"""Compare matched AMI-only and AMI plus HarperValley supervised updates."""

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json, development_splits
from .ami_training import load_selection
from .audio import read_audio, silero
from .dense_teacher import SAMPLES, file_digest, normalize, render
from .dense_teacher_train import SEEDS, external
from .eot_domain_probe import DECISION_MS, GAP, decision_audio, summarize
from .harper_valley_audit import audit_call
from .harper_valley_plan import REVISION
from .labels import Label, ami_speakers, read_rows
from .model import build
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .splits import speaker_groups
from .train import _auc, _threshold

EPOCHS, STEPS, PER_CLASS = 20, 13, 16
HARPER = "harper_valley"


def ami_cohorts(root):
    rows = read_rows(root / "labels" / "ami.jsonl")
    archive, = (root / "raw" / "ami_annotations").glob("*.zip")
    with ZipFile(archive) as archive:
        groups = speaker_groups((meeting, (speaker for speaker, _ in tracks.values()))
                                for meeting, tracks in ami_speakers(archive).items())
    assignments = development_splits({r.recording for r in rows if r.split == "train"}, groups)
    limits = load_selection(root)
    cohorts = {name: [] for name in ("train", "validation", "calibration")}
    official_held = {r.speaker for r in rows if r.split != "train"}
    for row in rows:
        if row.split == "train" and limits.get((row.audio, row.pause_start, row.label), 0) >= GAP:
            if row.speaker in official_held:
                raise ValueError("pilot enrolment overlaps official held-out AMI identities")
            role = assignments[row.recording]
            cohorts[role].append(replace(row, split=role, pause_samples=GAP, group=groups[row.recording]))
    assert_disjoint(cohorts, "speaker")
    assert_disjoint(cohorts, "group")
    return cohorts


def assert_disjoint(cohorts, field):
    scopes = [{getattr(row, field) for row in rows} for rows in cohorts.values()]
    if len(set.union(*scopes)) != sum(map(len, scopes)):
        raise ValueError(f"{field} identity crossed AMI pilot roles")


def harper_train(root):
    selection = root / "cache" / "harper-valley" / f"{REVISION}-selection.json"
    audit = root / "reports" / "harper-valley-label-audit-clock-recovery.json"
    report = json.loads(audit.read_text())
    if not report["agent_clock_recovery"] or file_digest(selection) != report["selection_sha256"]:
        raise ValueError("training selection differs from the HarperValley timing audit")
    roles = json.loads(selection.read_text())
    people = {split: {c[role] for c in cases for role in ("caller", "agent")} for split, cases in roles.items()}
    if people["train"] & (people["validation"] | people["test"]):
        raise ValueError("HarperValley speaker cross-contamination")
    raw = root / "raw" / "harper-valley"
    detect = silero()
    counts, evidence, rows = Counter(), [], []
    meter = Progress("real-label-harper-timing", len(roles["train"]), "calls")
    try:
        for case in roles["train"]:
            call = case["id"]
            for asset in case["files"]:
                if (raw / asset["path"]).stat().st_size != asset["bytes"]:
                    raise ValueError("staged HarperValley asset size changed")
            pauses, clock = [], {}
            outcomes = audit_call(raw, call, detect, recover_clock=True, pauses=pauses, evidence=clock)
            counts.update(outcomes)
            evidence.append({"call": call, **clock, "fixed_pauses": pauses, "outcomes": outcomes})
            audio = f"raw/harper-valley/data/audio/caller/{call}.wav"
            rows.extend(Label(HARPER, "en", str(case["caller"]), call, "harper-train", "train",
                              edge, GAP, label, "word-and-vad-timing-supported", audio) for edge, label in pauses)
            meter.advance(1)
        if counts != report["splits"]["train"]["counts"]:
            raise ValueError("reconstructed HarperValley training labels differ from the timing audit")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return rows, evidence, roles["validation"], file_digest(audit)


def materialize(root, rows, encoding="telephone"):
    ordered = sorted(rows, key=lambda row: (row.audio, row.pause_start))
    keys = {(row.audio, row.pause_start) for row in ordered}
    if len(keys) != len(ordered):
        raise ValueError("repeated pause in a pilot cohort")
    @lru_cache(maxsize=2)
    def audio(path):
        return read_audio(root / path)
    if encoding == "telephone":
        pcm = np.stack([render(row, audio(row.audio), DECISION_MS) for row in ordered])
        clips = normalize(pcm)
    else:
        clips = np.stack([decision_audio(audio(row.audio), row.pause_start, "clean")[-SAMPLES:] for row in ordered])
    truth = np.asarray([row.label == "shift" for row in ordered])
    if np.unique(truth).size != 2:
        raise ValueError("pilot cohort lacks a hold or a shift")
    return clips, truth


def predict(model, clips, device):
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(clips), 128):
            x = torch.as_tensor(clips[start:start + 128], device=device)
            elapsed = torch.full((len(x), 1), float(DECISION_MS), device=device)
            scores.extend(model(x, elapsed).flatten().cpu().tolist())
    return scores


def fit(output, ami, extra, validation, calibration, seed, device):
    config = {"parameters": 150000, "window": 3, "architecture": "a", "seed": seed}
    torch.manual_seed(seed)
    model = build(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    in_ami, ami_y = ami
    in_extra, extra_y = extra
    pools = [[np.flatnonzero(y == target) for target in (False, True)] for y in (ami_y, extra_y)]
    rng = [np.random.default_rng(seed_for("real-label-pilot", seed, stream)) for stream in ("shared", "additional")]
    labels = torch.as_tensor(np.tile(np.repeat([0., 1.], PER_CLASS), 2), dtype=torch.float32, device=device)
    elapsed = torch.full((len(labels), 1), float(DECISION_MS), device=device)
    meter = Progress(f"real-label-{output.name}", EPOCHS * STEPS, "steps")
    best_auc, record, state = -1, None, None
    history = []
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for _ in range(STEPS):
                indices = [np.concatenate([generator.choice(pool, PER_CLASS) for pool in by_class])
                           for generator, by_class in zip(rng, pools, strict=True)]
                x = torch.as_tensor(np.concatenate([in_ami[indices[0]], in_extra[indices[1]]]), device=device)
                loss = F.binary_cross_entropy_with_logits(model.logits(x, elapsed).flatten(), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                meter.advance(1)
            auc = _auc(predict(model, validation[0], device), validation[1].tolist())
            history.append({"epoch": epoch, "ami_validation_auc": auc})
            if auc > best_auc:
                best_auc, record = auc, history[-1]
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    model.load_state_dict(state)
    cal_scores = predict(model, calibration[0], device)
    threshold = _threshold(cal_scores, calibration[1].tolist())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"config": config, "state": state}, temporary)
    temporary.replace(checkpoint)
    return checkpoint, {"checkpoint_sha256": file_digest(checkpoint), "epochs": history, "selected": record,
                        "ami_calibration_threshold": threshold,
                        "ami_calibration": summarize(cal_scores, calibration[1].tolist(), [threshold], roc_limits=True)}


def harper_development(root, validation_cases):
    membership_path = root / "reports" / "harper-valley-development-stratified-provenance.json"
    prior = json.loads((root / "reports" / "harper-valley-development-stratified.json").read_text())
    held = json.loads(membership_path.read_text())
    membership = [(row["call"], row["fixed_pauses"]) for row in held]
    if hashlib.sha256(json.dumps(membership, separators=(",", ":")).encode()).hexdigest() != prior["pause_membership_sha256"]:
        raise ValueError("HarperValley validation pause membership changed")
    callers = {c["id"]: str(c["caller"]) for c in validation_cases}
    held_rows = [Label(HARPER, "en", callers[call], call, "harper-validation", "validation",
                       edge, GAP, label, "word-and-vad-timing-supported", f"raw/harper-valley/data/audio/caller/{call}.wav")
                 for call, pauses in membership for edge, label in pauses]
    return {condition: materialize(root, held_rows, condition) for condition in ("telephone", "native")}, prior


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real-label pilot uses staged audio on Vast")
    cohorts = ami_cohorts(root)
    harper, evidence, validation_cases, audit_hash = harper_train(root)
    selected = materialize(root, harper)
    ami = {role: materialize(root, rows) for role, rows in cohorts.items()}
    harper_dev, prior = harper_development(root, validation_cases)
    fits, results = {}, {}
    for seed in SEEDS:
        for arm, extra in (("ami-only", ami["train"]), ("ami-plus-harper", selected)):
            name = f"{arm}-{seed}"
            fit_dir = root / "runs" / "real-label-pilot" / name
            fits[name] = fit(fit_dir, ami["train"], extra, ami["validation"], ami["calibration"], seed, device)
            checkpoint, record = fits[name]
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            model = build(saved["config"]).to(device)
            model.load_state_dict(saved["state"])
            threshold = record["ami_calibration_threshold"]
            results[name] = {condition: summarize(predict(model, clips, device), labels.tolist(), [threshold],
                                                  roc_limits=True)
                             for condition, (clips, labels) in harper_dev.items()}
    thresholds = {name: record["ami_calibration_threshold"] for name, (_, record) in fits.items()}
    report = {"use": "diagnostic", "decision_ms": DECISION_MS, "epochs": EPOCHS, "steps_per_epoch": STEPS,
              "harper_audit_sha256": audit_hash, "harper_validation_membership_sha256": prior["pause_membership_sha256"],
              "ami_fit_scopes": {role: dict(Counter(r.label for r in rows)) for role, rows in cohorts.items()},
              "harper_fit_scope": dict(Counter(r.label for r in harper)), "harper_validation": results,
              "fits": {name: record for name, (_, record) in fits.items()},
              "external_telephone": external(root, fits, device, thresholds=thresholds)}
    _write_json(root / "reports" / "real-label-pilot-training-timing-provenance.json", evidence)
    _write_json(root / "reports" / "real-label-pilot.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    result = run(data_root(), "cuda")
    print(json.dumps({"external_telephone": result["external_telephone"]}, indent=2))
