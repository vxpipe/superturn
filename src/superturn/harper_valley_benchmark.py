"""Score fixed development models on speaker-held HarperValley timing labels."""

from collections import Counter, defaultdict
import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .ami_deadline import retrospective_limits
from .ami_external_baselines import load_smart_turn
from .audio import RATE, read_audio, silero
from .dense_teacher import SAMPLES, TEACHER_SHA256, file_digest
from .eot_domain_probe import DECISION_MS, GAP, decision_audio, load_validated_models, summarize
from .harper_valley_audit import audit_call
from .harper_valley_plan import REVISION
from .model import build
from .progress import Progress
from .sources import data_root
from .train import _auc

BASELINE = "s1-a-150000-29"
STUDENTS = ("150000-17-teacher", "150000-29-teacher")
TEACHERS = {"smart-turn-3s": SAMPLES, "smart-turn-8s": 8 * RATE}


def pinned_validation(root):
    path = root / "cache" / "harper-valley" / f"{REVISION}-selection.json"
    archived = root / "reports" / "harper-valley-label-audit-clock-recovery.json"
    report = json.loads(archived.read_text())
    if file_digest(path) != report["selection_sha256"] or not report["agent_clock_recovery"]:
        raise ValueError("HarperValley selection differs from the clock-recovery audit")
    roles = json.loads(path.read_text())
    if {c["caller"] for c in roles["validation"]} & {
            c[role] for split in ("train", "test") for c in roles[split] for role in ("caller", "agent")}:
        raise ValueError("HarperValley validation callers overlap other roles")
    return roles["validation"], report["splits"]["validation"], file_digest(archived)


def paired_intervals(scores, truth, callers):
    """Group bootstrap of each student minus the fixed 145k baseline."""
    grouped = defaultdict(list)
    for i, caller in enumerate(callers):
        grouped[caller].append(i)
    groups = list(grouped.values())
    rng = np.random.default_rng(60029)
    y = np.asarray(truth)
    predictions = {name: np.asarray(values) for name, values in scores.items()}
    deltas = defaultdict(list)
    for sampled in rng.integers(len(groups), size=(500, len(groups))):
        indices = [i for group in sampled for i in groups[group]]
        target = y[indices].tolist()
        if min(target) == max(target):
            continue

        def rates(name):
            values = predictions[name][indices].tolist()
            return np.asarray([_auc(values, target),
                               retrospective_limits(values, target)["max_recall_at_5pct_false_complete"]])

        reference = rates(BASELINE)
        for name in STUDENTS:
            deltas[name].append(rates(name) - reference)
    return {name: {"replicates": len(values), "delta_auc_95": np.quantile(np.asarray(values)[:, 0], [0.025, 0.975]).tolist(),
                   "delta_recall_95": np.quantile(np.asarray(values)[:, 1], [0.025, 0.975]).tolist()}
            for name, values in deltas.items()}


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("HarperValley benchmarking uses the staged train and validation selection on Vast")
    entries, expected, audit_hash = pinned_validation(root)
    detect = silero()
    predict, teacher_hash = load_smart_turn(root)
    if teacher_hash != TEACHER_SHA256:
        raise ValueError("Smart Turn does not match the previously evaluated teacher")
    baseline, baseline_provenance = load_validated_models(root, (BASELINE,))
    models = {name: model.to(device) for name, model, _ in baseline}
    fit_report = root / "reports" / "dense-teacher-gru.json"
    fits = json.loads(fit_report.read_text())["fit"]
    for name in STUDENTS:
        artifact = root / "runs" / "dense-teacher-diagnostic" / name / "best.pt"
        if file_digest(artifact) != fits[name]["checkpoint_sha256"]:
            raise ValueError(f"student checkpoint differs from its completed fit: {name}")
        stored = torch.load(artifact, map_location=device, weights_only=True)
        model = build(stored["config"]).to(device).eval()
        model.load_state_dict(stored["state"])
        models[name] = model
    raw = root / "raw" / "harper-valley"
    scores, counts = defaultdict(list), Counter()
    audit, truth, speakers = [], [], []
    meter = Progress("harper-valley-development-benchmark", len(entries), "calls")
    try:
        for case in entries:
            identifier = case["id"]
            pauses, evidence = [], {}
            for asset in case["files"]:
                if (raw / asset["path"]).stat().st_size != asset["bytes"]:
                    raise ValueError("staged HarperValley asset differs in size from the approved selection")
            counts.update(outcomes := audit_call(raw, identifier, detect, recover_clock=True,
                                                 pauses=pauses, evidence=evidence))
            audit.append({"call": identifier, **evidence, "outcomes": outcomes})
            if pauses:
                pcm = read_audio(raw / "data" / "audio" / "caller" / f"{identifier}.wav")
                clips = []
                for edge, label in pauses:
                    if not 0 <= edge <= len(pcm) - GAP:
                        raise ValueError("caller-local pause does not reach the 300 ms query")
                    truth.append(int(label == "shift"))
                    speakers.append(case["caller"])
                    rendered = decision_audio(pcm, edge, "telephone")
                    for name, length in TEACHERS.items():
                        scores[name].append(predict(rendered[-length:]))
                    clips.append(rendered[-SAMPLES:])
                audio = torch.as_tensor(np.stack(clips), device=device)
                elapsed = torch.full((len(audio), 1), float(DECISION_MS), device=device)
                with torch.inference_mode():
                    for name, model in models.items():
                        scores[name].extend(model(audio[:, -model.samples:], elapsed).flatten().cpu().tolist())
            meter.advance(1)
        if counts != expected["counts"] or (len(truth) - sum(truth), sum(truth)) != (
                counts["fixed_window_supported_hold"], counts["fixed_window_supported_shift"]):
            raise ValueError("reconstructed pauses differ from the frozen timing audit")
        for values in scores.values():
            if len(values) != len(truth) or not np.all((0 <= np.asarray(values)) & (np.asarray(values) <= 1)):
                raise ValueError("benchmark scores are incomplete or invalid")
        metrics = {name: summarize(values, truth, [0.5], roc_limits=True) for name, values in scores.items()}
        report = {"set": "validation", "label_scope": "fixed-window timing agreement",
                  "condition": "G.711 telephone", "decision_ms": DECISION_MS,
                  "audit_sha256": audit_hash, "student_fit_sha256": file_digest(fit_report),
                  "baseline": baseline_provenance, "teacher_sha256": teacher_hash,
                  "calls": len(entries), "callers": len(set(speakers)), "holds": len(truth) - sum(truth),
                  "shifts": sum(truth), "metrics": metrics,
                  "caller_bootstrap_delta_from_baseline": paired_intervals(scores, truth, speakers)}
        _write_json(root / "reports" / "harper-valley-development-timing-provenance.json", audit)
        _write_json(root / "reports" / "harper-valley-development-benchmark.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root(), "cuda"), indent=2))
