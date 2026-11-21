"""Test whether a short real-label fit improves the dense teacher student."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .ami_text_signal import CROSSVAL
from .dense_teacher import DEADLINES, SOURCES, file_digest, normalize
from .dense_teacher_train import external, load_pool
from .eot_domain_probe import DECISION_MS, summarize
from .model import build
from .progress import Progress
from .real_label_pilot import ami_cohorts, harper_development, harper_train, materialize, predict
from .shards import seed_for
from .sources import data_root
from .train import _threshold

UPDATES = 40
RUNS = ((17, 17), (29, 29), (41, 17), (53, 29))
RECIPES = {
    "dense-harper": (0, 64, 0),
    "dense-mix": (16, 48, 0),
    "dense-rehearsal": (4, 12, 48),
    "scratch-mix": (16, 48, 0),
}
RATES = (0.0001, 0.0003)
SCRATCH_RATE = 0.001


def fit_ami(root, ami):
    origin = root / "shards" / "ami-crossval" / CROSSVAL
    folds = sorted(origin.glob("fold-*/shards/out-of-fold/clean/*.parquet"))
    if not folds:
        raise ValueError("AMI cross-validation holdout shards are required")
    excluded = {name for path in folds for name in pq.read_table(path, columns=["group"]).column(0).to_pylist()}
    rows = [row for row in ami["train"] if row.group not in excluded]
    if {row.label for row in rows} != {"hold", "shift"}:
        raise ValueError("no balanced AMI fit pool survives the cross-validation exclusion")
    return rows, excluded


def voice_pool(root, rows):
    clips, labels = materialize(root, rows)
    by_label = {False: defaultdict(list), True: defaultdict(list)}
    ordered = sorted(rows, key=lambda row: (row.audio, row.pause_start))
    for index, (row, label) in enumerate(zip(ordered, labels, strict=True)):
        by_label[bool(label)][row.speaker].append(index)
    return clips, [list(voices.values()) for voices in by_label.values()]


def draw_voices(pool, count, rng):
    clips, labels = pool
    by_class = []
    for voices in labels:
        chosen = rng.integers(len(voices), size=count // 2)
        by_class.extend(rng.choice(voices[i]) for i in chosen)
    return clips[by_class], np.repeat(np.asarray([0, 1], dtype=np.float32), count // 2)


def replay_pool(root, excluded_ami):
    pcm, targets, sources, roles, groups, fingerprint = load_pool(root, root / "shards" / "dense-teacher")
    deadline = DEADLINES.index(DECISION_MS)
    # Retain one deterministic set per source. Training never draws fidelity rows.
    selected = []
    for source in SOURCES:
        rows = np.flatnonzero((sources == source) & (roles == "fit"))
        candidates = defaultdict(list)
        for index in rows:
            if source != "ami" or groups[index] not in excluded_ami:
                candidates[groups[index]].append(index)
        rng = np.random.default_rng(seed_for("fine-tune-rehearsal-sample", source))
        scopes = list(candidates.values())
        if not scopes:
            raise ValueError(f"{source} has no rehearsal groups after holdout exclusions")
        block = np.asarray([rng.choice(scopes[i]) for i in rng.integers(len(scopes), size=1024)])
        selected.append((normalize(pcm[block, deadline]), targets[block, deadline]))
    return selected, fingerprint


def draw_replay(pool, count, rng):
    audio, targets = [], []
    for clips, y in pool:
        indices = rng.integers(len(y), size=count // len(pool))
        audio.append(clips[indices])
        targets.append(y[indices])
    return np.concatenate(audio), np.concatenate(targets)


def tune(name, model, recipe, rate, pools, seed, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    rngs = [np.random.default_rng(seed_for("teacher-tune-v1", seed, name)) for name in ("ami", "harper", "replay")]
    elapsed = torch.full((sum(recipe), 1), float(DECISION_MS), device=device)
    progress = Progress(f"teacher-tune-{name}", UPDATES, "steps")
    model.train()
    try:
        for _ in range(UPDATES):
            batches = [draw(pool, count, rng) for draw, pool, count, rng in
                       zip((draw_voices, draw_voices, draw_replay), pools, recipe, rngs, strict=True) if count]
            audio = torch.as_tensor(np.concatenate([clips for clips, _ in batches]), device=device)
            labels = torch.as_tensor(np.concatenate([y for _, y in batches]), device=device)
            loss = F.binary_cross_entropy_with_logits(model.logits(audio, elapsed).flatten(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            progress.advance(1)
    except BaseException:
        progress.finish("failed")
        raise
    else:
        progress.finish()


def checkpoint(root, fits, seed, device):
    name = f"150000-{seed}-teacher"
    path = root / "runs" / "dense-teacher-diagnostic" / name / "best.pt"
    if file_digest(path) != fits[name]["checkpoint_sha256"]:
        raise ValueError("dense initialization differs from the prior fit report")
    state = torch.load(path, map_location=device, weights_only=True)
    return path, state


def summarize_model(model, calibration, developed, device):
    calibration_scores = predict(model, calibration[0], device)
    threshold = _threshold(calibration_scores, calibration[1].tolist())
    return threshold, {
        condition: summarize(predict(model, clips, device), y.tolist(), [threshold], roc_limits=True)
        for condition, (clips, y) in developed.items()
    }


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("teacher fine-tuning uses the staged Vast data")
    ami = ami_cohorts(root)
    approved, excluded_ami = fit_ami(root, ami)
    harper, evidence, validation_cases, audit_hash = harper_train(root)
    replay, replay_hash = replay_pool(root, excluded_ami)
    pools = (voice_pool(root, approved), voice_pool(root, harper), replay)
    calibration = materialize(root, ami["calibration"])
    development, prior = harper_development(root, validation_cases)
    fit_report = root / "reports" / "dense-teacher-gru.json"
    original = json.loads(fit_report.read_text())["fit"]
    fits, harper_metrics, thresholds, records = {}, {}, {}, {}
    for seed, dense_seed in RUNS:
        dense_path, dense = checkpoint(root, original, dense_seed, device)
        calibrate = {"dense-unmodified": (0, 0, 0), **RECIPES}
        for arm, recipe in calibrate.items():
            rates = (SCRATCH_RATE,) if arm == "scratch-mix" else (None,) if arm == "dense-unmodified" else RATES
            for rate in rates:
                name = f"{arm}-{seed}" if rate in (None, SCRATCH_RATE) else f"{arm}-{seed}-{rate:g}"
                torch.manual_seed(seed)
                config = {**dense["config"], "seed": seed, "shuffled": False}
                model = build(config).to(device)
                if arm != "scratch-mix":
                    model.load_state_dict(dense["state"])
                if rate is None:
                    path = dense_path
                else:
                    tune(name, model, recipe, rate, pools, seed, device)
                    path = root / "runs" / "teacher-finetune" / name / "best.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    torch.save({"config": config, "state": model.state_dict()}, temporary)
                    temporary.replace(path)
                thresholds[name], harper_metrics[name] = summarize_model(model, calibration, development, device)
                fits[name] = path, {}
                records[name] = {"arm": arm, "seed": seed, "initial_dense_seed": dense_seed if arm != "scratch-mix" else None,
                                 "rate": rate, "checkpoint_sha256": file_digest(path),
                                 "ami_calibration_threshold": thresholds[name]}
    metrics = external(root, fits, device, thresholds=thresholds)
    aggregates = {}
    for arm in ("dense-unmodified", *RECIPES):
        for rate in {record["rate"] for record in records.values() if record["arm"] == arm}:
            names = [name for name, record in records.items() if (record["arm"], record["rate"]) == (arm, rate)]
            key = arm if rate in (None, SCRATCH_RATE) else f"{arm}-{rate:g}"
            aggregates[key] = {
                "external_auc": mean(metrics[n]["auc"] for n in names),
                "external_recall_at_5pct": mean(metrics[n]["retrospective_roc_limits"]
                                                ["max_recall_at_5pct_false_complete"] for n in names),
                "harper_telephone_auc": mean(harper_metrics[n]["telephone"]["auc"] for n in names),
            }
    control, scratch = aggregates["dense-unmodified"], aggregates["scratch-mix"]
    for name, scores in aggregates.items():
        scores["exceeds_exploratory_auc_gate"] = name not in ("dense-unmodified", "scratch-mix") and (
            scores["external_auc"] >= control["external_auc"] + 0.03
            and scores["harper_telephone_auc"] >= control["harper_telephone_auc"] - 0.03
            and scores["external_auc"] >= scratch["external_auc"] + 0.08)
    membership = json.dumps([asdict(row) for row in (*approved, *harper)], sort_keys=True).encode()
    report = {"use": "diagnostic", "updates": UPDATES, "label_mix_ami_harper_teacher": RECIPES,
              "seeds_and_dense_initializations": RUNS, "ami_fit_labels": dict(Counter(r.label for r in approved)),
              "ami_cv_groups_excluded_from_updates": len(excluded_ami),
              "harper_fit_labels": dict(Counter(r.label for r in harper)),
              "fit_membership_sha256": hashlib.sha256(membership).hexdigest(),
              "trainer_sha256": file_digest(Path(__file__)),
              "harper_audit_sha256": audit_hash, "harper_validation_membership_sha256": prior["pause_membership_sha256"],
              "teacher_replay_report_sha256": replay_hash, "replay_draws_per_source": 1024,
              "external": metrics, "harper": harper_metrics, "fits": records, "aggregate": aggregates}
    _write_json(root / "reports" / "teacher-finetune-harper-provenance.json", evidence)
    _write_json(root / "reports" / "teacher-finetune.json", report)
    return aggregates


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(run(data_root(), "cuda"), indent=2))
