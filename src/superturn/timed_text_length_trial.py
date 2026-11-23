"""Compare capped length balancing with unweighted replay at fixed English coverage."""

from hashlib import sha256
import json

import numpy as np

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root
from .timed_text_pause_diagnostic import FREEZE, pack_rows
from .turn_text_train import binary_metrics, digest, load_fit, predict, run as train

SEEDS = (17, 29, 43, 71, 101)


def short_hold_delta(rows, unweighted, balanced):
    short = [index for index, row in enumerate(rows)
             if not row["label"] and 300 <= row["gap_ms"] < 800 and row["words"] >= 8]
    turns = {}
    for index in short:
        turns.setdefault(rows[index]["caller"], []).append(index)
    difference = (np.asarray(balanced) >= .5).astype(int) - (np.asarray(unweighted) >= .5)
    grouped = np.stack([difference[:, indices].sum(axis=1) for indices in turns.values()], axis=1)
    counts = np.array([len(indices) for indices in turns.values()])
    rng = np.random.default_rng(29)
    draw_seeds = rng.integers(len(SEEDS), size=(4000, len(SEEDS)))
    draw_turns = rng.integers(len(turns), size=(4000, len(turns)))
    resampled = grouped[draw_seeds[:, :, None], draw_turns[:, None, :]].mean(axis=1).sum(axis=1)
    return {"holds": len(short), "turns": len(turns), "delta": float(grouped.sum() / (len(SEEDS) * len(short))),
            "paired_seed_and_turn_ci95": np.quantile(resampled / counts[draw_turns].sum(axis=1), [.025, .975]).tolist()}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("publisher trial inputs stay on Vast")
    freeze_path = root / "reports" / FREEZE
    freeze = json.loads(freeze_path.read_text())
    rows, _, membership = pack_rows(root)
    if (freeze["membership_sha256"] != membership
            or freeze["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()):
        raise ValueError("publisher development reference changed")
    rows = [row for row in rows if row["has_previous"]]
    dataset = root / "cache/turn-text-focused-en-v1"
    base = root / "cache/turn-text-qwen3-06b"
    corpus_hash = digest(dataset / "manifest.json")
    bands = {
        "all": list(range(len(rows))),
        "short_8_plus_holds": [i for i, row in enumerate(rows)
                               if not row["label"] and row["words"] >= 8 and 300 <= row["gap_ms"] < 800],
        "long_gap": [i for i, row in enumerate(rows) if row["gap_ms"] >= 1500],
    }
    if tuple(map(len, bands.values())) != (924, 327, 436):
        raise ValueError("unexpected publisher trial slice counts")
    trials = {"replay": [], "balanced": []}
    arm_scores = {"replay": [], "balanced": []}
    meter = Progress("english-length-trial", 2 * len(SEEDS), "fits")
    for seed in SEEDS:
        for arm in trials:
            output = root / "runs" / f"turn-text-qwen3-length-{arm}-{seed}"
            train(dataset, base, output, seed=seed, backbone="qwen3", balance_length=arm == "balanced")
            checkpoint = output / "checkpoint.pt"
            tokenizer, model = load_fit(checkpoint, base, "cuda", corpus_hash)
            scores = predict(tokenizer, model, [row["text"] for row in rows], "cuda")
            del model
            arm_scores[arm].append(scores)
            metrics = {name: binary_metrics([(scores[i], rows[i]["label"]) for i in indices])
                       for name, indices in bands.items()}
            trials[arm].append({"seed": seed, "checkpoint_sha256": digest(checkpoint), "views": metrics})
            meter.advance(1)
    meter.finish()
    delta = short_hold_delta(rows, arm_scores["replay"], arm_scores["balanced"])
    rollup = {}
    for arm, fits in trials.items():
        rollup[arm] = {}
        for name in bands:
            view = [item["views"][name] for item in fits]
            rollup[arm][name] = {
                key: {"mean": float(np.mean(values)), "range": [min(values), max(values)]}
                for key in view[0] if view[0][key] is not None
                if (values := [item[key] for item in view])
            }
    def diff(view, metric):
        return rollup["balanced"][view][metric]["mean"] - rollup["replay"][view][metric]["mean"]

    report = {"scope": "fixed English development at 0.5, paired seeds, no calibration or test",
              "freeze_sha256": digest(freeze_path), "dataset_manifest_sha256": corpus_hash,
              "seeds": SEEDS, "trials": trials, "rollup": rollup, "short_hold_comparison": delta,
              "gate": {"short_hold_delta": delta["delta"] <= -.05 and delta["paired_seed_and_turn_ci95"][1] < 0,
                       "shift_recall": diff("all", "shift_recall") >= -.02,
                       "long_gap_auc": diff("long_gap", "auc") >= -.01,
                       "all_auc": diff("all", "auc") >= -.005}}
    _write_json(root / "reports/publisher-timed-english-length-trial.json", report)
    print(json.dumps({"rollup": rollup, "short_hold_comparison": delta, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run(data_root())
