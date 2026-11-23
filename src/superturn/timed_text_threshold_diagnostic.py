"""Measure threshold headroom for frozen English text checkpoints on publisher development."""

import json

import numpy as np

from .ami_calibration import _write_json
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .timed_text_length_trial import SEEDS, publisher_rows, report_path
from .turn_text_train import binary_metrics, digest, load_fit, predict


def threshold_for_ba(scores, labels):
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    shifts = int(labels.sum())
    holds = len(labels) - shifts
    if not shifts or not holds:
        raise ValueError("both classes are required to select a threshold")
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    recall = labels.cumsum()[ends] / shifts
    false_complete = (~labels).cumsum()[ends] / holds
    best = int(np.argmax(recall - false_complete))
    if recall[best] <= false_complete[best]:
        return float(np.nextafter(scores[0], np.inf))
    return float(scores[ends[best]])


def diagnostics(scores, labels, folds, gaps):
    scores = np.asarray(scores, dtype=np.float64)
    decisions = np.zeros(len(labels), dtype=bool)
    for index in range(5):
        held = folds == index
        threshold = threshold_for_ba(scores[~held], labels[~held])
        decisions[held] = scores[held] >= threshold

    def summarize_view(selected):
        values, truth = scores[selected], labels[selected]
        baseline = binary_metrics(list(zip(values, truth, strict=True)))
        calls = decisions[selected]
        recall = float(calls[truth].mean()) if truth.any() else None
        false_complete = float(calls[~truth].mean()) if (~truth).any() else None
        return {**baseline, "cross_fitted_shift_recall": recall, "cross_fitted_false_complete": false_complete,
                "cross_fitted_balanced_accuracy": (1 + recall - false_complete) / 2
                if recall is not None and false_complete is not None else None}

    return {
        "all": summarize_view(np.ones(len(labels), dtype=bool)),
        "gap_300_to_799_ms": summarize_view((gaps >= 300) & (gaps < 800)),
        "gap_at_least_800_ms": summarize_view(gaps >= 800),
        "optimistic_same_row_maximum": binary_metrics(list(zip(scores, labels, strict=True)),
                                                       threshold_for_ba(scores, labels))["balanced_accuracy"],
    }


def run(root):
    rows, freeze_path = publisher_rows(root)
    archive = json.loads(report_path(root, score_only=False).read_text())
    corpus_sha = digest(root / "cache/turn-text-focused-en-v1/manifest.json")
    freeze_sha = digest(freeze_path)
    if (archive["freeze_sha256"], archive["dataset_manifest_sha256"]) != (freeze_sha, corpus_sha):
        raise ValueError("publisher or training reference receipt changed")
    f = np.asarray([seed_for("publisher-threshold-fold-v1", row["caller"]) % 5 for row in rows])
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    gaps = np.asarray([row["gap_ms"] for row in rows])
    if len(set(f)) != 5:
        raise ValueError("publisher diagnostic requires five non-empty turn folds")
    texts = [row["text"] for row in rows]
    fits, predictions = [], []
    meter = Progress("english-threshold-headroom", len(SEEDS), "frozen checkpoints")
    for seed in SEEDS:
        stored, = [item for item in archive["trials"]["balanced"] if item["seed"] == seed]
        checkpoint = root / "runs" / f"turn-text-qwen3-length-balanced-{seed}/checkpoint.pt"
        checkpoint_sha = digest(checkpoint)
        if checkpoint_sha != stored["checkpoint_sha256"]:
            raise ValueError(f"weighted checkpoint changed for seed {seed}")
        tokenizer, model = load_fit(checkpoint, root / "cache/turn-text-qwen3-06b", "cuda", corpus_sha)
        scores = predict(tokenizer, model, texts, "cuda")
        del model
        if binary_metrics(list(zip(scores, labels, strict=True))) != stored["views"]["all"]:
            raise ValueError(f"weighted seed {seed} scores differ from the archived scorecard")
        predictions.append(scores)
        fits.append({"seed": seed, "checkpoint_sha256": checkpoint_sha,
                     "views": diagnostics(scores, labels, f, gaps)})
        meter.advance(1)
    meter.finish()
    report = {
        "scope": "Provisional publisher development only. Threshold headroom, no calibration or test.",
        "fold_unit": "Publisher turn group. Persistent speaker identity is unavailable.",
        "freeze_sha256": freeze_sha, "dataset_manifest_sha256": corpus_sha, "fits": fits,
        "probability_mean": diagnostics(np.mean(predictions, axis=0), labels, f, gaps),
    }
    _write_json(root / "reports/publisher-timed-english-threshold-headroom.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
