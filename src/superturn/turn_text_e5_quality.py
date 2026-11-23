"""Score a completed caller-pool E5 fit on the frozen provisional English view."""

import argparse
import json
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .sources import data_root
from .timed_text_length_trial import publisher_rows, report_path
from .turn_text_train import binary_metrics, digest, load_fit, predict

FROZEN_PUBLISHER_SHA = "433ffbec3412ab8a962a8e058dbd8fc6323c6cd9425201cf1e46228f2cb08b78"
PREVIOUS_E5_SHA = "12626ca9d011bdb29ad33d223f5f0ec765a862c398cb28cd26e230239912d711"


def diagnostics(root, dataset, base, output):
    import pyarrow.parquet as pq

    from .audio import RATE
    from .harper_partial_stt import english_membership
    from .timed_text_pause_diagnostic import SHARD
    from .timed_text_visibility import length_scores, tags
    from .turn_metrics import auc

    quality_path = root / "reports" / f"{output.name}-english-quality.json"
    quality = json.loads(quality_path.read_text())
    rows, freeze = publisher_rows(root)
    corpus_sha = digest(dataset / "manifest.json")
    e5_checkpoint = output / "checkpoint.pt"
    qwen = json.loads(report_path(root, score_only=False).read_text())
    qwen_seed, = [fit for fit in qwen["trials"]["balanced"] if fit["seed"] == 29]
    qwen_checkpoint = root / "runs/turn-text-qwen3-length-balanced-29/checkpoint.pt"
    if (digest(freeze) != FROZEN_PUBLISHER_SHA or quality["freeze_sha256"] != FROZEN_PUBLISHER_SHA
            or qwen["freeze_sha256"] != FROZEN_PUBLISHER_SHA or qwen["dataset_manifest_sha256"] != corpus_sha
            or quality["training_manifest_sha256"] != corpus_sha
            or quality["checkpoint_sha256"] != digest(e5_checkpoint)
            or qwen_seed["checkpoint_sha256"] != digest(qwen_checkpoint)):
        raise ValueError("the failure diagnostic inputs differ from the locked own-model comparison")
    texts = [row["text"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    lengths = np.asarray([row["words"] for row in rows])
    by_model = {"length_only_grouped_cv": length_scores(rows)}
    for name, checkpoint, backbone, expected in (
            ("e5_caller", e5_checkpoint, base, quality["publisher"]),
            ("qwen_b1_29", qwen_checkpoint, root / "cache/turn-text-qwen3-06b", qwen_seed["views"]["all"])):
        tokenizer, model = load_fit(checkpoint, backbone, "cuda", corpus_sha)
        scores = np.asarray(predict(tokenizer, model, texts, "cuda"))
        del model
        if binary_metrics(list(zip(scores, labels, strict=True))) != expected:
            raise ValueError(f"{name} did not reproduce its frozen scorecard")
        by_model[name] = scores

    members, _, _ = english_membership(root)
    anchors = {rank: stop / RATE for rank, _, stop, _, _ in members}
    raw = pq.read_table(root / SHARD, columns=["words", "messages"]).to_pylist()
    visible = [(i, tags(row, anchors[row["rank"]], raw[row["call"]])) for i, row in enumerate(rows)
               if (row["gap_ms"] >= 1500 if row["label"] else 300 <= row["gap_ms"] < 800)]
    cells = {
        "number_ending_holds": [i for i, t in visible if not labels[i] and t[0] == "number"],
        "closed_1_to_3_word_shifts": [i for i, t in visible if labels[i] and t[0] == "closed_proxy" and t[3] == "1-3"],
        "closed_16_plus_other_continuation_holds": [i for i, t in visible if not labels[i] and
                                                   (t[0], t[1], t[3]) == ("closed_proxy", "other", "16+")],
    }
    length_bins = {}
    for low, high, band in ((1, 3, "1-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, np.inf, "16+")):
        included = (lengths >= low) & (lengths <= high)
        truth = labels[included]
        length_bins[band] = {
            "holds": int((~truth).sum()), "shifts": int(truth.sum()),
            "auc": {name: auc(scores[included], truth) if truth.any() and (~truth).any() else None
                    for name, scores in by_model.items()},
        }
    report = {"scope": "Owned fit diagnostics on provisional publisher development, fixed threshold 0.5",
              "quality_receipt_sha256": digest(quality_path), "freeze_sha256": FROZEN_PUBLISHER_SHA,
              "within_length_bins": length_bins, "arms": {}}
    for model, scores in by_model.items():
        arm = {"all_auc": auc(scores, labels), "class_score_distribution": {}, "step_a_cells": {}}
        for name, mask in (("holds", ~labels), ("shifts", labels)):
            values = scores[mask]
            arm["class_score_distribution"][name] = {
                "rows": len(values), "mean": float(values.mean()), "p25_p75": np.percentile(values, [25, 75]).tolist(),
                "share_at_least_0_9": float(np.mean(values >= .9)),
                "histogram_0_to_1_ten_bins": np.histogram(values, bins=10, range=(0, 1))[0].tolist(),
            }
        for name, indices in cells.items():
            decisions = scores[indices] >= .5
            arm["step_a_cells"][name] = {"rows": len(indices), "called_shift": int(decisions.sum()),
                                         "error_count": int((decisions != labels[indices]).sum())}
        report["arms"][model] = arm
    _write_json(root / "reports" / "publisher-timed-english-e5-caller-diagnostics.json", report)
    print(json.dumps(report, indent=2))


def run(root, dataset, base, output):
    checkpoint = output / "checkpoint.pt"
    receipt = json.loads((output / "receipt.json").read_text())
    corpus_sha = digest(dataset / "manifest.json")
    if (receipt["backbone"] != "e5-caller" or receipt["dataset_manifest_sha256"] != corpus_sha
            or receipt["checkpoint_sha256"] != digest(checkpoint)):
        raise ValueError("the score request does not match a completed caller-pool E5 fit")
    rows, freeze = publisher_rows(root)
    if digest(freeze) != FROZEN_PUBLISHER_SHA:
        raise ValueError("the provisional publisher freeze differs from the preregistration")
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    tokenizer, model = load_fit(checkpoint, base, "cuda", corpus_sha)
    scores = predict(tokenizer, model, texts, "cuda")
    del model
    metrics = binary_metrics(list(zip(scores, labels, strict=True)))
    if (metrics["holds"], metrics["shifts"]) != (527, 397):
        raise ValueError("the provisional English label roster changed")
    english = receipt["history"][-1]["validation"]["language/en"]
    archived = json.loads(report_path(root, score_only=False).read_text())
    if archived["freeze_sha256"] != FROZEN_PUBLISHER_SHA or archived["dataset_manifest_sha256"] != corpus_sha:
        raise ValueError("the archived Qwen reference differs from the current comparison")
    qwen, = [fit for fit in archived["trials"]["balanced"] if fit["seed"] == 29]
    old_checkpoint = root / "runs/turn-text-e5-focused-en-v1/checkpoint.pt"
    if digest(old_checkpoint) != PREVIOUS_E5_SHA:
        raise ValueError("the previous E5 checkpoint changed")
    tokenizer, model = load_fit(old_checkpoint, base, "cuda", corpus_sha)
    old_scores = predict(tokenizer, model, texts, "cuda")
    result = {
        "scope": "Provisional English publisher word-cut development, fixed 0.5 cutoff",
        "freeze_sha256": digest(freeze), "training_manifest_sha256": corpus_sha,
        "checkpoint_sha256": receipt["checkpoint_sha256"], "fit_receipt_sha256": digest(output / "receipt.json"),
        "seed": receipt["seed"], "epoch": receipt["epochs"], "publisher": metrics,
        "matched_model_references": {
            "qwen_b1_seed_29_archived": qwen["views"]["all"],
            "earlier_e5_focused": binary_metrics(list(zip(old_scores, labels, strict=True))),
        },
        "synthetic_english_validation": english,
    }
    _write_json(root / "reports" / f"{output.name}-english-quality.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    command = diagnostics if args.diagnostics else run
    command(data_root(), args.dataset, args.base, args.output)
