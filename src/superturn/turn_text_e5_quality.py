"""Score a completed caller-pool E5 fit on the frozen provisional English view."""

import argparse
import json
from pathlib import Path

from .ami_calibration import _write_json
from .sources import data_root
from .timed_text_length_trial import publisher_rows, report_path
from .turn_text_train import binary_metrics, digest, load_fit, predict

FROZEN_PUBLISHER_SHA = "433ffbec3412ab8a962a8e058dbd8fc6323c6cd9425201cf1e46228f2cb08b78"
PREVIOUS_E5_SHA = "12626ca9d011bdb29ad33d223f5f0ec765a862c398cb28cd26e230239912d711"


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
    args = parser.parse_args()
    run(data_root(), args.dataset, args.base, args.output)
