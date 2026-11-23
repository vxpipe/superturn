"""Evaluate the frozen text fit on qualified English development inputs."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .harper_text_review import gap_band
from .sources import data_root
from .turn_text_train import LABELS, binary_metrics, digest, load_fit, predict

QUALIFICATION = "harper-text-dev-qualification.json"
PACK_RECEIPT = "harper-text-dev-packing.json"
SCORE_SCOPE = "English HarperValley development only"


def open_qualified_pack(root, checkpoint):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark inputs stay on Vast")
    reports = root / "reports"
    gate = reports / QUALIFICATION
    if not gate.is_file():
        raise ValueError("independent English development label qualification is missing")
    receipt = json.loads(gate.read_text())
    if (receipt.get("scope") != SCORE_SCOPE
            or receipt.get("status") != "qualified"
            or receipt.get("checkpoint_sha256") != digest(checkpoint)):
        raise ValueError("the qualification does not cover this development scope and frozen fit")
    evidence = receipt.get("independent_audit_reports", {})
    if not evidence or any(Path(name).name != name or name in {QUALIFICATION, PACK_RECEIPT}
                           or digest(reports / name) != checksum for name, checksum in evidence.items()):
        raise ValueError("the independent audit evidence is missing or changed")
    pack_receipt = json.loads((reports / PACK_RECEIPT).read_text())
    checksum = pack_receipt["serialized_sha256"]
    path = root / "cache/harper-valley/text-dev-v1.jsonl"
    if receipt.get("input_sha256") != checksum or digest(path) != checksum:
        raise ValueError("the qualified English input pack has changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != pack_receipt["rows"]
            or len({(row["call"], row["caller_sample"]) for row in rows}) != len(rows)
            or {row["label"] for row in rows} != set(LABELS)):
        raise ValueError("the English development input membership changed")
    return rows, checksum, digest(gate)


def breakdown(rows, scores):
    groups = defaultdict(list)
    calls = defaultdict(set)
    callers = defaultdict(lambda: defaultdict(Counter))
    for row, score in zip(rows, scores, strict=True):
        band = gap_band(row["gap_ms"])
        target, prediction = LABELS[row["label"]], int(score >= .5)
        for name in ("english", band):
            groups[name].append((score, target))
            calls[name].add(row["call"])
            callers[name][row["caller"]][target, prediction] += 1
    return {
        name: {
            **binary_metrics(groups[name]),
            "calls": len(calls[name]),
            "callers": len(callers[name]),
            "caller_bootstrap_balanced_accuracy_95": caller_interval(callers[name]),
            "true_shifts": sum(score >= .5 and label == 1 for score, label in groups[name]),
            "false_completes": sum(score >= .5 and label == 0 for score, label in groups[name]),
        }
        for name in ("english", "300_to_800_ms", "over_800_ms")
    }


def caller_interval(groups):
    if len(groups) < 2:
        return None
    counts = np.array([[rows[0, 0], rows[0, 1], rows[1, 0], rows[1, 1]] for rows in groups.values()])
    draws = np.random.default_rng(29).integers(0, len(counts), (5000, len(counts)))
    correct_hold, false_complete, missed_shift, correct_shift = counts[draws].sum(axis=1).T
    holds, shifts = correct_hold + false_complete, missed_shift + correct_shift
    valid = (holds > 0) & (shifts > 0)
    if not valid.any():
        return None
    balanced = (correct_hold[valid] / holds[valid] + correct_shift[valid] / shifts[valid]) / 2
    return np.quantile(balanced, [.025, .975]).tolist()


def evaluate(root, base, checkpoint, dataset_manifest, device):
    rows, input_sha, qualification_sha = open_qualified_pack(root, checkpoint)
    tokenizer, model = load_fit(checkpoint, base, device, digest(dataset_manifest))
    fields = ("text", "current_only", "last_word_missing")
    scored = {}
    for field in fields:
        eligible = [row for row in rows if row[field]]
        probabilities = predict(tokenizer, model, [row[field] for row in eligible], device)
        scored[field] = breakdown(eligible, probabilities)
    paired = [row for row in rows if row["last_word_missing"]]
    full_scores = predict(tokenizer, model, [row["text"] for row in paired], device)
    result = {
        "scope": SCORE_SCOPE,
        "threshold_source": "fixed 0.5 from the diagnostic synthetic fit",
        "checkpoint_sha256": digest(checkpoint),
        "input_sha256": input_sha,
        "qualification_sha256": qualification_sha,
        "available_context": scored["text"],
        "current_only": scored["current_only"],
        "last_word_missing": scored["last_word_missing"],
        "available_context_on_same_last_word_missing_rows": breakdown(paired, full_scores),
        "blank_after_removing_last_word": dict(Counter(
            f"{row['label']}/{gap_band(row['gap_ms'])}" for row in rows if not row["last_word_missing"])),
    }
    _write_json(root / "reports/harper-text-dev-scorecard.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    root = data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=root / "cache/turn-text-e5-small")
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/turn-text-e5-v1/checkpoint.pt")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("datasets/turn-completion-v1/manifest.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    evaluate(root, args.base, args.checkpoint, args.dataset_manifest, args.device)
