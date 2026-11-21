"""Prepare and verify the approved AMI training-only eligibility list."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .ami_reference import audit_sample
from .labels import read_rows
from .sources import data_root


def _path(root):
    return root / "labels" / "ami-verified-train.json"


def _train_fingerprint(root):
    manifest = json.loads((root / "shards" / "manifest.json").read_text())
    return manifest["sha256_by_split"]["train"]


def selection_hash(root):
    return hashlib.sha256(_path(root).read_bytes()).hexdigest()


def load_selection(root):
    report = json.loads(_path(root).read_text())
    if report["train_sha256"] != _train_fingerprint(root):
        raise ValueError("AMI verification applies to a different training shard set")
    limits = {tuple(key): limit for key, limit in report["pause_limits"]}
    if not limits or len(limits) != report["sharded_examples"] or min(limits.values()) < 3200:
        raise ValueError("AMI training selection is empty or has inconsistent counts")
    return limits


def prepare(root):
    label_path = root / "labels" / "ami.jsonl"
    rows = [row for row in read_rows(label_path) if row.split == "train"]
    pause_limits = {}
    votes = audit_sample(root, rows, end_ms=300, onset_ms=375, pause_limits=pause_limits)
    supported = {(row.audio, row.pause_start, row.label): row
                 for row, vote in votes.items() if vote == "supported"}
    available = set()
    for path in sorted((root / "shards" / "train" / "clean").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(
                columns=["source", "audio", "pause_start", "label"]):
            available.update((row["audio"], row["pause_start"], row["label"])
                             for row in batch.to_pylist() if row["source"] == "ami")
    matched = available & supported.keys()
    keys = sorted(key for key in matched if pause_limits[supported[key]] >= 3200)
    if not keys:
        raise ValueError("no AMI training shards passed review")
    summary = {
        "train_sha256": _train_fingerprint(root),
        "reviewed_examples": len(rows),
        "review_counts": dict(Counter(votes.values())),
        "tolerances_ms": {"caller_end": 300, "next_onset": 375},
        "sharded_examples": len(keys),
        "sharded_labels": dict(Counter(key[2] for key in keys)),
        "sharded_speakers": len({supported[key].speaker for key in keys}),
        "shorter_usable_pause": sum(
            any(pause_limits[supported[key]] < ms * 16 <= supported[key].pause_samples
                for ms in (300, 500)) for key in keys
        ),
        "excluded_under_200_ms": len(matched) - len(keys),
        "original_ami_training_shards": len(available),
        "evaluation": "original validation and frozen test splits",
    }
    target = _path(root)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        **summary, "pause_limits": [(key, pause_limits[supported[key]]) for key in keys],
    }, indent=2) + "\n")
    tmp.replace(target)
    report = root / "reports" / "ami-verified-training.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    annotation, = (root / "raw" / "ami_annotations").glob("*.zip")
    verifier = Path(__file__).with_name("ami_reference.py")
    report.write_text(json.dumps({
        **summary,
        "selection_sha256": selection_hash(root),
        "ami_labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(data_root()), indent=2))
