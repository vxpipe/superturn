"""Prepare and verify the approved AMI training-only eligibility list."""

from collections import Counter
import hashlib
import json

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
    keys = {tuple(key) for key in report["keys"]}
    if not keys or len(keys) != report["sharded_examples"]:
        raise ValueError("AMI training selection is empty or has inconsistent counts")
    return keys


def prepare(root):
    rows = [row for row in read_rows(root / "labels" / "ami.jsonl") if row.split == "train"]
    votes = audit_sample(root, rows, end_ms=300, onset_ms=375)
    supported = {(row.audio, row.pause_start, row.label): row.speaker
                 for row, vote in votes.items() if vote == "supported"}
    available = set()
    for path in sorted((root / "shards" / "train" / "clean").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(
                columns=["source", "audio", "pause_start", "label"]):
            available.update((row["audio"], row["pause_start"], row["label"])
                             for row in batch.to_pylist() if row["source"] == "ami")
    keys = sorted(available & supported.keys())
    if not keys:
        raise ValueError("no AMI training shards passed review")
    summary = {
        "train_sha256": _train_fingerprint(root),
        "reviewed_examples": len(rows),
        "review_counts": dict(Counter(votes.values())),
        "tolerances_ms": {"caller_end": 300, "next_onset": 375},
        "sharded_examples": len(keys),
        "sharded_labels": dict(Counter(key[2] for key in keys)),
        "sharded_speakers": len({supported[key] for key in keys}),
        "original_ami_training_shards": len(available),
        "evaluation": "original validation and frozen test splits",
    }
    target = _path(root)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({**summary, "keys": keys}, indent=2) + "\n")
    tmp.replace(target)
    report = root / "reports" / "ami-verified-training.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({**summary, "selection_sha256": selection_hash(root)}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(data_root()), indent=2))
