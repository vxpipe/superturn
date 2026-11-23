"""Compare synthetic turn prefixes with private received Harper transcripts at 300 ms."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json

import numpy as np

from .ami_calibration import _write_json
from .sources import data_root
from .timed_text_visibility import OPEN, number
from .turn_text_train import digest, read_dataset

HESITATIONS = {"uh", "um", "erm", "hmm", "uhh", "umm"}


def partial_geometry(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    result = {}
    for label, members in sorted(groups.items()):
        geometry, caller_words, previous_words = Counter(), [], []
        for row in members:
            if not row["text"]:
                geometry[row.get("missing", "no_text")] += 1
                continue
            previous, current = row["text"].split("\n")
            tokens = current.split()
            if not tokens:
                raise ValueError("a visible transcript has an empty caller turn")
            count = len(tokens)
            caller_words.append(count)
            previous_words.append(0 if previous == "---" else len(previous.split()))
            geometry["text"] += 1
            geometry["first_turn_marker"] += previous == "---"
            geometry["caller_one_to_three_words"] += count <= 3
            geometry["caller_four_to_seven_words"] += 4 <= count <= 7
            geometry["caller_eight_to_fifteen_words"] += 8 <= count <= 15
            geometry["caller_sixteen_or_more_words"] += count >= 16
            ending = tokens[-1]
            geometry["open_ending"] += ending in OPEN
            geometry["number_ending"] += number(ending)
            geometry["hesitation_word"] += bool(HESITATIONS.intersection(tokens))
            geometry["repeated_adjacent_word"] += any(a == b for a, b in zip(tokens, tokens[1:]))
        result[label] = {
            "rows": len(members), "counts": dict(geometry),
            "caller_words_p10_p50_p90": np.percentile(caller_words, [10, 50, 90]).tolist() if caller_words else None,
            "previous_words_p10_p50_p90": np.percentile(previous_words, [10, 50, 90]).tolist() if previous_words else None,
        }
    return result


def run(root, *, include_scale=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("received transcript comparisons only run on Vast")
    receipt_path = root / "reports/harper-text-streaming-capture-v2.json"
    receipt = json.loads(receipt_path.read_text())
    settings = sha256(json.dumps(receipt["provider_settings"], sort_keys=True).encode()).hexdigest()
    packed_path = (root / receipt["packed_path"]).resolve()
    if (not packed_path.is_relative_to(root.resolve()) or digest(packed_path) != receipt["packed_sha256"]
            or settings != receipt["provider_settings_sha256"] or receipt["rows"] != 358
            or receipt["provider_settings"]["repeated_pauses"] != 50
            or receipt["provider_settings"]["benchmark_batch_order"] !=
            "sort by SHA-256 of call:caller_sample, no salt, independent of playback length"):
        raise ValueError("expected the complete corrected Harper capture receipt")
    packed = json.loads(packed_path.read_text())
    if packed["provider_settings_sha256"] != settings or packed["input_sha256"] != receipt["trace_sha256"]:
        raise ValueError("packed transcript receipt does not bind to the corrected trace")
    if (len({row["id"] for row in packed["pauses"]}) != 358 or
            Counter(row["label"] for row in packed["pauses"]) != {"hold": 179, "shift": 179}):
        raise ValueError("the corrected Harper label roster changed")
    real = []
    for row in packed["pauses"]:
        decision = row["at_300_ms"]
        if decision and decision["since_stop_ns"] != 300_000_000:
            raise ValueError("received transcript uses the wrong deadline")
        real.append({"label": row["label"], "text": decision["text"] if decision else None,
                     "missing": row.get("scoring_exclusion", "no_text" if decision else "unobserved")})
    corpora = {"b1": "turn-text-focused-en-v1"}
    if include_scale:
        corpora["b1_plus_scale"] = "turn-text-en-scale-v1"
    synthetic = {}
    for name, folder in corpora.items():
        splits, manifest = read_dataset(root / "cache" / folder)
        synthetic[name] = {"manifest_sha256": manifest,
                           "train_english": partial_geometry(row for row in splits["train"] if row["language"] == "en")}
    report = {
        "scope": "Aggregate input geometry, provisional Harper labels, no training or classifier evaluation",
        "real_capture_receipt_sha256": digest(receipt_path), "real_packed_sha256": receipt["packed_sha256"],
        "real_received_at_300_ms": partial_geometry(real), "synthetic": synthetic,
    }
    suffix = "scale" if include_scale else "b1"
    _write_json(root / "reports" / f"harper-v2-synthetic-partial-geometry-{suffix}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-scale", action="store_true")
    run(data_root(), include_scale=parser.parse_args().include_scale)
