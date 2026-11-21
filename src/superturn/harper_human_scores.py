"""Score complete blind HarperValley judgments against the sealed private key."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
import re

from .sources import data_root

PACKET_SHA256 = "5bb9e19864b11fcc31e45e5a52e4cf6eba289f8ee01fe4fbd5513b405650bf8c"
FIELDS = ("example", "probability_percent", "abstain")


def judgments(payload, identifiers):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("expected the sealed reviewer CSV columns")
    scores = {}
    for row in reader:
        if None in row or None in row.values():
            raise ValueError("invalid reviewer CSV row")
        identifier = row["example"].strip()
        if identifier not in identifiers or identifier in scores:
            raise ValueError("unknown or duplicate blind example")
        probability = row["probability_percent"].strip()
        abstain = row["abstain"].strip().lower()
        if abstain in {"1", "true", "yes"} and not probability:
            scores[identifier] = None
        elif abstain in {"", "0", "false", "no"} and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", probability):
            value = Decimal(probability)
            if value > 100:
                raise ValueError("probability must be a decimal number from 0 to 100")
            scores[identifier] = value
        else:
            raise ValueError("each row needs either a probability or an explicit abstention")
    if scores.keys() != identifiers:
        raise ValueError("reviewer CSV is missing blind examples")
    return scores


def evaluate(labels, scores):
    totals = Counter(labels.values())
    if set(totals) != {0, 1}:
        raise ValueError("both observed floor labels are required")
    levels, abstentions = {}, Counter()
    for identifier, score in scores.items():
        label = labels[identifier]
        if score is None:
            abstentions[label] += 1
        else:
            levels.setdefault(score, Counter())[label] += 1

    def point(threshold, holds, shifts):
        return {
            "threshold_percent": str(threshold) if threshold is not None else None,
            "false_completes": holds,
            "shifts_detected": shifts,
            "false_complete_rate": holds / totals[0],
            "shift_recall": shifts / totals[1],
            "meets_nominal_80_5_on_packet": holds * 20 <= totals[0] and shifts * 5 >= totals[1] * 4,
        }

    curve = [point(None, 0, 0)]
    counts = Counter()
    for threshold, level in sorted(levels.items(), reverse=True):
        counts.update(level)
        curve.append(point(threshold, counts[0], counts[1]))
    matched = max(i for i, row in enumerate(curve) if row["false_completes"] * 20 <= totals[0])
    return {
        "holds": totals[0],
        "shifts": totals[1],
        "abstained_holds": abstentions[0],
        "abstained_shifts": abstentions[1],
        "matched_5_percent": curve[matched],
        "next_looser_threshold": curve[matched + 1] if matched + 1 < len(curve) else None,
        "curve": curve,
    }


def score_packet(root, ratings):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the sealed Harper packet and judgments stay on Vast")
    receipt_path = root / "reports/harper-human-baseline-packet.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt_digest = sha256(receipt_bytes).hexdigest()
    if receipt_digest != PACKET_SHA256:
        raise ValueError("receipt does not match the pinned packet seal")
    receipt = json.loads(receipt_bytes)
    listener = root / receipt["packet"]
    directory = listener.parent
    if sha256((listener / "GUIDE.md").read_bytes()).hexdigest() != receipt["guide_sha256"]:
        raise ValueError("sealed listener guide changed")
    if receipt.get("verified_clips") != receipt["clips"] or receipt.get("decision_ms") != 300:
        raise ValueError("expected a sealed 300 ms packet")
    key_bytes = (directory / "key.json").read_bytes()
    manifest_bytes = (directory / "audio-checksums.json").read_bytes()
    key_digest = sha256(key_bytes).hexdigest()
    manifest_digest = sha256(manifest_bytes).hexdigest()
    if key_digest != receipt["key_sha256"] or manifest_digest != receipt["audio_manifest_sha256"]:
        raise ValueError("sealed key or waveform manifest changed")
    labels = {identifier: value["label"] for identifier, value in json.loads(key_bytes).items()}
    if len(labels) != receipt["clips"] or labels.keys() != json.loads(manifest_bytes).keys():
        raise ValueError("sealed key and waveform membership differ")
    reviewers = []
    for reviewer, path in ratings.items():
        payload = path.read_bytes()
        digest = sha256(payload).hexdigest()
        if any(row["csv_sha256"] == digest for row in reviewers):
            raise ValueError("duplicate reviewer ballot")
        scores = judgments(payload, labels.keys())
        template = (listener / f"reviewer-{reviewer}.csv").read_bytes()
        rows = csv.DictReader(io.StringIO(template.decode("utf-8-sig"), newline=""))
        if list(scores) != [row["example"] for row in rows]:
            raise ValueError("reviewer CSV differs from the assigned example order")
        reviewers.append({"reviewer_slot": reviewer, "csv_sha256": digest, **evaluate(labels, scores)})
    return {
        "scope": "private retrospective English diagnostic on one connected caller-agent component",
        "decision_ms": 300,
        "rule": "Full response required. Abstentions always wait. The most permissive threshold within 5% preserves ties.",
        "packet_receipt_sha256": receipt_digest,
        "key_sha256": key_digest,
        "audio_manifest_sha256": manifest_digest,
        "scorer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "reviewers": reviewers,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for slot in ("a", "b"):
        parser.add_argument(f"--reviewer-{slot}", type=Path, help="filled reviewer CSV on Vast")
    args = parser.parse_args()
    ratings = {slot: path for slot, path in (("a", args.reviewer_a), ("b", args.reviewer_b)) if path}
    if not ratings:
        parser.error("provide at least one reviewer CSV")
    print(json.dumps(score_packet(data_root(), ratings), indent=2))
