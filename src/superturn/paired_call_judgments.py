"""Audit two independent paired-call ballots before admitting training labels."""

from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import file_digest
from .paired_call_review import JOB, verify
from .sources import data_root

CANDIDATE_FIELDS = ("example", "outcome", "brief_ack", "language", "confidence")
CONTINUOUS_FIELDS = ("recording", "focal_channel", "stop_ms", "next_onset_ms", "next_channel",
                     "outcome", "brief_ack", "language", "confidence")
OUTCOMES = {"resume", "handoff", "interruption", "uncertain", "invalid_pause"}
CLEAR = {"resume", "handoff"}
CHANNELS = {"left": 0, "right": 1}


def read_rows(path, expected):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"incorrect ballot columns: {path.name}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"incomplete ballot row: {path.name}")
            yield {key: value.strip().casefold() for key, value in row.items()}


def validate_judgment(row):
    if (row["outcome"] not in OUTCOMES or row["brief_ack"] not in {"", "yes", "no"}
            or row["confidence"] not in {"1", "2", "3"} or not row["language"]):
        raise ValueError("each judgment needs a valid outcome, language and confidence")
    languages = {"hi": "hindi", "id": "indonesian", "ur": "urdu"}
    row["language"] = languages.get(row["language"], row["language"])


def candidate_ballot(path, key):
    result = {}
    seen = set()
    for row in read_rows(path, CANDIDATE_FIELDS):
        name = row.pop("example")
        if name not in key or name in seen:
            raise ValueError("unknown or duplicate candidate in ballot")
        seen.add(name)
        if any(row.values()):
            validate_judgment(row)
            result[name] = row
    if seen != set(key):
        raise ValueError("candidate ballot membership changed")
    return result


def continuous_ballot(path, key):
    result = defaultdict(list)
    for row in read_rows(path, CONTINUOUS_FIELDS):
        name = row.pop("recording")
        channel = row.pop("focal_channel")
        if name not in key or channel not in CHANNELS:
            raise ValueError("unknown continuous clip or channel")
        validate_judgment(row)
        duration = key[name]["samples"] * 1000 // RATE
        stop = int(row.pop("stop_ms"))
        onset = row.pop("next_onset_ms")
        next_channel = row.pop("next_channel")
        if not 0 <= stop <= duration - 300:
            raise ValueError("stop has insufficient room for 300 ms of silence")
        if onset:
            onset = int(onset)
            if not stop + 300 <= onset < duration or next_channel not in CHANNELS:
                raise ValueError("invalid next onset or channel")
        elif next_channel or row["outcome"] in CLEAR:
            raise ValueError("a clear continuation or handoff needs a next onset and channel")
        else:
            onset = None
        result[name, CHANNELS[channel]].append((stop, onset, next_channel, row))
    for spans in result.values():
        spans.sort(key=lambda span: span[0])
        if any(second[0] - first[0] < 300 for first, second in zip(spans, spans[1:])):
            raise ValueError("overlapping 300 ms pause proposals in a single ballot")
    return result


def compare_rows(first, second, counts):
    counts["rated_by_both"] += 1
    same_language = first["language"] == second["language"] and first["language"] not in {"mixed", "other", "unknown"}
    same_outcome = first["outcome"] == second["outcome"]
    counts["same_known_language"] += same_language
    counts["same_outcome"] += same_outcome
    counts["either_uncertain_or_invalid"] += any(
        row["outcome"] in {"uncertain", "invalid_pause"} for row in (first, second))
    if same_language and same_outcome and first["outcome"] in CLEAR:
        counts[f"agreed_{first['outcome']}"] += 1


def compare_candidates(key, first, second):
    counts = defaultdict(Counter)
    for name, item in key.items():
        group = counts[item["source"], item["nominal_language"], item["focus_channel"]]
        group["proposals"] += 1
        group["rated_first"] += name in first
        group["rated_second"] += name in second
        if name in first and name in second:
            compare_rows(first[name], second[name], group)
    return [{"source": source, "nominal_language": language, "source_channel": channel, **values}
            for (source, language, channel), values in sorted(counts.items())]


def compare_continuous(first, second):
    counts = Counter()
    for name in first.keys() | second.keys():
        left, right = first.get(name, []), second.get(name, [])
        counts["marks_first"] += len(left)
        counts["marks_second"] += len(right)
        for a in left:
            match = [b for b in right if abs(a[0] - b[0]) <= 80]
            if not match:
                continue
            b, = match
            compare_rows(a[3], b[3], counts)
            counts["next_role_and_onset_agreement"] += (a[1] is not None and b[1] is not None
                                                       and a[2] == b[2] and abs(a[1] - b[1]) <= 80)
    return counts


def run(root, mode, first, second):
    if first.samefile(second):
        raise ValueError("supply separately completed reviewer ballots")
    verify(root)
    report_name = JOB if mode == "candidates" else "continuous-spring-floor-review"
    packet = json.loads((root / "reports" / f"{report_name}.json").read_text())
    key = json.loads((root / packet["packet"]).parent.joinpath("private-key.json").read_text())
    if mode == "candidates":
        a, b = (candidate_ballot(path, key) for path in (first, second))
        comparison = compare_candidates(key, a, b)
    else:
        a, b = (continuous_ballot(path, key) for path in (first, second))
        comparison = compare_continuous(a, b)
    if not a or not b:
        raise ValueError("both ballots must contain independent judgments")
    result = {"scope": "agreement and coverage only; training eligibility requires review",
              "packet_key_sha256": packet["key_sha256"], "mode": mode,
              "ballot_sha256": [file_digest(path) for path in (first, second)], "comparison": comparison}
    folder = root / "reports" / f"{JOB}-judgments"
    folder.mkdir(parents=True, exist_ok=True)
    write_to = folder / f"{mode}-{result['ballot_sha256'][0][:12]}-{result['ballot_sha256'][1][:12]}.json"
    _write_json(write_to, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("candidates", "continuous"))
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    run(data_root(), args.mode, args.first, args.second)
