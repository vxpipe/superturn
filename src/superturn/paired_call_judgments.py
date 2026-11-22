"""Audit two independent paired-call ballots before admitting training labels."""

from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import file_digest
from .paired_call_review import CONTINUOUS_REPORTS, JOB, WINDOW_FIELDS, verify
from .sources import data_root

CANDIDATE_FIELDS = ("example", "outcome", "brief_ack", "language", "confidence")
CONTINUOUS_FIELDS = ("recording", "focal_channel", "stop_ms", "next_onset_ms", "next_channel",
                     "outcome", "brief_ack", "language", "confidence")
OUTCOMES = {"resume", "handoff", "interruption", "uncertain", "invalid_pause"}
CLEAR = {"resume", "handoff"}
CHANNELS = {"left": 0, "right": 1}
LANGUAGES = {"hi": "hindi", "id": "indonesian", "ur": "urdu"}


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
            or row["confidence"] not in {"", "1", "2", "3"} or not row["language"]):
        raise ValueError("each judgment needs a valid outcome and language, and any confidence must be 1–3")
    row["language"] = LANGUAGES.get(row["language"], row["language"])


def window_ballot(path, key):
    seen, completed = set(), {}
    for row in read_rows(path, WINDOW_FIELDS):
        name = row.pop("recording")
        if name not in key or name in seen:
            raise ValueError("unknown or duplicate recording in window ballot")
        seen.add(name)
        if any(row.values()):
            if row["reviewed"] != "yes" or not row["left_language"] or not row["right_language"]:
                raise ValueError("a completed window needs a language entry for each channel")
            values = (row["left_language"], row["right_language"])
            completed[name] = tuple(LANGUAGES.get(value, value) for value in values)
    if seen != set(key):
        raise ValueError("window ballot membership changed")
    return completed


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


def compare_continuous(first, second, key=None, completed=None):
    counts = Counter()
    calls_by_outcome = defaultdict(set)
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
            temporal = (a[1] is not None and b[1] is not None and a[2] == b[2] and
                        abs(a[1] - b[1]) <= 80 and min(a[1], b[1]) - max(a[0], b[0]) >= 300)
            counts["next_role_and_onset_agreement"] += temporal
            if completed is not None:
                record, channel = name
                nominal = LANGUAGES[key[record]["nominal_language"]]
                outcomes = a[3]["outcome"], b[3]["outcome"]
                expected_next = channel if outcomes[0] == "resume" else 1 - channel
                window_languages = [ballot[record][channel] for ballot in completed]
                matches = (
                    temporal and outcomes[0] == outcomes[1] and outcomes[0] in CLEAR
                    and all(language == nominal for language in (*window_languages, a[3]["language"], b[3]["language"]))
                )
                joint = matches and CHANNELS[a[2]] == expected_next
                counts["joint_nominal_language_and_floor_matches"] += joint
                if joint:
                    counts[f"joint_{outcomes[0]}"] += 1
                    calls_by_outcome[outcomes[0]].add(key[record]["recording"])
                elif matches and outcomes[0] == "resume" and a[3]["brief_ack"] == b[3]["brief_ack"] == "yes":
                    counts["acknowledged_partner_first_resumes"] += 1
    if completed is not None:
        for outcome in sorted(CLEAR):
            counts[f"joint_{outcome}_calls"] = len(calls_by_outcome[outcome])
    return counts


def compare_anchor(key, events):
    counts = defaultdict(Counter)
    for recording, item in key.items():
        for ref in item["known_answers"]:
            outcome = ref["outcome"]
            group = counts[outcome]
            group["reference"] += 1
            for stop, onset, side, event in events.get((recording, CHANNELS["left"]), ()):
                if abs(stop - ref["stop_ms"]) <= 80:
                    group["found_stop"] += 1
                    group["matched_outcome_and_next_speaker"] += (
                        event["outcome"] == outcome and onset is not None and abs(onset - ref["onset_ms"]) <= 80
                        and side == ("left" if outcome == "resume" else "right")
                    )
    return counts


def run(root, mode, first, second, *, windows=None):
    if first.samefile(second):
        raise ValueError("supply separately completed reviewer ballots")
    report_name = JOB if mode == "candidates" else CONTINUOUS_REPORTS[mode]
    verify(root, (report_name,))
    packet = json.loads((root / "reports" / f"{report_name}.json").read_text())
    key = json.loads((root / packet["packet"]).parent.joinpath("private-key.json").read_text())
    is_datahive = mode == "continuous-datahive"
    if is_datahive and (not packet.get("window_ballot_required") or windows is None or windows[0].samefile(windows[1])):
        raise ValueError("supply the two separate DataHive window ballots")
    if mode == "candidates":
        a, b = (candidate_ballot(path, key) for path in (first, second))
        comparison = compare_candidates(key, a, b)
    else:
        a, b = (continuous_ballot(path, key) for path in (first, second))
        comparison = compare_continuous(a, b)
    if not is_datahive and (not a or not b):
        raise ValueError("both ballots must contain independent judgments")
    result = {"scope": "agreement and coverage only; training eligibility requires review",
              "packet_key_sha256": packet["key_sha256"], "mode": mode,
              "ballot_sha256": [file_digest(path) for path in (first, second)], "comparison": comparison}
    if mode == "continuous-control":
        result["scope"] = "English protocol control only"
        result["known_answer_counts_by_listener"] = [compare_anchor(key, ballot) for ballot in (a, b)]
    if is_datahive:
        completed = tuple(window_ballot(path, key) for path in windows)
        for events, reviewed in zip((a, b), completed, strict=True):
            if not {identifier for identifier, _ in events} <= reviewed.keys():
                raise ValueError("pause events require a completed window")
        both = completed[0].keys() & completed[1].keys()
        if not both:
            raise ValueError("no DataHive window has been reviewed by both listeners")
        result.pop("comparison")
        result["window_ballot_sha256"] = [file_digest(path) for path in windows]
        result["by_nominal_language"] = {}
        for language in sorted(packet["slices_by_nominal_language"]):
            scoped = {item for item, row in key.items() if row["nominal_language"] == language}
            shared = both & scoped
            matched_events = [{item: spans for item, spans in ballot.items() if item[0] in shared}
                              for ballot in (a, b)]
            result["by_nominal_language"][language] = {
                "windows_total": len(scoped), "windows_reviewed_by_both": len(shared),
                "windows_reviewed_per_listener": [len(scoped & ballot.keys()) for ballot in completed],
                "windows_without_events_per_listener": [
                    len(shared - {item[0] for item in ballot}) for ballot in matched_events
                ],
                "channel_language_agreement": Counter(
                    f"{left}/{right}" for item in sorted(shared)
                    for left, right in zip(completed[0][item], completed[1][item], strict=True)
                ),
                **compare_continuous(*matched_events, key, completed),
            }
    folder = root / "reports" / f"{JOB}-judgments"
    folder.mkdir(parents=True, exist_ok=True)
    hashes = [*result["ballot_sha256"], *result.get("window_ballot_sha256", [])]
    write_to = folder / f"{mode}-{'-'.join(digest[:12] for digest in hashes)}.json"
    _write_json(write_to, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("candidates", *CONTINUOUS_REPORTS))
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--windows", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    run(data_root(), args.mode, args.first, args.second, windows=args.windows)
