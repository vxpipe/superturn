"""Measure corrected transcript coverage at the archived HarperValley pauses."""

from collections import Counter, defaultdict
import hashlib
import json
import re
from statistics import median

from .ami_calibration import _write_json
from .audio import RATE
from .harper_valley_audit import BOUNDARY_TOLERANCE, GAP
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .sources import data_root

MEMBERSHIP = "bcde950f78d22a7874c184a0af82e0dc062a22360182ac28b44855133be8f0d5"
LEXICAL = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def tokens(text):
    return LEXICAL.findall(text.casefold().replace("’", "'"))


def caller_timing(segments):
    words, missing = [], []
    for segment in segments:
        if segment["speaker_role"] != "caller":
            continue
        start = float(segment["offset_ms"])
        offsets, durations = segment["word_offsets_ms"], segment["word_durations_ms"]
        ranges = [(round((start + offset) * RATE / 1000),
                   round((start + offset + duration) * RATE / 1000))
                  for offset, duration in zip(offsets, durations)] if offsets and durations else []
        if (not ranges or len(ranges) != max(len(offsets), len(durations)) or
                any(left < 0 or left >= right for left, right in ranges) or
                ranges != sorted(ranges)):
            missing.append((round(start * RATE / 1000),
                            round((start + segment["duration_ms"]) * RATE / 1000)))
            continue
        parts = [tokens(part) for part in (segment["transcript"] or "").split()]
        machine = [part[0] for part in parts] if all(len(part) == 1 for part in parts) else []
        corrected = tokens(segment["human_transcript"] or "")
        for index, (left, right) in enumerate(ranges):
            words.append((left, right, index, len(ranges), machine, corrected))
    return words, missing


def coverage(words, missing, edge):
    lower, deadline = edge - BOUNDARY_TOLERANCE, edge + GAP
    if any(left < deadline and right >= lower for left, right in missing):
        return "local_timing_unavailable", None
    if any(left < deadline and right > edge for left, right, *_ in words):
        return "timed_word_in_decision_silence", None
    endpoints = [word for word in words if lower <= word[1] <= edge]
    if not endpoints:
        return "no_timed_endpoint", None
    latest = max(word[1] for word in endpoints)
    anchors = [word for word in endpoints if word[1] == latest]
    if len(anchors) != 1:
        return "ambiguous_timed_endpoint", None
    start, _, index, count, machine, corrected = anchors[0]
    if any(word is not anchors[0] and word[0] < latest and word[1] > start for word in words):
        return "ambiguous_timed_endpoint", None
    lag_ms = (edge - latest) * 1000 / RATE
    if len(machine) != count:
        return "machine_word_count_mismatch", lag_ms
    if machine[:index + 1] != corrected[:index + 1]:
        return "corrected_prefix_mismatch", lag_ms
    if corrected != machine:
        return "prefix_agrees_segment_differs", lag_ms
    return ("aligned_segment_final" if index + 1 == count else "aligned_segment_interior"), lag_ms


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this audit requires staged Vast data")
    reports = root / "reports"
    archived = json.loads((reports / "harper-history-replication.json").read_text())
    membership = json.loads((reports / "harper-history-replication-membership.json").read_text())
    digest = hashlib.sha256(json.dumps(membership).encode()).hexdigest()
    if (digest != MEMBERSHIP or archived["membership_sha256"] != digest or archived["split"] != "train"
            or len({(call, edge) for call, edge, _ in membership}) != len(membership)):
        raise ValueError("the archived HarperValley cohort changed")
    cases, _, audit = pinned_split(root, "train")
    if archived["audit_sha256"] != audit:
        raise ValueError("the archived timing audit changed")
    callers = {case["id"]: case["caller"] for case in cases}
    by_call = defaultdict(list)
    for call, edge, label in membership:
        if call not in callers or label not in ("hold", "shift"):
            raise ValueError("unexpected call or label in the archived cohort")
        by_call[call].append((edge, label))
    totals, voices = defaultdict(Counter), defaultdict(set)
    aligned_counts, lag = Counter(), defaultdict(list)
    meter = Progress("harper-transcript-coverage", len(membership), "pauses")
    try:
        for call, pauses in by_call.items():
            path = root / "raw/harper-valley/data/transcript" / f"{call}.json"
            words, missing = caller_timing(json.loads(path.read_text()))
            for edge, label in pauses:
                outcome, lag_ms = coverage(words, missing, edge)
                totals[label][outcome] += 1
                voices[label, outcome].add(callers[call])
                if outcome.startswith("aligned_"):
                    voices[label, "all_aligned"].add(callers[call])
                    aligned_counts[callers[call]] += 1
                    lag[label].append(lag_ms)
                meter.advance(1)
        if {label: sum(counts.values()) for label, counts in totals.items()} != {"hold": 273, "shift": 255}:
            raise ValueError("unexpected archived cohort counts")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    result = {"membership_sha256": digest, "decision_ms": 300,
              "outcomes": totals, "calls": len(by_call),
              "callers": len({callers[call] for call in by_call}),
              "callers_with_at_least_ten_aligned": sum(count >= 10 for count in aligned_counts.values()),
              "aligned_endpoint_lag_ms": {label: [min(values), median(values), max(values)]
                                          for label, values in lag.items()},
              "callers_per_label_outcome": {label: {outcome: len(ids) for (name, outcome), ids
                                                     in voices.items() if name == label}
                                             for label in totals}}
    _write_json(reports / "harper-transcript-coverage.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
