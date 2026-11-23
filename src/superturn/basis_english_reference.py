"""Measure English Basis label agreement without feeding a model the examples."""

from collections import Counter, defaultdict

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .basis_stage1 import events, read_segment, samples
from .progress import Progress

REFERENCE_AUDIO_CAP = 1_400_000_000
EDGE_TOLERANCE_SECONDS = .35


def unique_assignments(options):
    """Keep unambiguous one-to-one temporal matches and expose collisions."""
    claims = Counter(candidate for matches in options for candidate in matches)
    for matches in options:
        if not matches:
            yield None, "unmatched"
        elif len(matches) != 1 or claims[matches[0]] != 1:
            yield None, "ambiguous"
        else:
            yield matches[0], "matched"


def match_reference(judgments, proposals, word_gaps, pause_options):
    metrics, confusion = Counter(), Counter()
    options = []
    for judgment in judgments:
        focal, = judgment["speakers"]
        options.append([
            index for index, (stop, onset, _, voice, _) in enumerate(proposals)
            if voice == focal and abs(stop / RATE - judgment["start"]) <= EDGE_TOLERANCE_SECONDS
            and abs(onset / RATE - judgment["end"]) <= EDGE_TOLERANCE_SECONDS
        ])
    for judgment, (index, status) in zip(judgments, unique_assignments(options), strict=True):
        category = pause_options[judgment["label"]]
        metrics[f"{category}_publisher"] += 1
        metrics[f"{category}_{status}"] += 1
        if index is None:
            continue
        if category == "finished":
            focal, = judgment["speakers"]
            independently_observed = [
                next_voice for stop, onset, voice, next_voice in word_gaps
                if voice == focal and abs(stop - judgment["start"]) <= EDGE_TOLERANCE_SECONDS
                and abs(onset - judgment["end"]) <= EDGE_TOLERANCE_SECONDS
            ]
            if len(independently_observed) != 1 or independently_observed[0] == focal:
                metrics["finished_handoff_unconfirmed"] += 1
                continue
        reference = "hold" if category == "hold" else "shift"
        predicted = proposals[index][2]
        confusion[f"{reference}_predicted_{predicted}"] += 1
    return metrics, confusion


def filter_human_exclusions(row, proposals):
    vetoes = [
        (mark["start"] * RATE, mark["end"] * RATE,
         "human_backchannel" if mark["question"] == "backchannel" else "human_overlap")
        for mark in row["human_judgments"]
        if mark["question"] == "backchannel" or
        (mark["question"] == "overlap" and mark["label"] in {"competitive", "simultaneous_start"})
    ]
    retained, counts = [], Counter()
    for event in proposals:
        stop, onset, _, _, _ = event
        reasons = {reason for start, end, reason in vetoes if start < onset + RATE and end > stop - RATE / 5}
        if reasons:
            counts.update(reasons)
        else:
            retained.append(event)
    return retained, counts


def agreement_scorecard(call_counts):
    rng = np.random.default_rng(23)
    labels = ("hold", "shift")
    correct = np.array([[call[f"{label}_predicted_{label}"] for label in labels]
                        for call in call_counts])
    counts = np.array([[sum(call[f"{label}_predicted_{outcome}"] for outcome in labels) for label in labels]
                       for call in call_counts])
    if not len(counts):
        return {}
    draws = rng.integers(0, len(counts), size=(5000, len(counts)))
    totals_correct, totals = correct.sum(axis=0), counts.sum(axis=0)
    drawn_correct, drawn_counts = correct[draws].sum(axis=1), counts[draws].sum(axis=1)
    report = {}
    for column, label in enumerate((*labels, "overall")):
        if label == "overall":
            hits, n = int(totals_correct.sum()), int(totals.sum())
            sampled_hits, sampled_n = drawn_correct.sum(axis=1), drawn_counts.sum(axis=1)
        else:
            hits, n = int(totals_correct[column]), int(totals[column])
            sampled_hits, sampled_n = drawn_correct[:, column], drawn_counts[:, column]
        valid = sampled_n > 0
        interval = (np.quantile(sampled_hits[valid] / sampled_n[valid], [.025, .975]).tolist()
                    if valid.any() else None)
        report[label] = {"correct": hits, "n": n, "agreement": hits / n if n else None,
                         "conversation_bootstrap_95": interval, "valid_bootstrap_draws": int(valid.sum())}
    return report


def run_reference_audit(root, fs, prefix, segments, judgments, audio_bytes, word_gaps, pause_options):
    if not 0 < audio_bytes <= REFERENCE_AUDIO_CAP:
        raise ValueError("the English reference audit exceeds its sealed audio cap")
    meter = Progress("basis-english-reference-audio", audio_bytes, "B")
    detect = silero()
    calls, attrition = defaultdict(Counter), Counter()
    try:
        for segment in segments:
            tracks = read_segment(fs, prefix, "en", segment, meter)
            activity = [(start, stop, voice) for voice, pcm in tracks.items()
                        for start, stop in detect(pcm)]
            connected = {person["speaker"]: samples(person["on_call"]) for person in segment["speakers"]}
            counts, proposals = events(activity, connected, samples(segment["muted_spans"]),
                                       min(map(len, tracks.values())))
            retained, veto_counts = filter_human_exclusions(segment, proposals)
            attrition.update(counts)
            attrition.update(veto_counts)
            local = calls[segment["conversation_id"]]
            local["segments"] += 1
            coverage, confusion = match_reference(judgments[segment["segment_id"]],
                                                  retained, word_gaps(segment), pause_options)
            local.update(coverage)
            local.update(confusion)
        if meter.done != audio_bytes:
            raise ValueError("English audit bytes did not match the sealed projection")
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    totals = sum(calls.values(), Counter())
    report = {
        "scope": "English Basis publisher-reference diagnostic on the audit role",
        "audio_bytes": meter.done,
        "calls": len(calls),
        "event_attrition": dict(attrition),
        "reference_counts": dict(totals),
        "contributing_calls": {
            label: sum(any(counts[f"{label}_predicted_{outcome}"] for outcome in ("hold", "shift"))
                       for counts in calls.values())
            for label in ("hold", "shift")
        },
        "anonymous_call_counts": sorted((dict(counts) for counts in calls.values()),
                                        key=lambda counts: sorted(counts.items())),
        "agreement": agreement_scorecard(list(calls.values())),
        "edge_tolerance_ms": round(EDGE_TOLERANCE_SECONDS * 1000),
    }
    _write_json(root / "reports/basis-english-reference-audit.json", report)
    return report
