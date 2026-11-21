"""Audit timing and label coverage for staged HarperValleyBank calls."""

import argparse
from collections import Counter
import hashlib
import json
from statistics import median

from .ami_calibration import _write_json
from .ami_pilot import annotated_pauses, group_words
from .audio import RATE, read_audio, silero
from .harper_valley_plan import REVISION
from .labels import Speech, turns
from .progress import Progress
from .sources import data_root

ROLES = ("caller", "agent")
GAP = RATE * 300 // 1000
MAX_ORIGIN_SPREAD_MS = 40
BOUNDARY_TOLERANCE = RATE * 160 // 1000


def recording_origin(segments):
    """Place a role's channel-local sample zero on the shared call timeline."""
    offsets = [float(row["start_ms"]) - float(row["offset_ms"]) for row in segments]
    if not offsets or max(offsets) - min(offsets) > MAX_ORIGIN_SPREAD_MS:
        return None
    return round(median(offsets) * RATE / 1000)


def transcript_words(segments, role, channel_length, origin, *, omissions=None, clock="offset_ms"):
    for utterance in segments:
        offsets = utterance.get("word_offsets_ms") or []
        durations = utterance.get("word_durations_ms") or []
        segment_start_ms = float(utterance[clock])
        incomplete = not offsets or len(offsets) != len(durations)
        for offset, duration in zip(offsets, durations):
            start_ms = segment_start_ms + float(offset)
            start = round(start_ms * RATE / 1000)
            end = round((start_ms + float(duration)) * RATE / 1000)
            if 0 <= start < end <= channel_length:
                yield Speech(role, start + origin, end + origin)
            else:
                incomplete = True
        if incomplete and omissions is not None:
            length_ms = float(utterance.get("duration_ms") or 0)
            start = max(0, round(segment_start_ms * RATE / 1000))
            end = min(channel_length, round((segment_start_ms + length_ms) * RATE / 1000))
            if not 0 <= start < end <= channel_length:
                start, end = 0, channel_length
            omissions.append(Speech(role, start + origin, end + origin))


def label_support(vad, word_stream, common_start, common_end, *, omissions=()):
    """Compare aligned two-party VAD and word-timing labels inside shared audio."""
    def crop(spans):
        for span in spans:
            start, end = max(common_start, span.start), min(common_end, span.end)
            if start < end:
                yield Speech(span.speaker, start, end)

    heard = list(crop(vad))
    words = list(crop(word_stream))
    reference = [(caller.end, following.start, label) for caller, following, label, _ in
                 annotated_pauses(group_words(words), {}, minimum_gap=GAP) if caller.speaker == "caller"]
    outcomes = Counter()
    for caller, gap, label in turns(heard, decision_ms=300):
        if caller.speaker != "caller":
            continue
        outcomes[f"candidate_{label}"] += 1
        edge = caller.end
        first = edge + gap
        if any(span.start < first + BOUNDARY_TOLERANCE and span.end > edge - BOUNDARY_TOLERANCE
               for span in omissions):
            outcomes[f"incomplete_timing_{label}"] += 1
            continue
        matches = [(end, onset, vote) for end, onset, vote in reference
                   if abs(end - edge) <= BOUNDARY_TOLERANCE and abs(onset - first) <= BOUNDARY_TOLERANCE]
        if len(matches) == 1:
            (word_end, word_onset, vote), = matches
            outcome = "supported" if vote == label else "disagreed"
            if vote == label and word_end <= edge and word_onset >= edge + GAP:
                outcomes[f"fixed_window_supported_{label}"] += 1
        else:
            outcome = "ambiguous" if matches else "unmatched"
        outcomes[f"{outcome}_{label}"] += 1
    return outcomes


def word_overlap(words, intervals):
    if not words:
        return 0.0
    return sum(any(word.start < end and start < word.end for start, end in intervals)
               for word in words) / len(words)


def use_conversation_clock(segments, channel_length, vad):
    """Confirm a metadata-defined clock on both alternating segment halves."""
    for half in (segments[::2], segments[1::2]):
        original = list(transcript_words(half, "agent", channel_length, 0))
        alternate = list(transcript_words(half, "agent", channel_length, 0, clock="start_ms"))
        if min(len(original), len(alternate)) < 4:
            return False
        baseline = word_overlap(original, vad)
        recovered = word_overlap(alternate, vad)
        if recovered < 0.8 or recovered - baseline < 0.2:
            return False
    return True


def audit_call(directory, identifier, detect, *, recover_clock=False):
    transcript = directory / "data" / "transcript" / f"{identifier}.json"
    segments = {role: [] for role in ROLES}
    for row in json.loads(transcript.read_text()):
        role = row["speaker_role"]
        if role not in segments:
            raise ValueError("unexpected HarperValleyBank transcript speaker role")
        segments[role].append(row)
    origins = {role: recording_origin(parts) for role, parts in segments.items()}
    if None in origins.values():
        return Counter(inconsistent_or_missing_channel_origin=1)
    audio = {role: read_audio(directory / "data" / "audio" / role / f"{identifier}.wav") for role in ROLES}
    support = {role: list(detect(audio[role])) for role in ROLES}
    counts = Counter()
    agent_clock = "offset_ms"
    if recover_clock and origins["agent"] <= -RATE and use_conversation_clock(
            segments["agent"], len(audio["agent"]), support["agent"]):
        # For this subset, the published agent PCM begins near conversation
        # zero. Caller PCM retains its independently verified recording origin.
        origins["agent"] = 0
        agent_clock = "start_ms"
        counts["agent_clock_recovered"] = 1
    baseline = min(origins.values())
    origins = {role: origin - baseline for role, origin in origins.items()}
    common_start = max(origins.values())
    common_end = min(origins[role] + len(audio[role]) for role in ROLES)
    if common_end - common_start < RATE:
        return counts + Counter(insufficient_shared_audio=1)
    vad = [Speech(role, start + origins[role], end + origins[role]) for role in ROLES
           for start, end in support[role]]
    omissions = []
    words = [word for role in ROLES for word in transcript_words(
        segments[role], role, len(audio[role]), origins[role], omissions=omissions,
        clock=agent_clock if role == "agent" else "offset_ms")]
    if omissions:
        counts["segments_with_incomplete_word_timing"] = len(omissions)
    # Presence of timed words and their overlap with detected speech is a check
    # on recording-local timestamps before either stream proposes boundaries.
    timed = {role: [span for span in words if span.speaker == role] for role in ROLES}
    if any(not spans for spans in timed.values()):
        return counts + Counter(missing_word_timing=1)
    overlap = {role: word_overlap(timed[role], [(start + origins[role], end + origins[role])
                                               for start, end in support[role]]) for role in ROLES}
    if min(overlap.values()) < 0.8:
        return counts + Counter(low_audio_word_overlap=1)
    return counts + Counter(audited_calls=1) + label_support(vad, words, common_start, common_end, omissions=omissions)


def run(root, *, recover_clock=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("HarperValleyBank audio auditing uses the campaign machine")
    manifest = root / "cache" / "harper-valley" / f"{REVISION}-selection.json"
    report = root / "reports" / "harper-valley-plan.json"
    expected = json.loads(report.read_text())["selection_sha256"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError("HarperValleyBank selection differs from the reviewed manifest")
    selection = json.loads(manifest.read_text())
    raw = root / "raw" / "harper-valley"
    calls = {split: selection[split] for split in ("train", "validation")}
    for cases in calls.values():
        for case in cases:
            for file in case["files"]:
                path = raw / file["path"]
                if not path.is_file() or path.stat().st_size != file["bytes"]:
                    raise ValueError("expected staged HarperValleyBank audio and transcripts for the audit")
    detect = silero()
    results = {}
    meter = Progress("harper-valley-label-audit", sum(map(len, calls.values())), "calls")
    try:
        for split, cases in calls.items():
            totals = Counter()
            coverage = {key: Counter() for key in (
                "supported_hold", "supported_shift", "fixed_window_supported_hold", "fixed_window_supported_shift",
            )}
            for case in cases:
                counts = audit_call(raw, case["id"], detect, recover_clock=recover_clock)
                totals.update(counts)
                for key, voices in coverage.items():
                    if counts[key]:
                        voices[case["caller"]] += counts[key]
                meter.advance(1)
            results[split] = {"calls": len(cases), "counts": totals,
                              "caller_speakers_with_support": {key: len(voices) for key, voices in coverage.items()},
                              "supported_pauses_per_caller": {
                                  key: {"median": median(voices.values()), "maximum": max(voices.values())}
                                  for key, voices in coverage.items() if voices
                              }}
        suffix = "-clock-recovery" if recover_clock else ""
        destination = root / "reports" / f"harper-valley-label-audit{suffix}.json"
        _write_json(destination, {"selection_sha256": expected, "decision_ms": 300,
                                  "boundary_tolerance_ms": BOUNDARY_TOLERANCE * 1000 // RATE,
                                  "agent_clock_recovery": recover_clock,
                                  "splits": results})
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-agent-clock", action="store_true")
    args = parser.parse_args()
    print(run(data_root(), recover_clock=args.recover_agent_clock))
