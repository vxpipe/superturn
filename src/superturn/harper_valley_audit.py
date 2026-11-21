"""Audit timing and label coverage for staged HarperValleyBank calls."""

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


def transcript_words(segments, role, channel_length, origin):
    for utterance in segments:
        offsets = utterance.get("word_offsets_ms") or []
        durations = utterance.get("word_durations_ms") or []
        if len(offsets) != len(durations):
            continue
        for offset, duration in zip(offsets, durations, strict=True):
            start_ms = float(utterance["offset_ms"]) + float(offset)
            start = round(start_ms * RATE / 1000)
            end = round((start_ms + float(duration)) * RATE / 1000)
            if 0 <= start < end <= channel_length:
                yield Speech(role, start + origin, end + origin)


def label_support(vad, word_stream, common_start, common_end):
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
        matches = [vote for end, onset, vote in reference
                   if abs(end - edge) <= BOUNDARY_TOLERANCE and abs(onset - first) <= BOUNDARY_TOLERANCE]
        if len(matches) == 1:
            outcome = "supported" if matches[0] == label else "disagreed"
        else:
            outcome = "ambiguous" if matches else "unmatched"
        outcomes[f"{outcome}_{label}"] += 1
    return outcomes


def audit_call(directory, identifier, detect):
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
    baseline = min(origins.values())
    origins = {role: origin - baseline for role, origin in origins.items()}
    audio = {role: read_audio(directory / "data" / "audio" / role / f"{identifier}.wav") for role in ROLES}
    common_start = max(origins.values())
    common_end = min(origins[role] + len(audio[role]) for role in ROLES)
    if common_end - common_start < RATE:
        return Counter(insufficient_shared_audio=1)
    vad = [Speech(role, start + origins[role], end + origins[role]) for role in ROLES
           for start, end in detect(audio[role])]
    words = [word for role in ROLES
             for word in transcript_words(segments[role], role, len(audio[role]), origins[role])]
    # Presence of timed words and their overlap with detected speech is a check
    # on recording-local timestamps before either stream proposes boundaries.
    timed = {role: [span for span in words if span.speaker == role] for role in ROLES}
    if any(not spans for spans in timed.values()):
        return Counter(missing_word_timing=1)
    support = {}
    for role in ROLES:
        intervals = [span for span in vad if span.speaker == role]
        support[role] = sum(any(word.start < span.end and span.start < word.end for span in intervals)
                            for word in timed[role]) / len(timed[role])
    if min(support.values()) < 0.8:
        return Counter(low_audio_word_overlap=1)
    return Counter(audited_calls=1) + label_support(vad, words, common_start, common_end)


def run(root):
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
            coverage = {"hold": set(), "shift": set()}
            for case in cases:
                counts = audit_call(raw, case["id"], detect)
                totals.update(counts)
                for label, voices in coverage.items():
                    if counts[f"supported_{label}"]:
                        voices.add(case["caller"])
                meter.advance(1)
            results[split] = {"calls": len(cases), "counts": totals,
                              "caller_speakers_with_support": {label: len(voices) for label, voices in coverage.items()}}
        destination = root / "reports" / "harper-valley-label-audit.json"
        _write_json(destination, {"selection_sha256": expected, "decision_ms": 300,
                                  "boundary_tolerance_ms": BOUNDARY_TOLERANCE * 1000 // RATE,
                                  "splits": results})
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    print(run(data_root()))
