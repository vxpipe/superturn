"""Freeze causal text inputs for the staged HarperValley development cohort."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .ami_calibration import _write_json
from .audio import RATE, silero
from .basis_stage1 import events
from .progress import Progress
from .sources import data_root
from .turn_generation import normalize
from .turn_training_dataset import model_text


def prefix_turns(vad, edge, common_start, observed_from_start):
    """Walk back through observed role runs only, ignoring events after the stop."""
    prefix = sorted((span.start, min(span.end, edge), span.speaker)
                    for span in vad if span.start < edge and span.end > common_start)
    if not prefix or prefix[-1][1:] != (edge, "caller"):
        return None, "no_caller_edge"
    runs = []
    for start, stop, role in prefix:
        if runs and role == runs[-1][2]:
            left, end, _, duration = runs[-1]
            runs[-1] = left, max(end, stop), role, duration + max(0, stop - max(end, start))
        else:
            runs.append((start, stop, role, stop - start))
    current = runs[-1]
    if current[0] < common_start:
        return None, "current_start_cropped"
    if len(runs) == 1:
        return (None, current), None if observed_from_start else "previous_turn_unknown"
    prior = runs[-2]
    if prior[0] < common_start or prior[1] > current[0] or prior[3] < RATE:
        return None, "previous_turn_ambiguous"
    if len(runs) >= 3 and runs[-3][1] > prior[0]:
        return None, "previous_turn_ambiguous"
    return (prior, current), None


def observed_words(segments, role, clock, origin, left, cutoff):
    words = []
    for row in segments:
        if row["speaker_role"] != role:
            continue
        start_ms = float(row[clock])
        start = round(start_ms * RATE / 1000) + origin
        if start >= cutoff:
            continue
        offsets, durations = row.get("word_offsets_ms") or [], row.get("word_durations_ms") or []
        segment_end = start + round(float(row["duration_ms"]) * RATE / 1000)
        if not offsets and segment_end > left:
            return None
        pieces = [normalize(piece).split() for piece in (row.get("transcript") or "").split()]
        for index, offset in enumerate(offsets):
            word_start_ms = start_ms + float(offset)
            word_start = round(word_start_ms * RATE / 1000) + origin
            if word_start < left or word_start >= cutoff:
                continue
            if (index >= len(durations) or index >= len(pieces)
                    or len(pieces[index]) != 1 or float(durations[index]) <= 0):
                return None
            word_end = round((word_start_ms + float(durations[index])) * RATE / 1000) + origin
            if word_start < word_end <= cutoff:
                words.append((word_start, word_end, pieces[index][0]))
    ordered = sorted(words)
    if any(left_word[1] > right_word[0] for left_word, right_word in zip(ordered, ordered[1:])):
        return None
    return [token for _, _, token in ordered]


def causal_example(segments, origins, agent_clock, vad, edge, common_start):
    turns, issue = prefix_turns(vad, edge, common_start, origins["caller"] == origins["agent"] == common_start)
    if issue:
        return None, issue
    previous, current = turns
    current_words = observed_words(segments, "caller", "offset_ms", origins["caller"], current[0], edge)
    if not current_words:
        return None, "current_text_unavailable"
    previous_words = []
    if previous is not None:
        previous_words = observed_words(segments, "agent", agent_clock, origins["agent"],
                                        previous[0], previous[1])
        if not previous_words:
            return None, "previous_text_unavailable"
    prior = normalize(" ".join(previous_words))
    partial = normalize(" ".join(current_words))
    lagged = normalize(" ".join(current_words[:-1]))
    return {
        "text": model_text(prior, partial),
        "current_only": model_text("", partial),
        "last_word_missing": model_text(prior, lagged) if lagged else None,
    }, None


def run(root):
    from .harper_partial_stt import pinned_membership
    from .harper_valley_audit import audit_call

    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark inputs stay on Vast")
    pinned, membership = pinned_membership(root, "train")
    by_call = defaultdict(dict)
    for _, call, local_stop, label, person in pinned:
        by_call[call][local_stop, label] = person
    raw = root / "raw/harper-valley"
    destination = root / "cache/harper-valley/text-dev-v1.jsonl"
    counts = Counter()
    records = []
    detect = silero()
    meter = Progress("harper-causal-text-dev", len(by_call), "calls")
    for call, candidates in by_call.items():
        transcript = json.loads((raw / "data/transcript" / f"{call}.json").read_text())
        reproduced, evidence = [], {}

        def observe(audio, origins, vad, _words, _omissions, start, end):
            tracks = {role: pcm[start - origins[role]:end - origins[role]] for role, pcm in audio.items()}
            spans = [(max(start, v.start) - start, min(end, v.end) - start, v.speaker)
                     for v in vad if v.start < end and v.end > start]
            _, qualified = events(spans, {role: [(0, end - start)] for role in tracks},
                                  [], end - start, tracks=tracks)
            for local_stop, local_next, label, focal, _ in qualified:
                native_stop = local_stop + start - origins["caller"]
                if focal != "caller" or (native_stop, label) not in candidates:
                    continue
                counts[f"{label}_waveform_eligible"] += 1
                packed, reason = causal_example(transcript, origins, evidence["agent_clock"], vad,
                                                local_stop + start, start)
                if reason:
                    counts[f"{label}_{reason}"] += 1
                    continue
                gap = (local_next - local_stop) / (RATE / 1000)
                records.append({
                    "call": call, "caller": candidates[native_stop, label], "caller_sample": native_stop,
                    "gap_ms": gap, "label": label, **packed,
                })
                counts[f"{label}_packed"] += 1
                counts[f"{label}_{'short' if gap <= 800 else 'long'}"] += 1
                if packed["last_word_missing"] is None:
                    counts[f"{label}_lagged_blank"] += 1
                if packed["text"].startswith("---\n"):
                    counts[f"{label}_first_turn"] += 1

        audit_call(raw, call, detect, recover_clock=True, pauses=reproduced, evidence=evidence, observe=observe)
        if sorted(reproduced) != sorted(candidates):
            raise ValueError("the Harper development membership changed")
        meter.advance(1)
    records.sort(key=lambda row: (row["call"], row["caller_sample"]))
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    temporary.replace(destination)
    result = {"scope": "causal English development inputs awaiting label qualification",
              "archived_membership_sha256": membership, "rows": len(records),
              "callers": len({row["caller"] for row in records}),
              "serialized_sha256": sha256(payload).hexdigest(), "counts": dict(counts),
              "text_timing": "machine transcript word ends at the Silero stop; a delivered-STT proxy"}
    _write_json(root / "reports/harper-text-dev-packing.json", result)
    meter.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(data_root())
