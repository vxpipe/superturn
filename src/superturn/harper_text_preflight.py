"""Check the open English benchmark and transcript coverage without scoring it."""

import argparse
from collections import Counter, defaultdict
import json

from .ami_calibration import _write_json
from .audio import RATE, silero
from .basis_stage1 import events
from .harper_partial_stt import pinned_membership
from .harper_transcript_coverage import caller_timing, coverage, tokens
from .harper_valley_audit import audit_call, recording_origin
from .harper_valley_plan import REVISION, SPLITS
from .progress import Progress
from .sources import data_root


def preceding_agent_text(segments, caller_origin, edge, words):
    """Check a candidate prior agent segment, pending turn-level validation."""
    deadline = (caller_origin + edge) * 1000 / RATE
    agent = []
    for row in segments:
        if row["speaker_role"] != "agent":
            continue
        start = float(row["start_ms"])
        end = start + float(row["duration_ms"])
        if 0 < end <= deadline:
            agent.append((end, row))
    if not agent:
        return "no_agent_segment_before_edge"
    end, prior = max(agent, key=lambda pair: pair[0])
    local_end = round(end * RATE / 1000) - caller_origin
    if not any(local_end <= word[0] < edge for word in words):
        return "prior_turn_boundary_unclear"
    offsets, durations = prior.get("word_offsets_ms") or [], prior.get("word_durations_ms") or []
    corrected = tokens(prior.get("human_transcript") or "")
    machine = tokens(prior.get("transcript") or "")
    if not offsets or len(offsets) != len(durations) or len(offsets) != len(corrected) or corrected != machine:
        return "prior_agent_word_timing_unavailable"
    start = float(prior["start_ms"])
    if any(float(offset) < 0 or float(duration) <= 0 or start + float(offset) + float(duration) > deadline
           for offset, duration in zip(offsets, durations, strict=True)):
        return "prior_agent_timing_crosses_cutoff"
    return "two_turn_timing_supported"


def runtime_label_coverage(raw, grouped, text_candidates):
    detect = silero()
    counts, attrition, voices = Counter(), Counter(), defaultdict(set)
    meter = Progress("harper-text-dev-waveform-labels", len(grouped), "calls")
    for call, pauses in grouped.items():
        strict = {}

        def observe(audio, origins, vad, _words, _omissions, start, end):
            tracks = {role: pcm[start - origins[role]:end - origins[role]] for role, pcm in audio.items()}
            span = [(max(start, event.start) - start, min(end, event.end) - start, event.speaker)
                    for event in vad if event.start < end and event.end > start]
            connected = {role: [(0, end - start)] for role in tracks}
            vetoes, events_found = events(span, connected, [], end - start, tracks=tracks)
            attrition.update(vetoes)
            strict.update({(stop + start - origins["caller"], outcome): onset - stop
                           for stop, onset, outcome, focal, _ in events_found if focal == "caller"})

        reproduced = []
        audit_call(raw, call, detect, recover_clock=True, pauses=reproduced, observe=observe)
        if sorted(reproduced) != sorted((edge, label) for edge, label, _ in pauses):
            raise ValueError("the archived Harper pause timing changed")
        for edge, label, caller in pauses:
            gap = strict.get((edge, label))
            if gap is None:
                counts[f"{label}_filtered"] += 1
                continue
            condition = "300_to_800_ms" if gap <= RATE * 8 // 10 else "over_800_ms"
            counts[f"{label}_{condition}"] += 1
            if (call, edge, label) in text_candidates:
                counts[f"{label}_{condition}_with_candidate_text"] += 1
                voices[label, condition].add(caller)
        meter.advance(1)
    meter.finish()
    return {"archived_labels_retained_by_runtime_band": dict(counts),
            "waveform_filter_counts_both_roles": dict(attrition),
            "candidate_text_callers_by_band": {f"{label}_{condition}": len(people)
                                               for (label, condition), people in voices.items()}}


def run(root, *, runtime_labels=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Harper text benchmark inventory uses staged Vast data")
    rows, membership_digest = pinned_membership(root, "train")
    path = root / "cache" / "harper-valley" / f"{REVISION}-selection.json"
    roles = json.loads(path.read_text())
    people = {role: {call[side] for call in calls for side in ("caller", "agent")}
              for role, calls in roles.items()}
    if len(set.union(*people.values())) != sum(map(len, people.values())):
        raise ValueError("a Harper identity crosses benchmark roles")
    grouped = defaultdict(list)
    for _, call, edge, label, caller in rows:
        grouped[call].append((edge, label, caller))
    counts, callers, calls = defaultdict(Counter), defaultdict(set), defaultdict(set)
    text_candidates = set()
    meter = Progress("harper-text-dev-transcripts", len(rows), "pauses")
    for call, pauses in grouped.items():
        transcript = root / "raw" / "harper-valley" / "data" / "transcript" / f"{call}.json"
        segments = json.loads(transcript.read_text())
        caller_origin = recording_origin([row for row in segments if row["speaker_role"] == "caller"])
        words, missing = caller_timing(segments)
        for edge, label, caller in pauses:
            outcome, _ = coverage(words, missing, edge)
            if outcome.startswith("aligned_") and caller_origin is not None:
                counts[label]["current_timing_supported"] += 1
                outcome = preceding_agent_text(segments, caller_origin, edge, words)
                if outcome == "two_turn_timing_supported":
                    callers[label].add(caller)
                    calls[label].add(call)
                    text_candidates.add((call, edge, label))
            counts[label][outcome] += 1
            meter.advance(1)
    meter.finish()
    report = {
        "scope": "archived English development pause timing feasibility, no model scores or test reads",
        "revision": REVISION,
        "development_membership_sha256": membership_digest,
        "development_pauses": dict(Counter(label for _, _, _, label, _ in rows)),
        "development_coverage": {label: {**counts[label], "two_turn_callers": len(callers[label]),
                                          "two_turn_calls": len(calls[label])}
                                 for label in ("hold", "shift")},
        "roles": {
            target: {"calls": len(roles[source]), "people": len(people[source]),
                     "caller_voices": len({call["caller"] for call in roles[source]})}
            for source, target in zip(SPLITS, ("dev", "calibration", "test"), strict=True)
        },
    }
    report["text_evidence"] = "corrected caller prefix and candidate preceding agent segment"
    if runtime_labels:
        report["runtime_label_coverage"] = runtime_label_coverage(
            root / "raw" / "harper-valley", grouped, text_candidates)
    _write_json(root / "reports" / "harper-text-benchmark-preflight.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-labels", action="store_true", help="replay strict short-gap eligibility")
    run(data_root(), runtime_labels=parser.parse_args().runtime_labels)
