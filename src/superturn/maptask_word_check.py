"""Check fixed Map Task acoustic pause candidates against independent word timings."""

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from .ami_calibration import _write_json
from .ami_pilot import _peak, group_words
from .audio import RATE, silero
from .dense_teacher import file_digest, normalize
from .harper_valley_audit import BOUNDARY_TOLERANCE, GAP
from .labels import Speech, turns
from .maptask_probe import FILES, SOURCE, analyze, stage
from .progress import Progress
from .sources import data_root

ACKNOWLEDGMENTS = {"okay", "ok", "right", "yeah", "yes", "uhhuh", "mhm", "mmhmm"}


def words_for(model, signal):
    words, uncertain, counts = [], [], Counter()
    for index, pcm in enumerate(signal):
        result = model.transcribe(normalize(pcm), language="en", word_timestamps=True,
                                  condition_on_previous_text=False, temperature=0)
        for segment in result["segments"]:
            for word in segment.get("words", ()):
                start, end = (round(word[key] * RATE) for key in ("start", "end"))
                token = re.sub(r"[^a-z]", "", word["word"].lower())
                if not token or not 0 <= start < end <= len(pcm):
                    counts["invalid"] += 1
                    continue
                own, other = (_peak(track, start, end) for track in (pcm, signal[1 - index]))
                if own < max(128, 1.414 * other):
                    counts["weak_or_nondominant"] += 1
                    continue
                span = Speech(str(index), start, end)
                if word["probability"] < 0.3:
                    counts["uncertain_confidence"] += 1
                    uncertain.append(span)
                else:
                    words.append((span, token))
    retained = []
    for word, token in words:
        peers = [(max(word.start, other.start), min(word.end, other.end))
                 for other, text in words if other.speaker != word.speaker and token == text
                 and max(word.start, other.start) < min(word.end, other.end)]
        slot = int(word.speaker)
        levels = [(_peak(signal[slot], start, end), _peak(signal[1 - slot], start, end))
                  for start, end in peers]
        if all(own >= 1.414 * max(1.0, other) for own, other in levels):
            retained.append((word, token))
        elif any(other >= 1.414 * max(1.0, own) for own, other in levels):
            counts["duplicate_spill"] += 1
        else:
            counts["uncertain_duplicate"] += 1
            uncertain.append(word)
    return retained, uncertain, counts


def check(candidates, words, uncertain=()):
    spans = group_words(s for s, _ in words)
    by_class = {"hold": Counter(), "shift": Counter()}
    errors, matched_word_edges, cases = [], set(), []
    word_votes = {(span.speaker, span.end, span.end + gap): vote
                  for span, gap, vote in turns(spans, decision_ms=300)}
    for caller, first, label in candidates:
        counters = by_class[label]
        ends = [span.end for span in spans if span.speaker == caller.speaker
                and abs(span.end - caller.end) <= BOUNDARY_TOLERANCE]
        row = {"caller_channel": caller.speaker, "end_sample": caller.end,
               "next_channel": first.speaker, "next_sample": first.start, "acoustic_label": label}
        if len(ends) != 1:
            outcome = "ambiguous_word_end" if ends else "no_word_end"
        else:
            end, = ends
            delta = (end - caller.end) * 1000 / RATE
            errors.append(delta)
            row["word_end_delta_ms"] = delta
            following = [span for span in spans if span.start >= end]
            next_spans = [span for span in following if span.start == following[0].start
                          and abs(span.start - first.start) <= BOUNDARY_TOLERANCE]
            if len(next_spans) != 1:
                outcome = "ambiguous_next_onset" if next_spans else "no_next_onset"
            else:
                next_span, = next_spans
                matched_word_edges.add((caller.speaker, end, next_span.start))
                tokens = [token for word, token in words if word.speaker == next_span.speaker
                          and next_span.start <= word.start < next_span.end]
                if any(span.end > caller.end and span.start < max(caller.end + GAP, next_span.start)
                       for span in uncertain):
                    outcome = "uncertain_intervening_word"
                elif next_span.speaker != first.speaker:
                    outcome = "next_channel_conflict"
                elif end > caller.end or any(span.end > caller.end and span.start < caller.end + GAP for span in spans):
                    outcome = "window_silence_unconfirmed"
                elif (next_span.speaker != caller.speaker and len(tokens) <= 2 and tokens
                      and all(token in ACKNOWLEDGMENTS for token in tokens)):
                    outcome = "acknowledgment_mediated"
                else:
                    vote = word_votes.get((caller.speaker, end, next_span.start))
                    if vote == label:
                        outcome = "supported"
                    elif vote is not None:
                        outcome = "label_conflict"
                    else:
                        outcome = "no_word_pause_vote"
        counters[outcome] += 1
        cases.append({**row, "outcome": outcome})
    return {"by_class": by_class, "cases": cases,
            "unmatched_word_candidates": len(word_votes.keys() - matched_word_edges)}, errors


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Map Task media stays on Vast")
    import torch
    import whisper

    checkpoint = Path.home() / ".cache" / "whisper" / "tiny.pt"
    if not checkpoint.is_file():
        raise ValueError("the existing Whisper Tiny checkpoint is required")
    model = whisper.load_model(str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu")
    previous = root / "reports" / "maptask-audio-probe.json"
    manifest = json.loads(previous.read_text())
    detect = silero()
    results, errors = {}, []
    meter = Progress("maptask-word-check", len(FILES), "dialogues")
    try:
        for name in FILES:
            source = root / "raw" / SOURCE / "probe"
            if file_digest(source / name) != manifest["files"][name]["sha256"]:
                raise ValueError("staged Map Task slice changed")
            signal, _, _ = stage(source, name)
            candidates = []
            acoustic = analyze(signal, detect, candidates=candidates)
            if acoustic["preliminary_dominant_pauses"] != manifest["files"][name]["preliminary_dominant_pauses"]:
                raise ValueError("screened Silero candidates do not reproduce")
            words, uncertain, dropped = words_for(model, signal)
            summary, deviations = check(candidates, words, uncertain)
            results[name] = {**summary, "retained_words": len(words), "dropped_words": dropped}
            errors.extend(deviations)
            meter.advance(1)
        totals, class_counts = Counter(), {"hold": Counter(), "shift": Counter()}
        for row in results.values():
            for label, group in row["by_class"].items():
                totals.update(group)
                class_counts[label].update(group)
        compared_outcomes = ("supported", "label_conflict", "next_channel_conflict")
        comparable_by_class = {label: sum(counts[key] for key in compared_outcomes)
                               for label, counts in class_counts.items()}
        comparable = sum(comparable_by_class.values())
        agreement = totals["supported"] / comparable if comparable else None
        median = float(np.median(errors)) if errors else None
        iqr = float(np.quantile(errors, .75) - np.quantile(errors, .25)) if errors else None
        report = {"method": "independent word timing and channel screening, not semantic labels",
                  "acoustic_report_sha256": file_digest(previous), "whisper_sha256": file_digest(checkpoint),
                  "word_end_delta_ms": {"count": len(errors), "median": median, "iqr": iqr},
                  "matched_label_agreement": agreement, "comparable_cases": comparable,
                  "comparable_by_class": comparable_by_class, "outcomes": totals,
                  "engineering_pass": bool(agreement is not None and agreement >= .7
                                           and min(comparable_by_class.values()) >= 5
                                           and abs(median) <= 80 and iqr <= 160
                                           and totals["next_channel_conflict"] <= 1),
                  "files": results}
        _write_json(root / "reports" / "maptask-word-check.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run(data_root()).items() if k != "files"}, indent=2))
