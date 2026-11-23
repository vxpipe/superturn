"""Check whether publisher word times support a causal deadline sensitivity."""

from collections import Counter, defaultdict
from hashlib import sha256
import json

import numpy as np
import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .audio import RATE, silero
from .progress import Progress
from .sources import data_root
from .timed_text_pause_diagnostic import FREEZE, SHARD, pack_rows
from .turn_generation import normalize
from .turn_text_train import digest

DEADLINES_MS = (0, 150, 300)


def aligned_words(raw):
    return sorted(
        (round(word["start"] * RATE), round(word["end"] * RATE), word["word"])
        for word in raw if word["word"] and word["start"] is not None and word["end"] is not None
        and 0 <= word["start"] < word["end"]
    )


def word_clock_pairs(words, spans, edges):
    """Match internal VAD stops to bounded overlapping word intervals."""
    offsets = []
    for start, stop in spans[:-1]:
        if any(abs(stop - edge) <= RATE * .32 for edge in edges):
            continue
        nearby = [end for left, end, _ in words if start < end and left < stop
                  and abs(stop - end) <= RATE * .32]
        if nearby:
            offsets.append(min(nearby, key=lambda end: abs(stop - end)) - stop)
    return np.array(offsets) * (1000 / RATE)


def rms(pcm):
    return max(1., float(np.sqrt(np.mean(np.square(pcm.astype(np.float64))))))


def straddle_energy(pcm, edge, end, resume):
    # Estimate the following quiet floor on separate 32 ms blocks. Use the
    # lower quartile so a longer tail cannot set its own reference.
    block = round(.032 * RATE)
    quiet = [rms(pcm[i:i + block]) for i in range(end, resume - block + 1, block)]
    if not quiet or not edge < end <= len(pcm):
        return "unmeasurable"
    difference = 20 * np.log10(rms(pcm[edge:end]) / np.quantile(quiet, .25))
    return "audible_tail" if difference >= 10 else "near_floor" if difference <= 3 else "intermediate"


def row_audit(row, edge, words, pcm, spans):
    outcome = Counter(rows=1)
    current = row["text"].split("\n")[1]
    gap_end = min(len(pcm), edge + round(row["gap_ms"] * RATE / 1000))
    resume = next((start for start, _ in spans if start >= edge), None) if not row["label"] else None
    if not row["label"]:
        if resume is None:
            outcome["hold_resume_unverified"] += 1
        elif abs(resume - gap_end) > round(.15 * RATE):
            outcome["hold_resume_outside_150_ms"] += 1
    crossing = [(start, end) for start, end, _ in words if start < edge < end <= edge + round(.3 * RATE)]
    duration = None
    if crossing:
        end = max(end for _, end in crossing)
        duration = (end - edge) * 1000 / RATE
        outcome["straddle"] += 1
        energy = straddle_energy(pcm, edge, end, min(resume, gap_end) if resume is not None else gap_end)
        outcome[energy] += 1
        quiet_end = min(resume, gap_end) if resume is not None else gap_end
        if energy == "audible_tail" and quiet_end - end < round(.3 * RATE):
            outcome["short_quiet_under_word_clock"] += 1
    for ms in DEADLINES_MS:
        cutoff = edge + round(ms * RATE / 1000)
        included = [(start, end, token) for start, end, token in words if start < edge and end <= cutoff]
        # Reproduce the frozen renderer's ordering for tied boundaries.
        rendered = normalize(" ".join(token for start, end, token in sorted(included, key=lambda w: (w[1], w[0], w[2]))))
        outcome[f"changed_at_{ms}_ms"] += rendered != current
        if resume is not None:
            outcome[f"resumed_words_at_{ms}_ms"] += sum(start + end >= 2 * resume for start, end, _ in included)
    return outcome, duration


def asymmetry(by_turn):
    counts = np.array([counts for counts in by_turn.values()])
    rng = np.random.default_rng(29)
    draws = rng.integers(len(counts), size=(3000, len(counts)))
    totals = counts[draws].sum(axis=1)
    valid = (totals[:, 0] > 0) & (totals[:, 2] > 0)
    hold, hold_hit, shift, shift_hit = totals[valid].T
    return np.quantile(shift_hit / shift - hold_hit / hold, [.025, .975]).tolist()


def run(root):
    from .harper_partial_stt import english_membership

    if not (root / ".campaign-machine").is_file():
        raise ValueError("publisher benchmark inputs stay on Vast")
    freeze = root / "reports" / FREEZE
    frozen = json.loads(freeze.read_text())
    rows, _, membership = pack_rows(root)
    if (frozen["membership_sha256"] != membership
            or frozen["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()):
        raise ValueError("deadline inputs differ from the original freeze")
    members, tracks, _ = english_membership(root)
    anchors = {rank: edge for rank, _, edge, _, _ in members}
    grouped = defaultdict(list)
    for row in rows:
        if row["has_previous"] and (row["gap_ms"] >= 1500 if row["label"] else 300 <= row["gap_ms"] < 800):
            grouped[row["call"]].append(row)
    source = pq.read_table(root / SHARD, columns=["words"])["words"].to_pylist()
    totals = {"hold": Counter(), "shift": Counter()}
    lengths = defaultdict(list)
    by_turn = {}
    offsets = []
    exclusions = []
    flagged_number_holds = 0
    from .timed_text_visibility import number

    detected = silero()
    progress = Progress("publisher-deadline-audit", len(grouped), "turns")
    for call, items in grouped.items():
        pcm = tracks[call]
        spans = detected(pcm)
        words = aligned_words(source[call])
        edges = [anchors[item["rank"]] for item in items]
        residual = word_clock_pairs(words, spans, edges)
        if residual.size:
            offsets.append(float(np.median(residual)))
        by_turn[call] = [0] * 4
        for row, edge in zip(items, edges, strict=True):
            name = "shift" if row["label"] else "hold"
            scores, duration = row_audit(row, edge, words, pcm, spans)
            totals[name].update(scores)
            if scores["straddle"] or scores["short_quiet_under_word_clock"]:
                exclusions.append({"rank": row["rank"], "straddle": bool(scores["straddle"]),
                                   "short_quiet": bool(scores["short_quiet_under_word_clock"])})
                flagged_number_holds += not row["label"] and number(row["text"].split()[-1])
            by_turn[call][2 * row["label"]] += 1
            by_turn[call][2 * row["label"] + 1] += scores["straddle"]
            if duration is not None:
                lengths[name].append(duration)
        progress.advance(1)
    progress.finish()
    if (totals["hold"]["rows"], totals["shift"]["rows"]) != (415, 397):
        raise ValueError("unexpected deadline audit membership")
    lower, median, upper = np.quantile(offsets, [.25, .5, .75]) if offsets else (None, None, None)
    delta = totals["shift"]["straddle"] / 397 - totals["hold"]["straddle"] / 415
    counts = totals["hold"] + totals["shift"]
    audit_mask = root / "cache/publisher-timed-english-deadline-mask.json"
    _write_json(audit_mask, {"freeze_sha256": digest(freeze), "excluded": sorted(exclusions, key=lambda row: row["rank"])})
    report = {
        "scope": "unscored, frozen English publisher timing sensitivity preflight",
        "freeze_sha256": digest(freeze),
        "counts": totals,
        "excluded_ranks_sha256": digest(audit_mask), "excluded_rows": len(exclusions),
        "excluded_number_ending_holds": flagged_number_holds,
        "median_straddle_ms": {name: float(np.median(values)) for name, values in lengths.items()},
        "internal_clock_turns_with_anchors": len(offsets), "total_turns": len(grouped),
        "word_end_minus_internal_vad_stop_median_ms": median,
        "word_end_minus_internal_vad_stop_iqr_ms": upper - lower if offsets else None,
        "shift_minus_hold_straddle_rate": delta, "paired_turn_ci95": asymmetry(by_turn),
        "class_asymmetry_abs_gt_10_points": abs(delta) > .10,
        "checks": {
            "frozen_stop_text_reproduced": not counts["changed_at_0_ms"],
            "no_observed_resume_words": not any(counts[f"resumed_words_at_{ms}_ms"] for ms in DEADLINES_MS),
            "all_hold_resumes_observed": not totals["hold"]["hold_resume_unverified"],
            "internal_clock_within_bounds": bool(offsets and abs(median) <= 40 and upper - lower <= 80),
            "internal_clock_dispersion_within_80_ms": bool(offsets and upper - lower <= 80),
            "audible_tail_majority": counts["audible_tail"] > counts["straddle"] / 2,
        },
    }
    _write_json(root / "reports/publisher-timed-english-deadline-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
