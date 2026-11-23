"""Seal and score an independent HarperValley development listening pilot."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE
from .shards import seed_for
from .sources import data_root

REVIEW_ROOT = Path("/dev/shm/superturn-harper-review")
DURATION = 20
SAMPLES = DURATION * RATE
AUDIT_TARGET = 11
UNIFORM_TARGET = 6
OUTCOMES = {"resume": "hold", "handoff": "shift"}
NEXT_CHANNEL = {"hold": "left", "shift": "right"}


@dataclass
class Window:
    call: str
    caller: int
    start: int
    rank: int
    stereo: np.ndarray
    proposals: list
    anchor: int | None
    cohort: str


class ListeningPacket:
    def __init__(self, destination, membership):
        if destination.exists():
            raise ValueError("the Harper listening packet already exists")
        self.destination = destination
        self.membership = membership
        self.targets = {"hold": [], "shift": []}
        self.uniform = []
        self.unavailable = Counter()

    def _window(self, call, caller, start, rank, tracks, proposals, anchor, cohort):
        stop = start + SAMPLES
        kept = []
        for edge, onset, label, focal, next_voice in proposals:
            if focal != "caller" or edge < start + 3 * RATE or onset + 2 * RATE > stop:
                continue
            kept.append({"stop_ms": (edge - start) / 16, "onset_ms": (onset - start) / 16,
                         "outcome": label, "next_channel": "left" if next_voice == "caller" else "right"})
        stereo = np.column_stack((tracks["caller"][start:stop], tracks["agent"][start:stop])).astype("<i2")
        return Window(call, caller, start, rank, stereo, kept, anchor, cohort)

    def consider(self, call, caller, tracks, proposals, caller_offset, archived):
        length = min(map(len, tracks.values()))
        limit = length - SAMPLES
        if limit >= 0:
            rank = seed_for("harper-uniform-call-v1", self.membership, call)
            if len(self.uniform) < 40 or rank < self.uniform[-1].rank:
                start = seed_for("harper-uniform-offset-v1", self.membership, call) % (limit + 1)
                self.uniform.append(self._window(call, caller, start, rank, tracks, proposals, None, "uniform"))
                self.uniform.sort(key=lambda row: row.rank)
                del self.uniform[40:]
        else:
            self.unavailable["uniform_calls_shorter_than_window"] += 1

        for edge, onset, label, focal, _ in proposals:
            if (focal != "caller" or onset - edge > RATE * 8 // 10
                    or (edge + caller_offset, label) not in archived):
                continue
            low, high = max(0, onset + 2 * RATE - SAMPLES), min(limit, edge - 3 * RATE)
            if low > high:
                self.unavailable[f"{label}_target_cannot_fit"] += 1
                continue
            rank = seed_for("harper-challenge-v1", self.membership, call, edge)
            start = low + rank % (high - low + 1)
            self.targets[label].append(self._window(call, caller, start, rank, tracks, proposals,
                                                    edge - start, f"challenge_{label}"))

    def _select(self):
        shifts = sorted(self.targets["shift"], key=lambda row: row.rank)
        if len(shifts) > AUDIT_TARGET:
            raise ValueError("unexpected growth in the pinned short-shift population")
        used_calls = {row.call for row in shifts}
        used_callers = {row.caller for row in shifts}
        selected = list(shifts)
        holds = sorted(self.targets["hold"], key=lambda row: row.rank)
        for _ in shifts:
            eligible = [row for row in holds if row.call not in used_calls]
            if not eligible:
                break
            row = min(eligible, key=lambda candidate: (candidate.caller in used_callers, candidate.rank))
            selected.append(row)
            used_calls.add(row.call)
            used_callers.add(row.caller)
        selected.extend([row for row in self.uniform if row.call not in used_calls][:UNIFORM_TARGET])
        return sorted(selected, key=lambda row: seed_for("harper-blind-order-v1", row.rank))

    def write(self, root):
        picked = self._select()
        if not picked:
            raise ValueError("Harper replay yielded no eligible excerpts")
        ids = [f"call-{row.rank:016x}" for row in picked]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate listening window identifiers")
        with TemporaryDirectory(dir=self.destination.parent, prefix="harper-review-") as temporary:
            packet = Path(temporary) / "packet"
            listening = packet / "harper-review"
            audio = listening / "audio"
            audio.mkdir(parents=True, mode=0o700)
            _write_json(listening / "packet.json", {"windows": ids, "duration": DURATION})
            private = {}
            for ident, row in zip(ids, picked, strict=True):
                path = audio / f"{ident}.wav"
                with wave.open(str(path), "wb") as handle:
                    handle.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
                    handle.writeframes(row.stereo.tobytes())
                private[ident] = {
                    "call": row.call, "start_sample": row.start, "cohort": row.cohort,
                    "anchor_ms": None if row.anchor is None else row.anchor / 16,
                    "proposals": row.proposals, "audio_sha256": sha256(path.read_bytes()).hexdigest(),
                }
            reference = packet / "private-reference.json"
            _write_json(reference, {"membership_sha256": self.membership, "windows": private})
            digest = sha256(reference.read_bytes()).hexdigest()
            packet.chmod(0o700)
            packet.rename(self.destination)
        report = {
            "scope": "blind English development listening pilot",
            "membership_sha256": self.membership, "private_reference_sha256": digest,
            "duration_seconds": DURATION, "windows": len(ids),
            "windows_by_cohort": dict(Counter(row.cohort for row in picked)),
            "unique_calls": len({row.call for row in picked}),
            "unique_callers": len({row.caller for row in picked}),
            "unavailable": dict(self.unavailable), "shift_population_expected": AUDIT_TARGET,
        }
        _write_json(root / "reports/harper-text-listening-packet.json", report)
        print(json.dumps(report, indent=2), flush=True)


def matches(proposals, marks, tolerance_ms):
    """Maximize matching count first, then minimize stop error."""
    ordered_proposals = sorted(enumerate(proposals), key=lambda pair: float(pair[1]["stop_ms"]))
    ordered_marks = sorted(enumerate(marks), key=lambda pair: float(pair[1]["stop_ms"]))
    empty = (0, 0., ())
    prior = [empty] * (len(ordered_marks) + 1)
    for i, proposal in ordered_proposals:
        current = [empty]
        for column, (j, mark) in enumerate(ordered_marks, 1):
            choices = [current[-1], prior[column]]
            error = abs(proposal["stop_ms"] - float(mark["stop_ms"]))
            if error <= tolerance_ms:
                count, negative_error, pairs = prior[column - 1]
                choices.append((count + 1, negative_error - error, (*pairs, (i, j))))
            current.append(max(choices, key=lambda result: result[:2]))
        prior = current
    return list(prior[-1][2])


def mark_label(mark):
    label = OUTCOMES.get(mark["outcome"])
    if label is None:
        return None, mark["outcome"]
    if mark.get("brief_ack") != "no":
        reason = "brief_acknowledgment" if mark.get("brief_ack") == "yes" else "acknowledgment_unknown"
        return None, reason
    channel = mark.get("next_channel")
    if channel != NEXT_CHANNEL[label]:
        return None, "inconsistent_next_channel" if channel else "next_channel_missing"
    try:
        gap = float(mark.get("next_onset_ms", "")) - float(mark["stop_ms"])
    except (TypeError, ValueError):
        return None, "next_onset_missing"
    if not np.isfinite(gap) or gap < 300:
        return None, "quiet_unconfirmed"
    return label, None


def score_window(record, marks, tolerance_ms):
    anchor = record["anchor_ms"]
    proposals = record["proposals"]
    if anchor is not None:
        proposals = [row for row in proposals if row["stop_ms"] == anchor]
        if len(proposals) != 1:
            raise ValueError("the hidden challenge interval has no unique target")
    counts = Counter(proposals=len(proposals), human_marks=len(marks))
    counts.update(f"{proposal['outcome']}_proposals" for proposal in proposals)
    paired = matches(proposals, marks, tolerance_ms)
    counts["unmatched_proposals"] = len(proposals) - len(paired)
    if anchor is None:
        counts["unmatched_human_marks"] = len(marks) - len(paired)
        matched_marks = {j for _, j in paired}
        for j, mark in enumerate(marks):
            label, _ = mark_label(mark)
            if label:
                counts[f"{label}_eligible_human_marks"] += 1
                if j not in matched_marks:
                    counts["unmatched_eligible_human_marks"] += 1
                    counts[f"{label}_unmatched_eligible_human_marks"] += 1
    for i, j in paired:
        proposal, mark = proposals[i], marks[j]
        proposed = proposal["outcome"]
        counts[f"{proposed}_matched_stops"] += 1
        label, reason = mark_label(mark)
        if label is None:
            counts[f"human_{reason}"] += 1
            counts[f"{proposed}_unusable_matched_outcomes"] += 1
            continue
        agreement = "outcome_agreement" if label == proposed else "outcome_disagreement"
        counts[agreement] += 1
        counts[f"{proposed}_{agreement}"] += 1
        if mark.get("next_channel") == proposal["next_channel"]:
            counts["next_channel_agreement"] += 1
        if mark.get("next_onset_ms") and abs(float(mark["next_onset_ms"]) - proposal["onset_ms"]) <= 80:
            counts["next_onset_within_80_ms"] += 1
    return counts


def confirmation_interval(totals, draws):
    drawn = totals[draws].sum(axis=1)
    supported = drawn[:, 1] > 0
    if np.count_nonzero(totals[:, 1]) < 2 or not supported.any():
        return None
    return np.quantile(drawn[supported, 0] / drawn[supported, 1], [.025, .975]).tolist()


def summarize_calls(calls):
    counts = sum(calls, Counter())
    draws = np.random.default_rng(29).integers(0, len(calls), (5000, len(calls)))
    totals = np.array([[call["outcome_agreement"], call["proposals"]] for call in calls])
    classes = {}
    for label in NEXT_CHANNEL:
        proposals = counts[f"{label}_proposals"]
        matched = counts[f"{label}_matched_stops"]
        agrees = counts[f"{label}_outcome_agreement"]
        disagrees = counts[f"{label}_outcome_disagreement"]
        eligible = agrees + disagrees
        paired_totals = np.array([[call[f"{label}_outcome_agreement"], call[f"{label}_proposals"]]
                                  for call in calls])
        classes[label] = {
            "proposals": proposals,
            "matched_stops": matched,
            "usable_matched_outcomes": eligible,
            "unusable_matched_outcomes": counts[f"{label}_unusable_matched_outcomes"],
            "agreed_outcomes": agrees,
            "disagreed_outcomes": disagrees,
            "matched_stop_coverage": matched / proposals if proposals else None,
            "matched_outcome_coverage": eligible / proposals if proposals else None,
            "agreement_given_usable_match": agrees / eligible if eligible else None,
            "confirmed_share_of_proposals": agrees / proposals if proposals else None,
            "confirmation_call_bootstrap_95": confirmation_interval(paired_totals, draws),
        }
    return {"events": counts, "calls": len(calls),
            "confirmed_share_of_proposals": counts["outcome_agreement"] / counts["proposals"]
            if counts["proposals"] else None, "call_bootstrap_95": confirmation_interval(totals, draws),
            "by_proposed_class": classes}


def score(root, ballot):
    reference = REVIEW_ROOT / "private-reference.json"
    receipt = json.loads((root / "reports/harper-text-listening-packet.json").read_text())
    if sha256(reference.read_bytes()).hexdigest() != receipt["private_reference_sha256"]:
        raise ValueError("the Harper listening reference changed")
    windows = json.loads(reference.read_text())["windows"]
    for ident, record in windows.items():
        audio = REVIEW_ROOT / "harper-review" / "audio" / f"{ident}.wav"
        if sha256(audio.read_bytes()).hexdigest() != record["audio_sha256"]:
            raise ValueError("a Harper listening waveform changed")
    ballot_bytes = ballot.read_bytes()
    saved = json.loads(ballot_bytes)["draft"]["windows"]
    if set(saved) != set(windows) or any(not row.get("reviewed") for row in saved.values()):
        raise ValueError("complete every listening window before scoring")
    reports = {}
    for tolerance in (250, 80):
        by_cohort = {}
        for ident, record in windows.items():
            source = saved[ident].get("events", [])
            marks = [mark for mark in source if mark["focal_channel"] == "left"
                     and 3000 <= float(mark["stop_ms"]) < 1000 * (DURATION - 2)
                     and (not mark.get("next_onset_ms") or float(mark["next_onset_ms"]) <= 1000 * (DURATION - 2))]
            cohort = record["cohort"]
            counts = by_cohort.setdefault(cohort, {}).setdefault(record["call"], Counter())
            counts.update(score_window(record, marks, tolerance))
        report = {cohort: summarize_calls(list(by_call.values())) for cohort, by_call in by_cohort.items()}
        challenge_calls = {}
        for cohort in ("challenge_hold", "challenge_shift"):
            for call, counts in by_cohort.get(cohort, {}).items():
                challenge_calls.setdefault(call, Counter()).update(counts)
        if challenge_calls:
            report["short_gap_challenge"] = summarize_calls(list(challenge_calls.values()))
        reports[str(tolerance)] = report
    result = {"scope": "directional factual listening pilot", "stop_tolerance_ms": reports,
              "primary_stop_tolerance_ms": 250, "sensitivity_stop_tolerance_ms": 80,
              "ballot_sha256": sha256(ballot_bytes).hexdigest(),
              "private_reference_sha256": receipt["private_reference_sha256"]}
    _write_json(root / "reports/harper-text-listening-score.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ballot", type=Path, help="complete review draft from the local-loopback reviewer")
    score(data_root(), parser.parse_args().ballot)
