"""Prepare and score a factual listening audit of pinned HarperValley development pauses."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import wave

from .ami_calibration import _write_json
from .audio import RATE, silero
from .harper_text_review import (
    QUALIFICATION_REPORT, QUALIFICATION_ROOT, ListeningPacket, Window,
    gap_band, mark_label, matches, score_window, summarize_calls, verified_packet,
)
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .turn_text_train import digest

QUOTAS = {("hold", "300_to_800_ms"): 30, ("hold", "over_800_ms"): 30,
          ("shift", "300_to_800_ms"): 14, ("shift", "over_800_ms"): 46}
DURATION = 30
SAMPLES = DURATION * RATE
POOL_COUNTS = {"hold": 179, "shift": 179}


@dataclass(frozen=True)
class Candidate:
    window: Window
    label: str
    band: str
    straddle: bool
    native_stop: int
    prior: str
    length: str


def select(candidates):
    selected, calls, callers = [], Counter(), Counter()
    for label, band in sorted(QUOTAS, key=lambda key: key != ("shift", "300_to_800_ms")):
        quota = QUOTAS[label, band]
        pool = [item for item in candidates if (item.label, item.band) == (label, band)]
        if len(pool) < quota or (label == "shift" and band == "300_to_800_ms" and len(pool) != quota):
            raise ValueError(f"Harper audit quota changed for {label}/{band}: {len(pool)} eligible for {quota}")
        flags = min(sum(item.straddle for item in pool), 5)
        for index in range(quota):
            available = [item for item in pool if calls[item.window.call] < 3]
            straddles = [item for item in available if item.straddle]
            eligible = straddles if index < flags and straddles else available
            if not eligible:
                raise ValueError(f"cannot meet {label}/{band} quota within the call cap")
            item = min(eligible, key=lambda row: (calls[row.window.call], callers[row.window.caller], row.window.rank))
            pool = [remaining for remaining in pool if remaining is not item]
            selected.append(item)
            calls[item.window.call] += 1
            callers[item.window.caller] += 1
    return sorted(selected, key=lambda row: seed_for("harper-qualification-order-v1", row.window.rank))


def raw_clock_audit(root):
    from .harper_valley_audit import transcript_words

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Harper word timings stay on Vast")
    source = root / "cache/harper-valley/text-dev-v1.jsonl"
    metadata = json.loads((root / "reports/harper-text-dev-packing.json").read_text())
    if digest(source) != metadata["serialized_sha256"]:
        raise ValueError("Harper development input hash changed")
    rows = defaultdict(list)
    for candidate in map(json.loads, source.read_text().splitlines()):
        rows[candidate["call"]].append(candidate)
    raw = root / "raw/harper-valley/data"
    results = {"hold": Counter(), "shift": Counter()}
    for call, candidates in rows.items():
        segments = [row for row in json.loads((raw / "transcript" / f"{call}.json").read_text())
                    if row["speaker_role"] == "caller"]
        with wave.open(str(raw / "audio/caller" / f"{call}.wav"), "rb") as audio:
            if (audio.getframerate(), audio.getsampwidth(), audio.getnchannels()) != (8000, 2, 1):
                raise ValueError("unexpected Harper caller track format")
            samples = audio.getnframes() * (RATE // audio.getframerate())
        omissions = []
        words = list(transcript_words(segments, "caller", samples, 0, omissions=omissions))
        for row in candidates:
            counts = results[row["label"]]
            edge = row["caller_sample"]
            crossing = [word for word in words if word.start < edge < word.end]
            counts["rows"] += 1
            counts["rows_with_crossing_word"] += bool(crossing)
            counts["crossing_words"] += len(crossing)
            counts["rows_with_crossing_ending_within_300_ms"] += any(word.end <= edge + .3 * RATE for word in crossing)
            counts["rows_with_timing_omissions_at_edge"] += any(word.start <= edge <= word.end for word in omissions)
    if {label: result["rows"] for label, result in results.items()} != POOL_COUNTS:
        raise ValueError("raw word audit did not cover the frozen class population")
    report = {"scope": "caller-local raw word times without label-support trimming",
              "causal_pack_sha256": digest(source), "counts_by_class": results}
    _write_json(root / "reports/harper-text-qualification-raw-word-clock.json", report)
    print(json.dumps(report, indent=2))


def pre_eligibility_audit(root):
    from .harper_valley_audit import audit_call
    from .harper_valley_benchmark import pinned_split
    from .labels import Speech, turns

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Harper eligibility audit stays on Vast")
    cases, _, roster = pinned_split(root, "train")
    stats, crossings = Counter(), Counter()
    detect = silero()
    progress = Progress("harper-pre-eligibility", len(cases), "calls")
    for case in cases:
        def observe(_audio, _origins, vad, words, omissions, start, end):
            clipped = [Speech(span.speaker, max(start, span.start), min(end, span.end))
                       for span in vad if max(start, span.start) < min(end, span.end)]
            for focal, _, label in turns(clipped, decision_ms=300):
                if focal.speaker != "caller":
                    continue
                crossings[f"{label}/pre_eligibility"] += 1
                crossings[f"{label}/raw_word_crossing"] += any(
                    word.speaker == "caller" and word.start < focal.end < word.end for word in words)
                crossings[f"{label}/timing_omission_at_stop"] += any(
                    omission.speaker == "caller" and omission.start <= focal.end <= omission.end
                    for omission in omissions)

        stats.update(audit_call(root / "raw/harper-valley", case["id"], detect, recover_clock=True,
                                observe=observe))
        progress.advance(1)
    progress.finish()
    per_class = {}
    for label in POOL_COUNTS:
        count = stats[f"candidate_{label}"]
        if not count or count != crossings[f"{label}/pre_eligibility"]:
            raise ValueError("raw word and archive candidate universes disagree")
        eligible = stats[f"fixed_window_supported_{label}"]
        per_class[label] = {
            "pre_eligibility": count, "raw_word_crossing": crossings[f"{label}/raw_word_crossing"],
            "timing_omission_at_stop": crossings[f"{label}/timing_omission_at_stop"],
            "fixed_window_supported": eligible, "dropped": count - eligible,
            "drop_rate": (count - eligible) / count,
            "drop_reasons": {kind: stats[f"{kind}_{label}"]
                             for kind in ("incomplete_timing", "disagreed", "ambiguous", "unmatched")}
                            | {"agreed_outside_fixed_window": stats[f"supported_{label}"] - eligible},
        }
    delta = per_class["shift"]["drop_rate"] - per_class["hold"]["drop_rate"]
    report = {"scope": "staged development calls with valid clock recovery before word-support filtering",
              "roster_audit_sha256": roster, "calls_total": len(cases), "calls_observed": stats["audited_calls"],
              "by_class": per_class, "shift_minus_hold_drop_rate": delta,
              "absolute_drop_difference_over_ten_points": abs(delta) > .10}
    _write_json(root / "reports/harper-text-pre-eligibility-word-clock.json", report)
    print(json.dumps(report, indent=2))


def build(root, code_revision):
    from .basis_stage1 import events
    from .harper_valley_audit import audit_call
    from .harper_valley_benchmark import pinned_split
    from .timed_text_visibility import QUESTION
    from .turn_text_number_layout import length_bin

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Harper listening audio stays on Vast")
    pack = root / "cache/harper-valley/text-dev-v1.jsonl"
    metadata = json.loads((root / "reports/harper-text-dev-packing.json").read_text())
    if digest(pack) != metadata["serialized_sha256"]:
        raise ValueError("Harper development rows changed")
    membership = metadata["archived_membership_sha256"]
    packet = ListeningPacket(QUALIFICATION_ROOT, membership, duration=DURATION)
    cases, _, roster_sha = pinned_split(root, "train")
    roles = {case["id"]: case for case in cases}
    calls = defaultdict(dict)
    population = Counter()
    for row in map(json.loads, pack.read_text().splitlines()):
        key = row["caller_sample"], row["label"]
        if key in calls[row["call"]]:
            raise ValueError("duplicate Harper development stop")
        calls[row["call"]][key] = row
        population[f"{row['label']}/{gap_band(row['gap_ms'])}"] += 1
    if (sum(map(len, calls.values())) != metadata["rows"] or not calls.keys() <= roles.keys()
            or Counter(row["label"] for rows in calls.values() for row in rows.values()) != POOL_COUNTS):
        raise ValueError("Harper development row counts changed")
    selected_candidates = []
    continuous = []
    rejected = defaultdict(list)
    exclusion_counts = Counter()
    detect = silero()
    progress = Progress("harper-qualification-listening", len(roles), "calls")
    for call, case in sorted(roles.items()):
        packed = calls.get(call, {})
        collected = {}
        reproduced = []

        def observe(audio, origins, vad, words, _omissions, start, end):
            tracks = {role: pcm[start - origins[role]:end - origins[role]] for role, pcm in audio.items()}
            spans = [(max(start, span.start) - start, min(end, span.end) - start, span.speaker)
                     for span in vad if span.start < end and span.end > start]
            rejected_events = []
            reasons, eligible = events(spans, {role: [(0, end - start)] for role in tracks},
                                       [], end - start, tracks=tracks, rejected=rejected_events)
            if reasons["ambiguous_overlap_boundary"]:
                exclusion_counts["unknown/ambiguous_overlap_boundary"] += reasons["ambiguous_overlap_boundary"]

            limit = end - start - SAMPLES
            if limit >= 0:
                rank = seed_for("harper-qualification-continuous-call-v1", membership, call)
                left = seed_for("harper-qualification-continuous-offset-v1", membership, call) % (limit + 1)
                packed_events = [proposal for proposal in eligible if proposal[3] == "caller"
                                 and (proposal[0] + start - origins["caller"], proposal[2]) in packed]
                continuous.append(packet._window(call, case["caller"], left, rank, tracks,
                                                 packed_events, None, "continuous"))
            for stop, onset, label, focal, next_voice, reason in rejected_events:
                if focal != "caller":
                    continue
                exclusion_counts[f"{label}/{reason}"] += 1
                low, high = max(0, onset + 2 * RATE - SAMPLES), min(limit, stop - 3 * RATE)
                if low <= high:
                    rank = seed_for("harper-qualification-excluded-v1", membership, call, stop, reason)
                    left = low + rank % (high - low + 1)
                    rejected[reason].append(packet._window(call, case["caller"], left, rank, tracks,
                                                           [(stop, onset, label, focal, next_voice)], stop - left,
                                                           f"excluded/{reason}"))
            for stop, onset, label, focal, next_voice in eligible:
                key = stop + start - origins["caller"], label
                row = packed.get(key)
                if focal != "caller" or row is None:
                    continue
                if round(row["gap_ms"] * RATE / 1000) != onset - stop:
                    raise ValueError("Harper waveform pause gap changed")
                bounds = max(0, onset + 2 * RATE - SAMPLES), min(end - start - SAMPLES, stop - 3 * RATE)
                if bounds[0] > bounds[1]:
                    packet.unavailable[f"{label}_target_cannot_fit"] += 1
                    continue
                rank = seed_for("harper-qualification-v1", membership, call, key[0], label)
                left = bounds[0] + rank % (bounds[1] - bounds[0] + 1)
                edge = stop + start
                straddle = any(word.speaker == "caller" and word.start < edge < word.end <= edge + .3 * RATE
                               for word in words)
                band = gap_band(row["gap_ms"])
                cohort = f"{label}/{band}/{'straddle' if straddle else 'other'}"
                window = packet._window(call, row["caller"], left, rank, tracks,
                                        [(stop, onset, label, focal, next_voice)], stop - left, cohort)
                prior, text = row["text"].split("\n")
                prior_type = "absent" if prior == "---" else "question" if prior.split()[0] in QUESTION else "statement"
                collected[key] = Candidate(window, label, band, straddle, key[0], prior_type,
                                           length_bin(len(text.split())))

        audit_call(root / "raw/harper-valley", call, detect, recover_clock=True, pauses=reproduced, observe=observe)
        confirmed = set(reproduced)
        selected_candidates.extend(candidate for key, candidate in collected.items() if key in confirmed)
        for reason, keys in (("unreproduced", packed.keys() - confirmed),
                             ("without_excerpt", (packed.keys() & confirmed) - collected.keys())):
            for key in keys:
                row = packed[key]
                packet.unavailable[f"{reason}/{row['label']}/{gap_band(row['gap_ms'])}"] += 1
        progress.advance(1)
    progress.finish()
    chosen = select(selected_candidates)
    pool = Counter(f"{row.label}/{row.band}/{'straddle' if row.straddle else 'other'}"
                   for row in selected_candidates)
    windows = [item.window for item in chosen]
    quota_use = Counter(window.call for window in windows)
    used_rejected = Counter()
    for reason in sorted(rejected, key=lambda key: (-sum(exclusion_counts[f"{label}/{key}"] for label in POOL_COUNTS), key)):
        options = [row for row in sorted(rejected[reason], key=lambda row: row.rank) if quota_use[row.call] < 3]
        selected = []
        for row in options:
            if quota_use[row.call] < 3:
                selected.append(row)
                quota_use[row.call] += 1
            if len(selected) == 4:
                break
        if len(selected) < 4:
            for row in selected:
                quota_use[row.call] -= 1
            continue
        windows.extend(selected)
        used_rejected[reason] = len(selected)
        if len(used_rejected) == 5:
            break
    if len(used_rejected) != 5 or len(continuous) < 20:
        raise ValueError(f"Harper diagnostic shortfall: exclusions={dict(used_rejected)}, continuous={len(continuous)}")
    windows.extend(sorted(continuous, key=lambda row: row.rank)[:20])
    targets = {item.window.rank: {"native_stop": item.native_stop, "caller": item.window.caller,
                                 "agent": roles[item.window.call]["agent"], "label": item.label,
                                 "gap_band": item.band, "straddle": item.straddle, "prior": item.prior,
                                 "length_bin": item.length} for item in chosen}
    allocation = Counter()
    for target in targets.values():
        for dimension in ("gap_band", "straddle", "prior", "length_bin"):
            allocation[f"{target['label']}/{dimension}/{target[dimension]}"] += 1
    details = {
        "causal_pack_sha256": metadata["serialized_sha256"], "causal_pack_rows": metadata["rows"],
        "archived_role": "train", "roster_audit_sha256": roster_sha, "selection_seed": "harper-qualification-v1",
        "renderer_sha256": digest(Path(__file__)), "code_revision": code_revision,
        "input_source_scope": "HarperValley English development",
        "population_by_class_and_gap": dict(population), "renderable_by_class_gap_straddle": dict(pool),
        "exclusions_by_class_and_reason": dict(exclusion_counts), "sampled_exclusions_by_reason": dict(used_rejected),
        "renderable_exclusions_by_reason": {reason: len(rows) for reason, rows in rejected.items()},
        "accepted_sample_allocation": dict(allocation), "accepted_sample_callers": len({item.window.caller for item in chosen}),
        "accepted_sample_agents": len({target["agent"] for target in targets.values()}),
        "max_anchored_excerpts_per_call": max(quota_use.values()), "continuous_eligible_calls": len(continuous),
        "continuous_windows": 20, "required_agreements_per_class": 59,
    }
    windows.sort(key=lambda row: seed_for("harper-qualification-mixed-order-v2", row.rank))
    packet.write(root, picked=windows, report_name=QUALIFICATION_REPORT,
                 qualification_pool=details, qualification_targets=targets)


def score(root, ballot):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Harper ballots stay on Vast")
    packet, receipt = verified_packet(root, QUALIFICATION_ROOT)
    pool = receipt["qualification_candidate_pool"]
    path = root / "cache/harper-valley/text-dev-v1.jsonl"
    if digest(path) != pool["causal_pack_sha256"]:
        raise ValueError("the factual audit's causal source changed")
    packed = {(row["call"], row["caller_sample"], row["label"]): row["caller"]
              for row in map(json.loads, path.read_text().splitlines())}
    if len(packed) != sum(POOL_COUNTS.values()):
        raise ValueError("the factual audit's source population changed")
    saved = json.loads(ballot.read_text())["draft"]["windows"]
    if saved.keys() != packet["windows"].keys() or any(not row.get("reviewed") for row in saved.values()):
        raise ValueError("review every Harper qualification window before scoring")
    views = {}
    for tolerance in (250, 80):
        calls = defaultdict(Counter)
        strata = defaultdict(lambda: defaultdict(Counter))
        continuous = Counter()
        excluded = defaultdict(Counter)
        for ident, record in packet["windows"].items():
            marks = [mark for mark in saved[ident].get("events", []) if mark["focal_channel"] == "left"]
            cohort = record["cohort"]
            if cohort == "continuous":
                valid = [(mark, label) for mark in marks if (label := mark_label(mark)[0])
                         and 3000 <= float(mark["stop_ms"]) and float(mark["next_onset_ms"]) <= 28000]
                pairs = matches(record["proposals"], [mark for mark, _ in valid], tolerance)
                continuous["windows"] += 1
                continuous["detected"] += len(record["proposals"])
                continuous.update(f"{label}_eligible_rater_events" for _, label in valid)
                for index, match in pairs:
                    label = valid[match][1]
                    if record["proposals"][index]["outcome"] == label:
                        continuous[f"{label}_detected_rater_events"] += 1
                continue
            counts = score_window(record, marks, tolerance)
            if cohort.startswith("excluded/"):
                excluded[cohort].update(counts)
                continue
            target = record["qualification_target"]
            key = record["call"], target["native_stop"], target["label"]
            if packed.get(key) != target["caller"]:
                raise ValueError("a bound qualification stop differs from the frozen causal input")
            calls[record["call"]].update(counts)
            strata[cohort][record["call"]].update(counts)
        continuous["not_confirmed_detector_events"] = continuous["detected"] - sum(
            continuous[f"{label}_detected_rater_events"] for label in POOL_COUNTS)
        recall = {label: continuous[f"{label}_detected_rater_events"] / continuous[f"{label}_eligible_rater_events"]
                  if continuous[f"{label}_eligible_rater_events"] else None for label in POOL_COUNTS}
        balance = abs(recall["hold"] - recall["shift"]) if all(rate is not None for rate in recall.values()) else None
        gap_totals = defaultdict(Counter)
        for key, clusters in strata.items():
            gap_totals[key.rsplit("/", 1)[0]].update(sum(clusters.values(), Counter()))
        weighted = sum(pool["population_by_class_and_gap"][key] * totals["outcome_agreement"] / totals["proposals"]
                       for key, totals in gap_totals.items()) / sum(POOL_COUNTS.values())
        views[str(tolerance)] = {"overall": summarize_calls(list(calls.values())),
                                 "strata": {key: summarize_calls(list(value.values()))
                                            for key, value in strata.items()},
                                 "gap_reweighted_confirmation": weighted,
                                 "excluded_by_reason": excluded,
                                 "continuous": {"counts": continuous, "recall_by_class": recall,
                                                "miss_rate_absolute_gap": balance,
                                                "screen_passed": balance is not None and balance < .10
                                                and min(recall.values()) >= .90}}
    measures = views["250"]["overall"]["by_proposed_class"]
    if any(measures[label]["proposals"] != 60 for label in ("hold", "shift")):
        raise ValueError("expected exactly 60 proposals for each class")
    passed = all(measures[label]["agreed_outcomes"] >= 59 for label in ("hold", "shift"))
    wilson_lower = {}
    z = 1.959963984540054
    for label in ("hold", "shift"):
        rate = measures[label]["agreed_outcomes"] / 60
        wilson_lower[label] = (
            rate + z**2 / 120 - z * sqrt(rate * (1 - rate) / 60 + z**2 / 14400)
        ) / (1 + z**2 / 60)
    report = {"scope": "HarperValley factual label confirmation, development only", "primary_tolerance_ms": 250,
              "passed_selected_event_confirmation": passed, "required_agreements_per_class": 59,
              "passed_continuous_event_screen": views["250"]["continuous"]["screen_passed"],
              "confirmation_wilson_lower_95_by_class": wilson_lower,
              "wilson_scope": "descriptive on targeted, clustered excerpt sample",
              "causal_pack_sha256": pool["causal_pack_sha256"],
              "ballot_sha256": digest(ballot), "private_reference_sha256": receipt["private_reference_sha256"],
              "by_tolerance_ms": views}
    _write_json(root / "reports/harper-text-qualification-listening-score.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--score", type=Path)
    modes.add_argument("--raw-clock-only", action="store_true")
    modes.add_argument("--pre-eligibility", action="store_true")
    parser.add_argument("--code-revision", help="Git revision used to render the packet")
    args = parser.parse_args()
    if args.raw_clock_only:
        raw_clock_audit(data_root())
    elif args.pre_eligibility:
        pre_eligibility_audit(data_root())
    elif args.score:
        score(data_root(), args.score)
    elif args.code_revision and len(args.code_revision) == 40 and all(ch in "0123456789abcdef" for ch in args.code_revision):
        build(data_root(), args.code_revision)
    else:
        parser.error("provide --code-revision to render the frozen packet")
