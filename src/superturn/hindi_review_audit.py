"""Freeze the source-anchored first Hindi ballot for independent review."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import file_digest
from .paired_call_judgments import CHANNELS, CLEAR, OUTCOMES, continuous_ballot, validate_judgment, window_ballot
from .paired_call_review import CONTINUOUS_REPORTS, JOB, verify
from .sources import data_root

FILES = {"reviewer-a.json", "suggested-pauses.json", "packet.json"}


def load_snapshot(folder, expected_manifest):
    manifest = folder / "SHA256SUMS"
    if file_digest(manifest) != expected_manifest:
        raise ValueError("Hindi ballot snapshot manifest changed")
    entries = [line.split(maxsplit=1)[::-1] for line in manifest.read_text().splitlines()]
    digests = dict(entries)
    if len(entries) != len(FILES) or set(digests) != FILES or any(
        file_digest(folder / name) != digest for name, digest in digests.items()
    ):
        raise ValueError("Hindi ballot snapshot failed provenance verification")
    return {name: json.loads((folder / name).read_text()) for name in FILES}


def judgment(row):
    fields = {key: row.get(key, "") for key in ("outcome", "language", "brief_ack", "confidence")}
    validate_judgment(fields)
    return fields


def assisted_judgment(row, proposal, duration_ms):
    fields = judgment(row)
    stop = proposal["stop_ms"]
    correction = row.get("rater_stop_ms")
    corrected_stop = stop if correction in (None, "") else int(correction)
    onset = row.get("next_onset_ms", proposal["next_onset_ms"])
    channel = row.get("next_channel", proposal["next_channel"])
    invalid = abs(corrected_stop - stop) > 100 or (
        onset != "" and int(onset) - max(stop, corrected_stop) < 300
    )
    if invalid:
        fields["outcome"] = "invalid_pause"
    elif onset == "":
        if channel or fields["outcome"] in CLEAR:
            raise ValueError("missing assisted next onset")
    elif int(onset) >= duration_ms or channel not in CHANNELS:
        raise ValueError("assisted next onset or channel is invalid")
    first = row.get("first_outcome")
    if first is not None and first not in OUTCOMES:
        raise ValueError("invalid original assisted outcome")
    return {"judgment": fields, "first_outcome": first, "entered_outcome": row["outcome"],
            "timing": {"proposal_stop_ms": stop, "rater_stop_ms": correction,
                       "next_onset_ms": onset, "next_channel": channel,
                       "next_onset_source": "rater" if "next_onset_ms" in row else "proposal",
                       "next_channel_source": "rater" if "next_channel" in row else "proposal"}}


def reconcile(packet, ballot, suggestions, short_key, window_key):
    draft = ballot["draft"]
    windows = set(packet["windows"])
    clips = set(packet["clips"])
    if windows != set(window_key) or not clips <= short_key.keys() or not draft["clips"].keys() <= clips:
        raise ValueError("Hindi packet membership changed")
    if suggestions["schema"] != 1 or suggestions["mode"] != "assisted":
        raise ValueError("expected the sealed assisted proposal manifest")
    if set(suggestions["recordings"]) != windows or any(
        suggestions["recordings"][id]["audio_sha256"] != window_key[id]["clip_sha256"] for id in windows
    ):
        raise ValueError("assisted proposals refer to different audio")
    proposals = {row["id"]: row for row in suggestions["proposals"]}
    if len(proposals) != len(suggestions["proposals"]) or not draft["assisted"].keys() <= proposals.keys():
        raise ValueError("unknown or duplicate assisted proposal")

    events = defaultdict(dict)
    call_names = {item["recording"] for item in window_key.values()}
    for id in clips:
        source = short_key[id]
        if source["source"] != "spring" or source["nominal_language"] != "hi" or source["recording"] not in call_names:
            raise ValueError("short clip is outside the staged SPRING Hindi calls")
        if type(source["focus_channel"]) is not int or source["focus_channel"] not in CHANNELS.values():
            raise ValueError("unexpected short-clip channel encoding")
        row = draft["clips"].get(id, {})
        if row.get("outcome"):
            key = (source["recording"], source["focus_channel"], source["native_edge_sample"])
            if "short" in events[key]:
                raise ValueError("duplicate short view for the same source pause")
            events[key]["short"] = {"id": id, "judgment": judgment(row)}

    for id, row in draft["assisted"].items():
        if not row.get("outcome"):
            continue
        proposal = proposals[id]
        window = window_key[proposal["recording"]]
        stop = proposal["stop_ms"]
        duration = window["samples"] * 1000 // RATE
        if not 0 <= stop <= duration - 300:
            raise ValueError("assisted stop is outside the source window")
        edge = window["native_start_sample"] + stop * RATE // 1000
        key = (window["recording"], CHANNELS[proposal["focal_channel"]], edge)
        if "assisted" in events[key]:
            raise ValueError("duplicate assisted view for the same source pause")
        events[key]["assisted"] = {
            "id": id,
            **assisted_judgment(row, proposal, duration),
        }

    rows = []
    counts = Counter()
    short_only, assisted_only = [], []
    for (recording, channel, edge), views in sorted(events.items()):
        outcomes = {view["judgment"]["outcome"] for view in views.values()}
        languages = {view["judgment"]["language"] for view in views.values()}
        if len(views) == 2:
            counts["cross_view_duplicates"] += 1
            if len(outcomes) > 1:
                counts["cross_view_conflicts"] += 1
            if len(languages) > 1:
                counts["cross_view_language_conflicts"] += 1
        elif "short" in views:
            short_only.append((recording, channel, edge))
        else:
            assisted_only.append((recording, channel, edge))
        if "assisted" in views:
            assisted = views["assisted"]
            counts["rated_assisted"] += 1
            if assisted["first_outcome"] and assisted["first_outcome"] != assisted["entered_outcome"]:
                counts["revised_assisted_outcomes"] += 1
            if assisted["judgment"]["outcome"] in CLEAR and (
                assisted["timing"]["next_onset_source"] == "proposal"
                or assisted["timing"]["next_channel_source"] == "proposal"
            ):
                counts["clear_assisted_with_proposal_timing"] += 1
        if "short" in views:
            counts["rated_short"] += 1
        outcome = next(iter(outcomes)) if len(outcomes) == 1 else None
        language = next(iter(languages)) if len(languages) == 1 else None
        counts[outcome or "conflict"] += 1
        rows.append({"recording": recording, "focal_channel": channel, "stop_sample": edge,
                     "outcome": outcome, "language": language, "views": views})
    counts["unrated_short"] = len(clips) - counts["rated_short"]
    counts["unrated_assisted"] = len(proposals) - counts["rated_assisted"]
    counts["unmatched_cross_view_within_80_ms"] = sum(
        any((recording, channel) == (other, side) and abs(edge - sample) <= 80 * RATE // 1000
            for other, side, sample in assisted_only)
        for recording, channel, edge in short_only
    )
    return rows, counts


def compare_followup(first, second, window_key, completed):
    """Match manual marks to first-review anchors without treating them as manual."""
    tolerance = 80 * RATE // 1000
    coverage = {row["recording"]: (row["native_start_sample"],
                                  row["native_start_sample"] + row["samples"] - RATE * 3 // 10)
                for row in window_key.values()}
    if len(coverage) != len(window_key):
        raise ValueError("expected one blind window per source recording")
    marks = defaultdict(list)
    for (id, channel), spans in second.items():
        window = window_key[id]
        start = window["native_start_sample"]
        for stop, onset, next_channel, fields in spans:
            marks[window["recording"], channel].append({
                "window": id, "stop_sample": start + stop * RATE // 1000,
                "next_sample": None if onset is None else start + onset * RATE // 1000,
                "next_channel": next_channel, "judgment": fields, "window_language": completed[id][channel],
            })

    possible = []
    uses = Counter()
    counts = Counter(first_anchors=len(first), manual_marks=sum(map(len, marks.values())))
    for event in first:
        group = event["recording"], event["focal_channel"]
        span = coverage.get(group[0])
        if span is None or not span[0] <= event["stop_sample"] <= span[1]:
            counts["outside_manual_coverage"] += 1
            continue
        candidates = [index for index, mark in enumerate(marks[group])
                      if abs(mark["stop_sample"] - event["stop_sample"]) <= tolerance]
        possible.append((event, group, candidates))
        uses.update((group, index) for index in candidates)

    comparison = []
    calls_by_floor = defaultdict(set)
    for event, group, candidates in possible:
        if not candidates:
            counts["unmatched_first_anchors"] += 1
            continue
        if len(candidates) != 1 or uses[group, candidates[0]] != 1:
            counts["ambiguous_first_anchors"] += 1
            continue
        mark = marks[group][candidates[0]]
        counts["matched"] += 1
        counts[f"matched_manual_{mark['judgment']['outcome']}"] += 1
        first_outcome = event["outcome"]
        second_outcome = mark["judgment"]["outcome"]
        if first_outcome is None:
            floor = "first_view_conflict"
        elif first_outcome != second_outcome:
            floor = "clear_contradiction" if first_outcome in CLEAR and second_outcome in CLEAR else "other_disagreement"
        elif first_outcome not in CLEAR:
            floor = "nonbinary_agreement"
        elif mark["judgment"]["language"] != "hindi" or mark["window_language"] != "hindi":
            floor = "language_unconfirmed"
        else:
            floor = f"agreed_{first_outcome}"
        counts[floor] += 1
        calls_by_floor[floor].add(event["recording"])

        timing = event["views"].get("assisted", {}).get("timing")
        onset_delta = None
        if not timing or timing["next_onset_ms"] == "":
            time_result = "timing_unavailable_first"
        elif mark["next_sample"] is None:
            time_result = "timing_unavailable_manual"
        elif timing["next_channel"] != mark["next_channel"]:
            time_result = "next_channel_disagreement"
        else:
            offset = event["stop_sample"] - int(timing["proposal_stop_ms"]) * RATE // 1000
            first_next = offset + int(timing["next_onset_ms"]) * RATE // 1000
            onset_delta = mark["next_sample"] - first_next
            quiet = min(first_next, mark["next_sample"]) - max(event["stop_sample"], mark["stop_sample"])
            if abs(onset_delta) <= tolerance and quiet >= RATE * 3 // 10:
                timing_class = "proposal" if "proposal" in (
                    timing["next_onset_source"], timing["next_channel_source"]
                ) else "entered"
                time_result = f"{timing_class}_onset_and_channel_corroborated"
            else:
                time_result = "onset_or_quiet_disagreement"
        counts[time_result] += 1
        comparison.append({
            "recording": event["recording"], "focal_channel": event["focal_channel"],
            "proposal_stop_sample": event["stop_sample"], "manual_mark": mark,
            "stop_delta_samples": mark["stop_sample"] - event["stop_sample"],
            "onset_delta_samples": onset_delta, "floor_comparison": floor, "timing_comparison": time_result,
        })
    counts["manual_marks_without_first_anchor"] = counts["manual_marks"] - len(uses)
    counts["manual_marks_with_ambiguous_anchor"] = len(uses) - counts["matched"]
    counts.update({f"{key}_calls": len(calls) for key, calls in calls_by_floor.items()})
    return comparison, dict(sorted(counts.items()))


def read_followup(events_csv, windows_csv, key):
    digests = {path: file_digest(path) for path in (events_csv, windows_csv)}
    second = continuous_ballot(events_csv, key)
    completed = window_ballot(windows_csv, key)
    if set(completed) != set(key):
        raise ValueError("finish the completion form for all review windows before comparing")
    if any(file_digest(path) != digest for path, digest in digests.items()):
        raise ValueError("a manual ballot file changed while being checked")
    return second, completed, digests


def run(root, snapshot, expected_manifest, reviewer_b=None, reviewer_b_windows=None):
    names = (JOB, CONTINUOUS_REPORTS["continuous"])
    verify(root, names)
    saved = load_snapshot(snapshot, expected_manifest)
    restored_packet = root / "runs" / JOB / "hindi-review-a" / "packet.json"
    if file_digest(restored_packet) != file_digest(snapshot / "packet.json"):
        raise ValueError("snapshot uses a different Hindi review packet")
    keys = []
    for name in names:
        report = json.loads((root / "reports" / f"{name}.json").read_text())
        keys.append(json.loads((root / report["packet"]).parent.joinpath("private-key.json").read_text()))
    rows, counts = reconcile(saved["packet.json"], saved["reviewer-a.json"], saved["suggested-pauses.json"], *keys)
    summary = {
        "scope": "first reviewer only; source-call training feasibility",
        "snapshot_manifest_sha256": expected_manifest,
        "event_key": "SPRING recording, focal channel, proposal stop sample at 16 kHz",
        "language_scope": "first-ballot entries can contain browser defaults",
        "source_calls": len({row["recording"] for row in rows}),
        "distinct_pauses": len(rows),
        "counts": dict(sorted(counts.items())),
        "admitted_training_rows": 0,
    }
    payload = {**summary, "events": rows}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    output = snapshot.parent / "audit"
    output.mkdir(mode=0o700, exist_ok=True)
    path = output / f"source-events-{digest}.json"
    if not path.exists():
        _write_json(path, payload)
    elif json.loads(path.read_text()) != payload:
        raise ValueError("sealed source-event receipt changed")
    print(json.dumps({**summary, "event_receipt_sha256": file_digest(path)}, indent=2))
    if reviewer_b is not None:
        second, completed, digests = read_followup(reviewer_b, reviewer_b_windows, keys[1])
        digest = digests[reviewer_b]
        matches, stats = compare_followup(rows, second, keys[1], completed)
        report = {
            "scope": "submitted manual ballot against proposal anchors; reviewer independence requires confirmation",
            "first_receipt_sha256": file_digest(path),
            "manual_ballot_sha256": digest,
            "manual_windows_sha256": digests[reviewer_b_windows],
            "completed_windows": len(completed),
            "completed_windows_without_marks": len(set(completed) - {id for id, _ in second}),
            "manual_hindi_channels": sum(language == "hindi" for pair in completed.values() for language in pair),
            "counts": stats,
            "admitted_training_rows": 0,
        }
        receipt = {**report, "matched_events": matches}
        receipt_id = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        review_path = output / f"followup-{receipt_id}.json"
        if not review_path.exists():
            _write_json(review_path, receipt)
        elif json.loads(review_path.read_text()) != receipt:
            raise ValueError("sealed follow-up comparison changed")
        print(json.dumps({**report, "comparison_sha256": file_digest(review_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--reviewer-b", type=Path, help="optional completed blind continuous CSV")
    parser.add_argument("--reviewer-b-windows", type=Path, help="completion and language CSV for the same reviewer")
    args = parser.parse_args()
    if (args.reviewer_b is None) != (args.reviewer_b_windows is None):
        parser.error("provide both manual reviewer files together")
    run(data_root(), args.snapshot, args.manifest_sha256, args.reviewer_b, args.reviewer_b_windows)
