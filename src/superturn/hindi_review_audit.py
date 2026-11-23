"""Freeze the source-anchored first Hindi ballot for independent review."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import file_digest
from .paired_call_judgments import CHANNELS, CLEAR, OUTCOMES, validate_judgment
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


def run(root, snapshot, expected_manifest):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    run(data_root(), args.snapshot, args.manifest_sha256)
