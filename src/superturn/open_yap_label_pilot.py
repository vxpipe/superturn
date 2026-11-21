"""Measure corroborated behavioral pause-label yield on the staged Open Yap calls."""

from collections import Counter
import json

import numpy as np

from .ami_calibration import _write_json
from .ami_pilot import _peak, group_words
from .audio import RATE, read_audio, silero
from .dense_teacher import file_digest
from .dral_label_pilot import isolation, summarize
from .harper_label_yield import one_to_one
from .labels import Speech
from .open_yap_preflight import SOURCE
from .progress import Progress
from .sources import data_root

SIDES = ("a", "b")
GAP = round(.3 * RATE)
TOLERANCE = round(.16 * RATE)


def timed_words(record, length):
    spans, uncertain, counts = [], [], Counter()
    for slot, name in enumerate(SIDES):
        for word in record["transcripts"][name]["words"]:
            start, end = word["start"], word["end"]
            if start is None or end is None:
                counts["untimed"] += 1
                continue
            start, end = round(start * RATE), round(end * RATE)
            if not 0 <= start < end <= length:
                counts["invalid_timing"] += 1
                continue
            span = Speech(str(slot), start, end)
            if word["type"] not in ("word", "filler"):
                uncertain.append(span)
            spans.append(span)
            counts[word["type"]] += 1
    return spans, uncertain, counts


def clock_check(audio, word_bouts, length):
    checks = {}
    for side in ("0", "1"):
        vad = [span.end for span in audio if span.speaker == side]
        lexical = [span.end for span in word_bouts if span.speaker == side]
        first, last = [], []
        for edge in vad:
            candidates = [other for other in lexical if abs(other - edge) <= TOLERANCE]
            if len(candidates) != 1:
                continue
            other, = candidates
            if sum(abs(other - match) <= TOLERANCE for match in vad) != 1:
                continue
            if edge < length / 4:
                first.append((other - edge) * 1000 / RATE)
            elif edge >= .75 * length:
                last.append((other - edge) * 1000 / RATE)
        early, late = [float(np.median(items)) if len(items) >= 5 else None for items in (first, last)]
        drift = late - early if early is not None and late is not None else None
        checks[side] = {"early_pairs": len(first), "late_pairs": len(last), "early_median_ms": early,
                        "late_median_ms": late, "median_change_ms": drift,
                        "passed": drift is not None and abs(drift) <= 40 and max(abs(early), abs(late)) <= 80}
    return checks


def inspect(tracks, record, detect):
    length = tracks.shape[1]
    activity = [list(detect(pcm)) for pcm in tracks]
    condition = isolation(tracks, activity)
    vad = [Speech(str(slot), start, end) for slot, rows in enumerate(activity) for start, end in rows]
    words, uncertain, word_counts = timed_words(record, length)
    clock = clock_check(group_words(vad), group_words(words), length)
    audio, audio_counts = summarize(vad, length)
    lexical, lexical_counts = summarize(words, length)
    conflicts = Counter()
    offsets = []
    supported = one_to_one(audio, lexical, conflicts, endpoint_deltas=offsets, anchor_first=True)
    retained, excluded = [], Counter()
    for slot, edge, next_start, label in supported:
        if any(span.start < edge + GAP and span.end > edge for span in words):
            excluded["word_during_vad_pause"] += 1
            continue
        if any(span.start < next_start and span.end > edge for span in uncertain):
            excluded["nonlexical_activity"] += 1
            continue
        following = int(slot) if label == "hold" else 1 - int(slot)
        levels = [_peak(tracks[int(slot)], edge - RATE // 5, edge),
                  _peak(tracks[following], next_start, next_start + RATE // 5)]
        quiet = max(_peak(track, edge + RATE // 25, edge + GAP) for track in tracks)
        if min(levels) < 128 or quiet > .5 * min(levels):
            excluded["quiet_or_level"] += 1
            continue
        retained.append((slot, edge, next_start, label))
    counts = Counter(row[-1] for row in retained)
    timing = {"endpoint_median_ms": float(np.median(offsets)) if offsets else None,
              "endpoint_iqr_ms": float(np.diff(np.percentile(offsets, [25, 75]))[0]) if offsets else None}
    pass_timing = all(row["passed"] for row in clock.values()) and offsets and (
        abs(timing["endpoint_median_ms"]) <= 80 and timing["endpoint_iqr_ms"] <= 160)
    accepted = bool(pass_timing and condition["passed"] and not (word_counts["untimed"] or word_counts["invalid_timing"]))
    result = {"channel_isolation": condition, "word_clock_proxy": clock, "endpoint_timing": timing,
              "acoustic_candidates": audio_counts, "word_candidates": lexical_counts, "word_counts": word_counts,
              "agreement": conflicts, "excluded": excluded, "provisional_supported": counts,
              "supported_by_focal_track": {slot: Counter(label for side, _, _, label in retained if side == slot)
                                            for slot in ("0", "1")}, "accepted": accepted}
    return result, retained if accepted else []


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Open Yap recordings stay on Vast")
    source = root / "raw" / SOURCE
    manifest_path = source / "MANIFEST.json"
    upstream = json.loads((root / "reports/open-yap-audio.json").read_text())
    if file_digest(manifest_path) != upstream["member_manifest_sha256"]:
        raise ValueError("Open Yap selection manifest changed")
    manifest = json.loads(manifest_path.read_text())
    detect = silero()
    reports, cohorts = [], {}
    meter = Progress("open-yap-label-pilot", len(manifest["calls"]), "calls")
    try:
        for key in manifest["calls"]:
            files = [source / f"{key}.{suffix}" for suffix in ("a.flac", "b.flac", "json")]
            for path in files:
                if file_digest(path) != manifest["members"][path.name]["sha256"]:
                    raise ValueError("Open Yap staged payload changed")
            tracks = np.stack([read_audio(path) for path in files[:2]])
            report, cohort = inspect(tracks, json.loads(files[2].read_text()), detect)
            reports.append(report)
            cohorts[key] = cohort
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    totals = Counter(row[-1] for rows in cohorts.values() for row in rows)
    viable = all(totals[label] >= 20 and all(sum(row[-1] == label for row in rows) >= 5 for rows in cohorts.values())
                 for label in ("hold", "shift"))
    report = {"scope": "behavioral next-speaker label yield; query anchored to the Silero edge",
              "deadline_ms": 300, "manifest_sha256": file_digest(manifest_path), "calls": reports,
              "supported": totals, "feasible": viable}
    if viable:
        output = root / "runs/open-yap-preflight/behavioral-pauses.json"
        _write_json(output, cohorts)
        report["labels_sha256"] = file_digest(output)
    _write_json(root / "reports/open-yap-label-pilot.json", report)
    return report


if __name__ == "__main__":
    report = run(data_root())
    print(json.dumps({"supported": report["supported"], "feasible": report["feasible"],
                      "calls": [{key: value for key, value in row.items() if key != "word_counts"}
                                for row in report["calls"]]}, indent=2))
