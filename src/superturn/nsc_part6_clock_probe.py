"""Test whether two local phone tracks support a trustworthy common timeline."""

from collections import Counter
import hashlib
from itertools import combinations
import json
import math

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, flac_streaminfo, read_audio, silero
from .dense_teacher import file_digest
from .nsc_part6_recording_pilot import SOURCE, verify_stage
from .progress import Progress
from .sources import data_root

HOP = RATE // 100
FRAME = 2 * HOP
PHRASE = 160
SEARCH = 1_000
MIN_CORR = .65
MIN_MARGIN = .10
MAX_COPY_DB = -12.
MATCH_TOLERANCE = 3 * HOP


def track_info(audio, timing, speaker):
    with audio.open("rb") as stream:
        native_rate, channels, bits, count, _ = flac_streaminfo(stream.read(42))
    pcm = read_audio(audio)
    if (native_rate, channels, bits, count) != (RATE, 1, 16, len(pcm)):
        raise ValueError("decoded audio differs from the native 16 kHz mono stream")
    with timing.open() as stream:
        rows = [json.loads(line) for line in stream]
    if not rows or any(int(row["spkid"]) != int(speaker) for row in rows):
        raise ValueError("annotation speaker differs from the staged track")
    intervals = [(round(float(row["start"]) * RATE), round(float(row["end"]) * RATE)) for row in rows]
    if any(not 0 <= start < end <= count for start, end in intervals):
        raise ValueError("annotation timing extends outside the native audio")
    if intervals != sorted(intervals) or any(before[1] > after[0] for before, after in zip(intervals, intervals[1:])):
        raise ValueError("the track's annotation segments are not sorted and disjoint")
    return pcm, intervals


def spectral_frames(pcm):
    signal = np.lib.stride_tricks.sliding_window_view(pcm, FRAME)[::HOP]
    bins = np.fft.rfftfreq(FRAME, 1 / RATE)
    edges = (200, 350, 500, 750, 1100, 1600, 2300, 3200)
    meter = np.empty((len(signal), len(edges) - 1))
    win = np.hanning(FRAME)
    for start in range(0, len(signal), 2048):
        fft = np.abs(np.fft.rfft(signal[start:start + 2048] * win)) ** 2
        meter[start:start + len(fft)] = np.column_stack(
            [fft[:, (bins >= lo) & (bins < hi)].mean(1) for lo, hi in zip(edges, edges[1:])])
    power = 10 * np.log10(np.maximum(meter.mean(axis=1), 1))
    bands = np.log(np.maximum(meter, 1))
    return bands, power


def candidate_windows(activity, count):
    per_third = [[], [], []]
    for start, end in activity:
        for edge in range(math.ceil(start / HOP), (end - FRAME) // HOP + 2 - PHRASE, PHRASE):
            per_third[min(2, 3 * edge * HOP // count)].append(edge)
    return [edge for subset in per_third
            for edge in (subset[i] for i in np.linspace(0, len(subset) - 1, min(len(subset), 8)).astype(int))]


def local_activity(pcm, vad, intervals):
    active = sum(end - start for start, end in vad)
    gaps = [(after[0] - before[1]) / RATE for before, after in zip(vad, vad[1:])]
    return {"decoded_samples": len(pcm), "duration_seconds": len(pcm) / RATE,
            "annotation_segments": len(intervals),
            "annotation_partition_spans_entire_track": intervals[0][0] == 0 and intervals[-1][1] == len(pcm)
            and all(before[1] == after[0] for before, after in zip(intervals, intervals[1:])),
            "active_fraction": active / len(pcm), "vad_spans": len(vad),
            "vad_pauses_ge_300ms": sum(gap >= .3 for gap in gaps),
            "vad_pauses_ge_1s": sum(gap >= 1 for gap in gaps),
            "zero_sample_fraction": float(np.mean(pcm == 0))}


def direct_power(power, activity):
    parts = [power[start // HOP:end // HOP] for start, end in activity if end - start >= RATE // 2]
    return float(np.median(np.concatenate(parts))) if parts else None


def match_phrases(source, receiver, edges, source_direct, receiver_direct, *, max_copy_db=MAX_COPY_DB):
    bands, source_power = source
    other, power = receiver
    if receiver_direct is None or source_direct is None:
        return [], Counter()
    matches, rejected = [], Counter()
    for start in edges:
        if np.median(source_power[start:start + PHRASE]) < source_direct - 6:
            rejected["quiet_reference"] += 1
            continue
        wanted = bands[start:start + PHRASE].copy()
        wanted -= wanted.mean(axis=0)
        left = max(0, start - SEARCH)
        right = min(len(other), start + SEARCH + PHRASE)
        if right - left < PHRASE or len(wanted) != PHRASE:
            rejected["missing_context"] += 1
            continue
        check = other[left:right]
        cum = np.cumsum(np.vstack((np.zeros_like(check[:1]), check)), axis=0)
        cum_sq = np.cumsum(np.vstack((np.zeros_like(check[:1]), check ** 2)), axis=0)
        means = cum[PHRASE:] - cum[:-PHRASE]
        sums = cum_sq[PHRASE:] - cum_sq[:-PHRASE]
        energy = (sums - means ** 2 / PHRASE).sum(1)
        fft_size = 1 << (len(check) + PHRASE - 2).bit_length()
        moving = np.fft.rfft(check, n=fft_size, axis=0)
        reference = np.fft.rfft(wanted[::-1], n=fft_size, axis=0)
        dot = np.fft.irfft(np.sum(moving * reference, axis=1), n=fft_size)[PHRASE - 1:len(check)]
        score = dot / np.sqrt(np.maximum(np.sum(wanted ** 2) * energy, 1e-12))
        peak = int(np.argmax(score))
        best = score[peak]
        comparison = np.r_[score[:max(0, peak - 12)], score[peak + 13:]]
        target = left + peak
        margin = best - comparison.max() if len(comparison) else 0.
        echo = float(np.median(power[target:target + PHRASE]) - receiver_direct)
        if not 0 < peak < len(score) - 1 or best < MIN_CORR or margin < MIN_MARGIN:
            rejected["weak_or_ambiguous_peak"] += 1
        elif max_copy_db is not None and echo > max_copy_db:
            rejected["copy_too_loud"] += 1
        else:
            alternatives = np.flatnonzero(score >= best - MIN_MARGIN)
            matches.append({"start": start * HOP, "copy": target * HOP, "correlation": float(best),
                            "copy_min": int(left + alternatives[0]) * HOP,
                            "copy_max": int(left + alternatives[-1]) * HOP,
                            "margin": float(margin), "relative_copy_db": echo})
    return matches, rejected


def inequality(point, reverse):
    latest_copy = point["copy_max"] / RATE
    if reverse:
        return -latest_copy, -1, -point["start"] / RATE + MATCH_TOLERANCE / RATE
    return point["start"] / RATE, 1, latest_copy + MATCH_TOLERANCE / RATE


def vertices(forward, reverse):
    limits = np.array([*(inequality(row, False) for row in forward),
                       *(inequality(row, True) for row in reverse)], dtype=float)
    points = []
    for first, second in combinations(limits, 2):
        matrix = np.stack((first[:2], second[:2]))
        if abs(np.linalg.det(matrix)) <= 1e-10:
            continue
        point = np.linalg.solve(matrix, (first[2], second[2]))
        if point[0] > 0 and np.all(limits[:, :2] @ point <= limits[:, 2] + 1e-8):
            points.append(point.tolist())
    return points


def clock_bounds(forward, reverse, duration_a, duration_b):
    # Require both directions in early, middle and late thirds. Hold out the
    # strongest middle match per direction while fitting the clock polygon.
    training, checks = [], []
    for rows, duration, reverse_direction in ((forward, duration_a, False), (reverse, duration_b, True)):
        thirds = {min(2, 3 * row["start"] // duration) for row in rows}
        if thirds != {0, 1, 2}:
            return None
        held = max((row for row in rows if 3 * row["start"] // duration == 1), key=lambda row: row["correlation"])
        checks.append(inequality(held, reverse_direction))
        training.append([row for row in rows if row is not held])
    fit_region = vertices(*training)
    fitted = vertices(forward, reverse)
    if not fit_region or not fitted:
        return None
    confirmed = all(a * slope + b * offset <= bound + 1e-8
                    for a, b, bound in checks for slope, offset in fit_region)
    bracket = (max(min(row["start"] for row in training[0]), min(row["copy"] for row in training[1])),
               min(max(row["start"] for row in training[0]), max(row["copy"] for row in training[1])))
    if bracket[0] >= bracket[1]:
        return None
    width = []
    for point in bracket:
        possible = [slope * point + offset * RATE for slope, offset in fitted]
        width.append((max(possible) - min(possible)) * 1000 / RATE)
    return {"slope_offset_seconds_vertices": fitted, "fit_only_vertices": fit_region,
            "held_out_confirms_entire_fit_region": confirmed, "bracket_a_samples": list(bracket),
            "clock_width_ms_at_bracket_ends": width}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("clock probe uses staged campaign audio only")
    folder = root / "raw" / SOURCE
    manifest = verify_stage(root)
    detect = silero()
    data, results = {}, {}
    meter = Progress("nsc-part6-recording-check", 4, "tracks")
    try:
        for pair in (1, 2):
            results[str(pair)] = {"tracks": []}
            for row in (row for row in manifest["files"] if row["pair"] == pair and row["file"].endswith(".flac")):
                audio = folder / row["file"]
                pcm, intervals = track_info(audio, audio.with_suffix(".jsonl"), row["speaker"])
                activity = list(detect(pcm))
                features = spectral_frames(pcm)
                data[pair, row["track"]] = (features, candidate_windows(activity, len(pcm)),
                                            direct_power(features[1], activity), len(pcm))
                results[str(pair)]["tracks"].append({**local_activity(pcm, activity, intervals),
                                                     "decoded_pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()})
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    for pair in (1, 2):
        directions = []
        counts = []
        for source, receiver in ((1, 2), (2, 1)):
            features, edges, own_direct, n = data[pair, source]
            other, _, direct, _ = data[pair, receiver]
            wrong, _, wrong_direct, _ = data[3 - pair, receiver]
            matches, rejected = match_phrases(features, other, edges, own_direct, direct)
            controls, _ = match_phrases(features, wrong, edges, own_direct, wrong_direct)
            counts.append({"anchors": len(edges), "matches": len(matches), "matched_details": matches,
                           "anchors_by_third": [sum(min(2, 3 * h * HOP // n) == i for h in edges) for i in range(3)],
                           "matches_by_third": [sum(min(2, 3 * row["start"] // n) == i for row in matches)
                                                for i in range(3)],
                           "rejected": rejected, "wrong_pair_matches": len(controls), "native_samples": n})
            directions.append(matches)
        a, b = (data[pair, track] for track in (1, 2))
        if len({item["decoded_pcm_sha256"] for item in results[str(pair)]["tracks"]}) != 2:
            raise ValueError("the staged participant tracks have identical PCM")
        bounds = clock_bounds(*directions, a[3], b[3]) if all(x["wrong_pair_matches"] == 0 for x in counts) else None
        results[str(pair)].update({"directions": counts, "clock_bounds": bounds})
    report = {"decision_ms": 300, "manifest_sha256": file_digest(folder / "MANIFEST.json"),
              "same_lengths_do_not_establish_alignment": True, "pairs": results,
              "pairs_with_feasible_affine_map": sum(row["clock_bounds"] is not None for row in results.values())}
    _write_json(root / "reports/nsc-part6-clock-probe.json", report)
    return report


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    report = run(data_root())
    print(json.dumps({key: {"matched": [row["matches"] for row in pair["directions"]],
                           "controls": [row["wrong_pair_matches"] for row in pair["directions"]],
                           "clock_bounded": pair["clock_bounds"] is not None}
                      for key, pair in report["pairs"].items()}, indent=2))
