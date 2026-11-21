"""Stage two hash-pinned original same-room dyads and test their native clocks."""

import argparse
import hashlib
import json
from math import gcd
import re
import subprocess
import tarfile

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly

from .ami_calibration import _write_json
from .audio import RATE, flac_streaminfo, read_audio, silero
from .dense_teacher import file_digest
from .language_failure_audit import read_receipt
from .nsc_part4_native_preflight import SOURCE, inventory
from .nsc_part6_clock_probe import (HOP, PHRASE, SEARCH, candidate_windows, clock_bounds, direct_power, local_activity,
                                    match_phrases, spectral_frames)
from .nsc_part6_preflight import BLOCK, RangeReader
from .progress import Progress
from .sources import budget_bytes, data_root, load_sources


JOB = "nsc-part4-native-clock"
PHRASE_SAMPLES = PHRASE * HOP
DIRECTIONAL_COPY_LIMIT_DB = -10.


def plan(preflight):
    paths = []
    seconds = preflight.get("sample_prefix_seconds")
    for pair, task in enumerate(preflight["selected"], 1):
        for channel, (speaker, files) in enumerate(sorted(task["people"].items()), 1):
            for fmt, entry in sorted(files.items()):
                if fmt == "flac" and seconds:
                    rate, channels, bits, _, _ = entry["format"]
                    transport_bound = (seconds + 2) * rate * channels * (bits // 8) + (256 << 10)
                    entry = {**entry, "read_bytes": min(entry["read_bytes"], transport_bound)}
                paths.append({"pair": pair, "channel": channel, "speaker": speaker, **entry,
                              "local": f"pair-{pair}-track-{channel}.{fmt}"})
    if len(paths) != 8 or len({row["speaker"] for row in paths}) != 4:
        raise ValueError("the preflight does not lock two complete, distinct dyads")
    return paths


def stage(root, preflight, preflight_sha):
    source = load_sources()[SOURCE]
    desired = plan(preflight)
    cap = source.get("max_total_bytes")
    payload = sum(row.get("read_bytes", row["bytes"]) for row in desired)
    if not preflight["payload_allowed"] or payload >= cap:
        raise ValueError("the preflight did not approve the full dyad transfer")
    folder = root / "raw" / SOURCE
    manifest = folder / "MANIFEST.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if previous["preflight_sha256"] != preflight_sha or len(previous["files"]) != len(desired):
            raise ValueError("staged files belong to a different preflight")
        for actual, expected in zip(previous["files"], desired, strict=True):
            if (any(actual.get(key) != v for key, v in expected.items())
                    or file_digest(folder / expected["local"]) != actual["sha256"]):
                raise ValueError("the staged files differ from the locked original payload")
        return folder, previous
    raw_bytes = sum(path.stat().st_size for path in (root / "raw").rglob("*") if path.is_file())
    if raw_bytes + cap > budget_bytes():
        raise ValueError("the pilot exceeds the remaining corpus download budget")
    fragments, inventory_bytes = inventory()
    if [list(item) for item in fragments] != preflight["archive_fragments"]:
        raise ValueError("the native archive inventory changed")
    reader = RangeReader(fragments, byte_limit=cap - inventory_bytes - preflight["inspected_metadata_bytes"],
                         request_limit=32)
    folder.mkdir(parents=True, exist_ok=True)
    keep = []
    meter = Progress(f"{JOB}-stage", payload, "B")
    try:
        for row in desired:
            offset, count = row["offset"], row["bytes"]
            header = tarfile.TarInfo.frombuf(reader.read(offset - BLOCK, BLOCK), "utf-8", "strict")
            if (header.name.rsplit("/", 1)[-1], header.size, header.isfile()) != (row["name"], count, True):
                raise ValueError("the source tar member changed")
            content = reader.read(offset, row.get("read_bytes", count))
            if row["local"].endswith(".flac") and list(flac_streaminfo(content[:42])) != row["format"]:
                raise ValueError("FLAC identity differs from the inspected header")
            path = folder / row["local"]
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
            keep.append({**row, "sha256": file_digest(path)})
            meter.advance(len(content))
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"source": SOURCE, "licence": source.licence, "attribution": source.attribution,
              "revision": source.get("revision"), "preflight_sha256": preflight_sha,
              "files": keep, "sample_prefix_seconds": preflight.get("sample_prefix_seconds"),
              "transfer_bytes_including_qualifying_preflight": reader.bytes + inventory_bytes
              + preflight["inspected_metadata_bytes"]}
    _write_json(manifest, report)
    return folder, report


def decode_prefix(path, seconds, rate):
    frames = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                             "-read_intervals", f"%+{seconds}", "-show_frames", "-show_streams",
                             "-show_entries",
                             "frame=pts,pkt_pts,pkt_dts,best_effort_timestamp,nb_samples:stream=time_base,sample_rate",
                             "-of", "json", str(path)], check=True, capture_output=True)
    probe = json.loads(frames.stdout)
    stream, = probe["streams"]
    if stream["time_base"] != f"1/{rate}" or int(stream["sample_rate"]) != rate:
        raise ValueError("native FLAC timestamps have an unexpected sample time base")
    observed = 0
    for frame in probe["frames"]:
        timestamp = next((frame[field] for field in ("pts", "pkt_pts", "pkt_dts", "best_effort_timestamp")
                          if field in frame), None)
        if timestamp != observed or frame["nb_samples"] <= 0:
            raise ValueError("native FLAC frame timestamps break continuity from sample zero")
        observed += frame["nb_samples"]
    if observed < seconds * rate:
        raise ValueError("complete native frames do not cover the fixed sample prefix")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "crccheck+explode",
                          "-i", str(path), "-map", "0:a:0", "-t", str(seconds), "-ar", str(rate),
                          "-ac", "1", "-f", "s16le", "pipe:1"],
                         check=True, capture_output=True).stdout
    if len(raw) != 2 * seconds * rate:
        raise ValueError("CRC-checked decoding did not cover the exact fixed sample prefix")
    pcm = np.frombuffer(raw, "<i2")
    if rate != RATE:
        divisor = gcd(RATE, rate)
        converted = resample_poly(pcm.astype(np.float32), RATE // divisor, rate // divisor)
        pcm = np.rint(converted).clip(-32768, 32767).astype("<i2")
    if len(pcm) != seconds * RATE:
        raise ValueError("analysis resampling changed prefix coverage")
    return pcm, {"native_prefix_sha256": hashlib.sha256(raw).hexdigest(),
                 "native_prefix_samples": seconds * rate, "native_contiguous_frame_count": len(probe["frames"])}


def source_track(folder, row, detect, prefix_seconds):
    path = folder / row["local"]
    rate, _, _, count, _ = row["format"]
    if prefix_seconds:
        pcm, prefix_receipt = decode_prefix(path, prefix_seconds, rate)
    else:
        pcm, prefix_receipt = read_audio(path), {}
        if abs(len(pcm) - round(count * RATE / rate)) > 2:
            raise ValueError("decoding changed the duration of an original track")
    with path.with_suffix(".jsonl").open() as stream:
        annotation = [json.loads(line) for line in stream]
    intervals, lexical = [], []
    for segment in annotation:
        if "spkid" in segment and int(segment["spkid"]) != int(row["speaker"]):
            raise ValueError("an annotation contains a different speaker identity")
        start, end = (round(float(segment[key]) * RATE) for key in ("start", "end"))
        if prefix_seconds:
            start, end = max(0, start), min(len(pcm), end)
        if 0 <= start < end <= len(pcm):
            intervals.append((start, end))
            text = re.sub(r"<[^>]*>|\[[^\]]*\]", "", str(segment.get("text", "")))
            if any(char.isalpha() for char in text):
                lexical.append((start, end))
    intervals.sort()
    if not intervals:
        raise ValueError("no valid recording-local annotation intervals")
    activity = list(detect(pcm))
    features = spectral_frames(pcm)
    anchors = []
    interval = 0
    for start in candidate_windows(activity, len(pcm)):
        while interval < len(intervals) and intervals[interval][1] <= start * HOP:
            interval += 1
        if interval < len(intervals) and intervals[interval][0] <= start * HOP < intervals[interval][1]:
            anchors.append(int(start))
    profile = {**local_activity(pcm, activity, intervals), **prefix_receipt, "native_rate": rate,
               "annotation_intervals_outside_pcm": len(annotation) - len(intervals)}
    return profile, (features, anchors, direct_power(features[1], activity), lexical, len(pcm))


def directional(matches, focal, partner):
    return [point for point in matches if point["relative_copy_db"] <= DIRECTIONAL_COPY_LIMIT_DB
            and any(start <= point["start"] and point["start"] + PHRASE_SAMPLES <= end for start, end in focal)
            and not any(start < point["copy_max"] + PHRASE_SAMPLES and end > point["copy_min"]
                        for start, end in partner)]


def disjoint_directions(forward, reverse):
    shared = [(i, j) for i, left in enumerate(forward) for j, right in enumerate(reverse)
              if abs(left["start"] - right["copy"]) < PHRASE_SAMPLES
              or abs(right["start"] - left["copy"]) < PHRASE_SAMPLES]
    left, right = ({row[index] for row in shared} for index in (0, 1))
    return ([point for i, point in enumerate(forward) if i not in left],
            [point for i, point in enumerate(reverse) if i not in right])


def null_matches(own, control, selected, direct, received):
    matches, rejected = match_phrases(own, control, selected, direct, received, max_copy_db=None)
    return {"matches": len(matches), "searched": len(matches) + rejected["weak_or_ambiguous_peak"]}


def envelope_offset(a, b):
    left, right = (np.clip(values, np.quantile(values, .1), np.quantile(values, .9))
                   for values in (a, b))
    left, right = left - left.mean(), right - right.mean()
    if not np.any(left) or not np.any(right):
        return None
    lags = correlation_lags(len(right), len(left))
    mask = np.abs(lags) <= SEARCH
    scores = correlate(right, left, method="fft")[mask] / (np.linalg.norm(left) * np.linalg.norm(right))
    best = int(np.argmax(scores))
    return {"lag_b_minus_a_ms": int(lags[mask][best]) * (1000 * HOP // RATE),
            "correlation": float(scores[best]), "scope": "coarse activity envelope only"}


def test_clock(tracks, audio, pair):
    results, anchors = [], []
    for a, b in ((1, 2), (2, 1)):
        own, selected, direct, own_words, _ = audio[pair, a]
        receiver, _, received, receiver_words, _ = audio[pair, b]
        control, _, unrelated, _, _ = audio[3 - pair, b]
        matches, rejected = match_phrases(own, receiver, selected, direct, received, max_copy_db=None)
        wrong_pair = null_matches(own, control, selected, direct, unrelated)
        shifted = tuple(np.roll(values, 3 * SEARCH, axis=0) for values in receiver)
        wrong_time = (null_matches(own, shifted, selected, direct, received) if len(receiver[0]) > 6 * SEARCH
                      else {"matches": 0, "searched": 0})
        owned = directional(matches, own_words, receiver_words)
        results.append({"candidates": len(selected), "matches": matches, "rejections": rejected,
                        "wrong_pair": wrong_pair, "wrong_time": wrong_time,
                        "directional_before_deduplication": len(owned)})
        anchors.append(owned)
    anchors = disjoint_directions(*anchors)
    for direction, points in zip(results, anchors, strict=True):
        direction["directional_anchors"] = points
    a, b = (audio[pair, channel] for channel in (1, 2))
    nulls_pass = all(row[key]["matches"] == 0 and row[key]["searched"] >= 3
                     for row in results for key in ("wrong_pair", "wrong_time"))
    sufficient = sum(map(len, anchors)) >= 10
    bounds = clock_bounds(*anchors, a[-1], b[-1]) if nulls_pass and sufficient else None
    passed = bool(bounds and bounds["held_out_confirms_entire_fit_region"]
                  and max(bounds["clock_width_ms_at_bracket_ends"]) <= 80.)
    drift = ([1e6 * (min(row[0] for row in bounds["fit_only_vertices"]) - 1),
              1e6 * (max(row[0] for row in bounds["fit_only_vertices"]) - 1)] if bounds else None)
    return {"tracks": tracks, "directions": results, "clock": bounds, "drift_ppm": drift,
            "passes": passed, "independent_anchor_minimum_met": sufficient, "nulls_pass": nulls_pass,
            "coarse_envelope_offset": None if passed else envelope_offset(a[0][1], b[0][1])}


def run(root, expected_sha, *, prefix=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("original data must remain on Vast")
    preflight_name = f"{SOURCE}-prefix" if prefix else SOURCE
    preflight, preflight_sha = read_receipt(root, preflight_name)
    if preflight_sha != expected_sha:
        raise ValueError("the expected native header preflight changed")
    folder, manifest = stage(root, preflight, preflight_sha)
    detect = silero()
    audio = {}
    profiles = {1: [], 2: []}
    meter = Progress(JOB, 4, "tracks")
    try:
        for row in manifest["files"]:
            if not row["local"].endswith(".flac"):
                continue
            pair, channel = row["pair"], row["channel"]
            profile, audio[pair, channel] = source_track(folder, row, detect, manifest.get("sample_prefix_seconds"))
            profiles[pair].append(profile)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    results = {pair: test_clock(profiles[pair], audio, pair) for pair in (1, 2)}
    report = {"scope": "native-clock engineering only; no assigned hold or shift labels",
              "sample_prefix_seconds": manifest.get("sample_prefix_seconds"),
              "preflight_sha256": expected_sha, "manifest_sha256": file_digest(folder / "MANIFEST.json"),
              "transfer_bytes_including_qualifying_preflight": manifest["transfer_bytes_including_qualifying_preflight"],
              "pairs": results, "both_pass": all(row["passes"] for row in results.values())}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps({pair: {"passed": row["passes"], "matched": [len(x["matches"]) for x in row["directions"]],
                             "directional": [len(x["directional_anchors"]) for x in row["directions"]]}
                      for pair, row in results.items()}))


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--prefix", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(data_root(), args.preflight_sha256, prefix=args.prefix)
