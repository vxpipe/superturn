"""Plan a bounded Hindi conversation pilot from original paired IITM WAVs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import PurePosixPath
import struct
from urllib.request import Request, urlopen

import numpy as np

from .acquire import parse_wav_header, wav_bytes
from .ami_calibration import _write_json
from .audio import RATE, read_audio, silero
from .dense_teacher import file_digest
from .dral_label_pilot import isolation, summarize
from .harper_label_yield import events, one_to_one
from .labels import Speech
from .progress import Progress
from .sources import budget_bytes, data_root, load_sources

SOURCE = "spring-hindi-paired-pilot"
WINDOW = 30 * RATE
WINDOWS = 6
META_MAX = 15_000_000


def stage_metadata(root, source):
    folder = root / "raw" / SOURCE
    expected = source.get("metadata_sha256")
    origin = source.get("origin")
    meter = Progress("spring-hindi-tables", len(expected), "tables")
    result = {}
    bytes_left = META_MAX
    try:
        for relative, digest in sorted(expected.items()):
            path = folder / relative
            if path.is_file():
                if file_digest(path) != digest:
                    raise ValueError(f"stored metadata changed: {relative}")
            else:
                request = Request(origin + relative, headers={"Accept-Encoding": "identity"})
                with urlopen(request, timeout=45) as response:
                    announced = response.headers.get("Content-Length")
                    if ((announced is not None and int(announced) > bytes_left)
                            or response.headers.get("Content-Encoding", "identity") != "identity"):
                        raise ValueError("metadata exceeds the pilot budget")
                    payload = response.read(bytes_left + 1)
                if len(payload) > bytes_left or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError(f"provider metadata changed: {relative}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".part")
                temporary.write_bytes(payload)
                temporary.replace(path)
            size = path.stat().st_size
            bytes_left -= size
            if bytes_left < 0:
                raise ValueError("metadata exceeds the pilot budget")
            result[relative] = {"path": relative, "bytes": size, "sha256": digest}
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return folder, result


def intervals(folder, split):
    path = folder / split
    audio = {row[0]: PurePosixPath(row[-1]).stem
             for line in (path / "wav.scp").read_text().splitlines() if (row := line.split())}
    speakers = dict(line.split() for line in (path / "utt2spk").read_text().splitlines() if line)
    by_call, native_tags = defaultdict(list), defaultdict(set)
    for line in (path / "segments").read_text().splitlines():
        if not line:
            continue
        key, record, begin, end = line.split()
        native = audio.get(record, record)
        if not native.startswith("hi_IN_") or not native.endswith(("_Left", "_Right")):
            continue
        recording, side = native.rsplit("_", 1)
        start, stop = round(float(begin) * RATE), round(float(end) * RATE)
        if not 0 <= start < stop or key not in speakers:
            raise ValueError("invalid provider interval or missing utterance key")
        by_call[recording].append(Speech(side, start, stop))
        native_tags[native].add(speakers[key])
    summary = {"calls": len(by_call), "intervals": sum(map(len, by_call.values())),
               "channels_with_one_speaker_key": sum(len(tags) == 1 for tags in native_tags.values()),
               "channels": len(native_tags)}
    return by_call, summary


def tiles(train, protected):
    """Whole windows avoid every interval listed by the provider in dev or eval."""
    choices = []
    for recording, spans in sorted(train.items()):
        if {span.speaker for span in spans} != {"Left", "Right"}:
            continue
        forbidden = protected.get(recording, [])
        first, last = min(x.start for x in spans), max(x.end for x in spans)
        candidates, _ = events(spans, first, last)
        for start in range(first // WINDOW * WINDOW, last - WINDOW + 1, WINDOW):
            stop = start + WINDOW
            if any(span.start < stop and span.end > start for span in forbidden):
                continue
            anchors = [(side, edge, onset, label) for side, edge, onset, label in candidates
                       if start + 3 * RATE <= edge and onset + 2 * RATE < stop]
            counts = Counter(label for _, _, _, label in anchors)
            if not all(counts[label] for label in ("hold", "shift")):
                continue
            priority = (-min(counts.values()), -len(anchors), recording, start)
            choices.append((priority, {"recording": recording, "start_native_sample": start,
                                       "frames": WINDOW, "segment_candidates": anchors}))
    seen, selection = set(), []
    for _, tile in sorted(choices, key=lambda pair: pair[0]):
        recording = tile["recording"]
        if recording not in seen:
            selection.append(tile)
            seen.add(recording)
        if len(selection) == WINDOWS:
            break
    return selection


def wav_headers(source, selection):
    origin = source.get("origin")
    meter = Progress("spring-hindi-wav-headers", len(selection) * 2, "channel headers")
    try:
        for tile in selection:
            channels = {}
            for side in ("Left", "Right"):
                url = f'{origin}Audio/{tile["recording"]}_{side}.wav'
                req = Request(url, headers={"Range": "bytes=0-4095", "Accept-Encoding": "identity"})
                with urlopen(req, timeout=30) as response:
                    length = response.headers.get("Content-Range", "")
                    etag = response.headers.get("ETag")
                    if (response.status != 206 or not length.startswith("bytes 0-4095/") or not etag
                            or response.headers.get("Content-Encoding", "identity") != "identity"):
                        raise ValueError("native source did not honor the bounded header range")
                    header = response.read(4097)
                if len(header) != 4096:
                    raise ValueError("unexpected native header length")
                fmt, offset = parse_wav_header(header)
                if struct.unpack("<HHIIHH", fmt[:16]) != (1, 1, RATE, 2 * RATE, 2, 16):
                    raise ValueError("expected native 16 kHz PCM channels")
                size = int(length.split("/")[-1])
                data_bytes, = struct.unpack_from("<I", header, offset - 4)
                if (offset + data_bytes > size or data_bytes % 2
                        or 2 * (tile["start_native_sample"] + tile["frames"]) > data_bytes):
                    raise ValueError("metadata interval extends past source payload")
                channels[side] = {"url": url, "bytes": size, "etag": etag, "offset": offset,
                                  "fmt": fmt.hex(), "data_bytes": data_bytes,
                                  "header_range_sha256": hashlib.sha256(header).hexdigest()}
                meter.advance(1)
            if channels["Left"]["data_bytes"] != channels["Right"]["data_bytes"]:
                raise ValueError("native channel lengths disagree")
            tile["tracks"] = channels
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("SPRING source metadata stays on Vast")
    source = load_sources()[SOURCE]
    baseline = sum(path.stat().st_size for path in (root / "raw").rglob("*") if path.is_file())
    if baseline + source.get("max_total_bytes") > budget_bytes():
        raise ValueError("the Hindi pilot reserve exceeds the campaign budget")
    folder, metadata = stage_metadata(root, source)
    train, train_summary = intervals(folder, "train")
    dev, dev_summary = intervals(folder, "dev")
    test, test_summary = intervals(folder, "eval")
    protected = {call: dev.get(call, []) + test.get(call, []) for call in dev.keys() | test.keys()}
    selection = tiles(train, protected)
    if not selection:
        raise ValueError("no protected-interval-safe train windows with both classes")
    wav_headers(source, selection)
    report = {"scope": "metadata feasibility only, role-play, no verified turn labels",
              "licence": source.licence, "metadata": metadata, "native_sample_rate": RATE,
              "provider_intervals": {"train": train_summary, "dev": dev_summary, "eval": test_summary},
              "provider_calls_crossing_train_dev": len(train.keys() & dev.keys()),
              "provider_calls_crossing_train_eval": len(train.keys() & test.keys()),
              "selection": selection,
              "planned_pcm_bytes": len(selection) * WINDOW * 2 * 2}
    output = root / "reports/spring-hindi-paired-plan.json"
    if output.is_file() and json.loads(output.read_text()) != json.loads(json.dumps(report)):
        raise ValueError("the candidate plan differs from the previous replay")
    _write_json(output, report)
    _write_json(folder / "MANIFEST.json", {"source": SOURCE, "licence": source.licence,
                                          "attribution": source.attribution,
                                          "selection_sha256": file_digest(output), "files": list(metadata.values())})
    return {"windows": len(selection), "planned_pcm_bytes": report["planned_pcm_bytes"],
            "plan_sha256": file_digest(output), "provider_intervals": report["provider_intervals"]}


def stage_audio(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("SPRING audio stays on Vast")
    source = load_sources()[SOURCE]
    folder = root / "raw" / SOURCE
    path = root / "reports/spring-hindi-paired-plan.json"
    plan_sha = file_digest(path)
    receipt = json.loads((folder / "MANIFEST.json").read_text())
    if receipt["selection_sha256"] != plan_sha:
        raise ValueError("the stage plan differs from the acquired metadata receipt")
    plan = json.loads(path.read_text())
    for item in plan["metadata"].values():
        if file_digest(folder / item["path"]) != item["sha256"]:
            raise ValueError("the staged metadata differs from the plan")
    reserve = plan["planned_pcm_bytes"] + 4096
    raw_bytes = sum(asset.stat().st_size for asset in (root / "raw").rglob("*") if asset.is_file())
    if (sum(item["bytes"] for item in plan["metadata"].values()) + reserve > source.get("max_total_bytes")
            or raw_bytes + reserve > budget_bytes()):
        raise ValueError("planned PCM exceeds allocated input budget")
    detect = silero()
    output = root / "reports/spring-hindi-paired-audio.json"
    old = json.loads(output.read_text()) if output.is_file() else None
    if old is not None and old["plan_sha256"] != plan_sha:
        raise ValueError("the saved audio evidence has a different plan")
    files, windows = {}, []
    meter = Progress("spring-hindi-audio-feasibility", len(plan["selection"]), "windows")
    try:
        for tile in plan["selection"]:
            start, frames = tile["start_native_sample"], tile["frames"]
            audio = []
            for side, track in sorted(tile["tracks"].items()):
                native_id = tile["recording"]
                relative = f"windows/{native_id}/{start}-{side}.wav"
                destination = folder / relative
                first = track["offset"] + 2 * start
                last = first + 2 * frames - 1
                if old is None or not destination.is_file():
                    req = Request(track["url"], headers={"Range": f"bytes={first}-{last}",
                                                         "If-Match": track["etag"], "Accept-Encoding": "identity"})
                    with urlopen(req, timeout=30) as response:
                        expected = f'bytes {first}-{last}/{track["bytes"]}'
                        if (response.status != 206 or response.headers.get("Content-Range") != expected
                                or response.headers.get("ETag") != track["etag"]
                                or response.headers.get("Content-Encoding", "identity") != "identity"):
                            raise ValueError("the source changed or refused the original byte range")
                        pcm = response.read(2 * frames + 1)
                    if len(pcm) != 2 * frames:
                        raise ValueError("native sample range is incomplete")
                    encoded = wav_bytes(bytes.fromhex(track["fmt"]), pcm)
                    if old is not None and hashlib.sha256(encoded).hexdigest() != old["files"][relative]:
                        raise ValueError("rebuilt WAV does not match its pinned content")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".part")
                    temporary.write_bytes(encoded)
                    temporary.replace(destination)
                digest = file_digest(destination)
                if old is not None and digest != old["files"][relative]:
                    raise ValueError("stored WAV differs from its source receipt")
                files[relative] = digest
                audio.append(read_audio(destination))
            tracks = np.stack(audio)
            if tracks.shape != (2, frames) or np.array_equal(tracks[0], tracks[1]):
                raise ValueError("the staged call is not a distinct aligned channel pair")
            activity = [detect(track) for track in tracks]
            screen = isolation(tracks, activity)
            vad, outcomes = summarize([Speech(side, lo, hi) for side, spans in zip(
                ("Left", "Right"), activity, strict=True) for lo, hi in spans], frames)
            metadata = [(side, edge - start, onset - start, label)
                        for side, edge, onset, label in tile["segment_candidates"]]
            agreement = Counter()
            one_to_one(vad, metadata, agreement)
            windows.append({"recording": tile["recording"], "start_native_sample": start,
                            "isolation": screen, "silero": outcomes, "metadata_agreement": agreement})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result = {"scope": "acoustic and coarse segment feasibility, no training labels",
              "plan_sha256": plan_sha, "files": files, "windows": windows}
    if old is not None and old != result:
        raise ValueError("audio feasibility results changed")
    _write_json(output, result)
    return {"windows": len(windows), "tracks": len(files),
            "passed_isolation": sum(w["isolation"]["passed"] for w in windows),
            "agreement": sum((w["metadata_agreement"] for w in windows), Counter()),
            "agreement_on_isolated_windows": sum((w["metadata_agreement"] for w in windows
                                                  if w["isolation"]["passed"]), Counter())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-audio", action="store_true")
    args = parser.parse_args()
    print(json.dumps((stage_audio if args.stage_audio else run)(data_root()), indent=2))
