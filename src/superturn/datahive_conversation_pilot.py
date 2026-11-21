"""Audit behavioral pause candidates from pinned stereo phone recordings."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json

import av
import numpy as np
import torch
import whisper

from .ami_calibration import _write_json
from .audio import RATE, silero
from .dense_teacher import file_digest
from .dral_label_pilot import inspect
from .open_speech_teacher_probe import WHISPER_SHA256
from .progress import Progress
from .sources import data_root, load_sources

SOURCE = "datahive-conversation-pilot"


def decode_tracks(path):
    resampler = av.AudioResampler(format="s16p", layout="stereo", rate=RATE)
    with av.open(str(path)) as container:
        stream, = container.streams.audio
        if stream.codec_context.name not in {"mp3", "mp3float"}:
            raise ValueError("expected native DataHive MP3")
        if stream.rate != 48000 or stream.channels != 2:
            raise ValueError("expected native 48 kHz stereo")
        frames = [output.to_ndarray() for frame in container.decode(stream)
                  for output in resampler.resample(frame)]
        frames.extend(frame.to_ndarray() for frame in resampler.resample(None))
    tracks = np.concatenate(frames, axis=1).astype("<i2", copy=False)
    if tracks.shape[0] != 2 or not tracks.shape[1] or np.array_equal(*tracks):
        raise ValueError("decoder did not produce two distinct aligned channels")
    return tracks


def language_profile(recognizer, tracks):
    scores = []
    for track in tracks:
        windows = []
        for segment in np.array_split(track, 3):
            # Fixed centre samples give coarse language coverage of the call.
            centre = len(segment) // 2
            signal = segment[max(0, centre - 15 * RATE):centre + 15 * RATE]
            audio = whisper.pad_or_trim(signal.astype(np.float32) / 32768)
            mel = whisper.log_mel_spectrogram(audio, n_mels=recognizer.dims.n_mels).to(recognizer.device)
            with torch.inference_mode():
                _, probabilities = recognizer.detect_language(mel)
            ranked = sorted(probabilities.items(), key=lambda pair: (-pair[1], pair[0]))
            windows.append(ranked[:2])
        scores.append(windows)
    return scores


def inventory(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("DataHive audio stays on Vast")
    source = load_sources()[SOURCE]
    folder = root / "raw" / SOURCE
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    checksums = source.get("audio_sha256")
    if (manifest["revision"] != source.get("revision")
            or {x["path"] for x in manifest["files"]} != set(source.get("files"))
            or sum(x["planned_bytes"] for x in manifest["files"]) > source.get("max_total_bytes")):
        raise ValueError("unexpected acquisition receipt")
    metadata = folder / "preview/metadata.jsonl"
    by_name = {}
    for line in metadata.read_text().splitlines():
        item = json.loads(line)
        by_name[item["file_name"]] = item
    return source, folder, metadata, by_name


def load_call(folder, filename, digest, by_name):
    path = folder / filename
    if file_digest(path) != digest:
        raise ValueError(f"publisher object hash mismatch: {filename}")
    info = by_name[path.name]
    language = path.name.split("_")[0]
    if language != info["language"]:
        raise ValueError("publisher file and metadata language disagree")
    tracks = decode_tracks(path)
    if abs(tracks.shape[1] / RATE - info["duration_seconds"]) > .2:
        raise ValueError("decoded duration differs from the common timeline")
    receipt = {
        "language": language, "samples": tracks.shape[1], "source_sha256": digest,
        "pcm_by_channel_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in tracks],
    }
    return tracks, receipt


def verify_rebuild(root):
    source, folder, metadata, by_name = inventory(root)
    report = json.loads((root / "reports/datahive-conversation-pilot.json").read_text())
    executable = {key: list(value) for key, value in av.library_versions.items()}
    if (report["revision"] != source.get("revision") or report["metadata_sha256"] != file_digest(metadata)
            or report["decoded_rate"] != RATE or report["av_version"] != av.__version__
            or report["decoder_libraries"] != executable):
        raise ValueError("the source or decoder differs from the recorded pilot")
    checksums = source.get("audio_sha256")
    if set(checksums) != report["calls"].keys():
        raise ValueError("the call selection changed")
    for filename, digest in sorted(checksums.items()):
        _, actual = load_call(folder, filename, digest, by_name)
        reference = report["calls"][filename]
        if any(reference[key] != value for key, value in actual.items()):
            raise ValueError(f"the decoded recording changed: {filename}")
        for side, edge, onset, label in reference["provisional_pauses"]:
            if (side not in ("0", "1") or label not in ("hold", "shift")
                    or not 0 <= edge < onset < actual["samples"]):
                raise ValueError(f"recorded pause is outside the rebuilt tracks: {filename}")
    return {"verified_calls": len(checksums), "revision": source.get("revision")}


def run(root):
    source, folder, metadata, by_name = inventory(root)
    checksums = source.get("audio_sha256")
    checkpoint = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(checkpoint) != WHISPER_SHA256:
        raise ValueError("existing Whisper Large v3 checkpoint has the wrong hash")
    recognize = whisper.load_model(str(checkpoint), device="cuda")
    detect = silero()
    receipts, totals = {}, defaultdict(Counter)
    offsets = defaultdict(list)
    meter = Progress("datahive-conversation-audit", len(checksums), "calls")
    try:
        for filename, digest in sorted(checksums.items()):
            tracks, receipt = load_call(folder, filename, digest, by_name)
            language = receipt["language"]
            status, deltas, pauses = inspect(tracks, detect, recognize, language)
            profile = language_profile(recognize, tracks)
            supported = Counter(row[-1] for row in pauses)
            totals[language].update(supported)
            totals[language]["calls_with_both_classes"] += all(supported[x] > 0 for x in ("hold", "shift"))
            totals[language]["isolation_pass"] += status["isolation"]["passed"]
            offsets[language].extend(deltas)
            receipts[filename] = {
                **receipt,
                "language_posteriors_by_channel": profile, "screen": status,
                "provisional_pauses": pauses,
            }
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    for language, values in offsets.items():
        if values:
            totals[language]["word_endpoint_median_ms"] = float(np.median(values))
            totals[language]["word_endpoint_iqr_ms"] = float(np.diff(np.percentile(values, (25, 75)))[0])
    report = {
        "scope": "train-side feasibility only, behavioral next-speaker agreement, no human adjudication",
        "revision": source.get("revision"), "bytes": source.get("max_total_bytes"),
        "speaker_scope": "global identities unavailable; keep the pilot as one component",
        "metadata_sha256": file_digest(metadata), "whisper_sha256": WHISPER_SHA256,
        "decoded_rate": RATE, "av_version": av.__version__, "decoder_libraries": av.library_versions,
        "languages": totals, "calls": receipts,
    }
    _write_json(root / "reports/datahive-conversation-pilot.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-rebuild", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify_rebuild(data_root()) if args.verify_rebuild else run(data_root())["languages"]
    print(json.dumps(result, indent=2))
