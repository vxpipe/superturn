"""Suggest shared gaps on the locally authorized stereo listening packet."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import file_digest

MODEL_SHA256 = "9ccdacc4719d8aa7e45a77536bfabec45a03ba1f2fad5e241ab4060b24238a85"
FRAME = 512
CONTEXT = 64
FRAME_MS = 32
MIN_GAP_MS = 300
MIN_BOUT_MS = 96
OVERLAP_MARGIN_MS = 150
CHANNELS = ("left", "right")


def probabilities(session, track):
    # The sequence model takes one row per 32 ms frame with preceding context.
    total = int(np.ceil(track.size / FRAME)) * FRAME
    padded = np.pad(track.astype(np.float32) / 32768, (CONTEXT, total - track.size))
    blocks = np.lib.stride_tricks.sliding_window_view(padded, FRAME + CONTEXT)[::FRAME]
    initial = np.zeros((1, 1, 128), dtype=np.float32)
    return session.run(["speech_probs"], {"input": np.ascontiguousarray(blocks), "h": initial, "c": initial})[0]


def speech_bouts(scores, duration_ms):
    """Use hysteresis and keep short audible replies and acknowledgments."""
    start = end = None
    bouts = []
    for index, score in enumerate(scores):
        time = index * FRAME_MS
        if score >= 0.50:
            if start is None:
                start = time
            end = None
        elif score < 0.35 and start is not None:
            if end is None:
                end = time
            if time - end >= 100:
                if end - start >= MIN_BOUT_MS:
                    bouts.append((start, end))
                start = end = None
    if start is not None and duration_ms - start >= MIN_BOUT_MS:
        bouts.append((start, duration_ms))
    return bouts


def shared_bouts(activity):
    """Union active tracks and retain the channel edges of each union."""
    unions = []
    for start, end, channel in sorted((start, end, channel)
                                      for channel, spans in enumerate(activity) for start, end in spans):
        if not unions or start > unions[-1][1]:
            unions.append([start, end, channel, {channel: end}])
        else:
            block = unions[-1]
            block[1] = max(block[1], end)
            if start == block[0] and channel != block[2]:
                block[2] = None
            block[3][channel] = max(block[3].get(channel, 0), end)
    return unions


def suggest(recording, tracks, activity):
    proposals = []
    counts = Counter()
    blocks = shared_bouts(activity)
    for (_, stop, _, ends), (onset, _, next_channel, _) in zip(blocks, blocks[1:]):
        if onset - stop < MIN_GAP_MS:
            continue
        counts["shared_gaps"] += 1
        focal = max(ends, key=ends.get)
        if len(ends) == 2 and stop - ends[1 - focal] <= OVERLAP_MARGIN_MS:
            counts["overlap_end_suppressed"] += 1
            continue
        if next_channel is None:
            counts["ambiguous_next_onset_suppressed"] += 1
            continue
        tail = tracks[:, max(0, stop - 200) * RATE // 1000:stop * RATE // 1000].astype(np.int32)
        own, other = np.max(np.abs(tail), axis=1)[[focal, 1 - focal]]
        if own < max(128, 1.414 * other):
            counts["crosstalk_suppressed"] += 1
            continue
        key = f"{recording}:{CHANNELS[focal]}:{stop}"
        proposals.append({
            "id": hashlib.sha256(key.encode()).hexdigest()[:16],
            "recording": recording,
            "focal_channel": CHANNELS[focal],
            "stop_ms": stop,
            "next_onset_ms": onset,
            "next_channel": CHANNELS[next_channel],
            "short_history": stop < 3000,
        })
    return proposals, counts


def run(bundle, model):
    import onnxruntime as ort

    if file_digest(model) != MODEL_SHA256:
        raise ValueError("Unexpected Silero sequence model checksum")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    pilot = bundle / "hindi-review-a"
    packet = json.loads((pilot / "packet.json").read_text())
    checksums = dict(line.split(maxsplit=1)[::-1] for line in (pilot / "SHA256SUMS").read_text().splitlines())
    suggestions = []
    recordings = {}
    for recording in packet["windows"]:
        if not re.fullmatch(r"call-[0-9a-f]{16}", recording) or recording in recordings:
            raise ValueError("Invalid or duplicate review recording")
        item = f"audio/{recording}.wav"
        path = pilot / item
        digest = file_digest(path)
        if checksums.get(item) != digest:
            raise ValueError(f"Review audio checksum differs for {recording}")
        with wave.open(str(path)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (2, 2, RATE):
                raise ValueError("Expected stereo 16 kHz PCM")
            duration = wav.getnframes() * 1000 // RATE
            tracks = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, 2).T
        activity = [speech_bouts(probabilities(session, track), duration) for track in tracks]
        detected, counts = suggest(recording, tracks, activity)
        suggestions.extend(detected)
        recordings[recording] = {"audio_sha256": digest, "duration_ms": duration,
                                 "proposals": len(detected), **counts}
    receipt = {
        "schema": 1, "mode": "assisted",
        "detector": {"name": "silero-vad", "version": "6.2.3", "onnx_sha256": MODEL_SHA256,
                     "frame_ms": FRAME_MS, "speech_on": 0.50, "speech_off": 0.35, "speech_pad_ms": 0,
                     "vad_min_silence_ms": 100, "vad_min_bout_ms": MIN_BOUT_MS, "min_shared_gap_ms": MIN_GAP_MS,
                     "overlap_margin_ms": OVERLAP_MARGIN_MS, "dominance_factor": 1.414, "min_focal_peak": 128},
        "recordings": recordings, "proposals": suggestions,
    }
    _write_json(pilot / "suggested-pauses.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/review/paired-call-floor-review"))
    parser.add_argument("--model", type=Path, default=Path("data/review/.models/silero_vad_16k_sequence-v6.2.3.onnx"))
    args = parser.parse_args()
    receipt = run(args.bundle, args.model)
    print(json.dumps({"proposals": len(receipt["proposals"]), "recordings": receipt["recordings"]}, indent=2))


if __name__ == "__main__":
    main()
