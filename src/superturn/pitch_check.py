"""Compare the spectral pitch feature with the YIN difference method on speech."""

import json
import random

import numpy as np
import torch

from .model import FRAME, HOP, AcousticFeatures
from .progress import Progress
from .train import _input, paths, records


def yin_pitch(frame):
    """Return Hz using the first strong YIN minimum, or NaN when unvoiced."""
    max_lag = 229
    width = len(frame) - max_lag
    shifted = np.lib.stride_tricks.sliding_window_view(frame, width)
    differences = np.mean((shifted[1:] - shifted[0]) ** 2, axis=-1)
    normalized = differences * np.arange(1, max_lag + 1) / np.maximum(np.cumsum(differences), 1e-12)
    troughs = np.flatnonzero((normalized[1:-1] <= normalized[:-2]) &
                             (normalized[1:-1] < normalized[2:])) + 1
    valid = troughs[(troughs >= 39) & (normalized[troughs] < 0.15)]
    if not len(valid) or np.mean(frame ** 2) < 1e-6:
        return np.nan
    return 16000 / (valid[0] + 1)


def _sample(root, count):
    rng = random.Random(17)
    results = []
    index = 0
    for rows in records(paths(root, "validation")):
        for row in rows:
            index += 1
            if len(results) < count:
                results.append(row)
            elif (position := rng.randrange(index)) < count:
                results[position] = row
    return results


def check_pitch(root, count=64):
    if count < 1:
        raise ValueError("pitch check needs at least one clip")
    rows = _sample(root, count)
    if not rows:
        raise ValueError("pitch check requires validation audio")
    frontend = AcousticFeatures()
    torch.set_num_threads(1)
    meter = Progress("pitch-reference", len(rows), "clips")
    reference = []
    feature = []
    voiced = []
    try:
        for row in rows:
            audio = _input(row, 16000).astype("float32") / 32768
            frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME)[::HOP]
            reference.extend(yin_pitch(frame) for frame in frames)
            with torch.inference_mode():
                power = frontend.power_spectrum(torch.from_numpy(audio).unsqueeze(0))
                hz, confidence = frontend.pitch_features(power)
            feature.extend(hz.flatten().tolist())
            voiced.extend((confidence.flatten() >= 0.35).tolist())
            meter.advance(1)

        expected, actual, voiced = np.asarray(reference), np.asarray(feature), np.asarray(voiced)
        reference_voiced = np.isfinite(expected)
        jointly_voiced = reference_voiced & voiced
        errors = np.abs(actual[jointly_voiced] - expected[jointly_voiced]) / expected[jointly_voiced]
        report = {
            "reference": "YIN",
            "split": "validation",
            "clips": len(rows),
            "frames": len(reference),
            "reference_voiced": int(reference_voiced.sum()),
            "jointly_voiced": int(jointly_voiced.sum()),
            "voiced_agreement": float(np.mean(reference_voiced == voiced)),
            "median_relative_hz_error": float(np.median(errors)) if len(errors) else None,
            "over_20pct_relative_error": float(np.mean(errors > 0.2)) if len(errors) else None,
        }
        path = root / "reports" / "pitch-reference.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return path
