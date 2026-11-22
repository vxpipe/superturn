"""Check the proposed 256-point spectrum against the current pitch feature."""

import argparse
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .augment import condition
from .model import FFT, FRAME, HOP, AcousticFeatures, parameter_count
from .pitch_check import _sample, yin_pitch
from .progress import Progress
from .sources import data_root
from .train import _input

JOB = "compact-frontend-preflight"
SHORT = 256
WINDOWS = 32
LAG_START, LAG_STOP = 40, 230


class ShortPitch(AcousticFeatures):
    """Diagnostic only. Align 16 ms windows to the existing 25 ms centers."""

    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(SHORT)
        self.bins = torch.arange(SHORT // 4 + 1, dtype=torch.float32)
        correction = torch.stack([(self.window * self.window.roll(lag)).sum() for lag in range(LAG_START, LAG_STOP)])
        self.lag_window = correction / self.window.square().sum()

    def power_spectrum(self, audio):
        trim = (FRAME - SHORT) // 2
        centered = audio - audio.mean(dim=1, keepdim=True)
        spectrum = torch.stft(centered[:, trim:-trim], n_fft=SHORT, hop_length=HOP,
                              window=self.window, center=False, return_complex=True)
        return spectrum[:, :len(self.bins)].abs().square().transpose(1, 2)

    def pitch_features(self, power):
        # Match the existing 62.5 Hz low-frequency exclusion at 256 points.
        power = power[..., 1:]
        harmonics = (self.bins[1:, None] * self.lags * (2 * torch.pi / SHORT)).cos()
        strength = power.sum(dim=-1, keepdim=True)
        correlation = (power @ harmonics) / strength.clamp_min(1e-8)
        corrected = (correlation / self.lag_window).clamp(-1, 1)
        lag_index = (corrected - self.lags * .0005).argmax(dim=-1, keepdim=True)
        pitch = 16000 / (lag_index.to(power.dtype) + LAG_START)
        raw_peak = correlation.gather(-1, lag_index)
        floor = strength.amax(dim=1, keepdim=True) * .0003
        reliable = (strength >= floor.clamp_min(.01)) & (raw_peak >= .18)
        confidence = corrected.gather(-1, lag_index).clamp(0, 1)
        return pitch, confidence * reliable.to(power.dtype)


def summarize(expected, predicted, voiced, mask):
    refs = mask & np.isfinite(expected)
    accepted = refs & voiced
    errors = np.abs(predicted[accepted] / expected[accepted] - 1)
    return {
        "frames": int(mask.sum()), "reference_voiced": int(refs.sum()),
        "matched_voiced": int(accepted.sum()),
        "voiced_recall": float(accepted.sum() / refs.sum()) if refs.any() else None,
        "relative_error_p50": float(np.median(errors)) if len(errors) else None,
        "relative_error_over_20pct": float(np.mean(errors > .20)) if len(errors) else None,
    }


def get_pitch(frontend, audio):
    with torch.inference_mode():
        hz, confidence = frontend.pitch_features(frontend.power_spectrum(torch.from_numpy(audio).unsqueeze(0)))
    return hz.flatten().numpy(), confidence.flatten().numpy() >= .35


def recurrence_cost(width):
    samples = 48000
    frames = (samples - FRAME) // HOP + 1
    steps = (frames + 3) // 4
    return 2 * steps * 3 * (width * width + width * width)


def estimates():
    widths = {size: min(range(32, 433, 4), key=lambda w: abs(parameter_count("a", w, 3) - size))
              for size in (972795, 1500000, 1900000)}
    return {str(size): {"width": width, "parameters": parameter_count("a", width, 3),
                        "recurrent_matrix_macs": recurrence_cost(width)}
            for size, width in widths.items()}


def run(root, *, half_rate=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real validation audio stays on Vast")
    torch.set_num_threads(1)
    models = {"current_512": AcousticFeatures(), "proposed_256": ShortPitch()}
    if half_rate:
        models["half_rate_256"] = AcousticFeatures(spectral_rate=8000)
    rows = _sample(root, WINDOWS)
    if len(rows) < WINDOWS:
        raise ValueError("insufficient archived validation clips")
    targets, outputs = [], {name: [] for name in models}
    receipt = hashlib.sha256()
    meter = Progress(JOB, len(rows), "clips")
    try:
        for row in rows:
            pcm = condition(_input(row, 16000, 300), "telephone", 0)
            receipt.update(pcm.astype("<i2").tobytes())
            audio = pcm.astype(np.float32) / 32768
            frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME)[::HOP]
            targets.extend(yin_pitch(frame) for frame in frames)
            for name, model in models.items():
                hz, voiced = get_pitch(model, audio)
                if len(hz) != len(frames):
                    raise ValueError("short and current frame centers differ")
                outputs[name].append((hz, voiced))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    targets = np.asarray(targets)
    t = np.arange(16000, dtype=np.float32) / 16000
    results = {}
    for name, model in models.items():
        hz, voiced = (np.concatenate(tuple(values)) for values in zip(*outputs[name], strict=True))
        bands = {"all": np.ones(len(targets), dtype=bool), "below_125hz": targets < 125,
                 "at_least_125hz": targets >= 125}
        tones = {}
        for f0 in (75, 80, 100, 120, 160, 210, 280):
            wave = np.sin(2 * np.pi * f0 * t).astype(np.float32) * .1
            answer, active = get_pitch(model, wave)
            tones[f0] = {"median_hz": float(np.median(answer)), "voiced_coverage": float(active.mean())}
        results[name] = {"yin": {band: summarize(targets, hz, voiced, mask) for band, mask in bands.items()},
                         "harmonic_free_tones": tones}
    report = {
        "scope": "signal preflight only; no training, latency or quality benchmark",
        "sample_geometry": {"phone_validation_clips": len(rows), "pcm_sha256": receipt.hexdigest(),
                            "hop_samples": HOP, "current_frame": FRAME, "current_fft": FFT,
                            "proposed_frame": SHORT, "proposed_fft": SHORT,
                            "lag_support": [LAG_START, LAG_STOP - 1]},
        "feature_evidence": results,
        "capacity_estimates": estimates(),
    }
    if half_rate:
        half = models["half_rate_256"]
        with torch.inference_mode():
            tones = torch.from_numpy(np.stack([np.sin(2 * np.pi * freq * t) for freq in (3000, 5000)]))
            power = half.power_spectrum(tones)[:, 2:-2].sum(dim=-1).mean(dim=1)
        report["half_rate_geometry"] = {
            "input_rate": 16000, "analysis_rate": 8000, "filter_taps": half.antialias.numel(),
            "lag_support_8khz": [LAG_START // 2, LAG_STOP // 2 - 1],
            "5000hz_to_3000hz_power_db": float(10 * (power[1] / power[0]).log10()),
        }
    name = f"{JOB}-half-rate" if half_rate else JOB
    _write_json(root / "reports" / f"{name}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-rate", action="store_true")
    run(data_root(), half_rate=parser.parse_args().half_rate)
