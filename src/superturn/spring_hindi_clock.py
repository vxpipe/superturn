"""Diagnose provider interval timing on the pinned SPRING-INX WAV windows."""

import argparse
from collections import Counter
import json

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, read_audio, silero
from .dense_teacher import file_digest
from .dral_label_pilot import summarize
from .harper_label_yield import one_to_one
from .labels import Speech
from .progress import Progress
from .sources import data_root
from .spring_hindi_pilot import SOURCE, intervals

FRAME = RATE // 50
SEARCH = 100
SIDES = ("Left", "Right")


def occupancy(spans, frames):
    centres = np.arange(frames // FRAME) * FRAME + FRAME // 2
    mask = np.zeros(len(centres), dtype=bool)
    for start, stop in spans:
        mask |= (centres >= start) & (centres < stop)
    return mask


def scan(reference, observed, lo, hi):
    """Positive lag moves publisher boundaries later."""
    extrema = np.arange(lo, hi)
    scores = {}
    for lag in range(-SEARCH, SEARCH + 1):
        a, b = reference[extrema - lag], observed[extrema]
        union = (a | b).sum()
        scores[lag] = float((a & b).sum() / union) if union else 0.
    best = max(scores, key=lambda lag: (scores[lag], -abs(lag), -lag))
    return {"zero_iou": scores[0], "peak_iou": scores[best], "lag_frames": best,
            "held_lag_scores": {str(lag): value for lag, value in scores.items()}}


def boundary_stats(provider, detected, end, shift):
    available = lambda items, offset: np.asarray(sorted(pair[end] + offset for pair in items
                                                        if FRAME <= pair[end] + offset))
    p, d = available(provider, shift), available(detected, 0)
    offsets = []
    if len(p) and len(d):
        separation = abs(p[:, None] - d)
        for i, j in enumerate(separation.argmin(axis=1)):
            if separation[:, j].argmin() == i and separation[i, j] <= RATE:
                offsets.append(float((d[j] - p[i]) * 1000 / RATE))
    return {"publisher": len(p), "detected": len(d), "paired": len(offsets),
            "within_160ms": sum(abs(x) <= 160 for x in offsets),
            "signed_ms_p25_p50_p75": np.percentile(offsets, (25, 50, 75)).tolist() if offsets else None}


def analyze(tile, provider, audio, detector):
    start, length = tile["start_native_sample"], tile["frames"]
    publisher = {side: [(x.start - start, x.end - start) for x in provider if x.speaker == side
                        and x.start < start + length and x.end > start] for side in SIDES}
    observed = {side: detector(track) for side, track in zip(SIDES, audio, strict=True)}
    vb, _ = summarize([Speech(side, lo, hi) for side in SIDES for lo, hi in observed[side]], length)
    pb = [(side, edge - start, onset - start, label) for side, edge, onset, label in tile["segment_candidates"]]
    near = Counter()
    tolerance = 160 * RATE // 1000
    for side, end, onset, label in vb:
        edges = [row for row in pb if side == row[0] and abs(end - row[1]) <= tolerance]
        onsets = [row for row in pb if side == row[0] and abs(onset - row[2]) <= tolerance]
        both = [row for row in edges if row in onsets]
        near.update({"edge_within_tolerance": bool(edges), "onset_within_tolerance": bool(onsets),
                     "both_within_tolerance": bool(both), "multiple_both": len(both) > 1,
                     "next_actor_disagreement": any(row[3] != label for row in both)})
    diagnostics = {}
    frames = length // FRAME
    for i, side in enumerate(SIDES):
        a = occupancy(publisher[side], length)
        b = occupancy(observed[side], length)
        c = occupancy(observed[SIDES[1 - i]], length)
        full = scan(a, b, SEARCH, frames - SEARCH)
        swapped = scan(a, c, SEARCH, frames - SEARCH)
        early = scan(a, b, SEARCH, frames // 2)
        late = scan(a, b, frames // 2, frames - SEARCH)
        shift = full["lag_frames"] * FRAME
        interiors = [(lo, hi) for lo, hi in publisher[side] if FRAME <= lo < hi <= length - FRAME]
        usable = [(lo, hi) for lo, hi in observed[side] if FRAME <= lo < hi <= length - FRAME]
        diagnostics[side] = {
            "publisher_core_speech_frames": int(a[SEARCH:-SEARCH].sum()),
            "observed_core_speech_frames": int(b[SEARCH:-SEARCH].sum()),
            "same_side": {key: value for key, value in full.items() if key != "held_lag_scores"},
            "swapped_side": {key: value for key, value in swapped.items() if key != "held_lag_scores"},
            "early_lag_ms": early["lag_frames"] * 20, "late_lag_ms": late["lag_frames"] * 20,
            "early_zero_iou": early["zero_iou"], "late_zero_iou": late["zero_iou"],
            "early_speech_frames": [int(mask[SEARCH:frames // 2].sum()) for mask in (a, b)],
            "late_speech_frames": [int(mask[frames // 2:frames - SEARCH].sum()) for mask in (a, b)],
            "early_lag_late_iou": late["held_lag_scores"][str(early["lag_frames"])],
            "late_lag_early_iou": early["held_lag_scores"][str(late["lag_frames"])],
            "start": boundary_stats(interiors, usable, 0, 0),
            "end": boundary_stats(interiors, usable, 1, 0),
            "start_shifted_diagnostic": boundary_stats(interiors, usable, 0, shift),
            "end_shifted_diagnostic": boundary_stats(interiors, usable, 1, shift),
        }
    agreement = Counter()
    one_to_one(vb, pb, agreement)
    return {"stages": near, "original_match": agreement, "sides": diagnostics}


def read_inputs(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("SPRING audio stays on Vast")
    plan_path = root / "reports/spring-hindi-paired-plan.json"
    audio_path = root / "reports/spring-hindi-paired-audio.json"
    plan = json.loads(plan_path.read_text())
    audio = json.loads(audio_path.read_text())
    if file_digest(plan_path) != audio["plan_sha256"]:
        raise ValueError("the audio timing receipt changed")
    folder = root / "raw" / SOURCE
    for item in plan["metadata"].values():
        if file_digest(folder / item["path"]) != item["sha256"]:
            raise ValueError("the provider timing table changed")
    train, _ = intervals(folder, "train")
    return plan, audio, folder, train, file_digest(audio_path)


def load_tracks(folder, receipt, tile):
    call, start = tile["recording"], tile["start_native_sample"]
    paths = [f"windows/{call}/{start}-{side}.wav" for side in SIDES]
    if any(file_digest(folder / key) != receipt["files"][key] for key in paths):
        raise ValueError("a window differs from its pinned source bytes")
    waveforms = [read_audio(folder / path) for path in paths]
    if any(len(wav) != tile["frames"] for wav in waveforms):
        raise ValueError("length disagrees with the native-time plan")
    return np.stack(waveforms)


def run(root):
    plan, audio, folder, train, audio_digest = read_inputs(root)
    detect = silero()
    result = []
    meter = Progress("spring-hindi-clock", len(plan["selection"]), "windows")
    try:
        for tile in plan["selection"]:
            call, start = tile["recording"], tile["start_native_sample"]
            waveforms = load_tracks(folder, audio, tile)
            result.append({"call": call, "native_start_sample": start,
                           **analyze(tile, train[call], waveforms, detect)})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    document = {"scope": "within-channel timing diagnostics, no labels, six 30-second windows",
                "frame_ms": 20, "lag_limit_ms": SEARCH * 20, "audio_sha256": audio_digest,
                "windows": result}
    _write_json(root / "reports/spring-hindi-clock.json", document)


def publisher_side_evidence(pauses, spans, start):
    def window(side, lo, hi):
        return any(span.speaker == SIDES[side] and min(span.end, hi) - max(span.start, lo) >= RATE // 10
                   for span in spans)

    for row in pauses:
        side = row["side"]
        next_side = side if row["label"] == "hold" else 1 - side
        edge, onset = row["edge"] + start, row["onset"] + start
        row["native_edge"], row["native_next_onset"] = edge, onset
        row["publisher_focal_side"] = window(side, edge - RATE // 5, edge)
        row["publisher_next_side"] = window(next_side, onset, onset + RATE // 5)
        row["publisher_competing_next_side"] = window(1 - next_side, onset, onset + RATE // 5)


def word_timing(root):
    import torch
    import whisper

    from .datahive_pause_rescue import inspect
    from .open_speech_teacher_probe import WHISPER_SHA256

    torch.set_num_threads(2)
    plan, audio, folder, train, audio_digest = read_inputs(root)
    clock = root / "reports/spring-hindi-clock.json"
    if json.loads(clock.read_text())["audio_sha256"] != audio_digest:
        raise ValueError("the clock evidence covers different source audio")
    weight = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(weight) != WHISPER_SHA256:
        raise ValueError("the existing Whisper weights changed")
    model = whisper.load_model(str(weight), device="cuda")
    detect = silero()
    windows = []
    meter = Progress("spring-hindi-word-timing", len(plan["selection"]), "windows")
    try:
        for tile in plan["selection"]:
            call, start = tile["recording"], tile["start_native_sample"]
            tracks = load_tracks(folder, audio, tile)
            result = inspect(tracks, detect, model, "hi")
            publisher_side_evidence(result["provisional_pauses"], train[call], start)
            windows.append({"call": call, "native_start_sample": start, **result})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"scope": "strict word and acoustic timing, pilot only", "clock_sha256": file_digest(clock),
              "audio_sha256": audio_digest, "whisper_sha256": WHISPER_SHA256, "windows": windows}
    _write_json(root / "reports/spring-hindi-word-timing.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-timing", action="store_true")
    args = parser.parse_args()
    (word_timing if args.word_timing else run)(data_root())
