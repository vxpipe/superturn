"""Capture causal, lagged Whisper partials on pinned HarperValley pauses."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import wave

import numpy as np
import torch

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import file_digest
from .eot_domain_probe import GAP
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .transcript import pack_partial_transcript
from .turn1_mini_probe import upsample_prefix

HOP = RATE // 5
HISTORY = 6 * RATE
LAGS = (150, 300, 500)
TINY_SHA256 = "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9"


def pinned_membership(root, role):
    reports = root / "reports"
    cases, _, audit = pinned_split(root, role)
    if role == "train":
        manifest = json.loads((reports / "harper-history-replication-membership.json").read_text())
        archived = json.loads((reports / "harper-history-replication.json").read_text())
        digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        if digest != MEMBERSHIP or archived["membership_sha256"] != digest or archived["audit_sha256"] != audit:
            raise ValueError("HarperValley training pauses changed")
    else:
        prior = json.loads((reports / "harper-valley-development-stratified.json").read_text())
        rows = json.loads((reports / "harper-valley-development-stratified-provenance.json").read_text())
        grouped = [(row["call"], row["fixed_pauses"]) for row in rows]
        digest = hashlib.sha256(json.dumps(grouped, separators=(",", ":")).encode()).hexdigest()
        if digest != prior["pause_membership_sha256"]:
            raise ValueError("HarperValley validation pauses changed")
        manifest = [(call, edge, label) for call, pauses in grouped for edge, label in pauses]
    expected = {"train": {"hold": 273, "shift": 255}, "validation": {"hold": 40, "shift": 48}}
    if Counter(label for _, _, label in manifest) != expected[role]:
        raise ValueError("HarperValley pause counts changed")
    by_call = {case["id"]: case for case in cases}
    return [(i, call, edge, label, by_call[call]["caller"])
            for i, (call, edge, label) in enumerate(manifest)], digest


def pilot_rows(rows):
    counts = Counter()
    used = set()
    selected = []
    for row in sorted(rows, key=lambda r: seed_for("harper-partial-stt-v1", r[1], r[2])):
        _, _, _, label, caller = row
        if counts[label] < 16 and (caller, label) not in used:
            used.add((caller, label))
            counts[label] += 1
            selected.append(row)
    if counts != {"hold": 16, "shift": 16}:
        raise ValueError("the fixed pilot needs 16 holds and 16 shifts")
    return sorted(selected)


def tick_at(decision, lag):
    """Choose the newest grid snapshot that could be visible at this decision."""
    return max(0, (decision - lag * RATE // 1000) // HOP * HOP)


def run(root, *, full=False, serial_cpu=False):
    import whisper

    if not (root / ".campaign-machine").is_file():
        raise ValueError("the speech replay uses staged Vast recordings")
    checkpoint = Path.home() / ".cache/whisper/tiny.pt"
    if whisper.__version__ != "20250625" or file_digest(checkpoint) != TINY_SHA256:
        raise ValueError("unexpected staged Whisper Tiny checkpoint")
    train, train_digest = pinned_membership(root, "train")
    development, dev_digest = pinned_membership(root, "validation")
    if {row[-1] for row in train} & {row[-1] for row in development}:
        raise ValueError("HarperValley caller identities cross roles")
    scopes = {"train": train, "validation": development} if full else {"train": pilot_rows(train)}
    requests = defaultdict(set)
    for rows in scopes.values():
        for _, call, edge, _, _ in rows:
            requests[call].update(tick_at(edge + GAP, lag) for lag in LAGS)
    total = sum(sum(tick > 0 for tick in ticks) for ticks in requests.values())
    model = whisper.load_model(str(checkpoint), device="cpu" if serial_cpu else "cuda").eval()
    device = model.device.type
    native_folder = root / "raw/harper-valley/data/audio/caller"

    @lru_cache(maxsize=2)
    def native(call):
        with wave.open(str(native_folder / f"{call}.wav"), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (8000, 1, 2):
                raise ValueError("unexpected native caller track")
            return np.frombuffer(wav.readframes(wav.getnframes()), "<i2")

    outputs, elapsed = {}, []
    meter = Progress("harper-partial-stt", total, "snapshots")
    try:
        with torch.inference_mode():
            for call, ticks in sorted(requests.items()):
                for tick in sorted(ticks - {0}):
                    start = max(0, tick - HISTORY)
                    track = native(call)
                    if tick // 2 > len(track):
                        raise ValueError("the caller track does not reach the scheduled snapshot")
                    audio = upsample_prefix(track[start // 2:tick // 2], tick - start)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    decoded = model.transcribe(audio.astype(np.float32) / 32768, language="en",
                                               fp16=device == "cuda", temperature=0, verbose=None,
                                               condition_on_previous_text=False, word_timestamps=True,
                                               no_speech_threshold=.6, logprob_threshold=-1,
                                               compression_ratio_threshold=2.4)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    elapsed.append((time.perf_counter() - started) * 1000)
                    words = [(word["word"], start + round(word["end"] * RATE))
                             for segment in decoded["segments"] for word in segment.get("words", ())]
                    outputs[call, tick] = pack_partial_transcript(decoded["text"]), words
                    meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result, coverage = {}, {}
    for role, rows in scopes.items():
        result[role], counts = [], Counter()
        for index, call, edge, label, _ in rows:
            decision = edge + GAP
            record = {"ordinal": index, "decision_sample": decision, "snapshots": {}}
            for lag in LAGS:
                tick = tick_at(decision, lag)
                packed, words = outputs.get((call, tick), (pack_partial_transcript(None), []))
                publication = tick + lag * RATE // 1000 if tick else None
                if publication is not None and publication > decision:
                    raise ValueError("a future hypothesis reached a decision")
                post_edge = sum(end > edge for _, end in words)
                guarded = packed
                if post_edge:
                    guarded = pack_partial_transcript("".join(word for word, end in words if end <= edge))
                counts[f"{label}/{lag}/rows"] += 1
                counts[f"{label}/{lag}/nonblank"] += any(packed)
                counts[f"{label}/{lag}/nonblank_after_word_time_guard"] += any(guarded)
                counts[f"{label}/{lag}/post_edge_word_timestamp"] += post_edge > 0
                record["snapshots"][str(lag)] = {"audio_sample": tick, "publication_sample": publication,
                                                "text_bytes": packed, "guarded_text_bytes": guarded,
                                                "post_edge_word_timestamps": post_edge}
            result[role].append(record)
        coverage[role] = dict(counts)
    stem = "harper-partial-stt" if full else "harper-partial-stt-pilot"
    destination = root / "runs" / stem
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "snapshots.json"
    _write_json(output, result)
    report = {"scope": "lagged rolling-window ASR replay; lag assumed, GPU decode timing diagnostic only",
              "checkpoint_sha256": TINY_SHA256, "whisper_version": whisper.__version__, "device": device,
              "grid_samples": HOP, "history_samples": HISTORY, "lags_ms": LAGS, "decode_count": total,
              "train_membership_sha256": train_digest, "validation_membership_sha256": dev_digest,
              "snapshot_file": output.relative_to(root).as_posix(), "snapshot_file_sha256": file_digest(output),
              "decode_ms_quantiles": np.quantile(elapsed, [.1, .5, .9]).tolist(), "coverage": coverage}
    _write_json(root / "reports" / f"{stem}.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(data_root(), full=args.full, serial_cpu=args.cpu), indent=2))
