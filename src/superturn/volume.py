"""Controlled gain and line-noise sensitivity of held-out turn decisions."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .augment import _bandlimit, _g711, _limit
from .evaluate import CONVERSATION_SOURCES, _predict_calls, _session
from .progress import Progress
from .shards import CONTEXT, seed_for
from .train import _stats, paths

SPEECH_LEVELS = (-55, -50, -45, -40, -35, -30, -25, -20)
SNR_LEVELS = (-5, 0, 5, 10, 15, 20, 30)
DECISION_MS = 300
LINE_NOISE_LEVEL = -50
SNR_SPEECH_LEVEL = -30
HISTORY = 8000


def _active_rms(pcm):
    frames = pcm.astype("float32").reshape(-1, 320) / 32768
    powers = np.mean(frames ** 2, axis=1)
    count = max(1, len(powers) // 4)
    return float(np.sqrt(np.mean(np.partition(powers, -count)[-count:])))


def _sample(root, limit):
    groups = defaultdict(list)
    serial = 0
    filters = [("source", "in", sorted(CONVERSATION_SOURCES)),
               ("pause_samples", ">=", DECISION_MS * 16), ("label", "!=", "weak-shift")]
    for path in paths(root, "test"):
        columns = ["source", "language", "label", "audio", "pause_start", "pcm"]
        table = pq.read_table(path, columns=columns, filters=filters)
        for row in table.to_pylist():
            key = (row["language"], row["label"])
            seed = seed_for("volume-audit-v1", row["source"], row["audio"], row["pause_start"])
            rank = (-seed, serial)
            serial += 1
            bucket = groups[key]
            if len(bucket) < limit or rank > bucket[0][0]:
                pcm = np.frombuffer(row["pcm"], dtype="<i2")
                reference_rms = _active_rms(pcm[CONTEXT - HISTORY:CONTEXT])
                if reference_rms <= 1e-6:
                    continue
                value = (key, seed, pcm, reference_rms)
                if len(bucket) < limit:
                    heapq.heappush(bucket, (rank, value))
                else:
                    heapq.heapreplace(bucket, (rank, value))
    return [value for key in sorted(groups) for _, value in sorted(groups[key])]


def _render(example, *, length, dbfs, line_noise):
    _, seed, pcm, reference_rms = example
    end = CONTEXT + DECISION_MS * 16
    adjusted = pcm[end - length:end].astype("float32") * (10 ** (dbfs / 20) / reference_rms)
    if line_noise is not None:
        random = np.random.default_rng(seed).normal(size=len(adjusted))
        noise = _bandlimit(random, 300, 3400)
        sigma = np.sqrt(np.mean(noise ** 2))
        adjusted += noise * (32768 * 10 ** (line_noise / 20) / max(sigma, 1e-12))
    return _g711(_limit(adjusted)).astype("float32") / 32768


def _trials():
    for dbfs in SPEECH_LEVELS:
        yield "quiet phone", dbfs, dbfs, None
        yield "phone with -50 dBFS line noise", dbfs, dbfs, LINE_NOISE_LEVEL
    for snr in SNR_LEVELS:
        yield "phone SNR at -30 dBFS speech", snr, SNR_SPEECH_LEVEL, SNR_SPEECH_LEVEL - snr


def _lowest_level(rows):
    for index, row in enumerate(rows):
        if all(other["meets_targets"] for other in rows[index:]):
            return row["tested_db"]
    return None


def benchmark_volume(root, model_path, threshold, *, per_label_language=24):
    if per_label_language < 1:
        raise ValueError("volume sweep needs at least one example per label and language")
    model_path = Path(model_path)
    session = _session(model_path)
    length = session.get_inputs()[0].shape[1]
    cases = _sample(root, per_label_language)
    if not cases:
        raise ValueError("volume sweep found no held-out conversation examples at 300 ms")
    trials = list(_trials())
    meter = Progress(f"volume-{model_path.parent.name}", len(cases) * len(trials), "decisions")
    grouped_levels = defaultdict(list)
    groups = []
    try:
        with ThreadPoolExecutor(max_workers=6) as workers:
            for condition, level, speech_dbfs, line_noise in trials:
                results = defaultdict(list)
                render = partial(_render, length=length, dbfs=speech_dbfs, line_noise=line_noise)
                for start in range(0, len(cases), 16):
                    batch = cases[start:start + 16]
                    pcm = np.stack(list(workers.map(render, batch)))
                    elapsed = np.full((len(batch), 1), DECISION_MS, dtype="float32")
                    for example, score in zip(batch, _predict_calls(session, pcm, elapsed)):
                        (language, label), *_ = example
                        pair = (float(score), int(label == "shift"))
                        for group in ("all", language):
                            results[group].append(pair)
                    meter.advance(len(batch))
                for language, rows in sorted(results.items()):
                    scores, labels = zip(*rows)
                    metrics = _stats(scores, labels, threshold)
                    supported = (metrics["holds"] >= 20 and metrics["shifts"] >= 20
                                 and metrics["false_complete"] <= 0.05 and metrics["recall"] >= 0.8)
                    row = {"condition": condition, "tested_db": level, "speech_dbfs": speech_dbfs,
                           "line_noise_dbfs": line_noise, "language": language,
                           "meets_targets": supported, **metrics}
                    groups.append(row)
                    if language == "all":
                        grouped_levels[condition].append(row)

        floors = {condition: _lowest_level(levels) for condition, levels in grouped_levels.items()}
        report = {
            "decision_ms": DECISION_MS,
            "threshold": threshold,
            "scope": "speaker-disjoint AMI and Indic DiarBench conversation examples",
            "speech_level_definition": "RMS of loudest quarter of 20 ms frames in the last 500 ms before pause",
            "level_measurement": "nominal before G.711 encoding, with fixed band-limited noise when specified",
            "qualifying_lowest_tested_db": floors,
            "examples": len(cases),
            "groups": groups,
        }
        destination = root / "reports" / model_path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "volume.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = ["# Telephone level and SNR sweep", "",
                 f"Fixed decision threshold {threshold:.4f} at {DECISION_MS} ms on {len(cases)} paired held-out pauses.",
                 "Speech and injected noise levels are measured before G.711 encoding.", "",
                 "| Sweep | Lowest tested level meeting both quality targets at this level and all higher levels |",
                 "|---|---:|"]
        for condition, floor in floors.items():
            value = "No tested level" if floor is None else f"{floor} {'dB SNR' if 'SNR' in condition else 'dBFS'}"
            lines.append(f"| {condition} | {value} |")
        lines += ["", "| Sweep | Speech dBFS | Noise dBFS | Holds | Shifts | False complete | Recall |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in groups:
            if row["language"] == "all":
                false_complete = "—" if row["false_complete"] is None else f"{row['false_complete']:.2%}"
                recall = "—" if row["recall"] is None else f"{row['recall']:.2%}"
                lines.append(f"| {row['condition']} | {row['speech_dbfs']} | {row['line_noise_dbfs']} | "
                             f"{row['holds']} | {row['shifts']} | {false_complete} | {recall} |")
        (destination / "volume.md").write_text("\n".join(lines) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report
