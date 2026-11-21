"""Deterministic, blind listening packets and judgment summaries."""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import RATE, read_audio, write_wave
from .labels import Label
from .progress import Progress
from .shards import seed_for


def choose(rows, per_language):
    groups = defaultdict(list)
    for row in rows:
        groups[row.language].append(row)
    for language, examples in sorted(groups.items()):
        ranked = sorted(examples, key=lambda row: seed_for(
            "listen-v1", language, row.source, row.recording, row.audio, row.speaker, row.pause_start,
        ))
        for index, row in enumerate(ranked[:per_language], 1):
            yield f"{language}-{index:03d}", row


def _excerpt(root, label):
    path = root / label.audio
    caller = read_audio(path)
    before = max(0, label.pause_start - 4 * RATE)
    edge = label.pause_start
    after = min(len(caller), edge + max(4 * RATE, label.pause_samples + 2 * RATE))
    if label.silence_start is not None:
        tail = caller[label.silence_start:label.silence_start + min(label.pause_samples, RATE)]
        return np.concatenate((caller[before:edge], tail, caller[edge:after]))
    if label.source == "ami":
        # Let the reviewer hear the other side of the exchange after the
        # caller's pause. Keep the headset mix below full scale.
        others = [read_audio(track)[before:after].astype(np.float32)
                  for track in sorted(path.parent.glob("*.Headset-*.wav")) if track != path]
        if others:
            other_pcm = np.stack(others).mean(axis=0)
            mix = (caller[before:after].astype(np.float32) + 0.7 * other_pcm) / 1.7
            return mix.astype("<i2")
    return caller[before:after]


def prepare(root, rows, *, per_language=50):
    packet = root / "labels" / "listening"
    packet.mkdir(parents=True, exist_ok=True)
    selected = dict(choose(rows, per_language))
    if not selected:
        raise ValueError("no labels to review")
    completed, template = packet / "judgments.csv", packet / "review.csv"
    if completed.is_file():
        raise ValueError("existing review judgments must be preserved before regenerating the packet")
    if template.is_file():
        with template.open(newline="") as handle:
            if any(row["heard"].strip() for row in csv.DictReader(handle)):
                raise ValueError("existing partial review judgments must be preserved before regenerating the packet")
    meter = Progress("listening-packet", len(selected), "examples")
    try:
        for identifier, label in selected.items():
            write_wave(packet / "audio" / f"{identifier}.wav", _excerpt(root, label))
            meter.advance(1)
        (packet / "key.json").write_text(json.dumps({key: asdict(row) for key, row in selected.items()}, indent=2) + "\n")
        with (packet / "review.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["example", "pause_at_ms", "heard", "notes"])
            writer.writerows((identifier, min(label.pause_start, 4 * RATE) // 16, "", "")
                             for identifier, label in selected.items())
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()


def summarize(root, judgments):
    packet = root / "labels" / "listening"
    key = {name: Label(**row) for name, row in json.loads((packet / "key.json").read_text()).items()}
    results = defaultdict(Counter)
    seen = set()
    with Path(judgments).open(newline="") as handle:
        for row in csv.DictReader(handle):
            identifier, heard = row["example"], row["heard"].strip().lower()
            if identifier not in key or identifier in seen:
                raise ValueError(f"unknown or duplicated example ID {identifier!r}")
            if heard not in {"hold", "shift", "unclear"}:
                raise ValueError(f"{identifier}: heard must be hold, shift, or unclear")
            seen.add(identifier)
            label = key[identifier]
            expected = "shift" if label.label == "weak-shift" else label.label
            for group in (("language", label.language), ("method", label.method)):
                results[group]["reviewed"] += 1
                results[group]["correct"] += heard == expected
                results[group]["unclear"] += heard == "unclear"
    missing = set(key) - seen
    if missing:
        raise ValueError(f"{len(missing)} listening examples lack judgments")
    report = root / "reports" / "listening.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps([
        {kind: name, **counts, "accuracy": counts["correct"] / counts["reviewed"]}
        for (kind, name), counts in sorted(results.items())
    ], indent=2) + "\n")
    (packet / "judgments.csv").write_bytes(Path(judgments).read_bytes())
    return report



def require_gate(root, per_language=50):
    path = root / "reports" / "listening.json"
    if not path.is_file():
        raise ValueError("record the listening judgments before starting campaign training")
    rows = json.loads(path.read_text())
    short = [row["language"] for row in rows if "language" in row and row["reviewed"] < per_language]
    if short:
        raise ValueError(f"listening review is below {per_language} for languages {', '.join(short)}")
    per_track = next((row for row in rows if row.get("method") == "per-track"), None)
    if per_track is None or per_track["accuracy"] < 0.9:
        raise ValueError("listening review requires at least 90% correct per-track labels")
