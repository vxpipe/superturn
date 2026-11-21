"""Train and select checkpoints on validation data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from .augment import augment
from .model import build
from .progress import Progress
from .shards import CONTEXT, seed_for

PAUSES = (200, 300, 500)


def paths(root, split, condition="clean"):
    return sorted((root / "shards" / split / condition).glob("*.parquet"))


def records(files, batch_size=16):
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield batch.to_pylist()


def eligible(row, config):
    if row["label"] == "weak-shift" and not config.get("weak", True):
        return False
    if config.get("sources") and row["source"] not in config["sources"]:
        return False
    fraction = config.get("fraction", 1)
    return seed_for(row["source"], row["audio"], row["pause_start"]) % 10000 < 10000 * fraction


def _input(row, samples, silence_ms=0):
    elapsed = round(silence_ms * 16)
    if not 0 <= elapsed <= row["pause_samples"]:
        raise ValueError(f"decision silence exceeds the labelled pause: {silence_ms} ms")
    end = CONTEXT + elapsed
    return np.frombuffer(row["pcm"], "<i2")[end - samples:end]


def _auc(scores, truth):
    pairs = sorted(zip(scores, truth))
    positive = sum(truth)
    negative = len(truth) - positive
    if not positive or not negative:
        return None
    rank_sum = 0.0
    from itertools import groupby
    rank = 1
    for _, tied in groupby(pairs, key=lambda pair: pair[0]):
        labels = [label for _, label in tied]
        rank_sum += sum(labels) * (rank + (len(labels) - 1) / 2)
        rank += len(labels)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def _threshold(scores, truth, target=0.05):
    holds = sorted((score for score, label in zip(scores, truth) if not label), reverse=True)
    if not holds:
        return 0.5
    boundary = holds[min(len(holds) - 1, int(target * len(holds)))]
    return float(np.nextafter(np.float32(boundary), np.float32(np.inf)))


def _stats(scores, truth, threshold):
    holds = sum(label == 0 for label in truth)
    shifts = sum(truth)
    return {
        "holds": holds, "shifts": shifts,
        "false_complete": (sum(not label and score >= threshold
                               for score, label in zip(scores, truth)) / holds if holds else None),
        "recall": (sum(label and score >= threshold for score, label in zip(scores, truth))
                   / shifts if shifts else None),
        "auc": _auc(scores, truth),
    }


def _validation(model, files, device):
    scores, truth = [], []
    model.eval()
    with torch.inference_mode():
        for rows in records(files):
            rows = [row for row in rows if row["pause_samples"] >= 4800 and row["label"] != "weak-shift"]
            if rows:
                audio = np.stack([_input(row, model.samples, 300) for row in rows]).astype(np.float32) / 32768
                probability = model(torch.from_numpy(audio).to(device),
                                    torch.full((len(rows), 1), 300.0, device=device))
                scores.extend(probability.cpu().flatten().tolist())
                truth.extend(int(row["label"] == "shift") for row in rows)
    threshold = _threshold(scores, truth)
    return _stats(scores, truth, threshold), threshold


def _assets(root, samples):
    from zipfile import ZipFile
    from .audio import read_audio, write_wave

    noise = sorted((root / "raw" / "musan" / "musan" / "noise").rglob("*.wav"))
    ir_root = root / "labels" / "impulses"
    if not ir_root.exists():
        for path in (root / "raw" / "openslr26").glob("*.zip"):
            with ZipFile(path) as archive:
                entries = sorted(n for n in archive.namelist() if n.lower().endswith(".wav"))
                for index, name in enumerate(entries[::max(1, len(entries) // 256)]):
                    write_wave(ir_root / f"{index:04d}.wav", read_audio(archive.read(name)))
    voices = {}
    for batch in records(paths(root, "train")):
        for row in batch:
            key = (row["source"], row["speaker"])
            if key not in voices:
                voices[key] = _input(row, samples)
        if len(voices) >= 96:
            break
    return noise, sorted(ir_root.glob("*.wav")), voices


def train(root, name, config, device):
    train_files = paths(root, "train")
    valid_files = paths(root, "validation")
    if not train_files or (not valid_files and not config.get("smoke")):
        raise ValueError("training requires train and validation shards")
    seed = config.get("seed", 17)
    torch.manual_seed(seed)
    if device == "cpu":
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.001))
    run = root / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    train_batches = 0
    for batch in records(train_files, config.get("batch_size", 16)):
        selection = [row for row in batch if eligible(row, config)]
        counts.update(row["language"] for row in selection)
        train_batches += bool(selection)
    if not train_batches:
        raise ValueError("no training examples match the run configuration")

    language_normalizer = sum(count ** 0.7 for count in counts.values()) / sum(counts.values())
    augmentation = config.get("augmentation", True)
    noise, impulse, voices = _assets(root, model.samples) if augmentation else ([], [], {})
    epochs = config.get("epochs", 5)
    steps = min(config.get("max_steps", train_batches * epochs), train_batches * epochs)
    meter = Progress(f"train-{name}", steps, "steps")
    metrics = []
    best = None
    completed = 0
    try:
        for epoch in range(epochs):
            model.train()
            rng = np.random.default_rng(seed_for(seed, epoch))
            loss_total, batches = 0.0, 0
            for batch in records(rng.permutation(train_files), config.get("batch_size", 16)):
                selected = [row for row in batch if eligible(row, config)]
                if not selected:
                    continue
                audio, gaps, truth, weights = [], [], [], []
                for index in rng.permutation(len(selected)):
                    row = selected[index]
                    gap_ms = rng.choice([v for v in PAUSES if v * 16 <= row["pause_samples"]])
                    pcm = _input(row, model.samples, gap_ms)
                    if augmentation:
                        other = [value for key, value in voices.items() if key != (row["source"], row["speaker"])]
                        pcm = augment(pcm, int(rng.integers(2**63)), noise_paths=noise,
                                      ir_paths=impulse, voices=other, pause_samples=gap_ms * 16)
                    audio.append(pcm)
                    # Equalize class exposure to elapsed silence within each available gap.
                    gaps.append([gap_ms])
                    positive = row["label"] != "hold"
                    truth.append([float(positive)])
                    weight = counts[row["language"]] ** -0.3 / language_normalizer
                    weight *= 0.35 if row["label"] == "weak-shift" else 1
                    weight *= 1 if positive else config.get("hold_weight", 2.0)
                    weights.append([weight])
                audio = np.asarray(audio, dtype=np.float32) / 32768
                inputs = torch.from_numpy(audio).to(device)
                gap = torch.as_tensor(np.asarray(gaps), dtype=torch.float32, device=device)
                targets = torch.tensor(truth, device=device)
                weights = torch.tensor(weights, device=device)
                optimizer.zero_grad()
                losses = F.binary_cross_entropy_with_logits(model.logits(inputs, gap), targets, reduction="none")
                loss = (losses * weights).mean()
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
                batches += 1
                completed += 1
                meter.advance(1)
                if completed >= steps:
                    break
            validation, threshold = _validation(model, valid_files, device)
            if not config.get("smoke") and validation["auc"] is None:
                raise ValueError("validation requires both 300 ms holds and shifts")
            record = {"epoch": epoch + 1, "loss": loss_total / batches, "threshold": threshold, **validation}
            metrics.append(record)
            print(json.dumps(record), flush=True)
            score = ((record["recall"] or 0), (record["auc"] or 0), -record["loss"])
            if best is None or score > best:
                best = score
                checkpoint = {"config": config, "state": model.state_dict(), "metrics": record,
                              "parameters": sum(p.numel() for p in model.parameters())}
                temp = run / "best.tmp"
                torch.save(checkpoint, temp)
                temp.replace(run / "best.pt")
            (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if completed >= steps:
                break
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return run / "best.pt"
