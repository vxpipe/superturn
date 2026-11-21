"""Fixed-length PCM in compressed, split-specific Parquet shards."""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .audio import RATE, read_audio
from .augment import CONDITIONS, condition
from .progress import Progress

CONTEXT = RATE * 8
TAIL = RATE * 3
SHARD_ROWS = 128


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "big")


def _key(row):
    return seed_for(row.source, row.recording, row.audio, row.speaker, row.pause_start)


def cap_training(rows, cap, disabled=False):
    counts = Counter(row.source for row in rows if row.split == "train")
    per_speaker = Counter()
    for row in sorted(rows, key=_key):
        key = (row.source, row.speaker)
        if row.split == "train":
            if not disabled and per_speaker[key] >= max(1, int(counts[row.source] * cap)):
                continue
            per_speaker[key] += 1
        yield row


def speaker_report(rows):
    speakers = defaultdict(set)
    counts = Counter()
    groups = defaultdict(set)
    for row in rows:
        key = (row.source, row.language)
        counts[key] += 1
        groups[(*key, row.split)].add(row.group)
        if row.speaker and row.speaker != "unknown":
            speakers[key].add(row.speaker)
    return [
        {"source": source, "language": language, "examples": counts[source, language],
         "known_speakers": len(speakers[source, language]),
         "below_50": len(speakers[source, language]) < 50,
         "groups_by_split": {split: len(groups[source, language, split])
                             for split in ("train", "validation", "test")}}
        for source, language in sorted(counts)
    ]


def clip(row, pcm):
    edge = row.pause_start
    tail_start = edge if row.silence_start is None else row.silence_start
    if not (0 <= edge <= tail_start <= len(pcm) and tail_start + row.pause_samples <= len(pcm)):
        raise ValueError(f"{row.audio}: label outside audio")
    before = pcm[max(0, edge - CONTEXT):edge]
    after = pcm[tail_start:tail_start + min(row.pause_samples, TAIL)]
    return np.concatenate((np.zeros(CONTEXT - len(before), "<i2"), before,
                           after, np.zeros(TAIL - len(after), "<i2")))


def make_shards(root, rows, *, sample=False):
    target = root / "shards"
    target.mkdir(parents=True, exist_ok=True)
    retained = list(cap_training(rows, 0.005, disabled=sample))
    if not retained:
        raise ValueError("no labelled pauses to shard")
    grouped = defaultdict(list)
    for row in sorted(retained, key=lambda row: (row.audio, row.pause_start)):
        grouped[row.split].append(row)
    meter = Progress("shard", sum(len(items) * (3 if split == "test" else 1)
                                  for split, items in grouped.items()), "clips")
    cached = lru_cache(maxsize=8)(read_audio)
    try:
        with TemporaryDirectory(dir=target.parent) as temp:
            staging = target.parent / temp
            frozen = hashlib.sha256()
            for split, items in sorted(grouped.items()):
                for name in CONDITIONS if split == "test" else ("clean",):
                    folder = staging / split / name
                    folder.mkdir(parents=True)
                    for start in range(0, len(items), SHARD_ROWS):
                        records = []
                        for row in items[start:start + SHARD_ROWS]:
                            seed = _key(row)
                            pcm = condition(clip(row, cached(root / row.audio)), name, seed)
                            metadata = asdict(row)
                            if split == "test":
                                frozen.update(json.dumps((name, metadata), sort_keys=True).encode())
                                frozen.update(pcm.tobytes())
                            records.append({**metadata, "condition": name, "pcm": pcm.tobytes()})
                            meter.advance(1)
                        pq.write_table(pa.Table.from_pylist(records),
                                       folder / f"{start // SHARD_ROWS:06d}.parquet", compression="zstd")
            (staging / "manifest.json").write_text(json.dumps(
                {"test_sha256": frozen.hexdigest(), "speaker_report": speaker_report(rows),
                 "examples_by_split": {name: len(items) for name, items in grouped.items()}}, indent=2) + "\n")
            for entry in staging.iterdir():
                destination = target / entry.name
                if destination.is_dir():
                    shutil.rmtree(destination)
                entry.replace(destination)
            for split in ("train", "validation", "test"):
                if split not in grouped and (target / split).is_dir():
                    shutil.rmtree(target / split)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
