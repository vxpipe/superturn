"""Fixed-length PCM in compressed, split-specific Parquet shards."""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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


def speaker_report(rows, retained):
    speakers = defaultdict(set)
    counts = Counter()
    groups = defaultdict(set)
    split_counts = defaultdict(Counter)
    voice_splits = defaultdict(set)
    for row in rows:
        key = (row.source, row.language)
        counts[key] += 1
        groups[(*key, row.split)].add(row.group)
        if row.speaker and row.speaker != "unknown":
            speakers[key].add(row.speaker)
            voice_splits[row.source, row.speaker].add(row.split)
    overlaps = sum(len(splits) > 1 for splits in voice_splits.values())
    if overlaps:
        raise ValueError(f"{overlaps} speakers appear in multiple splits")
    for row in retained:
        split_counts[row.source, row.language, row.split][row.label] += 1
    return [
        {"source": source, "language": language, "examples": counts[source, language],
         "known_speakers": len(speakers[source, language]),
         "below_50": len(speakers[source, language]) < 50,
         "groups_by_split": {split: len(groups[source, language, split])
                             for split in ("train", "validation", "test")},
         "retained_labels_by_split": {split: dict(split_counts[source, language, split])
                                     for split in ("train", "validation", "test")}}
        for source, language in sorted(counts)
    ]


def require_split_coverage(report, roles):
    sources = defaultdict(set)
    issues = []
    for row in report:
        source = row["source"]
        available = {name for name, count in row["groups_by_split"].items() if count}
        if roles[source] == "test":
            if available != {"test"}:
                issues.append(f"{source}/{row['language']} is a test-only source")
            continue
        sources[source].update(available)
        if sum(row["groups_by_split"].values()) >= 3 and len(available) < 3:
            issues.append(f"{source}/{row['language']} lacks {', '.join(sorted({'train', 'validation', 'test'} - available))}")
    for source, role in roles.items():
        if role.startswith("train-") and len(sources[source]) < 3:
            issues.append(f"{source} needs speaker groups in all three splits")
    if issues:
        raise ValueError("invalid split coverage: " + "; ".join(issues))


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
    report = speaker_report(rows, retained)
    grouped = defaultdict(list)
    for row in sorted(retained, key=lambda row: (row.audio, row.pause_start)):
        grouped[row.split].append(row)
    meter = Progress("shard", sum(len(items) * (3 if split == "test" else 1)
                                  for split, items in grouped.items()), "clips")
    cached = lru_cache(maxsize=8)(read_audio)
    try:
        with (TemporaryDirectory(dir=target.parent) as temp,
              ThreadPoolExecutor(max_workers=6) as renderers):
            staging = target.parent / temp
            fingerprints = {split: hashlib.sha256() for split in grouped}
            for split, items in sorted(grouped.items()):
                for name in CONDITIONS if split == "test" else ("clean",):
                    folder = staging / split / name
                    folder.mkdir(parents=True)
                    for start in range(0, len(items), SHARD_ROWS):
                        records = []
                        batch = items[start:start + SHARD_ROWS]
                        clips = [clip(row, cached(root / row.audio)) for row in batch]
                        if name == "clean":
                            rendered = clips
                        else:
                            rendered = renderers.map(condition, clips, [name] * len(batch), map(_key, batch))
                        for row, pcm in zip(batch, rendered):
                            metadata = asdict(row)
                            pcm_bytes = pcm.tobytes()
                            fingerprints[split].update(json.dumps((name, metadata), sort_keys=True).encode())
                            fingerprints[split].update(pcm_bytes)
                            records.append({**metadata, "condition": name, "pcm": pcm_bytes})
                            meter.advance(1)
                        pq.write_table(pa.Table.from_pylist(records),
                                       folder / f"{start // SHARD_ROWS:06d}.parquet", compression="zstd")
            (staging / "manifest.json").write_text(json.dumps(
                {"test_sha256": fingerprints.get("test", hashlib.sha256()).hexdigest(),
                 "sha256_by_split": {split: digest.hexdigest() for split, digest in fingerprints.items()},
                 "speaker_report": report,
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
