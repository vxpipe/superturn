"""Recover the verified 300 ms caller PCM for own-teacher distillation."""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .audio import RATE, read_audio
from .dense_teacher import SAMPLES, file_digest, render
from .labels import Label
from .own_head_distillation import DEADLINE, pool_rows
from .progress import Progress
from .shards import SHARD_ROWS
from .sources import data_root

RENDERS_PER_ROLE_SOURCE = 5


def source_state(root):
    folder, pool_hash, rows = pool_rows(root)
    reports = root / "reports"
    rendered_path = reports / "dense-teacher-targets.json"
    verified_path = reports / "own-speech-targets.json"
    rendered = json.loads(rendered_path.read_text())
    verified = json.loads(verified_path.read_text())
    ordinals = [ordinal for ordinal, _ in rows]
    eligible_hash = hashlib.sha256(json.dumps(ordinals).encode()).hexdigest()
    if (rendered["preview"] or rendered["pool_sha256"] != pool_hash
            or verified["pool_sha256"] != pool_hash or verified["pauses"] != len(rows)
            or verified["eligible_ordinals_sha256"] != eligible_hash
            or verified["decision_ms"] != DEADLINE):
        raise ValueError("the recovered caller pool differs from the verified causal cohort")
    contract = {"pool_sha256": pool_hash, "eligible_ordinals_sha256": eligible_hash, "pauses": len(rows),
                "verified_pcm_sha256": verified["pcm_sha256"], "verified_report_sha256": file_digest(verified_path),
                "source_receipt_sha256": file_digest(rendered_path), "decision_ms": DEADLINE, "samples": SAMPLES}
    return folder, rows, rendered["shard_sha256"], contract


def recover(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the real caller cache stays on Vast")
    folder, rows, shard_hashes, contract = source_state(root)
    directory = root / "cache/own-teacher-pool"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "telephone-300ms.npy"
    receipt = directory / "telephone-300ms.json"
    if target.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["inputs"] != contract or saved["file_sha256"] != file_digest(target):
            raise ValueError("the persistent caller PCM cache changed")
        return target, saved

    # Only the PCM column enters the cache. The historic score columns
    # are excluded from materialization and all downstream training.
    shards = defaultdict(list)
    for output, (ordinal, row) in enumerate(rows):
        index, within = divmod(ordinal, SHARD_ROWS)
        shards[f"{index:04d}.parquet"].append((output, within, row))
    temporary = target.with_suffix(".part")
    pcm = np.lib.format.open_memmap(temporary, mode="w+", dtype="<i2", shape=(len(rows), 1, SAMPLES))
    load_audio = lru_cache(maxsize=2)(read_audio)
    parity = Counter()
    strata = {(row["source"], row["role"]) for _, row in rows}
    meter = Progress("own-teacher-pool-pcm", len(rows), "pauses")
    try:
        for name, entries in shards.items():
            path = folder / "scored" / name
            if file_digest(path) != shard_hashes[name]:
                raise ValueError(f"the archived PCM shard changed: {name}")
            column = pq.read_table(path, columns=["pcm_300"]).column("pcm_300")
            for output, index, row in entries:
                stored = column[index].as_py()
                if len(stored) != 2 * SAMPLES:
                    raise ValueError("archived caller PCM has a different window or sample width")
                pcm[output, 0] = np.frombuffer(stored, "<i2")
                key = row["source"], row["role"]
                if parity[key] < RENDERS_PER_ROLE_SOURCE:
                    label = Label(**{field: row[field] for field in Label.__dataclass_fields__})
                    original = load_audio(root / label.audio)
                    stop = label.pause_start + DEADLINE * RATE // 1000
                    if stop < len(original):
                        changed = original.copy()
                        changed[stop:stop + RATE // 10] ^= 32767
                        for audio in (original, changed):
                            if render(label, audio, DEADLINE).astype("<i2", copy=False).tobytes() != stored:
                                raise ValueError(f"caller PCM failed causal re-render: {name}, row {index}")
                        parity[key] += 1
            meter.advance(len(entries))
        pcm.flush()
        if any(parity[key] != RENDERS_PER_ROLE_SOURCE for key in strata):
            raise ValueError("causal re-render did not cover each source and role")
        if hashlib.sha256(memoryview(pcm)).hexdigest() != contract["verified_pcm_sha256"]:
            raise ValueError("the recovered ordered PCM differs from the verified project-head input")
        del pcm
        saved = {"inputs": contract, "file_sha256": file_digest(temporary),
                 "causal_rerenders": {"/".join(key): n for key, n in sorted(parity.items())}}
        temporary.replace(target)
        _write_json(receipt, saved)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return target, saved


if __name__ == "__main__":
    _, manifest = recover(data_root())
    print(json.dumps(manifest, indent=2))
