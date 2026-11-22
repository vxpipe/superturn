"""Count published Dutch speaker IDs without reading any audio columns."""

import argparse
from collections import Counter
from hashlib import sha256
import json

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .labels import read_rows
from .progress import Progress
from .shards import seed_for
from .sources import budget_bytes, data_root, load_sources

JOB = "voxpopuli-dutch-inventory"
REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
FILES = tuple(f"nl/train-{part:05d}-of-00003.parquet" for part in range(3))
STAGE = "voxpopuli-dutch-exposure"
VALIDATION_AND_TEST = tuple(f"nl/{role}-00000-of-00001.parquet" for role in ("validation", "test"))


def open_remote(filesystem, path):
    return filesystem.open(f"datasets/facebook/voxpopuli/{path}", "rb", revision=REVISION,
                           block_size=65536, cache_type="none")


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("dataset metadata stays on Vast")
    remote = HfFileSystem()
    counts = Counter()
    shards = []
    meter = Progress(JOB, len(FILES), "train shards")
    try:
        for path in FILES:
            with open_remote(remote, path) as handle:
                file = pq.ParquetFile(handle, pre_buffer=False)
                groups = []
                for index in range(file.metadata.num_row_groups):
                    meta = file.metadata.row_group(index)
                    table = file.read_row_group(index, columns=["speaker_id"])
                    raw = table["speaker_id"].to_pylist()
                    voices = Counter(voice for voice in raw if voice and voice != "unknown")
                    counts.update(voices)
                    groups.append({"group_index": index, "rows": len(raw), "with_speaker_id": sum(voices.values()),
                                   "distinct_speakers": len(voices), "clips_at_cap_20": sum(min(20, n) for n in voices.values()),
                                   "audio_column_compressed_bytes": sum(
                                       meta.column(col).total_compressed_size
                                       for col in range(meta.num_columns)
                                       if meta.column(col).path_in_schema.startswith("audio."))})
                shards.append({"file": path, "groups": groups})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result = {"scope": "Dutch parliamentary speech, publisher train indices, metadata only",
              "revision": REVISION, "license": "CC0", "audio_fetched": False,
              "distinct_speakers": len(counts), "shards": shards}
    _write_json(root / "reports" / f"{JOB}.json", result)
    print(json.dumps(result, indent=2))


def stage(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real audio stays on Vast")
    registered = load_sources()[STAGE]
    if registered.get("revision") != REVISION:
        raise ValueError("the named source revision changed")
    [path], [group] = registered.get("files"), registered.get("row_groups")
    inventory_path = root / "reports" / f"{JOB}.json"
    index = json.loads(inventory_path.read_text())
    if index["revision"] != REVISION:
        raise ValueError("the inventory revision changed")
    [entry] = [row for row in index["shards"] if row["file"] == path]
    estimate = entry["groups"][group]["audio_column_compressed_bytes"]
    raw = root / "raw"
    used = sum(item.stat().st_size for item in raw.rglob("*") if item.is_file())
    cap = registered.get("max_total_bytes")
    if estimate >= cap:
        raise ValueError("audio column exceeds the opt-in transfer cap")
    folder = raw / STAGE
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "train-row-group-6.parquet"
    receipt = folder / "MANIFEST.json"
    if receipt.is_file():
        cached = json.loads(receipt.read_text())
        if (not output.is_file() or file_digest(output) != cached["local_parquet_sha256"]
                or file_digest(inventory_path) != cached["inventory_sha256"]):
            raise ValueError("the staged Dutch exposure selection changed")
        print(json.dumps(cached, indent=2))
        return
    if used + cap > budget_bytes():
        raise ValueError("not enough campaign download budget for the Dutch row group")
    filesystem = HfFileSystem()
    protected = {row.speaker for row in read_rows(root / "labels/voxpopuli.jsonl") if row.split != "train"}
    for reserved in VALIDATION_AND_TEST:
        with open_remote(filesystem, reserved) as handle:
            table = pq.ParquetFile(handle, pre_buffer=False).read(columns=["speaker_id"])
            protected.update(table["speaker_id"].to_pylist())
    with open_remote(filesystem, path) as handle:
        file = pq.ParquetFile(handle, pre_buffer=False)
        meta = file.read_row_group(group, columns=["audio_id", "speaker_id"])
        speakers = Counter()
        kept = []
        order = sorted(range(len(meta)), key=lambda i: seed_for(STAGE, meta["audio_id"][i].as_py()))
        for i in order:
            voice = meta["speaker_id"][i].as_py()
            if not voice or voice == "unknown" or voice in protected or speakers[voice] >= 20:
                continue
            speakers[voice] += 1
            kept.append(i)
        if not kept:
            raise ValueError("no Dutch rows remain after excluding held-out speaker IDs")
        meter = Progress(STAGE, estimate, "compressed audio bytes")
        try:
            audio = file.read_row_group(group, columns=["audio_id", "speaker_id", "audio"]).take(sorted(kept))
            temporary = output.with_suffix(".partial")
            pq.write_table(audio, temporary, compression="zstd")
            temporary.replace(output)
            meter.advance(estimate)
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
    result = {"scope": "train-only Dutch exposure audio, no teacher targets or new model",
              "source": registered.name, "licence": registered.licence, "attribution": registered.attribution,
              "revision": REVISION, "file": path, "row_group": group, "compressed_audio_bytes": estimate,
              "protected_speaker_ids": len(protected), "selected_speakers": len(speakers), "selected_rows": len(audio),
              "selection_sha256": sha256(json.dumps(audio["audio_id"].to_pylist()).encode()).hexdigest(),
              "inventory_sha256": file_digest(inventory_path), "local_parquet_sha256": file_digest(output)}
    _write_json(receipt, result)
    _write_json(root / "reports" / f"{STAGE}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true")
    (stage if parser.parse_args().stage else run)(data_root())
