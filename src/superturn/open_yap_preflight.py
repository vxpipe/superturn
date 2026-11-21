"""Inventory the public Open Yap sample with metadata-only byte ranges."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import tarfile

from .acquire import _request
from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .nsc_part6_preflight import BLOCK, RangeReader, pax_fields
from .progress import Progress
from .sources import data_root, load_sources
from .splits import speaker_groups

SOURCE = "open-yap-sample"
ARCHIVES = ("shard-00000.tar", "shard-00001.tar")
HEADER_LIMIT = 130
FILE_LIMIT = 64_000
RECORD_LIMIT = 2_000_000
VERIFICATION_SHA256 = "6a5766b6ad79e367815c9ccfaa89064be011afdc95c5065477e19e64802c42bb"


def read_small(url):
    with _request(url) as response:
        data = response.read(FILE_LIMIT + 1)
    if len(data) > FILE_LIMIT:
        raise ValueError("Open Yap metadata transfer exceeds its cap")
    return data


def scan(archive, repo, revision, budget, expected, verification):
    shard, size, oid = archive
    reader = RangeReader([archive], byte_limit=budget, request_limit=HEADER_LIMIT * 2,
                         repo=repo, revision=revision)
    records, audio = {}, defaultdict(dict)
    position = 0
    extension = {}
    meter = Progress(f"open-yap-metadata-{shard}", HEADER_LIMIT, "headers")
    try:
        for index in range(HEADER_LIMIT):
            header = reader.read(position, BLOCK)
            meter.advance(1)
            if not any(header):
                meter.skip(HEADER_LIMIT - index - 1)
                break
            item = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            content = position + BLOCK
            extent = item.size
            if item.type == tarfile.XHDTYPE:
                if not 0 < extent <= 4096:
                    raise ValueError("unexpected Open Yap tar extension")
                extension = pax_fields(reader.read(content, extent))
            elif item.isfile():
                name = extension.get("path", item.name)
                extent = int(extension.get("size", extent))
                extension.clear()
                key, _, suffix = name.rsplit("/", 1)[-1].partition(".")
                if key not in expected:
                    raise ValueError("archive member is absent from the public sample index")
                if suffix == "json":
                    if not 0 < extent <= RECORD_LIMIT or key in records:
                        raise ValueError("unexpected Open Yap record size or duplicate")
                    entry = verification[f"{key}.{suffix}"]
                    if extent != entry["byteLength"]:
                        raise ValueError("sample JSON differs from the publisher's manifest")
                    payload = reader.read(content, extent)
                    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                        raise ValueError("sample JSON failed transport integrity")
                    records[key] = json.loads(payload)
                elif suffix in ("a.flac", "b.flac"):
                    if suffix in audio[key]:
                        raise ValueError("duplicate audio member")
                    entry = verification[f"{key}.{suffix}"]
                    if extent != entry["byteLength"]:
                        raise ValueError("audio member differs from the publisher's manifest")
                    audio[key][suffix] = {"offset": content, "bytes": extent, "verification": entry}
            elif not item.isdir() or extent != 0:
                raise ValueError("unsupported tar member type")
            position = content + ((extent + BLOCK - 1) // BLOCK) * BLOCK
            if extent < 0 or position > size:
                raise ValueError("invalid tar member extent")
        else:
            raise ValueError("Open Yap header cap reached before the end marker")
        if set(audio) != set(records) or any(set(members) != {"a.flac", "b.flac"} for members in audio.values()):
            raise ValueError("Open Yap archive member pairing is incomplete")
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return {"archive": shard, "archive_bytes": size, "archive_oid": oid,
            "bytes_transferred": reader.bytes, "requests": reader.requests,
            "metadata": records, "ranges": audio}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Open Yap metadata inspection runs on Vast")
    source = load_sources()[SOURCE]
    repo, revision = source.get("repo"), source.get("revision")
    base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
    license_text = read_small(f"{base}/LICENSE.txt")
    if b"Open Yap 1K" not in license_text or b"CC-BY-4.0" not in license_text:
        raise ValueError("unexpected sample licence")
    metadata = read_small(f"{base}/metadata.jsonl")
    verification_bytes = read_small(f"{base}/verification.json")
    if hashlib.sha256(verification_bytes).hexdigest() != VERIFICATION_SHA256:
        raise ValueError("unexpected publisher verification file")
    verification = json.loads(verification_bytes)
    if hashlib.sha256(metadata).hexdigest() != verification["files"]["metadata.jsonl"]["sha256"]:
        raise ValueError("the Open Yap JSONL inventory changed")
    keys = {row["key"] for row in map(json.loads, metadata.splitlines())}
    if len(keys) != 16:
        raise ValueError("pinned Open Yap sample count changed")
    listing = read_small(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}")
    entries = {row["path"]: row for row in json.loads(listing)}
    archives = [(name, entries[name]["size"], entries[name]["lfs"]["oid"]) for name in ARCHIVES]
    for name, size, oid in archives:
        entry = verification["files"][name]
        if (size, oid) != (entry["byteLength"], entry["sha256"]):
            raise ValueError("archive listing differs from the publisher's manifest")
    base_cost = len(license_text) + len(metadata) + len(listing) + len(verification_bytes)
    member_hashes = verification["archive_members"]
    budget = source.get("max_total_bytes") - base_cost
    header_allowance = budget - sum(v["byteLength"] for name, v in member_hashes.items() if name.endswith(".json"))
    if header_allowance < HEADER_LIMIT * BLOCK * len(archives):
        raise ValueError("Open Yap metadata exceeds its staged budget")
    with ThreadPoolExecutor(max_workers=len(archives)) as pool:
        pending = [pool.submit(scan, archive, repo, revision, budget, keys, member_hashes) for archive in archives]
        shards = [future.result() for future in pending]
    if sum(shard["bytes_transferred"] for shard in shards) > budget:
        raise ValueError("Open Yap metadata transport exceeded its staged budget")
    records, voices = {}, {}
    for shard in shards:
        for key, metadata_row in shard.pop("metadata").items():
            if key in records or metadata_row["sample_rate"] != 48_000:
                raise ValueError("unexpected duplicate or sample rate")
            speaker_rows = metadata_row["speakers"]
            people = tuple(speaker_rows[side]["speaker_id"] for side in ("a", "b"))
            if not all(people) or people[0] == people[1]:
                raise ValueError("Open Yap needs two distinct speaker identifiers")
            voices[key] = people
            columns = {}
            for side in ("a", "b"):
                words = metadata_row["transcripts"][side]["words"]
                timed = [word for word in words if word["start"] is not None and word["end"] is not None]
                track = speaker_rows[side]
                columns[side] = {"words": len(words), "timed_words": len(timed),
                                 "recording_seconds": track["recording"]["duration_seconds"],
                                 "silence_profile": track["audio_metrics"]["silence_profile"]}
            records[key] = {"people": people, "seconds": metadata_row["duration_seconds"], "tracks": columns}
    if set(records) != keys or len(set().union(*map(set, voices.values()))) != 8:
        raise ValueError("Open Yap release identities did not match its source card")
    components = speaker_groups(voices.items())
    private = root / "runs/open-yap-preflight/metadata.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    _write_json(private, {"shards": shards, "records": records, "components": components})
    report = {"repository": repo, "revision": revision, "conversations": len(records), "speakers": 8,
              "pairs": len({tuple(sorted(people)) for people in voices.values()}),
              "connected_component_sizes": sorted(Counter(components.values()).values(), reverse=True),
              "timed_words": sum(track["timed_words"] for row in records.values() for track in row["tracks"].values()),
              "metadata_bytes": base_cost + sum(shard["bytes_transferred"] for shard in shards),
              "licence_sha256": hashlib.sha256(license_text).hexdigest(),
              "index_sha256": file_digest(private), "audio_bytes": 0}
    _write_json(root / "reports/open-yap-preflight.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
