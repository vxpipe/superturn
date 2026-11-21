"""Stage two voice-disjoint Open Yap calls for timing and label feasibility."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json

import av
import numpy as np

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .nsc_part6_preflight import RangeReader
from .open_yap_preflight import SOURCE
from .progress import Progress
from .sources import budget_bytes, data_root, load_sources

CAP = 100_000_000


def select(index):
    locations = {key: (shard, files) for shard in index["shards"] for key, files in shard["ranges"].items()}
    options = []
    for pair in combinations(sorted(locations), 2):
        first, second = pair
        people = index["records"][first]["people"] + index["records"][second]["people"]
        if len(set(people)) != 4:
            continue
        size = sum(file["bytes"] for key in pair for file in locations[key][1].values())
        if size <= CAP:
            options.append((index["components"][first] == index["components"][second], size, pair))
    if not options:
        raise ValueError("no four-voice sample fits the 100 MB pilot")
    same_component, size, pair = min(options)
    return pair, size, same_component, locations


def native_pcm(path, expected):
    digest = hashlib.sha256()
    samples = 0
    with av.open(str(path)) as media:
        for frame in media.decode(audio=0):
            if (frame.sample_rate, frame.layout.nb_channels, frame.format.bits) != (48_000, 1, 16):
                raise ValueError("expected native mono 48 kHz s16 audio")
            digest.update(np.ascontiguousarray(frame.to_ndarray(), dtype="<i2").tobytes())
            samples += frame.samples
    if samples != expected["samples"] or digest.hexdigest() != expected["pcmSha256"]:
        raise ValueError("decoded PCM does not match the publisher's sample clock receipt")
    return samples


def transfer(source, target, archive, descriptor):
    expected = descriptor["verification"]
    if not target.is_file() or file_digest(target) != expected["sha256"]:
        name, size, oid = (archive[key] for key in ("archive", "archive_bytes", "archive_oid"))
        reader = RangeReader([(name, size, oid)], repo=source.get("repo"), revision=source.get("revision"),
                             byte_limit=descriptor["bytes"] + 1, request_limit=1)
        data = reader.read(descriptor["offset"], descriptor["bytes"])
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Open Yap member failed transport integrity")
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(target)
    if "audio" in expected:
        return native_pcm(target, expected["audio"])
    return None


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Open Yap recordings may only be staged on Vast")
    receipt = json.loads((root / "reports/open-yap-preflight.json").read_text())
    source = load_sources()[SOURCE]
    index_path = root / "runs/open-yap-preflight/metadata.json"
    if receipt["revision"] != source.get("revision") or receipt["index_sha256"] != file_digest(index_path):
        raise ValueError("Open Yap sample plan does not match its pinned archive index")
    index = json.loads(index_path.read_text())
    pair, size, same_component, locations = select(index)
    raw = root / "raw"
    if sum(path.stat().st_size for path in raw.rglob("*") if path.is_file()) + size > budget_bytes():
        raise ValueError("bounded pilot exceeds the campaign raw data budget")
    folder = raw / SOURCE
    folder.mkdir(parents=True, exist_ok=True)
    meter = Progress("open-yap-audio", size, "bytes")
    manifest = {}
    try:
        with ThreadPoolExecutor(3) as pool:
            jobs = [(key, suffix, descriptor, pool.submit(transfer, source, folder / f"{key}.{suffix}",
                                                           locations[key][0], descriptor))
                    for key in pair for suffix, descriptor in locations[key][1].items()]
            for key, suffix, descriptor, task in jobs:
                samples = task.result()
                manifest[f"{key}.{suffix}"] = {"sha256": descriptor["verification"]["sha256"],
                                               "bytes": descriptor["bytes"], "samples": samples}
                meter.advance(descriptor["bytes"])
        for key in pair:
            if manifest[f"{key}.a.flac"]["samples"] != manifest[f"{key}.b.flac"]["samples"]:
                raise ValueError("caller and partner native sample counts differ")
        _write_json(folder / "MANIFEST.json", {"index_sha256": receipt["index_sha256"],
                                               "calls": pair, "members": manifest})
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"calls": len(pair), "speakers": 4, "components": 1 if same_component else 2,
              "selected_bytes": size, "audio_seconds": sum(manifest[f"{key}.a.flac"]["samples"] / 48_000 for key in pair),
              "member_manifest_sha256": file_digest(folder / "MANIFEST.json"), "decoded_pcm_verified": True}
    _write_json(root / "reports/open-yap-audio.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
