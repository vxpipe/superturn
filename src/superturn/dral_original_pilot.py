"""Plan and stage a bounded behavioral evaluation sample of original DRAL conversations."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import struct

from .acquire import _request, parse_wav_header, wav_bytes
from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .progress import Progress
from .sources import data_root, load_sources

SOURCE = "dral-original-pilot"
REPO = "jonavila/DRAL"
REVISION = "6a5386cf72d2a148a3d1e7db841bf02cbfa0ed40"
ORIGINALS = (("ES_015", "EN_024"), ("ES_050", "EN_043"), ("ES_070", "EN_111"), ("ES_075", "EN_108"))
HEADER_BYTES = 4096
MAX_BYTES = 160_000_000
WINDOWS_PERCENT = (35, 65)


def resolve(path):
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"


def read_range(path, total, start, count):
    end = start + count - 1
    with _request(resolve(path), headers={"Range": f"bytes={start}-{end}"}) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
            raise ValueError(f"publisher did not honor a bounded request for {path}")
        content = response.read(count + 1)
    if len(content) != count:
        raise ValueError(f"wrong bounded payload size for {path}")
    return content


def table(name):
    with _request(resolve(name)) as response:
        payload = response.read(60_001)
    if len(payload) > 60_000:
        raise ValueError("publisher metadata exceeded its transfer cap")
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))), len(payload)


def plan():
    conversations, conv_bytes = table("conversation.csv")
    participants, person_bytes = table("participant.csv")
    by_id = {row["id"]: row for row in conversations}
    producers = {row["id_unique"] for row in participants if row["is_producer"].strip()}
    with _request(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/recordings") as response:
        listing = response.read(1_000_001)
    if len(listing) > 1_000_000:
        raise ValueError("publisher WAV listing exceeded its transfer cap")
    files = {row["path"]: row for row in json.loads(listing) if row["type"] == "file"}
    speakers, selected = set(), []
    for component, pair in enumerate(ORIGINALS):
        dyad = None
        for recording in pair:
            row = by_id[recording]
            ids = [row[f"participant_id_{side}_unique"] for side in ("left", "right")]
            if row["original_or_reenacted"] != "OG" or len(set(ids)) != 2 or producers & set(ids):
                raise ValueError("selection requires two different nonproducer speakers in original dialogue")
            if dyad is None:
                dyad = set(ids)
                if dyad & speakers:
                    raise ValueError("pilot dyads share a participant")
                speakers.update(dyad)
            if dyad != set(ids):
                raise ValueError("paired languages have different participant identities")
            path = f"recordings/{recording}.wav"
            details = files[path]
            header = read_range(path, details["size"], 0, HEADER_BYTES)
            fmt, offset = parse_wav_header(header)
            encoding, channels, rate, _, align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if encoding != 1 or channels != 2 or bits != 16 or align != 4 or not 16_000 <= rate <= 48_000:
                raise ValueError(f"unexpected original WAV format for {recording}")
            frames = struct.unpack_from("<I", header, offset - 4)[0] // align
            if frames < 240 * rate or offset + frames * align > details["size"]:
                raise ValueError(f"original recording too short or incomplete for {recording}")
            selected.append({"recording": recording, "path": path, "component": component,
                             "language": recording.split("_")[0].lower(), "source_bytes": details["size"],
                             "publisher_sha256": details["lfs"]["oid"], "rate": rate,
                             "channel_sample_count": frames, "data_offset": offset, "fmt_hex": fmt.hex(),
                             "align": align, "people": ids})
    fixed_bytes = conv_bytes + person_bytes + len(listing) + HEADER_BYTES * len(selected)
    per_second = len(WINDOWS_PERCENT) * sum(row["rate"] * row["align"] for row in selected)
    seconds = min(50, (MAX_BYTES - 2 * fixed_bytes) // per_second)
    if seconds < 30:
        raise ValueError("pilot cap cannot support the predeclared two interior windows")
    for row in selected:
        rate, align, frames = row["rate"], row["align"], row["channel_sample_count"]
        row["ranges"] = [{"percent": point,
                          "start_byte": row["data_offset"] + (frames * point // 100 - seconds * rate // 2) * align,
                          "bytes": seconds * rate * align} for point in WINDOWS_PERCENT]
    return {"source": SOURCE, "revision": REVISION, "licence": "CC0-1.0", "window_seconds": seconds,
            "speakers": len(speakers), "calls_by_language": dict(Counter(r["language"] for r in selected)),
            "metadata_response_bytes": fixed_bytes, "payload_bytes": per_second * seconds,
            "transfer_cap_bytes": MAX_BYTES, "selection": selected}


def run(root, *, plan_only):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("DRAL acquisition and metadata checks run on Vast")
    if load_sources()[SOURCE].licence != "CC0-1.0":
        raise ValueError("source licence does not match the reviewed release")
    report = root / "reports/dral-original-pilot-plan.json"
    if plan_only:
        selection = plan()
        _write_json(report, selection)
        return {key: value for key, value in selection.items() if key != "selection"}
    selection = json.loads(report.read_text())
    if selection != plan():
        raise ValueError("pinned archive selection did not reproduce")
    folder = root / "raw" / SOURCE
    folder.mkdir(parents=True, exist_ok=True)
    meter = Progress("dral-original-pilot", selection["payload_bytes"], "B")
    manifest = []
    try:
        for row in selection["selection"]:
            for window in row["ranges"]:
                path = folder / f"{row['recording']}-{window['percent']}.wav"
                pcm = read_range(row["path"], row["source_bytes"], window["start_byte"], window["bytes"])
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(wav_bytes(bytes.fromhex(row["fmt_hex"]), pcm))
                temporary.replace(path)
                manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": file_digest(path)})
                meter.advance(len(pcm))
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    result = {"download_bytes": selection["payload_bytes"], "license": selection["licence"],
              "plan_sha256": file_digest(report), "files": manifest}
    _write_json(folder / "MANIFEST.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(data_root(), plan_only=args.plan_only), indent=2))
