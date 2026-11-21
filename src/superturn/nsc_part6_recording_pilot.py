"""Stage the exact two Singapore Part 6 pairs for a clock feasibility check."""

from itertools import combinations
import json
import re
import tarfile

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .nsc_part6_preflight import BLOCK, DESIGNS, REPO, REVISION, RangeReader, inventory
from .progress import Progress
from .sources import budget_bytes, data_root, load_sources

SOURCE = "nsc-part6-pilot"
LICENCE = "Singapore Open Data Licence v1.0"
HEADER_REPORT = "c39c09be40bb072e7b7eaee6ffeaf4de0cba58fd149d2c69107800a685981863"
PAYLOAD_BYTES = 38_003_262
TRANSFER_LIMIT = 40_000_000


def select(source_report):
    if (source_report["repository"], source_report["revision"]) != (REPO, REVISION):
        raise ValueError("catalogue report refers to another source")
    if set(source_report["designs"]) != set(DESIGNS):
        raise ValueError("catalogue report does not include every design")
    ranked = []
    for design, evidence in source_report["designs"].items():
        for pair in evidence["pairs"]:
            speakers = pair["speakers"]
            if len(speakers) != 2 or any(not str(s).isdigit() for s in speakers):
                raise ValueError("proposed recordings do not contain two speaker identifiers")
            if not re.fullmatch(r"[0-9]+", pair["session"]) or not re.fullmatch(r"[A-Za-z]+", pair["task"]):
                raise ValueError("invalid session or task identifier")
            if any(set(members) != {"jsonl", "flac"} for members in speakers.values()):
                raise ValueError("the pair lacks its original annotations or audio")
            size = sum(member["bytes"] for members in speakers.values() for member in members.values())
            ranked.append((size, design, pair["session"], pair["task"], pair))
    possible = [candidate for candidate in combinations(sorted(ranked, key=lambda row: row[:4]), 2)
                if candidate[0][1] != candidate[1][1]
                and not (set(candidate[0][-1]["speakers"]) & set(candidate[1][-1]["speakers"]))]
    if not possible:
        raise ValueError("two cross-design speaker-disjoint pairs are unavailable")
    chosen = min(possible, key=lambda pair: sum(row[0] for row in pair))
    if sum(row[0] for row in chosen) != PAYLOAD_BYTES:
        raise ValueError("the pair selection no longer matches the inspected byte commitment")
    return chosen


def pair_members(index, design, session, task, pair):
    for track, (speaker, members) in enumerate(sorted(pair["speakers"].items()), start=1):
        for fmt, entry in sorted(members.items()):
            yield {"pair": index, "track": track, "design": design, "file": f"pair-{index}-track-{track}.{fmt}",
                   "speaker": speaker, "session": session, "task": task, "bytes": entry["bytes"],
                   "archive_position": entry["byte_offset"]}


def verify_stage(root):
    report = root / "reports/nsc-part6-preflight.json"
    if file_digest(report) != HEADER_REPORT:
        raise ValueError("archive header report differs from the verified receipt")
    desired = [row for index, (_, design, session, task, pair) in enumerate(select(json.loads(report.read_text())), 1)
               for row in pair_members(index, design, session, task, pair)]
    directory = root / "raw" / SOURCE
    data = json.loads((directory / "MANIFEST.json").read_text())
    expected_provenance = {"preflight_sha256": HEADER_REPORT, "revision": REVISION, "source": SOURCE,
                           "repo": REPO, "license": LICENCE}
    if any(data.get(key) != value for key, value in expected_provenance.items()) or len(data["files"]) != len(desired):
        raise ValueError("staged recording manifest differs from the verified pair selection")
    for actual, expected in zip(data["files"], desired, strict=True):
        path = directory / expected["file"]
        if (any(actual.get(key) != value for key, value in expected.items())
                or path.stat().st_size != expected["bytes"] or file_digest(path) != actual["sha256"]):
            raise ValueError("staged recording differs from its selected member or digest")
    return data


def stage(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the recording pilot is restricted to the campaign machine")
    source = load_sources()[SOURCE]
    if source.licence != LICENCE or source.get("revision") != REVISION:
        raise ValueError("dataset registration differs from the reviewed source")
    report = root / "reports/nsc-part6-preflight.json"
    if file_digest(report) != HEADER_REPORT:
        raise ValueError("archive header report differs from the verified receipt")
    directory = root / "raw" / SOURCE
    manifest = directory / "MANIFEST.json"
    if manifest.is_file():
        return verify_stage(root)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("recording acquisition was interrupted before verification")
    existing = sum(path.stat().st_size for path in (root / "raw").rglob("*") if path.is_file())
    if existing + TRANSFER_LIMIT > budget_bytes():
        raise ValueError("the two-pair pilot exceeds the approved acquisition budget")
    review = json.loads(report.read_text())
    chosen = select(review)
    current, inventory_bytes = inventory()
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    meter = Progress("nsc-part6-stage", PAYLOAD_BYTES, "B")
    try:
        pulled = inventory_bytes
        for index, (size, design, session, task, pair) in enumerate(chosen, start=1):
            fragments = current[design]
            if [list(row) for row in fragments] != review["designs"][design]["fragments"]:
                raise ValueError("fragment inventory differs from the reviewed manifest")
            reader = RangeReader(fragments, byte_limit=size + 8193, request_limit=24)
            for row in pair_members(index, design, session, task, pair):
                position, count = row["archive_position"], row["bytes"]
                if position < BLOCK or position % BLOCK or count <= 0:
                    raise ValueError("invalid staged member range")
                fmt = row["file"].rsplit(".", 1)[-1]
                header = tarfile.TarInfo.frombuf(reader.read(position - BLOCK, BLOCK), "utf-8", "strict")
                name = f"SE{session}-Pt{task}-SP{row['speaker']}.{fmt}"
                if (header.name.rsplit("/", 1)[-1], header.size, header.type) != (name, count, tarfile.REGTYPE):
                    raise ValueError("archive member does not reproduce the paired header")
                content = reader.read(position, count)
                if fmt == "flac" and not content.startswith(b"fLaC"):
                    raise ValueError("the advertised track does not contain a native FLAC stream")
                path = directory / row["file"]
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
                files.append({**row, "sha256": file_digest(path)})
                meter.advance(count)
            pulled += reader.bytes
        if pulled > TRANSFER_LIMIT:
            raise ValueError("archive response bodies exceeded the recording-pilot cap")
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result = {"source": SOURCE, "repo": REPO, "revision": REVISION, "license": LICENCE,
              "preflight_sha256": HEADER_REPORT, "files": files, "payload_bytes": PAYLOAD_BYTES,
              "response_body_bytes": pulled, "pairs": 2, "speaker_ids": 4, "provisional_tracks": True}
    _write_json(manifest, result)
    return result


if __name__ == "__main__":
    metadata = stage(data_root())
    print(json.dumps({key: metadata[key] for key in ("payload_bytes", "response_body_bytes", "pairs", "speaker_ids")}))
