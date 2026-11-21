"""Inspect the pinned NSC Part 3 same-room tar headers without acquiring audio."""

from collections import Counter, defaultdict
import json
from pathlib import PurePosixPath
import re

from .acquire import _request
from .ami_calibration import _write_json
from .nsc_part6_preflight import RangeReader, REPO, REVISION, tar_headers
from .progress import Progress
from .sources import data_root

LIMIT = 64
INVENTORY_LIMIT = 64 << 10
MEMBER = re.compile(r"SE(?P<session>\d+)-SP(?P<speaker>\d+|_Boundary)\.(?P<format>flac|jsonl)")


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("archive inspection runs on Vast")
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/Part%203"
    with _request(url) as response:
        body = response.read(INVENTORY_LIMIT)
    if len(body) == INVENTORY_LIMIT:
        raise ValueError("archive inventory exceeds the read limit")
    fragments = sorted((row["path"], row["size"], row["lfs"]["oid"])
                       for row in json.loads(body)
                       if re.fullmatch(r"Part 3/SameRooms-[0-9]{2}\.tar", row["path"]))
    if [PurePosixPath(name).name for name, _, _ in fragments] != [
            f"SameRooms-{i:02d}.tar" for i in range(8)] or any(
                size <= 0 or not re.fullmatch(r"[a-f0-9]{64}", oid) for _, size, oid in fragments):
        raise ValueError("unexpected pinned Part 3 fragment layout")
    reader = RangeReader(fragments, byte_limit=256 << 10, request_limit=2 * LIMIT)
    groups = defaultdict(dict)
    track_counts = Counter()
    unrecognized_extensions = Counter()
    meter = Progress("nsc-part3-same-room-headers", LIMIT, "headers")
    parsed = 0
    try:
        for name, is_file, offset, size in tar_headers(reader, LIMIT):
            parsed += 1
            meter.advance(1)
            if not is_file or name is None:
                continue
            match = MEMBER.fullmatch(name)
            if match is None:
                unrecognized_extensions[PurePosixPath(name).suffix] += 1
                continue
            session, speaker, fmt = match.groups()
            if size <= 0 or (speaker, fmt) in groups[session]:
                raise ValueError("invalid or duplicate paired archive member")
            groups[session][speaker, fmt] = {"name": name, "logical_data_offset": offset, "bytes": size}
            track_counts[("room" if speaker == "_Boundary" else "speaker") + "_" + fmt] += 1
    except BaseException:
        meter.finish("failed")
        raise
    meter.skip(LIMIT - parsed)
    meter.finish()
    pairs = []
    for session, members in sorted(groups.items()):
        ids = sorted({speaker for speaker, _ in members if speaker != "_Boundary"})
        if (len(ids) == 2 and ("_Boundary", "flac") in members
                and all((speaker, fmt) in members for speaker in ids for fmt in ("flac", "jsonl"))):
            pairs.append({"session": session, "speaker_ids": ids, "members": list(members.values()),
                          "member_bytes": sum(row["bytes"] for row in members.values())})
    result = {
        "repository": REPO,
        "revision": REVISION,
        "fragments": fragments,
        "headers": parsed,
        "requests": reader.requests,
        "metadata_bytes": len(body) + reader.bytes,
        "audio_or_transcript_payload_bytes": 0,
        "recognized_sessions": len(groups),
        "recognized_tracks": track_counts,
        "unrecognized_extensions": unrecognized_extensions,
        "provisional_same_room_pairs": pairs,
    }
    _write_json(root / "reports/nsc-part3-same-room-preflight.json", result)
    print(json.dumps({"pairs": len(pairs), "metadata_bytes": result["metadata_bytes"],
                      "members_total_bytes": sum(row["member_bytes"] for row in pairs),
                      "recognized_tracks": track_counts,
                      "unrecognized_extensions": unrecognized_extensions}, indent=2))


if __name__ == "__main__":
    run(data_root())
