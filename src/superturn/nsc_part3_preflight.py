"""Inspect the pinned NSC Part 3 same-room tar headers without acquiring audio."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import PurePosixPath
import re
from urllib.parse import quote

from .acquire import _request
from .ami_calibration import _write_json
from .nsc_part6_preflight import RangeReader, REPO, REVISION, tar_headers
from .progress import Progress
from .sources import data_root

LIMIT = 1_000
INVENTORY_LIMIT = 64 << 10
MEMBER = re.compile(r"SE(?P<session>\d+)-SP(?P<speaker>\d+|_Boundary)\.(?P<format>flac|jsonl)")
SPEAKER_INDEX = "SpeakerMetaData/Part 3 (Same Room).csv"
SPEAKER_INDEX_SHA256 = "1f00f4ae1e056cd9e1b7c7689b497a5177d192c55d098ca4ac9199f3e7c6453d"


def indexed_partners(groups):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(SPEAKER_INDEX)}"
    with _request(url) as response:
        body = response.read(40_000)
    if sha256(body).hexdigest() != SPEAKER_INDEX_SHA256:
        raise ValueError("the pinned Part 3 speaker index changed")
    rows = list(csv.DictReader(StringIO(body.decode("utf-8-sig"))))
    partners = {tuple(map(int, row["SCD"].split("-"))): tuple(map(int, row["PARTNER ID"].split("-")))
                for row in rows}
    if (len(rows) != len(partners) or
            any(person == partner or partners.get(partner) != person for person, partner in partners.items())):
        raise ValueError("the publisher pairing index contains ambiguous identities")
    voice_keys = [(session, speaker) for session, members in groups.items()
                  for speaker, fmt in members if speaker != "_Boundary" and fmt == "flac"]
    voices = {(int(session), int(speaker)): (session, speaker) for session, speaker in voice_keys}
    if len(voice_keys) != len(voices):
        raise ValueError("numeric archive keys map to multiple original speaker tracks")
    exact = {identity: source for identity, source in voices.items() if identity in partners}
    swapped = sum((speaker, session) in partners for session, speaker in voices)
    paired = []
    for person, (session, speaker) in sorted(exact.items()):
        partner = partners[person]
        if person >= partner or partner not in exact:
            continue
        other_session, other_speaker = exact[partner]
        tracks = [groups[session][speaker, "flac"], groups[other_session][other_speaker, "flac"]]
        transcripts = [groups[key][voice, "jsonl"] for key, voice in ((session, speaker), (other_session, other_speaker))
                       if (voice, "jsonl") in groups[key]]
        room = [groups[key]["_Boundary", "flac"] for key in sorted({session, other_session})
                if ("_Boundary", "flac") in groups[key]]
        paired.append({"keys": [[session, speaker], [other_session, other_speaker]],
                       "speaker_tracks": tracks, "room_tracks": room, "transcript_members": transcripts})
    return {
        "sha256": SPEAKER_INDEX_SHA256,
        "bytes": len(body),
        "reciprocal_composite_records": len(partners),
        "speaker_flac_keys": len(voices),
        "exact_session_speaker_matches": len(exact),
        "swapped_speaker_session_matches": swapped,
        "candidate_pairs": paired,
        "pairs_with_two_transcripts_and_room": sum(len(pair["transcript_members"]) == 2 and bool(pair["room_tracks"])
                                                   for pair in paired),
    }


def run(root, limit, partner_index=False):
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
    reader = RangeReader(fragments, byte_limit=1 << 20, request_limit=2 * limit)
    groups = defaultdict(dict)
    track_counts = Counter()
    unrecognized_extensions = Counter()
    tag = f"{limit}-indexed" if partner_index else str(limit)
    meter = Progress(f"nsc-part3-same-room-headers-{tag}", limit, "headers")
    parsed = 0
    pairs = []
    try:
        for name, is_file, offset, size in tar_headers(reader, limit):
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
            members = groups[session]
            ids = sorted({speaker for speaker, _ in members if speaker != "_Boundary"})
            if (not pairs and len(ids) == 2 and ("_Boundary", "flac") in members
                    and all((speaker, fmt) in members for speaker in ids for fmt in ("flac", "jsonl"))):
                pairs.append({"session": session, "speaker_ids": ids, "members": list(members.values()),
                              "member_bytes": sum(row["bytes"] for row in members.values())})
                if not partner_index:
                    break
    except BaseException:
        meter.finish("failed")
        raise
    meter.skip(limit - parsed)
    meter.finish()
    result = {
        "repository": REPO,
        "revision": REVISION,
        "fragments": fragments,
        "headers": parsed,
        "header_limit": limit,
        "requests": reader.requests,
        "metadata_bytes": len(body) + reader.bytes,
        "audio_or_transcript_payload_bytes": 0,
        "recognized_sessions": len(groups),
        "recognized_tracks": track_counts,
        "unrecognized_extensions": unrecognized_extensions,
        "provisional_same_room_pairs": pairs,
        "header_members": [row for members in groups.values() for row in members.values()],
    }
    if partner_index:
        index = indexed_partners(groups)
        result["publisher_partner_index"] = index
        result["metadata_bytes"] += index["bytes"]
        print(json.dumps({"publisher_records": index["reciprocal_composite_records"],
                          "exact_keys": index["exact_session_speaker_matches"],
                          "swapped_keys": index["swapped_speaker_session_matches"],
                          "indexed_candidate_pairs": len(index["candidate_pairs"]),
                          "with_transcripts_and_room": index["pairs_with_two_transcripts_and_room"]}, indent=2))
    _write_json(root / f"reports/nsc-part3-same-room-preflight-{tag}.json", result)
    print(json.dumps({"pairs": len(pairs), "headers": parsed, "metadata_bytes": result["metadata_bytes"],
                      "members_total_bytes": sum(row["member_bytes"] for row in pairs),
                      "recognized_sessions": len(groups),
                      "recognized_tracks": track_counts,
                      "unrecognized_extensions": unrecognized_extensions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=int, default=LIMIT)
    parser.add_argument("--partner-index", action="store_true", help="join the pinned speaker pairing census")
    args = parser.parse_args()
    if not 1 <= args.headers <= LIMIT:
        parser.error(f"--headers must be between 1 and {LIMIT}")
    run(data_root(), args.headers, args.partner_index)
