"""Find roster-verified original NSC Part 4 pairs without reading audio."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
import re

from .acquire import _request
from .ami_calibration import _write_json
from .audio import flac_streaminfo
from .dense_teacher import file_digest
from .language_failure_audit import read_receipt
from .nsc_part6_preflight import INVENTORY_BYTES, REPO, REVISION, RangeReader, tar_headers
from .nsc_turn_provenance import REPORT, ROSTERS, part_rosters
from .progress import Progress
from .shards import seed_for
from .sources import budget_bytes, data_root, load_sources
from .splits import assign, speaker_groups


SOURCE = "nsc-part4-native-pilot"
KIND = "SameRooms"
HEADERS = 384
MEMBER = re.compile(r"SE([0-9]+)-SP([0-9]+)\.(flac|jsonl)")


def inventory():
    path = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/Part%204"
    with _request(path) as response:
        encoded = response.read(INVENTORY_BYTES)
    if len(encoded) >= INVENTORY_BYTES:
        raise ValueError("the pinned archive listing exceeded the metadata bound")
    parts = sorted((row["path"], row["size"], row["lfs"]["oid"]) for row in json.loads(encoded)
                   if re.fullmatch(rf"Part 4/{KIND}-[0-9]{{2}}\.tar", row["path"]))
    if [row[0] for row in parts] != [f"Part 4/{KIND}-{i:02d}.tar" for i in range(3)]:
        raise ValueError("the original same-room archive differs from the pinned fragment layout")
    return parts, len(encoded)


def choose(pairs, groups, cap):
    sizes = {session: sum(member["bytes"] for paths in people.values() for member in paths.values())
             for session, people in pairs.items()}
    rank = sorted((session for session in pairs if assign(SOURCE, groups[session], "train-conversation") == "train"),
                  key=lambda session: (seed_for(SOURCE, session), session))
    return next(((left, right) for left, right in combinations(rank, 2)
                 if groups[left] != groups[right] and sizes[left] + sizes[right] + (1 << 20) <= cap), ()), sizes


def scan(fragments, roster, reciprocated, groups, headers, cap):
    reader = RangeReader(fragments, byte_limit=512 * 1024, request_limit=2 * headers)
    found = defaultdict(lambda: defaultdict(dict))
    complete = {}
    meter = Progress(SOURCE, headers, "headers")
    parsed = 0
    try:
        for name, regular, start, size in tar_headers(reader, headers):
            parsed += 1
            meter.advance(1)
            key = MEMBER.fullmatch(name or "")
            if not regular or key is None:
                continue
            raw_session, raw_speaker, fmt = key.groups()
            session, speaker = str(int(raw_session)), str(int(raw_speaker))
            if session not in reciprocated or speaker not in roster[session]:
                continue
            files = found[session][speaker]
            if fmt in files or size <= 42 or not 0 < start < reader.ends[-1]:
                raise ValueError("invalid duplicate or empty original archive member")
            files[fmt] = {"name": name, "offset": start, "bytes": size}
            pair = found[session]
            if set(pair) == roster[session] and all(set(paths) == {"flac", "jsonl"} for paths in pair.values()):
                complete[session] = dict(pair)
                if choose(complete, groups, cap)[0]:
                    break
    except BaseException:
        meter.finish("failed")
        raise
    meter.skip(headers - parsed)
    meter.finish()
    return reader, complete, len(found), parsed


def run(root, headers=HEADERS):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("run the original recording check on Vast")
    census, census_sha = read_receipt(root, REPORT)
    if not census["parts"]["4"]["breadth_pass"]:
        raise ValueError("the source did not pass the roster breadth screen")
    source = load_sources()[SOURCE]
    if (source.get("repo"), source.get("revision")) != (REPO, REVISION):
        raise ValueError("the source registry does not match the inspected archives")
    roster_directory = root / "raw" / ROSTERS
    for file, expected in census["inputs"][ROSTERS]["files"].items():
        if file_digest(roster_directory / file) != expected:
            raise ValueError("a pinned participant roster changed")
    part4, reciprocated, *_ = part_rosters(roster_directory / "SpeakerMetaData")
    groups = speaker_groups((session, part4[session]) for session in reciprocated)
    fragments, listing_bytes = inventory()
    transport_cap = source.get("max_total_bytes")
    reader, pairs, mentioned, parsed = scan(fragments, part4, reciprocated, groups, headers, transport_cap)
    chosen, sizes = choose(pairs, groups, transport_cap)
    inventory_bytes = listing_bytes + reader.bytes
    raw_used = sum(path.stat().st_size for path in (root / "raw").rglob("*") if path.is_file())
    selected = []
    for session in chosen:
        people = pairs[session]
        for paths in people.values():
            item = paths["flac"]
            format_data = reader.read(item["offset"], 42)
            item["format"] = flac_streaminfo(format_data)
            rate, channels, bits, samples, _ = item["format"]
            if rate <= 0 or channels != 1 or bits != 16 or samples <= 0:
                raise ValueError("the selected original track cannot verify a native mono clock")
        selected.append({"session": session, "people": people, "pair_bytes": sizes[session]})
    result = {
        "scope": "native same-room headers and stream information only",
        "roster_census_sha256": census_sha, "repository": REPO, "revision": REVISION,
        "archive_fragments": fragments, "headers_budget": headers, "headers_checked": parsed,
        "range_requests": reader.requests,
        "inspected_metadata_bytes": listing_bytes + reader.bytes, "header_walk_bytes": inventory_bytes,
        "complete_reciprocal_pairs": len(pairs), "roster_session_headers_seen": mentioned,
        "selected": selected, "byte_cap": transport_cap,
        "raw_bytes_before_audio": raw_used, "global_budget_bytes": budget_bytes(),
        "payload_allowed": bool(selected) and raw_used + transport_cap <= budget_bytes(),
    }
    _write_json(root / "reports" / f"{SOURCE}.json", result)
    print(json.dumps({key: result[key] for key in ("complete_reciprocal_pairs", "inspected_metadata_bytes", "payload_allowed")},
                     indent=2))
    print("selected_pair_sizes", [row["pair_bytes"] for row in selected])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=int, default=HEADERS)
    args = parser.parse_args()
    if not 1 <= args.headers <= 896:
        parser.error("header budget must be between 1 and 896")
    run(data_root(), args.headers)
