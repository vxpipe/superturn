"""Check duration and room PCM checksums on a bounded set of indexed NSC pairs."""

import argparse
from hashlib import sha256
import json

from .ami_calibration import _write_json
from .audio import flac_streaminfo
from .nsc_part3_preflight import LIMIT, SPEAKER_INDEX_SHA256
from .nsc_part6_preflight import RangeReader, REPO, REVISION
from .progress import Progress
from .sources import data_root

HEADER_SIZE = 42
LIMIT_PAIRS = 8
MAX_DURATION_DELTA = .5
ZERO_MD5 = "0" * 32


def describe(reader, member):
    position, length = member["logical_data_offset"], member["bytes"]
    if length < HEADER_SIZE or position < 512 or position % 512 or position + length > reader.ends[-1]:
        raise ValueError("invalid FLAC extent")
    rate, channels, bits, samples, digest = flac_streaminfo(reader.read(position, HEADER_SIZE))
    if not rate:
        raise ValueError("invalid native sample rate")
    return {"rate": rate, "channels": channels, "bits": bits, "samples": samples, "pcm_md5": digest}


def run(root, pair_limit, expected_digest):
    if not 1 <= pair_limit <= LIMIT_PAIRS:
        raise ValueError(f"pair limit must be between 1 and {LIMIT_PAIRS}")
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the stream metadata audit stays on Vast")
    path = root / f"reports/nsc-part3-same-room-preflight-{LIMIT}-indexed.json"
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected_digest:
        raise ValueError("header census differs from the supplied receipt")
    index = json.loads(payload)
    publisher = index["publisher_partner_index"]
    if (index["repository"], index["revision"], publisher["sha256"]) != (REPO, REVISION, SPEAKER_INDEX_SHA256):
        raise ValueError("expected the pinned mirror and publisher pairing index")
    available = [row for row in publisher["candidate_pairs"]
                 if len(row["transcript_members"]) == 2 and 1 <= len(row["room_tracks"]) <= 2]
    count = min(pair_limit, len(available))
    sample = [available[i * len(available) // count] for i in range(count)]
    reader = RangeReader(index["fragments"], byte_limit=4 * HEADER_SIZE * count + 1, request_limit=8 * count)
    outcomes = []
    meter = Progress("nsc-part3-streaminfo", count, "pairs")
    try:
        for pair in sample:
            offsets = {track["logical_data_offset"] for track in pair["speaker_tracks"]}
            if len(pair["speaker_tracks"]) != 2 or len(offsets) != 2:
                raise ValueError("expected two distinct indexed speaker tracks")
            speaker = [describe(reader, member) for member in pair["speaker_tracks"]]
            room = [describe(reader, member) for member in pair["room_tracks"]]
            tracks = speaker + room
            formats = {(track["rate"], track["channels"], track["bits"]) for track in tracks}
            durations = [track["samples"] / track["rate"] for track in tracks if track["samples"]]
            span = max(durations) - min(durations) if len(durations) == len(tracks) else None
            compatible = len(formats) == 1 and span <= MAX_DURATION_DELTA if span is not None else None
            room_match = len(room) == 2 and room[0] == room[1] and room[0]["pcm_md5"] != ZERO_MD5
            close_md5 = [track["pcm_md5"] for track in speaker]
            copied_speech = (close_md5[0] == close_md5[1] != ZERO_MD5
                             or any(track["pcm_md5"] != ZERO_MD5 and track["pcm_md5"] in close_md5
                                    for track in room))
            outcomes.append({"keys": pair["keys"], "speaker": speaker, "room": room,
                             "same_native_format": len(formats) == 1, "duration_span_seconds": span,
                             "duration_compatible": compatible,
                             "duplicate_declared_room_format_and_checksum": room_match,
                             "shared_declared_speaker_checksum": copied_speech})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"scope": "container headers only, timing feasibility",
              "header_index_sha256": expected_digest,
              "streaminfo_payload_bytes": reader.bytes, "audio_frames_read": 0,
              "logical_range_requests": reader.requests, "maximum_duration_span_seconds": MAX_DURATION_DELTA,
              "indexed_pairs_with_two_transcripts_and_room": len(available), "examined_pairs": count,
              "duration_compatible_pairs": sum(row["duration_compatible"] is True for row in outcomes),
              "duration_incompatible_pairs": sum(row["duration_compatible"] is False for row in outcomes),
              "duration_unverifiable_pairs": sum(row["duration_compatible"] is None for row in outcomes),
              "pairs_with_two_room_tracks": sum(len(row["room"]) == 2 for row in outcomes),
              "duplicate_declared_room_format_and_checksum_pairs":
                  sum(row["duplicate_declared_room_format_and_checksum"] for row in outcomes),
              "shared_declared_speaker_checksum_pairs": sum(row["shared_declared_speaker_checksum"] for row in outcomes)}
    _write_json(root / "reports/nsc-part3-streaminfo.json", {**report, "pairs": outcomes})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=LIMIT_PAIRS)
    parser.add_argument("--census-sha256", required=True, help="SHA-256 from the reviewed census receipt")
    args = parser.parse_args()
    if not 1 <= args.pairs <= LIMIT_PAIRS:
        parser.error(f"--pairs must be between 1 and {LIMIT_PAIRS}")
    run(data_root(), args.pairs, args.census_sha256)
