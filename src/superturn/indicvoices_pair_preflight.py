"""Check original IndicVoices call pairing without staging an archive."""

import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import PurePosixPath
import re
import tarfile
import urllib.request
import wave

import numpy as np

from .audio import RATE, silero
from .release_cohort import write_once
from .sources import data_root, load_sources

JOB = "indicvoices-pair-preflight"
SOURCE = "indicvoices-hindi-prefix-probe"
PREFIX_BYTES = 8 * 1024 * 1024
ARCHIVE_BYTES = 14_322_746_939
CALL_SIDE = re.compile(r"(?P<call>[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})_(?P<side>[01])")


def fetch(source):
    etag = '"' + source.get("archive_etag") + '"'
    last = PREFIX_BYTES - 1
    request = urllib.request.Request(source.get("archive_url"), headers={
        "Range": f"bytes=0-{last}", "If-Match": etag, "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=30) as handle:
        if (handle.status, handle.headers.get("Content-Range"), handle.headers.get("ETag")) != (
            206, f"bytes 0-{last}/{ARCHIVE_BYTES}", etag,
        ):
            raise ValueError("the publisher did not return the pinned byte range")
        payload = handle.read(PREFIX_BYTES)
    if len(payload) != PREFIX_BYTES:
        raise ValueError("the bounded archive prefix is incomplete")
    return payload


def read_members(payload):
    members = defaultdict(dict)
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                parts = CALL_SIDE.fullmatch(path.stem)
                if not member.isfile() or parts is None or path.suffix not in (".json", ".wav"):
                    continue
                cap = 256 * 1024 if path.suffix == ".json" else 16 * 1024 * 1024
                if member.size > cap:
                    continue
                call, side = parts.groups()
                key = side, path.suffix
                if key in members[call]:
                    raise ValueError("the prefix has duplicate original member identities")
                data = archive.extractfile(member).read(cap)
                if len(data) == member.size:
                    members[call][key] = data
    except (EOFError, tarfile.ReadError) as error:
        if "end" not in str(error).lower():
            raise
    return members


def decode(wav):
    with wave.open(io.BytesIO(wav), "rb") as track:
        if (track.getnchannels(), track.getframerate(), track.getsampwidth()) != (1, 8000, 2):
            raise ValueError("expected native 8 kHz mono PCM")
        frames = track.getnframes()
        value = np.frombuffer(track.readframes(frames), dtype="<i2")
        if len(value) != frames:
            raise ValueError("truncated original WAV")
    return value


def own_intervals(sidecar, frames):
    speaker = sidecar.get("speaker_id")
    result = []
    for row in sidecar.get("verbatim", []):
        if not speaker or row.get("speaker_id") != speaker:
            continue
        try:
            start, stop = (round(float(row[field]) * RATE) for field in ("start", "end"))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if 0 <= start < stop <= frames:
            result.append((start, stop))
    return sorted(result)


def inspect_pairs(members, *, clock=False):
    from scipy.signal import resample_poly
    from .dral_label_pilot import isolation

    counts = Counter()
    pairs = []
    signals = {}
    detect = silero()
    for group in members.values():
        if set(group) != {(side, ext) for side in ("0", "1") for ext in (".wav", ".json")}:
            counts["incomplete_pair"] += 1
            continue
        counts["complete_pairs"] += 1
        audio = [decode(group[side, ".wav"]) for side in ("0", "1")]
        sidecars = [json.loads(group[side, ".json"]) for side in ("0", "1")]
        people = [metadata.get("speaker_id") for metadata in sidecars]
        identities = all(isinstance(person, str) and person for person in people) and len(set(people)) == 2
        equal_length = len(audio[0]) == len(audio[1])
        equal_pcm = np.array_equal(*audio)
        checks = {"distinct_sidecar_speakers": bool(identities), "equal_frames": equal_length,
                  "byte_equivalent_pcm": equal_pcm}
        if equal_length and not equal_pcm:
            tracks = np.stack([np.rint(resample_poly(track.astype(np.float32), RATE // 8000, 1))
                              .clip(-32768, 32767).astype(np.int16) for track in audio])
            activity = [detect(track) for track in tracks]
            checks.update({"seconds": len(audio[0]) / 8000, "silero_activity_segments": list(map(len, activity)),
                           "channel_isolation": isolation(tracks, activity)})
            if clock:
                from .nsc_part6_clock_probe import candidate_windows, direct_power, spectral_frames

                tracks_clock = []
                for track, spans, sidecar in zip(tracks, activity, sidecars, strict=True):
                    signal = spectral_frames(track)
                    own = own_intervals(sidecar, len(track))
                    tracks_clock.append((signal, candidate_windows(spans, len(track)),
                                         direct_power(signal[1], spans), own, len(track)))
                signals[len(pairs) + 1] = tracks_clock
        pairs.append(checks)
    clock_checks = compare_clock(signals) if clock else None
    return {"counts": counts, "pair_checks": pairs, **({"clock_checks": clock_checks} if clock else {}),
            "scope": "bounded original audio preflight; common clock, reuse terms and speaker semantics still pending"}


def compare_clock(signals):
    from .nsc_part4_native_clock import test_clock

    if set(signals) != {1, 2}:
        raise ValueError("the clock test requires the two expected complete pairs")
    mapped = {(pair, side): value for pair, tracks in signals.items()
              for side, value in enumerate(tracks, start=1)}
    report = []
    for pair, tracks in signals.items():
        result = test_clock([], mapped, pair)
        report.append({"sidecar_intervals": [len(track[3]) for track in tracks],
                       "passes": result["passes"], "nulls_pass": result["nulls_pass"],
                       "clock": result["clock"],
                       "coarse_envelope_offset": result["coarse_envelope_offset"],
                       "directions": [{"candidates": direction["candidates"],
                                       "matched": len(direction["matches"]),
                                       "anchors": len(direction["directional_anchors"]),
                                       "wrong_pair": direction["wrong_pair"],
                                       "wrong_time": direction["wrong_time"]}
                                      for direction in result["directions"]]})
    return report


def run(root, *, clock=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("IndicVoices audio stays on the campaign machine")
    source = load_sources()[SOURCE]
    report = {"archive_etag": source.get("archive_etag"), "range_bytes": PREFIX_BYTES,
              **inspect_pairs(read_members(fetch(source)), clock=clock)}
    name = f"{JOB}-clock" if clock else JOB
    write_once(root / "reports" / f"{name}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", action="store_true", help="look for mutual timing anchors and test wrong-time controls")
    run(data_root(), clock=parser.parse_args().clock)
