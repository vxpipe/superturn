"""Bounded, fit-only Basis label audit on synchronized tracks in memory."""

import json
import os
import resource
from collections import Counter
from pathlib import Path, PurePosixPath

from .ami_calibration import _write_json
from .audio import RATE, flac_streaminfo, read_audio, silero
from .basis_stage0_yield import json_lines
from .basis_split import independent_audit, voices
from .progress import Progress
from .yodas_mono_diarization import disjoint_bouts

LANGUAGES = ("en", "hi")
MAX_AUDIO_BYTES = 120_000_000


def events(activity, connected, muted, length):
    """Abstain on overlaps, short interjections, mutes and track transitions."""
    groups = list(disjoint_bouts(sorted(activity)))
    counts, proposed = Counter(), []
    for (_, last), (first, _) in zip(groups, groups[1:]):
        if len(last) != 1 or len(first) != 1:
            continue
        start, stop, focal = last[0]
        onset, end, next_voice = first[0]
        if onset - stop < 4800 or stop < 3 * RATE or onset + 2 * RATE > length:
            continue
        left, right = stop - 3 * RATE, onset + 2 * RATE
        if any(a < right and b > left for a, b in muted):
            counts["muted"] += 1
            continue
        active_people = {voice for voice, spans in connected.items()
                         if any(a < right and b > left for a, b in spans)}
        if (focal not in active_people or next_voice not in active_people
                or any(not any(a <= left and b >= right for a, b in connected[voice])
                       for voice in active_people)):
            counts["track_transition"] += 1
            continue
        if any(a < right_ and b > left_ and voice != owner
               for left_, right_, owner in ((max(start, stop - 1600), stop, focal),
                                             (onset, min(end, onset + RATE), next_voice))
               for a, b, voice in activity):
            counts["overlap"] += 1
            continue
        if next_voice != focal and end - onset < RATE:
            counts["short_other"] += 1
            continue
        label = "hold" if focal == next_voice else "shift"
        counts[label] += 1
        proposed.append((stop, onset, label))
    return counts, proposed


def samples(spans):
    return [(round(span["start"] * RATE), round(span["end"] * RATE)) for span in spans]


def read_segment(fs, prefix, language, row, meter):
    tracks = {}
    frames = set()
    for speaker, path in sorted(row["tracks"].items()):
        relative = PurePosixPath(path)
        if (len(relative.parts) != 4 or relative.parts[:3] != (language, "audio", row["conversation_id"])
                or not relative.name.endswith(f".{speaker}.flac")):
            raise ValueError("Basis sidecar references a track outside the reserved conversation")
        full = f"{prefix}/{path}"
        size = fs.info(full)["size"]
        if not 0 < size <= meter.total - meter.done:
            raise ValueError("Basis track exceeds the predeclared audio cap")
        with fs.open(full, "rb") as handle:
            payload = handle.read(size + 1)
        if len(payload) != size:
            raise ValueError("Basis track length changed")
        meter.advance(size)
        rate, channels, depth, count, _ = flac_streaminfo(payload[:42])
        if (rate, channels, depth) != (48000, 1, 16) or count == 0:
            raise ValueError("unexpected Basis native audio format")
        frames.add(count)
        tracks[speaker] = read_audio(payload)
        if abs(len(tracks[speaker]) - count / 3) > 1:
            raise ValueError("Basis resampling changed observed track duration")
    if len(frames) != 1 or set(tracks) != {person["speaker"] for person in row["speakers"]}:
        raise ValueError("Basis tracks do not match the synchronized segment roster")
    return tracks


def run_pilot(root, source, membership, calls, digest):
    from huggingface_hub import HfFileSystem
    from huggingface_hub.utils import HfHubHTTPError

    cgroup = Path("/sys/fs/cgroup")
    if any((cgroup / key).read_text().strip() != "0" for key in ("memory.swap.current", "memory.swap.max")):
        raise ValueError("Basis audit requires a swap-disabled campaign container")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    fs = HfFileSystem(token=os.environ["HF_TOKEN"])
    prefix = f'datasets/{source.get("repo")}@{source.get("revision")}'
    detect = silero()
    output = {"scope": "fit-only acquisition and label-rule pilot",
              "membership_sha256": digest, "languages": {}}
    planned = {}
    for language in LANGUAGES:
        fit = set(membership[language]["fit"])
        disjoint = independent_audit([call for call in calls[language] if call.name in fit])
        planned[language] = disjoint[:2]
    budget = sum(call.size for group in planned.values() for call in group)
    if not 0 < budget <= MAX_AUDIO_BYTES or any(len(group) < 2 for group in planned.values()):
        raise ValueError("fit-only audit cannot form its bounded four-call plan")
    audio = Progress("basis-fit-audit-audio", budget, "B")
    try:
        for language, group in planned.items():
            wanted = {call.name for call in group}
            path = f"{prefix}/{language}/segments.jsonl"
            size = fs.info(path)["size"]
            if not 0 < size < 200_000_000:
                raise ValueError("Basis audit metadata exceeded its input cap")
            metadata = Progress(f"basis-audit-{language}-metadata", size, "B")
            totals, seen = Counter(), set()
            with fs.open(path, "rb", block_size=8 * 1024 * 1024) as handle:
                for row in json_lines(handle, metadata):
                    if row["conversation_id"] not in wanted:
                        continue
                    seen.add(row["conversation_id"])
                    tracks = read_segment(fs, prefix, language, row, audio)
                    activity = [(start, stop, voice) for voice, pcm in tracks.items() for start, stop in detect(pcm)]
                    connected = {person["speaker"]: samples(person["on_call"]) for person in row["speakers"]}
                    counts, _ = events(activity, connected, samples(row["muted_spans"]),
                                       min(len(pcm) for pcm in tracks.values()))
                    totals.update(counts)
            metadata.finish()
            if seen != wanted:
                raise ValueError("Basis audit did not reconstruct all reserved fit calls")
            output["languages"][language] = {"calls": len(group), "roster": len(voices(group)),
                                             "automatic_proposals": dict(totals)}
        if audio.done != budget:
            raise ValueError("Basis audit track bytes do not match the sealed plan")
    except HfHubHTTPError:
        audio.finish("failed")
        raise RuntimeError("Basis fit-audit transfer failed") from None
    except BaseException:
        audio.finish("failed")
        raise
    audio.finish()
    _write_json(root / "reports/basis-stage1-fit-pilot.json", output)
    print(json.dumps(output, indent=2))
