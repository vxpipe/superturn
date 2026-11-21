"""Count ICSI speaker components and provisional pause gaps from original MRT files."""

from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import groupby
import json
import math
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .acquire import Plan, PlannedFile, _request, _write_manifest
from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root, load_sources
from .splits import speaker_groups

SOURCE = "icsi-transcript-census"
MEETING = re.compile(r"transcripts/(B[A-Za-z0-9]+)\.mrt")
CHANNEL = re.compile(r"chan[0-9]+\.sph")


def archive(root, source):
    path = root / "raw" / SOURCE / "ICSI_original_transcripts.zip"
    limit = source.get("max_total_bytes")
    if path.is_file():
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    else:
        with _request(source.get("url")) as response:
            data = response.read(limit + 1)
    if len(data) > limit or sha256(data).hexdigest() != source.get("sha256"):
        raise ValueError("ICSI transcription archive differs from its receipt")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    if not (path.parent / "MANIFEST.json").is_file():
        planned = PlannedFile(path.name, len(data), url=source.get("url"))
        _write_manifest(Plan(source, [planned]), path.parent, {
            path.name: {"bytes": len(data), "sha256": source.get("sha256")},
        })
    return data


def count_gaps(spans):
    counts = Counter()
    active_end, ending = -1, set()
    ambiguous = 0
    for start, cluster in groupby(sorted(spans), key=lambda span: span[0]):
        same_onset = list(cluster)
        gap = start - active_end
        if active_end >= 0 and 300 <= gap <= 3000:
            starting = {(speaker, usable) for _, _, speaker, usable in same_onset}
            if len(ending) == len(starting) == 1:
                (before, before_usable), = ending
                (after, after_usable), = starting
                if before_usable and after_usable:
                    counts[before, "hold" if before == after else "shift"] += 1
            else:
                ambiguous += 1
        for _, end, speaker, usable in same_onset:
            if end > active_end:
                active_end, ending = end, set()
            if end == active_end:
                ending.add((speaker, usable))
    return counts, ambiguous


def inspect_meeting(data, meeting):
    root = ET.fromstring(data)
    if root.get("Session") != meeting:
        raise ValueError("ICSI meeting filename and session differ")
    listed = {node.get("Name"): node.get("AudioFile", "") for node in root.findall("./Preamble/Channels/Channel")}
    people = root.findall("./Preamble/Participants/Participant")
    attendees = {person.attrib["Name"] for person in people}
    channels = {person.attrib["Name"]: listed.get(person.get("Channel"), "") for person in people}
    shared = Counter(name for name in channels.values() if CHANNEL.fullmatch(name))
    channels = {person: name for person, name in channels.items() if shared[name] == 1}
    if len(attendees) != len(people):
        raise ValueError("duplicate participant")
    transcript = root.find("./Transcript")
    if transcript is None:
        raise ValueError("ICSI meeting has no transcript")
    ceiling = float(transcript.attrib["EndTime"])
    spans, skipped = [], Counter()
    for node in transcript.iter("Segment"):
        start, end = float(node.attrib["StartTime"]), float(node.attrib["EndTime"])
        speaker = node.get("Participant")
        if not math.isfinite(ceiling) or not 0 <= start < end <= ceiling + .05:
            skipped["invalid_time"] += 1
            continue
        usable = speaker in channels and "DigitTask" not in node.attrib
        if not usable:
            skipped["no_individual_track_or_digit_task"] += 1
        spans.append((round(start * 1000), round(end * 1000), speaker or "", usable))
    counts, ambiguous = count_gaps(spans)
    return {"attendees": sorted(attendees), "channels": channels, "annotation_segments": len(spans),
            "shared_close_microphones": sum(count > 1 for count in shared.values()),
            "ambiguous_gap_boundary": ambiguous,
            "ignored_as_anchor": skipped, "untimed_internal_pause_tags": len(transcript.findall(".//Pause"))}, counts


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("ICSI source data stays on Vast")
    source = load_sources()[SOURCE]
    data = archive(root, source)
    meetings, gaps = {}, Counter()
    with ZipFile(BytesIO(data)) as zip_file:
        members = [(match[1], name) for name in sorted(zip_file.namelist()) if (match := MEETING.fullmatch(name))]
        if len(members) != 75 or len({meeting for meeting, _ in members}) != 75:
            raise ValueError("expected the complete ICSI meeting roster")
        meter = Progress("icsi-transcript-roster", len(members), "meetings")
        try:
            for meeting, name in members:
                meetings[meeting], counts = inspect_meeting(zip_file.read(name), meeting)
                gaps.update(counts)
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
    groups = speaker_groups((meeting, row["attendees"]) for meeting, row in meetings.items())
    components = Counter(groups.values())
    visits = Counter(person for row in meetings.values() for person in row["attendees"])
    mapped = {person for row in meetings.values() for person in row["channels"]}
    for meeting, group in groups.items():
        meetings[meeting]["component"] = group
    summary = {
        "scope": "original transcript and microphone metadata census; no acoustic pause labels",
        "source_sha256": source.get("sha256"), "archive_bytes": len(data), "audio_payload_bytes": 0,
        "meetings": len(meetings), "participants": len(visits), "mapped_participants": len(mapped),
        "participants_in_multiple_meetings": sum(count > 1 for count in visits.values()),
        "maximum_meetings_for_one_participant": max(visits.values()),
        "component_meeting_counts": sorted(components.values(), reverse=True),
        "shared_close_microphones": sum(row["shared_close_microphones"] for row in meetings.values()),
        "ambiguous_gap_boundaries": sum(row["ambiguous_gap_boundary"] for row in meetings.values()),
        "segment_final_gap_range_seconds": [.3, 3.],
        "provisional_segment_final_candidates": {
            label: sum(gaps[person, label] for person in mapped) for label in ("hold", "shift")
        },
        "participants_with_20_segment_final_gaps": sum(gaps[p, "hold"] + gaps[p, "shift"] >= 20 for p in mapped),
        "participants_with_20_each_class": sum(min(gaps[p, "hold"], gaps[p, "shift"]) >= 20 for p in mapped),
        "untimed_internal_pause_tags": sum(row["untimed_internal_pause_tags"] for row in meetings.values()),
    }
    _write_json(root / "reports/icsi-roster.json", {**summary, "meeting_index": meetings})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(data_root())
