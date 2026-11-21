"""Turn labels from aligned per-speaker speech intervals."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from .progress import Progress
from .splits import assign, speaker_groups


@dataclass(frozen=True)
class Speech:
    speaker: str
    start: int
    end: int


@dataclass(frozen=True)
class Label:
    source: str
    language: str
    speaker: str
    recording: str
    group: str
    split: str
    pause_start: int
    pause_samples: int
    label: str
    method: str
    audio: str
    annotated_backchannel: bool = False
    silence_start: int | None = None


def turns(speech, decision_ms=200):
    """Yield (ending speech, available silence in samples, label)."""
    minimum = round(16 * decision_ms)
    intervals = []
    # Merge overlapping/touching intervals from the same track before applying
    # the overlap rule to other tracks.
    by_speaker = defaultdict(list)
    for span in sorted(speech, key=lambda s: (s.start, s.end)):
        if span.start < 0 or span.end <= span.start:
            raise ValueError(f"invalid speech interval {span}")
        track = by_speaker[span.speaker]
        if track and span.start <= track[-1].end:
            span = Speech(span.speaker, track[-1].start, max(track.pop().end, span.end))
        track.append(span)
    active_until = 0
    for span in sorted((s for track in by_speaker.values() for s in track),
                       key=lambda s: (s.start, s.end, s.speaker)):
        intervals.append(span)

    for index, span in enumerate(intervals):
        overlap_before_end = active_until > span.end
        active_until = max(active_until, span.end)
        next_index = index + 1
        if overlap_before_end or next_index == len(intervals):
            continue
        following = intervals[next_index]
        gap = following.start - span.end
        if gap < minimum:
            continue
        if following.speaker == span.speaker:
            yield span, gap, "hold"
            continue
        if following.end - following.start < 16000 and next_index + 1 < len(intervals):
            resumed = intervals[next_index + 1]
            if resumed.speaker == span.speaker:
                # The model is only queried during the silence before the interjection.
                yield span, gap, "hold"
                continue
        if gap >= 3200:
            yield span, gap, "shift"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            handle.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def read_rows(path):
    with path.open() as handle:
        return [Label(**json.loads(line)) for line in handle]


def _archive_member(archive, name):
    return next((p for p in archive.namelist() if p == name or p.endswith("/" + name)), None)


def ami_speakers(archive):
    xml = ET.fromstring(archive.read(_archive_member(archive, "corpusResources/meetings.xml")))
    return {
        meeting.attrib["observation"]: {
            int(speaker.attrib["channel"]): (speaker.attrib["global_name"], speaker.attrib["nxt_agent"])
            for speaker in meeting.findall("speaker")
        }
        for meeting in xml.iter("meeting")
    }


def ami_backchannels(archive, meeting, agent, offset):
    """Map the explicitly marked AMI backchannel dialogue acts to slice samples."""
    stem = f"{meeting}.{agent}"
    acts = _archive_member(archive, f"dialogueActs/{stem}.dialog-act.xml")
    words = _archive_member(archive, f"words/{stem}.words.xml")
    if not acts or not words:
        return []
    def attr(element, key):
        return next((value for name, value in element.attrib.items()
                     if name.rsplit("}", 1)[-1] == key), "")
    starts = {}
    for word in ET.fromstring(archive.read(words)):
        if "starttime" in word.attrib:
            starts[attr(word, "id")] = round(float(word.attrib["starttime"]) * 16000) - offset
    return [
        starts[match[1]]
        for act in ET.fromstring(archive.read(acts))
        if any("id(ami_da_1)" in attr(child, "href") for child in act)
        for child in act
        if (match := re.search(r"#id\(([^)]+)\)", attr(child, "href"))) and match[1] in starts
    ]


def ami_labels(root, vad):
    raw = root / "raw" / "ami"
    files = json.loads((raw / "MANIFEST.json").read_text())["files"]
    archive_path, = (root / "raw" / "ami_annotations").glob("*.zip")
    meetings = defaultdict(list)
    for entry in files:
        meetings[entry["path"].split("/")[0]].append(entry)

    with ZipFile(archive_path) as archive:
        speakers = ami_speakers(archive)
        groups = speaker_groups((meeting, (s for s, _ in mapping.values()))
                                for meeting, mapping in speakers.items())
        meter = Progress("label-ami", len(meetings), "meetings")
        try:
            for meeting, tracks in sorted(meetings.items()):
                expected = speakers[meeting]
                paths, spans, backchannels = {}, [], defaultdict(list)
                offsets = {int(entry["start_s"] * 16000) for entry in tracks}
                if len(offsets) != 1:
                    raise ValueError(f"{meeting}: headset slices are misaligned")
                offset, = offsets
                channel_paths = {
                    int(Path(entry["path"]).stem.rsplit("Headset-", 1)[1]): raw / entry["path"]
                    for entry in tracks
                }
                missing = expected.keys() - channel_paths.keys()
                if missing:
                    raise ValueError(f"{meeting}: missing annotated headset channels {sorted(missing)}")
                extra = channel_paths.keys() - expected.keys()
                if any(vad(channel_paths[channel]) for channel in sorted(extra)):
                    print(f"{meeting}: skipping meeting with speech on an unannotated headset", flush=True)
                    meter.advance(1)
                    continue
                for channel, (speaker, agent) in sorted(expected.items()):
                    path = channel_paths[channel]
                    paths[speaker] = str(path.relative_to(root))
                    spans.extend(Speech(speaker, start, end) for start, end in vad(path))
                    for sample in ami_backchannels(archive, meeting, agent, offset):
                        backchannels[speaker].append(sample)
                group = groups[meeting]
                split = assign("ami", group, "train-conversation")
                for span, gap, label in turns(spans):
                    marked = any(
                        any(span.end <= sample <= span.end + gap + 3200 for sample in samples)
                        for speaker, samples in backchannels.items() if speaker != span.speaker
                    )
                    yield Label("ami", "en", span.speaker, meeting, group, split, span.end,
                                gap, label, "per-track", paths[span.speaker], marked)
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()
