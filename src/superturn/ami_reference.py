"""Compare AMI's VAD labels with manually timestamped word intervals."""

from bisect import bisect_left
from collections import Counter, defaultdict
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .labels import Speech, _archive_member, ami_speakers, read_rows, turns
from .progress import Progress


def _manual_spans(archive, meeting, speakers, offset, length):
    for speaker, agent in speakers.values():
        member = _archive_member(archive, f"words/{meeting}.{agent}.words.xml")
        if member is None:
            continue
        for word in ET.fromstring(archive.read(member)):
            if word.tag.rsplit("}", 1)[-1] != "w" or not {"starttime", "endtime"} <= word.attrib.keys():
                continue
            start = max(0, round(float(word.attrib["starttime"]) * 16000) - offset)
            end = min(length, round(float(word.attrib["endtime"]) * 16000) - offset)
            if end > start:
                yield Speech(speaker, start, end)


def check_ami_reference(root, tolerance_ms=200):
    raw = root / "raw"
    tracks = json.loads((raw / "ami" / "MANIFEST.json").read_text())["files"]
    windows = {}
    for track in tracks:
        meeting = track["path"].split("/", 1)[0]
        windows[meeting] = int(track["start_s"] * 16000), int(track["length_s"] * 16000)
    labels = defaultdict(list)
    for row in read_rows(root / "labels" / "ami.jsonl"):
        labels[row.recording].append(row)
    path, = (raw / "ami_annotations").glob("*.zip")
    meter = Progress("ami-reference", len(labels), "meetings")
    counts = defaultdict(Counter)
    tolerance = tolerance_ms * 16
    try:
        with ZipFile(path) as archive:
            speakers = ami_speakers(archive)
            for meeting, observed in sorted(labels.items()):
                offset, length = windows[meeting]
                reference = defaultdict(list)
                for span, _, label in turns(_manual_spans(archive, meeting, speakers[meeting], offset, length)):
                    reference[span.speaker].append((span.end, label))
                for speaker, entries in reference.items():
                    reference[speaker] = sorted(entries)
                for row in observed:
                    entries = reference[row.speaker]
                    i = bisect_left(entries, (row.pause_start, ""))
                    nearest = min(entries[max(0, i - 1):i + 1],
                                  key=lambda pair: abs(pair[0] - row.pause_start), default=None)
                    for group in (row.split, "all"):
                        stats = counts[group]
                        stats["labelled"] += 1
                        if row.annotated_backchannel:
                            stats[f"{row.label}_with_annotated_backchannel"] += 1
                        if nearest is not None and abs(nearest[0] - row.pause_start) <= tolerance:
                            stats["matched"] += 1
                            stats["agreed"] += nearest[1] == row.label
                meter.advance(1)
        report = {
            "reference": "AMI manual word timestamps",
            "tolerance_ms": tolerance_ms,
            "groups": {
                group: {**stats, "agreement_on_matched": stats["agreed"] / stats["matched"] if stats["matched"] else None}
                for group, stats in sorted(counts.items())
            },
        }
        destination = root / "reports" / "ami-reference.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination
