"""Compare AMI VAD labels with the aligned human transcript and dialogue acts."""

from bisect import bisect_left
from collections import Counter, defaultdict
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .labels import Speech, _archive_member, ami_speakers, read_rows, turns
from .progress import Progress


def _manual_spans(archive, meeting, speakers, offset, length, *, tags=("w",)):
    for speaker, agent in speakers.values():
        member = _archive_member(archive, f"words/{meeting}.{agent}.words.xml")
        if member is None:
            continue
        for word in ET.fromstring(archive.read(member)):
            if word.tag.rsplit("}", 1)[-1] not in tags or not {"starttime", "endtime"} <= word.attrib.keys():
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
            "reference": "word timestamps from AMI's force-aligned human transcripts",
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


def _manual_vote(row, spans, annotated, end_tolerance, onset_tolerance, errors, pause_limits):
    edge = row.pause_start
    caller_end = min((span.end for span in spans if span.speaker == row.speaker),
                     key=lambda end: abs(end - edge), default=None)
    events = [span for span in spans if span.start >= edge + 1280]
    if errors is not None and caller_end is not None and events:
        delta_end = caller_end - edge
        delta_onset = events[0].start - edge - row.pause_samples
        if max(abs(delta_end), abs(delta_onset)) <= 9600:
            errors.append((delta_end / 16, delta_onset / 16))
    if caller_end is None or abs(caller_end - edge) > end_tolerance:
        return "unresolved", "no caller word end within timing tolerance"
    speech_in_pause = sum(max(0, min(span.end, edge + 2880) - max(span.start, edge + 1280))
                          for span in spans)
    if speech_in_pause >= 640:
        return "contradicted", "speech inside the first 200 ms pause"
    if not events:
        return "unresolved", "no later annotated word"
    if abs(events[0].start - edge - row.pause_samples) > onset_tolerance:
        return "unresolved", "next word exceeds the onset timing tolerance"
    first = events[0]
    if first.speaker == row.speaker:
        expected = "hold"
    else:
        words = [first]
        for span in events[1:]:
            if span.speaker != first.speaker or span.start - words[-1].end > 3200:
                break
            words.append(span)
        marked = any(abs(point - first.start) <= 3200 for point in annotated[first.speaker])
        if marked:
            continuation = next((span for span in events if span.speaker != first.speaker), None)
            if continuation is None or continuation.start - words[-1].end > 3 * 16000:
                return "unresolved", "continuation after annotated backchannel is unclear"
            expected = "hold" if continuation.speaker == row.speaker else "shift"
        elif words[-1].end - first.start < 16000:
            # A short response may be an acknowledgement without a lexical tag.
            return "unresolved", "short other-speaker response without backchannel annotation"
        else:
            expected = "shift"
    if expected == row.label:
        if pause_limits is not None:
            pause_limits[row] = min(row.pause_samples, first.start - edge)
        return "supported", "timing and speaker order agree"
    return "contradicted", f"manual speaker order implies {expected} for {row.label} label"


def audit_sample(root, rows, *, explain=False, end_ms=200, onset_ms=250, errors=None,
                 pause_limits=None):
    """Check every sampled AMI label against raw words and dialogue acts."""
    from .labels import ami_backchannels

    raw = root / "raw"
    tracks = json.loads((raw / "ami" / "MANIFEST.json").read_text())["files"]
    windows = {track["path"].split("/")[0]: (int(track["start_s"] * 16000), int(track["length_s"] * 16000))
               for track in tracks}
    selected = defaultdict(list)
    for row in rows:
        selected[row.recording].append(row)
    archive_path, = (raw / "ami_annotations").glob("*.zip")
    results = {}
    with ZipFile(archive_path) as archive:
        speakers = ami_speakers(archive)
        for meeting, examples in selected.items():
            offset, length = windows[meeting]
            tracks = speakers[meeting]
            spans = sorted(_manual_spans(archive, meeting, tracks, offset, length),
                           key=lambda span: (span.start, span.end, span.speaker))
            annotated = {speaker: ami_backchannels(archive, meeting, agent, offset)
                         for speaker, agent in tracks.values()}
            for row in examples:
                verdict = _manual_vote(row, spans, annotated, end_ms * 16, onset_ms * 16,
                                       errors, pause_limits)
                results[row] = verdict if explain else verdict[0]
    return results
