"""Compare AMI turn labels by overlapping pause intervals, without a decision time."""

from collections import Counter, defaultdict
import hashlib
import json
from zipfile import ZipFile

from .ami_calibration import _write_json
from .ami_pilot import annotated_pauses, group_words
from .ami_reference import _manual_spans, _manual_vote
from .audio import RATE
from .labels import ami_backchannels, ami_speakers, read_rows
from .progress import Progress
from .sources import data_root


def matched_floor_intervals(row, intervals):
    start, end = row.pause_start, row.pause_start + row.pause_samples
    matches = []
    for caller, following, label, backchannel in intervals:
        if caller.speaker == row.speaker and min(end, following.start) > max(start, caller.end):
            matches.append((caller, following, label, backchannel))
    return matches


def floor_vote(row, matches):
    """Vote only when one reference pause from the same caller overlaps the label."""
    if len(matches) != 1:
        return "no matching reference pause" if not matches else "multiple matching reference pauses"
    return "agrees" if matches[0][2] == row.label else "opposite label"


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI floor audit uses the training server")
    raw = root / "raw"
    manifest = raw / "ami" / "MANIFEST.json"
    label_file = root / "labels" / "ami.jsonl"
    windows = {entry["path"].split("/")[0]: (round(entry["start_s"] * RATE), round(entry["length_s"] * RATE))
               for entry in json.loads(manifest.read_text())["files"]}
    grouped = defaultdict(list)
    for row in read_rows(label_file):
        if row.split in {"train", "validation"}:
            grouped[row.recording].append(row)
    archive_path, = (raw / "ami_annotations").glob("*.zip")
    totals = defaultdict(Counter)
    by_label = defaultdict(lambda: defaultdict(Counter))
    comparisons = defaultdict(lambda: defaultdict(Counter))
    eligible = defaultdict(lambda: defaultdict(Counter))
    meter = Progress("ami-floor-audit", len(grouped), "meetings")
    try:
        with ZipFile(archive_path) as archive:
            speakers = ami_speakers(archive)
            for meeting, rows in sorted(grouped.items()):
                tracks = speakers[meeting]
                offset, length = windows[meeting]
                words = sorted(_manual_spans(archive, meeting, tracks, offset, length),
                               key=lambda span: (span.start, span.end, span.speaker))
                marked = {speaker: ami_backchannels(archive, meeting, agent, offset)
                          for speaker, agent in tracks.values()}
                intervals = list(annotated_pauses(group_words(words), marked, minimum_gap=1))
                for row in rows:
                    matches = matched_floor_intervals(row, intervals)
                    result = floor_vote(row, matches)
                    if result == "agrees":
                        caller, following, _, _ = matches[0]
                        for elapsed in (100, 200, 300):
                            if following.start - caller.end >= elapsed * RATE // 1000:
                                eligible[row.split][row.label][elapsed] += 1
                    strict, _ = _manual_vote(row, words, marked, 200 * 16, 250 * 16, None, None)
                    totals[row.split][result] += 1
                    by_label[row.split][row.label][result] += 1
                    comparisons[row.split][strict][result] += 1
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    report = root / "reports" / "ami-floor-audit.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report, {
        "scope": "original AMI train and validation labels; positive overlap with one reference pause from the same caller",
        "inputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (label_file, manifest, archive_path)},
        "floor_votes": totals, "by_proposed_label": by_label, "strict_timing_comparison": comparisons,
        "eligible_at_reference_boundary_ms": eligible,
    })
    return report


if __name__ == "__main__":
    print(run(data_root()))
