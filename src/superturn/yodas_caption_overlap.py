"""Screen a pinned YODAS caption sample for reuse of development utterances."""

from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .progress import Progress
from .sources import data_root
from .yodas_segment_inventory import REPOSITORY, REVISION

JOB = "yodas-caption-overlap"
TEXT_BYTES = {"ar": 1006321, "hi": 1176700, "id": 617420, "nl": 836131, "pt": 695396, "tr": 886401}
CAPTION_ID = re.compile(r"^([A-Za-z0-9_-]{11})-(\d+)-(\d+)-(\d+)$")
NGRAM = 8


def tokens(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple("".join(char if unicodedata.category(char)[0] in "LMN" else " "
                         for char in normalized).split())


def grams(words):
    return (words[index:index + NGRAM] for index in range(len(words) - NGRAM + 1))


def development(path):
    source = pq.ParquetFile(path)
    counts = Counter()
    seen = set()
    for group in source.iter_batches(batch_size=32, columns=["words", "messages"]):
        for row in group.to_pylist():
            counts["turns"] += 1
            words = sorted((word for word in row["words"] or [] if word["word"] and word["start"] is not None),
                           key=lambda word: word["start"])
            lines = [" ".join(word["word"] for word in words)]
            lines.extend(msg["content"] for msg in row["messages"] or [] if isinstance(msg["content"], str))
            strings = set(gram for item in lines for gram in grams(tokens(item)))
            counts["turns_with_eight_words"] += bool(strings)
            seen.update(strings)
    return seen, dict(counts)


def text_payload(path, relative, size):
    if not path.exists():
        request = Request(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{relative}",
                          headers={"Accept-Encoding": "identity"})
        with urlopen(request, timeout=30) as reply:
            payload = reply.read(size + 1)
        if len(payload) != size:
            raise ValueError("caption index differs from pinned size")
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(path)
    body = path.read_bytes()
    if len(body) != size:
        raise ValueError("cached caption index differs from pinned size")
    return body


def candidates(payload, durations, reference):
    tracks = defaultdict(dict)
    stats = Counter()
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        stats["caption_rows"] += 1
        fields = line.split(maxsplit=1)
        match = CAPTION_ID.fullmatch(fields[0])
        if not match:
            stats["invalid_caption_keys"] += 1
            continue
        video, track, start, end = match.groups()
        position = int(start), int(end)
        words = tokens(fields[1]) if len(fields) > 1 else ()
        tracks[video, track][position] = words
    blocked = set()
    for (video, _), captions in tracks.items():
        previous, tail = None, ()
        for (start, end), words in sorted(captions.items()):
            # Adjacent non-overlapping captions, ignoring gaps over one second.
            window = tail + words if previous is not None and 0 <= start - previous <= 100 else words
            if any(gram in reference for gram in grams(window)):
                blocked.add(video)
                stats["matching_caption_windows"] += 1
            previous, tail = end, window[-(NGRAM - 1):]
    available, eligible = Counter(), Counter()
    for line in durations.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        key, seconds = fields
        match = CAPTION_ID.fullmatch(key)
        if match is None or not 5 <= float(seconds) <= 30:
            continue
        video, track, start, end = match.groups()
        if not tracks.get((video, track), {}).get((int(start), int(end))):
            stats["duration_rows_without_caption"] += 1
            continue
        available[video] += 1
        if video not in blocked:
            eligible[video] += 1
    stats.update({"matched_videos": len(blocked), "videos_with_eligible_captioned_duration": len(available),
                  "videos_after_text_screen": len(eligible), "max_segments_per_video_after_screen": max(
                      eligible.values(), default=0)})
    return dict(stats), sorted(blocked)


def run(root):
    from .multilingual_teacher_probe import DATA

    if not (root / ".campaign-machine").is_file():
        raise ValueError("caption metadata and development text stay on Vast")
    output = root / "manifests" / JOB
    output.mkdir(parents=True, exist_ok=True)
    dur_report = json.loads((root / "reports/yodas-segment-inventory.json").read_text())
    meter = Progress(JOB, len(TEXT_BYTES), "languages")
    reports, exclusions = {}, {}
    try:
        for language, size in TEXT_BYTES.items():
            duration = (root / "manifests/yodas-segment-inventory" / f"{language}-durations.txt").read_bytes()
            if hashlib.sha256(duration).hexdigest() != dur_report["languages"][language]["sha256"]:
                raise ValueError("duration selection changed")
            file = root / f"raw/livekit_multilingual_diagnostic/data/{language}/validation-00000-of-00001.parquet"
            if file_digest(file) != DATA[language]:
                raise ValueError("development cohort changed")
            reference, coverage = development(file)
            relative = f"data/{language}000/text/00000000.txt"
            payload = text_payload(output / f"{language}-captions.txt", relative, size)
            stats, excluded = candidates(payload, duration, reference)
            exclusions[language] = excluded
            reports[language] = {**coverage, "reference_grams": len(reference), "caption_path": relative,
                                 "caption_sha256": hashlib.sha256(payload).hexdigest(), **stats}
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    excluded_path = output / "excluded-videos.json"
    _write_json(excluded_path, exclusions)
    report = {"scope": "eight-token exact overlap screen; no audio fetched, no speaker overlap claim",
              "revision": REVISION, "caption_bytes": sum(TEXT_BYTES.values()), "audio_bytes": 0,
              "exclusion_sha256": file_digest(excluded_path), "languages": reports}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
