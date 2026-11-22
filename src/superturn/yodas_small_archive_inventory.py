"""Compare small pinned YODAS archives using captions and durations only."""

from collections import Counter
import hashlib
import json
from urllib.request import urlopen

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .progress import Progress
from .sources import data_root
from .yodas_caption_overlap import CAPTION_ID, candidates, development, text_payload, tokens
from .yodas_segment_inventory import LANGUAGE_BYTES, REPOSITORY, REVISION

JOB = "yodas-small-archive-inventory"
TOP = 3
METADATA_CAP = 8_000_000


def lookup(language, kind):
    relative = f"data/{language}000/{kind}"
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/{relative}"
    with urlopen(url, timeout=30) as response:
        entries = json.load(response)
    return {row["path"].rsplit("/", 1)[-1].split(".")[0]: row
            for row in entries if row["type"] == "file"}


def archive_options(lang):
    audio, text, duration = (lookup(lang, kind) for kind in ("audio", "text", "duration"))
    common = sorted(audio.keys() & text.keys() & duration.keys(), key=lambda key: (audio[key]["size"], key))
    return [(key, audio[key], text[key], duration[key]) for key in common[:TOP]]


def video_counts(text, duration, excluded):
    caption_ids = {identifier for line in text.decode("utf-8").splitlines()
                   if (fields := line.split(maxsplit=1)) and len(fields) == 2
                   and (identifier := fields[0]) and tokens(fields[1])}
    counts = Counter()
    for line in duration.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] not in caption_ids:
            continue
        key, seconds = fields
        match = CAPTION_ID.fullmatch(key)
        if match and match.group(1) not in excluded and 5 <= float(seconds) <= 30:
            counts[match.group(1)] += 1
    return counts


def run(root):
    from .multilingual_teacher_probe import DATA

    if not (root / ".campaign-machine").is_file():
        raise ValueError("YODAS metadata stays on Vast")
    plan = {lang: archive_options(lang) for lang in LANGUAGE_BYTES}
    if any(len(rows) != TOP for rows in plan.values()):
        raise ValueError("expected three pinned archive options in each language")
    cost = sum(text["size"] + duration["size"] for rows in plan.values()
               for _, _, text, duration in rows)
    if cost > METADATA_CAP:
        raise ValueError("small-archive census exceeds its metadata budget")
    output = root / "manifests" / JOB
    output.mkdir(parents=True, exist_ok=True)
    meter = Progress(JOB, sum(map(len, plan.values())), "sidecar pairs")
    summaries, membership = {}, {}
    try:
        for lang, options in plan.items():
            path = root / f"raw/livekit_multilingual_diagnostic/data/{lang}/validation-00000-of-00001.parquet"
            if file_digest(path) != DATA[lang]:
                raise ValueError("development exclusion text changed")
            dev, coverage = development(path)
            receipt, by_video = [], {}
            for key, audio, text, duration in options:
                captions = text_payload(output / f"{lang}-{key}-text.txt", text["path"], text["size"])
                timings = text_payload(output / f"{lang}-{key}-duration.txt", duration["path"], duration["size"])
                row, excluded = candidates(captions, timings, dev)
                counts = video_counts(captions, timings, set(excluded))
                if len(counts) != row["videos_after_text_screen"]:
                    raise ValueError("video membership differs from overlap counts")
                by_video[key] = {"eligible_video_counts": dict(counts), "excluded_videos": excluded}
                receipt.append({"shard": key, "audio_path": audio["path"], "archive_bytes": audio["size"],
                                "archive_sha256": audio["lfs"]["oid"], "text_path": text["path"],
                                "duration_path": duration["path"], "text_sha256": hashlib.sha256(captions).hexdigest(),
                                "duration_sha256": hashlib.sha256(timings).hexdigest(),
                                "segments_at_twenty_per_video": sum(min(20, amount) for amount in counts.values()),
                                **row})
                meter.advance(1)
            summaries[lang] = {**coverage, "options": receipt,
                               "distinct_videos_across_three": len({video for row in by_video.values()
                                                                    for video in row["eligible_video_counts"]})}
            membership[lang] = by_video
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    membership_path = output / "membership.json"
    _write_json(membership_path, membership)
    report = {"scope": "metadata-only; voice identities and audio are not screened",
              "revision": REVISION, "license": "CC BY 3.0", "audio_bytes": 0, "metadata_bytes": cost,
              "membership_sha256": file_digest(membership_path), "languages": summaries}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
