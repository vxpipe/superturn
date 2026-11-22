"""Check whether existing fit-side single-speaker pauses have words on both sides."""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, read_audio
from .dense_teacher import file_digest
from .progress import Progress
from .shards import seed_for
from .sources import data_root

JOB = "natural-hold-preflight"
PER_LANGUAGE = 8
TOLERANCE = round(.160 * RATE)
CHECK_SHIFT = round(.700 * RATE)


def flank_residuals(words, edge, onset):
    endings = [end - edge for start, end in words if (start + end) // 2 < edge]
    starts = [start - onset for start, end in words if (start + end) // 2 >= onset]
    return min(endings, key=abs, default=None), min(starts, key=abs, default=None)


def has_flanks(words, edge, onset):
    return all(value is not None and abs(value) <= TOLERANCE for value in flank_residuals(words, edge, onset))


def select(rows):
    strata = defaultdict(list)
    for row in rows:
        if (row["role"] == "fit" and row["source"] in {"emilia_yodas", "voxpopuli"}
                and row["label"] == "hold" and row["method"] == "single-track"
                and row["pause_start"] >= round(2.7 * RATE)
                and row["silence_start"] in (None, row["pause_start"])):
            strata[row["source"], row["language"]].append(row)
    selected = []
    for _, items in sorted(strata.items()):
        speakers = set()
        for row in sorted(items, key=lambda item: seed_for(JOB, item["audio"], item["pause_start"])):
            if row["speaker"] in speakers:
                continue
            selected.append(row)
            speakers.add(row["speaker"])
            if len(speakers) == PER_LANGUAGE:
                break
    return selected


def run(root):
    import torch
    import whisper

    from .open_speech_teacher_probe import WHISPER_SHA256

    if not (root / ".campaign-machine").is_file():
        raise ValueError("real audio stays on Vast")
    pool = root / "shards/dense-teacher/pool.json"
    inventory = json.loads((root / "reports/dense-teacher-inventory.json").read_text())
    encoded = pool.read_bytes()
    pool_sha = hashlib.sha256(encoded).hexdigest()
    if pool_sha != inventory["pool_sha256"]:
        raise ValueError("the isolated fit pool changed")
    candidates = select(json.loads(encoded))
    if not candidates:
        raise ValueError("no existing candidate holds")
    weight = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(weight) != WHISPER_SHA256:
        raise ValueError("the archived Whisper weights changed")
    torch.set_num_threads(2)
    model = whisper.load_model(str(weight), device="cuda")
    meter = Progress(JOB, len(candidates), "pauses")
    counts = defaultdict(Counter)
    timing = defaultdict(lambda: defaultdict(list))
    receipt = []
    try:
        for row in candidates:
            group = row["source"], row["language"]
            counts[group]["candidates"] += 1
            audio = read_audio(root / row["audio"])
            edge = row["pause_start"]
            onset = edge + row["pause_samples"]
            left, right = max(0, edge - 3 * RATE), min(len(audio), onset + 3 * RATE)
            item = {"source": row["source"], "language": row["language"], "speaker": row["speaker"],
                    "group": row["connected_group"], "audio": row["audio"],
                    "edge": edge, "onset": onset,
                    "rendered_pcm_sha256": hashlib.sha256(audio[left:right].astype("<i2").tobytes()).hexdigest()}
            if onset + RATE // 2 <= right and 4800 <= onset - edge <= 4 * RATE:
                decoded = model.transcribe(audio[left:right].astype(np.float32) / 32768,
                                           language=row["language"], word_timestamps=True,
                                           condition_on_previous_text=False, temperature=0, verbose=None)
                words = [(left + round(word["start"] * RATE), left + round(word["end"] * RATE))
                         for segment in decoded["segments"] for word in segment.get("words", ())
                         if word["probability"] >= .5 and word["end"] > word["start"]
                         and any(char.isalpha() for char in word["word"])]
                for label, shift in (("observed", 0), ("null_early", -CHECK_SHIFT), ("null_late", CHECK_SHIFT)):
                    item[label] = has_flanks(words, edge + shift, onset + shift)
                    counts[group][label] += item[label]
                for side, value in zip(("offset", "resume"), flank_residuals(words, edge, onset), strict=True):
                    if value is not None:
                        timing[group][side].append(1000 * value / RATE)
                    counts[group][f"matched_{side}"] += value is not None and abs(value) <= TOLERANCE
                counts[group]["word_enters_first_300ms"] += any(
                    begin < edge + round(.300 * RATE) and end > edge for begin, end in words)
                counts[group]["scored"] += 1
            else:
                counts[group]["insufficient_context"] += 1
            receipt.append(item)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    directory = root / "runs" / JOB
    directory.mkdir(parents=True, exist_ok=True)
    membership = directory / "membership.json"
    if membership.is_file() and json.loads(membership.read_text()) != receipt:
        raise ValueError("the archived within-segment preflight changed")
    _write_json(membership, receipt)
    report = {"scope": "speech-flank evidence only; no admitted labels or teacher training",
              "pool_sha256": pool_sha, "whisper_sha256": WHISPER_SHA256,
              "membership_sha256": file_digest(membership),
              "membership_size": len(receipt), "timing_tolerance_ms": 160,
              "strata": [{"source": source, "language": lang, **count,
                          "median_residual_ms_by_flank": {
                              side: float(np.median(values)) for side, values in timing[source, lang].items()}}
                         for (source, lang), count in sorted(counts.items())]}
    _write_json(root / "reports" / f"{JOB}-timing.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
