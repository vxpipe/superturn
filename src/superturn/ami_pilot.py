"""Compare two AMI training label methods on matching, held-out telephone data."""

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ami_reference import _manual_spans, audit_sample
from .ami_training import load_selection
from .audio import RATE, read_audio
from .augment import condition
from .labels import Speech, ami_backchannels, ami_speakers, read_rows
from .progress import Progress
from .shards import SHARD_ROWS, clip, seed_for
from .sources import data_root

DECISION_MS = 300
GAP = DECISION_MS * RATE // 1000
JOIN = RATE // 5
FRAME = RATE // 50


def group_words(words):
    """Join adjacent words without bridging a qualifying pause."""
    tracks = defaultdict(list)
    for word in sorted(words, key=lambda word: word.start):
        track = tracks[word.speaker]
        if track and word.start - track[-1].end < JOIN:
            prior = track.pop()
            word = Speech(word.speaker, prior.start, max(prior.end, word.end))
        track.append(word)
    return sorted((span for track in tracks.values() for span in track),
                  key=lambda span: (span.start, span.end, span.speaker))


def annotated_pauses(spans, backchannels, *, minimum_gap=GAP):
    """Anchor global silence to speech spans and use explicit backchannel tags."""
    active_until = 0
    for index, (caller, first) in enumerate(zip(spans, spans[1:])):
        active_until = max(active_until, caller.end)
        if active_until > caller.end or first.start - caller.end < minimum_gap:
            continue
        marked = first.end - first.start < RATE and any(
            abs(point - first.start) <= JOIN for point in backchannels.get(first.speaker, ())
        )
        if first.speaker == caller.speaker:
            label = "hold"
        elif marked:
            following = spans[index + 2:index + 3]
            if not following:
                continue
            later, = following
            if later.speaker == first.speaker or later.start - first.end > 3 * RATE:
                continue
            if later.speaker != caller.speaker and later.end - later.start < RATE:
                continue
            label = "hold" if later.speaker == caller.speaker else "shift"
        elif first.end - first.start >= RATE:
            label = "shift"
        else:
            continue
        yield caller, first, label, marked


def _peak(audio, start, end):
    start = max(0, start)
    end = min(len(audio), end)
    frames = (end - start) // FRAME
    if frames <= 0:
        return 0.0
    values = audio[start:start + frames * FRAME].astype(np.float32).reshape(-1, FRAME)
    return float(np.sqrt(np.max(np.mean(values * values, axis=1))))


def acoustic_status(caller, following, tracks, nonwords=(), *, dominance=1.414, quiet_ratio=0.5):
    """Require evidence for both active headsets and a quieter query window."""
    if caller.end + GAP > min(map(len, tracks.values())):
        return "incomplete slice"
    if any(span.start < caller.end + GAP and span.end > caller.end for span in nonwords):
        return "timed nonword event in pause"
    levels = []
    for span, start, end in (
        (caller, max(caller.start, caller.end - 8 * FRAME), caller.end),
        (following, following.start, min(following.end, following.start + 8 * FRAME)),
    ):
        peaks = {speaker: _peak(pcm, start, end) for speaker, pcm in tracks.items()}
        own = peaks.pop(span.speaker)
        if own < 128 or own < dominance * max(peaks.values(), default=0):
            return "headset dominance or level"
        levels.append(own)
    quiet = max(_peak(pcm, caller.end + 2 * FRAME, caller.end + GAP) for pcm in tracks.values())
    return "passed" if quiet <= quiet_ratio * min(levels) else "speech or noise in pause"


def _annotation_labels(root, original, current, *, screen=acoustic_status):
    split_by_meeting = {row.recording: row for row in original}
    current_by_meeting = defaultdict(list)
    for row in current:
        current_by_meeting[row.recording].append(row)
    tracks = defaultdict(list)
    raw = root / "raw" / "ami"
    for entry in json.loads((raw / "MANIFEST.json").read_text())["files"]:
        tracks[entry["path"].split("/")[0]].append(entry)
    archive_path, = (root / "raw" / "ami_annotations").glob("*.zip")
    results, screened_current = [], []
    counts = defaultdict(Counter)
    current_counts = Counter()
    meter = Progress("ami-pilot-labels", len(split_by_meeting), "meetings")
    try:
        with ZipFile(archive_path) as archive:
            mappings = ami_speakers(archive)
            for meeting, row in sorted(split_by_meeting.items()):
                mapping = mappings[meeting]
                paths, audio = {}, {}
                windows = {(round(entry["start_s"] * RATE), round(entry["length_s"] * RATE))
                           for entry in tracks[meeting]}
                if len(windows) != 1:
                    raise ValueError(f"{meeting}: headset slices are misaligned")
                (offset, length), = windows
                for entry in tracks[meeting]:
                    channel = int(Path(entry["path"]).stem.rsplit("Headset-", 1)[1])
                    if channel in mapping:
                        speaker = mapping[channel][0]
                        path = raw / entry["path"]
                        paths[speaker] = str(path.relative_to(root))
                        audio[speaker] = read_audio(path)
                if set(audio) != {speaker for speaker, _ in mapping.values()}:
                    raise ValueError(f"{meeting}: missing annotated headset")
                spans = group_words(_manual_spans(archive, meeting, mapping, offset, length))
                nonwords = list(_manual_spans(archive, meeting, mapping, offset, length,
                                              tags=("vocalsound", "disfmarker", "gap", "transformerror")))
                marked = {speaker: ami_backchannels(archive, meeting, agent, offset)
                          for speaker, agent in mapping.values()}
                for caller, following, label, backchannel in annotated_pauses(spans, marked):
                    status = screen(caller, following, audio, nonwords)
                    counts[row.split][status] += 1
                    if status == "passed":
                        results.append(replace(
                            row, speaker=caller.speaker, pause_start=caller.end, pause_samples=GAP,
                            label=label, method="annotation-acoustic", audio=paths[caller.speaker],
                            annotated_backchannel=backchannel, silence_start=None,
                        ))
                for baseline in current_by_meeting[meeting]:
                    edge = baseline.pause_start
                    following = next((span for span in spans if span.start >= edge + GAP), None)
                    if following is None:
                        current_counts["unmatched subsequent speech"] += 1
                        continue
                    caller = Speech(baseline.speaker, max(0, edge - 8 * FRAME), edge)
                    status = acoustic_status(caller, following, audio, nonwords)
                    current_counts[status] += 1
                    if status == "passed":
                        screened_current.append(baseline)
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return results, screened_current, counts, current_counts


def match_training(current, annotation):
    """Match the number of examples for each speaker and label."""
    groups = []
    for rows in (current, annotation):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row.speaker, row.label].append(row)
        groups.append(buckets)
    paired = [[], []]
    for key in sorted(groups[0].keys() & groups[1].keys()):
        count = min(len(group[key]) for group in groups)
        for selected, group in zip(paired, groups):
            candidates = sorted(group[key], key=lambda row: seed_for(
                "ami-pilot-match-v1", row.audio, row.pause_start, row.label))
            selected.extend(candidates[:count])
    return paired


def _render(root, destination, rows, split, channel="telephone"):
    """Write rendered examples in the training reader's normal shard layout."""
    folder = destination / "shards" / split / "clean"
    folder.parent.mkdir(parents=True, exist_ok=True)
    revised = sorted(rows, key=lambda row: (row.audio, row.pause_start))
    cached = lru_cache(maxsize=6)(read_audio)
    digest = hashlib.sha256()
    with TemporaryDirectory(dir=folder.parent) as staging:
        folder.mkdir(exist_ok=True)
        for index in range(0, len(revised), SHARD_ROWS):
            records = []
            for row in revised[index:index + SHARD_ROWS]:
                pcm = condition(clip(row, cached(root / row.audio)), channel, 0).tobytes()
                metadata = asdict(row)
                digest.update(json.dumps(metadata, sort_keys=True).encode())
                digest.update(pcm)
                records.append({**metadata, "condition": channel, "pcm": pcm})
            path = Path(staging) / f"{index // SHARD_ROWS:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")
        for path in folder.glob("*.parquet"):
            path.unlink()
        for path in Path(staging).glob("*.parquet"):
            path.replace(folder / path.name)
    return digest.hexdigest()


def _summary(rows):
    return {"examples": len(rows), "speakers": len({row.speaker for row in rows}),
            "labels": dict(Counter(row.label for row in rows))}


def supported_crosscheck(root, rows):
    limits = {}
    votes = audit_sample(root, rows, pause_limits=limits)
    return [replace(row, pause_samples=GAP) for row, status in votes.items()
            if status == "supported" and limits[row] >= GAP]


def prepare(root, destination):
    original = [row for row in read_rows(root / "labels" / "ami.jsonl") if row.split in {"train", "validation"}]
    limits = load_selection(root)
    current = [
        replace(row, pause_samples=GAP) for row in original
        if row.split == "train" and limits.get((row.audio, row.pause_start, row.label), 0) >= GAP
    ]
    annotation, eligible_current, screened, screened_current = _annotation_labels(root, original, current)
    matched = match_training(eligible_current, [row for row in annotation if row.split == "train"])
    validation = [row for row in annotation if row.split == "validation"]
    crosscheck = supported_crosscheck(root, [row for row in original if row.split == "validation"])
    for rows in [*matched, validation, crosscheck]:
        if {row.label for row in rows} != {"hold", "shift"}:
            raise ValueError("AMI label pilot needs holds and shifts in every cohort")
    if {row.speaker for rows in matched for row in rows} & {row.speaker for row in validation + crosscheck}:
        raise ValueError("AMI label pilot contains held-out speakers in training")

    shared = destination / "evaluation"
    validation_hash = _render(root, shared, validation, "validation")
    crosscheck_hash = _render(root, shared, crosscheck, "crosscheck")
    report = {
        "decision_ms": DECISION_MS, "condition": "telephone",
        "matched_training": {}, "acoustic_screen": screened,
        "current_acoustic_screen": screened_current,
        "validation": {**_summary(validation), "sha256": validation_hash},
        "original_supported_crosscheck": {**_summary(crosscheck), "sha256": crosscheck_hash},
    }
    for name, rows in zip(("current", "annotation"), matched):
        arm = destination / name
        training_hash = _render(root, arm, rows, "train")
        report["matched_training"][name] = {**_summary(rows), "sha256": training_hash}
        link = arm / "shards" / "validation" / "clean"
        link.mkdir(parents=True, exist_ok=True)
        for path in link.glob("*.parquet"):
            path.unlink()
        for path in (shared / "shards" / "validation" / "clean").glob("*.parquet"):
            os.link(path, link / path.name)
        manifest = {"sha256_by_split": {"train": training_hash, "validation": validation_hash}}
        (arm / "shards" / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return report


def train_and_score(folder, name, config, validation, archive, device):
    saved, scores = fit_and_score(folder, name, config, validation, ("crosscheck",), archive, device)
    from .train import _stats

    return {
        "checkpoint_sha256": saved["checkpoint_sha256"],
        "validation": saved["validation"],
        "original_supported_crosscheck": _stats(*scores["crosscheck"], saved["validation"]["threshold"]),
    }


def fit_and_score(folder, name, config, evaluation, splits, archive, device, *, expected_checkpoint=None):
    import torch

    from .model import build
    from .train import _input, completed_run, paths, records, train

    checkpoint = completed_run(folder, name, config, device)
    if expected_checkpoint is not None and (
            checkpoint is None or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_checkpoint):
        raise ValueError(f"{name} does not match the completed reference checkpoint")
    if checkpoint is None:
        checkpoint = train(folder, name, config, device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    archive.mkdir(parents=True, exist_ok=True)
    for filename in ("best.pt", "finished.json", "metrics.json"):
        temporary = archive / f".{filename}.tmp"
        shutil.copyfile(checkpoint.parent / filename, temporary)
        temporary.replace(archive / filename)
    model = build(config).to(device).eval()
    model.load_state_dict(saved["state"])
    results = {}
    with torch.inference_mode():
        for split in splits:
            scores, truth = [], []
            for rows in records(paths(evaluation, split)):
                audio = np.stack([_input(row, model.samples, DECISION_MS) for row in rows])
                pcm = torch.from_numpy(audio.astype(np.float32) / 32768).to(device)
                elapsed = torch.full((len(rows), 1), float(DECISION_MS), device=device)
                scores.extend(model(pcm, elapsed).cpu().flatten().tolist())
                truth.extend(int(row["label"] == "shift") for row in rows)
            results[split] = scores, truth
    return {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "validation": saved["metrics"],
    }, results


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI label pilot uses the existing campaign machine and its private data")
    sources = [root / "labels" / "ami.jsonl", root / "labels" / "ami-verified-train.json",
               root / "raw" / "ami" / "MANIFEST.json", *sorted((root / "raw" / "ami_annotations").glob("*.zip")),
               Path(__file__), Path(__file__).with_name("ami_reference.py")]
    source_digest = hashlib.sha256()
    for source in sources:
        source_digest.update(hashlib.sha256(source.read_bytes()).digest())
    pilot_id = source_digest.hexdigest()
    destination = root / "shards" / "ami-pilot" / pilot_id[:16]
    prepared_path = destination / "prepared.json"
    if prepared_path.is_file():
        prepared = json.loads(prepared_path.read_text())
    else:
        prepared = {"pilot_sha256": pilot_id, **prepare(root, destination)}
        temporary = prepared_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(prepared, indent=2) + "\n")
        temporary.replace(prepared_path)
    print(json.dumps(prepared, indent=2), flush=True)
    validation = destination / "evaluation"
    result = {**prepared, "runs": []}
    output = root / "reports" / "ami-label-pilot" / f"{pilot_id[:16]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {"architecture": "a", "parameters": 150000, "window": 3,
              "epochs": 20, "batch_size": 16, "augmentation": False}
    meter = Progress("ami-label-pilot", 4, "runs")
    try:
        for seed in (17, 29):
            for arm in ("current", "annotation"):
                name = f"ami-pilot-{arm}-{seed}"
                variant = {**config, "seed": seed}
                folder = destination / arm
                archive = root / "exports" / "ami-label-pilot" / pilot_id[:16] / name
                result["runs"].append({
                    "arm": arm, "seed": seed, "architecture": variant,
                    **train_and_score(folder, name, variant, validation, archive, device),
                })
                temporary = output.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(output)
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return output


if __name__ == "__main__":
    import fcntl

    root = data_root()
    with (root / "jobs" / "ami-label-pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, "cuda"))
