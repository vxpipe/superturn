"""Test whether future-word activity pretraining helps AMI turn detection."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .ami_calibration import CONFIG, _write_json, development_splits
from .ami_pilot import fit_and_score
from .ami_reference import _manual_spans
from .ami_transfer import COHORTS, FINE_SEEDS, PRETRAIN_SEEDS, paired_intervals, partial_auc
from .audio import RATE, read_audio
from .augment import condition
from .labels import ami_speakers, read_rows
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .splits import speaker_groups

FRAME = RATE // 10
HORIZONS = (0, 2, 5, 10, 20)
WINDOW_FRAMES = 30
CAP = 48
PRETEXT_EPOCHS = 4
PRETEXT_BATCH = 96
ARMS = ("scratch", "forecast", "shuffled")


def word_activity(words, speakers, length):
    """Rasterize overlapping annotated words on a 100 ms grid."""
    index = {speaker: row for row, speaker in enumerate(speakers)}
    frames = np.zeros((len(speakers), length // FRAME), dtype=bool)
    for word in words:
        frames[index[word.speaker], word.start // FRAME:(word.end + FRAME - 1) // FRAME] = True
    return frames


def anchors(activity, caller):
    """Select anchors using past frames only, then attach future activity targets."""
    own = activity[caller]
    other = np.delete(activity, caller, axis=0).any(axis=0)
    last_own = np.maximum.accumulate(np.where(own, np.arange(len(own)), -100))
    for frame in range(WINDOW_FRAMES, len(own) - HORIZONS[-1] + 1):
        gap = frame - 1 - int(last_own[frame - 1])
        if gap not in (0, 2, 3, 4, 5) or other[frame - max(gap, 1):frame].any():
            continue
        target = tuple(int(stream[frame + start:frame + end].any())
                       for stream in (own, other) for start, end in zip(HORIZONS, HORIZONS[1:]))
        yield frame, gap * 100, target


def prepare(root, folder, prior):
    raw = root / "raw"
    manifest = json.loads((raw / "ami" / "MANIFEST.json").read_text())
    by_meeting = defaultdict(dict)
    for entry in manifest["files"]:
        path = Path(entry["path"])
        channel = int(path.stem.rsplit("Headset-", 1)[1])
        by_meeting[path.parts[0]][channel] = entry
    archive, = (raw / "ami_annotations").glob("*.zip")
    by_voice = defaultdict(list)
    with ZipFile(archive) as metadata:
        roster = ami_speakers(metadata)
        meeting_groups = speaker_groups((meeting, (speaker for speaker, _ in tracks.values()))
                                        for meeting, tracks in roster.items())
        original_train = {row.recording for row in read_rows(root / "labels" / "ami.jsonl") if row.split == "train"}
        splits = development_splits(original_train, meeting_groups)
        meetings = sorted(meeting for meeting, split in splits.items() if split == "train")
        if len(meetings) != prior["train"]["meetings"] or len({meeting_groups[m] for m in meetings}) != prior["train"]["meeting_groups"]:
            raise ValueError("AMI future-activity meetings differ from the pilot training split")
        meter = Progress("ami-activity-anchors", len(meetings), "meetings")
        try:
            for meeting in meetings:
                tracks = roster[meeting]
                files = by_meeting[meeting]
                (offset, length), = {(round(files[channel]["start_s"] * RATE), round(files[channel]["length_s"] * RATE))
                                     for channel in tracks}
                channels = sorted(tracks)
                speakers = [tracks[channel][0] for channel in channels]
                frames = word_activity(_manual_spans(metadata, meeting, tracks, offset, length), speakers, length)
                for i, (channel, speaker) in enumerate(zip(channels, speakers)):
                    path = files[channel]["path"]
                    for frame, gap, target in anchors(frames, i):
                        priority = seed_for("ami-activity-v1", path, frame)
                        by_voice[speaker, bool(gap)].append((priority, path, frame, gap, target))
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()
    selected = sorted(item[1:] for candidates in by_voice.values() for item in sorted(candidates)[:CAP])
    folder.mkdir(parents=True, exist_ok=True)
    meter = Progress("ami-activity-render", len(selected), "clips")
    digest = hashlib.sha256()
    pcm = np.lib.format.open_memmap(folder / "pcm.npy", mode="w+", dtype="<i2",
                                    shape=(len(selected), FRAME * WINDOW_FRAMES))
    labels, gaps = [], []
    current = None
    try:
        for i, (path, frame, gap, target) in enumerate(selected):
            if path != current:
                audio = read_audio(raw / "ami" / path)
                current = path
            end = frame * FRAME
            clip = audio[end - FRAME * WINDOW_FRAMES:end]
            pcm[i] = condition(clip, "telephone", 0)
            digest.update(pcm[i].tobytes())
            labels.append(target)
            gaps.append(gap)
            meter.advance(1)
        pcm.flush()
        np.save(folder / "targets.npy", np.asarray(labels, dtype=np.uint8))
        np.save(folder / "gaps.npy", np.asarray(gaps, dtype=np.int16))
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    digest.update(np.asarray(labels, dtype=np.uint8).tobytes())
    digest.update(np.asarray(gaps, dtype="<i2").tobytes())
    return {"examples": len(selected), "meetings": len(meetings), "sha256": digest.hexdigest(),
            "speakers": len({speaker for speaker, _ in by_voice}), "elapsed_ms": dict(Counter(gaps)),
            "target_positives": np.sum(labels, axis=0).tolist(), "cap_per_speaker_per_anchor_type": CAP}


def pretrain(folder, archive, seed, arm, device, fingerprint):
    import torch
    from torch import nn
    from torch.nn import functional as F

    from .model import build

    path = archive / "activity.pt"
    report = archive / "metrics.json"
    if path.is_file() and report.is_file():
        info = json.loads(report.read_text())
        if info["data_sha256"] == fingerprint and info["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest():
            return path, info
    archive.mkdir(parents=True, exist_ok=True)
    pcm = np.load(folder / "pcm.npy", mmap_mode="r")
    labels = np.load(folder / "targets.npy")
    gaps = np.load(folder / "gaps.npy")
    if arm == "shuffled":
        rng = np.random.default_rng(seed_for("ami-activity-shuffle", seed))
        for gap in np.unique(gaps):
            indices = np.flatnonzero(gaps == gap)
            labels[indices] = labels[rng.permutation(indices)]
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build({**CONFIG, "architecture": "a"})
    model.decision[-1] = nn.Linear(model.decision[-1].in_features, labels.shape[1])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batches = (len(gaps) + PRETEXT_BATCH - 1) // PRETEXT_BATCH
    meter = Progress(f"ami-activity-{arm}-{seed}", batches * PRETEXT_EPOCHS, "steps")
    losses = []
    try:
        for epoch in range(PRETEXT_EPOCHS):
            order = np.random.default_rng(seed_for(seed, epoch)).permutation(len(gaps))
            loss_total = 0.0
            for start in range(0, len(order), PRETEXT_BATCH):
                batch = order[start:start + PRETEXT_BATCH]
                audio = torch.as_tensor(pcm[batch], dtype=torch.float32, device=device) / 32768
                elapsed = torch.as_tensor(gaps[batch, None], dtype=torch.float32, device=device)
                target = torch.as_tensor(labels[batch], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model.logits(audio, elapsed), target)
                loss.backward()
                optimizer.step()
                loss_total += loss.detach().item() * len(batch)
                meter.advance(1)
            losses.append(loss_total / len(gaps))
            print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch + 1, "loss": losses[-1]}), flush=True)
        state = {name: value.cpu() for name, value in model.state_dict().items() if not name.startswith("decision.")}
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        info = {"arm": arm, "seed": seed, "loss": losses, "data_sha256": fingerprint,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        _write_json(report, info)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return path, info


def run(root, pilot, device):
    import torch

    from .model import build
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI activity pretraining uses the training server")
    if len(pilot) != 16 or any(char not in "0123456789abcdef" for char in pilot):
        raise ValueError("calibration pilot must be a 16-character report ID")
    cohort = root / "shards" / "ami-calibration" / pilot
    prior = json.loads((cohort / "prepared.json").read_text())["cohorts"]
    files = [root / "raw" / "ami" / "MANIFEST.json", root / "labels" / "ami.jsonl",
             *sorted((root / "raw" / "ami_annotations").glob("*.zip"))]
    input_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in zip(("raw_manifest", "labels", "annotations"), files, strict=True)}
    origin = json.loads((root / "reports" / "ami-calibration" / f"{pilot}.json").read_text())
    split_code = ("ami_calibration", "ami_reference", "labels", "splits")
    if origin["inputs"] != input_hashes or any(
            origin["code"][name] != hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
            for name in split_code):
        raise ValueError("AMI inputs or split code differ from the pilot that assigned speakers")
    protocol = {"pilot": pilot, "cohorts": prior, "config": CONFIG, "fine_tune_seeds": FINE_SEEDS,
                "pretrain_seeds": PRETRAIN_SEEDS, "pretrain_epochs": PRETEXT_EPOCHS,
                "pretrain_batch_size": PRETEXT_BATCH, "horizon_edges_ms": [edge * 100 for edge in HORIZONS],
                "inputs": input_hashes,
                "code": {name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
                         for name in ("ami_activity", "ami_calibration", "ami_reference", "ami_pilot",
                                      "ami_transfer", "labels", "splits", "shards", "augment", "audio", "train", "model")}}
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    folder = root / "shards" / "ami-activity" / key
    exports = root / "exports" / "ami-activity" / key
    report = root / "reports" / "ami-activity" / f"{key}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    cache = folder / "prepared.json"
    if cache.is_file():
        sampled = json.loads(cache.read_text())
    else:
        sampled = prepare(root, folder, prior)
        _write_json(cache, sampled)
    fine = folder / "fine"
    fine.mkdir(exist_ok=True)
    if not (fine / "shards").is_symlink():
        (fine / "shards").symlink_to(cohort / "shards", target_is_directory=True)
    sources = {}
    groups = {split: [row["group"] for batch in records(paths(cohort, split), columns=["group"])
                      for row in batch] for split in COHORTS}
    scores_by_split = {split: defaultdict(list) for split in COHORTS}
    targets = {}
    result = {**protocol, "pretrain_data": sampled, "pretrain": [], "runs": []}
    meter = Progress("ami-activity", 2 * len(PRETRAIN_SEEDS) + len(ARMS) * len(FINE_SEEDS), "runs")
    try:
        for arm in ("forecast", "shuffled"):
            for seed in PRETRAIN_SEEDS:
                sources[arm, seed], info = pretrain(folder, exports / f"{arm}-{seed}", seed, arm,
                                                     device, sampled["sha256"])
                result["pretrain"].append(info)
                _write_json(report, result)
                meter.advance(1)
        for i, seed in enumerate(FINE_SEEDS):
            for arm in ARMS:
                name = f"ami-activity-{arm}-{seed}"
                config = {**CONFIG, "architecture": "a", "lr": 0.001, "seed": seed}
                if arm != "scratch":
                    source_seed = PRETRAIN_SEEDS[i % len(PRETRAIN_SEEDS)]
                    source = sources[arm, source_seed]
                    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                    starting = fine / "runs" / f"{name}-initial" / "best.pt"
                    marker = starting.with_name("source.json")
                    starting.parent.mkdir(parents=True, exist_ok=True)
                    if (not starting.is_file() or not marker.is_file()
                            or json.loads(marker.read_text()) != {
                                "source_sha256": source_hash,
                                "checkpoint_sha256": hashlib.sha256(starting.read_bytes()).hexdigest(),
                            }):
                        torch.manual_seed(seed)
                        state = build(config).state_dict()
                        state.update(torch.load(source, weights_only=True))
                        temporary = starting.with_suffix(".tmp")
                        torch.save({"state": state}, temporary)
                        temporary.replace(starting)
                        _write_json(marker, {"source_sha256": source_hash,
                                             "checkpoint_sha256": hashlib.sha256(starting.read_bytes()).hexdigest()})
                    config["warm_start"] = starting.parent.name
                checkpoint, scores = fit_and_score(fine, name, config, cohort, COHORTS, exports / name, device)
                for split, (probability, truth) in scores.items():
                    expected = Counter({0: prior[split]["labels"]["hold"], 1: prior[split]["labels"]["shift"]})
                    if Counter(truth) != expected or targets.setdefault(split, truth) != truth:
                        raise ValueError(f"unexpected labels in {split}")
                    scores_by_split[split][arm].append(probability)
                threshold = _threshold(*scores["calibration"])
                result["runs"].append({
                    "arm": arm, "config": config, **checkpoint, "threshold": threshold,
                    **{split: {**_stats(*pairs, threshold), "partial_auc": partial_auc(*pairs)}
                       for split, pairs in scores.items()},
                })
                _write_json(report, result)
                meter.advance(1)
        result["paired_meeting_group_bootstrap"] = {
            split: {
                "versus_scratch": paired_intervals(scores_by_split[split], groups[split], targets[split]),
                "versus_shuffled": paired_intervals({"scratch": scores_by_split[split]["shuffled"],
                                                    "forecast": scores_by_split[split]["forecast"]},
                                                   groups[split], targets[split]),
            } for split in ("assessment", "crosscheck")
        }
        _write_json(report, result)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", help="16-character calibration pilot ID")
    args = parser.parse_args()
    root = data_root()
    with (root / "jobs" / "ami-activity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, args.pilot, "cuda"))
