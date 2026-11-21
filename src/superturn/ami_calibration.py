"""Pilot independent epoch selection and threshold calibration on AMI telephone audio."""

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from .ami_pilot import _annotation_labels, _render, _summary, fit_and_score, supported_crosscheck
from .labels import ami_speakers, read_rows
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .splits import speaker_groups

DEVELOPMENT_SPLITS = ("train", "validation", "calibration")
ASSESSMENT_SPLITS = ("assessment", "crosscheck")
CONFIG = {"parameters": 150000, "window": 3, "epochs": 15, "batch_size": 16, "augmentation": False}
EXPERIMENTS = [(architecture, seed) for architecture in ("a", "b") for seed in (17, 29)]
TRAIN_SPEAKER_CAP = 12


def development_splits(train_meetings, groups):
    """Reserve two independent group samples, keeping the largest group for fit."""
    counts = Counter(groups[meeting] for meeting in train_meetings)
    largest = max(counts, key=lambda group: (counts[group], group))
    candidates = sorted(counts.keys() - {largest},
                        key=lambda group: (seed_for("ami-calibration-split-v1", group), group))
    size = round(len(candidates) * 0.25)
    if not size:
        raise ValueError("calibration pilot needs more connected meeting groups")
    assignments = {group: split for split, subset in (
        ("validation", candidates[:size]), ("calibration", candidates[size:2 * size]),
    ) for group in subset}
    return {meeting: assignments.get(groups[meeting], "train") for meeting in train_meetings}


def _write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(root, folder, paths):
    original = [row for row in read_rows(paths["labels"]) if row.split in {"train", "validation"}]
    with ZipFile(paths["annotations"]) as archive:
        speakers = {meeting: {speaker for speaker, _ in tracks.values()}
                    for meeting, tracks in ami_speakers(archive).items()}
    groups = speaker_groups(speakers.items())
    assignments = development_splits({r.recording for r in original if r.split == "train"}, groups)
    for row in original:
        if row.split == "validation":
            assignments[row.recording] = "assessment"
    speakers_by_split = defaultdict(set)
    meetings_by_split = defaultdict(list)
    for meeting, split in sorted(assignments.items()):
        speakers_by_split[split].update(speakers[meeting])
        meetings_by_split[split].append(meeting)
    if sum(map(len, speakers_by_split.values())) != len(set.union(*speakers_by_split.values())):
        raise ValueError("AMI headset voices overlap across development cohorts")

    annotation, _, screen, _ = _annotation_labels(root, original, [])
    cohorts = defaultdict(list)
    for row in annotation:
        split = assignments[row.recording]
        cohorts[split].append(replace(row, split=split, group=groups[row.recording]))
    training = sorted(cohorts["train"], key=lambda row: seed_for("ami-calibration-cap-v1", row.audio, row.pause_start))
    counts = Counter()
    cohorts["train"] = []
    for row in training:
        if counts[row.speaker] < TRAIN_SPEAKER_CAP:
            cohorts["train"].append(row)
            counts[row.speaker] += 1
    cohorts["crosscheck"] = [replace(row, split="crosscheck", group=groups[row.recording])
                            for row in supported_crosscheck(root, [r for r in original if r.split == "validation"])]
    meetings_by_split["crosscheck"] = sorted({row.recording for row in cohorts["crosscheck"]})
    report = {}
    for split in (*DEVELOPMENT_SPLITS, *ASSESSMENT_SPLITS):
        rows = cohorts[split]
        if {row.label for row in rows} != {"hold", "shift"}:
            raise ValueError(f"{split} cohort requires holds and shifts")
        meetings = meetings_by_split[split]
        report[split] = {**_summary(rows), "meeting_groups": len({groups[m] for m in meetings}),
                         "meetings": len(meetings), "sha256": _render(root, folder, rows, split)}
    _write_json(folder / "shards" / "manifest.json",
                {"sha256_by_split": {split: stats["sha256"] for split, stats in report.items()}})
    return {"cohorts": report, "acoustic_screen": screen,
            "train_speaker_cap": TRAIN_SPEAKER_CAP, "max_train_examples_per_speaker": max(counts.values())}


def run(root, device):
    from .train import _stats, _threshold

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI calibration uses the training server")
    annotation, = (root / "raw" / "ami_annotations").glob("*.zip")
    inputs = {"labels": root / "labels" / "ami.jsonl", "annotations": annotation,
              "raw_manifest": root / "raw" / "ami" / "MANIFEST.json"}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    sources = ("ami_calibration", "ami_pilot", "ami_reference", "audio", "augment",
               "labels", "shards", "splits", "train")
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
                   for name in sources}
    signature = json.dumps([hashes, code_hashes, CONFIG, EXPERIMENTS], sort_keys=True).encode()
    identifier = hashlib.sha256(signature).hexdigest()[:16]
    folder = root / "shards" / "ami-calibration" / identifier
    exports = root / "exports" / "ami-calibration" / identifier
    report = root / "reports" / "ami-calibration" / f"{identifier}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    prepared = folder / "prepared.json"
    if prepared.is_file():
        info = json.loads(prepared.read_text())
    else:
        info = prepare(root, folder, inputs)
        _write_json(prepared, info)
    result = {**info, "inputs": hashes, "code": code_hashes, "config": CONFIG, "decision_ms": 300,
              "condition": "telephone", "calibration_target_false_complete": 0.05, "runs": []}
    meter = Progress("ami-calibration", len(EXPERIMENTS), "runs")
    try:
        for architecture, seed in EXPERIMENTS:
            name = f"ami-calibration-{architecture}-{seed}"
            config = {**CONFIG, "architecture": architecture, "seed": seed}
            checkpoint, scores = fit_and_score(folder, name, config, folder, ("calibration", *ASSESSMENT_SPLITS),
                                               exports / name, device)
            for split, (_, labels) in scores.items():
                if Counter(labels) != Counter({0: info["cohorts"][split]["labels"]["hold"],
                                               1: info["cohorts"][split]["labels"]["shift"]}):
                    raise ValueError(f"{split} rows do not match the prepared cohort")
            threshold = _threshold(*scores["calibration"])
            _write_json(exports / name / "calibration.json", {
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "calibration_sha256": info["cohorts"]["calibration"]["sha256"],
                "decision_ms": result["decision_ms"],
                "threshold": threshold,
                "target_false_complete": result["calibration_target_false_complete"],
            })
            result["runs"].append({"architecture": architecture, "seed": seed, **checkpoint,
                                   "calibrated_threshold": threshold,
                                   "at_epoch_threshold": {split: _stats(*scores[split], checkpoint["validation"]["threshold"])
                                                          for split in ASSESSMENT_SPLITS},
                                   **{split: _stats(*pairs, threshold) for split, pairs in scores.items()}})
            _write_json(report, result)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    import fcntl

    root = data_root()
    with (root / "jobs" / "ami-calibration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, "cuda"))
