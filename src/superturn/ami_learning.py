"""Measure AMI training-speaker coverage and the effect of audio augmentation."""

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from .ami_calibration import CONFIG, _write_json
from .ami_pilot import _render, _summary, fit_and_score
from .ami_transfer import COHORTS, FINE_SEEDS, paired_intervals, partial_auc
from .labels import Label
from .progress import Progress
from .shards import seed_for
from .sources import data_root

ARMS = {
    "telephone-quarter": ("telephone", 0.25, False),
    "telephone-half": ("telephone", 0.5, False),
    "scratch": ("telephone", 1.0, False),
    "clean": ("clean", 1.0, False),
    "clean-augmented": ("clean", 1.0, True),
}


def select_speakers(rows, fraction):
    """Keep all clips for each selected caller, nested and stratified by meeting group."""
    voices = defaultdict(set)
    for row in rows:
        voices[row.group].add(row.speaker)
    chosen = {
        (group, speaker) for group, speakers in voices.items()
        for speaker in sorted(speakers, key=lambda speaker: (seed_for("ami-curve-v1", group, speaker), speaker))[
            :max(1, round(len(speakers) * fraction))]
    }
    return [row for row in rows if (row.group, row.speaker) in chosen]


def run(root, pilot, device):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI learning curve uses the training server")
    if len(pilot) != 16 or any(char not in "0123456789abcdef" for char in pilot):
        raise ValueError("calibration pilot must be a 16-character report ID")
    cohort = root / "shards" / "ami-calibration" / pilot
    prior = json.loads((cohort / "prepared.json").read_text())["cohorts"]
    source_files = ("ami_learning", "ami_pilot", "ami_transfer", "model", "train", "training_augmentation",
                    "augment", "audio", "shards")
    protocol = {
        "pilot": pilot, "cohorts": prior, "arms": ARMS, "seeds": FINE_SEEDS, "config": CONFIG,
        "code": {name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
                 for name in source_files},
    }
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    workspace = root / "shards" / "ami-learning" / key
    exports = root / "exports" / "ami-learning" / key
    report = root / "reports" / "ami-learning" / f"{key}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    labels = [field.name for field in fields(Label)]
    rows = [Label(**row) for batch in records(paths(cohort, "train"), columns=labels) for row in batch]
    if _summary(rows) != {key: prior["train"][key] for key in ("examples", "speakers", "labels")}:
        raise ValueError("AMI training rows do not match the calibrated cohort")
    data = {}
    prepared = workspace / "prepared.json"
    if prepared.is_file():
        data = json.loads(prepared.read_text())
    else:
        variants = {arm: (channel, fraction) for arm, (channel, fraction, _) in ARMS.items()
                    if arm != "clean-augmented"}
        meter = Progress("ami-learning-prepare", len(variants), "cohorts")
        try:
            for arm, (channel, fraction) in variants.items():
                folder = workspace / arm
                selected = select_speakers(rows, fraction)
                if {row.label for row in selected} != {"hold", "shift"}:
                    raise ValueError(f"training arm {arm} needs both labels")
                digest = _render(root, folder, selected, "train", channel)
                if arm == "scratch" and digest != prior["train"]["sha256"]:
                    raise ValueError("full telephone training differs from the calibrated cohort")
                valid = folder / "shards" / "validation"
                if not valid.is_symlink():
                    valid.symlink_to(cohort / "shards" / "validation", target_is_directory=True)
                _write_json(folder / "shards" / "manifest.json", {
                    "sha256_by_split": {"train": digest, "validation": prior["validation"]["sha256"]},
                })
                data[arm] = {**_summary(selected), "sha256": digest}
                meter.advance(1)
            data["clean-augmented"] = data["clean"]
            raw = workspace / "clean" / "raw"
            raw.mkdir(exist_ok=True)
            for source in ("musan", "openslr26"):
                if not (raw / source).is_symlink():
                    (raw / source).symlink_to(root / "raw" / source, target_is_directory=True)
            _write_json(prepared, data)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()

    groups = {split: [row["group"] for batch in records(paths(cohort, split), columns=["group"])
                      for row in batch] for split in COHORTS}
    scores_by_split = {split: defaultdict(list) for split in COHORTS}
    targets = {}
    result = {**protocol, "training": data, "runs": []}
    meter = Progress("ami-learning", len(ARMS) * len(FINE_SEEDS), "runs")
    try:
        for seed in FINE_SEEDS:
            for arm, (_, _, augmentation) in ARMS.items():
                folder = workspace / ("clean" if arm == "clean-augmented" else arm)
                name = f"ami-curve-{arm}-{seed}"
                config = {**CONFIG, "architecture": "a", "lr": 0.001, "seed": seed,
                          "augmentation": augmentation}
                if augmentation:
                    config["augmentation_workers"] = 4
                checkpoint, scores = fit_and_score(folder, name, config, cohort, COHORTS, exports / name, device)
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
        result["paired_meeting_group_bootstrap"] = {}
        for split in ("assessment", "crosscheck"):
            scores = scores_by_split[split]
            curve = {arm: scores[arm] for arm in ("scratch", "telephone-quarter", "telephone-half")}
            augmented = {"scratch": scores["clean"], "clean-augmented": scores["clean-augmented"]}
            result["paired_meeting_group_bootstrap"][split] = {
                "curve_vs_full": paired_intervals(curve, groups[split], targets[split]),
                "augmentation_vs_clean": paired_intervals(augmented, groups[split], targets[split]),
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
    with (root / "jobs" / "ami-learning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, args.pilot, "cuda"))
