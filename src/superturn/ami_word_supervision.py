"""Test whether past caller-word targets improve the audio-only AMI GRU."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .ami_crossval import BASE, SEEDS, paired_deltas, summarize_runs, write_shards
from .ami_pilot import fit_and_score
from .ami_text_signal import CROSSVAL, read_words, row_features
from .ami_transfer import partial_auc
from .progress import Progress
from .shards import seed_for
from .sources import data_root

ARMS = ("audio", "word-targets", "shuffled-targets")
AUXILIARY_WEIGHT = 0.5


def targets_for(rows, words):
    targets = []
    for row in rows:
        timing, categories, _, _, _ = row_features(row, words)
        six_seconds, two_seconds, elapsed = timing
        targets.append([min(six_seconds / 24, 1), min(two_seconds / 8, 1),
                        1 - elapsed / 6, *categories])
    return targets


def shuffle_targets(rows, targets, fold):
    """Keep the training groups' and classes' auxiliary target distributions."""
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[row["group"], row["label"]].append(index)
    shuffled = targets.copy()
    rng = np.random.default_rng(seed_for("ami-word-supervision-shuffle", fold))
    for indices in strata.values():
        for index, donor in zip(indices, rng.permutation(indices)):
            shuffled[index] = targets[donor]
    return shuffled


def prepare(folder, source, rows, targets, validation_hash):
    train_hash = write_shards(folder, [dict(row, auxiliary_targets=target)
                                       for row, target in zip(rows, targets, strict=True)], "train")
    validation = folder / "shards" / "validation"
    if not validation.is_symlink():
        validation.symlink_to(source / "shards" / "validation", target_is_directory=True)
    _write_json(folder / "shards" / "manifest.json", {
        "sha256_by_split": {"train": train_hash, "validation": validation_hash},
    })
    return train_hash


def run(root, device):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI word supervision uses the training server")
    folds = root / "shards" / "ami-crossval" / CROSSVAL
    reference = root / "reports" / "ami-crossval" / f"{CROSSVAL}.json"
    original = json.loads(reference.read_text())
    base = {**BASE, "window": 6}
    files = {
        "reference": reference,
        "raw_manifest": root / "raw" / "ami" / "MANIFEST.json",
        "annotations": next((root / "raw" / "ami_annotations").glob("*.zip")),
    }
    modules = ("ami_word_supervision", "ami_text_signal", "ami_crossval", "ami_pilot", "ami_transfer",
               "train", "model", "shards", "audio")
    files.update({module: Path(__file__).with_name(f"{module}.py") for module in modules})
    protocol = {"crossval": CROSSVAL, "config": base, "arms": ARMS, "seeds": SEEDS,
                "auxiliary_weight": AUXILIARY_WEIGHT,
                "hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    staging = root / "shards" / "ami-word-supervision" / key
    exports = root / "exports" / "ami-word-supervision" / key
    report = root / "reports" / "ami-word-supervision" / f"{key}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    predictions, meta = {}, {}
    result = {**protocol, "folds": [], "runs": []}
    meter = Progress("ami-word-supervision", original["folds"] * len(ARMS) * len(SEEDS), "runs")
    try:
        for fold, info in enumerate(original["splits"]):
            source = folds / f"fold-{fold}"
            rows = [row for batch in records(paths(source, "train")) for row in batch]
            groups = [row["group"] for batch in records(paths(source, "out-of-fold"), columns=["group"])
                      for row in batch]
            words = read_words(root, {row["recording"] for row in rows})
            targets = targets_for(rows, words)
            shuffled = shuffle_targets(rows, targets, fold)
            arm_folders = {arm: staging / f"fold-{fold}" / arm for arm in ARMS[1:]}
            train_hashes = {
                arm: prepare(arm_folders[arm], source, rows, values, info["validation"]["sha256"])
                for arm, values in zip(ARMS[1:], (targets, shuffled), strict=True)
            }
            control = staging / f"fold-{fold}" / "audio"
            control.mkdir(parents=True, exist_ok=True)
            if not (control / "shards").is_symlink():
                (control / "shards").symlink_to(source / "shards", target_is_directory=True)
            arm_folders["audio"] = control
            result["folds"].append({"fold": fold, "train_rows": len(rows), "train_hashes": train_hashes,
                                    "shuffled_rows_changed": sum(a != b for a, b in zip(targets, shuffled))})
            for seed in SEEDS:
                for arm in ARMS:
                    config = {**base, "seed": seed}
                    if arm != "audio":
                        config.update(auxiliary_size=len(targets[0]), auxiliary_weight=AUXILIARY_WEIGHT)
                    name = f"word-supervision-{fold}-{arm}-{seed}"
                    checkpoint, scores = fit_and_score(arm_folders[arm], name, config, source,
                                                       ("validation", "out-of-fold"), exports / name, device)
                    for split, (_, labels) in scores.items():
                        expected = info[split]["labels"]
                        if Counter(labels) != Counter({0: expected["hold"], 1: expected["shift"]}):
                            raise ValueError(f"fold {fold}/{split} labels changed")
                    probability, truth = scores["out-of-fold"]
                    if meta.setdefault(fold, (truth, groups)) != (truth, groups):
                        raise ValueError(f"fold {fold} scores do not align")
                    predictions[fold, arm, seed] = probability
                    threshold = _threshold(*scores["validation"])
                    evaluation = {
                        **_stats(probability, truth, threshold),
                        "partial_auc": partial_auc(probability, truth),
                        "partial_auc_to_5pct": partial_auc(probability, truth, max_fpr=0.05),
                    }
                    result["runs"].append({"fold": fold, "seed": seed, "arm": arm,
                                           **checkpoint, "evaluation": evaluation, "threshold": threshold})
                    _write_json(report, result)
                    meter.advance(1)
        result["summary"] = {}
        for arm in ARMS:
            runs = [row for row in result["runs"] if row["arm"] == arm]
            result["summary"][arm] = {
                **summarize_runs(runs),
                "partial_auc_to_5pct": float(np.mean([row["evaluation"]["partial_auc_to_5pct"] for row in runs])),
            }
        result["paired_bootstrap_vs_audio"] = paired_deltas(
            predictions, meta, variants=ARMS, baseline="audio", max_fpr=0.05)
        result["paired_bootstrap_vs_shuffled"] = paired_deltas(
            predictions, meta, variants=ARMS[1:], baseline="shuffled-targets", max_fpr=0.05)
        result["paired_bootstrap_max_fpr"] = 0.05
        _write_json(report, result)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(run(data_root(), "cuda"))
