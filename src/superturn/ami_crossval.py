"""Compare telephone model variants across connected AMI meeting-group folds."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ami_calibration import CONFIG, _write_json
from .ami_pilot import fit_and_score
from .ami_transfer import partial_auc
from .progress import Progress
from .shards import SHARD_ROWS, seed_for
from .sources import data_root

FOLDS = 5
SEEDS = (17, 29)
BASE = {**CONFIG, "architecture": "a", "epochs": 12, "fixed_epochs": True}
VARIANTS = {
    "gru-3s": {},
    "gru-1s": {"window": 1},
    "gru-6s": {"window": 6},
    "conformer-3s": {"architecture": "b"},
    "gru-3s-augmented": {"augmentation": True, "augmentation_workers": 4},
}


def assign_folds(rows, count=FOLDS):
    """Keep the largest component in training and balance other groups by size."""
    counts = Counter(row["group"] for row in rows)
    if len(counts) <= count:
        raise ValueError("need at least one training-only group and one group per fold")
    largest, *held_out = sorted(counts, key=lambda group: (
        -counts[group], seed_for("ami-crossval-group-v1", group), group))
    assignments = {largest: None}
    totals = [0] * count
    for group in held_out:
        fold = min(range(count), key=lambda fold: (totals[fold], fold))
        assignments[group] = fold
        totals[fold] += counts[group]
    return assignments


def summarize(rows):
    return {"examples": len(rows), "speakers": len({row["speaker"] for row in rows}),
            "groups": len({row["group"] for row in rows}),
            "labels": dict(Counter(row["label"] for row in rows))}


def write_shards(root, rows, split):
    destination = root / "shards" / split / "clean"
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for start in range(0, len(rows), SHARD_ROWS):
        batch = rows[start:start + SHARD_ROWS]
        for row in batch:
            digest.update(json.dumps({k: v for k, v in row.items() if k != "pcm"}, sort_keys=True).encode())
            digest.update(row["pcm"])
        path = destination / f"{start // SHARD_ROWS:06d}.parquet"
        temporary = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(batch), temporary, compression="zstd")
        temporary.replace(path)
    return digest.hexdigest()


def prepare(root, destination, rows, assignments):
    prepared = []
    meter = Progress("ami-crossval-prepare", FOLDS, "folds")
    try:
        for test in range(FOLDS):
            calibration = (test + 1) % FOLDS
            shards = defaultdict(list)
            for row in rows:
                fold = assignments[row["group"]]
                split = "out-of-fold" if fold == test else "validation" if fold == calibration else "train"
                shards[split].append(row)
            folder = destination / f"fold-{test}"
            info = {}
            for split in ("train", "validation", "out-of-fold"):
                cases = shards[split]
                if {row["label"] for row in cases} != {"hold", "shift"}:
                    raise ValueError(f"group split {test}/{split} needs both labels")
                info[split] = {**summarize(cases), "sha256": write_shards(folder, cases, split)}
            _write_json(folder / "shards" / "manifest.json", {
                "sha256_by_split": {split: values["sha256"] for split, values in info.items()},
            })
            raw = folder / "raw"
            raw.mkdir(exist_ok=True)
            for name in ("musan", "openslr26"):
                if not (raw / name).is_symlink():
                    (raw / name).symlink_to(root / "raw" / name, target_is_directory=True)
            prepared.append(info)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return prepared


def summarize_runs(runs):
    """Average discrimination within fits and pool their independently calibrated decisions."""
    holds = sum(row["evaluation"]["holds"] for row in runs)
    shifts = sum(row["evaluation"]["shifts"] for row in runs)
    return {
        **{key: mean(row["evaluation"][key] for row in runs) for key in ("auc", "partial_auc")},
        "false_complete": sum(row["evaluation"]["false_complete"] * row["evaluation"]["holds"]
                              for row in runs) / holds,
        "recall": sum(row["evaluation"]["recall"] * row["evaluation"]["shifts"] for row in runs) / shifts,
    }


def paired_deltas(predictions, meta, *, variants=VARIANTS, baseline="gru-3s"):
    """Bootstrap connected evaluation groups within each fold and pair by seed."""
    from .train import _auc

    rng = np.random.default_rng(23)
    output = {}
    for variant in sorted(set(variants) - {baseline}):
        output[variant] = {}
    for name, metric in (("auc", _auc), ("partial_auc", partial_auc)):
        resamples = defaultdict(list)
        for _ in range(400):
            folds = defaultdict(list)
            for fold, (truth, groups) in meta.items():
                indices = defaultdict(list)
                for index, group in enumerate(groups):
                    indices[group].append(index)
                chosen = [index for group in rng.choice(sorted(indices), len(indices)) for index in indices[group]]
                labels = [truth[i] for i in chosen]
                if len(set(labels)) < 2:
                    continue
                for variant in output:
                    effects = []
                    for seed in SEEDS:
                        base = predictions[fold, baseline, seed]
                        values = predictions[fold, variant, seed]
                        effects.append(metric([values[i] for i in chosen], labels)
                                       - metric([base[i] for i in chosen], labels))
                    folds[variant].append(mean(effects))
            for variant, effects in folds.items():
                if len(effects) == len(meta):
                    resamples[variant].append(mean(effects))
        for variant, values in resamples.items():
            output[variant][name] = {
                "quantiles": np.quantile(values, [0.025, 0.975]).tolist(),
                "valid_resamples": len(values),
            }
    return output


def run(root, pilot, device):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI cross-validation uses the training server")
    if len(pilot) != 16 or any(char not in "0123456789abcdef" for char in pilot):
        raise ValueError("calibration pilot must be a 16-character report ID")
    cohort = root / "shards" / "ami-calibration" / pilot
    prior = json.loads((cohort / "prepared.json").read_text())["cohorts"]
    rows = []
    for split in ("train", "validation", "calibration"):
        cases = [row for batch in records(paths(cohort, split)) for row in batch]
        if summarize(cases) != {
            **{key: prior[split][key] for key in ("examples", "speakers", "labels")},
            "groups": prior[split]["meeting_groups"],
        }:
            raise ValueError(f"{split} differs from the frozen calibration pilot")
        rows.extend(cases)
    speaker_groups = defaultdict(set)
    for row in rows:
        speaker_groups[row["speaker"]].add(row["group"])
    if any(len(groups) != 1 for groups in speaker_groups.values()):
        raise ValueError("AMI pool has speakers shared across meeting groups")
    assignments = assign_folds(rows)
    reserved = [row for row in rows if assignments[row["group"]] is None]
    protocol = {
        "pilot": pilot, "cohort_hashes": {split: prior[split]["sha256"]
                                          for split in ("train", "validation", "calibration")},
        "base": BASE, "variants": VARIANTS, "seeds": SEEDS, "folds": FOLDS,
        "always_training": summarize(reserved), "pool": summarize(rows),
        "code": {name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
                 for name in ("ami_crossval", "ami_pilot", "ami_transfer", "train", "model",
                              "augment", "audio", "training_augmentation", "shards")},
    }
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    destination = root / "shards" / "ami-crossval" / key
    exports = root / "exports" / "ami-crossval" / key
    report = root / "reports" / "ami-crossval" / f"{key}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    prepared = destination / "prepared.json"
    if prepared.is_file():
        info = json.loads(prepared.read_text())
    else:
        info = prepare(root, destination, rows, assignments)
        _write_json(prepared, info)
    result = {**protocol, "splits": info, "runs": []}
    predictions = {}
    meta = {}
    meter = Progress("ami-crossval", FOLDS * len(VARIANTS) * len(SEEDS), "runs")
    try:
        for fold in range(FOLDS):
            folder = destination / f"fold-{fold}"
            groups = [row["group"] for batch in records(paths(folder, "out-of-fold"), columns=["group"])
                      for row in batch]
            for variant, changes in VARIANTS.items():
                for seed in SEEDS:
                    config = {**BASE, **changes, "seed": seed}
                    name = f"cv-{fold}-{variant}-{seed}"
                    checkpoint, scores = fit_and_score(folder, name, config, folder,
                                                       ("validation", "out-of-fold"), exports / name, device)
                    for split, (_, truth) in scores.items():
                        if Counter(truth) != Counter({0: info[fold][split]["labels"]["hold"],
                                                       1: info[fold][split]["labels"]["shift"]}):
                            raise ValueError(f"fold {fold}/{split} labels changed")
                    probability, truth = scores["out-of-fold"]
                    if meta.setdefault(fold, (truth, groups)) != (truth, groups):
                        raise ValueError(f"fold {fold} scores do not align with earlier fits")
                    predictions[fold, variant, seed] = probability
                    threshold = _threshold(*scores["validation"])
                    evaluation = {**_stats(probability, truth, threshold),
                                  "partial_auc": partial_auc(probability, truth)}
                    result["runs"].append({
                        "fold": fold, "variant": variant, "seed": seed, **checkpoint,
                        "threshold": threshold, "evaluation": evaluation,
                    })
                    _write_json(report, result)
                    meter.advance(1)
        result["summary"] = {variant: summarize_runs([row for row in result["runs"] if row["variant"] == variant])
                             for variant in VARIANTS}
        result["paired_fold_bootstrap"] = paired_deltas(predictions, meta)
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
    with (root / "jobs" / "ami-crossval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, args.pilot, "cuda"))
