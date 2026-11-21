"""Compare non-AMI initialization against scratch on fixed AMI telephone cohorts."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ami_calibration import CONFIG, _write_json
from .ami_pilot import GAP, fit_and_score
from .augment import condition
from .progress import Progress
from .shards import CONTEXT, SHARD_ROWS, seed_for
from .sources import data_root

PRETRAIN = {**CONFIG, "architecture": "a", "epochs": 4, "batch_size": 64}
PRETRAIN_SEEDS = (17, 29)
FINE_SEEDS = (17, 29, 43, 59, 71)
TRAIN_QUOTAS = {
    ("common_voice", "hold"): 800, ("common_voice", "shift"): 800,
    ("emilia_yodas", "hold"): 256, ("emilia_yodas", "weak-shift"): 256,
}
VALIDATION_QUOTAS = {("common_voice", "hold"): 160, ("common_voice", "shift"): 160}
COHORTS = ("calibration", "assessment", "crosscheck")
ARMS = {"scratch": 0.001, "transfer": 0.001, "transfer-low-lr": 0.0001}


def select(files, quotas, split):
    """Choose 300 ms examples by a stable hash, with a cap per voice and label."""
    columns = ["source", "label", "speaker", "audio", "pause_start", "pause_samples"]
    candidates = defaultdict(list)
    meter = Progress(f"ami-transfer-select-{split}", len(files), "shards")
    try:
        for path in files:
            for index, row in enumerate(pq.read_table(path, columns=columns).to_pylist()):
                cell = (row["source"], row["label"])
                if cell in quotas and row["pause_samples"] >= GAP and row["speaker"] not in ("", None, "unknown"):
                    priority = seed_for("ami-transfer-v1", row["audio"], row["pause_start"])
                    candidates[cell].append((priority, path, index, row["speaker"]))
            meter.advance(1)
        keep = defaultdict(set)
        for cell, count in quotas.items():
            per_voice = Counter()
            selected = 0
            for _, path, index, speaker in sorted(candidates[cell]):
                if per_voice[speaker] == 2:
                    continue
                keep[path].add(index)
                per_voice[speaker] += 1
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise ValueError(f"non-AMI pretraining {split} needs {count} examples of {cell}, found {selected}")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return keep


def render(folder, selection, split):
    destination = folder / "shards" / split / "clean"
    destination.mkdir(parents=True, exist_ok=True)
    meter = Progress(f"ami-transfer-render-{split}", sum(map(len, selection.values())), "clips")
    digest = hashlib.sha256()
    records = []
    index = 0

    def write_batch():
        output = destination / f"{index:06d}.parquet"
        temporary = output.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
        temporary.replace(output)
        records.clear()

    try:
        for path, indices in sorted(selection.items()):
            for row in pq.read_table(path).take(sorted(indices)).to_pylist():
                pcm = np.frombuffer(row.pop("pcm"), "<i2").copy()
                pcm[CONTEXT + GAP:] = 0
                row.update(pause_samples=GAP, condition="telephone")
                digest.update(json.dumps(row, sort_keys=True).encode())
                row["pcm"] = condition(pcm, "telephone", 0).tobytes()
                digest.update(row["pcm"])
                records.append(row)
                if len(records) == SHARD_ROWS:
                    write_batch()
                    index += 1
                meter.advance(1)
        if records:
            write_batch()
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return digest.hexdigest()


def partial_auc(scores, labels, max_fpr=0.1):
    """Mean ROC recall from zero through max_fpr, with tied scores grouped."""
    from itertools import groupby

    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    true_rate = false_rate = area = 0.0
    for _, tied in groupby(sorted(zip(scores, labels), reverse=True), key=lambda item: item[0]):
        truth = [label for _, label in tied]
        next_true = true_rate + sum(truth) / positive
        next_false = false_rate + (len(truth) - sum(truth)) / negative
        width = next_false - false_rate
        if next_false >= max_fpr:
            width = max_fpr - false_rate
            edge = true_rate + (next_true - true_rate) * width / (next_false - false_rate)
            return (area + width * (true_rate + edge) / 2) / max_fpr
        area += width * (true_rate + next_true) / 2
        true_rate, false_rate = next_true, next_false
    raise ValueError("ROC does not reach the requested false-complete range")


def paired_intervals(scores, groups, labels):
    from .train import _auc

    indices = defaultdict(list)
    for index, group in enumerate(groups):
        indices[group].append(index)
    rng = np.random.default_rng(17)
    samples = []
    group_ids = sorted(indices)
    for _ in range(600):
        chosen = [index for group in rng.choice(group_ids, len(group_ids)) for index in indices[group]]
        truth = [labels[index] for index in chosen]
        if min(truth) == max(truth):
            continue
        samples.append(chosen)
    intervals = {
        arm: {
            name: np.quantile([
                np.mean([metric([a[i] for i in sample], [labels[i] for i in sample])
                         - metric([b[i] for i in sample], [labels[i] for i in sample])
                         for a, b in zip(values, scores["scratch"])])
                for sample in samples], [0.025, 0.975]).tolist()
            for name, metric in (("auc", _auc), ("partial_auc", partial_auc))
        } for arm, values in scores.items() if arm != "scratch"
    }
    return {"groups": len(group_ids), "valid_resamples": len(samples), "delta_vs_scratch": intervals}


def run(root, pilot, device):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI transfer uses the training server")
    if len(pilot) != 16 or any(char not in "0123456789abcdef" for char in pilot):
        raise ValueError("calibration pilot must be a 16-character report ID")
    cohort = root / "shards" / "ami-calibration" / pilot
    prior = json.loads((cohort / "prepared.json").read_text())["cohorts"]
    manifest = json.loads((root / "shards" / "manifest.json").read_text())["sha256_by_split"]
    quotas = {"train": TRAIN_QUOTAS, "validation": VALIDATION_QUOTAS}
    protocol = {
        "cohorts": prior, "pretrain_config": PRETRAIN, "pretrain_seeds": PRETRAIN_SEEDS,
        "fine_tune_config": CONFIG, "fine_tune_seeds": FINE_SEEDS, "fine_tune_lrs": ARMS,
        "source_shards": {split: manifest[split] for split in quotas},
        "pretrain_counts": {split: {f"{source}/{label}": n for (source, label), n in counts.items()}
                            for split, counts in quotas.items()},
        "code": {name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
                 for name in ("ami_transfer", "ami_calibration", "ami_pilot", "train", "augment", "model")},
    }
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    staging = root / "shards" / "ami-transfer" / key
    exports = root / "exports" / "ami-transfer" / key
    report = root / "reports" / "ami-transfer" / f"{key}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    pretrain = staging / "pretrain"
    finish = staging / "ami"
    manifest_path = pretrain / "shards" / "manifest.json"
    if not manifest_path.is_file():
        selections = {split: select(paths(root, split), counts, split) for split, counts in quotas.items()}
        hashes = {split: render(pretrain, selection, split) for split, selection in selections.items()}
        _write_json(manifest_path, {"sha256_by_split": hashes})
    staging.mkdir(parents=True, exist_ok=True)
    finish.mkdir(exist_ok=True)
    if not (finish / "shards").is_symlink():
        (finish / "shards").symlink_to(cohort / "shards", target_is_directory=True)
    protocol["rendered_source_shards"] = json.loads(manifest_path.read_text())["sha256_by_split"]
    groups = {split: [row["group"] for rows in records(paths(cohort, split), columns=["group"])
                      for row in rows] for split in COHORTS}
    fits = []
    downstream = []
    all_scores = {split: defaultdict(list) for split in COHORTS}
    fixed_labels = {}
    result = {**protocol, "pretrain": fits, "downstream": downstream}
    meter = Progress("ami-transfer", len(PRETRAIN_SEEDS) + len(FINE_SEEDS) * len(ARMS), "runs")
    try:
        for seed in PRETRAIN_SEEDS:
            name = f"ami-transfer-source-{seed}"
            archive = exports / name
            checkpoint, _ = fit_and_score(pretrain, name, {**PRETRAIN, "seed": seed}, pretrain, (), archive, device)
            link = finish / "runs" / name / "best.pt"
            link.parent.mkdir(parents=True, exist_ok=True)
            temporary = link.with_suffix(".tmp")
            temporary.unlink(missing_ok=True)
            os.link(archive / "best.pt", temporary)
            temporary.replace(link)
            fits.append({"seed": seed, **checkpoint})
            _write_json(report, result)
            meter.advance(1)
        for i, seed in enumerate(FINE_SEEDS):
            for arm, lr in ARMS.items():
                name = f"ami-transfer-{arm}-{seed}"
                config = {**CONFIG, "architecture": "a", "seed": seed, "lr": lr}
                if arm != "scratch":
                    config["warm_start"] = f"ami-transfer-source-{PRETRAIN_SEEDS[i % len(PRETRAIN_SEEDS)]}"
                checkpoint, scores = fit_and_score(finish, name, config, cohort, COHORTS, exports / name, device)
                for split, (probability, truth) in scores.items():
                    if Counter(truth) != Counter({0: prior[split]["labels"]["hold"], 1: prior[split]["labels"]["shift"]}):
                        raise ValueError(f"unexpected labels in {split}")
                    fixed_labels[split] = truth
                    all_scores[split][arm].append(probability)
                threshold = _threshold(*scores["calibration"])
                downstream.append({"arm": arm, "config": config, **checkpoint, "threshold": threshold,
                                   **{split: {**_stats(*pairs, threshold), "partial_auc": partial_auc(*pairs)}
                                      for split, pairs in scores.items()}})
                _write_json(report, result)
                meter.advance(1)
        result["paired_meeting_group_bootstrap"] = {
            split: paired_intervals(all_scores[split], groups[split], fixed_labels[split])
            for split in COHORTS if split != "calibration"
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
    with (root / "jobs" / "ami-transfer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, args.pilot, "cuda"))
