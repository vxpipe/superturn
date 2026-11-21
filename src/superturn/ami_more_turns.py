"""Test additional annotation-derived turns at a fixed training-step budget."""

import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .ami_crossval import BASE, SEEDS, paired_deltas, summarize, summarize_runs, write_shards
from .ami_pilot import _annotation_labels, acoustic_status, fit_and_score
from .ami_transfer import partial_auc
from .audio import read_audio
from .augment import condition
from .labels import read_rows
from .progress import Progress
from .shards import clip, seed_for
from .sources import data_root

ARMS = ("strict", "expanded")
CAP = 12
SCREEN = {"dominance": 1.2, "quiet_ratio": 0.65}


def extra_turns(baseline, candidates):
    """Add unique turns from fit voices, with at most twelve cases per voice."""
    counts = Counter(row["speaker"] for row in baseline)
    allowed = {(row["speaker"], row["group"]) for row in baseline}
    selected = {(row["audio"], row["pause_start"]) for row in baseline}
    extras = []
    for row in sorted(candidates, key=lambda row: seed_for("ami-extra-turns-v1", row.audio, row.pause_start)):
        key = row.audio, row.pause_start
        if (row.speaker, row.group) in allowed and counts[row.speaker] < CAP and key not in selected:
            extras.append(row)
            counts[row.speaker] += 1
            selected.add(key)
    return extras


def run(root, crossval, device):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI additional-turn pilot uses the training server")
    if len(crossval) != 16 or any(char not in "0123456789abcdef" for char in crossval):
        raise ValueError("cross-validation report ID must be 16 lowercase hex characters")
    origin = root / "reports" / "ami-crossval" / f"{crossval}.json"
    original = json.loads(origin.read_text())
    controls = {(row["fold"], row["seed"]): row["checkpoint_sha256"]
                for row in original["runs"] if row["variant"] == "gru-6s"}
    folds = root / "shards" / "ami-crossval" / crossval
    annotation = root / "raw" / "ami_annotations"
    inputs = {"labels": root / "labels" / "ami.jsonl", "raw_manifest": root / "raw" / "ami" / "MANIFEST.json",
              "annotations": next(annotation.glob("*.zip"))}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    prior = json.loads((root / "reports" / "ami-calibration" / f'{original["pilot"]}.json').read_text())
    if hashes != prior["inputs"]:
        raise ValueError("AMI recording or annotation selection changed since calibration")
    protocol = {"crossval": crossval, "screen": SCREEN, "max_new_voice_total": CAP,
                "inputs": hashes, "config": {**BASE, "window": 6},
                "code": {name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
                         for name in ("ami_more_turns", "ami_pilot", "ami_crossval", "ami_reference",
                                      "ami_transfer", "train", "model", "shards", "audio", "augment")}}
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    staging = root / "shards" / "ami-more-turns" / key
    exports = root / "exports" / "ami-more-turns" / key
    report = root / "reports" / "ami-more-turns" / f"{key}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    prepared = staging / "prepared.json"
    if prepared.is_file():
        data = json.loads(prepared.read_text())
    else:
        labels = [row for row in read_rows(inputs["labels"]) if row.split == "train"]
        candidates, _, counts, _ = _annotation_labels(root, labels, [], screen=partial(acoustic_status, **SCREEN))
        data = {"screen_counts": counts, "folds": []}
        cache = lru_cache(maxsize=6)(read_audio)
        meter = Progress("ami-more-turns-prepare", original["folds"], "folds")
        try:
            for fold, info in enumerate(original["splits"]):
                source = folds / f"fold-{fold}"
                folder = staging / f"fold-{fold}"
                train_files = paths(source, "train")
                baseline = [row for batch in records(train_files) for row in batch]
                # Annotated rows originally store the same connected group as the pilot.
                extras = extra_turns(baseline, candidates)
                if not extras:
                    raise ValueError(f"no additional supported voices/turns for fold {fold}")
                render = [dict(asdict(row), condition="telephone", pcm=condition(
                    clip(row, cache(root / row.audio)), "telephone", 0).tobytes()) for row in
                    sorted(extras, key=lambda row: (row.audio, row.pause_start))]
                expanded = baseline + render
                # Distribute the new rows across batches instead of appending a new block.
                expanded.sort(key=lambda row: seed_for("ami-extra-turns-order-v1", row["audio"], row["pause_start"]))
                digest = write_shards(folder, expanded, "train")
                valid = folder / "shards" / "validation"
                if not valid.is_symlink():
                    valid.symlink_to(source / "shards" / "validation", target_is_directory=True)
                _write_json(folder / "shards" / "manifest.json", {
                    "sha256_by_split": {"train": digest, "validation": info["validation"]["sha256"]},
                })
                batch = BASE["batch_size"]
                steps = BASE["epochs"] * sum((pq.ParquetFile(path).metadata.num_rows + batch - 1) // batch
                                              for path in train_files)
                data["folds"].append({"steps": steps, "strict": summarize(baseline), "expanded": summarize(expanded),
                                      "additional": summarize(render), "expanded_sha256": digest})
                meter.advance(1)
            _write_json(prepared, data)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()
    result = {**protocol, **data, "runs": []}
    predictions, meta = {}, {}
    meter = Progress("ami-more-turns", original["folds"] * len(SEEDS) * len(ARMS), "runs")
    try:
        for fold, info in enumerate(original["splits"]):
            source = folds / f"fold-{fold}"
            groups = [row["group"] for batch in records(paths(source, "out-of-fold"), columns=["group"])
                      for row in batch]
            for seed in SEEDS:
                for arm in ARMS:
                    if arm == "strict":
                        folder, name = source, f"cv-{fold}-gru-6s-{seed}"
                        config = {**protocol["config"], "seed": seed}
                    else:
                        folder, name = staging / f"fold-{fold}", f"more-turns-{fold}-{seed}"
                        config = {**protocol["config"], "seed": seed, "max_steps": data["folds"][fold]["steps"]}
                    checkpoint, scores = fit_and_score(folder, name, config, source, ("validation", "out-of-fold"),
                                                       exports / name, device, expected_checkpoint=(
                                                           controls[fold, seed] if arm == "strict" else None))
                    probability, truth = scores["out-of-fold"]
                    for split, (_, labels) in scores.items():
                        if Counter(labels) != Counter({0: info[split]["labels"]["hold"], 1: info[split]["labels"]["shift"]}):
                            raise ValueError(f"fold {fold} {split} labels changed")
                    if meta.setdefault(fold, (truth, groups)) != (truth, groups):
                        raise ValueError(f"fold {fold} predictions are misaligned")
                    predictions[fold, arm, seed] = probability
                    threshold = _threshold(*scores["validation"])
                    evaluation = {**_stats(probability, truth, threshold), "partial_auc": partial_auc(probability, truth)}
                    result["runs"].append({"fold": fold, "seed": seed, "arm": arm, **checkpoint,
                                           "threshold": threshold, "evaluation": evaluation})
                    _write_json(report, result)
                    meter.advance(1)
        result["summary"] = {arm: summarize_runs([row for row in result["runs"] if row["arm"] == arm]) for arm in ARMS}
        result["paired_fold_bootstrap"] = paired_deltas(predictions, meta, variants=ARMS, baseline="strict")
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
    parser.add_argument("crossval", help="16-character cross-validation report ID")
    args = parser.parse_args()
    root = data_root()
    with (root / "jobs" / "ami-more-turns.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, args.crossval, "cuda"))
