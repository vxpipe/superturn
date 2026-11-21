"""Compare synthetic and natural extra holds with a matched AMI control."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ami_pilot import GAP, train_and_score
from .augment import condition
from .progress import Progress
from .shards import CONTEXT, SHARD_ROWS, seed_for
from .sources import data_root


def balanced_quotas(capacities, total):
    quotas = Counter()
    for _ in range(total):
        available = [language for language, capacity in capacities.items() if quotas[language] < capacity]
        if not available:
            raise ValueError("not enough language-matched auxiliary holds")
        language = min(available, key=lambda language: (quotas[language], language))
        quotas[language] += 1
    return dict(sorted(quotas.items()))


def _identity(row):
    return row["source"], row["audio"], row["pause_start"]


def _auxiliary(files, total):
    read_sources = ["common_voice"]
    natural_sources = ["emilia_yodas", "voxpopuli"]
    filters = [("source", "in", read_sources + natural_sources), ("label", "=", "hold"),
               ("pause_samples", ">=", GAP)]
    columns = ["source", "language", "speaker", "audio", "pause_start", "silence_start"]
    groups = {name: defaultdict(dict) for name in ("read", "natural")}
    for path in files:
        for row in pq.read_table(path, columns=columns, filters=filters).to_pylist():
            arm = "read" if row["source"] in read_sources else "natural"
            if (row["silence_start"] is not None) != (arm == "read"):
                continue
            if not row["speaker"] or row["speaker"] == "unknown":
                continue
            identity = _identity(row)
            option = (seed_for("ami-mixture-hold-v1", *identity), identity, path)
            by_voice = groups[arm][row["language"]]
            voice = row["source"], row["speaker"]
            if voice not in by_voice or option < by_voice[voice]:
                by_voice[voice] = option
    shared = groups["read"].keys() & groups["natural"].keys()
    quotas = balanced_quotas({lang: min(len(groups[arm][lang]) for arm in groups) for lang in shared}, total)
    output = {}
    for arm, languages in groups.items():
        chosen = [entry for language, count in quotas.items()
                  for entry in sorted(languages[language].values())[:count]]
        with_pcm = {}
        files_to_ids = defaultdict(set)
        for _, identity, path in chosen:
            files_to_ids[path].add(identity)
        for path, wanted in files_to_ids.items():
            for row in pq.read_table(path, filters=filters).to_pylist():
                identity = _identity(row)
                if identity in wanted:
                    pcm = np.frombuffer(row["pcm"], "<i2").copy()
                    pcm[CONTEXT + GAP:] = 0
                    with_pcm[identity] = {**row, "pause_samples": GAP, "condition": "telephone",
                                          "pcm": condition(pcm, "telephone", 0).tobytes()}
        output[arm] = [with_pcm[identity] for _, identity, _ in chosen]
    return output, quotas


def _arm(root, rows, validation_files, validation_hash):
    directory = root / "shards"
    train_path = directory / "train" / "clean"
    valid_path = directory / "validation" / "clean"
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for index in range(0, len(rows), SHARD_ROWS):
        batch = rows[index:index + SHARD_ROWS]
        for row in batch:
            digest.update(json.dumps({key: value for key, value in row.items() if key != "pcm"},
                                     sort_keys=True).encode())
            digest.update(row["pcm"])
        file = train_path / f"{index // SHARD_ROWS:06d}.parquet"
        temporary = file.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(batch), temporary, compression="zstd")
        temporary.replace(file)
    for path in validation_files:
        link = valid_path / path.name
        if not link.exists():
            os.link(path, link)
    manifest = {"sha256_by_split": {"train": digest.hexdigest(), "validation": validation_hash}}
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest["sha256_by_split"]["train"]


def run(root, pilot, device, *, epochs=20):
    from .train import paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI mixture diagnostic needs the training server")
    if len(pilot) != 16 or any(char not in "0123456789abcdef" for char in pilot):
        raise ValueError("pilot must be a 16-character report ID")
    if epochs < 1:
        raise ValueError("epochs must be at least one")
    pilot_root = root / "shards" / "ami-pilot" / pilot
    prepared = json.loads((pilot_root / "prepared.json").read_text())
    validation = pilot_root / "evaluation"
    validation_files = paths(validation, "validation")
    if not validation_files or not paths(validation, "crosscheck"):
        raise ValueError("both frozen pilot validation cohorts are required")
    baseline = [row for batch in records(paths(pilot_root / "current", "train")) for row in batch]
    if len(baseline) != prepared["matched_training"]["current"]["examples"]:
        raise ValueError("pilot training rows do not match the prepared cohort")
    holds = [row for row in baseline if row["label"] == "hold"]
    shifts = [row for row in baseline if row["label"] == "shift"]
    auxiliary, quotas = _auxiliary(paths(root, "train"), len(holds))
    manifest = (root / "shards" / "manifest.json").read_bytes()
    config = {"architecture": "a", "parameters": 150000, "window": 3, "epochs": epochs,
              "batch_size": 16, "augmentation": False}
    signature = hashlib.sha256(pilot.encode() + manifest + json.dumps(config, sort_keys=True).encode())
    for source in (Path(__file__), Path(__file__).with_name("ami_pilot.py")):
        signature.update(source.read_bytes())
    mixture_id = signature.hexdigest()[:16]
    staging = root / "shards" / "ami-mixture" / mixture_id
    destination = root / "exports" / "ami-mixture" / mixture_id
    output = root / "reports" / "ami-mixture" / f"{mixture_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    treatments = {"repeat_ami": holds, **auxiliary}
    report = {"label_pilot": pilot, "case_counts": {"core": len(baseline), "extra_holds": len(holds),
                                                   "repeated_shifts": len(shifts)},
              "auxiliary_language_counts": quotas, "language_weighting": "uniform in all arms",
              "config": config,
              "validation": prepared["validation"],
              "original_supported_crosscheck": prepared["original_supported_crosscheck"],
              "original_shards_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
              "arms": {}, "runs": []}
    meter = Progress("ami-mixture-pilot", len(treatments) * 2, "runs")
    try:
        for arm, extra in treatments.items():
            folder = staging / arm
            # The train reader weights by its language field. Keep one weight
            # group for this diagnostic and retain the true language separately.
            rows = [{**row, "source_language": row["language"], "language": "und"}
                    for row in baseline + shifts + extra]
            training_hash = _arm(folder, rows, validation_files, prepared["validation"]["sha256"])
            report["arms"][arm] = {
                "labels": dict(Counter(row["label"] for row in rows)),
                "sources": dict(Counter(row["source"] for row in rows)),
                "training_sha256": training_hash,
            }
            for seed in (17, 29):
                name = f"ami-mix-{arm}-{seed}"
                result = train_and_score(folder, name, {**config, "seed": seed}, validation,
                                         destination / name, device)
                report["runs"].append({"arm": arm, "seed": seed, **result})
                temporary = output.with_suffix(".tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", help="16-character completed AMI label pilot ID")
    parser.add_argument("--epochs", type=int, default=20, help="epochs per arm and seed")
    args = parser.parse_args()
    root = data_root()
    with (root / "jobs" / "ami-mixture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(run(root, args.pilot, "cuda", epochs=args.epochs))
