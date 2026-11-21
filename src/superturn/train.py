"""Train and select checkpoints on validation data."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from .model import build
from .progress import Progress
from .shards import CONTEXT, seed_for
from .sources import REPO_ROOT
from .training_augmentation import TrainingAugmentation

PAUSES = (200, 300, 500)


def paths(root, split, condition="clean"):
    return sorted((root / "shards" / split / condition).glob("*.parquet"))


def records(files, batch_size=16, columns=None):
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pylist()


def eligible(row, config, verified_ami=None):
    if (verified_ami is not None and row["source"] == "ami"
            and (row["audio"], row["pause_start"], row["label"]) not in verified_ami):
        return False
    if row["label"] == "weak-shift" and not config.get("weak", True):
        return False
    if config.get("sources") and row["source"] not in config["sources"]:
        return False
    fraction = config.get("fraction", 1)
    return seed_for(row["source"], row["audio"], row["pause_start"]) % 10000 < 10000 * fraction


def _input(row, samples, silence_ms=0):
    elapsed = round(silence_ms * 16)
    if not 0 <= elapsed <= row["pause_samples"]:
        raise ValueError(f"decision silence exceeds the labelled pause: {silence_ms} ms")
    end = CONTEXT + elapsed
    return np.frombuffer(row["pcm"], "<i2")[end - samples:end]


def _auc(scores, truth):
    pairs = sorted(zip(scores, truth))
    positive = sum(truth)
    negative = len(truth) - positive
    if not positive or not negative:
        return None
    rank_sum = 0.0
    from itertools import groupby
    rank = 1
    for _, tied in groupby(pairs, key=lambda pair: pair[0]):
        labels = [label for _, label in tied]
        rank_sum += sum(labels) * (rank + (len(labels) - 1) / 2)
        rank += len(labels)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def _threshold(scores, truth, target=0.05):
    holds = sorted((score for score, label in zip(scores, truth) if not label), reverse=True)
    if not holds:
        return 0.5
    boundary = holds[min(len(holds) - 1, int(target * len(holds)))]
    return float(np.nextafter(np.float32(boundary), np.float32(np.inf)))


def _stats(scores, truth, threshold):
    holds = sum(label == 0 for label in truth)
    shifts = sum(truth)
    return {
        "holds": holds, "shifts": shifts,
        "false_complete": (sum(not label and score >= threshold
                               for score, label in zip(scores, truth)) / holds if holds else None),
        "recall": (sum(label and score >= threshold for score, label in zip(scores, truth))
                   / shifts if shifts else None),
        "auc": _auc(scores, truth),
    }


def _validation(model, files, device):
    scores, truth = [], []
    model.eval()
    with torch.inference_mode():
        for rows in records(files):
            rows = [row for row in rows if row["pause_samples"] >= 4800 and row["label"] != "weak-shift"]
            if rows:
                audio = np.stack([_input(row, model.samples, 300) for row in rows]).astype(np.float32) / 32768
                probability = model(torch.from_numpy(audio).to(device),
                                    torch.full((len(rows), 1), 300.0, device=device))
                scores.extend(probability.cpu().flatten().tolist())
                truth.extend(int(row["label"] == "shift") for row in rows)
    threshold = _threshold(scores, truth)
    return _stats(scores, truth, threshold), threshold


def _assets(root, samples, config, verified_ami=None):
    from zipfile import ZipFile
    from .audio import read_audio, write_wave

    noise = sorted((root / "raw" / "musan" / "noise" / "free-sound").rglob("*.wav"))
    if (root / ".campaign-machine").is_file() and not noise:
        raise ValueError("MUSAN Free Sound noise files are missing")
    ir_root = root / "labels" / "impulses"
    if not ir_root.exists():
        for path in (root / "raw" / "openslr26").glob("*.zip"):
            with ZipFile(path) as archive:
                entries = sorted(n for n in archive.namelist() if n.lower().endswith(".wav"))
                for index, name in enumerate(entries[::max(1, len(entries) // 256)]):
                    write_wave(ir_root / f"{index:04d}.wav", read_audio(archive.read(name)))
    voices = {}
    for batch in records(paths(root, "train")):
        for row in batch:
            if not eligible(row, config, verified_ami):
                continue
            key = (row["source"], row["speaker"])
            if key not in voices:
                voices[key] = _input(row, samples)
        if len(voices) >= 96:
            break
    return noise, sorted(ir_root.glob("*.wav")), voices


def _starting_checkpoint(root, config):
    name = config.get("warm_start")
    if name is None:
        return None
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("warm_start must be one run name")
    return root / "runs" / name / "best.pt"


def _run_signature(root, config, device):
    shard_manifest = root / "shards" / "manifest.json"
    if not shard_manifest.is_file():
        return None
    source = hashlib.sha256()
    for name in ("train.py", "model.py", "augment.py", "audio.py", "quant_aware.py",
                 "ami_training.py", "training_augmentation.py"):
        source.update(Path(__file__).with_name(name).read_bytes())
    if config.get("architecture") == "citrinet":
        source.update(Path(__file__).with_name("citrinet.py").read_bytes())
        source.update(version("nemo-toolkit").encode())
    assets = {}
    if config.get("augmentation", True):
        for name in ("musan", "openslr26"):
            manifest = root / "raw" / name / "MANIFEST.json"
            if manifest.is_file():
                assets[name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    try:
        decoder = version("av")
    except PackageNotFoundError:
        decoder = None
    warm_start = _starting_checkpoint(root, config)
    ami_hash = None
    if config.get("ami_verified_training", False):
        from .ami_training import selection_hash

        ami_hash = selection_hash(root)
    return {
        "config": config,
        "ami_verified_sha256": ami_hash,
        "warm_start_sha256": hashlib.sha256(warm_start.read_bytes()).hexdigest() if warm_start else None,
        "augmentation_assets": assets,
        "decoder": decoder,
        "device": device,
        "torch": str(torch.__version__),
        "source": source.hexdigest(),
        "shards": hashlib.sha256(shard_manifest.read_bytes()).hexdigest(),
    }


def _compatible_signature(recorded, expected):
    if recorded == expected:
        return True
    if recorded is None or {**recorded, "source": expected["source"]} != expected:
        return False
    path = REPO_ROOT / "configs" / "training-source-equivalence.json"
    if not path.is_file() or expected.get("config", {}).get("auxiliary_size"):
        return False
    equivalent = json.loads(path.read_text())
    previous = (equivalent.get("fixed_epochs_previous", [])
                if expected.get("config", {}).get("fixed_epochs", False) else equivalent["previous"])
    if isinstance(previous, str):
        previous = [previous]
    return recorded.get("source") in previous and equivalent["current"] == expected["source"]


def completed_run(root, name, config, device):
    run = root / "runs" / name
    marker, checkpoint = run / "finished.json", run / "best.pt"
    signature = _run_signature(root, config, device)
    if signature is not None and marker.is_file() and checkpoint.is_file():
        result = json.loads(marker.read_text())
        if (_compatible_signature(result.get("signature"), signature)
                and result.get("checkpoint_sha256") == hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            return checkpoint
    return None


def train(root, name, config, device):
    train_files = paths(root, "train")
    valid_files = paths(root, "validation")
    if not train_files or (not valid_files and not config.get("smoke")):
        raise ValueError("training requires train and validation shards")
    seed = config.get("seed", 17)
    torch.manual_seed(seed)
    if device == "cpu":
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build(config).to(device)
    warm_start = _starting_checkpoint(root, config)
    if warm_start is not None:
        model.load_state_dict(torch.load(warm_start, map_location="cpu", weights_only=True)["state"])
    if config["architecture"] == "citrinet":
        from .citrinet import initialize

        initialize(model, root, config["citrinet_modules"], pretrained=config["citrinet_pretrained"])
    parameter_total = sum(p.numel() for p in model.parameters())
    state_dict = type(model).state_dict
    if config.get("quantization_aware", False):
        from .quant_aware import enable, plain_state_dict

        enable(model)
        state_dict = plain_state_dict
    parameters = list(model.parameters())
    auxiliary = None
    if auxiliary_size := config.get("auxiliary_size"):
        auxiliary = torch.nn.Linear(model.decision[0].in_features, auxiliary_size).to(device)
        parameters.extend(auxiliary.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.get("lr", 0.001))
    run = root / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "finished.json"
    marker.unlink(missing_ok=True)
    signature = _run_signature(root, config, device)
    verified_ami = None
    if config.get("ami_verified_training", False):
        from .ami_training import load_selection

        verified_ami = load_selection(root)
    counts = Counter()
    train_batches = 0
    for batch in records(train_files, config.get("batch_size", 16),
                         columns=["source", "language", "audio", "label", "pause_start"]):
        selection = [row for row in batch if eligible(row, config, verified_ami)]
        counts.update(row["language"] for row in selection)
        train_batches += bool(selection)
    if not train_batches:
        raise ValueError("no training examples match the run configuration")

    language_normalizer = sum(count ** 0.7 for count in counts.values()) / sum(counts.values())
    augmentation = config.get("augmentation", True)
    assets = _assets(root, model.samples, config, verified_ami) if augmentation else None
    epochs = config.get("epochs", 5)
    fixed_epochs = config.get("fixed_epochs", False)
    steps = min(config.get("max_steps", train_batches * epochs), train_batches * epochs)
    meter = Progress(f"train-{name}", steps, "steps")
    metrics = []
    best = None
    completed = 0
    processors = config.get("augmentation_workers", 1)
    augmenter = TrainingAugmentation(processors, assets) if augmentation else None
    try:
        for epoch in range(epochs):
            model.train()
            rng = np.random.default_rng(seed_for(seed, epoch))
            loss_total, batches = 0.0, 0
            for batch in records(rng.permutation(train_files), config.get("batch_size", 16)):
                selected = [row for row in batch if eligible(row, config, verified_ami)]
                if not selected:
                    continue
                audio, jobs, gaps, truth, weights, auxiliary_truth = [], [], [], [], [], []
                for index in rng.permutation(len(selected)):
                    row = selected[index]
                    pause_samples = row["pause_samples"]
                    if verified_ami is not None and row["source"] == "ami":
                        pause_samples = min(pause_samples, verified_ami[row["audio"], row["pause_start"], row["label"]])
                    gap_ms = rng.choice([v for v in PAUSES if v * 16 <= pause_samples])
                    pcm = _input(row, model.samples, gap_ms)
                    if augmenter is not None:
                        jobs.append((pcm, int(rng.integers(2**63)),
                                     (row["source"], row["speaker"]), gap_ms * 16))
                    else:
                        audio.append(pcm)
                    # Equalize class exposure to elapsed silence within each available gap.
                    gaps.append([gap_ms])
                    positive = row["label"] != "hold"
                    truth.append([float(positive)])
                    if auxiliary is not None:
                        auxiliary_truth.append(row["auxiliary_targets"])
                    weight = counts[row["language"]] ** -0.3 / language_normalizer
                    weight *= 0.35 if row["label"] == "weak-shift" else 1
                    weight *= 1 if positive else config.get("hold_weight", 2.0)
                    weights.append([weight])
                if augmenter is not None:
                    audio = augmenter.render(jobs)
                audio = np.asarray(audio, dtype=np.float32) / 32768
                inputs = torch.from_numpy(audio).to(device)
                gap = torch.as_tensor(np.asarray(gaps), dtype=torch.float32, device=device)
                targets = torch.tensor(truth, device=device)
                weights = torch.tensor(weights, device=device)
                optimizer.zero_grad()
                if auxiliary is None:
                    logits = model.logits(inputs, gap)
                else:
                    features = model.decision_input(inputs, gap)
                    logits = model.decision(features)
                losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                loss = (losses * weights).mean()
                if auxiliary is not None:
                    auxiliary_labels = torch.as_tensor(auxiliary_truth, dtype=torch.float32, device=device)
                    auxiliary_loss = F.binary_cross_entropy_with_logits(auxiliary(features), auxiliary_labels)
                    loss = loss + config["auxiliary_weight"] * auxiliary_loss
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
                batches += 1
                completed += 1
                meter.advance(1)
                if completed >= steps:
                    break
            if fixed_epochs and completed < steps:
                metrics.append({"epoch": epoch + 1, "loss": loss_total / batches})
                print(json.dumps(metrics[-1]), flush=True)
                (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
                continue
            validation, threshold = _validation(model, valid_files, device)
            if not config.get("smoke") and validation["auc"] is None:
                raise ValueError("validation requires both 300 ms holds and shifts")
            record = {"epoch": epoch + 1, "loss": loss_total / batches, "threshold": threshold, **validation}
            metrics.append(record)
            print(json.dumps(record), flush=True)
            score = ((record["recall"] or 0), (record["auc"] or 0), -record["loss"])
            if best is None or score > best:
                best = score
                checkpoint = {"config": config, "state": state_dict(model), "metrics": record,
                              "parameters": parameter_total}
                temp = run / "best.tmp"
                torch.save(checkpoint, temp)
                temp.replace(run / "best.pt")
            (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if completed >= steps:
                break
    except BaseException:
        meter.finish("failed")
        raise
    else:
        if signature is not None:
            temp = marker.with_suffix(".tmp")
            result = {"signature": signature,
                      "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()}
            temp.write_text(json.dumps(result, sort_keys=True) + "\n")
            temp.replace(marker)
        meter.finish()
    finally:
        if augmenter is not None:
            augmenter.close()
    return run / "best.pt"
