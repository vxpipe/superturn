"""Compare AMI-calibrated detectors on LiveKit's general English task-oriented turns."""

import argparse
import hashlib
import json
from statistics import mean

import numpy as np

from .ami_calibration import _write_json
from .ami_deadline import retrospective_limits
from .ami_external_baselines import load_detectors
from .ami_text_signal import CROSSVAL
from .ami_transfer import partial_auc
from .audio import RATE, read_audio
from .augment import condition
from .progress import Progress
from .shards import CONTEXT, TAIL
from .sources import data_root

DECISION_MS = 300
GAP = DECISION_MS * RATE // 1000
DATASET_SHA256 = "52bf7ff71e83d0214d744693839f8a6b14e78e0f26db20dfdb7f589453a50103"
REFERENCES = ("smart-turn-v3.2", "livekit-v1.1-mini")


def decision_audio(pcm, edge, audio_condition):
    stop = edge + GAP
    history = pcm[max(0, edge - CONTEXT):stop]
    padded = np.pad(history, (CONTEXT + GAP - len(history), TAIL - GAP))
    rendered = condition(padded, audio_condition, 0)
    return rendered[GAP:CONTEXT + GAP].astype(np.float32) / 32768


def load_grus(root):
    import torch

    from .model import build

    report = json.loads((root / "reports" / "ami-crossval" / f"{CROSSVAL}.json").read_text())
    models = []
    for entry in report["runs"]:
        if entry["variant"] != "gru-6s":
            continue
        name = f'cv-{entry["fold"]}-gru-6s-{entry["seed"]}'
        checkpoint = root / "exports" / "ami-crossval" / CROSSVAL / name / "best.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint differs from AMI cross-validation")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build(saved["config"]).eval()
        model.load_state_dict(saved["state"])
        models.append((name, model, entry["threshold"]))
    if len(models) != 10:
        raise ValueError("expected all five AMI GRU folds and both seeds")
    return models


def load_validated_models(root, names):
    import torch

    from .model import build

    manifest = json.loads((root / "shards" / "manifest.json").read_text())
    validation_hash = manifest["sha256_by_split"]["validation"]
    models = []
    checkpoints = {}
    for name in names:
        report = root / "reports" / f"{name}-float-ami-validation" / "metrics.json"
        record = json.loads(report.read_text())
        checkpoint = root / "runs" / name / "best.pt"
        with checkpoint.open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        expected = {"name": name, "checkpoint_sha256": checkpoint_hash, "split": "validation",
                    "source": "ami", "condition": "telephone", "validation_sha256": validation_hash,
                    "decision_ms": DECISION_MS}
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError(f"{name} does not match its 300 ms validation report")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build(saved["config"]).eval()
        model.load_state_dict(saved["state"])
        threshold = next(point["threshold"] for point in record["thresholds"]
                         if point["origin"] == "ami_telephone_validation_fit")
        models.append((name, model, threshold))
        checkpoints[name] = {"sha256": checkpoint_hash, "threshold": threshold, "parameters": saved["parameters"]}
    return models, {"validation_sha256": validation_hash, "checkpoints": checkpoints}


def summarize(scores, truth, thresholds, *, roc_limits=False):
    from .train import _stats

    points = [_stats(scores, truth, threshold) for threshold in thresholds]
    metrics = {
        "auc": points[0]["auc"],
        "partial_auc_to_5pct": partial_auc(scores, truth, max_fpr=0.05),
        "recall": mean(point["recall"] for point in points),
        "false_complete": mean(point["false_complete"] for point in points),
    }
    if roc_limits:
        metrics["retrospective_roc_limits"] = retrospective_limits(scores, truth)
    return metrics


def run(root, *, stage_one=False, checkpoint_name=None, window_names=(), replay_references=False):
    import pyarrow.parquet as pq
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("LiveKit benchmark uses the existing training server")
    if sum(map(bool, (stage_one, checkpoint_name, window_names))) > 1:
        raise ValueError("choose one benchmark checkpoint scope")
    if replay_references and not checkpoint_name:
        raise ValueError("replaying reference scores requires a checkpoint name")
    torch.set_num_threads(1)
    data = root / "raw" / "livekit_eot_diagnostic" / "data/en/validation-00000-of-00002.parquet"
    with data.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError("LiveKit benchmark shard differs from the pinned revision")
    source = pq.ParquetFile(data)
    if source.metadata.num_rows != 400 or {"cohort", "silence_spans"} - set(source.schema_arrow.names):
        raise ValueError("expected the 400-turn English general benchmark shard")
    reuse_reference = stage_one or bool(window_names) or replay_references
    if reuse_reference:
        if window_names:
            names = window_names
            reference_report = root / "reports" / f"{names[0]}-eot-domain-probe.json"
        elif replay_references:
            names = [checkpoint_name]
            reference_report = root / "reports" / "capacity-window-eot-domain-probe.json"
        else:
            names = [f"s1-{arch}-{size}-{seed}" for arch in ("a", "b")
                     for size in (150000, 500000, 1000000) for seed in (17, 29)]
            reference_report = root / "reports" / "livekit-eot-domain-probe.json"
        models, provenance = load_validated_models(root, names)
        reference_bytes = reference_report.read_bytes()
        reference = json.loads(reference_bytes)
        if (reference["dataset_sha256"], reference["decision_ms"]) != (digest, DECISION_MS):
            raise ValueError("held-out reference scores use a different benchmark")
        if window_names:
            previous = reference["checkpoint_provenance"]["checkpoints"][names[0]]
            if provenance["checkpoints"][names[0]]["sha256"] != previous["sha256"]:
                raise ValueError("the 3-second checkpoint differs from the reference comparison")
        predict, defaults, smart_hash = None, {}, reference["smart_turn_sha256"]
        thresholds = {}
    else:
        if checkpoint_name:
            models, provenance = load_validated_models(root, [checkpoint_name])
        else:
            models = load_grus(root)
        predict, defaults, smart_hash = load_detectors(root)
        external = json.loads((root / "reports" / "ami-external-turn-baselines.json").read_text())
        if external["smart_turn_sha256"] != smart_hash:
            raise ValueError("Smart Turn differs from the AMI-calibrated checkpoint")
        thresholds = {name: [fold["threshold"] for fold in external["results"][f"{name}/300ms/calibrated"]["folds"]]
                      for name in defaults}
    result = {"turns": source.metadata.num_rows, "decision_ms": DECISION_MS, "dataset_sha256": DATASET_SHA256,
              "threshold_source": "AMI calibration groups", "smart_turn_sha256": smart_hash, "conditions": {}}
    if reuse_reference:
        result["threshold_source"] = "original AMI telephone validation"
        result["reference_report_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        key = ("stage_one_provenance" if stage_one else
               "checkpoint_provenance" if replay_references else "window_sweep_provenance")
        result[key] = provenance
    if checkpoint_name:
        result["threshold_source"] = "original AMI telephone validation"
        result["checkpoint_provenance"] = provenance
    result["reference_threshold_source"] = "five AMI calibration-group thresholds per reference model"
    conditions = ("clean", "telephone")
    job = ("capacity-window-eot-domain-probe" if window_names else
           "stage1-eot-domain-probe" if stage_one else
           f"{checkpoint_name}-eot-domain-probe" if checkpoint_name else "livekit-eot-domain-probe")
    meter = Progress(job, source.metadata.num_rows * len(conditions), "turns")
    try:
        for audio_condition in conditions:
            truth = []
            current = {name: [] for name in defaults}
            student = [[] for _ in models]
            for batch in source.iter_batches(batch_size=8, columns=["cohort", "language", "audio", "silence_spans"]):
                batch_audio, batch_labels = [], []
                for row in batch.to_pylist():
                    if row["cohort"] != "general" or row["language"] != "en":
                        raise ValueError("benchmark source unexpectedly contains another cohort or language")
                    pcm = read_audio(row["audio"]["bytes"])
                    for span in row["silence_spans"]:
                        edge = round(span["start"] * RATE)
                        end = min(len(pcm), round(span["end"] * RATE))
                        if not 0 <= edge <= end or end - edge < GAP:
                            continue
                        label = span["label"]
                        if label not in {"eot", "hold"}:
                            raise ValueError("unknown benchmark label")
                        audio = decision_audio(pcm, edge, audio_condition)
                        if predict is not None:
                            outputs = predict({"pause_start": edge}, DECISION_MS, audio, turn_samples=edge + GAP)
                            for name, score in outputs.items():
                                current[name].append(score)
                        batch_audio.append(audio)
                        batch_labels.append(int(label == "eot"))
                if batch_audio:
                    truth.extend(batch_labels)
                    audio = torch.from_numpy(np.stack(batch_audio))
                    elapsed = torch.full((len(audio), 1), float(DECISION_MS))
                    with torch.inference_mode():
                        for scores, (_, model, _) in zip(student, models, strict=True):
                            scores.extend(model(audio[:, -model.samples:], elapsed).flatten().tolist())
                meter.advance(batch.num_rows)
            own = {name: summarize(scores, truth, [threshold], roc_limits=bool(checkpoint_name or window_names))
                   for scores, (name, _, threshold) in zip(student, models, strict=True)}
            counts = {"holds": len(truth) - sum(truth), "shifts": sum(truth)}
            if reuse_reference:
                prior = reference["conditions"][audio_condition]
                if any(prior[label] != count for label, count in counts.items()):
                    raise ValueError("held-out case counts differ from the reference evaluation")
                if replay_references:
                    result["conditions"][audio_condition] = {**counts, **own,
                                                             **{name: prior[name] for name in REFERENCES}}
                else:
                    group = "stage_one" if stage_one else "capacity_windows"
                    result["conditions"][audio_condition] = {**prior, group: own}
            else:
                result["conditions"][audio_condition] = {
                    **counts,
                    **(own if checkpoint_name else {
                        "superturn-6s": {key: mean(fit[key] for fit in own.values())
                                         for key in next(iter(own.values()))},
                    }),
                    **{name: {"calibrated": summarize(scores, truth, thresholds[name],
                                                       roc_limits=bool(checkpoint_name)),
                              "default": summarize(scores, truth, [defaults[name]])}
                       for name, scores in current.items()},
                }
        report = root / "reports" / f"{job}.json"
        _write_json(report, result)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-one", action="store_true", help="score the twelve completed Stage 1 checkpoints")
    args = parser.parse_args()
    print(run(data_root(), stage_one=args.stage_one))
