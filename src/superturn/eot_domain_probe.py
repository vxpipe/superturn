"""Compare AMI-calibrated detectors on LiveKit's general English task-oriented turns."""

import hashlib
import json
from statistics import mean

import numpy as np

from .ami_calibration import _write_json
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
        models.append((model, entry["threshold"]))
    if len(models) != 10:
        raise ValueError("expected all five AMI GRU folds and both seeds")
    return models


def summarize(scores, truth, thresholds):
    from .train import _stats

    points = [_stats(scores, truth, threshold) for threshold in thresholds]
    return {
        "auc": points[0]["auc"],
        "partial_auc_to_5pct": partial_auc(scores, truth, max_fpr=0.05),
        "recall": mean(point["recall"] for point in points),
        "false_complete": mean(point["false_complete"] for point in points),
    }


def run(root):
    import pyarrow.parquet as pq
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("LiveKit benchmark uses the existing training server")
    torch.set_num_threads(1)
    data = root / "raw" / "livekit_eot_diagnostic" / "data/en/validation-00000-of-00002.parquet"
    with data.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError("LiveKit benchmark shard differs from the pinned revision")
    source = pq.ParquetFile(data)
    if source.metadata.num_rows != 400 or {"cohort", "silence_spans"} - set(source.schema_arrow.names):
        raise ValueError("expected the 400-turn English general benchmark shard")
    predict, defaults, smart_hash = load_detectors(root)
    grus = load_grus(root)
    external = json.loads((root / "reports" / "ami-external-turn-baselines.json").read_text())
    if external["smart_turn_sha256"] != smart_hash:
        raise ValueError("Smart Turn differs from the AMI-calibrated checkpoint")
    thresholds = {name: [fold["threshold"] for fold in external["results"][f"{name}/300ms/calibrated"]["folds"]]
                  for name in defaults}
    result = {"turns": source.metadata.num_rows, "decision_ms": DECISION_MS, "dataset_sha256": DATASET_SHA256,
              "threshold_source": "AMI calibration groups", "smart_turn_sha256": smart_hash, "conditions": {}}
    conditions = ("clean", "telephone")
    meter = Progress("livekit-eot-domain-probe", source.metadata.num_rows * len(conditions), "turns")
    try:
        for audio_condition in conditions:
            truth = []
            current = {name: [] for name in defaults}
            student = [[] for _ in grus]
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
                        outputs = predict({"pause_start": edge}, DECISION_MS, audio, turn_samples=edge + GAP)
                        for name, score in outputs.items():
                            current[name].append(score)
                        batch_audio.append(audio[-6 * RATE:])
                        batch_labels.append(int(label == "eot"))
                if batch_audio:
                    truth.extend(batch_labels)
                    audio = torch.from_numpy(np.stack(batch_audio))
                    elapsed = torch.full((len(audio), 1), float(DECISION_MS))
                    with torch.inference_mode():
                        for scores, (model, _) in zip(student, grus, strict=True):
                            scores.extend(model(audio, elapsed).flatten().tolist())
                meter.advance(batch.num_rows)
            own = [summarize(scores, truth, [threshold]) for scores, (_, threshold) in zip(student, grus, strict=True)]
            result["conditions"][audio_condition] = {
                "holds": len(truth) - sum(truth), "shifts": sum(truth),
                "superturn-6s": {key: mean(fit[key] for fit in own) for key in own[0]},
                **{name: {"calibrated": summarize(scores, truth, thresholds[name]),
                          "default": summarize(scores, truth, [defaults[name]])}
                   for name, scores in current.items()},
            }
        report = root / "reports" / "livekit-eot-domain-probe.json"
        _write_json(report, result)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(run(data_root()))
