"""Score local reference detectors on the same AMI telephone development folds."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .ami_crossval import summarize_runs
from .ami_deadline import TIMES, inputs
from .ami_text_signal import CROSSVAL
from .ami_transfer import partial_auc
from .audio import RATE, read_audio
from .progress import Progress
from .sources import data_root

SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"


def load_smart_turn(root):
    import onnxruntime as ort
    from _whisper_features import compute_whisper_log_mel_features

    smart_path = root / "cache" / "eot-competitors" / "smart-turn-v3.2-cpu.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    smart_turn = ort.InferenceSession(str(smart_path), sess_options=options, providers=["CPUExecutionProvider"])

    def predict(audio):
        features = compute_whisper_log_mel_features(audio, do_normalize=True)[None]
        return float(smart_turn.run(None, {"input_features": features})[0].item())

    return predict, hashlib.sha256(smart_path.read_bytes()).hexdigest()


def load_detectors(root):
    from livekit.local_inference import EOT, EOT_MAX_SAMPLES, EOT_THRESHOLDS

    if EOT_MAX_SAMPLES != 2 * RATE:
        raise ValueError("expected the LiveKit v1.1-mini two-second audio window")
    smart_turn, checkpoint_sha256 = load_smart_turn(root)
    mini = EOT()
    defaults = {"livekit-v1.1-mini": float(EOT_THRESHOLDS["en"]),
                "smart-turn-v3.2": float(np.nextafter(np.float32(0.5), np.float32(1)))}

    def predict(row, gap, audio):
        pcm = (audio * 32768).astype(np.int16)
        valid_samples = min(len(pcm), row["pause_start"] + gap * 16)
        recent = np.ascontiguousarray(pcm[-min(valid_samples, EOT_MAX_SAMPLES):])
        # The archived, causal history approximates turn length in the meeting clips.
        mini_score = mini.predict(recent, turn_samples=valid_samples)
        return {"livekit-v1.1-mini": float(mini_score), "smart-turn-v3.2": smart_turn(audio)}

    return predict, defaults, checkpoint_sha256


def run(root):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("External AMI baselines use the training server")
    base = root / "shards" / "ami-crossval" / CROSSVAL
    reference = json.loads((root / "reports" / "ami-crossval" / f"{CROSSVAL}.json").read_text())
    predict, defaults, checkpoint_sha256 = load_detectors(root)
    cached = {}
    cached_audio = lru_cache(maxsize=6)(read_audio)
    runs = defaultdict(list)
    meter = Progress("ami-external-turn-baselines", reference["folds"], "folds")
    try:
        for fold in range(reference["folds"]):
            source = base / f"fold-{fold}"
            views = {}
            for split in ("validation", "out-of-fold"):
                rows = [row for batch in records(paths(source, split)) for row in batch]
                truth = [int(row["label"] == "shift") for row in rows]
                missing = [row for row in rows if (row["audio"], row["pause_start"]) not in cached]
                if missing:
                    clips, _ = inputs(root, missing, 8 * RATE, cached_audio)
                    for i, row in enumerate(missing):
                        cached[row["audio"], row["pause_start"]] = [predict(row, gap, clips[gap][i]) for gap in TIMES]
                views[split] = truth, [cached[row["audio"], row["pause_start"]] for row in rows]
            for model, default in defaults.items():
                for timing in ("300ms", "100-300ms"):
                    selector = (lambda values: values[-1][model]) if timing == "300ms" else (
                        lambda values: max(point[model] for point in values))
                    cal_y, cal_values = views["validation"]
                    test_y, test_values = views["out-of-fold"]
                    scores = [selector(values) for values in test_values]
                    calibrated = _threshold([selector(values) for values in cal_values], cal_y)
                    for policy, threshold in (("default", default), ("calibrated", calibrated)):
                        runs[model, timing, policy].append({
                            "fold": fold, "threshold": threshold,
                            "evaluation": {**_stats(scores, test_y, threshold),
                                           "partial_auc": partial_auc(scores, test_y, max_fpr=0.05)},
                        })
            meter.advance(1)
        expected = sum(fold["out-of-fold"]["examples"] for fold in reference["splits"])
        if len(cached) != expected:
            raise ValueError(f"expected {expected} unique AMI development rows, scored {len(cached)}")
        report = root / "reports" / "ami-external-turn-baselines.json"
        _write_json(report, {
            "crossval": CROSSVAL, "rows": len(cached), "deadlines_ms": TIMES,
            "livekit_runtime": "0.3.2", "smart_turn_revision": SMART_TURN_REVISION,
            "smart_turn_sha256": checkpoint_sha256,
            "scope": "AMI telephone development; LiveKit turn length proxied by available history, capped at 8 s",
            "partial_auc_max_fpr": 0.05,
            "results": {"/".join(key): {"summary": summarize_runs(rows), "folds": rows}
                        for key, rows in runs.items()},
        })
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(run(data_root()))
