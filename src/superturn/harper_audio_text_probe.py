"""Measure the value of causal ASR partials in a compact audio/text probe."""

import argparse
from collections import defaultdict
import hashlib
import json
import wave

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch

from .ami_calibration import _write_json
from .ami_transfer import partial_auc
from .audio import RATE
from .dense_teacher import file_digest
from .eot_domain_probe import load_validated_models
from .harper_partial_stt import LAGS, english_membership, pinned_membership, tick_at
from .own_head_distillation import BASELINE, student_scores
from .progress import Progress
from .sources import data_root
from .train import _stats, _threshold
from .turn1_mini_probe import GAP, upsample_prefix
from .wav2vec_harper_probe import caller_folds, pause_rows, rates

WIDTH = 2048
PENALTY = .01


def byte_features(values):
    partial = bytes(values).lstrip(b"\0")
    features = np.zeros(WIDTH, dtype=np.float64)
    if partial:
        partial = b"\0" + partial + b"\0"
        for width in (2, 3, 4):
            for i in range(len(partial) - width + 1):
                digest = hashlib.blake2s(partial[i:i + width], digest_size=4).digest()
                bucket = int.from_bytes(digest, "little") % WIDTH
                features[bucket] += 1 if digest[-1] & 128 else -1
        features /= max(np.linalg.norm(features), 1)
    return features


def fit(features, labels, *, acoustic):
    offset = features[:, 0].mean() if acoustic else 0
    scale = max(features[:, 0].std(), 1e-8) if acoustic else 1
    if acoustic:
        features = features.copy()
        features[:, 0] = (features[:, 0] - offset) / scale

    def objective(weights):
        beta, intercept = weights[:-1], weights[-1]
        logits = features @ beta + intercept
        residual = expit(logits) - labels
        loss = np.mean(np.logaddexp(0, logits) - labels * logits) + PENALTY / 2 * (beta @ beta)
        gradient = np.r_[features.T @ residual / len(labels) + PENALTY * beta, residual.mean()]
        return loss, gradient

    result = minimize(objective, np.zeros(features.shape[1] + 1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    weights = result.x
    if acoustic:
        weights[0] /= scale
        weights[-1] -= offset * weights[0]
    return weights


def metrics(scores, labels, threshold=None):
    result = {**rates(scores, labels), "partial_auc_to_5pct": partial_auc(
        scores.tolist(), labels.tolist(), max_fpr=.05)}
    if threshold is not None:
        result["fixed"] = _stats(scores.tolist(), labels.tolist(), threshold)
    return result


def permuted(matrix, callers, seed):
    rng = np.random.default_rng(seed)
    result = matrix.copy()
    nonblank = np.any(matrix != 0, axis=1)
    for caller in np.unique(callers):
        indices = np.flatnonzero((callers == caller) & nonblank)
        result[indices] = matrix[rng.permutation(indices)]
    return result


def caller_audio(root, rows):
    by_call = defaultdict(list)
    for index, call, edge, label, caller in rows:
        by_call[call].append((index, edge, label, caller))
    clips = [None] * len(rows)
    for call, pauses in by_call.items():
        path = root / "raw/harper-valley/data/audio/caller" / f"{call}.wav"
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (8000, 1, 2):
                raise ValueError("unexpected native caller track")
            track = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
        spans = [(upsample_prefix(track, edge + GAP), edge, int(label == "shift"), caller)
                 for _, edge, label, caller in pauses]
        for (index, *_), (clip, _, _) in zip(pauses, pause_rows(spans), strict=True):
            clips[index] = clip
    return np.stack(clips).astype(np.float32)


def features_for(records, membership, lag):
    features = []
    for record, (index, _, edge, _, _) in zip(records, membership, strict=True):
        decision = edge + GAP
        snapshot = record["snapshots"][str(lag)]
        cutoff = tick_at(decision, lag)
        values = snapshot["guarded_text_bytes"]
        if (record["ordinal"] != index or record["decision_sample"] != decision
                or snapshot["audio_sample"] != cutoff or len(values) != 128
                or snapshot["publication_sample"] != (cutoff + lag * RATE // 1000 if cutoff else None)
                or (not cutoff and any(values))):
            raise ValueError("transcript does not match the pinned pause and publication clock")
        features.append(byte_features(values))
    return np.stack(features)


def evaluate(train_audio, train_text, labels, callers, dev_audio, dev_text, dev_labels, dev_callers):
    folds = caller_folds(callers)
    train = {"audio": train_audio, "text": train_text,
             "audio_text": np.column_stack((train_audio, train_text))}
    dev = {"audio": dev_audio, "text": dev_text,
           "audio_text": np.column_stack((dev_audio, dev_text))}
    for seed in range(10):
        name = f"shuffled_text_{seed}"
        train[name] = np.column_stack((train_audio, permuted(train_text, callers, seed)))
        dev[name] = np.column_stack((dev_audio, permuted(dev_text, dev_callers, seed)))
    report, frozen = {}, {}
    for arm, matrix in train.items():
        oof = np.empty(len(labels))
        for fold in np.unique(folds):
            predict = folds == fold
            weights = fit(matrix[~predict], labels[~predict], acoustic=arm != "text")
            oof[predict] = matrix[predict] @ weights[:-1] + weights[-1]
        threshold = _threshold(oof.tolist(), labels.tolist())
        weights = fit(matrix, labels, acoustic=arm != "text")
        scores = dev[arm] @ weights[:-1] + weights[-1]
        report[arm] = {"caller_oof": metrics(oof, labels), "validation": metrics(scores, dev_labels, threshold)}
        if arm in ("audio", "audio_text", "text"):
            frozen[arm] = {"weights": weights.tolist(), "threshold_logit": threshold,
                           "train_oof_logits": oof.tolist(), "validation_logits": scores.tolist()}
        if arm == "audio_text":
            report["audio_text_zero_text"] = {"validation": metrics(
                dev_audio[:, 0] * weights[0] + weights[-1], dev_labels, threshold)}
    return report, frozen


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this probe requires staged Vast data")
    provenance_path = root / "reports/harper-partial-stt.json"
    provenance = json.loads(provenance_path.read_text())
    snapshots_path = root / provenance["snapshot_file"]
    if file_digest(snapshots_path) != provenance["snapshot_file_sha256"]:
        raise ValueError("the causal replay artifact changed")
    snapshots = json.loads(snapshots_path.read_text())
    models, baseline_provenance = load_validated_models(root, [BASELINE])
    (name, audio_model, _), = models
    receipt = json.loads((root / "runs" / name / "finished.json").read_text())
    manifest_path = root / "shards/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    allowed = {"ami", "common_voice", "emilia_yodas", "voxpopuli"}
    train_sources = {row["source"] for row in manifest["speaker_report"] if row["retained_labels_by_split"]["train"]}
    if (name != BASELINE or receipt["signature"]["warm_start_sha256"] is not None
            or file_digest(manifest_path) != receipt["signature"]["shards"]
            or receipt["checkpoint_sha256"] != baseline_provenance["checkpoints"][name]["sha256"]
            or not train_sources or not train_sources <= allowed):
        raise ValueError("audio baseline does not have independently trained provenance")
    roles = {}
    for role in ("train", "validation"):
        members, digest = pinned_membership(root, role)
        if digest != provenance[f"{role}_membership_sha256"]:
            raise ValueError("the transcripts do not match the frozen HarperValley view")
        audio = student_scores(audio_model, caller_audio(root, members), device="cpu")
        roles[role] = (members, logit(np.clip(audio, 1e-5, 1 - 1e-5))[:, None],
                       np.array([label == "shift" for _, _, _, label, _ in members], dtype=np.int8),
                       np.array([caller for *_, caller in members]))
    train_rows, train_audio, labels, callers = roles["train"]
    dev_rows, dev_audio, dev_labels, dev_callers = roles["validation"]
    if set(callers) & set(dev_callers):
        raise ValueError("HarperValley callers cross train and validation")
    results, heads = {}, {}
    meter = Progress("harper-audio-text-probe", len(LAGS), "lag assumptions")
    for lag in LAGS:
        results[str(lag)], heads[str(lag)] = evaluate(
            train_audio, features_for(snapshots["train"], train_rows, lag), labels,
            callers, dev_audio, features_for(snapshots["validation"], dev_rows, lag), dev_labels, dev_callers)
        meter.advance(1)
    meter.finish()
    frozen = heads["300"]["audio_text"]
    weights = np.asarray(frozen["weights"])
    sensitivity = {}
    for lag in LAGS:
        text = features_for(snapshots["validation"], dev_rows, lag)
        score = np.column_stack((dev_audio, text)) @ weights[:-1] + weights[-1]
        sensitivity[str(lag)] = metrics(score, dev_labels, frozen["threshold_logit"])
    path = root / "runs/harper-audio-text-probe/heads.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, heads)
    report = {"scope": "English HarperValley diagnostic; 300 ms assumed STT lag is primary",
              "audio_baseline": baseline_provenance, "baseline_train_sources": sorted(train_sources),
              "text_width": WIDTH, "head_l2": PENALTY,
              "snapshot_report_sha256": file_digest(provenance_path), "metrics": results,
              "frozen_300_validation_lag_sensitivity": sensitivity,
              "head_file": path.relative_to(root).as_posix(), "head_file_sha256": file_digest(path)}
    _write_json(root / "reports/harper-audio-text-probe.json", report)
    return report


def score_english(root):
    reports = root / "reports"
    fit_path = reports / "harper-audio-text-probe.json"
    fit_report = json.loads(fit_path.read_text())
    head_file = root / fit_report["head_file"]
    if file_digest(head_file) != fit_report["head_file_sha256"]:
        raise ValueError("the fitted fusion heads changed")
    heads = json.loads(head_file.read_text())["300"]
    replay_path = reports / "english-partial-stt.json"
    replay = json.loads(replay_path.read_text())
    snapshots_path = root / replay["snapshot_file"]
    rows, tracks, digest = english_membership(root)
    if digest != replay["english_membership_sha256"] or file_digest(snapshots_path) != replay["snapshot_file_sha256"]:
        raise ValueError("the English replay membership changed")
    snapshots = json.loads(snapshots_path.read_text())["english"]
    models, provenance = load_validated_models(root, [BASELINE])
    (_, model, _), = models
    if provenance != fit_report["audio_baseline"]:
        raise ValueError("the frozen audio baseline changed")
    clips, labels, _ = zip(*pause_rows((tracks[group], edge, int(label == "shift"), group)
                                       for _, group, edge, label, _ in rows), strict=True)
    audio = student_scores(model, np.stack(clips).astype(np.float32), device="cpu")
    audio = logit(np.clip(audio, 1e-5, 1 - 1e-5))[:, None]
    labels = np.asarray(labels)
    scores = {}
    for lag in LAGS:
        text = features_for(snapshots, rows, lag)
        matrices = {"audio": audio, "text": text, "audio_text": np.column_stack((audio, text)),
                    "audio_text_zero_text": np.column_stack((audio, np.zeros_like(text)))}
        scores[str(lag)] = {}
        for arm, features in matrices.items():
            head = heads[arm.removesuffix("_zero_text")]
            weights = np.asarray(head["weights"])
            score = features @ weights[:-1] + weights[-1]
            scores[str(lag)][arm] = metrics(score, labels, head["threshold_logit"])
    if scores["300"]["audio"]["recalls_at_2_5_5_7_5pct"][1] != .135:
        raise ValueError("frozen audio-only baseline failed to reproduce on English")
    result = {"scope": "frozen Harper fusion evaluated on English development",
              "replay_sha256": file_digest(replay_path), "fit_report_sha256": file_digest(fit_path),
              "metrics": scores}
    _write_json(reports / "harper-audio-text-english.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = score_english(data_root()) if args.english else run(data_root())
    print(json.dumps(report["metrics"]["300"], indent=2))
