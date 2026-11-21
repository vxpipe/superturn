"""Test a newly fitted turn head over non-turn Wav2Vec2 Base representations."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import wave

import numpy as np
from scipy.special import expit
import torch
from torch.nn import functional as F
import torchaudio

from .ami_calibration import _write_json
from .audio import RATE, read_audio
from .dense_teacher import SAMPLES, file_digest
from .elapsed_turn_probe import paired_intervals
from .eot_domain_probe import GAP, decision_audio
from .harper_history_probe import recall_area
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_benchmark import pinned_split
from .model import AcousticFeatures, FRAME, HOP
from .progress import Progress
from .sources import data_root
from .train import _auc, _stats, _threshold, paths, records
from .turn1_mini_probe import english_rows, harper_rows, upsample_prefix

SEEDS = (17, 29, 43)
PENALTIES = (.01, .05, .25)
BUNDLE = "wav2vec2_fairseq_base_ls960.pth"
AMI = "bed0ba3de027429f"


def calibration_rows(root):
    cohort = json.loads((root / "reports/ami-calibration" / f"{AMI}.json").read_text())["cohorts"]["calibration"]
    digest = hashlib.sha256()
    cached = lru_cache(maxsize=6)(read_audio)
    from .labels import Label
    from .shards import CONTEXT, clip
    from .augment import condition
    from dataclasses import fields

    names = [field.name for field in fields(Label)]
    for batch in records(paths(root / "shards/ami-calibration" / AMI, "calibration")):
        for row in batch:
            metadata = {name: row[name] for name in names}
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            digest.update(row["pcm"])
            if row["label"] == "hold":
                label = Label(**metadata)
                if label.pause_samples < GAP:
                    raise ValueError("AMI calibration hold does not reach 300 ms")
                original = clip(label, cached(root / label.audio))
                end = CONTEXT + GAP
                original[end:] = 0
                pcm = condition(original, "telephone", 0)
                yield pcm[end - SAMPLES:end].astype(np.float32) / 32768, 0, row["group"]
    if digest.hexdigest() != cohort["sha256"]:
        raise ValueError("AMI calibration rows changed")


def training_rows(root):
    reports = root / "reports"
    archived = json.loads((reports / "harper-history-replication.json").read_text())
    membership = json.loads((reports / "harper-history-replication-membership.json").read_text())
    cases, _, audit = pinned_split(root, "train")
    if (hashlib.sha256(json.dumps(membership).encode()).hexdigest() != MEMBERSHIP or
            archived["membership_sha256"] != MEMBERSHIP or archived["audit_sha256"] != audit):
        raise ValueError("Harper training membership changed")
    callers = {case["id"]: case["caller"] for case in cases}
    folder = root / "raw/harper-valley/data/audio/caller"

    @lru_cache(maxsize=2)
    def native(call):
        with wave.open(str(folder / f"{call}.wav"), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (8000, 1, 2):
                raise ValueError("unexpected native Harper audio")
            return np.frombuffer(reader.readframes(reader.getnframes()), "<i2")

    for call, edge, label in membership:
        observed = upsample_prefix(native(call), edge + GAP)
        yield decision_audio(observed, edge, "telephone")[-SAMPLES:], int(label == "shift"), callers[call]


def pause_rows(rows):
    for pcm, edge, truth, group in rows:
        yield decision_audio(pcm, edge, "telephone")[-SAMPLES:], truth, group


def pooled(frames, stride, width):
    center = torch.arange(frames.shape[1], device=frames.device) * stride + (width - 1) / 2
    edge = SAMPLES - GAP
    previous = (center >= edge - RATE // 2) & (center < edge)
    silence = center >= edge
    return torch.cat((frames.mean(1), frames[:, previous].mean(1), frames[:, silence].mean(1)), dim=-1)


def speech_features(model, audio):
    audio = F.layer_norm(audio, (SAMPLES,), eps=1e-7)
    features, lengths = model.feature_extractor(audio, None)
    projection = model.encoder.feature_projection(features)
    layers = model.encoder.extract_features(features, lengths)
    return torch.stack([pooled(layer, 320, 400) for layer in (projection, *layers)], dim=1)


def extract(name, predict, audio):
    meter = Progress(f"wav2vec-features-{name}", len(audio), "pauses")
    output = []
    try:
        with torch.inference_mode():
            for batch in np.array_split(audio, (len(audio) + 31) // 32):
                output.append(predict(torch.as_tensor(batch, device="cuda")).cpu().numpy())
                meter.advance(len(batch))
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    features = np.concatenate(output)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite pooled features")
    return features


def fit(features, truth, training, evaluation, penalty, seed):
    train = torch.as_tensor(features[training]).double()
    target = torch.as_tensor(truth[training]).double()
    center, scale = train.mean(0), train.std(0, unbiased=False).clamp(min=1e-5)
    train = (train - center) / scale
    rng = torch.Generator().manual_seed(seed)
    layer = (torch.randn(train.shape[1], generator=rng, dtype=torch.float64) * .01).requires_grad_()
    weights = torch.zeros(train.shape[2], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS((layer, weights, bias), max_iter=80, line_search_fn="strong_wolfe")

    def logits(values):
        return (values.transpose(1, 2) @ layer.softmax(0)) @ weights + bias

    def update():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits(train), target)
        loss = loss + penalty * weights.square().sum() / 2 + .001 * layer.square().mean()
        loss.backward()
        return loss

    optimizer.step(update)
    with torch.no_grad():
        other = torch.as_tensor(features[evaluation]).double()
        return logits((other - center) / scale).numpy(), layer.softmax(0).tolist()


def fit_arm(features, truth, splits, caller_folds):
    train = splits["train"]
    search = {}
    for penalty in PENALTIES:
        score = np.full(len(train), np.nan)
        for fold in range(5):
            held = caller_folds == fold
            score[held], _ = fit(features, truth, train[~held], train[held], penalty, 17)
        if not np.isfinite(score).all():
            raise ValueError("incomplete held-caller head selection")
        search[penalty] = (recall_area(score, truth[train])[-1], score)
    penalty = max(PENALTIES, key=lambda value: search[value][0])
    evaluation = np.concatenate([indices for name, indices in splits.items() if name != "train"])
    repeats = [fit(features, truth, train, evaluation, penalty, seed) for seed in (17, 29, 43, 53, 71)]
    weight = np.mean([weights for _, weights in repeats], axis=0)
    score = np.full(len(truth), np.nan)
    score[train] = expit(search[penalty][1])
    score[evaluation] = np.mean([expit(values) for values, _ in repeats], axis=0)
    probe = features[train] - features[train].mean(0)
    projection = probe.transpose(0, 2, 1) @ weight.astype(np.float32)
    variance = np.linalg.eigvalsh(projection @ projection.T).clip(min=0)[::-1]
    rank = int(np.searchsorted(np.cumsum(variance), .95 * variance.sum()) + 1)
    return score, {"penalty": penalty, "selection_partial_auc": {p: value[0] for p, value in search.items()},
                   "layer_weights": weight.tolist(), "raw_pooled_rank_at_95pct": rank}


def rates(scores, truth):
    values = recall_area(scores, truth).tolist()
    return {"auc": _auc(scores.tolist(), truth.tolist()), "recalls_at_2_5_5_7_5pct": values[:3],
            "mean_recall": float(np.mean(values[:3])), "partial_auc": values[-1]}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this diagnostic uses only staged Vast audio")
    cohorts = {"train": list(training_rows(root)),
               "english": list(pause_rows(english_rows(root))),
               "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
               "calibration": list(calibration_rows(root))}
    splits, truth, groups, clips = {}, [], {}, []
    for name, rows in cohorts.items():
        audio, labels, identifiers = zip(*rows, strict=True)
        expected = {"train": (273, 255), "english": (532, 400), "harper": (40, 48), "calibration": (51, 0)}
        if (len(rows) - sum(labels), sum(labels)) != expected[name]:
            raise ValueError(f"{name} membership changed")
        splits[name] = np.arange(len(truth), len(truth) + len(rows))
        truth.extend(labels)
        groups[name] = identifiers
        clips.extend(audio)
    truth, clips = np.asarray(truth), np.stack(clips).astype(np.float32)
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    if torchaudio.__version__.split("+")[0] != "2.8.0" or bundle._path != BUNDLE or bundle.sample_rate != RATE:
        raise ValueError("unexpected encoder bundle")
    folder = root / "cache/wav2vec2-harper"
    folder.mkdir(parents=True, exist_ok=True)
    pretrained = bundle.get_model(dl_kwargs={"model_dir": str(folder), "map_location": "cpu", "weights_only": True})
    checkpoint = folder / BUNDLE
    if checkpoint.stat().st_size != 377565405:
        raise ValueError("unexpected publisher checkpoint size")
    voice, caller = np.unique(groups["train"], return_inverse=True)
    load, assignment = np.zeros(5, dtype=int), np.zeros(len(voice), dtype=int)
    weights = np.bincount(caller)
    for index in np.argsort(-weights, kind="stable"):
        fold = np.argmin(load)
        assignment[index] = fold
        load[fold] += weights[index]
    folds = assignment[caller]
    scores, heads = {}, {}
    for name in ("pretrained", *(f"random-{seed}" for seed in SEEDS), "acoustic"):
        if name == "acoustic":
            model = AcousticFeatures().to("cuda").eval()
            predict = lambda audio: pooled(model(audio), HOP, FRAME)[:, None, :]
        else:
            if name == "pretrained":
                model = pretrained
            else:
                torch.manual_seed(int(name.rsplit("-", 1)[1]))
                model = torchaudio.models.wav2vec2_model(**bundle._params)
            model = model.to("cuda").eval()
            predict = lambda audio: speech_features(model, audio)
        features = extract(name, predict, clips)
        scores[name], heads[name] = fit_arm(features, truth, splits, folds)
        heads[name]["pooled_features_sha256"] = hashlib.sha256(features.tobytes()).hexdigest()
        del features, model
    scores["random-mean"] = np.mean([scores[f"random-{seed}"] for seed in SEEDS], axis=0)
    report = {"model_file": BUNDLE, "model_sha256": file_digest(checkpoint), "decision_ms": 300,
              "pcm_sha256": hashlib.sha256(clips.tobytes()).hexdigest(), "harper_train_sha256": MEMBERSHIP,
              "heads": heads, "cohorts": {}}
    for name, idx in splits.items():
        if name == "calibration":
            continue
        results = {}
        for arm, score in scores.items():
            metrics = rates(score[idx], truth[idx])
            if name != "train":
                cal = splits["calibration"]
                threshold = _threshold(score[cal], truth[cal])
                metrics.update({"ami_threshold": threshold,
                                "ami_fixed": _stats(score[idx].tolist(), truth[idx].tolist(), threshold)})
            results[arm] = metrics
        report["cohorts"][name] = results
    english = splits["english"]
    report["english_paired"] = paired_intervals({arm: values[english] for arm, values in scores.items()},
                                                 truth[english], groups["english"], {"pretrained": "random-mean"})
    _write_json(root / "reports/wav2vec-harper-probe.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root()), indent=2))
