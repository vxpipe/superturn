"""Test a newly fitted turn head over non-turn Wav2Vec2 Base representations."""

from collections import defaultdict
from dataclasses import fields
from functools import lru_cache
import argparse
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
from .augment import condition
from .dense_teacher import SAMPLES, file_digest
from .elapsed_turn_probe import paired_intervals
from .eot_domain_probe import GAP, decision_audio
from .harper_history_probe import recall_area
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_benchmark import pinned_split
from .labels import Label
from .model import AcousticFeatures, FRAME, HOP
from .progress import Progress
from .shards import CONTEXT, clip
from .sources import data_root
from .train import _auc, _stats, _threshold, paths, records
from .turn1_mini_probe import english_rows, harper_rows, upsample_prefix

SEEDS = (17, 29, 43)
PENALTIES = (.01, .05, .25)
HEAD_SEEDS = (17, 29, 43, 53, 71)
BUNDLE = "wav2vec2_fairseq_base_ls960.pth"
CHECKPOINT_SHA256 = "1275f890a623b31b9915295dc01d5049c93dc6de8813406c16dfab9b6bd36056"
AMI = "bed0ba3de027429f"


def verified_ami_labels(root, split):
    cohort = json.loads((root / "reports/ami-calibration" / f"{AMI}.json").read_text())["cohorts"][split]
    digest = hashlib.sha256()
    names = [field.name for field in fields(Label)]
    labels = []
    for batch in records(paths(root / "shards/ami-calibration" / AMI, split)):
        for row in batch:
            metadata = {name: row[name] for name in names}
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            digest.update(row["pcm"])
            labels.append(Label(**metadata))
    if digest.hexdigest() != cohort["sha256"]:
        raise ValueError(f"AMI {split} rows changed")
    return labels


def render_ami(root, labels):
    cached = lru_cache(maxsize=6)(read_audio)
    end = CONTEXT + GAP
    for label in labels:
        if label.pause_samples < GAP:
            raise ValueError("AMI pause does not reach 300 ms")
        original = clip(label, cached(root / label.audio))
        original[end:] = 0
        pcm = condition(original, "telephone", 0)
        yield pcm[end - SAMPLES:end].astype(np.float32) / 32768, int(label.label == "shift"), label.group


def calibration_rows(root):
    return render_ami(root, [label for label in verified_ami_labels(root, "calibration") if label.label == "hold"])


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


def pooled(frames, stride, width, *, temporal=False):
    center = torch.arange(frames.shape[1], device=frames.device) * stride + (width - 1) / 2
    edge = SAMPLES - GAP
    previous = (center >= edge - RATE // 2) & (center < edge)
    silence = center >= edge
    parts = [frames.mean(1), frames[:, previous].mean(1), frames[:, silence].mean(1)]
    if temporal:
        tail = center >= edge - RATE // 4
        parts.append(frames[:, previous & tail].mean(1) - frames[:, previous & ~tail].mean(1))
    return torch.cat(parts, dim=-1)


def speech_features(model, audio, *, temporal=False, normalization_eps=1e-7):
    audio = F.layer_norm(audio, (SAMPLES,), eps=normalization_eps)
    features, lengths = model.feature_extractor(audio, None)
    projection = model.encoder.feature_projection(features)
    layers = model.encoder.extract_features(features, lengths)
    return torch.stack([pooled(layer, 320, 400, temporal=temporal) for layer in (projection, *layers)], dim=1)


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


def fit_head(features, truth, training, penalty, seed, *, normalizer=None, sample_weights=None):
    train = torch.as_tensor(features[training]).double()
    target = torch.as_tensor(truth[training]).double()
    origin = train if normalizer is None else torch.as_tensor(features[normalizer]).double()
    center, scale = origin.mean(0), origin.std(0, unbiased=False).clamp(min=1e-5)
    loss_weights = None if sample_weights is None else torch.as_tensor(sample_weights).double()
    if loss_weights is not None and (loss_weights.shape != target.shape or
            not torch.isfinite(loss_weights).all() or (loss_weights < 0).any() or not torch.isclose(
                loss_weights.sum(), torch.tensor(float(len(train)), dtype=torch.float64))):
        raise ValueError("loss weights must be nonnegative, finite and average one across training rows")
    train = (train - center) / scale
    rng = torch.Generator().manual_seed(seed)
    layer = (torch.randn(train.shape[1], generator=rng, dtype=torch.float64) * .01).requires_grad_()
    weights = torch.zeros(train.shape[2], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS((layer, weights, bias), max_iter=80, line_search_fn="strong_wolfe")

    def update():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head_logits(train, layer, weights, bias), target, weight=loss_weights)
        loss = loss + penalty * weights.square().sum() / 2 + .001 * layer.square().mean()
        loss.backward()
        return loss

    optimizer.step(update)
    return center, scale, layer.detach(), weights.detach(), bias.detach()


def head_logits(values, layer, weights, bias):
    return (values.transpose(1, 2) @ layer.softmax(0)) @ weights + bias


def fit(features, truth, training, evaluation, penalty, seed, *, normalizer=None, sample_weights=None):
    head = fit_head(
        features, truth, training, penalty, seed, normalizer=normalizer, sample_weights=sample_weights)
    return predict_logits(features[evaluation], head), head[2].softmax(0).tolist()


def predict_logits(features, head):
    center, scale, *params = head
    with torch.no_grad():
        other = torch.as_tensor(features).double()
        return head_logits((other - center) / scale, *params).numpy()


def rank_at_95pct(features):
    variance = np.linalg.eigvalsh(features @ features.T).clip(min=0)[::-1]
    return int(np.searchsorted(np.cumsum(variance), .95 * variance.sum()) + 1) if variance.sum() else 0


def fit_arm(features, truth, splits, caller_folds, meter, *, contrast_width=0, head_states=None):
    train = splits["train"]
    search = {}
    for penalty in PENALTIES:
        score = np.full(len(train), np.nan)
        for fold in range(5):
            held = caller_folds == fold
            score[held], _ = fit(features, truth, train[~held], train[held], penalty, 17)
            meter.advance(1)
        if not np.isfinite(score).all():
            raise ValueError("incomplete held-caller head selection")
        search[penalty] = (recall_area(score, truth[train])[-1], score)
    penalty = max(PENALTIES, key=lambda value: search[value][0])
    evaluation = np.concatenate([indices for name, indices in splits.items() if name != "train"])
    repeats = []
    for seed in HEAD_SEEDS:
        head = fit_head(features, truth, train, penalty, seed)
        repeats.append((predict_logits(features[evaluation], head), head[2].softmax(0).tolist()))
        if head_states is not None:
            head_states.append(head)
        meter.advance(1)
    weight = np.mean([weights for _, weights in repeats], axis=0)
    score = np.full(len(truth), np.nan)
    score[train] = expit(search[penalty][1])
    score[evaluation] = np.mean([expit(values) for values, _ in repeats], axis=0)
    probe = features[train] - features[train].mean(0)
    projection = probe.transpose(0, 2, 1) @ weight.astype(np.float32)
    diagnostics = {"penalty": penalty, "selection_partial_auc": {p: value[0] for p, value in search.items()},
                   "layer_weights": weight.tolist(), "raw_pooled_rank_at_95pct": rank_at_95pct(projection)}
    if contrast_width:
        diagnostics["raw_contrast_rank_at_95pct"] = rank_at_95pct(projection[:, -contrast_width:])
    return score, diagnostics


def rates(scores, truth):
    values = recall_area(scores, truth).tolist()
    return {"auc": _auc(scores.tolist(), truth.tolist()), "recalls_at_2_5_5_7_5pct": values[:3],
            "mean_recall": float(np.mean(values[:3])), "partial_auc": values[-1]}


def temporal_report(root, features, truth, splits, folds, english_groups, report):
    reference_path = root / "reports/wav2vec-harper-probe.json"
    reference = json.loads(reference_path.read_text())
    anchor = reference["heads"]["pretrained"]
    width = features.shape[-1] // 4
    original = np.ascontiguousarray(features[..., :3 * width])
    feature_hash = hashlib.sha256(original.tobytes()).hexdigest()
    if (reference["pcm_sha256"] != report["pcm_sha256"] or
            reference["model_sha256"] != report["model_sha256"] or
            anchor["pooled_features_sha256"] != feature_hash):
        raise ValueError("archived frozen inputs and features did not reproduce")
    sign_seed = 6903
    signs = np.random.default_rng(sign_seed).choice(np.array([-1, 1], dtype=np.float32), len(truth))
    control = features.copy()
    control[..., -width:] *= signs[:, None, None]
    scores, metrics = {}, {}
    temporal_heads = []
    for arm, values in (("frozen", original), ("temporal", features), ("random-sign", control)):
        meter = Progress(f"wav2vec-temporal-{arm}", 5 * len(PENALTIES) + 5, "fits")
        try:
            scores[arm], report["heads"][arm] = fit_arm(
                values, truth, splits, folds, meter, contrast_width=0 if arm == "frozen" else width,
                head_states=temporal_heads if arm == "temporal" else None,
            )
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
        for name, idx in splits.items():
            if name == "calibration":
                continue
            cal = splits["calibration"]
            result = rates(scores[arm][idx], truth[idx])
            if name != "train":
                threshold = _threshold(scores[arm][cal], truth[cal])
                result.update(ami_threshold=threshold,
                              ami_fixed=_stats(scores[arm][idx].tolist(), truth[idx].tolist(), threshold))
            metrics.setdefault(name, {})[arm] = result
            if arm == "frozen":
                for key, value in result.items():
                    if key != "ami_fixed" and not np.allclose(
                        value, reference["cohorts"][name]["pretrained"][key], rtol=0, atol=1e-10,
                    ):
                        raise ValueError(f"the archived frozen head {name} {key} did not reproduce")
    en = splits["english"]
    evaluation = np.concatenate([idx for name, idx in splits.items() if name != "train"])
    replay = json.loads((root / "reports/harper-ami-transfer.json").read_text())
    score_hash = hashlib.sha256(scores["frozen"][evaluation].tobytes()).hexdigest()
    if score_hash != replay["own_head_probability_sha256"]["caller"]:
        raise ValueError("the archived frozen probabilities did not reproduce")
    groups = {arm: values[en] for arm, values in scores.items()}
    paired = paired_intervals(groups, truth[en], english_groups, {"temporal": "frozen"}, partial_auc=True)
    matched = paired_intervals(groups, truth[en], english_groups, {"temporal": "random-sign"}, partial_auc=True)
    english = metrics["english"]
    recall = {arm: result["recalls_at_2_5_5_7_5pct"][1] for arm, result in english.items()}
    harper = metrics["harper"]
    gates = {
        "english_recall_gain_four_points": recall["temporal"] + 1e-12 >= recall["frozen"] + .04,
        "paired_english_recall_gain": paired["temporal"]["recall_delta_95"][0] > 0,
        "english_recall_and_partial_auc_above_random_sign": (
            recall["temporal"] > recall["random-sign"] and
            english["temporal"]["partial_auc"] > english["random-sign"]["partial_auc"]
        ),
        "english_partial_auc_within_tolerance": english["temporal"]["partial_auc"] >= english["frozen"]["partial_auc"] - .005,
        "harper_partial_auc_within_tolerance": harper["temporal"]["partial_auc"] >= harper["frozen"]["partial_auc"] - .03,
    }
    report.update(cohorts=metrics, reference_sha256=file_digest(reference_path),
                  original_features_sha256=feature_hash, frozen_probability_sha256=score_hash,
                  contrast_windows_before_edge_ms=[[250, 0], [500, 250]], control_sign_seed=sign_seed,
                  paired_english=paired, paired_english_random_sign=matched,
                  advancement=gates, promoted=all(gates.values()))
    if report["promoted"]:
        folder = root / "runs/wav2vec-harper-temporal"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "heads.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"heads": temporal_heads, "seeds": HEAD_SEEDS, "encoder_sha256": CHECKPOINT_SHA256,
                    "decision_ms": 300, "contrast_windows_before_edge_ms": report["contrast_windows_before_edge_ms"],
                    "ensemble": "mean-probability"}, temporary)
        payload = torch.load(temporary, map_location="cpu", weights_only=True)
        probabilities = np.mean([expit(predict_logits(features[evaluation], head))
                                 for head in payload["heads"]], axis=0)
        if not np.array_equal(probabilities, scores["temporal"][evaluation]):
            raise ValueError("serialized temporal head probabilities did not reproduce")
        temporary.replace(path)
        report["checkpoint"] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path),
                                "evaluation_probability_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest()}
    _write_json(root / "reports/wav2vec-harper-temporal.json", report)
    return report


def pinned_inputs(root):
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
    return clips, truth, splits, groups


def caller_folds(groups):
    voice, caller = np.unique(groups, return_inverse=True)
    load, assignment = np.zeros(5, dtype=int), np.zeros(len(voice), dtype=int)
    weights = np.bincount(caller)
    for index in np.argsort(-weights, kind="stable"):
        fold = np.argmin(load)
        assignment[index] = fold
        load[fold] += weights[index]
    return assignment[caller]


def run(root, *, temporal=False):
    clips, truth, splits, groups = pinned_inputs(root)
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    if torchaudio.__version__.split("+")[0] != "2.8.0" or bundle._path != BUNDLE or bundle.sample_rate != RATE:
        raise ValueError("unexpected encoder bundle")
    folder = root / "cache/wav2vec2-harper"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / BUNDLE
    if not checkpoint.is_file():
        if temporal:
            raise ValueError("the temporal diagnostic requires the staged encoder checkpoint")
        url = "https://download.pytorch.org/torchaudio/models/" + BUNDLE
        torch.hub.download_url_to_file(url, checkpoint, hash_prefix=CHECKPOINT_SHA256)
    if checkpoint.stat().st_size != 377565405 or file_digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("unexpected publisher checkpoint")
    pretrained = torchaudio.models.wav2vec2_model(**bundle._params)
    pretrained.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    folds = caller_folds(groups["train"])
    report = {"model_file": BUNDLE, "model_sha256": file_digest(checkpoint), "decision_ms": 300,
              "pcm_sha256": hashlib.sha256(clips.tobytes()).hexdigest(), "harper_train_sha256": MEMBERSHIP,
              "heads": {}, "cohorts": {}}
    if temporal:
        pretrained.requires_grad_(False).eval().to("cuda")
        features = extract("harper-temporal", lambda audio: speech_features(pretrained, audio, temporal=True), clips)
        return temporal_report(root, features, truth, splits, folds, groups["english"], report)
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
        meter = Progress(f"wav2vec-head-{name}", 5 * len(PENALTIES) + 5, "fits")
        try:
            scores[name], heads[name] = fit_arm(features, truth, splits, folds, meter)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()
        heads[name]["pooled_features_sha256"] = hashlib.sha256(features.tobytes()).hexdigest()
        del features, model
    scores["random-mean"] = np.mean([scores[f"random-{seed}"] for seed in SEEDS], axis=0)
    report["heads"] = heads
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", action="store_true", help="test pre-edge temporal contrast")
    torch.set_num_threads(2)
    print(json.dumps(run(data_root(), temporal=parser.parse_args().temporal), indent=2))
