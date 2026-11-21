"""Compare two general speech encoders on the same caller-only labels."""

from collections import defaultdict
import hashlib
import json
from urllib.request import Request, urlopen

import numpy as np
from scipy.special import expit
import torch
import torchaudio
from torchaudio.pipelines._wav2vec2.utils import _extend_model

from .ami_calibration import _write_json
from .dense_teacher import SAMPLES, file_digest
from .elapsed_turn_probe import paired_intervals
from .harper_history_probe import recall_area
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import OwnSpeechHead, encoder
from .progress import Progress
from .sources import data_root
from .train import _stats, _threshold
from .wav2vec_harper_probe import (
    CHECKPOINT_SHA256, HEAD_SEEDS, PENALTIES, caller_folds, extract, fit, fit_arm,
    pinned_inputs, pooled, predict_logits, rates, speech_features,
)

FILE = "hubert_fairseq_large_ll60k.pth"
URL = "https://download.pytorch.org/torchaudio/models/" + FILE
SIZE = 1_261_897_861
ETAG = '"a0f92396cd9cc108d0cbfbb5e65b77ff-76"'
DEPTH_SEEDS = (901, 902, 903, 904, 905)
HUBERT_EPS = 1e-5


def hubert(root):
    bundle = torchaudio.pipelines.HUBERT_LARGE
    if torchaudio.__version__.split("+")[0] != "2.8.0" or bundle._path != FILE or not bundle._normalize_waveform:
        raise ValueError("unexpected HuBERT conversion")
    folder = root / "cache/open-speech-encoder"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FILE
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        temporary = path.with_suffix(".part")
        meter = Progress("hubert-checkpoint", SIZE, "bytes")
        try:
            with urlopen(Request(URL, headers={"If-Match": ETAG}), timeout=60) as response:
                if response.headers["ETag"] != ETAG or int(response.headers["Content-Length"]) != SIZE:
                    raise ValueError("HuBERT checkpoint publisher revision changed")
                with temporary.open("wb") as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        meter.advance(len(chunk))
            if temporary.stat().st_size != SIZE:
                raise ValueError("HuBERT checkpoint transfer is incomplete")
            temporary.replace(path)
            _write_json(manifest, {"url": URL, "etag": ETAG, "bytes": SIZE, "sha256": file_digest(path)})
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
    provenance = json.loads(manifest.read_text())
    if provenance != {"url": URL, "etag": ETAG, "bytes": SIZE, "sha256": file_digest(path)} or path.stat().st_size != SIZE:
        raise ValueError("HuBERT cache differs from the pinned publisher revision")
    model = torchaudio.models.wav2vec2_model(**bundle._params)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.eval().to("cuda")
    tone = torch.sin(torch.arange(SAMPLES, device="cuda", dtype=torch.float32) / 13)[None, :]
    with torch.inference_mode():
        official = _extend_model(model, normalize_waveform=True).eval()
        for audio in (tone, tone * 1e-3):
            layers, _ = official.extract_features(audio)
            public = torch.stack([pooled(layer, 320, 400) for layer in layers], dim=1)
            if not torch.equal(speech_features(model, audio, normalization_eps=HUBERT_EPS)[:, 1:], public):
                raise ValueError("HuBERT pooled features diverge from the published single-clip path")
    return model, provenance


def depth_curve(name, features, truth, splits, groups, penalty):
    train = splits["train"]
    english, harper, calibration = (splits[key] for key in ("english", "harper", "calibration"))
    evaluation = np.r_[english, harper, calibration]
    strata = defaultdict(list)
    for row, caller in zip(train, groups["train"], strict=True):
        strata[caller, truth[row]].append(row)
    callers = np.asarray(groups["train"])
    labels = truth[english]

    def estimate(selected):
        _, identity = np.unique(callers[selected], return_inverse=True)
        counts = np.bincount(identity)
        weights = len(selected) / (len(counts) * counts[identity])
        scores = []
        for seed in HEAD_SEEDS:
            logits, _ = fit(features, truth, selected, evaluation, penalty, seed, sample_weights=weights)
            scores.append(expit(logits))
            meter.advance(1)
        p = np.mean(scores, axis=0)
        en_scores = p[:len(english)]
        hv_scores = p[len(english):-len(calibration)]
        threshold = _threshold(p[-len(calibration):], truth[calibration])
        return {"rows": len(selected), "callers": len(counts),
                "english": {**rates(en_scores, labels),
                            "ami_fixed": _stats(en_scores.tolist(), labels.tolist(), threshold)},
                "harper": {**rates(hv_scores, truth[harper]),
                           "ami_fixed": _stats(hv_scores.tolist(), truth[harper].tolist(), threshold)}}, en_scores

    meter = Progress(f"{name}-caller-depth", (2 * len(DEPTH_SEEDS) + 1) * len(HEAD_SEEDS), "fits")
    try:
        full, full_scores = estimate(train)
        draws, half_scores = [], []
        for seed in DEPTH_SEEDS:
            rng = np.random.default_rng(seed)
            order = [rng.permutation(indices) for indices in strata.values()]
            draw = {"seed": seed}
            for percent in (25, 50):
                selected = np.sort(np.concatenate([rows[:max(1, round(len(rows) * percent / 100))]
                                                   for rows in order]))
                draw[str(percent)], by_turn = estimate(selected)
            draws.append(draw)
            half_scores.append(by_turn)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    gain = [full["english"]["mean_recall"] - draw["50"]["english"]["mean_recall"] for draw in draws]
    turns = defaultdict(list)
    for index, key in enumerate(groups["english"]):
        turns[key].append(index)
    clusters = tuple(turns.values())
    rng = np.random.default_rng(6902)
    deltas = []
    for draw in rng.integers(len(clusters), size=(500, len(clusters))):
        idx = np.concatenate([clusters[i] for i in draw])
        if len(np.unique(labels[idx])) == 2:
            full_recall = recall_area(full_scores[idx], labels[idx])[:3].mean()
            half_recall = np.mean([recall_area(p[idx], labels[idx])[:3].mean() for p in half_scores])
            deltas.append(full_recall - half_recall)
    return {"full": full, "draws": draws, "penalty": penalty, "mean_half_to_full_recall_gain": float(np.mean(gain)),
            "mean_half_to_full_partial_auc_gain": full["english"]["partial_auc"] - float(np.mean(
                [draw["50"]["english"]["partial_auc"] for draw in draws]
            )),
            "gain_turn_bootstrap_95": np.quantile(deltas, [.025, .975]).tolist(),
            "positive_draws": sum(delta > 0 for delta in gain)}


def run(root):
    clips, truth, splits, groups = pinned_inputs(root)
    folds = caller_folds(groups["train"])
    source = root / "reports/wav2vec-harper-probe.json"
    reference = json.loads(source.read_text())
    pcm_hash = hashlib.sha256(clips.tobytes()).hexdigest()
    if pcm_hash != reference["pcm_sha256"]:
        raise ValueError("archived causal inputs did not reproduce")
    en, hv, cal = (splits[name] for name in ("english", "harper", "calibration"))
    evaluation = np.r_[en, hv, cal]
    provenance = {"wav2vec2-base": {"sha256": CHECKPOINT_SHA256}}
    scores, metrics, heads = {}, {}, {}
    outputs, teachers = {}, {}
    for name in ("wav2vec2-base", "hubert-large"):
        if name == "wav2vec2-base":
            model = encoder(root, {"encoder_sha256": CHECKPOINT_SHA256})
        else:
            model, provenance[name] = hubert(root)
        eps = HUBERT_EPS if name == "hubert-large" else 1e-7
        features = extract(name, lambda audio: speech_features(model, audio, normalization_eps=eps), clips)
        feature_hash = hashlib.sha256(features.tobytes()).hexdigest()
        if name == "wav2vec2-base" and feature_hash != reference["heads"]["pretrained"]["pooled_features_sha256"]:
            raise ValueError("archived frozen features did not reproduce")
        meter = Progress(f"open-teacher-{name}", 5 * len(PENALTIES) + 5, "fits")
        fitted = []
        try:
            scores[name], heads[name] = fit_arm(
                features, truth, splits, folds, meter, head_states=fitted,
            )
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
        heads[name]["pooled_features_sha256"] = feature_hash
        heads[name]["normalization_eps"] = eps
        teachers[name] = OwnSpeechHead(model, fitted, normalization_eps=eps)
        score = scores[name]
        threshold = _threshold(score[cal], truth[cal])
        metrics[name] = {key: {**rates(score[idx], truth[idx]), "ami_threshold": threshold,
                               "ami_fixed": _stats(score[idx].tolist(), truth[idx].tolist(), threshold)}
                         for key, idx in (("english", en), ("harper", hv))}
        outputs[name] = features
        if name == "wav2vec2-base":
            replay = json.loads((root / "reports/harper-ami-transfer.json").read_text())
            if hashlib.sha256(score[evaluation].tobytes()).hexdigest() != replay["own_head_probability_sha256"]["caller"]:
                raise ValueError("archived frozen predictions did not reproduce")
    candidate, baseline = metrics["hubert-large"], metrics["wav2vec2-base"]
    pairs = paired_intervals({key: value[en] for key, value in scores.items()}, truth[en], groups["english"],
                             {"hubert-large": "wav2vec2-base"}, partial_auc=True)
    en_new, en_old = candidate["english"], baseline["english"]
    gates = {
        "recall_gain_five_points": en_new["recalls_at_2_5_5_7_5pct"][1] + 1e-12 >= en_old["recalls_at_2_5_5_7_5pct"][1] + .05,
        "paired_recall_gain": pairs["hubert-large"]["recall_delta_95"][0] > 0,
        "english_partial_auc_gain": en_new["partial_auc"] > en_old["partial_auc"],
        "harper_partial_auc_within_tolerance": candidate["harper"]["partial_auc"] >= baseline["harper"]["partial_auc"] - .03,
    }
    report = {"decision_ms": 300, "pcm_sha256": pcm_hash, "reference_sha256": file_digest(source),
              "encoders": provenance, "heads": heads, "metrics": metrics, "paired_english": pairs,
              "advancement": gates, "advanced": all(gates.values())}
    if report["advanced"]:
        folder = root / "runs/open-speech-teacher"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "hubert-head.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"heads": teachers["hubert-large"].heads, "seeds": HEAD_SEEDS,
                    "encoder_sha256": provenance["hubert-large"]["sha256"], "normalization_eps": HUBERT_EPS,
                    "decision_ms": 300, "samples": SAMPLES, "ensemble": "mean-probability"}, temporary)
        payload = torch.load(temporary, map_location="cpu", weights_only=True)
        p = np.mean([expit(predict_logits(outputs["hubert-large"][evaluation], head)) for head in payload["heads"]], axis=0)
        if not np.array_equal(p, scores["hubert-large"][evaluation]):
            raise ValueError("serialized HuBERT predictions did not reproduce")
        temporary.replace(path)
        report["checkpoint"] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path),
                                "evaluation_probability_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
    path = root / "reports/open-speech-teacher.json"
    _write_json(path, report)
    report["multilingual_diagnostic"] = score_languages(root, teachers, "cuda")
    _write_json(path, report)
    selected = ("wav2vec2-base", "hubert-large") if report["advanced"] else ("wav2vec2-base",)
    report["depth"] = {key: depth_curve(key, outputs[key], truth, splits, groups, heads[key]["penalty"])
                       for key in selected}
    _write_json(path, report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root()), indent=2))
