"""Compare two general speech encoders on the same caller-only labels."""

import argparse
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
from .multilingual_teacher_probe import DATA as MULTILINGUAL_SHA256, score_languages
from .own_head_distillation import encoder, predictions
from .progress import Progress
from .qwen_speech_encoder import qwen_audio
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
WHISPER_SHA256 = "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"
WHISPER_URL = f"https://openaipublic.azureedge.net/main/whisper/models/{WHISPER_SHA256}/large-v3.pt"
WHISPER_SIZE = 3_087_371_615
ORIGINAL_HUBERT_REPORT_SHA256 = "d58f3e43b40ab2ef9dabad5b80e73850e2d374b484b7bd0372fe404ef0f9cefe"


def checkpoint(root, name, url, size, *, etag=None, sha256=None):
    folder = root / "cache/open-speech-encoder"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / url.rsplit("/", 1)[-1]
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        temporary = path.with_suffix(".part")
        meter = Progress(f"{name}-checkpoint", size, "bytes")
        try:
            headers = {"If-Match": etag} if etag else {}
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if ((etag and response.headers["ETag"] != etag)
                        or int(response.headers["Content-Length"]) != size):
                    raise ValueError("checkpoint publisher revision changed")
                with temporary.open("wb") as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        meter.advance(len(chunk))
            digest = file_digest(temporary)
            if temporary.stat().st_size != size or (sha256 and digest != sha256):
                raise ValueError("checkpoint transfer is incomplete or has the wrong hash")
            temporary.replace(path)
            _write_json(manifest, {"url": url, "etag": etag, "bytes": size, "sha256": digest})
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
    provenance = json.loads(manifest.read_text())
    digest = file_digest(path)
    if (provenance != {"url": url, "etag": etag, "bytes": size, "sha256": digest}
            or path.stat().st_size != size or (sha256 and digest != sha256)):
        raise ValueError("cache differs from the pinned publisher revision")
    return path, provenance


def hubert(root):
    bundle = torchaudio.pipelines.HUBERT_LARGE
    if torchaudio.__version__.split("+")[0] != "2.8.0" or bundle._path != FILE or not bundle._normalize_waveform:
        raise ValueError("unexpected HuBERT conversion")
    path, provenance = checkpoint(root, "hubert-large", URL, SIZE, etag=ETAG)
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
    def predict(audio, *, return_mid=False):
        return speech_features(model, audio, normalization_eps=HUBERT_EPS, return_mid=return_mid)

    return predict, provenance


def whisper_large(root):
    import whisper
    from torch.nn.functional import gelu

    if whisper.__version__ != "20250625" or whisper._MODELS["large-v3"] != WHISPER_URL:
        raise ValueError("unexpected Whisper checkpoint configuration")
    path, provenance = checkpoint(root, "whisper-large-v3", WHISPER_URL, WHISPER_SIZE, sha256=WHISPER_SHA256)
    whisper_model = whisper.load_model(str(path), device="cpu")
    core = whisper_model.encoder.eval().to("cuda")
    n_mels = whisper_model.dims.n_mels
    del whisper_model
    if (n_mels, len(core.blocks), *core.positional_embedding.shape) != (128, 32, 1500, 1280):
        raise ValueError("unexpected Whisper Large v3 encoder shape")

    def mel(audio):
        # Whisper applies a clip-relative log-energy floor.
        return torch.stack([whisper.log_mel_spectrogram(samples, n_mels=n_mels,
                                                       padding=whisper.audio.N_SAMPLES - SAMPLES)
                            for samples in audio])

    def states(features):
        hidden = gelu(core.conv2(gelu(core.conv1(features)))).transpose(1, 2)
        hidden = (hidden + core.positional_embedding).to(hidden.dtype)
        yield hidden
        for block in core.blocks[:-1]:
            hidden = block(hidden)
            yield hidden
        yield core.ln_post(core.blocks[-1](hidden))

    tone = torch.sin(torch.arange(SAMPLES, device="cuda", dtype=torch.float32) / 13)[None, :]
    with torch.inference_mode():
        features = mel(tone)
        for traced in states(features):
            pass
        if not torch.equal(traced, core(features)):
            raise ValueError("Whisper layer extraction must reproduce the official encoder")

    def predict(audio, *, return_mid=False):
        # Whisper mel frames are centered every 160 samples, then subsampled by two.
        pooled_batches, middle = [], []
        for batch in audio.split(8):
            summaries = []
            for index, layer in enumerate(states(mel(batch))):
                observed = layer[:, :SAMPLES // 320]
                summaries.append(pooled(observed, 320, 1))
                if return_mid and index == 16:
                    middle.append(observed.contiguous())
            pooled_batches.append(torch.stack(summaries, dim=1))
        summary = torch.cat(pooled_batches)
        return (summary, torch.cat(middle)) if return_mid else summary

    with torch.inference_mode():
        quiet = tone * 1e-3
        separate = torch.cat((predict(tone), predict(quiet)))
        joint = predict(torch.cat((tone, quiet)))
        if not torch.allclose(separate, joint, rtol=1e-3, atol=5e-4):
            raise ValueError("Whisper features changed when clips were batched together")
    return predict, {**provenance, "whisper_version": whisper.__version__, "padded_input_samples": whisper.audio.N_SAMPLES}


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


ENCODERS = {"hubert-large": hubert, "whisper-large-v3": whisper_large, "qwen3-asr-1.7b": qwen_audio}


def run(root, candidate_name="hubert-large", *, replay_hubert=False):
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
    for name in ("wav2vec2-base", candidate_name):
        if name == "wav2vec2-base":
            base = encoder(root, {"encoder_sha256": CHECKPOINT_SHA256})
            predict = lambda audio: speech_features(base, audio)
        else:
            predict, provenance[name] = ENCODERS[name](root)
        torch.cuda.reset_peak_memory_stats()
        features = extract(name, predict, clips)
        heads_peak = torch.cuda.max_memory_allocated()
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
        heads[name]["extraction_peak_allocated_bytes"] = heads_peak
        teachers[name] = lambda audio, elapsed, predict=predict, fitted=fitted: torch.as_tensor(
            predictions(predict(audio).cpu().numpy(), fitted), device=audio.device,
        )
        if name == candidate_name:
            candidate_heads = fitted
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
    candidate, baseline = metrics[candidate_name], metrics["wav2vec2-base"]
    pairs = paired_intervals({key: value[en] for key, value in scores.items()}, truth[en], groups["english"],
                             {candidate_name: "wav2vec2-base"}, partial_auc=True)
    en_new, en_old = candidate["english"], baseline["english"]
    gates = {
        "recall_gain_five_points": en_new["recalls_at_2_5_5_7_5pct"][1] + 1e-12 >= en_old["recalls_at_2_5_5_7_5pct"][1] + .05,
        "paired_recall_gain": pairs[candidate_name]["recall_delta_95"][0] > 0,
        "english_partial_auc_gain": en_new["partial_auc"] > en_old["partial_auc"],
        "harper_partial_auc_within_tolerance": candidate["harper"]["partial_auc"] >= baseline["harper"]["partial_auc"] - .03,
    }
    report = {"decision_ms": 300, "pcm_sha256": pcm_hash, "reference_sha256": file_digest(source),
              "encoders": provenance, "heads": heads, "metrics": metrics, "paired_english": pairs,
              "advancement": gates, "advanced": all(gates.values())}
    archived = None
    if replay_hubert:
        original_path = root / "reports/open-speech-teacher.json"
        if file_digest(original_path) != ORIGINAL_HUBERT_REPORT_SHA256:
            raise ValueError("the original completed HuBERT report differs from its archived digest")
        archived = json.loads(original_path.read_text())
        if candidate_name != "hubert-large" or any(
            report[key] != archived[key] for key in (
                "decision_ms", "reference_sha256", "pcm_sha256", "encoders", "metrics", "paired_english",
            )
        ):
            raise ValueError("HuBERT replay changed the original matched decision metrics or inputs")
        for name, values in heads.items():
            expected = archived["heads"][name]
            if any(json.loads(json.dumps(values[key])) != expected[key]
                   for key in ("penalty", "layer_weights", "selection_partial_auc", "pooled_features_sha256")):
                raise ValueError(f"HuBERT replay changed original features or head selection for {name}")
        report["archived_hubert_report_sha256"] = ORIGINAL_HUBERT_REPORT_SHA256

    folder = root / "runs/open-speech-teacher"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{candidate_name}-head.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({"heads": candidate_heads, "seeds": HEAD_SEEDS, "encoder": candidate_name,
                "encoder_sha256": provenance[candidate_name]["sha256"],
                "decision_ms": 300, "samples": SAMPLES, "ensemble": "mean-probability"}, temporary)
    payload = torch.load(temporary, map_location="cpu", weights_only=True)
    p = np.mean([expit(predict_logits(outputs[candidate_name][evaluation], head)) for head in payload["heads"]], axis=0)
    if not np.array_equal(p, scores[candidate_name][evaluation]):
        raise ValueError("serialized candidate predictions did not reproduce")
    temporary.replace(path)
    report["diagnostic_checkpoint"] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path),
                                       "evaluation_probability_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
    filename = f"open-speech-{candidate_name}.json"
    if candidate_name == "hubert-large" and archived is None:
        filename = "open-speech-teacher.json"
    path = root / "reports" / filename
    _write_json(path, report)
    paired_rows = {}
    thresholds = {name: value["english"]["ami_threshold"] for name, value in metrics.items()}
    report["multilingual_diagnostic"] = score_languages(
        root, teachers, "cuda", paired_rows=paired_rows, fixed_thresholds=thresholds,
        progress_name=f"open-teacher-languages-{candidate_name}",
    )
    if archived is not None:
        for lang, row in report["multilingual_diagnostic"].items():
            for name, values in row["models"].items():
                expected = archived["multilingual_diagnostic"][lang]["models"][name]
                if any(values[key] != expected[key] for key in ("auc", "partial_auc_to_5pct", "retrospective_roc_limits")):
                    raise ValueError(f"HuBERT replay changed the language ranking metrics for {lang}")
    paired_path = path.with_name(f"{path.stem}-language-pairs.json")
    _write_json(paired_path, {"dataset_sha256": MULTILINGUAL_SHA256, "languages": paired_rows})
    report["multilingual_paired_rows"] = {"path": paired_path.relative_to(root).as_posix(),
                                         "sha256": file_digest(paired_path)}
    _write_json(path, report)
    selected = ("wav2vec2-base", candidate_name) if report["advanced"] else ("wav2vec2-base",)
    report["depth"] = (archived["depth"] if archived is not None else {
        key: depth_curve(key, outputs[key], truth, splits, groups, heads[key]["penalty"]) for key in selected
    })
    _write_json(path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=ENCODERS, default="hubert-large")
    parser.add_argument("--replay-hubert", action="store_true", help="verify the original HuBERT report and retain paired scores")
    torch.set_num_threads(2)
    args = parser.parse_args()
    print(json.dumps(run(data_root(), args.candidate, replay_hubert=args.replay_hubert), indent=2))
