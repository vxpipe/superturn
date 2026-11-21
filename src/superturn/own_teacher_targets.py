"""Cache scores and aligned hidden features from the locked general teacher."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
import torch

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import SAMPLES, file_digest, normalize
from .model import FRAME, HOP
from .open_speech_teacher_probe import ENCODERS
from .own_head_distillation import DEADLINE, encoder, pool_rows, predictions
from .own_teacher_pool_pcm import recover
from .progress import Progress
from .sources import data_root
from .teacher_language_selection import GENERAL
from .wav2vec_harper_probe import CHECKPOINT_SHA256, extract, pinned_inputs, speech_features

SHARD_ROWS = 256
BATCH = 32
MIN_LANGUAGE_FIT = 200
BASE = "wav2vec2-base"


def teacher(root, selection):
    name = selection["provisional_default"]
    descriptor = selection["inputs"][name]
    path = root / descriptor["head_path"]
    if file_digest(path) != descriptor["head_sha256"]:
        raise ValueError("the nominated own head changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["ensemble"] != "mean-probability" or payload["encoder_sha256"] != descriptor["encoder_sha256"]:
        raise ValueError("the teacher head does not match its encoder receipt")
    if name == BASE:
        receipt_path = root / "reports/harper-ami-transfer.json"
        if file_digest(receipt_path) != descriptor["report_sha256"]:
            raise ValueError("the archived baseline caller receipt changed")
        receipt = json.loads(receipt_path.read_text())
        heads = payload["arm_states"]["caller"]
        expected = receipt["own_head_probability_sha256"]["caller"]
        model = encoder(root, {"encoder_sha256": CHECKPOINT_SHA256})

        def predict(audio, *, return_mid=False):
            return speech_features(model, audio, return_mid=return_mid)

        provenance = {"sha256": CHECKPOINT_SHA256}
    else:
        receipt = next((report for report in (root / "reports").glob("open-speech-*.json")
                        if file_digest(report) == descriptor["report_sha256"]), None)
        if receipt is None:
            raise ValueError("the locked encoder development receipt is missing")
        archived = json.loads(receipt.read_text())
        expected = archived["diagnostic_checkpoint"]["evaluation_probability_sha256"]
        heads = payload["heads"]
        predict, provenance = ENCODERS[name](root)
    if provenance["sha256"] != descriptor["encoder_sha256"]:
        raise ValueError("the staged encoder differs from the selected checkpoint")

    clips, _, splits, _ = pinned_inputs(root)
    feats = extract(f"{name}-teacher-replay", predict, clips)
    evaluation = np.concatenate([part for key, part in splits.items() if key != "train"])
    if hashlib.sha256(predictions(feats[evaluation], heads).tobytes()).hexdigest() != expected:
        raise ValueError("the locked teacher's archived scores did not reproduce")
    with torch.inference_mode():
        batch = torch.as_tensor(clips[evaluation[:8]], device="cuda")
        original = predict(batch)
        joint, intermediate = predict(batch, return_mid=True)
        if not torch.equal(original, joint) or not torch.isfinite(intermediate).all():
            raise ValueError("extracting middle-layer features changed the frozen score path")
    if "frame_centers_samples" in provenance:
        centers = np.asarray(provenance["frame_centers_samples"], dtype=np.float64)
    elif name == "whisper-large-v3":
        centers = np.arange(0, SAMPLES, 320, dtype=np.float64)
    else:
        centers = np.arange((SAMPLES - 400) // 320 + 1) * 320 + 199.5
    return name, descriptor, predict, heads, centers


def align(mid, centers, left_padding):
    """Interpolate native frame centers to the GRU grid and mask synthetic history."""
    student = np.arange((SAMPLES - FRAME) // (HOP * 4) + 1) * HOP * 4 + (FRAME - 1) / 2
    upper = np.searchsorted(centers, student).clip(1, len(centers) - 1)
    lower = upper - 1
    weight = (student - centers[lower]) / (centers[upper] - centers[lower])
    in_range = (student >= centers[0]) & (student <= centers[-1])
    support_min = np.where(weight == 1, centers[upper], centers[lower])
    mask = in_range[None, :] & (support_min[None, :] >= left_padding[:, None])
    w = torch.as_tensor(weight.clip(0, 1), dtype=mid.dtype, device=mid.device)[None, :, None]
    features = torch.lerp(mid[:, lower], mid[:, upper], w).cpu().numpy()
    features[~mask] = 0
    return features.astype(np.float16), mask


def language_targets(scores, roles, languages):
    fit = roles == "fit"
    if not np.all(np.isfinite(scores) & (0 <= scores) & (scores <= 1)):
        raise ValueError("invalid own teacher probabilities")
    logits = logit(scores.astype(np.float64).clip(1e-6, 1 - 1e-6))
    center, scale = float(logits[fit].mean()), float(logits[fit].std())
    if scale < 1e-6:
        raise ValueError("the teacher's fit scores have collapsed")
    normalized = (logits - center) / scale
    used = {}
    for language in sorted(set(languages) - {"und", ""}):
        match = languages == language
        sample = logits[fit & match]
        if len(sample) < MIN_LANGUAGE_FIT or sample.std() < 1e-6:
            continue
        language_center, language_scale = float(sample.mean()), float(sample.std())
        normalized[match] = (logits[match] - language_center) / language_scale
        used[language] = {"fit_rows": len(sample), "mean": language_center, "std": language_scale}
    return expit(normalized).astype(np.float32), {"global": {"fit_rows": int(fit.sum()), "mean": center, "std": scale},
                                                 "by_language": used, "minimum_fit_rows": MIN_LANGUAGE_FIT}


def run(root):
    path = root / "reports/own-teacher-language-selection.json"
    ranking_hash = file_digest(path)
    selection = json.loads(path.read_text())
    if set(selection["macro_partial_auc_to_5pct"]) != GENERAL:
        raise ValueError("the complete general-teacher ranking is required")
    pcm_path, pcm_receipt = recover(root)
    _, pool_hash, rows = pool_rows(root)
    pcm = np.load(pcm_path, mmap_mode="r", allow_pickle=False)
    name, descriptor, predict, heads, centers = teacher(root, selection)
    directory = root / "cache/own-teacher-targets" / f"{name}-{ranking_hash[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    inputs = {"teacher": name, "teacher_head_sha256": descriptor["head_sha256"], "encoder_sha256": descriptor["encoder_sha256"],
              "ranking_sha256": ranking_hash, "pool_sha256": pool_hash, "pcm_file_sha256": pcm_receipt["file_sha256"],
              "code_sha256": {filename: file_digest(Path(__file__).with_name(filename)) for filename in (
                  Path(__file__).name, "open_speech_teacher_probe.py", "qwen_speech_encoder.py",
                  "wav2vec_harper_probe.py",
              )},
              "deadline_ms": DEADLINE, "frame_centers_samples": centers.tolist(),
              "student_hop_samples": HOP * 4, "student_window_samples": FRAME,
              "native_mid_block": {"wav2vec2-base": 6, "hubert-large": 12, "whisper-large-v3": 16, "qwen3-asr-1.7b": 12}[name]}
    saved = json.loads(manifest.read_text()) if manifest.is_file() else {"inputs": inputs, "shards": {}}
    if saved["inputs"] != inputs:
        raise ValueError("the existing target cache uses different inputs")
    left = np.maximum(0, SAMPLES - DEADLINE * RATE // 1000 - np.asarray([r["pause_start"] for _, r in rows]))
    codes = [str(r["language"]).strip().lower().replace("_", "-").split("-")[0] for _, r in rows]
    aliases = {"cmn": "zh", "zho": "zh", "chi": "zh", "none": "und", "unknown": "und"}
    languages = np.asarray([aliases.get(code, code) for code in codes])
    roles = np.asarray([r["role"] for _, r in rows])
    raw = np.empty(len(rows), dtype=np.float32)
    valid_frames = np.empty(len(rows), dtype=np.int16)
    meter = Progress(f"own-targets-{name}", len(rows), "pauses")
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), SHARD_ROWS):
                stop = min(start + SHARD_ROWS, len(rows))
                target = directory / f"{start // SHARD_ROWS:04d}.npz"
                digest = saved["shards"].get(target.name)
                if digest is not None:
                    if digest != file_digest(target):
                        raise ValueError(f"persisted teacher shard changed: {target.name}")
                    with np.load(target, allow_pickle=False) as cached:
                        raw[start:stop] = cached["raw_scores"]
                        valid_frames[start:stop] = cached["valid"].sum(axis=1)
                else:
                    matrices, masks = [], []
                    for offset in range(start, stop, BATCH):
                        edge = min(offset + BATCH, stop)
                        audio = torch.as_tensor(normalize(pcm[offset:edge, 0]), device="cuda")
                        pooled, mid = predict(audio, return_mid=True)
                        if mid.shape[1] != len(centers):
                            raise ValueError("native teacher dimensions do not match the sample grid")
                        raw[offset:edge] = predictions(pooled.cpu().numpy(), heads)
                        matrix, mask = align(mid, centers, left[offset:edge])
                        if not np.isfinite(matrix).all():
                            raise ValueError("nonfinite aligned teacher feature")
                        matrices.append(matrix)
                        masks.append(mask)
                    matrix = np.concatenate(matrices)
                    valid = np.concatenate(masks)
                    valid_frames[start:stop] = valid.sum(axis=1)
                    if not valid_frames[start:stop].all() or np.ptp(matrix[valid].astype(np.float32)) < 1e-6:
                        raise ValueError("the teacher has missing or constant valid features")
                    temporary = target.with_suffix(".part")
                    with temporary.open("wb") as stream:
                        np.savez(stream, raw_scores=raw[start:stop], features=matrix, valid=valid)
                    temporary.replace(target)
                    saved["shards"][target.name] = file_digest(target)
                    _write_json(manifest, saved)
                meter.advance(stop - start)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    targets, normalizers = language_targets(raw, roles, languages)
    output = directory / "scores.npy"
    temporary = output.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.save(stream, targets, allow_pickle=False)
    temporary.replace(output)
    counts = Counter(f"{r['source']}/{r['role']}/{language}" for (_, r), language in zip(rows, languages, strict=True))
    sources = np.asarray([r["source"] for _, r in rows])
    distributions = {}
    for source, role in sorted(set(zip(sources, roles, strict=True))):
        idx = (sources == source) & (roles == role)
        distributions[f"{source}/{role}"] = {
            name: {"quantiles": np.quantile(values[idx], [.01, .1, .5, .9, .99]).tolist(),
                   "largest_tie": int(np.unique(values[idx], return_counts=True)[1].max())}
            for name, values in (("raw", raw), ("normalized", targets))
        }
        distributions[f"{source}/{role}"]["valid_frames"] = int(valid_frames[idx].sum())
    result = {"cache": directory.relative_to(root).as_posix(), "inputs": inputs, "manifest_sha256": file_digest(manifest),
              "scores_sha256": file_digest(output), "normalizers": normalizers, "coverage": counts,
              "distribution_by_source_role": distributions}
    _write_json(root / "reports/own-teacher-distillation-targets.json", result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root()), indent=2))
