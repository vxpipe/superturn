"""Fit the active-frame frontend arm against the frozen feature-first recipe."""

import argparse
import hashlib
import json

import numpy as np
from scipy.special import expit, logit
import torch

from .ami_calibration import _write_json
from .audio import RATE
from .augment import condition
from .dense_teacher import SAMPLES, SOURCES, file_digest, normalize
from .own_head_distillation import DEADLINE, pool_rows, predictions
from .own_teacher_feature_student import (
    DEVICE, EPOCHS, SEEDS, compare, fit_arm, read_inputs, replace_history, targets_for_features,
)
from .own_teacher_targets import teacher
from .progress import Progress
from .shards import seed_for
from .sources import data_root


def comfort_noise(seed=271):
    edge = SAMPLES - DEADLINE * RATE // 1000
    rng = np.random.default_rng(seed)
    raw = np.round(rng.normal(0, 110, edge)).astype(np.int16)
    noise = condition(raw, "telephone", 0).astype(np.float32)
    noise *= 10 ** (-55 / 20) / np.sqrt(np.mean(noise ** 2))
    return np.pad(noise, (0, SAMPLES - edge))


def history_inputs(root, pcm, scores, fit, output, target_sha):
    edge = SAMPLES - DEADLINE * RATE // 1000
    target_report = json.loads((root / "reports/own-teacher-distillation-targets.json").read_text())
    _, pool_sha, pool = pool_rows(root)
    prefix = np.zeros(len(pool), dtype=np.int32)
    for index in fit:
        ordinal, row = pool[index]
        observed = min(edge, row["pause_start"])
        keep = RATE * 8 // 10 + seed_for("own-student-history-v1", ordinal) % (edge - RATE * 8 // 10 + 1)
        if keep < observed:
            prefix[index] = edge - keep
    noise = torch.as_tensor(np.stack([comfort_noise(6611 + i) for i in range(32)]), device=DEVICE)
    manifest = output / "history-targets.json"
    path = output / "history-targets.npz"
    receipts = {"target_report_sha256": target_sha, "pool_sha256": pool_sha,
                "prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
                "noise_sha256": hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()}
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved["inputs"] != receipts or saved["sha256"] != file_digest(path):
            raise ValueError("archived augmented targets do not match the fixed recipe")
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(prefix, data["prefix"]):
                raise ValueError("archived augmented membership changed")
            return (prefix, noise, data["scores"]), saved
    selection = json.loads((root / "reports/own-teacher-language-selection.json").read_text())
    name, descriptor, predict, heads, _ = teacher(root, selection)
    if (name != target_report["inputs"]["teacher"]
            or descriptor["head_sha256"] != target_report["inputs"]["teacher_head_sha256"]):
        raise ValueError("the history targets do not use the original owned teacher")
    altered = np.flatnonzero(prefix)
    if not len(altered):
        raise ValueError("no staged fit row has enough real history to truncate")
    history_scores = scores[:, 0].copy()
    normalizers = target_report["normalizers"]
    aliases = {"cmn": "zh", "zho": "zh", "chi": "zh", "none": "und", "unknown": "und"}
    meter = Progress("own-student-history-targets", len(altered), "fit pauses")
    try:
        with torch.inference_mode():
            for left in range(0, len(altered), 32):
                idx = altered[left:left + 32]
                lengths = torch.as_tensor(prefix[idx], device=DEVICE)
                original = torch.as_tensor(normalize(pcm[idx, 0]), device=DEVICE)
                audio = replace_history(original, lengths, noise[idx % len(noise)])
                raw = predictions(predict(audio).cpu().numpy(), heads)
                logits = logit(np.asarray(raw, dtype=np.float64).clip(1e-6, 1 - 1e-6))
                for ordinal, value in zip(idx, logits, strict=True):
                    code = str(pool[ordinal][1]["language"]).strip().lower().replace("_", "-").split("-")[0]
                    code = aliases.get(code, code)
                    stats = normalizers["by_language"].get(code, normalizers["global"])
                    history_scores[ordinal] = expit((value - stats["mean"]) / stats["std"])
                meter.advance(len(idx))
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    if not np.isfinite(history_scores).all() or not np.array_equal(history_scores[prefix == 0], scores[prefix == 0, 0]):
        raise ValueError("history targets invalid or unmodified rows changed")
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez(stream, prefix=prefix, scores=history_scores)
    temporary.replace(path)
    saved = {"inputs": receipts, "sha256": file_digest(path), "altered_fit_rows": len(altered),
             "cache": path.relative_to(root).as_posix()}
    _write_json(manifest, saved)
    return (prefix, noise, history_scores), saved


def run(root, *, history=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("frontend training requires the staged Vast data")
    reference_path = root / "reports/own-teacher-feature-student.json"
    reference = json.loads(reference_path.read_text())
    cache, digests, pcm, targets, sources, roles, groups, target_sha = read_inputs(root)
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    if (reference["target_report_sha256"] != target_sha or reference["epochs"] != EPOCHS
            or reference["seeds"] != list(SEEDS) or reference["fit_rows"] != len(fit)
            or reference["fidelity_rows"] != len(fidelity)):
        raise ValueError("the baseline fit contract changed")
    archived = root / "runs/own-teacher-feature-student" / target_sha[:12]
    pca = archived / "pca.pt"
    if file_digest(pca) != reference["pca_sha256"]:
        raise ValueError("the archived fit-only PCA changed")
    variant = "own-student-history-views" if history else "own-student-active-stats"
    tag = f"{target_sha[:12]}-noise-bank" if history else target_sha[:12]
    output = root / "runs" / variant / tag
    output.mkdir(parents=True, exist_ok=True)
    active_sources = tuple(source for source in SOURCES if source in sources)
    torch.use_deterministic_algorithms(True)
    score_history, history_receipt = history_inputs(root, pcm, targets, fit, output, target_sha) if history else (None, None)
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    fits = {}
    for seed in SEEDS:
        name = f"feature-first-{seed}"
        fits[name] = fit_arm(pcm, targets, features, valid, sources, groups, fit, fidelity,
                             active_sources, "feature-first", seed, active_stats=True, score_history=score_history)
    metrics, _ = compare(root, output, fits)
    report = {"scope": "active frontend diagnostic on the original views", "history_receipt": history_receipt,
              "baseline_sha256": file_digest(reference_path), "target_sha256": target_sha,
              "pca_sha256": file_digest(pca), "fits": {key: row[2] for key, row in fits.items()},
              "metrics": metrics}
    _write_json(root / "reports" / f"{variant}.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-views", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(data_root(), history=args.history_views)
