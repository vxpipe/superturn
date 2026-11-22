"""Isolate E20 capacity and frontend comparisons against archived controls."""

import argparse
from collections import defaultdict
import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .balanced_development_scorecard import (
    JOB as SCORECARD, STUDENTS, UNSEEN_IN_FIT, balanced_threshold, calibration_split, cell, frozen, macro,
)
from .dense_teacher import SOURCES, file_digest
from .harper_history_probe import recall_area
from .language_failure_audit import TEACHER, read_receipt, read_verified
from .model import parameter_count
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import student_scores
from .own_student_failure_refresh import agree, load_model
from .own_student_fidelity_confirmation import LOCK, NEW_SEEDS, TRIAL, artifact
from .own_teacher_feature_student import DEVICE, fit_arm, read_inputs, targets_for_features
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows

EPOCHS = 20
DRAWS = 500


def training(root, job, candidates, target, spectral_rate):
    control, digest = read_receipt(root, LOCK)
    trial, trial_digest = read_receipt(root, TRIAL)
    if trial["lock_sha256"] != digest or not trial["gates"]["fixed_longer_schedule"]:
        raise ValueError("the current control is required")
    cache, digests, pcm, scores, sources, roles, groups, targets = read_inputs(root)
    pca = root / "runs/own-teacher-feature-student" / targets[:12] / "pca.pt"
    if control["target_sha256"] != targets or control["pca_sha256"] != file_digest(pca):
        raise ValueError("width test changed fitting data or PCA")
    width = (272 if target == 1_000_000 else
             min(range(32, 433, 4), key=lambda value: abs(parameter_count("a", value, 3) - target)))
    if abs(parameter_count("a", width, 3) - target) > .03 * target:
        raise ValueError("explicit width does not match the desired parameter count")
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    active = tuple(source for source in SOURCES if source in sources)
    directory = root / "runs" / job / digest[:12]
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {"control_sha256": digest, "trial_sha256": trial_digest, "targets_sha256": targets,
                  "pca_sha256": control["pca_sha256"], "width": width, "epochs": EPOCHS, "seeds": list(NEW_SEEDS)}
    if spectral_rate != 16000:
        provenance["spectral_rate"] = spectral_rate
    manifest = directory / "inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("cached wider checkpoints use different fit inputs")
    _write_json(manifest, provenance)
    torch.use_deterministic_algorithms(True)
    checkpoints = {}
    for seed, candidate in zip(NEW_SEEDS, candidates, strict=True):
        receipt = directory / f"{candidate}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            read_verified(root, saved["checkpoint"])
        else:
            fixed = {}

            def save_epoch(epoch, config, state):
                path = directory / f"{candidate}.pt"
                temporary = path.with_suffix(".partial")
                torch.save({"config": config, "state": state}, temporary)
                temporary.replace(path)
                fixed["checkpoint"] = artifact(root, path)

            _, _, run = fit_arm(pcm, scores, features, valid, sources, groups, fit, fidelity, active,
                                "feature-first", seed, parameters=target, width=width, epochs=EPOCHS,
                                spectral_rate=spectral_rate, epoch_checkpoint=save_epoch,
                                checkpoint_epochs=(EPOCHS,), progress_prefix=job)
            saved = {**fixed, "final_fidelity": run["history"][-1]}
            _write_json(receipt, saved)
        checkpoints[candidate] = saved["checkpoint"]
    return directory, provenance, checkpoints


def by_turn(groups):
    turns = defaultdict(list)
    for idx, group in enumerate(groups):
        turns[group].append(idx)
    return tuple(turns.values())


def sample_partition(buckets, truth, rng):
    while True:
        indices = np.concatenate([buckets[i] for i in rng.integers(len(buckets), size=len(buckets))])
        labels = truth[indices]
        if labels.any() and (~labels).any():
            return indices


def delta_draws(language, row, candidates, seed_draws, rng):
    truth = np.asarray(row["labels"], dtype=bool)
    groups = np.asarray(row["turn_ordinals"])
    cut = calibration_split(groups, language)
    calibration, evaluation = (by_turn(groups[mask]) for mask in (cut, ~cut))
    scores = {name: np.asarray(row["scores"][name]) for name in (*STUDENTS, *candidates)}
    draws = []
    for seeds in seed_draws:
        a = sample_partition(calibration, truth[cut], rng)
        b = sample_partition(evaluation, truth[~cut], rng)
        shift = truth[~cut][b]

        def ba(name):
            values = scores[name]
            threshold = balanced_threshold(values[cut][a], truth[cut][a])
            decision = values[~cut][b] >= threshold
            return (decision[shift].mean() + (~decision[~shift]).mean()) / 2

        draws.append(np.mean([ba(candidates[idx]) - ba(STUDENTS[idx]) for idx in seeds]))
    return draws


def avg_english_three(row, names):
    truth = np.asarray(row["labels"], dtype=bool)
    subset = ~calibration_split(row["turn_ordinals"], "en")
    return float(np.mean([recall_area(np.asarray(row["scores"][name])[subset], truth[subset])[:3].mean()
                          for name in names]))


def run(root, *, half_rate=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the wider-student probe requires Vast")
    job = "own-student-half-rate-e20" if half_rate else "own-student-wider-e20"
    stem = "half-rate" if half_rate else "wider"
    candidates = tuple(f"{stem}-{seed}" for seed in NEW_SEEDS)
    target, spectral_rate = (1_000_000, 8000) if half_rate else (1_500_000, 16000)
    directory, provenance, checkpoints = training(root, job, candidates, target, spectral_rate)
    frozen_rows, trial, _ = frozen(root)
    models = {name: load_model(root, path) for name, path in checkpoints.items()}
    paired = {}
    score_languages(root, models, DEVICE, paired_rows=paired, progress_name=f"{job}-multilingual")
    clips, labels, groups = zip(*pause_rows(english_rows(root)), strict=True)
    english = np.stack(clips)
    paired["en"] = {"labels": labels, "turn_ordinals": groups,
                    "scores": {name: student_scores(model, english).tolist() for name, model in models.items()}}
    rng = np.random.default_rng(50017)
    seed_draws = rng.integers(len(NEW_SEEDS), size=(DRAWS, len(NEW_SEEDS)))
    cells, boot = {}, {}
    for language, row in sorted(frozen_rows.items()):
        new = paired[language]
        agree(row, list(new["labels"]), list(new["turn_ordinals"]), language)
        row["scores"].update(new["scores"])
        cells[language] = cell(row, language)
        boot[language] = delta_draws(language, row, candidates, seed_draws, rng)
    standard = json.loads((root / "reports" / f"{SCORECARD}.json").read_text())
    if standard["trial_sha256"] != trial:
        raise ValueError("the balanced control scorecard changed")
    overall = macro(cells, sorted(cells))
    reference = overall["student_seed_mean_and_range"]["balanced_accuracy"]["mean"]
    expected = standard["macro"]["student_seed_mean_and_range"]["balanced_accuracy"]["mean"]
    if not np.isclose(reference, expected, atol=1e-12, rtol=0):
        raise ValueError("the paired control no longer matches the standard scorecard")
    deltas = {lang: float(np.mean([row["models"][name]["balanced_accuracy"] for name in candidates]) -
                          row["student_seed_mean_and_range"]["balanced_accuracy"]["mean"])
              for lang, row in cells.items()}
    delta = float(np.mean(list(deltas.values())))
    ci = np.quantile(np.mean(list(boot.values()), axis=0), [.025, .975]).tolist()
    english_guard = avg_english_three(frozen_rows["en"], candidates) - avg_english_three(frozen_rows["en"], STUDENTS)
    archive = directory / "paired-scores.json"
    _write_json(archive, frozen_rows)
    report = {
        "scope": ("8 kHz analysis frontend at fixed 973k capacity" if half_rate else
                  "512-point width-only diagnostic; larger arm exceeds the audio-only parameter budget"),
        "threshold_source": "standard whole-turn development halves, candidate-specific calibration",
        "pipeline_metrics": "not measured", "teacher": TEACHER, "provenance": provenance,
        "parameters": sum(value.numel() for value in next(iter(models.values())).parameters()),
        "macro": overall, "zero_supervision_macro": macro(cells, UNSEEN_IN_FIT), "languages": cells,
        "paired_balanced_delta": delta, "paired_seed_calibration_turn_ci95": ci, "language_deltas": deltas,
        "english_legacy_mean_2_5_5_7_5_delta": english_guard,
        "decision": ("balanced-accuracy noninferiority margin one point" if half_rate else
                     "balanced-accuracy improvement at least two points"),
        "passes": bool((ci[0] > -.01 if half_rate else delta >= .02 and ci[0] > 0) and
                       deltas["en"] >= -.01 and english_guard >= -.01),
        "paired_scores": artifact(root, archive), "checkpoints": checkpoints,
    }
    _write_json(root / "reports" / f"{job}.json", report)
    print(json.dumps({"delta": delta, "interval": ci, "passed": report["passes"]}))


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-rate", action="store_true")
    run(data_root(), half_rate=parser.parse_args().half_rate)
