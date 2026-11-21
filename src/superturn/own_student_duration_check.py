"""Compare fixed epoch 30 with six hash-pinned, exactly replayed epoch-20 controls."""

import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .dense_teacher import SOURCES, file_digest
from .failure_strata import failure_comparison
from .harper_history_probe import recall_area
from .language_failure_audit import TEACHER, read_receipt, read_verified
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import student_scores
from .own_student_failure_refresh import agree, load_model
from .own_student_fidelity_confirmation import EXTENDED_EPOCHS, LOCK, NEW_SEEDS, TRIAL, artifact
from .own_student_seed_uncertainty import paired_seed_turns
from .own_teacher_feature_student import DEVICE, fit_arm, read_inputs, targets_for_features
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows


JOB = "own-student-duration-check"
CANDIDATE_EPOCHS = 30


def cpu_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def verify_epoch(root, expected, config, state):
    payload = torch.load(read_verified(root, expected), map_location="cpu", weights_only=True)
    if config != payload["config"] or state.keys() != payload["state"].keys():
        raise ValueError("the epoch-20 model structure changed")
    for name, saved in payload["state"].items():
        current = state[name]
        if current.shape != saved.shape or current.dtype != saved.dtype or not torch.equal(
                cpu_bytes(current), cpu_bytes(saved)):
            raise ValueError(f"epoch-20 weight changed during replay: {name}")


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the duration check requires staged Vast data")
    lock, lock_sha = read_receipt(root, LOCK)
    result, trial_sha = read_receipt(root, TRIAL)
    if (result["lock_sha256"] != lock_sha or lock["seeds"] != list(NEW_SEEDS)
            or lock["epochs"] != EXTENDED_EPOCHS or not result["gates"]["fixed_longer_schedule"]):
        raise ValueError("the pinned 20-epoch recipe has not passed")
    previous = json.loads(read_verified(root, result["paired_scores"]).read_text())
    cache, digests, pcm, targets, sources, roles, groups, target_sha = read_inputs(root)
    pca = root / "runs/own-teacher-feature-student" / target_sha[:12] / "pca.pt"
    if lock["target_sha256"] != target_sha or lock["pca_sha256"] != file_digest(pca):
        raise ValueError("the fit data or archived PCA changed")
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    active = tuple(source for source in SOURCES if source in sources)
    directory = root / "runs" / JOB / lock_sha[:12]
    directory.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    fits = {}
    for seed in NEW_SEEDS:
        parent = lock["fits"][str(seed)]
        verified = False

        def check(epoch, config, state):
            nonlocal verified
            verify_epoch(root, parent["epoch20_checkpoint"], config, state)
            verified = epoch == EXTENDED_EPOCHS

        config, state, history = fit_arm(
            pcm, targets, features, valid, sources, groups, fit, fidelity, active, "feature-first", seed,
            parameters=1_000_000, epochs=CANDIDATE_EPOCHS, select_final=True, epoch_checkpoint=check,
            checkpoint_epochs=(EXTENDED_EPOCHS,), progress_prefix=JOB,
        )
        if not verified or history["history"][:EXTENDED_EPOCHS] != parent["history"]:
            raise ValueError(f"seed {seed} did not reproduce its first 20 epochs")
        if state is None:
            raise ValueError(f"seed {seed} did not pass score stability at epoch {CANDIDATE_EPOCHS}")
        name = f"epoch30-{seed}"
        path = directory / f"{name}.pt"
        torch.save({"config": config, "state": state}, path)
        fits[name] = {**history, "epoch20_reproduced": True, "checkpoint": artifact(root, path)}
        _write_json(directory / f"{name}.json", fits[name])
    checkpoint_lock = directory / "checkpoint-lock.json"
    _write_json(checkpoint_lock, {name: receipt["checkpoint"] for name, receipt in fits.items()})
    print(f"lock_sha256={file_digest(checkpoint_lock)}", flush=True)
    models = {name: load_model(root, receipt["checkpoint"]) for name, receipt in fits.items()}
    rows = {}
    score_languages(root, models, DEVICE, paired_rows=rows, progress_name=f"{JOB}-multilingual")
    clips, labels, turns = zip(*pause_rows(english_rows(root)), strict=True)
    audio = np.stack(clips)
    rows["en"] = {"labels": list(labels), "turn_ordinals": list(turns),
                  "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    control = tuple(f"epoch20-{seed}" for seed in NEW_SEEDS)
    candidate = tuple(f"epoch30-{seed}" for seed in NEW_SEEDS)
    metrics, failures = {}, {}
    audit, audit_sha = read_receipt(root, "own-student-failure-refresh")
    if audit_sha != result["failure_audit_sha256"]:
        raise ValueError("the verified failure audit changed")
    reference = json.loads(read_verified(root, audit["paired_rows"]).read_text())
    if rows.keys() != previous.keys():
        raise ValueError("development language membership changed")
    for language, row in rows.items():
        old = previous[language]
        agree(old, row["labels"], row["turn_ordinals"], language)
        base = reference[language]
        agree(base, row["labels"], row["turn_ordinals"], language)
        row["scores"].update({name: old["scores"][name] for name in control})
        truth = np.asarray(row["labels"])
        metrics[language] = {name: recall_area(np.asarray(values), truth).tolist()
                             for name, values in row["scores"].items()}
        groups = row["turn_ordinals"]
        score_pair = {key: np.mean([row["scores"][model] for model in names], axis=0)
                      for key, names in (("epoch20", control), ("epoch30", candidate))}
        failures[language] = failure_comparison({**score_pair, TEACHER: base["scores"][TEACHER]},
                                                TEACHER, truth, groups, base["features"],
                                                {"epoch30_vs_epoch20": ("epoch30", "epoch20")})
    effect = paired_seed_turns(rows, candidate, control)
    stats = {}
    for name, names in (("epoch20", control), ("epoch30", candidate)):
        values = np.mean([[m[model][1] for model in names] for m in metrics.values()], axis=0)
        english = float(np.mean([metrics["en"][model][:3] for model in names]))
        stats[name] = {"macro_seed_mean": float(values.mean()), "macro_seed_range": [float(values.min()), float(values.max())],
                       "english_three_budget_mean": english, "macro_per_seed": values.tolist()}
    english_change = stats["epoch30"]["english_three_budget_mean"] - stats["epoch20"]["english_three_budget_mean"]
    passed = effect["paired_seed_and_turn_95"][0] > 0 and english_change >= -.01
    path = directory / "paired-language-scores.json"
    _write_json(path, rows)
    report = {"scope": "development only; final fixed 973k schedule check",
              "fidelity_trial_sha256": trial_sha, "fidelity_trial_lock_sha256": lock_sha, "fits": fits,
              "checkpoint_lock": artifact(root, checkpoint_lock), "failure_strata": failures,
              "development": stats, "language_metrics": metrics, "paired_scores": artifact(root, path),
              "paired_duration_effect": effect, "english_three_budget_change": english_change,
              "epoch30_passes": bool(passed)}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps({key: report[key] for key in ("development", "paired_duration_effect", "epoch30_passes")}))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
