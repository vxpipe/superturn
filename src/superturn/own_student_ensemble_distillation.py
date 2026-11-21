"""Compare a fixed owned-teacher/independent-student blend with original targets."""

from hashlib import sha256
import json

import numpy as np
from scipy.special import expit, logit
import torch

from .ami_calibration import _write_json
from .dense_teacher import SOURCES, file_digest
from .failure_strata import failure_comparison
from .harper_history_probe import recall_area
from .language_failure_audit import TEACHER, read_receipt, read_verified
from .model import build
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import student_scores
from .own_student_ami_supervision import summarize
from .own_student_failure_refresh import ProbabilityMean, agree, load_model
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, EPOCHS, SEEDS, compare, fit_arm, read_inputs, targets_for_features
from .own_telephone_distillation import fixed_rates
from .progress import Progress
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows

NEW_SEEDS = (101, 113, 127)
ARMS = ("control", "blend", "parent")
CONTRASTS = (("blend", "parent"), ("blend", "control"))
WEIGHT = .5


def target_blend(teacher, ensemble):
    if (teacher.shape != ensemble.shape
            or not all(np.all(np.isfinite(array) & (array >= 0) & (array <= 1)) for array in (teacher, ensemble))):
        raise ValueError("invalid fit-target predictions")
    left, right = (logit(np.asarray(array, dtype=np.float64).clip(1e-5, 1 - 1e-5))
                   for array in (teacher, ensemble))
    left_mean, left_std = left.mean(), left.std()
    right_mean, right_std = right.mean(), right.std()
    if min(left_std, right_std) <= 1e-6:
        raise ValueError("a fit logit distribution is constant")
    values = expit(left_mean + left_std * ((1 - WEIGHT) * (left - left_mean) / left_std
                                          + WEIGHT * (right - right_mean) / right_std))
    quantiles = {name: np.quantile(array, [.1, .5, .9]).tolist()
                 for name, array in (("teacher", teacher), ("ensemble", ensemble), ("blend", values))}
    spreads = {name: row[-1] - row[0] for name, row in quantiles.items()}
    return values, {"quantiles_10_50_90": quantiles, "teacher_logit_mean_std": [float(left_mean), float(left_std)],
                    "ensemble_logit_mean_std": [float(right_mean), float(right_std)],
                    "probability_spread_ratio": spreads["blend"] / spreads["teacher"] if spreads["teacher"] > 0 else 0.}


def lock_targets(root, width, pcm, scores, fit, directory, target_sha):
    parents = [width["fits"][model_name(1_000_000, seed)]["checkpoint"] for seed in SEEDS]
    ensemble = ProbabilityMean([load_model(root, receipt) for receipt in parents])
    meter = Progress("ensemble-distill-fit-predictions", len(fit), "fit pauses")
    pieces = []
    try:
        for start in range(0, len(fit), 256):
            rows = fit[start:start + 256]
            pieces.append(student_scores(ensemble, pcm[rows, 0].astype(np.float32) / 32768))
            meter.advance(len(rows))
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    mean = np.concatenate(pieces)
    blended, stats = target_blend(scores[fit, 0], mean)
    blended = blended.astype(scores.dtype)
    report = {"scope": "in-sample fit-row target generation only", "owned_target_sha256": target_sha,
              "original_parents": parents, "fit_pauses": len(fit), "blend_weight": WEIGHT,
              "training_target_dtype": str(blended.dtype),
              "fit_indices_sha256": sha256(fit.astype("<i8").tobytes()).hexdigest(), **stats}
    path = directory / "fit-targets.npz"
    with path.with_suffix(".part").open("wb") as output:
        np.savez_compressed(output, indices=fit, ensemble=mean, blend=blended)
    path.with_suffix(".part").replace(path)
    report["target_file"] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    report["spread_precheck_passed"] = stats["probability_spread_ratio"] >= .5
    _write_json(root / "reports/own-student-ensemble-targets.json", report)
    if not report["spread_precheck_passed"]:
        raise ValueError("blended 10th-to-90th percentile spread is under half the teacher spread")
    targets = scores.copy()
    targets[fit, 0] = blended
    return targets, report


def assess(root, models, metrics, directory, prior_rows, archived, absent):
    rows = {}
    score_languages(root, models, DEVICE, paired_rows=rows, progress_name="ensemble-distill-multilingual")
    audio, truth, turns = zip(*pause_rows(english_rows(root)), strict=True)
    rows["en"] = {"labels": truth, "turn_ordinals": turns,
                  "scores": {name: student_scores(model, np.stack(audio)).tolist() for name, model in models.items()}}
    for lang, row in rows.items():
        old = prior_rows[lang]
        agree(old, list(row["labels"]), list(row["turn_ordinals"]), lang)
        row["scores"].update({f"parent-{seed}": old["scores"][f"973k-{seed}"] for seed in SEEDS})
    summary = summarize(rows, arm_names=ARMS, contrasts=CONTRASTS, arm_seeds=dict.fromkeys(ARMS[:2], NEW_SEEDS))
    values = np.asarray(summary["macro"])
    control, blend, parents = values
    low = np.asarray(summary["paired_macro_95"])[:, 1, 0]
    english = np.asarray(summary["by_language"]["en"])
    deltas = np.asarray([row[1][1] - row[2][1] for row in summary["by_language"].values()])
    ceiling = archived["macro"]["973k"][1]
    gap = ceiling - parents[1]
    if gap <= 0:
        raise ValueError("the verified independent ensemble did not exceed the parent seed mean")
    checks = {
        "plus_1_5_points_at_matched_5_with_positive_lb_vs_parents": bool(blend[1] - parents[1] >= .015 and low[0] > 0),
        "normalized_pauc_at_least_parents": bool(blend[2] >= parents[2]),
        "no_language_declines_over_3_points": bool(deltas.min() >= -.03),
        "english_across_budgets_within_1_point_vs_parents": bool(english[1, 0] - english[2, 0] >= -.01),
    }
    comparisons = {"blend_vs_control": ("blend", "control"), "blend_vs_ensemble": ("blend", "ensemble")}
    failures = {}
    for lang, row in rows.items():
        old = prior_rows[lang]
        names = {"ensemble": old["scores"]["973k"], TEACHER: old["scores"][TEACHER]}
        # Row-level independent-parent comparison uses the ensemble's own matched threshold.
        for arm in ARMS[:2]:
            names[arm] = np.mean([row["scores"][f"{arm}-{seed}"] for seed in NEW_SEEDS], axis=0)
        failures[lang] = failure_comparison(names, TEACHER, row["labels"], row["turn_ordinals"],
                                             old["features"], comparisons)
    english_seeds = {}
    y = np.asarray(rows["en"]["labels"])
    for name, chosen in (("control", NEW_SEEDS), ("blend", NEW_SEEDS), ("parent", SEEDS)):
        individual = np.asarray([recall_area(np.asarray(rows["en"]["scores"][f"{name}-{seed}"]), y)[:3] for seed in chosen])
        english_seeds[name] = {"seeds": chosen, "mean": individual.mean(0).tolist(),
                               "range": np.stack([individual.min(0), individual.max(0)], axis=1).tolist()}
    path = directory / "paired-language-scores.json"
    _write_json(path, rows)
    return {"summary": summary, "gate": checks, "clears_parent_quality_gate": all(checks.values()),
            "english_matched_5_delta_vs_parents": float(english[1, 1] - english[2, 1]),
            "matched_control_paired_advantage": bool(blend[1] > control[1] and low[1] > 0),
            "development_ensemble_ceiling": ceiling, "share_of_ensemble_gap": float((blend[1] - parents[1]) / gap),
            "english_seed_metrics_at_2_5_5_7_5": english_seeds, "fixed_ami_counts": fixed_rates({
                lang: {**row, "scores": {name: row["scores"][name] for name in models}} for lang, row in rows.items()
            }, metrics),
            "absent_fit_languages": absent,
            "absent_fit_macro": np.mean([summary["by_language"][lang] for lang in absent], axis=0).tolist(),
            "failure_strata": failures, "paired_scores": {"path": path.relative_to(root).as_posix(),
                                                          "sha256": file_digest(path)}}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("ensemble distillation needs staged Vast audio")
    width, width_sha = read_receipt(root, "own-student-width-probe")
    audit, audit_sha = read_receipt(root, "own-student-failure-refresh")
    prior_rows = json.loads(read_verified(root, audit["paired_rows"]).read_text())
    cache, shards, pcm, scores, sources, roles, groups, target_sha = read_inputs(root)
    if width["target_report_sha256"] != target_sha or audit["experiment_receipts"]["width"] != width_sha:
        raise ValueError("the parent and failure audit did not use the expected receipts")
    if any(width["fits"][model_name(1_000_000, seed)]["selected"]["epoch"] != EPOCHS for seed in SEEDS):
        raise ValueError("the expected unchanged parents were not all selected at the final epoch")
    fit, fidelity = (np.flatnonzero(roles == name) for name in ("fit", "fidelity"))
    directory = root / "runs/own-student-ensemble-distillation" / width_sha[:12]
    directory.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    targets, blend = lock_targets(root, width, pcm, scores, fit, directory, target_sha)
    if not np.array_equal(targets[fidelity], scores[fidelity]):
        raise ValueError("the blend changed fit-excluded fidelity targets")
    pca = root / "runs/own-teacher-feature-student" / target_sha[:12] / "pca.pt"
    if file_digest(pca) != width["pca_sha256"]:
        raise ValueError("archived feature projection changed")
    features, valid = targets_for_features(cache, shards, sources, roles, pca, reuse_pca=True)
    active = tuple(source for source in SOURCES if source in sources)
    fits = {}
    for seed in NEW_SEEDS:
        for arm, values in (("control", scores), ("blend", targets)):
            fits[f"{arm}-{seed}"] = fit_arm(pcm, values, features, valid, sources, groups, fit, fidelity, active,
                                          "feature-first", seed, parameters=1_000_000, select_final=True,
                                          progress_prefix=f"ensemble-distill-{arm}")
    report = {"scope": "matched fit-only owned-model target comparison, development checks only",
              "width_sha256": width_sha, "failure_refresh_sha256": audit_sha,
              "blend_target_receipt_sha256": file_digest(root / "reports/own-student-ensemble-targets.json"),
              "target_spread": blend, "new_seeds": NEW_SEEDS, "epoch": EPOCHS,
              "fits": {name: fit[2] for name, fit in fits.items()}}
    path = root / "reports/own-student-ensemble-distillation.json"
    _write_json(path, report)
    if any(row[1] is None for row in fits.values()):
        raise ValueError("a fixed final checkpoint failed held-source spread or tie checks")
    metrics, paired = compare(root, directory, fits, peers={f"blend-{seed}": f"control-{seed}" for seed in NEW_SEEDS})
    models = {name: load_model(root, row[2]["checkpoint"]) for name, row in fits.items()}
    report.update(assess(root, models, metrics, directory, prior_rows, audit, audit["absent_fit_languages"]))
    report.update({"english_harper_ami": metrics, "paired_english_control": paired})
    _write_json(path, report)
    print(json.dumps({key: report[key] for key in ("gate", "clears_parent_quality_gate", "summary",
                                                  "share_of_ensemble_gap")}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
