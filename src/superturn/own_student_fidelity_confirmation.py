"""Lock fidelity-based predictions before scoring six new 973k fits."""

import argparse
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
from .own_student_seed_uncertainty import fidelity_by_source, paired_seed_turns, rank_evidence
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, EPOCHS, SEEDS, fit_arm, read_inputs, targets_for_features
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows


TRIAL = "own-student-fidelity-confirmation"
LOCK = f"{TRIAL}-lock"
NEW_SEEDS = (149, 163, 179, 193, 211, 227)
EXTENDED_EPOCHS = 20


def artifact(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def train_and_lock(root):
    lock_path = root / "reports" / f"{LOCK}.json"
    if lock_path.exists():
        raise ValueError("the prospective ranking is already locked")
    width, width_sha = read_receipt(root, "own-student-width-probe")
    uncertainty, uncertainty_sha = read_receipt(root, "own-student-seed-uncertainty")
    observed_seeds = {row["seed"] for row in uncertainty["fidelity_selector"]["fits"]}
    if observed_seeds.intersection(NEW_SEEDS) or uncertainty["width_receipt_sha256"] != width_sha:
        raise ValueError("the trial needs new seeds and the pinned control")
    cache, digests, pcm, scores, sources, roles, groups, target_sha = read_inputs(root)
    pca = root / "runs/own-teacher-feature-student" / target_sha[:12] / "pca.pt"
    if target_sha != width["target_report_sha256"] or file_digest(pca) != width["pca_sha256"]:
        raise ValueError("the fit data or archived PCA changed")
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    active = tuple(source for source in SOURCES if source in sources)
    directory = root / "runs" / TRIAL / width_sha[:12]
    directory.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    fits = {}
    for seed in NEW_SEEDS:
        def save(config, state, name):
            path = directory / f"{name}-{seed}.pt"
            torch.save({"config": config, "state": state}, path)
            return artifact(root, path)

        fixed = {}

        def save_fixed(epoch, config, state):
            fixed[f"epoch{epoch}_checkpoint"] = save(config, state, f"epoch{epoch}")

        config, state, receipt = fit_arm(
            pcm, scores, features, valid, sources, groups, fit, fidelity, active, "feature-first", seed,
            parameters=1_000_000, epochs=EXTENDED_EPOCHS, epoch_checkpoint=save_fixed, progress_prefix=TRIAL,
        )
        if state is None or len(fixed) != 2:
            raise ValueError(f"seed {seed} lacked a stable selection or a required fixed checkpoint")
        fits[str(seed)] = {**receipt, **fixed, "selected_checkpoint": save(config, state, "selected")}
        _write_json(directory / f"fit-{seed}.json", fits[str(seed)])
    predicted = sorted(fits, key=lambda seed: (-fits[seed]["selected"]["selection_key"][0], int(seed)))
    early_fidelity = {seed: float(np.mean(list(fidelity_by_source(fit["history"][EPOCHS - 1]).values())))
                      for seed, fit in fits.items()}
    if not np.isfinite(list(early_fidelity.values())).all():
        raise ValueError("a fixed epoch-10 checkpoint lacks a finite fidelity score")
    early_rank = sorted(fits, key=lambda seed: (-early_fidelity[seed], int(seed)))
    _write_json(lock_path, {"width_sha256": width_sha, "uncertainty_sha256": uncertainty_sha,
                           "target_sha256": target_sha, "pca_sha256": width["pca_sha256"], "fits": fits,
                           "seeds": NEW_SEEDS, "epochs": EXTENDED_EPOCHS, "predicted_rank": predicted,
                           "chosen_fit": predicted[0], "epoch10_fidelity": early_fidelity,
                           "epoch10_predicted_rank": early_rank, "epoch10_chosen_fit": early_rank[0]})
    print(f"lock_sha256={file_digest(lock_path)}")


def score_locked(root, expected_sha):
    lock, digest = read_receipt(root, LOCK)
    if expected_sha != digest or lock["seeds"] != list(NEW_SEEDS) or lock["epochs"] != EXTENDED_EPOCHS:
        raise ValueError("the prospective ranking receipt changed")
    width, width_sha = read_receipt(root, "own-student-width-probe")
    audit, audit_sha = read_receipt(root, "own-student-failure-refresh")
    if lock["width_sha256"] != width_sha or audit["experiment_receipts"]["width"] != width_sha:
        raise ValueError("the original width control or failure audit changed")
    prior = json.loads(read_verified(root, audit["paired_rows"]).read_text())
    early = tuple(f"epoch10-{seed}" for seed in NEW_SEEDS)
    final = tuple(f"epoch20-{seed}" for seed in NEW_SEEDS)
    selected = tuple(f"selected-{seed}" for seed in NEW_SEEDS)
    models = {name: load_model(root, lock["fits"][str(seed)][kind])
              for kind, names in (("epoch10_checkpoint", early), ("epoch20_checkpoint", final),
                                  ("selected_checkpoint", selected))
              for seed, name in zip(NEW_SEEDS, names, strict=True)}
    rows = {}
    score_languages(root, models, DEVICE, paired_rows=rows, progress_name=f"{TRIAL}-multilingual")
    audio, truth, turns = zip(*pause_rows(english_rows(root)), strict=True)
    audio = np.stack(audio)
    rows["en"] = {"labels": truth, "turn_ordinals": turns,
                  "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    width_rows = json.loads(read_verified(root, width["paired_language_scores"]).read_text())
    language_metrics, failure = {}, {}
    chosen = f'selected-{lock["chosen_fit"]}'
    chosen_epoch10 = f'epoch10-{lock["chosen_fit"]}'
    for language, row in rows.items():
        labels, groups = list(row["labels"]), list(row["turn_ordinals"])
        previous = prior[language]
        agree(previous, labels, groups, language)
        agree(width_rows[language], labels, groups, language)
        truth = np.asarray(labels)
        language_metrics[language] = {name: recall_area(np.asarray(values), truth)[:3]
                                      for name, values in row["scores"].items()}
        failure[language] = failure_comparison(
            {TEACHER: previous["scores"][TEACHER], "chosen": row["scores"][chosen],
             "epoch10": row["scores"][chosen_epoch10], "parent_mean": previous["scores"]["973k"]},
            TEACHER, labels, groups, previous["features"],
            {"chosen_vs_epoch10": ("chosen", "epoch10"), "chosen_vs_parent_mean": ("chosen", "parent_mean")},
        )
    macro = {name: float(np.mean([values[name][1] for values in language_metrics.values()])) for name in models}
    fidelity = np.asarray([lock["fits"][str(seed)]["selected"]["selection_key"][0] for seed in NEW_SEEDS])
    outcomes = np.asarray([macro[name] for name in selected])
    evidence = rank_evidence(outcomes, fidelity)
    early_outcomes = np.asarray([macro[name] for name in early])
    early_evidence = rank_evidence(early_outcomes, np.asarray([lock["epoch10_fidelity"][str(seed)] for seed in NEW_SEEDS]))
    early_chosen = f'epoch10-{lock["epoch10_chosen_fit"]}'
    early_gain = macro[early_chosen] - float(early_outcomes.mean())
    chosen_gain = macro[chosen] - float(outcomes.mean())
    schedule = paired_seed_turns(rows, selected, early)
    fixed_schedule = paired_seed_turns(rows, final, early)
    english = language_metrics["en"]
    truth = np.asarray(rows["en"]["labels"])
    archived_english = np.mean([recall_area(np.asarray(width_rows["en"]["scores"][model_name(1_000_000, seed)]),
                                           truth)[0] for seed in SEEDS])
    english_epoch10 = float(np.mean([english[name][0] for name in early]))
    english_selected = float(np.mean([english[name][0] for name in selected]))
    english_final = float(np.mean([english[name][0] for name in final]))

    def selector_passes(stats, gain, name):
        return bool(stats["spearman"] is not None and stats["spearman"] >= .83
                    and stats["one_sided_exact_p"] < .05 and gain >= .01
                    and english[name][0] - archived_english >= -.01)

    gates = {
        "epoch10_selector_replication": selector_passes(early_evidence, early_gain, early_chosen),
        "extended_schedule_selector": selector_passes(evidence, chosen_gain, chosen),
        "fixed_longer_schedule": bool(fixed_schedule["paired_seed_and_turn_95"][0] > 0
                                      and english_final - english_epoch10 >= -.01),
        "fidelity_selected_longer_schedule": bool(schedule["paired_seed_and_turn_95"][0] > 0
                                                  and english_selected - english_epoch10 >= -.01),
    }
    path = root / "runs" / TRIAL / width_sha[:12] / "paired-language-scores.json"
    _write_json(path, rows)
    result = {"scope": "prospective model ranking on new seeds; reused development audio",
              "lock_sha256": digest, "failure_audit_sha256": audit_sha, "selected_fidelity": evidence,
              "chosen": chosen, "chosen_gain_vs_seed_mean": chosen_gain, "schedule": schedule,
              "epoch10_fidelity": early_evidence, "epoch10_chosen": early_chosen,
              "epoch10_chosen_gain_vs_seed_mean": early_gain, "fixed_schedule": fixed_schedule,
              "selected_epoch_by_seed": {seed: fit["selected"]["epoch"] for seed, fit in lock["fits"].items()},
              "english_3_budget_recall": {"chosen": english[chosen][0], "archived_seed_mean": archived_english,
                                          "epoch10_seed_mean": english_epoch10, "epoch20_seed_mean": english_final,
                                          "epoch10_chosen": english[early_chosen][0], "selected_seed_mean": english_selected},
              "gates": gates, "matched_5_macro": macro, "language_metrics": language_metrics,
              "failure_strata": failure, "paired_scores": artifact(root, path)}
    _write_json(root / "reports" / f"{TRIAL}.json", result)
    print(json.dumps({key: result[key] for key in ("gates", "epoch10_fidelity", "selected_fidelity",
                                                  "chosen_gain_vs_seed_mean", "schedule", "fixed_schedule")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-locked-sha256")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the prospective trial requires staged Vast data")
    if args.score_locked_sha256:
        score_locked(root, args.score_locked_sha256)
    else:
        train_and_lock(root)
