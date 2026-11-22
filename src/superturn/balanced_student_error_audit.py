"""Decompose the frozen student–teacher balanced-accuracy gap by class and language."""

from collections import defaultdict
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .balanced_development_scorecard import JOB as SCORECARD, STUDENTS, UNSEEN_IN_FIT, calibration_split, frozen
from .dense_teacher import file_digest
from .failure_strata import strata_masks
from .language_failure_audit import TEACHER, read_receipt, read_verified
from .sources import data_root

JOB = "balanced-student-error-audit"
PARTS = ("both_correct", "teacher_only_correct", "student_only_correct", "both_wrong")
CLASSES = ("hold", "shift")
REPLICATES = 500


def joint(student, teacher):
    """Return a seed-by-outcome-by-pause membership matrix."""
    return np.stack((student & teacher, ~student & teacher, student & ~teacher, ~student & ~teacher), axis=1)


def summarize(outcomes, mask):
    n = int(mask.sum())
    if not n:
        return {"n": 0}
    counts = outcomes[:, :, mask].sum(axis=2)
    means = counts.mean(axis=0)
    return {
        "n": n, "seed_mean_counts": dict(zip(PARTS, means.tolist(), strict=True)),
        "rates": dict(zip(PARTS, (means / n).tolist(), strict=True)),
        "teacher_minus_student_correct": float((means[1] - means[2]) / n),
    }


def bootstrap(differences, labels, groups, seed_draws, rng):
    """Resample the same student seeds for all languages and turns within language."""
    buckets = defaultdict(list)
    for index, group in enumerate(groups):
        buckets[group].append(index)
    turns = tuple(buckets.values())
    draws = []
    for seeds in seed_draws:
        while True:
            indices = np.concatenate([turns[i] for i in rng.integers(len(turns), size=len(turns))])
            shift = labels[indices]
            if shift.any() and (~shift).any():
                break
        rates = differences[seeds][:, indices].mean(axis=0)
        draws.append((rates[~shift].mean() / 2, rates[shift].mean() / 2))
    return np.asarray(draws)


def audit_language(language, row, metrics, archive, seed_draws, rng):
    labels = np.asarray(row["labels"], dtype=bool)
    groups = np.array(row["turn_ordinals"])
    fit = calibration_split(groups, language)
    ids = sorted(set(groups[fit].tolist()))
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != metrics["group_split_sha256"]:
        raise ValueError(f"calibration membership changed: {language}")
    eval_rows = ~fit
    truth = labels[eval_rows]
    expected = (metrics["eval_holds"], metrics["eval_shifts"])
    if ((~truth).sum(), truth.sum()) != expected:
        raise ValueError(f"evaluation class counts changed: {language}")

    def correct(name):
        return (np.asarray(row["scores"][name])[eval_rows] >= metrics["models"][name]["threshold"]) == truth

    teacher = correct(TEACHER)
    students = np.stack([correct(name) for name in STUDENTS])
    outcomes = joint(students, teacher)
    masks = strata_masks(groups, archive["features"])
    result = {stratum: {kind: summarize(outcomes, mask[eval_rows] & (truth == is_shift))
                        for kind, is_shift in zip(CLASSES, (False, True), strict=True)}
              for stratum, mask in masks.items()}
    gap = np.mean([result["all"][kind]["teacher_minus_student_correct"] for kind in CLASSES])
    expected_gap = (metrics["models"][TEACHER]["balanced_accuracy"] -
                    metrics["student_seed_mean_and_range"]["balanced_accuracy"]["mean"])
    if not np.isclose(gap, expected_gap, atol=1e-12, rtol=0):
        raise ValueError(f"paired decomposition does not reproduce balanced accuracy: {language}")
    draws = bootstrap(teacher.astype(np.int8) - students, truth, groups[eval_rows], seed_draws, rng)
    return {"strata": result, "gap": float(gap),
            "gap_ci95": np.quantile(draws.sum(axis=1), [.025, .975]).tolist()}, draws


def macro(cells, draws, languages):
    languages = tuple(languages)
    surfaces = {kind: {part: float(np.mean([cells[lang]["strata"]["all"][kind]["rates"][part]
                                          for lang in languages])) for part in PARTS} for kind in CLASSES}
    parts = {kind: (surface["teacher_only_correct"] - surface["student_only_correct"]) / 2
             for kind, surface in surfaces.items()}
    sampled = np.mean([draws[lang] for lang in languages], axis=0)
    return {"languages": list(languages), "macro_joint_rates": surfaces, "ba_gap_contributions": parts,
            "ba_gap_total": sum(parts.values()), "oracle_ba": 1 - np.mean([s["both_wrong"] for s in surfaces.values()]),
            "ba_gap_ci95": np.quantile(sampled.sum(axis=1), [.025, .975]).tolist()}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("use the private prediction archive on Vast")
    scorecard = root / "reports" / f"{SCORECARD}.json"
    report = json.loads(scorecard.read_text())
    rows, trial_sha, teacher_sha = frozen(root)
    if (report["trial_sha256"], report["owned_teacher_receipt_sha256"]) != (trial_sha, teacher_sha):
        raise ValueError("the scorecard predates the paired archive")
    teacher, _ = read_receipt(root, "own-student-failure-refresh")
    strata_archive = json.loads(read_verified(root, teacher["paired_rows"]).read_text())
    rng = np.random.default_rng(315104)
    seed_draws = rng.integers(len(STUDENTS), size=(REPLICATES, len(STUDENTS)))
    cells, draws = {}, {}
    for lang, row in sorted(rows.items()):
        cells[lang], draws[lang] = audit_language(lang, row, report["languages"][lang], strata_archive[lang],
                                                  seed_draws, rng)
    overall = macro(cells, draws, sorted(cells))
    expected = (report["macro"]["models"][TEACHER]["balanced_accuracy"] -
                report["macro"]["student_seed_mean_and_range"]["balanced_accuracy"]["mean"])
    if not np.isclose(overall["ba_gap_total"], expected, atol=1e-12, rtol=0):
        raise ValueError("paired macro gap disagrees with the sealed scorecard")
    result = {
        "scope": "fixed 300 ms development operating points; no thresholds fitted",
        "scorecard_sha256": file_digest(scorecard), "macro": overall,
        "zero_supervision_macro": macro(cells, draws, UNSEEN_IN_FIT), "languages": cells,
        "interval_scope": "resample student seeds jointly and evaluation turns separately within language; fixed cutoffs",
        "oracle_scope": "adjudicates model disagreements with labels; descriptive upper point for these systems",
    }
    _write_json(root / "reports" / f"{JOB}.json", result)
    print(json.dumps({"macro": overall, "six": result["zero_supervision_macro"]}, indent=2))


if __name__ == "__main__":
    run(data_root())
