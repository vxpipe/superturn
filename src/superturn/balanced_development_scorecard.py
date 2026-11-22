"""Score frozen turn-detector outputs with a fixed group-disjoint development split."""

import argparse
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .ami_transfer import partial_auc
from .failure_strata import decisions
from .language_failure_audit import TEACHER, read_receipt, read_verified
from .own_student_failure_refresh import agree
from .own_student_fidelity_confirmation import NEW_SEEDS, TRIAL
from .sources import data_root
from .train import _auc

JOB = "balanced-development-scorecard"
STUDENTS = tuple(f"epoch20-{seed}" for seed in NEW_SEEDS)
UNSEEN_IN_FIT = ("ar", "hi", "id", "nl", "pt", "tr")
METRICS = ("balanced_accuracy", "shift_recall", "false_complete_rate", "recall_at_5pct", "auc", "pauc_10")


def calibration_split(groups, language):
    """Assign an entire source turn to one half before reading its labels."""
    groups = list(groups)
    order = sorted(set(groups), key=lambda group: hashlib.sha256(
        f"superturn-balanced-v1:{language}:{group}".encode()).digest())
    calibration = set(order[::2])
    return np.array([group in calibration for group in groups], dtype=bool)


def balanced_threshold(scores, labels):
    """Use grouped ties and prefer the highest threshold at the maximum Youden point."""
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("threshold calibration requires both classes")
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    fp = ends + 1 - tp
    values = tp * negatives - fp * positives
    best = int(np.argmax(values))
    return float(scores[ends[best]] if values[best] > 0 else np.nextafter(scores[0], np.inf))


def evaluate(scores, labels, threshold):
    if len(scores) != len(labels) or not np.isfinite(scores).all() or len(np.unique(labels)) != 2:
        raise ValueError("evaluation scores must be finite and have both classes")
    predicted = scores >= threshold
    shift = float(predicted[labels].mean())
    false = float(predicted[~labels].mean())
    return {"balanced_accuracy": (shift + 1 - false) / 2, "shift_recall": shift,
            "false_complete_rate": false, "recall_at_5pct": float(decisions(scores, labels)[labels].mean()),
            "auc": float(_auc(scores.tolist(), labels.astype(int).tolist())),
            "pauc_10": float(partial_auc(scores.tolist(), labels.astype(int).tolist(), max_fpr=.10))}


def summarize(seeds):
    return {metric: {"mean": float(np.mean([row[metric] for row in seeds])),
                     "min": min(row[metric] for row in seeds), "max": max(row[metric] for row in seeds)}
            for metric in METRICS}


def cell(row, language, thresholds=None):
    labels = np.asarray(row["labels"], dtype=bool)
    groups = row["turn_ordinals"]
    if len(labels) != len(groups):
        raise ValueError("the evaluation labels do not match their turn groups")
    calibrated = thresholds is None
    fit = calibration_split(groups, language) if calibrated else np.zeros(len(labels), dtype=bool)
    evaluation = ~fit
    data = {}
    for name, values in row["scores"].items():
        scores = np.asarray(values, dtype=float)
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            raise ValueError("invalid frozen predictions")
        cutoff = balanced_threshold(scores[fit], labels[fit]) if calibrated else thresholds[name]
        data[name] = {"threshold": cutoff, **evaluate(scores[evaluation], labels[evaluation], cutoff)}
    return {"fit_holds": int((fit & ~labels).sum()), "fit_shifts": int((fit & labels).sum()),
            "eval_holds": int((evaluation & ~labels).sum()), "eval_shifts": int((evaluation & labels).sum()),
            "eval_turn_groups": len({group for group, keep in zip(groups, evaluation, strict=True) if keep}),
            "group_split_sha256": hashlib.sha256(json.dumps(sorted(set(group for group, flag in
                zip(groups, fit, strict=True) if flag))).encode()).hexdigest(),
            "student_seed_mean_and_range": summarize([data[name] for name in STUDENTS]),
            "models": data}


def macro(cells, languages):
    names = set.intersection(*(set(cells[language]["models"]) for language in languages))
    models = {name: {metric: float(np.mean([cells[language]["models"][name][metric] for language in languages]))
                     for metric in METRICS} for name in sorted(names)}
    return {"languages": list(languages), "models": models,
            "student_seed_mean_and_range": summarize([models[name] for name in STUDENTS]),
            "eval_holds": sum(cells[language]["eval_holds"] for language in languages),
            "eval_shifts": sum(cells[language]["eval_shifts"] for language in languages)}


def frozen(root):
    trial, trial_sha = read_receipt(root, TRIAL)
    teacher, teacher_sha = read_receipt(root, "own-student-failure-refresh")
    student_rows = json.loads(read_verified(root, trial["paired_scores"]).read_text())
    owned_rows = json.loads(read_verified(root, teacher["paired_rows"]).read_text())
    rows = {}
    for language, row in student_rows.items():
        agree(owned_rows[language], row["labels"], row["turn_ordinals"], language)
        rows[language] = {"labels": row["labels"], "turn_ordinals": row["turn_ordinals"],
                          "scores": {**{name: row["scores"][name] for name in STUDENTS},
                                     TEACHER: owned_rows[language]["scores"][TEACHER]}}
    if len(rows) != 14 or not set(UNSEEN_IN_FIT) <= rows.keys():
        raise ValueError("the fixed fourteen-language cohort changed")
    return rows, trial_sha, teacher_sha


def run(root, *, references=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("development scores stay on Vast")
    rows, trial, teacher = frozen(root)
    if references:
        from .balanced_scorecard_references import append_references

        provenance = append_references(root, rows)
    else:
        provenance = {}
    cells = {language: cell(row, language) for language, row in rows.items() if language != "harper-88"}
    report = {"status": "development only, no deployment thresholds",
              "threshold_source": "per-language alternating hash-ranked whole-turn calibration half; evaluate complementary half",
              "metric_scope": "old recall and AUC/pAUC are retrospective diagnostic metrics on the same evaluation half",
              "release_gate": {"macro_balanced_accuracy": .80, "each_language": .70, "deadline_ms": 300},
              "pipeline_metrics": "no replay at these newly fitted thresholds",
              "trial_sha256": trial, "owned_teacher_receipt_sha256": teacher, "references": provenance,
              "macro": macro(cells, sorted(cells)), "zero_supervision_macro": macro(cells, UNSEEN_IN_FIT),
              "languages": cells}
    if "harper-88" in rows:
        thresholds = {name: value["threshold"] for name, value in cells["en"]["models"].items()}
        report["harper-88"] = {"threshold_source": "the English development calibration half, frozen",
                               **cell(rows["harper-88"], "en", thresholds)}
    path = root / "reports" / f"{JOB}{'-references' if references else ''}.json"
    _write_json(path, report)
    print(json.dumps({"macro": report["macro"], "english": cells["en"], "six_language": report["zero_supervision_macro"]},
                     indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true")
    run(data_root(), references=parser.parse_args().references)
