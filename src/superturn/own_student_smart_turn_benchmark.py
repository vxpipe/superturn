"""Compare frozen 973k epoch-20 fits with Smart Turn on the same English pauses."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .ami_external_baselines import load_smart_turn
from .audio import RATE
from .dense_teacher import file_digest
from .eot_domain_probe import DATASET_SHA256, GAP, decision_audio
from .failure_strata import cutoff_at_budget, decisions
from .language_failure_audit import read_receipt, read_verified
from .own_student_failure_refresh import agree
from .own_student_fidelity_confirmation import NEW_SEEDS, TRIAL
from .own_student_seed_uncertainty import paired_seed_turns
from .progress import Progress
from .sources import data_root
from .turn1_mini_probe import english_rows


REPORT = "own-student-smart-turn-benchmark"
REFERENCES = {"smart-turn-3s": 48_000, "smart-turn-8s": 128_000}
FEATURES_SHA256 = "139b047d10dee42ad5b4894489012e515a882d8f5e2bffe8ad27b30728fc9968"


def reference_history(rows, truth, values):
    elapsed = np.asarray([(row[1] + GAP) / RATE for row in rows])
    scores = {name: np.asarray(data) for name, data in values.items()}
    global_thresholds = {name: cutoff_at_budget(score, truth) for name, score in scores.items()}
    results = {}
    for name, choose in (("at_most_3s", elapsed <= 3), ("more_than_3s", elapsed > 3)):
        y = truth[choose]
        selected = {model: score[choose] for model, score in scores.items()}
        max_difference = None
        if len(y):
            max_difference = float(np.max(np.abs(selected["smart-turn-3s"] - selected["smart-turn-8s"])))
        cell = {"holds": int((~y).sum()), "shifts": int(y.sum()), "max_absolute_score_difference": max_difference}
        if y.any() and not y.all():
            cell["matched_within_bin_recall"] = {
                model: float(decisions(score, y)[y].mean()) for model, score in selected.items()
            }
            cell["global_threshold_shifts"] = {
                model: int((score[y] >= global_thresholds[model]).sum()) for model, score in selected.items()
            }
        results[name] = cell
    return {"scope": "elapsed from extracted clip start; no deployment weighting",
            "median_elapsed_seconds": float(np.median(elapsed)), "bins": results}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the frozen benchmark uses staged Vast data")
    students, students_sha = read_receipt(root, TRIAL)
    benchmark, benchmark_sha = read_receipt(root, "smart-turn-matched-context")
    if benchmark["dataset_sha256"] != DATASET_SHA256 or benchmark["decision_ms"] != 300:
        raise ValueError("the frozen Smart Turn benchmark uses different timing or input")
    frontend = importlib.util.find_spec("_whisper_features")
    if frontend is None or frontend.origin is None or file_digest(Path(frontend.origin)) != FEATURES_SHA256:
        raise ValueError("set PYTHONPATH to the pinned Smart Turn benchmark frontend cache")
    archive = json.loads(read_verified(root, students["paired_scores"]).read_text())["en"]
    score, model_sha = load_smart_turn(root)
    if model_sha != benchmark["teacher_sha256"]:
        raise ValueError("the frozen Smart Turn checkpoint changed")
    rows = list(english_rows(root))
    labels = [row[2] for row in rows]
    turns = [row[3] for row in rows]
    agree(archive, labels, turns, "en")
    truth = np.asarray(labels, dtype=bool)
    values = {name: [] for name in REFERENCES}
    meter = Progress(REPORT, len(rows), "pauses")
    try:
        for pcm, edge, _, _ in rows:
            clip = decision_audio(pcm, edge, "telephone")
            for name, size in REFERENCES.items():
                values[name].append(score(clip[-size:]))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    candidate = tuple(f"epoch20-{seed}" for seed in NEW_SEEDS)
    scores = {**values, **{name: archive["scores"][name] for name in candidate}}

    def at_budget(name):
        fired = decisions(np.asarray(scores[name]), truth)
        return {"shift_recall": float(fired[truth].mean()), "false_completes": int(fired[~truth].sum()),
                "true_shifts": int(fired[truth].sum())}

    points = {name: at_budget(name) for name in scores}
    if (len(rows) != 932 or int(truth.sum()) != 400
            or points["smart-turn-3s"]["shift_recall"] != .2525 or points["smart-turn-8s"]["shift_recall"] != .355):
        raise ValueError("the historical reference operating points did not reproduce")
    paired = {"en": {"labels": labels, "turn_ordinals": turns, "scores": scores}}
    effects = {name: paired_seed_turns(paired, candidate, (name,) * len(candidate)) for name in REFERENCES}
    results = {"scope": "frozen benchmark comparison only; development thresholds matched retrospectively",
               "decision_ms": 300, "dataset_sha256": DATASET_SHA256, "rows": len(rows),
               "turns": len(set(turns)), "shifts": int(truth.sum()), "holds": int((~truth).sum()),
               "student_receipt_sha256": students_sha, "smart_turn_receipt_sha256": benchmark_sha,
               "smart_turn_frontend_sha256": FEATURES_SHA256,
               "smart_turn_checkpoint_sha256": model_sha, "by_model": points,
               "epoch20_mean_recall": float(np.mean([points[name]["shift_recall"] for name in candidate])),
               "epoch20_minus_smart_turn": effects, "smart_turn_history_strata": reference_history(rows, truth, values)}
    _write_json(root / "reports" / f"{REPORT}.json", results)
    print(json.dumps({key: results[key] for key in ("epoch20_mean_recall", "epoch20_minus_smart_turn",
                                                    "smart_turn_history_strata")}, indent=2))


if __name__ == "__main__":
    run(data_root())
