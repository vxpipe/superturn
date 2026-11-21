"""Compare frozen 973k epoch-20 fits with Smart Turn on the same English pauses."""

import json

import numpy as np

from .ami_calibration import _write_json
from .ami_external_baselines import load_smart_turn
from .eot_domain_probe import DATASET_SHA256, decision_audio
from .failure_strata import decisions
from .language_failure_audit import read_receipt, read_verified
from .own_student_failure_refresh import agree
from .own_student_fidelity_confirmation import NEW_SEEDS, TRIAL
from .own_student_seed_uncertainty import paired_seed_turns
from .progress import Progress
from .sources import data_root
from .turn1_mini_probe import english_rows


REPORT = "own-student-smart-turn-benchmark"
REFERENCES = {"smart-turn-3s": 48_000, "smart-turn-8s": 128_000}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the frozen benchmark uses staged Vast data")
    students, students_sha = read_receipt(root, TRIAL)
    benchmark, benchmark_sha = read_receipt(root, "smart-turn-matched-context")
    if benchmark["dataset_sha256"] != DATASET_SHA256 or benchmark["decision_ms"] != 300:
        raise ValueError("the frozen Smart Turn benchmark uses different timing or input")
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
               "smart_turn_checkpoint_sha256": model_sha, "by_model": points,
               "epoch20_mean_recall": float(np.mean([points[name]["shift_recall"] for name in candidate])),
               "epoch20_minus_smart_turn": effects}
    _write_json(root / "reports" / f"{REPORT}.json", results)
    print(json.dumps({key: results[key] for key in ("epoch20_mean_recall", "epoch20_minus_smart_turn")}, indent=2))


if __name__ == "__main__":
    run(data_root())
