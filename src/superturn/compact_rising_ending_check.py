"""Check fixed-threshold candidate errors on archived rising source-pitch endings."""

import json

import numpy as np

from .ami_calibration import _write_json
from .balanced_development_scorecard import STUDENTS, calibration_split
from .balanced_student_error_audit import joint, summarize
from .failure_strata import strata_masks
from .language_failure_audit import TEACHER, read_receipt, read_verified
from .own_student_failure_refresh import agree
from .own_student_fidelity_confirmation import NEW_SEEDS
from .own_student_wider_e20 import by_turn
from .sources import data_root

JOBS = {"own-student-wider-e20": "wider", "own-student-half-rate-e20": "half-rate"}
RISING_MARGIN = .03


def rising_interval(sets):
    rng = np.random.default_rng(119321)
    draws = []
    for seeds in rng.integers(len(NEW_SEEDS), size=(500, len(NEW_SEEDS))):
        counts, total = 0, 0
        for delta, turns in sets:
            sampled = np.concatenate([turns[i] for i in rng.integers(len(turns), size=len(turns))])
            counts += delta[seeds][:, sampled].mean(axis=0).sum()
            total += len(sampled)
        draws.append(counts / total)
    return np.quantile(draws, [.025, .975]).tolist()


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("private score audits run on Vast")
    source, source_sha = read_receipt(root, "own-student-failure-refresh")
    features = json.loads(read_verified(root, source["paired_rows"]).read_text())
    results = {}
    for job, prefix in JOBS.items():
        report, digest = read_receipt(root, job)
        pairs = json.loads(read_verified(root, report["paired_scores"]).read_text())
        language_cells, paired_rising = {}, {"hold": [], "shift": []}
        for lang, row in sorted(pairs.items()):
            archive = features[lang]
            agree(archive, row["labels"], row["turn_ordinals"], lang)
            masks = strata_masks(row["turn_ordinals"], archive["features"])
            final = ~calibration_split(row["turn_ordinals"], lang)
            truth = np.asarray(row["labels"], dtype=bool)[final]
            cutoff = report["languages"][lang]["models"]

            def decisions(name):
                return np.asarray(row["scores"][name])[final] >= cutoff[name]["threshold"]

            teacher = decisions(TEACHER) == truth
            old = np.stack([decisions(arm) == truth for arm in STUDENTS])
            new = np.stack([decisions(f"{prefix}-{seed}") == truth for seed in NEW_SEEDS])
            old_joint, new_joint = joint(old, teacher), joint(new, teacher)
            language_cells[lang] = {}
            for name in ("all", "short_history", "first_pause", "source_f0_rising", "source_f0_falling"):
                parts = {}
                for kind, check in (("shift", truth), ("hold", ~truth)):
                    selected = masks[name][final] & check
                    if selected.any():
                        delta = (new[:, selected].mean() - old[:, selected].mean()).item()
                        parts[kind] = {"n": int(selected.sum()), "candidate_minus_control_accuracy": delta,
                                       "candidate_joint_with_teacher": summarize(new_joint, selected),
                                       "control_joint_with_teacher": summarize(old_joint, selected)}
                        if name == "source_f0_rising":
                            groups = np.asarray(row["turn_ordinals"])[final][selected]
                            paired_rising[kind].append((new[:, selected].astype(np.int8) - old[:, selected],
                                                        by_turn(groups)))
                language_cells[lang][name] = parts
        rising = {kind: [row["source_f0_rising"][kind] for row in language_cells.values()
                         if kind in row["source_f0_rising"]] for kind in ("hold", "shift")}
        results[job] = {
            "training_report_sha256": digest, "languages": language_cells,
            "rising_pooled": {kind: {"pauses": sum(row["n"] for row in items),
                                     "accuracy_delta": float(np.average(
                                         [row["candidate_minus_control_accuracy"] for row in items],
                                         weights=[row["n"] for row in items])),
                                     "paired_seed_turn_delta_ci95": rising_interval(paired_rising[kind])}
                              for kind, items in rising.items() if items},
        }
        results[job]["passes_rising_guard"] = len(results[job]["rising_pooled"]) == 2 and all(
            row["paired_seed_turn_delta_ci95"][0] > -RISING_MARGIN
            for row in results[job]["rising_pooled"].values())
    data = {"scope": "frozen per-language development cutoffs at 300 ms; source-pitch strata are descriptive",
            "pitch_audit_sha256": source_sha, "arms": results}
    _write_json(root / "reports/compact-rising-ending-check.json", data)
    print(json.dumps({job: report["rising_pooled"] for job, report in results.items()}, indent=2))


if __name__ == "__main__":
    run(data_root())
