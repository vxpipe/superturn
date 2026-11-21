"""Compare matched real-label training with and without gain augmentation."""

import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .ami_transfer import partial_auc
from .dense_teacher import file_digest
from .own_head_distillation import student_scores
from .own_student_real_labels import SEEDS, STEPS, checked_inputs, fit
from .sources import data_root
from .train import _stats, _threshold
from .vxpipe_policy_probe import operating_points


def metrics(values, labels, cutoff):
    _, _, _, shifts, holds = operating_points(values, labels.astype(bool))
    point = np.flatnonzero(holds <= .05 * np.count_nonzero(labels == 0))[-1]
    return {"matched_recall": float(shifts[point] / np.count_nonzero(labels)),
            "matched_hold_errors": int(holds[point]),
            "partial_auc5": partial_auc(values.tolist(), labels.tolist(), max_fpr=.05),
            "fixed": _stats(values.tolist(), labels.tolist(), cutoff)}


def run(root):
    receipt, parent, data, replay, _, _ = checked_inputs(root)
    old_report_path = root / "reports/own-student-real-labels.json"
    archived = json.loads(old_report_path.read_text())
    paired = {}
    fits = {}
    x, labels, _ = data["english"]
    harper, harper_y, _ = data["harper-validation"]
    cal, cal_y, _ = data["ami-calibration"]
    quiet_gain = np.float32(10 ** (-6 / 20))
    loud_gain = np.float32(10 ** (6 / 20))
    safe = np.max(np.abs(x), axis=1) * loud_gain <= 1.
    gain_views = {"clean": x, "minus_6_db": x * quiet_gain, "clean_paired": x[safe],
                  "plus_6_db_paired": x[safe] * loud_gain, "minus_6_db_paired": x[safe] * quiet_gain,
                  "harper": harper}
    truth = {name: labels[safe] if name.endswith("_paired") else labels for name in gain_views}
    truth["harper"] = harper_y
    output = root / "runs/own-student-gain-robustness"
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS:
        provenance = parent["fits"][f"feature-first-{seed}"]["checkpoint"]
        for mode in ("control", "attenuation"):
            name = f"{mode}-{seed}"
            model, record, _ = fit(root, provenance, "full-ami", seed, data, replay,
                                   attenuate=mode == "attenuation", output_dir=output)
            cal_scores = student_scores(model, cal)
            threshold = _threshold(cal_scores.tolist(), cal_y.tolist())
            scores = {key: student_scores(model, clips) for key, clips in gain_views.items()}
            clean = metrics(scores["clean"], labels, threshold)
            expected = archived["metrics"][f"full-ami-{seed}"]["english"]["recalls_at_2_5_5_7_5pct"][1]
            if mode == "control" and clean["matched_recall"] != expected:
                raise ValueError("matched no-attenuation control did not reproduce")
            paired[name] = {
                "cutoff": threshold,
                "scenarios": {key: metrics(values, truth[key], threshold) for key, values in scores.items()},
                "pooled_clean_minus6": metrics(np.r_[scores["clean"], scores["minus_6_db"]],
                                               np.tile(labels, 2), threshold),
                "mean_absolute_probability_drift": {
                    key: float(np.abs(scores[key] - scores["clean_paired"]).mean())
                    for key in ("plus_6_db_paired", "minus_6_db_paired")},
            }
            fits[name] = record
    result = {"scope": "paired gain diagnostic, no final test or threshold updates",
              "parent_fit_report_sha256": file_digest(receipt), "control_receipt_sha256": file_digest(old_report_path),
              "updates": STEPS, "seeds": SEEDS, "samples": len(labels), "boost_safe_samples": int(safe.sum()),
              "metrics": paired, "fits": fits}
    _write_json(root / "reports/own-student-gain-robustness.json", result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root())["metrics"], indent=2))
