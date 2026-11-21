"""Train the score-only audio student on the verified own-teacher cache."""

import hashlib
import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .dense_teacher_train import SEEDS, fit_model
from .own_head_distillation import (
    DEADLINE, calibration_rows, evaluate_students, pause_rows, pool_rows,
)
from .own_teacher_pool_pcm import recover
from .sources import data_root
from .turn1_mini_probe import english_rows, harper_rows


def run(root):
    report_path = root / "reports/own-teacher-distillation-targets.json"
    provenance = json.loads(report_path.read_text())
    cache = root / provenance["cache"]
    scores = cache / "scores.npy"
    if (file_digest(scores) != provenance["scores_sha256"]
            or file_digest(cache / "manifest.json") != provenance["manifest_sha256"]
            or provenance["inputs"]["ranking_sha256"] != file_digest(root / "reports/own-teacher-language-selection.json")):
        raise ValueError("the verified teacher target cache changed")
    pcm_path, pcm_receipt = recover(root)
    _, pool_hash, rows = pool_rows(root)
    if (provenance["inputs"]["pcm_file_sha256"] != pcm_receipt["file_sha256"]
            or provenance["inputs"]["pool_sha256"] != pool_hash
            or provenance["inputs"]["deadline_ms"] != DEADLINE):
        raise ValueError("teacher targets and caller pool do not agree")
    pcm = np.load(pcm_path, mmap_mode="r", allow_pickle=False)
    targets = np.load(scores, allow_pickle=False)[:, None]
    if targets.shape != pcm.shape[:2] or not np.all(np.isfinite(targets) & (targets >= 0) & (targets <= 1)):
        raise ValueError("invalid or mismatched teacher probabilities")
    sources, roles, groups = (np.asarray([row[key] for _, row in rows])
                              for key in ("source", "role", "connected_group"))
    for source in np.unique(sources):
        observed = targets[(sources == source) & (roles == "fit")]
        if np.diff(np.quantile(observed, [.01, .99]))[0] < .05:
            raise ValueError(f"collapsed normalized teacher scores for {source}")
    torch.use_deterministic_algorithms(True)
    checksum = file_digest(report_path)
    output = root / "runs/own-teacher-score-student" / checksum[:12]
    fits = {}
    for seed in SEEDS:
        for shuffled in (False, True):
            name = f"score-only-{seed}-{'shuffle' if shuffled else 'true'}"
            config = {"architecture": "a", "window": 3, "parameters": 150000, "seed": seed, "shuffled": shuffled}
            fits["caller", seed, shuffled] = fit_model(output / name, config, pcm, targets, sources, roles, groups,
                                                      checksum, "cuda", deadlines=(DEADLINE,))
    cohorts = {"english": list(pause_rows(english_rows(root))),
               "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
               "calibration": list(calibration_rows(root))}
    metrics, intervals, control = evaluate_students(root, fits, cohorts)
    result = {"teacher": provenance["inputs"]["teacher"], "target_report_sha256": checksum, "decision_ms": DEADLINE,
              "scope": "audio-only score distillation, with archived checkpoints selected by fidelity BCE",
              "metrics": metrics, "paired_english": intervals, "control": control,
              "fits": {f"{seed}/{shuffled}": report for (_, seed, shuffled), (_, report) in fits.items()}}
    _write_json(root / "reports/own-teacher-score-student.json", result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = run(data_root())
    print(json.dumps({"teacher": result["teacher"], "english": result["metrics"]["english"]}, indent=2))
