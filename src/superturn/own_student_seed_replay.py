"""Reproduce the archived 973k seed before attributing a fresh-run gap."""

from hashlib import sha256
import json
import os

import numpy as np
import torch

from .ami_calibration import _write_json
from .dense_teacher import SOURCES, file_digest
from .harper_history_probe import recall_area
from .language_failure_audit import read_receipt, read_verified
from .model import build
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import student_scores
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, EPOCHS, SEEDS, fit_arm, read_inputs, targets_for_features
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the exact seed replay uses staged Vast data")
    width, width_sha = read_receipt(root, "own-student-width-probe")
    fresh, fresh_sha = read_receipt(root, "own-student-ensemble-distillation")
    original, original_sha = read_receipt(root, "own-teacher-feature-student")
    cache, shards, pcm, targets, sources, roles, groups, target_sha = read_inputs(root)
    if (width["archived_fit_report_sha256"] != original_sha or target_sha != width["target_report_sha256"]
            or fresh["width_sha256"] != width_sha or fresh["epoch"] != EPOCHS):
        raise ValueError("the replay inputs differ from the comparison receipts")
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    if (original["fit_rows"], original["fidelity_rows"], original["epochs"]) != (len(fit), len(fidelity), EPOCHS):
        raise ValueError("archived training membership or epoch count changed")
    pca = root / "runs/own-teacher-feature-student" / target_sha[:12] / "pca.pt"
    if file_digest(pca) != width["pca_sha256"] or width["pca_sha256"] != original["pca_sha256"]:
        raise ValueError("archived fit PCA changed")
    seed = SEEDS[0]
    name = model_name(1_000_000, seed)
    expected = width["fits"][name]
    checkpoint = torch.load(read_verified(root, expected["checkpoint"]), map_location="cpu", weights_only=True)
    archived = json.loads(read_verified(root, width["paired_language_scores"]).read_text())
    torch.use_deterministic_algorithms(False)
    teacher, valid = targets_for_features(cache, shards, sources, roles, pca, reuse_pca=True)
    active = tuple(source for source in SOURCES if source in sources)
    torch.use_deterministic_algorithms(True)
    other_teacher, other_valid = targets_for_features(cache, shards, sources, roles, pca, reuse_pca=True)
    projection_equal = np.array_equal(teacher, other_teacher) and np.array_equal(valid, other_valid)
    projection = {"historical_feature_sha256": sha256(teacher).hexdigest(), "historical_mask_sha256": sha256(valid).hexdigest(),
                  "deterministic_feature_sha256": sha256(other_teacher).hexdigest(),
                  "deterministic_mask_sha256": sha256(other_valid).hexdigest(), "matches": projection_equal}
    del other_teacher, other_valid
    config, state, replay = fit_arm(pcm, targets, teacher, valid, sources, groups, fit, fidelity, active,
                                    "feature-first", seed, parameters=1_000_000, progress_prefix="seed-replay")
    if state is None or config != checkpoint["config"] or state.keys() != checkpoint["state"].keys():
        raise ValueError("replayed student did not provide the archived architecture")
    different = []
    for key, value in state.items():
        prior = checkpoint["state"][key]
        if value.dtype != prior.dtype or value.shape != prior.shape or not torch.equal(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                prior.contiguous().reshape(-1).view(torch.uint8)):
            different.append(key)
    model = build(config).to(DEVICE).eval()
    model.load_state_dict(state)
    historical = build(config).to(DEVICE).eval()
    historical.load_state_dict(checkpoint["state"])
    rows = {}
    models = {"replay": model, "loaded_archive": historical}
    score_languages(root, models, DEVICE, paired_rows=rows, progress_name="exact-seed-multilingual")
    clips, truth, groups = zip(*pause_rows(english_rows(root)), strict=True)
    rows["en"] = {"labels": list(truth), "turn_ordinals": list(groups),
                  "scores": {key: student_scores(value, np.stack(clips)).tolist() for key, value in models.items()}}
    if set(archived) != set(rows):
        raise ValueError("development language membership changed")
    by_language = {}
    for lang, row in rows.items():
        prior = archived[lang]
        if row["labels"] != prior["labels"] or row["turn_ordinals"] != prior["turn_ordinals"]:
            raise ValueError(f"changed development membership for {lang}")
        actual, old = (np.asarray(values) for values in (row["scores"]["replay"], prior["scores"][name]))
        loaded = np.asarray(row["scores"]["loaded_archive"])
        y = np.asarray(row["labels"], dtype=bool)
        by_language[lang] = {"exact_scores": np.array_equal(actual, old),
                             "loaded_archive_reproduces_scores": np.array_equal(loaded, old),
                             "replay_matches_loaded_archive": np.array_equal(actual, loaded),
                             "max_score_error": float(np.max(np.abs(actual - old))),
                             "matched_5_replayed": float(recall_area(actual, y)[1]),
                             "matched_5_archived": float(recall_area(old, y)[1])}
    old_seeds, new_seeds = list(SEEDS), fresh["new_seeds"]
    old_macro = [float(np.mean([recall_area(np.asarray(row["scores"][model_name(1_000_000, value)]),
                                          np.asarray(row["labels"], dtype=bool))[1] for row in archived.values()]))
                 for value in old_seeds]
    fresh_rows = json.loads(read_verified(root, fresh["paired_scores"]).read_text())
    if set(fresh_rows) != set(archived):
        raise ValueError("fresh controls and original parents have different language sets")
    new_macro = []
    for value in new_seeds:
        recalls = []
        for lang, row in fresh_rows.items():
            if any(row[key] != archived[lang][key] for key in ("labels", "turn_ordinals")):
                raise ValueError(f"fresh control membership differs for {lang}")
            recalls.append(recall_area(np.asarray(row["scores"][f"control-{value}"]),
                                       np.asarray(row["labels"], dtype=bool))[1])
        new_macro.append(float(np.mean(recalls)))
    report = {"scope": "one exact historical training seed and score replay on staged development rows",
              "seed": seed, "width_sha256": width_sha, "ensemble_distillation_sha256": fresh_sha,
              "target_sha256": target_sha, "pca_sha256": width["pca_sha256"],
              "projection_order_check": projection,
              "parameter_tensors_differing_at_byte_level": different,
              "training_history_exact": replay["history"] == expected["history"],
              "differing_epochs": [epoch["epoch"] for epoch, saved in zip(replay["history"], expected["history"],
                                                                          strict=True) if epoch != saved],
              "selected_epoch": replay["selected"]["epoch"], "archived_epoch": expected["selected"]["epoch"],
              "development_scores_exact": all(row["exact_scores"] for row in by_language.values()),
              "development": by_language,
              "original_seed_macro_matched_5": dict(zip(old_seeds, old_macro, strict=True)),
              "fresh_control_seed_macro_matched_5": dict(zip(new_seeds, new_macro, strict=True)),
              "fit_order_sha256": [sha256(np.random.default_rng(seed_for("dense-teacher-order", seed, epoch))
                                            .permutation(fit).astype("<i8").tobytes()).hexdigest()
                                   for epoch in range(1, EPOCHS + 1)],
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
                              "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
                              "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                              "matmul_precision": torch.get_float32_matmul_precision(),
                              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                              "cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "cudnn_deterministic_flag": torch.backends.cudnn.deterministic,
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "torch_threads": torch.get_num_threads(), "training_loader_workers": 0},
              "archived_checkpoint": expected["checkpoint"]}
    _write_json(root / "reports/own-student-seed-replay.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
