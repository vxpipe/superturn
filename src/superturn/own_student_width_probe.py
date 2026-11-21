"""Compare wider compact students with the archived feature-first fit."""

from collections import defaultdict
import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .dense_teacher import SOURCES, file_digest
from .harper_history_probe import recall_area
from .language_failure_audit import read_receipt, read_verified
from .model import build
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import student_scores
from .own_teacher_feature_student import DEVICE, EPOCHS, SEEDS, compare, fit_arm, read_inputs, targets_for_features
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows

TARGETS = (150000, 600000, 1000000)


def model_name(size, seed):
    return f"feature-first-{seed}" if size == TARGETS[0] else f"feature-first-{size}-{seed}"


def summarize(rows):
    rng = np.random.default_rng(29032)
    results, bootstrap = {}, []
    for language, row in rows.items():
        truth = np.asarray(row["labels"], dtype=bool)
        groups = np.asarray(row["turn_ordinals"])
        scores = {size: np.asarray([row["scores"][model_name(size, seed)] for seed in SEEDS])
                  for size in TARGETS}

        def recalls(index):
            return np.array([np.mean([recall_area(item[index], truth[index])[1] for item in scores[size]])
                             for size in TARGETS])

        metrics = recalls(np.arange(len(truth)))
        results[language] = dict(zip(TARGETS, metrics.tolist(), strict=True))
        strata = defaultdict(list)
        for idx, group in enumerate(groups):
            strata[group].append(idx)
        groups = list(strata.values())
        draws = []
        for sample in rng.integers(len(groups), size=(400, len(groups))):
            idx = np.concatenate([groups[group] for group in sample])
            estimates = recalls(idx)
            draws.append(estimates[1:] - estimates[0])
        bootstrap.append(draws)
    interval = np.quantile(np.mean(bootstrap, axis=0), [.025, .975], axis=0).T
    summary = {size: float(np.mean([row[size] for row in results.values()])) for size in TARGETS}
    return {"matched_5pct_macro": summary, "matched_5pct_by_language": results,
            "delta_95": dict(zip(TARGETS[1:], interval.tolist(), strict=True))}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the width probe requires staged Vast data")
    reference, reference_hash = read_receipt(root, "own-teacher-feature-student")
    gap, gap_hash = read_receipt(root, "own-student-fidelity-gap")
    pool_rank = np.mean([cell["raw"] for cell in gap["fidelity_sources"].values()])
    if pool_rank >= .85:
        raise ValueError("high pool fidelity calls for investigating the conversation domain first")
    cache, digests, pcm, scores, sources, roles, groups, target_hash = read_inputs(root)
    fit, fidelity = (np.flatnonzero(roles == name) for name in ("fit", "fidelity"))
    if (reference["target_report_sha256"] != target_hash or gap["fit_report_sha256"] != reference_hash
            or reference["epochs"] != EPOCHS or reference["seeds"] != list(SEEDS)
            or reference["fit_rows"] != len(fit) or reference["fidelity_rows"] != len(fidelity)):
        raise ValueError("the matched width protocol no longer matches the archived control")
    pca = root / "runs/own-teacher-feature-student" / target_hash[:12] / "pca.pt"
    if file_digest(pca) != reference["pca_sha256"]:
        raise ValueError("the archived training-only PCA changed")
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    active = tuple(source for source in SOURCES if source in sources)
    output = root / "runs/own-student-width-probe" / target_hash[:12]
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    fits = {}
    sizes = {}
    for size in TARGETS:
        for seed in SEEDS:
            name = model_name(size, seed)
            if size == TARGETS[0]:
                saved = reference["fits"][name]
                payload = torch.load(read_verified(root, saved["checkpoint"]), map_location="cpu", weights_only=True)
                fits[name] = payload["config"], payload["state"], saved
            else:
                fits[name] = fit_arm(pcm, scores, features, valid, sources, groups, fit, fidelity, active,
                                     "feature-first", seed, parameters=size)
            sizes[size] = sum(p.numel() for p in build(fits[name][0]).parameters())
    peers = {model_name(size, seed): model_name(TARGETS[0], seed) for size in TARGETS[1:] for seed in SEEDS}
    metrics, intervals = compare(root, output, fits, peers=peers)
    report = {"scope": "width-only feature-first quality diagnostic", "archived_fit_report_sha256": reference_hash,
              "fidelity_gap_report_sha256": gap_hash, "parameter_counts": sizes,
              "target_report_sha256": target_hash, "pca_sha256": reference["pca_sha256"],
              "fits": {key: row[2] for key, row in fits.items()}, "metrics": metrics, "paired_english": intervals}
    report_path = root / "reports/own-student-width-probe.json"
    _write_json(report_path, report)
    if any(row[1] is None for row in fits.values()):
        print("At least one arm failed fidelity. Training outcomes saved without a macro comparison.")
        return report
    models, thresholds = {}, {}
    for name, (config, state, _) in fits.items():
        model = build(config).to(DEVICE).eval()
        model.load_state_dict(state)
        models[name] = model
        thresholds[name] = metrics["english"][name]["threshold"]
    rows = {}
    report["language_metrics"] = score_languages(root, models, DEVICE, fixed_thresholds=thresholds,
                                                 paired_rows=rows, progress_name="own-width-multilingual")
    clips, truth, groups = zip(*pause_rows(english_rows(root)), strict=True)
    audio = np.stack(clips)
    rows["en"] = {"labels": truth, "turn_ordinals": groups,
                  "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    report["summary"] = summarize(rows)
    path = output / "paired-language-scores.json"
    _write_json(path, rows)
    report["paired_language_scores"] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    _write_json(report_path, report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(run(data_root()).get("summary"), indent=2))
