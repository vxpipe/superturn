"""Measure frozen compact student fidelity on its pool and conversation views."""

from hashlib import sha256
import json

import numpy as np
from scipy.stats import spearmanr
import torch

from .ami_calibration import _write_json
from .harper_history_probe import recall_area
from .language_failure_audit import read_receipt, read_verified
from .model import build
from .open_speech_teacher_probe import ENCODERS
from .own_head_distillation import pool_rows, student_scores
from .own_teacher_feature_student import DEVICE, read_inputs
from .own_teacher_targets import SHARD_ROWS, predictions
from .progress import Progress
from .sources import data_root
from .wav2vec_harper_probe import extract, pinned_inputs

STUDENT = "feature-first-29"
TEACHER = "whisper-large-v3"


def rank(student, teacher):
    if not len(student) == len(teacher) or not np.isfinite(student).all() or not np.isfinite(teacher).all():
        raise ValueError("the paired scores are invalid")
    value = float(spearmanr(student, teacher).statistic)
    if not np.isfinite(value):
        raise ValueError("a paired ranking is constant")
    return value


def diagnostics(scores, raw):
    n_top = max(1, (len(raw) + 9) // 10)
    reference = np.argsort(-raw, kind="stable")[:n_top]
    predicted = np.argsort(-scores, kind="stable")[:n_top]
    return {"pauses": len(scores), "raw": rank(scores, raw),
            "top_decile_overlap": len(np.intersect1d(reference, predicted)) / n_top}


def fidelity(root, model, fit):
    cache, shards, pcm, normalized, sources, roles, _, digest = read_inputs(root)
    if digest != fit["target_report_sha256"]:
        raise ValueError("the selected checkpoint used different teacher targets")
    _, _, members = pool_rows(root)
    if int((roles == "fidelity").sum()) != fit["fidelity_rows"]:
        raise ValueError("the archived fidelity membership changed")
    language_codes = np.asarray([row["language"] for _, row in members])
    rng = np.random.default_rng(29019)
    sample = []
    for source in np.unique(sources):
        for language in np.unique(language_codes[sources == source]):
            cell = (sources == source) & (language_codes == language)
            rows = np.flatnonzero(cell & (roles == "fit"))
            count = min(len(rows), int((cell & (roles == "fidelity")).sum()))
            sample.extend(rng.choice(rows, count, replace=False))
    selected = np.sort(np.r_[np.flatnonzero(roles == "fidelity"), np.asarray(sample, dtype=int)])
    raw = np.empty(len(selected), dtype=np.float32)
    shard_ordinals = selected // SHARD_ROWS
    names = np.unique(shard_ordinals)
    meter = Progress("fidelity-gap-raw-targets", len(names), "shards")
    try:
        for ordinal in names:
            name = f"{ordinal:04d}.npz"
            path = read_verified(cache, {"path": name, "sha256": shards[name]})
            mask = shard_ordinals == ordinal
            with np.load(path, allow_pickle=False) as archive:
                raw[mask] = archive["raw_scores"][selected[mask] % SHARD_ROWS]
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    scores = student_scores(model, pcm[selected, 0].astype(np.float32) / 32768)
    source_names = sources[selected]
    languages = language_codes[selected]
    selected_roles = roles[selected]
    result = {}
    for source in np.unique(source_names):
        mask = (source_names == source) & (selected_roles == "fidelity")
        normalized_rank = rank(scores[mask], normalized[selected[mask], 0])
        expected = fit["fits"][STUDENT]["selected"]["sources"][source]["by_deadline"]["300"]["spearman"]
        if abs(normalized_rank - expected) > 1e-5:
            raise ValueError(f"the selected {source} fidelity score did not reproduce")
        sampled_fit = (source_names == source) & (selected_roles == "fit")
        result[source] = {**diagnostics(scores[mask], raw[mask]), "normalized": normalized_rank,
                          "sampled_fit": diagnostics(scores[sampled_fit], raw[sampled_fit]), "by_language": {}}
        for language in np.unique(languages[mask]):
            subset = mask & (languages == language)
            if subset.sum() >= 30 and np.ptp(raw[subset]) and np.ptp(scores[subset]):
                measured = diagnostics(scores[subset], raw[subset])
                fit_subset = sampled_fit & (languages == language)
                if fit_subset.sum() >= 30 and np.ptp(raw[fit_subset]) and np.ptp(scores[fit_subset]):
                    measured["sampled_fit"] = diagnostics(scores[fit_subset], raw[fit_subset])
                result[source]["by_language"][language] = measured
    return result


def conversations(root, model, fit_hash, fit, teacher):
    receipt, receipt_hash = read_receipt(root, "own-teacher-feature-languages")
    if receipt["fit_report_sha256"] != fit_hash:
        raise ValueError("the language scores are from another fit")
    paired = teacher["multilingual_paired_rows"]
    ref = json.loads(read_verified(root, paired).read_text())
    if ref["dataset_sha256"] != receipt["dataset_sha256"]:
        raise ValueError("the models' language source digests differ")
    paths = f'runs/own-teacher-feature-student/{fit["target_report_sha256"][:12]}/language-scores.json'
    student = json.loads(read_verified(root, {"path": paths, "sha256": receipt["paired_scores_sha256"]}).read_text())
    if set(student) != set(ref["languages"]):
        raise ValueError("the language coverage differs")
    result = {}
    for language, data in student.items():
        gold = ref["languages"][language]
        if any(data[key] != gold[key] for key in ("turn_ordinals", "labels")):
            raise ValueError(f"the language pause join changed for {language}")
        result[language] = diagnostics(np.asarray(data["scores"][STUDENT]), np.asarray(gold["scores"][TEACHER]))

    clips, truth, splits, _ = pinned_inputs(root)
    if sha256(clips.tobytes()).hexdigest() != teacher["pcm_sha256"]:
        raise ValueError("the original English rendering changed")
    head_receipt = teacher["diagnostic_checkpoint"]
    state = torch.load(read_verified(root, head_receipt), map_location="cpu", weights_only=True)
    predict, provenance = ENCODERS[TEACHER](root)
    if provenance["sha256"] != state["encoder_sha256"]:
        raise ValueError("the frozen encoder changed")
    evaluation = np.concatenate([part for name, part in splits.items() if name != "train"])
    probabilities = predictions(extract("fidelity-gap-english", predict, clips)[evaluation], state["heads"])
    if sha256(probabilities.tobytes()).hexdigest() != head_receipt["evaluation_probability_sha256"]:
        raise ValueError("the English teacher score archive did not reproduce")
    for name, key in (("english", "en"), ("harper", "harper")):
        idx = splits[name]
        positions = np.searchsorted(evaluation, idx)
        if not np.array_equal(evaluation[positions], idx):
            raise ValueError("the evaluation scores do not cover the cohort")
        model_scores = student_scores(model, clips[idx])
        expected = fit["metrics"][name][STUDENT]["recalls_at_2_5_5_7_5pct"][1]
        if recall_area(model_scores, truth[idx])[1] != expected:
            raise ValueError(f"the archived {name} student recall changed")
        result[key] = diagnostics(model_scores, probabilities[positions])
    return result, receipt_hash


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the fidelity probe requires staged Vast data")
    fit, fit_hash = read_receipt(root, "own-teacher-feature-student")
    teacher, teacher_hash = read_receipt(root, "open-speech-whisper-large-v3")
    path = read_verified(root, fit["fits"][STUDENT]["checkpoint"])
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model = build(weights["config"]).to(DEVICE).eval()
    model.load_state_dict(weights["state"])
    pool = fidelity(root, model, fit)
    dev, paired_hash = conversations(root, model, fit_hash, fit, teacher)
    report = {"decision_ms": 300, "student": STUDENT, "teacher": TEACHER,
              "fit_report_sha256": fit_hash, "teacher_report_sha256": teacher_hash,
              "student_language_report_sha256": paired_hash, "fidelity_sources": pool,
              "conversation": dev, "conversation_14_language_macro": float(np.mean(
                  [stats["raw"] for name, stats in dev.items() if name != "harper"]))}
    _write_json(root / "reports/own-student-fidelity-gap.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
