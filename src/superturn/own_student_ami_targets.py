"""Verify and cache owned-teacher scores on the canonical AMI cohorts."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr
import torch

from .ami_calibration import _write_json
from .ami_dialogue_audit import classify, dialogue_tracks
from .ami_text_signal import CROSSVAL
from .dense_teacher import file_digest
from .harper_ami_probe import FILTERED, ami_cohorts
from .language_failure_audit import read_receipt, read_verified
from .model import build
from .own_head_distillation import pool_rows, predictions, student_scores
from .own_student_real_labels import arrays
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, SEEDS
from .own_teacher_targets import teacher
from .sources import data_root
from .train import paths, records
from .wav2vec_harper_probe import extract, rates, render_ami, training_rows, verified_ami_labels


def described(rows):
    return {"pauses": len(rows), "labels": dict(Counter(row.label for row in rows)),
            "speakers": len({row.speaker for row in rows}), "groups": len({row.group for row in rows})}


def overlap(left, right):
    return {field: len({getattr(row, field) for row in left} & {row[field] for row in right})
            for field in ("speaker", "recording", "group")}


def independent_strata(root, cohorts, fit_clean):
    held = ("validation", "calibration")
    annotation, = (root / "raw/ami_annotations").glob("*.zip")
    meetings = sorted({row.recording for role in held for row in cohorts[role]})
    tracks, pairs = dialogue_tracks(root / "raw/ami/MANIFEST.json", annotation, meetings)
    cleaned = {"train": fit_clean}
    for role in held:
        labels = [classify(asdict(row), tracks[row.recording], pairs[row.recording])[0]
                  for row in cohorts[role]]
        cleaned[role] = np.flatnonzero(np.isin(labels, sorted(FILTERED)))
    return cleaned


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the AMI score cache is prepared on Vast")
    fit, cal, fit_clean, _ = ami_cohorts(root)
    cohorts = {"train": fit, "validation": verified_ami_labels(root, "validation"), "calibration": cal}
    for role, rows in cohorts.items():
        if role != "train" and any(overlap(fit, list(map(asdict, rows))).values()):
            raise ValueError(f"AMI fit and {role} share an identity, recording or group")
    _, pool_hash, members = pool_rows(root)
    if {row["source"] for _, row in members} != {"emilia_yodas", "voxpopuli"}:
        raise ValueError("AMI or HarperValley entered the existing distillation pool")
    cv, _ = read_receipt(root, f"ami-crossval/{CROSSVAL}")
    cv_folder = root / "shards/ami-crossval" / CROSSVAL
    cv_rows = [row for number in range(cv["folds"])
               for batch in records(paths(cv_folder / f"fold-{number}", "out-of-fold")) for row in batch]
    cv_intersection = overlap(fit, cv_rows)
    clean = independent_strata(root, cohorts, fit_clean)

    width, width_hash = read_receipt(root, "own-student-width-probe")
    regression, regression_hash = read_receipt(root, "own-student-hop-probe")
    if regression["width_report_sha256"] != width_hash or regression["summary"]["macro_noninferior"]:
        raise ValueError("the experiment expects the archived 10 ms control")
    selection, selection_hash = read_receipt(root, "own-teacher-language-selection")
    target_receipt, _ = read_receipt(root, "own-teacher-distillation-targets")
    if selection["provisional_default"] != "whisper-large-v3" or target_receipt["inputs"]["pool_sha256"] != pool_hash:
        raise ValueError("the original teacher or distillation inputs changed")

    materialized = {name: arrays(render_ami(root, rows)) for name, rows in cohorts.items()}
    fit_pcm = {sha256(clip.tobytes()).digest() for clip, _, _ in training_rows(root)}
    if len(fit_pcm) != 528 or any(sha256(clip.tobytes()).digest() in fit_pcm
                                 for audio, _, _ in materialized.values() for clip in audio):
        raise ValueError("the Harper-only speech-head train render did not remain separate")
    name, descriptor, predict, heads, _ = teacher(root, selection)
    clips = np.concatenate([items[0] for items in materialized.values()])
    raw = predictions(extract("ami-domain-teacher", predict, clips), heads)
    del predict
    torch.cuda.empty_cache()
    normalizers = target_receipt["normalizers"]
    stats = normalizers["by_language"].get("en", normalizers["global"])
    normalized = expit((logit(raw.astype(np.float64).clip(1e-6, 1 - 1e-6)) - stats["mean"]) / stats["std"])
    students = {}
    for seed in SEEDS:
        item = width["fits"][model_name(1_000_000, seed)]["checkpoint"]
        state = torch.load(read_verified(root, item), map_location="cpu", weights_only=True)
        model = build(state["config"]).to(DEVICE).eval()
        model.load_state_dict(state["state"])
        students[seed] = student_scores(model, clips)
    cache, evaluation = {}, {}
    start = 0
    for role, rows in cohorts.items():
        region = slice(start, start + len(rows))
        audio, labels, _ = materialized[role]
        selected = clean[role]
        values = {"teacher": raw[region], **{str(seed): values[region] for seed, values in students.items()}}
        evaluation[role] = {**described(rows), "clean_labels": dict(Counter(rows[i].label for i in selected)),
                            "render_sha256": sha256(audio.tobytes()).hexdigest(),
                            "metrics": {arm: {"all": rates(scores, labels), "clean": rates(scores[selected], labels[selected])
                                              if len(np.unique(labels[selected])) == 2 else None}
                                        for arm, scores in values.items()},
                            "student_teacher_spearman": {str(seed): float(spearmanr(scores[region], raw[region]).statistic)
                                                         for seed, scores in students.items()}}
        cache.update({f"{role}_raw": raw[region], f"{role}_normalized": normalized[region].astype(np.float32),
                      f"{role}_clean": selected, f"{role}_labels": labels})
        start = region.stop
    output = root / "runs/own-student-ami-targets"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "targets.npz"
    temporary = artifact.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez(stream, **cache)
    temporary.replace(artifact)
    report = {"scope": "AMI teacher and observed-floor diagnostic, no new model fit",
              "teacher": name, "teacher_head_sha256": descriptor["head_sha256"],
              "head_fit_corpus": "HarperValley caller training", "teacher_fit_clip_count": len(fit_pcm),
              "ranking_sha256": selection_hash, "width_sha256": width_hash, "hop_sha256": regression_hash,
              "historical_ami_oof": {"pauses": len(cv_rows), "overlap_with_fit": cv_intersection},
              "cohorts": evaluation, "cache": {"path": artifact.relative_to(root).as_posix(), "sha256": file_digest(artifact)}}
    _write_json(root / "reports/own-student-ami-targets.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
