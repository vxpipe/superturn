"""Continue the compact student on matched, participant-purged telephone targets."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
import torch

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .language_failure_audit import read_receipt, read_verified
from .model import build
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import predictions, student_scores
from .own_student_ami_supervision import BATCH, LR, REPLACEMENTS, STEPS, schedules, summarize, train
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, SEEDS, compare, read_inputs
from .own_teacher_targets import teacher
from .own_telephone_targets import HISTORY_BINS, render
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import extract, pause_rows, rates

ARMS = ("replay", "caller", "emilia", "shuffle", "emilia-rank")
CONTRASTS = (("caller", "replay"), ("caller", "emilia"), ("caller", "shuffle"), ("caller", "emilia-rank"))
WEIGHT = .20


def shuffled_targets(rows, targets, seed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["call"], int(np.digitize(row["edge_16khz"], HISTORY_BINS))].append(i)
    permutation = np.arange(len(targets))
    rng = np.random.default_rng(seed_for("telephone-caller-null-v1", seed))
    for items in groups.values():
        if len(items) > 1:
            order = rng.permutation(items)
            permutation[order] = np.roll(order, 1)
    if len(np.unique(permutation)) != len(targets):
        raise ValueError("the shuffled null must preserve every target value")
    shuffled = targets[permutation]
    return shuffled, {"singleton_rows": sum(len(items) == 1 for items in groups.values()),
                      "moved_rows": int(np.count_nonzero(permutation != np.arange(len(targets)))),
                      "changed_scores": int(np.count_nonzero(shuffled != targets)),
                      "mean_abs_delta": float(np.mean(abs(shuffled - targets))),
                      "spearman": float(spearmanr(shuffled, targets).statistic)}


def fixed_rates(rows, metrics):
    report = {}
    for language, row in rows.items():
        truth = np.asarray(row["labels"], dtype=bool)
        models = {}
        for name, values in row["scores"].items():
            fired = np.asarray(values) >= metrics["english"][name]["threshold"]
            models[name] = {"shift_hits": int((truth & fired).sum()), "false_completes": int((~truth & fired).sum())}
        report[language] = {"shifts": int(truth.sum()), "holds": int((~truth).sum()), "models": models}
    return report


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("telephone distillation uses staged Vast audio")
    receipt, telephone_hash = read_receipt(root, "own-telephone-targets")
    width, width_hash = read_receipt(root, "own-student-width-probe")
    _, _, pcm, archived, sources, roles, _, target_hash = read_inputs(root)
    if (width_hash != receipt["width_sha256"] or target_hash != receipt["normalization_receipt_sha256"]
            or target_hash != width["target_report_sha256"]):
        raise ValueError("the matched controls have different training provenance")
    rows = json.loads(read_verified(root, receipt["rows"]).read_text())
    caller_audio = render(root, rows)
    if sha256(caller_audio.tobytes()).hexdigest() != receipt["render_sha256"]:
        raise ValueError("the paired Harper render changed")
    with np.load(read_verified(root, receipt["scores"]), allow_pickle=False) as data:
        caller, emilia, indices = (data[key] for key in ("harper", "emilia", "emilia_rows"))
    if (not len(rows) == len(caller) == len(emilia) == len(np.unique(indices)) or
            not np.all((sources[indices] == "emilia_yodas") & (roles[indices] == "fit"))):
        raise ValueError("Emilia control must use unique fit rows only")
    quantiles = {name: np.quantile(values, [.1, .25, .5, .75, .9]).tolist()
                 for name, values in (("caller", caller), ("emilia", emilia))}
    if any(not np.isfinite(q).all() or q[-1] - q[0] < .05 for q in quantiles.values()):
        raise ValueError("a comparison target distribution collapsed")
    emilia_audio = pcm[indices, 0].astype(np.float32) / 32768
    rank_matched = np.empty_like(emilia)
    rank_matched[np.argsort(emilia, kind="stable")] = np.sort(caller)
    replay_mask = (roles == "fit") & ~np.isin(np.arange(len(roles)), indices)
    fit = [np.flatnonzero(replay_mask & (sources == name)) for name in np.unique(sources)]
    # Round-robin first over input voices, with the same exposure slots in every arm.
    owners = [SimpleNamespace(group=row["person"]) for row in rows]
    torch.use_deterministic_algorithms(True)
    fits, controls = {}, {}
    output = root / "runs/own-telephone-distillation" / telephone_hash[:12]
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        schedule = schedules(fit, owners, np.arange(len(rows)), seed)
        null, diagnostics = shuffled_targets(rows, caller, seed)
        supplemental = schedule[1]["all"]
        controls[seed] = {**diagnostics, "replay_sha256": sha256(schedule[0].tobytes()).hexdigest(),
                          "supplement_sha256": sha256(supplemental.tobytes()).hexdigest(),
                          "slot_mean_target": {label: float(values[supplemental].mean()) for label, values in
                                               (("caller", caller), ("shuffle", null), ("emilia", emilia),
                                                ("emilia-rank", rank_matched))}}
        parent = width["fits"][model_name(1_000_000, seed)]["checkpoint"]
        arms = {"replay": (caller_audio, caller), "caller": (caller_audio, caller),
                "emilia": (emilia_audio, emilia), "shuffle": (caller_audio, null),
                "emilia-rank": (emilia_audio, rank_matched)}
        for name, data in arms.items():
            fits[f"{name}-{seed}"] = train(root, parent, name, seed, (pcm, archived), data, schedule,
                                          supplement_weight=WEIGHT, progress_name="telephone-distill")
    metrics, _ = compare(root, output, fits, peers={})
    models = {}
    for name, (config, state, _) in fits.items():
        models[name] = build(config).to(DEVICE).eval()
        models[name].load_state_dict(state)
    report = {"scope": "matched phone soft-target continuation, developmental evaluation",
              "target_report_sha256": telephone_hash, "seeds": SEEDS, "steps": STEPS, "batch": BATCH,
              "learning_rate": LR, "replacement_slots": REPLACEMENTS, "supplement_loss": WEIGHT,
              "normalized_quantiles": quantiles, "controls": controls,
              "fits": {name: value[2] for name, value in fits.items()}, "metrics": metrics}
    path = root / "reports/own-telephone-distillation.json"
    _write_json(path, report)
    rows_by_language = {}
    score_languages(root, models, DEVICE, paired_rows=rows_by_language, progress_name="telephone-distill-multilingual")
    clips, truth, groups = zip(*pause_rows(english_rows(root)), strict=True)
    audio = np.stack(clips)
    rows_by_language["en"] = {"labels": truth, "turn_ordinals": groups,
                              "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    report["summary"] = summarize(rows_by_language, arm_names=ARMS, contrasts=CONTRASTS)
    report["ami_locked_counts"] = fixed_rates(rows_by_language, metrics)
    pairs = output / "paired-language-scores.json"
    _write_json(pairs, rows_by_language)
    report["paired_language_scores"] = {"path": pairs.relative_to(root).as_posix(), "sha256": file_digest(pairs)}
    _write_json(path, report)
    return report


def held_fidelity(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the teacher fidelity replay uses staged Vast audio")
    candidate, candidate_hash = read_receipt(root, "own-telephone-distillation")
    targets, target_hash = read_receipt(root, "own-telephone-targets")
    selection, selection_hash = read_receipt(root, "own-teacher-language-selection")
    if (candidate["target_report_sha256"] != target_hash or targets["selection_sha256"] != selection_hash):
        raise ValueError("the held teacher selection changed")
    _, _, encoder, heads, _ = teacher(root, selection)
    clips, truth, _ = zip(*pause_rows(harper_rows(root, native_prefix=True)), strict=True)
    audio, truth = np.stack(clips), np.asarray(truth)
    values = predictions(extract("telephone-fidelity-held", encoder, audio), heads)
    del encoder
    if rates(values, truth) != targets["original_teacher_harper"]:
        raise ValueError("the archived Harper teacher metrics changed")
    results = {}
    for name, fit in candidate["fits"].items():
        state = torch.load(read_verified(root, fit["checkpoint"]), map_location="cpu", weights_only=True)
        model = build(state["config"]).to(DEVICE).eval()
        model.load_state_dict(state["state"])
        scores = student_scores(model, audio)
        if rates(scores, truth)["recalls_at_2_5_5_7_5pct"] != candidate["metrics"]["harper"][name]["recalls_at_2_5_5_7_5pct"]:
            raise ValueError("the archived student ranking did not reproduce")
        results[name] = float(spearmanr(scores, values).statistic)
    report = {"scope": "Harper validation fidelity to the original learned-layer teacher",
              "training_report_sha256": candidate_hash, "pauses": len(truth),
              "parent_fidelity": targets["width_harper_spearman"], "student_fidelity": results}
    _write_json(root / "reports/own-telephone-held-fidelity.json", report)
    print(json.dumps(report, indent=2))


def fit_fidelity(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the teacher fidelity replay uses staged Vast audio")
    candidate, candidate_hash = read_receipt(root, "own-telephone-distillation")
    targets, target_hash = read_receipt(root, "own-telephone-targets")
    if candidate["target_report_sha256"] != target_hash:
        raise ValueError("the telephone target report changed")
    rows = json.loads(read_verified(root, targets["rows"]).read_text())
    audio = render(root, rows)
    if sha256(audio.tobytes()).hexdigest() != targets["render_sha256"]:
        raise ValueError("the caller training render changed")
    with np.load(read_verified(root, targets["scores"]), allow_pickle=False) as stored:
        values = stored["harper"]
    results = {}
    for name, fit in candidate["fits"].items():
        state = torch.load(read_verified(root, fit["checkpoint"]), map_location="cpu", weights_only=True)
        model = build(state["config"]).to(DEVICE).eval()
        model.load_state_dict(state["state"])
        scores = student_scores(model, audio)
        results[name] = {"spearman": float(spearmanr(scores, values).statistic),
                         "mean_abs_error": float(np.mean(abs(scores - values)))}
    report = {"scope": "training fidelity to the participant-purged caller targets",
              "training_report_sha256": candidate_hash, "pauses": len(values), "models": results}
    _write_json(root / "reports/own-telephone-fit-fidelity.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--held-fidelity", action="store_true")
    mode.add_argument("--fit-fidelity", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.held_fidelity:
        held_fidelity(data_root())
    elif args.fit_fidelity:
        fit_fidelity(data_root())
    else:
        print(json.dumps(run(data_root())["summary"], indent=2))
