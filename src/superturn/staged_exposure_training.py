"""Train and score the sealed six-language exposure and extra-Emilia arms."""

from collections import Counter, defaultdict
import json

import numpy as np
from scipy.special import expit, logit
import torch

from .ami_calibration import _write_json
from .balanced_development_scorecard import (
    JOB as SCORECARD, STUDENTS, UNSEEN_IN_FIT, balanced_threshold, calibration_split, cell, frozen,
    macro, summarize,
)
from .dense_teacher import SOURCES, file_digest, normalize
from .language_failure_audit import read_receipt, read_verified
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import predictions, student_scores
from .own_student_failure_refresh import agree, load_model
from .own_student_fidelity_confirmation import LOCK, NEW_SEEDS, artifact
from .own_student_wider_e20 import avg_english_three, by_turn, sample_partition
from .own_teacher_feature_student import DEVICE, fit_arm, read_inputs, targets_for_features
from .own_teacher_targets import teacher
from .sources import data_root
from .staged_language_exposure import JOB as SEAL
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import extract, pause_rows

JOB = "staged-exposure-training"
ARMS = ("exposure", "extra-emilia")
DRAWS = 500


def slot_weights(rows):
    voices = Counter((row["language"], row["source"], row["speaker"]) for row in rows)
    groups = Counter(language for language, _, _ in voices)
    if set(groups) != set(UNSEEN_IN_FIT):
        raise ValueError("the planned comparison needs a matched slot in all six languages")
    weights = np.array([
        1 / (len(groups) * groups[row["language"]] * voices[row["language"], row["source"], row["speaker"]])
        for row in rows
    ])
    return weights / weights.sum()


def load_slots(root, directory, pool_report):
    seal, seal_sha = read_receipt(root, SEAL)
    if seal["missing_languages"] or seal["base_pool_sha256"] != pool_report["inputs"]["pool_sha256"]:
        raise ValueError("the six-language match is incomplete or the base pool changed")
    pcms, members = {}, {}
    for arm in ARMS:
        folder = root / "runs" / SEAL
        record = seal["control" if arm == "extra-emilia" else arm]
        audio = folder / f"{arm}.npy"
        manifest = folder / f"{arm}.json"
        if file_digest(audio) != record["audio_sha256"] or file_digest(manifest) != record["membership_sha256"]:
            raise ValueError("supplement inputs have changed since sealing")
        pcms[arm] = np.load(audio, allow_pickle=False)
        members[arm] = json.loads(manifest.read_text())
    weights = slot_weights(members["exposure"])
    metadata = {"seal_sha256": seal_sha, "rows": len(weights), "loss_mass": .1, "batch_slots": 32}
    receipt_path = directory / "targets.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["slot_protocol"] != metadata:
            raise ValueError("the archived supplemental targets use other slots")
        targets = {arm: np.load(read_verified(root, receipt[arm]["artifact"]), allow_pickle=False) for arm in ARMS}
    else:
        selection, _ = read_receipt(root, "own-teacher-language-selection")
        name, descriptor, predict, heads, _ = teacher(root, selection)
        if descriptor["head_sha256"] != pool_report["inputs"]["teacher_head_sha256"]:
            raise ValueError("supplement teacher differs from the base teacher")
        receipt = {"slot_protocol": metadata, "teacher": name, "targets": "fit-only global logit normalization"}
        norm = pool_report["normalizers"]["global"]
        targets = {}
        for arm in ARMS:
            feats = extract(f"{JOB}-{arm}", predict, normalize(pcms[arm]))
            raw = predictions(feats, heads)
            if not np.isfinite(raw).all() or np.ptp(raw) < .05:
                raise ValueError(f"{arm} teacher targets are invalid or collapsed")
            target = expit((logit(raw.astype(np.float64).clip(1e-6, 1 - 1e-6)) -
                            norm["mean"]) / norm["std"]).astype(np.float32)
            path = directory / f"{arm}-scores.npy"
            np.save(path, target, allow_pickle=False)
            masses = defaultdict(float)
            voice_masses = defaultdict(float)
            for row, weight in zip(members[arm], weights, strict=True):
                masses[f'{row["gap_kind"]}/padded-{bool(row["missing_history_samples"])}'] += float(weight)
                voice_masses[row["source"], row["speaker"]] += float(weight)
            receipt[arm] = {"artifact": artifact(root, path), "geometry_mass": dict(masses),
                            "max_voice_or_call_mass": max(voice_masses.values()),
                            "target_quantiles": np.quantile(target, [.01, .1, .5, .9, .99]).tolist()}
            targets[arm] = target
        _write_json(receipt_path, receipt)
    return {arm: (pcms[arm], targets[arm], weights) for arm in ARMS}, artifact(root, receipt_path)


def train(root):
    control, digest = read_receipt(root, LOCK)
    directory = root / "runs" / JOB / digest[:12]
    directory.mkdir(parents=True, exist_ok=True)
    pool_report, _ = read_receipt(root, "own-teacher-distillation-targets")
    extras, extra_receipt = load_slots(root, directory, pool_report)
    cache, digests, pcm, scores, sources, roles, groups, target_hash = read_inputs(root)
    pca = root / "runs/own-teacher-feature-student" / target_hash[:12] / "pca.pt"
    if control["target_sha256"] != target_hash or control["pca_sha256"] != file_digest(pca):
        raise ValueError("the base replay inputs differ from the fixed E20 control")
    features, valid = targets_for_features(cache, digests, sources, roles, pca, reuse_pca=True)
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    active = tuple(source for source in SOURCES if source in sources)
    torch.use_deterministic_algorithms(True)
    checkpoints = {}
    for seed in NEW_SEEDS:
        for arm in ARMS:
            name = f"{arm}-{seed}"
            receipt = directory / f"{name}.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["supplement"] != extra_receipt:
                    raise ValueError("the archived arm used different targets")
                read_verified(root, saved["checkpoint"])
            else:
                path = directory / f"{name}.pt"
                fixed = {}

                def checkpoint(epoch, config, state):
                    temporary = path.with_suffix(".partial")
                    torch.save({"config": config, "state": state}, temporary)
                    temporary.replace(path)
                    fixed["checkpoint"] = artifact(root, path)

                _, _, fit_receipt = fit_arm(
                    pcm, scores, features, valid, sources, groups, fit, fidelity, active, "feature-first", seed,
                    parameters=1_000_000, epochs=20, epoch_checkpoint=checkpoint, checkpoint_epochs=(20,),
                    progress_prefix=f"{JOB}-{arm}", supplement=extras[arm],
                )
                saved = {**fixed, "supplement": extra_receipt, "history": fit_receipt["history"]}
                _write_json(receipt, saved)
            checkpoints[name] = saved["checkpoint"]
    return directory, checkpoints, extra_receipt


def paired_shift_draws(language, row, seed_draws, rng):
    truth = np.asarray(row["labels"], dtype=bool)
    groups = np.asarray(row["turn_ordinals"])
    cut = calibration_split(groups, language)
    fit_buckets, eval_buckets = (by_turn(groups[mask]) for mask in (cut, ~cut))
    scores = {name: np.asarray(values) for name, values in row["scores"].items()}
    draws = []
    for seeds in seed_draws:
        fit = sample_partition(fit_buckets, truth[cut], rng)
        test = sample_partition(eval_buckets, truth[~cut], rng)
        labels = truth[~cut][test]

        def recall(arm, idx):
            values = scores[f"{arm}-{NEW_SEEDS[idx]}"]
            threshold = balanced_threshold(values[cut][fit], truth[cut][fit])
            return (values[~cut][test][labels] >= threshold).mean()

        draws.append(np.mean([recall(ARMS[0], idx) - recall(ARMS[1], idx) for idx in seeds]))
    return draws


def summarize_arms(result):
    result["arms"] = {arm: summarize([result["models"][f"{arm}-{seed}"] for seed in NEW_SEEDS])
                      for arm in ARMS}
    return result


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this experiment requires the staged Vast data")
    directory, checkpoints, extra_receipt = train(root)
    rows, trial, teacher_receipt = frozen(root)
    peers = {name: load_model(root, checkpoint) for name, checkpoint in checkpoints.items()}
    paired = {}
    score_languages(root, peers, DEVICE, paired_rows=paired, progress_name=f"{JOB}-development")
    clips, truth, groups = zip(*pause_rows(english_rows(root)), strict=True)
    audio = np.stack(clips)
    paired["en"] = {"labels": list(truth), "turn_ordinals": list(groups),
                    "scores": {key: student_scores(model, audio).tolist() for key, model in peers.items()}}
    if paired.keys() != rows.keys():
        raise ValueError("the development cohort changed")
    for name, row in paired.items():
        agree(rows[name], list(row["labels"]), list(row["turn_ordinals"]), name)
        rows[name]["scores"].update(row["scores"])

    cells = {name: summarize_arms(cell(row, name)) for name, row in rows.items() if name != "harper-88"}
    overall = summarize_arms(macro(cells, sorted(cells)))
    six = summarize_arms(macro(cells, UNSEEN_IN_FIT))

    def delta(summary, metric):
        return summary["arms"][ARMS[0]][metric]["mean"] - summary["arms"][ARMS[1]][metric]["mean"]

    effect = delta(six, "shift_recall")
    seed_rng = np.random.default_rng(61937)
    draws = seed_rng.integers(len(NEW_SEEDS), size=(DRAWS, len(NEW_SEEDS)))
    boot = [paired_shift_draws(lang, rows[lang], draws, seed_rng) for lang in UNSEEN_IN_FIT]
    interval = np.quantile(np.mean(boot, axis=0), [.025, .975]).tolist()
    english_legacy = avg_english_three(rows["en"], [f"exposure-{seed}" for seed in NEW_SEEDS])
    english_legacy -= avg_english_three(rows["en"], STUDENTS)
    english_ba = cells["en"]["arms"]["exposure"]["balanced_accuracy"]["mean"]
    english_ba -= cells["en"]["student_seed_mean_and_range"]["balanced_accuracy"]["mean"]
    guards = {"six_language_ba_delta": delta(six, "balanced_accuracy"),
              "macro_ba_delta": delta(overall, "balanced_accuracy"),
              "english_ba_vs_base": english_ba, "english_legacy_vs_base": english_legacy}
    passed = (effect >= .03 and interval[0] > 0 and
              min(guards["six_language_ba_delta"], guards["macro_ba_delta"]) >= 0 and
              min(english_ba, english_legacy) >= -.01)
    path = directory / "paired-scores.json"
    _write_json(path, rows)
    report = {"scope": "staged-source diagnostic; standard development cohorts",
              "threshold_source": "fixed whole-turn calibration halves at 300 ms, separately for every fit",
              "primary": "equal-six shift recall B minus C", "primary_delta": effect,
              "paired_seed_calibration_turn_ci95": interval, "guards": guards, "passes": bool(passed),
              "trial_sha256": trial, "teacher_receipt_sha256": teacher_receipt, "inputs": extra_receipt,
              "macro": overall, "six_languages": six, "languages": cells,
              "checkpoints": checkpoints, "paired_scores": artifact(root, path), "pipeline_metrics": "unmeasured"}
    baseline, _ = read_receipt(root, LOCK)
    suite = {name: load_model(root, baseline["fits"][str(seed)]["epoch20_checkpoint"])
             for name, seed in zip(STUDENTS, NEW_SEEDS, strict=True)}
    clips, truth, groups = zip(*pause_rows(harper_rows(root, native_prefix=True)), strict=True)
    audio = np.stack(clips)
    harper = {"labels": list(truth), "turn_ordinals": list(groups),
              "scores": {key: student_scores(model, audio).tolist() for key, model in (suite | peers).items()}}
    thresholds = {name: cells["en"]["models"][name]["threshold"] for name in harper["scores"]}
    report["harper-88"] = summarize_arms(cell(harper, "en", thresholds))
    references, reference_sha = read_receipt(root, f"{SCORECARD}-references")
    if references["trial_sha256"] != trial:
        raise ValueError("benchmark aggregates use another baseline")
    for section in ("macro", "six_languages", "harper-88", "languages"):
        if section == "languages":
            pairs = ((value, references[section][key]) for key, value in cells.items())
        else:
            pairs = ((report[section], references["zero_supervision_macro" if section == "six_languages" else section]),)
        for value, ref in pairs:
            for key in ("eval_holds", "eval_shifts", "group_split_sha256"):
                if key in value and value[key] != ref[key]:
                    raise ValueError("the benchmark aggregates are for a different split")
            value["benchmark_aggregates"] = {key: model for key, model in ref["models"].items()
                                            if key.startswith(("smart-turn-", "livekit-"))}
    report["benchmark_scorecard_sha256"] = reference_sha
    _write_json(directory / "harper-88-scores.json", harper)
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps({"primary_delta": effect, "ci95": interval, "passed": bool(passed)}))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
