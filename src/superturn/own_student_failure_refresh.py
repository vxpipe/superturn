"""Refresh multilingual failure strata for width and matched continuation experiments."""

import argparse
from hashlib import sha256
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from .ami_calibration import _write_json
from .augment import condition
from .dense_teacher import file_digest
from .eot_domain_probe import DATASET_SHA256
from .failure_strata import cutoff_at_budget, failure_comparison, paired_macro
from .harper_history_probe import recall_area
from .language_failure_audit import EDGE, TEACHER, read_receipt, read_verified, scan, summarize
from .model import build
from .multilingual_teacher_probe import DATA
from .open_speech_teacher_probe import ENCODERS
from .own_head_distillation import pool_rows, student_scores
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, SEEDS
from .own_teacher_targets import predictions
from .progress import Progress
from .sources import data_root
from .train import _threshold
from .wav2vec_harper_probe import extract, pinned_inputs

EXPERIMENTS = {
    "width": ("own-student-width-probe", {"145k": "feature-first", "593k": "feature-first-600000",
                                        "973k": "feature-first-1000000"}),
    "hop": ("own-student-hop-probe", {"hop20": "hop20"}),
    "ami": ("own-student-ami-supervision", {"ami-replay": "replay", "ami-teacher": "teacher",
                                           "ami-observed": "observed", "ami-teacher-clean": "teacher-clean",
                                           "ami-observed-clean": "observed-clean"}),
    "telephone": ("own-telephone-distillation", {"telephone-replay": "replay", "telephone-caller": "caller",
                                                  "telephone-emilia": "emilia", "telephone-shuffle": "shuffle",
                                                  "telephone-rank-emilia": "emilia-rank"}),
    "ranking": ("own-student-tail-ranking", {"tail-pointwise": "pointwise", "tail-pairwise": "pairwise"}),
}
COMPARISONS = {
    "width_593k_vs_145k": ("593k", "145k"), "width_973k_vs_145k": ("973k", "145k"),
    "hop20_vs_10": ("hop20", "973k"), "ami_teacher_vs_replay": ("ami-teacher", "ami-replay"),
    "ami_observed_vs_teacher": ("ami-observed", "ami-teacher"),
    "ami_clean_observed_vs_teacher": ("ami-observed-clean", "ami-teacher-clean"),
    "telephone_caller_vs_replay": ("telephone-caller", "telephone-replay"),
    "telephone_caller_vs_emilia": ("telephone-caller", "telephone-emilia"),
    "telephone_caller_vs_shuffle": ("telephone-caller", "telephone-shuffle"),
    "telephone_caller_vs_rank_matched": ("telephone-caller", "telephone-rank-emilia"),
    "tail_pairwise_vs_pointwise": ("tail-pairwise", "tail-pointwise"),
    "soup_vs_common_parent": ("soup", f"parent-{SEEDS[0]}"), "soup_vs_sibling_mean": ("soup", "sibling-mean"),
    "noise_973k_vs_original": ("973k-noise", "973k"), "noise_145k_vs_original": ("145k-noise", "145k"),
}


class ProbabilityMean:
    def __init__(self, models):
        self.models = models

    def __call__(self, waveform, silence):
        return torch.stack([model(waveform, silence) for model in self.models]).mean(0)


def load_model(root, receipt):
    payload = torch.load(read_verified(root, receipt), map_location="cpu", weights_only=True)
    model = build(payload["config"]).to(DEVICE).eval()
    model.load_state_dict(payload["state"])
    return model


def english_reference(root, report):
    clips, truth, splits, groups = pinned_inputs(root)
    if sha256(clips.tobytes()).hexdigest() != report["pcm_sha256"]:
        raise ValueError("English teacher input PCM changed")
    head_ref = report["diagnostic_checkpoint"]
    state = torch.load(read_verified(root, head_ref), map_location="cpu", weights_only=True)
    predict, provenance = ENCODERS[TEACHER](root)
    if provenance["sha256"] != state["encoder_sha256"]:
        raise ValueError("the owned Whisper encoder changed")
    evaluation = np.concatenate([part for key, part in splits.items() if key != "train"])
    probabilities = predictions(extract("failure-refresh-english-teacher", predict, clips)[evaluation], state["heads"])
    if sha256(probabilities.tobytes()).hexdigest() != head_ref["evaluation_probability_sha256"]:
        raise ValueError("English owned teacher scores did not reproduce")
    positions = np.searchsorted(evaluation, splits["english"])
    if not np.array_equal(evaluation[positions], splits["english"]):
        raise ValueError("English scores do not cover the pinned pauses")
    english = {"labels": truth[splits["english"]].tolist(), "turn_ordinals": list(groups["english"]),
               "scores": {TEACHER: probabilities[positions].tolist()}}
    cal = splits["calibration"]
    return english, clips[cal], truth[cal]


def read_archives(root):
    leases, tables, digests = {}, {}, {}
    for name, (path, aliases) in EXPERIMENTS.items():
        report, digests[name] = read_receipt(root, path)
        leases[name] = report
        archive = json.loads(read_verified(root, report["paired_language_scores"]).read_text())
        tables[name] = archive, aliases
    return leases, tables, digests


def verify_ancestry(root, leases, digests, soup):
    telephone_targets, telephone_digest = read_receipt(root, "own-telephone-targets")
    parent = leases["width"]["fits"][model_name(1_000_000, SEEDS[0])]["checkpoint"]
    if (soup["ranking_sha256"] != digests["ranking"] or soup["common_parent"] != parent
            or leases["ranking"]["width_sha256"] != digests["width"]
            or leases["hop"]["width_report_sha256"] != digests["width"]
            or leases["ami"]["width_report_sha256"] != digests["width"]
            or leases["telephone"]["target_report_sha256"] != telephone_digest
            or telephone_targets["width_sha256"] != digests["width"]):
        raise ValueError("the matched continuation receipts differ from their width parent")


def agree(row, labels, turns, name):
    if row["labels"] != labels or row["turn_ordinals"] != turns:
        raise ValueError(f"pause order or labels differ: {name}")


def episode_scores(language, tables, reference):
    labels, turns = reference["labels"], reference["turn_ordinals"]
    result = {TEACHER: reference["scores"][TEACHER]}
    for experiment, (data, aliases) in tables.items():
        row = data[language]
        agree(row, labels, turns, f"{experiment}/{language}")
        for alias, prefix in aliases.items():
            values = [row["scores"][f"{prefix}-{seed}"] for seed in SEEDS]
            result[alias] = np.mean(values, axis=0).tolist()
            result.update({f"{alias}-{seed}": score for seed, score in zip(SEEDS, values, strict=True)})
            if alias == "973k":
                result.update({f"parent-{seed}": score for seed, score in zip(SEEDS, values, strict=True)})
    return result


def summary_guard(scores, truth):
    truth = np.asarray(truth, dtype=bool)

    def score(name, *, matched_5=False):
        values = recall_area(np.asarray(scores[name]), truth)
        return float(values[1] if matched_5 else np.mean(values[:3]))

    ranked = {arm: [score(f"{arm}-{seed}") for seed in SEEDS] for arm in ("973k", "tail-pointwise", "tail-pairwise")}
    retained = [score(f"sibling-{seed}") for seed in SEEDS]
    soup, parent = score("soup"), score(f"parent-{SEEDS[0]}")
    siblings = float(np.mean(retained))
    controls = {}
    for name, (arm, control) in COMPARISONS.items():
        if name.startswith("noise_"):
            continue
        arms = [[f"{label}-{seed}" for seed in SEEDS] if f"{label}-{SEEDS[0]}" in scores else [label]
                for label in (arm, control)]
        left, right = [[score(label) for label in names] for names in arms]
        delta = float(np.mean(left) - np.mean(right))
        left_5, right_5 = [[score(label, matched_5=True) for label in names] for names in arms]
        readouts = [np.asarray([recall_area(np.asarray(scores[label]), truth)[:3] for label in names])
                    for names in arms]
        controls[name] = {"delta": delta, "passes_minus_one_point_guard": delta >= -.01,
                          "primary_matched_5_delta": float(np.mean(left_5) - np.mean(right_5)),
                          "candidate_seed_means_at_2_5_5_7_5": readouts[0].mean(0).tolist(),
                          "candidate_seed_ranges_at_2_5_5_7_5":
                              np.stack([readouts[0].min(0), readouts[0].max(0)], axis=1).tolist(),
                          "control_seed_means_at_2_5_5_7_5": readouts[1].mean(0).tolist(),
                          "control_seed_ranges_at_2_5_5_7_5":
                              np.stack([readouts[1].min(0), readouts[1].max(0)], axis=1).tolist(),
                          "candidate_mean_and_seed_range": [float(np.mean(left)), min(left), max(left)],
                          "control_mean_and_seed_range": [float(np.mean(right)), min(right), max(right)]}
    return {"english_973k_independent_seed_values": ranked["973k"],
            "english_973k_probability_mean": score("973k"),
            "english_973k_probability_mean_gain": score("973k") - float(np.mean(ranked["973k"])),
            "english_tail_seed_means_and_ranges": {key: [float(np.mean(values)), min(values), max(values)]
                                                   for key, values in ranked.items()},
            "english_tail_guard": float(np.mean(ranked["tail-pairwise"]) - np.mean(ranked["tail-pointwise"])) >= -.01,
            "english_soup": soup, "english_common_parent": parent,
            "english_sibling_seed_values": retained, "english_sibling_probability_mean": score("sibling-mean"),
            "english_soup_guard": min(soup - parent, soup - siblings) >= -.01,
            "english_experiment_guards": controls}


def paired_noise_counts(scores, truth, features, language):
    y = np.asarray(truth, dtype=bool)
    short = np.asarray(features["history"]) != "full"
    result = {}
    for name in ("145k", "973k", "soup"):
        clean, noisy = (np.asarray(scores[key]) for key in (name, f"{name}-noise"))
        if not np.array_equal(clean[~short], noisy[~short]):
            raise ValueError(f"comfort noise changed full history: {language}/{name}")
        cutoff = cutoff_at_budget(clean, y)
        result[name] = {
            "original": summarize(clean >= cutoff, y, short),
            "noise": summarize(noisy >= cutoff, y, short),
            "changed_short_decisions": int((short & ((noisy >= cutoff) != (clean >= cutoff))).sum()),
        }
    return result


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the failure refresh requires the pinned Vast environment")
    leases, tables, digests = read_archives(root)
    soup, soup_digest = read_receipt(root, "own-student-weight-soup")
    verify_ancestry(root, leases, digests, soup)
    width = leases["width"]
    models = {}
    for size, name in ((150000, "145k"), (1000000, "973k")):
        models[name] = ProbabilityMean([load_model(root, width["fits"][model_name(size, seed)]["checkpoint"])
                                        for seed in SEEDS])
    for seed in SEEDS:
        models[f"sibling-{seed}"] = load_model(root, soup["siblings"][str(seed)]["checkpoint"])
    models["soup"] = load_model(root, soup["soup"])
    owned, owned_digest = read_receipt(root, f"open-speech-{TEACHER}")
    gold = json.loads(read_verified(root, owned["multilingual_paired_rows"]).read_text())
    if gold["dataset_sha256"] != DATA:
        raise ValueError("owned teacher and diagnostic sources differ")
    reference = gold["languages"]
    reference["en"], cal_pcm, cal_truth = english_reference(root, owned)
    cal = {name: student_scores(model, cal_pcm) for name, model in models.items()}
    cal["sibling-mean"] = np.mean([cal[f"sibling-{seed}"] for seed in SEEDS], axis=0)
    thresholds = {name: _threshold(scores.tolist(), cal_truth.tolist()) for name, scores in cal.items()}
    thresholds[TEACHER] = owned["metrics"][TEACHER]["english"]["ami_threshold"]
    old, old_digest = read_receipt(root, "language-failure-audit")
    paths = {lang: f"raw/livekit_multilingual_diagnostic/data/{lang}/validation-00000-of-00001.parquet" for lang in DATA}
    paths["en"] = "raw/livekit_eot_diagnostic/data/en/validation-00000-of-00002.parquet"
    hashes = {**DATA, "en": DATASET_SHA256}
    readers = {lang: pq.ParquetFile(read_verified(root, {"path": path, "sha256": hashes[lang]}))
               for lang, path in paths.items()}
    rng = np.random.default_rng(271)
    noise = condition(np.round(rng.normal(0, 110, EDGE)).astype(np.int16), "telephone", 0).astype(np.float32)
    noise *= 10 ** (-55 / 20) / np.sqrt(np.mean(noise ** 2))
    output, pair_rows = {}, {}
    meter = Progress("owned-student-failure-refresh", sum(source.metadata.num_rows for source in readers.values()), "turns")
    try:
        for language, source in readers.items():
            ref = reference[language]
            archived = episode_scores(language, tables, ref)
            turns, truth, features, evaluated, padded, _, membership = scan(
                source, language, models, noise, meter, original_scores=archived, lexical_tones=False)
            agree(ref, truth, turns, f"redecoded/{language}")
            if language != "en" and membership != old["languages"][language]["membership_sha256"]:
                raise ValueError(f"causal failure-audit row membership changed: {language}")
            archived.update({key: np.asarray(values).tolist() for key, values in evaluated.items()})
            archived["sibling-mean"] = np.mean([archived[f"sibling-{seed}"] for seed in SEEDS], axis=0).tolist()
            for name in ("145k", "973k", "soup"):
                archived[f"{name}-noise"] = np.asarray(padded[name]).tolist()
            result = failure_comparison(archived, TEACHER, truth, turns, features, COMPARISONS)
            y = np.asarray(truth, dtype=bool)
            result["noise_at_original_cutoff"] = paired_noise_counts(archived, truth, features, language)
            result["ami_fixed"] = {name: summarize(np.asarray(archived[name]) >= value, y, np.ones(len(y), dtype=bool))
                                   for name, value in thresholds.items()}
            result["matched_recall_and_pauc"] = {name: recall_area(np.asarray(values), y).tolist()
                                                 for name, values in archived.items()}
            result["membership_sha256"] = membership
            output[language] = result
            pair_rows[language] = {"labels": truth, "turn_ordinals": turns, "features": dict(features), "scores": archived}
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    _, _, pool = pool_rows(root)
    trained = {row["language"] for _, row in pool if row["role"] == "fit"}
    absent = sorted(set(output) - trained)
    means = {arm: np.mean([row["matched_recall_and_pauc"][arm] for row in output.values()], axis=0).tolist()
             for arm in ("145k", "973k", TEACHER, "tail-pointwise", "tail-pairwise", "soup", "sibling-mean")}
    missing = {arm: np.mean([output[language]["matched_recall_and_pauc"][arm] for language in absent], axis=0).tolist()
               for arm in means}
    guard = summary_guard(pair_rows["en"]["scores"], pair_rows["en"]["labels"])
    directory = root / "runs/own-student-weight-soup" / digests["ranking"][:12]
    rows_path = directory / "failure-strata-rows.json"
    _write_json(rows_path, pair_rows)
    report = {"scope": "development diagnostics at 300 ms; paired turn intervals, no persistent speaker IDs",
              "matched_metric_order": ["recall at 2.5%", "recall at 5%", "recall at 7.5%", "normalized pAUC to 10%"],
              "source_sha256": hashes, "experiment_receipts": digests, "soup_receipt_sha256": soup_digest,
              "teacher_report_sha256": owned_digest, "prior_audit_receipt_sha256": old_digest,
              "absent_fit_languages": absent, "absent_fit_macro": missing, "macro": means, "english_guard": guard,
              "noise_comparison": "synthetic leading padding on student only, matched cutoff refitted per view",
              "languages": output, "paired_rows": {"path": rows_path.relative_to(root).as_posix(),
                                                   "sha256": file_digest(rows_path)}}
    _write_json(root / "reports/own-student-failure-refresh.json", report)
    print(json.dumps({"macro": means, "english_guard": guard, "absent_fit": absent}, indent=2))


def summarize_archived(root):
    report, digest = read_receipt(root, "own-student-failure-refresh")
    leases, _, digests = read_archives(root)
    soup, soup_digest = read_receipt(root, "own-student-weight-soup")
    verify_ancestry(root, leases, digests, soup)
    if report["experiment_receipts"] != digests or report["soup_receipt_sha256"] != soup_digest:
        raise ValueError("the archived refresh used other continuation receipts")
    rows = json.loads(read_verified(root, report["paired_rows"]).read_text())
    for language, row in rows.items():
        if len(row["turn_ordinals"]) != len(row["labels"]) or set(row["labels"]) != {0, 1}:
            raise ValueError(f"invalid paired failure cohort: {language}")
    absent = {lang: rows[lang] for lang in report["absent_fit_languages"]}
    table = {"failure_refresh_sha256": digest, "effect": "absolute matched-5 recall differences",
             "per_language_model": "probability mean of seeds or an explicit single checkpoint",
             "all_development_languages": paired_macro(rows, COMPARISONS),
             "languages_without_fit_rows": paired_macro(absent, COMPARISONS),
             "noise_at_original_cutoff": {lang: paired_noise_counts(row["scores"], row["labels"], row["features"], lang)
                                          for lang, row in rows.items()},
             "english_guard": summary_guard(rows["en"]["scores"], rows["en"]["labels"])}
    _write_json(root / "reports/own-student-failure-deltas.json", table)
    print(json.dumps(table["all_development_languages"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-archived", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.summarize_archived:
        summarize_archived(data_root())
    else:
        run(data_root())
