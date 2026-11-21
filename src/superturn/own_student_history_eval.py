"""Compare frozen compact history experiments on the same development pauses."""

from collections import Counter, defaultdict
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from .ami_calibration import _write_json
from .dense_teacher import SAMPLES, file_digest
from .eot_domain_probe import decision_audio
from .harper_history_probe import recall_area
from .language_failure_audit import EDGE, pitch_pair, scan, summarize
from .model import build
from .multilingual_teacher_probe import DATA
from .own_head_distillation import predictions, student_scores
from .own_student_active_stats import comfort_noise
from .own_teacher_feature_student import SEEDS
from .own_teacher_targets import teacher
from .progress import Progress
from .sources import data_root
from .turn1_mini_probe import english_rows, harper_rows
from .vxpipe_policy_probe import operating_points

ARMS = {
    "original": "own-teacher-feature-student",
    "active": "own-student-active-stats",
    "history": "own-student-history-views",
}
TEACHER = "teacher"


class FrozenTeacher:
    def __init__(self, predict, heads):
        self.predict, self.heads = predict, heads

    def __call__(self, audio, elapsed):
        scores = predictions(self.predict(audio).cpu().numpy(), self.heads)
        return torch.from_numpy(np.asarray(scores, dtype=np.float32))


def load(root):
    models, reports = {}, {}
    for arm, source in ARMS.items():
        path = root / "reports" / f"{source}.json"
        reports[arm] = file_digest(path)
        fit = json.loads(path.read_text())["fits"]
        for seed in SEEDS:
            item = fit[f"feature-first-{seed}"]["checkpoint"]
            weight = root / item["path"]
            if file_digest(weight) != item["sha256"]:
                raise ValueError(f"checkpoint changed for {arm}-{seed}")
            saved = torch.load(weight, weights_only=True, map_location="cpu")
            if bool(saved["config"].get("active_stats", False)) != (arm != "original"):
                raise ValueError(f"unexpected frontend for {arm}-{seed}")
            model = build(saved["config"])
            model.load_state_dict(saved["state"])
            models[f"{arm}-{seed}"] = model.cuda().eval()
    selection = root / "reports/own-teacher-language-selection.json"
    name, _, predict, heads, _ = teacher(root, json.loads(selection.read_text()))
    if name != "whisper-large-v3":
        raise ValueError("the original matched-view archive is for our Whisper teacher")
    models[TEACHER] = FrozenTeacher(predict, heads)
    reports[TEACHER] = file_digest(selection)
    return models, reports


def independent_view(rows, models, noise, name):
    truth, groups, clips, history = [], [], [], []
    features = defaultdict(list)
    for decoded, edge, label, group in rows:
        rendered = decision_audio(decoded, edge, "telephone")[-SAMPLES:]
        count = min(EDGE, edge)
        source_pitch, _, _ = pitch_pair(decoded[edge - count:edge].astype(np.float32) / 32768, rendered)
        truth.append(label)
        groups.append(group)
        clips.append(rendered)
        history.append(EDGE - count)
        features["native_f0"].append(source_pitch)
        features["history"].append("full" if count == EDGE else "short")
    meter = Progress(f"own-student-history-eval-{name}", len(models), "models")
    try:
        original = np.stack(clips)
        padding = np.asarray(history) > 0
        changed = original[padding]
        for clip, prefix in zip(changed, np.asarray(history)[padding], strict=True):
            clip[:prefix] = noise[:prefix]
        raw, noisy = {}, {}
        for model_name, model in models.items():
            raw[model_name] = student_scores(model, original)
            noisy[model_name] = raw[model_name].copy()
            if padding.any():
                noisy[model_name][padding] = student_scores(model, changed)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return groups, truth, features, raw, noisy


def view_stats(scores, truth, features, groups, *, turns=True):
    arrays = {name: np.asarray(values) for name, values in scores.items()}
    y = np.asarray(truth, dtype=bool)
    matched, metrics = {}, {}
    for name, values in arrays.items():
        if len(values) != len(y) or not np.isfinite(values).all():
            raise ValueError(f"invalid scores for {name}")
        _, _, cutoffs, _, false = operating_points(values, y)
        matched[name] = values >= cutoffs[false <= .05 * (~y).sum()][-1]
        metrics[name] = recall_area(values, y).tolist()
    features = dict(features)
    if turns:
        counts = Counter(groups)
        features["qualifying_pauses_in_turn"] = ["one" if counts[g] == 1 else "multiple" for g in groups]
    strata = {}
    for feature in ("history", "native_f0", "position", "qualifying_pauses_in_turn"):
        if feature not in features:
            continue
        cells = np.asarray(features[feature])
        strata[feature] = {}
        for cell in sorted(set(cells)):
            subset = cells == cell
            result = {}
            for name, complete in matched.items():
                row = summarize(complete, y, subset)
                row["teacher_only_shifts"] = int((subset & y & matched[TEACHER] & ~complete).sum())
                row["both_missed_shifts"] = int((subset & y & ~matched[TEACHER] & ~complete).sum())
                result[name] = row
            strata[feature][cell] = result
    return {"metrics": metrics, "strata": strata}, arrays, y


def arm_means(metrics, cutoff=slice(0, 3)):
    return {arm: float(np.mean([metrics[f"{arm}-{seed}"][cutoff] for seed in SEEDS])) for arm in ARMS}


def turn_bootstrap(arrays, truth, groups, rng):
    group_labels = np.asarray(groups)
    buckets = [np.flatnonzero(group_labels == g) for g in np.unique(group_labels)]
    values = []
    for draw in rng.integers(len(buckets), size=(300, len(buckets))):
        index = np.concatenate([buckets[i] for i in draw])
        labels = truth[index]
        metrics = {key: recall_area(array[index], labels)
                   for key, array in arrays.items() if key != TEACHER}
        deltas = []
        for means in (arm_means(metrics), arm_means(metrics, 1)):
            deltas.append([means["active"] - means["original"], means["history"] - means["original"],
                           means["history"] - means["active"]])
        values.append(deltas)
    return np.asarray(values)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this comparison requires staged Vast data")
    models, inputs = load(root)
    noise = comfort_noise()
    sources = {}
    for language, digest in DATA.items():
        path = root / f"raw/livekit_multilingual_diagnostic/data/{language}/validation-00000-of-00001.parquet"
        if file_digest(path) != digest:
            raise ValueError(f"language source changed for {language}")
        sources[language] = pq.ParquetFile(path)
    archive = root / "reports/language-failure-audit.json"
    memberships = json.loads(archive.read_text())["languages"]
    teacher_report = json.loads((root / "reports/open-speech-whisper-large-v3.json").read_text())
    pairs = teacher_report["multilingual_paired_rows"]
    pair_path = root / pairs["path"]
    pair_sha = file_digest(pair_path)
    if pair_sha != pairs["sha256"]:
        raise ValueError("the archived original teacher probabilities changed")
    paired = json.loads(pair_path.read_text())
    if paired["dataset_sha256"] != DATA:
        raise ValueError("the teacher score source clips differ")
    inputs["original_teacher_language_scores"] = pair_sha
    results, bootstrap = {}, []
    rng = np.random.default_rng(29017)

    def record(tag, groups, truth, features, raw, noisy):
        result, _, _ = view_stats(raw, truth, features, groups, turns=tag != "harper")
        result_noisy, arrays, y = view_stats(noisy, truth, features, groups, turns=tag != "harper")
        results[tag] = {"original": result, "noise_prefix": result_noisy}
        if tag != "harper":
            bootstrap.append(turn_bootstrap(arrays, y, groups, rng))

    meter = Progress("own-student-history-languages", sum(p.metadata.num_rows for p in sources.values()), "turns")
    try:
        for language, source in sources.items():
            archived_teacher = paired["languages"][language]
            scores = {TEACHER: archived_teacher["scores"]["whisper-large-v3"]}
            groups, y, features, raw, noisy, _, membership = scan(
                source, language, models, noise, meter, original_scores=scores)
            if membership != memberships[language]["membership_sha256"]:
                raise ValueError(f"development membership changed for {language}")
            if groups != archived_teacher["turn_ordinals"] or y != archived_teacher["labels"]:
                raise ValueError(f"archived original teacher order changed for {language}")
            record(language, groups, y, features, raw, noisy)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    record("en", *independent_view(english_rows(root), models, noise, "english"))
    record("harper", *independent_view(harper_rows(root, native_prefix=True), models, noise, "harper"))
    macro, macro_at_5 = defaultdict(list), defaultdict(list)
    for tag, result in results.items():
        if tag == "harper":
            continue
        metrics = result["noise_prefix"]["metrics"]
        for arm, value in arm_means(metrics).items():
            macro[arm].append(value)
        for arm, value in arm_means(metrics, 1).items():
            macro_at_5[arm].append(value)
        macro[TEACHER].append(float(np.mean(metrics[TEACHER][:3])))
        macro_at_5[TEACHER].append(metrics[TEACHER][1])
    macro = {key: float(np.mean(values)) for key, values in macro.items()}
    macro_at_5 = {key: float(np.mean(values)) for key, values in macro_at_5.items()}
    ci, ci_at_5 = np.quantile(np.mean(bootstrap, axis=0), [.025, .975], axis=0).transpose(1, 2, 0).tolist()
    names = ("active-original", "history-original", "history-active")
    baseline_gap = macro[TEACHER] - macro["original"]
    summary = {"macro_mean_recall_noise_prefix": macro,
               "macro_recall_noise_prefix_at_5pct": macro_at_5,
               "paired_turn_bootstrap_95": dict(zip(names, ci, strict=True)),
               "paired_turn_bootstrap_at_5pct_95": dict(zip(names, ci_at_5, strict=True)),
               "teacher_gap_fraction_closed": {arm: (macro[arm] - macro["original"]) / baseline_gap
                                               if baseline_gap > 0 else None for arm in ("active", "history")}}
    single = defaultdict(list)
    language_gains = defaultdict(dict)
    for tag, item in results.items():
        if tag == "harper":
            continue
        part = item["noise_prefix"]
        means = arm_means(part["metrics"])
        for arm in ("active", "history"):
            language_gains[arm][tag] = means[arm] - means["original"]
        subset = part["strata"]["qualifying_pauses_in_turn"]["one"]
        for arm in ARMS:
            single[arm].append(float(np.mean([subset[f"{arm}-{seed}"]["shift_hits"] /
                                              subset[f"{arm}-{seed}"]["shifts"] for seed in SEEDS])))
        single[TEACHER].append(subset[TEACHER]["shift_hits"] / subset[TEACHER]["shifts"])
    single = {key: float(np.mean(values)) for key, values in single.items()}
    summary["one_pause_turn_matched_5pct_macro"] = single
    summary["noise_prefix_language_deltas"] = dict(language_gains)
    harper = arm_means(results["harper"]["original"]["metrics"])
    single_gap = single[TEACHER] - single["original"]
    whole_gap_at_5 = macro_at_5[TEACHER] - macro_at_5["original"]
    summary["diagnostic_gate"] = {}
    for arm in ("active", "history"):
        gap_closed = summary["teacher_gap_fraction_closed"][arm]
        single_closed = (single[arm] - single["original"]) / single_gap if single_gap > 0 else None
        closed_at_5 = ((macro_at_5[arm] - macro_at_5["original"]) / whole_gap_at_5
                       if whole_gap_at_5 > 0 else None)
        summary["diagnostic_gate"][arm] = {
            "macro_gain_at_least_3pp": macro[arm] - macro["original"] >= .03,
            "macro_gain_lower_bound_positive": summary["paired_turn_bootstrap_95"][f"{arm}-original"][0] > 0,
            "teacher_gap_closed_at_least_20pct": gap_closed is not None and gap_closed >= .20,
            "single_pause_gap_closed_at_least_as_much": single_closed is not None and closed_at_5 is not None
                                                      and single_closed >= closed_at_5,
            "no_language_loss_over_3pp": min(language_gains[arm].values()) >= -.03,
            "harper_loss_at_most_2pp": harper[arm] - harper["original"] >= -.02,
        }
    report = {"scope": "retrospective ranking, no freeze or model promotion", "metrics_order": [
                  "recall_at_2.5pct", "recall_at_5pct", "recall_at_7.5pct", "normalized_pauc_to_10pct"],
              "inputs": {**inputs, "membership_report": file_digest(archive)}, "summary": summary,
              "views": results}
    _write_json(root / "reports/own-student-history-eval.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
