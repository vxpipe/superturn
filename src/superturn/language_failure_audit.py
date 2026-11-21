"""Verify language-specific Superturn failures on pinned development turns."""

from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import re

import numpy as np
import pyarrow.parquet as pq
import torch

from .ami_calibration import _write_json
from .ami_transfer import partial_auc
from .audio import RATE, read_audio
from .augment import condition
from .dense_teacher import SAMPLES, file_digest
from .eot_domain_probe import GAP, decision_audio
from .model import build
from .multilingual_teacher_probe import DATA, REVISION, pause_spans
from .own_head_distillation import student_scores
from .pitch_check import yin_pitch
from .progress import Progress
from .sources import data_root
from .vxpipe_policy_probe import operating_points

EDGE = SAMPLES - GAP
FEATURE = "feature-first-29"
SECOND_PARENT = "feature-first-43"
TEACHER = "whisper-large-v3"
ADAPTED = ("full-ami-29", "full-ami-43")
PARENTS = dict(zip(ADAPTED, (FEATURE, SECOND_PARENT), strict=True))
HAN_AT_END = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff]+)[\W_]*$")
PINYIN_VERSION = "0.55.0"


def read_receipt(root, name):
    path = root / "reports" / f"{name}.json"
    return json.loads(path.read_text()), file_digest(path)


def read_verified(root, item):
    path = root / item["path"]
    if file_digest(path) != item["sha256"]:
        raise ValueError(f"changed artifact: {path.name}")
    return path


def load_inputs(root):
    fit, fit_hash = read_receipt(root, "own-teacher-feature-student")
    adapt, adapt_hash = read_receipt(root, "own-student-real-labels")
    feature, archive_hash = read_receipt(root, "own-teacher-feature-languages")
    teacher, teacher_hash = read_receipt(root, "open-speech-whisper-large-v3")
    if (feature["dataset_sha256"] != DATA or feature["fit_report_sha256"] != fit_hash
            or adapt["fit_report_sha256"] != fit_hash):
        raise ValueError("feature-first archive has different inputs")
    archive = read_verified(root, {
        "path": f'runs/own-teacher-feature-student/{fit["target_report_sha256"][:12]}/language-scores.json',
        "sha256": feature["paired_scores_sha256"],
    })
    paired = json.loads(read_verified(root, teacher["multilingual_paired_rows"]).read_text())
    if paired["dataset_sha256"] != DATA:
        raise ValueError("own teacher archive has different source data")
    thresholds = {FEATURE: feature["fixed_ami_thresholds"][FEATURE]}
    thresholds.update({name: adapt["metrics"][name]["english"]["fixed"]["ami-calibration"]["threshold"]
                       for name in ADAPTED})
    thresholds[SECOND_PARENT] = adapt["metrics"]["frozen-43"]["english"]["fixed"]["ami-calibration"]["threshold"]
    thresholds[TEACHER] = teacher["metrics"][TEACHER]["english"]["ami_threshold"]
    for child, parent in PARENTS.items():
        if adapt["fits"][child]["parent"] != fit["fits"][parent]["checkpoint"]:
            raise ValueError(f"unexpected parent for {child}")
    models = {}
    for name in (FEATURE, SECOND_PARENT, *ADAPTED):
        report = adapt if name in ADAPTED else fit
        payload = torch.load(read_verified(root, report["fits"][name]["checkpoint"]),
                             map_location="cpu", weights_only=True)
        model = build(payload["config"])
        model.load_state_dict(payload["state"])
        models[name] = model.cuda().eval()
    return models, thresholds, json.loads(archive.read_text()), paired["languages"], {
        "feature_fit": fit_hash, "adapted_fit": adapt_hash,
        "feature_archive": archive_hash, "teacher_report": teacher_hash,
    }


def pitch_pair(source, rendered):
    # Parallel 25 ms frames every 40 ms in the last 900 ms before the edge.
    starts = range(max(0, len(source) - 14400), len(source) - 399, 640)
    start_in_window = EDGE - len(source)
    centers = np.asarray([start - len(source) + 200 for start in starts])
    native = np.asarray([yin_pitch(source[i:i + 400]) for i in starts])
    coded = np.asarray([yin_pitch(rendered[start_in_window + i:start_in_window + i + 400]) for i in starts])

    def contour(values):
        early = values[(centers < -5600) & np.isfinite(values)]
        late = values[(centers >= -5600) & np.isfinite(values)]
        if len(early) < 3 or len(late) < 3:
            return "unmeasurable"
        semitones = 12 * np.log2(np.median(late) / np.median(early))
        if semitones > 2:
            return "rising"
        return "falling" if semitones < -2 else "steady"

    source_valid, codec_valid = np.isfinite(native), np.isfinite(coded)
    both = source_valid & codec_valid
    delta = np.abs(np.log2(native[both] / coded[both]))
    frame_counts = [len(native), int(source_valid.sum()), int(codec_valid.sum()), int(both.sum()),
                    int((np.abs(delta - 1) <= .1).sum())]
    return contour(native), contour(coded), frame_counts


def final_lexical_tone(words, edge):
    from pypinyin import Style, pinyin

    # The word timestamp is provider metadata. Abandon near-edge words
    # without a complete pre-edge interval and never read an upcoming word.
    eligible = [word for word in words if 0 <= word["start"] < word["end"] <= edge]
    if not eligible:
        return "unavailable"
    last = max(eligible, key=lambda word: word["end"])
    if edge - last["end"] > .16:
        return "unavailable"
    match = HAN_AT_END.search(last["word"])
    if match is None:
        return "unavailable"
    sounds = pinyin(match[1], style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[-1]
    tones = {sound[-1] for sound in sounds if sound and sound[-1] in "12345"}
    return f"tone-{next(iter(tones))}" if len(tones) == 1 else "ambiguous"


def scan(source, language, models, noise, meter, *, original_scores=None):
    members, groups, truth = [], [], []
    features = defaultdict(list)
    predictions, noise_predictions = defaultdict(list), defaultdict(list)
    pitch_counts = np.zeros((2, 5), dtype=int)
    turn = 0
    columns = ["id", "audio", "language", "cohort", "silence_spans"]
    if language == "zh":
        columns.append("words")
    for batch in source.iter_batches(batch_size=8, columns=columns):
        clips, padding = [], []
        for row in batch.to_pylist():
            decoded = read_audio(row["audio"]["bytes"])
            previous = 0
            previous_edge = -1
            for ordinal, edge, label in pause_spans(row, decoded, language):
                if edge < previous_edge:
                    raise ValueError(f"out-of-order provider pauses: {language} turn {turn}")
                history = min(EDGE, edge)
                rendered = decision_audio(decoded, edge, "telephone")[-SAMPLES:]
                if not members:
                    altered = decoded.copy()
                    altered[edge + GAP:] = 32767
                    if not np.array_equal(rendered, decision_audio(altered, edge, "telephone")[-SAMPLES:]):
                        raise ValueError("future decoded samples changed the causal render")
                raw = decoded[edge - history:edge].astype(np.float32) / 32768
                native_pitch, codec_pitch, counts = pitch_pair(raw, rendered)
                pitch_counts[label] += counts
                members.append((turn, row["id"], ordinal, edge, label))
                groups.append(turn)
                truth.append(label)
                features["history"].append("<1.2s" if history < 1.2 * RATE else "<2.7s" if history < EDGE else "full")
                features["position"].append("first" if previous == 0 else "later")
                features["native_f0"].append(native_pitch)
                features["g711_f0"].append(codec_pitch)
                if language == "zh":
                    tone = final_lexical_tone(row["words"] or [], edge / RATE)
                    features["lexical_tone"].append(tone)
                    features["lexical_tone_and_source_f0"].append(f"{tone}/{native_pitch}")
                clips.append(rendered)
                padding.append(EDGE - history)
                previous += 1
                previous_edge = edge
            turn += 1
        if clips:
            inputs = np.stack(clips)
            padded = inputs.copy()
            for clip, length in zip(padded, padding, strict=True):
                clip[:length] = noise[:length]
            for name, model in models.items():
                if original_scores is not None and name in original_scores:
                    start = len(truth) - len(clips)
                    clean = np.asarray(original_scores[name][start:len(truth)])
                    if len(clean) != len(inputs):
                        raise ValueError(f"missing archived scores for {language}/{name}")
                else:
                    clean = student_scores(model, inputs)
                changed = clean.copy()
                short = np.asarray(padding) > 0
                if short.any():
                    changed[short] = student_scores(model, padded[short])
                predictions[name].extend(clean)
                noise_predictions[name].extend(changed)
        meter.advance(batch.num_rows)
    membership = hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest()
    return groups, truth, features, predictions, noise_predictions, pitch_counts, membership


def summarize(accepted, truth, subset):
    shifts = subset & truth
    holds = subset & ~truth
    return {"shift_hits": int((accepted & shifts).sum()), "shifts": int(shifts.sum()),
            "false_completes": int((accepted & holds).sum()), "holds": int(holds.sum())}


def stratify(scores, noise_scores, targets, features, thresholds, groups):
    truth = np.asarray(targets, dtype=bool)
    matched, all_rows = {}, np.ones(len(truth), dtype=bool)
    report = {}
    for name, values in scores.items():
        values = np.asarray(values)
        if len(values) != len(truth) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid scores for {name}")
        _, _, choices, _, holds = operating_points(values, truth)
        cutoff = choices[holds <= .05 * (~truth).sum()][-1]
        matched[name] = values >= cutoff
        row = {"fixed": summarize(values >= thresholds[name], truth, all_rows),
               "matched": summarize(matched[name], truth, all_rows),
               "partial_auc_to_5pct": partial_auc(values.tolist(), targets, max_fpr=.05),
               "matched_cutoff": float(cutoff) if np.isfinite(cutoff) else None}
        if name in noise_scores:
            noisy = np.asarray(noise_scores[name])
            _, _, noisy_choices, _, noisy_holds = operating_points(noisy, truth)
            noisy_cutoff = noisy_choices[noisy_holds <= .05 * (~truth).sum()][-1]
            noise_matched = noisy >= noisy_cutoff
            row["comfort_noise_rematched"] = summarize(noise_matched, truth, all_rows)
            row["comfort_noise_at_original_cutoff"] = {}
            row["comfort_noise_rematched_by_history"] = {}
            for band in sorted(set(features["history"])):
                subset = np.asarray(features["history"]) == band
                row["comfort_noise_at_original_cutoff"][band] = {
                    **summarize(noisy >= cutoff, truth, subset),
                    "flipped": int((subset & ((noisy >= cutoff) != matched[name])).sum()),
                }
                row["comfort_noise_rematched_by_history"][band] = summarize(noise_matched, truth, subset)
        report[name] = row
    strata = {}
    for key, cells in features.items():
        strata[key] = {}
        for cell in sorted(set(cells)):
            subset = np.asarray(cells) == cell
            base = {name: summarize(decision, truth, subset) for name, decision in matched.items()}
            for name in (*PARENTS, *PARENTS.values()):
                base[name]["teacher_only_shifts"] = int((subset & truth & matched[TEACHER] & ~matched[name]).sum())
                base[name]["both_miss_shifts"] = int((subset & truth & ~matched[TEACHER] & ~matched[name]).sum())
            strata[key][cell] = {"models": base, "turns": len(set(np.asarray(groups)[subset]))}
    return report, strata


def adaptation_intervals(scores, labels, groups, rng):
    truth = np.asarray(labels, dtype=bool)
    group_array = np.asarray(groups)
    buckets = [np.flatnonzero(group_array == group) for group in np.unique(group_array)]
    deltas = []
    arrays = {name: np.asarray(scores[name]) for name in (*PARENTS, *PARENTS.values())}
    for draw in rng.integers(len(buckets), size=(300, len(buckets))):
        index = np.concatenate([buckets[i] for i in draw])
        y = truth[index]
        rates = {}
        for name, full in arrays.items():
            values = full[index]
            _, _, _, hits, holds = operating_points(values, y)
            recall = hits[holds <= .05 * (~y).sum()][-1] / y.sum()
            rates[name] = np.asarray([recall, partial_auc(values.tolist(), y.tolist(), max_fpr=.05)])
        deltas.append(np.stack([rates[child] - rates[parent] for child, parent in PARENTS.items()]))
    return np.asarray(deltas)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the audit requires staged Vast data")
    if version("pypinyin") != PINYIN_VERSION:
        raise ValueError(f"the lexical-tone audit requires pypinyin {PINYIN_VERSION}")
    models, thresholds, archived, teacher, receipts = load_inputs(root)
    sources = {}
    for language, sha in DATA.items():
        path = f"raw/livekit_multilingual_diagnostic/data/{language}/validation-00000-of-00001.parquet"
        sources[language] = pq.ParquetFile(read_verified(root, {"path": path, "sha256": sha}))
    rng = np.random.default_rng(271)
    noise = condition(np.round(rng.normal(0, 110, EDGE)).astype(np.int16), "telephone", 0).astype(np.float32)
    noise *= 10 ** (-55 / 20) / np.sqrt(np.mean(noise ** 2))
    meter = Progress("language-failure-audit", sum(p.metadata.num_rows for p in sources.values()), "turns")
    output, bootstrap = {}, []
    try:
        for language, source in sources.items():
            groups, truth, feats, scores, noise_scores, counts, membership = scan(source, language, models, noise, meter)
            baseline = archived[language]
            own_teacher = teacher[language]
            for rows in (baseline, own_teacher):
                if truth != rows["labels"] or groups != rows["turn_ordinals"]:
                    raise ValueError(f"changed pause order for {language}")
            replay = np.asarray(scores[FEATURE])
            original = np.asarray(baseline["scores"][FEATURE])
            error = float(np.max(np.abs(replay - original)))
            if error > 1e-5 or not np.array_equal(replay >= thresholds[FEATURE], original >= thresholds[FEATURE]):
                raise ValueError(f"feature-first score replay failed for {language}: {error:g}")
            scores[FEATURE] = original
            scores[TEACHER] = own_teacher["scores"][TEACHER]
            metrics, strata = stratify(scores, noise_scores, truth, feats, thresholds, groups)
            bootstrap.append(adaptation_intervals(scores, truth, groups, rng))
            output[language] = {"metrics": metrics, "strata": strata, "membership_sha256": membership,
                                "feature_replay_max_error": error, "turns": len(set(groups)),
                                "source_turns": source.metadata.num_rows,
                                "column_names": source.schema_arrow.names,
                                "pitch_frames": {label: dict(zip(
                                    ("total", "source_voiced", "g711_voiced", "both_voiced", "octave_disagreement"),
                                    total.tolist(), strict=True))
                                    for label, total in zip(("hold", "shift"), counts, strict=True)}}
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"scope": "provider-labelled development diagnostics, 300 ms causal G.711 audio",
              "revision": REVISION, "source_sha256": DATA, "inputs": receipts,
              "lexical_tone_dictionary_version": PINYIN_VERSION,
              "fixed_thresholds": thresholds, "adaptation_parents": PARENTS, "languages": output,
              "macro_paired_turn_bootstrap_95": dict(zip(
                  ADAPTED, np.quantile(np.mean(bootstrap, axis=0), [.025, .975], axis=0)
                  .transpose(1, 2, 0).tolist(), strict=True)),
              "bootstrap_metrics": ["matched_recall_delta", "partial_auc_to_5pct_delta"]}
    _write_json(root / "reports/language-failure-audit.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
