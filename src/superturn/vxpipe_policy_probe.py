"""Estimate VXPipe false starts and speculative drafts on pinned development pauses."""

import argparse
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from .ami_calibration import _write_json
from .ami_external_baselines import load_detectors, load_smart_turn
from .audio import RATE, read_audio, silero
from .dense_teacher import file_digest
from .eot_domain_probe import GAP, decision_audio, load_validated_models
from .harper_valley_audit import audit_call
from .labels import Speech, turns
from .multilingual_teacher_probe import DATA
from .open_speech_teacher_probe import ENCODERS
from .own_head_distillation import BASELINE, predictions, student_scores
from .progress import Progress
from .sources import data_root
from .turn1_mini_probe import english_spans
from .wav2vec_harper_probe import extract, pinned_inputs

TIMES = (300, 500, 700, 1000)
FALLBACK = 1000
MODELS = {"whisper-large-v3": "open-speech-whisper-large-v3",
          "qwen3-asr-1.7b": "open-speech-qwen3-asr-1.7b",
          "hubert-large": "open-speech-hubert-large"}


def english(root):
    return [(label, group, (end - edge) * 1000 / RATE, end == len(pcm))
            for pcm, edge, end, label, group in english_spans(root)]


def languages(root):
    result = {}
    paths = {lang: root / f"raw/livekit_multilingual_diagnostic/data/{lang}/validation-00000-of-00001.parquet"
             for lang in DATA}
    sources = {lang: pq.ParquetFile(path) for lang, path in paths.items()}
    meter = Progress("vxpipe-language-pause-lengths", sum(source.metadata.num_rows for source in sources.values()), "turns")
    try:
        for lang, source in sources.items():
            if file_digest(paths[lang]) != DATA[lang]:
                raise ValueError(f"unverified development audio for {lang}")
            found, group = [], 0
            for batch in source.iter_batches(batch_size=8, columns=["audio", "cohort", "language", "silence_spans"]):
                for row in batch.to_pylist():
                    if (row["language"], row["cohort"]) != (lang, "general"):
                        raise ValueError("unexpected language diagnostic cohort")
                    length = len(read_audio(row["audio"]["bytes"]))
                    for span in row["silence_spans"]:
                        edge = round(span["start"] * RATE)
                        end = min(length, round(span["end"] * RATE))
                        if not 0 <= edge <= end or end - edge < GAP:
                            continue
                        if span["label"] not in {"eot", "hold"}:
                            raise ValueError("unexpected development pause label")
                        found.append((int(span["label"] == "eot"), group, (end - edge) * 1000 / RATE, end == length))
                    group += 1
                meter.advance(batch.num_rows)
            result[lang] = found
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return result


def harper(root):
    archived = json.loads((root / "reports/harper-valley-development-stratified-provenance.json").read_text())
    detect, result = silero(), []
    for group, row in enumerate(archived):
        if not row["fixed_pauses"]:
            continue
        duration = {}

        def observe(audio, origins, vad, words, omissions, left, right):
            clipped = [Speech(part.speaker, max(left, part.start), min(right, part.end))
                       for part in vad if max(left, part.start) < min(right, part.end)]
            for span, gap, label in turns(clipped, decision_ms=300):
                if span.speaker == "caller":
                    duration[span.end - origins["caller"], label] = gap * 1000 / RATE

        reproduced = []
        audit_call(root / "raw/harper-valley", row["call"], detect, recover_clock=True,
                   pauses=reproduced, observe=observe)
        if [list(item) for item in reproduced] != row["fixed_pauses"]:
            raise ValueError("the archived HarperValley pause membership did not reproduce")
        for edge, label in reproduced:
            result.append((int(label == "shift"), group, duration[edge, label], False))
    return result


def operating_points(scores, truth):
    order = np.argsort(-scores, kind="stable")
    endpoints = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    shifts = np.cumsum(truth[order])[endpoints]
    holds = endpoints + 1 - shifts
    thresholds = np.r_[np.inf, scores[order][endpoints]]
    return order, endpoints, thresholds, np.r_[0, shifts], np.r_[0, holds]


def policy(truth, duration, censored, decision, reply, grace, confirm):
    hold = ~truth
    play = decision + reply + grace
    resume = duration + confirm
    starts = hold & (resume > play)
    unknown_starts = hold & censored & ~starts & np.isfinite(play)
    wasted = hold & ~censored & (resume > decision) & (resume <= play)
    unknown_wasted = hold & censored & ~starts & np.isfinite(play)
    triggered = hold & (resume > decision)
    raw_cutoff = hold & (decision == 300)
    n_shifts, n_holds = truth.sum(), hold.sum()
    return {
        "shift_reply_by_ms": {str(t + reply + grace): float(np.mean(play[truth] <= t + reply + grace)) for t in TIMES},
        "mean_shift_reply_ms": float(play[truth].mean()) if np.isfinite(play[truth]).all() else None,
        "early_shift_recall": float(np.mean(decision[truth] == 300)),
        "audible_false_starts_per_hold": [float(starts.sum() / n_holds), float((starts | unknown_starts).sum() / n_holds)],
        "wasted_drafts_per_hold": [float(wasted.sum() / n_holds), float((wasted | unknown_wasted).sum() / n_holds)],
        "wasted_drafts_per_eligible_pause": [float(wasted.mean()), float((wasted | unknown_wasted).mean())],
        "certain_audible_per_triggered_hold": float(starts.sum() / triggered.sum()) if triggered.any() else None,
        "false_start_if_every_resume_missed_per_hold_upper": float((hold & np.isfinite(decision)).sum() / n_holds),
        "raw_false_complete_300ms": float(raw_cutoff.sum() / n_holds),
        "raw_accuracy_300ms": float(np.mean((decision == 300) == truth)),
        "raw_balanced_accuracy_300ms": float(((decision[truth] == 300).sum() / n_shifts + 1 - raw_cutoff.sum() / n_holds) / 2),
    }


def compare(rows, scores, fixed, fixed_name="fixed-AMI"):
    truth = np.asarray([item[0] for item in rows], dtype=bool)
    duration = np.asarray([item[2] for item in rows])
    censored = np.asarray([item[3] for item in rows], dtype=bool)
    order, ends, thresholds, shifts, holds = operating_points(scores, truth)
    candidates = {fixed_name: fixed,
                  "matched-5pct-raw": thresholds[np.flatnonzero(holds <= .05 * (~truth).sum())[-1]],
                  "matched-80pct-recall": thresholds[np.flatnonzero(shifts >= .80 * truth.sum())[0]]}
    scenarios = {}
    for reply in TIMES:
        for grace, confirm in ((200, 0), (200, 100), (100, 0), (300, 0)):
            label = f"reply-{reply}-grace-{grace}-confirm-{confirm}"
            baselines = {str(t): policy(truth, duration, censored, np.full(len(truth), float(t)), reply, grace, confirm)
                         for t in TIMES}
            table = {}
            for name, threshold in candidates.items():
                fired = scores >= threshold
                table[name] = {
                    "threshold": float(threshold) if np.isfinite(threshold) else None,
                    "single-decision": policy(truth, duration, censored, np.where(fired, 300., np.inf), reply, grace, confirm),
                    "with-fallback": policy(truth, duration, censored, np.where(fired, 300., FALLBACK), reply, grace, confirm),
                }
            # Sweep tied score groups for the 300 ms fast path plus a 1000 ms fallback.
            # Compare conservative false-start bounds and mean answer time on shifts.
            resume = duration + confirm
            known_late = (~truth) & ((resume > FALLBACK + reply + grace) | censored)
            extra = (~truth) & (resume > 300 + reply + grace) & ~known_late
            extra_count = np.r_[0, np.cumsum(extra[order])[ends]]
            rates = (known_late.sum() + extra_count) / (~truth).sum()
            mean_time = FALLBACK + reply + grace - (FALLBACK - 300) * shifts / truth.sum()
            matches = {}
            for timer, baseline in baselines.items():
                limit = baseline["audible_false_starts_per_hold"][1]
                deadline = baseline["mean_shift_reply_ms"]
                fast_enough = np.flatnonzero(mean_time <= deadline + 1e-9)
                allowed = np.flatnonzero(rates <= limit + 1e-12)
                matches[timer] = {
                    "at_matched_mean_shift_reply_ms": float(rates[fast_enough].min()) if len(fast_enough) else None,
                    "at_matched_false_start_rate_mean_shift_reply_ms": float(mean_time[allowed].min()) if len(allowed) else None,
                    "timer_false_start_upper": limit, "timer_mean_shift_reply_ms": deadline,
                }
            scenarios[label] = {"silence_only": baselines, "thresholds": table, "matched_frontier": matches}
    return {"holds": int((~truth).sum()), "shifts": int(truth.sum()),
            "censored_holds": int((censored & ~truth).sum()), "scenarios": scenarios}


def reference_views(root, boundaries):
    predict, presets, smart_hash = load_detectors(root)
    smart, _ = load_smart_turn(root)
    expected = json.loads((root / "reports/s1-a-150000-29-eot-domain-probe.json").read_text())
    if expected["smart_turn_sha256"] != smart_hash:
        raise ValueError("the Smart Turn benchmark checkpoint changed")
    scores = {name: [] for name in (*presets, "smart-turn-3s")}
    presets["smart-turn-3s"] = presets["smart-turn-v3.2"]
    meter = Progress("vxpipe-reference-english", len(boundaries), "pauses")
    try:
        for pcm, edge, _, _, _ in english_spans(root):
            audio = decision_audio(pcm, edge, "telephone")
            for name, value in predict({"pause_start": edge}, 300, audio, turn_samples=edge + GAP).items():
                scores[name].append(value)
            scores["smart-turn-3s"].append(smart(audio[-48000:]))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    results = {name: {"english": compare(boundaries, np.asarray(values), presets[name], "preset")}
               for name, values in scores.items()}
    for name, recall in (("smart-turn-3s", .2525), ("smart-turn-v3.2", .355), ("livekit-v1.1-mini", .4675)):
        replayed = results[name]["english"]["scenarios"]["reply-300-grace-200-confirm-0"]
        actual = replayed["thresholds"]["matched-5pct-raw"]["single-decision"]["early_shift_recall"]
        if actual != recall:
            raise ValueError(f"the archived matched-5% benchmark did not reproduce for {name}")
    return results


def run(root, replay, *, extend=False, references=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("development replay stays on Vast")
    boundaries = {"english": english(root), "harper": harper(root)}
    if not extend:
        boundaries.update(languages(root))
    clips, truth, splits, _ = pinned_inputs(root)
    evaluation = np.r_[splits["english"], splits["harper"], splits["calibration"]]
    output = root / "reports/vxpipe-policy-development.json"
    if extend:
        result = json.loads(output.read_text())
    else:
        result = {"scope": "counterfactual development replay, with optimistic uninterrupted shift response",
                  "reply_times_ms": TIMES, "grace_default_ms": 200, "single_model_query_ms": 300,
                  "fallback_ms": FALLBACK, "models": {}}
        [(student_name, student, threshold)], _ = load_validated_models(root, [BASELINE])
        student_views = {}
        for cohort in ("english", "harper"):
            idx = splits[cohort]
            if [r[0] for r in boundaries[cohort]] != truth[idx].tolist():
                raise ValueError("the student replay label order changed")
            student_views[cohort] = compare(boundaries[cohort], student_scores(student, clips[idx], device="cpu"), threshold)
        result["models"][student_name] = student_views
        _write_json(output, result)
    for name, stem in MODELS.items():
        if extend and name not in replay:
            continue
        report_path = root / "reports" / f"{stem}.json"
        if not report_path.is_file():
            if name in replay:
                raise ValueError(f"requested replay is missing the locked report for {name}")
            continue
        report = json.loads(report_path.read_text())
        if "multilingual_paired_rows" not in report:
            if name in replay:
                raise ValueError(f"requested replay is missing the paired report for {name}")
            continue
        paired = report["multilingual_paired_rows"]
        pair_path = root / paired["path"]
        if file_digest(pair_path) != paired["sha256"]:
            raise ValueError("the paired language scores changed")
        rows = json.loads(pair_path.read_text())
        if rows["dataset_sha256"] != DATA:
            raise ValueError("the language score corpus changed")
        fixed = report["metrics"][name]["english"]["ami_threshold"]
        model = result["models"].get(name, {})
        if name in replay:
            locked = report["diagnostic_checkpoint"]
            head_path = root / locked["path"]
            if file_digest(head_path) != locked["sha256"]:
                raise ValueError("the frozen head changed")
            head = torch.load(head_path, map_location="cpu", weights_only=True)
            predict, provenance = ENCODERS[name](root)
            values = predictions(extract(f"vxpipe-{name}", predict, clips)[evaluation], head["heads"])
            if (provenance["sha256"] != head["encoder_sha256"]
                    or hashlib.sha256(values.tobytes()).hexdigest() != locked["evaluation_probability_sha256"]):
                raise ValueError("the archived decision scores did not reproduce")
            del predict
            torch.cuda.empty_cache()
            for cohort in ("english", "harper"):
                idx = splits[cohort]
                if [r[0] for r in boundaries[cohort]] != truth[idx].tolist():
                    raise ValueError("the replay label order changed")
                model[cohort] = compare(boundaries[cohort], values[idx - evaluation[0]], fixed)
        if not extend:
            for lang, row in rows["languages"].items():
                if [(r[0], r[1]) for r in boundaries[lang]] != list(zip(row["labels"], row["turn_ordinals"])):
                    raise ValueError(f"the timing and scores are misaligned on {lang}")
                model[lang] = compare(boundaries[lang], np.asarray(row["scores"][name]), fixed)
        result["models"][name] = model
        _write_json(output, result)
    if references:
        result["models"].update(reference_views(root, boundaries["english"]))
        _write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", choices=MODELS, nargs="*", default=list(MODELS))
    parser.add_argument("--extend", action="store_true", help="add teacher English and HarperValley replays to an existing table")
    parser.add_argument("--references", action="store_true", help="also replay the archived English vendor benchmarks")
    torch.set_num_threads(2)
    args = parser.parse_args()
    summary = run(data_root(), args.replay, extend=args.extend, references=args.references)
    print(json.dumps({"models": list(summary["models"]), "cohorts": list(next(iter(summary["models"].values()), {}))}))
