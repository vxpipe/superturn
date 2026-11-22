"""Replay frozen wire-format candidate scores against silence timers."""

import argparse
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .balanced_development_scorecard import UNSEEN_IN_FIT, balanced_threshold, calibration_split, evaluate
from .balanced_scorecard_references import MODELS as REFERENCES
from .dense_teacher import file_digest
from .language_failure_audit import read_receipt
from .sources import data_root
from .vxpipe_policy_probe import FALLBACK, TIMES, english, harper, languages, operating_points, policy

JOB = "e20-timer-replay"
CANDIDATES = ("e20-compact-int8", "e20-float-onnx", "whisper-large-v3", *REFERENCES)
SCENARIOS = [(reply, 200, confirm) for reply in TIMES for confirm in (0, 100)]


def observed_subset(rows, scores, mask):
    values = np.asarray(rows)
    truth = values[:, 0].astype(bool)
    censored = values[:, 3].astype(bool)
    eligible = mask & (truth | ~censored)
    if len(set(truth[eligible])) != 2:
        raise ValueError("pipeline replay needs observed holds and shifts")
    return (
        (truth[eligible], values[eligible, 2].astype(float), scores[eligible]),
        {"holds": int((~truth & eligible).sum()), "shifts": int((truth & eligible).sum()),
         "excluded_censored_holds": int((mask & ~truth & censored).sum()),
         "retained_censored_shifts": int((mask & truth & censored).sum())},
    )


def decision_stats(data, decision, reply, grace, confirm):
    truth, duration, _ = data
    values = policy(truth, duration, np.zeros(len(truth), dtype=bool), decision, reply, grace, confirm)
    return {
        "mean_shift_reply_ms": values["mean_shift_reply_ms"],
        "shift_reply_by_ms": values["shift_reply_by_ms"],
        "early_shift_recall": values["early_shift_recall"],
        "audible_false_starts_per_observed_hold": values["audible_false_starts_per_hold"][0],
        "wasted_drafts_per_observed_hold": values["wasted_drafts_per_hold"][0],
        "raw_false_complete_300ms_on_observed_holds": values["raw_false_complete_300ms"],
    }


def calibration_cutoffs(data, reply, grace, confirm):
    """Find equal-or-better timer tradeoffs on calibration rows only."""
    truth, duration, scores = data
    order, ends, thresholds, shifts, _ = operating_points(scores, truth)
    resume = duration + confirm
    hold = ~truth
    late = hold & (resume > FALLBACK + reply + grace)
    extra = hold & (resume > 300 + reply + grace) & ~late
    count = late.sum() + np.r_[0, np.cumsum(extra[order])[ends]]
    results = {}
    for timer in TIMES:
        # Compare integer numerator counts to avoid tolerance-dependent tie choices.
        fast = np.flatnonzero((FALLBACK - 300) * shifts >= (FALLBACK - timer) * truth.sum())
        limit = (hold & (resume > timer + reply + grace)).sum()
        safe = np.flatnonzero(count <= limit)
        fast_index = fast[np.argmin(count[fast])] if len(fast) else None
        safe_index = safe[np.argmax(shifts[safe])] if len(safe) else None
        results[str(timer)] = {
            "match_timer_latency": None if fast_index is None else float(thresholds[fast_index]),
            "match_timer_false_starts": None if safe_index is None else float(thresholds[safe_index]),
        }
    return results


def scenario(data, fixed, cutoffs, reply, grace, confirm):
    _, _, scores = data

    def at(threshold):
        if threshold is None:
            return None
        result = decision_stats(data, np.where(scores >= threshold, 300., FALLBACK), reply, grace, confirm)
        result["threshold"] = threshold if np.isfinite(threshold) else None
        result["always_fallback"] = not np.isfinite(threshold)
        return result

    policies = {str(timer): decision_stats(data, np.full(len(scores), float(timer)), reply, grace, confirm)
                for timer in TIMES}
    return {"silence_only": policies, "calibrated_balanced_threshold": at(fixed),
            "calibration_matched_thresholds": {timer: {objective: at(threshold) for objective, threshold in items.items()}
                                               for timer, items in cutoffs.items()}}


def run(root, scores_sha):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("development timing data stays on Vast")
    export, export_sha = read_receipt(root, "e20-onnx-quality")
    source = root / export["compact"]["candidate"]["path"]
    if file_digest(source) != export["compact"]["candidate"]["sha256"]:
        raise ValueError("the replayed ONNX candidate has changed")
    paired = source.with_name("paired-scores.json")
    if file_digest(paired) != scores_sha:
        raise ValueError("paired scores differ from the verified backup")
    members = json.loads(paired.read_text())
    benchmarks, benchmark_sha = read_receipt(root, "balanced-development-scorecard-references")
    keys = {}
    keys["en"], keys["harper-88"] = [], []
    boundaries = {
        "en": english(root, ordered_keys=keys["en"]),
        "harper-88": harper(root, ordered_keys=keys["harper-88"], caller_resume=True),
        **languages(root, ordered_keys=keys),
    }
    if set(boundaries) != set(members):
        raise ValueError("development cohorts changed")
    output, calibration, provenance = {}, {}, {}
    for name in ("en", *sorted(boundaries.keys() - {"en", "harper-88"}), "harper-88"):
        rows, row = boundaries[name], members[name]
        if ([y for y, *_ in rows] != row["labels"] or [group for group, _, _ in keys[name]] != row["turn_ordinals"]):
            raise ValueError(f"development roles or labels changed for {name}")
        path = root / "runs" / "balanced-development-scorecard" / f"reference-{name}.json"
        reference = json.loads(path.read_text())
        if (reference["ordered_pause_membership_sha256"] != hashlib.sha256(json.dumps(keys[name]).encode()).hexdigest()
                or reference["comparison"] != benchmarks["references"]):
            raise ValueError("ordered pause identities or benchmark checkpoints changed")
        provenance[name] = {"reference_receipt_sha256": file_digest(path),
                            "ordered_pause_membership_sha256": reference["ordered_pause_membership_sha256"]}
        truth = np.asarray(row["labels"], dtype=bool)
        fit = calibration_split(row["turn_ordinals"], name) if name != "harper-88" else np.zeros(len(rows), dtype=bool)
        test = ~fit
        models = {}
        for model in CANDIDATES:
            if name == "harper-88" and model == "whisper-large-v3":
                continue
            scores = np.asarray(reference["scores"][model] if model in REFERENCES else row["scores"][model], dtype=float)
            if scores.shape != truth.shape or not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
                raise ValueError("invalid archived model probabilities")
            expected = (benchmarks if model in REFERENCES else export)
            view = expected["harper-88"] if name == "harper-88" else expected["languages"][name]
            previous = view["models"][model]
            fixed = previous["threshold"]
            if fit.any() and balanced_threshold(scores[fit], truth[fit]) != fixed:
                raise ValueError(f"{name} {model} failed to reproduce its calibrated threshold")
            observed = evaluate(scores[test], truth[test], fixed)
            if any(abs(value - previous[key]) > 1e-12 for key, value in observed.items()):
                raise ValueError(f"{name} {model} failed to reproduce its held metrics")
            evaluated, counts = observed_subset(rows, scores, test)
            trained = observed_subset(rows, scores, fit)[0] if fit.any() else None
            scenarios = {}
            for reply, grace, confirm in SCENARIOS:
                scenario_key = f"reply-{reply}-grace-{grace}-confirm-{confirm}"
                calibration_key = (name if fit.any() else "en"), model, scenario_key
                if trained is not None:
                    calibration[calibration_key] = calibration_cutoffs(trained, reply, grace, confirm)
                scenarios[scenario_key] = scenario(evaluated, fixed, calibration[calibration_key], reply, grace, confirm)
            models[model] = {"counts": counts, "raw_scorecard": observed, "scenarios": scenarios}
        output[name] = models
    report = {"scope": "development replay with same held pauses per policy; censored holds omitted for pipeline metrics",
              "assumptions": {"one_model_decision_ms": 300, "fallback_ms": FALLBACK, "reply_shape": "uninterrupted shifts",
                              "grace_ms": 200, "confirmation_sensitivity_ms": [0, 100]},
              "archived_scorecard_sha256": export_sha, "archived_benchmark_scorecard_sha256": benchmark_sha,
              "paired_scores_sha256": scores_sha, "reference_receipts": provenance,
              "six_language_view": list(UNSEEN_IN_FIT), "languages": output}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps({"views": len(output), "english": output["en"]["e20-compact-int8"]["counts"],
                      "harper": output["harper-88"]["e20-compact-int8"]["counts"]}))


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-sha256", required=True)
    run(data_root(), parser.parse_args().scores_sha256)
