"""Compare one fixed owned-teacher blend on matching frozen development turns."""

from collections import defaultdict
import json

import numpy as np
from scipy.special import logit

from .ami_calibration import _write_json
from .balanced_development_scorecard import (
    METRICS, UNSEEN_IN_FIT, balanced_threshold, calibration_split, evaluate,
)
from .sources import data_root
from .teacher_language_selection import verified_pair_rows

JOB = "balanced-owned-teacher-blend"
WHISPER = "whisper-large-v3"
QWEN = "qwen3-asr-1.7b"
BLEND = "equal-normalized-logit-blend"
MODELS = (WHISPER, QWEN, BLEND)
REPLICATES = 500


def logit_normalized(scores, fit):
    values = logit(np.clip(scores, 1e-6, 1 - 1e-6))
    return (values - values[fit].mean()) / max(values[fit].std(), 1e-6)


def paired_deltas(reference, blend, labels, groups, rng):
    by_turn = defaultdict(list)
    for index, group in enumerate(groups):
        by_turn[group].append(index)
    turns = tuple(by_turn.values())
    draws = []
    difference = blend.astype(float) - reference
    while len(draws) < REPLICATES:
        sample = np.concatenate([turns[i] for i in rng.integers(len(turns), size=len(turns))])
        shift = labels[sample]
        if shift.any() and (~shift).any():
            delta = difference[sample]
            draws.append((delta[shift].mean() - delta[~shift].mean()) / 2)
    return np.asarray(draws)


def score(language, labels, groups, scores, rng):
    fit = calibration_split(groups, language)
    truth = labels.astype(bool)
    values = {name: scores[name] for name in (WHISPER, QWEN)}
    values[BLEND] = (logit_normalized(values[WHISPER], fit) + logit_normalized(values[QWEN], fit)) / 2
    output, decisions = {}, {}
    for name, evidence in values.items():
        cutoff = balanced_threshold(evidence[fit], truth[fit])
        output[name] = {"threshold": cutoff, **evaluate(evidence[~fit], truth[~fit], cutoff)}
        decisions[name] = evidence[~fit] >= cutoff
    draws = paired_deltas(decisions[WHISPER], decisions[BLEND], truth[~fit], groups[~fit], rng)
    return {"calibration_holds": int((fit & ~truth).sum()), "calibration_shifts": int((fit & truth).sum()),
            "evaluation_holds": int((~fit & ~truth).sum()), "evaluation_shifts": int((~fit & truth).sum()),
            "models": output, "paired_balanced_accuracy_delta_ci95": np.percentile(draws, [2.5, 97.5]).tolist()}, draws


def macro(languages, by_language, draws):
    return {
        "languages": list(languages),
        "models": {name: {metric: float(np.mean([by_language[lang]["models"][name][metric] for lang in languages]))
                          for metric in METRICS} for name in MODELS},
        "paired_balanced_accuracy_delta_ci95": np.percentile(
            np.mean([draws[lang] for lang in languages], axis=0), [2.5, 97.5]).tolist(),
    }


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("paired development scores stay on Vast")
    reports = [root / "reports" / f"open-speech-{name}.json" for name in (WHISPER, QWEN)]
    cohorts, provenance, _ = verified_pair_rows(root, reports)
    rng = np.random.default_rng(22112026)
    scores, draws = {}, {}
    for lang, cohort in sorted(cohorts.items()):
        scores[lang], draws[lang] = score(lang, *cohort, rng)
    overall = macro(tuple(scores), scores, draws)
    delta = (overall["models"][BLEND]["balanced_accuracy"] - overall["models"][WHISPER]["balanced_accuracy"])
    report = {
        "status": "development only; fixed historical heads, no external turn-detector outputs",
        "threshold_source": "same hashed whole-turn calibration/evaluation halves as the standard scorecard",
        "primary": "blend minus Whisper macro balanced accuracy; +2 points and positive paired lower bound",
        "scope": "archived common non-English turns; English and Harper-88 predictions unavailable for this score join",
        "pipeline_metrics": "no pipeline replay at the fitted thresholds",
        "models": provenance,
        "macro": overall,
        "zero_supervision_macro": macro(UNSEEN_IN_FIT, scores, draws),
        "languages": scores,
        "passes_exploratory_gate": bool(delta >= .02 and overall["paired_balanced_accuracy_delta_ci95"][0] > 0),
    }
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    run(data_root())
