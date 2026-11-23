"""Resample fixed paired text predictions by caller."""

from collections import defaultdict

import numpy as np

from .turn_text_train import binary_metrics

METRICS = (
    "balanced_accuracy", "shift_recall", "false_complete", "auc",
    "normalized_pauc_to_10_percent", "retrospective_recall_at_5_percent_false_complete",
)


def paired_intervals(items, baseline, *, draws=2000):
    if len(items) != len(baseline) or set(baseline) != {row["rank"] for row in items}:
        raise ValueError("both models must score the same inputs")
    callers = defaultdict(list)
    for index, row in enumerate(items):
        callers[row["caller"]].append(index)
    groups = list(callers.values())
    pairs = [(row["score"], row["label"]) for row in items]
    control = [(baseline[row["rank"]], row["label"]) for row in items]
    trial, reference = binary_metrics(pairs), binary_metrics(control)
    samples = defaultdict(list)
    rng = np.random.default_rng(29)
    for sampled in rng.integers(len(groups), size=(draws, len(groups))):
        indices = [i for group in sampled for i in groups[group]]
        if len({items[i]["label"] for i in indices}) < 2:
            continue
        left = binary_metrics([pairs[i] for i in indices])
        right = binary_metrics([control[i] for i in indices])
        for metric in METRICS:
            samples[metric].append(left[metric] - right[metric])
    return {
        "callers": len(groups),
        "replicates": len(samples["auc"]),
        "estimate": {metric: trial[metric] - reference[metric] for metric in METRICS},
        "paired_caller_bootstrap_95": {metric: np.quantile(values, [.025, .975]).tolist()
                                       for metric, values in samples.items()},
    }
