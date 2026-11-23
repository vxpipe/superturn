"""Shared threshold-free metrics for binary end-of-turn scores."""

import numpy as np

BUDGETS = np.array([.025, .05, .075])


def auc(scores, truth):
    pairs = sorted(zip(scores, truth))
    positive = sum(truth)
    negative = len(truth) - positive
    if not positive or not negative:
        return None
    rank_sum = 0.0
    from itertools import groupby
    rank = 1
    for _, tied in groupby(pairs, key=lambda pair: pair[0]):
        labels = [label for _, label in tied]
        rank_sum += sum(labels) * (rank + (len(labels) - 1) / 2)
        rank += len(labels)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def recall_area(scores, truth):
    order = np.argsort(-scores, kind="stable")
    boundaries = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    hits = np.cumsum(truth[order])[boundaries]
    false = boundaries + 1 - hits
    fpr = np.r_[0., false / np.sum(truth == 0)]
    tpr = np.r_[0., hits / np.sum(truth == 1)]
    keep = fpr < .10
    end = np.searchsorted(fpr, .10)
    ordinate = np.interp(.10, fpr[end - 1:end + 1], tpr[end - 1:end + 1])
    x, y = np.r_[fpr[keep], .10], np.r_[tpr[keep], ordinate]
    recalls = [tpr[fpr <= budget][-1] for budget in BUDGETS]
    return np.asarray([*recalls, np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / .10])
