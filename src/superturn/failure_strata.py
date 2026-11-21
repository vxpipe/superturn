"""Paired failure counts on matched hold-budget operating points."""

import numpy as np

from .vxpipe_policy_probe import operating_points

MEASURES = (
    "shift_misses", "false_completes", "student_only_misses", "shared_misses",
    "student_only_false_completes", "shared_false_completes",
)


def cutoff_at_budget(scores, truth, budget=.05):
    _, _, thresholds, _, holds = operating_points(scores, truth)
    return thresholds[holds <= budget * (~truth).sum()][-1]


def decisions(scores, truth, budget=.05):
    return scores >= cutoff_at_budget(scores, truth, budget)


def strata_masks(groups, features):
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    history = np.asarray(features["history"])
    position = np.asarray(features["position"])
    pitch = np.asarray(features["native_f0"])
    result = {"all": np.ones(len(groups), dtype=bool), "single_pause": sizes[inverse] == 1,
              "multiple_pauses": sizes[inverse] > 1, "short_history": history != "full",
              "full_history": history == "full", "very_short_history": history == "<1.2s",
              "first_pause": position == "first", "later_pause": position == "later"}
    for name in ("rising", "falling", "steady", "unmeasurable"):
        result[f"source_f0_{name}"] = pitch == name
    return result


def failure_counts(fired, teacher_fired, truth, masks):
    missed = truth & ~fired
    false = ~truth & fired
    event = np.asarray([missed, false, missed & teacher_fired, missed & ~teacher_fired,
                        false & ~teacher_fired, false & teacher_fired], dtype=np.int32)
    return masks @ event.T


def failure_comparison(scores, teacher_name, truth, groups, features, contrasts, *, draws=300, seed=271):
    truth = np.asarray(truth, dtype=bool)
    group_count = np.asarray(groups)
    scores = {name: np.asarray(value, dtype=float) for name, value in scores.items()}
    if set(truth) != {False, True} or any(x.shape != truth.shape or not np.isfinite(x).all() for x in scores.values()):
        raise ValueError("expected finite scores paired with both development classes")
    masks = strata_masks(group_count, features)
    names = tuple(masks)
    masks = np.asarray(list(masks.values()), dtype=np.int32)
    classes = np.asarray([truth, ~truth], dtype=np.int32)
    denominators = masks @ classes.T

    def counts(index, wanted):
        current = truth[index]
        teacher = decisions(scores[teacher_name][index], current)
        return {name: failure_counts(decisions(scores[name][index], current), teacher, current, masks[:, index])
                for name in wanted}

    full = counts(np.arange(len(truth)), scores)
    slices = {
        name: {"shifts": int(denominators[i, 0]), "holds": int(denominators[i, 1]),
               "turns": len(np.unique(group_count[masks[i].astype(bool)])),
               "models": {model: dict(zip(MEASURES, result[i].tolist(), strict=True)) for model, result in full.items()}}
        for i, name in enumerate(names)
    }
    selected = sorted({name for pair in contrasts.values() for name in pair})
    buckets = [np.flatnonzero(group_count == group) for group in np.unique(group_count)]
    rng = np.random.default_rng(seed)
    paired = []
    for draw in rng.integers(len(buckets), size=(draws, len(buckets))):
        index = np.concatenate([buckets[i] for i in draw])
        current = truth[index]
        if current.all() or not current.any():
            continue
        row = counts(index, selected)
        den = (masks[:, index] @ classes[:, index].T)[:, [0, 1, 0, 0, 1, 1]]
        delta = np.asarray([row[arm] - row[control] for arm, control in contrasts.values()])
        paired.append(np.divide(delta, den, out=np.full(delta.shape, np.nan), where=den > 0))
    if not paired:
        raise ValueError("no valid paired bootstrap draw")
    paired = np.asarray(paired)
    result = {}
    for j, (comparison, (arm, control)) in enumerate(contrasts.items()):
        result[comparison] = {}
        for i, name in enumerate(names):
            intervals = {}
            for k, measure in enumerate(MEASURES):
                values = paired[:, j, i, k]
                valid = values[np.isfinite(values)]
                intervals[measure] = (np.quantile(valid, [.025, .975]).tolist()
                                      if len(valid) >= max(20, draws // 2) else None)
            result[comparison][name] = {"candidate_minus_control_counts":
                                       dict(zip(MEASURES, (full[arm][i] - full[control][i]).tolist(), strict=True)),
                                       "paired_turn_rate_delta_95": intervals}
    return {"strata": slices, "deltas": result}


def paired_macro(rows, contrasts, *, draws=300):
    """Report equal-language recall deltas from probability-averaged arms."""
    rng = np.random.default_rng(91027)
    results = {}
    contrast_order = tuple(contrasts)
    names = sorted({name for pair in contrasts.values() for name in pair})
    for language, row in sorted(rows.items()):
        truth, groups = np.asarray(row["labels"], dtype=bool), np.asarray(row["turn_ordinals"])
        buckets = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        scores = {name: np.asarray(row["scores"][name]) for name in names}
        point, boot = None, []
        for i in range(draws + 1):
            index = np.arange(len(truth)) if i == 0 else np.concatenate([
                buckets[j] for j in rng.integers(len(buckets), size=len(buckets))
            ])
            y = truth[index]
            if not 0 < y.sum() < len(y):
                boot.append(np.full(len(contrasts), np.nan))
                continue
            recalls = {name: float(decisions(score[index], y)[y].mean()) for name, score in scores.items()}
            if i == 0:
                point = recalls
            else:
                boot.append([recalls[a] - recalls[b] for a, b in contrasts.values()])
        results[language] = point, np.asarray(boot)
    table = {}
    for index, name in enumerate(contrast_order):
        candidate, control = contrasts[name]
        point = np.mean([[row[0][candidate], row[0][control]] for row in results.values()], axis=0)
        sample = np.mean([row[1][:, index] for row in results.values()], axis=0)
        valid = sample[np.isfinite(sample)]
        table[name] = {"candidate": float(point[0]), "control": float(point[1]), "delta": float(point[0] - point[1]),
                       "paired_turn_delta_95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None}
    return table
