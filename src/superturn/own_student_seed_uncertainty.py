"""Measure paired seed and turn uncertainty on archived width predictions."""

from itertools import permutations
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from .ami_calibration import _write_json
from .failure_strata import decisions
from .language_failure_audit import read_receipt, read_verified
from .own_student_width_probe import model_name
from .own_teacher_feature_student import SEEDS
from .sources import data_root


def paired_seed_turns(rows, candidate, control, *, draws=1000, random_seed=7429):
    """Resample paired model seeds globally and turns independently per language."""
    if len(candidate) != len(control) or len(candidate) < 2:
        raise ValueError("expected at least two paired model seeds")
    rng = np.random.default_rng(random_seed)
    seed_draws = rng.integers(len(candidate), size=(draws, len(candidate)))
    seed_deltas, turn_deltas = [], []
    for language, row in sorted(rows.items()):
        truth = np.asarray(row["labels"], dtype=bool)
        turns = np.asarray(row["turn_ordinals"])
        c, r = (np.asarray([row["scores"][name] for name in names], dtype=float)
                for names in (candidate, control))
        if (turns.shape != truth.shape or c.shape != (len(candidate), len(truth))
                or r.shape != c.shape or not np.isfinite(c).all() or not np.isfinite(r).all()
                or not 0 < truth.sum() < len(truth)):
            raise ValueError(f"invalid paired outcomes for {language}")
        buckets = [np.flatnonzero(turns == group) for group in np.unique(turns)]

        def difference(index):
            y = truth[index]
            if not 0 < y.sum() < len(y):
                return np.full(len(candidate), np.nan)
            return np.asarray([float(decisions(left[index], y)[y].mean() - decisions(right[index], y)[y].mean())
                               for left, right in zip(c, r, strict=True)])

        seed_deltas.append(difference(np.arange(len(truth))))
        turn_deltas.append(np.stack([difference(np.concatenate([buckets[i] for i in assignment]))
                                     for assignment in rng.integers(len(buckets), size=(draws, len(buckets)))]))
    point_seed = np.mean(seed_deltas, axis=0)
    turn_seed = np.mean(turn_deltas, axis=0)

    def interval(values):
        valid = values[np.isfinite(values)]
        if len(valid) < draws // 2:
            raise ValueError("too few resamples had both classes in every language")
        return np.quantile(valid, [.025, .975]).tolist()

    return {"delta": float(point_seed.mean()), "seed_deltas": point_seed.tolist(), "draws": draws,
            "turn_only_95": interval(turn_seed.mean(1)), "seed_only_95": interval(point_seed[seed_draws].mean(1)),
            "paired_seed_and_turn_95": interval(np.take_along_axis(turn_seed, seed_draws, axis=1).mean(1))}


def rank_evidence(recall, fidelity):
    if not np.ptp(recall) or not np.ptp(fidelity):
        return {"spearman": None, "one_sided_exact_p": None, "leave_one_fit_out_range": None}
    value = float(spearmanr(recall, fidelity).statistic)
    recall_ranks, fidelity_ranks = rankdata(recall), rankdata(fidelity)
    observed = np.dot(recall_ranks, fidelity_ranks)
    null = [np.dot(recall_ranks, order) for order in permutations(fidelity_ranks)]
    leave_one_out = []
    for i in range(len(recall)):
        subset = np.arange(len(recall)) != i
        if np.ptp(recall[subset]) and np.ptp(fidelity[subset]):
            leave_one_out.append(float(spearmanr(recall[subset], fidelity[subset]).statistic))
    return {"spearman": value, "one_sided_exact_p": float(np.mean(np.asarray(null) >= observed)),
            "leave_one_fit_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None}


def fidelity_by_source(epoch):
    return {source: stats["by_deadline"]["300"]["spearman"] for source, stats in epoch["sources"].items()}


def fidelity_rank(width, fresh, replay):
    rows = []
    for label, report, seeds, scores in (
            ("parent", width, SEEDS, replay["original_seed_macro_matched_5"]),
            ("fresh_control", fresh, fresh["new_seeds"], replay["fresh_control_seed_macro_matched_5"])):
        for seed in seeds:
            name = model_name(1_000_000, seed) if label == "parent" else f"control-{seed}"
            fit = report["fits"][name]
            selected, last = fit["selected"], fit["history"][-1]
            sources, last_sources = fidelity_by_source(selected), fidelity_by_source(last)
            score = float(np.mean(list(sources.values())))
            if not np.isclose(score, selected["selection_key"][0], rtol=0, atol=1e-12):
                raise ValueError(f"{name} uses a different fidelity rank")
            rows.append({"arm": label, "seed": seed, "epoch": selected["epoch"], "matched_5_macro": scores[str(seed)],
                         "fidelity_source_macro_spearman": score, "fidelity_per_source": sources,
                         "last_epoch": last["epoch"], "last_epoch_fidelity_per_source": last_sources,
                         "last_epoch_fidelity_source_macro_spearman": float(np.mean(list(last_sources.values())))})
    axis = np.asarray([row["matched_5_macro"] for row in rows])
    selected = rank_evidence(axis, np.asarray([row["fidelity_source_macro_spearman"] for row in rows]))
    last = rank_evidence(axis, np.asarray([row["last_epoch_fidelity_source_macro_spearman"] for row in rows]))
    value = selected["spearman"]
    return {"fits": rows, "selected_epoch": selected, "last_epoch": last,
            "clears_exploratory_0_8_bar": value is not None and value >= .8,
            "macro_across_all_six_sample_sd": float(axis.std(ddof=1)),
            "scope": "six stage-comparable fits; held-source fidelity served epoch selection"}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the paired seed audit reads staged Vast reports")
    width, width_sha = read_receipt(root, "own-student-width-probe")
    replay, replay_sha = read_receipt(root, "own-student-seed-replay")
    fresh, fresh_sha = read_receipt(root, "own-student-ensemble-distillation")
    if replay["width_sha256"] != width_sha or replay["ensemble_distillation_sha256"] != fresh_sha:
        raise ValueError("the exact replay used other width or fresh-control reports")
    views = json.loads(read_verified(root, width["paired_language_scores"]).read_text())
    size_names = {size: tuple(model_name(size, seed) for seed in SEEDS) for size in (150000, 1000000)}
    interval = paired_seed_turns(views, size_names[1000000], size_names[150000])
    score = width["summary"]["matched_5pct_macro"]
    if not np.isclose(interval["delta"], score["1000000"] - score["150000"], rtol=0, atol=1e-12):
        raise ValueError("the paired replay changed the original width point estimate")
    result = {"width_receipt_sha256": width_sha, "seed_replay_receipt_sha256": replay_sha,
              "fresh_control_receipt_sha256": fresh_sha, "width_seeds": SEEDS,
              "scope": "development reanalysis only; width used just three paired seeds",
              "matched_5_width_gain": interval, "fidelity_selector": fidelity_rank(width, fresh, replay)}
    _write_json(root / "reports/own-student-seed-uncertainty.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(data_root())
