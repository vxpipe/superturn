"""Compare capped length balancing with unweighted replay at fixed English coverage."""

import argparse
from hashlib import sha256
import json

import numpy as np

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root
from .timed_text_pause_diagnostic import FREEZE, pack_rows
from .turn_metrics import auc
from .turn_text_train import binary_metrics, digest, load_fit, predict, run as train

SEEDS = (17, 29, 43, 71, 101)


def report_path(root, *, score_only):
    filename = "publisher-timed-english-length-trial"
    suffix = "-ranking" if score_only else ""
    return root / "reports" / f"{filename}{suffix}.json"


def short_hold_delta(rows, unweighted, balanced):
    short = [index for index, row in enumerate(rows)
             if not row["label"] and 300 <= row["gap_ms"] < 800 and row["words"] >= 8]
    turns = {}
    for index in short:
        turns.setdefault(rows[index]["caller"], []).append(index)
    difference = (np.asarray(balanced) >= .5).astype(int) - (np.asarray(unweighted) >= .5)
    grouped = np.stack([difference[:, indices].sum(axis=1) for indices in turns.values()], axis=1)
    counts = np.array([len(indices) for indices in turns.values()])
    rng = np.random.default_rng(29)
    draw_seeds = rng.integers(len(SEEDS), size=(4000, len(SEEDS)))
    draw_turns = rng.integers(len(turns), size=(4000, len(turns)))
    resampled = grouped[draw_seeds[:, :, None], draw_turns[:, None, :]].mean(axis=1).sum(axis=1)
    return {"holds": len(short), "turns": len(turns), "delta": float(grouped.sum() / (len(SEEDS) * len(short))),
            "paired_seed_and_turn_ci95": np.quantile(resampled / counts[draw_turns].sum(axis=1), [.025, .975]).tolist()}


def matched_ranking(rows, controls, balanced):
    labels = np.array([row["label"] for row in rows], dtype=bool)
    cell = np.array([not row["label"] and row["words"] >= 8 and 300 <= row["gap_ms"] < 800 for row in rows])
    lengths = np.digitize([row["words"] for row in rows], [4, 8, 16])
    target_recall = float(np.mean(np.asarray(controls)[:, labels] >= .5))
    turns = {}
    for index, row in enumerate(rows):
        turns.setdefault(row["caller"], []).append(index)
    groups = list(turns.values())
    full = np.arange(len(rows))

    def compare(scores, indices):
        truth = labels[indices]
        vals = scores[indices]
        shifts = vals[truth]
        hits = min(len(shifts), max(1, int(np.rint(target_recall * len(shifts)))))
        threshold = np.sort(shifts)[-hits]
        eligible = truth | cell[indices]
        numerator, denominator = 0., 0
        for band in range(4):
            mask = lengths[indices] == band
            positives = int(truth[mask].sum())
            pairs = positives * (int(mask.sum()) - positives)
            if pairs:
                numerator += pairs * auc(vals[mask], truth[mask])
                denominator += pairs
        return np.array([np.mean(vals[cell[indices]] >= threshold), auc(vals[eligible], truth[eligible]),
                         numerator / denominator, np.mean(shifts >= threshold)])

    replay, trial = np.asarray(controls), np.asarray(balanced)
    observed = np.mean([compare(a, full) for a in replay], axis=0), np.mean(
        [compare(a, full) for a in trial], axis=0)
    rng = np.random.default_rng(29)
    draws = []
    for _ in range(1000):
        selected = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        indices = rng.integers(len(SEEDS), size=len(SEEDS))
        draws.append(np.mean([compare(trial[i], selected) - compare(replay[i], selected) for i in indices], axis=0))
    names = ("matched_short_hold_false_complete", "cell_vs_shift_auc", "within_length_pair_weighted_auc",
             "matched_actual_shift_recall")
    report = {}
    for i, name in enumerate(names):
        control, intervention = float(observed[0][i]), float(observed[1][i])
        report[name] = {"replay": control, "balanced": intervention, "delta": intervention - control,
                        "paired_seed_turn_ci95": np.quantile(np.asarray(draws)[:, i], [.025, .975]).tolist()}
    return {"target_control_shift_recall": target_recall, "metrics": report}


def run(root, *, score_only=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("publisher trial inputs stay on Vast")
    freeze_path = root / "reports" / FREEZE
    freeze = json.loads(freeze_path.read_text())
    rows, _, membership = pack_rows(root)
    if (freeze["membership_sha256"] != membership
            or freeze["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()):
        raise ValueError("publisher development reference changed")
    rows = [row for row in rows if row["has_previous"]]
    dataset = root / "cache/turn-text-focused-en-v1"
    base = root / "cache/turn-text-qwen3-06b"
    corpus_hash = digest(dataset / "manifest.json")
    bands = {
        "all": list(range(len(rows))),
        "short_8_plus_holds": [i for i, row in enumerate(rows)
                               if not row["label"] and row["words"] >= 8 and 300 <= row["gap_ms"] < 800],
        "long_gap": [i for i, row in enumerate(rows) if row["gap_ms"] >= 1500],
    }
    if tuple(map(len, bands.values())) != (924, 327, 436):
        raise ValueError("unexpected publisher trial slice counts")
    trials = {"replay": [], "balanced": []}
    arm_scores = {"replay": [], "balanced": []}
    meter = Progress("english-length-trial", 2 * len(SEEDS), "fits")
    for seed in SEEDS:
        for arm in trials:
            output = root / "runs" / f"turn-text-qwen3-length-{arm}-{seed}"
            if not score_only:
                train(dataset, base, output, seed=seed, backbone="qwen3", balance_length=arm == "balanced")
            checkpoint = output / "checkpoint.pt"
            tokenizer, model = load_fit(checkpoint, base, "cuda", corpus_hash)
            scores = predict(tokenizer, model, [row["text"] for row in rows], "cuda")
            del model
            arm_scores[arm].append(scores)
            metrics = {name: binary_metrics([(scores[i], rows[i]["label"]) for i in indices])
                       for name, indices in bands.items()}
            trials[arm].append({"seed": seed, "checkpoint_sha256": digest(checkpoint), "views": metrics})
            meter.advance(1)
    meter.finish()
    delta = short_hold_delta(rows, arm_scores["replay"], arm_scores["balanced"])
    rollup = {}
    for arm, fits in trials.items():
        rollup[arm] = {}
        for name in bands:
            view = [item["views"][name] for item in fits]
            rollup[arm][name] = {
                key: {"mean": float(np.mean(values)), "range": [min(values), max(values)]}
                for key in view[0] if view[0][key] is not None
                if (values := [item[key] for item in view])
            }
    def diff(view, metric):
        return rollup["balanced"][view][metric]["mean"] - rollup["replay"][view][metric]["mean"]

    report = {"scope": "fixed English development at 0.5, paired seeds, no calibration or test",
              "freeze_sha256": digest(freeze_path), "dataset_manifest_sha256": corpus_hash,
              "seeds": SEEDS, "trials": trials, "rollup": rollup, "short_hold_comparison": delta,
              "gate": {"short_hold_delta": delta["delta"] <= -.05 and delta["paired_seed_and_turn_ci95"][1] < 0,
                       "shift_recall": diff("all", "shift_recall") >= -.02,
                       "long_gap_auc": diff("long_gap", "auc") >= -.01,
                       "all_auc": diff("all", "auc") >= -.005}}
    if score_only:
        from .timed_text_visibility import number

        ranking = matched_ranking(rows, arm_scores["replay"], arm_scores["balanced"])
        measures = ranking["metrics"]
        short = measures["matched_short_hold_false_complete"]
        cell = measures["cell_vs_shift_auc"]
        within = measures["within_length_pair_weighted_auc"]
        report["matched_ranking"] = ranking
        report["ranking_gate"] = {
            "matched_short_hold": short["delta"] <= -.05 and short["paired_seed_turn_ci95"][1] < 0,
            "cell_auc": cell["delta"] >= .03 and cell["paired_seed_turn_ci95"][0] > 0,
            "within_length_auc": within["delta"] >= -.005,
        }
        mask_path = root / "cache/publisher-timed-english-deadline-mask.json"
        audit_path = root / "reports/publisher-timed-english-deadline-audit.json"
        mask = json.loads(mask_path.read_text())
        deadline = json.loads(audit_path.read_text())
        excluded = {item["rank"] for item in mask["excluded"]}
        if (mask["freeze_sha256"] != digest(freeze_path)
                or deadline["excluded_ranks_sha256"] != digest(mask_path) or len(excluded) != 159):
            raise ValueError("publisher deadline sensitivity mask changed")
        selected = [i for i, row in enumerate(rows) if row["rank"] not in excluded]
        robust_rows = [rows[i] for i in selected]
        robust = {arm: np.asarray(scores)[:, selected] for arm, scores in arm_scores.items()}
        secondary = matched_ranking(robust_rows, robust["replay"], robust["balanced"])
        robust_short = short_hold_delta(robust_rows, robust["replay"], robust["balanced"])
        basic_guards = {}
        for name in ("all", "long_gap"):
            indices = [i for i in bands[name] if rows[i]["rank"] not in excluded]
            basic_guards[name] = {}
            for arm, scores in arm_scores.items():
                views = [binary_metrics([(values[i], rows[i]["label"]) for i in indices]) for values in scores]
                basic_guards[name][arm] = {
                    key: float(np.mean([view[key] for view in views]))
                    for key in ("shift_recall", "auc", "balanced_accuracy", "false_complete")
                }
        sensitivity = secondary["metrics"]
        report["deadline_sensitivity"] = {
            "excluded_at_deadline": len(excluded), "number_ending_holds_excluded": deadline["excluded_number_ending_holds"],
            "matched_ranking": secondary, "short_hold_comparison": robust_short, "guards_at_0_5": basic_guards,
            "checks": {
                "short_hold_at_0_5": delta["delta"] < 0 and robust_short["delta"] <= delta["delta"] / 2,
                "matched_short_hold": short["delta"] < 0
                and sensitivity["matched_short_hold_false_complete"]["delta"] <= short["delta"] / 2,
                "cell_auc": cell["delta"] > 0 and sensitivity["cell_vs_shift_auc"]["delta"] >= cell["delta"] / 2,
                "within_length_auc": sensitivity["within_length_pair_weighted_auc"]["delta"] >= -.005,
                "shift_recall": basic_guards["all"]["balanced"]["shift_recall"]
                >= basic_guards["all"]["replay"]["shift_recall"] - .02,
                "all_auc": basic_guards["all"]["balanced"]["auc"] >= basic_guards["all"]["replay"]["auc"] - .005,
                "long_gap_auc": basic_guards["long_gap"]["balanced"]["auc"]
                >= basic_guards["long_gap"]["replay"]["auc"] - .01,
            },
        }
        numeric = [i for i, row in enumerate(rows) if number(row["text"].split()[-1]) and
                   (row["gap_ms"] >= 1500 if row["label"] else 300 <= row["gap_ms"] < 800)]
        if (sum(not rows[i]["label"] for i in numeric), sum(rows[i]["label"] for i in numeric)) != (77, 67):
            raise ValueError("number-ending cohort changed")
        report["number_endings"] = {}
        for mode, indices in (("full", numeric), ("alignment_sensitive_rows_excluded",
                                                  [i for i in numeric if rows[i]["rank"] not in excluded])):
            report["number_endings"][mode] = {
                arm: [binary_metrics([(scores[i], rows[i]["label"]) for i in indices]) for scores in fits]
                for arm, fits in arm_scores.items()
            }
    _write_json(report_path(root, score_only=score_only), report)
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true", help="replay completed checkpoints and add ranking guards")
    run(data_root(), score_only=parser.parse_args().score_only)
