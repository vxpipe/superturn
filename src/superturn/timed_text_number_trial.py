"""Train and measure the final matched English number-dictation pilot."""

from hashlib import sha256
import json

import numpy as np

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root
from .timed_text_length_trial import SEEDS
from .timed_text_pause_diagnostic import FREEZE, pack_rows
from .timed_text_visibility import OPEN, number
from .turn_text_number_layout import length_bin
from .turn_text_train import binary_metrics, digest, length_balanced_weights, load_fit, predict, read_dataset, run as train


def rank_auc(values, truth):
    negatives = np.sort(values[~truth])
    positives = values[truth]
    left = np.searchsorted(negatives, positives, side="left")
    right = np.searchsorted(negatives, positives, side="right")
    return float(((left + right) / (2 * len(negatives))).mean())


def measure(rows, control, trial):
    truth = np.asarray([row["label"] for row in rows], dtype=bool)
    ends = [row["text"].split()[-1] for row in rows]
    numeric = np.asarray([number(word) and (row["gap_ms"] >= 1500 if row["label"] else 300 <= row["gap_ms"] < 800)
                          for row, word in zip(rows, ends, strict=True)])
    numeric_holds = numeric & ~truth
    numeric_shifts = numeric & truth
    bins = np.array([length_bin(row["words"]) for row in rows])
    closed_short_shifts = truth & (bins == "1-3") & np.asarray([word not in OPEN and not number(word) for word in ends])
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["caller"], []).append(i)
    groups = list(groups.values())

    def metrics(scores, indices, target):
        labels = truth[indices]
        values = scores[indices]
        positives = values[labels]
        n = max(1, round(target * len(positives)))
        decision = values >= np.sort(positives)[-n]
        cell = numeric_holds[indices]
        negative = cell | labels
        same_ends = numeric[indices]
        return (np.mean(decision[cell]),
                rank_auc(values[negative], labels[negative]),
                rank_auc(values[same_ends], labels[same_ends]))

    pool = np.arange(len(rows))
    recall = float(np.mean(control[:, truth] >= .5))
    old = np.mean([metrics(scores, pool, recall) for scores in control], axis=0)
    new = np.mean([metrics(scores, pool, recall) for scores in trial], axis=0)
    rng = np.random.default_rng(29)
    draws = []
    for _ in range(2000):
        indices = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        seeds = rng.integers(len(SEEDS), size=len(SEEDS))
        positive_rows = indices[truth[indices]]
        target = float(np.mean(control[seeds][:, positive_rows] >= .5))
        draws.append(np.mean([np.subtract(metrics(trial[k], indices, target), metrics(control[k], indices, target))
                              for k in seeds], axis=0))
    intervals = np.quantile(draws, [.025, .975], axis=0)
    labels = ("number_hold_false_complete", "cell_vs_all_shifts_auc", "number_hold_vs_number_shift_auc")
    matched = {name: {"control": float(old[i]), "candidate": float(new[i]), "delta": float(new[i] - old[i]),
                      "paired_seed_turn_ci95": intervals[:, i].tolist()} for i, name in enumerate(labels)}
    masks = {"all_shifts": truth, "number_holds": numeric_holds, "number_shifts": numeric_shifts,
             "closed_short_shifts": closed_short_shifts}
    for band in ("1-3", "4-7", "8-15", "16+"):
        masks[f"number_holds/{band}"] = numeric_holds & (bins == band)
        masks[f"number_shifts/{band}"] = numeric_shifts & (bins == band)
    by_group = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        replays = []
        for scores in (control, trial):
            hit_count = max(1, round(recall * int(truth.sum())))
            cutoffs = np.sort(scores[:, truth], axis=1)[:, -hit_count, None]
            replays.append(float(np.mean(scores[:, mask] >= cutoffs)))
        by_group[name] = {"n": int(mask.sum()), "control_positive_rate": replays[0],
                          "candidate_positive_rate": replays[1], "delta": replays[1] - replays[0]}
    views = {}
    for name, mask in (("all", np.ones(len(rows), dtype=bool)),
                       ("long_gap", np.array([row["gap_ms"] >= 1500 for row in rows]))):
        view = {}
        for arm, values in (("weighted", control), ("numbers", trial)):
            fits = [binary_metrics(list(zip(scores[mask].tolist(), truth[mask].tolist(), strict=True))) for scores in values]
            view[arm] = {key: {"mean": float(np.mean([fit[key] for fit in fits])),
                               "range": [min(fit[key] for fit in fits), max(fit[key] for fit in fits)]}
                         for key in fits[0]}
        views[name] = view
    return {"target_shift_recall": recall, "matched": matched, "positive_rates": by_group, "scorecard": views}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the number trial uses staged development inputs only on Vast")
    parent = root / "cache/turn-text-focused-en-v1"
    variant = root / "cache/turn-text-number-pilot-v1"
    old, old_hash = read_dataset(parent)
    new, new_hash = read_dataset(variant)
    if length_balanced_weights(old["train"])[0] != length_balanced_weights(new["train"])[0]:
        raise ValueError("weighted control and numeric variant have different loss factors")
    freeze_path = root / "reports" / FREEZE
    freeze = json.loads(freeze_path.read_text())
    rows, _, membership = pack_rows(root)
    if (freeze["membership_sha256"] != membership
            or freeze["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()):
        raise ValueError("publisher number trial reference changed")
    rows = [row for row in rows if row["has_previous"]]
    base = root / "cache/turn-text-qwen3-06b"
    length_result = json.loads((root / "reports/publisher-timed-english-length-trial-ranking.json").read_text())
    expected = {fit["seed"]: fit["checkpoint_sha256"] for fit in length_result["trials"]["balanced"]}
    mask_path = root / "cache/publisher-timed-english-deadline-mask.json"
    mask = json.loads(mask_path.read_text())
    audit = json.loads((root / "reports/publisher-timed-english-deadline-audit.json").read_text())
    if mask["freeze_sha256"] != digest(freeze_path) or audit["excluded_ranks_sha256"] != digest(mask_path):
        raise ValueError("number trial timing mask changed")
    excluded = {item["rank"] for item in mask["excluded"]}
    holdout_path = variant / "number-test.jsonl"
    meta = json.loads((variant / "manifest.json").read_text())["number_pilot"]["heldout"]["test"]
    if digest(holdout_path) != meta["sha256"]:
        raise ValueError("synthetic held-out file differs from its receipt")
    challenge = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    challenge = [row for row in challenge if number(row["text"].split()[-1])]
    texts = [row["text"] for row in rows + challenge]
    predictions, held, fingerprints = [[], []], [[], []], [[], []]
    progress = Progress("english-number-trial", len(SEEDS), "paired fits")
    for seed in SEEDS:
        original = root / "runs" / f"turn-text-qwen3-length-balanced-{seed}/checkpoint.pt"
        if digest(original) != expected[seed]:
            raise ValueError("weighted checkpoint differs from the paired control")
        output = root / "runs" / f"turn-text-qwen3-number-{seed}"
        train(variant, base, output, backbone="qwen3", seed=seed, balance_length=True)
        for arm, (checkpoint, corpus_hash) in enumerate(((original, old_hash), (output / "checkpoint.pt", new_hash))):
            tokenizer, model = load_fit(checkpoint, base, "cuda", corpus_hash)
            scores = predict(tokenizer, model, texts, "cuda")
            del model
            fingerprints[arm].append(digest(checkpoint))
            predictions[arm].append(scores[:len(rows)])
            held[arm].append(binary_metrics([(score, row["label"] == "shift")
                                              for score, row in zip(scores[len(rows):], challenge, strict=True)]))
        progress.advance(1)
    progress.finish()
    control, trial = np.array(predictions)
    core = measure(rows, control, trial)
    selected = [i for i, row in enumerate(rows) if row["rank"] not in excluded]
    clean = measure([rows[i] for i in selected], control[:, selected], trial[:, selected])
    primary = core["matched"]
    robust = clean["matched"]
    false_complete = primary["number_hold_false_complete"]
    cell_auc = primary["cell_vs_all_shifts_auc"]
    def held_delta(metric):
        return float(np.mean([b[metric] - a[metric] for a, b in zip(*held, strict=True)]))

    def recall_guards(view):
        rates = view["positive_rates"]
        return (all(result["delta"] >= -.05 for name, result in rates.items() if name.startswith("number_shifts"))
                and rates["closed_short_shifts"]["delta"] >= -.03)

    def overall_guards(view):
        all_rows = view["scorecard"]["all"]
        long_gap = view["scorecard"]["long_gap"]
        return (all_rows["numbers"]["auc"]["mean"] >= all_rows["weighted"]["auc"]["mean"] - .005
                and all_rows["numbers"]["shift_recall"]["mean"] >= all_rows["weighted"]["shift_recall"]["mean"] - .02
                and long_gap["numbers"]["auc"]["mean"] >= long_gap["weighted"]["auc"]["mean"] - .01)

    report = {"scope": "provisional English publisher development, last synthetic number round",
              "seeds": SEEDS, "checkpoint_sha256": fingerprints, "freeze_sha256": digest(freeze_path),
              "full": core, "timing_mask": clean, "excluded_rows": len(excluded),
              "synthetic_heldout": {"control": held[0], "candidate": held[1],
                                    "false_complete_delta": held_delta("false_complete"),
                                    "recall_delta": held_delta("shift_recall")},
              "gate": {
                  "number_hold_at_matched_recall": false_complete["delta"] <= -.08
                  and false_complete["paired_seed_turn_ci95"][1] < 0,
                  "cell_rank_gain": cell_auc["delta"] >= .03 and cell_auc["paired_seed_turn_ci95"][0] > 0,
                  "number_only_auc_guard": primary["number_hold_vs_number_shift_auc"]["delta"] >= 0,
                  "recall_guards": recall_guards(core) and recall_guards(clean),
                  "overall_guards": overall_guards(core) and overall_guards(clean),
                  "mask_keeps_effect": false_complete["delta"] < 0 and cell_auc["delta"] > 0
                  and robust["number_hold_false_complete"]["delta"] <= false_complete["delta"] / 2
                  and robust["cell_vs_all_shifts_auc"]["delta"] >= cell_auc["delta"] / 2,
                  "synthetic_learnability": held_delta("false_complete") <= -.30 and held_delta("shift_recall") >= -.10,
              }}
    _write_json(root / "reports/publisher-timed-english-number-trial.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
