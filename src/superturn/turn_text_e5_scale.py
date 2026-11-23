"""Compare new synthetic English dialogues with matched English replay in E5."""

import argparse
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np

from . import text_cpu_probe, turn_metrics, turn_text_train
from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root
from .timed_text_length_trial import publisher_rows
from .timed_text_visibility import OPEN, number
from .turn_text_e5_quality import FROZEN_PUBLISHER_SHA
from .turn_metrics import auc
from .turn_text_scale_trial import AUDIT
from .turn_text_train import (BASE_SHA256, binary_metrics, digest, length_balanced_weights, load_fit, predict, read_dataset,
                              run as train, scorecard, tokenize_fit_input)

PARENT_SHA = "d41293a18ad13155ab12032bfa77c128979d92bf5a56578cde6839843c5d2240"
TOKENIZER_SHA = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
SEEDS = (17, 29, 43, 71, 101)
RECIPE = {"epochs": 2, "layers": 12, "batch_size": 32, "backbone": "e5-caller", "balance_length": True}
ARMS = ("b1", "replay", "scale")


def mixture(rows):
    return Counter(f"{row['language']}/{row['label']}" for row in rows)


def replay_rows(parent, extra):
    if not extra or any(row["language"] != "en" for row in extra):
        raise ValueError("scale must append English examples only")
    rng = random.Random(29)
    queues = {}
    for key, count in sorted(mixture(extra).items()):
        pool = [row for row in parent if f"{row['language']}/{row['label']}" == key]
        if not pool:
            raise ValueError(f"B1 has no training examples for {key}")
        remaining = count
        queues[key] = deque()
        while remaining:
            rng.shuffle(pool)
            take = min(remaining, len(pool))
            queues[key].extend(pool[:take])
            remaining -= take
    return [*parent, *(queues[f"{row['language']}/{row['label']}"].popleft() for row in extra)]


def read_test(dataset, manifest):
    path = dataset / "test.jsonl"
    if digest(path) != manifest["splits"]["test"]["sha256"]:
        raise ValueError("synthetic test differs from the audited scale export")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != manifest["splits"]["test"]["examples"]:
        raise ValueError("synthetic test row count differs from the audit")
    return rows


def packed_overlap(tokenizer, roles):
    seen = set()
    for role, rows in roles.items():
        packed = set()
        for row in rows:
            ids, mask = tokenize_fit_input(tokenizer, row["text"], "e5-caller")
            packed.add((tuple(ids), tuple(mask)))
        if seen & packed:
            raise ValueError(f"E5 packed input overlaps earlier synthetic roles in {role}")
        seen.update(packed)


def prepare(parent, scale, base, replay, audit_path):
    from transformers import AutoTokenizer

    b1, parent_sha = read_dataset(parent)
    s, scale_sha = read_dataset(scale)
    manifest = json.loads((scale / "manifest.json").read_text())
    audit = manifest["augmentation"]
    if (audit != json.loads(audit_path.read_text()) or parent_sha != PARENT_SHA
            or audit["parent_manifest_sha256"] != parent_sha
            or not audit["release_for_teacher_fit"] or audit["excluded_slots"]
            or manifest["dialogue_count"] != 20000 or audit["target_dialogues"] != 20000):
        raise ValueError("paired training requires the locked B1 and completed scale audit")
    if s["train"][:len(b1["train"])] != b1["train"]:
        raise ValueError("scale train must begin with the exact B1 training rows")
    r = replay_rows(b1["train"], s["train"][len(b1["train"]):])
    if any((a["language"], a["label"]) != (b["language"], b["label"]) for a, b in zip(r, s["train"], strict=True)):
        raise ValueError("paired arms must match language and class at every position")
    if digest(base / "model.safetensors") != BASE_SHA256 or digest(base / "tokenizer.json") != TOKENIZER_SHA:
        raise ValueError("the E5 base weights or tokenizer differ from the pinned model")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, use_fast=True)
    test = read_test(scale, manifest)
    roles = {**s, "test": test}
    slot_counts = {role: len({row["slot"] for row in rows if row["slot"].startswith("en/en-scale-v1/")})
                   for role, rows in roles.items()}
    if slot_counts != {"train": 18000, "validation": 1000, "test": 1000}:
        raise ValueError("scale does not have its complete locked dialogue roles")
    packed_overlap(tokenizer, roles)
    replay.mkdir(parents=True, exist_ok=True)
    train_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in r).encode()
    (replay / "train.jsonl").write_bytes(train_payload)
    (replay / "validation.jsonl").write_bytes((scale / "validation.jsonl").read_bytes())
    _write_json(replay / "manifest.json", {
        "scope": manifest["scope"], "model_input_fields": ["text"], "first_turn_marker": "---",
        "splits": {"train": {"sha256": sha256(train_payload).hexdigest(), "examples": len(r)},
                   "validation": manifest["splits"]["validation"]},
        "parent_manifest_sha256": parent_sha, "scale_manifest_sha256": scale_sha, "replay_rng_seed": 29,
    })
    _, replay_sha = read_dataset(replay)
    training = {"b1": b1["train"], "replay": r, "scale": s["train"]}
    receipts = {}
    for arm, rows in training.items():
        weights, bins = length_balanced_weights(rows)
        language_label_mass = defaultdict(float)
        for row, weight in zip(rows, weights, strict=True):
            language_label_mass[f"{row['language']}/{row['label']}"] += weight
        receipts[arm] = {"bins": bins, "weighted_language_label_mass": dict(language_label_mass)}
    frequencies = list(Counter(row["text"] for row in r if row["language"] == "en").values())
    return {
        "scope": "Paired E5 scale vs English replay, provisional diagnostics only",
        "b1_sha256": parent_sha, "scale_sha256": scale_sha, "replay_sha256": replay_sha,
        "tokenizer_sha256": TOKENIZER_SHA, "weights_sha256": BASE_SHA256,
        "publisher_freeze_sha256": FROZEN_PUBLISHER_SHA, "synthetic_test_sha256": manifest["splits"]["test"]["sha256"],
        "audit_sha256": digest(audit_path), "dialogue_slots": slot_counts,
        "implementation_sha256": {Path(path).name: digest(Path(path)) for path in
                                  (__file__, turn_text_train.__file__, text_cpu_probe.__file__, turn_metrics.__file__)},
        "packed_cross_role_overlap": 0, "paired_seeds": SEEDS,
        "recipes": {arm: {**RECIPE, "epochs": 5 if arm == "b1" else RECIPE["epochs"]} for arm in ARMS},
        "training_distributions": receipts, "replay_english_repetition_range": [min(frequencies), max(frequencies)],
        "steps_by_arm": {arm: (5 if arm == "b1" else RECIPE["epochs"]) *
                         ((len(rows) + RECIPE["batch_size"] - 1) // RECIPE["batch_size"])
                         for arm, rows in training.items()},
        "threshold": .5, "bootstrap": {"draws": 2000, "seed": 29, "clusters": "publisher turn then paired seeds"},
        "optimizer": {"name": "AdamW", "learning_rate": 3e-5, "weight_decay": .01,
                      "linear_warmup_fraction": .1, "clip_grad_norm": 1., "token_budget": turn_text_train.TOKEN_BUDGET,
                      "checkpoint": "fixed final epoch without data-dependent selection"},
        "decision": {"comparator": "maximum of the two baseline mean AUCs, reselected in each paired bootstrap draw",
                     "success_auc_delta": .03, "ci_lower_bound_above": 0, "miss_auc_delta_at_most": .01},
    }


def fit_pairs(root, datasets, base, receipt):
    roster = {}
    meter = Progress("e5-scale", len(ARMS) * len(receipt["paired_seeds"]), "fits")
    for seed in receipt["paired_seeds"]:
        for name in ARMS:
            destination = root / "runs" / f"turn-text-e5-scale-v1-{name}-{seed}"
            train(datasets[name], base, destination, seed=seed, **receipt["recipes"][name])
            checkpoint = destination / "checkpoint.pt"
            run = json.loads((destination / "receipt.json").read_text())
            if run["checkpoint_sha256"] != digest(checkpoint):
                raise ValueError("trained scale checkpoint differs from its fit receipt")
            roster[f"{name}/{seed}"] = {"name": destination.name, "sha256": run["checkpoint_sha256"]}
            meter.advance(1)
    meter.finish()
    return roster


def guard_recall(probabilities, mask, threshold):
    n = int(mask.sum())
    if not n:
        return {"shifts": 0, "recall": None, "wilson95": None}
    successes = int((probabilities[mask] >= threshold).sum())
    p = successes / n
    z = 1.959964
    center = p + z ** 2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return {"shifts": n, "detected": successes, "recall": p,
            "wilson95": [float(np.clip((center + sign * margin) / (1 + z ** 2 / n), 0, 1))
                         for sign in (-1, 1)]}


def scale_delta(scores, labels, rows, seeds):
    means = {arm: np.mean([auc(scores[arm][seed][rows], labels[rows]) for seed in seeds]) for arm in ARMS}
    return float(means["scale"] - max(means["replay"], means["b1"]))


def paired_interval(scores, labels, groups, settings):
    random = np.random.default_rng(settings["seed"])
    seed_count = len(scores["scale"])
    deltas = []
    for _ in range(settings["draws"]):
        rows = np.concatenate([groups[index] for index in random.integers(len(groups), size=len(groups))])
        if labels[rows].any() and (~labels[rows]).any():
            seeds = random.integers(seed_count, size=seed_count)
            deltas.append(scale_delta(scores, labels, rows, seeds))
    return np.quantile(deltas, [.025, .975]).tolist()


def evaluate(root, scale, base, receipt, roster):
    for item in roster.values():
        if digest(root / "runs" / item["name"] / "checkpoint.pt") != item["sha256"]:
            raise ValueError("a checkpoint changed after the seed roster was frozen")
    rows, freeze = publisher_rows(root)
    if digest(freeze) != receipt["publisher_freeze_sha256"]:
        raise ValueError("publisher view differs from the reviewed receipt")
    test = read_test(scale, json.loads((scale / "manifest.json").read_text()))
    labels = np.array([row["label"] for row in rows], dtype=bool)
    if (int((~labels).sum()), int(labels.sum())) != (527, 397):
        raise ValueError("publisher roster differs from the frozen diagnostic")
    texts = [row["text"] for row in rows]
    lengths = np.array([row["words"] for row in rows])
    length_bins = {label: (lengths >= low) & (lengths <= high)
                   for label, low, high in (("1-3", 1, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, np.inf))}
    ending = [value.split()[-1] for value in texts]
    guards = {
        "number_ending_shifts": labels & np.array([number(token) for token in ending]),
        "closed_1_to_3_word_shifts": labels & np.array([
            row["words"] <= 3 and token not in OPEN and not number(token) for row, token in zip(rows, ending, strict=True)
        ]),
    }
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["caller"]].append(i)
    scores = {name: [] for name in ARMS}
    trials = []
    for seed in receipt["paired_seeds"]:
        trial = {"seed": seed}
        for name in ARMS:
            item = roster[f"{name}/{seed}"]
            checkpoint = root / "runs" / item["name"] / "checkpoint.pt"
            tokenizer, model = load_fit(checkpoint, base, "cuda", receipt[f"{name}_sha256"])
            probabilities = np.asarray(predict(tokenizer, model, texts, "cuda"))
            synthetic = scorecard(predict(tokenizer, model, [row["text"] for row in test], "cuda"), test)["language/en"]
            del model
            scores[name].append(probabilities)
            publisher = binary_metrics(list(zip(probabilities, labels, strict=True)), receipt["threshold"])
            trial[name] = {
                "checkpoint_sha256": item["sha256"], "publisher": publisher, "synthetic_english_test": synthetic,
                "synthetic_minus_publisher_auc": synthetic["auc"] - publisher["auc"],
                "shift_guards": {key: guard_recall(probabilities, mask, receipt["threshold"])
                                 for key, mask in guards.items()},
                "length_bins": {key: {"holds": int((mask & ~labels).sum()), "shifts": int((mask & labels).sum()),
                                      "auc": auc(probabilities[mask], labels[mask])}
                                for key, mask in length_bins.items()},
            }
        trials.append(trial)
    delta = scale_delta(scores, labels, np.arange(len(rows)), range(len(trials)))
    interval = paired_interval(scores, labels, list(groups.values()), receipt["bootstrap"])
    gate = receipt["decision"]
    return {"trials": trials, "auc_delta": delta, "seed_and_publisher_turn_ci95": interval,
            "result": "scale_helped" if delta >= gate["success_auc_delta"] and interval[0] > gate["ci_lower_bound_above"]
                      else "no_scale_gain" if delta <= gate["miss_auc_delta_at_most"] else "inconclusive"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument("--reviewed-receipt-sha256")
    args = parser.parse_args()
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the paired scale comparison runs on Vast")
    base = root / "cache/turn-text-e5-small"
    scale = root / "cache/turn-text-en-scale-v1"
    replay = root / "cache/turn-text-en-scale-v1-replay"
    path = root / "reports/e5-scale-preregistration.json"
    b1 = root / "cache/turn-text-focused-en-v1"
    receipt = prepare(b1, scale, base, replay, root / "reports" / AUDIT)
    if args.action == "prepare":
        _write_json(path, receipt)
        print(digest(path), flush=True)
        return
    approved = json.loads(path.read_text())
    if (digest(path) != args.reviewed_receipt_sha256
            or json.loads(json.dumps(receipt)) != approved):
        raise ValueError("the current scale trial differs from the reviewed preregistration")
    result_path = root / "reports/publisher-timed-e5-scale-v1.json"
    if result_path.is_file():
        if json.loads(result_path.read_text())["preregistration_sha256"] != args.reviewed_receipt_sha256:
            raise ValueError("existing score belongs to a different scale comparison")
        return
    roster_path = root / "reports/e5-scale-checkpoint-roster.json"
    if not roster_path.is_file():
        roster = fit_pairs(root, {"b1": b1, "replay": replay, "scale": scale}, base, receipt)
        _write_json(roster_path, {"preregistration_sha256": args.reviewed_receipt_sha256, "checkpoints": roster})
    locked = json.loads(roster_path.read_text())
    expected = {f"{name}/{seed}" for name in ARMS for seed in receipt["paired_seeds"]}
    if locked["preregistration_sha256"] != args.reviewed_receipt_sha256 or set(locked["checkpoints"]) != expected:
        raise ValueError("seed roster differs from the reviewed comparison")
    result = evaluate(root, scale, base, receipt, locked["checkpoints"])
    _write_json(result_path, {**result, "preregistration_sha256": args.reviewed_receipt_sha256,
                             "checkpoint_roster_sha256": digest(roster_path)})


if __name__ == "__main__":
    main()
