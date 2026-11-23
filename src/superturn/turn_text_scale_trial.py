"""Audit en-scale-v1 on Vast, build training splits and score the frozen English diagnostic."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root
from .timed_text_length_trial import publisher_rows, report_path
from .timed_text_visibility import OPEN, number
from .turn_generation import CATALOGUE, plan_checksum
from .turn_metrics import auc
from .turn_text_context_supplement import dev_blocklist, ngrams
from .turn_text_scale_generation import BUDGET, SCALE_VERSION, build_plan, clean_scale_row, scale_examples
from .turn_text_train import binary_metrics, digest, load_fit, predict, read_dataset, run as train
from .turn_training_dataset import input_key

SEEDS = (17, 29, 43)
AUDIT = "en-scale-v1-pretraining-audit.json"
RESULT = "publisher-timed-english-scale-teacher.json"


def evaluate_dialogues(rows, specs, base_splits, blocked):
    seen = {item["text"] for values in base_splits.values() for item in values}
    excluded, accepted = {}, defaultdict(list)
    for row in rows:
        slot = row["slot"]
        spec = specs[slot]
        caller = "‖".join(part for part in (row["caller_before_pause"], row["caller_after_pause"]) if part)
        fields = clean_scale_row({**row, "caller": caller}, spec)
        if fields is None or any(row[key] != value for key, value in (spec | fields).items()):
            excluded[slot] = "local_validator"
            continue
        if any(gram in blocked for text in (row["previous_agent"], caller.replace("‖", " "), row["next_agent"])
               for gram in ngrams(text)):
            excluded[slot] = "benchmark_eight_word_overlap"
            continue
        examples = [{"slot": slot, "label": item["label"], "language": "en", "text": input_key(item)}
                    for item in scale_examples(row)]
        keys = {item["text"] for item in examples}
        if len(keys) != len(examples) or seen & keys:
            excluded[slot] = "duplicate_model_input"
            continue
        seen.update(keys)
        accepted[row["split"]].extend(examples)
    return accepted, excluded


def audit_export(root, generation, parent, destination):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("benchmark-overlap validation runs only on Vast")
    source = generation / "dialogues.jsonl"
    receipt_path = generation / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    plan = build_plan(json.loads(CATALOGUE.read_text()))
    specs = {row["slot"]: row for row in plan}
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    spent = receipt["estimated_spend_including_unknown_requests_usd"]
    if (receipt["plan_sha256"] != plan_checksum(plan) or receipt["dialogues_sha256"] != digest(source)
            or not 0 <= spent <= BUDGET or receipt["accepted_dialogues"] != len(rows)
            or len({row["slot"] for row in rows}) != len(rows)
            or set(row["slot"] for row in rows) - specs.keys()):
        raise ValueError("generated rows or spend do not match the locked scale plan")
    base_splits, parent_sha = read_dataset(parent)
    additions, excluded = evaluate_dialogues(rows, specs, base_splits, dev_blocklist(root))
    retained = [row for row in rows if row["slot"] not in excluded]
    role_scenarios = {role: sorted({row["scenario"] for row in retained if row["split"] == role})
                      for role in ("train", "validation", "test")}
    train_scenarios = {slot[2] if slot[1] == "focused-v1" else slot[1]
                       for row in base_splits["train"] if (slot := row["slot"].split("/"))}
    eval_scenarios = set(role_scenarios["validation"] + role_scenarios["test"])
    if (eval_scenarios & (train_scenarios | set(role_scenarios["train"]))
            or set(role_scenarios["validation"]) & set(role_scenarios["test"])):
        raise ValueError("a scale scenario crossed the held-out roles")
    audit = {
        "scope": "Local validators and exact eight-word comparison on private Vast development sources",
        "plan_sha256": receipt["plan_sha256"], "dialogues_sha256": receipt["dialogues_sha256"],
        "parent_manifest_sha256": parent_sha, "spent_usd": spent, "target_dialogues": len(plan),
        "accepted_before_audit": len(rows), "accepted_after_audit": len(retained),
        "exclusions_by_reason": dict(Counter(excluded.values())), "excluded_slots": sorted(excluded),
        "retained_by_family_and_role": dict(Counter(f"{row['scale_family']}/{row['split']}" for row in retained)),
        "role_scenarios": role_scenarios,
        "release_for_teacher_fit": not excluded and len(rows) == len(plan) and receipt["stop"] == "complete",
    }
    _write_json(root / "reports" / AUDIT, audit)
    if not audit["release_for_teacher_fit"]:
        print(json.dumps({key: value for key, value in audit.items() if key != "excluded_slots"}, indent=2))
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix="scale-en-") as work:
        output = Path(work) / "dataset"
        output.mkdir()
        splits = {}
        for role in role_scenarios:
            values = [*base_splits.get(role, []), *additions[role]]
            payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in values).encode()
            (output / f"{role}.jsonl").write_bytes(payload)
            splits[role] = {"sha256": sha256(payload).hexdigest(), "examples": len(values),
                            "labels": dict(Counter(row["label"] for row in values)),
                            "languages": dict(Counter(row["language"] for row in values))}
        _write_json(output / "manifest.json", {
            "scope": "synthetic training and diagnostic validation", "model_input_fields": ["text"],
            "first_turn_marker": "---", "splits": splits, "augmentation": audit,
            "generation_receipt_sha256": digest(receipt_path), "dialogue_count": len(rows),
        })
        read_dataset(output)
        if destination.exists():
            if digest(destination / "manifest.json") != digest(output / "manifest.json"):
                raise ValueError("the existing scale export has different inputs")
            read_dataset(destination)
        else:
            output.rename(destination)
    print(json.dumps({"verified_synthetic_dialogues": len(rows), "splits": splits}, indent=2))
    return True


def paired_interval(reference, scaled, labels, groups):
    random = np.random.default_rng(29)
    samples = []
    for _ in range(2000):
        rows = np.concatenate([groups[index] for index in random.integers(len(groups), size=len(groups))])
        truth = labels[rows]
        if truth.any() and (~truth).any():
            seeds = random.integers(len(reference), size=len(reference))
            samples.append(np.mean([auc(scaled[i][rows], truth) - auc(reference[i][rows], truth) for i in seeds]))
    return np.quantile(samples, [.025, .975]).tolist()


def scale_teacher(root, dataset, parent):
    splits, dataset_sha = read_dataset(dataset)
    parent_splits, parent_sha = read_dataset(parent)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if not manifest["augmentation"]["release_for_teacher_fit"] or manifest["augmentation"]["parent_manifest_sha256"] != parent_sha:
        raise ValueError("scale training requires the completed zero-overlap audit")
    rows, freeze_path = publisher_rows(root)
    archive = json.loads(report_path(root, score_only=False).read_text())
    freeze_sha = digest(freeze_path)
    if archive["dataset_manifest_sha256"] != parent_sha or archive["freeze_sha256"] != freeze_sha:
        raise ValueError("the archived B1 reference differs from the frozen comparison")
    base = root / "cache/turn-text-qwen3-06b"
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    texts = [row["text"] for row in rows]
    ending = [text.split()[-1] for text in texts]
    guards = {
        "number_ending_shifts": labels & np.asarray([number(token) for token in ending]),
        "closed_1_to_3_word_shifts": labels & np.asarray([
            row["words"] <= 3 and token not in OPEN and not number(token) for row, token in zip(rows, ending, strict=True)
        ]),
    }
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["caller"]].append(index)
    reference, scaled, trials = [], [], []
    meter = Progress("english-scale-teacher", len(SEEDS), "paired seeds")
    for seed in SEEDS:
        archived, = [item for item in archive["trials"]["balanced"] if item["seed"] == seed]
        source = root / "runs" / f"turn-text-qwen3-length-balanced-{seed}" / "checkpoint.pt"
        if digest(source) != archived["checkpoint_sha256"]:
            raise ValueError(f"B1 seed {seed} checkpoint changed")
        tokenizer, model = load_fit(source, base, "cuda", parent_sha)
        control = np.asarray(predict(tokenizer, model, texts, "cuda"))
        del model
        baseline = binary_metrics(list(zip(control, labels, strict=True)))
        if baseline != archived["views"]["all"]:
            raise ValueError("B1 publisher score failed exact reproduction")
        destination = root / "runs" / f"turn-text-qwen3-en-scale-v1-{seed}"
        train(dataset, base, destination, seed=seed, backbone="qwen3", balance_length=True)
        tokenizer, model = load_fit(destination / "checkpoint.pt", base, "cuda", dataset_sha)
        candidate = np.asarray(predict(tokenizer, model, texts, "cuda"))
        del model
        candidate_metrics = binary_metrics(list(zip(candidate, labels, strict=True)))
        reference.append(control)
        scaled.append(candidate)
        trials.append({"seed": seed, "reference": baseline, "scale": candidate_metrics,
                       "checkpoint_sha256": digest(destination / "checkpoint.pt"),
                       "descriptive_shift_guards": {name: {
                           "shifts": int(mask.sum()),
                           "reference_recall": float(np.mean(control[mask] >= .5)) if mask.any() else None,
                           "scale_recall": float(np.mean(candidate[mask] >= .5)) if mask.any() else None,
                       } for name, mask in guards.items()}})
        meter.advance(1)
    gains = [trial["scale"]["auc"] - trial["reference"]["auc"] for trial in trials]
    delta = float(np.mean(gains))
    interval = paired_interval(reference, scaled, labels, list(groups.values()))
    report = {
        "scope": "Provisional publisher English development, three paired seeds, frozen threshold 0.5",
        "publisher_freeze_sha256": freeze_sha, "training_manifest_sha256": dataset_sha,
        "training_rows": {"reference": len(parent_splits["train"]), "scale": len(splits["train"])},
        "training_recipe": {"epochs": 3, "batch_size": 32, "unfrozen_layers": 4, "length_balanced_weights": True},
        "training_steps": {name: (len(values["train"]) + 31) // 32 * 3
                           for name, values in (("reference", parent_splits), ("scale", splits))},
        "trials": trials, "auc_delta": delta, "seed_and_publisher_turn_ci95": interval,
        "result": "scale_helped" if delta >= .03 and interval[0] > 0 else
                  "no_scale_gain" if delta <= .01 else "inconclusive",
    }
    _write_json(root / "reports" / RESULT, report)
    meter.finish()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train"))
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    root = data_root()
    parent = root / "cache/turn-text-focused-en-v1"
    dataset = args.dataset or root / "cache/turn-text-en-scale-v1"
    if args.action == "audit":
        if not audit_export(root, args.generation or root / "cache" / SCALE_VERSION, parent, dataset):
            raise SystemExit(1)
    else:
        scale_teacher(root, dataset, parent)


if __name__ == "__main__":
    main()
