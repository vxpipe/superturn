"""Export verified synthetic dialogues as grouped text training examples."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .ami_calibration import _write_json
from .splits import bucket, speaker_groups
from .turn_generation import (
    CATALOGUE, CLEANER_VERSION, GENERATOR, QUOTA, SYSTEM,
    clean_sample, make_plan, paired_examples, plan_checksum,
)


def verify_dialogues(source, plan):
    corpus = (source / "dialogues.jsonl").read_bytes()
    receipt = json.loads((source / "receipt.json").read_text())
    if (receipt["plan_sha256"] != plan_checksum(plan)
            or receipt["dialogues_sha256"] != sha256(corpus).hexdigest()):
        raise ValueError("the dialogue bytes or generation plan differ from the receipt")
    expected = {spec["slot"]: spec for spec in plan}
    rows = [json.loads(line) for line in corpus.splitlines()]
    if receipt["accepted_dialogues"] != len(rows) or len({row["slot"] for row in rows}) != len(rows):
        raise ValueError("the accepted catalogue entries have the wrong count or duplicate slots")
    if set(expected) != {row["slot"] for row in rows}:
        raise ValueError("finish all planned dialogue slots before exporting the full matrix")
    constraints = {
        "stop": "complete", "generator": GENERATOR,
        "cleaner_version": CLEANER_VERSION, "target_per_criterion": QUOTA,
        "generator_prompt_sha256": sha256(SYSTEM.encode()).hexdigest(),
        "target_dialogues": len(plan), "target_criteria": len(plan) // QUOTA,
        "complete_criteria": len(plan) // QUOTA,
        "accepted_by_language": dict(Counter(row["language"] for row in rows)),
        "derived_example_counts": {"hold": len(rows), "shift": len(rows)},
    }
    if any(receipt.get(key) != value for key, value in constraints.items()):
        raise ValueError("the generation receipt does not describe the verified matrix")
    for row in rows:
        spec = expected[row["slot"]]
        if any(row.get(key) != value for key, value in spec.items()):
            raise ValueError("a dialogue's factors changed after judging")
        texts = clean_sample(row, spec)
        if texts is None or any(row[key] != value for key, value in texts.items()):
            raise ValueError("accepted dialogue is incompatible with the recorded cleaner")
    return rows, receipt


def model_text(previous_turn, current_turn):
    if (len(previous_turn.splitlines()) > 1 or len(current_turn.splitlines()) != 1
            or any(char in previous_turn + current_turn for char in "\n\r")):
        raise ValueError("turns must be single-line text and the current turn must be present")
    return f"{previous_turn or '---'}\n{current_turn}"


def input_key(example):
    return model_text(example["previous_turn"], example["current_turn"])


def split_examples(rows):
    examples = {row["slot"]: paired_examples(row) for row in rows}
    # A scenario is one family across languages. Identical model inputs can
    # join families, including legitimate hold/shift ambiguities.
    groups = speaker_groups(
        (row["slot"], [f"scenario:{row['scenario']}",
                       *(f"input:{input_key(item)}" for item in examples[row["slot"]])])
        for row in rows
    )
    assignments = {group: "validation" if bucket("turn-text-synthetic-v1", group) < 15 else "train"
                   for group in sorted(set(groups.values()))}
    splits = {name: [] for name in ("train", "validation")}
    for slot in sorted(examples):
        split = assignments[groups[slot]]
        splits[split].extend(examples[slot])
    if not all(splits.values()):
        raise ValueError("dataset needs more than one independent synthetic split group")
    train_keys = {input_key(item) for item in splits["train"]}
    valid_keys = {input_key(item) for item in splits["validation"]}
    if train_keys & valid_keys:
        raise AssertionError("a repeated model input crossed the synthetic split")
    return splits, Counter(assignments.values())


def balanced_accuracy(rows, predicts_shift):
    totals, correct = Counter(), Counter()
    for row in rows:
        label = row["label"]
        totals[label] += 1
        correct[label] += (label == "shift") == predicts_shift(row)
    return sum(correct[label] / totals[label] for label in ("hold", "shift")) / 2


def shortcut_checks(splits):
    train, validation = splits["train"], splits["validation"]
    lengths = sorted({len(row["current_turn"]) for row in train})
    length_rule = max(
        ((threshold, direction) for threshold in lengths for direction in (True, False)),
        key=lambda rule: balanced_accuracy(train, lambda row: (len(row["current_turn"]) >= rule[0]) == rule[1]),
    )
    endings = {}
    for row in train:
        tail = row["current_turn"].split()[-1]
        endings.setdefault((row["language"], tail), Counter())[row["label"]] += 1

    def last_token(row):
        counts = endings.get((row["language"], row["current_turn"].split()[-1]), Counter())
        return counts["shift"] > counts["hold"]

    threshold, direction = length_rule
    return {
        "scope": "synthetic validation balanced accuracy for shortcut baselines",
        "current_character_count": balanced_accuracy(
            validation, lambda row: (len(row["current_turn"]) >= threshold) == direction),
        "length_threshold_characters": threshold,
        "length_predicts_shift_above_threshold": direction,
        "language_and_last_token": balanced_accuracy(validation, last_token),
        "first_turn_marker_only": balanced_accuracy(validation, lambda row: not row["previous_turn"]),
    }


def export(source, destination):
    plan = make_plan(json.loads(CATALOGUE.read_text()))
    rows, receipt = verify_dialogues(source, plan)
    splits, groups = split_examples(rows)
    if destination.exists():
        raise ValueError("the export destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "synthetic training and diagnostic validation",
        "plan_sha256": receipt["plan_sha256"],
        "dialogues_sha256": receipt["dialogues_sha256"],
        "model_input_fields": ["text"],
        "input_layout": "previous speaker turn or --- on line one, current caller turn on line two",
        "first_turn_marker": "---",
        "dialogues": len(rows),
        "shortcut_checks": shortcut_checks(splits),
        "splits": {},
    }
    with TemporaryDirectory(dir=destination.parent, prefix="turn-training-") as temporary:
        root = Path(temporary) / "dataset"
        root.mkdir()
        for split, examples in splits.items():
            records = [{"slot": item["slot"], "language": item["language"], "label": item["label"],
                        "text": input_key(item)} for item in examples]
            payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                              for record in records).encode("utf-8")
            (root / f"{split}.jsonl").write_bytes(payload)
            languages = Counter(example["language"] for example in examples)
            labels = Counter(example["label"] for example in examples)
            manifest["splits"][split] = {
                "sha256": sha256(payload).hexdigest(), "examples": len(examples),
                "labels": dict(labels), "languages": dict(languages), "groups": groups[split],
                "by_turn_count": {
                    str(turns): dict(Counter(example["label"] for example in examples
                                             if (2 if example["previous_turn"] else 1) == turns))
                    for turns in (1, 2)
                },
            }
        _write_json(root / "manifest.json", manifest)
        root.rename(destination)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
