"""Rebuild synthetic v1 with a fixed dose of harder incomplete turns."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from .ami_calibration import _write_json
from .shards import seed_for
from .turn_text_train import digest, read_dataset
from .turn_training_dataset import model_text


def near_end_choices(dialogue, held):
    before = dialogue["caller_before_pause"].split()
    after = dialogue["caller_after_pause"].split()
    words = [*before, *after]
    if dialogue["language"] in {"zh", "ja"}:
        return []
    first, _ = held["text"].split("\n")
    return [model_text("" if first == "---" else first, " ".join(words[:-count]))
            for count in (1, 2, 3) if count < len(after)]


def make_train(rows, validation, dialogues):
    existing = {row["text"] for row in [*rows, *validation]}
    by_context = defaultdict(list)
    for index, row in enumerate(rows):
        first, current = row["text"].split("\n")
        by_context[row["language"], first == "---"].append((index, row, len(current.split())))

    replaced = rows.copy()
    stats = Counter()
    for (language, one_turn), group in sorted(by_context.items()):
        holds = Counter(length for _, row, length in group if row["label"] == "hold")
        shifts = Counter(length for _, row, length in group if row["label"] == "shift")
        ordered = sorted(group, key=lambda item: seed_for("near-end-source-v1", item[1]["slot"]))
        quota = sum(holds.values()) // 2
        count = 0
        for index, row, length in ordered:
            if count == quota:
                break
            if row["label"] != "hold":
                continue
            candidates = [text for text in near_end_choices(dialogues[row["slot"]], row) if text not in existing]
            if not candidates:
                continue
            holds[length] -= 1
            # Prefer a cut at a word count with fewer holds than shifts.
            def priority(text):
                words = len(text.split("\n", 1)[1].split())
                return holds[words] >= shifts[words], seed_for("near-end-cut-v1", row["slot"], words)

            choice = min(candidates, key=priority)
            existing.add(choice)
            holds[len(choice.split("\n", 1)[1].split())] += 1
            replaced[index] = {**row, "text": choice}
            stats[f"{language}/{'one_turn' if one_turn else 'two_turn'}"] += 1
            count += 1
    return replaced, stats


def english_length_control(rows, validation, _dialogues):
    """Replace English holds with exact word-count-matched cuts of synthetic shifts."""
    existing = {row["text"] for row in [*rows, *validation]}
    original_holds, donor_cuts = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        if row["language"] != "en":
            continue
        first, current = row["text"].split("\n")
        words = current.split()
        context = first == "---"
        if row["label"] == "hold":
            original_holds[context, len(words)].append(index)
        elif len(words) >= 4:
            cut = f"{first}\n{' '.join(words[:-2])}"
            if cut not in existing:
                donor_cuts[context, len(words) - 2].append((row["slot"], cut))
                existing.add(cut)

    replaced, stats = rows.copy(), Counter()
    for (one_turn, length), originals in original_holds.items():
        indices = sorted(originals, key=lambda index: seed_for("near-end-matched-replaced-v1", rows[index]["slot"]))
        donors = sorted(donor_cuts[one_turn, length], key=lambda pair: seed_for("near-end-matched-donor-v1", pair[0]))
        for index, (slot, text) in zip(indices, donors):
            replaced[index] = {**rows[index], "text": text, "prefix_source_slot": slot}
            stats[f"en/{'one_turn' if one_turn else 'two_turn'}"] += 1
    return replaced, stats


def run(source, destination, *, length_control=False):
    splits, source_hash = read_dataset(source)
    manifest = json.loads((source / "manifest.json").read_text())
    reference = source / "dialogues.jsonl"
    if digest(reference) != manifest["dialogues_sha256"]:
        raise ValueError("the synthetic dialogue source changed")
    dialogues = {row["slot"]: row for row in map(json.loads, reference.read_text(encoding="utf-8").splitlines())}
    builder = english_length_control if length_control else make_train
    train, stats = builder(splits["train"], splits["validation"], dialogues)
    if not sum(stats.values()):
        raise ValueError("no strict near-end holds could be derived")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train).encode()
    manifest["splits"]["train"]["sha256"] = sha256(payload).hexdigest()
    manifest["augmentation"] = {
        "operation": ("replace English holds with equal-length two-word-short prefixes of shifts" if length_control
                      else "replace at most half of holds per language and context with 1–3-word-short prefixes"),
        "parent_manifest_sha256": source_hash,
        "replaced_holds_by_language_and_context": dict(stats),
    }
    manifest.pop("shortcut_checks", None)
    manifest.pop("plan_sha256", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix="near-end-") as temporary:
        output = Path(temporary) / "dataset"
        output.mkdir()
        (output / "train.jsonl").write_bytes(payload)
        copyfile(source / "validation.jsonl", output / "validation.jsonl")
        _write_json(output / "manifest.json", manifest)
        read_dataset(output)
        output.rename(destination)
    print(json.dumps(manifest["augmentation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--english-length-control", action="store_true")
    args = parser.parse_args()
    run(args.source, args.destination, length_control=args.english_length_control)
