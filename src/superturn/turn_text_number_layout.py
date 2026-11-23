"""Freeze the unchanged length and prior-form layout for an English number pilot."""

import argparse
from collections import Counter, defaultdict
from fcntl import LOCK_EX, LOCK_NB, flock
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from .ami_calibration import _write_json
from .shards import seed_for
from .sources import data_root
from .timed_text_visibility import QUESTION, number
from .turn_generation import FIELDS, clean_sample, paired_examples, plan_checksum, run
from .turn_text_train import digest, length_balanced_weights, read_dataset
from .turn_training_dataset import input_key

QUOTAS = {
    ("16+", "16+"): 5,
    ("8-15", "8-15"): 4,
    ("8-15", "16+"): 34,
    ("4-7", "8-15"): 30,
}
SPLIT_SALT = "english-number-pilot-slots-v1"
DIGITS = set("zero one two three four five six seven eight nine".split())
SCENARIOS = {
    "train": ("train reservation reference", "theatre booking reference", "library locker code",
              "community center membership", "carton inventory tally", "power meter reading",
              "garden equipment pickup reference", "bike repair ticket number"),
    "validation": ("school fundraiser claim reference", "science center group booking"),
    "test": ("recreation league registration code", "museum coat-check receipt"),
}
BOUNDS = {"4-7": [4, 7], "8-15": [8, 15], "16+": [16, 26]}


def length_bin(words):
    if words < 4:
        return "1-3"
    if words < 8:
        return "4-7"
    return "8-15" if words < 16 else "16+"


def validate_number_pair(fields, spec):
    prior = fields["previous_agent"].split()
    prefix = fields["caller_before_pause"].split()
    continuation = fields["caller_after_pause"].split()
    full = prefix + continuation
    if any(ch.isdecimal() for text in fields.values() for ch in text):
        return False
    expected_question = spec["prior_form"] == "question"
    if ((prior[0] in QUESTION) != expected_question
            or not spec["prefix_words"][0] <= len(prefix) <= spec["prefix_words"][1]
            or not spec["full_words"][0] <= len(full) <= spec["full_words"][1]
            or len(continuation) < 2 or prefix[-1] not in DIGITS
            or sum(word in DIGITS for word in prefix) < 2
            or sum(word in DIGITS for word in continuation) < 2
            or not 7 <= sum(word in DIGITS for word in full) <= 16):
        return False
    if spec["continuation"] == "number":
        return continuation[0] in DIGITS and continuation[-1] in DIGITS
    return not number(continuation[0]) and not number(continuation[-1])


def build_plan(donors):
    options = []
    # Reserve 59 number-resume and 14 lexical-resume training groups across the
    # entire positional layout. Each donor gets two independent candidates.
    ranked = sorted(donors, key=lambda row: seed_for("english-number-pilot-continuation-v1", row["old_slot"]))
    lexical = {row["old_slot"] for row in ranked[:14]}
    for index, donor in enumerate(donors):
        branch = "lexical" if donor["old_slot"] in lexical else "number"
        options.append(("train", index, donor, branch, 2))
    # Hold out whole task families. Keep both continuation types and prior
    # forms in each held-out split.
    for split in ("validation", "test"):
        for index in range(24):
            donor = ranked[index * 3 % len(ranked)]
            options.append((split, index, donor, "lexical" if index % 5 == 0 else "number", 1))
    plan = []
    for split, index, donor, branch, repeats in options:
        scenario = SCENARIOS[split][index % len(SCENARIOS[split])]
        criterion = f"en/number-v1/{split}/{index:03d}"
        for repeat in range(repeats):
            plan.append({
                "slot": f"{criterion}/{repeat}", "criterion": criterion, "language": "en",
                "native_script": "Latn", "language_name": "English", "number_pilot": True,
                "caller_first": False, "short_shift": False, "pause_style": f"dictation/{branch}",
                "scenario": scenario, "split": split, "layout_index": index,
                "number_validator_version": 1,
                "continuation": branch, "prior_form": donor["prior_form"],
                "prefix_words": BOUNDS[donor["hold_bin"]], "full_words": BOUNDS[donor["shift_bin"]],
            })
    return plan


def eligible_donors(rows):
    pairs = defaultdict(list)
    for index, row in enumerate(rows):
        if row["language"] == "en":
            pairs[row["slot"]].append((index, row))
    eligible = defaultdict(list)
    for slot, group in pairs.items():
        if len(group) != 2 or {row["label"] for _, row in group} != {"hold", "shift"}:
            continue
        (hold_index, hold), (shift_index, shift) = sorted(group, key=lambda item: item[1]["label"])
        prior, prefix = hold["text"].split("\n")
        shift_prior, full = shift["text"].split("\n")
        if prior == "---" or prior != shift_prior or not full.startswith(prefix + " "):
            continue
        before, after = prefix.split(), full.split()
        if number(before[-1]) or number(after[-1]):
            continue
        hold_bin, shift_bin = length_bin(len(before)), length_bin(len(after))
        eligible[hold_bin, shift_bin].append({
            "old_slot": slot, "hold_index": hold_index, "shift_index": shift_index,
            "hold_bin": hold_bin, "shift_bin": shift_bin, "prior_form": "question" if prior.split()[0] in QUESTION
            else "statement",
        })
    return eligible


def layout(root: Path):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("freeze the existing held development pipeline on Vast")
    splits, corpus_hash = read_dataset(root / "cache/turn-text-focused-en-v1")
    rows = splits["train"]
    if len(rows) != 4272 or sum(row["language"] == "en" for row in rows) != 840:
        raise ValueError("English pilot requires the pinned parent row counts")
    eligible = eligible_donors(rows)
    chosen = []
    for band, quota in QUOTAS.items():
        ranked = sorted(eligible[band], key=lambda row: seed_for(SPLIT_SALT, row["old_slot"]))
        if len(ranked) < quota * 3:
            raise ValueError(f"quota would replace over one third of the original groups in {band}")
        chosen.extend(ranked[:quota])
    indices = {row[key] for row in chosen for key in ("hold_index", "shift_index")}
    if len(indices) != 2 * sum(QUOTAS.values()):
        raise ValueError("a donor position is reused")
    report = {
        "scope": "preflight only; no generation or replacement",
        "parent_manifest_sha256": corpus_hash, "hash_selection_salt": SPLIT_SALT, "groups": len(chosen),
        "group_counts": dict(Counter(f"{row['hold_bin']}/{row['shift_bin']}/{row['prior_form']}" for row in chosen)),
        "eligible_groups": {"/".join(key): len(value) for key, value in sorted(eligible.items())},
    }
    _write_json(root / "cache/turn-text-number-pilot-layout.json", {**report, "donors": chosen})
    _write_json(root / "reports/turn-text-number-pilot-layout.json", report)
    print(json.dumps(report, indent=2))


def generate(layout_path, directory):
    donors = json.loads(layout_path.read_text())["donors"]
    if len(donors) != sum(QUOTAS.values()):
        raise ValueError("unexpected number of registered donor slots")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        run(build_plan(donors), directory, .5, 6, os.environ["GEMINI_API_KEY"], attempts_per_slot=4)


def text_distribution(items):
    counts = Counter()
    for row in items:
        prior, caller = row["text"].split("\n")
        form = "absent" if prior == "---" else "question" if prior.split()[0] in QUESTION else "statement"
        counts[row["language"], row["label"], length_bin(len(caller.split())), form] += 1
    return counts


def export(root, directory, destination):
    from .turn_text_context_supplement import dev_blocklist, ngrams

    if not (root / ".campaign-machine").is_file():
        raise ValueError("screen the number pilot against private development inputs on Vast")
    parent = root / "cache/turn-text-focused-en-v1"
    splits, corpus_hash = read_dataset(parent)
    layout_path = root / "cache/turn-text-number-pilot-layout.json"
    frozen = json.loads(layout_path.read_text())
    donors = frozen["donors"]
    plan = build_plan(donors)
    receipt = json.loads((directory / "receipt.json").read_text())
    path = directory / "dialogues.jsonl"
    if (frozen["parent_manifest_sha256"] != corpus_hash or receipt["plan_sha256"] != plan_checksum(plan)
            or receipt["dialogues_sha256"] != digest(path)):
        raise ValueError("pilot layout, parent inputs and generation receipts disagree")
    source = [json.loads(line) for line in path.read_text().splitlines()]
    if receipt["accepted_dialogues"] != len(source) or len({row["slot"] for row in source}) != len(source):
        raise ValueError("generation receipt has a different sample count")
    specs = {spec["slot"]: spec for spec in plan}
    old = splits["train"]
    train = old.copy()
    occupied = {row["text"] for name in ("train", "validation") for row in splits[name]}
    overlap = dev_blocklist(root)
    sequences, replacements = set(), {}
    holdouts = defaultdict(list)
    rejection = Counter()
    for row in sorted(source, key=lambda item: seed_for("english-number-selection-v1", item["slot"])):
        spec = specs[row["slot"]]
        clean = clean_sample(row, spec)
        if (clean is None or any(row[key] != value for key, value in clean.items())
                or any(row[key] != value for key, value in spec.items())):
            raise ValueError("generated sequence differs from its reserved slot")
        if row["split"] == "train" and row["layout_index"] in replacements:
            continue
        code = tuple(word for field in ("caller_before_pause", "caller_after_pause")
                     for word in row[field].split() if word in DIGITS)
        pair = paired_examples(row)
        texts = [input_key(example) for example in pair]
        if code in sequences or any(text in occupied for text in texts):
            rejection["duplicate"] += 1
            continue
        if any(key in overlap for field in FIELDS for key in ngrams(row[field])):
            rejection["benchmark_eight_word_overlap"] += 1
            continue
        sequences.add(code)
        occupied.update(texts)
        if row["split"] != "train":
            holdouts[row["split"]].extend({"slot": row["slot"], "language": "en", "text": text,
                                           "label": example["label"]}
                                          for text, example in zip(texts, pair, strict=True))
            continue
        index = row["layout_index"]
        donor = donors[index]
        for label, text in zip(("hold", "shift"), texts, strict=True):
            position = donor[f"{label}_index"]
            if old[position]["slot"] != donor["old_slot"] or old[position]["label"] != label:
                raise ValueError("parent no longer contains the original donor positions")
            train[position] = {**old[position], "text": text}
        replacements[index] = row["slot"]
    if len(replacements) != len(donors) or any(len(holdouts[key]) < 40 for key in ("test", "validation")):
        raise ValueError("generated pool failed to fill the registered training and held-out quotas")
    if text_distribution(old) != text_distribution(train):
        raise ValueError("paired replacement changed the language, label, length or prior layout")
    if length_balanced_weights(old)[0] != length_balanced_weights(train)[0]:
        raise ValueError("paired replacement changed a training position's length-balanced weight")
    endpoints = Counter(row["label"] for row in train if row["language"] == "en" and number(row["text"].split()[-1]))
    if endpoints["hold"] > 1.3 * endpoints["shift"]:
        raise ValueError("number-ending hold/shift balance exceeded 1.3:1")
    manifest = json.loads((parent / "manifest.json").read_text())
    for key in ("dialogues", "shortcut_checks"):
        manifest.pop(key, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix="number-pilot-") as temporary:
        output = Path(temporary) / "dataset"
        output.mkdir()
        payload = lambda records: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                           for row in records).encode()
        (output / "train.jsonl").write_bytes(payload(train))
        copyfile(parent / "validation.jsonl", output / "validation.jsonl")
        manifest["splits"]["train"]["sha256"] = digest(output / "train.jsonl")
        held = {}
        for split, records in sorted(holdouts.items()):
            name = output / f"number-{split}.jsonl"
            name.write_bytes(payload(records))
            held[split] = {"rows": len(records), "sha256": digest(name)}
        manifest["number_pilot"] = {
            "parent_manifest_sha256": corpus_hash, "layout_sha256": digest(layout_path),
            "generated_dialogues_sha256": digest(path), "replaced_groups": replacements,
            "english_number_endpoints": dict(endpoints), "rejected_pairs": dict(rejection), "heldout": held,
            "matched_distribution": True,
        }
        _write_json(output / "manifest.json", manifest)
        read_dataset(output)
        output.rename(destination)
    print(json.dumps({**manifest["number_pilot"], "replaced_groups": len(replacements)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-layout", type=Path, help="print a count-only generation plan check")
    parser.add_argument("--generate", type=Path, help="write resumable Gemini generation state to this directory")
    parser.add_argument("--export-source", type=Path)
    parser.add_argument("--dataset-out", type=Path)
    args = parser.parse_args()
    if args.export_source and args.dataset_out:
        export(data_root(), args.export_source, args.dataset_out)
    elif args.generate and args.parent_layout:
        generate(args.parent_layout, args.generate)
    elif args.parent_layout:
        donors = json.loads(args.parent_layout.read_text())["donors"]
        plan = build_plan(donors)
        print(json.dumps(dict(Counter(row["split"] for row in plan)), indent=2))
    else:
        layout(data_root())
