"""Generate and merge a targeted English synthetic dialogue supplement."""

import argparse
from collections import Counter
from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from .ami_calibration import _write_json
from .shards import seed_for
from .turn_generation import CATALOGUE, clean_sample, make_plan, paired_examples, plan_checksum, run
from .turn_text_train import digest, read_dataset
from .turn_training_dataset import input_key

YES_NO = ["yes", "yeah", "yep", "no", "nope", "correct", "okay", "ok", "sure", "right", "exactly"]


def focused_plan(catalogue, splits):
    train_slots = {row["slot"] for row in splits["train"]}
    domains = {}
    for spec in make_plan(catalogue):
        if spec["language"] == "en" and spec["domain"] != "banking" and spec["slot"] in train_slots:
            domains.setdefault(spec["domain"], {})[spec["scenario"]] = spec
    # Draw one scenario per represented domain before taking a second from any.
    groups = [sorted(members.values(), key=lambda row: seed_for("focused-english-scenarios", row["scenario"]))
              for _, members in sorted(domains.items())]
    chosen = [group[index] for index in range(max(map(len, groups))) for group in groups if index < len(group)][:24]
    if len(chosen) != 24:
        raise ValueError("24 synthetic training scenario families are required")
    plan = []
    for spec in chosen:
        for band, full_range, prefix_range in (
                ("4-10", [4, 10], [2, 6]), ("10-13", [10, 13], [4, 10]), ("14-23", [14, 23], [8, 13])):
            for starts_yes_no in (False, True):
                style = "yes-no" if starts_yes_no else "neutral"
                criterion = f"en/focused-v1/{spec['scenario']}/{band}/{style}"
                for sample in (1, 2):
                    plan.append({
                        **spec, "slot": f"{criterion}/{sample}", "criterion": criterion,
                        "caller_first": False, "short_shift": False,
                        "variant": ("en-US", "en-IN", "en-SG")[(len(plan) + sample) % 3],
                        "full_caller_word_range": full_range, "caller_prefix_word_range": prefix_range,
                        "starts_yes_no": starts_yes_no, "yes_no_opening_words": YES_NO,
                        "pause_style": f"{band}/{style}",
                        "notes": (
                            f"{spec['notes']} Write a natural, compact spoken answer or request. Count the full "
                            f"caller answer at {full_range[0]}–{full_range[1]} whitespace-delimited words and the "
                            f"pre-pause portion at {prefix_range[0]}–{prefix_range[1]}. After the pause include "
                            "at least two words that materially continue the same thought. At the end give the "
                            "agent a substantive response. "
                            + ("Start the caller with a listed affirmative or negative word, as part of the reply."
                               if starts_yes_no else "Begin without a listed affirmative or negative word.")
                        ),
                    })
    return plan


def export(parent, destination, generation, plan, splits, manifest_hash, *, replay=False):
    receipt = json.loads((generation / "receipt.json").read_text())
    raw = generation / "dialogues.jsonl"
    if (receipt["stop"] != "complete" or receipt["plan_sha256"] != plan_checksum(plan)
            or digest(raw) != receipt["dialogues_sha256"]):
        raise ValueError("complete the targeted generation ledger before exporting")
    by_slot = {spec["slot"]: spec for spec in plan}
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(by_slot) or {row["slot"] for row in rows} != set(by_slot):
        raise ValueError("the focused generation membership changed")
    existing = {row["text"] for row in [*splits["train"], *splits["validation"]]}
    additions = []
    covered = Counter()
    for row in rows:
        spec = by_slot[row["slot"]]
        clean = clean_sample(row, spec)
        if clean is None or any(row[key] != value for key, value in clean.items()) or any(
                row[key] != value for key, value in spec.items()):
            raise ValueError("a focused dialogue does not match the plan")
        examples = paired_examples(row)
        texts = [input_key(item) for item in examples]
        if any(text in existing for text in texts):
            continue
        existing.update(texts)
        additions.extend({"slot": item["slot"], "language": "en", "label": item["label"], "text": text}
                         for item, text in zip(examples, texts, strict=True))
        covered[row["pause_style"]] += 1
    if min((covered[style] for style in {spec["pause_style"] for spec in plan}), default=0) < 40:
        raise ValueError("the focused dialogue coverage is below 40 pairs per cell after deduplication")
    if replay:
        originals = {}
        for row in splits["train"]:
            if row["language"] == "en" and not row["text"].startswith("---\n"):
                originals.setdefault(row["slot"], []).append(row)
        for index in range(0, len(additions), 2):
            original_slot = additions[index]["slot"]
            scenario = by_slot[original_slot]["scenario"]
            candidates = sorted(slot for slot, rows in originals.items()
                                if slot.split("/")[1] == scenario and len(rows) == 2)
            donor = candidates[seed_for("focused-english-replay-v1", original_slot) % len(candidates)]
            additions[index:index + 2] = [
                {**row, "slot": original_slot, "replay_source_slot": donor} for row in originals[donor]
            ]
    train = [*splits["train"], *additions]
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train).encode()
    manifest = json.loads((parent / "manifest.json").read_text())
    manifest["dialogues"] += len(additions) // 2
    manifest["splits"]["train"].update(
        sha256=sha256(payload).hexdigest(), examples=len(train),
        languages=dict(Counter(row["language"] for row in train)),
        labels=dict(Counter(row["label"] for row in train)),
        by_turn_count={str(turns): dict(Counter(row["label"] for row in train
                                              if row["text"].startswith("---\n") == (turns == 1)))
                       for turns in (1, 2)},
    )
    manifest["augmentation"] = {"parent_manifest_sha256": manifest_hash,
                                "focused_dialogues_sha256": receipt["dialogues_sha256"],
                                "added_pairs_by_cell": dict(covered),
                                "replay_control": replay}
    for prior_key in ("dialogues_sha256", "shortcut_checks", "plan_sha256"):
        manifest.pop(prior_key, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix="focused-en-") as temporary:
        output = Path(temporary) / "dataset"
        output.mkdir()
        (output / "train.jsonl").write_bytes(payload)
        copyfile(parent / "validation.jsonl", output / "validation.jsonl")
        _write_json(output / "manifest.json", manifest)
        read_dataset(output)
        output.rename(destination)
    print(json.dumps(manifest["augmentation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("--parent", type=Path, default=Path("datasets/turn-completion-v1"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--replay-control", action="store_true")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--attempts-per-slot", type=int, default=8)
    args = parser.parse_args()
    splits, manifest_hash = read_dataset(args.parent)
    plan = focused_plan(json.loads(CATALOGUE.read_text()), splits)
    if args.export:
        export(args.parent, args.export, args.generation, plan, splits, manifest_hash, replay=args.replay_control)
    else:
        if args.budget_usd is None or not 0 < args.budget_usd <= 2:
            parser.error("set --budget-usd to at most 2 for the focused supplement")
        args.generation.mkdir(parents=True, exist_ok=True)
        with (args.generation / "run.lock").open("a") as lock:
            flock(lock, LOCK_EX | LOCK_NB)
            receipt = run(plan, args.generation, args.budget_usd, 6, os.environ["GEMINI_API_KEY"],
                          attempts_per_slot=args.attempts_per_slot)
        if receipt["stop"] != "complete":
            raise SystemExit(1)
