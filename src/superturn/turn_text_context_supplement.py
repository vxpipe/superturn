"""Generate an original synthetic prior-turn contrast without judge calls."""

import argparse
from collections import Counter
from fcntl import LOCK_EX, LOCK_NB, flock
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from .ami_calibration import _write_json
from .shards import seed_for
from .sources import data_root
from .turn_generation import clean_sample, paired_examples, plan_checksum, run
from .turn_text_train import digest, read_dataset
from .turn_training_dataset import input_key

SCENARIOS = (
    ("garden-tool-library", "A neighborhood tool library coordinating a garden equipment pickup"),
    ("community-cinema", "A local cinema helping a caller find a community screening"),
    ("dog-walking", "A dog-walking service coordinating an agreed pickup"),
    ("sports-center", "A sports center arranging a booking for recreational facilities"),
    ("shoe-repair", "A shoe repair shop discussing service and collection"),
    ("pottery-studio", "A pottery studio helping a caller arrange a class and supplies"),
    ("community-choir", "A community choir coordinator answering a member's practical questions"),
    ("fresh-produce-box", "A local produce box service discussing deliveries and substitutions"),
)
HELDOUT_FAMILIES = {"community-choir", "fresh-produce-box"}


def plan():
    rows = []
    for scenario, description in SCENARIOS:
        for kind, count in (("contrast", 15), ("invariant", 5)):
            criterion = f"en/context-v1/{scenario}/{kind}"
            for sample in range(count):
                rows.append({
                    "slot": f"{criterion}/{sample + 1:02}", "criterion": criterion, "language": "en",
                    "language_name": "English", "native_script": "Latn", "scenario": scenario,
                    "variant": ("en-US", "en-SG", "en-IN")[sample % 3], "situation": description,
                    "pair_type": kind, "pause_style": kind,
                    "concept": ("choose a day or time", "describe two requested details", "select one option",
                                "correct an earlier assumption", "specify a location or quantity")[sample % 5],
                })
    return rows


def generate(directory, budget):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        receipt = run(plan(), directory, budget, 6, os.environ["GEMINI_API_KEY"], attempts_per_slot=5)
    accepted = receipt["accepted_dialogues"]
    attempted = sum(receipt["attempted_by_pause_style"].values())
    if accepted < 140 or (attempted - accepted) / attempted > .15:
        raise ValueError(f"stop: {accepted} valid context pairs from {attempted} slot attempts")


def ngrams(text):
    for line in text.split("\n"):
        words = line.split()
        yield from zip(*(words[offset:] for offset in range(8)))


def dev_blocklist(root):
    from .harper_text_transcript_benchmark import transcript_pairs
    from .harper_valley_benchmark import pinned_split
    from .timed_text_pause_diagnostic import pack_rows

    blocked = {key for row in pack_rows(root)[0] for key in ngrams(row["text"])}
    calls, _, _ = pinned_split(root, "train")
    for case in calls:
        path = root / "raw/harper-valley/data/transcript" / f"{case['id']}.json"
        if path.exists():
            for _, views in transcript_pairs(json.loads(path.read_text())):
                for text in views.values():
                    blocked.update(ngrams(text))
    return blocked


def export(directory, parent, destination, *, root, replay):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("screen against the existing benchmark only on Vast")
    src = directory / "dialogues.jsonl"
    receipt = json.loads((directory / "receipt.json").read_text())
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()]
    specs = {spec["slot"]: spec for spec in plan()}
    attempted = sum(receipt["attempted_by_pause_style"].values())
    if (receipt["plan_sha256"] != plan_checksum(plan()) or receipt["dialogues_sha256"] != digest(src)
            or receipt["accepted_dialogues"] != len(rows) or len(rows) < 140
            or (attempted - len(rows)) / attempted > .15 or len({row["slot"] for row in rows}) != len(rows)):
        raise ValueError("the context generation receipt failed its preregistered checks")
    splits, parent_sha = read_dataset(parent)
    existing = {row["text"] for row in [*splits["train"], *splits["validation"]]}
    previous = dev_blocklist(root)
    fit, challenge, reject = [], [], Counter()
    for row in rows:
        spec = specs[row["slot"]]
        clean = clean_sample(row, spec)
        if (clean is None or any(row[key] != value for key, value in clean.items())
                or any(row[key] != value for key, value in spec.items())):
            raise ValueError("a generated dialogue differs from its reserved slot")
        pair = paired_examples(row)
        texts = [input_key(item) for item in pair]
        if not existing.isdisjoint(texts):
            reject["existing_exact_text"] += 1
            continue
        if any(key in previous for text in texts for key in ngrams(text)):
            reject["benchmark_eight_word_overlap"] += 1
            continue
        existing.update(texts)
        target = challenge if row["scenario"] in HELDOUT_FAMILIES else fit
        target.extend({"slot": row["slot"], "language": "en", "label": item["label"], "text": text}
                      for item, text in zip(pair, texts, strict=True))
    if (len(fit) + len(challenge)) // 2 < 140 or not challenge or not fit:
        raise ValueError("insufficient context pairs after private contamination screening")
    if replay:
        pools = {label: [row for row in splits["train"] if row["language"] == "en" and row["label"] == label
                        and not row["text"].startswith("---\n")] for label in ("hold", "shift")}
        fit = [{**pools[item["label"]][seed_for("context-replay-v1", item["slot"], index) % len(pools[item["label"]])],
                "slot": item["slot"]} for index, item in enumerate(fit)]
    train = [*splits["train"], *fit]
    manifest = json.loads((parent / "manifest.json").read_text())
    for key in ("dialogues", "shortcut_checks"):
        manifest.pop(key, None)
    payload = lambda records: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                       for row in records).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix="context-en-") as temporary:
        output = Path(temporary) / "dataset"
        output.mkdir()
        (output / "train.jsonl").write_bytes(payload(train))
        (output / "context-heldout.jsonl").write_bytes(payload(challenge))
        copyfile(parent / "validation.jsonl", output / "validation.jsonl")
        manifest["splits"]["train"].update(
            sha256=digest(output / "train.jsonl"), examples=len(train),
            labels=dict(Counter(row["label"] for row in train)),
            languages=dict(Counter(row["language"] for row in train)),
            groups=manifest["splits"]["train"]["groups"] + len(SCENARIOS) - len(HELDOUT_FAMILIES),
            by_turn_count={str(turns): dict(Counter(row["label"] for row in train
                                                  if row["text"].startswith("---\n") == (turns == 1)))
                           for turns in (1, 2)},
        )
        manifest["context_supplement"] = {"parent_manifest_sha256": parent_sha, "dialogues_sha256": digest(src),
                                          "replay_control": replay, "heldout_families": sorted(HELDOUT_FAMILIES),
                                          "heldout_sha256": digest(output / "context-heldout.jsonl"),
                                          "heldout_rows": len(challenge), "added_rows": len(fit),
                                          "added_labels": dict(Counter(row["label"] for row in fit)),
                                          "rejected_pairs": dict(reject)}
        _write_json(output / "manifest.json", manifest)
        read_dataset(output)
        output.rename(destination)
    print(json.dumps(manifest["context_supplement"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--budget-usd", type=float, default=1)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--replay-control", action="store_true")
    args = parser.parse_args()
    if not 0 < args.budget_usd <= 2:
        parser.error("use a positive budget of at most $2 for this pilot")
    if args.export:
        root = data_root()
        export(args.directory, args.parent or root / "cache/turn-text-focused-en-v1",
               args.export, root=root, replay=args.replay_control)
    else:
        generate(args.directory, args.budget_usd)
