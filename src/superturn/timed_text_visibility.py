"""Audit lexical visibility and length dependence on the frozen English development rows."""

import argparse
from collections import Counter
from hashlib import sha256
import json

import numpy as np
import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .ami_text_signal import fit
from .audio import RATE
from .shards import seed_for
from .sources import data_root
from .timed_text_pause_diagnostic import FREEZE, SHARD, pack_rows, pause_prefix
from .turn_generation import normalize
from .turn_metrics import auc
from .turn_text_train import digest, load_fit, predict

OPEN = set("""and but or so because cause if that which who when where the a an my your our his her their
this these to of for in on at with from by about as than is was are were be am have has had will would can
could i we you it's i'm um uh er like just also then plus""".split())
NUMBER = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
first second third fourth fifth sixth seventh eighth ninth tenth""".split())
CONNECTIVE = set("and but so because cause or then also plus which that".split())
FILLER = set("um uh er like well yeah okay oh".split())
QUESTION = set("can could would will do does did is are was have has what which when where who how why may".split())


def number(word):
    return word in NUMBER or any(character.isdecimal() for character in word)


def tags(row, anchor, raw):
    previous, current = row["text"].split("\n")
    end = current.split()[-1]
    ending = "open" if end in OPEN else "number" if number(end) else "closed_proxy"
    raw_prior = raw["messages"][-1]["content"]
    prior = "question" if previous.split()[0] in QUESTION or raw_prior.rstrip().endswith("?") else "statement"
    length = next(name for name, ceiling in (("1-3", 3), ("4-7", 7), ("8-15", 15), ("16+", float("inf")))
                  if row["words"] <= ceiling)
    gap = row["gap_ms"]
    gap_band = "1500+" if row["label"] else "300-449" if gap < 450 else "450-599" if gap < 600 else "600-799"
    if row["label"]:
        return ending, "n/a", prior, length, gap_band
    resume = anchor + gap / 1000
    words = sorted((word["start"], normalize(word["word"])) for word in raw["words"] or []
                   if word["word"] and word["start"] is not None
                   and max(anchor, resume - .08) <= word["start"] <= resume + .5)
    token = next((word.split()[0] for _, word in words if word), "")
    continuation = ("unknown" if not token else "connective" if token in CONNECTIVE else
                    "filler" if token in FILLER else "number" if number(token) else "other")
    return ending, continuation, prior, length, gap_band


def length_scores(rows):
    data = np.array([(row["words"], len(row["text"].split("\n", 1)[0].split())) for row in rows])
    labels = np.array([row["label"] for row in rows])
    turns = sorted({row["caller"] for row in rows}, key=lambda turn: seed_for("publisher-length-fold-v1", turn))
    assignments = {turn: index % 5 for index, turn in enumerate(turns)}
    folds = np.array([assignments[row["caller"]] for row in rows])
    out = np.empty(len(rows))
    for fold in range(5):
        test = folds == fold
        out[test], = fit(data[~test], labels[~test], [data[test]])
    return out


def length_auc(rows):
    return auc(length_scores(rows), [row["label"] for row in rows])


def summarize(rows, attributes, scores):
    buckets = {name: {} for name in ("ending/resume/length", "prior/length", "gap/ending")}
    repair = Counter()
    for row, (ending, continued, prior, length, gap), score in zip(rows, attributes, scores, strict=True):
        shift = bool(row["label"])
        positive = score >= .5
        outcome = ("shift_hit" if positive else "shift_missed") if shift else (
            "false_complete" if positive else "correct_hold")
        for key, parts in zip(buckets, ((ending, continued, length), (prior, length), (gap, ending)), strict=True):
            buckets[key].setdefault("/".join(parts), Counter())[outcome] += 1
        repair[outcome] += 1
        if outcome == "false_complete":
            if ending == "open" or (ending == "closed_proxy" and continued == "connective"):
                repair["oracle_repaired_holds"] += 1
            if row["words"] >= 8:
                repair["long_false_complete"] += 1
                repair["long_false_complete_open"] += ending == "open"
                repair["long_false_complete_afterthought_proxy"] += (
                    ending == "closed_proxy" and continued in {"other", "filler"} and gap == "600-799")
    holds = repair["false_complete"] + repair["correct_hold"]
    recall = repair["shift_hit"] / (repair["shift_hit"] + repair["shift_missed"])
    ba = (1 - repair["false_complete"] / holds + recall) / 2
    return {"counts": dict(repair), "strata": buckets, "auc": auc(scores, [row["label"] for row in rows]),
            "balanced_accuracy": ba,
            "oracle_repair_fixed_recall_ba": ba + repair["oracle_repaired_holds"] / (2 * holds)}


def boundary_audit(rows, anchors, source):
    counts = Counter()
    open_endings = Counter()
    last_word_gaps = []
    for row in rows:
        if not row["label"]:
            continue
        current = row["text"].split("\n")[1]
        word = current.split()[-1]
        if word not in OPEN:
            continue
        counts["open_ending_shifts"] += 1
        open_endings[word] += 1
        edge = anchors[row["rank"]]
        raw = source[row["call"]]
        aligned = [(item["start"], item["end"]) for item in raw["words"]
                   if item["word"] and item["start"] is not None and item["end"] is not None
                   and 0 <= item["start"] < item["end"]]
        preceding = [end for _, end in aligned if round(end * RATE) <= round(edge * RATE)]
        if preceding:
            last_word_gaps.append((edge - max(preceding)) * 1000)
        _, at_deadline = pause_prefix(raw, round((edge + .3) * RATE))
        if current != at_deadline:
            counts["changes_when_cut_at_deadline"] += 1
            if at_deadline and at_deadline.split()[-1] not in OPEN:
                counts["open_tag_changes_when_cut_at_deadline"] += 1
        if any(start < edge < end <= edge + .3 for start, end in aligned):
            counts["has_word_straddling_edge"] += 1
    return {"counts": dict(counts), "ending_token_counts": dict(open_endings),
            "last_pre_edge_word_gap_ms_p10_p50_p90": np.percentile(last_word_gaps, [10, 50, 90]).tolist()}


def number_rendering(rows, attributes):
    report = {"hold": Counter(), "shift": Counter()}
    for row, (ending, resumed, _, length, _) in zip(rows, attributes, strict=True):
        if ending != "number":
            continue
        token = row["text"].split()[-1]
        form = "decimal" if token.isdecimal() else "mixed" if any(ch.isdecimal() for ch in token) else "word"
        counts = report["shift" if row["label"] else "hold"]
        counts["total"] += 1
        counts[f"ending/{form}"] += 1
        counts[f"length/{length}"] += 1
        counts[f"resume/{resumed}"] += 1
    return report


def run(root, *, boundary_only=False, numbers_only=False, weighted_only=False):
    from .harper_partial_stt import english_membership

    if not (root / ".campaign-machine").is_file():
        raise ValueError("publisher benchmark inputs stay on Vast")
    freeze = root / "reports" / FREEZE
    frozen = json.loads(freeze.read_text())
    rows, _, membership = pack_rows(root)
    if (frozen["membership_sha256"] != membership
            or frozen["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()):
        raise ValueError("visibility diagnostic inputs differ from the freeze")
    members, _, _ = english_membership(root)
    anchors = {rank: stop / RATE for rank, _, stop, _, _ in members}
    rows = [row for row in rows if row["has_previous"] and
            (row["gap_ms"] >= 1500 if row["label"] else 300 <= row["gap_ms"] < 800)]
    if Counter(row["label"] for row in rows) != {0: 415, 1: 397}:
        raise ValueError("visibility diagnostic membership mismatch")
    source = pq.read_table(root / SHARD, columns=["words", "messages"]).to_pylist()
    if boundary_only:
        report = {"freeze_sha256": digest(freeze), **boundary_audit(rows, anchors, source)}
        _write_json(root / "reports/publisher-timed-english-word-boundary.json", report)
        print(json.dumps(report, indent=2))
        return
    attributes = [tags(row, anchors[row["rank"]], source[row["call"]]) for row in rows]
    if numbers_only:
        report = {"freeze_sha256": digest(freeze), "forms": number_rendering(rows, attributes)}
        _write_json(root / "reports/publisher-timed-english-number-forms.json", report)
        print(json.dumps(report, indent=2))
        return
    report = {"scope": "English publisher development only, fixed 0.5; oracle repair sensitivity is not a ceiling",
              "freeze_sha256": digest(freeze), "length_only_grouped_cv_auc": length_auc(rows), "arms": {}}
    if weighted_only:
        comparison = json.loads((root / "reports/publisher-timed-english-length-trial-ranking.json").read_text())
        fits = [(str(trial["seed"]), f"turn-text-qwen3-length-balanced-{trial['seed']}",
                 "turn-text-focused-en-v1", trial["checkpoint_sha256"]) for trial in comparison["trials"]["balanced"]]
    else:
        pilot = json.loads((root / "reports/publisher-timed-english-context-pilot.json").read_text())
        fits = [
            ("parent", "turn-text-qwen3-focused-v1", "turn-text-focused-en-v1", frozen["checkpoint_sha256"]["qwen"]),
            ("replay", "turn-text-qwen3-context-replay-v1", "turn-text-context-replay-v1",
             pilot["arms"]["replay"]["checkpoint_sha256"]),
        ]
    for arm, run_name, dataset, expected in fits:
        checkpoint = root / "runs" / run_name / "checkpoint.pt"
        if digest(checkpoint) != expected:
            raise ValueError("unexpected visibility diagnostic checkpoint")
        manifest = root / "cache" / dataset / "manifest.json"
        tokenizer, model = load_fit(checkpoint, root / "cache/turn-text-qwen3-06b", "cuda", digest(manifest))
        scores = predict(tokenizer, model, [row["text"] for row in rows], "cuda")
        report["arms"][arm] = summarize(rows, attributes, scores)
        del model
    name = "publisher-timed-english-weighted-text-visibility" if weighted_only else "publisher-timed-english-text-visibility"
    if weighted_only:
        grouped = {}
        for arm in report["arms"].values():
            for key, buckets in arm["strata"].items():
                group = grouped.setdefault(key, {})
                for bucket, counts in buckets.items():
                    group.setdefault(bucket, Counter()).update(counts)
        report["mean_strata_counts"] = {key: {bucket: {label: count / len(fits) for label, count in counts.items()}
                                              for bucket, counts in groups.items()} for key, groups in grouped.items()}
    _write_json(root / f"reports/{name}.json", report)
    print(json.dumps({name: {key: value for key, value in arm.items() if key != "strata"}
                      for name, arm in report["arms"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--boundary-only", action="store_true")
    modes.add_argument("--numbers-only", action="store_true")
    modes.add_argument("--weighted-only", action="store_true")
    args = parser.parse_args()
    run(data_root(), boundary_only=args.boundary_only, numbers_only=args.numbers_only, weighted_only=args.weighted_only)
