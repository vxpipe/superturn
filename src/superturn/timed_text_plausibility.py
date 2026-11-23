"""Build a blind, quota-sampled English publisher text failure panel on Vast."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .ami_calibration import _write_json
from .harper_text_plausibility import RATINGS
from .shards import seed_for
from .sources import data_root
from .timed_text_context_audit import hold_interval
from .timed_text_pause_diagnostic import FREEZE, SHARD, pack_rows
from .turn_metrics import auc
from .turn_text_train import digest, load_fit, predict

PANEL = Path("/dev/shm/superturn-publisher-text-plausibility")
REPORT = "publisher-text-plausibility-packet.json"
QUOTAS = {"long_hold_false_complete": 30, "long_hold_correct": 30,
          "short_shift_missed": 20, "short_shift_recovered": 20}


def select(rows):
    pools = {key: [] for key in QUOTAS}
    for row in rows:
        hit = row["score"] >= .5
        if not row["label"] and 300 <= row["gap_ms"] < 800 and row["words"] >= 8:
            pools["long_hold_false_complete" if hit else "long_hold_correct"].append(row)
        if row["label"] and row["gap_ms"] >= 1500 and 1 <= row["words"] <= 3:
            pools["short_shift_recovered" if hit else "short_shift_missed"].append(row)
    selected, turns, texts = [], set(), set()
    for group, pool in pools.items():
        count = 0
        for row in sorted(pool, key=lambda item: seed_for("publisher-plausibility-select-v1", item["rank"])):
            if row["caller"] in turns or row["text"] in texts:
                continue
            selected.append((group, row))
            turns.add(row["caller"])
            texts.add(row["text"])
            count += 1
            if count == QUOTAS[group]:
                break
        if count != QUOTAS[group]:
            raise ValueError(f"insufficient unique publisher turns for {group}: {count}")
    selected.sort(key=lambda item: seed_for("publisher-plausibility-display-v1", item[1]["rank"]))
    return selected, {key: len(pool) for key, pool in pools.items()}


def run(root):
    import pyarrow.parquet as pq
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("publisher benchmark text stays on Vast")
    torch.set_num_threads(4)
    checkpoint = root / "runs/turn-text-qwen3-focused-v1/checkpoint.pt"
    freeze = root / "reports" / FREEZE
    frozen = json.loads(freeze.read_text())
    rows, _, membership = pack_rows(root)
    if (frozen["membership_sha256"] != membership
            or frozen["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            or frozen["checkpoint_sha256"]["qwen"] != digest(checkpoint)):
        raise ValueError("publisher review inputs changed")
    rows = [row for row in rows if row["has_previous"]]
    tokenizer, model = load_fit(checkpoint, root / "cache/turn-text-qwen3-06b", "cpu",
                                digest(root / "cache/turn-text-focused-en-v1/manifest.json"))
    context = predict(tokenizer, model, [row["text"] for row in rows], "cpu")
    masked = predict(tokenizer, model, [row["masked"] for row in rows], "cpu")
    source = pq.read_table(root / SHARD, columns=["messages"])["messages"].to_pylist()
    for row, score, without in zip(rows, context, masked, strict=True):
        previous = source[row["call"]][-1]["content"]
        row.update(score=score, masked_score=without, question=previous.rstrip().endswith("?"))
    selected, populations = select(rows)
    packet, reference = [], {}
    for group, row in selected:
        ident = f"text-{seed_for('publisher-plausibility-id-v1', row['rank']):016x}"
        previous, current = row["text"].split("\n")
        packet.append({"id": ident, "previous": previous, "current": current})
        reference[ident] = {"group": group, "rank": row["rank"], "words": row["words"]}
    if len(reference) != sum(QUOTAS.values()):
        raise ValueError("duplicate blind item identifier")
    PANEL.mkdir(mode=0o700, exist_ok=True)
    _write_json(PANEL / "packet.json", {"mode": "plausibility", "items": packet})
    _write_json(PANEL / "private-reference.json", reference)
    labels = [row["label"] for row in rows]
    short = [row for row in rows if not row["label"] and 300 <= row["gap_ms"] < 800]
    deltas = {}
    for name, value in (("question_mark", True), ("other_previous", False)):
        subset = [row for row in short if row["question"] == value]
        if subset:
            deltas[name] = hold_interval(subset, {row["rank"]: row["masked_score"] for row in subset})
    report = {
        "scope": "blinded publisher diagnostic, publisher turn clusters, development only",
        "freeze_sha256": digest(freeze), "packet_sha256": digest(PANEL / "packet.json"),
        "reference_sha256": digest(PANEL / "private-reference.json"),
        "selection": dict(Counter(row["group"] for row in reference.values())), "population": populations,
        "threshold": .5, "caller_word_count_auc": auc([row["words"] for row in rows], labels),
        "caller_word_count_reverse_auc": auc([-row["words"] for row in rows], labels),
        "raw_prior_question_mark_auc": auc([int(row["question"]) for row in rows], labels),
        "short_hold_context_minus_masked": deltas,
    }
    _write_json(root / "reports" / REPORT, report)
    print(json.dumps(report, indent=2))


def score(root, ballots):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("publisher review ballots stay on Vast")
    receipt = json.loads((root / "reports" / REPORT).read_text())
    packet = PANEL / "packet.json"
    reference_path = PANEL / "private-reference.json"
    if digest(packet) != receipt["packet_sha256"] or digest(reference_path) != receipt["reference_sha256"]:
        raise ValueError("publisher review packet changed")
    reference = json.loads(reference_path.read_text())
    if len({ballot.resolve() for ballot in ballots}) != 2 or len({digest(ballot) for ballot in ballots}) != 2:
        raise ValueError("two independent reviewer drafts are required")
    ratings = []
    for ballot in ballots:
        draft = json.loads(ballot.read_text())["draft"]["clips"]
        if set(draft) != set(reference) or any(row.get("outcome") not in RATINGS for row in draft.values()):
            raise ValueError("complete every publisher plausibility item before scoring")
        ratings.append({ident: row["outcome"] for ident, row in draft.items()})
    first, second = ratings
    left, right = Counter(first.values()), Counter(second.values())
    total = len(reference)
    observed = sum(first[ident] == second[ident] for ident in reference) / total
    expected = sum(left[key] * right[key] for key in RATINGS) / total**2
    groups = {}
    permissible = {"could_end", "complete"}
    for group in QUOTAS:
        ids = [ident for ident, row in reference.items() if row["group"] == group]
        groups[group] = {
            "count": len(ids), "per_reviewer": [dict(Counter(rating[ident] for ident in ids)) for rating in ratings],
            "both_permissible_rate": sum(first[ident] in permissible and second[ident] in permissible
                                         for ident in ids) / len(ids),
        }
    result = {"packet_sha256": receipt["packet_sha256"], "ballot_sha256": [digest(ballot) for ballot in ballots],
              "four_choice_cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
              "groups": groups}
    _write_json(root / "reports/publisher-text-plausibility-score.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    if args.score:
        score(data_root(), args.score)
    else:
        run(data_root())
