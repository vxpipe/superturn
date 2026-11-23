"""Seal a blind text plausibility panel and summarize completed human drafts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .ami_calibration import _write_json
from .harper_text_failures import strata
from .shards import seed_for
from .sources import data_root

PANEL_ROOT = Path("/dev/shm/superturn-harper-text-plausibility")
RECEIPT_NAME = "harper-text-plausibility-packet.json"
RATINGS = {"incomplete", "could_end", "complete", "unrateable"}
QUOTAS = {"false_complete": 36, "correct_hold": 18, "full_turn": 18}
CHECKPOINT = "51d6f934707e0702fcd528b774222913ffb7e3ef37ae60bcf9da28e0b28d21d5"
RENDERED_INPUTS = "8f3b8ccbc7f9cba4dd62f488b0cd6c31c5e2cf52389cd4f4eae75b6b673b9a4b"


def pick_items(items):
    pools = defaultdict(list)
    for row in items:
        group = (
            "full_turn" if row["label"] else "false_complete" if row["score"] >= .5 else "correct_hold"
        )
        pools[group].append(row)
    order = lambda row: seed_for("harper-plausibility-v1", row["rank"])
    selected, errors, used_calls, used_text = [], [], set(), set()
    for row in sorted(pools["false_complete"], key=order):
        if row["call"] not in used_calls and row["text"] not in used_text:
            errors.append(row)
            selected.append(("false_complete", row, None))
            used_calls.add(row["call"])
            used_text.add(row["text"])
        if len(errors) == QUOTAS["false_complete"]:
            break
    if len(errors) != QUOTAS["false_complete"]:
        raise ValueError("insufficient distinct failure cases")
    anchors = sorted(errors, key=lambda row: seed_for("harper-plausibility-control-anchor-v1", row["rank"]))
    for group in ("correct_hold", "full_turn"):
        for anchor in anchors[:QUOTAS[group]]:
            possible = [row for row in pools[group] if row["call"] not in used_calls and row["text"] not in used_text]
            if not possible:
                raise ValueError(f"insufficient distinct calls for {group}")
            previous_words = len(anchor["text"].split("\n", 1)[0].split())
            row = min(possible, key=lambda row: (
                abs(row["words"] - anchor["words"]),
                abs(len(row["text"].split("\n", 1)[0].split()) - previous_words), order(row)))
            selected.append((group, row, abs(row["words"] - anchor["words"])))
            used_calls.add(row["call"])
            used_text.add(row["text"])
    return sorted(selected, key=lambda item: seed_for("harper-plausibility-display-v1", item[1]["rank"])), pools


def write_panel(root, items, report):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark text stays on Vast")
    if report["checkpoint_sha256"] != CHECKPOINT or report["rendered_inputs_sha256"] != RENDERED_INPUTS:
        raise ValueError("the panel requires the frozen Qwen score and English transcript inputs")
    selected, pools = pick_items(items)
    visible, private = [], {}
    for group, row, match_gap in selected:
        ident = f"text-{seed_for('harper-plausibility-id-v1', row['rank']):016x}"
        previous, current = row["text"].split("\n", 1)
        visible.append({"id": ident, "previous": previous, "current": current})
        length, _, features = strata(row["text"])
        private[ident] = {"group": group, "words": length, "match_gap_words": match_gap,
                          "yes_no": features["affirmation_or_denial_opening"],
                          "filler": features["filler_or_repeated_word"]}
    if len(private) != len(selected):
        raise ValueError("duplicate blind item identifiers")
    PANEL_ROOT.mkdir(mode=0o700)
    packet = PANEL_ROOT / "packet.json"
    reference = PANEL_ROOT / "private-reference.json"
    _write_json(packet, {"mode": "plausibility", "items": visible})
    _write_json(reference, private)
    receipt = {
        "scope": "blind transcript plausibility diagnostic",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "rendered_inputs_sha256": report["rendered_inputs_sha256"],
        "packet_sha256": sha256(packet.read_bytes()).hexdigest(),
        "reference_sha256": sha256(reference.read_bytes()).hexdigest(),
        "selection": dict(Counter(group for group, _, _ in selected)),
        "population": {name: len(rows) for name, rows in pools.items()},
        "exact_word_count_matched_controls": dict(Counter(group for group, _, gap in selected if gap == 0)),
        "by_group_and_yes_no_opening": dict(Counter(f"{row['group']}/{'yes' if row['yes_no'] else 'no'}"
                                                   for row in private.values())),
        "unique_calls": len({row["call"] for _, row, _ in selected}),
        "unique_callers": len({row["caller"] for _, row, _ in selected}),
    }
    _write_json(root / "reports" / RECEIPT_NAME, receipt)
    print(json.dumps(receipt, indent=2))


def score(root, ballot):
    receipt = json.loads((root / "reports" / RECEIPT_NAME).read_text())
    reference_bytes = (PANEL_ROOT / "private-reference.json").read_bytes()
    packet_bytes = (PANEL_ROOT / "packet.json").read_bytes()
    if (sha256(reference_bytes).hexdigest() != receipt["reference_sha256"]
            or sha256(packet_bytes).hexdigest() != receipt["packet_sha256"]):
        raise ValueError("the blind plausibility packet changed")
    reference = json.loads(reference_bytes)
    content = ballot.read_bytes()
    rated = json.loads(content)["draft"]["clips"]
    if set(rated) != set(reference) or any(row.get("outcome") not in RATINGS for row in rated.values()):
        raise ValueError("complete every plausibility item before scoring")
    counts = defaultdict(Counter)
    for ident, row in reference.items():
        outcome = rated[ident]["outcome"]
        group = row["group"]
        opener = "yes" if row["yes_no"] else "no"
        counts[group][outcome] += 1
        counts[f"{group}/affirmative_or_negative_opening/{opener}"][outcome] += 1
    full_count = counts["false_complete"]
    rated_failures = sum(full_count.values()) - full_count["unrateable"]
    result = {"scope": "quota-sampled blinded transcript plausibility diagnostic",
              "ratings_by_stratum": dict(counts), "packet_sha256": receipt["packet_sha256"],
              "false_complete_prefixes_with_usable_ratings": rated_failures,
              "false_complete_judged_clearly_complete": full_count["complete"] / rated_failures
              if rated_failures else None,
              "false_complete_judged_plausibly_or_clearly_complete": (
                  (full_count["could_end"] + full_count["complete"]) / rated_failures if rated_failures else None),
              "ballot_sha256": sha256(content).hexdigest()}
    _write_json(root / "reports/harper-text-plausibility-score.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ballot", type=Path)
    score(data_root(), parser.parse_args().ballot)
