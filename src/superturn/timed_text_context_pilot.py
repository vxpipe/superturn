"""Evaluate matched context-supplement and replay fits on frozen development inputs."""

from collections import defaultdict
from hashlib import sha256
import json

from .ami_calibration import _write_json
from .sources import data_root
from .timed_text_context_audit import hold_interval
from .timed_text_pause_diagnostic import FREEZE, pack_rows
from .turn_text_train import binary_metrics, digest, load_fit, pack_qwen, predict


def by_family(tokenizer, rows, scores):
    groups = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        groups[row["slot"]].append((row["text"], row["label"], score))
    contrasts, invariant = [], []
    for slot, ((left, left_label, a), (right, right_label, b)) in groups.items():
        left_prior, caller = left.split("\n")
        right_prior, alternate_caller = right.split("\n")
        tokens = tokenizer.encode(caller, add_special_tokens=False)
        first, second = pack_qwen(tokenizer, left), pack_qwen(tokenizer, right)
        if (not tokens or caller != alternate_caller or left_prior == right_prior or first == second
                or first[-len(tokens):] != tokens or second[-len(tokens):] != tokens
                or left_label != "hold" or right_label not in {"hold", "shift"}):
            raise ValueError("invalid held-out context pair")
        if right_label == "shift":
            contrasts.append((slot, a, b))
        else:
            invariant.extend((a, b))
    return {
        "contrast_pairs": len(contrasts), "invariant_pairs": len(invariant) // 2,
        "contrast_order_accuracy": sum(1 if a < b else .5 if a == b else 0
                                       for _, a, b in contrasts) / len(contrasts),
        "contrast_hold_false_complete": sum(a >= .5 for _, a, _ in contrasts) / len(contrasts),
        "invariant_hold_false_complete": sum(a >= .5 for a in invariant) / len(invariant),
    }


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark inputs stay on Vast")
    freeze = root / "reports" / FREEZE
    frozen = json.loads(freeze.read_text())
    rows, _, membership = pack_rows(root)
    if (frozen["membership_sha256"] != membership
            or frozen["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()):
        raise ValueError("publisher benchmark inputs changed")
    rows = [row for row in rows if row["has_previous"]]
    bands = {"all": rows, "300_to_799": [row for row in rows if 300 <= row["gap_ms"] < 800],
             "1500_plus": [row for row in rows if row["gap_ms"] >= 1500]}
    report = {"scope": "English development only, fixed 0.5 threshold, one paired seed",
              "freeze_sha256": digest(freeze), "arms": {}}
    predictions = {}
    for arm in ("new", "replay"):
        dataset = root / f"cache/turn-text-context-{arm}-v1"
        manifest = dataset / "manifest.json"
        supplement = json.loads(manifest.read_text())["context_supplement"]
        challenge_path = dataset / "context-heldout.jsonl"
        if (supplement["replay_control"] != (arm == "replay")
                or digest(challenge_path) != supplement["heldout_sha256"]):
            raise ValueError("context arm or challenge checksum mismatch")
        challenge = [json.loads(line) for line in challenge_path.read_text().splitlines()]
        checkpoint = root / f"runs/turn-text-qwen3-context-{arm}-v1/checkpoint.pt"
        tokenizer, model = load_fit(checkpoint, root / "cache/turn-text-qwen3-06b", "cuda", digest(manifest))
        views = {}
        for mode, field in (("context", "text"), ("masked", "masked")):
            scores = predict(tokenizer, model, [row[field] for row in rows], "cuda")
            by_rank = {row["rank"]: score for row, score in zip(rows, scores, strict=True)}
            views[mode] = {band: binary_metrics([(by_rank[row["rank"]], row["label"]) for row in selected])
                           for band, selected in bands.items()}
            predictions[arm, mode] = by_rank
        scores = predict(tokenizer, model, [row["text"] for row in challenge], "cuda")
        report["arms"][arm] = {"checkpoint_sha256": digest(checkpoint), "views": views,
                              "synthetic_heldout": by_family(tokenizer, challenge, scores),
                              "supplement": supplement}
        del model
    if any(report["arms"]["new"]["supplement"][key] != report["arms"]["replay"]["supplement"][key]
           for key in ("heldout_sha256", "added_labels", "added_rows", "parent_manifest_sha256")):
        raise ValueError("context and replay data differ outside the intervention")
    primary = hold_interval([{**row, "score": predictions["new", "context"][row["rank"]]}
                             for row in bands["300_to_799"]], predictions["replay", "context"])
    trial = report["arms"]["new"]
    heldout = trial["synthetic_heldout"]
    report["short_hold_comparison"] = primary
    report["length_strata"] = {}
    for band, selected in bands.items():
        for low, high in ((1, 4), (4, 8), (8, float("inf"))):
            group = [row for row in selected if low <= len(row["text"].split("\n")[1].split()) < high]
            if group:
                report["length_strata"][f"{band}/{low}_to_{high}_caller_words_exclusive"] = {
                    arm: binary_metrics([(predictions[arm, "context"][row["rank"]], row["label"])
                                         for row in group])
                    for arm in ("new", "replay")
                }
    report["gate"] = {
        "short_hold_improvement": (primary["delta_false_complete"] <= -.05
                                   and primary["paired_turn_bootstrap_95"][1] < 0),
        "long_gap_auc_preserved": trial["views"]["context"]["1500_plus"]["auc"] >= .8055,
        "heldout_contrast_order": heldout["contrast_order_accuracy"] >= .70,
        "invariant_holds_guard": heldout["invariant_hold_false_complete"] <= heldout["contrast_hold_false_complete"],
    }
    _write_json(root / "reports/publisher-timed-english-context-pilot.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
