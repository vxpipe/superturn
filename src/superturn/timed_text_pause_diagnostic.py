"""Score frozen text fits on existing publisher-timed English pauses."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .audio import RATE
from .harper_text_comparison import paired_intervals
from .harper_text_transcript_benchmark import length_matched, report_for
from .sources import data_root
from .turn_generation import normalize
from .turn_text_train import digest, load_fit, predict
from .turn_training_dataset import model_text

SHARD = "raw/livekit_eot_diagnostic/data/en/validation-00000-of-00002.parquet"
FREEZE = "publisher-timed-english-text-freeze-v1.json"


def pause_prefix(row, stop):
    words = sorted((round(word["end"] * RATE), round(word["start"] * RATE), word["word"])
                   for word in row["words"] or []
                   if word["word"] and word["start"] is not None and word["end"] is not None
                   and 0 <= word["start"] < word["end"] and round(word["end"] * RATE) <= stop)
    current = normalize(" ".join(word for _, _, word in words))
    earlier = row["messages"] or []
    prior = earlier[-1].get("content") if earlier and earlier[-1].get("role") == "assistant" else None
    previous = normalize(prior) if isinstance(prior, str) else ""
    return previous, current


def pack_rows(root):
    from .harper_partial_stt import english_membership
    from .turn1_mini_probe import DATASET_SHA256

    source = root / SHARD
    if digest(source) != DATASET_SHA256:
        raise ValueError("English publisher shard changed from its benchmark revision")
    members, tracks, membership_sha = english_membership(root)
    transcripts = pq.read_table(source, columns=["words", "messages", "silence_spans"]).to_pylist()
    report = Counter()
    rows = []
    for index, group, edge, label, _ in members:
        report[f"{label}/published"] += 1
        previous, current = pause_prefix(transcripts[group], edge)
        if not current:
            report[f"{label}/empty_caller_prefix"] += 1
            continue
        if not previous:
            report[f"{label}/previous_missing_or_unconfirmed"] += 1
        _, guarded = pause_prefix(transcripts[group], edge - RATE // 20)
        if not guarded:
            report[f"{label}/empty_caller_with_50ms_guard"] += 1
        span, = [span for span in transcripts[group]["silence_spans"]
                 if round(span["start"] * RATE) == edge and (span["label"] == "eot") == (label == "shift")]
        gap_ms = (min(round(span["end"] * RATE), len(tracks[group])) - edge) * 1000 / RATE
        rows.append({
            "rank": index, "call": group, "caller": group, "label": int(label == "shift"),
            "text": model_text(previous, current), "masked": model_text("", current),
            "guarded": model_text("", guarded) if guarded else None,
            "has_previous": bool(previous), "words": len(current.split()), "context": bool(previous), "gap_ms": gap_ms,
        })
    if (report["hold/published"], report["shift/published"]) != (532, 400):
        raise ValueError("the English diagnostic missed archived classes")
    return rows, dict(report), membership_sha


def run(root, dataset_manifest, qwen_base, qwen_checkpoint, e5_base, e5_checkpoint, *, freeze_only=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark inputs stay on Vast")
    rows, attrition, membership = pack_rows(root)
    frozen = {
        "source": "LiveKit End-of-Turn Benchmark data, LiveKit (CC BY 4.0)",
        "scoring_code_sha256": digest(Path(__file__)),
        "source_sha256": digest(root / SHARD),
        "membership_sha256": membership,
        "rendered_inputs_sha256": sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": {"qwen": digest(qwen_checkpoint), "e5": digest(e5_checkpoint)},
        "input_attrition": attrition,
        "decision": "paired turn-bootstrap lower 95% bound on Qwen-minus-E5 BA at 0.5 above zero",
        "primary_views": ["current_only", "available_previous"],
        "secondary_views": ["current_only_50ms_guard", "current_only_exact_word_count"],
        "cluster_unit": "publisher turn (no persistent focal caller identity)",
    }
    freeze = root / "reports" / FREEZE
    if freeze_only:
        if freeze.exists() and json.loads(freeze.read_text()) != frozen:
            raise ValueError("the English pause freeze already has different inputs")
        _write_json(freeze, frozen)
        print(json.dumps(frozen, indent=2))
        return
    if json.loads(freeze.read_text()) != frozen:
        raise ValueError("the English pause diagnostic differs from the frozen review")
    fields = {"current_only": "masked", "available_previous": "text",
              "current_only_50ms_guard": "guarded"}
    models = {}
    for name, base, checkpoint in (("qwen", qwen_base, qwen_checkpoint), ("e5", e5_base, e5_checkpoint)):
        tokenizer, model = load_fit(checkpoint, base, "cuda", digest(dataset_manifest))
        contexts = {
            "available_previous": [row for row in rows if row["has_previous"]],
            "current_only": rows,
            "current_only_50ms_guard": [row for row in rows if row["guarded"]],
        }
        results = {}
        for view, items in contexts.items():
            field = fields[view]
            scores = predict(tokenizer, model, [row[field] for row in items], "cuda")
            results[view] = [
                {**row, "score": score, "words": len(row[field].split("\n", 1)[1].split())}
                for row, score in zip(items, scores, strict=True)
            ]
        models[name] = results
        results["current_only_exact_word_count"] = length_matched(results["current_only"])
        del tokenizer, model
    report = {
        "scope": "publisher-timed English human-to-agent development diagnostic",
        "timing": "publisher silence span starts; word ends up to pause start; eligible at 300 ms",
        "threshold_source": "frozen 0.5 from synthetic training",
        **frozen,
        "freeze_sha256": digest(freeze),
        "views": {},
    }
    for view, qwen_items in models["qwen"].items():
        e5_items = models["e5"][view]
        main, baseline = report_for(qwen_items), report_for(e5_items)
        for scorecard in (main, baseline):
            scorecard["turns"] = scorecard.pop("calls")
            scorecard.pop("callers")
        delta = paired_intervals(qwen_items, {row["rank"]: row["score"] for row in e5_items})
        delta["turns"] = delta.pop("callers")
        delta["paired_turn_bootstrap_95"] = delta.pop("paired_caller_bootstrap_95")
        report["views"][view] = {"qwen": main, "e5": baseline, "qwen_minus_e5": delta}
    labels_by_text = {}
    for row in rows:
        labels_by_text.setdefault(row["masked"], set()).add(row["label"])
    report["same_text_conflicting_labels_current_only"] = sum(len(labels) == 2 for labels in labels_by_text.values())
    report["current_only_qwen_by_published_gap_ms"] = {}
    for lower, upper in ((300, 800), (800, 1500), (1500, float("inf"))):
        name = f"[{lower},{upper})"
        scorecard = report_for([row for row in models["qwen"]["current_only"] if lower <= row["gap_ms"] < upper])
        scorecard["turns"] = scorecard.pop("calls")
        scorecard.pop("callers")
        report["current_only_qwen_by_published_gap_ms"][name] = scorecard
    _write_json(root / "reports/publisher-timed-english-text-diagnostic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    parser.add_argument("--dataset-manifest", type=Path,
                        default=root / "cache/turn-text-focused-en-v1/manifest.json")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    run(root, args.dataset_manifest, root / "cache/turn-text-qwen3-06b",
        root / "runs/turn-text-qwen3-focused-v1/checkpoint.pt", root / "cache/turn-text-e5-small",
        root / "runs/turn-text-e5-focused-en-v1/checkpoint.pt", freeze_only=args.freeze_only)
