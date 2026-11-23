"""Measure the effect of preceding context on identical publisher-timed pauses."""

from collections import defaultdict
from hashlib import sha256
import json

import numpy as np

from .ami_calibration import _write_json
from .harper_text_comparison import paired_intervals
from .harper_text_transcript_benchmark import report_for
from .sources import data_root
from .timed_text_pause_diagnostic import FREEZE, pack_rows
from .turn_text_train import digest, load_fit, predict


def hold_interval(items, baseline):
    by_turn = defaultdict(list)
    for row in items:
        if row["label"] == 0:
            delta = int(row["score"] >= .5) - int(baseline[row["rank"]] >= .5)
            by_turn[row["caller"]].append(delta)
    counts = np.asarray([(sum(values), len(values)) for values in by_turn.values()])
    draws = np.random.default_rng(29).integers(len(counts), size=(2000, len(counts)))
    errors, totals = counts[draws].sum(axis=1).T
    observed_errors, observed_holds = counts.sum(axis=0)
    return {"holds": int(observed_holds), "turns": len(counts),
            "delta_false_complete": float(observed_errors / observed_holds),
            "paired_turn_bootstrap_95": np.quantile(errors / totals, [.025, .975]).tolist()}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark inputs stay on Vast")
    freeze_path = root / "reports" / FREEZE
    frozen = json.loads(freeze_path.read_text())
    rows, _, membership = pack_rows(root)
    checkpoint = root / "runs/turn-text-qwen3-focused-v1/checkpoint.pt"
    if (frozen["membership_sha256"] != membership
            or frozen["rendered_inputs_sha256"] != sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            or frozen["checkpoint_sha256"]["qwen"] != digest(checkpoint)):
        raise ValueError("publisher context audit inputs changed")
    rows = [row for row in rows if row["has_previous"]]
    manifest = root / "cache/turn-text-focused-en-v1/manifest.json"
    tokenizer, model = load_fit(checkpoint, root / "cache/turn-text-qwen3-06b", "cuda", digest(manifest))
    context = predict(tokenizer, model, [row["text"] for row in rows], "cuda")
    masked = predict(tokenizer, model, [row["masked"] for row in rows], "cuda")
    scored = [{**row, "score": score} for row, score in zip(rows, context, strict=True)]
    baseline = {row["rank"]: score for row, score in zip(rows, masked, strict=True)}
    views = {}
    for name, lo, hi in (("all", 300, float("inf")), ("300_to_799", 300, 800),
                         ("800_to_1499", 800, 1500), ("1500_plus", 1500, float("inf"))):
        selected = [row for row in scored if lo <= row["gap_ms"] < hi]
        control = {row["rank"]: baseline[row["rank"]] for row in selected}
        views[name] = {
            "with_previous": report_for(selected),
            "current_only": report_for([{**row, "score": control[row["rank"]]} for row in selected]),
            "hold_operating_point": hold_interval(selected, control),
            "both_class_comparison": paired_intervals(selected, control)
            if {row["label"] for row in selected} == {0, 1} else None,
        }
        for view in ("with_previous", "current_only"):
            views[name][view]["turns"] = views[name][view].pop("calls")
            views[name][view].pop("callers")
        if delta := views[name]["both_class_comparison"]:
            delta["turns"] = delta.pop("callers")
            delta["paired_turn_bootstrap_95"] = delta.pop("paired_caller_bootstrap_95")
    report = {"scope": "same-row Qwen context ablation; fixed 0.5 and publisher pause lengths",
              "freeze_sha256": digest(freeze_path), "views": views}
    _write_json(root / "reports/publisher-timed-english-context-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
