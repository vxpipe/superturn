"""Measure first shift decisions inside a 100–300 ms AMI validation window."""

import argparse
from collections import defaultdict
from dataclasses import fields
from functools import lru_cache
import hashlib
from itertools import groupby
import json

import numpy as np

from .ami_calibration import _write_json
from .audio import read_audio
from .augment import condition
from .labels import Label
from .progress import Progress
from .shards import CONTEXT, clip
from .sources import data_root

TIMES = (100, 200, 300)


def retrospective_limits(scores, truth):
    """Describe the empirical ROC envelope, keeping equal scores together."""
    positions = sorted(zip(scores, truth, strict=True), reverse=True)
    shifts = sum(truth)
    holds = len(truth) - shifts
    if not shifts or not holds:
        raise ValueError("operating range requires holds and shifts")
    completed, cutoffs = 0, 0
    max_recall = 0.0
    min_false_complete = None
    for _, tied in groupby(positions, key=lambda row: row[0]):
        labels = [label for _, label in tied]
        positives = sum(labels)
        completed += positives
        cutoffs += len(labels) - positives
        recall, false_complete = completed / shifts, cutoffs / holds
        if false_complete <= 0.05:
            max_recall = recall
        if min_false_complete is None and recall >= 0.80:
            min_false_complete = false_complete
    return {"max_recall_at_5pct_false_complete": max_recall,
            "min_false_complete_at_80pct_recall": min_false_complete}


def pooled_rates(outcomes, policy):
    summary = {}
    for deadline in TIMES:
        metrics = [entry[policy][deadline] for entry in outcomes]
        summary[deadline] = {
            key: sum(row[key] * row[count] for row in metrics) / sum(row[count] for row in metrics)
            for key, count in (("recall", "shifts"), ("false_complete", "holds"))
        }
    return summary


def inputs(root, rows, samples, cached_audio):
    names = [field.name for field in fields(Label)]
    by_time, truth = defaultdict(list), []
    for row in rows:
        label = Label(**{name: row[name] for name in names})
        if label.pause_samples < TIMES[-1] * 16:
            raise ValueError("deadline comparison expects every row to reach 300 ms")
        original = clip(label, cached_audio(root / label.audio))
        for elapsed in TIMES:
            end = CONTEXT + elapsed * 16
            prefix = original.copy()
            prefix[end:] = 0
            rendered = condition(prefix, "telephone", 0)
            if elapsed == TIMES[-1] and rendered.tobytes() != row["pcm"]:
                raise ValueError("300 ms render differs from the archived fold")
            by_time[elapsed].append(rendered[end - samples:end])
        truth.append(int(label.label == "shift"))
    return {t: np.stack(values).astype(np.float32) / 32768 for t, values in by_time.items()}, truth


def run(root, source, variant):
    import torch

    from .model import build
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI deadline diagnostic uses the training server")
    if len(source) != 16 or any(char not in "0123456789abcdef" for char in source):
        raise ValueError("cross-validation report ID must be 16 lowercase hex characters")
    origin = root / "reports" / "ami-crossval" / f"{source}.json"
    original = json.loads(origin.read_text())
    variants = original.get("summary", {})
    if variant not in variants:
        raise ValueError("variant is absent from the completed cross-validation report")
    torch.set_num_threads(1)
    folds = root / "shards" / "ami-crossval" / source
    exports = root / "exports" / "ami-crossval" / source
    runs = [entry for entry in original["runs"] if entry["variant"] == variant]
    outcomes = []
    cache = lru_cache(maxsize=6)(read_audio)
    meter = Progress("ami-deadline", len(runs), "models")
    try:
        for fold in sorted({entry["fold"] for entry in runs}):
            folder = folds / f"fold-{fold}"
            first = next(entry for entry in runs if entry["fold"] == fold)
            checkpoint = exports / f'cv-{fold}-{variant}-{first["seed"]}' / "best.pt"
            sample_count = round(torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]["window"] * 16000)
            data = {}
            for split in ("validation", "out-of-fold"):
                rows = [row for batch in records(paths(folder, split)) for row in batch]
                data[split] = inputs(root, rows, sample_count, cache)
            for entry in (row for row in runs if row["fold"] == fold):
                checkpoint = exports / f'cv-{fold}-{variant}-{entry["seed"]}' / "best.pt"
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["checkpoint_sha256"]:
                    raise ValueError(f"checkpoint changed: {checkpoint.parent.name}")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = build(saved["config"]).eval()
                model.load_state_dict(saved["state"])
                scores = {}
                with torch.inference_mode():
                    for split, (clips, _) in data.items():
                        scores[split] = np.stack([
                            model(torch.from_numpy(clips[t]), torch.full((len(clips[t]), 1), float(t)))
                            .flatten().numpy() for t in TIMES
                        ])
                calibration_truth = data["validation"][1]
                test_truth = data["out-of-fold"][1]
                timing = {}
                for i, deadline in enumerate(TIMES, 1):
                    calibration = scores["validation"][:i].max(axis=0).tolist()
                    evaluation = scores["out-of-fold"][:i].max(axis=0).tolist()
                    threshold = _threshold(calibration, calibration_truth)
                    timing[deadline] = {"threshold": threshold, **_stats(evaluation, test_truth, threshold)}
                fixed_threshold = timing[TIMES[-1]]["threshold"]
                fixed_policy = {
                    deadline: _stats(scores["out-of-fold"][:i].max(axis=0).tolist(), test_truth, fixed_threshold)
                    for i, deadline in enumerate(TIMES, 1)
                }
                fixed = _stats(scores["out-of-fold"][-1].tolist(), test_truth, entry["threshold"])
                outcomes.append({"fold": fold, "seed": entry["seed"], "cumulative": timing, "300_ms_only": fixed,
                                 "fixed_300_ms_cumulative_policy": fixed_policy,
                                 "retrospective_limits_300_ms": retrospective_limits(
                                     scores["out-of-fold"].max(axis=0).tolist(), test_truth)})
                meter.advance(1)
        result = {
            "crossval": source, "variant": variant, "summary": pooled_rates(outcomes, "cumulative"),
            "fixed_300_ms_cumulative_policy": pooled_rates(outcomes, "fixed_300_ms_cumulative_policy"),
            "runs": outcomes,
            "scope": "out-of-fold AMI reference pauses reaching 300 ms; earlier queries omit future audio",
        }
        destination = root / "reports" / "ami-deadline" / f"{source}-{variant}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, result)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crossval", help="completed AMI cross-validation report ID")
    parser.add_argument("variant", help="trained model variant in that report")
    args = parser.parse_args()
    print(run(data_root(), args.crossval, args.variant))
