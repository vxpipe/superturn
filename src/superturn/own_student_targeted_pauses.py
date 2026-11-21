"""Build and test matched training additions from pinned, novel AMI pauses."""

import argparse
from collections import Counter, defaultdict
import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import SAMPLES, file_digest
from .elapsed_turn_probe import paired_intervals
from .labels import Label
from .own_head_distillation import student_scores
from .own_student_real_labels import SEEDS, checked_inputs, compare, fit
from .pitch_check import yin_pitch
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import GAP
from .wav2vec_harper_probe import render_ami, verified_ami_labels

CAP = 2


def target_types(row, clip):
    edge = SAMPLES - GAP
    history = row["prior_turn_samples"]
    kinds = []
    if history is not None and history < 2.7 * RATE:
        kinds.append("short_history")
    tail = float(np.mean(np.square(clip[edge - RATE // 5:edge])))
    if row["label"]["label"] == "hold":
        if tail < 10 ** (-35 / 10):
            kinds.append("quiet_hold")
    else:
        reference = float(np.mean(np.square(clip[edge - RATE:edge - RATE // 5])))
        if tail > 0 and reference > 0 and abs(10 * np.log10(tail / reference)) <= 6:
            kinds.append("steady_shift")
        early, late = [], []
        for start in range(edge - 9 * RATE // 10, edge - 400, 640):
            hz = yin_pitch(clip[start:start + 400])
            if np.isfinite(hz):
                target = late if start >= edge - 35 * RATE // 100 else early
                target.append(hz)
        if len(early) >= 3 and len(late) >= 3 and np.median(late) / np.median(early) > 2 ** (2 / 12):
            kinds.append("rising_shift")
    return kinds


def selected_rows(root, registry):
    files = root / "raw/ami-hard-pauses"
    manifest = files / "MANIFEST.json"
    source = root / "labels/ami-hard-pauses.json"
    ledger = json.loads(source.read_text())
    plan = root / "reports/ami-hard-pauses-plan.json"
    if ledger["manifest_sha256"] != file_digest(manifest) or ledger["plan_sha256"] != file_digest(plan):
        raise ValueError("the selective AMI source differs from its receipt")
    for row in json.loads(manifest.read_text())["files"]:
        if file_digest(files / row["path"]) != row["sha256"]:
            raise ValueError("a selective AMI file changed")
    new = ledger["pauses"]
    labels = [Label(**row["label"]) for row in new]
    if not labels or any(row.split != "train" for row in labels):
        raise ValueError("the supplemental AMI source lacks verified train pauses")
    if len({row["pause_id"] for row in new}) != len(new):
        raise ValueError("duplicate supplemental pause")
    original = defaultdict(list)
    for index, row in enumerate(registry):
        original[row.speaker, row.label].append(index)
    target, ordinary, stats = defaultdict(list), defaultdict(list), Counter()
    for index, (row, (clip, _, _)) in enumerate(zip(new, render_ami(root, labels), strict=True)):
        kinds = target_types(row, clip)
        row["target_types"] = kinds
        key = row["label"]["speaker"], row["label"]["label"]
        if kinds:
            target[key].append((index, clip))
            stats.update(kinds)
        else:
            ordinary[key].append((index, clip))
    replacements = {"target": {}, "target-matched": {}, "ordinary": {}}
    choices = []
    for key in sorted(target):
        old = sorted(original[key], key=lambda i: seed_for("ami-targeted-replace-v1", registry[i].audio,
                                                            registry[i].pause_start))
        rank = lambda item: seed_for("ami-targeted-order-v1", new[item[0]]["pause_id"])
        hard = sorted(target[key], key=lambda item: ("short_history" not in new[item[0]]["target_types"], rank(item)))
        typical = sorted(ordinary[key], key=rank)
        for ordinal, (baseline_index, match) in enumerate(zip(old[:CAP], hard)):
            index, clip = match
            replacements["target"][baseline_index] = clip
            ordinary_id = None
            if ordinal < len(typical):
                other, normal_clip = typical[ordinal]
                ordinary_id = new[other]["pause_id"]
                replacements["target-matched"][baseline_index] = clip
                replacements["ordinary"][baseline_index] = normal_clip
            choices.append({"original_id": [registry[baseline_index].audio, registry[baseline_index].pause_start],
                            "speaker": key[0], "class": key[1], "target_pause_id": new[index]["pause_id"],
                            "ordinary_pause_id": ordinary_id, "target_types": new[index]["target_types"]})
    receipt = {"version": 1, "source_ledger_sha256": file_digest(source),
               "canonical_ami_ids": [[row.audio, row.pause_start, row.label] for row in registry],
               "replacements": choices}
    path = root / "reports/ami-targeted-training-dataset.json"
    if path.is_file() and json.loads(path.read_text()) != receipt:
        raise ValueError("rebuild selected different training examples")
    _write_json(path, receipt)
    matched = [row for row in choices if row["ordinary_pause_id"] is not None]
    result = {"independently_supported": len(new), "target_type_counts": stats,
              "new_target_rows": len(choices), "target_speakers": len({row["speaker"] for row in choices}),
              "target_classes": Counter(row["class"] for row in choices),
              "matched_new_pairs": len(matched), "matched_paired_voices": len({row["speaker"] for row in matched}),
              "matched_target_classes": Counter(row["class"] for row in matched),
              "matched_target_types": Counter(t for row in matched for t in row["target_types"]),
              "dataset_sha256": file_digest(path)}
    _write_json(root / "reports/ami-targeted-training-coverage.json", result)
    return replacements, result


def run(root):
    registry = verified_ami_labels(root, "train")
    replacements, coverage = selected_rows(root, registry)
    if not replacements["target"]:
        return {"coverage": coverage, "status": "no new target matched a fit voice and label"}
    receipt, parent, data, replay, _, _ = checked_inputs(root)
    output = root / "runs/own-student-targeted-pauses"
    output.mkdir(parents=True, exist_ok=True)
    predictions, records = {}, {}
    torch.use_deterministic_algorithms(True)
    base_x, labels, groups = data["ami-train"]
    for arm, changes in (("unchanged", {}), *replacements.items()):
        if arm != "unchanged" and not changes:
            continue
        clips = base_x.copy()
        for i, replacement in changes.items():
            clips[i] = replacement
        inputs = {**data, "ami-train": (clips, labels, groups)}
        directory = output / arm
        directory.mkdir(exist_ok=True)
        for seed in SEEDS:
            name = f"{arm}-{seed}"
            initial = parent["fits"][f"feature-first-{seed}"]["checkpoint"]
            model, records[name], _ = fit(root, initial, "full-ami", seed, inputs, replay, output_dir=directory)
            predictions[name] = {role: student_scores(model, data[role][0])
                                 for role in ("english", "harper-validation", "ami-calibration")}
    metrics = compare(predictions, data)
    archived = json.loads((root / "reports/own-student-real-labels.json").read_text())["metrics"]
    for seed in SEEDS:
        actual = metrics[f"unchanged-{seed}"]["english"]["recalls_at_2_5_5_7_5pct"]
        if actual != archived[f"full-ami-{seed}"]["english"]["recalls_at_2_5_5_7_5pct"]:
            raise ValueError("the original training control did not reproduce")
    y, turns = data["english"][1:]
    paired = {name: values["english"] for name, values in predictions.items() if name.endswith("-29")}
    comparisons = {"target-29": "unchanged-29"}
    if "ordinary-29" in paired:
        comparisons["target-matched-29"] = "ordinary-29"
    intervals = paired_intervals(paired, y, turns, comparisons, partial_auc=True)
    result = {"scope": "matched novel training-only sample replacements; English development",
              "coverage": coverage, "parent_fit_report_sha256": file_digest(receipt),
              "metrics": metrics, "intervals": intervals, "checkpoints": records}
    _write_json(root / "reports/own-student-targeted-pauses.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the training dataset build requires staged Vast recordings")
    result = (selected_rows(root, verified_ami_labels(root, "train"))[1]
              if args.dataset_only else run(root))
    print(json.dumps(result, indent=2))
