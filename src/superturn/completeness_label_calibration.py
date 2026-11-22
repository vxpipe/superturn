"""Calibrate an open semantic labeller against verified training-role pauses."""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .audio_judge import MODEL, REVISION
from .harper_transcript_coverage import MEMBERSHIP, caller_timing, coverage
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .sources import data_root
from .text_value_ceiling import PROMPT, scorer

JOB = "completeness-label-calibration"
LABELS = ("hold", "shift")
OUTCOMES = {"aligned_segment_final", "aligned_segment_interior", "prefix_agrees_segment_differs"}


def prefix(words, missing, edge):
    condition, _ = coverage(words, missing, edge)
    if condition not in OUTCOMES:
        return "", condition
    anchor = max((word for word in words if word[1] <= edge), key=lambda word: word[1])
    _, _, index, _, _, corrected = anchor
    return " ".join(corrected[:index + 1]), condition


def length_band(text):
    length = len(text.split())
    return "1_to_2" if length <= 2 else "3_to_6" if length <= 6 else "7_or_more"


def summarize(labels, texts, scores):
    counts = defaultdict(Counter)
    by_length = defaultdict(Counter)
    for truth, text, score in zip(labels, texts, scores, strict=True):
        group = counts[truth]
        group["reference"] += 1
        if not text or not np.isfinite(score) or score == 0:
            group["abstained"] += 1
            continue
        guess = "shift" if score > 0 else "hold"
        group["classified"] += 1
        group["correct"] += guess == truth
        cell = by_length[length_band(text)]
        cell[f"classified_{truth}"] += 1
        cell[f"correct_{truth}"] += guess == truth
    all_rows = sum(counts[label]["reference"] for label in LABELS)
    observed = sum(counts[label]["classified"] for label in LABELS)
    correct = sum(counts[label]["correct"] for label in LABELS)
    class_agreement = {label: (group["correct"] / group["classified"] if group["classified"] else None)
                       for label, group in counts.items()}
    coverage = {label: group["classified"] / group["reference"] for label, group in counts.items()}
    agreement = correct / observed if observed else None
    return {"counts": {label: dict(counts[label]) for label in LABELS},
            "agreement_of_classified": agreement, "class_agreement": class_agreement,
            "correct_of_all_reference_rows": correct / all_rows,
            "coverage": coverage, "counts_by_length": {key: dict(cell) for key, cell in by_length.items()},
            "selective_gate_passes": bool(agreement is not None and agreement >= .85
                           and all(class_agreement[label] is not None and class_agreement[label] >= .80
                                   for label in LABELS)),
            "all_rows_gate_passes": bool(correct / all_rows >= .85
                                         and all(counts[label]["correct"] / counts[label]["reference"] >= .80
                                                 for label in LABELS))}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("transcripts stay on Vast")
    cases, _, audit_sha = pinned_split(root, "train")
    reports = root / "reports"
    membership = json.loads((reports / "harper-history-replication-membership.json").read_text())
    if hashlib.sha256(json.dumps(membership).encode()).hexdigest() != MEMBERSHIP:
        raise ValueError("the verified HarperValley cohort changed")
    if Counter(row[-1] for row in membership) != {"hold": 273, "shift": 255}:
        raise ValueError("the calibration label coverage changed")
    callers = {row["id"]: row["caller"] for row in cases}
    texts, labels, conditions, voices = [], [], [], set()
    transcript_cache = {}
    attrition = defaultdict(Counter)
    meter = Progress(JOB, len(membership), "prefixes")
    try:
        for call, edge, label in membership:
            voice = callers[call]
            if call not in transcript_cache:
                file = root / "raw/harper-valley/data/transcript" / f"{call}.json"
                transcript_cache[call] = caller_timing(json.loads(file.read_text()))
            text, condition = prefix(*transcript_cache[call], edge)
            texts.append(text)
            labels.append(label)
            conditions.append(condition)
            attrition[label][condition] += 1
            if text:
                voices.add(voice)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    scores, weights, checks = scorer(root, {"caller": texts}, text_of=str)
    metrics = summarize(labels, texts, scores["caller"])
    strict = [text if condition.startswith("aligned_") else ""
              for text, condition in zip(texts, conditions, strict=True)]
    boundary = defaultdict(Counter)
    for text, label, condition, score in zip(texts, labels, conditions, scores["caller"], strict=True):
        if text:
            boundary[condition]["eligible_" + label] += 1
            if np.isfinite(score) and score:
                boundary[condition]["classified_" + label] += 1
                boundary[condition]["correct_" + label] += int((score > 0) == (label == "shift"))
    checks_pass = checks["finished_example_log_odds"] > 0 > checks["unfinished_example_log_odds"]
    result = {"scope": "semantic label calibration against HarperValley behavioral labels, train roles only",
              "model": MODEL, "revision": REVISION, "weight_sha256": weights, "known_answer_checks": checks,
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "harper_audit_sha256": audit_sha,
              "pause_membership_sha256": MEMBERSHIP, "callers_with_text": len(voices),
              "prefix_conditions_by_reference_label": attrition, "by_prefix_condition": boundary,
              "strict_alignment": summarize(labels, strict, scores["caller"]), **metrics}
    result["passes"] = checks_pass and metrics["all_rows_gate_passes"]
    _write_json(reports / f"{JOB}.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
