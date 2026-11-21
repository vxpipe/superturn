"""Rank locked Superturn speech heads on the paired language development rows."""

import argparse
from collections import defaultdict
import json

import numpy as np

from .ami_calibration import _write_json
from .ami_transfer import partial_auc
from .dense_teacher import file_digest
from .harper_history_probe import recall_area
from .multilingual_teacher_probe import DATA
from .sources import data_root
from .train import _auc
from .wav2vec_harper_probe import CHECKPOINT_SHA256

BASE = "wav2vec2-base"
GENERAL = {BASE, "hubert-large", "whisper-large-v3", "qwen3-asr-1.7b"}
SPECIALISTS = {"parakeet-tdt-0.6b-v3": {"de", "es", "fr", "it", "nl", "pt"}}
REPLICATES = 500


def verified_pair_rows(root, reports):
    cohorts = {}
    base_path = root / "reports/harper-ami-transfer.json"
    base = json.loads(base_path.read_text())
    base_head = base["head_checkpoint"]
    if base["encoder_sha256"] != CHECKPOINT_SHA256 or file_digest(root / base_head["path"]) != base_head["sha256"]:
        raise ValueError("the baseline caller head differs from its archived receipt")
    inputs = {BASE: {"report_sha256": file_digest(base_path), "head_path": base_head["path"],
                     "head_sha256": base_head["sha256"], "encoder_sha256": CHECKPOINT_SHA256, "arm": "caller"}}
    english = {}
    for path in reports:
        report = json.loads(path.read_text())
        candidate, = set(report["metrics"]) - {BASE}
        if (candidate not in GENERAL | SPECIALISTS.keys() or candidate in inputs or report["decision_ms"] != 300
                or report["encoders"][BASE]["sha256"] != CHECKPOINT_SHA256):
            raise ValueError("unexpected, duplicate or incompatible own-head report")
        checkpoint = report["diagnostic_checkpoint"]
        pair_rows = report["multilingual_paired_rows"]
        pair_path = root / pair_rows["path"]
        if file_digest(pair_path) != pair_rows["sha256"] or file_digest(root / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("head checkpoint or paired prediction digest differs")
        packed = json.loads(pair_path.read_text())
        if packed["dataset_sha256"] != DATA or set(packed["languages"]) != set(DATA):
            raise ValueError("paired predictions do not cover the same pinned development corpus")
        inputs[candidate] = {"report_sha256": file_digest(path), "head_path": checkpoint["path"],
                             "head_sha256": checkpoint["sha256"],
                             "encoder_sha256": report["encoders"][candidate]["sha256"],
                             "pairs_sha256": pair_rows["sha256"]}
        for name, metrics in report["metrics"].items():
            recall = metrics["english"]["recalls_at_2_5_5_7_5pct"][1]
            if name in english and recall != english[name]:
                raise ValueError("archived English reference results disagree")
            english[name] = recall
        for lang, row in packed["languages"].items():
            labels = np.asarray(row["labels"])
            groups = np.asarray(row["turn_ordinals"])
            scores = {name: np.asarray(values) for name, values in row["scores"].items()}
            if (labels.ndim != 1 or set(labels) != {0, 1} or groups.shape != labels.shape
                    or groups.dtype.kind not in "iu" or np.any(groups < 0) or np.any(np.diff(groups) < 0)
                    or set(scores) != {BASE, candidate}
                    or any(values.shape != labels.shape or not np.all((0 <= values) & (values <= 1))
                           for values in scores.values())):
                raise ValueError(f"invalid paired data for {candidate} on {lang}")
            summary = report["multilingual_diagnostic"][lang]
            if (summary["holds"] != (labels == 0).sum() or summary["shifts"] != (labels == 1).sum()
                    or groups[-1] >= summary["turns"]):
                raise ValueError(f"paired row counts differ from the locked report on {lang}")
            for name, values in scores.items():
                computed = readout(values, labels)
                expected = summary["models"][name]
                if not np.allclose(
                    [computed[key] for key in ("auc", "partial_auc_to_5pct", "recall_at_5pct")],
                    [expected["auc"], expected["partial_auc_to_5pct"],
                     expected["retrospective_roc_limits"]["max_recall_at_5pct_false_complete"]],
                    atol=1e-12, rtol=0,
                ):
                    raise ValueError(f"{name} paired probabilities disagree with its report on {lang}")
            previous = cohorts.get(lang)
            if previous is None:
                cohorts[lang] = (labels, groups, scores)
            else:
                old_labels, old_groups, old_scores = previous
                if not all(np.array_equal(a, b) for a, b in ((labels, old_labels), (groups, old_groups),
                                                            (scores[BASE], old_scores[BASE]))):
                    raise ValueError(f"{candidate} does not match the shared reference rows on {lang}")
                old_scores[candidate] = scores[candidate]
    return cohorts, inputs, english


def readout(scores, labels):
    point = recall_area(scores, labels)
    r = point[:3]
    return {"auc": _auc(scores.tolist(), labels.tolist()),
            "partial_auc_to_5pct": partial_auc(scores.tolist(), labels.tolist(), max_fpr=.05),
            "partial_auc_to_10pct": float(point[-1]),
            "recall_at_5pct": float(r[1]), "mean_recall_at_2_5_5_7_5pct": float(r.mean())}


def paired_recall(candidate, reference, truth, groups):
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    turns = tuple(members.values())
    rng = np.random.default_rng(6902)
    deltas = []
    for draw in rng.integers(len(turns), size=(REPLICATES, len(turns))):
        idx = np.concatenate([turns[i] for i in draw])
        if len(np.unique(truth[idx])) != 2:
            continue
        a = recall_area(candidate[idx], truth[idx])[:3]
        b = recall_area(reference[idx], truth[idx])[:3]
        deltas.append((a - b).mean())
    if not deltas:
        raise ValueError("paired bootstrap has no two-class turn resample")
    return {"turns": len(turns), "replicates": len(deltas), "interval_95": np.quantile(deltas, [.025, .975]).tolist()}


def select(cohorts, english):
    metrics = {}
    for lang, (truth, groups, scores) in sorted(cohorts.items()):
        metrics[lang] = {name: readout(values, truth) for name, values in scores.items()
                         if name in GENERAL or lang in SPECIALISTS[name]}
    common = [lang for lang, (truth, _, _) in sorted(cohorts.items()) if min(np.bincount(truth)) >= 30]
    if not common:
        raise ValueError("no language has at least 30 labelled holds and shifts")
    candidates = sorted(GENERAL & set.intersection(*(set(metrics[lang]) for lang in common)))
    if len(candidates) < 2:
        raise ValueError("ranking needs at least two general heads on the common development languages")
    macro = {name: float(np.mean([metrics[lang][name]["partial_auc_to_5pct"] for lang in common]))
             for name in candidates}
    sensitivity = {name: float(np.mean([metrics[lang][name]["partial_auc_to_10pct"] for lang in common]))
                   for name in candidates}
    default = max(candidates, key=lambda name: (macro[name], english[name]))
    table = {}
    for lang, (truth, groups, scores) in sorted(cohorts.items()):
        holds, shifts = int((truth == 0).sum()), int((truth == 1).sum())
        reference = metrics[lang][default]
        alternatives = {}
        for name, value in sorted(metrics[lang].items()):
            if name == default:
                continue
            auc_gain = value["partial_auc_to_5pct"] - reference["partial_auc_to_5pct"]
            recall_gain = value["mean_recall_at_2_5_5_7_5pct"] - reference["mean_recall_at_2_5_5_7_5pct"]
            result = {"partial_auc_gain": auc_gain, "mean_recall_gain": recall_gain, "passes_screen": False}
            if holds >= 100 and shifts >= 50 and auc_gain + 1e-12 >= .03 and recall_gain + 1e-12 >= .03:
                result["paired_mean_recall"] = paired_recall(scores[name], scores[default], truth, groups)
                result["passes_screen"] = result["paired_mean_recall"]["interval_95"][0] > 0
            alternatives[name] = result
        passing = [name for name, result in alternatives.items() if result["passes_screen"]]
        winner = max(passing, key=lambda name: metrics[lang][name]["partial_auc_to_5pct"]) if passing else default
        table[lang] = {"holds": holds, "shifts": shifts, "thin_low_cutoff_evidence": holds < 100 or shifts < 50,
                       "heads": metrics[lang], "alternatives": alternatives, "provisional_teacher": winner}
    return {"scope": "paired provider-labelled development only, independent confirmation required",
            "decision_ms": 300, "macro_languages": common, "macro_partial_auc_to_5pct": macro,
            "macro_partial_auc_to_10pct": sensitivity, "english_tiebreak_recall_at_5pct": english,
            "provisional_default": default, "languages": table}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="own encoder report file names under the campaign reports directory")
    args = parser.parse_args()
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("language ranking uses the private paired campaign reports")
    rows, inputs, english = verified_pair_rows(root, [root / "reports" / filename for filename in args.reports])
    result = {**select(rows, english), "inputs": inputs, "development_corpus_sha256": DATA}
    _write_json(root / "reports/own-teacher-language-selection.json", result)
    print(json.dumps(result, indent=2))
