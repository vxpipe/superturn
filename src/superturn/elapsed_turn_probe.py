"""Independently compare causal elapsed duration with frozen Superturn audio scores."""

from collections import defaultdict
import json
from statistics import mean

import numpy as np
import pyarrow.parquet as pq
import torch

from .ami_calibration import _write_json
from .ami_deadline import retrospective_limits
from .audio import RATE, read_audio
from .citrinet_frozen_probe import fit_readout
from .dense_teacher import SAMPLES, file_digest
from .eot_domain_probe import DATASET_SHA256, DECISION_MS, GAP, decision_audio
from .harper_valley_benchmark import STUDENTS
from .model import build
from .multilingual_teacher_probe import DATA
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .train import _auc

BIN_EDGES = (0, 1, 2, 4, 8, float("inf"))
LONG_TURN_SECONDS = 3.3


def metrics(scores, labels, indices):
    truth = labels[indices].tolist()
    row = {"holds": len(truth) - sum(truth), "shifts": sum(truth)}
    if set(truth) == {0, 1}:
        row["models"] = {name: measures(values[indices].tolist(), truth) for name, values in scores.items()}
    return row


def measures(scores, truth):
    return {"auc": _auc(scores, truth),
            **retrospective_limits(scores, truth)}


def paired_intervals(scores, labels, turns, pairs, *, partial_auc=False):
    from .harper_history_probe import recall_area

    grouped = defaultdict(list)
    for index, turn in enumerate(turns):
        grouped[turn].append(index)
    members = tuple(grouped.values())
    rng = np.random.default_rng(6902)
    auc_deltas = {name: [] for name in pairs}
    recall_deltas = {name: [] for name in pairs}
    partial_deltas = {name: [] for name in pairs}
    for drawn in rng.integers(len(members), size=(500, len(members))):
        indices = np.concatenate([members[i] for i in drawn])
        result = metrics(scores, labels, indices).get("models")
        if result:
            for name, baseline in pairs.items():
                reference = result[baseline]
                auc_deltas[name].append(result[name]["auc"] - reference["auc"])
                key = "max_recall_at_5pct_false_complete"
                recall_deltas[name].append(result[name][key] - reference[key])
                if partial_auc:
                    partial_deltas[name].append(recall_area(scores[name][indices], labels[indices])[-1]
                                                - recall_area(scores[baseline][indices], labels[indices])[-1])
    intervals = {name: {"reference": pairs[name], "replicates": len(auc_deltas[name]),
                        "auc_delta_95": np.quantile(auc_deltas[name], [.025, .975]).tolist(),
                        "recall_delta_95": np.quantile(recall_deltas[name], [.025, .975]).tolist()}
                 for name in pairs}
    if partial_auc:
        for name, interval in intervals.items():
            interval["partial_auc_delta_95"] = np.quantile(partial_deltas[name], [.025, .975]).tolist()
    return intervals


def crossfit_long(scores, labels, turns):
    groups = sorted(set(turns), key=lambda turn: seed_for("elapsed-diagnostic-fold", turn))
    by_turn = {turn: i % 5 for i, turn in enumerate(groups)}
    folds = np.asarray([by_turn[turn] for turn in turns])
    predictions = {f"{arm}-{name}": np.full(len(labels), np.nan)
                   for name in STUDENTS for arm in ("audio-calibrated", "combined")}
    for name in STUDENTS:
        p = scores[name].clip(1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        combined = np.column_stack([logit, np.log1p(scores["elapsed"])])
        for role, x in (("audio-calibrated", combined[:, :1]), ("combined", combined)):
            for fold in range(5):
                train, holdout = folds != fold, folds == fold
                if len(np.unique(labels[train])) != 2 or not holdout.any():
                    raise ValueError("the long-turn fold lacks examples or both training classes")
                out, _ = fit_readout(x[train], labels[train], x[holdout])
                predictions[f"{role}-{name}"][holdout] = out
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("incomplete out-of-fold predictions")
    return predictions


def checkpoints(root, device):
    report = root / "reports" / "dense-teacher-gru.json"
    known = json.loads(report.read_text())["fit"]
    models = {}
    for name in STUDENTS:
        path = root / "runs" / "dense-teacher-diagnostic" / name / "best.pt"
        if file_digest(path) != known[name]["checkpoint_sha256"]:
            raise ValueError("the frozen Superturn checkpoint changed")
        stored = torch.load(path, map_location=device, weights_only=True)
        model = build(stored["config"]).to(device).eval()
        model.load_state_dict(stored["state"])
        models[name] = model
    return models, file_digest(report)


def describe_language(source, language, models, device, meter):
    values, labels, turns = defaultdict(list), [], []
    for batch in source.iter_batches(batch_size=8, columns=["id", "audio", "language", "cohort", "silence_spans"]):
        clips = []
        for row in batch.to_pylist():
            if row["language"] != language or row["cohort"] != "general":
                raise ValueError("unexpected language or cohort in elapsed diagnostic")
            pcm = read_audio(row["audio"]["bytes"])
            for span in row["silence_spans"]:
                edge = round(span["start"] * RATE)
                end = min(len(pcm), round(span["end"] * RATE))
                if not 0 <= edge <= end or end - edge < GAP:
                    continue
                if span["label"] not in {"eot", "hold"}:
                    raise ValueError("unexpected pause label")
                values["elapsed"].append((edge + GAP) / RATE)
                labels.append(int(span["label"] == "eot"))
                turns.append(row["id"])
                clips.append(decision_audio(pcm, edge, "telephone")[-SAMPLES:])
        if clips:
            x = torch.as_tensor(np.stack(clips), device=device)
            delay = torch.full((len(x), 1), float(DECISION_MS), device=device)
            with torch.inference_mode():
                for name, model in models.items():
                    values[name].extend(model(x, delay).flatten().cpu().tolist())
        meter.advance(batch.num_rows)
    scores = {name: np.asarray(rows) for name, rows in values.items()}
    if any(len(rows) != len(labels) or not np.isfinite(rows).all() for rows in scores.values()):
        raise ValueError("invalid prepared ranking scores")
    truth = np.asarray(labels)
    elapsed = scores["elapsed"]
    result = {"overall": metrics(scores, truth, np.arange(len(truth))), "by_elapsed_seconds": {
        f"{left:g}–{right:g}": metrics(scores, truth, np.flatnonzero((elapsed >= left) & (elapsed < right)))
        for left, right in zip(BIN_EDGES[:-1], BIN_EDGES[1:], strict=True)}}
    long_turn = elapsed >= LONG_TURN_SECONDS
    long_truth = truth[long_turn]
    long_turns = np.asarray(turns)[long_turn]
    long_scores = {name: values[long_turn] for name, values in scores.items()}
    long_scores.update(crossfit_long(long_scores, long_truth, long_turns))
    result["long_turns"] = metrics(long_scores, long_truth, np.arange(len(long_truth)))
    if language == "en":
        result["paired_turn_bootstrap"] = paired_intervals(scores, truth, turns, {name: "elapsed" for name in STUDENTS})
        comparisons = {f"combined-{name}": f"audio-calibrated-{name}" for name in STUDENTS}
        result["long_turn_paired_bootstrap"] = paired_intervals(long_scores, long_truth, long_turns, comparisons)
    return result


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("elapsed diagnostic uses staged audio on Vast")
    models, checkpoint_report = checkpoints(root, device)
    raw = root / "raw"
    files = {"en": raw / "livekit_eot_diagnostic" / "data/en/validation-00000-of-00002.parquet"}
    files.update({lang: raw / "livekit_multilingual_diagnostic" / "data" / lang / "validation-00000-of-00001.parquet"
                  for lang in DATA})
    hashes = {"en": DATASET_SHA256, **DATA}
    sources = {}
    for language, path in files.items():
        if file_digest(path) != hashes[language]:
            raise ValueError(f"unexpected diagnostic shard: {language}")
        sources[language] = pq.ParquetFile(path)
    meter = Progress("elapsed-turn-probe", sum(source.metadata.num_rows for source in sources.values()), "turns")
    try:
        results = {language: describe_language(source, language, models, device, meter)
                   for language, source in sources.items()}
        outcomes = results["en"]["overall"]
        if (outcomes["holds"], outcomes["shifts"]) != (532, 400):
            raise ValueError("English pause eligibility no longer reproduces")
        key = "max_recall_at_5pct_false_complete"
        previous_en = json.loads((root / "reports" / "dense-teacher-gru.json").read_text())["external"]
        previous_multilingual = json.loads((root / "reports" / "multilingual-teacher-probe.json").read_text())["languages"]
        for language, row in results.items():
            previous = previous_en if language == "en" else previous_multilingual[language]["models"]
            for name in STUDENTS:
                observed = row["overall"]["models"][name]
                if abs(observed["auc"] - previous[name]["auc"]) > 1e-10 or observed[key] != previous[name]["retrospective_roc_limits"][key]:
                    raise ValueError(f"frozen student metrics differ on {language}: {name}")
        macro = {name: mean(results[language]["overall"]["models"][name][key] for language in DATA)
                 for name in ("elapsed", *STUDENTS)}
        long_names = results["en"]["long_turns"]["models"]
        long_macro = {name: mean(results[language]["long_turns"]["models"][name][key] for language in DATA)
                      for name in long_names}
        report = {"scope": "provider-labelled extracted-turn diagnostics", "decision_ms": DECISION_MS,
                  "elapsed_definition": "observed seconds from extracted clip start to decision",
                  "long_turn_min_elapsed_seconds": LONG_TURN_SECONDS,
                  "long_turn_readout": "five-fold held-turn logistic diagnostics, per language",
                  "sha256_by_language": hashes, "student_report_sha256": checkpoint_report,
                  "macro_non_english_recall_at_5pct": macro, "macro_non_english_long_recall_at_5pct": long_macro,
                  "languages": results}
        _write_json(root / "reports" / "elapsed-turn-probe.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = run(data_root(), "cuda")
    print(json.dumps({"english": result["languages"]["en"]["overall"],
                      "english_long": result["languages"]["en"]["long_turns"],
                      "macro_non_english_recall_at_5pct": result["macro_non_english_recall_at_5pct"],
                      "macro_non_english_long_recall_at_5pct": result["macro_non_english_long_recall_at_5pct"]}, indent=2))
