"""Fit and score a fixed single-reviewer Hindi supplement for the owned teacher."""

import argparse
from collections import Counter
import json

import numpy as np
from scipy.special import expit
import torch

from .audio import RATE
from .balanced_development_scorecard import (
    UNSEEN_IN_FIT, balanced_threshold, calibration_split, cell, evaluate, frozen, macro,
)
from .balanced_scorecard_references import multilingual
from .dense_teacher import file_digest
from .eot_domain_probe import decision_audio
from .hindi_review_pilot import JOB, SOURCE_SHA256
from .language_failure_audit import read_receipt, read_verified
from .owned_history_teacher import encoder, training
from .own_student_wider_e20 import by_turn, sample_partition
from .progress import Progress
from .release_cohort import write_once
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import HEAD_SEEDS, fit_head, predict_logits

ARMS = ("harper-only", "harper-plus-hindi", "harper-plus-shuffled-hindi")
BATCH = 8


def extract(predict, clips, name):
    meter = Progress(name, len(clips), "pauses")
    output = []
    try:
        for start in range(0, len(clips), BATCH):
            features = predict(clips[start:start + BATCH])
            if not np.isfinite(features).all():
                raise ValueError("nonfinite Whisper embeddings")
            output.append(features)
            meter.advance(len(features))
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return np.concatenate(output)


def supplement(rows):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    calls = np.asarray([row["recording"] for row in rows])
    counts = Counter(calls)
    weights = np.asarray([len(rows) / len(counts) / counts[call] for call in calls])
    mixed = sum(len(set(labels[calls == call])) == 2 for call in counts)
    if mixed < 2 or any(sum(labels == value) < 8 or len(set(calls[labels == value])) < 3 for value in (0, 1)):
        raise ValueError("too few eligible Hindi outcomes or mixed calls for a controlled pilot")
    null = labels.copy()
    for call in sorted(counts):
        indices = np.flatnonzero(calls == call)
        null[indices] = np.random.default_rng(seed_for(JOB, "shuffle-v1", call)).permutation(labels[indices])
    if np.array_equal(null, labels):
        raise ValueError("the prespecified null did not move the Hindi labels")
    return labels, null, weights, mixed


def fit(root, directory):
    membership, membership_sha = read_receipt(root, f"{JOB}-membership")
    if membership["source_events_sha256"] != SOURCE_SHA256 or membership["decision_ms"] != 300:
        raise ValueError("the Hindi membership is outside the registered pilot")
    labels, null, weights, mixed = supplement(membership["rows"])
    hindi = np.load(read_verified(root, membership["pcm"]), allow_pickle=False)
    if hindi.shape != (len(labels), 3 * RATE):
        raise ValueError("the admitted Hindi PCM has unexpected geometry")
    archive, archive_sha = read_receipt(root, "open-speech-whisper-large-v3")
    baseline = torch.load(read_verified(root, archive["diagnostic_checkpoint"]), map_location="cpu", weights_only=True)
    penalty = archive["heads"]["whisper-large-v3"]["penalty"]
    sets, english_labels, harper_sha, voices = training(root)
    waves = np.concatenate([sets[0], hindi])
    harper = np.arange(len(english_labels))
    both = np.arange(len(waves))
    report = {"scope": "training-only Hindi transfer feasibility; never gate or select on this development view",
              "membership_sha256": membership_sha, "teacher_report_sha256": archive_sha,
              "harper_fit_sha256": harper_sha, "harper_fit_speakers": voices, "penalty": penalty,
              "seeds": list(HEAD_SEEDS), "supplement_mass": len(labels) / len(waves),
              "hindi_count": len(labels), "mixed_calls": mixed, "shuffled_rows_changed": int((labels != null).sum()),
              "normalizer": "unmodified Harper fit only",
              "decision": "collect more labels only if Hindi BA gains >= 2 points over both controls, the two paired lower bounds are > 0, and English BA loses <= 1 point"}
    write_once(directory / "protocol.json", report)
    predict = encoder(root)
    features = extract(predict, waves, f"{JOB}-teacher-fit-features")
    heads = {arm: [] for arm in ARMS}
    archived_probabilities, refit_probabilities = [], []
    meter = Progress(f"{JOB}-head-fit", len(HEAD_SEEDS) * len(ARMS), "heads")
    try:
        for seed, archived in zip(HEAD_SEEDS, baseline["heads"], strict=True):
            control = fit_head(features, english_labels, harper, penalty, seed)
            archived_probabilities.append(expit(predict_logits(features[harper], archived)))
            refit_probabilities.append(expit(predict_logits(features[harper], control)))
            heads[ARMS[0]].append(control)
            meter.advance(1)
            for arm, targets in zip(ARMS[1:], (labels, null), strict=True):
                heads[arm].append(fit_head(features, np.concatenate([english_labels, targets]), both, penalty, seed,
                                          normalizer=harper, sample_weights=np.r_[np.ones(len(harper)), weights]))
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    drift = np.asarray(refit_probabilities) - np.asarray(archived_probabilities)
    replay = {"max_seed_probability_difference": float(np.max(np.abs(drift))),
              "max_ensemble_probability_difference": float(np.max(np.abs(drift.mean(axis=0)))),
              "ensemble_mean_absolute_difference": float(np.abs(drift.mean(axis=0)).mean()),
              "control": "same-machine refit, with the frozen historical teacher also reported"}
    write_once(directory / "baseline-replay.json", replay)
    print(json.dumps(replay), flush=True)
    path = directory / "teacher-heads.pt"
    torch.save(heads, path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)
    lock = {"heads": {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)},
            "protocol_sha256": file_digest(directory / "protocol.json"),
            "baseline_replay_sha256": file_digest(directory / "baseline-replay.json")}
    write_once(directory / "teacher-lock.json", lock)
    return heads, lock, predict, report


def score_view(language, loader, predict, heads, expected):
    labels, groups = [], []
    scores = {arm: [] for arm in ARMS}
    clips = []

    def flush():
        features = predict(np.stack(clips))
        for arm, fits in heads.items():
            values = np.stack([expit(predict_logits(features, head)) for head in fits]).mean(axis=0)
            scores[arm].extend(values.tolist())
        meter.advance(len(clips))
        clips.clear()

    meter = Progress(f"{JOB}-score-{language}", len(expected["labels"]), "pauses")
    try:
        for pcm, edge, label, group in loader:
            clips.append(decision_audio(pcm, edge, "telephone")[-3 * RATE:])
            labels.append(label)
            groups.append(group)
            if len(clips) == BATCH:
                flush()
        if clips:
            flush()
        if labels != expected["labels"] or groups != expected["turn_ordinals"]:
            raise ValueError("development membership or ordering changed")
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return {"labels": labels, "turn_ordinals": groups, "scores": {**expected["scores"], **scores}}


def hindi_interval(row):
    truth = np.asarray(row["labels"], dtype=bool)
    groups = np.asarray(row["turn_ordinals"])
    fit = calibration_split(groups, "hi")
    calibration, test = by_turn(groups[fit]), by_turn(groups[~fit])
    scores = [np.asarray(row["scores"][arm]) for arm in ARMS]
    rng = np.random.default_rng(30183)
    draws = []
    for _ in range(500):
        f = sample_partition(calibration, truth[fit], rng)
        e = sample_partition(test, truth[~fit], rng)
        ba = [evaluate(ps[~fit][e], truth[~fit][e], balanced_threshold(ps[fit][f], truth[fit][f]))
              ["balanced_accuracy"] for ps in scores]
        draws.append((ba[1] - ba[0], ba[1] - ba[2]))
    return np.quantile(draws, [.025, .975], axis=0).T.tolist()


def run(root):
    directory = root / "runs" / JOB
    analysis_contract = directory / "analysis-contract.json"
    write_once(analysis_contract, {
        "paired_contrasts": [f"{ARMS[1]} minus {arm}" for arm in (ARMS[0], ARMS[2])],
        "collection_signal": "Hindi BA gain of two points over both paired controls and the frozen archive, "
                             "positive paired lower bounds, and English BA within one point of matched refit",
        "archive": "report matched refit probability differences and archived-threshold decision disagreements",
        "language_subset": "six languages without conversational supervision before this pilot",
    })
    saved = directory / "teacher-lock.json"
    if saved.exists():
        lock = json.loads(saved.read_text())
        protocol = directory / "protocol.json"
        if (file_digest(protocol) != lock["protocol_sha256"]
                or file_digest(directory / "baseline-replay.json") != lock["baseline_replay_sha256"]):
            raise ValueError("the frozen Hindi teacher protocol changed")
        report = json.loads(protocol.read_text())
        heads = torch.load(read_verified(root, lock["heads"]), map_location="cpu", weights_only=True)
        predict = encoder(root)
    else:
        heads, lock, predict, report = fit(root, directory)
    reference, trial_sha, teacher_sha = frozen(root)
    references, reference_sha = read_receipt(root, "balanced-development-scorecard-references")
    if (references["trial_sha256"], references["owned_teacher_receipt_sha256"]) != (trial_sha, teacher_sha):
        raise ValueError("archived controls use different development inputs")
    harper = json.loads((root / "runs/owned-history-teacher/predictions-harper-88.json").read_text())["row"]
    reference["harper-88"] = {**harper, "scores": {}}
    loaders = {"hi": lambda: (row[:4] for row in multilingual(root, "hi")),
               "en": lambda: english_rows(root), "harper-88": lambda: harper_rows(root, native_prefix=True),
               **{lang: (lambda lang=lang: (row[:4] for row in multilingual(root, lang)))
                  for lang in reference if lang not in {"hi", "en", "harper-88"}}}
    rows, replay_drift = {}, {}
    for language, loader in loaders.items():
        expected = reference[language].copy()
        if language != "harper-88":
            benchmark = json.loads((root / "runs/balanced-development-scorecard"
                                    / f"reference-{language}.json").read_text())
            expected["scores"] = {**expected["scores"], **benchmark["scores"]}
        path = directory / f"teacher-predictions-{language}.json"
        if path.exists():
            saved_view = json.loads(path.read_text())
            if saved_view["heads_sha256"] != lock["heads"]["sha256"]:
                raise ValueError("cached scores came from a different locked head")
            row = saved_view["row"]
        else:
            row = score_view(language, loader(), predict, heads, expected)
            write_once(path, {"heads_sha256": lock["heads"]["sha256"], "row": row})
        if (row["labels"], row["turn_ordinals"]) != (expected["labels"], expected["turn_ordinals"]):
            raise ValueError("cached evaluation membership changed")
        rows[language] = row
        archived = harper["scores"]["h3"] if language == "harper-88" else expected["scores"]["whisper-large-v3"]
        error = np.abs(np.asarray(row["scores"][ARMS[0]]) - np.asarray(archived))
        replay_drift[language] = {"max_absolute_probability_difference": float(error.max()),
                                  "mean_absolute_probability_difference": float(error.mean())}
        print(json.dumps({"scored_language": language, "baseline_replay": replay_drift[language]}), flush=True)
    cells = {lang: cell(row, lang) for lang, row in rows.items() if lang != "harper-88"}
    harper_labels = np.asarray(rows["harper-88"]["labels"], dtype=bool)
    harper_metrics = references["harper-88"]
    harper_metrics["models"].update({
        arm: {"threshold": cells["en"]["models"][arm]["threshold"],
              **evaluate(np.asarray(values), harper_labels, cells["en"]["models"][arm]["threshold"])}
        for arm, values in rows["harper-88"]["scores"].items()
    })
    intervals = hindi_interval(rows["hi"])
    hindi_ba = [cells["hi"]["models"][arm]["balanced_accuracy"] for arm in ARMS]
    english_ba = [cells["en"]["models"][arm]["balanced_accuracy"] for arm in ARMS]
    for language, metrics in cells.items():
        scores = rows[language]["scores"]
        expected = metrics["models"]["whisper-large-v3"]
        actual = metrics["models"][ARMS[0]]
        replay_drift[language]["refit_minus_archived_balanced_accuracy"] = (
            actual["balanced_accuracy"] - expected["balanced_accuracy"]
        )
        cut = expected["threshold"]
        runs = ~calibration_split(rows[language]["turn_ordinals"], language)
        replay_drift[language]["evaluation_decisions_changed_at_archived_threshold"] = int(np.sum(
            (np.asarray(scores[ARMS[0]])[runs] >= cut) != (np.asarray(scores["whisper-large-v3"])[runs] >= cut)
        ))
    deltas = [hindi_ba[1] - hindi_ba[0], hindi_ba[1] - hindi_ba[2]]
    archived_hindi = cells["hi"]["models"]["whisper-large-v3"]["balanced_accuracy"]
    report = {**report, "head_lock": lock, "comparison_receipt_sha256": reference_sha,
              "baseline_replay": replay_drift,
              "threshold_source": "fixed development turn halves; English threshold used for HarperValley-88",
              "analysis_contract_sha256": file_digest(analysis_contract),
              "metrics": {"macro": macro(cells, sorted(cells)), "six-originally-unsupervised": macro(cells, UNSEEN_IN_FIT),
                          "languages": cells, "harper-88": harper_metrics},
              "hindi_ba_delta_order": [f"{ARMS[1]} minus {arm}" for arm in (ARMS[0], ARMS[2])],
              "hindi_ba_deltas": deltas, "hindi_ba_delta_95": intervals,
              "hindi_collection_signal": min(deltas) >= .02 and hindi_ba[1] - archived_hindi >= .02
              and min(interval[0] for interval in intervals) > 0 and english_ba[1] - english_ba[0] >= -.01,
              "pipeline": "unmeasured"}
    write_once(root / "reports" / f"{JOB}-teacher.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(2)
    run(data_root())
