"""Probe past-only lexical signal on the existing AMI development folds."""

from collections import Counter, defaultdict
import json
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

from .ami_calibration import _write_json
from .ami_crossval import paired_deltas, summarize_runs
from .ami_transfer import partial_auc
from .audio import RATE
from .labels import _archive_member, ami_speakers
from .progress import Progress
from .shards import seed_for
from .sources import data_root

CROSSVAL = "6dbe4ae7e8bdfd90"
CLASSES = (
    frozenset("and or but so because if though while when unless although yet".split()),
    frozenset("um uh er erm hmm ah uhm".split()),
    frozenset("of to on in with for from about by at over under into".split()),
    frozenset("a an the my your his her our their this that these those".split()),
    frozenset("who what where why when which how".split()),
    frozenset("yes no okay ok right yeah thanks thank sure alright".split()),
)


def preceding(words, edge):
    """Use words ending within the six seconds before the query boundary."""
    return [word for word in words if edge - 6 * RATE < word[1] <= edge]


def row_features(row, terms):
    edge = row["pause_start"]
    own = preceding(terms[row["recording"], row["speaker"]], edge)
    other = sorted((word for (meeting, speaker), words in terms.items()
                    if meeting == row["recording"] and speaker != row["speaker"]
                    for word in preceding(words, edge)), key=lambda word: word[1])
    lexical = [re.sub(r"[^a-z']", "", word[2].lower()) for word in own]
    final = lexical[-1] if lexical else ""
    elapsed = (edge - own[-1][1]) / RATE if own else 6.0
    timing = [len(own), sum(word[1] > edge - 2 * RATE for word in own), elapsed]
    words = [float(final in category) for category in CLASSES]
    words.append(float(len(lexical) >= 2 and lexical[-2] == final and bool(final)))
    punctuation = [float(own[-1][2].endswith(mark)) if own else 0.0 for mark in (".", "?", "!", ",")]
    other_last = re.sub(r"[^a-z']", "", other[-1][2].lower()) if other else ""
    other_elapsed = (edge - other[-1][1]) / RATE if other else 6.0
    interaction = [len(other), sum(word[1] > edge - 2 * RATE for word in other), other_elapsed]
    interaction += [float(other_last in category) for category in CLASSES]
    return timing, words, final, punctuation, interaction


def read_words(root, meeting_ids):
    raw = root / "raw"
    windows = {entry["path"].split("/")[0]: round(entry["start_s"] * RATE)
               for entry in json.loads((raw / "ami" / "MANIFEST.json").read_text())["files"]}
    archive_path, = (raw / "ami_annotations").glob("*.zip")
    result = defaultdict(list)
    with ZipFile(archive_path) as archive:
        roster = ami_speakers(archive)
        for meeting in sorted(meeting_ids):
            for speaker, agent in roster[meeting].values():
                member = _archive_member(archive, f"words/{meeting}.{agent}.words.xml")
                if not member:
                    continue
                for word in ET.fromstring(archive.read(member)):
                    if (word.tag.rsplit("}", 1)[-1] == "w"
                            and {"starttime", "endtime"} <= word.attrib.keys()
                            and (text := (word.text or "").strip())):
                        start = round(float(word.attrib["starttime"]) * RATE) - windows[meeting]
                        end = round(float(word.attrib["endtime"]) * RATE) - windows[meeting]
                        if end > max(start, 0):
                            result[meeting, speaker].append((start, end, text))
                result[meeting, speaker].sort(key=lambda word: (word[1], word[0]))
    return result


def fit(train, truth, other):
    offset = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-8)
    x = np.column_stack(((train - offset) / scale, np.ones(len(train))))
    weights = np.zeros(x.shape[1])
    for _ in range(250):
        prediction = 1 / (1 + np.exp(np.clip(-(x @ weights), -50, 50)))
        penalty = weights.copy()
        penalty[-1] = 0
        weights -= .1 * (x.T @ (prediction - truth) / len(truth) + .05 * penalty)
    return [np.column_stack(((data - offset) / scale, np.ones(len(data)))) @ weights for data in other]


def caller_words_matrix(features, vocabulary):
    return np.array([timing + categories + [float(final == key) for key in vocabulary]
                     for timing, categories, final, _, _ in features])


def run(root):
    from .train import _stats, _threshold, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI text diagnostic uses the training server")
    report = root / "reports" / "ami-crossval" / f"{CROSSVAL}.json"
    metadata = json.loads(report.read_text())
    workspace = root / "shards" / "ami-crossval" / CROSSVAL
    columns = ["recording", "speaker", "group", "pause_start", "label"]
    data = []
    for i in range(metadata["folds"]):
        fold = workspace / f"fold-{i}"
        data.append([[row for batch in records(paths(fold, split), columns=columns) for row in batch]
                     for split in ("train", "validation", "out-of-fold")])
    words = read_words(root, {row["recording"] for fold in data for split in fold for row in split})
    outcomes = defaultdict(list)
    null = defaultdict(list)
    predictions, fold_meta = {}, {}
    punctuation_rows = 0
    meter = Progress("ami-text-signal", len(data), "folds")
    try:
        for i, splits in enumerate(data):
            features = [[row_features(row, words) for row in cases] for cases in splits]
            common = [word for word, _ in Counter(row[2] for row in features[0] if row[2]).most_common(30)]
            matrices = defaultdict(list)
            for split in features:
                matrices["caller-timing"].append(np.array([row[0] for row in split]))
                lexical = caller_words_matrix(split, common)
                matrices["caller-words"].append(lexical)
                matrices["caller-punctuation"].append(np.column_stack((lexical, [row[3] for row in split])))
                matrices["caller-plus-other-timing"].append(np.column_stack((lexical, [row[4][:3] for row in split])))
                matrices["all-speaker-past"].append(np.column_stack((lexical, [row[4] for row in split])))
            punctuation_rows += sum(any(row[3]) for row in features[2])
            y, cy, ty = [np.array([int(row["label"] == "shift") for row in cases]) for cases in splits]

            fold_meta[i] = (ty.tolist(), [row["group"] for row in splits[2]])

            def score(arm, labels, *, keep=False):
                train, cal, evaluation = matrices[arm]
                calibration, decisions = (v.tolist() for v in fit(train, labels, (cal, evaluation)))
                threshold = _threshold(calibration, cy.tolist())
                if keep:
                    predictions[i, arm, 0] = decisions
                return {"fold": i, "evaluation": {**_stats(decisions, ty.tolist(), threshold),
                                                    "partial_auc": partial_auc(decisions, ty.tolist())}}

            for arm in matrices:
                outcomes[arm].append(score(arm, y, keep=True))
            groups = defaultdict(list)
            for index, row in enumerate(splits[0]):
                groups[row["group"]].append(index)
            for seed in range(24):
                rng = np.random.default_rng(seed_for("ami-text-null-v1", i, seed))
                labels = y.copy()
                for indices in groups.values():
                    labels[indices] = y[rng.permutation(indices)]
                null[seed].append(score("caller-words", labels))
            meter.advance(1)
        destination = root / "reports" / "ami-text-signal.json"
        _write_json(destination, {
            "crossval": CROSSVAL, "summary": {arm: summarize_runs(rows) for arm, rows in outcomes.items()},
            "null_summary": [summarize_runs(rows) for rows in null.values()], "fold_results": outcomes,
            "paired_fold_bootstrap_vs_caller_timing": paired_deltas(
                predictions, fold_meta, variants=outcomes, baseline="caller-timing", seeds=(0,)),
            "reference_gru_6s": metadata["summary"]["gru-6s"],
            "punctuated_out_of_fold_rows": punctuation_rows,
            "scope": "same AMI development folds; aligned transcript words ending before pause only; no text exported",
        })
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    print(run(data_root()))
