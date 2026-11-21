"""Audit AMI next-speaker labels against human dialogue acts."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from statistics import mean
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

from .ami_calibration import _write_json
from .ami_external_baselines import load_detectors
from .ami_text_signal import CROSSVAL, caller_words_matrix, fit, read_words, row_features
from .audio import RATE
from .labels import _archive_member, ami_speakers, read_rows
from .progress import Progress
from .shards import seed_for
from .sources import data_root

DECISION_MS = 300
GAP = DECISION_MS * RATE // 1000
LINK = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?$")
INCOMPLETE = {"Fragment", "Stall"}


@dataclass(frozen=True)
class Act:
    identifier: str
    kind: str
    words: tuple


def entries(archive, filename):
    member = _archive_member(archive, filename)
    return list(ET.fromstring(archive.read(member)).iter()) if member else []


def identifier(element):
    return next((value for key, value in element.attrib.items() if key.rsplit("}", 1)[-1] == "id"), "")


def links(element):
    return [(child.get("role"), match[1], match[2] or match[1])
            for child in element if (match := LINK.search(child.get("href", "")))]


def read_track(archive, meeting, agent, offset, types):
    words, times = [], {}
    for entry in entries(archive, f"words/{meeting}.{agent}.words.xml"):
        word = identifier(entry)
        if not word:
            continue
        words.append(word)
        if entry.tag.rsplit("}", 1)[-1] == "w" and {"starttime", "endtime"} <= entry.attrib.keys():
            start, end = (round(float(entry.get(key)) * RATE) - offset for key in ("starttime", "endtime"))
            if start < end:
                times[word] = start, end
    positions = {word: i for i, word in enumerate(words)}
    by_word = defaultdict(list)
    for entry in entries(archive, f"dialogueActs/{meeting}.{agent}.dialog-act.xml"):
        if entry.tag.rsplit("}", 1)[-1] != "dact":
            continue
        pointers = links(entry)
        kind = next((types.get(first) for role, first, _ in pointers if role == "da-aspect"), None)
        if not kind:
            continue
        selected = tuple(word for role, first, last in pointers
                         if role is None and first in positions and last in positions
                         for word in words[positions[first]:positions[last] + 1] if word in times)
        act = Act(identifier(entry), kind, tuple(times[word] for word in selected))
        for word in selected:
            by_word[word].append(act)
    return times, by_word


def classify(row, tracks, pairs):
    edge = row["pause_start"]
    speaker = row["speaker"]
    timing, acts = tracks[speaker]
    matches = [word for word, (_, end) in timing.items() if end == edge]
    if len(matches) != 1 or len(acts[matches[0]]) != 1:
        return "uncovered", None
    act, = acts[matches[0]]
    done = max(end for _, end in act.words) == edge
    complete = done and act.kind not in INCOMPLETE | {"Backchannel"}
    if row["label"] == "hold":
        if not done or act.kind in INCOMPLETE:
            return "interior-or-fragment-hold", 0
        if complete:
            return "terminal-act-hold", 1
        return "other-hold", None
    following = [(start, voice, word) for voice, (times, _) in tracks.items()
                 for word, (start, _) in times.items() if start >= edge + GAP]
    onset = min((start for start, _, _ in following), default=None)
    first = [(voice, word) for start, voice, word in following if start == onset]
    response = False
    if len(first) == 1 and first[0][0] != speaker and not row["annotated_backchannel"]:
        (voice, word), = first
        response = any((act.identifier, next_act.identifier) in pairs for next_act in tracks[voice][1][word])
    if complete and (response or act.kind.startswith("Elicit")):
        return "response-or-elicit-shift", 1
    return "other-shift", 1 if complete else None


def endpoints(row, intervals):
    start = row["pause_start"]
    ends = [item.pause_start for item in intervals
            if min(start + GAP, item.pause_start + item.pause_samples) > max(start, item.pause_start)]
    if len(ends) != 1:
        return "unmatched"
    delta = abs(ends[0] - start) / 16
    return "<=50ms" if delta <= 50 else "50-120ms" if delta <= 120 else ">120ms"


def mean_auc(folds, field, selected, target="truth"):
    from .train import _auc

    values = []
    for fold in folds:
        rows = [row for row in fold if selected(row) and row[target] is not None]
        auc = _auc([row[field] for row in rows], [row[target] for row in rows])
        if auc is not None:
            values.append(auc)
    return {"auc": mean(values) if values else None, "folds": len(values)}


def contrast(folds, models, selected, target="truth"):
    return {
        "counts": dict(Counter(row[target] for fold in folds for row in fold
                               if selected(row) and row[target] is not None)),
        "models": {model: mean_auc(folds, model, selected, target) for model in models},
    }


def score_gru(root, entries, fold, audio):
    import torch

    from .model import build

    outputs = []
    for entry in entries:
        path = root / "exports" / "ami-crossval" / CROSSVAL / f'cv-{fold}-gru-6s-{entry["seed"]}' / "best.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["checkpoint_sha256"]:
            raise ValueError(f"reference GRU checkpoint changed in fold {fold}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build(saved["config"]).eval()
        pcm = torch.from_numpy(audio[:, -model.samples:])
        with torch.inference_mode():
            model.load_state_dict(saved["state"])
            outputs.append(model(pcm, torch.full((len(pcm), 1), float(DECISION_MS))).flatten().numpy())
    return np.mean(outputs, axis=0)


def summarize(folds, models):
    inside = "interior-or-fragment-hold"
    terminal = "terminal-act-hold"
    response = "response-or-elicit-shift"
    primary = lambda row: row["stratum"] in {inside, response}
    result = {
        "all": contrast(folds, models, lambda row: True),
        "clean_contrast": contrast(folds, models, primary),
        "terminal_hold_vs_response_shift": contrast(folds, models, lambda row: row["stratum"] in {terminal, response}),
        "interior_hold_vs_other_shift": contrast(folds, models,
                                                lambda row: row["stratum"] in {inside, "other-shift"}),
        "dialogue_act_proxy": contrast(folds, models, lambda row: True, target="proxy"),
        "endpoint_agreement": {kind: contrast(folds, models, lambda row: row["endpoint"] == kind)
                               for kind in ("<=50ms", "50-120ms", ">120ms", "unmatched")},
        "strata": dict(Counter(row["stratum"] for fold in folds for row in fold)),
        "backchannels_by_stratum": dict(Counter(row["stratum"] for fold in folds for row in fold if row["backchannel"])),
    }
    null = []
    rng = np.random.default_rng(seed_for("ami-dialogue-audit-null-v1"))
    for _ in range(24):
        shuffled = []
        for fold in folds:
            blocks = defaultdict(list)
            for row in fold:
                blocks[row["group"], row["truth"]].append(row)
            shuffled.append([dict(row, stratum=kind) for block in blocks.values()
                             for row, kind in zip(block, rng.permutation([row["stratum"] for row in block]))])
        value = mean_auc(shuffled, "gru-6s", primary)["auc"]
        if value is not None:
            null.append(value)
    result["clean_contrast_gru_null"] = {
        "replicates": len(null),
        "p95": float(np.quantile(null, 0.95)) if null else None,
    }
    return result


def run(root):
    import torch

    from .train import _input, paths, records

    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI dialogue audit uses the training server")
    torch.set_num_threads(1)
    folds = root / "shards" / "ami-crossval" / CROSSVAL
    report_dir = root / "reports"
    reference = json.loads((report_dir / "ami-crossval" / f"{CROSSVAL}.json").read_text())
    inputs = []
    for index in range(reference["folds"]):
        folder = folds / f"fold-{index}"
        inputs.append([[row for batch in records(paths(folder, split)) for row in batch]
                       for split in ("train", "out-of-fold")])
    meetings = {row["recording"] for fold in inputs for row in fold[1]}
    other_meetings = {row["recording"] for fold in inputs for row in fold[0]}
    raw = root / "raw"
    manifest = raw / "ami" / "MANIFEST.json"
    annotation, = (raw / "ami_annotations").glob("*.zip")
    offsets = {track["path"].split("/")[0]: round(track["start_s"] * RATE)
               for track in json.loads(manifest.read_text())["files"]}
    tracks, pair_links = {}, {}
    with ZipFile(annotation) as archive:
        roster = ami_speakers(archive)
        types = {identifier(entry): entry.get("gloss") for entry in entries(archive, "ontologies/da-types.xml")}
        if not types:
            raise ValueError("AMI dialogue act ontology missing")
        for meeting in meetings:
            tracks[meeting] = {speaker: read_track(archive, meeting, agent, offsets[meeting], types)
                               for speaker, agent in roster[meeting].values()}
            pairs = entries(archive, f"dialogueActs/{meeting}.adjacency-pairs.xml")
            pointers = [{role: first for role, first, _ in links(entry)}
                        for entry in pairs if entry.tag.rsplit("}", 1)[-1] == "adjacency-pair"]
            pair_links[meeting] = {(pair["source"], pair["target"]) for pair in pointers if {"source", "target"} <= pair.keys()}
    labels = root / "labels" / "ami.jsonl"
    intervals = defaultdict(list)
    for row in read_rows(labels):
        if row.recording in meetings and row.split in {"train", "validation"}:
            intervals[row.recording, row.speaker].append(row)
    words = read_words(root, meetings | other_meetings)
    predict, _, smart_hash = load_detectors(root)
    results = []
    meter = Progress("ami-dialogue-audit", len(inputs), "folds")
    try:
        for fold, (training, evaluation) in enumerate(inputs):
            train_features = [row_features(row, words) for row in training]
            eval_features = [row_features(row, words) for row in evaluation]
            vocabulary = [word for word, _ in Counter(row[2] for row in train_features if row[2]).most_common(30)]
            text_scores, = fit(caller_words_matrix(train_features, vocabulary),
                               np.array([row["label"] == "shift" for row in training]),
                               (caller_words_matrix(eval_features, vocabulary),))
            audio = np.stack([_input(row, 8 * RATE, DECISION_MS) for row in evaluation]).astype(np.float32) / 32768
            references = [entry for entry in reference["runs"] if entry["fold"] == fold and entry["variant"] == "gru-6s"]
            gru = score_gru(root, references, fold, audio)
            scored = []
            for i, row in enumerate(evaluation):
                meeting, speaker = row["recording"], row["speaker"]
                stratum, proxy = classify(row, tracks[meeting], pair_links[meeting])
                scores = predict(row, DECISION_MS, audio[i])
                scored.append({**scores, "gru-6s": gru[i], "caller-words": text_scores[i],
                               "stratum": stratum, "proxy": proxy, "truth": int(row["label"] == "shift"),
                               "group": row["group"], "backchannel": row["annotated_backchannel"],
                               "endpoint": endpoints(row, intervals[meeting, speaker])})
            results.append(scored)
            meter.advance(1)
        destination = report_dir / "ami-dialogue-audit.json"
        _write_json(destination, {
            "crossval": CROSSVAL, "rows": sum(map(len, results)), "smart_turn_sha256": smart_hash,
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (manifest, annotation, labels)},
            **summarize(results, ("gru-6s", "caller-words", "livekit-v1.1-mini", "smart-turn-v3.2")),
        })
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    print(run(data_root()))
