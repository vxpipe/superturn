"""Audit the frozen Hindi pilot annotations using training-side audio only."""

from collections import Counter
from fractions import Fraction
from itertools import combinations
import json

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch

from .audio import silero
from .hindi_review_pilot import EDGE_MARGIN, JOB
from .labels import Speech, turns
from .language_failure_audit import read_receipt, read_verified
from .owned_history_teacher import encoder
from .release_cohort import write_once
from .sources import data_root
from .spring_hindi_clock import load_tracks, read_inputs
from .spring_hindi_pilot import intervals
from .wav2vec_harper_probe import predict_logits


def combine(left, right):
    result = Counter()
    for a, size_a in left.items():
        for b, size_b in right.items():
            result[a + b] += size_a * size_b
    return result


def within_group(scores, labels):
    shift = int(labels.sum())
    pairs = shift * (len(labels) - shift)
    if pairs == 0:
        return
    ranks = np.rint(2 * rankdata(scores, method="average")).astype(int)

    def auc(indices):
        return Fraction(int(ranks[list(indices)].sum()) - shift * (shift + 1), 2 * pairs)

    observed = auc(np.flatnonzero(labels))
    null = Counter(auc(indices) for indices in combinations(range(len(labels)), shift))
    return pairs, observed, null


def ranking(rows, scores, *, condition_on_side):
    calls = sorted({row["recording"] for row in rows})
    summary = []
    observed = Fraction(0)
    null = Counter({Fraction(0): 1})
    for ordinal, call in enumerate(calls, 1):
        sides = (0, 1) if condition_on_side else (None,)
        results = []
        for side in sides:
            idx = [i for i, row in enumerate(rows) if row["recording"] == call
                   and (side is None or row["channel"] == side)]
            value = within_group(scores[idx], np.asarray([rows[i]["label"] for i in idx], dtype=bool))
            if value is not None:
                results.append(value)
        if not results:
            continue
        pairs = sum(value[0] for value in results)
        pair_weighted = sum(value[0] * value[1] for value in results) / pairs
        intra_call = Counter({Fraction(0): 1})
        for count, _, possible in results:
            intra_call = combine(intra_call, Counter({auc * count / pairs: n for auc, n in possible.items()}))
        null = combine(null, intra_call)
        observed += pair_weighted
        summary.append({"call_ordinal": ordinal, "comparable_pairs": pairs, "strata": len(results),
                        "auc": float(pair_weighted)})
    allocations = sum(null.values())
    return {"calls": summary, "allocations": allocations, "support": len(null),
            "equal_call_auc": float(observed / len(summary)) if summary else None,
            "upper_tail": sum(n for value, n in null.items() if value >= observed) / allocations if summary else None}


def acoustic_rule(root, rows):
    plan, audio, source, train, audio_sha = read_inputs(root)
    dev, _ = intervals(source, "dev")
    evaluation, _ = intervals(source, "eval")
    excluded = dev.keys() | evaluation.keys()
    candidate_calls = sum(call not in excluded and {span.speaker for span in spans} == {"Left", "Right"}
                          for call, spans in train.items())
    detect = silero()
    by_class = {name: Counter() for name in ("hold", "shift")}
    supported = {name: set() for name in by_class}
    for tile in plan["selection"]:
        call, start = tile["recording"], tile["start_native_sample"]
        tracks = load_tracks(source, audio, tile)
        spans = [Speech(str(side), lo, hi) for side, track in enumerate(tracks) for lo, hi in detect(track)]
        events = list(turns(spans, decision_ms=300))
        for row in (row for row in rows if row["recording"] == call):
            label = ("hold", "shift")[row["label"]]
            edge = row["stop_sample"] - start
            matches = [result for span, _, result in events
                       if span.speaker == str(row["channel"]) and abs(span.end - edge) <= EDGE_MARGIN]
            key = "unmatched" if not matches else "ambiguous" if len(matches) != 1 else matches[0]
            by_class[label][key] += 1
            if matches == [label]:
                supported[label].add(call)
    return {"audio_sha256": audio_sha, "prediction_by_review_label": by_class,
            "correct_calls_by_class": {label: len(calls) for label, calls in supported.items()},
            "publisher_train_calls": len(train), "train_exclusive_two_side_calls": candidate_calls,
            "scope": "cue-exposed reviewer, source training subset; label-rule feasibility only"}


def run(root):
    membership, membership_sha = read_receipt(root, f"{JOB}-membership")
    teacher, teacher_sha = read_receipt(root, "open-speech-whisper-large-v3")
    rows = membership["rows"]
    pcm = np.load(read_verified(root, membership["pcm"]), allow_pickle=False)
    archived = torch.load(read_verified(root, teacher["diagnostic_checkpoint"]),
                          map_location="cpu", weights_only=True)
    features = encoder(root)(pcm)
    values = np.stack([expit(predict_logits(features, head)) for head in archived["heads"]]).mean(axis=0)
    report = {"membership_sha256": membership_sha, "frozen_teacher_sha256": teacher_sha,
              "primary_call_and_channel": ranking(rows, values, condition_on_side=True),
              "secondary_call_only": ranking(rows, values, condition_on_side=False),
              "acoustic_rule": acoustic_rule(root, rows)}
    write_once(root / "reports/hindi-pilot-source-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    run(data_root())
