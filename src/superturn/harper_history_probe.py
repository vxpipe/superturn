"""Replicate the frozen Turn-1-Mini history comparison on HarperValley train."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import wave

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .augment import condition
from .dense_teacher import SAMPLES, normalize
from .harper_valley_audit import audit_call
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .shards import TAIL
from .sources import data_root
from .train import _threshold
from .turn_metrics import BUDGETS, recall_area
from .turn1_mini_probe import (
    ARTIFACT, CHECKPOINT, FRAME, GAP, LICENSE, REVISION, SIZE, STEP, Turn1,
    cache_artifact, upsample_prefix,
)

FOLDS = 5


def read_calls(root, entries, expected):
    detect = silero()
    raw = root / "raw/harper-valley"
    counts, calls = Counter(), []
    meter = Progress("harper-history-pauses", len(entries), "calls")
    try:
        for case in entries:
            for asset in case["files"]:
                if (raw / asset["path"]).stat().st_size != asset["bytes"]:
                    raise ValueError("staged HarperValley asset size changed")
            pauses = []
            identifier = case["id"]
            counts.update(audit_call(raw, identifier, detect, recover_clock=True, pauses=pauses))
            if pauses:
                with wave.open(str(raw / "data/audio/caller" / f"{identifier}.wav"), "rb") as reader:
                    if (reader.getframerate(), reader.getsampwidth(), reader.getnchannels()) != (8000, 2, 1):
                        raise ValueError("unexpected native HarperValley format")
                    native = np.frombuffer(reader.readframes(reader.getnframes()), "<i2")
                calls.append((case, native, pauses))
            meter.advance(1)
        if counts != expected["counts"]:
            raise ValueError("training pauses differ from the frozen audit")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return calls


def streams(native, edge, phase):
    stop = (edge + GAP - phase) // FRAME * FRAME + phase
    observed = upsample_prefix(native, stop)
    tail = TAIL - (stop - edge)
    full = normalize(condition(np.pad(observed, (0, tail)), "telephone", 0)[phase:stop])
    reset = full[-SAMPLES:]
    if len(full) % FRAME or (len(full) - len(reset)) % FRAME:
        raise ValueError("the two streams do not share a frame endpoint")
    return reset, full, edge + GAP - stop, stop


def predictions(scores, truth, folds, indices):
    decisions = np.zeros((len(indices), len(BUDGETS), 2), dtype=bool)
    thresholds = {}
    for fold in range(FOLDS):
        held = folds[indices] == fold
        fit = indices[~held]
        if not np.any(truth[fit] == 0):
            raise ValueError("fold needs calibration holds")
        grid = np.array([[_threshold(scores[fit, arm], truth[fit], budget)
                          for arm in range(2)] for budget in BUDGETS])
        decisions[held] = scores[indices[held], None, :] >= grid
        thresholds[fold] = grid.tolist()
    return decisions, thresholds


def rates(decisions, truth):
    return decisions[truth == 1].mean(axis=0), decisions[truth == 0].mean(axis=0)


def evaluate(scores, truth, callers, agents):
    voices, caller = np.unique(callers, return_inverse=True)
    partners, agent = np.unique(agents, return_inverse=True)
    weights = np.bincount(caller)
    load = np.zeros(FOLDS, dtype=int)
    assignments = np.zeros(len(voices), dtype=int)
    for index in np.argsort(-weights, kind="stable"):
        fold = np.argmin(load)
        assignments[index] = fold
        load[fold] += weights[index]
    folds = assignments[caller]
    indices = np.arange(len(truth))
    fitted, thresholds = predictions(scores, truth, folds, indices)
    recall, false = rates(fitted, truth)
    roc = np.column_stack([recall_area(arm, truth) for arm in scores.T])
    sensitivity = {}
    for two_way in (False, True):
        rng = np.random.default_rng(94217)
        samples = []
        for _ in range(500):
            weights = np.bincount(rng.integers(len(voices), size=len(voices)), minlength=len(voices))[caller]
            if two_way:
                weights *= np.bincount(rng.integers(len(partners), size=len(partners)), minlength=len(partners))[agent]
            picked = np.repeat(indices, weights)
            labels = truth[picked]
            if len(np.unique(labels)) != 2 or any(not np.any((folds[picked] != fold) & (labels == 0))
                                                  for fold in range(FOLDS)):
                continue
            decisions, _ = predictions(scores, truth, folds, picked)
            r, f = rates(decisions, labels)
            a, b = [recall_area(arm[picked], labels) for arm in scores.T]
            samples.append([np.diff(r, axis=1).mean(), r[1, 1] - r[1, 0],
                            f[1, 1] - f[1, 0], b[1] - a[1], b[3] - a[3]])
        sensitivity["caller-agent" if two_way else "caller"] = {
            "replicates": len(samples),
            "delta_95_intervals": dict(zip(("mean_recall", "recall_at_5", "false_at_5",
                                             "empirical_recall_at_5", "normalized_partial_auc"),
                                            np.quantile(samples, [.025, .975], axis=0).T.tolist(), strict=True)),
        }
    decision = fitted[:, 1, :]
    detail = []
    for index, voice in enumerate(voices):
        selected = caller == index
        row = {"caller": int(voice), "fold": int(assignments[index])}
        for label, name in ((0, "false_cutoff"), (1, "true_shift")):
            marked = decision[selected & (truth == label)]
            row[name] = {"total": len(marked), "hits": marked.sum(axis=0).tolist(),
                         "reset_only": int(np.sum(marked[:, 0] & ~marked[:, 1])),
                         "full_only": int(np.sum(marked[:, 1] & ~marked[:, 0]))}
        detail.append(row)
    return {"thresholds": thresholds, "recall": recall.tolist(), "false_cutoffs": false.tolist(),
            "mean_recall": recall.mean(axis=0).tolist(), "empirical_recall": roc[:3].tolist(),
            "normalized_partial_auc_0_to_10pct": roc[3].tolist(), "bootstrap": sensitivity,
            "by_caller_at_5pct": detail, "agents": len(partners)}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this inference uses only staged Vast audio")
    entries, expected, audit = pinned_split(root, "train")
    calls = read_calls(root, entries, expected)
    folder = root / "cache/eot-competitors/turn1-mini"
    cache_artifact(folder, "LICENSE", 11357, LICENSE)
    model = Turn1(cache_artifact(folder, ARTIFACT, SIZE, CHECKPOINT))
    rows = [(case, native, edge, label) for case, native, pauses in calls for edge, label in pauses]
    truth = np.asarray([label == "shift" for _, _, _, label in rows], dtype=int)
    if (len(truth) - truth.sum(), truth.sum()) != (273, 255):
        raise ValueError("unexpected HarperValley train cohort")
    rng = np.random.default_rng(94217)
    effects = []
    for index in rng.permutation(len(rows)):
        _, native, edge, _ = rows[index]
        reset, full, _, stop = streams(native, edge, 0)
        following = native[stop // 2:(stop + STEP) // 2]
        future = normalize(upsample_prefix(following, 2 * len(following))) if len(following) else full[:0]
        effects.extend(effect for pcm in (reset, full) if (effect := model.audit(pcm, future, rng)) is not None)
        if len(effects) >= 40:
            break
    if len(effects) < 40 or max(effects) > 1e-6:
        raise ValueError("partial chunks did not pass the causality check")

    def score(row):
        _, native, edge, _ = row
        result = []
        for phase in (0, FRAME // 2):
            reset, full, lag, _ = streams(native, edge, phase)
            values = [model.score(pcm, True) for pcm in (reset, full)]
            if any(age for _, _, age in values):
                raise ValueError("a mode failed to emit the aligned deadline frame")
            result.append(([instant for instant, _, _ in values], lag))
        return result

    collected = []
    meter = Progress("harper-history-scores", len(rows), "pauses")
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for batch in range(0, len(rows), 16):
                results = list(pool.map(score, rows[batch:batch + 16]))
                meter.advance(len(results))
                collected.extend(results)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    membership = [(case["id"], edge, label) for case, _, edge, label in rows]
    report = {"split": "train", "audit_sha256": audit, "holds": 273, "shifts": 255,
              "model_revision": REVISION, "model_sha256": CHECKPOINT,
              "membership_sha256": hashlib.sha256(json.dumps(membership).encode()).hexdigest(),
              "modes": ["reset3", "full"], "target_false_cutoffs": BUDGETS.tolist(),
              "future_checks": len(effects), "max_future_effect": max(effects), "phase_results": {}}
    for index, phase in enumerate((0, 10)):
        scores, lag = zip(*(result[index] for result in collected), strict=True)
        result = evaluate(np.asarray(scores), truth, [row[0]["caller"] for row in rows],
                          [row[0]["agent"] for row in rows])
        report["phase_results"][phase] = {**result, "lag_ms": [min(lag) * 1000 / RATE, max(lag) * 1000 / RATE]}
    _write_json(root / "reports/harper-history-replication-membership.json", membership)
    _write_json(root / "reports/harper-history-replication.json", report)
    return report


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    result = run(data_root())
    print(json.dumps({key: {name: item for name, item in value.items() if name not in ("by_caller_at_5pct", "thresholds")}
                      for key, value in result["phase_results"].items()}, indent=2))
