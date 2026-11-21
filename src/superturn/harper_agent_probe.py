"""Measure whether new agent-side labels improve Superturn on caller audio."""

from collections import Counter, defaultdict
import hashlib
import json
import wave

import numpy as np
from scipy.special import expit
import torch
import torchaudio

from .ami_calibration import _write_json
from .dense_teacher import SAMPLES, file_digest
from .elapsed_turn_probe import paired_intervals
from .eot_domain_probe import GAP, decision_audio
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .sources import data_root
from .train import _stats
from .turn1_mini_probe import english_rows, harper_rows, upsample_prefix
from .wav2vec_harper_probe import (
    BUNDLE, CHECKPOINT_SHA256, calibration_rows, extract, fit, pause_rows, rates,
    speech_features, training_rows,
)

HEAD_SEEDS = (17, 29, 43, 53, 71)
SHUFFLE_SEEDS = tuple(range(3301, 3311))
AGENT_MEMBERSHIP_SHA256 = "ce369a9d59cbf4480f99a82f369cf54e1e7da74ad70779250aecf42ed9f5bb32"


def agent_clips(root):
    reports = root / "reports"
    path = reports / "harper-strict-agent-membership.json"
    membership = json.loads(path.read_text())
    if (file_digest(path) != AGENT_MEMBERSHIP_SHA256 or
            membership["yield_sha256"] != file_digest(reports / "harper-label-yield.json")):
        raise ValueError("agent membership differs from the original yield inventory")
    rows = membership["rows"]
    entries, _, _ = pinned_split(root, "train")
    agents = {entry["id"]: entry["agent"] for entry in entries}
    keys = {(call, edge) for call, _, _, edge, _ in rows}
    counts = Counter(label for _, _, _, _, label in rows)
    if counts != {"hold": 253, "shift": 293} or len(keys) != len(rows):
        raise ValueError("unexpected agent pause identities or outcomes")
    data = {}
    wavs = root / "raw/harper-valley/data/audio/agent"
    for call, role, person, edge, label in rows:
        if role != "agent" or agents.get(call) != person:
            raise ValueError("agent is not a verified training voice")
        if call not in data:
            with wave.open(str(wavs / f"{call}.wav"), "rb") as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (8000, 1, 2):
                    raise ValueError("unexpected native agent microphone format")
                data[call] = np.frombuffer(reader.readframes(reader.getnframes()), "<i2")
        observed = upsample_prefix(data[call], edge + GAP)
        yield decision_audio(observed, edge, "telephone")[-SAMPLES:], int(label == "shift"), (person, call)


def shuffle(values, groups, seed):
    result = values.copy()
    rng = np.random.default_rng(seed)
    for members in groups.values():
        result[members] = rng.permutation(values[members])
    return result


def weights_for(caller, agent):
    caller_rate = np.bincount(caller, minlength=2) / len(caller)
    agent_counts = np.bincount(agent, minlength=2)
    weights = np.r_[np.full(len(caller), .5 / len(caller)), .5 * caller_rate[agent] / agent_counts[agent]]
    return weights * len(weights)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this experiment uses only staged Vast audio")
    reports = root / "reports"
    previous_path = reports / "wav2vec-harper-probe.json"
    reference = json.loads(previous_path.read_text())
    penalty = reference["heads"]["pretrained"]["penalty"]
    fixed_threshold = reference["cohorts"]["english"]["pretrained"]["ami_threshold"]
    if penalty != .05 or reference["model_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("the selected reference head differs from the archived fit")
    cohorts = {"caller": list(training_rows(root)), "english": list(pause_rows(english_rows(root))),
               "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
               "calibration": list(calibration_rows(root)), "agent": list(agent_clips(root))}
    inputs, labels, indices = [], [], {}
    for name, rows in cohorts.items():
        audio, truth, _ = zip(*rows, strict=True)
        inputs.extend(audio)
        indices[name] = np.arange(len(labels), len(labels) + len(truth))
        labels.extend(truth)
    clips, truth = np.stack(inputs).astype(np.float32), np.asarray(labels)
    prefix = indices["agent"][0]
    if hashlib.sha256(clips[:prefix].tobytes()).hexdigest() != reference["pcm_sha256"]:
        raise ValueError("original caller and evaluation PCM did not reproduce")
    folder = root / "cache/wav2vec2-harper"
    checkpoint = folder / BUNDLE
    if file_digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("the general speech encoder checkpoint changed")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = torchaudio.models.wav2vec2_model(**bundle._params)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to("cuda").eval()
    prediction = lambda batch: speech_features(model, batch)
    baseline = extract("agent-control-orig", prediction, clips[:prefix])
    if hashlib.sha256(baseline.tobytes()).hexdigest() != reference["heads"]["pretrained"]["pooled_features_sha256"]:
        raise ValueError("original speech features changed")
    supplemental = extract("agent-control-agent", prediction, clips[prefix:])
    features = np.concatenate((baseline, supplemental))
    del clips, inputs, model
    caller, agent, english, harper = (indices[key] for key in ("caller", "agent", "english", "harper"))
    both = np.r_[caller, agent]
    evaluation = np.r_[english, harper]
    combos = defaultdict(list)
    for position, (_, _, key) in enumerate(cohorts["agent"]):
        combos[key].append(position)
    movable = [members for members in combos.values() if len(set(truth[agent[members]])) == 2]
    options = [("caller-only", caller, truth, None), ("strict-agent", both, truth, weights_for(truth[caller], truth[agent]))]
    changes = {}
    for seed in SHUFFLE_SEEDS:
        permuted = truth.copy()
        permuted[agent] = shuffle(truth[agent], combos, seed)
        name = f"shuffle-{seed}"
        changes[name] = int(np.sum(permuted[agent] != truth[agent]))
        options.append((name, both, permuted, weights_for(permuted[caller], permuted[agent])))
    results, scores = {}, {}
    meter = Progress("harper-agent-heads", len(options) * len(HEAD_SEEDS), "fits")
    try:
        for name, train, target, weight in options:
            seed_scores = []
            for seed in HEAD_SEEDS:
                value, _ = fit(features, target, train, evaluation, penalty, seed,
                               normalizer=caller, sample_weights=weight)
                seed_scores.append(expit(value))
                meter.advance(1)
            score = np.zeros(len(truth))
            score[evaluation] = np.mean(seed_scores, axis=0)
            scores[name] = score[english]
            results[name] = {key: {**rates(score[part], truth[part]),
                                   "fixed_threshold": _stats(score[part].tolist(), truth[part].tolist(), fixed_threshold)}
                             for key, part in (("english", english), ("harper", harper))}
        for name in ("english", "harper"):
            old = reference["cohorts"][name]["pretrained"]
            new = results["caller-only"][name]
            for item in ("auc", "partial_auc", "recalls_at_2_5_5_7_5pct"):
                if not np.allclose(old[item], new[item], rtol=0, atol=1e-10):
                    raise ValueError(f"caller-only {name} {item} did not reproduce")
            if old["ami_fixed"] != new["fixed_threshold"]:
                raise ValueError(f"caller-only {name} fixed-threshold results changed")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    null = [metrics["english"]["recalls_at_2_5_5_7_5pct"][1]
            for name, metrics in results.items() if name.startswith("shuffle-")]
    report = {"deadline_ms": 300, "fixed_ami_threshold": fixed_threshold,
              "agent_membership_sha256": AGENT_MEMBERSHIP_SHA256,
              "reference_report_sha256": file_digest(previous_path), "fit_penalty": penalty,
              "results": results, "null_mean": float(np.mean(null)), "null_p95": float(np.percentile(null, 95)),
              "agent_calls_with_both_classes": len(movable),
              "agent_examples_available_to_shuffle": sum(map(len, movable)),
              "shuffled_examples_changed": changes,
              "paired_english": paired_intervals(scores, truth[english], [row[2] for row in cohorts["english"]],
                                                  {"strict-agent": "caller-only"})}
    _write_json(reports / "harper-agent-probe.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    value = run(data_root())
    print(json.dumps({key: item for key, item in value.items() if key != "results"}, indent=2))
