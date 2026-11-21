"""Test whether annotated AMI training pauses improve the frozen caller head."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json

import numpy as np
from scipy.special import expit
import torch
import torchaudio

from .ami_calibration import _write_json
from .ami_dialogue_audit import classify, dialogue_tracks
from .dense_teacher import file_digest
from .elapsed_turn_probe import paired_intervals
from .harper_agent_probe import HEAD_SEEDS, shuffle
from .progress import Progress
from .sources import data_root
from .train import _stats, _threshold
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import (
    AMI, BUNDLE, CHECKPOINT_SHA256, extract, fit_head, head_logits,
    pause_rows, rates, render_ami, speech_features, training_rows, verified_ami_labels,
)

PENALTY = .05
AMI_WEIGHT = .5
SHUFFLE_SEEDS = tuple(range(4801, 4821))
FILTERED = {"interior-or-fragment-hold", "response-or-elicit-shift"}


def ami_cohorts(root):
    training = verified_ami_labels(root, "train")
    calibration = verified_ami_labels(root, "calibration")
    for field in ("group", "speaker", "recording"):
        if {getattr(row, field) for row in training} & {getattr(row, field) for row in calibration}:
            raise ValueError(f"AMI fit and calibration {field} overlap")
    annotation, = (root / "raw/ami_annotations").glob("*.zip")
    manifest = root / "raw/ami/MANIFEST.json"
    archived = json.loads((root / "reports/ami-calibration" / f"{AMI}.json").read_text())
    if any(file_digest(path) != archived["inputs"][name]
           for name, path in (("annotations", annotation), ("raw_manifest", manifest))):
        raise ValueError("the AMI annotation archive or offsets changed")
    tracks, pairs = dialogue_tracks(manifest, annotation, sorted({row.recording for row in training}))
    labels = [classify(asdict(row), tracks[row.recording], pairs[row.recording])[0] for row in training]
    filtered = np.flatnonzero(np.isin(labels, sorted(FILTERED)))
    class_counts = Counter(training[i].label for i in filtered)
    if Counter(row.label for row in training) != {"hold": 253, "shift": 237} or class_counts != {"hold": 72, "shift": 69}:
        raise ValueError("verified AMI counts differ from the preregistered preflight")
    return training, calibration, filtered, Counter(labels)


def mixture_weights(caller, ami):
    caller_rate = np.bincount(caller, minlength=2) / len(caller)
    ami_count = np.bincount(ami, minlength=2)
    weight = np.r_[np.full(len(caller), 1 / len(caller)), AMI_WEIGHT * caller_rate[ami] / ami_count[ami]]
    return weight * len(weight) / (1 + AMI_WEIGHT)


def fit_scores(features, truth, fit_rows, evaluation, caller, seed, weight=None):
    head = fit_head(features, truth, fit_rows, PENALTY, seed, normalizer=caller, sample_weights=weight)
    return predict_head(features[evaluation], head), head


def predict_head(features, head):
    center, scale, *params = head
    return expit(head_logits((torch.as_tensor(features).double() - center) / scale, *params).numpy())


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this experiment uses only staged Vast audio")
    reports = root / "reports"
    reference_path = reports / "wav2vec-harper-probe.json"
    reference = json.loads(reference_path.read_text())
    anchor = reference["heads"]["pretrained"]
    if anchor["penalty"] != PENALTY or reference["model_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("unexpected archived frozen head")
    ami_labels, cal_labels, filtered, strata = ami_cohorts(root)
    cohorts = {
        "caller": list(training_rows(root)), "english": list(pause_rows(english_rows(root))),
        "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
        "calibration": list(render_ami(root, [row for row in cal_labels if row.label == "hold"])),
        "ami": list(render_ami(root, ami_labels)),
    }
    clips = np.stack([clip for rows in cohorts.values() for clip, _, _ in rows]).astype(np.float32)
    truth = np.asarray([label for rows in cohorts.values() for _, label, _ in rows])
    parts, start = {}, 0
    for name, rows in cohorts.items():
        parts[name] = np.arange(start, start + len(rows))
        start += len(rows)
    caller, english, harper, cal, ami = (parts[key] for key in cohorts)
    if len(cal) != 51 or hashlib.sha256(clips[:ami[0]].tobytes()).hexdigest() != reference["pcm_sha256"]:
        raise ValueError("the archived caller or development input did not reproduce")
    path = root / "cache/wav2vec2-harper" / BUNDLE
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    if (file_digest(path) != CHECKPOINT_SHA256 or bundle._path != BUNDLE
            or torchaudio.__version__.split("+")[0] != "2.8.0"):
        raise ValueError("unexpected general speech encoder")
    model = torchaudio.models.wav2vec2_model(**bundle._params)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval().to("cuda")
    predict = lambda audio: speech_features(model, audio)
    original = extract("ami-transfer-baseline", predict, clips[:ami[0]])
    if hashlib.sha256(original.tobytes()).hexdigest() != anchor["pooled_features_sha256"]:
        raise ValueError("the archived frozen features did not reproduce")
    features = np.concatenate((original, extract("ami-transfer-supplement", predict, clips[ami])))
    del clips, original, model, predict
    evaluation = np.r_[english, harper, cal]
    fit_sets = {"caller": np.array([], dtype=int), "ami": ami, "ami-filtered": ami[filtered]}
    scores, heads, nulls = {}, {}, {}
    meter = Progress("harper-ami-transfer", len(HEAD_SEEDS) * len(fit_sets) + 2 * len(SHUFFLE_SEEDS), "fits")
    try:
        for arm, supplement in fit_sets.items():
            fit_rows = np.r_[caller, supplement]
            weight = mixture_weights(truth[caller], truth[supplement]) if len(supplement) else None
            scores[arm], heads[arm] = [], []
            for seed in HEAD_SEEDS:
                score, head = fit_scores(features, truth, fit_rows, evaluation, caller, seed, weight)
                scores[arm].append(score)
                heads[arm].append(head)
                meter.advance(1)
            if not len(supplement):
                continue
            blocks = defaultdict(list)
            for i, position in enumerate(supplement - ami[0]):
                row = ami_labels[position]
                blocks[row.speaker, row.recording].append(i)
            labels = truth[supplement]
            movable = sum(len(idx) for idx in blocks.values() if len(np.unique(labels[idx])) == 2)
            draws = []
            for seed in SHUFFLE_SEEDS:
                permuted = shuffle(labels, blocks, seed)
                altered = truth.copy()
                altered[supplement] = permuted
                permuted_weight = mixture_weights(truth[caller], permuted)
                score, _ = fit_scores(features, altered, fit_rows, evaluation, caller, HEAD_SEEDS[0], permuted_weight)
                draws.append({"seed": seed, "changed": int(np.count_nonzero(permuted != labels)),
                              **rates(score[:len(english)], truth[english])})
                meter.advance(1)
            nulls[arm] = {"rows": len(supplement), "movable_rows": movable, "draws": draws,
                          "recall_p95": float(np.quantile([item["recalls_at_2_5_5_7_5pct"][1] for item in draws], .95)),
                          "matched_true_seed_recall": rates(scores[arm][0][:len(english)],
                                                            truth[english])["recalls_at_2_5_5_7_5pct"][1]}
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    averaged = {arm: np.mean(values, axis=0) for arm, values in scores.items()}
    results = {}
    for name, idx in (("english", english), ("harper", harper)):
        results[name] = {}
        relative = idx - english[0]
        for arm, score in averaged.items():
            threshold = _threshold(score[-len(cal):], truth[cal])
            results[name][arm] = {**rates(score[relative], truth[idx]), "ami_threshold": threshold,
                                  "ami_fixed": _stats(score[relative].tolist(), truth[idx].tolist(), threshold)}
        for key in ("auc", "partial_auc", "recalls_at_2_5_5_7_5pct", "ami_threshold"):
            if not np.allclose(results[name]["caller"][key], reference["cohorts"][name]["pretrained"][key],
                               rtol=0, atol=1e-10):
                raise ValueError(f"the archived frozen head {name} {key} did not reproduce")
    en = results["english"]
    recall = {name: value["recalls_at_2_5_5_7_5pct"][1] for name, value in en.items()}
    paired = paired_intervals({arm: values[:len(english)] for arm, values in averaged.items()}, truth[english],
                              [row[2] for row in cohorts["english"]],
                              {arm: "caller" for arm in nulls}, partial_auc=True)
    gates = {arm: {
        "english_recall_gain_at_least_five_points": recall[arm] + 1e-12 >= recall["caller"] + .05,
        "paired_recall_gain": paired[arm]["recall_delta_95"][0] > 0,
        "english_partial_auc_within_tolerance": en[arm]["partial_auc"] >= en["caller"]["partial_auc"] - .005,
        "matched_seed_exceeds_shuffle_null": nulls[arm]["matched_true_seed_recall"] > nulls[arm]["recall_p95"],
        "harper_partial_auc_direction": results["harper"][arm]["partial_auc"] >= results["harper"]["caller"]["partial_auc"],
    } for arm in nulls}
    passing = [arm for arm, checks in gates.items() if all(checks.values())]
    winner = max(passing, key=recall.__getitem__) if passing else None
    folder = root / "runs/harper-ami-transfer"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "heads.pt"
    temporary = target.with_suffix(".tmp")
    selected = ("caller", winner) if winner else ("caller",)
    torch.save({"encoder_sha256": CHECKPOINT_SHA256, "arm_states": {arm: heads[arm] for arm in selected},
                "seed_order": HEAD_SEEDS, "ensemble": "mean-probability"}, temporary)
    payload = torch.load(temporary, map_location="cpu", weights_only=True)
    for arm in selected:
        replay = np.mean([predict_head(features[evaluation], head)
                          for head in payload["arm_states"][arm]], axis=0)
        if not np.array_equal(replay, averaged[arm]):
            raise ValueError("the serialized head predictions did not reproduce")
    temporary.replace(target)
    report = {"decision_ms": 300, "encoder_sha256": CHECKPOINT_SHA256, "reference_sha256": file_digest(reference_path),
              "head_checkpoint": {"path": target.relative_to(root).as_posix(), "sha256": file_digest(target)},
              "original_features_sha256": anchor["pooled_features_sha256"],
              "own_head_probability_sha256": {arm: hashlib.sha256(averaged[arm].tobytes()).hexdigest()
                                              for arm in selected},
              "ami_weight_relative_to_caller": AMI_WEIGHT, "ami_strata": strata, "scores": results,
              "nulls": nulls, "paired_english": paired, "advancement": gates, "winner": winner}
    _write_json(reports / "harper-ami-transfer.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = run(data_root())
    print(json.dumps({key: result[key] for key in ("scores", "paired_english", "advancement", "winner")}, indent=2))
