"""Test real-label adaptation of the fidelity-locked compact audio student."""

from collections import Counter
import json

import numpy as np
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .ami_transfer import partial_auc
from .dense_teacher import file_digest
from .dense_teacher_train import batch
from .harper_partial_stt import pinned_membership
from .model import build
from .own_head_distillation import DEADLINE, rates, student_scores
from .own_teacher_feature_student import DEVICE, SEEDS, read_inputs
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .train import _stats, _threshold
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import pause_rows, render_ami, training_rows, verified_ami_labels

ARMS = ("frozen", "head", "full", "full-ami", "rehearsal", "shuffle")
STEPS = 64


def arrays(rows):
    clips, labels, groups = zip(*rows, strict=True)
    return np.stack(clips).astype(np.float32), np.asarray(labels), np.asarray(groups)


def fit(root, report, arm, seed, data, replay, *, attenuate=False, output_dir=None):
    path = root / report["path"]
    if file_digest(path) != report["sha256"]:
        raise ValueError("feature-first checkpoint changed")
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build(payload["config"]).to(DEVICE)
    model.load_state_dict(payload["state"])
    if arm == "frozen":
        return model, None, None
    torch.manual_seed(seed)
    if arm == "head":
        model.requires_grad_(False)
        model.decision.requires_grad_(True)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()),
                                  lr=.001 if arm == "head" else .0001)
    harper, labels, caller = data["harper-train"]
    training = labels.copy()
    if arm == "shuffle":
        rng = np.random.default_rng(seed_for("harper-adaptation-null", seed))
        for identity in np.unique(caller):
            rows = np.flatnonzero(caller == identity)
            training[rows] = rng.permutation(training[rows])
    harper_rng = np.random.default_rng(seed_for("harper-adaptation-order", seed))
    supplement_rng = np.random.default_rng(seed_for("harper-adaptation-supplement", seed))
    gain_rng = np.random.default_rng(seed_for("harper-adaptation-attenuation", seed)) if attenuate else None
    harper_pools = [np.flatnonzero(training == value) for value in (0, 1)]
    ami, ami_labels, _ = data["ami-train"]
    ami_pools = [np.flatnonzero(ami_labels == value) for value in (0, 1)]
    replay_pcm, targets, replay_groups = replay
    name = f"{arm}-{seed}" + ("-attenuated" if attenuate else "")
    meter = Progress(f"own-student-real-{name}", STEPS, "updates")
    losses = []
    try:
        for _ in range(STEPS):
            h = np.concatenate([harper_rng.choice(rows, 16) for rows in harper_pools])
            segments, truth = [harper[h]], [training[h]]
            weights = [np.full(32, 1 / 32)]
            if arm == "full-ami":
                a = np.concatenate([supplement_rng.choice(rows, 8) for rows in ami_pools])
                segments.append(ami[a])
                truth.append(ami_labels[a])
                weights.append(np.full(16, .5 / 16))
            if arm == "rehearsal":
                r = np.concatenate([supplement_rng.choice(rows, 8) for rows in replay_groups])
                audio, _ = batch(replay_pcm, r, np.zeros(len(r), dtype=int), DEVICE, (DEADLINE,))
                segments.append(audio.cpu().numpy())
                truth.append(targets[r, 0])
                weights.append(np.full(len(r), .25 / len(r)))
            x, y, w = [torch.as_tensor(np.concatenate(values), dtype=torch.float32, device=DEVICE)
                       for values in (segments, truth, weights)]
            delays = torch.full((len(x), 1), float(DEADLINE), device=DEVICE)
            logits = model.logits(x, delays).flatten()
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum()
            if gain_rng is not None:
                gain = torch.as_tensor(10 ** (gain_rng.uniform(-6, 0, (len(x), 1)) / 20),
                                       dtype=x.dtype, device=DEVICE)
                augmented = model.logits(x * gain, delays).flatten()
                loss_augmented = (F.binary_cross_entropy_with_logits(augmented, y, reduction="none") * w).sum() / w.sum()
                loss = (loss + loss_augmented) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    directory = root / "runs/own-student-real-labels" if output_dir is None else output_dir
    output = directory / f"{name}.pt"
    temporary = output.with_suffix(".part")
    torch.save({"config": payload["config"], "state": model.state_dict()}, temporary)
    temporary.replace(output)
    return model, {"path": output.relative_to(root).as_posix(), "sha256": file_digest(output),
                   "loss_first_last": [float(losses[0]), float(losses[-1])]}, float(np.mean(labels != training))


def compare(predictions, data):
    reports = {}
    for name, scores in predictions.items():
        thresholds = {source: _threshold(scores[source].tolist(), data[source][1].tolist())
                      for source in ("ami-calibration", "harper-validation")}
        reports[name] = {}
        for role in ("english", "harper-validation"):
            truth = data[role][1]
            roles = thresholds if role == "english" else {"ami-calibration": thresholds["ami-calibration"]}
            reports[name][role] = {**rates(scores[role], truth),
                                   "partial_auc_to_5pct": partial_auc(scores[role].tolist(), truth.tolist(), max_fpr=.05),
                                   "fixed": {source: {"threshold": cut, **_stats(scores[role].tolist(),
                                                                                  truth.tolist(), cut)}
                                             for source, cut in roles.items()}}
    return reports


def paired_means(predictions, labels, groups):
    scores = {arm: np.mean([predictions[f"{arm}-{seed}"]["english"] for seed in SEEDS], axis=0)
              for arm in ARMS}
    buckets = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    draws = np.random.default_rng(811).integers(len(buckets), size=(500, len(buckets)))
    from .harper_history_probe import recall_area

    deltas = {key: [] for key in ARMS if key != "frozen"}
    for draw in draws:
        index = np.concatenate([buckets[i] for i in draw])
        outcomes = {arm: recall_area(score[index], labels[index])[:3].mean() for arm, score in scores.items()}
        for arm in deltas:
            deltas[arm].append(outcomes[arm] - outcomes["frozen"])
    return {name: np.quantile(values, [.025, .975]).tolist() for name, values in deltas.items()}


def checked_inputs(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the experiment requires staged Vast data")
    receipt = root / "reports/own-teacher-feature-student.json"
    parent = json.loads(receipt.read_text())
    _, _, pcm, scores, sources, roles, _, pool_digest = read_inputs(root)
    if pool_digest != parent["target_report_sha256"] or "harper_valley" in sources or "ami" in sources:
        raise ValueError("unexpected adaptation-source membership in teacher pool")
    cal = verified_ami_labels(root, "calibration")
    ami_train = verified_ami_labels(root, "train")
    for field in ("speaker", "group", "recording"):
        if {getattr(row, field) for row in ami_train} & {getattr(row, field) for row in cal}:
            raise ValueError(f"AMI calibration and fitting overlap by {field}")
    fit_members, train_membership = pinned_membership(root, "train")
    valid_members, validation_membership = pinned_membership(root, "validation")
    if {row[-1] for row in fit_members} & {row[-1] for row in valid_members}:
        raise ValueError("Harper development voices cross fit/calibration")
    views = {"harper-train": arrays(training_rows(root)),
             "harper-validation": arrays(pause_rows(harper_rows(root, native_prefix=True))),
             "ami-train": arrays(render_ami(root, ami_train)),
             "ami-calibration": arrays(render_ami(root, [row for row in cal if row.label == "hold"])),
             "english": arrays(pause_rows(english_rows(root)))}
    if len(views["harper-train"][1]) != len(fit_members) or len(views["harper-validation"][1]) != len(valid_members):
        raise ValueError("Harper materialization and pinned roles disagree")
    replay = pcm, scores, [np.flatnonzero((roles == "fit") & (sources == source)) for source in np.unique(sources)]
    return receipt, parent, views, replay, train_membership, validation_membership


def run(root):
    receipt, parent, views, replay, train_membership, validation_membership = checked_inputs(root)
    train_output = root / "runs/own-student-real-labels"
    train_output.mkdir(parents=True, exist_ok=True)
    predictions, fits = {}, {}
    torch.use_deterministic_algorithms(True)
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}-{seed}"
            provenance = parent["fits"][f"feature-first-{seed}"]["checkpoint"]
            model, artifact, moved = fit(root, provenance, arm, seed, views, replay)
            fits[key] = {"parent": provenance, "checkpoint": artifact, "shuffled_labels_fraction": moved}
            predictions[key] = {role: student_scores(model, views[role][0])
                                for role in ("english", "harper-validation", "ami-calibration")}
    metrics = compare(predictions, views)
    means = {arm: {role: np.mean([predictions[f"{arm}-{seed}"][role] for seed in SEEDS], axis=0)
                   for role in ("english", "harper-validation", "ami-calibration")} for arm in ARMS}
    average_scores = compare(means, views)
    if metrics["frozen-29"]["english"]["fixed"]["ami-calibration"]["recall"] != .01:
        raise ValueError("the archived frozen model and AMI threshold did not reproduce")
    bootstrap = paired_means(predictions, views["english"][1], views["english"][2])
    report = {"scope": "development experiment, fixed training schedule; primary threshold-free English metrics",
              "steps": STEPS, "decision_ms": DEADLINE, "seeds": SEEDS, "fit_report_sha256": file_digest(receipt),
              "harper_train_membership_sha256": train_membership, "harper_validation_membership_sha256": validation_membership,
              "counts": {role: dict(Counter(map(int, values[1]))) for role, values in views.items()},
              "metrics": metrics, "averaged_score_metrics": average_scores,
              "paired_seed_mean_recall_delta_95": bootstrap, "fits": fits}
    _write_json(root / "reports/own-student-real-labels.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    report = run(data_root())
    print(json.dumps({key: value["english"] for key, value in report["metrics"].items()}, indent=2))
