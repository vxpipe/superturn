"""Compare limited Wav2Vec2 adaptation with the exact frozen caller baseline."""

from collections import defaultdict
import hashlib
import json
import math

import numpy as np
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .elapsed_turn_probe import paired_intervals
from .harper_agent_probe import HEAD_SEEDS, shuffle
from .progress import Progress
from .sources import data_root
from .train import _stats, _threshold
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import (
    BUNDLE, CHECKPOINT_SHA256, calibration_rows, extract, fit_head, head_logits,
    pause_rows, rates, speech_features, training_rows,
)

PENALTY = .05
HEAD_RATE = .0001
ADAPTER_RATE = .001
EPOCHS = 4
BATCH = 32
RANK = 2
ADAPTER_PARAMETERS = 36_864


class ValueAdapter(nn.Module):
    def __init__(self, base, generator):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(base.weight.new_empty(RANK, base.in_features))
        self.up = nn.Parameter(base.weight.new_zeros(base.out_features, RANK))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5), generator=generator)

    def forward(self, features):
        return self.base(features) + F.linear(F.linear(features, self.down), self.up)


def adapt_values(model, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    adapters = []
    for block in model.encoder.transformer.layers:
        attention = block.attention
        attention.v_proj = ValueAdapter(attention.v_proj, generator).eval()
        adapters.append(attention.v_proj)
    return adapters


def restore_values(model):
    for block in model.encoder.transformer.layers:
        attention = block.attention
        attention.v_proj = attention.v_proj.base


def normalization_drift(old, new):
    old_std = old.std(0).clip(min=1e-5)
    mean_change = (new.mean(0) - old.mean(0)) / old_std
    relative_std = new.std(0) / old_std
    return {"mean_change_abs_quantiles": np.quantile(np.abs(mean_change), [.5, .95, 1]).tolist(),
            "std_ratio_quantiles": np.quantile(relative_std, [.05, .5, .95]).tolist()}


def continued_fit(model, features, clips, labels, initial, batches, arm, seed, meter):
    center, scale, *original = [tensor.to("cuda") for tensor in initial]
    head = [value.clone().requires_grad_() for value in original]
    fit_encoder = arm != "head-only"
    adapters = adapt_values(model, seed) if fit_encoder else []
    tunable = [parameter for adapter in adapters for parameter in (adapter.down, adapter.up)]
    if fit_encoder and sum(p.numel() for p in tunable) != ADAPTER_PARAMETERS:
        raise ValueError("unexpected adapter parameter count")
    if fit_encoder:
        unchanged = extract(f"{arm}-{seed}-initial", lambda audio: speech_features(model, audio), clips)
        delta = np.abs(unchanged - features).max()
        if delta:
            with torch.no_grad():
                first = head_logits((torch.as_tensor(unchanged, device="cuda") - center) / scale, *head)
                original_logits = head_logits((torch.as_tensor(features, device="cuda") - center) / scale, *head)
                if (first - original_logits).abs().max().item() > 1e-6:
                    raise ValueError("zero adapters changed the initial logits")
    else:
        delta = 0.0
    cached = torch.as_tensor(features, device="cuda")
    target = torch.as_tensor(labels, dtype=torch.float64, device="cuda")
    parameter_groups = [{"params": head, "lr": HEAD_RATE}]
    if fit_encoder:
        parameter_groups.append({"params": tunable, "lr": ADAPTER_RATE})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0)
    history = []
    first_gradient = 0.0
    try:
        for epoch, orders in enumerate(batches):
            loss_sum = 0.0
            for step, idx in enumerate(orders):
                optimizer.zero_grad(set_to_none=True)
                batch = speech_features(model, torch.as_tensor(clips[idx], device="cuda")) if fit_encoder else cached[idx]
                layer, weights, _ = head
                logits = head_logits((batch - center) / scale, *head)
                loss = F.binary_cross_entropy_with_logits(logits, target[idx])
                loss = loss + PENALTY * weights.square().sum() / 2 + .001 * layer.square().mean()
                loss.backward()
                if fit_encoder and epoch == step == 0:
                    first_gradient = torch.linalg.vector_norm(torch.stack(
                        [param.grad.norm() for param in tunable])).item()
                    if not math.isfinite(first_gradient) or first_gradient <= 0:
                        raise ValueError("the adapter did not receive gradients")
                optimizer.step()
                loss_sum += loss.detach().item() * len(idx)
                meter.advance(1)
            residual = sum((adapter.up @ adapter.down).detach().square().sum().item() for adapter in adapters) ** .5
            if not math.isfinite(loss_sum) or (fit_encoder and (not math.isfinite(residual) or residual <= 0)):
                raise ValueError("nonfinite fit or unchanged adapter weights")
            history.append({"regularized_bce": loss_sum / len(labels), "residual_norm": residual})
        final = extract(f"{arm}-{seed}-final", lambda audio: speech_features(model, audio), clips) if fit_encoder else features
        with torch.no_grad():
            score = head_logits((torch.as_tensor(final, device="cuda") - center) / scale, *head).cpu().numpy()
        state = {"head": [tensor.detach().cpu() for tensor in (center, scale, *head)],
                 "adapters": [(adapter.down.detach().cpu(), adapter.up.detach().cpu()) for adapter in adapters]}
        return expit(score), state, {
            "epochs": history, "initial_max_pooled_difference": float(delta), "first_gradient_norm": first_gradient,
            "training_feature_drift": normalization_drift(features[:len(labels)], final[:len(labels)]),
        }
    finally:
        if fit_encoder:
            restore_values(model)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this experiment uses only staged Vast audio")
    reports = root / "reports"
    reference_path = reports / "wav2vec-harper-probe.json"
    reference = json.loads(reference_path.read_text())
    anchor = reference["heads"]["pretrained"]
    if anchor["penalty"] != PENALTY or reference["model_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("unexpected archived frozen head")
    cohorts = {"train": list(training_rows(root)), "english": list(pause_rows(english_rows(root))),
               "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
               "calibration": list(calibration_rows(root))}
    clips = np.stack([clip for rows in cohorts.values() for clip, _, _ in rows]).astype(np.float32)
    if hashlib.sha256(clips.tobytes()).hexdigest() != reference["pcm_sha256"]:
        raise ValueError("the archived causal inputs did not reproduce")
    truth = np.asarray([label for rows in cohorts.values() for _, label, _ in rows])
    partitions, start = {}, 0
    for name, rows in cohorts.items():
        partitions[name] = np.arange(start, start + len(rows))
        start += len(rows)
    train, english, cal = (partitions[name] for name in ("train", "english", "calibration"))
    checkpoint = root / "cache/wav2vec2-harper" / BUNDLE
    if file_digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("the general speech encoder checkpoint changed")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    if torchaudio.__version__.split("+")[0] != "2.8.0" or bundle._path != BUNDLE:
        raise ValueError("unexpected encoder implementation")
    model = torchaudio.models.wav2vec2_model(**bundle._params)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.requires_grad_(False).eval().to("cuda")
    features = extract("harper-adaptation-control", lambda audio: speech_features(model, audio), clips)
    if hashlib.sha256(features.tobytes()).hexdigest() != anchor["pooled_features_sha256"]:
        raise ValueError("the archived frozen features did not reproduce")

    membership = json.loads((reports / "harper-history-replication-membership.json").read_text())
    by_call = defaultdict(list)
    for idx, (call, _, label) in enumerate(membership):
        if (label == "shift") != truth[idx]:
            raise ValueError("the within-call shuffle rows are misaligned")
        by_call[cohorts["train"][idx][2], call].append(idx)
    if sum(map(len, by_call.values())) != len(train):
        raise ValueError("the within-call shuffle did not cover the training rows")
    arms = ("head-only", "adapter", "shuffle")
    scores = {arm: [] for arm in ("frozen", *arms)}
    diagnostics, candidate_states = {}, {}
    meter = Progress("harper-adapter-fit", len(HEAD_SEEDS) * len(arms) * EPOCHS * math.ceil(len(train) / BATCH), "updates")
    try:
        for seed in HEAD_SEEDS:
            initial = fit_head(features, truth, train, PENALTY, seed)
            center, scale, *head = initial
            with torch.no_grad():
                scores["frozen"].append(expit(head_logits((torch.as_tensor(features) - center) / scale, *head).numpy()))
            rng = np.random.default_rng(seed)
            batches = [[indices[start:start + BATCH] for start in range(0, len(train), BATCH)]
                       for indices in (rng.permutation(train) for _ in range(EPOCHS))]
            permuted = shuffle(truth[train], by_call, seed)
            changed = int(np.sum(permuted != truth[train]))
            if not changed:
                raise ValueError("the shuffle did not change any training labels")
            diagnostics[seed] = {"shuffle_changed_rows": changed}
            for arm in arms:
                labels = permuted if arm == "shuffle" else truth[train]
                score, state, details = continued_fit(model, features, clips, labels, initial, batches, arm, seed, meter)
                scores[arm].append(score)
                diagnostics[seed][arm] = {**details, "english": rates(score[english], truth[english])}
                if arm == "adapter":
                    candidate_states[seed] = state
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    averaged = {arm: np.mean(per_seed, axis=0) for arm, per_seed in scores.items()}
    results = {}
    for cohort in ("english", "harper"):
        idx = partitions[cohort]
        results[cohort] = {}
        for arm, score in averaged.items():
            threshold = _threshold(score[cal], truth[cal])
            results[cohort][arm] = {**rates(score[idx], truth[idx]), "ami_threshold": threshold,
                                    "ami_fixed": _stats(score[idx].tolist(), truth[idx].tolist(), threshold)}
        for key in ("auc", "partial_auc", "recalls_at_2_5_5_7_5pct"):
            if not np.allclose(results[cohort]["frozen"][key], reference["cohorts"][cohort]["pretrained"][key],
                               rtol=0, atol=1e-10):
                raise ValueError(f"frozen {cohort} {key} did not reproduce")

    en = results["english"]
    recall = {name: value["recalls_at_2_5_5_7_5pct"][1] for name, value in en.items()}
    comparator = max(("frozen", "head-only"), key=recall.__getitem__)
    comparison_scores = {arm: averaged[arm][english] for arm in ("head-only", "frozen")}
    comparison_scores.update({f"adapter-vs-{arm}": averaged["adapter"][english] for arm in ("head-only", "frozen")})
    paired = paired_intervals(comparison_scores, truth[english], [row[2] for row in cohorts["english"]],
                              {f"adapter-vs-{arm}": arm for arm in ("head-only", "frozen")}, partial_auc=True)
    harper = results["harper"]
    gates = {
        "english_recall_gain_at_least_five_points": recall["adapter"] + 1e-12 >= recall[comparator] + .05,
        "paired_recall_gain": paired["adapter-vs-head-only"]["recall_delta_95"][0] > 0,
        "english_pauc_within_tolerance": en["adapter"]["partial_auc"] >= en[comparator]["partial_auc"] - .005,
        "shuffle_does_not_improve_reference": recall["shuffle"] <= recall[comparator],
        "harper_pauc_direction": harper["adapter"]["partial_auc"] >= harper["frozen"]["partial_auc"],
    }
    report = {"decision_ms": 300, "reference_sha256": file_digest(reference_path),
              "head_lr": HEAD_RATE, "adapter_lr": ADAPTER_RATE, "adapter_parameters": ADAPTER_PARAMETERS,
              "epoch_budget": EPOCHS, "batch_size": BATCH, "results": results, "seeds": diagnostics,
              "english_paired": paired, "gates": gates, "advanced": all(gates.values())}
    if report["advanced"]:
        relative = "runs/harper-wav2vec-adaptation/adapter.pt"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_suffix(".tmp")
        torch.save({"encoder_sha256": CHECKPOINT_SHA256, "rank": RANK, "seeds": candidate_states}, staged)
        report["checkpoint"] = {"path": relative, "sha256": file_digest(staged)}
        staged.replace(destination)
    _write_json(reports / "harper-wav2vec-adaptation.json", report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = run(data_root())
    print(json.dumps({key: result[key] for key in ("results", "english_paired", "gates", "advanced")}, indent=2))
