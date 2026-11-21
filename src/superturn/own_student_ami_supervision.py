"""Compare replay, owned-teacher scores, and observed AMI floor labels."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json

import numpy as np
from scipy.special import xlogy
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .harper_ami_probe import ami_cohorts
from .language_failure_audit import read_receipt, read_verified
from .model import HOP, build
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import DEADLINE, rates, student_scores
from .own_student_hop_probe import recall_metrics
from .own_student_real_labels import arrays
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, SEEDS, compare, read_inputs
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows, render_ami, verified_ami_labels

ARMS = ("replay", "teacher", "observed", "teacher-clean", "observed-clean")
CONTRASTS = (("teacher", "replay"), ("observed", "teacher"), ("observed-clean", "teacher-clean"))
STEPS = 64
BATCH = 64
REPLACEMENTS = 16
LR = .0001


def component_order(rows, indices, rng):
    groups = defaultdict(list)
    for i in indices:
        groups[rows[i].group].append(i)
    sources = [rng.permutation(group) for group in groups.values()]
    return [int(sources[i][depth]) for depth in range(max(map(len, sources)))
            for i in rng.permutation(len(sources)) if depth < len(sources[i])]


def schedules(fit, rows, clean, seed):
    rng = np.random.default_rng(seed_for("ami-supervision-replay-v1", seed))
    if any(not len(group) for group in fit) or BATCH % len(fit) or REPLACEMENTS % len(fit):
        raise ValueError("the replay source mixture cannot be matched")

    def replay_batch(count):
        return rng.permutation(np.concatenate([rng.choice(group, count // len(fit)) for group in fit]))

    replay = np.asarray([np.r_[replay_batch(BATCH - REPLACEMENTS), replay_batch(REPLACEMENTS)] for _ in range(STEPS)])
    matched = {}
    for name, subset in (("all", np.arange(len(rows))), ("clean", clean)):
        order = []
        cells = np.random.default_rng(seed_for("ami-supervision-order-v1", seed, name))
        while len(order) < STEPS * REPLACEMENTS:
            order.extend(component_order(rows, subset, cells))
        matched[name] = np.asarray(order[:STEPS * REPLACEMENTS]).reshape(STEPS, REPLACEMENTS)
        if not np.array_equal(np.unique(matched[name]), subset):
            raise ValueError("a matching schedule missed eligible rows")
    return replay, matched


def train(root, parent, arm, seed, replay, supplement, order, *, supplement_weight=None,
          progress_name="ami-supervision"):
    weights = torch.load(read_verified(root, parent), map_location="cpu", weights_only=True)
    model = build(weights["config"]).to(DEVICE)
    if model.features.hop != HOP or sum(p.numel() for p in model.parameters()) != 972795:
        raise ValueError("the control is not the 973k 10 ms student")
    model.load_state_dict(weights["state"])
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    replay_pcm, replay_target = replay
    audio, targets = supplement
    replay_rows, supplements = order
    elapsed = torch.full((BATCH, 1), float(DEADLINE), device=DEVICE)
    loss_weights = None
    if supplement_weight is not None:
        loss_weights = torch.full((BATCH,), (1 - supplement_weight) / (BATCH - REPLACEMENTS), device=DEVICE)
        loss_weights[-REPLACEMENTS:] = supplement_weight / REPLACEMENTS
    meter = Progress(f"{progress_name}-{arm}-{seed}", STEPS, "updates")
    try:
        for step, rows in enumerate(replay_rows):
            clip_batch = replay_pcm[rows, 0].astype(np.float32) / 32768
            y = replay_target[rows, 0].copy()
            if arm != "replay":
                subset = supplements["clean" if arm.endswith("-clean") else "all"][step]
                clip_batch[-REPLACEMENTS:] = audio[subset]
                y[-REPLACEMENTS:] = targets[subset]
            x = torch.as_tensor(clip_batch, device=DEVICE)
            y = torch.as_tensor(y, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(x, elapsed).flatten()
            if loss_weights is None:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, weight=loss_weights, reduction="sum")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return weights["config"], {key: value.detach().cpu() for key, value in model.state_dict().items()}, {
        "parent": parent, "last_loss": float(loss.detach())}


def summarize(rows, *, arm_names=ARMS, contrasts=CONTRASTS):
    rng = np.random.default_rng(29062)
    individual, draws = {}, {}
    paired = [(arm_names.index(arm), arm_names.index(ref)) for arm, ref in contrasts]
    for language, row in rows.items():
        truth = np.asarray(row["labels"], dtype=bool)
        groups = np.asarray(row["turn_ordinals"])
        values = [np.asarray([row["scores"][f"{arm}-{seed}"] for seed in SEEDS]) for arm in arm_names]

        def metrics(idx):
            return np.asarray([recall_metrics(arm[:, idx], truth[idx]) for arm in values])

        individual[language] = metrics(np.arange(len(truth))).tolist()
        buckets = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        samples = []
        for pick in rng.integers(len(buckets), size=(400, len(buckets))):
            outcome = metrics(np.concatenate([buckets[i] for i in pick]))
            samples.append([outcome[arm] - outcome[ref] for arm, ref in paired])
        draws[language] = samples

    def limits(values):
        return np.quantile(values, [.025, .975], axis=0).transpose(1, 2, 0).tolist()

    return {"arm_order": arm_names, "contrast_order": contrasts,
            "metrics_order": ["mean_recall_2.5_5_7.5pct", "matched_5pct_recall", "normalized_pauc_10pct"],
            "macro": np.mean(list(individual.values()), axis=0).tolist(), "by_language": individual,
            "paired_macro_95": limits(np.mean(list(draws.values()), axis=0)), "paired_english_95": limits(draws["en"])}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI adaptation uses staged Vast data")
    audit, audit_hash = read_receipt(root, "own-student-ami-targets")
    width, width_hash = read_receipt(root, "own-student-width-probe")
    if audit["width_sha256"] != width_hash:
        raise ValueError("the AMI audit used different parent checkpoints")
    _, _, pcm, targets, source, roles, _, target_hash = read_inputs(root)
    if target_hash != width["target_report_sha256"]:
        raise ValueError("replay is not the width experiment's distillation pool")
    replay_groups = [np.flatnonzero((source == name) & (roles == "fit")) for name in np.unique(source)]
    train_rows, _, clean, _ = ami_cohorts(root)
    val_rows = verified_ami_labels(root, "validation")
    ami, y, _ = arrays(render_ami(root, train_rows))
    val, val_y, _ = arrays(render_ami(root, val_rows))
    for role, audio in (("train", ami), ("validation", val)):
        if sha256(audio.tobytes()).hexdigest() != audit["cohorts"][role]["render_sha256"]:
            raise ValueError(f"the AMI {role} render changed")
    with np.load(read_verified(root, audit["cache"]), allow_pickle=False) as stored:
        if not np.array_equal(y, stored["train_labels"]) or not np.array_equal(clean, stored["train_clean"]):
            raise ValueError("AMI fit membership changed since teacher scoring")
        teacher = stored["train_normalized"]
        val_clean = stored["validation_clean"]
    binary = y.astype(np.float32)
    entropy = -(xlogy(teacher, teacher) + xlogy(1 - teacher, 1 - teacher))
    torch.use_deterministic_algorithms(True)
    fits, membership = {}, {}
    output = root / "runs/own-student-ami-supervision" / audit_hash[:12]
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        order = schedules(replay_groups, train_rows, clean, seed)
        membership[seed] = {"replay": sha256(order[0].tobytes()).hexdigest(),
                            **{kind: sha256(ids.tobytes()).hexdigest() for kind, ids in order[1].items()}}
        parent = width["fits"][model_name(1_000_000, seed)]["checkpoint"]
        for arm in ARMS:
            fits[f"{arm}-{seed}"] = train(root, parent, arm, seed, (pcm, targets),
                                         (ami, teacher if arm.startswith("teacher") else binary), order)
    peers = {f"{arm}-{seed}": f"{ref}-{seed}" for arm, ref in CONTRASTS for seed in SEEDS}
    metrics, intervals = compare(root, output, fits, peers=peers)
    models, ami_validation = {}, {}
    for name, (config, state, _) in fits.items():
        models[name] = build(config).to(DEVICE).eval()
        models[name].load_state_dict(state)
        scores = student_scores(models[name], val)
        ami_validation[name] = {"all": rates(scores, val_y), "clean": rates(scores[val_clean], val_y[val_clean])
                                if len(np.unique(val_y[val_clean])) == 2 else None}
    report = {"scope": "matched continuation on AMI floor proxies", "ami_target_audit_sha256": audit_hash,
              "width_report_sha256": width_hash, "seeds": SEEDS, "steps": STEPS, "batch": BATCH,
              "learning_rate": LR, "replacement_slots": REPLACEMENTS, "schedule_sha256": membership,
              "soft_target_entropy_nats": {"full": float(entropy.mean()), "clean": float(entropy[clean].mean())},
              "hard_target_entropy_nats": 0,
              "clean_validation_size": len(val_clean), "ami_validation": ami_validation,
              "fits": {name: values[2] for name, values in fits.items()}, "metrics": metrics, "paired_english": intervals}
    path = root / "reports/own-student-ami-supervision.json"
    _write_json(path, report)
    rows = {}
    score_languages(root, models, DEVICE, paired_rows=rows, progress_name="ami-supervision-multilingual")
    clips, truth, groups = zip(*pause_rows(english_rows(root)), strict=True)
    audio = np.stack(clips)
    rows["en"] = {"labels": truth, "turn_ordinals": groups,
                  "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    report["summary"] = summarize(rows)
    pairs = output / "paired-language-scores.json"
    _write_json(pairs, rows)
    report["paired_language_scores"] = {"path": pairs.relative_to(root).as_posix(), "sha256": file_digest(pairs)}
    _write_json(path, report)
    return report


def frozen_thresholds(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the score replay uses staged Vast data")
    report, digest = read_receipt(root, "own-student-ami-supervision")
    rows = json.loads(read_verified(root, report["paired_language_scores"]).read_text())
    languages = {}
    for language, row in rows.items():
        labels = np.asarray(row["labels"], dtype=bool)
        shifts, holds = int(labels.sum()), int((~labels).sum())
        models = {}
        for name, scores in row["scores"].items():
            threshold = report["metrics"]["english"][name]["threshold"]
            fired = np.asarray(scores) >= threshold
            models[name] = {"threshold": threshold, "shift_hits": int((labels & fired).sum()),
                            "false_completes": int((~labels & fired).sum())}
        languages[language] = {"shifts": shifts, "holds": holds, "models": models}
    output = {"scope": "development scores, each AMI calibration cutoff applied unchanged",
              "training_report_sha256": digest, "languages": languages}
    _write_json(root / "reports/own-student-ami-supervision-locked.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-thresholds", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.frozen_thresholds:
        frozen_thresholds(data_root())
    else:
        print(json.dumps(run(data_root())["summary"], indent=2))
