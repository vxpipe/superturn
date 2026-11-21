"""Compare score and frame-feature training for the compact own-teacher student."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .dense_teacher import SAMPLES, SOURCES, file_digest
from .dense_teacher_train import BATCH, EPOCHS, batch, evaluate_fidelity, shuffled_targets
from .elapsed_turn_probe import paired_intervals
from .eot_domain_probe import load_validated_models
from .model import FRAME, HOP, build
from .multilingual_teacher_probe import DATA, score_languages
from .own_head_distillation import (
    BASELINE, DEADLINE, calibration_rows, pause_rows, pool_rows, rates, student_scores,
)
from .own_teacher_pool_pcm import recover
from .own_teacher_targets import SHARD_ROWS
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .train import _stats, _threshold
from .turn1_mini_probe import english_rows, harper_rows

DEVICE = "cuda"
DEADLINES = (DEADLINE,)
ARMS = ("score", "feature-first", "joint")
SEEDS = (29, 17, 43)
PCA_SIZE = 64
TAIL_FRAMES = 25
JOINT_WEIGHT = .1


def read_inputs(root):
    report_path = root / "reports/own-teacher-distillation-targets.json"
    report = json.loads(report_path.read_text())
    cache = root / report["cache"]
    pcm_path, pcm_receipt = recover(root)
    _, pool_hash, pool = pool_rows(root)
    inputs = report["inputs"]
    names = ("scores.npy", "manifest.json")
    if (file_digest(cache / names[0]) != report["scores_sha256"]
            or file_digest(cache / names[1]) != report["manifest_sha256"]
            or inputs["pcm_file_sha256"] != pcm_receipt["file_sha256"]
            or inputs["pool_sha256"] != pool_hash or inputs["deadline_ms"] != DEADLINE
            or inputs["student_hop_samples"] != HOP * 4 or inputs["student_window_samples"] != FRAME
            or inputs["ranking_sha256"] != file_digest(root / "reports/own-teacher-language-selection.json")):
        raise ValueError("the precomputed targets, causal PCM and pause membership differ")
    manifest = json.loads((cache / names[1]).read_text())
    if manifest["inputs"] != inputs:
        raise ValueError("teacher representations have different inputs")
    pcm = np.load(pcm_path, mmap_mode="r", allow_pickle=False)
    targets = np.load(cache / names[0], allow_pickle=False)[:, None]
    if pcm.shape != (len(pool), 1, SAMPLES) or targets.shape != (len(pool), 1):
        raise ValueError("unexpected distillation dimensions")
    if not np.all(np.isfinite(targets) & (targets >= 0) & (targets <= 1)):
        raise ValueError("invalid teacher scores")
    sources, roles, groups = (np.asarray([row[key] for _, row in pool])
                              for key in ("source", "role", "connected_group"))
    return cache, manifest["shards"], pcm, targets, sources, roles, groups, file_digest(report_path)


def tail_shards(cache, digests, count):
    expected = {f"{start // SHARD_ROWS:04d}.npz" for start in range(0, count, SHARD_ROWS)}
    if set(digests) != expected:
        raise ValueError("teacher feature shards are incomplete")
    frames = (SAMPLES - FRAME) // (HOP * 4) + 1
    for name, digest in sorted(digests.items()):
        path = cache / name
        if file_digest(path) != digest:
            raise ValueError(f"teacher shard changed: {name}")
        start = int(path.stem) * SHARD_ROWS
        stop = min(start + SHARD_ROWS, count)
        with np.load(path, allow_pickle=False) as archive:
            features, mask = archive["features"], archive["valid"]
        if features.shape[:2] != (stop - start, frames) or mask.shape != features.shape[:2]:
            raise ValueError("teacher features do not match the fixed student frame grid")
        tail = slice(-TAIL_FRAMES, None)
        yield start, stop, features[:, tail].copy(), mask[:, tail].copy()


def targets_for_features(cache, digests, sources, roles, output, *, reuse_pca=False):
    fit = roles == "fit"
    if reuse_pca:
        pca = torch.load(output, map_location=DEVICE, weights_only=True)
        selected = pca["fit_row_ordinals"].cpu().numpy()
        if not fit[selected].all():
            raise ValueError("the archived PCA includes a non-fit row")
        meter = Progress("own-distill-replay-pca", len(roles), "pauses")
        try:
            projected = project_features(cache, digests, len(roles), pca["origin"], pca["basis"], pca["scale"], meter)
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
        return projected
    rng = np.random.default_rng(60217)
    active = np.unique(sources)
    count = min(512, *(np.sum(fit & (sources == source)) for source in active))
    if not count:
        raise ValueError("PCA needs fit examples from each source")
    selected = np.zeros(len(roles), dtype=bool)
    for source in active:
        rows = np.flatnonzero(fit & (sources == source))
        selected[rng.choice(rows, count, replace=False)] = True
    chosen = []
    meter = Progress("own-distill-reduce-features", len(roles) * 2, "pauses")
    try:
        for start, stop, matrix, mask in tail_shards(cache, digests, len(roles)):
            for row in np.flatnonzero(selected[start:stop]):
                eligible = np.flatnonzero(mask[row])
                if not len(eligible):
                    raise ValueError("no real frame remains in a selected deadline tail")
                chosen.append(matrix[row, rng.choice(eligible, 4, replace=len(eligible) < 4)])
            meter.advance(stop - start)
        values = torch.as_tensor(np.concatenate(chosen), dtype=torch.float32, device=DEVICE)
        origin = values.mean(0)
        centered = values - origin
        eigenvalues, vectors = torch.linalg.eigh(centered.T @ centered / len(values))
        basis = vectors[:, -PCA_SIZE:]
        columns = torch.arange(PCA_SIZE, device=DEVICE)
        pivots = basis[basis.abs().argmax(dim=0), columns]
        basis = basis * torch.where(pivots < 0, -1., 1.)
        scale = eigenvalues[-PCA_SIZE:].clamp_min(1e-6).sqrt()
        projected, valid = project_features(cache, digests, len(roles), origin, basis, scale, meter)
        torch.save({"origin": origin.cpu(), "basis": basis.cpu(), "scale": scale.cpu(),
                    "fit_row_ordinals": torch.from_numpy(np.flatnonzero(selected))}, output)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return projected, valid


def project_features(cache, digests, count, origin, basis, scale, meter):
    projected = np.empty((count, TAIL_FRAMES, PCA_SIZE), dtype=np.float32)
    valid = np.empty(projected.shape[:2], dtype=bool)
    for start, stop, matrix, mask in tail_shards(cache, digests, count):
        data = torch.as_tensor(matrix, dtype=torch.float32, device=DEVICE)
        reduced = ((data - origin) @ basis / scale).cpu().numpy()
        reduced[~mask] = 0
        projected[start:stop], valid[start:stop] = reduced, mask
        meter.advance(stop - start)
    if not valid.any(axis=1).all() or not np.isfinite(projected).all():
        raise ValueError("projected targets contain empty or nonfinite rows")
    return projected, valid


def score_features(model, audio, elapsed, tail_frames=TAIL_FRAMES):
    states = model.stem(model.features(audio).transpose(1, 2)).transpose(1, 2)
    sequence, _ = model.sequence(states)
    decision = torch.cat((sequence[:, -1], elapsed / 1000), dim=-1)
    return model.decision(decision).flatten(), sequence[:, -tail_frames:]


def stable_scores(metrics):
    probes = [data["by_deadline"][str(DEADLINE)] for data in metrics.values()]
    return all(data["spearman"] is not None and data["score_spread"] >= .05
               and data["largest_tie"] / sample["pauses"] <= .02
               for data, sample in zip(probes, metrics.values(), strict=True))


def replace_history(audio, prefix, noise):
    positions = torch.arange(audio.shape[-1], device=audio.device)
    return torch.where(positions < prefix[:, None], noise, audio)


def fit_arm(pcm, scores, teacher, valid, sources, groups, fit, fidelity, active, arm, seed, *,
            shuffle_features=False, active_stats=False, score_history=None, parameters=150000, hop=HOP,
            select_final=False, progress_prefix="own-feature"):
    config = {"architecture": "a", "window": 3, "parameters": parameters}
    if hop != HOP:
        config["hop"] = hop
    tail_frames = teacher.shape[1]
    if active_stats:
        config["active_stats"] = True
    torch.manual_seed(seed)
    model = build(config).to(DEVICE)
    projection = torch.nn.Linear(model.sequence.hidden_size, PCA_SIZE, device=DEVICE)
    optimizer = torch.optim.AdamW([*model.parameters(), *projection.parameters()], lr=.001)
    row_weights = np.zeros(len(sources), dtype=np.float32)
    for source in active:
        rows = fit[sources[fit] == source]
        row_weights[rows] = len(fit) / (len(active) * len(rows))
    target_ordinals = np.arange(len(sources))
    if shuffle_features:
        target_ordinals = shuffled_targets(target_ordinals[:, None], sources, groups, fit, seed)[:, 0]
    j = np.zeros(BATCH, dtype=int)
    x, elapsed = batch(pcm, fit[:2], j[:2], DEVICE, DEADLINES)
    with torch.inference_mode():
        torch.testing.assert_close(score_features(model, x, elapsed, tail_frames)[0], model.logits(x, elapsed).flatten(),
                                   rtol=0, atol=0)
    history, winner, best_state = [], None, None
    name = f"{arm}-{seed}" + ("-shuffled-features" if shuffle_features else "")
    if active_stats:
        name += "-active-stats"
    if score_history is not None:
        prefix, noise, history_scores = score_history
        name += "-history"
    if parameters != 150000:
        name += f"-{parameters}"
    if hop != HOP:
        name += f"-hop{hop}"
    meter = Progress(f"{progress_prefix}-{name}", EPOCHS * ((len(fit) + BATCH - 1) // BATCH), "updates")
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = np.random.default_rng(seed_for("dense-teacher-order", seed, epoch)).permutation(fit)
            use_scores = arm != "feature-first" or epoch > 5
            use_features = arm == "joint" or not use_scores
            score_losses, feature_losses = [], []
            feature_positions = np.zeros(tail_frames)
            position_counts = np.zeros(tail_frames)
            for start in range(0, len(order), BATCH):
                rows = order[start:start + BATCH]
                audio, elapsed = batch(pcm, rows, j[:len(rows)], DEVICE, DEADLINES)
                truth = scores[rows, 0]
                if use_scores and score_history is not None:
                    replaces = ((rows + epoch) % 2 == 0) & (prefix[rows] > 0)
                    lengths = torch.as_tensor(np.where(replaces, prefix[rows], 0), device=DEVICE)
                    audio = replace_history(audio, lengths, noise[rows % len(noise)])
                    truth = np.where(replaces, history_scores[rows], truth)
                weight = torch.as_tensor(row_weights[rows], device=DEVICE)
                logits, states = score_features(model, audio, elapsed, tail_frames)
                losses = []
                if use_scores:
                    y = torch.as_tensor(truth, device=DEVICE)
                    score_loss = F.binary_cross_entropy_with_logits(logits, y, weight=weight)
                    losses.append(score_loss)
                    score_losses.append(score_loss.detach())
                if use_features:
                    source_rows = target_ordinals[rows]
                    mask = torch.as_tensor(valid[rows] & valid[source_rows], device=DEVICE)
                    y = torch.as_tensor(teacher[source_rows], device=DEVICE)
                    guess = projection(states)
                    frame_loss = ((guess - y).abs().mean(-1) + 1 - F.cosine_similarity(guess, y, dim=-1)) / 2
                    pause_loss = (frame_loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                    feature_loss = (pause_loss * weight).mean()
                    losses.append(feature_loss * (JOINT_WEIGHT if use_scores else 1))
                    feature_losses.append(feature_loss.detach())
                    feature_positions += (frame_loss.detach() * mask).sum(0).cpu().numpy()
                    position_counts += mask.sum(0).cpu().numpy()
                optimizer.zero_grad(set_to_none=True)
                sum(losses).backward()
                torch.nn.utils.clip_grad_norm_([*model.parameters(), *projection.parameters()], 5.,
                                               error_if_nonfinite=True)
                optimizer.step()
                meter.advance(1)
            stats = evaluate_fidelity(model, pcm, scores, fidelity, sources, active, DEVICE, deadlines=DEADLINES)
            observed = {"epoch": epoch, "sources": stats,
                        "score_loss": float(torch.stack(score_losses).mean()) if score_losses else None,
                        "feature_loss": float(torch.stack(feature_losses).mean()) if feature_losses else None,
                        "feature_loss_by_position": (feature_positions / position_counts.clip(min=1)).tolist()
                        if feature_losses else None}
            if use_scores and stable_scores(stats):
                rank = np.mean([data["by_deadline"][str(DEADLINE)]["spearman"] for data in stats.values()])
                bce = np.mean([data["bce"] for data in stats.values()])
                observed["selection_key"] = [rank, -bce, -epoch]
                choose = epoch == EPOCHS if select_final else winner is None or observed["selection_key"] > winner["selection_key"]
                if epoch > 5 and choose:
                    winner, best_state = observed, deepcopy(model.state_dict())
            history.append(observed)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return config, best_state, {"history": history, "selected": winner,
                                "feature_moved_fraction": float(np.mean(target_ordinals[fit] != fit))}


def compare(root, output, fits, *, peers=None):
    views = {"english": list(pause_rows(english_rows(root))),
             "harper": list(pause_rows(harper_rows(root, native_prefix=True))),
             "calibration": list(calibration_rows(root))}
    cohorts = {name: np.stack([clip for clip, _, _ in rows]).astype(np.float32) for name, rows in views.items()}
    labels = {name: np.asarray([truth for _, truth, _ in rows]) for name, rows in views.items()}
    (_, baseline, baseline_threshold), = load_validated_models(root, [BASELINE])[0]
    predictions = {"direct": {name: student_scores(baseline, clips, device="cpu")
                              for name, clips in cohorts.items()}}
    thresholds = {"direct": baseline_threshold}
    for name, (config, state, report) in fits.items():
        if state is None:
            continue
        if "checkpoint" not in report:
            path = output / f"{name}.pt"
            temporary = path.with_suffix(".part")
            torch.save({"config": config, "state": state}, temporary)
            temporary.replace(path)
            report["checkpoint"] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
        model = build(config).to(DEVICE).eval()
        model.load_state_dict(state)
        predictions[name] = {cohort: student_scores(model, clips) for cohort, clips in cohorts.items()}
        thresholds[name] = _threshold(predictions[name]["calibration"].tolist(), labels["calibration"].tolist())
    result = {}
    for cohort in ("english", "harper"):
        result[cohort] = {}
        for name, arms in predictions.items():
            score, y = arms[cohort], labels[cohort]
            result[cohort][name] = {**rates(score, y), "threshold": thresholds[name],
                                    "fixed": _stats(score.tolist(), y.tolist(), thresholds[name])}
    scores = {name: probabilities["english"] for name, probabilities in predictions.items()}
    if peers is None:
        peers = {f"{arm}-{SEEDS[0]}": f"score-{SEEDS[0]}" for arm in ARMS[1:]
                 if f"{arm}-{SEEDS[0]}" in scores and f"score-{SEEDS[0]}" in scores}
    peers = {arm: reference for arm, reference in peers.items() if arm in scores and reference in scores}
    intervals = paired_intervals(scores, labels["english"], [group for _, _, group in views["english"]],
                                 peers, partial_auc=True) if peers else {}
    return result, intervals


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the experiment requires staged Vast audio")
    cache, digests, pcm, targets, sources, roles, groups, digest = read_inputs(root)
    fit, fidelity = (np.flatnonzero(roles == role) for role in ("fit", "fidelity"))
    active = tuple(source for source in SOURCES if source in sources)
    torch.use_deterministic_algorithms(True)
    output = root / "runs/own-teacher-feature-student" / digest[:12]
    output.mkdir(parents=True, exist_ok=True)
    pca_path = output / "pca.pt"
    teacher, valid = targets_for_features(cache, digests, sources, roles, pca_path)
    fits = {}
    for arm in ARMS:
        for seed in SEEDS:
            fits[f"{arm}-{seed}"] = fit_arm(pcm, targets, teacher, valid, sources, groups, fit, fidelity,
                                           active, arm, seed)
    seed = SEEDS[0]
    eligible = [(fits[f"{arm}-{seed}"][2]["selected"]["selection_key"], arm) for arm in ARMS[1:]
                if fits[f"{arm}-{seed}"][2]["selected"] is not None]
    if eligible:
        _, winner = max(eligible)
        fits[f"{winner}-{seed}-shuffled-features"] = fit_arm(
            pcm, targets, teacher, valid, sources, groups, fit, fidelity, active, winner, seed, shuffle_features=True)
    result, intervals = compare(root, output, fits)
    report = {"target_report_sha256": digest, "code_sha256": file_digest(Path(__file__)),
              "seeds": SEEDS, "fit_rows": len(fit), "fidelity_rows": len(fidelity),
              "decision_ms": DEADLINE, "epochs": EPOCHS, "tail_frames": TAIL_FRAMES,
              "pca_sha256": file_digest(pca_path), "joint_weight": JOINT_WEIGHT,
              "fits": {name: row[2] for name, row in fits.items()}, "metrics": result, "paired_english": intervals}
    _write_json(root / "reports/own-teacher-feature-student.json", report)
    return report


def multilingual(root):
    report_path = root / "reports/own-teacher-feature-student.json"
    report = json.loads(report_path.read_text())
    models, thresholds = {}, {}
    for arm in ARMS:
        name = f"{arm}-{SEEDS[0]}"
        selected = report["fits"][name]["selected"]
        if selected is None:
            continue
        artifact = report["fits"][name]["checkpoint"]
        path = root / artifact["path"]
        if file_digest(path) != artifact["sha256"]:
            raise ValueError(f"fidelity-locked checkpoint changed: {name}")
        payload = torch.load(path, map_location=DEVICE, weights_only=True)
        model = build(payload["config"]).to(DEVICE).eval()
        model.load_state_dict(payload["state"])
        models[name] = model
        thresholds[name] = report["metrics"]["english"][name]["threshold"]
    if not models:
        raise ValueError("no primary-seed checkpoint cleared the fidelity gate")
    rows = {}
    results = score_languages(root, models, DEVICE, fixed_thresholds=thresholds,
                              paired_rows=rows, progress_name="own-feature-multilingual")
    output = root / "runs/own-teacher-feature-student" / report["target_report_sha256"][:12] / "language-scores.json"
    _write_json(output, rows)
    result = {"scope": "locked primary-seed development comparison",
              "fit_report_sha256": file_digest(report_path), "dataset_sha256": DATA,
              "fixed_ami_thresholds": thresholds, "languages": results,
              "paired_scores_sha256": file_digest(output)}
    _write_json(root / "reports/own-teacher-feature-languages.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multilingual", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = multilingual(data_root()) if args.multilingual else run(data_root())
    print(json.dumps(report["languages"] if args.multilingual else report["metrics"], indent=2))
