"""Compare real and shuffled teacher targets in two Superturn GRUs."""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .audio import RATE, read_audio
from .dense_teacher import DEADLINES, SAMPLES, SOURCES, file_digest
from .eot_domain_probe import DATASET_SHA256, DECISION_MS, GAP, decision_audio, summarize
from .model import build
from .progress import Progress
from .shards import seed_for
from .sources import data_root

SEEDS = (17, 29)
SIZES = (150_000, 1_000_000)
EPOCHS = 10
BATCH = 128
CORPORA = {
    "all": SOURCES,
    "emilia-only": ("emilia_yodas",),
    "without-emilia": ("ami", "voxpopuli"),
}


def load_pool(root, folder):
    report_path = root / "reports" / "dense-teacher-targets.json"
    report = json.loads(report_path.read_text())
    if report["preview"] or report["pool_sha256"] != file_digest(folder / "pool.json"):
        raise ValueError("training requires the complete inventoried target pool")
    count = report["pauses"]
    pcm = np.empty((count, len(DEADLINES), SAMPLES), dtype="<i2")
    targets = np.empty((count, len(DEADLINES)), dtype=np.float32)
    sources, roles, groups, speakers, recordings = [], [], [], [], []
    start = 0
    for name, digest in sorted(report["shard_sha256"].items()):
        path = folder / "scored" / name
        if file_digest(path) != digest:
            raise ValueError(f"teacher target shard changed: {name}")
        rows = pq.read_table(path).to_pylist()
        stop = start + len(rows)
        for j, deadline in enumerate(DEADLINES):
            pcm[start:stop, j, :] = np.stack([np.frombuffer(row[f"pcm_{deadline}"], "<i2") for row in rows])
            targets[start:stop, j] = [row[f"target_{deadline}"] for row in rows]
        sources.extend(row["source"] for row in rows)
        roles.extend(row["role"] for row in rows)
        groups.extend(row["connected_group"] for row in rows)
        speakers.extend(row["speaker"] for row in rows)
        recordings.extend(row["recording"] for row in rows)
        start = stop
    if start != count or not np.all((0 <= targets) & (targets <= 1)):
        raise ValueError("teacher targets have missing or invalid values")
    if set(sources) != set(SOURCES) or set(roles) != {"fit", "fidelity"}:
        raise ValueError("unexpected source or role in scored target pool")
    for kind, identities in (("group", groups), ("speaker", speakers), ("recording", recordings)):
        scopes = defaultdict(set)
        for source, identity, role in zip(sources, identities, roles, strict=True):
            scopes[source, identity].add(role)
        if any(len(values) != 1 for values in scopes.values()):
            raise ValueError(f"{kind} crossed fit and fidelity roles")
    return pcm, targets, np.asarray(sources), np.asarray(roles), np.asarray(groups), file_digest(report_path)


def shuffled_targets(targets, sources, groups, fit, seed):
    """Shuffle deadlines separately within the same source and connected group."""
    strata = defaultdict(list)
    for i in fit:
        strata[sources[i], groups[i]].append(i)
    result = targets.copy()
    rng = np.random.default_rng(seed_for("dense-teacher-shuffle-v1", seed))
    for rows in strata.values():
        for j in range(len(DEADLINES)):
            result[rows, j] = targets[rng.permutation(rows), j]
    return result


def batch(pcm, i, j, device):
    audio = torch.as_tensor(pcm[i, j], dtype=torch.float32, device=device) / 32768
    elapsed = torch.as_tensor(np.asarray(DEADLINES)[j, None], dtype=torch.float32, device=device)
    return audio, elapsed


def evaluate_fidelity(model, pcm, targets, fidelity, sources, active_sources, device):
    results = {}
    model.eval()
    with torch.inference_mode():
        for source in active_sources:
            rows = fidelity[sources[fidelity] == source]
            if not len(rows):
                raise ValueError(f"{source} is missing from the fidelity pool")
            i = np.repeat(rows, len(DEADLINES))
            j = np.tile(np.arange(len(DEADLINES)), len(rows))
            predictions, states, losses = [], [], []
            for start in range(0, len(i), BATCH):
                sl = slice(start, start + BATCH)
                x, elapsed = batch(pcm, i[sl], j[sl], device)
                features = model.decision_input(x, elapsed)
                logits = model.decision(features).flatten()
                y = torch.as_tensor(targets[i[sl], j[sl]], device=device)
                losses.append(F.binary_cross_entropy_with_logits(logits, y, reduction="none").cpu().numpy())
                predictions.append(logits.sigmoid().cpu().numpy())
                states.append(features[:, :-1].cpu().numpy())
            scores = np.concatenate(predictions)
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite student scores")
            by_deadline = {}
            for index, deadline in enumerate(DEADLINES):
                s, t = scores[index::len(DEADLINES)], targets[rows, index]
                rank_s, rank_t = rankdata(s), rankdata(t)
                lower, upper = np.quantile(s, [0.01, 0.99])
                by_deadline[str(deadline)] = {
                    "spearman": float(np.corrcoef(rank_s, rank_t)[0, 1])
                    if np.ptp(rank_s) and np.ptp(rank_t) else None,
                    "score_spread": float(upper - lower),
                    "largest_tie": int(np.unique(s, return_counts=True)[1].max()),
                }
            results[source] = {"pauses": len(rows), "bce": float(np.concatenate(losses).mean()),
                               "state_std": float(np.concatenate(states).std(axis=0).mean()),
                               "by_deadline": by_deadline}
    return results


def fit_model(output, config, pcm, targets, sources, roles, groups, cache_hash, device):
    source = Path(__file__)
    params = {"config": config, "target_report_sha256": cache_hash, "epochs": EPOCHS, "batch_size": BATCH,
              "code_sha256": {name: file_digest(source.with_name(name))
                              for name in (source.name, "model.py", "shards.py")},
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "device": torch.cuda.get_device_name(device) if device == "cuda" else device}
    checkpoint, report_path = output / "best.pt", output / "fit.json"
    if checkpoint.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if report["inputs"] != params or report["checkpoint_sha256"] != file_digest(checkpoint):
            raise ValueError(f"saved diagnostic fit differs from its provenance: {output.name}")
        return checkpoint, report
    output.mkdir(parents=True, exist_ok=True)
    fit = np.flatnonzero(roles == "fit")
    fidelity = np.flatnonzero(roles == "fidelity")
    active_sources = tuple(source for source in SOURCES if source in sources)
    weights = np.zeros(len(roles), dtype=np.float32)
    for source in active_sources:
        indices = fit[sources[fit] == source]
        if not len(indices):
            raise ValueError(f"{source} is missing from the fit pool")
        weights[indices] = len(fit) / (len(active_sources) * len(indices))
    seed = config["seed"]
    y = shuffled_targets(targets, sources, groups, fit, seed) if config["shuffled"] else targets
    torch.manual_seed(seed)
    model = build(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    fit_i = np.repeat(fit, len(DEADLINES))
    fit_j = np.tile(np.arange(len(DEADLINES)), len(fit))
    steps = (len(fit_i) + BATCH - 1) // BATCH
    meter = Progress(f"dense-teacher-{output.name}", EPOCHS * steps, "steps")
    best_loss, best, history = float("inf"), None, []
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = np.random.default_rng(seed_for("dense-teacher-order", seed, epoch)).permutation(len(fit_i))
            for start in range(0, len(order), BATCH):
                ids = order[start:start + BATCH]
                i, j = fit_i[ids], fit_j[ids]
                audio, elapsed = batch(pcm, i, j, device)
                truth = torch.as_tensor(y[i, j], device=device)
                weight = torch.as_tensor(weights[i], device=device)
                logits = model.logits(audio, elapsed).flatten()
                loss = F.binary_cross_entropy_with_logits(logits, truth, weight=weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                meter.advance(1)
            stats = evaluate_fidelity(model, pcm, targets, fidelity, sources, active_sources, device)
            loss = float(np.mean([value["bce"] for value in stats.values()]))
            history.append({"epoch": epoch, "macro_bce": loss, "sources": stats})
            if loss < best_loss:
                best_loss = loss
                best = (epoch, deepcopy(model.state_dict()))
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    epoch, state = best
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"config": config, "state": state}, temporary)
    temporary.replace(checkpoint)
    report = {"inputs": params, "parameters": sum(p.numel() for p in model.parameters()),
              "epochs": history, "selected": history[epoch - 1],
              "checkpoint_sha256": file_digest(checkpoint), "moved_targets": {
                  source: float(np.mean(y[fit[sources[fit] == source]] != targets[fit[sources[fit] == source]]))
                  for source in active_sources}}
    _write_json(report_path, report)
    return checkpoint, report


def external(root, fits, device):
    path = root / "raw" / "livekit_eot_diagnostic" / "data/en/validation-00000-of-00002.parquet"
    if file_digest(path) != DATASET_SHA256:
        raise ValueError("external diagnostic differs from the matched teacher comparison")
    models = {}
    for name, (checkpoint, _) in fits.items():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model = build(saved["config"]).to(device).eval()
        model.load_state_dict(saved["state"])
        models[name] = model
    truth, scores = [], defaultdict(list)
    source = pq.ParquetFile(path)
    meter = Progress("dense-teacher-external", source.metadata.num_rows, "turns")
    try:
        for rows in source.iter_batches(batch_size=8, columns=["audio", "cohort", "language", "silence_spans"]):
            clips = []
            for row in rows.to_pylist():
                if row["cohort"] != "general" or row["language"] != "en":
                    raise ValueError("external diagnostic contains an unexpected cohort")
                pcm = read_audio(row["audio"]["bytes"])
                for span in row["silence_spans"]:
                    edge = round(span["start"] * RATE)
                    end = min(len(pcm), round(span["end"] * RATE))
                    if not 0 <= edge <= end or end - edge < GAP:
                        continue
                    if span["label"] not in {"eot", "hold"}:
                        raise ValueError("external diagnostic contains an unexpected label")
                    truth.append(int(span["label"] == "eot"))
                    clips.append(decision_audio(pcm, edge, "telephone")[-SAMPLES:])
            if clips:
                x = torch.as_tensor(np.stack(clips), device=device)
                elapsed = torch.full((len(clips), 1), float(DECISION_MS), device=device)
                with torch.inference_mode():
                    for name, model in models.items():
                        scores[name].extend(model(x, elapsed).flatten().cpu().tolist())
            meter.advance(rows.num_rows)
        if (len(truth), sum(truth)) != (932, 400):
            raise ValueError("external diagnostic no longer matches the matched teacher counts")
        results = {name: summarize(values, truth, [0.5], roc_limits=True) for name, values in scores.items()}
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return results


def run(root, device, corpus):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("dense teacher diagnostics use the existing training server")
    folder = root / "shards" / "dense-teacher"
    suffix = "" if corpus == "all" else f"-{corpus}"
    output = root / "runs" / f"dense-teacher-diagnostic{suffix}"
    pcm, targets, sources, roles, groups, fingerprint = load_pool(root, folder)
    selected = CORPORA[corpus]
    if corpus != "all":
        eligible = np.isin(sources, selected)
        pcm, targets, sources, roles, groups = (values[eligible]
                                              for values in (pcm, targets, sources, roles, groups))
    coverage = {source: {role: int(np.count_nonzero((sources == source) & (roles == role)))
                         for role in ("fit", "fidelity")} for source in selected}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    fits = {}
    for size in (SIZES if corpus == "all" else SIZES[:1]):
        for seed in SEEDS:
            for shuffle in (False, True):
                name = f"{size}-{seed}-{'shuffle' if shuffle else 'teacher'}"
                config = {"architecture": "a", "window": SAMPLES / RATE, "parameters": size,
                          "seed": seed, "shuffled": shuffle}
                fits[name] = fit_model(output / name, config, pcm, targets, sources, roles, groups, fingerprint, device)
    del pcm, targets, sources, roles, groups
    metrics = external(root, fits, device)
    report = {"use": "diagnostic", "dataset_sha256": DATASET_SHA256, "decision_ms": DECISION_MS,
              "epochs": EPOCHS, "batch_size": BATCH, "corpus": corpus, "coverage": coverage,
              "selector": "macro-source teacher fidelity BCE",
              "external_half_probability_threshold": "0.5 for description only",
              "fit": {name: result for name, (_, result) in fits.items()}, "external": metrics}
    _write_json(root / "reports" / f"dense-teacher-gru{suffix}.json", report)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=CORPORA, default="all")
    args = parser.parse_args()
    print(json.dumps(run(data_root(), "cuda", args.corpus), indent=2))
