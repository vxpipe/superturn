"""Test upper-tail score ordering against the same-draw pointwise student."""

from collections import Counter, defaultdict
from hashlib import sha256
import json

import numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from .ami_calibration import _write_json
from .dense_teacher import SAMPLES, file_digest
from .eot_domain_probe import GAP
from .language_failure_audit import read_receipt, read_verified
from .model import HOP, build
from .multilingual_teacher_probe import score_languages
from .own_head_distillation import DEADLINE, pool_rows, student_scores
from .own_student_ami_supervision import BATCH, LR, STEPS, summarize
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, SEEDS, compare, read_inputs
from .own_telephone_distillation import fixed_rates
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_rows
from .wav2vec_harper_probe import pause_rows

MIN_CELL = 50
MIN_DELTA = .5
PAIRS = 8
WEIGHT = .20
ARMS = ("pointwise", "pairwise")
CONTRASTS = (("pairwise", "pointwise"), ("pairwise", "parent"))
HISTORY_BANDS = (19200, SAMPLES - GAP)


def cell_groups(rows, mask):
    groups = defaultdict(list)
    for i in np.flatnonzero(mask):
        row = rows[i]
        band = int(np.searchsorted(HISTORY_BANDS, row["pause_start"], side="right"))
        groups[row["source"], row["language"], band].append(int(i))
    return groups


def pair_candidates(rows, teacher_logits, mask):
    cells, coverage = {}, {}
    for cell, members in sorted(cell_groups(rows, mask).items()):
        key = "/".join(map(str, cell))
        all_rows = len(members)
        members = [i for i in members if rows[i]["speaker"]]
        coverage[key] = {"fit_rows": all_rows, "with_speaker": len(members)}
        if len(members) < MIN_CELL:
            coverage[key]["status"] = "small_cell"
            continue
        ranked = np.asarray(sorted(members, key=lambda i: (-teacher_logits[i], i)))
        upper_cutoff = teacher_logits[ranked[(len(ranked) + 4) // 5 - 1]]
        partner_cutoff = teacher_logits[ranked[(2 * len(ranked) + 4) // 5 - 1]]
        upper = ranked[teacher_logits[ranked] >= upper_cutoff]
        partners = ranked[teacher_logits[ranked] >= partner_cutoff]
        voices = np.asarray([rows[i]["speaker"] for i in partners])
        valid, cross_voice, eligible = [], 0, 0
        for high in upper:
            different = voices != rows[high]["speaker"]
            delta = teacher_logits[high] - teacher_logits[partners]
            possible = partners[different & (delta >= MIN_DELTA)]
            cross_voice += int(different.sum())
            eligible += len(possible)
            if len(possible):
                valid.append((int(high), possible))
        coverage[key].update({"cross_voice_comparisons": cross_voice, "eligible_pairs": eligible,
                              "delta_rejections": cross_voice - eligible,
                              "upper_rows_with_ties": len(upper), "partner_rows_with_ties": len(partners),
                              "eligible_high_rows": len(valid), "status": "eligible" if valid else "no_pair"})
        if valid:
            cells[cell] = valid
    return cells, coverage


def schedule(cells, rows, fit, seed):
    rng = np.random.default_rng(seed_for("own-tail-rank-v1", seed))
    sources = sorted({rows[i]["source"] for i in fit})
    groups = [np.asarray([i for i in fit if rows[i]["source"] == source]) for source in sources]
    available = {source: [key for key in cells if key[0] == source] for source in sources}
    active = [source for source in sources if available[source]]
    if not active or PAIRS % len(active) or (BATCH - 2 * PAIRS) % len(groups):
        raise ValueError("cannot build the matched source-balanced continuation")
    order, draw_counts = [], Counter()
    for _ in range(STEPS):
        replay = rng.permutation(np.concatenate([
            rng.choice(group, (BATCH - 2 * PAIRS) // len(groups)) for group in groups
        ]))
        chosen = []
        for source in active:
            tags = available[source]
            for _ in range(PAIRS // len(active)):
                cell = tags[rng.integers(len(tags))]
                pairs = cells[cell]
                high, lower = pairs[rng.integers(len(pairs))]
                chosen.extend((high, int(rng.choice(lower))))
                draw_counts["/".join(map(str, cell))] += 1
        order.append(np.r_[replay, chosen])
    return np.asarray(order, dtype="<i8"), draw_counts


def fit(parent, pcm, probabilities, teacher_logits, order, seed, arm):
    torch.manual_seed(seed)
    model = build(parent["config"]).to(DEVICE)
    if model.features.hop != HOP or sum(p.numel() for p in model.parameters()) != 972795:
        raise ValueError("the control is not the archived 973k 10 ms student")
    model.load_state_dict(parent["state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    elapsed = torch.full((BATCH, 1), float(DEADLINE), device=DEVICE)
    values = []
    meter = Progress(f"own-tail-rank-{arm}-{seed}", len(order), "updates")
    try:
        for indices in order:
            clips = torch.as_tensor(pcm[indices, 0].astype(np.float32) / 32768, device=DEVICE)
            targets = torch.as_tensor(probabilities[indices], dtype=torch.float32, device=DEVICE)
            teacher = teacher_logits[indices[-2 * PAIRS:]].reshape(PAIRS, 2)
            pairs = torch.as_tensor(expit(teacher[:, 0] - teacher[:, 1]), dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(clips, elapsed).flatten()
            point = F.binary_cross_entropy_with_logits(logits, targets)
            tail = logits[-2 * PAIRS:].reshape(PAIRS, 2)
            rank = F.binary_cross_entropy_with_logits(tail[:, 0] - tail[:, 1], pairs)
            loss = point if arm == "pointwise" else (1 - WEIGHT) * point + WEIGHT * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            values.append([float(item.detach()) for item in (point, rank, loss)])
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return parent["config"], {key: value.detach().cpu() for key, value in model.state_dict().items()}, {
        "mean_point_rank_total_loss": np.mean(values, axis=0).tolist()}


def fidelity(model, pcm, rows, probabilities, mask, name):
    groups = {cell: members for cell, members in cell_groups(rows, mask).items() if len(members) >= 30}
    indices = sorted(i for members in groups.values() for i in members)
    if not indices:
        raise ValueError("no eligible held fidelity cell")
    meter = Progress(f"own-tail-rank-fidelity-{name}", len(indices), "pauses")
    predictions = {}
    try:
        for part in np.array_split(indices, (len(indices) + 255) // 256):
            values = student_scores(model, pcm[part, 0].astype(np.float32) / 32768)
            predictions.update(zip(part, values, strict=True))
            meter.advance(len(part))
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result = {}
    for cell, members in sorted(groups.items()):
        teacher = probabilities[members]
        student = np.asarray([predictions[i] for i in members])
        cutoff = np.quantile(teacher, .8)
        top = teacher >= cutoff
        count = int(top.sum())
        predicted_cutoff = np.partition(student, len(student) - count)[len(student) - count]
        above, boundary = student > predicted_cutoff, student == predicted_cutoff
        credit = (count - int(above.sum())) / int(boundary.sum())
        informative = 1 < count <= len(student) / 2 and np.ptp(teacher[top]) > 0
        rho = spearmanr(teacher[top], student[top]).statistic if informative and np.ptp(student[top]) else None
        overlap = ((top & above).sum() + credit * (top & boundary).sum()) / count if informative else None
        result["/".join(map(str, cell))] = {"rows": len(members), "top_rows": count,
                                           "top_spearman": float(rho) if rho is not None else None,
                                           "top_overlap": float(overlap) if overlap is not None else None,
                                           "teacher_boundary_ties": int((teacher == cutoff).sum()),
                                           "student_boundary_ties": int(boundary.sum())}
    return result


def gate(summary, metrics, outcome):
    pointwise, pairwise, parent = np.asarray(summary["macro"])
    low_budget, low_5, _ = np.asarray(summary["paired_macro_95"])[0, :, 0]
    gains = np.asarray([np.asarray(rows)[1, 1] - rows[0][1] for rows in summary["by_language"].values()])
    cohort_deltas = {}
    for cohort in ("english", "harper"):
        scores = metrics[cohort]
        cohort_deltas[cohort] = float(np.mean([
            scores[f"pairwise-{seed}"]["recalls_at_2_5_5_7_5pct"][1]
            - scores[f"pointwise-{seed}"]["recalls_at_2_5_5_7_5pct"][1]
            for seed in SEEDS
        ]))
    fidelity_delta = []
    for seed in SEEDS:
        control = outcome[f"pointwise-{seed}"]
        for cell, value in outcome[f"pairwise-{seed}"].items():
            other = control[cell]["top_spearman"]
            if value["top_spearman"] is not None and other is not None:
                fidelity_delta.append(value["top_spearman"] - other)
    mean_fidelity_delta = float(np.mean(fidelity_delta)) if fidelity_delta else None
    checks = {"macro_matched_5_plus_3_points_and_lb_positive": bool(pairwise[1] - pointwise[1] >= .03 and low_5 > 0),
              "macro_average_plus_1_5_points_and_lb_positive": bool(pairwise[0] - pointwise[0] >= .015 and low_budget > 0),
              "matched_5_at_least_parent": bool(pairwise[1] >= parent[1]),
              "normalized_pauc_10pct_point_loss_at_most_0_005": bool(pairwise[2] - pointwise[2] >= -.005),
              "no_language_point_loss_over_3_points": bool(gains.min() >= -.03),
              "english_and_harper_nonnegative_point_change": all(value >= 0 for value in cohort_deltas.values()),
              "held_top_region_spearman_improves": mean_fidelity_delta is not None and mean_fidelity_delta > 0}
    return {"checks": checks, "passes_development_screen": all(checks.values()),
            "english_harper_matched_5_deltas": cohort_deltas,
            "matched_5_minimum_language_delta": float(gains.min()),
            "held_top_spearman_mean_delta": mean_fidelity_delta}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("upper-tail distillation needs staged Vast data")
    width, width_sha = read_receipt(root, "own-student-width-probe")
    _, _, pcm, probabilities, _, roles, _, target_sha = read_inputs(root)
    _, _, pool = pool_rows(root)
    if target_sha != width["target_report_sha256"]:
        raise ValueError("the 973k parent uses a different training pool")
    rows = [row for _, row in pool]
    probabilities = probabilities[:, 0]
    teacher_logits = logit(probabilities.clip(1e-5, 1 - 1e-5))
    fit_indices = np.flatnonzero(roles == "fit")
    cells, coverage = pair_candidates(rows, teacher_logits, roles == "fit")
    torch.use_deterministic_algorithms(True)
    outcome, fits, planned = {}, {}, {}
    output = root / "runs/own-student-tail-ranking" / target_sha[:12]
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        parent_receipt = width["fits"][model_name(1_000_000, seed)]["checkpoint"]
        parent = torch.load(read_verified(root, parent_receipt), map_location="cpu", weights_only=True)
        model = build(parent["config"]).to(DEVICE).eval()
        model.load_state_dict(parent["state"])
        parent_name = f"parent-{seed}"
        outcome[parent_name] = fidelity(model, pcm, rows, probabilities, roles == "fidelity", parent_name)
        fits[parent_name] = parent["config"], parent["state"], {"checkpoint": parent_receipt}
        order, counts = schedule(cells, rows, fit_indices, seed)
        selected = order[:, -2 * PAIRS:].reshape(-1, 2)
        voices, repeats = [], []
        for side in selected.T:
            people = Counter((rows[i]["source"], rows[i]["speaker"]) for i in side)
            voices.append(len(people))
            repeats.append(max(people.values()))
        planned[seed] = {"parent": parent_receipt, "schedule_sha256": sha256(order.tobytes()).hexdigest(),
                         "pair_draws_by_cell": counts, "distinct_rows": len(np.unique(order)),
                         "distinct_high_and_lower_rows": [len(np.unique(side)) for side in selected.T],
                         "distinct_high_and_lower_voices": voices, "maximum_draws_from_any_voice": repeats}
        for arm in ARMS:
            key = f"{arm}-{seed}"
            fits[key] = fit(parent, pcm, probabilities, teacher_logits, order, seed, arm)
    metrics, paired = compare(root, output, fits, peers={f"pairwise-{seed}": f"pointwise-{seed}" for seed in SEEDS})
    models = {}
    for name, (config, state, _) in fits.items():
        models[name] = build(config).to(DEVICE).eval()
        models[name].load_state_dict(state)
        if name not in outcome:
            outcome[name] = fidelity(models[name], pcm, rows, probabilities, roles == "fidelity", name)
    labels = {}
    score_languages(root, models, DEVICE, paired_rows=labels, progress_name="own-tail-rank-multilingual")
    english = list(pause_rows(english_rows(root)))
    clips, truth, grouping = zip(*english, strict=True)
    audio = np.stack(clips)
    labels["en"] = {"labels": truth, "turn_ordinals": grouping,
                    "scores": {name: student_scores(model, audio).tolist() for name, model in models.items()}}
    summary = summarize(labels, arm_names=(*ARMS, "parent"), contrasts=CONTRASTS)
    covered = {rows[i]["language"] for i in fit_indices}
    transfer = {}
    for name, languages in (("covered", covered), ("no_fit_rows", set(labels) - covered)):
        members = sorted(set(labels) & languages)
        transfer[name] = {"languages": members,
                          "macro": np.mean([summary["by_language"][language] for language in members], axis=0).tolist()}
    path = output / "paired-language-scores.json"
    _write_json(path, labels)
    report = {"scope": "development only. Own-teacher targets and matched audio draws.",
              "target_sha256": target_sha, "width_sha256": width_sha, "pair_weight": WEIGHT, "pairs_per_batch": PAIRS,
              "updates": STEPS, "batch": BATCH, "lr": LR, "teacher_logit_min_delta": MIN_DELTA,
              "cells": coverage, "pair_sources": sorted({cell[0] for cell in cells}),
              "pair_languages": sorted({cell[1] for cell in cells}), "plans": planned,
              "fits": {key: row[2] for key, row in fits.items()},
              "summary": summary, "transfer": transfer, "fidelity": outcome, "paired_english": paired,
              "gate": gate(summary, metrics, outcome),
              "english_harper_and_fixed_thresholds": metrics, "fixed_ami_counts": fixed_rates(labels, metrics),
              "paired_language_scores": {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}}
    _write_json(root / "reports/own-student-tail-ranking.json", report)
    print(json.dumps({"macro": summary["macro"], "paired_macro_95": summary["paired_macro_95"], "gate": report["gate"],
                      "transfer": transfer}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
