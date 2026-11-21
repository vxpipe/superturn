"""Average three archived-draw continuations from one fixed student parent."""

import json
from hashlib import sha256

import numpy as np
from scipy.special import logit
import torch

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .language_failure_audit import read_receipt, read_verified
from .own_head_distillation import pool_rows
from .own_student_tail_ranking import fit, pair_candidates, schedule
from .own_teacher_feature_student import SEEDS, read_inputs
from .sources import data_root


def average_states(models):
    first = models[0]
    result = {}
    if any(model["config"] != first["config"] or list(model["state"]) != list(first["state"]) for model in models):
        raise ValueError("sibling architecture or state keys differ")
    for key, reference in first["state"].items():
        items = [model["state"][key] for model in models]
        if any((item.dtype, item.shape) != (reference.dtype, reference.shape) for item in items):
            raise ValueError(f"mismatched sibling state: {key}")
        if reference.is_floating_point():
            value = torch.stack([item.double() for item in items]).mean(0).to(reference.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite averaged state: {key}")
            result[key] = value
        elif all(torch.equal(item, reference) for item in items):
            result[key] = reference
        else:
            raise ValueError(f"non-floating states differ: {key}")
    return {"config": first["config"], "state": result}


def save(root, target, payload):
    temporary = target.with_suffix(".part")
    torch.save(payload, temporary)
    temporary.replace(target)
    return {"path": target.relative_to(root).as_posix(), "sha256": file_digest(target)}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the sibling continuation needs staged Vast audio")
    ranking, ranking_sha = read_receipt(root, "own-student-tail-ranking")
    _, _, pcm, probabilities, _, roles, _, receipt = read_inputs(root)
    if ranking["target_sha256"] != receipt:
        raise ValueError("the previous replay used a different fit pool")
    _, _, pool = pool_rows(root)
    rows = [row for _, row in pool]
    probabilities = probabilities[:, 0]
    teacher_logits = logit(probabilities.clip(1e-5, 1 - 1e-5))
    mask = roles == "fit"
    cells, _ = pair_candidates(rows, teacher_logits, mask)
    directory = root / "runs/own-student-weight-soup" / ranking_sha[:12]
    directory.mkdir(parents=True, exist_ok=True)

    seed_zero = SEEDS[0]
    parent_ref = ranking["plans"][str(seed_zero)]["parent"]
    parent = torch.load(read_verified(root, parent_ref), map_location="cpu", weights_only=True)
    torch.use_deterministic_algorithms(True)
    siblings, models = {}, []
    for seed in SEEDS:
        order, _ = schedule(cells, rows, np.flatnonzero(mask), seed)
        schedule_sha = sha256(order.tobytes()).hexdigest()
        if schedule_sha != ranking["plans"][str(seed)]["schedule_sha256"]:
            raise ValueError("a sibling draw schedule differs from the recorded ranking comparison")
        if seed == seed_zero:
            checkpoint = ranking["fits"][f"pointwise-{seed}"]["checkpoint"]
            payload = torch.load(read_verified(root, checkpoint), map_location="cpu", weights_only=True)
        else:
            config, state, _ = fit(parent, pcm, probabilities, teacher_logits, order, seed, "pointwise")
            payload = {"config": config, "state": state}
            checkpoint = save(root, directory / f"sibling-{seed}.pt", payload)
        siblings[str(seed)] = {"checkpoint": checkpoint, "draws_sha256": schedule_sha}
        models.append(payload)
    soup = save(root, directory / "soup.pt", average_states(models))
    report = {"method": "equal-weight parameter mean of pointwise siblings from the predeclared first width seed",
              "ranking_sha256": ranking_sha, "fit_target_sha256": receipt, "common_parent_seed": seed_zero,
              "common_parent": parent_ref, "siblings": siblings, "soup": soup}
    _write_json(root / "reports/own-student-weight-soup.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
