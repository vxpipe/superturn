"""Compare recurrent depth at matched parameters and fixed student history."""

import json
from statistics import mean

import torch

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import SAMPLES, SOURCES, file_digest
from .dense_teacher_train import SEEDS, external, fit_model, load_pool
from .eot_domain_probe import DECISION_MS, summarize
from .harper_valley_benchmark import pinned_validation
from .model import build
from .multilingual_teacher_probe import DATA, score_languages
from .real_label_pilot import harper_development, predict
from .sources import data_root

DESIGNS = {2: (104, 144975), 3: (83, 143568), 4: (72, 145327)}
STEM_WIDTH = 104
TEACHER_SECONDS = 8


def config_for(layers, seed):
    return {"architecture": "a", "window": SAMPLES / RATE, "width": STEM_WIDTH, "layers": layers,
            "gru_hidden_size": DESIGNS[layers][0], "seed": seed, "shuffled": False}


def verify_control():
    original = build({"architecture": "a", "window": 3, "width": STEM_WIDTH}).eval()
    explicit = build(config_for(2, 17)).eval()
    explicit.load_state_dict(original.state_dict(), strict=True)
    audio = torch.randn(2, SAMPLES, generator=torch.Generator().manual_seed(17))
    delay = torch.full((2, 1), float(DECISION_MS))
    with torch.inference_mode():
        torch.testing.assert_close(original(audio, delay), explicit(audio, delay), atol=0, rtol=0)
    for layers, (_, expected) in DESIGNS.items():
        with torch.device("meta"):
            model = build(config_for(layers, 17))
        if sum(p.numel() for p in model.parameters()) != expected:
            raise ValueError(f"{layers}-layer parameter count does not match the planned comparison")


def get_recall(metrics):
    return metrics["retrospective_roc_limits"]["max_recall_at_5pct_false_complete"]


def compare(language_metrics, records):
    macro = {name: mean(get_recall(r["models"][name]) for r in language_metrics.values()) for name in records}
    comparisons = {}
    for depth in DESIGNS:
        names = [f"depth-{depth}-{seed}" for seed in SEEDS]
        delta = [macro[name] - macro[f"depth-2-{seed}"] for name, seed in zip(names, SEEDS, strict=True)]
        per_language = {
            language: mean(get_recall(r["models"][name]) - get_recall(r["models"][f"depth-2-{seed}"])
                           for name, seed in zip(names, SEEDS, strict=True))
            for language, r in language_metrics.items()}
        collapsed = {name: [source for source, item in records[name]["selected"]["sources"].items()
                            if item["by_deadline"][str(DECISION_MS)]["score_spread"] < 0.05]
                     for name in names}
        comparisons[str(depth)] = {
            "macro_recall": [macro[name] for name in names],
            "paired_macro_recall_delta": delta, "per_language_mean_delta": per_language,
            "collapsed_fidelity_sources": collapsed,
            "advance_exploratory": depth != 2 and min(delta) > 0 and mean(delta) >= 0.03
            and min(per_language.values()) >= -0.05 and not any(collapsed.values()),
        }
    return comparisons


def run(root, device):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the matched GRU depth study uses existing data on Vast")
    verify_control()
    folder = root / "shards" / "dense-teacher"
    pool = load_pool(root, folder, teacher_seconds=TEACHER_SECONDS)
    inventory = json.loads((folder / "pool.json").read_text())
    history = {
        source: {role: sum(row["pause_start"] + DECISION_MS * RATE // 1000 >= RATE * TEACHER_SECONDS
                           for row in inventory if (row["source"], row["role"]) == (source, role))
                 for role in ("fit", "fidelity")} for source in SOURCES}
    fits = {}
    for depth, (_, expected) in DESIGNS.items():
        for seed in SEEDS:
            name = f"depth-{depth}-{seed}"
            path = root / "runs" / "gru-depth-study" / name
            fits[name] = fit_model(path, config_for(depth, seed), *pool, device)
            if fits[name][1]["parameters"] != expected:
                raise ValueError("depth fit changed the matched parameter budget")
    del pool, inventory
    models = {}
    for name, (checkpoint, info) in fits.items():
        if file_digest(checkpoint) != info["checkpoint_sha256"]:
            raise ValueError("depth checkpoint changed before evaluation")
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        models[name] = build(saved["config"]).to(device).eval()
        models[name].load_state_dict(saved["state"], strict=True)
    multilingual = score_languages(root, models, device)
    english = external(root, fits, device)
    cases, _, _ = pinned_validation(root)
    harper, archived = harper_development(root, cases)
    harper_scores = {
        condition: {name: summarize(predict(model, clips, device), y.tolist(), [0.5], roc_limits=True)
                    for name, model in models.items()}
        for condition, (clips, y) in harper.items()}
    records = {name: details for name, (_, details) in fits.items()}
    result = {
        "use": "diagnostic", "teacher_seconds": TEACHER_SECONDS, "student_seconds": SAMPLES / RATE,
        "full_recorded_teacher_history": history, "decision_ms": DECISION_MS,
        "multilingual_input_sha256": DATA, "harper_membership_sha256": archived["pause_membership_sha256"],
        "multilingual": multilingual, "english": english, "harper": harper_scores, "fit": records,
        "depth_comparison": compare(multilingual, records),
        "threshold_note": "0.5 descriptive; recall comparisons use retrospective per-language ≤5% false cutoffs",
    }
    _write_json(root / "reports" / "gru-depth-study.json", result)
    return result["depth_comparison"]


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(run(data_root(), "cuda"), indent=2))
