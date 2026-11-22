"""Render an aggregate balanced-accuracy report as a Markdown scorecard."""

import argparse
import json
from pathlib import Path

MODEL_LABELS = {
    "Superturn 973k E20 (six seeds)": None,
    "Owned Whisper large-v3 teacher": "whisper-large-v3",
    "Smart Turn v3, 3 s": "smart-turn-3s",
    "Smart Turn v3, 8 s": "smart-turn-8s",
    "LiveKit audio v1.1-mini": "livekit-v1.1-mini",
}
METRICS = ("balanced_accuracy", "shift_recall", "false_complete_rate", "recall_at_5pct", "auc", "pauc_10")


def ordered_rows(report):
    yield "Macro (14 languages)", report["macro"]
    languages = report["languages"]
    yield "en", languages["en"]
    for language in sorted(languages.keys() - {"en"}):
        yield language, languages[language]
    yield "Zero supervision (six-language macro)", report["zero_supervision_macro"]
    if "harper-88" in report:
        yield "HarperValley-88", report["harper-88"]


def metric_cells(row, model):
    if model is None:
        values = row["student_seed_mean_and_range"]
        return [f"{100 * value['mean']:.2f} ({100 * value['min']:.2f}–{100 * value['max']:.2f})"
                for value in (values[name] for name in METRICS)]
    if model in row["models"]:
        return [f"{100 * row['models'][model][name]:.2f}" for name in METRICS]
    return ["—"] * len(METRICS)


def render(report):
    lines = [
        "# Balanced-accuracy model scorecard",
        "",
        "Telephone development at 300 ms. Thresholds maximize balanced accuracy on each",
        "language's calibration half of hash-assigned whole turns. Evaluation uses the",
        "complementary half. HarperValley-88 uses the frozen English threshold and",
        "provides directional evidence. Provider turns lack global speaker IDs,",
        "so the development split cannot establish the unseen-voice release gate.",
        "",
        "The release target is 80% macro balanced accuracy and 70% per supported language.",
        "All metrics use a 0–100 scale. Student entries show the six-seed mean and range.",
        "Recall at 5% false completes is a retrospective operating point on the evaluation",
        "half. pAUC integrates to a 10% false-complete rate, normalized to [0, 1].",
        "Pipeline replay at these thresholds is pending.",
        "",
        "| View | Calibration holds / shifts | Evaluation holds / shifts |",
        "|---|---:|---:|",
    ]
    rows = list(ordered_rows(report))
    for name, row in rows:
        calibration = f"{row['fit_holds']} / {row['fit_shifts']}" if "fit_holds" in row else "Per language"
        if name == "HarperValley-88":
            english = report["languages"]["en"]
            calibration = f"English {english['fit_holds']} / {english['fit_shifts']}"
        lines.append(f"| {name} | {calibration} | {row['eval_holds']} / {row['eval_shifts']} |")
    for name, model in MODEL_LABELS.items():
        if model is not None and model not in report["macro"]["models"]:
            continue
        lines.extend([
            "",
            f"## {name}",
            "",
            "| View | Balanced acc. | Shift recall | False completes | Recall at ≤5% FC | AUC | pAUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for view, row in rows:
            lines.append("| " + " | ".join([view, *metric_cells(row, model)]) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(render(json.loads(args.report.read_text())))
