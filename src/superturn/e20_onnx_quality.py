"""Export one fixed 973k E20 development checkpoint and verify ONNX quality."""

import argparse
import json

import numpy as np

from .ami_calibration import _write_json
from .balanced_development_scorecard import JOB as SCORECARD, UNSEEN_IN_FIT, cell, frozen, macro
from .evaluate import _predict_calls, _session, export
from .language_failure_audit import read_receipt, read_verified
from .multilingual_teacher_probe import score_languages
from .own_student_failure_refresh import agree
from .own_student_fidelity_confirmation import LOCK, NEW_SEEDS, TRIAL, artifact
from .release_cohort import write_once
from .sources import data_root
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import pause_rows

JOB = "e20-onnx-quality"
SEED = NEW_SEEDS[0]


def export_fixed(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the archived model and development audio stay on Vast")
    lock, lock_sha = read_receipt(root, LOCK)
    trial, trial_sha = read_receipt(root, TRIAL)
    if trial["lock_sha256"] != lock_sha or not trial["gates"]["fixed_longer_schedule"]:
        raise ValueError("the retained E20 schedule is required")
    checkpoint = lock["fits"][str(SEED)]["epoch20_checkpoint"]
    path = read_verified(root, checkpoint)
    directory = root / "runs" / JOB / checkpoint["sha256"][:12]
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "artifacts.json"
    protocol = {"seed": SEED, "checkpoint": checkpoint, "e20_lock_sha256": lock_sha,
                "e20_trial_sha256": trial_sha, "selection": "first fixed trial seed; no ranking",
                "spectrum": "stft"}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["protocol"] != protocol:
            raise ValueError("the sealed export comes from a different checkpoint")
        for name in ("float", "int8", "export"):
            read_verified(root, receipt[name])
    else:
        quant = export(path, destination=directory, limit=False)
        metadata = quant.with_name("export.json")
        summary = json.loads(metadata.read_text())
        receipt = {"protocol": protocol, "float": artifact(root, quant.with_name("float.onnx")),
                   "int8": artifact(root, quant), "export": artifact(root, metadata),
                   "int8_bytes": summary["int8_bytes"], "parameters": summary["parameters"],
                   "parity_max_abs_error": summary["float_max_abs_error"]}
        write_once(receipt_path, receipt)
    return receipt


def compact_fixed(root, receipt):
    from .compact_onnx_frontend import compact_frontend

    source = read_verified(root, receipt["int8"])
    directory = source.parent
    manifest = directory / "compact-v2.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["original"] != receipt["int8"]:
            raise ValueError("the compact graph has a different ancestor")
        read_verified(root, saved["candidate"])
    else:
        output = directory / "int8-compact-v2.onnx"
        stats = compact_frontend(source, output)
        saved = {**stats, "original": receipt["int8"], "candidate": artifact(root, output)}
        write_once(manifest, saved)
    return saved


class BatchOneOnnx:
    def __init__(self, path):
        self.session = _session(path)

    def __call__(self, clips, elapsed):
        import torch

        return torch.from_numpy(_predict_calls(self.session, clips.numpy(), elapsed.numpy()))


def quality(root, receipt, compact):
    directory = read_verified(root, receipt["int8"]).parent
    peers = {name: BatchOneOnnx(read_verified(root, source)) for name, source in
             (("e20-float-onnx", receipt["float"]), ("e20-stock-int8", receipt["int8"]),
              ("e20-compact-int8", compact["candidate"]))}
    reference, trial, _ = frozen(root)
    archived, _ = read_receipt(root, f"{SCORECARD}-references")
    if archived["trial_sha256"] != trial:
        raise ValueError("the comparison scorecard differs from the E20 trial")
    paired = {}
    score_languages(root, peers, "cpu", paired_rows=paired, progress_name=JOB)
    for name, examples in (("en", english_rows(root)), ("harper-88", harper_rows(root, native_prefix=True))):
        audio, labels, groups = zip(*pause_rows(examples), strict=True)
        clips = np.stack(audio)
        gap = np.full((len(clips), 1), 300, dtype=np.float32)
        paired[name] = {"labels": list(labels), "turn_ordinals": list(groups),
                        "scores": {model: _predict_calls(peer.session, clips, gap).tolist()
                                   for model, peer in peers.items()}}
    harper = paired.pop("harper-88")
    if set(paired) != set(reference):
        raise ValueError("the frozen development languages changed")
    scores, cells = {}, {}
    fixed = f"epoch20-{SEED}"
    for language, row in paired.items():
        before = reference[language]
        agree(before, row["labels"], row["turn_ordinals"], language)
        before["scores"].update(row["scores"])
        cells[language] = cell(before, language)
        cells[language]["benchmark_aggregates"] = {name: value for name, value in archived["languages"][language]["models"].items()
                                                   if name.startswith(("livekit-", "smart-turn-"))}
        original = np.asarray(before["scores"][fixed])
        float_onnx, stock, small = (np.asarray(row["scores"][name]) for name in peers)
        scores[language] = {"float_vs_archived_abs_error_max": float(np.max(np.abs(original - float_onnx))),
                            "float_vs_archived_abs_error_p95": float(np.quantile(np.abs(original - float_onnx), .95)),
                            "compact_vs_stock_abs_error_max": float(np.max(np.abs(small - stock))),
                            "compact_vs_stock_abs_error_p95": float(np.quantile(np.abs(small - stock), .95)),
                            "compact_vs_float_auc": (cells[language]["models"]["e20-compact-int8"]["auc"] -
                                                      cells[language]["models"]["e20-float-onnx"]["auc"])}

    from .balanced_development_scorecard import evaluate

    labels = np.asarray(harper["labels"], dtype=bool)
    original_harper = archived["harper-88"]
    if (int((~labels).sum()), int(labels.sum())) != (original_harper["eval_holds"], original_harper["eval_shifts"]):
        raise ValueError("the frozen HarperValley direction cohort changed")
    harper_cell = {**original_harper, "models": dict(original_harper["models"])}
    for name, values in harper["scores"].items():
        threshold = cells["en"]["models"][name]["threshold"]
        harper_cell["models"][name] = {"threshold": threshold, **evaluate(np.asarray(values), labels, threshold)}
    macro_all = macro(cells, sorted(cells))
    macro_six = macro(cells, UNSEEN_IN_FIT)
    for result, key in ((macro_all, "macro"), (macro_six, "zero_supervision_macro")):
        result["benchmark_aggregates"] = {name: value for name, value in archived[key]["models"].items()
                                          if name.startswith(("livekit-", "smart-turn-"))}
    _write_json(directory / "paired-scores.json", {**reference, "harper-88": harper})
    report = {"scope": "single fixed E20 seed, development-only export verification", "artifacts": receipt,
              "compact": compact, "macro": macro_all, "six_language": macro_six, "languages": cells,
              "harper-88": harper_cell, "numerical_by_language": scores,
              "float_parity_on_development": max(stats["float_vs_archived_abs_error_max"] for stats in scores.values()),
              "audio_only_size_pass": compact["compact_bytes"] <= 1_000_000,
              "per_language_auc_loss_within_one_point": all(stats["compact_vs_float_auc"] >= -.01 for stats in scores.values()),
              "latency": None}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps({key: value for key, value in report.items() if key in (
        "float_parity_on_development", "audio_only_size_pass", "per_language_auc_loss_within_one_point", "latency",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    root = data_root()
    receipt = export_fixed(root)
    if args.quality:
        quality(root, receipt, compact_fixed(root, receipt))
    else:
        print(json.dumps(compact_fixed(root, receipt) if args.compact else receipt, indent=2))
