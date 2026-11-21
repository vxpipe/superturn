"""Score a frozen, manually supported subset of the English AMI test split."""

import hashlib
import json

import pyarrow.parquet as pq

from .ami_reference import audit_sample
from .augment import CONDITIONS
from .labels import read_rows

DECISION_MS = 300
SCOPE_FILE = "ami-reference-test-scope.json"


def _test_fingerprint(root):
    manifest = json.loads((root / "shards" / "manifest.json").read_text())
    return manifest["test_sha256"]


def _case_id(audio, boundary, label):
    key = json.dumps((audio, boundary, label))
    return hashlib.sha256(key.encode()).hexdigest()


def _cases(root, split, condition, *, with_pcm=False):
    columns = ["audio", "pause_start", "label"]
    if with_pcm:
        columns.extend(("pause_samples", "pcm"))
    filters = [("source", "=", "ami"), ("pause_samples", ">=", DECISION_MS * 16)]
    for path in sorted((root / "shards" / split / condition).glob("*.parquet")):
        for row in pq.read_table(path, columns=columns, filters=filters).to_pylist():
            yield _case_id(row["audio"], row["pause_start"], row["label"]), row


def prepare(root):
    destination = root / "reports" / SCOPE_FILE
    if not destination.exists() and (root / "reports" / "experiments" / "selection.json").exists():
        raise ValueError("freeze the AMI reference scope before selecting models")
    label_path = root / "labels" / "ami.jsonl"
    rows = [row for row in read_rows(label_path) if row.split == "test"]
    pause_limits = {}
    votes = audit_sample(root, rows, pause_limits=pause_limits)
    available = {key for key, _ in _cases(root, "test", "clean")}
    included = [
        (_case_id(row.audio, row.pause_start, row.label), row)
        for row, vote in votes.items() if vote == "supported" and pause_limits[row] >= DECISION_MS * 16
    ]
    selected = {key: row for key, row in included if key in available}
    if {row.label for row in selected.values()} != {"hold", "shift"}:
        raise ValueError("AMI reference scope needs both verified holds and shifts")
    annotation, = (root / "raw" / "ami_annotations").glob("*.zip")
    report = {
        "split": "test",
        "decision_ms": DECISION_MS,
        "test_sha256": _test_fingerprint(root),
        "annotations_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "caller_end_tolerance_ms": 200,
        "next_onset_tolerance_ms": 250,
        "examples": len(selected),
        "holds": sum(row.label == "hold" for row in selected.values()),
        "shifts": sum(row.label == "shift" for row in selected.values()),
        "speakers": len({row.speaker for row in selected.values()}),
        "case_ids": sorted(selected),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and json.loads(destination.read_text()) != report:
        raise ValueError("AMI reference test scope differs from the frozen scope")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


def benchmark_selected(root):
    import numpy as np

    from .evaluate import _predict_calls, _session
    from .train import _input, _stats

    reports = root / "reports"
    scope_path = reports / SCOPE_FILE
    scope = json.loads(scope_path.read_text())
    if scope["test_sha256"] != _test_fingerprint(root):
        raise ValueError("AMI reference test scope belongs to a different frozen test set")
    selection_bytes = (reports / "experiments" / "selection.json").read_bytes()
    selection = json.loads(selection_bytes)
    if selection["split"] != "validation" or not selection["candidates"]:
        raise ValueError("AMI reference benchmark needs validation-selected candidates")
    case_ids = set(scope["case_ids"])
    conditions = {}
    for condition in CONDITIONS:
        matched = {key: row for key, row in _cases(root, "test", condition, with_pcm=True) if key in case_ids}
        if matched.keys() != case_ids:
            raise ValueError(f"AMI reference case membership changed in {condition}")
        conditions[condition] = [matched[key] for key in scope["case_ids"]]

    for candidate in selection["candidates"]:
        name, threshold = candidate["name"], candidate["threshold"]
        model = root / "runs" / name / "int8.onnx"
        session = _session(model)
        length = session.get_inputs()[0].shape[1]
        groups = []
        for condition, rows in conditions.items():
            audio = np.stack([_input(row, length, DECISION_MS) for row in rows]).astype("float32") / 32768
            silence = np.full((len(rows), 1), DECISION_MS, dtype="float32")
            scores = _predict_calls(session, audio, silence).tolist()
            groups.append({"condition": condition,
                           **_stats(scores, [int(row["label"] == "shift") for row in rows], threshold)})
        out = reports / name / "ami-reference-test.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            **{key: value for key, value in scope.items() if key != "case_ids"},
            "scope_sha256": hashlib.sha256(scope_path.read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "threshold": threshold,
            "groups": groups,
        }, indent=2) + "\n")
        print(out)


def compare_validation(root):
    """Compare complete Stage 1 checkpoints before making a selection decision."""
    import numpy as np
    import torch

    from .augment import condition
    from .model import build
    from .progress import Progress
    from .train import _input, _stats, _threshold, completed_run

    files = []
    for directory in sorted((root / "runs").glob("s1-*")):
        marker = directory / "finished.json"
        if marker.is_file():
            signature = json.loads(marker.read_text())["signature"]
            checkpoint = completed_run(root, directory.name, signature["config"], signature["device"])
            if checkpoint is not None:
                files.append(checkpoint)
    if not files:
        raise ValueError("no verified Stage 1 checkpoints are available")

    annotations = [row for row in read_rows(root / "labels" / "ami.jsonl") if row.split == "validation"]
    limits = {}
    votes = audit_sample(root, annotations, pause_limits=limits)
    supported = {_case_id(row.audio, row.pause_start, row.label)
                 for row, vote in votes.items() if vote == "supported" and limits[row] >= DECISION_MS * 16}
    rows = list(_cases(root, "validation", "clean", with_pcm=True))
    confirmed = [index for index, (key, _) in enumerate(rows) if key in supported]
    truth = [int(row["label"] == "shift") for _, row in rows]
    confirmed_truth = [truth[index] for index in confirmed]
    if len(set(truth)) != 2 or len(set(confirmed_truth)) != 2:
        raise ValueError("AMI validation needs both holds and shifts in each label scope")
    telephone = [{**row, "pcm": condition(np.frombuffer(row["pcm"], "<i2"), "telephone", 0).tobytes()}
                 for _, row in rows]
    torch.set_num_threads(1)
    results = []
    meter = Progress("stage1-ami-validation", len(files), "checkpoints")
    try:
        for checkpoint in files:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = build(saved["config"]).eval()
            model.load_state_dict(saved["state"])
            scores = []
            with torch.inference_mode():
                for start in range(0, len(telephone), 16):
                    batch = telephone[start:start + 16]
                    audio = np.stack([_input(row, model.samples, DECISION_MS) for row in batch])
                    normalized = torch.from_numpy(audio.astype("float32") / 32768)
                    scores.extend(model(normalized, torch.full((len(batch), 1), float(DECISION_MS))).flatten().tolist())
            thresholds = {"pooled_clean_validation": saved["metrics"]["threshold"],
                          "ami_telephone_validation_fit": _threshold(scores, truth)}
            confirmed_scores = [scores[index] for index in confirmed]
            results.append({
                "name": checkpoint.parent.name,
                "epoch": saved["metrics"]["epoch"],
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "thresholds": [{
                    "origin": origin,
                    "threshold": threshold,
                    "all_ami": _stats(scores, truth, threshold),
                    "annotation_supported": _stats(confirmed_scores, confirmed_truth, threshold),
                } for origin, threshold in thresholds.items()],
            })
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    out = root / "reports" / "stage1-ami-validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "shards" / "manifest.json").read_text())
    out.write_text(json.dumps({
        "split": "validation", "source": "ami", "condition": "telephone",
        "validation_sha256": manifest["sha256_by_split"]["validation"],
        "decision_ms": DECISION_MS, "checkpoints": results,
    }, indent=2) + "\n")
    return out
