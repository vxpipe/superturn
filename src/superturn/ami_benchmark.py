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


def _test_cases(root, condition, *, with_pcm=False):
    columns = ["audio", "pause_start", "label"]
    if with_pcm:
        columns.extend(("pause_samples", "pcm"))
    filters = [("source", "=", "ami"), ("pause_samples", ">=", DECISION_MS * 16)]
    for path in sorted((root / "shards" / "test" / condition).glob("*.parquet")):
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
    available = {key for key, _ in _test_cases(root, "clean")}
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
        matched = {key: row for key, row in _test_cases(root, condition, with_pcm=True) if key in case_ids}
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
