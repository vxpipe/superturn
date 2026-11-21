"""Score a frozen, manually supported subset of the English AMI test split."""

import hashlib
import json

import pyarrow.parquet as pq

from .ami_reference import audit_sample
from .augment import CONDITIONS
from .labels import read_rows

DECISION_MS = 300
SCOPE_FILE = "ami-reference-test-scope.json"
AMI_THRESHOLD_ORIGIN = "ami_telephone_validation_fit"


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


class AmiTelephoneValidation:
    def __init__(self, root):
        import numpy as np

        from .augment import condition

        annotations = [row for row in read_rows(root / "labels" / "ami.jsonl") if row.split == "validation"]
        limits = {}
        votes = audit_sample(root, annotations, pause_limits=limits)
        supported = {_case_id(row.audio, row.pause_start, row.label)
                     for row, vote in votes.items()
                     if vote == "supported" and limits[row] >= DECISION_MS * 16}
        rows = list(_cases(root, "validation", "clean", with_pcm=True))
        self.confirmed = [index for index, (key, _) in enumerate(rows) if key in supported]
        self.truth = [int(row["label"] == "shift") for _, row in rows]
        self.confirmed_truth = [self.truth[index] for index in self.confirmed]
        if len(set(self.truth)) != 2 or len(set(self.confirmed_truth)) != 2:
            raise ValueError("AMI validation needs both holds and shifts in each label scope")
        self.telephone = [
            {**row, "pcm": condition(np.frombuffer(row["pcm"], "<i2"), "telephone", 0).tobytes()}
            for _, row in rows
        ]
        manifest = json.loads((root / "shards" / "manifest.json").read_text())
        self.scope = {"split": "validation", "source": "ami", "condition": "telephone",
                      "validation_sha256": manifest["sha256_by_split"]["validation"],
                      "decision_ms": DECISION_MS}

    def _batches(self, samples):
        import numpy as np

        from .train import _input

        for start in range(0, len(self.telephone), 16):
            rows = self.telephone[start:start + 16]
            audio = np.stack([_input(row, samples, DECISION_MS) for row in rows])
            yield audio.astype("float32") / 32768, np.full((len(rows), 1), DECISION_MS, dtype="float32")

    def _report(self, model_path, scores, pooled_threshold):
        from .train import _stats, _threshold

        thresholds = {"pooled_clean_validation": pooled_threshold,
                      AMI_THRESHOLD_ORIGIN: _threshold(scores, self.truth)}
        confirmed_scores = [scores[index] for index in self.confirmed]
        return {
            **self.scope,
            "name": model_path.parent.name,
            "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "thresholds": [{
                "origin": origin,
                "threshold": threshold,
                "all_ami": _stats(scores, self.truth, threshold),
                "annotation_supported": _stats(confirmed_scores, self.confirmed_truth, threshold),
            } for origin, threshold in thresholds.items()],
        }

    def float_checkpoint(self, checkpoint):
        import torch

        from .model import build

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        scores = []
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = build(saved["config"]).eval()
            model.load_state_dict(saved["state"])
            with torch.inference_mode():
                for audio, elapsed in self._batches(model.samples):
                    scores.extend(model(torch.from_numpy(audio), torch.from_numpy(elapsed)).flatten().tolist())
        finally:
            torch.set_num_threads(prior_threads)
        return {**self._report(checkpoint, scores, saved["metrics"]["threshold"]),
                "epoch": saved["metrics"]["epoch"]}

    def int8_export(self, model_path, pooled_threshold):
        from .evaluate import _predict_calls, _session

        session = _session(model_path)
        samples = session.get_inputs()[0].shape[1]
        scores = []
        for audio, elapsed in self._batches(samples):
            scores.extend(_predict_calls(session, audio, elapsed).tolist())
        return self._report(model_path, scores, pooled_threshold)


def compare_validation(root):
    """Compare complete Stage 1 checkpoints before making a selection decision."""
    from .progress import Progress
    from .train import completed_run

    files = []
    for marker in sorted((root / "runs").glob("s1-*/finished.json")):
        signature = json.loads(marker.read_text())["signature"]
        checkpoint = completed_run(root, marker.parent.name, signature["config"], signature["device"])
        if checkpoint is not None:
            files.append(checkpoint)
    if not files:
        raise ValueError("no verified Stage 1 checkpoints are available")

    validation = AmiTelephoneValidation(root)
    results = []
    meter = Progress("stage1-ami-validation", len(files), "checkpoints")
    try:
        for checkpoint in files:
            results.append(validation.float_checkpoint(checkpoint))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    out = root / "reports" / "stage1-ami-validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        **validation.scope, "checkpoints": results,
    }, indent=2) + "\n")
    return out
