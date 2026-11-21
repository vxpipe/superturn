"""Export ONNX artifacts and measure quality and single-core latency."""

from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from .augment import CONDITIONS
from .model import build
from .progress import Progress
from .train import PAUSES, _input, _stats, _threshold, paths, records

BASELINES = (300, 500, 1000, 1500)
DECISIONS = tuple(range(200, 2501, 100))


def _session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _predict(session, audio, gaps):
    result, = session.run(["p_complete"], {"audio": audio, "silence_ms": gaps})
    return result.flatten()


def _compact_initializers(path):
    graph = onnx.load(path)
    # Use portable float kernels for convolution and recurrence, with the
    # packed int8 weights dequantized by the runtime.
    weights = {node.input[1] for node in graph.graph.node if node.op_type == "Conv"}
    weights.update(name for node in graph.graph.node if node.op_type == "GRU" for name in node.input[1:3])
    weights.add("positions")
    prefix = []
    for tensor in list(graph.graph.initializer):
        if tensor.name not in weights or tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(tensor)
        scale = np.float32(max(np.abs(values).max() / 127, 1e-10))
        codes = np.rint(values / scale).clip(-127, 127).astype("int8")
        name = tensor.name
        graph.graph.initializer.remove(tensor)
        graph.graph.initializer.extend([
            numpy_helper.from_array(codes, name + ".int8"),
            numpy_helper.from_array(np.asarray(scale), name + ".scale"),
        ])
        prefix.append(onnx.helper.make_node("DequantizeLinear", [name + ".int8", name + ".scale"], [name]))
    original = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend(prefix + original)
    onnx.checker.check_model(graph)
    onnx.save(graph, path)


def export(checkpoint, *, limit=True):
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = build(config).eval()
    model.load_state_dict(saved["state"])
    torch.set_num_threads(1)
    float_path = checkpoint.with_name("float.onnx")
    quant_path = checkpoint.with_name("int8.onnx")
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.15, (2, model.samples)).astype("float32")
    gap = np.asarray([[200], [500]], dtype="float32")
    inputs = (torch.from_numpy(audio[:1]), torch.from_numpy(gap[:1]))
    torch.onnx.export(model, inputs, float_path, input_names=["audio", "silence_ms"],
                      output_names=["p_complete"], opset_version=17, dynamo=False,
                      do_constant_folding=True, dynamic_axes={
                          "audio": {0: "batch"}, "silence_ms": {0: "batch"}, "p_complete": {0: "batch"},
                      })
    with torch.inference_mode():
        expected = model(torch.from_numpy(audio), torch.from_numpy(gap)).numpy().flatten()
    observed = _predict(_session(float_path), audio, gap)
    parity = float(np.max(np.abs(expected - observed)))
    if not np.isfinite(parity) or parity >= 1e-3:
        raise ValueError(f"float export parity error {parity}")
    quantize_dynamic(float_path, quant_path, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=True,
                     extra_options={"MatMulConstBOnly": True})
    _compact_initializers(quant_path)
    quantized = _predict(_session(quant_path), audio, gap)
    if not np.isfinite(quantized).all():
        raise ValueError("nonfinite scores from int8 model")
    size = quant_path.stat().st_size
    details = {"float_max_abs_error": parity, "int8_bytes": size, "parameters": saved["parameters"],
               "threshold": saved["metrics"]["threshold"], "config": config}
    quant_path.with_name("export.json").write_text(json.dumps(details, indent=2) + "\n")
    if limit and (size > 1_000_000 or not 100000 <= saved["parameters"] <= 1_000_000):
        raise ValueError(f"export has {size} bytes and {saved['parameters']} parameters")
    return quant_path


def _summarize(entries, threshold):
    summaries = {}
    for pause in PAUSES:
        rows = [entry for entry in entries if entry[0] >= pause]
        metrics = _stats([row[2][DECISIONS.index(pause)] for row in rows], [row[1] for row in rows], threshold)
        if pause != 300:
            metrics.pop("auc")
        summaries[str(pause)] = metrics

    shifts = [entry for entry in entries if entry[1]]
    holds = [entry for entry in entries if not entry[1]]
    def first_decision(entry):
        gap, _, scores = entry
        return next((ms for ms, score in zip(DECISIONS, scores) if ms <= gap and score >= threshold), None)

    shift_times = [first_decision(entry) for entry in shifts]
    times = [value if value is not None else 2500
             for value, entry in zip(shift_times, shifts) if value is not None or entry[0] >= 2500]
    summaries["decision_ms"] = {
        "p50": float(np.percentile(times, 50)) if times else None,
        "p90": float(np.percentile(times, 90)) if times else None,
        "false_complete": sum(first_decision(entry) is not None or entry[0] >= 2500
                              for entry in holds) / len(holds) if holds else None,
        "unresolved_shifts": len(shifts) - len(times),
        "recall": len(times) / len(shifts) if shifts else None,
    }
    summaries["baselines"] = {
        str(ms): {
            "false_complete": sum(row[0] >= ms for row in holds) / len(holds) if holds else None,
            "recall": sum(row[0] >= ms for row in shifts) / len(shifts) if shifts else None,
            "decision_ms": ms,
        }
        for ms in BASELINES
    }
    return summaries


def benchmark(root, model_path, threshold, *, split="test", samples=250, name=None):
    import pyarrow.parquet as pq

    if samples < 1:
        raise ValueError("latency samples must be positive")
    model_path = Path(model_path)
    session = _session(model_path)
    length = session.get_inputs()[0].shape[1]
    conditions = CONDITIONS if split == "test" else ("clean",)
    work = {condition: paths(root, split, condition) for condition in conditions}
    total = sum(pq.ParquetFile(path).metadata.num_rows
                for files in work.values() for path in files)
    if not total:
        raise ValueError(f"no {split} benchmark examples")
    meter = Progress(f"benchmark-{name or model_path.parent.name}", total + samples, "items")
    groups = defaultdict(list)
    try:
        for condition, files in work.items():
            for rows in records(files):
                meter.advance(len(rows))
                rows = [row for row in rows if row["label"] in {"shift", "hold"}
                        and row["pause_samples"] >= DECISIONS[0] * 16]
                if not rows:
                    continue
                for row in rows:
                    decision_ms = [ms for ms in DECISIONS if ms * 16 <= row["pause_samples"]]
                    audio = np.stack([_input(row, length, ms) for ms in decision_ms]).astype("float32") / 32768
                    gaps = np.asarray(decision_ms, dtype="float32").reshape(-1, 1)
                    scores = _predict(session, audio, gaps)
                    example = (row["pause_samples"] / 16, int(row["label"] == "shift"), scores)
                    for key in (("all", "all", condition), (row["source"], "all", condition),
                                ("all", row["language"], condition),
                                (row["source"], row["language"], condition)):
                        groups[key].append(example)

        audio = np.zeros((1, length), "float32")
        gap = np.full((1, 1), 300, "float32")
        for _ in range(20):
            _predict(session, audio, gap)
        times = []
        previous_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        if previous_affinity:
            os.sched_setaffinity(0, {min(previous_affinity)})
        try:
            for _ in range(samples):
                start = time.perf_counter_ns()
                _predict(session, audio, gap)
                times.append((time.perf_counter_ns() - start) / 1e6)
                meter.advance(1)
        finally:
            if previous_affinity:
                os.sched_setaffinity(0, previous_affinity)

        if threshold is None:
            if split != "validation":
                raise ValueError("threshold selection requires validation data")
            usable = [row for row in groups["all", "all", "clean"] if row[0] >= 300]
            threshold = _threshold([row[2][DECISIONS.index(300)] for row in usable],
                                   [row[1] for row in usable])
        destination = root / "reports" / (name or model_path.parent.name)
        destination.mkdir(parents=True, exist_ok=True)
        report = {"threshold": threshold, "split": split,
                  "latency_ms": {f"p{p}": float(np.percentile(times, p)) for p in (50, 99)},
                  "groups": [{"source": s, "language": lang, "condition": c, **_summarize(rows, threshold)}
                             for (s, lang, c), rows in sorted(groups.items())]}
        (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

        def percentage(value):
            return "—" if value is None else f"{value:.1%}"

        lines = ["# Superturn benchmark", "",
                 f"Split `{split}` at threshold {threshold:.4f}. CPU p50 {report['latency_ms']['p50']:.2f} ms, "
                 f"p99 {report['latency_ms']['p99']:.2f} ms.", "",
                 "| Source | Language | Condition | Pause (ms) | Holds | Shifts | False complete | Recall | AUC |",
                 "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in report["groups"]:
            for pause in PAUSES:
                metrics = row[str(pause)]
                lines.append(f"| {row['source']} | {row['language']} | {row['condition']} | {pause} | "
                             f"{metrics['holds']} | {metrics['shifts']} | {percentage(metrics['false_complete'])} | "
                             f"{percentage(metrics['recall'])} | {percentage(metrics.get('auc'))} |")
        lines += ["", "| Source | Language | Condition | Decision p50 (ms) | Decision p90 (ms) | Unresolved shifts | Cumulative false complete | Timeout | Timeout false complete | Timeout recall |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["groups"]:
            for timeout, metrics in row["baselines"].items():
                lines.append(f"| {row['source']} | {row['language']} | {row['condition']} | "
                             f"{row['decision_ms']['p50']} | {row['decision_ms']['p90']} | "
                             f"{row['decision_ms']['unresolved_shifts']} | "
                             f"{percentage(row['decision_ms']['false_complete'])} | {timeout} | "
                             f"{percentage(metrics['false_complete'])} | {percentage(metrics['recall'])} |")
        (destination / "report.md").write_text("\n".join(lines) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


def quantization_auc(root, model_path, threshold):
    model_path = Path(model_path)
    reports = [
        benchmark(root, path, None if path == model_path else threshold, split="validation", samples=25,
                  name=f"{path.parent.name}-{path.stem}-validation")
        for path in (model_path.with_name("float.onnx"), model_path)
    ]
    return [next(row["300"]["auc"] for row in report["groups"]
                 if (row["source"], row["language"], row["condition"]) == ("all", "all", "clean"))
            for report in reports]
