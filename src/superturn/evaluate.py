"""Export ONNX artifacts and measure quality and single-core latency."""

from collections import defaultdict
import hashlib
import json
import math
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
CONVERSATION_SOURCES = {"ami", "indic_diarbench"}


def _session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    model = path if isinstance(path, bytes) else str(path)
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


def _predict(session, audio, gaps):
    result, = session.run(["p_complete"], {"audio": audio, "silence_ms": gaps})
    return result.flatten()


def _predict_calls(session, audio, gaps):
    """Score each example as an independent batch-one runtime call."""
    return np.fromiter(
        (_predict(session, audio[index:index + 1], gaps[index:index + 1])[0]
         for index in range(len(audio))),
        dtype=np.float32, count=len(audio),
    )


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


def export(checkpoint, *, limit=True, spectrum="stft"):
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
    if spectrum == "convolution":
        cases = np.tile(audio, (4, 1))
        cases[2, :-4800] = 0
        cases[3, -4800:] = 0
        cases[4] *= .001
        cases[5] = 0
        phase = 2 * np.pi * 160 * np.arange(model.samples) / 16000
        cases[6] = .2 * np.sin(phase)
        cases[7] = .03 * np.sin(2 * phase) + .01 * cases[7]
        audio = cases
        gap = np.tile(gap, (4, 1))
    elif spectrum != "stft":
        raise ValueError(f"unsupported export spectrum {spectrum!r}")
    with torch.inference_mode():
        expected = model(torch.from_numpy(audio), torch.from_numpy(gap)).numpy().flatten()
    if spectrum == "convolution":
        from .spectral import ConvolutionSpectrum

        model.features = ConvolutionSpectrum(active_stats=model.features.active_stats, hop=model.features.hop,
                                             spectral_rate=model.features.spectral_rate).eval()
    inputs = (torch.from_numpy(audio[:1]), torch.from_numpy(gap[:1]))
    torch.onnx.export(model, inputs, float_path, input_names=["audio", "silence_ms"],
                      output_names=["p_complete"], opset_version=17, dynamo=False,
                      do_constant_folding=True, dynamic_axes={
                          "audio": {0: "batch"}, "silence_ms": {0: "batch"}, "p_complete": {0: "batch"},
                      })
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
    if spectrum != "stft":
        details["spectrum"] = spectrum
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
    times = sorted(value if value is not None else 2500
                   for value, entry in zip(shift_times, shifts)
                   if value is not None or entry[0] >= 2500)

    def shift_percentile(fraction):
        # Unresolved shifts are right-censored. Put them last so a small
        # subset of easy decisions cannot certify the full shift population.
        rank = math.ceil(fraction * len(shifts)) - 1
        return times[rank] if 0 <= rank < len(times) else None

    summaries["decision_ms"] = {
        "p50": shift_percentile(0.50),
        "p90": shift_percentile(0.90),
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


def _acceptance(model_path, report):
    rows = {(row["source"], row["language"], row["condition"]): row for row in report["groups"]}
    conversation = rows.get(("conversation", "all", "telephone"))
    telephone = rows.get(("all", "all", "telephone"))
    clean = rows.get(("all", "all", "clean"))
    noise = rows.get(("all", "all", "telephone-noise"))
    if not conversation or not telephone or not clean or not noise:
        return {}

    threshold_metrics = conversation["300"]
    false_complete = threshold_metrics["false_complete"]
    recall = threshold_metrics["recall"]
    decision = conversation["decision_ms"]["p50"]
    overall = telephone["300"]["false_complete"]
    failing_languages = [
        {"language": language, "holds": row["300"]["holds"], "false_complete": row["300"]["false_complete"]}
        for (source, language, condition_name), row in rows.items()
        if source == "all" and language != "all" and condition_name == "telephone"
        and overall is not None and row["300"]["false_complete"] is not None
        and row["300"]["false_complete"] > 2 * overall
    ]
    unmeasured = [
        language for (source, language, condition_name), row in rows.items()
        if source == "all" and language != "all" and condition_name == "telephone"
        and row["300"]["false_complete"] is None
    ]
    clean_rate = clean["300"]["false_complete"]
    noise_rate = noise["300"]["false_complete"]
    delta = None if clean_rate is None or noise_rate is None else noise_rate - clean_rate
    cumulative = conversation["decision_ms"]["false_complete"]
    decision_recall = conversation["decision_ms"]["recall"]
    beats_timeout = {
        timeout: decision is not None and cumulative is not None and metrics["false_complete"] is not None
        and decision < metrics["decision_ms"] and cumulative <= metrics["false_complete"]
        and decision_recall is not None and metrics["recall"] is not None
        and decision_recall >= metrics["recall"]
        for timeout, metrics in conversation["baselines"].items()
    }
    artifact_bytes = model_path.stat().st_size
    metadata = model_path.with_name("export.json")
    parameters = json.loads(metadata.read_text()).get("parameters") if metadata.is_file() else None
    return {
        "artifact": {"passed": artifact_bytes <= 1_000_000 and parameters is not None
                     and 100_000 <= parameters <= 1_000_000,
                     "bytes": artifact_bytes, "parameters": parameters},
        "cpu_p99": _cpu_acceptance(report["latency_ms"]),
        "conversation_telephone_300ms": {
            "historical_80_recall_at_5_false_complete": (
                false_complete is not None and recall is not None and false_complete <= 0.05 and recall >= 0.8
            ),
            **threshold_metrics,
        },
        "conversation_telephone_decision": {"passed": decision is not None and decision <= 500,
                                           **conversation["decision_ms"]},
        "language_gap_telephone": {"passed": overall is not None and not failing_languages and not unmeasured,
                                   "failing": failing_languages, "no_holds": unmeasured},
        "telephone_noise_delta": {"passed": delta is not None and delta <= 0.03, "delta": delta},
        "conversation_telephone_timeouts": {"passed": all(beats_timeout.values()), "timeouts": beats_timeout},
    }


def _cpu_acceptance(latency):
    if latency is None:
        return {"passed": None, "milliseconds": None}
    return {"passed": latency["p99"] < 2, "milliseconds": latency["p99"]}


def _measure_latency(session, clips, samples, condition, name):
    if not clips:
        raise ValueError(f"no {condition} audio for the CPU latency benchmark")
    for index in range(20):
        _predict(session, *clips[index % len(clips)])
    times = []
    meter = Progress(f"latency-{name}", samples, "calls")
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    try:
        if affinity:
            os.sched_setaffinity(0, {min(affinity)})
        for index in range(samples):
            audio, gap = clips[index % len(clips)]
            start = time.perf_counter_ns()
            _predict(session, audio, gap)
            times.append((time.perf_counter_ns() - start) / 1e6)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    finally:
        if affinity:
            os.sched_setaffinity(0, affinity)
    return {**{f"p{p}": float(np.percentile(times, p)) for p in (50, 99)},
            "samples": samples, "condition": condition, "input_clips": len(clips)}


def benchmark(root, model_path, threshold, *, split="test", samples=250, name=None, quality_first=False):
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
    run_name = name or model_path.parent.name
    meter = Progress(f"benchmark-{run_name}", total, "items")
    groups = defaultdict(list)
    latency_inputs = []
    latency_condition = "telephone" if split == "test" else "clean"
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
                    scores = _predict_calls(session, audio, gaps).tolist()
                    if condition == latency_condition and len(latency_inputs) < 16:
                        latency_index = DECISIONS.index(300) if 300 in decision_ms else 0
                        latency_inputs.append((audio[latency_index:latency_index + 1].copy(),
                                               gaps[latency_index:latency_index + 1].copy()))
                    example = (row["pause_samples"] / 16, int(row["label"] == "shift"), scores)
                    keys = [("all", "all", condition), (row["source"], "all", condition),
                            ("all", row["language"], condition),
                            (row["source"], row["language"], condition)]
                    if row["source"] in CONVERSATION_SOURCES:
                        keys.extend([("conversation", "all", condition),
                                     ("conversation", row["language"], condition)])
                    for key in keys:
                        groups[key].append(example)

        if threshold is None:
            if split != "validation":
                raise ValueError("threshold selection requires validation data")
            usable = [row for row in groups["all", "all", "clean"] if row[0] >= 300]
            threshold = _threshold([row[2][DECISIONS.index(300)] for row in usable],
                                   [row[1] for row in usable])
        destination = root / "reports" / run_name
        destination.mkdir(parents=True, exist_ok=True)
        report = {"threshold": threshold, "split": split,
                  "latency_ms": None,
                  "groups": [{"source": s, "language": lang, "condition": c, **_summarize(rows, threshold)}
                             for (s, lang, c), rows in sorted(groups.items())]}
        if split == "test":
            report["acceptance"] = _acceptance(model_path, report)
        checks = report.get("acceptance", {})
        if quality_first:
            # This entry point applies one legacy cutoff to mixed-source shards.
            # Independent per-language calibration and held-out phone evaluation
            # are required for the release balanced-accuracy and latency gates.
            report["release_balanced_accuracy"] = {"passed": None, "status": "independent cohort required",
                                                    "macro_target": .80, "per_language_target": .70,
                                                    "deadline_ms": 300}
        else:
            report["latency_ms"] = _measure_latency(session, latency_inputs, samples, latency_condition, run_name)
        if "cpu_p99" in checks:
            checks["cpu_p99"] = _cpu_acceptance(report["latency_ms"])
        (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

        def percentage(value):
            return "—" if value is None else f"{value:.1%}"

        latency = report["latency_ms"]
        timing = (f"CPU p50 {latency['p50']:.2f} ms, p99 {latency['p99']:.2f} ms." if latency else
                  "CPU latency deferred pending the telephone conversation quality target.")
        lines = ["# Superturn benchmark", "",
                 f"Split `{split}` at threshold {threshold:.4f}. {timing}", "",
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
        if report.get("acceptance"):
            lines += ["", "## Measured acceptance gates", "", "| Gate | Result |", "|---|---|"]
            for gate, details in report["acceptance"].items():
                if gate == "conversation_telephone_300ms":
                    continue
                status = "Pass" if details["passed"] else "Open"
                lines.append(f"| {gate.replace('_', ' ')} | {status} |")
        if quality_first:
            lines += ["", "Release balanced accuracy at 300 ms awaits independent calibration and test data.",
                      "The telephone conversation scores above use the historical single threshold."]
        (destination / "report.md").write_text("\n".join(lines) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


def quantization_auc(root, model_path, threshold):
    """Score only the 300 ms validation point needed for the export gate."""
    import pyarrow.parquet as pq

    model_path = Path(model_path)
    sessions = {
        "float": _session(model_path.with_name("float.onnx")),
        "int8": _session(model_path),
    }
    validation = paths(root, "validation")
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in validation)
    if not total:
        raise ValueError("quantization check requires validation examples")
    meter = Progress(f"quantization-{model_path.parent.name}", total, "examples")
    by_kind = {kind: defaultdict(list) for kind in sessions}
    length = sessions["float"].get_inputs()[0].shape[1]
    try:
        for batch in records(validation):
            rows = [row for row in batch if row["label"] in {"hold", "shift"}
                    and row["pause_samples"] >= 4800]
            if rows:
                audio = np.stack([_input(row, length, 300) for row in rows]).astype("float32") / 32768
                elapsed = np.full((len(rows), 1), 300, dtype="float32")
                for kind, session in sessions.items():
                    for row, score in zip(rows, _predict_calls(session, audio, elapsed)):
                        pair = (float(score), int(row["label"] == "shift"))
                        for group in (("all", "all"), (row["source"], "all"),
                                      ("all", row["language"]), (row["source"], row["language"])):
                            by_kind[kind][group].append(pair)
            meter.advance(len(batch))

        aucs = []
        for kind, groups in by_kind.items():
            combined = groups["all", "all"]
            if not combined:
                raise ValueError("quantization check found no labelled 300 ms pauses")
            all_scores, all_labels = zip(*combined)
            applied = threshold if kind == "float" else _threshold(all_scores, all_labels)
            aucs.append(_stats(all_scores, all_labels, applied)["auc"])
            values = []
            for (source, language), pairs in sorted(groups.items()):
                scores, labels = zip(*pairs)
                values.append({"source": source, "language": language, "condition": "clean",
                               "300": _stats(scores, labels, applied)})
            path = root / "reports" / f"{model_path.parent.name}-{kind}-validation" / "metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"threshold": applied, "split": "validation", "groups": values}, indent=2) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return aucs


def release_quality(root, model_path, plan_path, fit_roster_path, cohort_path, audio_path, *,
                    role, text_path=None, plan_sha256, calibration_sha256=None):
    """Collect 300 ms scores with the same ORT engine as the legacy benchmark."""
    from fractions import Fraction
    from .balanced_development_scorecard import METRICS, balanced_threshold, evaluate
    from .release_cohort import (
        CONTRACT, DEADLINE, digest, ensure_disjoint, identities, validate_manifest,
        validate_plan, verified_json, write_once,
    )

    if not (root / ".campaign-machine").is_file():
        raise ValueError("private telephone recordings stay on Vast")
    plan = verified_json(plan_path, plan_sha256)
    validate_plan(plan)
    artifact_bytes = Path(model_path).read_bytes()
    if role not in ("calibration", "test") or hashlib.sha256(artifact_bytes).hexdigest() != plan["artifact_sha256"]:
        raise ValueError("unknown role or the registered ONNX candidate changed")
    fit = verified_json(fit_roster_path, plan["fit_roster_sha256"])
    if fit["identity_namespace"] != plan["identity_namespace"] or not fit["coverage_receipt"]:
        raise ValueError("fit and model-selection identities must use the same verified roster")
    fit_ids = identities(fit["speakers"])
    if not fit_ids:
        raise ValueError("the model-fit identity roster is empty")
    private = root / "runs" / "release-quality" / plan["panel"] / plan["interface"]
    private.mkdir(parents=True, exist_ok=True)
    panel = {key: plan[key] for key in ("contract", "languages", "identity_namespace",
                                       "calibration_manifest_sha256", "test_manifest_sha256")}
    panel_path = private.parent / "panel.json"
    if role == "calibration":
        write_once(panel_path, panel)
    elif json.loads(panel_path.read_text()) != panel:
        raise ValueError("the registered panel changed across candidate interfaces")
    lock_path = private / "calibration.json"
    attempt = {"contract": CONTRACT, "plan_sha256": plan_sha256, "onnxruntime": ort.__version__}
    lock = None
    if role == "test":
        lock = verified_json(lock_path, calibration_sha256)
        if lock["attempt"] != attempt:
            raise ValueError("test requires an unchanged calibration lock for this artifact and runtime")
    elif lock_path.exists():
        registered = json.loads(lock_path.read_text())
        if registered["attempt"] != attempt:
            raise ValueError("an incompatible release attempt already exists")

    cohort = verified_json(cohort_path, plan[f"{role}_manifest_sha256"])
    indexes, membership = validate_manifest(cohort, plan, role)
    if lock is None:
        ensure_disjoint(fit_ids, membership)
    else:
        ensure_disjoint(fit_ids, [set(values) for values in lock["private_membership"]], membership)

    if digest(audio_path) != cohort["audio_sha256"]:
        raise ValueError("the sealed telephone PCM collection changed")
    pcm = np.load(audio_path, mmap_mode="r", allow_pickle=False)
    if pcm.shape != (len(cohort["pauses"]), 48_000) or pcm.dtype != np.dtype("<i2"):
        raise ValueError("expected one three-second int16 window per pause")
    combined = plan["interface"] == "audio-text"
    partials = None
    if combined:
        if text_path is None or digest(text_path) != cohort["text_sha256"]:
            raise ValueError("the sealed decision-time text collection changed")
        partials = np.load(text_path, mmap_mode="r", allow_pickle=False)
        if partials.shape != (len(pcm), 128) or partials.dtype != np.uint8:
            raise ValueError("expected one 128-byte packed partial per pause")
    elif text_path is not None:
        raise ValueError("the registered audio-only artifact takes no text")
    graph = onnx.load_model_from_string(artifact_bytes)
    tensors = onnx.external_data_helper._get_all_tensors(graph)
    if any(onnx.external_data_helper.uses_external_data(value) for value in tensors):
        raise ValueError("the frozen ONNX candidate must embed its weights")
    session = _session(artifact_bytes)
    expected_inputs = {"audio": ("tensor(float)", 48_000), "silence_ms": ("tensor(float)", 1)}
    if combined:
        expected_inputs["text_bytes"] = ("tensor(int32)", 128)
    def batch_one(item):
        return len(item.shape) == 2 and (not isinstance(item.shape[0], int) or item.shape[0] == 1)

    actual_inputs = {item.name: (item.type, item.shape[1]) for item in session.get_inputs() if batch_one(item)}
    outputs = session.get_outputs()
    if (actual_inputs != expected_inputs or len(session.get_inputs()) != len(expected_inputs) or len(outputs) != 1
            or outputs[0].name != "p_complete" or outputs[0].type != "tensor(float)"
            or not batch_one(outputs[0]) or outputs[0].shape[1] != 1):
        raise ValueError("ONNX graph does not implement the registered release interface")

    modes = ("combined", "masked_text", "masked_audio") if combined else ("audio",)
    scores = defaultdict(list)
    truth = np.asarray([row["label"] == "shift" for row in cohort["pauses"]])
    elapsed = np.full((1, 1), DEADLINE, dtype=np.float32)
    meter = Progress(f"release-quality-{plan['panel']}-{role}", len(pcm), "pauses")
    try:
        for index, (row, clip) in enumerate(zip(cohort["pauses"], pcm, strict=True)):
            if hashlib.sha256(clip.tobytes()).hexdigest() != row["pcm_sha256"]:
                raise ValueError("pause order or rendered PCM digest changed")
            audio = clip[None].astype(np.float32) / 32768
            if combined:
                text = partials[index:index + 1].astype(np.int32)
                for mode, signal, tokens in (("combined", audio, text), ("masked_text", audio, np.zeros_like(text)),
                                             ("masked_audio", np.zeros_like(audio), text)):
                    value, = session.run(["p_complete"], {"audio": signal, "silence_ms": elapsed, "text_bytes": tokens})
                    scores[mode].append(float(value.item()))
            else:
                value, = _predict(session, audio, elapsed)
                scores["audio"].append(float(value))
            meter.advance(1)
        scores = {mode: np.asarray(scores[mode], dtype=np.float64) for mode in modes}
        if any(not np.all(np.isfinite(values) & (0 <= values) & (values <= 1)) for values in scores.values()):
            raise ValueError("release artifact returned invalid probabilities")
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()

    counts = {language: {"holds": int((~truth[rows]).sum()), "shifts": int(truth[rows].sum()),
                         "calls": len({cohort["pauses"][index]["call"] for index in rows}),
                         "focal_speakers": len({cohort["pauses"][index]["speaker"] for index in rows})}
              for language, rows in indexes.items()}
    if lock is None:
        def cutoff(score, labels):
            boundary = balanced_threshold(score, labels)
            return None if boundary > score.max() else boundary

        receipt = {"attempt": attempt, "counts": counts, "threshold_rule": "max Youden; highest tied cutoff; null always holds",
                   "private_membership": [sorted(value) for value in membership],
                   "thresholds": {mode: {lang: cutoff(values[rows], truth[rows]) for lang, rows in indexes.items()}
                                  for mode, values in scores.items() if mode != "masked_text"}}
        write_once(lock_path, receipt)
        return {"status": "calibrated", "calibration_sha256": digest(lock_path)}
    primary_mode = modes[0]
    results = {}
    primary_ba = {}
    for mode, values in scores.items():
        thresholds = lock["thresholds"][primary_mode] if mode == "masked_text" else lock["thresholds"][mode]
        languages = {}
        for lang, rows in indexes.items():
            threshold = thresholds[lang]
            effective = np.inf if threshold is None else threshold
            labels = truth[rows]
            predicted = values[rows] >= effective
            tp, fp = int((labels & predicted).sum()), int((~labels & predicted).sum())
            languages[lang] = {**counts[lang], "threshold": threshold, "correct_shifts": tp, "false_completes": fp,
                               **evaluate(values[rows], labels, effective)}
            if mode == primary_mode:
                holds, shifts = counts[lang]["holds"], counts[lang]["shifts"]
                primary_ba[lang] = (Fraction(tp, shifts) + Fraction(holds - fp, holds)) / 2
        macro = {metric: float(np.mean([cell[metric] for cell in languages.values()])) for metric in METRICS}
        results[mode] = {"macro": macro, "languages": languages}
    passed = (sum(primary_ba.values()) / len(primary_ba) >= Fraction(4, 5)
              and min(primary_ba.values()) >= Fraction(7, 10))
    report = {"attempt": attempt, "artifact_sha256": plan["artifact_sha256"], "decision_ms": DEADLINE,
              "calibration_sha256": calibration_sha256, "calibration_counts": lock["counts"], "models": results,
              "balanced_accuracy_pass": bool(passed), "latency_ms": None,
              "scope": "held-out telephone point estimate conditional on supplied adjudication and rosters"}
    destination = root / "reports" / "release-quality" / plan["panel"] / plan["interface"]
    destination.mkdir(parents=True, exist_ok=True)
    write_once(destination / "test.json", report)
    return report
