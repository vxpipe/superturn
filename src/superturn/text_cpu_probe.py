"""Stage pinned text backbones and measure CPU inference on synthetic inputs."""

import argparse
from contextlib import contextmanager
import json
from math import ceil
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns, sleep

import numpy as np

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root
from .turn_text_train import digest

E5 = ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "MIT")
MULTI = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
         "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", "Apache-2.0")
ENGLISH = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "Apache-2.0")
CAUSAL = ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "Apache-2.0")
SOURCES = {"me5": E5, "minilm-multi": MULTI, "minilm-en": ENGLISH, "smollm2": CAUSAL}
LENGTHS = (16, 32, 64, 128)
LANGUAGES = ("en", "hi", "ta", "te", "ml", "kn", "zh", "ja")
E5_DEPTHS = {"me5-1": 1, "me5-6": 6}


def stage(root, names):
    from huggingface_hub import hf_hub_download

    meter = Progress("text-cpu-stage", 3 * len(names), "files")
    for name in names:
        repo, revision, _ = SOURCES[name]
        target = root / name
        quantized = ("model_int8.onnx" if name == "smollm2" else
                     "model_qint8_avx512_vnni.onnx" if name == "me5" else "model_quint8_avx2.onnx")
        for filename in ("tokenizer.json", "onnx/model.onnx", f"onnx/{quantized}"):
            hf_hub_download(repo_id=repo, revision=revision, filename=filename, local_dir=target, token=False)
            meter.advance(1)
    meter.finish()


def cpus():
    affinity = sorted(os.sched_getaffinity(0))
    quota_file = Path("/sys/fs/cgroup/cpu.max")
    quota_record = quota_file.read_text().strip() if quota_file.is_file() else None
    if quota_record:
        quota, period = quota_record.split()
        allowed = len(affinity) if quota == "max" else max(1, min(len(affinity), int(quota) // int(period)))
    else:
        allowed = len(affinity)
    cores = {}
    siblings_by_cpu = {}
    for cpu in affinity:
        sibling_file = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        siblings = sibling_file.read_text().strip() if sibling_file.is_file() else str(cpu)
        cores.setdefault(siblings, cpu)
        siblings_by_cpu[cpu] = siblings
    placement = list(cores.values())[:allowed]
    cpuinfo = Path("/proc/cpuinfo").read_text().split("\n\n")[placement[0]]
    fields = dict(line.split(":", 1) for line in cpuinfo.splitlines() if ":" in line)
    fields = {key.strip(): value.strip() for key, value in fields.items()}
    return affinity, placement, {"model": fields.get("model name"), "cpu_max": quota_record,
                                 "allowed_logical_cpus": affinity, "selected_cpus": placement,
                                 "benchmark_cpu": placement[0],
                                 "reported_thread_siblings": {cpu: siblings_by_cpu[cpu] for cpu in placement},
                                 "flags": [flag for flag in fields.get("flags", "").split()
                                           if flag.startswith(("avx", "vnni"))]}


@contextmanager
def busy_cores(placement):
    workers = []
    receipts = {"pinned_workers": len(placement)}

    def cgroup():
        path = Path("/sys/fs/cgroup/cpu.stat")
        return dict((key, int(value)) for key, value in (line.split() for line in path.read_text().splitlines())) \
            if path.is_file() else {}

    def worker_ticks(worker):
        values = Path(f"/proc/{worker.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(values[11]) + int(values[12])

    try:
        for cpu in placement:
            program = f"import os\nos.sched_setaffinity(0, {{{cpu}}})\nn=1\nwhile True: n=(n*17+3)&65535"
            workers.append(subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL))
        if workers:
            sleep(.5)
        if any(worker.poll() is not None for worker in workers):
            raise RuntimeError("a pinned background CPU worker exited")
        initial = [worker_ticks(worker) for worker in workers]
        start = cgroup()
        yield receipts
        end = cgroup()
        ticks = [worker_ticks(worker) - previous for worker, previous in zip(workers, initial, strict=True)]
        receipts["workers_with_cpu_ticks"] = sum(tick > 0 for tick in ticks)
        receipts["cgroup_throttled_usec"] = end.get("throttled_usec", 0) - start.get("throttled_usec", 0)
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait()


def get_session(model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(source, sess_options=options, providers=["CPUExecutionProvider"])


def graph_paths(root, name):
    folder = root / name / "onnx"
    quant = ("model_int8.onnx" if name == "smollm2" else "model_qint8_avx512_vnni.onnx" if name == "me5"
             else "model_quint8_avx2.onnx")
    return {"fp32": folder / "model.onnx", "int8": folder / quant}


def e5_vocabulary_size(root, corpus, report_name):
    import onnx
    from onnx import numpy_helper
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "me5" / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    with corpus.open() as stream:
        records = [json.loads(line) for line in stream]
    if not records:
        raise ValueError("vocabulary sizing needs at least one synthetic row")
    text_ids = {token for row in records for token in pack_encoder(tokenizer, row["text"], 128, "me5").ids}
    tokenizer_config = json.loads(tokenizer.to_str())
    text_ids.update(token["id"] for token in tokenizer_config["added_tokens"] if token["special"])
    keep = sorted(text_ids)
    receipt = {
        "scope": "Embedding size measurement only. The temporary graph has no deployment tokenizer.",
        "source": E5[:2], "corpus_sha256": digest(corpus), "corpus_rows": len(records),
        "languages": sorted({row["language"] for row in records}), "retained_token_rows": len(keep), "graphs": {},
    }
    meter = Progress("text-cpu-vocabulary-size", 2, "graphs")
    with TemporaryDirectory(dir=root) as temporary:
        destination = Path(temporary) / "model.onnx"
        for precision, source in graph_paths(root, "me5").items():
            model = onnx.load(source)
            table, = [tensor for tensor in model.graph.initializer
                      if "word_embeddings" in tensor.name and len(tensor.dims) == 2]
            old_rows, width = table.dims
            if keep[-1] >= old_rows or len(keep) >= old_rows:
                raise ValueError("synthetic tokenizer IDs do not fit the staged embedding table")
            companions = [tensor for tensor in model.graph.initializer
                          if tensor.name.startswith(table.name.removesuffix("_quantized"))
                          and tensor.name != table.name and list(tensor.dims) == [old_rows]]
            for tensor in [table, *companions]:
                tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor)[keep], tensor.name))
            onnx.save(model, destination)
            onnx.checker.check_model(str(destination))
            receipt["graphs"][precision] = {
                "original_bytes": source.stat().st_size, "pruned_bytes": destination.stat().st_size,
                "original_token_rows": old_rows, "embedding_width": width,
                "original_sha256": digest(source), "original_artifact": str(source.relative_to(root)),
            }
            meter.advance(1)
    meter.finish()
    _write_json(root / report_name, receipt)
    print(json.dumps(receipt, indent=2))


def e5_block_paths(root, name):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = graph_paths(root, "me5")["fp32"]
    depth = E5_DEPTHS[name]
    target = root / name
    target.mkdir(exist_ok=True)
    fp, quant = target / "model.onnx", target / "model_uint8.onnx"
    parity_path = target / "probe-head-parity.json"
    reference = None
    valid = False
    if fp.is_file():
        try:
            onnx.checker.check_model(str(fp))
            valid = depth != 1 or (parity_path.is_file() and quant.is_file() and
                                  json.loads(parity_path.read_text())["artifacts"] == {
                                      "base": digest(source), "fp32": digest(fp), "int8": digest(quant),
                                  })
        except onnx.checker.ValidationError:
            pass
    if not valid:
        model = onnx.load(source)
        layer = [node for node in model.graph.node
                 if node.name == f"/encoder/layer.{depth - 1}/output/LayerNorm/Add_1"]
        if len(layer) != 1:
            raise RuntimeError(f"could not locate mE5 block {depth} output")
        output = layer[0].output[0]
        info = next((item for item in model.graph.value_info if item.name == output), None)
        if info is None:
            info = model.graph.value_info.add()
        info.CopyFrom(model.graph.output[0])
        info.name = output
        extracted = onnx.utils.Extractor(model).extract_model([item.name for item in model.graph.input], [output])
        if depth == 1:
            reference = extracted.SerializeToString()
            add_e5_scalar_head(extracted)
        temporary = fp.with_name("model.partial.onnx")
        onnx.save(extracted, temporary)
        onnx.checker.check_model(str(temporary))
        temporary.replace(fp)
    if not quant.is_file() or reference is not None:
        temporary = quant.with_name("model_uint8.partial.onnx")
        quantize_dynamic(fp, temporary, per_channel=True, weight_type=QuantType.QUInt8,
                         op_types_to_quantize=["MatMul", "Gather"])
        onnx.checker.check_model(str(temporary))
        temporary.replace(quant)
    if reference is not None:
        parity = verify_e5_scalar_head(root, reference, fp, quant)
        parity["artifacts"] = {"base": digest(source), "fp32": digest(fp), "int8": digest(quant)}
        _write_json(parity_path, parity)
    return {"fp32": fp, "int8": quant}


def e5_probe_head_weights(width):
    random = np.random.default_rng(19)
    return (random.normal(0, (2 * width) ** -.5, (2 * width, 128)).astype(np.float32),
            random.normal(0, 128 ** -.5, (128, 1)).astype(np.float32))


def add_e5_scalar_head(model):
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    state = graph.output[0]
    width = state.type.tensor_type.shape.dim[-1].dim_value
    if width <= 0:
        raise ValueError("expected a fixed encoder hidden width")
    first, second = e5_probe_head_weights(width)
    opset = next(item.version for item in model.opset_import if item.domain == "")
    sum_input, sum_attrs = (["probe_axis"], {"keepdims": 0}) if opset >= 13 else ([], {"keepdims": 0, "axes": [1]})
    graph.input.extend([
        helper.make_tensor_value_info("caller_mask", TensorProto.FLOAT, ["batch_size", "sequence_length", 1]),
        helper.make_tensor_value_info("last_caller", TensorProto.INT64, ["batch_size", 2]),
    ])
    graph.initializer.extend([numpy_helper.from_array(first, "probe_first"), numpy_helper.from_array(second, "probe_second")])
    if sum_input:
        graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "probe_axis"))
    graph.node.extend([helper.make_node(op, inputs, [output], name=output, **attrs)
                       for op, inputs, output, attrs in (
        ("GatherND", [state.name, "last_caller"], "probe_last", {}),
        ("Mul", [state.name, "caller_mask"], "probe_masked", {}),
        ("ReduceSum", ["probe_masked", *sum_input], "probe_sum", sum_attrs),
        ("ReduceSum", ["caller_mask", *sum_input], "probe_count", sum_attrs),
        ("Div", ["probe_sum", "probe_count"], "probe_mean", {}),
        ("Concat", ["probe_last", "probe_mean"], "probe_summary", {"axis": 1}),
        ("MatMul", ["probe_summary", "probe_first"], "probe_hidden", {}),
        ("Relu", ["probe_hidden"], "probe_activation", {}),
        ("MatMul", ["probe_activation", "probe_second"], "probe_logit", {}),
        ("Sigmoid", ["probe_logit"], "score", {}),
    )])
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch_size", 1]))


def verify_e5_scalar_head(root, reference, fp32, int8):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "me5" / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    base, fp, quantized = (get_session(model) for model in (reference, fp32, int8))
    base_inputs = {item.name for item in base.get_inputs()}
    head_inputs = {item.name for item in fp.get_inputs()}
    fixtures = (
        "---\ncan you tell me when the next bus is due",
        "which appointment works for you\nthe one next week after lunch please",
        "आपको कौन सा समय चाहिए\nमैं अगले शुक्रवार को दोपहर के बाद आ सकता हूँ",
        "你想约什么时候\n下周五下午比较方便",
    )
    differences = {"fp32": [], "int8": []}
    for text in fixtures:
        feed, caller = encoder_feed(tokenizer, text, 128, "me5-1", head_inputs)
        hidden = base.run(None, {key: feed[key] for key in base_inputs})[0][0]
        first, second = e5_probe_head_weights(hidden.shape[-1])
        pooled = np.r_[hidden[caller[-1]], hidden[caller].mean(axis=0)]
        logit = (np.maximum(pooled @ first, 0.) @ second).item()
        expected = 1. / (1. + np.exp(-logit))
        for key, session in (("fp32", fp), ("int8", quantized)):
            actual = session.run(None, feed)[0].item()
            if not np.isfinite(actual):
                raise ValueError("the scalar head produced a non-finite probability")
            differences[key].append(abs(actual - expected))
    maximum = {key: max(values) for key, values in differences.items()}
    if maximum["fp32"] > 2e-5 or maximum["int8"] > .08:
        raise ValueError(f"E5 scalar probe failed score parity: {maximum}")
    return {"fixture_count": len(fixtures), "maximum_absolute_score_delta": maximum,
            "dummy_head_only": True, "fp32_tolerance": 2e-5, "int8_tolerance": .08}


def causal_paths(root):
    import onnx
    from onnx import TensorProto, helper

    paths = graph_paths(root, "smollm2")
    result = {kind: path.with_name(f"hidden-{kind}.onnx") for kind, path in paths.items()}
    if all(path.is_file() for path in result.values()):
        return result
    fp32 = onnx.load(paths["fp32"])
    projection, = [node for node in fp32.graph.node if "logits" in node.output and node.op_type == "MatMul"]
    hidden = projection.input[0]
    width = next(node.dims[1] for node in fp32.graph.initializer if node.name == "model.embed_tokens.weight")
    for kind, destination in result.items():
        if destination.is_file():
            continue
        model = fp32 if kind == "fp32" else onnx.load(paths[kind])
        if not any(hidden in node.output for node in model.graph.node):
            raise ValueError("vendor INT8 graph exposes a different hidden-state boundary")
        if not any(item.name == hidden for item in model.graph.value_info):
            model.graph.value_info.append(helper.make_tensor_value_info(
                hidden, TensorProto.FLOAT, ["batch_size", "sequence_length", width]))
        present = [item.name for item in model.graph.output if item.name.startswith("present.")]
        inputs = [item.name for item in model.graph.input]
        cut = onnx.utils.Extractor(model).extract_model(inputs, [hidden, *present])
        onnx.save(cut, destination)
    return result


def number_stats(lengths):
    return {"n": len(lengths), **{f"p{p}": float(np.percentile(lengths, p)) for p in (50, 90, 99)},
            "over_64": sum(x > 64 for x in lengths) / len(lengths),
            "over_128": sum(x > 128 for x in lengths) / len(lengths)} if lengths else {"n": 0}


def token_needs(tokenizer, records, name):
    report = {}
    prefix = "query:" if name.startswith("me5") else ""
    special = (len(tokenizer.encode("\n", add_special_tokens=False).ids)
               if name == "smollm2" else tokenizer.num_special_tokens_to_add(is_pair=True))
    unknown_id = next((ident for token in ("[UNK]", "<unk>")
                       if (ident := tokenizer.token_to_id(token)) is not None), None)
    for language in LANGUAGES:
        caller, short_prior, full = [], [], []
        unknown = 0
        for row in records:
            if row["language"] != language:
                continue
            previous, current = row["text"].split("\n")
            current_ids = tokenizer.encode(current, add_special_tokens=False).ids
            prior_ids = tokenizer.encode(previous, add_special_tokens=False).ids
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
            caller.append(len(current_ids))
            if unknown_id is not None:
                unknown += current_ids.count(unknown_id)
            short_prior.append(len(current_ids) + min(16, len(prior_ids)) + len(prefix_ids) + special)
            if name == "smollm2":
                full.append(len(tokenizer.encode(previous + "\n" + current).ids))
            else:
                fixed = tokenizer.encode(encoder_prefix(previous, name), add_special_tokens=False)
                prior = len(prior_ids) if previous != "---" else 0
                full.append(len(fixed.ids) + prior + len(current_ids) + special)
        report[language] = {"caller": number_stats(caller), "caller_plus_16_prior": number_stats(short_prior),
                            "full_two_lines": number_stats(full),
                            "caller_unknown_token_share": unknown / sum(caller) if caller and sum(caller) else None}
    return report


def encoder_screen(detail, languages, lengths, idle, loaded, load, expected_workers, *, timed_scope=None, serving_cap=None):
    counts = {language: detail["token_needs"][language]["full_two_lines"] for language in languages}
    missing_languages = [language for language, count in counts.items() if not count["n"]]
    p90 = max((count["p90"] for count in counts.values() if count["n"]), default=None)
    retained_p90 = min(p90, serving_cap) if p90 is not None and serving_cap is not None else p90
    bracket = next((length for length in lengths if retained_p90 is not None and length >= retained_p90), None)
    key = str(bracket)
    gates = {
        "all_screen_languages_represented": not missing_languages,
        "measured_at_or_above_required_tokens": bracket is not None,
        "idle_p99_under_5_ms": bracket is not None
        and idle[key]["synthetic_text_to_scalar"]["p99_ms"] < 5,
        "loaded_p99_at_most_10_ms": bracket is not None and loaded is not None
        and loaded[key]["synthetic_text_to_scalar"]["p99_ms"] <= 10,
        "all_pinned_load_workers_ran": loaded is not None
        and load["workers_with_cpu_ticks"] == expected_workers,
    }
    return {
        "languages": languages, "missing_languages": missing_languages, "largest_p90": p90,
        "conservative_measured_bucket": bracket, "p90_exceeds_128": p90 is not None and p90 > LENGTHS[-1],
        "serving_token_cap": serving_cap,
        "timed_input_scope": timed_scope or "English synthetic strings at matching token lengths",
        "checks": gates, "passed": all(gates.values()),
    }


def timed(call, warmups, iterations):
    for i in range(warmups):
        call(i)
    durations = []
    for i in range(iterations):
        start = perf_counter_ns()
        call(i)
        durations.append((perf_counter_ns() - start) / 1_000_000)
    return timing_stats(durations)


def timing_stats(durations):
    return dict(zip(("p50_ms", "p99_ms"), np.percentile(durations, [50, 99]).tolist(), strict=True))


def encoder_prefix(previous, name):
    prefix = "query:" if name.startswith("me5") else ""
    return " ".join(part for part in (prefix, "---" if previous == "---" else "") if part)


def pack_encoder(tokenizer, text, limit, name):
    from tokenizers import Encoding

    previous, current = (" ".join(line.split()) for line in text.split("\n"))
    fixed = tokenizer.encode(encoder_prefix(previous, name), add_special_tokens=False)
    prior = tokenizer.encode(previous if previous != "---" else "", add_special_tokens=False)
    caller = tokenizer.encode(current, add_special_tokens=False)
    remaining = limit - tokenizer.num_special_tokens_to_add(is_pair=True) - len(fixed.ids)
    if remaining <= 0:
        raise ValueError("model token budget cannot represent the two inputs")
    caller.truncate(remaining, direction="left")
    extra = remaining - len(caller.ids)
    if extra:
        prior.truncate(extra, direction="left")
    else:
        prior = tokenizer.encode("", add_special_tokens=False)
    return tokenizer.post_process(Encoding.merge([fixed, prior]), caller, add_special_tokens=True)


def native_p90_inputs(tokenizer, records, name, language, token_stats, *, serving_cap=None):
    if not token_stats["n"]:
        raise ValueError(f"no synthetic examples for {language}")
    original_length = ceil(token_stats["p90"])
    length = min(original_length, serving_cap) if serving_cap is not None else original_length
    if not 8 <= length <= LENGTHS[-1]:
        raise ValueError(f"{language} p90 of {length} tokens is outside the measured input range")
    by_length = []
    for row in records:
        if row["language"] != language:
            continue
        text = row["text"]
        packed = pack_encoder(tokenizer, text, LENGTHS[-1], name)
        if len(packed.ids) >= min(original_length, LENGTHS[-1]) and 1 in packed.sequence_ids:
            by_length.append((len(packed.ids), text))
    # Use the nearest distinct examples without stretching or repeating caller text.
    cases = list(dict.fromkeys(text for _, text in sorted(by_length)))[:32]
    if not cases:
        raise ValueError(f"no real-length synthetic fixtures for {language} at {length} tokens")
    return length, cases


def truncation_counts(tokenizer, records, name, language, limit):
    from collections import Counter

    counts = Counter()
    prefix = "query: " if name.startswith("me5") else ""
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False).ids)
    for row in records:
        if row["language"] != language:
            continue
        previous, current = row["text"].split("\n")
        packed = pack_encoder(tokenizer, row["text"], limit, name)
        roles = packed.sequence_ids
        caller_before = len(tokenizer.encode(current, add_special_tokens=False).ids)
        caller_after = roles.count(1)
        counts["rows"] += 1
        counts["caller_trimmed"] += caller_after < caller_before
        if previous != "---":
            prior_before = len(tokenizer.encode(previous, add_special_tokens=False).ids)
            prior_after = max(0, roles.count(0) - prefix_tokens)
            counts["rows_with_prior"] += 1
            counts["prior_trimmed"] += prior_after < prior_before
            counts["all_prior_tokens_dropped"] += prior_after == 0
    return dict(counts)


def encoder_feed(tokenizer, text, limit, name, inputs):
    packed = pack_encoder(tokenizer, text, limit, name)
    feed = {"input_ids": np.array([packed.ids], dtype=np.int64),
            "attention_mask": np.array([packed.attention_mask], dtype=np.int64)}
    if "token_type_ids" in inputs:
        feed["token_type_ids"] = np.array([packed.type_ids], dtype=np.int64)
    caller = np.flatnonzero(np.asarray([role == 1 for role in packed.sequence_ids]))
    if not caller.size:
        raise ValueError("expected at least one caller token")
    if "caller_mask" in inputs:
        mask = np.zeros((1, len(packed.ids), 1), dtype=np.float32)
        mask[0, caller, 0] = 1
        feed["caller_mask"] = mask
        feed["last_caller"] = np.array([[0, caller[-1]]], dtype=np.int64)
    return feed, caller


def encoder_trials(session, tokenizer, texts, warmups, iterations, name, lengths=LENGTHS, *, exact=None):
    inputs = {item.name for item in session.get_inputs()}
    has_scalar_head = "caller_mask" in inputs
    result = {}
    for length in lengths:
        fixtures = exact
        if fixtures is None:
            fixtures = []
            for text in texts:
                previous, caller = text.split("\n")
                words = caller.split() * length
                for size in range(1, len(words) + 1):
                    value = previous + "\n" + " ".join(words[:size])
                    if len(pack_encoder(tokenizer, value, length, name).ids) == length:
                        fixtures.append(value)
                        break

        def prepare(text):
            feed, caller_indices = encoder_feed(tokenizer, text, length, name, inputs)
            if feed["input_ids"].shape[1] != length:
                raise ValueError("synthetic fixture did not use the requested length")
            return feed, caller_indices

        def predict(prepared):
            feed, caller_indices = prepared
            hidden, *_ = session.run(None, feed)
            if has_scalar_head:
                return float(hidden[0, 0])
            if hidden.ndim != 3 or hidden.shape[:2] != feed["input_ids"].shape:
                raise ValueError("encoder did not return token states")
            # A representative scalar readout. The trained classifier is measured after L2.
            states = hidden[0, caller_indices]
            logit = float((states[-1] + states.mean(axis=0))[0])
            return 1 / (1 + np.exp(-logit))

        for i in range(warmups):
            predict(prepare(fixtures[i % len(fixtures)]))
        kernel, end_to_end = [], []
        for i in range(iterations):
            begin = perf_counter_ns()
            prepared = prepare(fixtures[i % len(fixtures)])
            ready = perf_counter_ns()
            predict(prepared)
            done = perf_counter_ns()
            kernel.append((done - ready) / 1_000_000)
            end_to_end.append((done - begin) / 1_000_000)
        result[str(length)] = {"model_and_readout": timing_stats(kernel),
                               "synthetic_text_to_scalar": timing_stats(end_to_end)}
    return result


def causal_runner(session):
    inputs = {item.name: item for item in session.get_inputs()}
    past = [name for name in inputs if name.startswith("past_key_values.")]
    outputs = [item.name for item in session.get_outputs()]
    states = {name: "present." + name.removeprefix("past_key_values.") for name in past}
    if (not past or set(states.values()) != set(outputs[1:])
            or set(inputs) - set(past) != {"input_ids", "attention_mask", "position_ids"}):
        raise ValueError("unsupported causal cache graph interface")
    blank = {}
    for name in past:
        entry = inputs[name]
        if entry.type != "tensor(float)" or len(entry.shape) != 4:
            raise ValueError("expected batch/head/history/channel float32 cache")
        blank[name] = np.empty((1, entry.shape[1], 0, entry.shape[3]), np.float32)

    def forward(tokens, prefix=None):
        cache = blank if prefix is None else prefix
        existing = next(iter(cache.values())).shape[-2]
        count = len(tokens)
        feed = {"input_ids": np.array([tokens], dtype=np.int64),
                "attention_mask": np.ones((1, existing + count), dtype=np.int64),
                "position_ids": np.arange(existing, existing + count, dtype=np.int64).reshape(1, -1), **cache}
        values = dict(zip(outputs, session.run(outputs, feed), strict=True))
        hidden = values[outputs[0]][0, -1]
        return hidden, {name: values[present] for name, present in states.items()}

    return forward


def causal_trials(session, tokenizer, texts, warmups, iterations, precision):
    forward = causal_runner(session)
    token_sets = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    sequences = [(tokens * (136 // len(tokens) + 1))[:136] for tokens in token_sets]
    conditions = {}
    parity = {}

    def classify(ids, cache=None):
        hidden, _ = forward(ids, cache)
        return 1 / (1 + np.exp(-float(hidden[0])))

    for size in LENGTHS:
        conditions[f"cold_{size}"] = timed(lambda i: classify(sequences[i % len(sequences)][:size]), warmups, iterations)
    for size in (64, 128):
        caches = [forward(tokens[:size])[1] for tokens in sequences]
        for count in (4, 8):
            key = f"cached_{size}_append_{count}"
            differences = []
            for tokens, cache in zip(sequences, caches, strict=True):
                expected, _ = forward(tokens[:size + count])
                actual, _ = forward(tokens[size:size + count], cache)
                differences.append(float(np.max(np.abs(expected - actual))))
                if precision == "fp32" and not np.allclose(expected, actual, rtol=.003, atol=.003):
                    raise ValueError("float32 cached decode does not match full recomputation")
            parity[key] = max(differences)
            conditions[key] = timed(lambda i: classify(sequences[i % len(sequences)][size:size + count],
                                                       caches[i % len(caches)]), warmups, iterations)
    return {"model_and_scalar": conditions, "append_vs_full_max_absolute_hidden_delta": parity,
            "rolling_window_refill_reference": "cold_128",
            "over_128_token_context_stress_only": ["cached_128_append_4", "cached_128_append_8"]}


def run(root, names, corpus, warmups, iterations, report_name, *, lengths=LENGTHS, precisions=("fp32", "int8"),
        native_p90=None, serving_cap=None):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    records = [json.loads(line) for line in corpus.read_text().splitlines()]
    fixture = ["Can you confirm the booking reference?\nI have it in front of me.",
               "Let me check the time.\nFriday after lunch should work for us.",
               "We need to change our booking.\nThe one for two people."]
    affinity, placement, hardware = cpus()
    result = {"scope": "batch-one synthetic CPU screening, no quality scoring", "hardware": hardware,
              "ort": ort.__version__, "warmups": warmups, "iterations": iterations,
              "lengths": lengths if not native_p90 else None,
              "cpu_execution_provider_only": True, "ort_intra_threads": 1, "ort_inter_threads": 1,
              "language_rows": dict((language, sum(row["language"] == language for row in records))
                                    for language in LANGUAGES), "encoders": {}}
    meter = Progress("text-cpu-encoders", 2 * len(precisions) * len(names), "model conditions")
    os.sched_setaffinity(0, {placement[0]})
    try:
        for name in names:
            original = "me5" if name in E5_DEPTHS else name
            tokenizer = Tokenizer.from_file(str(root / original / "tokenizer.json"))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            detail = {"license": SOURCES[original][2], "source": SOURCES[original][:2],
                      "token_needs": token_needs(tokenizer, records, original)}
            model_lengths, exact = lengths, None
            if native_p90:
                stats = detail["token_needs"][native_p90]["full_two_lines"]
                length, exact = native_p90_inputs(tokenizer, records, name, native_p90, stats, serving_cap=serving_cap)
                model_lengths = (length,)
                detail["native_p90_fixtures"] = {"language": native_p90, "count": len(exact), "length": length,
                                                 "synthetic_corpus_sha256": digest(corpus)}
                if serving_cap is not None:
                    detail["serving_cap_truncation"] = truncation_counts(tokenizer, records, name, native_p90, serving_cap)
            result["encoders"][name] = detail
            paths = (causal_paths(root) if name == "smollm2" else e5_block_paths(root, name) if name in E5_DEPTHS
                     else graph_paths(root, name))
            if name == "me5-1":
                detail["scalar_head_parity"] = json.loads((root / name / "probe-head-parity.json").read_text())
            for precision, path in paths.items():
                if precision not in precisions:
                    continue
                session = get_session(path)
                measure = (lambda: causal_trials(session, tokenizer, fixture, warmups, iterations, precision)
                           if name == "smollm2" else encoder_trials(session, tokenizer, fixture, warmups,
                                                                   iterations, name, model_lengths, exact=exact))
                idle = measure()
                meter.advance(1)
                with busy_cores(placement[1:]) as load:
                    loaded = measure() if len(placement) > 1 else None
                meter.advance(1)
                detail[precision] = {"bytes": path.stat().st_size, "sha256": digest(path),
                                     "onnx_artifact": str(path.relative_to(root)),
                                     "idle": idle, "loaded": loaded, "load_verification": load}
                if precision == "int8" and name != "smollm2":
                    languages = (native_p90,) if native_p90 else ("en",) if name == "minilm-en" else LANGUAGES
                    detail["screen_at_p90_bucket"] = encoder_screen(
                        detail, languages, model_lengths, idle, loaded, load, expected_workers=len(placement) - 1,
                        timed_scope=f"{native_p90} synthetic inputs at rounded-up p90 and declared cap" if native_p90 else None,
                        serving_cap=serving_cap,
                    )
                _write_json(root / report_name, result)
    finally:
        os.sched_setaffinity(0, affinity)
        meter.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage", action="store_true")
    action.add_argument("--e5-vocabulary-size", action="store_true", help="measure temporary pruned E5 graphs only")
    parser.add_argument("--models", nargs="+", choices=(*SOURCES, *E5_DEPTHS), required=True)
    parser.add_argument("--artifacts", type=Path, default=data_root() / "cache/text-cpu-l1")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/turn-completion-v1/train.jsonl"))
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--report-name", default="encoder-latency.json")
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "int8"), default=("fp32", "int8"))
    parser.add_argument("--native-p90", choices=LANGUAGES, help="time existing same-language inputs at their p90 length")
    parser.add_argument("--cap-tokens", type=int, help="time capped inputs using the original native p90 fixtures")
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name:
        parser.error("use a plain report filename")
    if args.stage:
        stage(args.artifacts, args.models)
    elif args.e5_vocabulary_size:
        if args.models != ["me5"]:
            parser.error("the size probe requires --models me5")
        e5_vocabulary_size(args.artifacts, args.corpus, args.report_name)
    else:
        if args.iterations <= 0 or args.warmups < 0:
            parser.error("use valid timing counts")
        lengths = tuple(sorted(set(args.lengths)))
        if lengths[0] < 8 or lengths[-1] > LENGTHS[-1] or ("smollm2" in args.models and lengths != LENGTHS):
            parser.error("encoder lengths must be 8–128; the causal lengths are fixed")
        if args.native_p90 and ("smollm2" in args.models or
                                ("minilm-en" in args.models and args.native_p90 != "en")):
            parser.error("native p90 requires an encoder for the requested language")
        if args.cap_tokens is not None and (not args.native_p90 or not 8 <= args.cap_tokens <= LENGTHS[-1]):
            parser.error("a declared cap requires native p90 and a length from 8 to 128")
        run(args.artifacts, args.models, args.corpus, args.warmups, args.iterations, args.report_name,
            lengths=lengths, precisions=tuple(dict.fromkeys(args.precisions)), native_p90=args.native_p90,
            serving_cap=args.cap_tokens)
