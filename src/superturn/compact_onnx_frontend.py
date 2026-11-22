"""Pack STFT lookup tables and store GRU biases in half precision."""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

PITCH = (127, 190)
MEL = (129, 40)


def shorten_names(graph):
    """Keep the public names and abbreviate internal single-graph symbols."""
    public = {item.name for item in (*graph.input, *graph.output)}
    if graph.sparse_initializer or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                                       for node in graph.node for a in node.attribute):
        raise ValueError("unsupported graph structure for internal name compaction")
    renamed = {}

    def short(name):
        if not name or name in public:
            return name
        if name not in renamed:
            renamed[name] = f"x{len(renamed):x}"
        return renamed[name]

    for tensor in graph.initializer:
        tensor.name = short(tensor.name)
    for value in graph.value_info:
        value.name = short(value.name)
    for node in graph.node:
        node.name = ""
        node.input[:] = [short(name) for name in node.input]
        node.output[:] = [short(name) for name in node.output]


def compact_frontend(source, destination):
    model = onnx.load(source)
    graph = model.graph
    before = Path(source).stat().st_size
    nodes, compacted = [], {}

    def store(name, values):
        graph.initializer.append(numpy_helper.from_array(np.asarray(values), name))

    for node in graph.node:
        tensors = [attribute.t for attribute in node.attribute
                   if attribute.type == onnx.AttributeProto.TENSOR] if node.op_type == "Constant" else []
        if len(tensors) != 1 or tuple(tensors[0].dims) not in (MEL, PITCH):
            nodes.append(node)
            continue
        value = numpy_helper.to_array(tensors[0])
        out, = node.output
        prefix = out + ".compact."
        if value.shape == MEL:
            flat = value.reshape(-1)
            positions = np.flatnonzero(flat).astype(np.int64)
            if len(positions) > 2 * MEL[0]:
                raise ValueError("expected the native sparse mel bank")
            store(prefix + "length", np.array([len(flat)], dtype=np.int64))
            store(prefix + "positions", positions)
            store(prefix + "values", flat[positions])
            store(prefix + "shape", np.array(MEL, dtype=np.int64))
            nodes += [
                helper.make_node("ConstantOfShape", [prefix + "length"], [prefix + "zeros"]),
                helper.make_node("ScatterElements", [prefix + "zeros", prefix + "positions", prefix + "values"],
                                 [prefix + "dense"], axis=0),
                helper.make_node("Reshape", [prefix + "dense", prefix + "shape"], [out]),
            ]
            compacted["mel_exact_nonzero"] = len(positions)
        else:
            bins = np.arange(2, 129, dtype=np.float32)[:, None]
            lags = np.arange(40, 230, dtype=np.float32)
            frequency = np.asarray(2 * np.pi / 512, dtype=np.float32)
            error = float(np.max(np.abs(np.cos(bins * lags * frequency) - value)))
            if error > 2e-7:
                raise ValueError("the pitch basis is not the native 512-point formula")
            store(prefix + "bins", bins)
            store(prefix + "lags", lags)
            store(prefix + "frequency", frequency)
            nodes += [
                helper.make_node("Mul", [prefix + "bins", prefix + "lags"], [prefix + "products"]),
                helper.make_node("Mul", [prefix + "products", prefix + "frequency"], [prefix + "phase"]),
                helper.make_node("Cos", [prefix + "phase"], [out]),
            ]
            compacted["pitch_cosine_numpy_max_abs_error"] = error

    if set(compacted) != {"mel_exact_nonzero", "pitch_cosine_numpy_max_abs_error"}:
        raise ValueError("STFT export did not expose both expected frontend tables")
    # Weights are already packed int8. Store just the two bias tensors in
    # FP16 to meet the file budget, then use the existing FP32 GRU kernels.
    bias_names = {node.input[3] for node in graph.node if node.op_type == "GRU"}
    bias_nodes, bias_errors = [], []
    for tensor in list(graph.initializer):
        if tensor.name not in bias_names:
            continue
        values = numpy_helper.to_array(tensor)
        packed = values.astype(np.float16)
        if values.dtype != np.float32 or not np.isfinite(packed).all():
            raise ValueError("expected finite float32 GRU bias initializers")
        name = tensor.name
        graph.initializer.remove(tensor)
        store(name + ".fp16", packed)
        bias_nodes.append(helper.make_node("Cast", [name + ".fp16"], [name], to=onnx.TensorProto.FLOAT))
        bias_errors.append(float(np.max(np.abs(packed.astype(np.float32) - values))))
    if len(bias_nodes) != len(bias_names) or not bias_errors:
        raise ValueError("the STFT candidate has unexpected GRU bias storage")
    del graph.node[:]
    graph.node.extend(bias_nodes + nodes)
    shorten_names(graph)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    size = Path(destination).stat().st_size
    return {**compacted, "gru_bias_fp16_max_abs_error": max(bias_errors),
            "source_bytes": before, "compact_bytes": size}
