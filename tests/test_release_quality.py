import hashlib
import json

import pytest

from superturn.release_cohort import CONTRACT, digest, ensure_disjoint, validate_manifest, write_once


def test_manifest_detects_shared_counterpart_even_when_focal_speakers_differ(tmp_path):
    plan = {"identity_namespace": "shared-people", "panel": "a", "languages": ["en"]}
    manifest = {"role": "calibration", "panel": "a", "identity_namespace": "shared-people",
                "decision_ms": 300, "condition": "telephone", "adjudication_receipt": "verified",
                "participants": {"call": ["first", "bridge"]}, "pauses": [
                    {"pause": f"p{i}", "call": "call", "speaker": "first", "edge_sample": i * 8000,
                     "source_rate": 8000, "quiet_samples": 2400, "label": label, "language": "en",
                     "pcm_sha256": f"{i:064x}"}
                    for i, label in enumerate(("hold", "shift"))
                ]}
    _, membership = validate_manifest(manifest, plan, "calibration")
    with pytest.raises(ValueError, match="share a person"):
        ensure_disjoint({"fit-speaker"}, membership, ({"second", "bridge"}, {"other"}, set(), set()))
    manifest["participants"]["call"][1] = "unknown"
    with pytest.raises(ValueError, match="fully resolved"):
        validate_manifest(manifest, plan, "calibration")
    lock = tmp_path / "lock.json"
    write_once(lock, {"value": .5})
    write_once(lock, {"value": .5})
    with pytest.raises(ValueError, match="incompatible release attempt"):
        write_once(lock, {"value": .6})


@pytest.mark.parametrize("case", ("audio", "audio-text", "always-hold"))
def test_frozen_onnx_release_rejects_changed_calibration(tmp_path, case):
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from onnx import TensorProto as T, helper as h

    from superturn.evaluate import release_quality

    (tmp_path / ".campaign-machine").touch()
    model = tmp_path / "fixture.onnx"
    combined = case == "audio-text"
    nodes = [
        h.make_node("Gather", ["audio", "index"], ["x"], axis=1),
        h.make_node("Add", ["x", "one"], ["shifted"]),
        h.make_node("Mul", ["shifted", "half"], ["acoustic" if combined else "p_complete"]),
    ]
    inputs = [h.make_tensor_value_info("audio", T.FLOAT, [1, 48000]),
              h.make_tensor_value_info("silence_ms", T.FLOAT, [1, 1])]
    initializers = [h.make_tensor("index", T.INT64, [1], [0]), h.make_tensor("one", T.FLOAT, [1], [1]),
                    h.make_tensor("half", T.FLOAT, [1], [.5])]
    if combined:
        inputs.append(h.make_tensor_value_info("text_bytes", T.INT32, [1, 128]))
        initializers.append(h.make_tensor("token_weight", T.FLOAT, [1], [1 / 1024]))
        nodes += [h.make_node("Gather", ["text_bytes", "index"], ["byte"], axis=1),
                  h.make_node("Cast", ["byte"], ["token"], to=T.FLOAT),
                  h.make_node("Mul", ["token", "token_weight"], ["lexical"]),
                  h.make_node("Add", ["acoustic", "lexical"], ["p_complete"])]
    graph = h.make_graph(nodes, "audio_score", inputs,
                         [h.make_tensor_value_info("p_complete", T.FLOAT, [1, 1])], initializers)
    onnx.save(h.make_model(graph, ir_version=10, opset_imports=[h.make_opsetid("", 17)]), model)

    def save(name, value):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        return path

    interface = "audio-text" if combined else "audio"
    plan = {"contract": CONTRACT, "identity_namespace": "fixture", "panel": "generated",
            "languages": ["en"], "interface": interface, "artifact_sha256": digest(model)}

    def render(role, amplitudes):
        audio = tmp_path / f"{role}.npy"
        pcm = np.zeros((2, 48000), dtype="<i2")
        pcm[:, 0] = amplitudes
        np.save(audio, pcm)
        speaker, other = f"{role}-left", f"{role}-right"
        cohort = {"role": role, "panel": plan["panel"], "identity_namespace": plan["identity_namespace"],
                  "decision_ms": 300, "condition": "telephone", "adjudication_receipt": "synthetic",
                  "audio_sha256": digest(audio), "participants": {role: [speaker, other]}, "pauses": [
                      {"pause": f"{role}-{i}", "call": role, "speaker": speaker, "edge_sample": i * 8000,
                       "source_rate": 8000, "quiet_samples": 2400, "label": label, "language": "en",
                       "pcm_sha256": hashlib.sha256(pcm[i].tobytes()).hexdigest()}
                      for i, label in enumerate(("hold", "shift"))
                  ]}
        text = None
        if combined:
            text = tmp_path / f"{role}-text.npy"
            partials = np.zeros((2, 128), dtype=np.uint8)
            partials[1, 0] = 120
            np.save(text, partials)
            cohort["text_sha256"] = digest(text)
        receipt = save(role, cohort)
        plan[f"{role}_manifest_sha256"] = digest(receipt)
        return receipt, audio, text

    calibration = render("calibration", (8000, -8000) if case == "always-hold" else (-8000, 8000))
    test = render("test", (-10000, 10000))
    fit = save("fit", {"identity_namespace": "fixture", "coverage_receipt": "synthetic", "speakers": ["fit"]})
    plan["fit_roster_sha256"] = digest(fit)
    plan_path = save("plan", plan)

    def run(paths, role, **options):
        cohort, audio, text = paths
        return release_quality(tmp_path, model, plan_path, fit, cohort, audio, role=role,
                               plan_sha256=digest(plan_path), text_path=text, **options)

    sealed = run(calibration, "calibration")
    report = run(test, "test", calibration_sha256=sealed["calibration_sha256"])
    primary = report["models"]["combined" if combined else "audio"]
    assert report["balanced_accuracy_pass"] is (case != "always-hold")
    assert primary["macro"]["balanced_accuracy"] == (.5 if case == "always-hold" else 1.0)
    if combined:
        assert report["models"]["masked_text"]["languages"]["en"]["threshold"] == primary["languages"]["en"]["threshold"]
        assert report["models"]["masked_text"]["macro"]["balanced_accuracy"] == .5
    if case == "always-hold":
        assert primary["languages"]["en"]["threshold"] is None
        assert primary["languages"]["en"]["correct_shifts"] == 0
    assert report["latency_ms"] is None
    lock = tmp_path / f"runs/release-quality/generated/{interface}/calibration.json"
    lock.write_text(lock.read_text() + "\n")
    with pytest.raises(ValueError, match="digest changed"):
        run(test, "test", calibration_sha256=sealed["calibration_sha256"])
