import numpy as np
from onnx import TensorProto, helper
import pytest

from superturn.text_cpu_probe import add_e5_scalar_head, e5_probe_head_weights, encoder_screen, get_session


def test_encoder_screen_requires_language_coverage_and_unclipped_token_lengths():
    detail = {"token_needs": {"en": {"full_two_lines": {"n": 10, "p90": 51.0}},
                              "kn": {"full_two_lines": {"n": 0}}}}
    durations = {str(n): {"synthetic_text_to_scalar": {"p99_ms": 4.9}} for n in (64, 128)}
    load = {"workers_with_cpu_ticks": 2}

    def screen():
        return encoder_screen(detail, ("en", "kn"), (64, 128), durations, durations, load,
                              expected_workers=2)

    missing = screen()
    assert not missing["passed"]
    assert missing["missing_languages"] == ["kn"]

    detail["token_needs"]["kn"]["full_two_lines"] = {"n": 10, "p90": 128.1}
    over_limit = screen()
    assert not over_limit["passed"]
    assert over_limit["p90_exceeds_128"]
    assert over_limit["conservative_measured_bucket"] is None

    declared = encoder_screen(detail, ("en", "kn"), (64, 128), durations, durations, load,
                              expected_workers=2, serving_cap=64)
    assert declared["passed"] and declared["largest_p90"] == 128.1
    assert declared["serving_token_cap"] == declared["conservative_measured_bucket"] == 64

    detail["token_needs"]["kn"]["full_two_lines"]["p90"] = 63.1
    within_limit = screen()
    assert within_limit["passed"]
    assert within_limit["conservative_measured_bucket"] == 64


@pytest.mark.parametrize("opset", [11, 13])
def test_one_block_scalar_pools_only_caller_tokens(opset):
    shape = ["batch_size", "sequence_length", 4]
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["state"])], "probe", [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, shape),
    ], [helper.make_tensor_value_info("state", TensorProto.FLOAT, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)
    add_e5_scalar_head(model)
    session = get_session(model.SerializeToString())
    states = np.arange(20, dtype=np.float32).reshape(1, 5, 4) / 23
    selection = [1, 3]
    mask = np.zeros((1, 5, 1), dtype=np.float32)
    mask[0, selection, 0] = 1
    actual = session.run(None, {"input": states, "caller_mask": mask,
                                "last_caller": np.array([[0, 3]], dtype=np.int64)})[0].item()
    first, second = e5_probe_head_weights(4)
    visible = states[0, selection]
    expected_logit = (np.maximum(np.r_[visible[-1], visible.mean(axis=0)] @ first, 0.) @ second).item()
    assert abs(actual - 1. / (1. + np.exp(-expected_logit))) < 2e-6
