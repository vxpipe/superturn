from pathlib import Path

import pytest

from superturn.turn_text_train import QUERY_PREFIX, TOKEN_BUDGET, pack, pack_qwen, read_dataset


class PairTokenizer:
    bos_token_id = 0
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [1 if part == "---" else int(part) if part.isdigit() else 2048 for part in text.split()]

    def num_special_tokens_to_add(self, *, pair):
        assert pair
        return 4


def test_qwen_receipt_refuses_e5_base_files(tmp_path, monkeypatch):
    from superturn import turn_text_train as module

    qwen = tmp_path / "qwen"
    e5 = tmp_path / "e5"
    for path, weights in ((qwen, b"qwen weights"), (e5, b"e5 weights")):
        path.mkdir()
        (path / "model.safetensors").write_bytes(weights)
        (path / "tokenizer.json").write_bytes(b"fixture tokenizer")
    expected = module.digest(qwen / "model.safetensors")
    monkeypatch.setattr(module, "QWEN_SHA256", expected)
    receipt = {
        "backbone": "qwen3", "base": module.QWEN_BASE, "base_revision": module.QWEN_REVISION,
        "base_weights_sha256": expected, "dataset_manifest_sha256": "fixture",
        "tokenizer_sha256": module.digest(qwen / "tokenizer.json"), "epochs": 3,
        "token_budget": TOKEN_BUDGET, "query_prefix": "",
    }
    assert module.check_fit_base(receipt, qwen, "fixture", 3) == "qwen3"
    with pytest.raises(ValueError, match="evaluation inputs do not match"):
        module.check_fit_base(receipt, e5, "fixture", 3)


def test_pack_preserves_first_turn_marker_and_recent_caller_tokens():
    raw = "---\n" + " ".join(map(str, range(100, 280)))
    result = pack(PairTokenizer(), raw)
    assert len(result) == TOKEN_BUDGET
    assert result[:5] == [0, 2048, 1, 2, 2]
    assert result[-2:] == [279, 2]


def test_pack_keeps_previous_tail_after_reserving_current_turn():
    previous = " ".join(map(str, range(20, 190)))
    current = " ".join(map(str, range(200, 220)))
    result = pack(PairTokenizer(), previous + "\n" + current)
    assert len(result) == TOKEN_BUDGET
    assert result[1] == PairTokenizer().encode(QUERY_PREFIX, add_special_tokens=False)[0]
    assert result[-23:] == [2, 2, *range(200, 220), 2]
    assert 189 in result and 20 not in result


def test_trainer_reads_the_published_immutable_dataset():
    rows, _ = read_dataset(Path("datasets/turn-completion-v1"))
    assert [len(rows[name]) for name in ("train", "validation")] == [3696, 616]


def test_decoder_keeps_opening_marker_and_last_actual_caller_word():
    class DecoderTokenizer(PairTokenizer):
        def encode(self, text, *, add_special_tokens):
            if text == "\n":
                return [999]
            return super().encode(text, add_special_tokens=add_special_tokens)

    current = " ".join(map(str, range(100, 280)))
    result = pack_qwen(DecoderTokenizer(), "---\n" + current)
    assert len(result) == TOKEN_BUDGET
    assert result[:2] == [1, 999]
    assert result[-1] == 279

def test_length_balancing_keeps_language_and_length_mass():
    import pytest

    from superturn.turn_text_train import length_balanced_weights

    rows = [
        *[{"language": "en", "label": "hold", "text": "---\nwait"}] * 7,
        {"language": "en", "label": "shift", "text": "---\nyes"},
        {"language": "en", "label": "hold", "text": "---\nthere were just three"},
        {"language": "en", "label": "shift", "text": "---\nthat works for me"},
        {"language": "fr", "label": "shift", "text": "---\noui"},
    ]
    weights, bins = length_balanced_weights(rows)
    assert weights[7] == 3
    assert weights[8:] == [1., 1., 1.]
    assert sum(weights[:8]) == pytest.approx(8)
    assert bins["en/1-3"]["rows_at_cap"] == 1
