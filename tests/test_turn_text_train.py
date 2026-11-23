from pathlib import Path

from superturn.turn_text_train import QUERY_PREFIX, TOKEN_BUDGET, pack, read_dataset


class PairTokenizer:
    bos_token_id = 0
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [1 if part == "---" else int(part) if part.isdigit() else 2048 for part in text.split()]

    def num_special_tokens_to_add(self, *, pair):
        assert pair
        return 4


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
