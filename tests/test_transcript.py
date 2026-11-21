from superturn.transcript import TEXT_BYTES, pack_partial_transcript


def test_partial_transcript_normalizes_and_right_aligns_bytes():
    packed = pack_partial_transcript("Ａ，猫!\0")
    expected = list("a猫".encode("utf-8"))
    assert packed == [0] * (TEXT_BYTES - len(expected)) + expected
    assert pack_partial_transcript(None) == [0] * TEXT_BYTES


def test_partial_transcript_keeps_exact_suffix_even_inside_utf8_character():
    text = "界" * 50
    assert pack_partial_transcript(text) == list(text.encode("utf-8")[-TEXT_BYTES:])
