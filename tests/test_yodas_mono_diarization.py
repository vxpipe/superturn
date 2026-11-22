from superturn.yodas_mono_diarization import predictions


def test_nested_overlap_keeps_the_gap_after_the_longer_speaker():
    spans = [(0, 32000, "A"), (8000, 12800, "B"), (41600, 48000, "B")]
    words = [(30400, 32000), (41600, 43200)]
    gap = (32000, 41600, False)

    assert predictions(spans, words) == ([gap], [gap])


def test_word_in_a_different_bout_does_not_support_the_gap():
    spans = [(0, 10000, "A"), (10500, 11000, "A"), (16000, 21000, "B")]
    words = [(9700, 10000), (16000, 17000)]
    gap = (11000, 16000, False)

    assert predictions(spans, words) == ([gap], [])
