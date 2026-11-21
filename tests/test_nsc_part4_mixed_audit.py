from collections import Counter

from superturn.audio import RATE
from superturn.nsc_part4_mixed_audit import audit, cross_check


def test_mixed_pause_requires_full_quiet_window_and_attributable_flanks():
    left = ("these", "are", "four", "words")
    right = ("here", "is", "my", "reply")
    onset = 3 * RATE // 2
    activity = [(0, RATE), (onset, onset + RATE)]
    words = [(start + i * RATE // 4, start + (i + 1) * RATE // 4, word)
             for start, phrase in zip((0, onset), (left, right), strict=True)
             for i, word in enumerate(phrase)]
    reference = [Counter({left: 1}), Counter({right: 1})]

    rows, counts = audit(activity, words, reference)
    assert counts["shift"] == 1
    assert cross_check(rows, rows) == {"shift": 1}
    assert not cross_check(rows * 2, rows)

    swapped, _ = audit(activity, words, reference[::-1])
    assert (swapped[0]["first"], swapped[0]["second"]) == (rows[0]["second"], rows[0]["first"])

    intrusive = (RATE + 320, RATE + 640, None)
    rejected, counts = audit(activity, sorted([*words, intrusive]), reference)
    assert not rejected
    assert counts["word_inside_pause"] == 1

    rejected, counts = audit(activity, words[:-1], reference)
    assert not rejected
    assert counts["short_flank"] == 1
