from superturn.icsi_roster import count_gaps


def test_nested_speech_preserves_the_true_end_and_exact_boundary():
    spans = [(0, 5000, "a", True), (2000, 3000, "b", True), (5300, 6100, "c", True)]
    assert count_gaps(spans) == ({("a", "shift"): 1}, 0)


def test_simultaneous_voices_cannot_supply_a_unique_end_or_start():
    tied_end = [(0, 1000, "a", True), (200, 1000, "b", True), (1600, 2000, "c", True)]
    tied_start = [(0, 1000, "a", True), (1600, 2000, "b", True), (1600, 2300, "", False)]
    assert count_gaps(tied_end) == ({}, 1)
    assert count_gaps(tied_start) == ({}, 1)
