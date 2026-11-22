import numpy as np

from superturn.review_pause_suggestions import RATE, suggest


def test_shared_pause_suggestions_require_dominant_unambiguous_speakers():
    tracks = np.zeros((2, RATE * 4), dtype=np.int16)
    tracks[0, : RATE] = 1200
    tracks[1, RATE * 2:RATE * 3] = 1200
    activity = [[(200, 1000)], [(2000, 3000)]]

    suggestions, counts = suggest("call-test", tracks, activity)
    assert len(suggestions) == counts["shared_gaps"] == 1
    assert suggestions[0]["focal_channel"] == "left"
    assert suggestions[0]["next_channel"] == "right"
    assert suggestions[0]["stop_ms"] == 1000
    assert suggestions[0]["next_onset_ms"] == 2000

    tracks[1, : RATE] = 1100
    suggestions, counts = suggest("call-test", tracks, activity)
    assert not suggestions
    assert counts["crosstalk_suppressed"] == 1

    activity[1].insert(0, (800, 950))
    suggestions, counts = suggest("call-test", tracks, activity)
    assert not suggestions
    assert counts["overlap_end_suppressed"] == 1
