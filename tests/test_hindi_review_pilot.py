import numpy as np

from superturn.audio import RATE
from superturn.hindi_review_pilot import ballot_outcome, supported_edge
from superturn.paired_call_review import QUIET


def test_preflight_excludes_assisted_outcome_changed_after_first_choice():
    event = {"outcome": "resume", "views": {
        "assisted": {"first_outcome": "handoff", "judgment": {"confidence": "", "language": "hindi"}},
    }}
    assert ballot_outcome(event) == (None, "assisted_first_and_final_disagree")
    event["views"]["assisted"]["first_outcome"] = "resume"
    assert ballot_outcome(event) == ("resume", None)
    event["views"]["assisted"]["judgment"]["language"] = "mixed"
    assert ballot_outcome(event) == (None, "review_language_not_hindi")


def test_preflight_requires_focal_activity_and_bilateral_quiet_without_moving_edge():
    edge = 3 * RATE
    tracks = np.zeros((2, edge + QUIET), dtype=np.int16)
    tracks[0, edge - RATE // 5:edge] = 1024
    clean = [[(edge - RATE // 5, edge)], []]
    assert supported_edge(tracks, clean, 0, edge)
    assert not supported_edge(tracks, clean, 1, edge)
    assert not supported_edge(tracks, [clean[0], [(edge + QUIET - 1, edge + QUIET)]], 0, edge)
    assert not supported_edge(tracks, clean, 0, edge + 1)
