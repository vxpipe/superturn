import numpy as np

from superturn.timed_text_threshold_diagnostic import threshold_for_ba


def test_balanced_threshold_keeps_equal_scores_together():
    scores = np.array([.9, .8, .8, .2])
    labels = np.array([True, True, False, False])
    threshold = threshold_for_ba(scores, labels)
    assert threshold == .9
    assert (scores >= threshold).tolist() == [True, False, False, False]


def test_balanced_threshold_can_reject_all_scores():
    scores = np.array([.5, .5])
    threshold = threshold_for_ba(scores, np.array([True, False]))
    assert not np.any(scores >= threshold)
