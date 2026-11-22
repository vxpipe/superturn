import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from superturn.e20_timer_replay import calibration_cutoffs, observed_subset, scenario


def test_timer_matching_uses_calibration_only():
    truth = np.array([True, True, False, False])
    duration = np.array([300., 300., 650., 1200.])
    calibration = truth, duration, np.array([.9, .8, .1, .2])
    cutoffs = calibration_cutoffs(calibration, 300, 200, 0)
    expected = cutoffs["500"]["match_timer_latency"]
    assert expected == .8

    held_scores = np.array([.4, .5, .8, .9])
    results = scenario((truth, duration, held_scores), expected, cutoffs, 300, 200, 0)
    timer = results["silence_only"]["500"]
    matched = results["calibration_matched_thresholds"]["500"]["match_timer_latency"]
    assert matched["threshold"] == .8
    assert matched["mean_shift_reply_ms"] > timer["mean_shift_reply_ms"]
    assert matched["audible_false_starts_per_observed_hold"] == .5


def test_censored_shift_stays_and_censored_hold_is_counted():
    rows = [(1, 0, 300, True), (0, 1, 700, True), (0, 2, 850, False)]
    data, counts = observed_subset(rows, np.array([.9, .8, .7]), np.ones(3, dtype=bool))
    assert data[0].tolist() == [True, False]
    assert counts == {"holds": 1, "shifts": 1, "excluded_censored_holds": 1, "retained_censored_shifts": 1}
