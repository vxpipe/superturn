import pytest

from superturn.harper_text_comparison import paired_intervals


def test_shared_caller_resample_leaves_identical_predictions_at_zero_delta():
    rows = [
        {"rank": 1, "label": 0, "score": .1, "caller": 1},
        {"rank": 2, "label": 1, "score": .2, "caller": 1},
        {"rank": 3, "label": 0, "score": .6, "caller": 2},
        {"rank": 4, "label": 1, "score": .8, "caller": 2},
    ]
    baseline = {row["rank"]: row["score"] for row in rows}
    result = paired_intervals(rows, baseline, draws=50)
    assert result["replicates"] == 50
    assert result["estimate"]["auc"] == 0
    assert result["paired_caller_bootstrap_95"]["balanced_accuracy"] == [0, 0]
    with pytest.raises(ValueError, match="same inputs"):
        paired_intervals(rows, {1: .1})
