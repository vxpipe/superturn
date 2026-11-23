from superturn.harper_text_review import matches, score_window


def test_match_maximizes_cardinality_before_timing_distance():
    proposals = [{"stop_ms": 100}, {"stop_ms": 350}]
    marks = [{"stop_ms": 0}, {"stop_ms": 220}]
    assert matches(proposals, marks, 150) == [(0, 0), (1, 1)]


def test_wrong_next_speaker_is_a_matched_label_disagreement():
    record = {"anchor_ms": 7000, "proposals": [
        {"stop_ms": 7000, "onset_ms": 7500, "outcome": "hold", "next_channel": "left"},
    ]}
    marks = [{"stop_ms": 7100, "next_onset_ms": 7530, "outcome": "handoff", "next_channel": "right"}]
    scores = score_window(record, marks, 250)
    assert scores["outcome_disagreement"] == scores["next_onset_within_80_ms"] == 1
    assert scores["unmatched_proposals"] == scores["next_channel_agreement"] == 0
