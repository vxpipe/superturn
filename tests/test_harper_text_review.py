from superturn.harper_text_review import matches, score_window, summarize_calls


def test_match_maximizes_cardinality_before_timing_distance():
    proposals = [{"stop_ms": 100}, {"stop_ms": 350}]
    marks = [{"stop_ms": 0}, {"stop_ms": 220}]
    assert matches(proposals, marks, 150) == [(0, 0), (1, 1)]


def test_wrong_next_speaker_is_a_matched_label_disagreement():
    record = {"anchor_ms": 7000, "proposals": [
        {"stop_ms": 7000, "onset_ms": 7500, "outcome": "hold", "next_channel": "left"},
    ]}
    marks = [{"stop_ms": 7100, "next_onset_ms": 7530, "outcome": "handoff",
              "next_channel": "right", "brief_ack": "no"}]
    scores = score_window(record, marks, 250)
    assert scores["outcome_disagreement"] == scores["next_onset_within_80_ms"] == 1
    assert scores["unmatched_proposals"] == scores["next_channel_agreement"] == 0


def test_scorecard_distinguishes_stop_coverage_confirmation_and_missed_events():
    window = {"anchor_ms": None, "proposals": [
        {"stop_ms": 4000, "onset_ms": 4550, "outcome": "hold", "next_channel": "left"},
        {"stop_ms": 7000, "onset_ms": 7500, "outcome": "shift", "next_channel": "right"},
        {"stop_ms": 10000, "onset_ms": 10400, "outcome": "shift", "next_channel": "right"},
    ]}
    marks = [
        {"stop_ms": 4050, "next_onset_ms": 4550, "outcome": "resume",
         "next_channel": "left", "brief_ack": "no"},
        {"stop_ms": 7000, "next_onset_ms": 7500, "outcome": "handoff",
         "next_channel": "left", "brief_ack": "no"},
        {"stop_ms": 13000, "next_onset_ms": 13500, "outcome": "handoff",
         "next_channel": "right", "brief_ack": "no"},
        {"stop_ms": 16000, "next_onset_ms": 16100, "outcome": "handoff",
         "next_channel": "right", "brief_ack": "no"},
    ]
    report = summarize_calls([score_window(window, marks, 80)])
    holds, shifts = (report["by_proposed_class"][label] for label in ("hold", "shift"))
    assert shifts["matched_stop_coverage"] == 0.5
    assert shifts["unusable_matched_outcomes"] == 1
    assert shifts["agreement_given_usable_match"] is None
    assert shifts["confirmed_share_of_proposals"] == 0
    assert holds["confirmed_share_of_proposals"] == holds["agreement_given_usable_match"] == 1
    assert holds["confirmation_call_bootstrap_95"] is None
    assert report["events"]["shift_unmatched_eligible_human_marks"] == 1
    assert report["events"]["shift_eligible_human_marks"] == 1
