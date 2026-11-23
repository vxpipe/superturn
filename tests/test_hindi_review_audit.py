import pytest

from superturn.hindi_review_audit import compare_followup, read_followup, reconcile
from superturn.paired_call_judgments import CONTINUOUS_FIELDS, WINDOW_FIELDS


def test_reconcile_deduplicates_source_pauses_and_keeps_disagreements():
    packet = {"windows": ["window"], "clips": ["clip", "later"]}
    window_key = {"window": {
        "recording": "source", "native_start_sample": 100_000, "samples": 480_000, "clip_sha256": "abc",
    }}
    short_key = {
        "clip": {"source": "spring", "nominal_language": "hi", "recording": "source",
                 "focus_channel": 1, "native_edge_sample": 116_000},
        "later": {"source": "spring", "nominal_language": "hi", "recording": "source",
                  "focus_channel": 0, "native_edge_sample": 132_000},
    }
    suggestions = {
        "schema": 1, "mode": "assisted", "recordings": {"window": {"audio_sha256": "abc"}},
        "proposals": [{"id": "p", "recording": "window", "focal_channel": "right", "stop_ms": 1000,
                       "next_onset_ms": 1500, "next_channel": "left"}],
    }
    hold = {"outcome": "resume", "language": "hindi"}
    shift = {"outcome": "handoff", "language": "hi"}
    ballot = {"draft": {"clips": {"clip": hold, "later": hold}, "assisted": {"p": shift}}}
    rows, counts = reconcile(packet, ballot, suggestions, short_key, window_key)
    assert len(rows) == 2
    assert counts["conflict"] == counts["cross_view_duplicates"] == counts["cross_view_conflicts"] == 1
    assert counts["resume"] == counts["clear_assisted_with_proposal_timing"] == 1
    assert counts["unmatched_cross_view_within_80_ms"] == 0
    assert rows[1]["stop_sample"] == 116_000
    assert rows[1]["outcome"] is None
    assert rows[1]["views"].keys() == {"short", "assisted"}

    ballot["draft"]["assisted"]["p"] = {**hold, "rater_stop_ms": 0}
    rows, _ = reconcile(packet, ballot, suggestions, short_key, window_key)
    assert rows[1]["views"]["assisted"]["judgment"]["outcome"] == "invalid_pause"
    assert rows[1]["stop_sample"] == 116_000


def test_followup_keeps_ambiguous_matches_and_proposal_timing_separate():
    window_key = {"window": {"recording": "call", "native_start_sample": 16_000, "samples": 480_000}}
    fields = {"outcome": "handoff", "language": "hindi"}
    second = {("window", 0): [(1000, 1400, "right", fields),
                              (2000, 2400, "right", fields),
                              (4000, 4400, "right", fields)]}
    proposal = {"proposal_stop_ms": 1000, "next_onset_ms": 1400, "next_channel": "right",
                "next_onset_source": "proposal", "next_channel_source": "proposal"}
    anchors = [
        {"recording": "call", "focal_channel": 0, "stop_sample": stop,
         "outcome": "handoff", "language": "hindi", "views": views}
        for stop, views in [(32_000, {"assisted": {"timing": proposal}}),
                            (47_900, {}), (48_100, {})]
    ]
    matches, counts = compare_followup(anchors, second, window_key, {"window": ("hindi", "unknown")})
    assert len(matches) == counts["agreed_handoff"] == counts["proposal_onset_and_channel_corroborated"] == 1
    assert counts["ambiguous_first_anchors"] == 2
    assert counts["manual_marks_without_first_anchor"] == counts["manual_marks_with_ambiguous_anchor"] == 1
    assert "entered_onset_and_channel_corroborated" not in counts


def test_followup_requires_explicit_completion_even_without_events(tmp_path):
    key = {"window": {"samples": 480_000}}
    events = tmp_path / "events.csv"
    events.write_text(",".join(CONTINUOUS_FIELDS) + "\n")
    windows = tmp_path / "windows.csv"
    header = ",".join(WINDOW_FIELDS) + "\n"
    windows.write_text(header + "window,,,\n")
    with pytest.raises(ValueError, match="completion form"):
        read_followup(events, windows, key)
    windows.write_text(header + "window,yes,hindi,mixed\n")
    marks, completed, digests = read_followup(events, windows, key)
    assert not marks
    assert completed == {"window": ("hindi", "mixed")}
    assert set(digests) == {events, windows}
