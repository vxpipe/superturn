from superturn.hindi_review_audit import reconcile


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
