from superturn.text_partial_geometry import partial_geometry


def test_partial_geometry_preserves_missing_hold_and_shift_denominators():
    summary = partial_geometry([
        {"label": "hold", "text": None, "missing": "missing_previous_stt"},
        {"label": "hold", "text": "what can i check\num the the"},
        {"label": "shift", "text": "---\nfriday please"},
        {"label": "shift", "text": None},
    ])
    assert summary["hold"]["rows"] == summary["shift"]["rows"] == 2
    assert summary["hold"]["counts"]["missing_previous_stt"] == 1
    assert summary["hold"]["counts"]["hesitation_word"] == 1
    assert summary["hold"]["counts"]["repeated_adjacent_word"] == 1
    assert summary["shift"]["counts"]["first_turn_marker"] == 1
    assert summary["shift"]["counts"]["no_text"] == 1
