from superturn.harper_text_transcript_benchmark import transcript_pairs


def test_full_and_partial_match_the_runtime_format_without_future_text():
    segments = [
        {"speaker_role": "caller", "start_timestamp_ms": 0, "duration_ms": 900,
         "human_transcript": "I need to move", "index": 0},
        {"speaker_role": "caller", "start_timestamp_ms": 1100, "duration_ms": 600,
         "human_transcript": "my appointment", "index": 1},
        {"speaker_role": "agent", "start_timestamp_ms": 2200, "duration_ms": 500,
         "human_transcript": "Sure, what day?", "index": 2},
        {"speaker_role": "caller", "start_timestamp_ms": 3000, "duration_ms": 600,
         "human_transcript": "Id like next Tuesday please", "index": 3},
        {"speaker_role": "agent", "start_timestamp_ms": 4400, "duration_ms": 500,
         "human_transcript": "Let me check", "index": 4},
    ]
    first, second = [views for _, views in transcript_pairs(segments)]
    assert first["interior_cut"] == "---\ni need to"
    assert first["full"] == "---\ni need to move my appointment"
    assert second["last_two_words_cut"] == "sure what day\nid like next"
    assert second["full"] == "sure what day\nid like next tuesday please"
