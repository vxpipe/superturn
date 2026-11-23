from types import SimpleNamespace

from superturn.harper_text_inputs import causal_example, prefix_turns


def speech(start, end, role):
    return SimpleNamespace(start=start * 16, end=end * 16, speaker=role)


def transcript(role, offset, start, words, timings, durations):
    return {"speaker_role": role, "offset_ms": offset, "start_ms": start, "duration_ms": 2500,
            "transcript": words, "word_offsets_ms": timings, "word_durations_ms": durations}


def test_recovered_clock_and_future_words_preserve_the_causal_prefix():
    turns = [speech(0, 1100, "agent"), speech(1200, 2000, "caller"),
             speech(2200, 2600, "caller"), speech(3000, 3500, "agent")]
    segments = [transcript("agent", 700, 0, "which account", [10, 500], [100, 100]),
                transcript("caller", 1200, 1200, "my savings account future", [10, 420, 1050, 1500], [90] * 4)]
    origins = {"caller": 0, "agent": 0}
    expected = {"text": "which account\nmy savings account", "current_only": "---\nmy savings account",
                "last_word_missing": "which account\nmy savings"}
    assert causal_example(segments, origins, "start_ms", turns, 2600 * 16, 0) == (expected, None)
    segments[1]["transcript"] = "my savings account arbitrary changed words"
    assert causal_example(segments, origins, "start_ms", turns[:3], 2600 * 16, 0) == (expected, None)


def test_cropped_context_is_not_promoted_to_a_first_turn():
    turns = [speech(1200, 2000, "caller")]
    assert prefix_turns(turns, 2000 * 16, 1000 * 16, False)[1] == "previous_turn_unknown"
    assert prefix_turns(turns, 2000 * 16, 1000 * 16, True)[1] is None
