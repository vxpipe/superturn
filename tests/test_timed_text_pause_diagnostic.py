from superturn.audio import RATE
from superturn.timed_text_pause_diagnostic import pause_prefix


def test_prefix_stops_at_word_end_and_refuses_stale_assistant_context():
    row = {
        "words": [
            {"start": .1, "end": .4, "word": "Can"},
            {"start": .4, "end": 1.0, "word": "You"},
            {"start": .9, "end": 1.1, "word": "Help"},
        ],
        "messages": [{"role": "assistant", "content": "What do you need?"}],
    }
    assert pause_prefix(row, RATE) == ("what do you need", "can you")
    assert pause_prefix(row, RATE - RATE // 20) == ("what do you need", "can")
    row["messages"].append({"role": "user", "content": "Earlier conversation"})
    assert pause_prefix(row, RATE) == ("", "can you")
