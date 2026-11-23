from superturn.turn_text_near_end import make_train


def test_near_end_replacement_preserves_label_context_and_splits():
    rows, dialogues = [], {}
    for index in range(4):
        slot = str(index)
        prior = "---" if index < 2 else "what date should we use"
        beginning = f"i am free after {index}"
        rows.extend([
            {"slot": slot, "language": "en", "text": f"{prior}\n{beginning}", "label": "hold"},
            {"slot": slot, "language": "en", "text": f"{prior}\nnext friday {index}", "label": "shift"},
        ])
        dialogues[slot] = {"language": "en", "caller_before_pause": beginning,
                           "caller_after_pause": "in the late afternoon"}
    validation = [{"text": "---\ni am free after 0 in the late"}]
    derived, stats = make_train(rows, validation, dialogues)
    assert stats == {"en/one_turn": 1, "en/two_turn": 1}
    for old, row in zip(rows, derived, strict=True):
        assert (old["slot"], old["label"]) == (row["slot"], row["label"])
        assert row["text"].split("\n")[0] == old["text"].split("\n")[0]
        if row != old:
            assert row["label"] == "hold"
            assert row["text"].startswith(old["text"] + " ")
            assert row["text"] not in [example["text"] for example in validation]
