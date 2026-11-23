from superturn.turn_training_dataset import input_key, model_text, split_examples


def test_synthetic_split_keeps_pairs_scenarios_and_duplicate_inputs_together():
    rows = []
    for index in range(60):
        for language in ("en", "fr"):
            rows.append({
                "scenario": f"scenario-{index}", "slot": f"{language}/{index}", "language": language,
                "previous_agent": "same opening" if index < 2 else f"opening {index}",
                "caller_before_pause": "same prefix" if index < 2 else f"prefix {index}",
                "caller_after_pause": f"suffix {index}", "short_caller_reply": "",
            })
    splits, groups = split_examples(rows)
    assert sum(groups.values()) == 59
    assignment = {}
    scenario_assignment = {}
    for split, examples in splits.items():
        assert len(examples) % 2 == 0
        for example in examples:
            slot = example["slot"]
            assert assignment.setdefault(slot, split) == split
            assert scenario_assignment.setdefault(slot.split("/")[-1], split) == split
    assert scenario_assignment["0"] == scenario_assignment["1"]
    assert not ({input_key(item) for item in splits["train"]}
                & {input_key(item) for item in splits["validation"]})


def test_input_identity_uses_only_model_visible_fields():
    one = {"previous_turn": "", "current_turn": "okay", "language": "en", "label": "hold"}
    another = {**one, "language": "fil", "label": "shift"}
    assert input_key(one) == input_key(another)
    assert input_key(one) == "---\nokay"
    assert model_text("what time works", "around five") == "what time works\naround five"
