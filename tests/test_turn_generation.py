import json

from superturn import turn_generation as gen


def test_quota_and_native_language_filters():
    catalogue = json.loads(gen.CATALOGUE.read_text())
    plan = gen.make_plan(catalogue)
    assert len(plan) == 14 * 77 * 2
    assert len({row["slot"] for row in plan}) == len(plan)
    hindi = next(row for row in plan if row["language"] == "hi")
    assert gen.clean_sample(dict.fromkeys(gen.FIELDS, "haan theek hai"), hindi) is None
    assert gen.normalize("नेहा, ठीक है।") == "नेहा ठीक है"
    assert gen.normalize("ok,thanks") == "ok thanks"
    assert gen.normalize("അ\u200dമ്മ") == "അ\u200dമ്മ"
    smoke = gen.smoke_plan(plan)
    assert len(smoke) == len({row["native_script"] for row in plan})
    assert {(row["caller_first"], row["short_shift"]) for row in smoke} == {
        (True, True), (True, False), (False, True), (False, False),
    }


def test_generator_resumes_without_duplicate_calls(monkeypatch, tmp_path):
    spec = gen.make_plan(json.loads(gen.CATALOGUE.read_text()))[0]
    sample = dict(zip(gen.FIELDS, ("", "check my account", "the savings one",
                                   "i can check that for you", "", ""), strict=True))
    calls = []

    def fake_request(path, key, payload):
        calls.append(path)
        generated = json.dumps([{"slot": spec["slot"], **sample}])
        return {"usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 60},
                "candidates": [{"content": {"parts": [{"text": generated}]}}]}

    monkeypatch.setattr(gen, "request_json", fake_request)
    receipt = gen.run([spec], tmp_path, .05, 1, "fixture-key")
    assert receipt["accepted_dialogues"] == 1
    assert receipt["stop"] == "complete"
    assert 0 < receipt["estimated_spend_including_unknown_requests_usd"] < .05
    assert gen.run([spec], tmp_path, .05, 1, "fixture-key") == receipt
    assert len(calls) == 1
    row = json.loads((tmp_path / "dialogues.jsonl").read_text())
    hold, shift = gen.paired_examples(row)
    assert hold["current_turn"] == "check my account"
    assert shift["current_turn"] == "check my account the savings one"
    assert hold["slot"] == shift["slot"] and set(hold) == set(shift)
    _, opening = gen.paired_examples({**row, "short_caller_reply": "just savings"})
    assert opening["previous_turn"] == ""
    assert opening["current_turn"] == "check my account the savings one"


def test_context_pairs_change_previous_turn_and_hide_continuations():
    from superturn.turn_text_context_supplement import plan

    row = {
        "previous_agent": "which day and which time suit you",
        "next_agent": "which day would work best for you",
        "caller_before_pause": "i can do next tuesday",
        "caller_after_pause": "at about three",
        "short_caller_reply": "",
        "final_agent_reply": "i will pencil that in",
    }
    for kind, label in (("contrast", "shift"), ("invariant", "hold")):
        spec = next(spec for spec in plan() if spec["pair_type"] == kind)
        sample = row if kind == "contrast" else {
            **row, "final_agent_reply": "", "short_caller_reply": "if there is a morning slot",
        }
        assert gen.clean_sample(sample, spec) == sample
        left, right = gen.paired_examples({**spec, **sample})
        assert (left["label"], right["label"]) == ("hold", label)
        assert left["current_turn"] == right["current_turn"] == row["caller_before_pause"]
        assert left["previous_turn"] != right["previous_turn"]
        assert not {"caller_after_pause", "short_caller_reply", "final_agent_reply"} & left.keys()
        assert gen.clean_sample({**sample, "next_agent": row["previous_agent"]}, spec) is None
