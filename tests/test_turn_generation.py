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


def test_generator_applies_judge_and_resumes_without_duplicate_calls(monkeypatch, tmp_path):
    spec = gen.make_plan(json.loads(gen.CATALOGUE.read_text()))[0]
    sample = dict(zip(gen.FIELDS, ("what would you like", "check my account", "the savings one",
                                   "i can check that for you"), strict=True))
    calls = []

    def fake_request(path, key, payload):
        calls.append(path)
        if path == "decisions":
            return {"usage": {"total_input_tokens": 200},
                    "answers": {question: {"noul": .9} for question in payload["questions"]}}
        generated = json.dumps([{"slot": spec["slot"], **sample}])
        return {"usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 60},
                "candidates": [{"content": {"parts": [{"text": generated}]}}]}

    monkeypatch.setattr(gen, "request_json", fake_request)
    receipt = gen.run([spec], tmp_path, .05, 1, "fixture-key")
    assert receipt["accepted_dialogues"] == 1
    assert receipt["stop"] == "complete"
    assert 0 < receipt["estimated_spend_including_unknown_requests_usd"] < .05
    assert gen.run([spec], tmp_path, .05, 1, "fixture-key") == receipt
    assert len(calls) == 2
