import json

from superturn import turn_generation as generator
from superturn.turn_generation import CATALOGUE
from superturn.turn_text_scale_generation import TOTALS, build_plan, clean_scale_row, scale_examples
from superturn.turn_text_scale_trial import evaluate_dialogues, harper_continuous_turns
from superturn.turn_text_train import scorecard


def test_scale_plan_is_disjoint_and_exact():
    plan = build_plan(json.loads(CATALOGUE.read_text()))
    assert len({row["slot"] for row in plan}) == len(plan) == sum(TOTALS.values())
    scenario = {role: {row["scenario"] for row in plan if row["split"] == role}
                for role in ("train", "validation", "test")}
    assert len(scenario["validation"]) == len(scenario["test"]) == 6
    assert not scenario["train"] & (scenario["validation"] | scenario["test"])
    assert not scenario["validation"] & scenario["test"]


def test_scale_validates_multiple_holds_and_short_shifts():
    spec = {"slot": "1", "scale_family": "G1", "shape": "G1", "prior": "none", "split": "train"}
    raw = {"previous_agent": "", "caller": "I called the repair shop this morning ‖ "
           "but the manager said the part had not arrived ‖ so can you check which day will work",
           "next_agent": "I can check our delivery schedule"}
    row = {**spec, **clean_scale_row(raw, spec)}
    assert [item["label"] for item in scale_examples(row)] == ["hold", "hold", "shift"]
    block = {("this", "morning", "but", "the", "manager", "said", "the", "part")}
    accepted, rejected = evaluate_dialogues([row], {"1": spec}, {}, block)
    assert not accepted and rejected == {"1": "benchmark_eight_word_overlap"}
    question = {"previous_agent": "Which shelf was it on?", "caller": "the bottom one", "next_agent": "I will check"}
    short = {"prior": "usual", "shape": "G3-short"}
    assert clean_scale_row(question, short)
    assert clean_scale_row({**question, "caller": "okay"}, short) is None
    assert clean_scale_row({**question, "next_agent": "i will check your account"}, short) is None
    assert clean_scale_row({**question, "next_agent": "try four one five four two five nine"}, short) is None


def test_scorecard_supports_shift_only_and_several_holds():
    rows = [{"text": "---\ntext", "language": "en", "slot": slot, "label": label}
            for slot, label in (("1", "hold"), ("1", "hold"), ("1", "shift"), ("2", "shift"))]
    report = scorecard([.2, .8, .6, .7], rows)["all"]
    assert report["dialogue_pairs"] == 2 and report["comparable_dialogues"] == 1
    assert report["shift_only_dialogues"] == 1 and report["within_dialogue_ranking"] == .5


def test_overlap_transcript_joins_segments_from_the_same_speaker_only():
    parts = [("caller", "Can you check"), ("caller", "which times are available next week?"),
             ("agent", "I can look"), ("agent", ""), ("agent", "The calendar is open")]
    rows = [{"index": index, "start_timestamp_ms": index * 400, "speaker_role": speaker,
             "human_transcript": text} for index, (speaker, text) in enumerate(parts)]
    assert list(harper_continuous_turns(reversed(rows))) == [
        "can you check which times are available next week", "i can look", "the calendar is open",
    ]


def test_retry_keeps_a_separate_spend_reservation(monkeypatch, tmp_path):
    spec = {"scale_family": "G3", "slot": "en/one", "language": "en", "criterion": "example",
            "native_script": "Latn", "pause_style": "G3", "split": "train"}
    delays = []

    def fake_batch(rows, key, attempt):
        if attempt == 1:
            raise generator.ApiError("HTTP 503 from model")
        assert attempt == 2
        return [{**rows[0], "previous_agent": "where will you be", "caller_before_pause": "at home",
                 "caller_after_pause": "", "next_agent": "someone can meet you there"}], .006

    monkeypatch.setattr(generator, "generate_batch", fake_batch)
    monkeypatch.setattr(generator.time, "sleep", delays.append)
    receipt = generator.run([spec], tmp_path, .10, 1, "test")
    assert receipt["stop"] == "complete" and receipt["accepted_dialogues"] == 1
    assert receipt["estimated_spend_including_unknown_requests_usd"] == .044
    assert receipt["provider_backoffs"] == 1 and delays == [5]
