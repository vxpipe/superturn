from collections import Counter

import pytest

from superturn.splits import assign, speaker_groups, split_recordings


def test_split_is_reproducible_and_order_independent():
    recordings = [(f"conv{i}", [f"spk{i}", f"spk{i + 1000}"]) for i in range(200)]
    first = split_recordings("icsi", "train-conversation", recordings)
    second = split_recordings("icsi", "train-conversation", list(reversed(recordings)))
    assert first == second


def test_no_speaker_crosses_splits_when_speakers_repeat_across_recordings():
    # Series of four meetings with the same four people, as in AMI.
    recordings = []
    for series in range(50):
        people = [f"p{series}-{k}" for k in range(4)]
        recordings += [(f"S{series}{part}", people) for part in "abcd"]
    splits = split_recordings("ami", "train-conversation", recordings)
    speaker_split: dict[str, set[str]] = {}
    for recording, speakers in recordings:
        for speaker in speakers:
            speaker_split.setdefault(speaker, set()).add(splits[recording])
    assert all(len(s) == 1 for s in speaker_split.values())


def test_every_dataset_is_split_in_roughly_the_agreed_shares():
    for source in ("icsi", "ami", "voxpopuli"):
        counts = Counter(assign(source, f"g{i}", "train-read") for i in range(5000))
        assert 0.77 < counts["train"] / 5000 < 0.83
        assert 0.08 < counts["validation"] / 5000 < 0.12
        assert 0.08 < counts["test"] / 5000 < 0.12


def test_test_only_sources_never_reach_training():
    assert {assign("fleurs", f"g{i}", "test") for i in range(500)} == {"test"}


def test_groups_merge_transitively():
    groups = speaker_groups([("a", ["x"]), ("b", ["x", "y"]), ("c", ["y"]), ("d", ["z"])])
    assert groups["a"] == groups["b"] == groups["c"] == "a"
    assert groups["d"] == "d"


def test_language_stratification_preserves_speakers_and_small_language_holdouts():
    from superturn.splits import stratified_groups

    groups = [(f"en-{index}", "en") for index in range(50)]
    groups += [(f"de-{index}", "de") for index in range(6)]
    groups += [("de-0", "en"), ("pt-only", "pt"), ("it-a", "it"), ("it-b", "it")]
    split = stratified_groups("cv", "train-read", groups)
    assert split == stratified_groups("cv", "train-read", reversed(groups))
    for language in ("en", "de"):
        assert {split[group] for group, tag in groups if tag == language} == {"train", "validation", "test"}
    assert {split[group] for group, tag in groups if tag in {"pt", "it"}} == {"train"}
    assert set(stratified_groups("fleurs", "test", groups).values()) == {"test"}


def test_conversation_plan_matches_language_duration_shares_with_connected_voices():
    from superturn.conversation_splits import plan

    rows = []
    for language in ("en", "hi"):
        for duration in (800, 2100, 3200, 6100, None):
            for i in range(30):
                key = f"{language}-{duration}-{i}"
                rows.append({"pause_id": key, "recording": key, "language": language,
                             "prior_turn_ms": duration, "speakers": [key]})
    rows[1]["speakers"].append(rows[0]["recording"])
    rows[1]["customer"] = rows[2]["customer"] = "example-customer"

    result = plan(rows, "calls")
    assert result == plan(reversed(rows), "calls")
    recording_splits = result["recordings"]
    assert len({recording_splits[row["recording"]] for row in rows[:3]}) == 1
    for stratum in result["coverage"]:
        if stratum["duration_band"] == "all_durations":
            continue
        assert all(stratum["by_split"][split]["groups"] for split in ("train", "validation", "test"))
        assert abs(stratum["by_split"]["validation"]["pauses"] - 3) <= 1
        assert abs(stratum["by_split"]["test"]["pauses"] - 3) <= 1
    assert set(plan(rows, "calls", test_only=True)["recordings"].values()) == {"test"}


def test_conversation_plan_preserves_language_holdouts_when_length_bands_are_sparse():
    from superturn.splits import stratified_duration_groups

    rows = [(f"voice-{i}", "en", prior) for i, prior in enumerate((800, 1400, 2900, 6200))]
    assignment, summary = stratified_duration_groups("calls", "train-conversation", rows)
    assert Counter(assignment.values()) == {"train": 2, "validation": 1, "test": 1}
    joint = [cell for cell in summary if cell["duration_band"] != "all_durations"]
    assert len(joint) == 4 and all(cell["independent_groups"] == 1 for cell in joint)


def test_expanded_conversation_plan_cannot_move_or_merge_frozen_participants():
    from superturn.conversation_splits import plan

    rows = [
        {"pause_id": "one", "recording": "used", "language": "en", "speakers": ["a"], "fixed_split": "test"},
        {"pause_id": "two", "recording": "new", "language": "en", "speakers": ["a", "b"]},
    ]
    assert plan(rows, "calls")["recordings"] == {"new": "test", "used": "test"}
    rows[1]["fixed_split"] = "train"
    with pytest.raises(ValueError, match="bridges different fixed splits"):
        plan(rows, "calls")


def test_small_conversation_plan_repairs_joint_coverage_without_breaking_other_cells():
    from superturn.splits import stratified_duration_groups

    rows = [(str(group), "en", duration) for duration, voices in (
        (800, (0, 1, 2)), (1800, (0, 1, 4)), (4500, (0, 2, 3, 4)), (7000, (0, 2, 3, 4)),
    ) for group in voices]
    _, result = stratified_duration_groups("x", "train-conversation", rows)
    assert all(not cell["empty_splits"] for cell in result)
