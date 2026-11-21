from collections import Counter

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
