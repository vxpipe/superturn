from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from superturn import ami_mixture
from superturn.ami_activity import anchors
from superturn.ami_calibration import development_splits
from superturn.ami_crossval import assign_folds
from superturn.ami_floor_audit import floor_vote, matched_floor_intervals, reference_context
from superturn.ami_deadline import retrospective_limits
from superturn.ami_dialogue_audit import Act, classify
from superturn.ami_learning import select_speakers
from superturn.ami_more_turns import extra_turns
from superturn.ami_text_signal import preceding
from superturn.ami_transfer import partial_auc
from superturn.ami_pilot import acoustic_status, annotated_pauses, group_words, match_training
from superturn.labels import Label, Speech


def test_text_signal_uses_word_end_and_excludes_future_words():
    assert preceding([(0, 8000, "first"), (9000, 9600, "last"), (9400, 9700, "tail")], 9600) == [
        (0, 8000, "first"), (9000, 9600, "last"),
    ]


def test_dialogue_act_proxy_distinguishes_a_hold_inside_an_act_from_a_new_act():
    row = {"pause_start": 9600, "speaker": "caller", "label": "hold", "annotated_backchannel": False}
    first = (8000, 9600)
    later = (14500, 18000)
    tracks = {"caller": ({"first": first, "later": later}, {"first": [Act("a", "Inform", (first, later))]})}
    assert classify(row, tracks, set()) == ("interior-or-fragment-hold", 0)
    tracks["caller"][1]["first"] = [Act("a", "Inform", (first,), terminal_known=False)]
    assert classify(row, tracks, set()) == ("uncovered", None)
    tracks["caller"][1]["first"] = [Act("a", "Inform", (first,))]
    assert classify(row, tracks, set()) == ("terminal-act-hold", 1)
    tracks["answerer"] = ({"answer": later}, {"answer": [Act("b", "Inform", (later,))]})
    row["label"] = "shift"
    assert classify(row, tracks, {("a", "b")}) == ("other-shift", 1)
    tracks["caller"][0].pop("later")
    assert classify(row, tracks, {("a", "b")}) == ("response-or-elicit-shift", 1)


def test_retrospective_limits_keep_threshold_ties_together():
    assert retrospective_limits([.9, .8, .2, .1], [1, 1, 0, 0]) == {
        "max_recall_at_5pct_false_complete": 1,
        "min_false_complete_at_80pct_recall": 0,
    }
    assert retrospective_limits([.9, .8, .8, .1], [1, 1, 0, 0]) == {
        "max_recall_at_5pct_false_complete": .5,
        "min_false_complete_at_80pct_recall": .5,
    }


def test_floor_audit_matches_short_overlapping_pause_without_fixed_time():
    row = Label("ami", "en", "a", "meeting", "group", "train", 6200, 8000,
                "shift", "fixture", "fixture.wav")
    caller = Speech("a", 0, 6000)
    reply = Speech("b", 7600, 26000)
    assert not list(annotated_pauses([caller, reply], {}))
    intervals = list(annotated_pauses([caller, reply], {}, minimum_gap=1))
    matches = matched_floor_intervals(row, intervals)
    assert floor_vote(row, matches) == "agrees"
    assert floor_vote(replace(row, label="hold"), matches) == "opposite label"
    assert floor_vote(row, matched_floor_intervals(replace(row, pause_start=7600), intervals)) == "no matching reference pause"
    ambiguous = intervals + [(Speech("a", 9000, 9900), Speech("a", 12000, 15000), "hold", False)]
    assert floor_vote(row, matched_floor_intervals(row, ambiguous)) == "multiple matching reference pauses"


def test_unmatched_floor_context_keeps_history_and_overlap_separate():
    row = Label("ami", "en", "a", "meeting", "group", "train", 40000, 4800,
                "hold", "fixture", "fixture.wav")
    words = [Speech("a", 0, 8000), Speech("b", 41000, 42000), Speech("a", 44800, 45000)]
    assert reference_context(row, words) == {
        "no_caller_word_in_preceding_2s": True,
        "caller_word_overlaps_pause": False,
        "other_word_overlaps_pause": True,
        "no_annotated_word_overlaps_pause": False,
    }


def test_annotation_labels_use_backchannels_and_group_following_words():
    caller = Speech("a", 0, 6400)
    brief = Speech("b", 12000, 14000)
    resume = Speech("a", 17000, 18500)
    assert not list(annotated_pauses(group_words([caller, brief, resume]), {}))
    assert list(annotated_pauses(group_words([caller, brief, resume]), {"b": [12000]})) == [
        (caller, brief, "hold", True)
    ]
    reply = Speech("b", 16000, 35000)
    complete = Speech("b", 12000, 35000)
    assert list(annotated_pauses(group_words([caller, brief, reply]), {"b": [12000]})) == [
        (caller, complete, "shift", False)
    ]
    separated_reply = Speech("b", 20000, 39500)
    late_caller = Speech("a", 48000, 49500)
    pauses = list(annotated_pauses(group_words([caller, brief, separated_reply, late_caller]),
                                  {"b": [12000]}))
    assert all(first != caller for first, _, _, _ in pauses)


def test_acoustic_screen_requires_quiet_and_dominant_headsets():
    tracks = {speaker: np.zeros(26000, dtype=np.int16) for speaker in ("a", "b")}
    caller, following = Speech("a", 0, 6400), Speech("b", 12000, 20000)
    tracks["a"][4000:6400] = 4000
    tracks["b"][12000:14000] = 4000
    assert acoustic_status(caller, following, tracks) == "passed"
    assert acoustic_status(caller, following, tracks, [Speech("b", 7200, 7600)]) == "timed nonword event in pause"
    tracks["b"][8000:9000] = 3000
    assert acoustic_status(caller, following, tracks) == "speech or noise in pause"


def test_ami_match_preserves_speaker_and_label_counts():
    a = Label("ami", "en", "a", "meeting", "group", "train", 6400, 4800,
              "hold", "per-track", "a.wav")
    b = replace(a, speaker="b", label="shift", audio="b.wav")
    selected_current, selected_annotation = match_training(
        [a, b, replace(a, pause_start=8000)],
        [replace(a, pause_start=6000), replace(b, pause_start=6900), replace(b, label="hold")],
    )
    assert [(row.speaker, row.label) for row in selected_current] == [("a", "hold"), ("b", "shift")]
    assert [(row.speaker, row.label) for row in selected_annotation] == [("a", "hold"), ("b", "shift")]


def test_auxiliary_holds_match_languages_and_cap_the_pause(tmp_path, monkeypatch):
    length = ami_mixture.CONTEXT + ami_mixture.GAP + 1
    audio = np.ones(length, dtype="<i2").tobytes()
    rows = []
    for source, silence_start in (("common_voice", 32000), ("emilia_yodas", None)):
        for language in ("ja", "ko"):
            rows.append({"source": source, "language": language, "speaker": f"voice-{language}",
                         "audio": f"{source}/{language}.wav", "pause_start": 20000,
                         "pause_samples": 8000, "label": "hold", "silence_start": silence_start,
                         "pcm": audio})
    path = tmp_path / "auxiliary.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    monkeypatch.setattr(ami_mixture, "condition", lambda pcm, mode, seed: pcm)
    auxiliary, quotas = ami_mixture._auxiliary([path], 2)
    assert quotas == {"ja": 1, "ko": 1}
    assert {row["source"] for row in auxiliary["read"]} == {"common_voice"}
    assert {row["source"] for row in auxiliary["natural"]} == {"emilia_yodas"}
    assert all(row["pause_samples"] == ami_mixture.GAP and row["pcm"][-2:] == bytes(2)
               for rows in auxiliary.values() for row in rows)
    with pytest.raises(ValueError, match="not enough language-matched"):
        ami_mixture._auxiliary([path], 3)


def test_calibration_keeps_connected_recordings_together():
    groups = {f"m{meeting}": f"g{meeting // 2}" for meeting in range(18)}
    groups["extra"] = "g0"
    assignments = development_splits(list(groups), groups)
    assert assignments == development_splits(list(reversed(groups)), groups)
    assert assignments["m0"] == assignments["m1"] == assignments["extra"] == "train"
    assert set(assignments.values()) == {"train", "validation", "calibration"}
    for meeting in range(0, 18, 2):
        assert assignments[f"m{meeting}"] == assignments[f"m{meeting + 1}"]


def test_partial_auc_handles_tied_scores_at_the_right_edge():
    truth = [1, 1, 0, 0]
    assert partial_auc([0.9, 0.8, 0.2, 0.1], truth) == 1
    assert partial_auc([0.1, 0.2, 0.8, 0.9], truth) == 0
    assert partial_auc([0.5] * 4, truth) == pytest.approx(0.05)


def test_learning_curve_is_nested_and_keeps_whole_speakers():
    rows = [Label("ami", "en", f"{group}-{voice}", group, group, "train",
                  pause, 4800, "hold", "annotation", "fixture.wav")
            for group in ("a", "b") for voice in range(8) for pause in (6400, 20000)]
    quarter = select_speakers(rows, 0.25)
    half = select_speakers(rows, 0.5)
    assert len(quarter) == 8
    assert len(half) == 16
    assert set(quarter) < set(half) < set(rows)
    assert set(quarter) == set(select_speakers(list(reversed(rows)), 0.25))
    for group in ("a", "b"):
        assert {row.pause_start for row in quarter if row.group == group} == {6400, 20000}
    for speaker in {row.speaker for row in quarter}:
        assert sum(row.speaker == speaker for row in quarter) == 2


def test_activity_anchor_depends_on_past_frames_and_uses_disjoint_future_bins():
    activity = np.zeros((2, 60), dtype=bool)
    activity[0, 27] = True
    activity[1, 31] = True
    activity[0, 34] = True
    observed = {frame: (gap, target) for frame, gap, target in anchors(activity, 0)}
    assert observed[30] == (200, (0, 1, 0, 0, 1, 0, 0, 0))
    activity[:, 30:] = False
    silent_future = {frame: (gap, target) for frame, gap, target in anchors(activity, 0)}
    assert silent_future[30] == (200, (0,) * 8)
    activity[1, 29] = True
    assert 30 not in {frame for frame, _, _ in anchors(activity, 0)}


def test_crossval_retains_largest_component_and_balances_disjoint_folds():
    rows = [{"group": str(group)} for group, count in enumerate([12, 4, 4, 3, 2, 2, 1, 1])
            for _ in range(count)]
    assignments = assign_folds(rows, count=3)
    assert assignments == assign_folds(reversed(rows), count=3)
    assert assignments["0"] is None
    assert set(assignments.values()) == {None, 0, 1, 2}
    counts = [sum(assignments[row["group"]] == fold for row in rows) for fold in range(3)]
    assert max(counts) - min(counts) <= 1


def test_extra_turns_preserve_fit_voice_group_and_cap():
    a = Label("ami", "en", "a", "meeting", "fit", "train", 6400, 4800,
              "hold", "annotation", "fixture.wav")
    baseline = [{"speaker": "a", "group": "fit", "audio": "fixture.wav", "pause_start": pause}
                for pause in range(6390, 6401)]
    added = replace(a, pause_start=10000)
    candidates = [a, replace(a, speaker="b"), replace(a, group="held", pause_start=8000), added, added]
    assert extra_turns(baseline, candidates) == [added]


def test_reference_scoring_never_retrains_a_missing_checkpoint(tmp_path, monkeypatch):
    import importlib

    pytest.importorskip("torch")
    from superturn.ami_pilot import fit_and_score

    train_module = importlib.import_module("superturn.train")
    monkeypatch.setattr(train_module, "completed_run", lambda *args: None)
    monkeypatch.setattr(train_module, "train", lambda *args: pytest.fail("reference checkpoint would be overwritten"))
    with pytest.raises(ValueError, match="completed reference checkpoint"):
        fit_and_score(tmp_path, "reference", {}, tmp_path, (), tmp_path, "cpu", expected_checkpoint="sha")
