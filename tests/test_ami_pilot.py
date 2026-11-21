from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from superturn import ami_mixture
from superturn.ami_calibration import development_splits
from superturn.ami_learning import select_speakers
from superturn.ami_transfer import partial_auc
from superturn.ami_pilot import acoustic_status, annotated_pauses, group_words, match_training
from superturn.labels import Label, Speech


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
