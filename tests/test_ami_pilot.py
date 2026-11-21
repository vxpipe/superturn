from dataclasses import replace

import numpy as np

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
