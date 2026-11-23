from collections import Counter

import numpy as np
import pytest

from superturn.turn_text_e5_scale import mixture, packed_overlap, replay_rows, scale_delta


def test_replay_matches_english_label_mass_with_balanced_repetition():
    hold = {"text": "---\none", "language": "en", "label": "hold"}
    other_hold = {**hold, "text": "---\ntwo"}
    shift = {**hold, "text": "---\nthree", "label": "shift"}
    hindi = {**shift, "language": "hi"}
    base = [hold, other_hold, shift, hindi]
    extra = [hold, shift] + [{**hold, "text": f"---\nnew {i}"} for i in range(4)]
    replay = replay_rows(base, extra)
    assert replay[:len(base)] == base
    assert mixture(replay) == mixture(base + extra)
    assert [row["label"] for row in replay] == [row["label"] for row in base + extra]
    assert replay.count(hindi) == 1
    counts = Counter(row["text"] for row in replay if row["label"] == "hold")
    assert counts.keys() == {hold["text"], other_hold["text"]}
    assert abs(counts[hold["text"]] - counts[other_hold["text"]]) == 1


def test_packed_overlap_rejects_cross_role_token_collision(monkeypatch):
    monkeypatch.setattr("superturn.turn_text_e5_scale.tokenize_fit_input",
                        lambda tokenizer, text, backbone: ([len(text)], [1]))
    with pytest.raises(ValueError, match="overlaps.*test"):
        packed_overlap(None, {"train": [{"text": "one"}], "test": [{"text": "two"}]})


def test_scale_must_beat_the_stronger_baseline():
    y = np.array([True, True, False, False])
    scores = {
        "replay": [np.array([0.1, .2, .3, .4])],
        "b1": [np.array([.9, .8, .3, .4])],
        "scale": [np.array([.9, .2, .3, .4])],
    }
    assert scale_delta(scores, y, np.arange(4), [0]) == -.5
