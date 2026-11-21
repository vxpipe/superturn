"""Regression checks for source leakage and deadline rendering."""

from dataclasses import replace

import numpy as np

from superturn.dense_teacher import normalize, render, source_candidates
from superturn.labels import Label


def label(**fields):
    row = Label("emilia_yodas", "en", "one", "recording-a", "group-a", "train",
                16000, 6000, "hold", "single-track", "a.wav")
    return replace(row, **fields)


def test_connected_group_reserves_train_rows_linked_to_held_out_voice():
    rows = [
        label(),
        label(recording="bridge", group="group-b", audio="bridge.wav"),
        label(recording="held-out", group="group-b", speaker="two", split="test"),
        label(recording="unrelated", group="group-c", speaker="three", audio="safe.wav"),
    ]
    eligible, _ = source_candidates(rows)
    assert [row.audio for row in eligible] == ["safe.wav"]


def test_deadline_render_cannot_hear_audio_after_the_decision():
    row = label()
    rng = np.random.default_rng(763)
    original = rng.integers(-24000, 24000, 24000, dtype=np.int16)
    for deadline in (200, 300):
        end = row.pause_start + deadline * 16
        altered = original.copy()
        altered[end:] = -altered[end:]
        current = render(row, original, deadline)
        longer = render(row, original, deadline, samples=128000)
        assert current.shape == (48000,)
        assert longer.shape == (128000,)
        np.testing.assert_array_equal(current, render(row, altered, deadline))
        np.testing.assert_array_equal(longer, render(row, altered, deadline, samples=128000))
        np.testing.assert_array_equal(current, longer[-48000:])
        np.testing.assert_array_equal(current, normalize(current) * 32768)
