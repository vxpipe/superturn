import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from superturn.owned_history_teacher import GAP, S3, S8, variants


def test_history_ablation_changes_only_observed_earlier_audio():
    rng = np.random.default_rng(19)
    pcm = rng.integers(-1000, 1000, size=S8 + GAP, dtype=np.int16)
    for observed in (S3 - 1000, S3 + 1000, S8):
        edge = observed - GAP
        available, (short, full, ablated) = variants(pcm, edge, "fixture")
        assert available == observed
        np.testing.assert_array_equal(short, full[-S3:])
        np.testing.assert_array_equal(full[-S3:], ablated[-S3:])
        np.testing.assert_array_equal(full[:S8 - observed], ablated[:S8 - observed])
        if observed <= S3:
            np.testing.assert_array_equal(full, ablated)
        else:
            assert not np.array_equal(full[-observed:-S3], ablated[-observed:-S3])

        after_deadline = pcm.copy()
        after_deadline[observed:] = 2000
        _, replay = variants(after_deadline, edge, "fixture")
        for first, second in zip((short, full, ablated), replay, strict=True):
            np.testing.assert_array_equal(first, second)
