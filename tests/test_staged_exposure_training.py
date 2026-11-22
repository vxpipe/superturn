import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from superturn.staged_exposure_training import shuffle_targets


def test_shuffle_keeps_sources_language_and_geometry_intact():
    rows = [
        {"language": language, "source": source, "gap_kind": kind, "missing_history_samples": missing}
        for language, source, kind, missing in (
            ("nl", "common_voice", "tail", 0),
            ("nl", "common_voice", "tail", 0),
            ("nl", "voxpopuli_dutch", "tail", 0),
            ("nl", "common_voice", "tail", 100),
            ("nl", "common_voice", "inside", 0),
            ("pt", "common_voice", "tail", 0),
        )
    ]
    targets = np.asarray([.1, .2, .3, .4, .5, .6], dtype=np.float32)
    shuffled, coverage = shuffle_targets(rows, targets)
    np.testing.assert_array_equal(shuffled, targets[[1, 0, 2, 3, 4, 5]])
    assert coverage["nl/common_voice"] == {"rows": 4, "different_targets": 2, "singleton_rows": 2}
    assert coverage["nl/voxpopuli_dutch"]["different_targets"] == 0
