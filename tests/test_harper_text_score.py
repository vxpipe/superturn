import pytest

from superturn.harper_text_score import breakdown, open_qualified_pack


def test_missing_human_qualification_stops_before_checkpoint_or_pack_reads(tmp_path):
    (tmp_path / ".campaign-machine").touch()
    with pytest.raises(ValueError, match="label qualification is missing"):
        open_qualified_pack(tmp_path, tmp_path / "absent-checkpoint.pt")


def test_breakdown_preserves_one_class_stratum():
    rows = [
        {"label": "hold", "gap_ms": 450, "caller": 1, "call": "a"},
        {"label": "shift", "gap_ms": 950, "caller": 2, "call": "b"},
    ]
    result = breakdown(rows, [.1, .7])
    assert result["english"]["balanced_accuracy"] == 1
    assert result["english"]["true_shifts"] == 1
    assert result["300_to_800_ms"]["balanced_accuracy"] is None
    assert result["300_to_800_ms"]["holds"] == 1
    assert result["over_800_ms"]["balanced_accuracy"] is None
