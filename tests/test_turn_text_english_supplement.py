from collections import Counter
import json
from pathlib import Path

from superturn.turn_generation import CATALOGUE, make_plan
from superturn.turn_text_english_supplement import focused_plan
from superturn.turn_text_train import read_dataset


def test_focused_plan_covers_six_cells_without_using_validation_scenarios():
    catalogue = json.loads(CATALOGUE.read_text())
    original = {row["slot"]: row for row in make_plan(catalogue)}
    splits, _ = read_dataset(Path("datasets/turn-completion-v1"))
    validation = {original[row["slot"]]["scenario"] for row in splits["validation"]}
    focused = focused_plan(catalogue, splits)
    assert len(focused) == 288
    assert len({row["scenario"] for row in focused}) == 24
    assert set(Counter(row["pause_style"] for row in focused).values()) == {48}
    assert not validation & {row["scenario"] for row in focused}
    assert all(not row["caller_first"] and not row["short_shift"] and row["domain"] != "banking" for row in focused)
