from collections import Counter

import numpy as np

from superturn.harper_text_qualification import QUOTAS, Candidate, select
from superturn.harper_text_review import Window


def test_qualification_keeps_all_short_shifts_and_caps_anchored_calls():
    candidates = []
    short = set()
    for group, ((label, band), quota) in enumerate(QUOTAS.items()):
        mandatory = label == "shift" and band == "300_to_800_ms"
        for index in range(quota if mandatory else quota + 10):
            rank = len(candidates)
            window = Window(f"call-{group * 20 + index // 2}", index // 3, 0, rank, np.empty((0, 2)), [], 0, "fixture")
            candidates.append(Candidate(window, label, band, index < 7, rank, "question", "4-7"))
            if mandatory:
                short.add(rank)
    chosen = select(candidates)
    assert set(item.window.rank for item in chosen) >= short
    assert Counter((item.label, item.band) for item in chosen) == QUOTAS
    assert max(Counter(item.window.call for item in chosen).values()) <= 3
    assert [item.window.rank for item in select(list(reversed(candidates)))] == [
        item.window.rank for item in chosen
    ]
