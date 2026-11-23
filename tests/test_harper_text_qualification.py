from collections import Counter
import json

import numpy as np

from superturn import harper_text_qualification as qualification
from superturn.harper_text_qualification import QUOTAS, Candidate, select
from superturn.harper_text_review import Window
from superturn.turn_text_train import digest


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


def test_confirmation_and_continuous_gates_are_independent(tmp_path, monkeypatch):
    (tmp_path / ".campaign-machine").touch()
    source = tmp_path / "cache/harper-valley/text-dev-v1.jsonl"
    source.parent.mkdir(parents=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    draft, windows, rows, population = {}, {}, [], Counter()

    def proposal(label, stop):
        return {"outcome": label, "stop_ms": stop, "onset_ms": stop + 500,
                "next_channel": "left" if label == "hold" else "right"}

    def judgment(label, stop):
        return {"focal_channel": "left", "stop_ms": stop, "next_onset_ms": stop + 500, "brief_ack": "no",
                "outcome": "resume" if label == "hold" else "handoff", "next_channel": proposal(label, stop)["next_channel"]}

    for label, short in (("hold", 95), ("shift", 14)):
        for index in range(179):
            ident = f"{label}-{index}"
            rows.append({"call": ident, "caller": index, "caller_sample": 8000, "label": label})
            band = "300_to_800_ms" if index < short else "over_800_ms"
            population[f"{label}/{band}"] += 1
            if index < QUOTAS[label, "300_to_800_ms"] or short <= index < short + QUOTAS[label, "over_800_ms"]:
                windows[ident] = {"call": ident, "anchor_ms": 4000, "cohort": f"{label}/{band}/other",
                                  "qualification_target": {"native_stop": 8000, "caller": index, "label": label},
                                  "proposals": [proposal(label, 4000)]}
                draft[ident] = {"reviewed": True, "events": [judgment(label, 4000)]}
    for index in range(20):
        ident = f"continuous-{index}"
        windows[ident] = {"call": ident, "anchor_ms": None, "cohort": "continuous",
                          "proposals": [proposal("hold", 4000), proposal("shift", 7000)]}
        draft[ident] = {"reviewed": True, "events": [judgment("hold", 4000), judgment("shift", 7000)]}
    for index in range(20):
        ident = f"excluded-{index}"
        windows[ident] = {"call": ident, "anchor_ms": 4000, "cohort": f"excluded/reason-{index // 4}",
                          "proposals": [proposal("hold", 4000)]}
        draft[ident] = {"reviewed": True, "events": []}
    source.write_text("\n".join(map(json.dumps, rows)) + "\n")
    receipt = {"qualification_candidate_pool": {"causal_pack_sha256": digest(source),
                                                 "population_by_class_and_gap": dict(population)},
               "private_reference_sha256": "synthetic"}
    monkeypatch.setattr(qualification, "verified_packet", lambda *_: ({"windows": windows}, receipt))
    ballot = tmp_path / "ballot.json"
    report = reports / "harper-text-qualification-listening-score.json"

    def run():
        ballot.write_text(json.dumps({"draft": {"windows": draft}}))
        qualification.score(tmp_path, ballot)
        return json.loads(report.read_text())

    draft["hold-0"]["events"][0]["brief_ack"] = "unsure"
    assert run()["passed_selected_event_confirmation"]
    draft["hold-1"]["events"] = []
    for index in range(3):
        windows[f"continuous-{index}"]["proposals"].pop()
    result = run()
    assert not result["passed_selected_event_confirmation"]
    assert not result["passed_continuous_event_screen"]
    assert result["by_tolerance_ms"]["250"]["continuous"]["recall_by_class"] == {"hold": 1, "shift": .85}
    assert result["by_tolerance_ms"]["250"]["overall"]["by_proposed_class"]["hold"]["agreed_outcomes"] == 58
