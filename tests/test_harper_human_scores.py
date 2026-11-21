import json
from hashlib import sha256

import pytest

from superturn.harper_human_scores import evaluate, judgments, score_packet


def test_matched_cutoff_keeps_ties_and_abstentions_indivisible():
    labels = {f"hold-{i}": 0 for i in range(20)}
    labels.update({"shift-1": 1, "shift-2": 1, "shift-3": 1})
    scores = dict.fromkeys(labels, 0)
    scores.update({"hold-0": 60, "hold-1": 60, "shift-1": 90, "shift-2": 60, "shift-3": None})
    result = evaluate(labels, scores)
    assert result["matched_5_percent"]["shifts_detected"] == 1
    assert result["next_looser_threshold"]["false_completes"] == 2
    assert result["next_looser_threshold"]["shifts_detected"] == 2
    assert result["abstained_shifts"] == 1
    assert result["curve"][-1]["shifts_detected"] == 2


@pytest.mark.parametrize("body", ["a,,\n", "a,30,yes\n", "a,1,\na,2,\n", "b,30,\n", ""])
def test_rejects_incomplete_or_ambiguous_ballots(body):
    with pytest.raises(ValueError):
        judgments(("example,probability_percent,abstain\n" + body).encode(), {"a"})


def test_packet_requires_matching_seal(tmp_path, monkeypatch):
    (tmp_path / ".campaign-machine").touch()
    (tmp_path / "reports").mkdir()
    (tmp_path / "packet" / "listening").mkdir(parents=True)
    key = tmp_path / "packet/key.json"
    key.write_text(json.dumps({"a": {"label": 0}, "b": {"label": 1}}))
    manifest = tmp_path / "packet/audio-checksums.json"
    manifest.write_text(json.dumps({"a": "fixture-a", "b": "fixture-b"}))
    guide = tmp_path / "packet/listening/GUIDE.md"
    guide.write_text("fixture guide")
    receipt = tmp_path / "reports/harper-human-baseline-packet.json"
    receipt.write_text(json.dumps({
        "packet": "packet/listening", "decision_ms": 300, "clips": 2, "verified_clips": 2,
        "guide_sha256": sha256(guide.read_bytes()).hexdigest(),
        "key_sha256": sha256(key.read_bytes()).hexdigest(),
        "audio_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr("superturn.harper_human_scores.PACKET_SHA256", sha256(receipt.read_bytes()).hexdigest())
    (tmp_path / "packet/listening/reviewer-a.csv").write_text("example,probability_percent,abstain\nb,,\na,,\n")
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("example,probability_percent,abstain\nb,80.50,\na,,yes\n")
    report, = score_packet(tmp_path, {"a": ratings})["reviewers"]
    assert report["abstained_holds"] == 1
    assert report["matched_5_percent"]["threshold_percent"] == "80.50"
    with pytest.raises(ValueError, match="duplicate reviewer ballot"):
        score_packet(tmp_path, {"a": ratings, "b": ratings})
    assert report["matched_5_percent"]["meets_nominal_80_5_on_packet"]
    key.write_text("{}")
    with pytest.raises(ValueError, match="sealed key"):
        score_packet(tmp_path, {"a": ratings})
    receipt.write_text(json.dumps({"key_sha256": sha256(key.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="pinned packet seal"):
        score_packet(tmp_path, {"a": ratings})
