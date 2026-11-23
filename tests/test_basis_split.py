from types import SimpleNamespace

from superturn.basis_split import Call, inventory, split, summarize, voices


def test_inventory_includes_all_participants_across_segments():
    paths = [
        "hi/audio/one/first.spk_a.flac",
        "hi/audio/one/second.spk_a.flac",
        "hi/audio/one/second.spk_b.flac",
    ]
    rows = [SimpleNamespace(path=path, size=10) for path in paths]
    assert inventory(rows, "hi") == [Call("one", frozenset({"spk_a", "spk_b"}), 30)]


def test_reservation_keeps_audit_and_holdout_speakers_disjoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERTURN_DATA", str(tmp_path))
    people = [frozenset((f"{index}-a", f"{index}-b")) for index in range(14)]
    calls = [Call(f"call-{index:02}", speakers, 10) for index, speakers in enumerate(people)]
    calls.extend(Call(f"bridge-{index:02}", first | second, 100)
                 for index, (first, second) in enumerate(zip(people, people[1:])))
    batches, audit, trial = split(calls, 6, 32, independent=True)

    assert summarize(calls)["components"] == 1
    assert len(audit) == 4 and len(voices(audit)) == 8
    assert set(audit) <= set(batches["train"])
    assert voices(batches["train"]).isdisjoint(voices(batches["calibration"]) | voices(batches["test"]))
    assert voices(batches["calibration"]).isdisjoint(voices(batches["test"]))
    for role in ("calibration", "test"):
        stats = summarize(batches[role])
        assert stats["speakers"] >= 6
        assert stats["components"] >= 2
    assert sum(len(batch) for batch in batches.values()) < len(calls)
    assert split(list(calls), 6, 32, independent=True) == (batches, audit, trial)
