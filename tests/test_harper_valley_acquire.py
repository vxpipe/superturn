from hashlib import sha256
from io import BytesIO
import json

import pytest

from superturn import harper_valley_acquire as acquire
from superturn.harper_valley_plan import REVISION, SPLITS


def test_selection_hash_and_exact_paths_are_verified(tmp_path):
    cache = tmp_path / "cache" / "harper-valley"
    cache.mkdir(parents=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    selection = {}
    for index, split in enumerate(SPLITS):
        sid = f"{index:016x}"
        paths = [f"data/audio/{role}/{sid}.wav" for role in ("caller", "agent")] + [f"data/transcript/{sid}.json"]
        selection[split] = [{"id": sid, "caller": 2 * index, "agent": 2 * index + 1,
                             "files": [{"path": path, "bytes": 3} for path in paths]}]

    def write():
        payload = json.dumps(selection).encode()
        expected = sha256(payload).hexdigest()
        (cache / f"{REVISION}-selection.json").write_bytes(payload)
        (reports / "harper-valley-plan.json").write_text(json.dumps({
            "selection_sha256": expected, "revision": REVISION, "metadata_bytes": 10, "projected_bytes": 37,
            "splits": {split: {"calls": 1, "bytes": 9} for split in SPLITS},
        }))
        return expected

    expected = write()
    assert acquire.verified_selection(tmp_path, expected) == selection
    with pytest.raises(ValueError, match="hash mismatch"):
        acquire.verified_selection(tmp_path, "bad")
    selection["validation"][0]["agent"] = selection["train"][0]["agent"]
    with pytest.raises(ValueError, match="split constraints"):
        acquire.verified_selection(tmp_path, write())
    selection["validation"][0]["files"][0]["path"] = "../escape"
    with pytest.raises(ValueError, match="payload path"):
        acquire.verified_selection(tmp_path, write())


@pytest.mark.parametrize("payload", [b"ab", b"abcd"])
def test_wrong_size_file_is_not_staged(tmp_path, monkeypatch, payload):
    file = {"path": "data/audio/test.wav", "bytes": 3}
    monkeypatch.setattr(acquire, "_request", lambda _: BytesIO(payload))
    with pytest.raises(ValueError, match="planned"):
        acquire.fetch_file(tmp_path, file)
    assert not list(tmp_path.glob("data/audio/*"))


def test_exact_size_file_is_staged_and_reused(tmp_path, monkeypatch):
    file = {"path": "data/audio/test.wav", "bytes": 3}
    monkeypatch.setattr(acquire, "_request", lambda _: BytesIO(b"abc"))
    assert acquire.fetch_file(tmp_path, file)
    assert (tmp_path / file["path"]).read_bytes() == b"abc"
    monkeypatch.setattr(acquire, "_request", lambda _: pytest.fail("reused file should not make a request"))
    assert not acquire.fetch_file(tmp_path, file)
