import io

import pyarrow as pa
import pyarrow.parquet as pq

from superturn import diarbench_inventory
from superturn import basis_stage0
from superturn.basis_split import Call
from superturn.basis_stage0 import four_roles_possible


def test_four_roles_preserve_whole_speaker_components():
    assert not four_roles_possible([530])
    assert not four_roles_possible([402, 6, 2, 2, 2, 2, 2])
    assert four_roles_possible([2, 2, 29, 29, 1, 1])


def test_joint_cut_never_places_a_shared_voice_in_two_roles(monkeypatch):
    def call(name, *people):
        return Call(name, frozenset(people), 1)

    calls = {
        "hi": [call("h-fit", "a", "b"), call("h-bridge1", "a", "shared"),
               call("h-bridge2", "b", "shared")],
        "en": [call("e-test", "c", "d"), call("e-bridge", "c", "shared")],
    }

    monkeypatch.setattr(basis_stage0, "candidate_rosters",
                        lambda *_: {"hi": {"fit": frozenset({"a", "b"})},
                                    "en": {"test": frozenset({"c", "d"})}})
    result = basis_stage0.joint_cut(calls, 1)
    assert result["shared_language_roster"] == 1
    assert result["joint_role_roster"] == {"fit": 3, "dev": 0, "calibration": 0, "test": 2}
    assert result["languages"]["hi"]["excluded_calls"]["calls"] == 0
    assert result["languages"]["en"]["excluded_calls"]["calls"] == 1


def test_metadata_projection_does_not_fetch_audio_or_annotations(monkeypatch):
    table = pa.table({"audio": [b"audio sample"], "annotated_transcript": ["human words"],
                      "recording_id": ["session"], "sample_id": ["clip"], "dataset_type": ["Near field"],
                      "num_speakers": [2], "duration_seconds": [5.0]})
    stream = io.BytesIO()
    pq.write_table(table, stream)
    payload = stream.getvalue()
    forbidden = []
    metadata = pq.ParquetFile(io.BytesIO(payload)).metadata.row_group(0)
    for index in (0, 1):
        column = metadata.column(index)
        start = column.dictionary_page_offset or column.data_page_offset
        forbidden.append((start, start + column.total_compressed_size))

    def request(_url, *, headers):
        start, end = (int(value) for value in headers["Range"].split("=")[1].split("-"))
        assert all(end < left or start >= right for left, right in forbidden)
        result = io.BytesIO(payload[start:end + 1])
        result.status = 206
        result.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        return result

    monkeypatch.setattr(diarbench_inventory, "_request", request)
    rows, _ = diarbench_inventory.project("Hindi/test.parquet", len(payload), "repo")
    assert rows == [{"recording_id": "session", "sample_id": "clip", "dataset_type": "Near field",
                     "num_speakers": 2, "duration_seconds": 5.0}]
