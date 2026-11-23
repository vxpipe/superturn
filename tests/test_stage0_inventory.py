import io

import pyarrow as pa
import pyarrow.parquet as pq

from superturn import diarbench_inventory
from superturn import basis_stage0
from superturn.basis_english_pilot import vote_census
from superturn.basis_english_reference import match_reference
from superturn.basis_split import Call
from superturn.basis_stage0 import four_roles_possible
from superturn.basis_stage0_yield import count_gaps, json_lines
from superturn.basis_stage1 import events


def test_english_vote_census_keeps_all_annotated_calls_separate_from_consensus():
    rows = [
        {"conversation_id": "a", "question": "pause", "label": "holding",
         "votes": [{"option": "holding", "count": 4}, {"option": "normal_gap", "count": 1}]},
        {"conversation_id": "b", "question": "pause", "label": "normal_gap",
         "votes": [{"option": "normal_gap", "count": 2}]},
        {"conversation_id": "c", "question": "overlap"},
    ]
    summary, priorities, annotated = vote_census(rows, {"a": "fit", "b": "fit"}, {"a", "b", "c"})
    assert annotated == {"a", "b", "c"}
    assert priorities == {"a": {"hold": 1}}
    assert summary["fit"]["pause_finished"] == 1
    assert summary["fit"]["consensus_finished"] == 0
    assert summary["excluded"]["overlap"] == 1


def test_english_reference_matches_before_comparing_the_outcome():
    judgments = [
        {"speakers": ["a"], "start": 3.0, "end": 3.5, "label": "holding"},
        {"speakers": ["b"], "start": 8.0, "end": 8.5, "label": "finished"},
        {"speakers": ["a"], "start": 17.0, "end": 17.5, "label": "holding"},
    ]
    proposals = [(48_000, 56_000, "shift", "a", "b"),
                 (128_000, 136_000, "shift", "b", "a")]
    coverage, confusion = match_reference(judgments, proposals, [], {"holding": "hold", "finished": "finished"})
    assert coverage["hold_matched"] == coverage["hold_unmatched"] == coverage["finished_handoff_unconfirmed"] == 1
    assert confusion == {"hold_predicted_shift": 1}


def test_basis_pause_rule_does_not_promote_a_backchannel_to_a_shift():
    counts, proposals = events([(20_000, 64_000, "a"), (72_000, 80_000, "b"),
                                (88_000, 120_000, "a")],
                               {speaker: [(0, 128_000)] for speaker in ("a", "b")}, [], 128_000)
    assert counts["short_other"] == counts["focal_backchannel"] == 1
    assert not proposals


def test_four_roles_preserve_whole_speaker_components():
    assert not four_roles_possible([530])
    assert not four_roles_possible([402, 6, 2, 2, 2, 2, 2])
    assert four_roles_possible([2, 2, 29, 29, 1, 1])


def test_metadata_capacity_counts_resumes_and_substantive_handoffs():
    counts, speakers = count_gaps([(0, 4_000, "a"), (4_400, 5_000, "a"),
                                   (5_300, 7_100, "b"), (7_700, 8_100, "a")])
    assert counts == {"hold": 1, "shift": 1, "short_other": 1}
    assert speakers == {"hold": {"a"}, "shift": {"a"}}


def test_timing_json_reader_handles_split_records_and_final_line():
    class Chunked(io.BytesIO):
        def read(self, size):
            return super().read(min(size, 4))

    class Meter:
        def __init__(self):
            self.bytes = 0

        def advance(self, size):
            self.bytes += size

    payload = b'{"a": 1}\n{"b": 2}'
    meter = Meter()
    assert list(json_lines(Chunked(payload), meter)) == [{"a": 1}, {"b": 2}]
    assert meter.bytes == len(payload)


def test_fit_audio_candidates_exclude_mutes_and_short_responses():
    activity = [(0, 64_000, "a"), (70_400, 80_000, "a"), (86_400, 113_600, "b"),
                (123_200, 129_600, "a")]
    connected = {"a": [(0, 200_000)], "b": [(0, 200_000)]}
    result, candidates = events(activity, connected, [], 200_000)
    assert result == {"hold": 1, "shift": 1, "short_other": 1}
    assert [candidate[2] for candidate in candidates] == ["hold", "shift"]
    result, candidates = events(activity, connected, [(0, 64_000)], 200_000)
    assert result == {"muted": 2, "short_other": 1}
    assert not candidates


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
