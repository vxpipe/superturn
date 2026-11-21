import json

import numpy as np

from superturn.audio import RATE, write_wave
from superturn.harper_valley_audit import audit_call, label_support, recording_origin, transcript_words
from superturn.labels import Speech


def test_recording_local_word_times_translate_to_shared_call_timeline():
    segments = [
        {"start_ms": 400, "offset_ms": 700, "word_offsets_ms": [0, 250], "word_durations_ms": [100, 100]},
        {"start_ms": 1200, "offset_ms": 1500, "word_offsets_ms": [], "word_durations_ms": []},
    ]
    origin = recording_origin(segments)
    assert origin == -4800
    assert list(transcript_words(segments, "caller", 32000, origin)) == [
        Speech("caller", 6400, 8000), Speech("caller", 10400, 12000),
    ]
    segments[-1]["start_ms"] += 100
    assert recording_origin(segments) is None


def test_shared_view_keeps_overlap_from_before_second_channel_started():
    streams = [
        Speech("agent", 0, 12000),
        Speech("caller", 4000, 8000),
        Speech("agent", 14000, 32000),
    ]
    assert not label_support(streams, streams, 3000, 32000)


def test_caller_pause_requires_speaker_order_agreement():
    streams = [Speech("caller", 0, 8000), Speech("agent", 14000, 32000)]
    mismatched = [Speech("caller", 0, 8000), Speech("caller", 14000, 32000)]
    assert label_support(streams, streams, 0, 32000) == {
        "candidate_shift": 1, "supported_shift": 1, "fixed_window_supported_shift": 1,
    }
    assert label_support(streams, mismatched, 0, 32000) == {"candidate_shift": 1, "disagreed_shift": 1}


def test_boundary_agreement_can_leave_the_fixed_query_window_unsupported():
    heard = [Speech("caller", 0, 8000), Speech("agent", 14000, 32000)]
    words = [Speech("caller", 0, 10000), Speech("agent", 16000, 34000)]
    assert label_support(heard, words, 0, 34000) == {"candidate_shift": 1, "supported_shift": 1}


def test_missing_word_timings_cannot_certify_a_silent_query():
    heard = [Speech("caller", 0, 8000), Speech("agent", 14000, 32000)]
    segments = [{"offset_ms": 560, "duration_ms": 120, "word_offsets_ms": [0], "word_durations_ms": []}]
    missing = []
    assert not list(transcript_words(segments, "caller", 32000, 0, omissions=missing))
    assert missing == [Speech("caller", 8960, 10880)]
    assert label_support(heard, heard, 0, 32000, omissions=missing) == {
        "candidate_shift": 1, "incomplete_timing_shift": 1,
    }


def test_audit_places_nonzero_channel_offsets_on_one_timeline(tmp_path):
    filename = tmp_path / "data" / "transcript" / "example.json"
    filename.parent.mkdir(parents=True)
    filename.write_text(json.dumps([
        {"speaker_role": "caller", "start_ms": 0, "offset_ms": 300,
         "word_offsets_ms": [0], "word_durations_ms": [500]},
        {"speaker_role": "agent", "start_ms": 800, "offset_ms": 600,
         "word_offsets_ms": [0], "word_durations_ms": [1200]},
    ]))
    for role, duration in (("caller", 3), ("agent", 4)):
        write_wave(tmp_path / "data" / "audio" / role / "example.wav", np.zeros(duration * RATE, dtype=np.int16))

    def detect(pcm):
        return [(4800, 12800)] if len(pcm) == 3 * RATE else [(9600, 28800)]

    assert audit_call(tmp_path, "example", detect) == {
        "audited_calls": 1, "candidate_shift": 1, "supported_shift": 1, "fixed_window_supported_shift": 1,
    }


def test_agent_clock_recovers_only_when_both_segment_halves_agree(tmp_path):
    filename = tmp_path / "data" / "transcript" / "example.json"
    filename.parent.mkdir(parents=True)

    def segment(role, start, offset, words):
        return {"speaker_role": role, "start_ms": start, "offset_ms": offset,
                "word_offsets_ms": words, "word_durations_ms": [100] * len(words)}

    segments = [
        segment("caller", 0, 0, [0, 100, 200, 300]),
        segment("agent", 1000, 3000, [0, 100, 200, 300]),
        segment("agent", 4000, 6000, [0, 100, 200, 300]),
    ]
    filename.write_text(json.dumps(segments))
    for role, duration in (("caller", 7), ("agent", 8)):
        write_wave(tmp_path / "data" / "audio" / role / "example.wav", np.zeros(duration * RATE, dtype=np.int16))

    def detect(pcm):
        return [(0, 8000)] if len(pcm) == 7 * RATE else [(16000, 36000), (64000, 84000)]

    assert audit_call(tmp_path, "example", detect)["low_audio_word_overlap"] == 1
    recovered = audit_call(tmp_path, "example", detect, recover_clock=True)
    assert recovered["agent_clock_recovered"] == recovered["audited_calls"] == 1

    segments[2] = segment("agent", 5000, 7000, [0, 100, 200, 300])
    filename.write_text(json.dumps(segments))
    assert audit_call(tmp_path, "example", detect, recover_clock=True)["agent_clock_recovered"] == 0
