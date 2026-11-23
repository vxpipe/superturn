import json

import numpy as np

from superturn.harper_streaming_capture import Episode, capture_batch, clean_text, repeat_diagnostics
from superturn.timed_text_arrival import PauseTrace, Revision, pack_trace_file, replay


class Clock:
    ns = 1_000_000_000

    def now(self):
        return self.ns

    def wait(self, seconds):
        self.ns += round(seconds * 1_000_000_000)


class Stream:
    received = 0
    real_received = 0
    decoded = 0

    def accept_waveform(self, rate, samples):
        assert rate == 16000
        self.received += len(samples)
        self.real_received += np.count_nonzero(samples)

    def input_finished(self):
        pass


class Recognizer:
    def __init__(self, clock):
        self.clock = clock

    def create_stream(self):
        return Stream()

    def is_ready(self, stream):
        return stream.received >= (stream.decoded + 1) * 5120

    def decode_streams(self, streams):
        self.clock.wait(.010)
        for stream in streams:
            stream.decoded += 1

    def get_result(self, stream):
        real_decoded = min(stream.decoded, stream.real_received // 5120)
        return ("hello 5 there" if real_decoded > 1 else "hello 5") if real_decoded else ""


def test_paced_capture_replays_only_received_revisions():
    row = {"call": "generated", "caller_sample": 12800, "label": "shift"}
    case = Episode(row, 0, 8000, np.ones(17600), 12800, None, 0, 5120, np.ones(5120))
    clock = Clock()
    events, _ = capture_batch(Recognizer(clock), [case], clock=clock.now, wait=clock.wait)
    event, = events
    assert event["prior_status"] == "final_before_caller_onset"
    assert event["prior_final_received_ns"] == 1_350_000_000
    assert event["revisions"][1]["received_ns"] == 2_150_000_000
    decisions = replay(PauseTrace(**event["trace"]), [Revision(**row) for row in event["revisions"]])
    assert decisions.at_300_ms.text == "hello five\nhello five"
    assert decisions.benchmark_grid[1].text == "hello five\nhello five there"
    assert event["digit_cleaner_changed_text_at_300"]
    assert event["raw_pair_at_300"] == ["hello 5", "hello 5"]
    assert clean_text(clean_text("Yes 5, 11.")) == "yes five one one"
    repeat = repeat_diagnostics(events, events)
    assert repeat["state_agreement"]["all"]["same_raw_provider_text_in_both"] == 1
    assert repeat["arrival_quantiles"]["paired_b_minus_a_visible_revision_ms"]["median_abs"] == 0


def test_late_revision_cannot_extend_observed_audio():
    row = {"call": "generated", "caller_sample": 1, "label": "shift"}
    case = Episode(row, 0, 0, np.ones(5120), 1, None, None, None, None)
    clock = Clock()
    events, _ = capture_batch(Recognizer(clock), [case], clock=clock.now, wait=clock.wait)
    event, = events
    assert event["trace"]["captured_through_ns"] == 1_320_000_000
    assert event["revisions"] == []
    assert event["after_observed_audio_revisions"][0]["received_ns"] == 1_330_000_000


def test_missing_prior_abstains_instead_of_appearing_as_first_turn(tmp_path):
    (tmp_path / ".campaign-machine").touch()
    source, destination = tmp_path / "capture.json", tmp_path / "packed.json"
    source.write_text(json.dumps({"provider_settings_sha256": "a" * 64, "pauses": [{
        "id": "generated", "label": "hold", "scoring_exclusion": "missing_previous_stt",
    }]}))
    pack_trace_file(tmp_path, source, destination)
    packed = json.loads(destination.read_text())
    assert packed["coverage"] == {"hold/300_ms/missing_previous_stt": 1}
    assert packed["pauses"][0]["at_300_ms"] is None
