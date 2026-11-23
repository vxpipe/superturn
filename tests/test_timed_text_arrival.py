from dataclasses import asdict, replace
import json

from superturn.timed_text_arrival import MS, PauseTrace, Revision, pack_trace_file, replay


TRACE = PauseTrace("caller-2", "", 10 * MS, 8000, 8000, None, 2_000 * MS)
STOP = 1010


def event(sequence, offset_ms, text, turn="caller-2"):
    return Revision(sequence, (STOP + offset_ms) * MS, turn, text)


def test_previous_turn_and_causal_replacements():
    trace = replace(TRACE, previous_turn="Which time?")
    result = replay(trace, [event(3, 390, "Afternoon."), event(1, 100, "At three!"), event(2, 110, "wrong", "agent")])
    assert result.at_300_ms.text == "which time\nat three"
    assert result.at_300_ms.revision_sequence == 1
    assert [tick.since_stop_ns // MS for tick in result.ticks] == [300, 400, 500, 600, 700]
    assert result.ticks[1].text == "which time\nafternoon"
    assert [tick.since_stop_ns // MS for tick in result.benchmark_grid] == [300, 400, 500, 600, 700, 800]
    assert result.ended_by == "timeout"


def test_late_initial_arrival_and_blanks_use_actual_cadence():
    result = replay(TRACE, [event(1, 354, "wait"), event(3, 450, ""), event(4, 600, "let me think")])
    assert result.at_300_ms.text is None
    assert [(tick.since_stop_ns // MS, tick.text) for tick in result.ticks] == [
        (354, "---\nwait"), (454, None), (554, None), (654, "---\nlet me think"), (754, "---\nlet me think"),
    ]
    assert result.benchmark_grid[0].text is None
    assert result.benchmark_grid[1].text == "---\nwait"
    assert result.benchmark_grid[2].text is None


def test_same_receipt_time_uses_sequence_and_never_revives_retracted_text():
    result = replay(TRACE, [event(5, 299, ""), event(1, 280, "old"), event(4, 299, "older")])
    assert result.at_300_ms.revision_sequence == 5
    assert result.at_300_ms.text is None
    assert result.ticks == ()


def test_resume_at_tick_precedes_inference():
    trace = replace(TRACE, resume_sample=8000 + 3200)
    result = replay(trace, [event(1, 280, "hello"), event(2, 400, "new")])
    assert [tick.since_stop_ns // MS for tick in result.ticks] == [300]
    assert [tick.since_stop_ns // MS for tick in result.benchmark_grid] == [300]
    assert result.ended_by == "resume"


def test_unobserved_deadline_is_distinct_from_available_deadline_without_text():
    trace = replace(TRACE, captured_through_ns=(STOP + 299) * MS)
    assert replay(trace, []).at_300_ms is None
    assert replay(trace, []).ended_by == "capture_ended"
    assert replay(TRACE, []).at_300_ms.text is None


def test_trace_packer_keeps_only_aggregate_coverage_on_stdout(tmp_path, capsys):
    (tmp_path / ".campaign-machine").touch()
    source = tmp_path / "synthetic.json"
    packed = tmp_path / "packed.json"
    source.write_text(json.dumps({
        "provider_settings_sha256": "a" * 64,
        "pauses": [{"id": "fixture", "label": "hold", "trace": asdict(TRACE),
                    "revisions": [asdict(event(1, 70, "fixture text"))]}],
    }))
    pack_trace_file(tmp_path, source, packed)
    report = capsys.readouterr().out
    assert "fixture text" not in report
    assert json.loads(report)["coverage"]["hold/300_ms/text"] == 1
    assert json.loads(packed.read_text())["pauses"][0]["at_300_ms"]["text"] == "---\nfixture text"
