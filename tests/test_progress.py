import json

from superturn.progress import Progress, fmt_duration, read_all, table


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def meter(tmp_path, total=1000, clock=None):
    clock = clock or Clock()
    lines = []
    m = Progress(
        "download", total, "B", every=0, path=tmp_path / "download.json",
        clock=clock, wall=lambda: 1_700_000_000 + clock.t, out=lambda s, flush=True: lines.append(s),
    )
    return m, clock, lines


def test_no_estimate_until_enough_work_is_measured(tmp_path):
    m, clock, _ = meter(tmp_path)
    m.advance(100)
    clock.t += 5  # too early
    assert m.remaining() is None
    clock.t += 45
    assert m.rate() == 2.0  # 100 B in 50 s
    assert m.remaining() == 450.0  # 900 B left at 2 B/s


def test_resumed_work_is_done_but_not_counted_in_the_pace(tmp_path):
    m, clock, _ = meter(tmp_path)
    m.skip(800)  # an earlier run fetched these
    m.advance(20)
    clock.t += 60
    assert m.rate() == 20 / 60
    assert abs(m.remaining() - 180 / (20 / 60)) < 1e-6


def test_progress_file_and_table(tmp_path):
    m, clock, lines = meter(tmp_path)
    m.advance(500)
    clock.t += 100
    m.report()
    record = json.loads((tmp_path / "download.json").read_text())
    assert record["fraction"] == 0.5 and record["remaining_s"] == 100.0
    assert "50.0%" in lines[-1] and "left 1m40s" in lines[-1]
    m.advance(500)
    m.finish()
    rows = table(read_all(tmp_path), now=1_700_000_000 + clock.t)
    assert "100.0%" in rows and "done after 1m40s" in rows


def test_durations_read_naturally():
    assert fmt_duration(42) == "42s"
    assert fmt_duration(125) == "2m05s"
    assert fmt_duration(3 * 3600 + 7 * 60) == "3h07m"
    assert fmt_duration(None) == "?"


def test_download_meter_counts_live_bytes_credits_the_rest_and_skips_resumed(tmp_path):
    import io

    from superturn.acquire import PlannedFile, _Counted, _local, _once

    m, clock, _ = meter(tmp_path, total=300)

    @_once
    def fetch(plan, f, target):
        # Stands in for a fetcher reading 60 bytes through _request.
        response = _Counted(io.BytesIO(b"x" * 60), _local.counter)
        response.read()
        return {"ok": True}

    first = PlannedFile("a.bin", 100)
    fetch(None, first, tmp_path, m)
    assert m.done == 100 and m.skipped == 0  # 60 live, 40 credited at the end

    fetch(None, first, tmp_path, m)  # already done: a resumed run
    assert m.done == 200 and m.skipped == 100
    assert _local.counter is None
