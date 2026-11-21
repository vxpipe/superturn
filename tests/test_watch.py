from superturn.watch import IDLE_SECONDS, STALL_SECONDS, events


def step(fraction, remaining=None, updated=1000.0, state="running"):
    return {"step": "download", "state": state, "unit": "B", "done": fraction * 100, "total": 100,
            "fraction": fraction, "elapsed_s": 60, "rate_per_s": None, "remaining_s": remaining,
            "finishes_at": None if remaining is None else updated + remaining,
            "started_at": 900, "updated_at": updated, "skipped": 0}


def status(now, jobs, progress=()):
    return {"time": now, "session": "vxpipe", "windows": [], "jobs": jobs, "progress": list(progress)}


def job(state, exit=None, started=900.0, ended=None):
    return {"state": state, "exit": exit, "started_at": started, "ended_at": ended}


def test_each_event_is_sent_once():
    msgs, state = events({"last_summary": 1000}, status(1000, {"download": job("running")}, [step(0.01)]))
    assert msgs == ["job download 0% started"]

    msgs, state = events(state, status(1030, {"download": job("running")}, [step(0.2, 600, 1030)]))
    assert msgs == ["download 0% started", "estimate: download 20% (10m00s left, about 00:27 UTC)"]

    msgs, state = events(state, status(1060, {"download": job("running")}, [step(0.3, 500, 1060)]))
    assert msgs == ["download 25% (8m20s left)"]

    msgs, state = events(state, status(1700, {"download": job("done", 0, ended=1690)},
                                       [{**step(1.0, 0, 1690, state="done"), "elapsed_s": 790}]))
    assert msgs == ["job download 100% finished in 13m10s", "download 50%", "download 75%",
                    "download 100% finished in 13m10s"]


def test_summary_every_hour_while_steps_run():
    state = {"jobs": {"train": "running"}, "last_summary": 1000}
    running = {"train": job("running")}
    msgs, state = events(state, status(1000 + 3599, running, [step(0.4, 900, 4599)]), summary_every=3600)
    assert not any(m.startswith("progress:") for m in msgs)
    msgs, state = events(state, status(1000 + 3600, running, [step(0.5, 800, 4600)]), summary_every=3600)
    assert any(m.startswith("progress: download 50%") for m in msgs)


def test_failure_stall_and_idle_are_reported():
    msgs, state = events({}, status(1000, {"label": job("running")}, [step(0.5, 60, 1000)]))
    later = 1000 + STALL_SECONDS + 1
    msgs, state = events(state, status(later, {"label": job("running")}, [step(0.5, 60, 1000)]))
    assert any("has not updated" in m for m in msgs)

    msgs, state = events(state, status(later + 10, {"label": job("failed", 3, ended=later)}))
    assert msgs == [f"job label FAILED (exit 3) in {(later - 900) // 60:.0f}m{(later - 900) % 60:02.0f}s; see logs/label.log"]

    msgs, state = events(state, status(later + 20 + IDLE_SECONDS, {"label": job("failed", 3, ended=later)}))
    assert any("still rented" in m for m in msgs)
    msgs, state = events(state, status(later + 30 + IDLE_SECONDS, {"label": job("failed", 3, ended=later)}))
    assert msgs == []


def test_job_without_window_or_exit_is_lost():
    msgs, _ = events({"jobs": {"train": "running"}}, status(1000, {"train": job("lost")}))
    assert "stopped without an exit status" in msgs[0]
