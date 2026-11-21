"""Campaign status and the notifications sent while watching it.

`collect_status` runs on the campaign machine (`superturn status --json`): the
progress records of every long step, the jobs started by `bin/job-run`
(running, finished or failed) and the tmux windows of the job session.

`events` runs locally (`superturn watch-events`, called by `bin/vast-watch`):
it compares a new status with the state saved from the previous check and
returns the messages worth sending to the user's phone, so each event is
sent once: a job started, finished or failed; a step's first time estimate
once its pace is measured; a step that stopped updating; a regular summary;
and the machine sitting idle (still costing money).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .progress import fmt_duration, read_all

STALL_SECONDS = 15 * 60
IDLE_SECONDS = 20 * 60
LONG_STEP_SECONDS = 180
PERCENTAGES = (0, 25, 50, 75, 100)


def _tmux_windows(session: str) -> list[str]:
    try:
        out = subprocess.run(
            ["tmux", "list-windows", "-t", session, "-F", "#W"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.split() if out.returncode == 0 else []


def collect_status(data: Path, session: str, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    windows = _tmux_windows(session)
    jobs = {}
    jobs_dir = data / "jobs"
    for script in sorted(jobs_dir.glob("*.sh")) if jobs_dir.exists() else []:
        name = script.stem
        exit_file, started_file = jobs_dir / f"{name}.exit", jobs_dir / f"{name}.started"
        started = float(started_file.read_text().strip()) if started_file.exists() else None
        if exit_file.exists():
            code = int(exit_file.read_text().strip() or 1)
            state = "done" if code == 0 else "failed"
            ended = exit_file.stat().st_mtime
        else:
            code, ended = None, None
            state = "running" if name in windows else "lost"  # no window, no exit: killed
        jobs[name] = {"state": state, "exit": code, "started_at": started, "ended_at": ended}
    return {"time": now, "session": session, "windows": windows,
            "jobs": jobs, "progress": read_all(data / "progress")}


def _when(epoch: float | None) -> str:
    return "?" if epoch is None else time.strftime("%H:%M UTC", time.gmtime(epoch))


def _step_line(r: dict) -> str:
    if r["remaining_s"] is None:
        left = "measuring pace"
    else:
        left = f"{fmt_duration(r['remaining_s'])} left, about {_when(r['finishes_at'])}"
    return f"{r['step']} {100 * r['fraction']:.0f}% ({left})"


def events(prev: dict, status: dict, summary_every: float = 3600) -> tuple[list[str], dict]:
    """Messages to send for `status`, given the state saved last time."""
    now = status["time"]
    seen_jobs = prev.get("jobs", {})
    estimated = set(prev.get("estimated", []))
    stalled = set(prev.get("stalled", []))
    milestones = dict(prev.get("milestones", {}))
    msgs: list[str] = []

    for name, job in status["jobs"].items():
        before = seen_jobs.get(name)
        if before == job["state"]:
            continue
        took = ""
        if job["started_at"] and job["ended_at"]:
            took = f" in {fmt_duration(job['ended_at'] - job['started_at'])}"
        if job["state"] == "running" and before is None:
            msgs.append(f"job {name} 0% started")
        elif job["state"] == "done":
            msgs.append(f"job {name} 100% finished{took}")
        elif job["state"] == "failed":
            msgs.append(f"job {name} FAILED (exit {job['exit']}){took}; see logs/{name}.log")
        elif job["state"] == "lost":
            msgs.append(f"job {name} stopped without an exit status (tmux window gone); see logs/{name}.log")

    for record in status["progress"]:
        key = f"{record['step']}@{record['started_at']}"
        elapsed = record["elapsed_s"]
        remaining = record["remaining_s"]
        is_long = elapsed >= LONG_STEP_SECONDS or (
            record["state"] == "running" and remaining is not None
            and elapsed + remaining >= LONG_STEP_SECONDS
        )
        if not is_long and key not in milestones:
            continue
        previous = milestones.get(key, -1)
        reached = 100 if record["state"] == "done" else min(99, int(100 * record["fraction"]))
        for percentage in PERCENTAGES:
            if previous < percentage <= reached:
                suffix = " started" if percentage == 0 else ""
                if percentage == 100:
                    suffix = f" finished in {fmt_duration(elapsed)}"
                elif percentage and remaining is not None and record["state"] == "running":
                    suffix = f" ({fmt_duration(remaining)} left)"
                msgs.append(f"{record['step']} {percentage}%{suffix}")
                milestones[key] = percentage

    running = [r for r in status["progress"] if r["state"] == "running"]
    for r in running:
        key = f"{r['step']}@{r['started_at']}"
        if r["remaining_s"] is not None and key not in estimated:
            estimated.add(key)
            msgs.append(f"estimate: {_step_line(r)}")
        if now - r["updated_at"] > STALL_SECONDS and key not in stalled:
            stalled.add(key)
            msgs.append(f"{r['step']} has not updated for {fmt_duration(now - r['updated_at'])}: stuck or killed?")

    last_summary = prev.get("last_summary", 0)
    if running and now - last_summary >= summary_every:
        msgs.append("progress: " + "; ".join(_step_line(r) for r in running))
        last_summary = now

    active = any(j["state"] == "running" for j in status["jobs"].values())
    idle_since = prev.get("idle_since")
    idle_sent = prev.get("idle_sent", False)
    if active:
        idle_since, idle_sent = None, False
    else:
        idle_since = idle_since or now
        if status["jobs"] and not idle_sent and now - idle_since >= IDLE_SECONDS:
            msgs.append(f"no job running for {fmt_duration(now - idle_since)}; the machine is still rented")
            idle_sent = True

    state = {
        "jobs": {n: j["state"] for n, j in status["jobs"].items()},
        "estimated": sorted(estimated),
        "stalled": sorted(stalled),
        "milestones": milestones,
        "last_summary": last_summary,
        "idle_since": idle_since,
        "idle_sent": idle_sent,
    }
    return msgs, state
