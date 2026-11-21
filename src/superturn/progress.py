"""Progress and time-to-finish estimates for long steps.

Every long step (download, labelling, sharding, training, benchmarks) counts
its work with a `Progress`: the total is known up front (bytes planned, files
to label, training steps), the pace is measured from the work done so far,
and the remaining time is the remaining work at that pace. Work skipped
because an earlier run already did it (a resumed download) counts as done
but not towards the pace, so a resume does not report an impossibly fast rate.

Each step prints a line every `every` seconds (on a timer) and keeps
`$SUPERTURN_DATA/progress/<step>.json` current, so `superturn progress`
(`bin/vast-progress` from the local computer) can show every step's state.
The final record keeps the measured duration, for the campaign labnotes.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .sources import data_root

# No estimate until this much of the measured work is done (a share of the
# total) and this many seconds have passed; early pace is noisy.
MIN_SHARE = 0.005
MIN_SECONDS = 20.0


def progress_dir() -> Path:
    return data_root() / "progress"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def fmt_amount(value: float, unit: str) -> str:
    if unit == "B":
        for scale, name in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
            if value >= scale:
                return f"{value / scale:.2f} {name}"
        return f"{value:.0f} B"
    return f"{value:,.0f} {unit}"


class Progress:
    """Thread-safe work counter with a measured pace and remaining time."""

    def __init__(
        self,
        step: str,
        total: float,
        unit: str = "items",
        every: float = 30.0,
        path: Path | None = None,
        clock=time.monotonic,
        wall=time.time,
        out=print,
    ):
        self.step, self.total, self.unit, self.every = step, float(total), unit, every
        self.path = path if path is not None else progress_dir() / f"{step}.json"
        self.clock, self.wall, self.out = clock, wall, out
        self.done = 0.0
        self.skipped = 0.0
        self.started = clock()
        self.started_wall = wall()
        self.state = "running"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._write()
        if every > 0:  # report on a timer, so slow work without byte counts still reports
            threading.Thread(target=self._tick, daemon=True).start()

    def _tick(self) -> None:
        while not self._stop.wait(self.every):
            self.report()

    # ---------------------------------------------------------- counting

    def advance(self, amount: float) -> None:
        with self._lock:
            self.done += amount

    def skip(self, amount: float) -> None:
        """Count work an earlier run already did: done, but not paced."""
        with self._lock:
            self.done += amount
            self.skipped += amount

    # ---------------------------------------------------------- estimates

    def elapsed(self) -> float:
        return self.clock() - self.started

    def rate(self) -> float | None:
        """Measured work per second, or None while there is too little to go on."""
        measured = self.done - self.skipped
        to_measure = self.total - self.skipped
        elapsed = self.elapsed()
        if measured <= 0 or elapsed < MIN_SECONDS or to_measure <= 0:
            return None
        if measured / to_measure < MIN_SHARE:
            return None
        return measured / elapsed

    def remaining(self) -> float | None:
        if self.done >= self.total:
            return 0.0
        rate = self.rate()
        return None if rate is None else (self.total - self.done) / rate

    def snapshot(self) -> dict:
        with self._lock:
            rate, remaining = self.rate(), self.remaining()
            return {
                "step": self.step,
                "state": self.state,
                "unit": self.unit,
                "done": self.done,
                "skipped": self.skipped,
                "total": self.total,
                "fraction": min(1.0, self.done / self.total) if self.total else 1.0,
                "elapsed_s": round(self.elapsed(), 1),
                "rate_per_s": rate,
                "remaining_s": None if remaining is None else round(remaining, 1),
                "finishes_at": None if remaining is None else round(self.wall() + remaining),
                "started_at": round(self.started_wall),
                "updated_at": round(self.wall()),
            }

    def line(self) -> str:
        s = self.snapshot()
        rate = "measuring pace" if s["rate_per_s"] is None else f"{fmt_amount(s['rate_per_s'], self.unit)}/s"
        eta = s["remaining_s"]
        finish = "" if eta is None else time.strftime(" (about %H:%M UTC)", time.gmtime(s["finishes_at"]))
        return (
            f"[{self.step}] {100 * s['fraction']:5.1f}%  "
            f"{fmt_amount(s['done'], self.unit)} of {fmt_amount(s['total'], self.unit)}  "
            f"{rate}  elapsed {fmt_duration(s['elapsed_s'])}  "
            f"left {fmt_duration(eta)}{finish}"
        )

    # ---------------------------------------------------------- reporting

    def report(self) -> None:
        self.out(self.line(), flush=True)
        self._write()

    def finish(self, state: str = "done") -> None:
        self._stop.set()
        self.state = state
        self.out(f"[{self.step}] {state} in {fmt_duration(self.elapsed())}", flush=True)
        self._write()

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=1))
            os.replace(tmp, self.path)
        except OSError:
            pass  # progress is advisory; never fail the step over it


def read_all(directory: Path | None = None) -> list[dict]:
    directory = directory or progress_dir()
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return records


def table(records: list[dict], now: float | None = None) -> str:
    now = time.time() if now is None else now
    rows = []
    for r in records:
        if r["state"] == "running":
            when = "?" if r["finishes_at"] is None else time.strftime("%H:%M UTC", time.gmtime(r["finishes_at"]))
            left = f"{fmt_duration(r['remaining_s'])} left, about {when}"
            stale = now - r["updated_at"]
            if stale > 600:
                left += f" (no update for {fmt_duration(stale)}: stopped?)"
        else:
            left = f"{r['state']} after {fmt_duration(r['elapsed_s'])}"
        rows.append(
            f"{r['step']:<16} {100 * r['fraction']:5.1f}%  "
            f"{fmt_amount(r['done'], r['unit'])} of {fmt_amount(r['total'], r['unit'])}  {left}"
        )
    return "\n".join(rows) if rows else "no progress recorded yet"
