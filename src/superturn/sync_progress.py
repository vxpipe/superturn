"""Run an R2 transfer and publish rclone byte counts to the campaign watcher."""

import json
import subprocess
import sys

from .progress import Progress


def copy(step, source, destination):
    command = ["rclone", "copy", "--transfers", "16", "--check-first", "--use-json-log",
               "--stats", "10s", "--stats-log-level", "NOTICE", source, destination]
    meter = None
    last_bytes = 0
    with subprocess.Popen(command, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stderr:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                continue
            stats = record.get("stats")
            if not stats:
                print(record.get("msg", ""), flush=True)
                continue
            total = stats.get("totalBytes", 0)
            received = stats.get("bytes", 0)
            if total > 0 and meter is None:
                meter = Progress(step, total, "B")
            if meter is not None:
                if received > last_bytes:
                    meter.advance(received - last_bytes)
                    last_bytes = received
                meter.report()
        status = process.wait()
    if meter is not None:
        if status == 0:
            meter.advance(max(0, meter.total - last_bytes))
        meter.finish("done" if status == 0 else "failed")
    return status


if __name__ == "__main__":
    raise SystemExit(copy(*sys.argv[1:]))
