"""Command line: `superturn plan|download|progress|status|watch-events|check-licences|inspect-archive|refresh-url-lists`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .progress import Progress, read_all as read_progress, table as progress_table
from .acquire import _hf_auth, check_licences, download, inspect_archive, plan_source
from .sources import budget_bytes, load_sources, select


def _gb(n: int) -> str:
    return f"{n / 1e9:8.1f} GB"


def _plans(args):
    sources = load_sources()
    return [plan_source(s, sample=args.sample) for s in select(sources, args.only)]


def _over_budget(plans, args) -> bool:
    total, budget = sum(p.size for p in plans), budget_bytes()
    if args.only or args.sample or total <= budget:
        return False
    print(
        f"plan is {total / 1e9:.1f} GB, over the {budget / 1e9:.0f} GB budget in "
        "configs/sources.toml; reduce the selection",
        file=sys.stderr,
    )
    return not args.allow_over_budget


def cmd_plan(args) -> int:
    plans = _plans(args)
    print(f"{'source':18s} {'role':20s} {'files':>7s} {'size':>11s}  licence")
    for plan in plans:
        source = plan.source
        print(
            f"{source.name:18s} {source.role:20s} {len(plan.files):7d} "
            f"{_gb(plan.size)}  {source.licence}"
        )
        if plan.note:
            print(f"{'':18s} note: {plan.note}")
    total = sum(p.size for p in plans)
    print(f"{'total':18s} {'':20s} {'':7s} {_gb(total)}  (budget {budget_bytes() / 1e9:.0f} GB)")
    return 2 if _over_budget(plans, args) else 0


def cmd_download(args) -> int:
    plans = _plans(args)
    if _over_budget(plans, args):
        return 2
    failed = []
    meter = Progress("download-sample" if args.sample else "download", sum(p.size for p in plans), "B")
    for plan in plans:
        name = plan.source.name
        print(f"== {name}: {len(plan.files)} files, up to {_gb(plan.size).strip()}", flush=True)
        try:
            download(plan, workers=args.workers, meter=meter)
        except (Exception, SystemExit) as error:  # keep going; report at the end
            print(f"!! {name} failed: {error}", file=sys.stderr, flush=True)
            failed.append(name)
        meter.report()
    meter.finish("failed" if failed else "done")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_status(args) -> int:
    import json
    import os

    from .sources import data_root
    from .watch import collect_status

    status = collect_status(data_root(), os.environ.get("TMUX_SESSION", "vxpipe"))
    print(json.dumps(status))
    return 0


def cmd_watch_events(args) -> int:
    """Read a status (stdin) and the saved state; print messages, save state."""
    import json

    from .watch import events

    path = Path(args.state)
    prev = json.loads(path.read_text()) if path.exists() else {}
    msgs, state = events(prev, json.load(sys.stdin), summary_every=args.summary_minutes * 60)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))
    for msg in msgs:
        print(msg)
    return 0


def cmd_progress(args) -> int:
    print(progress_table(read_progress()))
    return 0


def cmd_check_licences(args) -> int:
    problems = check_licences(load_sources())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def cmd_inspect_archive(args) -> int:
    """Show the member order at the start of an archive, to check that a
    capped stream reaches the parts we need (for example MUSAN noise)."""
    url = args.url
    headers = {}
    if not url.startswith("http"):  # REPO:PATH on Hugging Face
        from huggingface_hub import hf_hub_url

        repo, path = url.split(":", 1)
        url, headers = hf_hub_url(repo, path, repo_type="dataset"), _hf_auth()
    directories: dict[str, list[int]] = {}
    for name, size, read in inspect_archive(url, int(args.mb * 1e6), headers):
        key = "/".join(name.split("/")[: args.depth])
        if key not in directories:
            directories[key] = [0, 0, read]
            print(f"{read / 1e6:9.1f} MB  first member in {key}")
        directories[key][0] += 1
        directories[key][1] += size
    for key, (count, size, first) in directories.items():
        print(f"{key}: {count} members, {size / 1e6:.1f} MB, starts at {first / 1e6:.1f} MB")
    return 0


def cmd_refresh_url_lists(args) -> int:
    from .url_lists import refresh

    refresh()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="superturn")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, text in (
        ("plan", cmd_plan, "price each source from provider listings; downloads nothing"),
        ("download", cmd_download, "download sources into $SUPERTURN_DATA/raw"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--only", nargs="+", help="source names (default: all)")
        p.add_argument("--sample", action="store_true", help="smoke-test subset only")
        p.add_argument("--allow-over-budget", action="store_true", help="proceed despite the budget")
        p.set_defaults(func=func)
        if name == "download":
            p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("status", help="campaign status as JSON: jobs, tmux windows, progress (on the machine)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watch-events", help="turn a status on stdin into notifications (bin/vast-watch)")
    p.add_argument("--state", required=True, help="file holding what was already notified")
    p.add_argument("--summary-minutes", type=float, default=60)
    p.set_defaults(func=cmd_watch_events)

    p = sub.add_parser("progress", help="show each long step's progress and estimated time left")
    p.set_defaults(func=cmd_progress)

    p = sub.add_parser("check-licences", help="fail on raw data without a configured licence")
    p.set_defaults(func=cmd_check_licences)

    p = sub.add_parser(
        "inspect-archive", help="list member order at the start of an archive (reads --mb, saves nothing)"
    )
    p.add_argument("url", help="HTTPS URL or REPO:PATH of a Hugging Face dataset file")
    p.add_argument("--mb", type=float, default=5.0, help="megabytes to read (default 5)")
    p.add_argument("--depth", type=int, default=2, help="path components to group by")
    p.set_defaults(func=cmd_inspect_archive)

    p = sub.add_parser("refresh-url-lists", help="rebuild the AMI headset URL list")
    p.set_defaults(func=cmd_refresh_url_lists)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
