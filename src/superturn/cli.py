"""Command line: `superturn plan|download|check-licences`."""

from __future__ import annotations

import argparse
import sys

from .acquire import check_licences, download, plan_source
from .sources import load_sources, select


def _gb(n: int) -> str:
    return f"{n / 1e9:8.1f} GB"


def cmd_plan(args) -> int:
    sources = load_sources()
    total = 0
    print(f"{'source':18s} {'role':20s} {'files':>7s} {'size':>11s}  licence")
    for source in select(sources, args.only):
        plan = plan_source(source, sample=args.sample)
        total += plan.size
        print(
            f"{source.name:18s} {source.role:20s} {len(plan.files):7d} "
            f"{_gb(plan.size)}  {source.licence}"
        )
        if plan.note:
            print(f"{'':18s} note: {plan.note}")
    print(f"{'total':18s} {'':20s} {'':7s} {_gb(total)}")
    return 0


def cmd_download(args) -> int:
    sources = load_sources()
    failed = []
    for source in select(sources, args.only):
        plan = plan_source(source, sample=args.sample)
        print(f"== {source.name}: {len(plan.files)} files, {_gb(plan.size).strip()}", flush=True)
        try:
            download(plan, workers=args.workers)
        except (Exception, SystemExit) as error:  # keep going; report at the end
            print(f"!! {source.name} failed: {error}", file=sys.stderr, flush=True)
            failed.append(source.name)
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_check_licences(args) -> int:
    problems = check_licences(load_sources())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


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
        p.set_defaults(func=func)
        if name == "download":
            p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("check-licences", help="fail on raw data without a configured licence")
    p.set_defaults(func=cmd_check_licences)

    p = sub.add_parser("refresh-url-lists", help="rebuild the AMI and ICSI headset URL lists")
    p.set_defaults(func=cmd_refresh_url_lists)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
