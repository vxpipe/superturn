"""Plan and perform dataset downloads.

Planning reads only provider file listings and release metadata, so it is
safe to run anywhere. Downloading is meant for the campaign machine.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .sources import Source, data_root, matches, raw_dir, read_url_list

USER_AGENT = "superturn-downloader"


@dataclass
class PlannedFile:
    path: str  # path relative to the source's raw directory
    size: int
    url: str | None = None  # for urllist/http sources


@dataclass
class Plan:
    source: Source
    files: list[PlannedFile] = field(default_factory=list)
    revision: str | None = None
    note: str = ""

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


# ---------------------------------------------------------------- planning


def plan_source(source: Source, sample: bool = False) -> Plan:
    if source.kind == "hf":
        return _plan_hf(source, sample)
    if source.kind == "urllist":
        return _plan_urllist(source, sample)
    if source.kind == "http":
        return _plan_http(source, sample)
    if source.kind == "mdc":
        return _plan_mdc(source, sample)
    raise ValueError(source.kind)


def _plan_hf(source: Source, sample: bool) -> Plan:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = source.get("repo")
    revision = source.get("revision") or api.dataset_info(repo).sha
    include = source.get("sample_include") if sample else source.get("include")
    exclude = source.get("exclude", [])
    files = [
        PlannedFile(path=entry.path, size=entry.size or 0)
        for entry in api.list_repo_tree(
            repo, repo_type="dataset", recursive=True, revision=revision
        )
        if getattr(entry, "size", None) is not None
        and matches(entry.path, include, exclude)
    ]
    return Plan(source, sorted(files, key=lambda f: f.path), revision)


def _plan_urllist(source: Source, sample: bool) -> Plan:
    prefix = source.get("strip_prefix")
    wanted = source.get("sample_match") if sample else None
    files = [
        PlannedFile(path=url.removeprefix(prefix), size=size, url=url)
        for size, url in read_url_list(source)
        if wanted is None or wanted in url
    ]
    return Plan(source, files)


def _head_size(url: str) -> int:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def _plan_http(source: Source, sample: bool) -> Plan:
    urls = source.get("sample_urls" if sample else "urls", [])
    return Plan(
        source,
        [PlannedFile(path=u.rsplit("/", 1)[1], size=_head_size(u), url=u) for u in urls],
    )


def _plan_mdc(source: Source, sample: bool) -> Plan:
    request = urllib.request.Request(
        source.get("release_metadata"), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        locales = json.load(response)["locales"]
    wanted = source.get("sample_locales" if sample else "locales")
    ids = source.get("dataset_ids", {})
    files, missing = [], []
    for locale in wanted:
        if locale not in locales:
            raise SystemExit(f"common_voice: no release for locale {locale!r}")
        files.append(PlannedFile(path=locale, size=int(locales[locale]["size"])))
        if locale not in ids:
            missing.append(locale)
    note = f"dataset IDs not yet recorded for: {', '.join(missing)}" if missing else ""
    return Plan(source, files, revision=source.get("release"), note=note)


# ------------------------------------------------------------- downloading


def download(plan: Plan, workers: int = 8) -> None:
    target = raw_dir(plan.source)
    target.mkdir(parents=True, exist_ok=True)
    kind = plan.source.kind
    if kind == "hf":
        _download_hf(plan, target, workers)
    elif kind in ("urllist", "http"):
        _download_urls(plan, target, workers)
    elif kind == "mdc":
        _download_mdc(plan, target)
    _write_manifest(plan, target)


def _download_hf(plan: Plan, target: Path, workers: int) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=plan.source.get("repo"),
        repo_type="dataset",
        revision=plan.revision,
        allow_patterns=[f.path for f in plan.files],
        local_dir=target,
        max_workers=workers,
        token=os.environ.get("HF_TOKEN"),
    )
    for f in plan.files:
        actual = (target / f.path).stat().st_size
        if actual != f.size:
            raise RuntimeError(f"{f.path}: expected {f.size} bytes, got {actual}")


def _fetch(url: str, dest: Path, size: int, attempts: int = 5) -> None:
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with open(part, mode) as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
            if part.stat().st_size != size:
                raise OSError(f"size {part.stat().st_size} != {size}")
            part.rename(dest)
            return
        except OSError as error:
            if attempt == attempts:
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2**attempt)


def _download_urls(plan: Plan, target: Path, workers: int) -> None:
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(_fetch, f.url, target / f.path, f.size) for f in plan.files
        ]
        for future in cf.as_completed(futures):
            future.result()


def _download_mdc(plan: Plan, target: Path) -> None:
    from datacollective import download_dataset

    ids = plan.source.get("dataset_ids", {})
    missing = [f.path for f in plan.files if f.path not in ids]
    if missing:
        raise SystemExit(
            "common_voice: accept the terms for these locales in the Mozilla Data "
            "Collective web UI and record their dataset IDs in configs/sources.toml: "
            + ", ".join(missing)
        )
    for f in plan.files:
        done = target / f.path / ".complete"
        if done.exists():
            continue
        (target / f.path).mkdir(parents=True, exist_ok=True)
        download_dataset(ids[f.path], download_directory=str(target / f.path))
        done.touch()


def _write_manifest(plan: Plan, target: Path) -> None:
    manifest = {
        "source": plan.source.name,
        "licence": plan.source.licence,
        "attribution": plan.source.attribution,
        "role": plan.source.role,
        "revision": plan.revision,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [{"path": f.path, "size": f.size} for f in plan.files],
    }
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------- licence check


def check_licences(sources: dict[str, Source]) -> list[str]:
    """Return problems: raw data from unknown sources or without a manifest."""
    problems = []
    raw = data_root() / "raw"
    if not raw.exists():
        return problems
    for entry in sorted(raw.iterdir()):
        if entry.name not in sources:
            problems.append(f"{entry.name}: not a configured source")
        elif not (entry / "MANIFEST.json").exists():
            problems.append(f"{entry.name}: no MANIFEST.json (incomplete download?)")
    return problems
