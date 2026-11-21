"""Plan and perform partial dataset downloads.

Every source downloads only the bytes it uses (see configs/sources.toml).
Planning reads only listings and file metadata (parquet footers, HEAD
requests), so it is safe to run anywhere. Downloading is for the campaign
machine.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import struct
import tarfile
import threading
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .sources import Source, data_root, expand, matches, raw_dir, read_url_list

USER_AGENT = "superturn-downloader"
WAV_BYTES_PER_SECOND = 16_000 * 2  # AMI headsets: 16 kHz, 16-bit, mono


@dataclass
class PlannedFile:
    path: str  # path relative to the source's raw directory
    size: int  # bytes this plan downloads (an upper bound for streams)
    url: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Plan:
    source: Source
    files: list[PlannedFile] = field(default_factory=list)
    revision: str | None = None
    note: str = ""

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def _request(url: str, method: str = "GET", headers: dict | None = None, data: bytes | None = None):
    all_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    response = urllib.request.urlopen(
        urllib.request.Request(url, data=data, method=method, headers=all_headers), timeout=120
    )
    counter = getattr(_local, "counter", None)
    return _Counted(response, counter) if counter else response


# Bytes read by the file each download thread is fetching, for progress.
_local = threading.local()


class _FileCounter:
    def __init__(self, meter):
        self.meter = meter
        self.count = 0

    def add(self, n: int) -> None:
        self.count += n
        self.meter.advance(n)


class _Counted:
    """An HTTP response whose reads are counted towards the download's progress."""

    def __init__(self, response, counter: _FileCounter):
        self._response, self._counter = response, counter

    def read(self, n: int = -1) -> bytes:
        chunk = self._response.read(n)
        self._counter.add(len(chunk))
        return chunk

    def __getattr__(self, name):
        return getattr(self._response, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._response.__exit__(*exc)


def _head_size(url: str) -> int:
    with _request(url, "HEAD") as response:
        return int(response.headers["Content-Length"])


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def _hf_auth() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


# ---------------------------------------------------------------- planning


def plan_source(source: Source, sample: bool = False) -> Plan:
    return {
        "wav_slices": _plan_wav_slices,
        "hf_rowgroups": _plan_hf_rowgroups,
        "hf_tar_stream": _plan_hf_tar_stream,
        "http_tar_stream": _plan_http_tar_stream,
        "http": _plan_http,
    }[source.kind](source, sample)


def wav_window(sizes: list[int], minutes: float, start_fraction: float) -> tuple[float, float]:
    """(start_s, length_s) of the window shared by all tracks of one meeting."""
    duration = min((s - 44) / WAV_BYTES_PER_SECOND for s in sizes)
    length = min(minutes * 60.0, duration)
    start = min(duration * start_fraction, duration - length)
    return max(start, 0.0), length


def _plan_wav_slices(source: Source, sample: bool) -> Plan:
    prefix = source.get("strip_prefix")
    wanted = source.get("sample_match") if sample else None
    minutes = source.pick("minutes_per_meeting", sample)
    meetings: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for size, url in read_url_list(source):
        if wanted is None or wanted in url:
            meetings[url.removeprefix(prefix).split("/")[0]].append((size, url))
    files = []
    for meeting, tracks in sorted(meetings.items()):
        start, length = wav_window([s for s, _ in tracks], minutes, source.get("start_fraction"))
        for _, url in sorted(tracks, key=lambda t: t[1]):
            files.append(
                PlannedFile(
                    path=url.removeprefix(prefix),
                    size=44 + int(length * WAV_BYTES_PER_SECOND),
                    url=url,
                    extra={"meeting": meeting, "start_s": start, "length_s": length},
                )
            )
    return Plan(source, files)


def _hf_revision(source: Source) -> str:
    return source.get("revision") or _hf_api().dataset_info(source.get("repo")).sha


def _hf_paths(source: Source, revision: str, patterns: list[str]) -> list[tuple[str, int]]:
    tree = _hf_api().list_repo_tree(
        source.get("repo"), repo_type="dataset", recursive=True, revision=revision
    )
    return sorted(
        (e.path, e.size) for e in tree if getattr(e, "size", None) is not None and matches(e.path, patterns)
    )


def select_row_groups(sizes: list[int], max_bytes: int) -> list[int]:
    """Row groups taken in order until the cap; always at least one."""
    chosen, total = [], 0
    for index, size in enumerate(sizes):
        if chosen and total + size > max_bytes:
            break
        chosen.append(index)
        total += size
    return chosen


def _row_group_sizes(repo: str, path: str, revision: str) -> list[int]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    with fs.open(f"datasets/{repo}/{path}", "rb", revision=revision, block_size=256 * 1024) as f:
        meta = pq.ParquetFile(f).metadata
    return [
        sum(meta.row_group(i).column(c).total_compressed_size for c in range(meta.num_columns))
        for i in range(meta.num_row_groups)
    ]


def _plan_hf_rowgroups(source: Source, sample: bool) -> Plan:
    revision = _hf_revision(source)
    cap = source.pick("max_bytes_per_file", sample)
    files = []
    for path, _ in _hf_paths(source, revision, source.pick("files", sample)):
        sizes = _row_group_sizes(source.get("repo"), path, revision)
        groups = select_row_groups(sizes, cap)
        files.append(
            PlannedFile(path=path, size=sum(sizes[i] for i in groups), extra={"row_groups": groups})
        )
    return Plan(source, files, revision)


def _plan_hf_tar_stream(source: Source, sample: bool) -> Plan:
    revision = _hf_revision(source)
    languages = source.pick("languages", sample)
    cap = source.pick("max_bytes_per_file", sample)
    archives = dict(_hf_paths(source, revision, expand(source.get("files"), languages)))
    extras = dict(_hf_paths(source, revision, expand(source.get("extra_files", []), languages)))
    files = [PlannedFile(path=p, size=min(s, cap), extra={"stream": True}) for p, s in archives.items()]
    files += [PlannedFile(path=p, size=s, extra={"stream": False}) for p, s in extras.items()]
    return Plan(source, sorted(files, key=lambda f: f.path), revision)


def _plan_http_tar_stream(source: Source, sample: bool) -> Plan:
    cap = source.pick("max_bytes_per_file", sample)
    return Plan(
        source,
        [
            PlannedFile(path=u.rsplit("/", 1)[1], size=min(_head_size(u), cap), url=u)
            for u in source.get("urls")
        ],
    )


def _plan_http(source: Source, sample: bool) -> Plan:
    urls = source.pick("urls", sample, [])
    return Plan(
        source, [PlannedFile(path=u.rsplit("/", 1)[1], size=_head_size(u), url=u) for u in urls]
    )


# ------------------------------------------------------- streaming helpers


class CountingReader:
    """File-like wrapper that counts bytes read from an HTTP response."""

    def __init__(self, raw):
        self.raw = raw
        self.count = 0

    def read(self, n: int = -1) -> bytes:
        chunk = self.raw.read(n)
        self.count += len(chunk)
        return chunk


def stream_tar(fileobj, target: Path, max_bytes: int, keep: list[str] | None = None) -> dict:
    """Extract regular members from a tar stream until `max_bytes` have been
    read from the network, finishing the member in progress. Returns counts."""
    reader = CountingReader(fileobj)
    kept = skipped = 0
    with tarfile.open(fileobj=reader, mode="r|*") as archive:
        for member in archive:
            if member.isfile() and (keep is None or matches(member.name, keep)):
                destination = target / member.name
                if not destination.resolve().is_relative_to(target.resolve()):
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, open(destination, "wb") as out:
                    while chunk := source.read(1 << 20):
                        out.write(chunk)
                kept += 1
            else:
                skipped += 1
            if reader.count >= max_bytes:
                break
    return {"members_kept": kept, "members_skipped": skipped, "bytes_read": reader.count}


def parse_wav_header(header: bytes) -> tuple[bytes, int]:
    """Return (fmt chunk payload, byte offset of the data chunk payload)."""
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    offset, fmt = 12, None
    while offset + 8 <= len(header):
        chunk, size = header[offset : offset + 4], struct.unpack("<I", header[offset + 4 : offset + 8])[0]
        if chunk == b"fmt ":
            fmt = header[offset + 8 : offset + 8 + size]
        if chunk == b"data":
            if fmt is None:
                raise ValueError("data chunk before fmt chunk")
            return fmt, offset + 8
        offset += 8 + size + (size & 1)
    raise ValueError("no data chunk in header")


def wav_bytes(fmt: bytes, pcm: bytes) -> bytes:
    """A complete WAV file from a fmt payload and PCM data."""
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def inspect_archive(url: str, max_bytes: int, headers: dict | None = None) -> list[tuple[str, int, int]]:
    """List (member name, size, bytes read so far) from the start of a tar
    archive until `max_bytes` have been read. Nothing is written to disk."""
    members = []
    with _request(url, headers=headers) as response:
        reader = CountingReader(response)
        with tarfile.open(fileobj=reader, mode="r|*") as archive:
            for member in archive:
                members.append((member.name, member.size, reader.count))
                if reader.count >= max_bytes:
                    break
    return members


# ------------------------------------------------------------- downloading


def download(plan: Plan, workers: int = 8, meter=None) -> None:
    """Fetch every planned file. `meter` (a progress.Progress in bytes) is
    advanced as bytes arrive, or on completion where reads are not counted
    (parquet row groups), and credited without pacing for files already done."""
    target = raw_dir(plan.source)
    target.mkdir(parents=True, exist_ok=True)
    fetch = {
        "wav_slices": _fetch_wav_slice,
        "hf_rowgroups": _fetch_rowgroups,
        "hf_tar_stream": _fetch_hf_tar,
        "http_tar_stream": _fetch_http_tar,
        "http": _fetch_http,
    }[plan.source.kind]
    results = {}
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, plan, f, target, meter): f.path for f in plan.files}
        for future in cf.as_completed(futures):
            results[futures[future]] = future.result()
    _write_manifest(plan, target, results)


def _done(target: Path, path: str) -> Path:
    return target / ".done" / (path.replace("/", "__") + ".json")


def _once(fetch):
    """Skip files already fetched; record each file's result for the manifest."""

    def wrapper(plan: Plan, f: PlannedFile, target: Path, meter=None):
        marker = _done(target, f.path)
        if marker.exists():
            if meter:
                meter.skip(f.size)
            return json.loads(marker.read_text())
        counter = _FileCounter(meter) if meter else None
        _local.counter = counter
        try:
            result = fetch(plan, f, target) or {}
        finally:
            _local.counter = None
        if meter:  # credit what was not counted live; streams stop near f.size
            meter.advance(max(0, f.size - counter.count))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(result))
        return result

    return wrapper


@_once
def _fetch_wav_slice(plan: Plan, f: PlannedFile, target: Path) -> dict:
    with _request(f.url, headers={"Range": "bytes=0-4095"}) as response:
        fmt, data_offset = parse_wav_header(response.read())
    channels, rate, _, block_align = struct.unpack("<HIIH", fmt[2:14])
    first = data_offset + int(f.extra["start_s"] * rate) * block_align
    count = int(f.extra["length_s"] * rate) * block_align
    with _request(f.url, headers={"Range": f"bytes={first}-{first + count - 1}"}) as response:
        if response.status != 206:
            raise RuntimeError(f"{f.url}: server ignored the range request")
        pcm = response.read()
    if len(pcm) != count:
        raise RuntimeError(f"{f.url}: expected {count} bytes, got {len(pcm)}")
    destination = target / f.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(wav_bytes(fmt, pcm))
    return {"start_s": f.extra["start_s"], "length_s": f.extra["length_s"], "channels": channels, "rate": rate}


@_once
def _fetch_rowgroups(plan: Plan, f: PlannedFile, target: Path) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    remote = f"datasets/{plan.source.get('repo')}/{f.path}"
    with fs.open(remote, "rb", revision=plan.revision, block_size=4 << 20) as handle:
        table = pq.ParquetFile(handle).read_row_groups(f.extra["row_groups"])
    destination = target / f.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    return {"row_groups": f.extra["row_groups"], "rows": table.num_rows}


def _hf_file_url(plan: Plan, path: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(plan.source.get("repo"), path, repo_type="dataset", revision=plan.revision)


@_once
def _fetch_hf_tar(plan: Plan, f: PlannedFile, target: Path) -> dict:
    url = _hf_file_url(plan, f.path)
    if not f.extra.get("stream"):
        destination = target / f.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _request(url, headers=_hf_auth()) as response:
            destination.write_bytes(response.read())
        return {"bytes": destination.stat().st_size}
    with _request(url, headers=_hf_auth()) as response:
        return stream_tar(response, target / f.path.rsplit("/", 1)[0], f.size)


@_once
def _fetch_http_tar(plan: Plan, f: PlannedFile, target: Path) -> dict:
    with _request(f.url) as response:
        return stream_tar(response, target, f.size, plan.source.get("keep"))


@_once
def _fetch_http(plan: Plan, f: PlannedFile, target: Path) -> dict:
    destination = target / f.path
    with _request(f.url) as response, open(destination, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    if destination.stat().st_size != f.size:
        raise RuntimeError(f"{f.url}: size {destination.stat().st_size} != {f.size}")
    return {"bytes": f.size}


def _write_manifest(plan: Plan, target: Path, results: dict) -> None:
    manifest = {
        "source": plan.source.name,
        "licence": plan.source.licence,
        "attribution": plan.source.attribution,
        "role": plan.source.role,
        "revision": plan.revision,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [
            {"path": f.path, "planned_bytes": f.size, **results.get(f.path, {})} for f in plan.files
        ],
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
