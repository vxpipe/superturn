"""Verify bounded sequential access to selected captioned YODAS gzip objects."""

from collections import Counter
import gzip
import io
import json
from pathlib import PurePosixPath
import tarfile
from urllib.request import Request, urlopen

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .progress import Progress
from .sources import budget_bytes, data_root, load_sources
from .yodas_caption_overlap import CAPTION_ID
from .yodas_segment_inventory import REPOSITORY, REVISION
from .yodas_small_archive_inventory import JOB as INVENTORY

JOB = "yodas-audio-prefix-probe"
BYTES_PER_ARCHIVE = 8 * 1024 * 1024
MAX_MEMBER = 8 * 1024 * 1024
MIN_INDEX_VIDEOS = 10


def fetch(path, entry):
    size = BYTES_PER_ARCHIVE
    expected = f"bytes 0-{size - 1}/{entry['archive_bytes']}"
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{entry['audio_path']}"
    request = Request(url, headers={"Range": f"bytes=0-{size - 1}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60) as response:
        if (response.status != 206 or response.headers.get("Content-Range") != expected
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("the source refused the exact bounded gzip range")
        value = response.read(size + 1)
    if len(value) != size:
        raise ValueError("incorrect prefix length")
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(value)
    temporary.replace(path)


def inspect(payload, allowed):
    counts, complete = Counter(), Counter()
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        try:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.size > MAX_MEMBER:
                        counts["size_limited"] += 1
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None or len(extracted.read()) != member.size:
                        raise EOFError("incomplete tar member")
                    counts["complete_members"] += 1
                    name = PurePosixPath(member.name)
                    counts[f"suffix_{name.suffix.casefold()}"] += 1
                    key = CAPTION_ID.fullmatch(name.stem)
                    if key and key.group(1) in allowed and name.suffix.casefold() in {".wav", ".flac"}:
                        complete[key.group(1)] += 1
        except (EOFError, tarfile.ReadError):
            counts["ended_at_prefix_boundary"] += 1
    return {**counts, "matched_videos": len(complete), "matched_members": sum(complete.values()),
            "largest_matched_video_members": max(complete.values(), default=0)}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("real audio stays on Vast")
    report = json.loads((root / "reports" / f"{INVENTORY}.json").read_text())
    source = load_sources()[JOB]
    if source.get("repo") != REPOSITORY or source.get("revision") != REVISION or source.role != "metadata":
        raise ValueError("the probe has a different registered source")
    private = root / "manifests" / INVENTORY / "membership.json"
    if report["revision"] != REVISION or file_digest(private) != report["membership_sha256"]:
        raise ValueError("the screened video inventory changed")
    membership = json.loads(private.read_text())
    selection = {}
    for lang, row in report["languages"].items():
        eligible = [entry for entry in row["options"] if entry["videos_after_text_screen"] >= MIN_INDEX_VIDEOS]
        if not eligible:
            raise ValueError(f"no metadata archive meets the minimum video count in {lang}")
        selection[lang] = min(eligible, key=lambda entry: entry["archive_bytes"])
    raw = root / "raw"
    folder = raw / JOB
    folder.mkdir(parents=True, exist_ok=True)
    targets = {lang: folder / f"{lang}-{row['shard']}-prefix.tar.gz" for lang, row in selection.items()}
    current = sum(path.stat().st_size for path in raw.rglob("*") if path.is_file())
    reserved = sum(BYTES_PER_ARCHIVE for path in targets.values() if not path.exists())
    if current + reserved > budget_bytes() or len(targets) * BYTES_PER_ARCHIVE > source.get("max_total_bytes"):
        raise ValueError("the YODAS transport probe would exceed the campaign download cap")
    meter = Progress(JOB, len(selection) * BYTES_PER_ARCHIVE, "bytes")
    result = {}
    try:
        for lang, row in selection.items():
            target = targets[lang]
            if not target.exists():
                fetch(target, row)
                meter.advance(BYTES_PER_ARCHIVE)
            else:
                meter.skip(BYTES_PER_ARCHIVE)
            prefix = target.read_bytes()
            if len(prefix) != BYTES_PER_ARCHIVE:
                raise ValueError("cached prefix differs from the requested byte range")
            allowed = membership[lang][row["shard"]]["eligible_video_counts"]
            stats = inspect(prefix, allowed)
            result[lang] = {"source": row["audio_path"], "full_archive_sha256": row["archive_sha256"],
                            "full_archive_bytes": row["archive_bytes"], "prefix_bytes": len(prefix),
                            "prefix_sha256": file_digest(target), "video_candidates_in_full_index": len(allowed),
                            **stats}
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"scope": "bounded transport and member geometry only; no training audio accepted",
              "revision": REVISION, "license": source.licence, "attribution": source.attribution,
              "requested_audio_bytes": len(selection) * BYTES_PER_ARCHIVE, "languages": result}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
