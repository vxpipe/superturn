"""Inspect published YODAS v1 duration indices without fetching speech."""

from collections import Counter
import hashlib
import json
import re
from urllib.request import Request, urlopen

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root

REPOSITORY = "espnet/yodas"
REVISION = "0cad6a0ebec75e945fef92084052be69457fe9d1"
LANGUAGE_BYTES = {"ar": 339180, "hi": 234903, "id": 244162, "nl": 366732, "pt": 276556, "tr": 367777}
VIDEO = re.compile(r"^([A-Za-z0-9_-]{11})-[0-9]")


def inventory(payload):
    """Duration entries describe clips; they do not identify the active speaker."""
    counts, by_video = Counter(), Counter()
    for line in payload.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        counts["rows"] += 1
        match = VIDEO.match(fields[0])
        if len(fields) != 2 or match is None:
            counts["unrecognized"] += 1
            continue
        try:
            seconds = float(fields[1])
        except ValueError:
            counts["unrecognized"] += 1
            continue
        if not 0 < seconds < 3600:
            counts["invalid_duration"] += 1
            continue
        counts["parsed"] += 1
        if 5 <= seconds <= 30:
            counts["duration_5_to_30s"] += 1
            by_video[match.group(1)] += 1
    return {**counts, "distinct_videos_5_to_30s": len(by_video),
            "max_entries_from_one_video": max(by_video.values(), default=0)}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("inspect source data on Vast")
    output = root / "manifests/yodas-segment-inventory"
    output.mkdir(parents=True, exist_ok=True)
    meter = Progress("yodas-segment-inventory", len(LANGUAGE_BYTES), "duration indices")
    cells = {}
    try:
        for lang, size in LANGUAGE_BYTES.items():
            path = output / f"{lang}-durations.txt"
            relative = f"data/{lang}000/duration/00000000.txt"
            if not path.exists():
                request = Request(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{relative}",
                                  headers={"Accept-Encoding": "identity"})
                with urlopen(request, timeout=30) as reply:
                    payload = reply.read(size + 1)
                if len(payload) != size:
                    raise ValueError("pinned index has incorrect size")
                temporary = path.with_suffix(".partial")
                temporary.write_bytes(payload)
                temporary.replace(path)
            body = path.read_bytes()
            if len(body) != size:
                raise ValueError("staged index has incorrect size")
            cells[lang] = {"path": relative, "bytes": size, "sha256": hashlib.sha256(body).hexdigest(),
                           **inventory(body)}
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result = {"scope": "one published duration index per language; metadata only",
              "repository": REPOSITORY, "revision": REVISION, "license": "CC BY 3.0",
              "audio_bytes": 0, "metadata_bytes": sum(LANGUAGE_BYTES.values()), "languages": cells}
    _write_json(root / "reports/yodas-segment-inventory.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(data_root())
