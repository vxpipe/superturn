"""Stage the exact HarperValleyBank selection for a pilot on Vast.ai."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from .acquire import _request
from .ami_calibration import _write_json
from .harper_valley_plan import BUDGET, RAW, REVISION, SPLITS
from .progress import Progress
from .sources import data_root, load_sources

SOURCE = "harper-valley"


def verified_selection(root, expected):
    plan = json.loads((root / "reports" / "harper-valley-plan.json").read_text())
    manifest = root / "cache" / "harper-valley" / f"{REVISION}-selection.json"
    payload = manifest.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected or plan["selection_sha256"] != expected:
        raise ValueError("HarperValleyBank selection hash mismatch")
    selected = json.loads(payload)
    if set(selected) != set(SPLITS) or plan["revision"] != REVISION:
        raise ValueError("HarperValleyBank selection has an unexpected scope")
    paths = set()
    voices = set()
    total = plan["metadata_bytes"]
    for split in SPLITS:
        split_voices = set()
        split_bytes = 0
        for call in selected[split]:
            sid = call["id"]
            if not re.fullmatch(r"[0-9a-f]{16}", sid):
                raise ValueError("unexpected HarperValleyBank conversation identifier")
            files = call["files"]
            allowed = {f"data/audio/{role}/{sid}.wav" for role in ("caller", "agent")}
            allowed.add(f"data/transcript/{sid}.json")
            if len(files) != 3 or {file["path"] for file in files} != allowed or paths & allowed:
                raise ValueError("unexpected or duplicate HarperValleyBank payload path")
            paths.update(allowed)
            split_voices.update((call["caller"], call["agent"]))
            for file in files:
                if type(file["bytes"]) is not int or file["bytes"] <= 0:
                    raise ValueError("invalid HarperValleyBank payload byte count")
                split_bytes += file["bytes"]
        receipt = plan["splits"][split]
        if voices & split_voices or split_bytes != receipt["bytes"] or len(selected[split]) != receipt["calls"]:
            raise ValueError("HarperValleyBank selection no longer matches split constraints")
        total += split_bytes
        voices.update(split_voices)
    if total != plan["projected_bytes"] or not 0 < total <= BUDGET:
        raise ValueError("HarperValleyBank selection exceeds or differs from its recorded budget")
    return selected


def fetch_file(raw, file):
    path = raw / file["path"]
    size = file["bytes"]
    if path.is_file() and path.stat().st_size == size:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with _request(f"{RAW}/{file['path']}") as response, NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            remaining = size
            while chunk := response.read(min(remaining + 1, 1 << 20)):
                if len(chunk) > remaining:
                    raise ValueError("HarperValleyBank download exceeds planned file size")
                stream.write(chunk)
                remaining -= len(chunk)
            if remaining:
                raise ValueError("HarperValleyBank download is shorter than planned")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def run(root, expected, splits):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("HarperValleyBank staging uses the campaign machine")
    if not splits or set(splits) - set(SPLITS):
        raise ValueError("select one or more valid HarperValleyBank splits")
    selected = verified_selection(root, expected)
    source = load_sources()[SOURCE]
    raw = root / "raw" / SOURCE
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / ".acquire.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [file for split in SPLITS if split in splits for call in selected[split] for file in call["files"]]
        meter = Progress("harper-valley-acquisition", sum(file["bytes"] for file in files), "B")
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                fetched = pool.map(lambda file: fetch_file(raw, file), files)
                for file, downloaded in zip(files, fetched, strict=True):
                    (meter.advance if downloaded else meter.skip)(file["bytes"])
            staged = []
            for split in SPLITS:
                for call in selected[split]:
                    for file in call["files"]:
                        path = raw / file["path"]
                        if path.is_file() and path.stat().st_size == file["bytes"]:
                            staged.append({"split": split, "path": file["path"], "bytes": file["bytes"]})
            destination = raw / "MANIFEST.json"
            _write_json(destination, {"source": SOURCE, "licence": source.licence,
                                      "attribution": source.attribution, "role": source.role, "revision": REVISION,
                                      "selection_sha256": expected, "files": staged})
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_sha256")
    parser.add_argument("--split", nargs="+", choices=SPLITS, default=("train", "validation"))
    args = parser.parse_args()
    print(run(data_root(), args.selection_sha256, args.split))
