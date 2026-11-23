"""Count Indic DiarBench recording groups without reading audio or annotations."""

import io
import json
from collections import Counter, defaultdict
from urllib.parse import quote

from .acquire import _hf_paths, _request
from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root, load_sources

REVISION = "92877bad8aab6e598167d91c6ee02aa8ca6ede09"
COLUMNS = {"recording_id", "sample_id", "dataset_type", "num_speakers", "duration_seconds"}
BYTE_LIMIT = 2_000_000


class MetadataOnlyParquet(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.cursor, self.bytes = url, size, 0, 0
        trailer = self.fetch(size - 8, size)
        footer = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or not 0 < footer < min(size - 8, BYTE_LIMIT - 8):
            raise ValueError("unexpected Indic DiarBench Parquet footer")
        self.ranges = [(0, 4), (size - footer - 8, size)]

    def fetch(self, start, stop):
        if self.bytes + stop - start > BYTE_LIMIT:
            raise ValueError("Indic DiarBench metadata transport limit exceeded")
        with _request(self.url + f"?download=true&offset={start}",
                      headers={"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity"}) as response:
            if (response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{stop - 1}/{self.size}"
                    or response.headers.get("Content-Encoding", "identity") != "identity"):
                raise ValueError("Indic DiarBench origin did not honor a metadata byte range")
            result = response.read(stop - start + 1)
        if len(result) != stop - start:
            raise ValueError("Indic DiarBench returned an incorrect metadata byte range")
        self.bytes += len(result)
        return result

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.cursor = {io.SEEK_SET: 0, io.SEEK_CUR: self.cursor, io.SEEK_END: self.size}[whence] + offset
        return self.cursor

    def tell(self):
        return self.cursor

    def read(self, size=-1):
        start = self.cursor
        stop = self.size if size < 0 else min(start + size, self.size)
        if stop - start > BYTE_LIMIT:
            raise ValueError("unbounded Parquet read refused")
        # Parquet may read a block larger than its footer. The disallowed
        # regions remain zero and never cross the network.
        data = bytearray(stop - start)
        for left, right in self.ranges:
            begin, end = max(start, left), min(stop, right)
            if begin < end:
                data[begin - start:end - start] = self.fetch(begin, end)
        self.cursor = stop
        return bytes(data)


def project(path, size, repo):
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/datasets/{repo}/resolve/{REVISION}/{quote(path, safe='/')}"
    with MetadataOnlyParquet(url, size) as source:
        reader = pq.ParquetFile(source)
        for group in range(reader.metadata.num_row_groups):
            row_group = reader.metadata.row_group(group)
            for index in range(row_group.num_columns):
                column = row_group.column(index)
                if column.path_in_schema in COLUMNS:
                    start = column.dictionary_page_offset or column.data_page_offset
                    source.ranges.append((start, start + column.total_compressed_size))
        rows = reader.read(columns=sorted(COLUMNS), use_threads=False).to_pylist()
        return rows, source.bytes


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Indic DiarBench inventory runs on Vast")
    source = load_sources()["indic_diarbench"]
    repo = source.get("repo")
    paths = _hf_paths(source, REVISION, source.get("files"))
    if not paths:
        raise ValueError("the pinned Indic DiarBench file inventory is empty")
    report = {"revision": REVISION, "verified_cross_recording_speaker_ids": False,
              "protected_source_role": source.role, "projected_bytes": 0, "languages": {}}
    by_language = defaultdict(list)
    meter = Progress("diarbench-stage0-inventory", len(paths), "language files")
    try:
        for path, size in paths:
            language, _ = path.split("/")
            rows, used = project(path, size, repo)
            by_language[language].extend(rows)
            report["projected_bytes"] += used
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    for language, rows in sorted(by_language.items()):
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("the Indic DiarBench inventory contains repeated clips")
        reports = defaultdict(list)
        for row in rows:
            reports[row["dataset_type"]].append(row)
        report["languages"][language] = {
            "clips": len(rows), "recordings": len({row["recording_id"] for row in rows}),
            "conditions": {condition: {"clips": len(items),
                                       "recordings": len({row["recording_id"] for row in items}),
                                       "seconds": round(sum(row["duration_seconds"] for row in items), 1),
                                       "speakers_per_clip": Counter(row["num_speakers"] for row in items)}
                           for condition, items in sorted(reports.items())},
        }
    _write_json(root / "reports" / "diarbench-stage0-inventory.json", report)
    print(json.dumps({"languages": len(report["languages"]), "projected_bytes": report["projected_bytes"]}, indent=2))


if __name__ == "__main__":
    run(data_root())
