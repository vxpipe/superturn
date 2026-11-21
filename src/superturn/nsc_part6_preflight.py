"""Check selective tar access and pairing for the Singapore NSC Part 6 mirror."""

from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
import json
import re
import tarfile
from urllib.parse import quote

from .acquire import _request
from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root

REPO = "recursal/reprocessed_singapore_national_speech_corpus"
REVISION = "5e8294c3fb5f2634fd1d7004f7c0264969fe6687"
DESIGNS = ("BanksTelcosInsurance", "HDBMOEMSF", "HotelsHolidaysRestaurant")
BLOCK = 512
HEADERS = 256
REQUESTS = 330
INVENTORY_BYTES = 64 << 10
BYTES = ((1 << 20) - INVENTORY_BYTES) // len(DESIGNS)
PAIR_TARGET = 6
MEMBER = re.compile(r"SE(?P<session>[0-9]+)-Pt(?P<task>[A-Za-z]+)-SP(?P<speaker>[0-9]+)\.(?P<format>flac|jsonl)")


def inventory():
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/Part%206"
    with _request(url) as response:
        payload = response.read(INVENTORY_BYTES)
    if len(payload) == INVENTORY_BYTES:
        raise ValueError("the archive inventory exceeds its byte budget")
    entries = json.loads(payload)
    discovered = {}
    for design in DESIGNS:
        group = sorted((item["path"], item["size"], item["lfs"]["oid"])
                       for item in entries if re.fullmatch(rf"Part 6/{design}-[0-9]{{2}}\.tar", item["path"]))
        if len(group) != 4 or any(name != f"Part 6/{design}-{i:02d}.tar" or size <= 0
                                  for i, (name, size, _) in enumerate(group)):
            raise ValueError("the pinned mirror has an unexpected fragment layout")
        discovered[design] = group
    return discovered, len(payload)


class RangeReader:
    def __init__(self, parts, *, byte_limit=BYTES, request_limit=REQUESTS):
        self.parts = parts
        self.ends = tuple(accumulate(size for _, size, _ in parts))
        self.byte_limit = byte_limit
        self.request_limit = request_limit
        self.bytes = 0
        self.requests = 0

    def read(self, position, count):
        data = bytearray()
        while count:
            index = bisect_right(self.ends, position)
            if index >= len(self.parts):
                raise ValueError("tar member extends beyond the logical archive")
            name, size, _ = self.parts[index]
            start = position - (self.ends[index - 1] if index else 0)
            length = min(count, size - start)
            end = start + length - 1
            if self.bytes + length + 1 > self.byte_limit or self.requests >= self.request_limit:
                raise ValueError("the archive read exceeded its transport budget")
            self.requests += 1
            path = quote(name, safe="/")
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}?download=true&offset={start}"
            with _request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as response:
                if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
                        or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError("the mirror did not honor the exact byte range")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length != str(length):
                    raise ValueError("the mirror promised an unexpected byte count")
                result = response.read(length + 1)
            self.bytes += len(result)
            if len(result) != length:
                raise ValueError("the mirror returned an unexpected byte count")
            data.extend(result)
            position += length
            count -= length
        return bytes(data)


def pax_fields(raw):
    result = {}
    while raw:
        digits, _, _ = raw.partition(b" ")
        length = int(digits)
        if not 0 < length <= len(raw) or raw[length - 1:length] != b"\n":
            raise ValueError("invalid PAX metadata length")
        key, sep, value = raw[len(digits) + 1:length - 1].partition(b"=")
        if not sep or not key:
            raise ValueError("invalid PAX metadata record")
        if key.startswith(b"GNU.sparse.") or key == b"SCHILY.realsize":
            raise ValueError("sparse archive members are unsupported")
        if key in {b"path", b"size"}:
            result[key.decode("ascii")] = value.decode("utf-8")
            if key == b"size" and not re.fullmatch(rb"[0-9]+", value):
                raise ValueError("invalid PAX size")
        raw = raw[length:]
    return result


def scan(design, fragments):
    reader = RangeReader(fragments)
    meter = Progress(f"nsc-part6-headers-{design}", HEADERS, "headers")
    groups = defaultdict(lambda: defaultdict(dict))
    position, parsed = 0, 0
    extended, shared = {}, {}
    pairs = []
    types = Counter()
    try:
        for _ in range(HEADERS):
            header = reader.read(position, BLOCK)
            if not any(header):
                break
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            parsed += 1
            meter.advance(1)
            content = position + BLOCK
            size = member.size
            if size < 0:
                raise ValueError("negative tar member length")
            if member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
                if not 0 < size <= 4096:
                    raise ValueError("unsupported tar metadata length")
                metadata = reader.read(content, size)
                if member.type == tarfile.GNUTYPE_LONGNAME:
                    extended["path"] = metadata.rstrip(b"\0").decode("utf-8")
                elif member.type == tarfile.XGLTYPE:
                    shared.update(pax_fields(metadata))
                else:
                    extended.update(pax_fields(metadata))
            elif member.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
                fields = {**shared, **extended}
                name = fields.get("path", member.name).rsplit("/", 1)[-1]
                size = int(fields.get("size", size))
                extended.clear()
                if member.isdir() and size != 0:
                    raise ValueError("unexpected nonempty directory member")
                if match := MEMBER.fullmatch(name):
                    if not member.isfile() or size <= 0:
                        raise ValueError("invalid paired archive member")
                    session, task, speaker, fmt = match.groups()
                    members = groups[session, task][speaker]
                    if fmt in members:
                        raise ValueError("duplicate speaker member")
                    members[fmt] = {"byte_offset": content, "bytes": size}
                    types[fmt] += 1
            else:
                raise ValueError("unsupported tar member type")
            position = content + ((size + BLOCK - 1) // BLOCK) * BLOCK
            if size < 0 or position > reader.ends[-1]:
                raise ValueError("invalid tar member extent")
            pairs = [{"session": session, "task": task, "speakers": voices}
                     for (session, task), voices in sorted(groups.items())
                     if len(voices) == 2 and all(set(parts) == {"flac", "jsonl"} for parts in voices.values())]
            if len(pairs) >= PAIR_TARGET:
                break
    except BaseException:
        meter.finish("failed")
        raise
    meter.skip(HEADERS - parsed)
    meter.finish()
    return {"fragments": fragments, "headers": parsed, "logical_range_requests": reader.requests,
            "metadata_bytes": reader.bytes, "recognized_members": types, "pairs": pairs}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("archive metadata preflight is restricted to the campaign machine")
    parts, inventory_bytes = inventory()
    with ThreadPoolExecutor(max_workers=len(DESIGNS)) as pool:
        futures = {design: pool.submit(scan, design, parts[design]) for design in DESIGNS}
        result = {design: future.result() for design, future in futures.items()}
    report = {"repository": REPO, "revision": REVISION, "designs": result,
              "prefix_pairs_are_provisional": True,
              "inventory_bytes": inventory_bytes,
              "metadata_bytes": inventory_bytes + sum(row["metadata_bytes"] for row in result.values()),
              "audio_or_transcript_payload_bytes": 0}
    _write_json(root / "reports/nsc-part6-preflight.json", report)
    return report


if __name__ == "__main__":
    result = run(data_root())
    print(json.dumps({name: {k: len(row[k]) if k == "pairs" else row[k]
                             for k in ("headers", "metadata_bytes", "pairs")}
                      for name, row in result["designs"].items()}, indent=2))
