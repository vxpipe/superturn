"""Build the AMI and ICSI per-headset URL lists in configs/url-lists/.

Both corpora are served as plain files from the University of Edinburgh. AMI
files are found by probing meeting IDs from the download page; ICSI files come
from the download script its download form generates. Each list row is
`size<TAB>url`, with the size from a HEAD request, so planning needs no
network access to these servers.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .sources import REPO_ROOT

AMI_PAGE = "https://groups.inf.ed.ac.uk/ami/download/"
AMI_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ICSI_PAGE = "https://groups.inf.ed.ac.uk/ami/icsi/download/"
ICSI_FORM = "https://homepages.inf.ed.ac.uk/cgi/simonk/ICSI/estimate.cgi"
HEADERS = {"User-Agent": "superturn-downloader"}


def _get(url: str, data: bytes | None = None) -> str:
    request = urllib.request.Request(url, data=data, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read().decode("utf-8", "replace")


def _size(url: str, attempts: int = 4) -> int | None:
    for _ in range(attempts):
        try:
            request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                return int(response.headers["Content-Length"])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
        except (OSError, KeyError, ValueError):
            continue
    raise RuntimeError(f"could not size {url}")


def _sized(urls: list[str]) -> list[tuple[int, str]]:
    with cf.ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(_size, urls))
    return sorted((s, u) for s, u in zip(sizes, urls) if s is not None)


def ami_urls() -> list[str]:
    page = _get(AMI_PAGE)
    values = sorted(set(re.findall(r"""<option[^>]*value=['"]([A-Z]{2}\d{4}[a-z]?)['"]""", page, re.I)))
    meetings = []
    for value in values:  # series IDs (ES2002) expand to their meetings (ES2002a-d)
        meetings += [value] if value[-1].islower() else [value + s for s in "abcd"] + [value]
    return [
        f"{AMI_BASE}/{m}/audio/{m}.Headset-{h}.wav" for m in meetings for h in range(6)
    ]


def icsi_urls() -> list[str]:
    page = _get(ICSI_PAGE)
    sessions = re.findall(r"option value='([A-Za-z0-9]+)'", page)
    form = [("sessionID", s) for s in sessions] + [("mediaType", "Headset-"), ("detailed", "no")]
    result = _get(ICSI_FORM, urllib.parse.urlencode(form).encode())
    script_url = re.search(r'href="([^"]*wget\.sh)"', result).group(1)
    script = _get(script_url)
    return sorted(u.replace("/ami//", "/ami/") for u in re.findall(r"https://\S+\.sph", script))


def write(name: str, rows: list[tuple[int, str]]) -> Path:
    path = REPO_ROOT / "configs" / "url-lists" / f"{name}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{size}\t{url}\n" for size, url in sorted(rows, key=lambda r: r[1])))
    return path


def refresh() -> None:
    for name, urls in (("ami-headsets", ami_urls()), ("icsi-headsets", icsi_urls())):
        rows = _sized(urls)
        path = write(name, rows)
        print(f"{path.relative_to(REPO_ROOT)}: {len(rows)} files, {sum(s for s, _ in rows) / 1e9:.1f} GB")
