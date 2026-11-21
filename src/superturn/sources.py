"""Dataset source registry, read from configs/sources.toml."""

from __future__ import annotations

import fnmatch
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "sources.toml"
KINDS = {"wav_slices", "hf_rowgroups", "hf_tar_stream", "hf_files", "http_tar_stream", "http"}
ROLES = {
    "train-conversation",
    "train-read",
    "train-spontaneous",
    "test",
    "noise",
    "reverb",
    "metadata",
}


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    licence: str
    attribution: str
    role: str
    order: int
    settings: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.settings.get(key, default)

    def pick(self, key: str, sample: bool, default=None):
        """The `sample_<key>` setting in sample mode if present, else `<key>`."""
        if sample and f"sample_{key}" in self.settings:
            return self.settings[f"sample_{key}"]
        return self.settings.get(key, default)


def data_root() -> Path:
    return Path(os.environ.get("SUPERTURN_DATA", REPO_ROOT / "data")).resolve()


def raw_dir(source: Source) -> Path:
    return data_root() / "raw" / source.name


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def budget_bytes(path: Path = CONFIG) -> int:
    return int(float(_load(path)["budget_gb"]) * 1e9)


def load_sources(path: Path = CONFIG) -> dict[str, Source]:
    sources = {}
    for name, entry in _load(path)["sources"].items():
        entry = dict(entry)
        kind = entry.pop("kind")
        role = entry.pop("role")
        if kind not in KINDS:
            raise ValueError(f"{name}: unknown kind {kind!r}")
        if role not in ROLES:
            raise ValueError(f"{name}: unknown role {role!r}")
        for required in ("licence", "attribution"):
            if not entry.get(required):
                raise ValueError(f"{name}: missing {required}")
        sources[name] = Source(
            name=name,
            kind=kind,
            licence=entry.pop("licence"),
            attribution=entry.pop("attribution"),
            role=role,
            order=int(entry.pop("order", 9)),
            settings=entry,
        )
    return dict(sorted(sources.items(), key=lambda kv: (kv[1].order, kv[0])))


def select(sources: dict[str, Source], only: list[str] | None) -> list[Source]:
    if not only:
        return [source for source in sources.values() if not source.get("opt_in", False)]
    unknown = [n for n in only if n not in sources]
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(unknown)}")
    return [sources[n] for n in sources if n in only]


def matches(path: str, include: list[str], exclude: list[str] = ()) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in include) and not any(
        fnmatch.fnmatchcase(path, p) for p in exclude
    )


def read_url_list(source: Source) -> list[tuple[int, str]]:
    rows = []
    for line in (REPO_ROOT / source.get("list")).read_text().splitlines():
        if line.strip():
            size, url = line.split("\t")
            rows.append((int(size), url))
    return rows


def expand(templates: list[str], languages: list[str] | None) -> list[str]:
    """Fill `{language}` in path templates."""
    if not languages:
        return list(templates)
    return [t.format(language=lang) for lang in languages for t in templates]
