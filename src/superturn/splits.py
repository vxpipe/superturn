"""Deterministic, reproducible, speaker-disjoint train/validation/test splits.

Every dataset is split on its own, so each dataset appears in every split in
about the same proportion. Within a dataset, recordings are grouped so that
no speaker appears in more than one split: two recordings that share any
speaker belong to the same group (AMI, for example, reuses the same four
people across each series of meetings). A group's split depends only on the
source name and the group key, so it is identical on every machine and run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

# Percent of groups per split, for training sources.
SPLITS = (("train", 80), ("validation", 10), ("test", 10))


def bucket(source: str, group: str) -> int:
    digest = hashlib.sha256(f"superturn-split-v1\0{source}\0{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def assign(source: str, group: str, role: str) -> str:
    """Split for one group. Test-only sources always go to test."""
    if role == "test":
        return "test"
    b = bucket(source, group)
    edge = 0
    for name, share in SPLITS:
        edge += share
        if b < edge:
            return name
    raise AssertionError("split shares must sum to 100")


def speaker_groups(recordings: Iterable[tuple[str, Iterable[str]]]) -> dict[str, str]:
    """Map each recording to a group key such that recordings sharing a speaker
    share a group. Input: (recording_id, speaker_ids). The key is the smallest
    recording ID in the group, so it does not depend on input order."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    recording_ids = []
    for recording, speakers in recordings:
        node = f"r\0{recording}"
        recording_ids.append(recording)
        find(node)
        for speaker in speakers:
            union(node, f"s\0{speaker}")

    members: dict[str, list[str]] = {}
    for recording in recording_ids:
        members.setdefault(find(f"r\0{recording}"), []).append(recording)
    return {r: min(group) for group in members.values() for r in group}


def split_recordings(
    source: str, role: str, recordings: Iterable[tuple[str, Iterable[str]]]
) -> dict[str, str]:
    """Map each recording ID to its split, speaker-disjoint within the source."""
    groups = speaker_groups(recordings)
    return {r: assign(source, g, role) for r, g in groups.items()}
