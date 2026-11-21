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


def stratified_groups(
    source: str, role: str, groups: Iterable[tuple[str, str]]
) -> dict[str, str]:
    """Assign whole speaker groups to approximate 80/10/10 within each language.

    Count an independent group once per language. Allocate validation and
    test groups for small languages when at least three groups are available.
    A multilingual group receives one split for all its languages.
    """
    from collections import Counter, defaultdict

    languages = defaultdict(set)
    for group, language in groups:
        languages[group].add(language)
    if role == "test":
        return dict.fromkeys(sorted(languages), "test")

    totals = Counter(language for tags in languages.values() for language in tags)
    goals = {}
    for language, total in totals.items():
        held_out = max(1, round(total * 0.1)) if total >= 3 else 0
        goals[language] = {"train": total - 2 * held_out, "validation": held_out, "test": held_out}
    counts = {language: Counter() for language in totals}
    remaining = totals.copy()
    assignments = {}
    order = sorted(languages, key=lambda group: (
        min(totals[tag] for tag in languages[group]), -len(languages[group]),
        hashlib.sha256(f"superturn-language-split-v2\0{source}\0{group}".encode()).digest(),
    ))
    for group in order:
        tags = sorted(languages[group])

        def priority(split):
            mandatory = 0.0
            deficit = 0.0
            for tag in tags:
                gap = goals[tag][split] - counts[tag][split]
                # Preserve space for a language's held-out splits as the
                # number of still-unassigned groups approaches its deficit.
                unfilled = sum(counts[tag][name] == 0 and goal > 0
                               for name, goal in goals[tag].items())
                if remaining[tag] <= unfilled and gap > 0 and counts[tag][split] == 0:
                    mandatory += 1 / totals[tag]
                deficit += gap / totals[tag]
            return mandatory, deficit

        split = max((name for name, _ in SPLITS), key=priority)
        assignments[group] = split
        for tag in tags:
            counts[tag][split] += 1
            remaining[tag] -= 1
    # Repair uncovered splits where a multilingual group can move without
    # removing another language's last group from a required split.
    for tag in sorted(totals, key=lambda name: (totals[name], name)):
        for split, _ in SPLITS:
            if not goals[tag][split] or counts[tag][split]:
                continue
            candidates = [
                group for group in order
                if tag in languages[group] and all(
                    not goals[other][assignments[group]] or counts[other][assignments[group]] > 1
                    for other in languages[group]
                )
            ]
            if candidates:
                group = candidates[0]
                previous = assignments[group]
                assignments[group] = split
                for other in languages[group]:
                    counts[other][previous] -= 1
                    counts[other][split] += 1
    return assignments
