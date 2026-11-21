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
from collections import Counter, defaultdict
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
    groups = {}
    for members_of_group in members.values():
        key = min(members_of_group)
        groups.update(dict.fromkeys(members_of_group, key))
    return groups


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


DURATION_BANDS = ("under_1200_ms", "1200_to_2699_ms", "2700_to_5999_ms", "6000_ms_or_more")


def duration_band(milliseconds: int | None) -> str:
    """Bin available turn history as measured at a pause."""
    if milliseconds is None:
        return "unknown"
    if type(milliseconds) is not int or milliseconds < 0:
        raise ValueError("prior_turn_ms must be a nonnegative integer or null")
    return DURATION_BANDS[sum(milliseconds >= edge for edge in (1200, 2700, 6000))]


def stratified_duration_groups(
    source: str, role: str, rows: Iterable[tuple[str, str, int | None]], *, fixed: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Allocate whole groups by joint language and duration pause counts."""
    volumes = defaultdict(Counter)
    for group, language, prior_turn_ms in rows:
        volumes[group][language, duration_band(prior_turn_ms)] += 1
        volumes[group][language, "all_durations"] += 1
        volumes[group]["", "all_durations"] += 1

    totals, group_totals = Counter(), Counter()
    for frequencies in volumes.values():
        totals.update(frequencies)
        group_totals.update(frequencies.keys())

    pause_counts = defaultdict(Counter)
    group_counts = defaultdict(Counter)
    remaining = group_totals.copy()
    assignments = dict(fixed or {})
    order = sorted(volumes.keys() - assignments.keys(), key=lambda group: (
        min(group_totals[cell] for cell in volumes[group]),
        -sum(volumes[group].values()),
        hashlib.sha256(f"superturn-duration-split-v1\0{source}\0{group}".encode()).digest(),
    ))

    def place(group, split, direction):
        for cell, weight in volumes[group].items():
            pause_counts[cell][split] += direction * weight
            group_counts[cell][split] += direction

    shares = dict(SPLITS)
    if any(group not in volumes or split not in shares or (role == "test" and split != "test")
           for group, split in assignments.items()):
        raise ValueError("invalid fixed group or split")
    for group, split in assignments.items():
        place(group, split, 1)
        remaining.subtract(volumes[group].keys())

    def cost(cell, split, delta):
        total = totals[cell]
        before = pause_counts[cell][split] / total - shares[split] / 100
        after = before + delta / total
        weight = 1 if cell[1] != "all_durations" else .1 if cell[0] else .01
        return weight * (after * after - before * before)

    for group in order:
        cells = sorted(volumes[group].items())

        def priority(split):
            coverage, increase = 0.0, 0.0
            for cell, weight in cells:
                if group_totals[cell] >= len(SPLITS):
                    missing = sum(group_counts[cell][name] == 0 for name in shares)
                    if remaining[cell] <= missing and group_counts[cell][split] == 0:
                        coverage += 1 / group_totals[cell]
                increase += cost(cell, split, weight)
            return -coverage, increase

        split = "test" if role == "test" else min(shares, key=priority)
        assignments[group] = split
        place(group, split, 1)
        remaining.subtract(volumes[group].keys())

    if role != "test":
        # Fill uncovered cells by moving a group only if its other cells
        # retain coverage. Counts can still be constrained by large groups.
        for cell in sorted(totals, key=lambda key: (group_totals[key], key)):
            if group_totals[cell] < len(SPLITS):
                continue
            for split in shares:
                if group_counts[cell][split]:
                    continue
                candidates = [group for group in order if cell in volumes[group] and all(
                    group_totals[tag] < len(SPLITS) or group_counts[tag][assignments[group]] > 1
                    for tag in volumes[group]
                )]
                if candidates:
                    def move_cost(group):
                        previous = assignments[group]
                        return sum(cost(tag, split, weight) + cost(tag, previous, -weight)
                                   for tag, weight in sorted(volumes[group].items()))

                    group = min(candidates, key=move_cost)
                    place(group, assignments[group], -1)
                    assignments[group] = split
                    place(group, split, 1)
                elif len(order) <= 48:
                    # A donor's previous split can sometimes be replenished
                    # by one other group in the same small planning problem.
                    for donor in order:
                        if cell not in volumes[donor]:
                            continue
                        previous = assignments[donor]
                        donors_last_tags = {tag for tag in volumes[donor]
                                            if group_totals[tag] >= len(SPLITS) and group_counts[tag][previous] == 1}
                        replacement = next((other for other in order if assignments[other] != previous
                                            and donors_last_tags <= volumes[other].keys()
                                            and all(group_totals[tag] < len(SPLITS)
                                                    or group_counts[tag][assignments[other]] > 1
                                                    or (assignments[other] == split and tag in volumes[donor])
                                                    for tag in volumes[other])), None)
                        if replacement is not None:
                            place(donor, previous, -1)
                            place(replacement, assignments[replacement], -1)
                            assignments[donor], assignments[replacement] = split, previous
                            place(donor, split, 1)
                            place(replacement, previous, 1)
                            break

    summary = []
    for cell, total in sorted(totals.items()):
        language, band = cell
        groups = group_totals[cell]
        summary.append({"language": language or None, "duration_band": band, "pauses": total,
                        "independent_groups": groups,
                        "by_split": {split: {"pauses": pause_counts[cell][split], "groups": group_counts[cell][split]}
                                     for split in shares},
                        "empty_splits": [split for split in shares if not group_counts[cell][split]]})
    return assignments, summary
