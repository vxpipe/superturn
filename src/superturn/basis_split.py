"""Plan a speaker-disjoint Basis split from track metadata only."""

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath

from .ami_calibration import _write_json
from .progress import Progress
from .shards import seed_for
from .sources import data_root, load_sources
from .splits import speaker_groups

ROLES = ("train", "calibration", "test")


@dataclass(frozen=True)
class Call:
    name: str
    speakers: frozenset[str]
    size: int


def inventory(entries, language):
    people = defaultdict(set)
    sizes = defaultdict(int)
    seen = set()
    for entry in entries:
        path = PurePosixPath(entry.path)
        if path.suffix != ".flac":
            continue
        if len(path.parts) != 4 or path.parts[:2] != (language, "audio") or entry.path in seen:
            raise ValueError("unexpected or repeated Basis track path")
        _, _, call, name = path.parts
        _, dot, speaker = name.removesuffix(".flac").rpartition(".")
        if not dot or not speaker.startswith("spk_") or not entry.size:
            raise ValueError("a listed track lacks a speaker ID or byte count")
        seen.add(entry.path)
        people[call].add(speaker)
        sizes[call] += entry.size
    if not people:
        raise ValueError(f"no conversations for {language}")
    return [Call(call, frozenset(voices), sizes[call]) for call, voices in sorted(people.items())]


def voices(calls):
    return frozenset(speaker for call in calls for speaker in call.speakers)


def reserve(calls, protected, held, target, rank, trial, role, independent):
    available = [call for call in calls if call.speakers.isdisjoint(protected | held)]
    covered, blocked = frozenset(), set().union(*(rank[speaker] for speaker in held))
    groups = []
    while len(covered) < target:
        proposals = [call for call in available if call.speakers - covered
                     and (not independent or call.speakers.isdisjoint(covered))]
        if not proposals:
            break

        def cost(call):
            added = call.speakers - covered
            cuts = len(set().union(*(rank[speaker] for speaker in added)) - blocked)
            return (cuts / min(len(added), target - len(covered)),
                    call.size / len(added), seed_for("basis-split", trial, role, call.name))

        proposals.sort(key=cost)
        # Deterministic multi-start explores the three cheapest alternatives.
        index = seed_for("basis-reserve", trial, role, len(groups)) % min(3, len(proposals))
        selected = proposals[index]
        blocked.update(*(rank[speaker] for speaker in selected.speakers))
        covered |= selected.speakers
        groups.append(selected.speakers)
    return (groups if independent else [covered]) if len(covered) >= target else None


def split(calls, target, trials, *, independent=False):
    indexes = defaultdict(set)
    for call in calls:
        for speaker in call.speakers:
            indexes[speaker].add(call.name)
    audit = independent_audit(calls)
    if len(audit) != 4:
        return None
    protected = voices(audit)
    candidates = []
    meter = Progress("basis-speaker-split", trials, "plans")
    try:
        for trial in range(trials):
            cal_groups = reserve(calls, protected, frozenset(), target, indexes, trial, "calibration", independent)
            if cal_groups:
                calibration = frozenset().union(*cal_groups)
                test_groups = reserve(calls, protected, calibration, target, indexes, trial, "test", independent)
            else:
                test_groups = None
            if test_groups:
                held = calibration.union(*test_groups)
                batches = {
                    "train": [call for call in calls if call.speakers.isdisjoint(held)],
                    "calibration": [call for call in calls if any(call.speakers <= group for group in cal_groups)],
                    "test": [call for call in calls if any(call.speakers <= group for group in test_groups)],
                }
                candidates.append((batches, audit, trial))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    if not candidates:
        return None
    return max(candidates, key=lambda result: (
        len(voices(result[0]["train"])),
        sum(len(result[0][role]) for role in ROLES),
        -sum(call.size for role in ("calibration", "test") for call in result[0][role]),
    ))


def independent_audit(calls):
    """Find four mutually voice-disjoint, training-only calls."""
    selected, covered = [], frozenset()
    for call in sorted(calls, key=lambda row: (
            row.size, seed_for("basis-audit", row.name))):
        if len(call.speakers) >= 2 and call.speakers.isdisjoint(covered):
            selected.append(call)
            covered |= call.speakers
        if len(selected) == 4:
            break
    return selected


def summarize(calls):
    groups = speaker_groups((call.name, call.speakers) for call in calls)
    component_sizes = Counter(groups.values())
    return {"calls": len(calls), "speakers": len(voices(calls)),
            "components": len(component_sizes), "largest_component_calls": max(component_sizes.values(), default=0),
            "whole_call_bytes": sum(call.size for call in calls)}


def run(root, language, target, trials, independent):
    from huggingface_hub import HfApi

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Basis processing stays on the campaign machine")
    source = load_sources()["basis_conversations"]
    revision = source.get("revision")
    api = HfApi(token=os.environ["HF_TOKEN"])
    entries = api.list_repo_tree(source.get("repo"), repo_type="dataset", revision=revision,
                                 path_in_repo=f"{language}/audio", recursive=True)
    calls = inventory(entries, language)
    report = {"revision": revision, "language": language, "source": summarize(calls),
              "target_speakers_per_held_role": target, "trials": trials,
              "independent_holdout_groups": independent}
    result = split(calls, target, trials, independent=independent)
    if result is not None:
        batches, audit, trial = result
        speaker_sets = [voices(batches[role]) for role in ROLES]
        if any(first & second for index, first in enumerate(speaker_sets)
               for second in speaker_sets[index + 1:]):
            raise AssertionError("a speaker crossed roles")
        retained = {call.name for batch in batches.values() for call in batch}
        if not {call.name for call in audit} <= {call.name for call in batches["train"]}:
            raise AssertionError("an audit call left the training role")
        report.update({"roles": {role: summarize(batch) for role, batch in batches.items()},
                       "excluded_bridging": summarize([call for call in calls if call.name not in retained]),
                       "training_only_audit": summarize(audit), "selected_trial": trial})
        membership = {role: sorted(call.name for call in batch) for role, batch in batches.items()}
        membership["training_only_audit"] = sorted(call.name for call in audit)
        report["membership_sha256"] = sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    suffix = "-independent" if independent else ""
    if target != 30:
        suffix += f"-{target}"
    _write_json(root / "reports" / f"basis-split-{language}{suffix}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--independent", action="store_true", help="reserve disjoint groups inside held-out roles")
    args = parser.parse_args()
    if not args.language.isascii() or not args.language.isalpha() or min(args.target, args.trials) < 1:
        parser.error("use an ASCII language code and positive target and trial counts")
    run(data_root(), args.language.lower(), args.target, args.trials, args.independent)
