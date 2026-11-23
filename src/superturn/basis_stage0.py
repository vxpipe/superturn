"""Check four-role feasibility without breaking Basis speaker components."""

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import product

from .ami_calibration import _write_json
from .basis_split import inventory, reserve, summarize, voices
from .progress import Progress
from .sources import data_root, load_sources
from .splits import speaker_groups


def four_roles_possible(components, minimum=30):
    """Use every component intact, with two populated fit-side roles."""
    states = {(False, False, 0, 0)}
    for size in components:
        states = {next_state for fit, dev, calibration, test in states for next_state in (
            (True, dev, calibration, test),
            (fit, True, calibration, test),
            (fit, dev, min(minimum, calibration + size), test),
            (fit, dev, calibration, min(minimum, test + size)),
        )}
    return (True, True, minimum, minimum) in states


CUT_ROLES = ("fit", "dev", "calibration", "test")


def recording_seconds(repo, revision, token):
    """Read the published recording manifest without audio or transcripts."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    path = f"datasets/{repo}@{revision}/manifests/lhotse/recordings.jsonl"
    size = fs.info(path)["size"]
    if not 0 < size <= 3_000_000:
        raise ValueError("Basis recording manifest exceeded its metadata byte cap")
    with fs.open(path, "rb") as source:
        payload = source.read(size + 1)
    if len(payload) != size:
        raise ValueError("Basis recording manifest length differs from the listing")
    durations = defaultdict(float)
    segments = set()
    for line in payload.splitlines():
        row = json.loads(line)
        paths = {tuple(source["source"].split("/")[:3]) for source in row["sources"]}
        if len(paths) != 1 or row["id"] in segments or row["sampling_rate"] != 48_000:
            raise ValueError("unexpected Basis recording manifest")
        (language, folder, call), = paths
        if folder != "audio" or not row["id"].startswith(f"{call}_") or row["num_samples"] <= 0:
            raise ValueError("Basis recording manifest does not identify its parent call")
        durations[language, call] += row["num_samples"] / row["sampling_rate"]
        segments.add(row["id"])
    return durations, size


def search_cut(rows, trials, *, held_target=30, dev_target=10):
    """Find candidates from metadata, with complete calls and four distinct roles."""
    if len(voices(rows)) < 2 * held_target + dev_target + 2:
        return None
    index = defaultdict(set)
    for call in rows:
        for person in call.speakers:
            index[person].add(call.name)
    best, best_rank = None, None
    for trial in range(trials):
        roster, held = {}, frozenset()
        for role, target in (("calibration", held_target), ("test", held_target), ("dev", dev_target)):
            groups = reserve(rows, frozenset(), held, target, index, trial, role,
                             independent=trial % 3 == 0)
            if groups is None:
                break
            roster[role] = frozenset().union(*groups)
            held |= roster[role]
        if len(roster) == 3:
            roster["fit"] = voices(rows) - held
            batches = {role: [call for call in rows if call.speakers <= roster[role]] for role in CUT_ROLES}
            fit = len(voices(batches["fit"]))
            if fit:
                rank = (fit, min(len(batches[role]) for role in CUT_ROLES[2:]),
                        sum(map(len, batches.values())))
                if best_rank is None or rank > best_rank:
                    best, best_rank = roster, rank
    return best


def candidate_rosters(calls, excluded, trials, held_target):
    meter = Progress("basis-stage0-joint-cut", len(calls), "languages")
    try:
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(search_cut, [row for row in rows if row.speakers.isdisjoint(excluded)],
                                trials, held_target=held_target): language for language, rows in calls.items()}
            result = {}
            for job in as_completed(jobs):
                result[jobs[job]] = job.result()
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return dict(sorted(result.items()))


def joint_cut(calls, trials, durations=None, *, membership_out=None, held_target=30):
    frequencies = Counter(person for rows in calls.values() for person in voices(rows))
    multilingual = {person for person, count in frequencies.items() if count > 1}
    names = [call.name for rows in calls.values() for call in rows]
    if len(names) != len(set(names)):
        raise ValueError("a Basis conversation spans multiple language directories")
    assignment, candidates = {}, set()
    for language, roster in candidate_rosters(calls, multilingual, trials, held_target).items():
        if roster is None:
            continue
        candidates.add(language)
        for role, people in roster.items():
            assignment.update(dict.fromkeys(people, role))

    # Give each shared voice a single role, favouring its existing neighbours.
    # Calls joining multiple roles are excluded in the final whole-call pass.
    shared_calls = [call for language in candidates for call in calls[language]
                    if call.speakers & multilingual]
    people = sorted(multilingual)

    def retained_roles(roles):
        retained = 0
        proposed = dict(zip(people, roles, strict=True))
        for call in shared_calls:
            neighbours = {assignment.get(speaker, proposed.get(speaker)) for speaker in call.speakers}
            retained += len(neighbours) == 1 and None not in neighbours
        return retained

    # The pinned release has only four multilingual people. Enumerate their
    # roles against the selected monolingual candidate rosters.
    if len(people) > 6:
        raise ValueError("the pinned Basis cross-language roster changed")
    proposed = max(product(CUT_ROLES, repeat=len(people)), key=retained_roles)
    assignment.update(zip(people, proposed, strict=True))

    roles = {role: set() for role in CUT_ROLES}
    membership, report = {}, {}
    for language, rows in calls.items():
        if language not in candidates:
            report[language] = {"provisional_candidate": False, "source": summarize(rows)}
            continue
        batches = {role: [] for role in CUT_ROLES}
        excluded = []
        for call in rows:
            assignments = {assignment.get(person) for person in call.speakers}
            if len(assignments) == 1 and None not in assignments:
                role, = assignments
                batches[role].append(call)
                roles[role].update(call.speakers)
            else:
                excluded.append(call)
        details = {role: summarize(batch) for role, batch in batches.items()}
        if durations is not None:
            for role, batch in batches.items():
                details[role]["recorded_seconds"] = round(sum(durations[language, call.name] for call in batch), 3)
        report[language] = {"provisional_candidate": True, "source": summarize(rows), "roles": details,
                            "excluded_calls": summarize(excluded)}
        membership[language] = {role: sorted(call.name for call in batch) for role, batch in batches.items()}
    if sum(map(len, roles.values())) != len(set().union(*roles.values())):
        raise AssertionError("a retained Basis person crossed roles")
    digest = sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    if membership_out is not None:
        membership_out.update(membership)
    return {"languages": report, "membership_sha256": digest, "shared_language_roster": len(multilingual),
            "joint_role_roster": {role: len(people) for role, people in roles.items()}}


def run(root, cut_trials=None, *, estimate=False, pilot=False, review_packet=False, held_target=30):
    from huggingface_hub import HfApi

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Basis processing stays on Vast")
    source = load_sources()["basis_conversations"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    files = defaultdict(list)
    revision = source.get("revision")
    for entry in api.list_repo_tree(source.get("repo"), repo_type="dataset", revision=revision, recursive=True):
        parts = entry.path.split("/")
        if len(parts) == 4 and parts[1] == "audio" and parts[3].endswith(".flac"):
            files[parts[0]].append(entry)
    calls = {language: inventory(entries, language) for language, entries in sorted(files.items())}
    if cut_trials is not None:
        durations, received = recording_seconds(source.get("repo"), revision, os.environ["HF_TOKEN"])
        if set(durations) != {(language, call.name) for language, rows in calls.items() for call in rows}:
            raise ValueError("Basis recording manifest does not cover the pinned file inventory")
        membership = {} if estimate or pilot or review_packet else None
        report = {"revision": revision, "scope": "joint metadata-only four-role candidate",
                  "minimum_held_roster": held_target, "provisional_dev_roster": 10,
                  "trials_per_language": cut_trials,
                  "recording_manifest_bytes": received,
                  **joint_cut(calls, cut_trials, durations, membership_out=membership, held_target=held_target)}
        suffix = "" if held_target == 30 else f"-{held_target}"
        receipt = root / "reports" / f"basis-stage0-joint-cut{suffix}.json"
        if pilot or review_packet:
            from .basis_stage1 import run_pilot

            prior = json.loads(receipt.read_text())
            if prior["membership_sha256"] != report["membership_sha256"]:
                raise ValueError("the Basis audit membership changed")
            run_pilot(root, source, membership, calls, report["membership_sha256"], review_packet=review_packet)
            return
        if estimate:
            from .basis_stage0_yield import estimate_yield

            if receipt.is_file():
                prior = json.loads(receipt.read_text())
                if report["membership_sha256"] != prior["membership_sha256"]:
                    raise ValueError("the Basis split did not reproduce its verified receipt")
            else:
                _write_json(receipt, report)
            estimate_yield(root, source, membership, report["membership_sha256"], suffix=suffix)
            return
        _write_json(receipt, report)
        print(json.dumps(report, indent=2))
        return
    languages_by_speaker = defaultdict(set)
    for language, rows in calls.items():
        for speaker in voices(rows):
            languages_by_speaker[speaker].add(language)
    graph = speaker_groups((f"{language}/{call.name}", call.speakers)
                           for language, rows in calls.items() for call in rows)
    report = {"revision": revision, "scope": "whole release-wide components preserved",
              "global_components": len(set(graph.values())),
              "speakers_in_multiple_languages": sum(len(group) > 1 for group in languages_by_speaker.values()),
              "minimum_held_speakers": 30, "languages": {}}
    meter = Progress("basis-stage0-components", len(calls), "languages")
    for language, rows in calls.items():
        by_component = defaultdict(list)
        for call in rows:
            by_component[graph[f"{language}/{call.name}"]].append(call)
        rosters = sorted((len(voices(component)) for component in by_component.values()), reverse=True)
        report["languages"][language] = {"calls": len(rows), "speakers": len(voices(rows)),
                                         "global_component_rosters": rosters,
                                         "four_roles_possible": four_roles_possible(rosters)}
        meter.advance(1)
    meter.finish()
    _write_json(root / "reports/basis-stage0-components.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-trials", type=int, help="search candidate splits allowing bridge-call exclusions")
    parser.add_argument("--held-target", type=int, default=30, help="reserve this many rostered held-out voices")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--estimate-yield", action="store_true", help="estimate paused turns on the verified split")
    action.add_argument("--fit-audit-pilot", action="store_true", help="audit a small fit-only set")
    action.add_argument("--fit-review-packet", action="store_true", help="prepare the blind fit-only listening packet")
    args = parser.parse_args()
    if args.cut_trials is not None and args.cut_trials < 1:
        parser.error("cut trials must be positive")
    if args.held_target < 30:
        parser.error("held-out roster must have at least 30 voices")
    if (args.estimate_yield or args.fit_audit_pilot or args.fit_review_packet) and args.cut_trials is None:
        parser.error("yield estimation and auditing require the verified cut's trial count")
    if (args.fit_audit_pilot or args.fit_review_packet) and args.held_target != 40:
        parser.error("the fit audit uses the 40-roster Stage 0 receipt")
    run(data_root(), args.cut_trials, estimate=args.estimate_yield,
        pilot=args.fit_audit_pilot, review_packet=args.fit_review_packet, held_target=args.held_target)
