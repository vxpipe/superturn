"""Join NSC turn metadata to the mirrored participant rosters without reading audio."""

from collections import Counter, defaultdict
import csv
import json
import re

import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .dense_teacher import file_digest
from .sources import data_root, load_sources
from .splits import assign, speaker_groups


REPORT = "nsc-turn-provenance"
ROSTERS = "nsc-speaker-rosters"
PARTS = (4, 5, 6)
CALL = re.compile(r"SE([0-9]+)(?:-Pt([A-Za-z0-9]+))?")
SPEAKER = re.compile(r"SP([0-9]+)")
FIELDS = ("call_id", "speaker", "split", "language", "eot_kind", "eot_len", "response_dur", "n_holds",
          "response_overlap", "capped_at_response", "src_sr")
SHA256 = {
    4: "c89956feac2623d60255ad0701d3d0f5aa6e30ae4802f3bcfaa210eb04334dce",
    5: "e143a66ff02beabb1945f555cac340448306828f16890e833faf46e45d96b735",
    6: "1799649c2df5bc9e426140896039ba30f4c093218a7d501b6a6a8c3a949cea4d",
}


def roster_id(value):
    value = value.strip()
    if not value.isdecimal():
        raise ValueError("invalid source roster key")
    return str(int(value))


def read_roster(path, speaker_column, *, partner_column=None):
    sessions = defaultdict(set)
    links = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            session, speaker = (roster_id(row[key]) for key in ("Session ID", speaker_column))
            sessions[session].add(speaker)
            if partner_column:
                links[session].add((speaker, roster_id(row[partner_column])))
    return dict(sessions), links


def part_rosters(directory):
    part4, links = read_roster(directory / "Part 4.csv", "Speaker ID", partner_column="Partner ID")
    reciprocated = {session for session, people in part4.items()
                    if len(people) == 2 and len(links[session]) == 2
                    and all(left != right and (right, left) in links[session] for left, right in links[session])}
    debate, _ = read_roster(directory / "Part 5 (Debate).csv", "Person ID")
    finance, _ = read_roster(directory / "Part 5 (Finance).csv", "Person ID")
    main, _ = read_roster(directory / "Part 5 (Main).csv", "Person ID")
    return part4, reciprocated, debate, finance, main


def read_part(path):
    calls = {}
    for row in pq.read_table(path, columns=FIELDS).to_pylist():
        call, speaker = row["call_id"], row["speaker"]
        match, identity = CALL.fullmatch(call), SPEAKER.fullmatch(speaker)
        if not match or not identity:
            raise ValueError("a sidecar key does not match the pinned recording schema")
        call_data = calls.setdefault(call, {"session": roster_id(match[1]), "topic": match[2], "people": set(),
                                           "split": set(), "counts": defaultdict(Counter), "rates": Counter()})
        call_data["people"].add(roster_id(identity[1]))
        call_data["split"].add(row["split"])
        call_data["rates"][row["src_sr"]] += 1
        counts = call_data["counts"][row["language"]]
        counts["turns"] += 1
        counts[f'end_{row["eot_kind"]}'] += 1
        counts["publisher_hold_spans"] += row["n_holds"]
        if (row["eot_kind"] == "response" and row["response_dur"] is not None
                and row["response_dur"] >= .5 and row["eot_len"] >= .3):
            counts["response_with_300ms_tail"] += 1
            counts["response_with_300ms_tail_and_known_overlap"] += row["response_overlap"] is not None
            counts["response_with_300ms_tail_marked_capped"] += row["capped_at_response"] is True
    if any(len(call["split"]) != 1 for call in calls.values()):
        raise ValueError("a publisher split cuts a single call across sets")
    return calls


def roster_evidence(part, calls, rosters):
    part4, reciprocated, debate, finance, main = rosters
    matched = {}
    excluded = Counter()
    for name, call in calls.items():
        session, topic, visible = call["session"], call["topic"], call["people"]
        if part == 4:
            named = part4.get(session)
            if named and session not in reciprocated:
                excluded["nonreciprocal_or_ambiguous_roster"] += 1
                continue
        elif part == 5:
            if topic not in {"01", "02", "03", "Fin", "Pos", "Neg"}:
                excluded["unknown_recording_topic"] += 1
                continue
            roster = debate if topic.isdecimal() else finance
            named = roster.get(session)
            if named and named != main.get(session):
                excluded["main_and_topic_rosters_disagree"] += 1
                continue
        else:
            excluded["no_independent_part6_person_roster"] += 1
            continue
        if not named:
            excluded["session_absent_from_roster"] += 1
        elif len(named) != 2 or not visible <= named:
            excluded["ambiguous_pair_or_unmatched_speaker"] += 1
        else:
            matched[name] = named
    return matched, dict(excluded)


def summarize(calls, matched):
    source_people = {p for call in calls.values() for p in call["people"]}
    split_people = defaultdict(set)
    for name, call in calls.items():
        split_people[next(iter(call["split"]))].update(matched.get(name, set()))
    groups = speaker_groups(matched.items())
    group_people, languages, proposed = defaultdict(set), defaultdict(Counter), defaultdict(Counter)
    native_rates = Counter()
    for name, people in matched.items():
        group_people[groups[name]].update(people)
        proposed_split = assign(REPORT, groups[name], "train-conversation")
        native_rates.update(calls[name]["rates"])
        for language, reason_counts in calls[name]["counts"].items():
            languages[language].update(reason_counts)
            proposed[proposed_split].update(reason_counts)
    people = set().union(*matched.values())
    largest = max(map(len, group_people.values()), default=0)
    return {"sidecar_calls": len(calls), "sidecar_speaker_keys": len(source_people),
            "roster_verified_calls": len(matched), "roster_verified_people": len(people),
            "two_visible_channel_calls": sum(len(calls[name]["people"]) == 2 for name in matched),
            "publisher_cross_split_people": sum(sum(p in group for group in split_people.values()) > 1 for p in people),
            "components": len(group_people), "largest_component_people": largest,
            "largest_component_fraction": largest / len(people) if people else None,
            "breadth_pass": len(people) >= 150 and largest / len(people) <= .5,
            "matched_native_sample_rates": dict(native_rates), "by_publisher_language": dict(languages),
            "proposed_component_split": dict(proposed)}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the metadata census runs on the staged Vast machine")
    registry = load_sources()
    paths = {key: root / "raw" / key for key in (REPORT, ROSTERS)}
    inputs = {}
    for key, directory in paths.items():
        source = registry[key]
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        if manifest["revision"] != source.get("revision") or manifest["licence"] != source.licence:
            raise ValueError("the acquisition manifest does not match the registered revision and licence")
        files = {name: file_digest(directory / name) for name in source.get("files")}
        if sum((directory / name).stat().st_size for name in files) != source.get("max_total_bytes"):
            raise ValueError("the census input byte total changed")
        inputs[key] = {"revision": manifest["revision"], "files": files}
    rosters = part_rosters(paths[ROSTERS] / "SpeakerMetaData")
    combined_calls, combined_people, results = {}, {}, {}
    cohorts = {}
    for part in PARTS:
        base = f"meta/dialogue/imda_part{part}"
        if inputs[REPORT]["files"][f"{base}.parquet"] != SHA256[part]:
            raise ValueError("the sidecar differs from its upstream SHA-256")
        calls = read_part(paths[REPORT] / f"{base}.parquet")
        matched, excluded = roster_evidence(part, calls, rosters)
        cohorts[part] = set().union(*matched.values())
        results[part] = {"exclusions": excluded, **summarize(calls, matched)}
        combined_calls.update({f"{part}/{call}": row for call, row in calls.items()})
        combined_people.update({f"{part}/{call}": people for call, people in matched.items()})
    result = {"scope": "metadata only; timing candidates need native-channel validation",
              "identity_scope": "exact original session and numeric person-code roster joins",
              "language_scope": "publisher code; Part 4 is code-switched and requires acoustic language confirmation",
              "hold_scope": "all reported hold spans; sidecars do not include per-hold durations",
              "inputs": inputs, "parts": results, "joined_parts_shared_codes": len(cohorts[4] & cohorts[5]),
              "joint": summarize(combined_calls, combined_people)}
    _write_json(root / "reports" / f"{REPORT}.json", result)
    keys = ("roster_verified_calls", "roster_verified_people", "publisher_cross_split_people",
            "largest_component_fraction", "breadth_pass")
    print(json.dumps({part: {key: row[key] for key in keys} for part, row in results.items()}))


if __name__ == "__main__":
    run(data_root())
