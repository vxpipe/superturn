"""Check four-role feasibility without breaking Basis speaker components."""

import json
import os
from collections import defaultdict

from .ami_calibration import _write_json
from .basis_split import inventory, voices
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


def run(root):
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
    run(data_root())
