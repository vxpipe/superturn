"""Seal the English Basis pilot census without downloading audio."""

import json
import os
import resource
from collections import Counter, defaultdict
from pathlib import Path

from .ami_calibration import _write_json
from .basis_split import inventory, summarize
from .basis_stage0 import CUT_ROLES, joint_cut, recording_seconds
from .basis_stage0_yield import json_lines
from .progress import Progress
from .sources import data_root, load_sources

FIT_AUDIO_CEILING = 5_000_000_000
PAUSE_OPTIONS = {"holding": "hold", "normal_gap": "finished", "awkward_gap": "finished",
                 "finished": "finished"}


def vote_census(rows, roles, eligible_calls):
    summary = defaultdict(Counter)
    priorities = defaultdict(Counter)
    annotated = set()
    for row in rows:
        call = row["conversation_id"]
        if call not in eligible_calls:
            raise ValueError("a judgment references a call missing from the English inventory")
        role = roles.get(call, "excluded")
        annotated.add(call)
        question = row["question"]
        summary[role][question] += 1
        if question != "pause":
            continue
        label = PAUSE_OPTIONS.get(row["label"])
        if label is None:
            summary[role]["pause_other"] += 1
            continue
        summary[role][f"pause_{label}"] += 1
        votes = sum(item["count"] for item in row["votes"])
        supporting = sum(item["count"] for item in row["votes"]
                         if PAUSE_OPTIONS.get(item["option"]) == label)
        if votes >= 4 and supporting / votes >= .8:
            summary[role][f"consensus_{label}"] += 1
            priorities[call][label] += 1
    return summary, priorities, annotated


def run(root):
    from huggingface_hub import HfApi, HfFileSystem

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Basis processing stays on Vast")
    cgroup = Path("/sys/fs/cgroup")
    if any((cgroup / key).read_text().strip() != "0" for key in ("memory.swap.current", "memory.swap.max")):
        raise ValueError("Basis metadata processing requires disabled swap")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    source = load_sources()["basis_conversations"]
    repo, revision = source.get("repo"), source.get("revision")
    token = os.environ["HF_TOKEN"]
    entries = defaultdict(list)
    api = HfApi(token=token)
    for entry in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True):
        parts = entry.path.split("/")
        if len(parts) == 4 and parts[1] == "audio" and parts[3].endswith(".flac"):
            entries[parts[0]].append(entry)
    calls = {language: inventory(rows, language) for language, rows in sorted(entries.items())}
    duration, recording_bytes = recording_seconds(repo, revision, token)
    receipt_dir = root / "reports"
    prior = json.loads((receipt_dir / "basis-stage0-joint-cut-40.json").read_text())
    membership = {}
    split = joint_cut(calls, prior["trials_per_language"], duration,
                      membership_out=membership, held_target=prior["minimum_held_roster"])
    if split["membership_sha256"] != prior["membership_sha256"]:
        raise ValueError("the existing four-role split did not reproduce")
    roles = {call: role for role, batch in membership["en"].items() for call in batch}
    english = {call.name: call for call in calls["en"]}
    fs = HfFileSystem(token=token)
    path = f"datasets/{repo}@{revision}/en/human_judgments.jsonl"
    size = fs.info(path)["size"]
    if not 0 < size < 3_000_000:
        raise ValueError("the English judgment file exceeds its metadata cap")
    meter = Progress("basis-english-judgments", size, "B")
    with fs.open(path, "rb") as handle:
        summary, priorities, annotated = vote_census(json_lines(handle, meter), roles, english)
    if size != meter.done:
        raise ValueError("the English judgment file differs from the pinned size")
    meter.finish()
    excluded = [call for name, call in english.items() if name not in roles]
    fit_annotated = [english[call] for call in membership["en"]["fit"] if priorities[call]]
    all_fit_annotated = [english[call] for call in membership["en"]["fit"] if call in annotated]
    projected = {
        "fit_human_annotated_calls": len(all_fit_annotated),
        "fit_human_annotated_whole_track_bytes": sum(call.size for call in all_fit_annotated),
        "fit_consensus_annotated_calls": len(fit_annotated),
        "fit_consensus_annotated_whole_track_bytes": sum(call.size for call in fit_annotated),
        "fit_audio_ceiling_bytes": FIT_AUDIO_CEILING,
    }
    result = {
        "scope": "English metadata census, existing calibration and test stay sealed",
        "revision": revision,
        "membership_sha256": split["membership_sha256"],
        "original": summarize(english.values()),
        "roles": split["languages"]["en"]["roles"],
        "excluded": {**summarize(excluded),
                     "recorded_seconds": round(sum(duration["en", call.name] for call in excluded), 3)},
        "judgments_by_role": {role: dict(summary[role]) for role in (*CUT_ROLES, "excluded")},
        "projection": projected,
        "metadata_bytes": recording_bytes + size,
        "audio_bytes_downloaded": 0,
    }
    _write_json(receipt_dir / "basis-english-pilot-census.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(data_root())
