"""Verify the private cohort and immutable calibration receipt for the release benchmark."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

DEADLINE = 300
BLOCKED_IDS = {"", "unknown", "und", "anonymous", "none", "null", "n/a"}
CONTRACT = "superturn-telephone-ba-v1"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_json(path, expected):
    with Path(path).open("rb") as handle:
        encoded = handle.read()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("sealed release receipt digest changed")
    return json.loads(encoded)


def write_once(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", dir=Path(path).parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if Path(path).read_text() != text:
            raise ValueError("an incompatible release attempt already exists") from None
    finally:
        temporary.unlink()


def identities(values):
    if (not isinstance(values, list) or any(not isinstance(x, str) or x != x.strip()
            or x.casefold() in BLOCKED_IDS for x in values) or len(values) != len(set(values))):
        raise ValueError("speaker identities must be globally comparable and fully resolved")
    return set(values)


def validate_plan(plan):
    languages = plan["languages"]
    if (plan["contract"] != CONTRACT or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", plan["panel"])
            or not isinstance(languages, list) or not languages
            or any(not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)*", x) for x in languages)
            or len(set(languages)) != len(languages) or not plan["identity_namespace"]
            or plan["interface"] not in ("audio", "audio-text")):
        raise ValueError("invalid release protocol")
    for key in ("artifact_sha256", "fit_roster_sha256", "calibration_manifest_sha256", "test_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", plan[key]):
            raise ValueError("release protocol needs byte digests for all frozen inputs")
    if plan["calibration_manifest_sha256"] == plan["test_manifest_sha256"]:
        raise ValueError("calibration and test need different memberships")


def validate_manifest(manifest, plan, role):
    if (manifest["role"] != role or manifest["panel"] != plan["panel"]
            or manifest["identity_namespace"] != plan["identity_namespace"]
            or manifest["decision_ms"] != DEADLINE or manifest["condition"] != "telephone"
            or not manifest["adjudication_receipt"]):
        raise ValueError("wrong cohort protocol, role or telephone condition")
    recordings = {call: identities(people) for call, people in manifest["participants"].items()}
    if not recordings or any(not call or len(people) < 2 for call, people in recordings.items()):
        raise ValueError("each call needs a full resolved caller and counterpart roster")
    seen, positions, pcm, presentations = set(), set(), set(), defaultdict(list)
    for index, row in enumerate(manifest["pauses"]):
        pause, call, edge = (row[key] for key in ("pause", "call", "edge_sample"))
        position = call, row["speaker"], edge
        rate, quiet = row["source_rate"], row["quiet_samples"]
        if (not isinstance(pause, str) or not pause or pause in seen or position in positions
                or row["speaker"] not in recordings.get(call, set()) or type(edge) is not int or edge < 0
                or type(rate) is not int or rate <= 0 or type(quiet) is not int or quiet * 1000 < rate * DEADLINE
                or row["label"] not in ("hold", "shift") or row["language"] not in plan["languages"]
                or not re.fullmatch(r"[0-9a-f]{64}", row["pcm_sha256"]) or row["pcm_sha256"] in pcm):
            raise ValueError("duplicate, unresolved, or invalid release pause")
        seen.add(pause)
        positions.add(position)
        pcm.add(row["pcm_sha256"])
        presentations[row["language"]].append(index)
    if set(presentations) != set(plan["languages"]):
        raise ValueError(f"{role} is missing a pre-registered supported language")
    for indexes in presentations.values():
        if {manifest["pauses"][index]["label"] for index in indexes} != {"hold", "shift"}:
            raise ValueError(f"{role} needs both labels in every supported language")
    return presentations, (set.union(*recordings.values()), set(recordings), seen, pcm)


def ensure_disjoint(fit, calibration, test=None):
    cohorts = (calibration,) if test is None else (calibration, test)
    if any(fit & data[0] for data in cohorts):
        raise ValueError("fit and held-out cohorts share a person")
    if test is not None and any(left & right for left, right in zip(calibration, test, strict=True)):
        raise ValueError("calibration and test share a person, call, pause or rendered input")
