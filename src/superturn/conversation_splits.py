"""Plan private splits for newly labeled conversations."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .splits import speaker_groups, stratified_duration_groups


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty identifier")
    return value


def plan(rows, source, *, test_only=False):
    calls = defaultdict(set)
    customers = {}
    fixed_recordings = {}
    pauses = {}
    for row in rows:
        pause = _identifier(row["pause_id"], "pause_id")
        recording = _identifier(row["recording"], "recording")
        language = _identifier(row["language"], "language").lower()
        speakers = row["speakers"]
        if not isinstance(speakers, list) or not speakers:
            raise ValueError("each pause needs at least one known speaker")
        calls[recording].update("speaker:" + _identifier(speaker, "speaker") for speaker in speakers)
        customer = row.get("customer")
        if customer is not None:
            customer = _identifier(customer, "customer")
            if recording in customers and customers[recording] != customer:
                raise ValueError("a recording has conflicting customer IDs")
            customers[recording] = customer
            calls[recording].add("customer:" + customer)
        fixed = row.get("fixed_split")
        if fixed is not None:
            if fixed not in ("train", "validation", "test"):
                raise ValueError("invalid fixed_split")
            if recording in fixed_recordings and fixed_recordings[recording] != fixed:
                raise ValueError("a recording has conflicting fixed splits")
            fixed_recordings[recording] = fixed
        if pause in pauses:
            raise ValueError("duplicate pause ID")
        pauses[pause] = recording, language, row.get("prior_turn_ms")
    if not pauses:
        raise ValueError("the fresh cohort has no pauses")

    groups = speaker_groups(calls.items())
    fixed_groups = {}
    for recording, split in sorted(fixed_recordings.items()):
        group = groups[recording]
        if group in fixed_groups and fixed_groups[group] != split:
            raise ValueError("a speaker or customer component bridges different fixed splits")
        fixed_groups[group] = split
    entries = [(groups[recording], language, elapsed) for recording, language, elapsed in pauses.values()]
    role = "test" if test_only else "train-conversation"
    assignments, coverage = stratified_duration_groups(source, role, entries, fixed=fixed_groups)
    return {"algorithm": "superturn-duration-split-v1", "source": source,
            "recordings": {recording: assignments[groups[recording]] for recording in sorted(groups)},
            "coverage": coverage}


def write_plan(input_path: Path, output_path: Path, source: str, *, test_only=False):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("the split receipt must be separate from the input manifest")
    payload = input_path.read_bytes()
    data = plan((json.loads(line) for line in payload.splitlines() if line.strip()), source, test_only=test_only)
    data["input_sha256"] = hashlib.sha256(payload).hexdigest()
    encoded = (json.dumps(data, indent=2) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent) as handle:
        handle.write(encoded)
        handle.flush()
        try:
            os.link(handle.name, output_path)
        except FileExistsError:
            if output_path.read_bytes() != encoded:
                raise ValueError("split receipt already exists with a different plan")
    return {"algorithm": data["algorithm"], "source": source, "coverage": data["coverage"]}
