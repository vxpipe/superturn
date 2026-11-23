"""Estimate pause capacity using publisher ASR timestamps, without audio."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .ami_calibration import _write_json
from .progress import Progress
from .yodas_mono_diarization import disjoint_bouts

BYTE_LIMIT = 540_000_000


def count_gaps(spans):
    counts, contributors = Counter(), defaultdict(set)
    bouts = list(disjoint_bouts(sorted(spans)))
    for (_, last), (first, _) in zip(bouts, bouts[1:]):
        if len(last) != 1 or len(first) != 1:
            continue
        _, edge, focal = last[0]
        onset, end, following = first[0]
        if edge < 3_000 or onset - edge < 300:
            continue
        if focal != following and end - onset < 1_000:
            counts["short_other"] += 1
            continue
        label = "hold" if focal == following else "shift"
        counts[label] += 1
        contributors[label].add(focal)
    return counts, contributors


def json_lines(handle, meter):
    remainder = b""
    while chunk := handle.read(8 * 1024 * 1024):
        meter.advance(len(chunk))
        lines, separator, tail = (remainder + chunk).rpartition(b"\n")
        remainder = tail
        if separator:
            for line in lines.splitlines():
                yield json.loads(line)
    if remainder:
        yield json.loads(remainder)


def fit_vote_counts(fs, prefix, membership):
    """Census publisher pause votes on fit calls only."""
    finished = {"normal_gap", "awkward_gap", "finished"}
    report = {}
    for language, roles in membership.items():
        fit_calls = set(roles["fit"])
        path = f"{prefix}/{language}/human_judgments.jsonl"
        if not fs.exists(path):
            continue
        size = fs.info(path)["size"]
        if not 0 < size < 3_000_000:
            raise ValueError("the Basis voting inventory exceeded its metadata cap")
        counts = Counter()
        with fs.open(path, "rb") as handle:
            data = handle.read(size + 1)
        if len(data) != size:
            raise ValueError("the Basis voting inventory did not match its listing")
        for line in data.splitlines():
            row = json.loads(line)
            if row["question"] != "pause" or row["conversation_id"] not in fit_calls:
                continue
            label = row["label"]
            options = {"holding"} if label == "holding" else finished if label in finished else set()
            votes = sum(item["count"] for item in row["votes"])
            supporting = sum(item["count"] for item in row["votes"] if item["option"] in options)
            if votes >= 4 and supporting / votes >= .8:
                counts["holding" if label == "holding" else "perceived_finished"] += 1
        report[language] = dict(counts)
    return report


def estimate_yield(root, source, membership, digest, *, suffix=""):
    from huggingface_hub import HfFileSystem

    cgroup = Path("/sys/fs/cgroup")
    if any((cgroup / name).read_text().strip() != "0"
           for name in ("memory.swap.max", "memory.swap.current")):
        raise ValueError("Basis metadata processing requires swap disabled")
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    selected = {call: (language, role) for language, roles in membership.items()
                for role, calls in roles.items() for call in calls}
    token = os.environ["HF_TOKEN"]
    fs = HfFileSystem(token=token)
    repo, revision = source.get("repo"), source.get("revision")
    prefix = f"datasets/{repo}@{revision}"
    path = f"{prefix}/manifests/lhotse/supervisions.jsonl"
    size = fs.info(path)["size"]
    if not 0 < size <= BYTE_LIMIT:
        raise ValueError("Basis timing manifest exceeded its transfer cap")
    streams = defaultdict(list)
    observed = set()
    meter = Progress("basis-stage0-timing-capacity", size, "B")
    try:
        with fs.open(path, "rb", block_size=8 * 1024 * 1024) as handle:
            for row in json_lines(handle, meter):
                segment = row["recording_id"]
                call, _, _ = segment.rpartition("_")
                if call not in selected:
                    continue
                observed.add(call)
                start = round(row["start"] * 1_000)
                stop = round((row["start"] + row["duration"]) * 1_000)
                if 0 <= start < stop:
                    streams[segment].append((start, stop, row["speaker"]))
        if meter.done != size:
            raise ValueError("Basis timing manifest length did not match the pinned file")
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    assigned = defaultdict(Counter)
    speakers = defaultdict(lambda: defaultdict(set))
    for segment, spans in streams.items():
        key = selected[segment.rpartition("_")[0]]
        counts, contributors = count_gaps(spans)
        assigned[key].update(counts)
        for label, people in contributors.items():
            speakers[key][label].update(people)
    output = {"scope": "unverified metadata capacity from uncorrected ASR timestamps",
              "membership_sha256": digest, "source_bytes": size,
              "publisher_high_consensus_pause_votes_fit": fit_vote_counts(fs, prefix, membership),
              "languages": {}}
    for language, roles in membership.items():
        output["languages"][language] = {}
        for role, calls in roles.items():
            key = language, role
            output["languages"][language][role] = {
                "calls_with_timing": len(observed.intersection(calls)),
                "calls": len(calls),
                "candidate_counts": dict(assigned[key]),
                "candidate_focal_total": len(set().union(*speakers[key].values())),
                "candidate_focal_speakers": {label: len(people) for label, people in speakers[key].items()},
            }
    _write_json(root / "reports" / f"basis-stage0-timing-capacity{suffix}.json", output)
    print(json.dumps(output, indent=2))
