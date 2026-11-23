"""Seal the English Basis pilot census without downloading audio."""

import argparse
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
from .yodas_mono_diarization import disjoint_bouts

FIT_AUDIO_CEILING = 5_000_000_000
PAUSE_OPTIONS = {"holding": "hold", "normal_gap": "finished", "awkward_gap": "finished",
                 "finished": "finished"}


def decision_window_counts(rows):
    counts = defaultdict(Counter)
    for row in rows:
        category = PAUSE_OPTIONS.get(row["label"])
        if row["question"] != "pause" or category is None:
            continue
        width_ms = round((row["end"] - row["start"]) * 1000)
        band = "under_300_ms" if width_ms < 300 else "300_to_800_ms" if width_ms <= 800 else "over_800_ms"
        counts[category][band] += 1
    return {category: dict(bands) for category, bands in counts.items()}


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


def observed_word_gaps(segment):
    words = sorted((word["start"], word["end"], utterance["speaker"])
                   for utterance in segment["transcript"] for word in utterance["words"]
                   if word["start"] < word["end"])
    bouts = list(disjoint_bouts(words))
    return [(last[0][1], first[0][0], last[0][2], first[0][2])
            for (_, last), (first, _) in zip(bouts, bouts[1:])
            if len(last) == len(first) == 1 and first[0][0] - last[0][1] >= .3]


def timing_support(segment, judgments):
    """Check whether released word times independently support the judgment edges."""
    gaps = observed_word_gaps(segment)
    results = Counter()
    for judgment in judgments:
        category = PAUSE_OPTIONS[judgment["label"]]
        focal, = judgment["speakers"]
        matches = [(focal == next_voice) for stop, onset, voice, next_voice in gaps
                   if voice == focal and abs(judgment["start"] - stop) <= .35
                   and abs(judgment["end"] - onset) <= .35]
        if len(matches) != 1:
            status = "missing" if not matches else "ambiguous"
        else:
            status = "supported" if matches[0] == (category == "hold") else "role_mismatch"
        results[f"{category}_{status}"] += 1
    return results


def inspect_fit_timings(fs, prefix, fit_judgments, audio_paths, *, retain=False):
    sidecar = f"{prefix}/en/segments.jsonl"
    size = fs.info(sidecar)["size"]
    if not 0 < size <= 130_000_000:
        raise ValueError("the English word timing file exceeds its metadata cap")
    meter = Progress("basis-english-word-time-preflight", size, "B")
    supported, seen, audio_bytes = Counter(), set(), 0
    first_track, segments = None, []
    with fs.open(sidecar, "rb", block_size=8 * 1024 * 1024) as handle:
        for row in json_lines(handle, meter):
            ident = row["segment_id"]
            if ident not in fit_judgments:
                continue
            if ident in seen:
                raise ValueError("a fit pause segment appears twice")
            seen.add(ident)
            supported.update(timing_support(row, fit_judgments[ident]))
            if retain:
                segments.append(row)
            for path in row["tracks"].values():
                audio_bytes += audio_paths[path]
                first_track = first_track or path
    if meter.done != size or seen != set(fit_judgments):
        raise ValueError("the English word timing file did not cover the fit judgment list")
    meter.finish()
    return {"fit_pause_segments": len(seen), "fit_pause_segment_whole_track_bytes": audio_bytes,
            "word_timing_counts": dict(supported), "word_sidecar_bytes": size}, first_track, segments


def flac_seek_table(handle, size):
    """Read only FLAC metadata block headers without fetching audio frames."""
    if handle.read(4) != b"fLaC":
        raise ValueError("the selected Basis track does not have a native FLAC header")
    offset, requested, has_seek_table = 4, 4, False
    for _ in range(32):
        handle.seek(offset)
        header = handle.read(4)
        requested += 4
        if len(header) != 4:
            raise ValueError("the Basis FLAC metadata header is incomplete")
        block_type = header[0] & 127
        length = int.from_bytes(header[1:], "big")
        offset += 4 + length
        if offset >= size:
            raise ValueError("the Basis FLAC metadata overlaps the listed end")
        has_seek_table |= block_type == 3 and length >= 18
        if header[0] & 128:
            return {"sampled_track_has_seek_table": has_seek_table, "header_bytes_requested": requested}
    raise ValueError("the Basis FLAC metadata exceeded the bounded header walk")


def run(root, *, reference_audit=False, publisher_windows=False):
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
    prefix = f"datasets/{repo}@{revision}"
    fs = HfFileSystem(token=token)
    path = f"{prefix}/en/human_judgments.jsonl"
    size = fs.info(path)["size"]
    if not 0 < size < 3_000_000:
        raise ValueError("the English judgment file exceeds its metadata cap")
    meter = Progress("basis-english-judgments", size, "B")
    with fs.open(path, "rb") as handle:
        rows = list(json_lines(handle, meter))
    if size != meter.done:
        raise ValueError("the English judgment file differs from the pinned size")
    meter.finish()
    window_counts = decision_window_counts(rows)
    if publisher_windows:
        report = {"scope": "English publisher judgment interval widths",
                  "revision": revision, "decision_window_counts": window_counts,
                  "metadata_bytes": size}
        _write_json(root / "reports/basis-english-publisher-decision-windows.json", report)
        print(json.dumps(report, indent=2))
        return
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
    fit_judgments = defaultdict(list)
    summary, priorities, annotated = vote_census(rows, roles, english)
    for row in rows:
        if (roles.get(row["conversation_id"]) == "fit" and row["question"] == "pause"
                and row["label"] in PAUSE_OPTIONS and len(row.get("speakers", [])) == 1):
            fit_judgments[row["segment_id"]].append(row)
    excluded = [call for name, call in english.items() if name not in roles]
    fit_annotated = [english[call] for call in membership["en"]["fit"] if priorities[call]]
    all_fit_annotated = [english[call] for call in membership["en"]["fit"] if call in annotated]
    audio_sizes = {entry.path: entry.size for entry in entries["en"]}
    timing, first_track, segments = inspect_fit_timings(fs, prefix, fit_judgments, audio_sizes,
                                                       retain=reference_audit)
    if first_track:
        with fs.open(f"{prefix}/{first_track}", "rb", cache_type="none") as handle:
            timing.update(flac_seek_table(handle, audio_sizes[first_track]))
    projected = {
        "fit_human_annotated_calls": len(all_fit_annotated),
        "fit_human_annotated_whole_track_bytes": sum(call.size for call in all_fit_annotated),
        "fit_consensus_annotated_calls": len(fit_annotated),
        "fit_consensus_annotated_whole_track_bytes": sum(call.size for call in fit_annotated),
        "fit_audio_ceiling_bytes": FIT_AUDIO_CEILING,
        **timing,
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
        "publisher_decision_window_counts": window_counts,
        "projection": projected,
        "metadata_bytes": recording_bytes + size + timing["word_sidecar_bytes"],
        "audio_bytes_downloaded": 0,
    }
    _write_json(receipt_dir / "basis-english-pilot-census.json", result)
    print(json.dumps(result, indent=2))
    if reference_audit:
        from .basis_english_reference import run_reference_audit

        audit = run_reference_audit(root, fs, prefix, segments, fit_judgments,
                                    timing["fit_pause_segment_whole_track_bytes"], observed_word_gaps,
                                    PAUSE_OPTIONS)
        print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--reference-audit", action="store_true",
                        help="also run the bounded publisher-reference audio diagnostic")
    actions.add_argument("--publisher-windows", action="store_true",
                         help="count English judgment intervals by runtime silence band")
    args = parser.parse_args()
    run(data_root(), reference_audit=args.reference_audit, publisher_windows=args.publisher_windows)
