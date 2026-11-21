"""Census staged telephone pauses for a training-only soft-target comparison."""

from collections import Counter, defaultdict
from hashlib import sha256
import json

import numpy as np
import torch

from .ami_calibration import _write_json
from .audio import RATE, silero
from .datahive_conversation_pilot import inventory, load_call
from .dense_teacher import file_digest
from .dral_label_pilot import isolation
from .harper_partial_stt import pinned_membership
from .harper_valley_audit import GAP, audit_call
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .sources import data_root

ROLES = ("caller", "agent")
MIN_HEAD_ROWS = 300


def overlaps(spans, start, end):
    return any(left < end and right > start for left, right in spans)


def harper(root, detect, meter):
    calls, _, audit_hash = pinned_split(root, "train")
    raw = root / "raw/harper-valley"
    original, membership_hash = pinned_membership(root, "train")
    by_id = {call["id"]: call for call in calls}
    previous = defaultdict(list)
    for _, call, edge, _, _ in original:
        previous[call].append(edge)
    fit_participants = [{by_id[call][role] for role in ROLES} for _, call, _, _, _ in original]
    retains, failures, candidates = [], Counter(), []
    for call in calls:
        identifier = call["id"]
        if any(not (raw / item["path"]).is_file() or (raw / item["path"]).stat().st_size != item["bytes"]
               for item in call["files"]):
            raise ValueError("a pinned Harper training call is incomplete")
        voices = {call[role] for role in ROLES}
        remaining = [index for index, pair in enumerate(fit_participants) if not (pair & voices)]
        retains.append({"rows": len(remaining), "speakers": len({original[i][-1] for i in remaining}),
                        "labels": dict(Counter(original[i][3] for i in remaining))})

        def observe(audio, origins, vad, _, omissions, common_start, common_end):
            activity = {role: [(span.start, span.end) for span in vad if span.speaker == role] for role in ROLES}
            omitted = [(span.start, span.end) for span in omissions]
            for role in ROLES:
                other = ROLES[1 - ROLES.index(role)]
                digest = sha256(audio[role].tobytes()).hexdigest()
                for _, edge in activity[role]:
                    if edge < common_start or edge + GAP > common_end or overlaps(activity[role], edge, edge + GAP):
                        continue
                    local = edge - origins[role]
                    candidates.append({"call": identifier, "role": role, "person": call[role],
                                       "edge_16khz": local, "audio_sha256": digest,
                                       "counterpart_speech_in_query": overlaps(activity[other], edge, edge + GAP),
                                       "unverified_words_in_query": overlaps(omitted, edge, edge + GAP),
                                       "near_original_caller_pause": role == "caller" and any(
                                           abs(local - old) <= 160 * RATE // 1000 for old in previous[identifier]),
                                       "purged_head_supported": len(remaining) >= MIN_HEAD_ROWS
                                       and set(retains[-1]["labels"]) == {"hold", "shift"}})

        counts = audit_call(raw, identifier, detect, recover_clock=True, observe=observe)
        if not counts["audited_calls"]:
            failures.update(counts)
        meter.advance(1)
    summary = {"audited_calls": len(calls) - sum(failures[name] for name in
               ("inconsistent_or_missing_channel_origin", "insufficient_shared_audio", "missing_word_timing", "low_audio_word_overlap")),
               "planned_calls": len(calls), "clock_screen_exclusions": dict(failures),
               "purged_fit_rows_min_max": [min(x["rows"] for x in retains), max(x["rows"] for x in retains)],
               "purged_fit_callers_min_max": [min(x["speakers"] for x in retains), max(x["speakers"] for x in retains)],
               "purged_fit_class_minima": {label: min(x["labels"].get(label, 0) for x in retains) for label in ("hold", "shift")}}
    for role in ROLES:
        selected = [row for row in candidates if row["role"] == role]
        admissible = [row for row in selected if row["purged_head_supported"] and not (
            row["counterpart_speech_in_query"] or row["unverified_words_in_query"] or row["near_original_caller_pause"])]
        summary[role] = {"local_pauses": len(selected), "new_quiet_pauses": len(admissible),
                         "new_quiet_calls": len({row["call"] for row in admissible}),
                         "new_quiet_people": len({row["person"] for row in admissible}),
                         "counterpart_active": sum(row["counterpart_speech_in_query"] for row in selected)}
    return candidates, {**summary, "clock_audit_sha256": audit_hash, "teacher_membership_sha256": membership_hash}


def datahive(root, detect, meter):
    source, folder, metadata, by_name = inventory(root)
    calls = []
    for filename, digest in sorted(source.get("audio_sha256").items()):
        tracks, receipt = load_call(folder, filename, digest, by_name)
        active = [detect(track) for track in tracks]
        screen = isolation(tracks, active)
        count = 0
        if screen["passed"]:
            for slot, spans in enumerate(active):
                for _, edge in spans:
                    lo = max(0, edge - RATE // 5)
                    if edge + GAP > tracks.shape[1] or any(overlaps(channel, edge, edge + GAP) for channel in active):
                        continue
                    peak = np.max(np.abs(tracks[:, lo:edge].astype(np.int32)), axis=1)
                    silence = np.max(np.abs(tracks[:, edge:edge + GAP].astype(np.int32)))
                    count += bool(peak[slot] >= max(128, 1.414 * peak[1 - slot]) and silence <= .5 * peak[slot])
        calls.append({"nominal_language": receipt["language"], "source_sha256": receipt["source_sha256"],
                      "isolation_passed": screen["passed"], "quiet_pause_candidates": count})
        meter.advance(1)
    return {"metadata_sha256": file_digest(metadata), "calls": calls}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the telephone inventory uses staged Vast audio")
    calls, _, _ = pinned_split(root, "train")
    detect = silero()
    meter = Progress("own-telephone-inventory", len(calls) + 6, "staged calls")
    try:
        candidates, harper_summary = harper(root, detect, meter)
        datahive_summary = datahive(root, detect, meter)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    output = root / "runs/own-telephone-inventory"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "harper-training-candidates.json"
    _write_json(artifact, candidates)
    report = {"scope": "training-side inventory of 300 ms acoustic queries, no labels or teacher scoring",
              "harper": harper_summary, "datahive": datahive_summary,
              "harper_candidates": {"path": artifact.relative_to(root).as_posix(), "sha256": file_digest(artifact)}}
    _write_json(root / "reports/own-telephone-inventory.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
