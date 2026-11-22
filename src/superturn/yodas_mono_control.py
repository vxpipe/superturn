"""Freeze verified HarperValley positive controls for a mixed-microphone label audit."""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .datahive_codec_control import observe_call
from .dense_teacher import file_digest
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .shards import seed_for
from .sources import data_root

JOB = "yodas-mono-control"
CALLS = 20


def choose(entries, labels):
    """Use one training call per caller and prefer calls with both reference classes."""
    chosen, voices = [], set()
    ranked = sorted(entries, key=lambda row: (-len(labels[row["id"]]), seed_for(JOB, row["id"])))
    for row in ranked:
        call, voice = row["id"], row["caller"]
        if labels[call] and voice not in voices:
            chosen.append(row)
            voices.add(voice)
        if len(chosen) == CALLS:
            break
    if len(chosen) != CALLS:
        raise ValueError("the archive does not support twenty distinct training-role callers")
    return chosen


def pcm_digest(pcm):
    return hashlib.sha256(np.ascontiguousarray(pcm, dtype="<i2").tobytes()).hexdigest()


def mono(tracks):
    """Mix both aligned inputs with fixed gains and integer arithmetic."""
    return (tracks.astype(np.int32).sum(axis=0) // 2).astype("<i2")


def run(root):
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("HarperValley recordings stay on Vast")
    torch.set_num_threads(2)
    entries, _, audit = pinned_split(root, "train")
    reports = root / "reports"
    members = json.loads((reports / "harper-history-replication-membership.json").read_text())
    if hashlib.sha256(json.dumps(members).encode()).hexdigest() != MEMBERSHIP:
        raise ValueError("the verified HarperValley membership changed")
    labels, per_call = defaultdict(set), defaultdict(set)
    for call, edge, label in members:
        labels[call].add(label)
        per_call[call].add((edge, label))
    cases = choose(entries, labels)
    detect = silero()
    rows, counts, next_counts, seconds = [], Counter(), Counter(), 0.
    meter = Progress("yodas-mono-control-cohort", len(cases), "calls")
    try:
        for case in cases:
            call = case["id"]
            context = {}
            tracks, positive = observe_call(root / "raw/harper-valley", call, per_call[call], detect,
                                            word_context=context)
            mixed = mono(tracks)
            reference = []
            for edge, onset, label in sorted(positive):
                next_bouts = [span for span in context["activity"] if span.start == onset]
                if len(next_bouts) != 1:
                    next_counts["ambiguous"] += 1
                    continue
                bout, = next_bouts
                counts[label] += 1
                role = "caller" if bout.speaker == "0" else "agent"
                next_counts[f"{label}_then_{role}"] += 1
                reference.append({"edge_sample": edge, "next_onset_sample": onset, "floor_label": label,
                                  "next_role": role, "next_bout_samples": bout.end - bout.start})
            if not reference:
                raise ValueError("a selected control call has no usable next-onset reference")
            seconds += len(mixed) / RATE
            rows.append({"call": call, "caller": case["caller"], "agent": case["agent"],
                         "samples": len(mixed), "source_pcm_sha256": [pcm_digest(track) for track in tracks],
                         "mono_pcm_sha256": pcm_digest(mixed), "reference": reference})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    if min(counts[label] for label in ("hold", "shift")) < 20:
        raise ValueError("the fixed control has fewer than twenty known pauses in a class")
    private = root / "runs" / JOB / "membership.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"scope": "HarperValley train roles only", "archive_membership_sha256": MEMBERSHIP,
               "audit_sha256": audit, "native_analysis_rate": RATE, "mix": "floor((caller + agent) / 2)",
               "calls": rows}
    if private.exists() and json.loads(private.read_text()) != json.loads(json.dumps(receipt)):
        raise ValueError("a different mono-control membership already exists")
    _write_json(private, receipt)
    report = {"scope": "known-answer control geometry only; diarization and ASR have not run",
              "calls": len(rows), "unique_callers": len({row["caller"] for row in rows}),
              "unique_agents": len({row["agent"] for row in rows}), "seconds": seconds,
              "reference_counts": dict(counts), "next_speaker_by_floor_label": dict(next_counts),
              "cohort_sha256": file_digest(private),
              "harper_audit_sha256": audit, "harper_archive_membership_sha256": MEMBERSHIP}
    _write_json(reports / f"{JOB}-cohort.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
