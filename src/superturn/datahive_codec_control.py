"""Validate the telephone pause extractor before acquiring additional DataHive audio."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .datahive_conversation_pilot import decode_tracks
from .datahive_pause_rescue import GAP, TOLERANCE, inspect
from .dense_teacher import file_digest
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_audit import audit_call
from .harper_valley_benchmark import pinned_split
from .labels import turns
from .open_speech_teacher_probe import WHISPER_SHA256
from .progress import Progress
from .shards import seed_for
from .sources import data_root

JOB = "datahive-codec-control"
CALLS = 10


def select_calls(entries, membership):
    labels = defaultdict(set)
    for call, _, label in membership:
        labels[call].add(label)
    selected, speakers = [], set()
    for row in sorted(entries, key=lambda row: seed_for(JOB, row["id"])):
        call, caller = row["id"], row["caller"]
        if len(labels[call]) == 2 and caller not in speakers:
            selected.append(call)
            speakers.add(caller)
        if len(selected) == CALLS:
            return selected
    raise ValueError("the control cannot select ten distinct caller voices with both labels")


def encode(tracks, target):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "s16le", "-ar", str(RATE), "-ac", "2", "-i", "pipe:0",
                    "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "128k",
                    "-joint_stereo", "1", str(target)], input=tracks.T.tobytes(), check=True)
    decoded = decode_tracks(target)
    if decoded.shape != tracks.shape:
        raise ValueError("MP3 round trip changed the sample count")
    return decoded


def compare(reference, candidates, activity):
    """Keep label out of matching and enforce reciprocal uniqueness."""
    candidates = [row for row in candidates if row["side"] == 0 and row["publisher_language_supported"]]
    matches = [[j for j, row in enumerate(candidates)
                if abs(edge - row["edge"]) <= TOLERANCE and abs(onset - row["onset"]) <= TOLERANCE]
               for edge, onset, _ in reference]
    uses = Counter(index for indices in matches for index in indices)
    counts, deltas, matched = Counter(), [], {}
    for (edge, onset, label), indices in zip(reference, matches, strict=True):
        counts["reference_" + label] += 1
        if len(indices) != 1 or uses[indices[0]] != 1:
            counts["abstained_" + label] += 1
            continue
        hit = candidates[indices[0]]
        deadline = edge + GAP
        if (hit["word_edge"] > edge or min(hit["onset"], hit["word_onset"]) < deadline
                or any(left < deadline and edge < right for left, right in activity)):
            counts["archived_deadline_conflict_" + label] += 1
            counts["abstained_" + label] += 1
            continue
        counts["matched_" + label] += 1
        matched[edge] = (hit["edge"], hit["onset"], hit["label"])
        if hit["label"] == label:
            counts["agreed_" + label] += 1
        deltas.append((hit["edge"] - edge) * 1000 / RATE)
    counts["extra_or_unverified_candidates"] = len(candidates) - len(matched)
    return counts, deltas, matched


def paired(reference, native, mp3):
    counts, shifts = Counter(), []
    for edge, _, _ in reference:
        if edge in native and edge in mp3:
            counts["both"] += 1
            native_edge, native_onset, native_label = native[edge]
            mp3_edge, mp3_onset, mp3_label = mp3[edge]
            counts["same_label_in_both"] += native_label == mp3_label
            shifts.append(((mp3_edge - native_edge) * 1000 / RATE,
                           (mp3_onset - native_onset) * 1000 / RATE))
        elif edge in native:
            counts["native_only"] += 1
        elif edge in mp3:
            counts["mp3_only"] += 1
        else:
            counts["neither"] += 1
    return counts, shifts


def observe_call(raw, call, member, detect):
    observed = {}

    def seal(audio, origins, primary, words, omissions, left, right):
        tracks = np.stack([audio[role][left - origins[role]:right - origins[role]]
                           for role in ("caller", "agent")]).astype("<i2", copy=False)
        reference = [(span.end - left, span.end + gap - left, label)
                     for span, gap, label in turns(primary, decision_ms=300)
                     if span.speaker == "caller" and (span.end - origins["caller"], label) in member
                     and span.end - left >= 3 * RATE and span.end + gap + 2 * RATE <= right]
        observed.update(tracks=tracks, reference=reference)

    reproduced = []
    audit_call(raw, call, detect, recover_clock=True, pauses=reproduced, observe=seal)
    if set(reproduced) != member or not observed.get("reference"):
        raise ValueError("the archived control labels did not reproduce")
    return observed["tracks"], observed["reference"]


def summary(counts, deltas):
    matched = sum(counts["matched_" + label] for label in ("hold", "shift"))
    agreed = sum(counts["agreed_" + label] for label in ("hold", "shift"))
    agreement = agreed / matched if matched else None
    offset = median(deltas) if deltas else None
    passed = (min(counts["matched_hold"], counts["matched_shift"]) >= 10
              and agreement is not None and agreement >= .85 and abs(offset) <= 80)
    return {"counts": dict(counts), "label_agreement_among_matched": agreement,
            "signed_endpoint_median_ms": offset, "passes": passed}


def run(root):
    import torch
    import whisper

    if not (root / ".campaign-machine").is_file():
        raise ValueError("recordings stay on Vast")
    entries, _, audit_sha = pinned_split(root, "train")
    reports = root / "reports"
    membership = json.loads((reports / "harper-history-replication-membership.json").read_text())
    if hashlib.sha256(json.dumps(membership).encode()).hexdigest() != MEMBERSHIP:
        raise ValueError("the archived control membership changed")
    chosen = select_calls(entries, membership)
    members = defaultdict(set)
    for call, edge, label in membership:
        members[call].add((edge, label))

    weight = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(weight) != WHISPER_SHA256:
        raise ValueError("the cached ASR model changed")
    torch.set_num_threads(2)
    model = whisper.load_model(str(weight), device="cuda")
    detect = silero()
    totals = {"native": Counter(), "mp3": Counter()}
    offsets = {"native": [], "mp3": []}
    paired_counts, paired_offsets = Counter(), []
    per_call = []
    meter = Progress(JOB, 2 * len(chosen), "call-renders")
    try:
        with TemporaryDirectory() as tmp:
            for call in chosen:
                tracks, reference = observe_call(root / "raw/harper-valley", call, members[call], detect)
                result, matches = {}, {}
                for name, rendered in (("native", tracks), ("mp3", encode(tracks, Path(tmp) / "control.mp3"))):
                    observed = inspect(rendered, detect, model, "en")
                    activity = [span for track in rendered for span in detect(track)]
                    counts, deltas, matches[name] = compare(reference, observed["provisional_pauses"], activity)
                    totals[name].update(counts)
                    offsets[name].extend(deltas)
                    result[name] = {"comparison": summary(counts, deltas), "isolation": observed["isolation"],
                                    "attrition": observed["silero_vs_words"]}
                    meter.advance(1)
                counts, deltas = paired(reference, matches["native"], matches["mp3"])
                paired_counts.update(counts)
                paired_offsets.extend(deltas)
                per_call.append({"control": result, "paired_counts": dict(counts)})
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    outcomes = {name: summary(totals[name], offsets[name]) for name in totals}
    report = {"scope": "HarperValley train-label extractor control, no new data or model fit",
              "source_audit_sha256": audit_sha, "pause_membership_sha256": MEMBERSHIP,
              "selected_calls_sha256": hashlib.sha256(json.dumps(chosen).encode()).hexdigest(),
              "whisper_sha256": WHISPER_SHA256,
              "mp3": {"sample_rate": 48000, "bitrate": 128000, "joint_stereo": True},
              "paired_counts": dict(paired_counts),
              "paired_mp3_minus_native_median_ms": [median(axis) for axis in zip(*paired_offsets)],
              "arms": outcomes, "calls": per_call, "passes": all(arm["passes"] for arm in outcomes.values())}
    _write_json(reports / f"{JOB}.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(data_root())["arms"], indent=2))
