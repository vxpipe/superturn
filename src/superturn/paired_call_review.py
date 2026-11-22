"""Prepare a blind floor-label review from the staged paired-call pilots."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .dense_teacher import file_digest
from .progress import Progress
from .shards import seed_for
from .sources import data_root

JOB = "paired-call-floor-review"
PRE = 3 * RATE
POST = 3 * RATE
QUIET = 300 * RATE // 1000
PER_CALL = 12
CONTINUOUS_REPORTS = {
    "continuous": "continuous-spring-floor-review",
    "continuous-datahive": "continuous-datahive-floor-review",
    "continuous-control": "continuous-harper-floor-review",
}
WINDOW_FIELDS = ("recording", "reviewed", "left_language", "right_language")

GUIDE = """# Review a conversational pause

At 3.000 s the left-channel speaker reaches a proposed pause. The other
speaker is on the right. Listen in stereo. Use the whole clip to identify
what happened. Work independently and retain your assigned row order.

Record an outcome from this list:

- resume: the left speaker continues, possibly after a brief acknowledgment.
- handoff: the right speaker takes a substantive turn or gives a short
  answer, and the left speaker does not resume the previous thought.
- interruption: the right speaker cuts into unfinished left speech.
- uncertain: the outcome is inaudible, ambiguous, or outside the clip.
- invalid_pause: speech overlaps the interval from 3.000 to 3.300 s, or
  there is no audible left-channel stop near 3.000 s.

Use brief_ack=yes when a brief right-channel acknowledgment accompanies a
left-channel continuation. Record the language of the left speech using
a plain language name, mixed, or unknown. Use confidence of 1 (low), 2
(moderate), or 3 (high). Never infer an outcome just from silence.

The audio consists of excerpts from IIT Madras SPRING-INX Hindi R1 and
DataHive multilingual-conversations-sample, CC BY 4.0. Channel order may
be swapped to place the focal speaker on the left. These labels describe
observed behavior for a training feasibility review.
[SPRING-INX](https://asr.iitm.ac.in/)
[DataHive](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample)
"""

CONTINUOUS_GUIDE = """# Review continuous paired speech

Listen independently in stereo. The two channels preserve the original
recording over each 30-second slice. Mark each clearly audible stop
followed by at least 300 ms without speech on either channel. Record the
channel that stopped, both boundary times in milliseconds from file
start, and the next speaker. If speech never resumes in the file, mark
the outcome uncertain.

Use the outcomes resume, handoff, interruption, or uncertain. Count
brief acknowledgments separately. A short substantive answer may be a
handoff. Record the focal speaker's language as heard and confidence
from 1 to 3. Listen to this continuous packet before opening any
extracted-pause clips, so the proposal system cannot bias your marks.

Audio excerpts from IIT Madras SPRING-INX Hindi R1 (CC BY 4.0).
[Publisher](https://asr.iitm.ac.in/)
"""


def candidates(activity, length):
    """Find focal offsets followed by 300 ms of quiet on both channels."""
    result = []
    for channel, spans in enumerate(activity):
        other = activity[1 - channel]
        for index, (_, edge) in enumerate(spans):
            if edge < PRE or edge + POST > length:
                continue
            if index + 1 < len(spans) and spans[index + 1][0] < edge + QUIET:
                continue
            if any(start < edge + QUIET and end > edge for start, end in other):
                continue
            result.append((edge, channel))
    return sorted(result)


def staged_calls(root):
    from .datahive_conversation_pilot import inventory, load_call
    from .spring_hindi_clock import load_tracks, read_inputs

    plan, audio, spring, _, _ = read_inputs(root)
    for tile in plan["selection"]:
        yield ("spring", tile["recording"], "hi", tile["start_native_sample"],
               lambda tile=tile: load_tracks(spring, audio, tile))
    source, folder, _, metadata = inventory(root)
    receipt = json.loads((root / "reports/datahive-conversation-pilot.json").read_text())
    for name, digest in sorted(source.get("audio_sha256").items()):
        reference = receipt["calls"][name]

        def load(name=name, digest=digest, reference=reference):
            tracks, actual = load_call(folder, name, digest, metadata)
            if any(actual[key] != reference[key] for key in actual):
                raise ValueError("DataHive decoded tracks changed")
            return tracks

        yield "datahive", name, reference["language"], 0, load


def write_stereo(path, tracks):
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        output.writeframes(np.asarray(tracks.T, dtype="<i2").tobytes())


def prepare(root):
    import torch

    from .dral_label_pilot import isolation

    if not (root / ".campaign-machine").is_file():
        raise ValueError("review audio stays on Vast")
    torch.set_num_threads(2)
    parent = root / "runs" / JOB
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / "packet-v1"
    if destination.exists():
        raise ValueError("the sealed packet is preserved")
    detect = silero()
    staged = list(staged_calls(root))
    meter = Progress(JOB, len(staged), "calls")
    counts, by_source, bindings = Counter(), Counter(), {}
    with TemporaryDirectory(dir=parent) as tmp:
        packet = Path(tmp) / "packet"
        listening = packet / "listening"
        (listening / "audio").mkdir(parents=True)
        (listening / "GUIDE.md").write_text(GUIDE)
        try:
            for source, name, lang, offset, load in staged:
                tracks = load()
                activity = [detect(track) for track in tracks]
                counts[source, lang, "calls"] += 1
                if isolation(tracks, activity)["passed"]:
                    counts[source, lang, "clean_calls"] += 1
                    eligible = candidates(activity, tracks.shape[1])
                    counts[source, lang, "eligible"] += len(eligible)
                    selected = sorted(eligible, key=lambda row: seed_for(JOB, source, name, *row))[:PER_CALL]
                    counts[source, lang, "selected"] += len(selected)
                    for edge, channel in selected:
                        token = seed_for(JOB, "clip", source, name, offset, edge, channel)
                        clip = f"sample-{token:016x}"
                        if clip in bindings:
                            raise ValueError("duplicate blind identifier")
                        path = listening / "audio" / f"{clip}.wav"
                        pcm = tracks[[channel, 1 - channel], edge - PRE:edge + POST]
                        write_stereo(path, pcm)
                        bindings[clip] = {"source": source, "recording": name, "nominal_language": lang,
                                          "focus_channel": channel, "native_edge_sample": offset + edge,
                                          "clip_sha256": file_digest(path)}
                        by_source[source] += 1
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
        if not bindings:
            raise ValueError("no staged pauses clear the paired-channel screen")
        for reviewer in ("a", "b"):
            ordered = sorted(bindings, key=lambda clip: seed_for(JOB, reviewer, clip))
            with (listening / f"reviewer-{reviewer}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("example", "outcome", "brief_ack", "language", "confidence"))
                writer.writerows((clip, "", "", "", "") for clip in ordered)
        _write_json(packet / "private-key.json", bindings)
        packet.rename(destination)
    report = {"scope": "pending two independent reviews; training feasibility only",
              "clips": len(bindings), "clips_by_source": by_source,
              "strata": [{"source": source, "nominal_language": lang, "stage": stage, "count": count}
                         for (source, lang, stage), count in sorted(counts.items())],
              "key_sha256": file_digest(destination / "private-key.json"),
              "guide_sha256": file_digest(destination / "listening" / "GUIDE.md"),
              "packet": str(destination.relative_to(root) / "listening")}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(report, indent=2))


def spring_slices(root):
    from .spring_hindi_clock import load_tracks, read_inputs

    plan, receipt, folder, _, digest = read_inputs(root)
    return [(tile["recording"], tile["start_native_sample"], "hi",
             lambda tile=tile: load_tracks(folder, receipt, tile), {}) for tile in plan["selection"]], digest


def datahive_slices(root):
    import torch

    from .dral_label_pilot import isolation

    torch.set_num_threads(2)
    detector = silero()
    window = 30 * RATE
    receipt = root / "reports/datahive-conversation-pilot.json"
    slices = []
    for source, name, language, _, load in staged_calls(root):
        if source != "datahive":
            continue
        tracks = load()
        activity = [detector(track) for track in tracks]
        if not isolation(tracks, activity)["passed"]:
            continue
        grid = range(window, tracks.shape[1] - window + 1, window)

        def eligible(start):
            return all(sum(max(0, min(b, start + window) - max(a, start)) for a, b in spans)
                       >= 2 * RATE for spans in activity)

        for start in sorted(grid, key=lambda point: seed_for(JOB, "datahive", name, point)):
            if eligible(start):
                clip = tracks[:, start:start + window].copy()
                slices.append((name, start, language, lambda clip=clip: clip, {}))
                break
    return slices, file_digest(receipt)


def harper_slice(root):
    import torch

    from .datahive_codec_control import observe_call
    from .harper_transcript_coverage import MEMBERSHIP
    from .yodas_mono_control import JOB as COHORT, pcm_digest

    path = root / "runs" / COHORT / "membership.json"
    report = json.loads((root / "reports" / f"{COHORT}-cohort.json").read_text())
    digest = file_digest(path)
    if digest != report["cohort_sha256"]:
        raise ValueError("the sealed HarperValley control changed")
    cohort = json.loads(path.read_text())
    if cohort["scope"] != "HarperValley train roles only" or cohort["archive_membership_sha256"] != MEMBERSHIP:
        raise ValueError("the anchor must use the existing training-only cohort")
    window = 30 * RATE
    choices = []
    outcomes = {("hold", "caller"): "resume", ("shift", "agent"): "handoff"}
    for call in cohort["calls"]:
        name = call["call"]
        for start in range(0, call["samples"] - window + 1, RATE):
            known = [{"stop_ms": round((ref["edge_sample"] - start) * 1000 / RATE),
                      "onset_ms": round((ref["next_onset_sample"] - start) * 1000 / RATE),
                      "outcome": outcomes[ref["floor_label"], ref["next_role"]]}
                     for ref in call["reference"]
                     if (ref["floor_label"], ref["next_role"]) in outcomes
                     and start + PRE <= ref["edge_sample"] and ref["next_onset_sample"] + POST <= start + window]
            count = Counter(row["outcome"] for row in known)
            if min(count["resume"], count["handoff"]) > 0:
                choices.append((-min(count.values()), -len(known), seed_for(JOB, "harper", name, start), start, call, known))
    if not choices:
        raise ValueError("no training slice contains verified examples of both floor outcomes")
    _, _, _, start, call, known = min(choices, key=lambda row: row[:3])
    name = call["call"]
    members = json.loads((root / "reports/harper-history-replication-membership.json").read_text())
    archived = {(edge, label) for identifier, edge, label in members if identifier == name}
    torch.set_num_threads(2)
    tracks, _ = observe_call(root / "raw/harper-valley", name, archived, silero())
    if [pcm_digest(track) for track in tracks] != call["source_pcm_sha256"]:
        raise ValueError("the known-answer stereo rendering changed")
    return [(name, start, "en", lambda: tracks[:, start:start + window], {"known_answers": known})], digest


def prepare_continuous(root, source="spring"):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("review audio stays on Vast")
    picker, guide = ({
        "spring": (spring_slices, CONTINUOUS_GUIDE),
        "datahive": (datahive_slices, CONTINUOUS_GUIDE.replace(
            "IIT Madras SPRING-INX Hindi R1 (CC BY 4.0)", "DataHive multilingual conversation samples (CC BY 4.0)"
        ).replace("https://asr.iitm.ac.in/", "https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample")),
        "harper": (harper_slice, CONTINUOUS_GUIDE.replace(
            "IIT Madras SPRING-INX Hindi R1 (CC BY 4.0)", "HarperValleyBank (CC BY 4.0)"
        ).replace("https://asr.iitm.ac.in/", "https://github.com/cricketclub/gridspace-stanford-harper-valley")
                   + "\nThis optional English control checks channel and timing conventions.\n"),
    })[source]
    parent = root / "runs" / JOB
    parent.mkdir(parents=True, exist_ok=True)
    version = "v2" if source == "datahive" else "v1"
    destination = parent / f"continuous-{source}-{version}"
    if destination.exists():
        raise ValueError("the sealed continuous packet is preserved")
    slices, receipt_sha = picker(root)
    if not slices:
        raise ValueError("no continuous paired windows passed the screen")
    key = {}
    job = f"continuous-{source}-floor-review"
    meter = Progress(job, len(slices), "slices")
    with TemporaryDirectory(dir=parent) as tmp:
        packet = Path(tmp) / "packet"
        listening = packet / "listening"
        (listening / "audio").mkdir(parents=True)
        if source == "datahive":
            guide += """
First complete the windows form without consulting any pause proposals.
Use reviewed=yes after listening to the full window, even if there are
no qualifying pauses. Report each channel's audible language. Use mixed,
unknown, or no_speech where applicable. Then mark pause events in the
separate event form. Leave incomplete windows blank in both forms.
"""
        (listening / "GUIDE.md").write_text(guide)
        try:
            for name, start, language, load, extra in slices:
                identifier = f"call-{seed_for(JOB, 'continuous', name, start):016x}"
                path = listening / "audio" / f"{identifier}.wav"
                tracks = load()
                write_stereo(path, tracks)
                key[identifier] = {"recording": name, "native_start_sample": start,
                                   "samples": tracks.shape[1], "clip_sha256": file_digest(path),
                                   "nominal_language": language, **extra}
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
        for reviewer in ("a", "b"):
            order = sorted(key, key=lambda name: seed_for(JOB, "continuous", reviewer, name))
            with (listening / f"reviewer-{reviewer}-order.txt").open("w") as stream:
                stream.write("\n".join(order) + "\n")
            with (listening / f"reviewer-{reviewer}.csv").open("w", newline="") as stream:
                csv.writer(stream).writerow(("recording", "focal_channel", "stop_ms", "next_onset_ms",
                                            "next_channel", "outcome", "brief_ack", "language", "confidence"))
            if source == "datahive":
                with (listening / f"reviewer-{reviewer}-windows.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(WINDOW_FIELDS)
                    writer.writerows((identifier, "", "", "") for identifier in order)
        _write_json(packet / "private-key.json", key)
        packet.rename(destination)
    report = {"scope": "independent continuous annotation before reading candidate clips",
              "slices": len(key), "seconds": sum(entry["samples"] for entry in key.values()) / RATE,
              "source_receipt_sha256": receipt_sha,
              "key_sha256": file_digest(destination / "private-key.json"),
              "packet": str(destination.relative_to(root) / "listening")}
    if source == "datahive":
        report["selection"] = ("one hashed 30-second non-opening grid slot per staged isolated call; each "
                               "channel has at least 2 seconds of Silero activity; feasibility only")
        report["slices_by_nominal_language"] = Counter(item["nominal_language"] for item in key.values())
        report["window_ballot_required"] = True
    elif source == "harper":
        report["scope"] = "optional English protocol control, separate from training and non-English feasibility"
        report["known_answer_counts"] = Counter(row["outcome"] for item in key.values() for row in item["known_answers"])
    _write_json(root / "reports" / f"{job}.json", report)
    print(json.dumps(report, indent=2))


def verify(root, reports=None):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("review audio stays on Vast")
    if reports is None:
        reports = [JOB, CONTINUOUS_REPORTS["continuous"]]
        reports.extend(name for mode, name in CONTINUOUS_REPORTS.items() if mode != "continuous"
                       and (root / "reports" / f"{name}.json").exists())
    for report_name in reports:
        report = json.loads((root / "reports" / f"{report_name}.json").read_text())
        listening = root / report["packet"]
        key_path = listening.parent / "private-key.json"
        if file_digest(key_path) != report["key_sha256"]:
            raise ValueError("review key changed")
        key = json.loads(key_path.read_text())
        clips = {path.stem: path for path in (listening / "audio").glob("*.wav")}
        if set(clips) != set(key):
            raise ValueError("review clips differ from the key")
        for identifier, path in clips.items():
            item = key[identifier]
            frames = item.get("samples", PRE + POST)
            if file_digest(path) != item["clip_sha256"]:
                raise ValueError("review audio changed")
            with wave.open(str(path), "rb") as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (
                    2, 2, RATE, frames
                ):
                    raise ValueError("review clip has incorrect geometry")
        print(f"{report_name}: verified {len(clips)} stereo files and their private key")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--continuous-spring", action="store_true")
    actions.add_argument("--continuous-datahive", action="store_true")
    actions.add_argument("--continuous-control", action="store_true")
    actions.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = data_root()
    if args.verify:
        verify(root)
    elif args.continuous_spring:
        prepare_continuous(root)
    elif args.continuous_datahive:
        prepare_continuous(root, "datahive")
    elif args.continuous_control:
        prepare_continuous(root, "harper")
    else:
        prepare(root)
