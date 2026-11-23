"""Materialize the single-reviewer Hindi training pilot on the staged call tracks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .dense_teacher import file_digest
from .paired_call_review import QUIET
from .progress import Progress
from .sources import data_root

SOURCE_SHA256 = "4c74777f37cca157dfa84d7505fe0c68a1202a24b607749b58baed278478ab83"
JOB = "hindi-single-reviewer-pilot"
OUTCOMES = {"resume": 0, "handoff": 1}
EDGE_MARGIN = RATE * 80 // 1000


def ballot_outcome(event):
    outcome = event["outcome"]
    if outcome not in OUTCOMES:
        return None, "unclear_or_conflicting"
    views = event["views"]
    if {view["judgment"]["language"] for view in views.values()} - {"hindi"}:
        return None, "review_language_not_hindi"
    if any(view["judgment"]["confidence"] == "1" for view in views.values()):
        return None, "low_reported_confidence"
    if "assisted" in views and views["assisted"]["first_outcome"] != outcome:
        return None, "assisted_first_and_final_disagree"
    return outcome, None


def supported_edge(tracks, activity, focal, edge):
    """Keep the original anchor and require native channel and quiet evidence."""
    if edge < 3 * RATE - QUIET or edge + QUIET > tracks.shape[1]:
        return False
    offsets = [end for _, end in activity[focal] if abs(end - edge) <= EDGE_MARGIN]
    if len(offsets) != 1:
        return False
    if any(start < edge + QUIET and end > edge for channel in activity for start, end in channel):
        return False
    pre = tracks[:, edge - RATE // 5:edge].astype(np.float64)
    own, other = np.sqrt(np.square(pre).mean(axis=1))[[focal, 1 - focal]]
    return own >= max(128., 1.414 * other)


def materialize(root, ballot_path):
    from .dral_label_pilot import isolation
    from .eot_domain_probe import decision_audio
    from .spring_hindi_clock import load_tracks, read_inputs

    if file_digest(ballot_path) != SOURCE_SHA256:
        raise ValueError("the single-reviewer source-event receipt changed")
    ballot = json.loads(ballot_path.read_text())
    plan, audio, source, publisher_train, audio_sha = read_inputs(root)
    if any(tile["recording"] not in publisher_train for tile in plan["selection"]):
        raise ValueError("only publisher train calls may enter the Hindi fit")
    events = ballot["events"]
    keys = {(row["recording"], row["focal_channel"], row["stop_sample"]) for row in events}
    geometry = {tile["recording"]: tile for tile in plan["selection"]}
    if len(keys) != len(events) or set(geometry) != {key[0] for key in keys} or len(geometry) != 6:
        raise ValueError("unexpected source-call or event membership")

    directory = root / "runs" / JOB
    output = root / "reports" / f"{JOB}-membership.json"
    if output.exists():
        raise ValueError("the pilot membership is already locked")
    directory.mkdir(parents=True, exist_ok=True)
    detector = silero()
    counts = Counter()
    rows, pcm = [], []
    meter = Progress(f"{JOB}-screen", len(geometry), "calls")
    try:
        for recording, tile in geometry.items():
            tracks = load_tracks(source, audio, tile)
            activity = [detector(track) for track in tracks]
            clean = isolation(tracks, activity)["passed"]
            for event in (row for row in events if row["recording"] == recording):
                outcome, exclusion = ballot_outcome(event)
                channel = event["focal_channel"]
                edge = event["stop_sample"] - tile["start_native_sample"]
                if exclusion is None:
                    if not clean or not supported_edge(tracks, activity, channel, edge):
                        exclusion = "insufficient_native_acoustic_support"
                    elif outcome == "resume" and any(
                        view["judgment"]["brief_ack"] == "yes" for view in event["views"].values()
                    ):
                        exclusion = "acknowledgment_needs_separate_analysis"
                if exclusion is not None:
                    counts[exclusion] += 1
                    continue
                rows.append({"recording": recording, "channel": channel, "stop_sample": event["stop_sample"],
                             "label": OUTCOMES[outcome], "language": "hi", "rater_count": 1,
                             "language_source": "publisher_split", "timing_source": "silero_proposal",
                             "views": sorted(event["views"])})
                pcm.append(decision_audio(tracks[channel], edge, "telephone")[-3 * RATE:])
                counts[outcome] += 1
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()

    clip_path = directory / "pcm.npy"
    with clip_path.with_suffix(".part").open("wb") as stream:
        np.save(stream, np.asarray(pcm, dtype=np.float32), allow_pickle=False)
    clip_path.with_suffix(".part").replace(clip_path)
    summary = {"scope": "single-reviewer training feasibility only", "source_events_sha256": SOURCE_SHA256,
               "source_audio_sha256": audio_sha, "decision_ms": 300,
               "counts": dict(sorted(counts.items())), "source_calls": len(geometry),
               "calls_by_class": {outcome: len({row["recording"] for row in rows if row["label"] == label})
                                  for outcome, label in OUTCOMES.items()},
               "pcm": {"path": clip_path.relative_to(root).as_posix(), "sha256": file_digest(clip_path)}}
    _write_json(output, {**summary, "rows": rows})
    print(json.dumps({**summary, "membership_sha256": file_digest(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ballot", type=Path)
    args = parser.parse_args()
    materialize(data_root(), args.ballot)
