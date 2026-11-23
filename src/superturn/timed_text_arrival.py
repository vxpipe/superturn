"""Replay received STT revisions against the application's silence clock."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .ami_calibration import _write_json
from .sources import data_root
from .turn_generation import normalize
from .turn_text_train import digest
from .turn_training_dataset import model_text

MS = 1_000_000
FIRST_NS = 300 * MS
STEP_NS = 100 * MS
TIMEOUT_NS = 800 * MS


@dataclass(frozen=True)
class Revision:
    """A complete replacement for one caller turn, timestamped at receipt."""

    sequence: int
    received_ns: int
    turn_id: str
    text: str


@dataclass(frozen=True)
class PauseTrace:
    turn_id: str
    previous_turn: str
    first_sample_ns: int
    sample_rate: int
    stop_sample: int
    resume_sample: int | None
    captured_through_ns: int

    def sample_time(self, sample):
        return self.first_sample_ns + sample * 1_000_000_000 // self.sample_rate


@dataclass(frozen=True)
class Decision:
    since_stop_ns: int
    revision_sequence: int | None
    text: str | None


@dataclass(frozen=True)
class Replay:
    at_300_ms: Decision | None
    ticks: tuple[Decision, ...]
    benchmark_grid: tuple[Decision, ...]
    ended_by: str


def replay(trace, revisions):
    """Rebuild potential scoring ticks without reading the label or future transcript."""
    if (trace.sample_rate <= 0 or trace.stop_sample < 0
            or trace.resume_sample is not None and trace.resume_sample < trace.stop_sample):
        raise ValueError("invalid sample clock or pause boundaries")
    stop = trace.sample_time(trace.stop_sample)
    if trace.captured_through_ns < stop:
        raise ValueError("the stop lies after the captured trace")
    ordered = sorted(revisions, key=lambda event: (event.received_ns, event.sequence))
    if (len({event.sequence for event in ordered}) != len(ordered)
            or any(event.received_ns > trace.captured_through_ns for event in ordered)):
        raise ValueError("invalid STT revision sequence or capture clock")
    current_turn = [event for event in ordered if event.turn_id == trace.turn_id]
    final = stop + TIMEOUT_NS
    ended_by = "timeout"
    if trace.resume_sample is not None:
        resume = trace.sample_time(trace.resume_sample)
        if resume <= final:
            final, ended_by = resume, "resume"
    if trace.captured_through_ns < final:
        ended_by = "capture_ended"

    def observed(at):
        # The timeout or caller resumption takes priority at an equal time.
        return at < final and at <= trace.captured_through_ns

    previous = normalize(trace.previous_turn)
    index = 0
    latest = None

    def decision(at):
        nonlocal index, latest
        while index < len(current_turn) and current_turn[index].received_ns <= at:
            latest = current_turn[index]
            index += 1
        normalized = normalize(latest.text) if latest else ""
        return Decision(at - stop, latest.sequence if latest else None,
                        model_text(previous, normalized) if normalized else None)

    first = stop + FIRST_NS
    grid = tuple(decision(at) for at in range(first, stop + TIMEOUT_NS + 1, STEP_NS)
                 if at <= trace.captured_through_ns
                 and (at < final or at == final and ended_by == "timeout"))
    index, latest = 0, None
    if not observed(first):
        return Replay(None, (), grid, ended_by)
    at_300 = decision(first)
    ticks = []
    tick = first
    eligible = at_300
    # The first call waits for available caller text and begins a 100 ms cadence.
    while eligible.text is None and index < len(current_turn):
        tick = current_turn[index].received_ns
        if not observed(tick):
            break
        eligible = decision(tick)
    if eligible.text is not None:
        ticks.append(eligible)
        tick += STEP_NS
        while observed(tick):
            ticks.append(decision(tick))
            tick += STEP_NS
    return Replay(at_300, tuple(ticks), grid, ended_by)


def pack_trace_file(root, source, destination):
    if (not (root / ".campaign-machine").is_file() or
            any(not path.resolve().is_relative_to(root.resolve()) for path in (source, destination))
            or source.resolve() == destination.resolve()):
        raise ValueError("benchmark traces and packed inputs stay under the Vast data directory")
    capture = json.loads(source.read_text())
    provider = capture["provider_settings_sha256"]
    if len(provider) != 64 or any(ch not in "0123456789abcdef" for ch in provider):
        raise ValueError("the trace must name immutable provider settings")
    results = []
    counts = Counter()
    ids = set()
    for event in capture["pauses"]:
        ident, label = event["id"], event["label"]
        if ident in ids or label not in {"hold", "shift"}:
            raise ValueError("duplicate pause ID or invalid label")
        ids.add(ident)
        if exclusion := event.get("scoring_exclusion"):
            if exclusion != "missing_previous_stt":
                raise ValueError("unknown scoring exclusion")
            counts[f"{label}/300_ms/{exclusion}"] += 1
            results.append({"id": ident, "label": label, "scoring_exclusion": exclusion,
                            "at_300_ms": None, "ticks": [], "benchmark_grid": [], "ended_by": "abstained"})
            continue
        trace = PauseTrace(**event["trace"])
        observed = replay(trace, [Revision(**row) for row in event["revisions"]])
        first = observed.at_300_ms
        coverage = "unobserved" if first is None else "text" if first.text else "no_text"
        counts[f"{label}/300_ms/{coverage}"] += 1
        counts[f"{label}/ended_by/{observed.ended_by}"] += 1
        results.append({"id": ident, "label": label, **asdict(observed)})
    if not results:
        raise ValueError("the trace contains no pauses")
    receipt = {"scope": "Vast-only received transcript coverage",
               "input_sha256": digest(source), "provider_settings_sha256": provider, "coverage": dict(counts)}
    _write_json(destination, {**receipt, "pauses": results})
    print(json.dumps({**receipt, "output_sha256": digest(destination)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("packed", type=Path)
    args = parser.parse_args()
    pack_trace_file(data_root(), args.trace, args.packed)
