"""Replay frozen HarperValley development pauses through a local streaming ASR."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from math import sqrt
import re
import time

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .harper_text_inputs import prefix_turns
from .harper_valley_audit import audit_call
from .progress import Progress
from .sources import data_root
from .timed_text_arrival import FIRST_NS, PauseTrace, Revision, pack_trace_file, replay
from .turn_generation import normalize
from .turn_text_train import digest

MODEL = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
MODEL_REVISION = "672fbf1b30579d6585301139bb363f42a0ad4a24"
MODEL_FILES = {
    "encoder-epoch-99-avg-1-chunk-16-left-128.onnx": "a423883ce5754507fd941755ab0b5bc426a84ac670cbe21cf060e9e2c66dc660",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx": "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
    "joiner-epoch-99-avg-1-chunk-16-left-128.onnx": "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
    "tokens.txt": "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
}
RUNTIME = "1.12.23+cuda12.cudnn9"
FRAME = RATE // 50
NS_PER_SAMPLE = 1_000_000_000 // RATE
PRIOR_QUIET = RATE * 3 // 10
EOF_PAD = np.zeros(RATE * 66 // 100, dtype=np.float32)
LOOKAHEAD = RATE * 8 // 10
DIGITS = "zero one two three four five six seven eight nine".split()


def clean_text(text):
    text = normalize(text)
    return normalize(re.sub(r"[0-9]", lambda match: f" {DIGITS[int(match[0])]} ", text))


def pause_id(row):
    return sha256(f"{row['call']}:{row['caller_sample']}".encode()).hexdigest()


@dataclass(frozen=True)
class Episode:
    row: dict
    origin: int
    caller_start: int
    caller_pcm: np.ndarray
    stop: int
    resume: int | None
    agent_start: int | None
    agent_finalize: int | None
    agent_pcm: np.ndarray | None


@dataclass
class Track:
    stream: object
    start: int
    pcm: np.ndarray
    finish: bool
    position: int = 0
    finished: bool = False

    def feed(self, cursor):
        end = min(len(self.pcm), cursor - self.start)
        if end > self.position:
            self.stream.accept_waveform(RATE, self.pcm[self.position:end])
            self.position = end
        if self.finish and not self.finished and self.position == len(self.pcm):
            # The pinned transducer only drains full chunks after EOF.
            # Once the agent endpoint closes, provide the documented padding.
            self.stream.accept_waveform(RATE, EOF_PAD)
            self.stream.input_finished()
            self.finished = True


def episode_for(row, audio, origins, vad, common_start, common_end):
    stop = row["caller_sample"] + origins["caller"]
    turns, issue = prefix_turns(vad, stop, common_start, origins["caller"] == origins["agent"] == common_start)
    if issue:
        raise ValueError(f"frozen causal membership no longer has its turn boundaries: {issue}")
    prior, current = turns
    resume = min((span.start for span in vad if span.speaker == "caller" and span.start > stop), default=None)
    limit = min(common_end, stop + LOOKAHEAD)
    caller_end = min(limit, resume) if resume is not None else limit
    caller_start = current[0]

    def pcm(role, start, end):
        offset = origins[role]
        return audio[role][start - offset:end - offset].astype(np.float32) / 32768

    agent_start, agent_finalize, agent_pcm = None, None, None
    if prior is not None:
        agent_start = prior[0]
        # Use the 300 ms quiet gate to close prior agent ASR before the
        # caller begins. A late or missing prior explicitly abstains.
        candidate_finalize = prior[1] + PRIOR_QUIET
        if candidate_finalize < caller_start:
            agent_finalize = candidate_finalize
            agent_pcm = pcm("agent", agent_start, agent_finalize)
    origin = agent_start if agent_finalize is not None else caller_start
    return Episode(row, origin, caller_start, pcm("caller", caller_start, caller_end),
                   stop, resume, agent_start, agent_finalize, agent_pcm)


def capture_batch(recognizer, episodes, *, clock=time.monotonic_ns, wait=time.sleep):
    start_ns = clock()
    tracks, finals, caller_revisions = [], {}, defaultdict(list)
    received_raw = defaultdict(list)
    packet_lags = defaultdict(int)
    first_samples = [start_ns - episode.origin * NS_PER_SAMPLE for episode in episodes]
    for index, episode in enumerate(episodes):
        tracks.append((index, "caller", Track(recognizer.create_stream(), episode.caller_start, episode.caller_pcm, False)))
        if episode.agent_pcm is not None:
            tracks.append((index, "agent", Track(recognizer.create_stream(), episode.agent_start, episode.agent_pcm, True)))
    duration = max(track.start + len(track.pcm) - episodes[index].origin for index, _, track in tracks)
    boundaries = set(range(FRAME, duration + FRAME, FRAME))
    for index, _, track in tracks:
        boundaries.add(track.start + len(track.pcm) - episodes[index].origin)
    pacing_lag = 0
    for offset in sorted(boundaries):
        deadline = start_ns + offset * NS_PER_SAMPLE
        while (remaining := deadline - clock()) > 0:
            wait(remaining / 1_000_000_000)
        pacing_lag = max(pacing_lag, clock() - deadline)
        for index, _, track in tracks:
            release_lag = clock() - deadline
            before = track.position
            track.feed(episodes[index].origin + offset)
            if track.position > before:
                packet_lags[index] = max(packet_lags[index], release_lag)
        ready = [track.stream for _, _, track in tracks if recognizer.is_ready(track.stream)]
        while ready:
            recognizer.decode_streams(ready)
            ready = [stream for stream in ready if recognizer.is_ready(stream)]
        for index, role, track in tracks:
            if not track.position:
                continue
            if role == "agent" and (not track.finished or index in finals):
                continue
            raw = recognizer.get_result(track.stream)
            received = clock()
            cleaned = clean_text(raw)
            if role == "agent":
                finals[index] = received, raw, cleaned
                continue
            raw_history = received_raw[index]
            if not raw_history or raw_history[-1][1] != raw:
                raw_history.append((received, raw))
            history = caller_revisions[index]
            if (history and history[-1].text != cleaned) or (not history and cleaned):
                history.append(Revision(len(history), received, "caller", cleaned))

    result = []
    for index, episode in enumerate(episodes):
        row = episode.row
        first = first_samples[index]
        previous_time, previous_raw, previous_text = finals.get(index, (None, "", ""))
        caller_start_ns = first + episode.caller_start * NS_PER_SAMPLE
        missing_prior = episode.agent_start is not None and (not previous_text or previous_time >= caller_start_ns)
        history = caller_revisions[index]
        captured_through = first + (episode.caller_start + len(episode.caller_pcm)) * NS_PER_SAMPLE
        # A delayed revision cannot extend observed audio past its watermark.
        received = [item for item in history if item.received_ns <= captured_through]
        cutoff = min(captured_through, first + episode.stop * NS_PER_SAMPLE + FIRST_NS)
        current_raw = next((value for stamp, value in reversed(received_raw[index]) if stamp <= cutoff), "")
        trace = PauseTrace("caller", previous_text if not missing_prior else "", first, RATE,
                           episode.stop, episode.resume, captured_through)
        result.append({
            "id": pause_id(row),
            "label": row["label"],
            "trace": asdict(trace),
            "revisions": [asdict(revision) for revision in received],
            "after_observed_audio_revisions": [asdict(revision) for revision in history
                                               if revision.received_ns > captured_through],
            "prior_status": ("missing_at_caller_onset" if missing_prior else
                             "first_turn" if episode.agent_start is None else "final_before_caller_onset"),
            "prior_final_received_ns": previous_time,
            "caller_start_sample": episode.caller_start,
            "digit_cleaner_changed_text_at_300": bool(current_raw and (
                clean_text(current_raw) != normalize(current_raw) or previous_text != normalize(previous_raw))),
            "raw_pair_at_300": [previous_raw, current_raw] if not missing_prior and current_raw else None,
            "max_packet_release_lag_ms": packet_lags[index] / 1_000_000,
            "scheduled_active_tracks_at_300": sum(
                track.start - episodes[other].origin <= episode.stop - episode.origin + PRIOR_QUIET
                < track.start + len(track.pcm) - episodes[other].origin
                for other, _, track in tracks
            ),
        })
    return result, pacing_lag / 1_000_000


def open_recognizer(model_dir):
    import torch
    import sherpa_onnx

    if not torch.cuda.is_available() or version("sherpa-onnx") != RUNTIME:
        raise ValueError("the frozen CUDA sherpa-onnx environment is unavailable")
    for name, expected in MODEL_FILES.items():
        if digest(model_dir / name) != expected:
            raise ValueError(f"unexpected streaming ASR artifact: {name}")
    paths = {name.split("-")[0].split(".")[0]: str(model_dir / name) for name in MODEL_FILES}
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **paths, num_threads=1, provider="cuda", enable_endpoint_detection=False,
        decoding_method="greedy_search", cudnn_conv_algo_search=1,
    )


def capture_episodes(recognizer, episodes, batch_size, name):
    captured, pacing_lags = [], []
    meter = Progress(name, len(episodes), "pauses")
    for start in range(0, len(episodes), batch_size):
        batch = episodes[start:start + batch_size]
        events, max_lag = capture_batch(recognizer, batch)
        captured.extend(events)
        pacing_lags.append(max_lag)
        meter.advance(len(batch))
    meter.finish()
    return captured, max(pacing_lags)


def observe_at_300(event):
    if event["prior_status"] == "missing_at_caller_onset":
        return None
    revisions = [Revision(**revision) for revision in event["revisions"]]
    return replay(PauseTrace(**event["trace"]), revisions).at_300_ms


def repeat_diagnostics(first_run, second_run):
    counts = defaultdict(Counter)
    arrival_times = defaultdict(list)
    for first, second in zip(first_run, second_run, strict=True):
        if (first["id"], first["label"]) != (second["id"], second["label"]):
            raise ValueError("repeat capture membership or order changed")
        label = first["label"]
        a, b = (observe_at_300(event) for event in (first, second))
        left, right = a.text if a else None, b.text if b else None
        for group in ("all", label):
            counts[group]["rows"] += 1
            counts[group]["exact_packed_state_agreement"] += left == right
            counts[group]["text_in_both_runs"] += bool(left and right)
            counts[group]["same_packed_text_in_both"] += bool(left and left == right)
            counts[group]["same_raw_provider_text_in_both"] += bool(
                left and right and first["raw_pair_at_300"] == second["raw_pair_at_300"])
            counts[group]["same_missing_prior_state"] += first["prior_status"] == second["prior_status"]
            counts[group][f"a_{'text' if left else 'abstain'}_b_{'text' if right else 'abstain'}"] += 1
        visible, first_arrivals = [], []
        for run, event, decision in (("a", first, a), ("b", second, b)):
            trace = PauseTrace(**event["trace"])
            if event["revisions"]:
                first_ns = event["revisions"][0]["received_ns"]
                elapsed = (first_ns - trace.sample_time(event["caller_start_sample"])) / 1_000_000
                arrival_times[f"{run}_first_nonempty_revision_relative_onset_ms"].append(elapsed)
                first_arrivals.append(elapsed)
            if decision and decision.text:
                revision = next(row for row in event["revisions"] if row["sequence"] == decision.revision_sequence)
                elapsed = (revision["received_ns"] - trace.sample_time(trace.stop_sample)) / 1_000_000
                arrival_times[f"{run}_visible_revision_relative_stop_ms"].append(elapsed)
                visible.append(elapsed)
        for name, values in (("visible_revision", visible), ("first_revision", first_arrivals)):
            if len(values) == 2:
                arrival_times[f"paired_b_minus_a_{name}_ms"].append(values[1] - values[0])

    agreement = {}
    for label, values in counts.items():
        agreement[label] = dict(values)
        if available := values["text_in_both_runs"]:
            proportion = values["same_raw_provider_text_in_both"] / available
            z = 1.959963984540054
            denominator = 1 + z ** 2 / available
            center = (proportion + z ** 2 / (2 * available)) / denominator
            radius = z * sqrt(proportion * (1 - proportion) / available + z ** 2 / (4 * available ** 2)) / denominator
            agreement[label]["raw_agreement_when_both_available_wilson_95"] = [
                max(0., center - radius), min(1., center + radius),
            ]
    return {"state_agreement": agreement,
            "arrival_quantiles": {name: {
                "n": len(values), "not_observed": len(first_run) - len(values),
                "p25_p50_p75": np.percentile(values, [25, 50, 75]).tolist(),
                "median_abs": float(np.median(np.abs(values))) if name.startswith("paired") else None,
            } for name, values in arrival_times.items()}}


def run(root, model_dir, *, batch_size=8, limit=None):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("HarperValley streaming capture only runs on Vast")
    cache = root / "cache/harper-valley"
    reports = root / "reports"
    pack_path = cache / "text-dev-v1.jsonl"
    source_receipt = json.loads((reports / "harper-text-dev-packing.json").read_text())
    pack_sha = digest(pack_path)
    rows = [json.loads(line) for line in pack_path.read_text().splitlines()]
    if (pack_sha != source_receipt["serialized_sha256"] or source_receipt["rows"] != len(rows)
            or len({(row["call"], row["caller_sample"]) for row in rows}) != len(rows)):
        raise ValueError("the frozen HarperValley development input pack changed")
    if limit is not None:
        chosen = [row for label in ("hold", "shift") for row in
                  [row for row in rows if row["label"] == label][:limit]]
    else:
        chosen = rows

    recognize = open_recognizer(model_dir)
    # Warm the CUDA path with generated silence, before opening real data.
    warmup = recognize.create_stream()
    warmup.accept_waveform(RATE, np.zeros(RATE, dtype=np.float32))
    while recognize.is_ready(warmup):
        recognize.decode_stream(warmup)

    by_call = defaultdict(list)
    for row in chosen:
        by_call[row["call"]].append(row)
    detect = silero()
    episodes = []
    prepare = Progress("harper-streaming-prepare", len(by_call), "calls")
    for call, group in by_call.items():
        def collect(audio, origins, vad, _words, _omissions, start, end):
            episodes.extend(episode_for(row, audio, origins, vad, start, end) for row in group)

        audit_call(root / "raw/harper-valley", call, detect, recover_clock=True, observe=collect)
        prepare.advance(1)
    prepare.finish()
    if len(episodes) != len(chosen):
        raise ValueError("failed to reproduce the frozen HarperValley calls")

    settings = {
        "model": MODEL, "revision": MODEL_REVISION, "model_license": "Apache-2.0", "assets_sha256": MODEL_FILES,
        "runtime": RUNTIME, "provider": "cuda", "provider_device": 0, "num_threads": 1,
        "decoding": "greedy_search", "audio_rate": RATE, "packet_ms": 20, "batch_size": batch_size,
        "clock": "client monotonic; chunks released at their last sample and revisions stamped after decode",
        "prior": "agent stream closes after 300 ms real quiet, drains with 660 ms EOF padding, and must arrive before caller onset",
        "caller": "separate caller stream opens on onset and ends on resumption or 800 ms after Silero stop",
        "benchmark_batch_order": "sort by SHA-256 of call:caller_sample, no salt, independent of playback length",
        "repeated_pauses": 0 if limit is not None else 50,
        "cleaner": "repository normalizer plus ASCII digit-by-digit spelling",
    }
    settings_sha = sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    limits = "smoke-v2" if limit is not None else "v2"
    trace_path = cache / f"text-dev-streaming-trace-{limits}.json"
    packed_path = cache / f"text-dev-streaming-{limits}.json"
    report_path = reports / f"harper-text-streaming-capture-{limits}.json"
    episodes.sort(key=lambda ep: pause_id(ep.row))
    captured, max_lag = capture_episodes(recognize, episodes, batch_size, "harper-streaming-capture")
    rec_by_id = {pause_id(row): row for row in chosen}
    counts = Counter()
    for event in captured:
        label = event["label"]
        counts[f"{label}/prior/{event['prior_status']}"] += 1
        counts[f"{label}/captured"] += 1
        if event["max_packet_release_lag_ms"] > 20:
            counts[f"{label}/packet_release_lag_over_20_ms"] += 1
        if event["prior_status"] == "missing_at_caller_onset":
            event["scoring_exclusion"] = "missing_previous_stt"
        else:
            decision = observe_at_300(event)
            if decision and decision.text:
                counts[f"{label}/300_ms/text"] += 1
                if event["digit_cleaner_changed_text_at_300"]:
                    counts[f"{label}/300_ms/digit_cleaner_changed"] += 1
                if decision.text != rec_by_id[event["id"]]["text"]:
                    counts[f"{label}/300_ms/differs_from_word_cut"] += 1
    envelope = {"provider_settings_sha256": settings_sha, "frozen_pack_sha256": pack_sha, "pauses": captured}
    _write_json(trace_path, envelope)
    pack_trace_file(root, trace_path, packed_path)
    active_tracks = [event["scheduled_active_tracks_at_300"] for event in captured]
    post_stop_gaps = [rec_by_id[event["id"]]["gap_ms"] for event in captured]
    timing_correlation = float(np.corrcoef(active_tracks, post_stop_gaps)[0, 1]) if (
        np.std(active_tracks) > 0 and np.std(post_stop_gaps) > 0) else None
    receipt = {
        "scope": "provisional HarperValley streaming-ASR condition, awaiting independent labels",
        "frozen_pack_sha256": pack_sha, "provider_settings_sha256": settings_sha, "provider_settings": settings,
        "trace_sha256": digest(trace_path), "packed_sha256": digest(packed_path),
        "trace_path": trace_path.relative_to(root).as_posix(), "packed_path": packed_path.relative_to(root).as_posix(),
        "rows": len(captured), "counts": dict(counts), "max_batch_pacing_lag_ms": max_lag,
        "scheduled_active_tracks_at_300_p25_p50_p75": np.percentile(active_tracks, [25, 50, 75]).tolist(),
        "scheduled_active_tracks_at_300_and_post_stop_gap_correlation": timing_correlation,
    }
    if limit is None:
        repeated = episodes[:settings["repeated_pauses"]]
        a, first_lag = capture_episodes(recognize, repeated, batch_size, "harper-streaming-repeat-a")
        b, second_lag = capture_episodes(recognize, repeated, batch_size, "harper-streaming-repeat-b")
        repeat_path = cache / "text-dev-streaming-repeat-v2.json"
        _write_json(repeat_path, {"provider_settings_sha256": settings_sha,
                                  "frozen_pack_sha256": pack_sha, "repeat_a": a, "repeat_b": b})
        receipt.update(repeat_diagnostics(a, b), repeat_trace_sha256=digest(repeat_path),
                       repeat_trace_path=repeat_path.relative_to(root).as_posix(),
                       repeat_max_batch_pacing_lag_ms=[first_lag, second_lag])
    _write_json(report_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_per_class is not None and args.limit_per_class < 1):
        parser.error("batch size and any per-class limit must be positive")
    root = data_root()
    run(root, root / "cache/harper-streaming-model", batch_size=args.batch_size, limit=args.limit_per_class)
