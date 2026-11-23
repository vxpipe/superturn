"""Replay frozen HarperValley development pauses through a local streaming ASR."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import re
import time

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .harper_text_inputs import prefix_turns
from .harper_valley_audit import audit_call
from .progress import Progress
from .sources import data_root
from .timed_text_arrival import PauseTrace, Revision, pack_trace_file, replay
from .turn_generation import normalize
from .turn_text_train import digest

MODEL = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
MODEL_REVISION = "672fbf1b30579d6585301139bb363f42a0ad4a24"
MODEL_FILES = {
    "encoder-epoch-99-avg-1-chunk-16-left-128.onnx": "a423883ce5754507fd941755ab0b5bc426a84ac670cbe21cf060e9e2c66dc660",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx": "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
    "joiner-epoch-99-avg-1-chunk-16-left-128.onnx": "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
    "tokens.txt": "7c025e7de139195720829e522dac7d1947ae0a321a8c74b7b13f2926aca49139",
}
RUNTIME = "1.12.23+cuda12.cudnn9"
FRAME = RATE // 50
NS_PER_SAMPLE = 1_000_000_000 // RATE
PRIOR_QUIET = RATE * 3 // 10
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
    digits_changed = set()
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
            track.feed(episodes[index].origin + offset)
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
            if cleaned != normalize(raw):
                digits_changed.add(index)
            if role == "agent":
                finals[index] = received, cleaned
                continue
            history = caller_revisions[index]
            if (history and history[-1].text != cleaned) or (not history and cleaned):
                history.append(Revision(len(history), received, "caller", cleaned))

    result = []
    for index, episode in enumerate(episodes):
        row = episode.row
        first = first_samples[index]
        previous_time, previous_text = finals.get(index, (None, ""))
        caller_start_ns = first + episode.caller_start * NS_PER_SAMPLE
        missing_prior = episode.agent_start is not None and (not previous_text or previous_time >= caller_start_ns)
        history = caller_revisions[index]
        captured_through = first + (episode.caller_start + len(episode.caller_pcm)) * NS_PER_SAMPLE
        # Revisions delivered after the final observed caller sample can
        # still arrive before a decision. The trace records true receipt.
        captured_through = max(captured_through, history[-1].received_ns if history else 0)
        trace = PauseTrace("caller", previous_text if not missing_prior else "", first, RATE,
                           episode.stop, episode.resume, captured_through)
        result.append({
            "id": pause_id(row),
            "label": row["label"],
            "trace": asdict(trace),
            "revisions": [asdict(revision) for revision in history],
            "prior_status": ("missing_at_caller_onset" if missing_prior else
                             "first_turn" if episode.agent_start is None else "final_before_caller_onset"),
            "prior_final_received_ns": previous_time,
            "caller_start_sample": episode.caller_start,
            "digit_cleaner_changed_text": index in digits_changed,
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
        "prior": "separate agent stream finalizes after 300 ms quiet and must arrive before caller onset",
        "caller": "separate caller stream opens on onset and ends on resumption or 800 ms after Silero stop",
        "cleaner": "repository normalizer plus ASCII digit-by-digit spelling",
    }
    settings_sha = sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    limits = "smoke" if limit is not None else "v1"
    trace_path = cache / f"text-dev-streaming-trace-{limits}.json"
    packed_path = cache / f"text-dev-streaming-{limits}.json"
    report_path = reports / ("harper-text-streaming-capture-smoke.json" if limit is not None else
                             "harper-text-streaming-capture.json")
    captured = []
    preparation = Progress("harper-streaming-capture", len(episodes), "pauses")
    pacing_lags = []
    # Length sorting avoids leaving idle batch slots behind very long episodes.
    episodes.sort(key=lambda ep: ep.caller_start + len(ep.caller_pcm) - ep.origin)
    for start in range(0, len(episodes), batch_size):
        batch = episodes[start:start + batch_size]
        events, max_lag = capture_batch(recognize, batch)
        captured.extend(events)
        pacing_lags.append(max_lag)
        preparation.advance(len(batch))
    preparation.finish()
    rec_by_id = {pause_id(row): row for row in chosen}
    counts = Counter()
    for event in captured:
        label = event["label"]
        counts[f"{label}/prior/{event['prior_status']}"] += 1
        counts[f"{label}/captured"] += 1
        if event["digit_cleaner_changed_text"]:
            counts[f"{label}/digit_cleaner_changed"] += 1
        if event["prior_status"] == "missing_at_caller_onset":
            event["scoring_exclusion"] = "missing_previous_stt"
        else:
            revisions = [Revision(**revision) for revision in event["revisions"]]
            decision = replay(PauseTrace(**event["trace"]), revisions).at_300_ms
            if decision and decision.text:
                counts[f"{label}/300_ms/text"] += 1
                if decision.text != rec_by_id[event["id"]]["text"]:
                    counts[f"{label}/300_ms/differs_from_word_cut"] += 1
    envelope = {"provider_settings_sha256": settings_sha, "frozen_pack_sha256": pack_sha, "pauses": captured}
    _write_json(trace_path, envelope)
    pack_trace_file(root, trace_path, packed_path)
    receipt = {
        "scope": "provisional HarperValley streaming-ASR condition, awaiting independent labels",
        "frozen_pack_sha256": pack_sha, "provider_settings_sha256": settings_sha, "provider_settings": settings,
        "trace_sha256": digest(trace_path), "packed_sha256": digest(packed_path),
        "trace_path": trace_path.relative_to(root).as_posix(), "packed_path": packed_path.relative_to(root).as_posix(),
        "rows": len(captured), "counts": dict(counts), "max_batch_pacing_lag_ms": max(pacing_lags),
    }
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
