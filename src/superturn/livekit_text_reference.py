"""Score LiveKit's local text reference on the two frozen English diagnostics."""

import argparse
import asyncio
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .harper_text_review import gap_band
from .progress import Progress
from .sources import data_root
from .timed_text_length_trial import publisher_rows
from .turn_text_train import LABELS, binary_metrics, digest

MODEL = "livekit/turn-detector"
TAG = "v0.4.1-intl"
REVISION = "87e35fcb1e60a569bea70346191c4886ea92e281"
RUNTIME = "1.8.13"
ONNX = "onnx/model_q8.onnx"
PINNED_HASHES = {
    ONNX: "bd2c30776882138a1d95a07faddc13756fe1a35bef6323505f1124fca349bc9c",
    "LICENSE": "4f5b74db4fd26299a166688de8a3bde046fd7d9cc13ecab64532c24310bc18d7",
    "languages.json": "05a16e11e4282173331dbf140624048e581a727ba45c361bed35e3014ad68187",
}
FILES = ("LICENSE", "languages.json", "config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json", ONNX)


def model_files(root, *, download=False):
    from huggingface_hub import hf_hub_download

    meter = Progress("livekit-text-reference-stage", len(FILES), "files") if download else None
    hashes = {}
    try:
        for name in FILES:
            path = Path(hf_hub_download(MODEL, name, revision=TAG, cache_dir=root / "cache/livekit-text-hub",
                                        token=False, local_files_only=not download))
            parents = len(Path(name).parts) - 1
            if path.parents[parents].name != REVISION:
                raise ValueError("the publisher's LiveKit model tag changed")
            checksum = digest(path)
            if name in PINNED_HASHES and checksum != PINNED_HASHES[name]:
                raise ValueError("the pinned LiveKit weights, license or operating point changed")
            hashes[name] = checksum
            if name == "languages.json":
                threshold = float(json.loads(path.read_text())["en"]["threshold"])
            if meter is not None:
                meter.advance(1)
    except Exception:
        if meter is not None:
            meter.finish("failed")
        raise
    if meter is not None:
        meter.finish()
    return hashes, threshold


def harper_rows(root):
    path = root / "cache/harper-valley/text-dev-v1.jsonl"
    receipt = json.loads((root / "reports/harper-text-dev-packing.json").read_text())
    checksum = digest(path)
    if checksum != receipt["serialized_sha256"] or receipt["scope"] != "causal English development inputs awaiting label qualification":
        raise ValueError("the frozen HarperValley development inputs changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (len(rows) != receipt["rows"]
            or len({(row["call"], row["caller_sample"]) for row in rows}) != len(rows)
            or {row["label"] for row in rows} != set(LABELS)):
        raise ValueError("the frozen HarperValley cohort changed")
    return [{**row, "label": LABELS[row["label"]]} for row in rows], checksum


def chat_context(text, context):
    previous, current = text.split("\n")
    if previous != "---":
        context.add_message(role="assistant", content=previous)
    context.add_message(role="user", content=current)
    return context


def streaming_rows(root, frozen, input_sha):
    receipt_path = root / "reports/harper-text-streaming-capture.json"
    receipt = json.loads(receipt_path.read_text())
    trace = root / "cache/harper-valley/text-dev-streaming-trace-v1.json"
    packed = root / "cache/harper-valley/text-dev-streaming-v1.json"
    if (receipt["frozen_pack_sha256"] != input_sha or receipt["rows"] != len(frozen)
            or digest(trace) != receipt["trace_sha256"] or digest(packed) != receipt["packed_sha256"]):
        raise ValueError("the full Harper streaming capture does not match the frozen reference inputs")
    capture = json.loads(packed.read_text())
    if (capture["input_sha256"] != receipt["trace_sha256"]
            or capture["provider_settings_sha256"] != receipt["provider_settings_sha256"]):
        raise ValueError("the Harper streaming capture settings changed")

    def identity(row):
        return sha256(f"{row['call']}:{row['caller_sample']}".encode()).hexdigest()

    events = capture["pauses"]
    by_id = {event["id"]: event for event in events}
    if len(events) != len(by_id) or set(by_id) != {identity(row) for row in frozen}:
        raise ValueError("streaming and word-cut reference membership differs")
    rows = []
    for row in frozen:
        event = by_id[identity(row)]
        if LABELS[event["label"]] != row["label"]:
            raise ValueError("streaming and word-cut reference labels differ")
        rows.append({**row, "capture": event})
    return rows, digest(receipt_path)


async def summarize_streaming(rows, prediction, threshold):
    views = defaultdict(list)
    counts = Counter()
    triggered_shifts, committed_shifts = [], []
    meter = Progress("livekit-text-reference-streaming", len(rows), "pauses")
    try:
        for row in rows:
            event = row["capture"]
            label = event["label"]
            counts[label] += 1
            counts[f"{label}/trace_end/{event['ended_by']}"] += 1
            if not row["label"] and row["gap_ms"] > 800:
                counts["hold/lasts_past_800_ms"] += 1
            scores = {}
            first = event["at_300_ms"]
            if first is not None and first["text"]:
                text = first["text"]
                scores[text] = await prediction(text)
                for view in ("english", gap_band(row["gap_ms"])):
                    views[view].append((scores[text], row["label"]))
                counts[f"{label}/has_300_ms_score"] += 1
            elif exclusion := event.get("scoring_exclusion"):
                counts[f"{label}/300_ms/{exclusion}"] += 1
            else:
                counts[f"{label}/300_ms/{'unobserved' if first is None else 'no_text'}"] += 1
            trigger = None
            if any(tick["text"] for tick in event["ticks"]):
                counts[f"{label}/has_scoring_tick"] += 1
            for tick in event["ticks"]:
                text = tick["text"]
                if text is None:
                    continue
                if text not in scores:
                    scores[text] = await prediction(text)
                if scores[text] >= threshold:
                    trigger = tick["since_stop_ns"] / 1_000_000
                    break
            if trigger is not None:
                counts[f"{label}/model_trigger_before_timeout"] += 1
                if row["label"]:
                    triggered_shifts.append(trigger)
                    committed_shifts.append(trigger)
            elif row["label"] and event["ended_by"] == "timeout":
                committed_shifts.append(800)
            meter.advance(1)
    except Exception:
        meter.finish("failed")
        raise
    meter.finish()

    def latency(values):
        return {"shifts": len(values), "p50_ms": float(np.median(values)) if values else None,
                "p90_ms": float(np.quantile(values, .9)) if values else None}

    return {"at_300_ms": {name: binary_metrics(views[name], threshold)
                          for name in ("english", "300_to_800_ms", "over_800_ms")},
            "coverage_and_sequence_counts": dict(counts),
            "hold_false_trigger_rate": counts["hold/model_trigger_before_timeout"] / counts["hold"],
            "hold_false_trigger_rate_when_scored": counts["hold/model_trigger_before_timeout"]
            / counts["hold/has_scoring_tick"] if counts["hold/has_scoring_tick"] else None,
            "shift_first_model_trigger": latency(triggered_shifts),
            "shift_commit_with_observed_800_ms_fallback": latency(committed_shifts),
            "elapsed_s": meter.elapsed()}


async def score(root, *, streaming=False):
    from livekit.agents import llm
    from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor
    from livekit.plugins.turn_detector.base import EOUModelBase
    from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual

    for package in ("livekit-agents", "livekit-plugins-turn-detector"):
        if version(package) != RUNTIME:
            raise ValueError("the LiveKit Agents reference runtime changed")
    if _EUORunnerMultilingual.model_revision() != TAG:
        raise ValueError("the Agents plugin does not resolve the pinned multilingual text model")

    class LocalTextModel(EOUModelBase):
        def _inference_method(self):
            return _EUORunnerMultilingual.INFERENCE_METHOD

    hashes, threshold = model_files(root)
    harper, harper_checksum = harper_rows(root)
    if streaming:
        harper, capture_checksum = streaming_rows(root, harper, harper_checksum)
    else:
        publisher, freeze = publisher_rows(root)
    executor = InferenceProcExecutor(
        runners={_EUORunnerMultilingual.INFERENCE_METHOD: _EUORunnerMultilingual},
        initialize_timeout=300, close_timeout=5, memory_warn_mb=2000, memory_limit_mb=0,
        ping_interval=5, ping_timeout=60, high_ping_threshold=2.5,
        mp_ctx=multiprocessing.get_context("spawn"), loop=asyncio.get_running_loop(), http_proxy=None,
    )
    result = {"scope": "English-only reference diagnostics on provisional frozen inputs",
              "model": MODEL, "model_tag": TAG, "revision": REVISION, "agents_runtime": RUNTIME,
              "threshold_source": "released languages.json English threshold",
              "license": "LiveKit Model License", "asset_sha256": hashes,
              "decision_guard": "Reference only. Excluded from training, threshold fitting and model selection.",
              "harper_input_sha256": harper_checksum}
    name = "livekit-text-streaming-reference" if streaming else "livekit-text-reference"
    if streaming:
        result.update(scope="English-only reference on provisional Harper streaming inputs",
                      timing="Captured client receipt times for local English streaming ASR",
                      harper_streaming_capture_receipt_sha256=capture_checksum)
    else:
        result.update(timing="Existing frozen transcript inputs, not a streaming STT replay",
                      publisher_freeze_sha256=digest(freeze))
    await executor.start()
    try:
        await executor.initialize()
        model = LocalTextModel("multilingual", inference_executor=executor)

        async def prediction(text):
            context = chat_context(text, llm.ChatContext())
            probability = await model.predict_end_of_turn(context)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("LiveKit returned an invalid turn probability")
            return probability

        if streaming:
            result["harper"] = await summarize_streaming(harper, prediction, threshold)
        for source, rows in () if streaming else (("publisher", publisher), ("harper", harper)):
            groups = defaultdict(list)
            callers = defaultdict(set)
            calls = defaultdict(set)
            meter = Progress(f"livekit-text-reference-{source}", len(rows), "decisions")
            try:
                for row in rows:
                    probability = await prediction(row["text"])
                    for band in ("english", gap_band(row["gap_ms"])):
                        groups[band].append((probability, row["label"]))
                        callers[band].add(row["caller"])
                        calls[band].add(row["call"])
                    meter.advance(1)
            except Exception:
                meter.finish("failed")
                raise
            meter.finish()
            counts = ({"publisher_turns": len(calls["english"])} if source == "publisher"
                      else {"calls": len(calls["english"]), "callers": len(callers["english"])})
            result[source] = {"elapsed_s": meter.elapsed(), "cohort": counts,
                              "views": {band: binary_metrics(values, threshold) for band, values in groups.items()}}
    finally:
        await executor.aclose()
    _write_json(root / f"reports/{name}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--stage", action="store_true", help="download only the pinned reference assets")
    modes.add_argument("--streaming", action="store_true", help="score the full receipt-bound Harper streaming capture")
    args = parser.parse_args()
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the LiveKit reference runs only on Vast")
    os.environ["HF_HUB_CACHE"] = str(root / "cache/livekit-text-hub")
    if args.stage:
        model_files(root, download=True)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.pop("LIVEKIT_REMOTE_EOT_URL", None)
        logging.getLogger("livekit").setLevel(logging.WARNING)
        asyncio.run(score(root, streaming=args.streaming))
