"""Score LiveKit's local text reference on the two frozen English diagnostics."""

import argparse
import asyncio
from collections import defaultdict
from importlib.metadata import version
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path

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
FILES = (*PINNED_HASHES, "config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json")


def model_files(root, *, download=False):
    from huggingface_hub import hf_hub_download

    meter = Progress("livekit-text-reference-stage", len(FILES), "files") if download else None
    paths = {}
    try:
        for name in FILES:
            path = Path(hf_hub_download(MODEL, name, revision=TAG, cache_dir=root / "cache/livekit-text-hub",
                                        token=False, local_files_only=not download))
            parents = len(Path(name).parts) - 1
            if path.parents[parents].name != REVISION:
                raise ValueError("the publisher's LiveKit model tag changed")
            paths[name] = path
            if meter is not None:
                meter.advance(1)
        hashes = {name: digest(path) for name, path in paths.items()}
        if any(hashes[name] != value for name, value in PINNED_HASHES.items()):
            raise ValueError("the pinned LiveKit weights, license or operating point changed")
    except Exception:
        if meter is not None:
            meter.finish("failed")
        raise
    if meter is not None:
        meter.finish()
    threshold = float(json.loads(paths["languages.json"].read_text())["en"]["threshold"])
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


async def score(root):
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
    publisher, freeze = publisher_rows(root)
    harper, harper_checksum = harper_rows(root)
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
              "timing": "Existing frozen transcript inputs, not a streaming STT replay",
              "decision_guard": "Reference only. Excluded from training, threshold fitting and model selection.",
              "publisher_freeze_sha256": digest(freeze), "harper_input_sha256": harper_checksum}
    await executor.start()
    try:
        await executor.initialize()
        model = LocalTextModel("multilingual", inference_executor=executor)
        for name, rows in (("publisher", publisher), ("harper", harper)):
            groups = defaultdict(list)
            callers = defaultdict(set)
            calls = defaultdict(set)
            meter = Progress(f"livekit-text-reference-{name}", len(rows), "decisions")
            try:
                for row in rows:
                    context = chat_context(row["text"], llm.ChatContext())
                    probability = await model.predict_end_of_turn(context)
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("LiveKit returned an invalid turn probability")
                    for band in ("english", gap_band(row["gap_ms"])):
                        groups[band].append((probability, row["label"]))
                        callers[band].add(row["caller"])
                        calls[band].add(row["call"])
                    meter.advance(1)
            except Exception:
                meter.finish("failed")
                raise
            meter.finish()
            counts = ({"publisher_turns": len(calls["english"])} if name == "publisher"
                      else {"calls": len(calls["english"]), "callers": len(callers["english"])})
            result[name] = {"elapsed_s": meter.elapsed(), "cohort": counts,
                            "views": {band: binary_metrics(values, threshold) for band, values in groups.items()}}
    finally:
        await executor.aclose()
    _write_json(root / "reports/livekit-text-reference.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="download only the pinned reference assets")
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
        asyncio.run(score(root))
