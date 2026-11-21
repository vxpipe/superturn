"""Blinded audio-language review of English per-track labels."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from .progress import Progress
from .sources import data_root

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
REVISION = "0a095220c30b7b31434169c3086508ef3ea5bf0a"


def fetch(root):
    from huggingface_hub import HfApi, hf_hub_url

    from .acquire import _request

    model_dir = root / "audit-models" / "qwen2-audio" / REVISION
    model_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    files = [file for file in info.siblings if file.rfilename.endswith((".safetensors", ".json", ".txt", ".jinja"))]
    meter = Progress("fetch-audio-judge", sum(file.size for file in files), "B")

    def download(file):
        path = model_dir / file.rfilename
        if path.is_file() and path.stat().st_size == file.size:
            meter.skip(file.size)
            return
        partial = path.with_suffix(path.suffix + ".partial")
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset == file.size:
            meter.skip(offset)
            partial.replace(path)
            return
        headers = {"Range": f"bytes={offset}-"} if 0 < offset < file.size else {}
        url = hf_hub_url(MODEL, file.rfilename, revision=REVISION)
        with _request(url, headers=headers) as response:
            resumed = bool(headers) and response.status == 206
            if resumed:
                meter.skip(offset)
            with partial.open("ab" if resumed else "wb") as output:
                while block := response.read(1 << 20):
                    output.write(block)
                    meter.advance(len(block))
        if partial.stat().st_size != file.size:
            raise ValueError(f"truncated audio judge file {file.rfilename}")
        partial.replace(path)

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(download, files))
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return model_dir


def _signals(root, row, read):
    import numpy as np

    from .audio import RATE

    path = root / row.audio
    pcm = read(path)
    edge = row.pause_start
    stop = edge + row.pause_samples
    left = max(0, stop - 1920)
    right = min(len(pcm), stop + 4 * RATE)
    tracks = [read(other)[left:right].astype("float32")
              for other in sorted(path.parent.glob("*.Headset-*.wav")) if other != path]
    mix = np.sum(tracks, axis=0) / max(1, len(tracks) ** 0.5)
    signals = [pcm[max(0, edge - 2 * RATE):edge].astype("float32"),
               pcm[left:right].astype("float32"), np.asarray(mix, dtype="float32")]
    return [np.clip(audio / 32768, -1, 1) for audio in signals]


PROMPT = (
    "Judge one English meeting pause. Recording 1 is the caller's headset just before "
    "the pause. Recordings 2 and 3 cover the same four seconds just after the pause. "
    "Recording 2 is the caller's headset and recording 3 combines the other headsets. "
    "Nearby microphones can pick up the same voice, so compare who has the clear speech. "
    "If the caller continues, including after a brief acknowledgement, choose A. "
    "If a different speaker takes the turn, choose B. If the recordings do not "
    "establish who continues, choose C. Answer with only A, B, or C."
)


def _reviewer(model_dir, device):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    if processor.feature_extractor.sampling_rate != 16000:
        raise ValueError("audio judge sampling rate must be 16 kHz")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16, device_map=device,
    ).eval()
    content = [{"type": "text", "text": PROMPT}]
    for index in range(3):
        content.extend(({"type": "text", "text": f"Recording {index + 1}."},
                        {"type": "audio", "audio_url": f"track-{index + 1}"}))
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=False)

    choices = ("hold", "shift", "unclear")
    ids = [processor.tokenizer.encode(" " + character, add_special_tokens=False) for character in "ABC"]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("the audio judge requires a single-token answer per category")
    choice_ids = [tokens[0] for tokens in ids]

    def review(signals):
        batch = processor(text=prompt, audio=signals, sampling_rate=16000,
                          padding=True, return_tensors="pt").to(device)
        batch["input_features"] = batch["input_features"].to(dtype=torch.bfloat16)
        with torch.inference_mode():
            scores = model(**batch, use_cache=False).logits[0, -1, choice_ids]
        return choices[int(scores.argmax())]

    return review


def run(root, device):
    from collections import Counter
    from functools import lru_cache
    import json

    from .ami_reference import audit_sample
    from .audio import read_audio
    from .labels import read_rows
    from .listening import choose
    from .shards import seed_for

    raw_labels = read_rows(root / "labels" / "ami.jsonl")
    targets = [row for _, row in choose(raw_labels, 50)]
    pool = [row for _, row in choose(
        [row for row in raw_labels if row.split == "train" and row not in targets], 800,
    )]
    manual = audit_sample(root, pool)
    anchors = [
        row for label in ("hold", "shift")
        for row in [example for example in pool if example.label == label and manual[example] == "supported"][:20]
    ]
    distribution = Counter(row.label for row in anchors)
    if any(distribution[label] < 10 for label in ("hold", "shift")):
        raise ValueError(f"only {dict(distribution)} independent calibration examples")
    model_dir = root / "audit-models" / "qwen2-audio" / REVISION
    review = _reviewer(model_dir, device)
    read = lru_cache(maxsize=16)(read_audio)
    meter = Progress("audio-language-review", len(anchors) + len(targets), "pauses")
    summaries = {}
    cases = []
    try:
        for scope, examples in (("calibration", anchors), ("per_track", targets)):
            summary = Counter()
            for row in examples:
                answer = review(_signals(root, row, read))
                outcome = ("unresolved" if answer == "unclear"
                           else "supported" if answer == row.label else "contradicted")
                identity = seed_for("label-audit", row.source, row.audio, row.speaker, row.pause_start)
                cases.append({"scope": scope, "id": f"{identity:016x}", "label": row.label,
                              "judgment": answer, "outcome": outcome})
                summary[outcome] += 1
                meter.advance(1)
            summaries[scope] = dict(summary)
        report = {"model": MODEL, "revision": REVISION, "blind": True,
                  "decision": "single-token likelihood among three choices",
                  "groups": summaries, "cases": cases}
        destination = root / "reports" / "audio-language-review-choice.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


def main():
    parser = ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.fetch:
        print(fetch(data_root()))
        return 0
    print(run(data_root(), args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
