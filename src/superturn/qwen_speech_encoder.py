"""Stage the published Qwen3-ASR 1.7B audio tower without its decoder."""

import hashlib
import json
import struct
from urllib.request import Request, urlopen

import numpy as np
import torch

from .ami_calibration import _write_json
from .audio import RATE
from .dense_teacher import SAMPLES, file_digest
from .eot_domain_probe import GAP
from .progress import Progress
from .sources import data_root

REPO = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
MODEL_BYTES = 4_076_193_080
MODEL_ETAG = '"082792968b53db8c7c1d695f8fa0f75d9aef41ed4f79ffb3e61c32d0f988e9f2"'
HEADER_BYTES = 88_120
HEADER_SHA256 = "2332189910c12f20e73be544d67a1ce2e1c7ef6de7b5d665b45a79dcdf4e7d73"
AUDIO_BYTES = 628_657_408
METADATA = {
    "config.json": (2399, "117ac8e63e2af7cae3665e5a632d6eb03f5f384915519ceb6403c15ec6533f63"),
    "processor_config.json": (487, "bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a"),
}


def read_range(start, end):
    # Distinct URL query values prevent a cached redirect for one range from
    # carrying another range's CDN signature.
    request = Request(f"{URL}/model.safetensors?download=true&range={start}-{end}",
                      headers={"Range": f"bytes={start}-{end}"})
    response = urlopen(request, timeout=60)
    if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{MODEL_BYTES}"
            or response.headers.get("ETag") != MODEL_ETAG
            or response.headers.get("Content-Length") != str(end - start + 1)):
        response.close()
        raise ValueError("the publisher did not return the exact pinned audio byte range")
    return response


def stage_tower(folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "audio-tower.safetensors"
    receipt = folder / "audio-tower.json"
    release = {"repo": REPO, "revision": REVISION, "source_etag": MODEL_ETAG, "payload_bytes": AUDIO_BYTES,
               "source_header_sha256": HEADER_SHA256}
    if target.is_file() and receipt.is_file():
        provenance = json.loads(receipt.read_text())
        if provenance != {**release, "sha256": file_digest(target)}:
            raise ValueError("the cached Qwen audio tower differs from its receipt")
        return target, provenance

    with read_range(0, HEADER_BYTES - 1) as response:
        header = response.read(HEADER_BYTES)
    if hashlib.sha256(header).hexdigest() != HEADER_SHA256:
        raise ValueError("Qwen safetensors header changed")
    prefix = "model.audio_tower."
    weights = {key.removeprefix(prefix): value for key, value in json.loads(header[8:]).items()
               if key.startswith(prefix)}
    offsets = sorted(item["data_offsets"] for item in weights.values())
    if (len(offsets) != 393 or {p["dtype"] for p in weights.values()} != {"BF16"}
            or offsets[0][0] != 0 or offsets[-1][1] != AUDIO_BYTES
            or any(left[1] != right[0] for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("Qwen audio tensors no longer occupy the identified contiguous prefix")
    packed = json.dumps(weights, separators=(",", ":")).encode()
    packed += b" " * (-len(packed) % 8)
    temporary = target.with_suffix(".part")
    meter = Progress("qwen-audio-checkpoint", AUDIO_BYTES, "bytes")
    try:
        with read_range(HEADER_BYTES, HEADER_BYTES + AUDIO_BYTES - 1) as response, temporary.open("wb") as stream:
            stream.write(struct.pack("<Q", len(packed)))
            stream.write(packed)
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
                meter.advance(len(chunk))
        if temporary.stat().st_size != 8 + len(packed) + AUDIO_BYTES:
            raise ValueError("the Qwen audio tower transfer was incomplete")
        provenance = {**release, "sha256": file_digest(temporary)}
        temporary.replace(target)
        _write_json(receipt, provenance)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return target, provenance


def qwen_audio(root):
    import transformers
    from safetensors.torch import load_file
    from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig
    from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
    from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder

    if transformers.__version__ != "5.24.0":
        raise ValueError("unexpected Qwen3-ASR implementation version")
    folder = root / "cache/open-speech-encoder/qwen3-asr-1.7b"
    target, provenance = stage_tower(folder)
    metadata = {}
    for name, (size, digest) in METADATA.items():
        path = folder / name
        if not path.is_file():
            with urlopen(f"{URL}/{name}", timeout=20) as response:
                data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Qwen {name} differs from the published revision")
            path.write_bytes(data)
        if file_digest(path) != digest:
            raise ValueError(f"cached Qwen {name} changed")
        metadata[name] = json.loads(path.read_text())
    acoustic = metadata["config.json"]["audio_config"]
    config = Qwen3ASREncoderConfig.from_dict(acoustic, attn_implementation="eager")
    model = Qwen3ASREncoder(config)
    model.load_state_dict(load_file(target), strict=True)
    model.eval().to("cuda")
    frontend = Qwen3ASRFeatureExtractor.from_dict(metadata["processor_config.json"]["feature_extractor"])
    centers = (np.arange(3)[:, None] * 100 + np.arange(13) * 8).ravel() * 160
    pre = (centers >= SAMPLES - GAP - RATE // 2) & (centers < SAMPLES - GAP)
    post = centers >= SAMPLES - GAP

    def predict(audio):
        features = frontend(audio.cpu().numpy(), sampling_rate=RATE, padding="longest", truncation=False)
        spec = features.input_features.to(audio.device)
        valid = features.attention_mask.to(audio.device)
        if spec.shape[1:] != (128, 300) or not bool(valid.all()):
            raise ValueError("Qwen front end did not render the exact three-second audio")
        states = []
        for mel, mask in zip(spec, valid, strict=True):
            output = model(mel[None], mask[None], output_hidden_states=True)
            layers = torch.stack((*output.hidden_states[:-1], output.last_hidden_state)).float()
            if layers.shape != (25, 39, 1024):
                raise ValueError(f"unexpected Qwen acoustic frame geometry: {layers.shape}")
            states.append(torch.cat((layers.mean(1), layers[:, pre].mean(1), layers[:, post].mean(1)), dim=-1))
        return torch.stack(states)

    with torch.inference_mode():
        tone = torch.sin(torch.arange(SAMPLES, device="cuda", dtype=torch.float32) / 13)[None, :]
        quiet = tone * 1e-3
        if not torch.allclose(predict(torch.cat((tone, quiet))), torch.cat((predict(tone), predict(quiet))),
                              rtol=1e-4, atol=1e-5):
            raise ValueError("Qwen audio features changed when clips were batched together")
    return predict, {**provenance, "transformers": transformers.__version__, "frame_centers_samples": centers.tolist()}


if __name__ == "__main__":
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the Qwen audio tower is staged only on Vast")
    torch.set_num_threads(2)
    _, provenance = qwen_audio(root)
    print(json.dumps(provenance, indent=2))
