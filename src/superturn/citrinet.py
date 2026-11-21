"""Audio-only hold/shift model initialized from the LibriSpeech Citrinet."""

import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import NamedTemporaryFile

import torch
from torch import nn

from .progress import Progress

REPOSITORY = "nvidia/stt_en_citrinet_256_ls"
REVISION = "c7cdacff0262bbb2a17771d20ec9f9f3b5cff6ed"
ARCHIVE = "stt_en_citrinet_256_ls.nemo"
SIZE = 36_693_216
SHA256 = "a00c8fb75a2bcc37271275293bce950863e4e069d43f6cf193d42fcf605aa21c"
NEMO_VERSION = "3.0.0"


def fetch(root):
    from huggingface_hub import hf_hub_url

    from .acquire import _request

    if not (root / ".campaign-machine").is_file():
        raise ValueError("the pinned encoder is downloaded on the campaign machine")
    path = root / "audit-models" / "citrinet-256-ls" / REVISION / ARCHIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == SIZE:
        with path.open("rb") as archive:
            if hashlib.file_digest(archive, "sha256").hexdigest() == SHA256:
                return path

    url = hf_hub_url(REPOSITORY, ARCHIVE, revision=REVISION)
    meter = Progress("fetch-citrinet-256-ls", SIZE, "B")
    temporary = None
    try:
        digest = hashlib.sha256()
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with _request(url) as response:
                downloaded = 0
                while block := response.read(1 << 20):
                    downloaded += len(block)
                    if downloaded > SIZE:
                        raise ValueError("Citrinet download exceeds the pinned size")
                    output.write(block)
                    digest.update(block)
                    meter.advance(len(block))
        if downloaded != SIZE or digest.hexdigest() != SHA256:
            raise ValueError("Citrinet download differs from the pinned checkpoint")
        temporary.replace(path)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def archive_contents(path):
    """Return the trusted archive's encoder configuration and tensor state."""
    from nemo.utils.model_utils import maybe_update_config_version
    from omegaconf import OmegaConf

    names = {"model_config.yaml", "model_weights.ckpt"}
    with tarfile.open(path, "r:*") as archive:
        members = {}
        for member in archive:
            name = PurePosixPath(member.name).name
            if name in names:
                if name in members or not member.isfile():
                    raise ValueError(f"invalid Citrinet archive member {name}")
                members[name] = member
        if members.keys() != names:
            raise ValueError("incomplete Citrinet archive")
        with archive.extractfile(members["model_config.yaml"]) as source:
            settings = maybe_update_config_version(OmegaConf.load(BytesIO(source.read())))
        with archive.extractfile(members["model_weights.ckpt"]) as source:
            state = torch.load(BytesIO(source.read()), weights_only=True, map_location="cpu")
    config = OmegaConf.to_container(settings, resolve=True)
    modules = {name: dict(config[name]) for name in ("preprocessor", "encoder")}
    targets = {
        "preprocessor": "AudioToMelSpectrogramPreprocessor",
        "encoder": "ConvASREncoder",
    }
    for name, target in targets.items():
        if modules[name].pop("_target_").rsplit(".", 1)[-1] != target:
            raise ValueError(f"unsupported Citrinet {name} class")
    if modules["preprocessor"].get("sample_rate") != 16000:
        raise ValueError("Citrinet checkpoint must use 16 kHz PCM")
    return modules, {name: tensor for name, tensor in state.items() if name.split(".")[0] in modules}


def initialize(model, root, expected_modules, *, pretrained):
    modules, state = archive_contents(fetch(root))
    if modules != expected_modules:
        raise ValueError("Citrinet checkpoint configuration and model differ")
    for name in ("preprocessor", "encoder") if pretrained else ("preprocessor",):
        prefix = name + "."
        getattr(model, name).load_state_dict({key.removeprefix(prefix): value for key, value in state.items()
                                             if key.startswith(prefix)})


class CitrinetTurnModel(nn.Module):
    def __init__(self, config):
        from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
        from nemo.collections.asr.modules.conv_asr import ConvASREncoder

        super().__init__()
        self.samples = round(config["window"] * 16000)
        settings = config["citrinet_modules"]
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**settings["preprocessor"])
        self.encoder = ConvASREncoder(**settings["encoder"])
        width = settings["encoder"]["jasper"][-1]["filters"]
        self.decision = nn.Sequential(nn.Linear(2 * width + 1, 128), nn.GELU(), nn.Linear(128, 1))

    def decision_input(self, audio, silence_ms):
        if audio.shape[1] != self.samples:
            raise ValueError("Citrinet expects the fixed PCM prefix ending at the decision")
        lengths = torch.full((audio.shape[0],), self.samples, dtype=torch.long, device=audio.device)
        features, lengths = self.preprocessor(input_signal=audio, length=lengths)
        encoded, lengths = self.encoder(audio_signal=features, length=lengths)
        steps = torch.arange(encoded.shape[-1], device=encoded.device)
        mask = steps[None, :] < lengths[:, None]
        mean = (encoded * mask[:, None, :]).sum(dim=-1) / lengths[:, None]
        last_index = (lengths[:, None, None] - 1).expand(-1, encoded.shape[1], 1)
        last = encoded.gather(-1, last_index).squeeze(-1)
        return torch.cat((mean, last, silence_ms / 1000), dim=-1)

    def logits(self, audio, silence_ms):
        return self.decision(self.decision_input(audio, silence_ms))

    def forward(self, audio, silence_ms):
        return self.logits(audio, silence_ms).sigmoid()
