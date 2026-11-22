"""Exportable acoustic features and small turn models."""

import math

import torch
from torch import nn
from torch.nn import functional as F

FFT = 512
FRAME = 400
HOP = 160


class AcousticFeatures(nn.Module):
    def __init__(self, active_stats=False, *, hop=HOP, spectral_rate=16000):
        super().__init__()
        if spectral_rate not in (8000, 16000):
            raise ValueError("the acoustic frontend supports 8 kHz or 16 kHz analysis")
        self.active_stats = active_stats
        self.hop = hop
        self.spectral_rate = spectral_rate
        self.decimation = 16000 // spectral_rate
        self.fft = FFT // self.decimation
        self.frame = FRAME // self.decimation
        self.frame_hop = hop // self.decimation
        self.lag_start, lag_stop = 40 // self.decimation, 230 // self.decimation
        frequency = torch.arange(129) * (16000 / FFT)
        def mel(hz):
            return 2595 * torch.log10(1 + hz / 700)

        edges = torch.linspace(0, mel(torch.tensor(4000.0)), 42)
        bins = mel(frequency).unsqueeze(1)
        filters = torch.minimum((bins - edges[:-2]) / (edges[1:-1] - edges[:-2]),
                                (edges[2:] - bins) / (edges[2:] - edges[1:-1])).clamp_min(0)
        self.register_buffer("mel", filters, persistent=False)
        self.register_buffer("window", torch.hann_window(self.frame), persistent=False)
        if self.decimation == 2:
            indices = torch.arange(-16, 17, dtype=torch.float32)
            taps = .45 * torch.sinc(indices * .45) * torch.hann_window(33, periodic=False)
            self.register_buffer("antialias", (taps / taps.sum()).reshape(1, 1, -1), persistent=False)
        if active_stats:
            self.activity_floor = 10 ** (-75 / 10)
            # Unnormalized, one-sided STFT power for the retained band.
            power_floor = self.activity_floor * (FFT / 2) * (3 * FRAME / 8)
            self.register_buffer("energy_floor", torch.tensor(power_floor), persistent=False)
            self.register_buffer("mel_floor", (power_floor * filters.sum(dim=0) / len(frequency)).clamp_min(1e-8),
                                 persistent=False)
        self.register_buffer("bins", torch.arange(129, dtype=torch.float32), persistent=False)
        self.register_buffer("lags", torch.arange(self.lag_start, lag_stop, dtype=torch.float32), persistent=False)
        padded = F.pad(self.window, (0, self.fft - self.frame))
        window_correlation = torch.stack([(padded * padded.roll(lag)).sum() for lag in range(self.lag_start, lag_stop)])
        self.register_buffer("lag_window", window_correlation / padded.square().sum(), persistent=False)

    def pitch_features(self, power):
        # DC and sub-pitch rumble can otherwise appear perfectly periodic.
        pitch_power = power[..., 2:]
        harmonics = (self.bins[2:, None] * self.lags * (2 * math.pi / self.fft)).cos()
        strength = pitch_power.sum(dim=-1, keepdim=True)
        correlation = (pitch_power @ harmonics) / strength.clamp_min(1e-8)
        corrected = (correlation / self.lag_window).clamp(-1, 1)
        lag_index = (corrected - self.lags * (0.0005 * self.decimation)).argmax(dim=-1, keepdim=True)
        pitch = self.spectral_rate / (lag_index.to(power.dtype) + self.lag_start)
        raw_peak = correlation.gather(-1, lag_index)
        floor = strength.amax(dim=1, keepdim=True) * 0.0003
        reliable = (strength >= floor.clamp_min(0.01)) & (raw_peak >= 0.18)
        confidence = corrected.gather(-1, lag_index).clamp(0, 1)
        return pitch, confidence * reliable.to(power.dtype)

    def analysis_audio(self, audio):
        centered = audio - audio.mean(dim=1, keepdim=True)
        if self.decimation == 2:
            return F.conv1d(centered.unsqueeze(1), self.antialias, stride=2, padding=16).squeeze(1)
        return centered

    def power_spectrum(self, audio):
        # Left-align the physical Hann window inside the FFT.
        padding = (0, self.fft - self.frame)
        window = F.pad(self.window, padding)
        padded = F.pad(self.analysis_audio(audio), padding)
        spectrum = torch.stft(padded, n_fft=self.fft, hop_length=self.frame_hop, window=window,
                              center=False, return_complex=False)
        power = spectrum[:, :129].square().sum(dim=-1).transpose(1, 2)
        return power if self.decimation == 1 else 4 * power

    def forward(self, audio):
        power = self.power_spectrum(audio)
        if self.active_stats:
            return self.active_features(audio, power)
        mel = (power @ self.mel).clamp_min(1e-8).log()
        energy = power.sum(dim=-1, keepdim=True).clamp_min(1e-8).log()
        pitch, voicing = self.pitch_features(power)
        pitch = pitch.log()
        valid = voicing >= 0.35
        weights = valid.to(audio.dtype)
        pitch_mean = (pitch * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1)
        pitch = (pitch - pitch_mean) * weights
        pitch = pitch / (pitch.square().mean(dim=1, keepdim=True) + 1e-4).sqrt()
        energy = energy - energy.mean(dim=1, keepdim=True)
        energy = energy / (energy.square().mean(dim=1, keepdim=True) + 1e-4).sqrt()
        return torch.cat((mel, pitch, voicing, energy), dim=-1)

    @staticmethod
    def normalize_active(values, active):
        count = active.sum(dim=1, keepdim=True).clamp_min(1)
        centered = (values - (values * active).sum(dim=1, keepdim=True) / count) * active
        variance = centered.square().sum(dim=1, keepdim=True) / count
        return centered / (variance + 1e-4).sqrt()

    def active_features(self, audio, power):
        framed = audio.unsqueeze(1)
        mean = F.avg_pool1d(framed, FRAME, self.hop)
        variance = F.avg_pool1d(framed.square(), FRAME, self.hop) - mean.square()
        active = (variance.transpose(1, 2) > self.activity_floor).to(power.dtype)
        # Base activity on the native frames before whole-window centering.
        power = power * active
        mel = torch.maximum(power @ self.mel, self.mel_floor).log()
        spectral_energy = power.sum(dim=-1, keepdim=True)
        energy = self.normalize_active(torch.maximum(spectral_energy, self.energy_floor).log(), active)
        pitch, voicing = self.pitch_features(power)
        pitch = self.normalize_active(pitch.log(), (voicing >= 0.35).to(power.dtype))
        return torch.cat((mel, pitch, voicing, energy), dim=-1)


class StemBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv1d(inputs, inputs, 5, stride=2, padding=2, groups=inputs),
                         nn.Conv1d(inputs, outputs, 1), nn.GELU())


class ConformerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.conv_norm = nn.LayerNorm(width)
        self.conv = nn.Sequential(nn.Conv1d(width, width, 7, groups=width, padding=3), nn.GELU(),
                                  nn.Conv1d(width, width, 1))
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))

    def forward(self, x):
        batch, frames, width = x.shape
        qkv = self.qkv(self.attn_norm(x)).reshape(batch, frames, 3, 4, width // 4)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        context = ((q @ k.transpose(-2, -1)) * (self.width // 4) ** -0.5).softmax(dim=-1) @ v
        x = x + self.projection(context.transpose(1, 2).reshape(batch, frames, width))
        x = x + self.conv(self.conv_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ff(x)


class TurnModel(nn.Module):
    def __init__(self, architecture, width, window, layers=None, gru_hidden_size=None, *,
                 active_stats=False, hop=HOP, spectral_rate=16000):
        super().__init__()
        self.architecture = architecture
        self.samples = round(window * 16000)
        self.features = AcousticFeatures(active_stats=active_stats, hop=hop, spectral_rate=spectral_rate)
        self.stem = nn.Sequential(StemBlock(43, width // 2), StemBlock(width // 2, width))
        output_width = width
        if architecture == "a":
            if gru_hidden_size is not None:
                output_width = gru_hidden_size
            self.sequence = nn.GRU(width, output_width, num_layers=layers or 2, batch_first=True)
        elif architecture == "b":
            frames = ((self.samples - FRAME) // hop + 4) // 4
            self.positions = nn.Parameter(torch.zeros(1, frames, width))
            self.sequence = nn.Sequential(*(ConformerBlock(width) for _ in range(layers or 3)))
            self.pool = nn.Linear(width, 1)
        else:
            raise ValueError(f"unknown architecture {architecture!r}")
        self.decision = nn.Sequential(nn.Linear(output_width + 1, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def decision_input(self, audio, silence_ms, *, return_state=False):
        x = self.stem(self.features(audio).transpose(1, 2)).transpose(1, 2)
        if self.architecture == "a":
            sequence, state = self.sequence(x)
            x = sequence[:, -1]
        else:
            state = None
            x = self.sequence(x + self.positions)
            x = (self.pool(x).softmax(dim=1) * x).sum(dim=1)
        features = torch.cat((x, silence_ms / 1000), dim=-1)
        return (features, state) if return_state else features

    def logits(self, audio, silence_ms):
        return self.decision(self.decision_input(audio, silence_ms))

    def forward(self, audio, silence_ms):
        return self.logits(audio, silence_ms).sigmoid()


def build(config):
    if config["architecture"] == "citrinet":
        from .citrinet import CitrinetTurnModel

        return CitrinetTurnModel(config)
    width = config.get("width")
    hop = config.get("hop", HOP)
    if hop not in (HOP, 2 * HOP) or (hop != HOP and config["architecture"] != "a"):
        raise ValueError("only GRU models support a 20 ms feature hop")
    gru_hidden_size = config.get("gru_hidden_size")
    if gru_hidden_size is not None and (width is None or config["architecture"] != "a"):
        raise ValueError("a custom GRU hidden size requires a fixed GRU stem width")
    if width is None:
        target = config.get("parameters", 150000)
        sizes = {value: parameter_count(config["architecture"], value, config["window"], config.get("layers"))
                 for value in range(32, 433, 4)}
        width = min((value for value, size in sizes.items() if size <= 1_000_000),
                    key=lambda value: abs(sizes[value] - target))
    return TurnModel(config["architecture"], width, config["window"], config.get("layers"), gru_hidden_size,
                     active_stats=config.get("active_stats", False), hop=hop,
                     spectral_rate=config.get("spectral_rate", 16000))


def parameter_count(architecture, width, window, layers=None):
    with torch.device("meta"):
        return sum(p.numel() for p in TurnModel(architecture, width, window, layers).parameters())
