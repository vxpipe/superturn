"""Export-only spectral frontend using a fixed real Fourier filter bank."""

import math

import torch
from torch.nn import functional as F

from .model import AcousticFeatures, HOP


class ConvolutionSpectrum(AcousticFeatures):
    """Compute the original first 129 windowed DFT bins at the configured hop."""

    def __init__(self, active_stats=False, *, hop=HOP, spectral_rate=16000):
        super().__init__(active_stats=active_stats, hop=hop, spectral_rate=spectral_rate)
        phase = (-2 * math.pi / self.fft) * torch.arange(129, dtype=torch.float64)[:, None] * torch.arange(self.frame)
        kernels = torch.stack((phase.cos(), phase.sin()), dim=1).reshape(-1, 1, self.frame)
        kernels = (kernels * self.window).float()
        self.register_buffer("fourier_kernels", kernels, persistent=False)

    def power_spectrum(self, audio):
        analysis = self.analysis_audio(audio)
        spectrum = F.conv1d(analysis.unsqueeze(1), self.fourier_kernels, stride=self.frame_hop)
        power = spectrum.reshape(audio.shape[0], 129, 2, -1).square().sum(dim=2).transpose(1, 2)
        return power if self.decimation == 1 else 4 * power
