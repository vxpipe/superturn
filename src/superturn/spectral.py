"""Export-only spectral frontend using a fixed real Fourier filter bank."""

import math

import torch
from torch.nn import functional as F

from .model import AcousticFeatures, FFT, FRAME, HOP


class ConvolutionSpectrum(AcousticFeatures):
    """Compute the original first 129 windowed DFT bins at the configured hop."""

    def __init__(self, active_stats=False, *, hop=HOP):
        super().__init__(active_stats=active_stats, hop=hop)
        phase = (-2 * math.pi / FFT) * torch.arange(129, dtype=torch.float64)[:, None] * torch.arange(FRAME)
        kernels = torch.stack((phase.cos(), phase.sin()), dim=1).reshape(-1, 1, FRAME)
        kernels = (kernels * self.window).float()
        self.register_buffer("fourier_kernels", kernels, persistent=False)

    def power_spectrum(self, audio):
        centered = audio - audio.mean(dim=1, keepdim=True)
        spectrum = F.conv1d(centered.unsqueeze(1), self.fourier_kernels, stride=self.hop)
        return spectrum.reshape(audio.shape[0], 129, 2, -1).square().sum(dim=2).transpose(1, 2)
