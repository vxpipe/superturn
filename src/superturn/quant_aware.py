"""Fake-quantized weights for an export recovery fine-tune."""

import torch
from torch import nn
from torch.nn.utils import parametrize


class _Int8Weight(nn.Module):
    def __init__(self, per_channel):
        super().__init__()
        self.per_channel = per_channel

    def forward(self, weight):
        with torch.no_grad():
            if self.per_channel:
                axes = tuple(range(1, weight.ndim))
                scale = (weight.abs().amax(dim=axes) / 127).clamp_min(1e-10)
            else:
                scale = (weight.abs().max() / 127).clamp_min(1e-10)
            zero = torch.zeros_like(scale, dtype=torch.int32)
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(weight, scale, zero, 0, -127, 127)
        return torch.fake_quantize_per_tensor_affine(weight, scale, zero, -127, 127)


def _quantize_linear_input(_layer, args):
    (inputs,) = args
    with torch.no_grad():
        lower, upper = inputs.aminmax()
        lower = lower.clamp_max(0)
        upper = upper.clamp_min(0)
        scale = ((upper - lower) / 255).clamp_min(1e-10)
        zero = (-lower / scale).round().clamp(0, 255).to(torch.int32)
    return (torch.fake_quantize_per_tensor_affine(inputs, scale, zero, 0, 255),)


def enable(model):
    for layer in list(model.modules()):
        if isinstance(layer, nn.Linear):
            layer.register_forward_pre_hook(_quantize_linear_input)
            parametrize.register_parametrization(layer, "weight", _Int8Weight(True))
        elif isinstance(layer, nn.Conv1d):
            parametrize.register_parametrization(layer, "weight", _Int8Weight(False))
        elif isinstance(layer, nn.GRU):
            names = [name for name, _ in layer.named_parameters(recurse=False) if name.startswith("weight_")]
            for name in names:
                parametrize.register_parametrization(layer, name, _Int8Weight(False))
    if hasattr(model, "positions"):
        parametrize.register_parametrization(model, "positions", _Int8Weight(False))


def plain_state_dict(model):
    """Save trainable parameters with the ordinary export model's names."""
    result = {}
    for key, tensor in model.state_dict().items():
        prefix, marker, param = key.rpartition("parametrizations.")
        if marker and param.endswith(".original"):
            key = prefix + param.removesuffix(".original")
        result[key] = tensor
    return result
