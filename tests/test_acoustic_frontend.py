import pytest

torch = pytest.importorskip("torch")

from superturn.model import AcousticFeatures
from superturn.spectral import ConvolutionSpectrum


def test_half_rate_retains_frame_geometry_and_export_spectrum():
    generator = torch.Generator().manual_seed(119)
    pcm = torch.randn((2, 48000), generator=generator) * .1
    native = AcousticFeatures(active_stats=True, spectral_rate=8000)
    export = ConvolutionSpectrum(active_stats=True, spectral_rate=8000)

    power = native.power_spectrum(pcm)

    assert power.shape == (2, 298, 129)
    assert native(pcm).shape == (2, 298, 43)
    torch.testing.assert_close(power, export.power_spectrum(pcm), rtol=1e-4, atol=2e-4)


def test_half_rate_convolution_export_preserves_analysis_rate(tmp_path):
    import json

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from superturn.evaluate import export
    from superturn.model import build

    config = {"architecture": "a", "window": 3, "width": 32, "spectral_rate": 8000}
    model = build(config)
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": config, "state": model.state_dict(), "metrics": {"threshold": .5},
                "parameters": sum(value.numel() for value in model.parameters())}, checkpoint)

    output = export(checkpoint, limit=False, spectrum="convolution")

    receipt = json.loads(output.with_name("export.json").read_text())
    assert receipt["config"]["spectral_rate"] == 8000
    assert receipt["float_max_abs_error"] < 1e-3
