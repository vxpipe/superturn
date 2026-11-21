"""Exercise the pinned encoder adapter with generated audio and a small archive."""

from io import BytesIO
import json
import tarfile

import pytest


def test_archive_initialization_and_pcm_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("nemo.collections.asr.modules")
    from superturn import citrinet

    modules = {
        "preprocessor": {"sample_rate": 16000, "dither": 0, "features": 16},
        "encoder": {"feat_in": 16, "activation": "relu", "jasper": [
            {"filters": 16, "repeat": 1, "kernel": [5], "stride": [2], "dilation": [1],
             "dropout": 0.0, "residual": False, "separable": True},
        ]},
    }
    config = {"window": 1, "citrinet_modules": modules}
    source = citrinet.CitrinetTurnModel(config)
    weights = BytesIO()
    torch.save({key: value for key, value in source.state_dict().items() if not key.startswith("decision.")},
               weights)
    model_config = {
        "preprocessor": {"_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                         **modules["preprocessor"]},
        "encoder": {"_target_": "nemo.collections.asr.modules.ConvASREncoder", **modules["encoder"]},
    }
    path = tmp_path / "generated.nemo"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in (("model_config.yaml", json.dumps(model_config).encode()),
                              ("model_weights.ckpt", weights.getvalue())):
            entry = tarfile.TarInfo(f"./{name}")
            entry.size = len(payload)
            archive.addfile(entry, BytesIO(payload))

    monkeypatch.setattr(citrinet, "fetch", lambda root: path)
    pretrained = citrinet.CitrinetTurnModel(config)
    random = citrinet.CitrinetTurnModel(config)
    for model, initialize_encoder in ((pretrained, True), (random, False)):
        citrinet.initialize(model, tmp_path, modules, pretrained=initialize_encoder)
        for key, tensor in source.preprocessor.state_dict().items():
            torch.testing.assert_close(model.preprocessor.state_dict()[key], tensor)
    for key, tensor in source.encoder.state_dict().items():
        torch.testing.assert_close(pretrained.encoder.state_dict()[key], tensor)
    assert any(not torch.equal(expected, actual) for expected, actual in
               zip(source.encoder.parameters(), random.encoder.parameters(), strict=True))

    audio = torch.randn(2, 16000) / 10
    elapsed = torch.full((2, 1), 300.0)
    scores = pretrained.eval()(audio, elapsed)
    assert scores.shape == (2, 1) and scores.isfinite().all()
    torch.testing.assert_close(pretrained(audio[:1], elapsed[:1]), scores[:1])
    with pytest.raises(ValueError, match="fixed PCM prefix"):
        pretrained(audio[:, :4000], elapsed)

    loss = pretrained.train().logits(audio, elapsed).square().mean()
    loss.backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all()
               for parameter in pretrained.parameters())
