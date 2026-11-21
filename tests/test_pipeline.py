from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from superturn.audio import write_wave
from superturn.corpora import indicative_speech, read_pauses, spontaneous_pauses
from superturn.labels import Label
from superturn.shards import CONTEXT, TAIL, make_shards


def test_single_track_and_timestamp_labels():
    row = {"annotated_transcript": [{"speaker_id": "s", "start_time": 0.5, "end_time": 1.5}]}
    assert indicative_speech(row)[0].end == 24000
    assert list(spontaneous_pauses([(1000, 10000), (14000, 24000)], 28000)) == [
        (10000, 4000, "hold", None), (24000, 4000, "weak-shift", None)
    ]
    words = [(3000, 5000, .9), (5500, 7500, .9), (8000, 9000, .9), (10000, 15000, .9)]
    assert list(read_pauses([(0, 16000)], 22000, words)) == [
        (16000, 6000, "shift", None), (7500, 5680, "hold", 16320)
    ]


def _generated_rows(root):
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        for label, hz in (("hold", 160), ("shift", 260)):
            voice = f"fixture-{index}-{label}"
            audio = Path("generated") / f"{voice}.wav"
            time = np.arange(16000) / 16000
            pcm = np.concatenate(((15000 * np.sin(2 * np.pi * hz * time)).astype("int16"),
                                  np.resize(np.array([-9, 11], dtype="int16"), 8000)))
            write_wave(root / audio, pcm)
            rows.append(Label("fixture", "en", voice, voice, voice, split, 16000, 8000,
                              label, "generated", str(audio)))
    return rows


def test_generated_shards_are_fixed_and_frozen(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq

    monkeypatch.setenv("SUPERTURN_DATA", str(tmp_path))
    rows = _generated_rows(tmp_path)
    make_shards(tmp_path, rows)
    manifest = (tmp_path / "shards" / "manifest.json").read_text()
    make_shards(tmp_path, list(reversed(rows)))
    assert (tmp_path / "shards" / "manifest.json").read_text() == manifest
    assert json.loads(manifest)["examples_by_split"] == {"test": 2, "train": 2, "validation": 2}
    for name in ("clean", "telephone", "telephone-noise"):
        table = pq.read_table(tmp_path / "shards" / "test" / name / "000000.parquet")
        assert len(table["pcm"][0].as_py()) == (CONTEXT + TAIL) * 2


@pytest.mark.parametrize("architecture", ["a", "b"])
def test_generated_training_and_int8_export(tmp_path, monkeypatch, architecture):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from superturn.evaluate import benchmark, export, quantization_auc
    from superturn.train import completed_run, train

    from dataclasses import replace

    monkeypatch.setenv("SUPERTURN_DATA", str(tmp_path))
    make_shards(tmp_path, [replace(row, source="ami") for row in _generated_rows(tmp_path)])
    config = {"architecture": architecture, "parameters": 150000, "window": 1,
              "epochs": 1, "augmentation": False}
    checkpoint = train(tmp_path, "fixture", config, "cpu")
    assert completed_run(tmp_path, "fixture", config, "cpu") == checkpoint
    assert completed_run(tmp_path, "fixture", {**config, "epochs": 2}, "cpu") is None
    result = export(checkpoint)
    assert result.stat().st_size < 1000000
    assert all(auc is not None for auc in quantization_auc(tmp_path, result, 0.5))
    report = benchmark(tmp_path, result, 0.5, samples=3)
    assert all(row["300"]["auc"] is not None for row in report["groups"])
    assert report["acceptance"]["artifact"]["passed"]
    assert "conversation_telephone_300ms" in report["acceptance"]


def test_pitch_on_known_reference_tone():
    torch = pytest.importorskip("torch")
    from superturn.model import AcousticFeatures

    frontend = AcousticFeatures()
    for hz in (100, 160, 230, 340):
        tone = torch.sin(2 * np.pi * hz * torch.arange(400) / 16000)
        power = torch.fft.rfft(tone * frontend.window, 512).abs().square()[:129]
        pitch, voicing = frontend.pitch_features(power.reshape(1, 1, -1))
        assert abs(pitch.item() - hz) / hz < 0.06
        assert voicing.item() > 0.65


def test_experiment_queue_continues_after_failed_run(tmp_path, monkeypatch):
    import importlib
    import json

    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from superturn.pipeline import experiments

    evaluate = importlib.import_module("superturn.evaluate")
    trainer = importlib.import_module("superturn.train")
    monkeypatch.setenv("SUPERTURN_DATA", str(tmp_path))
    runs, benchmarks = {}, []

    def train(root, name, config, device):
        runs[name] = config
        if name == "s1-a-150000-17":
            raise RuntimeError("fixture training failure")
        path = root / "runs" / name / "best.pt"
        path.parent.mkdir(parents=True)
        torch.save({"metrics": {"auc": .9, "recall": .8, "loss": .5}}, path)
        return path

    def export(path):
        path.with_name("export.json").write_text(json.dumps({"threshold": .6}))
        return path.with_name("int8.onnx")

    def quantization_auc(root, path, threshold):
        destination = root / "reports" / f"{path.parent.name}-int8-validation"
        destination.mkdir(parents=True)
        (destination / "metrics.json").write_text(json.dumps({"threshold": .61}))
        return .90, .895

    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr(evaluate, "export", export)
    monkeypatch.setattr(evaluate, "quantization_auc", quantization_auc)
    monkeypatch.setattr(evaluate, "benchmark", lambda root, path, threshold: benchmarks.append(threshold))
    assert experiments(tmp_path, {"epochs": 1}, "cpu") == 1
    assert len(runs) == 34 and benchmarks == [.61] * 3
    report = json.loads((tmp_path / "reports" / "experiments" / "queue.json").read_text())
    assert len(report["models"]) == 3
    assert report["failures"] == [["s1-a-150000-17", "fixture training failure"]]


def test_input_window_includes_real_elapsed_pause():
    pytest.importorskip("torch")
    from superturn.train import _input

    pcm = np.zeros(CONTEXT + TAIL, dtype="<i2")
    pcm[CONTEXT - 10:CONTEXT] = 11
    pcm[CONTEXT:CONTEXT + 3200] = 22
    row = {"pcm": pcm.tobytes(), "pause_samples": 3200}
    assert list(_input(row, 3202, 200)[:3]) == [11, 11, 22]
    assert _input(row, 3202, 200)[-1] == 22
    with pytest.raises(ValueError):
        _input(row, 3202, 300)


def test_source_locales_share_a_common_language_code():
    from superturn.corpora import language_code

    assert {language_code(value) for value in ("Hindi", "hi_IN", "hi")} == {"hi"}
    assert language_code("cmn_hans_cn") == "zh"
    assert language_code("Bodo") == "brx"


def test_emilia_yodas_reads_observed_provider_id(tmp_path):
    import json

    from superturn.corpora import _emilia_clips

    clip = tmp_path / "DE" / "clip"
    clip.parent.mkdir()
    clip.with_suffix(".json").write_text(json.dumps({"_id": "clip-id", "language": "de", "speaker": "voice"}))
    clip.with_suffix(".mp3").touch()
    assert list(_emilia_clips(tmp_path))[0][:4] == ("clip-id", "clip-id", "voice", "de")


def test_listening_selection_is_order_independent_and_capped(tmp_path):
    from superturn.listening import choose

    rows = _generated_rows(tmp_path)
    examples = dict(choose(rows, 3))
    assert examples == dict(choose(reversed(rows), 3))
    assert list(examples) == ["en-001", "en-002", "en-003"]


def test_yin_reference_distinguishes_tone_and_silence():
    pytest.importorskip("torch")
    from superturn.pitch_check import yin_pitch

    tone = np.sin(2 * np.pi * 160 * np.arange(400) / 16000)
    assert yin_pitch(tone) == 160
    assert np.isnan(yin_pitch(np.zeros_like(tone)))


def test_listening_gate_requires_matching_human_judgments(tmp_path):
    from dataclasses import replace
    import csv

    from superturn.listening import prepare, require_gate, summarize

    rows = [replace(row, method="per-track") for row in _generated_rows(tmp_path)]
    prepare(tmp_path, rows, per_language=3)
    packet = tmp_path / "labels" / "listening"
    with pytest.raises(ValueError, match="record the listening"):
        require_gate(tmp_path, per_language=3)
    import json
    answers = json.loads((packet / "key.json").read_text())
    with (packet / "review.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["example", "heard", "notes"])
        writer.writerows((key, row["label"], "") for key, row in answers.items())
    with pytest.raises(ValueError, match="partial review judgments"):
        prepare(tmp_path, rows, per_language=3)
    summarize(tmp_path, packet / "review.csv")
    require_gate(tmp_path, per_language=3)
    with pytest.raises(ValueError, match="below 4"):
        require_gate(tmp_path, per_language=4)
