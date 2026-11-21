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
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    import shutil

    from superturn.evaluate import benchmark, export, quantization_auc
    from superturn.train import completed_run, train

    from dataclasses import replace

    monkeypatch.setenv("SUPERTURN_DATA", str(tmp_path))
    rows = [replace(row, source="ami") for row in _generated_rows(tmp_path)]
    make_shards(tmp_path, rows)
    config = {"architecture": architecture, "parameters": 150000, "window": 1,
              "epochs": 1, "augmentation": architecture == "a", "augmentation_workers": 2}
    checkpoint = train(tmp_path, "fixture", config, "cpu")
    assert completed_run(tmp_path, "fixture", config, "cpu") == checkpoint
    assert completed_run(tmp_path, "fixture", {**config, "epochs": 2}, "cpu") is None
    result = export(checkpoint)
    assert result.stat().st_size < 1000000
    assert all(auc is not None for auc in quantization_auc(tmp_path, result, 0.5))
    report = benchmark(tmp_path, result, 0.5, samples=3)
    assert all(row["300"]["auc"] is not None for row in report["groups"])
    assert report["latency_ms"]["condition"] == "telephone"
    assert report["acceptance"]["artifact"]["passed"]
    assert "conversation_telephone_300ms" in report["acceptance"]
    alternative = tmp_path / "alternative" / "best.pt"
    alternative.parent.mkdir()
    shutil.copyfile(checkpoint, alternative)
    accelerated = export(alternative, spectrum="convolution")
    assert accelerated.stat().st_size < 1_000_000
    graph = onnx.load(accelerated).graph
    assert all(node.op_type != "STFT" for node in graph.node)
    convolution_weights = {node.input[1] for node in graph.node if node.op_type == "Conv"}
    assert any(node.op_type == "Constant" and node.output[0] in convolution_weights
               and any(attr.t.data_type == onnx.TensorProto.FLOAT and tuple(attr.t.dims) == (258, 1, 400)
                       for attr in node.attribute)
               for node in graph.node)
    if architecture == "a":
        import importlib
        import json

        evaluate = importlib.import_module("superturn.evaluate")
        with monkeypatch.context() as patch:
            patch.setattr(evaluate, "_measure_latency", lambda *args: pytest.fail("quality gate did not defer latency"))
            deferred = benchmark(tmp_path, result, 1.0, samples=3, name="quality-first", quality_first=True)
            validation = benchmark(tmp_path, result, 1.0, split="validation", name="validation-only", quality_first=True)
        assert deferred["latency_ms"] is None
        assert validation["latency_ms"] is None
        assert deferred["acceptance"]["cpu_p99"] == {"passed": None, "milliseconds": None}
        assert not deferred["acceptance"]["conversation_telephone_300ms"]["passed"]
        assert "CPU latency deferred" in (tmp_path / "reports/quality-first/report.md").read_text()
        with monkeypatch.context() as patch:
            patch.setattr(evaluate, "_acceptance",
                          lambda path, report: {"conversation_telephone_300ms": {"passed": True},
                                                "cpu_p99": {"passed": None, "milliseconds": None}})
            patch.setattr(evaluate, "_measure_latency", lambda *args: {"p50": 1.0, "p99": 1.5})
            qualified = benchmark(tmp_path, result, .5, name="qualified", quality_first=True)
        assert qualified["acceptance"]["cpu_p99"] == {"passed": True, "milliseconds": 1.5}

        from superturn import ami_benchmark
        from superturn.labels import write_rows
        from superturn.volume import benchmark_volume

        sensitivity = benchmark_volume(tmp_path, result, 0.5, per_label_language=1)
        assert sensitivity["examples"] == 2
        assert len(sensitivity["qualifying_lowest_tested_db"]) == 3
        assert all(level is None for level in sensitivity["qualifying_lowest_tested_db"].values())
        write_rows(tmp_path / "labels" / "ami.jsonl", rows)
        references = tmp_path / "raw" / "ami_annotations"
        references.mkdir(parents=True)
        (references / "fixture.zip").write_bytes(b"mocked annotations")

        def supported(root, offered, *, pause_limits):
            assert all(row.split == "test" for row in offered)
            pause_limits.update((row, 4800) for row in offered)
            return dict.fromkeys(offered, "supported")

        monkeypatch.setattr(ami_benchmark, "audit_sample", supported)
        ami_benchmark.prepare(tmp_path)
        selected = tmp_path / "reports" / "experiments"
        selected.mkdir()
        (selected / "selection.json").write_text(json.dumps({
            "split": "validation", "candidates": [{"name": "fixture", "threshold": .5}],
        }))
        ami_benchmark.benchmark_selected(tmp_path)
        reference = json.loads((tmp_path / "reports" / "fixture" / "ami-reference-test.json").read_text())
        assert reference["threshold"] == .5 and reference["examples"] == 2
        assert len(reference["groups"]) == 3
        assert all(row["holds"] == 1 and row["shifts"] == 1 for row in reference["groups"])
        shutil.copytree(checkpoint.parent, tmp_path / "runs" / "s1-fixture")

        def supported_validation(root, offered, *, pause_limits):
            assert all(row.split == "validation" for row in offered)
            pause_limits.update((row, 4800) for row in offered)
            return dict.fromkeys(offered, "supported")

        monkeypatch.setattr(ami_benchmark, "audit_sample", supported_validation)
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            comparison = json.loads(ami_benchmark.compare_validation(tmp_path).read_text())
            assert torch.get_num_threads() == 2
        finally:
            torch.set_num_threads(prior_threads)
        assert comparison["split"] == "validation"
        assert [row["name"] for row in comparison["checkpoints"]] == ["s1-fixture"]
        assert all(row["annotation_supported"]["holds"] == 1 and row["annotation_supported"]["shifts"] == 1
                   for row in comparison["checkpoints"][0]["thresholds"])
        phone = ami_benchmark.AmiTelephoneValidation(tmp_path).int8_export(result, .5)
        assert phone["condition"] == "telephone"
        assert phone["thresholds"][1]["all_ami"]["shifts"] == 1


@pytest.mark.parametrize("architecture", ["a", "b"])
def test_quantization_recovery_checkpoint_exports(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from superturn.evaluate import export
    from superturn.train import completed_run, train

    make_shards(tmp_path, _generated_rows(tmp_path))
    config = {"architecture": architecture, "parameters": 150000, "window": 1,
              "epochs": 1, "augmentation": False}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = train(tmp_path, "baseline", config, device)
    retry = {**config, "lr": 0.0001, "warm_start": "baseline", "quantization_aware": True}
    recovery = train(tmp_path, "recovery", retry, device)
    assert completed_run(tmp_path, "recovery", retry, device) == recovery
    before = torch.load(baseline, map_location="cpu", weights_only=True)["state"]
    after = torch.load(recovery, map_location="cpu", weights_only=True)["state"]
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert export(recovery).stat().st_size < 1000000


def test_ami_review_filters_training_without_rebuilding_shards(tmp_path, monkeypatch):
    from dataclasses import replace
    import json
    from superturn import ami_training
    from superturn.labels import write_rows

    rows = [replace(row, source="ami") for row in _generated_rows(tmp_path)]
    make_shards(tmp_path, rows)
    manifest = tmp_path / "shards" / "manifest.json"
    before = manifest.read_bytes()
    write_rows(tmp_path / "labels" / "ami.jsonl", rows)
    references = tmp_path / "raw" / "ami_annotations"
    references.mkdir(parents=True)
    (references / "fixture.zip").write_bytes(b"mocked annotations")

    def audit(root, offered, **kwargs):
        assert all(row.split == "train" for row in offered)
        assert kwargs["end_ms"] == 300 and kwargs["onset_ms"] == 375
        kwargs["pause_limits"].update((row, 4000) for row in offered if row.label == "hold")
        return {row: "supported" if row.label == "hold" else "contradicted" for row in offered}

    monkeypatch.setattr(ami_training, "audit_sample", audit)
    summary = ami_training.prepare(tmp_path)
    assert summary["sharded_labels"] == {"hold": 1}
    accepted = next(row for row in rows if row.split == "train" and row.label == "hold")
    assert summary["shorter_usable_pause"] == 1
    assert ami_training.load_selection(tmp_path) == {(accepted.audio, accepted.pause_start, "hold"): 4000}
    assert manifest.read_bytes() == before
    stale = json.loads(before)
    stale["sha256_by_split"]["train"] = "stale"
    manifest.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="different training shard set"):
        ami_training.load_selection(tmp_path)


def test_ami_training_retains_only_reviewed_pauses():
    pytest.importorskip("torch")
    from superturn.train import eligible

    kept = {("fixture.wav", 16000, "hold"): 4000}
    row = {"source": "ami", "audio": "fixture.wav", "pause_start": 16000, "label": "hold"}
    assert eligible(row, {}, kept)
    assert not eligible({**row, "label": "shift"}, {}, kept)
    assert eligible({**row, "source": "common_voice"}, {}, kept)


def test_decision_percentiles_include_censored_shifts():
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from superturn.evaluate import _summarize

    report = _summarize([(400, 1, [.7, .1, .1]), (400, 1, [.1] * 3),
                         (400, 1, [.1] * 3), (400, 1, [.1] * 3),
                         (500, 0, [.1] * 4)], threshold=.5)
    assert report["decision_ms"]["recall"] == .25
    assert report["decision_ms"]["unresolved_shifts"] == 3
    assert report["decision_ms"]["p50"] is None
    assert report["decision_ms"]["p90"] is None


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


@pytest.mark.parametrize("conversation_ranking", (False, True))
def test_experiment_queue_continues_after_failed_run(tmp_path, monkeypatch, conversation_ranking):
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
    if conversation_ranking:
        from superturn import ami_benchmark, listening, pipeline, shards, volume

        (tmp_path / ".campaign-machine").touch()
        (tmp_path / "shards").mkdir()
        (tmp_path / "shards" / "manifest.json").write_text(json.dumps({"speaker_report": []}))
        monkeypatch.setattr(shards, "require_split_coverage", lambda report, sources: None)
        monkeypatch.setattr(listening, "require_gate", lambda root: None)
        monkeypatch.setattr(pipeline, "_run_sync", lambda root: None)
        monkeypatch.setattr(volume, "benchmark_volume", lambda root, path, threshold: None)

        class TelephoneValidation:
            scope = {"split": "validation", "source": "ami", "condition": "telephone"}

            def __init__(self, root):
                pass

            def report(self, recall):
                return {"thresholds": [{"origin": ami_benchmark.AMI_THRESHOLD_ORIGIN,
                                        "threshold": .64, "all_ami": {"recall": recall, "auc": .9}}]}

            def float_checkpoint(self, checkpoint):
                name = checkpoint.parent.name
                recall = .9 if name.startswith("s4-") else .85 if name.startswith("s1-b-500000-") else .7
                return self.report(recall)

            def int8_export(self, artifact, pooled_threshold):
                assert pooled_threshold == .61
                return self.report(.92 if artifact.parent.name == "s4-4" else .85)

        monkeypatch.setattr(ami_benchmark, "AmiTelephoneValidation", TelephoneValidation)

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
        recall = .92 if path.parent.name == "s4-2" else .8
        (destination / "metrics.json").write_text(json.dumps({
            "threshold": .61, "groups": [{"source": "all", "language": "all",
                                         "300": {"recall": recall, "auc": .895}}],
        }))
        return .90, .87 if path.parent.name == "s4-1" else .895

    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr(evaluate, "export", export)
    monkeypatch.setattr(evaluate, "quantization_auc", quantization_auc)
    monkeypatch.setattr(evaluate, "benchmark",
                        lambda root, path, threshold, *, quality_first: benchmarks.append((threshold, quality_first)))
    assert experiments(tmp_path, {"epochs": 1}, "cpu", stage_one_only=True) == 1
    assert len(runs) == 12 and all(name.startswith("s1-") for name in runs)
    assert benchmarks == []
    receipt = tmp_path / "reports" / "experiments" / "stage1-queue.json"
    assert json.loads(receipt.read_text())["failures"] == [["s1-a-150000-17", "fixture training failure"]]
    assert not receipt.with_name("selection.json").exists()
    for path in (tmp_path / "runs").glob("*/best.pt"):
        path.unlink()
        path.parent.rmdir()
    runs.clear()
    assert experiments(tmp_path, {"epochs": 1}, "cpu", stage_two_only=True) == 1
    assert len(runs) == 20 and sum(name.startswith("s2-") for name in runs) == 8
    assert benchmarks == []
    stage2 = json.loads(receipt.with_name("stage2-ranking.json").read_text())
    assert {row["seconds"] for row in stage2["windows"]} == {1, 2, 4, 8}
    assert not receipt.with_name("selection.json").exists()
    for path in (tmp_path / "runs").glob("*/best.pt"):
        path.unlink()
        path.parent.rmdir()
    runs.clear()
    assert experiments(tmp_path, {"epochs": 1}, "cpu") == 1
    expected_threshold = .64 if conversation_ranking else .61
    assert len(runs) == 35 and benchmarks == [(expected_threshold, True)] * 3
    assert runs["s4-1-qat"]["warm_start"] == "s4-1"
    assert runs["s4-1-qat"]["quantization_aware"]
    report = json.loads((tmp_path / "reports" / "experiments" / "queue.json").read_text())
    assert len(report["models"]) == 3
    assert report["models"][0] == ("s4-4" if conversation_ranking else "s4-2")
    assert report["failures"] == [["s1-a-150000-17", "fixture training failure"]]
    if conversation_ranking:
        assert (runs["s2-0-1s"]["architecture"], runs["s2-0-1s"]["parameters"]) == ("b", 500000)
        stage1 = json.loads((tmp_path / "reports" / "experiments" / "stage1-ranking.json").read_text())
        assert stage1["advanced"][0] == "s1-b-500000-17"
        selection = json.loads((tmp_path / "reports" / "experiments" / "selection.json").read_text())
        assert selection["source"] == "ami" and selection["condition"] == "telephone"
        assert selection["candidates"][0]["threshold"] == expected_threshold


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


def test_codec_augmentations_return_mono_int16():
    from superturn.augment import _opus_roundtrip, _pitch_shift

    pcm = (10000 * np.sin(2 * np.pi * 160 * np.arange(16000) / 16000)).astype("int16")
    opus = _opus_roundtrip(pcm)
    pitch = _pitch_shift(pcm, 1.1)
    assert opus.dtype == pitch.dtype == np.dtype("int16")
    assert len(opus) == len(pcm)
    assert 0.95 * len(pcm) <= len(pitch) <= 1.05 * len(pcm)
    assert np.any(opus) and np.any(pitch)


def test_training_augmentation_preserves_seeded_pcm():
    from superturn.training_augmentation import TrainingAugmentation

    rng = np.random.default_rng(29)
    voices = {("fixture", str(i)): rng.integers(-5000, 5000, 48000, dtype=np.int16)
              for i in range(4)}
    items = list(voices.items())
    jobs = [(items[seed % len(items)][1], seed, items[seed % len(items)][0], 200 * 16)
            for seed in range(32)]
    serial = TrainingAugmentation(1, ([], [], voices))
    parallel = TrainingAugmentation(2, ([], [], voices))
    try:
        assert np.array_equal(serial.render(jobs), parallel.render(jobs))
    finally:
        parallel.close()


def test_checkpoint_equivalence_binds_both_source_hashes(tmp_path, monkeypatch):
    import importlib
    import json

    pytest.importorskip("torch")
    train = importlib.import_module("superturn.train")
    monkeypatch.setattr(train, "REPO_ROOT", tmp_path)
    equivalent = tmp_path / "configs" / "training-source-equivalence.json"
    equivalent.parent.mkdir()
    equivalent.write_text(json.dumps({"previous": "threaded", "current": "process"}))
    recorded = {"config": {"seed": 17}, "source": "threaded", "shards": "same"}
    expected = {**recorded, "source": "process"}
    assert train._compatible_signature(recorded, expected)
    assert not train._compatible_signature(recorded, {**expected, "source": "other"})
    assert not train._compatible_signature(recorded, {**expected, "shards": "other"})
    assert not train._compatible_signature(expected, recorded)
    equivalent.write_text(json.dumps({"previous": ["threaded", "process"], "current": "fixed-epoch"}))
    updated = {**expected, "source": "fixed-epoch"}
    assert train._compatible_signature(recorded, updated)
    assert train._compatible_signature(expected, updated)
    fixed = {"fixed_epochs": True, "seed": 17}
    assert not train._compatible_signature({**expected, "config": fixed}, {**updated, "config": fixed})


def test_pitch_rejects_constant_offset_and_low_energy_noise():
    torch = pytest.importorskip("torch")
    from superturn.model import AcousticFeatures

    frontend = AcousticFeatures()
    rng = torch.Generator().manual_seed(19)
    audio = torch.stack((torch.full((16000,), .05), torch.randn(16000, generator=rng) * .0001))
    _, confidence = frontend.pitch_features(frontend.power_spectrum(audio))
    assert torch.all(confidence < .35)
