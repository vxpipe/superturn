"""Command-line entry points for preparation and the first campaign."""

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

from .labels import ami_labels, read_rows, write_rows
from .sources import REPO_ROOT, data_root, load_sources
from .splits import stratified_groups

LABEL_SOURCES = ("ami", "indic_diarbench", "fleurs", "common_voice", "voxpopuli", "emilia_yodas")


def _device(value):
    if value == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


READ_SOURCES = {"common_voice", "fleurs"}
PARALLEL_SOURCES = READ_SOURCES | {"emilia_yodas"}


def _init_label_worker(device, read):
    from .audio import silero
    from .corpora import whisper_aligner

    global _worker_vad, _worker_align
    _worker_vad = silero()
    _worker_align = whisper_aligner(device) if read else None


def _label_locale(root, source, locale):
    from .corpora import corpus_labels

    return list(corpus_labels(root, source, _worker_vad, _worker_align, locale))


def _locale_cache(root, source, device):
    manifest = root / "raw" / source / "MANIFEST.json"
    if not manifest.is_file():
        return None
    digest = hashlib.sha256(manifest.read_bytes())
    for filename in ("audio.py", "corpora.py", "labels.py"):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    digest.update(device.encode())
    for package in ("av", "openai-whisper", "silero-vad", "torch"):
        try:
            digest.update(version(package).encode())
        except PackageNotFoundError:
            digest.update(b"unavailable")
    return root / "labels" / ".locale-cache" / source / digest.hexdigest()


def _parallel_labels(root, source, device, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    from .corpora import parallel_locales
    from .progress import Progress

    locales = parallel_locales(root / "raw" / source, source)
    if not locales:
        raise ValueError(f"{source}: no downloaded locales")
    meter = Progress(f"label-{source}", len(locales), "languages")
    cache = _locale_cache(root, source, device)
    by_locale = {}
    pending = []
    failures = []
    try:
        for locale in locales:
            cached = cache / f"{locale}.jsonl" if cache is not None else None
            if cached is not None and cached.is_file():
                by_locale[locale] = read_rows(cached)
                meter.skip(1)
            else:
                pending.append(locale)
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                     initializer=_init_label_worker, initargs=(device, source in READ_SOURCES)) as pool:
                futures = {pool.submit(_label_locale, root, source, locale): locale for locale in pending}
                for future in as_completed(futures):
                    locale = futures[future]
                    try:
                        rows = future.result()
                    except Exception as error:
                        failures.append(locale)
                        print(f"{source}/{locale} failed: {error}", flush=True)
                    else:
                        if cache is not None:
                            write_rows(cache / f"{locale}.jsonl", rows)
                        by_locale[locale] = rows
                        meter.advance(1)
        if failures:
            raise RuntimeError(f"{source}: {len(failures)} failed locales ({', '.join(sorted(failures))})")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return [row for locale in locales for row in by_locale[locale]]


def label(root, sources, device, workers=1):
    from .audio import silero
    from .corpora import corpus_labels, whisper_aligner

    if workers < 1:
        raise ValueError("label workers must be positive")
    detect = align = None
    registry = load_sources()
    for source in sources:
        output = root / "labels" / f"{source}.jsonl"
        if source in PARALLEL_SOURCES and workers > 1:
            labels = _parallel_labels(root, source, device, workers)
        else:
            if detect is None and source != "indic_diarbench":
                detect = silero()
            if align is None and source in READ_SOURCES:
                align = whisper_aligner(device)
            rows = ami_labels(root, detect) if source == "ami" else corpus_labels(root, source, detect, align)
            labels = list(rows)
        splits = stratified_groups(source, registry[source].role, ((row.group, row.language) for row in labels))
        labels = [replace(row, split=splits[row.group]) for row in labels]
        write_rows(output, labels)
        coverage = defaultdict(lambda: defaultdict(set))
        for row in labels:
            coverage[row.language][row.split].add(row.group)
        print(json.dumps({
            language: {split: len(groups) for split, groups in sorted(by_split.items())}
            for language, by_split in sorted(coverage.items())
        }), flush=True)
        print(f"{source}: {len(labels)} labels, {len({row.speaker for row in labels if row.speaker != 'unknown'})} known speakers",
              flush=True)
        if not labels:
            raise ValueError(f"{source}: no pause labels")


def shard(root, sources, sample=False):
    from .shards import make_shards

    rows = [row for source in sources for row in read_rows(root / "labels" / f"{source}.jsonl")]
    make_shards(root, rows, sample=sample)


def smoke(root, device, *, acquire=False):
    from .cli import main
    from .evaluate import benchmark, export
    from .progress import Progress
    from .train import train

    meter = Progress("smoke-pipeline", 4, "stages")
    try:
        if acquire:
            for command in (["download", "--sample"], ["check-licences"],
                            ["inspect-archive", "https://www.openslr.org/resources/17/musan.tar.gz", "--mb", "50"]):
                if main(command):
                    raise RuntimeError(f"smoke acquisition command failed: {command[0]}")
        if acquire:
            meter.advance(1)
        else:
            meter.skip(1)
        meter.report()
        label(root, LABEL_SOURCES, device)
        meter.advance(1)
        meter.report()
        shard(root, LABEL_SOURCES, sample=True)
        config = {"architecture": "a", "window": 1, "parameters": 150000,
                  "epochs": 1, "max_steps": 3, "seed": 17, "smoke": True, "augmentation": False}
        artifact = export(train(root, "smoke", config, device))
        meter.advance(1)
        meter.report()
        details = json.loads(artifact.with_name("export.json").read_text())
        benchmark(root, artifact, details["threshold"], samples=25, name="smoke")
        meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()


def _run_sync(root):
    if (root / ".campaign-machine").is_file():
        subprocess.run([str(REPO_ROOT / "bin" / "sync"), "push", "runs", "reports"],
                       cwd=REPO_ROOT, check=True)


def _ranking_key(item):
    metrics = item[2]
    telephone = metrics.get("ami_telephone", metrics)
    return (telephone["recall"] or 0, telephone["auc"] or 0,
            metrics["recall"] or 0, metrics["auc"] or 0, -metrics["loss"])


def _rank(results, count=1):
    good = sorted(results, key=_ranking_key, reverse=True)
    if not good:
        raise ValueError("all runs in stage failed")
    return good[:count]


def experiments(root, base, device):
    import torch
    from .ami_benchmark import AMI_THRESHOLD_ORIGIN, AmiTelephoneValidation
    from .evaluate import benchmark, export, quantization_auc
    from .progress import Progress
    from .train import completed_run, train
    from .volume import benchmark_volume

    telephone_validation = None
    if (root / ".campaign-machine").is_file():
        from .listening import require_gate
        from .shards import require_split_coverage

        manifest = json.loads((root / "shards" / "manifest.json").read_text())
        registry = load_sources()
        require_split_coverage(manifest["speaker_report"], {name: registry[name].role for name in LABEL_SOURCES})
        if base.get("ami_verified_training", False):
            from .ami_training import load_selection

            selection = load_selection(root)
            print(f"Approved training scope: {len(selection)} verified AMI pauses. "
                  "Validation and test use original splits.", flush=True)
        else:
            require_gate(root)
        telephone_validation = AmiTelephoneValidation(root)
    # Stages select from validation. Only final, selected exports read test shards.
    meter = Progress("experiments", 37, "runs")
    failures = []
    final_models = []
    destination = root / "reports" / "experiments"
    destination.mkdir(parents=True, exist_ok=True)

    def record_telephone_validation(report, name, format_name):
        path = root / "reports" / f"{name}-{format_name}-ami-validation" / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        return next(row for row in report["thresholds"] if row["origin"] == AMI_THRESHOLD_ORIGIN)

    def attempt(name, config):
        reused = False
        try:
            merged = {**base, **config}
            output = completed_run(root, name, merged, device)
            reused = output is not None
            if reused:
                print(f"{name}: reuse verified training checkpoint", flush=True)
            else:
                output = train(root, name, merged, device)
            result = torch.load(output, map_location="cpu", weights_only=True)["metrics"]
            if telephone_validation is not None:
                phone = record_telephone_validation(telephone_validation.float_checkpoint(output), name, "float")
                result["ami_telephone"] = phone["all_ami"]
            _run_sync(root)
            return name, merged, result
        except Exception as error:
            print(f"{name} failed: {error}", flush=True)
            failures.append((name, str(error)))
        finally:
            if reused:
                meter.skip(1)
            else:
                meter.advance(1)

    def stage(variants):
        return [result for name, config in variants if (result := attempt(name, config)) is not None]

    try:
        first = stage([
            (f"s1-{arch}-{size}-{seed}", {"architecture": arch, "parameters": size, "seed": seed, "window": 3})
            for arch in ("a", "b") for size in (150000, 500000, 1000000) for seed in (17, 29)
        ])
        means = []
        for key in sorted({(item[1]["architecture"], item[1]["parameters"]) for item in first}):
            seeds = [item for item in first if (item[1]["architecture"], item[1]["parameters"]) == key]
            if len(seeds) != 2:
                continue
            name, config, _ = _rank(seeds)[0]
            metrics = {metric: sum(item[2][metric] or 0 for item in seeds) / 2
                       for metric in ("recall", "auc", "loss")}
            if telephone_validation is not None:
                metrics["ami_telephone"] = {
                    metric: sum(item[2]["ami_telephone"][metric] or 0 for item in seeds) / 2
                    for metric in ("recall", "auc")
                }
            means.append((name, config, metrics))
        stage1_order = _rank(means, len(means))
        top_two = stage1_order[:2]
        scope = telephone_validation.scope if telephone_validation is not None else {"split": "validation"}
        stage1_report = {**scope, "groups": [
            {"checkpoint": name, "architecture": config["architecture"],
             "parameters": config["parameters"], "mean_validation": metrics}
            for name, config, metrics in stage1_order
        ], "advanced": [name for name, _, _ in top_two]}
        (destination / "stage1-ranking.json").write_text(json.dumps(stage1_report, indent=2) + "\n")
        second = stage([
            (f"s2-{index}-{window}s", {**config, "window": window})
            for index, (_, config, _) in enumerate(top_two) for window in (1, 2, 4, 8)
        ])
        _, winner, _ = _rank(second)[0]
        mixes = {
            "ami": ["ami"], "read": ["ami", "common_voice"],
            "spontaneous": ["ami", "voxpopuli", "emilia_yodas"], "all": None,
        }
        third = stage([
            (f"s3-{mix}-{weak}", {**winner, "sources": sources, "weak": weak})
            for mix, sources in mixes.items() for weak in (True, False)
        ])
        _, winner, _ = _rank(third)[0]
        stage([(f"s3b-{fraction}", {**winner, "fraction": fraction}) for fraction in (0.25, 0.5, 1)])
        fourth = stage([(f"s4-{weight}", {**winner, "hold_weight": weight}) for weight in (1, 2, 4)])
        ranked = _rank(fourth + third + second + first, len(fourth + third + second + first))
        candidates = []
        for name, config, _ in ranked:
            if len(candidates) == 3:
                break
            try:
                model = export(root / "runs" / name / "best.pt")
                metadata = json.loads(model.with_name("export.json").read_text())
                full_auc, int8_auc = quantization_auc(root, model, metadata["threshold"])
                metadata["validation_auc"] = {"float": full_auc, "int8": int8_auc}
                model.with_name("export.json").write_text(json.dumps(metadata, indent=2) + "\n")
                if full_auc is None or int8_auc is None:
                    raise ValueError("validation needs both holds and shifts for the export check")
                if full_auc - int8_auc > 0.01:
                    source_name = name
                    name = f"{name}-qat"
                    retry = {**config, "epochs": 1, "lr": config.get("lr", 0.001) / 10,
                             "quantization_aware": True, "warm_start": source_name}
                    print(f"{source_name}: validation AUC drop {full_auc - int8_auc:.4f}, starting QAT",
                          flush=True)
                    checkpoint = completed_run(root, name, retry, device)
                    if checkpoint is None:
                        checkpoint = train(root, name, retry, device)
                    model = export(checkpoint)
                    metadata = json.loads(model.with_name("export.json").read_text())
                    qat_float, qat_int8 = quantization_auc(root, model, metadata["threshold"])
                    metadata["validation_auc"] = {"original_float": full_auc, "float": qat_float, "int8": qat_int8}
                    model.with_name("export.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    if qat_float is None or qat_int8 is None or max(full_auc, qat_float) - qat_int8 > 0.01:
                        raise ValueError("int8 validation AUC remains outside tolerance after QAT")
                validation = json.loads((root / "reports" / f"{name}-int8-validation" / "metrics.json").read_text())
                metadata["float_threshold"] = metadata["threshold"]
                metadata["threshold"] = validation["threshold"]
                overall = next(row["300"] for row in validation["groups"]
                               if row["source"] == "all" and row["language"] == "all")
                candidate = {"name": name, **overall}
                if telephone_validation is not None:
                    report = telephone_validation.int8_export(model, validation["threshold"])
                    phone = record_telephone_validation(report, name, "int8")
                    metadata["threshold"] = phone["threshold"]
                    metadata["threshold_scope"] = telephone_validation.scope
                    candidate = {"name": name, **phone["all_ami"], "pooled_clean": overall}
                model.with_name("export.json").write_text(json.dumps(metadata, indent=2) + "\n")
                candidates.append({**candidate, "threshold": metadata["threshold"]})
            except Exception as error:
                print(f"{name} export failed: {error}", flush=True)
                failures.append((name, str(error)))
        if len(candidates) < 3:
            raise ValueError(f"only {len(candidates)} exports passed validation")
        candidates.sort(key=lambda item: (
            item["recall"], item["auc"], item.get("pooled_clean", item)["recall"],
        ), reverse=True)
        (destination / "selection.json").write_text(json.dumps({**scope, "candidates": candidates}, indent=2) + "\n")
        for candidate in candidates:
            name, threshold = candidate["name"], candidate["threshold"]
            model = root / "runs" / name / "int8.onnx"
            try:
                benchmark(root, model, threshold)
                if (root / ".campaign-machine").is_file():
                    benchmark_volume(root, model, threshold)
                _run_sync(root)
                final_models.append(name)
            except Exception as error:
                print(f"{name} benchmark failed: {error}", flush=True)
                failures.append((name, str(error)))
            finally:
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish("failed" if failures else "done")
    finally:
        (destination / "queue.json").write_text(json.dumps({"models": final_models, "failures": failures}, indent=2) + "\n")
        _run_sync(root)
    return int(bool(failures))


def register(sub):
    handlers = {
        "label": "label downloaded corpus selections",
        "shard": "build compressed shards and freeze test conditions",
        "train": "train a model from a JSON configuration",
        "export": "export and check float and int8 ONNX",
        "benchmark": "report quality and single-core CPU latency",
        "benchmark-volume": "measure level and SNR sensitivity on held-out conversations",
        "benchmark-ami-reference": "score validation-selected models on supported English AMI test pauses",
        "compare-ami-validation": "compare completed Stage 1 checkpoints on AMI telephone validation",
        "smoke": "exercise the pipeline on a downloaded sample",
        "experiments": "run the first campaign experiment matrix",
        "listening": "sample labelled audio and score human judgments",
        "audit-labels": "audit sampled pauses with independent WebRTC speech activity",
        "check-pitch": "compare pitch and voicing with a YIN speech reference",
        "check-ami-reference": "compare AMI labels with the aligned human transcript",
    }
    for name, description in handlers.items():
        parser = sub.add_parser(name, help=description)
        parser.set_defaults(func=command)
        if name in {"label", "train", "smoke", "experiments", "audit-labels"}:
            parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name == "label":
            parser.add_argument("--workers", type=int, default=1)
        if name == "audit-labels":
            parser.add_argument("--detector", choices=("webrtc", "speechbrain"), default="webrtc")
        if name in {"label", "shard", "listening"}:
            parser.add_argument("--only", nargs="+", choices=LABEL_SOURCES, default=LABEL_SOURCES)
        if name == "smoke":
            parser.add_argument("--acquire-sample", action="store_true")
        if name == "listening":
            parser.add_argument("--judgments", type=Path, help="completed review.csv from the listening packet")
        if name in {"train", "experiments"}:
            parser.add_argument("--config", type=Path, required=True)
        if name == "train":
            parser.add_argument("--name", required=True)
        if name in {"export", "benchmark", "benchmark-volume"}:
            parser.add_argument("model", type=Path)
        if name == "export":
            parser.add_argument("--spectrum", choices=("stft", "convolution"), default="stft",
                                help="spectral implementation in the exported graph")
        if name in {"benchmark", "benchmark-volume"}:
            parser.add_argument("--threshold", type=float)
        if name == "benchmark":
            parser.add_argument("--split", choices=("validation", "test"), default="test")
            parser.add_argument("--samples", type=int, default=250)
        if name == "benchmark-ami-reference":
            parser.add_argument("--prepare", action="store_true", help="freeze the reference scope before seeing model scores")


def command(args):
    root = data_root()
    name = args.command
    if name == "label":
        label(root, args.only, _device(args.device), args.workers)
    elif name == "shard":
        shard(root, args.only)
    elif name == "benchmark-volume":
        from .volume import benchmark_volume

        threshold = args.threshold
        if threshold is None:
            threshold = json.loads(args.model.with_name("export.json").read_text())["threshold"]
        benchmark_volume(root, args.model, threshold)
    elif name == "benchmark-ami-reference":
        from .ami_benchmark import benchmark_selected, prepare

        if args.prepare:
            print(prepare(root))
        else:
            benchmark_selected(root)
    elif name == "compare-ami-validation":
        from .ami_benchmark import compare_validation

        print(compare_validation(root))
    elif name == "audit-labels":
        from .label_audit import audit, label_hashes

        rows = [row for source in LABEL_SOURCES for row in read_rows(root / "labels" / f"{source}.jsonl")]
        print(audit(root, rows, label_hashes(root, LABEL_SOURCES), detector=args.detector, device=_device(args.device)))
    elif name == "check-ami-reference":
        from .ami_reference import check_ami_reference

        print(check_ami_reference(root))
    elif name == "check-pitch":
        from .pitch_check import check_pitch

        print(check_pitch(root))
    elif name == "listening":
        from .listening import prepare, summarize

        if args.judgments:
            print(summarize(root, args.judgments))
        else:
            prepare(root, [row for source in args.only for row in read_rows(root / "labels" / f"{source}.jsonl")])
    elif name == "smoke":
        smoke(root, _device(args.device), acquire=args.acquire_sample)
    elif name == "experiments":
        return experiments(root, json.loads(args.config.read_text()), _device(args.device))
    elif name == "train":
        from .train import train

        if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
            raise ValueError("run name must be one directory name")
        train(root, args.name, json.loads(args.config.read_text()), _device(args.device))
    else:
        from .evaluate import benchmark, export

        if name == "export":
            print(export(args.model, spectrum=args.spectrum))
        else:
            threshold = args.threshold
            if threshold is None:
                threshold = json.loads(args.model.with_name("export.json").read_text())["threshold"]
            benchmark(root, args.model, threshold, split=args.split, samples=args.samples)
    return 0
