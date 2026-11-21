"""Command-line entry points for preparation and the first campaign."""

import json
from collections import defaultdict
from dataclasses import replace
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


def label(root, sources, device):
    from .audio import silero
    from .corpora import corpus_labels, whisper_aligner

    detect = silero() if any(source != "indic_diarbench" for source in sources) else None
    align = whisper_aligner(device) if any(source in {"common_voice", "fleurs"} for source in sources) else None
    registry = load_sources()
    for source in sources:
        output = root / "labels" / f"{source}.jsonl"
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


def _rank(results, count=1):
    good = sorted(results, key=lambda item: (item[2]["recall"] or 0, item[2]["auc"] or 0, -item[2]["loss"]),
                  reverse=True)
    if not good:
        raise ValueError("all runs in stage failed")
    return good[:count]


def experiments(root, base, device):
    import torch
    from .evaluate import benchmark, export, quantization_auc
    from .progress import Progress
    from .train import completed_run, train

    if (root / ".campaign-machine").is_file():
        from .listening import require_gate
        from .shards import require_split_coverage

        manifest = json.loads((root / "shards" / "manifest.json").read_text())
        registry = load_sources()
        require_split_coverage(manifest["speaker_report"], {name: registry[name].role for name in LABEL_SOURCES})
        require_gate(root)
    # Stages select from validation. Only final, selected exports read test shards.
    meter = Progress("experiments", 37, "runs")
    failures = []
    final_models = []

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
            means.append((name, config, metrics))
        second = stage([
            (f"s2-{index}-{window}s", {**config, "window": window})
            for index, (_, config, _) in enumerate(_rank(means, 2)) for window in (1, 2, 4, 8)
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
        for name, _, _ in ranked:
            if len(final_models) == 3:
                break
            try:
                model = export(root / "runs" / name / "best.pt")
                metadata = json.loads(model.with_name("export.json").read_text())
                full_auc, int8_auc = quantization_auc(root, model, metadata["threshold"])
                metadata["validation_auc"] = {"float": full_auc, "int8": int8_auc}
                if full_auc is None or int8_auc is None or full_auc - int8_auc > 0.01:
                    model.with_name("export.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    raise ValueError(f"int8 validation AUC {int8_auc}, float AUC {full_auc}")
                validation = json.loads((root / "reports" / f"{name}-int8-validation" / "metrics.json").read_text())
                metadata["float_threshold"] = metadata["threshold"]
                metadata["threshold"] = validation["threshold"]
                model.with_name("export.json").write_text(json.dumps(metadata, indent=2) + "\n")
                benchmark(root, model, metadata["threshold"])
                _run_sync(root)
                final_models.append(name)
                meter.advance(1)
            except Exception as error:
                print(f"{name} export or benchmark failed: {error}", flush=True)
                failures.append((name, str(error)))
        if len(final_models) < 3:
            raise ValueError(f"only {len(final_models)} exports passed validation")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish("failed" if failures else "done")
    finally:
        destination = root / "reports" / "experiments"
        destination.mkdir(parents=True, exist_ok=True)
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
        "smoke": "exercise the pipeline on a downloaded sample",
        "experiments": "run the first campaign experiment matrix",
        "listening": "sample labelled audio and score human judgments",
        "check-pitch": "compare pitch and voicing with a YIN speech reference",
    }
    for name, description in handlers.items():
        parser = sub.add_parser(name, help=description)
        parser.set_defaults(func=command)
        if name in {"label", "train", "smoke", "experiments"}:
            parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        if name in {"export", "benchmark"}:
            parser.add_argument("model", type=Path)
        if name == "benchmark":
            parser.add_argument("--threshold", type=float)
            parser.add_argument("--split", choices=("validation", "test"), default="test")
            parser.add_argument("--samples", type=int, default=250)


def command(args):
    root = data_root()
    name = args.command
    if name == "label":
        label(root, args.only, _device(args.device))
    elif name == "shard":
        shard(root, args.only)
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
            print(export(args.model))
        else:
            threshold = args.threshold
            if threshold is None:
                threshold = json.loads(args.model.with_name("export.json").read_text())["threshold"]
            benchmark(root, args.model, threshold, split=args.split, samples=args.samples)
    return 0
