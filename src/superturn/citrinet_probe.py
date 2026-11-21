"""Compare pretrained and random Citrinet end-of-turn classifiers on conversation data."""

import argparse
from importlib.metadata import version
import json

from .ami_benchmark import AMI_THRESHOLD_ORIGIN, AmiTelephoneValidation
from .ami_calibration import _write_json
from .citrinet import NEMO_VERSION, REPOSITORY, REVISION, SHA256, archive_contents, fetch
from .eot_domain_probe import DATASET_SHA256, run as compare_domain
from .pipeline import _run_sync
from .progress import Progress
from .sources import data_root


def check_model(root, config):
    import torch

    from .citrinet import CitrinetTurnModel, initialize

    model = CitrinetTurnModel(config)
    initialize(model, root, config["citrinet_modules"], pretrained=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if not 8_000_000 <= count <= 12_000_000:
        raise ValueError(f"Citrinet parameter count {count:,} is outside the pilot budget")
    device = "cuda"
    model.to(device).eval()
    generator = torch.Generator().manual_seed(config["seed"])
    audio = (torch.randn(config["batch_size"], model.samples, generator=generator) / 10).to(device)
    elapsed = torch.full((len(audio), 1), 300.0, device=device)
    with torch.inference_mode():
        batch = model(audio, elapsed)
        single = model(audio[:1], elapsed[:1])
    if batch.shape != (len(audio), 1) or single.shape != (1, 1) or not batch.isfinite().all():
        raise ValueError("Citrinet batch output has invalid shape or scores")
    torch.testing.assert_close(single, batch[:1], rtol=0.0001, atol=0.00001)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    optimizer.zero_grad()
    loss = model.logits(audio, elapsed).square().mean()
    loss.backward()
    if not loss.isfinite() or not all(parameter.grad is not None and parameter.grad.isfinite().all()
                                      for parameter in model.parameters() if parameter.requires_grad):
        raise ValueError("Citrinet generated-batch training produced invalid gradients")
    optimizer.step()
    return count


def run(root, *, sweep_job="capacity-window-sweep"):
    from .train import completed_run, train

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Citrinet fine-tuning uses the existing campaign machine")
    if (root / "jobs" / f"{sweep_job}.exit").read_text().strip() != "0":
        raise ValueError("the GRU window sweep must complete successfully before the transfer pilot")
    reports = root / "reports"
    sweep = json.loads((reports / "capacity-window-validation.json").read_text())
    skip = [2.5] if sweep_job == "capacity-window-shortlist" else []
    if ({item["seconds"] for item in sweep["ranked"]} != {1, 1.5, 2, 2.5, 3} - set(skip)
            or sweep.get("skipped_windows", []) != skip):
        raise ValueError("transfer pilot requires all scheduled GRU windows")
    diagnostics = json.loads((reports / "capacity-window-eot-domain-probe.json").read_text())
    if diagnostics["dataset_sha256"] != DATASET_SHA256 or diagnostics["decision_ms"] != 300:
        raise ValueError("transfer pilot requires the pinned 300 ms external diagnostic")
    telephone = diagnostics["conditions"]["telephone"]
    diagnostic_windows = telephone["capacity_windows"]
    ceiling = "max_recall_at_5pct_false_complete"
    smart_turn = telephone["smart-turn-v3.2"]["calibrated"]["retrospective_roc_limits"][ceiling]
    if any(item["telephone_validation"]["recall"] >= 0.8
           and item["telephone_validation"]["false_complete"] <= 0.05
           and diagnostic_windows[item["name"]]["retrospective_roc_limits"][ceiling] > smart_turn
           for item in sweep["ranked"]):
        print("A GRU window met the accuracy gate and beat Smart Turn.")
        return None
    validation = AmiTelephoneValidation(root)
    if any(sweep[key] != value for key, value in validation.scope.items()):
        raise ValueError("transfer pilot AMI validation differs from the window sweep")
    if version("nemo-toolkit") != NEMO_VERSION:
        raise ValueError(f"the transfer pilot requires NeMo {NEMO_VERSION}")

    modules, _ = archive_contents(fetch(root))
    config = {
        "architecture": "citrinet", "window": sweep["ranked"][0]["seconds"], "seed": 29,
        "epochs": 20, "fixed_epochs": True, "batch_size": 16, "lr": 0.0003,
        "sources": ["ami"], "ami_verified_training": True, "hold_weight": 1,
        "augmentation": True, "augmentation_workers": 8, "citrinet_modules": modules,
    }
    parameter_count = check_model(root, config)
    print(f"Verified archived Citrinet configuration with {parameter_count:,} parameters", flush=True)
    result = {**validation.scope, "window_seconds": config["window"],
              "parameters": parameter_count,
              "configuration": {key: value for key, value in config.items() if key != "citrinet_modules"},
              "upstream": {"repository": REPOSITORY, "revision": REVISION, "sha256": SHA256,
                           "nemo_version": NEMO_VERSION}, "arms": {}}
    meter = Progress("citrinet-probe", 2, "fits")
    try:
        for arm in ("random", "pretrained"):
            name = f"citrinet-256-ls-ami-{int(config['window'] * 1000)}ms-{arm}"
            variant = {**config, "citrinet_pretrained": arm == "pretrained"}
            checkpoint = completed_run(root, name, variant, "cuda")
            if checkpoint is None:
                checkpoint = train(root, name, variant, "cuda")
            telephone = validation.float_checkpoint(checkpoint)
            validation_dir = reports / f"{name}-float-ami-validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            _write_json(validation_dir / "metrics.json", telephone)
            diagnostic = compare_domain(root, checkpoint_name=name, replay_references=True)
            point = next(row for row in telephone["thresholds"] if row["origin"] == AMI_THRESHOLD_ORIGIN)
            result["arms"][arm] = {"name": name, "ami_telephone": point,
                                   "checkpoint_sha256": telephone["checkpoint_sha256"],
                                   "diagnostic": diagnostic.name}
            _run_sync(root)
            meter.advance(1)
        destination = reports / "citrinet-probe.json"
        _write_json(destination, result)
        _run_sync(root)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-job", choices=("capacity-window-sweep", "capacity-window-shortlist"),
                        default="capacity-window-sweep")
    args = parser.parse_args()
    if result := run(data_root(), sweep_job=args.sweep_job):
        print(result)
