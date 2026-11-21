"""Train the 10M accuracy probe and score a frozen telephone comparison."""

import argparse
import json

from .ami_benchmark import AMI_THRESHOLD_ORIGIN, AmiTelephoneValidation
from .ami_calibration import _write_json
from .eot_domain_probe import run as compare_domain
from .pipeline import _run_sync
from .progress import Progress
from .sources import REPO_ROOT, data_root

NAME = "capacity-gru-10m-29"
PARAMETERS = 9_919_695
WINDOWS = (3, 1, 1.5, 2, 2.5)


def run(root, *, window_sweep=False, completed_shortlist=False):
    from .ami_training import load_selection
    from .model import parameter_count
    from .train import completed_run, train

    if not (root / ".campaign-machine").is_file():
        raise ValueError("the capacity probe uses the existing campaign machine")
    config = json.loads((REPO_ROOT / "configs" / "capacity-10m.json").read_text())
    load_selection(root)
    telephone_validation = AmiTelephoneValidation(root)
    windows = tuple(seconds for seconds in WINDOWS if seconds != 2.5) if completed_shortlist else (
        WINDOWS if window_sweep else (config["window"],))
    names, outcomes = [], []
    job = "capacity-window-shortlist" if completed_shortlist else "capacity-window-sweep"
    meter = Progress(job, len(windows), "windows") if window_sweep or completed_shortlist else None
    try:
        for seconds in windows:
            variant = {**config, "window": seconds}
            if parameter_count(variant["architecture"], variant["width"], seconds) != PARAMETERS:
                raise ValueError("the capacity probe architecture differs from the planned model")
            name = NAME if seconds == config["window"] else f"capacity-gru-10m-29-{int(seconds * 1000)}ms"
            names.append(name)
            checkpoint = completed_run(root, name, variant, "cuda")
            if checkpoint is None:
                if completed_shortlist:
                    raise ValueError(f"{name} needs a completed checkpoint before shortlist evaluation")
                checkpoint = train(root, name, variant, "cuda")
            telephone = telephone_validation.float_checkpoint(checkpoint)
            path = root / "reports" / f"{name}-float-ami-validation" / "metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, telephone)
            point = next(record for record in telephone["thresholds"] if record["origin"] == AMI_THRESHOLD_ORIGIN)
            outcomes.append({"name": name, "seconds": seconds, "parameters": PARAMETERS,
                             "telephone_validation": point["all_ami"]})
            if meter is not None:
                if not completed_shortlist:
                    _run_sync(root)
                meter.advance(1)
        if meter is not None:
            if completed_shortlist and any(
                item["telephone_validation"]["recall"] >= 0.8
                and item["telephone_validation"]["false_complete"] <= 0.05
                for item in outcomes if item["seconds"] in (2, 3)
            ):
                raise ValueError("the 2.5-second window was conditional on both endpoints missing the target")
            ranked = sorted(outcomes, key=lambda item: (
                item["telephone_validation"]["false_complete"] <= 0.05,
                item["telephone_validation"]["recall"], item["telephone_validation"]["auc"]), reverse=True)
            _write_json(root / "reports" / "capacity-window-validation.json", {
                **telephone_validation.scope, "ranked": ranked,
                "skipped_windows": [2.5] if completed_shortlist else [],
            })
            result = compare_domain(root, window_names=names)
        else:
            result = compare_domain(root, checkpoint_name=NAME)
    except BaseException:
        if meter is not None:
            meter.finish("failed")
        raise
    else:
        if meter is not None:
            meter.finish()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--window-sweep", action="store_true",
                      help="compare 1 to 3 seconds at half-second steps at fixed capacity")
    mode.add_argument("--completed-shortlist", action="store_true",
                      help="score four completed windows if both 2 and 3 seconds missed the accuracy target")
    args = parser.parse_args()
    print(run(data_root(), window_sweep=args.window_sweep, completed_shortlist=args.completed_shortlist))
