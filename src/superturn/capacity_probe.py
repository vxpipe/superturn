"""Train the 10M accuracy probe and score a frozen telephone comparison."""

import json

from .ami_benchmark import AmiTelephoneValidation
from .ami_calibration import _write_json
from .eot_domain_probe import run as compare_domain
from .sources import REPO_ROOT, data_root

NAME = "capacity-gru-10m-29"
PARAMETERS = 9_919_695


def run(root):
    from .ami_training import load_selection
    from .model import parameter_count
    from .train import completed_run, train

    if not (root / ".campaign-machine").is_file():
        raise ValueError("the capacity probe uses the existing campaign machine")
    config = json.loads((REPO_ROOT / "configs" / "capacity-10m.json").read_text())
    if parameter_count(config["architecture"], config["width"], config["window"]) != PARAMETERS:
        raise ValueError("the capacity probe architecture differs from the planned model")
    load_selection(root)
    checkpoint = completed_run(root, NAME, config, "cuda") or train(root, NAME, config, "cuda")
    telephone = AmiTelephoneValidation(root).float_checkpoint(checkpoint)
    path = root / "reports" / f"{NAME}-float-ami-validation" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, telephone)
    return compare_domain(root, checkpoint_name=NAME)


if __name__ == "__main__":
    print(run(data_root()))
