"""Test frozen Citrinet representations with the same AMI-only linear readout."""

import hashlib
import json

import numpy as np

from .ami_benchmark import DECISION_MS, AmiTelephoneValidation
from .ami_calibration import _write_json
from .ami_deadline import retrospective_limits
from .ami_training import load_selection, selection_hash
from .augment import condition
from .citrinet import CitrinetTurnModel, REVISION, SHA256, initialize
from .labels import read_rows
from .progress import Progress
from .sources import data_root
from .train import _input, _stats, _threshold, completed_run, paths, records


def training_audio(root, samples):
    verified = load_selection(root)
    validation_speakers = {row.speaker for row in read_rows(root / "labels" / "ami.jsonl")
                           if row.split == "validation"}
    minimum = DECISION_MS * 16
    audio, labels, speakers = [], [], set()

    def eligible(row):
        return row["source"] == "ami" and min(row["pause_samples"], verified.get(
            (row["audio"], row["pause_start"], row["label"]), 0)) >= minimum

    columns = ["source", "audio", "pause_start", "label", "pause_samples"]
    for path in paths(root, "train"):
        if not any(eligible(row) for batch in records([path], columns=columns) for row in batch):
            continue
        for batch in records([path]):
            for row in batch:
                if not eligible(row):
                    continue
                pcm = condition(np.frombuffer(row["pcm"], "<i2"), "telephone", 0)
                audio.append(_input({**row, "pcm": pcm}, samples, DECISION_MS))
                labels.append(int(row["label"] == "shift"))
                speakers.add(row["speaker"])
    if speakers & validation_speakers or len(set(labels)) != 2:
        raise ValueError("the frozen probe needs both labels and speaker-disjoint AMI splits")
    return np.asarray(audio, dtype=np.float32) / 32768, np.asarray(labels), len(speakers)


def encode(model, batches):
    import torch

    features = []
    with torch.inference_mode():
        for audio, elapsed in batches:
            values = model.decision_input(torch.from_numpy(audio).cuda(), torch.from_numpy(elapsed).cuda())
            # Elapsed silence is identical for every example in this probe.
            features.append(values[:, :-1].cpu().numpy())
    return np.concatenate(features).astype(np.float64)


def fit_readout(training, truth, validation):
    from scipy.optimize import minimize
    from scipy.special import expit

    center = training.mean(axis=0)
    scale = training.std(axis=0).clip(min=1e-8)
    x = np.column_stack(((training - center) / scale, np.ones(len(training))))

    def objective(weights):
        logits = x @ weights
        penalty = weights.copy()
        penalty[-1] = 0
        loss = np.mean(np.logaddexp(0, logits) - truth * logits) + 0.025 * np.dot(penalty, penalty)
        gradient = x.T @ (expit(logits) - truth) / len(truth) + 0.05 * penalty
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                      options={"gtol": 1e-8, "ftol": 1e-13, "maxiter": 2000})
    gradient = float(np.abs(result.jac).max())
    if not result.success or not np.isfinite(result.fun) or gradient > 1e-5:
        raise ValueError(f"frozen encoder readout did not converge: {result.message}, gradient {gradient:g}")
    logits = ((validation - center) / scale) @ result.x[:-1] + result.x[-1]
    return logits, {"iterations": result.nit, "objective": float(result.fun), "gradient_max_abs": gradient}


def run(root):
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("the frozen Citrinet probe uses the existing campaign machine")
    torch.set_num_threads(1)
    previous = (root / "reports" / "citrinet-probe.json").read_bytes()
    prior = json.loads(previous)
    record = prior["arms"]["pretrained"]
    name = record["name"]
    run_dir = root / "runs" / name
    config = json.loads((run_dir / "finished.json").read_text())["signature"]["config"]
    checkpoint = completed_run(root, name, config, "cuda")
    if checkpoint is None or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
        raise ValueError("the full fine-tune checkpoint differs from the completed comparison")
    validation = AmiTelephoneValidation(root)
    if any(prior[key] != value for key, value in validation.scope.items()):
        raise ValueError("the AMI validation cohort differs from the fine-tuning probe")

    samples = round(config["window"] * 16000)
    training, truth, speakers = training_audio(root, samples)
    elapsed = np.full((len(training), 1), DECISION_MS, dtype=np.float32)
    train_batches = [(training[start:start + 16], elapsed[start:start + 16])
                     for start in range(0, len(training), 16)]
    result = {**validation.scope, "window_seconds": config["window"],
              "training": {"holds": len(truth) - int(truth.sum()), "shifts": int(truth.sum()), "speakers": speakers,
                           "eligibility_sha256": selection_hash(root)},
              "upstream": {"revision": REVISION, "sha256": SHA256},
              "fine_tune_report_sha256": hashlib.sha256(previous).hexdigest(), "arms": {}}
    arms = ("random", "original_asr", "fine_tuned")
    meter = Progress("citrinet-frozen-encoder-probe", len(arms), "encoders")
    try:
        for arm in arms:
            torch.manual_seed(config["seed"])
            model = CitrinetTurnModel(config)
            if arm == "fine_tuned":
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state"])
            else:
                initialize(model, root, config["citrinet_modules"], pretrained=arm == "original_asr")
            model.cuda().eval()
            train_features = encode(model, train_batches)
            validation_features = encode(model, validation._batches(samples))
            logits, optimizer = fit_readout(train_features, truth, validation_features)
            scores = logits.astype(np.float32).tolist()
            threshold = _threshold(scores, validation.truth)
            supported_scores = [scores[index] for index in validation.confirmed]
            result["arms"][arm] = {"threshold_logit": threshold, "readout_convergence": optimizer,
                                   "all_ami": _stats(scores, validation.truth, threshold),
                                   "annotation_supported": _stats(supported_scores, validation.confirmed_truth, threshold),
                                   "all_ami_roc_limits": retrospective_limits(scores, validation.truth),
                                   "supported_roc_limits": retrospective_limits(supported_scores, validation.confirmed_truth)}
            meter.advance(1)
        destination = root / "reports" / "citrinet-frozen-encoder-probe.json"
        _write_json(destination, result)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    print(run(data_root()))
