"""Measure score spread, ties and recall limits on the AMI telephone validation set."""

import argparse

import numpy as np

from .ami_benchmark import AmiTelephoneValidation
from .ami_calibration import _write_json
from .ami_deadline import retrospective_limits
from .eot_domain_probe import load_validated_models
from .progress import Progress
from .sources import data_root


def score_spread(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("telephone score spread requires finite scores")
    _, counts = np.unique(scores, return_counts=True)
    levels = (0, 0.01, 0.5, 0.99, 1)
    return {"count": len(scores), "distinct": len(counts), "largest_tie": int(counts.max()),
            "quantiles": dict(zip((str(level) for level in levels), np.quantile(scores, levels).tolist(),
                                  strict=True))}


def recurrent_context_spread(states, truth, scores, head, decision_ms):
    import torch

    from .train import _auc

    early_scores = []
    with torch.inference_mode():
        for batch in states:
            elapsed = torch.full((len(batch), 1), decision_ms / 1000)
            early_scores.extend(head(torch.cat((batch[:, 0], elapsed), dim=-1)).sigmoid().flatten().tolist())
    report = {}
    for index, (name, predictions) in enumerate((("before_silence", early_scores), ("decision", scores))):
        state = torch.cat([batch[:, index] for batch in states])
        report[name] = {"state_rms_std": float(state.var(dim=0, correction=0).mean().sqrt()),
                        "score_spread": score_spread(predictions), "auc": _auc(predictions, truth)}
    return report


def run(root, names):
    import torch

    from .model import FRAME, HOP
    from .train import _stats

    if not (root / ".campaign-machine").is_file():
        raise ValueError("telephone score audit uses the existing campaign machine")
    torch.set_num_threads(1)
    validation = AmiTelephoneValidation(root)
    models, provenance = load_validated_models(root, names)
    if validation.scope["validation_sha256"] != provenance["validation_sha256"]:
        raise ValueError("the checkpoint scoring scope differs from the validation data")
    output = {**validation.scope, "checkpoint_provenance": provenance["checkpoints"], "models": {}}
    meter = Progress("ami-telephone-score-audit", len(models), "models")
    try:
        for name, model, threshold in models:
            scores, states = [], []
            capture = None
            if getattr(model, "architecture", None) == "a":
                # Two stride-2, size-5 layers reach six spectrogram frames beyond 4*i.
                # Window-wide feature normalization remains shared in this diagnostic.
                silence_samples = validation.scope["decision_ms"] * 16
                before_silence = ((model.samples - silence_samples - FRAME) // HOP - 6) // 4

                def observe_sequence(_module, _inputs, output):
                    states.append(output[0][:, [before_silence, -1], :].clone())

                capture = model.sequence.register_forward_hook(observe_sequence)
            try:
                with torch.inference_mode():
                    for audio, silence in validation._batches(model.samples):
                        scores.extend(model(torch.from_numpy(audio), torch.from_numpy(silence)).flatten().tolist())
            finally:
                if capture is not None:
                    capture.remove()
            if len(scores) != len(validation.truth):
                raise ValueError(f"{name} score count differs from validation labels")
            output["models"][name] = {
                "scores": score_spread(scores),
                "hold_scores": score_spread([score for score, label in zip(scores, validation.truth) if not label]),
                "shift_scores": score_spread([score for score, label in zip(scores, validation.truth) if label]),
                "validation_operating_point": _stats(scores, validation.truth, threshold),
                "retrospective_roc_limits": retrospective_limits(scores, validation.truth),
            }
            if states:
                output["models"][name]["recurrent_context"] = recurrent_context_spread(
                    states, validation.truth, scores, model.decision, validation.scope["decision_ms"])
            meter.advance(1)
        destination = root / "reports" / "ami-telephone-score-audit.json"
        _write_json(destination, output)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    print(run(data_root(), args.checkpoints))
