"""Score real telephone transcript completions and controlled caller prefixes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .ami_calibration import _write_json
from .shards import seed_for
from .sources import data_root
from .turn_generation import normalize
from .turn_metrics import auc
from .turn_text_train import LABELS, binary_metrics, digest, load_fit, predict, read_dataset
from .turn_training_dataset import model_text


def transcript_pairs(segments):
    """Join neighboring same-speaker segments, retaining unknown runs as barriers."""
    runs = []
    for segment in sorted(segments, key=lambda row: (float(row["start_timestamp_ms"]), row["index"])):
        role = segment["speaker_role"]
        start = float(segment["start_timestamp_ms"])
        end = start + float(segment["duration_ms"])
        words = normalize(segment.get("human_transcript") or "").split()
        if runs and runs[-1]["role"] == role:
            run = runs[-1]
            run["end"] = max(end, run["end"])
            run["words"].extend(words)
            run["complete_text"] &= bool(words)
        else:
            runs.append({"role": role, "start": start, "end": end, "words": words, "complete_text": bool(words)})
    for index, current in enumerate(runs[:-1]):
        if current["role"] != "caller" or len(current["words"]) < 4 or not current["complete_text"]:
            continue
        following = runs[index + 1]
        if (following["role"] != "agent" or not following["complete_text"]
                or len(following["words"]) < 3 or current["end"] > following["start"]):
            continue
        previous = runs[index - 1] if index else None
        if previous and (previous["role"] != "agent" or not previous["complete_text"]
                         or previous["end"] > current["start"]):
            continue
        prior = " ".join(previous["words"]) if previous else ""
        words = current["words"]
        yield index, {
            "interior_cut": model_text(prior, " ".join(words[:len(words) // 2])),
            "last_two_words_cut": model_text(prior, " ".join(words[:-2])),
            "full": model_text(prior, " ".join(words)),
        }


def length_matched(items):
    bins = defaultdict(lambda: {0: [], 1: []})
    for item in items:
        bins[item["context"], item["words"]][item["label"]].append(item)
    result = []
    for group in bins.values():
        keep = min(map(len, group.values()))
        for members in group.values():
            ordered = sorted(members, key=lambda item: item["rank"])
            result.extend(ordered[:keep])
    return result


def report_for(items):
    result = binary_metrics([(item["score"], item["label"]) for item in items])
    result["calls"] = len({item["call"] for item in items})
    result["callers"] = len({item["caller"] for item in items})
    result["length_only_auc"] = auc([item["words"] for item in items], [item["label"] for item in items])
    result["median_scores"] = {}
    for name, target in LABELS.items():
        group = [item["score"] for item in items if item["label"] == target]
        result["median_scores"][name] = float(np.median(group)) if group else None
    return result


def run(root, dataset, base, checkpoint, device):
    from .harper_valley_benchmark import pinned_split

    if not (root / ".campaign-machine").is_file():
        raise ValueError("real benchmark transcripts stay on Vast")
    cases, _, selection_audit = pinned_split(root, "train")
    synthetic, manifest = read_dataset(dataset)
    training_inputs = {row["text"] for row in synthetic["train"]}
    transcript_root = root / "raw/harper-valley/data/transcript"
    pairs, covered = [], Counter()
    fingerprint = sha256()
    for case in sorted(cases, key=lambda row: row["id"]):
        path = transcript_root / f"{case['id']}.json"
        if not path.is_file():
            covered["transcripts_missing"] += 1
            continue
        payload = path.read_bytes()
        fingerprint.update(path.stem.encode() + bytes.fromhex(sha256(payload).hexdigest()))
        covered["transcripts_read"] += 1
        for index, views in transcript_pairs(json.loads(payload)):
            if not training_inputs.isdisjoint(views.values()):
                covered["pairs_excluded_exact_training_input_overlap"] += 1
                continue
            context = "one_turn" if views["full"].startswith("---\n") else "two_turn"
            pairs.append((case["id"], case["caller"], index, context, views))
    if not pairs:
        raise ValueError("no eligible real transcript pairs")
    tokenizer, model = load_fit(checkpoint, base, device, manifest)
    result = {
        "scope": "real HarperValley English development transcript diagnostic",
        "label_contract": "recorded speaker handoff vs controlled word-prefix cut",
        "observed_300ms_pauses_required": False,
        "threshold_source": "fixed synthetic fit threshold 0.5",
        "checkpoint_sha256": digest(checkpoint),
        "development_selection_audit_sha256": selection_audit,
        "transcripts_sha256": fingerprint.hexdigest(),
        "transcript_inventory": dict(covered),
        "identical_interior_and_last_two_word_cuts": sum(
            views["interior_cut"] == views["last_two_words_cut"] for _, _, _, _, views in pairs),
        "length_only_within_turn_ranking": 1.0,
        "views": {},
    }
    for view in ("interior_cut", "last_two_words_cut"):
        inputs = [pair[4][field] for pair in pairs for field in (view, "full")]
        scores = iter(predict(tokenizer, model, inputs, device))
        current_only = [model_text("", text.split("\n", 1)[1]) for text in inputs]
        masked_scores = iter(predict(tokenizer, model, current_only, device))
        items, masked, outcomes, masked_outcomes = [], [], [], []
        for call, caller, index, context, samples in pairs:
            hold_score, shift_score = next(scores), next(scores)
            masked_hold, masked_shift = next(masked_scores), next(masked_scores)
            outcomes.append(float(shift_score > hold_score) + .5 * (shift_score == hold_score))
            masked_outcomes.append(float(masked_shift > masked_hold) + .5 * (masked_shift == masked_hold))
            for field, score, mask, label in ((view, hold_score, masked_hold, "hold"),
                                              ("full", shift_score, masked_shift, "shift")):
                item = {
                    "call": call, "caller": caller, "context": context, "score": score, "label": LABELS[label],
                    "words": len(samples[field].split("\n")[1].split()),
                    "rank": seed_for("harper-transcript-length-match-v1", call, index, label),
                }
                items.append(item)
                masked.append({**item, "score": mask})
        result["views"][view] = {
            "all": report_for(items),
            "one_turn": report_for([item for item in items if item["context"] == "one_turn"]),
            "two_turn": report_for([item for item in items if item["context"] == "two_turn"]),
            "exact_word_count_and_context_matched": report_for(length_matched(items)),
            "within_turn_ranking": sum(outcomes) / len(outcomes),
            "current_only_ablation": report_for(masked),
            "current_only_within_turn_ranking": sum(masked_outcomes) / len(masked_outcomes),
        }
    _write_json(root / "reports/harper-real-transcript-benchmark.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    root = data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/turn-completion-v1"))
    parser.add_argument("--base", type=Path, default=root / "cache/turn-text-e5-small")
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/turn-text-e5-v1/checkpoint.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(root, args.dataset, args.base, args.checkpoint, args.device)
