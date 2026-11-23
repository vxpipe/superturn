"""Fine-tune a pinned multilingual encoder using only the synthetic corpus."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import random

from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root

BASE = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
BASE_SHA256 = "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"
QWEN_BASE = "Qwen/Qwen3-0.6B-Base"
QWEN_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
QWEN_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
TOKEN_BUDGET = 128
QUERY_PREFIX = "query: "
LABELS = {"hold": 0, "shift": 1}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_dataset(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["scope"] != "synthetic training and diagnostic validation"
            or manifest["model_input_fields"] != ["text"] or manifest["first_turn_marker"] != "---"):
        raise ValueError("training requires the two-line synthetic corpus")
    splits = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        detail = manifest["splits"][split]
        if digest(path) != detail["sha256"]:
            raise ValueError(f"{split} does not match its dataset checksum")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != detail["examples"]:
            raise ValueError(f"unexpected {split} example count")
        if any(row["label"] not in LABELS or len(row["text"].split("\n")) != 2
               or not all(row["text"].split("\n")) for row in rows):
            raise ValueError("unexpected synthetic input format")
        splits[split] = rows
    first, second = splits.values()
    if {row["text"] for row in first} & {row["text"] for row in second}:
        raise ValueError("the same model input crossed dataset splits")
    return splits, digest(directory / "manifest.json")


def pack(tokenizer, text):
    """Apply the encoder query prefix and truncate from the previous turn first."""
    previous, current = text.split("\n")
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    limit = TOKEN_BUDGET - n_special
    prior = tokenizer.encode(QUERY_PREFIX + previous, add_special_tokens=False)
    caller = tokenizer.encode(current, add_special_tokens=False)
    required = len(prior) if previous == "---" else len(tokenizer.encode(QUERY_PREFIX, add_special_tokens=False))
    caller = caller[-(limit - required):]
    prior_tail = min(len(prior) - required, limit - required - len(caller))
    prior = prior[:required] + (prior[-prior_tail:] if prior_tail else [])
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if n_special != 4 or bos is None or eos is None:
        raise ValueError("expected the pinned XLM-R pair token layout")
    return [bos, *prior, eos, eos, *caller, eos]


def pack_qwen(tokenizer, text):
    previous, current = text.split("\n")
    prior = tokenizer.encode(previous, add_special_tokens=False)
    separator = tokenizer.encode("\n", add_special_tokens=False)
    caller = tokenizer.encode(current, add_special_tokens=False)
    minimum_prior = len(prior) if previous == "---" else 0
    caller = caller[-(TOKEN_BUDGET - len(separator) - minimum_prior):]
    keep = TOKEN_BUDGET - len(separator) - len(caller)
    return [*(prior[-keep:] if keep else []), *separator, *caller]


def build_model(base_path, layers, backbone="e5"):
    import torch
    from transformers import AutoModel

    class EncoderClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            precision = {"dtype": torch.float32} if backbone == "qwen3" else {}
            self.encoder = AutoModel.from_pretrained(base_path, local_files_only=True, use_safetensors=True, **precision)
            self.encoder.requires_grad_(False)
            if layers:
                blocks = self.encoder.layers if backbone == "qwen3" else self.encoder.encoder.layer
                blocks[-layers:].requires_grad_(True)
            self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            if self.backbone == "qwen3":
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=False).last_hidden_state
                rows = torch.arange(len(hidden), device=hidden.device)
                final = attention_mask.sum(dim=1) - 1
                return self.head(hidden[rows, final]).squeeze(-1)
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return self.head(pooled).squeeze(-1)

    return EncoderClassifier()


def binary_metrics(items, threshold=.5):
    import numpy as np

    from .turn_metrics import auc, recall_area

    holds = sum(not target for _, target in items)
    shifts = len(items) - holds
    false_complete = sum(not target and value >= threshold for value, target in items) / holds if holds else None
    recall = sum(target and value >= threshold for value, target in items) / shifts if shifts else None
    report = {"holds": holds, "shifts": shifts, "threshold": threshold, "balanced_accuracy": None,
              "false_complete": false_complete, "shift_recall": recall, "auc": None,
              "retrospective_recall_at_5_percent_false_complete": None, "normalized_pauc_to_10_percent": None}
    if holds and shifts:
        probabilities, truth = zip(*items, strict=True)
        limits = recall_area(np.array(probabilities), np.array(truth))
        report.update(balanced_accuracy=(1 - false_complete + recall) / 2, auc=auc(probabilities, truth),
                      retrospective_recall_at_5_percent_false_complete=float(limits[1]),
                      normalized_pauc_to_10_percent=float(limits[3]))
    return report


def scorecard(scores, rows):
    groups = defaultdict(list)
    pairs = defaultdict(lambda: defaultdict(list))
    for score, row in zip(scores, rows, strict=True):
        example = (score, LABELS[row["label"]])
        pairs[row["slot"]][row["label"]].append(score)
        groups["all"].append(example)
        groups[f"language/{row['language']}"].append(example)
        groups["one_turn" if row["text"].startswith("---\n") else "two_turn"].append(example)
    report = {name: binary_metrics(items) for name, items in groups.items()}
    comparable = [pair for pair in pairs.values() if pair.get("hold") and pair.get("shift")]
    margins = [shift - hold for pair in comparable for shift in pair["shift"] for hold in pair["hold"]]
    report["all"]["dialogue_pairs"] = len(margins)
    report["all"]["comparable_dialogues"] = len(comparable)
    report["all"]["shift_only_dialogues"] = sum(not pair.get("hold") for pair in pairs.values())
    report["all"]["within_dialogue_ranking"] = (sum(
        1. if delta > 0 else .5 if delta == 0 else 0. for delta in margins) / len(margins) if margins else None)
    return report


def length_balanced_weights(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        length = len(row["text"].split("\n", 1)[1].split())
        band = next(name for name, end in (("1-3", 3), ("4-7", 7), ("8-15", 15), ("16+", float("inf")))
                    if length <= end)
        groups[row["language"], band].append(index)
    weights = [1.] * len(rows)
    report = {}
    for (language, band), indices in sorted(groups.items()):
        counts = Counter(rows[index]["label"] for index in indices)
        by_label = {label: 1. for label in counts}
        if len(counts) == 2:
            rare = min(counts, key=counts.get)
            common = next(label for label in counts if label != rare)
            by_label[rare] = min(3., len(indices) / (2 * counts[rare]))
            by_label[common] = (len(indices) - counts[rare] * by_label[rare]) / counts[common]
        for index in indices:
            weights[index] = by_label[rows[index]["label"]]
        report[f"{language}/{band}"] = {"counts": dict(counts), "weights": by_label,
                                       "rows_at_cap": sum(weights[index] == 3. for index in indices)}
    return weights, report


def run(dataset, base, output, *, epochs=3, batch_size=32, layers=4, seed=29, device="cuda", backbone="e5",
        balance_length=False):
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    splits, corpus_hash = read_dataset(dataset)
    qwen = backbone == "qwen3"
    expected_sha = QWEN_SHA256 if qwen else BASE_SHA256
    if digest(base / "model.safetensors") != expected_sha:
        raise ValueError("the downloaded encoder does not match its pinned weights")
    receipt = {
        "scope": "fixed-epoch synthetic-only diagnostic", "base": QWEN_BASE if qwen else BASE,
        "base_revision": QWEN_REVISION if qwen else REVISION,
        "base_weights_sha256": expected_sha, "tokenizer_sha256": digest(base / "tokenizer.json"),
        "dataset_manifest_sha256": corpus_hash, "token_budget": TOKEN_BUDGET, "query_prefix": QUERY_PREFIX,
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "top_layers_trainable": layers,
        "device": device, "learning_rate": 2e-5, "torch": str(torch.__version__),
        "turn_boundary": "native tokenizer pair separators",
    }
    if qwen:
        receipt.update(backbone=backbone, query_prefix="", turn_boundary="newline; last real caller token",
                       inference_dtype="float32")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "right"
    if not qwen and pack(tokenizer, "---\nhello") != tokenizer(f"{QUERY_PREFIX}---", "hello")["input_ids"]:
        raise ValueError("the tokenizer changed its native pair layout")
    encode = pack_qwen if qwen else pack
    packed = {name: [encode(tokenizer, row["text"]) for row in rows] for name, rows in splits.items()}
    train = splits["train"]
    truth = [LABELS[row["label"]] for row in train]
    weights = None
    if balance_length:
        weights, distribution = length_balanced_weights(train)
        receipt["training_loss_weights"] = {"method": "language-length-class-v1", "bins": distribution}
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    torch.manual_seed(seed)
    model = build_model(base, layers, backbone).to(device)
    params = {name: weight for name, weight in model.named_parameters() if weight.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=receipt["learning_rate"], weight_decay=.01)
    steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, steps // 10, steps)
    target = torch.nn.BCEWithLogitsLoss(reduction="none" if weights is not None else "mean")
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    history = []
    initial_epoch = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["receipt"] != receipt or set(params) != set(saved["weights"]):
            raise ValueError("training checkpoint belongs to a different run")
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(saved["weights"][name])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        initial_epoch = len(history)
    meter = Progress("turn-text-fit", steps, "batches", path=output / "progress.json")
    meter.skip(initial_epoch * steps_per_epoch)

    def forward(sequences):
        inputs = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(device)
        mixed_precision = use_bf16 and (not qwen or model.training)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=mixed_precision):
            return model(**inputs).float()

    if qwen:
        probe = [encode(tokenizer, "---\nhello"), encode(tokenizer, "what day would work for you\nfriday works for me")]
        model.eval()
        with torch.inference_mode():
            single = forward(probe[:1])[0]
            mixed = forward(probe)[0]
        if not torch.allclose(single, mixed, rtol=0, atol=.0001):
            raise ValueError("right-padding changed the decoder's caller-token logit")

    for epoch in range(initial_epoch, epochs):
        random.Random(seed + epoch).shuffle(order := list(range(len(train))))
        torch.manual_seed(seed + epoch)
        model.train()
        total = 0.
        for position in range(0, len(order), batch_size):
            indices = order[position:position + batch_size]
            logits = forward([packed["train"][i] for i in indices])
            labels = torch.tensor([truth[i] for i in indices], device=device, dtype=torch.float32)
            loss = target(logits, labels)
            if weights is not None:
                factors = torch.tensor([weights[i] for i in indices], device=device, dtype=torch.float32)
                loss = (loss * factors).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(indices)
            meter.advance(1)
        model.eval()
        probabilities = []
        for position in range(0, len(packed["validation"]), batch_size):
            with torch.inference_mode():
                probabilities.extend(forward(packed["validation"][position:position + batch_size]).sigmoid().tolist())
        metrics = scorecard(probabilities, splits["validation"])
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), "validation": metrics})
        print(json.dumps({"epoch": epoch + 1, **metrics["all"]}), flush=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"receipt": receipt, "weights": {name: value.detach().cpu() for name, value in params.items()},
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history},
                   temporary)
        temporary.replace(checkpoint)
    _write_json(output / "receipt.json", {
        **receipt, "trainable_parameters": sum(value.numel() for value in params.values()),
        "checkpoint_sha256": digest(checkpoint), "history": history,
    })
    meter.finish()


def check_fit_base(receipt, base, corpus_hash, trained_epochs):
    backbone = receipt.get("backbone", "e5")
    if backbone not in {"e5", "qwen3"}:
        raise ValueError("unsupported model backbone in the fit receipt")
    qwen = backbone == "qwen3"
    expected_sha = QWEN_SHA256 if qwen else BASE_SHA256
    expected_base, expected_revision = (QWEN_BASE, QWEN_REVISION) if qwen else (BASE, REVISION)
    if (receipt["dataset_manifest_sha256"] != corpus_hash or receipt["base_weights_sha256"] != expected_sha
            or receipt["base"] != expected_base or receipt["base_revision"] != expected_revision
            or digest(base / "model.safetensors") != expected_sha
            or receipt["tokenizer_sha256"] != digest(base / "tokenizer.json")
            or trained_epochs != receipt["epochs"]
            or receipt["token_budget"] != TOKEN_BUDGET or receipt["query_prefix"] != ("" if qwen else QUERY_PREFIX)):
        raise ValueError("evaluation inputs do not match the completed synthetic fit")
    return backbone


def load_fit(checkpoint, base, device, corpus_hash):
    import torch
    from transformers import AutoTokenizer

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    receipt = saved["receipt"]
    backbone = check_fit_base(receipt, base, corpus_hash, len(saved["history"]))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "right"
    model = build_model(base, receipt["top_layers_trainable"], backbone).to(device).eval()
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(saved["weights"]) != set(parameters):
        raise ValueError("the checkpoint is missing trained parameters")
    with torch.inference_mode():
        for name, parameter in parameters.items():
            parameter.copy_(saved["weights"][name])
    return tokenizer, model


def predict(tokenizer, model, texts, device):
    import torch

    encode = pack_qwen if model.backbone == "qwen3" else pack
    tokens = [encode(tokenizer, text) for text in texts]
    use_bf16 = model.backbone != "qwen3" and device == "cuda" and torch.cuda.is_bf16_supported()
    scores = []
    with torch.inference_mode():
        for index in range(0, len(tokens), 32):
            inputs = tokenizer.pad({"input_ids": tokens[index:index + 32]},
                                   padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                scores.extend(model(**inputs).float().sigmoid().tolist())
    return scores


def evaluate(dataset, base, output, device):
    splits, corpus_hash = read_dataset(dataset)
    rows = splits["validation"]
    checkpoint = output / "checkpoint.pt"
    tokenizer, model = load_fit(checkpoint, base, device, corpus_hash)
    by_view = {}
    modified = {}
    for view in ("available_context", "current_only", "last_word_missing"):
        texts = []
        changed = Counter()
        for row in rows:
            previous, current = row["text"].split("\n")
            if view == "current_only":
                previous = "---"
            if view == "last_word_missing" and " " in current:
                current = current.rsplit(" ", 1)[0]
            text = f"{previous}\n{current}"
            if text != row["text"]:
                changed[row["language"]] += 1
            texts.append(text)
        by_view[view] = scorecard(predict(tokenizer, model, texts, device), rows)
        modified[view] = dict(changed)
    result = {"scope": "synthetic validation only", "checkpoint_sha256": digest(checkpoint),
              "dataset_manifest_sha256": corpus_hash, "views": by_view,
              "modified_rows_by_language": modified,
              "last_word_missing": "drop final whitespace-delimited word when a nonempty prefix remains"}
    _write_json(output / "synthetic-scorecard.json", result)
    print(json.dumps({key: metrics["all"] for key, metrics in by_view.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/turn-completion-v1"))
    parser.add_argument("--base", type=Path, default=data_root() / "cache/turn-text-e5-small")
    parser.add_argument("--output", type=Path, default=data_root() / "runs/turn-text-e5-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--backbone", choices=("e5", "qwen3"), default="e5")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluate", action="store_true", help="score the final checkpoint on synthetic validation")
    parser.add_argument("--balance-length", action="store_true", help="rebalance classes within training length bins")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.layers <= 12:
        parser.error("positive epochs and batch size, and 0–12 trainable encoder layers are required")
    if args.evaluate:
        evaluate(args.dataset, args.base, args.output, args.device)
    else:
        run(args.dataset, args.base, args.output, epochs=args.epochs, batch_size=args.batch_size,
            layers=args.layers, seed=args.seed, device=args.device, backbone=args.backbone,
            balance_length=args.balance_length)
