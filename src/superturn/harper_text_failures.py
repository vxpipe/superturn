"""Aggregate text errors without writing transcripts or per-row scores."""

from collections import Counter, defaultdict

from .turn_text_train import LABELS

FILLERS = frozenset(("uh", "um", "erm", "hmm", "ah", "uhh", "umm"))
SHORT_ANSWER = frozenset(("yes", "yeah", "yep", "no", "nope", "correct", "okay", "ok", "sure", "right", "exactly"))
OPEN_END = frozenset(("a", "an", "the", "my", "your", "to", "of", "for", "with", "and", "or", "but", "because", "if"))
REPAIR = ("i mean", "sorry", "actually", "no wait", "let me rephrase")


def word_band(length):
    for maximum, band in ((3, "1–3"), (6, "4–6"), (9, "7–9"), (13, "10–13"), (23, "14–23")):
        if length <= maximum:
            return band
    return "24+"


def strata(text):
    previous, current = text.split("\n", 1)
    words = current.split()
    if not words:
        raise ValueError("empty caller transcript")
    features = {
        "affirmation_or_denial_opening": len(words) <= 10 and words[0] in SHORT_ANSWER,
        "filler_or_repeated_word": any(word in FILLERS for word in words)
        or any(left == right for left, right in zip(words, words[1:])),
        "possible_repair_phrase": any(f" {phrase} " in f" {current} " for phrase in REPAIR),
        "ends_in_function_word": words[-1] in OPEN_END,
        "previous_turn_at_least_15_words": previous != "---" and len(previous.split()) >= 15,
    }
    return len(words), previous == "---", features


def audit(real_items, training_rows):
    counts = defaultdict(Counter)
    sources = defaultdict(set)
    length_control = defaultdict(lambda: defaultdict(Counter))
    contexts = {row["text"].startswith("---\n") for row in real_items}
    for dataset, rows in (("real", real_items), ("synthetic_train", training_rows)):
        for row in rows:
            if dataset == "synthetic_train" and row["language"] != "en":
                continue
            label = row["label"] if dataset == "synthetic_train" else ("shift" if row["label"] else "hold")
            length, first, features = strata(row["text"])
            if dataset == "synthetic_train" and first not in contexts:
                continue
            wrong = dataset == "real" and (row["score"] >= .5) != bool(LABELS[label])
            groups = [f"length/{word_band(length)}", *(f"{key}/{'yes' if value else 'no'}"
                                                       for key, value in features.items())]
            for group in groups:
                target = counts[group, label]
                target[dataset] += 1
                if dataset == "real":
                    target["errors"] += wrong
                    sources[group, label].add(row["caller"])
            if dataset == "real":
                for feature, value in features.items():
                    cells = length_control[feature, label][length, first]
                    cells[value, "rows"] += 1
                    cells[value, "errors"] += wrong
    summary = {}
    for (group, label), count in sorted(counts.items()):
        total = count["real"]
        summary.setdefault(group, {})[label] = {
            "real": total, "real_callers": len(sources[group, label]), "errors": count["errors"],
            "error_rate": count["errors"] / total if total else None, "synthetic_train": count["synthetic_train"],
        }
    conditioned = {}
    for (feature, label), cells in length_control.items():
        numerator, support = 0., 0
        for cell in cells.values():
            count = min(cell[False, "rows"], cell[True, "rows"])
            if count:
                support += count
                numerator += count * (cell[True, "errors"] / cell[True, "rows"]
                                      - cell[False, "errors"] / cell[False, "rows"])
        conditioned.setdefault(feature, {})[label] = {
            "matched_rows_per_side": support,
            "feature_present_minus_absent_error_rate": numerator / support if support else None,
        }
    return {"scope": "aggregate text forms, fixed threshold 0.5",
            "synthetic_train_context_matched": True,
            "rates_by_label_and_stratum": summary, "exact_length_and_context_adjusted": conditioned}
