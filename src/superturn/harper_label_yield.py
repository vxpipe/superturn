"""Inventory additional behavioral labels in the staged HarperValley training calls."""

from collections import Counter, defaultdict
import hashlib
import json

from .ami_calibration import _write_json
from .ami_pilot import group_words
from .audio import RATE, silero
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_audit import BOUNDARY_TOLERANCE, GAP, ROLES, audit_call, label_support
from .harper_valley_benchmark import pinned_split
from .label_audit import _spans
from .labels import Speech
from .progress import Progress
from .shards import seed_for
from .sources import data_root

VOICE_CAP = 24
PER_CALL_ROLE_CLASS_CAP = 4


def events(track, left, right):
    """Classify both roles symmetrically using the next sustained speech bout."""
    spans = sorted((span for span in group_words(track) if left <= span.start < right),
                   key=lambda span: (span.start, span.end, span.speaker))
    active = max((span.end for span in track if span.start < left), default=left)
    outcomes, found = Counter(), []
    for index, span in enumerate(spans):
        overlap = active > span.end
        active = max(active, span.end)
        if span.end + GAP > right or index + 1 == len(spans):
            outcomes["recording_boundary"] += 1
            continue
        next_span = spans[index + 1]
        if overlap or next_span.start < span.end:
            outcomes["overlap"] += 1
            continue
        if next_span.start - span.end < GAP:
            outcomes["short_gap"] += 1
            continue
        if min(next_span.end, right) - next_span.start < RATE:
            outcomes["brief_next_bout"] += 1
            continue
        if index + 2 < len(spans) and spans[index + 2].start < next_span.start + RATE:
            outcomes["competing_next_bout"] += 1
            continue
        label = "hold" if next_span.speaker == span.speaker else "shift"
        found.append((span.speaker, span.end, next_span.start, label))
        outcomes[f"candidate_{label}"] += 1
    return found, outcomes


def one_to_one(first, second, counts, *, endpoint_deltas=None):
    def near(source, other):
        return [[j for j, (role, end, onset, _) in enumerate(other)
                 if role == ar and abs(end - ae) <= BOUNDARY_TOLERANCE
                 and abs(onset - ao) <= BOUNDARY_TOLERANCE] for ar, ae, ao, _ in source]

    left, right = near(first, second), near(second, first)
    matches = []
    for i, hits in enumerate(left):
        if len(hits) == 1 and right[hits[0]] == [i]:
            a, b = first[i], second[hits[0]]
            label = a[3]
            if endpoint_deltas is not None:
                endpoint_deltas.append((b[1] - a[1]) * 1000 / RATE)
            counts[f"comparable_first_{label}"] += 1
            counts[f"comparable_second_{b[3]}"] += 1
            edge = max(a[1], b[1])
            onset = min(a[2], b[2])
            if a[3] != b[3]:
                counts[f"next_speaker_conflict_first_{label}"] += 1
            elif onset - edge < GAP:
                counts[f"shared_deadline_conflict_{label}"] += 1
            else:
                counts[f"agreement_{label}"] += 1
                matches.append((a[0], edge, onset, label))
    counts["first_unmatched"] += len(first) - sum(counts[f"comparable_first_{label}"] for label in ("hold", "shift"))
    counts["second_unmatched"] += len(second) - sum(counts[f"comparable_second_{label}"] for label in ("hold", "shift"))
    return matches


def without_omissions(items, omissions):
    return [row for row in items if not any(
        word.start < row[2] + BOUNDARY_TOLERANCE and word.end > row[1] - BOUNDARY_TOLERANCE
        for word in omissions)]


def capped(rows):
    voices, calls = Counter(), Counter()
    kept = []
    for row in sorted(rows, key=lambda row: seed_for("harper-label-yield", *row)):
        call, role, voice, _, label = row
        key = call, role, label
        if voices[voice] < VOICE_CAP and calls[key] < PER_CALL_ROLE_CLASS_CAP:
            voices[voice] += 1
            calls[key] += 1
            kept.append(row)
    return {"counts_before_cap": dict(Counter(row[-1] for row in rows)),
            "counts_after_cap": dict(Counter(row[-1] for row in kept)),
            "speakers_after_cap": len(voices),
            "calls_after_cap": len({row[0] for row in kept}),
            "speakers_per_class": {label: len({row[2] for row in kept if row[-1] == label})
                                   for label in ("hold", "shift")}}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("this inventory requires staged Vast audio")
    raw = root / "raw/harper-valley"
    reports = root / "reports"
    entries, expected, audit = pinned_split(root, "train")
    membership = json.loads((reports / "harper-history-replication-membership.json").read_text())
    if hashlib.sha256(json.dumps(membership).encode()).hexdigest() != MEMBERSHIP:
        raise ValueError("existing caller cohort changed")
    old = defaultdict(list)
    for call, edge, label in membership:
        old[call].append((edge, label))
    labels, audits = defaultdict(list), defaultdict(Counter)
    detect = silero()
    archived, reproduced, counts = Counter(), [], Counter()
    meter = Progress("harper-label-yield", len(entries), "calls")
    try:
        for case in entries:
            call = case["id"]
            for asset in case["files"]:
                if (raw / asset["path"]).stat().st_size != asset["bytes"]:
                    raise ValueError("staged HarperValley asset size changed")

            def observe(audio, origins, primary, words, omissions, left, right):
                strict = []
                audits["strict_agent"].update(label_support(primary, words, left, right,
                                             omissions=omissions, fixed_pauses=strict, anchor_role="agent"))
                prior_edges = [edge + origins["caller"] for edge, _ in old[call]]
                for edge, label in strict:
                    if not any(abs(edge - prior) <= GAP for prior in prior_edges):
                        labels["strict_agent"].append((call, "agent", case["agent"], edge - origins["agent"], label))
                secondary = [Speech(role, begin + origins[role], end + origins[role])
                             for role in ROLES for begin, end in _spans(audio[role], 0, len(audio[role]))]
                silero_events, s = events(primary, left, right)
                web_events, w = events(secondary, left, right)
                word_events, t = events(words, left, right)
                audits["silero"].update(s)
                audits["webrtc"].update(w)
                audits["words"].update(t)

                for role in ROLES:
                    first, second, lexical = ([row for row in pool if row[0] == role]
                                              for pool in (silero_events, web_events, word_events))
                    agreement = Counter()
                    combined = one_to_one(first, second, agreement)
                    audits[f"silero_vs_webrtc_{role}"].update(agreement)
                    word_comparison = Counter()
                    comparable = without_omissions(combined, omissions)
                    reference = without_omissions(lexical, omissions)
                    word_comparison["dual_timing_omission"] = len(combined) - len(comparable)
                    word_comparison["word_timing_omission"] = len(lexical) - len(reference)
                    triple = one_to_one(comparable, reference, word_comparison)
                    audits[f"dual_vs_words_{role}"].update(word_comparison)
                    for name, pool in (("dual", combined), ("triple", triple)):
                        for _, edge, _, label in pool:
                            if not any(abs(edge - prior) <= GAP for prior in prior_edges):
                                labels[f"{name}_{role}"].append((call, role, case[role], edge - origins[role], label))

            pauses = []
            counts.update(audit_call(raw, call, detect, recover_clock=True, pauses=pauses, observe=observe))
            reproduced.extend((call, edge, label) for edge, label in pauses)
            archived.update(label for _, label in pauses)
            meter.advance(1)
        if counts != expected["counts"] or reproduced != [tuple(row) for row in membership]:
            raise ValueError("the archived timing audit did not reproduce")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    result = {"set": "train", "audit_sha256": audit, "old_membership_sha256": MEMBERSHIP,
              "old": archived, "verified_calls": counts["audited_calls"],
              "voice_cap": VOICE_CAP, "per_call_role_class_cap": PER_CALL_ROLE_CLASS_CAP,
              "candidates": {key: capped(value) for key, value in labels.items()}, "audit": audits}
    _write_json(reports / "harper-label-yield.json", result)
    return result


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    print(json.dumps(run(data_root()), indent=2))
