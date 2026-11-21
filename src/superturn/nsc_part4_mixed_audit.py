"""Check behavioral pause yield on fixed NSC same-room microphone openings."""

import argparse
from collections import Counter
import json
import re

from .ami_calibration import _write_json
from .audio import RATE, silero
from .dense_teacher import file_digest
from .progress import Progress
from .sources import data_root

NGRAM = 4
MIN_PAUSE = 300 * RATE // 1000
WORD_TOLERANCE = 160 * RATE // 1000
MAX_PAUSE = 2 * RATE
JOB = "nsc-part4-mixed-audit"


def tokens(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold().replace("’", "'"))


def transcript(path):
    """Index complete source transcripts without using their time coordinates."""
    phrases = Counter()
    for line in path.read_text().splitlines():
        text = str(json.loads(line).get("text", ""))
        for fragment in re.split(r"<[^>]*>|\[[^\]]*\]", text):
            words = tokens(fragment)
            phrases.update(tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1))
    return phrases


def owner(phrase, reference):
    counts = [phrases[phrase] for phrases in reference]
    if counts == [1, 0]:
        return 1
    if counts == [0, 1]:
        return 2
    return None


def transcribe(model, pcm):
    output = model.transcribe(pcm.astype("float32") / 32768, word_timestamps=True,
                              condition_on_previous_text=False, temperature=0, verbose=None)
    words = []
    for segment in output["segments"]:
        reliable = segment["no_speech_prob"] <= .6 and segment["avg_logprob"] >= -1
        for word in segment["words"]:
            start, end = (round(word[key] * RATE) for key in ("start", "end"))
            parts = tokens(word["word"])
            # Keep uncertainty in the time sequence so an intervening word
            # can't become an apparent quiet interval by being filtered out.
            if parts or end > start:
                token = parts[0] if reliable and len(parts) == 1 and word["probability"] >= .5 else None
                words.append((start, end, token))
    return output["language"], sorted(words, key=lambda word: word[:2])


def flank(words, start, end, *, final):
    contained = [word for word in words if start <= word[0] and word[1] <= end and word[1] > word[0]]
    if len(contained) < NGRAM:
        return None, "short"
    selected = contained[-NGRAM:] if final else contained[:NGRAM]
    if any(word[2] is None for word in selected):
        return None, "uncertain"
    return selected, None


def audit(activity, words, references):
    result, counts = [], Counter()
    for (start, edge), (onset, end) in zip(activity, activity[1:]):
        gap = onset - edge
        if not MIN_PAUSE <= gap <= MAX_PAUSE:
            continue
        counts["vad_gaps_300_to_2000_ms"] += 1
        deadline = edge + MIN_PAUSE
        if any(left < deadline and edge < right for left, right, _ in words):
            counts["word_inside_pause"] += 1
            continue
        before, before_reason = flank(words, start - WORD_TOLERANCE, edge + WORD_TOLERANCE, final=True)
        after, after_reason = flank(words, onset - WORD_TOLERANCE, end + WORD_TOLERANCE, final=False)
        if before_reason or after_reason:
            counts["short_flank" if "short" in (before_reason, after_reason) else "uncertain_flank"] += 1
            continue
        if (not edge - WORD_TOLERANCE <= before[-1][1] <= edge
                or not max(onset - WORD_TOLERANCE, deadline) <= after[0][0] <= onset + WORD_TOLERANCE):
            counts["word_vad_mismatch"] += 1
            continue
        before, after = (tuple(word[2] for word in flank_words) for flank_words in (before, after))
        first, second = owner(before, references), owner(after, references)
        if first is None or second is None:
            counts["unattributed_flank"] += 1
            continue
        label = "hold" if first == second else "shift"
        if label == "shift" and end - onset < RATE // 2:
            counts["short_partner_response"] += 1
            continue
        counts[label] += 1
        result.append({"edge": edge, "gap": gap, "first": first, "second": second, "label": label,
                       "phrases": (before, after)})
    return result, counts


def cross_check(primary, secondary):
    primary_counts = Counter(row["phrases"] for row in primary)
    by_phrases = {}
    for row in secondary:
        by_phrases.setdefault(row["phrases"], []).append(row)
    confirmed = Counter()
    for row in primary:
        matches = by_phrases.get(row["phrases"], ())
        if (primary_counts[row["phrases"]] == len(matches) == 1
                and abs(matches[0]["gap"] - row["gap"]) <= 80 * RATE // 1000):
            confirmed[row["label"]] += 1
    return confirmed


def run(root, expected_clock_sha):
    import torch
    import whisper

    from .nsc_part4_native_clock import decode_prefix
    from .nsc_part4_native_preflight import SOURCE
    from .open_speech_teacher_probe import WHISPER_SHA256

    if not (root / ".campaign-machine").is_file():
        raise ValueError("original data must remain on Vast")
    folder = root / "raw" / SOURCE
    manifest_file = folder / "MANIFEST.json"
    clock_file = root / "reports/nsc-part4-native-clock.json"
    manifest_sha = file_digest(manifest_file)
    if (file_digest(clock_file) != expected_clock_sha
            or json.loads(clock_file.read_text())["manifest_sha256"] != manifest_sha):
        raise ValueError("the source manifest is not bound to the verified native-clock report")
    manifest = json.loads(manifest_file.read_text())
    seconds = manifest["sample_prefix_seconds"]
    if seconds != 120:
        raise ValueError("the two fixed openings must be 120 seconds")
    for row in manifest["files"]:
        if file_digest(folder / row["local"]) != row["sha256"]:
            raise ValueError("a native opening or transcript changed")
    references = {pair: [transcript(folder / f"pair-{pair}-track-{mic}.jsonl") for mic in (1, 2)]
                  for pair in (1, 2)}
    weight = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(weight) != WHISPER_SHA256:
        raise ValueError("the cached ASR model changed")
    torch.set_num_threads(2)
    model = whisper.load_model(str(weight), device="cuda")
    detect = silero()
    tracks = {}
    meter = Progress(JOB, 4, "tracks")
    try:
        for row in manifest["files"]:
            if row["local"].endswith(".flac"):
                pcm, receipt = decode_prefix(folder / row["local"], seconds, row["format"][0])
                language, words = transcribe(model, pcm)
                tracks[row["pair"], row["channel"]] = (language, words, detect(pcm), receipt)
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    pairs = []
    for pair in (1, 2):
        references_pair = references[pair]
        metrics, candidates = [], []
        for mic in (1, 2):
            language, words, activity, receipt = tracks[pair, mic]
            rows, counts = audit(activity, words, references_pair)
            unrelated, _ = audit(activity, words, references[3 - pair])
            swapped, _ = audit(activity, words, references_pair[::-1])
            if [(row["edge"], row["first"], row["second"]) for row in rows] != [
                    (row["edge"], 3 - row["first"], 3 - row["second"]) for row in swapped]:
                raise ValueError("transcript swap failed to invert the speaker assignments")
            metrics.append({"language": language, "counts": dict(counts), "unrelated_rows": len(unrelated),
                            "swap_inverts_assignments": True, **receipt})
            candidates.append(rows)
        pairs.append({"microphones": metrics,
                      "shared_gap_confirmations": dict(cross_check(*candidates)),
                      "cross_transcript_ngram_overlap": len(references_pair[0].keys() & references_pair[1].keys())})
    total = Counter()
    for pair in pairs:
        total.update(pair["shared_gap_confirmations"])
    passes = (all(total[label] >= 10 for label in ("hold", "shift"))
              and all(pair["shared_gap_confirmations"].get(label, 0) >= 3
                      for pair in pairs for label in ("hold", "shift"))
              and all(mic["unrelated_rows"] == 0 for pair in pairs for mic in pair["microphones"]))
    report = {"scope": "behavioral label feasibility only; no caller-only inputs admitted",
              "primary_microphone": 1, "cross_mic_not_independent": True,
              "source_manifest_sha256": manifest_sha, "clock_sha256": expected_clock_sha,
              "whisper_sha256": WHISPER_SHA256,
              "pairs": pairs, "confirmed_total": dict(total), "passes": passes}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-sha256", required=True)
    args = parser.parse_args()
    run(data_root(), args.clock_sha256)
