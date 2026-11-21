"""Test independent word timing against Silero on the six staged phone calls."""

from collections import Counter
import csv
from importlib.metadata import version
import json

import numpy as np
import torch
import whisper

from .ami_calibration import _write_json
from .ami_pilot import _peak
from .audio import RATE, read_audio, silero
from .datahive_conversation_pilot import inventory, load_call
from .dense_teacher import file_digest
from .dral_label_pilot import isolation, summarize
from .harper_label_yield import one_to_one
from .label_audit import _spans
from .labels import Speech
from .open_speech_teacher_probe import WHISPER_SHA256
from .progress import Progress
from .sources import data_root

BLOCK = 20 * RATE
CONTEXT = 5 * RATE
GAP = 300 * RATE // 1000
TOLERANCE = 160 * RATE // 1000
NULL_OFFSETS = (-700, 700)
ORIGINAL_SHA = "f25b618eb7a1aab261b5a15eddc7376431cc70ad3c7343267f1806852075bae1"


def language_vote(model, pcm):
    audio = whisper.pad_or_trim(pcm.astype(np.float32) / 32768)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    with torch.inference_mode():
        _, probabilities = model.detect_language(mel)
    return sorted(probabilities.items(), key=lambda x: (-x[1], x[0]))[:2]


def reference_languages(root, model):
    """Small read-speech control on distinct publisher speaker IDs."""
    raw = root / "raw/common_voice"
    result = {}
    for language in ("hi", "pa-IN", "ur", "id"):
        folder = raw / f"audio/{language}/train/{language}_train_0"
        available = {path.name: path for path in folder.glob("*.mp3")}
        tsv = raw / f"transcript/{language}/train.tsv"
        with tsv.open(newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            eligible = sorted((row["path"], row["client_id"]) for row in rows
                              if row["path"] in available and row["client_id"])
        speakers, chosen = set(), []
        for filename, speaker in eligible:
            if speaker not in speakers:
                path = available[filename]
                pcm = read_audio(path)
                if len(pcm) < 2 * RATE:
                    continue
                speakers.add(speaker)
                chosen.append({"file": filename, "sha256": file_digest(path), "samples": len(pcm),
                               "top_two": language_vote(model, pcm)})
                if len(chosen) == 12:
                    break
        if not chosen:
            raise ValueError(f"{language} reference has no staged speakers")
        result[language] = {"inventory_sha256": file_digest(tsv), "clips": chosen,
                            "predictions": Counter(x["top_two"][0][0] for x in chosen)}
    return result


def recognize_words(model, tracks):
    """Keep timed words from fixed ASR blocks chosen independently of Silero."""
    words, weak, blocks = [], [], [[], []]
    length = tracks.shape[1]
    for side, track in enumerate(tracks):
        for start in range(0, length, BLOCK):
            stop = min(start + BLOCK, length)
            left = max(0, start - CONTEXT)
            right = min(length, stop + CONTEXT)
            sample = track[left:right]
            ranked = language_vote(model, sample)
            blocks[side].append({"start": start, "end": stop, "top_two": ranked})
            decoded = model.transcribe(sample.astype(np.float32) / 32768, language=ranked[0][0],
                                       word_timestamps=True, condition_on_previous_text=False,
                                       temperature=0)
            for segment in decoded["segments"]:
                for word in segment.get("words", ()):
                    begin = left + round(word["start"] * RATE)
                    end = left + round(word["end"] * RATE)
                    if not (left <= begin < end <= right and start <= (begin + end) // 2 < stop):
                        continue
                    text = word["word"]
                    span = Speech(str(side), begin, end)
                    if word["probability"] >= .5 and any(char.isalpha() for char in text):
                        words.append(span)
                    else:
                        weak.append(span)
    return words, weak, blocks


def pair_events(acoustic, lexical, words, weak, tracks, *, late_word_samples=0, deltas=None):
    """Match both boundaries while holding the original Silero query edge fixed."""
    counts, retained = Counter(), []
    following = [[j for j, b in enumerate(lexical)
                  if a[0] == b[0] and abs(a[1] - b[1]) <= TOLERANCE and abs(a[2] - b[2]) <= TOLERANCE]
                 for a in acoustic]
    uses = Counter(j for row in following for j in row)
    for (side, edge, onset, label), hits in zip(acoustic, following, strict=True):
        if len(hits) != 1 or uses[hits[0]] != 1:
            counts["missing_or_ambiguous_match"] += 1
            continue
        _, word_edge, word_onset, word_label = lexical[hits[0]]
        counts["boundary_matched"] += 1
        if deltas is not None:
            deltas.append([(word_edge - edge) * 1000 / RATE, (word_onset - onset) * 1000 / RATE])
        if word_label != label:
            counts["next_speaker_conflict"] += 1
            continue
        if word_edge > edge + late_word_samples:
            counts["word_offset_after_query_start"] += 1
            continue
        if min(onset, word_onset) < edge + GAP:
            counts["next_onset_before_deadline"] += 1
            continue
        if any(span.start < edge + GAP and span.end > edge
               and not (late_word_samples and span.speaker == side and span.start < edge and span.end == word_edge)
               for span in words):
            counts["word_in_query"] += 1
            continue
        if any(span.start < min(onset, word_onset) + TOLERANCE and span.end > edge - TOLERANCE
               for span in weak):
            counts["uncertain_word"] += 1
            continue
        focal = int(side)
        next_side = focal if label == "hold" else 1 - focal
        peaks = []
        for who, left, right in ((focal, word_edge - RATE // 5, word_edge),
                                 (next_side, onset, onset + RATE // 5)):
            own = _peak(tracks[who], max(0, left), right)
            other = _peak(tracks[1 - who], max(0, left), right)
            if own < max(128, 1.414 * other):
                break
            peaks.append(own)
        if len(peaks) != 2:
            counts["not_channel_dominant"] += 1
            continue
        quiet = max(_peak(track, edge, edge + GAP) for track in tracks)
        if quiet > .5 * min(peaks):
            counts["energy_in_query"] += 1
            continue
        counts[f"supported_{label}"] += 1
        retained.append({"side": focal, "edge": edge, "onset": onset, "label": label,
                         "word_edge": word_edge, "word_onset": word_onset})
    return retained, counts


def language_support(row, blocks, activity, nominal):
    who = row["side"]
    other = who if row["label"] == "hold" else 1 - who
    detected = []
    for side, point in ((who, row["edge"] - 1), (other, row["onset"])):
        cell = blocks[side][point // BLOCK]
        start, end = cell["start"], cell["end"]
        voiced = sum(max(0, min(stop, end) - max(begin, start)) for begin, stop in activity[side])
        (language, probability), (_, runner_up) = cell["top_two"]
        supported = voiced >= 4 * RATE and probability >= .7 and probability - runner_up >= .25
        detected.append(language if supported else "uncertain")
    return {"supported_block_languages": detected,
            "publisher_language_supported": all(lang == nominal for lang in detected),
            "hindustani_block_support": nominal == "hi" and all(lang in {"hi", "ur"} for lang in detected)}


def inspect(tracks, detect, model, nominal):
    length = tracks.shape[1]
    activity = [detect(track) for track in tracks]
    channel_status = isolation(tracks, activity)
    primary, silero_stats = summarize([Speech(str(i), lo, hi) for i, items in enumerate(activity)
                                        for lo, hi in items], length)
    secondary, web_stats = summarize([Speech(str(i), lo, hi) for i, track in enumerate(tracks)
                                       for lo, hi in _spans(track, 0, length)], length)
    web_agreement = Counter()
    one_to_one(primary, secondary, web_agreement)
    words, weak, blocks = recognize_words(model, tracks)
    lexical, word_stats = summarize(words, length)
    deltas = []
    candidates, timing = pair_events(primary, lexical, words, weak, tracks, deltas=deltas)
    _, offset_sensitivity = pair_events(primary, lexical, words, weak, tracks,
                                        late_word_samples=80 * RATE // 1000)
    offsets = {}
    for ms in NULL_OFFSETS:
        delta = ms * RATE // 1000
        shifted = [Speech(span.speaker, span.start + delta, span.end + delta) for span in words
                   if 0 <= span.start + delta < span.end + delta <= length]
        uncertain = [Speech(span.speaker, span.start + delta, span.end + delta) for span in weak
                     if 0 <= span.start + delta < span.end + delta <= length]
        null_events, _ = summarize(shifted, length)
        _, offsets[str(ms)] = pair_events(primary, null_events, shifted, uncertain, tracks)
    for row in candidates:
        row.update(language_support(row, blocks, activity, nominal))
    return {"isolation": channel_status, "silero": silero_stats, "webrtc": web_stats,
            "webrtc_agreement": web_agreement, "lexical": word_stats, "ambiguous_words": len(weak),
            "silero_vs_words": timing, "reciprocal_boundary_deltas_ms": deltas,
            "late_word_offset_sensitivity": offset_sensitivity, "time_shifted_controls": offsets, "lid_blocks": blocks,
            "provisional_pauses": candidates}


def run(root):
    source, folder, metadata, by_name = inventory(root)
    original = root / "reports/datahive-conversation-pilot.json"
    if file_digest(original) != ORIGINAL_SHA:
        raise ValueError("the original audit report changed")
    old = json.loads(original.read_text())
    checkpoint = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(checkpoint) != WHISPER_SHA256:
        raise ValueError("existing Whisper Large v3 checkpoint changed")
    model = whisper.load_model(str(checkpoint), device="cuda")
    detect = silero()
    checksums = source.get("audio_sha256")
    calls = {}
    meter = Progress("datahive-pause-rescue", len(checksums) + 1, "stages")
    try:
        control = reference_languages(root, model)
        meter.advance(1)
        for filename, digest in sorted(checksums.items()):
            tracks, receipt = load_call(folder, filename, digest, by_name)
            if any(value != old["calls"][filename][key] for key, value in receipt.items()):
                raise ValueError(f"input changed for {filename}")
            calls[filename] = {**receipt, **inspect(tracks, detect, model, receipt["language"])}
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    result = {"scope": "behavioral and language feasibility, one train-side component",
              "source_revision": source.get("revision"), "metadata_sha256": file_digest(metadata),
              "original_report_sha256": ORIGINAL_SHA, "whisper_sha256": WHISPER_SHA256,
              "dependencies": {key: version(key) for key in ("av", "torch", "openai-whisper", "silero-vad")},
              "native_timeline": {"rate": RATE, "deadline_samples": GAP, "tolerance_samples": TOLERANCE,
                                  "lexical_block_samples": BLOCK, "lexical_context_samples": CONTEXT},
              "lid_reference": control, "calls": calls}
    _write_json(root / "reports/datahive-pause-rescue.json", result)


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
