"""Check behavioral hold and shift label yield in the bounded DRAL pilot."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import wave

import numpy as np

from .ami_calibration import _write_json
from .ami_pilot import _peak
from .audio import RATE, silero
from .dense_teacher import file_digest
from .dral_original_pilot import SOURCE
from .harper_label_yield import events, one_to_one, without_omissions
from .label_audit import _spans
from .labels import Speech
from .maptask_word_check import words_for
from .progress import Progress
from .sources import data_root


def read_tracks(path, digest):
    if file_digest(path) != digest:
        raise ValueError("staged original audio differs from the manifest")
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate(), reader.getsampwidth(), reader.getnchannels()) != (RATE, 2, 2):
            raise ValueError("expected paired 16 kHz original tracks")
        return np.frombuffer(reader.readframes(reader.getnframes()), "<i2").reshape(-1, 2).T


def isolation(tracks, activity):
    dominance, leakage = [], [[], []]
    for slot, spans in enumerate(activity):
        opposite = 1 - slot
        for start, end in spans:
            own = _peak(tracks[slot], start, end)
            partner = _peak(tracks[opposite], start, end)
            dominance.append(own >= 1.414 * max(1., partner))
            if own and not any(low < end and high > start for low, high in activity[opposite]):
                leakage[slot].append(20 * np.log10(max(1., partner) / own))
    ratio = float(np.mean(dominance)) if dominance else 0.
    median = [float(np.median(values)) if values else None for values in leakage]
    passed = ratio >= .6 and all(value is not None and value <= -20 for value in median)
    return {"dominance": ratio, "opposite_to_own_db_p50_by_channel": median, "passed": passed}


def interior(items, length):
    return [row for row in items if row[1] >= 3 * RATE and row[2] + 2 * RATE <= length]


def summarize(track, length):
    items, counts = events(track, 0, length)
    kept = interior(items, length)
    counts["insufficient_window_context"] = len(items) - len(kept)
    return kept, counts


def inspect(tracks, detect, recognize, language):
    length = tracks.shape[1]
    activity = [list(detect(pcm)) for pcm in tracks]
    acoustic_status = isolation(tracks, activity)
    if not acoustic_status["passed"]:
        return {"isolation": acoustic_status}, [], []
    primary, p = summarize([Speech(str(i), lo, hi) for i, values in enumerate(activity)
                            for lo, hi in values], length)
    secondary, s = summarize([Speech(str(i), lo, hi) for i, pcm in enumerate(tracks)
                              for lo, hi in _spans(pcm, 0, length)], length)
    acoustic = Counter()
    consensus = one_to_one(primary, secondary, acoustic)
    report = {"isolation": acoustic_status, "silero": p, "webrtc": s, "acoustic_comparison": acoustic}
    if not consensus:
        return report, [], []
    words, uncertain, exclusions = words_for(recognize, tracks, language=language)
    linguistic, lexical = summarize([span for span, _ in words], length)
    word_comparison = Counter()
    supported = one_to_one(without_omissions(consensus, uncertain), without_omissions(linguistic, uncertain),
                           word_comparison, endpoint_deltas=(delays := []))
    report.update({"word_candidates": lexical, "word_exclusions": exclusions,
                   "word_comparison": word_comparison,
                   "supported": Counter(row[3] for row in supported)})
    return report, delays, supported


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("DRAL audio stays on Vast")
    import torch
    import whisper

    torch.set_num_threads(2)
    raw = root / "raw" / SOURCE
    manifest = json.loads((raw / "MANIFEST.json").read_text())
    plan_path = root / "reports/dral-original-pilot-plan.json"
    if file_digest(plan_path) != manifest["plan_sha256"]:
        raise ValueError("staged conversation selection does not match the plan")
    plan = json.loads(plan_path.read_text())["selection"]
    entries = {row["file"]: row["sha256"] for row in manifest["files"]}
    expected = {f"{row['recording']}-{window['percent']}.wav" for row in plan for window in row["ranges"]}
    if set(entries) != expected:
        raise ValueError("one or more planned recordings were not staged")
    checkpoint = Path.home() / ".cache/whisper/tiny.pt"
    if not checkpoint.is_file():
        raise ValueError("existing Whisper Tiny weights are required")
    recognize = whisper.load_model(str(checkpoint), device="cuda")
    detect = silero()
    results, totals = {}, defaultdict(Counter)
    timing, pairs = defaultdict(list), defaultdict(set)
    meter = Progress("dral-label-pilot", len(entries), "windows")
    try:
        for row in plan:
            language, component = row["language"], row["component"]
            for window in row["ranges"]:
                filename = f"{row['recording']}-{window['percent']}.wav"
                tracks = read_tracks(raw / filename, entries[filename])
                report, deltas, supported = inspect(tracks, detect, recognize, language)
                results[filename] = report
                timings = {key: report.get(key, {}) for key in ("acoustic_comparison", "word_comparison")}
                for key, value in timings.items():
                    totals[language].update({f"{key}.{name}": count for name, count in value.items()})
                totals[language]["isolation_passed_windows"] += report["isolation"]["passed"]
                timing[language].extend(deltas)
                for _, _, _, label in supported:
                    totals[language][label] += 1
                    pairs[language, label].add(component)
                meter.advance(1)
        aggregate = {}
        for lang, counts in totals.items():
            offsets = timing[lang]
            aggregate[lang] = {"counts": counts,
                               "dyads_by_class": {label: len(pairs[lang, label]) for label in ("hold", "shift")},
                               "word_endpoint_median_ms": float(np.median(offsets)) if offsets else None,
                               "word_endpoint_iqr_ms": float(np.percentile(offsets, 75) - np.percentile(offsets, 25))
                               if offsets else None}
        report = {"deadline_ms": 300, "source_plan_sha256": manifest["plan_sha256"],
                  "whisper_sha256": file_digest(checkpoint), "languages": aggregate, "windows": results}
        _write_json(root / "reports/dral-label-pilot.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(json.dumps(run(data_root())["languages"], indent=2))
