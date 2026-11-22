"""Check blinded mono diarization and word timing against sealed two-track pauses."""

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, silero
from .dense_teacher import file_digest
from .progress import Progress
from .sources import data_root

COHORT_SHA256 = "376bc28baa24c7758f871e63fe9c4e788cc765997f83ad2bd9f9570530f2fb9a"
ASSETS = {
    "pipeline": ("speaker-diarization-3.1", "84fd25912480287da0247647c3d2b4853cb3ee5d",
                 "config.yaml", "5402e3ca79b6cfa5b0ec283eed920cafe45ee39b"),
    "segmentation": ("segmentation-3.0", "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
                     "pytorch_model.bin", "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"),
    "embedding": ("wespeaker-voxceleb-resnet34-LM", "837717ddb9ff5507820346191109dc79c958d614",
                  "pytorch_model.bin", "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"),
}
WORD_TOLERANCE = round(.160 * RATE)
MATCH_TOLERANCE = round(.250 * RATE)


def assets(root, *, download):
    from huggingface_hub import hf_hub_download

    cache = root / "cache/pyannote"
    paths = {}
    for key, (repo, revision, name, expected) in ASSETS.items():
        path = Path(hf_hub_download(f"pyannote/{repo}", name, revision=revision, cache_dir=cache / "hub",
                                    local_files_only=not download))
        if name == "config.yaml":
            payload = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        else:
            actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"official pyannote {key} checksum mismatch")
        paths[key] = path
        if download:
            print(f"verified pyannote {key}: {path.stat().st_size:,} bytes", flush=True)
    return paths


def pipeline(paths):
    import torch
    import yaml
    from pyannote.audio import Pipeline

    config = yaml.safe_load(paths["pipeline"].read_text())
    for key in ("segmentation", "embedding"):
        config["pipeline"]["params"][key] = str(paths[key])
    # These publisher checkpoints contain Lightning metadata. Load only the
    # two official files verified in assets(), using their exact local paths.
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        with NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
            yaml.safe_dump(config, handle)
            handle.flush()
            model = Pipeline.from_pretrained(handle.name)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
    model.to(torch.device("cuda"))
    return model


def nearest_speaker(words, spans):
    by_side = defaultdict(list)
    for begin, end in words:
        center = (begin + end) // 2
        possible = {side for start, stop, side in spans if start <= center < stop}
        if len(possible) == 1:
            side, = possible
            by_side[side].append((begin, end))
    return by_side


def disjoint_bouts(spans):
    """Return the first and last uniquely attributed segments in each active block."""
    if not spans:
        return
    start, end, _ = spans[0]
    first = last = [spans[0]]
    for span in spans[1:]:
        begin, stop, _ = span
        if begin > end:
            yield first, last
            start, end = begin, stop
            first = last = [span]
            continue
        if begin == start:
            first = [*first, span]
        if stop >= end:
            last = [*last, span] if stop == end else [span]
            end = stop
    yield first, last


def speaker_spans(diarization, length):
    intervals = ((max(0, round(segment.start * RATE)), min(length, round(segment.end * RATE)), side)
                 for segment, _, side in diarization.itertracks(yield_label=True))
    return sorted(span for span in intervals if span[0] < span[1])


def predictions(spans, words):
    word_sides = nearest_speaker(words, spans)
    all_quiet, word_supported = [], []
    bouts = list(disjoint_bouts(spans))
    for (_, last), (first, _) in zip(bouts, bouts[1:]):
        if len({span[2] for span in last}) != 1 or len({span[2] for span in first}) != 1:
            continue
        start, edge, side = min(last)
        onset, stop, next_side = max(first)
        if onset - edge < round(.300 * RATE):
            continue
        event = (edge, onset, side == next_side)
        all_quiet.append(event)
        left = any(start <= (begin + end) // 2 < edge and abs(end - edge) <= WORD_TOLERANCE
                   for begin, end in word_sides[side])
        right = any(onset <= (begin + end) // 2 < stop and abs(begin - onset) <= WORD_TOLERANCE
                    for begin, end in word_sides[next_side])
        if left and right:
            word_supported.append(event)
    return all_quiet, word_supported


def anchored_speaker_check(references, true_activity, spans, counts):
    """Check identity only at verified locations, independently of pause recovery."""
    for ref in references:
        left = ref["edge_sample"] - RATE // 10
        right = ref["next_onset_sample"] + RATE // 10
        truth = [{span.speaker for span in true_activity if span.start <= point < span.end}
                 for point in (left, right)]
        mono = [{side for start, stop, side in spans if start <= point < stop} for point in (left, right)]
        label = ref["floor_label"]
        expected_right = "0" if ref["next_role"] == "caller" else "1"
        if truth != [{"0"}, {expected_right}] or any(len(sides) != 1 for sides in mono):
            counts[label]["abstain"] += 1
        else:
            counts[label]["agreement" if (mono[0] == mono[1]) == (truth[0] == truth[1]) else "inversion"] += 1


def score(references, predicted, counts, residuals):
    pairs = []
    for i, ref in enumerate(references):
        for j, (edge, onset, _) in enumerate(predicted):
            delta = edge - ref["edge_sample"]
            delay = onset - ref["next_onset_sample"]
            if max(abs(delta), abs(delay)) <= MATCH_TOLERANCE:
                pairs.append((abs(delta) + abs(delay), i, j, delta, delay))
    matched, used = set(), set()
    for _, i, j, delta, delay in sorted(pairs):
        if i in matched or j in used:
            continue
        matched.add(i)
        used.add(j)
        ref = references[i]
        label = ref["floor_label"]
        correct = predicted[j][2] == (ref["next_role"] == "caller")
        counts[label]["agreement" if correct else "inversion"] += 1
        residuals[label]["edge"].append(1000 * delta / RATE)
        residuals[label]["onset"].append(1000 * delay / RATE)
    for i, ref in enumerate(references):
        if i not in matched:
            counts[ref["floor_label"]]["abstain"] += 1
    return len(predicted) - len(used)


def timing(values):
    if not values:
        return None
    low, median, high = np.percentile(values, [25, 50, 75]).tolist()
    return {"n": len(values), "signed_median_ms": median, "iqr_ms": high - low,
            "within_80ms": float(np.mean(np.abs(values) <= 80))}


def rates(counts):
    matched = counts["agreement"] + counts["inversion"]
    return {"matched": matched, "references": counts.total(),
            "coverage": matched / counts.total(),
            "agreement_among_matched": counts["agreement"] / matched if matched else None}


def run(root):
    import torch
    import whisper

    from .datahive_codec_control import observe_call
    from .harper_transcript_coverage import MEMBERSHIP
    from .open_speech_teacher_probe import WHISPER_SHA256
    from .yodas_mono_control import JOB, mono, pcm_digest

    private = root / "runs" / JOB / "membership.json"
    weight = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(private) != COHORT_SHA256 or file_digest(weight) != WHISPER_SHA256:
        raise ValueError("the sealed HarperValley cohort or Whisper checkpoint changed")
    cohort = json.loads(private.read_text())
    raw_membership = root / "reports/harper-history-replication-membership.json"
    members = json.loads(raw_membership.read_text())
    if hashlib.sha256(json.dumps(members).encode()).hexdigest() != MEMBERSHIP:
        raise ValueError("the HarperValley archive membership changed")
    by_call = defaultdict(set)
    for key, edge, label in members:
        by_call[key].add((edge, label))
    torch.set_num_threads(2)
    diarize = pipeline(assets(root, download=False))
    asr = whisper.load_model(str(weight), device="cuda")
    detect = silero()
    counts = defaultdict(Counter)
    acoustic_counts = defaultdict(Counter)
    anchored_counts = defaultdict(Counter)
    residuals = defaultdict(lambda: defaultdict(list))
    acoustic_residuals = defaultdict(lambda: defaultdict(list))
    label_calls = Counter()
    unmatched = 0
    meter = Progress("yodas-mono-diarization", len(cohort["calls"]), "calls")
    try:
        for call in cohort["calls"]:
            context = {}
            tracks, _ = observe_call(root / "raw/harper-valley", call["call"], by_call[call["call"]], detect,
                                     word_context=context)
            pcm = mono(tracks)
            if pcm_digest(pcm) != call["mono_pcm_sha256"]:
                raise ValueError("the mono rendering changed")
            audio = pcm.astype(np.float32) / 32768
            diarization = diarize({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": RATE})
            decoded = asr.transcribe(audio, language="en", word_timestamps=True, condition_on_previous_text=False,
                                     temperature=0, verbose=None)
            words = [(round(word["start"] * RATE), round(word["end"] * RATE))
                     for segment in decoded["segments"] for word in segment.get("words", ())
                     if word["probability"] >= .5 and word["end"] > word["start"]
                     and any(char.isalpha() for char in word["word"])]
            spans = speaker_spans(diarization, len(audio))
            acoustic, predicted = predictions(spans, words)
            unmatched += score(call["reference"], predicted, counts, residuals)
            score(call["reference"], acoustic, acoustic_counts, acoustic_residuals)
            anchored_speaker_check(call["reference"], context["activity"], spans, anchored_counts)
            label_calls.update({ref["floor_label"] for ref in call["reference"]})
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    totals = sum(counts.values(), Counter())
    overall = {key: timing(residuals["hold"][key] + residuals["shift"][key]) for key in ("edge", "onset")}
    agreement = totals["agreement"] / totals.total()
    passed = (agreement >= .85 and min(label_calls.values()) >= 10 and
              all(row and abs(row["signed_median_ms"]) <= 80 and row["iqr_ms"] <= 160
                  for row in overall.values()))
    report = {"scope": "HarperValley training-role mixed mono control", "cohort_sha256": COHORT_SHA256,
              "next_speaker_agreement_all": agreement, "passed": passed, "counts": dict(totals),
              "matched_rates": rates(totals), "matched_rates_by_floor_label": {
                  label: rates(value) for label, value in counts.items()},
              "diarization_only_diagnostic": {label: {**dict(value), **rates(value)}
                                              for label, value in acoustic_counts.items()},
              "reference_anchored_identity_diagnostic": {label: {**dict(value), **rates(value)}
                                                         for label, value in anchored_counts.items()},
              "pyannote_min_duration_off": diarize.segmentation.min_duration_off,
              "counts_by_floor_label": counts, "calls_by_floor_label": label_calls,
              "timing": overall, "timing_by_floor_label": {
                  label: {key: timing(values) for key, values in streams.items()}
                  for label, streams in residuals.items()}, "unmatched_predictions": unmatched,
              "matching_window_ms": 250, "word_support_window_ms": 160, "whisper_sha256": WHISPER_SHA256,
              "pyannote_assets": ASSETS}
    _write_json(root / "reports/yodas-mono-diarization-v2.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="download and check the official pinned weights")
    args = parser.parse_args()
    root = data_root()
    if not (root / ".campaign-machine").is_file():
        raise ValueError("run the mono control on Vast")
    if args.stage:
        assets(root, download=True)
    else:
        run(root)
