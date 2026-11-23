"""Replay verified AMI pauses with native and G.711 mono inputs."""

import argparse
import json
import os
import wave
from collections import Counter, defaultdict

import numpy as np

from .ami_calibration import _write_json
from .augment import condition
from .audio import RATE
from .dense_teacher import file_digest
from .progress import Progress
from .sources import data_root
from .yodas_mono_diarization import predictions, rates, score, speaker_spans, timing

PLAN_DIGEST = "e192a177e020b379d147d75cd378026fa2a94dbbd1ca417e69843db42c07cbde"
LABEL_DIGEST = "d0b013882bed710e2d067f00c49c6db5537a62574444c87e364071612e3797b9"
COMMUNITY_REPO = "pyannote/speaker-diarization-community-1"
COMMUNITY_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
ARMS = ("clean", "telephone")


def load_control(root):
    plan = root / "reports/ami-hard-pauses-plan.json"
    labels = root / "labels/ami-hard-pauses.json"
    raw = root / "raw/ami-hard-pauses"
    manifest = raw / "MANIFEST.json"
    if file_digest(plan) != PLAN_DIGEST or file_digest(labels) != LABEL_DIGEST:
        raise ValueError("the archived AMI control inputs changed")
    receipt = json.loads(labels.read_text())
    if receipt["plan_sha256"] != PLAN_DIGEST or receipt["manifest_sha256"] != file_digest(manifest):
        raise ValueError("the AMI reconstruction manifest changed")
    membership = {}
    for row in receipt["pauses"]:
        label = row["label"]
        if label["split"] != "train":
            raise ValueError("only archived training-role pauses can enter the control")
        membership[row["pause_id"]] = label
    tracks = {row["path"]: row for row in json.loads(manifest.read_text())["files"]}
    cases = []
    matched = set()
    for tile in json.loads(plan.read_text())["tiles"]:
        start, meeting = tile["start"], tile["recording"]
        references = []
        for row in tile["pauses"]:
            caller, following = row["caller"], row["following"]
            key = f"ami:{meeting}:{caller['speaker']}:{caller['end']}"
            if key not in membership:
                continue
            label = membership[key]
            if (label["label"], label["pause_start"]) != (row["label"], caller["end"] - start):
                raise ValueError("the verified pause no longer agrees with its plan")
            references.append({"edge_sample": caller["end"] - start,
                               "next_onset_sample": following["start"] - start,
                               "floor_label": label["label"],
                               "next_role": "caller" if following["speaker"] == caller["speaker"] else "other"})
            matched.add(key)
        if references:
            files = [tracks[f"{meeting}/{start}/{speaker}.wav"] for speaker in sorted(tile["tracks"])]
            cases.append((tile["frames"], files, references))
    if matched != membership.keys() or sum(len(case[2]) for case in cases) != len(membership):
        raise ValueError("the sealed reference membership was not reconstructed exactly")
    return raw, cases


def mix_tracks(raw, frames, files):
    summed = np.zeros(frames, dtype=np.int32)
    for entry in files:
        path = raw / entry["path"]
        if path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError("a restored AMI audio track changed")
        with wave.open(str(path), "rb") as track:
            if (track.getframerate(), track.getnchannels(), track.getsampwidth(), track.getnframes()) != (
                    RATE, 1, 2, frames):
                raise ValueError("the native AMI tracks are not aligned 16 kHz mono PCM")
            summed += np.frombuffer(track.readframes(frames), dtype="<i2")
    return (summed // len(files)).astype("<i2")


def above_telephone_band(audio):
    window = RATE // 10
    blocks = audio[:len(audio) // window * window].reshape(-1, window)
    power = np.abs(np.fft.rfft(blocks * np.hanning(window), axis=-1)) ** 2
    first_bin = 3400 * window // RATE + 1
    return float(power[:, first_bin:].sum() / max(power[:, 1:].sum(), 1))


def community_snapshot(root, *, download):
    from huggingface_hub import snapshot_download

    return snapshot_download(COMMUNITY_REPO, revision=COMMUNITY_REVISION,
                             cache_dir=root / "cache/pyannote/community-1",
                             token=os.environ.get("HF_TOKEN"),
                             allow_patterns=["config.yaml", "*/*.bin", "plda/*.npz"],
                             local_files_only=not download)


def run(root, *, verify_inputs=False, stage_model=False):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("AMI recordings stay on Vast")
    if stage_model:
        community_snapshot(root, download=True)
        return
    raw, cases = load_control(root)
    if verify_inputs:
        for frames, files, _ in cases:
            mix_tracks(raw, frames, files)
        references = Counter(ref["floor_label"] for _, _, rows in cases for ref in rows)
        print(json.dumps({"windows": len(cases), "references": references}, indent=2))
        return

    import torch
    from pyannote.audio import Pipeline

    torch.set_num_threads(2)
    diarize = Pipeline.from_pretrained(community_snapshot(root, download=False))
    diarize.to(torch.device("cuda"))
    counts = {arm: defaultdict(Counter) for arm in ARMS}
    residuals = {arm: defaultdict(lambda: defaultdict(list)) for arm in ARMS}
    surplus = Counter()
    band_power = {arm: [] for arm in ARMS}
    meter = Progress("ami-mono-control", len(cases) * len(ARMS), "renders")
    try:
        for frames, files, references in cases:
            pcm = mix_tracks(raw, frames, files)
            for arm in ARMS:
                rendered = condition(pcm, arm, 0)
                band_power[arm].append(above_telephone_band(rendered))
                waveform = torch.from_numpy(rendered.astype(np.float32) / 32768).unsqueeze(0)
                annotation = diarize({"waveform": waveform, "sample_rate": RATE}).speaker_diarization
                events, _ = predictions(speaker_spans(annotation, frames), ())
                surplus[arm] += score(references, events, counts[arm], residuals[arm])
                meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"scope": "AMI train-only short-window same-versus-changed speaker control",
              "plan_sha256": PLAN_DIGEST, "labels_sha256": LABEL_DIGEST,
              "windows": len(cases), "diarizer": COMMUNITY_REPO,
              "diarizer_revision": COMMUNITY_REVISION, "renders": {}}
    for arm in ARMS:
        total = sum(counts[arm].values(), Counter())
        report["renders"][arm] = {"counts": total, "rates": rates(total),
                                  "all_reference_agreement": total["agreement"] / total.total(),
                                  "median_energy_fraction_above_3400_hz": float(np.median(band_power[arm])),
                                  "unmatched_predictions": surplus[arm],
                                  "classes": {
                                      label: {"counts": stats, **rates(stats),
                                              "timing": {key: timing(values)
                                                         for key, values in residuals[arm][label].items()}}
                                      for label, stats in counts[arm].items()}}
    _write_json(root / "reports/ami-mono-control.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-inputs", action="store_true", help="check restored audio and labels only")
    action.add_argument("--stage-model", action="store_true", help="fetch the pinned Community-1 checkpoint")
    args = parser.parse_args()
    if not args.stage_model:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    run(data_root(), verify_inputs=args.verify_inputs, stage_model=args.stage_model)
