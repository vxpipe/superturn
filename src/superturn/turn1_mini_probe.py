"""Evaluate the pinned single-channel Turn-1-Mini on fixed causal pause queries."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .audio import RATE, read_audio
from .augment import condition
from .dense_teacher import SAMPLES, file_digest, normalize
from .elapsed_turn_probe import LONG_TURN_SECONDS, metrics, paired_intervals
from .eot_domain_probe import DATASET_SHA256, DECISION_MS, GAP, decision_audio
from .harper_valley_benchmark import STUDENTS, pinned_validation
from .progress import Progress
from .shards import TAIL
from .sources import data_root

REVISION = "d548240397d5449901559eb09f87cdfcad70420e"
REPO = "https://huggingface.co/p99lab/turn-1-mini"
ARTIFACT = "onnx/turn-1-mini.step.onnx"
CHECKPOINT = "e747b182d685c7db489a3ea2e12bdcd65bac506fcddec00ed3cd0bf9f209fd11"
LICENSE = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
SIZE = 21825125
STEP, FRAME = 2560, 320
FRAMES_PER_STEP = STEP // FRAME


def cache_artifact(folder, name, size, digest):
    path = folder / name.rsplit("/", 1)[-1]
    if not path.is_file():
        with urlopen(f"{REPO}/resolve/{REVISION}/{name}", timeout=60) as response:
            value = response.read(size + 1)
        if len(value) != size:
            raise ValueError("pinned publisher asset size changed")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(value)
        if file_digest(temporary) != digest:
            raise ValueError("pinned publisher asset hash changed")
        temporary.replace(path)
    if path.stat().st_size != size or file_digest(path) != digest:
        raise ValueError("pinned cache asset has changed")
    return path


class Turn1:
    def __init__(self, checkpoint):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(checkpoint), options, providers=["CPUExecutionProvider"])
        self.zeros = {i.name: np.zeros(i.shape, dtype=np.float32)
                      for i in self.session.get_inputs() if i.name.startswith("state_")}
        expected = {f"state_{name}" for name in ("audio", "conv1", "conv2", "k", "v", "k_valid", "h6")}
        if self.zeros.keys() != expected:
            raise ValueError("unexpected single-channel streaming state")
        self.outputs = ["probs", *(f"new_{name}" for name in self.zeros)]

    def step(self, state, audio):
        probs, *new = self.session.run(self.outputs, {"audio": audio.reshape(1, STEP), **state})
        if probs.shape != (FRAMES_PER_STEP,) or not np.all((0 <= probs) & (probs <= 1)):
            raise ValueError("unexpected single-channel probabilities")
        return probs, dict(zip(self.zeros, new, strict=True))

    def past(self, pcm):
        complete = len(pcm) // STEP * STEP
        state = self.zeros
        frames = []
        for start in range(0, complete, STEP):
            probs, state = self.step(state, pcm[start:start + STEP])
            frames.extend(probs.tolist())
        return state, frames, pcm[complete:]

    def audit(self, pcm, future, rng):
        state, _, suffix = self.past(pcm)
        valid = len(suffix) // FRAME
        if not valid:
            return None
        remaining = STEP - len(suffix)
        real = np.pad(future[:remaining], (0, max(0, remaining - len(future))))
        fillers = (np.zeros(remaining, np.float32), real,
                   rng.uniform(-0.75, 0.75, remaining).astype(np.float32))
        scores = [self.step(state, np.concatenate((suffix, pad)))[0][:valid] for pad in fillers]
        return max(float(np.max(np.abs(scores[0] - x))) for x in scores[1:])

    def score(self, pcm, use_partial):
        state, frames, suffix = self.past(pcm)
        if use_partial and len(suffix) >= FRAME:
            probs, _ = self.step(state, np.pad(suffix, (0, STEP - len(suffix))))
            frames.extend(probs[:len(suffix) // FRAME].tolist())
        deadline = len(pcm)
        first = max(0, (deadline - GAP + FRAME - 1) // FRAME - 1)
        if not frames or first >= len(frames):
            raise ValueError("no completed score frame in the decision interval")
        return frames[-1], max(frames[first:]), deadline - FRAME * len(frames)


def render(pcm, edge):
    stop = edge + GAP
    if not 0 <= edge <= len(pcm) - GAP:
        raise ValueError("ineligible fixed-deadline pause")
    recent = decision_audio(pcm, edge, "telephone")[-SAMPLES:]
    # Process only the observed caller prefix when producing full-history audio.
    full = normalize(condition(np.pad(pcm[:stop], (0, TAIL - GAP)), "telephone", 0)[:stop])
    return recent, full, normalize(pcm[stop:stop + STEP])


def english_rows(root):
    path = root / "raw/livekit_eot_diagnostic/data/en/validation-00000-of-00002.parquet"
    if file_digest(path) != DATASET_SHA256:
        raise ValueError("English comparison shard changed")
    for group, row in enumerate(pq.ParquetFile(path).iter_batches(batch_size=1,
                                               columns=["audio", "language", "cohort", "silence_spans"])):
        row, = row.to_pylist()
        if row["cohort"] != "general" or row["language"] != "en":
            raise ValueError("unexpected benchmark cohort")
        pcm = read_audio(row["audio"]["bytes"])
        for span in row["silence_spans"]:
            edge = round(span["start"] * RATE)
            end = min(len(pcm), round(span["end"] * RATE))
            if not 0 <= edge <= end or end - edge < GAP:
                continue
            if span["label"] not in {"eot", "hold"}:
                raise ValueError("unexpected benchmark label")
            yield pcm, edge, int(span["label"] == "eot"), group


def harper_rows(root):
    import hashlib

    cases, _, _ = pinned_validation(root)
    report = json.loads((root / "reports/harper-valley-development-stratified.json").read_text())
    provenance = json.loads((root / "reports/harper-valley-development-stratified-provenance.json").read_text())
    membership = [(row["call"], row["fixed_pauses"]) for row in provenance]
    if hashlib.sha256(json.dumps(membership, separators=(",", ":")).encode()).hexdigest() != report["pause_membership_sha256"]:
        raise ValueError("HarperValley pause membership changed")
    allowed = {case["id"]: case["caller"] for case in cases}
    for call, pauses in membership:
        if pauses:
            pcm = read_audio(root / "raw/harper-valley/data/audio/caller" / f"{call}.wav")
            for edge, label in pauses:
                yield pcm, edge, int(label == "shift"), allowed[call]


def run(root):
    import torch

    from .elapsed_turn_probe import checkpoints

    if not (root / ".campaign-machine").is_file():
        raise ValueError("Turn-1-Mini inference uses only staged Vast audio")
    folder = root / "cache/eot-competitors/turn1-mini"
    folder.mkdir(parents=True, exist_ok=True)
    cache_artifact(folder, "LICENSE", 11357, LICENSE)
    model = Turn1(cache_artifact(folder, ARTIFACT, SIZE, CHECKPOINT))
    students, student_hash = checkpoints(root, "cuda")
    cohorts = {"english": list(english_rows(root)), "harper": list(harper_rows(root))}
    expected = {"english": (532, 400), "harper": (40, 48)}
    for key, rows in cohorts.items():
        shifts = sum(row[2] for row in rows)
        if (len(rows) - shifts, shifts) != expected[key]:
            raise ValueError("the existing pause counts did not reproduce")

    rng = np.random.default_rng(317)
    audit = []
    for rows in cohorts.values():
        for index in rng.choice(len(rows), size=25, replace=False):
            pcm, edge, _, _ = rows[index]
            recent, full, future = render(pcm, edge)
            audit.extend(value for clip in (recent, full) if (value := model.audit(clip, future, rng)) is not None)
    use_partial = bool(audit and max(audit) <= 1e-6)

    def evaluate(row):
        pcm, edge, label, group = row
        recent, full, _ = render(pcm, edge)
        return recent, label, group, len(full) / RATE, model.score(recent, use_partial), model.score(full, use_partial)

    results = {}
    meter = Progress("turn1-mini-probe", sum(map(len, cohorts.values())), "pauses")
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            for name, rows in cohorts.items():
                comparison, lag, lengths, labels, groups = defaultdict(list), defaultdict(list), [], [], []
                for start in range(0, len(rows), 16):
                    recent = []
                    for pcm, label, group, length, short, full in workers.map(evaluate, rows[start:start + 16]):
                        recent.append(pcm)
                        labels.append(label)
                        groups.append(group)
                        lengths.append(length)
                        for mode, (instant, cumulative, age) in (("matched-3s", short), ("full-history", full)):
                            comparison[f"{mode}-instant"].append(instant)
                            comparison[f"{mode}-cumulative"].append(cumulative)
                            lag[mode].append(age * 1000 / RATE)
                    audio = torch.as_tensor(np.stack(recent), device="cuda")
                    elapsed = torch.full((len(recent), 1), float(DECISION_MS), device="cuda")
                    with torch.inference_mode():
                        for key, student in students.items():
                            comparison[key].extend(student(audio, elapsed).flatten().cpu().tolist())
                    meter.advance(len(recent))
                scores, truth = {key: np.asarray(values) for key, values in comparison.items()}, np.asarray(labels)
                results[name] = {"overall": metrics(scores, truth, np.arange(len(labels))),
                                 "frame_lag_ms": {mode: [min(values), max(values)] for mode, values in lag.items()}}
                if name == "english":
                    results[name]["elapsed_at_least_3_3s"] = metrics(scores, truth,
                                                                    np.flatnonzero(np.asarray(lengths) >= LONG_TURN_SECONDS))
                    results[name]["turn_bootstrap"] = {
                        student: paired_intervals(scores, truth, groups, {"matched-3s-instant": student})
                        for student in STUDENTS}
        report = {"model_checkpoint_sha256": CHECKPOINT, "model_revision": REVISION,
                  "publisher_license_sha256": LICENSE, "student_reference_sha256": student_hash,
                  "deadline_ms": DECISION_MS, "max_future_effect": max(audit, default=None),
                  "future_checks": len(audit), "partial_chunk_used": use_partial,
                  "full_history": "G.711 on observed prefix; Harper history starts at channel start",
                  "thresholds": "retrospective ROC, no fitted thresholds", "results": results}
        _write_json(root / "reports/turn1-mini-probe.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    print(json.dumps(run(data_root()), indent=2))
