"""Reproduce benchmark-only augmented scorecard rows on Vast."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .ami_calibration import _write_json
from .ami_external_baselines import load_smart_turn
from .audio import RATE, read_audio
from .dense_teacher import TEACHER_SHA256, file_digest
from .eot_domain_probe import DATASET_SHA256, GAP, decision_audio
from .language_failure_audit import TEACHER, read_receipt
from .multilingual_teacher_probe import DATA, pause_spans
from .own_head_distillation import predictions, student_scores
from .own_student_failure_refresh import agree, load_model
from .own_student_fidelity_confirmation import LOCK, NEW_SEEDS, TRIAL
from .own_student_smart_turn_benchmark import FEATURES_SHA256
from .own_teacher_targets import teacher
from .progress import Progress
from .turn1_mini_probe import english_rows, harper_rows
from .wav2vec_harper_probe import extract

MODELS = ("smart-turn-3s", "smart-turn-8s", "livekit-v1.1-mini")


def decoder(root):
    from livekit.local_inference import EOT, EOT_MAX_SAMPLES

    feature = importlib.util.find_spec("_whisper_features")
    if feature is None or not feature.origin or file_digest(Path(feature.origin)) != FEATURES_SHA256:
        raise ValueError("the pinned Smart Turn frontend is missing")
    if EOT_MAX_SAMPLES != 2 * RATE:
        raise ValueError("the archived LiveKit Mini window changed")
    smart, digest = load_smart_turn(root)
    if digest != TEACHER_SHA256:
        raise ValueError("the archived Smart Turn weights changed")
    mini = EOT()

    def predict(pcm, edge, label, group, ordinal):
        clip = decision_audio(pcm, edge, "telephone")
        livekit_pcm = (clip * 32768).astype(np.int16)
        livekit_pcm = np.ascontiguousarray(livekit_pcm[-min(edge + GAP, EOT_MAX_SAMPLES):])
        short = np.pad(clip[-3 * RATE:], (5 * RATE, 0))
        long_score = smart(clip)
        short_score = long_score if np.array_equal(short, clip) else smart(short)
        values = (short_score, long_score, float(mini.predict(livekit_pcm, turn_samples=edge + GAP)))
        return label, group, (ordinal, edge), values

    return predict, {"smart_turn_sha256": digest, "smart_turn_frontend_sha256": FEATURES_SHA256,
                     "livekit_runtime": "0.3.2", "livekit_note": "2 s plus elapsed turn length from supplied clip start",
                     "smart_turn_note": "3 s left-padded to 8 s; native 8 s scored separately"}


def multilingual(root, language):
    path = root / "raw/livekit_multilingual_diagnostic/data" / language / "validation-00000-of-00001.parquet"
    if file_digest(path) != DATA[language]:
        raise ValueError("the multilingual benchmark shard changed")
    dataset = pq.ParquetFile(path)
    for group, row in enumerate(row for batch in dataset.iter_batches(
            batch_size=8, columns=["audio", "language", "cohort", "silence_spans"]) for row in batch.to_pylist()):
        pcm = read_audio(row["audio"]["bytes"])
        for ordinal, edge, label in pause_spans(row, pcm, language):
            yield pcm, edge, label, group, ordinal


def owned_harper(root, rows):
    lock, sha = read_receipt(root, LOCK)
    trial, _ = read_receipt(root, TRIAL)
    if trial["lock_sha256"] != sha or not trial["gates"]["fixed_longer_schedule"]:
        raise ValueError("the frozen E20 checkpoints changed")
    chunks = np.stack([decision_audio(pcm, edge, "telephone")[-3 * RATE:] for pcm, edge, _, _, _ in rows])
    result = {f"epoch20-{seed}": student_scores(
        load_model(root, lock["fits"][str(seed)]["epoch20_checkpoint"]), chunks).tolist() for seed in NEW_SEEDS}
    selected, _ = read_receipt(root, "own-teacher-language-selection")
    name, _, predict, heads, _ = teacher(root, selected)
    if name != TEACHER:
        raise ValueError("the owned speech teacher has changed")
    result[TEACHER] = predictions(extract("balanced-scorecard-harper", predict, chunks), heads).tolist()
    return result


def append_references(root, rows):
    predict, provenance = decoder(root)
    directory = root / "runs/balanced-development-scorecard"
    directory.mkdir(parents=True, exist_ok=True)
    for language in (*sorted(rows, key=lambda lang: (lang != "en", lang)), "harper-88"):
        harper = language == "harper-88"
        if language == "en":
            stimuli = [(*sample, i) for i, sample in enumerate(english_rows(root))]
        elif harper:
            stimuli = [(*sample, i) for i, sample in enumerate(harper_rows(root, native_prefix=True))]
        else:
            stimuli = list(multilingual(root, language))
        source_sha = DATASET_SHA256 if language == "en" else DATA.get(language)
        keys = [(group, item, edge) for _, edge, _, group, item in stimuli]
        membership = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
        labels = [label for _, _, label, _, _ in stimuli]
        groups = [group for _, _, _, group, _ in stimuli]
        if not harper:
            agree(rows[language], labels, groups, language)
        else:
            if (labels.count(0), labels.count(1)) != (40, 48):
                raise ValueError("the independent Harper direction cohort changed")
            rows[language] = {"labels": labels, "turn_ordinals": groups, "scores": owned_harper(root, stimuli)}
        path = directory / f"reference-{language}.json"
        key = {"source_sha256": source_sha, "comparison": provenance, "ordered_pause_membership_sha256": membership}
        if path.is_file():
            receipt = json.loads(path.read_text())
            if any(receipt[k] != v for k, v in key.items()):
                raise ValueError("archived reference provenance changed")
        else:
            scores = defaultdict(list)
            meter = Progress(f"balanced-reference-{language}", len(stimuli), "pauses")
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for expected, result in zip(stimuli, pool.map(lambda x: predict(*x), stimuli), strict=True):
                        label, group, point, values = result
                        if (expected[2], expected[3], (expected[4], expected[1])) != (label, group, point):
                            raise ValueError("benchmark predictions lost pause identity")
                        for name, value in zip(MODELS, values, strict=True):
                            scores[name].append(value)
                        meter.advance(1)
            except BaseException:
                meter.finish("failed")
                raise
            meter.finish()
            receipt = {**key, "scores": dict(scores)}
            _write_json(path, receipt)
        rows[language]["scores"].update(receipt["scores"])
        print(json.dumps({"benchmarked_language": language, "rows": len(labels)}), flush=True)
    return provenance
