"""Produce caller soft targets with both participants excluded from each head fit."""

from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import json
import wave

import numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr
import torch

from .ami_calibration import _write_json
from .audio import read_audio
from .dense_teacher import SAMPLES, file_digest
from .eot_domain_probe import GAP, decision_audio
from .harper_partial_stt import pinned_membership
from .harper_valley_benchmark import pinned_split
from .language_failure_audit import read_receipt, read_verified
from .model import build
from .own_head_distillation import pool_rows, predictions, student_scores
from .own_student_width_probe import model_name
from .own_teacher_feature_student import DEVICE, SEEDS, read_inputs
from .own_teacher_targets import teacher
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import harper_rows, upsample_prefix
from .wav2vec_harper_probe import extract, fit_head, pause_rows, predict_logits, rates, training_rows

PENALTY = .05
VOICE_CAP = 48
HISTORY_BINS = (19200, SAMPLES - GAP)


def candidates(root, receipt):
    rows = json.loads(read_verified(root, receipt["harper_candidates"]).read_text())
    selected = sorted((row for row in rows if row["role"] == "caller" and row["purged_head_supported"]
                       and not any(row[key] for key in ("counterpart_speech_in_query", "unverified_words_in_query",
                                                       "near_original_caller_pause"))),
                      key=lambda row: seed_for("harper-soft-v1", row["call"], row["edge_16khz"]))
    if len(selected) != receipt["harper"]["caller"]["new_quiet_pauses"]:
        raise ValueError("the candidate receipt changed")
    count, keep = Counter(), []
    for row in selected:
        person = row["person"]
        if count[person] < VOICE_CAP:
            keep.append(row)
            count[person] += 1
    return sorted(keep, key=lambda row: (row["call"], row["edge_16khz"]))


def render(root, rows):
    @lru_cache(maxsize=1)
    def native(call, digest):
        path = root / "raw/harper-valley/data/audio/caller" / f"{call}.wav"
        if sha256(read_audio(path).tobytes()).hexdigest() != digest:
            raise ValueError("a staged caller track changed after census")
        with wave.open(str(path), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (8000, 1, 2):
                raise ValueError("unexpected caller audio format")
            return np.frombuffer(reader.readframes(reader.getnframes()), "<i2")

    clips = []
    for row in rows:
        edge = row["edge_16khz"]
        prefix = upsample_prefix(native(row["call"], row["audio_sha256"]), edge + GAP)
        clips.append(decision_audio(prefix, edge, "telephone")[-SAMPLES:])
    return np.stack(clips)


def emilia_control(pool, caller):
    """Match the available history bin and prefer underrepresented voices."""
    eligible = defaultdict(list)
    for idx, (_, row) in enumerate(pool):
        if row["source"] == "emilia_yodas" and row["role"] == "fit" and row["language"] == "en":
            eligible[int(np.digitize(row["pause_start"], HISTORY_BINS))].append(idx)
    used, match, gaps = Counter(), [], []
    for row in caller:
        bin_index = int(np.digitize(row["edge_16khz"], HISTORY_BINS))
        supported = [bucket for bucket, options in eligible.items() if options]
        bucket = min(supported, key=lambda group: (abs(group - bin_index), group))
        options = eligible[bucket]
        index = min(options, key=lambda i: (used[pool[i][1]["speaker"]],
                                            seed_for("telephone-emilia-v1", pool[i][0])))
        options.remove(index)
        used[pool[index][1]["speaker"]] += 1
        match.append(index)
        gaps.append(bucket - bin_index)
    return np.asarray(match, dtype=np.int64), gaps, len(used)


def purged_scores(features, labels, dyads, query_actors):
    grouped = defaultdict(list)
    for index, pair in enumerate(query_actors):
        grouped[pair].append(index)
    outputs = np.empty(len(query_actors), dtype=np.float64)
    kept = []
    meter = Progress("own-telephone-purged-heads", len(grouped), "dyads")
    try:
        for actors, queries in grouped.items():
            retained = np.array([index for index, pair in enumerate(dyads) if not (actors & pair)])
            if len(retained) < 300 or len(np.unique(labels[retained])) != 2:
                raise ValueError("insufficient head labels after participant purge")
            head = fit_head(features, labels, retained, PENALTY, seed=29)
            outputs[queries] = expit(predict_logits(features[queries], head))
            kept.append(len(retained))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    return outputs, [min(kept), max(kept)], len(grouped)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("telephone targets use staged Vast audio")
    census, census_hash = read_receipt(root, "own-telephone-inventory")
    rows = candidates(root, census)
    train, membership = pinned_membership(root, "train")
    calls, _, audit = pinned_split(root, "train")
    if (membership, audit) != (census["harper"]["teacher_membership_sha256"], census["harper"]["clock_audit_sha256"]):
        raise ValueError("Harper source membership changed")
    people = {call["id"]: frozenset((call["caller"], call["agent"])) for call in calls}
    case_order = [people[call] for _, call, _, _, _ in train]
    caller_actors = [people[row["call"]] for row in rows]
    query_order = case_order + caller_actors + caller_actors
    original, labels, groups = zip(*training_rows(root), strict=True)
    if tuple(int(label == "shift") for _, _, _, label, _ in train) != labels or tuple(row[-1] for row in train) != groups:
        raise ValueError("head-label render order changed")
    audio = render(root, rows)
    _, _, pool = pool_rows(root)
    _, _, replay, _, _, _, _, target_hash = read_inputs(root)
    control, history_gaps, emilia_speakers = emilia_control(pool, rows)
    emilia = replay[control, 0].astype(np.float32) / 32768
    validation, held_labels, _ = zip(*pause_rows(harper_rows(root, native_prefix=True)), strict=True)
    selection, selection_hash = read_receipt(root, "own-teacher-language-selection")
    name, descriptor, predict, heads, _ = teacher(root, selection)
    if name != "whisper-large-v3":
        raise ValueError("the planned speech encoder changed")
    # The equal layer mean is fixed before fitting and cannot expose a target dyad.
    full = extract("own-telephone-fixed-layer-pool", predict, np.concatenate((original, audio, emilia, validation)))
    held = predictions(full[-len(validation):], heads)
    features = full[:-len(validation)].mean(axis=1, keepdims=True)
    del full, predict
    width, width_hash = read_receipt(root, "own-student-width-probe")
    fidelity = {}
    for seed in SEEDS:
        item = width["fits"][model_name(1_000_000, seed)]["checkpoint"]
        state = torch.load(read_verified(root, item), map_location="cpu", weights_only=True)
        model = build(state["config"]).to(DEVICE).eval()
        model.load_state_dict(state["state"])
        fidelity[seed] = float(spearmanr(student_scores(model, np.stack(validation)), held).statistic)
    scores, retained, dyads = purged_scores(features, np.asarray(labels), case_order, query_order)
    raw, raw_emilia = np.split(scores[len(labels):], 2)
    target_receipt, normalizer_hash = read_receipt(root, "own-teacher-distillation-targets")
    if normalizer_hash != target_hash:
        raise ValueError("control PCM and normalization provenance disagree")
    normalizers = target_receipt["normalizers"]
    stats = normalizers["by_language"].get("en", normalizers["global"])
    def normalize(values):
        return expit((logit(values.clip(1e-6, 1 - 1e-6)) - stats["mean"]) / stats["std"]).astype(np.float32)

    normalized, normalized_emilia = normalize(raw), normalize(raw_emilia)
    spread = np.quantile(normalized, [.1, .25, .5, .75, .9]).tolist()
    if not np.isfinite(normalized).all() or spread[-1] - spread[0] < .05:
        raise ValueError("the caller soft targets collapsed")
    out = root / "runs/own-telephone-targets"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "caller-rows.json"
    _write_json(manifest, rows)
    score_path = out / "paired-targets.npz"
    temporary = score_path.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez(stream, harper=normalized, emilia=normalized_emilia, emilia_rows=control)
    temporary.replace(score_path)
    report = {"scope": "purged telephone soft targets, own trained head", "census_sha256": census_hash,
              "selection_sha256": selection_hash, "teacher_encoder_sha256": descriptor["encoder_sha256"],
              "width_sha256": width_hash, "normalization_receipt_sha256": target_hash,
              "penalty": PENALTY, "layers": "uniform mean", "retained_label_rows_min_max": retained,
              "dyads": dyads, "pauses": len(rows), "people": len({row["person"] for row in rows}),
              "raw_quantiles": np.quantile(raw, [.1, .25, .5, .75, .9]).tolist(),
              "emilia_raw_quantiles": np.quantile(raw_emilia, [.1, .25, .5, .75, .9]).tolist(),
              "emilia_history_bin_deltas": dict(Counter(str(gap) for gap in history_gaps)),
              "emilia_speakers": emilia_speakers,
              "normalized_quantiles": spread, "purged_labels": rates(scores[:len(labels)], np.asarray(labels)),
              "original_teacher_harper": rates(held, np.asarray(held_labels)), "width_harper_spearman": fidelity,
              "render_sha256": sha256(audio.tobytes()).hexdigest(),
              "rows": {"path": manifest.relative_to(root).as_posix(), "sha256": file_digest(manifest)},
              "scores": {"path": score_path.relative_to(root).as_posix(), "sha256": file_digest(score_path)}}
    _write_json(root / "reports/own-telephone-targets.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
