"""Test real earlier audio in an owned Whisper teacher while keeping student input at 3 s."""

import argparse
from collections import defaultdict
import hashlib
import json
from functools import lru_cache
import wave

import numpy as np
import torch

from .ami_calibration import _write_json
from .audio import RATE
from .balanced_development_scorecard import UNSEEN_IN_FIT, balanced_threshold, calibration_split, cell, evaluate, frozen, macro
from .balanced_scorecard_references import multilingual
from .dense_teacher import file_digest
from .eot_domain_probe import GAP, decision_audio
from .harper_partial_stt import pinned_membership
from .language_failure_audit import read_receipt, read_verified
from .open_speech_teacher_probe import WHISPER_SHA256, WHISPER_SIZE, WHISPER_URL, checkpoint
from .own_head_distillation import predictions
from .progress import Progress
from .release_cohort import write_once
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_rows, harper_rows, upsample_prefix
from .wav2vec_harper_probe import HEAD_SEEDS, fit_head, training_rows

JOB = "owned-history-teacher"
S3 = 3 * RATE
S8 = 8 * RATE
NOISE_DBFS = -55
ARMS = ("h3", "h8", "h8-ablated")
DRAWS = 500
BATCH = 16


def variants(pcm, edge, key):
    full = decision_audio(pcm, edge, "telephone")
    if full.shape != (S8,):
        raise ValueError("the archived causal eight-second renderer changed")
    seen = min(S8, edge + GAP)
    ablated = full.copy()
    if seen > S3:
        ablated[-seen:-S3] = (np.random.default_rng(seed_for(JOB, key, edge))
                               .standard_normal(seen - S3, dtype=np.float32) * 10 ** (NOISE_DBFS / 20))
    return seen, (full[-S3:], full, ablated)


def encoder(root):
    import whisper

    path = root / "cache/open-speech-encoder/large-v3.pt"
    if (whisper.__version__ != "20250625" or whisper._MODELS["large-v3"] != WHISPER_URL
            or not path.is_file() or not path.with_suffix(".json").is_file()):
        raise ValueError("this experiment requires the staged pinned Whisper model")
    path, _ = checkpoint(root, "whisper-large-v3", WHISPER_URL, WHISPER_SIZE, sha256=WHISPER_SHA256)
    core = whisper.load_model(str(path), device="cpu").encoder.eval().to("cuda")
    if (core.conv1.in_channels, len(core.blocks), *core.positional_embedding.shape) != (128, 32, 1500, 1280):
        raise ValueError("unexpected Whisper encoder")
    pooled = []
    masks = {}

    def pool(states):
        frames, before, after = active
        visible = states[:, :frames]
        return torch.cat((visible.mean(1), visible[:, before].mean(1), visible[:, after].mean(1)), dim=-1)

    def first(_, args):
        pooled.append(pool(args[0]))

    def intermediate(_, __, output):
        pooled.append(pool(output))

    core.blocks[0].register_forward_pre_hook(first)
    for block in core.blocks[:-1]:
        block.register_forward_hook(intermediate)
    for samples in (S3, S8):
        centers = torch.arange(samples // 320, device="cuda") * 320
        edge = samples - GAP
        masks[samples] = (len(centers), (centers >= edge - RATE // 2) & (centers < edge), centers >= edge)
    active = masks[S3]

    def predict(waves):
        nonlocal active
        samples = waves.shape[1]
        active = masks[samples]
        results = []
        with torch.inference_mode():
            tensor = torch.as_tensor(waves, dtype=torch.float32, device="cuda")
            for batch in tensor.split(8):
                pooled.clear()
                mel = torch.stack([whisper.log_mel_spectrogram(
                    clip, n_mels=128, padding=whisper.audio.N_SAMPLES - samples
                ) for clip in batch])
                pooled.append(pool(core(mel)))
                if len(pooled) != 33:
                    raise ValueError("Whisper layer readout changed")
                results.append(torch.stack(pooled, dim=1).cpu().numpy())
        return np.concatenate(results)

    for samples in (S3, S8):
        t = np.arange(samples, dtype=np.float32)
        probe = np.stack((np.sin(t / 13) * .1, np.sin(t / 37) * .001))
        singleton = np.concatenate([predict(row[None]) for row in probe])
        if not np.allclose(singleton, predict(probe), rtol=1e-3, atol=5e-4):
            raise ValueError("Whisper history features changed under batching")
    return predict


def training(root):
    rows, digest = pinned_membership(root, "train")
    base_path = root / "raw/harper-valley/data/audio/caller"

    @lru_cache(maxsize=2)
    def native(call):
        with wave.open(str(base_path / f"{call}.wav"), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (8000, 1, 2):
                raise ValueError("unexpected native caller track")
            return np.frombuffer(reader.readframes(reader.getnframes()), "<i2")

    waves, labels, speakers = [], [], []
    for (ordinal, call, edge, label, speaker), (archived, old_label, old_speaker) in zip(
        rows, training_rows(root), strict=True
    ):
        observed = upsample_prefix(native(call), edge + GAP)
        seen, arms = variants(observed, edge, ("train", ordinal))
        if seen != S8:
            continue
        if not np.array_equal(arms[0], archived) or (label == "shift") != old_label or speaker != old_speaker:
            raise ValueError("the historical three-second fit stimuli changed")
        waves.append(arms)
        labels.append(old_label)
        speakers.append(speaker)
    if len(waves) != 528 or (len(labels) - sum(labels), sum(labels)) != (273, 255):
        raise ValueError("the verified eight-second labelled fit subset changed")
    return [np.stack(clips) for clips in zip(*waves, strict=True)], np.asarray(labels), digest, len(set(speakers))


def fit(root, directory, predict):
    archive, teacher_sha = read_receipt(root, "open-speech-whisper-large-v3")
    selected, _ = read_receipt(root, "own-teacher-language-selection")
    old = selected["inputs"]["whisper-large-v3"]
    if old["report_sha256"] != teacher_sha or selected["provisional_default"] != "whisper-large-v3":
        raise ValueError("the registered three-second teacher changed")
    base_head = torch.load(read_verified(root, archive["diagnostic_checkpoint"]), map_location="cpu", weights_only=True)
    sets, truth, membership, voices = training(root)
    penalty = archive["heads"]["whisper-large-v3"]["penalty"]
    protocol = {"teacher_report_sha256": teacher_sha, "encoder_sha256": WHISPER_SHA256,
                "fit_membership_sha256": membership, "fit_holds": 273, "fit_shifts": 255,
                "fit_speakers": voices, "noise_dbfs": NOISE_DBFS, "penalty": penalty,
                "seeds": list(HEAD_SEEDS), "pcm_sha256": [hashlib.sha256(clip.tobytes()).hexdigest() for clip in sets]}
    write_once(directory / "protocol.json", protocol)
    target = directory / "heads.pt"
    if target.exists():
        saved = torch.load(target, map_location="cpu", weights_only=True)
        if saved["protocol"] != protocol:
            raise ValueError("cached teachers use another fit protocol")
        return saved["heads"]
    heads = {}
    meter = Progress(f"{JOB}-fit", len(truth) * len(ARMS), "clips")
    try:
        for arm, waves in zip(ARMS, sets, strict=True):
            blocks = []
            for start in range(0, len(waves), BATCH):
                block = waves[start:start + BATCH]
                blocks.append(predict(block))
                meter.advance(len(block))
            features = np.concatenate(blocks)
            del blocks
            if not np.isfinite(features).all():
                raise ValueError("frozen encoder returned invalid features")
            heads[arm] = [fit_head(features, truth, np.arange(len(truth)), penalty, seed) for seed in HEAD_SEEDS]
            if arm == "h3":
                drift = np.max(np.abs(predictions(features, heads[arm]) - predictions(features, base_head["heads"])))
                if drift > 1e-5:
                    raise ValueError(f"three-second fit head drifted from the archived teacher: {drift}")
            del features
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    pending = target.with_suffix(".partial")
    torch.save({"protocol": protocol, "heads": heads}, pending)
    pending.replace(target)
    return heads


def score_view(loader, heads, predict, meter, archived_h3=None):
    bands, labels, groups = [], [], []
    scores = {arm: [] for arm in ARMS}
    chunks = []

    def flush():
        actual = [np.stack(items) for items in zip(*chunks, strict=True)]
        short = np.asarray([np.array_equal(a, b) for a, b in zip(actual[1], actual[2], strict=True)])
        for arm, wave in zip(ARMS, actual, strict=True):
            if arm == "h3" and archived_h3 is not None:
                continue
            if arm == "h8-ablated":
                features = full.copy()
                if not short.all():
                    features[~short] = predict(wave[~short])
            else:
                features = predict(wave)
                if arm == "h8":
                    full = features
            scores[arm].extend(predictions(features, heads[arm]).tolist())
        meter.advance(len(chunks))
        chunks.clear()

    for ordinal, (pcm, edge, label, group) in enumerate(loader):
        seen, waves = variants(pcm, edge, (group, ordinal))
        bands.append(seen)
        labels.append(label)
        groups.append(group)
        chunks.append(waves)
        if len(chunks) == BATCH:
            flush()
    if chunks:
        flush()
    if archived_h3 is not None:
        if len(archived_h3) != len(labels):
            raise ValueError("archived three-second control has a different cohort size")
        scores["h3"] = list(archived_h3)
        if not np.isfinite(scores["h3"]).all():
            raise ValueError("archived three-second control is invalid")
    return {"labels": labels, "turn_ordinals": groups, "recorded_samples": bands, "scores": scores}


def scope(samples):
    return {"short": samples <= S3, "middle": (samples > S3) & (samples < S8),
            "full": samples >= S8, "long": samples > S3}


def ba(values, labels, cutoff):
    if len(np.unique(labels)) != 2:
        return None
    decision = values >= cutoff
    return float((decision[labels].mean() + (~decision[~labels]).mean()) / 2)


def band_analysis(rows):
    from .own_student_wider_e20 import by_turn, sample_partition

    rng = np.random.default_rng(88003)
    strata, boot = {}, defaultdict(list)
    for lang, row in rows.items():
        truth = np.asarray(row["labels"], dtype=bool)
        groups = np.asarray(row["turn_ordinals"])
        fit_mask = calibration_split(groups, lang)
        test = ~fit_mask
        scores = {arm: np.asarray(row["scores"][arm], dtype=float) for arm in ARMS}
        cutoff = {arm: balanced_threshold(values[fit_mask], truth[fit_mask]) for arm, values in scores.items()}
        masks = scope(np.asarray(row["recorded_samples"])[test])
        cell_by_band = {}
        for name, mask in masks.items():
            labels = truth[test][mask]
            per_arm = {arm: evaluate(values[test][mask], labels, cutoff[arm]) for arm, values in scores.items()} if len(
                np.unique(labels)) == 2 else None
            cell_by_band[name] = {"holds": int((~labels).sum()), "shifts": int(labels.sum()),
                                  "turns": len(set(groups[test][mask])), "metrics": per_arm}
        strata[lang] = cell_by_band
        if cell_by_band["long"]["metrics"] is None:
            continue
        cal_groups, test_groups = by_turn(groups[fit_mask]), by_turn(groups[test][masks["long"]])
        held_truth = truth[test][masks["long"]]
        for _ in range(DRAWS):
            calibration = sample_partition(cal_groups, truth[fit_mask], rng)
            evaluation = sample_partition(test_groups, held_truth, rng)
            values = {}
            for arm, source in scores.items():
                threshold = balanced_threshold(source[fit_mask][calibration], truth[fit_mask][calibration])
                values[arm] = ba(source[test][masks["long"]][evaluation], held_truth[evaluation], threshold)
            boot[lang].append([values["h8"] - values["h3"], values["h8"] - values["h8-ablated"]])

    summaries = {}
    for band in ("short", "middle", "full", "long"):
        cells = {lang: row[band]["metrics"] for lang, row in strata.items() if row[band]["metrics"] is not None}
        summaries[band] = {"languages": list(cells), "models": {
            arm: float(np.mean([metric[arm]["balanced_accuracy"] for metric in cells.values()]))
            for arm in ARMS
        } if cells else {}}
    if len(boot) != len(rows):
        raise ValueError("long-history gate requires both labels in every development language")
    interval = np.quantile(np.mean(list(boot.values()), axis=0), [.025, .975], axis=0).T.tolist()
    return strata, summaries, interval


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the owned-history probe uses staged Vast recordings")
    directory = root / "runs" / JOB
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    reference, trial_sha, _ = frozen(root)
    predict = encoder(root)
    heads = fit(root, directory, predict)
    heads_sha = file_digest(directory / "heads.pt")
    loaders = {"en": lambda: english_rows(root),
               **{lang: (lambda lang=lang: (row[:4] for row in multilingual(root, lang))) for lang in reference if lang != "en"},
               "harper-88": lambda: harper_rows(root, native_prefix=True)}
    meter = Progress(f"{JOB}-evaluate", sum(len(row["labels"]) for row in reference.values()) + 88, "pauses")
    views = {}
    try:
        for lang, loader in loaders.items():
            path = directory / f"predictions-{lang}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["heads_sha256"] != heads_sha:
                    raise ValueError("cached development probabilities use another teacher")
                row = saved["row"]
                meter.skip(len(row["labels"]))
            else:
                # Verify the full English control, then reuse archived H3 on
                # other languages with their pinned labels and turn ordering.
                h3 = reference[lang]["scores"]["whisper-large-v3"] if lang in reference and lang != "en" else None
                row = score_view(loader(), heads, predict, meter, archived_h3=h3)
                _write_json(path, {"heads_sha256": heads_sha, "row": row})
            if lang in reference:
                old = reference[lang]
                if row["labels"] != old["labels"] or row["turn_ordinals"] != old["turn_ordinals"]:
                    raise ValueError("paired development membership changed")
                error = max(abs(a - b) for a, b in zip(row["scores"]["h3"], old["scores"]["whisper-large-v3"], strict=True))
                if error > 1e-5:
                    raise ValueError(f"frozen three-second replay differs on {lang}: {error}")
                old["scores"].update(row["scores"])
            views[lang] = row
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()

    harper = views.pop("harper-88")
    strata, bands, interval = band_analysis(views)
    cells = {lang: cell(row, lang) for lang, row in reference.items()}
    overall = macro(cells, sorted(cells))
    six = macro(cells, UNSEEN_IN_FIT)
    short, long = bands["short"]["models"], bands["long"]["models"]
    english = cells["en"]["models"]
    guards = {"h8_vs_h3_long_ba": long["h8"] - long["h3"], "h8_vs_ablated_long_ba": long["h8"] - long["h8-ablated"],
              "h8_vs_h3_short_ba": short["h8"] - short["h3"],
              "h8_vs_h3_macro_ba": overall["models"]["h8"]["balanced_accuracy"] - overall["models"]["h3"]["balanced_accuracy"],
              "h8_vs_h3_english_ba": english["h8"]["balanced_accuracy"] - english["h3"]["balanced_accuracy"]}
    passed = (guards["h8_vs_h3_long_ba"] >= .02 and guards["h8_vs_ablated_long_ba"] >= .015
              and min(bounds[0] for bounds in interval) > 0 and abs(guards["h8_vs_h3_short_ba"]) < .01
              and min(guards["h8_vs_h3_macro_ba"], guards["h8_vs_h3_english_ba"]) >= 0)
    labels = np.asarray(harper["labels"], dtype=bool)
    if labels.sum() != 48:
        raise ValueError("HarperValley direction membership changed")
    harper_scores = {arm: evaluate(np.asarray(values), labels, english[arm]["threshold"])
                     for arm, values in harper["scores"].items()}
    archived, benchmark_sha = read_receipt(root, "balanced-development-scorecard-references")
    if archived["trial_sha256"] != trial_sha:
        raise ValueError("reference benchmark aggregates use another split")
    for current, existing in ((overall, archived["macro"]), (six, archived["zero_supervision_macro"]),
                              *((value, archived["languages"][lang]) for lang, value in cells.items())):
        current["benchmark_aggregates"] = {name: value for name, value in existing["models"].items()
                                          if name.startswith(("livekit-", "smart-turn-"))}
    report = {"scope": "owned Whisper window diagnostic on frozen development groups", "trial_sha256": trial_sha,
              "benchmark_sha256": benchmark_sha, "head_sha256": heads_sha,
              "threshold_source": "whole-turn development calibration halves, fixed by language and arm",
              "h3_control": "archived H3 on 13 languages, after reproducing H3 fit and English; Harper scored directly",
              "pipeline_metrics": "unmeasured", "passes": bool(passed), "interval_order": ["h8-h3", "h8-h8-ablated"],
              "paired_turn_ci95_conditional_on_fixed_head_ensemble": interval, "guards": guards,
              "macro": overall, "zero_supervision_macro": six, "languages": cells, "bands": bands, "strata": strata,
              "harper-88": {**archived["harper-88"], "models": {**archived["harper-88"]["models"], **harper_scores}}}
    _write_json(root / "reports" / f"{JOB}.json", report)
    print(json.dumps({"passes": bool(passed), "guards": guards, "long_intervals": interval}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run(data_root())
