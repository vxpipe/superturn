"""Measure whether an online turn clock adds signal to frozen audio scores."""

from collections import Counter, defaultdict
import hashlib
import json
import wave

import numpy as np
import torch

from .ami_calibration import _write_json
from .audio import RATE, silero
from .citrinet_frozen_probe import fit_readout
from .dense_teacher import SAMPLES
from .eot_domain_probe import GAP, decision_audio
from .harper_history_probe import recall_area
from .harper_partial_stt import pinned_membership
from .harper_valley_audit import audit_call
from .harper_valley_benchmark import pinned_split
from .language_failure_audit import read_receipt, read_verified
from .own_head_distillation import student_scores
from .own_student_failure_refresh import agree, load_model
from .own_student_fidelity_confirmation import LOCK, NEW_SEEDS, TRIAL
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import english_spans, upsample_prefix


JOB = "harper-online-turn-probe"
ARMS = {"audio": (0,), "audio_elapsed": (0, 1), "elapsed": (1,)}
BINS = (0, 1, 2.5, 5, float("inf"))
NATIVE_RATE = 8000
FRAME = 512
VXPIPE_ACTIVITY_PROFILE = {"revision": "5077b67898093472307e1bc723b15fb29e93920a",
                           "onset_frames": 4, "offset_frames": 16,
                           "onset_threshold": .5, "offset_threshold": .35}


def native_wave(path):
    with wave.open(str(path), "rb") as media:
        if (media.getframerate(), media.getnchannels(), media.getsampwidth()) != (NATIVE_RATE, 1, 2):
            raise ValueError("HarperValley audio differs from the native input format")
        return np.frombuffer(media.readframes(media.getnframes()), "<i2")


def speech_events(vad, pcm, offset, role):
    """Replay the VXPipe activity counters on a causal 8 to 16 kHz interpolation."""
    vad.reset_states()
    samples = np.repeat(pcm.astype(np.float32) / 32768, 2)
    samples[2::2] = (samples[1:-1:2] + samples[3::2]) / 2
    samples = torch.from_numpy(samples)
    speaking, consecutive = False, 0
    profile = VXPIPE_ACTIVITY_PROFILE
    for stop in range(FRAME, len(samples) + 1, FRAME):
        probability = vad(samples[stop - FRAME:stop], RATE).item()
        if speaking:
            condition, required, kind = probability < profile["offset_threshold"], profile["offset_frames"], "end"
        else:
            condition, required, kind = probability >= profile["onset_threshold"], profile["onset_frames"], "start"
        consecutive = consecutive + 1 if condition else 0
        if consecutive == required:
            yield stop + offset, role, kind, stop - FRAME * required + offset
            speaking, consecutive = not speaking, 0


def turn_clock(events, deadlines):
    """Return elapsed time using only the VAD events already emitted at a deadline."""
    by_time = iter(sorted(events))
    event = next(by_time, None)
    start = None
    agent_since_onset = False
    agent_active = False
    caller_active = False
    reason = None
    results = []
    for deadline in deadlines:
        while event is not None and event[0] <= deadline:
            _, role, kind, position = event
            if role == "agent":
                agent_active = kind == "start"
                if agent_active:
                    agent_since_onset = True
            else:
                caller_active = kind == "start"
                if caller_active and (start is None or agent_since_onset or agent_active):
                    reason = "overlap_start" if agent_active else "after_agent" if agent_since_onset else "call_open"
                    start = position
                if caller_active:
                    agent_since_onset = False
            event = next(by_time, None)
        results.append((None if start is None else (deadline - start) / RATE, reason, caller_active, agent_active))
    return results


def harper_rows(root, models):
    from silero_vad import load_silero_vad

    vad = load_silero_vad(onnx=True)
    detect = silero()
    raw = root / "raw" / "harper-valley"
    work, membership = [], {}
    for role in ("train", "validation"):
        pauses, membership[role] = pinned_membership(root, role)
        cases, _, _ = pinned_split(root, role)
        grouped = defaultdict(list)
        for _, call, edge, label, _ in pauses:
            grouped[call].append((edge, label))
        work.extend((role, case, sorted(grouped[case["id"]])) for case in cases if case["id"] in grouped)
    rows = []
    timing = Counter()
    meter = Progress(JOB, len(work), "calls")
    try:
        for split, case, pauses in work:
            for entry in case["files"]:
                if (raw / entry["path"]).stat().st_size != entry["bytes"]:
                    raise ValueError("a staged call asset changed in size")
            call = case["id"]
            evidence, audited = {}, []
            audit_call(raw, call, detect, recover_clock=True, evidence=evidence, pauses=audited)
            if sorted(audited) != pauses:
                raise ValueError("call timing differs from the frozen pause inventory")
            origin = evidence["selected_origins"]
            shift = origin["agent"] - origin["caller"]
            pcm = {who: native_wave(raw / f"data/audio/{who}/{call}.wav") for who in ("caller", "agent")}
            events = [*speech_events(vad, pcm["caller"], 0, "caller"),
                      *speech_events(vad, pcm["agent"], shift, "agent")]
            deadlines = [edge + GAP for edge, _ in pauses]
            clocks = turn_clock(events, deadlines)
            clips = np.stack([decision_audio(upsample_prefix(pcm["caller"], edge + GAP), edge, "telephone")[-SAMPLES:]
                              for edge, _ in pauses])
            scores = np.mean([student_scores(model, clips) for model in models], axis=0)
            for (edge, label), score, (elapsed, source, calling, agent) in zip(pauses, scores, clocks, strict=True):
                timing[f"{label}:{source}"] += 1
                timing[f"{label}:caller_active_at_query"] += calling
                timing[f"{label}:agent_active_at_query"] += agent
                if elapsed is not None:
                    rows.append((split, case["caller"], case["agent"], label == "shift", score, elapsed))
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    if sum(timing[f"{label}:None"] for label in ("hold", "shift")) > .05 * 616:
        raise ValueError("the online clock failed to cover 95% of the pinned pauses")
    return rows, dict(timing), membership


def inputs(audio, elapsed):
    probability = np.asarray(audio).clip(1e-6, 1 - 1e-6)
    return np.column_stack((np.log(probability / (1 - probability)), np.log(np.maximum(.032, elapsed))))


def predict(x, truth, fit, score):
    return {name: fit_readout(x[fit][:, columns], truth[fit], score[:, columns])[0]
            for name, columns in ARMS.items()}


def oof(x, truth, callers):
    voices = sorted(set(callers), key=lambda voice: seed_for(JOB, voice))
    by_voice = {voice: i % 5 for i, voice in enumerate(voices)}
    folds = np.asarray([by_voice[voice] for voice in callers])
    result = {name: np.full(len(x), np.nan) for name in ARMS}
    for fold in range(5):
        held = folds == fold
        for name, values in predict(x, truth, ~held, x[held]).items():
            result[name][held] = values
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("out-of-fold scoring did not cover the cohort")
    return result


def score_view(scores, truth, elapsed, *, overall=True):
    bins, weighted_delta, weight = {}, 0., 0
    for left, right in zip(BINS[:-1], BINS[1:], strict=True):
        using = (elapsed >= left) & (elapsed < right)
        labels = truth[using]
        values = {name: recall_area(value[using], labels).tolist() for name, value in scores.items()
                  if labels.any() and not labels.all()}
        shifts = int(labels.sum())
        bins[f"{left:g}–{right:g}"] = {"holds": len(labels) - shifts, "shifts": shifts, "metrics": values}
        if values:
            weighted_delta += shifts * np.mean(np.asarray(values["audio_elapsed"][:3]) - values["audio"][:3])
            weight += shifts
    if not weight:
        raise ValueError("no turn-length bin has holds and shifts")
    return {"bins": bins, "within_bin_shift_weighted_recall_delta": weighted_delta / weight,
            "overall": {name: recall_area(values, truth).tolist() for name, values in scores.items()} if overall else None}


def intervals(scores, truth, callers, agents):
    rng = np.random.default_rng(19619)
    _, caller = np.unique(callers, return_inverse=True)
    _, agent = np.unique(agents, return_inverse=True)
    deltas = []
    for _ in range(1000):
        weights = [np.bincount(rng.integers(n, size=n), minlength=n)[index]
                   for index in (caller, agent) for n in (int(index.max()) + 1,)]
        picked = np.repeat(np.arange(len(truth)), weights[0] * weights[1])
        labels = truth[picked]
        if labels.any() and not labels.all():
            audio, combined = (recall_area(scores[name][picked], labels) for name in ("audio", "audio_elapsed"))
            deltas.append(float((combined[:3] - audio[:3]).mean()))
    if len(deltas) < 500:
        raise ValueError("too few bootstrap replicates contain both classes")
    return np.quantile(deltas, [.025, .975]).tolist()


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the turn clock uses staged Vast audio")
    lock, digest = read_receipt(root, LOCK)
    reference, trial_sha = read_receipt(root, TRIAL)
    if reference["lock_sha256"] != digest or not reference["gates"]["fixed_longer_schedule"]:
        raise ValueError("fixed epoch-20 checkpoints changed")
    models = [load_model(root, lock["fits"][str(seed)]["epoch20_checkpoint"]) for seed in NEW_SEEDS]
    rows, timing, membership = harper_rows(root, models)
    roles, callers, agents, labels, probabilities, elapsed = zip(*rows, strict=True)
    truth, elapsed = np.asarray(labels), np.asarray(elapsed)
    x = inputs(probabilities, elapsed)
    combined = oof(x, truth, callers)
    harper = score_view(combined, truth, elapsed)
    train = np.asarray(roles) == "train"
    validation_only = score_view(predict(x, truth, train, x[~train]), truth[~train], elapsed[~train])
    archived = json.loads(read_verified(root, reference["paired_scores"]).read_text())["en"]
    external = [(edge, label, turn) for _, edge, _, label, turn in english_spans(root, native_pcm=True)]
    ext_edges, ext_labels, groups = zip(*external, strict=True)
    agree(archived, list(ext_labels), list(groups), "en")
    ext_audio = np.mean([archived["scores"][f"epoch20-{seed}"] for seed in NEW_SEEDS], axis=0)
    ext_elapsed = (np.asarray(ext_edges) + GAP) / RATE
    ext_truth = np.asarray(ext_labels)
    elapsed_distribution = {name: np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist()
                            for name, values in (("harper_online", elapsed), ("external_clip_start", ext_elapsed))}
    print(json.dumps({"elapsed_quantiles_seconds": elapsed_distribution}), flush=True)
    ext_scores = predict(x, truth, np.ones(len(x), dtype=bool), inputs(ext_audio, ext_elapsed))
    ext = score_view(ext_scores, ext_truth, ext_elapsed, overall=False)
    gain = float(np.mean(np.asarray(harper["overall"]["audio_elapsed"][:3]) - harper["overall"]["audio"][:3]))
    ci = intervals(combined, truth, callers, agents)
    result = {"scope": "development diagnostic; frozen six-seed E20 mean; no model input changes",
              "controller": VXPIPE_ACTIVITY_PROFILE, "anchor": "pinned speech edge plus 300 ms",
              "elapsed_quantiles_seconds_0_10_25_50_75_90_100": elapsed_distribution,
              "limitations": ["VAD-inferred agent activity instead of known app playback",
                              "HarperValley train labels used by the upstream owned speech teacher",
                              "the caller-agent intervals condition on fixed out-of-fold predictions",
                              "the external clock starts at the extracted clip boundary"],
              "trial_sha256": trial_sha, "timing_counts": timing, "pause_membership": membership,
              "score_membership_sha256": hashlib.sha256(json.dumps(rows, default=float).encode()).hexdigest(),
              "coverage": {"callers": len(set(callers)), "agents": len(set(agents)), "holds": int((~truth).sum()),
                           "shifts": int(truth.sum())},
              "harper_oof": harper, "harper_train_to_validation": validation_only, "external_within_bin_only": ext,
              "paired_mean_recall_delta": gain, "caller_agent_bootstrap_delta_95": ci,
              "worth_proposing": bool(gain >= .03 and ci[0] > 0 and harper["within_bin_shift_weighted_recall_delta"] > 0
                                      and ext["within_bin_shift_weighted_recall_delta"] >= 0)}
    _write_json(root / "reports" / f"{JOB}.json", result)
    print(json.dumps({key: result[key] for key in ("coverage", "paired_mean_recall_delta",
                                                  "caller_agent_bootstrap_delta_95", "worth_proposing")}))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run(data_root())
