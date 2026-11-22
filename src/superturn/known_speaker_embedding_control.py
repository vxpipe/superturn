"""Calibrate and check a speaker-similarity screen on known staged call identities."""

from collections import defaultdict
import hashlib
import json
import os

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE, read_audio, silero
from .augment import condition
from .dense_teacher import file_digest
from .harper_valley_benchmark import pinned_split
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .yodas_mono_diarization import assets

JOB = "known-speaker-embedding-control"
WINDOW = RATE * 3
SPEECH_MIN = RATE * 2
CALL_CAP = 3
IMPOSTOR_BUDGET = .01


def embedding_model(root):
    import torch
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    weight = assets(root, download=False)["embedding"]
    old = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        speaker = PretrainedSpeakerEmbedding(str(weight), device=torch.device("cuda"))
    finally:
        if old is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old
    return speaker, file_digest(weight)


def speech_window(pcm, spans):
    """Take the most speech-dense complete 3-second window on a half-second grid."""
    grid = range(0, len(pcm) - WINDOW + 1, RATE // 2)
    cover = lambda start: sum(max(0, min(stop, start + WINDOW) - max(edge, start)) for edge, stop in spans)
    start = max(grid, key=cover, default=None)
    if start is None or cover(start) < SPEECH_MIN:
        return None
    return pcm[start:start + WINDOW], start


def unit(embeddings):
    vectors = np.asarray(embeddings)
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(magnitudes == 0):
        raise ValueError("the embedding model produced invalid vectors")
    return vectors / magnitudes


def pair_scores(people, identities, *, codec_pair):
    """Score one person pair using its best distinct-recording comparison."""
    positives, negatives = [], []
    for i, person in enumerate(identities):
        left = people[person]
        for other in identities[i:]:
            right = people[other]
            a, b = codec_pair
            candidates = [float(recording_a[a] @ recording_b[b])
                          for recording_a in left for recording_b in right if recording_a[0] != recording_b[0]]
            if candidates:
                (positives if person == other else negatives).append(max(candidates))
    return np.asarray(positives), np.asarray(negatives)


def threshold(negatives):
    if len(negatives) < 100 or not np.isfinite(negatives).all():
        raise ValueError("the calibration cohort needs at least one hundred finite impostor pairs")
    limit = int(IMPOSTOR_BUDGET * len(negatives))
    index = len(negatives) - limit - 1
    return float(np.nextafter(np.partition(negatives, index)[index], np.inf))


def stats(positive, negative, floor):
    if not len(positive) or not len(negative):
        raise ValueError("both pair types must be represented")
    return {"same_speaker_pairs": len(positive), "different_speaker_pairs": len(negative),
            "same_speaker_matched": int((positive >= floor).sum()),
            "false_matches": int((negative >= floor).sum()),
            "same_speaker_recall": float(np.mean(positive >= floor)),
            "different_speaker_false_match_rate": float(np.mean(negative >= floor)),
            "same_speaker_cosine_p10": float(np.percentile(positive, 10)),
            "different_speaker_cosine_p99": float(np.percentile(negative, 99))}


def run(root):
    import torch

    if not (root / ".campaign-machine").is_file():
        raise ValueError("speaker recordings remain on Vast")
    torch.set_num_threads(2)
    calls, _, audit_sha = pinned_split(root, "train")
    decoder, checkpoint_sha = embedding_model(root)
    detect = silero()
    available = defaultdict(list)
    for row in calls:
        for side in ("caller", "agent"):
            available[row[side]].append((row["id"], side))
    people, members = defaultdict(list), []
    meter = Progress(JOB, len(available), "speakers")
    try:
        for voice, options in sorted(available.items()):
            clips = []
            chosen = sorted(options, key=lambda pair: seed_for(JOB, voice, *pair))
            used = set()
            for call, channel in chosen:
                if call in used:
                    continue
                path = root / "raw/harper-valley/data/audio" / channel / f"{call}.wav"
                pcm = read_audio(path)
                local = speech_window(pcm, detect(pcm))
                if local is None:
                    continue
                native, edge = local
                g711 = condition(native, "telephone", 0)
                clips.extend([native, g711])
                recorded = {"speaker": voice, "call": call, "channel": channel, "start_sample": edge,
                            "native_sha256": hashlib.sha256(native.astype("<i2").tobytes()).hexdigest()}
                members.append(recorded)
                used.add(call)
                if len(used) == CALL_CAP:
                    break
            if clips:
                audio = torch.from_numpy(np.stack(clips).astype(np.float32) / 32768).unsqueeze(1)
                values = unit(decoder(audio))
                call_ids = [r["call"] for r in members[-(len(values) // 2):]]
                people[voice] = [(call, values[2 * ix], values[2 * ix + 1]) for ix, call in enumerate(call_ids)]
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    multiple = [voice for voice, rows in people.items() if len(rows) >= 2]
    # Alternate hash-ranked voices with two or more calls, then distribute singleton voices.
    rounds = (multiple, people.keys() - set(multiple))
    roles = [[], []]
    for partition in rounds:
        for index, voice in enumerate(sorted(partition, key=lambda key: seed_for(JOB, "split", key))):
            roles[index % 2].append(voice)
    a, b = roles
    native_a, native_b = [pair_scores(people, group, codec_pair=(1, 1)) for group in roles]
    cross_a, cross_b = [pair_scores(people, group, codec_pair=(1, 2)) for group in roles]
    phone_a, phone_b = [pair_scores(people, group, codec_pair=(2, 2)) for group in roles]
    cutoff = max(threshold(native_a[1]), threshold(cross_a[1]))
    private = root / "runs" / JOB
    private.mkdir(parents=True, exist_ok=True)
    membership = private / "membership.json"
    _write_json(membership, {"calibration_speakers": a, "assessment_speakers": b, "windows": members})
    report = {"scope": "known staged training-role speaker matches; separate hashed calibration and assessment speakers",
              "comparison": "maximum of up to 9 distinct-call similarities per person pair",
              "harper_audit_sha256": audit_sha, "embedding_checkpoint_sha256": checkpoint_sha,
              "membership_sha256": file_digest(membership), "cosine_threshold": cutoff,
              "calibration_speakers": len(a), "assessment_speakers": len(b), "windows": len(members),
              "calibration_native": stats(*native_a, cutoff), "assessment_native": stats(*native_b, cutoff),
              "calibration_native_to_g711": stats(*cross_a, cutoff), "assessment_native_to_g711": stats(*cross_b, cutoff)}
    independent_vetoes = {}
    for name, (calibration, assessment) in {
        "native": (native_a, native_b), "native_to_g711": (cross_a, cross_b),
        "both_g711": (phone_a, phone_b),
    }.items():
        if not len(calibration[0]):
            raise ValueError("known-speaker calibration needs recorded same-speaker pairs")
        verify_all = float(calibration[0].min())
        narrow = threshold(calibration[1])
        independent_vetoes[name] = {
            "false_match_control_cutoff": narrow, "assessment_at_false_match_control": stats(*assessment, narrow),
            "high_recall_veto_cutoff": verify_all, "assessment_at_high_recall_veto": stats(*assessment, verify_all),
        }
    report["calibration_only_cutoffs_by_pair_condition"] = independent_vetoes
    _write_json(root / "reports" / f"{JOB}-codec.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(data_root())
