"""Audit sampled label structure on the campaign machine with independent VAD."""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json

from .ami_reference import audit_sample
from .audio import RATE, read_audio
from .listening import choose
from .progress import Progress
from .shards import seed_for

FRAME = 320
REVIEW_SIZE = 50
EDGE = 200 * 16


def _spans(pcm, left, right):
    import webrtcvad

    vad = webrtcvad.Vad(2)
    left = max(0, left // FRAME * FRAME)
    right = min(right, len(pcm))
    spans = []
    for start in range(left, right - FRAME + 1, FRAME):
        if vad.is_speech(pcm[start:start + FRAME].astype("<i2", copy=False).tobytes(), RATE):
            if spans and start - spans[-1][1] <= 4 * FRAME:
                spans[-1] = (spans[-1][0], start + FRAME)
            else:
                spans.append((start, start + FRAME))
    return [(start, end) for start, end in spans if end - start >= 3 * FRAME]


def _overlap(spans, start, end):
    return sum(max(0, min(right, end) - max(left, start)) for left, right in spans)


def _decision_window(row, pcm):
    edge = row.pause_start
    tail = row.silence_start if row.silence_start is not None else edge
    if not (3200 <= row.pause_samples and 0 < edge <= tail and tail + row.pause_samples <= len(pcm)):
        return None
    return edge, tail


def _single_track(row, pcm, edge, tail, detect):
    stop = tail + row.pause_samples
    spans = detect(pcm, max(0, edge - RATE), stop + 2 * RATE)
    if _overlap(spans, edge - RATE // 2, edge + FRAME) < 3 * FRAME:
        return "unresolved", "missing preceding speech"
    # Ignore disagreement exactly at the two acoustic boundary estimates.
    interior_start = tail + min(5 * FRAME, row.pause_samples // 4)
    interior_end = stop - (2 * FRAME if tail == edge else 0)
    if _overlap(spans, interior_start, interior_end) >= 3 * FRAME:
        return "contradicted", "speech inside pause interior"
    if row.silence_start is not None:
        if _overlap(spans, edge, min(tail, edge + RATE)) < 3 * FRAME:
            return "unresolved", "original cut continuation unconfirmed"
        return "supported", "speech before and after cut and quiet original tail"
    if row.label == "hold" or row.method == "human-timestamps":
        if _overlap(spans, stop - 3 * FRAME, stop + RATE) < 3 * FRAME:
            return "unresolved", "resumed speech unconfirmed"
    return "supported", "acoustic pause structure"


def _per_track(row, path, edge, read, detect):
    right = edge + row.pause_samples + 3 * RATE
    tracks = {track: detect(read(track), edge - 2 * RATE, right)
              for track in sorted(path.parent.glob("*.Headset-*.wav"))}
    caller = tracks[path]
    endings = [end for start, end in caller if abs(end - edge) <= EDGE]
    if not endings:
        return "unresolved", "caller acoustic offset unconfirmed"
    reference_edge = min(endings, key=lambda end: abs(end - edge))
    if any(_overlap(spans, reference_edge + 3 * FRAME, edge + 3200) >= 2 * FRAME
           for spans in tracks.values()):
        return "contradicted", "independent detector finds speech during decision silence"
    events = sorted((start, end, track) for track, spans in tracks.items()
                    for start, end in spans if start >= reference_edge)
    if not events:
        return "unresolved", "next speech unconfirmed"
    start, end, next_track = events[0]
    if abs(start - edge - row.pause_samples) > EDGE:
        return "unresolved", "next acoustic onset differs"
    if next_track == path:
        expected = "hold"
    elif end - start < RATE and any(track == path for _, _, track in events[1:2]):
        expected = "hold"
    else:
        expected = "shift"
    if expected != row.label:
        return "contradicted", "next detected headset implies different label"
    return "supported", "independent per-headset speech activity"


def audit(root, rows, source_signatures, per_language=REVIEW_SIZE, *, detector="webrtc", device="cpu"):
    detector_settings = {"mode": 2, "frame_ms": 20}
    detect = _spans
    if detector == "speechbrain":
        from .audit_models import SPEECHBRAIN_REVISION, speechbrain_detector

        detect = speechbrain_detector(root, device)
        detector_settings = {"revision": SPEECHBRAIN_REVISION, "activation": 0.5, "deactivation": 0.25}
    selected = list(choose(rows, per_language))
    sampled_ami = {row for _, row in selected if row.source == "ami"}
    ami_rows = [row for row in rows if row.source == "ami"]
    if len(sampled_ami) < per_language:
        reserve = [(key, row) for key, row in choose(ami_rows, 2 * per_language) if row not in sampled_ami]
        selected.extend((f"ami-{key}", row) for key, row in reserve[:per_language - len(sampled_ami)])
    if not selected:
        raise ValueError("no labels to audit")
    read = lru_cache(maxsize=32)(read_audio)
    ami_reference = audit_sample(root, sampled_ami | {row for _, row in selected if row.source == "ami"}) if ami_rows else {}
    by_language, by_method = defaultdict(Counter), defaultdict(Counter)
    by_source, by_label = defaultdict(Counter), defaultdict(Counter)
    ami_detector = Counter()
    ami_manual = Counter()
    cases = []
    meter = Progress(f"label-audit-{detector}", len(selected), "pauses")
    try:
        for key, row in selected:
            path = root / row.audio
            pcm = read(path)
            window = _decision_window(row, pcm)
            if window is None:
                outcome, reason = "invalid", "label refers to samples outside the recording"
            elif row.method == "per-track":
                outcome, reason = _per_track(row, path, window[0], read, detect)
            else:
                outcome, reason = _single_track(row, pcm, *window, detect)
            detector_vote = outcome
            reference_vote = ami_reference.get(row)
            if reference_vote is not None:
                ami_detector[detector_vote] += 1
                ami_manual[reference_vote] += 1
                if outcome != "invalid":
                    if detector_vote == "unresolved":
                        outcome = reference_vote
                        reason = "aligned word timing and dialogue-act evidence"
                    elif reference_vote not in {"unresolved", detector_vote}:
                        outcome = "unresolved"
                        reason = "acoustic and aligned word timing disagree"
            if not key.startswith("ami-"):
                by_language[row.language][outcome] += 1
            by_method[row.method][outcome] += 1
            by_source[row.source][outcome] += 1
            by_label[row.label][outcome] += 1
            identity = seed_for("label-audit", row.source, row.audio, row.speaker, row.pause_start)
            cases.append({"example": key, "id": f"{identity:016x}", "language": row.language, "source": row.source,
                          "method": row.method, "label": row.label, "outcome": outcome, "reason": reason,
                          "detector_vote": detector_vote, "manual_ami_vote": reference_vote})
            meter.advance(1)

        report = {
            "version": 2,
            "method": "structural acoustic audit and separate AMI annotation evidence",
            "detector": detector,
            "limitations": [
                "Read-cut support verifies the acoustic splice and continuation, not a word boundary or semantic hold.",
                "On mixed audio the acoustic detector does not identify speakers; Indic labels use human diarization.",
                "Weak shifts remain segment-end proxies.",
                "The supportive fraction counts unresolved examples in the denominator.",
            ],
            "detector_settings": {**detector_settings, "edge_tolerance_ms": 200},
            "samples_per_language": per_language,
            "label_file_sha256": source_signatures,
            "languages": {key: dict(value) for key, value in sorted(by_language.items())},
            "methods": {key: dict(value) for key, value in sorted(by_method.items())},
            "sources": {key: dict(value) for key, value in sorted(by_source.items())},
            "labels": {key: dict(value) for key, value in sorted(by_label.items())},
            "ami_detector_votes": dict(ami_detector),
            "ami_manual_votes": dict(ami_manual),
            "cases": cases,
        }
        suffix = "" if detector == "webrtc" else "-" + detector
        destination = root / "reports" / f"label-audit{suffix}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return destination


def label_hashes(root, sources):
    result = {}
    for name in sources:
        with (root / "labels" / f"{name}.jsonl").open("rb") as handle:
            result[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result
