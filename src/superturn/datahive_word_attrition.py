"""Find where timed words stop corroborating the frozen HarperValley pause control."""

from collections import Counter, defaultdict
import hashlib
import json

from .ami_calibration import _write_json
from .audio import RATE, silero
from .datahive_codec_control import observe_call, select_calls
from .datahive_pause_rescue import GAP, TOLERANCE, recognize_words, language_support
from .dense_teacher import file_digest
from .dral_label_pilot import summarize
from .harper_transcript_coverage import MEMBERSHIP
from .harper_valley_benchmark import pinned_split
from .open_speech_teacher_probe import WHISPER_SHA256
from .progress import Progress
from .sources import data_root

JOB = "datahive-word-attrition"


def classify(edge, onset, words, omitted):
    """Require observed words immediately around the fixed decision window."""
    if any(span.start < edge + GAP and edge < span.end for span in words):
        return "word_inside_query", None
    if any(span.start < onset + TOLERANCE and edge - TOLERANCE < span.end for span in omitted):
        return "uncertain_flank", None
    if not any(span.speaker == "0" and edge - TOLERANCE <= span.end <= edge for span in words):
        return "no_caller_word_end", None
    following = [span for span in words if edge + GAP <= span.start <= onset + TOLERANCE]
    if not following:
        return "no_next_word", None
    first = min(span.start for span in following)
    roles = {span.speaker for span in following if span.start <= first + RATE // 20}
    if first < onset - TOLERANCE or len(roles) != 1:
        return "contradictory_or_ambiguous_next_word", None
    return "witnesses_present", "hold" if roles == {"0"} else "shift"


def attrition(reference, audio, words, omitted, activity, blocks=None):
    grouped, _ = summarize(words, audio.shape[1])
    totals = defaultdict(Counter)
    for edge, onset, label in reference:
        counts = totals[label]
        counts["reference"] += 1
        state, predicted = classify(edge, onset, words, omitted)
        counts[state] += 1
        if predicted is not None:
            counts["raw_agrees" if predicted == label else "raw_inverts"] += 1
        matches = [item for item in grouped if item[0] == "0" and abs(item[1] - edge) <= TOLERANCE
                   and abs(item[2] - onset) <= TOLERANCE]
        compatible = len(matches) == 1 and matches[0][-1] == label
        counts["grouped_agrees"] += compatible
        if predicted == label:
            counts["raw_agrees_grouped_missing"] += not compatible
            if blocks is not None:
                signal = {"side": 0, "edge": edge, "onset": onset, "label": predicted}
                counts["raw_agrees_with_english_lid"] += language_support(
                    signal, blocks, activity, "en")["publisher_language_supported"]
        if label == "shift" and not compatible:
            nearby = [(left, right) for left, right in activity[1] if abs(left - onset) <= TOLERANCE]
            caller_resumes = any(left < onset + 2 * RATE and onset < right for left, right in activity[0])
            if len(nearby) == 1 and nearby[0][1] - nearby[0][0] < RATE and not caller_resumes:
                counts["short_partner_bout_without_caller_resume_2s"] += 1
    return totals


def run(root):
    import torch
    import whisper

    if not (root / ".campaign-machine").is_file():
        raise ValueError("recordings stay on Vast")
    entries, _, audit_sha = pinned_split(root, "train")
    output = root / "reports"
    membership = json.loads((output / "harper-history-replication-membership.json").read_text())
    if hashlib.sha256(json.dumps(membership).encode()).hexdigest() != MEMBERSHIP:
        raise ValueError("the archived control membership changed")
    chosen = select_calls(entries, membership)
    members = defaultdict(set)
    for call, edge, label in membership:
        members[call].add((edge, label))
    weight = root / "cache/open-speech-encoder/large-v3.pt"
    if file_digest(weight) != WHISPER_SHA256:
        raise ValueError("the cached ASR model changed")
    torch.set_num_threads(2)
    model = whisper.load_model(str(weight), device="cuda")
    detect = silero()
    totals = {arm: defaultdict(Counter) for arm in ("publisher", "whisper")}
    meter = Progress(JOB, len(chosen), "calls")
    try:
        for call in chosen:
            context = {}
            tracks, reference = observe_call(root / "raw/harper-valley", call, members[call], detect,
                                             word_context=context)
            activity = [detect(track) for track in tracks]
            words, weak, blocks = recognize_words(model, tracks)
            for arm, word_stream, omissions, lid in (("publisher", context["words"], context["omissions"], None),
                                                       ("whisper", words, weak, blocks)):
                for label, counts in attrition(reference, tracks, word_stream, omissions, activity, lid).items():
                    totals[arm][label].update(counts)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    meter.finish()
    report = {"scope": "train-control attrition only; no labels admitted",
              "publisher_timings_used_to_select_reference": True,
              "source_audit_sha256": audit_sha, "pause_membership_sha256": MEMBERSHIP,
              "selected_calls_sha256": hashlib.sha256(json.dumps(chosen).encode()).hexdigest(),
              "whisper_sha256": WHISPER_SHA256, "arms": totals}
    _write_json(output / f"{JOB}.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
