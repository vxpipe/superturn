"""Measure how long annotated holds remain open after the 300 ms decision."""

import json

from .ami_calibration import _write_json
from .audio import RATE
from .eot_domain_probe import DATASET_SHA256, DECISION_MS
from .sources import data_root
from .turn1_mini_probe import english_spans

TIME_TO_AUDIO_MS = (300, 500, 700, 1000)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the English development diagnostic stays on the campaign machine")
    holds = []
    shifts = 0
    for pcm, edge, stop, truth, _ in english_spans(root):
        if truth:
            shifts += 1
        else:
            holds.append((stop - edge, stop == len(pcm)))
    if (len(holds), shifts) != (532, 400):
        raise ValueError("the pinned 300 ms diagnostic membership changed")
    rows = []
    for arrival in TIME_TO_AUDIO_MS:
        time = (DECISION_MS + arrival) * RATE // 1000
        observed = sum(duration > time for duration, _ in holds)
        unresolved = sum(censored and duration <= time for duration, censored in holds)
        rows.append({"time_from_decision_ms": arrival, "time_from_speech_stop_ms": DECISION_MS + arrival,
                     "observed_open_holds": observed, "censored_before_playback": unresolved,
                     "open_hold_fraction_lower": observed / len(holds),
                     "open_hold_fraction_upper": (observed + unresolved) / len(holds)})
    result = {"dataset_sha256": DATASET_SHA256, "decision_ms": DECISION_MS, "holds": len(holds),
              "shifts": shifts, "holds_ending_at_clip_end": sum(censored for _, censored in holds),
              "assumptions": "All holds trigger at 300 ms. Immediate cancellation at annotated resumption. "
              "Point-in-time hold duration before clip end is observed; a hold ending at clip end is right-censored. "
              "No model scores, VAD delay, playback or system cutoff measurement.",
              "english_development": rows}
    _write_json(root / "reports/english-hold-survival.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
