"""Measure acoustic-to-annotation timing differences on AMI training only."""

from collections import Counter
import json

import numpy as np

from .ami_reference import audit_sample
from .labels import read_rows
from .progress import Progress
from .sources import data_root

TOLERANCES = ((100, 125), (200, 250), (300, 375), (400, 500), (500, 625))


def evaluate(root):
    rows = [row for row in read_rows(root / "labels" / "ami.jsonl") if row.split == "train"]
    errors = []
    sweep = []
    meter = Progress("ami-timing-calibration", len(TOLERANCES), "settings")
    try:
        for index, (end_ms, onset_ms) in enumerate(TOLERANCES):
            votes = audit_sample(root, rows, end_ms=end_ms, onset_ms=onset_ms,
                                 errors=errors if index == 0 else None)
            counts = Counter(votes.values())
            resolved = counts["supported"] + counts["contradicted"]
            sweep.append({"caller_end_ms": end_ms, "next_onset_ms": onset_ms, **counts,
                          "support_among_resolved": counts["supported"] / resolved if resolved else None})
            meter.advance(1)
        if not errors:
            raise ValueError("no AMI acoustic and annotation timings could be paired")
        pairs = np.asarray(errors)
        stats = []
        for index, boundary in enumerate(("caller_end", "next_onset")):
            values = pairs[:, index]
            stats.append({"boundary": boundary, "median_signed_ms": float(np.median(values)),
                          "median_absolute_ms": float(np.median(np.abs(values))),
                          "p90_absolute_ms": float(np.percentile(np.abs(values), 90))})
        report = {"split": "train", "scanned": len(rows), "pairing_limit_ms": 600,
                  "paired": len(errors), "offsets": stats, "sweep": sweep}
        path = root / "reports" / "ami-timing.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(data_root()), indent=2))
