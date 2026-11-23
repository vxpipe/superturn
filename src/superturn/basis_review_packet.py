"""Prepare fit-side blind listening windows in the campaign memory filesystem."""

import hashlib
import json
import wave
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .ami_calibration import _write_json
from .audio import RATE
from .shards import seed_for

REVIEW_ROOT = Path("/dev/shm/superturn-basis-review")
WINDOW_SAMPLES = 60 * RATE


def usable_windows(connected, muted, length):
    for start in range(0, length - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        end = start + WINDOW_SAMPLES
        if any(left < end and right > start for left, right in muted):
            continue
        present = [voice for voice, intervals in connected.items()
                   if any(left < end and right > start for left, right in intervals)]
        if len(present) == 2 and all(any(left <= start and end <= right for left, right in connected[voice])
                                     for voice in present):
            yield start, end, sorted(present)


class ListeningPacket:
    def __init__(self, digest):
        self.digest = digest
        self.by_call = defaultdict(list)

    def consider(self, language, row, tracks, connected, muted, predictions):
        call, segment = row["conversation_id"], row["segment_id"]
        length = min(map(len, tracks.values()))
        candidates = self.by_call[language, call]
        for start, end, people in usable_windows(connected, muted, length):
            rank = seed_for("basis-listening-v1", self.digest, segment, start)
            if len(candidates) == 2 and rank >= candidates[-1][0]:
                continue
            ident = f"call-{rank:016x}"
            stereo = np.column_stack([tracks[person][start:end] for person in people]).astype("<i2")
            sides = dict(zip(people, ("left", "right"), strict=True))
            references = [{"stop_ms": (stop - start) / 16, "onset_ms": (onset - start) / 16,
                           "outcome": label, "focal_channel": sides[focal], "next_channel": sides[next_voice]}
                          for stop, onset, label, focal, next_voice in predictions
                          if start + 3 * RATE <= stop and onset + 2 * RATE <= end
                          and focal in sides and next_voice in sides]
            candidates.append((rank, ident, stereo, references))
            candidates.sort(key=lambda item: item[0])
            del candidates[2:]

    def write(self, root):
        if REVIEW_ROOT.exists():
            raise ValueError("the Basis listening packet already exists")
        picked = []
        for language in ("en", "hi"):
            groups = [rows for (locale, _), rows in self.by_call.items() if locale == language and rows]
            if len(groups) < 2:
                raise ValueError("the Basis pilot lacks dyadic windows from both selected calls")
            first = [rows[0] for rows in groups]
            extra = [row for rows in groups for row in rows[1:]]
            if not extra:
                raise ValueError("the Basis pilot lacks a third continuous window")
            picked.extend(first + [min(extra, key=lambda row: row[0])])
        picked.sort(key=lambda item: seed_for("basis-blind-order-v1", item[1]))
        ids = [item[1] for item in picked]
        if len(set(ids)) != 6:
            raise ValueError("Basis review window IDs repeated")
        with TemporaryDirectory(dir=REVIEW_ROOT.parent, prefix="basis-packet-") as temporary:
            packet = Path(temporary) / "packet"
            listening = packet / "hindi-review-b"
            audio = listening / "audio"
            audio.mkdir(parents=True, mode=0o700)
            (listening / "reviewer-b-order.txt").write_text("\n".join(ids) + "\n")
            receipt = {}
            for _, ident, stereo, references in picked:
                path = audio / f"{ident}.wav"
                with wave.open(str(path), "wb") as handle:
                    handle.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
                    handle.writeframes(stereo.tobytes())
                receipt[ident] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "automatic_proposals": references}
            _write_json(packet / "private-reference.json", {"membership_sha256": self.digest, "windows": receipt})
            packet.chmod(0o700)
            packet.replace(REVIEW_ROOT)
        report = {"scope": "blind fit-only Basis review", "membership_sha256": self.digest,
                  "windows": len(picked), "minutes": len(picked), "languages": ["en", "hi"],
                  "packet_sha256": hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()}
        _write_json(root / "reports/basis-listening-packet.json", report)
        print(json.dumps(report, indent=2))
