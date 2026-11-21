"""Prepare a blind caller-only human baseline from pinned Harper validation audio."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np

from .ami_calibration import _write_json
from .audio import write_wave
from .dense_teacher import SAMPLES, file_digest
from .harper_partial_stt import pinned_membership
from .progress import Progress
from .shards import seed_for
from .sources import data_root
from .turn1_mini_probe import harper_rows
from .wav2vec_harper_probe import pause_rows

GUIDE = """# HarperValleyBank caller-only floor-completion judgments at 300 ms

Each clip is the caller channel for a single three-second model input.
The candidate pause starts 300 milliseconds before the end of the clip.
Rate how likely it is that the caller finished their turn at that pause.
Use a score from 0 to 100, where 0 means certain they will continue,
50 means equally likely, and 100 means certain they finished. If the
clip is inaudible or the task cannot be rated, record an abstention and
leave the probability field blank.

Work independently and follow the assigned CSV order. Keep playback
level fixed and play each clip once without displaying its waveform.
Use only the supplied clip and its silence. Record judgments before
consulting any other listener or viewing additional context.

Audio adapted from HarperValleyBank by Wu, Nafziger, Scodary and Maas
(CC BY 4.0). These are shortened caller-channel excerpts rendered to
16 kHz PCM after G.711 μ-law.
[Publisher source](https://github.com/cricketclub/gridspace-stanford-harper-valley)
"""


def prepare(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the Harper packet stays on Vast")
    members, membership_sha = pinned_membership(root, "validation")
    decoded = list(pause_rows(harper_rows(root, native_prefix=True)))
    if len(decoded) != len(members) or any(
            len(audio) != SAMPLES or truth != int(row[3] == "shift") or speaker != row[4]
            for (audio, truth, speaker), row in zip(decoded, members, strict=True)):
        raise ValueError("the blind audio differs from the pinned validation rows")
    directory = root / "runs/harper-human-baseline"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / membership_sha[:12]
    if destination.exists():
        raise ValueError("the existing packet may hold judgments and is preserved")
    permutation = np.random.default_rng(seed_for("harper-human-packet-v1", membership_sha)).permutation(len(members))
    bindings, checksums = {}, {}
    with TemporaryDirectory(dir=directory) as temporary:
        output = Path(temporary) / "packet"
        public = output / "listening"
        public.mkdir(parents=True)
        (public / "GUIDE.md").write_text(GUIDE)
        meter = Progress("harper-human-baseline", len(members), "caller clips")
        try:
            for index, offset in enumerate(permutation, 1):
                audio, truth, _ = decoded[offset]
                identifier = f"sample-{index:03d}"
                path = public / "audio" / f"{identifier}.wav"
                pcm = np.rint(audio * 32768).clip(-32768, 32767).astype("<i2")
                write_wave(path, pcm)
                checksums[identifier] = file_digest(path)
                _, call, boundary, _, speaker = members[offset]
                bindings[identifier] = {"label": truth, "call": call, "edge_16khz": boundary, "speaker": speaker}
                meter.advance(1)
        except BaseException:
            meter.finish("failed")
            raise
        meter.finish()
        for reviewer in ("a", "b"):
            rng = np.random.default_rng(seed_for("harper-human-listener-v1", reviewer, membership_sha))
            with (public / f"reviewer-{reviewer}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["example", "probability_percent", "abstain"])
                writer.writerows((identifier, "", "") for identifier in rng.permutation(list(bindings)))
        private = output / "key.json"
        _write_json(private, bindings)
        manifest = output / "audio-checksums.json"
        _write_json(manifest, checksums)
        output.rename(destination)
    report = {"scope": "blind listening preparation, pending authorized raters",
              "membership_sha256": membership_sha, "clips": len(bindings), "decision_ms": 300,
              "label_counts": {"holds": len(bindings) - sum(row["label"] for row in bindings.values()),
                               "shifts": sum(row["label"] for row in bindings.values())},
              "callers": len({row["speaker"] for row in bindings.values()}),
              "packet": (destination / "listening").relative_to(root).as_posix(),
              "key_sha256": file_digest(destination / "key.json"),
              "audio_manifest_sha256": file_digest(destination / "audio-checksums.json"),
              "render_order_sha256": sha256(permutation.astype("<i8").tobytes()).hexdigest()}
    _write_json(root / "reports/harper-human-baseline-packet.json", report)


def seal(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the Harper packet stays on Vast")
    report_path = root / "reports/harper-human-baseline-packet.json"
    report = json.loads(report_path.read_text())
    path = root / report["packet"]
    listener = path if path.name == "listening" else path / "listening"
    manifest = listener.parent / "audio-checksums.json"
    if (file_digest(manifest) != report["audio_manifest_sha256"] or
            file_digest(listener.parent / "key.json") != report["key_sha256"]):
        raise ValueError("the blind packet manifests changed")
    checksums = json.loads(manifest.read_text())
    if len(checksums) != report["clips"] or len(list((listener / "audio").glob("*.wav"))) != len(checksums):
        raise ValueError("the packet has missing or extra audio")
    for name, expected in checksums.items():
        audio = listener / "audio" / f"{name}.wav"
        if file_digest(audio) != expected:
            raise ValueError("a blind clip changed")
        with wave.open(str(audio), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes()) != (16000, 1, 2, SAMPLES):
                raise ValueError("incorrect blind clip input geometry")
    for rater in ("a", "b"):
        with (listener / f"reviewer-{rater}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if (len(rows) != len(checksums) or {row["example"] for row in rows} != set(checksums)
                or any(row["probability_percent"] or row["abstain"] for row in rows)):
            raise ValueError("the packet is changed or already contains ratings")
    (listener / "GUIDE.md").write_text(GUIDE)
    report["packet"] = listener.relative_to(root).as_posix()
    report["guide_sha256"] = file_digest(listener / "GUIDE.md")
    report["verified_clips"] = len(checksums)
    _write_json(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true", help="verify existing listening packet before use")
    args = parser.parse_args()
    if not args.seal:
        prepare(data_root())
    seal(data_root())
