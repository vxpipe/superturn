"""Measure channel separation and preliminary pause yield on two Map Task slices."""

from collections import Counter
import hashlib
import json
from math import gcd
from pathlib import Path
import struct
import wave

import numpy as np
from scipy.signal import resample_poly

from .acquire import _request, parse_wav_header, wav_bytes
from .ami_calibration import _write_json
from .ami_pilot import _peak, acoustic_status
from .audio import RATE, silero
from .labels import Speech, turns
from .progress import Progress
from .sources import data_root, load_sources

BASE = "https://groups.inf.ed.ac.uk/maptask/signals/dialogues/"
FILES = ("q1ec1.mix.wav", "q5nc1.mix.wav")
SECONDS = 90
MAX_TOTAL_BYTES = 20_000_000
SOURCE = "maptask-audio"


def read_range(url, start, count):
    end = start + count - 1
    with _request(url, headers={"Range": f"bytes={start}-{end}"}) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError("Map Task server did not honor the exact bounded range")
        payload = response.read(count + 1)
        if len(payload) != count:
            raise ValueError("wrong Map Task byte-range length")
    return payload


def stage(folder, name):
    path = folder / name
    if not path.is_file():
        url = BASE + name
        fmt, offset = parse_wav_header(read_range(url, 0, 4096))
        encoding, channels, rate, _, alignment, bits = struct.unpack("<HHIIHH", fmt[:16])
        if (encoding, channels, alignment, bits) != (1, 2, 4, 16) or not 8000 <= rate <= 24000:
            raise ValueError("unexpected Map Task WAV format")
        count = SECONDS * rate * alignment
        if len(FILES) * (count + 4096) > MAX_TOTAL_BYTES:
            raise ValueError("Map Task byte-range probe exceeds its designated cap")
        pcm = read_range(url, offset, count)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(wav_bytes(fmt, pcm))
        temporary.replace(path)
    with wave.open(str(path), "rb") as audio:
        channels, rate, size = audio.getnchannels(), audio.getframerate(), audio.getsampwidth()
        if (channels, size, audio.getnframes()) != (2, 2, rate * SECONDS) or not 8000 <= rate <= 24000:
            raise ValueError("cached Map Task slice differs from its bounded format")
        frames = np.frombuffer(audio.readframes(audio.getnframes()), "<i2").reshape(-1, 2).T
    divisor = gcd(rate, RATE)
    signals = np.rint(resample_poly(frames.astype(np.float32), RATE // divisor, rate // divisor, axis=-1))
    return signals.clip(-32768, 32767).astype(np.int16), rate, path


def analyze(signal, detect, *, candidates=None):
    intervals = [list(detect(channel)) for channel in signal]
    kept = []
    ratios = [[], []]
    for channel, spans in enumerate(intervals):
        for start, end in spans:
            own = _peak(signal[channel], start, end)
            other = _peak(signal[1 - channel], start, end)
            if own >= 1.414 * max(1.0, other):
                kept.append(Speech(str(channel), start, end))
            if own and not any(begin < end and start < stop for begin, stop in intervals[1 - channel]):
                ratios[channel].append(20 * np.log10(max(1.0, other) / own))
    raw = [Speech(str(i), start, end) for i, spans in enumerate(intervals) for start, end in spans]
    spans = {str(i): {"vad_segments": len(rows), "speech_seconds": sum(b - a for a, b in rows) / RATE,
                      "opposite_to_own_db_p50": float(np.median(ratios[i])) if ratios[i] else None}
             for i, rows in enumerate(intervals)}
    def counts(segments, *, screen=False):
        by_channel = Counter()
        track_audio = {str(i): channel for i, channel in enumerate(signal)}
        for span, gap, label in turns(segments, decision_ms=300):
            if screen:
                following = [row for row in segments if row.start == span.end + gap]
                if len(following) != 1 or acoustic_status(span, following[0], track_audio) != "passed":
                    continue
                if candidates is not None:
                    candidates.append((span, following[0], label))
            by_channel[span.speaker, label] += 1
        return {str(i): {label: by_channel[str(i), label] for label in ("hold", "shift")} for i in range(2)}
    return {"channels": spans, "dominant_segments": len(kept), "all_segments": len(raw),
            "preliminary_raw_pauses": counts(raw), "preliminary_dominant_pauses": counts(kept, screen=True)}


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("Map Task media stays on Vast")
    source = load_sources()[SOURCE]
    raw = root / "raw" / SOURCE
    folder = raw / "probe"
    folder.mkdir(parents=True, exist_ok=True)
    detect, records = silero(), {}
    meter = Progress("maptask-audio-probe", len(FILES), "dialogues")
    try:
        for name in FILES:
            signal, rate, path = stage(folder, name)
            records[name] = {"sample_rate": rate, "seconds": SECONDS, "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **analyze(signal, detect)}
            meter.advance(1)
        report = {"method": "preliminary single-timing-source acoustic screen", "deadline_ms": 300,
                  "roles": "left and right; giver/follower mapping unverified", "files": records}
        _write_json(raw / "MANIFEST.json", {
            "source": SOURCE, "licence": source.licence, "attribution": source.attribution, "role": source.role,
            "files": [{"path": f"probe/{name}", "url": BASE + name, "bytes": info["bytes"],
                       "sha256": info["sha256"], "start_s": 0, "length_s": info["seconds"]}
                      for name, info in records.items()],
        })
        _write_json(root / "reports" / "maptask-audio-probe.json", report)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
    return report


if __name__ == "__main__":
    print(json.dumps(run(data_root()), indent=2))
