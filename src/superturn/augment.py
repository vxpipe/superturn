"""Seeded augmentations and benchmark channels at 16 kHz."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from io import BytesIO
from functools import lru_cache

import numpy as np

from .audio import RATE, read_audio

CONDITIONS = ("clean", "telephone", "telephone-noise")


def _ffmpeg(pcm, *options):
    return np.frombuffer(subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
         *options, "-f", "s16le", "-ar", str(RATE), "-ac", "1", "pipe:1"],
        input=pcm.astype("<i2").tobytes(), capture_output=True, check=True,
    ).stdout, "<i2")


def _mono_frame(av, pcm):
    frame = av.AudioFrame.from_ndarray(pcm.astype("<i2", copy=False)[None, :], format="s16", layout="mono")
    frame.sample_rate = RATE
    frame.time_base = Fraction(1, RATE)
    return frame


def _pitch_shift(pcm, factor):
    shifted_rate = round(RATE * factor)
    tempo = 1 / factor
    try:
        import av
    except ImportError:
        pass
    else:
        try:
            graph = av.filter.Graph()
            graph.threads = 1
            nodes = [
                graph.add_abuffer(sample_rate=RATE, format="s16", layout="mono", channels=1,
                                  time_base=Fraction(1, RATE)),
                graph.add("asetrate", str(shifted_rate)),
                graph.add("aresample", str(RATE)),
                graph.add("atempo", str(tempo)),
                graph.add("aformat", f"sample_fmts=s16:channel_layouts=mono:sample_rates={RATE}"),
                graph.add("abuffersink"),
            ]
            for left, right in zip(nodes, nodes[1:]):
                left.link_to(right)
            graph.configure()
            graph.push(_mono_frame(av, pcm))
            graph.push(None)
            output = []
            while True:
                try:
                    output.append(graph.pull().to_ndarray().reshape(-1))
                except av.error.EOFError:
                    break
            return np.concatenate(output) if output else np.empty(0, dtype="<i2")
        except (av.error.FFmpegError, AttributeError):
            pass
    return _ffmpeg(pcm, "-af", f"asetrate={shifted_rate},aresample={RATE},atempo={tempo}")


def _opus_roundtrip(pcm):
    try:
        import av
    except ImportError:
        pass
    else:
        try:
            dst = BytesIO()
            with av.open(dst, "w", format="ogg") as media:
                stream = media.add_stream("libopus", rate=RATE)
                stream.codec_context.layout = "mono"
                stream.codec_context.bit_rate = 12000
                for frame in (_mono_frame(av, pcm), None):
                    for packet in stream.encode(frame):
                        media.mux(packet)
            return read_audio(dst.getvalue())
        except (av.error.FFmpegError, AttributeError):
            pass
    encoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "12k", "-f", "ogg", "pipe:1"],
        input=pcm.astype("<i2").tobytes(), capture_output=True, check=True,
    ).stdout
    return read_audio(encoded)


def _bandlimit(signal, low, high):
    spectrum = np.fft.rfft(signal)
    hz = np.fft.rfftfreq(len(signal), 1 / RATE)
    spectrum[(hz < low) | (hz > high)] = 0
    return np.fft.irfft(spectrum, n=len(signal))


def _resample(signal, n):
    return np.interp(np.linspace(0, len(signal) - 1, n), np.arange(len(signal)), signal)


def _g711(pcm, alaw=False):
    import audioop

    # Band-limit before decimation and apply the ITU integer codec at 8 kHz.
    base = _limit(_bandlimit(pcm, 300, 3400))[::2].tobytes()
    compressed = audioop.lin2alaw(base, 2) if alaw else audioop.lin2ulaw(base, 2)
    decoded = audioop.alaw2lin(compressed, 2) if alaw else audioop.ulaw2lin(compressed, 2)
    samples = np.frombuffer(decoded, "<i2").astype(np.float32)
    return np.interp(np.arange(len(pcm)) / 2, np.arange(len(samples)), samples)


def _noise(rng, count, color):
    white = rng.normal(size=count)
    if color:
        spectrum = np.fft.rfft(white)
        bins = np.arange(len(spectrum)).clip(1)
        spectrum /= bins ** (color / 2)
        white = np.fft.irfft(spectrum, n=count)
    return white


def _mix(signal, noise, snr):
    signal = np.asarray(signal, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(signal)))
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    return signal if noise_rms == 0 else signal + noise * (rms / noise_rms * 10 ** (-snr / 20))


def _limit(signal):
    return np.round(np.clip(signal, -32768, 32767)).astype("<i2")


def condition(pcm, name, seed):
    if name == "clean":
        return pcm
    signal = _g711(pcm)
    if len(signal) != len(pcm):
        raise ValueError("G.711 round trip changed the clip duration")
    if name == "telephone-noise":
        rng = np.random.default_rng(seed)
        signal = _mix(signal, _bandlimit(_noise(rng, len(signal), 0), 300, 3400), 10)
    elif name != "telephone":
        raise ValueError(f"unknown condition {name!r}")
    return _limit(signal)


@lru_cache(maxsize=32)
def _asset(path):
    return read_audio(path).astype("float32")


def _tile(signal, count, rng):
    repeated = np.tile(signal, 1 + (count + len(signal)) // len(signal))
    start = rng.integers(0, len(signal))
    return repeated[start:start + count]


def augment(pcm, seed, *, noise_paths=(), ir_paths=(), voices=(), pause_samples=0):
    rng = np.random.default_rng(seed)
    signal = pcm.astype("float32")
    n = len(signal)
    speech_end = n - pause_samples
    effect = rng.choice(("none", "noise", "hum", "hiss", "telephone", "music-or-noise",
                         "babble", "room", "low-pass", "speed", "pitch", "opus"))
    if effect == "noise":
        signal = _mix(signal, _noise(rng, n, rng.integers(0, 3)), rng.uniform(0, 30))
    elif effect == "hum":
        hz = rng.choice((50, 60))
        hum = np.sin(np.arange(n) / RATE * (2 * np.pi * hz))
        signal = _mix(signal, hum, rng.uniform(10, 30))
    elif effect == "hiss":
        signal = _mix(signal, _bandlimit(_noise(rng, n, 0), 2000, 7900), rng.uniform(10, 30))
    elif effect == "telephone":
        signal = _g711(pcm, alaw=bool(rng.integers(2)))
        loss = rng.uniform(0, 0.05)
        for start in range(0, n, 320):
            if rng.random() < loss:
                signal[start:start + 320] = 0
            elif rng.random() < 0.025:
                signal[start:start + 320] = np.roll(signal[start:start + 320], int(rng.integers(-32, 33)))
    elif effect == "music-or-noise" and noise_paths:
        signal = _mix(signal, _tile(_asset(noise_paths[int(rng.integers(len(noise_paths)))]), n, rng), rng.uniform(0, 30))
    elif effect == "babble" and len(voices) >= 3:
        indices = rng.choice(len(voices), min(len(voices), rng.integers(3, 9)), replace=False)
        babble = sum(_tile(voices[i].astype(np.float32), n, rng) for i in indices)
        signal = _mix(signal, babble, rng.uniform(0, 30))
    elif effect == "room" and ir_paths:
        impulse = _asset(ir_paths[int(rng.integers(len(ir_paths)))])[:RATE]
        peak = np.max(np.abs(impulse))
        if peak:
            impulse = impulse[np.argmax(np.abs(impulse) >= peak * 0.1):]
            size = n + len(impulse) - 1
            signal = np.fft.irfft(np.fft.rfft(signal, size) * np.fft.rfft(impulse, size), size)[:n]
            signal /= np.linalg.norm(impulse)
    elif effect == "low-pass":
        signal = _bandlimit(signal, 0, 400)
    elif effect in {"speed", "pitch"} and speech_end > 0:
        speech = pcm[:speech_end]
        if effect == "speed":
            transformed = _resample(speech, round(speech_end / rng.uniform(0.9, 1.1)))
        else:
            factor = 2 ** (rng.uniform(-2, 2) / 12)
            transformed = _pitch_shift(speech, factor)
        transformed = np.pad(transformed, (max(0, speech_end - len(transformed)), 0))
        signal = np.concatenate((transformed[-speech_end:], signal[speech_end:]))
    elif effect == "opus":
        signal = _opus_roundtrip(pcm)
    if len(signal) < n:
        signal = np.pad(signal, (n - len(signal), 0))
    return _limit(signal[-n:] * rng.uniform(0.65, 1.35))
