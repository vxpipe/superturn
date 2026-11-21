"""Read or convert audio to mono 16 kHz PCM."""

import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 16000


def read_audio(source):
    if isinstance(source, Path) and source.suffix.lower() == ".wav":
        try:
            with wave.open(str(source)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, RATE):
                    return np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
        except wave.Error:
            pass
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0" if isinstance(source, bytes) else str(source),
         "-f", "s16le", "-ac", "1", "-ar", str(RATE), "pipe:1"],
        input=source if isinstance(source, bytes) else None, capture_output=True, check=True,
    )
    return np.frombuffer(pcm.stdout, "<i2")


def write_wave(path, pcm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def silero():
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=True)

    def detect(source):
        pcm = read_audio(source) if isinstance(source, Path) else source
        samples = torch.from_numpy(pcm.astype(np.float32) / 32768)
        timestamps = get_speech_timestamps(samples, model, sampling_rate=RATE, speech_pad_ms=0)
        return [(span["start"], span["end"]) for span in timestamps]

    return detect
