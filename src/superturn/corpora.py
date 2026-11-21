"""Label downloaded corpus selections according to their source contracts."""

import csv
import hashlib
import json
from collections import defaultdict

from .audio import RATE, read_audio, write_wave
from .labels import Label, Speech, turns
from .progress import Progress
from .splits import assign

MIN_GAP = 3200
READ_TAIL_GUARD = 320


INDIC_CODES = {
    "assamese": "as", "bengali": "bn", "bodo": "brx", "dogri": "doi",
    "gujarati": "gu", "hindi": "hi", "kannada": "kn", "kashmiri": "ks",
    "konkani": "kok", "maithili": "mai", "malayalam": "ml", "manipuri": "mni",
    "marathi": "mr", "nepali": "ne", "odia": "or", "oriya": "or",
    "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd",
    "tamil": "ta", "telugu": "te", "urdu": "ur",
}


def language_code(value):
    locale = value.strip().lower()
    if locale in INDIC_CODES:
        return INDIC_CODES[locale]
    base = locale.replace("_", "-").split("-", 1)[0]
    return "zh" if base == "cmn" else base


def indicative_speech(row):
    return [
        Speech(str(span["speaker_id"]), round(float(span["start_time"]) * RATE),
               round(float(span["end_time"]) * RATE))
        for span in row["annotated_transcript"]
        if float(span["end_time"]) > float(span["start_time"])
    ]


def spontaneous_pauses(speech, samples):
    for current, following in zip(speech, speech[1:]):
        gap = following[0] - current[1]
        if gap >= MIN_GAP:
            yield current[1], gap, "hold", None
    if speech and samples - speech[-1][1] >= MIN_GAP:
        yield speech[-1][1], samples - speech[-1][1], "weak-shift", None


def read_pauses(speech, samples, words):
    if not speech:
        return
    end = speech[-1][1]
    if samples - end >= MIN_GAP:
        yield end, samples - end, "shift", None
    tail = end + READ_TAIL_GUARD
    if samples - tail < MIN_GAP:
        return
    # Select high-confidence word ends inside the sentence and reuse only
    # the original recording's VAD-verified trailing silence.
    for start, boundary, confidence in words[1:-2]:
        if (confidence >= 0.4 and speech[0][0] + 4000 <= start < boundary <= end - 4000
                and any(left <= boundary <= right for left, right in speech)):
            yield boundary, samples - tail, "hold", tail


def whisper_aligner(device):
    import torch
    import whisper
    from whisper.timing import find_alignment
    from whisper.tokenizer import LANGUAGES, get_tokenizer

    model = whisper.load_model("tiny", device=device)

    def align(pcm, text, language):
        code = language_code(language)
        if code not in LANGUAGES or not text.strip() or not RATE // 2 <= len(pcm) <= 30 * RATE:
            return []
        tokenizer = get_tokenizer(True, language=code, task="transcribe")
        tokens = tokenizer.encode(" " + text.strip())
        if len(tokens) + len(tokenizer.sot_sequence) + 2 > model.dims.n_text_ctx:
            return []
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(pcm.astype("float32") / 32768),
                                          n_mels=model.dims.n_mels).to(device)
        with torch.no_grad():
            alignment = find_alignment(model, tokenizer, tokens, mel, len(pcm) // 160)
        return [(round(word.start * RATE), round(word.end * RATE), word.probability)
                for word in alignment if any(char.isalnum() for char in word.word)]

    return align


def _parquet(raw):
    import pyarrow.parquet as pq

    for path in sorted(raw.rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                yield path, row


def _table_clips(raw, source):
    for path, row in _parquet(raw):
        if source == "indic_diarbench":
            yield row["sample_id"], row["recording_id"], None, row["language"], "", row["audio"]["bytes"], row
        elif row["speaker_id"]:
            yield (row["audio_id"], row["audio_id"], str(row["speaker_id"]), path.parent.name,
                   "", row["audio"]["bytes"], None)


def _read_clips(raw, source):
    if source == "common_voice":
        metadata = sorted(raw.glob("transcript/*/train.tsv"))
        for table in metadata:
            language = table.parent.name
            clips = {p.name: p for p in (raw / "audio" / language).rglob("*.mp3")}
            with table.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                    name = row["path"].rsplit("/", 1)[-1]
                    clip = clips.get(name if name.endswith(".mp3") else name + ".mp3")
                    if clip and row["client_id"]:
                        yield clip.name, clip.name, row["client_id"], language, row["sentence"], clip, None
    else:
        for table in sorted(raw.glob("data/*/test.tsv")):
            language = table.parent.name
            clips = {p.name: p for p in table.parent.rglob("*.wav")}
            with table.open(encoding="utf-8", newline="") as handle:
                for columns in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if len(columns) >= 4 and (clip := clips.get(columns[1])):
                        # FLEURS TSV has an utterance ID but no global speaker ID.
                        yield clip.name, clip.name, None, language, columns[3], clip, None


def _emilia_clips(raw):
    for metadata in sorted(raw.rglob("*.json")):
        if metadata.name == "MANIFEST.json" or ".done" in metadata.parts:
            continue
        audio = metadata.with_suffix(".mp3")
        if audio.is_file():
            row = json.loads(metadata.read_text())
            speaker = row.get("speaker")
            if speaker:
                key = str(row.get("_id") or row.get("id") or metadata.stem)
                yield key, key, str(speaker), row["language"], "", audio, None


def _clip_count(raw, source):
    if source in {"indic_diarbench", "voxpopuli"}:
        import pyarrow.parquet as pq

        return sum(pq.ParquetFile(path).metadata.num_rows for path in raw.rglob("*.parquet"))
    return sum(1 for _ in (_emilia_clips(raw) if source == "emilia_yodas" else _read_clips(raw, source)))


def corpus_labels(root, source, vad, align=None):
    raw = root / "raw" / source
    role = "test" if source in {"indic_diarbench", "fleurs"} else "train"
    read = source in {"common_voice", "fleurs"}
    if source in {"indic_diarbench", "voxpopuli"}:
        clips = _table_clips(raw, source)
    elif read:
        clips = _read_clips(raw, source)
    else:
        clips = _emilia_clips(raw)
    count = _clip_count(raw, source)
    if not count:
        raise ValueError(f"{source}: no usable downloaded clips")
    meter = Progress(f"label-{source}", count, "clips")
    try:
        for sample, recording, speaker, language, text, audio, row in clips:
            language = language_code(language)
            pcm = read_audio(audio)
            if row:
                spans = indicative_speech(row)
                groups = defaultdict(list)
                for span, gap, label in turns(spans):
                    if span.end + gap <= len(pcm):
                        groups[f"{recording}:{span.speaker}"].append((span.end, gap, label, None))
                method = "human-timestamps"
            else:
                speech = vad(pcm)
                if read:
                    examples = read_pauses(speech, len(pcm), align(pcm, text, language) if speech else [])
                    method = "cut"
                else:
                    examples = spontaneous_pauses(speech, len(pcm))
                    method = "single-track"
                groups = {speaker or "unknown": list(examples)}

            if any(groups.values()):
                if isinstance(audio, bytes):
                    digest = hashlib.sha256(sample.encode()).hexdigest()
                    audio = root / "labels" / "audio" / source / f"{digest}.wav"
                    write_wave(audio, pcm)
                for voice, examples in groups.items():
                    group = recording if role == "test" else voice
                    split = assign(source, group, role)
                    for edge, gap, label, tail in examples:
                        yield Label(source, language, voice, recording, group, split, edge, gap,
                                    label, method, str(audio.relative_to(root)), silence_start=tail)
            meter.advance(1)
    except BaseException:
        meter.finish("failed")
        raise
    else:
        meter.finish()
