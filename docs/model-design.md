# Model design

Status: combined input contract approved 2026-11-21. Implementation pending.

## Call contract

The application's VAD reports a pause. The application calls Superturn with
the last 3 s of the caller's audio through the decision instant, including
the pause (zero-padded at the start if shorter), the elapsed silence in
milliseconds and the current live partial transcript bytes. Superturn returns
P(turn complete). The application may call again as the pause grows, for
example at 200, 300 and 500 ms.

The first version is stateless: one call per decision, no state carried between
calls. A streaming variant that carries recurrent state across the turn can
come later if longer context helps.

## Input

- 16 kHz mono PCM, the same stream the application already feeds its VAD and
  speech-to-text, so no extra conversion is needed at call time.
- Phone callers are narrowband (G.711 at 8 kHz on the phone network) even when
  the carrier delivers 16 kHz. The model band-limits its input to 0–4 kHz, so
  phone, wideband and browser audio look alike to it.
- 40-band log-mel, 25 ms window, 10 ms hop.
- Pitch track: log-F0 and voicing probability per frame, z-normalised over the
  window. The normalisation is what makes "falling" relative to this speaker's
  own range. Energy per frame, normalised the same way.
- Silence elapsed, as a scalar, joined after the recurrent layer.
- The exported graph computes acoustic features and text byte n-gram hashes.
  The application passes raw PCM and packs the transcript bytes.

The combined contract uses 48,000 samples (3 seconds) per call.

## Network

The single artifact has a 2 MB int8 budget and a p99 target of at most 3 ms
per call for both branches on one CPU core. The audio development track
compared these architectures on the same data:

**A. Recurrent baseline**

1. Convolutional stem: two depthwise-separable 1D conv blocks, stride 2 each,
   10 ms frames to 40 ms frames. About 20k parameters.
2. One or two GRU layers, 128 units, over the ~75 frames of a 3 s window.
3. Final state plus silence elapsed, a small dense layer, sigmoid.

About 0.1–0.3M parameters.

**B. Small transformer (conformer-style)**

1. Same convolutional stem, to 40 ms frames.
2. Three or four blocks, model width 128, 4 attention heads, feed-forward 256,
   each with a depthwise conv module. About 130k parameters per block.
3. Learned positions; a summary token or attention pooling, joined with
   silence elapsed, then a dense layer and sigmoid.

About 0.4–0.6M parameters. Attention over ~75 frames is cheap. It can relate
the final syllables directly to the earlier part of the window, which is where
the speaker's pitch range and pace come from. Quantisation needs more care
(softmax and layer norm stay in float).

Model size and latency are measured after export. The campaign tests whether
data quality and coverage matter more than the choice between architectures.

## Pretraining

Fit our own teacher head on real labels with a permitted general-purpose
speech encoder. Distil it into the small audio branch using eligible causal
audio. Use existing turn detectors for benchmark comparison only.

The first version trains directly on hold/shift labels. If it falls short on
languages or conditions with little labelled data, pretrain our own tiny
encoder on large unlabelled audio with self-supervised targets that need no
human labels:

- Future voice activity: from the audio up to a pause, predict whether the same
  speaker is speaking again 200, 400, 600 and 800 ms later. Silero VAD supplies
  the targets on any single-speaker recording, so every pause in every corpus
  becomes a training example.
- Prosody continuation: predict the coming pitch and energy contour.

After pretraining, fine-tune on hold/shift labels from multi-speaker conversations, where
"someone else took the turn" is known from the per-speaker tracks.

Hand-computed pitch and energy inputs mean the model does not have to learn to
extract them itself, which is most of what a tiny model would otherwise lack.

## Training

- Loss: binary cross-entropy on hold/shift labels. For the VXPipe policy
  diagnostic, prioritize shift recall and distinguish long holds likely to
  survive until playback from shorter holds that cost a discarded draft.
- Optional auxiliary targets during training only: filler at the end, and time
  until the speaker resumes. They are dropped at export.
- Language balance: temperature sampling across languages so small languages
  are not drowned out.
- Augmentation on the fly: telephone channel, MUSAN noise, room reverb, gain,
  speed and pitch perturbation, low-pass filtered speech.
- Optional distillation from a larger teacher model on unlabelled pauses.

## Export

ONNX, int8. Dynamic quantisation first; quantisation-aware training only if
int8 loses accuracy against float.

## Transcript input

The current live STT partial arrives independently of the PCM. Take the
snapshot visible at the 300 ms decision and require no word-to-audio
alignment. Pack it by applying NFKC, Unicode lowercase, then removing
Unicode punctuation and NUL. UTF-8 encode, take the last 128 bytes and
left-pad with zeros. The graph consumes bytes as integers, including
truncated UTF-8 sequences at the leading boundary. An absent partial
becomes 128 zeros. The reference packer is
`superturn.transcript.pack_partial_transcript`. The planned interface is:

| Tensor | Type and shape |
| --- | --- |
| `audio` | Float32 `[batch, 48000]`, mono PCM in [-1, 1] |
| `silence_ms` | Float32 `[batch, 1]` |
| `text_bytes` | Int32 `[batch, 128]` |
| `p_complete` | Float32 `[batch, 1]` |

Compute rolling UTF-8 byte n-gram hashes in the graph with integer
operations and a shared embedding table. Use byte-space boundaries for
spaced writing and short character windows for unspaced writing. A small
text GRU summarizes the byte features. Fuse its state, the audio state and
elapsed silence in one probability head. Train with lag and blank-text
dropout. Use real pause points and calibrate any open LLM text labeller
on verified real labels before using its output.

## Evaluation

- Held-out sets: FLEURS, Indic DiarBench, held-out AMI meetings, and local
  call recordings.
- Conditions: wideband, telephone (8 kHz G.711), noisy.
- Metrics: false-complete rate at a fixed pause length, accuracy, and median
  pause length needed to reach a confident decision. Compare against a larger
  teacher model and against silence-only timeouts.
- Measure combined quality, audio-only with text zeroed, and text-only
  with PCM zeroed. Compare the application policy against silence timers
  by language at live STT latency and at 150/300/500 ms lag sensitivities.
  Report responsiveness, false starts and wasted drafts. A language
  qualifies when the combined model passes and blank text preserves the
  current audio-only performance.
- Measure single-core p99 after the quality checks. Profile the STFT first
  and test a smaller FFT or larger hop if needed. Treat application-side
  caching as a separate interface experiment.
