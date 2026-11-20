# Training data options

Status: research, 2026-11-20. Nothing downloaded yet.

## Licence policy

The model, weights and code are released free for commercial and
non-commercial use. Training data must therefore allow commercial training and
publishing the trained weights, with no share-alike on the model. Attribution
requirements are fine; list every source in the model card. Non-commercial,
paid or research-only data is excluded.

## What a label is

Every example is a pause in one speaker's audio, labelled by what happened
next:

- **complete (shift)**: the speaker stopped and another speaker took the turn.
- **incomplete (hold)**: the speaker paused, then continued the same turn.

The model sees the audio up to the pause and must predict which one.

## Selected sources

| Source | Licence (as read 2026-11-20) | Languages | Role |
|---|---|---|---|
| Basis Conversations 1500 | Basis Data License 1.0 (custom) | 22 | Main training data: ~1,500 h of real conversation, one 48 kHz FLAC per speaker on a shared clock, 2,645 speakers. Labels from Silero VAD on each track. |
| Common Voice | CC0 (sentence text confirmed CC0; confirm the audio terms where it is downloaded) | 100+ | Language and speaker breadth: real voices on everyday microphones, cut at word boundaries for holds. |
| VoxPopuli (data only) | CC0 | 16+ European | Spontaneous mid-sentence pauses and sentence endings. |
| Emilia-YODAS (YODAS subset only) | CC BY 4.0; upstream YODAS2 is CC BY 3.0 | 6 | Spontaneous in-the-wild speech. |
| FLEURS | CC BY 4.0 | ~100 | Held-out test set only: the same sentences in every language. |
| MUSAN | CC BY 4.0 | n/a | Noise, music and babble for augmentation. |
| OpenSLR 26 (simulated RIRs) | Apache 2.0 | n/a | Room reverberation. |

### Licence notes

- **Basis Data License 1.0** allows training for research and commercial
  purposes and publishing, distributing and selling the resulting models. It
  states that a model is not a copy of the dataset and imposes no share-alike.
  Conditions: attribution to Basis in the model card; no redistribution of the
  audio beyond short excerpts (30 s per clip, 5 min per publication); no
  attempt to identify or contact participants; no voice clones of
  participants. Models made before any termination may still be used. The
  model can be open, but the training audio cannot be republished; others can
  request access from Basis to reproduce training.
- Basis access is approved per person by Basis and is non-transferable; each
  person downloading needs their own access. Cloud compute and storage are
  allowed when the provider is bound by written confidentiality terms and does
  not use the data for itself. Required attribution: "Developed using Basis
  Conversations 1500, published by Basis", with the repository URL and release
  identifier, in the model card. Basis may terminate only for an uncured
  material breach (30 days to cure); models made before termination remain
  usable. No field-of-use restrictions; no indemnity; California law.
- **VoxPopuli**: only the data is CC0. Its released pretrained models and
  code are CC BY-NC 4.0 and must not be used.
- **Emilia**: download only the YODAS subset. The original Emilia subset is
  CC BY-NC 4.0. The gated access terms add an indemnity clause and allow access
  to be revoked.
- **OpenSLR 28** is labelled Apache 2.0 but bundles third-party RIR sets and
  MUSAN-derived noise under other terms. Not used unless each component is
  checked.

## Excluded

| Source | Reason |
|---|---|
| Seamless Interaction | CC BY-NC 4.0 |
| Original Emilia subset | CC BY-NC 4.0 |
| Global-Conversational-Speech | Paid commercial licence |
| Switchboard, Fisher, CALLHOME (LDC) | Paid, restricted licence |

## Channel augmentation

Telephone channel: resample to 8 kHz, G.711 μ-law/A-law round trip, band-pass
300–3400 Hz, packet loss and jitter, level changes. Production callers arrive
over the phone, so this matters more than room noise.

## Our own calls

Production call recordings are the closest match to real use, but they are
customer data: local evaluation only, never used for training, committed or
published.

## Labelling tools

- **Silero VAD**: pause boundaries and lengths on each per-speaker track.
- **pyannote**: diarization for mixed, single-track audio such as podcasts or call
  recordings. Its boundary errors fall exactly where labels matter, so treat
  diarized labels as lower quality than per-track labels.
- **Teacher model**: a larger turn model can score unlabelled pauses, so the
  small model can be distilled from it on large unlabelled audio.
- **Forced alignment**: word boundaries in real single-speaker recordings, so
  utterances can be cut at a word boundary to make incomplete examples from real
  voices. Cut only at word boundaries and pad with real silence from the same
  recording; cutting mid-word would teach the model to spot edits instead of
  unfinished speech.
- **Speaker augmentation**: pitch shift, formant shift, speed perturbation and
  vocal tract length perturbation on real speech to widen speaker variety.

## Rejected: preset-voice TTS

A TTS model with a fixed set of voices gives only that many speakers. The model
would learn those voices rather than turn-taking in general. Synthetic incomplete
examples come from truncating real recordings instead, which already covers
tens of thousands of speakers across the multilingual read-speech sets.

## Rejected: reversed audio

Reversing audio does not make speech language-neutral for this task. It turns
falling phrase endings into rising ones and moves final-syllable lengthening to
the start, so the model would learn the opposite of the real cues. Language
independence should come from many languages in the training data. Low-pass
filtered speech (around 400 Hz) keeps the pitch contour and removes most words.
It is worth trying as an augmentation, and as a test that the model relies on
prosody rather than words.
