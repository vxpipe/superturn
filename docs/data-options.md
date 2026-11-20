# Training data options

Status: research, 2026-11-20. Nothing downloaded yet. Sources and download
selections are codified in `configs/sources.toml`.

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
| Indic DiarBench | CC BY 4.0 | 22 Indian languages | Test set only: ~108 h of natural multi-speaker conversation (53 h near-field from one microphone per speaker, mixed to mono), human-corrected speaker-attributed timestamps. |
| AMI, ICSI | CC BY 4.0 | English | Training conversations with one headset per speaker (see below). |
| MUSAN | CC BY 4.0 | n/a | Noise, music and babble for augmentation. |
| OpenSLR 26 (simulated RIRs) | Apache 2.0 | n/a | Room reverberation. |

### Per-speaker conversation sources besides Basis

English meetings recorded with a close-talk microphone per speaker. They give
real hold/shift labels from per-track VAD and need no access approval, so the
per-track labelling pipeline can be built and tested on them before Basis
access is granted. Meetings have more overlap and more than two speakers, so
they are weaker than one-to-one calls.

| Source | Licence | Languages | Notes |
|---|---|---|---|
| AMI Meeting Corpus | CC BY 4.0 per the current Edinburgh download page; older releases were under a modified NC-SA licence, so use only the current release | English | ~100 h, individual headset microphones. |
| ICSI Meeting Corpus | CC BY 4.0 per its Edinburgh licence page | English | ~70 h, close-talk microphones. |
| NOTSOFAR-1 | CC BY 4.0 (reported by a derivative card; confirm on the official page) | English | Close-talk per speaker in train and dev splits only. |
| Full-duplex dual-track set (arXiv 2509.04093) | CC BY 4.0 (verify) | Chinese, English | ~15 h, one track per speaker. Small. |

No large, permissively licensed, multilingual per-speaker conversation source
other than Basis was found on 2026-11-20. Without Basis, non-English shift
labels come only from the weak segment ends of spontaneous speech and from
read-speech sentence ends.

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

## Download plan and sizes

Measured on 2026-11-20 by `bin/plan-downloads` from provider file listings
(Hugging Face trees, HEAD requests, Common Voice release metadata). Sizes are
the selections in `configs/sources.toml`, not whole datasets.

| Source | Selection | Files | Size | Whole dataset |
|---|---|---|---|---|
| Basis Conversations 1500 | everything | 13,764 | 254.6 GB | same |
| Common Voice 27.0 | 25 locales | 25 | 168.3 GB | 947 GB, 295 locales |
| VoxPopuli | 2 training shards per language | 29 | 75.8 GB | ~339 GB per-language data |
| FLEURS | test split, all languages | 204 | 50.7 GB | 878 GB including a parquet copy |
| AMI | individual headsets, 171 meetings | 687 | 46.6 GB | adds mixes, lapels, arrays, video |
| ICSI | individual headsets | 922 | 34.1 GB | adds mixes |
| Emilia-YODAS | 5 shards (~1 GB each) per language | 30 | 31.6 GB | 2.1 TB |
| Indic DiarBench | everything | 23 | 12.1 GB | same |
| MUSAN | everything | 1 | 11.1 GB | same |
| OpenSLR 26 | 16 kHz simulated RIRs | 1 | 0.2 GB | same |
| **Total** | | | **685 GB** | |

Basis is uneven across languages: Spanish 82 GB, Arabic 48 GB, Hindi 33 GB,
English 28 GB and Portuguese 21 GB of audio; most other languages have
1–7 GB, and French, German, Polish, Dutch, Mingrelian and Ukrainian less than
1.2 GB each. Its 22 languages are not 22 equal training sets.

Common Voice: Spanish (52 GB), German (37 GB) and French (31 GB) are left out
because Basis, VoxPopuli and Emilia-YODAS cover them; Kannada and Gujarati
have no release. Each locale is one archive, so a locale cannot be partly
downloaded, and Mozilla Data Collective allows 30 dataset downloads per day
after its terms are accepted in the web UI per dataset.

### How downloads are made cheaper

- Only the parts used are listed: test splits only for test sets, a few shards
  of the very large sets, individual headset tracks only for AMI and ICSI.
- Each plan is priced exactly before anything downloads.
- Hugging Face downloads pin the dataset commit and verify every file size;
  HTTP downloads resume and verify sizes; everything is idempotent.
- Small, essential sources download first (`order` in the config), so
  labelling can start while the large ones are still arriving.
- Each source is processed into shards and its raw files deleted once the
  shards are synced to persistent storage, so the machine needs ~1 TB of disk
  rather than raw plus processed copies. Later campaigns pull shards, never raw
  data.

## Speaker variety

The model must work for voices it has never heard, so every source needs many
speakers, and no speaker may dominate.

| Source | Speakers (as stated) |
|---|---|
| Basis | 2,645 |
| Common Voice, 25 chosen locales | 81,574 contributors in total; the smallest locale (pa-IN) has 75 |
| Indic DiarBench | 485 in meetings plus ~750 in-the-wild |
| AMI, ICSI, VoxPopuli, FLEURS, Emilia-YODAS | not stated on the cards; counted from the data during labelling |

Rules, enforced in code during labelling and splitting:

- Speaker counts are computed from the data for every source and language and
  reported; a source or language below 50 speakers is flagged.
- Examples are capped per speaker, so one voice cannot exceed a fixed share of
  any source's training examples.
- Splits are speaker-disjoint (see the milestone), and benchmark results come
  only from voices never seen in training.

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
