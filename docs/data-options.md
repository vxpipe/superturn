# Training data options

Status: 2026-11-20. Nothing downloaded yet. Sources and download selections
are codified in `configs/sources.toml`.

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
| Common Voice 17.0 | CC0 | 25 chosen languages | Language and speaker breadth: real voices on everyday microphones, cut at word boundaries for holds. Downloaded from the `fsicoli/common_voice_17_0` copy on Hugging Face, whose per-clip metadata (speaker, sentence) is in separate TSV files. |
| VoxPopuli (data only) | CC0 | 16+ European | Spontaneous mid-sentence pauses and sentence endings. |
| Emilia-YODAS (YODAS subset only) | CC BY 4.0; upstream YODAS2 is CC BY 3.0 | 6 | Spontaneous in-the-wild speech. |
| FLEURS | CC BY 4.0 | ~100 | Held-out test set only: the same sentences in every language. |
| Indic DiarBench | CC BY 4.0 | 22 Indian languages | Test set only: ~108 h of natural multi-speaker conversation (53 h near-field from one microphone per speaker, mixed to mono), human-corrected speaker-attributed timestamps. |
| AMI | CC BY 4.0 | English | The only training conversations: one headset per speaker, real hold/shift labels (see below). |
| MUSAN Free Sound | CC BY 4.0 corpus; selected recordings in the US public domain | n/a | Environmental noise for augmentation. |
| OpenSLR 26 (simulated RIRs) | Apache 2.0 | n/a | Room reverberation. |

### Per-speaker conversation sources

Basis Conversations 1500 was the intended main source (22 languages, one track
per speaker) but access was not granted (2026-11-20), so the model is trained
without it. The per-speaker conversation data is English meetings recorded with
a close-talk microphone per speaker. They give real hold/shift labels from
per-track VAD. Meetings have more overlap and more than two speakers, so they
are weaker than one-to-one calls.

| Source | Licence | Languages | Notes |
|---|---|---|---|
| AMI Meeting Corpus | CC BY 4.0 per the current Edinburgh download page; older releases were under a modified NC-SA licence, so use only the current release | English | ~100 h, individual headset microphones. Selected (4-minute window per meeting). |
| ICSI Meeting Corpus | CC BY 4.0 per its Edinburgh licence page | English | ~70 h, close-talk microphones. Not selected: files are compressed and only downloadable whole (34 GB). |
| NOTSOFAR-1 | CC BY 4.0 (reported by a derivative card; confirm on the official page) | English | Close-talk per speaker in train and dev splits only. Candidate. |
| Full-duplex dual-track set (arXiv 2509.04093) | CC BY 4.0 (verify) | Chinese, English | ~15 h, one track per speaker. Small. Candidate. |

No other large, permissively licensed, multilingual per-speaker conversation
source was found on 2026-11-20. Consequences:

- Real shift labels ("someone else took the turn") exist only in English
  training data. In other languages, shift examples come from read-speech
  sentence ends and the weak segment ends of spontaneous speech.
- Non-English conversational behaviour is measured only on test sets: Indic
  DiarBench (22 Indian languages, human-labelled turns). The benchmark must
  report how far non-English results fall behind English; that gap decides
  whether more conversation data is needed.

### Licence notes

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
| Basis Conversations 1500 | Access requested 2026-11-20 and not granted. Its licence (Basis Data License 1.0) would have allowed commercial training and open weights. |
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

The whole download is capped at 20 GB (`budget_gb` in `configs/sources.toml`);
`bin/plan-downloads` and `bin/download` refuse a selection over it. A model of
100k–1M parameters deciding hold or shift from prosody needs a few hundred
thousand labelled pauses with many voices, languages and conditions, not
hundreds of hours per language. So every source downloads only the bytes it
uses:

- **Time slices of uncompressed WAV** (AMI): HTTP range requests fetch the same
  4-minute window from every headset track of each of the 171 meetings, so all
  speakers are kept.
- **Parquet row groups** (Indic DiarBench, VoxPopuli on Hugging Face): the
  file footer is read, then only the chosen row groups are fetched. The
  publisher's row-group size is the smallest unit: ~124 MB for Indic
  DiarBench, ~0.5 GB for VoxPopuli.
- **Partial tar streams** (Emilia-YODAS, FLEURS and Common Voice on Hugging
  Face): the archive is read from the start and the connection closed
  at a byte cap, keeping the members already read. Metadata files are fetched
  whole first.

Measured on 2026-11-21 by `bin/plan-downloads`:

| Source | What is fetched | Size |
|---|---|---|
| AMI | 4-minute window per meeting, all 687 headset tracks | 5.3 GB |
| Indic DiarBench | row groups up to 160 MB per language, 22 languages (test) | 3.0 GB |
| Common Voice 17.0 | metadata TSV plus the first 120 MB of each of 25 language archives (~5 h each) | 2.7 GB |
| FLEURS | first 60 MB of the test archive, 30 languages (test) | 1.8 GB |
| Emilia-YODAS | first 300 MB of one archive, 6 languages | 1.8 GB |
| VoxPopuli | one row group each for Spanish, Italian, Polish | 1.8 GB |
| MUSAN Free Sound | 100 direct noise files and the subset licence | 0.09 GB |
| AMI annotations | Manual v1.6.2 ZIP | 0.023 GB |
| OpenSLR 26 | 16 kHz simulated RIRs | 0.2 GB |
| **Total** | | **16.7 GB** |

The smoke-test sample (`--sample`) is 0.9 GB.

Not downloaded: ICSI, whose files are shorten-compressed and can only be
fetched whole; whole Common Voice locales (947 GB for all 295); whole FLEURS
(878 GB); Emilia beyond one partial archive per language (2.1 TB).

Partial streams and how their risks are handled:

- Common Voice: Mozilla's own archives keep the per-clip metadata (speaker ID,
  sentence) inside the archive, possibly after the clips, so a partial stream
  could miss it; they also need an account, terms accepted per language and
  allow 30 downloads a day. The Hugging Face copy of release 17.0 keeps the
  metadata in separate TSV files, so it is fetched whole first. Release 17.0
  (2024) is older than 27.0, which does not matter for ~5 h per language.
  Because the metadata is known first, the smoke test reports how many
  speakers the streamed clips cover per language; if the start of an archive
  is concentrated on few speakers, the byte budget is spread across the
  archive's split files.
- The smoke inspection of OpenSLR's MUSAN tar found music at the start.
  For augmentation we fetch 100 named recordings from the Free Sound
  noise subset and its licence file, using the pinned mirror revision in
  `configs/sources.toml`. The subset licence marks all selected Free Sound
  recordings public domain. Other MUSAN subsets have distinct terms.
  [Corpus](https://www.openslr.org/17/)
  [Pinned Free Sound files](https://huggingface.co/datasets/bilguun/musan-noise/tree/d179de7b0c6adf58bf295495b481386b8fdb97c1/noise/free-sound)
- Emilia-YODAS audio has been through source separation and quality
  filtering, so it is mostly clean; noise comes from augmentation for every
  source.

## Speaker variety

The model must work for voices it has never heard, so every source needs many
speakers, and no speaker may dominate.

| Source | Speakers (as stated) |
|---|---|
| Common Voice, 25 chosen locales | 81,574 contributors in total; the smallest locale (pa-IN) has 75 |
| Indic DiarBench | 485 in meetings plus ~750 in-the-wild |
| AMI, VoxPopuli, FLEURS, Emilia-YODAS | not stated on the cards; counted from the data during labelling |

Rules, enforced in code during labelling and splitting:

- Speaker counts are computed from the data for every source and language and
  reported; a source or language below 50 speakers is flagged.
- Examples are capped per speaker, so one voice cannot exceed a fixed share of
  any source's training examples.
- Splits are speaker-disjoint (see the milestone), and benchmark results come
  only from voices never seen in training.

## Labelling tools

- **Silero VAD** v6.2.3 (MIT code and ONNX model): pause boundaries and lengths
  on each per-speaker or single-speaker track.
  [Silero VAD](https://github.com/snakers4/silero-vad/tree/v6.2.3)
- **pyannote**: diarization for mixed, single-track audio such as podcasts or call
  recordings. Its boundary errors fall exactly where labels matter, so treat
  diarized labels as lower quality than per-track labels.
- **Teacher model**: a larger turn model can score unlabelled pauses, so the
  small model can be distilled from it on large unlabelled audio.
- **Forced alignment**: word boundaries in real single-speaker recordings, so
  utterances can be cut at a word boundary to make incomplete examples from real
  voices. Cut only at word boundaries and pad with real silence from the same
  recording; cutting mid-word would teach the model to spot edits instead of
  unfinished speech. The campaign aligns known transcripts with multilingual
  Whisper `tiny` and its cross-attention timestamp routine. Whisper code and
  weights are MIT. Languages without tokenizer support contribute natural
  ends only. [Whisper](https://github.com/openai/whisper)
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

### AMI manual annotations

The University of Edinburgh's [manual annotations v1.6.2](https://groups.inf.ed.ac.uk/ami/download/)
are CC BY 4.0. The 22,887,865-byte ZIP is included in the full and smoke
plans. Labelling uses the headset-to-speaker map and dialogue acts.
