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
| ICSI Meeting Corpus | CC BY 4.0 for signals and transcription | English | ~70 h, close-talk microphones. Edinburgh serves individual headset files. Embedded Shorten prevents direct PCM ranges. |
| NOTSOFAR-1 recorded meetings | [CC BY 4.0](https://huggingface.co/datasets/microsoft/NOTSOFAR#data-license) | English | Close-talk audio and timed transcripts. The publisher reports speaker overlap between training and Dev 1. Any new development split needs verified identities. |
| otoSpeech task-oriented releases | [CC BY 4.0](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-task-oriented-v2#licensing-information) | English | Separate speaker channels and stable pseudonymous IDs. The current Hugging Face credential has no access. |
| IMDA NSC Parts 4–6 turn provenance | [Singapore Open Data Licence v1.0](https://data.gov.sg/open-data-licence), with IMDA attribution | Singapore English and code-switched speech | Metadata census only. The [Scicom clean dialogue sidecars](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/tree/506f620ab39e981ffd777813038eef7c2ba23325/meta/dialogue) retain source terms. Join Parts 4–5 against [Recursal's mirrored rosters](https://huggingface.co/datasets/recursal/reprocessed_singapore_national_speech_corpus/tree/5e8294c3fb5f2634fd1d7004f7c0264969fe6687/SpeakerMetaData) before counting independent speakers. |
| IMDA NSC Part 4 original same-room pilot | [Singapore Open Data Licence v1.0](https://data.gov.sg/open-data-licence), with IMDA attribution | Singapore English with code-switching | Independent-phone [original sessions](https://huggingface.co/datasets/recursal/reprocessed_singapore_national_speech_corpus/blob/5e8294c3fb5f2634fd1d7004f7c0264969fe6687/Docs/Part%203%20-%206.md). Header and clock feasibility only. Qualify both channels and anchors before considering any training use. |

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
| MagicData dual-track Chinese and English (arXiv 2509.04093) | Publisher's audio licence is [CC BY-NC-ND 4.0](https://magichub.com/datasets/multi-stream-spontaneous-conversation-training-datasets_english/) |

## Channel augmentation

Telephone channel: resample to 8 kHz, G.711 μ-law/A-law round trip, band-pass
300–3400 Hz, packet loss and jitter, level changes. Production callers arrive
over the phone, so this matters more than room noise.

## Project-collected calls

Collect separate caller and partner tracks with consent for training,
evaluation and release of the trained weights. Verify capture timing on
the two tracks. Keep speakers disjoint across splits and reserve a group
of previously unseen speakers for evaluation. Store the recordings and
identifiable derivatives privately and publish aggregate results.

## Download plan and sizes

The whole download is capped at 25 GB (`budget_gb` in `configs/sources.toml`);
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

For the standalone LiveKit EoT Benchmark diagnostic, select
`--only livekit_eot_diagnostic` on the campaign server. The registered CC BY
4.0 English general shard is pinned to 109,936,364 bytes. It adds about
0.11 GB to the existing campaign selection and stays outside the frozen
suite. [Dataset card](https://huggingface.co/datasets/livekit/eot-bench-data)

The opt-in `livekit_multilingual_diagnostic` selection adds the 13
non-English general shards at the same revision, 1,389,421,846 bytes.
Use the provider's natural task-oriented pause labels for a fixed 300 ms
diagnostic by language. The shards stay outside fitting and the frozen
suite. This release has no global speaker IDs. CC BY 4.0 is recorded on
the pinned dataset card. [Dataset card](https://huggingface.co/datasets/livekit/eot-bench-data/blob/ae834af9e9bddf73d1f550c190aaeb99bf31f523/README.md)

HCRC Map Task dialogue WAVs have a separate CC BY 4.0 licence.
The opt-in `maptask-audio` probe fetches two ninety-second stereo
slices and measures channel separation and preliminary pause yield.
The probe uses only the dialogue audio. The publisher's NXT annotation
archive uses a separate CC BY-SA licence. [Audio directory licence](https://groups.inf.ed.ac.uk/maptask/signals/dialogues/00LICENSE.html) · [Published design](https://groups.inf.ed.ac.uk/maptask/maptask-description.html)

HarperValleyBank is a separate, opt-in English conversation pilot under
CC BY 4.0. `python -m superturn.harper_valley_plan` pins a speaker-disjoint
file list, and `python -m superturn.harper_valley_acquire` stages that list
by split on the campaign machine. The proposed 614 MB includes both
channels, transcripts and the metadata used for the split. The default
staging scope is train and validation. Audit timing and label agreement
before using the pilot for training. [Publisher source](https://github.com/cricketclub/gridspace-stanford-harper-valley)

ICSI is an unstaged option for English speaker breadth. The publisher's
chooser returned individual headset URLs and estimated 356 MB for
`Bdb001`. A 1 KiB range probe on Vast confirmed 16 kHz mono NIST SPHERE
with embedded Shorten. HTTP ranges work on the compressed bytes.
An exact decoded sample window needs a decoding pass from the beginning
or a seek index. For a pilot, select only channels needed by the
core word timings, pin the compressed object hashes, and record the
decoder and decoded sample intervals. Verify speaker-channel mapping
and timing before accepting any hold or shift label.
[Official ICSI download and licence](https://groups.inf.ed.ac.uk/ami/icsi/download/)

The opt-in `icsi-transcript-census` uses the original MRT transcriptions
under the same licence. It reads the publisher's meeting, participant,
channel and segment records for the entire roster and produces aggregate
speaker recurrence and provisional annotation-gap counts. It acquires
no audio and produces no training labels.

The opt-in DataHive conversation pilot pins six unscripted stereo calls
in Hindi and Indonesian plus the metadata at a total of 61.55 MB.
Callers and callees are isolated on left and right channels of a common
MP3 timeline. The publisher licenses the release under CC BY 4.0 and
explicitly permits commercial model training. Repeated participants
lack published stable IDs, so these calls form a single train-side
feasibility pool until speaker linkage can be resolved. Decode and hash
the tracks, screen channel isolation and check next-speaker labels with
acoustic timing and independent word evidence before any model fit.
[Publisher dataset card](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample)

SPRING-INX Hindi R1 is a separate pilot using the publisher's
original paired call channels and manually transcribed segments.
The Download Data page identifies the first release as CC BY 4.0.
Its directions describe mostly mock conversations and monologues,
so the paired subset serves as recorded human role-play evidence.
The publisher's train, dev and eval utterance lists can share the
same calls and their speaker fields do not prove participant identity.
The pilot excludes intersecting dev or eval intervals and keeps entire
selected recordings on the training side of Superturn. The manual
segment edges are coarse timing evidence. No test or train labels are
enrolled until the acoustic screen and independent review.
[IIT Madras Download Data](https://asr.iitm.ac.in/)

Other unstaged full downloads include Common Voice locales (947 GB for
all 295), FLEURS (878 GB) and Emilia beyond one partial archive per
language (2.1 TB).

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

### Singapore National Speech Corpus Part 6

The opt-in `nsc-part6-pilot` selection uses four original phone
tracks and their timed annotations from two provisional dyads in
two service designs. It checks audio and timing feasibility before
any dataset expansion or use in fitting. The pinned Recursal mirror
supplies FLAC and paired JSONL files. Verified tar member ranges
account for 38,003,262 payload bytes.

IMDA's corpus distribution form specifies the Singapore Open Data
Licence version 1.0, which permits commercial derived applications
and requires conspicuous attribution. The grant excludes personal
data and unlicensed third-party rights. Cite IMDA, the corpus version,
the date of access and the licence in any derivative product. Source
recordings and transcripts stay on campaign storage. [IMDA corpus
distribution and licence](https://docs.google.com/forms/d/e/1FAIpQLSd3k8wFF4GQP4yo_lDAXKjCltfYk-dE-yYpegTnCB20kr7log/viewform)
· [Singapore Open Data Licence](https://data.gov.sg/open-data-licence)
· [Pinned Recursal mirror](https://huggingface.co/datasets/recursal/reprocessed_singapore_national_speech_corpus/tree/5e8294c3fb5f2634fd1d7004f7c0264969fe6687)

### Singapore National Speech Corpus Part 3 metadata

The opt-in `nsc-part3-metadata` preflight uses the same mirror revision
and Singapore Open Data Licence v1.0. It reads same-room TAR headers
and can join them to the publisher's paired speaker-code index. The
probe processes only the participant and partner codes from the index.
It does not download audio or transcript payloads. A header and index
match still needs channel, clock and speaker-independence validation.
[Pinned Part 3 documentation](https://huggingface.co/datasets/recursal/reprocessed_singapore_national_speech_corpus/blob/5e8294c3fb5f2634fd1d7004f7c0264969fe6687/Docs/Part%203%20-%206.md)

### AppTek Call-Center Dialogues metadata

AppTek publishes split-channel English VoIP role-play and manual
segment times for evaluation and analysis, under CC BY-SA 4.0.
The current source registry rejects this licence. The release
remains outside the registry and any evaluation queue pending
a project decision on evaluation-only sources.
[Pinned AppTek dataset card](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues/blob/82a72903da3c868604f6dc20ddc8788b25ed4f86/README.md)

### AMI manual annotations

The University of Edinburgh's [manual annotations v1.6.2](https://groups.inf.ed.ac.uk/ami/download/)
are CC BY 4.0. The 22,887,865-byte ZIP is included in the full and smoke
plans. Labelling uses the headset-to-speaker map and dialogue acts.

### Archived Open Yap English pilot

The pinned Open Yap sample contains 16 opt-in English conversations from
eight voices, delivered as paired 48 kHz tracks on a shared timeline.
The sample's CC BY 4.0 grant is separate from the full corpus's access
agreement. Respect the release's requests about speaker privacy and
reproduction. The selected conversations were used in a timing and label-yield
diagnostic. The project ended the English acquisition track to focus on
underrepresented languages.
[Dataset card](https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k/blob/cae5c0617b2e225422076599ef99516b73c637e3/README.md)
· [Licence](https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k/blob/cae5c0617b2e225422076599ef99516b73c637e3/LICENSE.txt)
