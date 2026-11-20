# First trained turn model

Status: specified 2026-11-20, not started.

## Outcome

A first round of experiments that ends in a decision: which model to take
forward, or what the next round must test. The candidate is a 100k–1M
parameter model that runs on CPU, exported as a single int8 ONNX file of at
most 1 MB and trained only on permissively licensed data. It takes raw 16 kHz
mono PCM and the silence elapsed, and returns P(turn complete).

This is not expected to be a single training run. The work is organised as a
local build-out followed by one planned cloud campaign that runs a whole
experiment matrix; a second campaign is planned only if the first one's
decision record calls for it.

## Design sources

- [Training data options](../../docs/data-options.md): sources, licences,
  exclusions, labelling tools.
- [Model design](../../docs/model-design.md): call contract, input, candidate
  networks, training and evaluation.
- [Cloud campaign plan](../../docs/campaign-plan.md): machine, storage and
  campaign stages.

## Out of scope

- Voice activity detection and barge-in. The calling application runs Silero
  VAD for both.
- Application integration: the ring buffer, pause trigger and turn decisions
  live in the application.
- Transcript input.
- Pretraining on unlabelled audio, unless the experiments show a gap that needs
  it; that becomes a follow-up milestone.

## Contracts

### Model interface

- Inputs: `audio`, float32 `[batch, samples]`, 16 kHz mono PCM in [-1, 1],
  the audio before the decision (zero-padded at the start when shorter; length
  fixed by the window experiment); `silence_ms`, float32 `[batch, 1]`.
- Output: `p_complete`, float32 `[batch, 1]`.
- Feature extraction (band-limit to 0–4 kHz, log-mel, pitch, voicing, energy,
  normalisation) is inside the ONNX graph. The application passes raw PCM and
  computes nothing.
- Runs on CPU with ONNX Runtime; no GPU at inference.

### Labels

- **shift (complete)**: a speaker's speech is followed by at least the decision
  silence, and the next speech on any track belongs to another speaker, who
  starts no earlier than 200 ms after the first speaker stopped. Overlapping
  takeovers are excluded, because they are interruptions rather than turn ends.
- **hold (incomplete)**: a speaker pauses for at least the decision silence and
  then continues, with no other speaker starting in between.
- Backchannels ("mm-hm", "yeah") from the other speaker during a pause do not
  make it a shift. Speech shorter than 1 s on the other track, with the first
  speaker continuing afterwards, is treated as a backchannel. Basis human
  backchannel labels are used to check this rule.
- Each example records: source, language, speaker ID, conversation or
  recording ID, pause start, pause length, label, and how the label was made
  (per-track, diarized or cut).

### Splits and held-out test sets

- A held-out test set is data the model never trains on, kept aside to measure
  how it does on conversations and voices it has not heard.
- For Basis, a fixed share of whole conversations per language (about 10%) is
  set aside before any training. Splitting by conversation and by speaker
  means no conversation and no voice appears in both training and test; a
  random split by clip would let the model recognise voices it trained on and
  overstate its accuracy.
- FLEURS is used only as a test set.
- A validation split, also by conversation and speaker, is used to pick
  checkpoints and thresholds; test sets are never used for choices.
- Test sets are frozen once built; their manifests record a content hash.

## Checkpoints

### A. Local build-out (no cloud)

Everything runs end to end locally on a small sample before a machine is
rented.

- [ ] Python project with `uv`: package `superturn`, PyTorch, ONNX,
  ONNX Runtime, Silero VAD, pyannote. Versions pinned in the lockfile.
- [ ] `data/` (git-ignored) layout: `raw/<source>/`, `shards/`, `tests/`,
  `runs/`, `reports/`. The root is configurable by environment variable; sync
  to and from the campaign bucket is one command.
- [ ] Source manifest in `docs/` for each dataset: URL, version or download
  date, licence, attribution text, and the hash of each downloaded archive.
- [ ] Download scripts, one per source, idempotent and resumable, with a
  `--sample` mode for local runs:
  - Basis Conversations 1500: access is requested on its dataset page and
    reviewed by Basis; each person downloading needs their own approved
    access.
  - Common Voice: per language from its current distribution site; confirm and
    record the audio licence there before use. Capped slice per language
    (about 20–50 h).
  - VoxPopuli: transcribed data only, downloaded directly. Its released code
    and models (CC BY-NC) are not used or vendored.
  - Emilia-YODAS: accept the access terms, then download only the YODAS
    subset; refuse any file outside it. About 1–2k h.
  - FLEURS: all languages, test use only.
  - MUSAN and OpenSLR 26: noise and room impulse responses.
- [ ] Licence check: fails if any file under `data/raw/` belongs to a source
  without a recorded permissive licence.
- [ ] Labelling (rules above):
  - Basis: Silero VAD per speaker track, then hold/shift labels.
  - Read speech (Common Voice, FLEURS): a permissively licensed multilingual
    forced aligner, with its licence and models recorded (aligners whose
    models are non-commercial are excluded). Holds are cut at word boundaries
    away from the sentence end, followed by real silence from the same
    recording, never digital zeros. Shifts come from the natural sentence end.
  - Spontaneous speech (VoxPopuli, Emilia-YODAS): Silero VAD pauses inside a
    segment are holds; segment ends are weak shifts in their own label class,
    so their weight can be tuned or dropped.
  - Mixed single-track audio, if used: pyannote diarization, flagged as
    diarized and kept out of test sets.
- [ ] Shard writer: pause-centred clips (window plus decision silence plus a
  margin) with labels and metadata, compressed, in fixed-size shards.
- [ ] Augmentation, on the fly, seeded and logged: telephone channel (8 kHz
  resample, G.711 μ-law and A-law round trip, 300–3400 Hz band-pass, packet
  loss 0–5%, jitter); Opus at 16 kHz and low bitrates; MUSAN noise at SNR
  0–30 dB; OpenSLR 26 reverb; gain; speed 0.9–1.1×; pitch ±2 semitones;
  low-pass filtered speech (around 400 Hz) at a small rate.
- [ ] Pre-rendered test copies per condition: clean, telephone, telephone plus
  noise at 10 dB SNR.
- [ ] Feature front end as exportable PyTorch modules; parity test of ONNX
  Runtime against PyTorch on 1,000 clips, maximum absolute difference below
  1e-3 (float); pitch accuracy check against a reference pitch tracker.
- [ ] Models A (recurrent) and B (conformer-style) at configurable sizes;
  one-command training from a config file with a fixed seed, per-epoch metrics,
  and checkpoint choice on the validation split.
- [ ] Export to int8 ONNX and the benchmark command (metrics below).
- [ ] Experiment queue runner: takes the matrix below, runs it unattended, keeps
  going past a failed run, and syncs each finished run to the bucket.
- [ ] Local smoke test: the whole pipeline on the sample in one command, from
  download to benchmark report, with a tiny training run.

### B. Campaign 1: data

Follows the [cloud campaign plan](../../docs/campaign-plan.md).

- [ ] Machine provisioned to the plan (datacenter-hosted, under confidentiality
  terms suitable for Basis data); smoke test passes on it.
- [ ] All sources downloaded, hashes recorded, licence check passes.
- [ ] Labels, splits, frozen test sets and shards built; labelling report per
  source and language (counts, hold/shift balance, pause lengths).
- [ ] 50 randomly sampled labelled pauses per language listened to and marked
  correct or wrong in labnotes; Basis per-track labels at least 90% correct.
- [ ] Shards, test sets and manifests synced to the bucket.

### C. Campaign 1: experiment round

Staged so later stages use what earlier ones found. All stages are queued
before the campaign starts; each stage picks its inputs from the previous
stage's results automatically.

- [ ] **Stage 1, architecture and size**: model A and model B at about 150k,
  500k and 1M parameters; 3 s window; all data; two seeds each.
- [ ] **Stage 2, window**: the best two from stage 1 at 1, 2, 4 and 8 s.
- [ ] **Stage 3, data mix**: the best configuration on Basis only; Basis plus
  read-speech cuts; Basis plus spontaneous speech; all data; with and without
  the weak-shift class.
- [ ] **Stage 4, cut-off cost**: the false-complete loss weight at three
  values on the best configuration.
- [ ] **Stage 5, export**: the best three configurations exported to int8
  ONNX; int8 against float on validation, AUC drop at most 0.01
  (quantisation-aware training if not).
- [ ] Every run's metrics, checkpoint and report synced to the bucket.

### D. Benchmark and decision

One command produces `data/reports/<run>/report.md` plus machine-readable
results. Every number is reported per language, per condition (clean,
telephone, telephone plus noise) and overall.

- [ ] **False-complete rate** at 200, 300 and 500 ms of silence: the share of
  holds the model calls complete at the chosen threshold.
- [ ] **Complete recall** at the same points: the share of shifts called
  complete.
- [ ] **Time to decision**: for shifts, the median and 90th-percentile silence
  at which the model first calls complete, checking every 100 ms from 200 ms,
  with a 2.5 s fallback.
- [ ] **ROC AUC** at 300 ms.
- [ ] **Baselines** on the same sets: silence-only timeouts at 300, 500, 1,000
  and 1,500 ms.
- [ ] **CPU latency**: p50 and p99 per call on one CPU core with ONNX Runtime,
  batch 1.
- [ ] **Local production calls**: the same metrics on local call recordings
  when available, kept out of the repository and out of training.
- [ ] **Decision record** in labnotes: which model goes forward and why, or
  what a second campaign must test, with its expected machine time.

## Acceptance gates

The targets below are our own starting proposals, chosen for the product's
needs; they are not taken from any other project. They are revisited after
stage 1 results, and any change is recorded here with its reason.

- [ ] Only permissively licensed sources in training and test data; the
  licence check passes.
- [ ] int8 ONNX file at most 1 MB, 100k–1M parameters, raw PCM in.
- [ ] p99 inference under 2 ms per call on one CPU core.
- [ ] At the chosen threshold and 300 ms of silence, on Basis held-out
  conversations, telephone condition: false-complete rate at most 5% with
  complete recall at least 80%.
- [ ] Median time to decision on shifts at most 500 ms of silence, telephone
  condition.
- [ ] No language in the test sets has a false-complete rate more than twice
  the overall rate; languages that miss are listed with their example counts.
- [ ] Telephone plus noise (10 dB) costs at most 3 percentage points of
  false-complete rate against clean.
- [ ] Beats every silence-only baseline on time to decision at an equal or
  lower false-complete rate.
- [ ] Campaign record in labnotes: machine, hours, cost, what ran and what
  failed.
- [ ] Benchmark report, model card draft with dataset attributions, and
  labnotes evidence committed; no audio, shards or model files committed.

## Design review

Not yet reviewed. Review before checkpoint A finishes: label rules, test-set
construction, the experiment stages, and whether the targets match the
product.
