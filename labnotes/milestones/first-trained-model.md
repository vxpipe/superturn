# First trained turn model

Status: checkpoint B in progress on a South Korea RTX 5090 datacenter rental.
Last updated 2026-11-21.

## Handover

### Done

- Registered AMI manual annotations v1.6.2 (22,887,865 bytes, CC BY 4.0).
  Provider metadata puts the full selection at 16.7 GB and smoke at 0.9 GB.
  Emilia access is available. AMI is the English conversation training source.
  Indic DiarBench is held out for non-English conversation evaluation.
- Implemented source labellers, speaker counts and caps, compressed shards,
  seeded augmentation, frozen test conditions, both acoustic models,
  config-driven training, int8 ONNX export, benchmarking and the staged
  experiment queue. The runbook has the campaign commands. The smoke command
  keeps sample cache keys separate from the full campaign.
- The Python 3.12 unit tests pass. Generated CPU fixtures also trained and
  exported both architectures using an existing Python 3.14 Torch environment.
  The 1 s artifacts were 300,447 bytes and 388,961 bytes and matched Torch
  probabilities in the float ONNX parity probe.
- The previous session verified Hugging Face access, SSH configuration, the
  Vast.ai API key and a read-only R2 listing. No live campaign has run.

- Rented the approved South Korea datacenter RTX 5090 at $0.5304/hour.
  Fixed the fresh-image uv PATH after the first bootstrap stopped, and
  restarted bootstrap in `vxpipe`.
- GPT-6 Astra reviewed the local pipeline. Fixed elapsed-pause PCM, VoxPopuli
  language decoding, Common Voice TSV parsing, babble arithmetic and
  unreachable timeout decisions. Added language-stratified speaker groups
  and quarter-point progress notifications.

- Real-data smoke passed after adjusting the Emilia-YODAS adapter to its
  observed `_id` field. The retry took 2m27s with cached data. It exercised
  all six speech sources, made 3,106 shard copies and ran three GPU training
  steps before exporting a 300,447-byte int8 model and benchmarking it.
- Started the full download in `vxpipe`. Added split coverage auditing,
  the listening packet workflow, a YIN pitch reference check and R2
  copy verification for preparation. Verified a smoke report copy to R2
  and back by file size. Switched MUSAN to the pinned Free Sound noise
  subset and resumed acquisition from completed files.

### Waiting on

- Full download and licence check. The 50-per-language listening gate
  still applies before the real training queue.

### Next step

After the download exits cleanly, run the preparation command in the
[campaign runbook](../../docs/campaign-plan.md). Review real source coverage
and the listening results before starting the experiment matrix. Keep
`bin/vast-watch` running throughout. After all gates pass, send the final
PushOver message prefixed with `COMPLETE`. Leave the machine running for
at least one hour after that message so the user can review, then fetch, sync
and destroy it unless the user asks to keep it.

### Evidence

- [South Korea campaign](../20261121-0142-south-korea-campaign.md)
- [Local turn pipeline](../20261121-0112-local-turn-pipeline.md)
- [Dataset sizes and scaffold](../20261120-1600-dataset-sizes-and-scaffold.md)
- [Campaign runbook](../../docs/campaign-plan.md)

## Outcome

A first round of experiments that ends in a decision: which model to take
forward, or what the next round must test. The candidate is a 100k–1M
parameter model that runs on CPU, exported as a single int8 ONNX file of at
most 1 MB and trained only on permissively licensed data. It takes raw 16 kHz
mono PCM and the silence elapsed, and returns P(turn complete). It must be
speaker independent: it is judged only on voices it never trained on.

This is not expected to be a single training run. The work is organised as a
local build-out, followed by one planned cloud campaign that downloads the
data once and runs a whole experiment matrix. A second campaign is planned
only if the first one's decision record calls for it.

## Design sources

- [Training data options](../../docs/data-options.md): sources, licences,
  sizes, speaker variety, exclusions, labelling tools.
- [Model design](../../docs/model-design.md): call contract, input, candidate
  networks, training and evaluation.
- [Cloud campaign plan](../../docs/campaign-plan.md): machine, storage and the
  `bin/` runbook.
- `configs/sources.toml` and `configs/campaign.env`: the codified selection and
  machine defaults.

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
  speaker continuing afterwards, is treated as a backchannel. The AMI
  dialogue-act annotations, which mark backchannels, are used to check this
  rule.
- Indic DiarBench labels come from its human-corrected speaker-attributed
  timestamps instead of VAD on separate tracks.
- Each example records: source, language, speaker ID, recording ID, group,
  split, pause start, pause length, label, and how the label was made
  (per-track, human timestamps, diarized or cut).

### Splits by source, language and speaker group

Implemented in `src/superturn/splits.py`, tested in `tests/test_splits.py`.

- Aim for 80% train, 10% validation and 10% test of independent groups in
  each source and language. Give small languages validation and test groups
  when at least three independent groups can be allocated. Record missing
  coverage when the available groups cannot support all three splits.
- Recordings that share any speaker form one group, including multilingual
  speakers and AMI meetings that reuse participants. A group always has one
  assignment across its languages.
- The `superturn-language-split-v2` ordering hashes the source and group key,
  then assigns groups against the language quotas. Reordering inputs leaves
  the result identical. The shard report records group counts by split.
- Test-role sources (FLEURS, Indic DiarBench) are test only.
- Validation picks checkpoints and thresholds; test sets are never used for
  choices. Test manifests are frozen with a content hash.

### Speaker variety

- Speaker counts are computed from the data per source and language and
  reported; any source or language below 50 speakers is flagged.
- Training examples are capped per speaker (initially at most 0.5% of a
  source's training examples), so no voice dominates.
- All reported metrics are on voices never seen in training.

### Downloads

- Only datasets with a recorded permissive licence; the licence check fails
  otherwise.
- Never downloaded locally; only on the campaign machine.
- Raw downloads stay on the campaign machine (20 GB on a 200 GB disk) and,
  with labels, shards and results, are pushed to private persistent storage
  (`SUPERTURN_REMOTE`, Cloudflare R2, all under `<bucket>/superturn/`) before the machine is released, so a
  later campaign pulls them instead of downloading and labelling again.

### Progress and time estimates

- Every long step (download, labelling, sharding, test copies, each training
  run, the experiment queue, benchmarks) counts its work with
  `superturn.progress.Progress` against a total known up front (bytes,
  recordings, clips, training steps). Its log line and
  `$SUPERTURN_DATA/progress/<step>.json` give done/total, the pace measured
  from the work done so far (resumed work excluded) and the time left;
  `bin/vast-progress` shows them all from the local computer.
- The user is told each step's estimate once its pace is measured, and the
  campaign labnotes record estimate against actual per step.

## Checkpoints

### A. Local build-out (no downloads)

Everything is exercised locally with unit tests and small generated audio
fixtures. The first run on real data is the smoke test on the campaign
machine.

- [x] Python project with `uv` (Python 3.12), package `superturn`.
- [x] Source registry `configs/sources.toml` with licence, attribution, role,
  selection and smoke-test sample per source.
- [x] `bin/plan-downloads`: exact sizes from provider listings, no downloads.
  Total capped by `budget_gb` (20 GB) and refused above it; current
  selection 16.7 GB, smoke-test sample 0.9 GB.
- [x] `bin/download`: Hugging Face (pinned commit, size-verified), HTTP and
  URL lists (resumable, size-verified), Common Voice via the Mozilla Data
  Collective API; writes a manifest per source; ends with the licence check.
- [x] Partial fetchers, unit-tested: WAV time slices by HTTP range, parquet
  row groups, tar streams stopped at a byte cap.
- [x] `superturn inspect-archive`: member order at the start of an archive,
  nothing saved.
- [x] `superturn refresh-url-lists`: rebuilds the AMI headset list.
- [x] Reproducible, per-dataset, speaker-disjoint splits with tests.
- [x] Campaign scripts: `bin/vast-offers`, `vast-up`, `vast-push`,
  `vast-bootstrap`, `vast-run`, `vast-ssh`, `vast-fetch`, `vast-down`, and
  `bin/sync` for persistent storage.
- [x] Lock PyTorch, ONNX, ONNX Runtime, Silero VAD and Whisper for the
  campaign. Indic DiarBench uses human timestamps. Add pyannote when a
  mixed source needs automatic diarization.
- [x] Labelling (rules above), each with fixture tests:
  - per-track labeller (AMI): Silero VAD per track, hold/shift;
  - human-timestamp labeller (Indic DiarBench);
  - read speech (Common Voice, FLEURS): multilingual Whisper tiny as a
    forced aligner, with MIT code and weights recorded in the data options; holds cut at word boundaries away from the
    sentence end and followed by real silence from the same recording, never
    digital zeros; shifts from the natural sentence end;
  - spontaneous speech (VoxPopuli, Emilia-YODAS): pauses inside a segment are
    holds; segment ends are weak shifts in their own label class.
- [x] Speaker counting and per-speaker caps; speaker report per source and
  language.
- [x] Shard writer: pause-centred clips at 16 kHz (window plus decision
  silence plus a margin) with labels and metadata, compressed, fixed-size
  shards.
- [x] Augmentation, on the fly, seeded and logged: babble (3-8 other
  speakers from our own training split mixed in); white, pink and brown
  noise, mains hum and line hiss; telephone channel (8 kHz
  resample, G.711 μ-law and A-law round trip, 300–3400 Hz band-pass, packet
  loss 0–5%, jitter); Opus at 16 kHz and low bitrates; MUSAN noise at SNR
  0–30 dB; OpenSLR 26 reverb; gain; speed 0.9–1.1×; pitch ±2 semitones;
  low-pass filtered speech (around 400 Hz) at a small rate.
- [x] Pre-rendered test copies per condition: clean, telephone, telephone plus
  noise at 10 dB SNR.
- [x] Feature front end as exportable PyTorch modules; parity test of ONNX
  Runtime against PyTorch, maximum absolute difference below 1e-3 (float);
  pitch accuracy check against a reference pitch tracker.
- [x] Models A (recurrent) and B (conformer-style) at configurable sizes;
  one-command training from a config file with a fixed seed, per-epoch metrics,
  checkpoint choice on validation.
- [x] Export to int8 ONNX and the benchmark command (metrics in D).
- [x] Experiment queue runner: takes the stage C matrix, runs it unattended,
  keeps going past a failed run, syncs each finished run.
- [x] One smoke command: on the downloaded sample, labels, shards, a tiny
  training run, export and a benchmark report, end to end.

### B. Campaign 1: data

Follows the [campaign runbook](../../docs/campaign-plan.md#runbook).

- [x] `.env` filled (Emilia access is already granted).
- [x] Machine rented with `bin/vast-up`; smoke test passes on it, including
  the speaker count of the streamed Common Voice sample and the MUSAN member
  order from `superturn inspect-archive`.
- [ ] All sources downloaded, manifests written, licence check passes.
- [ ] Labels, speaker report, splits, frozen test sets and shards built per
  source.
- [ ] Raw downloads, labels, shards and test copies pushed to storage
  (`bin/sync push`) and listed back.
- [ ] 50 randomly sampled labelled pauses per language listened to and marked
  correct or wrong in labnotes; per-track labels at least 90% correct.

### C. Campaign 1: experiment round

Staged so later stages use what earlier ones found. All stages are queued
before the campaign starts; each stage picks its inputs from the previous
stage's results automatically.

- [ ] **Stage 1, architecture and size**: model A and model B at about 150k,
  500k and 1M parameters; 3 s window; all data; two seeds each.
- [ ] **Stage 2, window**: the best two from stage 1 at 1, 2, 4 and 8 s.
- [ ] **Stage 3, data mix**: the best configuration on English conversations
  only;
  plus read-speech cuts; plus spontaneous speech; all data; with and without
  the weak-shift class.
- [ ] **Stage 3b, learning curve**: the best configuration on 25%, 50% and
  100% of the training data. Flat from 50% to 100% means more data will not
  help; still rising means a later campaign fetches more of the source or
  language that is short.
- [ ] **Stage 4, cut-off cost**: the false-complete loss weight at three
  values on the best configuration.
- [ ] **Stage 5, export**: the best three configurations exported to int8
  ONNX; int8 against float on validation, AUC drop at most 0.01
  (quantisation-aware training if not).
- [ ] Every run's metrics, checkpoint and report fetched locally
  (`bin/vast-fetch`).

### D. Benchmark and decision

One command produces `data/reports/<run>/report.md` plus machine-readable
results. Every number is reported per source, per language, per condition
(clean, telephone, telephone plus noise) and overall, on unseen voices only.

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
- [ ] Splits reproduce exactly from the code; no speaker in more than one
  split; every training source present in train, validation and test.
- [ ] int8 ONNX file at most 1 MB, 100k–1M parameters, raw PCM in.
- [ ] p99 inference under 2 ms per call on one CPU core.
- [ ] At the chosen threshold and 300 ms of silence, on held-out conversation
  test splits, telephone condition: false-complete rate at most 5% with
  complete recall at least 80%.
- [ ] Median time to decision on shifts at most 500 ms of silence, telephone
  condition.
- [ ] No language in the test sets has a false-complete rate more than twice
  the overall rate; languages that miss are listed with their example counts.
  The English-to-non-English gap on conversation test sets is reported
  separately, since only English has real conversation training data.
- [ ] Telephone plus noise (10 dB) costs at most 3 percentage points of
  false-complete rate against clean.
- [ ] Beats every silence-only baseline on time to decision at an equal or
  lower false-complete rate.
- [ ] Campaign record in labnotes: machine, hours, cost, what ran and what
  failed.
- [ ] Benchmark report, model card draft with dataset attributions, and
  labnotes evidence committed; no audio, shards or model files committed.

## Design review

Local review on 2026-11-21 retained the initial targets. Candidate labels
exclude overlapping takeovers and score only elapsed pauses the caller could
actually reach. Validation selects architecture, data mix, loss weight and
threshold. Test shards remain reserved for the final reports. Treat
Whisper-aligned read-speech cuts and spontaneous segment ends as separate
labelling methods when interpreting the campaign. Indic DiarBench measures
transfer from English conversation training to Indian languages. FLEURS
metadata supports test-only inclusion and does not identify individual voices.
Review per-source counts and the label listening check before the training
queue. Revisit targets only with campaign evidence.
