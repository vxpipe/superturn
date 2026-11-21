# South Korea campaign

## Provisioning and review

- Rented one verified RTX 5090 in a South Korea datacenter at $0.5304/hour
  including 200 GiB. The offer listed 16 effective CPU cores and 607 Mbps
  download speed. No local dataset downloads were made.
- Initial bootstrap reached SSH and installed system prerequisites and uv.
  The parent shell then failed to find uv after the child setup exited.
  Added the local bin directory to remote command, bootstrap and job PATHs.
  Restarted the existing rental's bootstrap as a `vxpipe` job.
- GPT-6 Astra reviewed the pipeline. Addressed elapsed pause audio, integer
  PCM overflow during babble, corpus language/TSV formats, datacenter
  filtering and decision-time coverage. The rental was verified against
  the datacenter and price filters before it was created.
- Added deterministic language quotas within sources and kept speaker groups
  intact across languages. The benchmark groups canonical language codes
  across sources. Added quarter-point notifications for long steps and asked
  GPT-6 Luna to monitor the campaign and watcher.

## Real-data smoke

- Recovery bootstrap passed in 1m05s and pulled the project storage prefix.
  The first smoke download transferred 951.89 MB in 1m55s, about 8.27 MB/s.
  The first pipeline stopped after 3m42s at Emilia-YODAS. All 100 inspected
  metadata files used the `_id` key, while the adapter expected `id`.
  The fix handles both schemas. The retry reused 949.75 MB of planned data
  and completed in 2m27s. Its bootstrap and smoke jobs both exited 0.
- The smoke speaker/label counts were AMI English 3/17, Common Voice Telugu
  4/62, Emilia-YODAS German 23/207, FLEURS Hindi 0 known IDs/260, Indic
  DiarBench Kashmiri 9/214 and VoxPopuli Slovenian 12/1,028. MUSAN's first
  inspected 50 MB contained music. The full acquisition will show whether
  its byte cap reaches the noise subdirectory.
- Smoke wrote 937 training, 192 validation and 659 test pause examples.
  The three test conditions yielded 3,106 shard copies across the run.
  Three GPU training steps ran with the Python 3.12 campaign stack. The
  recurrent model had 144,975 parameters, a 300,447-byte int8 export and
  float parity error of 2.98e-8. CPU p99 was 0.89 ms on 25 repetitions
  for the 1 s smoke model. Quality gates remain open.
- Launched the full acquisition as the `download` job. Checkpoint B's
  remaining gates cover the full selection, label and shard inventory, R2
  verification and listening review.

## Full acquisition

- The first measured estimate was 22m08s remaining, revised to 19m08s at
  5% with a 14.98 MB/s pace on the original plan. The 84.24 kB smoke
  report R2 write and one-way size check passed in 2s.
- The MUSAN tar opened with music in smoke. Provider metadata confirmed
  that 100 individually selectable Free Sound noise files at the pinned
  mirror revision total 87,894,158 bytes, plus 109 bytes for their public
  domain licence. Replaced the 1.5 GB archive stream and stopped the
  original download while it still processed AMI. Resume credits each
  previously completed file without counting it toward the new ETA.

- The revised plan measured 16.68 GB. The resumed acquisition credited
  4.25 GB of completed files and reached 15.98 GB in 4m38s. Its terminal
  failure was isolated to an AMI HTTP 503. Three-worker AMI retry completed
  in 1m56s, with 4.41 GB reused and about 0.87 GB newly fetched. All nine
  source manifests had the expected licences and nonempty selections,
  including 687 AMI tracks and 100 public-domain MUSAN waveforms.
- The verified manifest counts were 687 AMI tracks, one AMI annotation
  archive, 22 Indic partitions, 60 FLEURS files, 101 MUSAN files including
  the licence, one OpenSLR26 archive, six Emilia archives, three VoxPopuli
  partitions and 50 Common Voice files.

## Real-data preparation

- Started simultaneous AMI, Common Voice/FLEURS, and
  Indic/VoxPopuli/Emilia lanes. Indic processed 498 clips in 34s and
  VoxPopuli processed 3,000 clips in 4m07.
- The AMI lane stopped at meeting 7. Five of 171 meetings have a fourth
  downloaded headset with no speaker mapping. Silero found no speech on
  three extra tracks and speech on two. The corrected labeller ignores
  the silent extras and skips the two entire ambiguous meetings. The
  repaired AMI job is running.
- The initial Common Voice estimate was about 2h12m for 74,299 clips. The
  Emilia estimate was about 41m for 29,834 clips. A 1,000-clip Common
  Voice probe took 1m07s to decode and run VAD. It found 968 clips that
  qualified for alignment, so skipping ineligible clips alone could not
  significantly address the estimate.
- Installed PyAV 17.1.0 and the compiler prerequisites for Whisper's
  Triton kernels. Added spawned workers per locale and moved the global
  speaker assignment after the results merge. The separate cached-sample
  run passed with the same label counts as earlier smoke for Common
  Voice, FLEURS and Emilia-YODAS. The full read-speech and Emilia work
  will use this path.

- The superseded Common Voice/FLEURS and remaining Emilia lane processes
  were stopped after the separate sample test. Completed Indic and
  VoxPopuli outputs were retained. The prepared sources now run in
  independent `read-labels`, `label-emilia` and `label-ami` windows.
- Initial parallel Common Voice workers processed 23–31 clips/s each,
  compared with about 9 clips/s in one serial lane. The first language
  batch completed in about 2m03s–2m25s per locale. The aggregate estimate
  was revised to 14m15s remaining at 6 of 25 locales, before FLEURS.
- Added independent AMI manual-timestamp and YIN pitch reports,
  holdout gate summaries for final benchmarks, and input signatures for
  resuming completed locale or training work.

- AMI completed in 10m40s. The initial remaining estimate was 13m45s
  at 6 of 171 meetings. The repaired probe processed all 171 selections
  and used 169 meetings for labels. It produced 7,665 labels from 184
  known speakers. Speaker counts by split were train 160, validation
  12, and test 12, with zero overlap.
- The manual word-timestamp comparison matched 1,212 AMI pauses within
  200 ms and agreed on 1,103 (91.0% of matches). The report
  `reports/ami-reference.json` also counts 151 holds and 111 shifts with
  a nearby reference backchannel. Timestamp matching covers a subset
  and does not count as a listening judgment.

- Parallel Emilia-YODAS completed 29,834 clips across six locales in
  12m12s, compared with the initial serial estimate of about 41 minutes.
  It produced 46,444 labels from 1,535 known speakers. Every selected
  language has independent groups in train, validation, and test.
  Indic DiarBench produced 4,536 labels from 223 scoped speaker IDs.
  VoxPopuli produced 3,059 labels from 263 known speakers.
- Started the raw and manifest R2 backup while the read-speech workers
  were still active.

## Data preparation results

- Common Voice covered 25 locales in 16m00s and produced 283,800 labels
  from 999 known speakers. FLEURS covered 30 locales in 1m09s and
  produced 34,211 labels with speaker identities unavailable. The
  combined read job completed in 17m12s.
- The six speech sources produced 379,715 candidates. Caps retained
  296,744 unique examples, partitioned into 180,948 train, 42,847
  validation and 72,949 test. Reordering the label inputs reproduced
  the recorded source and language splits. All 43 test languages have
  holds and shifts at 300 ms. Sharding produced 442,642 condition
  copies in 15m02s and recorded content hashes by split.
- R2 copied 4.34 GB of labels and 40.68 GB of shards. One-way size
  verification found 4,074 matching label files and 3,460 matching
  shard files with zero differences. Raw copy and verification remain
  in the separate backup job.
- YIN compared 6,272 validation frames. Removing DC/sub-pitch energy
  and gating weak evidence increased voiced agreement from 52.8% to
  70.9%. Median relative pitch error remained 0.76% among jointly
  voiced frames.
- The user requested automated reviews in place of reviewing every
  language by ear. A deterministic sample covered 50 examples in each
  of 43 languages, with 33 supplemental AMI examples to audit 50
  per-track pauses. WebRTC with separate manual AMI evidence supported
  17 of 50 per-track examples, contradicted 16 and left 17 unresolved.
  SpeechBrain supported four, contradicted nine and left 37 unresolved.
  These audits check acoustic structure and do not establish semantic
  label accuracy. The first blinded Qwen2-Audio generation run abstained
  or produced an unrecognized answer on all 40 calibration examples.
  A constrained single-token follow-up is pending.
- The user requested a model card with speech-level and SNR limits and
  identified `HashNuke/superturn` as the future release repository.
  The controlled telephone sweeps and model card draft are in the code.
  Publish the reviewed ONNX there and pin the release as a Git submodule
  once the repository exists.

## Review and training preflight

- Qwen2-Audio-7B-Instruct was used on the rented machine as an independent
  audit probe. Its 16.81 GB of auditor weights finished downloading in
  3m48s. The pinned model's Apache 2.0 weights were used for review only.
  The constrained follow-up supported 17 of 40 manual-reference
  calibration examples, contradicted 18 and left five unresolved.
  It supported 16 of the 50 fixed per-track samples and was biased toward
  shift. Neither audio-language configuration passed calibration.
  [Auditor model card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- The direct manual timestamp and dialogue-act inventory supported
  989 of the 7,665 AMI labels. At 300 ms, the supported subset has 789
  train pauses from 132 speakers, 104 validation pauses from 12, and 59
  test pauses from 10. The test subset has 28 holds and 31 shifts. Most
  other labels remain unresolved by this automatic check. Requested the
  user's approval to use the smaller, reference-supported review scope.
- The validation-only pilot ran 256 GPU steps in 39s including validation.
  The observed training rate was about nine steps/s. Its validation AUC
  was 0.5775, and recall was 7.33% at 5.00% false completes. It is a pace
  probe and no decision used the test split.
- Generated-audio augmentation ran at about 110 examples/s with one
  worker and 619/s with eight. In-process PyAV reduced the measured
  Opus call from 61 to 22 ms and pitch from 31 to under 1 ms.
  A generated forward/backward probe used 64 examples per step and
  peaked below 1 GB allocation on the 8 s, near-1M candidate.
