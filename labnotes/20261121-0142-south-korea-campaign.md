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
