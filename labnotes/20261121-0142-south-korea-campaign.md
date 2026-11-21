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
