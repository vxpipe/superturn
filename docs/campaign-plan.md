# Cloud campaign plan

Status: scripts written 2026-11-20; no campaign run yet.

Datasets are never downloaded locally. Downloading is the expensive, slow
step (the full selection is 430 GB; see
[data options](data-options.md#download-plan-and-sizes)), so rented machines
are not created and destroyed ad hoc. Work is grouped into planned campaigns:
one machine, rented once, that downloads, labels, runs a whole round of
experiments and benchmarks, and brings back everything worth keeping before it
is released.

Every step below is a script in `bin/`. Do not run ad hoc commands on the
machine for anything that will be repeated; add or fix a script instead.

## Before a campaign

- Code runs locally against small generated audio fixtures and unit tests
  (`uv run pytest`). No dataset audio is downloaded locally.
- `bin/plan-downloads` prices the selection from provider listings; it
  downloads nothing.
- Access is in place: Emilia access terms accepted for the Hugging Face
  account in `HF_TOKEN`; Common Voice terms accepted for each locale in the
  Mozilla Data Collective web UI and the dataset IDs recorded in
  `configs/sources.toml` (30 downloads per day at most).
- `.env` filled from `.env.example`: tokens and the SSH key registered with
  vast.ai; optionally a storage remote.

## Machine

`configs/campaign.env` holds the offer query and disk size:

- one GPU with 24 GB or more; the models are small, the GPU mostly needs to be
  kept fed;
- 24 or more CPU cores: decoding, VAD, alignment and augmentation are
  CPU-bound;
- 1 TB of disk: sources are processed one at a time and their raw files
  deleted after their shards are written;
- at least 500 Mb/s download;
- verified hosts with a reliability record above 0.98. No selected source
  requires datacenter hosting.

## Keeping results

`bin/vast-down` first runs `bin/vast-fetch`, which copies reports, logs,
download manifests, run metrics, checkpoints and ONNX exports to
`data/campaigns/<instance>/` on the local computer. These are small: a model
file is 1–4 MB. Raw audio and shards stay on the machine and are lost with it.

Persistent storage (`SUPERTURN_REMOTE`, any rclone remote such as an
S3-compatible bucket) is optional. It only matters for keeping processed
shards between campaigns, so a later campaign can skip re-downloading and
re-processing (~430 GB, a few hours of machine time). Decide after the first
campaign, when the shard size and the processing time are known.

## Runbook

1. `bin/vast-offers` lists matching offers, cheapest first.
2. `bin/vast-up OFFER_ID` rents the machine, waits for it, copies the repo and
   `.env` (`bin/vast-push`) and installs everything (`bin/vast-bootstrap`).
3. Smoke test: `bin/vast-run -n smoke 'bin/download --sample'`, then the
   pipeline's smoke command on the sample. Fix and `bin/vast-push` until it
   passes. About 3 GB.
4. Acquire: `bin/vast-run -n download 'bin/download'`. Rerun after an
   interruption; it resumes. Ends with the licence check.
5. Prepare, experiment, export and benchmark: the commands the milestone adds
   for each stage, each started with `bin/vast-run` so it survives
   disconnects and logs to `$REMOTE_DATA/logs/`.
6. `bin/vast-ssh` to inspect; `bin/vast-fetch` copies results here at any
   time.
7. `bin/vast-down --yes` fetches results (and pushes shards if a storage
   remote is set), then destroys the machine only if that succeeded.

A failed run must not stop the queue; it is logged and the next run starts.

## Later campaigns

Without persistent storage, a later campaign downloads and processes the data
again. With it, it starts from the stored shards (`bin/sync pull`) and needs
the machine for training only.
