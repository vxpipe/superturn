# Cloud campaign plan

Status: scripts written 2026-11-20; no campaign run yet.

Datasets are never downloaded locally. Downloading is the expensive, slow
step (the full selection is 685 GB; see
[data options](data-options.md#download-plan-and-sizes)), so rented machines
are not created and destroyed ad hoc. Work is grouped into planned campaigns:
one machine, rented once, that downloads, labels, runs a whole round of
experiments and benchmarks, and saves everything worth keeping before it is
released.

Every step below is a script in `bin/`. Do not run ad hoc commands on the
machine for anything that will be repeated; add or fix a script instead.

## Before a campaign

- Code runs locally against small generated audio fixtures and unit tests
  (`uv run pytest`). No dataset audio is downloaded locally.
- `bin/plan-downloads` prices the selection from provider listings; it
  downloads nothing.
- Access is in place: Basis approved for the Hugging Face account in
  `HF_TOKEN`; Emilia access terms accepted; Common Voice terms accepted for
  each locale in the Mozilla Data Collective web UI and the dataset IDs
  recorded in `configs/sources.toml` (30 downloads per day at most).
- `.env` filled from `.env.example`: tokens, the SSH key registered with
  vast.ai, and the rclone remote for persistent storage.

## Machine

`configs/campaign.env` holds the offer query and disk size:

- one GPU with 24 GB or more; the models are small, the GPU mostly needs to be
  kept fed;
- 24 or more CPU cores: decoding, VAD, alignment and augmentation are
  CPU-bound;
- 1 TB of disk: sources are processed one at a time and their raw files
  deleted after their shards are synced;
- at least 500 Mb/s download;
- datacenter-hosted only. The Basis licence allows cloud compute and storage
  only under written confidentiality terms, so community-hosted machines are
  not used.

## Persistent storage

A private rclone remote (`SUPERTURN_REMOTE`) holds what must outlive a
machine: processed shards, frozen test sets, raw-download manifests,
checkpoints, ONNX exports, reports and logs. Basis audio and anything derived
from it that still contains audio stay private: the licence forbids
redistributing it. Only trained models, reports and code are published.

## Runbook

1. `bin/vast-offers` lists matching offers, cheapest first.
2. `bin/vast-up OFFER_ID` rents the machine, waits for it, copies the repo and
   `.env` (`bin/vast-push`) and installs everything (`bin/vast-bootstrap`).
3. Smoke test: `bin/vast-run -n smoke 'bin/download --sample'`, then the
   pipeline's smoke command on the sample. Fix and `bin/vast-push` until it
   passes. About 4 GB.
4. Acquire: `bin/vast-run -n download 'bin/download'`. Rerun after an
   interruption; it resumes. Ends with the licence check.
5. Prepare, experiment, export and benchmark: the commands the milestone adds
   for each stage, each started with `bin/vast-run` so it survives
   disconnects and logs to `$REMOTE_DATA/logs/`.
6. `bin/vast-ssh` to inspect; `bin/sync push` copies results to storage.
7. `bin/vast-down --yes` pushes results once more and destroys the machine
   only if that push succeeds.

A failed run must not stop the queue; it is logged and the next run starts.

## Later campaigns

Later campaigns start from the shards in storage (`bin/sync pull`), so they
need a machine for training only. Raw data is downloaded again only when a new
source is added.
