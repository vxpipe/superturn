# Cloud campaign plan

Status: scripts written 2026-11-20; no campaign run yet.

Datasets are never downloaded locally. Downloads are capped at 20 GB and
fetch only the parts of each dataset that are used (18.3 GB; see
[data options](data-options.md#download-plan-and-sizes)). Rented machines are
still not created and destroyed ad hoc. Work is grouped into planned campaigns:
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
  account in `HF_TOKEN`. All other sources are public.
- `.env` filled from `env.sample`: tokens and the SSH key registered with
  vast.ai; optionally a storage remote.

## Machine

`configs/campaign.env` holds the offer query and disk size:

- one GPU with 24 GB or more; the models are small, the GPU mostly needs to be
  kept fed;
- 16 or more CPU cores: decoding, VAD, alignment and augmentation are
  CPU-bound;
- 200 GB of disk: 20 GB of downloads plus decoded audio, shards,
  checkpoints and logs;
- at least 300 Mb/s download;
- verified hosts with a reliability record above 0.98. No selected source
  requires datacenter hosting.

## Keeping results

`bin/vast-down` first runs `bin/vast-fetch`, which copies reports, logs,
download manifests, run metrics, checkpoints and ONNX exports to
`data/campaigns/<instance>/` on the local computer. These are small: a model
file is 1–4 MB.

Raw downloads, labels, shards and test copies are kept in persistent storage
(`SUPERTURN_REMOTE`, a private Cloudflare R2 bucket via rclone; see
`env.sample`). The bucket may be shared with other work, so everything goes
under one project directory, `<bucket>/superturn/<dir>` (`raw`, `labels`,
`shards`, `tests`, `runs`, `reports`, `logs`, `manifests`); `bin/sync` only
copies, never deletes. `bin/vast-down` pushes them with `bin/sync push` before it
destroys the machine; `bin/vast-up` runs `bin/sync pull` right after bootstrap, so a later campaign
starts from them. Syncing is only ever between the campaign machine and the
bucket: `bin/sync` refuses to run on any machine `bin/vast-bootstrap` has not
marked, so nothing large crosses the local connection. Only results come
here, with `bin/vast-fetch`.
This guards against a source being changed, gated or withdrawn and saves the
download and labelling hours. Expected size: about 18 GB raw plus 15–40 GB
derived. The bucket must stay private: some sources' terms (Emilia) allow use
but not redistribution.

## Runbook

1. `bin/vast-offers` lists matching offers, cheapest first.
2. `bin/vast-up OFFER_ID` rents the machine, waits for it, copies the repo and
   `.env` (`bin/vast-push`) and installs everything (`bin/vast-bootstrap`).
3. Smoke test: `bin/vast-run -n smoke 'bin/download --sample'`, then the
   pipeline's smoke command on the sample, and `uv run superturn
   inspect-archive https://www.openslr.org/resources/17/musan.tar.gz --mb 50`
   to see whether MUSAN noise falls inside its cap. Fix and `bin/vast-push`
   until it passes. About 1 GB.
4. Acquire: `bin/vast-run -n download 'bin/download'`. Rerun after an
   interruption; it resumes. Ends with the licence check.
5. Prepare, experiment, export and benchmark: the commands the milestone adds
   for each stage, each started with `bin/vast-run` as a window of the tmux
   session `vxpipe`, so it survives disconnects, logging to
   `$REMOTE_DATA/logs/` and recording its exit status in `$REMOTE_DATA/jobs/`.
   Start `bin/vast-watch` locally in the background with the first job: it
   checks every 5 minutes and sends job, estimate, stall, idle and hourly
   progress notifications through `pushnotify`.
6. `bin/vast-progress` shows each long step's progress, measured pace and
   estimated time left (from `$REMOTE_DATA/progress/<step>.json`); `bin/vast-ssh` to inspect; `bin/vast-fetch` copies results here at any
   time.
7. `bin/vast-down --yes` fetches results (and pushes raw downloads, labels
   and shards if a storage remote is set), then destroys the machine only if that succeeded.

A failed run must not stop the queue; it is logged and the next run starts.

## Later campaigns

A later campaign runs `bin/sync pull` after bootstrap and starts from the
stored raw downloads, labels and shards. It needs the machine for training
only, unless the labelling code changed, in which case it relabels from the
stored raw files without downloading. Without persistent storage it downloads
and processes everything again.
