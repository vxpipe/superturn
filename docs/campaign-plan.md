# Cloud campaign plan

Status: local fixture pipeline verified 2026-11-21. The first campaign is pending.

Datasets are never downloaded locally. Downloads are capped at 20 GB and
fetch only the parts of each dataset that are used (16.7 GB; see
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
- verified datacenter hosts with a reliability record above 0.98.

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
3. Run `bin/vast-run -n smoke 'bin/campaign-smoke --device cuda'` and start
   `bin/vast-watch` locally in the background. Smoke downloads the 0.9 GB
   sample into an isolated data directory, prints the MUSAN member order,
   labels every speech source, builds shards, trains for up to three steps,
   exports ONNX and writes `reports/smoke/report.md`. Smoke progress and
   reports appear under the main campaign root. Fix and `bin/vast-push` until
   it passes.
4. Acquire with `bin/vast-run -n download 'bin/download'`. Rerun after an
   interruption to resume. The job ends with the licence check.
5. After download passes, start preparation and the experiment matrix with
   these commands. Review the speaker report and automated label audit before
   starting experiments.
   ```sh
   bin/vast-run -n prepare 'bin/campaign-prepare --device cuda'
   bin/vast-run -n experiments 'uv run --no-sync python -m superturn.ami_training && uv run --no-sync superturn experiments --device cuda --config configs/experiments.json'
   ```
   Use `bin/campaign-prepare --finish` after separately completed label jobs.
   That path verifies the derived outputs. Run or verify the independent raw
   and manifest backup before closing the data checkpoint.
   The shard manifest shows per-language group and label counts after caps.
   The automated label audit summarizes 50 sampled pauses per language and
   retains acoustic contradictions and unresolved cases. The on-machine
   listening packet remains available for manual follow-up.
   The experiment queue can reuse a finished training run when the shard
   fingerprints, configuration, device, Torch version and training code match.
   One exact training source hash transition is permitted for the tested
   move from threads to spawned augmentation workers. The original
   checkpoint metadata and checksum stay in effect.
   Use `superturn compare-ami-validation` on the training server to compare
   completed Stage 1 checkpoints on held-out AMI telephone audio. The report
   includes each saved threshold and an AMI validation fit for diagnosing
   the effect of threshold calibration.
   The original experiment queue ranked configurations by recall on AMI telephone
   validation at up to 5% false completes, then by AUC on the same rows.
   Pooled clean validation scores broke ties. The legacy evaluator uses
   that AMI calibration and the historical recall-at-5% gate for continuity.
   For release, freeze per-language thresholds on independent calibration
   before scoring held-out telephone calls from unseen voices at 300 ms.
   Defer CPU latency until balanced accuracy reaches at least 80% macro and
   70% per supported language. Keep the pooled float-to-int8 AUC export check.
   Each job runs in the tmux session `vxpipe`, logs under
   `$REMOTE_DATA/logs/`, and records its exit status under
   `$REMOTE_DATA/jobs/`. The watcher logs estimates, stalls and step
   progress locally. Use PushOver for experiment summaries with the outcome,
   measured model performance and any verified cause.
6. `bin/vast-progress` shows each long step's progress, measured pace and
   estimated time left (from `$REMOTE_DATA/progress/<step>.json`); `bin/vast-ssh` to inspect; `bin/vast-fetch` copies results here at any
   time. Use `bin/vast-fetch --results-only` for reports and model files
   without the larger pause label files.
   To include the AMI reference diagnostic, freeze its scope with
   `superturn benchmark-ami-reference --prepare` before any selected model
   scores test data. Run `superturn benchmark-ami-reference` on the training
   server once the validation selection is available. It uses the existing
   selected models and thresholds, with no change to the main benchmark.
7. `bin/vast-down --yes` fetches results (and pushes raw downloads, labels
   and shards if a storage remote is set), then destroys the machine only if that succeeded.

A failed run must not stop the queue; it is logged and the next run starts.

To resume architecture and size comparisons while reserving the frozen test,
run only Stage 1. It verifies and reuses finished checkpoints and stops
after writing the validation ranking and `stage1-queue.json`.

```sh
bin/vast-run -n experiments-stage-one 'uv run --no-sync superturn experiments --config configs/experiments.json --device cuda --stage-one-only'
```

The isolated AMI label pilot can run against the original campaign data
with `bin/vast-run -n ami-label-pilot 'uv run --no-sync python -m superturn.ami_pilot && bin/sync push exports reports'`.
It stores the matched data beneath `shards/ami-pilot/`, the versioned
validation report beneath `reports/ami-label-pilot/`, and the four
checkpoint copies beneath `exports/ami-label-pilot/`.

Compare matched training mixtures using the completed label pilot's report ID.

```sh
bin/vast-run -n ami-mixture 'uv run --no-sync python -m superturn.ami_mixture d57f1d5f05a30a91 --epochs 20 && bin/sync push exports reports && bin/sync verify exports reports'
```

The versioned reports and checkpoints live beneath `reports/ami-mixture/`
and `exports/ami-mixture/`.

Run the AMI telephone calibration pilot on the existing AMI slices.

```sh
bin/vast-run -n ami-calibration 'uv run --no-sync python -m superturn.ami_calibration && bin/sync push exports reports && bin/sync verify exports reports'
```

The pilot uses connected AMI meeting groups for training, epoch selection,
and threshold calibration. It evaluates with the original reference-screened
AMI validation recordings. Its reports and exports use the `ami-calibration/`
subdirectory. The original campaign shards are outside the pilot directory.

Use the completed calibration pilot's report ID for the transfer probe.

```sh
bin/vast-run -n ami-transfer 'uv run --no-sync python -m superturn.ami_transfer bed0ba3de027429f && bin/sync push exports reports && bin/sync verify exports reports'
```

The pretraining samples come from existing non-AMI train and validation
shards. The transfer probe uses the calibration pilot's unchanged AMI
cohorts. The reports and checkpoints are under `ami-transfer/`.

The AMI learning curve and augmentation comparison use the same calibration
pilot ID.

```sh
bin/vast-run -n ami-learning 'uv run --no-sync python -m superturn.ami_learning bed0ba3de027429f && bin/sync push exports reports && bin/sync verify exports reports'
```

The reports and checkpoints are under `ami-learning/`.

The supervised future-speech pilot uses the same AMI calibration pilot.

```sh
bin/vast-run -n ami-activity 'uv run --no-sync python -m superturn.ami_activity bed0ba3de027429f && bin/sync push exports reports && bin/sync verify exports reports'
```

This uses existing AMI word timings and speaker headsets from training
meetings. Find its reports and checkpoints under `ami-activity/`.

Compare AMI models across connected meeting-group folds.

```sh
bin/vast-run -n ami-crossval 'uv run --no-sync python -m superturn.ami_crossval bed0ba3de027429f && bin/sync push exports reports && bin/sync verify exports reports'
```

The pilot holds each evaluation group out of training and threshold
calibration. Results and checkpoints are under `ami-crossval/`.

Use a completed group cross-validation report to test additional AMI turns.

```sh
bin/vast-run -n ami-more-turns 'uv run --no-sync python -m superturn.ami_more_turns 6dbe4ae7e8bdfd90 && bin/sync push exports reports && bin/sync verify exports reports'
```

This keeps fold speakers and optimizer steps fixed. Results are under
`ami-more-turns/`.

For an export-only latency comparison, copy the checkpoint into a separate
output directory and run `superturn export CHECKPOINT --spectrum convolution`.
This option stores the fixed Fourier filter bank as float coefficients in
the int8 artifact. Check the artifact size, validation parity and one-core
p99 for the specific model before selecting it.

## Later campaigns

A later campaign runs `bin/sync pull` after bootstrap and starts from the
stored raw downloads, labels and shards. It needs the machine for training
only, unless the labelling code changed, in which case it relabels from the
stored raw files without downloading. Without persistent storage it downloads
and processes everything again.


## Model distribution

Fetch the selected int8 ONNX and its reports with `bin/vast-fetch`. Publish
the approved ONNX and completed model card under the user's Hugging Face model
repository `HashNuke/superturn`. Pin that repository's release commit as a Git
submodule after it exists. Keep the repository's normal code and benchmark
evidence in GitHub, with the model release in the Hugging Face repository.

## Rebuild selected conversation data

Keep the selection receipt with each training run. It records the source
revision or original index, the source byte ranges, timing and speaker
group assignments, input hashes and the project commit that generated it.
Store the receipt and derived files only where their source terms allow.
A source with restricted redistribution can be fetched again from its
publisher using the same access grant.

For the AMI hard-pause pilot, restore the original AMI selection, its word
annotations and the verified AMI development cohorts. On the campaign
machine, restore the pinned receipts from approved private storage and
run:

```sh
uv run --no-sync python -m superturn.ami_hard_pauses --plan-only
uv run --no-sync python -m superturn.ami_hard_pauses
uv run --no-sync python -m superturn.own_student_targeted_pauses --dataset-only
```

The plan under `reports/ami-hard-pauses-plan.json` pins every selected
recording, channel, sample range and candidate pause. The raw manifest,
also copied to `manifests/ami-hard-pauses.json` by `bin/sync`, pins
the audio hashes. The regenerated label manifest under
`labels/ami-hard-pauses.json` pins pause IDs and labels. A rebuild
must also reproduce `reports/ami-targeted-training-dataset.json`.
That training receipt binds each new item to the original item it
replaces, retaining the speaker and label counts. Keep the old receipt
with the old run when deliberately choosing a different cohort. This
pilot draws from existing AMI training identities and leaves all held-out
roles fixed. Run `python -m superturn.ami_hard_pauses --verify-rebuild`
for a fresh download into temporary staging and comparison with the
pinned audio and label hashes.
