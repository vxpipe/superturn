# Cloud campaign plan

Status: proposal, 2026-11-20.

Downloading the datasets is the expensive, slow step: roughly 1 TB in total
(estimate). Rented GPU machines are therefore not created and destroyed ad
hoc. Work is grouped into planned campaigns: one machine, rented once, that
downloads, labels, runs a whole round of experiments and benchmarks, and saves
everything worth keeping before it is released.

## Before any campaign: everything runs locally first

Every script (download, labelling, augmentation, features, training, export,
benchmark) runs end to end on a small local sample before a machine is rented:
a few Basis conversations, a few FLEURS and Common Voice languages, a few MUSAN
files. Cloud time is for scale, not debugging.

## Machine

- One GPU with 24 GB or more. The models are small; the GPU mostly needs to be
  kept fed.
- 32 or more CPU cores: decoding, VAD, alignment and on-the-fly augmentation are
  CPU-bound.
- 2–3 TB of local NVMe: raw downloads, decoded audio and shards side by side.
- Hosted in a datacenter under written confidentiality terms. The Basis
  licence allows cloud compute and storage only under such obligations, so
  community-hosted machines are not used for Basis data.

## Persistent storage

A private object-storage bucket holds what must outlive a machine:

- processed shards (pause-centred clips with labels), so later campaigns start
  from shards instead of re-downloading raw corpora;
- frozen test sets and their manifests;
- checkpoints, ONNX exports, benchmark reports and logs.

Basis audio and anything derived from it that still contains its audio stay
private: the licence forbids redistributing it. Only trained models, reports
and code are published.

## Campaign stages

1. **Provision**: rent the machine, attach storage, install the locked
   environment, run the local smoke test on the machine.
2. **Acquire**: run all download scripts in parallel; verify hashes against the
   source manifest; run the licence check.
3. **Prepare**: decode to 16 kHz mono, label, cut pause-centred clips, build
   splits, freeze test sets, write shards. Sync shards and test sets to the
   bucket as soon as each source finishes.
4. **Experiment**: run the experiment queue (see the milestone) unattended.
   Each run writes its checkpoint, metrics and benchmark report and syncs them
   to the bucket when it finishes.
5. **Select and export**: export the best candidates to int8 ONNX, run the full
   benchmark including CPU latency, and write the comparison report.
6. **Save and release**: confirm everything is in the bucket, record the
   campaign in labnotes (machine, hours, cost, what ran, what failed), then
   release the machine.

A failed run must not stop the queue; it is logged and the next run starts.

## Later campaigns

Later campaigns start from the shards in the bucket, so they need a machine for
training only. Raw data is downloaded again only when a new source is added.
