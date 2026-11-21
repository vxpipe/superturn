# Resume the original Stage 1 matrix

- Seven of twelve original architecture/size runs were complete. Earlier
  diagnostics measured weak conversational quality and identified a faster
  3 s export. The original Stage 1 matrix still needs five Conformer runs
  before it can rank the two architecture families across all sizes.
- Added an explicit Stage 1 stop to the experiment CLI. It uses the
  original training config and checkpoint names, then writes the
  Stage 1 validation ranking and a separate receipt. This stops ahead
  of Stage 2, model selection, and the frozen test.
- Started `experiments-stage-one` on the existing Vast.ai instance.
  The job first runs the queue fixtures, then invokes
  `superturn experiments --config configs/experiments.json --device cuda --stage-one-only`
  and verifies the runs and reports backup. This work uses the approved
  original data while the HarperValleyBank pilot decision is pending.
- Local pytest passed 51 with 14 skips. Both server queue fixtures passed.
  Astra reviewed checkpoint compatibility, the early return and the
  separate receipt. The live job confirmed reuse of all seven original
  checkpoints and started `s1-b-150000-29`.
- At 330 of 14,135 optimizer steps the measured pace was 11 steps/s,
  with 20m55s estimated remaining for that run. The user prioritized
  turn quality and asked to defer further speed benchmarks until a
  model meets the quality targets.
