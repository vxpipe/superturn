# AMI learning curve and augmentation

- Uses AMI calibration pilot `bed0ba3de027429f`, the same validation,
  calibration, and original validation views as the transfer probe.
  Frozen test is excluded. Five paired seed runs use the same 150k GRU
  and 15-epoch schedule as the scratch arm of the
  [transfer probe](20261121-0653-ami-transfer-probe.md).
- One connected meeting group supplies 260 of 490 training clips.
  To keep the largest group represented throughout, keep a nested
  quarter, half, and all caller speakers within each connected group.
  Each represented speaker contributes all their available training
  clips, at most 12. The fractions are approximate with at least one
  caller per group. Fixed epochs allow more steps at larger sizes.
  Keep the same G.711 rendering. Require the full-size shard fingerprint
  to reproduce the prior AMI cohort.
- The augmentation comparison uses the same 490 clips rendered as
  original headset PCM in both arms. The existing augmentation path
  uses only the configured noise and reverb assets and AMI training
  voices for babble. Both experiments evaluate the same G.711 clips
  and fit thresholds on the independent calibration cohort.
- Report AUC and mean ROC recall up to 10% false completes alongside
  fixed-threshold 300 ms recall and false-complete rate. Connected-group
  bootstrap quantiles are descriptive with three groups in the paired
  assessment views.
- Started `ami-learning` in `vxpipe` using
  `uv run --no-sync python -m superturn.ami_learning bed0ba3de027429f`.
- Astra verified the nested sampling and preserved split roles.
  The augmentation path draws from the training RNG, so the arms
  share input rows and initialization seeds while their pause draws
  and batch ordering can differ. Local pytest passed 48 with 13 skips.
- An annotation-only inventory of the 99 training-cohort meetings
  found 163,953 100 ms bins with one annotated caller speaking and
  9,753 with global word silence and caller activity in the preceding
  500 ms. Counts exclude the first 3 s and last 2 s of each slice.
  These are potential future-activity training anchors before any acoustic
  screening or per-voice cap. No held-out meeting audio was read.

## Results

Report `reports/ami-learning/fcf013cd27c37a68.json`.
The five-seed means at 300 ms are below. Each seed has its own threshold
fitted on the separate calibration cohort.

| Training | Training clips / speakers | Assessment AUC | Cross-check AUC | Cross-check partial AUC | Cross-check false completes | Cross-check recall |
|---|---:|---:|---:|---:|---:|---:|
| G.711, quarter | 123 / 25 | 0.522 | 0.537 | 0.055 | 9.58% | 12.50% |
| G.711, half | 247 / 47 | 0.547 | 0.554 | 0.075 | 7.04% | 6.25% |
| G.711, full | 490 / 92 | 0.651 | 0.631 | 0.111 | 3.38% | 8.13% |
| Clean, augmentation off | 490 / 92 | 0.466 | 0.511 | 0.056 | 5.35% | 3.75% |
| Clean, augmentation on | 490 / 92 | 0.499 | 0.545 | 0.102 | 3.38% | 7.50% |

The full G.711 arm reproduced the previous training fingerprint and
scratch metrics. Ranking improved as the G.711 training cohort grew,
with more voices, clips and optimizer updates at each size. This design
does not identify which part accounts for how much of the gain.
The augmentation pipeline improved results when starting from clean
headsets. G.711 training had the highest AUC on both development views,
and all arms had low mean recall at their fixed thresholds. The cross-check
partial AUC improvement with augmentation was +0.046 (descriptive
meeting-group bootstrap quantiles +0.017 to +0.124).

All 25 runs finished in 5m00s after preparation. At 9/25 the estimate was
3m33s remaining. The job exited 0 after 61 server tests passed. R2
matched 240 exports and 41 reports with zero differences.

Next, try denser supervised future-activity targets from
the existing annotations on training voices only. Match scratch and a
shuffled-target control before treating a gain as evidence of transfer.
