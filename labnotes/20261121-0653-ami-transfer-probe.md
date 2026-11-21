# AMI transfer probe

- Source pilot `bed0ba3de027429f` fixes the AMI telephone partitions.
  Connected meeting groups keep the 490 training, 104 checkpoint
  selection and 106 calibration examples speaker-disjoint. The 35-case
  assessment and 103-case cross-check share speakers from the original
  validation split. Both are development sets. The final test stays frozen.
- The old mixed checkpoints included speakers from the pilot's new
  checkpoint selection and calibration partitions. The transfer source
  uses existing Common Voice and Emilia-YODAS training shards. The
  first run stopped before rendering because the proposed 800 weak
  shifts per spontaneous corpus exceeded availability. At 300 ms, with
  the per-voice cap, Emilia has 296 eligible weak shifts and VoxPopuli has
  12. Use 800 holds and 800 shifts from Common Voice, and 256 of each class
  from Emilia, balanced within each source. Keep a cap of two examples
  per voice and class. Spontaneous segment ends keep their weak labels.
  The second attempt exposed a validation quota problem after rendering
  2,112 training clips in 40 s. The original Common Voice validation
  partition has 176 eligible holds and 203 eligible shifts, so use 160
  of each. Validate both selections before rendering G.711 audio.
- Pretrain the 150k GRU for four epochs with seeds 17 and 29. Train
  15-epoch scratch and transfer runs on the same AMI training partition
  with seeds 17, 29, 43, 59 and 71. Compare a matched 0.001 learning rate
  and a 0.0001 transfer learning rate. Select the epoch using the
  existing trainer's recall-first criterion, then fit each threshold on
  the separate calibration set. Report ordinary AUC, mean ROC recall
  from 0% through 10% false completes (normalised partial AUC), and a
  paired bootstrap interval across connected meeting groups for the mean
  transfer effect across the five seeds. The interval conditions on the
  fitted runs. There are only three original-validation meeting groups,
  so the intervals are descriptive.
- Ran `ami-transfer` on the existing Vast.ai instance with
  `uv run --no-sync python -m superturn.ami_transfer bed0ba3de027429f`.
  Local tests passed 47 with 13 skips. No dataset download was needed.
- The successful preparation rendered 2,112 training clips in 39 s.
  The 30 s sample predicted 9 s remaining. Validation rendered 320
  clips in 5 s. The rendered source cohorts have 897 and 99 known
  speakers respectively, with no shared source/speaker IDs. All rows
  are G.711 and eligible at 300 ms. The only speaker overlap among
  AMI evaluation views is 11 speakers shared by assessment and
  cross-check. Neither pilot directory contains test shards.
- When seeking another review in the user's tmux session, identify as Codex
  and share the relevant measurements, constraints, and file paths.
- The largest connected group accounts for 260 of the 490 AMI
  training clips. A whole-group learning curve would add that group
  by itself in the transition from roughly half to all examples.
  Use nested samples of whole caller speakers within each training
  group if running that curve, and keep all development cohorts
  fixed.

## Readout

The two source checkpoints scored 0.924 and 0.855 AUC on Common Voice
validation. Source transfer into AMI was inconsistent. The report is
`reports/ami-transfer/c31adb87800b5ca0.json`.

| AMI training | Assessment AUC | Assessment partial AUC | Cross-check AUC | Cross-check partial AUC | Cross-check false completes | Cross-check recall |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 0.651 | 0.255 | 0.631 | 0.111 | 3.38% | 8.13% |
| Transfer, matched learning rate | 0.550 | 0.135 | 0.624 | 0.145 | 3.38% | 6.25% |
| Transfer, lower learning rate | 0.355 | 0.006 | 0.435 | 0.068 | 6.76% | 6.88% |

Figures are five-seed means. Scores and thresholds are computed per seed.
The matched-learning-rate arm isolates initialization. The lower-learning-rate
arm changes both initialization and learning rate.

The matched transfer cross-check partial AUC delta was +0.034, with descriptive
meeting-group bootstrap quantiles [-0.038, +0.065] and a deterioration of 0.120
on the assessment view. The two views share voices and together span only
three meeting groups. This source mixture and training schedule produced no
consistent improvement, so keep the scratch comparison and proceed with
training-size and augmentation probes on the AMI cohorts.

The 17-run series took 2m43s after preparation. At 9/17 the estimate
was 1m20s remaining. The full job took 4m11s and exited 0. Server pytest
passed 60 tests in 35 s, and R2 matched 165 export files and 40 reports
with zero differences.

The user switched PushOver to consolidated experiment outcomes and verified
causes. The local watcher now prints its events without sending them to the
phone. Its one-shot check passed.
