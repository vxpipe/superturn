# Additional annotation-derived AMI turns

- Reuse the five folds and 6 s G.711 GRU checkpoints in cross-validation
  report `6dbe4ae7e8bdfd90`. The held-out and calibration rows stay fixed.
  Each expanded fit keeps every original row and adds unique turns only
  from its existing speakers and meeting groups, with up to 12 rows per
  voice. Verify the archived control checkpoint hashes against the report.
- Retain the annotation rule for turn changes and backchannels. For the
  potential additions, lower the required headset dominance ratio from
  1.414 to 1.2 and raise the permissible pause-to-speech RMS ratio from
  0.50 to 0.65. The minimum signal level and rejection of timed nonword
  events in the pause remain in place. This changes the acoustic screen
  as well as the clip count. The expanded row order is deterministic and
  separate from the control. Any effect is an outcome of this combined
  change.
- Hold the number of gradient updates fixed to each control's 12 epochs,
  with two identical initialization seeds per fold and independent
  threshold fitting. Expanded fits traverse fewer epochs to match steps.
- Started `ami-more-turns` on Vast.ai. The source data is from existing
  training-role meetings. No additional recording download was needed.

## Results

- The relaxed screen passed 768 candidate turns across the original training
  meetings. Matching the fit speakers and applying the cap added 28
  examples per fold, from 21 existing voices. The expanded cohorts have
  550–554 examples. Baseline cohorts have 522–526. All fits used 396 steps.
- Report `reports/ami-more-turns/9785a8a8d90e919f.json`. The strict models
  reproduced the archived checkpoint hashes and prior metrics. At 300 ms
  the strict run averaged AUC 0.553, partial AUC 0.052, recall 10.26% and
  false completes 8.37%. Expanded training reached AUC 0.489, partial AUC
  0.049, recall 10.00% and false completes 8.57%. The descriptive paired
  AUC delta interval was -0.095 to -0.023. This modest addition combined
  with relaxed screening and a distinct batch order did not help.
- The 20 comparisons took 1m56s after preparation. The estimate at 5/20
  was 1m30s remaining. The job exited 0 and passed 64 tests. R2 verified
  503 exports and 44 reports with zero differences.
- The Astra review found that a missing strict control could previously
  start training before comparison of its checksum. The completed
  experiment reused all controls. Scoring now checks the reference hash
  before writing archives and fails before touching an invalid control.
  The added regression passed on the server (11 focused tests).

## Validation-only int8 deployment check

Exported the 1, 3 and 6 s GRUs from fold 0, seed 17, with the existing
exporter. Measured the resulting ONNX Runtime models on one pinned CPU
core in 1,000 batch-1 calls over 16 G.711 calibration clips. All three
had 144,975 parameters and approximately 302 KB. Float export maximum
absolute error was below 6e-8.

| Window | p50 (ms) | p99 (ms) |
|---|---:|---:|
| 1 s | 0.885 | 0.930 |
| 3 s | 2.528 | 2.598 |
| 6 s | 5.018 | 5.106 |

The 6 s model's quality and latency both fall short of their targets.
An ONNX Runtime profile with 40 generated inputs attributed roughly
1.98 ms and 3.96 ms per call to STFT for the 3 and 6 s windows.
The next cost was recurrence at 0.34 and 0.66 ms. This identifies the STFT
as the primary latency bottleneck on this host. Report
`reports/ami-int8-window-probe.json`. R2 verified 515 exports and 45
reports with zero differences. No frozen test was read.
