# AMI telephone threshold calibration

- The earlier pilot used 19 holds to select the checkpoint and fit the
  threshold. At that sample size even one cutoff exceeds 5%. The extended
  run missed 5% on every original reference cross-check. See the
  [training run](20261121-0142-rtx-5090-training.md).
- The AMI roster links the original training recordings into 27 groups.
  One connected group spans 46 recordings, most span four recordings,
  and two groups have one or two. Keeping the largest group in the training
  cohort lets the two new validation cohorts draw speakers from multiple
  groups. A fixed hash orders the other groups. Six go to epoch selection
  and six go to threshold calibration. All recordings that share a headset
  speaker stay together. The original validation recordings provide a
  separate assessment cohort. No existing split is rewritten.
- The same word-timing, backchannel and acoustic screen applies on every
  cohort. Training uses at most 12 examples per speaker. All clips are
  rendered as G.711 and evaluated at 300 ms. The two 150k architectures
  each use seeds 17 and 29, a 3 s window, and 15 epochs. Each best checkpoint
  is selected before fitting the threshold on the calibration cohort and
  measuring its transfer to the original validation microphones.
- Started `ami-calibration` under `vxpipe` using the existing slices.
  Local pytest passed 46 with 13 GPU and optional dependency skips.
- The two new development cohorts contain 104 epoch-selection examples
  from 21 speakers (70 holds, 34 shifts) and 106 calibration examples from
  21 different speakers (51 holds, 55 shifts). Training retained 490
  clips across 92 speakers (253 holds, 237 shifts). The two assessment
  views use the existing reference-screened 35 and 103 examples from
  original validation and share speakers with each other.
- At the separately fitted thresholds, the 103-case cross-check measured
  18.75% recall at 8.45% false completes (GRU 17), 3.13% recall at 0%
  (GRU 29), 3.13% recall at 0% (Conformer 17) and 9.38% recall at 4.23%
  (Conformer 29). Conformer 29 had the highest AUC, 0.739. Its threshold
  from epoch selection alone yielded 15.63% recall at 5.63% false
  completes. Transfer across speakers is still variable. Each of the
  three calibrated runs within 5% missed at least 90% of completed turns
  on the cross-check. The frozen test remained unused.
- The Astra review found two workflow issues. Calibration now checks the
  prepared label counts before fitting any threshold. Each export archives
  a calibration sidecar bound to the checkpoint hash and the calibration
  shard hash, since the training checkpoint retains the epoch-selection
  threshold. Cohort provenance includes the source helpers that render
  the inputs.
- The final provenance-keyed report is
  `reports/ami-calibration/bed0ba3de027429f.json`. Its cohort hashes and all
  four runs' metrics match the earlier cross-check report
  `reports/ami-calibration/8bfb73ec2d851e73.json`. All four calibration
  sidecars matched their checkpoint and cohort hashes. The final job
  exited 0, server pytest passed 59 tests, and R2 matched 114 export
  files and 39 reports with zero differences.

## Claude review and experiment order

- Claude reviewed the calibration pilot and trainer read-only in the other
  tmux pane. The current AUC and low recall indicate weak ranking near the
  operating point. Claude's estimated AUC for 80% recall assumes
  equal-variance Gaussian scores. The milestone continues to use measured
  recall at 5% false completes. On the 103-case cross-check, one extra
  detected shift moves recall by 3.13 points and four false cutoffs on 71
  holds exceed the target.
- Compare small mixed-source pretraining followed by AMI fine-tuning
  against the same model trained from scratch. The existing Stage 1
  checkpoints used AMI speakers that the calibration pilot now reserves,
  so their weights cannot initialize this speaker-disjoint comparison.
  Use only existing non-AMI training data for pretraining. A below-chance
  direct score on AMI suggests possible domain mismatch and cannot prove
  that the learned representation will transfer.
- Hold the new AMI cohorts fixed. Compare ranking over multiple seeds,
  including partial AUC near 5% and uncertainty resampled by speaker.
  Report recall with the independently calibrated threshold. Both original
  validation views have already guided experiments. They are development
  evidence. The final benchmark still uses the frozen test.
- If transfer yield is small, compare AMI training sample sizes by
  connected meeting group and test augmentation. If improvement remains
  weak, try predicting future speech activity from denser word timings in
  just the training meetings before fitting the turn classifier. Keep
  inference audio-only. The current pilots use uniform language weights
  and retain the trainer's 2.0 weight for holds.
