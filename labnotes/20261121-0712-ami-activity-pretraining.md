# Future speech activity pretraining

- The [AMI learning curve](20261121-0704-ami-learning-curve.md) showed
  stronger telephone ranking with more training evidence. Use the
  99 training-cohort meetings in `bed0ba3de027429f` and their existing
  word timings for a denser supervised task. Verify source and split
  hashes against the calibration report before reading headset audio.
- Rasterize aligned words to 100 ms bins. Target presence of words on
  the caller track and on any other speaker track in four future windows
  with edges 0, 200, 500, 1,000 and 2,000 ms after the anchor. Select
  anchors using activity up to the query boundary. Include caller-only
  speech and globally silent 200–500 ms pauses after recent caller
  speech. Cap at 48 per caller in each of the two anchor types. Render
  each 3 s caller prefix as G.711 after cutting away the future.
- Pretrain the 150k GRU on eight binary targets for four epochs, with
  seeds 17 and 29. Compare the real targets to target vectors permuted
  at each elapsed pause length, preserving the joint target distribution
  and elapsed-time cue. Start both arms with the same weights and audio
  batches. Reinitialize the entire decision head to each downstream
  seed before normal turn-detection training. Use the same 15 epochs and
  five downstream seeds for forecast, shuffled and scratch.
- Use the original reference-screened AMI development scores and separate
  calibration. Keep the frozen test reserved. The 8-output head is
  training-only. The final 1-output turn model contract is unchanged.
- Started `ami-activity` in `vxpipe` and passed 49 local tests with
  13 skips. The new fixture checks that changing future frames only
  affects targets at the anchor, and that speech on another track in
  the pause excludes the anchor.

## Results

- Rendered 9,328 G.711 prefixes from 111 training speakers. There were
  5,216 active-speaker anchors and 4,112 pause anchors, including 1,060 at
  300 ms. Anchor selection took 3 s and rendering took 9 s. Four pretraining
  runs and 15 downstream runs finished in 3m12s.
- After four epochs, mean training BCE was 0.442 with future-word targets
  and 0.459 with shuffled targets. These are training losses. They show
  that the pretext objective learned some structure in the training set.
- Report `reports/ami-activity/a4129bb1788b8fbc.json`. Five-seed means at
  300 ms follow. Each threshold is fit on the independent calibration cohort.

| Initialization | Assessment AUC | Assessment partial AUC | Cross-check AUC | Cross-check partial AUC | Cross-check false completes | Cross-check recall |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 0.651 | 0.255 | 0.631 | 0.111 | 3.38% | 8.13% |
| Future activity | 0.577 | 0.191 | 0.595 | 0.158 | 2.54% | 6.25% |
| Shuffled activity | 0.649 | 0.192 | 0.647 | 0.099 | 2.25% | 3.13% |

The forecast arm improved cross-check partial AUC and fell on both views'
overall AUC. Its cross-check partial-AUC difference versus scratch was
+0.046, with descriptive connected-group resampling quantiles from -0.171
to +0.072. There were only three meeting groups. The effect on deployment
recall was weak and inconsistent across development views.

Astra reviewed the split guard, the past-only input boundary, the target
shuffle and checkpoint reuse. Pretraining cache identity records inputs
and code. It omits runtime versions, so reproducing it on a new environment
would require a clean rerun. The job exited 0 after 62 server tests passed,
and R2 matched 293 exports and 42 reports with zero differences.
The frozen test was unused.

## Silence probe and independent review

- Rescored all 15 downstream checkpoints on CPU with one thread on the
  103-case cross-check. Used normal audio and the same `_input` waveform
  with its last 4,800 samples set to zero, keeping `silence_ms=300`.
  Scratch mean AUC was 0.631 → 0.657, forecast 0.595 → 0.600, and
  shuffled 0.647 → 0.681. None of the arms lost ranking on this
  intervention. The artificial silence changes the input distribution,
  and gain here cannot rule out smaller pause-content or crosstalk effects.
- Asked Claude for read-only review from the other tmux pane and identified
  as Codex. Claude noted that the two small validation views cannot
  distinguish effects of the observed size reliably. The approximate
  AUC standard errors are 0.06 and 0.095 before clustering. The estimated
  AUC needed for the operating target assumes equal-variance Gaussian
  scores and does not replace the measured 5% false-complete/80% recall gate.
- Compare the next model variants using connected-meeting-group folds
  across the present training, checkpoint-selection and calibration
  cohorts. Keeping the 260-example group always in training leaves about
  440 examples for out-of-fold scoring. Report fold-level paired metrics
  so score scales from different fitted models cannot distort the AUC.
  Keep architecture, window, or telephone augmentation comparisons fixed
  across the same folds. If comparing a pretrained arm, restrict source
  pretraining to voices present in every relevant training fold.
