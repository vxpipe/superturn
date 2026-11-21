# AMI meeting-group cross-validation

- [The future-activity probe](20261121-0712-ami-activity-pretraining.md)
  used two original-validation views sharing three meeting groups.
  Pool only train, checkpoint selection and threshold calibration from
  calibration pilot `bed0ba3de027429f`. Keep the largest connected
  component (260 clips) in training. Balance the remaining components
  by row count across five outer folds. For each fit, hold out the next
  fold for threshold calibration and train on the other rows. Score
  each eligible group once per seed. The original assessment and frozen
  test audio stay outside the experiment.
- Train from scratch for 12 fixed epochs with two seeds per fold. The
  150k GRU compares 1, 3 and 6 s windows. The 150k Conformer uses 3 s.
  One extra 3 s GRU arm applies the existing augmentation pipeline to
  G.711 training audio. Random augmentation can change pause/batch
  draws and sometimes applies another telephone transform. The comparison
  tests that pipeline as a whole.
- Compute AUC and partial AUC within each fitted score scale, then average
  across folds and paired seeds. Pool decisions after calibrating separate
  thresholds per fit. Fold bootstrap resamples meeting components within
  each outer fold and reports variation conditional on those fits. The
  folds share parts of their training data. The largest component supplies
  training evidence and no out-of-fold evidence.
- Started `ami-crossval` in `vxpipe` with
  `uv run --no-sync python -m superturn.ami_crossval bed0ba3de027429f`.
  Local pytest passed 50 with 13 skips. Astra found no blocking leakage
  or compatibility issue in the fixed-epoch trainer and pilot.
- Rehashed the three source cohorts from stored metadata and PCM. All
  three matched the calibration report. The pool has 700 rows, 134 voices
  and 27 components. The always-training component has 260 rows and 45
  voices. The five outer folds score 440 rows from 89 voices and 26
  components. Each fit uses 522–526 rows in training and 87–89 for
  calibration. Checked reuse of all seven completed original Stage 1
  checkpoints with the updated trainer. All seven still verify.

## Conversation-data options

Checked publisher documentation and file metadata while cross-validation ran.
This was a source review. The experiments still use the existing data.

- HarperValleyBank is CC BY 4.0 and contains 1,446 simulated
  customer-support calls by 59 human speakers. There are separate caller
  and agent WAVs and stable role-specific speaker IDs. The corrected
  transcript text is human sourced. Word timings and dialogue-act labels
  are machine outputs, so they need an acoustic and speaker-order audit.
  The GitHub tree metadata gives 2,725,318,056 WAV bytes and 24,359,883
  JSON bytes, without fetching any audio. Combined with the measured
  17,056,204,439 bytes of current raw data, the full audio and JSON
  inventory would reach 19,805,882,378 bytes. A small, speaker-stratified
  selection could test whether the telephone domain transfers without
  taking up the remaining budget. Assign speakers first and discard
  cross-split calls if the participant graph links the splits.
  [Publisher repository and licence](https://github.com/cricketclub/gridspace-stanford-harper-valley)
  · [Corpus paper](https://arxiv.org/abs/2010.13929)
- HCRC Map Task offers stereo files assembled from two original speaker
  recordings and NXT annotations under CC BY 4.0. A small subset could
  add goal-directed Scottish English. Split on participant identity and
  check the close microphones for leakage.
  [University download and licence](https://groups.inf.ed.ac.uk/maptask/maptasknxt.html)
- The CC BY 4.0 Perit sample has 45 performed support calls and a 3.52 GB
  file inventory, above the remaining budget in full. Real-TurnTurk has
  original Turkish two-track speech under Apache 2.0 from six speakers.
  Both are candidates for a narrow domain check. Neither gives broad
  speaker coverage.
  [Perit dataset](https://huggingface.co/datasets/perit-ai/perit-duplex-support-en)
  · [Real-TurnTurk dataset](https://huggingface.co/datasets/tugrulbayrak/Real-TurnTurk)
- The directly annotated TurnBench sets and SpokenWOZ have non-commercial
  terms. DUEL offers its anonymized transcripts and movement data under
  PDDL and requires separate permission for audio. Exclude these sources
  from openly publishable model training under the project rules.
  [TurnBench](https://www.sesame.com/journal/turnbench)
  · [SpokenWOZ](https://spokenwoz.github.io/)
  · [DUEL corpus paper](https://aclanthology.org/L16-1281.pdf)

## Results

Report `reports/ami-crossval/6dbe4ae7e8bdfd90.json`. All 50 fits finished
in 9m48s after the 3 s preparation. At 6/50 the reported estimate was
7m20s remaining. There are 245 holds and 195 shifts across the outer folds.
The decision rates pool two calibrated fits per fold at 300 ms.
The job exited 0 after 63 server tests passed. R2 matched 443 exports
and 43 reports with zero differences.

| G.711 training | Mean fold AUC | Partial AUC up to 10% | False completes | Complete recall |
|---|---:|---:|---:|---:|
| GRU, 3 s | 0.479 | 0.054 | 12.45% | 13.85% |
| GRU, 1 s | 0.507 | 0.061 | 8.78% | 11.54% |
| GRU, 6 s | 0.553 | 0.052 | 8.37% | 10.26% |
| Conformer, 3 s | 0.414 | 0.042 | 8.16% | 7.95% |
| GRU, 3 s with augmentation | 0.524 | 0.071 | 14.29% | 17.18% |

- The 6 s model improved overall AUC by 0.074 versus the 3 s GRU.
  The paired descriptive interval was +0.039 to +0.113. Partial AUC
  was effectively flat. The tested augmentation increased recall and
  false completes together. No arm transferred below 5% false completes.
- A one-off 15-feature logistic baseline on the same folds scored 0.552
  mean AUC and 0.089 mean partial AUC. Its features were 20 ms frame
  log-RMS over the pre-pause 2.7 s, 90th-percentile signal level,
  dispersion and duration fraction, and relative level, spread, fraction
  and slope over the final 160/480/1,000 ms. Standardize using fit rows,
  run 250 gradient updates with learning rate 0.1 and L2 weight 0.05,
  and choose calibration thresholds using the next fold. This is a
  diagnostic score, not a deployment selection.
- Rescored both the 3 s and 6 s GRUs on the same held-out clips, comparing
  only hold/shift pairs from the same speaker or the same connected group.
  Mean within-speaker AUC was 0.495 and 0.560 respectively, compared with
  all-pair AUC 0.479 and 0.553. Within-group AUC was 0.486 and 0.553.
  Speaker score offsets alone cannot explain the weak rankings.
- These estimates used a larger, different validation set and fixed
  epochs. They do not directly replicate the earlier 103-case result.
  Keep the frozen test and original assessment views reserved.
