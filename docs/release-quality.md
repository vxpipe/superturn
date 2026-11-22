# Independent telephone release quality

Use `superturn benchmark-release-quality` after the telephone cohorts and
a candidate ONNX artifact are frozen. The older `benchmark` command remains
a mixed-shard diagnostic. Its test labels and global threshold cannot verify
the 300 ms release quality target.

The calibration and test panels need independently adjudicated holds and
shifts on tracked, authorized voices. Use the same cohort for the audio-only
and combined model comparisons. Keep test gold under separate custody until
the calibration receipt is sealed. Place all materials on Vast and keep
identity data, audio, text bytes and receipts out of Git.

## Inputs

- The protocol JSON uses `contract: "superturn-telephone-ba-v1"`, a stable
  `panel` ID, supported `languages`, a shared `identity_namespace`, and
  `interface` of `audio` or `audio-text`. It includes
  `artifact_sha256`, `fit_roster_sha256`, `calibration_manifest_sha256` and
  `test_manifest_sha256`. Preserve its SHA-256 in the preregistration
  record before opening test labels. When comparing interfaces, record
  both plans and calibration digests before either test. Use the same
  panel ID, language list and manifests for both interfaces. The plan
  freezes one artifact per interface.
- The private fit-roster JSON includes the `identity_namespace`, a
  `coverage_receipt` linking the approved roster reconciliation, and a
  `speakers` list. Include the project voices used for fitting, teacher
  supervision, data-dependent export and model selection. Resolve aliases
  across sources before creating the roster. The code checks ID equality.
  Verification that IDs denote distinct people belongs to the cohort audit.
- Each private cohort manifest contains the `panel`, `identity_namespace`,
  `role` (`calibration` or `test`), `decision_ms` of 300, `condition` of
  `telephone`, an `adjudication_receipt`, and `audio_sha256`. The
  `participants` map binds stable recording IDs to all callers and
  counterparts. Each object in ordered `pauses` contains `pause`, `call`,
  `speaker`, `edge_sample`, `source_rate`, `quiet_samples`, `language`,
  `label` (`hold` or `shift`) and `pcm_sha256`. Confirm during rendering
  that each row is eligible and causally cut at the deadline, with no
  post-deadline audio in the 16 kHz caller channel. Use globally comparable
  IDs and include both classes for every language in each role.
- The audio file is a little-endian signed int16 NumPy array with one
  48,000-sample input per manifest row. For combined input, the manifest also
  includes `text_sha256`. Supply a uint8 NumPy array with one row of 128
  packed bytes per pause. An unavailable partial is an all-zero row.
  Freeze text from the live transcript as of the decision. The audio-only
  command can use a manifest that records a text digest.

## Calibration and test

Run each operation on Vast with the same protocol, artifact and fit roster.

```sh
superturn benchmark-release-quality calibration model.onnx plan.json fit-roster.json calibration.json calibration.npy --plan-sha256 "$PLAN_SHA"
superturn benchmark-release-quality test model.onnx plan.json fit-roster.json test.json test.npy --plan-sha256 "$PLAN_SHA" --calibration-sha256 "$CALIBRATION_SHA"
```

Add `--text path.npy` for the combined artifact. Record the calibration digest
outside the writable run directory before opening the test panel. The private
lock holds the calibration roster and thresholds under
`runs/release-quality/<panel>/<interface>/calibration.json`. The test command
verifies both digests, checks roster overlap and PCM hashes, and writes only
aggregate metrics under `reports/release-quality/<panel>/<interface>/test.json`.
Reusing a panel and interface with changed inputs or outputs is refused.

The cutoff maximizes calibration balanced accuracy, choosing the highest
score when thresholds tie. An always-hold cutoff is represented by null.
The combined mask with blank text keeps the combined cutoff. The text-only
diagnostic fits its own cutoff on calibration. Legacy recall at 5% false
completes, AUC and pAUC use the same test rows as diagnostic ranking metrics.
The point quality gate is at least 80% macro balanced accuracy and 70% in
each registered language. Use the reported class, call and speaker counts
to judge whether the panel can support an independent release claim. Pipeline
replay, matched reference models and CPU latency follow as separate checks.
