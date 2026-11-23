# Text-only turn model training

The first diagnostic uses the original
[multilingual E5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3)
under MIT. Its published model card describes multilingual embedding
pretraining. The experiment uses a fresh binary classification head.
Pin the exact encoder, tokenizer and data checksums before fitting.

Run on the campaign machine with PyTorch and the pinned Transformers
dependency. Stage the encoder files in
`$SUPERTURN_DATA/cache/turn-text-e5-small`. Run
`uv run --no-sync python -m superturn.turn_text_train` to fit, and add
`--evaluate` to replay the final checkpoint on synthetic validation.
The default corpus is `datasets/turn-completion-v1`. The dataset loader
checks the two published train and validation hashes.

Use the preceding speaker and caller as the two segments of an
XLM-R pair. The query prefix is internal to E5. The physical model
input has native pair separators because XLM-R treats newline as
ordinary whitespace. Keep the genuine first-turn marker and reserve
the current turn before retaining the tail of the previous turn.
The 128-token cap includes special tokens.

The fixed recipe uses seed 29, the last four encoder layers, a linear
shift head on attention-masked mean pooling, unweighted binary
cross-entropy, AdamW at 2e-5, 10% warmup, batch size 32 and three epochs.
The checkpoint stores the trainable weights, optimizer, scheduler and
epoch history. Restart the same command to resume from a completed epoch.

The scorecard uses threshold 0.5 and contains synthetic diagnostic
hold and shift counts, AUC, low-false-complete recall, normalized partial
AUC, language subsets and context views. The last-word mask drops a
whitespace-delimited word where a nonempty prefix remains. It is a text
perturbation. HarperValley has its own development label audit and
causal inputs. The release score and CPU latency follow that audit.

The English development scorer is
`uv run --no-sync python -m superturn.harper_text_score` on Vast. It uses
only the frozen development pack at threshold 0.5. The scorecard includes
the 300–800 ms and longer-gap bands, a current-turn-only view and a
last-word-missing comparison on identical nonblank rows.

The scorer requires `reports/harper-text-dev-qualification.json` before
reading the development input pack. That receipt records scope
`English HarperValley development only`, status `qualified`,
`input_sha256`, `checkpoint_sha256`, and `independent_audit_reports`
as report filenames mapped to their SHA-256 checksums. Issue it only
when the independent audit supports use of the entire development pack.
The two current listening panels audit a subset and their completion
alone does not qualify all 358 inputs.

Reserve the separate HarperValley calibration inputs on Vast with
`uv run --no-sync python -m superturn.harper_text_inputs --role calibration`.
This reuses the pinned historical validation roster and clocks, and
reports word-timed first-query coverage. The calibration label audit
and cutoff fit are separate steps. The text test role stays sealed.

For the real-transcript challenge, run
`uv run --no-sync python -m superturn.harper_text_transcript_benchmark`
in a campaign job. It uses the pinned HarperValley development roster
and human-corrected telephone transcripts. It compares complete caller
runs followed by an agent reply with the same runs cut halfway or two
words early. It reports exact current-word-count and context matching,
a word-count-only baseline, and previous-turn masking. This challenge
uses deliberate cuts. The 300 ms real-pause scorecard has its own audit.

For the near-end training intervention, derive a new version from the
published synthetic corpus with
`python -m superturn.turn_text_near_end datasets/turn-completion-v1 DESTINATION`.
The broad version replaces some early holds with later prefixes while
keeping all shifts. Add `--english-length-control` to instead replace
English holds by word-count-matched prefixes cut two words before the
end of completed synthetic turns. Both keep the original validation bytes.
Use the derived directory with the trainer's `--dataset` and use a
different `--output`. For transcript evaluation use the matching
`--dataset` and `--checkpoint` with a distinct `--report-name`.

Build a targeted English supplement with
`python -m superturn.turn_text_english_supplement RUN_DIR --budget-usd 2`.
Resume using the same directory. Production generation uses the existing
Gemini ledger and validation, including exact word spans and the
affirmative/negative opening condition. No real transcript text is used.
The published supplement contains `dialogues.jsonl` and `receipt.json`.
Run the same command with that directory and `--export DESTINATION` to
derive training rows. Add `--replay-control` for the same-size English
exposure control. Both export the original validation file verbatim.

The general-language transfer comparison uses
[Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd)
under Apache 2.0. Stage the pinned weights on Vast and use
`--backbone qwen3 --base BASE_DIR` with the same trainer. The 128-token
input preserves the caller suffix and a newline between turns. The
classifier reads the last caller token and tests that right padding
preserves its logit. The scoring command reads the backbone from the
fit receipt and checks the expected weights. This is a GPU quality
comparison. CPU release qualifications have their own gate.

To compare two frozen fits on the same transcript inputs, pass
`--compare-base REFERENCE_BASE --compare-checkpoint REFERENCE_FIT` to the
transcript benchmark. Both fits must reference the same synthetic
manifest. It reports Qwen-minus-reference style differences with a
paired caller bootstrap on the full near-end set and the frozen
word-count-matched subset. Threshold 0.5 stays fixed for the operating
point metrics. Recall at 5% false completes comes from the resampled ROC.

Run the companion publisher-timed English diagnostic on Vast with
`python -m superturn.timed_text_pause_diagnostic --freeze-only`, then
rerun without the flag. It checks the previously staged LiveKit
benchmark-data shard and compares two frozen Superturn checkpoints at
the marked pauses. The inputs use timestamped words up to the publisher
pause start. Treat the result as offline text evidence. The native
telephone and unseen-speaker release gate uses qualified calls.
Run `python -m superturn.timed_text_context_audit` after the diagnostic
to compare Qwen with and without the previous turn on identical rows.
The report separates published pause-length bands and reports a
false-complete interval for bands with holds only.

### LiveKit text reference

The English reference uses the multilingual text plugin at
`v0.4.1-intl`, revision `87e35fcb1e60a569bea70346191c4886ea92e281`.
The artifact is the 396,316,457-byte INT8 ONNX model and the operating
point is the publisher's English threshold of 0.011. The model's terms
require the LiveKit Agents framework and limit use of its results to
the reference comparison. The local Agents inference process runs on
Vast using the published plugin. The scoring run has no model-service
connection and writes aggregate metrics only. [Pinned model license](https://huggingface.co/livekit/turn-detector/blob/87e35fcb1e60a569bea70346191c4886ea92e281/LICENSE)

From the Vast project directory, install the two pinned packages into
an isolated overlay and run the stage and score entry points. The
reference does not modify the project environment.

```sh
livekit_overlay="$SUPERTURN_DATA/cache/livekit-text-python"
uv pip install --python .venv/bin/python --target "$livekit_overlay" \
  'livekit-agents==1.8.13' 'livekit-plugins-turn-detector==1.8.13' \
  'transformers==5.24.0' 'onnxruntime==1.32.0'
PYTHONPATH="$livekit_overlay" .venv/bin/python -m superturn.livekit_text_reference --stage
PYTHONPATH="$livekit_overlay" .venv/bin/python -m superturn.livekit_text_reference
```

The stage retrieves the pinned weight, tokenizer and metadata files.
The scorer verifies the publisher and HarperValley input checksums. It
uses the same two-line text inputs as the current diagnostics, mapping
each line to its LiveKit speaker role and omitting the missing-turn
marker. The resulting `reports/livekit-text-reference.json` has the
released operating point, retrospective ROC metrics, and hold and
shift counts for each diagnostic. Model selection and qualification use
the independent English process described in the milestone.

After the full HarperValley streaming capture is available, run the
same scoring command with `--streaming`. The reference checks the
v2 fixed-hash batch receipt against the frozen pack, reports coverage and the
scorecard at 300 ms, and simulates the received-text calls through the
800 ms timeout. This writes `reports/livekit-text-streaming-reference.json`
separately from the timing-proxy comparison.

The English loss-weighting comparison is
`python -m superturn.timed_text_length_trial` on Vast. It keeps the same
synthetic rows and seeds in both arms and reweights the two classes
within language and caller-length bands. Build the score sensitivity
mask with `python -m superturn.timed_text_deadline_audit`. Run
`python -m superturn.timed_text_length_trial --score-only` to measure
matched-recall performance and replay the metrics with timing-sensitive
rows excluded. The original freeze remains the development reference.

To reproduce the last synthetic English number trial, run these on Vast
with the existing focused English synthetic parent and publisher freeze.
The three commands reuse versioned synthetic rows and make no model API
calls.

```sh
python -m superturn.turn_text_number_layout
python -m superturn.turn_text_number_layout \
  --export-source datasets/turn-completion-en-number-v1 \
  --dataset-out "$SUPERTURN_DATA/cache/turn-text-number-pilot-v1"
python -m superturn.timed_text_number_trial
```

The final command reuses the five weighted controls and fits five
matched candidates. It reports fixed-threshold and matched-recall
metrics, separate number-ending hold and shift cells, timing sensitivity,
and the held-out synthetic checks.

The received transcript packer is
`python -m superturn.timed_text_arrival TRACE.json PACKED.json` on Vast.
The trace identifies provider settings by SHA-256 and binds each pause
to the monotonic time of its first PCM sample, sample rate, VAD stop
and resume, and capture end. Revision events carry receipt time,
sequence number and caller-turn ID. The adapter supplies complete
replacement text, including blank retractions. The packer renders the
300 ms snapshot and the fixed 300–800 ms benchmark grid separately from
application ticks. The 800 ms snapshot is for diagnostics. A late transcript
starts the 100 ms cadence on receipt. A simultaneous caller resume or
800 ms timeout takes priority. The output file stays on Vast. Standard
output contains counts and hashes.
