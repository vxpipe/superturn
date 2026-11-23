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
