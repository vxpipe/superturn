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
