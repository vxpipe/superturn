# Synthetic turn transcripts

The first multilingual matrix uses the 14 language entries and every one
of the 77 scenarios across 22 domains in the pinned barge-in-detector
catalogue. A criterion is one language and one scenario. The first quota
is two accepted dialogues per criterion, for 2,156 dialogues in all.
Regional variants and six pause styles rotate across scenarios. Each
criterion has one dialogue with a short substantive reply as the shift.
Some dialogues begin with the caller and have an empty previous turn.
This matrix is monolingual and native-script.
It does not cross every regional, intent, register or code-switch factor.

The catalogue's MIT notice is in
[the third-party notice](third-party/barge-in-detector-MIT.txt).
[Source catalogue](https://github.com/vxpipe/barge-in-detector/tree/23fa741b19cdf5858aa0d53f6304282f9445d10f/generation/factors)

A record contains the preceding agent turn, caller text before and after
a plausible pause, and the next substantive agent reply. Derive two
training decisions from that conversation. The prefix followed by caller
continuation is a hold. A complete caller turn followed by an actual
handoff is a shift. Use the later short caller turn as the shift when
there is a preceding turn. For caller-first dialogues, use the completed
opening as the shift so the first-turn marker stays label-balanced.
Keep both decisions in one split and remove label,
future turns, pause markers and catalogue metadata from the model input.
Use synthetic-only material, distinct scenario descriptions from the
source catalogue and fictional details.

Inspect the matrix with `uv run --no-sync python -m superturn.turn_generation --plan`.
For an approved API run, load `GEMINI_API_KEY` from the ignored environment
and supply the total spend cap with `--budget-usd`. The destination defaults
to `data/turn-generation-v1`. Run the isolated `--smoke` option first to
check one dialogue per native script. The full run can reuse accepted smoke
rows and count their spend via `--seed-smoke data/turn-generation-smoke`.
Restart with the same directory and total cap to continue unfilled slots.
Increase `--attempts-per-slot` to retry only exhausted slots if necessary.
The runner keeps an atomic ledger and reserves budget before
sending each batch. The spend receipt uses uncached list prices and keeps a
conservative charge for requests without usage receipts. The provider's
billing statement is authoritative.

Gemini 3.8 Flash generates structured text. Local code normalizes
punctuation and Unicode, checks script, rejects contact-like strings and
deduplicates accepted samples. The main run uses one model call per batch.
Accepted records and their receipt are candidates for publication after a
synthetic spot check. The full runner ledger remains under ignored data.
Review the provider terms for original generated content and the competitive
use restriction before running.
[Gemini terms](https://ai.google.dev/gemini-api/terms) · [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

Export the completed matrix with
`uv run --no-sync python -m superturn.turn_training_dataset data/turn-generation-v1 NEW_OUTPUT_DIRECTORY`.
The exporter verifies the generation receipt, exact quota slots and cleaned
text. It writes `train.jsonl`, `validation.jsonl`
and a checksum manifest. All examples from one scenario, including all
languages and both labels of each dialogue, stay in one split. Identical
two-turn inputs join scenario groups before assigning the splits.
The synthetic validation split is for diagnostics. Each model input is
exactly two text lines. The first contains the preceding speaker's turn
or `---` for a genuine first turn. The second contains the current caller
turn. The exporter inserts the marker after generation.
