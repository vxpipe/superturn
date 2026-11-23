# Synthetic turn transcripts

The first multilingual matrix uses the 14 language entries and every one
of the 77 scenarios across 22 domains in the pinned barge-in-detector
catalogue. A criterion is one language and one scenario. The first quota
is two accepted dialogues per criterion, for 2,156 dialogues in all.
Regional variants rotate where the catalogue specifies them. Each
criterion has one complete-sounding hold extension and one planning
pause in a shorter reply. This matrix is monolingual and native-script.
It does not cross every regional, intent, register or code-switch factor.

The catalogue's MIT notice is in
[the third-party notice](third-party/barge-in-detector-MIT.txt).
[Source catalogue](https://github.com/vxpipe/barge-in-detector/tree/23fa741b19cdf5858aa0d53f6304282f9445d10f/generation/factors)

A record contains the preceding agent turn, caller text before and after
a plausible pause, and the next substantive agent reply. Derive two
training decisions from that conversation. The prefix followed by caller
continuation is a hold. The complete caller turn followed by an actual
handoff is a shift. Keep both decisions in one split and remove label,
future turns, pause markers and catalogue metadata from the model input.
Use synthetic-only material, distinct scenario descriptions from the
source catalogue and fictional details.

Inspect the matrix with `uv run --no-sync python -m superturn.turn_generation --plan`.
For an approved API run, load `GEMINI_API_KEY` from the ignored environment
and supply the total spend cap with `--budget-usd`. The destination defaults
to `data/turn-generation-v1`. Restart with the same directory to continue
unfilled slots. The runner keeps an atomic ledger and reserves budget before
sending each batch. The spend receipt uses uncached list prices and keeps a
conservative charge for requests without usage receipts. The provider's
billing statement is authoritative.

Gemini 3.8 Flash generates structured text. Gemini Decision 1 checks
scenario match, natural spoken language, the pause boundary and the
substantive handoff. Local code normalizes punctuation and Unicode, checks
script, rejects contact-like strings and deduplicates accepted samples.
Accepted records and their receipt are candidates for publication after a
synthetic spot check. The full runner ledger remains under ignored data.
Review the provider terms for original generated content and the competitive
use restriction before running.
[Gemini terms](https://ai.google.dev/gemini-api/terms) · [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
