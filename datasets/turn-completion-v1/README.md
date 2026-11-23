# Synthetic turn completion v1

Gemini 3.8 Flash generated 2,156 fictional phone dialogues across all
14 languages and all 77 scenarios in the pinned barge-in-detector
catalogue. Each of the 1,078 language-scenario combinations has two
dialogues. Each dialogue provides one hold and one shift.

| Split | Dialogues | One-turn examples | Two-turn examples | Total examples |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,848 | 560 | 3,136 | 3,696 |
| Synthetic validation | 308 | 56 | 560 | 616 |
| Total | 2,156 | 616 | 3,696 | 4,312 |

Train and validation have disjoint scenario families and model-visible
text. Within each split and context length, holds and shifts are
balanced. Each language has 154 dialogues and 308 labeled examples.

Read the `text` field as written. It contains exactly two lines. The
first is the previous speaker's turn or `---` for a caller-first example.
The second is the caller's current turn. The exporter inserts the
marker. The other fields are labels or provenance.

- `dialogues.jsonl` holds the original normalized synthetic turns and slot metadata.
- `train.jsonl` and `validation.jsonl` hold the derived model inputs.
- `manifest.json` pins the training files and records the synthetic checks.
- `generation-receipt.json` pins the dialogue bytes and run provenance.

The estimated API spend was $3.051092, including the earlier smoke
test. The production run used generation with local checks. Seven
accepted smoke dialogues were carried over. Native-script and schema
checks are structural. The length-only control reached 62.7% balanced
accuracy on synthetic validation, and the first-turn marker alone
reached 50.0%. These are properties of the synthetic corpus.

[Generation and restart procedure](../../docs/turn-generation.md)
