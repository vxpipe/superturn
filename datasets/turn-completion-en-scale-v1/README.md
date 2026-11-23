# English synthetic telephone dialogues

Gemini 3.8 Flash generated these fictional telephone conversations for
turn-completion research. The project stopped generation with 18,536
dialogues saved out of 20,000 planned. The publication audit excluded
69 dialogues with exact eight-word matches in private development
material and 228 with duplicate model inputs. This snapshot contains
18,239 dialogues across 16,367 train, 927 validation and 945 test slots.

Use the explicit `split` field to select a role. The two-line input is the
previous agent turn (`---` for a first turn) and a caller prefix at a
generated pause, or the full caller turn. Internal pauses yield hold
examples and the full turn yields a shift example. The next agent turn
is excluded from model input. The generation receipt pins the rows,
prompt and plan and records the approximately $11.15 estimated spend.

This is a partial synthetic dataset. No model was trained on this
snapshot. The JSONL contains fictional dialogue only.
