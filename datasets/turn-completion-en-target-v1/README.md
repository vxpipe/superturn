# Targeted English turn completion v1

Gemini 3.8 Flash generated 288 fictional phone dialogues across 24 scenario
families from the original synthetic training split. Each of six cells has
48 dialogues. Cells cross a full caller length of 4–10, 10–13 or 14–23
words with a neutral or affirmative/negative opening. The intentionally
overlapping ranges describe the prompt cells. The export validates each
dialogue against its own assigned span. The generated prefix is a hold and
the full caller turn before the next agent reply is a shift.

Parsing kept the original provider receipt and SHA-256. No real transcript
text was included in the requests. To derive the 576 new training examples,
run the supplement exporter described in
[text model training](../../docs/turn-text-training.md). It keeps the
original validation split and derives a deterministic same-size replay
control from the original English training pairs.
