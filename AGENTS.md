# Superturn

Superturn is a small, audio-only end-of-turn model for voice agents. Given the
caller's recent audio at a pause, it returns the probability that the caller
has finished their turn rather than pausing mid-thought.

## Scope

- Turn detection only. Voice activity detection belongs to the calling
  application, which runs Silero VAD and invokes Superturn when it sees a pause.
- Labelling uses Silero VAD for pause boundaries on per-speaker tracks and
  pyannote for diarization of mixed single-track audio.
- Audio in, probability out. Transcripts are not a model input; an application
  may combine the score with transcript rules on its own side.
- The model must work across languages and on telephone audio (8 kHz,
  G.711, line noise), not only clean wideband microphones.
- Size budget: about 1 MB of weights or less, int8, CPU inference in a few
  milliseconds. Export to ONNX.

## Repository layout

- `docs/` holds decisions and research: data sources and licences, labelling
  rules, model design, and evaluation results.
- Data generation and training code lives in this repository. Datasets,
  downloaded corpora, features and checkpoints do not; keep them under a data
  directory that git ignores.

## Labnotes

- For research, data, training and benchmark tasks, create a labnotes file with
  `bin/create-labnotes` and a two-to-four-word hyphenated task name.
- Update it as work progresses: what worked, what did not, barriers,
  workarounds, decisions and their rationale, commands run, and the evidence
  (counts, metrics, report paths) behind each claim.
- Keep labnotes factual and useful to the next person resuming the task. They
  do not replace design documents under `docs/`.
- Do not create labnotes for simple status checks or read-only inspection.
- Commit labnotes with the work they document.

## Milestones

- Milestone specifications and progress checklists live in
  `labnotes/milestones/`. Start with `labnotes/milestones/index.md`, whose
  checklist gives the order. `bin/create-labnotes` files stay at the top level
  of `labnotes/`.
- Milestone filenames and titles are descriptive and unnumbered; only the index
  owns the order.
- Each milestone states its outcome, design sources, scope, contracts,
  checkpoints and acceptance gates. Read its design sources before starting.
- Track progress with Markdown task lists. A written specification is not an
  implemented milestone: check a task only when it is done and verified, and
  check the index entry only when every acceptance gate passes.
- Record blockers and approved scope or target changes in the milestone rather
  than checking off partial work. Keep the milestone, its evidence and the
  index in sync with commits.

## Working rules

- Record each dataset's exact licence in `docs/` before using it. Keep
  non-commercial (NC) data out of any model meant for production unless that
  decision is written down.
- Never commit audio, transcripts or recordings from real customer calls.
  Use them only for local evaluation.
- Keep a fixed held-out test set per language, separate from training sources,
  and report results per language and per audio condition (wideband,
  telephone, noisy).
- The headline metric is the false "complete" rate at a fixed pause length:
  how often the model ends a turn the caller had not finished. Report it
  alongside overall accuracy and latency saved.
- Do not use reversed audio as training data: reversing flips pitch contours
  and syllable lengthening, the cues the model is meant to learn.
- Use `uv` for Python environments and run shell commands through an
  interactive zsh that sources `~/.zshrc`.
- Commit only when asked; stage exact paths.
