# Superturn

## Goal

Build a small, open, audio-only end-of-turn model for voice agents. Given the
caller's recent audio at a pause and how long the silence has lasted, it
returns the probability that the caller has finished their turn rather than
pausing mid-thought. It must:

- run on CPU in a few milliseconds, as an int8 ONNX file of at most 1 MB
  (100k–1M parameters);
- take raw 16 kHz mono PCM, with all feature extraction inside the model;
- work across many languages and on telephone audio (8 kHz G.711, line noise);
- be speaker independent: judged only on voices it never trained on;
- be released free for commercial and non-commercial use (Apache 2.0), trained
  only on permissively licensed data.

The current milestone, with a handover section on where work stands, is in
`labnotes/milestones/`. Start there.

## Scope

- Turn detection only. Voice activity detection and barge-in belong to the
  calling application, which runs Silero VAD and invokes Superturn when it
  sees a pause.
- Audio in, probability out. Transcripts are not a model input; an application
  may combine the score with transcript rules on its own side.
- Labelling uses Silero VAD for pause boundaries on per-speaker tracks and
  pyannote for diarization of mixed single-track audio.

## Repository layout

- `configs/sources.toml`: every dataset source, its licence, attribution,
  role and exact download selection. `configs/campaign.env`: campaign machine
  defaults. `configs/url-lists/`: generated file lists.
- `src/superturn/`: the package (downloads, splits, and later labelling,
  features, models, training, export, benchmark). `tests/`: unit tests.
- `bin/`: every operation that is run more than once.
- `docs/`: decisions and research. `labnotes/`: working notes and milestones.
- Datasets, shards, checkpoints and reports live under `$SUPERTURN_DATA`
  (default `data/`, git-ignored), never in git.

## Working rules

- Anything that will be run more than once is a script in `bin/` or a
  `superturn` subcommand, committed to the repo: setup, downloads, campaign
  machine provisioning, labelling, training, export, benchmarks, syncing. Do
  not rely on ad hoc commands typed on a machine.
- Datasets are never downloaded locally. Local work uses unit tests and small
  generated audio fixtures. Real data is downloaded only on the campaign
  machine, following `docs/campaign-plan.md`.
- Cloud machines are rented for planned campaigns, not created and destroyed
  ad hoc. A campaign downloads once, runs a whole queue of experiments and
  benchmarks, and syncs results to persistent storage before release.
- Record each dataset's exact licence in `configs/sources.toml` and `docs/`
  before using it. Only permissive licences that allow commercial training and
  publishing the weights; no non-commercial or share-alike-on-the-model data.
- Splits are reproducible, per dataset and speaker-disjoint
  (`src/superturn/splits.py`). Never split by clip. Never tune on test sets.
- Guard speaker variety: count speakers per source and language from the
  data, cap examples per speaker, and report metrics on unseen voices only.
- Never commit audio, transcripts, shards, checkpoints or recordings from real
  customer calls. Use customer calls only for local evaluation.
- Never commit secrets. Tokens and keys go in `.env` (git-ignored); see
  `.env.example`.
- Report results per language and per audio condition (wideband, telephone,
  noisy). The headline metric is the false "complete" rate at a fixed pause
  length: how often the model ends a turn the caller had not finished.
- Do not use reversed audio or preset-voice TTS as training data.
- Use `uv` for Python environments (Python 3.12) and run shell commands
  through an interactive zsh that sources `~/.zshrc`.
- Run `uv run pytest` before committing code. Commit only when asked; stage
  exact paths.

## Handover

- Before ending a session, update the milestone's handover section (done,
  waiting on, next step) and the current labnotes file, so another agent can
  continue from the milestone alone.

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
