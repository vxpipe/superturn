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

Read `~/AGENTS.md` for the user's shared environment instructions.
Start with `~/notes/superturn/milestones/index.md`. The active milestone
includes the handover.

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
- `docs/`: durable project decisions and research. Working labnotes and
  milestones live in the shared `~/notes` Git repository.
- Datasets, shards, checkpoints and reports live under `$SUPERTURN_DATA`
  (default `data/`, git-ignored), never in git.

## Campaign machine and storage

Only select vast.ai offers marked as datacenter machines. Use
`datacenter=True` when searching for campaign offers.

Real data is only ever handled on a rented vast.ai machine. The local
computer drives it with these scripts; read `docs/campaign-plan.md` for the
full runbook.

- Credentials are in `.env` (see `env.sample`): `HF_TOKEN`,
  `SUPERTURN_SSH_KEY`, and the R2 storage lines. The vast.ai API key is in
  `~/.config/vastai/`. Call the vast CLI as `uvx --python 3.12 vastai ...`
  (newer Pythons fail to build its dependencies); `bin/_lib.sh` wraps it.
- The current machine's instance ID is in `.campaign/instance`
  (git-ignored). If it is missing, no campaign machine is recorded; check
  `uvx --python 3.12 vastai show instances` before renting another.
- Check on a running campaign:
  - `bin/vast-progress`: every long step's progress, measured pace and
    estimated time left, plus the jobs in `vxpipe` and disk use. Start here.
  - `bin/vast-ssh [command]`: a shell or one command on the machine, e.g.
    `bin/vast-ssh tail -f /workspace/data/logs/<job>.log`.
  - `bin/vast-fetch`: copy results here (reports, logs, metrics, model files,
    labels; no audio) to `data/campaigns/<instance>/`.
- All work on the campaign machine runs in the tmux session `vxpipe`, so
  nothing is lost when a connection drops. Start every job with
  `bin/vast-run -n NAME 'command'` (after `bin/vast-push` for code changes):
  it becomes window NAME of `vxpipe`, logs to
  `/workspace/data/logs/NAME.log` and records its exit status. Never run
  long work in a plain ssh session. Inspect with
  `bin/vast-ssh tmux attach -t vxpipe`.
- Run `bin/vast-watch` locally from the first job until `bin/vast-down`.
  It prints step progress, estimates, stalls and availability events locally.
  Use `pushnotify "superturn: ..."` for consolidated experiment results.
  State what was attempted, what succeeded or failed, and any verified cause.
  Lead with model performance when available. Keep secrets and machine
  addresses out of messages.
- Never run tmux commands that create or remove sessions or windows on the
  local computer: agents run inside the user's own tmux, where `$TMUX` makes
  tmux reach the user's server. Test tmux behaviour on the campaign machine.
- Lifecycle: `bin/vast-offers` → `bin/vast-up OFFER_ID` (rents, pushes,
  bootstraps, pulls stored data) → work → `bin/vast-down --yes` (fetches
  results, pushes to storage, then destroys the machine). Renting and
  destroying cost money: confirm with the user first.
- Persistent storage is a private Cloudflare R2 bucket that may be shared
  with other work. Everything this project stores goes under
  `<bucket>/superturn/` (`raw`, `labels`, `shards`, `tests`, `runs`,
  `reports`, `logs`, `manifests`). `bin/sync push|pull` copies (never
  deletes) and runs only on the campaign machine, never locally: the local
  connection is slow, and data moves only between vast.ai and R2.

## Working rules

- Every long step (download, labelling, sharding, training, benchmarks)
  counts its work with `superturn.progress.Progress`, so its log and
  `bin/vast-progress` show done/total, the pace measured from the work done
  so far, and the time left. When starting such a step, tell the user the
  estimate once the pace is measured, and record estimate against actual in
  the labnotes.

- Anything that will be run more than once is a script in `bin/` or a
  `superturn` subcommand, committed to the repo: setup, downloads, campaign
  machine provisioning, labelling, training, export, benchmarks, syncing. Do
  not rely on ad hoc commands typed on a machine.
- Downloads are capped at 25 GB in total (`budget_gb` in
  `configs/sources.toml`). Fetch only the parts of a dataset that are used
  (time slices, parquet row groups, partial archive streams), never whole
  archives for a slice of them. Grow a source only when an experiment shows
  the model is short of that kind of data.
- Datasets are never downloaded locally. Local work uses unit tests and small
  generated audio fixtures. Real data is downloaded only on the campaign
  machine, following `docs/campaign-plan.md`.
- Cloud machines are rented for planned campaigns, not created and destroyed
  ad hoc. A campaign downloads once, runs a whole queue of experiments and
  benchmarks, and copies results back (`bin/vast-fetch`) before release.
- Record each dataset's exact licence in `configs/sources.toml` and `docs/`
  before using it. Only permissive licences that allow commercial training and
  publishing the weights; no non-commercial or share-alike-on-the-model data.
- Splits are reproducible, per dataset and speaker-disjoint
  (`src/superturn/splits.py`). Never split by clip. Never tune on test sets.
- Guard speaker variety: count speakers per source and language from the
  data, cap examples per speaker, and report metrics on unseen voices only.
- Never commit audio, transcripts, shards, checkpoints or recordings from real
  customer calls. Use customer calls only for local evaluation.
- This is a public GitHub repository, including its history and docs.
  Keep secrets out of both the public repository and the shared notes
  repository:
  tokens, API keys, passwords, private keys, the R2 account ID, endpoint or
  bucket name, or vast.ai instance addresses. They go in `.env`
  (git-ignored; see `env.sample`, which holds placeholders only). Refer to
  them by variable name in docs and labnotes, never by value. Check
  `git diff --cached` before every commit; if a secret is ever committed,
  tell the user at once so it can be revoked, since rewriting history does
  not unpublish it.
- Report results per language and per audio condition (wideband, telephone,
  noisy). The headline metric is the false "complete" rate at a fixed pause
  length: how often the model ends a turn the caller had not finished.
- Do not use reversed audio or preset-voice TTS as training data.
- Use `uv` for Python environments (Python 3.12) and run shell commands
  through an interactive zsh that sources `~/.zshrc`.
- When asking Claude for suggestions in the user's tmux session, identify
  yourself as Codex and include the relevant evidence, constraints and paths.
- Run `uv run pytest` before committing code. Commit only when asked; stage
  exact paths.

## Handover

- For each new experiment or research direction, ask GPT-6.2 Astra at
  `xhigh` reasoning for the next step. Give it the current metrics,
  labnotes, constraints and open questions. Then send Astra's recommendation
  and supporting evidence to Claude for independent verification. Identify
  yourself as Codex when contacting Claude. Record what the reviews agree on,
  any unresolved disagreement, and the resulting decision in the labnotes.

- Before ending a session, update the milestone's handover section and
  the current labnotes file under `~/notes/superturn/`.

## Labnotes

- Follow `~/notes/AGENTS.md`. Create research, data, training and benchmark
  notes with `~/notes/bin/create-labnotes superturn TASK-NAME`, using a
  two-to-four-word task name. Files live in `~/notes/superturn/labnotes/`.
- Update it as work progresses: what worked, what did not, barriers,
  workarounds, decisions and their rationale, commands run, and the evidence
  (counts, metrics, report paths) behind each claim.
- Keep labnotes factual and useful to the next person resuming the task. They
  do not replace design documents under `docs/`.
- Do not create labnotes for simple status checks or read-only inspection.
- When committing project work, commit the related labnotes and milestones
  in `~/notes` as their own checkpoint. Reference the implementation commit
  where useful. Push both repositories when the user asks for updates to
  their remotes.

## Milestones

- Milestone specifications and progress checklists live in
  `~/notes/superturn/milestones/`. Its `index.md` defines the order.
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
