# Dataset sizes and project scaffold

2026-11-20. Checkpoint A of [First trained turn model](milestones/first-trained-model.md).

## Dataset sizes

Measured with `bin/plan-downloads` from provider listings; nothing downloaded.

- Full selection: 685 GB (table in `docs/data-options.md`). Smoke-test sample
  (`--sample`): 3.3 GB.
- Basis is 254.6 GB but uneven: es 82.4, ar 47.5, hi 32.6, en 28.0, pt 20.6 GB
  of audio; ka 7.2, ja 6.7, kk 5.3, ru 4.3, he 4.1, tr 2.7, zh 2.4, xh 2.0,
  it 2.0, zu 1.2, ko 1.2, de 1.1, fr 0.8, pl 0.5, nl 0.3, xmf 0.2 GB, uk ~0.
- Common Voice 27.0 (2026-09-11 release metadata): 947 GB for all 295 locales.
  The 25 chosen locales are 168.3 GB, 81,574 contributors, ~3,459 validated
  hours. es/de/fr dropped (52/37/31 GB; covered elsewhere); kn and gu have no
  release. Locales are single archives (no partial download).
- Mozilla Data Collective: terms must be accepted per dataset in the web UI;
  30 dataset downloads per organisation per day; Python SDK
  `datacollective` 0.7.1, `download_dataset(dataset_id, download_directory=...)`.
- Hugging Face `usedStorage` overstates sizes (history and parquet copies):
  FLEURS shows 891 GB but the test split is 50.7 GB; plan sizes are summed
  from the file tree instead.
- AMI individual headsets: 171 meetings, 687 files, 46.6 GB. The download page
  lists series IDs (ES2002) that expand to meetings (ES2002a–d); the site's own
  size estimator does not expand them and reports only 12 GB. The Hugging Face
  `edinburghcstr/ami` copy is segmented per utterance, which loses the pauses,
  so it is not usable.
- ICSI individual headsets: 922 files, 34.1 GB (the site estimate said 31 GB).
  URLs come from the download script the form generates. A first probe with
  `curl` left some sizes blank on timeouts; `superturn refresh-url-lists`
  retries HEAD requests and fails loudly instead.
- Indic DiarBench (`sarvamai/indic-diarbench`): CC BY 4.0, 22 Indian
  languages, ~108 h, 12.1 GB; mono mixes with human-corrected
  speaker-attributed timestamps. Added as a test-only source.

## Basis not available

Basis access was not granted (user, 2026-11-20). Removed from
`configs/sources.toml`; the selection drops from 685 GB to 430 GB. Real
conversation training data is now English only (AMI, ICSI); see
`docs/data-options.md`. Without Basis no source needs datacenter hosting: the
offer query now allows verified community hosts (40 offers from $0.24/h,
against 5 datacenter-only). Persistent storage became optional:
`bin/vast-fetch` brings results back locally.

## 20 GB budget

The user capped downloads at 20 GB: 430 GB is far more than a 100k-1M
parameter turn model needs. Every source now fetches only the bytes it uses:

- AMI WAVs are uncompressed (RIFF header, data at byte 44, 16 kHz 16-bit
  mono), so HTTP range requests fetch a 4-minute window per meeting, the same
  window for every headset: 5.3 GB instead of 46.6 GB, all speakers kept.
- ICSI files are NIST SPHERE with `embedded-shorten-v2.00` compression, so they
  cannot be range-sliced; ICSI is dropped.
- Parquet footers (read with pyarrow over `HfFileSystem`, a few KB each) give
  row-group sizes: VoxPopuli ~0.47-0.54 GB per row group, Indic DiarBench
  ~124 MB (Hindi: 94 rows, 10 row groups). Only chosen row groups are fetched.
- Tar archives (Emilia-YODAS, FLEURS, MUSAN, Common Voice) are streamed and the
  connection closed at a byte cap.
- Mozilla Data Collective: `POST {api}/datasets/{id}/download` with a Bearer
  key returns `downloadUrl`, a presigned URL that can be streamed. The SDK also
  has `list_datasets` (with a `locale` filter) and a sample-download endpoint.
- Result: 18.3 GB total, 0.9 GB smoke sample. Disk 200 GB; the offer query
  (24 GB GPU, 16 vCPUs, 300 Mb/s, verified, reliability > 0.98) matched 64
  offers from $0.12/h. Persistent storage is unnecessary: re-downloading
  20 GB is cheap.

## Partial-stream risks

- Common Voice switched to `fsicoli/common_voice_17_0` on Hugging Face (CC0
  tag; copy of the official 17.0 release; 122 locales, all 25 chosen present).
  Layout: `transcript/<lang>/<split>.tsv` (columns include client_id, path,
  sentence, gender) and `audio/<lang>/<split>/<lang>_<split>_<n>.tar`. The TSV
  is fetched whole, the tar streamed to the cap: 2.7 GB for 25 languages.
  Removes the Mozilla account, API key, per-language terms and the 30/day
  limit. The `datacollective` dependency is dropped.
- MUSAN: no noise-only copy on Hugging Face (`noisy-alpaca-test/MUSAN-noise-
  audio-only` is speech pre-mixed with noise; `csukuangfj/musan` is empty).
  `superturn inspect-archive` reads the start of an archive and lists member
  order without saving; run it in the smoke test. Babble and synthetic noise
  are added to the augmentation plan so noise does not depend on MUSAN.

## Decisions

- No downloads locally (user instruction); code is tested with generated
  fixtures, and the first real data is the smoke test on the campaign machine.
- Superseded 2026-11-21: process each source into shards and delete its raw
  files, so ~1 TB of disk is enough. Now raw files are kept (20 GB on a 200 GB
  disk) and stored in R2 with the derived data. `disk_space>=1600` matched only 3 datacenter offers;
  `>=1000` with 24+ vCPUs and 500 Mb/s matched 13, from $0.74/h.
- Splits: per dataset, 80/10/10 by hash of source and group; groups merged
  across shared speakers (union-find); test-role sources test only.
- Superseded 2026-11-20: datacenter vast.ai hosts only (needed for the
  Basis licence). With Basis dropped, verified community hosts are allowed.
- `uvx vastai` needs `--python 3.12`; under Python 3.15 it tries to build
  Pillow from source and fails.

## 2026-11-21: credentials, storage and tooling

- `.env`: `HF_TOKEN` checked with `HfApi.whoami` and a file listing of each
  Hugging Face source (all five ok, nothing downloaded). `bin/plan-downloads`:
  18.1 GB total after the Common Voice switch.
- Persistent storage: a private Cloudflare R2 bucket, possibly shared with
  other work. Listed read-only with boto3 (rclone is not installed locally):
  reachable, empty. User decisions: keep raw downloads as well as derived
  data, so later campaigns skip download and labelling; keep everything under
  `<bucket>/superturn/`; sync only between vast.ai and R2, never locally.
- `bin/sync` enforces both (fixed `superturn/` directory, `store:<bucket>`
  only, plain directory names, copy never sync, refuses without the
  `.campaign-machine` marker that `bin/vast-bootstrap` writes). Checked with a
  stub rclone in a scratch directory: pushes and pulls land in
  `<bucket>/superturn/<dir>`; a path in the remote, `../x`, `/etc` and an
  unmarked machine are refused with no rclone call.
- `bin/vast-up` pulls stored data after bootstrap; `bin/vast-down` pushes
  before destroying; `bin/vast-fetch` also brings back labels.
- Expected derived size (estimate, to be measured): labels 10-50 MB, shards
  10-35 GB (11 s clips while windows up to 8 s are tested), test copies
  2-5 GB.
- Progress: `superturn.progress.Progress` measures pace from work done
  (resumed work excluded), prints done/total, pace and time left every 30 s,
  and writes `$SUPERTURN_DATA/progress/<step>.json`; `bin/vast-progress`
  shows them. Wired into `superturn download` (bytes counted live through
  `_request`, parquet row groups credited on completion). Unit tests only.
- RTX 5090 offers on vast.ai (2026-11-21): 29 offers, $0.49-1.67/h, nine
  meeting the campaign filters at $0.53-0.75/h.

- Jobs and notifications (user, 2026-11-21): every campaign job runs as a
  window of the tmux session `vxpipe` (`bin/job-run`, started by
  `bin/vast-run`; command in `jobs/NAME.sh`, exit status in `jobs/NAME.exit`).
  `bin/vast-watch` polls `superturn status` every 5 minutes and sends events
  through `pushnotify`; `superturn.watch.events` decides what to send, once
  each (unit-tested). A local tmux test went to the user's own tmux server
  (`$TMUX` overrides `TMUX_TMPDIR`) and its `kill-server` killed the user's
  session; tmux is now tested only on the campaign machine.

## Not yet verified

- The campaign scripts (`bin/vast-up` onwards) have not run against a real
  instance; `bin/vast-offers` has.
- The Common Voice download path has not run (no dataset IDs yet).

## Evidence

- `uv run pytest`: 27 passed (2026-11-21).
- `bin/vast-offers` returns offers with the committed query.
