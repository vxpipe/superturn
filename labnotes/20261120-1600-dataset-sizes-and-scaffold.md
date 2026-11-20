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

## Decisions

- No downloads locally (user instruction); code is tested with generated
  fixtures, and the first real data is the smoke test on the campaign machine.
- Process each source into shards and delete its raw files after syncing, so
  ~1 TB of disk is enough. `disk_space>=1600` matched only 3 datacenter offers;
  `>=1000` with 24+ vCPUs and 500 Mb/s matched 13, from $0.74/h.
- Splits: per dataset, 80/10/10 by hash of source and group; groups merged
  across shared speakers (union-find); test-role sources test only.
- Datacenter vast.ai hosts only, because the Basis licence requires
  confidentiality terms for cloud compute and storage.
- `uvx vastai` needs `--python 3.12`; under Python 3.15 it tries to build
  Pillow from source and fails.

## Not yet verified

- The campaign scripts (`bin/vast-up` onwards) have not run against a real
  instance; `bin/vast-offers` has.
- The Common Voice download path has not run (no dataset IDs yet).

## Evidence

- `uv run pytest`: 9 passed.
- `bin/vast-offers` returns offers with the committed query.
