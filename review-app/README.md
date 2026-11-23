# Superturn review app

A local React and Fastify app for the Hindi listening pilot. It reads
the verified audio bundle in the parent repository's git-ignored `data/`
directory. Use Node.js 22.12 or newer and npm.

```sh
cd review-app
npm ci
npm run build
npm start
```

Open `http://127.0.0.1:8765/`, or `http://rocksalt:8765/` on the
configured tailnet. Ratings save automatically to
`hindi-review-a/ballots/<reviewer-id>.json` in the local data bundle.
Use the same browser and address to reopen your draft. Wait for
**Saved to rocksalt** before closing a tab. CSV downloads are optional.
Old drafts in the same browser migrate automatically. Each browser
session has its own review file.

In **Suggested pauses**, set the minimum and maximum detected silence
in milliseconds, then play around a pause and select the outcome you
heard. The filter only changes which items are displayed. Exports
include all saved ratings, including ones outside the current filter.
The stereo timeline shows the whole recording and marks each suggestion.
Click a marker to select it or scrub the timeline to seek. The highlighted
band initially shows the 300 ms decision window. The full detected gap
appears after the first rating.
The assisted CSV keeps the detector stop as its time origin. Its audit
JSON records detector settings, audio checksums, timing corrections, and
the first and latest outcomes. Keep assisted and manual ballots separate.
The short-clip CSV retains the full set of template IDs with blank rows
for clips outside this pilot or left unrated.

Run `npm run dev` for React hot reload at `http://127.0.0.1:5173/` and
the local Fastify API on port 8766. Browsers store drafts separately
for each origin. Use the production address for the actual review.

Set `REVIEW_BUNDLE` to an absolute path if the copied
`paired-call-floor-review` directory lives elsewhere. Keep both the
`hindi-review-a` packet and `packet-v1/listening/reviewer-a.csv` in that
directory. The app serves the review manifest and allowlisted audio IDs
over loopback.

Set `REVIEW_BALLOTS` to an alternate local ballot directory when
testing. The normal directory is git-ignored and private to the server
account.

## Prepare suggestions

For the locally authorized Hindi packet, run from the Superturn checkout.
Use the existing NumPy environment with ONNX Runtime available.

```sh
mkdir -p data/review/.models
curl -fL https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad_16k_sequence.onnx \
  -o data/review/.models/silero_vad_16k_sequence-v6.2.3.onnx
uv run --no-sync python -m superturn.review_pause_suggestions
```

The command checks the pinned model and audio hashes before writing
`hindi-review-a/suggested-pauses.json`. Restart Fastify after generating
it. Silero proposes shared gaps on the separate speaker tracks. Mixed
audio would require separately validated speaker diarization. This queue
supports assisted ratings of selected proposals and cannot measure how
many pauses the detector missed.

## Independent Hindi review

Run `npm start -- --independent` after building and open
`http://127.0.0.1:8767/` from rocksalt or an explicitly authorized SSH
forward. This instance serves the six uncued stereo recordings in the
sealed listening order. Open only this instance for the second review.
The first review and its suggestions are served by the original app.

Pause and scrub to each boundary, capture the playhead, adjust the
millisecond fields and save the pause. Listen to the whole recording
and explicitly mark it reviewed with both channel languages. Drafts
save under `hindi-review-b/ballots/` with a separate browser session.
The reviewer never needs to download or export anything.

After confirming the independent reviewer and identifying their saved
draft, prepare both CSVs from the same revision as the operator:

```sh
npm run prepare-independent -- SAVED_DRAFT.json \
  ../data/review/paired-call-floor-review/hindi-review-b/reviewer-b-order.txt \
  NEW_BALLOT_DIR
```

Pass the two files in that directory to `superturn.hindi_review_audit`
with `--reviewer-b` and `--reviewer-b-windows`. The preparation command
requires all six windows to be marked reviewed and refuses an existing
output directory. The audit checks timing and agreement against the
sealed packet.

## Private Basis listening pilot

Build the fit-only memory-backed packet on the Vast host with
`superturn.basis_stage0 --fit-review-packet` and its pinned split
arguments. From rocksalt, run `bin/vast-push`, then
`bin/vast-basis-review start` and `bin/vast-basis-review tunnel`.
The app listens on port 8769 through the SSH forward. An authorized
reviewer on another machine can open their own forward with
`ssh -N -L 8769:127.0.0.1:8769 rocksalt`, then visit
`http://127.0.0.1:8769/`.

Listen to the six recordings and mark the pauses you hear. Ratings
save to the private packet on the Vast host. The server sets no-store
headers and the app keeps no backup draft in browser storage. Audio and
ratings are in memory-backed storage and disappear when the host restarts.
