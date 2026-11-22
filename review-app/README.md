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
configured tailnet. Use the same URL to keep your browser drafts. The
app reads earlier drafts and offers a separate export for manual marks.

In **Suggested pauses**, set the minimum and maximum detected silence
in milliseconds, then play around a pause and select the outcome you
heard. The filter only changes which items are displayed. Exports
include all saved ratings, including ones outside the current filter.
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
