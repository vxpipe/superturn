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

Open `http://127.0.0.1:8765/`. The app keeps drafts in the current
browser, with the same storage key and default address as the original
static Hindi reviewer. Use the export controls to save the two CSV
ballots to disk. The candidate ballot includes the full set of template
IDs, with blank rows for clips outside this pilot or left unrated.

Run `npm run dev` for React hot reload at `http://127.0.0.1:5173/` and
the local Fastify API on port 8766. Browsers store drafts separately
for each origin. Use the production address for the actual review.

Set `REVIEW_BUNDLE` to an absolute path if the copied
`paired-call-floor-review` directory lives elsewhere. Keep both the
`hindi-review-a` packet and `packet-v1/listening/reviewer-a.csv` in that
directory. The app serves the review manifest and allowlisted audio IDs
over loopback.
