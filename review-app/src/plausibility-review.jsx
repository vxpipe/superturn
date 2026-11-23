import { useState } from 'react';
import { useReviewDraft } from './use-review-draft.js';

const defaults = { clips: {}, windows: {}, assisted: {}, pauseRange: {} };
const choices = [
  ['incomplete', 'Clearly incomplete'],
  ['could_end', 'Either stopping or continuing is plausible'],
  ['complete', 'Clearly complete'],
  ['unrateable', 'Cannot assess this text'],
];

export function PlausibilityReview({ packet, serverDraft }) {
  const { draft, save, status, failed, retry, conflict } = useReviewDraft(serverDraft, defaults, true, true);
  const items = packet.items;
  const nextUnread = items.findIndex(item => !draft.clips[item.id]?.outcome);
  const [index, setIndex] = useState(Math.max(0, nextUnread));
  const item = items[index];
  const reviewed = items.filter(row => draft.clips[row.id]?.outcome).length;
  function choose(outcome) {
    save({ ...draft, clips: { ...draft.clips, [item.id]: { outcome } } });
    if (index + 1 < items.length) setIndex(index + 1);
  }
  return <main>
    <header className="page-header">
      <h1>English transcript review</h1>
      <p>Given these words, could the caller reasonably finish here?</p>
      <div className="draft-status">
        <span className="storage-note" role="status">{status}</span>
        {failed && <button type="button" onClick={retry}>
          {conflict === null ? 'Retry save' : 'Save this tab’s draft'}
        </button>}
      </div>
    </header>
    <section className="sheet text-review" aria-labelledby="plausibility-title">
      <div className="sheet-top">
        <h2 id="plausibility-title">Item {index + 1} <span>of {items.length}</span></h2>
        <span role="status">{reviewed} reviewed</span>
      </div>
      <dl className="transcript-turns">
        <dt>Previous speaker</dt>
        <dd>{item.previous === '---' ? 'No previous turn' : item.previous}</dd>
        <dt>Caller</dt>
        <dd>{item.current}</dd>
      </dl>
      <div className="actions" aria-label="Could the caller reasonably finish here?">
        {choices.map(([key, label]) => <button key={key} type="button"
          aria-pressed={draft.clips[item.id]?.outcome === key} onClick={() => choose(key)}>{label}</button>)}
      </div>
      <div className="actions sheet-bottom">
        <button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</button>
        <button type="button" disabled={index + 1 === items.length} onClick={() => setIndex(index + 1)}>Next</button>
        {nextUnread >= 0 && <button type="button" onClick={() => setIndex(nextUnread)}>Next unrated</button>}
      </div>
    </section>
  </main>;
}
