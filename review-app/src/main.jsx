import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AudioTimeline } from './audio-timeline.jsx';
import { candidateCsv, continuousCsv, judgmentDefaults } from './ballot.js';
import { Judgment } from './judgment.jsx';
import { defaultRange, getPauseQueue, PauseQueue } from './pause-queue.jsx';
import { useReviewDraft } from './use-review-draft.js';
import './review.css';

function download(name, contents, type = 'text/csv;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const defaults = { clips: {}, windows: {}, assisted: {}, pauseRange: defaultRange };

function ShortClips({ packet, draft, save }) {
  const [index, setIndex] = useState(() => Math.max(0, packet.clips.findIndex(id => !draft.clips[id]?.outcome)));
  const id = packet.clips[index];
  const clip = { ...judgmentDefaults, ...draft.clips[id] };
  const update = (field, value) => save({
    ...draft, clips: { ...draft.clips, [id]: { ...clip, [field]: value } },
  });
  return <section className="sheet" aria-labelledby="item-title">
    <div className="sheet-top">
      <h2 id="item-title">Clip {index + 1} <span>of {packet.clips.length}</span></h2>
      <label className="jump">Jump to
        <select value={index} onChange={event => setIndex(Number(event.target.value))}>
          {packet.clips.map((item, i) => <option value={i} key={item}>
            {i + 1}{draft.clips[item]?.outcome ? ' · reviewed' : ''}
          </option>)}
        </select>
      </label>
    </div>
    <p className="instruction">The focal speaker is on the left. Choose Invalid pause if they did not stop near
      3.000 s, or if either channel has speech between 3.000 and 3.300 s.</p>
    <AudioTimeline key={id} recording={id} durationHint={6} label={`Clip ${index + 1} playback`}
      selected={{ id, stop_ms: 3000, focal_channel: 'left' }} reveal={Boolean(clip.outcome)} />
    <Judgment row={clip} update={update} />
    <div className="actions sheet-bottom">
      <button disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</button>
      <button className="primary" disabled={index + 1 === packet.clips.length}
        onClick={() => setIndex(index + 1)}>Next clip</button>
      <button className="export" onClick={() => download('hindi-short-clips.csv', candidateCsv(packet.ballotIds, draft.clips))}>
        Export short-clip CSV
      </button>
    </div>
  </section>;
}

function Reviewer({ packet, serverDraft }) {
  const { draft, save, status, failed, retry, conflict } = useReviewDraft(serverDraft, defaults);
  const [view, setView] = useState('assisted');
  const [practiceOpen, setPracticeOpen] = useState(false);
  const [legacyError, setLegacyError] = useState('');
  const hasManualDraft = Object.values(draft.windows).some(recording => recording.events?.length);
  const queue = getPauseQueue(packet.receipt?.proposals ?? [], draft.pauseRange, draft.assisted);
  const steps = [
    ['assisted', 'Suggested pauses', queue.rated, queue.visible.length],
    ['clips', 'Short clips', packet.clips.filter(id => draft.clips[id]?.outcome).length, packet.clips.length],
  ];

  function exportManual() {
    try {
      download('hindi-continuous-manual.csv', continuousCsv(packet.windows, draft.windows));
      setLegacyError('');
    } catch (problem) {
      setLegacyError(problem.message);
    }
  }

  return <main>
    <header className="page-header">
      <h1>Hindi pause review</h1>
      <p>Use stereo headphones to hear which speaker continues after each pause.</p>
      <div className="draft-status">
        <span className="storage-note" role="status">{status}</span>
        {failed && <button type="button" onClick={retry}>
          {conflict === null ? 'Retry save' : 'Save this tab’s draft'}
        </button>}
      </div>
    </header>
    <div className="workspace">
      <aside>
        <nav aria-label="Review steps">
          {steps.map(([key, label, reviewed, total]) =>
            <button key={key} aria-current={view === key ? 'step' : undefined} onClick={() => setView(key)}>
              <span>{label}</span><span>{reviewed} / {total}</span>
            </button>)}
        </nav>
        <details onToggle={event => setPracticeOpen(event.currentTarget.open)}>
          <summary>English practice audio</summary>
          <p>Listen for the caller on the left. Check whether that speaker continues or the person on the right takes over.</p>
          {practiceOpen && <AudioTimeline recording="practice" durationHint={30} label="English practice audio" />}
        </details>
        {hasManualDraft && <details>
          <summary>Earlier manual draft</summary>
          <button type="button" onClick={exportManual}>Export manual CSV</button>
          {legacyError && <p role="alert" className="error">{legacyError}</p>}
        </details>}
      </aside>
      {view === 'assisted'
        ? <PauseQueue packet={packet} queue={queue} draft={draft} save={save} download={download} />
        : <ShortClips packet={packet} draft={draft} save={save} />}
    </div>
    <footer>
      Audio excerpts from <a href="https://asr.iitm.ac.in/">SPRING-INX Hindi R1</a> and{' '}
      <a href="https://github.com/cricketclub/gridspace-stanford-harper-valley">HarperValleyBank</a> (CC BY 4.0).
    </footer>
  </main>;
}

function App() {
  const [review, setReview] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const read = path => fetch(path).then(response => {
      if (!response.ok) throw new Error(`Review request failed (${response.status})`);
      return response.json();
    });
    Promise.all([read('/api/packet'), read('/api/draft')])
      .then(([packet, serverDraft]) => setReview({ packet, serverDraft }))
      .catch(problem => setError(problem.message));
  }, []);
  if (error) return <main><h1>Review packet unavailable</h1><p role="alert">{error}</p>
    <button onClick={() => location.reload()}>Retry</button></main>;
  return review ? <Reviewer {...review} /> : <main><p role="status">Loading review packet…</p></main>;
}

createRoot(document.getElementById('root')).render(<App />);
