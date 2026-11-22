import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { candidateCsv, continuousCsv, judgmentDefaults } from './ballot.js';
import { Judgment } from './judgment.jsx';
import { defaultRange, PauseQueue } from './pause-queue.jsx';
import './review.css';

const storageKey = 'superturn-hindi-review-v2';
const priorStorageKey = 'superturn-hindi-review-v1';

function download(name, contents, type = 'text/csv;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function readDraft() {
  const defaults = { clips: {}, windows: {}, assisted: {}, pauseRange: defaultRange };
  try {
    const saved = localStorage.getItem(storageKey) ?? localStorage.getItem(priorStorageKey) ?? '{}';
    return { ...defaults, ...JSON.parse(saved) };
  } catch {
    return defaults;
  }
}

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
    <audio key={id} controls preload="metadata" aria-label={`Clip ${index + 1} playback`}
      src={`/api/audio/${id}`} />
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

function Reviewer({ packet }) {
  const [draft, setDraft] = useState(readDraft);
  const [view, setView] = useState('assisted');
  const [saveError, setSaveError] = useState('');
  const [legacyError, setLegacyError] = useState('');
  const hasManualDraft = Object.values(draft.windows).some(recording => recording.events?.length);
  const steps = [
    ['assisted', 'Suggested pauses', packet.receipt?.proposals.map(proposal => proposal.id) ?? []],
    ['clips', 'Short clips', packet.clips],
  ];

  function save(next) {
    setDraft(next);
    try {
      localStorage.setItem(storageKey, JSON.stringify(next));
      setSaveError('');
    } catch {
      setSaveError('Browser storage is unavailable. Export your ballots before closing this page.');
    }
  }

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
      <p className="storage-note" role="status">{saveError || 'Drafts are saved in this browser.'}</p>
    </header>
    <div className="workspace">
      <aside>
        <nav aria-label="Review steps">
          {steps.map(([key, label, items]) =>
            <button key={key} aria-current={view === key ? 'step' : undefined} onClick={() => setView(key)}>
              <span>{label}</span><span>{items.filter(id => draft[key][id]?.outcome).length} / {items.length}</span>
            </button>)}
        </nav>
        <details>
          <summary>English practice audio</summary>
          <p>Listen for the caller on the left. Check whether that speaker continues or the person on the right takes over.</p>
          <audio controls preload="none" aria-label="English practice audio" src="/api/audio/practice" />
        </details>
        {hasManualDraft && <details>
          <summary>Earlier manual draft</summary>
          <button type="button" onClick={exportManual}>Export manual CSV</button>
          {legacyError && <p role="alert" className="error">{legacyError}</p>}
        </details>}
      </aside>
      {view === 'assisted'
        ? <PauseQueue packet={packet} draft={draft} save={save} download={download} />
        : <ShortClips packet={packet} draft={draft} save={save} />}
    </div>
    <footer>
      Audio excerpts from <a href="https://asr.iitm.ac.in/">SPRING-INX Hindi R1</a> and{' '}
      <a href="https://github.com/cricketclub/gridspace-stanford-harper-valley">HarperValleyBank</a> (CC BY 4.0).
    </footer>
  </main>;
}

function App() {
  const [packet, setPacket] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    fetch('/api/packet').then(response => {
      if (!response.ok) throw new Error(`Packet request failed (${response.status})`);
      return response.json();
    }).then(setPacket).catch(problem => setError(problem.message));
  }, []);
  if (error) return <main><h1>Review packet unavailable</h1><p role="alert">{error}</p>
    <button onClick={() => location.reload()}>Retry</button></main>;
  return packet ? <Reviewer packet={packet} /> : <main><p role="status">Loading review packet…</p></main>;
}

createRoot(document.getElementById('root')).render(<App />);
