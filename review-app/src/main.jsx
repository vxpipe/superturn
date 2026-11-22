import { useEffect, useId, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  candidateCsv, clipOptions, confidenceOptions, continuousCsv, eventDefaults,
  judgmentDefaults, languages, outcomeOptions, sides,
} from './ballot.js';
import './review.css';

const storageKey = 'superturn-hindi-review-v1';
const setField = (row, field, value) => ({ ...row, [field]: value });
const acknowledgment = [['no', 'No'], ['yes', 'Yes']];

function SelectField({ label, value, options, onChange, placeholder }) {
  return <label>{label}
    <select value={value} onChange={event => onChange(event.target.value)}>
      {placeholder && <option value="">{placeholder}</option>}
      {options.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
    </select>
  </label>;
}

function Judgment({ row, update, clip }) {
  const group = useId();
  const options = clip ? clipOptions : outcomeOptions;
  return <>
    <fieldset className="outcome">
      <legend>What happened after the pause?</legend>
      <div className="choices">
        {options.map(([id, label]) => <label key={id}>
          <input type="radio" name={group} checked={row.outcome === id} onChange={() => update('outcome', id)} />
          <span>{label}</span>
        </label>)}
      </div>
    </fieldset>
    <div className="fields">
      <SelectField label="Brief acknowledgment" value={row.brief_ack} options={acknowledgment}
        onChange={value => update('brief_ack', value)} />
      <SelectField label="Focal language as heard" value={row.language} options={languages}
        onChange={value => update('language', value)} />
      <SelectField label="Confidence" value={row.confidence} options={confidenceOptions}
        onChange={value => update('confidence', value)} />
      {clip && row.outcome && <button type="button" onClick={() => update('outcome', '')}>Clear rating</button>}
    </div>
  </>;
}

function Timestamp({ label, value, onChange, playbackTime }) {
  return <div className="timestamp">
    <label>{label}
      <input type="number" inputMode="numeric" min="0" max="30000" placeholder="ms" value={value}
        onChange={event => onChange(event.target.value)} />
    </label>
    <button type="button" onClick={() => onChange(String(playbackTime()))}>Use playback time</button>
  </div>;
}

function WindowReview({ windowState, update, audioRef }) {
  function edit(index, field, value) {
    update({ ...windowState, events: windowState.events.map((row, i) => (
      i === index ? setField({ ...eventDefaults, ...row }, field, value) : row
    )) });
  }
  const playbackTime = () => Math.round(audioRef.current.currentTime * 1000);
  return <>
    {windowState.events.map((event, index) => {
      const row = { ...eventDefaults, ...event };
      const change = field => value => edit(index, field, value);
      return <section className="event" key={index} aria-label={`Pause ${index + 1}`}>
        <div className="event-title">
          <h3>Pause {index + 1}</h3>
          <button type="button"
            onClick={() => update({ ...windowState, events: windowState.events.filter((_, i) => i !== index) })}>
            Remove pause
          </button>
        </div>
        <div className="fields">
          <SelectField label="Speaker who paused" value={row.focal_channel} options={sides}
            onChange={change('focal_channel')} />
          <SelectField label="Next speaker" value={row.next_channel} options={sides}
            placeholder="Choose if heard" onChange={change('next_channel')} />
        </div>
        <div className="fields">
          <Timestamp label="Stop (ms)" value={row.stop_ms} onChange={change('stop_ms')} playbackTime={playbackTime} />
          <Timestamp label="Next speech (ms)" value={row.next_onset_ms}
            onChange={change('next_onset_ms')} playbackTime={playbackTime} />
        </div>
        <Judgment row={row} update={(field, value) => edit(index, field, value)} />
      </section>;
    })}
    <div className="actions window-actions">
      <button type="button" onClick={() => update({ ...windowState, events: [...windowState.events, eventDefaults] })}>
        Add a pause
      </button>
      <label className="inline">
        <input type="checkbox" checked={windowState.done}
          onChange={event => update({ ...windowState, done: event.target.checked })} />
        Fully listened
      </label>
    </div>
  </>;
}

function download(name, contents) {
  const url = URL.createObjectURL(new Blob([contents], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function readDraft() {
  try {
    const { clips, windows } = JSON.parse(localStorage.getItem(storageKey) ?? '{}');
    return { clips: clips ?? {}, windows: windows ?? {} };
  } catch {
    return { clips: {}, windows: {} };
  }
}

function Reviewer({ packet }) {
  const [draft, setDraft] = useState(readDraft);
  const [view, setView] = useState('windows');
  const [positions, setPositions] = useState(() => ({
    windows: Math.max(0, packet.windows.findIndex(id => !draft.windows[id]?.done)),
    clips: Math.max(0, packet.clips.findIndex(id => !draft.clips[id]?.outcome)),
  }));
  const [error, setError] = useState('');
  const [saveError, setSaveError] = useState('');
  const audioRef = useRef(null);
  const isClip = view === 'clips';
  const index = positions[view];
  const id = packet[view][index];
  const clip = { ...judgmentDefaults, ...draft.clips[id] };
  const windowState = { done: false, events: [], ...draft.windows[id] };
  const counts = {
    windows: packet.windows.filter(item => draft.windows[item]?.done).length,
    clips: packet.clips.filter(item => draft.clips[item]?.outcome).length,
  };

  function update(row) {
    const next = { ...draft, [view]: { ...draft[view], [id]: row } };
    setDraft(next);
    try {
      localStorage.setItem(storageKey, JSON.stringify(next));
      setSaveError('');
    } catch {
      setSaveError('Browser storage is unavailable. Export your ballots before closing this page.');
    }
  }
  function jump(destination) {
    setPositions(previous => ({ ...previous, [view]: destination }));
  }
  function exportBallot() {
    try {
      if (isClip) download('hindi-short-clips.csv', candidateCsv(packet.ballotIds, draft.clips));
      else download('hindi-continuous.csv', continuousCsv(packet.windows, draft.windows));
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }

  return <main>
    <header className="page-header">
      <h1>Hindi pause review</h1>
      <p>Use stereo headphones. Review the continuous recordings first, then rate the short clips.</p>
      <p className="storage-note" role="status">{saveError || 'Drafts are saved in this browser.'}</p>
    </header>
    <div className="workspace">
      <aside>
        <nav aria-label="Review steps">
          {[['windows', 'Continuous recordings'], ['clips', 'Short clips']].map(([key, label]) =>
            <button key={key} aria-current={view === key ? 'step' : undefined} onClick={() => {
              setView(key);
              setError('');
            }}>
              <span>{label}</span><span> {counts[key]} / {packet[key].length}</span>
            </button>)}
        </nav>
        <details>
          <summary>English practice audio</summary>
          <p>Listen for the caller on the left. Check whether that speaker continues or the person on the right takes over.</p>
          <audio controls preload="none" aria-label="English practice audio" src="/api/audio/practice" />
        </details>
      </aside>
      <section className="sheet" aria-labelledby="item-title">
        <div className="sheet-top">
          <h2 id="item-title">{isClip ? 'Clip' : 'Recording'} {index + 1} <span>of {packet[view].length}</span></h2>
          <label className="jump">Jump to
            <select value={index} onChange={event => jump(Number(event.target.value))}>
              {packet[view].map((item, i) => <option value={i} key={item}>
                {i + 1}{(isClip ? draft.clips[item]?.outcome : draft.windows[item]?.done) ? ' · reviewed' : ''}
              </option>)}
            </select>
          </label>
        </div>
        <p className="instruction">{isClip
          ? 'The focal speaker is on the left. Choose Invalid pause if they did not stop near 3.000 s, or if either channel has speech between 3.000 and 3.300 s.'
          : 'Mark stops followed by at least 300 ms of quiet on both channels. Enter times from the start of the recording. A brief acknowledgment followed by the original speaker continuing counts as a resume.'}</p>
        <audio key={id} ref={audioRef} controls preload="metadata"
          aria-label={`${isClip ? 'Clip' : 'Recording'} ${index + 1} playback`} src={`/api/audio/${id}`} />
        {isClip
          ? <Judgment clip row={clip} update={(field, value) => update(setField(clip, field, value))} />
          : <WindowReview windowState={windowState} update={update} audioRef={audioRef} />}
        <div className="actions sheet-bottom">
          <button disabled={index === 0} onClick={() => jump(index - 1)}>Previous</button>
          <button className="primary" disabled={index === packet[view].length - 1}
            onClick={() => jump(index + 1)}>Next {isClip ? 'clip' : 'recording'}</button>
          <button className="export" onClick={exportBallot}>Export {isClip ? 'short-clip' : 'continuous'} CSV</button>
        </div>
        {error && <p role="alert" className="error">{error}</p>}
      </section>
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
