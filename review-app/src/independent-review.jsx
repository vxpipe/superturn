import { useState } from 'react';
import { AudioTimeline } from './audio-timeline.jsx';
import { continuousCsv, eventDefaults, languages, sides } from './ballot.js';
import { Judgment, SelectField } from './judgment.jsx';
import { useReviewDraft } from './use-review-draft.js';

const defaults = { clips: {}, windows: {}, assisted: {}, pauseRange: {} };
const blankMark = { ...eventDefaults, focal_channel: '', language: '' };
const options = [['no', 'No'], ['yes', 'Yes']];

function MarkEditor({ id, number, duration, events, save, languageOptions, harper }) {
  const initialMark = harper ? { ...blankMark, focal_channel: 'left', language: 'english' } : blankMark;
  const [mark, setMark] = useState(initialMark);
  const [editing, setEditing] = useState(null);
  const [error, setError] = useState('');
  const update = (field, value) => setMark(previous => ({ ...previous, [field]: value }));
  const reset = () => { setMark(initialMark); setEditing(null); setError(''); };
  function saveMark(event) {
    event.preventDefault();
    const next = [...events];
    next[editing ?? next.length] = mark;
    try {
      continuousCsv([id], { [id]: { events: next } }, duration * 1000);
      save(next);
      reset();
    } catch {
      setError('Check the focal channel, language and outcome. Enter times at least 300 ms apart. Marks on the same channel must also be at least 300 ms apart.');
    }
  }

  return <>
    <AudioTimeline recording={id} durationHint={duration} label={`Recording ${number} playback`}
      renderControls={({ ready, capture }) => <div className="timeline-controls">
        <button type="button" disabled={!ready} onClick={() => update('stop_ms', String(capture()))}>Use playhead as stop</button>
        <button type="button" disabled={!ready} onClick={() => update('next_onset_ms', String(capture()))}>Use playhead as next speech</button>
      </div>}
    />
    <form className="manual-mark" onSubmit={saveMark}>
      <h3>{editing === null ? 'Mark a pause' : 'Edit pause'}</h3>
      <p className="instruction">Pause playback and scrub to each boundary. Check and adjust the times below before saving.</p>
      <div className="fields">
        <SelectField label="Speaker who paused" value={mark.focal_channel} options={sides} placeholder="Choose channel"
          onChange={value => update('focal_channel', value)} />
        <label>Stop (ms)
          <input type="number" min="0" max={duration * 1000 - 300} required value={mark.stop_ms}
            onChange={event => update('stop_ms', event.target.value)} />
        </label>
        <label>Next speech (ms)
          <input type="number" min="0" max={duration * 1000 - 1} value={mark.next_onset_ms}
            onChange={event => update('next_onset_ms', event.target.value)} />
        </label>
        <SelectField label="Next speaker" value={mark.next_channel} options={sides} placeholder="Choose if heard"
          onChange={value => update('next_channel', value)} />
      </div>
      <Judgment row={mark} update={update} languageOptions={languageOptions} />
      <p className="instruction">A brief acknowledgment is a short listener response, such as “mm-hm,” while the first speaker keeps the turn.</p>
      <SelectField label="Brief acknowledgment heard" value={mark.brief_ack} options={options}
        placeholder="Leave blank if unsure" onChange={value => update('brief_ack', value)} />
      <div className="actions sheet-bottom">
        <button className="primary" type="submit">Save pause</button>
        {editing !== null && <button type="button" onClick={reset}>Cancel edit</button>}
      </div>
      {error && <p role="alert" className="error">{error}</p>}
    </form>
    {events.length > 0 && <section className="manual-mark" aria-label="Saved pauses">
      <h3>Saved pauses</h3>
      <ol className="manual-marks">
        {events.map((event, index) => <li key={index}>
          <span>{(Number(event.stop_ms) / 1000).toFixed(3)} s · {event.focal_channel} · {event.outcome.replace('_', ' ')}</span>
          <div className="actions">
            <button type="button" onClick={() => { setEditing(index); setMark(event); setError(''); }}>Edit</button>
            <button type="button" onClick={() => {
              save(events.filter((_, i) => i !== index));
              reset();
            }}>Remove</button>
          </div>
        </li>)}
      </ol>
    </section>}
  </>;
}

export function IndependentReview({ packet, serverDraft }) {
  const { draft, save, status, failed, retry, conflict } = useReviewDraft(serverDraft, defaults, true, packet.basis || packet.harper);
  const languageOptions = packet.basis || packet.harper ? [['english', 'English'], ...languages] : languages;
  const channelLanguages = [...languageOptions, ['no_speech', 'No speech']];
  const [index, setIndex] = useState(() => Math.max(0, packet.windows.findIndex(id => !draft.windows[id]?.reviewed)));
  const id = packet.windows[index];
  const window = draft.windows[id] ?? {};
  const reviewed = packet.windows.filter(key => draft.windows[key]?.reviewed).length;
  const update = changes => save({ ...draft, windows: { ...draft.windows, [id]: { ...window, ...changes } } });
  return <main>
    <header className="page-header">
      <h1>{packet.harper ? 'HarperValleyBank pause review' : packet.basis ? 'Conversation pause review' : 'Independent Hindi review'}</h1>
      <p>{packet.harper
        ? 'Listen in stereo. The caller is on the left. Mark caller pauses with at least 300 ms of silence on both channels, then record who speaks next.'
        : 'Listen in stereo. Mark clearly audible stops followed by at least 300 ms of silence on both channels.'}</p>
      <div className="draft-status">
        <span className="storage-note" role="status">{status}</span>
        {failed && <button type="button" onClick={retry}>
          {conflict === null ? 'Retry save' : 'Save this tab’s draft'}
        </button>}
      </div>
    </header>
    <div className="workspace">
      <aside>
        <p className="queue-count">{reviewed} of {packet.windows.length} recordings reviewed</p>
        <nav aria-label="Recordings">
          {packet.windows.map((key, i) => <button type="button" key={key}
            aria-current={i === index ? 'step' : undefined} onClick={() => setIndex(i)}>
            <span>Recording {i + 1}</span>{draft.windows[key]?.reviewed && <span>Reviewed</span>}
          </button>)}
        </nav>
      </aside>
      <section className="sheet" aria-labelledby="recording-title">
        <h2 id="recording-title">Recording {index + 1} <span>of {packet.windows.length}</span></h2>
        <p className="instruction">Listen to the full recording, even if you hear no pauses. If speech never resumes, choose Uncertain and leave the next speech fields blank.</p>
        <MarkEditor key={id} id={id} number={index + 1} duration={packet.duration ?? 30} events={window.events ?? []}
          save={events => update({ events, reviewed: false })} languageOptions={languageOptions} harper={packet.harper} />
        <section className="manual-mark" aria-label="Recording completion">
          <h3>Finish this recording</h3>
          {!packet.harper && <div className="fields">
            <SelectField label="Left channel language" value={window.left_language ?? ''} options={channelLanguages}
              placeholder="Choose language" onChange={left_language => update({ left_language, reviewed: false })} />
            <SelectField label="Right channel language" value={window.right_language ?? ''} options={channelLanguages}
              placeholder="Choose language" onChange={right_language => update({ right_language, reviewed: false })} />
          </div>}
          <div className="actions sheet-bottom">
            <button type="button" className="primary"
              disabled={window.reviewed || (!packet.harper && (!window.left_language || !window.right_language))}
              onClick={() => update({ reviewed: true })}>{window.reviewed ? 'Recording reviewed' : 'Mark recording reviewed'}</button>
            <button type="button" disabled={index + 1 === packet.windows.length} onClick={() => setIndex(index + 1)}>Next recording</button>
          </div>
        </section>
      </section>
    </div>
    {packet.harper
      ? <footer>Audio adapted from <a href="https://github.com/cricketclub/gridspace-stanford-harper-valley">HarperValleyBank</a> (CC BY 4.0).</footer>
      : packet.basis
      ? <footer>Private listening excerpts from Basis Conversations.</footer>
      : <footer>Audio excerpts from <a href="https://asr.iitm.ac.in/">SPRING-INX Hindi R1</a> (CC BY 4.0).</footer>}
  </main>;
}
