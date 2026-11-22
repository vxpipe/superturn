import { useRef, useState } from 'react';
import { assistedCsv, judgmentDefaults, sides } from './ballot.js';
import { Judgment, SelectField } from './judgment.jsx';

export const defaultRange = { min: '500', max: '5000' };

function NumberField({ label, value, update, min = 0, max = 30000, step = 1, placeholder }) {
  return <label>{label}
    <input type="number" inputMode="numeric" min={min} max={max} step={step}
      value={value} placeholder={placeholder} onChange={event => update(event.target.value)} />
  </label>;
}

function PauseAudio({ proposal }) {
  const player = useRef(null);
  const stopAt = useRef(null);
  const [error, setError] = useState('');
  const position = proposal.stop_ms / 1000;
  const start = Math.max(0, position - 4);

  function play(near) {
    const audio = player.current;
    stopAt.current = near ? position + 3 : null;
    if (near) audio.currentTime = start;
    audio.play().then(() => setError('')).catch(problem => setError(problem.message));
  }

  return <div className="pause-audio">
    <audio ref={player} controls preload="metadata" src={`/api/audio/${proposal.recording}`}
      aria-label="Recording playback"
      onLoadedMetadata={() => { player.current.currentTime = start; }}
      onTimeUpdate={() => {
        if (stopAt.current !== null && player.current.currentTime >= stopAt.current) {
          player.current.pause();
          stopAt.current = null;
        }
      }} />
    <div className="actions">
      <button type="button" className="primary" onClick={() => play(true)}>Play around pause</button>
      <button type="button" onClick={() => play(false)}>Continue playing</button>
    </div>
    {error && <p role="alert" className="error">Playback failed. {error}</p>}
  </div>;
}

export function PauseQueue({ packet, draft, save, download }) {
  const { receipt, windows } = packet;
  const [marked, setMarked] = useState('');
  const [error, setError] = useState('');
  if (!receipt) return <section className="sheet">
    <h2>Suggested pauses</h2>
    <p>The pause manifest is missing. The short clips are ready to review.</p>
  </section>;

  const { assisted: ratings, pauseRange: range } = draft;
  const min = Number(range.min);
  const max = Number(range.max);
  const valid = Number.isInteger(min) && Number.isInteger(max) && min >= 300 && max <= 30000 && max >= min;
  const visible = valid ? receipt.proposals.filter(proposal => {
    const gap = proposal.next_onset_ms - proposal.stop_ms;
    return gap >= min && gap <= max;
  }) : [];
  const rated = visible.filter(proposal => ratings[proposal.id]?.outcome).length;
  const proposal = visible.find(pause => pause.id === marked)
    ?? visible.find(pause => !ratings[pause.id]?.outcome) ?? visible[0];
  const index = visible.indexOf(proposal);
  const row = { ...judgmentDefaults, ...ratings[proposal?.id] };
  const hasRatings = receipt.proposals.some(pause => ratings[pause.id]?.outcome);

  function update(field, value) {
    const updated = { ...row, [field]: value };
    if (field === 'outcome' && value && !row.first_outcome) {
      updated.first_outcome = value;
      updated.committed_at = new Date().toISOString();
    }
    setMarked(proposal.id);
    save({ ...draft, assisted: { ...ratings, [proposal.id]: updated } });
  }

  function exportBallots() {
    try {
      const csv = assistedCsv(windows, receipt.proposals, ratings);
      download('hindi-assisted-pauses.csv', csv);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }
  function exportAudit() {
    download('hindi-assisted-audit.json', JSON.stringify({ ...receipt, filter: range, ratings }, null, 2),
      'application/json');
  }

  return <section className="sheet" aria-labelledby="item-title">
    <div className="sheet-top">
      <h2 id="item-title">Suggested pauses</h2>
      <span className="queue-count" role="status">{rated} of {visible.length} reviewed</span>
    </div>
    <div className="fields filter">
      <NumberField label="Min pause (ms)" min={300} value={range.min}
        update={min => save({ ...draft, pauseRange: { ...range, min } })} />
      <NumberField label="Max pause (ms)" min={300} value={range.max}
        update={max => save({ ...draft, pauseRange: { ...range, max } })} />
      <button type="button" onClick={() => save({ ...draft, pauseRange: defaultRange })}>Reset range</button>
    </div>
    {!valid && <p role="alert" className="error">Use a minimum of at least 300 ms and a maximum between that value and 30,000 ms.</p>}
    {valid && !visible.length && <p className="instruction">No proposals in this range. Widen the range to see more.</p>}
    {proposal && <>
      <div className="sheet-top proposal-title">
        <h3>Recording {windows.indexOf(proposal.recording) + 1}
          {' · '}{(proposal.stop_ms / 1000).toFixed(2)} s · {proposal.focal_channel === 'left' ? 'Left' : 'Right'} speaker
        </h3>
        <label className="jump">Jump to pause
          <select value={proposal.id} onChange={event => setMarked(event.target.value)}>
            {visible.map((pause, i) => <option key={pause.id} value={pause.id}>
              {i + 1}{ratings[pause.id]?.outcome ? ' · reviewed' : ''}
            </option>)}
          </select>
        </label>
      </div>
      <p className="instruction">Listen for whether that speaker continues or the other speaker takes over.
        A brief acknowledgment followed by the original speaker continuing is a resume.</p>
      <PauseAudio key={proposal.id} proposal={proposal} />
      <Judgment row={row} update={update} compact>
        <div className="actions">
          <button disabled={index === 0} onClick={() => setMarked(visible[index - 1].id)}>Previous</button>
          <button className="primary" disabled={index + 1 === visible.length}
            onClick={() => setMarked(visible[index + 1].id)}>Next pause</button>
        </div>
      </Judgment>
      {row.outcome && <details key={proposal.id} className="annotation-details">
        <summary>Review or correct detector timing</summary>
        <p className="instruction">The scoring stop stays at {proposal.stop_ms} ms. A stop correction
          over 100 ms or a corrected gap under 300 ms marks the item invalid on export.</p>
        <div className="fields">
          <NumberField label="Stop as heard (ms)" value={row.rater_stop_ms ?? ''} placeholder={proposal.stop_ms}
            update={value => update('rater_stop_ms', value)} />
          <NumberField label="Next speech (ms)" value={row.next_onset_ms ?? proposal.next_onset_ms}
            update={value => update('next_onset_ms', value)} />
          <SelectField label="Next speaker" value={row.next_channel ?? proposal.next_channel}
            options={sides} placeholder="Not heard" onChange={value => update('next_channel', value)} />
        </div>
      </details>}
    </>}
    <div className="actions sheet-bottom">
      <button disabled={!hasRatings} onClick={exportBallots}>Export assisted CSV</button>
      <button disabled={!hasRatings} onClick={exportAudit}>Export audit JSON</button>
    </div>
    {error && <p role="alert" className="error">{error}</p>}
  </section>;
}
