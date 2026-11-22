import { useId } from 'react';
import { languages } from './ballot.js';

const options = [
  ['resume', 'Resume / hold', 'The speaker who paused comes back and continues their thought.'],
  ['handoff', 'Handoff / shift', 'The first speaker finishes and the other takes a turn. A short answer can count.'],
  ['interruption', 'Interruption', 'There’s a real pause, but the other speaker cuts into an unfinished thought.'],
  ['uncertain', 'Uncertain', 'You can’t tell what happened. For example, the audio is unclear or ends too soon.'],
  ['invalid_pause', 'Invalid pause', 'The marked speaker didn’t stop near the marker, or either speaker talks in the first 300 ms.'],
];

export function SelectField({ label, value, options, onChange, placeholder }) {
  return <label>{label}
    <select value={value} onChange={event => onChange(event.target.value)}>
      {placeholder && <option value="">{placeholder}</option>}
      {options.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
    </select>
  </label>;
}

export function Judgment({ row, update, compact, children }) {
  const group = useId();
  const clear = row.outcome && <button className="clear-rating" type="button" onClick={() => update('outcome', '')}>
    Clear selection
  </button>;
  const fields = <div className="fields">
    <SelectField label="Focal language as heard" value={row.language} options={languages}
      onChange={value => update('language', value)} />
  </div>;
  return <>
    <fieldset className="outcome">
      <legend>What happened after the pause?</legend>
      <div className="choices">
        {options.map(([id, label, description]) => {
          const titleId = `${group}-${id}`;
          const hintId = `${titleId}-help`;
          return <label key={id}>
            <input type="radio" name={group} checked={row.outcome === id}
              aria-labelledby={titleId} aria-describedby={hintId} onChange={() => update('outcome', id)} />
            <span>
              <span id={titleId} className="choice-title">{label}</span>
              <span id={hintId} className="choice-help">{description}</span>
            </span>
          </label>;
        })}
      </div>
    </fieldset>
    {children}
    {compact ? row.outcome && <details className="annotation-details">
      <summary>Language as heard</summary>
      {fields}
      {clear}
    </details> : <>{fields}{clear}</>}
  </>;
}
