import { useId } from 'react';
import { clipOptions, confidenceOptions, languages } from './ballot.js';

const acknowledgment = [['no', 'No'], ['yes', 'Yes']];

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
    Clear rating
  </button>;
  const fields = <div className="fields">
    <SelectField label="Brief acknowledgment" value={row.brief_ack} options={acknowledgment}
      onChange={value => update('brief_ack', value)} />
    <SelectField label="Focal language as heard" value={row.language} options={languages}
      onChange={value => update('language', value)} />
    <SelectField label="Confidence" value={row.confidence} options={confidenceOptions}
      onChange={value => update('confidence', value)} />
  </div>;
  return <>
    <fieldset className="outcome">
      <legend>What happened after the pause?</legend>
      <div className="choices">
        {clipOptions.map(([id, label]) => <label key={id}>
          <input type="radio" name={group} checked={row.outcome === id} onChange={() => update('outcome', id)} />
          <span>{label}</span>
        </label>)}
      </div>
    </fieldset>
    {children}
    {compact ? row.outcome && <details className="annotation-details">
      <summary>Rating details</summary>
      {fields}
      {clear}
    </details> : <>{fields}{clear}</>}
  </>;
}
