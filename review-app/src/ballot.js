export const languages = [
  ['hindi', 'Hindi'], ['urdu', 'Urdu'], ['mixed', 'Mixed'],
  ['other', 'Other'], ['unknown', 'Unknown'],
];
export const sides = [['left', 'Left'], ['right', 'Right']];
export const judgmentDefaults = { outcome: '', brief_ack: '', language: 'hindi', confidence: '' };
export const eventDefaults = {
  focal_channel: 'left', stop_ms: '', next_onset_ms: '', next_channel: '', ...judgmentDefaults,
};
const judgmentFields = ['outcome', 'brief_ack', 'language', 'confidence'];
const eventFields = ['focal_channel', 'stop_ms', 'next_onset_ms', 'next_channel', ...judgmentFields];

function csv(fields, rows) {
  return [fields, ...rows]
    .map(row => row.map(value => `"${String(value ?? '').replaceAll('"', '""')}"`).join(','))
    .join('\r\n') + '\r\n';
}

export function completedWindowsCsv(windows, recordings) {
  return csv(['recording', 'reviewed', 'left_language', 'right_language'], windows.map(id => {
    const { reviewed, left_language, right_language } = recordings[id] ?? {};
    if (reviewed !== true || !left_language || !right_language) {
      throw new Error('Review every recording and record both channel languages before preparing the ballot.');
    }
    return [id, 'yes', left_language, right_language];
  }));
}

export function candidateCsv(ballotIds, ratings) {
  return csv(['example', ...judgmentFields], ballotIds.map(id => {
    const row = ratings[id];
    return row?.outcome ? [id, ...judgmentFields.map(field => row[field])] : [id, '', '', '', ''];
  }));
}

export function continuousCsv(windows, recordings) {
  const rows = [];
  windows.forEach((id, recordingIndex) => {
    const lastStops = { left: [], right: [] };
    (recordings[id]?.events ?? []).forEach((event, index) => {
      const row = { ...eventDefaults, ...event };
      const { outcome, focal_channel: focal, stop_ms: stop, next_onset_ms: onset, next_channel: next } = row;
      const validStop = /^\d+$/.test(stop) && Number(stop) <= 29700;
      const validNext = onset === '' ? !next && !['resume', 'handoff'].includes(outcome)
        : /^\d+$/.test(onset) && Number(onset) >= Number(stop) + 300 && Number(onset) < 30000
          && sides.some(([side]) => side === next);
      if (!outcome || !row.language || !sides.some(([side]) => side === focal)
          || !validStop || !validNext) {
        throw new Error(`Check the outcome and timestamps for recording ${recordingIndex + 1}, pause ${index + 1}.`);
      }
      if (lastStops[focal].some(previous => Math.abs(previous - Number(stop)) < 300)) {
        throw new Error(`Recording ${recordingIndex + 1} has overlapping pauses on the ${focal} channel.`);
      }
      lastStops[focal].push(Number(stop));
      rows.push([id, ...eventFields.map(field => row[field])]);
    });
  });
  return csv(['recording', ...eventFields], rows);
}

export function assistedCsv(windows, proposals, ratings) {
  const recordings = {};
  for (const proposal of proposals) {
    const rating = ratings[proposal.id];
    if (!rating?.outcome) continue;
    const { recording, focal_channel, stop_ms } = proposal;
    const nextOnset = rating.next_onset_ms ?? proposal.next_onset_ms;
    const nextChannel = rating.next_channel ?? proposal.next_channel;
    const corrected = rating.rater_stop_ms;
    const invalid = rating.outcome === 'invalid_pause'
      || (corrected && Math.abs(Number(corrected) - stop_ms) > 100)
      || (nextOnset !== '' && Number(nextOnset) - Math.max(stop_ms, Number(corrected || stop_ms)) < 300);
    const entry = {
      ...rating, focal_channel, stop_ms,
      outcome: invalid ? 'invalid_pause' : rating.outcome,
      next_onset_ms: invalid ? '' : nextOnset,
      next_channel: invalid ? '' : nextChannel,
    };
    (recordings[recording] ??= { events: [] }).events.push(entry);
  }
  return continuousCsv(windows, recordings);
}
