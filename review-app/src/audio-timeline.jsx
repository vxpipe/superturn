import { useEffect, useRef, useState } from 'react';
import './audio-timeline.css';

const percent = (value, duration) => `${Math.min(100, value / duration * 100)}%`;
const formatTime = seconds => {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
};

function WavePeaks({ waveform, focal }) {
  const canvas = useRef(null);
  useEffect(() => {
    const { width, height } = canvas.current;
    const ctx = canvas.current.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    if (!waveform) return;
    for (const [lane, peaks] of waveform.channels.entries()) {
      const center = height * (lane + 0.5) / 2;
      const step = width / peaks.length;
      ctx.fillStyle = focal === ['left', 'right'][lane] ? '#a1d6c0' : '#6c9c91';
      for (let bin = 0; bin < peaks.length; bin++) {
        const amplitude = 1 + Math.sqrt(peaks[bin] / 128) * height * 0.19;
        ctx.fillRect(bin * step, center - amplitude, Math.max(1, step * 0.62), amplitude * 2);
      }
    }
  }, [waveform, focal]);
  return <canvas ref={canvas} width="960" height="176" aria-hidden="true" />;
}

export function AudioTimeline({ recording, durationHint, selected, markers = [], onSelect,
  reveal = false, reviewed = {}, label = 'Recording playback', renderControls }) {
  const audio = useRef(null);
  const edge = selected ? selected.stop_ms / 1000 : 0;
  const start = selected ? Math.max(0, edge - 4) : 0;
  const end = selected ? edge + 3 : null;
  const stopAt = useRef(end);
  const [duration, setDuration] = useState(durationHint);
  const [position, setPosition] = useState(start);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [waveform, setWaveform] = useState(null);
  const [waveError, setWaveError] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const read = new AbortController();
    fetch(`/api/waveform/${recording}`, { signal: read.signal }).then(response => {
      if (!response.ok) throw new Error('Waveform unavailable');
      return response.json();
    }).then(setWaveform).catch(() => {
      if (!read.signal.aborted) setWaveError(true);
    });
    return () => read.abort();
  }, [recording]);

  useEffect(() => {
    if (!ready) return;
    const player = audio.current;
    player.pause();
    stopAt.current = end;
    player.currentTime = start;
    setPosition(start);
  }, [ready, start, end, selected?.id]);

  function seek(time) {
    audio.current.currentTime = time;
    setPosition(time);
  }

  function play() {
    audio.current.play().then(() => setError('')).catch(() => setError('Audio could not play. Try again.'));
  }

  function toggle() {
    if (playing) audio.current.pause();
    else {
      if (audio.current.ended) {
        stopAt.current = end;
        seek(start);
      }
      play();
    }
  }

  function onTimeUpdate() {
    if (stopAt.current !== null && audio.current.currentTime >= stopAt.current) {
      audio.current.pause();
      seek(Math.min(stopAt.current, audio.current.duration));
      stopAt.current = null;
      return;
    }
    setPosition(audio.current.currentTime);
  }

  const visibleMarkers = selected && !onSelect ? [selected] : markers;
  const ticks = duration >= 20 ? [0, 5, 10, 15, 20, 25, duration] : [0, duration / 2, duration];

  return <section className="timeline-player" aria-label={label}>
    <audio ref={audio} preload="metadata" src={`/api/audio/${recording}`}
      onLoadedMetadata={() => {
        setDuration(audio.current.duration);
        setReady(true);
      }}
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
      onEnded={() => { setPlaying(false); stopAt.current = null; }}
      onTimeUpdate={onTimeUpdate} onError={() => setError('This recording could not load. Refresh to try again.')} />
    <div className="timeline-plot">
      {ticks.slice(1, -1).map(time =>
        <span key={time} className="timeline-grid" style={{ left: percent(time, duration) }} aria-hidden="true" />)}
      <WavePeaks waveform={waveform} focal={selected?.focal_channel} />
      <div className="timeline-channels" aria-hidden="true"><span>L</span><span>R</span></div>
      {visibleMarkers.map(marker => {
        const active = marker.id === selected?.id;
        const actualSpan = !active ? reviewed[marker.id]?.first_outcome : reveal;
        const right = (actualSpan && marker.next_onset_ms) || marker.stop_ms + 300;
        const style = { left: percent(marker.stop_ms / 1000, duration),
          width: percent((right - marker.stop_ms) / 1000, duration) };
        return <span key={marker.id} className="timeline-region" data-current={active}
          aria-hidden="true" style={style} />;
      })}
      <span className="timeline-playhead" aria-hidden="true" style={{ left: percent(position, duration) }} />
      <input className="timeline-seek" type="range" min="0" max={duration} step=".01"
        disabled={!ready} value={position} aria-label="Seek recording" aria-valuetext={formatTime(position)}
        onChange={event => { stopAt.current = null; seek(Number(event.target.value)); }} />
      {onSelect && visibleMarkers.map((marker, index) =>
        <button key={marker.id} type="button" className="timeline-marker"
          style={{ left: percent(marker.stop_ms / 1000, duration) }}
          aria-label={`Select pause ${index + 1} at ${(marker.stop_ms / 1000).toFixed(2)} seconds`}
          aria-current={marker.id === selected?.id ? 'true' : undefined} onClick={() => onSelect(marker.id)} />)}
    </div>
    <div className="timeline-ticks" aria-hidden="true">
      {ticks.map(time => <span key={time} style={{ left: percent(time, duration) }}>{formatTime(time)}</span>)}
    </div>
    <div className="timeline-controls">
      <button type="button" className="timeline-play" disabled={!ready} onClick={toggle}>
        <i data-playing={playing} aria-hidden="true" />
        {playing ? 'Pause' : 'Play'}
      </button>
      {selected && <button type="button" disabled={!ready}
        onClick={() => { stopAt.current = end; seek(start); play(); }}>Replay context</button>}
      <span className="timeline-clock">{formatTime(position)} / {formatTime(duration)}</span>
      <label className="timeline-speed">Speed
        <select aria-label="Playback speed" value={speed} onChange={event => {
          const value = Number(event.target.value);
          audio.current.playbackRate = value;
          setSpeed(value);
        }}>
          {[0.75, 1, 1.25, 1.5].map(value => <option key={value} value={value}>{value}×</option>)}
        </select>
      </label>
    </div>
    {renderControls?.({
      ready: ready && !playing,
      capture: () => Math.round(audio.current.currentTime * 1000),
    })}
    {error && <p role="alert" className="timeline-error">{error}</p>}
    {waveError && <p className="timeline-error">Waveform unavailable. Audio controls are still available.</p>}
  </section>;
}
