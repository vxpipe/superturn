import { useEffect, useRef, useState } from 'react';

const draftKey = 'superturn-hindi-review-v2';
const previousKey = 'superturn-hindi-review-v1';
const pendingKey = 'superturn-hindi-review-pending';

function browserDraft(key) {
  try {
    return JSON.parse(localStorage.getItem(key));
  } catch {
    return null;
  }
}

export function useReviewDraft(server, defaults) {
  const [initial] = useState(() => {
    const pending = browserDraft(pendingKey);
    const local = browserDraft(draftKey) ?? browserDraft(previousKey);
    return {
      draft: { ...defaults, ...(pending?.draft ?? server.draft ?? local) },
      pending: Boolean(pending || (!server.draft && local)),
      revision: pending?.revision ?? server.revision,
    };
  });
  const [draft, setDraft] = useState(initial.draft);
  const [status, setStatus] = useState(server.draft ? 'Saved to rocksalt' : 'Ratings save to rocksalt automatically');
  const [failed, setFailed] = useState(false);
  const revision = useRef(initial.revision);
  const latest = useRef(initial.draft);
  const waiting = useRef(null);
  const saving = useRef(false);
  const [conflict, setConflict] = useState(null);

  function storeBrowser() {
    try {
      localStorage.setItem(draftKey, JSON.stringify(latest.current));
      localStorage.setItem(pendingKey, JSON.stringify({ draft: latest.current, revision: revision.current }));
    } catch {
      // Server persistence remains available when browser storage is blocked.
    }
  }

  async function flush() {
    if (saving.current || !waiting.current) return;
    saving.current = true;
    setFailed(false);
    setConflict(null);
    setStatus('Saving to rocksalt…');
    try {
      while (waiting.current) {
        const next = waiting.current;
        waiting.current = null;
        const response = await fetch('/api/draft', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ revision: revision.current, draft: next }),
        });
        if (response.status === 409) {
          const read = await fetch('/api/draft');
          if (!read.ok) throw new Error('Could not verify the saved draft');
          const current = await read.json();
          if (JSON.stringify(current.draft) !== JSON.stringify(next)) {
            setConflict(current.revision);
            throw new Error('Another tab saved a newer draft. This tab still has your changes.');
          }
          revision.current = current.revision;
        } else {
          if (!response.ok) throw new Error('Server save failed. Keep this tab open and retry.');
          ({ revision: revision.current } = await response.json());
        }
        if (waiting.current) storeBrowser();
      }
      try { localStorage.removeItem(pendingKey); } catch { /* Disk save already succeeded. */ }
      setStatus('Saved to rocksalt');
    } catch (error) {
      waiting.current = latest.current;
      setFailed(true);
      setStatus(error.message);
    } finally {
      saving.current = false;
    }
  }

  function save(next) {
    latest.current = waiting.current = next;
    setDraft(next);
    storeBrowser();
    if (conflict === null) void flush();
  }

  useEffect(() => {
    if (initial.pending) save(initial.draft);
  }, [initial]);

  function retry() {
    if (conflict !== null) {
      revision.current = conflict;
      storeBrowser();
    }
    void flush();
  }

  return { draft, save, status, failed, retry, conflict };
}
