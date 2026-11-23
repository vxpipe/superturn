import { mkdir, mkdtemp, readFile, rename, rm, rmdir, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { completedWindowsCsv, continuousCsv } from './src/ballot.js';

async function prepare([draftPath, orderPath, destination]) {
  if (!draftPath || !orderPath || !destination) {
    throw new Error('Usage: npm run prepare-independent -- SAVED_DRAFT.json LISTENING_ORDER.txt NEW_OUTPUT_DIR');
  }
  const windows = (await readFile(orderPath, 'utf8')).trim().split(/\r?\n/);
  const { revision, draft } = JSON.parse(await readFile(draftPath, 'utf8'));
  if (windows.length !== 6 || new Set(windows).size !== windows.length
      || !windows.every(id => /^call-[a-f0-9]{16}$/.test(id))
      || !Number.isInteger(revision) || revision < 1
      || !draft?.windows || Object.keys(draft.windows).some(id => !windows.includes(id))) {
    throw new Error('The saved draft does not match the independent Hindi packet.');
  }
  if (Object.values(draft.windows).some(window => window.events?.some(mark => !mark.language || !mark.focal_channel))) {
    throw new Error('Every pause needs an explicitly chosen focal channel and language.');
  }
  const events = continuousCsv(windows, draft.windows);
  const completed = completedWindowsCsv(windows, draft.windows);
  const output = resolve(destination);
  await mkdir(output, { mode: 0o700 });
  let temporary;
  try {
    temporary = await mkdtemp(join(dirname(output), `.${basename(output)}-`));
    await Promise.all([
      writeFile(join(temporary, 'reviewer-b.csv'), events, { mode: 0o600 }),
      writeFile(join(temporary, 'reviewer-b-windows.csv'), completed, { mode: 0o600 }),
    ]);
    await rename(temporary, output);
  } catch (error) {
    await rmdir(output);
    throw error;
  } finally {
    if (temporary) await rm(temporary, { recursive: true, force: true });
  }
  console.log(`Prepared both ballots from saved revision ${revision}.`);
}

prepare(process.argv.slice(2)).catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
