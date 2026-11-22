import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const cookie = 'superturn_reviewer';
const reviewerId = request => request.headers.cookie?.match(
  /(?:^|;\s*)superturn_reviewer=([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?=;|$)/,
)?.[1];

export function registerDraftStore(server, directory) {
  const writes = new Map();
  const filename = id => join(directory, `${id}.json`);
  const read = id => readFile(filename(id), 'utf8').then(JSON.parse).catch(error => {
    if (error.code === 'ENOENT') return { revision: 0, draft: null };
    throw error;
  });

  server.get('/api/draft', async (request, reply) => {
    let id = reviewerId(request);
    if (!id) {
      id = randomUUID();
      reply.header('Set-Cookie', `${cookie}=${id}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict`);
    }
    reply.header('Cache-Control', 'no-store');
    return read(id);
  });

  server.put('/api/draft', {
    bodyLimit: 256 * 1024,
    schema: {
      body: {
        type: 'object', additionalProperties: false, required: ['revision', 'draft'],
        properties: {
          revision: { type: 'integer', minimum: 0 },
          draft: {
            type: 'object', additionalProperties: false, required: ['clips', 'windows', 'assisted', 'pauseRange'],
            properties: Object.fromEntries(['clips', 'windows', 'assisted', 'pauseRange']
              .map(key => [key, { type: 'object' }])),
          },
        },
      },
    },
  }, async (request, reply) => {
    const id = reviewerId(request);
    if (!id) return reply.code(400).send({ error: 'Reload to initialize the reviewer session' });
    const { revision, draft } = request.body;
    const pending = (writes.get(id) ?? Promise.resolve()).catch(() => {}).then(async () => {
      const current = await read(id);
      if (revision !== current.revision) {
        return reply.code(409).send({ error: 'Another tab saved a newer draft' });
      }
      const record = { revision: revision + 1, updatedAt: new Date().toISOString(), draft };
      await mkdir(directory, { recursive: true, mode: 0o700 });
      const temporary = join(directory, `.${randomUUID()}.tmp`);
      try {
        await writeFile(temporary, JSON.stringify(record, null, 2) + '\n', { mode: 0o600 });
        await rename(temporary, filename(id));
      } finally {
        await rm(temporary, { force: true });
      }
      return { revision: record.revision };
    });
    writes.set(id, pending);
    try {
      return await pending;
    } finally {
      if (writes.get(id) === pending) writes.delete(id);
    }
  });
}
