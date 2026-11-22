import { access, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';

const appDir = fileURLToPath(new URL('.', import.meta.url));
const bundle = resolve(process.env.REVIEW_BUNDLE ?? join(appDir, '../data/review/paired-call-floor-review'));
const audioDir = join(bundle, 'hindi-review-a/audio');
const dev = process.argv.includes('--dev');
const defaults = { dev: 8766, production: 8765 };
const server = Fastify();
const valid = (id, prefix) => new RegExp(`^${prefix}-[a-f0-9]{16}$`).test(id);

async function start() {
  const [manifest, template] = await Promise.all([
    readFile(join(bundle, 'hindi-review-a/packet.json'), 'utf8'),
    readFile(join(bundle, 'packet-v1/listening/reviewer-a.csv'), 'utf8'),
  ]);
  const { clips, windows } = JSON.parse(manifest);
  const [header, ...lines] = template.trimEnd().split(/\r?\n/);
  const ballotIds = lines.map(line => line.split(',')[0]);
  if (header !== 'example,outcome,brief_ack,language,confidence'
      || !ballotIds.every(id => valid(id, 'sample'))
      || new Set(ballotIds).size !== ballotIds.length
      || !clips.length || !windows.length
      || !clips.every(id => ballotIds.includes(id))
      || !windows.every(id => valid(id, 'call'))) {
    throw new Error('Unexpected review packet or candidate ballot template');
  }
  const ids = new Set([...clips, ...windows, 'practice']);
  if (ids.size !== clips.length + windows.length + 1) {
    throw new Error('Duplicate review audio IDs');
  }
  await Promise.all([...ids].map(id => access(join(audioDir, `${id}.wav`))));
  const receipt = await readFile(join(bundle, 'hindi-review-a/suggested-pauses.json'), 'utf8')
    .then(JSON.parse)
    .catch(error => { if (error.code !== 'ENOENT') throw error; return null; });
  if (receipt && (receipt.schema !== 1 || receipt.mode !== 'assisted'
      || !receipt.proposals.every(proposal => windows.includes(proposal.recording)))) {
    throw new Error('Unexpected suggested-pause receipt');
  }

  const dist = join(appDir, 'dist');
  if (!dev) await access(join(dist, 'index.html'));
  await server.register(fastifyStatic, { root: dist, serve: !dev });
  server.get('/api/packet', async (_request, reply) => {
    reply.header('Cache-Control', 'no-store');
    return { clips, windows, ballotIds, receipt };
  });
  server.get('/api/audio/:id', async (request, reply) => {
    const { id } = request.params;
    if (!ids.has(id)) return reply.code(404).send({ error: 'Audio is outside this review packet' });
    return reply.sendFile(`${id}.wav`, audioDir, { cacheControl: false });
  });

  const port = dev ? defaults.dev : Number(process.env.PORT ?? defaults.production);
  await server.listen({ port, host: '127.0.0.1' });
  console.log(`Superturn review listening at http://127.0.0.1:${port}/`);
}

start().catch(error => {
  console.error(`Cannot start Superturn review: ${error.message}`);
  process.exitCode = 1;
});
