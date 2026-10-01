const { readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');

function load() {
  const context = vm.createContext({ setTimeout, clearTimeout, Promise, Error, Set, Math, Number, String, encodeURIComponent });
  vm.runInContext(readBuilderModule('server_pipeline.mjs'), context);
  return context;
}

const fastSleep = () => Promise.resolve();

test('browser build options map to the server pipeline body', () => {
  const c = load();
  assert.deepEqual(JSON.parse(JSON.stringify(c.serverBuildBody({ buildMode: 'fresh_rebuild', sceneScope: 'from_selected', videoSeedMode: 'random' }))), {
    build_mode: 'fresh_rebuild', scope: 'from_selected', video_seed_mode: 'randomize', max_auto_retries: 3, stitch: true,
  });
  const selected = c.serverBuildBody({ sceneScope: 'selected', videoSeedMode: 'keep', maxAutoRetries: 99 });
  assert.equal(selected.stitch, false, 'selected-scenes runs do not stitch');
  assert.equal(selected.video_seed_mode, 'keep');
  assert.equal(selected.max_auto_retries, 5);
  assert.equal(c.serverBuildBody({ sceneScope: 'weird' }).scope, 'all');
  assert.equal(c.serverBuildBody({}).build_mode, 'resume_missing');
});

test('project id is the last folder name of the project path', () => {
  const c = load();
  assert.equal(c.projectIdFromFolder('C:\\ComfyUI\\output\\Higher Ground'), 'Higher Ground');
  assert.equal(c.projectIdFromFolder('/home/me/out/Song/'), 'Song');
  assert.equal(c.projectIdFromFolder(''), '');
});

test('follows a job to success and reports every poll', async () => {
  const c = load();
  const states = [{ id: 'j1', status: 'queued' }, { id: 'j1', status: 'running', progress: { percent: 40 } }, { id: 'j1', status: 'succeeded', result: { ok: 1 } }];
  let index = 0;
  const updates = [];
  const job = await c.followServerJob({
    start: async () => states[0],
    getJob: async () => states[Math.min(++index, states.length - 1)],
    cancelJob: async () => { throw new Error('should not cancel'); },
    onUpdate: (item) => updates.push(item.status),
    sleep: fastSleep,
  });
  assert.equal(job.status, 'succeeded');
  assert.deepEqual(updates, ['queued', 'running', 'succeeded']);
});

test('failed jobs throw the server message', async () => {
  const c = load();
  await assert.rejects(
    c.followServerJob({
      start: async () => ({ id: 'j', status: 'running' }),
      getJob: async () => ({ id: 'j', status: 'failed', error: { message: 'Scene 3 needs a MiniMax H3 prompt.' } }),
      cancelJob: async () => {},
      sleep: fastSleep,
    }),
    /Scene 3 needs a MiniMax H3 prompt/,
  );
  await assert.rejects(
    c.followServerJob({ start: async () => ({ id: 'j', status: 'running' }), getJob: async () => ({ id: 'j', status: 'interrupted' }), cancelJob: async () => {}, sleep: fastSleep }),
    /interrupted/,
  );
});

test('cancel is sent once and a cancelled job reports it', async () => {
  const c = load();
  let cancels = 0;
  let polls = 0;
  await assert.rejects(
    c.followServerJob({
      start: async () => ({ id: 'j', status: 'running' }),
      getJob: async () => ({ id: 'j', status: ++polls >= 3 ? 'cancelled' : 'running' }),
      cancelJob: async () => { cancels += 1; },
      shouldCancel: () => true,
      sleep: fastSleep,
    }),
    /cancelled/,
  );
  assert.equal(cancels, 1);
});

test('brief poll failures do not abandon a running job, repeated ones do', async () => {
  const c = load();
  let calls = 0;
  const job = await c.followServerJob({
    start: async () => ({ id: 'j', status: 'running' }),
    getJob: async () => { calls += 1; if (calls <= 2) throw new Error('network'); return { id: 'j', status: 'succeeded' }; },
    cancelJob: async () => {},
    sleep: fastSleep,
  });
  assert.equal(job.status, 'succeeded');
  await assert.rejects(
    c.followServerJob({
      start: async () => ({ id: 'j', status: 'running' }),
      getJob: async () => { throw new Error('server gone'); },
      cancelJob: async () => {},
      sleep: fastSleep,
      maxPollErrors: 3,
    }),
    /server gone/,
  );
});

test('progress text names the stage and message', () => {
  const c = load();
  assert.deepEqual(JSON.parse(JSON.stringify(c.describeJobProgress({ progress: { percent: 140, stage: 'rendering_video', message: 'Scene 2/9' } }))), { percent: 100, text: 'rendering video: Scene 2/9' });
  assert.equal(c.describeJobProgress({ status: 'queued' }).text, 'queued');
});

test('Build Full Video offers a server run and routes it to the server pipeline', () => {
  const source = readBuilderSource();
  assert.match(source, /key: "runWhere"/);
  assert.match(source, /runOnServer: options\.runWhere === "server"/);
  assert.match(source, /if \(options\.runOnServer\) return buildFullVideoOnServer\(options\)/);
  // The browser loop stays the default: the server path must be opt-in only.
  assert.doesNotMatch(source, /runOnServer: true/);
  // A server run must save first and refuse to continue when the save fails.
  assert.match(source, /autoSaveSessionQuiet\("Build Full Video on server"\)/);
  assert.match(source, /The project could not be saved first/);
  assert.match(source, /scene_ids: batchTargetItems\(sceneScope\)\.map\(\(item\) => item\.segment\.id\)/);
});
