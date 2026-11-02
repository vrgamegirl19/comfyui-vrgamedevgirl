const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

function fixture(options = {}) {
  const scenes = [
    { id: 'a', label: 'First', start: 0, end: 5, minimax_h3_prompt: 'Original A', api_video_prompt: 'Prior Seedance A' },
    { id: 'b', label: 'Second', start: 5, end: 10, minimax_h3_prompt: 'Original B' },
  ];
  const calls = [], notices = [];
  const state = { projectVideoEngine: 'minimax_h3' };
  let active = options.noActiveScene ? null : scenes[1];
  let conversions = 0, history = 0, saved = 0;
  const c = vm.createContext({
    GEMMA_VIDEO_PROMPT_TIMEOUT_MS: 1000,
    toast: message => notices.push(message),
    postJson: async (url, body) => {
      calls.push([url, body]);
      if (url.includes('backup_seedance')) {
        if (options.failBackup) throw new Error('Disk full');
        assert.equal(conversions, 0);
        assert.equal(scenes[0].api_video_prompt, 'Prior Seedance A');
        return { backup_id: 'backup', backup_path: 'project/project_context/prompt_backups/backup.json' };
      }
      conversions++;
      assert.equal(scenes[0].api_video_prompt, 'Prior Seedance A', 'no partial batch writes while LLM calls are running');
      if (options.failSecond && conversions === 2) throw new Error('Invalid formatting plan');
      if (options.failFirst && conversions === 1) throw new Error('Invalid formatting plan');
      if (options.switchScene && conversions === 1) active = scenes[0];
      if (options.changeSource && conversions === 2) scenes[0].minimax_h3_prompt = 'User edit while running';
      if (options.cancel && conversions === 1) state.batchCancelled = true;
      return { prompt: `Converted ${body.scene_id}` };
    },
  });
  vm.runInContext(readBuilderModule('api_video.mjs'), c);
  vm.runInContext(readBuilderModule('seedance_conversion.mjs'), c);
  const refs = segment => ({ image_references: [{ path: segment.id + '.png' }], reference_aliases: { '<Subject 1>': '@Image 1' }, video_paths: [], audio_paths: [], original_soundtrack: true });
  const converter = c.createSeedanceConversion({
    state, activeSegment: () => active, allEditableSegments: () => scenes, activeProjectFolderForSave: () => 'project',
    updateActiveFromInputs() {}, saveApiPanel() {}, getSourceReferences: refs,
    textGemmaRunnerPayload: () => ({ text_runner: 'llm_api' }), getTextModel: () => 'model',
    createProgressWindow: () => ({ set() {} }), pushHistory: () => { history++; },
    saveSession: async () => { saved++; return options.failSave ? { stale: true } : { ok: true }; }, syncInspector() {}, render() {},
  });
  return { ...converter, scenes, state, calls, notices, c, counts: () => ({ conversions, history, saved }) };
}

test('backs up all originals before LLM calls and commits a complete converted batch once', async () => {
  const f = fixture();
  await f.convertAllToSeedancePrompts();
  assert.match(f.calls[0][0], /backup_seedance/);
  assert.equal(f.calls[0][1].scenes[0].minimax_h3_prompt, 'Original A');
  assert.equal(f.calls[0][1].scenes[0].api_video_prompt, 'Prior Seedance A');
  assert.deepEqual(f.scenes.map(s => s.minimax_h3_prompt), ['Original A', 'Original B']);
  assert.deepEqual(f.scenes.map(s => s.api_video_prompt), ['Converted a', 'Converted b']);
  assert.equal(f.state.projectVideoEngine, 'minimax_h3');
  assert.equal(f.scenes[0].api_video_settings.use_mapped_references, false);
  assert.equal(f.scenes[0].api_video_image_references[0].path, 'a.png');
  assert.deepEqual(f.counts(), { conversions: 2, history: 1, saved: 1 });
  assert.equal(f.calls[1][1].backup_id, 'backup');
  assert.equal('source_prompt' in f.calls[1][1], false, 'conversion reads the actual saved backup');
});

for (const [name, options, requests] of [
  ['failed backup', { failBackup: true }, 0],
  ['failed second scene', { failSecond: true }, 2],
  ['cancellation', { cancel: true }, 1],
]) {
  test(`${name} leaves every prior Seedance prompt intact`, async () => {
    const f = fixture(options);
    await f.convertAllToSeedancePrompts();
    assert.equal(f.scenes[0].api_video_prompt, 'Prior Seedance A');
    assert.equal(f.scenes[1].api_video_prompt, undefined);
    assert.deepEqual(f.counts(), { conversions: requests, history: 0, saved: 0 });
    assert.match(f.notices.at(-1), /No prompts were changed/);
  });
}

test('user edits during conversion are kept instead of overwritten by stale conversion results', async () => {
  const f = fixture({ changeSource: true });
  await f.convertAllToSeedancePrompts();
  assert.equal(f.scenes[0].minimax_h3_prompt, 'User edit while running');
  assert.equal(f.scenes[0].api_video_prompt, 'Prior Seedance A');
  assert.equal(f.counts().saved, 0);
});

test('an unsuccessful session save reports that converted text needs saving and retains the backup path', async () => {
  const f = fixture({ failSave: true });
  await f.convertAllToSeedancePrompts();
  assert.equal(f.scenes[0].api_video_prompt, 'Converted a');
  assert.match(f.notices.at(-1), /could not be saved/);
  assert.match(f.notices.at(-1), /save the project again/);
  assert.match(f.notices.at(-1), /prompt_backups/);
});

test('deduplicated reference images keep Subject and Picture aliases bound to actual upload order', () => {
  const f = fixture();
  const assignments = f.c.seedanceImageAssignments([
    { path: 'wren.png', label: '<Subject 1>', picture_label: '<Picture 1>' },
    { path: 'wren.png', label: '<Subject 1>', picture_label: '<Picture 2>' },
    { path: 'foyer.png', label: '<Subject 2>', picture_label: '<Picture 3>' },
  ]);
  assert.equal(assignments.images.length, 2);
  assert.equal(assignments.aliases['<Subject 1>'], '@Image 1');
  assert.equal(assignments.aliases['<Picture 2>'], '@Image 1');
  assert.equal(assignments.aliases['<Subject 2>'], '@Image 2');
  assert.equal(assignments.aliases['<Picture 3>'], '@Image 2');
});

test('converted prompt references remain stable and later reference edits require reconversion', async () => {
  const f = fixture();
  await f.convertAllToSeedancePrompts();
  const scene = f.scenes[0];
  assert.equal(f.c.validateApiVideoScene(scene, f.c.apiVideoReferences(scene)).length, 0);
  scene.api_video_image_references[0].path = 'another-character.png';
  assert.match(f.c.validateApiVideoScene(scene, f.c.apiVideoReferences(scene))[0], /references changed after conversion/);
});

test('source prompt selection follows the active engine and falls back to existing source prompts', () => {
  const f = fixture();
  const scene = { i2v_prompt: 'LTX', minimax_h3_prompt: 'H3', api_video_prompt: 'Seedance' };
  assert.equal(f.c.seedanceSourcePrompt(scene, 'minimax_h3').prompt, 'H3');
  assert.equal(f.c.seedanceSourcePrompt(scene, 'api_nodes').prompt, 'Seedance');
  assert.equal(f.c.seedanceSourcePrompt(scene, 'ltx').prompt, 'LTX');
  assert.equal(f.c.seedanceSourcePrompt({ minimax_h3_prompt: 'H3' }, 'api_nodes').prompt, 'H3');
});

test('single-scene conversion backs up, requests and changes only the selected scene', async () => {
  const f = fixture();
  const untouched = JSON.stringify(f.scenes[0]);
  await f.convertSceneToSeedancePrompt();
  assert.equal(f.calls[0][1].scenes.length, 1);
  assert.equal(f.calls[0][1].scenes[0].id, 'b');
  assert.equal(f.calls[0][1].scenes[0].source_prompt, 'Original B');
  assert.equal(f.calls[1][1].scene_id, 'b');
  assert.equal(JSON.stringify(f.scenes[0]), untouched);
  assert.equal(f.scenes[1].api_video_prompt, 'Converted b');
  assert.equal(f.scenes[1].minimax_h3_prompt, 'Original B');
  assert.deepEqual(f.counts(), { conversions: 1, history: 1, saved: 1 });
});

test('single-scene failure keeps every scene intact and no selection never falls back to Convert All', async () => {
  const f = fixture({ failFirst: true });
  const before = JSON.stringify(f.scenes);
  await f.convertSceneToSeedancePrompt();
  assert.equal(JSON.stringify(f.scenes), before);
  assert.equal(f.counts().saved, 0);
  const empty = fixture({ noActiveScene: true });
  await empty.convertSceneToSeedancePrompt();
  assert.equal(empty.calls.length, 0);
  assert.match(empty.notices[0], /Select a scene/);
});

test('switching selection during conversion does not change its original target', async () => {
  const f = fixture({ switchScene: true });
  await f.convertSceneToSeedancePrompt();
  assert.equal(f.scenes[0].api_video_prompt, 'Prior Seedance A');
  assert.equal(f.scenes[1].api_video_prompt, 'Converted b');
});
