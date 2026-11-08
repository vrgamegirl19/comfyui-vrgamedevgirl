const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture(scenes) {
  const events = [];
  const c = vm.createContext({ Map, state: { videoType: 'singing', autoTimeSingerCuesBeforePrompt: true },
    miniMaxH3CutPlanForSegment: scene => scene,
    miniMaxH3OfficialShotPlan: scene => Array(scene.shots || 1).fill({}),
    normalizeVideoType: value => value,
    isMiniMaxSingerAssignmentMode: () => true,
    isInstrumentalLyricText: text => text === '[instrumental]',
    flattenLyricForPrompt: text => String(text || '').trim(),
    selectedPerformerSubjectsForSegment: () => [{ id: 'singer', name: 'Singer' }],
    normalizeLyricCueMapForSegment: scene => scene.lyric_cue_map || [],
    miniMaxEffectiveCueEnd: (scene, cues, index, cue) => cue.end ?? 8,
    sceneDisplayName: scene => scene.id, segmentIndexInfo: () => ({ index: 0 }),
    autoTimeMiniMaxSingerCuesForSegment: async (scene, options) => {
      events.push(`whisper:${scene.id}`);
      assert.equal(options.unloadAfter, false);
      if (scene.fail) return false;
      scene.lyric_cue_map = [{ text: scene.lyric_text, start: 0, end: 4 }];
      return true;
    },
    postJson: async () => { events.push('unload'); },
  });
  vm.runInContext(functionSource(readBuilderModule('minimax_prompt.mjs'), 'ensureAutoTimedSingerCuesBeforePrompt'), c);
  vm.runInContext(readBuilderModule('batch_singer_timing.mjs'), c);
  return { c, events, scenes, prepare: options => c.prepareBatchSingerTiming(scenes, c.ensureAutoTimedSingerCuesBeforePrompt, options) };
}
const scene = (id, shots = 2, extra = {}) => ({ id, shots, lyric_text: 'Sung words', ...extra });

test('Whisper processes only multi-shot scenes and unloads once before any prompt', async () => {
  const f = fixture([scene('one'), scene('single', 1), scene('two')]);
  const errors = await f.prepare();
  assert.equal(errors.size, 0);
  for (const item of f.scenes) f.events.push(`prompt:${item.id}`);
  assert.deepEqual(f.events, ['whisper:one', 'whisper:two', 'unload', 'prompt:one', 'prompt:single', 'prompt:two']);
});

test('a fully timed or single-shot batch does not load or clear Whisper', async () => {
  const f = fixture([scene('single', 1), scene('timed', 3, { lyric_performance_mode: 'cue_map', lyric_cue_map: [{ text: 'Manual', start: 0, end: null }] })]);
  await f.prepare();
  assert.deepEqual(f.events, []);
});

test('timing failures remain scene-specific and Whisper still unloads', async () => {
  const bad = scene('bad', 2, { fail: true });
  const f = fixture([bad, scene('good')]);
  const errors = await f.prepare();
  assert.match(errors.get(bad).message, /Whisper timing did not complete/);
  assert.equal(errors.size, 1);
  assert.deepEqual(f.events, ['whisper:bad', 'whisper:good', 'unload']);
});

test('cancelling between scenes frees Whisper before propagating the stop', async () => {
  const f = fixture([scene('one'), scene('two')]);
  let calls = 0;
  await assert.rejects(f.prepare({ assertNotStopped: () => { if (++calls === 2) throw new Error('Stopped by user'); } }), /Stopped by user/);
  assert.deepEqual(f.events, ['whisper:one', 'unload']);
});

test('LLM All completes timing before prompting and disables per-scene re-timing', async () => {
  const f = fixture([scene('one'), scene('single', 1), scene('two')]);
  const failures = {};
  Object.assign(f.c, {
    currentVideoMode: () => 't2v', videoModeDisplayLabel: () => 'T2V', promptRunnerActionName: () => 'LLM API',
    createProgressWindow: () => ({ set() {}, close() {} }), normalizeBatchScope: () => 'all', batchScopeLabel: () => 'all',
    batchTargetItems: () => f.scenes.map(segment => ({ segment })),
    videoVisionReferenceEnabled: () => false, createI2VButton: {}, sceneVideoConceptPromptText: () => 'Scene concept', projectInput: { value: 'project' },
    saveSessionForSceneVideo: async () => {}, assertBatchNotStopped() {}, syncInspector() {}, render() {},
    generateI2VPromptForSegment: async (item, _progress, _percent, _label, options) => {
      assert.equal(options.skipAutoTiming, true);
      assert.equal(options.unloadAfter, false);
      f.events.push(`prompt:${item.id}`);
    },
    gemmaBatchFailureStore: () => failures, autoSaveSessionQuiet: async () => {},
    runClearMemoryWorkflowQuiet: async () => { f.events.push('prompt-unload'); },
    toast() {}, isRecoverableBuildGemmaError: () => false, saveGemmaJunkDebug: async () => '',
  });
  vm.runInContext(functionSource(readBuilderModule('batch_prompts.mjs'), 'i2vAllScenes'), f.c);
  await f.c.i2vAllScenes({ forceTextOnly: true, throwOnError: true });
  assert.deepEqual(f.events, ['whisper:one', 'whisper:two', 'unload', 'prompt:one', 'prompt:single', 'prompt:two', 'prompt-unload']);
});

test('Story Board All pre-times only targeted scenes, keeps the API prompt phase loaded and cleans up', async () => {
  const f = fixture([scene('one'), scene('single', 1), scene('two', 2, { video_prompt: 'Already done' })]);
  Object.assign(f.c.state, {
    mode: 'image_to_video_prep', gemmaSettings: { text_runner: 'llm_api' },
    onBeforeCreateVideoPrompts: (items, options) => f.c.prepareBatchSingerTiming(items, f.c.ensureAutoTimedSingerCuesBeforePrompt, options),
  });
  Object.assign(f.c, {
    currentRows: () => f.scenes, createToast() {}, gemmaAllButton: {},
    promptRunnerName: () => 'LLM API', promptRunnerGenericName: () => 'LLM API',
    createStoryboardProgressWindow: () => ({ set() {}, close() {} }), keepGemmaLoadedInput: { checked: false },
    createScenePromptForActiveMode: async (item, options) => {
      assert.equal(options.skipAutoTiming, true);
      assert.equal(options.unloadAfter, false);
      f.events.push(`prompt:${item.id}`);
    }, saveStoryboard: async () => {}, renderTable() {},
  });
  const fs = require('node:fs'), path = require('node:path');
  const source = fs.readFileSync(path.join(__dirname, '../web/storyboard_builder/prompt_generation.mjs'), 'utf8');
  vm.runInContext(functionSource(source, 'createAllPromptsWithGemma'), f.c);
  await f.c.createAllPromptsWithGemma({ onlyMissing: true });
  assert.deepEqual(f.events, ['whisper:one', 'unload', 'prompt:one', 'prompt:single', 'unload']);
});
