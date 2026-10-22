const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture(reply, extra = {}) {
  const scenes = [
    { id: 'one', minimax_h3_prompt: 'Original one' },
    { id: 'two', minimax_h3_prompt: 'Original two' },
    { id: 'empty', minimax_h3_prompt: '' },
  ];
  const calls = [];
  const saves = [];
  let history = 0;
  const button = { disabled: false, textContent: '', title: '' };
  const input = { value: scenes[0].minimax_h3_prompt };
  const context = vm.createContext({
    GEMMA_VIDEO_PROMPT_TIMEOUT_MS: 123,
    postJson: async (url, payload) => {
      calls.push({ url, payload });
      return reply(payload);
    },
    applyCompactButtonLabel: (control, label) => { control.textContent = label; },
    toast: () => {},
    console: { warn: () => {} },
  });
  vm.runInContext(readBuilderModule('minimax_prompt_enhance.mjs'), context);
  const actions = context.createMiniMaxPromptEnhance({
    activeProjectFolderForSave: () => '/project',
    activeSegment: () => scenes[0],
    allEditableSegments: () => scenes,
    autoSaveSessionQuiet: async (reason) => { saves.push(reason); },
    createProgressWindow: () => ({ set() {}, close() {} }),
    gemmaRunnerLabel: () => 'Test LLM',
    gemmaRunnerLine: () => 'Test LLM',
    miniMaxEnhancePromptButton: button,
    miniMaxPrompt: input,
    miniMaxTextGemmaModelSelect: { value: 'selected-model' },
    miniMaxGemmaModelSelect: { value: 'vision-model' },
    miniMaxMmprojSelect: { value: 'vision-projector' },
    miniMaxEnhanceReferenceImages: (scene) => [{ path: `${scene.id}.png`, label: '<Subject 1>', picture_label: '<Picture 1>', describe: true }],
    pushHistory: () => { history += 1; },
    render() {},
    requireActiveSegment: () => scenes[0],
    sceneDisplayName: (scene) => scene.id,
    syncMiniMaxH3Panel() {},
    textGemmaRunnerPayload: () => ({ runner: 'llm_api' }),
    updateActiveFromInputs() {},
    updateMiniMaxPromptCharacterStatus() {},
    ...extra,
  });
  return { actions, scenes, calls, saves, button, input, history: () => history };
}

test('single enhancement saves the result with undo and the selected runner/model', async () => {
  const f = fixture(async () => ({ prompt: 'Enhanced one' }));
  await f.actions.enhanceMiniMaxH3PromptWithLLM({ useReferences: false });
  assert.equal(f.scenes[0].minimax_h3_prompt, 'Enhanced one');
  assert.equal(f.input.value, 'Enhanced one');
  assert.equal(f.history(), 1);
  assert.equal(f.saves.length, 1);
  assert.equal(f.button.disabled, false);
  assert.equal(f.calls[0].url, '/vrgdg/music_builder/enhance_minimax_h3_prompt');
  assert.equal(f.calls[0].payload.current_prompt, 'Original one');
  assert.equal(f.calls[0].payload.model_file, 'selected-model');
  assert.equal(f.calls[0].payload.runner, 'llm_api');
  assert.equal(f.calls[0].payload.use_reference_images, false);
  assert.equal(f.calls[0].payload.image_references, undefined);
});

test('a rejected single enhancement leaves the saved prompt and history alone', async () => {
  const f = fixture(async () => { throw new Error('Changed lyrics'); });
  await f.actions.enhanceMiniMaxH3PromptWithLLM({ useReferences: false });
  assert.equal(f.scenes[0].minimax_h3_prompt, 'Original one');
  assert.equal(f.history(), 0);
  assert.equal(f.saves.length, 0);
  assert.equal(f.button.disabled, false);
});

test('batch skips empty scenes, preserves a failed scene, and continues saving successes', async () => {
  const f = fixture(async ({ scene_id }) => {
    if (scene_id === 'one') throw new Error('Changed timing');
    return { prompt: 'Enhanced two' };
  });
  await f.actions.enhanceAllMiniMaxH3Prompts({ useReferences: false });
  assert.equal(f.scenes[0].minimax_h3_prompt, 'Original one');
  assert.equal(f.scenes[1].minimax_h3_prompt, 'Enhanced two');
  assert.equal(f.scenes[2].minimax_h3_prompt, '');
  assert.equal(f.calls.length, 2);
  assert.equal(f.calls[0].payload.unload_after, false);
  assert.equal(f.calls[1].payload.unload_after, true);
  assert.equal(f.history(), 1);
  assert.equal(f.saves.length, 1);
});

test('reference enhancement sends mapped images with the vision model and projector', async () => {
  const f = fixture(async () => ({ prompt: 'With reference descriptions' }));
  f.scenes[0].minimax_h3_prompt_reference_binding = { prompt: 'Original one', signature: 'stable' };
  await f.actions.enhanceMiniMaxH3PromptWithLLM({ useReferences: true });
  const payload = f.calls[0].payload;
  assert.equal(payload.use_reference_images, true);
  assert.equal(payload.model_file, 'vision-model');
  assert.equal(payload.mmproj_file, 'vision-projector');
  assert.equal(payload.image_references[0].path, 'one.png');
  assert.equal(payload.image_references[0].label, '<Subject 1>');
  assert.equal(f.scenes[0].minimax_h3_prompt_reference_binding.prompt, 'With reference descriptions');
  assert.equal(f.scenes[0].minimax_h3_prompt_reference_binding.signature, 'stable');
});

test('reference batch uses each scene own mapped images', async () => {
  const f = fixture(async ({ scene_id }) => ({ prompt: `Enhanced ${scene_id}` }));
  await f.actions.enhanceAllMiniMaxH3Prompts({ useReferences: true });
  assert.equal(f.calls.length, 2);
  assert.equal(f.calls[0].payload.image_references[0].path, 'one.png');
  assert.equal(f.calls[1].payload.image_references[0].path, 'two.png');
});

test('missing or stale references keep the saved prompt without issuing a request', async () => {
  for (const refs of [() => [], () => { throw new Error('Reference order changed'); }]) {
    const f = fixture(async () => ({ prompt: 'Wrong' }), { miniMaxEnhanceReferenceImages: refs });
    await f.actions.enhanceMiniMaxH3PromptWithLLM({ useReferences: true });
    assert.equal(f.calls.length, 0);
    assert.equal(f.scenes[0].minimax_h3_prompt, 'Original one');
    assert.equal(f.history(), 0);
    assert.equal(f.button.disabled, false);
  }
});

test('chooser resolves both image choices and cancellation and removes its keyboard listener', async () => {
  for (const mode of ['with', 'without', 'cancel', 'escape', 'backdrop']) {
    const keys = new Map();
    let removed = false;
    const makeElement = () => ({
      style: {}, children: [], events: {},
      append(...items) { this.children.push(...items); },
      setAttribute() {}, focus() {},
      addEventListener(name, handler) { this.events[name] = handler; },
      remove() { removed = true; },
    });
    const body = makeElement();
    const c = vm.createContext({
      document: { body, createElement: makeElement,
        addEventListener: (name, handler) => keys.set(name, handler),
        removeEventListener: name => keys.delete(name),
      },
      makeButton: text => ({ ...makeElement(), textContent: text }),
    });
    vm.runInContext(functionSource(readBuilderModule('minimax_prompt_enhance.mjs'), 'chooseMiniMaxEnhanceReferences'), c);
    const pending = c.chooseMiniMaxEnhanceReferences();
    const backdrop = body.children[0];
    const box = backdrop.children[0];
    if (mode === 'escape') keys.get('keydown')({ key: 'Escape', preventDefault() {} });
    else if (mode === 'backdrop') backdrop.events.pointerdown({ target: backdrop });
    else box.children[{ with: 2, without: 3, cancel: 4 }[mode]].onclick();
    assert.equal(await pending, mode === 'with' ? true : mode === 'without' ? false : null);
    assert.equal(keys.size, 0);
    assert.equal(removed, true);
  }
});

test('image reference enhancement preserves frame offsets and subject/location labels', () => {
  const items = [
    { kind: 'start_frame', label: 'Start', image: { path: 'start.png' } },
    { kind: 'subject', label: 'Singer', image: { path: 'singer.png' } },
    { kind: 'extra', label: 'Friend', image: { path: 'friend.png' } },
    { kind: 'location', label: 'Room', image: { path: 'room.png' } },
  ];
  const c = vm.createContext({
    miniMaxH3ModeForSegment: () => 'image_reference_to_video', isRefmodPipelineActive: () => false,
    miniMaxH3ImageReferencePromptItems: () => items,
    miniMaxH3CombinedSubjectPlan: () => ({ subjects: [
      { label: '<Subject 1>', pictureLabel: '<Picture 2>', name: 'Singer', kind: 'subject' },
      { label: '<Subject 2>', pictureLabel: '<Picture 3>', name: 'Friend', kind: 'extra' },
      { label: '<Subject 3>', pictureLabel: '<Picture 4>', name: 'environment', kind: 'location' },
    ] }),
    miniMaxPromptReferenceMismatch: () => '',
    miniMaxReferencePurposeText: item => `${item.kind} reference`,
  });
  vm.runInContext(functionSource(readBuilderModule('minimax_prompt.mjs'), 'miniMaxEnhanceReferenceImages'), c);
  const refs = c.miniMaxEnhanceReferenceImages({ minimax_h3_prompt: 'saved' });
  assert.equal(refs.length, 4);
  assert.equal(refs[0].describe, false);
  assert.equal(refs[1].label, '<Subject 1>');
  assert.equal(refs[1].picture_label, '<Picture 2>');
  assert.equal(refs[2].path, 'friend.png');
  assert.equal(refs[2].kind, 'extra');
  assert.equal(refs[3].label, '<Subject 3>');
  assert.equal(refs[3].picture_label, '<Picture 4>');
  assert.equal(refs[3].describe, true);
});
