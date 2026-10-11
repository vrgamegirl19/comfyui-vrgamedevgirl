const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');
async function harness() {
  const { cloneMiniMaxH3Settings } = await import('../web/music_video_builder/minimax_h3.mjs');
  const messages = [];
  const context = vm.createContext({ cloneMiniMaxH3Settings, toast: (...args) => messages.push(args) });
  vm.runInContext(readBuilderModule('timeline_continuity.mjs'), context);
  return { context, messages };
}
test('bulk off covers inherited, locked, inactive and overlay settings and preserves unrelated state', async () => {
  const state = {
    miniMaxH3Settings: { continuity_mode: 'latent_continuation_masked', continuity_prompt_from_last_frame: true, seed: 42 },
    segments: [{ id: 'inherit' }, { id: 'locked', use_scene_minimax_h3_settings: true, minimax_h3_settings: { continuity_mode: 'latent_continuation_masked', continuity_prompt_from_last_frame: true, seed: 99 }, minimax_h3_masked_quick_prev: { continuity_mode: 'latent_continuation_masked' }, video_path: 'keep.mp4' }, { id: 'inactive', use_scene_minimax_h3_settings: false, minimax_h3_settings: { continuity_mode: 'latent_continuation_masked' } }, { id: 'empty-lock', use_scene_minimax_h3_settings: true }],
    overlaySegments: [{ id: 'overlay', minimax_h3_settings: { continuity_mode: 'latent_continuation_masked' } }],
  };
  const { context } = await harness();
  context.disableAllLatentContinuation(state);
  for (const scene of [...state.segments, ...state.overlaySegments]) {
    const settings = scene.minimax_h3_settings || state.miniMaxH3Settings;
    assert.equal(settings.continuity_mode, 'off');
    assert.equal(settings.continuity_prompt_from_last_frame, false);
    assert.equal(scene.minimax_h3_masked_quick_prev, undefined);
  }
  assert.equal(state.miniMaxH3Settings.seed, 42);
  assert.equal(state.segments[1].minimax_h3_settings.seed, 99);
  assert.equal(state.segments[1].use_scene_minimax_h3_settings, true);
  assert.equal(state.segments[1].video_path, 'keep.mp4');
  const off = JSON.stringify(state);
  context.disableAllLatentContinuation(state);
  assert.equal(JSON.stringify(state), off);
});
test('button captures history before mutation, refreshes and saves once', async () => {
  const state = { miniMaxH3Settings: { continuity_mode: 'latent_continuation_masked' }, segments: [{ id: 'a' }] };
  const { context, messages } = await harness();
  const button = {}, order = [];
  context.wireLatentContinuationOffAll({ button, state,
    pushHistory: () => { order.push('history'); assert.equal(state.miniMaxH3Settings.continuity_mode, 'latent_continuation_masked'); },
    syncInspector: () => order.push('inspector'), render: () => order.push('render'),
    autoSaveSessionQuiet: async () => { order.push('save'); assert.equal(state.miniMaxH3Settings.continuity_mode, 'off'); },
  });
  await button.onclick();
  assert.deepEqual(order, ['history', 'inspector', 'render', 'save']);
  assert.equal(messages.length, 1);
});
test('save failure is displayed', async () => {
  const { context, messages } = await harness();
  const button = {};
  context.wireLatentContinuationOffAll({ button, state: {}, pushHistory() {}, syncInspector() {}, render() {}, autoSaveSessionQuiet: async () => { throw new Error('Save failed'); } });
  await button.onclick();
  assert.deepEqual(messages, [['Error: Save failed', true]]);
});

test('bulk-off button is the rightmost timeline header control', () => {
  const { functionSource } = require('./builder_source.cjs');
  class Element {
    constructor() { this.style = {}; this.children = []; this.options = []; }
    append(...items) { for (const item of items) { this.children.push(item); item.parentElement = this; } }
    addEventListener() {}
  }
  const element = () => new Element();
  const context = vm.createContext({ document: { createElement: element },
    makeButton: label => Object.assign(element(), { textContent: label }),
    makeCheckbox: () => ({ wrapper: element(), input: element() }),
    makeSelect: element, WAVEFORM_MODES: {}, TIMELINE_HEIGHT: 400,
    createTimelineToolWindows: () => ({ toolsButton: element(), deleteAllButton: element(), refreshDeleteActions() {} }),
  });
  vm.runInContext(functionSource(readBuilderModule('timeline_view.mjs'), 'buildTimelineView'), context);
  const view = context.buildTimelineView({ overlay: element(), preview: element(), previewStage: element() });
  const button = view.latentContinuationOffAllButton;
  assert.equal(button.textContent, 'Latent Continuation: Off All');
  assert.equal(button.parentElement.children.at(-1), button);
  assert.equal(button.parentElement.children.at(-2), view.freezeTimingControl.wrapper);
});

test('undo snapshot includes project continuity settings alongside scene overrides', () => {
  const { functionSource } = require('./builder_source.cjs');
  const state = { miniMaxH3Settings: { continuity_mode: 'latent_continuation_masked' }, segments: [{ id: 'a' }] };
  const source = functionSource(readBuilderModule('history.mjs'), 'historySnapshot');
  const globals = { state, historySnapshotReplacer: (key, value) => value };
  for (const match of source.matchAll(/\b(normalize\w+)\(/g)) globals[match[1]] = value => value;
  const context = vm.createContext(globals);
  vm.runInContext(source, context);
  const snapshot = JSON.parse(context.historySnapshot());
  assert.deepEqual(snapshot.miniMaxH3Settings, state.miniMaxH3Settings);
  assert.deepEqual(snapshot.segments, state.segments);
});
