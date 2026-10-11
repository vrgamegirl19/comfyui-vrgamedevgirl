const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');

function fixture(state) {
  const messages = [];
  const c = vm.createContext({ state, toast: (...args) => messages.push(args) });
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  vm.runInContext(readBuilderModule('timeline_video_settings.mjs'), c);
  vm.runInContext(functionSource(readBuilderModule('minimax_panel.mjs'), 'miniMaxH3SettingsForSegment'), c);
  return { c, messages };
}

test('unlock all makes base and overlay scenes follow global mode and passes while preserving content', () => {
  const scene = { id: 'locked', use_scene_minimax_h3_settings: true, use_scene_i2v_video_settings: true,
    minimax_h3_settings: { video_mode: 'reference_to_video', render_pass: 'single', seed: 42 },
    i2v_video_settings: { steps: 12 }, minimax_h3_masked_quick_prev: { locked: true },
    video_path: 'keep.mp4', minimax_h3_prompt: 'keep prompt', locked_video: true };
  const state = { miniMaxH3Settings: { video_mode: 'image_reference_to_video', render_pass: 'two_pass' },
    segments: [scene, { id: 'unlocked', minimax_h3_mode: 'reference_to_video' }],
    overlaySegments: [{ ...scene, id: 'overlay' }], timingFrozen: true };
  const { c } = fixture(state);
  const settings = JSON.stringify(scene.minimax_h3_settings);
  c.unlockAllSceneVideoSettings(state);
  for (const item of [...state.segments, ...state.overlaySegments]) {
    assert.equal(item.use_scene_minimax_h3_settings, false);
    assert.equal(item.use_scene_i2v_video_settings, false);
    assert.equal(item.minimax_h3_mode, 'image_reference_to_video');
    assert.equal(item.minimax_h3_masked_quick_prev, undefined);
    const effective = c.miniMaxH3SettingsForSegment(item);
    assert.equal(effective.video_mode, 'image_reference_to_video');
    assert.equal(effective.render_pass, 'two_pass');
  }
  assert.equal(JSON.stringify(scene.minimax_h3_settings), settings);
  assert.equal(scene.i2v_video_settings.steps, 12);
  assert.equal(scene.video_path, 'keep.mp4');
  assert.equal(scene.minimax_h3_prompt, 'keep prompt');
  assert.equal(scene.locked_video, true);
  assert.equal(state.timingFrozen, true);
  const once = JSON.stringify(state);
  c.unlockAllSceneVideoSettings(state);
  assert.equal(JSON.stringify(state), once);
});

test('unlock button captures undo before clearing locks and refreshes and saves once', async () => {
  const state = { segments: [{ id: 'a', use_scene_minimax_h3_settings: true }] };
  const { c, messages } = fixture(state);
  const button = {}, order = [];
  c.wireUnlockAllScenes({ button, state,
    pushHistory() { assert.equal(state.segments[0].use_scene_minimax_h3_settings, true); order.push('history'); },
    syncInspector() { order.push('inspector'); }, render() { order.push('render'); },
    async autoSaveSessionQuiet() { assert.equal(state.segments[0].use_scene_minimax_h3_settings, false); order.push('save'); },
  });
  await button.onclick();
  assert.deepEqual(order, ['history', 'inspector', 'render', 'save']);
  assert.equal(messages.length, 1);
});
