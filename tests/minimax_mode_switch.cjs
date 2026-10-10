const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');

function fixture(mode, locked = false) {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  const scene = {
    id: 'scene', use_scene_minimax_h3_settings: locked,
    minimax_h3_settings: { video_mode: mode },
    minimax_h3_scene_image_use: 'exact_start_frame',
    minimax_h3_use_scene_image_as_start_frame: true,
  };
  const other = { id: 'other', minimax_h3_scene_image_use: 'exact_start_frame', minimax_h3_use_scene_image_as_start_frame: true };
  const protectedScene = { ...scene, id: 'locked', use_scene_minimax_h3_settings: true };
  const overlay = { ...other, id: 'overlay', track: 'overlay' };
  Object.assign(c, {
    state: { miniMaxH3Settings: { video_mode: mode } },
    activeSegment: () => scene, videoSettingsSegment: () => scene,
    allEditableSegments: () => [scene, other, protectedScene, overlay],
    segmentTrack: item => item.track || 'main',
    wizardVideoSettings: { global: !locked },
    requireActiveSegment: () => scene,
    miniMaxModeButtons: ['image_to_video', 'reference_to_video'].map(mode => ({ dataset: { minimaxH3Mode: mode } })),
    pushHistory() {}, autoSaveSessionQuiet: async () => {},
    syncMiniMaxH3Panel() {
      c.sceneImageControl = c.miniMaxH3SceneImageUseForSegment(scene);
    },
  });
  const panel = readBuilderModule('minimax_panel.mjs');
  for (const name of ['miniMaxH3SettingsForSegment', 'miniMaxH3ModeForSegment',
    'miniMaxH3SceneImageUseForSegment', 'setMiniMaxH3ModeForSegment',
    'setMiniMaxH3RenderPassForSegment', 'clearMiniMaxImageReferenceStartFrameOnModeSwitch']) {
    vm.runInContext(functionSource(panel, name), c);
  }
  const events = readBuilderModule('minimax_panel_events.mjs');
  const start = events.indexOf('  for (const button of miniMaxModeButtons) {');
  const end = events.indexOf('  for (const button of miniMaxPassButtons.refmod', start);
  vm.runInContext(events.slice(start, end), c);
  return { c, scene, other, protectedScene, overlay };
}

for (const locked of [false, true]) {
  for (const mode of ['image_to_video', 'image_reference_to_video']) {
    test(`${locked ? 'locked scene' : 'project'}: ${mode} to references clears the start-image requirement`, async () => {
      const { c, scene, other, protectedScene, overlay } = fixture(mode, locked);
      await c.miniMaxModeButtons[1].onclick();
      assert.equal(c.miniMaxH3ModeForSegment(scene), 'reference_to_video');
      assert.equal(c.miniMaxH3SceneImageUseForSegment(scene), 'off');
      assert.equal(scene.minimax_h3_use_scene_image_as_start_frame, false);
      assert.equal(c.sceneImageControl, 'off');
      assert.equal(other.minimax_h3_use_scene_image_as_start_frame, locked);
      assert.equal(protectedScene.minimax_h3_use_scene_image_as_start_frame, true);
      assert.equal(overlay.minimax_h3_use_scene_image_as_start_frame, true);
    });
  }
  test(`${locked ? 'locked scene' : 'project'}: Image + Reference through I2V to references clears the forced start frame`, async () => {
    const { c, scene } = fixture('image_reference_to_video', locked);
    await c.miniMaxModeButtons[0].onclick();
    assert.equal(c.miniMaxH3ModeForSegment(scene), 'image_to_video');
    await c.miniMaxModeButtons[1].onclick();
    assert.equal(c.miniMaxH3ModeForSegment(scene), 'reference_to_video');
    assert.equal(c.miniMaxH3SceneImageUseForSegment(scene), 'off');
    assert.equal(scene.minimax_h3_use_scene_image_as_start_frame, false);
  });
}

test('saved reference mode discards a legacy scene start-frame setting', () => {
  const { c, scene } = fixture('reference_to_video');
  c.clearMiniMaxImageReferenceStartFrameOnModeSwitch(null, 'reference_to_video');
  assert.equal(c.miniMaxH3SceneImageUseForSegment(scene), 'off');
  assert.equal(scene.minimax_h3_use_scene_image_as_start_frame, false);
});

test('reference mode permits environment inspiration without inheriting image framing', () => {
  const { c, scene } = fixture('reference_to_video');
  scene.minimax_h3_scene_image_use = 'environment_framing_inspiration';
  assert.equal(c.miniMaxH3SceneImageUseForSegment(scene), 'environment_inspiration');
  assert.equal(scene.minimax_h3_use_scene_image_as_start_frame, false);
});
