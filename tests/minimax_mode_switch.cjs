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
    miniMaxModeSelect: { value: mode },
    pushHistory() {}, autoSaveSessionQuiet: async () => {},
    syncMiniMaxH3Panel() {
      c.sceneImageControl = c.miniMaxH3SceneImageUseForSegment(scene);
    },
  });
  const panel = readBuilderModule('minimax_panel.mjs');
  for (const name of ['editableMiniMaxH3Settings', 'miniMaxH3SettingsForSegment', 'miniMaxH3ModeForSegment',
    'miniMaxH3SceneImageUseForSegment', 'setMiniMaxH3ModeForSegment',
    'setMiniMaxH3RenderPassForSegment', 'clearMiniMaxImageReferenceStartFrameOnModeSwitch']) {
    vm.runInContext(functionSource(panel, name), c);
  }
  const events = readBuilderModule('minimax_panel_events.mjs');
  const start = events.indexOf('  miniMaxModeSelect.onchange = async () => {');
  const end = events.indexOf('  for (const button of miniMaxPassButtons.refmod', start);
  vm.runInContext(events.slice(start, end), c);
  return { c, scene, other, protectedScene, overlay };
}

for (const locked of [false, true]) {
  for (const mode of ['image_to_video', 'image_reference_to_video']) {
    test(`${locked ? 'locked scene' : 'project'}: ${mode} to references clears the start-image requirement`, async () => {
      const { c, scene, other, protectedScene, overlay } = fixture(mode, locked);
      c.miniMaxModeSelect.value = 'reference_to_video';
      await c.miniMaxModeSelect.onchange();
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
    c.miniMaxModeSelect.value = 'image_to_video';
    await c.miniMaxModeSelect.onchange();
    assert.equal(c.miniMaxH3ModeForSegment(scene), 'image_to_video');
    c.miniMaxModeSelect.value = 'reference_to_video';
    await c.miniMaxModeSelect.onchange();
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

for (const locked of [false, true]) {
  test(`${locked ? 'locked scene' : 'project'}: dropdown selects every mode with the existing pass defaults`, async () => {
    const { c, scene } = fixture('text_to_video', locked);
    let history = 0;
    const saves = [];
    c.pushHistory = () => history++;
    c.autoSaveSessionQuiet = async reason => saves.push(reason);
    const modes = ['image_to_video', 'image_reference_to_video', 'reference_to_video', 'video_to_video', 'text_to_video'];
    for (const mode of modes) {
      c.miniMaxModeSelect.value = mode;
      await c.miniMaxModeSelect.onchange();
      const settings = c.miniMaxH3SettingsForSegment(scene);
      assert.equal(settings.video_mode, mode);
      assert.equal(settings.render_pass, mode === 'image_reference_to_video' ? 'two_pass' : 'single');
      if (locked) assert.equal(c.state.miniMaxH3Settings.video_mode, 'text_to_video');
    }
    assert.equal(history, modes.length);
    assert.equal(saves.length, modes.length);
    assert.ok(saves.every(reason => reason === (locked ? 'MiniMax H3 locked scene mode' : 'MiniMax H3 project mode')));
  });
}

test('dropdown tracks restored modes and is hidden only for the RefMod pipeline', () => {
  const panel = readBuilderModule('minimax_panel.mjs');
  const start = panel.indexOf('    miniMaxModeSelect.value = mode;');
  const end = panel.indexOf('    // 2 Pass Advanced', start);
  const refmod = { modeChooser: { style: {} }, note: { style: {} }, pipelineButtons: [] };
  const c = vm.createContext({ miniMaxModeSelect: {}, miniMaxPassButtons: { refmod }, settings: {} });
  for (const pipeline of ['standard', 'refmod', 'standard']) {
    for (const mode of ['text_to_video', 'image_to_video', 'image_reference_to_video', 'reference_to_video', 'video_to_video']) {
      c.mode = mode;
      c.settings.pipeline = pipeline;
      vm.runInContext(`{${panel.slice(start, end)}}`, c);
      assert.equal(c.miniMaxModeSelect.value, mode);
      assert.equal(refmod.modeChooser.style.display, pipeline === 'refmod' ? 'none' : 'flex');
      assert.equal(refmod.note.style.display, pipeline === 'refmod' ? 'block' : 'none');
    }
  }
});

test('changing the dropdown without a scene restores the displayed mode without saving', async () => {
  const { c } = fixture('reference_to_video', true);
  c.requireActiveSegment = () => null;
  c.syncMiniMaxH3Panel = () => { c.miniMaxModeSelect.value = 'reference_to_video'; };
  c.pushHistory = () => assert.fail('No scene means no mutation');
  c.autoSaveSessionQuiet = () => assert.fail('No scene means no save');
  c.miniMaxModeSelect.value = 'text_to_video';
  await c.miniMaxModeSelect.onchange();
  assert.equal(c.miniMaxModeSelect.value, 'reference_to_video');
});

test('Standard mode chooser renders one labeled native dropdown with all five modes', () => {
  const document = { createElement: tag => ({
    tag, style: {}, children: [], attributes: {},
    append(...items) { this.children.push(...items); },
    setAttribute(name, value) { this.attributes[name] = value; },
  }) };
  const c = vm.createContext({ document, BUILDER_FONT_STACK: 'sans-serif' });
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  const controls = readBuilderModule('controls.mjs');
  for (const name of ['makeSelect', 'makeField']) vm.runInContext(functionSource(controls, name), c);
  const layout = readBuilderModule('minimax_panel_layout.mjs');
  const start = layout.indexOf('  const miniMaxModeSelect =');
  const end = layout.indexOf('  const miniMaxPassChooser', start);
  vm.runInContext(layout.slice(start, end) + '\nthis.chooser = miniMaxModeChooser; this.select = miniMaxModeSelect;', c);
  assert.equal(c.chooser.tag, 'label');
  assert.equal(c.chooser.children[0].textContent, 'Video mode');
  assert.equal(c.select.tag, 'select');
  assert.equal(c.select.attributes['aria-label'], 'MiniMax H3 video mode');
  assert.equal(c.select.value, 'text_to_video');
  assert.deepEqual(c.select.children.map(option => [option.value, option.textContent]), [
    ['text_to_video', 'Text to Video'],
    ['image_to_video', 'Image to Video'],
    ['image_reference_to_video', 'Image + Reference 2 Pass'],
    ['reference_to_video', 'Reference to Video'],
    ['video_to_video', 'Video to Video'],
  ]);
});
