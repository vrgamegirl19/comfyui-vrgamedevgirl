const { functionSource, readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = readBuilderSource();
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function fixture() {
  const c = vm.createContext({ state: { miniMaxH3Settings: { video_mode: 'text_to_video', render_pass: 'single' } } });
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  vm.runInContext(functionSource(source, 'miniMaxH3SettingsForSegment'), c);
  return c;
}
for (const renderPass of ['single', 'two_pass', 'three_pass']) {
  test(`legacy locked reference scene inherits loaded ${renderPass} project before normalization`, () => {
    const c = fixture();
    c.session = { minimax_h3_settings: { video_mode: 'reference_to_video', ...(renderPass === 'single' ? { render_pass: 'single' } : {}) }, minimax_h3_two_pass: renderPass === 'two_pass', minimax_h3_advanced_two_pass: renderPass === 'three_pass' };
    const load = functionSource(source, 'loadSessionFromProject');
    const start = load.indexOf('state.miniMaxH3Settings = cloneMiniMaxH3Settings(session.');
    const end = load.indexOf('state.miniMaxH3ThreePassEnabled =', start);
    assert.ok(start >= 0 && start < load.indexOf('ensureAllSegmentRuntimeFields();'));
    vm.runInContext(load.slice(start, load.indexOf('\n', end)), c);
    const scene = { use_scene_minimax_h3_settings: true, minimax_h3_settings: { video_mode: 'reference_to_video' } };
    assert.equal(c.miniMaxH3SettingsForSegment(scene).render_pass, renderPass);
    c.state.miniMaxH3Settings.render_pass = 'single';
    assert.equal(c.miniMaxH3SettingsForSegment(scene).render_pass, renderPass);
  });
  test(`explicit scene ${renderPass} survives a different project mode`, () => {
    const c = fixture();
    const scene = { use_scene_minimax_h3_settings: true, minimax_h3_settings: { video_mode: 'reference_to_video', render_pass: renderPass } };
    assert.equal(c.miniMaxH3SettingsForSegment(scene).render_pass, renderPass);
    const reloaded = JSON.parse(JSON.stringify(scene));
    c.state.miniMaxH3Settings.render_pass = 'three_pass';
    assert.equal(c.miniMaxH3SettingsForSegment(reloaded).render_pass, renderPass);
  });
}
test('legacy Image + Reference stays two pass through runtime normalization and scene lookup', () => {
  const c = fixture();
  c.segment = { use_scene_minimax_h3_settings: true, minimax_h3_mode: 'image_reference_to_video', minimax_h3_settings: {} };
  const runtime = functionSource(source, 'ensureSegmentRuntimeFields');
  const start = runtime.indexOf('    if (segment.use_scene_minimax_h3_settings) {');
  const end = runtime.indexOf('    if (segment.minimax_h3_prompt == null)', start);
  vm.runInContext(runtime.slice(start, end), c);
  assert.equal(c.segment.minimax_h3_settings.render_pass, 'two_pass');
  assert.equal(c.miniMaxH3SettingsForSegment(c.segment).render_pass, 'two_pass');
});
test('save response restores project settings before normalizing scenes', () => {
  const refresh = section('if (data.session && options.refreshFromSavedSession === true)', 'projectInput.value = state.projectFolder;');
  assert.ok(refresh.indexOf('state.miniMaxH3Settings =') < refresh.indexOf('ensureAllSegmentRuntimeFields();'));
});
test('changing a locked scene pass does not change project or another locked scene', () => {
  const c = fixture();
  vm.runInContext(functionSource(source, 'setMiniMaxH3RenderPassForSegment'), c);
  const first = { use_scene_minimax_h3_settings: true, minimax_h3_settings: { video_mode: 'reference_to_video', render_pass: 'two_pass' } };
  const second = JSON.parse(JSON.stringify(first));
  c.setMiniMaxH3RenderPassForSegment(first, 'three_pass');
  assert.equal(c.miniMaxH3SettingsForSegment(first).render_pass, 'three_pass');
  assert.equal(c.miniMaxH3SettingsForSegment(second).render_pass, 'two_pass');
  assert.equal(c.state.miniMaxH3Settings.render_pass, 'single');
});
test('successful bulk deletion removes videos, backups and thumbnails then clears assignments', async () => {
  const c = fixture();
  const scene = { video_path: 'current.mp4', video_history: ['current.mp4', 'old.mp4'], video_backup_paths: ['backup.mp4'], video_thumbnail_path: 'current.jpg' };
  const deleted = [];
  let saved = false;
  const player = { pause() {}, removeAttribute() {}, load() {}, dataset: {}, style: {} };
  Object.assign(c, { allEditableSegments: () => [scene], confirmDestructiveAction: async () => ({ confirmed: true, optionChecked: false }),
    deleteAllTimelineVideosButton: {}, pauseTimelineForEditing() {}, pushHistory() {}, projectInput: { value: '/project' },
    postJson: async (url, data) => { deleted.push(data.path); return { deleted: true }; },
    ensureSegmentRuntimeFields() {}, deleteStaleSceneLatents: async () => {}, previewVideo: player, sceneAudio: player,
    syncInspector() {}, syncPreview() {}, activeSegment() {}, renderList() {}, render() {}, updateSelectedMediaTools() {},
    autoSaveSessionQuiet: async () => { saved = true; }, toast() {},
  });
  vm.runInContext(section('async function deleteAllTimelineVideos()', 'async function deleteAllTimelineImages()'), c);
  await c.deleteAllTimelineVideos();
  assert.deepEqual(deleted, ['current.mp4', 'old.mp4', 'backup.mp4', 'current.jpg']);
  assert.equal(scene.video_path, '');
  assert.equal(scene.video_history.length, 0);
  assert.equal(saved, true);
});
test('partial bulk deletion retains scene references and reports failure', async () => {
  const c = fixture();
  const scene = { video_path: 'locked.mp4', video_history: ['locked.mp4', 'old.mp4'] };
  let message = '';
  const deleted = [];
  Object.assign(c, { allEditableSegments: () => [scene], confirmDestructiveAction: async () => ({ confirmed: true, optionChecked: false }),
    deleteAllTimelineVideosButton: {}, pauseTimelineForEditing() {}, pushHistory() {}, projectInput: { value: '/project' },
    postJson: async (url, data) => { deleted.push(data.path); if (data.path === 'locked.mp4') throw Error('locked'); },
    toast: text => { message = text; },
  });
  vm.runInContext(section('async function deleteAllTimelineVideos()', 'async function deleteAllTimelineImages()'), c);
  await c.deleteAllTimelineVideos();
  assert.deepEqual(deleted, ['locked.mp4', 'old.mp4']);
  assert.equal(scene.video_path, 'locked.mp4');
  assert.equal(scene.video_history.length, 2);
  assert.match(message, /could not be deleted/);
  assert.match(message, /retry/);
  assert.equal(c.deleteAllTimelineVideosButton.disabled, false);
});
