const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

function fixture() {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  vm.runInContext(readBuilderModule('video_mode_resume.mjs'), c);
  const values = new Map();
  const storage = { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  const state = () => ({ projectFolder: 'Z:\\projects\\band', projectVideoEngine: 'minimax_h3',
    miniMaxH3Settings: c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video' }),
    segments: [{ id: 'first' }, { id: 'last', use_scene_minimax_h3_settings: true,
      minimax_h3_settings: c.cloneMiniMaxH3Settings({ video_mode: 'image_to_video' }) }],
    overlaySegments: [], activeId: 'first' });
  return { c, storage, state };
}

for (const mode of ['text_to_video', 'image_to_video', 'image_reference_to_video', 'reference_to_video', 'video_to_video']) {
  test(`browser restart restores unsaved ${mode} and the selected locked scene`, () => {
    const { c, storage, state } = fixture();
    const current = state();
    c.restoreVideoMode(storage, current);
    current.miniMaxH3Settings = c.cloneMiniMaxH3Settings({ video_mode: mode,
      render_pass: mode === 'image_reference_to_video' ? 'two_pass' : 'single' });
    current.segments[1].minimax_h3_settings.video_mode = mode;
    current.activeId = 'last';
    c.rememberVideoMode(storage, current);
    const reopened = state();
    assert.equal(c.restoreVideoMode(storage, reopened), 'last');
    assert.equal(reopened.miniMaxH3Settings.video_mode, mode);
    assert.equal(reopened.segments[1].minimax_h3_settings.video_mode, mode);
    assert.equal(reopened.miniMaxH3Settings.render_pass, current.miniMaxH3Settings.render_pass);
  });
}

test('a successful save updates the baseline before a later unsaved mode switch', () => {
  const { c, storage, state } = fixture();
  const current = state();
  c.restoreVideoMode(storage, current);
  current.miniMaxH3Settings.video_mode = 'image_to_video';
  c.recordSavedVideoMode(current, { projectFolder: current.projectFolder,
    session: { video_engine: current.projectVideoEngine, minimax_h3_settings: current.miniMaxH3Settings,
      segments: current.segments, overlay_segments: [] } });
  const saved = JSON.parse(JSON.stringify(current));
  current.miniMaxH3Settings.video_mode = 'image_reference_to_video';
  current.miniMaxH3Settings.render_pass = 'two_pass';
  c.rememberVideoMode(storage, current);
  c.restoreVideoMode(storage, saved);
  assert.equal(saved.miniMaxH3Settings.video_mode, 'image_reference_to_video');
});

test('newer saved modes and other projects take precedence over a local checkpoint', () => {
  const { c, storage, state } = fixture();
  const current = state();
  c.restoreVideoMode(storage, current);
  current.miniMaxH3Settings.video_mode = 'image_reference_to_video';
  c.rememberVideoMode(storage, current);
  const newer = state();
  newer.miniMaxH3Settings.video_mode = 'video_to_video';
  assert.equal(c.restoreVideoMode(storage, newer), '');
  assert.equal(newer.miniMaxH3Settings.video_mode, 'video_to_video');
  const other = state();
  other.projectFolder += '-other';
  assert.equal(c.restoreVideoMode(storage, other), '');
  assert.equal(other.miniMaxH3Settings.video_mode, 'reference_to_video');
});

test('blocked browser storage leaves saved settings intact', () => {
  const { c, state } = fixture();
  const blocked = { getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); } };
  const current = state();
  assert.equal(c.restoreVideoMode(blocked, current), '');
  c.rememberVideoMode(blocked, current);
  assert.equal(current.miniMaxH3Settings.video_mode, 'reference_to_video');
});

test('explicit save invalidates an older closing checkpoint even when returning to its original mode', () => {
  const { c, storage, state } = fixture();
  const current = state();
  c.restoreVideoMode(storage, current);
  current.miniMaxH3Settings.video_mode = 'image_reference_to_video';
  c.rememberVideoMode(storage, current);
  current.miniMaxH3Settings.video_mode = 'reference_to_video';
  c.recordSavedVideoMode(current, { projectFolder: current.projectFolder,
    session: { video_engine: current.projectVideoEngine, minimax_h3_settings: current.miniMaxH3Settings,
      segments: current.segments, overlay_segments: [] } }, storage);
  const reopened = state();
  assert.equal(c.restoreVideoMode(storage, reopened), '');
  assert.equal(reopened.miniMaxH3Settings.video_mode, 'reference_to_video');
});

test('project load restores selection after mode checkpoint and honors preserveSelection', () => {
  const { c, storage, state } = fixture();
  c.state = state();
  c.state.activeId = 'last';
  c.localStorage = storage;
  c.session = { active_scene_id: 'last' };
  c.options = {};
  c.previousActiveId = 'first';
  const source = readBuilderModule('session.mjs');
  const start = source.indexOf('      state.activeTrack = session.active_track || "base";');
  const end = source.indexOf('      syncZImageSettingsPanel();', start);
  vm.runInContext(`{${source.slice(start, end)}}`, c);
  assert.equal(c.state.activeId, 'last');
  c.options.preserveSelection = true;
  vm.runInContext(`{${source.slice(start, end)}}`, c);
  assert.equal(c.state.activeId, 'first');
  c.options.preserveSelection = false;
  c.session.active_scene_id = 'deleted-scene';
  vm.runInContext(`{${source.slice(start, end)}}`, c);
  assert.equal(c.state.activeId, 'first');
});

test('pass profiles preserve Image + Reference mode through JSON reload', () => {
  const { c } = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'image_reference_to_video', render_pass: 'two_pass',
    ref_pass_profiles: { two_pass: { video_mode: 'reference_to_video' } } });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings))).video_mode, 'image_reference_to_video');
});
