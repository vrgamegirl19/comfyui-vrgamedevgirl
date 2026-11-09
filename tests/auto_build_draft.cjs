const assert = require('node:assert/strict');
const { test } = require('node:test');
const vm = require('node:vm');
const { File } = require('node:buffer');
const fs = require('node:fs');
const path = require('node:path');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');

class Element {
  constructor(tag) { Object.assign(this, { tag, children: [], style: {}, value: '', listeners: {}, disabled: false }); }
  append(...nodes) {
    for (const node of nodes) { this.children.push(node); node.parent = this; }
    if (this.tag === 'select' && !this.value) this.value = this.children[0]?.value || '';
  }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  remove() { this.parent.children = this.parent.children.filter(node => node !== this); this.parent = null; }
}
const walk = node => [node, ...node.children.flatMap(walk)];
function fixture(folder = 'Z:/projects/one', savedDraft = null) {
  const document = { body: new Element('body'), createElement: tag => new Element(tag) };
  const state = { projectFolder: folder, projectVideoEngine: 'minimax_h3', autoBuildDraft: savedDraft, textGemmaRunner: 'llm_api', llmApiKey: 'test' };
  const saves = [], errors = [];
  let fail = false;
  const c = vm.createContext({ document, File, atob, Uint8Array,
    COMPACT_TOOLBAR_ICONS: {}, escapeHtml: value => value,
    makeButton: text => Object.assign(new Element('button'), { textContent: text }),
    makeCheckbox: (text, checked) => {
      const wrapper = Object.assign(new Element('label'), { textContent: text });
      const input = Object.assign(new Element('input'), { checked, type: 'checkbox' });
      wrapper.append(input);
      return { wrapper, input };
    },
    normalizeProjectVideoEngine: value => value,
    readFileAsDataUrl: async file => `data:${file.type};base64,${Buffer.from(await file.arrayBuffer()).toString('base64')}`,
    toast: message => errors.push(message),
  });
  for (const name of ['scenes', 'shot_presets', 'defaults_presets']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/storyboard_builder', `${name}.mjs`), 'utf8')
      .replace(/^import [^;]*;\r?\n/gm, '').replace(/^export /gm, ''), c);
  }
  c.normalizeMiniMaxShortFilmPlanningMode = value => value || 'guided_film';
  for (const name of ['normalizeBuilderStoryboardDefaults', 'builderMotionSpeedGuidance']) {
    vm.runInContext(functionSource(readBuilderModule('model_settings.mjs'), name), c);
  }
  vm.runInContext(readBuilderModule('auto_build_presets.mjs'), c);
  vm.runInContext(readBuilderModule('auto_build_draft.mjs'), c);
  vm.runInContext(readBuilderModule('auto_build.mjs'), c);
  let storyOptions;
  const api = c.createAutoBuild({ syncInspector() {}, render() {}, autoSaveSessionQuiet: async () => {},
    openStoryboardBuilderFromProject: options => { storyOptions = options; }, state, projectInput: { value: folder }, audioInput: { value: '' },
    gemmaRunnerLabel: () => 'LLM API', llmApiVisionModelSelected: () => true,
    saveSession: async () => { if (fail) throw new Error('disk full'); saves.push(JSON.parse(JSON.stringify(state.autoBuildDraft))); return {}; },
  });
  return { c, api, state, storyOptions: () => storyOptions, saves, errors, document, fail: () => { fail = true; },
    nodes: tag => walk(document.body).filter(node => node.tag === tag),
    button: text => walk(document.body).find(node => node.textContent === text),
  };
}

test('closing saves all inputs and reopening a reloaded project restores file contents and options', async () => {
  const f = fixture(); f.api.openAutoBuildModal();
  const files = f.nodes('input').filter(node => node.type === 'file');
  for (let i = 0; i < files.length; i++) {
    files[i].files = [new File([`contents-${i}`], i ? `image-${i}.png` : 'song.wav', { type: i ? 'image/png' : 'audio/wav', lastModified: 42 })];
    files[i].onchange();
  }
  f.nodes('textarea')[0].value = '  My complete lyrics\nsecond line  ';
  f.nodes('select').find(node => node.children.some(option => option.value === 'custom')).value = 'custom';
  f.nodes('input').find(node => node.type === 'number').value = '7.5';
  f.nodes('input').filter(node => node.type === 'checkbox').forEach(node => { node.checked = true; });
  f.nodes('details')[0].open = true;
  await f.button('Close').onclick();
  assert.equal(f.document.body.children.length, 0);
  const saved = f.saves[0];
  assert.equal(saved.lyrics, '  My complete lyrics\nsecond line  ');
  assert.equal(saved.use_current_settings, true);
  assert.equal(saved.skip_prompts, true);
  const reloaded = fixture('Z:/projects/one', JSON.parse(JSON.stringify(saved)));
  reloaded.api.openAutoBuildModal();
  assert.equal(reloaded.nodes('textarea')[0].value, saved.lyrics);
  assert.equal(reloaded.nodes('select').find(node => node.children.some(option => option.value === 'custom')).value, 'custom');
  assert.equal(reloaded.nodes('input').find(node => node.type === 'number').value, '7.5');
  assert.ok(walk(reloaded.document.body).some(node => node.textContent === 'song.wav'));
  await reloaded.button('Cancel').onclick();
  assert.deepEqual(reloaded.saves[0], saved);
  const restored = f.c.restoreAutoBuildFile(saved.song);
  assert.equal(await restored.text(), 'contents-0');
  assert.equal(restored.name, 'song.wav');
  assert.equal(restored.lastModified, 42);
});

test('drafts stay within their project and an older project has no inherited inputs', () => {
  const f = fixture('Z:/projects/two', { project_folder: 'Z:/projects/one', lyrics: 'private to one' });
  f.api.openAutoBuildModal();
  assert.equal(f.nodes('textarea')[0].value, '');
  assert.equal(f.c.autoBuildDraftForProject({ project_folder: 'Z:\\projects\\one\\' }, 'z:/projects/one').project_folder, 'Z:\\projects\\one\\');
});

test('a failed save keeps the modal and its inputs available', async () => {
  const f = fixture(); f.api.openAutoBuildModal(); f.fail();
  f.nodes('textarea')[0].value = 'Do not lose this';
  await f.button('Close').onclick();
  assert.equal(f.document.body.children.length, 1);
  assert.equal(f.nodes('textarea')[0].value, 'Do not lose this');
  assert.match(f.errors[0], /disk full/);
  assert.equal(f.button('Close').disabled, false);
});

test('backdrop dismissal saves a draft and cannot dismiss a running build', async () => {
  const f = fixture(); f.api.openAutoBuildModal();
  const backdrop = f.document.body.children[0];
  f.button('Build My Music Video').disabled = true;
  backdrop.listeners.pointerdown({ target: backdrop });
  assert.equal(f.saves.length, 0);
  f.button('Build My Music Video').disabled = false;
  backdrop.listeners.pointerdown({ target: backdrop });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.saves.length, 1);
  assert.equal(f.document.body.children.length, 0);
});


test('opening Story Board applies Medium shots before packaging scenes', () => {
  const f = fixture();
  f.state.segments = [{ id: 'one', shot_type: '', camera_motion: '' }, { id: 'two' }];
  f.api.openAutoBuildModal();
  f.button('Go to Story Board').onclick();
  assert.equal(f.state.builderStoryboardDefaults.camera_flow, 'medium_shots');
  assert.ok(f.state.segments.every(scene => scene.shot_type && scene.camera_motion));
  assert.match(f.state.segments[0].shot_type, /waist up/);
  assert.equal(f.state.segments[0].scene_defaults.camera_flow, 'medium_shots');
  assert.ok(f.nodes('input').find(node => node.parent.textContent === 'Use current Scene Defaults and story settings').checked);
});

test('Speed and Music style combine and synchronize scene overrides and Story Board labels', async () => {
  const f = fixture();
  f.state.segments = [{ scene_defaults: { camera_motion_speed: 9, camera_flow: 'intimate_closeups', video_style: 'custom', fx_preset: 'old' }, story_beat: 'Keep this' }];
  f.api.openAutoBuildModal();
  const speed = f.nodes('select').find(node => node['aria-label'] === 'Speed');
  const music = f.nodes('select').find(node => node['aria-label'] === 'Music style');
  speed.value = 'slow'; await speed.onchange();
  music.value = 'rap'; await music.onchange();
  const defaults = f.state.builderStoryboardDefaults, scene = f.state.segments[0];
  assert.equal(defaults.defaults_speed_preset, 'slow');
  assert.equal(defaults.defaults_music_preset, 'rap');
  assert.equal(defaults.camera_flow, 'medium_shots');
  assert.equal(defaults.camera_motion_speed, 2);
  assert.equal(defaults.minimax_h3_cut_frequency, 1);
  assert.equal(scene.scene_defaults.camera_motion_speed, 2);
  assert.equal(scene.scene_defaults.video_style, '');
  assert.equal(scene.scene_defaults.fx_preset, '');
  assert.equal(scene.performance_style, 'rap_hiphop');
  assert.equal(scene.facial_performance, 'rap_high_intensity');
  assert.equal(scene.story_beat, 'Keep this');
  assert.match(scene.character_motion, /2\/10/);
  f.button('Go to Story Board').onclick();
  f.state.builderStoryboardDefaults.defaults_speed_preset = 'fast';
  f.storyOptions().onClose();
  assert.equal(speed.value, 'fast');
});

test('every Auto Build preset uses the shared Story Board definitions in either order', () => {
  const f = fixture();
  for (const speed of ['static', 'slow', 'medium', 'fast']) {
    for (const music of ['rock', 'metal', 'rap', 'soft', 'pop']) {
      const a = { segments: [{}] }, b = { segments: [{}] };
      f.c.applyAutoBuildPresets(a, { recommended: true, speed, music });
      f.c.applyAutoBuildPresets(b, { recommended: true, music });
      f.c.applyAutoBuildPresets(b, { speed });
      assert.deepEqual(a.builderStoryboardDefaults, b.builderStoryboardDefaults);
      assert.equal(a.segments[0].performance_style, b.segments[0].performance_style);
      assert.equal(a.segments[0].facial_performance, b.segments[0].facial_performance);
      assert.ok(a.segments[0].shot_type);
    }
  }
});

test('preserved defaults fill new scenes without overwriting reviewed scene framing', () => {
  const f = fixture();
  const state = { segments: [{}] };
  f.c.applyAutoBuildPresets(state, { recommended: true, music: 'soft', speed: 'fast' });
  state.segments[0].shot_type = 'My reviewed shot';
  state.segments[0].camera_motion = 'My reviewed move';
  state.segments.push({});
  f.c.fillAutoBuildSceneFraming(state);
  assert.equal(state.segments[0].shot_type, 'My reviewed shot');
  assert.equal(state.segments[0].camera_motion, 'My reviewed move');
  assert.ok(state.segments[1].shot_type);
  assert.equal(state.builderStoryboardDefaults.defaults_speed_preset, 'fast');
});

test('preserving empty defaults still records Medium shots, while explicit Off stays Off', () => {
  const f = fixture();
  const fresh = { segments: [{}] };
  f.c.fillAutoBuildSceneFraming(fresh);
  assert.equal(fresh.builderStoryboardDefaults.camera_flow, 'medium_shots');
  assert.ok(fresh.segments[0].shot_type);
  const off = { builderStoryboardDefaults: { camera_flow: 'off' }, segments: [{}] };
  f.c.fillAutoBuildSceneFraming(off);
  assert.equal(off.builderStoryboardDefaults.camera_flow, 'off');
  assert.equal(off.segments[0].shot_type, '');
});
