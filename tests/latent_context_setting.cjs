const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../web/VRGDG_MusicVideoBuilderUI.js'), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const getter = section('  function miniMaxH3SettingsForSegment(', '  function miniMaxH3ModeForSegment(');
const saver = section('  function saveMiniMaxH3SettingsFromPanel(', '  function saveMiniMaxSceneInputsFromPanel(');
const bindingEnd = source.indexOf('    control.addEventListener("input", saveMiniMaxH3SettingsFromPanel);');
const bindings = source.slice(source.lastIndexOf('  for (const control of [', bindingEnd), source.indexOf('\n  }', bindingEnd) + 4);
const panelLine = source.match(/miniMaxLatentContextFrames.value = String\([^\n]+/)[0];
const renderLine = source.match(/const latentContextFrames = [\s\S]*?;/)[0];

function fixture() {
  const control = () => {
    const events = {};
    return { value: '', checked: false, events, addEventListener: (name, fn) => { events[name] = fn; } };
  };
  const context = { state: { miniMaxH3Settings: { latent_context_frames: 22, video_mode: 'text_to_video' } }, saved: 0 };
  for (const name of new Set((saver + bindings).match(/\bminiMax[A-Z]\w*/g))) {
    context[name] = { ...control(), input: control(), dataset: {} };
  }
  Object.assign(context, {
    DEFAULT_MINIMAX_H3_SETTINGS: { latent_context_frames: 22 },
    miniMaxLoraSlots: [], miniMaxAccelerationControls: [], twoPassControls: [], advancedTwoPassControls: [],
    selected: { id: 'a', minimax_h3_latent_context_frames: 22 },
    activeSegment: () => context.selected,
    cloneMiniMaxH3Settings: settings => ({ latent_context_frames: 22, ...settings }),
    normalizeMiniMaxH3Mode: value => value || 'text_to_video',
    normalizeMiniMaxH3LocationTransitionPreset: () => 'normal',
    autoSaveSessionQuiet: async () => { context.saved++; },
  });
  vm.createContext(context);
  vm.runInContext(getter + saver
    + section('  const persistMiniMaxSettings =', '  for (const picker of [miniMaxDiffusionModelPicker')
    + bindings, context);
  return {
    context,
    choose(value) { context.miniMaxLatentContextFrames.value = String(value); context.miniMaxLatentContextFrames.events.change(); },
    display() { return vm.runInContext(`{ const segment = activeSegment(); const settings = miniMaxH3SettingsForSegment(segment); ${panelLine} miniMaxLatentContextFrames.value; }`, context); },
    renderValue() { return vm.runInContext(`{ const segment = activeSegment(); const miniMaxSettings = miniMaxH3SettingsForSegment(segment); ${renderLine} latentContextFrames; }`, context); },
  };
}

test('changing context persists immediately and survives selecting another scene', () => {
  const f = fixture();
  assert.equal(typeof f.context.miniMaxLatentContextFrames.events.input, 'function');
  f.choose(39);
  assert.equal(f.context.saved, 1);
  f.context.selected = { id: 'b', minimax_h3_latent_context_frames: 22 };
  assert.equal(f.display(), '39');
  assert.equal(f.renderValue(), 39);
  f.context.selected = { id: 'a', minimax_h3_latent_context_frames: 22 };
  assert.equal(f.display(), '39');
});

test('scene overrides stay independent from project settings', () => {
  const f = fixture(); f.choose(39);
  const scene = { id: 'b', minimax_h3_latent_context_frames: 22, use_scene_minimax_h3_settings: true,
    minimax_h3_settings: { latent_context_frames: 56 } };
  f.context.selected = scene;
  assert.equal(f.display(), '56');
  f.choose(16);
  assert.equal(f.renderValue(), 16);
  assert.equal(f.context.state.miniMaxH3Settings.latent_context_frames, 39);
  f.context.selected = { id: 'a' };
  assert.equal(f.display(), '39');
  f.context.selected = scene;
  assert.equal(f.display(), '16');
});

test('saved project and scene settings survive serialization', () => {
  const f = fixture(); f.choose(56);
  f.context.state = JSON.parse(JSON.stringify(f.context.state));
  f.context.selected = { id: 'loaded', minimax_h3_latent_context_frames: 22 };
  assert.equal(f.display(), '56');
  assert.equal(f.renderValue(), 56);
});
