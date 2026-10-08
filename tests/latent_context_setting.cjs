const { functionSource, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = readBuilderSource();
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const getter = functionSource(source, 'miniMaxH3SettingsForSegment');
const saver = functionSource(source, 'saveMiniMaxH3SettingsFromPanel');
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
    miniMaxKeyframes: {transitionStyle: {...control(), value: "natural"}, transitionDirection: control()},
    wizardVideoSettings: { global: false },
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
    + bindings
    + section('  for (const control of [miniMaxKeyframes.transitionStyle', '  miniMaxUseLoras.input.addEventListener'), context);
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


test('transition event handlers save globally, lock overrides, and reload without losing either scope', () => {
  const f = fixture();
  const c = f.context;
  c.miniMaxKeyframes.transitionStyle.value = 'surreal_morph';
  c.miniMaxKeyframes.transitionDirection.value = 'petals become stars';
  c.miniMaxKeyframes.transitionStyle.events.change();
  assert.equal(c.saved, 1);
  assert.equal(c.state.miniMaxH3Settings.i2v_transition_style, 'surreal_morph');
  assert.equal(c.selected.minimax_h3_settings, undefined);
  c.selected = {id:'locked',use_scene_minimax_h3_settings:true,minimax_h3_settings:{}};
  c.miniMaxKeyframes.transitionStyle.value = 'camera_reveal';
  c.miniMaxKeyframes.transitionDirection.value = 'orbit right';
  c.miniMaxKeyframes.transitionDirection.events.change();
  assert.equal(c.saved, 2);
  c.state = JSON.parse(JSON.stringify(c.state));
  c.selected = JSON.parse(JSON.stringify(c.selected));
  assert.equal(c.miniMaxH3SettingsForSegment(c.selected).i2v_transition_direction, 'orbit right');
  assert.equal(c.state.miniMaxH3Settings.i2v_transition_direction, 'petals become stars');
  c.selected.use_scene_minimax_h3_settings = false;
  assert.equal(c.miniMaxH3SettingsForSegment(c.selected).i2v_transition_style, 'surreal_morph');
});

test('global wizard transition edits target project settings even with a locked selected scene', () => {
  const c = fixture().context;
  c.wizardVideoSettings.global = true;
  c.selected = {id:'locked',use_scene_minimax_h3_settings:true,minimax_h3_settings:{i2v_transition_style:'custom'}};
  c.miniMaxKeyframes.transitionStyle.value = 'dreamlike_dissolve';
  c.miniMaxKeyframes.transitionStyle.events.change();
  assert.equal(c.state.miniMaxH3Settings.i2v_transition_style, 'dreamlike_dissolve');
  assert.equal(c.selected.minimax_h3_settings.i2v_transition_style, 'custom');
});
