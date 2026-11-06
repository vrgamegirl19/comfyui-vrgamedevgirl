import { test } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { assertMiniMaxLorasInstalled, miniMaxPickerLoraName, syncInstalledMiniMaxLoraPicker } from '../web/music_video_builder/minimax_lora_validation.mjs';
import { MINIMAX_H3_DMAD_LORA_NAME as DMAD } from '../web/music_video_builder/minimax_h3.mjs';
const { functionSource, readBuilderModule } = createRequire(import.meta.url)('./builder_source.cjs');
const refiner = 'refiner.safetensors';
const settings = (overrides = {}) => ({ use_dmad: true, use_loras: true, lora_count: 1, render_pass: 'two_pass', two_pass_lora_name: refiner, loras: [{ name: DMAD, strength: 1, apply_to: 'pass1' }], ...overrides });

test('missing selections cannot appear as valid picker choices or disappear from saved settings', () => {
  const picker = { input: { style: {}, value: '', setCustomValidity(message) { this.validationMessage = message; } }, options: ['[none]', 'installed.safetensors'] };
  syncInstalledMiniMaxLoraPicker(picker, DMAD);
  assert.equal(picker.input.value, '');
  assert.match(picker.input.placeholder, /Missing installed LoRA/);
  assert.ok(picker.input.validationMessage);
  assert.equal(picker.options.includes(DMAD), false);
  assert.equal(miniMaxPickerLoraName(picker), DMAD);
  picker.options.push(DMAD);
  syncInstalledMiniMaxLoraPicker(picker);
  assert.equal(picker.input.value, DMAD);
  assert.equal(picker.input.validationMessage, '');
  picker.input.value = '[none]';
  syncInstalledMiniMaxLoraPicker(picker);
  assert.equal(miniMaxPickerLoraName(picker), '[none]');
});

test('DMAD refuses unavailable, disabled, wrong-pass, absent and zero-strength adapters', () => {
  assert.throws(() => assertMiniMaxLorasInstalled(settings(), ['dmad_minimax_h3_4step_full_critic.safetensors', refiner]), /DMAD rendering requires/);
  for (const patch of [{ use_loras: false }, { lora_count: 0 }, { loras: [] }, { loras: [{ name: DMAD, apply_to: 'pass2' }] }, { loras: [{ name: DMAD, strength: 0, apply_to: 'pass1' }] }]) {
    assert.throws(() => assertMiniMaxLorasInstalled(settings(patch), [DMAD, refiner]), /DMAD rendering requires/);
  }
  assert.doesNotThrow(() => assertMiniMaxLorasInstalled(settings(), [DMAD, refiner]));
  assert.throws(() => assertMiniMaxLorasInstalled(settings(), [DMAD]), /not installed/);
  assert.doesNotThrow(() => assertMiniMaxLorasInstalled({ use_dmad: false, use_loras: false, render_pass: 'single' }, []));
});

test('single-scene rendering refuses missing DMAD before audio preparation or queueing', async () => {
  const scene = { id: 'scene1' };
  const c = vm.createContext({
    activeSegment: () => scene,
    saveMiniMaxH3SettingsFromPanel() {},
    miniMaxH3SettingsForSegment: () => settings(),
    getJson: async () => ({ loras: [refiner] }),
    assertMiniMaxLorasInstalled,
    prepareEditedAudio: () => { assert.fail('audio preparation must not run'); },
  });
  vm.runInContext(`${functionSource(readBuilderModule('video_render.mjs'), 'renderMiniMaxSceneVideoWithProgress')}; globalThis.renderScene = renderMiniMaxSceneVideoWithProgress;`, c);
  await assert.rejects(c.renderScene(scene, 0, null), /DMAD rendering requires/);
});

test('Render All refuses missing DMAD in a locked scene before starting the batch', async () => {
  let message = '';
  const c = vm.createContext({
    updateActiveFromInputs() {}, saveI2VVideoSettingsFromPanel() {}, saveMiniMaxH3SettingsFromPanel() {},
    state: { projectVideoEngine: 'minimax_h3' }, normalizeProjectVideoEngine: value => value,
    normalizeBatchScope: () => 'all', batchTargetItems: () => [{ segment: { id: 'locked' } }],
    selectedSegmentVideoPath: () => '', miniMaxH3SettingsForSegment: () => settings(),
    getJson: async () => ({ loras: [refiner] }), assertMiniMaxLorasInstalled,
    toast: value => { message = value; },
    createProgressWindow: () => { assert.fail('batch must not start'); },
  });
  vm.runInContext(`${functionSource(readBuilderModule('batch_render.mjs'), 'renderAllScenes')}; globalThis.renderAll = renderAllScenes;`, c);
  await c.renderAll();
  assert.match(message, /DMAD rendering requires/);
});
