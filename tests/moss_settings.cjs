const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// The settings MOSS-Image is run with (web/music_video_builder/moss_settings.mjs): shared by the Reference Builder, the Thumbnail dialog
// and the MOSS-Image tab of Models > Select models.

const context = vm.createContext({});
vm.runInContext(readBuilderModule('moss_settings.mjs'), context);
const run = (code) => JSON.parse(JSON.stringify(vm.runInContext(code, context)));

function fakeStorage(initial = {}) {
  const data = { ...initial };
  return { data, getItem: (key) => (key in data ? data[key] : null), setItem: (key, value) => { data[key] = String(value); } };
}
const KEY = vm.runInContext('MOSS_REFERENCE_STORAGE_KEY', context);

test('the settings are kept under the key the Reference Builder has always used', () => {
  assert.equal(KEY, 'vrgdg_moss_image_reference_settings');
});

test('a fresh install gets the Turbo preset at 1920x1088 with the Edit LoRA name', () => {
  const defaults = run('DEFAULT_MOSS_REFERENCE_SETTINGS');
  assert.equal(defaults.preset, 'turbo');
  assert.equal(defaults.steps, 8);
  assert.equal(defaults.cfg, 1);
  assert.equal(defaults.lora_name, 'moss_image_sft_8_steps_dmd_lora.safetensors');
  assert.equal(defaults.edit_lora_name, 'moss_image_edit_lora.safetensors');
  assert.equal(defaults.width, 1920);
  assert.equal(defaults.height, 1088);
});

test('saved settings go over the defaults and anything unusable gives the defaults', () => {
  context.storage = fakeStorage({ [KEY]: JSON.stringify({ steps: 20, preset: 'custom' }) });
  const saved = run('loadMossReferenceSettings(storage)');
  assert.equal(saved.steps, 20);
  assert.equal(saved.preset, 'custom');
  assert.equal(saved.unet_name, 'moss_image_sft_bf16.safetensors', 'what was not saved keeps its default');
  for (const bad of ['not json', '[1,2]', 'null', '7']) {
    context.storage = fakeStorage({ [KEY]: bad });
    assert.equal(run('loadMossReferenceSettings(storage)').steps, 8, `"${bad}" gives the defaults`);
  }
  context.storage = null;
  assert.equal(run('loadMossReferenceSettings(storage)').steps, 8, 'no storage gives the defaults');
  context.storage = { getItem() { throw new Error('blocked'); } };
  assert.equal(run('loadMossReferenceSettings(storage)').steps, 8, 'blocked storage gives the defaults');
});

test('saving writes the settings and reports a browser that keeps nothing', () => {
  context.storage = fakeStorage();
  context.settings = { steps: 12 };
  assert.equal(run('saveMossReferenceSettings(settings, storage)'), true);
  assert.deepEqual(JSON.parse(context.storage.data[KEY]), { steps: 12 });
  context.storage = { setItem() { throw new Error('full'); } };
  assert.equal(run('saveMossReferenceSettings(settings, storage)'), false);
});

test('sizes are made a multiple of 16 between 256 and 4096', () => {
  assert.equal(run('mossSize(1000, 1)'), 1008);
  assert.equal(run('mossSize(1920, 1)'), 1920);
  assert.equal(run('mossSize(100, 1)'), 256);
  assert.equal(run('mossSize(9000, 1)'), 4096);
  for (const bad of ['0', '-5', "'abc'", 'NaN']) assert.equal(run(`mossSize(${bad}, 1088)`), 1088, `${bad} gives the fallback`);
});

test('normalizing clamps what is out of range and falls back for what is empty or wrong', () => {
  const settings = run(`normalizeMossSettings({
    unet_name: '  ', clip_name: ' my_clip.safetensors ', vae_name: '', lora_name: '', edit_lora_name: null, preset: 'bogus',
    steps: 999, cfg: 99, width: 1001, height: 'x', seed: -4, seed_mode: 'fixed',
  })`);
  assert.equal(settings.unet_name, 'moss_image_sft_bf16.safetensors');
  assert.equal(settings.clip_name, 'my_clip.safetensors', 'names are trimmed');
  assert.equal(settings.vae_name, 'flux2-vae.safetensors');
  assert.equal(settings.lora_name, '[none]', 'an empty LoRA means none');
  assert.equal(settings.edit_lora_name, 'moss_image_edit_lora.safetensors');
  assert.equal(settings.preset, 'custom', 'an unknown preset is custom');
  assert.equal(settings.steps, 200);
  assert.equal(settings.cfg, 20);
  assert.equal(settings.width, 1008);
  assert.equal(settings.height, 1088);
  assert.equal(settings.seed, 0);
  assert.equal(settings.seed_mode, 'fixed');
  assert.equal(run("normalizeMossSettings({ seed_mode: 'whatever' }).seed_mode"), 'random');
  assert.equal(run('normalizeMossSettings(null).steps'), 8);
  assert.equal(run('normalizeMossSettings([1]).steps'), 8);
});

test('the Reference Builder request carries the saved settings and a trimmed prompt', () => {
  const payload = run(`mossReferencePayload('  a red door  ', { unet_name: 'u.safetensors', steps: 0, cfg: 0, width: 0, height: 0, seed: 0, lora_name: '' })`);
  assert.equal(payload.prompt, 'a red door');
  assert.equal(payload.unet_name, 'u.safetensors');
  assert.equal(payload.lora_name, '[none]');
  assert.equal(payload.steps, 8, 'zero steps fall back');
  assert.equal(payload.cfg, 1);
  assert.equal(payload.width, 1920);
  assert.equal(payload.height, 1088);
  assert.equal(payload.seed_mode, 'random');
  assert.equal(payload.batch_size, 1);
});
