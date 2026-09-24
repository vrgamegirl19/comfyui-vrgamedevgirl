const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../web/VRGDG_MusicVideoBuilderUI.js'), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function fixture() {
  const c = vm.createContext({});
  vm.runInContext(section('const MINIMAX_H3_MODE_OPTIONS =', 'const LTX_23_MODEL_DOWNLOADS ='), c);
  return c;
}
test('all three mode profiles survive switching and project JSON reload', () => {
  const c = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', ref_pass_mode: 'single', steps: 17, use_te_speed: true, ref_image_size: 'match' });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  settings.pass1_use_feedforward = true;
  settings.pass2_use_feedforward = false;
  settings.two_pass_pass1_steps = 25;
  settings.two_pass_use_fast_vae_decode = true;
  settings = c.selectMiniMaxH3PassSettings(settings, 'advanced');
  settings.pass1_use_feedforward = false;
  settings.pass2_use_feedforward = true;
  settings.advanced_two_pass_pass2_steps = 3;
  settings.two_pass_use_fast_vae_decode = false;
  settings = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings)));
  assert.equal(settings.ref_pass_mode, 'advanced');
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  assert.equal(settings.steps, 17);
  assert.equal(settings.ref_image_size, 'match');
  assert.equal(settings.use_te_speed, true);
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(settings.two_pass_pass1_steps, 25);
  assert.equal(settings.pass1_use_feedforward, true);
  assert.equal(settings.pass2_use_feedforward, false);
  assert.equal(settings.two_pass_use_fast_vae_decode, true);
  settings = c.selectMiniMaxH3PassSettings(settings, 'advanced');
  assert.equal(settings.advanced_two_pass_pass2_steps, 3);
  assert.equal(settings.pass1_use_feedforward, false);
  assert.equal(settings.pass2_use_feedforward, true);
  assert.equal(settings.two_pass_use_fast_vae_decode, false);
  for (const profile of Object.values(settings.ref_pass_profiles)) assert.equal(profile.ref_pass_profiles, undefined);
});
test('reference mode ignores the retired Turbo toggle and leaves steps editable', () => {
  const c = fixture();
  const settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', use_turbo_lora: true, use_loras: true, steps: 19, loras: [{ name: 'turbo.safetensors', strength: 1 }] });
  assert.equal(settings.use_turbo_lora, false);
  assert.equal(settings.use_loras, true);
  assert.equal(settings.steps, 19);
  assert.equal(settings.loras[0].name, 'turbo.safetensors');
});
test('pass selector has three exclusive buttons outside the main mode grid', async () => {
  const c = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', ref_pass_mode: 'single' });
  const segment = { id: 'scene' };
  Object.assign(c, { miniMaxPassButtons: ['single', 'two_pass', 'advanced'].map(passMode => ({ dataset: { passMode } })),
    requireActiveSegment: () => segment, pushHistory() {}, state: {},
    saveMiniMaxH3SettingsFromPanel: () => settings,
    syncMiniMaxH3Panel() { settings = c.state.miniMaxH3Settings; },
    autoSaveSessionQuiet: async () => {},
  });
  const start = source.indexOf('  for (const button of miniMaxPassButtons) {', source.indexOf('  miniMaxUseTurboLora.input.addEventListener'));
  const end = source.indexOf('  miniMaxAudioMode.addEventListener', start);
  vm.runInContext(source.slice(start, end), c);
  for (const button of c.miniMaxPassButtons) {
    await button.onclick();
    assert.equal(settings.ref_pass_mode, button.dataset.passMode);
    assert.equal(c.miniMaxPassButtons.filter(b => b.dataset.passMode === settings.ref_pass_mode).length, 1);
  }
  assert.ok(source.includes('miniMaxModeChooser, miniMaxPassChooser, miniMaxSubTabs.wrapper'));
  assert.ok(!source.includes('miniMaxModeChooser.append(miniMaxTwoPassButton)'));
});

test('new two-pass settings default to two refinement steps, random seeds and acceleration off', () => {
  const c = fixture();
  const settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video' });
  assert.equal(settings.two_pass_pass2_steps, 2);
  assert.equal(settings.two_pass_pass1_seed, -1);
  assert.equal(settings.two_pass_pass2_seed, -1);
  for (const mode of ['single', 'two_pass', 'advanced']) {
    const selected = c.selectMiniMaxH3PassSettings(settings, mode);
    for (const key of ['te_speed', 'feedforward', 'block_sparse_attention']) {
      assert.equal(Boolean(selected[`use_${key}`]), false);
      for (const pass of [1, 2]) assert.equal(Boolean(selected[`pass${pass}_use_${key}`] ?? selected[`two_pass_use_${key}`]), false);
    }
    assert.equal(selected.use_fast_vae_decode, false);
    assert.equal(selected.two_pass_use_fast_vae_decode, false);
  }
  const saved = c.cloneMiniMaxH3Settings({ ...settings, two_pass_pass2_steps: 7, two_pass_pass1_seed: 123, pass1_use_te_speed: true });
  assert.equal(saved.two_pass_pass2_steps, 7);
  assert.equal(saved.two_pass_pass1_seed, 123);
  assert.equal(saved.pass1_use_te_speed, true);
});
