const { readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');

const DMAD_LORA = 'dmad_minimax_h3_4step_full_critic_comfyui.safetensors';
const REFINER = 'minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors';

function fixture() {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  return c;
}
// Objects made inside the vm context have another Object.prototype, so compare through JSON.
const plain = (value) => JSON.parse(JSON.stringify(value));

test('DMAD fills in the LoRAs, the sampler settings and the sigma shift', () => {
  const c = fixture();
  const on = plain(c.setMiniMaxH3Dmad(c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'two_pass' }), true));
  assert.equal(on.use_dmad, true);
  assert.deepEqual(on.loras, [{ name: DMAD_LORA, strength: 1, apply_to: 'pass1' }]);
  assert.equal(on.use_loras, true);
  assert.equal(on.lora_count, 1);
  assert.equal(on.two_pass_lora_name, REFINER);
  assert.equal(on.two_pass_lora_strength, 0.7);
  assert.equal(on.two_pass_pass1_sampler, 'lcm');
  assert.equal(on.two_pass_pass1_scheduler, 'simple');
  assert.equal(on.two_pass_pass1_steps, 4);
  assert.equal(on.two_pass_pass2_steps, 3);
  assert.equal(on.two_pass_pass2_denoise, 0.2);
  assert.equal(on.sampler_name, 'lcm');
  assert.equal(on.steps, 4);
  assert.deepEqual([on.sigma_pass1_enabled, on.sigma_pass1_video, on.sigma_pass1_audio], [true, 12, 2]);
  assert.equal(on.sigma_pass2_enabled, false);
});

test('Standard puts back what DMAD replaced and keeps LoRAs added meanwhile', () => {
  const c = fixture();
  const before = c.cloneMiniMaxH3Settings({
    video_mode: 'reference_to_video', render_pass: 'two_pass', steps: 17, sampler_name: 'euler', two_pass_pass1_steps: 25,
    two_pass_lora_name: 'my_turbo.safetensors', two_pass_lora_strength: 0.9,
    use_loras: true, lora_count: 1, loras: [{ name: 'style.safetensors', strength: 0.4, apply_to: 'both' }],
    sigma_pass2_enabled: true, sigma_pass2_video: 7, sigma_pass2_audio: 1.5,
  });
  const on = c.setMiniMaxH3Dmad(before, true);
  assert.deepEqual(plain(on.loras.map((item) => item.name)), [DMAD_LORA, 'style.safetensors']);
  assert.equal(on.lora_count, 2);
  const off = plain(c.setMiniMaxH3Dmad(on, false));
  assert.equal(off.use_dmad, false);
  assert.deepEqual(off.dmad_backup, {});
  assert.deepEqual(off.loras, [{ name: 'style.safetensors', strength: 0.4, apply_to: 'both' }]);
  assert.equal(off.use_loras, true);
  assert.equal(off.lora_count, 1);
  for (const key of ['steps', 'sampler_name', 'two_pass_pass1_steps', 'two_pass_lora_name', 'two_pass_lora_strength',
    'sigma_pass1_enabled', 'sigma_pass2_enabled', 'sigma_pass2_video', 'sigma_pass2_audio']) {
    assert.equal(off[key], plain(before)[key], key);
  }
  // A LoRA added while DMAD was on stays when DMAD goes.
  const added = plain(on);
  added.loras.push({ name: 'later.safetensors', strength: 1, apply_to: 'pass2' });
  added.lora_count = 3;
  const kept = plain(c.setMiniMaxH3Dmad(added, false));
  assert.deepEqual(kept.loras.map((item) => item.name), ['style.safetensors', 'later.safetensors']);
});

test('Standard with no extra LoRAs before leaves the LoRA list off', () => {
  const c = fixture();
  const off = plain(c.setMiniMaxH3Dmad(c.setMiniMaxH3Dmad(c.cloneMiniMaxH3Settings({}), true), false));
  assert.equal(off.use_loras, false);
  assert.equal(off.lora_count, 0);
  assert.deepEqual(off.loras, []);
});

test('turning DMAD on twice or off twice changes nothing', () => {
  const c = fixture();
  const on = c.setMiniMaxH3Dmad(c.cloneMiniMaxH3Settings({ render_pass: 'single', video_mode: 'reference_to_video' }), true);
  assert.deepEqual(plain(c.setMiniMaxH3Dmad(on, true)), plain(on));
  const plainSettings = c.cloneMiniMaxH3Settings({ render_pass: 'single' });
  assert.deepEqual(plain(c.setMiniMaxH3Dmad(plainSettings, false)), plain(plainSettings));
});

test('DMAD has no 2 Pass Advanced', () => {
  const c = fixture();
  const on = c.setMiniMaxH3Dmad(c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'three_pass' }), true);
  assert.equal(on.render_pass, 'two_pass');
  // A saved project that holds both is read as 2 Pass.
  assert.equal(c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'three_pass', use_dmad: true }).render_pass, 'two_pass');
  assert.equal(c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'three_pass' }).render_pass, 'three_pass');
});

test('DMAD fills in the saved pass profiles too, so Single and 2 Pass agree', () => {
  const c = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single', steps: 17 });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  const on = c.setMiniMaxH3Dmad(settings, true);
  const twoPass = c.selectMiniMaxH3PassSettings(on, 'two_pass');
  assert.equal(twoPass.use_dmad, true);
  assert.equal(twoPass.render_pass, 'two_pass');
  assert.equal(twoPass.two_pass_pass1_sampler, 'lcm');
  assert.equal(twoPass.loras[0].name, DMAD_LORA);
  const back = c.selectMiniMaxH3PassSettings(twoPass, 'single');
  assert.equal(back.use_dmad, true);
  assert.equal(back.sampler_name, 'lcm');
  assert.equal(back.steps, 4);
  const off = c.selectMiniMaxH3PassSettings(c.setMiniMaxH3Dmad(back, false), 'two_pass');
  assert.equal(off.use_dmad, false);
  assert.equal(off.two_pass_pass1_sampler, 'res_multistep');
  assert.equal(off.use_loras, false);
});

test('a pass profile saved under the other sampling method is not used after a video profile switches it', () => {
  const c = fixture();
  // Standard project with a cached 2 Pass profile.
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single' });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  assert.equal(settings.ref_pass_profiles.two_pass.two_pass_pass1_sampler, 'res_multistep');
  // A video profile with DMAD on is applied: it carries the DMAD values but not the pass cache.
  const dmadProfile = c.setMiniMaxH3Dmad(c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single' }), true);
  const applied = c.cloneMiniMaxH3Settings({ ...settings, ...plain(dmadProfile), ref_pass_profiles: settings.ref_pass_profiles });
  assert.equal(applied.use_dmad, true);
  const twoPass = c.selectMiniMaxH3PassSettings(applied, 'two_pass');
  assert.equal(twoPass.use_dmad, true);
  assert.equal(twoPass.two_pass_pass1_sampler, 'lcm', 'the stale Standard profile did not win');
  assert.equal(twoPass.two_pass_lora_name, REFINER);
  assert.equal(twoPass.loras[0].name, DMAD_LORA);
});

test('the RefMod pipeline keeps working with DMAD', () => {
  const c = fixture();
  const on = c.setMiniMaxH3Dmad(c.cloneMiniMaxH3Settings({ pipeline: 'refmod', render_pass: 'two_pass' }), true);
  assert.equal(on.pipeline, 'refmod');
  assert.equal(on.video_mode, 'reference_to_video');
  assert.equal(on.use_dmad, true);
  assert.equal(on.render_pass, 'two_pass');
});

test('sigma shift settings are clamped and survive a project reload', () => {
  const c = fixture();
  const settings = c.cloneMiniMaxH3Settings({ sigma_pass1_enabled: true, sigma_pass1_video: 500, sigma_pass1_audio: 0, sigma_pass2_video: 'bad' });
  assert.equal(settings.sigma_pass1_enabled, true);
  assert.equal(settings.sigma_pass1_video, 100);
  assert.equal(settings.sigma_pass1_audio, 0.01);
  assert.equal(settings.sigma_pass2_video, 12);
  const reloaded = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings)));
  assert.deepEqual(plain(reloaded), plain(settings));
  const defaults = c.cloneMiniMaxH3Settings({});
  assert.equal(defaults.sigma_pass1_enabled, false);
  assert.equal(defaults.use_dmad, false);
});

function tabFixture() {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('models.mjs'), c);
  vm.runInContext(readBuilderModule('dmad_models_tab.mjs'), c);
  return c;
}

test('the Dmad Models tab sits between MiniMax H3 and Custom Nodes in the Models dialog', () => {
  const dialogs = readBuilderModule('dialogs.mjs');
  const minimax = dialogs.indexOf('id: "minimax"');
  const dmad = dialogs.indexOf('{ id: "dmad", label: "Dmad Models"');
  const nodes = dialogs.indexOf('{ id: "custom-nodes", label: "Custom Nodes"');
  assert.ok(minimax > 0 && minimax < dmad && dmad < nodes, 'tab order');
  assert.ok(dialogs.includes('stopTabWork();'), 'leaving the tab stops its polling');
});

test('the Dmad Models tab lists both LoRAs from the same names the settings use', () => {
  const c = tabFixture();
  const merged = plain(c.mergeDmadModels([]));
  assert.deepEqual(merged.map((item) => item.filename), [DMAD_LORA, REFINER]);
  for (const item of merged) {
    assert.ok(item.url.startsWith('https://huggingface.co/') && item.url.includes('/resolve/'), item.id);
    assert.equal(item.installed, false);
  }
  const fromServer = plain(c.mergeDmadModels([{ id: 'dmad_lora', installed: true, size: 1956195976, destination: 'C:/models/loras' }]));
  assert.equal(fromServer[0].installed, true);
  assert.equal(fromServer[1].installed, false);
  assert.equal(fromServer[0].label, merged[0].label, 'the dialog keeps its own labels and links');
});

test('the Dmad Models tab says what each file is doing', () => {
  const c = tabFixture();
  assert.equal(c.formatBytes(1956195976), '1.82 GB');
  assert.equal(c.formatBytes(150 * 1024 * 1024), '150 MB');
  assert.equal(c.dmadModelStateText({ installed: false, job: {} }), 'Not installed');
  assert.equal(c.dmadModelStateText({ installed: true, size: 1956195976, job: {} }), 'Installed (1.82 GB)');
  assert.match(c.dmadModelStateText({ installed: false, job: { state: 'downloading', downloaded: 524288000, total: 1048576000 } }), /^Downloading… 50% \(500 MB of 1000 MB\)$/);
  assert.equal(c.dmadModelStateText({ installed: false, job: { state: 'downloading', downloaded: 0, total: 0 } }), 'Starting…');
  assert.equal(c.dmadModelStateText({ installed: false, job: { state: 'error', error: 'No space.' } }), 'Failed: No space.');
  assert.equal(c.dmadModelStateText({ installed: false, job: { state: 'cancelled' } }), 'Cancelled');
});

test('the panel offers lcm and carries the sampling method row', () => {
  const layout = readBuilderSource();
  assert.ok(layout.includes('MINIMAX_H3_SAMPLING_OPTIONS'));
  assert.ok(layout.includes('"lcm"'));
});
