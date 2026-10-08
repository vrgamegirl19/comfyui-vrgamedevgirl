const { readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = readBuilderSource();
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function fixture() {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  return c;
}
test('all three mode profiles survive switching and project JSON reload', () => {
  const c = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single', steps: 17, use_te_speed: true, ref_image_size: 'match' });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  settings.pass1_use_feedforward = true;
  settings.pass2_use_feedforward = false;
  settings.two_pass_pass1_steps = 25;
  settings.two_pass_use_fast_vae_decode = true;
  settings = c.selectMiniMaxH3PassSettings(settings, 'three_pass');
  settings.pass1_use_feedforward = false;
  settings.pass2_use_feedforward = true;
  settings.advanced_two_pass_pass2_steps = 3;
  settings.two_pass_use_fast_vae_decode = false;
  settings = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings)));
  assert.equal(settings.render_pass, 'three_pass');
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  assert.equal(settings.steps, 17);
  assert.equal(settings.ref_image_size, 'match');
  assert.equal(settings.use_te_speed, true);
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(settings.two_pass_pass1_steps, 25);
  assert.equal(settings.pass1_use_feedforward, true);
  assert.equal(settings.pass2_use_feedforward, false);
  assert.equal(settings.two_pass_use_fast_vae_decode, true);
  settings = c.selectMiniMaxH3PassSettings(settings, 'three_pass');
  assert.equal(settings.advanced_two_pass_pass2_steps, 3);
  assert.equal(settings.pass1_use_feedforward, false);
  assert.equal(settings.pass2_use_feedforward, true);
  assert.equal(settings.two_pass_use_fast_vae_decode, false);
  for (const profile of Object.values(settings.ref_pass_profiles)) assert.equal(profile.ref_pass_profiles, undefined);
});
test('saved ref pass mode and advanced profile migrate to scene-scoped render pass', () => {
  const c = fixture();
  const legacy = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', ref_pass_mode: 'advanced' });
  assert.equal(legacy.render_pass, 'three_pass');
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single',
    ref_pass_profiles: { advanced: { advanced_two_pass_pass2_steps: 5 } } });
  settings = c.selectMiniMaxH3PassSettings(settings, 'three_pass');
  assert.equal(settings.advanced_two_pass_pass2_steps, 5);
  assert.equal(settings.ref_pass_mode, undefined);
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
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single' });
  const segment = { id: 'scene' };
  Object.assign(c, { miniMaxPassButtons: ['single', 'two_pass', 'three_pass'].map(passMode => ({ dataset: { passMode } })),
    requireActiveSegment: () => segment, pushHistory() {}, state: {}, wizardVideoSettings: { global: false },
    clearMiniMaxImageReferenceStartFrameOnModeSwitch() {},
    setMiniMaxH3RenderPassForSegment: (scene, passMode) => { c.state.miniMaxH3Settings.render_pass = passMode; },
    setMiniMaxH3ModeForSegment: (scene, mode) => { c.state.miniMaxH3Settings.video_mode = mode; },
    saveMiniMaxH3SettingsFromPanel: () => settings,
    syncMiniMaxH3Panel() { settings = c.state.miniMaxH3Settings; },
    autoSaveSessionQuiet: async () => {},
  });
  const start = source.indexOf('  for (const button of miniMaxPassButtons) {', source.indexOf('  miniMaxUseTurboLora.input.addEventListener'));
  const end = source.indexOf('  miniMaxAudioMode.addEventListener', start);
  vm.runInContext(source.slice(start, end), c);
  for (const button of c.miniMaxPassButtons) {
    await button.onclick();
    assert.equal(settings.render_pass, button.dataset.passMode);
    assert.equal(c.miniMaxPassButtons.filter(b => b.dataset.passMode === settings.render_pass).length, 1);
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
  for (const mode of ['single', 'two_pass', 'three_pass']) {
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

test('LoRA presets select installed files, save steps, and leave missing selections unchanged', async () => {
  const c = fixture();
  let saved = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'two_pass', two_pass_lora_name: 'chosen.safetensors', advanced_two_pass_pass2_steps: 9 });
  let saves = 0;
  const four = 'H3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors';
  const eight = 'H3/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors';
  Object.assign(c, {
    state: { miniMaxH3TwoPassEnabled: true, miniMaxH3ThreePassEnabled: false },
    miniMaxTwoPassLoraPresetButtons: [4, 8].map(preset => ({ dataset: { preset: String(preset) } })),
    miniMaxTwoPassLoraPreset: { dataset: { preset: '4' } },
    miniMaxTwoPassLoraPicker: { input: { value: 'chosen.safetensors' }, options: [] },
    miniMaxTwoPassLoraStatus: { textContent: '' },
    installed: [eight, four],
    getJson: async () => ({ loras: c.installed }),
    videoSettingsSegment: () => null,
    miniMaxH3ModeForSegment: () => saved.video_mode,
    twoPassControls: [{ steps: { value: '20' } }, { steps: { value: '2' } }],
    pushHistory() {}, syncMiniMaxH3Panel() {},
    saveMiniMaxH3SettingsFromPanel() {
      saved = c.cloneMiniMaxH3Settings({ ...saved, two_pass_lora_name: c.miniMaxTwoPassLoraPicker.input.value, two_pass_lora_preset: Number(c.miniMaxTwoPassLoraPreset.dataset.preset), two_pass_pass2_steps: Number(c.twoPassControls[1].steps.value) });
    },
    autoSaveSessionQuiet: async () => { saves++; },
  });
  const start = source.indexOf('  for (const button of miniMaxTwoPassLoraPresetButtons)', source.indexOf('wireSearchablePicker(miniMaxTurboLoraPicker'));
  const end = source.indexOf('  wireSearchablePicker(miniMaxTwoPassLoraPicker', start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(source.slice(start, end), c);
  for (const [index, steps] of [[1, 4], [0, 2]]) {
    await c.miniMaxTwoPassLoraPresetButtons[index].onclick();
    assert.equal(saved.two_pass_pass2_steps, steps);
    assert.equal(saved.two_pass_lora_preset, steps * 2);
    assert.equal(c.twoPassControls[0].steps.value, '20');
    assert.equal(saved.advanced_two_pass_pass2_steps, 9);
    assert.equal(saved.two_pass_lora_name, index === 1 ? eight : four);
  }
  assert.equal(saves, 2);
  c.twoPassControls[1].steps.value = '7';
  c.saveMiniMaxH3SettingsFromPanel();
  saved = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(saved)));
  saved = c.selectMiniMaxH3PassSettings(saved, 'single');
  saved = c.selectMiniMaxH3PassSettings(saved, 'two_pass');
  assert.equal(saved.two_pass_lora_preset, 4);
  assert.equal(saved.two_pass_pass2_steps, 7);
  assert.equal(saved.two_pass_lora_name, four);
  c.installed = [];
  await c.miniMaxTwoPassLoraPresetButtons[1].onclick();
  assert.equal(saved.two_pass_lora_name, four);
  assert.equal(saved.two_pass_pass2_steps, 7);
  assert.equal(saved.two_pass_lora_preset, 4);
  assert.match(c.miniMaxTwoPassLoraStatus.textContent, /No matching 8-step LoRA installed/);
  assert.equal(saves, 2);
  assert.ok(c.miniMaxTwoPassLoraPresetButtons.every(button => !button.disabled));
  c.state.miniMaxH3ThreePassEnabled = true;
  await c.miniMaxTwoPassLoraPresetButtons[1].onclick();
  assert.equal(saved.two_pass_pass2_steps, 7);
  assert.equal(saves, 2);
});

test('installed LoRA matching respects preferred names, subfolders and recognized fallbacks', () => {
  const c = fixture();
  const preferred4 = 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors';
  const user4 = 'minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors';
  const user8 = 'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors';
  const fallback8 = 'minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors';
  assert.equal(c.miniMaxInstalledPass2Lora(4, [user4, `H3/${preferred4}`]), `H3/${preferred4}`);
  assert.equal(c.miniMaxInstalledPass2Lora(4, [user4]), user4);
  const windowsPath = `H3${String.fromCharCode(92)}${preferred4.toUpperCase()}`;
  assert.equal(c.miniMaxInstalledPass2Lora(4, [windowsPath]), windowsPath);
  assert.equal(c.miniMaxInstalledPass2Lora(8, [fallback8, user8]), user8);
  assert.equal(c.miniMaxInstalledPass2Lora(8, [`H3/${fallback8}`]), `H3/${fallback8}`);
  for (const name of [
    'minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors',
    'minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors',
    'minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors',
    'minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors',
  ]) assert.equal(c.miniMaxInstalledPass2Lora(4, [name]), name);
  assert.equal(c.miniMaxInstalledPass2Lora(8, [preferred4]), '');
  assert.equal(c.miniMaxInstalledPass2Lora(4, [user8, 'unknown_4step.safetensors']), '');
});


test('retired VRAM presets map into the 8-24 GB range and tiling settings are no longer options', () => {
  const c = fixture();
  assert.equal(c.cloneMiniMaxH3Settings({}).advanced_two_pass_vram_preset, '16gb');
  for (const [saved, expected] of [['8gb', '8gb'], ['12gb', '12gb'], ['16gb', '16gb'], ['24gb', '24gb'], ['32gb', '24gb'], ['custom', '24gb'], ['bogus', '16gb'], [undefined, '16gb']]) {
    assert.equal(c.cloneMiniMaxH3Settings({ advanced_two_pass_vram_preset: saved }).advanced_two_pass_vram_preset, expected, String(saved));
  }
  // Tiles, chunks, fades and the upscaler device are derived at run time (minimax/tile_plan.py).
  const defaults = c.cloneMiniMaxH3Settings({});
  for (const key of ['advanced_two_pass_tile_size_mode', 'advanced_two_pass_tile_width', 'advanced_two_pass_grid_rows',
    'advanced_two_pass_chunk_length', 'advanced_two_pass_fade_width', 'advanced_two_pass_overlap_mode',
    'advanced_two_pass_brightness_match', 'advanced_two_pass_dynamic_fade', 'advanced_two_pass_upscaler_device',
    'two_pass_final_width', 'two_pass_final_height', 'advanced_two_pass_pass2_megapixels']) {
    assert.equal(key in defaults, false, key);
  }
});

test('MiniMax scene render keeps only the final pass video (no pass 1 backup)', () => {
  const start = source.indexOf('async function renderMiniMaxSceneVideoWithProgress');
  const end = source.indexOf('async function stitchRenderedScenes', start);
  assert.ok(start >= 0 && end > start);
  const body = source.slice(start, end);
  assert.equal(body.includes('collect_minimax_h3_stage_backup'), false);
  assert.equal(body.includes('find_minimax_h3_stage_outputs'), false);
  assert.equal(body.includes('canCleanupScratch'), false);
  assert.ok(body.includes('cleanup_minimax_h3_output'));
});

test('one output resolution is shared by every pass mode and legacy saves migrate to it', () => {
  const c = fixture();
  const defaults = c.cloneMiniMaxH3Settings({});
  assert.equal(defaults.resolution_preset, '1k');
  assert.equal(defaults.megapixels, 0.5625);
  // A preset resolves per aspect ratio; custom keeps the typed megapixels.
  assert.equal(c.cloneMiniMaxH3Settings({ resolution_preset: '2k', aspect_ratio: '9:16 (Portrait Widescreen)' }).megapixels, 1.9922);
  assert.equal(c.cloneMiniMaxH3Settings({ resolution_preset: '1k' }).megapixels, 0.5625);
  assert.equal(c.cloneMiniMaxH3Settings({ resolution_preset: 'custom', megapixels: 1.3 }).megapixels, 1.3);
  const size = c.miniMaxH3FrameSize(1.9922, '16:9 (Widescreen)');
  assert.equal(size.width, 1920);
  assert.equal(size.height, 1088);
  // Legacy saves took the resolution of the pass type they rendered with.
  const single = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single', megapixels: 0.9 });
  assert.equal(single.resolution_preset, 'custom');
  assert.equal(single.megapixels, 0.9);
  const two = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'two_pass', two_pass_final_width: 1920, two_pass_final_height: 1080 });
  assert.equal(two.resolution_preset, '2k');
  assert.equal(two.megapixels, 1.9922);
  const advanced = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'three_pass', advanced_two_pass_pass2_resolution_preset: '4k' });
  assert.equal(advanced.resolution_preset, '4k');
  assert.equal(advanced.megapixels, 7.9688);
  const advancedCustom = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'three_pass', advanced_two_pass_pass2_resolution_preset: 'custom', advanced_two_pass_pass2_megapixels: 3.3 });
  assert.equal(advancedCustom.resolution_preset, 'custom');
  assert.equal(advancedCustom.megapixels, 3.3);
  // Saved settings reload unchanged.
  const reloaded = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(advanced)));
  assert.equal(reloaded.resolution_preset, '4k');
  assert.equal(reloaded.megapixels, 7.9688);
});

test('switching pass mode keeps the same output resolution', () => {
  const c = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'reference_to_video', render_pass: 'single', resolution_preset: '1k' });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(settings.resolution_preset, '1k');
  settings.resolution_preset = '4k';
  settings.megapixels = 7.9688;
  settings = c.selectMiniMaxH3PassSettings(settings, 'three_pass');
  assert.equal(settings.resolution_preset, '4k');
  assert.equal(settings.megapixels, 7.9688);
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  assert.equal(settings.resolution_preset, '4k');
  assert.equal(settings.megapixels, 7.9688);
});

test('VRAM presets size an equal tile grid from the output resolution', () => {
  const c = fixture();
  const fourK = (key) => c.miniMaxH3TilePlan(key, 7.9688, '16:9 (Widescreen)');
  const grid = (plan) => `${plan.rows}x${plan.cols}`;
  // Same grids as tests/test_minimax_tile_plan.py (the Python planner the graph uses).
  assert.equal(grid(fourK('24gb')), '3x4');
  assert.equal(grid(fourK('16gb')), '4x5');
  assert.equal(grid(fourK('12gb')), '5x5');
  assert.equal(grid(fourK('8gb')), '6x7');
  assert.equal(grid(c.miniMaxH3TilePlan('24gb', 1.9922, '16:9 (Widescreen)')), '2x2');
  assert.equal(grid(c.miniMaxH3TilePlan('16gb', 1.9922, '16:9 (Widescreen)')), '2x2');
  // 2 Pass Advanced is for cards up to 24 GB: retired presets map to 24 GB and there is no 32 GB preset.
  assert.equal(vm.runInContext("MINIMAX_H3_VRAM_PRESETS['32gb']", c), undefined);
  assert.equal(vm.runInContext('Object.keys(MINIMAX_H3_VRAM_PRESETS).join()', c), '8gb,12gb,16gb,24gb');
  assert.equal(grid(fourK('32gb')), grid(fourK('24gb')));
  assert.equal(fourK('24gb').chunk, 153);
  assert.equal(fourK('24gb').overlap, 160);
  for (const key of ['8gb', '12gb', '16gb', '24gb']) assert.equal(fourK(key).chunk % 17, 0);
});

test('LoRAs default to the first pass and the target dropdown is labeled "LoRA target"', () => {
  const c = fixture();
  const settings = c.cloneMiniMaxH3Settings({
    use_loras: true, lora_count: 4,
    loras: [
      { name: 'a.safetensors', strength: 1 },
      { name: 'b.safetensors', strength: 1, apply_to: 'both' },
      { name: 'c.safetensors', strength: 1, apply_to: 'pass2' },
      { name: 'd.safetensors', strength: 1, apply_to: 'bogus' },
    ],
  });
  assert.equal(Array.from(settings.loras, (item) => item.apply_to).join(), 'pass1,both,pass2,pass1');
  // Explicit choices survive a project reload.
  const reloaded = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings)));
  assert.equal(Array.from(reloaded.loras, (item) => item.apply_to).join(), 'pass1,both,pass2,pass1');
  const menu = section('const applyTo = makeSelect([', 'const row = document.createElement("div");');
  assert.ok(menu.includes('], "pass1");'), 'the dropdown starts on Pass 1 only');
  assert.ok(menu.indexOf('Pass 1 only') < menu.indexOf('Pass 2 only') && menu.indexOf('Pass 2 only') < menu.indexOf('Both passes'));
  assert.ok(menu.includes('makeField("LoRA target", applyTo)'));
  assert.equal(source.includes('2-pass target'), false);
  assert.ok(source.includes('slot.applyTo.value = ["both", "pass1", "pass2"].includes(item.apply_to) ? item.apply_to : "pass1";'));
});


test('I2V pass switching keeps its mode and settings separate from reference profiles', () => {
  const c = fixture();
  let settings = c.cloneMiniMaxH3Settings({ video_mode: 'image_to_video', render_pass: 'single',
    i2v_pass_settings_version: 1, steps: 17, ref_pass_profiles: { two_pass: { two_pass_pass1_steps: 99, diffusion_model_name: 'ref-custom.safetensors' } } });
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(settings.video_mode, 'image_to_video');
  assert.equal(settings.render_pass, 'two_pass');
  assert.equal(settings.two_pass_pass1_steps, 20);
  assert.match(settings.diffusion_model_name, /fl2va/);
  assert.match(settings.two_pass_lora_name, /fl2v/);
  settings.two_pass_pass1_steps = 31;
  settings.two_pass_pass2_steps = 7;
  settings.two_pass_pass2_seed = 123;
  settings.two_pass_lora_name = 'custom-i2v.safetensors';
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  assert.equal(settings.steps, 17);
  settings = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings)));
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(settings.video_mode, 'image_to_video');
  assert.equal(settings.two_pass_pass1_steps, 31);
  assert.equal(settings.two_pass_pass2_steps, 7);
  assert.equal(settings.two_pass_pass2_seed, 123);
  assert.equal(settings.two_pass_lora_name, 'custom-i2v.safetensors');
  assert.equal(settings.ref_pass_profiles.two_pass.two_pass_pass1_steps, 99);
  for (const profile of Object.values(settings.i2v_pass_profiles)) {
    assert.equal(profile.i2v_pass_profiles, undefined);
    assert.equal(profile.ref_pass_profiles, undefined);
  }
});

test('legacy I2V saves stay single pass and advanced is not offered for new I2V settings', () => {
  const c = fixture();
  const legacy = c.cloneMiniMaxH3Settings({ video_mode: 'image_to_video', render_pass: 'two_pass' });
  assert.equal(legacy.render_pass, 'single');
  assert.equal(legacy.i2v_pass_settings_version, 1);
  const selected = c.selectMiniMaxH3PassSettings(legacy, 'two_pass');
  assert.equal(c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(selected))).render_pass, 'two_pass');
  assert.equal(c.selectMiniMaxH3PassSettings(selected, 'three_pass').render_pass, 'two_pass');
});

test('I2V LoRA presets skip reference-to-video adapters', () => {
  const c = fixture();
  const ref = 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors';
  const i2v = 'H3/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors';
  assert.equal(c.miniMaxInstalledPass2Lora(4, [ref, i2v], 'image_to_video'), i2v);
  assert.equal(c.miniMaxInstalledPass2Lora(4, [ref], 'image_to_video'), '');
  assert.equal(c.miniMaxInstalledPass2Lora(4, [ref, i2v], 'reference_to_video'), ref);
});


test('actual pass-button handlers keep global and locked-scene I2V settings in I2V', async () => {
  for (const locked of [false, true]) {
    const c = fixture();
    const globalSettings = c.cloneMiniMaxH3Settings({ video_mode: locked ? 'reference_to_video' : 'image_to_video', render_pass: 'single', i2v_pass_settings_version: 1 });
    const scene = locked ? { use_scene_minimax_h3_settings: true, minimax_h3_settings: c.cloneMiniMaxH3Settings({ video_mode: 'image_to_video', render_pass: 'single', i2v_pass_settings_version: 1 }) } : null;
    Object.assign(c, {
      state: { miniMaxH3Settings: globalSettings },
      wizardVideoSettings: { global: !locked },
      miniMaxPassButtons: [{ dataset: { passMode: 'two_pass' } }],
      requireActiveSegment: () => scene,
      pushHistory() {}, clearMiniMaxImageReferenceStartFrameOnModeSwitch() {}, syncMiniMaxH3Panel() {},
      saveMiniMaxH3SettingsFromPanel: () => locked ? scene.minimax_h3_settings : c.state.miniMaxH3Settings,
      setMiniMaxH3RenderPassForSegment() {},
      setMiniMaxH3ModeForSegment: (segment, mode) => { segment.minimax_h3_settings.video_mode = mode; },
      autoSaveSessionQuiet: async () => {},
    });
    const start = source.indexOf('  for (const button of miniMaxPassButtons) {', source.indexOf('  for (const button of miniMaxPassButtons.refmod'));
    const end = source.indexOf('  miniMaxAudioMode.addEventListener', start);
    vm.runInContext(source.slice(start, end), c);
    await c.miniMaxPassButtons[0].onclick();
    const selected = locked ? scene.minimax_h3_settings : c.state.miniMaxH3Settings;
    assert.equal(selected.video_mode, 'image_to_video');
    assert.equal(selected.render_pass, 'two_pass');
    if (locked) {
      assert.equal(c.state.miniMaxH3Settings.video_mode, 'reference_to_video');
      assert.equal(c.state.miniMaxH3Settings.render_pass, 'single');
    }
  }
});
test('I2V rejects saved between-scene continuity for both passes', () => {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  for (const pass of ['single', 'two_pass']) {
    assert.equal(c.isMiniMaxH3ContinuityAllowedForMode('latent_continuation_masked', 'image_to_video', pass), false);
    assert.equal(c.isMiniMaxH3ContinuityAllowedForMode('off', 'image_to_video', pass), true);
    assert.equal(c.isMiniMaxH3ContinuityAllowedForMode('latent_continuation_masked', 'reference_to_video', pass), true);
  }
});

test('panel hides direction controls without a continuity prompt and hides I2V continuity settings', () => {
  const panel = readBuilderModule('minimax_panel.mjs');
  const start = panel.indexOf('const promptFromLastFrame =');
  const end = panel.indexOf('miniMaxPrompt.disabled', start);
  const code = panel.slice(start, end);
  for (const [mode, enabled] of [['image_to_video', false], ['reference_to_video', false], ['reference_to_video', true]]) {
    const c = vm.createContext({ mode, segment: {}, miniMaxH3FrameContinuityPromptEnabled: () => enabled,
      miniMaxContinuationDirectionField: {style: {}}, miniMaxContinuationStartField: {style: {}}, miniMaxContinuitySection: {style: {}} });
    vm.runInContext(code, c);
    assert.equal(c.miniMaxContinuationDirectionField.style.display, enabled ? 'flex' : 'none');
    assert.equal(c.miniMaxContinuationStartField.style.display, enabled ? 'flex' : 'none');
    assert.equal(c.miniMaxContinuitySection.style.display, mode === 'image_to_video' ? 'none' : '');
  }
});
