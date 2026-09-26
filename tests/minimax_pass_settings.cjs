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
    requireActiveSegment: () => segment, pushHistory() {}, state: {},
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


test('old VRAM presets migrate fade only and preserve custom settings', () => {
  const c = fixture();
  for (const [preset, tile, chunk] of [['8gb',352,51],['12gb',512,85],['16gb',576,272],['24gb',672,153]]) {
    const old = { advanced_two_pass_defaults_version: 3, advanced_two_pass_vram_preset: preset,
      advanced_two_pass_tile_size_mode: 'specific_size', advanced_two_pass_tile_width: tile, advanced_two_pass_tile_height: tile,
      advanced_two_pass_chunk_length: chunk, advanced_two_pass_fade_width: 128, advanced_two_pass_fade_height: 128,
      advanced_two_pass_spatial_w_overlap: 128, advanced_two_pass_spatial_h_overlap: 128,
      advanced_two_pass_overlap_mode: 'earlier', advanced_two_pass_overlap_blend: 'smoothstep' };
    const migrated = c.cloneMiniMaxH3Settings(old);
    assert.equal(migrated.advanced_two_pass_fade_width, 64);
    assert.equal(migrated.advanced_two_pass_fade_height, 64);
    assert.equal(migrated.advanced_two_pass_tile_width, tile);
    assert.equal(migrated.advanced_two_pass_chunk_length, chunk);
    assert.equal(migrated.advanced_two_pass_vram_preset, preset);
    assert.equal(c.cloneMiniMaxH3Settings({...old,advanced_two_pass_fade_width:80}).advanced_two_pass_fade_width,80);
    assert.equal(c.cloneMiniMaxH3Settings({...old,advanced_two_pass_defaults_version:4}).advanced_two_pass_fade_width,128);
  }
});

test('final collection waits for in-flight stage backup and retains scratch on copy failure', async () => {
  for (const fail of [false, true]) {
    const segment = {};
    let finishCopy, copyCalls = 0;
    const c = vm.createContext({ twoPass:true, threePass:true, segment, built:{output_folder:'/scratch'},
      renderStartedAt:1, projectFolder:'/project', slotNumber:1, state:{}, console,
      mediaPathKey:p=>p, activateSegmentVideoPath(){}, syncPreview(){}, renderList(){}, render(){}, autoSaveSessionQuiet:async()=>{},
      postJson:async (url) => {
        if (url.endsWith('find_minimax_h3_stage_outputs')) return {stage1_path:'/scratch/stage1.mp4'};
        copyCalls++;
        if (fail) throw Error('copy failed');
        await new Promise(resolve=>{finishCopy=resolve;});
        return {backup_path:'/project/backup.mp4'};
      }
    });
    const start=source.indexOf('      let liveStageBackupsRegistered = false;');
    const end=source.indexOf('      const videos = await waitForVideos(',start);
    vm.runInContext(source.slice(start,end)+';globalThis.register=registerLiveThreePassBackups;',c);
    const first=c.register();
    assert.equal(c.register(),first);
    await new Promise(resolve=>setImmediate(resolve));
    if (!fail) finishCopy();
    await first;
    const a=source.indexOf('      // Finish any in-flight backup');
    const b=source.indexOf('      pushHistory();',a);
    const ready=await vm.runInContext('(async()=>{'+source.slice(a,b)+'return canCleanupScratch;})()',c);
    assert.equal(ready,!fail);
    if (!fail) { assert.equal(segment.minimax_h3_stage1_backup_path,'/project/backup.mp4'); assert.equal(copyCalls,1); }
  }
});
