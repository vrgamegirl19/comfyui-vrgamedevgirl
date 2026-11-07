const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// How each image workflow is asked to make the thumbnail (web/music_video_builder/thumbnail_engines.mjs).
function load() {
  const context = vm.createContext({});
  vm.runInContext(readBuilderModule('thumbnail_engines.mjs'), context);
  return context;
}

const plain = (value) => JSON.parse(JSON.stringify(value));
const request = (c, engine, settings = {}, extra = {}) => plain(c.thumbnailEngineRequest({
  engine, prompt: 'A neon portrait.', sourcePath: 'C:/p/thumbnail/source_0001.png', change: 0.65, settings, seed: 42,
  loraPayload: (s) => ({ use_custom_loras: Boolean(s.use_loras), lora_count: s.lora_count || 0 }), defaultNbModel: 'nb-default', ...extra,
}));

test('how much a restyle workflow may change the frame maps onto each workflow\'s own control, in the right direction', () => {
  const c = load();
  // Z-Image and Ernie: the step to start at, 1 changes the frame most and 8 least.
  assert.equal(c.startStepFor(1), 1);
  assert.equal(c.startStepFor(0), 8);
  assert.equal(c.startStepFor(0.65), 3);
  assert.ok(c.startStepFor(0.2) > c.startStepFor(0.8));
  // Krea 2: denoise 0 to 10, more changes more.
  assert.equal(c.creativityFor(0), 0);
  assert.equal(c.creativityFor(1), 10);
  assert.equal(c.creativityFor(0.65), 7);
  // MOSS-Image: denoise, never all the way to 1 (that would ignore the frame) and never so low nothing changes.
  assert.equal(c.denoiseFor(0), 0.2);
  assert.equal(c.denoiseFor(1), 0.95);
  assert.equal(c.denoiseFor(0.65), 0.65);
  for (const bad of ['x', null, undefined, NaN]) assert.equal(c.denoiseFor(bad), 0.65);
});

test('NanoBanana gets the frame as a reference image and needs its key', () => {
  const c = load();
  const built = request(c, 'nano_banana', { api_key: 'key', model: '' });
  assert.equal(built.url, '/vrgdg/workflow_runner/build_nb_image_prompt');
  assert.deepEqual(built.payload, { api_key: 'key', model: 'nb-default', prompt: 'A neon portrait.', image_ingredients: [{ path: 'C:/p/thumbnail/source_0001.png' }] });
  assert.throws(() => request(c, 'nano_banana', { api_key: '  ' }), /NanoBanana API key is missing.*another image workflow/);
});

test('Flux/Klein gets the frame as an ingredient, the Builder\'s models and size, and its LoRA fields', () => {
  const c = load();
  const built = request(c, 'flux_klein', { unet_name: 'u', clip_name: 'c', vae_name: 'v', width: 1024, height: 576, use_loras: true, lora_count: 2 });
  assert.equal(built.url, '/vrgdg/workflow_runner/build_flux_klein_prompt');
  assert.deepEqual(built.payload, {
    prompt: 'A neon portrait.', image_ingredients: [{ path: 'C:/p/thumbnail/source_0001.png' }], unet_name: 'u', clip_name: 'c', vae_name: 'v',
    width: 1024, height: 576, seed: 42, use_custom_loras: true, lora_count: 2,
  });
  const defaults = request(c, 'flux_klein', {});
  assert.equal(defaults.payload.width, 1280);
  assert.equal(defaults.payload.height, 720);
});

test('Krea 2 restyles the frame at the chosen creativity with the Builder\'s settings and LoRAs', () => {
  const c = load();
  const built = request(c, 'krea2_2pass', {
    unet_name: 'k.safetensors', aspect_ratio: '1:1', cfg: 9, use_loras: true, lora_count: 1,
    loras: [{ name: 'style.safetensors', first_pass_strength: 0.3, second_pass_strength: 0.8 }, { name: 'unused.safetensors' }],
  });
  assert.equal(built.url, '/vrgdg/workflow_runner/build_krea2_2pass_prompt');
  const p = built.payload;
  assert.equal(p.use_image_to_image, true);
  assert.equal(p.image_to_image_path, 'C:/p/thumbnail/source_0001.png');
  assert.equal(p.image_to_image_creativity, 7);
  assert.equal(p.aspect_ratio, '1:1');
  assert.equal(p.cfg, 1.2, 'cfg is kept to what Krea 2 allows');
  assert.equal(p.seed, 42);
  assert.equal(p.batch_size, 1);
  assert.equal(p.use_custom_loras, true);
  assert.equal(p.lora_count, 1);
  assert.equal(p.lora_1, 'style.safetensors');
  assert.equal(p.first_pass_strength_1, 0.3);
  assert.equal(p.second_pass_strength_1, 0.8);
  assert.equal(p.lora_2, '[none]', 'a LoRA past the count is off');
  assert.equal(request(c, 'krea2_2pass', {}).payload.aspect_ratio, '16:9 (Widescreen)');
});

test('Z-Image and Ernie start from the frame at a step that follows the chosen change', () => {
  const c = load();
  const z = request(c, 'zimage', { unet_name: 'z.safetensors', first_pass_width: 960, second_pass_width: 1280 });
  assert.equal(z.url, '/vrgdg/workflow_runner/build_zimage_prompt');
  assert.equal(z.payload.use_image_to_image, true);
  assert.equal(z.payload.image_to_image_start_at_step, 3);
  assert.equal(z.payload.image_to_image_path, 'C:/p/thumbnail/source_0001.png');
  assert.equal(z.payload.first_pass_width, 960);
  assert.equal(z.payload.first_pass_height, 720, 'what the settings leave out falls back to the Builder\'s default');
  assert.equal(z.payload.second_pass_width, 1280);
  assert.equal(z.payload.ltx_two_pass_mode, false);
  const e = request(c, 'ernie_image', { width: 1024, height: 576, use_loras: true, lora_count: 1, loras: [{ name: 'e.safetensors', strength: 0.5 }] });
  assert.equal(e.url, '/vrgdg/workflow_runner/build_ernie_image_prompt');
  assert.equal(e.payload.image_to_image_start_at_step, 3);
  assert.equal(e.payload.width, 1024);
  assert.equal(e.payload.lora_1, 'e.safetensors');
  assert.equal(e.payload.strength_1, 0.5);
  assert.equal(request(c, 'ernie_image', {}, { change: 1 }).payload.image_to_image_start_at_step, 1);
  assert.equal(request(c, 'zimage', {}, { change: 0 }).payload.image_to_image_start_at_step, 8);
});

test('MOSS-Image restyles the frame with image to image at a denoise that follows the chosen change', () => {
  const c = load();
  const moss = request(c, 'moss_image', { unet_name: 'm.safetensors', clip_name: 'q.safetensors', vae_name: 'f.safetensors', lora_name: 'turbo.safetensors', steps: 8, cfg: 1, width: 1280, height: 720 });
  assert.equal(moss.url, '/vrgdg/workflow_runner/build_moss_image_prompt');
  assert.deepEqual(moss.payload, {
    prompt: 'A neon portrait.', unet_name: 'm.safetensors', clip_name: 'q.safetensors', vae_name: 'f.safetensors', lora_name: 'turbo.safetensors',
    steps: 8, cfg: 1, width: 1280, height: 720, seed: 42, seed_mode: 'fixed', batch_size: 1,
    use_image_to_image: true, image_to_image_path: 'C:/p/thumbnail/source_0001.png', denoise: 0.65,
  });
  assert.equal(request(c, 'moss_image', {}, { change: 0.3 }).payload.denoise, 0.3);
  assert.equal(request(c, 'moss_image', {}).payload.lora_name, '[none]');
});

test('MOSS-Image Edit gets the frame as a reference for the edit LoRA, with no image to image and none of the Turbo sampling settings', () => {
  const c = load();
  const settings = {
    unet_name: 'm.safetensors', clip_name: 'q.safetensors', vae_name: 'f.safetensors', edit_lora_name: 'e.safetensors',
    lora_name: 'turbo.safetensors', steps: 8, cfg: 1, width: 1920, height: 1088,
  };
  const built = request(c, 'moss_image_edit', settings);
  assert.equal(built.url, '/vrgdg/workflow_runner/build_moss_image_prompt');
  assert.deepEqual(built.payload, {
    prompt: 'A neon portrait.', image_ingredients: [{ path: 'C:/p/thumbnail/source_0001.png' }], use_edit: true,
    unet_name: 'm.safetensors', clip_name: 'q.safetensors', vae_name: 'f.safetensors', edit_lora_name: 'e.safetensors', seed: 42, seed_mode: 'fixed',
  });
  for (const turbo of ['steps', 'cfg', 'lora_name', 'width', 'height', 'denoise', 'use_image_to_image']) {
    assert.equal(turbo in built.payload, false, `${turbo} is for text to image: the builder uses the edit LoRA's own sampler`);
  }
  assert.equal(request(c, 'moss_image_edit', {}).payload.edit_lora_name, '', 'the builder falls back to moss_image_edit_lora.safetensors');
  assert.equal(request(c, 'moss_image_edit', null).payload.use_edit, true, 'no settings at all still builds');
  assert.equal(built.label, 'MOSS-Image Edit (local, edit)');
});

test('Qwen Image Edit gets the frame as a reference and nothing about models: the builder picks the newest installed one', () => {
  const c = load();
  const built = request(c, 'qwen_image_edit', { unet_name: 'ignored.safetensors', steps: 8, cfg: 1, width: 1920 });
  assert.equal(built.url, '/vrgdg/workflow_runner/build_qwen_image_edit_prompt');
  assert.deepEqual(built.payload, {
    prompt: 'A neon portrait.', image_ingredients: [{ path: 'C:/p/thumbnail/source_0001.png' }], seed: 42, seed_mode: 'fixed',
  });
  assert.equal(built.label, 'Qwen Image Edit (local, edit)');
  assert.deepEqual(request(c, 'qwen_image_edit', null).payload, built.payload, 'no settings at all still builds');
});

test('Overlay asks the Builder to put the layers on the frame itself: its own route, the project and the look, no prompt', () => {
  const c = load();
  const built = plain(c.thumbnailOverlayRequest({
    sourcePath: ' C:/p/thumbnail/source_0001.png ', title: 'HEAT', projectFolder: 'C:/p', overlay: { color: 'cyan', position: 'top', strength: 80 },
  }));
  assert.equal(built.url, '/vrgdg/music_builder/thumbnail/overlay');
  assert.deepEqual(built.payload, { project_folder: 'C:/p', source_path: 'C:/p/thumbnail/source_0001.png', title: 'HEAT', color: 'cyan', position: 'top', strength: 80 });
  assert.match(built.label, /^Overlay/);
  const bare = plain(c.thumbnailOverlayRequest({ sourcePath: 'a.png', projectFolder: 'C:/p' }));
  assert.equal(bare.payload.title, '', 'no title is fine: only the grade is applied');
  assert.equal('color' in bare.payload, false, 'the pack picks its own defaults for what is missing');
  assert.throws(() => c.thumbnailOverlayRequest({ sourcePath: '  ', projectFolder: 'C:/p' }), /Choose the frame first/);
  assert.throws(() => c.thumbnailOverlayRequest({ sourcePath: 'a.png', projectFolder: '' }), /Create or load a project/);
});

test('a missing prompt, a missing frame or an unknown workflow is refused with a message to act on', () => {
  const c = load();
  assert.throws(() => c.thumbnailEngineRequest({ engine: 'flux_klein', prompt: '  ', sourcePath: 'a.png', settings: {} }), /Write the prompt first/);
  assert.throws(() => c.thumbnailEngineRequest({ engine: 'flux_klein', prompt: 'x', sourcePath: '', settings: {} }), /Choose the frame first/);
  assert.throws(() => c.thumbnailEngineRequest({ engine: 'nope', prompt: 'x', sourcePath: 'a.png', settings: {} }), /Unknown image workflow: nope/);
  assert.equal(request(c, 'moss_image', null).payload.width, 1280, 'no settings at all still builds');
});

test('every workflow has a label and a route', () => {
  const c = load();
  const engines = ['qwen_image_edit', 'nano_banana', 'flux_klein', 'moss_image_edit', 'krea2_2pass', 'zimage', 'ernie_image', 'moss_image'];
  const labels = vm.runInContext('THUMBNAIL_ENGINE_LABELS', c);
  // Overlay has a label too, but no build route: it is not an image workflow (see thumbnailOverlayRequest).
  assert.deepEqual(Object.keys(labels), ['overlay', ...engines]);
  assert.match(labels.overlay, /^Overlay: your frame exactly, no AI/);
  assert.throws(() => request(c, 'overlay', {}), /Unknown image workflow: overlay/);
  for (const engine of engines) {
    const built = request(c, engine, { api_key: 'key' });
    assert.match(built.url, /^\/vrgdg\/workflow_runner\/build_\w+_prompt$/);
    assert.equal(built.label, labels[engine]);
  }
});
