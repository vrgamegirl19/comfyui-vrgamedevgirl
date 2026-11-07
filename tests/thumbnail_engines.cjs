const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// How each way of making the thumbnail is asked to do it (web/music_video_builder/thumbnail_engines.mjs).
function load() {
  const context = vm.createContext({});
  vm.runInContext(readBuilderModule('thumbnail_engines.mjs'), context);
  return context;
}

const plain = (value) => JSON.parse(JSON.stringify(value));
const request = (c, engine, settings = {}, extra = {}) => plain(c.thumbnailEngineRequest({
  engine, prompt: 'A neon portrait.', sourcePath: 'C:/p/thumbnail/source_0001.png', settings, seed: 42, defaultNbModel: 'nb-default', ...extra,
}));

test('NanoBanana gets the frame as a reference image and needs its key', () => {
  const c = load();
  const built = request(c, 'nano_banana', { api_key: 'key', model: '' });
  assert.equal(built.url, '/vrgdg/workflow_runner/build_nb_image_prompt');
  assert.deepEqual(built.payload, { api_key: 'key', model: 'nb-default', prompt: 'A neon portrait.', image_ingredients: [{ path: 'C:/p/thumbnail/source_0001.png' }] });
  assert.equal(request(c, 'nano_banana', { api_key: 'key', model: 'nb-chosen' }).payload.model, 'nb-chosen');
  assert.throws(() => request(c, 'nano_banana', { api_key: '  ' }), /NanoBanana API key is missing.*another image workflow/);
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

const apiRequest = (c, runner, extra = {}) => plain(c.thumbnailApiImageRequest({
  prompt: ' Add the title "HEAT". ', sourcePath: ' C:/p/thumbnail/source_0001.png ', projectFolder: 'C:/p', runner, model: '', ...extra,
}));

test('the API image engine sends the frame, the prompt and the LLM Runner\'s own provider and key to the pack', () => {
  const c = load();
  const built = apiRequest(c, { llm_api_provider: 'google', llm_api_key: ' key-1 ', llm_api_model: 'gemini-chat' });
  assert.equal(built.url, '/vrgdg/music_builder/thumbnail/api_image');
  assert.deepEqual(built.payload, {
    project_folder: 'C:/p', source_path: 'C:/p/thumbnail/source_0001.png', prompt: 'Add the title "HEAT".', provider: 'gemini', api_key: 'key-1', model: '',
  });
  assert.match(built.label, /^API image/);
  assert.equal(apiRequest(c, { llm_api_provider: 'grok', llm_api_key: 'k' }, { model: ' grok-imagine-image ' }).payload.model, 'grok-imagine-image', 'a typed model is used');
  assert.equal(apiRequest(c, { llm_api_provider: 'openai', llm_api_key: 'k' }).payload.provider, 'openai');
});

test('the LLM Runner\'s provider names map to the image providers, and the others cannot draw', () => {
  const c = load();
  const named = (provider) => plain(c.thumbnailApiImageProvider({ llm_api_provider: provider }));
  assert.deepEqual(named('google'), { name: 'gemini', label: 'Gemini' });
  assert.deepEqual(named('Gemini'), { name: 'gemini', label: 'Gemini' });
  assert.deepEqual(named(' grok '), { name: 'grok', label: 'Grok' });
  assert.deepEqual(named('xai'), { name: 'grok', label: 'Grok' });
  assert.deepEqual(named('openai'), { name: 'openai', label: 'OpenAI' });
  for (const other of ['anthropic', 'deepseek', 'openrouter', '', undefined]) assert.equal(named(other), null, String(other));
  assert.equal(c.thumbnailApiImageProvider(null), null);
  assert.equal(c.thumbnailApiImageProvider(undefined), null);
});

test('the API image engine refuses what is missing with a message to act on, before anything is sent', () => {
  const c = load();
  const gemini = { llm_api_provider: 'google', llm_api_key: 'k' };
  assert.throws(() => apiRequest(c, gemini, { prompt: '  ' }), /Write the prompt first/);
  assert.throws(() => apiRequest(c, gemini, { sourcePath: ' ' }), /Choose the frame first/);
  assert.throws(() => apiRequest(c, gemini, { projectFolder: '' }), /Create or load a project/);
  assert.throws(() => apiRequest(c, { llm_api_provider: 'anthropic', llm_api_key: 'k' }), /cannot draw pictures.*Gemini \(google\), Grok or OpenAI/);
  assert.throws(() => apiRequest(c, { llm_api_provider: 'grok', llm_api_key: '  ' }), /Add your Grok API key in the LLM Runner first/);
  assert.throws(() => apiRequest(c, { llm_api_provider: 'google' }), /Add your Gemini API key in the LLM Runner first/);
  assert.throws(() => apiRequest(c, null), /cannot draw pictures/);
});

test('a missing prompt, a missing frame or an unknown workflow is refused with a message to act on', () => {
  const c = load();
  assert.throws(() => c.thumbnailEngineRequest({ engine: 'qwen_image_edit', prompt: '  ', sourcePath: 'a.png', settings: {} }), /Write the prompt first/);
  assert.throws(() => c.thumbnailEngineRequest({ engine: 'qwen_image_edit', prompt: 'x', sourcePath: '', settings: {} }), /Choose the frame first/);
  assert.throws(() => c.thumbnailEngineRequest({ engine: 'nope', prompt: 'x', sourcePath: 'a.png', settings: {} }), /Unknown image workflow: nope/);
  for (const removed of ['flux_klein', 'moss_image_edit', 'krea2_2pass', 'zimage', 'ernie_image', 'moss_image']) {
    assert.throws(() => request(c, removed, {}), new RegExp(`Unknown image workflow: ${removed}`), `${removed} is no longer offered`);
  }
});

test('every engine has a label, and the ComfyUI workflows have a route', () => {
  const c = load();
  const labels = vm.runInContext('THUMBNAIL_ENGINE_LABELS', c);
  // Overlay and API image have labels too, but no build route: they are not ComfyUI workflows (see their own request functions).
  assert.deepEqual(Object.keys(labels), ['overlay', 'qwen_image_edit', 'nano_banana', 'api_image']);
  assert.match(labels.overlay, /^Overlay: your frame exactly, no AI/);
  assert.match(labels.api_image, /^API image: Gemini, Grok or OpenAI of the LLM Runner/);
  assert.throws(() => request(c, 'overlay', {}), /Unknown image workflow: overlay/);
  assert.throws(() => request(c, 'api_image', {}), /Unknown image workflow: api_image/);
  for (const engine of ['qwen_image_edit', 'nano_banana']) {
    const built = request(c, engine, { api_key: 'key' });
    assert.match(built.url, /^\/vrgdg\/workflow_runner\/build_\w+_prompt$/);
    assert.equal(built.label, labels[engine]);
  }
});
