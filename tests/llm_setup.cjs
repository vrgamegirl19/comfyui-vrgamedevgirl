const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture(initial = { text_runner: 'llm_api', llm_api_key: '', llm_api_model: 'test-model' }) {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('llm_setup.mjs'), c);
  let current = { ...initial };
  const dialogs = [];
  const setup = c.createLlmSetup({ getRunnerPayload: () => ({ ...current }), openRunner: options => {
    dialogs.push(options);
    return { cancelSetup: options.onCancel };
  } });
  const dispose = c.registerLlmSetup(setup);
  return { c, dialogs, dispose, update: value => { current = { ...value }; } };
}

test('missing key pauses a reference description and resumes with fresh credentials and the same image', async () => {
  const f = fixture();
  const payload = { text_runner: 'llm_api', llm_api_key: '', llm_api_model: 'test-model', image_path: 'reference.png', name: 'Hero' };
  let finished = false;
  const pending = f.c.prepareLlmRequest('/vrgdg/music_builder/describe_reference_image', payload).then(value => {
    finished = true; return value;
  });
  await Promise.resolve();
  assert.equal(finished, false);
  assert.equal(f.dialogs.length, 1);
  assert.match(f.dialogs[0].setupMessage, /API key/);
  assert.match(f.dialogs[0].validateSetup(), /API key/);
  f.update({ text_runner: 'llm_api', llm_api_key: 'test-only-key', llm_api_model: 'chosen-model' });
  assert.equal(f.dialogs[0].validateSetup(), '');
  f.dialogs[0].onSaved();
  const sent = await pending;
  assert.equal(sent.llm_api_key, 'test-only-key');
  assert.equal(sent.llm_api_model, 'chosen-model');
  assert.equal(sent.image_path, payload.image_path);
  assert.equal(sent.name, payload.name);
  assert.equal(payload.llm_api_key, '', 'the original captured request is not mutated');
});

test('setup can switch runners and reuse the original action', async () => {
  const f = fixture();
  const pending = f.c.prepareLlmRequest('/describe_reference_image', { text_runner: 'llm_api', image_data: 'test-image' });
  f.update({ text_runner: 'qwen_local', qwen_model_file: 'qwen.gguf', qwen_mmproj_file: 'vision.gguf', model_file: 'qwen.gguf', mmproj_file: 'vision.gguf' });
  f.dialogs[0].onSaved();
  const payload = await pending;
  assert.equal(payload.text_runner, 'qwen_local');
  assert.equal(payload.mmproj_file, 'vision.gguf');
  assert.equal(payload.image_data, 'test-image');
});

test('cancel sends no request and keeps the caller data', async () => {
  const f = fixture();
  const pending = f.c.prepareLlmRequest('/describe_reference_image', { text_runner: 'llm_api', name: 'Hero' });
  f.dialogs[0].onCancel();
  await assert.rejects(pending, error => error.code === 'LLM_SETUP_CANCELLED');
});

test('concurrent requests share one dialog and use the strongest vision requirement', async () => {
  const f = fixture({ text_runner: 'builtin' });
  const text = f.c.prepareLlmRequest('/generate', { text_runner: 'builtin', user_notes: 'Original notes' });
  const image = f.c.prepareLlmRequest('/describe_reference_image', { text_runner: 'builtin', image_path: 'photo.png' });
  assert.equal(f.dialogs.length, 1);
  f.update({ text_runner: 'builtin', gemma_model_file: 'gemma.gguf', model_file: 'gemma.gguf' });
  assert.match(f.dialogs[0].validateSetup(), /mmproj/);
  f.update({ text_runner: 'builtin', gemma_model_file: 'gemma.gguf', model_file: 'gemma.gguf', mmproj_file: 'vision.gguf' });
  assert.equal(f.dialogs[0].validateSetup(), '');
  f.dialogs[0].onSaved();
  const result = await Promise.all([text, image]);
  assert.equal(result[0].user_notes, 'Original notes');
  assert.equal(result[1].image_path, 'photo.png');
});

test('closing Builder cancels pending setup and unregisters its handler', async () => {
  const f = fixture();
  const pending = f.c.prepareLlmRequest('/generate', { text_runner: 'llm_api' });
  f.dispose();
  await assert.rejects(pending, error => error.code === 'LLM_SETUP_CANCELLED');
  const payload = { text_runner: 'llm_api' };
  assert.equal(await f.c.prepareLlmRequest('/generate', payload), payload);
});

test('configured LLM and non-LLM requests do not open setup or change payloads', async () => {
  const f = fixture();
  for (const payload of [
    { text_runner: 'llm_api', llm_api_key: 'test-only-key', llm_api_model: 'model' },
    { prompt: 'a normal render', seed: 10 },
    { runner: { text_runner: 'llm_api' }, keys: {} },
  ]) assert.equal(await f.c.prepareLlmRequest('/some_route', payload), payload);
  assert.equal(f.dialogs.length, 0);
});

test('configuration checks cover local models, vision projectors and server choices', () => {
  const { c } = fixture();
  assert.match(c.missingLlmSetup({ text_runner: 'lm_studio' }), /already loaded/);
  assert.match(c.missingLlmSetup({ text_runner: 'own_server' }), /URL/);
  assert.match(c.missingLlmSetup({ text_runner: 'own_server', own_server_url: 'http://localhost/v1' }), /model/);
  assert.match(c.missingLlmSetup({ text_runner: 'qwen_local' }), /GGUF/);
  assert.match(c.missingLlmSetup({ text_runner: 'builtin', model_file: 'model.gguf' }, true), /mmproj/);
  assert.equal(c.missingLlmSetup({ text_runner: 'builtin', model_file: 'model.gguf' }), '');
});

test('postJson waits for setup before starting its network timeout and sends once', async () => {
  const f = fixture();
  let requests = 0, timers = 0;
  Object.assign(f.c, {
    errorLogContext: () => ({}), errorLogHeaders: () => ({}), reportProjectError() {},
    AbortController, setTimeout: () => ++timers, clearTimeout() {},
    api: { fetchApi: async (url, request) => {
      requests++;
      assert.equal(JSON.parse(request.body).llm_api_key, 'test-only-key');
      return { ok: true, json: async () => ({ ok: true, description: 'Resumed' }) };
    } },
    ownServerPayloadTimeoutMs: () => 0,
  });
  vm.runInContext(functionSource(readBuilderModule('comfy_api.mjs'), 'postJson'), f.c);
  const pending = f.c.postJson('/describe_reference_image', { text_runner: 'llm_api' });
  await Promise.resolve();
  assert.equal(requests, 0);
  assert.equal(timers, 0);
  f.update({ text_runner: 'llm_api', llm_api_key: 'test-only-key', llm_api_model: 'model' });
  f.dialogs[0].onSaved();
  assert.equal((await pending).description, 'Resumed');
  assert.equal(requests, 1);
  assert.equal(timers, 1);
});

function modalFixture() {
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.attributes = {}; this.value = ''; }
    append(...nodes) { for (const node of nodes) { node.remove(); node.parentElement = this; this.children.push(node); } }
    prepend(node) { node.remove(); node.parentElement = this; this.children.unshift(node); }
    insertBefore(node, next) { node.remove(); node.parentElement = this; this.children.splice(next ? this.children.indexOf(next) : this.children.length, 0, node); }
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(node => node !== this); this.parentElement = null; }
    addEventListener() {}
    setAttribute(name, value) { this.attributes[name] = value; }
    get options() { return this.children; }
    get nextSibling() { return this.parentElement?.children[this.parentElement.children.indexOf(this) + 1] || null; }
    set innerHTML(value) { this.children = []; }
    focus() { this.focused = true; }
  }
  const focus = new Element('input'), parent = new Element('parent-tool'), mmproj = new Element('select');
  parent.append(mmproj);
  const listeners = {};
  const document = { activeElement: focus, body: new Element('body'), createElement: tag => new Element(tag),
    addEventListener: (name, handler) => { listeners[name] = handler; }, removeEventListener: name => { delete listeners[name]; } };
  document.body.append(parent);
  const state = { textGemmaRunner: 'llm_api', llmApiModel: 'test-model' };
  const posts = [], events = [];
  const c = vm.createContext({ document, state,
    makeButton: text => Object.assign(new Element('button'), { textContent: text }),
    makeInput: (value, type) => Object.assign(new Element('input'), { value, type }),
    makeSelect: (options, value) => {
      const select = Object.assign(new Element('select'), { value });
      select.append(...options.map(value => Object.assign(new Element('option'), { value })));
      return select;
    },
    makeField: (label, input) => { const field = new Element('label'); field.textContent = label; field.append(input); return field; },
    getJson: async () => ({ providers: [{ id: 'openai', models: ['test-model'], default_model: 'test-model' }] }),
    postJson: async (url, payload) => { posts.push({ url, payload }); return {}; },
    globalLlmSettingsPayload: () => ({}), rememberGlobalLlmKeys() {}, toast() {},
    applyGemmaLocalTokenLimits() {}, GEMMA_LOCAL_MIN_TOKENS: 512,
  });
  for (const name of ['normalizeGemmaContextLimit', 'normalizeOutputTokenLimit', 'normalizeGemmaGpuLayers',
    'normalizeLmStudioContextLimit', 'normalizeOwnServerTimeoutMinutes']) c[name] = value => Number(value) || 512;
  vm.runInContext(readBuilderModule('gemma_runner.mjs'), c);
  const runner = c.createGemmaRunner({ state, mmprojSelect: mmproj, t2iTextGemmaModelSelect: new Element('select'),
    autoSaveSessionQuiet: async () => events.push('saved'), syncBuilderLlmModelSelectsFromRunner() {},
    updatePromptRunnerButtonLabels() {}, textGemmaRunnerPayload: () => ({}), createProgressWindow() {} });
  const walk = node => [node, ...node.children.flatMap(walk)];
  return { c, runner, state, document, parent, mmproj, focus, listeners, posts, events,
    find: text => walk(document.body).find(node => node.textContent === text) };
}

test('setup dialog saves and resumes over the existing tool, restoring focus and shared controls', async () => {
  const f = modalFixture();
  let resumed = 0;
  const dialog = f.runner.openGemmaRunnerModal({ setupMessage: 'Enter your API key',
    validateSetup: () => f.state.llmApiKey ? '' : 'Enter your API key', onSaved: () => resumed++ });
  await Promise.resolve(); await Promise.resolve();
  assert.match(dialog.style.cssText, /z-index:100100/);
  assert.equal(f.parent.parentElement, f.document.body);
  const save = f.find('Save & Continue');
  await save.onclick();
  assert.equal(resumed, 0);
  assert.equal(f.posts.length, 0, 'invalid setup stays open without saving');
  const apiKey = f.find('API key').children[0];
  apiKey.value = 'test-only-key';
  await save.onclick();
  assert.equal(resumed, 1);
  assert.equal(f.posts.length, 1);
  assert.equal(f.posts[0].url, '/vrgdg/music_builder/llm_runner');
  assert.equal(dialog.parentElement, null);
  assert.equal(f.mmproj.parentElement, f.parent);
  assert.equal(f.focus.focused, true);
  assert.equal(f.listeners.keydown, undefined);
});

test('Escape cancels setup without dismissing the original tool or running its action', () => {
  const f = modalFixture();
  let cancelled = 0;
  f.runner.openGemmaRunnerModal({ setupMessage: 'Choose a model', onSaved: () => assert.fail('Must not resume'), onCancel: () => cancelled++ });
  let stopped = false;
  f.listeners.keydown({ key: 'Escape', preventDefault() {}, stopImmediatePropagation() { stopped = true; } });
  assert.equal(cancelled, 1);
  assert.equal(stopped, true);
  assert.equal(f.parent.parentElement, f.document.body);
  assert.equal(f.mmproj.parentElement, f.parent);
  assert.equal(f.posts.length, 0);
});
