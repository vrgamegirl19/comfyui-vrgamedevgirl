import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// hf_token_box.mjs (and the Models tab that holds it) import ComfyUI's scripts/api.js, which only exists in the browser. The loader stubs it, so
// this test swaps in a mocked fetchApi that plays the Builder's token routes (core/hf_token.py).
register('./stub_comfy_loader.mjs', import.meta.url);
const { api } = await import('./stub/scripts/api.js');

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.style = { cssText: '' };
    this.listeners = {};
    this.dataset = {};
    this.textContent = '';
    this.value = '';
    this.disabled = false;
    this.checked = false;
    this.title = '';
  }
  append(...nodes) { this.children.push(...nodes); }
  prepend(...nodes) { this.children.unshift(...nodes); }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  setAttribute() {}
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  click() {
    for (const listener of this.listeners.click || []) listener({ type: 'click', target: this });
    if (typeof this.onclick === 'function') return this.onclick({ type: 'click', target: this });
    return undefined;
  }
  press(key) {
    for (const listener of this.listeners.keydown || []) listener({ type: 'keydown', key, target: this });
  }
}
const opened = [];
globalThis.document = { createElement: (tag) => new FakeElement(tag), createTextNode: (text) => ({ text }), body: new FakeElement('body') };
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
globalThis.window = {
  dispatchEvent: () => {},
  setTimeout: (fn) => setTimeout(fn, 1),
  clearTimeout: (id) => clearTimeout(id),
  open: (...args) => { opened.push(args); },
};

const { renderHfTokenBox, hfTokenProblem, HF_TOKEN_URL, HF_TOKEN_TEST_URL, HF_TOKEN_PAGE } = await import('../web/music_video_builder/hf_token_box.mjs');
const { renderCatalogTab } = await import('../web/music_video_builder/catalog_models_tab.mjs');

const SECRET = 'hf_secretTOKEN123';
const flush = () => new Promise((resolve) => setTimeout(resolve, 25));
const settle = async () => { for (let i = 0; i < 12; i += 1) await flush(); };
const textOf = (node) => (node.text ?? '') + (node.textContent || '') + (node.children || []).map(textOf).join('|');
const find = (node, test, found = []) => {
  if (test(node)) found.push(node);
  for (const child of node.children || []) find(child, test, found);
  return found;
};
const buttonNamed = (node, name) => find(node, (item) => item.tagName === 'button' && item.textContent === name)[0];
const inputOf = (node) => find(node, (item) => item.tagName === 'input' && item.type === 'password')[0];
const statusOf = (box) => box.element.children[3].textContent;
// What a browser reads back from `display`: the property when set, else the one in the style text.
const displayOf = (element) => element.style.display ?? (/display:\s*none/.test(element.style.cssText) ? 'none' : 'flex');

// A pretend Builder: the token routes plus `reply(url, body)` for what each one answers (a failure is `{ status, error }`).
function server(reply) {
  const calls = [];
  api.fetchApi = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    calls.push({ url, body });
    const answer = reply(url, body);
    const failed = answer.status && answer.status >= 400;
    return { ok: !failed, status: answer.status || 200, headers: { get: () => null }, json: async () => (failed ? { ok: false, error: answer.error } : { ok: true, ...answer }) };
  };
  return calls;
}

test('a token problem is named before anything is sent', () => {
  assert.match(hfTokenProblem(''), /Paste your Hugging Face token first/);
  assert.match(hfTokenProblem('   '), /Paste your Hugging Face token first/);
  for (const bad of ['not_a_token', 'hf_two words', `hf_${'a'.repeat(500)}`]) assert.match(hfTokenProblem(bad), /starts with hf_/, bad);
  assert.equal(hfTokenProblem(`  ${SECRET}  `), '');
});

test('the box shows only when the installer says whether a token is saved, and says what it holds', () => {
  const box = renderHfTokenBox();
  assert.equal(displayOf(box.element), 'none', 'hidden until the installer has answered');
  box.setSaved(undefined);
  assert.equal(box.element.style.display, 'none', 'an installer that does not know about tokens has no box');
  box.setSaved(false);
  assert.equal(box.element.style.display, 'flex');
  assert.equal(statusOf(box), 'No token is saved yet.');
  assert.equal(buttonNamed(box.element, 'Remove Token').disabled, true, 'nothing to remove');
  box.setSaved(true);
  assert.equal(statusOf(box), 'A token is saved on this computer.');
  assert.equal(buttonNamed(box.element, 'Remove Token').disabled, false);
});

test('Save Token sends the pasted token once, trimmed, empties the field and says the gated files can be installed', async () => {
  const calls = server(() => ({ configured: true }));
  const changes = [];
  const box = renderHfTokenBox({ onChange: (saved) => changes.push(saved) });
  box.setSaved(false);
  const field = inputOf(box.element);
  assert.equal(field.type, 'password', 'the token is not shown while it is typed');
  field.value = `  ${SECRET}  `;
  await buttonNamed(box.element, 'Save Token').click();
  assert.deepEqual(calls, [{ url: HF_TOKEN_URL, body: { api_key: SECRET } }]);
  assert.equal(field.value, '', 'the token does not stay in the page');
  assert.match(statusOf(box), /Token saved\./);
  assert.deepEqual(changes, [true]);
  assert.equal(buttonNamed(box.element, 'Remove Token').disabled, false);
  assert.doesNotMatch(textOf(box.element), new RegExp(SECRET), 'nothing on the page shows the token');
  // The answer the server gives when the tab asks again does not replace the message the person just got.
  box.setSaved(true);
  assert.match(statusOf(box), /Token saved\./);
});

test('Enter in the field saves it like the button', async () => {
  const calls = server(() => ({ configured: true }));
  const box = renderHfTokenBox();
  box.setSaved(false);
  const field = inputOf(box.element);
  field.value = SECRET;
  field.press('Enter');
  await flush();
  assert.equal(calls.length, 1);
  field.value = 'x';
  field.press('a');
  await flush();
  assert.equal(calls.length, 1, 'only Enter saves');
});

test('a token that is not shaped like one is refused in the box and never sent', async () => {
  const calls = server(() => ({}));
  const changes = [];
  const box = renderHfTokenBox({ onChange: (saved) => changes.push(saved) });
  box.setSaved(false);
  await buttonNamed(box.element, 'Save Token').click();
  assert.match(statusOf(box), /Paste your Hugging Face token first/);
  inputOf(box.element).value = 'nope';
  await buttonNamed(box.element, 'Save Token').click();
  assert.match(statusOf(box), /starts with hf_/);
  assert.equal(inputOf(box.element).value, 'nope', 'what was typed stays so it can be fixed');
  assert.deepEqual([calls.length, changes.length], [0, 0]);
});

test('an error from the Builder is shown in the box and the buttons are usable again', async () => {
  server(() => ({ status: 400, error: 'The token could not be saved: disk full' }));
  const box = renderHfTokenBox();
  box.setSaved(false);
  inputOf(box.element).value = SECRET;
  await buttonNamed(box.element, 'Save Token').click();
  assert.match(statusOf(box), /disk full/);
  assert.equal(box.element.children[3].style.color, '#fca5a5');
  assert.equal(buttonNamed(box.element, 'Save Token').disabled, false);
  assert.equal(inputOf(box.element).value, SECRET, 'the token stays to try again');
});

test('Test Token checks the typed token, else the saved one, and says whose it is', async () => {
  const calls = server((url) => (url === HF_TOKEN_TEST_URL ? { connected: true, user: 'alice' } : {}));
  const box = renderHfTokenBox();
  box.setSaved(false);
  await buttonNamed(box.element, 'Test Token').click();
  assert.match(statusOf(box), /Paste a token first, or save one/);
  assert.equal(calls.length, 0, 'nothing to test');
  inputOf(box.element).value = SECRET;
  await buttonNamed(box.element, 'Test Token').click();
  assert.deepEqual(calls, [{ url: HF_TOKEN_TEST_URL, body: { api_key: SECRET } }]);
  assert.match(statusOf(box), /belongs to alice/);
  assert.equal(inputOf(box.element).value, SECRET, 'testing does not save it');
  inputOf(box.element).value = '';
  box.setSaved(true);
  await buttonNamed(box.element, 'Test Token').click();
  assert.deepEqual(calls[1], { url: HF_TOKEN_TEST_URL, body: {} }, 'no token typed: the server checks the saved one');
});

test('a token Hugging Face does not accept says so', async () => {
  server(() => ({ status: 400, error: 'Hugging Face did not accept that token. Check that you copied the whole token and that it was not revoked.' }));
  const box = renderHfTokenBox();
  box.setSaved(true);
  await buttonNamed(box.element, 'Test Token').click();
  assert.match(statusOf(box), /did not accept that token/);
});

test('Remove Token clears it on the server and tells the tab', async () => {
  const calls = server(() => ({ configured: false }));
  const changes = [];
  const box = renderHfTokenBox({ onChange: (saved) => changes.push(saved) });
  box.setSaved(true);
  await buttonNamed(box.element, 'Remove Token').click();
  assert.deepEqual(calls, [{ url: HF_TOKEN_URL, body: { api_key: '' } }]);
  assert.match(statusOf(box), /Token removed\./);
  assert.deepEqual(changes, [false]);
  assert.equal(buttonNamed(box.element, 'Remove Token').disabled, true);
});

test('Create a Token opens the Hugging Face token page in a new tab', () => {
  opened.length = 0;
  const box = renderHfTokenBox();
  box.setSaved(false);
  buttonNamed(box.element, 'Create a Token').click();
  assert.deepEqual(opened.at(-1), [HF_TOKEN_PAGE, '_blank', 'noopener,noreferrer']);
});

// -- in the Models tab ----------------------------------------------------------------------------------------------------------------------

const GB = 1024 ** 3;
const gatedModel = (saved) => ({
  id: 'flux9', label: 'Flux 9B', category: 'image', groups: ['Alpha'], default: false, gated: true, token_saved: saved,
  gated_message: saved ? null : 'Hugging Face asks for a login before it gives this file out. Paste your Hugging Face token in the box at the top of this tab.',
  format: 'safetensors', folder: 'diffusion_models', filename: 'flux9.safetensors', download_size: 2 * GB, installed: false, size: 0, path: '',
  destination: 'C:/ComfyUI/models/diffusion_models', note: '', page: 'https://huggingface.co/x/flux9', job: { state: 'idle', downloaded: 0, total: 0, error: '' },
});

// A pretend installer whose status follows the token it was given.
function installerWithToken({ saved = false, reportsToken = true } = {}) {
  const state = { saved, calls: [] };
  api.fetchApi = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    state.calls.push({ url, body });
    if (url === HF_TOKEN_URL) {
      state.saved = Boolean(body.api_key);
      return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({ ok: true, configured: state.saved }) };
    }
    const reply = { ok: true, models: [gatedModel(state.saved)], ...(reportsToken ? { hf_token_saved: state.saved } : {}) };
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => reply };
  };
  return state;
}

test('in the tab, saving a token turns a file that waited for a login into one that can be installed', async () => {
  const state = installerWithToken({ saved: false });
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: [{ title: 'Alpha', note: '' }] });
  await settle();
  assert.match(textOf(body), /Login needed/);
  assert.equal(buttonNamed(body, 'Install'), undefined, 'a file that waits for a login has no Install button');
  assert.equal(find(body, (node) => node.tagName === 'div' && node.textContent === 'Hugging Face token')[0] !== undefined, true, 'the token box is in the tab');
  inputOf(body).value = SECRET;
  await buttonNamed(body, 'Save Token').click();
  await settle();
  assert.deepEqual(state.calls.find((call) => call.url === HF_TOKEN_URL).body, { api_key: SECRET });
  assert.doesNotMatch(textOf(body), /Login needed/, 'the tab asked the installer again and the file is open now');
  assert.ok(buttonNamed(body, 'Install'), 'the file can be installed');
  assert.match(textOf(body), /Token saved\./, 'and the box still says what happened');
  assert.doesNotMatch(textOf(body), new RegExp(SECRET));
  // Removing the token puts the file back to waiting.
  await buttonNamed(body, 'Remove Token').click();
  await settle();
  assert.match(textOf(body), /Login needed/);
  stop();
});

test('in the tab, an installer that does not report a token has no box', async () => {
  installerWithToken({ saved: false, reportsToken: false });
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: [{ title: 'Alpha', note: '' }] });
  await settle();
  const title = find(body, (node) => node.tagName === 'div' && node.textContent === 'Hugging Face token')[0];
  assert.ok(title, 'the box exists in the tree');
  assert.equal(body.children.find((child) => child.children.includes(title)).style.display, 'none', 'but it is hidden');
  stop();
});
