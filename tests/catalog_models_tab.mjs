import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// catalog_models_tab.mjs imports ComfyUI's scripts/api.js, which only exists in the browser. The loader stubs it, so this test swaps in a
// mocked fetchApi that plays the Builder's model installer routes (core/model_downloads.py).
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
  tick(checked) {
    this.checked = checked;
    for (const listener of this.listeners.change || []) listener({ type: 'change', target: this });
  }
}
const events = [];
const opened = [];
FakeElement.prototype.appendChild = function appendChild(node) { this.children.push(node); };
FakeElement.prototype.remove = function remove() {};
globalThis.document = {
  createElement: (tag) => new FakeElement(tag),
  createTextNode: (text) => ({ text }),
  body: new FakeElement('body'),
};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
globalThis.window = {
  dispatchEvent: (event) => { events.push(event); },
  setTimeout: (fn) => setTimeout(fn, 1),
  clearTimeout: (id) => clearTimeout(id),
  open: (...args) => { opened.push(args); },
};

const tab = await import('../web/music_video_builder/catalog_models_tab.mjs');
const { renderCatalogTab, canInstallModel, needsHfToken, catalogModelsForGroup, downloadTotals, catalogRecommended, catalogSelected, catalogInstallLabel, catalogGroupSummary, modelSizeText } = tab;

const GB = 1024 ** 3;
const model = (id, extra = {}) => ({
  id, label: `Label ${id}`, category: 'image', groups: ['Alpha'], default: false, gated: false, format: 'safetensors', folder: 'vae',
  filename: `${id}.safetensors`, download_size: 2 * GB, installed: false, size: 0, path: '', destination: 'C:/ComfyUI/models/vae',
  note: '', page: `https://huggingface.co/x/${id}`, job: { state: 'idle', downloaded: 0, total: 0, error: '' }, ...extra,
});

test('a file can be started when it is missing, idle and not behind a login', () => {
  assert.equal(canInstallModel(model('a')), true);
  assert.equal(canInstallModel(model('a', { installed: true })), false);
  assert.equal(canInstallModel(model('a', { gated: true })), false);
  assert.equal(canInstallModel(model('a', { job: { state: 'downloading' } })), false);
  assert.equal(canInstallModel(model('a', { job: { state: 'error', error: 'x' } })), true, 'a failed file can be tried again');
  assert.equal(canInstallModel(null), false);
  // A gated file waits for the Builder's Hugging Face token.
  assert.equal(needsHfToken(model('a', { gated: true })), true);
  assert.equal(needsHfToken(model('a', { gated: true, token_saved: true })), false);
  assert.equal(needsHfToken(model('a')), false);
  assert.equal(canInstallModel(model('a', { gated: true, token_saved: true })), true);
});

test('a card lists its own category and group, and a shared file is on both of its cards', () => {
  const models = [
    model('own', { groups: ['Alpha'] }),
    model('shared', { groups: ['Alpha', 'Beta'] }),
    model('other_tab', { category: 'llm', groups: ['Alpha'] }),
    model('beta_only', { groups: ['Beta'] }),
  ];
  assert.deepEqual(catalogModelsForGroup(models, 'image', 'Alpha').map((item) => item.id), ['own', 'shared']);
  assert.deepEqual(catalogModelsForGroup(models, 'image', 'Beta').map((item) => item.id), ['shared', 'beta_only']);
  assert.deepEqual(catalogModelsForGroup(null, 'image', 'Alpha'), []);
});

test('downloadTotals, labels and summaries count only what would still be downloaded', () => {
  const models = [model('a'), model('b', { installed: true }), model('c', { gated: true }), model('d', { download_size: 0 })];
  assert.deepEqual(downloadTotals(models), { count: 2, bytes: 2 * GB });
  assert.equal(catalogInstallLabel('Install Selected', models), 'Install Selected (2 files, 2.00 GB)');
  assert.equal(catalogInstallLabel('Install Selected', [model('a')]), 'Install Selected (1 file, 2.00 GB)');
  assert.equal(catalogInstallLabel('Install Selected', []), 'Install Selected');
  assert.equal(catalogGroupSummary(models), '1 of 4 installed · 2.00 GB to download');
  assert.equal(catalogGroupSummary([model('a', { installed: true })]), '1 of 1 installed');
  assert.equal(modelSizeText(model('d', { download_size: 0 })), 'size unknown');
});

test('recommended files are the default ones that are missing, and a ticked file shared by two cards counts once', () => {
  const models = [model('a', { default: true }), model('b', { default: true, installed: true }), model('c'), model('d', { default: true, gated: true })];
  assert.deepEqual(catalogRecommended(models).map((item) => item.id), ['a']);
  const picked = catalogSelected([...models, models[2]], new Set(['a', 'b', 'c', 'd', 'c']));
  assert.deepEqual(picked.map((item) => item.id), ['a', 'c', 'c'], 'an installed or gated file is never started');
});

// A pretend installer: starting a file moves it along by one step per status call, then installs it.
function installer(models) {
  const state = { calls: [], models: structuredClone(models), running: new Set() };
  const advance = () => {
    for (const item of state.models) {
      if (!state.running.has(item.id)) continue;
      if (item.job.state === 'downloading' && item.job.downloaded >= 50) {
        item.installed = true;
        item.size = item.download_size;
        item.path = `C:/ComfyUI/models/${item.folder}/${item.filename}`;
        item.job = { state: 'done', downloaded: 100, total: 100, error: '' };
        state.running.delete(item.id);
      } else {
        item.job = { state: 'downloading', downloaded: item.job.downloaded + 50, total: 100, error: '' };
      }
    }
  };
  api.fetchApi = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    state.calls.push({ url, body });
    if (url.endsWith('/models/install')) {
      for (const id of body.ids) {
        state.running.add(id);
        state.models.find((item) => item.id === id).job = { state: 'downloading', downloaded: 0, total: 100, error: '' };
      }
      return { ok: true, json: async () => ({ ok: true, started: body.ids, models: structuredClone(state.models) }) };
    }
    if (url.endsWith('/models/cancel')) {
      for (const id of body.ids) {
        state.running.delete(id);
        state.models.find((item) => item.id === id).job = { state: 'cancelled', downloaded: 0, total: 0, error: '' };
      }
      return { ok: true, json: async () => ({ ok: true, cancelled: body.ids }) };
    }
    advance();
    return { ok: true, json: async () => ({ ok: true, models: structuredClone(state.models) }) };
  };
  return state;
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 25));
const settle = async () => { for (let i = 0; i < 12; i += 1) await flush(); };
const textOf = (node) => (node.text ?? '') + (node.textContent || '') + (node.children || []).map(textOf).join('|');
const find = (node, test, found = []) => {
  if (test(node)) found.push(node);
  for (const child of node.children || []) find(child, test, found);
  return found;
};
const buttons = (node) => find(node, (item) => item.tagName === 'button');
// The tab has a text field of its own now (the Hugging Face token), so a check box is an input that is one.
const checkboxes = (node) => find(node, (item) => item.tagName === 'input' && item.type === 'checkbox');
const buttonStarting = (node, prefix) => buttons(node).find((button) => button.textContent.startsWith(prefix));

const CARDS = [{ title: 'Alpha', note: 'The first card.' }, { title: 'Beta', note: 'The second card.' }];
const SERVER = [
  model('a1'), model('shared', { groups: ['Alpha', 'Beta'] }), model('b1', { groups: ['Beta'] }), model('b2', { groups: ['Beta'], gated: true }),
  model('llm_one', { category: 'llm' }),
];

test('the tab shows a card per group with a check box per file and nothing ticked', async () => {
  installer(SERVER);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS, intro: 'Pick some.' });
  await settle();
  const text = textOf(body);
  assert.match(text, /Pick some\./);
  assert.match(text, /The first card\./);
  assert.match(text, /Label a1/);
  assert.match(text, /a1\.safetensors · 2\.00 GB/);
  assert.doesNotMatch(text, /Label llm_one/, 'another tab\'s file is not shown');
  assert.equal(buttonStarting(body, 'Install Selected').disabled, true);
  assert.equal(buttonStarting(body, 'Install Recommended').style.display, 'none', 'nothing here is recommended');
  assert.equal(checkboxes(body).filter((box) => box.checked).length, 0);
  stop();
});

test('ticking files and clicking Install Selected installs exactly those, then tells the Builder to reload its lists', async () => {
  events.length = 0;
  const server = installer(SERVER);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  // The first check box of a card is the card's own. The file boxes follow in the order of the files.
  const [, a1Box, sharedBox] = checkboxes(body);
  a1Box.tick(true);
  sharedBox.tick(true);
  assert.equal(buttonStarting(body, 'Install Selected').textContent, 'Install Selected (2 files, 4.00 GB)');
  buttonStarting(body, 'Install Selected').click();
  await settle();
  const install = server.calls.find((call) => call.url === '/vrgdg/video_builder/models/install');
  assert.deepEqual(install.body.ids, ['a1', 'shared']);
  const text = textOf(body);
  assert.equal((text.match(/Installed \(2\.00 GB\)/g) || []).length >= 2, true);
  const changed = events.filter((event) => event.type === 'vrgdg-models-changed').flatMap((event) => event.detail.ids).sort();
  assert.deepEqual(changed, ['a1', 'shared']);
  assert.equal(buttonStarting(body, 'Install Selected').disabled, true, 'installed files are no longer selected');
  // Nothing keeps polling once everything has finished.
  const before = server.calls.length;
  await settle();
  assert.equal(server.calls.length, before);
  stop();
});

test('a card\'s own check box ticks every file that can still be installed, and skips a gated one', async () => {
  const server = installer(SERVER);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  const betaBox = checkboxes(body).filter((box) => box.title.startsWith('Tick or untick'))[1];
  betaBox.tick(true);
  assert.equal(buttonStarting(body, 'Install Selected').textContent, 'Install Selected (2 files, 4.00 GB)', 'shared and b1, not the gated b2');
  buttonStarting(body, 'Install Selected').click();
  await settle();
  const install = server.calls.find((call) => call.url === '/vrgdg/video_builder/models/install');
  assert.deepEqual(install.body.ids.sort(), ['b1', 'shared']);
  stop();
});

test('a gated file is installed like any other once the Builder has a Hugging Face token for it', async () => {
  const gatedWithToken = SERVER.map((item) => (item.id === 'b2' ? { ...item, token_saved: true, gated_message: null } : item));
  const server = installer(gatedWithToken);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  const text = textOf(body);
  assert.doesNotMatch(text, /Login needed/);
  assert.doesNotMatch(text, /Hugging Face asks for a login/);
  const gatedBox = checkboxes(body).filter((box) => !box.title.startsWith('Tick or untick'))[4];
  assert.equal(gatedBox.disabled, false, 'b2 can be ticked');
  gatedBox.tick(true);
  buttonStarting(body, 'Install Selected').click();
  await settle();
  assert.deepEqual(server.calls.find((call) => call.url.endsWith('/models/install')).body.ids, ['b2']);
  stop();
});

test('a gated file with no token says what is missing, in the words the Builder sent', async () => {
  installer(SERVER.map((item) => (item.id === 'b2' ? { ...item, gated_message: 'Add your Hugging Face token in App Settings.' } : item)));
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  assert.match(textOf(body), /Add your Hugging Face token in App Settings\./);
  stop();
});

test('a gated file has no Install button, only its page, and its check box is off', async () => {
  opened.length = 0;
  installer(SERVER);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  const text = textOf(body);
  assert.match(text, /Login needed/);
  assert.match(text, /Hugging Face asks for a login/);
  const gatedBox = checkboxes(body).find((box) => box.title.startsWith('Needs a Hugging Face login'));
  assert.equal(gatedBox.disabled, true);
  const pages = buttons(body).filter((button) => button.textContent === 'Open Model Page');
  assert.equal(pages.length, 5, 'every row has its page, the shared file twice');
  pages[4].click();
  assert.equal(opened.at(-1)[0], 'https://huggingface.co/x/b2');
  assert.equal(opened.at(-1)[1], '_blank');
  stop();
});

test('Install Recommended starts only the default files that are missing, in every card', async () => {
  const server = installer([
    model('core1', { category: 'minimax', groups: ['Alpha'], default: true }),
    model('core2', { category: 'minimax', groups: ['Alpha'], default: true, installed: true }),
    model('extra', { category: 'minimax', groups: ['Alpha'] }),
    model('lora', { category: 'minimax', groups: ['Beta'], default: true }),
  ]);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'minimax', groups: CARDS });
  await settle();
  const top = buttonStarting(body, 'Install Recommended');
  assert.equal(top.textContent, 'Install Recommended (2 files, 4.00 GB)');
  assert.notEqual(top.style.display, 'none');
  top.click();
  await settle();
  const install = server.calls.find((call) => call.url === '/vrgdg/video_builder/models/install');
  assert.deepEqual(install.body.ids, ['core1', 'lora']);
  assert.equal(server.models.find((item) => item.id === 'extra').installed, false, 'an optional file is left alone');
  stop();
});

test('a download can be cancelled from its row', async () => {
  const server = installer(SERVER);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  buttons(body).find((button) => button.textContent === 'Install').click();
  await new Promise((resolve) => setTimeout(resolve, 5));
  const cancel = buttons(body).find((button) => button.textContent === 'Cancel');
  assert.ok(cancel, 'a running download offers Cancel');
  cancel.click();
  await settle();
  const call = server.calls.find((item) => item.url === '/vrgdg/video_builder/models/cancel');
  assert.deepEqual(call.body.ids, ['a1']);
  assert.match(textOf(body), /Cancelled/);
  stop();
});

test('without any file of its category (an older Builder) the tab shows the plain links instead', async () => {
  installer([model('llm_one', { category: 'llm' })]);
  const body = new FakeElement('div');
  let fallbackCalls = 0;
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS, fallback: () => { fallbackCalls += 1; } });
  await settle();
  assert.equal(fallbackCalls, 1);
  stop();
});

test('when the installer cannot be reached the tab shows the plain links instead', async () => {
  api.fetchApi = async () => { throw new Error('connection refused'); };
  const body = new FakeElement('div');
  let fallbackCalls = 0;
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS, fallback: () => { fallbackCalls += 1; } });
  await settle();
  assert.equal(fallbackCalls, 1);
  stop();
});

test('leaving the tab stops its polling', async () => {
  const server = installer(SERVER);
  const body = new FakeElement('div');
  const stop = renderCatalogTab(body, { category: 'image', groups: CARDS });
  await settle();
  buttons(body).find((button) => button.textContent === 'Install').click();
  await new Promise((resolve) => setTimeout(resolve, 3));
  stop();
  const after = server.calls.length;
  await settle();
  assert.ok(server.calls.length - after <= 1, 'at most one request already in flight when the tab was left');
});
