import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// dmad_models_tab.mjs imports ComfyUI's scripts/api.js, which only exists in the browser. The loader stubs it, so this
// test swaps in a mocked fetchApi that plays the Builder's model installer routes.
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

const { renderDmadModelsTab } = await import('../web/music_video_builder/dmad_models_tab.mjs');

const LORA = 'dmad_minimax_h3_4step_full_critic_comfyui.safetensors';
const REFINER = 'minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors';
const model = (id, filename, extra = {}) => ({
  id, label: id === 'dmad_lora' ? 'DMAD 4-step LoRA' : 'Refiner LoRA', filename, installed: false, size: 0, path: '',
  destination: 'C:/ComfyUI/models/loras', note: '', page: `https://example.test/${id}`,
  job: { state: 'idle', downloaded: 0, total: 0, error: '' }, ...extra,
});

// A pretend installer: each status call moves a started download along by one step.
function installer() {
  const state = {
    calls: [],
    models: [model('dmad_lora', LORA), model('dmad_refiner', REFINER)],
    running: new Set(),
    steps: 0,
    failRefiner: false,
  };
  const advance = () => {
    for (const item of state.models) {
      if (!state.running.has(item.id)) continue;
      state.steps += 1;
      if (item.id === 'dmad_refiner' && state.failRefiner) {
        item.job = { state: 'error', downloaded: 10, total: 100, error: 'The download server answered 503 Service Unavailable.' };
        state.running.delete(item.id);
      } else if (item.job.state === 'downloading' && item.job.downloaded >= 50) {
        item.installed = true;
        item.size = 1956195976;
        item.path = `C:/ComfyUI/models/loras/${item.filename}`;
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
const textOf = (node) => (node.text ?? '') + (node.textContent || '') + (node.children || []).map(textOf).join('|');
const buttonsIn = (node, found = []) => {
  if (node.tagName === 'button') found.push(node);
  for (const child of node.children || []) buttonsIn(child, found);
  return found;
};
const buttonLabelled = (node, label) => buttonsIn(node).find((button) => button.textContent === label);
const settle = async () => { for (let i = 0; i < 12; i += 1) await flush(); };

test('the tab lists both LoRAs as missing and where they will go', async () => {
  const server = installer();
  const body = new FakeElement('div');
  const stop = renderDmadModelsTab(body);
  await settle();
  const text = textOf(body);
  assert.match(text, /DMAD 4-step LoRA/);
  assert.match(text, /Refiner LoRA/);
  assert.equal((text.match(/Not installed/g) || []).length, 2);
  assert.match(text, /Installs to: C:\/ComfyUI\/models\/loras/);
  assert.equal(buttonLabelled(body, 'Install Missing Files').disabled, false);
  assert.equal(server.calls.every((call) => call.url === '/vrgdg/video_builder/models/status'), true);
  stop();
});

test('Install Missing Files downloads both, follows progress, then tells the panels to reload their LoRA lists', async () => {
  events.length = 0;
  const server = installer();
  const body = new FakeElement('div');
  const stop = renderDmadModelsTab(body);
  await settle();
  buttonLabelled(body, 'Install Missing Files').click();
  await settle();
  const install = server.calls.find((call) => call.url === '/vrgdg/video_builder/models/install');
  assert.deepEqual(install.body.ids, ['dmad_lora', 'dmad_refiner']);
  const text = textOf(body);
  assert.equal((text.match(/Installed \(1\.82 GB\)/g) || []).length, 2);
  assert.equal(buttonLabelled(body, 'Install Missing Files').disabled, true);
  const changed = events.filter((event) => event.type === 'vrgdg-models-changed');
  assert.ok(changed.length >= 1, 'the Builder is told to reload the LoRA lists');
  assert.deepEqual(changed.flatMap((event) => event.detail.ids).sort(), ['dmad_lora', 'dmad_refiner']);
  // Nothing keeps polling once everything is installed.
  const before = server.calls.length;
  await settle();
  assert.equal(server.calls.length, before);
  stop();
});

test('a failed download shows why and offers Try Again', async () => {
  const server = installer();
  server.failRefiner = true;
  const body = new FakeElement('div');
  const stop = renderDmadModelsTab(body);
  await settle();
  buttonLabelled(body, 'Install Missing Files').click();
  await settle();
  const text = textOf(body);
  assert.match(text, /Failed: The download server answered 503 Service Unavailable\./);
  assert.ok(buttonLabelled(body, 'Try Again'));
  stop();
});

test('a download can be cancelled', async () => {
  const server = installer();
  const body = new FakeElement('div');
  const stop = renderDmadModelsTab(body);
  await settle();
  // Start only the first file by clicking its own Install button.
  buttonLabelled(body, 'Install').click();
  await new Promise((resolve) => setTimeout(resolve, 5));
  const cancelButton = buttonLabelled(body, 'Cancel');
  assert.ok(cancelButton, 'a running download offers Cancel');
  cancelButton.click();
  await settle();
  const cancelCall = server.calls.find((call) => call.url === '/vrgdg/video_builder/models/cancel');
  assert.deepEqual(cancelCall.body.ids, ['dmad_lora']);
  assert.match(textOf(body), /Cancelled/);
  stop();
});

test('Open Model Page opens the Hugging Face page in a new tab and leaving the tab stops polling', async () => {
  opened.length = 0;
  const server = installer();
  const body = new FakeElement('div');
  const stop = renderDmadModelsTab(body);
  await settle();
  buttonsIn(body).find((button) => button.textContent === 'Open Model Page').click();
  assert.equal(opened.length, 1);
  assert.match(opened[0][0], /^https:\/\/(huggingface\.co|example\.test)\//);
  assert.equal(opened[0][1], '_blank');
  buttonLabelled(body, 'Install Missing Files').click();
  await new Promise((resolve) => setTimeout(resolve, 3));
  stop();
  const after = server.calls.length;
  await settle();
  assert.ok(server.calls.length - after <= 1, 'at most one request already in flight when the tab was left');
});

test('when the installer cannot be reached the tab says so and keeps the model links', async () => {
  api.fetchApi = async () => { throw new Error('connection refused'); };
  const body = new FakeElement('div');
  const stop = renderDmadModelsTab(body);
  await settle();
  const text = textOf(body);
  assert.match(text, /Could not reach the Builder's model installer: connection refused/);
  assert.match(text, /DMAD 4-step LoRA/);
  assert.ok(buttonLabelled(body, 'Open Model Page'));
  stop();
});
