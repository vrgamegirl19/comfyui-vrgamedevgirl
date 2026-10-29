import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// refmod_card.mjs imports ComfyUI's scripts/api.js, which only exists in the browser. The loader stubs it with a shared
// `api` object, so this test swaps in a mocked fetchApi.
register('./stub_comfy_loader.mjs', import.meta.url);
const { api } = await import('./stub/scripts/api.js');

// Just enough DOM for buildRefmodPicker.
class FakeOption {
  constructor(text = '', value = '') { this.text = text; this.value = value; }
}
class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.style = {};
    this.listeners = {};
    this._value = '';
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  dispatch(type) {
    for (const listener of this.listeners[type] || []) listener({ type, target: this });
    if (typeof this[`on${type}`] === 'function') this[`on${type}`]({ type, target: this });
  }
  get options() { return this.children.filter((child) => child instanceof FakeOption); }
  get value() { return this._value; }
  set value(next) {
    if (this.tagName !== 'select') { this._value = next; return; }
    this._value = this.options.some((option) => option.value === next) ? next : '';
  }
}
globalThis.Option = FakeOption;
globalThis.document = { createElement: (tag) => new FakeElement(tag), createTextNode: (text) => ({ text }) };
globalThis.window = { dispatchEvent() {}, confirm: () => true };
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };

const card = await import('../web/music_video_builder/refmod_card.mjs');
const { buildRefmodPicker, loadRefmodLibrary } = card;

const entry = (name, extra = {}) => ({
  name, folder: name.split('/')[0], type: name.split('/')[0], kind: 'video', frames: 4, tokens: 1536,
  canvas: [512, 512], description: '', has_preview: false, path: `models/refmods/${name}.safetensors`, ...extra,
});

// The server's library, and every request made to it. `hold` makes the next requests wait until released.
const server = { library: [], calls: [], held: [], hold: false, fail: false };
api.fetchApi = (url) => {
  server.calls.push(url);
  // The folders are read when the request arrives, so a held request answers with the library as it was then.
  const refmods = server.library.map((item) => ({ ...item }));
  const failed = server.fail;
  const respond = () => {
    if (failed) return Promise.reject(new Error('offline'));
    return Promise.resolve({ ok: true, json: async () => ({ ok: true, refmods }) });
  };
  if (!server.hold) return respond();
  return new Promise((resolve, reject) => server.held.push(() => respond().then(resolve, reject)));
};
const flush = () => new Promise((resolve) => setImmediate(resolve));
const releaseAll = async () => {
  const pending = server.held.splice(0);
  for (const release of pending) release();
  await flush();
};
const reset = async (library = []) => {
  server.library = library;
  server.calls = [];
  server.held = [];
  server.hold = false;
  server.fail = false;
  card.refmodLibraryChanged?.();
  await loadRefmodLibrary(true);
  server.calls = [];
};
const selectsIn = (node, found = []) => {
  if (node instanceof FakeElement) {
    if (node.tagName === 'select') found.push(node);
    for (const child of node.children) selectsIn(child, found);
  }
  return found;
};
const locationCard = () => ({ id: 'loc_safehouse', name: 'Safehouse', reference_type: 'environment', source: 'refmod' });
const openPicker = async (item = locationCard()) => {
  const panel = buildRefmodPicker({ item, kind: 'location' });
  await flush();
  const selects = selectsIn(panel);
  return { item, panel, modSelect: selects[selects.length - 1] };
};
const choices = (select) => select.options.map((option) => option.value).filter(Boolean);

test('opening the Saved RefMod dropdown shows a RefMod made after the card was built', async () => {
  await reset([entry('background/safehouse_floors')]);
  const { modSelect } = await openPicker();
  assert.deepEqual(choices(modSelect), ['background/safehouse_floors']);

  // Made in RefMods Studio (or another tab, or through the API) while the Builder stays open.
  server.library.push(entry('background/rooftop'));
  modSelect.dispatch('pointerdown');
  modSelect.dispatch('focus');
  await flush();
  assert.deepEqual(choices(modSelect), ['background/safehouse_floors', 'background/rooftop']);
  assert.equal(server.calls.length, 1, 'pointerdown and focus share one request');
});

test('opening the dropdown also drops a RefMod deleted elsewhere and flags the card that used it', async () => {
  await reset([entry('background/safehouse_floors'), entry('background/rooftop')]);
  const item = { ...locationCard(), refmod: { name: 'background/rooftop', kind: 'video', tokens: 1536, strength: 1 } };
  const { modSelect, panel } = await openPicker(item);
  assert.equal(modSelect.value, 'background/rooftop');

  server.library = [entry('background/safehouse_floors')];
  modSelect.dispatch('focus');
  await flush();
  // The dropdown no longer offers it and the info line says it is missing, as when a card is built after a delete.
  assert.deepEqual(choices(modSelect), ['background/safehouse_floors']);
  const texts = JSON.stringify(panel, (key, value) => (key === 'listeners' ? undefined : value));
  assert.match(texts, /background\/rooftop was not found in models\/refmods/);
});

test('loads that overlap share one request', async () => {
  await reset([entry('background/safehouse_floors')]);
  server.hold = true;
  const loads = [loadRefmodLibrary(true), loadRefmodLibrary(true), loadRefmodLibrary()];
  assert.equal(server.calls.length, 1);
  await releaseAll();
  const results = await Promise.all(loads);
  assert.deepEqual(results.map((library) => library.map((item) => item.name)), Array(3).fill(['background/safehouse_floors']));
});

test('many cards opened together make one request', async () => {
  await reset([entry('background/safehouse_floors')]);
  const pickers = [await openPicker(), await openPicker(), await openPicker()];
  server.calls = [];
  server.library.push(entry('background/rooftop'));
  for (const { modSelect } of pickers) modSelect.dispatch('pointerdown');
  await flush();
  assert.equal(server.calls.length, 1);
  for (const { modSelect } of pickers) assert.ok(choices(modSelect).includes('background/rooftop'));
});

test('a RefMods Studio save makes the next card read the folders again', async () => {
  await reset([entry('background/safehouse_floors')]);
  server.library.push(entry('background/rooftop'));
  assert.equal(typeof card.refmodLibraryChanged, 'function', 'refmod_card.mjs exports refmodLibraryChanged');
  card.refmodLibraryChanged();
  const { modSelect } = await openPicker();
  assert.ok(choices(modSelect).includes('background/rooftop'));
});

test('a read that started before a change does not answer a load made after it', async () => {
  await reset([entry('background/safehouse_floors')]);
  server.hold = true;
  const early = loadRefmodLibrary(true);
  server.library.push(entry('background/rooftop'));
  card.refmodLibraryChanged?.();
  const late = loadRefmodLibrary(true);
  assert.equal(server.calls.length, 2);
  await releaseAll();
  assert.deepEqual((await early).map((item) => item.name), ['background/safehouse_floors']);
  assert.deepEqual((await late).map((item) => item.name), ['background/safehouse_floors', 'background/rooftop']);
  assert.deepEqual((await loadRefmodLibrary()).map((item) => item.name), ['background/safehouse_floors', 'background/rooftop']);
});

test('an unchanged library leaves the open dropdown alone', async () => {
  await reset([entry('background/safehouse_floors')]);
  const { modSelect } = await openPicker();
  const before = modSelect.options;
  modSelect.dispatch('focus');
  await flush();
  assert.equal(server.calls.length, 1);
  assert.deepEqual(modSelect.options, before);
  modSelect.options.forEach((option, index) => assert.equal(option, before[index], 'the same option elements'));
});

test('a failed refresh keeps the list the picker already has', async () => {
  await reset([entry('background/safehouse_floors')]);
  const { modSelect } = await openPicker();
  server.fail = true;
  modSelect.dispatch('focus');
  await flush();
  assert.deepEqual(choices(modSelect), ['background/safehouse_floors']);
});

test('syncRefmodCards brings cards in line with the library and leaves the rest alone', () => {
  const library = [entry('identity/hero', { frames: 5, tokens: 2880 }), entry('background/stage', { kind: 'image', frames: 1, tokens: 576 })];
  const cards = [
    { id: 'hero', name: 'Hero', refmod: { name: 'identity/hero', kind: 'video', tokens: 3456, frames: 6, strength: 0.8 } },
    { id: 'stage', refmod: { name: 'background/stage', kind: 'video', tokens: 900, frames: 4, strength: 1 } },
    { id: 'gone', refmod: { name: 'identity/gone', kind: 'video', tokens: 700, frames: 2, strength: 1 } },
    { id: 'plain', source: 'image' },
    null,
  ];
  assert.equal(card.syncRefmodCards(cards, library), true);
  assert.deepEqual(cards[0].refmod, { name: 'identity/hero', kind: 'video', tokens: 2880, frames: 5, strength: 0.8 });
  assert.equal(cards[0].name, 'Hero');
  assert.deepEqual(cards[1].refmod, { name: 'background/stage', kind: 'image', tokens: 576, frames: 1, strength: 1 });
  assert.deepEqual(cards[2].refmod, { name: 'identity/gone', kind: 'video', tokens: 700, frames: 2, strength: 1 }, 'a RefMod missing from the library keeps its numbers');
  assert.equal(card.syncRefmodCards(cards, library), false, 'a second call has nothing to change');
  assert.equal(card.syncRefmodCards(undefined, undefined), false);
});

test('a card saved with old numbers takes the library\'s when its picker loads', async () => {
  await reset([entry('identity/hero', { frames: 5, tokens: 2880 })]);
  const item = {
    id: 'hero', name: 'Hero', reference_type: 'character', source: 'refmod',
    refmod: { name: 'identity/hero', folder: 'identity', type: 'identity', kind: 'video', tokens: 3456, frames: 6, strength: 0.8 },
  };
  const changes = [];
  const panel = buildRefmodPicker({ item, kind: 'subject', onChange: (...args) => changes.push(args) });
  await flush();
  assert.deepEqual(item.refmod, { name: 'identity/hero', folder: 'identity', type: 'identity', kind: 'video', tokens: 2880, frames: 5, strength: 0.8 });
  assert.deepEqual(changes, [[false]], 'the Builder is told once, without a redraw');

  // Nothing changed in the library, so opening the dropdown does not tell the Builder again.
  const selects = selectsIn(panel);
  selects[selects.length - 1].dispatch('focus');
  await flush();
  assert.deepEqual(changes, [[false]]);
});
