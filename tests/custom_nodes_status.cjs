const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// What the Custom Nodes dialog shows and allows (web/music_video_builder/custom_nodes_status.mjs).
function load() {
  const context = vm.createContext({});
  vm.runInContext(readBuilderModule('custom_nodes_status.mjs'), context);
  return context;
}

const plain = (value) => JSON.parse(JSON.stringify(value));
const upscalerPack = { id: 'minimax_h3_latent_upscaler', label: 'Comfyui_Minimax_h3_latent_Upscaler' };
const builtInStatus = { id: 'minimax_h3_latent_upscaler', installed: false, built_in: true, built_in_since: '0.44.0' };

test('the red note is the ComfyUI-Manager text unless the server gave its own reason', () => {
  const c = load();
  const fallback = c.managerNoteText({});
  assert.match(fallback, /ComfyUI-Manager is not installed in ComfyUI’s Python.*--enable-manager.*manager_requirements\.txt/);
  assert.equal(c.managerNoteText(undefined), fallback);
  assert.equal(c.managerNoteText({ manager_note: '  ' }), fallback);
  assert.equal(c.managerNoteText({ manager_note: 'ComfyUI is not reachable.' }), 'ComfyUI is not reachable.');
});

test('a pack built into ComfyUI says so, and only the packs that are neither installed nor built in are asked for', () => {
  const c = load();
  assert.deepEqual(plain(c.nodeBadge(upscalerPack, builtInStatus)), { text: 'Built in', kind: 'ok' });
  assert.deepEqual(plain(c.nodeBadge(upscalerPack, { installed: false, built_in: false })), { text: 'Missing', kind: 'missing' });
  assert.deepEqual(plain(c.nodeBadge(upscalerPack, { installed: true, built_in: false })), { text: 'Installed', kind: 'ok' });
  assert.deepEqual(plain(c.nodeBadge({ id: 'vrgdg', current: true }, undefined)), { text: 'Installed', kind: 'ok' });
  assert.deepEqual(plain(c.nodeBadge(upscalerPack, undefined)), { text: 'Missing', kind: 'missing' });

  const nodes = [
    { id: 'kjnodes', installed: true },
    { id: 'gguf', installed: false },
    builtInStatus,
    { id: 'old_comfyui_upscaler', installed: false, built_in: false },
  ];
  assert.deepEqual(plain(c.missingNodeIds(nodes)), ['gguf', 'old_comfyui_upscaler']);
  assert.deepEqual(plain(c.missingNodeIds(undefined)), []);
});

test('the built-in line names the ComfyUI version it came with and the one in use', () => {
  const c = load();
  assert.equal(c.builtInLine(builtInStatus, '0.45.0'), 'Built into ComfyUI 0.44.0 and newer, so this pack is not needed. Yours is 0.45.0. Only older ComfyUI needs it.');
  assert.equal(c.builtInLine(builtInStatus), 'Built into ComfyUI 0.44.0 and newer, so this pack is not needed. Only older ComfyUI needs it.');
  assert.equal(c.builtInLine({ installed: false, built_in: false }, '0.43.0'), '');
  assert.equal(c.builtInLine(undefined, '0.43.0'), '');
});

test('the button of a card is Included, Built in, Reinstall or Install, and needs the Manager only to install', () => {
  const c = load();
  const state = (item, status, manager) => plain(c.installButtonState(item, status, manager));
  assert.deepEqual(state({ id: 'vrgdg', current: true }, undefined, true), { text: 'Included', disabled: true });
  assert.deepEqual(state(upscalerPack, builtInStatus, true), { text: 'Built in', disabled: true });
  assert.deepEqual(state(upscalerPack, { installed: true }, true), { text: 'Reinstall Requirements', disabled: false });
  assert.deepEqual(state(upscalerPack, { installed: false }, true), { text: 'Install', disabled: false });
  assert.deepEqual(state(upscalerPack, { installed: false }, false), { text: 'Install', disabled: true });
  // Built in wins over a missing Manager: there is nothing to install either way.
  assert.deepEqual(state(upscalerPack, builtInStatus, false), { text: 'Built in', disabled: true });
});
