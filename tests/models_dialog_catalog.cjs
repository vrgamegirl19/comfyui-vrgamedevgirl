const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const { fire, loadDialog, makeContext, walk } = require('./models_dialog_harness.cjs');

// The Models dialog with the real dialogs.mjs and the real model catalog (core/model_catalog.json), served the way the Builder's installer serves
// it. Every file of the catalog has to be on a card of its tab: a card whose title is not a group name of the catalog would show nothing.

const CATALOG = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'core', 'model_catalog.json'), 'utf8'));
const ROWS = CATALOG.map((entry) => ({
  ...entry, group: entry.group || entry.category, installed: false, size: 0, path: '', destination: `C:/ComfyUI/models/${entry.folder}`,
  job: { state: 'idle', downloaded: 0, total: 0, error: '' },
}));
const flush = () => new Promise((resolve) => setImmediate(resolve));

async function openTab(label, rows = ROWS, status = {}) {
  const calls = [];
  const context = makeContext({
    getJson: async (url) => { calls.push(['get', url]); return url.endsWith('/models/status') ? { models: rows, ...status } : {}; },
    postJson: async (url, body) => { calls.push(['post', url, JSON.parse(JSON.stringify(body))]); return { models: rows }; },
  });
  loadDialog(context);
  context.args = { modelSelections: [] };
  vm.runInContext('showModelDownloadModal(args)', context);
  const all = () => walk(context.document.body);
  const button = (text) => all().find((node) => node.tag === 'button' && node.textContent === text);
  fire(button('Download models'), 'click');
  fire(button(label), 'click');
  await flush();
  await flush();
  return { all, button, calls, text: () => all().map((node) => node.textContent || '').join('\n') };
}

for (const [label, category] of [['Image Models', 'image'], ['LLM Models', 'llm'], ['MiniMax H3', 'minimax']]) {
  test(`${label}: every file of the catalog is on a card, and no file of another tab is`, async () => {
    const { text, calls } = await openTab(label);
    assert.ok(calls.some((call) => call[1] === '/vrgdg/video_builder/models/status'), 'the tab asks the installer what is there');
    const shown = text();
    for (const entry of CATALOG) {
      const there = shown.includes(`${entry.filename} · `);
      assert.equal(there, entry.category === category, `${entry.id} (${entry.category}) ${there ? 'is' : 'is not'} on the ${label} tab`);
    }
  });
}

test('the file a card shares with another is on both cards', async () => {
  const { text } = await openTab('Image Models');
  const occurrences = text().split('qwen_3_4b.safetensors · ').length - 1;
  assert.equal(occurrences, 2, 'Qwen 3 4B is on the ZImage and the Flux/Klein 4B cards');
  assert.equal(text().split('flux2-vae.safetensors · ').length - 1, 3, 'Flux2 VAE is on the MOSS-Image, Flux/Klein 4B and Ernie Image cards');
});

test('Install Recommended shows on the MiniMax H3 and LLM tabs and not on Image Models, where nothing is recommended', async () => {
  for (const [label, shown] of [['MiniMax H3', true], ['LLM Models', true], ['Image Models', false]]) {
    const { all } = await openTab(label);
    const buttons = all().filter((node) => node.tag === 'button' && /^Install Recommended/.test(node.textContent || ''));
    assert.ok(buttons.length > 0, `${label} has the button`);
    assert.equal(buttons[0].style.display !== 'none', shown, label);
  }
});

test('the Hugging Face token box is on every install tab once the installer says whether a token is saved, and not on an older installer', async () => {
  const boxOf = (tab) => tab.all().find((node) => node.tag === 'div' && node.textContent === 'Hugging Face token').parentElement;
  for (const label of ['Image Models', 'LLM Models', 'MiniMax H3']) {
    const fresh = await openTab(label, ROWS, { hf_token_saved: false });
    assert.equal(boxOf(fresh).style.display, 'flex', `${label}: the box is shown`);
    assert.match(fresh.text(), /No token is saved yet\./);
    const saved = await openTab(label, ROWS, { hf_token_saved: true });
    assert.match(saved.text(), /A token is saved on this computer\./);
    const older = await openTab(label);
    assert.equal(boxOf(older).style.display, 'none', `${label}: an installer that does not know about tokens has no box`);
  }
});

test('the Flux/Klein 9B card sends the person to the token box', async () => {
  const { text } = await openTab('Image Models', ROWS, { hf_token_saved: false });
  assert.match(text(), /save your Hugging Face token in the box at the top of this tab/);
});

test('an older Builder whose installer has none of these files still shows the plain download links', async () => {
  const { text, all } = await openTab('MiniMax H3', ROWS.filter((row) => row.category !== 'minimax'));
  const shown = text();
  assert.match(shown, /Required diffusion model, Qwen3-VL text encoder, video VAE, and audio VAE/);
  assert.equal(all().some((node) => node.tag === 'button' && node.textContent === 'Install Selected'), false);
});
