const assert = require('node:assert/strict');
const { test } = require('node:test');
const vm = require('node:vm');
const { File } = require('node:buffer');
const { readBuilderModule } = require('./builder_source.cjs');

class Element {
  constructor(tag) { Object.assign(this, { tag, children: [], style: {}, value: '', listeners: {}, disabled: false }); }
  append(...nodes) {
    for (const node of nodes) { this.children.push(node); node.parent = this; }
    if (this.tag === 'select' && !this.value) this.value = this.children[0]?.value || '';
  }
  setAttribute() {}
  addEventListener(type, fn) { this.listeners[type] = fn; }
  remove() { this.parent.children = this.parent.children.filter(node => node !== this); this.parent = null; }
}
const walk = node => [node, ...node.children.flatMap(walk)];
function fixture(folder = 'Z:/projects/one', savedDraft = null) {
  const document = { body: new Element('body'), createElement: tag => new Element(tag) };
  const state = { projectFolder: folder, projectVideoEngine: 'minimax_h3', autoBuildDraft: savedDraft, textGemmaRunner: 'llm_api', llmApiKey: 'test' };
  const saves = [], errors = [];
  let fail = false;
  const c = vm.createContext({ document, File, atob, Uint8Array,
    COMPACT_TOOLBAR_ICONS: {}, escapeHtml: value => value,
    makeButton: text => Object.assign(new Element('button'), { textContent: text }),
    makeCheckbox: (text, checked) => {
      const wrapper = Object.assign(new Element('label'), { textContent: text });
      const input = Object.assign(new Element('input'), { checked, type: 'checkbox' });
      wrapper.append(input);
      return { wrapper, input };
    },
    normalizeProjectVideoEngine: value => value,
    readFileAsDataUrl: async file => `data:${file.type};base64,${Buffer.from(await file.arrayBuffer()).toString('base64')}`,
    toast: message => errors.push(message),
  });
  vm.runInContext(readBuilderModule('auto_build_draft.mjs'), c);
  vm.runInContext(readBuilderModule('auto_build.mjs'), c);
  const api = c.createAutoBuild({ state, projectInput: { value: folder }, audioInput: { value: '' },
    gemmaRunnerLabel: () => 'LLM API', llmApiVisionModelSelected: () => true,
    saveSession: async () => { if (fail) throw new Error('disk full'); saves.push(JSON.parse(JSON.stringify(state.autoBuildDraft))); return {}; },
  });
  return { c, api, state, saves, errors, document, fail: () => { fail = true; },
    nodes: tag => walk(document.body).filter(node => node.tag === tag),
    button: text => walk(document.body).find(node => node.textContent === text),
  };
}

test('closing saves all inputs and reopening a reloaded project restores file contents and options', async () => {
  const f = fixture(); f.api.openAutoBuildModal();
  const files = f.nodes('input').filter(node => node.type === 'file');
  for (let i = 0; i < files.length; i++) {
    files[i].files = [new File([`contents-${i}`], i ? `image-${i}.png` : 'song.wav', { type: i ? 'image/png' : 'audio/wav', lastModified: 42 })];
    files[i].onchange();
  }
  f.nodes('textarea')[0].value = '  My complete lyrics\nsecond line  ';
  f.nodes('select')[0].value = 'custom';
  f.nodes('input').find(node => node.type === 'number').value = '7.5';
  f.nodes('input').filter(node => node.type === 'checkbox').forEach(node => { node.checked = true; });
  f.nodes('details')[0].open = true;
  await f.button('Close').onclick();
  assert.equal(f.document.body.children.length, 0);
  const saved = f.saves[0];
  assert.equal(saved.lyrics, '  My complete lyrics\nsecond line  ');
  assert.equal(saved.use_current_settings, true);
  assert.equal(saved.skip_prompts, true);
  const reloaded = fixture('Z:/projects/one', JSON.parse(JSON.stringify(saved)));
  reloaded.api.openAutoBuildModal();
  assert.equal(reloaded.nodes('textarea')[0].value, saved.lyrics);
  assert.equal(reloaded.nodes('select')[0].value, 'custom');
  assert.equal(reloaded.nodes('input').find(node => node.type === 'number').value, '7.5');
  assert.ok(walk(reloaded.document.body).some(node => node.textContent === 'song.wav'));
  await reloaded.button('Cancel').onclick();
  assert.deepEqual(reloaded.saves[0], saved);
  const restored = f.c.restoreAutoBuildFile(saved.song);
  assert.equal(await restored.text(), 'contents-0');
  assert.equal(restored.name, 'song.wav');
  assert.equal(restored.lastModified, 42);
});

test('drafts stay within their project and an older project has no inherited inputs', () => {
  const f = fixture('Z:/projects/two', { project_folder: 'Z:/projects/one', lyrics: 'private to one' });
  f.api.openAutoBuildModal();
  assert.equal(f.nodes('textarea')[0].value, '');
  assert.equal(f.c.autoBuildDraftForProject({ project_folder: 'Z:\\projects\\one\\' }, 'z:/projects/one').project_folder, 'Z:\\projects\\one\\');
});

test('a failed save keeps the modal and its inputs available', async () => {
  const f = fixture(); f.api.openAutoBuildModal(); f.fail();
  f.nodes('textarea')[0].value = 'Do not lose this';
  await f.button('Close').onclick();
  assert.equal(f.document.body.children.length, 1);
  assert.equal(f.nodes('textarea')[0].value, 'Do not lose this');
  assert.match(f.errors[0], /disk full/);
  assert.equal(f.button('Close').disabled, false);
});

test('backdrop dismissal saves a draft and cannot dismiss a running build', async () => {
  const f = fixture(); f.api.openAutoBuildModal();
  const backdrop = f.document.body.children[0];
  f.button('Build My Music Video').disabled = true;
  backdrop.listeners.pointerdown({ target: backdrop });
  assert.equal(f.saves.length, 0);
  f.button('Build My Music Video').disabled = false;
  backdrop.listeners.pointerdown({ target: backdrop });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.saves.length, 1);
  assert.equal(f.document.body.children.length, 0);
});
