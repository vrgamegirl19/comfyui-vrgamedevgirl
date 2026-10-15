const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const source = readBuilderModule('project_files.mjs');
const json = (...prompts) => JSON.stringify({ scenes: prompts.map(prompt => ({ prompt })) });

function fixture(choose) {
  const context = vm.createContext({
    numericPromptKey: key => Number(String(key).match(/\d+/)?.[0] || 0),
    showMiniMaxPromptSourceChoice: choose || (() => { throw Error('Unexpected source chooser'); }),
  });
  for (const name of ['parsePromptTextBlocks', 'miniMaxPromptEditorSources', 'chooseMiniMaxPromptEditorSource']) {
    vm.runInContext(functionSource(source, name), context);
  }
  return context;
}

test('populated session prompts beat an existing empty export', async () => {
  const c = fixture();
  const current = json('Story prompt', '', 'Another scene');
  const selected = await c.chooseMiniMaxPromptEditorSource(current, json('', '', ''));
  assert.equal(selected.id, 'scenes');
  assert.equal(selected.content, current);
  assert.equal(selected.count, 2);
  assert.equal(JSON.parse(selected.content).scenes.length, 3, 'Preserve blank scene positions');
});

test('a populated export is selected when session prompts are blank', async () => {
  const c = fixture();
  const exported = json('Saved-file prompt');
  const selected = await c.chooseMiniMaxPromptEditorSource(json(''), exported);
  assert.equal(selected.id, 'file');
  assert.equal(selected.content, exported);
});

test('a missing export uses session prompts without a question', async () => {
  assert.equal((await fixture().chooseMiniMaxPromptEditorSource(json('Current'), '')).id, 'scenes');
});

test('both populated sources ask and honor either selection', async () => {
  for (const selectedId of ['scenes', 'file']) {
    let asked = 0;
    const c = fixture(sources => {
      asked++;
      assert.equal(sources.length, 2);
      assert.equal(sources[0].count, 1);
      assert.equal(sources[1].count, 2);
      return sources.find(item => item.id === selectedId);
    });
    const selected = await c.chooseMiniMaxPromptEditorSource(json('Story'), json('File 1', 'File 2'));
    assert.equal(asked, 1);
    assert.equal(selected.id, selectedId);
  }
});

test('identical populated sources still offer the requested choice', async () => {
  let asked = false;
  const c = fixture(sources => { asked = true; return sources[0]; });
  await c.chooseMiniMaxPromptEditorSource(json('Same'), json('Same'));
  assert.equal(asked, true);
});

test('cancellation returns no selection and neither source is altered', async () => {
  const current = json('Keep current');
  const exported = json('Keep file');
  assert.equal(await fixture(() => null).chooseMiniMaxPromptEditorSource(current, exported), null);
  assert.equal(JSON.parse(current).scenes[0].prompt, 'Keep current');
  assert.equal(JSON.parse(exported).scenes[0].prompt, 'Keep file');
});

test('no prompts produces no source; whitespace and JSON metadata do not count', async () => {
  assert.equal(await fixture().chooseMiniMaxPromptEditorSource(json('  '), JSON.stringify({ scenes: [] })), null);
  assert.equal(await fixture().chooseMiniMaxPromptEditorSource(json(''), JSON.stringify({ version: 1, scene_count: 3, type: 'storyboard_video_prompts' })), null);
});

test('malformed JSON export cannot hide valid session prompts', async () => {
  assert.equal((await fixture().chooseMiniMaxPromptEditorSource(json('Current'), '{"scenes":')).id, 'scenes');
});

test('pass-two-only exports are usable prompt sources', async () => {
  const exported = JSON.stringify({ scenes: [{ prompt: '', pass2_prompt: 'Second pass' }] });
  const selected = await fixture().chooseMiniMaxPromptEditorSource(json(''), exported);
  assert.equal(selected.id, 'file');
  assert.equal(selected.count, 1);
});

test('source chooser supports choosing, canceling, Escape and backdrop dismissal', async () => {
  for (const action of ['scenes', 'file', 'cancel', 'escape', 'backdrop']) {
    const listeners = new Map();
    const element = () => ({
      children: [], style: {}, removed: false,
      append(...children) { this.children.push(...children); },
      remove() { this.removed = true; }, focus() {},
    });
    const document = {
      body: element(), createElement: element,
      addEventListener(name, fn) { listeners.set(name, fn); },
      removeEventListener(name) { listeners.delete(name); },
    };
    const c = vm.createContext({ document, makeButton: text => ({ ...element(), textContent: text }) });
    vm.runInContext(functionSource(source, 'showMiniMaxPromptSourceChoice'), c);
    const sources = [{ id: 'scenes', label: 'Current scenes', count: 2 }, { id: 'file', label: 'Saved file', count: 1 }];
    const result = c.showMiniMaxPromptSourceChoice(sources);
    const backdrop = document.body.children[0];
    const buttons = backdrop.children[0].children[2].children;
    if (action === 'escape') listeners.get('keydown')({ key: 'Escape', preventDefault() {}, stopPropagation() {} });
    else if (action === 'backdrop') backdrop.onclick({ target: backdrop });
    else buttons[action === 'scenes' ? 0 : action === 'file' ? 1 : 2].onclick();
    const selected = await result;
    assert.equal(selected?.id || null, ['scenes', 'file'].includes(action) ? action : null);
    assert.equal(backdrop.removed, true);
    assert.equal(listeners.size, 0);
  }
});

test('opening the editor leaves prompts intact; Save applies the chosen source and updates its export', async () => {
  for (const selectedId of ['scenes', 'file']) {
    const c = fixture(sources => sources.find(item => item.id === selectedId));
    const segments = [{ id: 's1', minimax_h3_prompt: 'Current 1' }, { id: 's2', minimax_h3_prompt: 'Current 2' }];
    const current = json('Current 1', 'Current 2');
    const exported = json('File 1', 'File 2');
    const elements = [];
    const element = tag => ({
      tag, children: [], style: {},
      append(...children) { this.children.push(...children); }, remove() {}, focus() {},
    });
    const writes = [];
    let sessionSaves = 0;
    Object.assign(c, {
      document: { body: element('body'), createElement: tag => { const e = element(tag); elements.push(e); return e; } },
      makeButton: text => ({ ...element('button'), textContent: text }),
      showPromptFormatHint: async () => true,
      promptKindFile: () => 'project/prompts/video_prompts.json',
      formatPromptBlocks: () => current,
      loadPromptTextFile: async () => exported,
      ensureOriginalPromptBackup: async () => {},
      backupCurrentPromptState: async () => {},
      allEditableSegments: () => segments,
      pushHistory() {}, syncInspector() {}, render() {},
      sceneDisplayName: (_segment, index) => `Scene ${index + 1}`,
      savePromptTextFile: async (path, content) => { writes.push({ path, content }); return { path }; },
      saveSession: async () => { sessionSaves++; },
      promptKindLabel: () => 'MiniMax H3',
      toast: message => { if (/Error|failed|undefined/i.test(message)) throw Error(message); },
    });
    for (const name of ['setSegmentPromptForEdit', 'applyPromptBlocksToSegments', 'saveMiniMaxPromptExport', 'editFinalPromptList']) {
      vm.runInContext(functionSource(source, name), c);
    }
    await c.editFinalPromptList('minimax');
    assert.equal(segments[0].minimax_h3_prompt, 'Current 1');
    assert.equal(sessionSaves, 0);
    assert.equal(writes.length, 0);
    const textarea = elements.find(e => e.tag === 'textarea');
    assert.equal(textarea.value, selectedId === 'scenes' ? current : exported);
    const editor = c.document.body.children[0];
    await editor.children[3].children[1].onclick();
    assert.equal(segments[0].minimax_h3_prompt, selectedId === 'scenes' ? 'Current 1' : 'File 1');
    assert.equal(sessionSaves, 1);
    assert.equal(writes.length, 1);
    assert.equal(JSON.parse(writes[0].content).scenes[0].prompt, segments[0].minimax_h3_prompt);
  }
});

test('canceling source selection never creates an editor or backup', async () => {
  const c = fixture(() => null);
  Object.assign(c, {
    showPromptFormatHint: async () => true,
    promptKindFile: () => 'project/prompts/video_prompts.json',
    formatPromptBlocks: () => json('Current'),
    loadPromptTextFile: async () => json('File'),
    ensureOriginalPromptBackup: () => { throw Error('Must not write a backup on cancellation'); },
    document: { createElement: () => { throw Error('Must not open an editor on cancellation'); } },
    toast: message => { throw Error(message); },
  });
  vm.runInContext(functionSource(source, 'editFinalPromptList'), c);
  await c.editFinalPromptList('minimax');
});
