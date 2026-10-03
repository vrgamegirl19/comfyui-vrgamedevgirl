const { functionSource, readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const source = readBuilderSource();

// ---- the dialog -------------------------------------------------------------------------------------------
function fakeDom() {
  const documentListeners = {};
  const makeNode = (tag) => ({
    tag, style: {}, dataset: {}, attrs: {}, children: [], listeners: {}, textContent: '', checked: false, removed: false, focused: false,
    append(...kids) { this.children.push(...kids); },
    setAttribute(name, value) { this.attrs[name] = value; },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    remove() { this.removed = true; },
    focus() { this.focused = true; },
  });
  const document = {
    body: makeNode('body'),
    createElement: makeNode,
    createTextNode: (text) => ({ text }),
    addEventListener: (type, fn) => { documentListeners[type] = fn; },
    removeEventListener: (type) => { delete documentListeners[type]; },
  };
  return { document, documentListeners };
}

function openDialog(options) {
  const { document, documentListeners } = fakeDom();
  const inputs = [];
  const context = vm.createContext({
    document,
    makeButton: (label) => ({ ...document.createElement('button'), label }),
    makeCheckbox: (label, checked) => {
      const input = document.createElement('input');
      input.checked = Boolean(checked);
      inputs.push(input);
      return { wrapper: document.createElement('label'), input, label };
    },
  });
  vm.runInContext(`${readBuilderModule('confirm_dialog.mjs')};globalThis.confirmDestructiveAction = confirmDestructiveAction;`, context);
  const result = context.confirmDestructiveAction(options);
  const backdrop = document.body.children[0];
  const box = backdrop.children[0];
  const actions = box.children[box.children.length - 1];
  const [cancel, confirm] = actions.children;
  return { result, backdrop, box, cancel, confirm, document, documentListeners, inputs };
}

test('the dialog shows the title, message and details and waits for OK or Cancel', async () => {
  const d = openDialog({ title: 'Delete this scene?', message: ['You are about to delete a scene.', 'Second line.'], details: ['Scene 3'] });
  assert.equal(d.box.children[0].textContent, 'Delete this scene?');
  assert.equal(d.box.children[1].textContent, 'You are about to delete a scene.');
  assert.equal(d.box.children[2].textContent, 'Second line.');
  assert.equal(d.box.children[3].children[0].textContent, 'Scene 3');
  assert.equal(d.cancel.label, 'Cancel');
  assert.equal(d.confirm.label, 'OK');
  assert.equal(d.backdrop.attrs.role, 'alertdialog');
  assert.equal(d.cancel.focused, true, 'Cancel has focus so Enter cannot delete by accident');
  assert.equal(d.document.body.children.length, 1);
  d.confirm.onclick();
  const answer = await d.result;
  assert.equal(answer.confirmed, true);
  assert.equal(d.backdrop.removed, true);
});

test('Cancel, Escape and clicking outside the box all cancel', async () => {
  const cancelled = openDialog({ title: 'x' });
  cancelled.cancel.onclick();
  assert.equal((await cancelled.result).confirmed, false);

  const escaped = openDialog({ title: 'x' });
  let prevented = false, stopped = false;
  escaped.documentListeners.keydown({ key: 'Escape', preventDefault() { prevented = true; }, stopPropagation() { stopped = true; } });
  assert.equal((await escaped.result).confirmed, false);
  assert.equal(prevented && stopped, true);
  assert.equal(escaped.documentListeners.keydown, undefined, 'the key handler is removed');

  const outside = openDialog({ title: 'x' });
  outside.backdrop.onclick({ target: outside.box });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(outside.backdrop.removed, false, 'clicking inside the box does not close it');
  outside.backdrop.onclick({ target: outside.backdrop });
  assert.equal((await outside.result).confirmed, false);

  const other = openDialog({ title: 'x' });
  other.documentListeners.keydown({ key: 'Enter', preventDefault() { throw Error('Enter must not act'); }, stopPropagation() {} });
  other.cancel.onclick();
  assert.equal((await other.result).confirmed, false);
});

test('the optional checkbox is reported and defaults to unchecked', async () => {
  const defaulted = openDialog({ title: 'x', option: { label: 'Also delete images' } });
  assert.ok(defaulted.box.children.some((child) => child.tag === 'label'), 'the checkbox is shown');
  assert.equal(defaulted.inputs[0].checked, false);
  defaulted.confirm.onclick();
  assert.deepEqual({ ...(await defaulted.result) }, { confirmed: true, optionChecked: false });

  const ticked = openDialog({ title: 'x', option: { label: 'Also delete images' } });
  ticked.inputs[0].checked = true;
  ticked.confirm.onclick();
  assert.deepEqual({ ...(await ticked.result) }, { confirmed: true, optionChecked: true });

  const cancelled = openDialog({ title: 'x', option: { label: 'Also delete images', checked: true } });
  cancelled.cancel.onclick();
  assert.equal((await cancelled.result).confirmed, false);

  const noOption = openDialog({ title: 'x' });
  assert.equal(noOption.inputs.length, 0);
  noOption.confirm.onclick();
  assert.equal((await noOption.result).optionChecked, false);
});

// ---- every delete asks first ---------------------------------------------------------------------------------
const FIRST_CHANGE = ['pushHistory(', 'postJson(', 'state.segments =', 'state.overlaySegments ='];
for (const [name, ask] of [
  ['deleteSegment', 'confirmDestructiveAction('],
  ['deleteAllSegments', 'confirmDestructiveAction('],
  ['deleteSelectedMedia', 'confirmDeleteMediaAction('],
  ['deleteAllTimelineVideos', 'confirmDestructiveAction('],
  ['deleteAllTimelineImages', 'confirmDestructiveAction('],
]) {
  test(`${name} asks for confirmation before changing anything and uses no native confirm`, () => {
    const body = functionSource(source, name);
    const askAt = body.indexOf(ask);
    assert.ok(askAt >= 0, `${name} must call ${ask}`);
    assert.equal(body.includes('window.confirm('), false);
    // Cancel must stop the action: a return guard has to follow the confirmation.
    const guard = body.slice(askAt).match(/if \(!(?:\w+\.)?confirmed\) return|if \(!ok\) return/);
    assert.ok(guard, `${name}: Cancel must return before anything is deleted`);
    for (const change of FIRST_CHANGE) {
      const changeAt = body.indexOf(change);
      if (changeAt >= 0) assert.ok(askAt < changeAt, `${name}: the confirmation must come before ${change}`);
    }
  });
}

test('the shared media dialog also uses the OK / Cancel dialog', () => {
  const body = functionSource(source, 'confirmDeleteMediaAction');
  assert.ok(body.includes('confirmDestructiveAction('));
  assert.equal(body.includes('document.createElement'), false);
});

// ---- behaviour of the scene delete ------------------------------------------------------------------------------
function deleteSegmentContext(answer) {
  const calls = [];
  const segment = { id: 'a', label: 'Verse', start: 1, end: 5, image_history: ['i.png'] };
  const c = vm.createContext({
    calls, segment, state: { segments: [segment], overlaySegments: [], undoStack: [1], redoStack: [1], projectFolder: '/p', activeId: 'a', activeTrack: 'base', sceneAudioGlobalTime: 0 },
    projectInput: { value: '/p' }, activeSegment: () => segment, segmentTrack: () => 'base', sceneSlotNumber: () => 1,
    selectedSegmentVideoPath: () => '', formatTime: (value) => `${value}s`,
    confirmDestructiveAction: async (options) => { calls.push(['confirm', options]); return answer; },
    postJson: async (url) => { calls.push(['post', url]); return { renamed: [] }; },
    pushHistory: () => calls.push(['pushHistory']), toast() {}, rewriteRenamedScenePaths() {}, updateHistoryButtons() {},
    closeBaseTimelineGap: () => 0, renumberGenericBaseSceneLabels() {}, currentGlobalTime: () => 0, syncInspector() {}, render() {},
    loadDirtyLatentBadges() {}, syncPromptJsonFromSegments: async () => {}, syncI2VMotionJsonFromSegments: async () => {},
    autoSaveSessionQuiet: async () => { calls.push(['save']); },
  });
  vm.runInContext(functionSource(source, 'deleteSegment'), c);
  return c;
}

test('cancelling the scene delete leaves the timeline and the project files untouched', async () => {
  const c = deleteSegmentContext({ confirmed: false });
  await c.deleteSegment();
  assert.deepEqual(c.calls.map((call) => call[0]), ['confirm']);
  assert.equal(c.state.segments.length, 1);
  assert.equal(c.state.undoStack.length, 1);
});

test('confirming the scene delete removes it, and the dialog names the scene', async () => {
  const c = deleteSegmentContext({ confirmed: true });
  await c.deleteSegment();
  const [, options] = c.calls.find((call) => call[0] === 'confirm');
  assert.equal(options.title, 'Delete this scene?');
  assert.match(options.details.join(' '), /Verse/);
  assert.match(options.message.join(' '), /removed_scene_assets/);
  assert.ok(options.details.includes('This scene has generated media.'));
  assert.equal(c.state.segments.length, 0);
  assert.ok(c.calls.some((call) => call[0] === 'post' && /renumber_scenes_after_removal/.test(call[1])));
});

test('cancelling delete-all-segments keeps every segment and confirming removes them', async () => {
  for (const confirmed of [false, true]) {
    const calls = [];
    const c = vm.createContext({
      calls, state: { segments: [{ id: 'a' }, { id: 'b' }], overlaySegments: [{ id: 'c' }], selectedSegmentIds: [] },
      toast() {}, pauseTimelineForEditing() {}, pushHistory: () => calls.push('pushHistory'), sceneAudio: { removeAttribute() {}, load() {} },
      freezeTimingControl: { input: {} }, segmentLayer: {}, sceneListPane: {}, syncTimelineTrimModeButton() {}, syncInspector() {}, render() {},
      syncPromptJsonFromSegments: async () => {}, syncI2VMotionJsonFromSegments: async () => {}, autoSaveSessionQuiet: async () => {},
      confirmDestructiveAction: async (options) => { calls.push(options.title); return { confirmed }; },
    });
    vm.runInContext(functionSource(source, 'deleteAllSegments'), c);
    await c.deleteAllSegments();
    assert.equal(calls[0], 'Delete ALL 3 segments?');
    assert.equal(c.state.segments.length, confirmed ? 0 : 2);
    assert.equal(calls.includes('pushHistory'), confirmed);
  }
});
