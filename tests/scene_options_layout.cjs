const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.style = { cssText: '' };
    this.dataset = {};
    this.events = {};
    this.value = '';
  }
  append(...children) {
    for (const child of children) {
      child.remove();
      child.parentElement = this;
      this.children.push(child);
    }
  }
  remove() {
    if (this.parentElement) {
      const parent = this.parentElement;
      parent.children.splice(parent.children.indexOf(this), 1);
      this.parentElement = null;
    }
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  removeEventListener(name) { delete this.events[name]; }
  setAttribute(name, value) { this[name] = value; }
  cloneNode() { return new Element(this.tagName); }
  get options() { return this.children; }
}

function fixture() {
  const document = { createElement: tag => new Element(tag),
    createTextNode: text => Object.assign(new Element('#text'), { textContent: text }),
    body: new Element('body') };
  const c = vm.createContext({ document, window: {}, BUILDER_FONT_STACK: 'sans-serif',
    normalizeSceneRenderWaitHours: value => Number(value || 2), MAX_SCENE_RENDER_WAIT_HOURS: 24,
    normalizeNotificationSettings: () => ({ mode: 'off', volume: 0.5 }),
    normalizeContinuityMode: value => value || 'off', normalizeAutoImg2ImgStartStep: () => 5,
    normalizeAutoImg2ImgCreativity: () => 0.5, TIMELINE_HEIGHT: 300,
    WAVEFORM_MODES: { medium: {} }, toast() {}, showMultiSelectHint() {},
    createTimelineToolWindows: () => ({ toolsButton: new Element('button'),
      deleteAllButton: new Element('button'), refreshDeleteActions() {} }) });
  for (const name of ['makeButton', 'makeInput', 'makeCheckbox', 'makeSelect', 'makeField',
    'makePickerField', 'makeSettingsSection', 'normalizeProjectVideoEngine', 'normalizeVideoType']) {
    vm.runInContext(functionSource(readBuilderModule('controls.mjs'), name), c);
  }
  vm.runInContext(readBuilderModule('elevenlabs_voice_design.mjs'), c);
vm.runInContext(readBuilderModule('elevenlabs.mjs'), c);
  return c;
}

function parameters(source) {
  return Object.fromEntries(source.slice(source.indexOf('{') + 1, source.indexOf('})'))
    .split(',').map(name => name.trim()).filter(Boolean).map(name => [name, new Element()]));
}

test('timeline has a single Scenes lock with no Freeze SRT timing checkbox', () => {
  const c = fixture();
  vm.runInContext(functionSource(readBuilderModule('timeline_view.mjs'), 'buildTimelineView'), c);
  const layout = c.buildTimelineView({ overlay: new Element(), preview: new Element(), previewStage: new Element() });
  assert.equal(layout.freezeTimingControl, undefined);
  // The Export button sits right after the lock, last in the header.
  assert.equal(layout.sceneDragLockButton.parentElement.children.at(-2), layout.sceneDragLockButton);
  assert.equal(layout.sceneDragLockButton.parentElement.children.at(-1), layout.exportVideoButton);
  assert.equal(layout.exportVideoButton.textContent, 'Export');
  assert.equal(layout.exportVideoButton.disabled, true);
  const texts = node => [node.textContent, ...node.children.flatMap(texts)];
  assert.ok(!texts(layout.sceneDragLockButton.parentElement).includes('Freeze SRT timing'));
});

test('inspector has three tabs and migrates old Scene selection to Image', () => {
  const c = fixture();
  const module = readBuilderModule('inspector.mjs');
  vm.runInContext(functionSource(module, 'buildInspectorTabs'), c);
  const layout = c.buildInspectorTabs();
  assert.deepEqual(Array.from(layout.inspectorTabs.children, button => button.textContent), ['Image', 'Video', 'Audio']);
  assert.equal(layout.scenePanel.parentElement, undefined);
  const source = functionSource(module, 'createInspector');
  vm.runInContext(source, c);
  const state = {};
  const inspector = c.createInspector({ ...parameters(source), ...layout, state,
    activeSegment: () => ({ id: 'scene1' }), applyLayoutSizes() {} });
  inspector.setInspectorTab('scene');
  assert.equal(state.inspectorTab, 'image');
  assert.equal(layout.imagePanel.style.display, 'flex');
  assert.equal(layout.videoPanel.style.display, 'none');
  // Changing right-panel tabs must not hide Scene Options inside Settings.
  inspector.setInspectorTab('video');
  assert.equal(layout.scenePanel.style.display, undefined);
});

test('Scene Options reuses the scene controls and survives closing and reopening Settings', () => {
  const c = fixture();
  const source = functionSource(readBuilderModule('project_setup.mjs'), 'createProjectSetup');
  vm.runInContext(source, c);
  const scenePanel = new Element();
  const input = new Element('input');
  input.value = 'My scene';
  let edits = 0;
  input.addEventListener('input', () => edits++);
  scenePanel.append(input);
  const state = {};
  let syncs = 0;
  const setup = c.createProjectSetup({ ...parameters(source), state, scenePanel,
    settingsModalControls: {}, syncInspector: () => syncs++, getPreferredProjectRoot: () => '',
    autoSaveSessionQuiet: async () => {} });
  for (let i = 0; i < 2; i++) {
    setup.openSettingsModal();
    const backdrop = c.document.body.children.at(-1);
    const box = backdrop.children[0];
    const section = box.children.find(child => child.children[0]?.textContent === 'Scene Options');
    assert.ok(section);
    assert.equal(section.open, false);
    assert.equal(section.children[1].children.at(-1), scenePanel);
    assert.equal(input.value, 'My scene');
    input.events.input();
    box.children[0].children[1].onclick();
    assert.equal(c.document.body.children.length, 0);
  }
  assert.equal(edits, 2);
  assert.equal(syncs, 2);
});

for (const dragLocked of [true, false]) {
  test(`Scenes Unlocked releases a saved timing freeze when drag lock is ${dragLocked}`, async () => {
    const c = fixture();
    const module = readBuilderModule('timeline_view.mjs');
    c.SCENE_DRAG_LOCK_KEY = 'scenes-locked';
    const stored = [];
    c.localStorage = { setItem: (key, value) => stored.push(value) };
    for (const name of ['syncSceneDragLockButton', 'wireSceneDragLock']) vm.runInContext(functionSource(module, name), c);
    const button = new Element('button');
    const state = { timingFrozen: true, timelineDragLocked: dragLocked };
    const history = [], saved = [];
    let syncs = 0;
    c.wireSceneDragLock({ button, state,
      pushHistory: () => history.push(state.timingFrozen), syncInspector: () => syncs++, render() {},
      autoSaveSessionQuiet: async () => saved.push(state.timingFrozen) });
    assert.equal(button['aria-pressed'], 'true');
    await button.onclick();
    assert.equal(state.timingFrozen, false);
    assert.equal(state.timelineDragLocked, false);
    assert.equal(button['aria-pressed'], 'false');
    assert.deepEqual(history, [true]);
    assert.deepEqual(saved, [false]);
    assert.equal(syncs, 1);
    assert.deepEqual(stored, ['0']);
    await button.onclick();
    assert.equal(state.timelineDragLocked, true);
    assert.equal(state.timingFrozen, false, 'ordinary scene lock keeps its original drag-only behavior');
    assert.deepEqual(stored, ['0', '1']);
    assert.equal(saved.length, 1);
    // Project reload or undo can restore a timing freeze after the button was unlocked.
    state.timelineDragLocked = false;
    state.timingFrozen = true;
    c.syncSceneDragLockButton(button, state);
    assert.equal(button['aria-pressed'], 'true');
  });
}

test('scene lock still works when browser storage is unavailable', async () => {
  const c = fixture();
  c.SCENE_DRAG_LOCK_KEY = 'scenes-locked';
  c.localStorage = { setItem() { throw new Error('Storage blocked'); } };
  const module = readBuilderModule('timeline_view.mjs');
  for (const name of ['syncSceneDragLockButton', 'wireSceneDragLock']) vm.runInContext(functionSource(module, name), c);
  const button = new Element('button'), state = {};
  c.wireSceneDragLock({ button, state, pushHistory() {}, syncInspector() {}, render() {}, autoSaveSessionQuiet: async () => {} });
  await button.onclick();
  assert.equal(state.timelineDragLocked, false);
  await button.onclick();
  assert.equal(state.timelineDragLocked, true);
});

test('one shared timing lock blocks every scene drag and unlock restores editing', () => {
  const c = fixture();
  const viewport = new Element();
  Object.assign(c, { state: { timingFrozen: true, timelineDragLocked: false }, timelineViewport: viewport,
    segmentTrack: () => 'base', hasLockedVideo: () => false,
    window: { addEventListener() {}, removeEventListener() {} } });
  vm.runInContext('let activeSegmentDragCleanup = null;\n' +
    functionSource(readBuilderModule('media_import.mjs'), 'makeDragHandle'), c);
  for (const id of ['scene1', 'scene2']) {
    const segment = { id, start: 0, end: 10 };
    const handle = new Element();
    let editing = 0;
    const event = { button: 0, isPrimary: true, pointerId: 1, clientX: 100,
      preventDefault: () => editing++, stopPropagation() {} };
    c.makeDragHandle(handle, segment, 'end');
    c.state.timingFrozen = true;
    handle.events.pointerdown(event);
    assert.equal(editing, 0);
    assert.equal(segment.end, 10);
    c.state.timingFrozen = false;
    handle.events.pointerdown(event);
    assert.equal(editing, 1);
  }
});

test('scenes locked in place refuse a drag, say how to unlock once, and drag again when unlocked', () => {
  const c = fixture();
  const listeners = {};
  const toasts = [];
  Object.assign(c, { state: { timingFrozen: false }, timelineViewport: new Element(),
    segmentTrack: () => 'base', hasLockedVideo: () => false, toast: (message) => toasts.push(message),
    window: { addEventListener: (name, handler) => { listeners[name] = handler; }, removeEventListener: (name) => { delete listeners[name]; } } });
  vm.runInContext('let activeSegmentDragCleanup = null;\n' +
    functionSource(readBuilderModule('media_import.mjs'), 'makeDragHandle'), c);
  const segment = { id: 'scene1', start: 0, end: 10 };
  const handle = new Element();
  let editing = 0;
  const event = { button: 0, isPrimary: true, pointerId: 1, clientX: 100, preventDefault: () => editing++, stopPropagation() {} };
  c.makeDragHandle(handle, segment, 'move');
  // Locked by default (no saved choice): a press does not start a drag, and a click needs no note.
  handle.events.pointerdown(event);
  assert.equal(editing, 0);
  listeners.pointermove({ clientX: 102 });
  assert.equal(toasts.length, 0, 'a small wobble during a click says nothing');
  listeners.pointermove({ clientX: 140 });
  assert.equal(toasts.length, 1);
  assert.match(toasts[0], /Scenes Locked/);
  assert.equal(listeners.pointermove, undefined, 'the note is shown once per drag');
  c.state.timelineDragLocked = false;
  handle.events.pointerdown(event);
  assert.equal(editing, 1, 'unlocked scenes drag as before');
});
