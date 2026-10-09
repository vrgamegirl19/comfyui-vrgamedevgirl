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
    'makePickerField', 'makeSettingsSection', 'normalizeProjectVideoEngine']) {
    vm.runInContext(functionSource(readBuilderModule('controls.mjs'), name), c);
  }
  return c;
}

function parameters(source) {
  return Object.fromEntries(source.slice(source.indexOf('{') + 1, source.indexOf('})'))
    .split(',').map(name => name.trim()).filter(Boolean).map(name => [name, new Element()]));
}

test('timeline owns the global timing checkbox at the far right', () => {
  const c = fixture();
  vm.runInContext(functionSource(readBuilderModule('timeline_view.mjs'), 'buildTimelineView'), c);
  const layout = c.buildTimelineView({ overlay: new Element(), preview: new Element(), previewStage: new Element() });
  const wrapper = layout.freezeTimingControl.wrapper;
  assert.equal(wrapper.parentElement.children.at(-1), wrapper);
  assert.match(wrapper.style.cssText, /margin-left:auto/);
  assert.equal(wrapper.children.at(-1).textContent, 'Freeze SRT timing');
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

test('global timing change saves the shared state without a selected scene', async () => {
  const c = fixture();
  const source = functionSource(readBuilderModule('timeline_events.mjs'), 'wireTimelineControls');
  vm.runInContext(source, c);
  const input = new Element('input');
  const state = { timingFrozen: false };
  const saved = [];
  const history = [];
  c.wireTimelineControls({ ...parameters(source), state, freezeTimingControl: { input },
    snapToBeatsControl: { input: new Element('input') },
    pushHistory: () => history.push(state.timingFrozen), syncInspector() {}, render() {},
    autoSaveSessionQuiet: async () => saved.push(state.timingFrozen) });
  for (const checked of [true, false]) {
    input.checked = checked;
    input.events.change();
    await Promise.resolve();
    assert.equal(state.timingFrozen, checked);
  }
  assert.deepEqual(history, [false, true]);
  assert.deepEqual(saved, [true, false]);
});

test('one shared timing lock blocks every scene drag and unlock restores editing', () => {
  const c = fixture();
  const viewport = new Element();
  Object.assign(c, { state: { timingFrozen: true }, timelineViewport: viewport,
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
