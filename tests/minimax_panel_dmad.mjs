import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// Builds the real MiniMax panel (minimax_panel_layout.mjs) in a small fake DOM, to check that the sampling method row,
// the DMAD note and the sigma shift controls are created and placed. ComfyUI's scripts/api.js is stubbed.
register('./stub_comfy_loader.mjs', import.meta.url);

class FakeElement {
  constructor(tagName) {
    this.tagName = String(tagName).toLowerCase();
    this.children = [];
    this.style = { cssText: '' };
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
    this.textContent = '';
    this.innerHTML = '';
    this.value = '';
    this.checked = false;
    this.disabled = false;
  }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.children.push(node); return node; }
  prepend(...nodes) { this.children.unshift(...nodes); }
  insertBefore(node, ref) {
    const index = this.children.indexOf(ref);
    if (index < 0) this.children.push(node); else this.children.splice(index, 0, node);
    return node;
  }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  remove() {}
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  removeEventListener() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  contains() { return false; }
  focus() {}
  click() {}
  get lastChild() { return this.children[this.children.length - 1] ?? null; }
  get firstChild() { return this.children[0] ?? null; }
  get firstElementChild() { return this.children.find((child) => child instanceof FakeElement) ?? null; }
}
const walk = (node, visit) => {
  if (!(node instanceof FakeElement)) return;
  visit(node);
  for (const child of node.children) walk(child, visit);
};

globalThis.document = {
  createElement: (tag) => new FakeElement(tag),
  createTextNode: (text) => ({ text, textContent: text, set data(value) { this.textContent = value; } }),
  createDocumentFragment: () => new FakeElement('fragment'),
  body: new FakeElement('body'),
  addEventListener() {},
  removeEventListener() {},
};
globalThis.window = {
  dispatchEvent() {}, addEventListener() {}, removeEventListener() {}, setTimeout, clearTimeout,
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  innerWidth: 1920, innerHeight: 1080,
};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node' }, configurable: true });

const { buildMiniMaxPanel } = await import('../web/music_video_builder/minimax_panel_layout.mjs');

const build = () => buildMiniMaxPanel({
  miniMaxGemmaModelSelect: new FakeElement('select'),
  miniMaxMmprojSelect: new FakeElement('select'),
  miniMaxReferencesButton: new FakeElement('button'),
  miniMaxSceneVideoButton: new FakeElement('button'),
  miniMaxTextGemmaModelSelect: new FakeElement('select'),
  miniMaxVideoReferencesButton: new FakeElement('button'),
  updateMiniMaxPromptSaveButtonState() {},
});

test('the panel puts the sampling method row right under the pipeline row', () => {
  const panel = build();
  const dmad = panel.miniMaxPassButtons.dmad;
  assert.ok(dmad, 'the DMAD controls ride on the pass buttons like the RefMod ones');
  const order = panel.miniMaxEnginePanel.children;
  const index = (node) => order.indexOf(node);
  assert.ok(index(dmad.chooser) > 0);
  assert.ok(index(dmad.note) === index(dmad.chooser) + 1);
  assert.ok(index(dmad.steps) === index(dmad.chooser) + 2);
  assert.ok(index(dmad.status) === index(dmad.chooser) + 3);
  assert.ok(index(panel.miniMaxPassChooser) > index(dmad.chooser), 'single / 2 pass come after the sampling method');
  assert.deepEqual(dmad.buttons.map((button) => button.dataset.minimaxH3Sampling), ['standard', 'dmad']);
  assert.deepEqual(dmad.buttons.map((button) => button.textContent), ['Standard sampling', 'DMAD 4-step']);
});

test('the sampling row comes after the pipeline row and before the pass buttons', () => {
  const panel = build();
  const order = panel.miniMaxEnginePanel.children;
  const pipelineRow = order.find((node) => node.attributes['aria-label'] === 'Video pipeline');
  const samplingRow = order.find((node) => node.attributes['aria-label'] === 'Sampling method');
  assert.ok(pipelineRow && samplingRow);
  assert.equal(order.indexOf(samplingRow), order.indexOf(pipelineRow) + 1);
  assert.ok(order.indexOf(panel.miniMaxPassChooser) > order.indexOf(samplingRow));
});

test('the sigma shift section has pass 1 and pass 2 controls with the node defaults', () => {
  const panel = build();
  const { sigmaControls, sigmaSection } = panel.miniMaxPassButtons.dmad;
  assert.equal(sigmaControls.length, 2);
  assert.deepEqual(sigmaControls.map((control) => control.pass), [1, 2]);
  for (const control of sigmaControls) {
    assert.equal(control.enabled.input.checked, false);
    assert.equal(control.video.value, '12');
    assert.equal(control.audio.value, '3');
    assert.equal(control.video.min, '0.01');
    assert.equal(control.video.max, '100');
    // The checkbox label text is the wrapper's last child, which the panel renames for Single / 2 Pass.
    control.label.textContent = 'Use sigma shift';
    assert.equal(control.enabled.wrapper.lastChild.textContent, 'Use sigma shift');
  }
  let insideAdvanced = false;
  walk(panel.miniMaxAdvancedSettings, (node) => { if (node === sigmaSection) insideAdvanced = true; });
  assert.ok(insideAdvanced, 'sigma shift sits in Advanced Settings next to the Sampler section');
});

test('the DMAD row has a steps box for pass 1 and pass 2, hidden until DMAD is on', () => {
  const panel = build();
  const { steps, stepsControls } = panel.miniMaxPassButtons.dmad;
  assert.match(steps.style.cssText, /display:none/);
  assert.deepEqual(stepsControls.map((control) => control.pass), [1, 2]);
  assert.deepEqual(stepsControls.map((control) => control.label.textContent), ['Pass 1 steps', 'Pass 2 steps']);
  for (const { input } of stepsControls) {
    assert.equal(input.type, 'number');
    assert.equal(input.min, '1');
    assert.equal(input.max, '1000');
    assert.equal(input.step, '1');
  }
});

test('the DMAD steps boxes and the Sampler / Pass 1 / Pass 2 step fields stay in step', async () => {
  const { wireMiniMaxDmadSteps } = await import('../web/music_video_builder/minimax_dmad_steps.mjs');
  const panel = build();
  const { stepsControls } = panel.miniMaxPassButtons.dmad;
  const state = { miniMaxH3TwoPassEnabled: false };
  const saved = [];
  let persisted = 0;
  wireMiniMaxDmadSteps({
    stepsControls, twoPassControls: panel.twoPassControls, singlePassSteps: panel.miniMaxSteps, state,
    save: () => saved.push([panel.miniMaxSteps.value, panel.twoPassControls[0].steps.value, panel.twoPassControls[1].steps.value]),
    persist: () => { persisted += 1; },
  });
  const fire = (node, type) => (node.listeners[type] || []).forEach((listener) => listener({ target: node }));
  const [box1, box2] = stepsControls.map((control) => control.input);

  // Single pass: pass 1 is the Sampler steps and pass 2 does nothing.
  box1.value = '8';
  fire(box1, 'input');
  assert.equal(panel.miniMaxSteps.value, '8');
  assert.deepEqual(saved.at(-1), ['8', panel.twoPassControls[0].steps.value, panel.twoPassControls[1].steps.value]);
  const before = saved.length;
  box2.value = '9';
  fire(box2, 'input');
  assert.equal(saved.length, before, 'pass 2 has nothing to edit in single pass');
  panel.miniMaxSteps.value = '10';
  fire(panel.miniMaxSteps, 'input');
  assert.equal(box1.value, '10');

  // 2 Pass: the boxes edit the Pass 1 and Pass 2 sections.
  state.miniMaxH3TwoPassEnabled = true;
  box1.value = '6';
  fire(box1, 'input');
  box2.value = '5';
  fire(box2, 'input');
  assert.equal(panel.twoPassControls[0].steps.value, '6');
  assert.equal(panel.twoPassControls[1].steps.value, '5');
  assert.equal(panel.miniMaxSteps.value, '10', 'the single pass steps are left alone in 2 Pass');
  panel.twoPassControls[1].steps.value = '7';
  fire(panel.twoPassControls[1].steps, 'input');
  assert.equal(box2.value, '7');
  panel.miniMaxSteps.value = '99';
  fire(panel.miniMaxSteps, 'input');
  assert.equal(box1.value, '6', 'the hidden single pass field does not move pass 1');

  fire(box1, 'change');
  assert.equal(persisted, 1);
});

test('lcm is a sampler choice for single pass and for both 2 Pass passes', () => {
  const panel = build();
  const optionValues = (select) => select.children.map((option) => option.value);
  assert.ok(optionValues(panel.miniMaxSamplerName).includes('lcm'));
  for (const control of panel.twoPassControls) assert.ok(optionValues(control.sampler).includes('lcm'));
  for (const control of panel.twoPassControls) assert.ok(optionValues(control.scheduler).includes('simple'));
  assert.ok(optionValues(panel.miniMaxScheduler).includes('simple'));
});
