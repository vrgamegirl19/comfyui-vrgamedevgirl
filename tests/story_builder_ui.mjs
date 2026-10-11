// Smoke test for the Story Builder window: opens it on a small fake DOM, renders every tab, and saves the scene plan.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';

// ComfyUI's own scripts are not available outside the browser.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (/scripts\/(app|api)\.js$/.test(specifier)) {
      return { url: 'data:text/javascript,export const app = {}; export const api = (globalThis.fakeComfyApi ||= {});', shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  // The Builder's .js files are ES modules too.
  load(url, context, nextLoad) {
    if (url.includes('/web/') && url.endsWith('.js')) return { ...nextLoad(url, { ...context, format: 'module' }), format: 'module' };
    return nextLoad(url, context);
  },
});

class FakeElement {
  constructor(tagName) {
    this.tagName = String(tagName).toUpperCase();
    this.children = [];
    this.style = { cssText: '' };
    this.dataset = {};
    this.listeners = {};
    this.value = '';
    this.parent = null;
    this.ownText = '';
  }
  append(...nodes) {
    for (const node of nodes) {
      const child = typeof node === 'string' ? Object.assign(new FakeElement('#text'), { ownText: node }) : node;
      child.parent = this;
      this.children.push(child);
    }
  }
  appendChild(node) { this.append(node); return node; }
  insertBefore(node) { this.append(node); return node; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this); }
  set textContent(value) { this.children = []; this.ownText = String(value); }
  get textContent() { return this.ownText + this.children.map((child) => child.textContent).join(''); }
  set innerHTML(value) { this.children = []; this.ownText = String(value).replace(/<[^>]+>/g, ' '); }
  get innerHTML() { return this.ownText; }
  get options() { return this.children.filter((child) => child.tagName === 'OPTION'); }
  get selectedOptions() { return this.options.filter((option) => option.value === this.value); }
  addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
  removeEventListener() {}
  setAttribute(name, value) { this[name] = value; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  focus() {}
  click() { this.onclick?.(); }
  all() { return [this, ...this.children.flatMap((child) => child.all())]; }
}

globalThis.HTMLElement = FakeElement;
globalThis.document = {
  body: new FakeElement('body'),
  head: new FakeElement('head'),
  createElement: (tag) => new FakeElement(tag),
  createTextNode: (text) => Object.assign(new FakeElement('#text'), { ownText: String(text) }),
  getElementById: () => null,
  addEventListener() {},
};
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.dispatchEvent = () => true;
globalThis.localStorage = { getItem: () => null, setItem() {} };

const { createStoryBuilder } = await import('../web/music_video_builder/story_builder.mjs');
const { normalizeBuilderStoryLayer } = await import('../web/music_video_builder/model_settings.mjs');

function project() {
  const segments = Array.from({ length: 6 }, (_, index) => ({
    id: `seg_${index + 1}`, start: index * 5, end: (index + 1) * 5, lyric_text: index === 5 ? '' : `line ${index + 1}`,
    timeline_note: index === 1 ? 'She runs to the porch.' : '', i2v_notes: '', notes: '',
  }));
  const layer = {
    story_flow_version: 2,
    story_book: { title: 'T', premise: 'P', chapters: [
      { id: 'ch_1', title: 'Home', setting_id: 'setting_home', paragraphs: ['Mara packs.'] },
      { id: 'ch_2', title: 'Away', setting_id: 'setting_train', paragraphs: ['Mara rides.'] },
    ] },
    story_book_text: 'Chapter 1: Home\nSetting: Home\n\nMara packs.',
    story_settings: [{ id: 'setting_home', name: 'Home', look: 'Pine.' }, { id: 'setting_train', name: 'Train', look: '' }],
    story_acts: [
      { id: 'act_1', chapter_id: 'ch_1', title: 'Pack', summary: 'Packing.', cast_ids: ['mara'], scene_ids: ['seg_1', 'seg_2', 'seg_3'] },
      { id: 'act_2', chapter_id: 'ch_2', title: 'Ride', summary: 'Riding.', cast_ids: ['mara'], scene_ids: ['seg_4', 'seg_5', 'seg_6'] },
    ],
    story_scene_plan: segments.map((segment, index) => ({
      scene_id: segment.id, act_id: index < 3 ? 'act_1' : 'act_2', story_slice: `slice ${index + 1}`, spot: index < 3 ? 'Kitchen' : 'Sleeper Car',
      mood: 'calm', performance_mode: index === 2 ? 'no_lip_sync' : 'singing', note_sources: [],
    })),
  };
  return {
    segments,
    overlaySegments: [],
    timelineMarkers: [{ id: 'm1', start: 6, end: 12, label: 'Story event', note: 'She finds the letter.' }],
    builderStoryLayer: layer,
    fluxReferenceBuilder: { subjects: [{ id: 'mara', name: 'Mara', reference_type: 'character', image: { path: 'C:/m.png' } }], locations: [], scene_map: {} },
  };
}

function setup() {
  const state = project();
  const calls = { saves: [], hosted: [], storyboard: 0, errors: [], order: [] };
  const builder = createStoryBuilder({
    autoSaveSessionQuiet: async (label) => { calls.saves.push(label); },
    createProgressWindow: () => ({ set() {}, close() {} }),
    getLlmPayload: () => ({ llm_runner: 'lm_studio' }),
    normalizeBuilderStoryLayer,
    openFluxReferenceBuilderModal: (options) => {
      calls.hosted.push(options.focusedSection);
      assert.ok(options.host instanceof FakeElement, 'the Reference Builder is hosted inside the tab');
      // Like the real one: Save writes back the copy of the cards taken when it opened.
      const copy = JSON.parse(JSON.stringify(state.fluxReferenceBuilder));
      return { save: async () => { calls.order.push('hosted save'); state.fluxReferenceBuilder = copy; }, close() {} };
    },
    openLyricMappingWorkflowModal() {},
    openLyricReviewModal() {},
    openStoryboardClassic: () => { calls.storyboard += 1; },
    pushHistory() {},
    referenceBuilderPayload: () => state.fluxReferenceBuilder,
    render() {},
    state,
    storyboardScenePayload: () => state.segments.map((segment) => ({ id: segment.id, lyrics: segment.lyric_text })),
  });
  return { state, calls, builder };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const findButton = (label) => document.body.all().find((node) => node.tagName === 'BUTTON' && node.textContent.trim() === label);

test('every tab renders without an error', async () => {
  const { calls, builder } = setup();
  const originalError = console.error;
  console.error = (...args) => calls.errors.push(args.join(' '));
  try {
    builder.openStoryBuilder();
    await tick();
    for (const tab of ['song', 'cast', 'story', 'acts', 'scenes', 'places', 'prompts']) {
      builder.openStoryBuilder({ tab });
      await tick();
      assert.ok(!document.body.textContent.includes('This tab failed to open'), `${tab} tab failed to render`);
    }
  } finally {
    console.error = originalError;
  }
  assert.deepEqual(calls.errors, []);
  assert.deepEqual(calls.hosted, ['subjects', 'locations']);
  findButton('Open Storyboard').onclick();
  assert.equal(calls.storyboard, 1);
  findButton('Close').onclick();
  await tick();
  assert.equal(document.body.children.length, 0, 'closing removes the window');
});

test('building locations keeps the new cards when the hosted Reference Builder saves its copy', async () => {
  const { state, calls, builder } = setup();
  globalThis.fakeComfyApi.fetchApi = async () => {
    calls.order.push('build');
    return { ok: true, json: async () => ({ ok: true, story_layer: {}, locations: [
      { name: 'Home - Kitchen', description: 'Pine kitchen.', story_setting_id: 'setting_home', story_act_ids: ['act_1'], scene_ids: ['seg_1', 'seg_2'] },
    ] }) };
  };
  try {
    builder.openStoryBuilder({ tab: 'places' });
    await tick();
    await findButton('Build Locations From Story').onclick();
    await tick();
    assert.deepEqual(calls.order.slice(0, 2), ['hosted save', 'build'], 'the hosted builder is saved before the build');
    assert.deepEqual(state.fluxReferenceBuilder.locations.map((card) => card.name), ['Home - Kitchen']);
    assert.equal(state.fluxReferenceBuilder.scene_map.seg_1, state.fluxReferenceBuilder.locations[0].id);
    findButton('Close').onclick();
    await tick();
    assert.deepEqual(state.fluxReferenceBuilder.locations.map((card) => card.name), ['Home - Kitchen'], 'closing keeps the cards');
  } finally {
    delete globalThis.fakeComfyApi.fetchApi;
  }
});

test('saving the scene plan sets each scene performance and keeps the notes', async () => {
  const { state, calls, builder } = setup();
  builder.openStoryBuilder({ tab: 'scenes' });
  await tick();
  await findButton('Save Scene Plan And Notes').onclick();
  assert.equal(state.segments[2].performance_mode, 'no_lip_sync');
  assert.equal(state.segments[2].lyric_no_lip_sync, true);
  assert.equal(state.segments[0].performance_mode, 'singing');
  assert.equal(state.segments[1].timeline_note, 'She runs to the porch.');
  assert.ok(calls.saves.includes('story builder scene plan edited'));
  findButton('Close').onclick();
  await tick();
});
