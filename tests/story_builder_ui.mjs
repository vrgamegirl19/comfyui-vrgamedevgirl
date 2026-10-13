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

function setup(extra = {}) {
  const state = project();
  const calls = { saves: [], hosted: [], storyboard: 0, errors: [], order: [], beats: [], prompts: 0, promptArgs: [], generated: [], defaults: [], progress: [] };
  const builder = createStoryBuilder({
    autoSaveSessionQuiet: async (label) => { calls.saves.push(label); },
    createProgressWindow: () => ({ set(message) { calls.progress.push(message); }, close() {} }),
    createSceneBeats: async (options) => { calls.beats.push(options); state.segments[0].story_beat = 'Mara folds a shirt.'; },
    createVideoPrompts: async (options) => { calls.prompts += 1; calls.promptArgs.push(options); },
    getLlmPayload: () => ({ llm_runner: 'lm_studio' }),
    normalizeBuilderStoryLayer,
    openFluxReferenceBuilderModal: (options) => {
      calls.hosted.push(options.focusedSection);
      assert.ok(options.host instanceof FakeElement, 'the Reference Builder is hosted inside the tab');
      // Like the real one: Save writes back the copy of the cards taken when it opened.
      const copy = JSON.parse(JSON.stringify(state.fluxReferenceBuilder));
      return { save: async () => { calls.order.push('hosted save'); state.fluxReferenceBuilder = copy; }, close() {},
        generateMissingSubjectImages: () => calls.generated.push('characters'),
        generateMissingLocationImages: () => calls.generated.push('locations') };
    },
    openLyricMappingWorkflowModal() {},
    openLyricReviewModal() {},
    // Like the Storyboard loader: the focused Scene Defaults view arrives asynchronously.
    openSceneDefaults: async (options) => {
      assert.ok(options.host instanceof FakeElement, 'Scene Defaults is hosted inside the Song tab');
      calls.defaults.push('open');
      return { save: async () => { calls.defaults.push('save'); }, close: () => { calls.defaults.push('close'); } };
    },
    openStoryboardClassic: () => { calls.storyboard += 1; },
    pushHistory() {},
    referenceBuilderPayload: () => state.fluxReferenceBuilder,
    render() {},
    state,
    storyboardScenePayload: () => state.segments.map((segment) => ({ id: segment.id, lyrics: segment.lyric_text })),
    ...extra,
  });
  return { state, calls, builder };
}

test('each tab offers Story Mode\'s own instructions, and the steps send the project folder', async () => {
  const opened = [];
  const bodies = [];
  const { state, builder } = setup({ projectFolder: () => 'C:/projects/song', openInstructions: (key, segment) => opened.push([key, segment.id]) });
  state.projectVideoEngine = 'minimax_h3';
  state.miniMaxH3Settings = { video_mode: 'reference_to_video' };
  globalThis.fakeComfyApi.fetchApi = async (url, options) => {
    bodies.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ ok: true, story_layer: {} }) };
  };
  try {
    builder.openStoryBuilder({ tab: 'acts' });
    await tick();
    findButton('Edit Acts Instructions').onclick();
    builder.openStoryBuilder({ tab: 'prompts' });
    await tick();
    findButton('Edit Scene Beat Instructions').onclick();
    findButton('Edit MiniMax Prompt Instructions').onclick();
    assert.deepEqual(opened, [['story_mode_acts', 'seg_1'], ['story_mode_scene_beat', 'seg_1'],
      ['minimax_h3_story_mode_reference_to_video', 'seg_1']]);
    builder.openStoryBuilder({ tab: 'acts' });
    await tick();
    await findButton('Rebuild All Acts').onclick();
    await tick();
    assert.equal(bodies.at(-1)?.project_folder, 'C:/projects/song');
  } finally {
    delete globalThis.fakeComfyApi.fetchApi;
  }
  findButton('Close').onclick();
  await tick();
});

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
    assert.deepEqual(calls.generated, [], 'building locations never opens the image generator');
    findButton('Close').onclick();
    await tick();
    assert.deepEqual(state.fluxReferenceBuilder.locations.map((card) => card.name), ['Home - Kitchen'], 'closing keeps the cards');
  } finally {
    delete globalThis.fakeComfyApi.fetchApi;
  }
});

test('beats and prompts tab creates beats and prompts and saves edits', async () => {
  const { state, calls, builder } = setup();
  state.projectVideoEngine = 'minimax_h3';
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  assert.equal(findButton('Create Video Prompts').disabled, true, 'prompts need a beat first');
  await findButton('Create Missing Beats').onclick();
  await tick();
  assert.deepEqual(calls.beats, [{ overwrite: false }]);
  assert.equal(findButton('Create Video Prompts').disabled, false);
  await findButton('Create Video Prompts').onclick();
  await tick();
  assert.equal(calls.prompts, 1);
  const boxes = document.body.all().filter((node) => node.tagName === 'TEXTAREA');
  assert.equal(boxes.length, state.segments.length * 2, 'a beat and a prompt box per scene');
  assert.equal(boxes[0].value, 'Mara folds a shirt.');
  boxes[1].value = 'Close-up, Mara folds a shirt.';
  await findButton('Save Beats And Prompts').onclick();
  assert.equal(state.segments[0].minimax_h3_prompt, 'Close-up, Mara folds a shirt.');
  assert.equal(state.segments[0].minimax_h3_prompt_origin, 'manual');
  assert.equal(state.segments[1].minimax_h3_prompt_origin, undefined, 'unchanged prompts keep their origin');
  assert.ok(calls.saves.includes('story builder beats and prompts edited'));
  findButton('Close').onclick();
  await tick();
});

test('Reference to Video blocks prompts for scenes with no image and opens the image generator', async () => {
  const { state, calls, builder } = setup();
  state.projectVideoEngine = 'minimax_h3';
  state.miniMaxH3Settings = { video_mode: 'reference_to_video' };
  state.segments[0].story_beat = 'A beat.';
  // Mara has an image and is mapped to scenes 1-3; scenes 4-6 only have an imageless location.
  state.fluxReferenceBuilder.subject_scene_map = { seg_1: ['mara'], seg_2: ['mara'], seg_3: ['mara'] };
  state.fluxReferenceBuilder.locations = [{ id: 'loc_1', name: 'Train - Sleeper Car', story_setting_id: 'setting_train', image: {} }];
  state.fluxReferenceBuilder.scene_map = { seg_4: 'loc_1', seg_5: 'loc_1', seg_6: 'loc_1' };
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  assert.equal(findButton('Create Video Prompts').disabled, true);
  assert.ok(document.body.textContent.includes('Scenes 4-6 have no character or location image'));
  await findButton('Generate Missing Location Images').onclick();
  await tick();
  assert.deepEqual(calls.generated, ['locations'], 'it opens the Places tab and its image generator');
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  state.fluxReferenceBuilder.locations[0].image = { path: 'C:/car.png' };
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  assert.equal(findButton('Create Video Prompts').disabled, false);
  findButton('Close').onclick();
  await tick();
});

test('the story cast checklist saves who the story is about', async () => {
  const { state, builder } = setup();
  state.fluxReferenceBuilder.subjects.push({ id: 'joe', name: 'Joe', reference_type: 'character', image: {} });
  builder.openStoryBuilder({ tab: 'story' });
  await tick();
  const boxes = document.body.all().filter((node) => node.tagName === 'INPUT' && node.type === 'checkbox');
  assert.equal(boxes.length, 2, 'one tick per character; no option to add new people');
  assert.ok(!document.body.textContent.includes('Find Cast In Story'), 'no scan for other characters');
  const [mara, joe] = boxes;
  assert.equal(mara.checked && joe.checked, true, 'every card is ticked when none were picked');
  const roles = document.body.all().filter((node) => node.tagName === 'SELECT' && node.options.some((option) => option.value === 'love_interest') && node.options.some((option) => option.value === 'main'));
  assert.deepEqual(roles.map((select) => select.value), ['main', 'supporting'], 'the first character is the main character');
  // Make Joe the main character: Mara becomes supporting.
  roles[1].value = 'main';
  await roles[1].onchange();
  assert.equal(state.builderStoryLayer.story_main_cast_id, 'joe');
  assert.deepEqual(state.builderStoryLayer.story_cast_roles, { mara: 'supporting', joe: 'main' });
  assert.equal(roles[0].value, 'supporting');
  roles[0].value = 'love_interest';
  await roles[0].onchange();
  assert.deepEqual(state.builderStoryLayer.story_cast_roles, { mara: 'love_interest', joe: 'main' });
  // A relationship between them.
  await findButton('Add Relationship').onclick();
  assert.deepEqual(state.builderStoryLayer.story_relationships, [{ a: 'mara', b: 'joe', type: 'love_interest', note: '' }]);
  joe.checked = false;
  await joe.onchange();
  assert.deepEqual(state.builderStoryLayer.story_cast_ids, ['mara']);
  assert.equal(state.builderStoryLayer.story_main_cast_id, 'mara', 'an unticked main character hands over to a ticked one');
  mara.checked = false;
  await mara.onchange();
  assert.deepEqual(state.builderStoryLayer.story_cast_ids, ['mara'], 'unticking everyone is refused');
  findButton('Close').onclick();
  await tick();
});

test('a scene cast set with the chips is the user\'s, reaches the scene mapping and keeps the direction', async () => {
  const { state, builder } = setup();
  state.fluxReferenceBuilder.subjects.push({ id: 'joe', name: 'Joe', reference_type: 'character', image: {} });
  state.builderStoryLayer.story_scene_plan[0].cast_ids = ['mara'];
  state.builderStoryLayer.story_scene_plan[0].cast_source = 'story';
  builder.openStoryBuilder({ tab: 'scenes' });
  await tick();
  const joeChips = document.body.all().filter((node) => node.tagName === 'BUTTON' && node.textContent === 'Joe');
  assert.equal(joeChips.length, 6, 'one chip per character per scene');
  joeChips[0].onclick();
  assert.equal(joeChips[0].textContent, '✓ Joe');
  const notes = document.body.all().filter((node) => node.tagName === 'TEXTAREA' && node.placeholder.startsWith('Your direction for this scene'));
  notes[0].value = 'Joe arrives with flowers.';
  await findButton('Save Scene Plan And Notes').onclick();
  const entry = state.builderStoryLayer.story_scene_plan.find((item) => item.scene_id === 'seg_1');
  assert.deepEqual([entry.cast_ids, entry.cast_source, entry.user_note], [['mara', 'joe'], 'manual', 'Joe arrives with flowers.']);
  assert.deepEqual(state.fluxReferenceBuilder.subject_scene_map.seg_1, ['mara', 'joe']);
  const untouched = state.builderStoryLayer.story_scene_plan.find((item) => item.scene_id === 'seg_2');
  assert.equal(untouched.cast_source, '', 'scenes you did not change are not marked as yours');
  findButton('Close').onclick();
  await tick();
});

test('the Song tab hosts Scene Defaults and saves them when you leave it', async () => {
  const { calls, builder } = setup();
  builder.openStoryBuilder({ tab: 'song' });
  await tick();
  assert.ok(document.body.textContent.includes('Scene Defaults'));
  assert.deepEqual(calls.defaults, ['open']);
  builder.openStoryBuilder({ tab: 'story' });
  await tick();
  assert.deepEqual(calls.defaults, ['open', 'save', 'close']);
  findButton('Close').onclick();
  await tick();
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

test('continued scenes become masked continuations, can be ticked by hand and carry a direction', async () => {
  const { state, calls, builder } = setup();
  state.projectVideoEngine = 'minimax_h3';
  state.miniMaxH3Settings = { video_mode: 'reference_to_video', render_pass: 'single', continuity_mode: 'off' };
  builder.openStoryBuilder({ tab: 'scenes' });
  await tick();
  const modeSelect = document.body.all().find((node) => node.tagName === 'SELECT' && node.options.some((option) => option.value === 'location'));
  modeSelect.value = 'location';
  await modeSelect.onchange();
  await tick();
  const masked = (segment) => segment.use_scene_minimax_h3_settings === true
    && segment.minimax_h3_settings.continuity_mode === 'latent_continuation_masked'
    && segment.minimax_h3_settings.location_transition_preset === 'masked'
    && segment.minimax_h3_settings.continuity_prompt_from_last_frame === true;
  // Kitchen is scenes 1-3 (chapter 1), Sleeper Car is scenes 4-6 (chapter 2): a new chapter never continues.
  assert.deepEqual(state.segments.map(masked), [false, true, true, false, true, true]);
  assert.equal(state.builderStoryLayer.story_continuation_mode, 'location');
  assert.ok(document.body.textContent.includes('MASKED: Scene 1 → Scene 2'), 'a visible marker names both scenes');
  assert.ok(document.body.textContent.includes('4 scenes continue from the scene before'));

  // Untick scene 2 by hand, give scene 3 a direction.
  const ticks = document.body.all().filter((node) => node.tagName === 'INPUT' && node.type === 'checkbox'
    && node.parent?.textContent.includes('Continue from'));
  assert.equal(ticks.length, 5, 'every scene but the first can continue');
  ticks[0].checked = false;
  ticks[0].onchange();
  const directions = document.body.all().filter((node) => node.tagName === 'TEXTAREA' && node.placeholder.startsWith('During the continuation'));
  directions[1].value = 'She drops the letter and runs out.';
  await findButton('Save Scene Plan And Notes').onclick();
  await tick();
  assert.deepEqual(state.segments.map(masked), [false, false, true, false, true, true]);
  assert.equal(state.segments[1].story_masked_continuation, undefined, 'the story turned its own change back off');
  assert.equal(state.segments[2].minimax_h3_continuation_direction, 'She drops the letter and runs out.');
  const plan = Object.fromEntries(state.builderStoryLayer.story_scene_plan.map((entry) => [entry.scene_id, entry]));
  assert.deepEqual([plan.seg_2.continues_previous, plan.seg_2.continuation_source], [false, 'manual']);
  assert.deepEqual([plan.seg_3.continues_previous, plan.seg_3.continuation_source], [true, 'auto']);

  // The mode no longer overrides the hand-made choice.
  builder.openStoryBuilder({ tab: 'scenes' });
  await tick();
  assert.equal(state.segments[1].use_scene_minimax_h3_settings, false);

  // Beats & Prompts marks the continued scenes and edits their direction.
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  assert.ok(document.body.textContent.includes('3 scenes are masked continuations'));
  const promptDirections = document.body.all().filter((node) => node.tagName === 'TEXTAREA' && node.placeholder === 'What happens once the carried-on moment settles.');
  assert.equal(promptDirections.length, 3);
  promptDirections[0].value = 'She sits on the bed.';
  await findButton('Save Beats And Prompts').onclick();
  assert.equal(state.segments[2].minimax_h3_continuation_direction, 'She sits on the bed.');
  assert.ok(calls.saves.includes('story builder continuation directions'));
  findButton('Close').onclick();
  await tick();
});

test('reset scene plan clears the plan and the masked continuations it set', async () => {
  const { state, calls, builder } = setup();
  state.projectVideoEngine = 'minimax_h3';
  state.miniMaxH3Settings = { video_mode: 'reference_to_video', render_pass: 'single', continuity_mode: 'off' };
  builder.openStoryBuilder({ tab: 'scenes' });
  await tick();
  const modeSelect = document.body.all().find((node) => node.tagName === 'SELECT' && node.options.some((option) => option.value === 'location'));
  modeSelect.value = 'location';
  await modeSelect.onchange();
  await tick();
  assert.equal(state.segments[1].story_masked_continuation, true);
  await findButton('Reset Scene Plan').onclick();
  await tick();
  assert.deepEqual(state.builderStoryLayer.story_scene_plan, []);
  assert.equal(state.segments[1].use_scene_minimax_h3_settings, false, 'the masked scene is back to the project settings');
  assert.equal(state.segments[1].timeline_note, 'She runs to the porch.', 'notes are kept');
  assert.ok(calls.saves.includes('story builder scene plan reset'));
  assert.equal(findButton('Reset Scene Plan').disabled, true);
  findButton('Close').onclick();
  await tick();
});

test('the premise saves on its own and nothing is shown as out of date', async () => {
  const { state, calls, builder } = setup();
  builder.openStoryBuilder({ tab: 'story' });
  await tick();
  assert.ok(!document.body.textContent.includes('The story is out of date'));
  const premise = document.body.all().find((node) => node.tagName === 'TEXTAREA' && node.placeholder.startsWith('A son rides'));
  premise.value = 'Mara leaves home for good.';
  await findButton('Save Premise').onclick();
  assert.equal(state.builderStoryLayer.overall_story_idea, 'Mara leaves home for good.');
  assert.ok(calls.saves.includes('story builder premise'));
  { const t = document.body.textContent; const i = t.indexOf('out of date'); assert.ok(i < 0, 'no out-of-date warnings: ' + t.slice(Math.max(0, i - 120), i + 80)); }
  findButton('Close').onclick();
  await tick();
});

test('scenes are grouped in act sections, and Replan asks for a location and story, then replans the act or ticked scenes', async () => {
  const { state, calls, builder } = setup();
  const bodies = [];
  globalThis.fakeComfyApi.fetchApi = async (url, options) => {
    bodies.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ ok: true, story_layer: { story_scene_plan: state.builderStoryLayer.story_scene_plan } }) };
  };
  try {
    builder.openStoryBuilder({ tab: 'scenes' });
    await tick();
    const text = document.body.textContent;
    assert.ok(text.includes('Chapter 1: Home') && text.includes('Chapter 2: Away'), 'each act is a section under its chapter');
    assert.ok(!findButton('Replan Out-Of-Date Acts (1)'));
    // Replan This Act opens a window for the location and a bit of story, and can replan only the ticked scenes.
    findButton('Replan This Act').onclick();
    const boxes = document.body.all().filter((node) => node.tagName === 'TEXTAREA' && /^(Where it happens|What must happen)/.test(node.placeholder));
    assert.equal(boxes.length, 2, 'a location box and a story box');
    boxes[0].value = 'A rooftop at dusk.';
    boxes[1].value = 'She burns the letter.';
    const mode = document.body.all().find((node) => node.tagName === 'SELECT' && node.options?.some((option) => option.value === 'some'));
    mode.value = 'some';
    mode.onchange();
    const sceneTicks = document.body.all().filter((node) => node.tagName === 'INPUT' && node.type === 'checkbox' && node.parent?.textContent?.startsWith('Scene '));
    sceneTicks[1].checked = true;
    await findButton('Replan').onclick();
    await tick();
    assert.deepEqual(bodies.filter((body) => body.scene_ids).map((body) => [body.act_id, body.scene_ids]), [['act_1', ['seg_2']]], 'only the ticked scene is replanned');
    assert.ok(bodies.some((body) => body.act_id === 'act_1' && !body.scene_ids), 'the act description is rewritten from the new plan');
    assert.deepEqual(calls.beats.at(-1), { overwrite: true, sceneIds: ['seg_2'] }, 'the replanned scene gets a new beat from its new plan');
    const act = state.builderStoryLayer.story_acts[0];
    assert.deepEqual([act.location_note, act.direction], ['A rooftop at dusk.', 'She burns the letter.'], 'the act keeps your words');
  } finally {
    delete globalThis.fakeComfyApi.fetchApi;
  }
  findButton('Close').onclick();
  await tick();
});

test('beats and prompts are redone for ticked scenes or one scene only', async () => {
  const { state, calls, builder } = setup();
  state.projectVideoEngine = 'minimax_h3';
  for (const segment of state.segments) segment.story_beat = `beat of ${segment.id}`;
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  assert.equal(findButton('Redo Beats (0)').disabled, true);
  assert.equal(findButton('Redo Prompts (0)').disabled, true);
  // Changing a plan marks nothing out of date; a scene's own Redo Beat runs just that scene.
  state.builderStoryLayer.story_scene_plan[1].story_slice = 'Mara burns the letter.';
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  assert.ok(!document.body.textContent.includes('OUT OF DATE'));
  assert.equal(findButton('Redo Beats (0)').disabled, true);
  const oneBeat = document.body.all().filter((node) => node.tagName === 'BUTTON' && node.textContent === 'Redo Beat');
  await oneBeat[1].onclick();
  await tick();
  assert.deepEqual(calls.beats.at(-1), { overwrite: true, sceneIds: ['seg_2'] });
  // A ticked scene is added to both buttons.
  builder.openStoryBuilder({ tab: 'prompts' });
  await tick();
  const ticks = document.body.all().filter((node) => node.tagName === 'INPUT' && node.type === 'checkbox');
  ticks[2].checked = true;
  ticks[2].onchange();
  assert.ok(findButton('Redo Beats (1)'));
  await findButton('Redo Prompts (1)').onclick();
  await tick();
  assert.deepEqual(calls.promptArgs.at(-1), { sceneIds: ['seg_3'], storyMode: true });
  assert.ok(findButton('Redo Prompts (0)'), 'ticks are cleared after a run');
  findButton('Close').onclick();
  await tick();
});

test('planning all scenes runs act by act and says which act and scenes it is on', async () => {
  const { state, calls, builder } = setup();
  const bodies = [];
  globalThis.fakeComfyApi.fetchApi = async (url, options) => {
    bodies.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ ok: true, story_layer: { story_scene_plan: state.builderStoryLayer.story_scene_plan } }) };
  };
  try {
    builder.openStoryBuilder({ tab: 'scenes' });
    await tick();
    await findButton('Replan All Scenes').onclick();
    await tick();
    assert.deepEqual(bodies.map((body) => [body.act_id, body.unload_after]), [['act_1', false], ['act_2', true]],
      'one call per act, and the model is only let go after the last one');
    const text = calls.progress.join(' | ');
    assert.ok(text.includes('Act 1 of 2: Pack') && text.includes('Act 2 of 2: Ride'));
    assert.ok(text.includes('Scenes 1-3 (3 scenes)') && text.includes('Waiting for the LLM: 0s'));
    assert.ok(text.includes('Planned 2 acts.'));
  } finally {
    delete globalThis.fakeComfyApi.fetchApi;
  }
  findButton('Close').onclick();
  await tick();
});

test('Update Story From This Scene keeps the change and adapts the act', async () => {
  const { state, calls, builder } = setup();
  state.fluxReferenceBuilder.subjects.push({ id: 'joe', name: 'Joe', reference_type: 'character', image: {} });
  const bodies = [];
  globalThis.fakeComfyApi.fetchApi = async (url, options) => {
    const body = JSON.parse(options.body);
    bodies.push([url, body]);
    const layerSent = body.story_layer;
    const plan = layerSent.story_scene_plan.map((entry) => (entry.scene_id === 'seg_2'
      ? { ...entry, story_slice: 'Joe arrives with flowers.', interaction: 'Joe offers the flowers.' } : entry));
    const acts = layerSent.story_acts.map((act) => (act.id === 'act_1' ? { ...act, summary: 'Packing, until Joe arrives.' } : act));
    return { ok: true, json: async () => ({ ok: true, chapter_id: 'ch_1', other_scenes: [{ scene_id: 'seg_3', reason: 'Joe is there now' }],
      story_layer: { story_scene_plan: plan, story_acts: acts, story_book: layerSent.story_book, story_book_text: layerSent.story_book_text } }) };
  };
  try {
    builder.openStoryBuilder({ tab: 'scenes' });
    await tick();
    const joeChips = document.body.all().filter((node) => node.tagName === 'BUTTON' && node.textContent === 'Joe');
    joeChips[1].onclick();
    const notes = document.body.all().filter((node) => node.tagName === 'TEXTAREA' && node.placeholder.startsWith('Your direction for this scene'));
    notes[1].value = 'Joe arrives with flowers.';
    const updates = document.body.all().filter((node) => node.tagName === 'BUTTON' && node.textContent === 'Update Story From This Scene');
    assert.equal(updates.length, 6);
    await updates[1].onclick();
    await tick();
    const [url, sent] = bodies[0];
    assert.ok(url.endsWith('/story_mode/revise_scene'));
    assert.equal(sent.scene_id, 'seg_2');
    const sentEntry = sent.story_layer.story_scene_plan.find((entry) => entry.scene_id === 'seg_2');
    assert.deepEqual([sentEntry.cast_ids, sentEntry.cast_source, sentEntry.user_note], [['joe'], 'manual', 'Joe arrives with flowers.'],
      'your change is saved before the LLM step');
    assert.equal(state.builderStoryLayer.story_acts[0].summary, 'Packing, until Joe arrives.');
    assert.deepEqual(calls.beats, [{ overwrite: true, sceneIds: ['seg_2'] }], 'the updated scene gets its new beat for 7. Beats & Prompts');
    const redo = document.body.all().filter((node) => node.tagName === 'BUTTON' && node.textContent.startsWith('Save And Redo This Scene'));
    await redo[2].onclick();
    await tick();
    assert.deepEqual(calls.beats.at(-1), { overwrite: true, sceneIds: ['seg_3'] }, 'the per-scene button redoes only that scene');
    assert.deepEqual(state.fluxReferenceBuilder.subject_scene_map.seg_2, ['joe']);
    const text = document.body.textContent;
    assert.ok(!text.includes('out of date'), 'nothing is shown as out of date');
  } finally {
    delete globalThis.fakeComfyApi.fetchApi;
  }
  findButton('Close').onclick();
  await tick();
});

test('each scene in the Scenes tab plays its part of the song, and stops at its end', async () => {
  const { state, builder } = setup();
  state.audioPath = 'C:/song.mp3';
  const played = [];
  let lastAudio = null;
  FakeElement.prototype.play = function play() {
    lastAudio = this;
    played.push(this.currentTime);
    this.paused = false;
    return Promise.resolve();
  };
  FakeElement.prototype.pause = function pause() { this.paused = true; };
  try {
    builder.openStoryBuilder({ tab: 'scenes' });
    await tick();
    const plays = document.body.all().filter((node) => node.tagName === 'BUTTON' && node.textContent === '▶ Play Scene Audio');
    assert.equal(plays.length, 6, 'one player button per scene');
    plays[1].onclick();
    assert.deepEqual(played, [5], 'scene 2 starts at its own start');
    assert.equal(lastAudio.src.includes(encodeURIComponent('C:/song.mp3')), true);
    // At the scene's end (10s) the player stops by itself.
    lastAudio.currentTime = 7;
    lastAudio.ontimeupdate();
    assert.equal(plays[1].textContent, '■ Stop 0:02.0 / 0:05.0');
    lastAudio.currentTime = 10;
    lastAudio.ontimeupdate();
    assert.equal(plays[1].textContent, '▶ Play Scene Audio');
    assert.equal(lastAudio.paused, true);
    plays[1].onclick();
    plays[2].onclick();
    assert.equal(plays[1].textContent, '▶ Play Scene Audio', 'starting scene 3 stops scene 2');
    assert.deepEqual(played, [5, 5, 10]);
    plays[2].onclick();
    assert.equal(plays[2].textContent, '▶ Play Scene Audio', 'a second click stops it');
    findButton('Close').onclick();
    await tick();
  } finally {
    delete FakeElement.prototype.play;
    delete FakeElement.prototype.pause;
  }
});
