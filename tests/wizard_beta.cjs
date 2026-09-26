const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/VRGDG_WizardBeta.js'), 'utf8');
const builder = fs.readFileSync(path.join(__dirname, '../web/VRGDG_MusicVideoBuilderUI.js'), 'utf8');

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.disabled = false; this.value = ''; this.style = {};  }
  set textContent(value) { this.text = value; this.children = []; }
  get textContent() { return (this.text || '') + this.children.map(child => child.textContent).join(''); }
  append(...children) { for (const child of children) { child.remove(); child.parent = this; this.children.push(child); } }
  prepend(child) { child.remove(); child.parent = this; this.children.unshift(child); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); this.parent = null; }
  replaceChildren(...children) { this.text = ''; this.children.forEach(child => { child.parent = null; }); this.children = []; this.append(...children); }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener() {}
  focus() {}
  getClientRects() { return this.hidden ? [] : [{}]; }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(selector.split(',').includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)]); }
}


const modeOptions = {
  ltx: ['i2v', 't2v', 'rtv', 'ingredients', 'id_lora', 'flf'].map(value => ({ value, label: value })),
  minimax_h3: ['text_to_video', 'image_to_video', 'reference_to_video', 'image_reference_to_video', 'video_to_video'].map(value => ({ value, label: value })),
};
function fixture(overrides = {}) {
  const document = { createElement: tag => new Element(tag), createTextNode: text => { const node = new Element('#text'); node.textContent = text; return node; }, getElementById: () => null, body: new Element('body'), head: new Element('head') };
  const data = { engine: 'minimax_h3', mode: 'text_to_video', performance: 'speaking', performances: [{ value: 'speaking', label: 'Speaking' }], audioMode: 'built_in_audio', audioModes: [{ value: 'input_audio', label: 'Input audio' }, { value: 'built_in_audio', label: 'Built-in audio' }], imageMode: 'zimage', imageModes: [{ value: 'zimage', label: 'ZImage' }], modes: modeOptions, lyrics: '', direction: '', scenes: [], referenceCount: 0, ...overrides };
  const events = [];
  const api = {
    snapshot: () => data, flush() {}, configure: values => Object.assign(data, values),
    imageUrl: path => `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}`,
    mountSettings: (holder, kind) => { events.push(`mount:${kind}`); return () => events.push(`restore:${kind}`); },
    save: async (draft, edits) => { events.push('save'); data.draft = JSON.parse(JSON.stringify(draft)); for (const [id, fields] of edits) Object.assign(data.scenes.find(scene => scene.id === id), fields); },
    openTiming: async kind => { events.push(`timing:${kind}`); data.scenes = [{ id: 'a', start: 0, end: 8, lyric_text: 'A' }]; },
    render: async () => events.push('render'), openReferences() {}, openRunner() {}, openLyrics() {}, openStoryboard() {}, openScene() {}, generateImages() {}, generatePrompts() {},
  };
  const context = vm.createContext({ document, window: { confirm: () => true }, api });
  vm.runInContext(source.replaceAll('export function ', 'function ').replace('export const ', 'const '), context);
  vm.runInContext('openWizardBeta(api)', context);
  const find = (tag, text) => document.body.querySelectorAll(tag).find(element => element.textContent === text);
  const click = async text => { const control = find('button', text); assert.ok(control, text); assert.equal(control.disabled, false); await control.onclick(); };
  const input = label => document.body.querySelectorAll('label').find(element => element.children[0]?.textContent === label)?.children[1];
  return { context, document, api, data, events, click, input, find };
}

test('starts at engine selection and advances through ordered steps', async () => {
  const f = fixture();
  assert.ok(f.find('h2', '1. Engine'));
  await f.click('Next →'); assert.ok(f.input('Video mode'));
  await f.click('Next →'); assert.ok(f.input('Characters (optional)'));
  assert.equal(f.input('Text-to-image model'), undefined);
});

test('image mode shows image model selection and restores native panel when leaving', async () => {
  const f = fixture({ mode: 'image_to_video' });
  await f.click('3 Inputs'); assert.ok(f.input('Text-to-image model'));
  assert.ok(f.events.includes('mount:image'));
  await f.click('4 Models & LoRAs');
  assert.ok(f.events.includes('restore:image')); assert.ok(f.events.includes('mount:video'));
});

test('timing choices replace fixed durations and skip transcription for built-in audio', async () => {
  const f = fixture(); await f.click('5 Sound & timing');
  assert.equal(f.input('Number of scenes'), undefined);
  assert.equal(f.input('Scene length (seconds)'), undefined);
  assert.equal(f.find('button', '1. Existing scenes').disabled, true);
  assert.equal(f.find('button', '2. No scenes yet').disabled, true);
  await f.click('3. Manual timing');
  assert.deepEqual(f.events, ['save', 'timing:manual']);
  await f.click('6 Scenes');
  assert.equal(f.find('button', 'Create scenes from this setup'), undefined);
});

test('input audio timing requires a file and gates choices by existing scenes', async () => {
  const f = fixture({ audioMode: 'input_audio' }); await f.click('5 Sound & timing');
  for (const label of ['1. Existing scenes', '2. No scenes yet', '3. Manual timing']) assert.equal(f.find('button', label).disabled, true);
  f.data.audioPath = 'song.wav'; await f.click('5 Sound & timing');
  assert.equal(f.find('button', '1. Existing scenes').disabled, true);
  await f.click('2. No scenes yet');
  assert.equal(f.find('button', '2. No scenes yet').disabled, false);
  assert.ok(f.document.body.textContent.includes('recreate them from the audio'));
  await f.click('1. Existing scenes');
  assert.deepEqual(f.events, ['save', 'timing:new', 'save', 'timing:existing']);
  const g = fixture({ audioModes: [{ value: 'built_in_audio', disabled: true }] });
  await g.click('Save Project'); assert.deepEqual(g.events, []);
});

test('save does not prepare or render, preserves stage and handles failure', async () => {
  const f = fixture(); await f.click('3 Inputs');
  const characters = f.input('Characters (optional)'); characters.value = 'A detective'; characters.oninput();
  await f.click('Save Project'); assert.deepEqual(f.events, ['save']);
  assert.equal(f.data.draft.page, 2); assert.equal(f.data.draft.characters, 'A detective');
  f.api.save = async () => { throw new Error('Disk full'); };
  await f.click('Save Project'); assert.ok(f.document.body.textContent.includes('Disk full'));
});

test('story direction stays staged until save and remembers the scene substep', async () => {
  const f = fixture({ direction: 'Before' });
  await f.click('6 Scenes'); await f.click('2 Story direction');
  const direction = f.input('Story direction'); direction.value = 'After'; direction.oninput();
  assert.equal(f.data.direction, 'Before'); await f.click('Save Project');
  assert.equal(f.data.draft.direction, 'After'); assert.equal(f.data.draft.sceneStep, 1);
});

test('every mode has the correct media requirements', () => {
  const f = fixture();
  const needs = (engine, mode) => f.context.wizardBetaNeeds(engine, mode);
  assert.equal(needs('ltx', 't2v').images, false);
  assert.equal(needs('ltx', 'ingredients').references, true);
  assert.equal(needs('ltx', 'flf').endFrame, true);
  assert.equal(needs('ltx', 'id_lora').images, true);
  assert.equal(needs('minimax_h3', 'video_to_video').video, true);
  assert.equal(needs('minimax_h3', 'image_reference_to_video').images, true);
  assert.equal(needs('minimax_h3', 'image_reference_to_video').references, true);
});

function adapterFixture() {
  const state = { projectVideoEngine: 'minimax_h3', miniMaxH3Settings: { video_mode: 'text_to_video', audio_mode: 'built_in_audio' }, videoModelMode: 't2v', imageModelMode: 'zimage', videoType: 'speaking', projectFolder: '/project', segments: [], overlaySegments: [], fluxKleinSettings: {}, fluxReferenceBuilder: { subjects: [], locations: [], subject_scene_map: {}, scene_map: {} }, i2vVideoSettings: {} };
  const events = [];
  const context = { state, wizardGlobalVideoSettings: false, audioInput: { value: '' }, projectInput: { value: '/project' }, audio: { duration: 0 },
    MINIMAX_H3_MODE_OPTIONS: modeOptions.minimax_h3, MINIMAX_H3_AUDIO_MODE_OPTIONS: [{ value: 'input_audio', label: 'Input' }, { value: 'built_in_audio', label: 'Built-in' }], VIDEO_TYPE_OPTIONS: [],
    normalizeProjectVideoEngine: x => x, normalizeVideoType: x => x,
    makeEditorImageUrl: path => `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}`,
    cloneMiniMaxH3Settings: x => ({ ...x }), cloneI2VVideoSettings: x => ({ ...x }), normalizeFluxReferenceBuilder: x => x, normalizeBuilderStoryLayer: x => x, normalizeLyricMapper: x => x,
    currentVideoMode: () => state.videoModelMode, allEditableSegments: () => state.segments, activeSegment: () => state.segments[0], imageModeDisplayLabel: x => x,
    openWizardBeta: api => { context.api = api; }, wizardBetaNeeds: fixture().context.wizardBetaNeeds,
    applyBulkSegmentTimings: async times => { state.segments = times.map((time, i) => ({ id: String(i), ...time })); },
    saveSession: async () => events.push('save'), createSilentTimelineAudioForDuration: async () => events.push('silent'), chooseProjectAudioFile: async () => events.push('audio'),
    confirmAndRunFullBuild: async () => events.push('ltx-build'),
    chooseBatchModeAction: async () => 'resume', renderAllScenes: async () => events.push('minimax-render'),
    miniMaxH3ModeForSegment: scene => scene.minimax_h3_mode || state.miniMaxH3Settings.video_mode,
    segmentImageSource: () => null, createProgressWindow: () => ({ set() {}, close() {} }),
    runMiniMaxH3PromptGeneration: async (scene, mode) => { events.push(`prompt:${mode}`); return { prompt: 'Native MiniMax prompt' }; },
    autoSaveSessionQuiet: async () => {},
  };
  for (const name of ['updateActiveFromInputs','saveI2VVideoSettingsFromPanel','saveMiniMaxH3SettingsFromPanel','saveMiniMaxSceneInputsFromPanel','saveZImageSettingsFromPanel','saveFluxKleinSettingsFromPanel','saveErnieImageSettingsFromPanel','saveKrea2TwoPassSettingsFromPanel','saveNBImageSettingsFromPanel','saveFlowGptBrowserSettingsFromPanel','syncProjectVideoEngineUI','syncVideoModePanel','syncI2VVideoSettingsPanel','syncFluxKleinPanel','syncZImageSettingsPanel','syncErnieImagePanel','syncKrea2TwoPassPanel','syncNBImagePanel','syncFlowGptBrowserPanel','syncInspector','render','pushHistory','openGemmaRunnerModal','openLyricReviewModal','openStoryboardBuilderFromProject','confirmAndRunZImageAll','importTimelineImagesFromFolder','syncVideoTypeControl','assertBatchNotStopped']) context[name] = () => {};
  vm.createContext(context);
  const start = builder.indexOf('  function openWizardBetaFromBuilder()');
  const end = builder.indexOf('  function openWizardFromBuilder()', start);
  vm.runInContext(builder.slice(start, end) + '\nopenWizardBetaFromBuilder();', context);
  return { context, state, events, api: context.api };
}

test('adapter preserves selected engine/mode/audio when saving and rendering existing text scenes', async () => {
  const f = adapterFixture();
  const draft = { engine: 'minimax_h3', mode: 'text_to_video', audioMode: 'built_in_audio', sceneSeconds: 5, sceneCount: 2, locations: [], characters: 'A detective', locationsText: 'A city', direction: 'Find a clue', sound: 'Rain', lyrics: '', videoPath: '', imageSource: 'generate' };
  f.state.segments = [{ id: 'a', start: 0, end: 5 }, { id: 'b', start: 5, end: 10 }];
  await f.api.save(draft, new Map());
  assert.equal(f.state.miniMaxH3Settings.video_mode, 'text_to_video');
  assert.equal(f.state.miniMaxH3Settings.audio_mode, 'built_in_audio');
  assert.equal(f.state.segments.length, 2); assert.equal(f.state.segments[1].end, 10);
  assert.ok(!f.events.includes('audio'));
  await f.api.render();
  assert.ok(!f.events.includes('prompt:text_to_video')); assert.ok(f.events.includes('minimax-render')); assert.ok(!f.events.includes('ltx-build'));
});

test('adapter exposes all LTX modes and constrains multipass MiniMax audio', () => {
  const f = adapterFixture();
  f.api.configure({ engine: 'ltx', mode: 'flf', audioMode: 'silent' });
  assert.equal(f.api.snapshot().mode, 'flf'); assert.equal(f.api.snapshot().audioMode, 'silent');
  assert.equal(f.api.snapshot().modes.ltx.filter(item => !item.disabled).length, 6);
  f.api.configure({ engine: 'minimax_h3', mode: 'image_reference_to_video' });
  assert.equal(f.state.videoModelMode, 'i2v');
  assert.equal(f.api.snapshot().audioModes.find(item => item.value === 'built_in_audio').disabled, true);
});


test('reference Inputs uses independent titles and descriptions below images, without generic boxes', async () => {
  const f = fixture({ mode: 'reference_to_video', draft: {
    singer: { name: 'hero.png', data: 'data:image/png;base64,eA==', title: 'Hero', description: 'Old description' },
    locations: [{ name: 'city.png', data: 'data:image/png;base64,eA==' }, { name: 'forest.png', data: 'data:image/png;base64,eA==' }],
  } });
  await f.click('3 Inputs');
  assert.equal(f.input('Characters (optional)'), undefined);
  assert.equal(f.input('Locations (optional)'), undefined);
  assert.equal(f.input('Character title').value, 'Hero');
  const title = f.input('Character title'); title.value = 'Raven'; title.oninput();
  const first = f.input('Location 1 description'); first.value = 'Rainy city'; first.oninput();
  const second = f.input('Location 2 description'); second.value = 'Moonlit forest'; second.oninput();
  assert.equal(f.data.draft.singer.title, 'Hero');
  await f.click('Save Project');
  assert.equal(f.data.draft.singer.title, 'Raven');
  assert.equal(f.data.draft.locations[0].description, 'Rainy city');
  assert.equal(f.data.draft.locations[1].description, 'Moonlit forest');
});

test('reference metadata saves to Reference Builder and reloads, including cleared descriptions', async () => {
  const f = adapterFixture();
  const draft = { engine: 'minimax_h3', mode: 'reference_to_video', characters: 'Hidden character text', locationsText: 'Hidden location text', lyrics: '', direction: '',
    singer: { name: 'hero.png', data: 'image', title: 'Raven', description: 'Black coat' },
    locations: [{ name: 'city.png', data: 'city', title: 'City', description: 'Rainy streets' }, { name: 'forest.png', data: 'forest', title: 'Forest', description: 'Moonlit trees' }],
  };
  await f.api.save(draft, new Map());
  const refs = f.state.fluxReferenceBuilder;
  assert.equal(refs.subjects[0].name, 'Raven'); assert.equal(refs.subjects[0].description, 'Black coat');
  assert.equal(refs.locations.length, 2); assert.equal(refs.locations[1].description, 'Moonlit trees');
  refs.subjects[0].name = 'Edited in Reference Builder';
  assert.equal(f.api.snapshot().draft.singer.title, 'Edited in Reference Builder');
  draft.singer.title = 'Renamed'; draft.singer.description = '';
  await f.api.save(draft, new Map());
  assert.equal(refs.subjects[0].name, 'Renamed'); assert.equal(refs.subjects[0].description, '');
  assert.equal(refs.subjects[0].image.name, 'hero.png');
});

test('focused reference buttons save inputs first and reload edited metadata on return', async () => {
  const f = fixture({ mode: 'reference_to_video', draft: { singer: { name: 'a.png', title: 'Before', data: 'a' }, locations: [] } });
  const opened = [];
  f.api.openReferences = (section, onClose) => { opened.push(section); f.data.draft.singer.title = 'Edited'; f.data.draft.subjects[0].title = 'Edited'; onClose(); };
  await f.click('3 Inputs');
  await f.click('Edit Subjects');
  assert.deepEqual(opened, ['subjects']);
  assert.equal(f.events[0], 'save');
  await f.click('Save Project');
  assert.equal(f.data.draft.singer.title, 'Edited');
  await f.click('Edit Locations');
  assert.equal(f.find('button', 'Edit Mappings'), undefined);
  f.data.scenes = [{ id: 'a', start: 0, end: 8 }];
  await f.click('6 Scenes'); await f.click('5 Edit Mappings'); await f.click('Edit Mappings');
  assert.deepEqual(opened, ['subjects', 'locations', 'mapping']);
  f.api.save = async () => { throw new Error('Save failed'); };
  await f.click('3 Inputs');
  await f.click('Edit Subjects');
  assert.equal(opened.length, 3);
});

test('focused reference editor only mounts its selected section; full editor keeps all tabs', () => {
  const start = builder.indexOf('    const referenceTabs = [');
  const end = builder.indexOf('    const footer =', start);
  for (const section of ['', 'subjects', 'locations', 'mapping']) {
    const context = vm.createContext({
      document: { createElement: tag => new Element(tag) }, focusedSection: section,
      miniMaxProject: true, options: {}, wizardLocationMode: false,
      subjectCard: new Element('subjects'), extrasCard: new Element('extras'),
      locationsCard: new Element('locations'), mappingCard: new Element('mapping'),
      tabBar: new Element('nav'), tabContent: new Element('main'), tabShell: new Element('section'),
    });
    vm.runInContext(builder.slice(start, end), context);
    assert.equal(context.tabContent.children.length, 1);
    assert.equal(context.tabContent.children[0].tagName, section || 'subjects');
    assert.equal(context.tabShell.children.includes(context.tabBar), !section);
    if (section) {
      vm.runInContext('setReferenceTab("extras")', context);
      assert.equal(context.tabContent.children[0].tagName, section);
    } else assert.equal(context.tabBar.children.length, 4);
  }
});

test('saved reference image paths render after editor return and survive another wizard save', async () => {
  const adapter = adapterFixture();
  const draft = { engine: 'minimax_h3', mode: 'reference_to_video', characters: '', locationsText: '', lyrics: '', direction: '',
    singer: { name: 'hero.png', data: 'data:image/png;base64,aA==' },
    locations: [{ name: 'city.png', data: 'data:image/png;base64,Yw==' }, { name: 'forest.png', data: 'data:image/png;base64,Zg==' }],
  };
  await adapter.api.save(draft, new Map());
  const f = fixture({ mode: 'reference_to_video', draft });
  f.api.openReferences = (section, onClose) => {
    const refs = adapter.state.fluxReferenceBuilder;
    refs.subjects[0].name = 'Edited hero';
    refs.subjects[0].description = 'Edited description';
    for (const ref of [...refs.subjects, ...refs.locations]) {
      ref.image = { name: ref.image.name, data: '', path: `C:\\My Project\\references\\${ref.image.name}` };
    }
    f.data.draft = adapter.api.snapshot().draft;
    onClose();
  };
  await f.click('3 Inputs');
  assert.equal(f.document.body.querySelectorAll('img')[0].src, draft.singer.data);
  await f.click('Edit Subjects');
  const expected = ['hero.png', 'city.png', 'forest.png'].map(name => adapter.api.imageUrl(`C:\\My Project\\references\\${name}`));
  assert.deepEqual(f.document.body.querySelectorAll('img').map(img => img.src), expected);
  assert.equal(f.input('Character title').value, 'Edited hero');
  assert.equal(f.input('Character description').value, 'Edited description');
  await f.click('Save Project');
  await adapter.api.save(f.data.draft, new Map());
  const reloaded = fixture({ mode: 'reference_to_video', draft: adapter.api.snapshot().draft });
  await reloaded.click('3 Inputs');
  assert.deepEqual(reloaded.document.body.querySelectorAll('img').map(img => img.src), expected);
});

test('runner and performance selections survive creating a project on first save', async () => {
  const f = adapterFixture();
  f.state.projectFolder = '';
  f.context.projectInput.value = '';
  f.state.textGemmaRunner = 'qwen_local';
  f.state.qwenModelFile = 'chosen.gguf';
  f.state.qwenMmprojFile = 'vision.gguf';
  f.api.configure({ performance: 'speaking' });
  let syncedPerformance;
  f.context.syncVideoTypeControl = () => { syncedPerformance = f.state.videoType; };
  f.context.newProject = async () => {
    f.state.projectFolder = '/new-project';
    f.state.textGemmaRunner = 'builtin';
    f.state.qwenModelFile = '';
    f.state.qwenMmprojFile = '';
    f.state.videoType = 'singing';
    return true;
  };
  await f.api.save({ engine: 'minimax_h3', mode: 'text_to_video', characters: '', locationsText: '', lyrics: '', direction: '', subjects: [], locations: [] }, new Map());
  assert.equal(f.state.textGemmaRunner, 'qwen_local');
  assert.equal(f.state.qwenModelFile, 'chosen.gguf');
  assert.equal(f.state.qwenMmprojFile, 'vision.gguf');
  assert.equal(f.api.snapshot().performance, 'speaking');
  assert.equal(syncedPerformance, 'speaking');
});

test('first wizard opening loads existing references and saves without duplicating them', async () => {
  const f = adapterFixture();
  f.api.configure({ mode: 'reference_to_video' });
  f.state.wizardBetaDraft = null;
  f.state.fluxReferenceBuilder.subjects = [{ id: 'hero', name: 'Hero', description: 'Detective', image: { path: '/hero.png', name: 'hero.png' } }];
  f.state.fluxReferenceBuilder.locations = [{ id: 'city', name: 'City', description: 'At night', image: { path: '/city.png', name: 'city.png' } }];
  const wizard = fixture(f.api.snapshot());
  wizard.api.save = f.api.save;
  assert.equal(wizard.find('span', 'Saved draft loaded'), undefined);
  await wizard.click('3 Inputs');
  assert.deepEqual(wizard.document.body.querySelectorAll('img').map(img => img.src), ['/hero.png', '/city.png'].map(f.api.imageUrl));
  await wizard.click('Save Project');
  assert.deepEqual(f.state.fluxReferenceBuilder.subjects.map(ref => ref.id), ['hero']);
  assert.deepEqual(f.state.fluxReferenceBuilder.locations.map(ref => ref.id), ['city']);
  assert.equal(f.state.wizardBetaDraft.subjects[0].description, 'Detective');
  assert.equal(f.state.wizardBetaDraft.locations[0].description, 'At night');
});

test('LLM Runner stays in the header immediately before Save Project', async () => {
  const f = fixture();
  let opened = 0;
  f.api.openRunner = () => opened++;
  const header = f.document.body.querySelectorAll('header')[0];
  assert.deepEqual(header.children.filter(child => child.tagName === 'button').map(child => child.textContent), ['Configure LLM Runner', 'Save Project', 'Close']);
  await f.click('Configure LLM Runner');
  assert.equal(opened, 1);
  for (const step of ['3 Inputs', '6 Scenes']) {
    await f.click(step);
    assert.equal(f.document.body.querySelectorAll('main')[0].querySelectorAll('button').filter(child => child.textContent === 'Configure LLM Runner').length, 0);
    assert.equal(f.document.body.querySelectorAll('button').filter(child => child.textContent === 'Configure LLM Runner').length, 1);
  }
  await f.click('Configure LLM Runner');
  assert.equal(opened, 2);
  assert.equal(f.events.includes('save'), false);
});

test('subjects added in Edit Subjects remain visible in LTX and MiniMax wizard drafts', async () => {
  for (const [engine, mode] of [['ltx', 'rtv'], ['minimax_h3', 'reference_to_video']]) {
    const adapter = adapterFixture();
    adapter.api.configure({ engine, mode });
    await adapter.api.save({ engine, mode, characters: '', locationsText: '', lyrics: '', direction: '', subjects: [], locations: [] }, new Map());
    adapter.state.fluxReferenceBuilder.subjects.push({
      id: 'editor_subject', name: 'New hero', description: 'Red coat',
      image: { name: 'hero.png', path: 'C:\\references\\hero.png', data: '' },
    });
    const refreshed = adapter.api.snapshot().draft;
    assert.equal(refreshed.subjects.length, 1);
    assert.equal(refreshed.subjects[0].referenceId, 'editor_subject');
    assert.equal(refreshed.subjects[0].path, 'C:\\references\\hero.png');
    const wizard = fixture({ engine, mode, draft: refreshed });
    await wizard.click('3 Inputs');
    assert.equal(wizard.document.body.querySelectorAll('img')[0].src, adapter.api.imageUrl('C:\\references\\hero.png'));
    await adapter.api.save(refreshed, new Map());
    assert.equal(adapter.state.fluxReferenceBuilder.subjects.length, 1);
    assert.equal(adapter.api.snapshot().draft.subjects[0].path, 'C:\\references\\hero.png');
  }
});


test('mapping editor is available only in Scenes after scenes exist', async () => {
  const f = fixture();
  await f.click('3 Inputs');
  assert.equal(f.find('button', 'Edit Mappings'), undefined);
  await f.click('6 Scenes');
  assert.equal(f.find('button', 'Edit Mappings'), undefined);
  await f.click('5 Sound & timing'); await f.click('3. Manual timing'); await f.click('6 Scenes');
  await f.click('5 Edit Mappings');
  assert.equal(f.find('button', 'Edit Mappings').disabled, false);
});

test('image-to-video reference editors respect image model capabilities for both engines', () => {
  const f = adapterFixture();
  const opened = [];
  f.context.openFluxReferenceBuilderModal = options => opened.push(options);
  for (const engine of ['ltx', 'minimax_h3']) {
    const mode = engine === 'ltx' ? 'i2v' : 'image_to_video';
    for (const imageMode of ['zimage', 'ernie_image', 'krea2_2pass', 'nano_banana', 'flux_klein', 'flow_gpt']) {
      f.api.configure({ engine, mode, imageMode });
      for (const section of ['subjects', 'locations', 'mapping']) {
        f.api.openReferences(section);
        assert.equal(opened.at(-1).focusedSection, section);
        assert.equal(opened.at(-1).textOnlyMode, ['zimage', 'ernie_image', 'krea2_2pass'].includes(imageMode));
      }
    }
    f.api.configure({ mode: engine === 'ltx' ? 'rtv' : 'reference_to_video', imageMode: 'zimage' });
    f.api.openReferences('subjects');
    assert.equal(opened.at(-1).textOnlyMode, false);
    f.api.configure({ mode: engine === 'ltx' ? 't2v' : 'text_to_video', imageMode: 'nano_banana' });
    f.api.openReferences('locations');
    assert.equal(opened.at(-1).textOnlyMode, true);
  }
});

test('Scenes step opens separate storyboard windows and refreshes story on return', async () => {
  const f = fixture();
  const opened = [];
  f.api.openStoryboard = (section, onClose) => { opened.push(section); f.data.direction = 'Updated story'; onClose(); };
  await f.click('6 Scenes');
  await f.click('6 Storyboard Scenes');
  assert.equal(f.find('button', 'Storyboard Scenes').disabled, true);
  await f.click('1 Scene Defaults'); await f.click('Scene Defaults');
  await f.click('3 Story Layer'); await f.click('Story Layer');
  await f.click('2 Story direction');
  assert.equal(f.input('Story direction').value, 'Updated story');
  await f.click('5 Sound & timing'); await f.click('3. Manual timing'); await f.click('6 Scenes');
  await f.click('6 Storyboard Scenes'); await f.click('Storyboard Scenes');
  assert.deepEqual(opened, ['defaults', 'story', 'scenes']);
  await f.click('Save Project');
  assert.equal(f.data.draft.direction, 'Updated story');
});

test('focused storyboard routes allow Image Prep only for Image to Video', () => {
  const f = adapterFixture();
  const opened = [];
  f.context.openStoryboardBuilderFromProject = options => opened.push(options);
  for (const [engine, mode, expected] of [['ltx', 'i2v', true], ['ltx', 't2v', false], ['ltx', 'rtv', false], ['minimax_h3', 'image_to_video', true], ['minimax_h3', 'reference_to_video', false]]) {
    f.api.configure({ engine, mode });
    f.api.openStoryboard('defaults');
    assert.equal(opened.at(-1).focusedSection, 'defaults');
    assert.equal(opened.at(-1).allowImagePrep, expected);
  }
});

test('focused storyboard windows mount only their own content and relevant actions', () => {
  const story = fs.readFileSync(path.join(__dirname, '../web/VRGDG_StoryboardBuilderUI.js'), 'utf8');
  const start = story.indexOf('  if (!focusedSection || focusedSection === "defaults") middleContent.append');
  const end = story.indexOf('  shell.append(header', start);
  for (const focusedSection of ['', 'defaults', 'story', 'scenes']) {
    for (const allowImagePrep of [true, false]) {
      const c = { focusedSection, allowImagePrep };
      for (const key of ['middleContent', 'sceneDefaultsPanel', 'storyLayerPanel', 'tableWrap', 'headerActions', 'footerActions', 'close', 'save', 'steps', 'header']) c[key] = new Element(key);
      c.header.append(c.steps);
      vm.runInNewContext(story.slice(start, end), c);
      assert.deepEqual(c.middleContent.children.map(x => x.tagName), focusedSection ? [{ defaults: 'sceneDefaultsPanel', story: 'storyLayerPanel', scenes: 'tableWrap' }[focusedSection]] : ['sceneDefaultsPanel', 'storyLayerPanel', 'tableWrap']);
      assert.equal(c.header.children.includes(c.steps), !focusedSection || (focusedSection !== 'story' && allowImagePrep));
      if (focusedSection && focusedSection !== 'scenes') assert.deepEqual(c.footerActions.children, [c.save]);
    }
  }
});

test('wizard global video scope reads global LTX settings despite a locked selection', () => {
  const c = { wizardGlobalVideoSettings: true, state: { i2vVideoSettings: { width: 1280 } }, scene: { use_scene_i2v_video_settings: true, i2v_video_settings: { width: 640 } } };
  c.activeSegment = () => c.scene;
  const helper = builder.slice(builder.indexOf('  function videoSettingsSegment()'), builder.indexOf('  function activeSegment()', builder.indexOf('  function videoSettingsSegment()')));
  const getter = builder.slice(builder.indexOf('  function activeI2VVideoSettings()'), builder.indexOf('  function videoVisionReferenceEnabled'));
  vm.createContext(c); vm.runInContext(helper + getter, c);
  assert.equal(c.activeI2VVideoSettings().width, 1280);
  c.wizardGlobalVideoSettings = false;
  assert.equal(c.activeI2VVideoSettings().width, 640);
});

test('global LTX save bypasses multi-selection writes and preserves scene overrides', () => {
  const start = builder.indexOf('    if (segment?.use_scene_i2v_video_settings || (!wizardGlobalVideoSettings');
  const end = builder.indexOf('    updateI2VLoraVisibility();', start);
  const c = { wizardGlobalVideoSettings: true, segment: null, state: {}, settings: { width: 1920 }, hasMultiSceneBatchSelection: () => true,
    applyVideoSettingsToMultiSelection: () => { throw Error('Must not overwrite selected scenes'); } };
  vm.runInNewContext(builder.slice(start, end), c);
  assert.equal(c.state.i2vVideoSettings.width, 1920);
});

test('MiniMax wizard pass selection works without a scene and preserves locked scene settings', async () => {
  const locked = { use_scene_minimax_h3_settings: true, minimax_h3_settings: { ref_pass_mode: 'single' } };
  const button = { dataset: { passMode: 'advanced' } };
  const c = { wizardGlobalVideoSettings: true, miniMaxPassButtons: [button], state: { miniMaxH3Settings: {} },
    requireActiveSegment: () => { throw Error('Wizard must not require a scene'); },
    pushHistory() {}, clearMiniMaxImageReferenceStartFrameOnModeSwitch() {}, cloneMiniMaxH3Settings: settings => settings,
    saveMiniMaxH3SettingsFromPanel: target => { assert.equal(target, null); return {}; },
    selectMiniMaxH3PassSettings: (settings, mode) => ({ ref_pass_mode: mode }), syncMiniMaxH3Panel() {}, autoSaveSessionQuiet: async () => {}, locked };
  const start = builder.lastIndexOf('  for (const button of miniMaxPassButtons) {');
  vm.runInNewContext(builder.slice(start, builder.indexOf('  miniMaxAudioMode.addEventListener', start)), c);
  await button.onclick();
  assert.equal(c.state.miniMaxH3Settings.ref_pass_mode, 'advanced');
  assert.equal(locked.minimax_h3_settings.ref_pass_mode, 'single');
});

test('wizard video panel mounts only global tabs and restores timeline controls on exit', () => {
  const f = adapterFixture();
  const c = f.context;
  Element.prototype.before = function(other) { other.remove(); other.parent = this.parent; this.parent.children.splice(this.parent.children.indexOf(this), 0, other); };
  Object.defineProperty(Element.prototype, 'parentNode', { get() { return this.parent; }, configurable: true });
  Element.prototype.replaceWith = function(other) { this.before(other); this.remove(); };
  const source = new Element('panels');
  const models = new Element('models'), settings = new Element('settings');
  source.append(models, settings, new Element('speakers'), new Element('prompt'));
  const original = new Element('tabs'); original.append(new Element('nav'), source);
  const passHome = new Element('passHome'); c.miniMaxPassChooser = new Element('passes'); passHome.append(c.miniMaxPassChooser);
  c.miniMaxSubTabs = { wrapper: original, setActive: value => { assert.equal(value, 'models'); } };
  c.useSceneMiniMaxH3Settings = { wrapper: new Element('sceneLock') };
  c.useSceneMiniMaxH3SettingsNote = new Element('lockNote');
  c.miniMaxModePanels = { reference: new Element('sceneMedia') };
  settings.append(c.useSceneMiniMaxH3Settings.wrapper, c.useSceneMiniMaxH3SettingsNote, c.miniMaxModePanels.reference);
  c.document = { createElement: tag => new Element(tag), createComment: () => new Element('comment') };
  c.syncMiniMaxH3Panel = () => {};
  c.makeSubTabs = tabs => { assert.deepEqual(Array.from(tabs, tab => tab.label), ['Models', 'Video Settings']); const wrapper = new Element('globalTabs'); for (const tab of tabs) wrapper.append(tab.content); return { wrapper }; };
  const holder = new Element('holder');
  const restore = f.api.mountSettings(holder, 'video');
  assert.equal(c.wizardGlobalVideoSettings, true);
  assert.equal(models.parent.tagName, 'div');
  assert.equal(source.children.includes(models), false);
  assert.equal(settings.children.includes(c.useSceneMiniMaxH3Settings.wrapper), false);
  assert.equal(source.children.some(child => child.tagName === 'prompt'), true);
  restore();
  assert.equal(c.wizardGlobalVideoSettings, false);
  assert.equal(source.children[0], models);
  assert.equal(source.children[1], settings);
  assert.equal(settings.children[0], c.useSceneMiniMaxH3Settings.wrapper);
  assert.equal(passHome.children[0], c.miniMaxPassChooser);
});

test('LTX wizard keeps FLF and ID-LoRA settings mounted and restores their panels', () => {
  for (const mode of ['flf', 'id_lora']) {
    const f = adapterFixture(), c = f.context;
    f.api.configure({ engine: 'ltx', mode });
    const source = new Element('panels');
    const models = new Element('models'), settings = new Element('settings');
    source.append(models, settings);
    const original = new Element('tabs'); original.append(new Element('nav'), source);
    c.videoSubTabs = { wrapper: original, setActive() {} };
    c.useSceneI2VVideoSettings = { wrapper: new Element('sceneLock') };
    c.useSceneI2VVideoSettingsNote = new Element('lockNote');
    c.createSceneVideoActions = new Element('sceneActions');
    c.rtvSceneImageAnchorSection = new Element('sceneAnchor');
    c.idLoraVoiceSettingsSection = new Element('identitySettings');
    c.flfGuideSettingsSection = new Element('flfSettings');
    c.createSceneVideoButtons = [];
    settings.append(c.useSceneI2VVideoSettings.wrapper, c.useSceneI2VVideoSettingsNote, c.createSceneVideoActions, c.rtvSceneImageAnchorSection, c.idLoraVoiceSettingsSection, c.flfGuideSettingsSection);
    c.document = { createElement: tag => new Element(tag), createComment: () => new Element('comment') };
    c.syncMiniMaxH3Panel = () => {};
    c.makeSubTabs = tabs => { const wrapper = new Element('globalTabs'); for (const tab of tabs) wrapper.append(tab.content); return { wrapper }; };
    const holder = new Element('holder');
    const restore = f.api.mountSettings(holder, 'video');
    assert.equal(c.wizardGlobalVideoSettings, true);
    assert.ok(holder.querySelectorAll('identitySettings').includes(c.idLoraVoiceSettingsSection));
    assert.ok(holder.querySelectorAll('flfSettings').includes(c.flfGuideSettingsSection));
    assert.equal(holder.querySelectorAll('sceneLock').length, 0);
    assert.equal(holder.querySelectorAll('sceneAnchor').length, 0);
    restore();
    assert.equal(c.wizardGlobalVideoSettings, false);
    assert.equal(source.children[1], settings);
    assert.equal(c.idLoraVoiceSettingsSection.parent, settings);
    assert.equal(c.flfGuideSettingsSection.parent, settings);
  }
});

test('reference uploads append subjects and locations, replace one, and remove one', async () => {
  const f = fixture({ mode: 'reference_to_video' });
  f.context.FileReader = class { readAsDataURL(file) { this.result = `data:${file.name}`; this.onload(); } };
  Element.prototype.click = function() {};
  const picker = index => f.document.body.querySelectorAll('input').filter(item => item.type === 'file')[index];
  const choose = async (index, names) => { const p = picker(index); p.files = names.map(name => ({ name })); await p.onchange(); };
  await f.click('3 Inputs');
  await choose(0, ['hero.png']);
  const title = f.input('Character title'); title.value = 'Hero'; title.oninput();
  await choose(0, ['friend.png']);
  await choose(1, ['city.png']); await choose(1, ['forest.png']);
  await f.click('Save Project');
  assert.equal(f.data.draft.subjects.length, 2); assert.equal(f.data.draft.locations.length, 2);
  assert.equal(f.data.draft.subjects[0].title, 'Hero');
  await f.click('Replace'); await choose(0, ['newhero.png']);
  await f.click('Save Project');
  assert.equal(f.data.draft.subjects[0].name, 'newhero.png');
  assert.equal(f.data.draft.subjects[0].title, 'Hero');
  assert.equal(f.data.draft.subjects[1].name, 'friend.png');
  await f.click('Remove'); await f.click('Save Project');
  assert.equal(f.data.draft.subjects.length, 1);
  assert.equal(f.data.draft.subjects[0].name, 'friend.png');
  assert.equal(f.data.draft.locations.length, 2);
});

test('removing a saved reference preserves surviving IDs and mappings when adding another', async () => {
  const f = adapterFixture();
  const draft = { engine: 'minimax_h3', mode: 'reference_to_video', characters: '', locationsText: '', lyrics: '', direction: '',
    subjects: [{ name: 'a.png', data: 'a' }, { name: 'b.png', data: 'b' }], locations: [{ name: 'c.png', data: 'c' }, { name: 'd.png', data: 'd' }] };
  await f.api.save(draft, new Map());
  const subjectId = draft.subjects[1].referenceId, locationId = draft.locations[1].referenceId;
  const deletedSubject = draft.subjects[0].referenceId, deletedLocation = draft.locations[0].referenceId;
  f.state.fluxReferenceBuilder.subject_scene_map.scene = [deletedSubject, subjectId];
  f.state.fluxReferenceBuilder.scene_map.scene = locationId;
  draft.removedReferenceIds = [deletedSubject, deletedLocation];
  draft.subjects.shift(); draft.locations.shift();
  draft.subjects.push({ name: 'e.png', data: 'e' }); draft.locations.push({ name: 'f.png', data: 'f' });
  await f.api.save(draft, new Map());
  const refs = f.state.fluxReferenceBuilder;
  assert.equal(refs.subjects.find(item => item.id === subjectId).image.name, 'b.png');
  assert.equal(refs.locations.find(item => item.id === locationId).image.name, 'd.png');
  assert.equal(refs.scene_map.scene, locationId);
  assert.deepEqual(Array.from(refs.subject_scene_map.scene), [subjectId]);
  assert.notEqual(draft.subjects[1].referenceId, subjectId);
  assert.notEqual(draft.locations[1].referenceId, locationId);
  assert.equal(f.api.snapshot().draft.subjects.length, 2);
});

test('timing adapter routes directly to existing editors and keeps transcribed timing and lyrics', async () => {
  const f = adapterFixture();
  f.context.audioInput.value = 'song.wav';
  f.api.configure({ audioMode: 'input_audio' });
  f.state.wizardBetaDraft = { engine: 'minimax_h3', mode: 'text_to_video', audioMode: 'input_audio', direction: 'Story', characters: '', locationsText: '', sound: '' };
  const calls = [];
  f.context.createScenesFromTimestampedLyrics = async () => { calls.push('new'); f.state.segments = [{ id: 'a', start: 1.2, end: 7.4, lyric_text: 'Exact transcribed line' }]; };
  f.context.transcribeLyricsForTimeline = async () => calls.push('existing');
  f.context.openLyricMappingWorkflowModal = options => { assert.equal(options.manualOnly, true); calls.push('manual-audio'); options.onClose(); };
  f.context.openBulkSegmentsModal = options => { assert.equal(options.initialMode, 'ranges'); calls.push('manual-ranges'); options.onClose(); };
  await f.api.openTiming('new');
  assert.equal(f.state.segments[0].start, 1.2); assert.equal(f.state.segments[0].end, 7.4);
  assert.equal(f.state.segments[0].lyric_text, 'Exact transcribed line');
  f.context.window = { confirm: () => { throw Error('No extra confirmation before the editor'); } };
  await f.api.openTiming('new');
  assert.deepEqual(calls, ['new', 'new']);
  await f.api.openTiming('existing'); await f.api.openTiming('manual');
  f.api.configure({ audioMode: 'built_in_audio' });
  await assert.rejects(f.api.openTiming('existing'), /input audio/);
  await f.api.openTiming('manual');
  assert.deepEqual(calls, ['new', 'new', 'existing', 'manual-audio', 'manual-ranges']);
});

test('Models step no longer includes LLM configuration', async () => {
  const f = fixture(); await f.click('4 Models & LoRAs');
  assert.equal(f.find('button', 'Configure prompt-writing LLM'), undefined);
});


test('unavailable timing choices explain prerequisites and use a non-busy cursor', async () => {
  const f = fixture({ audioMode: 'input_audio' });
  await f.click('5 Sound & timing');
  assert.equal(f.find('button', '2. No scenes yet').title, 'Choose an audio file above first.');
  assert.ok(f.document.body.textContent.includes('Choose an audio file above first.'));
  assert.ok(source.includes('.wb-button:disabled { opacity:.5;cursor:not-allowed; }'));
  assert.ok(source.includes('.wb-dialog[aria-busy="true"] .wb-button:disabled { cursor:wait; }'));
});


test('transcription routes open their existing options windows before doing work', async () => {
  const calls = [];
  const context = {
    showTimestampedTranscribeModal: async () => { calls.push('Create Scenes From Timestamped Lines'); return null; },
    showTranscribeLyricsModal: async () => { calls.push('Transcribe Lines For Timeline'); return null; },
  };
  for (const name of ['createScenesFromTimestampedLyrics', 'transcribeLyricsForTimeline']) {
    const start = builder.indexOf(`  async function ${name}(`);
    const cancel = builder.indexOf('if (!options) return', start);
    const end = builder.indexOf(';', cancel) + 1;
    vm.runInNewContext(builder.slice(start, end) + '\n}', context);
    await context[name]();
  }
  assert.deepEqual(calls, ['Create Scenes From Timestamped Lines', 'Transcribe Lines For Timeline']);
});

test('both transcription windows prefill the saved wizard lyrics without changing line breaks', () => {
  for (const name of ['showTranscribeLyricsModal', 'showTimestampedTranscribeModal']) {
    const start = builder.indexOf(`  function ${name}(`);
    const inputStart = builder.indexOf('      const lyrics = document.createElement("textarea");', start);
    const end = builder.indexOf('      const language =', inputStart);
    for (const sourceText of ['Verse one\nVerse two\n\n[Instrumental]\nChorus', '']) {
      const c = { state: { lyricMapper: { source_text: sourceText } }, document: { createElement: tag => new Element(tag) } };
      vm.runInNewContext(builder.slice(inputStart, end) + '\nthis.lyrics = lyrics;', c);
      assert.equal(c.lyrics.value, sourceText);
      c.lyrics.value = 'Edited in transcription window';
      assert.equal(c.state.lyricMapper.source_text, sourceText);
    }
  }
});

test('wizard saves newly entered lyrics before opening the transcription editor', async () => {
  const f = fixture({ audioMode: 'input_audio', audioPath: 'song.wav' });
  await f.click('5 Sound & timing');
  const input = f.input('Lyrics or dialogue (optional)');
  input.value = 'New first line\nNew second line'; input.oninput();
  f.api.openTiming = async kind => {
    assert.equal(kind, 'new');
    assert.equal(f.data.draft.lyrics, 'New first line\nNew second line');
  };
  await f.click('2. No scenes yet');
});

test('manual timing Space and Down Arrow split without triggering playback or repeated splits', () => {
  const start = builder.indexOf('    const onLyricMappingKeydown = (event) => {');
  const end = builder.indexOf('    const setActiveTab', start);
  const c = { activeLyricMappingTab: 'manual_timing', backdrop: { contains: target => !target.outside }, pane: { querySelector: () => ({ click: () => c.splits++ }) }, splits: 0 };
  vm.runInNewContext(builder.slice(start, end) + '\nthis.handle = onLyricMappingKeydown;', c);
  const event = extra => ({ key: ' ', code: 'Space', target: { tagName: 'AUDIO' }, preventDefault() { this.prevented = true; }, stopImmediatePropagation() { this.stopped = true; }, ...extra });
  const space = event(); c.handle(space);
  assert.equal(c.splits, 1); assert.equal(space.prevented, true); assert.equal(space.stopped, true);
  const repeated = event({ repeat: true }); c.handle(repeated);
  assert.equal(c.splits, 1); assert.equal(repeated.prevented, true);
  c.handle(event({ key: 'ArrowDown', code: 'ArrowDown' })); assert.equal(c.splits, 2);
  for (const target of [{ tagName: 'INPUT' }, { tagName: 'TEXTAREA' }, { tagName: 'SELECT' }, { isContentEditable: true }, { outside: true }]) {
    const e = event({ target }); c.handle(e); assert.equal(e.prevented, undefined);
  }
  c.handle(event({ ctrlKey: true })); assert.equal(c.splits, 2);
  c.activeLyricMappingTab = 'transcribe'; c.handle(event()); assert.equal(c.splits, 2);
  assert.ok(builder.includes('window.removeEventListener("keydown", onLyricMappingKeydown, true)'));
});

test('manual timing custom playback returns keyboard focus to the split pane', async () => {
  const start = builder.indexOf('        const playback = document.createElement("div");', builder.indexOf('  function openLyricMappingWorkflowModal'));
  const end = builder.indexOf('        const controls =', start);
  let focused = 0;
  const c = { pane: { focus: () => focused++ }, document: { createElement: tag => new Element(tag) },
    makeButton: label => { const b = new Element('button'); b.textContent = label; return b; },
    manualTimingAudio: new Element('audio'), formatTime: String, toast: message => { throw Error(message); },
    audioPanel: new Element('div'), addSplit: new Element('button'), undoSplit: new Element('button') };
  Object.assign(c.manualTimingAudio, { paused: true, currentTime: 3, duration: 177,
    async play() { this.paused = false; }, pause() { this.paused = true; } });
  vm.runInNewContext(builder.slice(start, end) + '\nthis.playPause = playPause; this.seek = seek;', c);
  await c.playPause.onclick(); assert.equal(c.manualTimingAudio.paused, false); assert.equal(focused, 1);
  await c.playPause.onclick(); assert.equal(c.manualTimingAudio.paused, true); assert.equal(focused, 2);
  c.seek.value = '12.5'; c.seek.oninput(); c.seek.onchange();
  assert.equal(c.manualTimingAudio.currentTime, 12.5); assert.equal(focused, 3);
  assert.ok(builder.includes('manualTimingAudio.controls = false;'));
});


test('Scenes guides users through ordered substeps and keeps runner available in the header', async () => {
  const f = fixture({ scenes: [{ id: 'a', start: 0, end: 8 }] });
  let runnerOpened = 0; f.api.openRunner = () => runnerOpened++;
  await f.click('6 Scenes');
  assert.ok(f.find('button', 'Scene Defaults'));
  assert.equal(f.input('Story direction'), undefined);
  assert.equal(f.find('button', 'Generate video prompts'), undefined);
  await f.click('Next →'); assert.ok(f.input('Story direction'));
  await f.click('Next →'); assert.ok(f.find('button', 'Story Layer'));
  await f.click('Configure LLM Runner'); assert.equal(runnerOpened, 1);
  await f.click('Next →'); assert.ok(f.find('button', 'Align lyrics / dialogue'));
  assert.ok(f.find('h3', '4. Align lyrics / dialogue (Optional)'));
  await f.click('Skip / Next →'); assert.ok(f.find('button', 'Edit Mappings'));
  await f.click('Next →'); assert.ok(f.find('button', 'Storyboard Scenes'));
  assert.equal(f.input('Video prompt'), undefined);
  await f.click('Next →'); assert.ok(f.find('h2', '7. Render'));
});


test('wizard Render All routes resume and redo directly to rendering for both engines', async () => {
  for (const engine of ['minimax_h3', 'ltx']) {
    for (const action of ['resume_missing', 'redo_videos', null]) {
      const f = adapterFixture();
      f.state.projectVideoEngine = engine;
      const calls = [];
      f.context.chooseBatchModeAction = async options => {
        assert.equal(options.title, 'Render All?');
        assert.deepEqual(Array.from(options.choices, choice => choice.value), ['resume_missing', 'redo_videos']);
        return action;
      };
      f.context.renderAllScenes = async options => calls.push(options);
      await f.api.render();
      assert.equal(calls.length, action ? 1 : 0);
      if (action) { assert.equal(calls[0].sceneScope, 'all'); assert.equal(calls[0].forceVideos, action === 'redo_videos'); }
      assert.deepEqual(f.events, []);
    }
  }
});

test('final wizard step offers Render All and saves before rendering', async () => {
  const f = fixture({ scenes: [{ id: 'a', start: 0, end: 8 }] });
  await f.click('7 Render');
  assert.equal(f.find('button', 'Build Full Video'), undefined);
  await f.click('Render All');
  assert.deepEqual(f.events, ['save', 'render']);
  const empty = fixture(); await empty.click('7 Render');
  assert.equal(empty.find('button', 'Render All').disabled, true);
});


test('own images are assigned after scenes exist, via storyboard or the folder importer', async () => {
  const f = fixture({ engine: 'ltx', mode: 'i2v', draft: { imageSource: 'upload', startImage: { name: 'old.png', data: 'old' } } });
  await f.click('3 Inputs');
  assert.equal(f.find('strong', 'Starting image'), undefined);
  assert.ok(f.document.body.textContent.includes('After creating scenes'));
  await f.click('6 Scenes'); await f.click('6 Storyboard Scenes');
  assert.equal(f.find('button', 'Fill Timeline Images From Folder').disabled, true);
  f.data.scenes = [{ id: 'a', start: 0, end: 8 }];
  await f.click('6 Storyboard Scenes');
  assert.equal(f.find('button', 'Fill Timeline Images From Folder').disabled, false);
  assert.equal(f.find('button', 'Generate missing scene images'), undefined);
  const files = [{ name: '1.png' }, { name: '2.png' }];
  let imported;
  f.api.importImages = async values => { imported = values; };
  const folder = f.document.body.querySelectorAll('input').find(input => input.type === 'file');
  assert.equal(folder.attributes.webkitdirectory, '');
  folder.files = files; await folder.onchange();
  assert.deepEqual(Array.from(imported), files);
  assert.equal(f.events.at(-1), 'save');
});
