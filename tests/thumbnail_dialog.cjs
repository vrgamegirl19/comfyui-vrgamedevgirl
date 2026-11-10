const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule, readBuilderSource } = require('./builder_source.cjs');

// The Thumbnail dialog (web/music_video_builder/thumbnail.mjs) driven against a small fake DOM: scrubbing to a moment, taking that exact
// frame, writing the prompt from it and making the thumbnail. What the browser draws is not checked here, only what the dialog does.

function fakeElement(tag, all) {
  const node = {
    tag, style: {}, dataset: {}, attrs: {}, children: [], listeners: {}, value: '', textContent: '', disabled: false, title: '',
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = kids; },
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    setAttribute(key, value) { this.attrs[key] = value; },
    remove() { this.removed = true; },
    pause() {},
  };
  all.push(node);
  return node;
}

const API_IMAGE_CAPABILITY = {
  providers: {
    gemini: { label: 'Gemini', models: ['gemini-3-pro-image-preview', 'gemini-3.1-flash-image-preview'], default: 'gemini-3-pro-image-preview' },
    grok: { label: 'Grok', models: ['grok-imagine-image-2.0', 'grok-imagine-image'], default: 'grok-imagine-image-2.0' },
    openai: { label: 'OpenAI', models: ['gpt-image-2', 'gpt-image-1'], default: 'gpt-image-2' },
  },
};

function harness({
  segments, frameAvailable = true, postJson, engineSettings = {}, overlayAvailable = false, apiImage = null, runner = { text_runner: 'builtin' }, intro = {},
  videoIntroAvailable = true, browserAi = false, browserAiDirect = false, browserSettings = { provider: 'flow_nano_banana', failure_mode: 'last_successful_image' },
  images = async () => [{ filename: 'gen.png', subfolder: '', type: 'output' }],
} = {}) {
  const all = [];
  const calls = [];
  const toasts = [];
  const saves = [];
  const queued = [];
  const control = (tag) => fakeElement(tag, all);
  const context = vm.createContext({
    document: { createElement: control, body: { append(...kids) { all.body = kids; } } },
    window: { confirm: () => true },
    api: {},
    BUILDER_FONT_STACK: 'sans-serif',
    DEFAULT_NB_IMAGE_MODEL: 'nb-model',
    makeEditorImageUrl: (path) => `image:${path}`,
    makeEditorVideoUrl: (path, bust) => `video:${path}?${bust ?? ''}`,
    toast: (message, isError) => toasts.push([message, Boolean(isError)]),
    makeButton: (label) => Object.assign(control('button'), { textContent: label }),
    makeInput: (value = '', type = 'text') => Object.assign(control('input'), { value, type }),
    makeSelect: (options, value) => Object.assign(control('select'), { value, options: options.map((option) => option.value) }),
    makeCheckbox: (label, checked) => ({ wrapper: control('label'), input: { checked } }),
    makeField: (label, input) => Object.assign(control('label'), { label, input }),
    postJson: async (url, payload) => {
      calls.push([url, payload]);
      return postJson(url, payload);
    },
    queueWorkflowPrompt: async (prompt) => {
      queued.push(prompt);
      return { prompt_id: 'p1' };
    },
    waitForImages: async (...args) => images(...args),
    BROWSER_IMAGE_PROVIDERS: { FLOW_NANO_BANANA: 'flow_nano_banana', GPT_IMAGE: 'gpt_image', META_AI: 'meta_ai' },
  });
  // The real Browser AI helpers: the provider order after a failure, and the label and timeout of each provider.
  vm.runInContext(readBuilderModule('browser_image_fallback.mjs'), context);
  const builderSource = readBuilderSource();
  for (const name of ['normalizeFlowGptBrowserProvider', 'browserImageProviderLabel', 'browserImageProviderTimeout']) {
    vm.runInContext(functionSource(builderSource, name), context);
  }
  vm.runInContext(readBuilderModule('thumbnail_state.mjs'), context);
  vm.runInContext(readBuilderModule('thumbnail_engines.mjs'), context);
  vm.runInContext(readBuilderModule('thumbnail.mjs'), context);
  const state = {
    segments, projectFolder: 'C:/p', thumbnailAvailable: true, thumbnailFrameAvailable: frameAvailable, thumbnailOverlayAvailable: overlayAvailable,
    thumbnailApiImage: apiImage, thumbnailIntro: context.normalizeThumbnailIntro(intro), nbImageSettings: { api_key: 'key', model: 'nb-model' },
    videoIntroAvailable, thumbnailBrowserAi: browserAi, thumbnailBrowserAiDirect: browserAiDirect, flowGptBrowserSettings: browserSettings,
  };
  const deps = {
    state, projectInput: { value: 'C:/p' }, runnerPayload: () => runner,
    engineSettings: (engine) => ({ nano_banana: state.nbImageSettings, browser_ai: state.flowGptBrowserSettings, ...engineSettings })[engine],
    sceneImagePath: (segment) => segment.image || '', sceneVideoPath: (segment) => segment.video || '',
    sceneLabel: (segment, index) => `Scene ${index + 1}`, save: async () => { saves.push(JSON.stringify(state.thumbnailIntro.source_path)); },
    createProgressWindow: () => ({ set() {}, close() {} }),
  };
  const open = () => context.openThumbnailDialog(deps);
  const find = (predicate) => all.find(predicate);
  const button = (label) => find((node) => node.tag === 'button' && node.textContent === label);
  return { context, state, calls, toasts, saves, queued, open, find, button, all };
}

const scenes = () => [
  { id: 'a', start: 0, end: 2, video: 'C:/v/a.mp4', video_cache_bust: 7, lyric_text: 'first line' },
  { id: 'b', start: 2, end: 4, video: 'C:/v/b.mp4', image: 'C:/i/b.png', lyric_text: 'second line' },
];

test('scrubbing shows the scene that has that moment, parked on that frame', () => {
  const h = harness({ segments: scenes() });
  h.open();
  const video = h.find((node) => node.tag === 'video');
  const scrub = h.find((node) => node.type === 'range');
  assert.equal(scrub.min, '0');
  assert.equal(scrub.max, '95');
  assert.equal(video.src, 'video:C:/v/a.mp4?7', 'it opens on the first frame');
  scrub.value = '60';
  scrub.listeners.input[0]();
  assert.equal(video.src, 'video:C:/v/b.mp4?', 'frame 60 is in scene b');
  // The listener of the first scene is stale by now: only the newest one may move the player.
  video.listeners.loadedmetadata[0]();
  assert.equal(video.currentTime, undefined, 'a load for a scene that is no longer shown does not seek');
  video.listeners.loadedmetadata.at(-1)();
  assert.equal(video.currentTime, (12 + 0.25) / 24, 'parked a quarter frame into frame 12 of the scene video');
  const readout = h.all.find((node) => /frame 60/.test(node.textContent));
  assert.match(readout.textContent, /Scene 2 {2}· {2}0:02\.50 {2}· {2}frame 60 {2}\(video\)/);
});

test('the step buttons move one frame or one second, and the scene buttons jump to a scene start', () => {
  const h = harness({ segments: scenes() });
  h.open();
  const scrub = h.find((node) => node.type === 'range');
  const press = (label) => h.button(label).onclick();
  press('+ 1 frame');
  assert.equal(scrub.value, '1');
  press('+ 1 s');
  assert.equal(scrub.value, '25');
  press('\u2212 1 frame');
  assert.equal(scrub.value, '24');
  press('Scene \u25B6');
  assert.equal(scrub.value, '48', 'the next scene starts at 2 s');
  press('+ 1 frame');
  press('\u25C0 Scene');
  assert.equal(scrub.value, '48', 'from inside a scene it goes back to that scene\'s start');
  press('\u25C0 Scene');
  assert.equal(scrub.value, '0', 'from a scene start it goes to the one before');
  press('\u2212 1 s');
  assert.equal(scrub.value, '0', 'it stops at the first frame');
});

test('Use this frame takes exactly that frame of that scene video and remembers where it was', async () => {
  const h = harness({ segments: scenes(), postJson: async () => ({ image_path: 'C:/p/thumbnail/source_0001.png' }) });
  h.open();
  const scrub = h.find((node) => node.type === 'range');
  scrub.value = '60';
  scrub.listeners.input[0]();
  await h.button('Use this frame').onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(h.calls[0])), ['/vrgdg/music_builder/thumbnail/frame', { project_folder: 'C:/p', video_path: 'C:/v/b.mp4', seconds: 0.5 }]);
  const intro = h.state.thumbnailIntro;
  assert.equal(intro.source_path, 'C:/p/thumbnail/source_0001.png');
  assert.equal(intro.source_kind, 'frame');
  assert.equal(intro.source_frame, 60);
  assert.equal(intro.source_seconds, 2.5);
  assert.equal(intro.scene_id, 'b');
  assert.equal(h.saves.length, 1, 'the choice is saved with the project');
  const caption = h.all.find((node) => /video frame/.test(node.textContent));
  assert.match(caption.textContent, /Scene 2 {2}· {2}0:02\.50 {2}· {2}frame 60 {2}\(video frame\)/);
});

test('a scene with no video uses its image, and a build without the frame route only offers images', async () => {
  const noVideo = [{ id: 'a', start: 0, end: 2, image: 'C:/i/a.png' }];
  const h = harness({ segments: noVideo });
  h.open();
  await h.button('Use this frame').onclick();
  assert.equal(h.calls.length, 0, 'nothing to extract');
  assert.equal(h.state.thumbnailIntro.source_path, 'C:/i/a.png');
  assert.equal(h.state.thumbnailIntro.source_kind, 'image');

  const old = harness({ segments: scenes(), frameAvailable: false });
  old.open();
  const video = old.find((node) => node.tag === 'video');
  assert.equal(video.src, undefined, 'no video is loaded');
  assert.ok(old.find((node) => node.tag === 'img' && node.src === 'image:C:/i/b.png'), 'scene b shows its image');
  const scrub = old.find((node) => node.type === 'range');
  assert.equal(scrub.min, '48', 'scene a has neither a usable video nor an image, so the picker starts at scene b');
});

test('the LLM prompt and the thumbnail start from the chosen frame, not from a scene image', async () => {
  const replies = {
    '/vrgdg/music_builder/thumbnail/frame': { image_path: 'C:/p/thumbnail/source_0001.png' },
    '/vrgdg/music_builder/thumbnail/prompt': { prompt: 'Recrop the attached image.', characters: 25 },
    '/vrgdg/workflow_runner/build_nb_image_prompt': { prompt: {} },
    '/vrgdg/music_builder/thumbnail/save': { image_path: 'C:/p/thumbnail/thumbnail_0001.png' },
  };
  const h = harness({ segments: scenes(), postJson: async (url) => replies[url] });
  h.open();
  const scrub = h.find((node) => node.type === 'range');
  scrub.value = '60';
  scrub.listeners.input[0]();
  await h.button('Use this frame').onclick();
  const titleInput = h.all.find((node) => node.tag === 'input' && node.maxLength === 120);
  titleInput.value = 'RUN';
  await h.button('Write prompt with LLM').onclick();
  const prompt = h.calls.find(([url]) => url.endsWith('/thumbnail/prompt'))[1];
  assert.equal(prompt.image_path, 'C:/p/thumbnail/source_0001.png');
  assert.equal(prompt.title, 'RUN');
  assert.equal(prompt.scene_lyrics, 'second line', 'the lyrics are the chosen scene\'s');
  assert.equal(prompt.text_runner, 'builtin');
  // The reply did not have the title, so the mandatory sentence was added before the prompt reached the box.
  const withTitle = `Recrop the attached image. ${h.context.thumbnailTitleClause('RUN', 'edit')}`;
  assert.equal(h.state.thumbnailIntro.prompt, withTitle);
  await h.button('Generate thumbnail').onclick();
  const built = h.calls.find(([url]) => url.endsWith('/build_nb_image_prompt'))[1];
  assert.deepEqual(JSON.parse(JSON.stringify(built.image_ingredients)), [{ path: 'C:/p/thumbnail/source_0001.png' }]);
  assert.equal(built.prompt, withTitle);
  const intro = h.state.thumbnailIntro;
  assert.equal(intro.image_path, 'C:/p/thumbnail/thumbnail_0001.png');
  assert.equal(intro.history.length, 1);
  assert.equal(intro.history[0].scene_id, 'b');
});

test('writing the prompt or generating before a frame is chosen asks for the frame and calls nothing', async () => {
  const h = harness({ segments: scenes() });
  h.open();
  await h.button('Write prompt with LLM').onclick();
  await h.button('Generate thumbnail').onclick();
  assert.equal(h.calls.length, 0);
  assert.equal(h.toasts.length, 2);
  assert.match(h.toasts[0][0], /Choose the frame first/);
  assert.equal(h.toasts[0][1], true);
});

test('with no scene to take a frame from the dialog still opens, for the person\'s own picture and intro video', async () => {
  for (const segments of [[{ id: 'a', start: 0, end: 2 }], []]) {
    const h = harness({ segments, postJson: async () => ({ image_path: 'C:/p/thumbnail/thumbnail_0001.png', width: 1920, height: 1080 }) });
    assert.ok(h.open(), 'it opens');
    assert.equal(h.toasts.length, 0, 'nothing is refused');
    assert.equal(h.find((node) => node.tag === 'video' && node.src), undefined, 'no scene video is loaded');
    const scrub = h.find((node) => node.type === 'range');
    assert.equal(scrub.min, '0');
    assert.equal(scrub.max, '0');
    // The frame buttons do nothing without a scene.
    await h.button('Use this frame').onclick();
    assert.equal(h.calls.length, 0);
    // The scrubber and the generating controls are hidden: what is left is the picture, the history and the person's own files.
    const parentOf = (child) => h.all.find((node) => node.children.includes(child));
    assert.equal(parentOf(parentOf(scrub)).style.display, 'none', 'the picker');
    const left = parentOf(parentOf(h.button('Generate thumbnail')));
    assert.equal(left.style.display, 'none', 'the generating controls');
    assert.equal(parentOf(left).style.gridTemplateColumns, 'minmax(0,1fr)', 'the thumbnail preview takes the whole row');
    // The own picture still works with no scene at all.
    const input = h.find((node) => node.tag === 'input' && /thumbnail picture/.test(node.placeholder || ''));
    input.value = 'C:/mine/cover.png';
    await h.button('Use this picture').onclick();
    assert.equal(h.state.thumbnailIntro.image_path, 'C:/p/thumbnail/thumbnail_0001.png');
  }
});

test('reopening starts on the frame that was chosen', () => {
  const h = harness({ segments: scenes() });
  h.state.thumbnailIntro = h.context.normalizeThumbnailIntro({
    source_path: 'C:/p/thumbnail/source_0003.png', source_kind: 'frame', source_frame: 70, source_seconds: 70 / 24, scene_id: 'b',
  });
  h.open();
  const scrub = h.find((node) => node.type === 'range');
  assert.equal(scrub.value, '70');
  const shown = h.all.find((node) => node.tag === 'img' && node.src === 'image:C:/p/thumbnail/source_0003.png');
  assert.ok(shown, 'the frame in use is shown');
});

const frameReplies = {
  '/vrgdg/music_builder/thumbnail/frame': { image_path: 'C:/p/thumbnail/source_0001.png' },
  '/vrgdg/music_builder/thumbnail/prompt': { prompt: 'A close-up of a singer.', characters: 23 },
  '/vrgdg/music_builder/thumbnail/save': { image_path: 'C:/p/thumbnail/thumbnail_0001.png' },
};

async function pickFrame(h) {
  h.open();
  const scrub = h.find((node) => node.type === 'range');
  scrub.value = '60';
  scrub.listeners.input[0]();
  await h.button('Use this frame').onclick();
  return { select: h.find((node) => node.tag === 'select') };
}

test('only Overlay, Qwen Image Edit, NanoBanana, the API image and Browser AI (Flow, GPT Image, Meta AI) are offered, and only where the build has them', async () => {
  const bare = harness({ segments: scenes() });
  bare.open();
  assert.deepEqual(Array.from(bare.find((node) => node.tag === 'select').options), ['qwen_image_edit', 'nano_banana']);
  const full = harness({ segments: scenes(), overlayAvailable: true, apiImage: API_IMAGE_CAPABILITY });
  full.open();
  assert.deepEqual(Array.from(full.find((node) => node.tag === 'select').options), ['overlay', 'qwen_image_edit', 'nano_banana', 'api_image']);
  const withBrowser = harness({ segments: scenes(), overlayAvailable: true, apiImage: API_IMAGE_CAPABILITY, browserAi: true });
  withBrowser.open();
  assert.deepEqual(Array.from(withBrowser.find((node) => node.tag === 'select').options), ['overlay', 'qwen_image_edit', 'nano_banana', 'api_image', 'browser_flow', 'browser_gpt', 'browser_meta']);
  // A project saved with a Browser AI engine opens on it where the build has it, and on NanoBanana where it does not (the app, when its ComfyUI has no Browser AI node).
  for (const engine of ['browser_flow', 'browser_gpt', 'browser_meta']) {
    const saved = harness({ segments: scenes(), browserAi: true, intro: { engine } });
    saved.open();
    assert.equal(saved.find((node) => node.tag === 'select').value, engine);
    const without = harness({ segments: scenes(), intro: { engine } });
    without.open();
    assert.equal(without.find((node) => node.tag === 'select').value, 'nano_banana', engine);
  }
  // A project saved when Browser AI was one engine with a provider setting opens on the engine of that site.
  const legacy = harness({ segments: scenes(), browserAi: true, intro: { engine: 'browser_ai', browser_ai: { provider: 'meta_ai' } } });
  legacy.open();
  assert.equal(legacy.find((node) => node.tag === 'select').value, 'browser_meta');
  const legacyBare = harness({ segments: scenes(), browserAi: true, intro: { engine: 'browser_ai' } });
  legacyBare.open();
  assert.equal(legacyBare.find((node) => node.tag === 'select').value, 'browser_flow');
  // A project saved with a workflow that was taken out of the list opens on NanoBanana.
  for (const gone of ['flux_klein', 'moss_image_edit', 'krea2_2pass', 'zimage', 'ernie_image', 'moss_image']) {
    const old = harness({ segments: scenes(), overlayAvailable: true, apiImage: API_IMAGE_CAPABILITY, intro: { engine: gone } });
    old.open();
    assert.equal(old.find((node) => node.tag === 'select').value, 'nano_banana', gone);
  }
});

test('NanoBanana is given the editing prompt and the Builder\'s own key and model', async () => {
  const replies = { ...frameReplies, '/vrgdg/workflow_runner/build_nb_image_prompt': { prompt: {} } };
  const h = harness({ segments: scenes(), postJson: async (url) => replies[url] });
  await pickFrame(h);
  await h.button('Write prompt with LLM').onclick();
  assert.equal(h.calls.find(([url]) => url.endsWith('/thumbnail/prompt'))[1].engine_kind, 'edit');
  await h.button('Generate thumbnail').onclick();
  const [, built] = h.calls.find(([url]) => url.endsWith('/build_nb_image_prompt'));
  assert.equal(built.api_key, 'key');
  assert.equal(built.model, 'nb-model');
  assert.equal(h.state.thumbnailIntro.engine, 'nano_banana');
  assert.equal(h.state.thumbnailIntro.history[0].engine, 'nano_banana');
});

test('there is no Change control, and switching to a workflow that wants another kind of prompt flags the prompt', async () => {
  const h = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, postJson: async (url) => frameReplies[url] });
  const { select } = await pickFrame(h);
  assert.equal(h.all.find((node) => node.attrs && node.attrs['aria-label'] === 'How much to change the frame'), undefined);
  await h.button('Write prompt with LLM').onclick();
  select.value = 'api_image';
  select.listeners.change[0]();
  assert.equal(h.all.find((node) => /different kind of prompt/.test(node.textContent)), undefined, 'NanoBanana and the API image share the editing prompt');
  select.value = 'qwen_image_edit';
  select.listeners.change[0]();
  assert.ok(h.all.find((node) => /different kind of prompt/.test(node.textContent)), 'Qwen Image Edit wants a prompt about the change only');
});

test('the thumbnail is the first frame, so there is no length to set and the stitch sends none', async () => {
  const h = harness({ segments: scenes(), postJson: async () => ({ intro_frames: 1 }) });
  h.open();
  assert.equal(h.all.find((node) => /Seconds shown/.test(node.textContent)), undefined);
  assert.equal(h.all.find((node) => node.tag === 'input' && node.type === 'number'), undefined, 'no number box for seconds');
  const box = h.all.find((node) => node.tag === 'label');
  assert.ok(h.context.normalizeThumbnailIntro({ seconds: 5 }).seconds === undefined, 'an old saved length is dropped');
  h.state.thumbnailIntro = h.context.normalizeThumbnailIntro({ enabled: true, image_path: 'C:/p/thumbnail/thumbnail_0001.png', seconds: 5 });
  h.state.thumbnailAvailable = true;
  const stitched = await h.context.applyThumbnailIntroAfterStitch(h.state, { final_video_path: 'C:/p/FINAL_VIDEO.mp4' }, { projectFolder: 'C:/p' });
  const [url, sent] = h.calls.at(-1);
  assert.equal(url, '/vrgdg/music_builder/thumbnail/prepend');
  assert.deepEqual(Object.keys(sent).sort(), ['image_path', 'input_path', 'project_folder', 'replace_source']);
  assert.equal(stitched.thumbnail_intro.intro_frames, 1);
  void box;
});

test('the title is mandatory: it is in the prompt the LLM wrote, it comes back when the prompt loses it, and nothing is added without a title', async () => {
  const replies = { ...frameReplies, '/vrgdg/workflow_runner/build_nb_image_prompt': { prompt: {} } };
  const h = harness({ segments: scenes(), postJson: async (url) => replies[url] });
  await pickFrame(h);
  const titleInput = h.all.find((node) => node.tag === 'input' && node.maxLength === 120);
  titleInput.value = 'HEAT';
  const clause = h.context.thumbnailTitleClause('HEAT', 'edit');
  // The reply ('A close-up of a singer.') has no title, so the box already has the sentence when it is shown.
  await h.button('Write prompt with LLM').onclick();
  assert.equal(h.state.thumbnailIntro.prompt, `A close-up of a singer. ${clause}`);
  const promptBox = h.all.find((node) => node.tag === 'textarea' && String(node.value).startsWith('A close-up of a singer.'));
  assert.ok(promptBox, 'the prompt box shows it');
  assert.equal(h.toasts.length, 0, 'nothing is announced for a prompt that was written with it');

  // The prompt is edited and the title is lost: Generate puts it back, shows it and says so.
  promptBox.value = 'Make it pop.';
  await h.button('Generate thumbnail').onclick();
  const built = h.calls.find(([url]) => url.endsWith('/build_nb_image_prompt'))[1];
  assert.equal(built.prompt, `Make it pop. ${clause}`, 'the request carries the title');
  assert.equal(promptBox.value, `Make it pop. ${clause}`, 'the box shows what was sent');
  assert.equal(h.state.thumbnailIntro.history[0].prompt, `Make it pop. ${clause}`);
  assert.deepEqual(h.toasts.at(-1), ['The title text was missing from the prompt, so it was added.', false]);

  // The spelling typed in the title field wins over how the LLM wrote it.
  h.calls.length = 0;
  promptBox.value = 'The word \u201CHeat\u201D in huge letters.';
  await h.button('Generate thumbnail').onclick();
  assert.equal(h.calls.find(([url]) => url.endsWith('/build_nb_image_prompt'))[1].prompt, 'The word "HEAT" in huge letters.');

  // No title: the prompt goes as it is.
  titleInput.value = '';
  h.calls.length = 0;
  const toasts = h.toasts.length;
  promptBox.value = 'Make it pop.';
  await h.button('Generate thumbnail').onclick();
  assert.equal(h.calls.find(([url]) => url.endsWith('/build_nb_image_prompt'))[1].prompt, 'Make it pop.');
  assert.equal(h.toasts.length, toasts);
});

// -- Overlay: the frame kept exactly, with the grade and the title put on top ---------------------------------------------------------

const overlayReplies = {
  '/vrgdg/music_builder/thumbnail/frame': { image_path: 'C:/p/thumbnail/source_0001.png' },
  '/vrgdg/music_builder/thumbnail/overlay': { image_path: 'C:/p/thumbnail/thumbnail_0001.png', width: 1280, height: 736 },
};

function overlayControls(h) {
  const selects = h.all.filter((node) => node.tag === 'select');
  return {
    engine: selects[0], color: selects[1], position: selects[2],
    strength: h.all.find((node) => node.attrs && node.attrs['aria-label'] === 'How strong the grade, the glow and the vignette are'),
  };
}

test('Overlay is only offered by a build that has it, and a project saved with it opens on the first engine that is offered', async () => {
  const without = harness({ segments: scenes(), intro: { engine: 'overlay' } });
  without.open();
  const engine = overlayControls(without).engine;
  assert.equal(engine.value, 'nano_banana', 'saved with Overlay, opened where it is missing');
  const offered = without.all.filter((node) => node.tag === 'select')[0];
  assert.ok(offered, 'the engine list is there');
  const withIt = harness({ segments: scenes(), overlayAvailable: true, intro: { engine: 'overlay' } });
  withIt.open();
  assert.equal(overlayControls(withIt).engine.value, 'overlay', 'saved with Overlay, opened where it is offered');
});

test('choosing Overlay hides the prompt and the LLM button, and says nothing is redrawn', async () => {
  const h = harness({ segments: scenes(), overlayAvailable: true, postJson: async (url) => overlayReplies[url] });
  await pickFrame(h);
  const { engine } = overlayControls(h);
  const hint = h.all.find((node) => /edits the frame by instruction|keeps every pixel/.test(node.textContent));
  assert.ok(hint);
  engine.value = 'overlay';
  engine.listeners.change[0]();
  assert.match(hint.textContent, /keeps every pixel of your frame/);
  assert.match(hint.textContent, /No AI redraws anything, so the person and the scene cannot change/);
  assert.equal(h.button('Write prompt with LLM').style.display, 'none');
  engine.value = 'nano_banana';
  engine.listeners.change[0]();
  assert.equal(h.button('Write prompt with LLM').style.display, '', 'the LLM button is back for an image workflow');
  assert.match(hint.textContent, /redraws the frame, so the person, the face and the scene can change: use Overlay to keep them exactly/);
});

test('Generate with Overlay asks the Builder for the overlay and nothing else: no prompt, no LLM, no ComfyUI, no separate save', async () => {
  const h = harness({ segments: scenes(), overlayAvailable: true, postJson: async (url) => overlayReplies[url] });
  await pickFrame(h);
  const { engine, color, position, strength } = overlayControls(h);
  const titleInput = h.all.find((node) => node.tag === 'input' && node.maxLength === 120);
  titleInput.value = 'HEAT';
  engine.value = 'overlay';
  engine.listeners.change[0]();
  color.value = 'cyan';
  position.value = 'top';
  strength.value = '80';
  await h.button('Generate thumbnail').onclick();
  const urls = h.calls.map(([url]) => url);
  assert.deepEqual(urls, ['/vrgdg/music_builder/thumbnail/frame', '/vrgdg/music_builder/thumbnail/overlay'], 'only the frame and the overlay');
  const [, sent] = h.calls.at(-1);
  assert.deepEqual(JSON.parse(JSON.stringify(sent)), {
    project_folder: 'C:/p', source_path: 'C:/p/thumbnail/source_0001.png', title: 'HEAT', color: 'cyan', position: 'top', strength: 80,
  });
  const intro = h.state.thumbnailIntro;
  assert.equal(intro.image_path, 'C:/p/thumbnail/thumbnail_0001.png', 'the Builder saved it, so that is the picture');
  assert.equal(intro.engine, 'overlay');
  assert.deepEqual(JSON.parse(JSON.stringify(intro.overlay)), { color: 'cyan', position: 'top', strength: 80 }, 'the look is remembered');
  assert.equal(intro.history.length, 1);
  assert.equal(intro.history[0].engine, 'overlay');
  assert.equal(intro.history[0].prompt, '');
  assert.equal(h.toasts.length, 0);
  assert.equal(h.state.thumbnailIntro.prompt, '', 'no prompt was written or required');
});

test('Overlay without a title still makes the picture, and says there is no text', async () => {
  const h = harness({ segments: scenes(), overlayAvailable: true, postJson: async (url) => overlayReplies[url] });
  await pickFrame(h);
  const { engine } = overlayControls(h);
  engine.value = 'overlay';
  engine.listeners.change[0]();
  await h.button('Generate thumbnail').onclick();
  assert.equal(h.calls.at(-1)[1].title, '');
  const status = h.all.find((node) => /no title text because the title is empty/.test(node.textContent));
  assert.ok(status);
});

test('Overlay reopens with the color, position and strength that were chosen, and strength 0 is kept', () => {
  const h = harness({ segments: scenes(), overlayAvailable: true, intro: { engine: 'overlay', overlay: { color: 'red', position: 'bottom', strength: 0 } } });
  h.open();
  const { color, position, strength } = overlayControls(h);
  assert.equal(color.value, 'red');
  assert.equal(position.value, 'bottom');
  assert.equal(strength.value, '0');
});

test('an Overlay that fails says why and leaves the dialog usable', async () => {
  const h = harness({
    segments: scenes(), overlayAvailable: true,
    postJson: async (url) => {
      if (url.endsWith('/thumbnail/overlay')) throw new Error('The frame was not found: C:/gone.png');
      return overlayReplies[url];
    },
  });
  await pickFrame(h);
  const { engine } = overlayControls(h);
  engine.value = 'overlay';
  engine.listeners.change[0]();
  await h.button('Generate thumbnail').onclick();
  assert.equal(h.toasts.at(-1)[1], true);
  assert.match(h.toasts.at(-1)[0], /The frame was not found/);
  assert.equal(h.button('Generate thumbnail').disabled, false, 'it can be tried again');
  assert.equal(h.state.thumbnailIntro.image_path, '');
});

test('Qwen Image Edit gets its own kind of prompt, the frame as a reference, and the models are left to the builder', async () => {
  const replies = { ...frameReplies, '/vrgdg/workflow_runner/build_qwen_image_edit_prompt': { prompt: {} } };
  const h = harness({ segments: scenes(), postJson: async (url) => replies[url] });
  const { select } = await pickFrame(h);
  const hint = h.all.find((node) => /edits the frame by instruction/.test(node.textContent));
  const titleInput = h.all.find((node) => node.tag === 'input' && node.maxLength === 120);
  titleInput.value = 'HEAT';
  select.value = 'qwen_image_edit';
  select.listeners.change[0]();
  assert.match(hint.textContent, /^Qwen Image Edit edits the frame by instruction/);
  assert.match(hint.textContent, /newest Qwen Image Edit model you have installed \(2511 before 2509\)/);
  assert.match(hint.textContent, /use Overlay when nothing may change/);
  assert.match(hint.textContent, /put the lettering you want in Style notes/);
  const styleInput = h.all.find((node) => node.tag === 'textarea' && /Optional look/.test(node.placeholder || ''));
  styleInput.value = 'very artistic and graffiti style text and lines';
  await h.button('Write prompt with LLM').onclick();
  assert.equal(h.calls.find(([url]) => url.endsWith('/thumbnail/prompt'))[1].engine_kind, 'qwen_edit', 'a prompt about the change, not the picture');
  // The reply ('A close-up of a singer.') has no title, so the sentence that is added carries the style notes, not sans-serif capitals.
  assert.match(h.state.thumbnailIntro.prompt, /Add the text "HEAT" as lettering in this style: very artistic and graffiti style text and lines/);
  assert.doesNotMatch(h.state.thumbnailIntro.prompt, /sans-serif/);
  await h.button('Generate thumbnail').onclick();
  const [, built] = h.calls.find(([url]) => url.endsWith('/build_qwen_image_edit_prompt'));
  assert.deepEqual(JSON.parse(JSON.stringify(built.image_ingredients)), [{ path: 'C:/p/thumbnail/source_0001.png' }]);
  assert.match(built.prompt, /Add the text "HEAT" as lettering in this style: very artistic and graffiti style text and lines/, 'the title and its style are in what was sent');
  for (const modelField of ['unet_name', 'clip_name', 'vae_name', 'lora_name', 'steps', 'cfg']) assert.equal(modelField in built, false, modelField);
  assert.equal(h.state.thumbnailIntro.engine, 'qwen_image_edit');
  assert.equal(h.state.thumbnailIntro.history[0].engine, 'qwen_image_edit');
});

// -- API image: the frame edited by the image API of the LLM Runner's provider --------------------------------------------------------

const GEMINI_RUNNER = { text_runner: 'builtin', llm_api_provider: 'google', llm_api_key: 'g-key' };
const apiReplies = {
  ...frameReplies,
  '/vrgdg/music_builder/thumbnail/api_image': { image_path: 'C:/p/thumbnail/thumbnail_0001.png', width: 1280, height: 720, provider: 'gemini', model: 'gemini-3-pro-image-preview' },
};

function apiControls(h) {
  const models = h.find((node) => node.tag === 'datalist');
  return {
    engine: h.find((node) => node.tag === 'select'),
    model: h.find((node) => node.tag === 'input' && node.attrs.list === 'vrgdg-thumbnail-api-models'),
    models,
    group: h.all.find((node) => node.children && node.children.includes(models)),
    provider: h.all.find((node) => /^(Provider: |The LLM Runner's provider cannot draw)/.test(node.textContent)),
    hint: h.all.find((node) => /edits the frame by instruction|image API of the provider/.test(node.textContent)),
    prompt: h.all.find((node) => node.tag === 'textarea' && /^The edit prompt/.test(node.placeholder || '')),
  };
}

const chooseApiImage = (h) => {
  const controls = apiControls(h);
  controls.engine.value = 'api_image';
  controls.engine.listeners.change[0]();
  return controls;
};

test('the API image engine shows the LLM Runner\'s provider and that provider\'s image models, and only for itself', () => {
  const h = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: GEMINI_RUNNER });
  h.open();
  const { group, models, model, provider, hint, engine } = apiControls(h);
  assert.equal(group.style.display, 'none', 'the default engine has no use for it');
  engine.value = 'api_image';
  engine.listeners.change[0]();
  assert.equal(group.style.display, 'flex');
  assert.equal(provider.textContent, 'Provider: Gemini, with the key from the LLM Runner.');
  assert.deepEqual(models.children.map((option) => option.value), ['gemini-3-pro-image-preview', 'gemini-3.1-flash-image-preview']);
  assert.equal(model.placeholder, 'Empty: gemini-3-pro-image-preview');
  assert.match(hint.textContent, /image API of the provider set in the LLM Runner \(Gemini, Grok or OpenAI\) with that provider's key/);
  assert.match(hint.textContent, /use Overlay to keep them exactly/);
  engine.value = 'qwen_image_edit';
  engine.listeners.change[0]();
  assert.equal(group.style.display, 'none');
});

test('the model list follows the provider, and a provider that cannot draw is said so in the dialog', () => {
  const grok = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: { llm_api_provider: 'grok', llm_api_key: 'k' } });
  grok.open();
  const shown = chooseApiImage(grok);
  assert.equal(shown.provider.textContent, 'Provider: Grok, with the key from the LLM Runner.');
  assert.deepEqual(shown.models.children.map((option) => option.value), ['grok-imagine-image-2.0', 'grok-imagine-image']);

  const claude = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: { llm_api_provider: 'anthropic', llm_api_key: 'k' } });
  claude.open();
  const none = chooseApiImage(claude);
  assert.match(none.provider.textContent, /cannot draw pictures. Pick Gemini \(google\), Grok or OpenAI in the LLM Runner/);
  assert.deepEqual(none.models.children, []);
  assert.equal(none.model.placeholder, '');
});

test('Generate with the API image sends the frame, the prompt, the provider and its key to the pack, and nothing to ComfyUI', async () => {
  const h = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: GEMINI_RUNNER, postJson: async (url) => apiReplies[url] });
  await pickFrame(h);
  const { engine, model } = chooseApiImage(h);
  const titleInput = h.all.find((node) => node.tag === 'input' && node.maxLength === 120);
  titleInput.value = 'HEAT';
  model.value = ' gemini-3.1-flash-image-preview ';
  await h.button('Write prompt with LLM').onclick();
  const asked = h.calls.find(([url]) => url.endsWith('/thumbnail/prompt'))[1];
  assert.equal(asked.engine_kind, 'edit', 'the API image is given the editing prompt');
  assert.equal(asked.llm_api_provider, 'google', 'the LLM writes it with the same runner');
  await h.button('Generate thumbnail').onclick();
  assert.deepEqual(h.calls.map(([url]) => url), [
    '/vrgdg/music_builder/thumbnail/frame', '/vrgdg/music_builder/thumbnail/prompt', '/vrgdg/music_builder/thumbnail/api_image',
  ], 'no workflow is built or queued, and nothing is saved separately');
  const [, sent] = h.calls.at(-1);
  assert.deepEqual(JSON.parse(JSON.stringify(sent)), {
    project_folder: 'C:/p', source_path: 'C:/p/thumbnail/source_0001.png', prompt: `A close-up of a singer. ${h.context.thumbnailTitleClause('HEAT', 'edit')}`,
    provider: 'gemini', api_key: 'g-key', model: 'gemini-3.1-flash-image-preview',
  });
  const intro = h.state.thumbnailIntro;
  assert.equal(intro.engine, 'api_image');
  assert.equal(intro.image_path, 'C:/p/thumbnail/thumbnail_0001.png', 'the pack saved it, so that is the picture');
  assert.equal(intro.api_image.model, 'gemini-3.1-flash-image-preview', 'the model is remembered');
  assert.equal(intro.history.length, 1);
  assert.equal(intro.history[0].engine, 'api_image');
  assert.equal(intro.history[0].prompt, sent.prompt);
  assert.equal(engine.disabled, false);
});

test('the API image title is mandatory too, and an empty model means the provider\'s first', async () => {
  const h = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: { llm_api_provider: 'openai', llm_api_key: 'o-key' }, postJson: async (url) => apiReplies[url] });
  await pickFrame(h);
  const { prompt } = chooseApiImage(h);
  h.all.find((node) => node.tag === 'input' && node.maxLength === 120).value = 'HEAT';
  prompt.value = 'Make it pop.';
  await h.button('Generate thumbnail').onclick();
  const [, sent] = h.calls.at(-1);
  assert.equal(sent.prompt, `Make it pop. ${h.context.thumbnailTitleClause('HEAT', 'edit')}`);
  assert.equal(sent.provider, 'openai');
  assert.equal(sent.model, '', 'the pack uses the provider\'s first image model');
});

test('the API image engine says what is missing before it calls anything, and the dialog stays usable', async () => {
  for (const [runner, message] of [
    [{ llm_api_provider: 'anthropic', llm_api_key: 'k' }, /cannot draw pictures/],
    [{ llm_api_provider: 'grok', llm_api_key: '' }, /Add your Grok API key in the LLM Runner first/],
  ]) {
    const h = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner, postJson: async (url) => apiReplies[url] });
    await pickFrame(h);
    chooseApiImage(h).prompt.value = 'Make it pop.';
    await h.button('Generate thumbnail').onclick();
    assert.equal(h.calls.some(([url]) => url.endsWith('/thumbnail/api_image')), false, 'nothing was sent');
    assert.equal(h.toasts.at(-1)[1], true);
    assert.match(h.toasts.at(-1)[0], message);
    assert.equal(h.button('Generate thumbnail').disabled, false);
    assert.equal(h.state.thumbnailIntro.image_path, '');
  }
});

test('an API image that the provider refuses shows its message and can be tried again', async () => {
  const h = harness({
    segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: GEMINI_RUNNER,
    postJson: async (url) => {
      if (url.endsWith('/thumbnail/api_image')) throw new Error('Gemini HTTP 429: quota exceeded');
      return apiReplies[url];
    },
  });
  await pickFrame(h);
  chooseApiImage(h).prompt.value = 'Make it pop.';
  await h.button('Generate thumbnail').onclick();
  assert.match(h.toasts.at(-1)[0], /Gemini HTTP 429: quota exceeded/);
  assert.equal(h.toasts.at(-1)[1], true);
  assert.equal(h.button('Generate thumbnail').disabled, false, 'it can be tried again');
  assert.equal(h.state.thumbnailIntro.image_path, '');
  assert.equal(h.state.thumbnailIntro.history.length, 0);
});

test('the API image reopens with the model that was chosen', () => {
  const h = harness({ segments: scenes(), apiImage: API_IMAGE_CAPABILITY, runner: GEMINI_RUNNER, intro: { engine: 'api_image', api_image: { model: 'my-own-model' } } });
  h.open();
  const { engine, model } = apiControls(h);
  assert.equal(engine.value, 'api_image');
  assert.equal(model.value, 'my-own-model');
});

test('the build says which engines it has: the overlay and the providers of the API image', async () => {
  const ask = async (reply) => {
    const h = harness({ segments: scenes() });
    h.context.api.fetchApi = async () => reply;
    const state = {};
    await h.context.loadThumbnailCapabilities(state);
    return state;
  };
  const full = await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true, overlay: true, api_image: API_IMAGE_CAPABILITY }) });
  assert.equal(full.thumbnailOverlayAvailable, true);
  assert.deepEqual(Object.keys(full.thumbnailApiImage.providers), ['gemini', 'grok', 'openai']);
  const older = await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true }) });
  assert.equal(older.thumbnailAvailable, true);
  assert.equal(older.thumbnailOverlayAvailable, false);
  assert.equal(older.thumbnailApiImage, null, 'a build without the route does not offer the engine');
  const missing = await ask({ ok: false, json: async () => null });
  assert.equal(missing.thumbnailAvailable, false);
  assert.equal(missing.thumbnailApiImage, null);
});

// The person's own thumbnail picture and intro video.
const PICK = '/vrgdg/music_builder/pick_path';
const SAVE = '/vrgdg/music_builder/thumbnail/save';
const pictureField = (h) => h.find((node) => node.tag === 'input' && /thumbnail picture/.test(node.placeholder || ''));
const videoField = (h) => h.find((node) => node.tag === 'input' && /intro video/.test(node.placeholder || ''));
const savedReply = { image_path: 'C:/p/thumbnail/thumbnail_0004.png', width: 1920, height: 1080 };

test('a picture of their own is copied into the project, becomes the thumbnail, switches the first frame on and joins the history', async () => {
  const replies = { [PICK]: { path: 'C:/mine/cover.jpg' }, [SAVE]: savedReply };
  const h = harness({ segments: scenes(), postJson: async (url) => replies[url] });
  h.open();
  await h.button('Choose a picture...').onclick();
  assert.deepEqual(h.calls.map(([url]) => url), [PICK, SAVE], 'the file dialog, then the route that keeps every thumbnail: no LLM, no ComfyUI, no overlay');
  assert.deepEqual(JSON.parse(JSON.stringify(h.calls[0][1])), { kind: 'image' });
  assert.deepEqual(JSON.parse(JSON.stringify(h.calls[1][1])), { project_folder: 'C:/p', source_path: 'C:/mine/cover.jpg' });
  const intro = h.state.thumbnailIntro;
  assert.equal(intro.image_path, 'C:/p/thumbnail/thumbnail_0004.png');
  assert.equal(intro.enabled, true, 'choosing it is asking for it');
  assert.deepEqual(JSON.parse(JSON.stringify(intro.history)), [{ path: 'C:/p/thumbnail/thumbnail_0004.png', prompt: '', engine: 'own', scene_id: '' }]);
  assert.equal(h.saves.length, 1, 'it is saved with the project');
  assert.ok(h.all.find((node) => node.tag === 'img' && node.src === 'image:C:/p/thumbnail/thumbnail_0004.png' && node.style.display === 'block'), 'the preview shows it');
  assert.equal(pictureField(h).value, 'C:/mine/cover.jpg');
  assert.match(h.all.find((node) => /Your picture is the thumbnail/.test(node.textContent)).textContent, /1920\u00D71080/);
});

test('a typed or pasted path works too, quotes are dropped, and an empty one asks for a picture without calling anything', async () => {
  const h = harness({ segments: scenes(), postJson: async () => savedReply });
  h.open();
  await h.button('Use this picture').onclick();
  assert.equal(h.calls.length, 0);
  assert.match(h.toasts[0][0], /Choose a picture first/);
  assert.equal(h.toasts[0][1], true);
  pictureField(h).value = '  "C:\\Users\\me\\My Pictures\\cover.png" ';
  await h.button('Use this picture').onclick();
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0][1].source_path, 'C:\\Users\\me\\My Pictures\\cover.png');
});

test('cancelling the picture dialog changes nothing, and a picture that cannot be read says why and leaves the dialog usable', async () => {
  const cancelled = harness({ segments: scenes(), postJson: async () => ({ path: '' }) });
  cancelled.open();
  await cancelled.button('Choose a picture...').onclick();
  assert.deepEqual(cancelled.calls.map(([url]) => url), [PICK]);
  assert.equal(cancelled.state.thumbnailIntro.image_path, '');
  assert.equal(cancelled.toasts.length, 0);

  let failing = true;
  const h = harness({
    segments: scenes(),
    postJson: async (url) => {
      if (url === PICK) return { path: 'C:/mine/notes.txt' };
      if (failing) throw new Error('Thumbnail image was not found: C:/mine/notes.txt');
      return savedReply;
    },
  });
  h.open();
  await h.button('Choose a picture...').onclick();
  assert.equal(h.state.thumbnailIntro.image_path, '');
  assert.deepEqual(Array.from(h.state.thumbnailIntro.history), []);
  assert.deepEqual(h.toasts.at(-1), ['Thumbnail image was not found: C:/mine/notes.txt', true]);
  assert.ok(h.all.find((node) => node.textContent === 'Thumbnail image was not found: C:/mine/notes.txt'), 'the dialog shows it too');
  assert.equal(h.button('Use this picture').disabled, false, 'not left busy');
  failing = false;
  await h.button('Choose a picture...').onclick();
  assert.equal(h.state.thumbnailIntro.image_path, 'C:/p/thumbnail/thumbnail_0004.png');
});

test('a picture the person chose is in the history with the generated ones, and an earlier one can be picked again', async () => {
  const h = harness({
    segments: scenes(),
    postJson: async (url) => (url === PICK ? { path: 'C:/mine/cover.jpg' } : savedReply),
    intro: { history: [{ path: 'C:/p/thumbnail/thumbnail_0002.png', prompt: 'x', engine: 'nano_banana', scene_id: 'a' }] },
  });
  h.open();
  await h.button('Choose a picture...').onclick();
  // The history is painted again after every change, and the fake DOM keeps the strips it replaced: the last two pictures are the current ones.
  const thumbs = h.all.filter((node) => node.tag === 'img' && node.title === 'Use this thumbnail').slice(-2);
  assert.deepEqual(thumbs.map((node) => node.src), ['image:C:/p/thumbnail/thumbnail_0004.png', 'image:C:/p/thumbnail/thumbnail_0002.png'], 'newest first');
  thumbs[1].onclick();
  assert.equal(h.state.thumbnailIntro.image_path, 'C:/p/thumbnail/thumbnail_0002.png');
  assert.equal(h.state.thumbnailIntro.history.length, 2, 'nothing is lost');
});

test('an intro video of their own is chosen, switched on, previewed where it is and kept with the project', async () => {
  const h = harness({ segments: scenes(), postJson: async () => ({ path: 'D:/clips/my intro.mp4' }) });
  h.open();
  await h.button('Choose a video...').onclick();
  assert.deepEqual(h.calls.map(([url, payload]) => [url, payload.kind]), [[PICK, 'video']], 'only the file dialog: the file is used where it is, nothing is copied or probed');
  const intro = h.state.thumbnailIntro;
  assert.equal(intro.video_intro_path, 'D:/clips/my intro.mp4');
  assert.equal(intro.video_intro_enabled, true);
  assert.equal(videoField(h).value, 'D:/clips/my intro.mp4');
  const videos = h.all.filter((node) => node.tag === 'video');
  const preview = videos[1];
  assert.equal(preview.src.startsWith('video:D:/clips/my intro.mp4'), true);
  assert.equal(preview.style.display, 'block');
  assert.equal(preview.controls, true);
  assert.equal(videos[0].src, 'video:C:/v/a.mp4?7', 'the scrubber\'s player is left alone');
  assert.equal(h.saves.length, 1);
  preview.duration = 4.5;
  preview.videoWidth = 1280;
  preview.videoHeight = 720;
  preview.listeners.loadedmetadata[0]();
  assert.match(h.all.find((node) => /long, /.test(node.textContent)).textContent, /^0:04\.50 long, 1280\u00D7720$/);
  preview.listeners.error[0]();
  assert.match(h.all.find((node) => /cannot play that file/.test(node.textContent)).textContent, /still tried when the final video is made/);
});

test('typing the path of the intro video works, and clearing it switches the intro off', () => {
  const h = harness({ segments: scenes() });
  h.open();
  const field = videoField(h);
  field.value = ' "E:/intro.mov" ';
  field.listeners.change[0]();
  assert.equal(h.state.thumbnailIntro.video_intro_path, 'E:/intro.mov');
  assert.equal(h.state.thumbnailIntro.video_intro_enabled, true);
  field.value = '';
  field.listeners.change[0]();
  assert.equal(h.state.thumbnailIntro.video_intro_path, '');
  assert.equal(h.state.thumbnailIntro.video_intro_enabled, false);
  const preview = h.all.filter((node) => node.tag === 'video')[1];
  assert.equal(preview.style.display, 'none');
  assert.ok(h.all.find((node) => node.textContent === 'No intro video chosen.'));
});

test('reopening shows the intro video that was chosen, and a switched-off one keeps its file but is not used', () => {
  const h = harness({ segments: scenes(), intro: { video_intro_enabled: false, video_intro_path: 'D:/clips/intro.mp4' } });
  h.open();
  assert.equal(videoField(h).value, 'D:/clips/intro.mp4');
  const preview = h.all.filter((node) => node.tag === 'video')[1];
  assert.equal(preview.src.startsWith('video:D:/clips/intro.mp4'), true);
  assert.equal(h.context.videoIntroForStitch(h.state), null, 'off means the stitch leaves the video alone');
});

test('a failing file dialog says so and leaves the dialog usable', async () => {
  const h = harness({ segments: scenes(), postJson: async () => { throw new Error('Native file dialog is not available on this system. Type the path instead.'); } });
  h.open();
  await h.button('Choose a video...').onclick();
  assert.match(h.toasts.at(-1)[0], /Type the path instead/);
  await h.button('Choose a picture...').onclick();
  assert.equal(h.toasts.length, 2);
  assert.equal(h.state.thumbnailIntro.video_intro_path, '');
});

test('a build without the intro video route does not offer it, and its stitch leaves the video alone', async () => {
  const h = harness({ segments: scenes(), videoIntroAvailable: false, intro: { video_intro_enabled: true, video_intro_path: 'D:/clips/intro.mp4' } });
  h.open();
  const row = h.all.find((node) => node.children.includes(videoField(h)));
  const column = h.all.find((node) => node.children.includes(row));
  assert.equal(column.style.display, 'none');
  assert.equal(h.all.find((node) => node.children.includes(column)).style.gridTemplateColumns, 'minmax(0,1fr)', 'the picture takes the whole row');
  const stitched = await h.context.applyVideoIntroAfterStitch(h.state, { final_video_path: 'C:/p/FINAL_VIDEO.mp4' }, { projectFolder: 'C:/p' });
  assert.equal(h.calls.length, 0);
  assert.deepEqual({ ...stitched }, { final_video_path: 'C:/p/FINAL_VIDEO.mp4' });
});

test('the stitch puts the intro video in front of the video: the whole file, in place, and nothing else is sent', async () => {
  const h = harness({
    segments: scenes(), postJson: async () => ({ intro_frames: 96, intro_seconds: 4, method: 'copy', duration: 64 }),
    intro: { video_intro_enabled: true, video_intro_path: 'D:/clips/intro.mp4' },
  });
  const progress = [];
  const stitched = await h.context.applyVideoIntroAfterStitch(h.state, { final_video_path: 'C:/p/FINAL_VIDEO.mp4', other: 1 }, {
    projectFolder: 'C:/p', progress: { set: (text, percent) => progress.push([text, percent]) },
  });
  assert.equal(h.calls.length, 1);
  const [url, sent] = h.calls[0];
  assert.equal(url, '/vrgdg/music_builder/video_intro/prepend');
  assert.deepEqual(JSON.parse(JSON.stringify(sent)), { input_path: 'C:/p/FINAL_VIDEO.mp4', intro_path: 'D:/clips/intro.mp4', replace_source: true, project_folder: 'C:/p' });
  assert.equal(stitched.video_intro.intro_frames, 96);
  assert.equal(stitched.final_video_path, 'C:/p/FINAL_VIDEO.mp4');
  assert.equal(stitched.other, 1, 'the stitch answer comes back unchanged apart from the intro');
  assert.equal(progress.length, 1);
  // Nothing happens when it is off, or when there is no video to put it in front of.
  h.state.thumbnailIntro = h.context.normalizeThumbnailIntro({ video_intro_enabled: false, video_intro_path: 'D:/clips/intro.mp4' });
  await h.context.applyVideoIntroAfterStitch(h.state, { final_video_path: 'C:/p/FINAL_VIDEO.mp4' }, { projectFolder: 'C:/p' });
  h.state.thumbnailIntro = h.context.normalizeThumbnailIntro({ video_intro_enabled: true, video_intro_path: 'D:/clips/intro.mp4' });
  await h.context.applyVideoIntroAfterStitch(h.state, { final_video_path: '' }, { projectFolder: 'C:/p' });
  assert.equal(h.calls.length, 1);
});

test('an intro video that cannot be added leaves the plain video, says why, and the thumbnail still goes in', async () => {
  const h = harness({
    segments: scenes(),
    postJson: async (url) => {
      if (url.endsWith('/video_intro/prepend')) throw new Error('The intro video was not found: D:/clips/intro.mp4');
      return { intro_frames: 1 };
    },
    intro: { enabled: true, image_path: 'C:/p/thumbnail/thumbnail_0001.png', video_intro_enabled: true, video_intro_path: 'D:/clips/intro.mp4' },
  });
  const plain = { final_video_path: 'C:/p/FINAL_VIDEO.mp4' };
  const withIntro = await h.context.applyVideoIntroAfterStitch(h.state, plain, { projectFolder: 'C:/p' });
  assert.deepEqual({ ...withIntro }, plain);
  assert.match(h.toasts[0][0], /The intro video could not be added, so the video starts with the first scene\. The intro video was not found/);
  assert.equal(h.toasts[0][1], true);
  const done = await h.context.applyThumbnailIntroAfterStitch(h.state, withIntro, { projectFolder: 'C:/p' });
  assert.equal(done.thumbnail_intro.intro_frames, 1);
});

test('with both on, the intro video is joined first and the thumbnail second, so the picture stays the first frame', async () => {
  const order = [];
  const h = harness({
    segments: scenes(),
    postJson: async (url) => { order.push(url); return {}; },
    intro: { enabled: true, image_path: 'C:/p/thumbnail/thumbnail_0001.png', video_intro_enabled: true, video_intro_path: 'D:/clips/intro.mp4' },
  });
  // What video_render.mjs does after the stitch and the overlays.
  const withIntro = await h.context.applyVideoIntroAfterStitch(h.state, { final_video_path: 'C:/p/FINAL_VIDEO.mp4' }, { projectFolder: 'C:/p' });
  const done = await h.context.applyThumbnailIntroAfterStitch(h.state, withIntro, { projectFolder: 'C:/p' });
  assert.deepEqual(order, ['/vrgdg/music_builder/video_intro/prepend', '/vrgdg/music_builder/thumbnail/prepend']);
  assert.deepEqual(Object.keys(done).sort(), ['final_video_path', 'thumbnail_intro', 'video_intro']);
  const render = require('node:fs').readFileSync(require('node:path').join(__dirname, '..', 'web', 'music_video_builder', 'video_render.mjs'), 'utf8');
  assert.ok(render.indexOf('applyVideoIntroAfterStitch(state, overlaid') < render.indexOf('applyThumbnailIntroAfterStitch(state, withIntro'), 'the Builder runs them in that order');
});

test('the build says whether it can join an intro video', async () => {
  const ask = async (reply) => {
    const h = harness({ segments: scenes() });
    h.context.api.fetchApi = async () => reply;
    const state = {};
    await h.context.loadThumbnailCapabilities(state);
    return state;
  };
  assert.equal((await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true, video_intro: true }) })).videoIntroAvailable, true);
  assert.equal((await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true }) })).videoIntroAvailable, false, 'an older pack');
  assert.equal((await ask({ ok: false, json: async () => null })).videoIntroAvailable, false);
});

const browserReplies = {
  ...frameReplies,
  '/vrgdg/workflow_runner/build_flow_gpt_image_prompt': { prompt: { '1': {} }, provider_label: 'GPT Image', image_count: 1 },
};
const buildCalls = (h) => h.calls.filter(([url]) => url.endsWith('/build_flow_gpt_image_prompt')).map(([, payload]) => JSON.parse(JSON.stringify(payload)));

// Browser AI is three engines in the list, one for each site: Flow, GPT Image and Meta AI. The label is the Image panel's name of the site.
const BROWSER_SITES = [
  ['browser_flow', 'flow_nano_banana', 'Flow Nano Banana'],
  ['browser_gpt', 'gpt_image', 'GPT Image'],
  ['browser_meta', 'meta_ai', 'Meta AI'],
];
const pickEngine = (select, engine) => {
  select.value = engine;
  select.listeners.change[0]();
};
async function chooseBrowserAi(h, engine) {
  const { select } = await pickFrame(h);
  pickEngine(select, engine);
  return { select };
}

test('each Browser AI engine names its own site, says what it does, and takes the editing prompt', async () => {
  for (const [engine, , label] of BROWSER_SITES) {
    // The Image panel's provider is something else: the engine decides the site.
    const h = harness({ segments: scenes(), browserAi: true, browserSettings: { provider: 'bing', failure_mode: 'try_other_provider' }, postJson: async (url) => browserReplies[url] });
    await chooseBrowserAi(h, engine);
    const hint = h.all.find((node) => new RegExp(`^${label} opens in your own browser`).test(node.textContent));
    assert.ok(hint, `the hint names ${label}`);
    assert.match(hint.textContent, /uploads your frame as a reference/);
    assert.match(hint.textContent, new RegExp(`login button, set to ${label}\\.`));
    assert.match(hint.textContent, /no other site is tried/);
    assert.match(hint.textContent, /redraws the frame/, 'it redraws the frame like the other AI workflows');
    await h.button('Write prompt with LLM').onclick();
    assert.equal(h.calls.find(([url]) => url.endsWith('/thumbnail/prompt'))[1].engine_kind, 'edit', engine);
  }
});

test('there is no provider dropdown: each site is its own entry in the engine list', async () => {
  const h = harness({ segments: scenes(), browserAi: true, browserSettings: { provider: 'gpt_image' }, postJson: async (url) => browserReplies[url] });
  const { select } = await pickFrame(h);
  for (const [engine] of BROWSER_SITES) {
    pickEngine(select, engine);
    assert.equal(h.all.find((node) => node.label === 'Browser AI provider'), undefined, engine);
    assert.equal(h.all.find((node) => node.tag === 'select' && node.options.includes('meta_ai')), undefined, engine);
  }
  assert.equal(h.all.filter((node) => node.tag === 'select' && node.options.includes('browser_gpt')).length, 1, 'the engine list is the only place to choose');
});

test('Generate with a Browser AI engine builds the workflow for that site, waits for the picture in ComfyUI and keeps it', async () => {
  let shown = false;
  const h = harness({
    // The Image panel is set to Flow: the engine chosen here is GPT Image, and that is the site that is asked.
    segments: scenes(), browserAi: true, browserSettings: { provider: 'flow_nano_banana', failure_mode: 'last_successful_image', gpt_timeout_seconds: 600 },
    postJson: async (url) => browserReplies[url],
    // While ComfyUI is still waiting for the picture, the dialog says which site is working in the browser.
    images: async () => {
      shown = h.all.some((node) => /Asking GPT Image in your browser/.test(node.textContent));
      return [{ filename: 'gen.png', subfolder: '', type: 'output' }];
    },
  });
  await chooseBrowserAi(h, 'browser_gpt');
  const titleInput = h.all.find((node) => node.tag === 'input' && node.maxLength === 120);
  titleInput.value = 'HEAT';
  await h.button('Write prompt with LLM').onclick();
  await h.button('Generate thumbnail').onclick();
  const [built] = buildCalls(h);
  assert.equal(built.provider, 'gpt_image');
  assert.equal(built.aspect_ratio, '16:9');
  assert.deepEqual(built.image_ingredients, [{ path: 'C:/p/thumbnail/source_0001.png' }], 'the chosen frame is the reference');
  assert.match(built.prompt, /HEAT/);
  assert.equal(built.reuse_open_project, false);
  assert.equal(built.timeout_seconds, 601, 'the Image panel\'s 600 seconds, nudged by a second');
  assert.equal(h.queued.length, 1, 'the workflow is queued in ComfyUI once');
  const save = h.calls.find(([url]) => url.endsWith('/thumbnail/save'));
  assert.deepEqual(JSON.parse(JSON.stringify(save[1].image)), { filename: 'gen.png', subfolder: '', type: 'output' });
  assert.equal(h.state.thumbnailIntro.image_path, 'C:/p/thumbnail/thumbnail_0001.png');
  assert.equal(h.state.thumbnailIntro.engine, 'browser_gpt');
  assert.equal(h.state.thumbnailIntro.history[0].engine, 'browser_gpt');
  assert.equal(h.state.thumbnailIntro.browser_ai, undefined, 'the engine says which site, so there is no other setting');
  assert.equal(shown, true, 'the status says the site is working in the browser');
});

test('each Browser AI engine asks its own site with that site\'s own wait', async () => {
  for (const [engine, provider, seconds] of [['browser_flow', 'flow_nano_banana', 421], ['browser_gpt', 'gpt_image', 601], ['browser_meta', 'meta_ai', 501]]) {
    const h = harness({
      segments: scenes(), browserAi: true, postJson: async (url) => browserReplies[url],
      browserSettings: { provider: 'gpt_image', gpt_timeout_seconds: 600, flow_timeout_seconds: 420, meta_timeout_seconds: 500 },
    });
    await chooseBrowserAi(h, engine);
    await h.button('Write prompt with LLM').onclick();
    await h.button('Generate thumbnail').onclick();
    assert.deepEqual(buildCalls(h).map((payload) => [payload.provider, payload.timeout_seconds]), [[provider, seconds]], engine);
    assert.equal(h.state.thumbnailIntro.engine, engine, 'kept with the project');
  }
  // Opening the dialog again starts on the engine that was used.
  const again = harness({ segments: scenes(), browserAi: true, intro: { engine: 'browser_meta' }, browserSettings: { provider: 'gpt_image' }, postJson: async (url) => browserReplies[url] });
  again.open();
  assert.equal(again.find((node) => node.tag === 'select').value, 'browser_meta');
});

test('a second try with the same prompt changes the workflow a little, so ComfyUI really runs it again', async () => {
  const h = harness({ segments: scenes(), browserAi: true, browserSettings: { provider: 'gpt_image', gpt_timeout_seconds: 600 }, postJson: async (url) => browserReplies[url] });
  await chooseBrowserAi(h, 'browser_gpt');
  await h.button('Write prompt with LLM').onclick();
  await h.button('Generate thumbnail').onclick();
  await h.button('Generate thumbnail').onclick();
  await h.button('Generate thumbnail').onclick();
  assert.deepEqual(buildCalls(h).map((payload) => payload.timeout_seconds), [601, 600, 601]);
  // At the longest wait it goes down instead of up.
  const long = harness({ segments: scenes(), browserAi: true, browserSettings: { provider: 'gpt_image', gpt_timeout_seconds: 2400 }, postJson: async (url) => browserReplies[url] });
  await chooseBrowserAi(long, 'browser_gpt');
  await long.button('Write prompt with LLM').onclick();
  await long.button('Generate thumbnail').onclick();
  assert.equal(buildCalls(long)[0].timeout_seconds, 2399);
});

test('a Browser AI engine never tries another site after its own fails, whatever the Image panel\'s failure setting says', async () => {
  const fail = async () => {
    throw new Error('try 1 failed');
  };
  for (const failure_mode of ['try_other_provider', 'last_successful_image']) {
    const h = harness({
      segments: scenes(), browserAi: true, images: fail, postJson: async (url) => browserReplies[url],
      browserSettings: { provider: 'gpt_image', failure_mode, gpt_timeout_seconds: 600, flow_timeout_seconds: 420, meta_timeout_seconds: 500 },
    });
    await chooseBrowserAi(h, 'browser_gpt');
    await h.button('Write prompt with LLM').onclick();
    await h.button('Generate thumbnail').onclick();
    assert.deepEqual(buildCalls(h).map((payload) => payload.provider), ['gpt_image'], `no other provider is tried (${failure_mode})`);
    assert.equal(h.state.thumbnailIntro.image_path, '', 'nothing was made');
    assert.ok(h.toasts.some(([message, isError]) => isError && /try 1 failed/.test(message)), 'the reason is shown');
    assert.equal(h.button('Generate thumbnail').disabled, false, 'the dialog can be used again');
  }
});

// A build that drives Chrome itself (H3 Studio): one route makes the picture and keeps it, so there is no workflow to build, queue or save.
const directReplies = {
  ...frameReplies,
  '/vrgdg/music_builder/thumbnail/browser_ai': { image_path: 'C:/p/thumbnail/thumbnail_0001.png', provider: 'gpt_image', provider_label: 'GPT Image', width: 1024, height: 576 },
  '/vrgdg/music_builder/thumbnail/browser_ai/open': { provider: 'gpt_image', provider_label: 'GPT Image', url: 'https://chatgpt.com/images', debug_port: 9223 },
};
const directCalls = (h, suffix) => h.calls.filter(([url]) => url.endsWith(suffix)).map(([, payload]) => JSON.parse(JSON.stringify(payload)));

test('on a build that drives Chrome itself, each Browser AI engine says nothing needs installing and offers the sign-in button', async () => {
  for (const [engine, , label] of BROWSER_SITES) {
    const direct = harness({ segments: scenes(), browserAi: true, browserAiDirect: true, postJson: async (url) => directReplies[url] });
    await chooseBrowserAi(direct, engine);
    const hint = direct.all.find((node) => new RegExp(`^${label} opens in Chrome`).test(node.textContent));
    assert.ok(hint, `the hint names ${label}`);
    assert.match(hint.textContent, /Nothing needs installing/);
    assert.match(hint.textContent, /Each Browser AI engine has its own Chrome window and sign-in/);
    assert.match(hint.textContent, /no other site is tried/);
    assert.equal(direct.button('Open Chrome and sign in').style.display, '', `the button is there for ${engine}`);
  }
  // The ComfyUI workflow build has no sign-in button here: the pack's Image panel has its own.
  const workflow = harness({ segments: scenes(), browserAi: true, postJson: async (url) => browserReplies[url] });
  await chooseBrowserAi(workflow, 'browser_gpt');
  assert.equal(workflow.button('Open Chrome and sign in').style.display, 'none');
  assert.match(workflow.all.find((node) => /^GPT Image opens in your own browser/.test(node.textContent)).textContent, /Sign in there first with the Image panel's login button, set to GPT Image/);
  // The other engines do not show it either.
  const other = harness({ segments: scenes(), browserAi: true, browserAiDirect: true, postJson: async (url) => directReplies[url] });
  await pickFrame(other);
  assert.equal(other.button('Open Chrome and sign in').style.display, 'none');
});

test('Open Chrome and sign in opens the site of the chosen engine and says what to do next', async () => {
  const h = harness({ segments: scenes(), browserAi: true, browserAiDirect: true, browserSettings: { provider: 'flow_nano_banana' }, postJson: async (url) => directReplies[url] });
  await chooseBrowserAi(h, 'browser_gpt');
  const button = h.button('Open Chrome and sign in');
  await button.onclick();
  assert.deepEqual(directCalls(h, '/thumbnail/browser_ai/open'), [{ provider: 'gpt_image' }]);
  assert.ok(h.all.some((node) => /GPT Image is open in Chrome\. Sign in there, leave the window open, then click Generate\./.test(node.textContent)));
  assert.equal(button.disabled, false);
  // A failure is shown where the person looks and the button works again.
  const broken = harness({ segments: scenes(), browserAi: true, browserAiDirect: true, postJson: async (url) => { if (url.endsWith('/browser_ai/open')) throw new Error('Could not find Google Chrome.'); return directReplies[url]; } });
  await chooseBrowserAi(broken, 'browser_flow');
  await broken.button('Open Chrome and sign in').onclick();
  assert.ok(broken.all.some((node) => /Could not find Google Chrome\./.test(node.textContent)));
  assert.equal(broken.button('Open Chrome and sign in').disabled, false);
});

test('Generate with a Browser AI engine on a build that drives Chrome itself sends the frame to its route and keeps what comes back, with no ComfyUI queue', async () => {
  const h = harness({ segments: scenes(), browserAi: true, browserAiDirect: true, browserSettings: { provider: 'flow_nano_banana', failure_mode: 'last_successful_image', gpt_timeout_seconds: 600 }, postJson: async (url) => directReplies[url] });
  await chooseBrowserAi(h, 'browser_gpt');
  h.all.find((node) => node.tag === 'input' && node.maxLength === 120).value = 'HEAT';
  await h.button('Write prompt with LLM').onclick();
  await h.button('Generate thumbnail').onclick();
  const [made] = directCalls(h, '/thumbnail/browser_ai');
  assert.equal(made.provider, 'gpt_image', 'the engine decides the site, not the Image panel');
  assert.equal(made.source_path, 'C:/p/thumbnail/source_0001.png');
  assert.equal(made.project_folder, 'C:/p');
  assert.equal(made.timeout_seconds, 600, 'the Image panel\'s wait, not nudged: there is no ComfyUI cache to get around');
  assert.equal(made.reuse_open_project, false);
  assert.match(made.prompt, /HEAT/);
  assert.equal(h.queued.length, 0, 'nothing is queued in ComfyUI');
  assert.equal(directCalls(h, '/workflow_runner/build_flow_gpt_image_prompt').length, 0);
  assert.equal(directCalls(h, '/thumbnail/save').length, 0, 'the route already kept the picture');
  assert.equal(h.state.thumbnailIntro.image_path, 'C:/p/thumbnail/thumbnail_0001.png');
  assert.equal(h.state.thumbnailIntro.engine, 'browser_gpt');
  assert.equal(h.state.thumbnailIntro.history[0].engine, 'browser_gpt');
});

test('a Browser AI engine on a build that drives Chrome itself never tries another site after its own fails', async () => {
  const postJson = async (url, payload) => {
    if (!url.endsWith('/thumbnail/browser_ai')) return directReplies[url];
    throw new Error(`${payload.provider} is not signed in`);
  };
  for (const failure_mode of ['try_other_provider', 'last_successful_image']) {
    const h = harness({
      segments: scenes(), browserAi: true, browserAiDirect: true, postJson,
      browserSettings: { provider: 'gpt_image', failure_mode, gpt_timeout_seconds: 600, flow_timeout_seconds: 420, meta_timeout_seconds: 500 },
    });
    await chooseBrowserAi(h, 'browser_gpt');
    await h.button('Write prompt with LLM').onclick();
    await h.button('Generate thumbnail').onclick();
    assert.deepEqual(directCalls(h, '/thumbnail/browser_ai').map((p) => p.provider), ['gpt_image'], `no other provider is tried (${failure_mode})`);
    assert.equal(h.state.thumbnailIntro.image_path, '', 'nothing was made');
    assert.ok(h.toasts.some(([message, isError]) => isError && /gpt_image is not signed in/.test(message)), 'the reason is shown');
  }
});

test('on a build that drives Chrome itself, each engine opens its own site for sign-in and the chosen one makes the picture', async () => {
  const h = harness({
    segments: scenes(), browserAi: true, browserAiDirect: true, postJson: async (url) => directReplies[url],
    browserSettings: { provider: 'gpt_image', failure_mode: 'try_other_provider', gpt_timeout_seconds: 600, flow_timeout_seconds: 420, meta_timeout_seconds: 500 },
  });
  const { select } = await chooseBrowserAi(h, 'browser_gpt');
  // Each site gets its own sign-in, one after the other.
  for (const engine of ['browser_flow', 'browser_meta']) {
    pickEngine(select, engine);
    await h.button('Open Chrome and sign in').onclick();
  }
  assert.deepEqual(directCalls(h, '/thumbnail/browser_ai/open'), [{ provider: 'flow_nano_banana' }, { provider: 'meta_ai' }]);
  await h.button('Write prompt with LLM').onclick();
  await h.button('Generate thumbnail').onclick();
  assert.deepEqual(directCalls(h, '/thumbnail/browser_ai').map((p) => [p.provider, p.timeout_seconds]), [['meta_ai', 500]], 'only the chosen site, with its own wait');
  assert.equal(h.state.thumbnailIntro.engine, 'browser_meta');
});

test('the build says whether it drives Chrome itself for Browser AI', async () => {
  const ask = async (reply) => {
    const h = harness({ segments: scenes() });
    h.context.api.fetchApi = async () => reply;
    const state = {};
    await h.context.loadThumbnailCapabilities(state);
    return state;
  };
  const direct = await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true, browser_ai: true, browser_ai_direct: true }) });
  assert.deepEqual([direct.thumbnailBrowserAi, direct.thumbnailBrowserAiDirect], [true, true]);
  const workflow = await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true, browser_ai: true }) });
  assert.deepEqual([workflow.thumbnailBrowserAi, workflow.thumbnailBrowserAiDirect], [true, false], 'the pack runs it as a ComfyUI workflow');
  // Driving Chrome itself without Browser AI on offer is not a thing the dialog can use.
  const odd = await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true, browser_ai_direct: true }) });
  assert.deepEqual([odd.thumbnailBrowserAi, odd.thumbnailBrowserAiDirect], [false, false]);
  const none = await ask({ ok: false, json: async () => null });
  assert.deepEqual([none.thumbnailBrowserAi, none.thumbnailBrowserAiDirect], [false, false]);
});

test('the build says whether it has Browser AI', async () => {
  const ask = async (reply) => {
    const h = harness({ segments: scenes() });
    h.context.api.fetchApi = async () => reply;
    const state = {};
    await h.context.loadThumbnailCapabilities(state);
    return state;
  };
  assert.equal((await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true, browser_ai: true }) })).thumbnailBrowserAi, true);
  assert.equal((await ask({ ok: true, json: async () => ({ ok: true, version: 1, frame: true }) })).thumbnailBrowserAi, false, 'a build that does not say browser_ai does not have the engine');
  assert.equal((await ask({ ok: false, json: async () => null })).thumbnailBrowserAi, false);
});
