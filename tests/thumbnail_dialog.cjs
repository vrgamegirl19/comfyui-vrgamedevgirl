const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

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
} = {}) {
  const all = [];
  const calls = [];
  const toasts = [];
  const saves = [];
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
    makeField: (label, input) => control('label'),
    postJson: async (url, payload) => {
      calls.push([url, payload]);
      return postJson(url, payload);
    },
    queueWorkflowPrompt: async () => ({ prompt_id: 'p1' }),
    waitForImages: async () => [{ filename: 'gen.png', subfolder: '', type: 'output' }],
  });
  vm.runInContext(readBuilderModule('thumbnail_state.mjs'), context);
  vm.runInContext(readBuilderModule('thumbnail_engines.mjs'), context);
  vm.runInContext(readBuilderModule('thumbnail.mjs'), context);
  const state = {
    segments, projectFolder: 'C:/p', thumbnailAvailable: true, thumbnailFrameAvailable: frameAvailable, thumbnailOverlayAvailable: overlayAvailable,
    thumbnailApiImage: apiImage, thumbnailIntro: context.normalizeThumbnailIntro(intro), nbImageSettings: { api_key: 'key', model: 'nb-model' },
  };
  const deps = {
    state, projectInput: { value: 'C:/p' }, runnerPayload: () => runner,
    engineSettings: (engine) => ({ nano_banana: state.nbImageSettings, ...engineSettings })[engine],
    sceneImagePath: (segment) => segment.image || '', sceneVideoPath: (segment) => segment.video || '',
    sceneLabel: (segment, index) => `Scene ${index + 1}`, save: async () => { saves.push(JSON.stringify(state.thumbnailIntro.source_path)); },
    createProgressWindow: () => ({ set() {}, close() {} }),
  };
  const open = () => context.openThumbnailDialog(deps);
  const find = (predicate) => all.find(predicate);
  const button = (label) => find((node) => node.tag === 'button' && node.textContent === label);
  return { context, state, calls, toasts, saves, open, find, button, all };
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

test('with nothing to show the dialog says so and does not open', () => {
  const h = harness({ segments: [{ id: 'a', start: 0, end: 2 }] });
  assert.equal(h.open(), null);
  assert.match(h.toasts[0][0], /No scene has a rendered video or an image yet/);
  const none = harness({ segments: [] });
  assert.equal(none.open(), null);
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

test('only Overlay, Qwen Image Edit, NanoBanana and the API image are offered, and only where the build has them', async () => {
  const bare = harness({ segments: scenes() });
  bare.open();
  assert.deepEqual(Array.from(bare.find((node) => node.tag === 'select').options), ['qwen_image_edit', 'nano_banana']);
  const full = harness({ segments: scenes(), overlayAvailable: true, apiImage: API_IMAGE_CAPABILITY });
  full.open();
  assert.deepEqual(Array.from(full.find((node) => node.tag === 'select').options), ['overlay', 'qwen_image_edit', 'nano_banana', 'api_image']);
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
