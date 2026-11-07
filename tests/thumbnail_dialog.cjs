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

function harness({ segments, frameAvailable = true, postJson } = {}) {
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
    fluxKleinLoraPayload: () => ({}),
    makeButton: (label) => Object.assign(control('button'), { textContent: label }),
    makeInput: (value = '', type = 'text') => Object.assign(control('input'), { value, type }),
    makeSelect: (options, value) => Object.assign(control('select'), { value }),
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
  vm.runInContext(readBuilderModule('thumbnail.mjs'), context);
  const state = {
    segments, projectFolder: 'C:/p', thumbnailAvailable: true, thumbnailFrameAvailable: frameAvailable,
    thumbnailIntro: context.normalizeThumbnailIntro(), nbImageSettings: { api_key: 'key', model: 'nb-model' }, fluxKleinSettings: {},
  };
  const deps = {
    state, projectInput: { value: 'C:/p' }, runnerPayload: () => ({ text_runner: 'builtin' }),
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
  assert.equal(h.state.thumbnailIntro.prompt, 'Recrop the attached image.');
  await h.button('Generate thumbnail').onclick();
  const built = h.calls.find(([url]) => url.endsWith('/build_nb_image_prompt'))[1];
  assert.deepEqual(JSON.parse(JSON.stringify(built.image_ingredients)), [{ path: 'C:/p/thumbnail/source_0001.png' }]);
  assert.equal(built.prompt, 'Recrop the attached image.');
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
