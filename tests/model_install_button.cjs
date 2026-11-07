const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');
const { fire, loadDialog, makeContext, makeNode, walk } = require('./models_dialog_harness.cjs');

// The download buttons that install the file themselves (web/music_video_builder/model_install_button.mjs): the MOSS-Image Edit LoRA is
// called adapter_model.safetensors on Hugging Face, so its button has the Builder save it as moss_image_edit_lora.safetensors.

const STATUS = '/vrgdg/video_builder/models/status';
const INSTALL = '/vrgdg/video_builder/models/install';
const CANCEL = '/vrgdg/video_builder/models/cancel';
const ITEM = {
  label: 'Edit LoRA (instruction editing)',
  url: 'https://huggingface.co/chinoll/Moss-Image-Edit/resolve/main/adapter_model.safetensors',
  install: 'moss_edit_lora',
  filename: 'moss_image_edit_lora.safetensors',
};
const flush = () => new Promise((resolve) => setImmediate(resolve));
const entry = (extra = {}) => ({
  id: 'moss_edit_lora', installed: false, path: '', destination: 'C:/ComfyUI/models/loras',
  job: { state: 'idle', downloaded: 0, total: 0, error: '' }, ...extra,
});
const downloading = (downloaded, total = 100) => entry({ job: { state: 'downloading', downloaded, total, error: '' } });
const installed = () => entry({ installed: true, path: 'C:/ComfyUI/models/loras/moss_image_edit_lora.safetensors', size: 1, job: { state: 'done', downloaded: 1, total: 1, error: '' } });
const failed = (error) => entry({ job: { state: 'error', downloaded: 0, total: 0, error } });

// A button wired with fakes for the server, the toasts, the browser and the clock. `server.models` is what the status route answers.
function wired({ models = [entry()], installAnswer = null, installFails = false } = {}) {
  const context = makeContext();
  vm.runInContext(readBuilderModule('model_install_button.mjs'), context);
  const server = { models, calls: [] };
  const notes = [];
  const opened = [];
  const announced = [];
  const timers = [];
  const deps = {
    get: async (url) => { server.calls.push([url]); return { models: server.models }; },
    post: async (url, body) => {
      server.calls.push([url, body]);
      if (installFails) throw new Error('404 Not Found');
      return url === INSTALL ? { models: installAnswer || server.models } : {};
    },
    notify: (message, isError) => notes.push([message, Boolean(isError)]),
    open: (url) => opened.push(url),
    announce: (ids) => announced.push(ids),
    setTimer: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
    clearTimer: () => {},
  };
  const button = makeNode('button');
  context.args = { button, item: ITEM, deps };
  const stop = vm.runInContext('wireInstallButton(args.button, args.item, args.deps)', context);
  const click = async () => { fire(button, 'click'); await flush(); };
  const runTimer = async () => { const next = timers.shift(); await next.fn(); await flush(); };
  return { button, server, notes, opened, announced, timers, stop, click, runTimer, context };
}

test('the button text follows the state of the install', () => {
  const context = makeContext();
  vm.runInContext(readBuilderModule('model_install_button.mjs'), context);
  context.args = { item: ITEM, cases: [null, downloading(30), entry({ job: { state: 'downloading', downloaded: 5, total: 0, error: '' } }), installed(), failed('x')] };
  const texts = vm.runInContext('args.cases.map((model) => installButtonText(args.item, model))', context);
  assert.equal(texts[0], 'Edit LoRA (instruction editing)');
  assert.equal(texts[1], 'Installing Edit LoRA (instruction editing)… 30% (click to cancel)');
  assert.equal(texts[2], 'Installing Edit LoRA (instruction editing)… (click to cancel)', 'no percentage without a known size');
  assert.equal(texts[3], 'Edit LoRA (instruction editing) · installed');
  assert.equal(texts[4], 'Edit LoRA (instruction editing) · try again');
});

test('it starts with the item label and the name it will be saved as', async () => {
  const h = wired();
  assert.equal(h.button.textContent, ITEM.label);
  await flush();
  assert.equal(h.button.textContent, ITEM.label);
  assert.match(h.button.title, /moss_image_edit_lora\.safetensors/);
  assert.deepEqual(h.server.calls, [[STATUS]], 'it asks once what is installed');
  assert.equal(h.timers.length, 0, 'nothing to follow');
});

test('a file that is already there shows as installed, with where it is', async () => {
  const h = wired({ models: [installed()] });
  await flush();
  assert.match(h.button.textContent, /· installed$/);
  assert.match(h.button.title, /moss_image_edit_lora\.safetensors/);
  await h.click();
  assert.equal(h.server.calls.some((call) => call[0] === INSTALL), false, 'it is not installed twice');
  assert.match(h.notes.at(-1)[0], /already installed/);
});

test('a click installs by id, follows the download and says when it is done', async () => {
  const h = wired({ installAnswer: [downloading(10)] });
  await flush();
  await h.click();
  const install = h.server.calls.find((call) => call[0] === INSTALL);
  assert.deepEqual(JSON.parse(JSON.stringify(install[1])), { ids: ['moss_edit_lora'] }, 'only an id goes to the server, never a url or a name');
  assert.match(h.button.textContent, /10%/);
  assert.equal(h.timers.length, 1);

  h.server.models = [downloading(60)];
  await h.runTimer();
  assert.match(h.button.textContent, /60%/);
  assert.equal(h.timers[0].ms, 1000, 'it follows a running download once a second');
  assert.equal(h.notes.length, 0);

  h.server.models = [installed()];
  await h.runTimer();
  assert.match(h.button.textContent, /· installed$/);
  assert.equal(h.timers.length, 0, 'it stops asking when the download is over');
  assert.deepEqual(h.notes, [['Edit LoRA (instruction editing) installed in C:/ComfyUI/models/loras as moss_image_edit_lora.safetensors.', false]]);
  assert.deepEqual(JSON.parse(JSON.stringify(h.announced)), [['moss_edit_lora']], 'the LoRA lists are told to reload');
  assert.deepEqual(h.opened, [], 'the browser download page was never opened');
});

test('a click while it downloads cancels', async () => {
  const h = wired({ models: [downloading(20)] });
  await flush();
  await h.click();
  const cancel = h.server.calls.find((call) => call[0] === CANCEL);
  assert.deepEqual(JSON.parse(JSON.stringify(cancel[1])), { ids: ['moss_edit_lora'] });
  assert.equal(h.server.calls.some((call) => call[0] === INSTALL), false);
});

test('an install that cannot start says why, and so does one that fails on the way', async () => {
  const early = wired({ installAnswer: [failed('ComfyUI has no \'loras\' model folder. Is ComfyUI running?')] });
  await flush();
  await early.click();
  assert.deepEqual(early.notes, [["Edit LoRA (instruction editing): ComfyUI has no 'loras' model folder. Is ComfyUI running?", true]]);
  assert.match(early.button.textContent, /· try again$/);

  const late = wired({ installAnswer: [downloading(5)] });
  await flush();
  await late.click();
  late.server.models = [failed('The download server answered 503 Service Unavailable.')];
  await late.runTimer();
  assert.deepEqual(late.notes, [['Edit LoRA (instruction editing): The download server answered 503 Service Unavailable.', true]]);
  assert.equal(late.button.title, 'The download server answered 503 Service Unavailable.');
});

test('a build without the installer falls back to the browser download and says what to rename it to', async () => {
  const h = wired({ installFails: true });
  await flush();
  await h.click();
  assert.deepEqual(h.opened, [ITEM.url]);
  assert.equal(h.notes[0][1], true);
  assert.match(h.notes[0][0], /could not be reached \(404 Not Found\)/);
  assert.match(h.notes[0][0], /save the file as moss_image_edit_lora\.safetensors/);
  assert.equal(h.timers.length, 0);
});

test('stopping the button ends its polling', async () => {
  const h = wired({ installAnswer: [downloading(10)] });
  await flush();
  await h.click();
  const before = h.server.calls.length;
  h.stop();
  await h.runTimer();
  assert.equal(h.server.calls.length, before, 'no more questions once it is stopped');
  assert.equal(h.timers.length, 0);
});

// The Models dialog with the real models.mjs and dialogs.mjs: Download models > Image Models > MOSS-Image.
function openImageModels(overrides = {}) {
  const calls = [];
  const context = makeContext({
    getJson: async (url) => { calls.push(['get', url]); return {}; },
    postJson: async (url, body) => { calls.push(['post', url, JSON.parse(JSON.stringify(body))]); return { models: [downloading(10)] }; },
    ...overrides,
  });
  loadDialog(context);
  context.args = { modelSelections: [] };
  vm.runInContext('showModelDownloadModal(args)', context);
  const all = () => walk(context.document.body);
  const button = (label) => all().find((node) => node.tag === 'button' && node.textContent === label);
  fire(button('Download models'), 'click');
  fire(button('Image Models'), 'click');
  return { context, calls, all, button };
}

test('in the Models dialog the Edit LoRA button installs through the Builder instead of opening the page', async () => {
  const { context, calls, button } = openImageModels();
  await flush();
  fire(button('Edit LoRA (instruction editing)'), 'click');
  await flush();
  assert.deepEqual(calls.filter((call) => call[0] === 'post'), [['post', INSTALL, { ids: ['moss_edit_lora'] }]]);
  assert.deepEqual(context.opened, [], 'no browser tab: the file would keep the name adapter_model.safetensors');
});

test('every other MOSS-Image button still opens its model page', async () => {
  const { context, calls, button } = openImageModels();
  await flush();
  fire(button('8-step Turbo LoRA'), 'click');
  fire(button('Flux2 VAE'), 'click');
  assert.deepEqual(context.opened, [
    'https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/resolve/main/lora.safetensors',
    'https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors',
  ]);
  assert.deepEqual(calls.filter((call) => call[0] === 'post'), []);
});

test('leaving the tab stops a running install from being followed', async () => {
  const { context, calls, button } = openImageModels();
  await flush();
  fire(button('Edit LoRA (instruction editing)'), 'click');
  await flush();
  assert.equal(context.timers.length, 1, 'the running download is being followed');
  fire(button('LTX'), 'click');
  const asked = calls.filter((call) => call[0] === 'get').length;
  for (const timer of context.timers.splice(0)) await timer();
  await flush();
  assert.equal(calls.filter((call) => call[0] === 'get').length, asked, 'the old tab asks nothing more');
  assert.equal(context.timers.length, 0);
});

test('the MOSS-Image folder note no longer tells people to rename the Edit LoRA', () => {
  const context = makeContext();
  vm.runInContext(readBuilderModule('models.mjs'), context);
  const hint = vm.runInContext('MODEL_FOLDER_HINTS["MOSS-Image"]', context);
  assert.match(hint, /Edit LoRA button installs/);
  assert.match(hint, /moss_image_edit_lora\.safetensors/);
  assert.doesNotMatch(hint, /Rename it to moss_image_edit_lora/);
});
