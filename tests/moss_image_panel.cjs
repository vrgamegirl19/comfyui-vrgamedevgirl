const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// The MOSS-Image tab of Models > Select models (web/music_video_builder/moss_image_panel.mjs), the "MOSS-Image (models)" choice in the
// Image tab's model list, and the Models dialog opening on that tab, driven against a small fake DOM with a real parent/child tree.

function detach(node) {
  const parent = node.parentElement;
  if (parent) parent.children.splice(parent.children.indexOf(node), 1);
  node.parentElement = null;
}

function makeNode(tag) {
  const node = {
    tag, style: {}, dataset: {}, attrs: {}, children: [], listeners: {}, value: '', textContent: '', parentElement: null, disabled: false,
    get nextSibling() {
      const parent = this.parentElement;
      return parent ? parent.children[parent.children.indexOf(this) + 1] || null : null;
    },
    append(...kids) {
      for (const kid of kids) {
        if (typeof kid === 'string') continue;
        detach(kid);
        kid.parentElement = this;
        this.children.push(kid);
      }
    },
    prepend(...kids) {
      for (const kid of kids.reverse()) {
        detach(kid);
        kid.parentElement = this;
        this.children.unshift(kid);
      }
    },
    insertBefore(child, reference) {
      detach(child);
      child.parentElement = this;
      const index = reference ? this.children.indexOf(reference) : -1;
      if (index < 0) this.children.push(child);
      else this.children.splice(index, 0, child);
    },
    replaceChildren(...kids) {
      for (const child of this.children) child.parentElement = null;
      this.children = [];
      this.append(...kids);
    },
    querySelector(selector) {
      const wanted = /^option\[value="(.*)"\]$/.exec(selector);
      return this.children.find((child) => child.tag === 'option' && wanted && child.value === wanted[1]) || null;
    },
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    setAttribute(key, value) { this.attrs[key] = value; },
    remove() { detach(this); this.removed = true; },
  };
  return node;
}

function fire(node, type) {
  for (const fn of node.listeners[type] || []) fn({ preventDefault() {}, stopPropagation() {} });
}

function fakeStorage(initial = {}) {
  const data = { ...initial };
  return { data, getItem: (key) => (key in data ? data[key] : null), setItem: (key, value) => { data[key] = String(value); } };
}

function makeContext() {
  const documentListeners = {};
  const body = makeNode('body');
  const context = vm.createContext({
    document: {
      createElement: makeNode,
      body,
      addEventListener(type, fn) { (documentListeners[type] ||= []).push(fn); },
      removeEventListener(type, fn) { documentListeners[type] = (documentListeners[type] || []).filter((item) => item !== fn); },
    },
    window: { open() {} },
    BUILDER_FONT_STACK: 'sans-serif',
    getJson: async () => ({}),
    postJson: async () => ({}),
    toast() {},
    makeEditorImageUrl: () => '',
    renderDmadModelsTab: () => () => {},
    makeButton: (label) => Object.assign(makeNode('button'), { textContent: label }),
    makeMiniButton: (label) => Object.assign(makeNode('button'), { textContent: label }),
    makeInput: (value = '', type = 'text') => Object.assign(makeNode('input'), { value, type }),
    makeSelect: (options, value) => {
      const select = Object.assign(makeNode('select'), { value });
      for (const item of options) select.append(Object.assign(makeNode('option'), { value: item.value ?? item }));
      return select;
    },
    makeField: (label, control) => { const field = makeNode('label'); field.append(control); return field; },
    makeSettingsSection: (title, children) => { const section = makeNode('details'); section.append(...children); return section; },
    makeSearchableLoraPicker: (value) => ({ wrapper: makeNode('div'), input: Object.assign(makeNode('input'), { value }), list: makeNode('div'), options: [], matches: [], activeIndex: -1 }),
    // The real one wires typing and the suggestion list; the test only needs the change callback to call as typing would.
    wireSearchablePicker: (picker, onChange) => { picker.onChange = onChange; },
  });
  vm.runInContext(readBuilderModule('moss_settings.mjs'), context);
  vm.runInContext(readBuilderModule('moss_image_panel.mjs'), context);
  context.documentListeners = documentListeners;
  return context;
}

const choices = (unets = ['moss_image_sft_bf16.safetensors'], clip = ['qwen3.5_9b_int8_convrot.safetensors'], vae = ['flux2-vae.safetensors']) => ({ unets, clip, vae });
const fetcher = ({ models = choices(), loras = ['[none]', 'moss_image_edit_lora.safetensors'] } = {}) => async (url) => {
  if (url.endsWith('/i2v_choices')) return models;
  if (url.endsWith('/lora_list')) return { loras };
  throw new Error(`unexpected ${url}`);
};

function openPanel({ fetchJson = fetcher(), saved } = {}) {
  const context = makeContext();
  const storage = fakeStorage(saved ? { vrgdg_moss_image_reference_settings: JSON.stringify(saved) } : {});
  context.args = { fetchJson, storage };
  const panel = vm.runInContext('createMossImagePanel(args)', context);
  const stored = () => JSON.parse(storage.data.vrgdg_moss_image_reference_settings || '{}');
  return { context, panel, storage, stored };
}

test('the tab is called MOSS-Image and its content has a place to go back to', () => {
  const { panel } = openPanel();
  assert.equal(panel.modelSelection.label, 'MOSS-Image');
  assert.equal(panel.modelSelection.content.parentElement, panel.holder, 'the Models dialog puts a tab back where it found it');
});

test('it opens on the saved settings', () => {
  const { panel } = openPanel({ saved: { unet_name: 'mine.safetensors', preset: 'custom', steps: 20, cfg: 2.5, width: 1280, height: 720, seed: 9, seed_mode: 'fixed' } });
  assert.equal(panel.pickers.unet.input.value, 'mine.safetensors');
  assert.equal(panel.controls.preset.value, 'custom');
  assert.equal(panel.controls.steps.value, '20');
  assert.equal(panel.controls.cfg.value, '2.5');
  assert.equal(panel.controls.width.value, '1280');
  assert.equal(panel.controls.seed.value, '9');
  assert.equal(panel.controls.seedMode.value, 'fixed');
});

test('opening the Builder saves nothing the user did not change', async () => {
  const { panel, storage } = openPanel();
  await panel.ready;
  assert.deepEqual(Object.keys(storage.data), []);
});

test('the pickers list what is installed, a subfolder spelling is adopted and a missing model stays as saved', async () => {
  const { panel } = openPanel({
    saved: { clip_name: 'not_installed.safetensors' },
    fetchJson: fetcher({ models: choices(['image\\moss_image_sft_bf16.safetensors', 'ltx\\video.safetensors'], ['other_clip.safetensors']) }),
  });
  await panel.ready;
  assert.equal(panel.pickers.unet.input.value, 'image\\moss_image_sft_bf16.safetensors');
  assert.equal(panel.pickers.unet.options.length, 2);
  assert.equal(panel.pickers.clip.input.value, 'not_installed.safetensors', 'it is never swapped for another model');
  assert.deepEqual([...panel.pickers.clip.options], ['other_clip.safetensors']);
});

test('the Edit LoRA note shows only when the LoRA list is known and the file is not in it', async () => {
  const missing = openPanel({ fetchJson: fetcher({ loras: ['[none]', 'something_else.safetensors'] }) });
  assert.equal(missing.panel.editLoraNote.style.display, 'none', 'nothing is said before the list is known');
  await missing.panel.ready;
  assert.equal(missing.panel.editLoraNote.style.display, 'block');
  assert.match(missing.panel.editLoraNote.textContent, /moss_image_edit_lora\.safetensors/);
  assert.match(missing.panel.editLoraNote.textContent, /v0\.42\.1/);

  const installed = openPanel({ fetchJson: fetcher({ loras: ['[none]', 'moss\\moss_image_edit_lora.safetensors'] }) });
  await installed.panel.ready;
  assert.equal(installed.panel.editLoraNote.style.display, 'none');
  assert.equal(installed.panel.pickers.editLora.input.value, 'moss\\moss_image_edit_lora.safetensors');

  missing.panel.pickers.editLora.input.value = '[none]';
  missing.panel.pickers.editLora.onChange();
  assert.equal(missing.panel.editLoraNote.style.display, 'none', 'choosing no Edit LoRA is fine');
  assert.equal(missing.stored().edit_lora_name, '[none]');
});

test('a ComfyUI that cannot be asked leaves the saved names alone', async () => {
  const { panel } = openPanel({ saved: { unet_name: 'mine.safetensors' }, fetchJson: async () => { throw new Error('offline'); } });
  await panel.ready;
  assert.equal(panel.pickers.unet.input.value, 'mine.safetensors');
  assert.equal(panel.editLoraNote.style.display, 'none');
});

test('a preset sets its LoRA, steps and CFG and is saved', () => {
  const { panel, stored } = openPanel({ saved: { preset: 'custom', steps: 30, cfg: 4, lora_name: 'x.safetensors' } });
  panel.controls.preset.value = 'quality';
  fire(panel.controls.preset, 'change');
  assert.equal(panel.pickers.lora.input.value, '[none]');
  assert.equal(panel.controls.steps.value, '50');
  assert.equal(panel.controls.cfg.value, '3');
  assert.deepEqual({ preset: stored().preset, steps: stored().steps, cfg: stored().cfg, lora: stored().lora_name }, { preset: 'quality', steps: 50, cfg: 3, lora: '[none]' });
  panel.controls.preset.value = 'turbo';
  fire(panel.controls.preset, 'change');
  assert.equal(stored().lora_name, 'moss_image_sft_8_steps_dmd_lora.safetensors');
  assert.equal(stored().steps, 8);
});

test('changing steps, CFG or the sampling LoRA by hand makes the preset Custom', () => {
  for (const change of [
    (panel) => { panel.controls.steps.value = '12'; fire(panel.controls.steps, 'input'); fire(panel.controls.steps, 'change'); },
    (panel) => { panel.controls.cfg.value = '2'; fire(panel.controls.cfg, 'input'); fire(panel.controls.cfg, 'change'); },
    (panel) => { panel.pickers.lora.input.value = 'other.safetensors'; panel.pickers.lora.onChange(); },
  ]) {
    const { panel, stored } = openPanel();
    assert.equal(panel.controls.preset.value, 'turbo');
    change(panel);
    assert.equal(panel.controls.preset.value, 'custom');
    assert.equal(stored().preset, 'custom');
  }
});

test('sizes become a multiple of 16 and the boxes show what was kept', () => {
  const { panel, stored } = openPanel();
  panel.controls.width.value = '1000';
  panel.controls.height.value = '50';
  fire(panel.controls.width, 'change');
  assert.equal(panel.controls.width.value, '1008');
  assert.equal(panel.controls.height.value, '256');
  assert.equal(stored().width, 1008);
  assert.equal(stored().height, 256);
});

test('every change is saved, on top of the settings the panel does not show', () => {
  const { panel, stored } = openPanel({ saved: { reference_only_setting: 'kept' } });
  panel.controls.seed.value = '77';
  panel.controls.seedMode.value = 'fixed';
  fire(panel.controls.seed, 'change');
  assert.equal(stored().seed, 77);
  assert.equal(stored().seed_mode, 'fixed');
  assert.equal(stored().reference_only_setting, 'kept');
  assert.match(panel.status.textContent, /Saved/);
  panel.pickers.unet.input.value = 'new_unet.safetensors';
  panel.pickers.unet.onChange();
  assert.equal(stored().unet_name, 'new_unet.safetensors');
});

test('the Reference Builder and the Thumbnail dialog read what the panel saved', () => {
  const { context, panel, storage } = openPanel();
  panel.controls.steps.value = '14';
  fire(panel.controls.steps, 'change');
  context.storage = storage;
  assert.equal(vm.runInContext('loadMossReferenceSettings(storage).steps', context), 14);
  assert.equal(vm.runInContext('mossReferencePayload("x", loadMossReferenceSettings(storage)).steps', context), 14);
});

function imageModelSelect(context) {
  const select = context.makeSelect([{ value: 'zimage' }, { value: 'flux_klein' }, { value: 'custom_image' }], 'zimage');
  select.dataset.activeMode = 'zimage';
  const switched = [];
  select.onchange = (event) => switched.push([select.value, event]);
  return { select, switched };
}

test('MOSS-Image (models) is in the Image tab model list, before + Custom', () => {
  const context = makeContext();
  const { select } = imageModelSelect(context);
  context.args = { select };
  vm.runInContext('addMossImageOption(args.select, () => {})', context);
  assert.deepEqual(select.children.map((option) => option.value), ['zimage', 'flux_klein', 'moss_image_settings', 'custom_image']);
  assert.equal(select.children[2].textContent, 'MOSS-Image (models)');
});

test('choosing it opens the models and puts the list back, and the other choices still switch the mode', () => {
  const context = makeContext();
  const { select, switched } = imageModelSelect(context);
  let opened = 0;
  context.args = { select, open: () => { opened += 1; } };
  vm.runInContext('addMossImageOption(args.select, args.open)', context);
  select.value = 'moss_image_settings';
  select.onchange({ type: 'change' });
  assert.equal(opened, 1);
  assert.equal(select.value, 'zimage', 'the list is back on the mode in use');
  assert.equal(switched.length, 0, 'it is not a mode, so no mode is switched');
  select.value = 'flux_klein';
  select.onchange({ type: 'change' });
  assert.equal(opened, 1);
  assert.equal(switched.length, 1);
  assert.equal(switched[0][0], 'flux_klein');
});

// The Models dialog itself, with the real models.mjs and dialogs.mjs.
function openDialog(select) {
  const context = makeContext();
  vm.runInContext(readBuilderModule('models.mjs'), context);
  vm.runInContext(readBuilderModule('dialogs.mjs'), context);
  const first = makeNode('div');
  const firstHolder = makeNode('div');
  firstHolder.append(first);
  const { panel } = (() => {
    context.storage = fakeStorage();
    context.args = { fetchJson: fetcher(), storage: context.storage };
    return { panel: vm.runInContext('createMossImagePanel(args)', context) };
  })();
  context.args = { modelSelections: [{ label: 'Video', content: first }, panel.modelSelection], select };
  vm.runInContext('showModelDownloadModal(args)', context);
  const all = [];
  const walk = (node) => { all.push(node); node.children.forEach(walk); };
  walk(context.document.body);
  const find = (predicate) => all.find(predicate);
  return { context, panel, first, firstHolder, find };
}

test('the Models dialog opens on the MOSS-Image tab when asked to', () => {
  const { panel, first, firstHolder, find } = openDialog('MOSS-Image');
  const title = find((node) => node.tag === 'div' && node.textContent === 'Select models');
  assert.ok(title, 'the title says Select models');
  const body = find((node) => node.style.cssText === 'max-width:800px;');
  assert.equal(body.children[0], panel.modelSelection.content, 'the MOSS-Image tab is showing');
  assert.equal(panel.modelSelection.content.style.display, 'flex');
  assert.equal(first.parentElement, firstHolder, 'the other tab was not touched');
  const selection = body.parentElement;
  assert.equal(selection.style.display, 'block');
});

test('closing the dialog puts the MOSS-Image tab back where it was', () => {
  const { panel, find } = openDialog('MOSS-Image');
  find((node) => node.tag === 'button' && node.textContent === 'Close').onclick();
  assert.equal(panel.modelSelection.content.parentElement, panel.holder);
});

test('without a tab named, or with one that does not exist, the dialog opens as it always did', () => {
  for (const select of ['', 'No such tab']) {
    const { find } = openDialog(select);
    const body = find((node) => node.style.cssText === 'max-width:800px;');
    assert.equal(body.children.length, 0, 'no tab is showing yet');
    assert.notEqual(body.parentElement.style.display, 'block', 'the Select models side is not open (it starts as display:none)');
    assert.match(body.parentElement.style.cssText, /^display:none;/);
  }
});
