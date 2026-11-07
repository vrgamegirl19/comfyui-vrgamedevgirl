const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// A small fake DOM with a real parent/child tree, and a vm context that holds the Builder's Models dialog code
// (moss_settings.mjs, moss_image_panel.mjs, model_install_button.mjs, models.mjs, dialogs.mjs). Shared by the tests of the MOSS-Image
// tab and of the download buttons that install themselves.

function detach(node) {
  const parent = node.parentElement;
  if (parent) parent.children.splice(parent.children.indexOf(node), 1);
  node.parentElement = null;
}

function makeNode(tag) {
  return {
    tag, style: {}, dataset: {}, attrs: {}, children: [], listeners: {}, value: '', textContent: '', parentElement: null, disabled: false, title: '',
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
}

function fire(node, type) {
  for (const fn of node.listeners[type] || []) fn({ preventDefault() {}, stopPropagation() {} });
}

function fakeStorage(initial = {}) {
  const data = { ...initial };
  return { data, getItem: (key) => (key in data ? data[key] : null), setItem: (key, value) => { data[key] = String(value); } };
}

// Every node under `root`, root first.
function walk(root) {
  const all = [];
  const visit = (node) => { all.push(node); node.children.forEach(visit); };
  visit(root);
  return all;
}

// `overrides` replace the globals the code reaches for (getJson, postJson, toast, window...). The returned context also has:
// `opened` (urls given to window.open), `timers` (functions given to window.setTimeout, not run until the test runs them),
// `events` (events given to window.dispatchEvent).
function makeContext(overrides = {}) {
  const documentListeners = {};
  const body = makeNode('body');
  const opened = [];
  const timers = [];
  const events = [];
  class CustomEvent {
    constructor(type, init = {}) { this.type = type; this.detail = init.detail; }
  }
  const context = vm.createContext({
    document: {
      createElement: makeNode,
      body,
      addEventListener(type, fn) { (documentListeners[type] ||= []).push(fn); },
      removeEventListener(type, fn) { documentListeners[type] = (documentListeners[type] || []).filter((item) => item !== fn); },
    },
    window: {
      open: (url) => opened.push(url),
      setTimeout: (fn) => timers.push(fn),
      clearTimeout() {},
      dispatchEvent: (event) => events.push(event),
    },
    CustomEvent,
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
    // The real one wires typing and the suggestion list; the tests only need the change callback to call as typing would.
    wireSearchablePicker: (picker, onChange) => { picker.onChange = onChange; },
    ...overrides,
  });
  vm.runInContext(readBuilderModule('moss_settings.mjs'), context);
  vm.runInContext(readBuilderModule('moss_image_panel.mjs'), context);
  context.opened = opened;
  context.timers = timers;
  context.events = events;
  context.documentListeners = documentListeners;
  return context;
}

// The Models dialog itself, with the real models.mjs, model_install_button.mjs and dialogs.mjs.
function loadDialog(context) {
  vm.runInContext(readBuilderModule('models.mjs'), context);
  vm.runInContext(readBuilderModule('model_install_button.mjs'), context);
  vm.runInContext(readBuilderModule('dialogs.mjs'), context);
}

module.exports = { fakeStorage, fire, loadDialog, makeContext, makeNode, walk };
