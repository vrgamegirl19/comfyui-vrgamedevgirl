const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const source = functionSource(readBuilderModule('reference_mapping_tools.mjs'), 'createWizardMappingTools');

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = { cssText: '' }; this.hidden = false; }
  append(...children) {
    for (const child of children) {
      child.remove(); child.parent = this; this.children.push(child);
    }
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }
  replaceChildren(...children) { this.children.forEach((child) => { child.parent = null; }); this.children = []; this.append(...children); }
  contains(target) { return this === target || this.children.some((child) => child.contains(target)); }
}

test('Wizard mapping tools reuse the existing actions and keep their handlers', () => {
  const context = vm.createContext({
    document: { createElement: (tag) => new Element(tag) },
    makeField: (_label, control) => { const field = new Element('label'); field.append(control); return field; },
  });
  vm.runInContext(`${source}\nglobalThis.create = createWizardMappingTools;`, context);
  const original = new Element('original');
  const actions = ['assignScenes', 'extractLocations', 'autoMapLocations'].map(() => new Element('button'));
  const theme = new Element('input');
  original.append(...actions, theme);
  let runs = 0;
  actions[0].onclick = () => { runs++; };
  const { card, render } = context.create({
    cardStyle: 'display:flex', assignScenes: actions[0], extractLocations: actions[1],
    autoMapLocations: actions[2], locationStyleTheme: theme,
  });
  render();
  assert.ok(actions.every((button) => card.contains(button)));
  assert.ok(card.contains(theme));
  actions[0].onclick();
  render();
  actions[0].onclick();
  assert.equal(runs, 2);
  actions[1].hidden = true;
  render();
  assert.equal(card.contains(actions[1]), false);
  assert.equal(card.contains(actions[2]), true);
});
