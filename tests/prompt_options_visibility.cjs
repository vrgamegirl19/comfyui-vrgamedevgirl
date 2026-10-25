const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.value = '';
  }
  append(...children) { this.children.push(...children); }
  addEventListener() {}
  remove() {}
}

function fixture(engine, mode, pipeline = 'standard', imageModel = 'zimage') {
  const document = { createElement: tag => new Element(tag), body: new Element('body'),
    createTextNode: text => Object.assign(new Element('#text'), { textContent: text }) };
  const state = { projectVideoEngine: engine, imageModelMode: imageModel,
    miniMaxH3Settings: { video_mode: mode, pipeline } };
  const calls = [];
  const c = vm.createContext({ document, BUILDER_FONT_STACK: 'sans-serif' });
  for (const name of ['makeButton', 'makeCheckbox', 'makeInput', 'makeField', 'normalizeProjectVideoEngine']) {
    vm.runInContext(functionSource(readBuilderModule('controls.mjs'), name), c);
  }
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  vm.runInContext(readBuilderModule('prompt_creators.mjs'), c);
  const creators = c.createPromptCreators({ state, currentVideoMode: () => mode,
    editFinalPromptList: kind => calls.push(['edit', kind]),
    countPromptFindReplaceMatches: (text, kinds) => {
      calls.push(['preview', Array.from(kinds)]);
      return { matches: 1, fields: 1, scenes: 1 };
    } });
  const nodes = node => [node, ...node.children.filter(child => child instanceof Element).flatMap(nodes)];
  const open = () => {
    document.body.children = [];
    creators.openPromptOptionsModal();
    return nodes(document.body);
  };
  return { state, calls, open };
}

for (const [engine, mode, image, structured] of [
  ['minimax_h3', 'reference_to_video', false, true],
  ['minimax_h3', 'text_to_video', false, false],
  ['minimax_h3', 'video_to_video', false, false],
  ['minimax_h3', 'image_to_video', true, false],
  ['minimax_h3', 'image_reference_to_video', true, true],
  ['ltx', 'i2v', true, false],
  ['ltx', 't2v', false, false],
  ['ltx', 'rtv', false, false],
  ['ltx', 'ingredients', false, false],
  ['ltx', 'flf', true, false],
  ['ltx', 'id_lora', true, false],
  ['ltx', 'import', false, false],
]) {
  test(`${engine} ${mode}: shows relevant tools and limits replacement to those prompts`, () => {
    const f = fixture(engine, mode);
    const elements = f.open();
    const texts = elements.map(node => node.textContent);
    const mini = engine === 'minimax_h3';
    const video = mini || mode !== 'import';
    assert.equal(texts.includes('Image'), image);
    assert.equal(texts.includes('Create Concept Prompts'), image);
    assert.equal(texts.includes('Create Motion Notes'), video && !mini);
    assert.equal(texts.includes('Edit MiniMax H3 Prompts'), mini);
    assert.equal(texts.includes('MiniMax H3 prompt validation'), mini);
    assert.equal(texts.includes('Use Structured outputs'), structured);
    assert.equal(texts.includes('Do not add lyric lines to prompt'), mini);
    if (image || video) {
      elements.find(node => node.placeholder?.startsWith('Find exact')).value = 'the woman';
      elements.find(node => node.textContent === 'Preview Matches').onclick();
      assert.deepEqual(f.calls[0], ['preview', [
        ...(image ? ['t2i'] : []), ...(mini ? ['minimax'] : video ? ['i2v'] : []),
      ]]);
      const grid = elements.find(node => node.children.some(child => child.children?.some(heading => heading.textContent === 'Video')));
      assert.ok(grid.style.cssText.includes(`repeat(${image && video ? 2 : 1},`));
    } else {
      assert.ok(!texts.includes('Find / Replace Prompts'));
    }
  });
}

test('RefMod hides image tools even with stale image mode settings', () => {
  const f = fixture('minimax_h3', 'image_to_video', 'refmod');
  const texts = f.open().map(node => node.textContent);
  assert.ok(!texts.includes('Image'));
  assert.ok(texts.includes('Use Structured outputs'));
});

test('reopening reads the current engine and preserves prompt editor routing', () => {
  const f = fixture('minimax_h3', 'reference_to_video');
  f.open().find(node => node.textContent === 'Edit MiniMax H3 Prompts').onclick();
  assert.deepEqual(f.calls.pop(), ['edit', 'minimax']);
  f.state.projectVideoEngine = 'ltx';
  const elements = f.open();
  assert.ok(!elements.some(node => node.textContent === 'Edit MiniMax H3 Prompts'));
});

test('LTX text-to-video labels and actions use the selected mode', () => {
  const f = fixture('ltx', 't2v');
  const elements = f.open();
  elements.find(node => node.textContent === 'Edit Text to Video Prompts').onclick();
  assert.deepEqual(f.calls.pop(), ['edit', 'i2v']);
  assert.ok(!elements.some(node => /Image to Video/.test(node.textContent || '')));
});
