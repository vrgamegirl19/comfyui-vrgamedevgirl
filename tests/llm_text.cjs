const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// Labels and messages written with "Gemma" read as the selected LLM Runner (controls.mjs llmText / bindLlmText / toast,
// llm_runner.mjs createLlmRunner).
function fixture(runner = 'builtin') {
  const sent = [], reported = [];
  const element = () => ({ style: {}, textContent: '', title: '', setAttribute() {}, append() {}, appendChild() {}, remove() {} });
  const c = vm.createContext({
    app: {}, BUILDER_FONT_STACK: 'sans-serif', BUY_ME_A_COFFEE_URL: '', reportProjectError: (message) => reported.push(message),
    document: { createElement: element, body: { appendChild() {} } },
    window: { dispatchEvent: (event) => sent.push(event.detail) },
    CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init.detail; } },
    setTimeout: () => 0,
  });
  vm.runInContext(readBuilderModule('controls.mjs'), c);
  vm.runInContext(readBuilderModule('llm_runner.mjs'), c);
  const state = { textGemmaRunner: runner };
  const buttons = Object.fromEntries(['gemmaT2IAllButton', 'gemmaVideoAllButton', 'createT2IButton', 'ernieCreateT2IButton',
    'krea2TwoPassCreateT2IButton', 'createFluxPromptButton', 'createNBPromptButton', 'flowGptCreatePromptButton',
    'zEnhanceGemmaButton', 'createI2VButton'].map((name) => [name, element()]));
  const llm = c.createLlmRunner({ state, currentVideoMode: () => 'i2v', ...buttons });
  return { c, state, llm, sent, reported, buttons, element };
}

test('until a runner is set, text is shown as written, and anything that is not text is left alone', () => {
  const sent = [];
  const c = vm.createContext({ app: {}, BUILDER_FONT_STACK: '', BUY_ME_A_COFFEE_URL: '', reportProjectError() {} });
  vm.runInContext(readBuilderModule('controls.mjs'), c);
  assert.equal(c.llmText('Gemma Describe'), 'Gemma Describe');
  assert.equal(c.llmText(42), 42);
  assert.equal(c.llmText(undefined), undefined);
  assert.equal(sent.length, 0);
});

test('Gemma Local keeps the wording as written', () => {
  const { c } = fixture('builtin');
  assert.equal(c.llmText('Gemma Describe'), 'Gemma Describe');
  assert.equal(c.llmText('Gemma returned an empty I2V prompt.'), 'Gemma returned an empty I2V prompt.');
});

test('every other runner reads by its own name', () => {
  const read = (runner, text) => fixture(runner).c.llmText(text);
  assert.equal(read('lm_studio', 'Gemma Describe'), 'LM Studio Describe');
  assert.equal(read('lm_studio', 'Auto Map Locations with Gemma'), 'Auto Map Locations with LM Studio');
  assert.equal(read('llm_api', 'Gemma T2I'), 'LLM API T2I');
  assert.equal(read('qwen_local', 'Gemma Enhance Prompt'), 'Qwen Local Enhance Prompt');
  assert.equal(read('own_server', 'Gemma created T2I from notes.'), 'Custom Server created T2I from notes.');
  assert.equal(read('lm_studio', 'Use vision Gemma to describe character reference images.'), 'Use LM Studio vision to describe character reference images.');
  assert.equal(read('lm_studio', 'Gemma Local T2I'), 'LM Studio T2I');
});

test('a model name and the built-in Gemma4 drafts are not renamed', () => {
  const { c } = fixture('lm_studio');
  assert.equal(c.llmText('LTX 2.5 distilled transformer, Gemma 4 text encoders'), 'LTX 2.5 distilled transformer, Gemma 4 text encoders');
  assert.equal(c.llmText('Gemma4 draft is ready.'), 'Gemma4 draft is ready.');
  assert.equal(c.llmText('Gemma-3 is not renamed either, but Gemma is.'), 'Gemma-3 is not renamed either, but LM Studio is.');
});

test('toasts, and the error log line of an error toast, name the selected runner', () => {
  const { c, sent, reported } = fixture('llm_api');
  c.toast('Gemma returned an empty reference description.', true);
  c.toast('No Gemma here? Fine.');
  assert.deepEqual(sent.map((detail) => detail.message), ['LLM API returned an empty reference description.', 'No LLM API here? Fine.']);
  assert.equal(sent[0].isError, true);
  assert.deepEqual(reported, ['LLM API returned an empty reference description.'], 'only the error toast is logged, with the runner name');
});

test('a button follows the runner when it changes and leaves a busy label alone', () => {
  const { c, state, llm } = fixture('builtin');
  const describe = c.makeButton('Gemma Describe', 'primary');
  const plain = c.makeButton('Save');
  assert.equal(describe.textContent, 'Gemma Describe');
  assert.equal(plain.textContent, 'Save');

  state.textGemmaRunner = 'lm_studio';
  llm.updatePromptRunnerButtonLabels();
  assert.equal(describe.textContent, 'LM Studio Describe');
  assert.equal(plain.textContent, 'Save');

  describe.textContent = 'Working...';
  state.textGemmaRunner = 'llm_api';
  llm.updatePromptRunnerButtonLabels();
  assert.equal(describe.textContent, 'Working...', 'a label something else changed is not overwritten');

  c.bindLlmText(describe, 'textContent', 'Gemma Describe');
  assert.equal(describe.textContent, 'LLM API Describe');
  state.textGemmaRunner = 'qwen_local';
  llm.updatePromptRunnerButtonLabels();
  assert.equal(describe.textContent, 'Qwen Local Describe', 'binding the same element again keeps one entry');
});

test('tooltips and placeholders follow the runner too', () => {
  const { c, state, llm, element } = fixture('lm_studio');
  const box = element();
  c.bindLlmText(box, 'title', 'Video notes. Gemma uses this for camera motion.');
  c.bindLlmText(box, 'placeholder', 'Scene notes for Gemma...');
  assert.equal(box.title, 'Video notes. LM Studio uses this for camera motion.');
  assert.equal(box.placeholder, 'Scene notes for LM Studio...');
  state.textGemmaRunner = 'builtin';
  llm.updatePromptRunnerButtonLabels();
  assert.equal(box.title, 'Video notes. Gemma uses this for camera motion.');
  assert.equal(box.placeholder, 'Scene notes for Gemma...');
});

test('the main prompt buttons still get the runner name', () => {
  const { state, llm, buttons } = fixture('lm_studio');
  llm.updatePromptRunnerButtonLabels();
  assert.equal(buttons.createT2IButton.textContent, 'LM Studio T2I');
  assert.equal(buttons.createI2VButton.textContent, 'LM Studio I2V');
  state.textGemmaRunner = 'builtin';
  llm.updatePromptRunnerButtonLabels();
  assert.equal(buttons.createT2IButton.textContent, 'Gemma Local T2I');
});
