const { functionSource, readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');

const source = readBuilderSource();

test('wizard storyboard prompting registers the Storyboard pipeline once without opening its UI', async () => {
  const opened = [];
  const runner = async (scene) => ({ prompt: `prompt for ${scene.id}` });
  const c = {
    storyboardPipeline: { runner: null },
    openStoryboardBuilderFromProject(options) {
      opened.push(options);
      c.storyboardPipeline.runner = runner;
    },
  };
  vm.createContext(c);
  vm.runInContext(functionSource(source, 'storyboardPromptPipeline'), c);
  assert.equal(await c.storyboardPromptPipeline()({ id: 'a' }).then((data) => data.prompt), 'prompt for a');
  assert.equal(c.storyboardPromptPipeline(), runner);
  assert.equal(JSON.stringify(opened), JSON.stringify([{ registerPromptPipelineOnly: true }]));
});

test('wizard bridge uses the registered Storyboard pipeline', () => {
  const wizard = readBuilderModule('wizard_bridge.mjs');
  assert.match(wizard, /await storyboardPromptPipeline\(\)\(sceneForPrompt, \{/);
  assert.doesNotMatch(wizard, /createStoryboardVideoPromptViaBuilder/);
});

function locationFixture(response) {
  const calls = { posts: [], toasts: [], saves: [], updated: 0 };
  const c = {
    calls,
    state: { textGemmaRunner: 'builtin', gemmaContextLimit: 8192, gemmaOutputTokenLimit: 1024 },
    t2iTextGemmaModelSelect: { value: 'gemma.gguf' }, gemmaModelSelect: { value: '' },
    i2vTextGemmaModelSelect: { value: '' }, i2vGemmaModelSelect: { value: '' },
    toast: (message, isError) => calls.toasts.push([message, Boolean(isError)]),
    createProgressWindow: () => ({ set() {}, close() {} }),
    gemmaRunnerLine: () => '',
    textGemmaRunnerPayload: () => ({}),
    normalizeGemmaContextLimit: (value) => value,
    normalizeOutputTokenLimit: (value) => value,
    postJson: async (url, payload) => { calls.posts.push([url, payload]); return response; },
    autoSaveSessionQuiet: async (label) => calls.saves.push(label),
  };
  vm.createContext(c);
  vm.runInContext(functionSource(source, 'createDetailedLocationDescriptionWithGemma'), c);
  return c;
}

test('detailed location description updates the location and refreshes the caller', async () => {
  const c = locationFixture({ text: 'A rain-soaked neon alley at night.' });
  const location = { name: 'Alley', description: 'neon alley' };
  const button = { textContent: 'Detailed Description', disabled: false };
  await c.createDetailedLocationDescriptionWithGemma(location, button, () => { c.calls.updated += 1; });
  assert.equal(location.description, 'A rain-soaked neon alley at night.');
  assert.equal(c.calls.updated, 1);
  assert.equal(c.calls.posts[0][1].target, 'location_description_detail');
  assert.deepEqual(c.calls.saves, ['Detailed location description: Alley']);
  assert.equal(button.disabled, false);
});

test('both reference builders call the shared detailed location description', () => {
  assert.match(readBuilderModule('ingredients_builder.mjs'), /createDetailedLocationDescriptionWithGemma\(locationById\(location\.id\) \|\| location, detailedDescription, renderAll\)/);
  assert.match(readBuilderModule('reference_locations.mjs'), /createDetailedLocationDescriptionWithGemma\(location, detailedDescription, renderAll\)/);
  assert.equal(source.split('async function createDetailedLocationDescriptionWithGemma(').length - 1, 1);
});
