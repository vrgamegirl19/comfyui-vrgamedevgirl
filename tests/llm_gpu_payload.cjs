const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

// The runner settings every LLM request carries (web/music_video_builder/llm_runner.mjs): the GPU of Gemma Local and Qwen Local rides along with the
// GPU layers, so each writer (prompts, enhance, reference description, storyboard, wizard) runs the model on the GPU chosen in the LLM Runner.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('prompt_text.mjs'), context);
vm.runInContext(functionSource(readBuilderModule('comfy_api.mjs'), 'normalizeOwnServerTimeoutMinutes'), context);
vm.runInContext(readBuilderModule('llm_runner.mjs'), context);
const payload = (state) => vm.runInContext('(state) => createLlmRunner({ state }).textGemmaRunnerPayload()', context)(state);

test('a request carries the GPU chosen for the local model', () => {
  assert.equal(payload({ textGemmaRunner: 'builtin', gemmaGpuDevice: 1 }).gpu_device, 1);
  assert.equal(payload({ textGemmaRunner: 'qwen_local', gemmaGpuDevice: 0 }).gpu_device, 0);
});

test('no choice, or one that is not a GPU number, is automatic', () => {
  assert.equal(payload({ textGemmaRunner: 'builtin' }).gpu_device, -1);
  assert.equal(payload({ textGemmaRunner: 'builtin', gemmaGpuDevice: 'auto' }).gpu_device, -1);
  assert.equal(payload({ textGemmaRunner: 'builtin', gemmaGpuDevice: -3 }).gpu_device, -1);
});

test('the GPU layers still travel with it', () => {
  const sent = payload({ textGemmaRunner: 'builtin', gemmaGpuLayers: 20, gemmaGpuDevice: 1 });
  assert.equal(sent.n_gpu_layers, 20);
  assert.equal(sent.gpu_device, 1);
});
