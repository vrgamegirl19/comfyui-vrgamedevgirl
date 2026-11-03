const assert = require('node:assert/strict');
const { test } = require('node:test');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

function reportedLoras(prompt) {
  const source = readBuilderModule('video_render.mjs');
  const start = source.indexOf('      const exactModelChainLoras =');
  const end = source.indexOf('      const formatExactLoras =', start);
  assert.ok(start >= 0 && end > start);
  const c = vm.createContext({ built: { prompt }, twoPass: true, threePass: false });
  vm.runInContext(source.slice(start, end) + '\nresult = exactTwoPassLoras;', c);
  return JSON.parse(JSON.stringify(c.result));
}

test('manual-sigma enhancer reports DMAD and Refiner from the actual pass-2 guider', () => {
  const result = reportedLoras({
    '124': { inputs: { model: ['141', 0] } },
    '141': { class_type: 'DiffusionModelLoaderKJ', inputs: {} },
    '192': { class_type: 'ManualSigmas', inputs: { sigmas: '0.7, 0.494118, 0' } },
    '193': { class_type: 'BasicGuider', inputs: { model: ['patch', 0] } },
    patch: { class_type: 'BlockSparseAttention', inputs: { model: ['refiner', 0] } },
    refiner: { class_type: 'LoraLoaderModelOnly', inputs: { model: ['207', 0], lora_name: 'refiner.safetensors', strength_model: 1 } },
    '207': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['141', 0], lora_name: 'dmad.safetensors', strength_model: 1 } },
  });
  assert.deepEqual(result.pass1, []);
  assert.deepEqual(result.pass2.map(lora => [lora.name, lora.strength]),
    [['dmad.safetensors', 1], ['refiner.safetensors', 1]]);
});

test('normal two-pass Turbo and optional pass-1 LoRAs are still reported', () => {
  const prompt = {
    '124': { inputs: { model: ['character', 0] } },
    character: { class_type: 'LoraLoaderModelOnly', inputs: { model: ['141', 0], lora_name: 'character.safetensors', strength_model: 0.6 } },
    '141': { class_type: 'DiffusionModelLoaderKJ', inputs: {} },
    '192': { class_type: 'BasicScheduler', inputs: { model: ['207', 0] } },
    '193': { class_type: 'BasicGuider', inputs: { model: ['207', 0] } },
    '207': { class_type: 'LoraLoaderModelOnly', inputs: { model: ['141', 0], lora_name: 'turbo.safetensors', strength_model: 1 } },
  };
  const result = reportedLoras(prompt);
  assert.deepEqual(result.pass1.map(lora => lora.name), ['character.safetensors']);
  assert.deepEqual(result.pass2.map(lora => lora.name), ['turbo.safetensors']);
});
