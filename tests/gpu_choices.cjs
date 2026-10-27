const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

// The GPU list of the LLM Runner (web/music_video_builder/gpu_choices.mjs) and the number rules (prompt_text.mjs). The dialog itself is checked in the app.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('gpu_choices.mjs'), context);
vm.runInContext(functionSource(readBuilderModule('prompt_text.mjs'), 'normalizeGemmaGpuDevice'), context);
const helper = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));

const rtx5090 = { number: 0, name: 'NVIDIA GeForce RTX 5090', total_gb: 31.8, free_gb: 25.4, slot: 0, visible: true, comfyui_gpu: true, label: 'GPU 0: NVIDIA GeForce RTX 5090 (31.8 GB, 25.4 GB free) - ComfyUI renders here' };
const rtx4090Hidden = { number: null, name: 'NVIDIA GeForce RTX 4090', total_gb: 24, free_gb: 15, slot: 1, visible: false, comfyui_gpu: false, label: 'NVIDIA GeForce RTX 4090 (24.0 GB, 15.0 GB free) - not visible to ComfyUI' };
const rtx4090First = { number: 0, name: 'NVIDIA GeForce RTX 4090', total_gb: 24, free_gb: 15, slot: 1, visible: true, comfyui_gpu: true, label: 'GPU 0: NVIDIA GeForce RTX 4090 (24.0 GB, 15.0 GB free) - ComfyUI renders here' };
const rtx5090Second = { number: 1, name: 'NVIDIA GeForce RTX 5090', total_gb: 31.8, free_gb: 25.4, slot: 0, visible: true, comfyui_gpu: false, label: 'GPU 1: NVIDIA GeForce RTX 5090 (31.8 GB, 25.4 GB free)' };

test('the GPU number is automatic unless it is a whole number from 0 up', () => {
  const normalize = helper('normalizeGemmaGpuDevice');
  for (const value of [undefined, null, '', 'auto', -1, -5, true, false, 'abc', NaN]) assert.equal(normalize(value), -1, String(value));
  assert.deepEqual([0, 1, '1', 2.4, 500].map(normalize), [0, 1, 1, 2, 63]);
});

test('a ComfyUI that cannot list its GPUs offers automatic and says how to get the list', () => {
  const choices = plain(helper('gpuChoices')(null, -1));
  assert.deepEqual(choices.options.map((o) => o.value), ['-1']);
  assert.match(choices.details, /Update the comfyui-vrgamedevgirl pack/);
  assert.equal(choices.selected, '-1');
  assert.deepEqual(plain(helper('gpuChoices')({}, -1)).options.map((o) => o.value), ['-1']);
});

test('a computer without an NVIDIA card says the model runs on the CPU', () => {
  assert.match(helper('gpuChoices')({ gpus: [] }, -1).details, /runs on the CPU/);
});

test('a hidden card is listed but cannot be chosen, and the launcher note comes with it', () => {
  const note = 'ComfyUI can only see 1 of the 2 GPUs (CUDA_VISIBLE_DEVICES is "1").';
  const choices = plain(helper('gpuChoices')({ gpus: [rtx5090, rtx4090Hidden], note }, -1));
  assert.deepEqual(choices.options.map((o) => [o.value, o.disabled]), [['-1', false], ['0', false], ['hidden-1', true]]);
  assert.equal(choices.options[1].label, rtx5090.label);
  assert.match(choices.details, /GPU 0: NVIDIA GeForce RTX 5090, 31\.8 GB, 25\.4 GB free - ComfyUI renders here\n/);
  assert.match(choices.details, /Hidden from ComfyUI: NVIDIA GeForce RTX 4090, 24 GB, 15 GB free\s+\(Task Manager and nvidia-smi: GPU 1\)/);
  assert.ok(choices.details.includes(note));
});

test('each card shows ComfyUI\'s number and the number Task Manager uses when they differ', () => {
  const choices = plain(helper('gpuChoices')({ gpus: [rtx4090First, rtx5090Second], note: '' }, 1));
  assert.deepEqual(choices.options.map((o) => o.value), ['-1', '0', '1']);
  assert.match(choices.details, /GPU 0: NVIDIA GeForce RTX 4090, 24 GB, 15 GB free - ComfyUI renders here {2}\(Task Manager and nvidia-smi: GPU 1\)/);
  assert.match(choices.details, /GPU 1: NVIDIA GeForce RTX 5090, 31\.8 GB, 25\.4 GB free {2}\(Task Manager and nvidia-smi: GPU 0\)/);
  assert.equal(choices.selected, '1');
});

test('a card whose numbers agree has no Task Manager remark', () => {
  assert.doesNotMatch(helper('gpuCardLine')(rtx5090), /Task Manager/);
});

test('a saved choice that is not available stays on the list, so saving cannot drop it by accident', () => {
  const choices = plain(helper('gpuChoices')({ gpus: [rtx5090, rtx4090Hidden] }, 1));
  assert.deepEqual(choices.options.map((o) => o.value), ['-1', '0', 'hidden-1', '1']);
  assert.equal(choices.options[3].label, 'GPU 1 (not visible to ComfyUI now)');
  assert.equal(choices.selected, '1');
  assert.match(choices.details, /The saved choice, GPU 1, is not visible to ComfyUI now/);
  const old = plain(helper('gpuChoices')(null, 1));
  assert.deepEqual(old.options.map((o) => o.value), ['-1', '1'], 'an old ComfyUI keeps the choice too');
  assert.doesNotMatch(old.details, /saved choice/);
});

test('a saved automatic choice or a number that is listed adds nothing', () => {
  assert.equal(helper('gpuChoices')({ gpus: [rtx5090] }, -1).options.length, 2);
  assert.equal(helper('gpuChoices')({ gpus: [rtx5090] }, 0).options.length, 2);
});
