const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

// What the Builder says when ComfyUI refuses a workflow at queue time: the node, the input and the value, not just "failed validation".
function load() {
  const context = vm.createContext({});
  vm.runInContext(functionSource(readBuilderModule('comfy_api.mjs'), 'formatNodeErrors'), context);
  return context;
}

test('a value that is not in the list names the node, the input and what it got', () => {
  const c = load();
  const text = c.formatNodeErrors({
    1068: { class_type: 'UNETLoader', errors: [{ type: 'value_not_in_list', message: 'Value not in list', extra_info: { input_name: 'unet_name', received_value: 'flux/x.safetensors' } }] },
    1064: { class_type: 'VAELoader', errors: [{ message: 'Value not in list', extra_info: { input_name: 'vae_name', received_value: 'flux/ae.safetensors' } }] },
  });
  assert.equal(text, [
    'Node 1064: VAELoader | vae_name: Value not in list (got "flux/ae.safetensors")',
    'Node 1068: UNETLoader | unet_name: Value not in list (got "flux/x.safetensors")',
  ].join('\n'));
});

test('errors without the extra info keep the older one line form, and no errors give nothing', () => {
  const c = load();
  assert.equal(c.formatNodeErrors({ 7: { class_type: 'KSampler', errors: [{ message: 'Required input is missing' }, 'plain text'] } }),
    'Node 7: KSampler | Required input is missing | plain text');
  assert.equal(c.formatNodeErrors({ 9: { exception_message: 'boom' } }), 'Node 9: boom');
  assert.equal(c.formatNodeErrors({ 3: {} }), 'Node 3: {}');
  for (const none of [undefined, null, {}, 'x', 5]) assert.equal(c.formatNodeErrors(none), '');
});
