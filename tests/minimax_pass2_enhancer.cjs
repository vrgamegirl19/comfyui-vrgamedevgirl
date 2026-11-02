const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture() {
  const elements = [];
  const element = () => {
    const node = { style: {}, children: [], setAttribute() {}, append(...children) {
      this.children.push(...children);
      for (const child of children) child.parentElement = this;
    } };
    elements.push(node);
    return node;
  };
  const c = vm.createContext({ document: { createElement: element },
    makeCheckbox(label, checked) { return { label, input: { checked }, wrapper: element() }; },
    makeSearchableLoraPicker(value) { return { input: { value }, wrapper: element() }; },
    makeField(label, child) { const field = element(); field.append(child); return field; },
  });
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  vm.runInContext(readBuilderModule('minimax_pass2_enhancer.mjs'), c);
  return { c, elements };
}

test('enhancer exposes both downloads and ComfyUI-format defaults', () => {
  const { c, elements } = fixture();
  const controls = c.createMiniMaxPass2Enhancer();
  assert.equal(controls.enabled.input.checked, false);
  assert.match(controls.dmad.input.value, /full_critic_comfyui\.safetensors$/);
  assert.match(controls.refiner.input.value, /rank_16_bf16\.safetensors$/);
  const links = elements.filter(e => e.href);
  assert.equal(links.length, 2);
  assert.match(links[0].href, /ZhengmingYu\/DMAD\/resolve\/main\/minimax_h3\/.*comfyui/);
  assert.match(links[1].href, /Kijai\/MiniMax-H3-experimental\/resolve\/main\/loras\//);
  for (const link of links) assert.equal(link.rel, 'noopener noreferrer');
});

test('toggle hides overridden controls without changing either pass and restores normal controls', () => {
  const { c } = fixture();
  const controls = c.createMiniMaxPass2Enhancer();
  const twoPassControls = [1, 2].map(pass => Object.fromEntries(
    ['steps', 'denoise', 'sampler', 'scheduler', 'seed'].map(key =>
      [key, { value: `${pass}-${key}`, parentElement: { style: {} } }])
  ));
  const normalValues = JSON.stringify(twoPassControls);
  controls.turboLayout = { style: {} };
  controls.turboDownload = { style: {} };
  const options = { twoPass: true, threePass: false, twoPassControls,
    turboPicker: { input: {} }, turboStrength: {} };
  const settings = c.cloneMiniMaxH3Settings({ two_pass_use_enhancer: true,
    two_pass_pass1_steps: 23, two_pass_pass2_steps: 9 });
  assert.equal(c.syncMiniMaxPass2Enhancer(controls, settings, options), true);
  assert.equal(controls.details.style.display, 'flex');
  assert.equal(options.turboPicker.input.disabled, true);
  assert.equal(controls.turboLayout.style.display, 'none');
  assert.equal(controls.turboDownload.style.display, 'none');
  for (const key of ['steps', 'denoise', 'sampler', 'scheduler']) {
    assert.equal(twoPassControls[0][key].disabled, false);
    assert.equal(twoPassControls[1][key].disabled, true);
    assert.equal(twoPassControls[1][key].parentElement.style.display, 'none');
  }
  assert.equal(twoPassControls[1].seed.disabled, undefined);
  c.syncMiniMaxPass2Enhancer(controls, { ...settings, two_pass_use_enhancer: false }, options);
  assert.equal(options.turboPicker.input.disabled, false);
  assert.equal(controls.turboLayout.style.display, 'grid');
  assert.equal(controls.turboDownload.style.display, '');
  assert.equal(controls.details.style.display, 'none');
  for (const key of ['steps', 'denoise', 'sampler', 'scheduler']) {
    assert.equal(twoPassControls[1][key].disabled, false);
    assert.equal(twoPassControls[1][key].parentElement.style.display, '');
  }
  for (let pass = 0; pass < 2; pass++) {
    const original = JSON.parse(normalValues)[pass];
    for (const key of Object.keys(original)) assert.equal(twoPassControls[pass][key].value, original[key].value);
  }
  assert.equal(c.syncMiniMaxPass2Enhancer(controls, settings, { ...options, threePass: true }), false);
  assert.equal(controls.wrapper.style.display, 'none');
});

test('saved enhancer and filenames survive reload and pass-profile switching', () => {
  const { c } = fixture();
  let settings = c.cloneMiniMaxH3Settings({ ...c.cloneMiniMaxH3Settings({}), video_mode: 'reference_to_video', render_pass: 'two_pass',
    two_pass_use_enhancer: true, two_pass_enhancer_dmad_name: 'H3/dmad.safetensors',
    two_pass_enhancer_refiner_name: 'H3/refiner.safetensors', two_pass_pass2_steps: 9 });
  settings = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(settings)));
  settings = c.selectMiniMaxH3PassSettings(settings, 'single');
  settings = c.selectMiniMaxH3PassSettings(settings, 'two_pass');
  assert.equal(settings.two_pass_use_enhancer, true);
  assert.equal(settings.two_pass_enhancer_dmad_name, 'H3/dmad.safetensors');
  assert.equal(settings.two_pass_enhancer_refiner_name, 'H3/refiner.safetensors');
  assert.equal(settings.two_pass_pass2_steps, 9);
});


test('model-list refresh populates both enhancer pickers and preserves selected paths', async () => {
  const { c } = fixture();
  const controls = c.createMiniMaxPass2Enhancer();
  controls.dmad.input.value = 'H3/dmad.safetensors';
  controls.refiner.input.value = 'H3/refiner.safetensors';
  const loras = ['[none]', controls.dmad.input.value, controls.refiner.input.value];
  Object.assign(c, { miniMaxPass2Enhancer: controls, getJson: async () => ({ loras }),
    REQUIRED_LTX_MSR_LORA: 'msr', REQUIRED_LTX_INGREDIENTS_LORA: 'ingredients', REQUIRED_LTX_ID_LORA: 'id' });
  for (const name of ['zLoraSlots', 'ernieLoraSlots', 'fluxLoraSlots', 'i2vLoraSlots',
    'zEnhanceLoraSlots', 'krea2TwoPassLoraSlots', 'miniMaxLoraSlots']) c[name] = [];
  for (const name of ['miniMaxTwoPassLoraPicker', 'miniMaxThreePassLoraPicker',
    'ltxMsrLoraPicker', 'ltxIngredientsLoraPicker', 'ltxIdLoraPicker']) c[name] = { input: { value: '' } };
  vm.runInContext(functionSource(readBuilderModule('model_pickers.mjs'), 'refreshLoraChoices'), c);
  await c.refreshLoraChoices();
  assert.deepEqual(Array.from(controls.dmad.options), loras);
  assert.deepEqual(Array.from(controls.refiner.options), loras);
  assert.equal(controls.dmad.input.value, 'H3/dmad.safetensors');
  assert.equal(controls.refiner.input.value, 'H3/refiner.safetensors');
});
