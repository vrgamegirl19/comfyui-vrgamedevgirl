const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');
const generation = readBuilderModule('reference_generation.mjs');
class Element {
  constructor() { this.style = {}; this.children = []; }
  append(...children) { this.children.push(...children); }
}
for (const batch of [false, true]) {
  test(`Krea 2 card has both sizes and no Z-image enhancer (batch=${batch})`, () => {
    const context = vm.createContext({
      document: { createElement: () => new Element() },
      makeField: (label, element) => ({ label, element }),
    });
    vm.runInContext(`${functionSource(generation, 'referenceEngineCards')}; globalThis.cards = referenceEngineCards;`, context);
    const settings = { first_pass_width: 768, width: 1536 };
    const controls = { readKrea2: () => settings };
    for (const key of ['zModel', 'zClip', 'zVae', 'kreaModel', 'kreaClip', 'kreaVae', 'mossModel', 'mossClip', 'mossVae', 'mossLora']) controls[key] = { wrapper: new Element() };
    const card = context.cards(controls, { batch }).find(item => item.key === 'krea2');
    assert.equal(card.heading, 'Krea 2');
    assert.equal(card.settings().krea2, settings);
    assert.match(card.tagline, /two passes/);
    assert.doesNotMatch(card.tagline, /Z.image/i);
    const labels = card.body.flatMap(item => item.children || [item]).map(item => item.label);
    assert.ok(labels.includes('First width'));
    assert.ok(labels.includes('Final width'));
  });
}
test('Krea reference requests use the two-pass endpoint and omit Z-image settings from older saves', () => {
  const c = vm.createContext({});
  vm.runInContext(`${functionSource(generation, 'krea2ReferencePayload')}; globalThis.payload = krea2ReferencePayload;`, c);
  const payload = c.payload(' Singer ', { krea_unet_name: 'custom-krea.safetensors', use_zimage_enhance: true, z_unet_name: 'z_image.safetensors', width: 1536, height: 2048 });
  assert.equal(payload.unet_name, 'custom-krea.safetensors');
  assert.equal(payload.prompt, 'Singer');
  assert.equal(payload.width, 1536);
  assert.equal(payload.height, 2048);
  assert.equal(payload.use_zimage_enhance, undefined);
  assert.equal(payload.z_unet_name, undefined);
  assert.doesNotMatch(generation, /build_krea2_prompt/);
  assert.equal((generation.match(/build_krea2_2pass_prompt/g) || []).length, 3);
});
