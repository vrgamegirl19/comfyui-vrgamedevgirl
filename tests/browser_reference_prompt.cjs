const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const context = vm.createContext({});
vm.runInContext(functionSource(readBuilderModule('image_generation.mjs'), 'browserImageReferencePrompt'), context);
const build = context.browserImageReferencePrompt;
const settings = { reference_context: { has_subject_reference: true, has_location_reference: true } };
const prompt = 'Using the provided character reference image and location reference image, create a cinematic profile close-up of the woman beside a round hay bale.';

test('preserves the existing reference-aware opening without generic duplicate', () => {
  assert.equal(build(prompt, settings), prompt);
  assert.equal(build(build(prompt, settings), settings), prompt);
});

test('still supplies reference instructions when missing or incomplete', () => {
  assert.match(build('Create a close-up beside a hay bale.', settings), /^Using the provided character reference and location reference as visual context/);
  assert.match(build('Using the provided character reference image, create a close-up.', settings), /^Using the provided character reference and location reference as visual context/);
  assert.equal(build('Create a close-up.', {}), 'Create a close-up.');
});

test('keeps previous-scene guidance without repeating the reference opening', () => {
  for (const purpose of ['continuity', 'same_location_composition_diversity']) {
    const configured = { ...settings, previous_scene_image_attached: true, previous_scene_image_purpose: purpose };
    const result = build(prompt, configured);
    assert.ok(result.startsWith(prompt));
    assert.equal((result.match(/Using the provided/g) || []).length, 1);
    assert.match(result, /Treat the (?:last|previous) scene image/);
    assert.equal(build(result, configured), result);
  }
});

test('retains generic fallback and recognizes the exact generated instruction', () => {
  const configured = { image_ingredients: [{ path: 'reference.png' }] };
  const result = build('Create a close-up.', configured);
  assert.match(result, /^Using the provided reference images as visual context/);
  assert.equal(build(result, configured), result);
});
