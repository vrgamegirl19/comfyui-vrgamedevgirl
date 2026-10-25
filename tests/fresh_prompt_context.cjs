const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource } = require('./builder_source.cjs');

const source = fs.readFileSync(path.join(__dirname, '../web/storyboard_builder/scenes.mjs'), 'utf8');
const c = vm.createContext({ slimSceneForRequest: scene => structuredClone(scene) });
for (const name of ['freshPromptSceneContext', 'storyboardSceneCardContext']) {
  vm.runInContext(functionSource(source, name), c);
}

test('fresh scene context excludes saved prompt outputs and preserves authored direction', () => {
  const scene = {
    video_prompt: 'STALE VIDEO', image_prompt: 'STALE IMAGE', minimax_h3_pass2_prompt: 'STALE PASS TWO',
    timeline_note: 'Keep her dramatic wink.', notes: 'Current direction', motion_summary: 'Track beside her.',
    scene_defaults: { performance_style: 'theatrical' },
    image_data: 'INLINE IMAGE', image_path: 'start-frame.png',
    subject_refs: [{ name: 'Singer', image: { path: 'singer.png', data: 'INLINE REF' } }],
    location_ref: null,
  };
  const original = JSON.stringify(scene);
  const result = c.storyboardSceneCardContext(scene);
  const text = JSON.stringify(result);
  assert.doesNotMatch(text, /STALE|INLINE IMAGE|INLINE REF/);
  assert.equal(result.timeline_note, scene.timeline_note);
  assert.equal(result.motion_summary, scene.motion_summary);
  assert.equal(result.scene_defaults.performance_style, 'theatrical');
  assert.equal(result.image_path, 'start-frame.png');
  assert.equal(result.subject_refs[0].image.has_inline_image, true);
  assert.equal(JSON.stringify(scene), original);
});

test('nested cards and legacy prompt aliases are filtered while frame inventory stays', () => {
  const result = c.freshPromptSceneContext({
    scenes: [{ scene_card: { i2v_prompt: 'STALE I2V', t2v_prompt: 'STALE T2V',
      minimax_h3_prompt: 'STALE H3', pass2_prompt: 'STALE SECOND PASS', notes: 'Current notes' },
      first_frame_visual_inventory: { text: 'The approved frame shows a red coat.' } }],
  });
  assert.doesNotMatch(JSON.stringify(result), /STALE/);
  assert.equal(result.scenes[0].scene_card.notes, 'Current notes');
  assert.match(result.scenes[0].first_frame_visual_inventory.text, /approved frame/);
});
