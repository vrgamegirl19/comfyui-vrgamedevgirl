const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture(sequence, key = 'custom', framing = true) {
  const context = vm.createContext({
    state: { builderStoryboardDefaults: { camera_flow: key }, segments: [] },
    STORYBOARD_CAMERA_FLOW_PRESETS: {
      [key]: { label: 'Test coverage', framing_candidates: framing, sequence },
      energetic: { sequence: [] },
    },
    normalizeStoryboardCustomCameraFlowSequence: () => sequence,
    miniMaxH3ResolveFramingEntry: (_scene, entry) => entry,
    sceneVideoConceptPromptText: () => '',
  });
  const source = readBuilderModule('minimax_prompt.mjs');
  for (const name of ['miniMaxH3SelectedFramingEntries', 'miniMaxH3PerShotFramingLines']) {
    vm.runInContext(functionSource(source, name), context);
  }
  return context;
}

const plan = (count) => Array.from({ length: count }, (_, i) => ({ number: i + 1, time: i * 2.5 }));
const coverage = [
  { id: 'wide', shot: 'wide shot', camera: 'pan left' },
  { id: 'close', shot: 'close-up shot', camera: 'push in' },
  { id: 'full', shot: 'full-body shot', camera: 'orbit around subject' },
  { id: 'detail', shot: 'detail shot', camera: 'rack focus' },
];

test('a populated scene card anchors only Shot 1 in a multi-shot scene', () => {
  const c = fixture(coverage);
  const scene = { id: 's1', shot_type: 'waist-up shot', camera_motion: 'track right' };
  const entries = c.miniMaxH3SelectedFramingEntries(scene, plan(4));
  assert.ok(entries[0].shot.startsWith(scene.shot_type));
  assert.equal(entries[0].camera, scene.camera_motion);
  assert.equal(new Set(entries.map(e => e.shot)).size, 4);
  assert.equal(new Set(entries.map(e => e.camera)).size, 4);
  const text = c.miniMaxH3PerShotFramingLines(scene, plan(4)).join('\n');
  assert.match(text, /OVERRIDES SCENE-WIDE FRAMING/);
  assert.match(text, /different camera angle/);
  assert.match(text, /never locks the camera/);
});

test('single-shot scenes retain the exact saved camera setup', () => {
  const c = fixture(coverage);
  const scene = { shot_type: 'waist-up shot', camera_motion: 'track right' };
  const entries = c.miniMaxH3SelectedFramingEntries(scene, plan(1));
  assert.equal(entries[0].shot, scene.shot_type);
  assert.equal(entries[0].camera, scene.camera_motion);
  assert.match(c.miniMaxH3PerShotFramingLines(scene, plan(1)).join('\n'), /SCENE CARD FRAMING/);
});

test('stale repeated candidate IDs are rebuilt instead of reused', () => {
  const c = fixture(coverage);
  const scene = { id: 's1', minimax_h3_framing_shot_ids: ['wide', 'wide', 'wide'] };
  const entries = c.miniMaxH3SelectedFramingEntries(scene, plan(3));
  assert.equal(new Set(entries.map(e => e.shot)).size, 3);
  assert.equal(new Set(entries.map(e => e.camera)).size, 3);
  assert.equal(new Set(scene.minimax_h3_framing_shot_ids).size, 3);
});

test('changing a pan direction does not count as distinct camera coverage', () => {
  const c = fixture([
    { id: 'a', shot: 'medium shot', camera: 'pan left' },
    { id: 'b', shot: 'wide shot', camera: 'pan right' },
    { id: 'c', shot: 'close-up shot', camera: 'push in' },
  ]);
  const entries = c.miniMaxH3SelectedFramingEntries({ id: 's1' }, plan(3));
  assert.equal(entries.filter(e => e && /pan/.test(e.camera)).length, 1);
  assert.equal(entries.filter(Boolean).length, 2);
});

test('an exhausted custom pool asks for new coverage instead of repeating', () => {
  const c = fixture(coverage.slice(0, 1));
  const text = c.miniMaxH3PerShotFramingLines({ id: 's1' }, plan(4)).join('\n');
  assert.match(text, /Shot 2 framing: choose a new shot type/);
  assert.match(text, /Shot 4 framing: choose a new shot type/);
  assert.match(text, /author new distinct coverage instead of recycling/);
});

test('multi-shot plans also work for presets without framing_candidates', () => {
  const c = fixture(coverage, 'test_preset', false);
  assert.equal(c.miniMaxH3SelectedFramingEntries({ id: 's1' }, plan(4)).filter(Boolean).length, 4);
});

test('frame continuation preserves its starting frame instead of restaging Shot 1', () => {
  const c = fixture(coverage);
  const text = c.miniMaxH3PerShotFramingLines({ id: 's1' }, plan(4), true).join('\n');
  assert.match(text, /Shot 1 framing: begin exactly as Attached Picture 1 shows it/);
  assert.match(text, /Shot 2 framing:/);
});


test('shot-size aliases and repeated angle families do not count as new coverage', () => {
  const c = fixture([
    { id: 'a', shot: 'slightly low-angle waist-up shot', camera: 'track left' },
    { id: 'b', shot: 'medium upper-body shot', camera: 'push in' },
    { id: 'c', shot: 'low-angle close-up shot', camera: 'rack focus' },
    { id: 'd', shot: 'full-body shot', camera: 'orbit around subject' },
  ]);
  const scene = { id: 's1', shot_type: 'low-angle waist-up shot', camera_motion: 'track right' };
  const entries = c.miniMaxH3SelectedFramingEntries(scene, plan(4));
  assert.equal(entries.filter(Boolean).length, 2);
  assert.equal(entries.filter(e => e && /low.angle/.test(e.shot)).length, 1);
  assert.ok(entries.filter(Boolean).some(e => /full-body/.test(e.shot)));
});

test('unqualified candidate angles are made explicit and distinct', () => {
  const c = fixture(coverage);
  const entries = c.miniMaxH3SelectedFramingEntries({ id: 's1' }, plan(4));
  const angles = entries.map(e => e.shot.split('camera angle: ')[1]);
  assert.ok(angles.every(Boolean));
  assert.equal(new Set(angles).size, 4);
});

test('the subway example gets distinct coverage with the real Medium shots preset', async () => {
  const { STORYBOARD_CAMERA_FLOW_PRESETS } = await import('../web/storyboard_builder/shot_presets.mjs');
  const c = fixture([]);
  c.STORYBOARD_CAMERA_FLOW_PRESETS = STORYBOARD_CAMERA_FLOW_PRESETS;
  c.state.builderStoryboardDefaults.camera_flow = 'medium_shots';
  const scene = {
    id: 'subway',
    shot_type: 'slightly low-angle waist-up shot',
    camera_motion: 'controlled lateral track right',
    story_beat: 'She raises her fist while singing on the subway platform',
  };
  const entries = c.miniMaxH3SelectedFramingEntries(scene, plan(4));
  assert.ok(entries.every(Boolean));
  assert.equal(new Set(entries.map(e => e.shot)).size, 4);
  assert.equal(new Set(entries.map(e => e.camera)).size, 4);
  assert.equal(entries.filter(e => /low.angle/.test(e.shot)).length, 1);
  assert.equal(entries.filter(e => /waist|upper.body/.test(e.shot)).length, 1);
});
