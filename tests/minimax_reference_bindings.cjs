const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../web/VRGDG_MusicVideoBuilderUI.js'), 'utf8');
const start = source.indexOf('  function miniMaxPromptReferenceSignature(');
const end = source.indexOf('  async function runMiniMaxH3PromptGeneration(', start);
assert.ok(start >= 0 && end > start);

function fixture() {
  const segment = { id: 'scene' };
  const items = [
    { kind: 'subject', source_id: 'lead', label: 'Lead', image: { path: 'lead.png' } },
    { kind: 'location', source_id: 'room', label: 'Room', image: { path: 'room.png' } },
  ];
  const context = {
    segment, items,
    miniMaxH3ModeForSegment: (item) => item.mode || 'reference_to_video',
    miniMaxOrderedImageReferenceItemsForSegment: () => items,
    miniMaxH3ImageReferencePromptItems: () => items,
    miniMaxReferenceBuilderImagePathsForSegment: () => items.map((item) => item.image.path),
    mediaPathKey: (value) => String(value || '').toLowerCase(),
    selectedSegmentImagePath: () => '',
    miniMaxH3FrameContinuityPromptEnabled: () => false,
    sceneDisplayName: (item, index) => item.label || `Scene ${index + 1}`,
    allEditableSegments: () => [segment],
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  return context;
}

const currentPrompt = 'subject_definitions:\n<Subject 1> is the Lead in <Picture 1>.\n<Subject 2> is the environment in <Picture 2>.\n\nsummary:\nA scene.';
const oldPrompt = 'subject_definitions:\n<Subject 1> is the environment in <Picture 1>.\n\nsummary:\nA scene.';

test('legacy prompt with matching references renders without review', () => {
  const c = fixture();
  assert.equal(c.miniMaxPromptReferenceMismatch(c.segment, currentPrompt, 'reference_to_video', ['lead.png', 'room.png']), '');
});

test('legacy location-only prompt is stale when subjects are mapped ahead of it', () => {
  const c = fixture();
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, oldPrompt, 'reference_to_video', ['lead.png', 'room.png']), /defines 1 image reference/);
});

test('legacy prompt detects a location and subject order swap', () => {
  const c = fixture();
  c.items.reverse();
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, currentPrompt, 'reference_to_video', ['room.png', 'lead.png']), /described as a subject/);
});

test('legacy prompt detects a same-kind reorder when subject names are explicit', () => {
  const c = fixture();
  c.items[1] = { kind: 'subject', source_id: 'support', label: 'Support', image: { path: 'support.png' } };
  const prompt = 'subject_definitions:\n<Subject 1> is the Lead in <Picture 1>.\n<Subject 2> is the Support in <Picture 2>.\n\nsummary:\nA scene.';
  c.items.reverse();
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, prompt, 'reference_to_video'), /names Lead/);
});

test('saved signature detects same-kind reorder without asking for review', () => {
  const c = fixture();
  c.items[1].kind = 'subject';
  c.rememberMiniMaxPromptReferences(c.segment, currentPrompt, c.miniMaxPromptReferenceSignature(c.segment));
  c.items.reverse();
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, currentPrompt, 'reference_to_video'), /reference order or images changed/);
});

test('saved signature detects a reference image change', () => {
  const c = fixture();
  c.rememberMiniMaxPromptReferences(c.segment, currentPrompt, c.miniMaxPromptReferenceSignature(c.segment));
  c.items[0].image.path = 'replacement.png';
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, currentPrompt, 'reference_to_video'), /reference order or images changed/);
});

test('signature stays compact when an image is stored in memory', () => {
  const c = fixture();
  c.items[0].image = { data: 'x'.repeat(100000) };
  assert.ok(c.miniMaxPromptReferenceSignature(c.segment).length < 40);
});

test('saved matching signature survives serialization and allows render', () => {
  const c = fixture();
  c.rememberMiniMaxPromptReferences(c.segment, currentPrompt, c.miniMaxPromptReferenceSignature(c.segment));
  const loaded = JSON.parse(JSON.stringify(c.segment));
  assert.equal(c.miniMaxPromptReferenceMismatch(loaded, currentPrompt, 'reference_to_video'), '');
});

test('editing shot prose alone does not require reference review', () => {
  const c = fixture();
  c.rememberMiniMaxPromptReferences(c.segment, currentPrompt, c.miniMaxPromptReferenceSignature(c.segment));
  assert.equal(c.miniMaxPromptReferenceMismatch(c.segment, `${currentPrompt}\n[Shot 1] Move slowly.`, 'reference_to_video'), '');
});

test('editing shot prose cannot hide a changed signed reference order', () => {
  const c = fixture();
  c.rememberMiniMaxPromptReferences(c.segment, currentPrompt, c.miniMaxPromptReferenceSignature(c.segment));
  c.items.reverse();
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, `${currentPrompt}\n[Shot 1] Move slowly.`, 'reference_to_video'), /described as a subject/);
});

test('picture number beyond supplied image count is blocked', () => {
  const c = fixture();
  const prompt = currentPrompt.replace('<Picture 2>', '<Picture 3>');
  assert.match(c.miniMaxPromptReferenceMismatch(c.segment, prompt, 'reference_to_video', ['lead.png', 'room.png']), /only 2 image references/);
});

test('non-reference modes never require binding checks', () => {
  const c = fixture();
  assert.equal(c.miniMaxPromptReferenceMismatch(c.segment, oldPrompt, 'text_to_video'), '');
});

test('batch preflight reports every stale target before rendering', () => {
  const c = fixture();
  const scenes = [
    { segment: { label: 'Scene 1', minimax_h3_prompt: currentPrompt }, index: 0 },
    { segment: { label: 'Scene 2', minimax_h3_prompt: oldPrompt }, index: 1 },
    { segment: { label: 'Scene 3', minimax_h3_prompt: oldPrompt }, index: 2 },
  ];
  const problems = c.miniMaxBatchReferenceProblems(scenes);
  assert.equal(problems.length, 2);
  assert.match(problems[0], /Scene 2/);
  assert.match(problems[1], /Scene 3/);
});

test('Render All preflight checks only scenes that will be rendered', () => {
  const c = fixture();
  const scenes = [
    { segment: { label: 'Scene 1', minimax_h3_prompt: currentPrompt }, index: 0 },
    { segment: { label: 'Scene 2', minimax_h3_prompt: oldPrompt, video_path: 'existing.mp4' }, index: 1 },
    { segment: { label: 'Scene 3', minimax_h3_prompt: oldPrompt }, index: 2 },
  ];
  Object.assign(c, {
    state: { projectVideoEngine: 'minimax_h3', projectFolder: 'project' },
    projectInput: { value: 'project' },
    normalizeBatchScope: (value) => value || 'all',
    batchTargetItems: () => scenes,
    segmentIndexInfo: (item) => ({ index: scenes.findIndex(({ segment }) => segment === item) }),
    selectedSegmentVideoPath: (item) => item.video_path || '',
    normalizeProjectVideoEngine: () => 'minimax_h3',
    miniMaxH3SettingsForSegment: () => ({ audio_mode: 'built_in_audio' }),
    validateMiniMaxSceneReadyForVideo: () => [],
  });
  const a = source.indexOf('  function validateRenderAllReady(');
  const b = source.indexOf('  function audioFallbackTargetScenes(', a);
  assert.ok(a >= 0 && b > a);
  vm.runInContext(source.slice(a, b), c);
  assert.deepEqual(Array.from(c.validateRenderAllReady({})).map((value) => value.match(/Scene \d+/)?.[0]), ['Scene 3']);
  assert.equal(c.validateRenderAllReady({ forceVideos: true }).filter((value) => /Scene [23]/.test(value)).length, 2);
});
