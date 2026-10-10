const { functionSource, readBuilderSource } = require('./builder_source.cjs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const source = readBuilderSource();
const quote = functionSource(source, 'miniMaxH3EnsureQuotedLyricInShot');

function run(segment, description, shotIndex = 0, shotCount = 1, settings = { audio_mode: 'input_audio' }) {
  const context = vm.createContext({
    state: {},
    hasEmotionExpressionInput: (s) => Boolean(s.emotion_expression_tags || s.facial_performance_custom),
    miniMaxH3FrameContinuityPromptEnabled: () => false,
    segmentUsesNoLipSyncPerformance: (s) => Boolean(s.lyric_no_lip_sync),
    miniMaxH3SettingsForSegment: () => settings,
    isMiniMaxSingerAssignmentMode: () => false,
    isInstrumentalLyricText: (t) => /\[instrumental\]/i.test(String(t || '')),
    escapeRegExp: (v) => String(v).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
    selectedPerformerSubjectsForSegment: () => [{ id: 'dave', name: 'dave' }],
    miniMaxH3PerformerLabel: () => '<Subject 1> (dave)',
    miniMaxH3SubjectLabelMapForSegment: () => new Map(),
    miniMaxH3ModeForSegment: () => 'reference_to_video',
  });
  vm.runInContext(quote, context);
  return context.miniMaxH3EnsureQuotedLyricInShot(segment, description, shotIndex, shotCount);
}

const lyric = 'If you open up your eyes and feel the dark,\nLike the heavy world is falling all apart';

test('a plain copy of the lyric is put in double quotes', () => {
  const out = run({ lyric_text: lyric }, 'He paces. He sings the lyric line, If you open up your eyes and feel the dark, Like the heavy world is falling all apart.');
  assert.ok(out.includes('"If you open up your eyes and feel the dark, Like the heavy world is falling all apart"'), out);
});

test('a shot without the lyric gets the sung line added in quotes', () => {
  const out = run({ lyric_text: lyric }, 'A man paces slowly.');
  assert.equal(out, 'A man paces slowly. <Subject 1> (dave) sings the lyric line, "If you open up your eyes and feel the dark, Like the heavy world is falling all apart".');
});

test('an already quoted lyric is left alone', () => {
  const said = 'He sings the lyric line, "If you open up your eyes and feel the dark, Like the heavy world is falling all apart".';
  assert.equal(run({ lyric_text: lyric }, said), said);
});

test('emotion-tagged exact lyrics stay intact without extra quotation or repetition', () => {
  const said = '<Subject 1> sings with pleading eyes. <d>[English, desperate, singing] If you open up your eyes and feel the dark, Like the heavy world is falling all apart.</d>';
  assert.equal(run({ lyric_text: lyric, emotion_expression_tags: 'Desperate' }, said), said);
});

test('lyric lines are shared across cuts in order', () => {
  const lines = 'one two\nthree four\nfive six';
  assert.ok(run({ lyric_text: lines }, 'Shot a.', 0, 2).includes('"one two three four"'));
  assert.ok(run({ lyric_text: lines }, 'Shot b.', 1, 2).includes('"five six"'));
});

test('instrumental, visual-only, no-character and built-in audio scenes are untouched', () => {
  assert.equal(run({ lyric_text: '[instrumental]' }, 'A man paces.'), 'A man paces.');
  assert.equal(run({ lyric_text: lyric, lyric_no_lip_sync: true }, 'A man paces.'), 'A man paces.');
  assert.equal(run({ lyric_text: lyric, no_character_present: true }, 'A man paces.'), 'A man paces.');
  assert.equal(run({ lyric_text: lyric }, 'A man paces.', 0, 1, { audio_mode: 'built_in_audio' }), 'A man paces.');
});
