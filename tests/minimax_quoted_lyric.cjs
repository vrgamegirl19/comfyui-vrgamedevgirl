const { functionSource, readBuilderSource } = require('./builder_source.cjs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const source = readBuilderSource();
const helpers = ['miniMaxH3EnsureQuotedLyricInShot', 'miniMaxH3PunctuatedCueText', 'miniMaxH3CapitalizeCueText']
  .map((name) => functionSource(source, name)).join('\n');

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
  vm.runInContext(helpers, context);
  return context.miniMaxH3EnsureQuotedLyricInShot(segment, description, shotIndex, shotCount);
}

const lyric = 'If you open up your eyes and feel the dark,\nLike the heavy world is falling all apart';
const tagged = '<d>[English] If you open up your eyes and feel the dark, Like the heavy world is falling all apart.</d>';

test('a quoted "sings the lyric line" becomes a <d> tag', () => {
  const sand = 'I was tracing circles in the shifting sand\nWatching hours slip right through my open hand';
  const out = run({ lyric_text: sand }, 'He sings the lyric line, "I was tracing circles in the shifting sand Watching hours slip right through my open hand," in sync with <Audio 1>, his warm smile softening his eyes.');
  assert.equal(out, 'He sings <d>[English] I was tracing circles in the shifting sand Watching hours slip right through my open hand.</d> in sync with <Audio 1>, his warm smile softening his eyes.');
});

test('a plain copy of the lyric is put in a <d> tag', () => {
  const out = run({ lyric_text: lyric }, 'He paces. He sings If you open up your eyes and feel the dark, Like the heavy world is falling all apart in sync with <Audio 1>.');
  assert.equal(out, `He paces. He sings ${tagged} in sync with <Audio 1>.`);
});

test('a shot without the lyric gets the sung line added in a <d> tag', () => {
  const out = run({ lyric_text: lyric }, 'A man paces slowly.');
  assert.equal(out, `A man paces slowly. <Subject 1> (dave) sings ${tagged} in sync with <Audio 1>.`);
});

test('an already tagged lyric is left alone', () => {
  const said = `He sings ${tagged} in sync with <Audio 1>.`;
  assert.equal(run({ lyric_text: lyric }, said), said);
});

test('emotion-tagged exact lyrics stay intact without extra tags or repetition', () => {
  const said = '<Subject 1> sings with pleading eyes. <d>[English, desperate, singing] If you open up your eyes and feel the dark, Like the heavy world is falling all apart.</d>';
  assert.equal(run({ lyric_text: lyric, emotion_expression_tags: 'Desperate' }, said), said);
});

test('lyric lines are shared across cuts in order', () => {
  const lines = 'one two\nthree four\nfive six';
  assert.ok(run({ lyric_text: lines }, 'Shot a.', 0, 2).includes('<d>[English] One two three four.</d>'));
  assert.ok(run({ lyric_text: lines }, 'Shot b.', 1, 2).includes('<d>[English] Five six.</d>'));
});

test('instrumental, visual-only, no-character and built-in audio scenes are untouched', () => {
  assert.equal(run({ lyric_text: '[instrumental]' }, 'A man paces.'), 'A man paces.');
  assert.equal(run({ lyric_text: lyric, lyric_no_lip_sync: true }, 'A man paces.'), 'A man paces.');
  assert.equal(run({ lyric_text: lyric, no_character_present: true }, 'A man paces.'), 'A man paces.');
  assert.equal(run({ lyric_text: lyric }, 'A man paces.', 0, 1, { audio_mode: 'built_in_audio' }), 'A man paces.');
});
