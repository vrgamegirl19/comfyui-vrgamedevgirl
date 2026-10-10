const { functionSource, readBuilderSource } = require('./builder_source.cjs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const source = readBuilderSource();

async function harness({ enabled = true, single = false, audio = 'input_audio', performance = 'singing', performers = [{ id: 'a', name: 'Alice' }, { id: 'b', name: 'Bob' }] } = {}) {
  const policy = await import('../web/music_video_builder/lyric_free_performance.mjs');
  const emotion = await import('../web/music_video_builder/emotion_expression.mjs');
  const plan = single ? [{ number: 1, time: 0 }] : [{ number: 1, time: 0 }, { number: 2, time: 2, timecode: '00:02.000' }];
  const context = vm.createContext({
    ...policy,
    ...emotion,
    state: { omitLyricsFromVideoPrompts: enabled, videoType: performance },
    normalizeVideoType: (v) => v,
    miniMaxH3SettingsForSegment: () => ({ audio_mode: audio }),
    miniMaxH3CutPlanForSegment: () => ({ exact_duration_seconds: 4 }),
    miniMaxH3OfficialShotPlan: () => plan,
    normalizeLyricCueMapForSegment: (s) => s.lyric_cue_map || [],
    segmentUsesNoLipSyncPerformance: (s) => Boolean(s.lyric_no_lip_sync),
    isInstrumentalLyricText: (v) => /instrumental/i.test(v),
    selectedPerformerSubjectsForSegment: () => performers,
    miniMaxH3SubjectLabelMapForSegment: () => new Map(),
    miniMaxH3PerformerLabel: (p) => p.id === 'b' ? '<Subject 2>' : '<Subject 1>',
    flattenLyricForPrompt: (v) => String(v || '').replace(/\s+/g, ' ').trim(),
    storyboardFacialPerformancePreset: () => ({ direction: 'Her eyes water. Her brows lift.' }),
    normalizeMiniMaxH3Mode: (v) => v,
    miniMaxH3PostCutShotText: (v) => v,
    miniMaxH3ModeForSegment: () => 'text_to_video',
    enforceMiniMaxH3CueOnShotDescription: (_s, d) => d,
    miniMaxH3EnsureQuotedLyricInShot: (s, d) => `${d} "${s.lyric_text}"`,
  });
  for (const name of ['omitLyricsForSegment', 'lyricFreeDirections', 'lyricFreeFacialText', 'lyricFreeShot', 'miniMaxH3OfficialShotBodyFromDescriptions']) {
    vm.runInContext(functionSource(source, name), context);
  }
  return (segment, descriptions) => context.miniMaxH3OfficialShotBodyFromDescriptions(segment, descriptions, 'text_to_video');
}

const mixed = {
  lyric_text: 'Secret song words', lyric_performance_mode: 'cue_map', facial_performance: 'crying',
  lyric_cue_map: [
    { type: 'instrumental', start: 0, end: 2 },
    { type: 'vocal', start: 2, end: 4, text: 'Secret song words', singer_id: 'b' },
  ],
};

test('single-shot assembly keeps owned acting without repeating singing or articulation', async () => {
  const run = await harness({ single: true, performers: [{ id: 'a', name: 'Alice' }] });
  const scene = { lyric_text: 'Secret song words', facial_performance: 'custom', facial_performance_custom: 'angry' };
  const draft = '<Subject 1> sings in sync with <Audio 1> [angry, singing] while she grips the zipper pull with her right hand. <Subject 1> looks down at the zipper, then toward the camera. <Subject 1>\'s mouth follows only the audible vocal phrasing.';
  const out = run(scene, [draft]);
  assert.equal((out.match(/sings\b/g) || []).length, 1);
  assert.equal((out.match(/audible vocal phrasing/g) || []).length, 1);
  assert.match(out, /she grips the zipper pull with her right hand/);
  assert.match(out, /<Subject 1> looks down at the zipper/);
});

test('performance deduplication preserves singer identity and decimal cue timing', async () => {
  const { remainingLyricFreeContract, lyricFreeShotDirection } = await import('../web/music_video_builder/lyric_free_performance.mjs');
  const contract = lyricFreeShotDirection('<Subject 2>', false, '2.5s–4s', false).replace('sings with passion', 'sings');
  assert.equal(remainingLyricFreeContract('<Subject 2> sings in sync with <Audio 1> during 2.5s–4s.', contract), '');
  assert.equal(remainingLyricFreeContract('<Subject 1> watches as <Subject 2> walks. <Subject 1> sings in sync with <Audio 1> during 2.5s–4s.', contract), contract);
  assert.equal(remainingLyricFreeContract('<Subject 2> sings in sync with <Audio 1>.', contract), contract);
});

test('multi-shot final assembly omits lyrics and articulation, and keeps opening instrumental', async () => {
  const run = await harness();
  const out = run(mixed, ['A wide camera tracks left. Her mouth moves.', '<Subject 2> sings [sad, singing] with watery eyes. Her eyes water. A dolly moves closer.']);
  assert.doesNotMatch(out, /Secret song words|mouth|lips?|jaw|<d>/i);
  assert.doesNotMatch(out.split('[Shot 2]')[0], /sings|singing/i);
  assert.match(out, /<Subject 2> sings in sync.*<Audio 1> during 2s–4s/);
  assert.match(out, /\[sad, singing\] with watery eyes/);
  assert.match(out, /eyes water/);
  assert.match(out, /wide camera tracks left/);
});

test('single continuous shot preserves delayed vocal timing and permits articulation', async () => {
  const run = await harness({ single: true });
  const out = run(mixed, ['A camera tracks left.']);
  assert.match(out, /during 2s–4s/);
  assert.match(out, /mouth and jaw movement follows only the audible vocal/);
  assert.doesNotMatch(out, /Secret song words/);
});

test('instrumental, B-roll and no-character scenes never get singing', async () => {
  const run = await harness({ single: true });
  for (const segment of [{ lyric_text: '[instrumental]' }, { ...mixed, lyric_no_lip_sync: true }, { ...mixed, no_character_present: true }]) {
    assert.doesNotMatch(run(segment, ['A camera tracks left. Her mouth moves.']), /sings|mouth|jaw/i);
  }
});

test('unchecked, built-in audio and speaking retain existing assembly', async () => {
  for (const options of [{ enabled: false }, { audio: 'built_in_audio' }, { performance: 'speaking' }]) {
    const run = await harness({ ...options, single: true });
    assert.match(run(mixed, ['A camera tracks left.']), /Secret song words/);
  }
});

test('custom emotion overrides inherited direction and keeps the LLM progression', async () => {
  const { emotionExpressionInput, hasEmotionExpressionInput } = await import('../web/music_video_builder/emotion_expression.mjs');
  assert.equal(emotionExpressionInput({}, { defaultFacialPerformance: 'custom', defaultFacialPerformanceCustom: 'Angry' }).facial_performance_custom, 'Angry');
  assert.equal(hasEmotionExpressionInput({ facial_performance: 'off' }), false);
  assert.equal(hasEmotionExpressionInput({ facial_performance: 'off', emotion_expression_tags: 'Sad' }), true);
  const run = await harness();
  const out = run({ ...mixed, emotion_expression_tags: 'Start happy, then end sad' }, [
    'She begins with bright eyes. A camera tracks left.',
    '<Subject 2> sings [happy, singing] with bright eyes. By the end, [sad, singing] accompanies a lowered gaze.',
  ]);
  assert.match(out, /\[happy, singing\].*By the end, \[sad, singing\]/);
  assert.doesNotMatch(out, /with passion|mouth|jaw|Secret song words/);
});
