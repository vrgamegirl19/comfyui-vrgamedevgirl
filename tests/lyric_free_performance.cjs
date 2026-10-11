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

test('inline singing is merged once, without oral directives or losing physical action', async () => {
  const run = await harness({ single: true, performers: [{ id: 'a', name: 'Alice' }] });
  const draft = '<Subject 1> walks slowly alongside the table; <Subject 1> sings passionately in sync with the audible vocals, with natural mouth and jaw articulation, narrowed eyes, and brows drawn in a defiant expression. Dust drifts through the warehouse.';
  const out = run({ lyric_text: 'Secret song words', facial_performance_custom: 'Rock/Punk' }, [draft]);
  assert.equal((out.match(/sings\b/g) || []).length, 1);
  assert.equal((out.match(/<Audio 1>/g) || []).length, 1);
  assert.doesNotMatch(out, /mouth|lips?|jaw/i);
  assert.match(out, /walks slowly alongside the table/);
  assert.match(out, /narrowed eyes/);
  assert.match(out, /brows drawn/);
  assert.match(out, /Dust drifts/);
});

test('articulation removal protects exact words inside dialogue tags and quotes', async () => {
  const { stripSingingArticulation } = await import('../web/music_video_builder/lyric_free_performance.mjs');
  const out = stripSingingArticulation('She walks, her mouth and jaw moving to the vocals. She sings <d>[English, angry] My lips are sealed.</d> and holds a note.');
  assert.match(out, /She walks\./);
  assert.match(out, /<d>\[English, angry\] My lips are sealed\.<\/d>/);
  assert.doesNotMatch(out, /jaw moving/);
});

test('merging one singer never overwrites a different scheduled vocal interval', async () => {
  const { mergeLyricFreeContract } = await import('../web/music_video_builder/lyric_free_performance.mjs');
  const first = '<Subject 1> sings in sync with the vocals in <Audio 1> during 1s–2s.';
  const second = '<Subject 1> sings in sync with the vocals in <Audio 1> during 3s–4s.';
  const [text, remaining] = mergeLyricFreeContract(first, second);
  assert.equal(text, first);
  assert.equal(remaining, second);
});

test('single-shot assembly keeps owned acting without repeating singing or articulation', async () => {
  const run = await harness({ single: true, performers: [{ id: 'a', name: 'Alice' }] });
  const scene = { lyric_text: 'Secret song words', facial_performance: 'custom', facial_performance_custom: 'angry' };
  const draft = '<Subject 1> sings in sync with <Audio 1> [angry, singing] while she grips the zipper pull with her right hand. <Subject 1> looks down at the zipper, then toward the camera. <Subject 1>\'s mouth follows only the audible vocal phrasing.';
  const out = run(scene, [draft]);
  assert.equal((out.match(/sings\b/g) || []).length, 1);
  assert.doesNotMatch(out, /mouth|lips?|jaw|audible vocal phrasing/i);
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
  assert.equal((out.match(/sings\b/g) || []).length, 1);
  assert.match(out, /\[sad, singing\] with watery eyes/);
  assert.match(out, /eyes water/);
  assert.match(out, /wide camera tracks left/);
});

test('single continuous shot preserves delayed vocal timing without articulation', async () => {
  const run = await harness({ single: true });
  const out = run(mixed, ['A camera tracks left.']);
  assert.match(out, /during 2s–4s/);
  assert.doesNotMatch(out, /mouth|lips?|jaw/i);
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

test('tagged pronoun singing merges once and removes an already-appended canonical duplicate', async () => {
  const { mergeLyricFreeContract } = await import('../web/music_video_builder/lyric_free_performance.mjs');
  const contract = '<Subject 1> sings in sync with the vocals in <Audio 1>.';
  const draft = '<Subject 1> keeps her stance planted and carries a restrained pulse through her shoulders as she [intense, singing] sings in sync with the vocals in <Audio 1>. <Subject 1> holds an engaged gaze toward the camera, her brows tightening. Warm light falls across the doorway. ' + contract;
  const [text, extra] = mergeLyricFreeContract(draft, contract);
  assert.equal(extra, '');
  assert.equal((text.match(/sings\b/g) || []).length, 1);
  assert.equal((text.match(/<Audio 1>/g) || []).length, 1);
  assert.match(text, /she \[intense, singing\] sings in sync with the vocals in <Audio 1>\./);
  assert.match(text, /restrained pulse through her shoulders/);
  assert.match(text, /Warm light falls/);
  const twice = mergeLyricFreeContract(text, contract);
  assert.equal(twice[0], text);
  assert.equal(twice[1], '');
});

test('pronoun merging preserves timed cues, requires clear ownership, and ignores quotes', async () => {
  const { mergeLyricFreeContract } = await import('../web/music_video_builder/lyric_free_performance.mjs');
  const contract = '<Subject 1> sings in sync with the vocals in <Audio 1> during 2.5s–4s.';
  const draft = '<Subject 1> stands beside the doorway. She [intense, singing] sings in sync with <Audio 1> during 2.5s–4s.';
  const [text, extra] = mergeLyricFreeContract(draft, contract);
  assert.equal(extra, '');
  assert.equal((text.match(/sings\b/g) || []).length, 1);
  assert.match(text, /\[intense, singing\]/);
  assert.match(text, /during 2.5s–4s/);
  for (const uncertain of ['<Subject 1> watches <Subject 2> as she [sad, singing] sings.', '<Subject 1> holds a sign saying "she sings".', '<Subject 1> stands nearby. She sings during 1s–2s.']) {
    assert.deepEqual(mergeLyricFreeContract(uncertain, contract), [uncertain, contract]);
  }
  const named = '<Subject 1> [intense, singing] sings in sync with <Audio 1>. <Subject 2> sings in sync with <Audio 1> during 1s–2s.';
  const [multi] = mergeLyricFreeContract(named, contract);
  assert.match(multi, /<Subject 2> sings in sync with <Audio 1> during 1s–2s\./);
});


test('the reported shoulder-pulse wording produces one singing cue in final assembly', async () => {
  const run = await harness({ single: true, performers: [{ id: 'a', name: 'Alice' }] });
  const draft = '<Subject 1> keeps her stance planted and carries a restrained pulse through her shoulders as she [intense, singing] sings in sync with the vocals in <Audio 1>. <Subject 1> holds an engaged gaze toward the camera, her brows tightening. Warm light falls across the doorway.';
  const out = run({ lyric_text: 'Secret song words', facial_performance_custom: 'intense' }, [draft]);
  assert.equal((out.match(/sings\b/g) || []).length, 1);
  assert.equal((out.match(/<Audio 1>/g) || []).length, 1);
  assert.match(out, /she \[intense, singing\] sings/);
  assert.match(out, /restrained pulse through her shoulders/);
});
