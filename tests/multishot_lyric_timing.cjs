const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture(shots = 2, options = {}) {
  const scene = { lyric_text: 'First words then later words', start: 0, end: 8, ...options.scene };
  let transcriptions = 0;
  const c = vm.createContext({
    state: { videoType: 'singing', autoTimeSingerCuesBeforePrompt: false, ...options.state },
    miniMaxH3CutPlanForSegment: () => ({}),
    miniMaxH3OfficialShotPlan: () => Array(shots).fill({}),
    normalizeVideoType: value => value,
    isMiniMaxSingerAssignmentMode: () => !options.builtIn,
    isInstrumentalLyricText: text => text === '[instrumental]',
    flattenLyricForPrompt: text => String(text || '').trim(),
    selectedPerformerSubjectsForSegment: () => options.noSinger ? [] : [{ id: 'singer', name: 'Singer' }],
    normalizeLyricCueMapForSegment: segment => segment.lyric_cue_map || [],
    miniMaxEffectiveCueEnd: (segment, cues, index, cue) => cue.end ?? segment.end - segment.start,
    sceneDisplayName: () => 'Scene 1', segmentIndexInfo: () => ({ index: 0 }),
    autoTimeMiniMaxSingerCuesForSegment: async segment => {
      transcriptions++;
      assert.equal(segment.lyric_shot_word_timing_enabled, true);
      assert.equal(segment.lyric_performance_mode, 'cue_map');
      return !options.fail;
    },
  });
  vm.runInContext(functionSource(readBuilderModule('minimax_prompt.mjs'), 'ensureAutoTimedSingerCuesBeforePrompt'), c);
  return { scene, c, run: () => c.ensureAutoTimedSingerCuesBeforePrompt(scene), count: () => transcriptions };
}

test('multi-shot singing times lyrics even when the global option is off', async () => {
  const f = fixture();
  assert.equal(await f.run(), true);
  assert.equal(f.count(), 1);
});

test('single-shot scenes stay opt-in, including continuous plans', async () => {
  const f = fixture(1);
  await f.run();
  assert.equal(f.count(), 0);
  f.c.state.autoTimeSingerCuesBeforePrompt = true;
  await f.run();
  assert.equal(f.count(), 1);
});

test('complete manual cue timing is preserved without transcription', async () => {
  const f = fixture(2, { scene: { lyric_performance_mode: 'cue_map', lyric_cue_map: [{ text: 'Manual words', start: 0, end: null }] } });
  const before = JSON.stringify(f.scene);
  assert.equal(await f.run(), true);
  assert.equal(f.count(), 0);
  assert.equal(JSON.stringify(f.scene), before);
});

test('a blank cue start is not mistaken for a complete timing at zero', async () => {
  const f = fixture(2, { scene: { lyric_performance_mode: 'cue_map',
    lyric_cue_map: [{ text: 'Untimed words', start: null, end: null }] } });
  assert.equal(await f.run(), true);
  assert.equal(f.count(), 1);
});

test('visual-only, speaking, instrumental and built-in-audio scenes skip singing transcription', async () => {
  for (const options of [{ scene: { no_character_present: true } }, { scene: { lyric_no_lip_sync: true } },
    { scene: { performance_mode: 'speaking' } }, { scene: { lyric_text: '[instrumental]' } }, { builtIn: true }]) {
    const f = fixture(2, options);
    await f.run();
    assert.equal(f.count(), 0);
  }
});

test('missing singers and failed Whisper timing stop prompt generation', async () => {
  await assert.rejects(fixture(2, { noSinger: true }).run(), /mapped singer/);
  await assert.rejects(fixture(2, { fail: true }).run(), /prompt generation was stopped/);
});
