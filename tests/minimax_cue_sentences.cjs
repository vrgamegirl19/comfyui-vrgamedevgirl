const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');
const source = readBuilderModule('minimax_prompt.mjs');
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'minimax_cue_sentence_cases.json'), 'utf8'));
const c = vm.createContext({
  normalizeMiniMaxH3ShotDescription: text => String(text).replace(/\bAudio\s+1\b/g, '<Audio 1>')
    .replace(/<+Audio 1>+/g, '<Audio 1>').replace(/\s+/g, ' ').trim(),
  miniMaxH3FrameContinuityPromptEnabled: () => false,
  isMiniMaxSingerAssignmentMode: () => true,
  normalizeLyricCueMapForSegment: segment => [segment.cue],
  selectedPerformerSubjectsForSegment: () => [{ id: 'singer' }],
  miniMaxH3SubjectLabelMapForSegment: () => ({}),
  miniMaxH3PerformerLabel: () => '<Subject 1>',
  miniMaxH3FallbackShotDescription: () => 'A steady cinematic shot continues the scene.',
  escapeRegExp: text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
});
vm.runInContext(source.slice(source.indexOf('const MINIMAX_H3_CRAFT_ACTING'),
  source.indexOf('function miniMaxH3ReferenceSceneGroundingContract')), c);
for (const name of ['miniMaxH3PunctuatedCueText', 'miniMaxH3CapitalizeCueText',
  'miniMaxH3MergeTimedCue', 'enforceMiniMaxH3CueOnShotDescription']) {
  vm.runInContext(functionSource(source, name), c);
}
vm.runInContext(functionSource(readBuilderModule('lyric_free_performance.mjs'), 'stripSingingArticulation'), c);
const results = cases.map(item => {
  const segment = { lyric_performance_mode: 'cue_map', cue: item.cue };
  const enforce = text => c.enforceMiniMaxH3CueOnShotDescription(segment, text, 0, 'reference_to_video');
  const merged = enforce(item.description);
  const clean = c.stripSingingArticulation(merged);
  for (const text of item.contains) assert.ok(clean.includes(text), `${item.name}: missing ${text}\n${clean}`);
  assert.doesNotMatch(clean, /assigned vocal cue|then sings|after the cue\.|<\/d>[”"’']|<\/d>\s+in sync/, item.name);
  assert.equal((clean.match(/<d>/g) || []).length, 1, item.name);
  const visualProse = clean.replace(/<d>[\s\S]*?<\/d>/gi, '');
  assert.equal((visualProse.match(/\b(?:sings|singing)\b/g) || []).length, 1, item.name);
  assert.equal(enforce(merged), merged, `${item.name}: cue processing must be idempotent`);
  return clean;
});
process.stdout.write(JSON.stringify(results));
