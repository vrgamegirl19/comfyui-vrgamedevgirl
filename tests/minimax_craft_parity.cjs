// Prints the Builder's performance craft, cue header, cue delivery and cut snapping results for
// tests/minimax_craft_cases.json, so tests/test_minimax_performance_craft.py can compare them with the Python twins.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const prompt = readBuilderModule('minimax_prompt.mjs');
const cues = readBuilderModule('lyric_cues.mjs');
const start = prompt.indexOf('const MINIMAX_H3_CRAFT_ACTING');
const end = prompt.indexOf('function miniMaxH3ReferenceSceneGroundingContract');
const context = vm.createContext({});
vm.runInContext(`${prompt.slice(start, end)}\n${functionSource(cues, 'snapCutsToVocalGaps')}
this.api = { miniMaxH3PerformanceCraftInstruction, miniMaxH3CueHeader, miniMaxH3CueDelivery, snapCutsToVocalGaps };`, context);
const { api } = context;
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'minimax_craft_cases.json'), 'utf8'));
process.stdout.write(JSON.stringify({
  craft: cases.craft.map((item) => api.miniMaxH3PerformanceCraftInstruction(item)),
  header: cases.header.map((item) => api.miniMaxH3CueHeader(item)),
  delivery: cases.delivery.map(([text, body]) => api.miniMaxH3CueDelivery(text, body)),
  snap: cases.snap.map(([cutList, words, duration]) => api.snapCutsToVocalGaps(cutList, words, duration)),
}));
