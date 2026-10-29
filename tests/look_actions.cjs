const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// What the effect and filter screens do to a project's blocks (web/music_video_builder/look_actions.mjs). A build that cannot draw them (this pack: no
// capability route answers) must leave the project's `video_effects` and `video_filters` as they were and send none to the stitch.
const toasts = [];
const context = vm.createContext({ document: { querySelectorAll: () => [] }, toast: (text) => toasts.push(text) });
for (const name of ['transition_model.mjs', 'transition_dom.mjs', 'transition_actions.mjs', 'look_model.mjs', 'look_actions.mjs']) vm.runInContext(readBuilderModule(name), context);
const act = (name) => vm.runInContext(name, context);
// Results come from another realm, so compare their JSON, not their prototypes.
const plain = (value) => JSON.parse(JSON.stringify(value));

const segments = [{ id: 'a', start: 0, end: 2 }, { id: 'b', start: 2, end: 4 }];
const savedEffects = () => [
  { id: 'x', look: 'Glow', start: 1.5, seconds: 1, intensity: 0.5, params: { radius: 0.1 } },
  { look: 'Flash', start: 0.5, seconds: 0.25 },
];
const savedFilters = () => [{ id: 'f9', look: 'Sepia', start: 0, seconds: 4 }];
const stateOf = (extra) => ({ segments, videoEffects: savedEffects(), videoFilters: savedFilters(), ...extra });

test('a build that cannot draw them keeps the saved lists as they were and sends none to the stitch', () => {
  const state = stateOf({ effectsAvailable: false, filtersAvailable: false });
  assert.deepEqual(plain(act('looksForSession')(state, 'effect')), savedEffects());
  assert.deepEqual(plain(act('looksForSession')(state, 'filter')), savedFilters());
  assert.deepEqual(plain(state.videoEffects), savedEffects(), 'nothing was rewritten');
  assert.deepEqual(plain(act('looksForStitch')(state, 'effect', segments, 0)), []);
  assert.deepEqual(plain(act('looksForStitch')(state, 'filter', segments, 0)), []);
  for (const kind of ['effect', 'filter']) {
    assert.equal(act('lookLaneVisible')(state, kind), false);
    assert.equal(act('lookLaneHeight')(state, kind), 0);
  }
  assert.deepEqual(plain(act('looksForSession')({ segments }, 'effect')), [], 'a project without the key saves an empty list');
});

test('a build that can draw one kind reads it into blocks and leaves the other kind alone', () => {
  const state = stateOf({ effectsAvailable: true, filtersAvailable: false });
  const list = plain(act('looksForSession')(state, 'effect'));
  assert.deepEqual(list.map((e) => [e.id, e.look, e.start]), [['e1', 'Flash', 0.5], ['x', 'Glow', 1.5]]);
  assert.deepEqual(plain(act('looksForStitch')(state, 'effect', segments, 0)).map((i) => [i.look, i.start, i.seconds, i.intensity]), [['Flash', 0.5, 0.25, 1], ['Glow', 1.5, 1, 0.5]]);
  assert.deepEqual(plain(act('looksForStitch')(state, 'filter', segments, 0)), [], 'filters cannot be drawn here');
  assert.deepEqual(plain(act('looksForSession')(state, 'filter')), savedFilters());
  assert.equal(act('lookLaneVisible')(state, 'effect'), true);
  assert.equal(act('lookLaneHeight')(state, 'effect'), 1 * 26 + 4, 'the two blocks do not overlap and share a row');
});

test('overlapping blocks take a row each', () => {
  const state = stateOf({ filtersAvailable: true, videoFilters: [{ look: 'Sepia', start: 0, seconds: 3 }, { look: 'Noir', start: 1, seconds: 1 }] });
  assert.equal(act('lookLaneHeight')(state, 'filter'), 2 * 26 + 4);
});

test('the lane shows once the project has a block or the tab was opened', () => {
  const empty = { segments, videoEffects: [], videoFilters: [], effectsAvailable: true, filtersAvailable: true };
  assert.equal(act('lookLaneVisible')(empty, 'effect'), false);
  assert.equal(act('lookLaneVisible')({ ...empty, effectLaneOpen: true }, 'effect'), true);
  assert.equal(act('lookLaneVisible')({ ...empty, effectLaneOpen: true }, 'filter'), false, 'the other lane stays shut');
  assert.equal(act('lookLaneVisible')({ ...empty, effectLaneOpen: true, effectsAvailable: false }, 'effect'), false);
});

test('a change asks for an undo step first, saves a new sorted list and then tells the page', () => {
  const state = stateOf({ effectsAvailable: true });
  act('lookEntries')(state, 'effect');
  const before = state.videoEffects;
  const calls = [];
  act('saveLooks')(state, 'effect', [...before, { look: 'Shake', start: 0.1, seconds: 0.5 }], 'Added', { pushHistory: () => calls.push(['history', before.length]), commit: (reason) => calls.push(['commit', reason]) });
  assert.deepEqual(calls, [['history', 2], ['commit', 'Added']]);
  assert.notEqual(state.videoEffects, before, 'a new list, so the history keeps the old one');
  assert.deepEqual(plain(state.videoEffects).map((e) => e.look), ['Shake', 'Flash', 'Glow']);
  assert.equal(before.length, 2);
});

test('only one block of the timeline is the chosen one', () => {
  const state = stateOf({ effectsAvailable: true, filtersAvailable: true, activeTransitionId: 't1' });
  act('selectLook')(state, 'effect', 'x');
  assert.deepEqual([state.activeEffectId, state.activeFilterId, state.activeTransitionId], ['x', '', ''], 'a transition is let go');
  act('selectLook')(state, 'filter', 'f9');
  assert.deepEqual([state.activeEffectId, state.activeFilterId], ['', 'f9'], 'so is the other kind');
  act('deselectLooks')(state);
  assert.deepEqual([state.activeEffectId, state.activeFilterId], ['', '']);
});

test('a removed block stops being the chosen one, and a copy is placed after its block inside the timeline', () => {
  const state = stateOf({ effectsAvailable: true });
  const ctx = { pushHistory: () => {}, commit: () => {} };
  act('lookEntries')(state, 'effect');
  state.activeEffectId = 'x';
  act('duplicateLook')(state, 'effect', 'x', ctx);
  const copy = plain(state.videoEffects).find((e) => e.id !== 'x' && e.look === 'Glow');
  assert.deepEqual([copy.start, copy.seconds, copy.intensity, copy.params], [2.5, 1, 0.5, { radius: 0.1 }]);
  assert.equal(state.activeEffectId, copy.id, 'the copy is the chosen one');
  assert.notEqual(copy.id, 'x');
  act('removeLook')(state, 'effect', copy.id, ctx);
  assert.equal(state.activeEffectId, '');
  assert.deepEqual(plain(state.videoEffects).map((e) => e.id), ['e1', 'x']);
  // The end of the timeline holds a copy back.
  state.videoEffects = [{ id: 'z', look: 'Glow', start: 3.5, seconds: 0.5, intensity: 1, params: {} }];
  act('duplicateLook')(state, 'effect', 'z', ctx);
  assert.deepEqual(plain(state.videoEffects).map((e) => e.start).sort(), [3.5, 3.5], 'a block of half a second ends with the timeline at 4 s');
});
