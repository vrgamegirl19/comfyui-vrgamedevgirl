const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// What the transition screens do to a project's blocks (web/music_video_builder/transition_actions.mjs). A build that cannot draw transitions (this
// pack: no capability route answers) must leave the project's `scene_transitions` as they were and send none to the stitch.
const context = vm.createContext({});
for (const name of ['transition_model.mjs', 'transition_dom.mjs', 'transition_actions.mjs']) vm.runInContext(readBuilderModule(name), context);
const act = (name) => vm.runInContext(name, context);
// Results come from another realm, so compare their JSON, not their prototypes.
const plain = (value) => JSON.parse(JSON.stringify(value));

const segments = [{ id: 'a', start: 0, end: 2 }, { id: 'b', start: 2, end: 4 }];
const saved = () => [
  { id: 'x', transition: 'WipeLeft', start: 1.75, seconds: 0.5, params: {} },
  { after_scene: 'a', transition: 'Fade', seconds: 0.5, align: 'end', params: { n: 1 } },
];
const stateOf = (extra) => ({ segments, sceneTransitions: saved(), ...extra });

test('a build that cannot draw transitions keeps the saved list as it was and sends none to the stitch', () => {
  const state = stateOf({ transitionsAvailable: false });
  assert.deepEqual(plain(act('transitionsForSession')(state)), saved());
  assert.deepEqual(plain(state.sceneTransitions), saved(), 'nothing was rewritten');
  assert.deepEqual(plain(act('transitionsForStitch')(state, segments, 0)), []);
  assert.equal(act('transitionLaneVisible')(state), false);
  assert.equal(act('transitionLaneHeight')(state), 0);
});

test('a build that can draw them reads an earlier per-cut entry as a block, and the stitch gets the blocks', () => {
  const state = stateOf({ transitionsAvailable: true });
  const list = plain(act('transitionsForSession')(state));
  assert.deepEqual(list.map((e) => [e.transition, e.start, e.seconds]), [['Fade', 1.5, 0.5], ['WipeLeft', 1.75, 0.5]]);
  assert.ok(list.every((e) => /^\w+$/.test(e.id)));
  assert.deepEqual(plain(act('transitionsForStitch')(state, segments, 0)).map((i) => [i.transition, i.start, i.seconds]), [['Fade', 1.5, 0.5], ['WipeLeft', 1.75, 0.5]]);
  assert.equal(act('transitionLaneVisible')(state), true);
  assert.equal(act('transitionLaneHeight')(state), 2 * 30 + 4, 'two overlapping blocks take two rows');
});

test('the lane shows only while the project has a transition, whatever tab was opened', () => {
  const empty = { segments, sceneTransitions: [], transitionsAvailable: true };
  assert.equal(act('transitionLaneVisible')(empty), false, 'an empty lane is not drawn');
  assert.equal(act('transitionLaneHeight')(empty), 0);
  const one = { ...empty, sceneTransitions: [{ transition: 'Fade', start: 1.5, seconds: 0.5 }] };
  assert.equal(act('transitionLaneVisible')(one), true);
  assert.equal(act('transitionLaneVisible')({ ...one, transitionsAvailable: false }), false, 'a build that cannot draw transitions shows no lane');
  assert.equal(act('transitionLaneVisible')({ ...one, sceneTransitions: [] }), false, 'the lane goes when its last block is removed');
});

test('an earlier per-cut entry waits for the scenes, and is saved as it was until they are loaded', () => {
  const state = { segments: [], sceneTransitions: saved(), transitionsAvailable: true };
  assert.deepEqual(plain(act('transitionEntries')(state)), []);
  assert.deepEqual(plain(act('transitionsForSession')(state)), saved());
  state.segments = segments;
  assert.equal(act('transitionEntries')(state).length, 2);
});

test('a change asks for an undo step first, saves a new sorted list and then tells the page', () => {
  const state = stateOf({ transitionsAvailable: true });
  act('transitionEntries')(state);
  const before = state.sceneTransitions;
  const calls = [];
  act('saveTransitions')(state, [...before, { transition: 'Cube', start: 0.5, seconds: 0.5 }], 'Added', { pushHistory: () => calls.push(['history', before.length]), commit: (reason) => calls.push(['commit', reason]) });
  assert.deepEqual(calls, [['history', 2], ['commit', 'Added']]);
  assert.equal(before.length, 2, 'the list the undo history holds is not changed');
  assert.deepEqual(plain(state.sceneTransitions).map((e) => e.transition), ['Cube', 'Fade', 'WipeLeft']);
});

test('removing the selected block clears the selection', () => {
  const state = stateOf({ transitionsAvailable: true, activeTransitionId: 'x' });
  act('removeTransition')(state, 'x', { pushHistory() {}, commit() {} });
  assert.equal(state.activeTransitionId, '');
  assert.ok(!plain(state.sceneTransitions).some((e) => e.id === 'x'));
});
