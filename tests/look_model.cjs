const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The rules of the effect lane and the filter lane (web/music_video_builder/look_model.mjs): which frames a block covers, how far it has run, how far it can be
// moved and stretched, what the stitch is sent. The H3 Studio app's Rust twin (crates/h3-media/src/looks.rs, `plan` and `block_state`) runs the frame cases of
// tests/look_frame_cases.json too, so the live preview and the exported video agree. Regenerate that file with
// `node scripts/export_look_frame_cases.mjs` after changing either rule.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('look_model.mjs'), context);
const lm = (name) => vm.runInContext(name, context);
// Results come from another realm, so compare their JSON, not their prototypes.
const plain = (value) => JSON.parse(JSON.stringify(value));

const block = (id, start, seconds, look = 'Glow') => ({ id, look, start, seconds, intensity: 1, params: {} });

test('the kinds name their session keys, state keys and ids', () => {
  const kinds = plain(lm('LOOK_KINDS'));
  assert.equal(kinds.filter.sessionKey, 'video_filters');
  assert.equal(kinds.effect.sessionKey, 'video_effects');
  assert.deepEqual(plain(lm('LOOK_ORDER')), ['filter', 'effect'], 'filters are drawn first');
  const keys = Object.values(kinds).flatMap((k) => [k.listKey, k.availableKey, k.activeKey]);
  assert.equal(new Set(keys).size, keys.length, 'no state key is shared by the two lanes');
});

test('a list is kept sorted, with an id, a length and a strength in range for every block', () => {
  const list = plain(lm('normalizeLookEntries')([
    { look: 'Glow', start: 2, seconds: 0.5 },
    { id: 'x', look: 'Flash', start: 1, seconds: 0, intensity: 3 },
    { id: 'x', look: 'Bloom', start: 1.5, intensity: -1 },
    { look: '', start: 4 },
    { look: 'Shake' },
    'junk',
    null,
  ], 'effect'));
  assert.deepEqual(list.map((e) => [e.id, e.look, e.start, e.seconds, e.intensity]), [
    ['x', 'Flash', 1, 0.05, 1],
    ['e2', 'Bloom', 1.5, 1, 0],
    ['e1', 'Glow', 2, 0.5, 1],
  ]);
  assert.deepEqual(plain(lm('normalizeLookEntries')([{ look: 'Sepia', start: 0 }], 'filter')).map((e) => e.id), ['f1'], 'filters have their own ids');
  assert.deepEqual(plain(lm('normalizeLookEntries')('no', 'effect')), []);
});

test('blocks may overlap and are stacked in rows', () => {
  const list = [block('a', 0, 2), block('b', 1, 2), block('c', 2, 1), block('d', 3.5, 1)];
  const { rows, count } = plain(lm('lookRows')(list));
  assert.deepEqual(rows, { a: 0, b: 1, c: 0, d: 0 });
  assert.equal(count, 2);
  assert.equal(plain(lm('lookRows')([])).count, 1);
});

test('a block covers whole frames and at least one', () => {
  const frames = (entry, total = 0) => plain(lm('lookFrames')(entry, total));
  assert.deepEqual(frames({ start: 1, seconds: 0.5 }), { first: 24, last: 36 });
  assert.deepEqual(frames({ start: 1, seconds: 0.01 }), { first: 24, last: 25 }, 'it takes a frame at least');
  assert.deepEqual(frames({ start: 3.9, seconds: 2 }, 96), { first: 94, last: 96 }, 'it ends with the video');
  assert.equal(frames({ start: 4, seconds: 1 }, 96), null, 'it starts after the video');
});

test('a block runs from just after 0 to just before 1 over its frames, and its time counts from its first frame', () => {
  const entry = { start: 1, seconds: 0.5 };
  const at = (frame) => plain(lm('lookStateAt')(entry, frame));
  assert.equal(at(23), null);
  assert.equal(at(36), null);
  assert.deepEqual(at(24), { progress: 0.5 / 12, time: 0 });
  assert.deepEqual(at(35), { progress: 11.5 / 12, time: 11 / 24 });
});

test('the player and the exported video agree on every frame case', () => {
  const file = JSON.parse(fs.readFileSync(path.join(__dirname, 'look_frame_cases.json'), 'utf8'));
  assert.ok(file.cases.length >= 500);
  const near = (a, b) => (a === null || b === null ? a === b : Math.abs(a - b) < 1e-9);
  for (const c of file.cases) {
    const entry = { start: c.start, seconds: c.seconds };
    assert.deepEqual(plain(lm('lookFrames')(entry, c.total)), c.frames, JSON.stringify(c));
    for (const state of c.states) {
      const run = plain(lm('lookStateAt')(entry, state.frame, c.total));
      assert.ok(near(run ? run.progress : null, state.progress) && near(run ? run.time : null, state.time), JSON.stringify([c, state, run]));
    }
  }
});

test('a block moves inside the timeline and keeps its length', () => {
  const entry = block('a', 1, 2);
  assert.equal(lm('moveLook')(entry, -3, 10), 0);
  assert.equal(lm('moveLook')(entry, 9, 10), 8);
  assert.equal(lm('moveLook')(entry, 4.25, 10), 4.25);
  assert.equal(lm('moveLook')(entry, 99, 0), 99, 'no limit without a length of the timeline');
});

test('an edge stretches a block and the other edge stays', () => {
  const entry = block('a', 1, 2);
  assert.deepEqual(plain(lm('trimLookStart')(entry, 0.5)), { start: 0.5, seconds: 2.5 });
  assert.deepEqual(plain(lm('trimLookStart')(entry, 5)), { start: 2.95, seconds: 0.05 }, 'it cannot pass the end');
  assert.deepEqual(plain(lm('trimLookStart')(entry, -1)), { start: 0, seconds: 3 });
  assert.deepEqual(plain(lm('trimLookEnd')(entry, 6, 10)), { start: 1, seconds: 5 });
  assert.deepEqual(plain(lm('trimLookEnd')(entry, 0, 10)), { start: 1, seconds: 0.05 }, 'it cannot pass the start');
  assert.deepEqual(plain(lm('trimLookEnd')(entry, 99, 10)), { start: 1, seconds: 9 }, 'it ends with the timeline');
});

test('a new block goes over the selected range, else the scene asked for, else the playhead', () => {
  const place = (options) => plain(lm('placeNewLook')(options));
  assert.deepEqual(place({ playhead: 2.01, seconds: 2, total: 10 }), { start: 2, seconds: 2 }, 'snapped to frames');
  assert.deepEqual(place({ playhead: 9, seconds: 2, total: 10 }), { start: 8, seconds: 2 }, 'moved back inside the timeline');
  assert.deepEqual(place({ playhead: 1, seconds: 2, total: 10, range: { start: 3, end: 5 } }), { start: 3, seconds: 2 });
  assert.deepEqual(place({ playhead: 1, seconds: 2, total: 10, span: { start: 0, end: 10 } }), { start: 0, seconds: 10 }, 'the whole video');
  assert.deepEqual(place({ playhead: 1, seconds: 2, total: 10, range: { start: 3, end: 3.01 }, span: { start: 4, end: 6 } }), { start: 4, seconds: 2 }, 'a range too small to use is ignored');
});

test('the stitch gets the blocks that lie on the stitched scenes, on the clock of their timing', () => {
  const scenes = [{ start: 2, end: 4 }, { start: 4, end: 6 }];
  const list = [block('a', 0, 1), block('b', 1.5, 1), block('c', 3, 1), block('d', 5.5, 2), block('e', 8, 1)];
  const items = plain(lm('lookStitchItems')(list, scenes, 2));
  assert.deepEqual(items.map((i) => [i.look, i.start, i.seconds]), [['Glow', 0, 0.5], ['Glow', 1, 1], ['Glow', 3.5, 2]]);
  assert.deepEqual(Object.keys(items[0]).sort(), ['intensity', 'look', 'params', 'seconds', 'start']);
});
