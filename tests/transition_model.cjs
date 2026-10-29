const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The rules of the transition lane (web/music_video_builder/transition_model.mjs): which frames a block replaces, how far it can be moved and
// stretched, what the stitch is sent. The H3 Studio app's Rust twin (crates/h3-media/src/transition.rs, `plan`) runs the window cases of
// tests/transition_window_cases.json too, so the live preview and the exported video agree. Regenerate that file with
// `node scripts/export_transition_window_cases.mjs` after changing either rule.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('transition_model.mjs'), context);
const tm = (name) => vm.runInContext(name, context);
// Results come from another realm, so compare their JSON, not their prototypes.
const plain = (value) => JSON.parse(JSON.stringify(value));

const scenes = (...lengths) => {
  let at = 0;
  return lengths.map((length, i) => ({ id: `s${i + 1}`, start: (at += length) - length, end: at }));
};
const block = (id, start, seconds, transition = 'WipeLeft') => ({ id, transition, start, seconds, params: {} });
const windowOf = (entry, segments) => plain(tm('transitionWindow')(entry, segments));
const shape = (entry, segments) => {
  const w = windowOf(entry, segments);
  return [w.first, w.frames, w.cut];
};

test('a block over a cut takes frames from both sides', () => {
  const w = windowOf(block('a', 1.75, 0.5), scenes(2, 2));
  assert.deepEqual([w.first, w.frames, w.cut, w.before, w.after], [42, 12, 0, 6, 6]);
});

test('a block that ends or starts at a cut belongs to that cut', () => {
  const clips = scenes(2, 2, 2);
  assert.deepEqual(shape(block('a', 1.5, 0.5), clips), [36, 12, 0]);
  assert.deepEqual(shape(block('b', 4.0, 0.5), clips), [96, 12, 1]);
});

test('a block inside a scene lies on no cut', () => {
  assert.deepEqual(shape(block('a', 0.5, 0.5), scenes(2, 2)), [12, 12, null]);
  assert.equal(windowOf(block('a', 1.4583, 0.5 - 1 / 24), scenes(2, 2)).cut, null, 'one frame short of the cut');
  assert.equal(windowOf(block('a', 2.5, 0.5), scenes(2, 2)).scene, 1);
});

test('a block over two cuts is cut down to the nearer cut, and over a whole scene sits on the later one', () => {
  assert.deepEqual(shape(block('a', 0.5, 38 / 24), scenes(1, 1, 1)), [12, 36, 0]);
  assert.deepEqual(shape(block('a', 1, 1), scenes(1, 1, 1)), [24, 24, 1]);
});

test('a block reaching past the end of the last scene is cut at it', () => {
  assert.deepEqual(shape(block('a', 3.5, 2), scenes(2, 2)), [84, 12, null]);
});

test('the progress is never exactly 0 or 1', () => {
  const w = windowOf(block('a', 1.75, 0.5), scenes(2, 2));
  const at = (frame) => tm('transitionProgressAt')(w, frame);
  assert.equal(at(41), null);
  assert.equal(at(42), 1 / 13);
  assert.equal(at(53), 12 / 13);
  assert.equal(at(54), null);
});

test('an entry of the earlier per-cut version reads as the block it stood for', () => {
  const clips = scenes(2, 2);
  const read = (align) => plain(tm('normalizeTransitionEntries')([{ after_scene: 's1', transition: 'Swirl', seconds: 0.5, align }], clips))[0].start;
  assert.deepEqual([read('center'), read('end'), read('start')], [1.75, 1.5, 2]);
  assert.deepEqual(plain(tm('normalizeTransitionEntries')([{ after_scene: 'gone', transition: 'Swirl' }], clips)), [], 'a deleted scene drops it');
});

test('entries get ids of their own, a length in range and the order of the timeline', () => {
  const list = plain(tm('normalizeTransitionEntries')([
    { id: 'x', transition: 'A', start: 5, seconds: 99 },
    { id: 'x', transition: 'B', start: 1, seconds: 0.01 },
    { transition: 'C', start: 3 },
    { transition: '', start: 0 },
    null,
  ]));
  assert.deepEqual(list.map((e) => [e.id, e.transition, e.start, e.seconds]), [['t1', 'B', 1, 0.1], ['t2', 'C', 3, 0.5], ['x', 'A', 5, 10]]);
});

test('overlapping blocks are stacked in rows', () => {
  const { rows, count } = plain(tm('transitionRows')([block('a', 0, 1), block('b', 0.5, 1), block('c', 1.5, 1)]));
  assert.deepEqual([rows.a, rows.b, rows.c, count], [0, 1, 0, 2]);
  assert.equal(tm('transitionRows')([]).count, 1);
});

test('a block cannot be moved or stretched over its neighbours', () => {
  const list = [block('a', 1, 1), block('b', 3, 1), block('c', 6, 1)];
  const move = tm('moveTransition');
  assert.equal(move(list, 'b', 0, 20), 2, 'stops at the end of the block before');
  assert.equal(move(list, 'b', 9, 20), 5, 'stops at the start of the block after');
  assert.equal(move(list, 'c', 9, 8), 7, 'and at the end of the timeline');
  assert.deepEqual(plain(tm('trimTransitionStart')(list, 'b', 0)), { start: 2, seconds: 2 });
  assert.deepEqual(plain(tm('trimTransitionStart')(list, 'b', 3.95)), { start: 3.9, seconds: 0.1 }, 'never shorter than the least length');
  assert.deepEqual(plain(tm('trimTransitionEnd')(list, 'b', 9, 20)), { start: 3, seconds: 3 });
  assert.deepEqual(plain(tm('trimTransitionEnd')(list, 'c', 99, 8)), { start: 6, seconds: 2 });
  assert.deepEqual(plain(tm('trimTransitionEnd')(list, 'c', 99, 0)), { start: 6, seconds: 10 }, 'no limit but the longest length');
});

test('times snap to the scene edges within a few pixels, and to beats only when asked', () => {
  const snap = tm('snapTransitionTime');
  const options = { points: [2, 4], beats: [3.01], pxPerSecond: 100 };
  assert.equal(snap(2.05, options), 2);
  assert.equal(snap(2.2, options), 2.2);
  assert.equal(snap(3.0, options), 3.0);
  assert.equal(snap(3.0, { ...options, snapBeats: true }), 3.01);
  assert.equal(snap(0.03, options), 0);
});

test('a new block is centered on a cut near the playhead, else at the playhead, and never on another block', () => {
  const clips = scenes(2, 2, 2);
  const place = (list, options) => plain(tm('placeNewTransition')(list, clips, options));
  assert.deepEqual(place([], { playhead: 2.2, seconds: 0.5, total: 6 }), { start: 1.75, seconds: 0.5 });
  assert.deepEqual(place([], { playhead: 1.0, seconds: 0.5, total: 6 }), { start: 1.0, seconds: 0.5 });
  assert.deepEqual(place([block('a', 1.75, 0.5)], { playhead: 2.0, seconds: 0.5, total: 6 }), { start: 2.25, seconds: 0.5 }, 'after the block in the way');
  assert.deepEqual(place([block('a', 5.75, 0.25)], { playhead: 5.8, seconds: 0.5, total: 6 }), { start: 5.25, seconds: 0.5 }, 'before it near the end');
  assert.equal(place([block('a', 0, 6)], { playhead: 1, seconds: 0.5, total: 6 }), null, 'no room');
  assert.deepEqual(place([], { playhead: 0, seconds: 0.5, total: 6, range: { start: 3, end: 4 } }), { start: 3, seconds: 1 }, 'the selected range');
});

test('the stitch gets the blocks on the clock of its scene timing', () => {
  const clips = scenes(2, 2, 2);
  const list = [block('a', 0.5, 0.5, 'FlashColor'), block('b', 1.75, 0.5), block('c', 5.0, 0.5)];
  const items = (segments, offset) => plain(tm('transitionStitchItems')(list, segments, offset));
  assert.deepEqual(items(clips, 0).map((i) => [i.transition, i.start, i.seconds]), [['FlashColor', 0.5, 0.5], ['WipeLeft', 1.75, 0.5], ['WipeLeft', 5, 0.5]]);
  // Only the last two scenes: the first block is outside, the second starts 2 s earlier.
  assert.deepEqual(items(clips.slice(1), 2).map((i) => [i.start, i.seconds]), [[0, 0.25], [3, 0.5]], 'a block that starts before the first scene is cut at its start');
});

test('the window cases the export planner runs too', () => {
  const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'transition_window_cases.json'), 'utf8'));
  assert.ok(cases.length > 500);
  for (const [n, c] of cases.entries()) {
    const segments = c.clips.map(([start, end], i) => ({ id: `s${i}`, start, end }));
    const w = windowOf({ start: c.start, seconds: c.seconds }, segments);
    const got = w.frames >= 2 ? { first: w.first, frames: w.frames, cut: w.cut } : null;
    assert.deepEqual(got, c.expect, `case ${n}: ${JSON.stringify(c)}`);
  }
});
