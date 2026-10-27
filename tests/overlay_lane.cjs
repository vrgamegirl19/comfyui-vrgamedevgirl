const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The editing rules of the overlay lane (web/music_video_builder/overlay_lane.mjs): moving, trimming, splitting, duplicating, snapping.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('overlay_layers.mjs'), context);
vm.runInContext(readBuilderModule('overlay_lane.mjs'), context);
const ov = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));
const norm = (list) => plain(ov('normalizeOverlayLayers')(list));

const dust = { kind: 'media', id: 'dust', start: 10, end: 14, path: 'overlay/AirParticles-008.mov', source_start: 1, speed: 1 };

test('a clip moves keeping its length and stays inside the timeline', () => {
  const clips = norm([dust]);
  const move = ov('moveOverlayLayer');
  assert.deepEqual(plain(move(clips, 'dust', 20, 100)).map((c) => [c.start, c.end]), [[20, 24]]);
  assert.deepEqual(plain(move(clips, 'dust', -5, 100)).map((c) => [c.start, c.end]), [[0, 4]]);
  assert.deepEqual(plain(move(clips, 'dust', 99, 100)).map((c) => [c.start, c.end]), [[96, 100]]);
  assert.deepEqual(plain(move(clips, 'dust', 99, 0)).map((c) => [c.start, c.end]), [[99, 103]], 'no limit when the length of the timeline is not known');
  assert.equal(move(clips, 'other', 5, 100)[0].start, 10, 'another id leaves the clip alone');
});

test('trimming the start moves the media start with it, and the clip never gets shorter than 0.2 s', () => {
  const clips = norm([dust]);
  const trim = ov('trimOverlayLayerStart');
  assert.deepEqual(plain(trim(clips, 'dust', 12)).map((c) => [c.start, c.end, c.source_start]), [[12, 14, 3]], 'trimming in skips 2 s of the file');
  assert.deepEqual(plain(trim(clips, 'dust', 8)).map((c) => [c.start, c.end, c.source_start]), [[8, 14, 0]], 'extending left cannot start before the file does');
  assert.deepEqual(plain(trim(clips, 'dust', 99)).map((c) => [c.start, c.end]), [[13.8, 14]]);
  const fast = norm([{ ...dust, speed: 2 }]);
  assert.equal(plain(trim(fast, 'dust', 12))[0].source_start, 5, 'a clip played at double speed skips twice as much of the file');
  assert.equal(plain(trim(norm([{ kind: 'grain', id: 'g', start: 5, end: 9 }]), 'g', 6))[0].start, 6);
});

test('trimming the end extends or shortens the clip within the timeline', () => {
  const clips = norm([dust]);
  const trim = ov('trimOverlayLayerEnd');
  assert.equal(plain(trim(clips, 'dust', 30, 100))[0].end, 30);
  assert.equal(plain(trim(clips, 'dust', 500, 100))[0].end, 100);
  assert.equal(plain(trim(clips, 'dust', 10.05, 100))[0].end, 10.2);
  assert.equal(plain(trim(clips, 'dust', 500, 0))[0].end, 500);
});

test('a clip is split at the playhead into two that carry on from each other', () => {
  const clips = norm([{ ...dust, fade_in: 0.5, fade_out: 0.6 }, { kind: 'grain', id: 'g', start: 0, end: 20 }]);
  const split = (list, id, time) => plain(ov('splitOverlayLayer')(list, id, time));
  const got = split(clips, 'dust', 12);
  assert.deepEqual(got.map((c) => c.id), ['dust', 'dust_2', 'g']);
  assert.deepEqual([got[0].start, got[0].end, got[0].fade_in, got[0].fade_out], [10, 12, 0.5, 0]);
  assert.deepEqual([got[1].start, got[1].end, got[1].fade_in, got[1].fade_out], [12, 14, 0, 0.6]);
  assert.equal(got[1].source_start, 3, 'the second half starts the file where the first one stopped');
  assert.equal(got[1].path, got[0].path);
  const again = split(norm(got), 'dust_2', 13);
  assert.deepEqual(again.map((c) => c.id), ['dust', 'dust_2', 'dust_3', 'g']);
  // Too close to an edge, or outside the clip, is no split at all.
  assert.equal(ov('splitOverlayLayer')(clips, 'dust', 10.1), clips);
  assert.equal(ov('splitOverlayLayer')(clips, 'dust', 13.9), clips);
  assert.equal(ov('splitOverlayLayer')(clips, 'dust', 30), clips);
  assert.equal(ov('splitOverlayLayer')(clips, 'nope', 12), clips);
});

test('a clip is duplicated after itself, or before it when the timeline ends', () => {
  const clips = norm([dust]);
  const dup = (list, id, total) => plain(ov('duplicateOverlayLayer')(list, id, total));
  let got = dup(clips, 'dust', 100);
  assert.deepEqual(got.map((c) => [c.id, c.start, c.end]), [['dust', 10, 14], ['dust_2', 14, 18]]);
  got = dup(clips, 'dust', 16);
  assert.deepEqual(got.map((c) => [c.id, c.start, c.end]), [['dust', 10, 14], ['dust_2', 6, 10]], 'no room after it, so it goes before');
  assert.equal(dup(clips, 'nope', 100).length, 1);
});

test('clips are deleted by id and added on top with their own ids', () => {
  const clips = norm([dust, { kind: 'grain', id: 'grain', start: 0, end: 10 }]);
  assert.deepEqual(plain(ov('deleteOverlayLayer')(clips, 'dust')).map((c) => c.id), ['grain']);
  const added = plain(ov('addOverlayLayer')(clips, { kind: 'grain', start: 2, end: 6 }));
  assert.deepEqual(added.map((c) => c.id), ['dust', 'grain', 'grain_2']);
  assert.equal(added[2].strength, 28, 'a new clip gets the defaults');
  assert.deepEqual(plain(ov('addOverlayLayer')([], { kind: 'vignette', start: 0, end: 5 })).map((c) => c.id), ['vignette']);
});

test('times snap to beats and scene edges within a few pixels, only when snap is on', () => {
  const snap = ov('snapOverlayTime');
  const beats = [1, 2, 3.5];
  const options = { snap: true, beats, edges: [10, 12], pxPerSecond: 40, total: 60 };
  assert.equal(snap(2.1, options), 2, '0.1 s is 4 px at this zoom');
  assert.equal(snap(2.3, options), 2.3, '0.3 s is 12 px, too far');
  assert.equal(snap(11.9, options), 12, 'scene edges count');
  assert.equal(snap(0.15, options), 0, 'the start counts');
  assert.equal(snap(59.9, options), 60, 'the end counts');
  assert.equal(snap(2.1, { ...options, snap: false }), 2.1);
  assert.equal(snap(2.3, { ...options, pxPerSecond: 10 }), 2, 'zoomed out, 8 px is a wider stretch of time');
  assert.equal(snap('x', options), 0);
  assert.equal(snap(5, { snap: true }), 5);
});

test('a block says what the clip is', () => {
  const label = ov('overlayLayerLabel');
  assert.equal(label({ kind: 'media', path: 'C:\\p\\overlay\\AirParticles-008.mov' }), 'AirParticles-008');
  assert.equal(label({ kind: 'media', path: '' }), 'Clip');
  assert.equal(label({ kind: 'grain' }), 'Grain');
  assert.equal(label({ kind: 'vignette' }), 'Vignette');
  assert.equal(label({ kind: 'text', text: 'Verse 2\nsecond line' }), 'Verse 2');
  assert.equal(label({ kind: 'text', text: '  ' }), 'Text');
  assert.equal(label({ kind: 'lut', lut: 'Teal Orange.cube' }), 'Teal Orange');
  assert.equal(label({ kind: 'glitch' }), 'glitch');
  assert.equal(label(null), 'Overlay');
});

test('the lane shows only in a build that can draw overlays, once there is an overlay or the library asked for it', () => {
  const visible = ov('overlayLaneVisible');
  assert.equal(visible({ overlayAvailable: false, overlayLayers: [{}] }), false);
  assert.equal(visible({ overlayAvailable: true, overlayLayers: [] }), false);
  assert.equal(visible({ overlayAvailable: true, overlayLayers: [{}] }), true);
  assert.equal(visible({ overlayAvailable: true, overlayLayers: [], overlayLaneOpen: true }), true);
  assert.equal(visible(null), false);
});

test('clips that overlap are stacked in rows, the rest share one', () => {
  const rows = (list) => plain(ov('overlayLaneRowsFor')(list));
  const clip = (id, start, end) => ({ id, start, end });
  assert.deepEqual(rows([]), { rows: {}, count: 1 });
  assert.deepEqual(rows([clip('a', 0, 4), clip('b', 4, 8), clip('c', 9, 10)]), { rows: { a: 0, b: 0, c: 0 }, count: 1 }, 'clips that only touch share a row');
  assert.deepEqual(rows([clip('a', 0, 10), clip('b', 2, 6), clip('c', 3, 5), clip('d', 7, 9)]), { rows: { a: 0, b: 1, c: 2, d: 1 }, count: 3 });
  assert.deepEqual(rows([clip('late', 8, 9), clip('early', 0, 2)]), { rows: { late: 0, early: 0 }, count: 1 }, 'the order of the list does not matter for the rows');
  const many = Array.from({ length: 10 }, (_, i) => clip(`c${i}`, 0, 5));
  assert.equal(rows(many).count, 6, 'at most six rows, the rest share the last one');
  assert.equal(rows(many).rows.c9, 5);
  assert.equal(ov('overlayLaneHeight')([]), 34);
  assert.equal(ov('overlayLaneHeight')([clip('a', 0, 5), clip('b', 1, 6), clip('c', 2, 7)]), 94);
});

test('the Delete key removes the selected clip, unless the person is typing or nothing is selected', () => {
  const clips = norm([{ kind: 'grain', id: 'g', start: 0, end: 5 }, { kind: 'vignette', id: 'v', start: 0, end: 5 }]);
  const run = (key, { target = null, active = 'g', available = true, root = { contains: () => true }, mods = {} } = {}) => {
    const state = { overlayAvailable: available, activeOverlayId: active, overlayLayers: clips };
    const log = { history: 0, commits: [], prevented: 0 };
    const handler = ov('createOverlayDeleteKey')({ state, root, pushHistory: () => { log.history += 1; }, commit: (reason) => log.commits.push(reason) });
    handler({ key, target, ...mods, preventDefault: () => { log.prevented += 1; }, stopPropagation: () => {} });
    return { ids: state.overlayLayers.map((c) => c.id), active: state.activeOverlayId, ...log };
  };
  const gone = run('Delete');
  assert.deepEqual([gone.ids, gone.active, gone.history, gone.commits, gone.prevented], [['v'], '', 1, ['Overlay deleted'], 1]);
  assert.deepEqual(run('Backspace').ids, ['v'], 'Backspace does the same (the Delete key of a Mac)');
  assert.deepEqual(run('x').ids, ['g', 'v'], 'other keys do nothing');
  assert.deepEqual(run('Delete', { active: '' }).ids, ['g', 'v'], 'nothing selected');
  assert.deepEqual(run('Delete', { active: 'gone' }).ids, ['g', 'v'], 'a selection that is no longer on the timeline');
  assert.deepEqual(run('Delete', { available: false }).ids, ['g', 'v'], 'a build that cannot draw overlays has none to delete');
  assert.deepEqual(run('Delete', { mods: { ctrlKey: true } }).ids, ['g', 'v'], 'with Ctrl, Alt or Cmd the key belongs to something else');
  assert.deepEqual(run('Delete', { target: { tagName: 'TEXTAREA' } }).ids, ['g', 'v'], 'typing in the text box of a title');
  assert.deepEqual(run('Backspace', { target: { tagName: 'INPUT', type: 'number' } }).ids, ['g', 'v'], 'typing a number');
  assert.deepEqual(run('Delete', { target: { tagName: 'SELECT' } }).ids, ['g', 'v']);
  assert.deepEqual(run('Delete', { target: { tagName: 'DIV', isContentEditable: true } }).ids, ['g', 'v']);
  assert.deepEqual(run('Delete', { target: { tagName: 'INPUT', type: 'range' } }).ids, ['v'], 'a slider is not a place to type');
  assert.deepEqual(run('Delete', { target: { tagName: 'DIV' } }).ids, ['v'], 'a key pressed inside the Builder');
  assert.deepEqual(run('Delete', { target: null, root: null }).ids, ['v'], 'a key with no focused element');
  assert.deepEqual(run('Delete', { target: { tagName: 'DIV' }, root: { contains: () => false } }).ids, ['g', 'v'], 'a key pressed in a dialog outside the Builder');
});
