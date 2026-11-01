const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The pure helpers of the overlay library panel (web/music_video_builder/overlay_library_panel.mjs). The panel itself is checked in the app.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('overlay_inspector.mjs'), context);
vm.runInContext(readBuilderModule('overlay_library_panel.mjs'), context);
const helper = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));

test('a clip is described by its size, length, frame rate and file size', () => {
  const meta = helper('overlayLayerMeta');
  assert.equal(meta({ width: 4096, height: 2304, duration: 7.9166, fps: 24, size_bytes: 206973492 }), '4096x2304 · 7.9 s · 24 fps · 197 MB');
  assert.equal(meta({ width: 1920, height: 1080, duration: 12, fps: 23.976, size_bytes: 3 * 1024 ** 3 }), '1920x1080 · 12.0 s · 24 fps · 3.0 GB');
  assert.equal(meta({ size_bytes: 10 }), '1 MB');
  assert.equal(meta({}), '');
  assert.equal(meta(null), '');
});

test('the library can be filtered by category', () => {
  const items = [{ id: 'dust/a.mov', category: 'dust' }, { id: 'snow/b.mov', category: 'snow' }, { id: 'dust/c.mov', category: 'dust' }];
  const filter = helper('filterOverlayItems');
  assert.deepEqual(plain(filter(items, 'dust')).map((i) => i.id), ['dust/a.mov', 'dust/c.mov']);
  assert.equal(filter(items, 'all').length, 3);
  assert.equal(filter(items, '').length, 3);
  assert.equal(filter(items, 'smoke').length, 0);
  assert.deepEqual(plain(filter(null, 'dust')), []);
});

test('categories are counted in the order the library lists them, with readable names', () => {
  const counts = plain(helper('overlayCategoryCounts')([{ category: 'dust' }, { category: 'light_leaks' }, { category: 'dust' }, { category: 'custom' }]));
  assert.deepEqual(counts, [
    { category: 'dust', count: 2, label: 'Dust' },
    { category: 'light_leaks', count: 1, label: 'Light leaks' },
    { category: 'custom', count: 1, label: 'custom' },
  ]);
  assert.deepEqual(plain(helper('overlayCategoryCounts')(undefined)), []);
});

test('the font chosen for new text and captions is used when the fonts list has it, and the default of each is used when it does not', () => {
  const chosen = (fonts, name) => helper('overlayChosenFont')(fonts, name);
  const fonts = [{ name: 'Anton', bundled: true }, { name: 'Arial Black', bundled: false, system: true }];
  assert.equal(chosen(fonts, 'Arial Black'), 'Arial Black');
  assert.equal(chosen(fonts, ' arial black '), 'Arial Black', 'the name of the list is used, whatever the case');
  assert.equal(chosen(fonts, ''), '', 'no choice is the default');
  assert.equal(chosen(fonts, 'Removed Font'), '', 'a font that is gone falls back to the default');
  assert.equal(chosen(['Anton'], 'Anton'), 'Anton', 'a list of plain names works too');
  assert.equal(chosen(undefined, 'Anton'), '');
});

test('a new clip is placed where the person is looking', () => {
  const place = (kind, options) => plain(helper('overlayLayerPlacement')(kind, options));
  // Grain and vignette: the selected range, or the whole timeline.
  assert.deepEqual(place('grain', { total: 60, range: { in: 10, out: 20 } }), { start: 10, end: 20 });
  assert.deepEqual(place('grain', { total: 60, range: { in: null, out: null } }), { start: 0, end: 60 });
  assert.deepEqual(place('vignette', { total: 60, range: { in: 30, out: 20 } }), { start: 0, end: 60 }, 'a backwards range is not used');
  assert.deepEqual(place('vignette', { total: 0 }), { start: 0, end: 1 });
  // A media clip: at the playhead, as long as its file, never past the end.
  assert.deepEqual(place('media', { playhead: 5, total: 60, duration: 7.9 }), { start: 5, end: 12.9 });
  assert.deepEqual(place('media', { playhead: 58, total: 60, duration: 7.9 }), { start: 52.1, end: 60 });
  assert.deepEqual(place('media', { playhead: 5, total: 3, duration: 7.9 }), { start: 0, end: 3 });
  assert.deepEqual(place('media', { playhead: 5, total: 60, duration: 0.2 }), { start: 5, end: 6 }, 'at least a second');
});

test('the exact picture is asked for from the scene under the playhead', () => {
  const build = (options) => plain(helper('overlayFramePayload')(options));
  const clips = [{ kind: 'grain', id: 'g' }];
  const got = build({ playhead: 7.25, segment: { start: 4, end: 8 }, videoPath: 'C:/p/rendered_scene_videos/video_0002-audio.mp4', clips, track: { enabled: false }, projectFolder: 'C:/p' });
  assert.deepEqual(got.payload, { input_path: 'C:/p/rendered_scene_videos/video_0002-audio.mp4', seek: 3.25, time_offset: 4, overlay_layers: clips, overlay_layer_track: { enabled: false }, project_folder: 'C:/p' });
  assert.equal(build({ playhead: 1, segment: null, videoPath: 'x' }).error, 'Put the playhead on a scene first.');
  assert.match(build({ playhead: 1, segment: { start: 0, end: 4 }, videoPath: '  ' }).error, /no rendered video/);
  assert.equal(build({ playhead: 3, segment: { start: 4, end: 8 }, videoPath: 'v.mp4', clips: null }).payload.seek, 0, 'a playhead before the scene start is the first frame');
  assert.deepEqual(build({ playhead: 3, segment: { start: 0, end: 8 }, videoPath: 'v.mp4' }).payload.overlay_layer_track, { enabled: true });
});

test('removing the captions takes out every lyric caption and nothing else', () => {
  const without = (list) => plain(helper('overlayWithoutCaptions')(list));
  const clips = [{ id: 'g', kind: 'grain' }, { id: 'c1', kind: 'text', group: 'lyric_captions' }, { id: 't', kind: 'text', group: 'title' }, { id: 'c2', kind: 'text', group: 'lyric_captions' }];
  assert.deepEqual(without(clips), { clips: [clips[0], clips[2]], removed: 2 });
  assert.deepEqual(without([clips[0]]), { clips: [clips[0]], removed: 0 });
  assert.deepEqual(without(null), { clips: [], removed: 0 });
});

test('captions cover the selected range, or everything when no range is selected', () => {
  const span = helper('overlayCaptionSpan');
  assert.deepEqual(plain(span({ in: 10, out: 20 })), { from: 10, to: 20 });
  assert.deepEqual(plain(span({ start: -3, end: 8 })), { from: 0, to: 8 });
  assert.equal(span({ in: 5, out: 5 }), null);
  assert.equal(span({ in: 9, out: 2 }), null);
  assert.equal(span(null), null);
  assert.equal(span({}), null);
});
