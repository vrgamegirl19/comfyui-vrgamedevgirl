const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The overlay data (web/music_video_builder/overlay_layers.mjs). The Rust twin (h3-core/src/overlay_layers.rs in the app) and a Python
// twin run the same cases, so a rule only changes together with tests/overlay_layer_cases.json.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('overlay_layers.mjs'), context);
const ov = (name) => vm.runInContext(name, context);
// Results come from another realm, so compare their JSON, not their prototypes.
const plain = (value) => JSON.parse(JSON.stringify(value));

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'overlay_layer_cases.json'), 'utf8'));
const functions = { normalize_overlay_layers: 'normalizeOverlayLayers', normalize_overlay_layer_track: 'normalizeOverlayLayerTrack' };

test('every shared case gives the recorded result', () => {
  assert.ok(cases.length >= 30);
  for (const item of cases) {
    const got = plain(ov(functions[item.fn])(JSON.parse(JSON.stringify(item.raw))));
    assert.deepEqual(got, item.return, item.name);
  }
});

test('normalizing a result again changes nothing', () => {
  for (const item of cases) {
    const once = plain(ov(functions[item.fn])(item.raw));
    const twice = plain(ov(functions[item.fn])(once));
    assert.deepEqual(twice, once, item.name);
  }
});

test('the saved clips are not changed by normalizing them', () => {
  const saved = [{ kind: 'media', opacity: 7, id: 'a' }, { kind: 'media', id: 'a' }];
  const copy = JSON.parse(JSON.stringify(saved));
  ov('normalizeOverlayLayers')(saved);
  assert.deepEqual(saved, copy);
});

test('a list longer than the limit is cut to the limit, in order', () => {
  const many = Array.from({ length: ov('MAX_OVERLAY_LAYERS') + 2 }, (_, i) => ({ kind: 'grain', id: `g${i}` }));
  const got = plain(ov('normalizeOverlayLayers')(many));
  assert.equal(got.length, ov('MAX_OVERLAY_LAYERS'));
  assert.equal(got[0].id, 'g0');
  assert.equal(got[got.length - 1].id, `g${ov('MAX_OVERLAY_LAYERS') - 1}`);
});

test('every known kind has its own fields and no clip ends before it starts', () => {
  for (const kind of plain(ov('OVERLAY_KINDS'))) {
    const [clip] = plain(ov('normalizeOverlayLayers')([{ kind, start: 5, end: 1 }]));
    assert.equal(clip.kind, kind);
    assert.ok(clip.end > clip.start, kind);
    assert.ok(clip.fade_in + clip.fade_out <= clip.end - clip.start + 1e-9, kind);
  }
});

test('numeric text is read as a plain decimal and nothing else', () => {
  const read = (text) => plain(ov('normalizeOverlayLayers')([{ kind: 'grain', strength: text }]))[0].strength;
  assert.equal(read('42.5'), 42.5);
  assert.equal(read(' 7 '), 7);
  assert.equal(read('+7.'), 7);
  assert.equal(read('.5'), 0.5);
  assert.equal(read('1e1'), 10);
  for (const refused of ['', 'abc', '0x10', '1e', 'e1', '--1', '1 2', 'inf', 'NaN', '1e999', '.']) {
    assert.equal(read(refused), 28, JSON.stringify(refused));
  }
});
