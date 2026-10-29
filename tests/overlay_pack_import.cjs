const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The rules of the screen that lists what a folder holds and lets the person choose what to import
// (web/music_video_builder/overlay_pack_import.mjs). The screen itself is checked in the app.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('overlay_pack_import.mjs'), context);
const helper = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));

const scan = {
  mode: 'packs',
  packs: [
    { id: 'guid-1', name: 'Dust Overlay 1', folder: '4_Dust_Overlay_1', size_bytes: 3 * 1024 * 1024, note: 'Blended as screen.', image_url: '/img?a=1' },
    { id: 'guid-2', name: 'Film Border 4', folder: '4_Film_4', size_bytes: 500 * 1024 },
  ],
  hidden: 2,
  unsupported: 1,
  not_overlays: 5,
};

test('sizes are written as a person reads them', () => {
  const bytes = helper('packImportBytes');
  assert.equal(bytes(0), '1 KB');
  assert.equal(bytes(850 * 1024), '850 KB');
  assert.equal(bytes(12.34 * 1024 ** 2), '12.3 MB');
  assert.equal(bytes(1.4 * 1024 ** 3), '1.4 GB');
  assert.equal(bytes(undefined), '1 KB');
});

test('packs and files both become entries with a name, a folder, a size and a picture when there is one', () => {
  const entries = plain(helper('packImportEntries')(scan));
  assert.deepEqual(entries.map((e) => [e.id, e.name, e.folder, e.size, e.image]), [
    ['guid-1', 'Dust Overlay 1', '4_Dust_Overlay_1', 3145728, '/img?a=1'],
    ['guid-2', 'Film Border 4', '4_Film_4', 512000, ''],
  ]);
  const files = plain(helper('packImportEntries')({ mode: 'files', files: [{ id: 'Smoke Pack/Smoke 1.mp4', name: 'Smoke 1', folder: 'Smoke Pack', size_bytes: 10 }] }));
  assert.deepEqual(files.map((e) => [e.id, e.name, e.folder]), [['Smoke Pack/Smoke 1.mp4', 'Smoke 1', 'Smoke Pack']]);
  assert.deepEqual(plain(helper('packImportEntries')(null)), []);
  assert.deepEqual(plain(helper('packImportEntries')({ mode: 'packs' })), []);
});

test('the search keeps the entries that hold every word, whatever the case, in the name or the folder', () => {
  const entries = helper('packImportEntries')(scan);
  const match = (query) => plain(helper('packImportMatches')(entries, query)).map((e) => e.id);
  assert.deepEqual(match(''), ['guid-1', 'guid-2']);
  assert.deepEqual(match('  '), ['guid-1', 'guid-2']);
  assert.deepEqual(match('DUST'), ['guid-1']);
  assert.deepEqual(match('4_film'), ['guid-2'], 'the folder counts');
  assert.deepEqual(match('dust 1'), ['guid-1'], 'every word has to be there');
  assert.deepEqual(match('dust border'), []);
});

test('choosing all shown or clearing them leaves what the search hides as it was', () => {
  const entries = helper('packImportEntries')(scan);
  const shown = [entries[0]];
  const select = helper('packImportSelectShown');
  const first = select(new Set(['guid-2']), shown, true);
  assert.deepEqual([...first].sort(), ['guid-1', 'guid-2']);
  const second = select(first, shown, false);
  assert.deepEqual([...second], ['guid-2'], 'the hidden one stays chosen');
  assert.deepEqual([...first].sort(), ['guid-1', 'guid-2'], 'the set that was passed in is not changed');
});

test('the button line says how many are chosen and how big they are together', () => {
  const entries = helper('packImportEntries')(scan);
  const text = helper('packImportChosenText');
  assert.equal(text(entries, new Set()), 'Nothing chosen yet');
  assert.equal(text(entries, new Set(['guid-1'])), '1 chosen, 3.0 MB');
  assert.equal(text(entries, new Set(['guid-1', 'guid-2', 'gone'])), '2 chosen, 3.5 MB', 'an id that is not listed is not counted');
});

test('the summary says what can be imported and what was left out and why', () => {
  const summary = helper('packImportSummary');
  assert.equal(summary(scan), '2 overlays can be imported. 2 already imported are not shown. 1 overlay pack in a format this app cannot use yet. 5 other packs (not overlays) left out.');
  assert.equal(summary(scan, 1), '1 overlay can be imported. 2 already imported are not shown. 1 overlay pack in a format this app cannot use yet. 5 other packs (not overlays) left out.');
  assert.equal(summary({ mode: 'files', files: [{ id: 'a' }], hidden: 1, truncated: true }), '1 video found. 1 already imported is not shown. only the first 3000 are shown.');
  assert.equal(summary({ mode: 'packs', packs: [] }), '0 overlays can be imported.');
});
