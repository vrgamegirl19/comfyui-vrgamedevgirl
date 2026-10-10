import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  createdCardRows, filterRefmods, formatBytes, formatCanvas, groupByType, prettyRefmodName, prettyTypeName, refmodKindLabel, typeLabel,
} from '../web/music_video_builder/refmods_viewer_data.mjs';

const LIBRARY = [
  { name: 'identity/darrel_noclothes', type: 'identity', kind: 'video', frames: 4, tokens: 3000, size: 5000, description: 'Tall man, beard', tags: ['lead'] },
  { name: 'identity/brad', type: 'identity', kind: 'image', frames: 1, tokens: 900, size: 9000, description: '', tags: [] },
  { name: 'clothing_men/warm-clothing', type: 'clothing_men', kind: 'image', frames: 1, tokens: 1200, size: 2000, description: 'Wool coat', tags: ['winter'] },
  { name: 'vehicle/old_truck', type: 'vehicle', kind: 'image', frames: 1, tokens: 700, size: 100, description: 'Rusty pickup', tags: [] },
];

test('names read nicely', () => {
  assert.equal(prettyRefmodName('identity/darrel_noclothes'), 'Darrel Noclothes');
  assert.equal(prettyRefmodName('clothing_men/warm-clothing'), 'Warm Clothing');
  assert.equal(prettyTypeName('clothing_men'), 'Clothing Men');
  assert.equal(typeLabel('clothing_men', { clothing_men: 'Clothing (men)' }), 'Clothing (men)');
  assert.equal(typeLabel('creature'), 'Creature');
});

test('groupByType counts each category and follows the given order', () => {
  const groups = groupByType(LIBRARY, ['identity', 'clothing_men', 'background']);
  assert.deepEqual(groups, [
    { type: 'identity', count: 2 },
    { type: 'clothing_men', count: 1 },
    { type: 'vehicle', count: 1 },
  ]);
  assert.deepEqual(groupByType([]), []);
});

test('filterRefmods filters by type, searches every word and sorts', () => {
  assert.deepEqual(filterRefmods(LIBRARY, { type: 'identity' }).map((item) => item.name), ['identity/brad', 'identity/darrel_noclothes']);
  assert.deepEqual(filterRefmods(LIBRARY, { query: 'wool coat' }).map((item) => item.name), ['clothing_men/warm-clothing']);
  assert.deepEqual(filterRefmods(LIBRARY, { query: 'winter' }).map((item) => item.name), ['clothing_men/warm-clothing']);
  assert.deepEqual(filterRefmods(LIBRARY, { query: 'darrel truck' }), []);
  assert.deepEqual(filterRefmods(LIBRARY, { sort: 'tokens' }).map((item) => item.tokens), [3000, 1200, 900, 700]);
  assert.deepEqual(filterRefmods(LIBRARY, { sort: 'size' }).map((item) => item.size), [9000, 5000, 2000, 100]);
  assert.deepEqual(filterRefmods(LIBRARY, { type: 'vehicle', query: 'brad' }), []);
});

test('details are formatted', () => {
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(2048), '2 KB');
  assert.equal(formatBytes(5 * 1024 * 1024), '5.0 MB');
  assert.equal(formatCanvas([1024, 576]), '1024 x 576');
  assert.equal(formatCanvas([0, 0]), '');
  assert.equal(refmodKindLabel(LIBRARY[0]), 'Video (4 images)');
  assert.equal(refmodKindLabel(LIBRARY[1]), 'Picture');
});

test('the created card lists only the details it has', () => {
  const rows = createdCardRows({
    name: 'darrel', folder: 'identity', kind: 'video', imageCount: 4, quality: 'high', canvas: [1024, 576], tokens: 3000, mode: 'Full Reference', size: 2048, typeLabel: 'Identity',
  });
  assert.deepEqual(rows, [
    ['Category', 'Identity'], ['Kind', 'Video (4 images)'], ['Images used', '4'], ['Quality', 'High'], ['Canvas', '1024 x 576'],
    ['Tokens', '3,000'], ['Mode', 'Full Reference'], ['File size', '2 KB'],
  ]);
  assert.deepEqual(createdCardRows({ folder: 'clothing_men', kind: 'image' }), [['Category', 'Clothing Men'], ['Kind', 'Picture']]);
});
