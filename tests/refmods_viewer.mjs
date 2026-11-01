import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  addedMessage, createdCardRows, filterRefmods, formatBytes, formatCanvas, groupByType, isSignInRequired, isWebAddress, prettyRefmodName,
  prettyTypeName, providerLabel, refmodKindLabel, serverHost, sharedMessage, statusLabel, typeLabel,
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

test('quick add: backgrounds are locations, the rest subjects, and fields follow the RefMod', async () => {
  const { cardFieldsFromRefmod, entriesForScope, quickAddScopeOf, usedRefmodNames } = await import('../web/music_video_builder/refmod_quick_add_data.mjs');
  const library = [
    { name: 'identity/darrel', folder: 'identity', type: 'identity', kind: 'video', frames: 4, tokens: 3000, description: 'Tall man' },
    { name: 'clothing_women/red_dress', folder: 'clothing_women', type: 'clothing_women', kind: 'image', frames: 1, tokens: 800, description: '' },
    { name: 'background/meadow', folder: 'background', type: 'background', kind: 'image', frames: 1, tokens: 600, description: 'A meadow' },
    { name: 'pose_motion/dance', folder: 'pose_motion', type: 'pose_motion', kind: 'video', frames: 8, tokens: 900, description: '' },
  ];
  assert.equal(quickAddScopeOf(library[2]), 'location');
  assert.deepEqual(entriesForScope(library, 'location').map((entry) => entry.name), ['background/meadow']);
  assert.deepEqual(entriesForScope(library, 'subject').map((entry) => entry.name), ['identity/darrel', 'clothing_women/red_dress', 'pose_motion/dance']);

  const character = cardFieldsFromRefmod(library[0]);
  assert.equal(character.name, 'Darrel');
  assert.equal(character.description, 'Tall man');
  assert.equal(character.reference_type, 'character');
  assert.equal(character.source, 'refmod');
  assert.deepEqual(character.refmod, { name: 'identity/darrel', folder: 'identity', type: 'identity', kind: 'video', tokens: 3000, frames: 4, strength: 1 });

  const dress = cardFieldsFromRefmod(library[1]);
  assert.equal(dress.reference_type, 'outfit');
  assert.equal(dress.clothing_set, 'women');
  assert.equal(dress.refmod.kind, 'image');
  assert.equal(cardFieldsFromRefmod(library[3]).reference_type, 'other');

  const place = cardFieldsFromRefmod(library[2]);
  assert.equal(place.reference_type, undefined);
  assert.equal(place.source, 'refmod');

  const used = usedRefmodNames({ subjects: [{ refmod: { name: 'identity/darrel' } }, { name: 'blank' }], locations: [{ refmod: { name: 'background/meadow' } }] });
  assert.deepEqual([...used].sort(), ['background/meadow', 'identity/darrel']);
});

test('server RefMods can be searched by who shared them', () => {
  const shared = [
    { id: 'a'.repeat(24), name: 'identity/ann', type: 'identity', uploader: 'Robin', description: '', tags: [] },
    { id: 'b'.repeat(24), name: 'identity/bob', type: 'identity', uploader: 'Alex', description: '', tags: [] },
  ];
  assert.deepEqual(filterRefmods(shared, { query: 'robin' }).map((item) => item.id), ['a'.repeat(24)]);
  assert.deepEqual(filterRefmods(shared, { query: 'identity alex' }).map((item) => item.id), ['b'.repeat(24)]);
  assert.deepEqual(groupByType(shared), [{ type: 'identity', count: 2 }]);
});

test('the server line under the title shows only the host', () => {
  assert.equal(serverHost('https://aidigitalcreations.com/'), 'aidigitalcreations.com');
  assert.equal(serverHost('http://127.0.0.1:8787'), '127.0.0.1:8787');
  assert.equal(serverHost(''), '');
  assert.equal(serverHost(undefined), '');
});

test('sharing and adding say what happened', () => {
  assert.equal(sharedMessage({ duplicate: false, replaced: false }, 'identity/darrel_noclothes'), 'Shared Darrel Noclothes with the server.');
  assert.equal(sharedMessage({ replaced: true }, 'identity/brad'), 'Replaced your earlier Brad on the server.');
  assert.equal(sharedMessage({ duplicate: true }, 'identity/brad'), 'Brad is already on the server.');
  assert.equal(addedMessage({ name: 'identity/darrel' }), 'Added Darrel to your library.');
  assert.equal(addedMessage({ name: 'identity/darrel (2)', renamed: true }), 'Added to your library as Darrel (2), because you already have one with that name.');
  assert.equal(addedMessage({ name: 'identity/darrel', already: true }), 'Darrel is already in your library.');
});

test('a share that waits for approval says so', () => {
  assert.equal(sharedMessage({ status: 'pending' }, 'identity/brad'), 'Shared Brad with the server. Others see it once the owner of the server approves it.');
  assert.equal(sharedMessage({ status: 'pending', replaced: true }, 'identity/brad'), 'Replaced your earlier Brad on the server. Others see it once the owner of the server approves it.');
  assert.equal(sharedMessage({ status: 'approved' }, 'identity/brad'), 'Shared Brad with the server.');
  assert.equal(statusLabel('pending'), 'Waiting for approval');
  assert.equal(statusLabel('hidden'), 'Hidden');
  assert.equal(statusLabel('approved'), '');
  assert.equal(statusLabel(undefined), '');
});

test('ways to sign in have names, and only web pages are opened', () => {
  assert.equal(providerLabel('github'), 'GitHub');
  assert.equal(providerLabel('discord'), 'Discord');
  assert.match(providerLabel('dev'), /^Test sign-in/);
  assert.equal(providerLabel('google'), 'Google');
  assert.equal(isWebAddress('https://github.com/login/oauth/authorize?state=x'), true);
  assert.equal(isWebAddress('http://127.0.0.1:8787/auth/dev?state=x'), true);
  for (const bad of ['javascript:alert(1)', 'file:///etc/passwd', 'data:text/html,<script>', 'ftp://x.test', '', undefined, 'https://a b']) {
    assert.equal(isWebAddress(bad), false, String(bad));
  }
  assert.equal(isSignInRequired({ code: 'signin_required' }), true);
  assert.equal(isSignInRequired({ code: 'banned' }), false);
  assert.equal(isSignInRequired(new Error('x')), false);
  assert.equal(isSignInRequired(undefined), false);
});
