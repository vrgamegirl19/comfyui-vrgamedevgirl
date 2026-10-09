import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { assignLabels, attachRefmodLabels, enforceCastLabels, isIdentityCard, composeRefmodItems, clothingChoices, referencePayload, sceneSubjectCards, tokenReport, tokenStatusText, totalTokens } from '../web/music_video_builder/refmod_labels.mjs';

const cases = JSON.parse(readFileSync(new URL('./refmod_scene_cases.json', import.meta.url), 'utf8'));

for (const testCase of cases) {
  test(`shared case: ${testCase.name}`, () => {
    const subjects = sceneSubjectCards(testCase.scene || {}, testCase.subjects);
    const items = composeRefmodItems(subjects, testCase.extras, testCase.location, testCase.all_subjects, testCase.override);
    const labelled = assignLabels(items, testCase.include_audio);
    const actual = labelled.map((item) => ({ card_id: item.card_id, category: item.category, label: item.label, mod_name: item.mod_name }));
    assert.deepEqual(actual, testCase.expected);
    assert.equal(totalTokens(items), testCase.expected_tokens);
  });
}

test('payload carries the visual mods in order with their labels', () => {
  const [first] = cases;
  const items = assignLabels(composeRefmodItems(first.subjects, first.extras, first.location, first.all_subjects, first.override));
  const payload = referencePayload(items);
  assert.equal(payload[0].name, 'identity/brad');
  assert.equal(payload[0].label, '<Video 1>');
  assert.equal(payload.length, items.length);
});

const scene = (extra = {}) => [
  { card_id: 'a', name: 'The man', category: 'character', label: '<Picture 1>', wears: '', ...extra },
  { card_id: 'b', name: 'the woman', category: 'character', label: '<Picture 2>', wears: '' },
];

test('every <Subject n> mention becomes the RefMod label with the name beside it', () => {
  const text = 'detailed_description:\nA style.\n\n[Shot 1] <Subject 1> (The man) sings. <Subject 2> (the woman) sways.\n\n[Shot 2] At 00:03.000, <Subject 2> (the woman) turns while <Subject 1> (The man) waits.';
  const result = attachRefmodLabels(text, scene());
  assert.match(result, /<Picture 1> \(The man\) sings\. <Picture 2> \(the woman\) sways\./);
  assert.match(result, /<Picture 2> \(the woman\) turns while <Picture 1> \(The man\) waits/);
  assert.doesNotMatch(result, /<Subject/);
});

test('running it again changes nothing', () => {
  const text = '[Shot 1] <Subject 1> (The man) sings and <Subject 1> (The man) waves.';
  const once = attachRefmodLabels(text, scene());
  assert.match(once, /^\[Shot 1\] <Picture 1> \(The man\) sings and <Picture 1> \(The man\) waves\./);
  assert.equal(attachRefmodLabels(once, scene()), once);
});

test('a subject without parentheses still gets its label and name', () => {
  assert.match(attachRefmodLabels('[Shot 1] <Subject 1> sings.', scene()), /<Picture 1> \(The man\) sings\./);
});

test('a RefMod the shots never name is added in one short sentence', () => {
  const items = [...scene(), { card_id: 'l', name: 'Meadow', category: 'background', label: '<Picture 3>', wears: '' }];
  const result = attachRefmodLabels('[Shot 1] <Subject 1> (The man) sings. <Subject 2> (the woman) sways.', items);
  assert.match(result, /The setting is <Picture 3>\./);
});

test('clothing goes right after the character who wears it, with its name', () => {
  const items = [...scene(), { card_id: 'c', name: 'red hat', category: 'clothing', label: '<Picture 3>', wears: 'a' }];
  const result = attachRefmodLabels('[Shot 1] <Subject 1> (The man) sings. <Subject 2> (the woman) sways.', items);
  assert.match(result, /<Picture 1> \(The man\), wearing <Picture 3> \(red hat\), sings\./);
});

test('clothing the writer never named is worn by its character', () => {
  const items = [
    { card_id: 'a', name: 'Darrel', category: 'character', label: '<Video 1>', wears: '' },
    { card_id: 'c', name: 'Warm Clothing', category: 'clothing', label: '<Video 2>', wears: 'a' },
  ];
  const result = attachRefmodLabels('[Shot 1] A close-up of the hillside at dusk.', items);
  assert.match(result, /<Video 1> \(Darrel\), wearing <Video 2> \(Warm Clothing\)[,.]/);
  assert.equal(attachRefmodLabels(result, items), result);
});

test('a prop is placed next to the main character and a vehicle is stood beside, never "in the scene" as a person', () => {
  const items = [
    { card_id: 'a', name: 'Darrel', category: 'character', label: '<Video 1>', wears: '' },
    { card_id: 'p', name: 'Gold Chain', category: 'object', reference_type: 'prop', label: '<Picture 1>', wears: '' },
    { card_id: 'v', name: 'Black Charger', category: 'object', reference_type: 'vehicle', label: '<Picture 2>', wears: '' },
  ];
  const result = attachRefmodLabels('[Shot 1] <Subject 1> (Darrel) sings.', items);
  assert.match(result, /<Picture 1> \(Gold Chain\) is placed in the scene next to <Video 1> \(Darrel\)\./);
  assert.match(result, /<Video 1> \(Darrel\) stands beside <Picture 2> \(Black Charger\)\./);
});

test('a person the writer named without a label gets the label, and quoted lyrics are left alone', () => {
  const people = [{ label: '<Subject 1>', name: 'Darrel' }];
  const text = 'A close-up opens on Darrel (Darrel); Darrel sings, "Darrel is here.", as he shifts.';
  assert.equal(
    enforceCastLabels(text, people),
    'A close-up opens on <Subject 1> (Darrel); <Subject 1> sings, "Darrel is here.", as he shifts.',
  );
  assert.equal(enforceCastLabels('<Subject 1> (Darrel) sings.', people), '<Subject 1> (Darrel) sings.');
});

test('a garment written as if it were a person is dropped, then worn by its character', () => {
  const people = [{ label: '<Subject 1>', name: 'Darrel' }];
  const garments = [{ label: '<Subject 2>', name: 'Warm Clothing' }];
  const items = [
    { card_id: 'a', name: 'Darrel', category: 'character', label: '<Video 1>', wears: '' },
    { card_id: 'c', name: 'Warm Clothing', category: 'clothing', label: '<Video 2>', wears: 'a' },
  ];
  const fixed = enforceCastLabels('Darrel (Darrel) turns. Warm Clothing (Warm Clothing) remains still beside him. Dusk light fills the wall.', people, garments);
  assert.equal(fixed, '<Subject 1> (Darrel) turns. Dusk light fills the wall.');
  const result = attachRefmodLabels(`[Shot 1] ${fixed}`, items);
  assert.match(result, /<Video 1> \(Darrel\), wearing <Video 2> \(Warm Clothing\), turns\./);
  assert.doesNotMatch(result, /Subject/);
});

test('a garment the writer names in plain words gets its label, with no second wearing clause', () => {
  const people = [{ label: '<Subject 1>', name: 'Darrel Noclothes' }];
  const garments = [{ label: '<Subject 2>', name: 'Warm Clothing' }];
  const items = [
    { card_id: 'a', name: 'Darrel Noclothes', category: 'character', label: '<Video 1>', wears: '' },
    { card_id: 'c', name: 'Warm Clothing', category: 'clothing', label: '<Video 2>', wears: 'a' },
  ];
  const writer = '[Shot 1] A close-up opens on <Subject 1> (Darrel Noclothes), wearing Warm Clothing, as a push moves; <Subject 1> sings, "Next.", while shifting.';
  const result = attachRefmodLabels(enforceCastLabels(writer, people, garments), items);
  assert.equal(
    result,
    '[Shot 1] A close-up opens on <Video 1> (Darrel Noclothes), wearing <Video 2> (Warm Clothing), as a push moves; <Video 1> (Darrel Noclothes) sings, "Next.", while shifting.',
  );
});

test('only identities are performers: clothing, props and vehicles RefMods are not', () => {
  const refmod = (referenceType) => ({ id: 'x', name: 'x', source: 'refmod', reference_type: referenceType, refmod: { name: 'a/b', kind: 'video' } });
  assert.equal(isIdentityCard(refmod('character')), true);
  assert.equal(isIdentityCard({ id: 'plain', name: 'Plain card' }), true);
  for (const type of ['outfit', 'prop', 'vehicle', 'object', 'creature', 'style', 'environment']) {
    assert.equal(isIdentityCard(refmod(type)), false, type);
  }
});

test('items without a label (strength 0) are ignored', () => {
  const items = [{ card_id: 'a', name: 'The man', category: 'character', label: '', wears: '' }];
  assert.equal(attachRefmodLabels('[Shot 1] <Subject 1> (The man) sings.', items), '[Shot 1] <Subject 1> (The man) sings.');
});

const person = (name, tokens, strength = 1, category = 'character') => ({ name, tokens, strength, category });

test('token report flags the lightest character when it is more than 2x lighter', () => {
  const report = tokenReport([person('The man', 520), person('Darrel', 2394), person('Brandon', 1440)]);
  assert.deepEqual(report.imbalance, { weak: 'The man', strong: 'Darrel', ratio: 4.6 });
  assert.equal(report.total, 4354);
  assert.equal(report.over_limit, false);
});

test('strength counts toward the balance and a balanced scene has no warning', () => {
  assert.equal(tokenReport([person('A', 520), person('B', 1000, 0.6)]).imbalance, null);
  assert.equal(tokenReport([person('A', 520), person('B', 1100, 0.5)]).imbalance, null);
  assert.equal(tokenReport([person('A', 1000), person('Meadow', 5000, 1, 'background')]).imbalance, null);
});

test('over the limit is reported and the status line names it', () => {
  const items = [person('A', 3500), person('B', 3500)];
  assert.equal(tokenReport(items).over_limit, true);
  assert.match(tokenStatusText(items), /7,000 tokens\. Over 6,000/);
  assert.equal(tokenStatusText([]), '');
});

test('clothing choices list every RefMod clothing card per character and what the scene wears now', () => {
  const hat = { id: 'h', name: 'Red hat', reference_type: 'outfit', source: 'refmod', refmod: { name: 'clothing_men/hat' }, wears: 'a' };
  const dress = { id: 'd', name: 'Dress', reference_type: 'outfit', source: 'refmod', refmod: { name: 'clothing_women/dress' } };
  const man = { id: 'a', name: 'The man', reference_type: 'character', source: 'refmod', refmod: { name: 'identity/man', kind: 'image', tokens: 500, strength: 1 } };
  const subjects = [man, hat, dress];
  const items = composeRefmodItems([man], [], null, subjects, undefined);
  let [row] = clothingChoices(items, subjects, undefined);
  assert.equal(row.character, 'The man');
  assert.equal(row.current, 'h');
  assert.equal(row.overridden, false);
  assert.deepEqual(row.options.map((option) => option.id), ['h', 'd']);
  const swapped = composeRefmodItems([man], [], null, subjects, { a: 'd' });
  [row] = clothingChoices(swapped, subjects, { a: 'd' });
  assert.equal(row.current, 'd');
  assert.equal(row.overridden, true);
  assert.deepEqual(clothingChoices(composeRefmodItems([man], [], null, [man], undefined), [man], undefined), []);
});
