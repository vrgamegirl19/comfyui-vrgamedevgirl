import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { assignLabels, attachRefmodLabels, composeRefmodItems, clothingChoices, referencePayload, sceneSubjectCards, tokenReport, tokenStatusText, totalTokens } from '../web/music_video_builder/refmod_labels.mjs';

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

test('each label goes after the first mention of its subject in every shot', () => {
  const text = 'detailed_description:\nA style.\n\n[Shot 1] <Subject 1> (The man) sings. <Subject 2> (the woman) sways.\n\n[Shot 2] At 00:03.000, <Subject 2> (the woman) turns while <Subject 1> (The man) waits.';
  const result = attachRefmodLabels(text, scene());
  assert.match(result, /<Subject 1> \(The man\) <Picture 1> sings\. <Subject 2> \(the woman\) <Picture 2> sways\./);
  assert.match(result, /<Subject 2> \(the woman\) <Picture 2> turns while <Subject 1> \(The man\) <Picture 1> waits/);
});

test('labels are added once and running it again changes nothing', () => {
  const text = '[Shot 1] <Subject 1> (The man) sings and <Subject 1> (The man) waves.';
  const once = attachRefmodLabels(text, scene());
  assert.equal(once.match(/<Picture 1>/g).length, 1);
  assert.equal(attachRefmodLabels(once, scene()), once);
});

test('a subject without parentheses still gets its label', () => {
  assert.match(attachRefmodLabels('[Shot 1] <Subject 1> sings.', scene()), /<Subject 1> <Picture 1> sings\./);
});

test('a RefMod the shots never name is added in one short sentence', () => {
  const items = [...scene(), { card_id: 'l', name: 'Meadow', category: 'background', label: '<Picture 3>', wears: '' }];
  const result = attachRefmodLabels('[Shot 1] <Subject 1> (The man) sings. <Subject 2> (the woman) sways.', items);
  assert.match(result, /The setting is <Picture 3>\./);
});

test('clothing goes right after the character who wears it', () => {
  const items = [...scene(), { card_id: 'c', name: 'red hat', category: 'clothing', label: '<Picture 3>', wears: 'a' }];
  const result = attachRefmodLabels('[Shot 1] <Subject 1> (The man) sings. <Subject 2> (the woman) sways.', items);
  assert.match(result, /<Picture 1>, wearing <Picture 3>,/);
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
