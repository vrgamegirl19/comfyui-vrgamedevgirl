import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { applyStoryLocations, storyCardName } from '../web/music_video_builder/story_locations_data.mjs';

// Same cases as tests/test_story_locations.py runs against storyboard/story_locations.py.
const CASES = JSON.parse(readFileSync(new URL('./story_locations_cases.json', import.meta.url), 'utf8'));

test('card names', () => {
  for (const item of CASES.card_names) assert.equal(storyCardName(item.setting, item.spot), item.name);
});

for (const item of CASES.apply) {
  test(`apply: ${item.name}`, () => {
    let count = 0;
    const newId = (taken) => { count += 1; const id = `loc_new_${count}`; taken.add(id); return id; };
    const { refs, added, updated } = applyStoryLocations(item.refs, item.locations, newId, item.replace_descriptions);
    assert.deepEqual(refs, item.refs_after);
    assert.equal(added, item.added);
    assert.equal(updated, item.updated);
    assert.notEqual(refs, item.refs, 'the input is not changed');
  });
}
