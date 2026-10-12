import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  applyStoryBookText, normalizeStoryModeFields, parseStoryBookText, resolveStoryContinuations, storyBookToText,
  syncStorySettings,
} from '../web/music_video_builder/story_mode_data.mjs';

// Same cases as tests/test_story_mode_data.py runs against storyboard/story_mode.py.
const CASES = JSON.parse(readFileSync(new URL('./story_mode_cases.json', import.meta.url), 'utf8'));

for (const item of CASES.parse) {
  test(`parse: ${item.name}`, () => {
    const { book, settingNames } = parseStoryBookText(item.text);
    assert.deepEqual(book, item.book);
    assert.deepEqual(settingNames, item.setting_names);
    const text = storyBookToText(book, syncStorySettings(book, settingNames));
    assert.equal(text, item.text_out);
    assert.deepEqual(parseStoryBookText(text).book, book, 'the text form reads back to the same book');
  });
}

for (const item of CASES.normalize) {
  test(`normalize: ${item.name}`, () => {
    assert.deepEqual(normalizeStoryModeFields(item.input), item.output);
  });
}

for (const item of CASES.continuations) {
  test(`continuations: ${item.name}`, () => {
    const plan = resolveStoryContinuations(item.plan, item.scenes, item.acts, item.mode);
    assert.deepEqual(Object.fromEntries(plan.map((entry) => [entry.scene_id, [entry.continues_previous, entry.continuation_source]])), item.expect);
  });
}

test('editing the book text keeps a setting look and drops unused settings', () => {
  const layer = { story_settings: [{ id: 'setting_home', name: 'Home', look: 'warm wood' }, { id: 'setting_gone', name: 'Gone' }] };
  const fields = applyStoryBookText(layer, 'Chapter 1: Back\nSetting: Home\n\nShe opens the door.');
  assert.equal(fields.story_flow_version, 2);
  assert.deepEqual(fields.story_settings, [{ id: 'setting_home', name: 'Home', look: 'warm wood' }]);
  assert.equal(fields.story_book.chapters[0].setting_id, 'setting_home');
});
