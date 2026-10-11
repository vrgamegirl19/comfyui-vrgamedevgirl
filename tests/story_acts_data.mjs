import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { mergeActs, moveActBorder, splitAct, validateStoryActs } from '../web/music_video_builder/story_acts_data.mjs';
import { fillSceneCast, normalizeSpotMinScenes, planSegmentFields } from '../web/music_video_builder/story_mode_data.mjs';

// Same cases as tests/test_story_acts.py runs against storyboard/story_acts.py.
const CASES = JSON.parse(readFileSync(new URL('./story_acts_cases.json', import.meta.url), 'utf8'));

for (const item of CASES.validate) {
  test(`validate: ${item.name}`, () => {
    const acts = item.acts === 'base' ? CASES.acts : item.acts;
    assert.deepEqual(validateStoryActs(acts, CASES.chapters, CASES.scene_ids), item.problems);
  });
}

for (const item of CASES.move) {
  test(`move: ${item.name}`, () => {
    const acts = moveActBorder(CASES.acts, item.act_index, item.scene_id, CASES.scene_ids);
    assert.deepEqual(acts.map((act) => act.scene_ids), item.scene_ids_after);
  });
}

for (const item of CASES.merge) {
  test(`merge: ${item.name}`, () => {
    const acts = mergeActs(CASES.acts, item.act_index);
    assert.deepEqual(acts.map((act) => act.id), item.ids_after);
    assert.deepEqual(acts.map((act) => act.scene_ids), item.scene_ids_after);
    assert.deepEqual(acts.map((act) => act.cast_ids), item.cast_after);
    assert.deepEqual(acts.map((act) => act.summary), item.summary_after);
  });
}

for (const item of CASES.split) {
  test(`split: ${item.name}`, () => {
    const acts = splitAct(CASES.acts, item.act_index, item.scene_id);
    assert.deepEqual(acts.map((act) => act.id), item.ids_after);
    assert.deepEqual(acts.map((act) => act.scene_ids), item.scene_ids_after);
    assert.deepEqual(acts.map((act) => act.title), item.titles_after);
  });
}

test('plan performance maps to timeline fields', () => {
  assert.deepEqual(planSegmentFields('singing'), { performance_mode: 'singing', lyric_no_lip_sync: false });
  assert.deepEqual(planSegmentFields('speaking'), { performance_mode: 'speaking', lyric_no_lip_sync: false });
  assert.deepEqual(planSegmentFields('story'), { performance_mode: 'no_lip_sync', lyric_no_lip_sync: true });
  assert.deepEqual(planSegmentFields('instrumental'), { performance_mode: 'no_lip_sync', lyric_no_lip_sync: true });
});

test('scenes per spot is kept between 1 and 20', () => {
  assert.equal(normalizeSpotMinScenes('4'), 4);
  assert.equal(normalizeSpotMinScenes(0), 1);
  assert.equal(normalizeSpotMinScenes(99), 20);
  assert.equal(normalizeSpotMinScenes(''), 3);
  assert.equal(normalizeSpotMinScenes('x'), 3);
});

test(`fill cast: ${CASES.fill_cast.name}`, () => {
  const item = CASES.fill_cast;
  const { map, changed } = fillSceneCast(item.subject_scene_map, item.acts, item.subject_ids);
  assert.deepEqual(map, item.map_after);
  assert.deepEqual(changed, item.changed);
});
