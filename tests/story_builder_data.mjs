import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  STORY_BUILDER_TABS, castWithoutImages, referenceImageGaps, sceneRangeText, storyBuilderBlocker, storyBuilderProgress,
  storySubjectCard, relinkStoryLocations, cardForSpot, planSpotsFromMapping, spotNameOfCard,
} from '../web/music_video_builder/story_builder_data.mjs';

const sceneIds = ['s1', 's2', 's3'];
const book = { title: 'T', chapters: [{ id: 'ch_1', title: 'One', setting_id: 'setting_home', paragraphs: ['p'] }] };
const acts = [{ id: 'act_1', chapter_id: 'ch_1', title: 'A', scene_ids: sceneIds }];

test('tabs follow the order a video is made', () => {
  assert.deepEqual(STORY_BUILDER_TABS.map((tab) => tab.id), ['song', 'cast', 'story', 'places', 'acts', 'scenes', 'prompts']);
});

test('an empty project has only empty tabs', () => {
  const progress = storyBuilderProgress({});
  assert.deepEqual(Object.values(progress), ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty']);
  assert.equal(storyBuilderBlocker('story', progress), 'Create the timeline scenes first (Song tab).');
});

test('progress moves tab by tab', () => {
  const refs = {
    subjects: [{ id: 'mara', name: 'Mara', reference_type: 'character', image: { path: 'C:/m.png' } },
      { id: 'joe', name: 'Joe', reference_type: 'character', image: {} }],
    locations: [{ id: 'loc_1', name: 'Home - Kitchen', story_setting_id: 'setting_home' }, { id: 'loc_2', name: 'Old' }],
    scene_map: { s1: 'loc_1', s2: 'loc_2' },
  };
  const layer = {
    story_book: book,
    story_acts: acts,
    story_scene_plan: [{ scene_id: 's1' }, { scene_id: 's2' }],
  };
  const segments = [{ story_beat: 'x' }, {}, {}];
  const progress = storyBuilderProgress({ sceneIds, refs, layer, segments });
  assert.deepEqual(progress, {
    song: 'done', cast: 'progress', story: 'done', acts: 'done', scenes: 'progress', places: 'progress', prompts: 'progress',
  });
  assert.deepEqual(castWithoutImages(refs), ['Joe']);
  assert.equal(storyBuilderBlocker('places', progress), '');
});

test('image gaps name the cards scenes use without an image, and the scenes with none', () => {
  const refs = {
    subjects: [{ id: 'mara', name: 'Mara', image: { path: 'C:/m.png' } }, { id: 'joe', name: 'Joe', image: {} },
      { id: 'ann', name: 'Ann', image: {} }, { id: 'ref', name: 'Ref', source: 'refmod', refmod: { name: 'Ref' } }],
    locations: [{ id: 'loc_1', name: 'Home - Kitchen', image: {} }, { id: 'loc_2', name: 'Street', image: { data: 'x' } }],
    subject_scene_map: { s1: ['mara', 'joe'], s2: ['joe'], s3: ['ref'], s4: 'joe' },
    scene_map: { s1: 'loc_1', s2: 'loc_1', s4: 'loc_2' },
  };
  assert.deepEqual(referenceImageGaps(refs, ['s1', 's2', 's3', 's4', 's5']), {
    characters: ['Joe'], locations: ['Home - Kitchen'], scenes: [2, 5],
  });
  assert.equal(sceneRangeText([48, 49, 50, 52]), 'Scenes 48-50, 52');
  assert.equal(sceneRangeText([7]), 'Scene 7');
});

test('used location cards that lost their story link are linked again by name', () => {
  const settings = [{ id: 'setting_home', name: 'Home' }, { id: 'setting_home_office', name: 'Home Office' }];
  const refs = {
    locations: [{ id: 'a', name: 'Home - Kitchen' }, { id: 'b', name: 'Home Office - Desk' }, { id: 'c', name: 'Home - Porch' },
      { id: 'd', name: 'Street' }, { id: 'e', name: 'Home', story_setting_id: 'setting_other' }],
    scene_map: { s1: 'a', s2: 'b', s3: 'd', s4: 'e' },
  };
  const fixed = relinkStoryLocations(refs, settings);
  assert.deepEqual(fixed.locations.map((card) => card.story_setting_id), ['setting_home', 'setting_home_office', undefined, undefined, 'setting_other']);
  assert.equal(relinkStoryLocations(fixed, settings), null, 'nothing left to repair');
  assert.equal(storyBuilderProgress({ sceneIds: ['s1', 's2'], refs: fixed }).places, 'done');
});

test('places are done once every scene has a story location, with or without images', () => {
  const refs = { locations: [{ id: 'loc_1', name: 'Home - Kitchen', story_setting_id: 'setting_home', image: {} }],
    scene_map: { s1: 'loc_1', s2: 'loc_1' } };
  assert.equal(storyBuilderProgress({ sceneIds, refs }).places, 'progress');
  refs.scene_map.s3 = 'loc_1';
  assert.equal(storyBuilderProgress({ sceneIds, refs }).places, 'done', 'location images are optional');
});

test('places built from the user list are done before any scene is planned, and the tab only needs the story', () => {
  const layer = { story_book: book, story_user_locations: "- Grandma's kitchen: yellow wallpaper\nThe night train" };
  const refs = { locations: [{ id: 'a', name: "Grandma's Kitchen" }, { id: 'b', name: 'Unrelated' }] };
  const progress = storyBuilderProgress({ sceneIds, refs, layer });
  assert.equal(progress.places, 'done');
  assert.equal(storyBuilderProgress({ sceneIds, refs: { locations: [{ id: 'b', name: 'Unrelated' }] }, layer }).places, 'empty');
  const planned = storyBuilderProgress({ sceneIds, refs, layer: { ...layer, story_scene_plan: [{ scene_id: 's1' }] } });
  assert.equal(planned.places, 'progress', 'once scenes are planned, every scene needs a place');
  assert.equal(storyBuilderBlocker('places', progress), '');
  assert.equal(storyBuilderBlocker('places', storyBuilderProgress({})), 'Write the story first (Story tab).');
  assert.equal(storyBuilderBlocker('acts', progress), '', 'acts can start once the places exist');
  const noPlaces = storyBuilderProgress({ sceneIds, layer: { story_book: book } });
  assert.equal(storyBuilderBlocker('acts', noPlaces), 'Build the places first (Places tab); the acts happen inside them.');
  const hasActs = storyBuilderProgress({ sceneIds, layer: { story_book: book, story_acts: acts } });
  assert.equal(storyBuilderBlocker('acts', hasActs), '', 'existing acts can still be rebuilt without place cards');
});

test('broken acts block the scene plan', () => {
  const layer = { story_book: book, story_acts: [{ id: 'act_1', chapter_id: 'ch_1', scene_ids: ['s1'] }] };
  const progress = storyBuilderProgress({ sceneIds, layer });
  assert.equal(progress.acts, 'progress');
  assert.equal(storyBuilderBlocker('scenes', progress), 'Create valid acts first (Acts tab).');
});

test('a found character becomes a subject card', () => {
  assert.deepEqual(storySubjectCard({ name: ' The Conductor ', role: 'supporting', description: 'Navy uniform.' }, 'subj_1'), {
    id: 'subj_1', name: 'The Conductor', description: 'Navy uniform.', reference_type: 'character',
    image: { path: '', data: '', name: '' },
  });
});

const placeSettings = [{ id: 'setting_home', name: 'Home' }, { id: 'setting_train', name: 'The Night Train' }];
const placeCards = [
  { id: 'loc_k', name: 'Home - Kitchen', story_setting_id: 'setting_home' },
  { id: 'loc_s', name: 'The Night Train - Sleeper Car', story_setting_id: 'setting_train' },
  { id: 'loc_g', name: "Grandma's kitchen" },
];

test('a card\'s spot name drops the setting prefix and a spot finds its card', () => {
  assert.equal(spotNameOfCard(placeCards[0], placeSettings), 'Kitchen');
  assert.equal(spotNameOfCard(placeCards[1], placeSettings), 'Sleeper Car');
  assert.equal(spotNameOfCard(placeCards[2], placeSettings), "Grandma's kitchen", 'a card with no story link keeps its name');
  assert.equal(cardForSpot(placeCards, 'Kitchen', placeSettings).id, 'loc_k', 'the spot name');
  assert.equal(cardForSpot(placeCards, 'home - kitchen', placeSettings).id, 'loc_k', 'the card name, any case');
  assert.equal(cardForSpot(placeCards, 'The sleeper car', placeSettings).id, 'loc_s', 'words of the shorter name inside the longer');
  assert.equal(cardForSpot(placeCards, 'Rooftop', placeSettings), null);
  assert.equal(cardForSpot(placeCards, '', placeSettings), null);
});

test('the scene plan spots follow the location mapping', () => {
  const plan = [{ scene_id: 's1', spot: 'Kitchen' }, { scene_id: 's2', spot: 'Kitchen' }, { scene_id: 's3', spot: 'Bench' }];
  const refs = { locations: placeCards, scene_map: { s1: 'loc_k', s2: 'loc_s' } };
  assert.deepEqual(planSpotsFromMapping(plan, refs, placeSettings), [
    { scene_id: 's1', spot: 'Kitchen' }, { scene_id: 's2', spot: 'Sleeper Car' }, { scene_id: 's3', spot: 'Bench' },
  ], 'an unmapped scene keeps its spot');
  assert.equal(planSpotsFromMapping(plan.slice(0, 1), refs, placeSettings), null, 'nothing changes, nothing is saved');
});
