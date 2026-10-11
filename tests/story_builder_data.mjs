import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  STORY_BUILDER_TABS, castWithoutImages, storyBuilderBlocker, storyBuilderProgress, storySubjectCard,
} from '../web/music_video_builder/story_builder_data.mjs';

const sceneIds = ['s1', 's2', 's3'];
const book = { title: 'T', chapters: [{ id: 'ch_1', title: 'One', setting_id: 'setting_home', paragraphs: ['p'] }] };
const acts = [{ id: 'act_1', chapter_id: 'ch_1', title: 'A', scene_ids: sceneIds }];

test('tabs follow the order a video is made', () => {
  assert.deepEqual(STORY_BUILDER_TABS.map((tab) => tab.id), ['song', 'cast', 'story', 'acts', 'scenes', 'places', 'prompts']);
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
