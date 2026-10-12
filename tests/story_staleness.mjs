// Story Builder change tracking: a change marks later steps out of date, never earlier ones.
import assert from 'node:assert/strict';
import { test } from 'node:test';

const { fillStoryStamps, stampStory, storyStaleness } = await import('../web/music_video_builder/story_staleness.mjs');

function project() {
  const segments = Array.from({ length: 4 }, (_, index) => ({
    id: `s${index + 1}`, start: index * 5, end: (index + 1) * 5, story_beat: `beat ${index + 1}`, minimax_h3_prompt: `prompt ${index + 1}`,
  }));
  const layer = {
    overall_story_idea: 'A girl leaves home.',
    story_book: { chapters: [
      { id: 'ch_1', title: 'Home', setting_id: 'setting_home', paragraphs: ['Mara packs.'] },
      { id: 'ch_2', title: 'Away', setting_id: 'setting_train', paragraphs: ['Mara rides.'] },
    ] },
    story_settings: [{ id: 'setting_home', name: 'Home', look: 'Pine.' }, { id: 'setting_train', name: 'Train', look: 'Steel.' }],
    story_acts: [
      { id: 'a1', chapter_id: 'ch_1', title: 'Pack', summary: 'Packing.', cast_ids: ['mara'], scene_ids: ['s1', 's2'] },
      { id: 'a2', chapter_id: 'ch_2', title: 'Ride', summary: 'Riding.', cast_ids: ['joe'], scene_ids: ['s3', 's4'] },
    ],
    story_scene_plan: segments.map((segment, index) => ({ scene_id: segment.id, act_id: index < 2 ? 'a1' : 'a2', story_slice: `slice ${index + 1}`, spot: index < 2 ? 'Kitchen' : 'Car' })),
  };
  const refs = {
    subjects: [{ id: 'mara', name: 'Mara', description: 'Red coat.' }, { id: 'joe', name: 'Joe', description: 'Conductor.' }],
    locations: [{ id: 'loc_k', name: 'Home - Kitchen', description: 'Pine kitchen.' }, { id: 'loc_c', name: 'Train - Car', description: 'Steel car.' }],
    scene_map: { s1: 'loc_k', s2: 'loc_k', s3: 'loc_c', s4: 'loc_c' },
    subject_scene_map: { s1: ['mara'], s2: ['mara'], s3: ['joe'], s4: ['joe'] },
  };
  const input = { layer, refs, segments, promptKey: 'minimax_h3_prompt' };
  input.layer.story_stamps = fillStoryStamps(input);
  return input;
}

const ids = (section) => Object.keys(section).sort();

test('a fresh project has every step stamped and nothing out of date', () => {
  const input = project();
  const stale = storyStaleness(input);
  assert.deepEqual(stale.book, []);
  for (const section of ['acts', 'plan', 'places', 'beats', 'prompts']) assert.deepEqual(stale[section], {}, section);
  assert.equal(fillStoryStamps(input), null, 'a second fill changes nothing');
});

test('changing one chapter marks only its acts and everything after them', () => {
  const input = project();
  input.layer.story_book.chapters[0].paragraphs = ['Mara packs and cries.'];
  const stale = storyStaleness(input);
  assert.deepEqual(stale.book, [], 'the story itself is not out of date (no trickle up)');
  assert.deepEqual(stale.acts, { a1: ['chapter text changed'] });
  assert.deepEqual(stale.plan, { s1: ['its act is out of date'], s2: ['its act is out of date'] });
  assert.deepEqual(ids(stale.places), ['s1', 's2']);
  assert.deepEqual(ids(stale.beats), ['s1', 's2']);
  assert.deepEqual(stale.prompts.s1, ['its beat is out of date']);
  assert.equal(stale.prompts.s3, undefined, 'the other chapter is untouched');
});

test('rebuilt acts are stamped and their scenes then need a replan', () => {
  const input = project();
  input.layer.story_book.chapters[0].paragraphs = ['Mara packs and cries.'];
  input.layer.story_acts[0].summary = 'Packing, in tears.';
  input.layer.story_stamps = stampStory(input, { acts: ['a1'] });
  const stale = storyStaleness(input);
  assert.deepEqual(stale.acts, {});
  assert.deepEqual(stale.plan, { s1: ['act changed'], s2: ['act changed'] });
  input.layer.story_stamps = stampStory(input, { plan: ['s1', 's2'] });
  assert.deepEqual(storyStaleness(input).plan, {}, 'replanning the act brings its scenes up to date');
});

test('a cast change marks the story and the acts and scenes with that person', () => {
  const input = project();
  input.refs.subjects[1].description = 'Old conductor with a limp.';
  const stale = storyStaleness(input);
  assert.deepEqual(stale.book, ['cast changed']);
  assert.deepEqual(stale.acts.a2, ['cast changed', 'the story is out of date']);
  assert.deepEqual(stale.acts.a1, ['the story is out of date']);
  assert.ok(stale.beats.s3.includes('cast changed'));
  assert.ok(!stale.beats.s1.includes('cast changed'), 'scenes without Joe only follow the story');
});

test('keeping the story as is clears everything after it', () => {
  const input = project();
  input.layer.overall_story_idea = 'A girl runs away.';
  assert.deepEqual(storyStaleness(input).book, ['premise changed']);
  assert.ok(Object.keys(storyStaleness(input).prompts).length === 4);
  input.layer.story_stamps = stampStory(input, { book: true });
  const stale = storyStaleness(input);
  assert.deepEqual([stale.book, stale.acts, stale.prompts], [[], {}, {}]);
});

test('editing a beat makes only its prompt out of date', () => {
  const input = project();
  input.segments[2].story_beat = 'Joe punches a ticket.';
  const stale = storyStaleness(input);
  assert.deepEqual(stale.beats, {});
  assert.deepEqual(stale.prompts, { s3: ['beat changed'] });
  assert.deepEqual(stale.plan, {});
});

test('a setting look change marks the places and beats of that setting', () => {
  const input = project();
  input.layer.story_settings[1].look = 'Rusty steel.';
  const stale = storyStaleness(input);
  assert.deepEqual(stale.places, { s3: ['setting look changed'], s4: ['setting look changed'] });
  assert.deepEqual(stale.beats.s3, ['its location is out of date']);
  assert.deepEqual(stale.acts, {});
});

test('fill drops stamps of outputs that are gone and stamps new ones', () => {
  const input = project();
  input.segments[0].story_beat = '';
  input.layer.story_acts.push({ id: 'a3', chapter_id: 'ch_2', title: 'Arrive', summary: '', cast_ids: [], scene_ids: [] });
  const filled = fillStoryStamps(input);
  assert.equal(filled.beats.s1, undefined);
  assert.ok(filled.acts.a3);
});
