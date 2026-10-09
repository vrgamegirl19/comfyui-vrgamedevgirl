const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture(videoType) {
  // Same scene/audio ordering that sent Scene 3's clip to Scene 4's latent lookup.
  const scenes = [0, 6.125, 10.73, 15.157, 21.699].map((start, index) => ({
    id: `s${index + 1}`, start, track: 'base',
    custom_audio_timeline_start: [0, 8, 16, 24, 21.969][index],
    video_path: `video_000${index + 1}-audio.mp4`, video_history: [],
  }));
  const overlays = [{ id: 'insert1', start: 14, track: 'overlay' },
    { id: 'insert2', start: 22, track: 'overlay' }];
  const c = vm.createContext({ state: { videoType, segments: scenes }, scenes, overlays,
    allEditableSegments: () => [...scenes, ...overlays],
    segmentTrack: scene => scene.track,
    segmentIndexInfo: scene => ({ index: (scene.track === 'base' ? scenes : overlays).indexOf(scene) }),
    audioTimelineStart: scene => scene.custom_audio_timeline_start ?? scene.start,
  });
  vm.runInContext(functionSource(readBuilderModule('scene_render_prep.mjs'),
    'previousAutoChainSourceSegment'), c);
  return c;
}

for (const videoType of ['speaking', 'singing', 'no_lip_sync']) {
  test(`${videoType}: continuity uses Scene 4 before Scene 5 despite independent audio timing`, () => {
    const c = fixture(videoType);
    const original = JSON.stringify(c.scenes);
    assert.equal(c.previousAutoChainSourceSegment(c.scenes[4]).id, 's4');
    assert.equal(c.previousAutoChainSourceSegment(c.scenes[3]).id, 's3');
    assert.equal(c.previousAutoChainSourceSegment(c.scenes[0]), null);
    assert.equal(JSON.stringify(c.scenes), original);
  });
}

test('continuity stays within each track and uses scene index to break equal-time ties', () => {
  const c = fixture('speaking');
  assert.equal(c.previousAutoChainSourceSegment(c.overlays[1]).id, 'insert1');
  assert.equal(c.previousAutoChainSourceSegment(c.overlays[0]), null);
  c.scenes[3].start = c.scenes[2].start;
  assert.equal(c.previousAutoChainSourceSegment(c.scenes[3]).id, 's3');
});

test('Scene 5 continuity request sends Scene 4 video to the Scene 4 latent lookup', async () => {
  const c = fixture('speaking');
  let request;
  Object.assign(c, {
    miniMaxH3ContinuityModeForSegment: () => 'latent_continuation_masked',
    miniMaxH3SettingsForSegment: () => ({ continuity_prompt_from_last_frame: false }),
    isMiniMaxH3LatentContinuationMode: () => true,
    sceneSlotNumber: scene => c.scenes.indexOf(scene) + 1,
    projectInput: { value: 'project' },
    selectedSegmentVideoPath: scene => scene.video_path,
    postJson: async (route, payload) => {
      request = payload;
      assert.equal(route, '/vrgdg/music_builder/check_latent_predecessor');
      assert.equal(payload.scene_number, 5);
      assert.equal(payload.predecessor_video_path, 'video_0004-audio.mp4');
      return { predecessor_exists: true, take_status: 'already_active' };
    },
  });
  vm.runInContext(functionSource(readBuilderModule('video_render.mjs'),
    'prepareMiniMaxH3ContinuityReference'), c);
  const result = await c.prepareMiniMaxH3ContinuityReference(c.scenes[4]);
  assert.ok(request);
  assert.equal(result.previousSegment.id, 's4');
  assert.equal(c.scenes[4].minimax_h3_continuity_source_scene_id, 's4');
});
