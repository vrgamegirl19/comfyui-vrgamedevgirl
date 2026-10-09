const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {readBuilderModule, functionSource} = require('./builder_source.cjs');

// Runs the Builder's stitchRenderedScenes with stubs and returns the payload it posts to the stitcher.
async function stitchPayload({builtIn, song = '/proj/song.wav', engine = 'minimax_h3'}) {
  let posted = null;
  const c = vm.createContext({
    Math, Number, String, Array, Date, Error, console,
    state: {segments: [], projectVideoEngine: engine, overlayTrack: {enabled: false}, overlaySegments: [], i2vVideoSettings: {}},
    normalizeProjectVideoEngine: (value) => value,
    overlayClipIsEnabled: () => false,
    ensureSceneLutsAppliedBeforeStitch: async () => {},
    ensureSceneAdjustsAppliedBeforeStitch: async () => {},
    ensureSceneFilmGrainAppliedBeforeStitch: async () => {},
    selectedSegmentVideoPath: (segment) => segment.video_path,
    currentProjectAudioPath: () => song,
    miniMaxH3SettingsForSegment: () => ({audio_mode: builtIn ? 'built_in_audio' : 'input_audio'}),
    currentVideoMode: () => 'image_to_video',
    usingSceneAudioMode: () => false,
    audioSourceStart: () => 0,
    audioChunkDuration: () => 0,
    sceneDisplayName: (_segment, index) => `Scene ${index + 1}`,
    isLikelyVideoPath: () => true,
    DEFAULT_LTX_INGREDIENTS_WIDTH: 1280,
    DEFAULT_LTX_INGREDIENTS_HEIGHT: 720,
    projectInput: {value: '/proj'},
    postJson: async (url, body) => { posted = {url, body}; return {final_video_path: '/proj/PREVIEW.mp4'}; },
  });
  vm.runInContext(functionSource(readBuilderModule('video_render.mjs'), 'stitchRenderedScenes'), c);
  const segments = [
    {start: 0, end: 6.76, video_path: '/proj/rendered_scene_videos/video_0001-audio.mp4'},
    {start: 6.76, end: 11.62, video_path: '/proj/rendered_scene_videos/video_0002-audio.mp4'},
  ];
  await c.stitchRenderedScenes(null, {segments, timelineOffset: 0, audioStart: 0, audioDuration: 11.62});
  assert.equal(posted.url, '/vrgdg/workflow_runner/stitch_scene_videos');
  return posted.body;
}

test('embedded-audio stitch sends the project song as song_path, not as the mux audio', async () => {
  const body = await stitchPayload({builtIn: true});
  assert.equal(body.use_embedded_scene_audio, true);
  assert.equal(body.audio_path, '');
  assert.equal(body.song_path, '/proj/song.wav');
});

test('project-song stitch keeps audio_path and sends no song_path', async () => {
  const body = await stitchPayload({builtIn: false});
  assert.equal(body.use_embedded_scene_audio, false);
  assert.equal(body.audio_path, '/proj/song.wav');
  assert.equal(body.song_path, '');
});
