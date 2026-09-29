const { functionSource, readBuilderModule } = require("./builder_source.cjs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const { test } = require("node:test");

const source = readBuilderModule("segments.mjs");
const c = vm.createContext({});
vm.runInContext(["scenePathKey", "rewriteRenamedScenePaths", "sortSegments", "renumberGenericBaseSceneLabels"].map((name) => functionSource(source, name)).join("\n"), c);

const root = "C:\\Projects\\song";
const renamed = [
  [`${root}\\rendered_scene_videos\\video_0017-audio.mp4`, `${root}\\rendered_scene_videos\\video_0016-audio.mp4`],
  [`${root}\\rendered_scene_videos_backup\\scene_0017`, `${root}\\rendered_scene_videos_backup\\scene_0016`],
  [`${root}\\rendered_scene_videos_backup\\scene_0017\\video_0017-audio_1.mp4`, `${root}\\rendered_scene_videos_backup\\scene_0016\\video_0016-audio_1.mp4`],
  [`${root}\\scene_image_previews\\scene_0017`, `${root}\\scene_image_previews\\scene_0016`],
];

test("stored scene paths follow renumbered files, including nested arrays and objects", () => {
  const segments = [{
    video_path: `${root}\\rendered_scene_videos\\video_0017-audio.mp4`,
    image_history: ["c:/projects/song/scene_image_previews/scene_0017/preview_1.png"],
    minimax_h3_settings: { stage_video: `${root}\\rendered_scene_videos_backup\\scene_0017\\video_0017-audio_1.mp4` },
    label: "Scene 16",
    custom_audio_path: `${root}\\scene_audio\\audio_0005.wav`,
  }];
  c.rewriteRenamedScenePaths(segments, renamed);
  assert.equal(segments[0].video_path, `${root}\\rendered_scene_videos\\video_0016-audio.mp4`);
  assert.equal(segments[0].image_history[0], `${root}\\scene_image_previews\\scene_0016/preview_1.png`);
  assert.equal(segments[0].minimax_h3_settings.stage_video, `${root}\\rendered_scene_videos_backup\\scene_0016\\video_0016-audio_1.mp4`);
  assert.equal(segments[0].label, "Scene 16");
  assert.equal(segments[0].custom_audio_path, `${root}\\scene_audio\\audio_0005.wav`);
});

test("a folder rename does not swallow a longer name that only shares its prefix", () => {
  const segments = [{ video_path: `${root}\\rendered_scene_videos_backup\\scene_00170\\x.mp4` }];
  c.rewriteRenamedScenePaths(segments, renamed);
  assert.equal(segments[0].video_path, `${root}\\rendered_scene_videos_backup\\scene_00170\\x.mp4`);
});

test("deleting a scene renumbers generic labels after it and keeps custom names", () => {
  const segments = [
    { start: 0, label: "Scene 1" },
    { start: 8, label: "4. Rooftop chorus" },
    { start: 12, label: "Fox girl bridge" },
    { start: 4, label: "Scene 3" },
    { start: 16, label: "" },
  ];
  assert.equal(c.renumberGenericBaseSceneLabels(segments), true);
  assert.deepEqual(segments.map((s) => s.label), ["Scene 1", "Scene 2", "3. Rooftop chorus", "Fox girl bridge", "Scene 5"]);
  assert.equal(c.renumberGenericBaseSceneLabels(segments), false);
});
