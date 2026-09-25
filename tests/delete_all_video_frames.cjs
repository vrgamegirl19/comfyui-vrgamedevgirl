const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../web/VRGDG_MusicVideoBuilderUI.js'), 'utf8');
const start = source.indexOf('async function deleteAllTimelineVideos()');
const fn = source.slice(start, source.indexOf('async function deleteAllTimelineImages()', start));
async function run(scene, answers, failPath = '') {
  const deleted = [], messages = [];
  let saved = false;
  const player = { pause() {}, removeAttribute() {}, load() {}, dataset: {}, style: {} };
  const c = vm.createContext({
    allEditableSegments: () => [scene], window: { confirm: () => answers.shift() },
    state: {}, projectInput: { value: '/project' }, deleteAllTimelineVideosButton: {},
    pauseTimelineForEditing() {}, pushHistory() {},
    postJson: async (url, data) => { if (data.path === failPath) throw Error('locked'); deleted.push(data.path); },
    ensureSegmentRuntimeFields() {}, deleteStaleSceneLatents: async () => {},
    previewVideo: player, previewImage: player, sceneAudio: player,
    syncInspector() {}, syncPreview() {}, activeSegment() {}, renderList() {}, render() {}, updateSelectedMediaTools() {},
    autoSaveSessionQuiet: async () => { saved = true; }, toast: message => messages.push(message),
  });
  vm.runInContext(fn, c);
  await c.deleteAllTimelineVideos();
  return { deleted, messages, saved };
}
test('removes old captured-frame history even when videos were already deleted', async () => {
  const scene = { image_history: ['frame1.png', 'frame2.png'], image_history_index: 1, custom_image_path: 'frame2.png',
    flf_rendered_start_frame_path: 'chain.png', minimax_h3_continuity_frame_path: 'continuity.png',
    custom_image_data: 'data:image/png;base64,AA', image: {}, minimax_h3_prompt: 'keep prompt',
    minimax_h3_reference_keys: ['subject:alice'] };
  const result = await run(scene, [true, true]);
  assert.deepEqual(result.deleted, ['frame1.png', 'frame2.png', 'chain.png', 'continuity.png']);
  assert.equal(scene.image_history.length, 0);
  assert.equal(scene.custom_image_path, '');
  assert.equal(scene.custom_image_data, '');
  assert.equal(scene.flf_rendered_start_frame_path, '');
  assert.equal(scene.image_assignment_cleared, true);
  assert.equal(scene.minimax_h3_prompt, 'keep prompt');
  assert.deepEqual(scene.minimax_h3_reference_keys, ['subject:alice']);
  assert.equal(result.saved, true);
});
test('declining scene-image deletion preserves images while deleting videos', async () => {
  const scene = { video_path: 'clip.mp4', image_history: ['image.png'], approved_image_path: 'image.png' };
  const result = await run(scene, [true, false]);
  assert.deepEqual(result.deleted, ['clip.mp4']);
  assert.equal(scene.video_path, '');
  assert.deepEqual(scene.image_history, ['image.png']);
  assert.equal(scene.approved_image_path, 'image.png');
});
test('cancelling the initial confirmation deletes nothing', async () => {
  const scene = { video_path: 'clip.mp4', image_history: ['frame.png'] };
  const result = await run(scene, [false]);
  assert.deepEqual(result.deleted, []);
  assert.equal(scene.video_path, 'clip.mp4');
});
test('a frame deletion failure preserves image references for retry', async () => {
  const scene = { image_history: ['frame.png'], custom_image_path: 'frame.png' };
  const result = await run(scene, [true, true], 'frame.png');
  assert.equal(scene.custom_image_path, 'frame.png');
  assert.deepEqual(scene.image_history, ['frame.png']);
  assert.equal(result.saved, false);
  assert.match(result.messages[0], /could not be deleted/);
});
