const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const source = functionSource(readBuilderModule('timeline_actions.mjs'), 'deleteSelectedMedia');

async function remove(scene, type) {
  const deleted = [];
  const context = vm.createContext({
    activeSegment: () => scene,
    selectedSegmentVideoPath: (item) => item.video_path || '',
    selectedSegmentImagePath: (item) => item.image_history?.[item.image_history_index] || item.approved_image_path || item.custom_image_path || '',
    selectedSegmentVideoThumbnailPath: () => '',
    confirmDeleteMediaAction: async () => true,
    confirmDestructiveAction: async () => ({ confirmed: true }),
    postJson: async (_, data) => { deleted.push(data.path); },
    projectInput: { value: '/project' },
    pushHistory() {},
    mediaPathKey: (path) => String(path || ''),
    normalizeSegmentVideoHistory() {},
    segmentImageSource: (item) => item.image_history?.length || item.custom_image_data || item.approved_image_path ? { path: item.approved_image_path } : null,
    ensureSegmentRuntimeFields() {},
    deleteStaleSceneLatents: async () => ({ deleted: true }),
    syncPreview() {}, syncInspector() {}, renderList() {}, render() {},
    autoSaveSessionQuiet: async () => {}, updateSelectedMediaTools() {}, toast() {},
  });
  vm.runInContext(`let mediaDeleteInFlight = false;\n${source}\nglobalThis.remove = deleteSelectedMedia;`, context);
  await context.remove({ segment: scene, type });
  return deleted;
}

test('Delete video removes the clicked scene video even while image preview is selected', async () => {
  const scene = { preview_mode: 'image', video_path: 'clip.mp4', video_history: ['clip.mp4'], video_history_index: 0, image_history: ['still.png'], image_history_index: 0 };
  assert.deepEqual(await remove(scene, 'video'), ['clip.mp4']);
  assert.equal(scene.video_path, '');
  assert.deepEqual(scene.image_history, ['still.png']);
});

test('Delete image removes the clicked scene image even while video preview is selected', async () => {
  const scene = { preview_mode: 'video', video_path: 'clip.mp4', image_history: ['still.png'], image_history_index: 0, approved_image_path: 'still.png' };
  assert.deepEqual(await remove(scene, 'image'), ['still.png']);
  assert.deepEqual(scene.image_history, []);
  assert.equal(scene.video_path, 'clip.mp4');
});

test('Delete image clears an unsaved in-memory image without a file request', async () => {
  const scene = { preview_mode: 'image', custom_image_data: 'data:image/png;base64,AA', image: {} };
  assert.deepEqual(await remove(scene, 'image'), []);
  assert.equal(scene.custom_image_data, '');
  assert.equal(scene.image, null);
});
