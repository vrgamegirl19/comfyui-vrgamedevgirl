const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const openMenuSource = functionSource(readBuilderModule('timeline_edit.mjs'), 'openSegmentContextMenu');

function openFor(segment) {
  const calls = [];
  let menu;
  const makeElement = () => ({
    style: {}, children: [],
    append(child) { this.children.push(child); },
    remove() {},
    contains(target) { return this === target || this.children.includes(target); },
    getBoundingClientRect() { return { width: 190, height: 300 }; },
  });
  const context = vm.createContext({
    state: { segments: [segment], multiSelectMode: false, projectFolder: '/project' },
    document: {
      querySelector: () => null,
      createElement: makeElement,
      body: { append(element) { menu = element; } },
    },
    window: { innerWidth: 1200, innerHeight: 800, addEventListener() {}, removeEventListener() {} },
    setTimeout() {},
    makeButton(label) { return { label, style: {}, disabled: false, onclick: null }; },
    pauseTimelineForEditing() {}, setActiveSegment() {},
    segmentTrack: () => 'base', baseSceneVideoTrimKind: () => '',
    currentGlobalTime: () => 2, projectInput: { value: '/project' },
    selectedSegmentVideoPath: (scene) => scene.video_path || '',
    selectedSegmentImagePath: (scene) => scene.image_path || '',
    selectedSegmentsForBatch: () => [], isSegmentMultiSelected: () => false,
    renderScenesFromMenu() {}, restoreVideoForSegment() {},
    mergeAdjacentBaseScene: async () => {}, hasLockedVideo: () => false,
    copyBaseSceneAsOverlay() {}, closeTimelineGapsFromMenu() {}, openSceneOptions() {},
    deleteSelectedMedia(options) { calls.push(['delete', options.type, options.segment]); },
    captureSelectedVideoFrameAsImage(scene) { calls.push(['frame', scene]); },
    toast() {},
  });
  vm.runInContext(`${openMenuSource}\nglobalThis.openMenu = openSegmentContextMenu;`, context);
  context.openMenu({ preventDefault() {}, stopPropagation() {}, clientX: 200, clientY: 300, currentTarget: { getBoundingClientRect: () => ({ left: 100, width: 200 }) } }, segment);
  return { labels: menu.children.map((button) => button.label), click: (label) => menu.children.find((button) => button.label === label).onclick(), calls };
}

test('scene menu only offers media actions for media present on that scene', () => {
  for (const [scene, expected] of [
    [{ id: 'empty', start: 0, end: 5 }, []],
    [{ id: 'image', start: 0, end: 5, image_path: 'still.png' }, ['Delete image']],
    [{ id: 'video', start: 0, end: 5, video_path: 'clip.mp4' }, ['Use frame as image', 'Delete video']],
    [{ id: 'both', start: 0, end: 5, image_path: 'still.png', video_path: 'clip.mp4' }, ['Use frame as image', 'Delete video', 'Delete image']],
  ]) {
    const menu = openFor(scene);
    assert.deepEqual(menu.labels.filter((label) => ['Use frame as image', 'Delete video', 'Delete image', 'Delete scene'].includes(label)), expected);
  }
});

test('scene media actions keep their clicked scene and media type', () => {
  const scene = { id: 'both', start: 0, end: 5, image_path: 'still.png', video_path: 'clip.mp4' };
  const menu = openFor(scene);
  menu.click('Use frame as image');
  menu.click('Delete video');
  menu.click('Delete image');
  assert.deepEqual(menu.calls.map(([action, type]) => [action, type === scene ? 'scene' : type]), [
    ['frame', 'scene'], ['delete', 'video'], ['delete', 'image'],
  ]);
  assert.ok(menu.calls.every((call) => call.at(-1) === scene));
});
