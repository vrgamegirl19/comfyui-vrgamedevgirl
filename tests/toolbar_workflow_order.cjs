const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function fixture() {
  const names = [
    'wizardButton', 'autoBuildButton', 'storyboardBuilderButton', 'fluxReferenceBuilderButton',
    'refModsStudioButton', 'refModsViewerButton', 'lyricMapperButton', 'gemmaRunnerButton', 'promptOptionsButton',
  ];
  const importActions = {
    children: [],
    append(...buttons) {
      for (const button of buttons) {
        this.children = this.children.filter(child => child !== button);
        this.children.push(button);
      }
    },
  };
  const state = { videoType: 'singing' };
  const c = vm.createContext({ importActions, state, videoTypeSelect: {} });
  for (const name of names) c[name] = { name, onclick: () => name };
  for (const name of ['normalizeVideoType', 'normalizeProjectVideoType']) {
    vm.runInContext(functionSource(readBuilderModule('controls.mjs'), name), c);
  }
  vm.runInContext(functionSource(readBuilderModule('toolbar.mjs'), 'syncWorkflowActionOrder'), c);
  vm.runInContext(readBuilderModule('notification_sounds.mjs'), c);
  c.sync = c.createNotificationSounds({ state, videoTypeSelect: c.videoTypeSelect,
    syncWorkflowActionOrder: c.syncWorkflowActionOrder }).syncVideoTypeControl;
  return { c, state, names, order: () => importActions.children.map(button => button.name) };
}

test('Singing workflow starts with Wizard, Auto Build, Ref Builder, Line Mapping, Story Builder', () => {
  const { c, order } = fixture();
  c.sync();
  assert.deepEqual(order(), [
    'wizardButton', 'autoBuildButton', 'fluxReferenceBuilderButton', 'lyricMapperButton', 'storyboardBuilderButton',
    'refModsStudioButton', 'refModsViewerButton', 'gemmaRunnerButton', 'promptOptionsButton',
  ]);
});

for (const videoType of ['speaking', 'no_lip_sync', 'story']) {
  test(`${videoType} keeps its original toolbar order when switching from Singing and back`, () => {
    const { c, state, names, order } = fixture();
    c.sync();
    state.videoType = videoType;
    c.sync();
    assert.deepEqual(order(), names);
    assert.equal(c.videoTypeSelect.value, videoType);
    state.videoType = 'singing';
    c.sync();
    assert.equal(order()[2], 'fluxReferenceBuilderButton');
    assert.equal(order()[3], 'lyricMapperButton');
    assert.equal(order()[4], 'storyboardBuilderButton');
  });
}

test('loading a project or configuring Wizard updates the order using the existing video-type sync', () => {
  const { c, state, names, order } = fixture();
  for (const videoType of ['story_mode', 'speaking', 'singing', 'no_lip_sync', 'singing']) {
    state.videoType = videoType;
    c.sync();
    c.sync();
    assert.equal(order().length, names.length);
    for (const name of names) {
      const button = c.importActions.children.find(item => item.name === name);
      assert.equal(button, c[name]);
      assert.equal(button.onclick(), name);
    }
    assert.equal(c.videoTypeSelect.value, videoType === 'story_mode' ? 'story' : videoType);
    if (videoType !== 'singing') assert.deepEqual(order(), names);
  }
});
