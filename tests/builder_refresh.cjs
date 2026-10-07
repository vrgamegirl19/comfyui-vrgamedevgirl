const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const context = vm.createContext({});
vm.runInContext(`${readBuilderModule('builder_refresh.mjs')}
  globalThis.queueRefresh = queueBuilderRefresh;
  globalThis.takeRefresh = takeBuilderRefresh;`, context);

function storage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

test('Builder refresh restores a saved project and view only once', () => {
  const session = storage();
  const view = { projectFolder: 'C:/projects/song', activeId: 'scene-3', inspectorTab: 'video' };
  context.queueRefresh(session, view, 1000);
  const restored = context.takeRefresh(session, 2000);
  assert.equal(restored.projectFolder, view.projectFolder);
  assert.equal(restored.activeId, view.activeId);
  assert.equal(restored.inspectorTab, view.inspectorTab);
  assert.equal(context.takeRefresh(session, 2001), null);
});

test('Builder refresh ignores expired or invalid handoffs', () => {
  const session = storage();
  context.queueRefresh(session, { projectFolder: 'C:/projects/song' }, 1000);
  assert.equal(context.takeRefresh(session, 1000 + 5 * 60 * 1000 + 1), null);
  context.queueRefresh(session, { projectFolder: '' }, 1000);
  assert.equal(context.takeRefresh(session, 1001), null);
  session.setItem('vrgdg:music-builder:refresh-resume', '{bad json');
  assert.equal(context.takeRefresh(session, 1001), null);
});

test('Builder refresh saves first and reloads only after a successful save', async () => {
  const events = [];
  let saveResult = { project_folder: 'C:/projects/song' };
  const refreshContext = vm.createContext({
    activeProjectFolderForSave: () => 'C:/projects/song',
    saveSession: async () => { events.push('save'); return saveResult; },
    queueBuilderRefresh: (_storage, view) => { events.push('queue'); assert.equal(view.activeId, 'scene-3'); },
    window: { sessionStorage: storage(), location: { reload: () => events.push('reload') } },
    node: { id: 12 },
    state: { activeId: 'scene-3', inspectorTab: 'video', leftPanelTab: 'scenes' },
    timelineViewport: { scrollLeft: 50 }, sceneListPane: { scrollTop: 80 },
    toast: (message) => events.push(message),
  });
  const refreshSource = functionSource(readBuilderModule('builder.mjs'), 'refreshBuilder');
  vm.runInContext(`${refreshSource}\nglobalThis.refresh = refreshBuilder;`, refreshContext);
  await refreshContext.refresh();
  assert.deepEqual(events, ['save', 'queue', 'reload']);
  events.length = 0;
  saveResult = { stale: true };
  await refreshContext.refresh();
  assert.equal(events[0], 'save');
  assert.equal(events.length, 2);
  assert.match(events[1], /could not be saved/);
});
