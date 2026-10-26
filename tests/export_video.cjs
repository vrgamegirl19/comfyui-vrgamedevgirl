const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

// The Export button of the timeline: when it is allowed, and what it does when it is clicked.
function harness({ postJson, stitch } = {}) {
  const log = [];
  const context = vm.createContext({
    toast: (message, isError) => log.push(['toast', message, Boolean(isError)]),
    showFinalVideoReadyModal: (path) => log.push(['ready', path]),
    selectedSegmentVideoPath: (segment) => segment?.video_path || '',
    postJson: async (url, payload) => {
      log.push(['post', url, payload]);
      return postJson(url, payload);
    },
  });
  vm.runInContext(functionSource(readBuilderModule('controls.mjs'), 'normalizeProjectVideoEngine'), context);
  vm.runInContext(readBuilderModule('export_video.mjs'), context);
  const button = { style: {}, disabled: true, textContent: '', title: '' };
  const progress = { set: (text, pct) => log.push(['progress', text, pct]), close: () => log.push(['progress closed']) };
  const wired = { button, state: null, projectInput: { value: 'C:\\Videos\\My Song' }, createProgressWindow: () => progress,
    stitchRenderedScenes: async (...args) => { log.push(['stitch', args[1]]); return stitch(...args); } };
  return { context, log, button, wired };
}

const scenes = (...paths) => paths.map((video_path, index) => ({ id: `s${index}`, video_path }));

test('Export is allowed only when every scene has its final video', () => {
  const { context } = harness();
  const ready = (state) => context.exportReadiness(state);
  assert.equal(ready({ segments: [] }).ready, false);
  assert.match(ready({ segments: [] }).reason, /Add scenes/);
  const all = ready({ segments: scenes('a.mp4', 'b.mp4', 'c.mp4') });
  assert.deepEqual({ ...all }, { ready: true, total: 3, rendered: 3, reason: '' });
  const missing = ready({ segments: scenes('a.mp4', '', 'c.mp4', '') });
  assert.equal(missing.ready, false);
  assert.equal(missing.rendered, 2);
  assert.match(missing.reason, /\(2 of 4 done\)\. Still without a video: Scene 2, Scene 4\./);
  const many = ready({ segments: scenes(...Array(7).fill('')) });
  assert.match(many.reason, /Scene 1, Scene 2, Scene 3, Scene 4 and 3 more/);
  // A scene that is rendering counts as not done even when it still has the old clip.
  const rendering = ready({ segments: [{ video_path: 'a.mp4' }, { video_path: 'b.mp4', video_status: 'running' }] });
  assert.equal(rendering.ready, false);
  assert.match(rendering.reason, /Scene 2/);
});

test('a MiniMax scene with only a draft is not ready, another engine does not care', () => {
  const { context } = harness();
  const segments = [{ video_path: 'a.mp4' }, { video_path: 'b.mp4', minimax_h3_draft: { id: 'd1' } }];
  const draft = context.exportReadiness({ projectVideoEngine: 'minimax_h3', segments });
  assert.equal(draft.ready, false);
  assert.match(draft.reason, /Finalize the drafts to export: Scene 2/);
  assert.equal(context.exportReadiness({ projectVideoEngine: 'ltx', segments }).ready, true);
});

test('the save dialog starts with the project folder name', () => {
  const { context } = harness();
  assert.equal(context.exportFileName('C:\\Videos\\My Song'), 'My Song.mp4');
  assert.equal(context.exportFileName('/home/me/videos/night drive/'), 'night drive.mp4');
  assert.equal(context.exportFileName('D:\\clips\\bad:name?'), 'bad_name_.mp4');
  assert.equal(context.exportFileName(''), 'video.mp4');
});

test('the button shows why it is off and turns on when every scene is rendered', () => {
  const { context, button } = harness();
  context.syncExportButton(null, { segments: [] });
  context.syncExportButton(button, { segments: scenes('a.mp4', '') });
  assert.equal(button.disabled, true);
  assert.match(button.title, /Still without a video: Scene 2/);
  context.syncExportButton(button, { segments: scenes('a.mp4', 'b.mp4') });
  assert.equal(button.disabled, false);
  assert.equal(button.textContent, 'Export');
  assert.match(button.title, /all 2 scene videos/);
  assert.equal(button.style.cursor, 'pointer');
});

test('clicking Export asks where to save, stitches with the export prefix, copies there and removes the stitch', async () => {
  const { context, log, button, wired } = harness({
    postJson: async (url) => (url.endsWith('pick_path') ? { path: 'D:\\Out\\song.mp4' } : { destination_path: 'D:\\Out\\song.mp4' }),
    stitch: async () => ({ final_video_path: 'C:\\Videos\\My Song\\EXPORT_VIDEO.mp4' }),
  });
  wired.state = { segments: scenes('a.mp4', 'b.mp4') };
  context.wireExportVideo(wired);
  await button.onclick();
  const posts = log.filter((entry) => entry[0] === 'post');
  assert.equal(posts[0][1], '/vrgdg/music_builder/pick_path');
  assert.deepEqual({ ...posts[0][2] }, { kind: 'save_video', file_name: 'My Song.mp4' });
  assert.deepEqual({ ...log.find((entry) => entry[0] === 'stitch')[1] }, { outputPrefix: 'EXPORT_VIDEO' });
  assert.equal(posts[1][1], '/vrgdg/music_builder/export_final_video');
  assert.deepEqual({ ...posts[1][2] }, {
    source_path: 'C:\\Videos\\My Song\\EXPORT_VIDEO.mp4', destination_path: 'D:\\Out\\song.mp4',
    project_folder: 'C:\\Videos\\My Song', remove_source: true,
  });
  // The place is asked for before the stitch starts.
  assert.ok(log.findIndex((entry) => entry[0] === 'post') < log.findIndex((entry) => entry[0] === 'stitch'));
  assert.deepEqual(log.find((entry) => entry[0] === 'ready'), ['ready', 'D:\\Out\\song.mp4']);
  assert.equal(button.disabled, false, 'the button is back after the export');
});

test('closing the save dialog stitches nothing', async () => {
  const { context, log, button, wired } = harness({ postJson: async () => ({ path: '' }), stitch: async () => assert.fail('stitched') });
  wired.state = { segments: scenes('a.mp4') };
  context.wireExportVideo(wired);
  await button.onclick();
  assert.equal(log.filter((entry) => entry[0] === 'post').length, 1);
  assert.equal(log.some((entry) => entry[0] === 'stitch' || entry[0] === 'ready'), false);
  assert.equal(button.disabled, false);
});

test('an unfinished project says what is missing and does not open a dialog', async () => {
  const { context, log, button, wired } = harness({ postJson: async () => assert.fail('posted'), stitch: async () => assert.fail('stitched') });
  wired.state = { segments: scenes('a.mp4', '') };
  context.wireExportVideo(wired);
  await button.onclick();
  assert.equal(log.length, 1);
  assert.equal(log[0][0], 'toast');
  assert.equal(log[0][2], true);
  assert.match(log[0][1], /Scene 2/);
});

test('a failed stitch or copy is reported, and a kept stitch is named', async () => {
  const failedStitch = harness({ postJson: async () => ({ path: 'D:\\Out\\song.mp4' }), stitch: async () => { throw new Error('FFmpeg could not join the scene clips.'); } });
  failedStitch.wired.state = { segments: scenes('a.mp4') };
  failedStitch.context.wireExportVideo(failedStitch.wired);
  await failedStitch.button.onclick();
  const stitchToast = failedStitch.log.find((entry) => entry[0] === 'toast');
  assert.equal(stitchToast[2], true);
  assert.equal(stitchToast[1], 'FFmpeg could not join the scene clips.');
  assert.equal(failedStitch.button.disabled, false);

  const failedCopy = harness({
    postJson: async (url) => { if (url.endsWith('pick_path')) return { path: 'D:\\Out\\song.mp4' }; throw new Error('The folder to save in does not exist: D:\\Out'); },
    stitch: async () => ({ final_video_path: 'C:\\Videos\\My Song\\EXPORT_VIDEO.mp4' }),
  });
  failedCopy.wired.state = { segments: scenes('a.mp4') };
  failedCopy.context.wireExportVideo(failedCopy.wired);
  await failedCopy.button.onclick();
  const copyToast = failedCopy.log.find((entry) => entry[0] === 'toast');
  assert.match(copyToast[1], /does not exist: D:\\Out/);
  assert.match(copyToast[1], /still in the project folder:\nC:\\Videos\\My Song\\EXPORT_VIDEO\.mp4/);
  assert.equal(failedCopy.log.some((entry) => entry[0] === 'ready'), false);
});

test('a second click during an export does nothing', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const { context, log, button, wired } = harness({
    postJson: async (url) => (url.endsWith('pick_path') ? gate.then(() => ({ path: 'D:\\Out\\song.mp4' })) : { destination_path: 'D:\\Out\\song.mp4' }),
    stitch: async () => ({ final_video_path: 'C:\\Videos\\My Song\\EXPORT_VIDEO.mp4' }),
  });
  wired.state = { segments: scenes('a.mp4') };
  context.wireExportVideo(wired);
  const first = button.onclick();
  await button.onclick();
  assert.equal(button.disabled, true);
  assert.equal(button.textContent, 'Exporting...');
  release();
  await first;
  assert.equal(log.filter((entry) => entry[0] === 'post' && entry[1].endsWith('pick_path')).length, 1);
  assert.equal(button.textContent, 'Export');
});
