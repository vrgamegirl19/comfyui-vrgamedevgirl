const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The overlay pass after the final stitch (web/music_video_builder/overlay_render.mjs): when it runs, what it sends, and what happens when it fails.
function harness({ available = true, reply = { ok: true, changed: true, applied: ['g'], skipped: [] }, fails = null, fetchReply = null } = {}) {
  const log = [];
  const context = vm.createContext({
    api: { fetchApi: async (url) => { log.push(['fetch', url]); if (fetchReply === 'throw') throw new Error('offline'); return fetchReply || { ok: true, json: async () => ({ ok: true, version: 1 }) }; } },
    toast: (message, isError) => log.push(['toast', message, Boolean(isError)]),
    postJson: async (url, payload, timeout) => { log.push(['post', url, JSON.parse(JSON.stringify(payload)), timeout]); if (fails) throw new Error(fails); return reply; },
  });
  vm.runInContext(readBuilderModule('overlay_layers.mjs'), context);
  vm.runInContext(readBuilderModule('overlay_render.mjs'), context);
  const state = { overlayAvailable: available, overlayLayerTrack: { enabled: true }, overlayLayers: [{ kind: 'grain', id: 'g', start: 0, end: 5 }, { kind: 'vignette', id: 'v', start: 0, end: 5, enabled: false }] };
  const progress = { set: (text, pct) => log.push(['progress', text, pct]) };
  return { context, log, state, progress, ov: (name) => vm.runInContext(name, context) };
}

test('a stitched video with overlays is sent through the overlay render and replaced in place', async () => {
  const { log, state, progress, ov } = harness();
  const stitched = { final_video_path: 'C:/p/FINAL_VIDEO.mp4', other: 1 };
  const got = await ov('applyOverlayAfterStitch')(state, stitched, { progress, timelineOffset: 12.5, projectFolder: 'C:/p' });
  assert.equal(got.final_video_path, 'C:/p/FINAL_VIDEO.mp4');
  assert.equal(got.other, 1);
  assert.equal(got.overlay_layers.changed, true);
  const [post] = log.filter((entry) => entry[0] === 'post');
  assert.equal(post[1], '/vrgdg/music_builder/overlay_layers/render');
  assert.equal(post[2].input_path, 'C:/p/FINAL_VIDEO.mp4');
  assert.equal(post[2].replace_source, true);
  assert.equal(post[2].time_offset, 12.5);
  assert.equal(post[2].project_folder, 'C:/p');
  assert.deepEqual(post[2].overlay_layers.map((clip) => clip.id), ['g'], 'a clip that is off is not sent');
  assert.deepEqual(log.find((entry) => entry[0] === 'progress').slice(1), ['Applying the overlays...', 97]);
});

test('nothing is sent when the build cannot draw overlays, they are off, or every layer is off', async () => {
  for (const change of [{ available: false }, { off: true }, { clipsOff: true }, { none: true }]) {
    const { log, state, ov } = harness({ available: change.available !== false });
    if (change.off) state.overlayLayerTrack = { enabled: false };
    if (change.clipsOff) state.overlayLayers.forEach((clip) => { clip.enabled = false; });
    if (change.none) state.overlayLayers = [];
    const stitched = { final_video_path: 'C:/p/FINAL_VIDEO.mp4' };
    assert.equal(await ov('applyOverlayAfterStitch')(state, stitched), stitched, JSON.stringify(change));
    assert.deepEqual(log, [], JSON.stringify(change));
  }
});

test('a failed overlay render keeps the plain stitched video and says why', async () => {
  const { log, state, ov } = harness({ fails: 'ffmpeg could not apply the overlays: bad graph' });
  const stitched = { final_video_path: 'C:/p/FINAL_VIDEO.mp4' };
  const got = await ov('applyOverlayAfterStitch')(state, stitched, {});
  assert.equal(got, stitched);
  const [toast] = log.filter((entry) => entry[0] === 'toast');
  assert.equal(toast[2], true);
  assert.match(toast[1], /could not be applied, so the video has none/);
  assert.match(toast[1], /bad graph/);
});

test('clips the render left out are listed', async () => {
  const { log, state, ov } = harness({ reply: { ok: true, changed: true, applied: ['g'], skipped: [{ id: 'dust', reason: 'the clip file was not found' }, { id: 'title', reason: 'text clips are not drawn yet' }] } });
  const got = await ov('applyOverlayAfterStitch')(state, { final_video_path: 'C:/p/FINAL_VIDEO.mp4' }, {});
  assert.equal(got.overlay_layers.skipped.length, 2);
  const [toast] = log.filter((entry) => entry[0] === 'toast');
  assert.match(toast[1], /dust: the clip file was not found; title: text clips are not drawn yet/);
});

test('a stitch with no video path is left alone', async () => {
  const { log, state, ov } = harness();
  const stitched = { final_video_path: '' };
  assert.equal(await ov('applyOverlayAfterStitch')(state, stitched, {}), stitched);
  assert.deepEqual(log, []);
});

test('the capability check says yes only when the build answers, and a missing route is not an error', async () => {
  for (const [fetchReply, expected] of [[null, true], [{ ok: false, status: 404, json: async () => ({}) }, false], [{ ok: true, json: async () => ({ ok: false }) }, false], [{ ok: true, json: async () => ({ ok: true }) }, false], ['throw', false]]) {
    const { state, log, ov } = harness({ fetchReply });
    state.overlayAvailable = null;
    assert.equal(await ov('loadOverlayCapabilities')(state), expected, JSON.stringify(fetchReply));
    assert.equal(state.overlayAvailable, expected);
    assert.equal(log.filter((entry) => entry[0] === 'toast').length, 0);
  }
});
