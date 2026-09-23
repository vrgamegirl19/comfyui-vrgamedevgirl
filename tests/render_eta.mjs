import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';
const source = readFileSync(new URL('../web/VRGDG_RenderETA.js', import.meta.url), 'utf8');
const { estimateRenderETA, formatRenderETA, renderETAProfile } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const now = Date.parse('2026-09-22T12:00:00Z');
const planned = (id, duration = 5, profile = 'settings-a') => ({ scene_id: id, eta_duration: duration, eta_profile: profile, video_mode: 'text_to_video' });
const sample = (id, total = 600000) => ({ ...planned(id), status: 'complete', total_ms: total });
const history = [{ id: 'old', video_engine: 'minimax_h3', status: 'complete', scenes: [sample('old-scene')], stitch_ms: 60000, eta_video_duration: 10 }];
function job() {
  return { id: 'live', video_engine: 'minimax_h3', status: 'running', skip_final_stitch: true,
    eta_plan: [planned('a'), planned('b', 10)], eta_video_duration: 15,
    scenes: [{ ...planned('a'), status: 'running', started_at: new Date(now - 240000).toISOString() }] };
}

test('counts down current scene and scales future work by duration', () => {
  const result = estimateRenderETA(job(), history, now);
  assert.equal(result.sceneMs, 360000);
  assert.equal(result.totalMs, 1560000);
  assert.equal(estimateRenderETA(job(), history, now + 1000).sceneMs, 359000);
});
test('includes estimated final stitching', () => {
  const log = job(); log.skip_final_stitch = false;
  assert.equal(estimateRenderETA(log, history, now).totalMs, 1650000);
});
test('stitching counts down without counting scenes again', () => {
  const log = job(); log.skip_final_stitch = false;
  log.scenes = [sample('a'), sample('b')];
  log.stitch_started_at = new Date(now - 30000).toISOString();
  const eta = estimateRenderETA(log, history, now);
  assert.equal(eta.totalMs, 60000);
  assert.equal(eta.stitching, true);
});
test('no history means estimating, not a made-up countdown', () => {
  assert.equal(estimateRenderETA(job(), [], now).sceneMs, null);
  assert.equal(estimateRenderETA(job(), [], now).totalMs, null);
});
test('learns from a completed scene in this batch', () => {
  const log = job();
  log.scenes = [sample('a'), { ...planned('b', 10), status: 'running', started_at: new Date(now).toISOString() }];
  assert.equal(estimateRenderETA(log, [], now).totalMs, 1200000);
});
test('does not mix new logs with different settings or engines', () => {
  const other = structuredClone(history);
  other[0].scenes[0].eta_profile = 'different-resolution';
  assert.equal(estimateRenderETA(job(), other, now).totalMs, null);
  other[0].scenes[0].eta_profile = 'settings-a'; other[0].video_engine = 'ltx';
  assert.equal(estimateRenderETA(job(), other, now).totalMs, null);
});
test('supports matching-mode legacy render logs', () => {
  const old = structuredClone(history);
  delete old[0].scenes[0].eta_profile; delete old[0].scenes[0].eta_duration;
  assert.equal(estimateRenderETA(job(), old, now).totalMs, 960000);
});
test('failed and completed scenes are removed from remaining work', () => {
  const log = job(); log.scenes = [{ ...planned('a'), status: 'failed' }];
  assert.equal(estimateRenderETA(log, history, now).totalMs, 1200000);
});
test('overrun does not claim the running scene is finished', () => {
  const eta = estimateRenderETA(job(), history, now + 600000);
  assert.equal(eta.sceneMs, -1);
  assert.equal(eta.totalMs, null);
  assert.equal(formatRenderETA(eta.sceneMs), 'Re-estimating…');
});
test('unknown stitching is identified separately', () => {
  const log = job(); log.skip_final_stitch = false;
  const old = structuredClone(history); delete old[0].stitch_ms;
  const eta = estimateRenderETA(log, old, now);
  assert.equal(eta.totalMs, 1560000);
  assert.equal(eta.stitchUnknown, true);
  log.stitch_started_at = new Date(now).toISOString();
  assert.equal(estimateRenderETA(log, old, now).totalMs, null);
});
test('finished, canceled, and failed logs do not resume countdowns', () => {
  for (const status of ['complete', 'canceled', 'failed']) {
    const log = job(); log.status = status;
    assert.equal(estimateRenderETA(log, history, now).totalMs, null);
  }
});
test('formats six minutes and an hour ten minutes', () => {
  assert.equal(formatRenderETA(360000), '06:00');
  assert.equal(formatRenderETA(4200000), '01:10:00');
  assert.equal(formatRenderETA(null), 'Estimating…');
});
test('profile ignores seeds and prompts but distinguishes resolution and passes', () => {
  const profile = settings => renderETAProfile('minimax_h3', 'text_to_video', settings);
  assert.equal(profile({ steps: 20, seed: 1, prompt: 'a' }), profile({ prompt: 'b', seed: 2, steps: 20 }));
  assert.notEqual(profile({ steps: 20, megapixels: 1 }), profile({ steps: 20, megapixels: 2 }));
  assert.notEqual(profile({ two_pass_pass2_steps: 5 }), profile({ two_pass_pass2_steps: 10 }));
});

test('uses legacy stitching history when durations were not recorded', () => {
  const log = job(); log.skip_final_stitch = false;
  const old = structuredClone(history); delete old[0].eta_video_duration;
  assert.equal(estimateRenderETA(log, old, now).totalMs, 1620000);
});

test('batch setup does not display zero before targets are known', () => {
  const log = job(); delete log.eta_plan;
  assert.equal(estimateRenderETA(log, history, now).totalMs, null);
});

const ui = readFileSync(new URL('../web/VRGDG_MusicVideoBuilderUI.js', import.meta.url), 'utf8');
function displayFixture() {
  let tick;
  let cleared = 0;
  const context = vm.createContext({
    liveETALog: null, builderETATimer: 0, overlay: { isConnected: true },
    builderETA: { style: {} }, builderSceneETA: {}, builderFullETA: {},
    state: { renderLogs: history, batchCancelled: false },
    formatRenderETA, estimateRenderETA: (log, logs) => estimateRenderETA(log, logs, now),
    positionBuilderETA() {},
    setInterval(callback) { tick = callback; return 1; },
    clearInterval() { cleared++; },
  });
  const start = ui.indexOf('  function refreshBuilderETA()');
  const end = ui.indexOf('  function startSingleSceneETA(', start);
  vm.runInContext(ui.slice(start, end), context);
  return { context, run: code => vm.runInContext(code, context), tick: () => tick(), cleared: () => cleared };
}
test('header updates once per tick and labels a selected batch accurately', () => {
  const f = displayFixture(); f.context.log = job(); f.context.log.scene_scope = 'selected';
  f.run('startBuilderETA(log)');
  assert.equal(f.context.builderSceneETA.textContent, 'Current Scene: 06:00');
  assert.equal(f.context.builderFullETA.textContent, 'Selected Scenes: 26:00');
  f.context.state.batchCancelled = true; f.tick();
  assert.equal(f.context.builderSceneETA.textContent, 'Current Scene: Stopping…');
});
test('completion stops timer and project reset clears the header', () => {
  const f = displayFixture(); f.context.log = job();
  f.run('startBuilderETA(log)');
  const previous = f.cleared();
  f.context.log.status = 'complete'; f.tick();
  assert.equal(f.context.builderFullETA.textContent, 'Full Video: Done');
  assert.ok(f.cleared() > previous);
  f.run('resetBuilderETA()');
  assert.equal(f.context.liveETALog, null);
  assert.equal(f.context.builderETA.style.display, 'none');
});
