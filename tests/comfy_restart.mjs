import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// comfy_restart.mjs imports controls.mjs, which imports ComfyUI's scripts/app.js (browser only). The loader stubs it.
register('./stub_comfy_loader.mjs', import.meta.url);

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.style = { cssText: '' };
    this.listeners = {};
    this.attributes = {};
    this.textContent = '';
    this.title = '';
    this.disabled = false;
    this.removed = false;
  }
  append(...nodes) { this.children.push(...nodes); }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  remove() { this.removed = true; }
}
globalThis.document = { createElement: (tag) => new FakeElement(tag), body: new FakeElement('body') };

const { restartComfyUi, runningJobCount, wireRestartComfyButton, confirmComfyRestart, COMFY_RESTART_URL } = await import('../web/music_video_builder/comfy_restart.mjs');

// A ComfyUI that answers /system_stats while `alive()` says so, and plays the restart route with `restart`.
function fakeApi({ restart, alive, queue }) {
  const calls = [];
  return {
    calls,
    fetchApi: async (route, options = {}) => {
      calls.push({ route, method: options.method || 'GET', body: options.body });
      if (route === COMFY_RESTART_URL) return restart(calls);
      if (route === '/system_stats') {
        if (!alive(calls)) throw new TypeError('Failed to fetch');
        return { ok: true, json: async () => ({}) };
      }
      if (route === '/queue') return queue ? { ok: true, json: async () => queue } : { ok: false, json: async () => ({}) };
      throw new Error(`unexpected ${route}`);
    },
  };
}

// A clock that moves only when the code waits, so a minute of waiting costs nothing.
function virtualClock() {
  let now = 0;
  return { now: () => now, wait: async (ms) => { now += ms; }, at: () => now };
}

// Down from `from` to `to` virtual ms. The clock tells the time of every call.
const downBetween = (clock, from, to) => () => !(clock.at() >= from && clock.at() < to);

test('it posts to the restart route, waits for ComfyUI to stop answering and then for it to answer again', async () => {
  const clock = virtualClock();
  const api = fakeApi({ restart: async () => ({ ok: true, json: async () => ({ ok: true, status: 'restarting' }) }), alive: downBetween(clock, 2000, 9000) });
  await restartComfyUi(api, clock);
  const restarts = api.calls.filter((call) => call.route === COMFY_RESTART_URL);
  assert.equal(restarts.length, 1);
  assert.equal(restarts[0].method, 'POST');
  assert.ok(clock.at() >= 9000, 'it waited until ComfyUI answered again');
  assert.ok(clock.at() < 12000, 'and stopped waiting when it did');
});

test('a connection that drops while the restart starts is not an error', async () => {
  const clock = virtualClock();
  const api = fakeApi({ restart: async () => { throw new TypeError('Failed to fetch'); }, alive: downBetween(clock, 1000, 5000) });
  await restartComfyUi(api, clock);
  assert.ok(clock.at() >= 5000);
});

test('an answer that refuses the restart gives its text and does not wait', async () => {
  const clock = virtualClock();
  const api = fakeApi({
    restart: async () => ({ ok: false, status: 501, json: async () => ({ ok: false, error: 'ComfyUI-Manager is not installed in this ComfyUI.' }) }),
    alive: () => true,
  });
  await assert.rejects(restartComfyUi(api, clock), /ComfyUI-Manager is not installed/);
  assert.equal(clock.at(), 0);
  assert.equal(api.calls.filter((call) => call.route === '/system_stats').length, 0);
});

test('an answer without a text still says what happened', async () => {
  const api = fakeApi({ restart: async () => ({ ok: false, status: 502, json: async () => { throw new Error('not json'); } }), alive: () => true });
  await assert.rejects(restartComfyUi(api, virtualClock()), /HTTP 502/);
});

test('a ComfyUI that never stops answering is reported, not waited for forever', async () => {
  const clock = virtualClock();
  const api = fakeApi({ restart: async () => ({ ok: true, json: async () => ({}) }), alive: () => true });
  await assert.rejects(restartComfyUi(api, clock), /did not restart/);
  assert.ok(clock.at() <= 21000);
});

test('a ComfyUI that does not come back is reported after three minutes', async () => {
  const clock = virtualClock();
  const api = fakeApi({ restart: async () => ({ ok: true, json: async () => ({}) }), alive: (calls) => calls.length < 3 });
  await assert.rejects(restartComfyUi(api, clock), /not answering after the restart/);
  assert.ok(clock.at() >= 180000);
});

test('the running and waiting jobs are counted from the queue, and an unreadable queue counts none', async () => {
  const busy = fakeApi({ restart: null, alive: () => true, queue: { queue_running: [[1]], queue_pending: [[2], [3]] } });
  assert.equal(await runningJobCount(busy), 3);
  assert.equal(await runningJobCount(fakeApi({ restart: null, alive: () => true })), 0);
  assert.equal(await runningJobCount({ fetchApi: async () => { throw new Error('down'); } }), 0);
});

test('the button asks first and does nothing when the answer is no', async () => {
  const button = new FakeElement('button');
  const clock = virtualClock();
  const api = fakeApi({ restart: async () => ({ ok: true, json: async () => ({}) }), alive: () => true });
  const toasts = [];
  wireRestartComfyButton(button, { api, toast: (...args) => toasts.push(args), confirm: async () => false, clock });
  await button.onclick();
  assert.equal(api.calls.some((call) => call.route === COMFY_RESTART_URL), false);
  assert.deepEqual(toasts, []);
  assert.equal(button.disabled, false);
});

test('the button is busy while ComfyUI restarts, says when it is back and is usable again', async () => {
  const button = new FakeElement('button');
  button.title = 'Restart ComfyUI';
  const clock = virtualClock();
  const seen = [];
  const api = fakeApi({
    restart: async () => ({ ok: true, json: async () => ({}) }),
    alive: () => {
      seen.push([button.disabled, button.title]);
      return !(clock.at() >= 1000 && clock.at() < 4000);
    },
    queue: { queue_running: [[1]], queue_pending: [] },
  });
  const toasts = [];
  let asked = -1;
  wireRestartComfyButton(button, { api, toast: (...args) => toasts.push(args), confirm: async (jobs) => { asked = jobs; return true; }, clock });
  await button.onclick();
  assert.equal(asked, 1, 'the question knows a job is running');
  assert.ok(seen.length > 0 && seen.every(([disabled, title]) => disabled && title === 'Restarting ComfyUI...'));
  assert.deepEqual(toasts.map(([text, isError]) => [/restarted and is ready/.test(text), Boolean(isError)]), [[false, false], [true, false]]);
  assert.equal(button.disabled, false);
  assert.equal(button.title, 'Restart ComfyUI');
});

test('a failed restart shows its error and frees the button', async () => {
  const button = new FakeElement('button');
  const api = fakeApi({ restart: async () => ({ ok: false, status: 501, json: async () => ({ error: 'No Manager.' }) }), alive: () => true });
  const toasts = [];
  wireRestartComfyButton(button, { api, toast: (...args) => toasts.push(args), confirm: async () => true, clock: virtualClock() });
  await button.onclick();
  assert.deepEqual(toasts[toasts.length - 1], ['No Manager.', true]);
  assert.equal(button.disabled, false);
});

test('a second click while it restarts does nothing', async () => {
  const button = new FakeElement('button');
  const clock = virtualClock();
  let release;
  const held = new Promise((resolve) => { release = resolve; });
  const api = fakeApi({ restart: async () => { await held; return { ok: true, json: async () => ({}) }; }, alive: downBetween(clock, 1000, 3000) });
  wireRestartComfyButton(button, { api, toast: () => {}, confirm: async () => true, clock });
  const first = button.onclick();
  while (!button.disabled) await new Promise((resolve) => setImmediate(resolve));
  await button.onclick();
  release();
  await first;
  assert.equal(api.calls.filter((call) => call.route === COMFY_RESTART_URL).length, 1);
});

test('the question names the lost jobs and answers true for Restart and false for Cancel', async () => {
  const run = async (jobs, press) => {
    document.body.children.length = 0;
    const answer = confirmComfyRestart(jobs);
    const backdrop = document.body.children[0];
    const box = backdrop.children[0];
    const [, note, actions] = box.children;
    const text = note.textContent;
    actions.children[press].onclick();
    return { text, answer: await answer, removed: backdrop.removed };
  };
  const one = await run(1, 0);
  assert.match(one.text, /1 running or queued job\./);
  assert.equal(one.answer, true);
  assert.equal(one.removed, true);
  assert.match((await run(2, 1)).text, /2 running or queued jobs\./);
  const none = await run(0, 1);
  assert.doesNotMatch(none.text, /job/);
  assert.equal(none.answer, false);
});
