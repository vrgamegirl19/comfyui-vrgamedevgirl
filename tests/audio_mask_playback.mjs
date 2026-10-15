// Smoke test for the Story Builder window: opens it on a small fake DOM, renders every tab, and saves the scene plan.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { after, test } from 'node:test';

// ComfyUI's own scripts are not available outside the browser.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (/scripts\/(app|api)\.js$/.test(specifier)) {
      return { url: 'data:text/javascript,export const app = {}; export const api = (globalThis.fakeComfyApi ||= {});', shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  // The Builder's .js files are ES modules too.
  load(url, context, nextLoad) {
    if (url.includes('/web/') && url.endsWith('.js')) return { ...nextLoad(url, { ...context, format: 'module' }), format: 'module' };
    return nextLoad(url, context);
  },
});

class FakeElement {
  constructor(tagName) {
    this.tagName = String(tagName).toUpperCase();
    this.children = [];
    this.style = { cssText: '' };
    this.dataset = {};
    this.listeners = {};
    this.value = '';
    this.parent = null;
    this.ownText = '';
  }
  append(...nodes) {
    for (const node of nodes) {
      const child = typeof node === 'string' ? Object.assign(new FakeElement('#text'), { ownText: node }) : node;
      child.parent = this;
      this.children.push(child);
    }
  }
  appendChild(node) { this.append(node); return node; }
  insertBefore(node) { this.append(node); return node; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this); }
  set textContent(value) { this.children = []; this.ownText = String(value); }
  get textContent() { return this.ownText + this.children.map((child) => child.textContent).join(''); }
  set innerHTML(value) { this.children = []; this.ownText = String(value).replace(/<[^>]+>/g, ' '); }
  get innerHTML() { return this.ownText; }
  get options() { return this.children.filter((child) => child.tagName === 'OPTION'); }
  get selectedOptions() { return this.options.filter((option) => option.value === this.value); }
  addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
  removeEventListener() {}
  setAttribute(name, value) { this[name] = value; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  focus() {}
  click() { this.onclick?.(); }
  all() { return [this, ...this.children.flatMap((child) => child.all())]; }
}

globalThis.HTMLElement = FakeElement;
globalThis.document = {
  body: new FakeElement('body'),
  head: new FakeElement('head'),
  createElement: (tag) => new FakeElement(tag),
  createTextNode: (text) => Object.assign(new FakeElement('#text'), { ownText: String(text) }),
  getElementById: () => null,
  addEventListener() {},
};
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.dispatchEvent = () => true;
globalThis.localStorage = { getItem: () => null, setItem() {} };


const { createAudioMask } = await import('../web/music_video_builder/audio_mask.mjs');
const store = await import('../web/music_video_builder/audio_mask_store.mjs');

class FakeAudio {
  constructor(src = '') { this.src = src; this.paused = true; this.volume = 1; this.slow = Boolean(FakeAudio.loading); this.currentTime = 0; this.preload = ''; }
  get readyState() { return this.slow && FakeAudio.loading ? 1 : 4; }
  play() { this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
  addEventListener() {}
}
globalThis.Audio = FakeAudio;

const NAMES = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];
const server = {};
const separated = [];
let now = 1000;
function sceneState(id) {
  const s = server[id];
  return { ok: true, exists: true, scene_id: id, duration: 5, stem_names: NAMES, separation: { duration_seconds: 5, start_seconds: s.start, source_path: String.raw`C:\Users\x\a.wav` },
    mix: s.mix, files: { masked_mix: `C:/p/${id}/masked_mix.wav` }, peaks: {} };
}
Object.assign(globalThis.fakeComfyApi ||= {}, {
  async fetchApi(url, init) {
    const body = JSON.parse(init.body);
    let data = { ok: true };
    if (url.endsWith('/audio_mask/state')) data = sceneState(body.scene_id);
    else if (url.endsWith('/audio_mask/states')) data = { ok: true, states: Object.fromEntries(body.scene_ids.map((id) => [id, sceneState(id)])) };
    else if (url.endsWith('/audio_mask/render')) {
      if (server[body.scene_id].missing) return { ok: false, status: 400, headers: { get: () => null }, json: async () => ({ ok: false, error: "Split this scene's audio into stems first." }) };
      const stems = {};
      for (const name of NAMES) { const i = body.stems[name] || {}; stems[name] = { mask: !!i.mask, regions: i.regions || [], db: Number(i.db) || 0, mute: !!i.mute }; }
      server[body.scene_id].mix = { stems, built_at: ++now };
      data = sceneState(body.scene_id);
    } else if (url.endsWith('/audio_mask/separate')) {
      separated.push(body.scene_id);
      server[body.scene_id].mix = null; // a new split removes the old masked mix
      data = sceneState(body.scene_id);
    } else if (url.endsWith('/audio_mask/list')) data = { ok: true, scene_ids: Object.keys(server) };
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => data };
  },
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const segments = [0, 1, 2].map((i) => ({ id: `s${i + 1}`, start: i * 5, end: i * 5 + 5, audio_mask: { version: 2, enabled: false, stems: {} } }));
const initialMix = { stems: Object.fromEntries(NAMES.map((n) => [n, { mask: false, regions: [], db: 0, mute: false }])), built_at: now };
for (const s of segments) server[s.id] = { start: s.start, mix: structuredClone(initialMix) };
const state = { segments, overlaySegments: [], audioMaskAuto: { enabled: false }, monitorStems: true, showTimelineStems: true };
const mainAudio = new FakeAudio('main'); const mainSceneAudio = new FakeAudio('scene');
let playing = false; let globalTime = 1; let active = segments[0];
const button = document.createElement('button'); const monitorButton = document.createElement('button'); const visibilityButton = document.createElement('button');
const overlay = document.createElement('div');
createAudioMask({ button, monitorButton, visibilityButton, overlay, state, activeSegment: () => active,
  currentProjectAudioPath: () => 'c:/Users/x/a.wav', getProjectFolder: () => 'C:/proj', autoSaveSessionQuiet: async () => true,
  currentGlobalTime: () => globalTime, isTimelinePlaying: () => playing, audio: mainAudio, sceneAudio: mainSceneAudio });


await sleep(1500);
playing = true;
await sleep(4500);
const muted = (segment) => Object.entries(segment.audio_mask.stems).filter(([, value]) => value.mute).map(([name]) => name);
test('stems show for a scene whose audio path is spelled differently by the page and the server', () => {
  assert.deepEqual(segments.map((segment) => Boolean(store.getStemLanes(segment.id))), [true, true, true]);
});
test('muting one stem in one scene mutes only that stem there and nothing in other scenes', async () => {
  store.requestStemEdit('s1', 'drums', { mute: true });
  await sleep(1800);
  active = segments[1]; globalTime = 6;
  store.requestStemEdit('s2', 'bass', { mute: true });
  await sleep(1800);
  assert.deepEqual(muted(segments[0]), ['drums']);
  assert.deepEqual(muted(segments[1]), ['bass']);
  assert.deepEqual(muted(segments[2]), []);
  assert.deepEqual(server.s1.mix.stems.drums.mute, true);
  assert.deepEqual(store.getStemLanes('s3').stems.filter((stem) => stem.mute), []);
});
test('the main audio is silenced only while the scene stem mix is playing', async () => {
  globalTime = 11; active = segments[2]; await sleep(400);
  assert.equal(mainAudio.volume, 0, 'the stem mix of scene 3 is playing');
  FakeAudio.loading = true; // the rebuilt mix of scene 3 has not loaded yet
  store.requestStemEdit('s3', 'piano', { mute: true });
  await sleep(2200);
  assert.equal(mainAudio.volume, 1, 'the main audio plays while the new mix loads');
  assert.equal(mainSceneAudio.volume, 1);
  FakeAudio.loading = false; await sleep(400);
  assert.equal(mainAudio.volume, 0);
});
test('Rebuild stems for all scenes splits and rebuilds every scene and keeps the mute settings', async () => {
  globalThis.window.confirm = () => true;
  const button = overlay.all().find((node) => node.textContent === 'Rebuild stems for all scenes');
  assert.ok(button, 'the button is in the Audio Mask window');
  const mutedBefore = segments.map(muted);
  playing = false;
  await button.onclick();
  assert.deepEqual(separated, ['s1', 's2', 's3']);
  for (const segment of segments) assert.ok(server[segment.id].mix, 'the mix of ' + segment.id + ' is built again');
  assert.deepEqual(segments.map(muted), mutedBefore);
  assert.deepEqual(Object.entries(server.s1.mix.stems).filter(([, value]) => value.mute).map(([name]) => name), ['drums']);
  assert.equal(button.textContent, 'Rebuild stems for all scenes');
});
test('stems that vanished from disk are forgotten so the scene is split again, not failed on every edit', async () => {
  server.s2.missing = true; // the files are gone on the server, the page still lists them
  playing = false;
  store.requestStemEdit('s2', 'vocals', { mute: true });
  await sleep(2200);
  assert.equal(store.getStemLanes('s2'), null, 's2 no longer shows stems that are not there');
  server.s2.missing = false;
});
after(() => process.exit(process.exitCode || 0));
