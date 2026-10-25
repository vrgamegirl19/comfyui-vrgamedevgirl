// Stem tracks in the timeline: a drag on a stem shows the green selection from the first click and grows it while the pointer moves, not only after it is
// released. Runs the real drag handlers on a fake canvas that records every frame the track paints.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';

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

// What the timeline's modules ask of a browser when they are loaded.
globalThis.window = globalThis;
globalThis.Path2D = class { rect() {} };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.document = { createElement: () => ({ style: {}, dataset: {}, append() {}, addEventListener() {} }), addEventListener() {}, body: { append() {} } };

const { attachStemEditing } = await import('../web/music_video_builder/timeline_view.mjs');
const { onStemEditRequest } = await import('../web/music_video_builder/audio_mask_store.mjs');

const WIDTH = 600; // 10 seconds on 600 pixels: 60 pixels a second
const REGION = 'rgba(34,197,94,.16)';
const EDGE = '#22c55e';

// A canvas that remembers what was painted since it was last cleared (a frame), and feeds the pointer events the track listens for.
function fakeCanvas() {
  const frames = [];
  const ctx = {
    fillStyle: '', globalAlpha: 1, font: '',
    clearRect() { frames.push([]); },
    fillRect(x, _y, width) { frames.at(-1)?.push({ color: ctx.fillStyle, x, width }); },
    fill() {}, fillText() {}, measureText: () => ({ width: 40 }),
  };
  const listeners = {};
  const canvas = {
    width: WIDTH, height: 60, style: {},
    addEventListener(name, handler) { (listeners[name] ||= []).push(handler); },
    getBoundingClientRect: () => ({ left: 0, width: WIDTH }),
    setPointerCapture() {},
    getContext: () => ctx,
  };
  const fire = (name, x) => (listeners[name] || []).forEach((handler) => handler({ button: 0, clientX: x, pointerId: 1, stopPropagation() {} }));
  return { canvas, frames, fire };
}

const peaks = Array.from({ length: 120 }, (_, i) => 0.2 + 0.6 * Math.abs(Math.sin(i / 7)));
const lanes = { duration: 10, enabled: true, mix: { peaks } };
const stemOf = (extra = {}) => ({ mask: false, mute: false, db: 0, peaks, regions: [], ...extra });

/** The kept regions the last frame shows in green, as `[x, width]`. */
const shownRegions = (frames) => (frames.at(-1) || []).filter((item) => item.color === REGION).map((item) => [Math.round(item.x), Math.round(item.width)]);

function track(stem) {
  const edits = [];
  onStemEditRequest((scene, name, patch) => edits.push({ scene, name, patch }));
  const view = fakeCanvas();
  attachStemEditing(view.canvas, 'seg_1', 'vocals', lanes, stem);
  return { ...view, edits };
}

test('the green selection starts at the click, before the pointer moves', () => {
  const { frames, fire } = track(stemOf());
  fire('pointerdown', 150);
  assert.ok(frames.length, 'the track was drawn when the button went down');
  const edges = frames.at(-1).filter((item) => item.color === EDGE);
  assert.ok(edges.some((item) => Math.abs(item.x - 150) <= 2), `a green marker stands where the pointer went down: ${JSON.stringify(frames.at(-1))}`);
});

test('the selection grows with the pointer while the button is down, and nothing is saved before it is released', () => {
  const { frames, fire, edits } = track(stemOf());
  fire('pointerdown', 150);
  fire('pointermove', 300);
  assert.deepEqual(shownRegions(frames), [[150, 150]]);
  fire('pointermove', 450);
  assert.deepEqual(shownRegions(frames), [[150, 300]]);
  // Dragging back past the start moves the other edge: the selection runs from the pointer to where it began.
  fire('pointermove', 60);
  assert.deepEqual(shownRegions(frames), [[60, 90]]);
  assert.deepEqual(edits, [], 'a drag in progress saves nothing');
});

test('releasing the button saves the selection as the part of the stem to keep', () => {
  const { fire, edits } = track(stemOf());
  fire('pointerdown', 150);
  fire('pointermove', 450);
  fire('pointerup', 450);
  assert.deepEqual(edits, [{ scene: 'seg_1', name: 'vocals', patch: { mask: true, regions: [{ start: 2.5, end: 7.5, fade_ms: 30 }] } }]);
});

test('a plain click selects nothing and saves nothing', () => {
  const { frames, fire, edits } = track(stemOf());
  fire('pointerdown', 150);
  fire('pointerup', 150);
  assert.deepEqual(edits, []);
  assert.deepEqual(shownRegions(frames), [], 'the marker of a click that selected nothing is gone');
});

test('a stem that is already masked still shows its regions and the new one together', () => {
  const { frames, fire } = track(stemOf({ mask: true, regions: [{ start: 1, end: 2, fade_ms: 30 }] }));
  fire('pointerdown', 300);
  fire('pointermove', 420);
  assert.deepEqual(shownRegions(frames), [[60, 60], [300, 120]]);
});
