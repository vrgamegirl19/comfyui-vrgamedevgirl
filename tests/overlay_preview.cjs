const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The rules of the live overlay preview (web/music_video_builder/overlay_preview.mjs): where the layers sit, how much shows when, the vignette falloff
// and where a looping clip is in its file. The layers themselves are checked in the app.
const context = vm.createContext({ api: { apiURL: (url) => `http://app${url}` } });
vm.runInContext(readBuilderModule('overlay_layers.mjs'), context);
vm.runInContext(readBuilderModule('overlay_preview.mjs'), context);
const ov = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));
const near = (a, b, digits = 3) => assert.ok(Math.abs(a - b) < 10 ** -digits, `${a} is not near ${b}`);

test('the layers sit on the picture, not on the black bars around it', () => {
  const rect = (...args) => plain(ov('overlayPreviewFrameRect')(...args));
  const wide = rect(1000, 500, 1920, 1080);
  near(wide.width, 888.889);
  near(wide.height, 500);
  near(wide.left, 55.556);
  near(wide.top, 0);
  const tall = rect(500, 1000, 1920, 1080);
  near(tall.width, 500);
  near(tall.height, 281.25);
  near(tall.top, 359.375);
  assert.deepEqual(rect(400, 300, 0, 0), { left: 0, top: 0, width: 400, height: 300 }, 'a picture of unknown size fills the box');
  assert.deepEqual(rect(0, 0, 1920, 1080), { left: 0, top: 0, width: 0, height: 0 });
});

test('a clip shows fully in the middle and ramps over its fades', () => {
  const clip = plain(ov('normalizeOverlayLayers')([{ kind: 'grain', start: 10, end: 20, fade_in: 2, fade_out: 4 }]))[0];
  const amount = (t) => ov('overlayEnvelope')(clip, t);
  assert.equal(amount(9.99), 0);
  assert.equal(amount(20.01), 0);
  assert.equal(amount(10), 0);
  near(amount(11), 0.5);
  assert.equal(amount(13), 1);
  assert.equal(amount(16), 1);
  near(amount(18), 0.5);
  assert.equal(amount(20), 0);
  const plainClip = { start: 1, end: 3, fade_in: 0, fade_out: 0 };
  assert.equal(ov('overlayEnvelope')(plainClip, 1), 1);
  assert.equal(ov('overlayEnvelope')(plainClip, 3), 1);
  assert.equal(ov('overlayEnvelope')(plainClip, 3.1), 0);
});

test('the vignette gradient darkens by the cosine to the fourth, like ffmpeg does', () => {
  const stops = (strength) => String(ov('overlayVignetteGradient')(strength)).match(/rgba\(0,0,0,([0-9.]+)\) (\d+)%/g).map((s) => Number(/,([0-9.]+)\)/.exec(s)[1]));
  const full = stops(1);
  assert.equal(full.length, 11);
  assert.equal(full[0], 0, 'the middle is not darkened');
  near(full[10], 1 - Math.cos(1.2) ** 4, 3);
  const mid = stops(0.7);
  near(mid[10], 1 - Math.cos(0.84) ** 4, 3);
  near(mid[10], 0.802, 2);
  for (let i = 1; i < mid.length; i += 1) assert.ok(mid[i] >= mid[i - 1], 'darker toward the corner');
  assert.deepEqual(stops(0), Array(11).fill(0), 'no strength, no vignette');
  assert.match(String(ov('overlayVignetteGradient')(0.5)), /^radial-gradient\(circle farthest-corner at 50% 50%, /);
  assert.deepEqual(stops(5), stops(1), 'a strength past the end is the end');
});

test('a media clip is at its source start first, loops from the start of the file, or stays on its last frame', () => {
  const time = (clip, t, duration) => ov('overlayMediaTime')(clip, t, duration);
  const clip = { start: 10, source_start: 2, speed: 1, loop: true };
  near(time(clip, 10, 8), 2);
  near(time(clip, 13, 8), 5);
  near(time(clip, 16.5, 8), 0.5, 3);
  near(time(clip, 26, 8), 2);
  near(time(clip, 28, 8), 4);
  const once = { ...clip, loop: false };
  near(time(once, 14, 8), 6);
  near(time(once, 30, 8), 7.95);
  near(time({ ...clip, speed: 2 }, 12, 8), 6);
  near(time({ ...clip, source_start: 0 }, 5, 8), 0, 3);
  near(time(clip, 12, 0), 4, 3);
  assert.equal(ov('overlayPreviewUrl')('overlay/a b.mov', 'C:/p/My Song'), 'http://app/vrgdg/music_builder/overlay_layers/preview?path=overlay%2Fa%20b.mov&project_folder=C%3A%2Fp%2FMy%20Song');
});

test('every blend mode a clip can ask for has a CSS blend mode', () => {
  const css = plain(ov('OVERLAY_CSS_BLEND'));
  for (const mode of plain(ov('OVERLAY_BLEND_MODES'))) assert.ok(css[mode], mode);
  assert.equal(css.add, 'plus-lighter');
  assert.equal(css.softlight, 'soft-light');
});

test('text is sized from a 1080 pixel tall picture, centered on x and y, and slides up while it fades in', () => {
  const clip = plain(ov('normalizeOverlayLayers')([{ kind: 'text', text: 'Hi', start: 4, end: 9, size: 100, outline_width: 4, shadow: 6, x: 0.25, y: 0.8, animation: 'slide_up', fade_in: 1 }]))[0];
  const metrics = (t, height) => plain(ov('overlayTextMetrics')(clip, t, height));
  const full = metrics(6, 1080);
  assert.deepEqual([full.fontSize, full.outline, full.shadow, full.left, full.top, full.offset], [100, 4, 6, 25, 80, 0]);
  const small = metrics(6, 540);
  assert.deepEqual([small.fontSize, small.outline, small.shadow], [50, 2, 3], 'half the picture, half the text');
  near(metrics(4, 1080).offset, 43.2, 2);
  near(metrics(4.4, 1080).offset, 43.2 * 0.6, 2);
  assert.equal(metrics(5, 1080).offset, 0);
  const short = plain(ov('normalizeOverlayLayers')([{ kind: 'text', start: 0, end: 5, animation: 'slide_up', fade_in: 0 }]))[0];
  near(plain(ov('overlayTextMetrics')(short, 0.2, 1000)).offset, 0.04 * 1000 * 0.5, 2);
  const still = plain(ov('normalizeOverlayLayers')([{ kind: 'text', start: 0, end: 5 }]))[0];
  assert.equal(plain(ov('overlayTextMetrics')(still, 0, 1080)).offset, 0, 'no animation, no offset');
  assert.equal(plain(ov('overlayTextMetrics')(still, 1, 0)).fontSize, 0);
});

test('a font is added to the document once, under one family name', () => {
  const added = [];
  class Face { constructor(family, source) { this.family = family; this.source = source; added.push(this); } load() { return Promise.resolve(this); } }
  const doc = { defaultView: { FontFace: Face }, fonts: { add() {} } };
  const family = (name) => ov('overlayFontFamily')(doc, name);
  assert.equal(family('Bebas Neue'), '"vrgdg-overlay-bebas-neue", sans-serif');
  assert.equal(family(' bebas neue '), '"vrgdg-overlay-bebas-neue", sans-serif');
  assert.equal(added.length, 1);
  assert.equal(added[0].source, 'url(http://app/vrgdg/music_builder/overlay_layers/font?name=Bebas%20Neue)');
  assert.equal(family(''), 'sans-serif');
  assert.equal(added.length, 1);
  assert.equal(ov('overlayFontFamily')({ defaultView: null, fonts: null }, 'Anton'), '"vrgdg-overlay-anton", sans-serif', 'a document without font support still gets a family name');
});
