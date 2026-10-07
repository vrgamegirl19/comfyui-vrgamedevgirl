import assert from 'node:assert/strict';
import { test } from 'node:test';
import { canvasFor, clampBox, DEFAULT_QUALITY, estimateTokens, expandBox, QUALITY_PRESETS, qualityScale, snap32, trimBoxFromPixels } from '../web/music_video_builder/refmod_trim.mjs';

function image(width, height, paint) {
  const data = new Uint8ClampedArray(width * height * 4).fill(255);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (paint(x, y)) {
        const i = (y * width + x) * 4;
        data[i] = 20; data[i + 1] = 20; data[i + 2] = 20;
      }
    }
  }
  return data;
}

test('snap32 rounds half up to multiples of 32 with a floor', () => {
  assert.equal(snap32(611), 608);
  assert.equal(snap32(661), 672);
  assert.equal(snap32(5), 32);
  assert.equal(snap32(528), 544);
});

test('finds the box around a subject on white', () => {
  const data = image(100, 80, (x, y) => x >= 30 && x < 70 && y >= 10 && y < 60);
  assert.deepEqual(trimBoxFromPixels(data, 100, 80), { x0: 30, y0: 10, x1: 70, y1: 60 });
});

test('ignores isolated noise in the background', () => {
  const data = image(100, 80, (x, y) => (x >= 30 && x < 70 && y >= 10 && y < 60) || (x === 2 && y === 70));
  assert.deepEqual(trimBoxFromPixels(data, 100, 80), { x0: 30, y0: 10, x1: 70, y1: 60 });
});

test('transparent pixels count as background', () => {
  const data = image(40, 40, () => true);
  for (let i = 0; i < data.length; i += 4) data[i + 3] = 0;
  for (let y = 10; y < 20; y += 1) for (let x = 10; x < 25; x += 1) data[(y * 40 + x) * 4 + 3] = 255;
  assert.deepEqual(trimBoxFromPixels(data, 40, 40), { x0: 10, y0: 10, x1: 25, y1: 20 });
});

test('an all-background image gives no box', () => {
  assert.equal(trimBoxFromPixels(image(20, 20, () => false), 20, 20), null);
});

test('expandBox adds a margin and stays inside the image', () => {
  assert.deepEqual(expandBox({ x0: 30, y0: 10, x1: 70, y1: 60 }, 16, 100, 80), { x0: 14, y0: 0, x1: 86, y1: 76 });
  assert.deepEqual(expandBox(null, 16, 100, 80), { x0: 0, y0: 0, x1: 100, y1: 80 });
});

test('clampBox keeps a box inside the image and above the minimum size', () => {
  assert.deepEqual(clampBox({ x0: -20, y0: -5, x1: 500, y1: 300 }, 400, 200), { x0: 0, y0: 0, x1: 400, y1: 200 });
  assert.deepEqual(clampBox({ x0: 100, y0: 100, x1: 105, y1: 101 }, 400, 200), { x0: 100, y0: 100, x1: 132, y1: 132 });
  assert.deepEqual(clampBox({ x0: 390, y0: 190, x1: 395, y1: 195 }, 400, 200), { x0: 368, y0: 168, x1: 400, y1: 200 });
});

test('canvas holds the widest and tallest image and snaps to 32 at full quality', () => {
  const canvas = canvasFor([{ width: 500, height: 560 }, { width: 480, height: 560 }, { width: 300, height: 1150 }], 1);
  assert.deepEqual([canvas.width, canvas.height], [448, 1024]);
});

test('quality scales the canvas down and never up', () => {
  const sizes = [{ width: 500, height: 560 }, { width: 300, height: 1000 }];
  const half = canvasFor(sizes, 0.5);
  assert.deepEqual([half.width, half.height], [320, 512]);
  const small = [{ width: 200, height: 300 }];
  assert.deepEqual([canvasFor(small, 1).width, canvasFor(small, 1).height], [320, 320]);
});

test('presets run from most to least detail', () => {
  const scales = QUALITY_PRESETS.map((preset) => preset.scale);
  assert.deepEqual([...scales].sort((x, y) => y - x), scales);
  assert.equal(qualityScale('balanced'), 0.6);
  assert.equal(qualityScale('unknown'), 0.6);
  assert.equal(DEFAULT_QUALITY, 'balanced');
});

test('estimates whole and trimmed tokens and every preset (same numbers as the Python twin)', () => {
  const sizes = [{ width: 611, height: 661 }, { width: 581, height: 661 }, { width: 330, height: 1179 }];
  const crops = [{ width: 500, height: 560 }, { width: 480, height: 560 }, { width: 300, height: 1150 }];
  const result = estimateTokens({ sizes, cropSizes: crops, quality: 'maximum' });
  assert.deepEqual(result.whole.canvas, [544, 1024]);
  assert.deepEqual(result.trimmed.canvas, [448, 1024]);
  assert.equal(result.trimmed.tokens, 3 * 14 * 32);
  assert.equal(result.cap, 5120);
  assert.ok(result.trimmed.fit > 0 && result.trimmed.fit <= 1);
  assert.equal(result.byQuality.length, QUALITY_PRESETS.length);
  const tokens = result.byQuality.map((entry) => entry.tokens);
  assert.deepEqual([...tokens].sort((x, y) => y - x), tokens);
  assert.equal(result.byQuality.find((entry) => entry.key === 'maximum').tokens, result.trimmed.tokens);
});

test('a lower quality always costs fewer tokens', () => {
  const sizes = [{ width: 600, height: 600 }, { width: 400, height: 1100 }];
  const maximum = estimateTokens({ sizes, quality: 'maximum' }).trimmed.tokens;
  const compact = estimateTokens({ sizes, quality: 'compact' }).trimmed.tokens;
  assert.ok(compact < maximum / 3);
});

test('with no crops the trimmed estimate equals the whole-image estimate', () => {
  const sizes = [{ width: 600, height: 600 }, { width: 400, height: 1100 }];
  const result = estimateTokens({ sizes, quality: 'balanced' });
  assert.equal(result.trimmed.tokens, result.whole.tokens);
});

test('no images gives no estimate', () => {
  assert.equal(estimateTokens({ sizes: [] }), null);
});
