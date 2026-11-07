const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule } = require('./builder_source.cjs');

// The Thumbnail intro's saved settings and when the stitch uses them (web/music_video_builder/thumbnail_state.mjs).
function load() {
  const context = vm.createContext({});
  vm.runInContext(readBuilderModule('thumbnail_state.mjs'), context);
  return context;
}

test('an empty or broken value loads as the defaults', () => {
  const c = load();
  for (const value of [undefined, null, 'x', 5, []]) {
    const intro = c.normalizeThumbnailIntro(value);
    assert.equal(intro.enabled, false);
    assert.equal(intro.image_path, '');
    assert.equal(intro.seconds, 2);
    assert.equal(intro.engine, 'nano_banana');
    assert.deepEqual(Array.from(intro.history), []);
  }
});

test('the length is kept between half a second and ten seconds', () => {
  const c = load();
  assert.equal(c.clampThumbnailSeconds(0), 0.5);
  assert.equal(c.clampThumbnailSeconds(99), 10);
  assert.equal(c.clampThumbnailSeconds('3.5'), 3.5);
  assert.equal(c.clampThumbnailSeconds('abc'), 2);
});

test('only a real engine and a real picture survive, and the history is limited', () => {
  const c = load();
  const history = Array.from({ length: 20 }, (_, index) => ({ path: `C:\\p\\thumbnail_${index}.png`, engine: 'bogus' }));
  history.push({ path: '' }, null, 'junk');
  const intro = c.normalizeThumbnailIntro({ enabled: true, engine: 'bogus', image_path: '  C:\\p\\a.png ', history });
  assert.equal(intro.enabled, true);
  assert.equal(intro.engine, 'nano_banana');
  assert.equal(intro.image_path, 'C:\\p\\a.png');
  assert.equal(intro.history.length, 12);
  assert.equal(intro.history.at(-1).path, 'C:\\p\\thumbnail_19.png');
  assert.equal(intro.history.at(-1).engine, 'nano_banana');
});

test('a new picture goes last in the history without a duplicate', () => {
  const c = load();
  const first = { path: 'a.png', prompt: '1', engine: 'flux_klein', scene_id: 's1' };
  const second = { path: 'b.png', prompt: '2', engine: 'nano_banana', scene_id: 's1' };
  let history = c.thumbnailHistoryWith([], first);
  history = c.thumbnailHistoryWith(history, second);
  history = c.thumbnailHistoryWith(history, { ...first, prompt: 'again' });
  assert.deepEqual(Array.from(history, (item) => item.path), ['b.png', 'a.png']);
  assert.equal(history.at(-1).prompt, 'again');
});

test('the stitch uses the picture only when this build can, it is on, and there is a picture', () => {
  const c = load();
  const on = { enabled: true, image_path: 'C:\\p\\a.png', seconds: 3 };
  assert.deepEqual({ ...c.thumbnailIntroForStitch({ thumbnailAvailable: true, thumbnailIntro: on }) }, { image_path: 'C:\\p\\a.png', seconds: 3 });
  assert.equal(c.thumbnailIntroForStitch({ thumbnailAvailable: false, thumbnailIntro: on }), null);
  assert.equal(c.thumbnailIntroForStitch({ thumbnailAvailable: true, thumbnailIntro: { ...on, enabled: false } }), null);
  assert.equal(c.thumbnailIntroForStitch({ thumbnailAvailable: true, thumbnailIntro: { ...on, image_path: '' } }), null);
  assert.equal(c.thumbnailIntroForStitch({ thumbnailAvailable: true }), null);
  assert.equal(c.thumbnailIntroForStitch(null), null);
});

test('only the full video gets the picture, not a preview of some scenes or of a slice of the song', () => {
  const c = load();
  assert.equal(c.stitchIsFullVideo({}), true);
  assert.equal(c.stitchIsFullVideo({ outputPrefix: 'EXPORT_VIDEO' }), true);
  assert.equal(c.stitchIsFullVideo({ segments: [{ id: 's1' }] }), false);
  assert.equal(c.stitchIsFullVideo({ audioDuration: 12 }), false);
  assert.equal(c.stitchIsFullVideo({ audioStart: 4 }), false);
  assert.equal(c.stitchIsFullVideo({ timelineOffset: 10 }), false);
});
