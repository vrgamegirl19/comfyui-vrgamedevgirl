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
    assert.equal(intro.seconds, undefined, 'the thumbnail is one frame: there is no length');
    assert.equal(intro.change, 0.65);
    assert.equal(intro.engine, 'nano_banana');
    assert.deepEqual(Array.from(intro.history), []);
  }
});

test('the change setting is kept between 0 and 1, and a missing one is 0.65', () => {
  const c = load();
  assert.equal(c.clampThumbnailChange(0), 0);
  assert.equal(c.clampThumbnailChange(2), 1);
  assert.equal(c.clampThumbnailChange(-1), 0);
  assert.equal(c.clampThumbnailChange('0.4'), 0.4);
  for (const bad of ['', null, undefined, 'abc', NaN]) assert.equal(c.clampThumbnailChange(bad), 0.65);
  assert.equal(c.normalizeThumbnailIntro({ change: 0.2 }).change, 0.2);
  assert.equal(c.normalizeThumbnailIntro({ change: 9 }).change, 1);
});

test('every image workflow is known, with the kind of prompt it needs', () => {
  const c = load();
  assert.deepEqual(Array.from(vm.runInContext('THUMBNAIL_ENGINES', c)), ['overlay', 'qwen_image_edit', 'nano_banana', 'flux_klein', 'moss_image_edit', 'krea2_2pass', 'zimage', 'ernie_image', 'moss_image']);
  assert.equal(c.thumbnailEngineKind('overlay'), 'overlay', 'it is not an image workflow: no prompt, no LLM, no ComfyUI');
  assert.equal(c.normalizeThumbnailIntro({ engine: 'overlay' }).engine, 'overlay');
  assert.equal(c.normalizeThumbnailIntro({ engine: 'qwen_image_edit' }).engine, 'qwen_image_edit');
  for (const edit of ['nano_banana', 'flux_klein', 'qwen_image_edit']) assert.equal(c.thumbnailEngineKind(edit), 'edit');
  assert.equal(c.thumbnailEngineKind('moss_image_edit'), 'moss_edit', 'MOSS-Image Edit follows its own short instruction');
  for (const restyle of ['krea2_2pass', 'zimage', 'ernie_image', 'moss_image']) assert.equal(c.thumbnailEngineKind(restyle), 'restyle');
  assert.equal(c.thumbnailEngineKind('unknown'), 'edit');
  assert.equal(c.normalizeThumbnailIntro({ engine: 'moss_image' }).engine, 'moss_image');
  assert.equal(c.normalizeThumbnailIntro({ engine: 'moss_image_edit' }).engine, 'moss_image_edit');
  assert.equal(c.normalizeThumbnailIntro({ history: [{ path: 'a.png', engine: 'moss_image_edit' }] }).history[0].engine, 'moss_image_edit');
  assert.equal(c.normalizeThumbnailIntro({ engine: 'bogus' }).engine, 'nano_banana');
  assert.equal(c.normalizeThumbnailIntro({ history: [{ path: 'a.png', engine: 'zimage' }] }).history[0].engine, 'zimage');
});

test('the overlay choices are kept to what the pack knows, and a missing or wrong one is the default', () => {
  const c = load();
  const plain = (value) => JSON.parse(JSON.stringify(value));
  assert.deepEqual(plain(c.normalizeThumbnailOverlay(undefined)), { color: 'yellow', position: 'auto', strength: 60 });
  assert.deepEqual(plain(c.normalizeThumbnailOverlay({ color: 'cyan', position: 'right', strength: 35.4 })), { color: 'cyan', position: 'right', strength: 35 });
  assert.deepEqual(plain(c.normalizeThumbnailOverlay({ color: 'chartreuse', position: 'diagonal', strength: 'abc' })), { color: 'yellow', position: 'auto', strength: 60 });
  assert.equal(c.normalizeThumbnailOverlay({ strength: 0 }).strength, 0, 'zero is a real choice: it leaves the frame alone');
  assert.equal(c.normalizeThumbnailOverlay({ strength: -4 }).strength, 0);
  assert.equal(c.normalizeThumbnailOverlay({ strength: 900 }).strength, 100);
  for (const empty of ['', null]) assert.equal(c.normalizeThumbnailOverlay({ strength: empty }).strength, 60);
  assert.deepEqual(plain(c.normalizeThumbnailIntro({}).overlay), { color: 'yellow', position: 'auto', strength: 60 });
  assert.deepEqual(plain(c.normalizeThumbnailIntro({ overlay: { color: 'red', strength: 20 } }).overlay), { color: 'red', position: 'auto', strength: 20 });
  assert.deepEqual(Array.from(vm.runInContext('THUMBNAIL_OVERLAY_COLORS', c)), ['yellow', 'white', 'orange', 'cyan', 'red']);
  assert.deepEqual(Array.from(vm.runInContext('THUMBNAIL_OVERLAY_POSITIONS', c)), ['auto', 'top', 'bottom', 'left', 'right']);
});

test('the title sentence asks for the exact text: an instruction for edits, a statement of the picture for restyles, nothing without a title', () => {
  const c = load();
  assert.match(c.thumbnailTitleClause('HEAT', 'edit'), /^Add the title text "HEAT" exactly as written and spelled in huge bold sans-serif capital letters .* Add no other text\.$/);
  assert.equal(c.thumbnailTitleClause('HEAT', 'moss_edit'), c.thumbnailTitleClause('HEAT', 'edit'));
  assert.match(c.thumbnailTitleClause('HEAT', 'restyle'), /^The title text "HEAT" is written exactly as spelled in huge bold .* No other text\.$/);
  for (const empty of ['', '   ', null, undefined]) assert.equal(c.thumbnailTitleClause(empty, 'edit'), '');
  assert.match(c.thumbnailTitleClause('  Never   look  back ', 'edit'), /"Never look back"/, 'spaces are tidied');
  assert.match(c.thumbnailTitleClause('He said "go"', 'edit'), /"He said 'go'"/, 'a double quote inside the title cannot end the quoted text');
});

test('a prompt always ends up with the title in it, exactly as typed', () => {
  const c = load();
  const ensure = (prompt, title, kind = 'edit') => c.ensureThumbnailTitle(prompt, title, kind);
  // Missing: the mandatory sentence goes on the end.
  assert.equal(ensure('A close-up.', 'HEAT'), `A close-up. ${c.thumbnailTitleClause('HEAT', 'edit')}`);
  assert.match(ensure('A close-up.', 'HEAT', 'restyle'), /The title text "HEAT" is written exactly as spelled/);
  // Already in quotes: the spelling and case that were typed are put back and the quotes made plain.
  assert.equal(ensure('The word “Heat” in huge letters.', 'HEAT'), 'The word "HEAT" in huge letters.');
  assert.equal(ensure("The word 'heat' glows.", 'Heat'), 'The word "Heat" glows.');
  assert.equal(ensure('Text "NEVER   LOOK back" at the top.', 'Never look back'), 'Text "Never look back" at the top.');
  // The word without quotes is not the title (warm heat on a cheek): the sentence is added instead of guessing which one.
  assert.match(ensure('Warm heat on his cheek.', 'HEAT'), /^Warm heat on his cheek\. Add the title text "HEAT"/);
  // Characters that mean something in a pattern are only characters.
  assert.match(ensure('x', 'What? (really) [now] $5 a+b'), /"What\? \(really\) \[now\] \$5 a\+b"/);
  assert.equal(ensure('Say "What? (really)" loud.', 'what? (REALLY)'), 'Say "what? (REALLY)" loud.');
  assert.match(ensure('x', 'He said "go"'), /"He said 'go'"/);
  // No title or no prompt changes nothing, and doing it twice changes nothing more.
  assert.equal(ensure('  A close-up. ', ''), 'A close-up.');
  assert.equal(ensure('A close-up.', null), 'A close-up.');
  assert.equal(ensure('', 'HEAT'), '');
  const once = ensure('A close-up.', 'HEAT');
  assert.equal(ensure(once, 'HEAT'), once);
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
  assert.deepEqual({ ...c.thumbnailIntroForStitch({ thumbnailAvailable: true, thumbnailIntro: on }) }, { image_path: 'C:\\p\\a.png' }, 'one frame: only the picture');
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

// The frame picker: which scene shows a timeline frame, and how far into its own video that is.
const scene = (id, start, end, extra = {}) => ({ id, start, end, video: `C:/v/${id}.mp4`, image: '', ...extra });

test('the picker lists scenes with something to show, in timeline order, as whole frames', () => {
  const c = load();
  const items = c.thumbnailTimelineItems([
    scene('b', 2, 4.5),
    scene('a', 0, 2),
    scene('empty', 4.5, 6, { video: '', image: '' }),
    scene('flat', 6, 6),
    scene('image only', 6, 8, { video: '', image: 'C:/i/8.png' }),
    { id: 'no times', video: 'x.mp4' },
  ]);
  assert.deepEqual(Array.from(items, (item) => item.id), ['a', 'b', 'image only']);
  assert.deepEqual(Array.from(items, (item) => [item.startFrame, item.endFrame]), [[0, 48], [48, 108], [144, 192]]);
});

test('the frame range runs from the first scene\'s first frame to the last scene\'s last frame', () => {
  const c = load();
  assert.deepEqual({ ...c.thumbnailFrameRange(c.thumbnailTimelineItems([scene('a', 1, 2), scene('b', 2, 5)])) }, { min: 24, max: 119 });
  assert.deepEqual({ ...c.thumbnailFrameRange([]) }, { min: 0, max: 0 });
});

test('a frame belongs to the scene that covers it, and a cut lands on the new scene', () => {
  const c = load();
  const items = c.thumbnailTimelineItems([scene('a', 0, 2), scene('b', 2, 4), scene('c', 5, 6)]);
  assert.equal(c.thumbnailItemAt(items, 0).id, 'a');
  assert.equal(c.thumbnailItemAt(items, 47).id, 'a');
  assert.equal(c.thumbnailItemAt(items, 48).id, 'b');
  assert.equal(c.thumbnailItemAt(items, 95).id, 'b');
  assert.equal(c.thumbnailItemAt(items, 96).id, 'c', 'the frame after a scene with a gap behind it shows the next scene');
  assert.equal(c.thumbnailItemAt(items, 110).id, 'c', 'a gap shows the next scene');
  assert.equal(c.thumbnailItemAt(items, 9999).id, 'c', 'past the end shows the last scene');
  assert.equal(c.thumbnailItemAt([], 5), null);
});

test('the moment inside a scene video is counted from the scene\'s first frame, and stays inside the scene', () => {
  const c = load();
  const [a, b] = c.thumbnailTimelineItems([scene('a', 0, 2), scene('b', 2, 4)]);
  assert.equal(c.thumbnailLocalSeconds(b, 48), 0);
  assert.equal(c.thumbnailLocalSeconds(b, 60), 0.5);
  assert.equal(c.thumbnailLocalSeconds(b, 95), 47 / 24);
  assert.equal(c.thumbnailLocalSeconds(b, 9999), 47 / 24);
  assert.equal(c.thumbnailLocalSeconds(a, -5), 0);
});

test('times read as minutes and seconds with hundredths', () => {
  const c = load();
  assert.equal(c.formatThumbnailTime(0), '0:00.00');
  assert.equal(c.formatThumbnailTime(5.5), '0:05.50');
  assert.equal(c.formatThumbnailTime(65.5), '1:05.50');
  assert.equal(c.formatThumbnailTime(-3), '0:00.00');
  assert.equal(c.formatThumbnailTime('abc'), '0:00.00');
});

test('the chosen source is kept only with a picture, and its kind falls back to an image', () => {
  const c = load();
  const kept = c.normalizeThumbnailIntro({ source_path: ' C:/p/source_0001.png ', source_kind: 'frame', source_frame: 298.9, source_seconds: 12.4 });
  assert.equal(kept.source_path, 'C:/p/source_0001.png');
  assert.equal(kept.source_kind, 'frame');
  assert.equal(kept.source_frame, 298);
  assert.equal(kept.source_seconds, 12.4);
  assert.equal(c.normalizeThumbnailIntro({ source_path: 'a.png', source_kind: 'bogus' }).source_kind, 'image');
  const none = c.normalizeThumbnailIntro({ source_kind: 'frame', source_frame: 5, source_seconds: -2 });
  assert.equal(none.source_kind, '');
  assert.equal(none.source_path, '');
  assert.equal(none.source_seconds, 0);
});
