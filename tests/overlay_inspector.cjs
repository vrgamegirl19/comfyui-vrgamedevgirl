const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule } = require('./builder_source.cjs');

// The settings of an overlay (web/music_video_builder/overlay_inspector.mjs): which controls each kind has, how a control maps to the clip,
// and the presets. The panel that draws them is checked in the app.
const context = vm.createContext({});
vm.runInContext(readBuilderModule('overlay_layers.mjs'), context);
vm.runInContext(readBuilderModule('overlay_inspector.mjs'), context);
const ov = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));
const norm = (list) => plain(ov('normalizeOverlayLayers')(list));
const keys = (clip, options) => plain(ov('overlayControlsFor')(clip, options)).map((c) => c.key);

test('every kind has the common controls and its own', () => {
  const common = ['enabled', 'locked', 'start', 'end', 'opacity', 'fade_in', 'fade_out'];
  assert.deepEqual(keys(norm([{ kind: 'grain' }])[0]), [...common, 'strength', 'chroma', 'size', 'animated']);
  assert.deepEqual(keys(norm([{ kind: 'vignette' }])[0]), [...common, 'strength']);
  assert.deepEqual(keys(norm([{ kind: 'media' }])[0]), [...common, 'blend_mode', 'scale', 'loop', 'speed', 'flip_h', 'brightness', 'contrast', 'tint']);
  assert.deepEqual(keys(norm([{ kind: 'lut' }])[0]), [...common, 'lut', 'strength']);
  assert.deepEqual(keys(norm([{ kind: 'glitch' }])[0]), common, 'an unknown kind has only the common ones');
  assert.deepEqual(keys(null), []);
});

test('a LUT clip lists the installed LUTs, or says there are none', () => {
  const clip = norm([{ kind: 'lut' }])[0];
  const select = (luts) => plain(ov('overlayControlsFor')(clip, { luts })).find((c) => c.key === 'lut');
  assert.deepEqual(select(['Teal Orange.cube', 'Warm.cube']).options.map((o) => [o.value, o.label]), [['', 'Choose a LUT'], ['Teal Orange.cube', 'Teal Orange'], ['Warm.cube', 'Warm']]);
  assert.equal(select([]).options[0].label, 'No LUTs installed');
});

test('percent controls show 0 to 100 and store 0 to 1', () => {
  const clip = norm([{ kind: 'vignette', strength: 0.55, opacity: 0.9 }])[0];
  const controls = Object.fromEntries(plain(ov('overlayControlsFor')(clip)).map((c) => [c.key, c]));
  assert.equal(vm.runInContext('(c, v) => c.toUi(v)', context)(ov('overlayControlsFor')(clip).find((c) => c.key === 'strength'), 0.55), 55);
  assert.equal(vm.runInContext('(c, v) => c.fromUi(v)', context)(ov('overlayControlsFor')(clip).find((c) => c.key === 'opacity'), 90), 0.9);
  assert.equal(controls.opacity.max, 100);
  assert.equal(controls.fade_in.max, 5);
  assert.equal(controls.start.kind, 'number');
  assert.equal(controls.opacity.kind, 'range');
});

test('a change is clamped by the clip rules and touches only that clip', () => {
  const clips = norm([{ kind: 'grain', id: 'g', start: 2, end: 8 }, { kind: 'vignette', id: 'v', start: 0, end: 5 }]);
  const change = (key, value, id = 'g') => plain(ov('applyOverlayChange')(clips, id, key, value));
  assert.equal(change('strength', 500)[0].strength, 60);
  assert.equal(change('strength', 12)[0].strength, 12);
  assert.equal(change('opacity', 0.5)[0].opacity, 0.5);
  assert.equal(change('opacity', -2)[0].opacity, 0);
  assert.equal(change('size', 'huge')[0].size, 'fine');
  assert.equal(change('size', 'coarse')[0].size, 'coarse');
  assert.equal(change('enabled', false)[0].enabled, false);
  assert.deepEqual(change('strength', 20).map((c) => c.id), ['g', 'v']);
  assert.equal(change('strength', 20)[1].strength, 0.7, 'the other clip is untouched');
  assert.equal(change('strength', 20, 'nope')[0].strength, 28, 'an unknown id changes nothing');
});

test('moving the start past the end, or the end before the start, keeps the clip as long as it was', () => {
  const clips = norm([{ kind: 'grain', id: 'g', start: 2, end: 8 }]);
  const change = (key, value) => plain(ov('applyOverlayChange')(clips, 'g', key, value))[0];
  assert.deepEqual([change('start', 10).start, change('start', 10).end], [10, 16]);
  assert.deepEqual([change('end', 1).start, change('end', 1).end], [0, 1]);
  assert.deepEqual([change('start', 5).start, change('start', 5).end], [5, 8]);
  assert.deepEqual([change('end', 20).start, change('end', 20).end], [2, 20]);
});

test('a clip can be reset, stretched, and the presets add unlocked clips over the range', () => {
  const clips = norm([{ kind: 'media', id: 'm', start: 3, end: 9, path: 'overlay/a.mov', source_start: 2, opacity: 0.2, speed: 1.5, tint: '#ff0000', locked: false }]);
  const reset = plain(ov('resetOverlayLayer')(clips, 'm'))[0];
  assert.deepEqual([reset.path, reset.start, reset.end, reset.source_start], ['overlay/a.mov', 3, 9, 2]);
  assert.deepEqual([reset.opacity, reset.speed, reset.tint, reset.locked], [0.9, 1, '', true]);
  assert.deepEqual(plain(ov('fitOverlayLayer')(clips, 'm', 0, 20)).map((c) => [c.start, c.end]), [[0, 20]]);
  assert.deepEqual(plain(ov('fitOverlayLayer')(clips, 'm', 5, 5)).map((c) => [c.start, c.end]), [[3, 9]], 'an empty span changes nothing');
  const preset = plain(ov('overlayPresetClips')('subtle_film', { total: 60, range: { in: 10, out: 20 } }));
  assert.deepEqual(preset.map((c) => [c.kind, c.start, c.end, c.locked]), [['grain', 10, 20, false], ['vignette', 10, 20, false]]);
  assert.deepEqual(preset.map((c) => [c.kind, c.strength]), [['grain', 4], ['vignette', 0.03]], 'Subtle film is grain 4 and a vignette of 3 %');
  assert.deepEqual(plain(ov('overlayPresetClips')('heavy_16mm', { total: 30 })), [], 'Heavy 16mm is gone');
  assert.deepEqual(Object.keys(plain(ov('OVERLAY_PRESETS'))), ['subtle_film']);
  assert.deepEqual(plain(ov('overlayPresetClips')('nope', { total: 30 })), []);
});

test('a text clip has the words, font, look and position controls, and the font list always holds the clip\'s own font', () => {
  const common = ['enabled', 'locked', 'start', 'end', 'opacity', 'fade_in', 'fade_out'];
  const clip = norm([{ kind: 'text', text: 'Hello', font: 'Caveat' }])[0];
  assert.deepEqual(keys(clip, { fonts: ['Anton', 'Caveat'] }), [...common, 'text', 'font', 'size', 'color', 'outline_width', 'outline_color', 'shadow', 'x', 'y', 'animation']);
  const controls = plain(ov('overlayControlsFor')(clip, { fonts: ['Anton', 'Caveat'] }));
  const control = (key) => controls.find((c) => c.key === key);
  assert.equal(control('text').kind, 'textarea');
  assert.deepEqual(control('font').options.map((o) => o.value), ['Anton', 'Caveat']);
  assert.equal(control('color').clearable, false, 'text always has a color, so there is no "No tint" button');
  assert.deepEqual(control('animation').options.map((o) => o.value), ['none', 'slide_up']);
  const options = (fonts, current) => plain(ov('overlayFontOptions')(fonts, current));
  assert.deepEqual(options(['Anton'], 'Pacifico').map((o) => [o.value, o.label]), [['Anton', 'Anton'], ['Pacifico', 'Pacifico (not installed)']]);
  assert.deepEqual(options(['Anton'], 'anton').map((o) => o.value), ['Anton'], 'a font is found by name whatever its case');
  assert.deepEqual(options(undefined, ''), []);
});

test('fonts that say where they come from are shown in groups, and the fonts installed on the computer are one of them', () => {
  const options = (fonts, current) => plain(ov('overlayFontOptions')(fonts, current));
  const list = [
    { name: 'Anton', file: 'Anton-Regular.ttf', bundled: true, system: false },
    { name: 'My Hand Font', file: 'My Hand Font.otf', bundled: false, system: false },
    { name: 'Arial', file: 'arial.ttf', bundled: false, system: true },
    { name: 'Arial Black', file: 'ariblk.ttf', bundled: false, system: true },
    { name: '' },
  ];
  assert.deepEqual(options(list, 'Pacifico').map((o) => [o.value, o.group || '']), [
    ['Anton', 'Included with the app'], ['My Hand Font', 'Your font files'], ['Arial', 'Installed on this computer'], ['Arial Black', 'Installed on this computer'], ['Pacifico', ''],
  ]);
  assert.deepEqual(options(['Anton', { name: 'Caveat' }], '').map((o) => o.group), [undefined, undefined], 'a list without sources has no groups');

  // The select puts each group under its own heading and an option without a group after them.
  const element = (tag) => ({ tag, children: [], append(...nodes) { this.children.push(...nodes); } });
  context.document = { createElement: element };
  const select = element('select');
  ov('appendOverlayOptions')(select, options(list, 'Pacifico'));
  const shape = (node) => (node.tag === 'optgroup' ? { group: node.label, options: node.children.map((o) => o.value) } : { option: node.value, label: node.textContent });
  assert.deepEqual(plain(select.children.map(shape)), [
    { group: 'Included with the app', options: ['Anton'] },
    { group: 'Your font files', options: ['My Hand Font'] },
    { group: 'Installed on this computer', options: ['Arial', 'Arial Black'] },
    { option: 'Pacifico', label: 'Pacifico (not installed)' },
  ]);
});

test('one font can be set on every text clip, or on those that overlap a scene, and on nothing else', () => {
  const clips = norm([
    { kind: 'text', id: 'a', text: 'One', start: 0, end: 4, font: 'Anton' },
    { kind: 'text', id: 'b', text: 'Two', start: 4, end: 9, font: 'Anton' },
    { kind: 'text', id: 'c', text: 'Three', start: 9, end: 12, font: 'Caveat' },
    { kind: 'grain', id: 'g', start: 0, end: 12 },
  ]);
  const apply = (range) => plain(ov('applyOverlayFontToText')(clips, 'Pacifico', range));
  const fonts = (result) => result.clips.filter((c) => c.kind === 'text').map((c) => c.font);

  const all = apply(null);
  assert.equal(all.count, 3);
  assert.deepEqual(fonts(all), ['Pacifico', 'Pacifico', 'Pacifico']);
  assert.deepEqual(all.clips.map((c) => c.id), ['a', 'b', 'c', 'g'], 'no clip is added, removed or moved');
  assert.deepEqual(all.clips.map((c) => [c.start, c.end]), [[0, 4], [4, 9], [9, 12], [0, 12]]);
  assert.deepEqual(all.clips.map((c) => c.text).slice(0, 3), ['One', 'Two', 'Three'], 'the words stay');

  // A clip that only touches the scene's edge is not in it.
  const scene = apply({ start: 4, end: 9 });
  assert.equal(scene.count, 1);
  assert.deepEqual(fonts(scene), ['Anton', 'Pacifico', 'Caveat']);
  const wide = apply({ start: 3, end: 10 });
  assert.equal(wide.count, 3, 'a caption that runs across the scene edge is in it');

  assert.equal(apply({ start: 20, end: 30 }).count, 0, 'a range with no text changes nothing');
  assert.deepEqual(fonts(apply({ start: 5, end: 5 })), ['Anton', 'Pacifico', 'Caveat'], 'a range with no length reaches only the clip it falls inside, never everything');
  assert.equal(plain(ov('applyOverlayFontToText')(clips, 'Pacifico', { start: 0, end: 12 })).clips.find((c) => c.id === 'g').kind, 'grain');
  assert.deepEqual(plain(ov('applyOverlayFontToText')(null, 'Pacifico')), { clips: [], count: 0 });
  assert.deepEqual(plain(clips.filter((c) => c.kind === 'text').map((c) => c.font)), ['Anton', 'Anton', 'Caveat'], 'the list it was given is not changed');
});

test('the words of a text clip are changed through the same path as every other setting', () => {
  const clips = norm([{ kind: 'text', id: 't', text: 'Hello', start: 2, end: 6 }]);
  const change = (key, value) => plain(ov('applyOverlayChange')(clips, 't', key, value))[0];
  assert.equal(change('text', 'Two\nlines').text, 'Two\nlines');
  assert.equal(change('size', 5000).size, 400, 'the size is limited');
  assert.equal(change('animation', 'typewriter').animation, 'none', 'an animation that does not exist is none');
  assert.equal(change('font', 'Oswald').font, 'Oswald');
});
