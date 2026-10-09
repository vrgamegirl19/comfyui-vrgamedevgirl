const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');
function fixture() {
  const c = vm.createContext({ crypto, postJson: async () => ({ audio_path: 'mix.wav', peaks: [0, 1] }) });
  vm.runInContext(readBuilderModule('audio_clip_editor.mjs'), c);
  return c;
}
function editorFixture() {
  const c = fixture(), events = {}, saves = [];
  class Element {
    constructor() { this.style = {}; this.children = []; }
    append(...items) { this.children.push(...items); }
    remove() {}
    contains(target) { return this.children.includes(target); }
    getContext() { return { beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }; }
  }
  Object.assign(c, {
    document: { createElement: () => new Element(), querySelector: () => null, body: new Element() },
    window: { innerWidth: 1000, innerHeight: 800, addEventListener: (name, fn) => { events[name] = fn; },
      removeEventListener: name => { delete events[name]; } },
    makeButton: label => Object.assign(new Element(), { textContent: label }), toast() {},
    TIMELINE_SCENE_AUDIO_TOP: 110, TIMELINE_SCENE_AUDIO_HEIGHT: 28,
  });
  const state = { videoType: 'speaking', pxPerSecond: 20, audioClips: [{ ...clip }],
    segments: [{ id: 's', start: 0, end: 10 }] };
  const history = [];
  const addButton = new Element();
  const editor = c.createAudioClipEditor({ state, addAudioClipButton: addButton,
    projectInput: { value: 'project' }, currentGlobalTime: () => 5,
    pushHistory: () => history.push(JSON.stringify(state)), render() {}, pauseTimelineForEditing() {},
    setActiveSegment() {}, autoSaveSessionQuiet: async reason => { saves.push(reason); } });
  const layer = new Element();
  editor.renderAudioClips(layer);
  const event = x => ({ button: 0, pointerId: 1, clientX: x, clientY: 100,
    preventDefault() {}, stopPropagation() {} });
  return { c, state, history, saves, editor, layer, events, event, addButton };
}
const clip = { id: 'a', scene_id: 's', path: 'original.wav', start: 3, source_start: 2,
  duration: 4, full_duration: 10, peaks: [0, .5, 1], name: 'Speech' };
test('split preserves source offsets and does not create scenes', () => {
  const c = fixture();
  const [left, right] = c.splitAudioClip(clip, 5);
  assert.equal(left.duration, 2);
  assert.equal(right.duration, 2);
  assert.equal(right.start, 5);
  assert.equal(right.source_start, 4);
  assert.equal(right.path, left.path);
  assert.notEqual(right.id, left.id);
  assert.equal(clip.duration, 4);
  assert.throws(() => c.splitAudioClip(clip, 2));
});
test('legacy imported audio from both entry paths becomes the same editable track', () => {
  const c = fixture();
  const state = { segments: [{ id: 's', start: 3, end: 7, custom_audio_path: 'audio.wav',
    custom_audio_source_start: 2, custom_audio_duration: 4, custom_audio_full_duration: 10 }] };
  const converted = c.audioClipsForState(state);
  assert.equal(converted[0].start, 3);
  assert.equal(converted[0].source_start, 2);
  assert.equal(converted[0].duration, 4);
  assert.equal(state.audioClips, undefined);
  state.audioClips = converted;
  const reloaded = JSON.parse(JSON.stringify(state));
  assert.equal(c.audioClipsForState(reloaded)[0].source_start, 2);
});
test('explicitly deleting all pieces stays empty rather than restoring original audio', () => {
  const c = fixture();
  const state = { audioClips: [], segments: [{ end: 6, custom_audio_path: 'audio.wav' }] };
  assert.equal(c.audioClipsForState(state).length, 0);
  assert.equal(c.audioEditDuration(state), 6);
});
test('dragging an audio piece moves audio without changing scene timing, and saves it', () => {
  const f = editorFixture();
  f.layer.children[0].onpointerdown(f.event(100));
  f.events.pointermove(f.event(140));
  f.events.pointerup({ ...f.event(140), type: 'pointerup' });
  assert.equal(f.state.audioClips[0].start, 5);
  assert.equal(f.state.audioClips[0].source_start, 2);
  assert.equal(f.state.segments[0].start, 0);
  assert.equal(f.state.segments[0].end, 10);
  assert.equal(f.history.length, 1);
  assert.equal(JSON.parse(f.history[0]).audioClips[0].start, 3);
  assert.equal(f.saves.length, 1);
  assert.equal(Object.keys(f.events).length, 0);
});
test('trimming the left audio edge advances its source offset', () => {
  const f = editorFixture();
  f.layer.children[0].children[2].onpointerdown(f.event(100));
  f.events.pointermove(f.event(120));
  f.events.pointerup({ ...f.event(120), type: 'pointerup' });
  assert.equal(f.state.audioClips[0].start, 4);
  assert.equal(f.state.audioClips[0].source_start, 3);
  assert.equal(f.state.audioClips[0].duration, 3);
});
test('clip menu splits at the playhead and deletion does not restore source audio', () => {
  const f = editorFixture();
  f.layer.children[0].oncontextmenu(f.event(100));
  f.c.document.body.children[0].children[0].onclick();
  assert.equal(f.state.audioClips.length, 2);
  assert.equal(f.state.audioClips[1].source_start, 4);
  assert.equal(f.state.segments.length, 1);
});
test('mix preparation is cached and rejects results after newer edits', async () => {
  const c = fixture();
  const state = { videoType: 'speaking', audioClips: [clip], segments: [{ end: 10 }] };
  let calls = 0;
  c.postJson = async () => { calls++; return { audio_path: 'mix.wav', peaks: [] }; };
  await c.prepareEditedAudio(state, 'project');
  await c.prepareEditedAudio(state, 'project');
  assert.equal(calls, 1);
  state.audioClips = [{ ...clip, start: 4 }];
  c.postJson = async () => {
    state.audioClips = [{ ...clip, start: 6 }];
    return { audio_path: 'stale.wav' };
  };
  await assert.rejects(c.prepareEditedAudio(state, 'project'), /Audio changed/);
  assert.equal(state.audioClipMixPath, 'mix.wav');
});
test('edited tracks cannot fall back to the original per-scene audio', () => {
  const source = readBuilderModule('timeline_state.mjs');
  const c = vm.createContext({ state: { videoType: 'speaking', audioClips: [], audioClipMixPath: 'edited.wav' },
    speakingAudioEditsActive: state => state.videoType === 'speaking' && Array.isArray(state.audioClips),
    audioInput: { value: 'original.wav' }, usingSceneAudioMode: () => true });
  for (const name of ['currentProjectAudioPath', 'usingSceneAudioPlaybackMode',
    'timelineAudioPathForSegment', 'timelineAudioSourceStartForSegment']) {
    vm.runInContext(functionSource(source, name), c);
  }
  assert.equal(c.usingSceneAudioPlaybackMode(), false);
  assert.equal(c.timelineAudioPathForSegment({ start: 4, custom_audio_path: 'old.wav' }), 'edited.wav');
  assert.equal(c.timelineAudioSourceStartForSegment({ start: 4, custom_audio_source_start: 2 }), 4);
  c.state.audioClipMixPath = '';
  assert.equal(c.currentProjectAudioPath(), '');
  assert.equal(c.timelineAudioPathForSegment({ custom_audio_path: 'old.wav' }), '');
});
test('additional clips append independent lanes without replacing dialogue', async () => {
  const f = editorFixture();
  f.c.FileReader = class {
    readAsDataURL(file) { this.result = `data:audio/wav;base64,${file.name}`; this.onload(); }
  };
  let calls = 0;
  f.c.postJson = async (route, payload) => {
    assert.equal(payload.preserve_source, true);
    calls++;
    return { saved_path: `source_${calls}.wav`, duration: 8, peaks: [0.2] };
  };
  await f.editor.importAdditionalAudio([{ name: 'score.wav' }, { name: 'another.wav' }]);
  assert.equal(f.state.audioClips.length, 3);
  assert.equal(f.state.audioClips[0].path, 'original.wav');
  assert.equal(f.state.audioClips[1].lane, 1);
  assert.equal(f.state.audioClips[2].lane, 2);
  assert.equal(f.state.audioClips[1].start, 5);
  assert.equal(f.state.audioClips[1].volume, 0.25);
  assert.equal(f.state.audioClips[1].include_in_generation, false);
  const reloaded = JSON.parse(JSON.stringify(f.state));
  assert.equal(reloaded.audioClips[2].lane, 2);
  const layer = { children: [], append(item) { this.children.push(item); } };
  f.editor.renderAudioClips(layer);
  assert.match(layer.children[1].style.cssText, /top:142px/);
  assert.match(layer.children[2].style.cssText, /top:174px/);
});
test('controls and clip edits are disabled for every other video type', async () => {
  const f = editorFixture();
  const types = ['singing', 'no_lip_sync', 'music', 'instrumental', 'music_video', ''];
  for (const type of types) {
    f.state.videoType = type;
    const layer = { append() { assert.fail('Audio editor appeared outside speaking'); } };
    f.editor.renderAudioClips(layer);
    assert.equal(f.addButton.style.display, 'none');
    await f.editor.importAdditionalAudio([{ name: 'score.wav' }]);
    assert.equal(await f.c.prepareEditedAudio(f.state, 'project'), null);
    assert.equal(f.c.speakingAudioLaneCount(f.state), 1);
  }
  assert.equal(f.state.audioClips.length, 1);
  assert.equal(f.history.length, 0);
});
test('volume zero and mute survive splitting and reload', () => {
  const f = editorFixture();
  const piece = { ...clip, volume: 0, muted: true, lane: 2, role: 'music', include_in_generation: false };
  const pieces = JSON.parse(JSON.stringify(f.c.splitAudioClip(piece, 5)));
  for (const part of pieces) {
    assert.equal(part.volume, 0); assert.equal(part.muted, true);
    assert.equal(part.lane, 2); assert.equal(part.include_in_generation, false);
  }
});
test('generation and playback use separately cached mixes', async () => {
  const c = fixture(), requests = [];
  c.postJson = async (route, payload) => {
    requests.push(payload);
    return { audio_path: payload.generation_only ? 'voice.wav' : 'voice_score.wav' };
  };
  const state = { videoType: 'speaking', audioClips: [clip], segments: [{ end: 10 }] };
  await c.prepareEditedAudio(state, 'project');
  await c.prepareEditedAudio(state, 'project', { generation: true });
  await c.prepareEditedAudio(state, 'project');
  assert.equal(requests.length, 2);
  assert.equal(state.audioClipMixPath, 'voice_score.wav');
  assert.equal(state.audioClipGenerationMixPath, 'voice.wav');
});
test('score joins generated scene audio at final stitch without duplicating dialogue', async () => {
  const c = fixture(); let payload;
  c.postJson = async (route, request) => { payload = request; return { audio_path: 'combined.wav' }; };
  const scenes = [{ id: 's', start: 0, end: 4 }];
  const state = { videoType: 'speaking', segments: scenes, audioClips: [
    { ...clip, role: 'dialogue' }, { ...clip, role: 'music', volume: 0.2 } ] };
  await c.prepareEmbeddedAudioWithClips(state, 'project', scenes, ['rendered.mp4']);
  assert.equal(payload.clips.length, 2);
  assert.equal(payload.clips[0].path, 'rendered.mp4');
  assert.equal(payload.clips[1].volume, 0.2);
  state.videoType = 'singing';
  assert.equal(await c.prepareEmbeddedAudioWithClips(state, 'project', scenes, ['rendered.mp4']), null);
});
test('volume and mute controls update only the selected audio piece', () => {
  const f = editorFixture();
  f.layer.children[0].oncontextmenu(f.event(100));
  const menu = f.c.document.body.children[0];
  const volume = menu.children[3].children[1];
  volume.value = '0'; volume.onchange();
  assert.equal(f.state.audioClips[0].volume, 0);
  menu.children[4].onclick();
  assert.equal(f.state.audioClips[0].muted, true);
  assert.equal(f.state.segments[0].end, 10);
});
test('switching away from speaking immediately redraws and hides the audio editor', async () => {
  const f = editorFixture(), calls = [];
  const source = readBuilderModule('toolbar.mjs');
  const start = source.indexOf('  videoTypeSelect.onchange = async');
  Object.assign(f.c, { state: f.state, videoTypeSelect: { value: 'no_lip_sync' },
    normalizeVideoType: value => value, syncVideoTypeControl() {},
    pauseTimelineForEditing: () => calls.push('pause'),
    autoSaveSessionQuiet: async () => {}, VIDEO_TYPE_OPTIONS: [],
    render: () => { f.editor.renderAudioClips({ append() {} }); calls.push('render'); } });
  vm.runInContext(source.slice(start, source.indexOf('  loadButton.onclick', start)), f.c);
  await f.c.videoTypeSelect.onchange();
  assert.deepEqual(calls, ['pause', 'render']);
  assert.equal(f.addButton.style.display, 'none');
});
