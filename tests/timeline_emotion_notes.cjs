const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

function harness() {
  const state = { videoType: 'speaking', showTimelineEmotionTags: true, segments: [{ id: 'a', emotion_expression_tags: 'Curious' }] };
  const saved = [], history = [];
  const context = vm.createContext({
    document: { createElement: () => ({ style: {}, setAttribute() {} }) },
  });
  vm.runInContext(readBuilderModule('timeline_emotion_notes.mjs'), context);
  const box = context.createTimelineEmotionNote({ segment: state.segments[0], state,
    left: 20, width: 120, top: 300, height: 60, isActive: false,
    pushHistory: () => history.push(state.segments[0].emotion_expression_tags),
    autoSaveSessionQuiet: reason => saved.push(reason) });
  return { context, state, box, saved, history };
}

test('emotion lane is speaking-only and preserves saved text when hidden', () => {
  const { context, state, box } = harness();
  assert.equal(context.timelineEmotionNotesVisible(state), true);
  assert.equal(box.value, 'Curious');
  state.videoType = 'singing';
  assert.equal(context.timelineEmotionNotesVisible(state), false);
  assert.equal(state.segments[0].emotion_expression_tags, 'Curious');
  state.videoType = 'speaking';
  state.showTimelineEmotionTags = false;
  assert.equal(context.timelineEmotionNotesVisible(state), false);
});

test('editing saves direction to the live scene and captures undo before the first change', () => {
  const { box, state, history, saved } = harness();
  box.onfocus();
  box.value = 'Start curious, then frightened';
  box.oninput();
  assert.deepEqual(history, ['Curious']);
  assert.equal(state.segments[0].emotion_expression_tags, box.value);
  state.segments = [{ id: 'a', emotion_expression_tags: box.value }];
  box.value = 'Start curious, then quietly frightened';
  box.oninput();
  box.onblur();
  assert.equal(history.length, 1);
  assert.equal(state.segments[0].emotion_expression_tags, box.value);
  assert.equal(saved.length, 1);
  box.onblur();
  assert.equal(saved.length, 1);
});

test('Emotion Tag button shows in speaking mode and restores its enabled appearance', () => {
  const state = { videoType: 'singing', showTimelineEmotionTags: true };
  const button = { style: {} };
  const context = vm.createContext({ state, emotionTagButton: button, lyricNoteButton: { style: {} } });
  vm.runInContext(functionSource(readBuilderModule('beat_calibration.mjs'), 'syncLyricNoteControls'), context);
  context.syncLyricNoteControls();
  assert.equal(button.style.display, 'none');
  state.videoType = 'speaking';
  context.syncLyricNoteControls();
  assert.equal(button.style.display, '');
  assert.equal(button.textContent, 'Hide Emotion Tags');
  state.showTimelineEmotionTags = false;
  context.syncLyricNoteControls();
  assert.equal(button.textContent, '+ Emotion Tag');
});

test('emotion notes fit below line notes and above the waveform', () => {
  const state = { videoType: 'speaking', showTimelineEmotionTags: true, showTimelineLyricNotes: true };
  const context = vm.createContext({ state, TIMELINE_NOTE_HEIGHT: 60, TIMELINE_NOTE_GAP: 18,
    timelineLyricNoteTop: () => 200, timelineEmotionNotesVisible: s => s.videoType === 'speaking' && s.showTimelineEmotionTags });
  for (const name of ['timelineEmotionNoteTop', 'timelineWaveTop']) {
    vm.runInContext(functionSource(readBuilderModule('timeline_view.mjs'), name), context);
  }
  assert.equal(context.timelineEmotionNoteTop(), 278);
  assert.equal(context.timelineWaveTop(), 352);
  state.showTimelineLyricNotes = false;
  assert.equal(context.timelineEmotionNoteTop(), 200);
});

test('Emotion Tag button opens line notes, supports undo, and ignores clicks outside speaking mode', () => {
  const state = { videoType: 'speaking', showTimelineEmotionTags: false, showTimelineLyricNotes: false };
  let history = 0, saves = 0;
  const button = {};
  const context = vm.createContext({ state, emotionTagButton: button, pushHistory: () => history++,
    syncLyricNoteControls() {}, render() {}, autoSaveSessionQuiet: () => saves++ });
  const source = readBuilderModule('timeline_events.mjs');
  vm.runInContext(source.slice(source.indexOf('  emotionTagButton.onclick ='), source.indexOf('  zoomOutButton.onclick =')), context);
  button.onclick();
  assert.equal(state.showTimelineEmotionTags, true);
  assert.equal(state.showTimelineLyricNotes, true);
  assert.equal(history, 1);
  assert.equal(saves, 1);
  state.videoType = 'singing';
  button.onclick();
  assert.equal(history, 1);
  assert.equal(state.showTimelineEmotionTags, true);
});

test('edited emotion direction reaches the real prompt payload alongside inherited facial settings', () => {
  const { box, state, context } = harness();
  box.onfocus();
  box.value = 'Quiet curiosity, growing alarm';
  box.oninput();
  box.onchange();
  state.defaultFacialPerformance = 'custom';
  state.defaultFacialPerformanceCustom = 'Natural';
  vm.runInContext(readBuilderModule('emotion_expression.mjs'), context);
  const payload = context.emotionExpressionInput(state.segments[0], state);
  assert.equal(payload.emotion_expression_tags, box.value);
  assert.equal(payload.facial_performance_custom, 'Natural');
  assert.equal(context.hasEmotionExpressionInput(state.segments[0], state), true);
});
