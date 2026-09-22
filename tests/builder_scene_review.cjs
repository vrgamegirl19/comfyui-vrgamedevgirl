const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../web/VRGDG_MusicVideoBuilderUI.js'), 'utf8').replace(/\r\n/g, '\n');
const modal = source.slice(source.indexOf('  function openLyricReviewModal('), source.indexOf('  function openLyricMappingWorkflowModal('));
function section(start, end) {
  const a = modal.indexOf(start);
  const b = modal.indexOf(end, a);
  assert.ok(a >= 0 && b > a, `Missing review section: ${start}`);
  return modal.slice(a, b);
}

function fixture(single = true) {
  const state = { segments: [
    { id: 'a', start: 0, end: 10, lyric_text: 'A' },
    { id: 'b', start: 10, end: 20, lyric_text: 'B' },
    { id: 'c', start: 20, end: 30, lyric_text: 'C' },
  ], fluxReferenceBuilder: { subjects: [{ id: 'singer', name: 'Singer' }] } };
  const rows = state.segments.map(segment => {
    const inputs = new Map([
      ['[data-review-start]', { value: String(segment.start) }],
      ['[data-review-end]', { value: String(segment.end) }],
      ['[data-review-lyric-text]', { value: segment.lyric_text }],
      ['[data-review-location]', { value: 'location' }],
      ["[data-review-facial-performance='1']", { value: 'singing' }],
    ]);
    return { inputs, isConnected: true,
      dataset: { reviewSegmentId: segment.id, reviewLastStart: String(segment.start), reviewLastEnd: String(segment.end) },
      querySelector: selector => inputs.get(selector),
      querySelectorAll: selector => selector.includes('singer-choice')
        ? [{ checked: true, value: 'Singer', dataset: { reviewSubjectId: 'singer' } }]
        : selector.includes('present-subject') ? [{ checked: true, value: 'singer' }] : [],
    };
  });
  const messages = [];
  const events = [];
  const backdrop = { isConnected: true, remove() { this.isConnected = false; events.push('close'); }, addEventListener() {} };
  const context = vm.createContext({ state, rows, messages, events, backdrop,
    isSingleScene: single, targetScene: state.segments[1], targetSceneIndex: 1, allScenes: state.segments,
    save: { disabled: false }, close: {}, cancel: {}, activeLyricReviewBackdrop: null,
    reviewAudio: { pause() {} }, clearReviewStopGuards() {},
    reviewRows: () => single ? [rows[1]] : rows,
    liveReviewSegmentForRow: row => state.segments.find(s => s.id === row.dataset.reviewSegmentId),
    parseBulkTimeValue: Number, formatTime: String,
    updateReviewTimingDisplay() {}, timelineDuration: () => Math.max(...state.segments.map(s => s.end)),
    syncInspector() {}, render() {}, toast: message => messages.push(message),
    hasLockedVideo: segment => Boolean(segment.locked),
    timingModeSelect: { value: 'lock' }, maybeWarnShortReviewScene: async () => {},
    pushHistory() {}, ensureAllSegmentRuntimeFields() {}, syncSingleSubjectPerformerLabel() {},
    rowList: { querySelectorAll: () => single ? [rows[1]] : rows },
    collectBoundaryOverlapLyricOverrides: () => new Map(rows.map(row => [row.dataset.reviewSegmentId, row.querySelector('[data-review-lyric-text]').value])),
    normalizeFluxReferenceBuilder: value => value,
    pendingReviewWordMoves: [], applyPendingReviewWordMoves() {},
    applyLyricSectionsFromReferenceText() {}, syncLyricMapperFromSegments() {},
    syncIngredientsSceneMapFromSubjectMappings: refs => ({ refs }), currentVideoMode: () => 't2v',
    sortSegments: segments => segments.sort((a, b) => a.start - b.start),
    saveSession: async () => { events.push('persist'); }, showInfoModal: info => messages.push(info.title),
    openLyricReviewModal: options => { events.push(options.singleSceneId || 'all'); },
  });
  vm.runInContext(section('    const syncReviewRowFromSegment =', '    const syncAllReviewRowsFromSegments =')
    + section('    const rememberReviewRowTiming =', '    const showShortReviewSceneConfirm =')
    + section('    const singleReviewTimingNeighbors =', '    const setReviewRowStartToPlayhead =')
    + section('    const applyReviewRowValues =', '    const copyLyricReviewFields =')
    + section('    activeLyricReviewBackdrop = backdrop;', '\n  }'), context);
  return { context, state, rows, events, messages, backdrop,
    run: code => vm.runInContext(code, context),
    times: () => state.segments.map(s => [s.start, s.end]),
  };
}

test('navigation persists lyrics, performers and location before closing', async () => {
  const f = fixture();
  f.rows[1].querySelector('[data-review-lyric-text]').value = 'Corrected lyrics';
  await f.run('navigateReviewScene({ singleSceneId: "c" })');
  assert.deepEqual(f.events, ['persist', 'close', 'c']);
  assert.equal(f.state.segments[1].lyric_text, 'Corrected lyrics');
  assert.deepEqual(Array.from(f.state.segments[1].lyric_singers), ['Singer']);
  assert.equal(f.state.fluxReferenceBuilder.scene_map.b, 'location');
  assert.deepEqual(Array.from(f.state.fluxReferenceBuilder.performer_scene_map.b), ['singer']);
  assert.equal(f.state.segments[0].lyric_text, 'A');
  assert.equal(f.state.segments[2].lyric_text, 'C');
  assert.deepEqual(f.messages, []);
});

test('failed save keeps the current editor and pending fields available', async () => {
  const f = fixture();
  f.context.saveSession = async () => { throw Error('Disk full'); };
  f.rows[1].querySelector('[data-review-lyric-text]').value = 'Keep me';
  await f.run('navigateReviewScene({ focusSceneId: "b" })');
  assert.equal(f.backdrop.isConnected, true);
  assert.equal(f.rows[1].querySelector('[data-review-lyric-text]').value, 'Keep me');
  assert.equal(f.context.save.disabled, false);
  assert.deepEqual(f.events, []);
  assert.deepEqual(f.messages, ['Line Review Save Error']);
});

test('concurrent navigation waits for one save and opens only one destination', async () => {
  const f = fixture();
  let finishSave;
  f.context.saveSession = () => new Promise(resolve => { finishSave = resolve; });
  const first = f.run('navigateReviewScene({ singleSceneId: "c" })');
  await new Promise(resolve => setImmediate(resolve));
  await f.run('navigateReviewScene({ singleSceneId: "a" })');
  assert.deepEqual(f.events, []);
  finishSave();
  await first;
  assert.deepEqual(f.events, ['close', 'c']);
});

test('closing during a pending save does not reopen the editor', async () => {
  const f = fixture();
  let finishSave;
  f.context.saveSession = () => new Promise(resolve => { finishSave = resolve; });
  const navigation = f.run('navigateReviewScene({ singleSceneId: "c" })');
  await new Promise(resolve => setImmediate(resolve));
  f.run('closeModal()');
  finishSave();
  await navigation;
  assert.deepEqual(f.events, ['close']);
});

test('duplicate modal opens are blocked but immediate close/reopen is allowed', () => {
  const f = fixture();
  const guard = section('    if (activeLyricReviewBackdrop', '    const focusSceneId');
  f.run('let opens = 0; function tryOpen() {' + guard + ' opens++; } tryOpen();');
  assert.equal(f.run('opens'), 0);
  f.run('closeModal(); tryOpen();');
  assert.equal(f.run('opens'), 1);
});

test('ordinary start edits change only the previous shared boundary', async () => {
  const f = fixture();
  f.rows[1].querySelector('[data-review-start]').value = '7';
  await f.run('handleReviewStartEdited(rows[1])');
  assert.deepEqual(f.times(), [[0, 7], [7, 20], [20, 30]]);
});

test('start edits cannot cross an entire previous scene', async () => {
  const f = fixture();
  f.context.reviewRows = () => [f.rows[2]];
  f.rows[2].querySelector('[data-review-start]').value = '5';
  await f.run('handleReviewStartEdited(rows[2])');
  assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
  assert.equal(f.rows[2].querySelector('[data-review-start]').value, '20');
});

test('ordinary end edits leave later boundaries unchanged', async () => {
  const f = fixture();
  f.rows[1].querySelector('[data-review-end]').value = '23';
  await f.run('handleReviewEndEdited(rows[1])');
  assert.deepEqual(f.times(), [[0, 10], [10, 23], [23, 30]]);
});

test('end edits cannot consume an entire next scene', async () => {
  const f = fixture();
  f.rows[1].querySelector('[data-review-end]').value = '35';
  await f.run('handleReviewEndEdited(rows[1])');
  assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
  assert.equal(f.rows[1].querySelector('[data-review-end]').value, '20');
});

for (const single of [true, false]) {
  test(`ripple preserves later scene durations (single=${single})`, async () => {
    const f = fixture(single);
    f.context.timingModeSelect.value = 'ripple';
    f.rows[0].querySelector('[data-review-end]').value = '15';
    if (single) f.context.reviewRows = () => [f.rows[0]];
    await f.run('handleReviewEndEdited(rows[0])');
    assert.deepEqual(f.times(), [[0, 15], [15, 25], [25, 35]]);
  });
}

for (const lockedIndex of [0, 1]) {
  test(`start edit respects lock on affected scene ${lockedIndex}`, async () => {
    const f = fixture();
    f.state.segments[lockedIndex].locked = true;
    f.rows[1].querySelector('[data-review-start]').value = '7';
    await f.run('handleReviewStartEdited(rows[1])');
    assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
  });
}

test('ripple respects locked later scenes', async () => {
  const f = fixture();
  f.state.segments[2].locked = true;
  f.context.timingModeSelect.value = 'ripple';
  f.rows[1].querySelector('[data-review-end]').value = '23';
  await f.run('handleReviewEndEdited(rows[1])');
  assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
});

test('frozen timing still allows saving lyric corrections', async () => {
  const f = fixture();
  f.state.timingFrozen = true;
  f.rows[1].querySelector('[data-review-start]').value = '7';
  f.rows[1].querySelector('[data-review-lyric-text]').value = 'Corrected';
  await f.run('saveReviewChanges(true)');
  assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
  assert.equal(f.state.segments[1].lyric_text, 'Corrected');
});

test('manual save retains the success notice and all-scene editing', async () => {
  const f = fixture(false);
  f.rows[0].querySelector('[data-review-lyric-text]').value = 'One';
  f.rows[2].querySelector('[data-review-lyric-text]').value = 'Three';
  await f.run('save.onclick()');
  assert.equal(f.state.segments[0].lyric_text, 'One');
  assert.equal(f.state.segments[2].lyric_text, 'Three');
  assert.deepEqual(f.messages, ['Line Review Saved']);
  assert.equal(f.backdrop.isConnected, true);
});

for (const direction of ['prevScene', 'nextScene']) {
  test(`${direction} button saves before navigating`, async () => {
    const f = fixture();
    f.context.prevScene = {};
    f.context.nextScene = {};
    f.run(section('      prevScene.onclick = () => {\n        if (targetSceneIndex', '      if (audioPath)'));
    await f.run(`${direction}.onclick()`);
    assert.deepEqual(f.events, ['persist', 'close', direction === 'prevScene' ? 'a' : 'c']);
  });
}

test('Open All Scenes saves the current card and focuses it in the full editor', async () => {
  const f = fixture();
  f.context.openAllButton = {};
  let focus;
  f.context.openLyricReviewModal = options => { focus = options.focusSceneId; };
  f.run(section('      openAllButton.onclick =', '      header.append(heading, openAllButton'));
  await f.run('openAllButton.onclick()');
  assert.deepEqual(f.events, ['persist', 'close']);
  assert.equal(focus, 'b');
});

test('first and last scene boundaries can be edited without missing neighbors', async () => {
  const f = fixture();
  f.context.reviewRows = () => [f.rows[0]];
  f.rows[0].querySelector('[data-review-start]').value = '2';
  await f.run('handleReviewStartEdited(rows[0])');
  f.context.reviewRows = () => [f.rows[2]];
  f.rows[2].querySelector('[data-review-end]').value = '33';
  await f.run('handleReviewEndEdited(rows[2])');
  assert.deepEqual(f.times(), [[2, 10], [10, 20], [20, 33]]);
});

test('lock-rest end edit respects the next scene video lock', async () => {
  const f = fixture();
  f.state.segments[2].locked = true;
  f.rows[1].querySelector('[data-review-end]').value = '23';
  await f.run('handleReviewEndEdited(rows[1])');
  assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
});

for (const retryFails of [false, true]) {
  test(`image-history save retry preserves navigation result (failure=${retryFails})`, async () => {
    const f = fixture();
    let attempts = 0;
    f.context.saveSession = async () => {
      if (++attempts === 1) throw Error('image_history');
      if (retryFails) throw Error('Disk full');
    };
    await f.run('navigateReviewScene({ singleSceneId: "c" })');
    assert.equal(attempts, 2);
    assert.equal(f.backdrop.isConnected, retryFails);
    assert.equal(f.context.save.disabled, false);
  });
}

test('splitting a locked scene leaves the scene and modal intact', async () => {
  const f = fixture();
  f.state.segments[1].locked = true;
  f.run(section('    const splitReviewRowAtPlayhead =', '    const choices = referenceBuilderSubjectChoices();'));
  await f.run('splitReviewRowAtPlayhead(rows[1], state.segments[1])');
  assert.deepEqual(f.times(), [[0, 10], [10, 20], [20, 30]]);
  assert.equal(f.backdrop.isConnected, true);
});

test('splitting an editable scene saves both pieces and opens the second card', async () => {
  const f = fixture();
  let id = 0;
  f.context.newSegment = (start, end) => ({ id: `split-${++id}`, start, end });
  f.context.reviewAudio.currentTime = 15;
  f.run(section('    const copyLyricReviewFields =', '    const choices = referenceBuilderSubjectChoices();'));
  await f.run('splitReviewRowAtPlayhead(rows[1], state.segments[1])');
  assert.deepEqual(f.times(), [[0, 10], [10, 15], [15, 20], [20, 30]]);
  assert.deepEqual(f.events, ['persist', 'close', 'split-2']);
  assert.equal(f.state.fluxReferenceBuilder.scene_map['split-1'], 'location');
  assert.equal(f.state.fluxReferenceBuilder.scene_map['split-2'], 'location');
});
