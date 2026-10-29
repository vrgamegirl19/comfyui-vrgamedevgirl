const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');

// Subjects: a project holds as many as it needs. Only the blank cards one typed count adds are bounded.
function referenceDataContext() {
  const c = vm.createContext({ console, document: { createElement: () => ({ style: {}, append() {} }) } });
  c.normalizeMiniMaxH3Voice = (value) => value || {};
  vm.runInContext(readBuilderModule('elevenlabs_voice_design.mjs'), c);
  vm.runInContext(readBuilderModule('elevenlabs.mjs'), c);
  vm.runInContext(functionSource(readBuilderModule('prompt_text.mjs'), 'defaultFluxReferenceBuilder'), c);
  vm.runInContext(readBuilderModule('reference_data.mjs'), c);
  return c;
}

test('a project keeps every subject it has, more than twelve included', () => {
  const c = referenceDataContext();
  const subjects = Array.from({ length: 15 }, (_, n) => ({ id: `s${n}`, name: `Person ${n}`, description: 'x' }));
  const refs = c.normalizeFluxReferenceBuilder({ subjects });
  assert.equal(refs.subjects.length, 15);
  assert.equal(refs.subject_count, 15);
  const again = c.normalizeFluxReferenceBuilder(JSON.parse(JSON.stringify(refs)));
  assert.equal(again.subjects.length, 15);
  assert.equal(again.subjects[14].name, 'Person 14');
});

test('typing a subject count adds blank cards, and one entry adds at most 100', () => {
  const c = referenceDataContext();
  assert.equal(c.normalizeFluxReferenceBuilder({ subject_count: 20 }).subjects.length, 20);
  assert.equal(c.normalizeFluxReferenceBuilder({ subject_count: 99999 }).subjects.length, vm.runInContext('MAX_BLANK_SUBJECTS_PER_ENTRY', c));
});

// Scenes: at most 8 Reference Builder images or RefMods are sent, whatever the scene maps.
function referencesContext({ pipeline = 'standard', continuity = false, cards, library = [] }) {
  const c = vm.createContext({ console });
  const libraryRequest = Promise.resolve(library);
  const calls = { saves: [], status: 0, buttons: 0 };
  Object.assign(c, {
    loadRefmodLibrary: () => libraryRequest,
    makeEditorImageUrl: () => '', escapeHtml: (v) => String(v), makeButton: () => ({}), toast: () => {},
    rtvReferenceImagePayload: (v) => v, hasReferenceImage: (image) => Boolean(image && (image.path || image.data)),
    normalizeMiniMaxH3Mode: (mode) => mode, normalizeMiniMaxH3Pipeline: (value) => value,
    normalizeProjectVideoEngine: (value) => value, expandSubjectReferencesForRender: (_refs, mapped) => mapped,
    normalizeFluxReferenceBuilder: (value) => value, mediaPathKey: (path) => path,
  });
  vm.runInContext(readBuilderModule('refmod_labels.mjs'), c);
  vm.runInContext(functionSource(readBuilderModule('refmod_card.mjs'), 'syncRefmodCards'), c);
  vm.runInContext(readBuilderModule('minimax_references.mjs'), c);
  const state = {
    projectVideoEngine: 'minimax_h3', miniMaxH3Settings: { pipeline },
    fluxReferenceBuilder: { use_subject_reference: true, subject_count: cards.length, subjects: cards, extra_subjects: [], locations: [], ingredients_sheets: [], scene_map: {} },
  };
  const segment = { id: 'seg_1', minimax_h3_reference_keys: pipeline === 'refmod' ? undefined : cards.map((card) => `subject:${card.id}`) };
  const factory = c.createMiniMaxReferences({
    activeSegment: () => segment, autoSaveSessionQuiet: async (reason) => { calls.saves.push(reason); }, currentVideoMode: () => 'rtv',
    firstLastFrameResolvedEndImageSource: () => ({}), logicalExtraSubjectsForScene: () => [],
    logicalReferenceSubjects: (refs) => refs.subjects, logicalSubjectIdsForScene: () => cards.map((card) => card.id),
    miniMaxH3ContinuityReferenceReserved: () => continuity, miniMaxH3ImageReferencePromptItems: () => [],
    miniMaxH3ModeForSegment: () => 'reference_to_video', miniMaxH3SceneImageIsPromptInspiration: () => false,
    miniMaxH3StartFrameCharacterInfluenceForSegment: () => 'full', openReferenceBuilderTargetChooser: () => {},
    pushHistory: () => {}, requireActiveSegment: () => segment, rtvReferenceBehaviorForSegment: () => '',
    rtvSceneImageAnchorPayload: () => null, sceneDisplayName: () => 'Scene 1', sceneReferenceMapValue: () => '',
    segmentImageSource: () => ({}), segmentIndexInfo: () => ({ index: 0 }), state, syncMiniMaxReferenceButtons: () => { calls.buttons += 1; },
    updateMiniMaxPromptCharacterStatus: () => { calls.status += 1; },
  });
  return { ...factory, segment, state, calls, limit: vm.runInContext('MAX_SCENE_REFERENCES', c) };
}

const imageCards = (count) => Array.from({ length: count }, (_, n) => ({
  id: `p${n}`, name: `Person ${n}`, description: 'x', reference_type: 'character', image: { path: `C:/refs/p${n}.png` },
}));
const refmodCards = (count, tokens) => Array.from({ length: count }, (_, n) => ({
  id: `p${n}`, name: `Person ${n}`, description: 'x', reference_type: 'character', source: 'refmod', image: { path: '' },
  refmod: { name: `identity/p${n}`, kind: 'video', tokens, strength: 1 },
}));

test('a scene uses at most 8 reference images, and the ninth is reported', () => {
  const eight = referencesContext({ cards: imageCards(8) });
  assert.equal(eight.limit, 8);
  const ok = eight.miniMaxH3ReferenceCapacityStatus(eight.segment);
  assert.equal(ok.count, 8);
  assert.equal(ok.limit, 8);
  assert.equal(ok.overflow, 0);
  assert.doesNotThrow(() => eight.assertMiniMaxH3ReferenceCapacity(eight.segment));

  const nine = referencesContext({ cards: imageCards(9) });
  const over = nine.miniMaxH3ReferenceCapacityStatus(nine.segment);
  assert.equal(over.overflow, 1);
  assert.throws(() => nine.assertMiniMaxH3ReferenceCapacity(nine.segment), /uses 9 reference images, but a scene can use at most 8\. Unmap or uncheck 1 of them/);
  assert.equal(nine.miniMaxReferenceKeysForSegment(nine.segment).length, 8);
});

test('many mapped subjects are all kept in the project and only eight are sent', () => {
  const c = referencesContext({ cards: imageCards(20) });
  assert.equal(c.miniMaxDesiredReferenceKeysForSegment(c.segment).length, 20);
  assert.equal(c.miniMaxReferenceKeysForSegment(c.segment).length, 8);
  assert.equal(c.miniMaxH3ReferenceCapacityStatus(c.segment).overflow, 12);
});

test('the continuity frame shares the ninth image with the references', () => {
  const c = referencesContext({ cards: imageCards(8), continuity: true });
  const status = c.miniMaxH3ReferenceCapacityStatus(c.segment);
  assert.equal(status.count, 9);
  assert.equal(status.limit, 9);
  assert.equal(status.overflow, 0);
});

test('a scene uses at most 8 RefMods and the token total only warns', () => {
  const eight = referencesContext({ pipeline: 'refmod', cards: refmodCards(8, 1100) });
  const status = eight.miniMaxH3ReferenceCapacityStatus(eight.segment);
  assert.equal(status.count, 8);
  assert.equal(status.limit, 8);
  assert.equal(status.overflow, 0);
  assert.match(status.tokenText, /8,800 tokens\. Over 8,192: this scene may not render or may crash/);
  assert.doesNotThrow(() => eight.assertMiniMaxH3ReferenceCapacity(eight.segment));

  const nine = referencesContext({ pipeline: 'refmod', cards: refmodCards(9, 100) });
  assert.equal(nine.miniMaxH3ReferenceCapacityStatus(nine.segment).overflow, 1);
  assert.throws(() => nine.assertMiniMaxH3ReferenceCapacity(nine.segment), /uses 9 RefMods, but a scene can use at most 8/);
});

test('cards that kept old RefMod numbers take the saved ones once, and the false warning goes', async () => {
  // Eight cards picked when each RefMod had 1,100 tokens. The RefMods were saved again with 900.
  const library = Array.from({ length: 8 }, (_, n) => ({ name: `identity/p${n}`, kind: 'video', tokens: 900, frames: 5 }));
  const c = referencesContext({ pipeline: 'refmod', cards: refmodCards(8, 1100), library });
  assert.match(c.miniMaxH3ReferenceCapacityStatus(c.segment).tokenText, /8,800 tokens\. Over 8,192/);

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(c.miniMaxH3ReferenceCapacityStatus(c.segment).tokenText, 'RefMods: 7,200 tokens.');
  const card = c.state.fluxReferenceBuilder.subjects[0].refmod;
  assert.deepEqual(card, { name: 'identity/p0', kind: 'video', tokens: 900, frames: 5, strength: 1 });
  assert.equal(c.calls.status, 1, 'the prompt status line is drawn again');
  assert.equal(c.calls.buttons, 1, 'the reference buttons are drawn again');
  assert.deepEqual(c.calls.saves, ['RefMod token counts'], 'the project is saved');

  // The same library read is not applied twice.
  c.miniMaxH3ReferenceCapacityStatus(c.segment);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(c.calls.saves, ['RefMod token counts']);
});

test('the standard pipeline leaves cards and the library alone', async () => {
  const c = referencesContext({ cards: imageCards(2), library: [{ name: 'identity/p0', kind: 'video', tokens: 1, frames: 1 }] });
  c.miniMaxH3ReferenceCapacityStatus(c.segment);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(c.calls, { saves: [], status: 0, buttons: 0 });
});
