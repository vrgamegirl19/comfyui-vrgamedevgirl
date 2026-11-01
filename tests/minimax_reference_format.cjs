const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');

test('browser and API use the same literal ownership and spatial clarity contract', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const context = vm.createContext({});
  vm.runInContext(functionSource(readBuilderModule('minimax_prompt.mjs'), 'miniMaxH3LiteralSceneInstruction'), context);
  const py = fs.readFileSync(path.join(__dirname, '../minimax/shot_prompt.py'), 'utf8').replace(/\r\n/g, '\n');
  const block = py.split('def literal_scene_instruction()')[1].split('\n\ndef ')[0];
  const pythonContract = block.split(/\r?\n/).filter(line => /^\s*"/.test(line) && !line.includes('"""'))
    .map(line => JSON.parse(line.trim())).join('');
  assert.equal(context.miniMaxH3LiteralSceneInstruction(), pythonContract);
});

function fixture(audioMode = 'input_audio', pipeline = 'standard', structured = true) {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_h3.mjs'), c);
  Object.assign(c, {
    state: { miniMaxH3Settings: { pipeline }, builderStoryboardDefaults: {}, failOnInvalidPromptFormats: true, useStructuredOutputs: structured },
    items: [
      { kind: 'subject', label: 'woman', description: '', image: { path: 'woman.png' } },
      { kind: 'location', label: 'warehouse', description: 'Rusted machinery and broken windows.', image: { path: 'warehouse.png' } },
    ],
    miniMaxH3SettingsForSegment: () => ({ audio_mode: audioMode }),
    miniMaxSceneDefaults: segment => ({ ...c.state.builderStoryboardDefaults, ...segment?.scene_defaults }),
    miniMaxH3ModeForSegment: () => 'reference_to_video',
    miniMaxH3CutPlanForSegment: () => ({ cut_times_seconds: [] }),
    miniMaxOrderedImageReferenceItemsForSegment: () => c.items,
    miniMaxH3ImageReferencePromptItems: () => [{ kind: 'start_frame', label: 'start' }, ...c.items],
    normalizeFluxReferenceBuilder: () => ({}), logicalExtraSubjectsForScene: () => [],
    miniMaxReferencePurposeText: item => item.kind === 'location' ? 'environment reference' : 'character identity reference',
    miniMaxH3CompactReferenceDescription: text => text,
    miniMaxH3VideoAssignmentLines: () => [],
    assertMiniMaxH3ReferenceCapacity() {},
    omitLyricsForSegment: () => false, isMiniMaxSingerAssignmentMode: () => false,
    escapeRegExp: text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
    parseMiniMaxH3ShotDescriptionPayload: text => [text],
    miniMaxH3OfficialShotBodyFromDescriptions: (segment, descriptions) => `[Shot 1] ${descriptions.join(' ')}`.trim(),
    attachRefmodLabels: text => text,
    miniMaxH3SubjectLabelMapForSegment: () => new Map(),
    enforceCastLabels: text => text,
    storyboardMiniMaxVideoStylePreset: value => (value === '1990s_grunge'
      ? { value, label: '1990s grunge', prompt_guidance: 'Washed-out color and flannel grit throughout.' } : { value: '', label: '', prompt_guidance: '' }),
  });
  const source = readBuilderModule('minimax_prompt.mjs');
  for (const name of ['miniMaxH3CleanSubjectNoun', 'miniMaxH3ReferenceSceneGroundingContract', 'miniMaxH3CompactReferenceShots', 'miniMaxH3CharacterReferenceBindings', 'miniMaxH3OfficialShotPlan',
    'miniMaxH3OfficialReferencePlan', 'miniMaxH3CombinedSubjectPlan', 'miniMaxH3OfficialAudioDefinition',
    'miniMaxH3OfficialSummary', 'miniMaxH3VideoStyleParts', 'miniMaxH3OpeningStyle', 'miniMaxH3DetailedOpening', 'miniMaxH3OfficialSoundscape',
    'miniMaxH3OfficialMusic', 'isRefmodPipelineActive', 'relabelRefmodPrompt',
    'miniMaxH3OfficialReferencePrompt', 'miniMaxH3ReferenceCompositionLeak', 'assertValidMiniMaxH3FinalPrompt',
    'miniMaxH3RefmodRoleText', 'repairRefmodShotDescriptions',
    'assembleMiniMaxH3OfficialPromptFromCreative', 'miniMaxH3PromptCharacterBudget']) {
    vm.runInContext(functionSource(source, name), c);
  }
  return c;
}

const scene = { minimax_h3_video_style: '1990s_grunge' };
const creative = 'The shot opens tight on <Subject 1> (the woman). Her gloved fingers test the zipper beside rusted machinery.';

test('native speech generation requires inline delivery inside dialogue, independently of strict format checking', () => {
  const c = fixture('built_in_audio');
  c.state.failOnInvalidPromptFormats = false;
  c.normalizeVideoType = value => value;
  const speakingScene = { performance_mode: 'speaking' };
  const plain = '[Shot 1] <Subject 1> speaks <d>[English, curious] The sky is turning black.</d> She takes a breath.';
  assert.throws(() => c.assertValidMiniMaxH3FinalPrompt(plain, speakingScene, 'reference_to_video', { requireNativeSpeechDelivery: true }),
    error => error.code === 'MINIMAX_H3_SPEECH_DELIVERY_MISSING');
  const delivered = '[Shot 1] <Subject 1> speaks <d>[English, curious] <breath> The sky is turning <i>black</i>.</d>';
  assert.doesNotThrow(() => c.assertValidMiniMaxH3FinalPrompt(delivered, speakingScene, 'reference_to_video', { requireNativeSpeechDelivery: true }));
  assert.throws(() => c.assertValidMiniMaxH3FinalPrompt(delivered + '\n[Shot 2] <d>[English, curious] We are not ready.</d>', speakingScene, 'reference_to_video', { requireNativeSpeechDelivery: true }),
    error => error.code === 'MINIMAX_H3_SPEECH_DELIVERY_MISSING');
  assert.doesNotThrow(() => c.assertValidMiniMaxH3FinalPrompt(plain, speakingScene, 'reference_to_video'));
  c.miniMaxH3SettingsForSegment = () => ({ audio_mode: 'input_audio' });
  assert.doesNotThrow(() => c.assertValidMiniMaxH3FinalPrompt(plain, speakingScene, 'reference_to_video', { requireNativeSpeechDelivery: true }));
});

test('a continuous-moment scene opens with the continuity paragraph; free cuts do not', () => {
  const c = fixture();
  c.MINIMAX_H3_CONTINUOUS_CUTS_OPENING = 'All shots follow one continuous performance.';
  assert.equal(c.miniMaxH3DetailedOpening(scene),
    'detailed_description:\nThe target video is in a 1990s grunge music-video style: Washed-out color and flannel grit throughout.\n\n');
  c.miniMaxH3CutPlanForSegment = () => ({ cut_times_seconds: [2], continuous: true });
  assert.equal(c.miniMaxH3DetailedOpening(scene),
    'detailed_description:\nThe target video is in a 1990s grunge music-video style: Washed-out color and flannel grit throughout. All shots follow one continuous performance.\n\n');
});

test('the style look is left out where an image sets the look, and custom wording is used as typed', () => {
  const c = fixture();
  c.miniMaxH3ModeForSegment = () => 'image_reference_to_video';
  assert.equal(c.miniMaxH3DetailedOpening(scene), 'detailed_description:\nThe target video is in a 1990s grunge music-video style.\n\n');
  c.miniMaxH3ModeForSegment = () => 'reference_to_video';
  const custom = { minimax_h3_video_style: 'custom', minimax_h3_video_style_custom: 'chalk-drawn dreamworld' };
  assert.equal(c.miniMaxH3DetailedOpening(custom), 'detailed_description:\nThe target video is in a chalk-drawn dreamworld music-video style.\n\n');
});

test('built-in speech preserves inline tags and matches spoken words without duplicating dialogue', () => {
  const c = fixture('built_in_audio');
  const source = readBuilderModule('minimax_prompt.mjs');
  Object.assign(c, { selectedPerformerSubjectsForSegment: () => [], hasEmotionExpressionInput: () => false,
    segmentUsesNoLipSyncPerformance: () => false, isInstrumentalLyricText: () => false,
    isMiniMaxSingerAssignmentMode: () => false });
  for (const name of ['miniMaxH3PunctuatedCueText', 'miniMaxH3CapitalizeCueText', 'normalizeMiniMaxH3DialogueTags', 'miniMaxH3EnsureQuotedLyricInShot']) {
    vm.runInContext(functionSource(source, name), c);
  }
  const draft = '<Subject 1> says <d>[English, curious] <breath> I was <i>not</i> expecting that. <pause></d>';
  assert.equal(c.normalizeMiniMaxH3DialogueTags(draft), draft);
  assert.equal(c.miniMaxH3EnsureQuotedLyricInShot({ lyric_text: 'I was not expecting that.' }, draft, 0, 1, 'reference_to_video'), draft);
  assert.equal(c.miniMaxH3PunctuatedCueText('<whisper>Keep quiet</whisper>'), '<whisper>Keep quiet.</whisper>');
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', draft + ' The camera holds on the speaker.');
  assert.doesNotThrow(() => c.assertValidMiniMaxH3FinalPrompt(prompt, scene, 'reference_to_video'));
});

test('default compact output still binds subjects and environment to reference pictures', () => {
  const c = fixture();
  delete c.state.useStructuredOutputs;
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', creative);
  assert.ok(prompt.startsWith('detailed_description:'));
  assert.match(prompt, /\[Shot 1\] The shot opens tight on <Subject 1>\./);
  assert.doesNotMatch(prompt, /<Subject 1>\s*\(|the woman from <Picture 1>/);
  assert.match(prompt, /environment from <Picture 2>/);
  assert.doesNotMatch(prompt, /<Subject \d+> is|visual authority|used as environment/);
  assert.doesNotMatch(prompt, /subject_definitions:|summary:|retention_analysis:|overall_soundscape:|non_diegetic_music:|first frame|exact opening/);
  const compactBudget = c.miniMaxH3PromptCharacterBudget(scene, 'reference_to_video', 7000);
  assert.equal(compactBudget.fixedChars, c.miniMaxH3OfficialReferencePrompt(scene, 'reference_to_video', '[Shot 1]').length);
  c.state.useStructuredOutputs = true;
  assert.ok(c.miniMaxH3PromptCharacterBudget(scene, 'reference_to_video', 7000).fixedChars > compactBudget.fixedChars);
  c.state.useStructuredOutputs = false;
  assert.equal(c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', creative), prompt);
});

test('compact bindings are inline, preserve existing picture mentions, and never introduce props', () => {
  const c = fixture();
  const subjects = c.miniMaxH3CombinedSubjectPlan(scene, 'reference_to_video').subjects;
  const source = '[Shot 1] The woman from <Picture 1> crosses the warehouse from <Picture 2>.\n\n[Shot 2] <Subject 1> (the woman) turns toward broken windows.';
  const result = c.miniMaxH3CompactReferenceShots(source, subjects);
  assert.ok(result.startsWith(source.split('\n\n')[0]));
  assert.match(result, /\[Shot 2\] <Subject 1> turns/);
  assert.equal((result.match(/<Picture 2>/g) || []).length, 2);
  assert.doesNotMatch(result, /table|coat|zipper|is the woman in/);
});

test('compact output removes every redundant character annotation but preserves speaker IDs', () => {
  const c = fixture();
  const subjects = c.miniMaxH3CombinedSubjectPlan(scene, 'reference_to_video').subjects;
  const text = '[Shot 1] A singing close-up shows <Subject 1> (the woman from <Picture 1>). <Subject 1> (the woman) turns. <Subject 1> (S1) sings.';
  const result = c.miniMaxH3CompactReferenceShots(text, subjects);
  assert.ok(result.startsWith('[Shot 1] A singing close-up shows <Subject 1>. <Subject 1> turns. <Subject 1> (S1) sings.'));
  assert.doesNotMatch(result, /the woman|from <Picture 1>/);
});

test('reference scene guidance distinguishes proposed image text from visible props', () => {
  const c = fixture();
  const contract = c.miniMaxH3ReferenceSceneGroundingContract();
  assert.match(contract, /saved image prompt is a proposed scene idea, not proof/);
  assert.match(contract, /Omit unsupported props/);
  assert.match(contract, /introduce its appearance and physical placement before using it/);
  assert.match(contract, /Do not output subject definitions/);
  assert.match(contract, /STAGING AND ACTION OWNERSHIP/);
  assert.match(contract, /subject label on first mention in each shot/);
  assert.match(contract, /use natural pronouns and possessives/);
  assert.match(contract, /repeat a label when the actor or speaker changes/);
  assert.doesNotMatch(readBuilderModule('minimax_prompt.mjs'), /Never refer to a character only by name, he, or she|never by pronoun/);
  assert.match(contract, /self-contained, physically coherent shot/);
  assert.match(contract, /before any action or camera instruction refers to it/);
  assert.match(contract, /final framing in chronological order/);
  assert.match(contract, /Preserve the beat's intended visual emphasis and endpoint/);
  assert.match(contract, /Make the character the actor/);
  assert.match(contract, /Do not invent gloves or accessories/);
  assert.match(contract, /physical setting around the character/);
  assert.match(contract, /bind the location picture to that setting/);
  assert.match(contract, /Then describe the camera independently/);
  assert.match(contract, /scene's story beat, storyboard details, scene-card directions, and selected camera settings/);
  assert.doesNotMatch(contract, /Do not default|streaks into bokeh/);
  assert.match(contract, /prompt determines opening framing, camera angle, staging, pose, composition, and action/);
});

test('reference mode rejects copying Picture 1 composition even with strict format checks off', () => {
  const c = fixture();
  c.state.failOnInvalidPromptFormats = false;
  const bad = 'Opening at eye level in the composition of <Picture 1>, a tight frame holds <Subject 1>.';
  assert.throws(() => c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', bad),
    error => error.code === 'MINIMAX_H3_REFERENCE_COMPOSITION_LEAK');
  assert.equal(c.miniMaxH3ReferenceCompositionLeak('An eye-level tight shot shows the woman from <Picture 1> in the warehouse from <Picture 2>.'), false);
});

test('legacy scene start flag cannot insert a picture before character references in R2V', () => {
  const c = fixture();
  Object.assign(c, {
    isRefmodPipeline: () => false,
    segmentImageSource: () => ({ path: 'old_scene.png' }),
    miniMaxH3StartFrameCharacterInfluenceForSegment: () => 'full_character',
    miniMaxPromptReferenceItemsForSegment: () => c.items,
    mediaPathKey: path => path,
  });
  vm.runInContext(functionSource(readBuilderModule('minimax_references.mjs'), 'miniMaxOrderedImageReferenceItemsForSegment'), c);
  const legacy = { minimax_h3_use_scene_image_as_start_frame: true };
  const refs = c.miniMaxOrderedImageReferenceItemsForSegment(legacy, 'reference_to_video');
  assert.equal(refs[0].image.path, 'woman.png');
  assert.equal(refs[1].image.path, 'warehouse.png');
  assert.equal(refs.length, 2);
  assert.equal(c.miniMaxOrderedImageReferenceItemsForSegment(legacy, 'image_reference_to_video')[0].kind, 'start_frame');
});

test('reference prompt saves picture bindings and soundscape around the shot prose', () => {
  const c = fixture();
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', creative);
  const sections = ['subject_definitions:', 'summary:', 'retention_analysis:', 'detailed_description:', 'overall_soundscape:', 'non_diegetic_music:'];
  let previous = -1;
  for (const section of sections) {
    assert.ok(prompt.indexOf(section) > previous, section);
    previous = prompt.indexOf(section);
  }
  assert.match(prompt, /<Subject 1> is the woman in <Picture 1>/);
  assert.match(prompt, /<Subject 2> is the environment in <Picture 2>/);
  assert.match(prompt, /<Audio 1>: fully_copy/);
  assert.ok(prompt.includes(`[Shot 1] ${creative}`));
  assert.doesNotMatch(prompt, /first frame|exact opening|keyframe completion/);
});

test('native audio reference prompt defines pictures without inventing an audio input', () => {
  const c = fixture('built_in_audio');
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', creative);
  assert.match(prompt, /subject_definitions:/);
  assert.match(prompt, /MiniMax generates the native audio/);
  assert.doesNotMatch(prompt, /<Audio 1>/);
});

test('Image + Reference keeps the start frame separate from subject numbering', () => {
  const c = fixture();
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'image_reference_to_video', creative);
  assert.match(prompt, /<Picture 1> is the first frame/);
  assert.match(prompt, /<Subject 1> is the woman in <Picture 2>/);
  assert.match(prompt, /keyframe completion/);
});

test('reference character budget reserves the complete saved wrapper', () => {
  const c = fixture();
  const budget = c.miniMaxH3PromptCharacterBudget(scene, 'reference_to_video', 7000);
  const empty = c.miniMaxH3OfficialReferencePrompt(scene, 'reference_to_video', '[Shot 1]');
  assert.equal(budget.fixedChars, empty.length);
  assert.equal(budget.shotDescriptionChars + budget.fixedChars, 7000);
  c.items.push({ kind: 'location', label: 'second set', description: 'Additional set reference.' });
  assert.ok(c.miniMaxH3PromptCharacterBudget(scene, 'reference_to_video', 7000).fixedChars > budget.fixedChars);
});

test('RefMod pipeline keeps its own prompt format', () => {
  const c = fixture('input_audio', 'refmod');
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'reference_to_video', creative);
  assert.ok(prompt.startsWith('detailed_description:'));
  assert.doesNotMatch(prompt, /subject_definitions:/);
});

test('Prompt Options checkbox starts off and saves changes with undo history', () => {
  const events = [];
  let checkbox;
  const c = vm.createContext({
    state: {},
    makeCheckbox(label, checked) {
      const input = { checked, addEventListener(type, callback) { this.change = callback; } };
      checkbox = { label, input, wrapper: {} };
      return checkbox;
    },
    document: { createElement: () => ({ style: {} }) },
    pushHistory: () => events.push('history'),
    autoSaveSessionQuiet: () => events.push('save'),
  });
  const source = readBuilderModule('prompt_creators.mjs');
  const start = source.indexOf('    const structuredOutputs = makeCheckbox(');
  const end = source.indexOf('    const omitLyricsNote', start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(source.slice(start, end), c);
  assert.equal(checkbox.label, 'Use Structured outputs');
  assert.equal(checkbox.input.checked, false);
  checkbox.input.checked = true;
  checkbox.input.change();
  assert.equal(c.state.useStructuredOutputs, true);
  assert.deepEqual(events, ['history', 'save']);
  checkbox.input.checked = false;
  checkbox.input.change();
  assert.equal(c.state.useStructuredOutputs, false);
});


test('compact image/reference binds only characters used by the shot with exact frame offsets', () => {
  const c = fixture('input_audio', 'standard', false);
  c.items = Array.from({ length: 4 }, (_, i) => ({ kind: 'subject', label: `character ${i + 1}` }));
  let prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'image_reference_to_video', '<Subject 1> walks beside the stage and turns toward <Subject 3> as the camera moves slowly forward.');
  assert.match(prompt, /Use <Picture 2> for <Subject 1>'s character identity/);
  assert.match(prompt, /Use <Picture 4> for <Subject 3>'s character identity/);
  assert.doesNotMatch(prompt, /<Subject 2>|<Subject 4>|subject_definitions:|retention_analysis:/);
  c.miniMaxH3ImageReferencePromptItems = () => [{ kind: 'start_frame' }, { kind: 'end_frame' }, ...c.items];
  prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'image_reference_to_video', '<Subject 1> turns toward the stage as the camera moves slowly forward.');
  assert.match(prompt, /Use <Picture 3> for <Subject 1>/);
  c.miniMaxReferencePurposeText = () => 'face identity and hair reference only; do not copy clothing';
  prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'image_reference_to_video', '<Subject 1> turns toward the stage as the camera moves slowly forward.');
  assert.match(prompt, /Use <Picture 3> for <Subject 1>'s face identity and hair\./);
  assert.doesNotMatch(prompt, /clothing|body-proportion/);
});

test('image/reference sends ordered images to both local and external vision writers', () => {
  const c = fixture();
  const images = [{ path: 'start.png' }, { path: 'character.png' }];
  c.miniMaxH3PromptVisionImages = () => images;
  vm.runInContext(functionSource(readBuilderModule('minimax_prompt.mjs'), 'miniMaxH3PromptVisionImagesForRunner'), c);
  for (const runner of ['builtin', 'lm_studio', 'llm_api', 'own_server']) {
    c.state.textGemmaRunner = runner;
    assert.equal(c.miniMaxH3PromptVisionImagesForRunner(scene, 'image_reference_to_video'), images);
  }
  c.state.textGemmaRunner = 'builtin';
  assert.equal(c.miniMaxH3PromptVisionImagesForRunner(scene, 'reference_to_video').length, 0);
});

test('compact image/reference budget reserves all potential character assignments', () => {
  const c = fixture('input_audio', 'standard', false);
  const budget = c.miniMaxH3PromptCharacterBudget(scene, 'image_reference_to_video', 7000);
  const wrapper = c.miniMaxH3OfficialReferencePrompt(scene, 'image_reference_to_video', '[Shot 1]');
  const reserved = c.miniMaxH3CharacterReferenceBindings('', c.miniMaxH3CombinedSubjectPlan(scene, 'image_reference_to_video').subjects, true);
  assert.equal(budget.fixedChars, wrapper.length + reserved.length + 2);
});

test('vision attachments keep start/end frames and four character pictures in renderer order', () => {
  const c = fixture();
  c.items = Array.from({ length: 4 }, (_, i) => ({ kind: 'subject', label: `character ${i + 1}`, image: { path: `character${i + 1}.png` } }));
  c.segmentImageSource = () => ({ path: 'start.png' });
  c.firstLastFrameEndImageSource = () => ({});
  c.mediaPathKey = value => value;
  const source = readBuilderModule('minimax_prompt.mjs');
  for (const name of ['miniMaxH3ImageReferencePromptItems', 'miniMaxH3PromptVisionImages', 'miniMaxH3PromptVisionImagesForRunner']) {
    vm.runInContext(functionSource(source, name), c);
  }
  c.state.textGemmaRunner = 'builtin';
  let images = c.miniMaxH3PromptVisionImagesForRunner(scene, 'image_reference_to_video');
  assert.deepEqual(Array.from(images, image => image.path), ['start.png', 'character1.png', 'character2.png', 'character3.png', 'character4.png']);
  c.firstLastFrameEndImageSource = () => ({ path: 'end.png' });
  images = c.miniMaxH3PromptVisionImagesForRunner(scene, 'image_reference_to_video');
  assert.deepEqual(Array.from(images, image => image.path), ['start.png', 'end.png', 'character1.png', 'character2.png', 'character3.png', 'character4.png']);
});

test('full character sheets preserve outfit and footwear; all panels are the same person', () => {
  const context = vm.createContext({
    miniMaxH3ModeForSegment: () => 'image_reference_to_video',
    miniMaxH3StartFrameCharacterInfluenceForSegment: () => 'full_character',
  });
  vm.runInContext(functionSource(readBuilderModule('minimax_references.mjs'), 'miniMaxReferencePurposeText'), context);
  const purpose = context.miniMaxReferencePurposeText({ kind: 'subject', reference_image_type: 'multi_view' }, {});
  assert.match(purpose, /complete outfit, footwear, accessories/);
  assert.match(purpose, /all panels depict the same character, not multiple people/);
  assert.match(purpose, /outside the start-frame crop/);
  const c = fixture('input_audio', 'standard', false);
  c.miniMaxReferencePurposeText = () => purpose;
  const prompt = c.assembleMiniMaxH3OfficialPromptFromCreative(scene, 'image_reference_to_video', '<Subject 1> walks beside the stage as the camera pulls back to show her full figure.');
  assert.match(prompt, /Use <Picture 2> for <Subject 1>'s character identity, face, hair, complete outfit, footwear, accessories, and body-proportion/);
});

test('a RefMod object is used by its character, a prop is part of the set, a creature is not a thing', () => {
  const c = fixture();
  const pictureItems = [
    { refmod: { category: 'character', card_id: 'a' } },
    { refmod: { category: 'character', card_id: 'w', reference_type: 'creature' } },
    { refmod: { category: 'object', reference_type: 'object', wears: 'w' } },
    { refmod: { category: 'object', reference_type: 'prop' } },
  ];
  const role = (index, name) => c.miniMaxH3RefmodRoleText(pictureItems[index], {
    subjectLabel: `<Subject ${index + 1}>`, pictureLabel: `<Picture ${index + 1}>`, pictureItems, shotList: '[Shot 1]', name,
  });
  const keys = role(2, 'car keys');
  assert.equal(keys.role, 'object');
  assert.match(keys.definition, /It is the car keys: show <Subject 2> holding and using it the way it is normally used/);
  const lamp = role(3, 'floor lamp');
  assert.equal(lamp.role, 'prop');
  assert.match(lamp.definition, /It is the floor lamp, part of the set: place it where it naturally belongs/);
  assert.equal(role(1, 'grey wolf'), null, 'a creature is a cast member, not a thing');
});
