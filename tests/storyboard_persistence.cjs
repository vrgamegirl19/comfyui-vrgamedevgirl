const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { spawnSync } = require("node:child_process");
const test = require("node:test");

const root = path.join(__dirname, "..");
const project = process.env.STORYBOARD_TEST_PROJECT;
const clone = (value) => JSON.parse(JSON.stringify(value));

function fixture(state, parentSave = async () => {}) {
  const messages = [];
  const context = vm.createContext({ console, Event: class {}, createToast: (...args) => messages.push(args),
    postJson: async (route, payload) => {
      const action = route.endsWith("/save") ? "save" : "load";
      const result = spawnSync(process.env.STORYBOARD_TEST_PYTHON,
        [path.join(__dirname, "test_storyboard_persistence.py"), action],
        { input: JSON.stringify(payload), encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr);
      return { storyboard: JSON.parse(result.stdout) };
    },
  });
  for (const file of ["music_video_builder/refmod_labels.mjs", "storyboard_builder/script_import.mjs",
    "storyboard_builder/video_style.mjs", "storyboard_builder/shot_presets.mjs",
    "storyboard_builder/scenes.mjs", "storyboard_builder/references.mjs", "storyboard_builder/persistence.mjs"]) {
    const source = fs.readFileSync(path.join(root, "web", file), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context, { filename: file });
  }
  const noop = () => {};
  const controls = {};
  const controlNames = ["adjacentLyricContextInput", "cameraFlowSelect", "cameraSpeedInput", "characterSpeedInput", "storyArcDetailSelect",
    "consistencyInput", "cutFrequencyInput", "facialCustomInput", "facialSelect", "fxCustomInput", "fxSelect",
    "imageAestheticSelect", "imageCustomStyleInput", "imageShotSelect", "imageWorldStyleSelect",
    "keepGemmaLoadedInput", "lyricStoryStrengthInput", "overallStoryIdeaInput", "performanceSelect", "shortFilmPlanningModeSelect",
    "songStoryBriefInput", "storyLayerEnabledInput", "temporalEffectCustomInput", "temporalEffectSelect",
    "temporalEnvironmentInput", "temporalExtrasInput", "temporalIntensityInput", "temporalProtectedCustomInput",
    "temporalProtectedSelect", "userStoryArcInput", "videoStyleCustomInput", "videoStyleSelect"];
  for (const name of controlNames) controls[name] = { value: "", checked: false, dispatchEvent: noop };
  Object.assign(controls.cameraFlowSelect, { value: "balanced" });
  Object.assign(controls.imageShotSelect, { value: "intimate" });
  Object.assign(controls.cameraSpeedInput, { value: "0" });
  Object.assign(controls.characterSpeedInput, { value: "0" });
  Object.assign(controls.cutFrequencyInput, { value: "0" });
  Object.assign(controls.shortFilmPlanningModeSelect, { value: "fully_custom" });
  Object.assign(controls.imageWorldStyleSelect, { value: "custom" });
  Object.assign(controls.imageCustomStyleInput, { value: "Ink on glass" });
  Object.assign(controls.fxSelect, { value: "custom" });
  Object.assign(controls.fxCustomInput, { value: '{"grain":0.3}' });
  Object.assign(controls.temporalIntensityInput, { value: "3" });
  Object.assign(controls.temporalProtectedSelect, { value: "all_referenced" });
  Object.assign(controls.lyricStoryStrengthInput, { value: "2" });
  controls.adjacentLyricContextInput.checked = true;
  const deps = { ...controls, state, payload: { onFocusedSave: parentSave },
    incomingProjectVideoEngine: "minimax_h3", openingMode: "image_to_video_prep", save: { disabled: false },
    facialPerformancePresets: [{ value: "" }], performanceStylePresets: [{ value: "" }],
    imageAestheticPresets: [{ value: "" }], imageShotFlowPresets: { intimate: {} },
    enforceStoryboardVideoFacialRequirements: (prompt) => prompt,
    syncStoryLayerFromInputs: () => {
      state.storyLayer = { enabled: controls.storyLayerEnabledInput.checked,
        overall_story_idea: controls.overallStoryIdeaInput.value, user_story_arc: controls.userStoryArcInput.value,
        song_story_brief: controls.songStoryBriefInput.value, lyric_story_strength: Number(controls.lyricStoryStrengthInput.value),
        image_world_style: controls.imageWorldStyleSelect.value, image_custom_style_direction: controls.imageCustomStyleInput.value };
    },
    storyboardDefaultsPayload: () => vm.runInContext("slimStoryboardForRequest", context)(state),
    setMode: (mode) => { state.mode = mode; },
  };
  for (const name of ["absorbSceneReferencesIntoCatalog", "renderTable", "syncReferenceMappingsToVideoCreator",
    "syncLyricStoryStrengthLabel", "refreshCameraFlowInfo", "refreshCameraSpeedInfo", "refreshCharacterSpeedInfo",
    "refreshConsistencyInfo", "refreshCutFrequencyInfo", "refreshFacialInfo", "refreshFxInfo", "refreshImageAestheticInfo",
    "refreshImageShotInfo", "refreshImageWorldStyleInfo", "refreshPerformanceInfo", "refreshTemporalEffectInfo", "refreshVideoStyleInfo"])
    deps[name] = noop;
  const api = vm.runInContext("createStoryboardPersistence", context)(deps);
  return { ...api, controls, messages, context, deps };
}

function stateFor(folder) {
  return { projectFolder: folder, projectVideoEngine: "minimax_h3", mode: "image_to_video_prep",
    performanceMode: "speaking", referenceBuilder: { subjects: [{ id: "subject1", name: "Singer",
      image: { data: "data:image/png;base64,A", name: "Portrait" }, source: "refmod",
      refmod: { name: "Singer", folder: "Singer", type: "character", kind: "image", tokens: 8, frames: 1, strength: 0.6 } }], locations: [] },
    customCameraFlowSequence: [{ shot: "Close-up", camera: "Pan left" }],
    storyLayer: {}, cameraMotionSpeed: 9, characterMotionSpeed: 9, globalConsistencyPhrase: "old phrase",
    scenes: [{ id: "scene1", scene_number: 1, label: "Edited label", lyrics: "Edited dialogue",
      project_video_engine: "minimax_h3", video_prompt_type: "flf", story_beat: "", prompt_summary: "",
      motion_summary: "Saved motion notes", audio_direction: "Saved sound", continuity: "Saved continuity",
      flf_start_state: "First", flf_transformation: "Turn", flf_end_state: "Last", flf_carry_forward: "Carry",
      image_prompt: "A glass room", video_prompt: "", minimax_h3_pass2_prompt: "Pass two",
      image_name: "Approved frame", image_data: "data:image/png;base64,A", lyric_singers: ["Singer"],
      lyric_no_lip_sync: true, lyric_instrumental: false, no_character_present: false,
      lyric_cue_map: [{ text: "Edited dialogue", start: 0, end: 1 }], lyric_shot_word_timing_enabled: true }],
  };
}

test("Save captures controls, writes disk, awaits the project, and restores edits", async () => {
  const state = stateFor(project);
  let release;
  let parentScenes;
  const pending = new Promise((resolve) => { release = resolve; });
  const saving = fixture(state, async (updates) => { parentScenes = clone(updates.scenes); await pending; });
  const promise = saving.saveStoryboard({ throwOnError: true });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.saving, true);
  assert.equal(saving.deps.save.disabled, true);
  assert.equal(saving.messages.length, 0);
  release();
  await promise;
  assert.equal(state.saving, false);
  const saved = JSON.parse(fs.readFileSync(path.join(project, "storyboard", "storyboard.json"), "utf8"));
  assert.equal(saved.camera_motion_speed, 0);
  assert.equal(saved.character_motion_speed, 0);
  assert.equal(saved.global_consistency_phrase, "");
  assert.equal(saved.send_adjacent_lyric_context, true);
  assert.equal(saved.prompt_summary, undefined);
  assert.equal(saved.scenes[0].prompt_summary, "");
  assert.equal(saved.fx_custom_json, '{"grain":0.3}');
  assert.equal(saved.story_layer.image_world_style, "custom");
  assert.equal(saved.story_layer.image_custom_style_direction, "Ink on glass");
  assert.equal(saved.reference_builder.subjects[0].image.data, "data:image/png;base64,A");
  assert.equal(saved.reference_builder.subjects[0].refmod.strength, 0.6);
  assert.deepEqual(saved.custom_camera_flow_sequence, [{ shot: "Close-up", camera: "Pan left" }]);
  const reopened = stateFor(project);
  reopened.scenes = parentScenes.map((scene) => ({ ...scene, prompt_summary: "Derived timeline summary" }));
  const loading = fixture(reopened);
  assert.equal(await loading.loadExisting(), true, JSON.stringify(loading.messages));
  for (const field of ["story_beat", "prompt_summary", "motion_summary", "audio_direction", "continuity",
    "flf_start_state", "flf_transformation", "flf_end_state", "flf_carry_forward", "minimax_h3_pass2_prompt",
    "image_name", "image_data", "lyrics", "lyric_no_lip_sync", "lyric_shot_word_timing_enabled"])
    assert.deepEqual(reopened.scenes[0][field], saved.scenes[0][field], field);
  assert.equal(loading.controls.imageWorldStyleSelect.value, "custom");
  assert.equal(loading.controls.imageCustomStyleInput.value, "Ink on glass");
  assert.equal(reopened.globalConsistencyPhrase, "");
  assert.equal(reopened.cameraMotionSpeed, 0);
  assert.equal(loading.controls.adjacentLyricContextInput.checked, true);
});

test("Explicit saves apply live scene edits and cleared prompts; exports preserve live lyrics", () => {
  const source = fs.readFileSync(path.join(root, "web/music_video_builder/storyboard_bridge.mjs"), "utf8");
  const start = source.indexOf("const applyStoryboardPrompts =");
  const end = source.indexOf("const findStoryboardSegment =", start);
  const segment = { id: "scene1", lyric_text: "Live dialogue", minimax_h3_prompt: "Old video", t2i_prompt: "Old image" };
  const noop = () => {};
  const context = vm.createContext({ state: { projectVideoEngine: "minimax_h3" },
    allEditableSegments: () => [segment], normalizeMiniMaxSpeakerAssignments: (value) => value,
    syncMiniMaxSpeakerAssignmentLegacyFields: noop, normalizeProjectVideoEngine: (value) => value,
    normalizeVideoPromptOrigin: (value) => value, normalizeBuilderStoryLayer: (value) => value,
    normalizeBuilderStoryboardDefaults: (value) => value, applyMiniMaxH3NativeVoiceBlock: (value) => value,
    ensureAllSegmentRuntimeFields: noop, ensureSegmentRuntimeFields: noop, syncInspector: noop, render: noop, toast: noop,
    autoSaveSessionQuiet: noop, setSegmentPromptForEdit: (target, kind, prompt) => { target[`${kind}_prompt`] = prompt; },
  });
  vm.runInContext(source.slice(start, end), context);
  const apply = vm.runInContext("applyStoryboardPrompts", context);
  apply({ scenes: [{ id: "scene1", lyrics: "Stale export dialogue", image_prompt: "New image" }] });
  assert.equal(segment.lyric_text, "Live dialogue");
  apply({ scenes: [{ id: "scene1", lyrics: "", image_prompt: "", video_prompt: "", motion_summary: "Edited motion",
    lyric_section: "", speaker_assignments: [], flf_end_state: "New end", timeline_note: "Edited director note" }] }, { saveSceneEdits: true });
  assert.equal(segment.lyric_text, "");
  assert.equal(segment.t2i_prompt, "");
  assert.equal(segment.minimax_h3_prompt, "");
  assert.equal(segment.video_notes, "Edited motion");
  assert.equal(segment.flf_end_state, "New end");
  assert.equal(segment.timeline_note, "Edited director note");
});

test("New projects retain current settings when no storyboard file exists", async () => {
  const state = stateFor(path.join(project, "new"));
  const api = fixture(state);
  assert.equal(await api.loadExisting(), true, JSON.stringify(api.messages));
  assert.equal(state.globalConsistencyPhrase, "old phrase");
  assert.equal(state.scenes[0].lyrics, "Edited dialogue");
});

test("Timeline prompt and note edits replace stale storyboard values on reopen, including clearing", async () => {
  const folder = path.join(project, "timeline-edits");
  const saved = stateFor(folder);
  Object.assign(saved.scenes[0], { timeline_note: "Old director note", notes: "Old planning notes",
    motion_summary: "Old video note", video_prompt: "Old prompt" });
  await fixture(saved).saveStoryboard({ throwOnError: true });
  for (const text of ["New timeline edit", ""]) {
    const current = stateFor(folder);
    Object.assign(current.scenes[0], { timeline_note: text, notes: text, motion_summary: text,
      video_prompt: text, image_prompt: text });
    const api = fixture(current);
    assert.equal(await api.loadExisting(), true);
    for (const field of ["timeline_note", "notes", "motion_summary", "video_prompt", "image_prompt"])
      assert.equal(current.scenes[0][field], text, field);
  }
});

test("Main timeline payload keeps canonical prompts and notes instead of old aliases", () => {
  const api = fixture(stateFor(project));
  const convert = vm.runInContext("scenesFromBuilderPayload", api.context);
  const scenes = convert({ scenes: [{ id: "scene1", video_prompt: "", i2v_prompt: "Old prompt",
    image_prompt: "Manual image prompt", motion_summary: "", video_notes: "Old video note",
    timeline_note: "Director note", notes: "Planning notes" }] });
  assert.equal(scenes[0].video_prompt, "");
  assert.equal(scenes[0].image_prompt, "Manual image prompt");
  assert.equal(scenes[0].motion_summary, "");
  assert.equal(scenes[0].timeline_note, "Director note");
  assert.equal(scenes[0].notes, "Planning notes");
});

test("Image and video LLM payloads include every normalized card field and all note types", () => {
  for (const mode of ["storyboard_prompts", "image_to_video_prep"]) {
    const state = stateFor(project);
    state.mode = mode;
    Object.assign(state.scenes[0], { timeline_note: "Director-note sentinel", notes: "Planning-note sentinel",
      motion_summary: "Motion-note sentinel", audio_direction: "Sound sentinel", continuity: "Continuity sentinel",
      camera_motion: "Camera sentinel", character_motion: "Action sentinel", include_microphone: true,
      facial_performance_custom: "Expression sentinel" });
    const api = fixture(state);
    for (const name of ["performance_presets.mjs", "gpt_payload.mjs"]) {
      const source = fs.readFileSync(path.join(root, "web/storyboard_builder", name), "utf8")
        .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
      vm.runInContext(source, api.context);
    }
    const payload = vm.runInContext("storyboardGptPayload", api.context)(state);
    const actual = clone(payload.scenes[0].scene_card);
    const normalized = clone(vm.runInContext("normalizeScene", api.context)(state.scenes[0]));
    for (const [field, value] of Object.entries(normalized)) {
      if (["image_data", "subject_refs", "location_ref"].includes(field)) continue;
      assert.deepEqual(actual[field], value, `${mode}: ${field}`);
    }
    assert.equal(payload.scenes[0].director_note, "Director-note sentinel");
    assert.equal(actual.has_inline_image, true);
    assert.equal(JSON.stringify(actual).includes("base64"), false);
    assert.match(payload.scenes[0].scene_card_instruction, /every populated field/);
  }
});

test("Story Arc request and GPT export use current timed Timeline Notes", async () => {
  const state = stateFor(project);
  state.referenceBuilder.locations = [{ id: "room", name: "Glass room" }];
  state.scenes[0].location_ref = state.referenceBuilder.locations[0];
  state.timelineMarkers = [{ start: 0, note: "Old note" }];
  let live = [{ start: 5, end: 10, note: "Reveal the conflict" },
    { start: 2, end: null, note: "Introduce the prop" }, { start: 0, note: " " }];
  state.getTimelineMarkers = () => live;
  const api = fixture(state);
  for (const name of ["performance_presets.mjs", "gpt_payload.mjs", "story_workflow.mjs", "story_layer.mjs"]) {
    vm.runInContext(fs.readFileSync(path.join(root, "web/storyboard_builder", name), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, ""), api.context);
  }
  const exported = vm.runInContext("storyLayerGptPayload", api.context)(state);
  assert.deepEqual(clone(exported.project_inputs.timeline_markers.map(note => note.start)), [2, 5]);
  assert.equal(exported.project_inputs.timeline_markers[0].end, null);
  let request;
  Object.assign(api.context, { createStoryboardProgressWindow: () => ({ set() {}, close() {} }),
    postJson: async (route, payload) => { request = payload; return { story_arc: "New arc" }; } });
  const control = { value: "", checked: true };
  const layer = vm.runInContext("createStoryLayer", api.context)({ state,
    imageCustomStyleInput: control, imageWorldStyleSelect: control, lyricStoryStrengthInput: { value: "7" },
    overallStoryIdeaInput: control, songStoryBriefInput: control, userStoryArcInput: { value: "" },
    storyLayerEnabledInput: control, refreshSetupPanelSummaries() {}, promptRunnerName: () => "Test LLM" });
  assert.equal(await layer.createStoryArcWithGemma(), "New arc");
  assert.equal(request.timeline_markers[1].note, "Reveal the conflict");
  live = [];
  assert.deepEqual(clone(vm.runInContext("storyLayerGptPayload", api.context)(state).project_inputs.timeline_markers), []);
});

test("Builder video prompting includes complete card context for both LTX and MiniMax", () => {
  const api = fixture(stateFor(project));
  const source = fs.readFileSync(path.join(root, "web/music_video_builder/storyboard_bridge.mjs"), "utf8");
  const start = source.indexOf("const storyboardVideoExtraNotes =");
  const end = source.indexOf("const ensureStoryboardRequiredStartingShot =", start);
  Object.assign(api.context, { normalizeMiniMaxShortFilmPlanningMode: () => "guided_film",
    normalizeProjectVideoEngine: value => value, miniMaxH3CutPlanForSegment: () => null });
  vm.runInContext(source.slice(start, end), api.context);
  const notes = vm.runInContext("storyboardVideoExtraNotes", api.context);
  for (const engine of ["ltx", "minimax_h3"]) {
    const scene = { project_video_engine: engine, timeline_note: "Director sentinel", notes: "Planning sentinel",
      audio_direction: "Sound sentinel", motion_summary: "Motion sentinel", include_microphone: true };
    const text = notes(scene, { scenes: [{ project_video_engine: engine }] });
    for (const value of ["Director sentinel", "Planning sentinel", "Sound sentinel", "Motion sentinel"])
      assert.ok(text.includes(value), `${engine}: ${value}`);
    assert.ok(text.includes('"include_microphone": true'));
  }
});

test("MiniMax creative input preserves complete long storyboard context", () => {
  const { functionSource, readBuilderModule } = require("./builder_source.cjs");
  const context = vm.createContext({
    state: { videoType: "no_lip_sync", builderStoryboardDefaults: {} },
    miniMaxH3SettingsForSegment: () => ({ audio_mode: "input_audio", aspect_ratio: "16:9" }),
    segmentUsesNoLipSyncPerformance: () => true,
    miniMaxDialogueAssignmentsForSegment: () => [],
    isInstrumentalLyricText: () => false,
    flattenLyricForPrompt: value => value || "",
    normalizeVideoType: value => value,
    miniMaxH3CutPlanForSegment: () => ({}),
    miniMaxH3OfficialShotPlan: () => [{ timecode: 0 }],
    miniMaxH3ModeLabel: value => value,
    miniMaxH3PromptCharacterBudget: () => ({ shotDescriptionChars: 6500 }),
    miniMaxH3LifeMovementBank: () => ["blink"],
    miniMaxH3MotionEnergyText: () => "",
    selectedCastCoverageContract: () => "",
    miniMaxH3SubjectLabelMapForSegment: () => new Map(),
    selectedPerformerSubjectsForSegment: () => [],
    miniMaxH3PerShotFramingLines: () => [],
    miniMaxH3CueShotContractText: () => "",
    miniMaxH3VocalCueMapText: () => "",
    castSafeText: (segment, value) => value,
    sceneVideoConceptPromptText: () => "",
    sceneCastRestrictionText: () => "",
    segmentMappedSubjectText: () => "",
    segmentMappedLocationText: () => "",
    normalizeMiniMaxH3Mode: value => value,
    segmentMappedExtraSubjectText: () => "",
    miniMaxH3ReferenceAssignmentLines: () => [],
    miniMaxI2VTransitionPrompt: () => "",
  });
  vm.runInContext(functionSource(readBuilderModule("minimax_prompt.mjs"),
    "miniMaxH3CreativePromptContextForSegment"), context);
  const sceneCard = JSON.stringify({ notes: "n".repeat(4500), timeline_note: "Director at the end" }, null, 2);
  const result = vm.runInContext("miniMaxH3CreativePromptContextForSegment", context)(
    { start: 0, end: 5 }, "text_to_video", { storyboardContext: sceneCard });
  assert.ok(result.includes(sceneCard), "The full card must reach the creative LLM without clipping");
});

test("Reopening retains added cards and deleted cards without losing new timeline scenes", async () => {
  const folder = path.join(project, "structure");
  const state = stateFor(folder);
  const api = fixture(state);
  state.scenes = [{ ...state.scenes[0], id: "added", label: "Added card" }];
  await api.saveStoryboard({ throwOnError: true });
  const reopened = stateFor(folder);
  reopened.scenes.push({ ...reopened.scenes[0], id: "new-timeline", scene_number: 2 });
  const loading = fixture(reopened);
  assert.equal(await loading.loadExisting(), true);
  assert.deepEqual(clone(reopened.scenes.map((scene) => scene.id)), ["added", "new-timeline"]);
  reopened.scenes = [];
  await loading.saveStoryboard({ throwOnError: true });
  const empty = stateFor(folder);
  assert.equal(await fixture(empty).loadExisting(), true);
  assert.equal(empty.scenes.length, 0);
});

test("A failed parent project save reports failure and releases the Save button", async () => {
  const state = stateFor(path.join(project, "failure"));
  const api = fixture(state, async () => { throw new Error("Disk full"); });
  await assert.rejects(api.saveStoryboard({ throwOnError: true }), /Disk full/);
  assert.equal(state.saving, false);
  assert.equal(api.deps.save.disabled, false);
  assert.equal(api.messages.length, 0);
});

test("Explicit saves preserve reference metadata and apply catalog deletions", () => {
  const source = fs.readFileSync(path.join(root, "web/music_video_builder/storyboard_bridge.mjs"), "utf8");
  const start = source.indexOf("const applyStoryboardReferenceMappings =");
  const end = source.indexOf("const applyStoryboardPrompts =", start);
  const state = { fluxReferenceBuilder: { subjects: [{ id: "removed", name: "Removed" }], locations: [] } };
  const context = vm.createContext({ state, normalizeFluxReferenceBuilder: (value) => value,
    allEditableSegments: () => [], render: () => {}, autoSaveSessionQuiet: () => { throw new Error("Unexpected background save"); } });
  vm.runInContext(fs.readFileSync(path.join(root, "web/music_video_builder/refmod_labels.mjs"), "utf8")
    .replace(/^export /gm, ""), context);
  vm.runInContext(source.slice(start, end), context);
  const refs = stateFor(project).referenceBuilder;
  refs.subjects[0].minimax_voice = { preset_id: "voice1" };
  vm.runInContext("applyStoryboardReferenceMappings", context)({ reference_builder: refs },
    { saveProject: false, replaceCatalog: true });
  assert.equal(state.fluxReferenceBuilder.subjects.length, 1);
  assert.equal(state.fluxReferenceBuilder.subjects[0].id, "subject1");
  assert.equal(state.fluxReferenceBuilder.subjects[0].refmod.strength, 0.6);
  assert.equal(state.fluxReferenceBuilder.subjects[0].minimax_voice.preset_id, "voice1");
});
