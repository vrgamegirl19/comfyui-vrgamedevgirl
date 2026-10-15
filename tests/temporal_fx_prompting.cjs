const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
const { functionSource, readBuilderModule } = require("./builder_source.cjs");
const styles = import("../web/storyboard_builder/video_style.mjs");
const source = readBuilderModule("batch_prompts.mjs");

async function context(defaults = {}, scene = {}) {
  const api = await styles;
  const sandbox = {
    ...api,
    state: { builderStoryboardDefaults: defaults },
    storyboardScenePayload: () => [{ id: "scene1", ...scene }],
    miniMaxH3CutPlanForSegment: () => ({ shot_count: 3 }),
  };
  vm.createContext(sandbox);
  vm.runInContext(functionSource(source, "sceneDefaultsFxPromptContext"), sandbox);
  return sandbox;
}

test("disabled FX adds no LLM guidance; scene Off overrides global temporal FX", async () => {
  const plain = await context();
  assert.equal(plain.sceneDefaultsFxPromptContext({ id: "scene1" }), "");
  const override = await context({ temporal_world_effect: "frozen_world" }, { temporal_world_effect_override: "off" });
  assert.equal(override.sceneDefaultsFxPromptContext({ id: "scene1" }), "");
});

test("temporal guidance carries intensity, protection, extras and time-passage controls", async () => {
  const api = await styles;
  const effect = api.storyboardTemporalWorldEffectForScene({ setting: "street" }, {
    temporal_world_effect: "realtime_subjects_timelapse_world",
    temporal_background_intensity: 9,
    temporal_protected_characters: "custom",
    temporal_protected_custom: "Singer A",
    temporal_allow_background_extras: false,
    temporal_environment_time_passage: false,
  });
  assert.match(effect.writing_guidance, /extreme 9\/10/);
  assert.match(effect.writing_guidance, /at least two concrete/);
  assert.match(effect.writing_guidance, /Protect only these named.*Singer A/);
  assert.match(effect.writing_guidance, /Do not add anonymous background people/);
  assert.match(effect.writing_guidance, /Do not add a day\/night/);
  assert.match(effect.writing_guidance, /primary audio at normal speed/);
  assert.equal(effect.exact_verbiage, undefined);
});

test("custom shot FX retains per-shot cues, timing, intensity and protection in LLM input", async () => {
  const sandbox = await context({
    fx_preset: "custom",
    fx_custom_json: JSON.stringify({ cues: ["Red light sweeps the wall.", "Blue light sweeps the ceiling."], intensity: 8, timing: "on the drum hit", avoid: "Keep her face readable." }),
  });
  const notes = sandbox.sceneDefaultsFxPromptContext({ id: "scene1" });
  const effects = JSON.parse(notes.split("Shot FX writing guidance:\n")[1].split("\n\n")[0]);
  assert.deepEqual(effects.map(effect => effect.shot), [1, 2, 3]);
  assert.equal(effects[0].cue, effects[2].cue);
  assert.notEqual(effects[0].cue, effects[1].cue);
  assert.equal(effects[1].intensity, 8);
  assert.equal(effects[1].timing, "on the drum hit");
  assert.equal(effects[1].protection, "Keep her face readable.");
});

test("shared MiniMax generator sends FX guidance to LLM and preserves returned prose", async () => {
  const sandbox = await context({ temporal_world_effect: "frozen_world" });
  const generated = "[Shot 1] She sings naturally while raindrops hang motionless behind her.";
  let request;
  Object.assign(sandbox, {
    ensureAutoTimedSingerCuesBeforePrompt: async () => {},
    miniMaxPromptReferenceSignature: () => "",
    miniMaxH3PromptVisionImagesForRunner: () => [],
    segmentUsesNoLipSyncPerformance: () => true,
    lyricFreePerformanceEnabled: () => true,
    effectiveVideoPerformanceModeForSegment: () => "visual_only",
    miniMaxH3SettingsForSegment: () => ({ audio_mode: "supplied" }),
    sceneCastGuardForSegment: () => null,
    miniMaxH3PromptCharacterBudget: () => ({ fixedChars: 0, hardLimit: 7000, shotDescriptionChars: 6500 }),
    textGemmaRunnerPayload: () => ({}),
    activeProjectFolderForSave: () => "project",
    miniMaxH3InstructionKey: () => "minimax",
    miniMaxTextGemmaModelSelect: { value: "test-model" },
    miniMaxH3CreativePromptContextForSegment: () => "Scene context",
    miniMaxH3SceneImageIsPromptInspiration: () => false,
    emotionExpressionInput: () => ({}),
    GEMMA_VIDEO_PROMPT_TIMEOUT_MS: 1000,
    postJson: async (_url, payload) => { request = payload; return { prompt: generated }; },
    castLeaks: () => [],
    feelingWordHits: () => [],
    hasEmotionExpressionInput: () => false,
    looksLikeUnfilledMiniMaxTemplate: () => false,
    assembleMiniMaxH3PromptFromCreative: (_segment, _mode, prompt) => prompt,
    assertValidMiniMaxH3FinalPrompt: () => {},
    rememberMiniMaxPromptReferences: () => {},
  });
  vm.runInContext(functionSource(source, "runMiniMaxH3PromptGeneration"), sandbox);
  const result = await sandbox.runMiniMaxH3PromptGeneration({ id: "scene1" }, "ref2va", { userNotes: "Keep the camera still." });
  assert.match(request.user_notes, /Keep the camera still/);
  assert.match(request.user_notes, /world frozen at one instant/);
  assert.match(request.user_notes, /naturally inside the appropriate shot descriptions/);
  assert.equal(result.prompt, generated);
});

test("all frontend FX appenders are removed", () => {
  for (const name of ["minimax_prompt.mjs", "storyboard_bridge.mjs", "batch_prompts.mjs", "video_render.mjs"]) {
    assert.doesNotMatch(readBuilderModule(name), /ensureBuilderManagedFx|ensureStoryboardRequiredTemporalWorldEffect/);
  }
  const standalone = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder/prompt_generation.mjs"), "utf8");
  assert.doesNotMatch(standalone, /applyBuilderManagedFx|FX accent:/);
});
