const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
const { functionSource, readBuilderModule } = require("./builder_source.cjs");

function fixture(failCollection = false) {
  const calls = [];
  const segment = { id: "s1", start: 0, end: 10, video_path: "draft.mp4", minimax_h3_draft: { id: "checkpoint", video_path: "draft.mp4" }, video_history: ["draft.mp4"], video_thumbnail_history: [] };
  const sandbox = {
    state: {}, projectInput: { value: "project" },
    sceneSlotNumber: () => 1, queueWorkflowPrompt: async graph => { calls.push(["queue", graph]); return { prompt_id: "p1" }; },
    waitForVideos: async (_id, update) => { update("Waiting for scene video..."); return [{ filename: "stage2.mp4" }]; }, resolveComfyVideoPath: () => "render.mp4",
    postJson: async (route, body) => {
      calls.push([route, body]);
      if (route.endsWith("collect_scene_video")) {
        if (failCollection) throw new Error("collection failed");
        return { video_path: "draft.mp4", thumbnail_path: "thumb.jpg" };
      }
      if (route.endsWith("trim_scene_video")) return { video_path: "trim.mp4" };
      return {};
    },
    pushHistory: () => {}, mediaPathKey: value => value,
    activateSegmentVideoPath: (scene, video) => { scene.video_path = video; },
    collectedSceneVideoFolder: () => "project/videos", syncPreview: () => {}, render: () => {},
    loadDirtyLatentBadges: () => {}, autoSaveSessionQuiet: async () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(functionSource(readBuilderModule("video_render.mjs"), "finishMiniMaxVideoRender"), sandbox);
  const built = { prompt: { "125": { class_type: "VRGDG_MiniMaxH3LoadDraft" } }, draft_id: "checkpoint", timing: {}, post_render_trim: { duration: 10 }, two_pass: {} };
  return { calls, segment, sandbox, built };
}

test("successful finalization overwrites the draft and clears its badge state", async () => {
  const { calls, segment, sandbox, built } = fixture();
  await sandbox.finishMiniMaxVideoRender(segment, 0, null, built, { renderPhase: "finalize", existingVideoAction: "backup" });
  const collect = calls.find(([route]) => route.endsWith("collect_scene_video"));
  assert.equal(collect[1].existing_action, "overwrite");
  assert.equal(segment.minimax_h3_draft, null);
  assert.equal(segment.minimax_h3_stage1_path, "");
  assert.equal(segment.minimax_h3_stage2_path, "draft.mp4");
});

test("failed finalization retains the draft video and checkpoint", async () => {
  const { segment, sandbox, built } = fixture(true);
  await assert.rejects(sandbox.finishMiniMaxVideoRender(segment, 0, null, built, { renderPhase: "finalize" }), /collection failed/);
  assert.equal(segment.video_path, "draft.mp4");
  assert.equal(segment.minimax_h3_draft.id, "checkpoint");
});

test("draft completion records pass 1 and the checkpoint in session-owned scene data", async () => {
  const { segment, sandbox, built } = fixture();
  await sandbox.finishMiniMaxVideoRender(segment, 0, null, built, { renderPhase: "draft" });
  assert.equal(segment.minimax_h3_draft.id, "checkpoint");
  assert.equal(segment.minimax_h3_draft.video_path, segment.video_path);
  assert.equal(segment.minimax_h3_stage1_path, segment.video_path);
  assert.equal(segment.minimax_h3_stage2_path, "");
});

test("scene cards show red DRAFT only for the selected draft video", () => {
  const source = fs.readFileSync(path.join(__dirname, "../web/music_video_builder/timeline_view.mjs"), "utf8");
  const sandbox = { document: { createElement: () => ({ style: {} }) } };
  vm.createContext(sandbox);
  vm.runInContext(functionSource(source, "appendMiniMaxDraftBadge"), sandbox);
  const badges = [];
  const element = { append: badge => badges.push(badge) };
  const scene = { video_path: "draft.mp4", minimax_h3_draft: { id: "one", video_path: "draft.mp4" } };
  sandbox.appendMiniMaxDraftBadge(element, scene);
  assert.equal(badges.length, 1);
  assert.equal(badges[0].textContent, "DRAFT");
  assert.match(badges[0].style.cssText, /right:5px;bottom:5px/);
  sandbox.appendMiniMaxDraftBadge(element, { ...scene, video_path: "older-final.mp4" });
  sandbox.appendMiniMaxDraftBadge(element, { ...scene, minimax_h3_draft: null });
  assert.equal(badges.length, 1);
});

async function runBatch(renderPhase, forceVideos = false) {
  const rendered = [];
  const renderOptions = [];
  const messages = [];
  let stitched = false;
  const segments = [
    { id: "draft", start: 0, end: 10, video_path: "draft.mp4", minimax_h3_draft: { id: "one" } },
    { id: "final", start: 10, end: 20, video_path: "final.mp4" },
  ];
  const sandbox = {
    state: { projectVideoEngine: "minimax_h3", miniMaxH3Settings: {}, segments },
    projectInput: { value: "project" }, renderAllButton: {}, createSceneVideoButtons: [], miniMaxSceneVideoButtons: [],
    gemmaThenCreateVideoButtons: [], console,
    normalizeProjectVideoEngine: value => value, normalizeBatchScope: value => value || "all",
    ensureAudioOrOfferSilentTimeline: async () => true,
    validateRenderAllReady: () => [], miniMaxBatchReferenceProblems: () => [],
    batchTargetItems: () => segments.map((segment, index) => ({ segment, index })),
    selectedSegmentVideoPath: segment => segment.video_path,
    createProgressWindow: () => ({ set: message => messages.push(message), close: () => {} }),
    renderETAScene: () => ({}), miniMaxH3ModeForSegment: () => "reference_to_video",
    sceneSlotNumber: segment => segments.indexOf(segment) + 1,
    sceneDisplayName: segment => segment.id,
    updateRenderLogSummary: () => ({ completed_scenes: 0 }), renderLogDuration: () => "0s",
    renderMiniMaxSceneVideoWithProgress: async (segment, _index, _progress, options) => {
      rendered.push([segment.id, options.renderPhase]); renderOptions.push(options); return segment.video_path;
    },
    stitchRenderedScenes: async () => { stitched = true; return { final_video_path: "final.mp4" }; },
    toast: message => messages.push(message),
  };
  for (const name of ["updateActiveFromInputs", "saveI2VVideoSettingsFromPanel", "saveMiniMaxH3SettingsFromPanel", "recoverSceneVideosFromProject", "startBuilderETA", "upsertRenderLog", "setButtonGroupState", "saveSessionForSceneVideo", "persistRenderLog", "assertBatchNotStopped", "runClearMemoryWorkflowQuiet", "autoSaveSessionQuiet", "showFinalVideoReadyModal", "syncInspector", "render"]) sandbox[name] = () => {};
  vm.createContext(sandbox);
  vm.runInContext(functionSource(readBuilderModule("batch_render.mjs"), "renderAllScenes"), sandbox);
  await sandbox.renderAllScenes({ renderPhase, forceVideos, suppressFinalModal: true, throwOnError: true });
  return { rendered, renderOptions, stitched, messages };
}

test("Render All Drafts runs pass 1 for every target and skips stitching", async () => {
  const result = await runBatch("draft", true);
  assert.deepEqual(result.rendered, [["draft", "draft"], ["final", "draft"]]);
  assert.equal(result.stitched, false);
});

test("Finalize All targets only drafts; normal Resume also finalizes pending drafts", async () => {
  for (const mode of ["finalize", "full"]) {
    const result = await runBatch(mode);
    assert.deepEqual(result.rendered, [["draft", "finalize"]]);
    assert.equal(result.stitched, true);
  }
});

test("draft progress identifies pass 1 without listing pass 2 steps", async () => {
  const { segment, sandbox, built } = fixture();
  built.draft_steps = { pass1: 20, pass2: 4 };
  const messages = [];
  await sandbox.finishMiniMaxVideoRender(segment, 0, { set: message => messages.push(message) }, built, { renderPhase: "draft", exactTwoPassStepsLine: "Pass 1 = 20; Pass 2 = 4" });
  const waiting = messages.find(message => message.includes("Waiting for scene video"));
  assert.match(waiting, /DRAFT — Pass 1 Only/);
  assert.match(waiting, /Pass 1 steps: 20/);
  assert.doesNotMatch(waiting, /Pass 2|Stage 2/);
});

test("only a Finalize that Render chose by itself may fall back to a full render", async () => {
  assert.equal((await runBatch("full")).renderOptions[0].draftAutoFinalize, true);
  assert.equal((await runBatch("finalize")).renderOptions[0].draftAutoFinalize, false);
});

function staleDraftSandbox() {
  const messages = [];
  const sandbox = {
    state: {}, projectInput: { value: "project" }, sceneSlotNumber: () => 1, renderList: () => {},
    miniMaxH3SettingsForSegment: () => ({ diffusion_model_name: "fasth3_omniref_qad_nvfp4_comfy.safetensors" }),
    postJson: async () => ({
      stale_draft: true, draft_models: ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
      current_model: "fasth3_omniref_qad_nvfp4_comfy.safetensors",
    }),
    // The full render path starts here; stop it so the test only checks which path was taken.
    prepareEditedAudio: async () => { throw new Error("FULL RENDER STARTED"); },
  };
  vm.createContext(sandbox);
  vm.runInContext(functionSource(readBuilderModule("video_render.mjs"), "renderMiniMaxSceneVideoWithProgress"), sandbox);
  const segment = { id: "s1", start: 0, end: 10, minimax_h3_draft: { id: "checkpoint" } };
  return { sandbox, segment, progress: { set: message => messages.push(message) }, messages };
}

test("a draft made with another model renders fresh when Render chose Finalize", async () => {
  const { sandbox, segment, progress, messages } = staleDraftSandbox();
  await assert.rejects(
    sandbox.renderMiniMaxSceneVideoWithProgress(segment, 0, progress, { renderPhase: "finalize", draftAutoFinalize: true }),
    /FULL RENDER STARTED/,
  );
  assert.match(messages[0], /made with minimax_h3_ref2va_pruned_int8_convrot\.safetensors/);
  assert.match(messages[0], /Rendering both passes with the current settings/);
});

test("an explicit Finalize of a draft made with another model stops with the reason", async () => {
  const { sandbox, segment, progress } = staleDraftSandbox();
  await assert.rejects(
    sandbox.renderMiniMaxSceneVideoWithProgress(segment, 0, progress, { renderPhase: "finalize" }),
    /now uses fasth3_omniref_qad_nvfp4_comfy\.safetensors[\s\S]*Render a new draft/,
  );
});
