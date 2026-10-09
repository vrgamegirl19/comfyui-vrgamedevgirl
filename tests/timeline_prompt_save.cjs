const { functionSource, readBuilderSource, readStoryboardSource } = require('./builder_source.cjs');
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const path = require("node:path");
const source = readBuilderSource();
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function fixture() {
  const a = { id: "a", i2v_prompt: "old", minimax_h3_prompt: "old" };
  const b = { id: "b", i2v_prompt: "other", minimax_h3_prompt: "other" };
  const saved = { story_layer: { brief: "keep" }, scenes: [
    { id: "a", scene_number: 1, video_prompt: "old", story_beat: "keep A beat" },
    { id: "b", scene_number: 2, video_prompt: "newer B", image_prompt: "keep B image" },
    { id: "c", scene_number: 3, video_prompt: "storyboard only" },
  ] };
  const button = () => ({ disabled: false, style: {}, textContent: "Save Updated Prompt" });
  const ctx = {
    selected: a, a, b, project: "project A", saved, writes: [], messages: [], sessionCalls: [],
    i2vPrompt: { value: "edited" }, miniMaxPrompt: { value: "edited" },
    saveI2VPromptButton: button(), saveMiniMaxPromptButton: button(),
    activeSegment: () => ctx.selected,
    activeProjectFolderForSave: () => ctx.project,
    storyboardScenePayload: () => [a, b].map((s, i) => ({ id: s.id, scene_number: i + 1, video_prompt: s.i2v_prompt })),
    saveSession: async (options) => { ctx.sessionCalls.push(options); if (ctx.sessionError) throw Error("session failed"); },
    postJson: async (url, payload) => {
      if (url.endsWith("/load")) { if (ctx.loadError) throw Error("load failed"); return { storyboard: saved }; }
      if (ctx.saveError) throw Error("save failed");
      ctx.writes.push(payload);
    },
    toast: (message) => ctx.messages.push(message),
  };
  vm.createContext(ctx);
  vm.runInContext(section("  // Mutable runtime state", "\n  setBuilderAutomaticMemoryCleanupEnabled") + "\n" +
    functionSource(source, "updateI2VPromptSaveButtonState") + "\n" +
    functionSource(source, "updateMiniMaxPromptSaveButtonState") + "\n" +
    functionSource(source, "saveStoryboardPromptFromTimeline") + "\n" + functionSource(source, "saveTimelinePrompt"), ctx);
  vm.runInContext('savedI2VPrompts.set(a, "old"); savedMiniMaxPrompts.set(a, "old");', ctx);
  return ctx;
}
for (const kind of ["i2v", "minimax"]) {
  test(`${kind}: saves just the target prompt, retaining other scenes and fields`, async () => {
    const c = fixture(); await c.saveTimelinePrompt(kind);
    const written = c.writes[0].storyboard;
    assert.equal(written.scenes[0].video_prompt, "edited");
    assert.equal(written.scenes[0].video_prompt_origin, "manual");
    assert.equal(written.scenes[0].story_beat, "keep A beat");
    assert.equal(written.scenes[1], c.saved.scenes[1]);
    assert.equal(written.scenes[2], c.saved.scenes[2]);
    assert.equal(written.story_layer, c.saved.story_layer);
    assert.equal(c.sessionCalls[0].throwOnError, true);
  });
  for (const failure of ["sessionError", "loadError", "saveError"]) {
    test(`${kind}: ${failure} leaves retry enabled and does not report success`, async () => {
      const c = fixture(); c[failure] = true;
      await c.saveTimelinePrompt(kind);
      assert.equal(c.writes.length, 0);
      assert.equal((kind === "i2v" ? c.saveI2VPromptButton : c.saveMiniMaxPromptButton).disabled, false);
      assert.equal(c.messages.some(m => m.includes("Prompt saved")), false);
      c[failure] = false; await c.saveTimelinePrompt(kind);
      assert.equal(c.writes.length, 1);
    });
  }
}
test("scene/project switching during save keeps the captured target and newer edits dirty", async () => {
  const c = fixture(); let resume;
  c.saveSession = () => new Promise(resolve => { resume = resolve; });
  const pending = c.saveTimelinePrompt("i2v");
  c.selected = c.b; c.project = "project B"; c.i2vPrompt.value = "new edit";
  await c.saveTimelinePrompt("minimax");
  resume(); await pending;
  assert.equal(c.writes.length, 1);
  assert.equal(c.writes[0].project_folder, "project A");
  assert.equal(c.writes[0].storyboard.scenes[0].video_prompt, "edited");
  assert.equal(c.saveI2VPromptButton.disabled, false);
});
test("missing project fails before persistence", async () => {
  const c = fixture(); c.project = ""; await c.saveTimelinePrompt("i2v");
  assert.equal(c.sessionCalls.length, 0); assert.equal(c.writes.length, 0);
});
test("scene-number fallback preserves stored scene identity", async () => {
  const c = fixture(); c.saved.scenes[0].id = "legacy-a";
  await c.saveTimelinePrompt("i2v");
  assert.equal(c.writes[0].storyboard.scenes[0].id, "legacy-a");
  assert.equal(c.writes[0].storyboard.scenes.length, 3);
});
test("missing target appends only that scene", async () => {
  const c = fixture(); c.saved.scenes = [c.saved.scenes[2]];
  await c.saveTimelinePrompt("i2v");
  assert.deepEqual(c.writes[0].storyboard.scenes.map(s => s.id), ["c", "a"]);
});

test("saving a prompt never overwrites a different authored card with the same number", async () => {
  const c = fixture();
  c.saved.source_scene_ids = ["a"];
  c.saved.scenes[0].id = "authored-card";
  await c.saveTimelinePrompt("i2v");
  assert.equal(c.writes[0].storyboard.scenes[0].id, "authored-card");
  assert.equal(c.writes[0].storyboard.scenes.at(-1).id, "a");
});

test("a stale project save keeps the prompt dirty and does not write the storyboard", async () => {
  const c = fixture();
  c.saveSession = async () => ({ stale: true });
  await c.saveTimelinePrompt("i2v");
  assert.equal(c.writes.length, 0);
  assert.equal(c.messages.some(message => message.includes("Prompt saved")), false);
  assert.equal(c.saveI2VPromptButton.disabled, false);
});

const storyboardSource = readStoryboardSource();
const storyboardSection = (start, end) => storyboardSource.slice(storyboardSource.indexOf(start), storyboardSource.indexOf(end, storyboardSource.indexOf(start)));
test("reopening storyboard uses the current timeline prompts, including intentionally blank prompts", () => {
  for (const savedPrompt of ["newer saved prompt", ""]) {
    const savedScene = { id: "a", scene_number: 1, image_prompt: "old prompt", video_prompt: "old prompt", video_prompt_origin: "manual" };
    const fresh = { id: "a", scene_number: 1, image_prompt: savedPrompt, video_prompt: savedPrompt, subject_refs: [] };
    const c = { state: {}, scenesToShow: [fresh], savedScenes: [savedScene], incomingScenes: [fresh], incomingById: new Map([["a", fresh]]), sourceIds: null, payloadVideoPromptType: "", currentLocationsCleared: false,
      normalizeStoryboardMiniMaxH3Mode: x => x, normalizeScene: x => x };
    vm.runInNewContext(storyboardSection("        state.scenes = scenesToShow.map(", "        if (currentLocationsCleared)"), c);
    assert.equal(c.state.scenes[0].video_prompt, savedPrompt);
    assert.equal(c.state.scenes[0].image_prompt, savedPrompt);
  }
});
for (const mode of ["image_to_video_prep", "storyboard_prompts"]) {
  test(`${mode}: selected LLM action calls only checked scenes`, async () => {
    const c = { state: { mode, scenes: [{ id: "a" }, { id: "b" }, { id: "c" }], selected: new Set(["b"]) },
      choosePromptGenerationScope: async () => "all",
      promptRunnerName: () => "LLM", promptRunnerGenericName: () => "LLM",
      createStoryboardProgressWindow: () => ({ set() {}, close() {} }),
      keepGemmaLoadedInput: { checked: true }, gemmaAllButton: {}, calls: [],
      saveStoryboard: async () => {}, createToast() {}, renderTable() {},
      createScenePromptForActiveMode: async s => c.calls.push(s.id),
    };
    vm.createContext(c);
    vm.runInContext(["getSelectedScenes", "isRecoverableStoryboardBatchError", "showStoryboardBatchFailures", "createAllPromptsWithGemma",
      "startAllPromptsWithGemma"].map((name) => functionSource(storyboardSource, name)).join("\n"), c);
    await c.startAllPromptsWithGemma();
    assert.deepEqual(c.calls, ["b"]);
  });
}
