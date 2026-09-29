import { postJson } from "./api.mjs";
import { createStoryboardProgressWindow, createToast, makeButton } from "./controls.mjs";
import { storyboardGptPayload } from "./gpt_payload.mjs";
import { isRecoverableStoryboardBatchError, showStoryboardBatchFailures } from "./prompt_generation.mjs";
import { normalizeScene, normalizeStoryLayer, slimStoryboardForRequest } from "./scenes.mjs";

export function sceneStoryBeatMissing(scene, flfMode) {
  return !String(scene.story_beat || "").trim()
    || (flfMode && [scene.flf_start_state, scene.flf_transformation, scene.flf_end_state, scene.flf_carry_forward].some((value) => !String(value || "").trim()));
}

export function adjacentLyricLine(lyrics, direction = "first") {
  const lines = String(lyrics || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return "";
  return direction === "last" ? lines[lines.length - 1] : lines[0];
}

export function confirmClearAllStoryBeats() {
  return new Promise((resolve) => {
  const confirmBackdrop = document.createElement("div");
  confirmBackdrop.style.cssText = "position:fixed;inset:0;z-index:100040;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:22px;";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(620px,calc(100vw - 44px));border:1px solid #991b1b;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;";
  const confirmHeader = document.createElement("div");
  confirmHeader.style.cssText = "padding:14px 16px;background:#3f0808;border-bottom:1px solid #991b1b;font-weight:900;color:#fecaca;";
  confirmHeader.textContent = "Clear all Storyboard story beats?";
  const body = document.createElement("div");
  body.style.cssText = "padding:16px;line-height:1.45;color:#e2e8f0;font-size:13px;";
  body.textContent = "This clears only the Scene Story Beat field in every scene. Lyrics, generated prompts, notes, images, subjects, locations, references, camera settings, and motion settings will remain unchanged.";
  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 16px 16px;";
  const cancel = makeButton("Cancel");
  const clear = makeButton("Yes, clear story beats", "primary");
  clear.style.borderColor = "#991b1b";
  clear.style.background = "#991b1b";
  const finish = (value) => { confirmBackdrop.remove(); resolve(value); };
  cancel.onclick = () => finish(false);
  clear.onclick = () => finish(true);
  confirmBackdrop.addEventListener("pointerdown", (event) => { if (event.target === confirmBackdrop) finish(false); });
  actions.append(cancel, clear);
  panel.append(confirmHeader, body, actions);
  confirmBackdrop.append(panel);
  document.body.append(confirmBackdrop);
});
}

export function createSceneBeats({
  currentRows, getSelectedScenes, promptRunnerName, renderTable, saveStoryboard, state,
  syncStoryLayerFromInputs,
}) {
  function sceneBeatGemmaPayload(scene, overrides = {}) {
    return {
    ...(state.gemmaSettings || {}),
    ...overrides,
    story_layer: normalizeStoryLayer(state.storyLayer),
    all_subjects: (Array.isArray(state.referenceBuilder?.subjects) ? state.referenceBuilder.subjects : [])
      .map((subject) => ({ name: String(subject?.name || ""), description: String(subject?.description || "") })),
    // A replacement request must not feed the old beat back to the model as
    // if it were authoritative. Lyrics, mappings, defaults, and references
    // remain; only the stale generated beat is cleared from this request.
    storyboard_payload: storyboardGptPayload(state, [{ ...scene, story_beat: "" }]),
    max_new_tokens: state.videoPromptType === "flf" ? 700 : 360,
    temperature: 0.35,
    top_p: 0.90,
  };
  }

  function propagateFlfEndStateToNextScene(scene) {
    if (state.videoPromptType !== "flf" && scene?.video_prompt_type !== "flf") return;
    const sceneIndex = state.scenes.findIndex((item) => item.id === scene?.id);
    if (sceneIndex < 0 || sceneIndex >= state.scenes.length - 1) return;
    const endState = String(scene.flf_end_state || "").trim();
    if (!endState) return;
    state.scenes[sceneIndex + 1].flf_start_state = endState;
  }

  async function createSceneBeatWithGemma(scene, { quiet = false, unloadAfter = true, previousBeat = "", previousLyrics = "", previousEndState = "", previousCarryForward = "", nextLyrics = "", progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    syncStoryLayerFromInputs();
    const normalized = normalizeScene(scene, 0);
    const sceneIndex = state.scenes.findIndex((item) => item.id === scene.id);
    if (!previousBeat && sceneIndex > 0) previousBeat = String(state.scenes[sceneIndex - 1]?.story_beat || "");
    if (!previousLyrics && sceneIndex > 0) previousLyrics = adjacentLyricLine(state.scenes[sceneIndex - 1]?.lyrics, "last");
    if (!previousEndState && sceneIndex > 0) previousEndState = String(state.scenes[sceneIndex - 1]?.flf_end_state || "");
    if (!previousCarryForward && sceneIndex > 0) previousCarryForward = String(state.scenes[sceneIndex - 1]?.flf_carry_forward || "");
    if (!nextLyrics && sceneIndex >= 0 && sceneIndex < state.scenes.length - 1) nextLyrics = adjacentLyricLine(state.scenes[sceneIndex + 1]?.lyrics, "first");
    if (!state.sendAdjacentLyricContext) {
      previousLyrics = "";
      nextLyrics = "";
    }
    if ((state.videoPromptType === "flf" || normalized.video_prompt_type === "flf") && sceneIndex > 0 && previousEndState.trim()) {
      scene.flf_start_state = previousEndState.trim();
    }
    try {
      progress?.set(`${progressLabel || normalized.label || "Scene"}: creating scene story beat with ${promptRunnerName()}...`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/scene_story_beat", sceneBeatGemmaPayload(scene, {
        unload_after: unloadAfter,
        previous_beat: previousBeat,
        previous_lyrics: previousLyrics,
        previous_end_state: previousEndState,
        previous_carry_forward: previousCarryForward,
        current_lyrics: normalized.lyrics,
        next_lyrics: nextLyrics,
        flf_mode: state.videoPromptType === "flf" || normalized.video_prompt_type === "flf",
      }), 240000);
      scene.story_beat = String(data.story_beat || "").trim();
      if (state.videoPromptType === "flf" || normalized.video_prompt_type === "flf") {
        scene.flf_start_state = sceneIndex > 0 && previousEndState.trim()
          ? previousEndState.trim()
          : String(data.flf_start_state || "").trim();
        scene.flf_transformation = String(data.flf_transformation || "").trim();
        scene.flf_end_state = String(data.flf_end_state || "").trim();
        scene.flf_carry_forward = String(data.flf_carry_forward || "").trim();
        propagateFlfEndStateToNextScene(scene);
      }
      if (!scene.story_beat) throw new Error(`${promptRunnerName()} returned an empty scene story beat.`);
      if (!quiet) createToast(`Scene story beat created for ${normalized.label || "scene"}.`);
      return scene.story_beat;
    } catch (error) {
      if (!quiet) createToast(`Scene story beat failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  async function createAllSceneBeatsWithGemma({ failedSceneIds = [] } = {}) {
    syncStoryLayerFromInputs();
    const flfMode = state.videoPromptType === "flf";
    const failedIds = new Set(failedSceneIds.map((value) => String(value)));
    const scenes = currentRows().filter((scene) => failedIds.size
      ? failedIds.has(String(scene.id || ""))
      : sceneStoryBeatMissing(scene, flfMode));
    if (!scenes.length) {
      createToast("No scene story beats are missing.");
      return;
    }
    const progress = createStoryboardProgressWindow(`Create Missing Scene Beats — ${promptRunnerName()}`);
    let created = 0;
    const failures = [];
    try {
      progress.set(`${failedIds.size ? "Retrying failed" : "Creating"} ${scenes.length} scene story beat${scenes.length === 1 ? "" : "s"}...`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        const scene = scenes[index];
        const allIndex = state.scenes.findIndex((item) => item.id === scene.id);
        const previousBeat = allIndex > 0 ? String(state.scenes[allIndex - 1]?.story_beat || "") : "";
        const previousLyrics = allIndex > 0 ? String(state.scenes[allIndex - 1]?.lyrics || "") : "";
        const previousEndState = allIndex > 0 ? String(state.scenes[allIndex - 1]?.flf_end_state || "") : "";
        const previousCarryForward = allIndex > 0 ? String(state.scenes[allIndex - 1]?.flf_carry_forward || "") : "";
        const nextLyrics = allIndex >= 0 && allIndex < state.scenes.length - 1 ? String(state.scenes[allIndex + 1]?.lyrics || "") : "";
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        try {
          await createSceneBeatWithGemma(scene, {
            quiet: true,
            unloadAfter: index === scenes.length - 1,
            previousBeat,
            previousLyrics,
            previousEndState,
            previousCarryForward,
            nextLyrics,
            progress,
            progressPercent: base,
            progressLabel: `Scene Beat ${index + 1}/${scenes.length}: ${scene.label || `Scene ${scene.scene_number || index + 1}`}`,
          });
          created += 1;
        } catch (error) {
          if (!isRecoverableStoryboardBatchError(error)) throw error;
          failures.push({ scene, error: String(error?.message || error) });
          progress.set(`Scene Beat ${index + 1}/${scenes.length} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      progress.set("Saving story beats...", 96);
      await saveStoryboard();
      progress.set(`Scene beats complete.\nCreated ${created} story beat${created === 1 ? "" : "s"}.${failures.length ? ` ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : ""}`, 100);
      progress.close(1600);
      createToast(`Created ${created} scene story beat${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      if (failures.length) showStoryboardBatchFailures(failures, (items) => createAllSceneBeatsWithGemma({
        failedSceneIds: items.map((item) => item.scene.id),
      }));
    } catch (error) {
      progress.set(`Scene beats stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, 100);
      createToast(`Scene beats stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, true);
    }
  }

  async function clearAllStoryboardStoryBeats() {
    if (!await confirmClearAllStoryBeats()) return;
    let changed = 0;
    for (const scene of state.scenes) {
      if (String(scene.story_beat || "").trim()) changed += 1;
      scene.story_beat = "";
    }
    renderTable();
    if (state.onStoryLayerChanged) {
      state.onStoryLayerChanged({
        scenes: state.scenes.map((scene) => ({
          id: scene.id,
          scene_number: scene.scene_number,
          story_beat: "",
        })),
      });
    }
    if (state.projectFolder) {
      try {
        await postJson("/vrgdg/storyboard/save", {
          project_folder: state.projectFolder,
          storyboard: slimStoryboardForRequest(state),
        });
        createToast(`Cleared story beats in ${changed} scene${changed === 1 ? "" : "s"} and saved Storyboard.`);
      } catch (error) {
        createToast(`Cleared story beats in this session, but could not save Storyboard:\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast(`Cleared story beats in ${changed} scene${changed === 1 ? "" : "s"}. Save the project to keep this change.`);
    }
  }
  async function handleReplaceSceneBeats() {
    const selectedScenes = getSelectedScenes();
    if (!selectedScenes.length) return;
    const isSingle = selectedScenes.length === 1;
    const message = isSingle
      ? "This will replace the scene beat in selected scene only."
      : `This will replace the scene beats in ${selectedScenes.length} selected scenes only.`;
    const confirmed = window.confirm(message);
    if (!confirmed) return;
    const runnerName = promptRunnerName();
    const progress = createStoryboardProgressWindow(`Replace Scene Beats — ${runnerName}`);
    let created = 0;
    const failures = [];
    try {
      progress.set(`Replacing scene beats for ${selectedScenes.length} selected scene${isSingle ? "" : "s"}...`, 5);
      for (let index = 0; index < selectedScenes.length; index += 1) {
        const scene = selectedScenes[index];
        const sceneLabel = scene.label || `Scene ${scene.scene_number || index + 1}`;
        const base = 8 + Math.round((index / Math.max(1, selectedScenes.length)) * 84);
        try {
          await createSceneBeatWithGemma(scene, {
            quiet: true,
            unloadAfter: index === selectedScenes.length - 1,
            progress,
            progressPercent: base,
            progressLabel: `Scene Beat ${index + 1}/${selectedScenes.length}: ${sceneLabel}`,
          });
          created += 1;
        } catch (error) {
          if (!isRecoverableStoryboardBatchError(error)) throw error;
          failures.push({ scene, error: String(error?.message || error) });
          progress.set(`Scene Beat ${index + 1}/${selectedScenes.length} skipped. Continuing...`, base);
        }
      }
      progress.set("Saving story beats...", 96);
      await saveStoryboard();
      progress.set(`Scene beats complete.\nReplaced ${created} story beat${created === 1 ? "" : "s"}.${failures.length ? ` ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : ""}`, 100);
      progress.close(1600);
      createToast(`Replaced ${created} scene story beat${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
    } catch (error) {
      progress.set(`Replace scene beats stopped after ${created}/${selectedScenes.length}:\n${String(error?.message || error)}`, 100);
      createToast(`Replace scene beats stopped after ${created}/${selectedScenes.length}:\n${String(error?.message || error)}`, true);
    } finally {
      renderTable();
    }
  }

  return {
    clearAllStoryboardStoryBeats, createAllSceneBeatsWithGemma, createSceneBeatWithGemma,
    handleReplaceSceneBeats, propagateFlfEndStateToNextScene,
  };
}
