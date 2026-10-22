import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import { applyCompactButtonLabel, makeButton, toast } from "./controls.mjs";

export const ENHANCE_PROMPT_LABEL = "Enhance\nPrompt";

export function chooseMiniMaxEnhanceReferences() {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100000;background:#000a;display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-label", "Enhance prompts");
    box.style.cssText = "background:#0f172a;color:#e2e8f0;border:1px solid #475569;border-radius:10px;padding:20px;max-width:480px;display:flex;flex-direction:column;gap:12px;";
    const title = document.createElement("strong");
    title.textContent = "Enhance prompts";
    const note = document.createElement("div");
    note.textContent = "Send the assigned references to add brief character appearance and location descriptions? With images uses your selected vision LLM; without images uses your text LLM.";
    const withImages = makeButton("With reference images", "primary");
    const withoutImages = makeButton("Without reference images");
    const cancel = makeButton("Cancel");
    const previousFocus = document.activeElement;
    const finish = (value) => {
      document.removeEventListener("keydown", onKey);
      backdrop.remove();
      previousFocus?.focus?.();
      resolve(value);
    };
    const onKey = (event) => {
      if (event.key === "Escape") { event.preventDefault(); finish(null); }
    };
    withImages.onclick = () => finish(true);
    withoutImages.onclick = () => finish(false);
    cancel.onclick = () => finish(null);
    backdrop.addEventListener("pointerdown", (event) => { if (event.target === backdrop) finish(null); });
    document.addEventListener("keydown", onKey);
    box.append(title, note, withImages, withoutImages, cancel);
    backdrop.append(box);
    document.body.append(backdrop);
    withImages.focus();
  });
}

// Both entry points choose the text-only or separate vision task. The backend validates shot rewrites;
// vision may additionally update only the explicitly mapped visual definitions.
export function createMiniMaxPromptEnhance({
  activeProjectFolderForSave, activeSegment, allEditableSegments, autoSaveSessionQuiet, createProgressWindow,
  gemmaRunnerLabel, gemmaRunnerLine, miniMaxEnhancePromptButton, miniMaxPrompt, miniMaxTextGemmaModelSelect,
  pushHistory, render, requireActiveSegment, sceneDisplayName, syncMiniMaxH3Panel, textGemmaRunnerPayload,
  updateActiveFromInputs, updateMiniMaxPromptCharacterStatus,
  miniMaxEnhanceReferenceImages, miniMaxGemmaModelSelect, miniMaxMmprojSelect,
}) {
  const savedPrompt = (segment) => String(segment?.minimax_h3_prompt || "").trim();

  async function requestEnhancedPrompt(segment, unloadAfter, useReferences) {
    const references = useReferences ? miniMaxEnhanceReferenceImages(segment) : [];
    if (useReferences && !references.length) throw new Error("This scene has no assigned reference images. Assign character/location references or choose Without reference images.");
    const data = await postJson("/vrgdg/music_builder/enhance_minimax_h3_prompt", {
      ...textGemmaRunnerPayload(),
      project_folder: activeProjectFolderForSave(),
      scene_id: segment.id || "",
      model_file: useReferences ? miniMaxGemmaModelSelect.value : miniMaxTextGemmaModelSelect.value,
      use_reference_images: useReferences,
      ...(useReferences ? { image_references: references, mmproj_file: miniMaxMmprojSelect.value } : {}),
      current_prompt: savedPrompt(segment),
      unload_after: unloadAfter,
    }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    const prompt = String(data.prompt || "").trim();
    if (!prompt) throw new Error("The LLM returned an empty prompt.");
    return prompt;
  }

  function applyEnhancedPrompt(segment, prompt) {
    if (segment.minimax_h3_prompt_reference_binding?.prompt === savedPrompt(segment)) {
      segment.minimax_h3_prompt_reference_binding = { ...segment.minimax_h3_prompt_reference_binding, prompt };
    }
    segment.minimax_h3_prompt = prompt;
    if (activeSegment()?.id === segment.id) {
      miniMaxPrompt.value = prompt;
      updateMiniMaxPromptCharacterStatus(segment);
    }
  }

  async function enhanceMiniMaxH3PromptWithLLM(options = {}) {
    updateActiveFromInputs();
    const segment = requireActiveSegment();
    if (!segment) return;
    if (!savedPrompt(segment)) {
      toast("Create this scene's MiniMax H3 prompt first, then enhance it.", true);
      return;
    }
    const useReferences = typeof options.useReferences === "boolean" ? options.useReferences : await chooseMiniMaxEnhanceReferences();
    if (useReferences === null) return;
    let progress = null;
    try {
      miniMaxEnhancePromptButton.disabled = true;
      miniMaxEnhancePromptButton.textContent = "Enhancing...";
      progress = createProgressWindow(`Enhancing ${sceneDisplayName(segment)} prompt`);
      progress.set(`Enhancing the MiniMax H3 prompt${useReferences ? " with reference images" : ""}...\n${gemmaRunnerLine({ vision: useReferences })}`, 30);
      const prompt = await requestEnhancedPrompt(segment, true, useReferences);
      pushHistory();
      applyEnhancedPrompt(segment, prompt);
      render();
      await autoSaveSessionQuiet("MiniMax H3 prompt enhanced");
      progress.set("Enhanced prompt saved. Undo brings back the previous prompt.", 100);
      progress.close(1200);
      toast(`Enhanced the MiniMax H3 prompt with ${gemmaRunnerLabel({ vision: useReferences })}.`);
    } catch (error) {
      const message = String(error?.message || error);
      progress?.set(`Error:\n${message}\n\nThe scene kept its prompt.`, 100);
      toast(message, true);
    } finally {
      miniMaxEnhancePromptButton.disabled = false;
      applyCompactButtonLabel(miniMaxEnhancePromptButton, ENHANCE_PROMPT_LABEL, { noMap: true, padding: "8px 6px", title: miniMaxEnhancePromptButton.title });
      syncMiniMaxH3Panel();
    }
  }

  async function enhanceAllMiniMaxH3Prompts(options = {}) {
    updateActiveFromInputs();
    const targets = allEditableSegments().filter((segment) => savedPrompt(segment));
    if (!targets.length) {
      toast("No scene has a MiniMax H3 prompt yet. Create the prompts first.", true);
      return;
    }
    const useReferences = typeof options.useReferences === "boolean" ? options.useReferences : await chooseMiniMaxEnhanceReferences();
    if (useReferences === null) return;
    const progress = createProgressWindow("Enhancing MiniMax H3 prompts");
    const failures = [];
    let enhanced = 0;
    pushHistory();
    for (const [index, segment] of targets.entries()) {
      const name = sceneDisplayName(segment);
      progress.set(`Scene ${index + 1} of ${targets.length}: ${name}\n${gemmaRunnerLine({ vision: useReferences })}`, Math.round((index / targets.length) * 100));
      try {
        applyEnhancedPrompt(segment, await requestEnhancedPrompt(segment, index === targets.length - 1, useReferences));
        enhanced += 1;
        await autoSaveSessionQuiet(`MiniMax H3 prompt enhanced: ${name}`);
      } catch (error) {
        failures.push(`${name}: ${String(error?.message || error)}`);
        console.warn(`[VRGDG Music Builder] Enhance prompt failed for ${name}:`, error);
      }
    }
    render();
    syncMiniMaxH3Panel();
    const summary = `Enhanced ${enhanced} of ${targets.length} MiniMax H3 prompts. Undo brings back the previous prompts.`;
    progress.set(failures.length ? `${summary}\n\nKept the old prompt for:\n${failures.join("\n")}` : summary, 100);
    if (!failures.length) progress.close(1500);
    toast(summary, Boolean(failures.length && !enhanced));
  }

  return { enhanceAllMiniMaxH3Prompts, enhanceMiniMaxH3PromptWithLLM };
}
