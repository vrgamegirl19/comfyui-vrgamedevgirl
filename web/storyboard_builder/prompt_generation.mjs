import { postJson, STORYBOARD_GEMMA_TIMEOUT_MS } from "./api.mjs";
import { createStoryboardProgressWindow, createToast, makeButton } from "./controls.mjs";
import { storyboardGptPayload } from "./gpt_payload.mjs";
import { normalizeReferenceBuilderCatalog } from "./references.mjs";
import {
  ensureStoryboardReferenceOpening,
  normalizeScene,
  normalizeStoryLayer,
  slimStoryboardForRequest,
} from "./scenes.mjs";
import { storyboardFxContract } from "./video_style.mjs";

export function confirmClearStoryboardPrompts() {
  return new Promise((resolve) => {
  const confirmBackdrop = document.createElement("div");
  confirmBackdrop.style.cssText = "position:fixed;inset:0;z-index:100040;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:22px;";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(620px,calc(100vw - 44px));border:1px solid #991b1b;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;";
  const header = document.createElement("div");
  header.style.cssText = "padding:14px 16px;background:#3f0808;border-bottom:1px solid #991b1b;font-weight:900;color:#fecaca;";
  header.textContent = "Clear all Storyboard prompts and notes?";
  const body = document.createElement("div");
  body.style.cssText = "padding:16px;line-height:1.45;color:#e2e8f0;font-size:13px;";
  body.textContent = "This clears prompt summaries, generated image/video prompts, and extra notes inside every scene card. It keeps lyrics, subjects, locations, reference images, shot type, camera motion, character motion, performance style, and microphone settings.";
  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 16px 16px;";
  const cancel = makeButton("Cancel");
  const clear = makeButton("Yes, clear prompts", "primary");
  clear.style.borderColor = "#991b1b";
  clear.style.background = "#991b1b";
  actions.append(cancel, clear);
  panel.append(header, body, actions);
  confirmBackdrop.append(panel);
  document.body.append(confirmBackdrop);
  const closeConfirm = (value) => {
    confirmBackdrop.remove();
    resolve(value);
  };
  cancel.onclick = () => closeConfirm(false);
  clear.onclick = () => closeConfirm(true);
  confirmBackdrop.addEventListener("pointerdown", (event) => {
    if (event.target === confirmBackdrop) closeConfirm(false);
  });
});
}

export function videoPromptTypeLabel(type) {
  if (type === "text_to_video") return "H3 T2V";
  if (type === "image_to_video") return "H3 I2V";
  if (type === "reference_to_video") return "H3 Reference";
  if (type === "video_to_video") return "H3 V2V";
  if (type === "id_lora") return "ID-LoRA I2V";
  if (type === "t2v") return "T2V";
  if (type === "rtv") return "RTV";
  return "I2V";
}

export function videoPromptTypeHint(type) {
  if (type === "text_to_video") return "MiniMax H3 Text to Video uses scene text and <Audio 1> without picture or video references.";
  if (type === "image_to_video") return "MiniMax H3 Image to Video uses the scene image as <Picture 1> and the authoritative opening frame.";
  if (type === "reference_to_video") return "MiniMax H3 Reference to Video uses this scene's ordered Reference Builder pictures and their exact <Picture N> tags.";
  if (type === "video_to_video") return "MiniMax H3 Video to Video uses the reference-video paths and purposes configured for this scene in Video Builder.";
  if (type === "id_lora") {
    return "ID-LoRA uses a scene image plus per-scene dialogue and a character voice sample from the ID-LoRA Ref Builder.";
  }
  if (type === "t2v") {
    return "T2V has no first frame, so choose an opening shot and describe the motion clearly.";
  }
  if (type === "rtv") {
    return "Reference to Video uses subject/location references plus an opening shot and motion direction.";
  }
  return "I2V already has a first frame, so use this mostly for camera movement, framing, and continuity.";
}

export function imagePromptImportJsonText(rawText) {
  const text = String(rawText || "").trim();
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) return fenced[1].trim();
  const firstArray = text.indexOf("[");
  const firstObject = text.indexOf("{");
  const starts = [firstArray, firstObject].filter((index) => index >= 0);
  if (!starts.length) return text;
  const start = Math.min(...starts);
  const end = Math.max(text.lastIndexOf("]"), text.lastIndexOf("}"));
  return end > start ? text.slice(start, end + 1).trim() : text.slice(start).trim();
}

export function parseVideoPromptImportJson(rawText) {
  return parseImagePromptImportJson(rawText, ["video_prompt", "i2v_prompt", "video", "prompt", "text"]);
}

export function parseImagePromptImportJson(rawText, promptKeys = ["image_prompt", "text_to_image_prompt", "t2i_prompt", "prompt", "text"]) {
  const text = imagePromptImportJsonText(rawText);
  if (!text) return [];
  const data = JSON.parse(text);
  const source = Array.isArray(data)
    ? data
    : Array.isArray(data.prompts)
      ? data.prompts
      : Array.isArray(data.scenes)
        ? data.scenes
        : data && typeof data === "object"
          ? Object.entries(data).map(([key, value]) => {
            if (value && typeof value === "object") return { scene: key, ...value };
            return { scene: key, prompt: value };
          })
          : [];
  const rows = [];
  for (const item of source) {
    if (!item || typeof item !== "object") continue;
    const sceneRaw = item.scene_number ?? item.sceneNumber ?? item.scene ?? item.number ?? item.id ?? "";
    const sceneNumber = Number(String(sceneRaw).match(/\d+/)?.[0] || sceneRaw || 0);
    const promptKey = promptKeys.find((key) => item[key] != null);
    const prompt = String(promptKey ? item[promptKey] : "").trim();
    if (!sceneNumber || !prompt) continue;
    rows.push({ sceneNumber, prompt });
  }
  return rows;
}

export function applyBuilderManagedFx(prompt, presetValue = "", customJson = "") {
  let text = String(prompt || "").trim();
  if (!text || !presetValue) return text;
  const timestampPattern = /((?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|\[\s*Shot\s+\d+[^\]]*\])\s*\n?)([\s\S]*?)(?=\n\s*(?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|\[\s*Shot\s+\d+[^\]]*\])|\n\s*(?:Audio(?:\s+1)?|overall_soundscape|non_diegetic_music|Continuity)\s*:|$)/gi;
  let index = 0;
  let matched = false;
  text = text.replace(timestampPattern, (whole, header, body) => {
    matched = true;
    const contract = storyboardFxContract(presetValue, customJson, index++);
    if (!contract || String(body || "").includes(contract.cue)) return whole;
    return `${header}${String(body || "").trim()} FX accent: ${contract.cue} Keep the mapped subject stable and readable.\n`;
  });
  if (matched) return text.trim();
  const contract = storyboardFxContract(presetValue, customJson, 0);
  return contract ? `${text}\n\nFX accent inside this shot: ${contract.cue} Keep the mapped subject stable and readable.`.trim() : text;
}

export function isRecoverableStoryboardBatchError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return [
    "did not return valid json shot descriptions",
    "returned 0 shot descriptions",
    "returned an invalid number of shot descriptions",
    "returned an empty",
    "request timed out",
    "backend may still be processing",
    "repeated/thought",
    "unfilled template",
    "placeholder",
  ].some((phrase) => message.includes(phrase));
}

export function showStoryboardBatchFailures(failures, retryHandler) {
  const items = Array.isArray(failures) ? failures : [];
  if (!items.length) return;
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:18px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(980px,calc(100vw - 36px));max-height:calc(100vh - 36px);overflow:auto;border:1px solid #991b1b;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 22px 80px rgba(0,0,0,.65);padding:16px;box-sizing:border-box;";
  const title = document.createElement("div");
  title.innerHTML = `<div style="font-size:17px;font-weight:900;color:#fecaca;">Storyboard skipped ${items.length} scene${items.length === 1 ? "" : "s"}</div><div style="font-size:12px;color:#cbd5e1;margin-top:5px;">Successful scenes were saved. Only these scenes will be retried.</div>`;
  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:10px;margin-top:14px;";
  items.forEach((item) => {
    const card = document.createElement("details");
    card.open = true;
    card.style.cssText = "border:1px solid #7f1d1d;border-radius:7px;background:#1f0808;padding:9px;";
    const summary = document.createElement("summary");
    summary.style.cssText = "cursor:pointer;font-weight:900;color:#fca5a5;";
    summary.textContent = `${item.scene.label || `Scene ${item.scene.scene_number || "?"}`}: ${item.error}`;
    const raw = document.createElement("pre");
    raw.style.cssText = "white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto;margin:9px 0 0;color:#fecaca;font-size:11px;line-height:1.4;";
    raw.textContent = item.error;
    card.append(summary, raw);
    list.append(card);
  });
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:16px;";
  const close = makeButton("Close");
  const retry = makeButton(`Retry ${items.length} Failed Scene${items.length === 1 ? "" : "s"}`, "primary");
  retry.onclick = async () => {
    retry.disabled = true;
    retry.textContent = "Retrying...";
    try {
      backdrop.remove();
      await retryHandler(items);
    } catch (error) {
      createToast(String(error?.message || error), true);
      retry.disabled = false;
      retry.textContent = `Retry ${items.length} Failed Scene${items.length === 1 ? "" : "s"}`;
    }
  };
  close.onclick = () => backdrop.remove();
  actions.append(close, retry);
  box.append(title, list, actions);
  backdrop.append(box);
  document.body.append(backdrop);
}

export function createPromptGeneration({
  currentRows, gemmaAllButton, getSelectedScenes, keepGemmaLoadedInput, promptRunnerGenericName,
  promptRunnerName, renderTable, saveStoryboard, state, syncReferenceMappingsToVideoCreator,
}) {
  async function clearAllStoryboardPrompts() {
    const confirmed = await confirmClearStoryboardPrompts();
    if (!confirmed) return;
    let changed = 0;
    for (const scene of state.scenes) {
      const before = [
        scene.prompt_summary,
        scene.motion_summary,
        scene.image_prompt,
        scene.video_prompt,
        scene.notes,
      ].map((value) => String(value || "")).join("\n");
      scene.prompt_summary = "";
      scene.motion_summary = "";
      scene.image_prompt = "";
      scene.video_prompt = "";
      scene.video_prompt_origin = "manual";
      scene.notes = "";
      if (scene.status && scene.status !== "draft") scene.status = "draft";
      const after = [
        scene.prompt_summary,
        scene.motion_summary,
        scene.image_prompt,
        scene.video_prompt,
        scene.notes,
      ].map((value) => String(value || "")).join("\n");
      if (before !== after) changed += 1;
    }
    renderTable();
    syncReferenceMappingsToVideoCreator();
    if (state.projectFolder) {
      try {
        await postJson("/vrgdg/storyboard/save", {
          project_folder: state.projectFolder,
          storyboard: slimStoryboardForRequest(state),
        });
        createToast(`Cleared prompts/notes in ${changed} scene${changed === 1 ? "" : "s"} and saved Storyboard.`);
      } catch (error) {
        createToast(`Cleared prompts/notes in this session, but could not save Storyboard:\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast(`Cleared prompts/notes in ${changed} scene${changed === 1 ? "" : "s"}. Save the project to keep this change.`);
    }
  }

  function openImportImagePromptsFromGptModal() {
    const isVideo = state.mode === "image_to_video_prep";
    const kindLabel = isVideo ? "Video" : "Image";
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100013;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(840px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = isVideo
      ? `<div style="font-size:17px;font-weight:900;color:#cffafe;">Import Video Prompts From GPT</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Paste the JSON code block from the video prompt GPT. This updates Video Prep prompts only.</div>`
      : `<div style="font-size:17px;font-weight:900;color:#cffafe;">Import Image Prompts From GPT</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Paste the JSON code block from the Krea 2 text-to-image GPT. This updates Image Prep prompts only.</div>`;
    const importClose = makeButton("Close");
    importHeader.append(importTitle, importClose);
    const help = document.createElement("div");
    help.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    help.innerHTML = `Accepted examples:<br><code>[{"scene":1,"${isVideo ? "video_prompt" : "image_prompt"}":"..."},{"scene":2,"prompt":"..."}]</code><br><code>{"scene1":"prompt text","scene2":"prompt text"}</code>`;
    const input = document.createElement("textarea");
    input.placeholder = "Paste GPT JSON output here...";
    input.spellcheck = false;
    input.style.cssText = "min-height:340px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const apply = makeButton(`Import ${kindLabel} Prompts`, "purple");
    actions.append(cancel, apply);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto;";
    body.append(help, input, status, actions);
    importBox.append(importHeader, body);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    const closeImport = () => importBackdrop.remove();
    importClose.onclick = closeImport;
    cancel.onclick = closeImport;
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) closeImport();
    });
    apply.onclick = () => {
      try {
        const rows = isVideo ? parseVideoPromptImportJson(input.value) : parseImagePromptImportJson(input.value);
        if (!rows.length) throw new Error(`No usable ${kindLabel.toLowerCase()} prompts found. Make sure each row has a scene number and ${isVideo ? "video_prompt" : "image_prompt"} or prompt.`);
        let updated = 0;
        const missing = [];
        for (const row of rows) {
          const scene = state.scenes.find((item) => Number(item.scene_number) === Number(row.sceneNumber));
          if (!scene) {
            missing.push(row.sceneNumber);
            continue;
          }
          if (isVideo) {
            scene.video_prompt = row.prompt;
            scene.video_prompt_origin = "manual";
            scene.status = "video_prompt_ready";
          } else {
            scene.image_prompt = row.prompt;
            scene.prompt_summary = "";
            scene.status = "image_prompt_ready";
          }
          updated += 1;
        }
        renderTable();
        status.textContent = `Updated ${updated} ${kindLabel} Prep prompt${updated === 1 ? "" : "s"}${missing.length ? `; missing scenes: ${missing.join(", ")}` : ""}.`;
        status.style.color = updated ? "#67e8f9" : "#fbbf24";
        createToast(`Imported ${updated} ${kindLabel.toLowerCase()} prompt${updated === 1 ? "" : "s"} from GPT.`);
        if (updated) closeImport();
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
        createToast(String(error?.message || error), true);
      }
    };
    input.focus();
  }

  function storyboardGemmaPayload(scene, overrides = {}) {
    const payload = storyboardGptPayload(state, [scene]);
    const imageStyle = normalizeStoryLayer(state.storyLayer);
    return {
      ...(state.gemmaSettings || {}),
      ...overrides,
      storyboard_payload: payload,
      image_world_style: imageStyle.image_world_style,
      image_custom_style_direction: imageStyle.image_custom_style_direction,
      max_new_tokens: 2000,
      temperature: 0.35,
      top_p: 0.90,
    };
  }

  async function createSceneImagePromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending image scene card to ${runnerName}...\nThis creates the text-to-image prompt for Image Prep.`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/gemma_image_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter, max_new_tokens: 1200 }), STORYBOARD_GEMMA_TIMEOUT_MS);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: ${genericName} response received.\nRunner: ${data.runner || runnerName}\nSaving image prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const prompt = ensureStoryboardReferenceOpening(applyStoryboardTriggerPhrases(data.prompt, scene), scene, state.imageMode);
      if (!prompt) throw new Error(`${genericName} returned an empty Storyboard image prompt.`);
      scene.image_prompt = prompt;
      scene.prompt_summary = "";
      scene.status = "image_prompt_ready";
      if (!quiet) createToast(`${genericName} created image prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || runnerName}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`${genericName} Storyboard image prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  function enforceStoryboardVideoFacialRequirements(prompt, scene) {
    let text = String(prompt || "").trim();
    const normalized = normalizeScene(scene, 0);
    const promptMentionsFace = /\b(?:woman|man|girl|boy|person|subject|singer|rapper|performer|speaker|character|face|eyes?|brows?|gaze|mouth|jaw|cheeks?|expression|smile|frown|sings?|singing|says|speaks?)\b/i.test(text);
    const hasCharacter = !normalized.no_character_present && (
      (Array.isArray(normalized.subject_refs) && normalized.subject_refs.length)
      || (Array.isArray(normalized.subjects) && normalized.subjects.length)
      || promptMentionsFace
    );
    if (!text || !hasCharacter) return text;
    const vocalStatus = normalized.vocal_status || {};
    const promptSaysSinging = /\b(?:sings?|singing|raps?|rapping)\b/i.test(text);
    const isSinging = promptSaysSinging || (String(normalized.performance_mode || vocalStatus.performance_mode || state.performanceMode || "").trim() === "singing"
      && vocalStatus.should_lip_sync !== false
      && !vocalStatus.instrumental
      && !vocalStatus.no_lip_sync
      && !normalized.lyric_no_lip_sync
      && Boolean(String(vocalStatus.lyric_text || normalized.lyrics || "").trim()));
    if (isSinging) {
      text = text
        .replace(/\bwith\s+a\s+quiet,\s*internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bwith\s+quiet\s+internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bquiet,\s*internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+intensity\b/gi, "controlled intensity")
        .replace(/\bquiet\s+performance\b/gi, "controlled performance")
        .replace(/\bquiet\s+emotion\b/gi, "restrained emotion")
        .replace(/\bquiet\s+singing\b/gi, "focused singing");
    }
    const hasBlink = /\bblink\w*\b/i.test(text);
    const hasEyeMovement = /\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b/i.test(text);
    const additions = [];
    if (!hasEyeMovement) additions.push("subtle natural eye movement");
    if (!hasBlink) additions.push("occasional natural blinking");
    if (additions.length) {
      const insert = `, ${additions.join(", ")}`;
      const faceSentence = text.match(/([^.]*(?:face|eyes?|brows?|gaze|expression)[^.]*)(\.)/i);
      if (faceSentence && typeof faceSentence.index === "number") {
        const nextSentence = `${faceSentence[1].trimEnd()}${insert}`;
        text = `${text.slice(0, faceSentence.index)}${nextSentence}${text.slice(faceSentence.index + faceSentence[1].length)}`;
      } else {
        text = `${text.replace(/\.+\s*$/, "")} with ${additions.join(", ")}.`;
      }
    }
    return text.replace(/\s{2,}/g, " ").trim();
  }

  function applyStoryboardTriggerPhrases(prompt, scene) {
    let text = enforceStoryboardVideoFacialRequirements(prompt, scene);
    const normalized = normalizeScene(scene, 0);
    const refs = normalizeReferenceBuilderCatalog(state.referenceBuilder || {});
    const parts = { start: [], end: [] };
    const add = (trigger, position = "start") => {
      const value = String(trigger || "").trim();
      if (!value) return;
      const key = position === "end" ? "end" : "start";
      if (!parts[key].some((item) => item.toLowerCase() === value.toLowerCase())) parts[key].push(value);
    };
    const subjectPosition = refs.subject_trigger_position === "end" ? "end" : "start";
    const locationPosition = refs.location_trigger_position === "end" ? "end" : "start";
    (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : []).forEach((subject) => {
      add(subject.trigger_phrase || subject.trigger || subject.Trigger, subjectPosition);
    });
    if (normalized.location_ref) {
      add(normalized.location_ref.trigger_phrase || normalized.location_ref.trigger || normalized.location_ref.Trigger, locationPosition);
    }
    add(normalized.trigger_phrase || normalized.trigger || normalized.Trigger, normalized.trigger_position === "end" ? "end" : "start");
    const stripBoundaryTrigger = (value, trigger) => {
      let current = String(value || "").trim();
      const escaped = String(trigger || "").trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!escaped) return current;
      const leading = new RegExp(`^\\s*${escaped}\\s*(?:,\\s*)?`, "i");
      const trailing = new RegExp(`(?:,\\s*)?${escaped}\\s*$`, "i");
      let previous = "";
      while (current && current !== previous) {
        previous = current;
        current = current.replace(leading, "").replace(trailing, "").trim();
      }
      return current;
    };
    [...parts.start, ...parts.end]
      .sort((a, b) => b.length - a.length)
      .forEach((trigger) => {
        text = stripBoundaryTrigger(text, trigger);
      });
    if (parts.start.length) {
      const prefix = parts.start.join(", ");
      if (!text.toLowerCase().startsWith(prefix.toLowerCase())) text = text ? `${prefix}, ${text}` : prefix;
    }
    if (parts.end.length) {
      const suffix = parts.end.join(", ");
      if (!text.toLowerCase().endsWith(suffix.toLowerCase())) text = text ? `${text}, ${suffix}` : suffix;
    }
    return text;
  }

  async function createSceneVideoPromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending scene card to ${runnerName}...\nThis can take a minute depending on runner/model speed.`, progressPercent);
      const callbackPayload = storyboardGptPayload(state, [scene]);
      if (state.onBeforeCreateVideoPrompt) {
        await state.onBeforeCreateVideoPrompt(scene, {
          storyboardPayload: callbackPayload,
          progress,
          progressPercent,
          progressLabel,
        });
      }
      if (state.projectVideoEngine === "minimax_h3" && !state.onCreateVideoPrompt) {
        throw new Error("Open Storyboard Builder from the Video Builder so MiniMax can use the scene's H3 mode, ordered references, exact timing, and LLM instructions.");
      }
      const data = state.onCreateVideoPrompt
        ? await state.onCreateVideoPrompt(scene, {
          unloadAfter,
          storyboardPayload: callbackPayload,
          progress,
          progressPercent,
          progressLabel,
        })
        : await postJson("/vrgdg/storyboard/gemma_video_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter }), STORYBOARD_GEMMA_TIMEOUT_MS);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: ${genericName} response received.\nRunner: ${data.runner || runnerName}\nSaving prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const rawPrompt = String(data?.prompt || data || "").trim();
      const prompted = data?.already_finalized ? rawPrompt : applyStoryboardTriggerPhrases(rawPrompt, scene);
      const prompt = data?.already_finalized ? prompted : applyBuilderManagedFx(prompted, state.fxPreset, state.fxCustomJson);
      if (!prompt) throw new Error(`${genericName} returned an empty Storyboard video prompt.`);
      scene.video_prompt = prompt;
      scene.video_prompt_origin = "gemma";
      scene.status = "video_prompt_ready";
      if (!quiet) createToast(`${genericName} created video prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || runnerName}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`${genericName} Storyboard prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  async function createScenePromptForActiveMode(scene, options = {}) {
    return state.mode === "image_to_video_prep"
      ? createSceneVideoPromptWithGemma(scene, options)
      : createSceneImagePromptWithGemma(scene, options);
  }

  function choosePromptGenerationScope(scenes = [], selected = false) {
    return new Promise((resolve) => {
    const promptKind = state.mode === "image_to_video_prep" ? "video" : "image";
    const promptField = `${promptKind}_prompt`;
    const scopeLabel = selected ? "selected" : "visible";
    const missingCount = scenes.filter((scene) => !String(scene[promptField] || "").trim()).length;
    const completedCount = scenes.length - missingCount;
    const runnerName = promptRunnerName();
    const choiceBackdrop = document.createElement("div");
    choiceBackdrop.style.cssText = "position:fixed;inset:0;z-index:100060;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:22px;box-sizing:border-box;";
    const panel = document.createElement("div");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-labelledby", "vrgdg-video-all-choice-title");
    panel.style.cssText = "width:min(650px,calc(100vw - 44px));border:1px solid #155e75;border-radius:11px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 90px rgba(0,0,0,.7);overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "padding:16px 18px;background:#083f4f;border-bottom:1px solid #155e75;";
    const title = document.createElement("div");
    title.id = "vrgdg-video-all-choice-title";
    title.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    title.textContent = `${runnerName} ${promptKind === "video" ? "Video" : "Image"} ${selected ? "Selected" : "All"}`;
    const subtitle = document.createElement("div");
    subtitle.style.cssText = "margin-top:4px;color:#bae6fd;font-size:12px;line-height:1.4;";
    subtitle.textContent = `Choose whether to preserve completed ${promptKind} prompts or regenerate every ${scopeLabel} scene.`;
    header.append(title, subtitle);
    const body = document.createElement("div");
    body.style.cssText = "padding:18px;display:flex;flex-direction:column;gap:14px;";
    const counts = document.createElement("div");
    counts.style.cssText = "border:1px solid #334155;border-radius:8px;background:#07111f;padding:12px;color:#e2e8f0;font-weight:800;line-height:1.45;";
    counts.textContent = `${missingCount} missing  •  ${completedCount} already complete  •  ${scenes.length} total ${scopeLabel} scene${scenes.length === 1 ? "" : "s"}`;
    const guidance = document.createElement("div");
    guidance.style.cssText = "color:#cbd5e1;font-size:13px;line-height:1.5;";
    guidance.textContent = `Only Missing keeps existing ${promptKind} prompts. Replace Existing regenerates every ${scopeLabel} scene, including manually edited prompts.`;
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;";
    const onlyMissing = makeButton(`Only Missing (${missingCount})`, missingCount > 0 ? "primary" : "");
    onlyMissing.style.minHeight = "46px";
    onlyMissing.disabled = missingCount === 0;
    onlyMissing.title = missingCount
      ? `Keep completed prompts and create only missing ${promptKind} prompts.`
      : `Every ${scopeLabel} scene already has an ${promptKind} prompt.`;
    const redoAll = makeButton(`Replace Existing (${scenes.length})`);
    redoAll.style.cssText += "min-height:46px;border-color:#d97706;background:#78350f;color:#fef3c7;";
    redoAll.title = `Replace ${promptKind} prompts for the ${scopeLabel} scenes.`;
    const cancel = makeButton("Cancel");
    cancel.style.cssText += "grid-column:1 / -1;min-height:40px;";
    actions.append(onlyMissing, redoAll, cancel);
    body.append(counts, guidance, actions);
    panel.append(header, body);
    choiceBackdrop.append(panel);
    document.body.append(choiceBackdrop);
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finish(null);
      }
    };
    const finish = (choice) => {
      document.removeEventListener("keydown", onKeyDown, true);
      choiceBackdrop.remove();
      resolve(choice);
    };
    onlyMissing.onclick = () => finish("missing");
    redoAll.onclick = () => finish("all");
    cancel.onclick = () => finish(null);
    choiceBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === choiceBackdrop) finish(null);
    });
    document.addEventListener("keydown", onKeyDown, true);
    requestAnimationFrame(() => (missingCount ? onlyMissing : redoAll).focus());
  });
  }

  async function createAllPromptsWithGemma({ onlyMissing = false, failedSceneIds = [] } = {}) {
    const visibleScenes = currentRows();
    if (!visibleScenes.length) {
      createToast("No storyboard scenes found.", true);
      return;
    }
    const videoMode = state.mode === "image_to_video_prep";
    const promptKind = videoMode ? "video" : "image";
    const promptField = videoMode ? "video_prompt" : "image_prompt";
    const failedIds = new Set(failedSceneIds.map((value) => String(value)));
    const scenes = failedIds.size
      ? visibleScenes.filter((scene) => failedIds.has(String(scene.id || "")))
      : onlyMissing
      ? visibleScenes.filter((scene) => !String(scene[promptField] || "").trim())
      : visibleScenes;
    if (!scenes.length) {
      createToast(`Every visible scene already has a ${promptKind} prompt.`);
      return;
    }
    gemmaAllButton.disabled = true;
    const previousText = gemmaAllButton.textContent;
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    const progress = createStoryboardProgressWindow(`Storyboard ${runnerName} All`);
    let created = 0;
    const failures = [];
    try {
      const keepLoaded = Boolean(keepGemmaLoadedInput.checked);
      progress.set(`${failedIds.size ? "Retrying failed Storyboard scenes" : `Starting Storyboard ${runnerName} All`}...\nMode: ${videoMode ? "Video Prep" : "Image Prep"}\nScope: ${failedIds.size ? `failed only (${scenes.length})` : onlyMissing ? `only missing (${scenes.length} of ${visibleScenes.length})` : `redo all (${scenes.length})`}\nKeep local LLM loaded: ${keepLoaded ? "yes" : "no"}`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        gemmaAllButton.textContent = `${runnerName} ${index + 1}/${scenes.length}`;
        const unloadAfter = keepLoaded ? index === scenes.length - 1 : true;
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        const label = `${runnerName} All ${index + 1}/${scenes.length}: ${scenes[index].label || `Scene ${scenes[index].scene_number || index + 1}`}`;
        try {
          progress.set(`${label}\nCreating storyboard ${promptKind} prompt...`, base);
          await createScenePromptForActiveMode(scenes[index], { quiet: true, unloadAfter, progress, progressPercent: base, progressLabel: label });
          created += 1;
        } catch (error) {
          if (!isRecoverableStoryboardBatchError(error)) throw error;
          failures.push({ scene: scenes[index], error: String(error?.message || error) });
          progress.set(`${label} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      progress.set("Saving storyboard prompts...", 96);
      await saveStoryboard();
      progress.set(`${runnerName} All complete.\nCreated ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}${failures.length ? `. ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped` : ""}${onlyMissing ? "; existing prompts were preserved" : ""}.`, 100);
      progress.close(1800);
      createToast(`${genericName} created ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}${onlyMissing ? "; existing prompts were preserved" : ""}.`, Boolean(failures.length));
      if (failures.length) {
        showStoryboardBatchFailures(failures, (items) => createAllPromptsWithGemma({
          failedSceneIds: items.map((item) => item.scene.id),
        }));
      }
    } catch (error) {
      if (created > 0) {
        progress.set(`Saving ${created} completed prompt${created === 1 ? "" : "s"} before stopping...`, 96);
        await saveStoryboard();
      }
      progress.set(`${runnerName} All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, 100);
      createToast(`${runnerName} All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, true);
    } finally {
      gemmaAllButton.disabled = false;
      gemmaAllButton.textContent = previousText;
      renderTable();
    }
  }

  async function startAllPromptsWithGemma() {
    let selectedScenes = getSelectedScenes();
    const scenes = selectedScenes.length ? selectedScenes : currentRows();
    if (!scenes.length) {
      createToast("No storyboard scenes found.", true);
      return;
    }
    const scope = await choosePromptGenerationScope(scenes, selectedScenes.length > 0);
    if (!scope) return;
    if (scope === "missing" && selectedScenes.length) {
      const field = state.mode === "image_to_video_prep" ? "video_prompt" : "image_prompt";
      selectedScenes = selectedScenes.filter((scene) => !String(scene[field] || "").trim());
      if (!selectedScenes.length) return;
    }
    if (selectedScenes.length > 0) {
      const runnerName = promptRunnerName();
      const genericName = promptRunnerGenericName();
      const videoMode = state.mode === "image_to_video_prep";
      const promptKind = videoMode ? "video" : "image";
      const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
      let created = 0;
      const failures = [];
      const keepLoaded = Boolean(keepGemmaLoadedInput.checked);
      const previousText = gemmaAllButton.textContent;
      gemmaAllButton.disabled = true;
      try {
        progress.set(`Starting Storyboard ${runnerName} for ${selectedScenes.length} selected scene${selectedScenes.length === 1 ? "" : "s"}...\nMode: ${videoMode ? "Video Prep" : "Image Prep"}\nKeep local LLM loaded: ${keepLoaded ? "yes" : "no"}`, 5);
        for (let index = 0; index < selectedScenes.length; index += 1) {
          gemmaAllButton.textContent = `${runnerName} ${index + 1}/${selectedScenes.length}`;
          const unloadAfter = keepLoaded ? index === selectedScenes.length - 1 : true;
          const base = 8 + Math.round((index / Math.max(1, selectedScenes.length)) * 84);
          const label = `${runnerName} ${index + 1}/${selectedScenes.length}: ${selectedScenes[index].label || `Scene ${selectedScenes[index].scene_number || index + 1}`}`;
          try {
            progress.set(`${label}\nCreating storyboard ${promptKind} prompt...`, base);
            await createScenePromptForActiveMode(selectedScenes[index], { quiet: true, unloadAfter, progress, progressPercent: base, progressLabel: label });
            created += 1;
          } catch (error) {
            if (!isRecoverableStoryboardBatchError(error)) throw error;
            failures.push({ scene: selectedScenes[index], error: String(error?.message || error) });
            progress.set(`${label} skipped. Continuing with remaining selected scenes...`, base);
          }
        }
        progress.set("Saving storyboard prompts...", 96);
        await saveStoryboard();
        progress.set(`${runnerName} complete.\nCreated ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}${failures.length ? `. ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped` : ""}.`, 100);
        progress.close(1800);
        createToast(`${genericName} created ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      } catch (error) {
        if (created > 0) {
          progress.set(`Saving ${created} completed prompt${created === 1 ? "" : "s"} before stopping...`, 96);
          await saveStoryboard();
        }
        progress.set(`${runnerName} stopped after ${created}/${selectedScenes.length} scenes:\n${String(error?.message || error)}`, 100);
        createToast(`${runnerName} stopped after ${created}/${selectedScenes.length} scenes:\n${String(error?.message || error)}`, true);
      } finally {
        gemmaAllButton.disabled = false;
        gemmaAllButton.textContent = previousText;
        renderTable();
      }
      return;
    }
    await createAllPromptsWithGemma({ onlyMissing: scope === "missing" });
  }

  return {
    clearAllStoryboardPrompts, createScenePromptForActiveMode, enforceStoryboardVideoFacialRequirements,
    openImportImagePromptsFromGptModal, startAllPromptsWithGemma,
  };
}
