import { postJson } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, toast } from "./controls.mjs";
import { sceneConceptPromptText } from "./image_prompts.mjs";
import { isInstrumentalLyricText, normalizeGemmaContextLimit, normalizeOutputTokenLimit } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { newSegment, normalizeVideoPromptOrigin } from "./segments.mjs";

export async function loadContextTextQuiet(path) {
  const filePath = String(path || "").trim();
  if (!filePath) return "";
  try {
    const data = await postJson("/vrgdg/music_builder/load_text_file", { path: filePath });
    return String(data.content || "").trim();
  } catch {
    return "";
  }
}

function formatSrtTimestamp(seconds) {
  const totalMs = Math.max(0, Math.round(Number(seconds || 0) * 1000));
  const ms = totalMs % 1000;
  const totalSeconds = Math.floor(totalMs / 1000);
  const sec = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const min = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  return `${String(hours).padStart(2, "0")}:${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

export function buildPromptCreatorSrtText(segments = []) {
  return [...segments]
    .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
    .map((segment, index) => {
      const start = Number(segment.start || 0);
      const end = Math.max(start + 0.1, Number(segment.end || start + 4));
      const label = String(segment.label || `SCENE ${index + 1}`).trim() || `SCENE ${index + 1}`;
      return `${index + 1}\n${formatSrtTimestamp(start)} --> ${formatSrtTimestamp(end)}\n${label}`;
    })
    .join("\n\n") + "\n";
}

function cleanPromptCreatorLyricText(text) {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .join(" ")
    .trim();
}

export function buildPromptCreatorLyricSegments(segments = []) {
  const mapping = {};
  const sorted = [...segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
  sorted.forEach((segment, index) => {
    mapping[`segment${index + 1}`] = cleanPromptCreatorLyricText(segment.lyric_text || "");
  });
  return mapping;
}

export function buildPromptCreatorWhisperPreview(lyricMapping = {}) {
  const entries = Object.entries(lyricMapping)
    .map(([key, value]) => {
      const match = String(key || "").match(/(\d+)/);
      return { index: match ? Number(match[1]) : 0, value: String(value || "") };
    })
    .filter((item) => item.index > 0)
    .sort((a, b) => a.index - b.index);
  return `# Lyrics to fix: (${entries.length} segments)\n\n${entries.map((item) => `lyricSegment${item.index}=${item.value}`).join("\n")}`;
}

export function lyricsFromSegmentsForPromptCreator(segments = []) {
  return [...segments]
    .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
    .map((segment) => cleanPromptCreatorLyricText(segment.lyric_text || ""))
    .filter((text) => text && !isInstrumentalLyricText(text))
    .join("\n");
}

function splitStorySourceIntoChunks(text, count) {
  const cleaned = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!cleaned) return [];
  const stanzas = cleaned.split(/\n\s*\n+/).map((item) => item.trim()).filter(Boolean);
  let units = stanzas.length >= count ? stanzas : cleaned.split("\n").map((item) => item.trim()).filter(Boolean);
  if (!units.length) units = [cleaned];
  const chunkCount = Math.max(1, Math.min(120, Number(count || units.length || 1)));
  const chunks = [];
  for (let index = 0; index < chunkCount; index += 1) {
    const start = Math.floor(index * units.length / chunkCount);
    const end = Math.max(start + 1, Math.floor((index + 1) * units.length / chunkCount));
    chunks.push(units.slice(start, end).join("\n"));
  }
  return chunks;
}

function promptKindLabel(kind) {
  if (kind === "minimax") return "MiniMax H3 video";
  return kind === "i2v" ? "image-to-video" : "text-to-image";
}

function segmentPromptFindReplaceTargets(segment, targetKinds = []) {
  const targets = [];
  const hasKind = (kind) => targetKinds.includes(kind);
  if (hasKind("t2i")) {
    targets.push(
      { key: "t2i_prompt", label: "T2I prompt" },
      { key: "flux_prompt", label: "Flux/Klein prompt" },
      { key: "nb_prompt", label: "Nano B prompt" },
      { key: "flow_gpt_prompt", label: "Flow/GPT prompt" },
      { key: "ernie_t2i_prompt", label: "Ernie prompt" },
      { key: "krea2_t2i_prompt", label: "Krea 2 prompt" },
    );
  }
  if (hasKind("i2v")) {
    targets.push({ key: "i2v_prompt", label: "I2V / T2V / video prompt" });
    targets.push({ key: "t2v_prompt", label: "T2V prompt" });
  }
  if (hasKind("minimax")) {
    targets.push({ key: "minimax_h3_prompt", label: "MiniMax H3 prompt" });
    targets.push({ key: "minimax_h3_pass2_prompt", label: "2nd Pass Prompt" });
  }
  const seen = new Set();
  return targets.filter((target) => {
    if (seen.has(target.key)) return false;
    seen.add(target.key);
    return Object.prototype.hasOwnProperty.call(segment || {}, target.key) || String(segment?.[target.key] || "").trim();
  });
}

function buildPromptFindReplaceRegex(findText, { caseSensitive = false } = {}) {
  const escaped = String(findText || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])`, caseSensitive ? "gu" : "giu");
}

function promptKindPrefix(kind) {
  return kind === "i2v" ? "I2V" : "Prompt";
}

function numericPromptKey(key, kind) {
  const text = String(key || "");
  const expected = kind === "i2v" ? /^(?:i2v|motion|prompt)\s*(\d+)$/i : /^(?:prompt|t2i|image)\s*(\d+)$/i;
  const match = text.match(expected) || text.match(/(\d+)/);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function parsePromptTextBlocks(text, kind) {
  const raw = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item || "").trim());
    }
    if (parsed && typeof parsed === "object") {
      if (kind === "minimax" && Array.isArray(parsed.scenes)) {
        return parsed.scenes.map((scene) => ({
          prompt: String(scene?.prompt || scene?.video_prompt || "").trim(),
          pass2_prompt: Object.prototype.hasOwnProperty.call(scene || {}, "pass2_prompt")
            || Object.prototype.hasOwnProperty.call(scene || {}, "minimax_h3_pass2_prompt")
            ? String(scene?.pass2_prompt || scene?.minimax_h3_pass2_prompt || "")
            : undefined,
        }));
      }
      return Object.keys(parsed)
        .sort((a, b) => numericPromptKey(a, kind) - numericPromptKey(b, kind))
        .map((key) => String(parsed[key] || "").trim());
    }
  } catch (_error) {
    // Not JSON; try key/value and then the normal blank-line format.
  }
  const keyed = [];
  const keyPattern = kind === "i2v"
    ? /^\s*(?:I2V|Motion|Prompt)\s*(\d+)\s*[:=]\s*(.*)\s*$/i
    : /^\s*(?:Prompt|T2I|Image)\s*(\d+)\s*[:=]\s*(.*)\s*$/i;
  for (const line of raw.split("\n")) {
    const match = line.match(keyPattern);
    if (!match) continue;
    keyed.push({ index: Number(match[1]), value: String(match[2] || "").trim() });
  }
  if (keyed.length) {
    keyed.sort((a, b) => a.index - b.index);
    return keyed.map((item) => item.value);
  }
  return raw.split(/\n\s*\n+/).map((item) => item.trim()).filter(Boolean);
}

async function loadPromptTextFile(path, fallback = "") {
  const filePath = String(path || "").trim();
  if (!filePath) return fallback;
  try {
    const data = await postJson("/vrgdg/music_builder/load_text_file", { path: filePath });
    return String(data.content || "");
  } catch (error) {
    const message = String(error?.message || error);
    if (/not found|was not found|cannot find/i.test(message)) return fallback;
    throw error;
  }
}

export async function savePromptTextFile(path, content) {
  const filePath = String(path || "").trim();
  if (!filePath) throw new Error("Create or load a project first.");
  return await postJson("/vrgdg/music_builder/save_text_file", { path: filePath, content: String(content || "") });
}

function promptTimestamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function showPromptReloadConfirm(kind, original = false) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = original
      ? `Reload original ${promptKindLabel(kind)} prompts?`
      : `Reload ${promptKindLabel(kind)} prompts?`;
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:9px;font-size:13px;color:#d4d4d8;line-height:1.45;";
    const source = document.createElement("div");
    source.textContent = original
      ? "This reads the original backup created by Prompt Options."
      : "This reads the current prompt text file in the project prompts folder.";
    const effect = document.createElement("div");
    effect.textContent = kind === "minimax"
      ? "It updates the MiniMax H3 prompt boxes for the scenes, then saves the session and video_prompts.json export."
      : kind === "i2v"
        ? "It updates the I2V prompt boxes for the scenes, then saves the project."
        : "It updates the T2I prompt boxes for the scenes, then saves the project.";
    const backup = document.createElement("div");
    backup.textContent = "The current scene prompts are backed up first.";
    body.append(source, effect, backup);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton("Reload", "primary");
    cancel.onclick = () => {
      backdrop.remove();
      resolve(false);
    };
    confirm.onclick = () => {
      backdrop.remove();
      resolve(true);
    };
    actions.append(cancel, confirm);
    box.append(heading, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

function showPromptClearConfirm(kind) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #991b1b;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = kind === "i2v" ? "Clear all I2V prompts?" : "Clear all T2I prompts?";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:9px;font-size:13px;color:#d4d4d8;line-height:1.45;";
    const scope = document.createElement("div");
    scope.textContent = kind === "i2v"
      ? "This clears only the saved image-to-video prompt text from every scene."
      : "This clears only the saved text-to-image prompt text from every scene, including model-specific T2I prompt copies.";
    const keep = document.createElement("div");
    keep.textContent = "Images, videos, notes, model settings, LoRAs, reference images, timing, and project paths are not changed.";
    const backup = document.createElement("div");
    backup.textContent = "The current prompt list is backed up first.";
    body.append(scope, keep, backup);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton(kind === "i2v" ? "Yes, clear I2V prompts" : "Yes, clear T2I prompts");
    confirm.style.background = "#991b1b";
    confirm.style.borderColor = "#dc2626";
    confirm.style.color = "#fff";
    cancel.onclick = () => {
      backdrop.remove();
      resolve(false);
    };
    confirm.onclick = () => {
      backdrop.remove();
      resolve(true);
    };
    actions.append(cancel, confirm);
    box.append(heading, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

export function createProjectFiles({
  activeSegment, allEditableSegments, autoSaveSessionQuiet, createProgressWindow, currentVideoMode,
  ernieGemmaModelSelect, ernieTextGemmaModelSelect, fluxGemmaModelSelect, gemmaModelSelect, gemmaRunnerLine,
  i2vGemmaModelSelect, i2vTextGemmaModelSelect, projectInput, pushHistory, render, saveSession,
  sceneDisplayName, sceneVideoConceptPromptText, segmentIndexInfo, state, storyIdeaInput,
  syncI2VMotionJsonFromSegments, syncInspector, syncPromptJsonFromSegments, t2iTextGemmaModelSelect,
  themeStyleInput, videoModeDisplayLabel, zEnhanceGemmaModelSelect,
}) {
  function projectContextPath(filename) {
    const folder = String(projectInput.value || state.projectFolder || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}project_context${separator}${filename}`;
  }

  function projectSceneNotesPath() {
    const folder = String(state.projectFolder || projectInput.value || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}SceneNotes.json`;
  }

  function projectReferenceBuilderLocationsPath() {
    const folder = String(state.projectFolder || projectInput.value || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}ReferenceBuilderLocations.json`;
  }

  async function locationExtractionStyleTheme(extraStyleTheme = "") {
    const parts = [];
    const globalStyleTheme = state.useVrgdgTextContext
      ? await loadContextTextQuiet(themeStyleInput.value || state.themeStylePath)
      : "";
    if (globalStyleTheme) parts.push(`Global theme/style:\n${globalStyleTheme}`);
    const localStyleTheme = String(extraStyleTheme || "").trim();
    if (localStyleTheme) parts.push(`Location extraction notes:\n${localStyleTheme}`);
    return parts.join("\n\n").trim();
  }

  function referenceBuilderSubjectLocationText() {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const parts = [];
    const subjects = (refs.subjects || [])
      .map((subject, index) => {
        const name = String(subject.name || `Character ${index + 1}`).trim();
        const description = String(subject.description || "").trim();
        return description ? `${name}: ${description}` : "";
      })
      .filter(Boolean);
    if (subjects.length) {
      parts.push(`Characters:\n${subjects.join("\n")}`);
    }
    const locations = (refs.locations || [])
      .map((location, index) => {
        const name = String(location.name || `Location ${index + 1}`).trim();
        const description = String(location.description || "").trim();
        return description ? `${name}: ${description}` : name;
      })
      .filter(Boolean);
    if (locations.length) {
      parts.push(`Locations:\n${locations.join("\n")}`);
    }
    return parts.join("\n\n").trim();
  }

  function builderStorySourcePath() {
    if (String(state.builderStorySourcePath || "").trim()) return String(state.builderStorySourcePath || "").trim();
    return projectContextPath("AgentStorySource.txt");
  }

  async function loadBuilderStorySource() {
    const path = builderStorySourcePath();
    if (!path) return "";
    const text = await loadContextTextQuiet(path);
    state.builderStorySourcePreview = text.slice(0, 500);
    if (text && !state.builderStorySourcePath) state.builderStorySourcePath = path;
    return text;
  }

  async function saveBuilderStorySource(text) {
    const path = builderStorySourcePath();
    if (!path) throw new Error("Create or load a project first, then save the Story Builder source.");
    const result = await postJson("/vrgdg/music_builder/save_text_file", { path, content: String(text || "") });
    state.builderStorySourcePath = result.path || path;
    state.builderStorySourcePreview = String(text || "").trim().slice(0, 500);
    await autoSaveSessionQuiet("Story Builder source saved");
    return result.path || path;
  }

  async function createStoryScenesFromSource(sceneCount = 0) {
    const source = await loadBuilderStorySource();
    if (!source.trim()) throw new Error("Add Story Source lyrics/script first.");
    const requestedCount = Number(sceneCount || 0);
    const chunks = splitStorySourceIntoChunks(source, requestedCount > 0 ? requestedCount : 12);
    const totalDuration = Math.max(Number(state.duration || 0), chunks.length * 4);
    const sceneDuration = Math.max(1, totalDuration / Math.max(1, chunks.length));
    const newScenes = chunks.map((chunk, index) => {
      const start = index * sceneDuration;
      const end = index === chunks.length - 1 ? totalDuration : (index + 1) * sceneDuration;
      const segment = newSegment(Number(start.toFixed(3)), Number(end.toFixed(3)));
      segment.label = `Scene ${index + 1}`;
      segment.lyric_text = chunk;
      segment.timeline_note = "";
      segment.notes = "";
      segment.source = "agent_story_builder";
      return segment;
    });
    pushHistory();
    state.segments = newScenes;
    state.overlaySegments = [];
    state.activeTrack = "base";
    state.activeId = newScenes[0]?.id || "";
    state.srtMode = false;
    state.timingFrozen = false;
    state.duration = Math.max(totalDuration, ...newScenes.map((item) => Number(item.end || 0)));
    syncInspector();
    render();
    await syncPromptJsonFromSegments("Story Builder scenes created");
    await syncI2VMotionJsonFromSegments("Story Builder scenes created");
    autoSaveSessionQuiet("Story Builder scenes created");
    return newScenes.length;
  }

  async function editContextTextFile(input, title, filename, gemmaTarget, options = {}) {
    let path = String(input.value || "").trim();
    if (!path) {
      path = projectContextPath(filename);
      if (!path) {
        toast("Create or choose a project folder first, then this editor can create the text file.", true);
        return;
      }
      input.value = path;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    let data;
    try {
      data = await postJson("/vrgdg/music_builder/load_text_file", { path });
    } catch (error) {
      const message = String(error?.message || error);
      if (!/not found|was not found|cannot find/i.test(message)) {
        toast(message, true);
        return;
      }
      data = { path, content: "" };
    }

    const box = document.createElement("div");
    box.style.cssText = `
      position:fixed;left:50%;top:8%;transform:translateX(-50%);
      z-index:100005;width:min(900px,calc(100vw - 36px));height:min(720px,calc(100vh - 56px));
      display:grid;grid-template-rows:auto auto auto minmax(0,1fr) auto;
      border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#e0f2fe;
      box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;
    `;
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid #155e75;background:#083344;";
    const heading = document.createElement("div");
    heading.textContent = title;
    heading.style.cssText = "font-size:13px;font-weight:900;";
    const close = makeButton("Close");
    close.style.padding = "5px 8px";
    header.append(heading, close);
    const pathText = document.createElement("div");
    pathText.textContent = data.path || path;
    pathText.style.cssText = "padding:8px 12px;border-bottom:1px solid #1f2937;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;";
    const gemmaHelp = document.createElement("div");
    const helperByTarget = {
      builder_style_theme: "Gemma uses only the text in this box as the user idea. It unloads the model after the draft is created.",
      builder_story_idea: "Gemma uses the text in this box plus the current theme/style file if one exists. It unloads the model after the draft is created.",
      builder_subjects_and_scenes: "Gemma uses the text in this box plus the current theme/style and story idea files if they exist. It unloads the model after the draft is created.",
    };
    gemmaHelp.textContent = options.helpText || helperByTarget[gemmaTarget] || "Edit this text file, then save it back to the current project.";
    gemmaHelp.style.cssText = "padding:8px 12px;border-bottom:1px solid #1f2937;color:#a1a1aa;font-size:11px;line-height:1.35;background:#0b1120;";
    const textarea = document.createElement("textarea");
    textarea.value = data.content || "";
    textarea.spellcheck = false;
    textarea.style.cssText = "width:100%;height:100%;box-sizing:border-box;border:0;resize:none;background:#020617;color:#fafafa;padding:12px;font-size:12px;line-height:1.45;outline:none;font-family:monospace;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;padding:10px 12px;border-top:1px solid #155e75;background:#111827;";
    const gemma = makeButton("Gemma4 Draft", "primary");
    const save = makeButton("Save", "primary");
    const cancel = makeButton("Cancel");
    if (options.showGemma !== false) actions.append(gemma);
    actions.append(cancel, save);
    box.append(header, pathText, gemmaHelp, textarea, actions);
    document.body.append(box);
    textarea.focus();
    close.onclick = () => box.remove();
    cancel.onclick = () => box.remove();
    gemma.onclick = async () => {
      const idea = String(textarea.value || "").trim();
      const modelFile = String(t2iTextGemmaModelSelect.value || gemmaModelSelect.value || ernieTextGemmaModelSelect.value || ernieGemmaModelSelect.value || zEnhanceGemmaModelSelect.value || i2vTextGemmaModelSelect.value || i2vGemmaModelSelect.value || fluxGemmaModelSelect.value || "").trim();
      if (!modelFile) {
        toast("Choose a Gemma4 model first in the Image tab model settings.", true);
        return;
      }
      if (!idea) {
        toast("Type a rough idea in this box first, then Gemma4 can clean it up.", true);
        return;
      }
      let progress = null;
      try {
        gemma.disabled = true;
        gemma.textContent = "Gemma...";
        progress = createProgressWindow(title.replace(/^Edit\s+/i, "Gemma4 "), { runnerAware: false });
        progress.set(`Creating draft from your notes...\n${gemmaRunnerLine({ forceBuiltin: true })}`, 25);
        const styleTheme = gemmaTarget === "builder_story_idea" || gemmaTarget === "builder_subjects_and_scenes"
          ? await loadContextTextQuiet(themeStyleInput.value)
          : "";
        const storyIdea = gemmaTarget === "builder_subjects_and_scenes"
          ? await loadContextTextQuiet(storyIdeaInput.value)
          : "";
        const data = await postJson("/vrgdg/gemma4/generate", {
          target: gemmaTarget,
          model_file: modelFile,
          notes: idea,
          style_theme: styleTheme,
          story_idea: storyIdea || idea,
          unload_after: true,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          gemma_output_token_limit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
          max_new_tokens: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
        }, 10 * 60 * 1000);
        const text = String(data.text || "").trim();
        if (!text) throw new Error("Gemma4 returned an empty draft.");
        textarea.value = text;
        progress.set(data.unloaded ? "Draft ready. Gemma4 was unloaded." : "Draft ready.", 100);
        progress.close(900);
        toast("Gemma4 draft is ready. Review it, then click Save.");
      } catch (error) {
        progress?.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      } finally {
        gemma.disabled = false;
        gemma.textContent = "Gemma4 Draft";
      }
    };
    save.onclick = async () => {
      try {
        save.disabled = true;
        save.textContent = "Saving...";
        const result = await postJson("/vrgdg/music_builder/save_text_file", { path: data.path || path, content: textarea.value });
        input.value = result.path || data.path || path;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        toast(`Saved text file:\n${result.path || data.path || path}`);
        box.remove();
        if (typeof options.afterSave === "function") {
          await options.afterSave(result, textarea.value);
        }
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        save.disabled = false;
        save.textContent = "Save";
      }
    };
  }

  function projectPromptsPath(filename) {
    const folder = String(projectInput.value || state.projectFolder || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}prompts${separator}${filename}`;
  }

  function projectGemmaDebugPath(filename) {
    const folder = String(projectInput.value || state.projectFolder || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}prompts${separator}gemma_debug${separator}${filename}`;
  }

  async function saveGemmaJunkDebug(error, context = {}) {
    const raw = String(error?.rawGemmaPrompt ?? error?.rawPrompt ?? "").trim();
    const cleaned = String(error?.cleanedGemmaPrompt ?? error?.cleanedPrompt ?? "").trim();
    if (!raw && !cleaned) return "";
    const path = projectGemmaDebugPath(`gemma_junk_${promptTimestamp()}.txt`);
    if (!path) return "";
    const segment = context.segment || activeSegment();
    const info = segment ? segmentIndexInfo(segment) : { index: -1 };
    const content = [
      "VRGDG Gemma junk/debug output",
      `Saved: ${new Date().toISOString()}`,
      `Context: ${context.label || ""}`,
      `Mode: ${videoModeDisplayLabel(currentVideoMode(), true)}`,
      `Scene: ${segment ? sceneDisplayName(segment, info.index) : ""}`,
      `Error: ${String(error?.message || error || "")}`,
      "",
      "===== RAW GEMMA OUTPUT =====",
      raw || "(empty)",
      "",
      "===== CLEANED OUTPUT SEEN BY VALIDATOR =====",
      cleaned || "(empty)",
      "",
      "===== SCENE INPUTS =====",
      `T2I/concept prompt:\n${sceneConceptPromptText(segment) || ""}`,
      "",
      `Video concept fallback:\n${sceneVideoConceptPromptText(segment) || ""}`,
      "",
      `I2V/T2V notes:\n${String(segment?.i2v_notes || "").trim()}`,
      "",
      `Lyrics/vocal line:\n${String(segment?.lyric_text || "").trim()}`,
      "",
      `Performer(s) / speaker(s):\n${Array.isArray(segment?.lyric_singers) ? segment.lyric_singers.join(", ") : String(segment?.lyric_singers || "")}`,
    ].join("\n");
    try {
      const result = await savePromptTextFile(path, content);
      const savedPath = result?.path || path;
      try {
        error.gemmaDebugPath = savedPath;
      } catch (_) {
        // Some thrown values are not extensible; returning the path is enough.
      }
      return savedPath;
    } catch (saveError) {
      console.warn("[VRGDG Music Builder] Failed to save Gemma junk debug output:", saveError);
      return "";
    }
  }

  function projectPromptBackupPath(kind, name) {
    const folder = String(projectInput.value || state.projectFolder || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}prompts${separator}backups${separator}${kind}_prompts_${name}.txt`;
  }

  function promptImageModeForEdit() {
    return state.imageModelMode || "zimage";
  }

  function segmentPromptForEdit(segment, kind) {
    if (kind === "minimax") return segment?.minimax_h3_prompt || "";
    if (kind === "i2v") return segment?.i2v_prompt || "";
    const mode = promptImageModeForEdit();
    if (mode === "flow_gpt") return segment?.flow_gpt_prompt || segment?.nb_prompt || segment?.t2i_prompt || "";
    if (mode === "nano_banana") return segment?.nb_prompt || segment?.t2i_prompt || "";
    if (mode === "flux_klein") return segment?.flux_prompt || segment?.t2i_prompt || "";
    return segment?.t2i_prompt || "";
  }

  function setSegmentPromptForEdit(segment, kind, value, options = {}) {
    const text = String(value || "").trim();
    if (kind === "minimax") {
      segment.minimax_h3_prompt = text;
      segment.minimax_h3_prompt_origin = "manual";
      return;
    }
    if (kind === "i2v") {
      segment.i2v_prompt = text;
      segment.i2v_prompt_origin = normalizeVideoPromptOrigin(options.origin);
      return;
    }
    const mode = promptImageModeForEdit();
    segment.t2i_prompt = text;
    if (mode === "flow_gpt") {
      segment.flow_gpt_prompt = text;
      segment.nb_prompt = text;
    } else if (mode === "nano_banana") {
      segment.nb_prompt = text;
    } else if (mode === "flux_klein") {
      segment.flux_prompt = text;
    } else {
      segment.flux_prompt = text;
      segment.nb_prompt = text;
      segment.flow_gpt_prompt = text;
    }
  }

  function countPromptFindReplaceMatches(findText, targetKinds, options = {}) {
    if (!String(findText || "").trim()) return { matches: 0, scenes: 0, fields: 0 };
    const regex = buildPromptFindReplaceRegex(findText, options);
    let matches = 0;
    const sceneIds = new Set();
    let fields = 0;
    for (const segment of allEditableSegments()) {
      for (const target of segmentPromptFindReplaceTargets(segment, targetKinds)) {
        const text = String(segment?.[target.key] || "");
        if (!text) continue;
        const found = text.match(regex);
        if (!found?.length) continue;
        matches += found.length;
        fields += 1;
        sceneIds.add(segment.id || sceneDisplayName(segment, segmentIndexInfo(segment).index));
      }
    }
    return { matches, scenes: sceneIds.size, fields };
  }

  async function replacePromptPhraseAcrossScenes(findText, replaceText, targetKinds, options = {}) {
    const regex = buildPromptFindReplaceRegex(findText, options);
    let matches = 0;
    let fields = 0;
    const sceneIds = new Set();
    pushHistory();
    for (const segment of allEditableSegments()) {
      let changed = false;
      for (const target of segmentPromptFindReplaceTargets(segment, targetKinds)) {
        const original = String(segment?.[target.key] || "");
        if (!original) continue;
        let localMatches = 0;
        const replaced = original.replace(regex, () => {
          localMatches += 1;
          return String(replaceText || "");
        });
        if (!localMatches || replaced === original) continue;
        segment[target.key] = replaced.trim();
        if (target.key === "i2v_prompt" || target.key === "t2v_prompt") {
          segment.i2v_prompt_origin = "manual";
        }
        matches += localMatches;
        fields += 1;
        changed = true;
      }
      if (changed) sceneIds.add(segment.id || sceneDisplayName(segment, segmentIndexInfo(segment).index));
    }
    syncInspector();
    render();
    await autoSaveSessionQuiet("prompt find replace");
    if (targetKinds.includes("minimax")) await saveMiniMaxPromptExport();
    return { matches, scenes: sceneIds.size, fields };
  }

  function promptKindFile(kind) {
    if (kind === "minimax") return projectPromptsPath("video_prompts.json");
    return projectPromptsPath(kind === "i2v" ? "i2v_prompts.txt" : "t2i_prompts.txt");
  }

  function formatPromptBlocks(kind, prompts = null) {
    const values = Array.isArray(prompts)
      ? prompts
      : allEditableSegments().map((segment) => segmentPromptForEdit(segment, kind));
    if (kind === "minimax") {
      const segments = allEditableSegments();
      const scenes = segments.map((segment, index) => {
        const provided = Array.isArray(prompts) ? prompts[index] : null;
        const promptText = provided != null && typeof provided === "object"
          ? String(provided.prompt || "").trim()
          : provided != null
            ? String(provided || "").trim()
            : String(segmentPromptForEdit(segment, kind) || "").trim();
        const pass2Text = provided != null && typeof provided === "object" && Object.prototype.hasOwnProperty.call(provided, "pass2_prompt")
          ? String(provided.pass2_prompt || "")
          : String(segment?.minimax_h3_pass2_prompt || "");
        return { scene: index + 1, label: `Scene ${index + 1}`, prompt: promptText, pass2_prompt: pass2Text };
      });
      return JSON.stringify({ version: 1, type: "storyboard_video_prompts", scene_count: scenes.length, scenes }, null, 2) + "\n";
    }
    return values.map((item) => String(item || "").trim()).join("\n\n").replace(/\s+$/g, "") + "\n";
  }

  function applyPromptBlocksToSegments(kind, prompts) {
    const values = Array.isArray(prompts) ? prompts : [];
    if (!values.length) throw new Error(`No ${promptKindLabel(kind)} prompts were found in that text.`);
    const segments = allEditableSegments();
    pushHistory();
    for (let index = 0; index < segments.length && index < values.length; index += 1) {
      const value = values[index];
      const promptText = typeof value === "object" && value
        ? String(value.prompt || "").trim()
        : String(value || "").trim();
      setSegmentPromptForEdit(segments[index], kind, promptText);
      if (
        kind === "minimax"
        && typeof value === "object"
        && value
        && Object.prototype.hasOwnProperty.call(value, "pass2_prompt")
        && value.pass2_prompt !== undefined
      ) {
        segments[index].minimax_h3_pass2_prompt = String(value.pass2_prompt || "");
      }
    }
    syncInspector();
    render();
  }

  async function saveMiniMaxPromptExport(sourceContent = "") {
    const path = promptKindFile("minimax");
    if (!path) return "";
    let exportData = null;
    try {
      const raw = String(sourceContent || await loadPromptTextFile(path, "")).trim();
      const parsed = raw ? JSON.parse(raw) : null;
      if (parsed && typeof parsed === "object" && Array.isArray(parsed.scenes)) exportData = parsed;
    } catch (_) {
      exportData = null;
    }
    if (!exportData) exportData = { version: 1, type: "storyboard_video_prompts", scenes: [] };
    const segments = allEditableSegments();
    exportData.scene_count = segments.length;
    exportData.scenes = segments.map((segment, index) => {
      const existing = exportData.scenes.find((scene) => String(scene?.scene_id || "") === String(segment?.id || "")) || exportData.scenes[index] || {};
      return {
        ...existing,
        scene: index + 1,
        scene_id: existing.scene_id || segment.id || "",
        label: existing.label || sceneDisplayName(segment, index),
        prompt: String(segment?.minimax_h3_prompt || "").trim(),
        pass2_prompt: String(segment?.minimax_h3_pass2_prompt || ""),
      };
    });
    return (await savePromptTextFile(path, JSON.stringify(exportData, null, 2) + "\n"))?.path || path;
  }

  async function ensureOriginalPromptBackup(kind) {
    const backupPath = projectPromptBackupPath(kind, "original");
    if (!backupPath) throw new Error("Create or load a project first.");
    const existing = await loadPromptTextFile(backupPath, null);
    if (existing !== null) return backupPath;
    const currentPath = promptKindFile(kind);
    const currentContent = await loadPromptTextFile(currentPath, formatPromptBlocks(kind));
    await savePromptTextFile(backupPath, currentContent || formatPromptBlocks(kind));
    return backupPath;
  }

  async function backupCurrentPromptState(kind, reason) {
    const backupPath = projectPromptBackupPath(kind, `${reason}_${promptTimestamp()}`);
    if (!backupPath) throw new Error("Create or load a project first.");
    await savePromptTextFile(backupPath, formatPromptBlocks(kind));
    return backupPath;
  }

  function showPromptFormatHint(kind, actionKey, force = false) {
    return new Promise((resolve) => {
      const prefKey = `${kind}_${actionKey}`;
      if (!force && state.promptToolsHintPrefs?.[prefKey] === false) {
        resolve(true);
        return;
      }
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(680px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = `${promptKindLabel(kind).replace(/^\w/, (letter) => letter.toUpperCase())} Prompt Formats`;
      heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
      const body = document.createElement("div");
      body.style.cssText = "display:flex;flex-direction:column;gap:10px;font-size:13px;color:#d4d4d8;line-height:1.45;max-height:60vh;overflow:auto;padding-right:4px;";
      const addText = (text) => {
        const item = document.createElement("div");
        item.textContent = text;
        body.append(item);
      };
      const addSection = (title, code) => {
        const label = document.createElement("div");
        label.textContent = title;
        label.style.cssText = "font-weight:900;color:#e0f2fe;margin-top:2px;";
        const block = document.createElement("pre");
        block.textContent = code;
        block.style.cssText = "margin:0;border:1px solid #334155;border-radius:6px;background:#020617;color:#f8fafc;padding:10px;white-space:pre-wrap;font-size:12px;line-height:1.4;overflow:auto;";
        body.append(label, block);
      };
      addText(`This tool edits the final ${promptKindLabel(kind)} prompt list for the current project.`);
      addText("Reload updates the scene prompt boxes from the file. Original reload uses the first backup this tool created.");
      if (kind === "i2v") {
        addSection("Blank-line format", "Video prompt for scene 1\n\nVideo prompt for scene 2\n\nVideo prompt for scene 3");
        addSection("Key/value format", "I2V1=Video prompt for scene 1\nI2V2=Video prompt for scene 2\nI2V3=Video prompt for scene 3");
        addSection("JSON format", "{\n  \"I2V1\": \"Video prompt for scene 1\",\n  \"I2V2\": \"Video prompt for scene 2\"\n}");
      } else {
        addSection("Blank-line format", "Prompt for scene 1\n\nPrompt for scene 2\n\nPrompt for scene 3");
        addSection("Key/value format", "Prompt1=Prompt for scene 1\nPrompt2=Prompt for scene 2\nPrompt3=Prompt for scene 3");
        addSection("JSON format", "{\n  \"Prompt1\": \"Prompt for scene 1\",\n  \"Prompt2\": \"Prompt for scene 2\"\n}");
      }
      const showAgain = makeCheckbox("Show this hint next time", true);
      showAgain.input.checked = state.promptToolsHintPrefs?.[prefKey] !== false;
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const confirm = makeButton("Continue", "primary");
      const finish = (result) => {
        if (!showAgain.input.checked) {
          state.promptToolsHintPrefs = { ...(state.promptToolsHintPrefs || {}), [prefKey]: false };
          autoSaveSessionQuiet("prompt format hint preference changed");
        }
        backdrop.remove();
        resolve(result);
      };
      cancel.onclick = () => finish(false);
      confirm.onclick = () => finish(true);
      actions.append(cancel, confirm);
      box.append(heading, body, showAgain.wrapper, actions);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  async function editFinalPromptList(kind) {
    if (!await showPromptFormatHint(kind, "edit")) return;
    const path = promptKindFile(kind);
    if (!path) {
      toast("Create or load a project first, then prompt files can be edited.", true);
      return;
    }
    try {
      await ensureOriginalPromptBackup(kind);
      const currentContent = await loadPromptTextFile(path, formatPromptBlocks(kind));
      const box = document.createElement("div");
      box.style.cssText = `
        position:fixed;left:50%;top:6%;transform:translateX(-50%);
        z-index:100006;width:min(980px,calc(100vw - 36px));height:min(760px,calc(100vh - 56px));
        display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;
        border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#e0f2fe;
        box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;
      `;
      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid #155e75;background:#083344;";
      const heading = document.createElement("div");
      heading.textContent = kind === "minimax" ? "Edit MiniMax H3 Prompts" : kind === "i2v" ? "Edit Image-to-Video Prompts" : "Edit Text-to-Image Prompts";
      heading.style.cssText = "font-size:13px;font-weight:900;";
      const headerActions = document.createElement("div");
      headerActions.style.cssText = "display:flex;align-items:center;gap:8px;";
      const hint = makeButton("?");
      hint.style.minWidth = "34px";
      const close = makeButton("Close");
      close.style.padding = "5px 8px";
      headerActions.append(hint, close);
      header.append(heading, headerActions);
      const pathText = document.createElement("div");
      pathText.textContent = path;
      pathText.style.cssText = "padding:8px 12px;border-bottom:1px solid #1f2937;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;";
      const textarea = document.createElement("textarea");
      textarea.value = currentContent || formatPromptBlocks(kind);
      textarea.spellcheck = false;
      textarea.style.cssText = "width:100%;height:100%;box-sizing:border-box;border:0;resize:none;background:#020617;color:#fafafa;padding:12px;font-size:12px;line-height:1.45;outline:none;font-family:monospace;";
      const actions = document.createElement("div");
      actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;padding:10px 12px;border-top:1px solid #155e75;background:#111827;";
      const save = makeButton("Save", "primary");
      const cancel = makeButton("Cancel");
      actions.append(cancel, save);
      box.append(header, pathText, textarea, actions);
      document.body.append(box);
      textarea.focus();
      close.onclick = () => box.remove();
      cancel.onclick = () => box.remove();
      hint.onclick = () => showPromptFormatHint(kind, "editor_help", true);
      save.onclick = async () => {
        try {
          save.disabled = true;
          save.textContent = "Saving...";
          await ensureOriginalPromptBackup(kind);
          await backupCurrentPromptState(kind, "before_edit");
          const prompts = parsePromptTextBlocks(textarea.value, kind);
          applyPromptBlocksToSegments(kind, prompts);
          if (kind === "minimax") await saveMiniMaxPromptExport(textarea.value);
          else await savePromptTextFile(path, textarea.value);
          await saveSession({ quiet: true, throwOnError: true });
          toast(`Saved and loaded ${prompts.length} ${promptKindLabel(kind)} prompt${prompts.length === 1 ? "" : "s"}.`);
          box.remove();
        } catch (error) {
          toast(String(error?.message || error), true);
        } finally {
          save.disabled = false;
          save.textContent = "Save";
        }
      };
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function reloadFinalPromptList(kind, original = false) {
    if (!await showPromptReloadConfirm(kind, original)) return;
    const path = original ? projectPromptBackupPath(kind, "original") : promptKindFile(kind);
    if (!path) {
      toast("Create or load a project first, then prompt files can be reloaded.", true);
      return;
    }
    try {
      await ensureOriginalPromptBackup(kind);
      await backupCurrentPromptState(kind, original ? "before_original_reload" : "before_reload");
      const content = await loadPromptTextFile(path, "");
      if (!String(content || "").trim()) throw new Error(`That ${promptKindLabel(kind)} prompt file is empty:\n${path}`);
      const prompts = parsePromptTextBlocks(content, kind);
      applyPromptBlocksToSegments(kind, prompts);
      if (kind === "minimax") await saveMiniMaxPromptExport(content);
      await saveSession({ quiet: true, throwOnError: true });
      toast(`Reloaded ${prompts.length} ${promptKindLabel(kind)} prompt${prompts.length === 1 ? "" : "s"} into the scene boxes.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function clearFinalPromptList(kind) {
    if (!await showPromptClearConfirm(kind)) return;
    const path = promptKindFile(kind);
    if (!path) {
      toast("Create or load a project first, then prompts can be cleared.", true);
      return;
    }
    try {
      await ensureOriginalPromptBackup(kind);
      await backupCurrentPromptState(kind, "before_clear");
      pushHistory();
      for (const segment of allEditableSegments()) {
        if (kind === "i2v") {
          segment.i2v_prompt = "";
          segment.i2v_prompt_origin = "manual";
        } else {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
        }
      }
      await savePromptTextFile(path, "");
      syncInspector();
      render();
      await saveSession({ quiet: true, throwOnError: true });
      toast(kind === "i2v" ? "Cleared all I2V prompts." : "Cleared all T2I prompts.");
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  return {
    builderStorySourcePath, clearFinalPromptList, countPromptFindReplaceMatches, createStoryScenesFromSource,
    editContextTextFile, editFinalPromptList, loadBuilderStorySource, locationExtractionStyleTheme,
    projectContextPath, projectPromptsPath, projectReferenceBuilderLocationsPath, projectSceneNotesPath,
    referenceBuilderSubjectLocationText, reloadFinalPromptList, replacePromptPhraseAcrossScenes,
    saveBuilderStorySource, saveGemmaJunkDebug, segmentPromptForEdit, setSegmentPromptForEdit,
  };
}

export function wireContextFileInputs({
  autoSaveSessionQuiet, clearConceptPromptNotesFromSegments, clearI2VMotionNotesFromSegments,
  editContextTextFile, editI2VMotionJsonButton, editPromptJsonButton, editStoryIdeaButton,
  editSubjectSceneButton, editThemeStyleButton, i2vMotionJsonInput, importI2VMotionJson, importPromptJson,
  loadDefaultContextPaths, loadVrgdgContextButton, promptJsonInput, pushHistory, render, state,
  storyIdeaInput, subjectSceneInput, syncInspector, themeStyleInput, useVrgdgTextContext,
}) {
  useVrgdgTextContext.input.addEventListener("change", () => {
    pushHistory();
    state.useVrgdgTextContext = Boolean(useVrgdgTextContext.input.checked);
  });
  loadVrgdgContextButton.onclick = loadDefaultContextPaths;
  themeStyleInput.addEventListener("input", () => {
    pushHistory();
    state.themeStylePath = themeStyleInput.value || "";
  });
  storyIdeaInput.addEventListener("input", () => {
    pushHistory();
    state.storyIdeaPath = storyIdeaInput.value || "";
  });
  subjectSceneInput.addEventListener("input", () => {
    pushHistory();
    state.subjectScenePath = subjectSceneInput.value || "";
  });
  editThemeStyleButton.onclick = () => editContextTextFile(themeStyleInput, "Edit Theme/Style Text", "themestyle.txt", "builder_style_theme", {
    afterSave: async () => {
      state.themeStylePath = themeStyleInput.value || "";
      await autoSaveSessionQuiet("theme/style text edited");
    },
  });
  editStoryIdeaButton.onclick = () => editContextTextFile(storyIdeaInput, "Edit Story Idea Text", "storyconcept.txt", "builder_story_idea", {
    afterSave: async () => {
      state.storyIdeaPath = storyIdeaInput.value || "";
      await autoSaveSessionQuiet("story idea text edited");
    },
  });
  editSubjectSceneButton.onclick = () => editContextTextFile(subjectSceneInput, "Edit Subject/Scene Text", "subjectsandscenes.txt", "builder_subjects_and_scenes", {
    afterSave: async () => {
      state.subjectScenePath = subjectSceneInput.value || "";
      await autoSaveSessionQuiet("subject/scene text edited");
    },
  });
  editPromptJsonButton.onclick = () => editContextTextFile(promptJsonInput, "Edit Prompt JSON", "ConceptPrompts.txt", null, {
    showGemma: false,
    helpText: "Save this file to re-import the updated concept prompts into the scene notes. To clear all concept notes, delete everything in this editor and save.",
    afterSave: async (_result, text) => {
      state.promptJsonPath = promptJsonInput.value || "";
      if (!String(text || "").trim()) {
        clearConceptPromptNotesFromSegments();
        syncInspector();
        render();
        toast("Cleared ConceptPrompts and scene concept notes.");
      } else {
        await importPromptJson();
      }
      await autoSaveSessionQuiet("prompt JSON edited");
    },
  });
  editI2VMotionJsonButton.onclick = () => editContextTextFile(i2vMotionJsonInput, "Edit I2V Motion Notes JSON", "I2VMotionNotes.txt", null, {
    showGemma: false,
    helpText: "Save this file to re-import the updated I2V motion notes into the scene motion boxes. To clear all motion notes, delete everything in this editor and save.",
    afterSave: async (_result, text) => {
      state.i2vMotionJsonPath = i2vMotionJsonInput.value || "";
      if (!String(text || "").trim()) {
        clearI2VMotionNotesFromSegments();
        syncInspector();
        render();
        toast("Cleared I2VMotionNotes and scene motion notes.");
      } else {
        await importI2VMotionJson();
      }
      await autoSaveSessionQuiet("I2V motion notes edited");
    },
  });
}
