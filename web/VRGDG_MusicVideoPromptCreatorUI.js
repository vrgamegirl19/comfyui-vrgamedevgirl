import { api } from "../../scripts/api.js";

const PROMPT_CREATOR_VERSION = "prompt-creator-2026-05-24-clean-gemma-panel";
const LYRIC_CREATOR_GPT_URL = "https://chatgpt.com/g/g-69979b391cc88191ae4fe298b59c236e-ai-lyric-creator";
const STYLE_THEME_GPT_URL = "https://chatgpt.com/g/g-69fb415a964c8191b4a737f84f37227f-ltx-2-3-style-theme-guide/c/69fb427d-4518-8331-bfd7-505c0f55d2cc";
const STORY_IDEA_GPT_URL = "https://chatgpt.com/g/g-69fb3cb767448191a6caa88be94940d5-ltx-2-3-story-concept-helper/c/69fb3e25-7e74-8326-abd6-7df9cf847a5b";
const SUBJECT_LOCATION_GPT_URL = "https://chatgpt.com/g/g-69fb38a997fc8191a2fa479e44a3c675-ltx-2-3-subject-and-location-creator/c/69fb39e2-2ba0-8328-94c0-6ac9c94d0c89";

function isLikelyEmbeddingModelId(modelId) {
  const text = String(modelId || "").toLowerCase();
  return text.includes("embedding") || text.includes("embed") || text.includes("nomic-embed") || text.includes("bge-") || text.includes("e5-");
}

function makeButton(label, kind = "neutral") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.style.cssText = `
    border: 1px solid ${kind === "primary" ? "#0891b2" : "#3f3f46"};
    border-radius: 6px;
    background: ${kind === "primary" ? "#06b6d4" : kind === "danger" ? "#991b1b" : "#27272a"};
    color: ${kind === "primary" ? "#082f49" : "#f4f4f5"};
    font-size: 12px;
    font-weight: 800;
    padding: 8px 11px;
    cursor: pointer;
  `;
  return button;
}

function createProgressWindow(title) {
  const box = document.createElement("div");
  box.style.cssText = `
    position:fixed;left:50%;top:16%;transform:translateX(-50%);
    z-index:100060;width:min(720px,calc(100vw - 40px));
    border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cffafe;
    box-shadow:0 22px 70px rgba(0,0,0,.55);overflow:hidden;font-family:Arial,sans-serif;
  `;
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid #155e75;background:#083344;";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-size:13px;font-weight:900;";
  const close = makeButton("Close");
  close.style.padding = "5px 8px";
  header.append(heading, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:12px;font-size:12px;line-height:1.45;white-space:pre-wrap;max-height:min(62vh,620px);overflow:auto;";
  body.textContent = "Starting...";
  const barOuter = document.createElement("div");
  barOuter.style.cssText = "height:8px;background:#164e63;border-radius:999px;margin:0 12px 12px;overflow:hidden;";
  const barInner = document.createElement("div");
  barInner.style.cssText = "width:20%;height:100%;background:#22d3ee;border-radius:999px;transition:width .2s ease;";
  barOuter.append(barInner);
  box.append(header, body, barOuter);
  document.body.append(box);
  close.onclick = () => box.remove();
  return {
    set(message, percent = null) {
      body.textContent = message;
      if (percent !== null) barInner.style.width = `${Math.max(5, Math.min(100, percent))}%`;
    },
    close(delay = 0) {
      setTimeout(() => box.remove(), delay);
    },
  };
}

function makeInput(value = "", type = "text") {
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
  return input;
}

function makeTextarea(value = "", rows = 6) {
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.rows = rows;
  textarea.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  return textarea;
}

function makeReadOnlyTextBox(value = "", rows = 5) {
  const textarea = makeTextarea(value, rows);
  textarea.readOnly = true;
  textarea.style.resize = "vertical";
  textarea.style.background = "#0f172a";
  textarea.style.color = "#cbd5e1";
  textarea.style.minHeight = "0";
  textarea.style.cursor = "default";
  return textarea;
}

function makeCompactPreviewBox(value = "", rows = 4) {
  const textarea = makeReadOnlyTextBox(value, rows);
  textarea.style.maxHeight = "120px";
  textarea.style.fontSize = "11px";
  textarea.style.opacity = "0.92";
  return textarea;
}

function makeSelect(options = [], value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
  for (const optionValue of options) {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = optionValue;
    select.append(option);
  }
  if (value) select.value = value;
  return select;
}

function makeCheckbox(checked = false) {
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(checked);
  input.style.cssText = "width:16px;height:16px;accent-color:#06b6d4;";
  return input;
}

function makeCheckboxField(label, input, hint = "") {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;align-items:flex-start;gap:8px;font-size:12px;color:#d4d4d8;font-weight:800;";
  const textWrap = document.createElement("span");
  textWrap.style.cssText = "display:flex;flex-direction:column;gap:4px;line-height:1.35;";
  const text = document.createElement("span");
  text.textContent = label;
  textWrap.append(text);
  if (hint) {
    const small = document.createElement("span");
    small.textContent = hint;
    small.style.cssText = "color:#a1a1aa;font-size:11px;font-weight:500;line-height:1.35;";
    textWrap.append(small);
  }
  wrapper.append(input, textWrap);
  return wrapper;
}

function makeField(label, control, hint = "") {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;color:#d4d4d8;font-weight:800;";
  const text = document.createElement("span");
  text.textContent = label;
  wrapper.append(text, control);
  if (hint) {
    const small = document.createElement("span");
    small.textContent = hint;
    small.style.cssText = "color:#a1a1aa;font-size:11px;font-weight:500;line-height:1.35;";
    wrapper.append(small);
  }
  return wrapper;
}

function makeCompactField(label, control, width = "110px", hint = "") {
  const field = makeField(label, control, hint);
  field.style.width = width;
  field.style.flex = `0 0 ${width}`;
  control.style.width = "100%";
  return field;
}

function makeHintButton(title, lines = []) {
  const button = makeButton("?");
  button.type = "button";
  button.style.cssText += "width:28px;min-width:28px;padding:7px 0;";
  button.onclick = () => showInfoPopup(title, lines);
  return button;
}

function makeCompactHintField(label, control, width = "110px", hintTitle = label, hintLines = []) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;color:#d4d4d8;font-weight:800;";
  wrapper.style.width = width;
  wrapper.style.flex = `0 0 ${width}`;
  const header = document.createElement("span");
  header.style.cssText = "display:flex;align-items:center;gap:6px;";
  const text = document.createElement("span");
  text.textContent = label;
  header.append(text, makeHintButton(hintTitle, hintLines));
  control.style.width = "100%";
  wrapper.append(header, control);
  return wrapper;
}

function makePickerField(label, input, button, hint = "") {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  row.append(input, button);
  return makeField(label, row, hint);
}

function showInfoPopup(title, lines = []) {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100070;background:rgba(0,0,0,.58);display:flex;align-items:center;justify-content:center;padding:18px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(560px,calc(100vw - 36px));border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#e0f2fe;box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 14px;border-bottom:1px solid #155e75;background:#083344;";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-size:14px;font-weight:900;";
  const close = makeButton("Close");
  close.style.padding = "5px 8px";
  header.append(heading, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:9px;font-size:12px;line-height:1.45;color:#d4f3ff;";
  for (const line of lines) {
    const item = document.createElement("div");
    item.textContent = line;
    item.style.cssText = "border:1px solid #164e63;border-radius:7px;background:#111827;padding:9px;";
    body.append(item);
  }
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  const finish = () => backdrop.remove();
  close.onclick = finish;
  backdrop.onclick = (event) => {
    if (event.target === backdrop) finish();
  };
}

function makePromptTextSection(label, textarea, buttons = [], hint = "") {
  const section = document.createElement("section");
  section.style.cssText = "border:1px solid #364152;border-radius:8px;background:#14191f;padding:14px;display:flex;flex-direction:column;gap:8px;";
  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;";
  const text = document.createElement("div");
  text.textContent = label;
  text.style.cssText = "font-size:12px;color:#d4d4d8;font-weight:900;";
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
  for (const button of buttons) actions.append(button);
  row.append(text, actions);
  section.append(row, textarea);
  if (hint) {
    const small = document.createElement("div");
    small.textContent = hint;
    small.style.cssText = "color:#a1a1aa;font-size:11px;line-height:1.35;";
    section.append(small);
  }
  return section;
}

function requestGemmaUserInput(title, currentText = "", hint = "") {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:18px;";
    const box = document.createElement("div");
    box.style.cssText = `
      width:min(720px,calc(100vw - 36px));
      border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#e0f2fe;
      box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;
      display:flex;flex-direction:column;
    `;
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 14px;border-bottom:1px solid #155e75;background:#083344;";
    const heading = document.createElement("div");
    heading.textContent = title;
    heading.style.cssText = "font-size:14px;font-weight:900;";
    const close = makeButton("Close");
    close.style.padding = "5px 8px";
    header.append(heading, close);

    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:9px;";
    const note = document.createElement("div");
    note.textContent = hint || "Type what you want Gemma4 to use as user input for this draft.";
    note.style.cssText = "color:#a1a1aa;font-size:12px;line-height:1.4;";
    const textarea = makeTextarea(currentText || "", 9);
    textarea.placeholder = "Enter your idea, notes, rough draft, or details for Gemma4...";
    body.append(note, textarea);

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;padding:12px 14px;border-top:1px solid #155e75;background:#111827;";
    const cancel = makeButton("Cancel");
    const run = makeButton("Run Gemma4", "primary");
    actions.append(cancel, run);
    box.append(header, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);

    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    close.onclick = () => finish(null);
    cancel.onclick = () => finish(null);
    run.onclick = () => finish(textarea.value || "");
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) finish(null);
    });
    textarea.focus();
    textarea.select();
  });
}

function makePanel(title) {
  const panel = document.createElement("section");
  panel.style.cssText = "border:1px solid #364152;border-radius:8px;background:#14191f;padding:14px;display:flex;flex-direction:column;gap:10px;";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-size:13px;font-weight:900;color:#bae6fd;";
  panel.append(heading);
  return panel;
}

async function postJson(url, payload) {
  const response = await api.fetchApi(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.ok === false) {
    throw new Error(data?.error || data?.message || response.statusText || "Request failed");
  }
  return data;
}

async function getJson(url) {
  const response = await api.fetchApi(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.ok === false) {
    throw new Error(data?.error || data?.message || response.statusText || "Request failed");
  }
  return data;
}

async function postForm(url, formData) {
  const response = await api.fetchApi(url, {
    method: "POST",
    body: formData,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.ok === false) {
    throw new Error(data?.error || data?.message || response.statusText || "Request failed");
  }
  return data;
}

function extractPromptCreatorText(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const readText = (nodeId) => {
    const output = outputs?.[String(nodeId)] || {};
    const text = output?.text ?? output?.ui?.text;
    const values = Array.isArray(text) ? text.flat(Infinity) : [text];
    return values.map((value) => String(value ?? "")).find((value) => value.trim()) || "";
  };
  const allText = [];
  for (const output of Object.values(outputs)) {
    const text = output?.text ?? output?.ui?.text;
    const values = Array.isArray(text) ? text.flat(Infinity) : [text];
    for (const value of values) {
      const stringValue = String(value ?? "");
      if (stringValue.trim()) allText.push(stringValue);
    }
  }
  return {
    whisper: readText(804) || readText("28:870") || readText(961) || allText.find((value) => /(?:lyricSegment|segment)\s*\d+/i.test(value)) || "",
    srt: readText("28:869") || readText(962) || allText.find((value) => /-->\s*\d{2}:/i.test(value)) || "",
  };
}

async function waitForPromptCreatorText(promptId, onStatus) {
  const started = Date.now();
  while (Date.now() - started < 30 * 60 * 1000) {
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const text = extractPromptCreatorText(data, promptId);
    if (text.whisper) return text;
    onStatus?.("Waiting for Whisper/SRT workflow output...");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Timed out waiting for the hidden Whisper/SRT workflow.");
}

async function queueWorkflowPrompt(prompt) {
  const clientId = api.clientId || crypto.randomUUID();
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.error) {
    const detailParts = [];
    if (data?.error?.message) detailParts.push(data.error.message);
    else if (typeof data?.error === "string") detailParts.push(data.error);
    if (data?.node_errors && typeof data.node_errors === "object") {
      for (const [nodeId, nodeError] of Object.entries(data.node_errors)) {
        const messages = [];
        if (Array.isArray(nodeError?.errors)) {
          for (const error of nodeError.errors) {
            messages.push(error?.message || error?.details || JSON.stringify(error));
          }
        }
        if (nodeError?.class_type || messages.length) {
          detailParts.push(`Node ${nodeId}${nodeError?.class_type ? ` (${nodeError.class_type})` : ""}: ${messages.join("; ")}`);
        }
      }
    }
    throw new Error(detailParts.filter(Boolean).join("\n") || `Queue failed (${response.status})`);
  }
  return data;
}

function setStatus(status, text, busy = false) {
  status.textContent = text;
  status.style.color = busy ? "#67e8f9" : "#d4d4d8";
}

function parseJsonSafe(text, fallback = {}) {
  try {
    return JSON.parse(text || "");
  } catch {
    return fallback;
  }
}

function parseOutputMapping(text, prefix, label) {
  const raw = String(text || "").trim();
  if (!raw) return {};
  if (/^\s*[\[{]/.test(raw)) {
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      throw new Error(`${label} is not valid JSON: ${error.message}`);
    }
    const output = {};
    if (Array.isArray(parsed)) {
      parsed.forEach((value, index) => {
        output[`${prefix}${index + 1}`] = String(value ?? "").trim();
      });
      return output;
    }
    if (!parsed || typeof parsed !== "object") {
      throw new Error(`${label} must be a JSON object or array.`);
    }
    for (const [key, value] of Object.entries(parsed)) {
      const match = String(key).match(/(\d+)/);
      if (!match) continue;
      output[`${prefix}${Number(match[1])}`] = String(value ?? "").trim();
    }
    return output;
  }

  const keyed = {};
  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    const match = line.match(/^\s*(?:Prompt|I2V|Motion)\s*(\d+)\s*[:=]\s*(.*)$/i);
    if (!match) continue;
    keyed[`${prefix}${Number(match[1])}`] = String(match[2] ?? "").trim();
  }
  if (Object.keys(keyed).length) return keyed;

  const blocks = raw
    .split(/\n\s*\n+/)
    .map((block) => block.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  const output = {};
  blocks.forEach((block, index) => {
    output[`${prefix}${index + 1}`] = block;
  });
  return output;
}

function prettyJson(value) {
  try {
    return JSON.stringify(value || {}, null, 2);
  } catch {
    return "{}";
  }
}

function parseLyricSegmentsToJson(text) {
  const parsedJson = parseJsonSafe(text, null);
  if (parsedJson && typeof parsedJson === "object" && !Array.isArray(parsedJson)) {
    const normalized = {};
    for (const [key, value] of Object.entries(parsedJson)) {
      const match = String(key).match(/^(?:lyricSegment|segment)\s*(\d+)$/i);
      if (match) normalized[`segment${Number(match[1])}`] = String(value ?? "").trim() || "Instrumental section.";
    }
    if (Object.keys(normalized).length) return normalized;
  }

  const segments = {};
  for (const line of String(text || "").split(/\r?\n/)) {
    const match = line.match(/^\s*(?:lyricSegment|segment)\s*(\d+)\s*[:=]\s*(.*)$/i);
    if (!match) continue;
    const index = Number(match[1]);
    if (!Number.isFinite(index) || index <= 0) continue;
    segments[`segment${index}`] = String(match[2] ?? "").trim() || "Instrumental section.";
  }
  return segments;
}

function normalizeInlineText(value) {
  return String(value || "").replace(/\r|\n/g, " ").split(/\s+/).filter(Boolean).join(" ");
}

function stripLeadingSubject(prompt, subjects = []) {
  let promptText = normalizeInlineText(prompt);
  const subjectList = subjects.map(normalizeInlineText).filter(Boolean);
  let changed = true;
  let guard = 0;
  while (changed && guard < 8) {
    changed = false;
    guard += 1;
    for (const subjectText of subjectList) {
      if (!promptText) break;
      const lowerPrompt = promptText.toLowerCase();
      const lowerSubject = subjectText.toLowerCase();
      if (lowerPrompt === lowerSubject) {
        promptText = "";
        changed = true;
        break;
      }
      if (lowerPrompt.startsWith(lowerSubject)) {
        promptText = promptText.slice(subjectText.length).replace(/^\s*[,;:.-]\s*/, "").trim();
        changed = true;
        break;
      }
    }
  }
  return promptText;
}

function prependSubjectToPrompts(prompts, subject, separator = ", ", previousSubjects = []) {
  const subjectText = normalizeInlineText(subject);
  if (!subjectText || !prompts || typeof prompts !== "object" || Array.isArray(prompts)) return prompts || {};
  const knownSubjects = [subjectText, ...(Array.isArray(previousSubjects) ? previousSubjects : [previousSubjects])];
  const output = {};
  for (const [key, value] of Object.entries(prompts)) {
    let promptText = stripLeadingSubject(value, knownSubjects);
    if (promptText) {
      promptText = `${subjectText}${separator}${promptText}`;
    } else {
      promptText = subjectText;
    }
    output[key] = promptText;
  }
  return output;
}

function buildPayload(controls, modelSelect) {
  return {
    project_folder: controls.projectFolder.value,
    audio_path: controls.audioPath.value,
    min_duration: controls.minDuration.value,
    max_duration: controls.maxDuration.value,
    bias: controls.bias.value,
    duration_preset: controls.durationPreset.value,
    use_srt_durations: controls.useSrtDurations.checked,
    fixed_scene_duration: controls.fixedSceneDuration.value,
    empty_segment_text: controls.emptySegmentText.value,
    whisper_language: controls.whisperLanguage.value,
    concept_match_mode: controls.conceptMatchMode.value,
    append_subject_to_prompts: controls.appendSubjectToPrompts.checked,
    repair_lyric_segments: controls.repairLyricSegments.checked,
    model_file: modelSelect.value,
    whisper_segments: controls.whisperSegments.value,
    full_lyrics: controls.fullLyrics.value,
    style_theme: controls.styleTheme.value,
    story_idea: controls.storyIdea.value,
    subject_locations: controls.subjectLocations.value,
    srt_text: controls.srtText.value,
    output_srt_path: controls.srtOutput.value,
    text_runner: controls.textGemmaRunner || "builtin",
    lmstudio_base_url: controls.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
    lmstudio_model: controls.lmStudioModel || "",
    lmstudio_api_key: controls.lmStudioApiKey || "",
    llm_api_provider: controls.llmApiProvider || "openai",
    llm_api_model: controls.llmApiModel || "",
    llm_api_key: controls.llmApiKey || "",
    text_gemma_runner: controls.textGemmaRunner || "builtin",
    lm_studio_base_url: controls.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
    lm_studio_model: controls.lmStudioModel || "",
    lm_studio_api_key: controls.lmStudioApiKey || "",
  };
}

function createModeChoiceModal() {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:9px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Start New Project";
    heading.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    const note = document.createElement("div");
    note.textContent = "Choose where you want to begin. Prompt Creator builds the SRT, lyric segments, concept prompts, and context files. Video Creation opens the editor directly.";
    note.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const promptCreator = makeButton("Start Prompt Creator", "primary");
    const videoCreator = makeButton("Start Video Creation");
    const cancel = makeButton("Cancel");
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    promptCreator.onclick = () => finish("prompt_creator");
    videoCreator.onclick = () => finish("video_creator");
    cancel.onclick = () => finish("");
    actions.append(promptCreator, videoCreator);
    box.append(heading, note, actions, cancel);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

function showPromptCreatorDraftProjectModal(projects = []) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:min(780px,calc(100vh - 40px));border:1px solid #155e75;border-radius:9px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Load Prompt Creator Draft";
    heading.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    const note = document.createElement("div");
    note.textContent = "Choose a recent project. Prompt Creator will load that project's saved draft if one exists.";
    note.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:7px;overflow:auto;max-height:min(480px,52vh);padding-right:3px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
    const close = makeButton("Close");
    const finish = (result) => {
      backdrop.remove();
      resolve(result);
    };
    close.onclick = () => finish(null);
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(null);
    });

    if (!projects.length) {
      const empty = document.createElement("div");
      empty.textContent = "No existing projects were found in the ComfyUI output folder.";
      empty.style.cssText = "border:1px dashed #3f3f46;border-radius:7px;padding:14px;color:#a1a1aa;font-size:12px;text-align:center;";
      list.append(empty);
    } else {
      for (const project of projects) {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;";
        const info = document.createElement("div");
        info.style.cssText = "display:flex;flex-direction:column;gap:4px;min-width:0;";
        const name = document.createElement("div");
        name.textContent = project.name || "Unnamed project";
        name.style.cssText = "font-size:13px;font-weight:900;color:#f8fafc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const updated = project.updated ? new Date(project.updated * 1000).toLocaleString() : "unknown date";
        const meta = document.createElement("div");
        meta.textContent = `${project.scene_count || 0} scene${Number(project.scene_count || 0) === 1 ? "" : "s"} | ${updated}`;
        meta.style.cssText = "font-size:11px;color:#a1a1aa;";
        const path = document.createElement("div");
        path.textContent = project.project_folder || "";
        path.style.cssText = "font-size:11px;color:#67e8f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        info.append(name, meta, path);
        const open = makeButton("Load Draft", "primary");
        open.onclick = () => finish({ project_folder: project.project_folder || "" });
        row.append(info, open);
        list.append(row);
      }
    }

    actions.append(close);
    box.append(heading, note, list, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.tabIndex = -1;
    backdrop.focus();
  });
}

function openPromptCreator(options = {}) {
  const existing = document.querySelector(".vrgdg-music-prompt-creator");
  if (existing) existing.remove();

  const state = {
    repairedSegments: {},
    conceptPrompts: {},
    i2vMotionNotes: {},
    extractedSubject: "",
    textGemmaRunner: "builtin",
    lmStudioBaseUrl: "http://127.0.0.1:1234/v1",
    lmStudioModel: "",
    lmStudioApiKey: "",
    llmApiProvider: "openai",
    llmApiModel: "",
    llmApiKey: "",
    llmApiChoices: null,
  };

  function gemmaRunnerLine() {
    return `Runner: ${state.textGemmaRunner === "llm_api" ? "LLM API" : state.textGemmaRunner === "lm_studio" ? "LM Studio" : "Gemma Local"}`;
  }

  const instructionLabels = {
    full_lyrics: "Full Lyrics",
    style_theme: "Style / Theme",
    story_idea: "Story Idea",
    subject_locations: "Subject and Locations",
    concept_prompts: "Concept Prompts",
    subject_extract: "Subject Extraction",
    i2v_motion_notes: "I2V Motion Notes",
  };

  const overlay = document.createElement("div");
  overlay.className = "vrgdg-music-prompt-creator";
  overlay.dataset.version = PROMPT_CREATOR_VERSION;
  overlay.style.cssText = "position:fixed;inset:24px;z-index:100010;border:1px solid #364152;border-radius:12px;background:#1f2328;color:#f8fafc;box-shadow:0 24px 90px rgba(0,0,0,.6);display:grid;grid-template-rows:auto minmax(0,1fr);overflow:hidden;font-family:Arial,sans-serif;";

  const topbar = document.createElement("div");
  topbar.style.cssText = "display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid #364152;background:#20242a;";
  const title = document.createElement("div");
  title.textContent = "Prompt Creator";
  title.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;margin-right:auto;";
  const backButton = makeButton("Back To Video Creator");
  const loadDraftButton = makeButton("Load Project Draft");
  const sendToVideoButton = makeButton("Send To Video Creator", "primary");
  const gemmaRunnerButton = makeButton("Gemma Runner");
  const saveDraftButton = makeButton("Save Project Draft", "primary");
  const closeButton = makeButton("Close");
  closeButton.onclick = () => overlay.remove();
  topbar.append(title, backButton, loadDraftButton, sendToVideoButton, gemmaRunnerButton, saveDraftButton, closeButton);

  const body = document.createElement("div");
  body.style.cssText = "min-height:0;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:14px;background:#1f2328;";

  const setupPanel = makePanel("Whisper / SRT Setup");
  const projectFolder = makeInput(options.projectFolder || "");
  projectFolder.readOnly = true;
  projectFolder.style.display = "none";
  const whisperLanguageOptions = [
    "english",
    "auto",
    "spanish",
    "french",
    "german",
    "italian",
    "portuguese",
    "japanese",
    "korean",
    "chinese",
    "russian",
    "arabic",
    "hindi",
  ];
  const whisperLanguage = makeSelect(whisperLanguageOptions, "english");
  const audioPath = makeInput("");
  const chooseAudioButton = makeButton("Choose Audio", "primary");
  const minDuration = makeInput("4", "number");
  const maxDuration = makeInput("10", "number");
  const bias = makeInput("0.7", "number");
  const durationPreset = makeSelect(["varied_no_repeat", "impact_weighted", "clustered_no_repeat"], "varied_no_repeat");
  const useSrtDurations = makeCheckbox(true);
  const fixedSceneDuration = makeInput("4", "number");
  const emptySegmentText = makeInput("Instrumental section.");
  const appendSubjectToPrompts = makeCheckbox(true);
  const repairLyricSegments = makeCheckbox(false);
  const srtOutput = makeInput("");
  srtOutput.style.display = "none";
  const srtText = makeTextarea("", 6);
  srtText.style.display = "none";
  const audioField = makePickerField("Audio file", audioPath, chooseAudioButton, "Choose the song/audio file used for Whisper and beat-aligned scene timing.");
  audioField.style.flex = "1 1 260px";
  const useSrtField = makeCheckboxField("Use SRT duration file", useSrtDurations, "When enabled, scene timing comes from the beat/SRT duration workflow. When disabled, the extractor uses fixed scene duration.");
  useSrtField.style.flex = "1 1 250px";
  const emptySegmentField = makeField("Empty lyric segment text", emptySegmentText, "Used by VRGDG Lyric Segment Text Cleaner for no-vocal or blank segments.");
  emptySegmentField.style.flex = "1 1 260px";
  const whisperLanguageField = makeField("Whisper language", whisperLanguage, "Language hint for the advanced Whisper/stable-ts extractor. Use auto only when the song language is unknown.");
  whisperLanguageField.style.flex = "1 1 260px";
  const appendSubjectField = makeCheckboxField("Append subject to ConceptPrompts", appendSubjectToPrompts, "When enabled, the extracted subject is added to the start of each concept prompt before saving.");
  appendSubjectField.style.flex = "1 1 260px";
  const repairLyricField = makeCheckboxField("Gemma lyric cleanup", repairLyricSegments);
  repairLyricField.title = "Optional extra Gemma pass that can mark intro/outro lyric bleed as instrumental and clean segment boundaries.";
  repairLyricField.style.cssText += "border:1px solid #3f3f46;border-radius:6px;background:#27272a;padding:6px 8px;";
  const durationPresetHints = [
    "varied_no_repeat: creates varied scene lengths and avoids repeating the same duration pattern back to back.",
    "impact_weighted: favors stronger beat/impact moments when choosing scene boundaries, so cuts feel more tied to the music.",
    "clustered_no_repeat: groups cuts around denser musical moments while still avoiding obvious repeated durations.",
  ];
  const setupControls = [
    whisperLanguageField,
    audioField,
    useSrtField,
    makeCompactField("Fixed scene duration", fixedSceneDuration, "140px", "Used when SRT is off."),
    emptySegmentField,
    appendSubjectField,
    makeCompactHintField("Min duration", minDuration, "110px", "Min Duration", [
      "Smallest scene length the beat/SRT setup should create.",
      "Use this to prevent cuts from becoming too fast or too short.",
    ]),
    makeCompactHintField("Max duration", maxDuration, "110px", "Max Duration", [
      "Largest scene length the beat/SRT setup should create.",
      "Use this to prevent long sections from staying on one scene for too long.",
    ]),
    makeCompactHintField("Bias", bias, "90px", "Bias", [
      "Controls how strongly the timing chooser leans toward preferred beat/duration choices.",
      "Lower values stay more even and conservative. Higher values allow stronger timing choices and more variation.",
      "0.7 is a balanced default.",
    ]),
    makeCompactHintField("Duration preset", durationPreset, "250px", "Duration Presets", durationPresetHints),
  ];
  const setupGrid = document.createElement("div");
  setupGrid.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;";
  setupGrid.append(...setupControls);
  setupPanel.append(setupGrid);

  const inputPanel = makePanel("User Inputs");
  const fullLyrics = makeTextarea("", 10);
  const styleTheme = makeTextarea("", 5);
  const storyIdea = makeTextarea("", 5);
  const subjectLocations = makeTextarea("", 7);
  const makeGemmaInputButton = (label, fieldKey, textarea) => {
    const button = makeButton(label, "primary");
    button.style.padding = "6px 9px";
    button.onclick = async () => {
      await runGemmaInputDraft(fieldKey, textarea, button);
    };
    return button;
  };
  const makeInstructionButton = (key) => {
    const button = makeButton("Edit Instructions");
    button.style.padding = "6px 9px";
    button.onclick = () => openInstructionEditor(key);
    return button;
  };
  const makeGptButton = (url) => {
    const button = makeButton("Use GPT");
    button.style.background = "#6d28d9";
    button.style.borderColor = "#7c3aed";
    button.style.color = "#f5f3ff";
    button.style.padding = "6px 9px";
    button.onclick = () => window.open(url, "_blank", "noopener,noreferrer");
    return button;
  };
  inputPanel.append(
    makePromptTextSection(
      "Full lyrics",
      fullLyrics,
      [
        (() => {
          const button = makeButton("Sonauto");
          button.style.padding = "6px 9px";
          button.onclick = () => window.open("https://sonauto.ai/", "_blank", "noopener,noreferrer");
          return button;
        })(),
        makeGemmaInputButton("Gemma4 Lyrics", "full_lyrics", fullLyrics),
        makeInstructionButton("full_lyrics"),
        repairLyricField,
        makeGptButton(LYRIC_CREATOR_GPT_URL),
      ]
    ),
    makePromptTextSection(
      "Style/theme",
      styleTheme,
      [
        makeGemmaInputButton("Gemma4", "style_theme", styleTheme),
        makeInstructionButton("style_theme"),
        makeGptButton(STYLE_THEME_GPT_URL),
      ]
    ),
    makePromptTextSection(
      "Story idea",
      storyIdea,
      [
        makeGemmaInputButton("Gemma4", "story_idea", storyIdea),
        makeInstructionButton("story_idea"),
        makeGptButton(STORY_IDEA_GPT_URL),
      ]
    ),
    makePromptTextSection(
      "Subject and locations",
      subjectLocations,
      [
        makeGemmaInputButton("Gemma4", "subject_locations", subjectLocations),
        makeInstructionButton("subject_locations"),
        makeGptButton(SUBJECT_LOCATION_GPT_URL),
      ]
    ),
  );
  const inputSections = Array.from(inputPanel.children).slice(1);
  const inputGrid = document.createElement("div");
  inputGrid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;";
  inputGrid.append(...inputSections);
  inputPanel.append(inputGrid);

  const whisperPanel = makePanel("Whisper Segments");
  const whisperSegments = makeCompactPreviewBox("", 3);
  whisperPanel.append(
    makeField("Numbered Whisper segments preview", whisperSegments, "Shown for review only. Downstream uses ConceptPrompts and the extracted subject."),
  );

  const modelPanel = makePanel("Gemma Settings");
  const modelSelect = makeSelect(["Loading models..."]);
  const conceptMatchDescriptions = {
    super_tight_literal: "Super tight and literal: use the lyric's exact visible objects and actions whenever possible.",
    medium: "Medium: keep at least one recognizable lyric object or action while still following story and style.",
    loose: "Loose: use the lyric as inspiration, but allow the story and visual theme to lead.",
    super_light: "Super light: mostly mood and timing; best for abstract visual sequences.",
  };
  const conceptMatchMode = makeSelect(
    ["super_tight_literal", "medium", "loose", "super_light"],
    "medium"
  );
  const conceptMatchInfoButton = makeButton("?");
  conceptMatchInfoButton.style.width = "34px";
  conceptMatchInfoButton.style.padding = "8px 0";
  const conceptMatchRow = document.createElement("div");
  conceptMatchRow.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;align-items:start;";
  conceptMatchRow.append(conceptMatchMode, conceptMatchInfoButton);
  const conceptMatchHint = document.createElement("div");
  conceptMatchHint.textContent = conceptMatchDescriptions[conceptMatchMode.value];
  conceptMatchHint.style.cssText = "color:#a1a1aa;font-size:11px;line-height:1.35;";
  conceptMatchMode.onchange = () => {
    conceptMatchHint.textContent = conceptMatchDescriptions[conceptMatchMode.value] || "";
  };
  conceptMatchInfoButton.onclick = () => showInfoPopup("Concept Lyric Match", [
    conceptMatchDescriptions.super_tight_literal,
    conceptMatchDescriptions.medium,
    conceptMatchDescriptions.loose,
    conceptMatchDescriptions.super_light,
  ]);
  const conceptMatchField = makeField("Concept lyric match", conceptMatchRow);
  conceptMatchField.append(conceptMatchHint);
  conceptMatchField.style.flex = "1 1 360px";
  setupControls.push(conceptMatchField);
  setupGrid.append(conceptMatchField);
  modelPanel.append(
    makeField("Gemma4 text model", modelSelect),
  );

  function syncRunnerControls() {
    controls.textGemmaRunner = state.textGemmaRunner;
    controls.lmStudioBaseUrl = state.lmStudioBaseUrl;
    controls.lmStudioModel = state.lmStudioModel;
    controls.lmStudioApiKey = state.lmStudioApiKey;
    controls.llmApiProvider = state.llmApiProvider;
    controls.llmApiModel = state.llmApiModel;
    controls.llmApiKey = state.llmApiKey;
  }

  function openGemmaRunnerModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100070;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(640px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Prompt Creator LLM Runner</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Prompt Creator is text-only, so it can use Gemma Local, LM Studio, or LLM API.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);
    const runner = makeSelect(["builtin", "lm_studio", "llm_api"], state.textGemmaRunner || "builtin");
    runner.options[0].textContent = "Gemma Local";
    runner.options[1].textContent = "LM Studio";
    runner.options[2].textContent = "LLM API";
    const baseUrl = makeInput(state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1");
    const model = makeInput(state.lmStudioModel || "");
    const modelSelectLm = makeSelect([""], "");
    const loadModels = makeButton("Load LM Studio Models");
    const modelPickerRow = document.createElement("div");
    modelPickerRow.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    const apiKey = makeInput(state.lmStudioApiKey || "", "password");
    const lmPanel = document.createElement("div");
    lmPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const note = document.createElement("div");
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    note.textContent = "In LM Studio, load your Gemma text model, open Local Server, start the server, then load the model list here.";
    modelSelectLm.onchange = () => {
      if (modelSelectLm.value) model.value = modelSelectLm.value;
    };
    loadModels.onclick = async () => {
      loadModels.disabled = true;
      loadModels.textContent = "Loading...";
      try {
        const data = await postJson("/vrgdg/music_builder/lm_studio_models", {
          lmstudio_base_url: baseUrl.value || "http://127.0.0.1:1234/v1",
          lmstudio_api_key: apiKey.value || "",
        });
        const allIds = Array.isArray(data.models) ? data.models.map((item) => String(item || "").trim()).filter(Boolean) : [];
        const ids = allIds.filter((id) => !isLikelyEmbeddingModelId(id));
        if (!allIds.length) throw new Error("LM Studio returned no models. Load a chat model and make sure the local server is running.");
        if (!ids.length) throw new Error("LM Studio only returned embedding models. Load a chat/text-generation model, then try again.");
        modelSelectLm.replaceChildren();
        ids.forEach((id) => {
          const option = document.createElement("option");
          option.value = id;
          option.textContent = id;
          modelSelectLm.append(option);
        });
        const current = String(model.value || "").trim();
        modelSelectLm.value = current && ids.includes(current) ? current : ids[0];
        model.value = modelSelectLm.value;
        setStatus(status, `Loaded ${ids.length} LM Studio model${ids.length === 1 ? "" : "s"}.`);
      } catch (error) {
        setStatus(status, String(error?.message || error), true);
      } finally {
        loadModels.disabled = false;
        loadModels.textContent = "Load LM Studio Models";
      }
    };
    modelPickerRow.append(makeField("Available LM Studio models", modelSelectLm), loadModels);
    lmPanel.append(
      note,
      makeField("LM Studio base URL", baseUrl),
      modelPickerRow,
      makeField("LM Studio model name", model),
      makeField("API key (usually blank for local LM Studio)", apiKey),
    );
    const apiPanel = document.createElement("div");
    apiPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const apiNote = document.createElement("div");
    apiNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    apiNote.textContent = "API key is session-only. It is not saved with the Prompt Creator draft and may need to be pasted again after refresh or restart.";
    const apiProvider = makeSelect(["openai"], state.llmApiProvider || "openai");
    const apiModel = makeSelect([""], state.llmApiModel || "");
    const llmApiKey = makeInput(state.llmApiKey || "", "password");
    const testApi = makeButton("Test LLM API", "primary");
    const apiStatus = document.createElement("div");
    apiStatus.style.cssText = "font-size:12px;color:#94a3b8;min-height:16px;";
    const llmProviders = () => Array.isArray(state.llmApiChoices?.providers) ? state.llmApiChoices.providers : [];
    const providerById = (id) => llmProviders().find((item) => String(item.id || "") === String(id || ""));
    const populateApiModels = () => {
      const provider = providerById(apiProvider.value) || llmProviders()[0] || { id: "openai", label: "OpenAI", models: ["gpt-4o"], default_model: "gpt-4o" };
      const models = Array.isArray(provider.models) && provider.models.length ? provider.models : [provider.default_model || ""].filter(Boolean);
      apiModel.replaceChildren();
      models.forEach((modelId) => {
        const option = document.createElement("option");
        option.value = modelId;
        option.textContent = modelId;
        apiModel.append(option);
      });
      const wanted = String(state.llmApiModel || "").trim();
      apiModel.value = wanted && models.includes(wanted) ? wanted : (provider.default_model || models[0] || "");
    };
    const populateApiProviders = () => {
      const providers = llmProviders().length ? llmProviders() : [{ id: "openai", label: "OpenAI", models: ["gpt-4o"], default_model: "gpt-4o" }];
      apiProvider.replaceChildren();
      providers.forEach((provider) => {
        const option = document.createElement("option");
        option.value = provider.id;
        option.textContent = String(provider.label || provider.id || "");
        apiProvider.append(option);
      });
      const wanted = String(state.llmApiProvider || "").trim();
      apiProvider.value = providers.some((provider) => String(provider.id) === wanted) ? wanted : String(providers[0]?.id || "openai");
      populateApiModels();
    };
    const loadApiChoices = async () => {
      try {
        apiStatus.textContent = "Loading LLM API model list...";
        const data = await getJson("/vrgdg/music_builder/llm_api_choices");
        state.llmApiChoices = { providers: Array.isArray(data.providers) ? data.providers : [] };
        populateApiProviders();
        apiStatus.textContent = "";
      } catch (error) {
        apiStatus.textContent = `Could not load API model list: ${String(error?.message || error)}`;
        apiStatus.style.color = "#fca5a5";
        populateApiProviders();
      }
    };
    apiProvider.onchange = () => {
      state.llmApiProvider = apiProvider.value || "openai";
      state.llmApiModel = "";
      populateApiModels();
    };
    apiModel.onchange = () => {
      state.llmApiModel = apiModel.value || "";
    };
    llmApiKey.oninput = () => {
      state.llmApiKey = llmApiKey.value || "";
      syncRunnerControls();
    };
    apiPanel.append(
      apiNote,
      makeField("Provider", apiProvider),
      makeField("Model", apiModel),
      makeField("API key", llmApiKey),
      testApi,
      apiStatus,
    );
    const syncVisibility = () => {
      state.textGemmaRunner = runner.value || "builtin";
      lmPanel.style.display = runner.value === "lm_studio" ? "flex" : "none";
      apiPanel.style.display = runner.value === "llm_api" ? "flex" : "none";
      syncRunnerControls();
    };
    runner.onchange = syncVisibility;
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Save Runner", "primary");
    actions.append(cancel, save);
    box.append(header, makeField("Text LLM runner", runner), lmPanel, apiPanel, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    syncVisibility();
    loadApiChoices();
    close.onclick = cancel.onclick = () => backdrop.remove();
    save.onclick = async () => {
      state.textGemmaRunner = runner.value || "builtin";
      state.lmStudioBaseUrl = baseUrl.value || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = model.value || "";
      state.lmStudioApiKey = apiKey.value || "";
      state.llmApiProvider = apiProvider.value || "openai";
      state.llmApiModel = apiModel.value || "";
      state.llmApiKey = runner.value === "llm_api" ? llmApiKey.value || "" : state.llmApiKey || "";
      syncRunnerControls();
      setStatus(status, `Prompt Creator runner set to ${state.textGemmaRunner === "llm_api" ? "LLM API" : state.textGemmaRunner === "lm_studio" ? "LM Studio" : "Gemma Local"}.`);
      backdrop.remove();
    };
    testApi.onclick = async () => {
      const provider = apiProvider.value || "openai";
      const modelId = apiModel.value || "";
      const key = llmApiKey.value || "";
      if (!key.trim()) {
        setStatus(status, "Enter an API key before testing LLM API.", true);
        return;
      }
      state.textGemmaRunner = "llm_api";
      state.llmApiProvider = provider;
      state.llmApiModel = modelId;
      state.llmApiKey = key;
      syncRunnerControls();
      testApi.disabled = true;
      testApi.textContent = "Testing...";
      try {
        const data = await postJson("/vrgdg/music_builder/test_llm_api", {
          provider,
          model: modelId,
          api_key: key,
          prompt: "Reply with OK only.",
        });
        apiStatus.textContent = `Test passed: ${data.used_provider || provider} / ${data.used_model || modelId}`;
        apiStatus.style.color = "#67e8f9";
        setStatus(status, "LLM API test passed.");
      } catch (error) {
        const message = String(error?.message || error);
        apiStatus.textContent = `Test failed: ${message}`;
        apiStatus.style.color = "#fca5a5";
        setStatus(status, `LLM API test failed: ${message}`, true);
      } finally {
        testApi.disabled = false;
        testApi.textContent = "Test LLM API";
      }
    };
  }

  function showInstructionWarning(label) {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100075;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = `Advanced Users: ${label} Instructions`;
      heading.style.cssText = "font-size:17px;font-weight:900;color:#fecaca;";
      const note = document.createElement("div");
      note.textContent = "Changing LLM instructions can break output formatting, cause missing JSON keys, or make Gemma return unusable text. Keep required output formats intact. Use Reset To Default if results get weird.";
      note.style.cssText = "font-size:13px;color:#f5d0d0;line-height:1.45;border:1px solid #7f1d1d;background:#450a0a;border-radius:7px;padding:10px;";
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const proceed = makeButton("Continue", "primary");
      actions.append(cancel, proceed);
      box.append(heading, note, actions);
      backdrop.append(box);
      document.body.append(backdrop);
      const finish = (value) => {
        backdrop.remove();
        resolve(value);
      };
      cancel.onclick = () => finish(false);
      proceed.onclick = () => finish(true);
      backdrop.onclick = (event) => {
        if (event.target === backdrop) finish(false);
      };
    });
  }

  function showLyricCleanupWarning() {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100075;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = "Enable Gemma Lyric Cleanup?";
      heading.style.cssText = "font-size:17px;font-weight:900;color:#fecaca;";
      const note = document.createElement("div");
      note.textContent = "This is an advanced optional pass. It can help fix instrumental intro/outro lyric bleed and some segment boundary mistakes, but it adds an extra Gemma run, can slow Prompt Creator down, and may change lyrics that were already correct. If it fails, Prompt Creator will fall back to the raw Whisper segments.";
      note.style.cssText = "font-size:13px;color:#f5d0d0;line-height:1.45;border:1px solid #7f1d1d;background:#450a0a;border-radius:7px;padding:10px;";
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Keep Off");
      const enable = makeButton("Enable Cleanup", "primary");
      actions.append(cancel, enable);
      box.append(heading, note, actions);
      backdrop.append(box);
      document.body.append(backdrop);
      const finish = (value) => {
        backdrop.remove();
        resolve(value);
      };
      cancel.onclick = () => finish(false);
      enable.onclick = () => finish(true);
      backdrop.onclick = (event) => {
        if (event.target === backdrop) finish(false);
      };
    });
  }

  async function chooseInstructionPreset(key) {
    const data = await postJson("/vrgdg/music_prompt_creator/list_instruction_presets", { key });
    const presets = Array.isArray(data.presets) ? data.presets : [];
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100080;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(720px,calc(100vw - 40px));max-height:min(680px,calc(100vh - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = `Load ${instructionLabels[key] || key} Preset`;
      heading.style.cssText = "font-size:17px;font-weight:900;color:#cffafe;";
      const list = document.createElement("div");
      list.style.cssText = "display:flex;flex-direction:column;gap:8px;overflow:auto;max-height:min(440px,52vh);";
      if (!presets.length) {
        const empty = document.createElement("div");
        empty.textContent = "No presets saved for this instruction yet.";
        empty.style.cssText = "border:1px dashed #3f3f46;border-radius:7px;padding:12px;color:#a1a1aa;text-align:center;font-size:12px;";
        list.append(empty);
      } else {
        for (const preset of presets) {
          const row = document.createElement("div");
          row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;";
          const info = document.createElement("div");
          info.style.cssText = "display:flex;flex-direction:column;gap:4px;min-width:0;";
          const name = document.createElement("div");
          name.textContent = preset.name || "Unnamed preset";
          name.style.cssText = "font-size:13px;font-weight:900;color:#f8fafc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
          const meta = document.createElement("div");
          meta.textContent = preset.updated ? new Date(preset.updated * 1000).toLocaleString() : "unknown date";
          meta.style.cssText = "font-size:11px;color:#a1a1aa;";
          info.append(name, meta);
          const load = makeButton("Load", "primary");
          load.onclick = () => {
            backdrop.remove();
            resolve(preset);
          };
          row.append(info, load);
          list.append(row);
        }
      }
      const close = makeButton("Close");
      close.onclick = () => {
        backdrop.remove();
        resolve(null);
      };
      box.append(heading, list, close);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  async function openInstructionEditor(key) {
    const label = instructionLabels[key] || key;
    if (!(await showInstructionWarning(label))) return;
    let data;
    try {
      data = await postJson("/vrgdg/music_prompt_creator/get_instruction", {
        project_folder: projectFolder.value || "",
        key,
      });
    } catch (error) {
      setStatus(status, `Could not load ${label} instructions:\n${error.message}`, true);
      return;
    }
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100076;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(920px,calc(100vw - 40px));height:min(760px,calc(100vh - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid #155e75;background:#083344;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:15px;font-weight:900;color:#cffafe;">Edit ${label} Instructions</div><div style="font-size:11px;color:#a5f3fc;margin-top:3px;">Saved instructions apply only to this Prompt Creator project unless you save them as a preset.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);
    const textarea = makeTextarea(data.text || data.default_text || "", 24);
    textarea.style.height = "100%";
    textarea.style.minHeight = "0";
    textarea.style.resize = "none";
    const body = document.createElement("div");
    body.style.cssText = "min-height:0;padding:14px;display:flex;flex-direction:column;gap:8px;";
    const pathNote = document.createElement("div");
    pathNote.textContent = data.has_custom ? `Using custom instructions: ${data.path}` : "Using built-in default instructions until you save a custom version.";
    pathNote.style.cssText = "font-size:11px;color:#a1a1aa;overflow-wrap:anywhere;";
    body.append(pathNote, textarea);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;padding:12px 14px;border-top:1px solid #155e75;background:#0f172a;";
    const loadPreset = makeButton("Load Preset");
    const savePreset = makeButton("Save As Preset");
    const reset = makeButton("Reset To Default");
    const cancel = makeButton("Cancel");
    const save = makeButton("Save Instructions", "primary");
    actions.append(loadPreset, savePreset, reset, cancel, save);
    box.append(header, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const closeModal = () => backdrop.remove();
    close.onclick = cancel.onclick = closeModal;
    save.onclick = async () => {
      try {
        const saved = await postJson("/vrgdg/music_prompt_creator/save_instruction", {
          project_folder: projectFolder.value || "",
          key,
          text: textarea.value || "",
        });
        pathNote.textContent = `Using custom instructions: ${saved.path}`;
        setStatus(status, `Saved ${label} custom instructions.`);
      } catch (error) {
        setStatus(status, `Could not save ${label} instructions:\n${error.message}`, true);
      }
    };
    reset.onclick = async () => {
      if (!window.confirm(`Reset ${label} instructions to the built-in default for this project?`)) return;
      try {
        const resetData = await postJson("/vrgdg/music_prompt_creator/reset_instruction", {
          project_folder: projectFolder.value || "",
          key,
        });
        textarea.value = resetData.text || resetData.default_text || "";
        pathNote.textContent = "Using built-in default instructions until you save a custom version.";
        setStatus(status, `Reset ${label} instructions to default.`);
      } catch (error) {
        setStatus(status, `Could not reset ${label} instructions:\n${error.message}`, true);
      }
    };
    savePreset.onclick = async () => {
      const name = window.prompt(`Preset name for ${label}:`, "");
      if (!name) return;
      try {
        await postJson("/vrgdg/music_prompt_creator/save_instruction_preset", {
          key,
          name,
          text: textarea.value || "",
        });
        setStatus(status, `Saved ${label} preset: ${name}`);
      } catch (error) {
        setStatus(status, `Could not save preset:\n${error.message}`, true);
      }
    };
    loadPreset.onclick = async () => {
      try {
        const preset = await chooseInstructionPreset(key);
        if (!preset?.name) return;
        const loaded = await postJson("/vrgdg/music_prompt_creator/load_instruction_preset", {
          key,
          name: preset.name,
        });
        textarea.value = loaded.text || "";
        setStatus(status, `Loaded ${label} preset: ${preset.name}. Click Save Instructions to use it in this project.`);
      } catch (error) {
        setStatus(status, `Could not load preset:\n${error.message}`, true);
      }
    };
  }

  const outputsPanel = makePanel("Generated Outputs");
  const repairedOutput = makeCompactPreviewBox("", 3);
  const conceptOutput = makeTextarea("", 14);
  const i2vMotionOutput = makeTextarea("", 10);
  const subjectOutput = makeTextarea("", 5);
  const previewNote = document.createElement("div");
  previewNote.textContent = "Pipeline preview only. These lyric segments are used to create ConceptPrompts, then ConceptPrompts are used downstream by the video builder.";
  previewNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const finalOutputGroup = makePanel("Editable Final Outputs");
  const outputNote = document.createElement("div");
  outputNote.textContent = "These are used downstream. Edit only the ConceptPrompts JSON or subject line if needed, then save.";
  outputNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const saveEditedOutputsButton = makeButton("Save Manual Edits", "primary");
  finalOutputGroup.append(
    outputNote,
    makeField("ConceptPrompts JSON", conceptOutput),
    makeField("I2V Motion Notes JSON", i2vMotionOutput),
    makeField("Extracted subject", subjectOutput),
    saveEditedOutputsButton,
  );
  outputsPanel.append(
    previewNote,
    makeField("Lyric segment JSON", repairedOutput),
    finalOutputGroup,
  );

  const taskPanel = makePanel("Run Tasks");
  const runAllButton = makeButton("Run", "primary");
  const runSkipWhisperButton = makeButton("Run: Skip Whisper/SRT", "primary");
  const instructionRow = document.createElement("div");
  instructionRow.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
  const conceptInstructionsButton = makeInstructionButton("concept_prompts");
  conceptInstructionsButton.textContent = "Edit Concept Instructions";
  const subjectInstructionsButton = makeInstructionButton("subject_extract");
  subjectInstructionsButton.textContent = "Edit Subject Instructions";
  const motionInstructionsButton = makeInstructionButton("i2v_motion_notes");
  motionInstructionsButton.textContent = "Edit I2V Motion Instructions";
  instructionRow.append(conceptInstructionsButton, subjectInstructionsButton, motionInstructionsButton);
  const status = document.createElement("div");
  status.style.cssText = "min-height:44px;border:1px solid #27272a;border-radius:7px;background:#09090b;color:#d4d4d8;padding:10px;font-size:12px;line-height:1.45;white-space:pre-wrap;";
  const runNote = document.createElement("div");
  runNote.textContent = "Runs the full prompt creator pipeline in order and auto-saves the generated files into this project. Use Save Manual Edits only if you change the final outputs afterward.";
  runNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  taskPanel.append(instructionRow, runAllButton, runSkipWhisperButton, runNote, status);

  const controls = {
    projectFolder,
    audioPath,
    minDuration,
    maxDuration,
    bias,
    durationPreset,
    useSrtDurations,
    fixedSceneDuration,
    emptySegmentText,
    whisperLanguage,
    conceptMatchMode,
    appendSubjectToPrompts,
    repairLyricSegments,
    srtOutput,
    fullLyrics,
    styleTheme,
    storyIdea,
    subjectLocations,
    whisperSegments,
    srtText,
  };
  syncRunnerControls();

  function updateDurationModeUi() {
    const usingSrt = useSrtDurations.checked;
    fixedSceneDuration.disabled = usingSrt;
    minDuration.disabled = !usingSrt;
    maxDuration.disabled = !usingSrt;
    bias.disabled = !usingSrt;
    durationPreset.disabled = !usingSrt;
    fixedSceneDuration.style.opacity = usingSrt ? "0.55" : "1";
    minDuration.style.opacity = usingSrt ? "1" : "0.55";
    maxDuration.style.opacity = usingSrt ? "1" : "0.55";
    bias.style.opacity = usingSrt ? "1" : "0.55";
    durationPreset.style.opacity = usingSrt ? "1" : "0.55";
  }
  useSrtDurations.addEventListener("change", updateDurationModeUi);
  updateDurationModeUi();

  repairLyricSegments.addEventListener("change", async () => {
    if (!repairLyricSegments.checked) return;
    const ok = await showLyricCleanupWarning();
    if (!ok) repairLyricSegments.checked = false;
  });

  async function loadConfig() {
    try {
      const config = await getJson("/vrgdg/music_prompt_creator/config");
      if (!config.workflow_template_exists) {
        setStatus(status, `Hidden Whisper workflow is missing:\n${config.workflow_template_path}`);
      }
    } catch (error) {
      setStatus(status, `Could not read prompt creator config:\n${error.message}`);
    }
    try {
      const choices = await getJson("/vrgdg/music_builder/gemma_choices");
      const models = Array.isArray(choices.models) && choices.models.length ? choices.models : ["[none]"];
      modelSelect.replaceChildren();
      for (const value of models) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        modelSelect.append(option);
      }
      const preferred = models.find((item) => /supergemma4.*q4_k_m/i.test(item)) || models[0];
      modelSelect.value = preferred;
    } catch (error) {
      modelSelect.replaceChildren();
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Could not load models";
      modelSelect.append(option);
    }
  }

  function applyDraft(draft) {
    if (!draft || typeof draft !== "object") return;
    audioPath.value = draft.audio_path || audioPath.value || "";
    minDuration.value = draft.min_duration ?? minDuration.value;
    maxDuration.value = draft.max_duration ?? maxDuration.value;
    bias.value = draft.bias ?? bias.value;
    durationPreset.value = draft.duration_preset || durationPreset.value;
    useSrtDurations.checked = draft.use_srt_durations ?? useSrtDurations.checked;
    fixedSceneDuration.value = draft.fixed_scene_duration ?? fixedSceneDuration.value;
    emptySegmentText.value = draft.empty_segment_text || emptySegmentText.value;
    whisperLanguage.value = draft.whisper_language || whisperLanguage.value;
    conceptMatchMode.value = draft.concept_match_mode || conceptMatchMode.value;
    appendSubjectToPrompts.checked = draft.append_subject_to_prompts ?? appendSubjectToPrompts.checked;
    repairLyricSegments.checked = draft.repair_lyric_segments ?? repairLyricSegments.checked;
    conceptMatchHint.textContent = conceptMatchDescriptions[conceptMatchMode.value] || "";
    updateDurationModeUi();
    fullLyrics.value = draft.full_lyrics || "";
    styleTheme.value = draft.style_theme || "";
    storyIdea.value = draft.story_idea || "";
    subjectLocations.value = draft.subject_locations || "";
    whisperSegments.value = draft.whisper_segments || "";
    srtText.value = draft.srt_text || "";
    repairedOutput.value = draft.corrected_segments_text || "";
    conceptOutput.value = draft.concept_prompts_text || "";
    i2vMotionOutput.value = draft.i2v_motion_notes_text || "";
    subjectOutput.value = draft.subject || "";
    state.repairedSegments = parseJsonSafe(repairedOutput.value, {});
    state.conceptPrompts = parseJsonSafe(conceptOutput.value, {});
    state.i2vMotionNotes = parseJsonSafe(i2vMotionOutput.value, {});
    state.extractedSubject = subjectOutput.value || "";
    state.textGemmaRunner = draft.text_gemma_runner || draft.text_runner || draft.textGemmaRunner || state.textGemmaRunner || "builtin";
    state.lmStudioBaseUrl = draft.lm_studio_base_url || draft.lmstudio_base_url || draft.lmStudioBaseUrl || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    state.lmStudioModel = draft.lm_studio_model || draft.lmstudio_model || draft.lmStudioModel || state.lmStudioModel || "";
    state.lmStudioApiKey = draft.lm_studio_api_key || draft.lmstudio_api_key || draft.lmStudioApiKey || state.lmStudioApiKey || "";
    state.llmApiProvider = draft.llm_api_provider || draft.llmApiProvider || state.llmApiProvider || "openai";
    state.llmApiModel = draft.llm_api_model || draft.llmApiModel || state.llmApiModel || "";
    syncRunnerControls();
  }

  async function loadDraft() {
    if (!projectFolder.value) return;
    try {
      const result = await postJson("/vrgdg/music_prompt_creator/load_draft", {
        project_folder: projectFolder.value,
      });
      if (result.found) {
        applyDraft(result.draft || {});
        setStatus(status, `Loaded prompt creator draft.\n${result.draft_path}`);
      }
    } catch (error) {
      setStatus(status, `Could not load prompt creator draft:\n${error.message}`);
    }
  }

  async function chooseAndLoadDraft() {
    let progress = null;
    try {
      loadDraftButton.disabled = true;
      loadDraftButton.textContent = "Loading...";
      const projectData = await getJson("/vrgdg/music_prompt_creator/list_drafts");
      const choice = await showPromptCreatorDraftProjectModal(projectData.projects || []);
      if (!choice?.project_folder) {
        return;
      }
      progress = createProgressWindow("Loading Prompt Creator Draft");
      projectFolder.value = choice.project_folder;
      progress.set("Loading selected project draft...", 65);
      const result = await postJson("/vrgdg/music_prompt_creator/load_draft", {
        project_folder: projectFolder.value,
      });
      if (!result.found) {
        throw new Error(`No prompt creator draft was found for this project.\n${result.draft_path}`);
      }
      applyDraft(result.draft || {});
      setStatus(status, `Loaded prompt creator draft.\n${result.draft_path}`);
      progress.set("Prompt Creator draft loaded.", 100);
      progress.close(900);
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      setStatus(status, String(error?.message || error), true);
    } finally {
      loadDraftButton.disabled = false;
      loadDraftButton.textContent = "Load Project Draft";
    }
  }

  async function chooseAudioFile() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*,video/*";
    input.style.display = "none";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      chooseAudioButton.disabled = true;
      chooseAudioButton.textContent = "Importing...";
      setStatus(status, `Importing audio into this project...\n${file.name}`, true);
      try {
        const form = new FormData();
        form.append("project_folder", projectFolder.value || "");
        form.append("audio", file, file.name);
        const imported = await postForm("/vrgdg/music_prompt_creator/import_audio", form);
        projectFolder.value = imported.project_folder || projectFolder.value;
        audioPath.value = imported.audio_path || file.path || file.name || "";
        setStatus(status, `Audio imported into project.\n${audioPath.value}`);
      } catch (error) {
        audioPath.value = file.path || file.name || "";
        setStatus(status, `Audio import failed:\n${error.message}`, false);
      } finally {
        chooseAudioButton.disabled = false;
        chooseAudioButton.textContent = "Choose Audio";
        input.remove();
      }
    };
    document.body.append(input);
    input.click();
  }

  async function runGemmaInputDraft(fieldKey, textarea, button) {
    const targetByField = {
      full_lyrics: "song_lyrics",
      style_theme: "builder_style_theme",
      story_idea: "builder_story_idea",
      subject_locations: "builder_subjects_and_scenes",
    };
    const target = targetByField[fieldKey];
    const modelFile = String(modelSelect.value || "").trim();
    if (!target) return;
    if (!modelFile && state.textGemmaRunner === "builtin") {
      setStatus(status, "Choose a Gemma4 text model first.");
      return;
    }

    const hintByField = {
      full_lyrics: "Type a song idea, rough lyrics, or any lyric direction. Gemma4 will turn this into full lyrics.",
      style_theme: "Type the style, genre, mood, world, lighting, color, wardrobe, or visual rules you want.",
      story_idea: "Type any story direction you want. Gemma4 will also use the full lyrics and current style/theme.",
      subject_locations: "Type the subject and possible locations you want. Gemma4 will use the current story idea, but your subject/location input takes priority.",
    };
    const titleByField = {
      full_lyrics: "Gemma4 Full Lyrics Input",
      style_theme: "Gemma4 Style/Theme Input",
      story_idea: "Gemma4 Story Idea Input",
      subject_locations: "Gemma4 Subject/Locations Input",
    };
    const userInput = await requestGemmaUserInput(
      titleByField[fieldKey] || "Gemma4 User Input",
      textarea.value || "",
      hintByField[fieldKey] || ""
    );
    if (userInput == null) return;
    const idea = String(userInput || "").trim();
    const payload = {
      project_folder: projectFolder.value || "",
      target,
      model_file: modelFile,
      notes: idea,
      lyrics: fullLyrics.value,
      style_theme: styleTheme.value,
      story_idea: fieldKey === "subject_locations" ? storyIdea.value : (idea || storyIdea.value),
      user_input: idea,
      subject_locations: fieldKey === "subject_locations" ? (idea || subjectLocations.value) : subjectLocations.value,
      unload_after: true,
      n_ctx: fieldKey === "style_theme" ? 8000 : 13000,
      max_new_tokens: fieldKey === "full_lyrics" ? 32000 : fieldKey === "style_theme" ? 600 : 8000,
      temperature: 0.75,
      top_p: 0.95,
      text_runner: state.textGemmaRunner || "builtin",
      lmstudio_base_url: state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
      lmstudio_model: state.lmStudioModel || "",
      lmstudio_api_key: state.lmStudioApiKey || "",
      llm_api_provider: state.llmApiProvider || "openai",
      llm_api_model: state.llmApiModel || "",
      llm_api_key: state.llmApiKey || "",
    };

    let progress = null;
    try {
      button.disabled = true;
      button.textContent = "Gemma...";
      progress = createProgressWindow(`Gemma4 ${fieldKey.replaceAll("_", " ")}`);
      setStatus(status, `Gemma4 is drafting ${fieldKey.replaceAll("_", " ")}...\n${gemmaRunnerLine()}`, true);
      progress.set(`Creating draft from your user input and context...\n${gemmaRunnerLine()}`, 25);
      const data = await postJson("/vrgdg/gemma4/generate", payload);
      const text = String(data.text || "").trim();
      if (!text) throw new Error("Gemma4 returned an empty draft.");
      textarea.value = text;
      progress.set(data.unloaded ? "Draft ready. Gemma4 was unloaded." : "Draft ready.", 100);
      progress.close(900);
      setStatus(status, data.unloaded ? "Draft ready. Gemma4 was unloaded." : "Draft ready.");
    } catch (error) {
      setStatus(status, `Error:\n${error.message}`);
      progress?.set(`Error:\n${error.message}`, 100);
    } finally {
      button.disabled = false;
      button.textContent = fieldKey === "full_lyrics" ? "Gemma4 Lyrics" : "Gemma4";
    }
  }

  async function runWhisperWorkflow(progress = null) {
    setStatus(status, "Running hidden Whisper/SRT workflow...", true);
    progress?.set("Step 1/6: Building hidden Whisper/SRT workflow...", 8);
    const built = await postJson("/vrgdg/music_prompt_creator/build_whisper_prompt", {
      project_folder: projectFolder.value,
      audio_path: audioPath.value,
      min_duration: minDuration.value,
      max_duration: maxDuration.value,
      bias: bias.value,
      duration_preset: durationPreset.value,
      use_srt_durations: useSrtDurations.checked,
      fixed_scene_duration: fixedSceneDuration.value,
      empty_segment_text: emptySegmentText.value,
      whisper_language: whisperLanguage.value,
      full_lyrics: fullLyrics.value,
    });
    progress?.set("Step 1/6: Queuing hidden Whisper/SRT workflow...", 14);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the hidden Whisper workflow but did not return a prompt_id.");
    progress?.set(`Step 1/6: Waiting for Whisper/SRT output...\nPrompt ID: ${promptId}`, 20);
    const text = await waitForPromptCreatorText(promptId, (message) => {
      progress?.set(`${message}\nPrompt ID: ${promptId}`, 28);
    });
    if (!text.whisper) throw new Error("Hidden Whisper workflow finished, but no numbered Whisper segments were found.");
    whisperSegments.value = text.whisper;
    if (text.srt) srtText.value = text.srt;
    srtOutput.value = built.expected_srt_path || srtOutput.value;
    projectFolder.value = built.project_folder || projectFolder.value;
    setStatus(status, "Hidden Whisper/SRT workflow finished.");
    return text;
  }

  async function prepareSegments(progress = null) {
    setStatus(status, "Preparing lyric segments...", true);
    progress?.set("Step 2/6: Preparing lyric segment JSON...", 38);
    state.repairedSegments = parseLyricSegmentsToJson(whisperSegments.value);
    if (repairLyricSegments.checked) {
      try {
        setStatus(status, `Cleaning lyric segments with Gemma...\n${gemmaRunnerLine()}`, true);
        progress?.set(`Step 2/6: Cleaning lyric segments with Gemma...\n${gemmaRunnerLine()}`, 38);
        const payload = buildPayload(controls, modelSelect);
        const result = await postJson("/vrgdg/music_prompt_creator/repair_segments", payload);
        if (result?.segments && Object.keys(result.segments).length) {
          state.repairedSegments = result.segments;
          setStatus(status, `Gemma lyric cleanup finished for ${result.segment_count || Object.keys(result.segments).length} segment(s).`);
        }
      } catch (error) {
        setStatus(status, `Gemma lyric cleanup failed, using raw Whisper segments instead:\n${error.message}`, true);
      }
    }
    const segmentCount = Object.keys(state.repairedSegments).length;
    if (!segmentCount) {
      throw new Error("No lyricSegment lines were found in the extractor output.");
    }
    repairedOutput.value = prettyJson(state.repairedSegments);
    setStatus(status, `Prepared ${segmentCount} lyric segment(s).`);
    return {
      segments: state.repairedSegments,
      segment_count: segmentCount,
    };
  }

  async function createConcepts(progress = null) {
    const payload = buildPayload(controls, modelSelect);
    payload.segments = state.repairedSegments && Object.keys(state.repairedSegments).length
      ? state.repairedSegments
      : parseJsonSafe(repairedOutput.value, {});
    setStatus(status, `Creating visual concept prompts with Gemma...\n${gemmaRunnerLine()}`, true);
    progress?.set(`Step 3/6: Creating ConceptPrompts JSON with Gemma...\n${gemmaRunnerLine()}`, 58);
    const result = await postJson("/vrgdg/music_prompt_creator/create_concepts", payload);
    state.conceptPrompts = result.prompts || {};
    conceptOutput.value = prettyJson(state.conceptPrompts);
    setStatus(status, `Created ${result.prompt_count || 0} concept prompt(s). Gemma unloaded.`);
    return result;
  }

  async function createI2VMotionNotes(progress = null) {
    const payload = buildPayload(controls, modelSelect);
    payload.prompts = state.conceptPrompts && Object.keys(state.conceptPrompts).length
      ? state.conceptPrompts
      : parseJsonSafe(conceptOutput.value, {});
    payload.subject = subjectOutput.value || state.extractedSubject || "";
    setStatus(status, `Creating I2V motion notes with Gemma...\n${gemmaRunnerLine()}`, true);
    progress?.set(`Step 5/6: Creating I2V motion notes JSON with Gemma...\n${gemmaRunnerLine()}`, 84);
    const result = await postJson("/vrgdg/music_prompt_creator/create_i2v_motion_notes", payload);
    state.i2vMotionNotes = result.motion_notes || {};
    i2vMotionOutput.value = prettyJson(state.i2vMotionNotes);
    const debugNote = result.debug_raw_output_path
      ? `\nRaw Gemma output saved:\n${result.debug_raw_output_path}`
      : "";
    const fallbackNote = Number(result.fallback_count || 0)
      ? `\nFallback notes used: ${result.fallback_count}/${result.motion_count || 0}`
      : "";
    setStatus(status, `Created ${result.motion_count || 0} I2V motion note(s). Gemma unloaded.${fallbackNote}${debugNote}`);
    return result;
  }

  async function extractSubject(progress = null) {
    setStatus(status, `Extracting subject line with Gemma...\n${gemmaRunnerLine()}`, true);
    progress?.set(`Step 4/6: Extracting the subject line with Gemma...\n${gemmaRunnerLine()}`, 76);
    const result = await postJson("/vrgdg/music_prompt_creator/extract_subject", buildPayload(controls, modelSelect));
    const previousSubject = state.extractedSubject || subjectOutput.value || "";
    state.extractedSubject = result.subject || "";
    subjectOutput.value = state.extractedSubject;
    if (appendSubjectToPrompts.checked && state.extractedSubject && state.conceptPrompts && Object.keys(state.conceptPrompts).length) {
      state.conceptPrompts = prependSubjectToPrompts(state.conceptPrompts, state.extractedSubject, ", ", [previousSubject]);
      conceptOutput.value = prettyJson(state.conceptPrompts);
    }
    setStatus(status, "Subject extracted. Gemma unloaded.");
    return result;
  }

  async function saveOutputs(progress = null) {
    setStatus(status, "Saving prompt creator outputs into the project folder...", true);
    progress?.set("Step 6/6: Saving prompt creator files into the project folder...", 92);
    const editedConceptPrompts = parseOutputMapping(conceptOutput.value, "Prompt", "ConceptPrompts");
    const editedI2VMotionNotes = parseOutputMapping(i2vMotionOutput.value, "Motion", "I2V Motion Notes");
    state.conceptPrompts = Object.keys(editedConceptPrompts).length ? editedConceptPrompts : (state.conceptPrompts || {});
    state.i2vMotionNotes = Object.keys(editedI2VMotionNotes).length ? editedI2VMotionNotes : (state.i2vMotionNotes || {});
    const payload = buildPayload(controls, modelSelect);
    payload.segments = state.repairedSegments && Object.keys(state.repairedSegments).length
      ? state.repairedSegments
      : parseJsonSafe(repairedOutput.value, {});
    payload.prompts = state.conceptPrompts && Object.keys(state.conceptPrompts).length
      ? state.conceptPrompts
      : parseJsonSafe(conceptOutput.value, {});
    payload.i2v_motion_notes = state.i2vMotionNotes && Object.keys(state.i2vMotionNotes).length
      ? state.i2vMotionNotes
      : parseJsonSafe(i2vMotionOutput.value, {});
    const previousSubject = state.extractedSubject || "";
    payload.subject = subjectOutput.value || state.extractedSubject || "";
    payload.previous_subject = previousSubject;
    if (appendSubjectToPrompts.checked && payload.subject && payload.prompts && Object.keys(payload.prompts).length) {
      payload.prompts = prependSubjectToPrompts(payload.prompts, payload.subject, ", ", [previousSubject]);
      state.conceptPrompts = payload.prompts;
      conceptOutput.value = prettyJson(payload.prompts);
    }
    state.extractedSubject = payload.subject || "";
    const result = await postJson("/vrgdg/music_prompt_creator/save_outputs", payload);
    projectFolder.value = result.project_folder || projectFolder.value;
    setStatus(status, `Saved prompt creator files.\n${result.project_folder}`);
    options.onSaved?.(result);
    return result;
  }

  async function sendToVideoCreator() {
    let progress = null;
    try {
      sendToVideoButton.disabled = true;
      sendToVideoButton.textContent = "Sending...";
      progress = createProgressWindow("Sending To Video Creator");
      const result = await saveOutputs(progress);
      progress.set("Opening Video Creator and importing this Prompt Creator project...", 96);
      if (typeof options.onSendToVideoCreator === "function") {
        await options.onSendToVideoCreator(result);
      } else {
        options.onSaved?.(result);
      }
      progress.set("Sent to Video Creator.", 100);
      progress.close(900);
      overlay.remove();
    } catch (error) {
      const message = error?.message || error;
      setStatus(status, `Error:\n${message}`);
      progress?.set(`Error:\n${message}`, 100);
    } finally {
      sendToVideoButton.disabled = false;
      sendToVideoButton.textContent = "Send To Video Creator";
    }
  }

  async function saveDraft(progress = null) {
    setStatus(status, "Saving prompt creator draft...", true);
    progress?.set("Saving prompt creator draft...", 80);
    const payload = buildPayload(controls, modelSelect);
    payload.corrected_segments_text = repairedOutput.value || "";
    payload.concept_prompts_text = conceptOutput.value || "";
    payload.i2v_motion_notes_text = i2vMotionOutput.value || "";
    payload.subject = subjectOutput.value || "";
    const result = await postJson("/vrgdg/music_prompt_creator/save_draft", payload);
    projectFolder.value = result.project_folder || projectFolder.value;
    setStatus(status, `Saved prompt creator draft.\n${result.draft_path}`);
    return result;
  }

  runAllButton.onclick = async () => {
    let progress = null;
    try {
      runAllButton.disabled = true;
      runSkipWhisperButton.disabled = true;
      runAllButton.textContent = "Running...";
      progress = createProgressWindow("Running Prompt Creator");
      await runWhisperWorkflow(progress);
      await prepareSegments(progress);
      await createConcepts(progress);
      await extractSubject(progress);
      await createI2VMotionNotes(progress);
      await saveOutputs(progress);
      progress.set("Prompt Creator finished. Files were saved into this project.", 100);
      progress.close(1200);
    } catch (error) {
      setStatus(status, `Error:\n${error.message}`);
      progress?.set(`Error:\n${error.message}`, 100);
    } finally {
      runAllButton.disabled = false;
      runSkipWhisperButton.disabled = false;
      runAllButton.textContent = "Run";
    }
  };

  runSkipWhisperButton.onclick = async () => {
    let progress = null;
    try {
      if (!String(whisperSegments.value || "").trim()) {
        throw new Error("No Whisper segments are loaded yet. Run the full pipeline once or load a saved project draft first.");
      }
      runAllButton.disabled = true;
      runSkipWhisperButton.disabled = true;
      runSkipWhisperButton.textContent = "Running...";
      progress = createProgressWindow("Running Prompt Creator");
      progress.set("Skipping Whisper/SRT setup and using the loaded Whisper segments.", 28);
      await prepareSegments(progress);
      await createConcepts(progress);
      await extractSubject(progress);
      await createI2VMotionNotes(progress);
      await saveOutputs(progress);
      progress.set("Prompt Creator finished from saved Whisper/SRT data. Files were saved into this project.", 100);
      progress.close(1200);
    } catch (error) {
      setStatus(status, `Error:\n${error.message}`);
      progress?.set(`Error:\n${error.message}`, 100);
    } finally {
      runAllButton.disabled = false;
      runSkipWhisperButton.disabled = false;
      runSkipWhisperButton.textContent = "Run: Skip Whisper/SRT";
    }
  };
  saveEditedOutputsButton.onclick = async () => {
    try {
      saveEditedOutputsButton.disabled = true;
      saveEditedOutputsButton.textContent = "Saving...";
      await saveOutputs();
    } catch (error) {
      setStatus(status, `Error:\n${error.message}`);
    } finally {
      saveEditedOutputsButton.disabled = false;
      saveEditedOutputsButton.textContent = "Save Manual Edits";
    }
  };
  sendToVideoButton.onclick = sendToVideoCreator;
  saveDraftButton.onclick = async () => {
    let progress = null;
    try {
      saveDraftButton.disabled = true;
      saveDraftButton.textContent = "Saving...";
      progress = createProgressWindow("Saving Prompt Creator Draft");
      await saveDraft(progress);
      progress.set("Draft saved. You can close and come back to this project later.", 100);
      progress.close(1000);
    } catch (error) {
      setStatus(status, `Error:\n${error.message}`);
      progress?.set(`Error:\n${error.message}`, 100);
    } finally {
      saveDraftButton.disabled = false;
      saveDraftButton.textContent = "Save Project Draft";
    }
  };
  loadDraftButton.onclick = chooseAndLoadDraft;
  gemmaRunnerButton.onclick = openGemmaRunnerModal;
  chooseAudioButton.onclick = chooseAudioFile;
  backButton.onclick = () => {
    overlay.remove();
    options.onBack?.();
  };

  const topGrid = document.createElement("div");
  topGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,420px);gap:14px;align-items:start;";
  topGrid.append(setupPanel, modelPanel);

  const processGrid = document.createElement("div");
  processGrid.style.cssText = "display:grid;grid-template-columns:minmax(560px,1fr) minmax(260px,340px);gap:14px;align-items:start;";
  processGrid.append(outputsPanel, whisperPanel);

  body.append(topGrid, inputPanel, taskPanel, processGrid);
  overlay.append(topbar, body);
  document.body.append(overlay);
  loadConfig().then(loadDraft);
  return overlay;
}

window.VRGDGMusicVideoPromptCreator = {
  open: openPromptCreator,
  chooseNewProjectMode: createModeChoiceModal,
};
