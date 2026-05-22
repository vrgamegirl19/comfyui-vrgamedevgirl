import { api } from "../../scripts/api.js";

const PROMPT_CREATOR_VERSION = "prompt-creator-2026-05-21";
const LYRIC_CREATOR_GPT_URL = "https://chatgpt.com/g/g-69979b391cc88191ae4fe298b59c236e-ai-lyric-creator";
const STYLE_THEME_GPT_URL = "https://chatgpt.com/g/g-69fb415a964c8191b4a737f84f37227f-ltx-2-3-style-theme-guide/c/69fb427d-4518-8331-bfd7-505c0f55d2cc";
const STORY_IDEA_GPT_URL = "https://chatgpt.com/g/g-69fb3cb767448191a6caa88be94940d5-ltx-2-3-story-concept-helper/c/69fb3e25-7e74-8326-abd6-7df9cf847a5b";
const SUBJECT_LOCATION_GPT_URL = "https://chatgpt.com/g/g-69fb38a997fc8191a2fa479e44a3c675-ltx-2-3-subject-and-location-creator/c/69fb39e2-2ba0-8328-94c0-6ac9c94d0c89";

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

function makePickerField(label, input, button, hint = "") {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  row.append(input, button);
  return makeField(label, row, hint);
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
    whisper: readText(961) || allText.find((value) => /lyricSegment\s*\d+/i.test(value)) || "",
    srt: readText(962) || allText.find((value) => /-->\s*\d{2}:/i.test(value)) || "",
  };
}

async function waitForPromptCreatorText(promptId, onStatus) {
  const started = Date.now();
  while (Date.now() - started < 30 * 60 * 1000) {
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const text = extractPromptCreatorText(data, promptId);
    if (text.whisper || text.srt) return text;
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

function prettyJson(value) {
  try {
    return JSON.stringify(value || {}, null, 2);
  } catch {
    return "{}";
  }
}

function buildPayload(controls, modelSelect) {
  return {
    project_folder: controls.projectFolder.value,
    audio_path: controls.audioPath.value,
    min_duration: controls.minDuration.value,
    max_duration: controls.maxDuration.value,
    bias: controls.bias.value,
    duration_preset: controls.durationPreset.value,
    model_file: modelSelect.value,
    whisper_segments: controls.whisperSegments.value,
    full_lyrics: controls.fullLyrics.value,
    style_theme: controls.styleTheme.value,
    story_idea: controls.storyIdea.value,
    subject_locations: controls.subjectLocations.value,
    srt_text: controls.srtText.value,
    output_srt_path: controls.srtOutput.value,
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

function openPromptCreator(options = {}) {
  const existing = document.querySelector(".vrgdg-music-prompt-creator");
  if (existing) existing.remove();

  const state = {
    repairedSegments: {},
    conceptPrompts: {},
    extractedSubject: "",
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
  const saveDraftButton = makeButton("Save Project Draft", "primary");
  const closeButton = makeButton("Close");
  closeButton.onclick = () => overlay.remove();
  topbar.append(title, backButton, saveDraftButton, closeButton);

  const body = document.createElement("div");
  body.style.cssText = "min-height:0;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:14px;background:#1f2328;";

  const setupPanel = makePanel("Whisper / SRT Setup");
  const projectFolder = makeInput(options.projectFolder || "");
  projectFolder.readOnly = true;
  projectFolder.style.display = "none";
  const projectFolderNote = document.createElement("div");
  projectFolderNote.textContent = projectFolder.value
    ? `Prompt Creator files will be saved into: ${projectFolder.value}`
    : "Prompt Creator files will be saved into the current project folder.";
  projectFolderNote.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#bae6fd;padding:9px;font-size:11px;line-height:1.4;overflow-wrap:anywhere;";
  const audioPath = makeInput("");
  const chooseAudioButton = makeButton("Choose Audio", "primary");
  const minDuration = makeInput("4", "number");
  const maxDuration = makeInput("10", "number");
  const bias = makeInput("0.7", "number");
  const durationPreset = makeSelect(["varied_no_repeat", "impact_weighted", "clustered_no_repeat"], "varied_no_repeat");
  const srtOutput = makeInput("");
  srtOutput.style.display = "none";
  const srtText = makeTextarea("", 6);
  srtText.style.display = "none";
  const workflowInfo = document.createElement("div");
  workflowInfo.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;word-break:break-word;";
  setupPanel.append(
    projectFolderNote,
    makePickerField("Audio file", audioPath, chooseAudioButton, "Choose the song/audio file used for Whisper and beat-aligned scene timing."),
    makeField("Min duration", minDuration),
    makeField("Max duration", maxDuration),
    makeField("Bias", bias),
    makeField("Duration preset", durationPreset),
    workflowInfo,
  );
  const setupControls = Array.from(setupPanel.children).slice(1);
  const setupGrid = document.createElement("div");
  setupGrid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;align-items:start;";
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
        makeGemmaInputButton("Gemma4 Lyrics", "full_lyrics", fullLyrics),
        makeGptButton(LYRIC_CREATOR_GPT_URL),
      ],
      "Type a song idea or rough lyrics, then Gemma4 or GPT can help turn it into full lyrics."
    ),
    makePromptTextSection(
      "Style/theme",
      styleTheme,
      [
        makeGemmaInputButton("Gemma4", "style_theme", styleTheme),
        makeGptButton(STYLE_THEME_GPT_URL),
      ],
      "Gemma uses the text in this box as the user idea and unloads after the draft is created."
    ),
    makePromptTextSection(
      "Story idea",
      storyIdea,
      [
        makeGemmaInputButton("Gemma4", "story_idea", storyIdea),
        makeGptButton(STORY_IDEA_GPT_URL),
      ],
      "Gemma uses the full lyrics, current style/theme, and your extra input to create a story idea."
    ),
    makePromptTextSection(
      "Subject and locations",
      subjectLocations,
      [
        makeGemmaInputButton("Gemma4", "subject_locations", subjectLocations),
        makeGptButton(SUBJECT_LOCATION_GPT_URL),
      ],
      "Gemma uses the current story idea plus your extra input. Your extra subject/location details take priority."
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
  const settingsNote = document.createElement("div");
  settingsNote.textContent = "Uses non-vision Gemma. unload_after_run=true, n_ctx=14848, max_new_tokens=32000, temperature=0.30, top_p=0.80.";
  settingsNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  modelPanel.append(makeField("Gemma4 text model", modelSelect), settingsNote);

  const outputsPanel = makePanel("Generated Outputs");
  const repairedOutput = makeCompactPreviewBox("", 3);
  const conceptOutput = makeTextarea("", 14);
  const subjectOutput = makeTextarea("", 5);
  const previewNote = document.createElement("div");
  previewNote.textContent = "Pipeline preview only. These repaired lyric segments are not used downstream by the video builder.";
  previewNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const finalOutputGroup = makePanel("Editable Final Outputs");
  const outputNote = document.createElement("div");
  outputNote.textContent = "These are used downstream. Edit only the ConceptPrompts JSON or subject line if needed, then save.";
  outputNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const saveEditedOutputsButton = makeButton("Save Manual Edits", "primary");
  finalOutputGroup.append(
    outputNote,
    makeField("ConceptPrompts JSON", conceptOutput),
    makeField("Extracted subject", subjectOutput),
    saveEditedOutputsButton,
  );
  outputsPanel.append(
    previewNote,
    makeField("Corrected lyric segment JSON", repairedOutput),
    finalOutputGroup,
  );

  const taskPanel = makePanel("Run Tasks");
  const runAllButton = makeButton("Run", "primary");
  const status = document.createElement("div");
  status.style.cssText = "min-height:44px;border:1px solid #27272a;border-radius:7px;background:#09090b;color:#d4d4d8;padding:10px;font-size:12px;line-height:1.45;white-space:pre-wrap;";
  const runNote = document.createElement("div");
  runNote.textContent = "Runs the full prompt creator pipeline in order and auto-saves the generated files into this project. Use Save Manual Edits only if you change the final outputs afterward.";
  runNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  taskPanel.append(runAllButton, runNote, status);

  const controls = {
    projectFolder,
    audioPath,
    minDuration,
    maxDuration,
    bias,
    durationPreset,
    srtOutput,
    fullLyrics,
    styleTheme,
    storyIdea,
    subjectLocations,
    whisperSegments,
    srtText,
  };

  async function loadConfig() {
    try {
      const config = await getJson("/vrgdg/music_prompt_creator/config");
      workflowInfo.textContent = config.workflow_template_exists
        ? `Hidden Whisper workflow found: ${config.workflow_template_path}`
        : `Hidden Whisper workflow is missing: ${config.workflow_template_path}`;
    } catch (error) {
      workflowInfo.textContent = `Could not read prompt creator config: ${error.message}`;
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
    fullLyrics.value = draft.full_lyrics || "";
    styleTheme.value = draft.style_theme || "";
    storyIdea.value = draft.story_idea || "";
    subjectLocations.value = draft.subject_locations || "";
    whisperSegments.value = draft.whisper_segments || "";
    srtText.value = draft.srt_text || "";
    repairedOutput.value = draft.corrected_segments_text || "";
    conceptOutput.value = draft.concept_prompts_text || "";
    subjectOutput.value = draft.subject || "";
    state.repairedSegments = parseJsonSafe(repairedOutput.value, {});
    state.conceptPrompts = parseJsonSafe(conceptOutput.value, {});
    state.extractedSubject = subjectOutput.value || "";
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
        projectFolderNote.textContent = projectFolder.value
          ? `Prompt Creator files will be saved into: ${projectFolder.value}`
          : "Prompt Creator files will be saved into the current project folder.";
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
    if (!modelFile) {
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
      target,
      model_file: modelFile,
      notes: idea,
      lyrics: fullLyrics.value,
      style_theme: styleTheme.value,
      story_idea: fieldKey === "subject_locations" ? storyIdea.value : (idea || storyIdea.value),
      user_input: idea,
      subject_locations: fieldKey === "subject_locations" ? (idea || subjectLocations.value) : subjectLocations.value,
      unload_after: true,
      n_ctx: 13000,
      max_new_tokens: fieldKey === "full_lyrics" ? 32000 : 8000,
      temperature: 0.75,
      top_p: 0.95,
    };

    let progress = null;
    try {
      button.disabled = true;
      button.textContent = "Gemma...";
      progress = createProgressWindow(`Gemma4 ${fieldKey.replaceAll("_", " ")}`);
      setStatus(status, `Gemma4 is drafting ${fieldKey.replaceAll("_", " ")}...`, true);
      progress.set("Creating draft from your user input and context...", 25);
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
    progress?.set("Step 1/5: Building hidden Whisper/SRT workflow...", 8);
    const built = await postJson("/vrgdg/music_prompt_creator/build_whisper_prompt", {
      project_folder: projectFolder.value,
      audio_path: audioPath.value,
      min_duration: minDuration.value,
      max_duration: maxDuration.value,
      bias: bias.value,
      duration_preset: durationPreset.value,
    });
    progress?.set("Step 1/5: Queuing hidden Whisper/SRT workflow...", 14);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the hidden Whisper workflow but did not return a prompt_id.");
    progress?.set(`Step 1/5: Waiting for Whisper/SRT output...\nPrompt ID: ${promptId}`, 20);
    const text = await waitForPromptCreatorText(promptId, (message) => {
      progress?.set(`${message}\nPrompt ID: ${promptId}`, 28);
    });
    if (!text.whisper) throw new Error("Hidden Whisper workflow finished, but no numbered Whisper segments were found.");
    whisperSegments.value = text.whisper;
    if (text.srt) srtText.value = text.srt;
    srtOutput.value = built.expected_srt_path || srtOutput.value;
    projectFolder.value = built.project_folder || projectFolder.value;
    projectFolderNote.textContent = projectFolder.value
      ? `Prompt Creator files will be saved into: ${projectFolder.value}`
      : "Prompt Creator files will be saved into the current project folder.";
    setStatus(status, "Hidden Whisper/SRT workflow finished.");
    return text;
  }

  async function repairSegments(progress = null) {
    setStatus(status, "Repairing Whisper segments with Gemma...", true);
    progress?.set("Step 2/5: Repairing Whisper segments with Gemma...", 38);
    const result = await postJson("/vrgdg/music_prompt_creator/repair_segments", buildPayload(controls, modelSelect));
    state.repairedSegments = result.segments || {};
    repairedOutput.value = prettyJson(state.repairedSegments);
    setStatus(status, `Repaired ${result.segment_count || 0} segment(s). Gemma unloaded.`);
    return result;
  }

  async function createConcepts(progress = null) {
    const payload = buildPayload(controls, modelSelect);
    payload.segments = state.repairedSegments && Object.keys(state.repairedSegments).length
      ? state.repairedSegments
      : parseJsonSafe(repairedOutput.value, {});
    setStatus(status, "Creating visual concept prompts with Gemma...", true);
    progress?.set("Step 3/5: Creating ConceptPrompts JSON with Gemma...", 58);
    const result = await postJson("/vrgdg/music_prompt_creator/create_concepts", payload);
    state.conceptPrompts = result.prompts || {};
    conceptOutput.value = prettyJson(state.conceptPrompts);
    setStatus(status, `Created ${result.prompt_count || 0} concept prompt(s). Gemma unloaded.`);
    return result;
  }

  async function extractSubject(progress = null) {
    setStatus(status, "Extracting subject line with Gemma...", true);
    progress?.set("Step 4/5: Extracting the subject line with Gemma...", 76);
    const result = await postJson("/vrgdg/music_prompt_creator/extract_subject", buildPayload(controls, modelSelect));
    state.extractedSubject = result.subject || "";
    subjectOutput.value = state.extractedSubject;
    setStatus(status, "Subject extracted. Gemma unloaded.");
    return result;
  }

  async function saveOutputs(progress = null) {
    setStatus(status, "Saving prompt creator outputs into the project folder...", true);
    progress?.set("Step 5/5: Saving prompt creator files into the project folder...", 90);
    const payload = buildPayload(controls, modelSelect);
    payload.segments = state.repairedSegments && Object.keys(state.repairedSegments).length
      ? state.repairedSegments
      : parseJsonSafe(repairedOutput.value, {});
    payload.prompts = state.conceptPrompts && Object.keys(state.conceptPrompts).length
      ? state.conceptPrompts
      : parseJsonSafe(conceptOutput.value, {});
    payload.subject = subjectOutput.value || state.extractedSubject || "";
    const result = await postJson("/vrgdg/music_prompt_creator/save_outputs", payload);
    projectFolder.value = result.project_folder || projectFolder.value;
    projectFolderNote.textContent = projectFolder.value
      ? `Prompt Creator files will be saved into: ${projectFolder.value}`
      : "Prompt Creator files will be saved into the current project folder.";
    setStatus(status, `Saved prompt creator files.\n${result.project_folder}`);
    options.onSaved?.(result);
    return result;
  }

  async function saveDraft(progress = null) {
    setStatus(status, "Saving prompt creator draft...", true);
    progress?.set("Saving prompt creator draft...", 80);
    const payload = buildPayload(controls, modelSelect);
    payload.corrected_segments_text = repairedOutput.value || "";
    payload.concept_prompts_text = conceptOutput.value || "";
    payload.subject = subjectOutput.value || "";
    const result = await postJson("/vrgdg/music_prompt_creator/save_draft", payload);
    projectFolder.value = result.project_folder || projectFolder.value;
    projectFolderNote.textContent = projectFolder.value
      ? `Prompt Creator files will be saved into: ${projectFolder.value}`
      : "Prompt Creator files will be saved into the current project folder.";
    setStatus(status, `Saved prompt creator draft.\n${result.draft_path}`);
    return result;
  }

  runAllButton.onclick = async () => {
    let progress = null;
    try {
      runAllButton.disabled = true;
      runAllButton.textContent = "Running...";
      progress = createProgressWindow("Running Prompt Creator");
      await runWhisperWorkflow(progress);
      await repairSegments(progress);
      await createConcepts(progress);
      await extractSubject(progress);
      await saveOutputs(progress);
      progress.set("Prompt Creator finished. Files were saved into this project.", 100);
      progress.close(1200);
    } catch (error) {
      setStatus(status, `Error:\n${error.message}`);
      progress?.set(`Error:\n${error.message}`, 100);
    } finally {
      runAllButton.disabled = false;
      runAllButton.textContent = "Run";
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
