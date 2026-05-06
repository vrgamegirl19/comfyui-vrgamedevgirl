import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_PromptCreatorUI_V2";
const MODAL_ID = "vrgdg-prompt-creator-ui-v2-modal";
const BANNER_URL = new URL("./ChatGPT Image May 5, 2026, 08_07_18 PM.png?v=20260505_2020_refresh", import.meta.url).href;
const TEXT_FIELDS = [
  {
    key: "full_lyrics",
    label: "Full Lyrics",
    placeholder: "Enter full lyrics",
    rows: 10,
    defaultValue: "Full Lyrics\n\n",
    helperText:
      "Enter the full lyrics or audio transcript here. This box automatically keeps the file header as: Full Lyrics",
  },
  {
    key: "style_theme",
    label: "Style/theme",
    placeholder: "Enter style/theme",
    rows: 6,
    defaultValue: `Surreal cinematic showroom aesthetic. Begin with sterile whites, cold grays, flat lighting, rigid symmetry, and polite stillness. Gradually shift toward harsh shadows, electric blues, defiant reds, cracked porcelain, broken symmetry, close-ups, Dutch angles, and low-angle heroic framing. Mood: controlled, eerie, then powerful and liberating.`,
    helperText:
      "Describe the overall visual language: art style, colors, mood, camera style, lighting, texture, and any rules that should apply to every scene.",
  },
  {
    key: "story_idea",
    label: "Story idea",
    placeholder: "Enter short story idea",
    rows: 6,
    defaultValue: `A singer is the only real person in a sterile showroom world filled with silent porcelain mannequins. Her honest voice exposes the fake perfection around her: porcelain cracks, staged rooms lose symmetry, and sterile cream lighting shifts into deep shadows, blues, and reds. Keep the video focused on her, the mannequins, and the illusion of polite control breaking apart. By the end, she stands powerful and free in the shattered showroom.`,
    helperText:
      "Enter a short story concept, or simply write something like: create a story for me based off lyrics and style/theme.",
  },
  {
    key: "subjects_and_scenes",
    label: "Subject and Locations",
    placeholder: "Enter subject and location details",
    rows: 6,
    defaultValue: "",
    helperText:
      "List the important characters, outfits, objects, recurring places, and locations. Include enough detail that later prompts can describe them consistently.",
  },
];

const SUBGRAPH_TARGET_NODE_ID = 28;
const SUBGRAPH_WIDGET_ORDER = [
  "fps",
  "output_filename",
  "min_duration",
  "max_duration",
  "bias",
  "seed",
  "duration_preset",
  "section_2_text",
  "section_4_text",
  "section_4_text_1",
  "language",
];
const SUBGRAPH_FIELDS = [
  {
    key: "fps",
    label: "FPS",
    type: "number",
    step: "1",
    defaultValue: "24",
    note: "Frames per second used by the subgraph timing/video logic. Set this to the same FPS in the Part 2 workflow.",
  },
  {
    key: "min_duration",
    label: "Min Duration",
    type: "number",
    step: "0.1",
    defaultValue: "3.0",
    note: "Minimum length of scene.",
  },
  {
    key: "max_duration",
    label: "Max Duration",
    type: "number",
    step: "0.1",
    defaultValue: "8.0",
    note: "Maximum length of scene.",
  },
  {
    key: "bias",
    label: "Bias",
    type: "number",
    step: "0.01",
    defaultValue: "0.60",
    note: "Controls how strongly beat impact affects scene cuts. Lower values are more even/random; higher values favor stronger beats and downbeats more.",
  },
  {
    key: "duration_preset",
    label: "Duration Preset",
    type: "select",
    defaultValue: "varied_no_repeat",
    options: ["impact_weighted", "varied_no_repeat", "clustered_no_repeat"],
    note: "impact_weighted follows strongest beats. varied_no_repeat avoids similar scene lengths back-to-back. clustered_no_repeat keeps lengths closer together while still avoiding repeats.",
  },
  {
    key: "language",
    label: "Whisper Language",
    type: "select",
    defaultValue: "auto",
    options: [
      "auto",
      "english",
      "spanish",
      "french",
      "german",
      "italian",
      "portuguese",
      "japanese",
      "korean",
      "chinese",
    ],
    note: "Language hint for Whisper transcription. Use auto to let Whisper detect it, or pick the song language for more consistent lyric timing.",
  },
];

function getStoredFieldValue(node, field) {
  const propertyName = `vrgdg_test_popup_${field.key}`;
  if (Object.prototype.hasOwnProperty.call(node.properties || {}, propertyName)) {
    return String(node.properties[propertyName] || "");
  }
  return String(field.defaultValue || "");
}

function getStoredSubgraphValue(node, field) {
  const propertyName = `vrgdg_test_popup_subgraph_${field.key}`;
  if (Object.prototype.hasOwnProperty.call(node.properties || {}, propertyName)) {
    return String(node.properties[propertyName] || "");
  }
  return String(field.defaultValue || "");
}

function hasStoredSubgraphValue(node, field) {
  const propertyName = `vrgdg_test_popup_subgraph_${field.key}`;
  return Object.prototype.hasOwnProperty.call(node.properties || {}, propertyName);
}

function formatFieldForDisplay(field, value) {
  if (field.key !== "full_lyrics") {
    return String(value || "");
  }

  const text = String(value || "").replace(/^\s+/, "");
  if (!text) {
    return "Full Lyrics\n\n";
  }
  if (/^Full Lyrics\b/i.test(text)) {
    return text;
  }
  return `Full Lyrics\n\n${text}`;
}

function formatFieldForSave(field, value) {
  return formatFieldForDisplay(field, value).trimEnd();
}

function createButton(label, styles = "") {
  const button = document.createElement("button");
  button.textContent = label;
  button.style.cssText = `
    border-radius: 8px;
    padding: 10px 14px;
    cursor: pointer;
    font-size: 13px;
    ${styles}
  `;
  return button;
}

function createPathHint() {
  const hint = document.createElement("div");
  hint.style.cssText = `
    font-size: 12px;
    line-height: 1.4;
    color: #94a3b8;
    margin-top: 6px;
    word-break: break-all;
  `;
  return hint;
}

function createSubgraphInput(field) {
  const input = field.type === "select" ? document.createElement("select") : document.createElement("input");
  if (field.type === "select") {
    for (const optionValue of field.options || []) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue;
      input.appendChild(option);
    }
  } else {
    input.type = field.type || "text";
    if (field.step) input.step = field.step;
  }
  input.style.cssText = `
    width: 100%;
    box-sizing: border-box;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid #4b5563;
    background: #0d1217;
    color: #f3f4f6;
    font-size: 13px;
  `;
  return input;
}

function coerceSubgraphValue(field, value) {
  const text = String(value ?? "").trim();
  if (field.type !== "number") {
    return text;
  }
  const numberValue = Number(text);
  return Number.isFinite(numberValue) ? numberValue : Number(field.defaultValue || 0);
}

function findTheGutsSubgraphNode() {
  const graph = app.graph;
  if (!graph) return null;

  const nodeById = graph.getNodeById?.(SUBGRAPH_TARGET_NODE_ID);
  if (nodeById) return nodeById;

  const nodes = Array.isArray(graph._nodes) ? graph._nodes : [];
  const byTitle = nodes.find((graphNode) => {
    const title = String(graphNode?.title || graphNode?.name || "").trim().toLowerCase();
    return title === "the guts" || title === "new subgraph";
  });
  if (byTitle) return byTitle;

  const requiredWidgetNames = new Set(SUBGRAPH_FIELDS.map((field) => field.key));
  return nodes.find((graphNode) => {
    const widgetNames = new Set((graphNode?.widgets || []).map((widget) => String(widget?.name || "")));
    return [...requiredWidgetNames].every((widgetName) => widgetNames.has(widgetName));
  }) || null;
}

function setSubgraphWidgetValue(targetNode, field, value) {
  const widget = (targetNode.widgets || []).find((item) => item?.name === field.key);
  if (widget) {
    widget.value = value;
    widget.callback?.(value, app.canvas, targetNode, app.canvas?.graph_mouse);
    return true;
  }

  const widgetIndex = SUBGRAPH_WIDGET_ORDER.indexOf(field.key);
  if (widgetIndex >= 0 && Array.isArray(targetNode.widgets_values)) {
    targetNode.widgets_values[widgetIndex] = value;
    return true;
  }

  return false;
}

function getSubgraphWidgetValue(targetNode, field) {
  const widget = (targetNode?.widgets || []).find((item) => item?.name === field.key);
  if (widget) {
    return widget.value;
  }

  const widgetIndex = SUBGRAPH_WIDGET_ORDER.indexOf(field.key);
  if (widgetIndex >= 0 && Array.isArray(targetNode?.widgets_values)) {
    return targetNode.widgets_values[widgetIndex];
  }

  return undefined;
}

async function fetchConfig() {
  const response = await api.fetchApi("/vrgdg/test_popup/config", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Config load failed (${response.status})`);
  }
  const data = await response.json();
  if (!data?.ok) {
    throw new Error(String(data?.error || "Config load failed"));
  }
  return data;
}

async function uploadAudio(file) {
  const form = new FormData();
  form.append("audio", file);

  const response = await api.fetchApi("/vrgdg/test_popup/upload_audio", {
    method: "POST",
    body: form,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) {
    throw new Error(String(data?.error || `Audio upload failed (${response.status})`));
  }
  return data;
}

async function saveText(payload) {
  const response = await api.fetchApi("/vrgdg/test_popup/save_text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) {
    throw new Error(String(data?.error || `Save failed (${response.status})`));
  }
  return data;
}

function ensureModal() {
  let overlay = document.getElementById(MODAL_ID);
  if (overlay) return overlay;

  overlay = document.createElement("div");
  overlay.id = MODAL_ID;
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.52);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    padding: 16px;
  `;

  const panel = document.createElement("div");
  panel.style.cssText = `
    width: min(1920px, calc(100vw - 32px));
    max-height: calc(100vh - 32px);
    overflow: auto;
    background: #1f2328;
    color: #f3f4f6;
    border: 1px solid #364152;
    border-radius: 12px;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
    padding: 18px;
    font-family: Arial, sans-serif;
  `;

  const titleRow = document.createElement("div");
  titleRow.style.cssText = `
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
  `;

  const banner = document.createElement("img");
  banner.src = BANNER_URL;
  banner.alt = "VRGDG Prompt Creator banner";
  banner.style.cssText = `
    display: block;
    width: 100%;
    height: auto;
    max-height: 200px;
    object-fit: contain;
    border-radius: 10px;
    border: 1px solid #364152;
    margin-bottom: 14px;
  `;

  const titleBlock = document.createElement("div");

  // const title = document.createElement("div");
  // title.textContent = "VRGDG Prompt Creator V2";
  // title.style.cssText = "font-size: 20px; font-weight: 700;";

  const subtitle = document.createElement("div");
  subtitle.textContent = "Save full lyrics, style/theme, story idea, and subject.";
  subtitle.style.cssText = "margin-top: 4px; font-size: 13px; color: #94a3b8;";

  titleBlock.append(subtitle);

  const closeButton = createButton(
    "Close UI Window",
    "border: 1px solid #dc2626; background: #ef4444; color: white; padding: 13px 20px; font-size: 14px; font-weight: 700;"
  );

  const audioSection = document.createElement("div");
  audioSection.style.cssText = `
    border: 1px solid #364152;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 16px;
    background: #14191f;
  `;

  const audioTitle = document.createElement("div");
  audioTitle.textContent = "Audio upload";
  audioTitle.style.cssText = "font-size: 15px; font-weight: 700; margin-bottom: 8px;";

  const audioHint = createPathHint();
  const audioFileName = document.createElement("div");
  audioFileName.style.cssText = "margin: 8px 0; font-size: 13px; color: #cbd5e1;";
  audioFileName.textContent = "No audio file selected.";

  const audioActions = document.createElement("div");
  audioActions.style.cssText = "display: flex; gap: 10px; align-items: center; flex-wrap: wrap;";

  const chooseAudioButton = createButton(
    "Choose and Upload Audio",
    "border: 1px solid #0f766e; background: #0f766e; color: white;"
  );

  const topSaveButton = createButton(
    "Save Text Files",
    "border: 1px solid #1d4ed8; background: #2563eb; color: white;"
  );

  const hiddenAudioInput = document.createElement("input");
  hiddenAudioInput.type = "file";
  hiddenAudioInput.accept = "audio/*,video/*";
  hiddenAudioInput.style.display = "none";

  audioActions.append(chooseAudioButton, topSaveButton, hiddenAudioInput);
  audioSection.append(audioTitle, audioHint, audioFileName, audioActions);

  const subgraphSection = document.createElement("div");
  subgraphSection.style.cssText = `
    border: 1px solid #364152;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 16px;
    background: #14191f;
  `;

  const subgraphTitleRow = document.createElement("div");
  subgraphTitleRow.style.cssText = `
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  `;

  const subgraphTitleBlock = document.createElement("div");

  const subgraphTitle = document.createElement("div");
  subgraphTitle.textContent = "The Guts Subgraph Settings";
  subgraphTitle.style.cssText = "font-size: 15px; font-weight: 700;";

  const subgraphHint = document.createElement("div");
  subgraphHint.textContent = "Applies these values to the open workflow subgraph, using node #28 first.";
  subgraphHint.style.cssText = "margin-top: 4px; font-size: 12px; color: #94a3b8;";

  subgraphTitleBlock.append(subgraphTitle, subgraphHint);

  const applySubgraphButton = createButton(
    "Apply Settings",
    "border: 1px solid #b45309; background: #d97706; color: white;"
  );

  subgraphTitleRow.append(subgraphTitleBlock, applySubgraphButton);

  const subgraphGrid = document.createElement("div");
  subgraphGrid.style.cssText = `
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
  `;

  const subgraphInputs = {};
  for (const field of SUBGRAPH_FIELDS) {
    const fieldWrap = document.createElement("label");
    fieldWrap.style.cssText = "display: block; font-size: 12px; color: #cbd5e1;";

    const fieldLabel = document.createElement("div");
    fieldLabel.textContent = field.label;
    fieldLabel.style.cssText = "margin-bottom: 5px; font-weight: 700;";

    const input = createSubgraphInput(field);

    const fieldNote = document.createElement("div");
    fieldNote.textContent = field.note || "";
    fieldNote.style.cssText = `
      margin-top: 5px;
      color: #94a3b8;
      font-size: 11px;
      line-height: 1.35;
    `;

    fieldWrap.append(fieldLabel, input, fieldNote);
    subgraphGrid.appendChild(fieldWrap);
    subgraphInputs[field.key] = input;
  }

  subgraphSection.append(subgraphTitleRow, subgraphGrid);

  const textGrid = document.createElement("div");
  textGrid.style.cssText = `
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 14px;
  `;

  const textareas = {};
  const pathHints = {};
  for (const field of TEXT_FIELDS) {
    const section = document.createElement("div");
    section.style.cssText = `
      border: 1px solid #364152;
      border-radius: 8px;
      padding: 14px;
      background: #14191f;
    `;

    const label = document.createElement("label");
    label.textContent = field.label;
    label.style.cssText = "display: block; margin-bottom: 8px; font-size: 14px; font-weight: 700;";

    const textarea = document.createElement("textarea");
    textarea.rows = field.rows;
    textarea.placeholder = field.placeholder;
    textarea.style.cssText = `
      width: 100%;
      box-sizing: border-box;
      resize: vertical;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #4b5563;
      background: #0d1217;
      color: #f3f4f6;
      font-size: 13px;
      line-height: 1.45;
    `;

    const helper = document.createElement("div");
    helper.textContent = field.helperText || "";
    helper.style.cssText = `
      margin-top: 6px;
      color: #94a3b8;
      font-size: 11px;
      line-height: 1.35;
    `;

    const hint = createPathHint();

    section.append(label, textarea, helper, hint);
    textGrid.appendChild(section);
    textareas[field.key] = textarea;
    pathHints[field.key] = hint;
  }

  const status = document.createElement("div");
  status.style.cssText = `
    min-height: 20px;
    margin-top: 16px;
    margin-bottom: 14px;
    font-size: 13px;
    color: #cbd5e1;
    white-space: pre-wrap;
  `;

  const actions = document.createElement("div");
  actions.style.cssText = "display: flex; gap: 10px; justify-content: flex-end; margin-top: 8px;";

  const saveButton = createButton(
    "Save Text Files",
    "border: 1px solid #1d4ed8; background: #2563eb; color: white;"
  );

  titleRow.append(titleBlock, closeButton);
  actions.append(saveButton);
  panel.append(banner, titleRow, audioSection, subgraphSection, textGrid, status, actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  const state = {
    node: null,
    config: null,
    status,
    saveButton,
    topSaveButton,
    chooseAudioButton,
    hiddenAudioInput,
    audioFileName,
    audioHint,
    textareas,
    pathHints,
    subgraphInputs,
  };

  function setStatus(message, isError = false) {
    status.textContent = message || "";
    status.style.color = isError ? "#fca5a5" : "#cbd5e1";
  }

  function closeModal() {
    overlay.style.display = "none";
    state.node = null;
    setStatus("");
  }

  function syncNodeProperties() {
    if (!state.node) return;
    state.node.properties = state.node.properties || {};
    for (const field of TEXT_FIELDS) {
      state.node.properties[`vrgdg_test_popup_${field.key}`] = String(textareas[field.key].value || "");
    }
    for (const field of SUBGRAPH_FIELDS) {
      state.node.properties[`vrgdg_test_popup_subgraph_${field.key}`] = String(subgraphInputs[field.key].value || "");
    }
  }

  function applySubgraphSettings() {
    const targetNode = findTheGutsSubgraphNode();
    if (!targetNode) {
      setStatus("Could not find The Guts subgraph. Open the workflow and make sure node #28 or a subgraph with these widgets exists.", true);
      return;
    }

    let updated = 0;
    const missing = [];
    for (const field of SUBGRAPH_FIELDS) {
      const value = coerceSubgraphValue(field, subgraphInputs[field.key].value);
      if (setSubgraphWidgetValue(targetNode, field, value)) {
        updated += 1;
      } else {
        missing.push(field.key);
      }
    }

    syncNodeProperties();
    app.graph?.setDirtyCanvas?.(true, true);

    const targetName = String(targetNode.title || targetNode.name || targetNode.id || "subgraph");
    if (missing.length) {
      setStatus(`Updated ${updated} fields on ${targetName}. Missing widgets: ${missing.join(", ")}`, true);
      return;
    }
    setStatus(`Updated ${updated} fields on ${targetName}.`);
  }

  async function ensureConfigLoaded() {
    if (state.config) return state.config;
    state.config = await fetchConfig();
    state.audioHint.textContent = `Target folder: ${String(state.config.audio_dir || "")}`;
    for (const field of TEXT_FIELDS) {
      pathHints[field.key].textContent = `Writes to: ${String(state.config.text_targets?.[field.key] || "")}`;
    }
    return state.config;
  }

  async function saveCurrentTexts() {
    saveButton.disabled = true;
    topSaveButton.disabled = true;
    setStatus("Saving text files...");
    try {
      await ensureConfigLoaded();
      const payload = {};
      for (const field of TEXT_FIELDS) {
        payload[field.key] = formatFieldForSave(field, textareas[field.key].value);
        textareas[field.key].value = payload[field.key];
      }
      const data = await saveText(payload);
      syncNodeProperties();
      setStatus(`Saved ${Object.keys(data.saved_paths || {}).length} text files.`);
    } catch (error) {
      setStatus(String(error?.message || error), true);
    } finally {
      saveButton.disabled = false;
      topSaveButton.disabled = false;
    }
  }

  async function handleAudioSelection() {
    const file = hiddenAudioInput.files?.[0];
    hiddenAudioInput.value = "";
    if (!file) return;

    chooseAudioButton.disabled = true;
    setStatus(`Uploading audio: ${file.name}`);
    try {
      await ensureConfigLoaded();
      const data = await uploadAudio(file);
      if (state.node) {
        state.node.properties = state.node.properties || {};
        state.node.properties.vrgdg_test_popup_audio_filename = String(data.filename || file.name);
      }
      audioFileName.textContent = `Current uploaded audio: ${String(data.filename || file.name)}`;
      setStatus(`Audio uploaded to ${String(data.path || "")}`);
    } catch (error) {
      setStatus(String(error?.message || error), true);
    } finally {
      chooseAudioButton.disabled = false;
    }
  }

  closeButton.addEventListener("click", closeModal);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeModal();
  });
  saveButton.addEventListener("click", saveCurrentTexts);
  topSaveButton.addEventListener("click", saveCurrentTexts);
  applySubgraphButton.addEventListener("click", applySubgraphSettings);
  chooseAudioButton.addEventListener("click", () => hiddenAudioInput.click());
  hiddenAudioInput.addEventListener("change", handleAudioSelection);

  for (const field of TEXT_FIELDS) {
    textareas[field.key].addEventListener("input", syncNodeProperties);
  }
  for (const field of SUBGRAPH_FIELDS) {
    subgraphInputs[field.key].addEventListener("input", syncNodeProperties);
    subgraphInputs[field.key].addEventListener("change", syncNodeProperties);
  }

  overlay.__vrgdgOpenForNode = async (node) => {
    state.node = node;
    state.node.properties = state.node.properties || {};

    for (const field of TEXT_FIELDS) {
      textareas[field.key].value = formatFieldForDisplay(field, getStoredFieldValue(state.node, field));
    }

    const targetSubgraphNode = findTheGutsSubgraphNode();
    for (const field of SUBGRAPH_FIELDS) {
      const liveValue = getSubgraphWidgetValue(targetSubgraphNode, field);
      subgraphInputs[field.key].value = hasStoredSubgraphValue(state.node, field)
        ? getStoredSubgraphValue(state.node, field)
        : String(liveValue ?? field.defaultValue ?? "");
    }
    syncNodeProperties();

    const audioName = String(state.node.properties.vrgdg_test_popup_audio_filename || "");
    audioFileName.textContent = audioName ? `Current uploaded audio: ${audioName}` : "No audio file selected.";

    setStatus("");
    overlay.style.display = "flex";

    try {
      await ensureConfigLoaded();
    } catch (error) {
      setStatus(String(error?.message || error), true);
    }

    setTimeout(() => textareas.full_lyrics.focus(), 0);
  };

  return overlay;
}

function attachButton(node) {
  if ((node.widgets || []).some((widget) => widget.type === "button" && widget.name === "Open Prompt Creator UI V2")) {
    return;
  }

  node.addWidget("button", "Open Prompt Creator UI V2", null, () => {
    const modal = ensureModal();
    modal.__vrgdgOpenForNode(node);
  });
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      this.properties = this.properties || {};
      attachButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.properties = this.properties || {};
      attachButton(this);
      return result;
    };
  },
});
