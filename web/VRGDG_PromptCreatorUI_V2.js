import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_PromptCreatorUI_V2";
const MODAL_ID = "vrgdg-prompt-creator-ui-v2-modal";
const TEXT_FIELDS = [
  {
    key: "full_lyrics",
    label: "Full lyrics",
    placeholder: "Enter full lyrics",
    rows: 10,
    defaultValue: "",
  },
  {
    key: "style_theme",
    label: "Style/theme",
    placeholder: "Enter style/theme",
    rows: 6,
    defaultValue: `Surreal cinematic showroom aesthetic. Begin with sterile whites, cold grays, flat lighting, rigid symmetry, and polite stillness. Gradually shift toward harsh shadows, electric blues, defiant reds, cracked porcelain, broken symmetry, close-ups, Dutch angles, and low-angle heroic framing. Mood: controlled, eerie, then powerful and liberating.`,
  },
  {
    key: "story_idea",
    label: "Story idea",
    placeholder: "Enter short story idea",
    rows: 6,
    defaultValue: `A singer is the only real person in a sterile showroom world filled with silent porcelain mannequins. Her honest voice exposes the fake perfection around her: porcelain cracks, staged rooms lose symmetry, and sterile cream lighting shifts into deep shadows, blues, and reds. Keep the video focused on her, the mannequins, and the illusion of polite control breaking apart. By the end, she stands powerful and free in the shattered showroom.`,
  },
  {
    key: "subjects_and_scenes",
    label: "Subject",
    placeholder: "Enter subject details",
    rows: 6,
    defaultValue: "",
  },
];

function getStoredFieldValue(node, field) {
  const propertyName = `vrgdg_test_popup_${field.key}`;
  if (Object.prototype.hasOwnProperty.call(node.properties || {}, propertyName)) {
    return String(node.properties[propertyName] || "");
  }
  return String(field.defaultValue || "");
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
    width: min(860px, calc(100vw - 32px));
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

  const titleBlock = document.createElement("div");

  const title = document.createElement("div");
  title.textContent = "VRGDG Prompt Creator V2";
  title.style.cssText = "font-size: 20px; font-weight: 700;";

  const subtitle = document.createElement("div");
  subtitle.textContent = "Save full lyrics, style/theme, story idea, and subject.";
  subtitle.style.cssText = "margin-top: 4px; font-size: 13px; color: #94a3b8;";

  titleBlock.append(title, subtitle);

  const closeButton = createButton(
    "Close",
    "border: 1px solid #4b5563; background: #2b3138; color: #f3f4f6;"
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

  const hiddenAudioInput = document.createElement("input");
  hiddenAudioInput.type = "file";
  hiddenAudioInput.accept = "audio/*,video/*";
  hiddenAudioInput.style.display = "none";

  audioActions.append(chooseAudioButton, hiddenAudioInput);
  audioSection.append(audioTitle, audioHint, audioFileName, audioActions);

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

    const hint = createPathHint();

    section.append(label, textarea, hint);
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
  panel.append(titleRow, audioSection, textGrid, status, actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  const state = {
    node: null,
    config: null,
    status,
    saveButton,
    chooseAudioButton,
    hiddenAudioInput,
    audioFileName,
    audioHint,
    textareas,
    pathHints,
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
    setStatus("Saving text files...");
    try {
      await ensureConfigLoaded();
      const payload = {};
      for (const field of TEXT_FIELDS) {
        payload[field.key] = String(textareas[field.key].value || "");
      }
      const data = await saveText(payload);
      syncNodeProperties();
      setStatus(`Saved ${Object.keys(data.saved_paths || {}).length} text files.`);
    } catch (error) {
      setStatus(String(error?.message || error), true);
    } finally {
      saveButton.disabled = false;
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
  chooseAudioButton.addEventListener("click", () => hiddenAudioInput.click());
  hiddenAudioInput.addEventListener("change", handleAudioSelection);

  for (const field of TEXT_FIELDS) {
    textareas[field.key].addEventListener("input", syncNodeProperties);
  }

  overlay.__vrgdgOpenForNode = async (node) => {
    state.node = node;
    state.node.properties = state.node.properties || {};

    for (const field of TEXT_FIELDS) {
      textareas[field.key].value = getStoredFieldValue(state.node, field);
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
