import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_PromptCreatorUI_V2";
const PART2_NODE_NAME = "VRGDG_Part2WorkflowUI";
const MODAL_ID = "vrgdg-prompt-creator-ui-v2-modal";
const PART2_MODAL_ID = "vrgdg-prompt-creator-part2-ui-modal";
const PART2_DRAFT_STORAGE_KEY = "vrgdg.part2.workflow.ui.draft.v1";
const PART2_OPTIONAL_LORA_NODE_NAME = "VRGDG_OptionalMultiLoraModelOnly";
const PART2_MAX_LORA_SLOTS = 20;
const PART2_Z_IMAGE_LORA_INNER_NODE_ID = 847;
const BANNER_URL = new URL("./ChatGPT Image May 5, 2026, 08_07_18 PM.png?v=20260505_2020_refresh", import.meta.url).href;
const PART2_BANNER_URL = new URL("./ChatGPT Image May 5, 2026, 08_07_18 PM-002.png?v=20260505_224432", import.meta.url).href;
const LYRIC_CREATOR_GPT_URL = "https://chatgpt.com/g/g-69979b391cc88191ae4fe298b59c236e-ai-lyric-creator";
const STYLE_THEME_GPT_URL = "https://chatgpt.com/g/g-69fb415a964c8191b4a737f84f37227f-ltx-2-3-style-theme-guide/c/69fb427d-4518-8331-bfd7-505c0f55d2cc";
const STORY_IDEA_GPT_URL = "https://chatgpt.com/g/g-69fb3cb767448191a6caa88be94940d5-ltx-2-3-story-concept-helper/c/69fb3e25-7e74-8326-abd6-7df9cf847a5b";
const SUBJECT_LOCATION_GPT_URL = "https://chatgpt.com/g/g-69fb38a997fc8191a2fa479e44a3c675-ltx-2-3-subject-and-location-creator/c/69fb39e2-2ba0-8328-94c0-6ac9c94d0c89";
const PART2_NODE_IDS = {
  modelLoader: 271,
  settings: 736,
  useSrtSwitch: 837,
  llmI2V: 811,
  llmT2I: 805,
  camera: 830,
  promptJson: 543,
  zImageModels: 797,
  optionalLoras: 842,
};
const PART2_MODEL_FIELDS = [
  { nodeId: 271, key: "unet_name", label: "LTX GGUF model", downloadUrl: "https://huggingface.co/Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF/tree/main" },
  { nodeId: 271, key: "vae_name", label: "Video VAE", downloadUrl: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
  { nodeId: 271, key: "clip_name1", label: "Gemma Clip Model", downloadUrl: "https://huggingface.co/Sikaworld1990/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition-Ltx-2/resolve/main/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors" },
  { nodeId: 271, key: "clip_name2", label: "Text Projection clip model", downloadUrl: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/text_encoders" },
  { nodeId: 271, key: "model_name", label: "Latent Upscaler", downloadUrl: "https://huggingface.co/prince-canuma/LTX-2.3-distilled/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" },
  { nodeId: 271, key: "vae_name_1", label: "Audio VAE", downloadUrl: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
  { nodeId: 797, key: "unet_name", label: "Z-Image Turbo Model", downloadUrl: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" },
  { nodeId: 797, key: "clip_name", label: "Z-Image Clip", downloadUrl: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" },
  { nodeId: 797, key: "vae_name", label: "Z-Image VAE", downloadUrl: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" },
];
const PART2_LLM_DOWNLOADS = [
  {
    label: "Download GGUF",
    url: "https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2/resolve/main/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
  },
  {
    label: "Download mmproj",
    url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf",
  },
];
const PART2_SETTING_FIELDS = [
  { key: "value", label: "Frames Per Second", type: "number", step: "1", note: "Must match the FPS used in the Part 1 workflow." },
  { key: "value_1", label: "Width", type: "number", step: "8" },
  { key: "value_2", label: "Height", type: "number", step: "8" },
  { key: "value_3", label: "Seed", type: "number", step: "1" },
  {
    key: "value_4",
    label: "Scene Duration When Fixed",
    type: "number",
    step: "1",
    fixedDurationOnly: true,
    note: "Only shown when Use SRT Duration is OFF. Going over 20 seconds can cause OOM issues during video creation.",
  },
];
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
  "switch",
  "scene_duration_seconds",
  "model_file",
];
const SUBGRAPH_FIELDS = [
  {
    key: "model_file",
    label: "LLM Model",
    type: "combo",
    defaultValue: "",
    headerControl: true,
    note: "Model used by the Part 1 prompt creator LLM.",
  },
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
  {
    key: "switch",
    label: "Use SRT Durations",
    type: "boolean",
    defaultValue: "true",
    note: "ON uses the SRT/beat timing for scene lengths. OFF uses one fixed scene duration instead.",
  },
  {
    key: "scene_duration_seconds",
    label: "Fixed Scene Duration Seconds",
    type: "number",
    step: "0.1",
    defaultValue: "4.0",
    fixedDurationOnly: true,
    note: "Only used when Use SRT Durations is OFF. Choose the fixed duration for each scene in seconds. Going over 20 seconds can cause OOM issues during video creation.",
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

function createLlmDownloadButtons() {
  const buttons = document.createElement("div");
  buttons.style.cssText = "display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end;";

  for (const item of PART2_LLM_DOWNLOADS) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.label;
    button.style.cssText = `
      border: 1px solid #2563eb;
      background: #1d4ed8;
      color: white;
      border-radius: 6px;
      padding: 5px 8px;
      cursor: pointer;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    `;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      window.open(item.url, "_blank", "noopener,noreferrer");
    });
    buttons.appendChild(button);
  }

  return buttons;
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
  const input = field.type === "select" || field.type === "boolean" || field.type === "combo" ? document.createElement("select") : document.createElement("input");
  if (field.type === "select" || field.type === "boolean" || field.type === "combo") {
    const options = field.type === "boolean" ? ["true", "false"] : field.options || [];
    for (const optionValue of options) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = field.type === "boolean"
        ? (optionValue === "true" ? "ON" : "OFF")
        : optionValue;
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
  if (field.type === "boolean") {
    return text.toLowerCase() !== "false";
  }
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

function getSubgraphWidgetOptions(targetNode, field) {
  const widget = (targetNode?.widgets || []).find((item) => item?.name === field.key);
  const options = widget?.options || {};
  const values = options.values || options.items || options.value || [];
  return Array.isArray(values) ? values.map((value) => String(value)) : [];
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

async function loadPart2ConceptPrompts() {
  const response = await api.fetchApi("/vrgdg/part2/load_concept_prompts", { cache: "no-store" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) {
    throw new Error(String(data?.error || `Concept prompts load failed (${response.status})`));
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

  const subgraphTitleActions = document.createElement("div");
  subgraphTitleActions.style.cssText = `
    display: flex;
    align-items: flex-end;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
  `;

  const applySubgraphButton = createButton(
    "Apply Settings",
    "border: 1px solid #b45309; background: #d97706; color: white;"
  );

  const subgraphGrid = document.createElement("div");
  subgraphGrid.style.cssText = `
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
  `;

  const subgraphInputs = {};
  const subgraphFieldWraps = {};
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

    if (field.key === "model_file") {
      const labelRow = document.createElement("div");
      labelRow.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 5px;";
      fieldLabel.style.marginBottom = "0";
      labelRow.append(fieldLabel, createLlmDownloadButtons());
      fieldWrap.append(labelRow, input, fieldNote);
    } else {
      fieldWrap.append(fieldLabel, input, fieldNote);
    }
    if (field.headerControl) {
      fieldWrap.style.minWidth = "320px";
      subgraphTitleActions.appendChild(fieldWrap);
    } else {
      subgraphGrid.appendChild(fieldWrap);
    }
    subgraphInputs[field.key] = input;
    subgraphFieldWraps[field.key] = fieldWrap;
  }

  subgraphTitleActions.appendChild(applySubgraphButton);
  subgraphTitleRow.append(subgraphTitleBlock, subgraphTitleActions);
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

    const labelRow = document.createElement("div");
    labelRow.style.cssText = `
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    `;

    const label = document.createElement("label");
    label.textContent = field.label;
    label.style.cssText = "display: block; font-size: 14px; font-weight: 700;";

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

    labelRow.appendChild(label);
    if (field.key === "full_lyrics") {
      const lyricsHelp = createButton(
        "If you have not yet created a song and need help creating lyrics then click here",
        "border: 1px solid #1d4ed8; background: #2563eb; color: white; padding: 8px 12px; font-size: 12px; font-weight: 700; line-height: 1.3; text-align: center; max-width: 360px;"
      );
      lyricsHelp.type = "button";
      lyricsHelp.addEventListener("click", () => {
        window.open(LYRIC_CREATOR_GPT_URL, "_blank", "noopener,noreferrer");
      });
      labelRow.appendChild(lyricsHelp);
    }

    const gptUrlByField = {
      style_theme: STYLE_THEME_GPT_URL,
      story_idea: STORY_IDEA_GPT_URL,
      subjects_and_scenes: SUBJECT_LOCATION_GPT_URL,
    };
    const gptUrl = gptUrlByField[field.key];
    if (gptUrl) {
      const useGptButton = createButton(
        "Use GPT",
        "border: 1px solid #7c3aed; background: #8b5cf6; color: white; padding: 8px 12px; font-size: 12px; font-weight: 700;"
      );
      useGptButton.type = "button";
      useGptButton.addEventListener("click", () => {
        window.open(gptUrl, "_blank", "noopener,noreferrer");
      });
      labelRow.appendChild(useGptButton);
    }

    section.append(labelRow, textarea, helper, hint);
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
    subgraphFieldWraps,
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

  function updateSubgraphVisibility() {
    const useSrtDurations = String(subgraphInputs.switch?.value || "true").toLowerCase() !== "false";
    for (const field of SUBGRAPH_FIELDS) {
      const fieldWrap = subgraphFieldWraps[field.key];
      if (!fieldWrap || !field.fixedDurationOnly) continue;
      fieldWrap.style.display = useSrtDurations ? "none" : "block";
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
    subgraphInputs[field.key].addEventListener("input", () => {
      syncNodeProperties();
      updateSubgraphVisibility();
    });
    subgraphInputs[field.key].addEventListener("change", () => {
      syncNodeProperties();
      updateSubgraphVisibility();
    });
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
      const selectedValue = hasStoredSubgraphValue(state.node, field)
        ? getStoredSubgraphValue(state.node, field)
        : String(liveValue ?? field.defaultValue ?? "");
      if (field.type === "combo") {
        fillSelectOptions(subgraphInputs[field.key], getSubgraphWidgetOptions(targetSubgraphNode, field), selectedValue);
      }
      subgraphInputs[field.key].value = selectedValue;
    }
    syncNodeProperties();
    updateSubgraphVisibility();

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

function getPart2Node(nodeId) {
  return app.graph?.getNodeById?.(nodeId) || null;
}

function getPart2NodeByType(typeName) {
  const nodes = app.graph?._nodes || [];
  return nodes.find((node) => node?.comfyClass === typeName || node?.type === typeName) || null;
}

function getPart2OptionalLoraNode() {
  const nodeById = getPart2Node(PART2_NODE_IDS.optionalLoras);
  if (nodeById?.comfyClass === PART2_OPTIONAL_LORA_NODE_NAME || nodeById?.type === PART2_OPTIONAL_LORA_NODE_NAME) {
    return nodeById;
  }
  return getPart2NodeByType(PART2_OPTIONAL_LORA_NODE_NAME);
}

function getSubgraphNodes(subgraph) {
  if (Array.isArray(subgraph?.nodes)) return subgraph.nodes;
  if (Array.isArray(subgraph?._nodes)) return subgraph._nodes;
  return [];
}

function getPart2ZImageOptionalLoraNode() {
  const zNode = getPart2Node(PART2_NODE_IDS.zImageModels);
  const subNodes = getSubgraphNodes(zNode?.subgraph);
  return (
    subNodes.find((node) => Number(node?.id) === PART2_Z_IMAGE_LORA_INNER_NODE_ID) ||
    subNodes.find((node) => node?.comfyClass === PART2_OPTIONAL_LORA_NODE_NAME || node?.type === PART2_OPTIONAL_LORA_NODE_NAME) ||
    null
  );
}

function getPart2ZImageUseLoraProxyValue() {
  const value = getPart2WidgetValue(PART2_NODE_IDS.zImageModels, "use_custom_loras", 6);
  return String(value ?? false).toLowerCase() === "true";
}

function getPart2Widget(node, name, fallbackIndex = -1) {
  if (!node) return null;
  const widgets = node.widgets || [];
  return widgets.find((widget) => widget?.name === name) || widgets[fallbackIndex] || null;
}

function getPart2WidgetOptions(widget) {
  const options = widget?.options || {};
  const values = options.values || options.items || options.value || [];
  return Array.isArray(values) ? values.map((value) => String(value)) : [];
}

function getPart2WidgetValue(nodeId, name, fallbackIndex = -1) {
  const node = getPart2Node(nodeId);
  const widget = getPart2Widget(node, name, fallbackIndex);
  if (widget) return widget.value;
  if (Array.isArray(node?.widgets_values) && fallbackIndex >= 0) return node.widgets_values[fallbackIndex];
  return "";
}

function setPart2WidgetValue(nodeId, name, value, fallbackIndex = -1) {
  const node = getPart2Node(nodeId);
  if (!node) return false;

  const widget = getPart2Widget(node, name, fallbackIndex);
  if (widget) {
    widget.value = value;
    widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
  }

  if (Array.isArray(node.widgets_values)) {
    const widgetIndex = (node.widgets || []).findIndex((item) => item?.name === name);
    const resolvedIndex = widgetIndex >= 0 ? widgetIndex : fallbackIndex;
    if (resolvedIndex >= 0) node.widgets_values[resolvedIndex] = value;
  }

  app.graph?.setDirtyCanvas?.(true, true);
  return true;
}

function fillSelectOptions(select, options, currentValue) {
  const optionSet = new Set(options.map((value) => String(value)));
  if (currentValue !== undefined && currentValue !== null && String(currentValue) && !optionSet.has(String(currentValue))) {
    optionSet.add(String(currentValue));
  }
  select.innerHTML = "";
  for (const value of optionSet) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  select.value = String(currentValue ?? "");
}

function setSelectValueAllowingDraft(select, value) {
  const text = String(value ?? "");
  const hasOption = Array.from(select.options || []).some((option) => option.value === text);
  if (text && !hasOption) {
    const option = document.createElement("option");
    option.value = text;
    option.textContent = text;
    select.appendChild(option);
  }
  select.value = text;
}

function createPart2Field(labelText, control, noteText = "") {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display: block; font-size: 12px; color: #cbd5e1;";

  const label = document.createElement("div");
  label.textContent = labelText;
  label.style.cssText = "margin-bottom: 5px; font-weight: 700;";

  const note = document.createElement("div");
  note.textContent = noteText || "";
  note.style.cssText = `
    margin-top: 5px;
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.35;
  `;

  wrapper.append(label, control, note);
  return wrapper;
}

function createPart2ModelField(field, control) {
  if (!field.downloadUrl) {
    return createPart2Field(field.label, control);
  }

  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display: block; font-size: 12px; color: #cbd5e1;";

  const labelRow = document.createElement("div");
  labelRow.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 5px;";

  const label = document.createElement("div");
  label.textContent = field.label;
  label.style.cssText = "font-weight: 700;";

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Download Model";
  button.style.cssText = `
    border: 1px solid #2563eb;
    background: #1d4ed8;
    color: white;
    border-radius: 6px;
    padding: 5px 8px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
  `;
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    window.open(field.downloadUrl, "_blank", "noopener,noreferrer");
  });

  labelRow.append(label, button);
  wrapper.append(labelRow, control);
  return wrapper;
}

function createPart2LlmField(control) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display: block; font-size: 12px; color: #cbd5e1;";

  const labelRow = document.createElement("div");
  labelRow.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 5px;";

  const label = document.createElement("div");
  label.textContent = "SuperGemma LLM Model";
  label.style.cssText = "font-weight: 700;";

  const note = document.createElement("div");
  note.textContent = "Download both files for the Gemma LLM Node.";
  note.style.cssText = `
    margin-top: 5px;
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.35;
  `;

  labelRow.append(label, createLlmDownloadButtons());
  wrapper.append(labelRow, control, note);
  return wrapper;
}

function stylePart2Input(input) {
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

function createPart2Section(titleText, hintText = "") {
  const section = document.createElement("div");
  section.style.cssText = `
    border: 1px solid #364152;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 16px;
    background: #14191f;
  `;

  const title = document.createElement("div");
  title.textContent = titleText;
  title.style.cssText = "font-size: 15px; font-weight: 700;";

  const hint = document.createElement("div");
  hint.textContent = hintText;
  hint.style.cssText = "margin-top: 4px; margin-bottom: 12px; font-size: 12px; color: #94a3b8;";

  section.append(title, hint);
  return section;
}

function ensurePart2Modal() {
  let overlay = document.getElementById(PART2_MODAL_ID);
  if (overlay) return overlay;

  overlay = document.createElement("div");
  overlay.id = PART2_MODAL_ID;
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
    width: min(1500px, calc(100vw - 32px));
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
  banner.src = PART2_BANNER_URL;
  banner.alt = "VRGDG Part 2 Workflow banner";
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
  const title = document.createElement("div");
  title.textContent = "Part 2 Workflow Controls";
  title.style.cssText = "font-size: 20px; font-weight: 700;";
  const subtitle = document.createElement("div");
  subtitle.textContent = "Control model pickers, render settings, SRT/fixed timing, camera motions, and copied prompt JSON.";
  subtitle.style.cssText = "margin-top: 4px; font-size: 13px; color: #94a3b8;";
  titleBlock.append(title, subtitle);

  const closeButton = createButton(
    "Close UI Window",
    "border: 1px solid #dc2626; background: #ef4444; color: white; padding: 13px 20px; font-size: 14px; font-weight: 700;"
  );

  const topApplyButton = createButton(
    "Apply Part 2 Settings",
    "border: 1px solid #b45309; background: #d97706; color: white; font-weight: 700;"
  );

  const titleActions = document.createElement("div");
  titleActions.style.cssText = "display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end;";
  titleActions.append(topApplyButton, closeButton);

  titleRow.append(titleBlock, titleActions);

  const controls = {
    modelSelects: {},
    settings: {},
    useSrt: null,
    lora: {
      useCustom: null,
      count: null,
      twoPass: null,
      slots: [],
      section: null,
    },
    zImageLora: {
      useCustom: null,
      count: null,
      slots: [],
      section: null,
    },
    cameraItems: null,
    cameraMode: null,
    promptJson: null,
    wrappers: {},
  };

  const modelSection = createPart2Section("Models", "Model dropdowns for LTX 2.3, Z-image and the LLM node.");
  const modelGrid = document.createElement("div");
  modelGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px;";

  for (const field of PART2_MODEL_FIELDS) {
    const select = stylePart2Input(document.createElement("select"));
    controls.modelSelects[`${field.nodeId}:${field.key}`] = select;
    modelGrid.appendChild(createPart2ModelField(field, select));
  }

  const llmSelect = stylePart2Input(document.createElement("select"));
  controls.modelSelects["llm"] = llmSelect;
  modelGrid.appendChild(createPart2LlmField(llmSelect));
  modelSection.appendChild(modelGrid);

  const loraSection = createPart2Section(
    "LTX Optional LoRAs",
    "Pick optional model-only LoRAs once. Image/style LoRAs can sometimes slow down or stiffen motion during the first video pass, so the workflow can use half strength while motion is being created, then full strength during the upscale/refine pass."
  );
  controls.lora.section = loraSection;
  const loraGrid = document.createElement("div");
  loraGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px;";

  const loraUseSelect = stylePart2Input(document.createElement("select"));
  for (const [value, label] of [["false", "OFF"], ["true", "ON"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    loraUseSelect.appendChild(option);
  }
  controls.lora.useCustom = loraUseSelect;
  loraGrid.appendChild(createPart2Field("Use Custom LoRAs", loraUseSelect, "OFF leaves both first and second pass models unchanged."));

  const loraCountInput = stylePart2Input(document.createElement("input"));
  loraCountInput.type = "number";
  loraCountInput.min = "0";
  loraCountInput.max = String(PART2_MAX_LORA_SLOTS);
  loraCountInput.step = "1";
  controls.lora.count = loraCountInput;
  loraGrid.appendChild(createPart2Field("LoRA Count", loraCountInput, "How many LoRA slots to show and apply."));

  const loraTwoPassSelect = stylePart2Input(document.createElement("select"));
  for (const [value, label] of [["true", "ON"], ["false", "OFF"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    loraTwoPassSelect.appendChild(option);
  }
  controls.lora.twoPass = loraTwoPassSelect;
  loraGrid.appendChild(createPart2Field("LTX Two Pass Strength", loraTwoPassSelect, "ON uses half of each selected LoRA strength on first pass to preserve motion, then the full selected strength on the upscale pass. OFF uses the selected strength on both passes."));

  for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
    const select = stylePart2Input(document.createElement("select"));
    const strength = stylePart2Input(document.createElement("input"));
    strength.type = "number";
    strength.step = "0.01";
    strength.min = "-100";
    strength.max = "100";

    const loraWrapper = createPart2Field(`LoRA ${i}`, select);
    const strengthWrapper = createPart2Field(`Strength ${i}`, strength, "Selected/full strength for the upscale pass. First pass uses half of this value when two-pass strength is ON.");
    controls.lora.slots.push({ select, strength, loraWrapper, strengthWrapper });
    loraGrid.append(loraWrapper, strengthWrapper);
  }

  loraSection.appendChild(loraGrid);

  const zImageLoraSection = createPart2Section(
    "Z-Image Optional LoRA",
    "Optional LoRA for the Z-Image still-image branch. This does not use two-pass strength; selected LoRAs apply at the strength you enter."
  );
  controls.zImageLora.section = zImageLoraSection;
  const zImageLoraGrid = document.createElement("div");
  zImageLoraGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px;";

  const zImageLoraUseSelect = stylePart2Input(document.createElement("select"));
  for (const [value, label] of [["false", "OFF"], ["true", "ON"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    zImageLoraUseSelect.appendChild(option);
  }
  controls.zImageLora.useCustom = zImageLoraUseSelect;
  zImageLoraGrid.appendChild(createPart2Field("Use Z-Image LoRAs", zImageLoraUseSelect, "OFF leaves the Z-Image model unchanged."));

  const zImageLoraCountInput = stylePart2Input(document.createElement("input"));
  zImageLoraCountInput.type = "number";
  zImageLoraCountInput.min = "0";
  zImageLoraCountInput.max = String(PART2_MAX_LORA_SLOTS);
  zImageLoraCountInput.step = "1";
  controls.zImageLora.count = zImageLoraCountInput;
  zImageLoraGrid.appendChild(createPart2Field("Z-Image LoRA Count", zImageLoraCountInput, "How many Z-Image LoRA slots to show and apply."));

  for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
    const select = stylePart2Input(document.createElement("select"));
    const strength = stylePart2Input(document.createElement("input"));
    strength.type = "number";
    strength.step = "0.01";
    strength.min = "-100";
    strength.max = "100";

    const loraWrapper = createPart2Field(`Z-Image LoRA ${i}`, select);
    const strengthWrapper = createPart2Field(`Z-Image Strength ${i}`, strength, "Applied at this exact strength.");
    controls.zImageLora.slots.push({ select, strength, loraWrapper, strengthWrapper });
    zImageLoraGrid.append(loraWrapper, strengthWrapper);
  }

  zImageLoraSection.appendChild(zImageLoraGrid);

  const settingsSection = createPart2Section("Main Settings", "FPS should match the FPS used in Part 1.");
  const settingsGrid = document.createElement("div");
  settingsGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px;";

  const useSrtSelect = stylePart2Input(document.createElement("select"));
  for (const [value, label] of [["true", "ON"], ["false", "OFF"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    useSrtSelect.appendChild(option);
  }
  controls.useSrt = useSrtSelect;
  settingsGrid.appendChild(createPart2Field("Use SRT Duration", useSrtSelect, "Match this to the Part 1 workflow."));

  const orderedPart2Settings = [
    ...PART2_SETTING_FIELDS.filter((field) => field.fixedDurationOnly),
    ...PART2_SETTING_FIELDS.filter((field) => !field.fixedDurationOnly),
  ];
  for (const field of orderedPart2Settings) {
    const input = stylePart2Input(document.createElement("input"));
    input.type = field.type || "text";
    if (field.step) input.step = field.step;
    controls.settings[field.key] = input;
    const wrapper = createPart2Field(field.label, input, field.note);
    controls.wrappers[field.key] = wrapper;
    settingsGrid.appendChild(wrapper);
  }
  settingsSection.appendChild(settingsGrid);

  const cameraSection = createPart2Section("Camera Motions", "Edit the camera motion list if you want a custom list or leave as is. Choose how the LLM receives motions for each scene.");
  const cameraMode = stylePart2Input(document.createElement("select"));
  for (const [value, label] of [["index", "Index-based"], ["random", "Random"], ["random no repeat", "Random No Repeat"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    cameraMode.appendChild(option);
  }
  controls.cameraMode = cameraMode;
  cameraSection.appendChild(createPart2Field("Selection Mode", cameraMode, "Index-based walks through the list in order and wraps. Random picks a seeded random motion. Random No Repeat walks a seeded shuffled order before repeating."));

  const cameraItems = document.createElement("textarea");
  cameraItems.rows = 8;
  stylePart2Input(cameraItems);
  cameraItems.style.resize = "vertical";
  controls.cameraItems = cameraItems;
  cameraSection.appendChild(createPart2Field("Camera Motion List", cameraItems, "One motion per line."));

  const promptSection = createPart2Section("Prompt JSON From Part 1", "Paste the JSON text created by the previous workflow in here.");
  const promptJson = document.createElement("textarea");
  promptJson.rows = 12;
  stylePart2Input(promptJson);
  promptJson.style.resize = "vertical";
  controls.promptJson = promptJson;
  const promptField = createPart2Field("Prompt JSON", promptJson, "This updates the Prompt Splitter node.");
  const promptHeader = promptField.querySelector("div");
  const pasteFromStep1Button = createButton(
    "Paste From Step 1",
    "border: 1px solid #2563eb; background: #1d4ed8; color: white; padding: 6px 9px; font-size: 11px; font-weight: 700;"
  );
  if (promptHeader) {
    promptHeader.style.display = "flex";
    promptHeader.style.alignItems = "center";
    promptHeader.style.justifyContent = "space-between";
    promptHeader.style.gap = "8px";
    promptHeader.appendChild(pasteFromStep1Button);
  } else {
    promptField.insertBefore(pasteFromStep1Button, promptJson);
  }
  promptSection.appendChild(promptField);

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
  actions.style.cssText = "display: flex; gap: 10px; justify-content: flex-end; margin-top: 8px; flex-wrap: wrap;";
  const applyButton = createButton(
    "Apply Part 2 Settings",
    "border: 1px solid #b45309; background: #d97706; color: white; font-weight: 700;"
  );
  actions.append(applyButton);

  panel.append(banner, titleRow, modelSection, loraSection, zImageLoraSection, settingsSection, cameraSection, promptSection, status, actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  function setStatus(message, isError = false) {
    status.textContent = message || "";
    status.style.color = isError ? "#fca5a5" : "#cbd5e1";
  }

  let suppressDraftSave = false;

  function collectPart2Draft() {
    const draft = {
      modelSelects: {},
      settings: {},
      useSrt: controls.useSrt.value,
      cameraItems: controls.cameraItems.value,
      cameraMode: controls.cameraMode.value,
      promptJson: controls.promptJson.value,
      lora: {
        useCustom: controls.lora.useCustom.value,
        count: controls.lora.count.value,
        twoPass: controls.lora.twoPass.value,
        slots: controls.lora.slots.map((slot) => ({
          lora: slot.select.value,
          strength: slot.strength.value,
        })),
      },
      zImageLora: {
        useCustom: controls.zImageLora.useCustom.value,
        count: controls.zImageLora.count.value,
        slots: controls.zImageLora.slots.map((slot) => ({
          lora: slot.select.value,
          strength: slot.strength.value,
        })),
      },
    };

    for (const [key, select] of Object.entries(controls.modelSelects)) {
      draft.modelSelects[key] = select.value;
    }
    for (const [key, input] of Object.entries(controls.settings)) {
      draft.settings[key] = input.value;
    }
    return draft;
  }

  function savePart2Draft() {
    if (suppressDraftSave) return;
    try {
      localStorage.setItem(PART2_DRAFT_STORAGE_KEY, JSON.stringify(collectPart2Draft()));
    } catch (error) {
      // Browser storage can be disabled; form still works normally.
    }
  }

  function loadPart2Draft() {
    try {
      const raw = localStorage.getItem(PART2_DRAFT_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (error) {
      return null;
    }
  }

  function clearPart2Draft() {
    try {
      localStorage.removeItem(PART2_DRAFT_STORAGE_KEY);
    } catch (error) {
      // ignore
    }
  }

  function applyPart2Draft(draft) {
    if (!draft || typeof draft !== "object") return false;
    suppressDraftSave = true;

    for (const [key, value] of Object.entries(draft.modelSelects || {})) {
      if (controls.modelSelects[key]) setSelectValueAllowingDraft(controls.modelSelects[key], value);
    }
    for (const [key, value] of Object.entries(draft.settings || {})) {
      if (controls.settings[key]) controls.settings[key].value = String(value ?? "");
    }

    if (draft.useSrt !== undefined) controls.useSrt.value = String(draft.useSrt);
    if (draft.cameraItems !== undefined) controls.cameraItems.value = String(draft.cameraItems);
    if (draft.cameraMode !== undefined) controls.cameraMode.value = String(draft.cameraMode);
    if (draft.promptJson !== undefined) controls.promptJson.value = String(draft.promptJson);

    if (draft.lora) {
      controls.lora.useCustom.value = String(draft.lora.useCustom ?? "false");
      controls.lora.count.value = String(draft.lora.count ?? "0");
      controls.lora.twoPass.value = String(draft.lora.twoPass ?? "true");
      for (let i = 0; i < controls.lora.slots.length; i++) {
        const slotDraft = draft.lora.slots?.[i] || {};
        if (slotDraft.lora !== undefined) setSelectValueAllowingDraft(controls.lora.slots[i].select, slotDraft.lora);
        if (slotDraft.strength !== undefined) controls.lora.slots[i].strength.value = String(slotDraft.strength);
      }
      updateLoraVisibility();
    }

    if (draft.zImageLora) {
      controls.zImageLora.useCustom.value = String(draft.zImageLora.useCustom ?? "false");
      controls.zImageLora.count.value = String(draft.zImageLora.count ?? "0");
      for (let i = 0; i < controls.zImageLora.slots.length; i++) {
        const slotDraft = draft.zImageLora.slots?.[i] || {};
        if (slotDraft.lora !== undefined) setSelectValueAllowingDraft(controls.zImageLora.slots[i].select, slotDraft.lora);
        if (slotDraft.strength !== undefined) controls.zImageLora.slots[i].strength.value = String(slotDraft.strength);
      }
      updateZImageLoraVisibility();
    }

    updateFixedDurationVisibility();
    suppressDraftSave = false;
    return true;
  }

  function closeModal() {
    savePart2Draft();
    overlay.style.display = "none";
    setStatus("");
  }

  function updateFixedDurationVisibility() {
    const useSrt = String(controls.useSrt.value || "true").toLowerCase() !== "false";
    if (controls.wrappers.value_4) {
      controls.wrappers.value_4.style.display = useSrt ? "none" : "block";
    }
  }

  function updateLoraVisibility() {
    const useLoras = String(controls.lora.useCustom.value || "false").toLowerCase() === "true";
    const rawCount = Number(controls.lora.count.value || 0);
    const count = useLoras ? Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Number.isFinite(rawCount) ? rawCount : 0)) : 0;

    controls.lora.count.parentElement.style.display = useLoras ? "block" : "none";
    controls.lora.twoPass.parentElement.style.display = useLoras ? "block" : "none";

    for (let i = 0; i < controls.lora.slots.length; i++) {
      const visible = useLoras && i < count;
      controls.lora.slots[i].loraWrapper.style.display = visible ? "block" : "none";
      controls.lora.slots[i].strengthWrapper.style.display = visible ? "block" : "none";
    }
  }

  function updateZImageLoraVisibility() {
    const useLoras = String(controls.zImageLora.useCustom.value || "false").toLowerCase() === "true";
    const rawCount = Number(controls.zImageLora.count.value || 0);
    const count = useLoras ? Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Number.isFinite(rawCount) ? rawCount : 0)) : 0;

    controls.zImageLora.count.parentElement.style.display = useLoras ? "block" : "none";

    for (let i = 0; i < controls.zImageLora.slots.length; i++) {
      const visible = useLoras && i < count;
      controls.zImageLora.slots[i].loraWrapper.style.display = visible ? "block" : "none";
      controls.zImageLora.slots[i].strengthWrapper.style.display = visible ? "block" : "none";
    }
  }

  function getLoraWidgetFallbackIndex(name) {
    if (name === "use_custom_loras") return 0;
    if (name === "lora_count") return 1;
    if (name === "ltx_two_pass_mode") return 2;
    let match = /^lora_(\d+)$/.exec(String(name || ""));
    if (match) return 3 + (Number(match[1]) - 1) * 2;
    match = /^strength_(\d+)$/.exec(String(name || ""));
    if (match) return 4 + (Number(match[1]) - 1) * 2;
    return -1;
  }

  function getNodeWidgetValueFlexible(node, name, fallbackValue = "") {
    const fallbackIndex = getLoraWidgetFallbackIndex(name);
    const widget = getPart2Widget(node, name, fallbackIndex);
    if (widget) return widget.value;
    if (Array.isArray(node?.widgets_values) && fallbackIndex >= 0) return node.widgets_values[fallbackIndex];
    return fallbackValue;
  }

  function refreshLoraControls(missing) {
    const loraNode = getPart2OptionalLoraNode();
    if (!loraNode) {
      controls.lora.section.style.display = "none";
      missing.push(PART2_OPTIONAL_LORA_NODE_NAME);
      return;
    }

    controls.lora.section.style.display = "block";
    const useWidget = getPart2Widget(loraNode, "use_custom_loras");
    const countWidget = getPart2Widget(loraNode, "lora_count");
    const twoPassWidget = getPart2Widget(loraNode, "ltx_two_pass_mode");

    controls.lora.useCustom.value = String(useWidget?.value ?? false).toLowerCase() === "true" ? "true" : "false";
    controls.lora.count.value = String(countWidget?.value ?? 0);
    controls.lora.twoPass.value = String(twoPassWidget?.value ?? true).toLowerCase() === "false" ? "false" : "true";

    if (!useWidget) missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.use_custom_loras`);
    if (!countWidget) missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.lora_count`);
    if (!twoPassWidget) missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.ltx_two_pass_mode`);

    for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
      const slot = controls.lora.slots[i - 1];
      const loraWidget = getPart2Widget(loraNode, `lora_${i}`);
      const strengthWidget = getPart2Widget(loraNode, `strength_${i}`);
      fillSelectOptions(slot.select, getPart2WidgetOptions(loraWidget), loraWidget?.value ?? "[none]");
      slot.strength.value = String(strengthWidget?.value ?? 1);
    }

    updateLoraVisibility();
  }

  function setPart2NodeWidgetValue(node, name, value) {
    if (!node) return false;
    const fallbackIndex = getLoraWidgetFallbackIndex(name);
    const widget = getPart2Widget(node, name, fallbackIndex);
    if (widget) {
      widget.value = value;
      widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
    }
    if (Array.isArray(node.widgets_values)) {
      const widgetIndex = (node.widgets || []).findIndex((item) => item?.name === name);
      const resolvedIndex = widgetIndex >= 0 ? widgetIndex : fallbackIndex;
      if (resolvedIndex >= 0) node.widgets_values[resolvedIndex] = value;
    }
    app.graph?.setDirtyCanvas?.(true, true);
    return Boolean(widget) || fallbackIndex >= 0;
  }

  function applyLoraControls(missing) {
    const loraNode = getPart2OptionalLoraNode();
    if (!loraNode) {
      missing.push(PART2_OPTIONAL_LORA_NODE_NAME);
      return 0;
    }

    let updated = 0;
    const useLoras = String(controls.lora.useCustom.value).toLowerCase() === "true";
    const rawCount = Number(controls.lora.count.value || 0);
    const count = Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Number.isFinite(rawCount) ? Math.round(rawCount) : 0));

    if (setPart2NodeWidgetValue(loraNode, "use_custom_loras", useLoras)) updated += 1;
    else missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.use_custom_loras`);
    if (setPart2NodeWidgetValue(loraNode, "lora_count", count)) updated += 1;
    else missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.lora_count`);
    if (setPart2NodeWidgetValue(loraNode, "ltx_two_pass_mode", String(controls.lora.twoPass.value).toLowerCase() !== "false")) updated += 1;
    else missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.ltx_two_pass_mode`);

    for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
      const slot = controls.lora.slots[i - 1];
      const selectedLora = i <= count && useLoras ? slot.select.value : "[none]";
      const strengthValue = Number(slot.strength.value);
      if (setPart2NodeWidgetValue(loraNode, `lora_${i}`, selectedLora || "[none]")) updated += 1;
      else missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.lora_${i}`);
      if (setPart2NodeWidgetValue(loraNode, `strength_${i}`, Number.isFinite(strengthValue) ? strengthValue : 1.0)) updated += 1;
      else missing.push(`${PART2_OPTIONAL_LORA_NODE_NAME}.strength_${i}`);
    }

    updateLoraVisibility();
    return updated;
  }

  function refreshZImageLoraControls(missing) {
    const loraNode = getPart2ZImageOptionalLoraNode();
    if (!loraNode) {
      controls.zImageLora.section.style.display = "none";
      missing.push("Z-Image optional LoRA node");
      return;
    }

    controls.zImageLora.section.style.display = "block";
    controls.zImageLora.useCustom.value = getPart2ZImageUseLoraProxyValue() ? "true" : "false";
    controls.zImageLora.count.value = String(getNodeWidgetValueFlexible(loraNode, "lora_count", 0));

    for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
      const slot = controls.zImageLora.slots[i - 1];
      const loraWidget = getPart2Widget(loraNode, `lora_${i}`, getLoraWidgetFallbackIndex(`lora_${i}`));
      const currentLora = getNodeWidgetValueFlexible(loraNode, `lora_${i}`, "[none]");
      fillSelectOptions(slot.select, getPart2WidgetOptions(loraWidget), currentLora);
      slot.strength.value = String(getNodeWidgetValueFlexible(loraNode, `strength_${i}`, 1));
    }

    updateZImageLoraVisibility();
  }

  function applyZImageLoraControls(missing) {
    const loraNode = getPart2ZImageOptionalLoraNode();
    if (!loraNode) {
      missing.push("Z-Image optional LoRA node");
      return 0;
    }

    let updated = 0;
    const useLoras = String(controls.zImageLora.useCustom.value).toLowerCase() === "true";
    const rawCount = Number(controls.zImageLora.count.value || 0);
    const count = Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Number.isFinite(rawCount) ? Math.round(rawCount) : 0));

    if (setPart2NodeWidgetValue(loraNode, "use_custom_loras", useLoras)) updated += 1;
    else missing.push("Z-Image LoRA use_custom_loras");
    if (setPart2WidgetValue(PART2_NODE_IDS.zImageModels, "use_custom_loras", useLoras, 6)) updated += 1;
    else missing.push("Z-Image use custom LoRAs toggle");
    if (setPart2NodeWidgetValue(loraNode, "lora_count", count)) updated += 1;
    else missing.push("Z-Image LoRA lora_count");
    setPart2NodeWidgetValue(loraNode, "ltx_two_pass_mode", false);

    for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
      const slot = controls.zImageLora.slots[i - 1];
      const selectedLora = i <= count && useLoras ? slot.select.value : "[none]";
      const strengthValue = Number(slot.strength.value);
      if (setPart2NodeWidgetValue(loraNode, `lora_${i}`, selectedLora || "[none]")) updated += 1;
      else missing.push(`Z-Image LoRA lora_${i}`);
      if (setPart2NodeWidgetValue(loraNode, `strength_${i}`, Number.isFinite(strengthValue) ? strengthValue : 1.0)) updated += 1;
      else missing.push(`Z-Image LoRA strength_${i}`);
    }

    updateZImageLoraVisibility();
    return updated;
  }

  async function pastePromptJsonFromStep1() {
    pasteFromStep1Button.disabled = true;
    const previousText = pasteFromStep1Button.textContent;
    pasteFromStep1Button.textContent = "Loading...";
    try {
      const data = await loadPart2ConceptPrompts();
      controls.promptJson.value = String(data.text ?? "");
      controls.promptJson.dispatchEvent(new Event("input", { bubbles: true }));
      savePart2Draft();
      setStatus("Loaded prompt JSON from Step 1. Click Apply Part 2 Settings when ready.");
    } catch (error) {
      setStatus(error?.message || "Could not load prompt JSON from Step 1.", true);
    } finally {
      pasteFromStep1Button.disabled = false;
      pasteFromStep1Button.textContent = previousText;
    }
  }

  function refreshPart2Controls() {
    const missing = [];
    suppressDraftSave = true;

    for (const field of PART2_MODEL_FIELDS) {
      const node = getPart2Node(field.nodeId);
      const widget = getPart2Widget(node, field.key);
      const current = getPart2WidgetValue(field.nodeId, field.key);
      fillSelectOptions(controls.modelSelects[`${field.nodeId}:${field.key}`], getPart2WidgetOptions(widget), current);
      if (!node || !widget) missing.push(field.label);
    }

    const llmNode = getPart2Node(PART2_NODE_IDS.llmI2V);
    const llmWidget = getPart2Widget(llmNode, null, 0);
    fillSelectOptions(controls.modelSelects.llm, getPart2WidgetOptions(llmWidget), getPart2WidgetValue(PART2_NODE_IDS.llmI2V, null, 0));
    if (!llmNode || !llmWidget) missing.push("SuperGemma LLM model");

    for (const field of PART2_SETTING_FIELDS) {
      controls.settings[field.key].value = String(getPart2WidgetValue(PART2_NODE_IDS.settings, field.key) ?? "");
      if (!getPart2Widget(getPart2Node(PART2_NODE_IDS.settings), field.key)) missing.push(field.label);
    }

    controls.useSrt.value = String(getPart2WidgetValue(PART2_NODE_IDS.useSrtSwitch, "switch", 0)).toLowerCase() === "false" ? "false" : "true";
    controls.cameraItems.value = String(getPart2WidgetValue(PART2_NODE_IDS.camera, "items", 1) || "");
    controls.cameraMode.value = String(getPart2WidgetValue(PART2_NODE_IDS.camera, "selection_mode", 3) || "index");
    controls.promptJson.value = String(getPart2WidgetValue(PART2_NODE_IDS.promptJson, null, 0) || "");
    updateFixedDurationVisibility();
    refreshLoraControls(missing);
    refreshZImageLoraControls(missing);
    suppressDraftSave = false;

    const draft = loadPart2Draft();
    if (draft) {
      applyPart2Draft(draft);
      setStatus(
        missing.length
          ? `Restored unsaved UI draft.\nLoaded with missing widgets:\n${missing.join("\n")}`
          : "Restored unsaved UI draft. Click Apply Part 2 Settings when ready."
      );
      return;
    }

    setStatus(missing.length ? `Loaded with missing widgets:\n${missing.join("\n")}` : "");
  }

  function applyPart2Settings() {
    const missing = [];
    let updated = 0;

    for (const field of PART2_MODEL_FIELDS) {
      if (setPart2WidgetValue(field.nodeId, field.key, controls.modelSelects[`${field.nodeId}:${field.key}`].value)) updated += 1;
      else missing.push(field.label);
    }

    const llmValue = controls.modelSelects.llm.value;
    if (setPart2WidgetValue(PART2_NODE_IDS.llmI2V, null, llmValue, 0)) updated += 1;
    else missing.push("SuperGemma LLM model");
    if (setPart2WidgetValue(PART2_NODE_IDS.llmT2I, null, llmValue, 0)) updated += 1;
    else missing.push("SuperGemma LLM model for text-to-image");

    for (const field of PART2_SETTING_FIELDS) {
      const rawValue = controls.settings[field.key].value;
      const numberValue = Number(rawValue);
      const value = Number.isFinite(numberValue) ? numberValue : rawValue;
      if (setPart2WidgetValue(PART2_NODE_IDS.settings, field.key, value)) updated += 1;
      else missing.push(field.label);
    }

    const useSrt = String(controls.useSrt.value).toLowerCase() !== "false";
    if (setPart2WidgetValue(PART2_NODE_IDS.useSrtSwitch, "switch", useSrt, 0)) updated += 1;
    else missing.push("Use SRT Duration toggle");

    if (setPart2WidgetValue(PART2_NODE_IDS.camera, "items", controls.cameraItems.value, 1)) updated += 1;
    else missing.push("Camera motion list");
    if (setPart2WidgetValue(PART2_NODE_IDS.camera, "selection_mode", controls.cameraMode.value, 3)) updated += 1;
    else missing.push("Camera motion mode");

    if (setPart2WidgetValue(PART2_NODE_IDS.promptJson, null, controls.promptJson.value, 0)) updated += 1;
    else missing.push("Prompt JSON");

    updated += applyLoraControls(missing);
    updated += applyZImageLoraControls(missing);
    updateFixedDurationVisibility();
    clearPart2Draft();
    setStatus(missing.length ? `Updated ${updated} settings.\nMissing:\n${missing.join("\n")}` : `Updated ${updated} Part 2 settings.`);
  }

  closeButton.addEventListener("click", closeModal);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeModal();
  });
  controls.useSrt.addEventListener("change", updateFixedDurationVisibility);
  controls.lora.useCustom.addEventListener("change", updateLoraVisibility);
  controls.lora.count.addEventListener("input", updateLoraVisibility);
  controls.lora.count.addEventListener("change", updateLoraVisibility);
  controls.zImageLora.useCustom.addEventListener("change", updateZImageLoraVisibility);
  controls.zImageLora.count.addEventListener("input", updateZImageLoraVisibility);
  controls.zImageLora.count.addEventListener("change", updateZImageLoraVisibility);
  pasteFromStep1Button.addEventListener("click", pastePromptJsonFromStep1);

  const draftControls = [
    ...Object.values(controls.modelSelects),
    ...Object.values(controls.settings),
    controls.useSrt,
    controls.cameraItems,
    controls.cameraMode,
    controls.promptJson,
    controls.lora.useCustom,
    controls.lora.count,
    controls.lora.twoPass,
    controls.zImageLora.useCustom,
    controls.zImageLora.count,
  ];
  for (const slot of controls.lora.slots) draftControls.push(slot.select, slot.strength);
  for (const slot of controls.zImageLora.slots) draftControls.push(slot.select, slot.strength);
  for (const control of draftControls) {
    control?.addEventListener?.("input", savePart2Draft);
    control?.addEventListener?.("change", savePart2Draft);
  }

  applyButton.addEventListener("click", applyPart2Settings);
  topApplyButton.addEventListener("click", applyPart2Settings);

  overlay.__vrgdgOpenPart2 = () => {
    overlay.style.display = "flex";
    refreshPart2Controls();
  };

  return overlay;
}

function attachButton(node) {
  if (!(node.widgets || []).some((widget) => widget.type === "button" && widget.name === "Open Prompt Creator UI V2")) {
    node.addWidget("button", "Open Prompt Creator UI V2", null, () => {
      const modal = ensureModal();
      modal.__vrgdgOpenForNode(node);
    });
  }

}

function attachPart2Button(node) {
  if ((node.widgets || []).some((widget) => widget.type === "button" && widget.name === "Open Part 2 Workflow UI")) {
    return;
  }

  node.addWidget("button", "Open Part 2 Workflow UI", null, () => {
    const modal = ensurePart2Modal();
    modal.__vrgdgOpenPart2();
  });
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME && nodeData.name !== PART2_NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      this.properties = this.properties || {};
      if (nodeData.name === PART2_NODE_NAME) attachPart2Button(this);
      else attachButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.properties = this.properties || {};
      if (nodeData.name === PART2_NODE_NAME) attachPart2Button(this);
      else attachButton(this);
      return result;
    };
  },
});
