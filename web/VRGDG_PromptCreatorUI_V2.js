import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_PromptCreatorUI_V2";
const PART2_NODE_NAME = "VRGDG_Part2WorkflowUI";
const PART3_NODE_NAME = "VRGDG_Part3WorkflowUI";
const MODAL_ID = "vrgdg-prompt-creator-ui-v2-modal";
const PART2_MODAL_ID = "vrgdg-prompt-creator-part2-ui-modal";
const PART2_DRAFT_STORAGE_KEY = "vrgdg.part2.workflow.ui.draft.v1";
const PART2_OPTIONAL_LORA_NODE_NAME = "VRGDG_OptionalMultiLoraModelOnly";
const PART2_MAX_LORA_SLOTS = 20;
const PART2_Z_IMAGE_LORA_INNER_NODE_ID = 847;
const PART2_Z_IMAGE_TRIGGER_NODE_ID = 867;
const PART2_LTX_TRIGGER_NODE_ID = 885;
const PART3_LTX_TRIGGER_NODE_ID = 884;
const BANNER_URL = new URL("./ChatGPT Image May 5, 2026, 08_07_18 PM.png?v=20260505_2020_refresh", import.meta.url).href;
const PART2_BANNER_URL = new URL("./ChatGPT Image May 5, 2026, 08_07_18 PM-002.png?v=20260505_224432", import.meta.url).href;
const LYRIC_CREATOR_GPT_URL = "https://chatgpt.com/g/g-69979b391cc88191ae4fe298b59c236e-ai-lyric-creator";
const STYLE_THEME_GPT_URL = "https://chatgpt.com/g/g-69fb415a964c8191b4a737f84f37227f-ltx-2-3-style-theme-guide/c/69fb427d-4518-8331-bfd7-505c0f55d2cc";
const STORY_IDEA_GPT_URL = "https://chatgpt.com/g/g-69fb3cb767448191a6caa88be94940d5-ltx-2-3-story-concept-helper/c/69fb3e25-7e74-8326-abd6-7df9cf847a5b";
const SUBJECT_LOCATION_GPT_URL = "https://chatgpt.com/g/g-69fb38a997fc8191a2fa479e44a3c675-ltx-2-3-subject-and-location-creator/c/69fb39e2-2ba0-8328-94c0-6ac9c94d0c89";
const ADVANCED_PROMPT_DETAILS_GPT_URL = "https://chatgpt.com/g/g-69fbebf16b3c81919db550b8d2e87db7-ltx-2-3-advanced-prompt-details";
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
const PART3_NODE_IDS = {
  ...PART2_NODE_IDS,
  llmI2V: 853,
  llmT2I: null,
  promptJson: 860,
  zImageModels: null,
};
const PART2_ADVANCED_NODE_NAME = "VRGDG_MultiCyclingTextPicker";
const PART2_ADVANCED_EASY_NODE_NAME = "VRGDG_EasyMultiCyclingTextPicker";
const PART2_ADVANCED_EASY_NODE_ID = 902;
const PART3_ADVANCED_EASY_NODE_ID = 887;
const PART2_MODEL_FIELDS = [
  { nodeId: 271, key: "unet_name", label: "LTX GGUF model", downloadUrl: "https://huggingface.co/Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF/tree/main" },
  { nodeId: 271, key: "vae_name", label: "Video VAE", downloadUrl: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
  { nodeId: 271, key: "clip_name1", label: "Gemma Clip Model", downloadUrl: "https://huggingface.co/Sikaworld1990/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition-Ltx-2/resolve/main/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors" },
  { nodeId: 271, key: "clip_name2", label: "Text Projection clip model", downloadUrl: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/text_encoders" },
  { nodeId: 271, key: "model_name", label: "Latent Upscaler", downloadUrl: "https://huggingface.co/prince-canuma/LTX-2.3-distilled/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" },
  { nodeId: 271, key: "vae_name_1", label: "Audio VAE", downloadUrl: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
  { nodeId: 797, key: "unet_name", label: "Z-Image Turbo Model", downloadUrl: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors", zImageOnly: true },
  { nodeId: 797, key: "clip_name", label: "Z-Image Clip", downloadUrl: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", zImageOnly: true },
  { nodeId: 797, key: "vae_name", label: "Z-Image VAE", downloadUrl: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", zImageOnly: true },
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
const PART2_ADVANCED_MAX_PICKERS = 20;
const PART2_ADVANCED_PRESET_ITEMS = {
  "Camera Motion": `Slow push-in
Track right
Track left
Dolly backward
Handheld follow
Over-the-shoulder push-in
Slow pan right
Slow pan left
Tilt up
Tilt down
Arc around subject
Orbit shot
Low-angle tracking shot
Crane rising move
Slow zoom-in`,
  "Character Movement/Motion": `Walks toward camera with confident swagger
Strides across the frame
Leans toward the camera
Points into the lens
Throws arms wide
Raises both hands overhead
Runs a hand through their hair
Slowly backs away from the camera
Drops to one knee
Throws their head back
Whips a jacket off one shoulder
Stomps forward with attitude
Tilts chin upward
Reaches toward the camera
Collapses dramatically to the floor`,
  Lighting: `Soft natural light
Hard direct sunlight
Warm tungsten light
Cool fluorescent light
Neon nightclub light
Moody low-key lighting
High-key studio lighting
Backlit silhouette
Rim lighting
Side lighting
Top-down lighting
Underlighting
Golden hour light
Blue hour light
Strobe lighting`,
  "Time of Day": `Pre-dawn
Dawn
Early morning
Mid-morning
Late morning
Noon
Early afternoon
Mid-afternoon
Late afternoon
Golden hour
Sunset
Dusk
Blue hour
Night
After midnight`,
  Weather: `Clear sky
Partly cloudy
Overcast
Light rain
Heavy rain
Thunderstorm
Drizzle
Fog
Mist
Snowfall
Blizzard
Hail
Strong wind
Dust storm
Humid haze`,
  Dialogue: "",
  "Facial Expression": `Calm expression
Serious expression
Confident smirk
Cold stare
Worried expression
Sad expression
Angry glare
Fearful expression
Surprised expression
Blank expression
Dreamy expression
Suspicious look
Pained expression
Defiant expression
Soft smile`,
  Emotion: `Joyful
Melancholic
Anxious
Furious
Heartbroken
Hopeful
Jealous
Lonely
Nostalgic
Conflicted
Euphoric
Ashamed
Determined
Vengeful
Peaceful`,
  Custom: "",
};
const PART2_ADVANCED_PRESETS = Object.keys(PART2_ADVANCED_PRESET_ITEMS);
const PART2_ADVANCED_SELECTION_MODES = ["index", "random", "random no repeat"];
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

function ensureGemma4ProgressOverlay() {
  let overlay = document.getElementById("vrgdg-gemma4-progress-overlay");
  if (overlay) return overlay;

  overlay = document.createElement("div");
  overlay.id = "vrgdg-gemma4-progress-overlay";
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100000;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.62);
  `;

  const panel = document.createElement("div");
  panel.style.cssText = `
    width: min(420px, calc(100vw - 32px));
    border: 1px solid #334155;
    border-radius: 8px;
    background: #111827;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
    padding: 18px;
    color: #f8fafc;
  `;

  const title = document.createElement("div");
  title.textContent = "Gemma4";
  title.style.cssText = "font-size: 16px; font-weight: 800; margin-bottom: 8px;";

  const message = document.createElement("div");
  message.textContent = "Starting...";
  message.style.cssText = "font-size: 13px; color: #cbd5e1; line-height: 1.45; white-space: pre-wrap;";

  const bar = document.createElement("div");
  bar.style.cssText = `
    height: 8px;
    margin-top: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: #1f2937;
  `;
  const fill = document.createElement("div");
  fill.style.cssText = `
    width: 42%;
    height: 100%;
    border-radius: 999px;
    background: #10b981;
    animation: vrgdgGemma4Pulse 1.1s ease-in-out infinite alternate;
  `;

  if (!document.getElementById("vrgdg-gemma4-progress-style")) {
    const style = document.createElement("style");
    style.id = "vrgdg-gemma4-progress-style";
    style.textContent = `
      @keyframes vrgdgGemma4Pulse {
        from { transform: translateX(-55%); opacity: 0.65; }
        to { transform: translateX(145%); opacity: 1; }
      }
    `;
    document.head.appendChild(style);
  }

  bar.appendChild(fill);
  panel.append(title, message, bar);
  overlay.appendChild(panel);
  overlay.__vrgdgSetMessage = (text) => {
    message.textContent = String(text || "");
  };
  overlay.__vrgdgSetTitle = (text) => {
    title.textContent = String(text || "Gemma4");
  };
  document.body.appendChild(overlay);
  return overlay;
}

function showGemma4Progress(message, title = "Gemma4") {
  const overlay = ensureGemma4ProgressOverlay();
  overlay.__vrgdgSetTitle?.(title);
  overlay.__vrgdgSetMessage?.(message);
  overlay.style.display = "flex";
  return overlay;
}

function hideGemma4Progress() {
  const overlay = document.getElementById("vrgdg-gemma4-progress-overlay");
  if (overlay) overlay.style.display = "none";
}

function createGemma4ModalShell(titleText) {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100001;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.66);
  `;

  const panel = document.createElement("div");
  panel.style.cssText = `
    width: min(680px, calc(100vw - 32px));
    max-height: calc(100vh - 48px);
    display: flex;
    flex-direction: column;
    border: 1px solid #334155;
    border-radius: 8px;
    background: #111827;
    box-shadow: 0 20px 70px rgba(0, 0, 0, 0.48);
    padding: 18px;
    color: #f8fafc;
  `;

  const title = document.createElement("div");
  title.textContent = titleText;
  title.style.cssText = "font-size: 16px; font-weight: 800; margin-bottom: 8px;";

  panel.appendChild(title);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  return { overlay, panel };
}

function requestGemma4Notes(targetLabel) {
  return new Promise((resolve) => {
    const { overlay, panel } = createGemma4ModalShell(`Gemma4 ${targetLabel}`);

    const body = document.createElement("div");
    body.textContent = targetLabel === "Song Lyrics"
      ? "Describe what the song should be about. You can include mood, genre, vocal type, story, phrases, or anything else."
      : targetLabel === "Subject and Locations"
        ? "Optional: add exact subject and location details. If you want a specific subject line, paste it here and say to use it verbatim."
      : "Add optional notes like song style, genre, main character details, constraints, or anything else.";
    body.style.cssText = "font-size: 13px; color: #cbd5e1; line-height: 1.45; margin-bottom: 10px;";

    const textarea = document.createElement("textarea");
    textarea.rows = 7;
    textarea.placeholder = targetLabel === "Subject and Locations"
      ? "Example: Use this subject line verbatim: subject: a woman, with black hair, wearing a red leather jacket."
      : "Optional notes...";
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
      margin-bottom: 14px;
    `;

    const actions = document.createElement("div");
    actions.style.cssText = "display: flex; justify-content: flex-end; gap: 8px;";

    const noThanks = createButton(
      "No thanks",
      "border: 1px solid #475569; background: #1f2937; color: #e5e7eb; padding: 8px 12px; font-size: 12px; font-weight: 700;"
    );
    const ok = createButton(
      "OK",
      "border: 1px solid #059669; background: #10b981; color: #052e1b; padding: 8px 12px; font-size: 12px; font-weight: 800;"
    );

    function finish(value) {
      overlay.remove();
      resolve(value);
    }

    noThanks.addEventListener("click", () => finish(""));
    ok.addEventListener("click", () => finish(String(textarea.value || "").trim()));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) finish("");
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish("");
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) finish(String(textarea.value || "").trim());
    });

    actions.append(noThanks, ok);
    panel.append(body, textarea, actions);
    setTimeout(() => textarea.focus(), 0);
  });
}

function requestGemma4AdvancedListNotes() {
  return new Promise((resolve) => {
    const { overlay, panel } = createGemma4ModalShell("Gemma4 List Guidance");

    const body = document.createElement("div");
    body.textContent = "Optional: tell Gemma what kind of details you want in these lists, such as fast camera movement, fast character motion, harsher lighting, calmer expressions, or any style rules it should follow.";
    body.style.cssText = "font-size: 13px; color: #cbd5e1; line-height: 1.45; margin-bottom: 10px;";

    const textarea = document.createElement("textarea");
    textarea.rows = 7;
    textarea.placeholder = "Example: Make the camera motion fast and energetic. Character movement should feel intense, with quick gestures and strong performance energy.";
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
      margin-bottom: 14px;
    `;

    const actions = document.createElement("div");
    actions.style.cssText = "display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap;";
    const cancel = createButton(
      "Cancel",
      "border: 1px solid #475569; background: #1f2937; color: #e5e7eb; padding: 8px 12px; font-size: 12px; font-weight: 700;"
    );
    const skip = createButton(
      "Use No Extra Notes",
      "border: 1px solid #2563eb; background: #1d4ed8; color: white; padding: 8px 12px; font-size: 12px; font-weight: 800;"
    );
    const create = createButton(
      "Create Lists",
      "border: 1px solid #059669; background: #10b981; color: #052e1b; padding: 8px 12px; font-size: 12px; font-weight: 800;"
    );

    function finish(value) {
      overlay.remove();
      resolve(value);
    }

    cancel.addEventListener("click", () => finish(null));
    skip.addEventListener("click", () => finish(""));
    create.addEventListener("click", () => finish(String(textarea.value || "").trim()));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) finish(null);
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(null);
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        finish(String(textarea.value || "").trim());
      }
    });

    actions.append(cancel, skip, create);
    panel.append(body, textarea, actions);
    setTimeout(() => textarea.focus(), 0);
  });
}

function showGemma4ResultDialog({ title, text, isError = false }) {
  return new Promise((resolve) => {
    const { overlay, panel } = createGemma4ModalShell(title);

    const message = document.createElement("div");
    message.textContent = text;
    message.style.cssText = `
      max-height: min(48vh, 420px);
      overflow: auto;
      white-space: pre-wrap;
      border: 1px solid ${isError ? "#7f1d1d" : "#334155"};
      border-radius: 8px;
      background: ${isError ? "#2b1215" : "#0d1217"};
      color: ${isError ? "#fecaca" : "#f3f4f6"};
      padding: 12px;
      font-size: 13px;
      line-height: 1.45;
      margin: 4px 0 14px;
    `;

    const actions = document.createElement("div");
    actions.style.cssText = "display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap;";

    const cancel = createButton(
      isError ? "Close" : "Cancel",
      "border: 1px solid #475569; background: #1f2937; color: #e5e7eb; padding: 8px 12px; font-size: 12px; font-weight: 700;"
    );
    const retry = createButton(
      "Try Again",
      "border: 1px solid #d97706; background: #f59e0b; color: #111827; padding: 8px 12px; font-size: 12px; font-weight: 800;"
    );
    const use = createButton(
      "Use This",
      "border: 1px solid #059669; background: #10b981; color: #052e1b; padding: 8px 12px; font-size: 12px; font-weight: 800;"
    );
    use.style.display = isError ? "none" : "";

    function finish(action) {
      overlay.remove();
      resolve(action);
    }

    cancel.addEventListener("click", () => finish("cancel"));
    retry.addEventListener("click", () => finish("retry"));
    use.addEventListener("click", () => finish("use"));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) finish("cancel");
    });

    actions.append(cancel, retry, use);
    panel.append(message, actions);
  });
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

async function generateGemma4Text(payload) {
  const response = await api.fetchApi("/vrgdg/gemma4/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) {
    throw new Error(String(data?.error || `Gemma4 generation failed (${response.status})`));
  }
  return data;
}

async function unloadGemma4Model(payload) {
  const response = await api.fetchApi("/vrgdg/gemma4/unload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) {
    throw new Error(String(data?.error || `Gemma4 unload failed (${response.status})`));
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

      const gemmaLyricsButton = createButton(
        "Gemma4 Lyrics",
        "border: 1px solid #059669; background: #10b981; color: #052e1b; padding: 8px 12px; font-size: 12px; font-weight: 800;"
      );
      gemmaLyricsButton.type = "button";
      gemmaLyricsButton.addEventListener("click", () => {
        runGemma4ForField(field.key);
      });
      labelRow.appendChild(gemmaLyricsButton);
    }

    const gptUrlByField = {
      style_theme: STYLE_THEME_GPT_URL,
      story_idea: STORY_IDEA_GPT_URL,
      subjects_and_scenes: SUBJECT_LOCATION_GPT_URL,
    };
    const gptUrl = gptUrlByField[field.key];
    if (gptUrl) {
      const gemma4Button = createButton(
        "Gemma4",
        "border: 1px solid #059669; background: #10b981; color: #052e1b; padding: 8px 12px; font-size: 12px; font-weight: 800;"
      );
      gemma4Button.type = "button";
      gemma4Button.addEventListener("click", () => {
        runGemma4ForField(field.key);
      });
      labelRow.appendChild(gemma4Button);

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

  async function runGemma4ForField(targetKey) {
    const labelByTarget = {
      full_lyrics: "Song Lyrics",
      style_theme: "Style/theme",
      story_idea: "Story idea",
      subjects_and_scenes: "Subject and Locations",
    };
    const targetLabel = labelByTarget[targetKey] || targetKey;
    const extraNotes = await requestGemma4Notes(targetLabel);
    const modelFile = String(subgraphInputs.model_file?.value || "").trim();
      if (!modelFile) {
        setStatus("Choose a Gemma4 model in the LLM Model dropdown first.", true);
        return;
      }
      const requestedDuration = targetKey === "full_lyrics"
        ? window.prompt("Song duration in seconds for Gemma4 lyrics? Use the same duration you plan to use.", "120")
        : "";
      if (targetKey === "full_lyrics" && requestedDuration === null) {
        setStatus("Gemma4 lyrics generation was cancelled.");
        return;
      }

      const lyrics = String(textareas.full_lyrics?.value || "").trim();
    const lyricsBody = lyrics.replace(/^Full Lyrics\b/i, "").trim();
    const styleTheme = String(textareas.style_theme?.value || "").trim();
    const storyIdea = String(textareas.story_idea?.value || "").trim();
    if ((targetKey === "style_theme" || targetKey === "story_idea") && !lyricsBody) {
      setStatus("Add full lyrics before running Gemma4 for this field.", true);
      return;
    }
    if (targetKey === "subjects_and_scenes" && !storyIdea) {
      setStatus("Add or generate a story idea before running Gemma4 for Subject and Locations.", true);
      return;
    }

    while (true) {
      const progress = showGemma4Progress(
        targetKey === "subjects_and_scenes"
          ? "Generating Subject and Locations...\nThe model will unload after this finishes."
          : `Generating ${targetLabel}...`
      );
      setStatus(`Gemma4 is generating ${targetLabel}...`);

      try {
        const data = await generateGemma4Text({
          target: targetKey === "full_lyrics" ? "song_lyrics" : targetKey,
          model_file: modelFile,
          lyrics,
          style_theme: styleTheme,
          story_idea: storyIdea,
          notes: extraNotes,
          unload_after: targetKey === "subjects_and_scenes",
          n_ctx: 13000,
          max_new_tokens: 32000,
          duration: targetKey === "full_lyrics" ? String(requestedDuration || "120") : "",
        });
        const text = String(data.text || "").trim();
        if (!text) {
          throw new Error("Gemma4 returned an empty response.");
        }
        progress.__vrgdgSetMessage?.(data.unloaded ? "Done. Gemma4 was unloaded." : "Done.");
        hideGemma4Progress();
        const action = await showGemma4ResultDialog({
          title: `Gemma4 ${targetLabel} Result`,
          text,
        });
        if (action === "retry") {
          continue;
        }
        if (action === "use") {
          textareas[targetKey].value = text;
          syncNodeProperties();
          app.graph?.setDirtyCanvas?.(true, true);
          setStatus(data.unloaded ? `Gemma4 filled ${targetLabel} and unloaded the model.` : `Gemma4 filled ${targetLabel}.`);
        } else {
          setStatus(`Gemma4 ${targetLabel} result was not applied.`);
        }
        break;
      } catch (error) {
        const message = String(error?.message || error);
        progress.__vrgdgSetMessage?.(`Failed:\n${message}`);
        hideGemma4Progress();
        setStatus(message, true);
        const action = await showGemma4ResultDialog({
          title: `Gemma4 ${targetLabel} Failed`,
          text: message,
          isError: true,
        });
        if (action === "retry") {
          continue;
        }
        break;
      }
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

function findPart2NodeDeep(nodeId) {
  const targetId = Number(nodeId);
  const topNode = getPart2Node(targetId);
  if (topNode) return topNode;

  for (const graph of [app.graph, app.canvas?.graph, ...getGraphSubgraphDefinitions()]) {
    const node = getSubgraphNodes(graph).find((item) => Number(item?.id) === targetId);
    if (node) return node;
  }

  return null;
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

function getGraphSubgraphDefinitions() {
  const graph = app.graph || {};
  return [
    ...(Array.isArray(graph.subgraphs) ? graph.subgraphs : []),
    ...(Array.isArray(graph._subgraphs) ? graph._subgraphs : []),
    ...(Array.isArray(graph.definitions?.subgraphs) ? graph.definitions.subgraphs : []),
    ...(Array.isArray(graph.extra?.definitions?.subgraphs) ? graph.extra.definitions.subgraphs : []),
  ];
}

function getSubgraphDefinitionForNode(node) {
  if (!node) return null;
  const possibleIds = new Set(
    [
      node.type,
      node.comfyClass,
      node.subgraph_id,
      node.subgraphId,
      node.properties?.subgraph_id,
      node.properties?.subgraphId,
      node.properties?.["Subgraph ID"],
    ]
      .filter((value) => value !== undefined && value !== null && String(value).trim())
      .map((value) => String(value))
  );

  if (!possibleIds.size) return null;
  return getGraphSubgraphDefinitions().find((subgraph) => possibleIds.has(String(subgraph?.id || ""))) || null;
}

function isAdvancedPickerNode(node) {
  const type = String(node?.comfyClass || node?.type || node?.properties?.["Node name for S&R"] || "");
  return type === PART2_ADVANCED_NODE_NAME || type === PART2_ADVANCED_EASY_NODE_NAME;
}

function isEasyAdvancedPickerNode(node) {
  const type = String(node?.comfyClass || node?.type || node?.properties?.["Node name for S&R"] || "");
  return type === PART2_ADVANCED_EASY_NODE_NAME;
}

function findAdvancedPickerInSubgraph(subgraph) {
  return getSubgraphNodes(subgraph).find(isAdvancedPickerNode) || null;
}

function findAdvancedPickersInGraph(graph) {
  return getSubgraphNodes(graph).filter(isAdvancedPickerNode);
}

function findAdvancedPickerById(graph, id) {
  if (id === null || id === undefined) return null;
  const node = getSubgraphNodes(graph).find((item) => Number(item?.id) === Number(id));
  return isAdvancedPickerNode(node) ? node : null;
}

function getAdvancedEasyPickerIdForWorkflowKind(workflowKind = "part2") {
  return workflowKind === "part3" ? PART3_ADVANCED_EASY_NODE_ID : PART2_ADVANCED_EASY_NODE_ID;
}

function getNodePosition(node) {
  const pos = node?.pos || node?.position;
  if (Array.isArray(pos)) return [Number(pos[0]) || 0, Number(pos[1]) || 0];
  return [0, 0];
}

function distanceBetweenNodes(a, b) {
  if (!a || !b) return Number.POSITIVE_INFINITY;
  const [ax, ay] = getNodePosition(a);
  const [bx, by] = getNodePosition(b);
  return Math.hypot(ax - bx, ay - by);
}

function findPreferredAdvancedPickerInGraph(graph, ownerNode = null) {
  const pickers = findAdvancedPickersInGraph(graph);
  if (!pickers.length) return null;
  const linkedPickers = pickers.filter((node) =>
    (node.outputs || []).some((output) =>
      String(output?.name || "") === "combined_formatted_text" &&
      Array.isArray(output?.links) &&
      output.links.length
    )
  );
  const linkedEasyPickers = linkedPickers.filter(isEasyAdvancedPickerNode);
  const easyPickers = pickers.filter(isEasyAdvancedPickerNode);
  if (linkedEasyPickers.length) {
    return ownerNode
      ? linkedEasyPickers.slice().sort((a, b) => distanceBetweenNodes(a, ownerNode) - distanceBetweenNodes(b, ownerNode))[0]
      : linkedEasyPickers[0];
  }
  if (linkedPickers.length) {
    return ownerNode
      ? linkedPickers.slice().sort((a, b) => distanceBetweenNodes(a, ownerNode) - distanceBetweenNodes(b, ownerNode))[0]
      : linkedPickers[0];
  }
  if (easyPickers.length && ownerNode) {
    return easyPickers.slice().sort((a, b) => distanceBetweenNodes(a, ownerNode) - distanceBetweenNodes(b, ownerNode))[0];
  }
  return easyPickers[0] || pickers[0];
}

function getPart2AdvancedPickerContext(ownerNode = null) {
  return getPart2AdvancedPickerContexts({ ownerNode, singleTarget: true, workflowKind: "part2" })[0] || null;
}

function getPart2AdvancedPickerContexts(options = {}) {
  const ownerNode = options.ownerNode || null;
  const singleTarget = Boolean(options.singleTarget);
  const workflowKind = options.workflowKind || "part2";
  const contexts = [];
  const seen = new Set();
  const addContext = (node, subgraph) => {
    if (!node || !subgraph) return;
    const key = `${String(subgraph?.id || subgraph?.name || "graph")}:${String(node?.id || node?.title || contexts.length)}`;
    if (seen.has(key)) return;
    seen.add(key);
    contexts.push({ node, subgraph, isTopGraph: subgraph === app.graph });
  };

  addContext(findAdvancedPickerById(app.graph, getAdvancedEasyPickerIdForWorkflowKind(workflowKind)), app.graph);
  if (singleTarget && contexts[0]) return [contexts[0]];

  addContext(findPreferredAdvancedPickerInGraph(app.graph, ownerNode), app.graph);
  if (singleTarget && contexts[0] && isEasyAdvancedPickerNode(contexts[0].node)) return [contexts[0]];

  const activeGraph = app.canvas?.graph;
  addContext(findAdvancedPickerInSubgraph(activeGraph), activeGraph);
  if (singleTarget && contexts[0]) return [contexts[0]];

  const outerNode = ownerNode || getPart2Node(PART2_NODE_IDS.camera);
  if (!outerNode) return contexts;

  const directSubgraph = outerNode.subgraph || outerNode.sub_graph || outerNode.graph;
  addContext(findAdvancedPickerInSubgraph(directSubgraph), directSubgraph);

  const outerType = String(outerNode.type || outerNode.comfyClass || "");
  const matchingDefinition = getGraphSubgraphDefinitions().find((subgraph) => String(subgraph?.id || "") === outerType);
  addContext(findAdvancedPickerInSubgraph(matchingDefinition), matchingDefinition);
  if (singleTarget && contexts[0]) return [contexts[0]];

  for (const subgraph of getGraphSubgraphDefinitions()) {
    addContext(findAdvancedPickerInSubgraph(subgraph), subgraph);
    if (singleTarget && contexts[0]) return [contexts[0]];
  }

  return contexts;
}

function getPart2AdvancedPickerNode() {
  return getPart2AdvancedPickerContext()?.node || null;
}

function getPart2ZImageOptionalLoraNode() {
  const zNode = getPart2Node(PART2_NODE_IDS.zImageModels);
  const subNodes = [
    ...getSubgraphNodes(zNode?.subgraph),
    ...getSubgraphNodes(getSubgraphDefinitionForNode(zNode)),
  ];
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

function setPart2WidgetValueStrict(nodeId, name, value) {
  const node = getPart2Node(nodeId);
  if (!node) return false;

  const widget = getPart2Widget(node, name);
  if (!widget) return false;

  widget.value = value;
  widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);

  if (Array.isArray(node.widgets_values)) {
    const widgetIndex = (node.widgets || []).findIndex((item) => item?.name === name);
    if (widgetIndex >= 0) node.widgets_values[widgetIndex] = value;
  }

  app.graph?.setDirtyCanvas?.(true, true);
  return true;
}

function getTriggerNodeId(kind, workflowKind = "part2") {
  if (kind === "zImage") return PART2_Z_IMAGE_TRIGGER_NODE_ID;
  return workflowKind === "part3" ? PART3_LTX_TRIGGER_NODE_ID : PART2_LTX_TRIGGER_NODE_ID;
}

function findWidgetByAliases(node, aliases, fallbackIndex = -1) {
  if (!node) return null;
  const accepted = new Set(aliases.map((alias) => String(alias).trim().toLowerCase()));
  const widgets = node.widgets || [];
  return widgets.find((widget) => accepted.has(String(widget?.name || "").trim().toLowerCase())) || widgets[fallbackIndex] || null;
}

function getTriggerWidgetValue(node, aliases, fallbackIndex, fallbackValue = "") {
  const widget = findWidgetByAliases(node, aliases, fallbackIndex);
  if (widget && Object.prototype.hasOwnProperty.call(widget, "value")) return widget.value;
  if (Array.isArray(node?.widgets_values) && fallbackIndex >= 0 && fallbackIndex < node.widgets_values.length) {
    return node.widgets_values[fallbackIndex];
  }
  return fallbackValue;
}

function setTriggerWidgetValue(node, aliases, fallbackIndex, value) {
  if (!node) return false;
  const widget = findWidgetByAliases(node, aliases, fallbackIndex);
  if (widget && Object.prototype.hasOwnProperty.call(widget, "value")) {
    widget.value = value;
    widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
  }

  if (Array.isArray(node.widgets_values)) {
    const accepted = new Set(aliases.map((alias) => String(alias).trim().toLowerCase()));
    const widgetIndex = (node.widgets || []).findIndex((item) => accepted.has(String(item?.name || "").trim().toLowerCase()));
    const resolvedIndex = widgetIndex >= 0 ? widgetIndex : fallbackIndex;
    if (resolvedIndex >= 0) node.widgets_values[resolvedIndex] = value;
  }

  app.graph?.setDirtyCanvas?.(true, true);
  return Boolean(widget) || (Array.isArray(node?.widgets_values) && fallbackIndex >= 0);
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

function getValidSlotCountDraftValue(value) {
  if (value === undefined || value === null || String(value).trim() === "") return null;
  const count = Number(value);
  if (!Number.isFinite(count)) return null;
  return String(Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Math.round(count))));
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

function createPart2TriggerControls(noteText) {
  const useTrigger = document.createElement("input");
  useTrigger.type = "checkbox";

  const triggerWord = stylePart2Input(document.createElement("input"));
  triggerWord.type = "text";
  triggerWord.placeholder = "Trigger word or LoRA name";

  const useWrapper = createPart2Field("Use Trigger Word", useTrigger, noteText);
  const wordWrapper = createPart2Field("Add Trigger Word", triggerWord, "This is sent to the trigger-word subgraph string_1 input.");

  return { useTrigger, triggerWord, useWrapper, wordWrapper };
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

  const note = document.createElement("div");
  note.textContent = "Download both files for the Gemma LLM Node.";
  note.style.cssText = `
    margin-top: 5px;
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.35;
  `;

  labelRow.append(label, buttons);
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
    workflowKind: "part2",
    workflowNode: null,
    modelSelects: {},
    modelFieldWrappers: {},
    settings: {},
    useSrt: null,
    lora: {
      useCustom: null,
      count: null,
      twoPass: null,
      slots: [],
      section: null,
      trigger: null,
    },
    zImageLora: {
      useCustom: null,
      count: null,
      slots: [],
      section: null,
      trigger: null,
    },
    advanced: {
      enabled: null,
      count: null,
      modeAll: null,
      pickers: [],
    },
    promptJson: null,
    wrappers: {},
  };

  function getWorkflowNodeIds() {
    return controls.workflowKind === "part3" ? PART3_NODE_IDS : PART2_NODE_IDS;
  }

  function isModelFieldVisible(field) {
    return controls.workflowKind !== "part3" || !field.zImageOnly;
  }

  const modelSection = createPart2Section("Models", "Model dropdowns for LTX 2.3, Z-image and the LLM node.");
  const modelGrid = document.createElement("div");
  modelGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px;";

  for (const field of PART2_MODEL_FIELDS) {
    const select = stylePart2Input(document.createElement("select"));
    controls.modelSelects[`${field.nodeId}:${field.key}`] = select;
    const wrapper = createPart2ModelField(field, select);
    controls.modelFieldWrappers[`${field.nodeId}:${field.key}`] = wrapper;
    modelGrid.appendChild(wrapper);
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

  const ltxTriggerControls = createPart2TriggerControls("Optional trigger word for the selected LTX LoRA.");
  controls.lora.trigger = ltxTriggerControls;
  ltxTriggerControls.wordWrapper.style.display = "none";
  loraGrid.append(ltxTriggerControls.useWrapper, ltxTriggerControls.wordWrapper);

  const loraCountInput = stylePart2Input(document.createElement("input"));
  loraCountInput.type = "number";
  loraCountInput.min = "0";
  loraCountInput.max = String(PART2_MAX_LORA_SLOTS);
  loraCountInput.step = "1";
  controls.lora.count = loraCountInput;
  const loraCountWrapper = createPart2Field("LoRA Count", loraCountInput, "How many LoRA slots to show and apply.");
  loraCountWrapper.style.display = "none";
  loraGrid.appendChild(loraCountWrapper);

  const loraTwoPassSelect = stylePart2Input(document.createElement("select"));
  for (const [value, label] of [["true", "ON"], ["false", "OFF"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    loraTwoPassSelect.appendChild(option);
  }
  controls.lora.twoPass = loraTwoPassSelect;
  const loraTwoPassWrapper = createPart2Field("LTX Two Pass Strength", loraTwoPassSelect, "ON uses half of each selected LoRA strength on first pass to preserve motion, then the full selected strength on the upscale pass. OFF uses the selected strength on both passes.");
  loraTwoPassWrapper.style.display = "none";
  loraGrid.appendChild(loraTwoPassWrapper);

  for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
    const select = stylePart2Input(document.createElement("select"));
    const strength = stylePart2Input(document.createElement("input"));
    strength.type = "number";
    strength.step = "0.01";
    strength.min = "-100";
    strength.max = "100";

    const loraWrapper = createPart2Field(`LoRA ${i}`, select);
    const strengthWrapper = createPart2Field(`Strength ${i}`, strength, "Selected/full strength for the upscale pass. First pass uses half of this value when two-pass strength is ON.");
    loraWrapper.style.display = "none";
    strengthWrapper.style.display = "none";
    controls.lora.slots.push({ select, strength, loraWrapper, strengthWrapper });
    loraGrid.append(loraWrapper, strengthWrapper);
  }

  loraSection.appendChild(loraGrid);

  const zImageLoraSection = createPart2Section(
    "Z-Image Optional LoRA",
    "Optional LoRA for the Z-Image still-image branch inside subgraph node 797. This does not use two-pass strength; selected LoRAs apply at the strength you enter."
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

  const zImageTriggerControls = createPart2TriggerControls("Optional trigger word for the selected Z-Image LoRA.");
  controls.zImageLora.trigger = zImageTriggerControls;
  zImageTriggerControls.wordWrapper.style.display = "none";
  zImageLoraGrid.append(zImageTriggerControls.useWrapper, zImageTriggerControls.wordWrapper);

  const zImageLoraCountInput = stylePart2Input(document.createElement("input"));
  zImageLoraCountInput.type = "number";
  zImageLoraCountInput.min = "0";
  zImageLoraCountInput.max = String(PART2_MAX_LORA_SLOTS);
  zImageLoraCountInput.step = "1";
  controls.zImageLora.count = zImageLoraCountInput;
  const zImageLoraCountWrapper = createPart2Field("Z-Image LoRA Count", zImageLoraCountInput, "How many Z-Image LoRA slots to show and apply.");
  zImageLoraCountWrapper.style.display = "none";
  zImageLoraGrid.appendChild(zImageLoraCountWrapper);

  for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
    const select = stylePart2Input(document.createElement("select"));
    const strength = stylePart2Input(document.createElement("input"));
    strength.type = "number";
    strength.step = "0.01";
    strength.min = "-100";
    strength.max = "100";

    const loraWrapper = createPart2Field(`Z-Image LoRA ${i}`, select);
    const strengthWrapper = createPart2Field(`Z-Image Strength ${i}`, strength, "Applied at this exact strength.");
    loraWrapper.style.display = "none";
    strengthWrapper.style.display = "none";
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

  const advancedSection = createPart2Section("Advanced Prompt Details", "Create custom details for the LLM.");
  const advancedIntro = document.createElement("div");
  advancedIntro.style.cssText = `
    border: 1px solid #243244;
    border-radius: 8px;
    background: #0d1217;
    padding: 10px 12px;
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.45;
    margin-bottom: 14px;
  `;
  advancedIntro.textContent = "Each active list adds one prompt detail category. The workflow supplies the scene index and random seed outside this subgraph, so this UI focuses on the editable category label, list text, selection behavior, and how multiple selected items are phrased.";
  advancedSection.appendChild(advancedIntro);

  const advancedGptRow = document.createElement("div");
  advancedGptRow.style.cssText = `
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    border: 1px solid #243244;
    border-radius: 8px;
    background: #0d1217;
    padding: 10px 12px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  `;

  const advancedGptNote = document.createElement("div");
  advancedGptNote.textContent = "Use this GPT to help create your lists based off your prompt, then change the selection mode to index. This will create fully custom lists that go with each prompt.";
  advancedGptNote.style.cssText = "color: #cbd5e1; font-size: 12px; line-height: 1.45; flex: 1 1 360px;";

  const advancedGptButton = createButton(
    "Open List Helper GPT",
    "border: 1px solid #2563eb; background: #1d4ed8; color: white; font-weight: 700;"
  );
  advancedGptButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    window.open(ADVANCED_PROMPT_DETAILS_GPT_URL, "_blank", "noopener,noreferrer");
  });

  const advancedGemmaButton = createButton(
    "Gemma4 Create Lists",
    "border: 1px solid #059669; background: #10b981; color: #052e1b; font-weight: 800;"
  );

  const advancedHelperButtons = document.createElement("div");
  advancedHelperButtons.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;";
  advancedHelperButtons.append(advancedGemmaButton, advancedGptButton);

  advancedGptRow.append(advancedGptNote, advancedHelperButtons);
  advancedSection.appendChild(advancedGptRow);

  const advancedCountRow = document.createElement("div");
  advancedCountRow.style.cssText = "display: grid; grid-template-columns: minmax(160px, 220px) minmax(180px, 260px) minmax(220px, 320px) 1fr; gap: 12px; align-items: end; margin-bottom: 14px;";

  const advancedEnabled = document.createElement("input");
  advancedEnabled.type = "checkbox";
  advancedEnabled.checked = false;
  controls.advanced.enabled = advancedEnabled;
  advancedCountRow.appendChild(createPart2Field("Advanced Settings", advancedEnabled, "Turn this on only when you want to add custom prompt detail lists."));

  const advancedCount = stylePart2Input(document.createElement("input"));
  advancedCount.type = "number";
  advancedCount.min = "0";
  advancedCount.max = String(PART2_ADVANCED_MAX_PICKERS);
  advancedCount.step = "1";
  advancedCount.value = "0";
  controls.advanced.count = advancedCount;
  advancedCountRow.appendChild(createPart2Field("Settings Count", advancedCount, ""));

  const advancedModeAll = createAdvancedSelectionModeSelect();
  controls.advanced.modeAll = advancedModeAll;
  advancedCountRow.appendChild(createPart2Field("Selection Mode For All", advancedModeAll, "Quickly sets every active list to the same selection mode. You can still change any list afterward."));

  const advancedSummary = document.createElement("div");
  advancedSummary.style.cssText = "font-size: 12px; color: #94a3b8; padding-bottom: 8px;";
  advancedCountRow.appendChild(advancedSummary);
  advancedSection.appendChild(advancedCountRow);

  const advancedApplyRow = document.createElement("div");
  advancedApplyRow.style.cssText = "display: flex; justify-content: flex-end; margin: -2px 0 14px;";
  const advancedApplyButton = createButton(
    "Apply Advanced Settings Only",
    "border: 1px solid #2563eb; background: #1d4ed8; color: white; font-weight: 700;"
  );
  advancedApplyRow.appendChild(advancedApplyButton);
  advancedSection.appendChild(advancedApplyRow);

  const advancedPickerList = document.createElement("div");
  advancedPickerList.style.cssText = "display: flex; flex-direction: column; gap: 12px;";
  advancedSection.appendChild(advancedPickerList);

  function createAdvancedPresetSelect() {
    const select = stylePart2Input(document.createElement("select"));
    for (const preset of PART2_ADVANCED_PRESETS) {
      const option = document.createElement("option");
      option.value = preset;
      option.textContent = preset;
      select.appendChild(option);
    }
    return select;
  }

  function createAdvancedSelectionModeSelect() {
    const select = stylePart2Input(document.createElement("select"));
    for (const mode of PART2_ADVANCED_SELECTION_MODES) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = mode === "index" ? "Index-based" : mode.replace(/\b\w/g, (char) => char.toUpperCase());
      select.appendChild(option);
    }
    return select;
  }

  function createAdvancedField(labelText, control, noteText = "") {
    const field = createPart2Field(labelText, control, noteText);
    const note = field.lastElementChild;
    if (note) {
      note.style.fontSize = "12px";
      note.style.lineHeight = "1.45";
      note.style.color = "#b6c2d1";
    }
    return field;
  }

  for (let i = 1; i <= PART2_ADVANCED_MAX_PICKERS; i++) {
    const picker = {
      wrapper: document.createElement("div"),
      header: document.createElement("button"),
      body: document.createElement("div"),
      preset: createAdvancedPresetSelect(),
      label: stylePart2Input(document.createElement("input")),
      selectionMode: createAdvancedSelectionModeSelect(),
      index: stylePart2Input(document.createElement("input")),
      seed: stylePart2Input(document.createElement("input")),
      pickCount: stylePart2Input(document.createElement("input")),
      template: stylePart2Input(document.createElement("input")),
      items: document.createElement("textarea"),
      collapsed: i > 1,
    };

    picker.wrapper.style.cssText = `
      border: 1px solid #364152;
      border-radius: 8px;
      background: #0f141a;
      overflow: hidden;
    `;
    picker.header.type = "button";
    picker.header.style.cssText = `
      width: 100%;
      border: 0;
      background: #14191f;
      color: #f3f4f6;
      padding: 10px 12px;
      cursor: pointer;
      text-align: left;
      font-size: 13px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    `;
    picker.body.style.cssText = "padding: 12px;";

    picker.index.type = "number";
    picker.index.step = "1";
    picker.seed.type = "number";
    picker.seed.step = "1";
    picker.pickCount.type = "number";
    picker.pickCount.min = "1";
    picker.pickCount.max = "50";
    picker.pickCount.step = "1";
    picker.items.rows = 8;
    stylePart2Input(picker.items);
    picker.items.style.resize = "vertical";

    const pickerGrid = document.createElement("div");
    pickerGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-bottom: 12px;";
    pickerGrid.append(
      createAdvancedField("Preset", picker.preset, "Pick a starter category. The list below stays editable after you choose one."),
      createAdvancedField("Label", picker.label, "The Label for this category."),
      createAdvancedField("Selection Mode", picker.selectionMode, "Index-based follows the scene number. Random picks from the list. Random No Repeat shuffles the list before repeating.")
    );

    const pickerRulesGrid = document.createElement("div");
    pickerRulesGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-bottom: 10px;";
    pickerRulesGrid.append(
      createAdvancedField(
        "Items Per Prompt",
        picker.pickCount,
        "How many entries this category adds to each scene. Example: with Facial Expression, 1 might output Soft smile. If set to 2, it might output start with Calm expression then follow with Cold stare."
      ),
      createAdvancedField(
        "Item Template",
        picker.template,
        "Used when Items Per Prompt is 2. Keep {item1} and {item2}; the node replaces them with the two selected list entries."
      )
    );
    picker.body.append(
      pickerGrid,
      pickerRulesGrid,
      createAdvancedField("List", picker.items, "One entry per line. Presets fill this list, but you can edit, remove, or add entries.")
    );
    picker.wrapper.append(picker.header, picker.body);
    advancedPickerList.appendChild(picker.wrapper);
    controls.advanced.pickers.push(picker);
  }

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

  const tabBar = document.createElement("div");
  tabBar.style.cssText = `
    display: flex;
    gap: 8px;
    margin-bottom: 14px;
    border-bottom: 1px solid #364152;
    flex-wrap: wrap;
  `;

  function createPart2TabButton(label) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.style.cssText = `
      border: 1px solid transparent;
      border-bottom: 0;
      background: transparent;
      color: #cbd5e1;
      border-radius: 8px 8px 0 0;
      padding: 9px 12px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
    `;
    return button;
  }

  const mainTabButton = createPart2TabButton("Workflow Settings");
  const advancedTabButton = createPart2TabButton("Advanced Settings");
  tabBar.append(mainTabButton, advancedTabButton);

  const mainTabPanel = document.createElement("div");
  mainTabPanel.append(modelSection, loraSection, zImageLoraSection, settingsSection, promptSection);

  const advancedTabPanel = document.createElement("div");
  advancedTabPanel.style.display = "none";
  advancedTabPanel.style.minHeight = "120px";
  advancedTabPanel.appendChild(advancedSection);

  function setPart2ActiveTab(tabName) {
    const isAdvanced = tabName === "advanced";
    mainTabPanel.style.display = isAdvanced ? "none" : "block";
    advancedTabPanel.style.display = isAdvanced ? "block" : "none";

    for (const [button, active] of [[mainTabButton, !isAdvanced], [advancedTabButton, isAdvanced]]) {
      button.style.background = active ? "#14191f" : "transparent";
      button.style.borderColor = active ? "#364152" : "transparent";
      button.style.color = active ? "#f3f4f6" : "#cbd5e1";
    }
  }

  mainTabButton.addEventListener("click", () => setPart2ActiveTab("main"));
  advancedTabButton.addEventListener("click", () => setPart2ActiveTab("advanced"));
  setPart2ActiveTab("main");

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

  panel.append(banner, titleRow, tabBar, mainTabPanel, advancedTabPanel, status, actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  function setStatus(message, isError = false) {
    status.textContent = message || "";
    status.style.color = isError ? "#fca5a5" : "#cbd5e1";
  }

  let suppressDraftSave = false;
  let hasDraft = false;

  function collectPart2Draft() {
    const draft = {
      modelSelects: {},
      settings: {},
      useSrt: controls.useSrt.value,
      advanced: collectAdvancedDraft(),
      promptJson: controls.promptJson.value,
      lora: {
        useCustom: controls.lora.useCustom.value,
        count: controls.lora.count.value,
        twoPass: controls.lora.twoPass.value,
        slots: controls.lora.slots.map((slot) => ({
          lora: slot.select.value,
          strength: slot.strength.value,
        })),
        trigger: {
          enabled: controls.lora.trigger.useTrigger.checked,
          word: controls.lora.trigger.triggerWord.value,
        },
      },
    };

    if (controls.workflowKind !== "part3") {
      draft.zImageLora = {
        useCustom: controls.zImageLora.useCustom.value,
        count: controls.zImageLora.count.value,
        slots: controls.zImageLora.slots.map((slot) => ({
          lora: slot.select.value,
          strength: slot.strength.value,
        })),
        trigger: {
          enabled: controls.zImageLora.trigger.useTrigger.checked,
          word: controls.zImageLora.trigger.triggerWord.value,
        },
      };
    }

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
      localStorage.setItem(getPart2DraftStorageKey(), JSON.stringify(collectPart2Draft()));
      hasDraft = true;
    } catch (error) {
      // Browser storage can be disabled; form still works normally.
    }
  }

  function loadPart2Draft() {
    try {
      const raw = localStorage.getItem(getPart2DraftStorageKey());
      return raw ? JSON.parse(raw) : null;
    } catch (error) {
      return null;
    }
  }

  function clearPart2Draft() {
    try {
      localStorage.removeItem(getPart2DraftStorageKey());
    } catch (error) {
      // ignore
    }
    hasDraft = false;
  }

  function getPart2DraftStorageKey() {
    const nodeId = controls.workflowNode?.id ?? "global";
    return `${PART2_DRAFT_STORAGE_KEY}.${controls.workflowKind}.node.${nodeId}`;
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
    if (draft.advanced !== undefined) applyAdvancedDraft(draft.advanced);
    if (draft.promptJson !== undefined) controls.promptJson.value = String(draft.promptJson);

    if (draft.lora) {
      controls.lora.useCustom.value = String(draft.lora.useCustom ?? "false");
      const loraDraftCount = getValidSlotCountDraftValue(draft.lora.count);
      if (loraDraftCount !== null) controls.lora.count.value = loraDraftCount;
      controls.lora.twoPass.value = String(draft.lora.twoPass ?? "true");
      for (let i = 0; i < controls.lora.slots.length; i++) {
        const slotDraft = draft.lora.slots?.[i] || {};
        if (slotDraft.lora !== undefined) setSelectValueAllowingDraft(controls.lora.slots[i].select, slotDraft.lora);
        if (slotDraft.strength !== undefined) controls.lora.slots[i].strength.value = String(slotDraft.strength);
      }
      if (draft.lora.trigger) {
        controls.lora.trigger.useTrigger.checked = Boolean(draft.lora.trigger.enabled);
        controls.lora.trigger.triggerWord.value = String(draft.lora.trigger.word ?? "");
      }
      updateLoraVisibility();
    }

    if (draft.zImageLora) {
      controls.zImageLora.useCustom.value = String(draft.zImageLora.useCustom ?? "false");
      const zImageDraftCount = getValidSlotCountDraftValue(draft.zImageLora.count);
      if (zImageDraftCount !== null) controls.zImageLora.count.value = zImageDraftCount;
      for (let i = 0; i < controls.zImageLora.slots.length; i++) {
        const slotDraft = draft.zImageLora.slots?.[i] || {};
        if (slotDraft.lora !== undefined) setSelectValueAllowingDraft(controls.zImageLora.slots[i].select, slotDraft.lora);
        if (slotDraft.strength !== undefined) controls.zImageLora.slots[i].strength.value = String(slotDraft.strength);
      }
      if (draft.zImageLora.trigger) {
        controls.zImageLora.trigger.useTrigger.checked = Boolean(draft.zImageLora.trigger.enabled);
        controls.zImageLora.trigger.triggerWord.value = String(draft.zImageLora.trigger.word ?? "");
      }
      updateZImageLoraVisibility();
    }

    updateFixedDurationVisibility();
    suppressDraftSave = false;
    hasDraft = true;
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
    controls.lora.trigger.useWrapper.style.display = "block";
    controls.lora.trigger.wordWrapper.style.display = controls.lora.trigger.useTrigger.checked ? "block" : "none";

    for (let i = 0; i < controls.lora.slots.length; i++) {
      const visible = useLoras && i < count;
      controls.lora.slots[i].loraWrapper.style.display = visible ? "block" : "none";
      controls.lora.slots[i].strengthWrapper.style.display = visible ? "block" : "none";
    }
  }

  function updateZImageLoraVisibility() {
    if (controls.workflowKind === "part3") {
      controls.zImageLora.section.style.display = "none";
      return;
    }

    const useLoras = String(controls.zImageLora.useCustom.value || "false").toLowerCase() === "true";
    const rawCount = Number(controls.zImageLora.count.value || 0);
    const count = useLoras ? Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Number.isFinite(rawCount) ? rawCount : 0)) : 0;

    controls.zImageLora.count.parentElement.style.display = useLoras ? "block" : "none";
    controls.zImageLora.trigger.useWrapper.style.display = "block";
    controls.zImageLora.trigger.wordWrapper.style.display = controls.zImageLora.trigger.useTrigger.checked ? "block" : "none";

    for (let i = 0; i < controls.zImageLora.slots.length; i++) {
      const visible = useLoras && i < count;
      controls.zImageLora.slots[i].loraWrapper.style.display = visible ? "block" : "none";
      controls.zImageLora.slots[i].strengthWrapper.style.display = visible ? "block" : "none";
    }
  }

  function updateWorkflowSpecificVisibility() {
    const isPart3 = controls.workflowKind === "part3";
    for (const field of PART2_MODEL_FIELDS) {
      const wrapper = controls.modelFieldWrappers[`${field.nodeId}:${field.key}`];
      if (wrapper) wrapper.style.display = isPart3 && field.zImageOnly ? "none" : "block";
    }
    zImageLoraSection.style.display = isPart3 ? "none" : "block";
  }

  function getAdvancedWidgetFallbackIndex(name) {
    const indexes = getAdvancedWidgetFallbackIndexes(name);
    return indexes.length ? indexes[0] : -1;
  }

  function getAdvancedWidgetFallbackIndexesForNode(node, name) {
    const indexes = getAdvancedWidgetFallbackIndexes(name);
    if (!Array.isArray(node?.widgets_values)) return indexes;

    const valueCount = node.widgets_values.length;
    const expectedCompactCount = 2 + PART2_ADVANCED_MAX_PICKERS * 6;
    const expectedNewCount = 2 + PART2_ADVANCED_MAX_PICKERS * 10;
    const expectedOldCount = 2 + PART2_ADVANCED_MAX_PICKERS * 12;

    if (valueCount === expectedCompactCount) return indexes.slice(0, 1);
    if (valueCount === expectedNewCount) return indexes.slice(1, 2);
    if (valueCount === expectedOldCount) return indexes.slice(-1);

    return indexes.slice(0, 1);
  }

  function getAdvancedWidgetFallbackIndexes(name) {
    if (name === "picker_count") return [0];
    if (name === "joiner") return [1];
    const textName = String(name || "");
    const compactMatch = /^(preset|items|label|selection_mode|two_item_template|pick_count)_(\d+)$/.exec(textName);
    const newMatch = /^(preset|items|label|max_items|split_mode|selection_mode|multi_format|two_item_template|keep_empty|pick_count)_(\d+)$/.exec(textName);
    const oldMatch = /^(preset|index|items|label|max_items|split_mode|selection_mode|seed|multi_format|two_item_template|keep_empty|pick_count)_(\d+)$/.exec(textName);
    const indexes = [];

    if (compactMatch) {
      const compactOffsets = {
        preset: 0,
        items: 1,
        label: 2,
        selection_mode: 3,
        two_item_template: 4,
        pick_count: 5,
      };
      indexes.push(2 + (Number(compactMatch[2]) - 1) * 6 + compactOffsets[compactMatch[1]]);
    }

    if (newMatch) {
      const newOffsets = {
        preset: 0,
        items: 1,
        label: 2,
        max_items: 3,
        split_mode: 4,
        selection_mode: 5,
        multi_format: 6,
        two_item_template: 7,
        keep_empty: 8,
        pick_count: 9,
      };
      indexes.push(2 + (Number(newMatch[2]) - 1) * 10 + newOffsets[newMatch[1]]);
    }

    if (oldMatch) {
      const oldOffsets = {
        preset: 0,
        index: 1,
        items: 2,
        label: 3,
        max_items: 4,
        split_mode: 5,
        selection_mode: 6,
        seed: 7,
        multi_format: 8,
        two_item_template: 9,
        keep_empty: 10,
        pick_count: 11,
      };
      indexes.push(2 + (Number(oldMatch[2]) - 1) * 12 + oldOffsets[oldMatch[1]]);
    }

    return [...new Set(indexes)];
  }

  function getAdvancedWidgetFallbackIndexOld(name) {
    if (name === "picker_count") return 0;
    if (name === "joiner") return 1;
    const match = /^(preset|index|items|label|max_items|split_mode|selection_mode|seed|multi_format|two_item_template|keep_empty|pick_count)_(\d+)$/.exec(String(name || ""));
    if (!match) return -1;
    const fieldOffsets = {
      preset: 0,
      index: 1,
      items: 2,
      label: 3,
      max_items: 4,
      split_mode: 5,
      selection_mode: 6,
      seed: 7,
      multi_format: 8,
      two_item_template: 9,
      keep_empty: 10,
      pick_count: 11,
    };
    return 2 + (Number(match[2]) - 1) * 12 + fieldOffsets[match[1]];
  }

  function getAdvancedWidget(node, name) {
    if (!node) return null;
    return (node.widgets || []).find((widget) => widget?.name === name) || null;
  }

  function getAdvancedWidgetValue(node, name, fallbackValue = "") {
    const widget = getAdvancedWidget(node, name);
    if (widget && Object.prototype.hasOwnProperty.call(widget, "value")) return widget.value;
    const fallbackIndexes = getAdvancedWidgetFallbackIndexesForNode(node, name);
    if (Array.isArray(node?.widgets_values)) {
      for (const fallbackIndex of fallbackIndexes) {
        if (fallbackIndex >= 0 && fallbackIndex < node.widgets_values.length) {
          const value = node.widgets_values[fallbackIndex];
          if (value !== undefined && value !== null && String(value) !== "") return value;
        }
      }
    }
    return fallbackValue;
  }

  function setAdvancedWidgetValue(node, name, value) {
    if (!node) return false;
    const fallbackIndexes = getAdvancedWidgetFallbackIndexesForNode(node, name);
    const widget = getAdvancedWidget(node, name);
    if (widget && Object.prototype.hasOwnProperty.call(widget, "value")) {
      widget.value = value;
      widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
    }

    if (Array.isArray(node.widgets_values)) {
      for (const fallbackIndex of fallbackIndexes) {
        if (fallbackIndex >= 0) node.widgets_values[fallbackIndex] = value;
      }
    }

    if (Array.isArray(node?.widgets)) {
      for (const fallbackIndex of fallbackIndexes) {
        const fallbackWidget = node.widgets[fallbackIndex];
        if (!fallbackWidget || fallbackWidget === widget || !Object.prototype.hasOwnProperty.call(fallbackWidget, "value")) continue;
        fallbackWidget.value = value;
        fallbackWidget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
      }
    }

    app.graph?.setDirtyCanvas?.(true, true);
    return Boolean(widget) || (Array.isArray(node.widgets_values) && fallbackIndexes.length > 0);
  }

  function setEasyAdvancedWidgetValue(node, name, value) {
    const widget = getAdvancedWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
    const index = (node.widgets || []).indexOf(widget);
    if (Array.isArray(node.widgets_values) && index >= 0) node.widgets_values[index] = value;
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
  }

  function getEasyAdvancedWidgetValue(node, name, fallback = "") {
    const widget = getAdvancedWidget(node, name);
    return widget?.value ?? fallback;
  }

  function syncAdvancedOuterProxyValue(name, value) {
    const outerNode = getPart2Node(PART2_NODE_IDS.camera);
    const widget = getPart2Widget(outerNode, name);
    if (widget && Object.prototype.hasOwnProperty.call(widget, "value")) {
      widget.value = value;
      widget.callback?.(value, app.canvas, outerNode, app.canvas?.graph_mouse);
    }

    if (Array.isArray(outerNode?.widgets_values)) {
      const widgetIndex = (outerNode.widgets || []).findIndex((item) => item?.name === name);
      if (widgetIndex >= 0) outerNode.widgets_values[widgetIndex] = value;
    }
  }

  function getSubgraphLinksArray(subgraph) {
    if (Array.isArray(subgraph?.links)) return subgraph.links;
    if (Array.isArray(subgraph?._links)) return subgraph._links;
    return null;
  }

  function findSubgraphInputSlot(subgraph, names) {
    const accepted = new Set(names.map((name) => String(name).trim().toLowerCase()));
    const inputs = Array.isArray(subgraph?.inputs) ? subgraph.inputs : [];
    return inputs.findIndex((input) => {
      const name = String(input?.name || "").trim().toLowerCase();
      const label = String(input?.label || input?.localized_name || "").trim().toLowerCase();
      return accepted.has(name) || accepted.has(label);
    });
  }

  function nextSubgraphLinkId(subgraph, links) {
    const current = Number(subgraph?.state?.lastLinkId || 0);
    const maxExisting = links.reduce((maxId, link) => Math.max(maxId, Number(link?.id || 0)), 0);
    const next = Math.max(current, maxExisting) + 1;
    subgraph.state = subgraph.state || {};
    subgraph.state.lastLinkId = next;
    return next;
  }

  function removeSubgraphTargetLink(subgraph, links, targetNode, targetSlot) {
    const input = targetNode?.inputs?.[targetSlot];
    const linkId = input?.link;
    if (linkId === null || linkId === undefined) return;

    const index = links.findIndex((link) => Number(link?.id) === Number(linkId));
    if (index >= 0) {
      const [oldLink] = links.splice(index, 1);
      const originInput = Array.isArray(subgraph?.inputs) ? subgraph.inputs[oldLink.origin_slot] : null;
      if (Array.isArray(originInput?.linkIds)) {
        originInput.linkIds = originInput.linkIds.filter((id) => Number(id) !== Number(linkId));
      }
    }
    input.link = null;
  }

  function ensureSubgraphInputLink(subgraph, sourceSlot, targetNode, targetInputName, type) {
    const links = getSubgraphLinksArray(subgraph);
    const targetSlot = (targetNode?.inputs || []).findIndex((input) => input?.name === targetInputName);
    if (!links || sourceSlot < 0 || targetSlot < 0) return false;

    const existing = links.find((link) =>
      Number(link?.origin_id) === -10 &&
      Number(link?.origin_slot) === Number(sourceSlot) &&
      Number(link?.target_id) === Number(targetNode.id) &&
      Number(link?.target_slot) === Number(targetSlot)
    );
    if (existing) {
      targetNode.inputs[targetSlot].link = existing.id;
      return false;
    }

    removeSubgraphTargetLink(subgraph, links, targetNode, targetSlot);
    const linkId = nextSubgraphLinkId(subgraph, links);
    links.push({
      id: linkId,
      origin_id: -10,
      origin_slot: sourceSlot,
      target_id: targetNode.id,
      target_slot: targetSlot,
      type,
    });

    targetNode.inputs[targetSlot].link = linkId;
    const sourceInput = Array.isArray(subgraph?.inputs) ? subgraph.inputs[sourceSlot] : null;
    if (sourceInput) {
      sourceInput.linkIds = Array.isArray(sourceInput.linkIds) ? sourceInput.linkIds : [];
      if (!sourceInput.linkIds.some((id) => Number(id) === Number(linkId))) {
        sourceInput.linkIds.push(linkId);
      }
    }
    return true;
  }

  function findSubgraphSourceNode(subgraph, names, type = "INT") {
    const accepted = new Set(names.map((name) => String(name).trim().toLowerCase()));
    return getSubgraphNodes(subgraph).find((node) => {
      const title = String(node?.title || node?.name || node?.label || "").trim().toLowerCase();
      const nodeType = String(node?.type || node?.comfyClass || "").trim().toLowerCase();
      const outputs = Array.isArray(node?.outputs) ? node.outputs : [];
      const hasTypedOutput = outputs.some((output) => String(output?.type || "").toUpperCase() === type);
      const outputNameMatches = outputs.some((output) => accepted.has(String(output?.name || output?.label || "").trim().toLowerCase()));
      return hasTypedOutput && (accepted.has(title) || accepted.has(nodeType) || outputNameMatches);
    }) || null;
  }

  function findFirstOutputSlot(node, type = "INT") {
    return (node?.outputs || []).findIndex((output) => String(output?.type || "").toUpperCase() === type);
  }

  function ensureSubgraphNodeLink(subgraph, sourceNode, sourceSlot, targetNode, targetInputName, type) {
    const targetSlot = (targetNode?.inputs || []).findIndex((input) => input?.name === targetInputName);
    if (!sourceNode || sourceSlot < 0 || targetSlot < 0) return false;

    if (typeof sourceNode.connect === "function") {
      const oldLinkId = targetNode.inputs?.[targetSlot]?.link;
      const result = sourceNode.connect(sourceSlot, targetNode, targetSlot);
      const newLinkId = targetNode.inputs?.[targetSlot]?.link;
      if (newLinkId !== null && newLinkId !== undefined) {
        return Number(oldLinkId) !== Number(newLinkId);
      }
      if (result) return true;
    }
    return false;
  }

  function findSourceFromInputLink(graph, targetNode, targetInputName) {
    const targetInput = (targetNode?.inputs || []).find((input) => input?.name === targetInputName);
    const linkId = targetInput?.link;
    if (linkId === null || linkId === undefined) return null;

    const nodes = getSubgraphNodes(graph);
    for (const node of nodes) {
      const outputs = Array.isArray(node?.outputs) ? node.outputs : [];
      for (let slot = 0; slot < outputs.length; slot++) {
        const links = outputs[slot]?.links;
        if (Array.isArray(links) && links.some((id) => Number(id) === Number(linkId))) {
          return { node, slot };
        }
      }
    }

    return null;
  }

  function ensureAdvancedInputSocket(node, name, type) {
    const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
    if (inputs.some((input) => input?.name === name)) return true;

    if (typeof node?.addInput === "function") {
      node.addInput(name, type);
      return true;
    }

    if (node) {
      node.inputs = inputs;
      node.inputs.push({
        localized_name: name,
        name,
        type,
        widget: { name },
        link: null,
      });
      return true;
    }

    return false;
  }

  function autoWireAdvancedTopGraphIndexSeed(graph, node, count) {
    if (!graph || !node) return 0;

    const indexSource = findSourceFromInputLink(graph, node, "index_1");
    const seedSource = findSourceFromInputLink(graph, node, "seed_1");
    let linked = 0;

    for (let i = 1; i <= count; i++) {
      ensureAdvancedInputSocket(node, `index_${i}`, "INT");
      ensureAdvancedInputSocket(node, `seed_${i}`, "INT");
      if (indexSource && ensureSubgraphNodeLink(graph, indexSource.node, indexSource.slot, node, `index_${i}`, "INT")) linked += 1;
      if (seedSource && ensureSubgraphNodeLink(graph, seedSource.node, seedSource.slot, node, `seed_${i}`, "INT")) linked += 1;
    }

    node?.setSize?.([node.size?.[0] || 430, node.computeSize?.()[1] || node.size?.[1] || 440]);
    app.graph?.setDirtyCanvas?.(true, true);
    return linked;
  }

  function autoWireAdvancedIndexSeed(subgraph, node, count, missing) {
    if (!subgraph || !node) {
      missing.push("node 830 inner subgraph wiring");
      return 0;
    }

    const indexNode = findSubgraphSourceNode(subgraph, ["index", "get_index"], "INT");
    const seedNode = findSubgraphSourceNode(subgraph, ["random seed", "seed", "random_seed"], "INT");
    const indexNodeSlot = findFirstOutputSlot(indexNode, "INT");
    const seedNodeSlot = findFirstOutputSlot(seedNode, "INT");
    let linked = 0;

    if (!indexNode) missing.push("inner index INT node");
    if (!seedNode) missing.push("inner random seed INT node");

    for (let i = 1; i <= count; i++) {
      ensureAdvancedInputSocket(node, `index_${i}`, "INT");
      ensureAdvancedInputSocket(node, `seed_${i}`, "INT");
      if (indexNode && ensureSubgraphNodeLink(subgraph, indexNode, indexNodeSlot, node, `index_${i}`, "INT")) linked += 1;
      if (seedNode && ensureSubgraphNodeLink(subgraph, seedNode, seedNodeSlot, node, `seed_${i}`, "INT")) linked += 1;
    }

    node?.setSize?.([node.size?.[0] || 430, node.computeSize?.()[1] || node.size?.[1] || 440]);
    app.graph?.setDirtyCanvas?.(true, true);
    return linked;
  }

  function getAdvancedCount() {
    if (!controls.advanced.enabled.checked) return 0;
    const rawCount = Number(controls.advanced.count.value || 1);
    return Math.max(1, Math.min(PART2_ADVANCED_MAX_PICKERS, Number.isFinite(rawCount) ? Math.round(rawCount) : 1));
  }

  function countAdvancedItems(text) {
    return String(text || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean).length;
  }

  function updateAdvancedPickerHeader(index) {
    const picker = controls.advanced.pickers[index - 1];
    if (!picker) return;
    const label = String(picker.label.value || picker.preset.value || `Setting ${index}`).trim();
    const itemCount = countAdvancedItems(picker.items.value);
    const mode = String(picker.selectionMode.value || "index");
    picker.header.innerHTML = "";
    const title = document.createElement("span");
    title.textContent = `${index}. ${label || `Setting ${index}`} · ${mode}`;
    const meta = document.createElement("span");
    meta.textContent = `${itemCount} items`;
    meta.style.cssText = "color: #94a3b8; font-weight: 600; white-space: nowrap;";
    picker.header.append(title, meta);
  }

  function updateAdvancedVisibility() {
    const count = getAdvancedCount();
    controls.advanced.count.value = String(count);
    controls.advanced.count.disabled = count <= 0;
    controls.advanced.modeAll.disabled = count <= 0;
    advancedSummary.textContent = count <= 0
      ? "Advanced prompt details are off"
      : `${count} active prompt detail ${count === 1 ? "list" : "lists"}`;

    for (let i = 1; i <= PART2_ADVANCED_MAX_PICKERS; i++) {
      const picker = controls.advanced.pickers[i - 1];
      const visible = i <= count;
      picker.wrapper.style.display = visible ? "block" : "none";
      picker.body.style.display = visible && !picker.collapsed ? "block" : "none";
      updateAdvancedPickerHeader(i);
    }
  }

  function applyAdvancedModeToActivePickers() {
    const count = getAdvancedCount();
    const mode = String(controls.advanced.modeAll.value || "index");
    for (let i = 1; i <= count; i++) {
      const picker = controls.advanced.pickers[i - 1];
      setSelectValueAllowingDraft(picker.selectionMode, mode);
      updateAdvancedPickerHeader(i);
    }
    savePart2Draft();
  }

  function applyAdvancedPreset(index) {
    const picker = controls.advanced.pickers[index - 1];
    if (!picker) return;
    const preset = String(picker.preset.value || "Custom");
    if (preset !== "Custom") {
      picker.label.value = preset;
      picker.items.value = PART2_ADVANCED_PRESET_ITEMS[preset] || "";
    }
    updateAdvancedPickerHeader(index);
    savePart2Draft();
  }

  function getAdvancedPresetForApply(picker) {
    const preset = String(picker?.preset?.value || "Custom").trim();
    return PART2_ADVANCED_PRESETS.includes(preset) ? preset : "Custom";
  }

  function getAdvancedLabelForApply(picker, preset, index) {
    const label = String(picker?.label?.value || "").trim();
    if (label) return label;
    if (preset && preset !== "Custom") return preset;
    return index === 1 ? "Camera Motion" : "";
  }

  function getAdvancedPickerDirectApplyValues(index) {
    const picker = controls.advanced.pickers[index - 1];
    const preset = getAdvancedPresetForApply(picker);
    const label = getAdvancedLabelForApply(picker, preset, index);
    const pickCount = Math.max(1, Math.min(50, Number.isFinite(Number(picker.pickCount.value)) ? Math.round(Number(picker.pickCount.value)) : 1));
    return {
      preset,
      items: stripAdvancedItemDirectives(picker.items.value),
      label,
      selection_mode: picker.selectionMode.value || "index",
      two_item_template: picker.template.value || "start with {item1} then follow with {item2}",
      pick_count: pickCount,
    };
  }

  function applyEasyAdvancedPickerState(node, count) {
    if (!node) return 0;
    let updated = 0;
    if (setEasyAdvancedWidgetValue(node, "picker_count", count)) updated += 1;
    if (setEasyAdvancedWidgetValue(node, "joiner", "newline")) updated += 1;

    for (let i = 1; i <= PART2_ADVANCED_MAX_PICKERS; i++) {
      const values = i <= count ? getAdvancedPickerDirectApplyValues(i) : {
        preset: i === 1 ? "Camera Motion" : "Custom",
        items: "",
        label: "",
        selection_mode: "index",
        two_item_template: "start with {item1} then follow with {item2}",
        pick_count: 1,
      };

      if (setEasyAdvancedWidgetValue(node, `preset_${i}`, values.preset)) updated += 1;
      if (setEasyAdvancedWidgetValue(node, `label_${i}`, values.label)) updated += 1;
      if (setEasyAdvancedWidgetValue(node, `selection_mode_${i}`, values.selection_mode)) updated += 1;
      if (setEasyAdvancedWidgetValue(node, `items_${i}`, values.items)) updated += 1;
      if (setEasyAdvancedWidgetValue(node, `pick_count_${i}`, values.pick_count)) updated += 1;
      if (setEasyAdvancedWidgetValue(node, `two_item_template_${i}`, values.two_item_template)) updated += 1;
    }

    node.setSize?.([Math.max(780, node.size?.[0] || 780), node.computeSize?.()[1] || node.size?.[1] || 440]);
    app.graph?.setDirtyCanvas?.(true, true);
    return updated;
  }

  function getAdvancedItemDirectives(items) {
    const directives = {};
    const lines = String(items || "").split(/\r?\n/);
    for (const line of lines) {
      const match = /^#\s*(?:VRGDG_)?(LABEL|SELECTION_MODE|PICK_COUNT|TEMPLATE):\s*(.*?)\s*$/.exec(line);
      if (!match) break;
      directives[match[1].toLowerCase()] = match[2];
    }
    return directives;
  }

  function withAdvancedItemDirectives(items, values) {
    const cleanItems = stripAdvancedItemDirectives(items);
    const cleanLabel = String(values?.label || "").trim();
    const selectionMode = String(values?.selectionMode || "index").trim();
    const pickCount = Math.max(1, Math.min(50, Number.isFinite(Number(values?.pickCount)) ? Math.round(Number(values.pickCount)) : 1));
    const template = String(values?.template || "").trim();
    const directives = [];
    if (cleanLabel) directives.push(`# VRGDG_LABEL: ${cleanLabel}`);
    if (selectionMode) directives.push(`# VRGDG_SELECTION_MODE: ${selectionMode}`);
    directives.push(`# VRGDG_PICK_COUNT: ${pickCount}`);
    if (template) directives.push(`# VRGDG_TEMPLATE: ${template}`);
    return `${directives.join("\n")}\n${cleanItems}`;
  }

  function withAdvancedLabelDirective(items, label) {
    const cleanItems = stripAdvancedItemDirectives(items);
    const cleanLabel = String(label || "").trim();
    return cleanLabel ? `# VRGDG_LABEL: ${cleanLabel}\n${cleanItems}` : cleanItems;
  }

  function stripAdvancedItemDirectives(items) {
    return String(items || "").replace(/^(#\s*(?:VRGDG_)?(?:LABEL|SELECTION_MODE|PICK_COUNT|TEMPLATE):.*(?:\r?\n|$))+/i, "");
  }

  function stripAdvancedLabelDirective(items) {
    return stripAdvancedItemDirectives(items);
  }

  function collectAdvancedDraft() {
    return {
      enabled: controls.advanced.enabled.checked,
      count: controls.advanced.count.value,
      modeAll: controls.advanced.modeAll.value,
      pickers: controls.advanced.pickers.map((picker) => ({
        preset: picker.preset.value,
        label: picker.label.value,
        selectionMode: picker.selectionMode.value,
        index: picker.index.value,
        seed: picker.seed.value,
        pickCount: picker.pickCount.value,
        template: picker.template.value,
        items: picker.items.value,
        collapsed: picker.collapsed,
      })),
    };
  }

  function applyAdvancedDraft(draft) {
    if (!draft || typeof draft !== "object") return;
    if (draft.enabled !== undefined) controls.advanced.enabled.checked = Boolean(draft.enabled);
    if (draft.count !== undefined) controls.advanced.count.value = String(draft.count);
    if (draft.modeAll !== undefined) setSelectValueAllowingDraft(controls.advanced.modeAll, draft.modeAll);
    for (let i = 0; i < controls.advanced.pickers.length; i++) {
      const pickerDraft = draft.pickers?.[i] || {};
      const picker = controls.advanced.pickers[i];
      if (pickerDraft.preset !== undefined) setSelectValueAllowingDraft(picker.preset, pickerDraft.preset);
      if (pickerDraft.label !== undefined) picker.label.value = String(pickerDraft.label);
      if (picker.preset.value && picker.preset.value !== "Custom") picker.label.value = picker.preset.value;
      if (pickerDraft.selectionMode !== undefined) setSelectValueAllowingDraft(picker.selectionMode, pickerDraft.selectionMode);
      if (pickerDraft.index !== undefined) picker.index.value = String(pickerDraft.index);
      if (pickerDraft.seed !== undefined) picker.seed.value = String(pickerDraft.seed);
      if (pickerDraft.pickCount !== undefined) picker.pickCount.value = String(pickerDraft.pickCount);
      if (pickerDraft.template !== undefined) picker.template.value = String(pickerDraft.template);
      if (pickerDraft.items !== undefined) picker.items.value = String(pickerDraft.items);
      if (pickerDraft.collapsed !== undefined) picker.collapsed = Boolean(pickerDraft.collapsed);
    }
    updateAdvancedVisibility();
  }

  function refreshAdvancedControls(missing) {
    const node = getPart2AdvancedPickerContexts({ ownerNode: controls.workflowNode, singleTarget: true, workflowKind: controls.workflowKind })[0]?.node || null;
    if (!node) {
      missing.push("VRGDG_EasyMultiCyclingTextPicker or VRGDG_MultiCyclingTextPicker");
      return;
    }

    const savedCount = Math.max(0, Math.min(PART2_ADVANCED_MAX_PICKERS, Number(getAdvancedWidgetValue(node, "picker_count", 0)) || 0));
    controls.advanced.enabled.checked = savedCount > 0;
    controls.advanced.count.value = String(savedCount);
    if (getAdvancedWidgetFallbackIndex("picker_count") < 0) missing.push("node 830 inner picker_count");
    const activeCount = getAdvancedCount();

    for (let i = 1; i <= PART2_ADVANCED_MAX_PICKERS; i++) {
      const picker = controls.advanced.pickers[i - 1];
      const presetWidget = getAdvancedWidget(node, `preset_${i}`);
      const labelWidget = getAdvancedWidget(node, `label_${i}`);
      const modeWidget = getAdvancedWidget(node, `selection_mode_${i}`);
      const requiredWidgets = [
        [`preset_${i}`, presetWidget],
        [`items_${i}`, getAdvancedWidget(node, `items_${i}`)],
        [`label_${i}`, labelWidget],
        [`selection_mode_${i}`, modeWidget],
        [`pick_count_${i}`, getAdvancedWidget(node, `pick_count_${i}`)],
        [`two_item_template_${i}`, getAdvancedWidget(node, `two_item_template_${i}`)],
      ];

      fillSelectOptions(picker.preset, getPart2WidgetOptions(presetWidget).length ? getPart2WidgetOptions(presetWidget) : PART2_ADVANCED_PRESETS, isEasyAdvancedPickerNode(node) ? getEasyAdvancedWidgetValue(node, `preset_${i}`, i === 1 ? "Camera Motion" : "Custom") : getAdvancedWidgetValue(node, `preset_${i}`, i === 1 ? "Camera Motion" : "Custom"));
      const rawItems = isEasyAdvancedPickerNode(node) ? getEasyAdvancedWidgetValue(node, `items_${i}`, "") : getAdvancedWidgetValue(node, `items_${i}`, "");
      const preset = String(picker.preset.value || "Custom");
      const itemDirectives = getAdvancedItemDirectives(rawItems);
      picker.label.value = isEasyAdvancedPickerNode(node)
        ? String(getEasyAdvancedWidgetValue(node, `label_${i}`, preset !== "Custom" ? preset : (i === 1 ? "Camera Motion" : "")))
        : preset !== "Custom"
          ? preset
          : String(getAdvancedWidgetValue(node, `label_${i}`, itemDirectives.label || (i === 1 ? "Camera Motion" : "")));
      fillSelectOptions(picker.selectionMode, getPart2WidgetOptions(modeWidget).length ? getPart2WidgetOptions(modeWidget) : PART2_ADVANCED_SELECTION_MODES, isEasyAdvancedPickerNode(node) ? getEasyAdvancedWidgetValue(node, `selection_mode_${i}`, "index") : itemDirectives.selection_mode || getAdvancedWidgetValue(node, `selection_mode_${i}`, "index"));
      picker.index.value = String(isEasyAdvancedPickerNode(node) ? 0 : getAdvancedWidgetValue(node, `index_${i}`, 0));
      picker.seed.value = String(isEasyAdvancedPickerNode(node) ? 0 : getAdvancedWidgetValue(node, `seed_${i}`, 0));
      picker.pickCount.value = String(isEasyAdvancedPickerNode(node) ? getEasyAdvancedWidgetValue(node, `pick_count_${i}`, 1) : itemDirectives.pick_count || getAdvancedWidgetValue(node, `pick_count_${i}`, 1));
      picker.template.value = String(isEasyAdvancedPickerNode(node) ? getEasyAdvancedWidgetValue(node, `two_item_template_${i}`, "start with {item1} then follow with {item2}") : itemDirectives.template || getAdvancedWidgetValue(node, `two_item_template_${i}`, "start with {item1} then follow with {item2}"));
      picker.items.value = isEasyAdvancedPickerNode(node) ? String(rawItems || "") : stripAdvancedItemDirectives(rawItems);

      if (i <= activeCount) {
        for (const [name, widget] of requiredWidgets) {
          if (!widget && getAdvancedWidgetFallbackIndex(name) < 0) missing.push(`node 830 inner ${name}`);
        }
      }
    }

    controls.advanced.modeAll.value = controls.advanced.pickers[0]?.selectionMode.value || "index";
    updateAdvancedVisibility();
  }

  function applyAdvancedControls(missing) {
    const contexts = getPart2AdvancedPickerContexts({ ownerNode: controls.workflowNode, singleTarget: true, workflowKind: controls.workflowKind });
    if (!contexts.length) {
      missing.push("VRGDG_EasyMultiCyclingTextPicker or VRGDG_MultiCyclingTextPicker");
      return 0;
    }

    let updated = 0;
    const count = getAdvancedCount();

    for (const context of contexts) {
      const node = context.node;

      if (isEasyAdvancedPickerNode(node)) {
        updated += applyEasyAdvancedPickerState(node, count);
        continue;
      }

      if (setAdvancedWidgetValue(node, "picker_count", count)) updated += 1;
      else missing.push(`node ${node?.id || 830} inner picker_count`);

      if (count <= 0) continue;

      for (let i = 1; i <= count; i++) {
        const picker = controls.advanced.pickers[i - 1];
        const preset = getAdvancedPresetForApply(picker);
        const label = getAdvancedLabelForApply(picker, preset, i);
        const values = {
          [`preset_${i}`]: preset,
          [`items_${i}`]: withAdvancedItemDirectives(picker.items.value, {
            label,
            selectionMode: picker.selectionMode.value || "index",
            pickCount: picker.pickCount.value,
            template: picker.template.value,
          }),
          [`label_${i}`]: label,
          [`selection_mode_${i}`]: picker.selectionMode.value,
          [`pick_count_${i}`]: Number.isFinite(Number(picker.pickCount.value)) ? Number(picker.pickCount.value) : 1,
          [`two_item_template_${i}`]: picker.template.value,
        };

        for (const [name, value] of Object.entries(values)) {
          if (setAdvancedWidgetValue(node, name, value)) updated += 1;
          else missing.push(`node ${node?.id || 830} inner ${name}`);
          if (i === 1) syncAdvancedOuterProxyValue(name, value);
        }
      }

      updated += context.isTopGraph
        ? autoWireAdvancedTopGraphIndexSeed(context.subgraph, node, count)
        : autoWireAdvancedIndexSeed(context.subgraph, node, count, missing);
    }

    syncAdvancedOuterProxyValue("picker_count", count);
    updateAdvancedVisibility();
    return updated;
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

  function getLoraSelectFallbackOptions(slotIndex = 0) {
    const ltxSlotOptions = Array.from(controls.lora.slots[slotIndex]?.select?.options || []).map((option) => option.value);
    if (ltxSlotOptions.length) return ltxSlotOptions;

    const loraNode = getPart2OptionalLoraNode();
    for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
      const widget = getPart2Widget(loraNode, `lora_${i}`);
      const options = getPart2WidgetOptions(widget);
      if (options.length) return options;
    }

    return ["[none]"];
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

  function refreshTriggerControls(kind, missing) {
    const trigger = kind === "zImage" ? controls.zImageLora.trigger : controls.lora.trigger;
    if (!trigger) return;

    if (kind === "zImage" && controls.workflowKind === "part3") {
      trigger.useWrapper.style.display = "none";
      trigger.wordWrapper.style.display = "none";
      return;
    }

    const nodeId = getTriggerNodeId(kind, controls.workflowKind);
    const node = findPart2NodeDeep(nodeId);
    if (!node) {
      trigger.useTrigger.checked = false;
      trigger.triggerWord.value = "";
      missing.push(`${kind === "zImage" ? "Z-Image" : "LTX"} trigger subgraph ${nodeId}`);
      return;
    }

    trigger.useTrigger.checked = String(getPart2WidgetValue(nodeId, null, 0) ?? false).toLowerCase() === "true";
    trigger.triggerWord.value = String(getPart2WidgetValue(nodeId, null, 1) ?? "");
  }

  function applyTriggerControls(kind, missing) {
    if (kind === "zImage" && controls.workflowKind === "part3") return 0;

    const trigger = kind === "zImage" ? controls.zImageLora.trigger : controls.lora.trigger;
    const nodeId = getTriggerNodeId(kind, controls.workflowKind);
    const node = findPart2NodeDeep(nodeId);
    const label = kind === "zImage" ? "Z-Image trigger" : "LTX trigger";
    if (!node) {
      missing.push(`${label} subgraph ${nodeId}`);
      return 0;
    }

    let updated = 0;
    const enabled = Boolean(trigger?.useTrigger?.checked);
    if (setPart2WidgetValue(nodeId, null, enabled, 0)) updated += 1;
    else missing.push(`${label} boolean switch`);

    const word = trigger.triggerWord.value || "";
    if (enabled) {
      if (setPart2WidgetValue(nodeId, null, word, 1)) updated += 1;
      else missing.push(`${label} string_1`);
    } else {
      setPart2WidgetValue(nodeId, null, "", 1);
    }

    return updated;
  }

  function refreshZImageLoraControls(missing) {
    if (controls.workflowKind === "part3") {
      controls.zImageLora.section.style.display = "none";
      return;
    }

    const loraNode = getPart2ZImageOptionalLoraNode();
    if (!loraNode) {
      controls.zImageLora.section.style.display = "none";
      missing.push("Z-Image subgraph optional LoRA node 847");
      return;
    }

    controls.zImageLora.section.style.display = "block";
    controls.zImageLora.useCustom.value = getPart2ZImageUseLoraProxyValue() ? "true" : "false";
    controls.zImageLora.count.value = String(getNodeWidgetValueFlexible(loraNode, "lora_count", 0));

    for (let i = 1; i <= PART2_MAX_LORA_SLOTS; i++) {
      const slot = controls.zImageLora.slots[i - 1];
      const loraWidget = getPart2Widget(loraNode, `lora_${i}`, getLoraWidgetFallbackIndex(`lora_${i}`));
      const currentLora = getNodeWidgetValueFlexible(loraNode, `lora_${i}`, "[none]") || "[none]";
      const loraOptions = getPart2WidgetOptions(loraWidget);
      fillSelectOptions(slot.select, loraOptions.length ? loraOptions : getLoraSelectFallbackOptions(i - 1), currentLora);
      slot.strength.value = String(getNodeWidgetValueFlexible(loraNode, `strength_${i}`, 1));
    }

    updateZImageLoraVisibility();
  }

  function applyZImageLoraControls(missing) {
    if (controls.workflowKind === "part3") {
      return 0;
    }

    const loraNode = getPart2ZImageOptionalLoraNode();
    if (!loraNode) {
      missing.push("Z-Image subgraph optional LoRA node 847");
      return 0;
    }

    let updated = 0;
    const useLoras = String(controls.zImageLora.useCustom.value).toLowerCase() === "true";
    const rawCount = Number(controls.zImageLora.count.value || 0);
    const count = Math.max(0, Math.min(PART2_MAX_LORA_SLOTS, Number.isFinite(rawCount) ? Math.round(rawCount) : 0));

    if (setPart2NodeWidgetValue(loraNode, "use_custom_loras", useLoras)) updated += 1;
    else missing.push("Z-Image LoRA use_custom_loras");
    if (setPart2WidgetValue(PART2_NODE_IDS.zImageModels, "use_custom_loras", useLoras, 6)) updated += 1;
    else missing.push("node 797.use_custom_loras");
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

  function stripJsonFence(text) {
    return String(text || "")
      .replace(/^\s*```(?:json)?\s*/i, "")
      .replace(/\s*```\s*$/i, "")
      .trim();
  }

  function extractPromptListForAdvancedGemma() {
    const text = stripJsonFence(controls.promptJson.value);
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        return parsed.map((item) => typeof item === "string" ? item : JSON.stringify(item)).map((item) => item.trim()).filter(Boolean);
      }
      if (parsed && typeof parsed === "object") {
        return Object.entries(parsed)
          .sort(([a], [b]) => {
            const an = Number(String(a).match(/\d+/)?.[0] || Number.MAX_SAFE_INTEGER);
            const bn = Number(String(b).match(/\d+/)?.[0] || Number.MAX_SAFE_INTEGER);
            return an - bn || String(a).localeCompare(String(b));
          })
          .map(([, value]) => typeof value === "string" ? value : JSON.stringify(value))
          .map((item) => item.trim())
          .filter(Boolean);
      }
    } catch {
      // Fall through to line mode.
    }
    return text.split(/\r?\n+/).map((line) => line.trim()).filter(Boolean);
  }

  function cleanGemmaAdvancedList(text, expectedCount) {
    const lines = String(text || "")
      .replace(/^\s*```(?:text)?\s*/i, "")
      .replace(/\s*```\s*$/i, "")
      .split(/\r?\n/)
      .map((line) => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, "").trim())
      .filter(Boolean);
    return lines.slice(0, expectedCount).join("\n");
  }

  async function runGemma4ForAdvancedLists() {
    const modelFile = String(controls.modelSelects.llm?.value || "").trim();
    if (!modelFile) {
      setStatus("Choose a SuperGemma model in the Part 2 LLM dropdown first.", true);
      return;
    }
    const prompts = extractPromptListForAdvancedGemma();
    if (!prompts.length) {
      setStatus("Paste or load Prompt JSON from Part 1 before using Gemma4 advanced lists.", true);
      return;
    }
    if (!controls.advanced.enabled.checked) {
      controls.advanced.enabled.checked = true;
    }
    const count = getAdvancedCount();
    if (count <= 0) {
      setStatus("Set Advanced Settings Count to at least 1 before using Gemma4.", true);
      return;
    }

    const extraNotes = await requestGemma4AdvancedListNotes();
    if (extraNotes === null) {
      setStatus("Gemma4 advanced list creation was cancelled.");
      return;
    }

    const progress = showGemma4Progress(`Gemma4 is creating ${count} advanced list${count === 1 ? "" : "s"}...\nUsing ${prompts.length} prompts.`, "Gemma4");
    advancedGemmaButton.disabled = true;
    setStatus(`Gemma4 is creating ${count} advanced list${count === 1 ? "" : "s"}...`);
    try {
      for (let i = 1; i <= count; i++) {
        const picker = controls.advanced.pickers[i - 1];
        if (!picker) continue;
        const label = String(picker.label.value || picker.preset.value || `Setting ${i}`).trim();
        progress.__vrgdgSetMessage?.(`Gemma4 is creating list ${i} of ${count}: ${label}\nUsing ${prompts.length} prompts.`);
        const data = await generateGemma4Text({
          target: "advanced_prompt_detail",
          model_file: modelFile,
          label,
          prompts,
          notes: extraNotes,
          n_ctx: 13000,
          max_new_tokens: Math.max(1024, prompts.length * 80),
          unload_after: i === count,
        });
        const listText = cleanGemmaAdvancedList(data.text, prompts.length);
        if (!listText) throw new Error(`Gemma4 returned an empty list for ${label}.`);
        picker.items.value = listText;
        picker.selectionMode.value = "index";
        updateAdvancedPickerHeader(i);
      }
      updateAdvancedVisibility();
      savePart2Draft();
      progress.__vrgdgSetMessage?.("Done. Gemma4 filled the advanced lists and unloaded the model.");
      hideGemma4Progress();
      setStatus(`Gemma4 filled ${count} advanced list${count === 1 ? "" : "s"}. Selection mode was set to Index-based.`);
    } catch (error) {
      hideGemma4Progress();
      setStatus(String(error?.message || error), true);
      await showGemma4ResultDialog({
        title: "Gemma4 Advanced Lists Failed",
        text: String(error?.message || error),
        isError: true,
      });
    } finally {
      advancedGemmaButton.disabled = false;
    }
  }

  function refreshPart2Controls() {
    const missing = [];
    suppressDraftSave = true;
    const nodeIds = getWorkflowNodeIds();
    updateWorkflowSpecificVisibility();

    for (const field of PART2_MODEL_FIELDS) {
      if (!isModelFieldVisible(field)) continue;
      const node = getPart2Node(field.nodeId);
      const widget = getPart2Widget(node, field.key);
      const current = getPart2WidgetValue(field.nodeId, field.key);
      fillSelectOptions(controls.modelSelects[`${field.nodeId}:${field.key}`], getPart2WidgetOptions(widget), current);
      if (!node || !widget) missing.push(`node ${field.nodeId}.${field.key}`);
    }

    const llmNode = getPart2Node(nodeIds.llmI2V);
    const llmWidget = getPart2Widget(llmNode, null, 0);
    fillSelectOptions(controls.modelSelects.llm, getPart2WidgetOptions(llmWidget), getPart2WidgetValue(nodeIds.llmI2V, null, 0));
    if (!llmNode || !llmWidget) missing.push(`node ${nodeIds.llmI2V} LLM model`);

    for (const field of PART2_SETTING_FIELDS) {
      controls.settings[field.key].value = String(getPart2WidgetValue(nodeIds.settings, field.key) ?? "");
      if (!getPart2Widget(getPart2Node(nodeIds.settings), field.key)) missing.push(`node ${nodeIds.settings}.${field.key}`);
    }

    controls.useSrt.value = String(getPart2WidgetValue(nodeIds.useSrtSwitch, "switch", 0)).toLowerCase() === "false" ? "false" : "true";
    controls.promptJson.value = String(getPart2WidgetValue(nodeIds.promptJson, null, 0) || "");
    updateFixedDurationVisibility();
    refreshAdvancedControls(missing);
    refreshLoraControls(missing);
    refreshTriggerControls("ltx", missing);
    refreshZImageLoraControls(missing);
    refreshTriggerControls("zImage", missing);
    updateLoraVisibility();
    updateZImageLoraVisibility();
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
    const nodeIds = getWorkflowNodeIds();

    for (const field of PART2_MODEL_FIELDS) {
      if (!isModelFieldVisible(field)) continue;
      if (setPart2WidgetValue(field.nodeId, field.key, controls.modelSelects[`${field.nodeId}:${field.key}`].value)) updated += 1;
      else missing.push(`node ${field.nodeId}.${field.key}`);
    }

    const llmValue = controls.modelSelects.llm.value;
    if (setPart2WidgetValue(nodeIds.llmI2V, null, llmValue, 0)) updated += 1;
    else missing.push(`node ${nodeIds.llmI2V} LLM model`);
    if (nodeIds.llmT2I) {
      if (setPart2WidgetValue(nodeIds.llmT2I, null, llmValue, 0)) updated += 1;
      else missing.push(`node ${nodeIds.llmT2I} LLM model`);
    }

    for (const field of PART2_SETTING_FIELDS) {
      const rawValue = controls.settings[field.key].value;
      const numberValue = Number(rawValue);
      const value = Number.isFinite(numberValue) ? numberValue : rawValue;
      if (setPart2WidgetValue(nodeIds.settings, field.key, value)) updated += 1;
      else missing.push(`node ${nodeIds.settings}.${field.key}`);
    }

    const useSrt = String(controls.useSrt.value).toLowerCase() !== "false";
    if (setPart2WidgetValue(nodeIds.useSrtSwitch, "switch", useSrt, 0)) updated += 1;
    else missing.push(`node ${nodeIds.useSrtSwitch}.switch`);

    updated += applyAdvancedControls(missing);

    if (setPart2WidgetValue(nodeIds.promptJson, null, controls.promptJson.value, 0)) updated += 1;
    else missing.push(`node ${nodeIds.promptJson} prompt JSON`);

    updated += applyLoraControls(missing);
    updated += applyTriggerControls("ltx", missing);
    updated += applyZImageLoraControls(missing);
    updated += applyTriggerControls("zImage", missing);

    updateFixedDurationVisibility();
    updateLoraVisibility();
    updateZImageLoraVisibility();
    clearPart2Draft();
    setStatus(missing.length ? `Updated ${updated} settings.\nMissing:\n${missing.join("\n")}` : `Updated ${updated} Part 2 settings.`);
  }

  function applyAdvancedOnlySettings() {
    const missing = [];
    const contexts = getPart2AdvancedPickerContexts({ ownerNode: controls.workflowNode, singleTarget: true, workflowKind: controls.workflowKind });
    const targetId = contexts[0]?.node?.id;
    const updated = applyAdvancedControls(missing);
    savePart2Draft();
    setStatus(
      missing.length
        ? `Updated ${updated} advanced settings${targetId ? ` on picker #${targetId}` : ""}.\nMissing:\n${missing.join("\n")}`
        : `Updated ${updated} advanced settings${targetId ? ` on picker #${targetId}` : ""}.`
    );
  }

  closeButton.addEventListener("click", closeModal);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeModal();
  });
  controls.useSrt.addEventListener("change", updateFixedDurationVisibility);
  controls.lora.useCustom.addEventListener("change", updateLoraVisibility);
  controls.lora.count.addEventListener("input", updateLoraVisibility);
  controls.lora.count.addEventListener("change", updateLoraVisibility);
  controls.lora.trigger.useTrigger.addEventListener("change", updateLoraVisibility);
  controls.zImageLora.useCustom.addEventListener("change", updateZImageLoraVisibility);
  controls.zImageLora.count.addEventListener("input", updateZImageLoraVisibility);
  controls.zImageLora.count.addEventListener("change", updateZImageLoraVisibility);
  controls.zImageLora.trigger.useTrigger.addEventListener("change", updateZImageLoraVisibility);
  pasteFromStep1Button.addEventListener("click", pastePromptJsonFromStep1);
  advancedGemmaButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    runGemma4ForAdvancedLists();
  });
  controls.advanced.enabled.addEventListener("change", () => {
    if (controls.advanced.enabled.checked && Number(controls.advanced.count.value || 0) <= 0) {
      controls.advanced.count.value = "1";
    }
    updateAdvancedVisibility();
    savePart2Draft();
  });
  controls.advanced.count.addEventListener("input", updateAdvancedVisibility);
  controls.advanced.count.addEventListener("change", updateAdvancedVisibility);
  controls.advanced.modeAll.addEventListener("change", applyAdvancedModeToActivePickers);

  for (let i = 1; i <= controls.advanced.pickers.length; i++) {
    const picker = controls.advanced.pickers[i - 1];
    picker.header.addEventListener("click", () => {
      picker.collapsed = !picker.collapsed;
      updateAdvancedVisibility();
      savePart2Draft();
    });
    picker.preset.addEventListener("change", () => applyAdvancedPreset(i));
    for (const control of [picker.label, picker.selectionMode, picker.index, picker.seed, picker.pickCount, picker.template, picker.items]) {
      control.addEventListener("input", () => updateAdvancedPickerHeader(i));
      control.addEventListener("change", () => updateAdvancedPickerHeader(i));
    }
  }

  const draftControls = [
    ...Object.values(controls.modelSelects),
    ...Object.values(controls.settings),
    controls.useSrt,
    controls.advanced.enabled,
    controls.advanced.count,
    controls.advanced.modeAll,
    controls.promptJson,
    controls.lora.useCustom,
    controls.lora.count,
    controls.lora.twoPass,
    controls.lora.trigger.useTrigger,
    controls.lora.trigger.triggerWord,
    controls.zImageLora.useCustom,
    controls.zImageLora.count,
    controls.zImageLora.trigger.useTrigger,
    controls.zImageLora.trigger.triggerWord,
  ];
  for (const slot of controls.lora.slots) {
    draftControls.push(slot.select, slot.strength);
  }
  for (const slot of controls.zImageLora.slots) {
    draftControls.push(slot.select, slot.strength);
  }
  for (const picker of controls.advanced.pickers) {
    draftControls.push(
      picker.preset,
      picker.label,
      picker.selectionMode,
      picker.index,
      picker.seed,
      picker.pickCount,
      picker.template,
      picker.items
    );
  }
  for (const control of draftControls) {
    control?.addEventListener?.("input", savePart2Draft);
    control?.addEventListener?.("change", savePart2Draft);
  }

  applyButton.addEventListener("click", applyPart2Settings);
  topApplyButton.addEventListener("click", applyPart2Settings);
  advancedApplyButton.addEventListener("click", applyAdvancedOnlySettings);

  overlay.__vrgdgOpenPart2 = (workflowKind = "part2", workflowNode = null) => {
    controls.workflowKind = workflowKind === "part3" ? "part3" : "part2";
    controls.workflowNode = workflowNode;
    title.textContent = controls.workflowKind === "part3" ? "Workflow 3 Controls" : "Part 2 Workflow Controls";
    subtitle.textContent = controls.workflowKind === "part3"
      ? "Control model pickers, render settings, SRT/fixed timing, prompt detail lists, and copied prompt JSON."
      : "Control model pickers, render settings, SRT/fixed timing, camera motions, and copied prompt JSON.";
    topApplyButton.textContent = controls.workflowKind === "part3" ? "Apply Workflow 3 Settings" : "Apply Part 2 Settings";
    applyButton.textContent = controls.workflowKind === "part3" ? "Apply Workflow 3 Settings" : "Apply Part 2 Settings";
    updateWorkflowSpecificVisibility();
    overlay.style.display = "flex";
    refreshPart2Controls();
  };

  return overlay;
}

function attachButton(node) {
  const buttonName = "Open Prompt Creator UI V2";
  const openUi = () => {
    const modal = ensureModal();
    modal.__vrgdgOpenForNode(node);
  };
  node.widgets = (node.widgets || []).filter((widget) => !(widget.type === "button" && widget.name === buttonName));

  const button = node.addWidget("button", buttonName, null, openUi);
  if (button) button.serialize = false;
}

function attachWorkflowButton(node, workflowKind) {
  const isPart3 = workflowKind === "part3";
  const buttonName = isPart3 ? "Open Workflow 3 UI" : "Open Part 2 Workflow UI";
  const openUi = () => {
    const modal = ensurePart2Modal();
    modal.__vrgdgOpenPart2(workflowKind, node);
  };
  node.widgets = (node.widgets || []).filter((widget) => !(widget.type === "button" && widget.name === buttonName));

  const button = node.addWidget("button", buttonName, null, openUi);
  if (button) button.serialize = false;
}

function attachUiForNode(node) {
  const nodeTypeName = node?.comfyClass || node?.type;
  if (nodeTypeName === PART2_NODE_NAME) attachWorkflowButton(node, "part2");
  else if (nodeTypeName === PART3_NODE_NAME) attachWorkflowButton(node, "part3");
  else if (nodeTypeName === NODE_NAME) attachButton(node);
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  loadedGraphNode(node) {
    attachUiForNode(node);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME && nodeData.name !== PART2_NODE_NAME && nodeData.name !== PART3_NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      this.properties = this.properties || {};
      attachUiForNode(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.properties = this.properties || {};
      attachUiForNode(this);
      return result;
    };
  },
});
