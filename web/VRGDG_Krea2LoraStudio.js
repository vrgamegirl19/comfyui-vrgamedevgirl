import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_Krea2LoraStudio";
const RECENT_PROJECTS_KEY = "vrgdg.krea2Studio.recentProjects.v1";

console.log("[VRGDG] Krea 2 LoRA Studio extension loaded");

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w?.name === name);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function jsonFetch(url, payload) {
  const response = await api.fetchApi(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await response.json();
  if (!response.ok || !data?.ok) throw new Error(data?.error || `HTTP ${response.status}`);
  return data;
}

function addStyles() {
  if (document.getElementById("vrgdg-krea2-studio-styles")) return;
  const style = document.createElement("style");
  style.id = "vrgdg-krea2-studio-styles";
  style.textContent = `
    .vrgdg-krea2-overlay {
      position: fixed;
      inset: 0;
      z-index: 10040;
      display: flex;
      align-items: stretch;
      justify-content: center;
      background: rgba(6, 8, 12, 0.82);
      color: #eef2f7;
      font-family: Inter, ui-sans-serif, system-ui, Segoe UI, sans-serif;
    }
    .vrgdg-krea2-shell {
      width: min(1500px, calc(100vw - 28px));
      height: calc(100vh - 28px);
      margin: 14px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 8px;
      background: #0d1017;
      box-shadow: 0 28px 90px rgba(0, 0, 0, 0.56);
    }
    .vrgdg-krea2-app {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: 0;
      height: 100%;
    }
    .vrgdg-krea2-rail {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 0;
      padding: 24px 18px;
      border-right: 1px solid rgba(148, 163, 184, 0.14);
      background: #0e1522;
    }
    .vrgdg-krea2-brand {
      display: grid;
      grid-template-columns: 34px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 34px;
    }
    .vrgdg-krea2-logo {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: #4f46e5;
      color: white;
      font-weight: 900;
    }
    .vrgdg-krea2-brand-title {
      font-size: 14px;
      font-weight: 840;
      color: #f8fafc;
    }
    .vrgdg-krea2-brand-subtitle {
      margin-top: 3px;
      color: #94a3b8;
      font-size: 12px;
    }
    .vrgdg-krea2-nav {
      display: grid;
      align-content: start;
      gap: 22px;
      min-height: 0;
      overflow: auto;
    }
    .vrgdg-krea2-nav-group {
      display: grid;
      gap: 8px;
    }
    .vrgdg-krea2-nav-label {
      color: #74849a;
      font-size: 10px;
      font-weight: 780;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      margin-bottom: 4px;
    }
    .vrgdg-krea2-nav-btn {
      display: grid;
      grid-template-columns: 24px 1fr;
      gap: 10px;
      align-items: center;
      min-height: 40px;
      border: 0;
      border-radius: 7px;
      background: transparent;
      color: #e5edf7;
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
      font-size: 12px;
    }
    .vrgdg-krea2-nav-btn:hover,
    .vrgdg-krea2-nav-btn.active {
      background: #1c2638;
    }
    .vrgdg-krea2-nav-ico {
      width: 20px;
      height: 20px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 5px;
      color: #cbd5e1;
      font-size: 12px;
    }
    .vrgdg-krea2-rail-card {
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 8px;
      background: #111a29;
      padding: 14px;
    }
    .vrgdg-krea2-rail-card-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 12px;
      font-weight: 760;
      color: #f1f5f9;
    }
    .vrgdg-krea2-mini-bar {
      height: 6px;
      overflow: hidden;
      border-radius: 999px;
      background: #263244;
      margin: 11px 0;
    }
    .vrgdg-krea2-mini-bar > div {
      height: 100%;
      border-radius: inherit;
      background: #64748b;
    }
    .vrgdg-krea2-body {
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
      min-height: 0;
      background: #0b111c;
    }
    .vrgdg-krea2-header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 22px 24px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.14);
      background: #0f1726;
    }
    .vrgdg-krea2-project-title {
      font-size: 16px;
      font-weight: 840;
      color: #f8fafc;
    }
    .vrgdg-krea2-content {
      display: grid;
      grid-template-columns: minmax(300px, 340px) minmax(0, 1fr);
      gap: 16px;
      min-height: 0;
      overflow: hidden;
      padding: 20px 24px 24px;
    }
    .vrgdg-krea2-panel {
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 8px;
      background: #101827;
      min-height: 0;
      max-height: 100%;
      overflow: auto;
    }
    .vrgdg-krea2-step {
      padding: 16px 18px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }
    .vrgdg-krea2-step:last-child {
      border-bottom: 0;
    }
    .vrgdg-krea2-step h3,
    .vrgdg-krea2-card h3 {
      margin: 0 0 12px;
      font-size: 14px;
      color: #f1f5f9;
      font-weight: 800;
    }
    .vrgdg-krea2-card {
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 8px;
      background: #101827;
      padding: 18px;
    }
    .vrgdg-krea2-stack {
      display: grid;
      gap: 16px;
      align-content: start;
      min-height: 0;
      max-height: 100%;
      overflow: auto;
      padding-right: 2px;
    }
    .vrgdg-krea2-empty {
      min-height: 360px;
      display: grid;
      place-items: center;
      text-align: center;
      color: #94a3b8;
    }
    .vrgdg-krea2-empty-icon {
      width: 54px;
      height: 54px;
      display: grid;
      place-items: center;
      margin: 0 auto 18px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 12px;
      color: #94a3b8;
      font-size: 24px;
    }
    .vrgdg-krea2-full-button {
      width: 100%;
      justify-content: center;
      min-height: 38px;
    }
    .vrgdg-krea2-details {
      margin-top: 10px;
    }
    .vrgdg-krea2-details summary {
      list-style: none;
      cursor: pointer;
    }
    .vrgdg-krea2-details summary::-webkit-details-marker {
      display: none;
    }
    .vrgdg-krea2-caption-editor {
      margin-top: 12px;
      display: grid;
      gap: 10px;
      padding-top: 12px;
      border-top: 1px solid rgba(148, 163, 184, 0.12);
    }
    .vrgdg-krea2-top {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      padding: 14px 16px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.18);
      background: #141923;
    }
    .vrgdg-krea2-title {
      font-size: 18px;
      font-weight: 760;
      line-height: 1.2;
    }
    .vrgdg-krea2-subtitle {
      margin-top: 4px;
      color: #aeb8c7;
      font-size: 12px;
    }
    .vrgdg-krea2-main {
      display: grid;
      grid-template-columns: minmax(320px, 430px) 1fr;
      min-height: 0;
    }
    .vrgdg-krea2-side {
      min-height: 0;
      overflow: auto;
      padding: 14px;
      border-right: 1px solid rgba(148, 163, 184, 0.16);
      background: #10141c;
    }
    .vrgdg-krea2-work {
      min-height: 0;
      overflow: auto;
      padding: 14px 16px 18px;
      background: #0d1017;
    }
    .vrgdg-krea2-section {
      margin-bottom: 16px;
    }
    .vrgdg-krea2-section h3 {
      margin: 0 0 9px;
      font-size: 13px;
      color: #dbe4f0;
      font-weight: 730;
    }
    .vrgdg-krea2-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 9px;
    }
    .vrgdg-krea2-field {
      display: grid;
      gap: 5px;
      margin-bottom: 9px;
    }
    .vrgdg-krea2-field label {
      color: #aeb8c7;
      font-size: 11px;
      line-height: 1.25;
    }
    .vrgdg-krea2-label-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .vrgdg-krea2-hint-btn {
      width: 20px;
      height: 20px;
      flex: 0 0 auto;
      border: 1px solid rgba(125, 211, 252, 0.42);
      border-radius: 50%;
      background: #10202a;
      color: #bae6fd;
      font-size: 12px;
      font-weight: 800;
      line-height: 18px;
      text-align: center;
      cursor: pointer;
    }
    .vrgdg-krea2-hint-btn:hover {
      background: #164e63;
      color: white;
    }
    .vrgdg-krea2-field input,
    .vrgdg-krea2-field select,
    .vrgdg-krea2-field textarea {
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
      border: 1px solid rgba(148, 163, 184, 0.26);
      border-radius: 7px;
      background: #181e29;
      color: #eef2f7;
      padding: 8px 9px;
      font: inherit;
      font-size: 12px;
      outline: none;
    }
    .vrgdg-krea2-field textarea {
      min-height: 82px;
      resize: vertical;
    }
    .vrgdg-krea2-field input:focus,
    .vrgdg-krea2-field select:focus,
    .vrgdg-krea2-field textarea:focus {
      border-color: #7dd3fc;
      box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.12);
    }
    .vrgdg-krea2-segment {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 11px;
    }
    .vrgdg-krea2-btn,
    .vrgdg-krea2-icon {
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 7px;
      background: #1b2230;
      color: #eef2f7;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.2;
      cursor: pointer;
    }
    .vrgdg-krea2-btn:hover,
    .vrgdg-krea2-icon:hover {
      background: #263041;
      border-color: rgba(125, 211, 252, 0.5);
    }
    .vrgdg-krea2-btn.primary {
      background: #0f766e;
      border-color: #2dd4bf;
      color: white;
      font-weight: 720;
    }
    .vrgdg-krea2-btn.hot {
      background: #7c2d12;
      border-color: #fb923c;
      color: white;
      font-weight: 720;
    }
    .vrgdg-krea2-btn.active {
      background: #164e63;
      border-color: #67e8f9;
      color: white;
    }
    .vrgdg-krea2-icon {
      width: 34px;
      height: 34px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      font-size: 17px;
    }
    .vrgdg-krea2-drop {
      display: grid;
      place-items: center;
      min-height: 165px;
      padding: 16px;
      border: 1px dashed rgba(125, 211, 252, 0.48);
      border-radius: 8px;
      background: #111827;
      text-align: center;
      color: #cbd5e1;
    }
    .vrgdg-krea2-drop.drag {
      border-color: #5eead4;
      background: #10201f;
    }
    .vrgdg-krea2-drop strong {
      display: block;
      color: #f8fafc;
      margin-bottom: 6px;
      font-size: 14px;
    }
    .vrgdg-krea2-hint {
      color: #8996a8;
      font-size: 11px;
      line-height: 1.35;
    }
    .vrgdg-krea2-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .vrgdg-krea2-check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: -2px 0 10px;
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.3;
      user-select: none;
    }
    .vrgdg-krea2-check input {
      width: 16px;
      height: 16px;
      accent-color: #14b8a6;
    }
    .vrgdg-krea2-advanced {
      display: none;
      margin-top: 10px;
      padding-top: 12px;
      border-top: 1px solid rgba(148, 163, 184, 0.16);
    }
    .vrgdg-krea2-advanced.open {
      display: block;
    }
    .vrgdg-krea2-settings-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 8px 10px;
    }
    .vrgdg-krea2-status {
      min-height: 28px;
      padding: 8px 16px;
      border-top: 1px solid rgba(148, 163, 184, 0.16);
      color: #aeb8c7;
      background: #141923;
      font-size: 12px;
    }
    .vrgdg-krea2-progress {
      position: fixed;
      inset: 0;
      z-index: 10080;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(3, 6, 12, 0.68);
    }
    .vrgdg-krea2-progress-box {
      width: min(560px, calc(100vw - 32px));
      border: 1px solid rgba(125, 211, 252, 0.36);
      border-radius: 8px;
      background: #111827;
      color: #eef2f7;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
      padding: 16px;
    }
    .vrgdg-krea2-progress-title {
      font-size: 16px;
      font-weight: 800;
      margin-bottom: 8px;
    }
    .vrgdg-krea2-progress-title.done {
      color: #67e8f9;
    }
    .vrgdg-krea2-progress-title.failed {
      color: #fca5a5;
    }
    .vrgdg-krea2-progress-text {
      white-space: pre-wrap;
      min-height: 70px;
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.45;
    }
    .vrgdg-krea2-progress-bar {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #263244;
      margin: 12px 0;
    }
    .vrgdg-krea2-progress-bar > div {
      width: 36%;
      height: 100%;
      border-radius: inherit;
      background: #14b8a6;
      animation: vrgdg-krea2-progress-slide 1.2s ease-in-out infinite alternate;
    }
    @keyframes vrgdg-krea2-progress-slide {
      from { transform: translateX(-18%); }
      to { transform: translateX(192%); }
    }
    .vrgdg-krea2-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .vrgdg-krea2-sample {
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 8px;
      overflow: hidden;
      background: #151b25;
    }
    .vrgdg-krea2-sample img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      display: block;
      background: #0b0f16;
    }
    .vrgdg-krea2-sample div {
      padding: 8px 10px;
      font-size: 12px;
      color: #dbe4f0;
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }
    @media (max-width: 920px) {
      .vrgdg-krea2-app {
        grid-template-columns: 1fr;
      }
      .vrgdg-krea2-rail {
        display: none;
      }
      .vrgdg-krea2-content {
        grid-template-columns: 1fr;
        padding: 12px;
      }
      .vrgdg-krea2-main {
        grid-template-columns: 1fr;
      }
      .vrgdg-krea2-side {
        border-right: none;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      }
      .vrgdg-krea2-shell {
        width: calc(100vw - 12px);
        height: calc(100vh - 12px);
        margin: 6px;
      }
    }
  `;
  document.head.appendChild(style);
}

function settingHints() {
  return {
    resolution_width: "Training bucket width written into the musubi dataset config.",
    resolution_height: "Training bucket height written into the musubi dataset config.",
    steps_per_run: "How many steps each button press trains before saving a LoRA and generating a sample.",
    total_target_steps: "The final step count where the trainer stops resuming.",
    network_dim: "LoRA rank. Higher can learn more but uses more memory and can overfit.",
    network_alpha: "LoRA alpha scaling. Matching rank is the normal Krea 2 default.",
    blocks_to_swap: "Moves Krea 2 blocks to CPU to reduce VRAM use. Zero is fastest if VRAM allows it.",
    clear_memory_before_text_encoder: "Clears Comfy models before text encoder caching to free VRAM.",
    learning_rate_preset: "Quick learning-rate selector used by the backend trainer.",
    learning_rate: "Used only when learning_rate_preset is Custom.",
    num_repeats: "Repeats each image-caption pair in the dataset.",
    cache_strategy: "Auto reuses cache when possible, force rebuilds, skip goes straight to training.",
    copy_latest_to_comfy_loras: "Leave off for Studio samples; the hidden workflow loads by path.",
    create_captions: "Simple fallback caption creation from caption_text. The LLM captioner is a separate placeholder.",
    caption_text: "Fallback caption text if create_captions is enabled.",
    add_trigger_word: "Prepends trigger_text to captions before training.",
    trigger_text: "Trigger word or phrase to prepend.",
    musubi_root: "Folder containing the native Krea 2 musubi-tuner scripts.",
    krea2_raw_dit: "Krea 2 RAW DiT checkpoint used for training.",
    vae: "Qwen Image VAE checkpoint used for training cache.",
    text_encoder: "Qwen3-VL text encoder checkpoint used for caption embedding cache.",
    fp8_base: "Uses fp8 base weights. Krea 2 expects fp8_scaled with this.",
    fp8_scaled: "Dynamic scaled fp8 weights. Required when fp8_base is on.",
    timestep_sampling: "Krea 2 docs recommend shift.",
    discrete_flow_shift: "Flow shift value used when timestep_sampling is shift. 2.5 is the current default.",
    aspect_ratio: "Aspect ratio used by the hidden Krea 2 sample workflow. Both resolution selector nodes are patched to this same value.",
  };
}

function settingLabel(key) {
  if (key === "aspect_ratio") return "sample aspect ratio";
  return key.replaceAll("_", " ");
}

function settingLabelHtml(key, hint) {
  return `
    <div class="vrgdg-krea2-label-line">
      <label>${esc(settingLabel(key))}</label>
      <button class="vrgdg-krea2-hint-btn" data-hint-key="${esc(key)}" title="What does this setting do?">?</button>
    </div>
  `;
}

function coerceByExisting(value, existing) {
  if (typeof existing === "boolean") return Boolean(value);
  if (typeof existing === "number") return Number(value);
  return value;
}

function cloneData(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function imageUrl(path) {
  return `/vrgdg/krea2_studio/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
}

function extractImagesFromHistory(historyPayload, promptId) {
  const item = historyPayload?.[promptId] || historyPayload;
  const outputs = item?.outputs || {};
  const images = [];
  for (const output of Object.values(outputs)) {
    for (const image of output?.images || []) images.push(image);
  }
  return images;
}

async function queuePrompt(prompt) {
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: api.clientId }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message || data?.error || `HTTP ${response.status}`);
  return data.prompt_id;
}

async function waitForImageOutput(promptId, onStatus) {
  for (let attempt = 0; attempt < 900; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    onStatus?.(`Sampling... ${Math.floor((attempt + 1) * 1.5)}s`);
    const response = await api.fetchApi(`/history/${promptId}`);
    const history = await response.json();
    const images = extractImagesFromHistory(history, promptId);
    if (images.length) return images;
  }
  throw new Error("Timed out waiting for the sample image.");
}

class Krea2Studio {
  constructor(node) {
    this.node = node;
    this.defaults = null;
    this.project = null;
    this.currentPreset = "Fast";
    this.settings = {};
    this.status = "Ready.";
    this.advancedOpen = false;
    this.autoContinue = true;
    this.stopRequested = false;
    this.isTraining = false;
    this.lastDroppedFiles = [];
    this.llmChoices = { models: [], mmproj: [], providers: [] };
    this.captionRunner = "builtin";
    this.captionGemmaModel = "";
    this.captionMmprojFile = "";
    this.lmStudioBaseUrl = "http://127.0.0.1:1234/v1";
    this.lmStudioModel = "";
    this.lmStudioApiKey = "";
    this.llmApiProvider = "openai";
    this.llmApiModel = "";
    this.llmApiKey = "";
    this.aspectRatio = "";
    this.samplePrompt = "";
    this.sampleModelChoices = { diffusion_models: [], text_encoders: [], vae: [] };
    this.sampleModelSettings = { diffusion_model: "", text_encoder: "", vae: "" };
    this.captionInstructions = "";
    this.captionUserNotes = "";
    this.captionOverwrite = false;
    this.projectList = [];
    this.overlay = null;
  }

  async open() {
    addStyles();
    const response = await api.fetchApi("/vrgdg/krea2_studio/defaults");
    this.defaults = await response.json();
    this.currentPreset = "Fast";
    this.settings = cloneData(this.defaults.presets.Fast);
    this.captionRunner = this.defaults.caption_runner || "builtin";
    this.lmStudioBaseUrl = this.defaults.lmstudio_base_url || this.lmStudioBaseUrl;
    this.aspectRatio = "3:4 (Portrait Standard)";
    this.samplePrompt = this.defaults.sample_prompt || "";
    this.sampleModelChoices = this.defaults.sample_model_choices || { diffusion_models: [], text_encoders: [], vae: [] };
    this.sampleModelSettings = {
      diffusion_model: this.defaults.sample_model_defaults?.diffusion_model || this.sampleModelChoices.diffusion_models?.[0] || "",
      text_encoder: this.defaults.sample_model_defaults?.text_encoder || this.sampleModelChoices.text_encoders?.[0] || "",
      vae: this.defaults.sample_model_defaults?.vae || this.sampleModelChoices.vae?.[0] || "",
    };
    this.captionInstructions = this.defaults.caption_instructions || "";
    this.captionUserNotes = "";
    await this.refreshLlmChoices();
    this.render();
  }

  async refreshLlmChoices() {
    try {
      const response = await api.fetchApi("/vrgdg/krea2_studio/llm_choices");
      const data = await response.json();
      if (!response.ok || !data?.ok) throw new Error(data?.error || `HTTP ${response.status}`);
      this.llmChoices = {
        models: Array.isArray(data.models) ? data.models : [],
        mmproj: Array.isArray(data.mmproj) ? data.mmproj : [],
        providers: Array.isArray(data.providers) ? data.providers : [],
      };
      if (!this.captionGemmaModel && this.llmChoices.models.length) this.captionGemmaModel = this.llmChoices.models[0];
      if (!this.captionMmprojFile && this.llmChoices.mmproj.length) this.captionMmprojFile = this.llmChoices.mmproj[0];
      const provider = this.llmChoices.providers.find((item) => item.id === this.llmApiProvider) || this.llmChoices.providers[0];
      if (provider) {
        this.llmApiProvider = provider.id || this.llmApiProvider;
        if (!this.llmApiModel) this.llmApiModel = provider.default_model || provider.models?.[0] || "";
      }
    } catch (error) {
      console.warn("[VRGDG] Krea 2 Studio could not load LLM choices:", error);
    }
  }

  render() {
    if (!this.overlay) {
      this.overlay = document.createElement("div");
      this.overlay.className = "vrgdg-krea2-overlay";
      document.body.appendChild(this.overlay);
    }

    const projectRoot = getWidget(this.node, "project_root")?.value || this.defaults.project_root;
    const projectName = getWidget(this.node, "project_name")?.value || this.defaults.project_name;
    const samplePrompt = this.samplePrompt || this.project?.sample_prompt || this.defaults.sample_prompt || "";
    const aspectRatio = this.aspectRatio || this.project?.aspect_ratio || "3:4 (Portrait Standard)";
    const captionInstructions = this.captionInstructions || this.project?.caption_instructions || this.defaults.caption_instructions || "";
    const captionUserNotes = this.captionUserNotes || this.project?.caption_user_notes || this.defaults.caption_user_notes || "";
    const sampleModels = this.sampleModelSettings || {};
    this.applyProjectCaptionSettings();
    const completed = Number(this.project?.completed_steps || 0);
    const target = Number(this.project?.total_target_steps || this.settings.total_target_steps || 0);
    const progressPercent = target > 0 ? Math.max(0, Math.min(100, Math.round((completed / target) * 100))) : 0;
    const importedImages = (this.project?.imported_files || []).filter((item) => item.type === "image");

    this.overlay.innerHTML = `
      <div class="vrgdg-krea2-shell" style="grid-template-rows:1fr;">
        <div class="vrgdg-krea2-app">
          <aside class="vrgdg-krea2-rail">
            <div>
              <div class="vrgdg-krea2-brand">
                <div class="vrgdg-krea2-logo">K2</div>
                <div>
                  <div class="vrgdg-krea2-brand-title">VRGDG Krea 2</div>
                  <div class="vrgdg-krea2-brand-subtitle">LoRA Studio</div>
                </div>
              </div>
              <div class="vrgdg-krea2-nav">
                <div class="vrgdg-krea2-nav-group">
                  <div class="vrgdg-krea2-nav-label">Project</div>
                  <button class="vrgdg-krea2-nav-btn" data-action="openProjectWindow"><span class="vrgdg-krea2-nav-ico">+</span><span>New Project</span></button>
                  <button class="vrgdg-krea2-nav-btn" data-action="openProjectWindow"><span class="vrgdg-krea2-nav-ico">O</span><span>Open Project</span></button>
                  <button class="vrgdg-krea2-nav-btn active" data-action="openProjectWindow">
                    <span class="vrgdg-krea2-nav-ico">P</span>
                    <span>Current Project<br><span class="vrgdg-krea2-hint">${esc(this.project?.project_name || projectName || "None")}</span></span>
                  </button>
                </div>
                <div class="vrgdg-krea2-nav-group">
                  <div class="vrgdg-krea2-nav-label">Settings</div>
                  <button class="vrgdg-krea2-nav-btn" data-action="scrollKreaModels"><span class="vrgdg-krea2-nav-ico">K</span><span>Krea Models</span></button>
                  <button class="vrgdg-krea2-nav-btn" data-action="toggleAdvanced"><span class="vrgdg-krea2-nav-ico">S</span><span>Training Settings</span></button>
                  <button class="vrgdg-krea2-nav-btn" data-action="clearCaptionMemory"><span class="vrgdg-krea2-nav-ico">M</span><span>Clear Memory</span></button>
                </div>
              </div>
            </div>
            <div class="vrgdg-krea2-rail-card">
              <div class="vrgdg-krea2-rail-card-title"><span>Training Chunk</span><span>${esc(completed)}/${esc(target || "?")}</span></div>
              <div class="vrgdg-krea2-mini-bar"><div style="width:${progressPercent}%;"></div></div>
              <div class="vrgdg-krea2-hint">${esc(this.status || "Ready to start training")}</div>
            </div>
          </aside>
          <main class="vrgdg-krea2-body">
            <header class="vrgdg-krea2-header">
              <div>
                <div class="vrgdg-krea2-project-title">${esc(this.project?.project_name || projectName || "Krea2Studio")}</div>
                <div class="vrgdg-krea2-subtitle">${esc(this.project?.project_dir || projectRoot || "Choose or create a project to begin.")}</div>
              </div>
              <div class="vrgdg-krea2-actions">
                <button class="vrgdg-krea2-btn" data-action="saveProject">Save Project</button>
                <button class="vrgdg-krea2-icon" data-action="close" title="Close">x</button>
              </div>
            </header>
            <div class="vrgdg-krea2-content">
              <section class="vrgdg-krea2-panel">
                <div class="vrgdg-krea2-step">
                  <h3>1. Preset</h3>
                  <div class="vrgdg-krea2-segment">
                    ${["Fast", "Medium", "Long"].map((name) => `<button class="vrgdg-krea2-btn ${this.currentPreset === name ? "active" : ""}" data-preset="${name}">${name}</button>`).join("")}
                  </div>
                  <div class="vrgdg-krea2-row">
                    <div class="vrgdg-krea2-field">${settingLabelHtml("resolution_width")}<input data-setting="resolution_width" type="number" value="${esc(this.settings.resolution_width)}"></div>
                    <div class="vrgdg-krea2-field">${settingLabelHtml("resolution_height")}<input data-setting="resolution_height" type="number" value="${esc(this.settings.resolution_height)}"></div>
                  </div>
                  <div class="vrgdg-krea2-field">
                    ${settingLabelHtml("aspect_ratio")}
                    <select data-bind="aspect_ratio">${this.defaults.aspect_ratios.map((item) => `<option ${item === aspectRatio ? "selected" : ""}>${esc(item)}</option>`).join("")}</select>
                  </div>
                  <details class="vrgdg-krea2-details" ${this.advancedOpen ? "open" : ""}>
                    <summary><button class="vrgdg-krea2-btn vrgdg-krea2-full-button" data-action="toggleAdvanced">${this.advancedOpen ? "Hide Settings" : "Show / Edit Settings"}</button></summary>
                    <div class="vrgdg-krea2-advanced open">
                      <div class="vrgdg-krea2-settings-grid">${this.renderSettings()}</div>
                      <div class="vrgdg-krea2-actions" style="margin-top:10px;">
                        <button class="vrgdg-krea2-btn" data-action="savePreset">Save Preset</button>
                      </div>
                    </div>
                  </details>
                </div>
                <div class="vrgdg-krea2-step" data-section="krea-models">
                  <h3>2. Krea Models</h3>
                  <div class="vrgdg-krea2-hint" style="margin-bottom:10px;">Sample image workflow models. These do not change training checkpoints.</div>
                  <div class="vrgdg-krea2-field">
                    <label>Sample diffusion model</label>
                    <select data-bind="sample_diffusion_model">${this.renderOptions(this.sampleModelChoices.diffusion_models, sampleModels.diffusion_model)}</select>
                  </div>
                  <div class="vrgdg-krea2-field">
                    <label>Sample text encoder</label>
                    <select data-bind="sample_text_encoder">${this.renderOptions(this.sampleModelChoices.text_encoders, sampleModels.text_encoder)}</select>
                  </div>
                  <div class="vrgdg-krea2-field">
                    <label>Sample VAE</label>
                    <select data-bind="sample_vae">${this.renderOptions(this.sampleModelChoices.vae, sampleModels.vae)}</select>
                  </div>
                </div>
                <div class="vrgdg-krea2-step">
                  <h3>3. Dataset</h3>
                  <div class="vrgdg-krea2-drop" data-dropzone>
                    <div>
                      <strong>Drop images and .txt captions here</strong>
                      <div>or click to browse</div>
                      <div class="vrgdg-krea2-hint" style="margin-top:8px;">Images are copied and renamed as image_###.</div>
                      <button class="vrgdg-krea2-btn" data-action="browseFiles" style="margin-top:12px;">Browse Files</button>
                      <input data-file-input type="file" multiple accept="image/*,.txt,.caption" style="display:none;">
                    </div>
                  </div>
                  ${this.renderImportList()}
                </div>
                <div class="vrgdg-krea2-step">
                  <h3>4. Captioning</h3>
                  ${this.renderCaptionRunner()}
                  <div class="vrgdg-krea2-actions" style="margin-top:10px;">
                    <button class="vrgdg-krea2-btn primary" data-action="captionPlaceholder">Generate Missing Captions</button>
                    <button class="vrgdg-krea2-btn" data-action="clearCaptionMemory">Clear Memory</button>
                  </div>
                  <label class="vrgdg-krea2-check" style="margin-top:10px;" title="When enabled, existing image_###.txt caption files are replaced using the current instructions.">
                    <input data-bind="caption_overwrite" type="checkbox" ${this.captionOverwrite ? "checked" : ""}>
                    <span>Overwrite existing captions</span>
                  </label>
                  <div class="vrgdg-krea2-caption-editor">
                    <div class="vrgdg-krea2-field">
                      <label>Caption instructions</label>
                      <textarea data-bind="caption_instructions">${esc(captionInstructions)}</textarea>
                    </div>
                    <div class="vrgdg-krea2-field">
                      <label>User input / global tags appended at the end</label>
                      <textarea data-bind="caption_user_notes">${esc(captionUserNotes)}</textarea>
                    </div>
                  </div>
                </div>
                <div class="vrgdg-krea2-step">
                  <h3>5. Train</h3>
                  <button class="vrgdg-krea2-btn hot vrgdg-krea2-full-button" data-action="trainChunk">${this.autoContinue ? "Start Training + Samples" : "Train One Chunk + Sample"}</button>
                  ${this.isTraining ? `<button class="vrgdg-krea2-btn vrgdg-krea2-full-button" data-action="stopTraining" style="margin-top:8px;">Stop After Current Chunk</button>` : ""}
                  <label class="vrgdg-krea2-check" style="margin-top:12px;" title="When enabled, Studio keeps training the next chunk after each sample is saved until total_target_steps is reached. Turn this off to review each LoRA before continuing.">
                    <input data-bind="auto_continue" type="checkbox" ${this.autoContinue ? "checked" : ""}>
                    <span>Continue next train chunk after each sample</span>
                  </label>
                </div>
              </section>
              <section class="vrgdg-krea2-stack">
                <div class="vrgdg-krea2-card">
                  <h3>Dataset</h3>
                  ${importedImages.length ? `<div class="vrgdg-krea2-gallery">${importedImages.slice(-12).map((item) => `<div class="vrgdg-krea2-sample"><img src="${esc(imageUrl(item.path))}"><div><span>${esc(item.name)}</span><span>dataset</span></div></div>`).join("")}</div>` : `<div class="vrgdg-krea2-empty"><div><div class="vrgdg-krea2-empty-icon">[]</div><div class="vrgdg-krea2-project-title">No dataset selected</div><div class="vrgdg-krea2-hint" style="margin-top:8px;">Add your training images and captions to get started.</div></div></div>`}
                </div>
                <div class="vrgdg-krea2-card">
                  <h3>Sample Prompt</h3>
                  <div class="vrgdg-krea2-field"><textarea data-bind="sample_prompt">${esc(samplePrompt)}</textarea></div>
                  <div class="vrgdg-krea2-hint">This prompt is used to generate each sample after a training chunk.</div>
                </div>
                <div class="vrgdg-krea2-card">
                  <h3>Samples</h3>
                  <div class="vrgdg-krea2-gallery">${this.renderGallery()}</div>
                </div>
              </section>
            </div>
          </main>
        </div>
      </div>
    `;
    this.bindEvents();
  }

  renderSettings() {
    const hints = settingHints();
    return Object.entries(this.settings).map(([key, value]) => {
      if (key === "image_guidance" || key === "resolution_width" || key === "resolution_height") return "";
      const hint = hints[key] || "";
      if (typeof value === "boolean") {
        return `<div class="vrgdg-krea2-field">${settingLabelHtml(key, hint)}<select data-setting="${esc(key)}"><option value="true" ${value ? "selected" : ""}>true</option><option value="false" ${!value ? "selected" : ""}>false</option></select></div>`;
      }
      if (key === "learning_rate_preset") {
        return `<div class="vrgdg-krea2-field">${settingLabelHtml(key, hint)}<select data-setting="${esc(key)}">${["Custom", "1e-4", "7e-5", "5e-5", "3e-5", "1e-5"].map((item) => `<option ${item === value ? "selected" : ""}>${item}</option>`).join("")}</select></div>`;
      }
      if (key === "cache_strategy") {
        return `<div class="vrgdg-krea2-field">${settingLabelHtml(key, hint)}<select data-setting="${esc(key)}">${["auto", "force", "skip"].map((item) => `<option ${item === value ? "selected" : ""}>${item}</option>`).join("")}</select></div>`;
      }
      if (key === "timestep_sampling") {
        return `<div class="vrgdg-krea2-field">${settingLabelHtml(key, hint)}<select data-setting="${esc(key)}">${["shift", "qwen_shift", "flux_shift"].map((item) => `<option ${item === value ? "selected" : ""}>${item}</option>`).join("")}</select></div>`;
      }
      const type = typeof value === "number" ? "number" : "text";
      const step = Number.isInteger(value) ? "1" : "0.000001";
      return `<div class="vrgdg-krea2-field">${settingLabelHtml(key, hint)}<input data-setting="${esc(key)}" type="${type}" step="${step}" value="${esc(value)}"></div>`;
    }).join("");
  }

  renderGallery() {
    const samples = this.project?.samples || [];
    const tiles = samples.map((sample) => `
      <div class="vrgdg-krea2-sample">
        <img src="${esc(imageUrl(sample.path))}">
        <div><span>Step ${esc(sample.step)}</span><span>${esc((sample.path || "").split(/[\\/]/).pop())}</span></div>
      </div>
    `);
    if (this.project?.xyz_plot_path) {
      tiles.push(`
        <div class="vrgdg-krea2-sample">
          <img src="${esc(imageUrl(this.project.xyz_plot_path))}">
          <div><span>XYZ Plot</span><span>steps</span></div>
        </div>
      `);
    }
    return tiles.length ? tiles.join("") : `<div class="vrgdg-krea2-hint">Samples will appear here after each chunk finishes.</div>`;
  }

  renderImportList() {
    const imported = this.project?.imported_files || [];
    const recent = this.lastDroppedFiles || [];
    const importedList = imported.slice(-12).map((item) => `
      <div class="vrgdg-krea2-hint">${esc(item.name || "")}${item.original_name ? ` from ${esc(item.original_name)}` : ""}</div>
    `).join("");
    const recentList = recent.map((name) => `<div class="vrgdg-krea2-hint">${esc(name)}</div>`).join("");
    if (!importedList && !recentList) {
      return `<div class="vrgdg-krea2-hint" style="margin-top:8px;">No files imported yet.</div>`;
    }
    return `
      <div style="margin-top:10px;">
        ${recentList ? `<div class="vrgdg-krea2-hint" style="color:#bae6fd;">Received drop:</div>${recentList}` : ""}
        ${importedList ? `<div class="vrgdg-krea2-hint" style="color:#bae6fd;margin-top:8px;">Project files:</div>${importedList}` : ""}
      </div>
    `;
  }

  renderCaptionRunner() {
    const providers = this.llmChoices.providers.length ? this.llmChoices.providers : [{ id: "openai", label: "OpenAI", models: ["gpt-4o"], default_model: "gpt-4o" }];
    const activeProvider = providers.find((item) => item.id === this.llmApiProvider) || providers[0];
    const apiModels = activeProvider?.models?.length ? activeProvider.models : [activeProvider?.default_model || ""].filter(Boolean);
    return `
      <div class="vrgdg-krea2-field">
        <label>Caption LLM runner</label>
        <select data-bind="caption_runner">
          <option value="builtin" ${this.captionRunner === "builtin" ? "selected" : ""}>Gemma Local</option>
          <option value="lm_studio" ${this.captionRunner === "lm_studio" ? "selected" : ""}>LM Studio</option>
          <option value="llm_api" ${this.captionRunner === "llm_api" ? "selected" : ""}>LLM API</option>
        </select>
      </div>
      <div class="vrgdg-krea2-settings-grid" data-caption-panel="builtin" style="display:${this.captionRunner === "builtin" ? "grid" : "none"};">
        <div class="vrgdg-krea2-field"><label>Gemma vision model</label><select data-bind="caption_gemma_model">${this.renderOptions(this.llmChoices.models, this.captionGemmaModel)}</select></div>
        <div class="vrgdg-krea2-field"><label>Vision mmproj</label><select data-bind="caption_mmproj_file">${this.renderOptions(this.llmChoices.mmproj, this.captionMmprojFile)}</select></div>
      </div>
      <div class="vrgdg-krea2-settings-grid" data-caption-panel="lm_studio" style="display:${this.captionRunner === "lm_studio" ? "grid" : "none"};">
        <div class="vrgdg-krea2-field"><label>LM Studio base URL</label><input data-bind="lmstudio_base_url" value="${esc(this.lmStudioBaseUrl)}"></div>
        <div class="vrgdg-krea2-field"><label>LM Studio model</label><input data-bind="lmstudio_model" value="${esc(this.lmStudioModel)}"></div>
        <div class="vrgdg-krea2-field"><label>LM Studio API key</label><input data-bind="lmstudio_api_key" type="password" value="${esc(this.lmStudioApiKey)}"></div>
        <div class="vrgdg-krea2-field"><label>Available models</label><button class="vrgdg-krea2-btn" data-action="loadLmStudioModels">Load LM Studio Models</button></div>
      </div>
      <div class="vrgdg-krea2-settings-grid" data-caption-panel="llm_api" style="display:${this.captionRunner === "llm_api" ? "grid" : "none"};">
        <div class="vrgdg-krea2-field"><label>Provider</label><select data-bind="llm_api_provider">${providers.map((provider) => `<option value="${esc(provider.id)}" ${provider.id === this.llmApiProvider ? "selected" : ""}>${esc(provider.label || provider.id)}</option>`).join("")}</select></div>
        <div class="vrgdg-krea2-field"><label>Model</label><select data-bind="llm_api_model">${this.renderOptions(apiModels, this.llmApiModel || activeProvider?.default_model || "")}</select></div>
        <div class="vrgdg-krea2-field"><label>API key</label><input data-bind="llm_api_key" type="password" value="${esc(this.llmApiKey)}"></div>
      </div>
    `;
  }

  renderOptions(values, selected) {
    const list = values?.length ? [...values] : [""];
    if (selected && !list.includes(selected)) list.unshift(selected);
    return list.map((value) => `<option value="${esc(value)}" ${String(value) === String(selected) ? "selected" : ""}>${esc(value || "(none found)")}</option>`).join("");
  }

  openProgressWindow(title, initialText) {
    const modal = document.createElement("div");
    modal.className = "vrgdg-krea2-progress";
    const started = Date.now();
    modal.innerHTML = `
      <div class="vrgdg-krea2-progress-box">
        <div class="vrgdg-krea2-progress-title">${esc(title)}</div>
        <div class="vrgdg-krea2-progress-text" data-progress-text>${esc(initialText || "Starting...")}</div>
        <div class="vrgdg-krea2-progress-bar"><div></div></div>
        <div class="vrgdg-krea2-actions">
          <button class="vrgdg-krea2-btn hot" data-progress-stop>Stop</button>
          <button class="vrgdg-krea2-btn" data-progress-clear>Clear Memory</button>
          <button class="vrgdg-krea2-btn" data-progress-close style="display:none;">Close</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    const titleEl = modal.querySelector(".vrgdg-krea2-progress-title");
    const textEl = modal.querySelector("[data-progress-text]");
    const closeButton = modal.querySelector("[data-progress-close]");
    const stopButton = modal.querySelector("[data-progress-stop]");
    const clearButton = modal.querySelector("[data-progress-clear]");
    const timer = setInterval(() => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      const current = textEl?.dataset.current || initialText || "Working...";
      if (textEl) textEl.textContent = `${current}\n\nElapsed: ${elapsed}s`;
    }, 1000);
    closeButton?.addEventListener("click", () => {
      clearInterval(timer);
      modal.remove();
    });
    return {
      set(message) {
        if (!textEl) return;
        textEl.dataset.current = String(message || "");
        textEl.textContent = String(message || "");
      },
      done(message, doneTitle = "Completed", failed = false) {
        clearInterval(timer);
        if (titleEl) {
          titleEl.textContent = String(doneTitle || "Completed");
          titleEl.classList.toggle("done", !failed);
          titleEl.classList.toggle("failed", Boolean(failed));
        }
        if (textEl) textEl.textContent = String(message || "Done.");
        const bar = modal.querySelector(".vrgdg-krea2-progress-bar");
        if (bar) bar.style.display = "none";
        if (stopButton) stopButton.style.display = "none";
        if (closeButton) closeButton.style.display = "";
      },
      onStop(handler) {
        stopButton?.addEventListener("click", handler);
      },
      onClear(handler) {
        clearButton?.addEventListener("click", handler);
      },
      close(delayMs = 0) {
        clearInterval(timer);
        setTimeout(() => modal.remove(), delayMs);
      },
    };
  }

  applyProjectCaptionSettings() {
    const settings = this.project?.caption_llm_settings || {};
    if (!settings || this.__captionSettingsAppliedFor === this.project?.project_dir) return;
    this.__captionSettingsAppliedFor = this.project?.project_dir || "";
    this.captionRunner = settings.caption_runner || this.captionRunner || "builtin";
    this.captionGemmaModel = settings.model_file || this.captionGemmaModel || "";
    this.captionMmprojFile = settings.mmproj_file || this.captionMmprojFile || "";
    this.lmStudioBaseUrl = settings.lmstudio_base_url || this.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    this.lmStudioModel = settings.lmstudio_model || this.lmStudioModel || "";
    this.lmStudioApiKey = settings.lmstudio_api_key || this.lmStudioApiKey || "";
    this.llmApiProvider = settings.llm_api_provider || this.llmApiProvider || "openai";
    this.llmApiModel = settings.llm_api_model || this.llmApiModel || "";
    this.aspectRatio = this.project?.aspect_ratio || this.aspectRatio || "3:4 (Portrait Standard)";
    this.samplePrompt = this.project?.sample_prompt || this.samplePrompt || this.defaults?.sample_prompt || "";
    this.sampleModelSettings = {
      ...this.sampleModelSettings,
      ...(this.project?.sample_model_settings || {}),
    };
    this.captionInstructions = this.project?.caption_instructions || this.captionInstructions || this.defaults?.caption_instructions || "";
    this.captionUserNotes = this.project?.caption_user_notes || this.captionUserNotes || "";
  }

  captionLlmSettings() {
    return {
      caption_runner: this.captionRunner,
      model_file: this.captionGemmaModel,
      mmproj_file: this.captionMmprojFile,
      lmstudio_base_url: this.lmStudioBaseUrl,
      lmstudio_model: this.lmStudioModel,
      lmstudio_api_key: this.lmStudioApiKey,
      llm_api_provider: this.llmApiProvider,
      llm_api_model: this.llmApiModel,
    };
  }

  imageCountHint() {
    if (this.currentPreset === "Fast") return "Fast is tuned for 10 images or fewer.";
    if (this.currentPreset === "Medium") return "Medium is tuned for about 20 images.";
    return "Long is tuned for more than 20 images.";
  }

  bindEvents() {
    this.overlay.addEventListener("dragover", (event) => event.preventDefault());
    this.overlay.addEventListener("drop", (event) => event.preventDefault());

    this.overlay.querySelector('[data-action="close"]')?.addEventListener("click", () => this.close());
    for (const button of this.overlay.querySelectorAll('[data-action="openProjectWindow"]')) {
      button.addEventListener("click", () => this.openProjectWindow());
    }
    this.overlay.querySelector('[data-action="saveProject"]')?.addEventListener("click", () => this.saveProjectWithStatus());
    this.overlay.querySelector('[data-action="scrollKreaModels"]')?.addEventListener("click", () => {
      this.overlay.querySelector('[data-section="krea-models"]')?.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    for (const button of this.overlay.querySelectorAll('[data-action="toggleAdvanced"]')) {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        this.advancedOpen = !this.advancedOpen;
        this.collectForm();
        this.render();
      });
    }
    this.overlay.querySelector('[data-action="savePreset"]')?.addEventListener("click", () => this.savePreset());
    this.overlay.querySelector('[data-action="captionPlaceholder"]')?.addEventListener("click", () => this.captionPlaceholder());
    for (const button of this.overlay.querySelectorAll('[data-action="clearCaptionMemory"]')) {
      button.addEventListener("click", () => this.clearCaptionMemory(true).catch((error) => {
        this.status = `Memory cleanup failed: ${error.message || error}`;
        this.render();
      }));
    }
    this.overlay.querySelector('[data-action="trainChunk"]')?.addEventListener("click", () => this.trainChunk());
    this.overlay.querySelector('[data-action="browseFiles"]')?.addEventListener("click", () => {
      this.overlay.querySelector("[data-file-input]")?.click();
    });
    this.overlay.querySelector('[data-action="loadLmStudioModels"]')?.addEventListener("click", () => this.loadLmStudioModels());
    this.overlay.querySelector("[data-file-input]")?.addEventListener("change", (event) => {
      this.importFiles(event.target.files || []);
      event.target.value = "";
    });
    this.overlay.querySelector('[data-action="stopTraining"]')?.addEventListener("click", () => {
      this.stopRequested = true;
      this.status = "Stop requested. Studio will stop after the current chunk and sample finish.";
      this.render();
    });
    const hints = settingHints();
    for (const button of this.overlay.querySelectorAll("[data-hint-key]")) {
      button.addEventListener("click", () => {
        const key = button.dataset.hintKey;
        window.alert(`${settingLabel(key)}\n\n${hints[key] || "No extra notes for this setting yet."}`);
      });
    }

    for (const button of this.overlay.querySelectorAll("[data-preset]")) {
      button.addEventListener("click", () => {
        this.currentPreset = button.dataset.preset;
        this.settings = cloneData(this.defaults.presets[this.currentPreset]);
        this.status = `${this.currentPreset} preset loaded.`;
        this.render();
      });
    }

    for (const input of this.overlay.querySelectorAll("[data-setting]")) {
      input.addEventListener("change", () => {
        const key = input.dataset.setting;
        const current = this.settings[key];
        let value = input.value;
        if (value === "true") value = true;
        if (value === "false") value = false;
        this.settings[key] = coerceByExisting(value, current);
      });
    }

    this.overlay.querySelector('[data-bind="auto_continue"]')?.addEventListener("change", (event) => {
      this.autoContinue = Boolean(event.target.checked);
      this.collectForm();
      this.render();
    });
    this.overlay.querySelector('[data-bind="caption_overwrite"]')?.addEventListener("change", (event) => {
      this.captionOverwrite = Boolean(event.target.checked);
    });

    for (const input of this.overlay.querySelectorAll("[data-bind]")) {
      input.addEventListener("input", () => {
        if (input.dataset.bind === "sample_prompt") this.samplePrompt = input.value;
        if (input.dataset.bind === "caption_instructions") this.captionInstructions = input.value;
        if (input.dataset.bind === "caption_user_notes") this.captionUserNotes = input.value;
      });
      input.addEventListener("change", () => {
        this.collectForm();
        if (input.dataset.bind === "caption_runner" || input.dataset.bind === "llm_api_provider") this.render();
      });
    }

    const drop = this.overlay.querySelector("[data-dropzone]");
    drop?.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.stopPropagation();
      drop.classList.add("drag");
    });
    drop?.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop?.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      drop.classList.remove("drag");
      this.importFiles(event.dataTransfer?.files || []);
    });
  }

  collectForm() {
    for (const input of this.overlay?.querySelectorAll("[data-setting]") || []) {
      const key = input.dataset.setting;
      const current = this.settings[key];
      let value = input.value;
      if (value === "true") value = true;
      if (value === "false") value = false;
      this.settings[key] = coerceByExisting(value, current);
    }
    const captionInstructions = this.overlay?.querySelector('[data-bind="caption_instructions"]')?.value ?? this.captionInstructions ?? "";
    const captionUserNotes = this.overlay?.querySelector('[data-bind="caption_user_notes"]')?.value ?? this.captionUserNotes ?? "";
    const aspectRatio = this.overlay?.querySelector('[data-bind="aspect_ratio"]')?.value ?? this.aspectRatio ?? "3:4 (Portrait Standard)";
    const samplePrompt = this.overlay?.querySelector('[data-bind="sample_prompt"]')?.value ?? this.samplePrompt ?? "";
    const sampleModelSettings = {
      diffusion_model: this.overlay?.querySelector('[data-bind="sample_diffusion_model"]')?.value || this.sampleModelSettings.diffusion_model || "",
      text_encoder: this.overlay?.querySelector('[data-bind="sample_text_encoder"]')?.value || this.sampleModelSettings.text_encoder || "",
      vae: this.overlay?.querySelector('[data-bind="sample_vae"]')?.value || this.sampleModelSettings.vae || "",
    };
    this.aspectRatio = aspectRatio;
    this.samplePrompt = samplePrompt;
    this.sampleModelSettings = sampleModelSettings;
    this.captionInstructions = captionInstructions;
    this.captionUserNotes = captionUserNotes;
    this.captionRunner = this.overlay?.querySelector('[data-bind="caption_runner"]')?.value || this.captionRunner || "builtin";
    this.captionOverwrite = Boolean(this.overlay?.querySelector('[data-bind="caption_overwrite"]')?.checked);
    this.captionGemmaModel = this.overlay?.querySelector('[data-bind="caption_gemma_model"]')?.value || this.captionGemmaModel || "";
    this.captionMmprojFile = this.overlay?.querySelector('[data-bind="caption_mmproj_file"]')?.value || this.captionMmprojFile || "";
    this.lmStudioBaseUrl = this.overlay?.querySelector('[data-bind="lmstudio_base_url"]')?.value || this.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    this.lmStudioModel = this.overlay?.querySelector('[data-bind="lmstudio_model"]')?.value || this.lmStudioModel || "";
    this.lmStudioApiKey = this.overlay?.querySelector('[data-bind="lmstudio_api_key"]')?.value || this.lmStudioApiKey || "";
    this.llmApiProvider = this.overlay?.querySelector('[data-bind="llm_api_provider"]')?.value || this.llmApiProvider || "openai";
    this.llmApiModel = this.overlay?.querySelector('[data-bind="llm_api_model"]')?.value || this.llmApiModel || "";
    this.llmApiKey = this.overlay?.querySelector('[data-bind="llm_api_key"]')?.value || this.llmApiKey || "";
    return {
      project_root: this.overlay?.querySelector('[data-bind="project_root"]')?.value || getWidget(this.node, "project_root")?.value || this.defaults?.project_root || "",
      project_name: this.overlay?.querySelector('[data-bind="project_name"]')?.value || getWidget(this.node, "project_name")?.value || this.defaults?.project_name || "",
      aspect_ratio: aspectRatio,
      sample_prompt: samplePrompt,
      sample_model_settings: sampleModelSettings,
      caption_instructions: captionInstructions,
      caption_user_notes: captionUserNotes,
      caption_final_instructions: this.finalCaptionInstructions(captionInstructions, captionUserNotes),
    };
  }

  finalCaptionInstructions(captionInstructions, captionUserNotes) {
    const base = String(captionInstructions || "").trim();
    const notes = String(captionUserNotes || "").trim();
    if (!notes) return base;
    return `${base}\n\nAdditional user notes / global tags:\n${notes}`;
  }

  async createProject() {
    try {
      const form = this.collectForm();
      this.status = "Creating project...";
      this.render();
      const data = await jsonFetch("/vrgdg/krea2_studio/create_project", {
        project_root: form.project_root,
        project_name: form.project_name,
        preset_name: this.currentPreset,
        settings: this.settings,
        aspect_ratio: form.aspect_ratio,
        sample_prompt: form.sample_prompt,
        sample_model_settings: form.sample_model_settings,
        caption_instructions: form.caption_instructions,
        caption_user_notes: form.caption_user_notes,
        caption_final_instructions: form.caption_final_instructions,
        caption_llm_settings: this.captionLlmSettings(),
      });
      this.project = data.project;
      this.rememberProject(this.project);
      this.status = `Project ready: ${this.project.project_dir}`;
      this.render();
    } catch (error) {
      this.status = `Project error: ${error.message || error}`;
      this.render();
    }
  }

  async saveProject() {
    if (!this.project?.project_dir) return;
    const form = this.collectForm();
    const data = await jsonFetch("/vrgdg/krea2_studio/save_project", {
      project_dir: this.project.project_dir,
      preset_name: this.currentPreset,
      settings: this.settings,
      aspect_ratio: form.aspect_ratio,
      sample_prompt: form.sample_prompt,
      sample_model_settings: form.sample_model_settings,
      caption_instructions: form.caption_instructions,
      caption_user_notes: form.caption_user_notes,
      caption_final_instructions: form.caption_final_instructions,
      caption_llm_settings: this.captionLlmSettings(),
      custom_presets: this.project.custom_presets || {},
    });
    this.project = data.project;
    this.rememberProject(this.project);
  }

  async saveProjectWithStatus() {
    try {
      if (!this.project?.project_dir) {
        this.status = "No project is loaded yet. Open Project... first.";
        this.render();
        return;
      }
      await this.saveProject();
      this.status = `Project saved: ${this.project.project_dir}`;
      this.render();
    } catch (error) {
      this.status = `Save project error: ${error.message || error}`;
      this.render();
    }
  }

  recentProjects() {
    try {
      const parsed = JSON.parse(localStorage.getItem(RECENT_PROJECTS_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  rememberProject(project) {
    const dir = String(project?.project_dir || "").trim();
    if (!dir) return;
    const item = {
      project_dir: dir,
      project_name: String(project?.project_name || dir.split(/[\\/]/).pop() || "Krea2Studio"),
      updated_at: new Date().toISOString(),
    };
    const recent = [item, ...this.recentProjects().filter((entry) => String(entry.project_dir || "") !== dir)].slice(0, 12);
    localStorage.setItem(RECENT_PROJECTS_KEY, JSON.stringify(recent));
  }

  async loadProject(projectDir) {
    const data = await jsonFetch("/vrgdg/krea2_studio/load_project", { project_dir: projectDir });
    this.project = data.project;
    this.currentPreset = this.project.preset_name || this.currentPreset || "Fast";
    this.settings = cloneData(this.project.settings || this.defaults.presets[this.currentPreset] || this.defaults.presets.Fast);
    this.aspectRatio = this.project.aspect_ratio || "3:4 (Portrait Standard)";
    this.samplePrompt = this.project.sample_prompt || this.defaults.sample_prompt || "";
    this.sampleModelSettings = {
      diffusion_model: this.project.sample_model_settings?.diffusion_model || this.defaults.sample_model_defaults?.diffusion_model || this.sampleModelChoices.diffusion_models?.[0] || "",
      text_encoder: this.project.sample_model_settings?.text_encoder || this.defaults.sample_model_defaults?.text_encoder || this.sampleModelChoices.text_encoders?.[0] || "",
      vae: this.project.sample_model_settings?.vae || this.defaults.sample_model_defaults?.vae || this.sampleModelChoices.vae?.[0] || "",
    };
    this.captionInstructions = this.project.caption_instructions || this.defaults.caption_instructions || "";
    this.captionUserNotes = this.project.caption_user_notes || "";
    this.__captionSettingsAppliedFor = "";
    this.applyProjectCaptionSettings();
    this.rememberProject(this.project);
    this.status = `Loaded project: ${this.project.project_dir}`;
    this.render();
  }

  async listProjects(projectRoot) {
    const data = await jsonFetch("/vrgdg/krea2_studio/list_projects", { project_root: projectRoot });
    return Array.isArray(data.projects) ? data.projects : [];
  }

  async openProjectWindow() {
    const form = this.collectForm();
    const root = form.project_root || this.defaults.project_root || "";
    let listed = [];
    try {
      listed = await this.listProjects(root);
    } catch (error) {
      console.warn("[VRGDG] Could not list Krea 2 projects:", error);
    }
    const recent = this.recentProjects();
    const byDir = new Map();
    for (const item of [...recent, ...listed]) {
      const dir = String(item.project_dir || "").trim();
      if (dir && !byDir.has(dir)) byDir.set(dir, item);
    }
    const projects = Array.from(byDir.values());

    const modal = document.createElement("div");
    modal.className = "vrgdg-krea2-overlay";
    modal.style.zIndex = "10050";
    modal.innerHTML = `
      <div class="vrgdg-krea2-shell" style="width:min(820px,calc(100vw - 28px));height:min(760px,calc(100vh - 28px));">
        <div class="vrgdg-krea2-top">
          <div>
            <div class="vrgdg-krea2-title">Krea 2 Project</div>
            <div class="vrgdg-krea2-subtitle">Create a new Studio project or load a previous one.</div>
          </div>
          <button class="vrgdg-krea2-icon" data-project-action="close" title="Close">x</button>
        </div>
        <div class="vrgdg-krea2-work">
          <div class="vrgdg-krea2-section">
            <h3>Create New</h3>
            <div class="vrgdg-krea2-field"><label>Project parent folder</label><input data-project-root value="${esc(root)}"></div>
            <div class="vrgdg-krea2-field"><label>Project name</label><input data-project-name value="${esc(form.project_name || this.defaults.project_name || "Krea2Studio")}"></div>
            <div class="vrgdg-krea2-actions">
              <button class="vrgdg-krea2-btn primary" data-project-action="create">Create Project</button>
              <button class="vrgdg-krea2-btn" data-project-action="refresh">Refresh List</button>
            </div>
          </div>
          <div class="vrgdg-krea2-section">
            <h3>Previous Projects</h3>
            <div class="vrgdg-krea2-field"><label>Custom project folder</label><input data-project-custom value=""></div>
            <div class="vrgdg-krea2-actions" style="margin-bottom:10px;">
              <button class="vrgdg-krea2-btn" data-project-action="loadCustom">Load Custom Folder</button>
            </div>
            <div data-project-list>
              ${projects.length ? projects.map((project) => `
                <button class="vrgdg-krea2-btn" data-project-dir="${esc(project.project_dir)}" style="width:100%;text-align:left;margin-bottom:8px;">
                  ${esc(project.project_name || project.project_dir)}
                  <div class="vrgdg-krea2-hint">${esc(project.project_dir)}${project.completed_steps ? ` | ${esc(project.completed_steps)}/${esc(project.total_target_steps || "?")} steps` : ""}</div>
                </button>
              `).join("") : `<div class="vrgdg-krea2-hint">No previous projects found in this folder yet.</div>`}
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    const close = () => modal.remove();
    modal.querySelector('[data-project-action="close"]')?.addEventListener("click", close);
    modal.querySelector('[data-project-action="create"]')?.addEventListener("click", async () => {
      const rootInput = modal.querySelector("[data-project-root]")?.value || root;
      const nameInput = modal.querySelector("[data-project-name]")?.value || "Krea2Studio";
      const rootWidget = getWidget(this.node, "project_root");
      const nameWidget = getWidget(this.node, "project_name");
      if (rootWidget) rootWidget.value = rootInput;
      if (nameWidget) nameWidget.value = nameInput;
      await this.createProject();
      close();
    });
    modal.querySelector('[data-project-action="refresh"]')?.addEventListener("click", async () => {
      close();
      await this.openProjectWindow();
    });
    modal.querySelector('[data-project-action="loadCustom"]')?.addEventListener("click", async () => {
      const custom = modal.querySelector("[data-project-custom]")?.value || "";
      if (!custom.trim()) return;
      await this.loadProject(custom);
      close();
    });
    for (const button of modal.querySelectorAll("[data-project-dir]")) {
      button.addEventListener("click", async () => {
        await this.loadProject(button.dataset.projectDir);
        close();
      });
    }
  }

  async importFiles(fileList) {
    try {
      const files = Array.from(fileList);
      this.lastDroppedFiles = files.map((file) => file.name || "(unnamed file)");
      if (!files.length) {
        this.status = "No files came through from that drop. Try Browse Files as a fallback.";
        this.render();
        return;
      }
      if (!this.project?.project_dir) {
        this.status = `Received ${files.length} file(s). Creating project before import...`;
        this.render();
        await this.createProject();
        if (!this.project?.project_dir) return;
      }
      this.status = `Importing ${files.length} file(s)...`;
      this.render();
      const form = new FormData();
      form.append("project_dir", this.project.project_dir);
      for (const file of files) form.append("files", file, file.name);
      const response = await api.fetchApi("/vrgdg/krea2_studio/import_files", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok || !data?.ok) throw new Error(data?.error || `HTTP ${response.status}`);
      this.project = data.project;
      const imageCount = (data.saved || []).filter((item) => item.type === "image").length;
      const captionCount = (data.saved || []).filter((item) => item.type === "caption").length;
      const orphanCount = data.manifest?.orphan_captions?.length || 0;
      this.status = `Imported ${imageCount} image(s) and ${captionCount} matched caption(s) as image_### files.${orphanCount ? ` ${orphanCount} orphan caption(s) were listed in import_manifest.json.` : ""}`;
      this.render();
    } catch (error) {
      this.status = `Import error: ${error.message || error}`;
      this.render();
    }
  }

  async captionPlaceholder() {
    let progress = null;
    try {
      await this.saveProject();
      this.status = "Generating missing captions...";
      this.render();
      progress = this.openProgressWindow(
        "Generating Captions",
        "Preparing image list and sending the caption request to the selected vision LLM..."
      );
      progress.onStop(async () => {
        progress.set("Stop requested. Sending Comfy interrupt, asking Krea Studio captioning to cancel, and clearing Gemma memory...");
        try {
          await api.fetchApi("/interrupt", { method: "POST" }).catch(() => null);
          await jsonFetch("/vrgdg/krea2_studio/cancel_captions", {});
          await this.clearCaptionMemory(false);
          progress.set("Stop request sent. If Gemma is inside one long model call, it may finish that current image before the backend returns.");
        } catch (error) {
          progress.set(`Stop request hit an error:\n${error.message || error}`);
        }
      });
      progress.onClear(async () => {
        progress.set("Clearing VRGDG Gemma/GGUF caches and CUDA memory...");
        try {
          await this.clearCaptionMemory(false);
          progress.set("Memory cleanup request finished.");
        } catch (error) {
          progress.set(`Memory cleanup failed:\n${error.message || error}`);
        }
      });
      const form = this.collectForm();
      progress.set(
        `Runner: ${this.captionRunner === "lm_studio" ? "LM Studio" : this.captionRunner === "llm_api" ? "LLM API" : "Gemma Local"}\n` +
        `Project: ${this.project.project_dir}\n\n` +
        "The backend is creating .txt files for images that do not already have captions."
      );
      const data = await jsonFetch("/vrgdg/krea2_studio/generate_captions_placeholder", {
        project_dir: this.project.project_dir,
        caption_final_instructions: form.caption_final_instructions,
        caption_runner: this.captionRunner,
        overwrite_existing: this.captionOverwrite,
        model_file: this.captionGemmaModel,
        mmproj_file: this.captionMmprojFile,
        lmstudio_base_url: this.lmStudioBaseUrl,
        lmstudio_model: this.lmStudioModel,
        lmstudio_api_key: this.lmStudioApiKey,
        llm_api_provider: this.llmApiProvider,
        llm_api_model: this.llmApiModel,
        llm_api_key: this.llmApiKey,
      });
      this.project = data.project || this.project;
      this.status = data.status;
      const created = Array.isArray(data.created) ? data.created : [];
      const createdLines = created.slice(0, 10).map((item) => `${item.caption_file}: ${item.caption}`).join("\n");
      progress.done(`${data.status}\n\n${createdLines || "No new captions were needed."}`, "Captioning Complete");
      this.render();
    } catch (error) {
      this.status = `Caption placeholder error: ${error.message || error}`;
      progress?.done(`Caption generation failed:\n${error.message || error}`, "Captioning Failed", true);
      this.render();
    }
  }

  async clearCaptionMemory(updateStatus = true) {
    const data = await jsonFetch("/vrgdg/krea2_studio/clear_memory", {});
    if (updateStatus) {
      this.status = data.status || "Memory cleanup complete.";
      this.render();
    }
    return data;
  }

  async loadLmStudioModels() {
    try {
      this.collectForm();
      this.status = "Loading LM Studio models...";
      this.render();
      const data = await jsonFetch("/vrgdg/krea2_studio/lm_studio_models", {
        lmstudio_base_url: this.lmStudioBaseUrl,
        lmstudio_api_key: this.lmStudioApiKey,
      });
      const models = Array.isArray(data.models) ? data.models.map((item) => String(item || "").trim()).filter(Boolean) : [];
      if (!models.length) throw new Error("LM Studio returned no models.");
      this.lmStudioModel = models[0];
      this.status = `Loaded ${models.length} LM Studio model(s). Selected ${this.lmStudioModel}.`;
      this.render();
    } catch (error) {
      this.status = `LM Studio model load failed: ${error.message || error}`;
      this.render();
    }
  }

  async savePreset() {
    if (!this.project) {
      this.status = "Create a project before saving custom presets.";
      this.render();
      return;
    }
    this.collectForm();
    const name = window.prompt("Preset name", `${this.currentPreset} Copy`);
    if (!name) return;
    this.project.custom_presets = this.project.custom_presets || {};
    this.project.custom_presets[name] = cloneData(this.settings);
    await this.saveProject();
    this.status = `Saved preset: ${name}`;
    this.render();
  }

  async trainChunk() {
    let progress = null;
    try {
      if (this.isTraining) return;
      this.isTraining = true;
      this.stopRequested = false;
      if (!this.project?.project_dir) await this.createProject();
      if (!this.project?.project_dir) return;
      await this.saveProject();
      progress = this.openProgressWindow(
        "Training Krea 2 LoRA",
        "Preparing project and starting the first training chunk..."
      );
      progress.onStop(async () => {
        this.stopRequested = true;
        progress.set("Stop requested. Studio will stop after the current chunk and sample finish. Sending Comfy interrupt too...");
        await api.fetchApi("/interrupt", { method: "POST" }).catch(() => null);
      });
      progress.onClear(async () => {
        progress.set("Clearing VRGDG Gemma/GGUF caches and CUDA memory...");
        try {
          await this.clearCaptionMemory(false);
          progress.set("Memory cleanup request finished. Training subprocesses may keep their own memory until the current chunk exits.");
        } catch (error) {
          progress.set(`Memory cleanup failed:\n${error.message || error}`);
        }
      });

      while (true) {
        const completed = Number(this.project?.completed_steps || 0);
        const target = Number(this.settings.total_target_steps || this.project?.total_target_steps || 0);
        if (target > 0 && completed >= target) {
          this.status = `Training complete at ${completed}/${target} steps.`;
          this.render();
          progress.set(`Training already complete at ${completed}/${target} steps. Creating XYZ plot...`);
          await this.createXyz(true);
          progress.done(`Training complete at ${completed}/${target} steps.\nXYZ plot is ready.`, "Training Complete");
          break;
        }

        this.status = `Training chunk ${completed}/${target || "?"}. This can take a while...`;
        progress.set(
          `Training chunk\n` +
          `Current steps: ${completed}/${target || "?"}\n` +
          `Chunk size: ${this.settings.steps_per_run || "?"} steps\n\n` +
          "The backend is running musubi. Detailed per-step progress is still in the terminal/log for now."
        );
        this.render();
        const data = await this.trainOneChunk();
        this.project = data.project;
        this.status = `Chunk finished at step ${data.result.completed_steps}. Sampling...`;
        progress.set(`Chunk finished at step ${data.result.completed_steps}/${data.result.total_target_steps}.\nGenerating sample image...`);
        this.render();
        await this.sampleLatest(data.result.latest_lora_path, data.result.completed_steps, progress);

        const newCompleted = Number(this.project?.completed_steps || data.result.completed_steps || 0);
        const newTarget = Number(this.project?.total_target_steps || this.settings.total_target_steps || 0);
        if (newTarget > 0 && newCompleted >= newTarget) {
          this.status = `Training complete at ${newCompleted}/${newTarget} steps.`;
          this.render();
          progress.set(`Training complete at ${newCompleted}/${newTarget} steps.\nCreating XYZ plot...`);
          await this.createXyz(true);
          progress.done(`Training complete at ${newCompleted}/${newTarget} steps.\nXYZ plot is ready.`, "Training Complete");
          break;
        }
        if (this.stopRequested) {
          this.status = `Stopped after step ${newCompleted}.`;
          progress.done(`Stopped after step ${newCompleted}.\nThe current sample has been saved.`, "Stopped");
          this.render();
          break;
        }
        if (!this.autoContinue) {
          this.status = `Sample saved for step ${newCompleted}. Auto-continue is off so you can review the LoRA.`;
          progress.done(`Sample saved for step ${newCompleted}.\nAuto-continue is off so you can review the LoRA.`, "Chunk Complete");
          this.render();
          break;
        }
        progress.set(`Sample saved for step ${newCompleted}.\nContinuing to next chunk...`);
      }
    } catch (error) {
      this.status = `Training error: ${error.message || error}`;
      progress?.done(`Training failed:\n${error.message || error}`, "Training Failed", true);
      this.render();
    } finally {
      this.isTraining = false;
      this.stopRequested = false;
      this.render();
    }
  }

  async trainOneChunk() {
    return jsonFetch("/vrgdg/krea2_studio/train_chunk", {
      project_dir: this.project.project_dir,
      settings: this.settings,
      sample_prompt: this.project.sample_prompt,
      aspect_ratio: this.project.aspect_ratio,
    });
  }

  async sampleLatest(loraPath, step, progress = null) {
    const build = await jsonFetch("/vrgdg/krea2_studio/build_sample_prompt", {
      project_dir: this.project.project_dir,
      lora_path: loraPath,
      aspect_ratio: this.project.aspect_ratio,
      sample_prompt: this.project.sample_prompt,
      sample_model_settings: this.project.sample_model_settings || this.sampleModelSettings,
      strength_model: 1,
    });
    if (!build?.prompt) throw new Error("The hidden sample workflow did not return a prompt to queue.");
    const promptId = await queuePrompt(build.prompt);
    const images = await waitForImageOutput(promptId, (message) => {
      this.status = message;
      progress?.set(`Sampling step ${step}...\n${message}`);
      const statusEl = this.overlay?.querySelector(".vrgdg-krea2-status");
      if (statusEl) statusEl.textContent = message;
    });
    if (!images.length) throw new Error("The hidden sample workflow finished without returning an image.");
    const saved = await jsonFetch("/vrgdg/krea2_studio/save_sample", {
      project_dir: this.project.project_dir,
      step,
      image: images[0],
    });
    this.project = saved.project;
    this.status = `Sample saved for step ${step}.`;
    progress?.set(`Sample saved for step ${step}.\n${saved.sample?.path || ""}`);
    this.render();
  }

  async createXyz(automatic = false) {
    try {
      if (!this.project?.project_dir) return;
      this.status = automatic ? "Training complete. Creating XYZ plot..." : "Creating XYZ plot...";
      this.render();
      const data = await jsonFetch("/vrgdg/krea2_studio/create_xyz", { project_dir: this.project.project_dir });
      this.project = data.project;
      this.status = automatic ? `Training complete. XYZ plot created: ${data.xyz_path}` : `XYZ plot created: ${data.xyz_path}`;
      this.render();
    } catch (error) {
      this.status = `XYZ error: ${error.message || error}`;
      this.render();
    }
  }

  close() {
    this.overlay?.remove();
    this.overlay = null;
  }
}

function ensureStudioButton(node) {
  const existing = (node.widgets || []).find((w) => w?.type === "button" && w?.name === "Open Krea 2 Studio");
  if (existing) {
    existing.serialize = false;
    return;
  }
  const button = node.addWidget("button", "Open Krea 2 Studio", null, () => {
    const studio = new Krea2Studio(node);
    studio.open().catch((error) => window.alert(`Krea 2 Studio failed to open: ${error.message || error}`));
  });
  button.serialize = false;
}

app.registerExtension({
  name: "vrgdg.Krea2LoraStudio",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      ensureStudioButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = origOnConfigure?.apply(this, arguments);
      ensureStudioButton(this);
      return result;
    };
  },
});
