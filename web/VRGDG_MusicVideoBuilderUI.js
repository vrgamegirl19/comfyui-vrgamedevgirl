import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "./VRGDG_MusicVideoPromptCreatorUI.js";

const NODE_NAME = "VRGDG_MusicVideoBuilderUI";
const BUILDER_UI_VERSION = "welcome-startup-2026-05-20";
const HIDDEN_WIDGETS = new Set(["audio_path", "project_folder", "session_path", "srt_path"]);
const DEFAULT_I2V_UNET = "LTX-2.3-22B-distilled-1.1-Q6_K.gguf";
const BAD_I2V_UNET_ALIASES = new Set(["LTX-2.3-22B-distilled-11-Q6_K.gguf"]);
const TIMELINE_HEIGHT = 210;
const TIMELINE_OVERLAY_TOP = 24;
const TIMELINE_OVERLAY_HEIGHT = 50;
const TIMELINE_SEGMENT_TOP = 86;
const TIMELINE_SEGMENT_HEIGHT = 62;
const TIMELINE_SCENE_AUDIO_TOP = TIMELINE_SEGMENT_TOP + TIMELINE_SEGMENT_HEIGHT + 10;
const TIMELINE_SCENE_AUDIO_HEIGHT = 28;
const TIMELINE_WAVE_TOP = 98;
const FLUX_GEMMA_TIMEOUT_MS = 30 * 60 * 1000;
const DEFAULT_NON_VISION_GEMMA_MODEL = "supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf";
const NB_IMAGE_MODELS = ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"];
const DEFAULT_NB_IMAGE_MODEL = "gemini-3-pro-image-preview";
const LTX_MODEL_DOWNLOADS = [
  { label: "LTX GGUF", url: "https://huggingface.co/Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF/tree/main" },
  { label: "Video VAE", url: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
  { label: "Gemma Clip", url: "https://huggingface.co/Sikaworld1990/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition-Ltx-2/resolve/main/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors" },
  { label: "Text Projection", url: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/text_encoders" },
  { label: "Latent Upscaler", url: "https://huggingface.co/prince-canuma/LTX-2.3-distilled/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" },
  { label: "Audio VAE", url: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
];
const ZIMAGE_MODEL_DOWNLOADS = [
  { label: "Z-Image Turbo", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" },
  { label: "Qwen CLIP", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" },
  { label: "Z-Image VAE", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" },
];
const FLUX_KLEIN_9B_MODEL_DOWNLOADS = [
  { label: "9B diffusion model", url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" },
  { label: "9B Qwen CLIP", url: "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" },
  { label: "9B VAE", url: "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors" },
];
const FLUX_KLEIN_4B_MODEL_DOWNLOADS = [
  { label: "4B diffusion model", url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors" },
  { label: "4B Qwen CLIP", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" },
  { label: "4B VAE", url: "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" },
];
const ERNIE_MODEL_DOWNLOADS = [
  { label: "Ernie diffusion model", url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/diffusion_models/ernie-image-turbo.safetensors" },
  { label: "Ministral text encoder", url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ministral-3-3b.safetensors" },
  { label: "Ernie VAE", url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/vae/flux2-vae.safetensors" },
];
const LLM_MODEL_DOWNLOADS = [
  { label: "SuperGemma GGUF", url: "https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2/resolve/main/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf" },
  { label: "Gemma Vision GGUF", url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" },
  { label: "Vision mmproj", url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf" },
];
const MODEL_FOLDER_HINTS = {
  "LLM / Vision": `ComfyUI/
models/
  LLM/
    supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf
    gemma-4-26B-A4B-it-UD-IQ2_M.gguf
    mmproj-BF16.gguf`,
  "ZImage": `ComfyUI/
models/
  text_encoders/
    qwen_3_4b.safetensors
  diffusion_models/
    z_image_turbo_bf16.safetensors
  vae/
    ae.safetensors`,
  "Flux/Klein 9B": `ComfyUI/
models/
  diffusion_models/
    flux-2-klein-9b-fp8.safetensors
  text_encoders/
    qwen_3_8b_fp8mixed.safetensors
  vae/
    full_encoder_small_decoder.safetensors`,
  "Flux/Klein 4B": `ComfyUI/
models/
  text_encoders/
    qwen_3_4b.safetensors
  diffusion_models/
    flux-2-klein-4b-fp8.safetensors
  vae/
    flux2-vae.safetensors`,
  "Ernie Image": `ComfyUI/
models/
  diffusion_models/
    ernie-image-turbo.safetensors
  text_encoders/
    ministral-3-3b.safetensors
  vae/
    flux2-vae.safetensors`,
  "LTX 2.3": `ComfyUI/
models/
  diffusion_models/
    ltx-2.3-distilled_1.1-Q6_k.gguf
  text_encoders/
    ltx-2.3-text_projection_bf16.safetensors
    abliterated-sikaworld-high-fidelity-edition.safetensors
  vae/
    LTX2.3_video_vae_bf16.safetensors
    LTX2.3_audio_vae_bf16.safetensors
  latent_upscale_models/
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors`,
};
const WAVEFORM_MODES = {
  small: { label: "Small wave", height: 150, gain: 1 },
  medium: { label: "Medium wave", height: 190, gain: 1.35 },
  large: { label: "Large wave", height: 240, gain: 1.85 },
};

function getWidget(node, name) {
  return (node?.widgets || []).find((widget) => widget?.name === name);
}

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
  const index = (node.widgets || []).indexOf(widget);
  if (Array.isArray(node.widgets_values) && index >= 0) node.widgets_values[index] = value;
  app.graph?.setDirtyCanvas?.(true, true);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgBuilderOriginalType")) {
    widget.__vrgdgBuilderOriginalType = widget.type;
    widget.__vrgdgBuilderOriginalComputeSize = widget.computeSize;
    widget.__vrgdgBuilderOriginalDraw = widget.draw;
  }
  widget.serialize = true;
  widget.hidden = !visible;
  if (visible) {
    widget.type = widget.__vrgdgBuilderOriginalType;
    if (widget.__vrgdgBuilderOriginalComputeSize) widget.computeSize = widget.__vrgdgBuilderOriginalComputeSize;
    else delete widget.computeSize;
    if (widget.__vrgdgBuilderOriginalDraw) widget.draw = widget.__vrgdgBuilderOriginalDraw;
    else delete widget.draw;
    return;
  }
  widget.type = "hidden";
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function hideInternalWidgets(node) {
  for (const widget of node?.widgets || []) {
    if (HIDDEN_WIDGETS.has(widget?.name)) setWidgetVisible(widget, false);
  }
  node?.setSize?.([420, 96]);
  app.graph?.setDirtyCanvas?.(true, true);
}

function makeButton(label, kind = "neutral") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.style.cssText = `
    border: 1px solid ${kind === "primary" ? "#0891b2" : "#3f3f46"};
    border-radius: 6px;
    background: ${kind === "primary" ? "#06b6d4" : "#27272a"};
    color: ${kind === "primary" ? "#082f49" : "#f4f4f5"};
    font-size: 12px;
    font-weight: 800;
    padding: 8px 11px;
    cursor: pointer;
  `;
  return button;
}

function makeInput(value = "", type = "text") {
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
  return input;
}

function makeCheckbox(label, checked = false) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;align-items:center;gap:8px;font-size:12px;color:#f4f4f5;font-weight:800;";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(checked);
  wrapper.append(input, document.createTextNode(label));
  return { wrapper, input };
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
  select.value = value;
  return select;
}

function makeSearchableLoraPicker(value = "[none]") {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:4px;position:relative;z-index:1;";
  const input = makeInput(value || "[none]");
  const list = document.createElement("div");
  list.style.cssText = "display:none;width:100%;max-height:180px;overflow:auto;border:1px solid #3f3f46;border-radius:6px;background:#18181b;box-shadow:0 8px 18px rgba(0,0,0,.32);";
  wrapper.append(input, list);
  return { wrapper, input, list, options: [], matches: [], activeIndex: -1 };
}

function makeField(label, control) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;color:#d4d4d8;font-weight:700;";
  const text = document.createElement("span");
  text.textContent = label;
  wrapper.append(text, control);
  return wrapper;
}

function makePickerField(label, input, button) {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  row.append(input, button);
  return makeField(label, row);
}

function makeEditField(label, input, button) {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  button.style.padding = "8px 10px";
  row.append(input, button);
  return makeField(label, row);
}

function makeMiniButton(label) {
  const button = makeButton(label);
  button.style.padding = "5px 8px";
  button.style.fontSize = "10px";
  button.style.borderRadius = "5px";
  return button;
}

function makeSettingsSection(title, children = [], open = true) {
  const details = document.createElement("details");
  details.open = Boolean(open);
  details.style.cssText = "border:1px solid #303038;border-radius:7px;background:#18181b;overflow:visible;";
  const summary = document.createElement("summary");
  summary.textContent = title;
  summary.style.cssText = "cursor:pointer;list-style:none;padding:10px 10px;font-size:12px;font-weight:900;color:#e4e4e7;border-bottom:1px solid #27272a;";
  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:9px;";
  body.append(...children);
  details.append(summary, body);
  return details;
}

function makeSettingsPanel(children = []) {
  const panel = document.createElement("div");
  panel.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #303038;border-radius:7px;background:#18181b;padding:9px;";
  panel.append(...children);
  return panel;
}

function makeSubTabs(tabs = []) {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const tabBar = document.createElement("div");
  tabBar.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:6px;position:sticky;top:44px;z-index:2;background:#202024;padding-bottom:2px;";
  const panels = document.createElement("div");
  panels.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const buttons = [];
  const setActive = (value) => {
    for (const item of tabs) {
      const active = item.value === value;
      item.content.style.display = active ? "flex" : "none";
    }
    for (const button of buttons) {
      const active = button.dataset.value === value;
      button.style.background = active ? "#06b6d4" : "#27272a";
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
  };
  for (const tab of tabs) {
    const button = makeButton(tab.label);
    button.dataset.value = tab.value;
    button.onclick = () => setActive(tab.value);
    buttons.push(button);
    tabBar.append(button);
    panels.append(tab.content);
  }
  wrapper.append(tabBar, panels);
  setActive(tabs[0]?.value || "");
  return { wrapper, setActive };
}

function toast(message, isError = false) {
  const element = document.createElement("div");
  element.textContent = message;
  element.style.cssText = `
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 100003;
    max-width: min(560px, calc(100vw - 36px));
    border: 1px solid ${isError ? "#991b1b" : "#155e75"};
    border-radius: 8px;
    background: ${isError ? "#450a0a" : "#083344"};
    color: ${isError ? "#fecaca" : "#cffafe"};
    padding: 12px 14px;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 18px 60px rgba(0,0,0,.45);
  `;
  document.body.appendChild(element);
  setTimeout(() => element.remove(), 6500);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function createProgressWindow(title, options = {}) {
  const box = document.createElement("div");
  const zIndex = Number(options.zIndex || 100004);
  box.style.cssText = `
    position: fixed;
    left: 50%;
    top: 54px;
    transform: translateX(-50%);
    z-index: ${zIndex};
    width: min(850px, calc(100vw - 560px));
    min-width: 520px;
    border: 1px solid #155e75;
    border-radius: 8px;
    background: #0f172a;
    color: #cffafe;
    box-shadow: 0 22px 70px rgba(0,0,0,.55);
    overflow: hidden;
    font-family: sans-serif;
  `;
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid #155e75;background:#083344;";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-size:13px;font-weight:900;";
  const headerActions = document.createElement("div");
  headerActions.style.cssText = "display:flex;align-items:center;gap:8px;";
  const minimize = makeButton("Min");
  minimize.title = "Minimize progress window";
  minimize.style.padding = "5px 8px";
  const close = makeButton("Close");
  close.style.padding = "5px 8px";
  headerActions.append(minimize, close);
  header.append(heading, headerActions);
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
  const restore = document.createElement("button");
  restore.type = "button";
  restore.textContent = title;
  restore.title = "Restore progress window";
  restore.style.cssText = `position:fixed;left:50%;top:54px;transform:translateX(-50%);z-index:${zIndex};display:none;max-width:min(850px,calc(100vw - 560px));min-width:260px;border:1px solid #155e75;border-radius:8px;background:#083344;color:#cffafe;padding:8px 12px;font-size:12px;font-weight:900;box-shadow:0 16px 48px rgba(0,0,0,.45);cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`;
  document.body.append(restore);
  const removeAll = () => {
    box.remove();
    restore.remove();
  };
  minimize.onclick = () => {
    box.style.display = "none";
    restore.style.display = "block";
  };
  restore.onclick = () => {
    restore.style.display = "none";
    box.style.display = "block";
  };
  close.onclick = removeAll;
  return {
    set(message, percent = null) {
      body.textContent = message;
      if (percent !== null) barInner.style.width = `${Math.max(5, Math.min(100, percent))}%`;
      const firstLine = String(message || title).split(/\r?\n/)[0] || title;
      restore.textContent = `${title}: ${firstLine}`;
    },
    setHtml(html, percent = null) {
      body.innerHTML = html;
      if (percent !== null) barInner.style.width = `${Math.max(5, Math.min(100, percent))}%`;
      restore.textContent = title;
    },
    close(delay = 0) {
      setTimeout(removeAll, delay);
    },
  };
}

function showFinalVideoReadyModal(videoPath) {
  const path = String(videoPath || "").trim();
  if (!path) return;
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.58);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(680px,calc(100vw - 48px));border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cffafe;box-shadow:0 22px 70px rgba(0,0,0,.6);overflow:hidden;font-family:sans-serif;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid #155e75;background:#083344;";
  const title = document.createElement("div");
  title.textContent = "Final Video Ready";
  title.style.cssText = "font-size:14px;font-weight:900;";
  const close = makeButton("Close");
  close.style.padding = "6px 10px";
  header.append(title, close);
  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:12px;padding:14px;";
  const message = document.createElement("div");
  message.textContent = "Your stitched final video is ready.";
  message.style.cssText = "font-size:12px;color:#e0f2fe;";
  const pathBox = document.createElement("div");
  pathBox.textContent = path;
  pathBox.style.cssText = "border:1px solid #334155;border-radius:6px;background:#020617;color:#bae6fd;padding:10px;font-size:11px;line-height:1.35;white-space:pre-wrap;overflow-wrap:anywhere;";
  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const open = makeButton("Open Video", "primary");
  const dismiss = makeButton("Close");
  actions.append(open, dismiss);
  body.append(message, pathBox, actions);
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  const finish = () => backdrop.remove();
  close.onclick = finish;
  dismiss.onclick = finish;
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) finish();
  });
  open.onclick = async () => {
    open.disabled = true;
    open.textContent = "Opening...";
    try {
      await postJson("/vrgdg/music_builder/open_local_file", { path }, 30000);
      finish();
    } catch (error) {
      open.disabled = false;
      open.textContent = "Open Video";
      toast(String(error?.message || error), true);
    }
  };
}

function showModelDownloadModal() {
  const groups = [
    { title: "LLM / Vision", note: "Use SuperGemma for text prompting. Use Gemma Vision GGUF plus mmproj for image-reference prompting.", downloads: LLM_MODEL_DOWNLOADS },
    { title: "ZImage", note: "Core ZImage Turbo diffusion model, Qwen text encoder, and VAE.", downloads: ZIMAGE_MODEL_DOWNLOADS },
    { title: "Flux/Klein 9B", note: "9B is higher quality. 4B is smaller and lighter.", downloads: FLUX_KLEIN_9B_MODEL_DOWNLOADS },
    { title: "Flux/Klein 4B", note: "4B is smaller and lighter.", downloads: FLUX_KLEIN_4B_MODEL_DOWNLOADS },
    { title: "Ernie Image", note: "Ernie diffusion model, Ministral text encoder, and VAE.", downloads: ERNIE_MODEL_DOWNLOADS },
    { title: "LTX 2.3", note: "High-quality image and video generation model.", downloads: LTX_MODEL_DOWNLOADS },
  ];
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:28px;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(1320px,calc(100vw - 56px));max-height:calc(100vh - 56px);overflow:auto;border:1px solid #155e75;border-radius:10px;background:#0f172a;color:#e4e4e7;box-shadow:0 22px 80px rgba(0,0,0,.6);";
  const header = document.createElement("div");
  header.style.cssText = "position:sticky;top:0;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:22px 24px;border-bottom:1px solid #155e75;background:#083344;z-index:1;";
  const title = document.createElement("div");
  title.textContent = "Download Models";
  title.style.cssText = "font-size:24px;font-weight:900;color:#e0f2fe;";
  const close = makeButton("Close");
  close.style.padding = "12px 16px";
  close.style.fontSize = "18px";
  header.append(title, close);
  const body = document.createElement("div");
  body.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;padding:18px 20px 20px;";
  for (const group of groups) {
    const card = document.createElement("div");
    card.style.cssText = "display:flex;flex-direction:column;gap:14px;border:1px solid #334155;border-radius:10px;background:#111827;padding:18px;";
    const titleRow = document.createElement("div");
    titleRow.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
    const groupTitle = document.createElement("div");
    groupTitle.textContent = group.title;
    groupTitle.style.cssText = "font-size:22px;font-weight:900;color:#f8fafc;";
    const folderButton = makeMiniButton("Folders");
    folderButton.style.fontSize = "13px";
    folderButton.style.padding = "8px 10px";
    folderButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const pre = document.createElement("pre");
      pre.textContent = MODEL_FOLDER_HINTS[group.title] || "No folder location listed yet.";
      pre.style.cssText = "white-space:pre-wrap;margin:0;padding:12px;border:1px solid #334155;border-radius:8px;background:#020617;color:#bae6fd;font-size:12px;line-height:1.45;overflow:auto;";
      showInfoModal({
        title: `${group.title} Folder Locations`,
        lines: [
          "Place the files here, then restart ComfyUI if the dropdowns do not refresh.",
          pre,
        ],
      });
    });
    titleRow.append(groupTitle, folderButton);
    const note = document.createElement("div");
    note.textContent = group.note;
    note.style.cssText = "font-size:17px;line-height:1.35;color:#c7d2fe;";
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;";
    for (const item of group.downloads) {
      const button = makeMiniButton(item.label);
      button.style.borderColor = "#2563eb";
      button.style.background = "#1d4ed8";
      button.style.color = "#eff6ff";
      button.style.fontWeight = "900";
      button.style.fontSize = "16px";
      button.style.padding = "12px 16px";
      button.style.borderRadius = "7px";
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        window.open(item.url, "_blank", "noopener,noreferrer");
      });
      buttons.append(button);
    }
    card.append(titleRow, note, buttons);
    body.append(card);
  }
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  close.onclick = () => backdrop.remove();
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) backdrop.remove();
  });
}

function showTextInputModal({ title, label, value = "", placeholder = "", confirmLabel = "Continue" } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = title || "Choose Value";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const fieldLabel = document.createElement("label");
    fieldLabel.textContent = label || "Value";
    fieldLabel.style.cssText = "font-size:12px;font-weight:900;color:#d4d4d8;";
    const input = makeInput(value || "");
    input.placeholder = placeholder || "";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton(confirmLabel, "primary");
    const finish = (result) => {
      backdrop.remove();
      resolve(result);
    };
    cancel.onclick = () => finish(null);
    confirm.onclick = () => finish(input.value.trim());
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") finish(input.value.trim());
      if (event.key === "Escape") finish(null);
    });
    actions.append(cancel, confirm);
    box.append(heading, fieldLabel, input, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    input.focus();
    input.select();
  });
}

function showInfoModal({ title, lines = [], confirmLabel = "Got it" } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = title || "Info";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:9px;font-size:13px;color:#d4d4d8;line-height:1.45;";
    for (const line of lines) {
      if (line instanceof HTMLElement) {
        body.append(line);
      } else {
        const item = document.createElement("div");
        item.textContent = line;
        body.append(item);
      }
    }
    const confirm = makeButton(confirmLabel, "primary");
    confirm.onclick = () => {
      backdrop.remove();
      resolve(true);
    };
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) confirm.click();
    });
    box.append(heading, body, confirm);
    backdrop.append(box);
    document.body.append(backdrop);
    confirm.focus();
  });
}

function showAddSegmentPositionModal(sceneLabel = "selected scene") {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(460px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Add Segment";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const body = document.createElement("div");
    body.textContent = `Add a new segment before or after ${sceneLabel}?`;
    body.style.cssText = "font-size:13px;line-height:1.45;color:#d4d4d8;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
    const before = makeButton("Before", "primary");
    const after = makeButton("After", "primary");
    const cancel = makeButton("Cancel");
    const finish = (result) => {
      backdrop.remove();
      resolve(result);
    };
    before.onclick = () => finish("before");
    after.onclick = () => finish("after");
    cancel.onclick = () => finish(null);
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(null);
    });
    actions.append(before, after, cancel);
    box.append(heading, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    before.focus();
  });
}

function pickProjectSessionFile() {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,application/json";
    input.style.display = "none";
    document.body.append(input);
    input.onchange = () => {
      const file = input.files?.[0] || null;
      input.remove();
      if (!file) {
        resolve("");
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const session = JSON.parse(String(reader.result || "{}"));
          const folder = String(session.project_folder || "").trim();
          if (!folder) throw new Error("That JSON does not include a project_folder. Choose the project's vrgdg_builder_session.json file.");
          resolve(folder);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = () => reject(new Error("Could not read the selected project session JSON."));
      reader.readAsText(file);
    };
    input.click();
  });
}

function showSaveProjectAsModal(defaultName = "") {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Save Project As";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const note = document.createElement("div");
    note.textContent = "Enter a project name. The copy will be saved as a new folder in ComfyUI output so it will not overwrite the current project.";
    note.style.cssText = "font-size:12px;color:#d4d4d8;line-height:1.45;";
    const input = makeInput(defaultName || "");
    input.placeholder = "New project name";
    const advanced = document.createElement("details");
    advanced.style.cssText = "border:1px solid #27272a;border-radius:6px;background:#18181b;padding:8px;";
    const summary = document.createElement("summary");
    summary.textContent = "Advanced: full folder path";
    summary.style.cssText = "cursor:pointer;font-size:12px;font-weight:900;color:#bae6fd;";
    const advancedInput = makeInput("");
    advancedInput.placeholder = "Optional full folder path";
    advancedInput.style.marginTop = "8px";
    const advancedNote = document.createElement("div");
    advancedNote.textContent = "Browsers cannot safely expose a real folder path from a normal folder picker, so paste a full path here only if you need a custom location.";
    advancedNote.style.cssText = "margin-top:6px;font-size:11px;color:#a1a1aa;line-height:1.35;";
    advanced.append(summary, advancedInput, advancedNote);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton("Save Copy", "primary");
    const finish = (result) => {
      backdrop.remove();
      resolve(result);
    };
    cancel.onclick = () => finish(null);
    confirm.onclick = () => finish((advancedInput.value || input.value || "").trim());
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") finish((advancedInput.value || input.value || "").trim());
      if (event.key === "Escape") finish(null);
    });
    actions.append(cancel, confirm);
    box.append(heading, note, input, advanced, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    input.focus();
    input.select();
  });
}

function showWelcomeProjectModal(projects = []) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:min(780px,calc(100vh - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Welcome to Video Creator";
    heading.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    const note = document.createElement("div");
    note.textContent = "Create a new project or open an existing one to get started.";
    note.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const create = makeButton("Create New Project", "primary");
    const close = makeButton("Close");
    actions.append(create, close);

    const listTitle = document.createElement("div");
    listTitle.textContent = "Existing projects";
    listTitle.style.cssText = "font-size:12px;font-weight:900;color:#bae6fd;margin-top:4px;";
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:7px;overflow:auto;max-height:min(420px,46vh);padding-right:3px;";

    const finish = (result) => {
      backdrop.remove();
      resolve(result);
    };
    create.onclick = () => finish({ action: "new" });
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
        row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:center;border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;";
        const info = document.createElement("div");
        info.style.cssText = "display:flex;flex-direction:column;gap:4px;min-width:0;";
        const name = document.createElement("div");
        name.textContent = project.name || "Unnamed project";
        name.style.cssText = "font-size:13px;font-weight:900;color:#f8fafc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const meta = document.createElement("div");
        const updated = project.updated ? new Date(project.updated * 1000).toLocaleString() : "unknown date";
        meta.textContent = `${project.scene_count || 0} scene${Number(project.scene_count || 0) === 1 ? "" : "s"} | ${updated}`;
        meta.style.cssText = "font-size:11px;color:#a1a1aa;";
        const path = document.createElement("div");
        path.textContent = project.project_folder || "";
        path.style.cssText = "font-size:11px;color:#67e8f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        info.append(name, meta, path);
        const open = makeButton("Open", "primary");
        const del = makeButton("Delete");
        del.style.borderColor = "#7f1d1d";
        del.style.color = "#fecaca";
        open.onclick = () => finish({ action: "load", project_folder: project.project_folder || "" });
        del.onclick = async (event) => {
          event.preventDefault();
          event.stopPropagation();
          const ok = await showDeleteProjectConfirm(project);
          if (!ok) return;
          try {
            del.disabled = true;
            del.textContent = "Deleting...";
            await postJson("/vrgdg/music_builder/delete_project", { project_folder: project.project_folder }, 120000);
            row.remove();
            if (!list.children.length) {
              const empty = document.createElement("div");
              empty.textContent = "No existing projects were found in the ComfyUI output folder.";
              empty.style.cssText = "border:1px dashed #3f3f46;border-radius:7px;padding:14px;color:#a1a1aa;font-size:12px;text-align:center;";
              list.append(empty);
            }
          } catch (error) {
            del.disabled = false;
            del.textContent = "Delete";
            toast(String(error?.message || error), true);
          }
        };
        row.append(info, open, del);
        list.append(row);
      }
    }

    box.append(heading, note, actions, listTitle, list);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.tabIndex = -1;
    backdrop.focus();
  });
}

function showLoadProjectModal(projects = []) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(800px,calc(100vw - 40px));max-height:min(820px,calc(100vh - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Load Project";
    heading.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Open a recent project from the ComfyUI output folder, or enter a custom project folder path.";
    note.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const customRow = document.createElement("div");
    customRow.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:end;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;";
    const customInput = makeInput("");
    customInput.placeholder = "Custom project folder path...";
    const customOpen = makeButton("Open Custom", "primary");
    customRow.append(makeField("Custom path", customInput), customOpen);
    const listTitle = document.createElement("div");
    listTitle.textContent = "Existing projects";
    listTitle.style.cssText = "font-size:12px;font-weight:900;color:#bae6fd;";
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:8px;overflow:auto;max-height:min(470px,48vh);padding-right:3px;";
    const finish = (result) => {
      backdrop.remove();
      resolve(result);
    };
    close.onclick = () => finish(null);
    customOpen.onclick = () => {
      const path = String(customInput.value || "").trim();
      if (path) finish({ action: "load", project_folder: path });
    };
    customInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") customOpen.click();
    });
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
        row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:center;border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;";
        const info = document.createElement("div");
        info.style.cssText = "display:flex;flex-direction:column;gap:4px;min-width:0;";
        const name = document.createElement("div");
        name.textContent = project.name || "Unnamed project";
        name.style.cssText = "font-size:13px;font-weight:900;color:#f8fafc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const meta = document.createElement("div");
        const updated = project.updated ? new Date(project.updated * 1000).toLocaleString() : "unknown date";
        meta.textContent = `${project.scene_count || 0} scene${Number(project.scene_count || 0) === 1 ? "" : "s"} | ${updated}`;
        meta.style.cssText = "font-size:11px;color:#a1a1aa;";
        const path = document.createElement("div");
        path.textContent = project.project_folder || "";
        path.style.cssText = "font-size:11px;color:#67e8f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        info.append(name, meta, path);
        const open = makeButton("Open", "primary");
        const del = makeButton("Delete");
        del.style.borderColor = "#7f1d1d";
        del.style.color = "#fecaca";
        open.onclick = () => finish({ action: "load", project_folder: project.project_folder || "" });
        del.onclick = async (event) => {
          event.preventDefault();
          event.stopPropagation();
          const ok = await showDeleteProjectConfirm(project);
          if (!ok) return;
          try {
            del.disabled = true;
            del.textContent = "Deleting...";
            await postJson("/vrgdg/music_builder/delete_project", { project_folder: project.project_folder }, 120000);
            row.remove();
            if (!list.children.length) {
              const empty = document.createElement("div");
              empty.textContent = "No existing projects were found in the ComfyUI output folder.";
              empty.style.cssText = "border:1px dashed #3f3f46;border-radius:7px;padding:14px;color:#a1a1aa;font-size:12px;text-align:center;";
              list.append(empty);
            }
          } catch (error) {
            del.disabled = false;
            del.textContent = "Delete";
            toast(String(error?.message || error), true);
          }
        };
        row.append(info, open, del);
        list.append(row);
      }
    }
    box.append(header, note, customRow, listTitle, list);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.tabIndex = -1;
    backdrop.focus();
  });
}

function showDeleteProjectConfirm(project) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Delete Project?";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
    const body = document.createElement("div");
    body.textContent = "This deletes the full project folder from disk. This cannot be undone.";
    body.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const name = document.createElement("div");
    name.textContent = project?.name || "Unnamed project";
    name.style.cssText = "font-size:13px;font-weight:900;color:#f8fafc;";
    const path = document.createElement("div");
    path.textContent = project?.project_folder || "";
    path.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton("Delete Project");
    confirm.style.borderColor = "#7f1d1d";
    confirm.style.background = "#991b1b";
    confirm.style.color = "#fee2e2";
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    cancel.onclick = () => finish(false);
    confirm.onclick = () => finish(true);
    actions.append(cancel, confirm);
    box.append(heading, body, name, path, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

async function postJson(url, payload, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await api.fetchApi(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data?.ok) throw new Error(String(data?.error || `Request failed (${response.status})`));
    return data;
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Request timed out. The backend may still be processing that file.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function getJson(url) {
  const response = await api.fetchApi(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) throw new Error(String(data?.error || `Request failed (${response.status})`));
  return data;
}

function makeImageViewUrl(image) {
  const params = new URLSearchParams();
  params.set("filename", image.filename || "");
  params.set("type", image.type || "output");
  if (image.subfolder) params.set("subfolder", image.subfolder);
  params.set("rand", String(Date.now()));
  return `/view?${params.toString()}`;
}

function makeEditorImageUrl(path) {
  return `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}&rand=${Date.now()}`;
}

function makeEditorVideoUrl(path) {
  return `/vrgdg/video_editor/video?path=${encodeURIComponent(path)}&rand=${Date.now()}`;
}

function extractImagesFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const images = [];
  for (const output of Object.values(outputs)) {
    if (Array.isArray(output?.images)) {
      for (const image of output.images) images.push(image);
    }
  }
  return images;
}

function extractPromptErrorFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const messages = [];
  const status = root?.status || {};
  if (status.status_str && !/success|completed/i.test(String(status.status_str))) {
    messages.push(`status: ${status.status_str}`);
  }
  const candidates = [
    root?.error,
    status?.error,
    status?.exception_message,
    status?.message,
    ...(Array.isArray(status?.messages) ? status.messages : []),
  ];
  const visit = (value) => {
    if (value == null) return;
    if (typeof value === "string") {
      if (value.trim() && !/execution_(start|cached|success)/i.test(value)) messages.push(value.trim());
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (typeof value === "object") {
      for (const key of ["exception_message", "error", "message", "node_id", "node_type", "class_type"]) {
        if (value[key] != null) visit(value[key]);
      }
    }
  };
  candidates.forEach(visit);
  return [...new Set(messages)].join("\n");
}

function promptHistoryFinished(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  if (!root || !Object.keys(root || {}).length) return false;
  const status = String(root?.status?.status_str || "").toLowerCase();
  if (status) return /success|completed|error|failed/i.test(status);
  return Boolean(root?.outputs);
}

function extractTextFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const values = [];
  for (const output of Object.values(outputs)) {
    const text = output?.text ?? output?.ui?.text;
    if (Array.isArray(text)) values.push(...text);
    else if (text != null) values.push(text);
  }
  return values.flat(Infinity).map((value) => String(value ?? "")).filter((value) => value.trim());
}

function resolveComfyVideoPath(video) {
  const params = video?.params || video || {};
  if (params.fullpath) return String(params.fullpath);
  const filename = params.filename || video?.filename || "";
  const subfolder = params.subfolder || video?.subfolder || "";
  if (filename && subfolder && /^[A-Za-z]:[\\/]/.test(String(subfolder))) {
    return `${String(subfolder).replace(/[\\/]+$/, "")}\\${filename}`;
  }
  return "";
}

function extractVideosFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const videos = [];
  for (const output of Object.values(outputs)) {
    for (const key of ["gifs", "videos", "animated"]) {
      if (Array.isArray(output?.[key])) {
        for (const video of output[key]) videos.push(video);
      }
    }
  }
  return videos;
}

async function waitForVideos(promptId, onStatus, shouldCancel) {
  const started = Date.now();
  while (Date.now() - started < 40 * 60 * 1000) {
    if (shouldCancel?.()) throw new Error("Stopped by user.");
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const promptError = extractPromptErrorFromHistory(data, promptId);
    if (promptError) throw new Error(`Scene video workflow failed:\n${promptError}`);
    const videos = extractVideosFromHistory(data, promptId);
    if (videos.length) return videos;
    if (promptHistoryFinished(data, promptId)) {
      throw new Error("Scene video workflow finished, but no video output was found in history.");
    }
    onStatus?.("Waiting for scene video...");
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error("Timed out waiting for the scene video.");
}

async function waitForImages(promptId, onStatus, shouldCancel) {
  const started = Date.now();
  while (Date.now() - started < 20 * 60 * 1000) {
    if (shouldCancel?.()) throw new Error("Stopped by user.");
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const promptError = extractPromptErrorFromHistory(data, promptId);
    if (promptError) throw new Error(`Image workflow failed:\n${promptError}`);
    const images = extractImagesFromHistory(data, promptId);
    if (images.length) return images;
    if (promptHistoryFinished(data, promptId)) {
      throw new Error("Image workflow finished, but no image output was found in history.");
    }
    onStatus?.("Waiting for image output...");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Timed out waiting for the ZImage preview.");
}

async function waitForText(promptId, onStatus, shouldCancel) {
  const started = Date.now();
  while (Date.now() - started < 5 * 60 * 1000) {
    if (shouldCancel?.()) throw new Error("Stopped by user.");
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const promptError = extractPromptErrorFromHistory(data, promptId);
    if (promptError) throw new Error(`Text workflow failed:\n${promptError}`);
    const text = extractTextFromHistory(data, promptId);
    if (text.length) return text;
    if (promptHistoryFinished(data, promptId)) {
      throw new Error("Text workflow finished, but no text output was found in history.");
    }
    onStatus?.("Waiting for cleanup result...");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for the cleanup result.");
}

function queueListsFromStatus(data) {
  return {
    running: Array.isArray(data?.queue_running) ? data.queue_running : [],
    pending: Array.isArray(data?.queue_pending) ? data.queue_pending : [],
  };
}

async function getComfyQueueStatus() {
  const response = await api.fetchApi("/queue");
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`Queue status request failed (${response.status})`);
  return queueListsFromStatus(data);
}

async function waitForComfyQueueIdle(onStatus, options = {}) {
  const timeoutMs = Number(options.timeoutMs || 10 * 60 * 1000);
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (options.shouldCancel?.()) throw new Error("Stopped by user.");
    const status = await getComfyQueueStatus();
    if (!status.running.length && !status.pending.length) return status;
    onStatus?.(`Waiting for ComfyUI queue to become idle...\nRunning: ${status.running.length}\nPending: ${status.pending.length}`);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for ComfyUI queue to become idle. Nothing new was queued.");
}

async function clearPendingComfyQueue() {
  const response = await api.fetchApi("/queue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clear: true }),
  }).catch(() => null);
  if (!response) return;
  if (!response.ok) {
    console.warn("[VRGDG Music Builder] Could not clear pending ComfyUI queue:", response.status);
  }
}

async function cancelComfyExecutionAndWaitIdle(onStatus, options = {}) {
  onStatus?.("Interrupting ComfyUI and clearing pending queue...");
  await api.fetchApi("/interrupt", { method: "POST" }).catch(() => null);
  await clearPendingComfyQueue();
  return await waitForComfyQueueIdle(onStatus, {
    timeoutMs: options.timeoutMs || 5 * 60 * 1000,
    shouldCancel: options.shouldCancel,
  });
}

async function queueWorkflowPrompt(prompt, options = {}) {
  await waitForComfyQueueIdle(options.onStatus, {
    timeoutMs: options.idleTimeoutMs,
    shouldCancel: options.shouldCancel,
  });
  const clientId = api.clientId || app?.clientId || crypto.randomUUID();
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.error) {
    throw new Error(data?.error?.message || data?.error || `Queue failed (${response.status})`);
  }
  if (data?.node_errors && Object.keys(data.node_errors).length) {
    const details = Object.entries(data.node_errors).map(([nodeId, error]) => {
      const messages = [error?.class_type, error?.exception_message, error?.message]
        .filter(Boolean)
        .map((value) => String(value));
      const inputErrors = Array.isArray(error?.errors)
        ? error.errors.map((item) => item?.message || item).filter(Boolean).map((value) => String(value))
        : [];
      return `Node ${nodeId}: ${[...messages, ...inputErrors].join(" | ") || JSON.stringify(error)}`;
    }).join("\n");
    throw new Error(`Workflow validation failed:\n${details}`);
  }
  return data;
}

function formatTime(value) {
  const total = Math.max(0, Number(value || 0));
  const minutes = Math.floor(total / 60);
  const seconds = Math.floor(total % 60);
  const hundredths = Math.floor((total - Math.floor(total)) * 100);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(hundredths).padStart(2, "0")}`;
}

function formatDurationSeconds(start, end) {
  const duration = Math.max(0, Number(end || 0) - Number(start || 0));
  return duration.toFixed(duration >= 10 ? 1 : 2);
}

function audioUrl(path) {
  return `/vrgdg/music_builder/audio?path=${encodeURIComponent(path)}&v=${Date.now()}`;
}

function newSegment(start = 0, end = 4) {
  const now = Date.now();
  return {
    id: `seg_${now}_${Math.floor(Math.random() * 10000)}`,
    track: "base",
    start,
    end,
    label: "New scene",
    notes: "",
    i2v_notes: "",
    t2i_prompt: "",
    enhance_notes: "",
    enhance_prompt: "",
    i2v_prompt: "",
    ref_image_path: "",
    use_vision_reference: false,
    use_i2v_vision_reference: true,
    custom_image_path: "",
    custom_image_data: "",
    custom_image_name: "",
    image: null,
    image_history: [],
    image_history_index: -1,
    preview_mode: "image",
    video_path: "",
    video_history: [],
    video_history_index: -1,
    video_output: null,
    video_status: "none",
    custom_audio_path: "",
    custom_audio_name: "",
    custom_audio_duration: 0,
    custom_audio_full_duration: 0,
    custom_audio_timeline_start: start,
    custom_audio_source_start: 0,
    custom_audio_peaks: [],
    custom_audio_beats: [],
    flux_image_ingredients: [],
    flux_notes: "",
    flux_prompt: "",
    nb_notes: "",
    nb_prompt: "",
    use_scene_zimage_settings: false,
    zimage_settings: null,
    use_scene_ernie_image_settings: false,
    ernie_image_settings: null,
    use_scene_flux_klein_settings: false,
    flux_klein_settings: null,
    use_scene_nb_image_settings: false,
    nb_image_settings: null,
    use_scene_i2v_video_settings: false,
    i2v_video_settings: null,
    source: "manual",
  };
}

function sortSegments(segments) {
  segments.sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
}

function openBuilder(node) {
  console.log(`[VRGDG Music Builder] UI version ${BUILDER_UI_VERSION}`);
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;";
  const shell = document.createElement("div");
  const normalShellStyle = `
    width: min(1800px, calc(100vw - 24px));
    height: min(920px, calc(100vh - 24px));
    display: grid;
    grid-template-rows: auto minmax(0,1fr) minmax(230px, 34vh);
    background: #18181b;
    color: #fafafa;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 24px 90px rgba(0,0,0,.55);
  `;
  const fullscreenShellStyle = `
    width: 100vw;
    height: 100vh;
    display: grid;
    grid-template-rows: auto minmax(0,1fr) minmax(230px, 34vh);
    background: #18181b;
    color: #fafafa;
    border: 0;
    border-radius: 0;
    overflow: hidden;
    box-shadow: none;
  `;
  shell.style.cssText = normalShellStyle;
  let builderFullscreen = false;

  const topbar = document.createElement("div");
  topbar.style.cssText = "position:relative;display:flex;gap:10px;align-items:center;justify-content:flex-end;flex-wrap:wrap;padding:12px;border-bottom:1px solid #27272a;background:#202024;";
  const audioInput = makeInput(String(getWidget(node, "audio_path")?.value || ""));
  const projectInput = makeInput(String(getWidget(node, "project_folder")?.value || ""));
  const srtInput = makeInput("");
  const pickAudioButton = makeButton("Pick");
  const pickSrtButton = makeButton("Pick");
  pickAudioButton.textContent = "Choose Audio";
  pickSrtButton.textContent = "Choose SRT";
  const settingsButton = makeButton("Settings");
  const loadButton = makeButton("Load Audio", "primary");
  const loadSrtButton = makeButton("Load SRT", "primary");
  const menuButton = makeButton("Menu");
  const loadSessionButton = makeButton("Load Project");
  const loadLastProjectButton = makeButton("Load Last Project");
  const newProjectButton = makeButton("New Project");
  const saveProjectAsButton = makeButton("Save Project As");
  const saveButton = makeButton("Quick Save", "primary");
  const autoSaveControl = makeCheckbox("Auto save", true);
  autoSaveControl.wrapper.style.cssText += "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:7px 10px;";
  const fullscreenButton = makeButton("Fullscreen");
  fullscreenButton.title = "Expand the Video Creator to fill the browser window without closing or resetting anything.";
  const closeButton = makeButton("Close");
  closeButton.onclick = () => overlay.remove();
  const promptCreatorButton = makeButton("Prompt Creator");
  const autoLoadAllButton = makeButton("Import Data From Prompt Creator");
  const fluxReferenceBuilderButton = makeButton("Reference Builder");
  const promptOptionsButton = makeButton("Prompt Options");
  const gemmaRunnerButton = makeButton("Gemma Runner");
  const clearMemoryButton = makeButton("Clear Memory");
  const renderAllButton = makeButton("Render All");
  const stitchPreviewButton = makeButton("Stitch Preview");
  const gemmaT2IAllButton = makeButton("Gemma T2I All");
  const gemmaVideoAllButton = makeButton("Gemma Video All");
  const zImageAllButton = makeButton("Image All");
  const fullBuildButton = makeButton("Build Full Video");
  const remakeModeButton = makeButton("Remake Mode");
  const stopWorkflowButton = makeButton("Stop");
  const downloadModelsButton = makeButton("Download Models");
  stopWorkflowButton.style.background = "#b91c1c";
  stopWorkflowButton.style.borderColor = "#7f1d1d";
  stopWorkflowButton.style.color = "#fee2e2";
  const menuDropdown = document.createElement("div");
  menuDropdown.style.cssText = "display:none;position:absolute;left:12px;top:52px;z-index:20;min-width:260px;border:1px solid #3f3f46;border-radius:8px;background:#18181b;box-shadow:0 18px 60px rgba(0,0,0,.55);padding:8px;gap:6px;flex-direction:column;";
  const styleMenuItem = (button) => {
    button.style.width = "100%";
    button.style.textAlign = "left";
    button.style.justifyContent = "flex-start";
  };
  for (const button of [newProjectButton, loadSessionButton, loadLastProjectButton, saveProjectAsButton, settingsButton, promptCreatorButton, autoLoadAllButton, gemmaT2IAllButton, gemmaVideoAllButton, zImageAllButton, renderAllButton, stitchPreviewButton, fullBuildButton, remakeModeButton]) {
    styleMenuItem(button);
    menuDropdown.append(button);
  }
  autoSaveControl.wrapper.style.marginTop = "4px";
  menuDropdown.append(autoSaveControl.wrapper);
  const projectActions = document.createElement("div");
  projectActions.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-right:auto;";
  projectActions.append(menuButton, saveButton);
  const batchActions = document.createElement("div");
  batchActions.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-left:1px solid #3f3f46;border-right:1px solid #3f3f46;padding:0 10px;";
  batchActions.style.display = "none";
  const importActions = document.createElement("div");
  importActions.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
  importActions.append(fluxReferenceBuilderButton, gemmaRunnerButton, promptOptionsButton);
  const utilityActions = document.createElement("div");
  utilityActions.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
  utilityActions.append(stopWorkflowButton, downloadModelsButton, clearMemoryButton, fullscreenButton, closeButton);
  topbar.append(projectActions, importActions, batchActions, utilityActions, menuDropdown);

  const main = document.createElement("div");
  main.style.cssText = "display:grid;grid-template-columns:260px 7px minmax(0,1fr) 7px 360px;min-height:0;";
  const segmentList = document.createElement("div");
  segmentList.style.cssText = "overflow:auto;padding:10px;border-right:1px solid #27272a;background:#202024;";
  segmentList.style.gridColumn = "1";
  const leftResizeHandle = document.createElement("div");
  leftResizeHandle.title = "Drag to resize scene list";
  leftResizeHandle.style.cssText = "cursor:col-resize;background:#18181b;border-left:1px solid #27272a;border-right:1px solid #27272a;";
  leftResizeHandle.style.gridColumn = "2";
  const preview = document.createElement("div");
  preview.style.cssText = "display:grid;grid-template-rows:minmax(0,1fr) auto;min-height:0;background:#09090b;";
  preview.style.gridColumn = "3";
  const previewStage = document.createElement("div");
  previewStage.style.cssText = "position:relative;width:100%;height:100%;display:flex;align-items:center;justify-content:center;min-height:0;overflow:hidden;";
  const previewEmpty = document.createElement("div");
  previewEmpty.textContent = "Create a segment, add a T2I prompt, then preview ZImage.";
  previewEmpty.style.cssText = "color:#71717a;font-size:13px;text-align:center;";
  const previewImage = document.createElement("img");
  previewImage.alt = "";
  previewImage.style.cssText = "display:none;max-width:100%;max-height:100%;object-fit:contain;background:#050505;";
  const previewVideo = document.createElement("video");
  previewVideo.controls = true;
  previewVideo.playsInline = true;
  previewVideo.muted = false;
  previewVideo.style.cssText = "display:none;max-width:100%;max-height:100%;object-fit:contain;background:#050505;";
  previewStage.append(previewEmpty, previewImage, previewVideo);
  const customImageFileInput = document.createElement("input");
  customImageFileInput.type = "file";
  customImageFileInput.accept = "image/png,image/jpeg,image/webp";
  customImageFileInput.style.display = "none";
  shell.append(customImageFileInput);
  const visionRefFileInput = document.createElement("input");
  visionRefFileInput.type = "file";
  visionRefFileInput.accept = "image/png,image/jpeg,image/webp";
  visionRefFileInput.style.display = "none";
  shell.append(visionRefFileInput);
  const i2iImageFileInput = document.createElement("input");
  i2iImageFileInput.type = "file";
  i2iImageFileInput.accept = "image/png,image/jpeg,image/webp";
  i2iImageFileInput.style.display = "none";
  shell.append(i2iImageFileInput);
  const projectAudioFileInput = document.createElement("input");
  projectAudioFileInput.type = "file";
  projectAudioFileInput.accept = "audio/wav,audio/mpeg,audio/flac,audio/mp4,audio/ogg,.wav,.mp3,.flac,.m4a,.ogg";
  projectAudioFileInput.style.display = "none";
  shell.append(projectAudioFileInput);
  const projectSrtFileInput = document.createElement("input");
  projectSrtFileInput.type = "file";
  projectSrtFileInput.accept = ".srt,text/plain";
  projectSrtFileInput.style.display = "none";
  shell.append(projectSrtFileInput);
  const inspector = document.createElement("div");
  inspector.style.cssText = "display:flex;flex-direction:column;gap:10px;padding:10px;border-left:1px solid #27272a;background:#202024;overflow:auto;min-height:0;";
  inspector.style.gridColumn = "5";
  const rightResizeHandle = document.createElement("div");
  rightResizeHandle.title = "Drag to resize settings panel";
  rightResizeHandle.style.cssText = "cursor:col-resize;background:#18181b;border-left:1px solid #27272a;border-right:1px solid #27272a;";
  rightResizeHandle.style.gridColumn = "4";
  main.append(segmentList, leftResizeHandle, preview, rightResizeHandle, inspector);

  const labelInput = makeInput("");
  const freezeTimingControl = makeCheckbox("Freeze SRT timing", false);
  const promptJsonInput = makeInput("");
  const importPromptJsonButton = makeButton("Import Prompt JSON", "primary");
  const editPromptJsonButton = makeButton("Edit");
  const i2vMotionJsonInput = makeInput("");
  const importI2VMotionJsonButton = makeButton("Import I2V Motion Notes", "primary");
  const editI2VMotionJsonButton = makeButton("Edit");
  const imageTriggerInput = makeInput("");
  imageTriggerInput.placeholder = "Optional image trigger word or phrase...";
  const useSceneErnieImageSettings = makeCheckbox("Use custom Ernie settings for this scene", false);
  const ernieImageTriggerInput = makeInput("");
  ernieImageTriggerInput.placeholder = imageTriggerInput.placeholder;
  const useSceneFluxKleinSettings = makeCheckbox("Use custom Flux/Klein settings for this scene", false);
  const fluxImageTriggerInput = makeInput("");
  fluxImageTriggerInput.placeholder = imageTriggerInput.placeholder;
  const useSceneNBImageSettings = makeCheckbox("Use custom NanoBanana settings for this scene", false);
  const useSceneI2VVideoSettings = makeCheckbox("Use custom video models/settings/LoRAs for this scene", false);
  const useSceneI2VVideoSettingsNote = document.createElement("div");
  useSceneI2VVideoSettingsNote.textContent = "Applies video model files, LoRAs, LoRA count, pass strengths, FPS, size, trigger phrase, and seed to this scene only.";
  useSceneI2VVideoSettingsNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const videoSettingsScopeNote = document.createElement("div");
  videoSettingsScopeNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const videoTriggerInput = makeInput("");
  videoTriggerInput.placeholder = "Optional video trigger word or phrase...";
  const useVrgdgTextContext = makeCheckbox("Use VRGDG text context files", true);
  const loadVrgdgContextButton = makeButton("Use Default TextFiles Paths", "primary");
  const themeStyleInput = makeInput("");
  const storyIdeaInput = makeInput("");
  const subjectSceneInput = makeInput("");
  const editThemeStyleButton = makeButton("Edit");
  const editStoryIdeaButton = makeButton("Edit");
  const editSubjectSceneButton = makeButton("Edit");
  const zimageSettingsPanel = document.createElement("div");
  zimageSettingsPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const zUnetPicker = makeSearchableLoraPicker("z_image_turbo_bf16.safetensors");
  const zClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const zVaePicker = makeSearchableLoraPicker("ae.safetensors");
  const useSceneZImageSettings = makeCheckbox("Use custom ZImage settings for this scene", false);
  const zFirstTitle = document.createElement("div");
  zFirstTitle.textContent = "First pass (low res)";
  zFirstTitle.style.cssText = "font-size:12px;color:#f4f4f5;font-weight:900;";
  const zFirstGrid = document.createElement("div");
  zFirstGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const zFirstWidth = makeInput("1280", "number");
  const zFirstHeight = makeInput("720", "number");
  zFirstGrid.append(makeField("Width", zFirstWidth), makeField("Height", zFirstHeight));
  const zSecondTitle = document.createElement("div");
  zSecondTitle.textContent = "2nd pass (upscale enhance)";
  zSecondTitle.style.cssText = "font-size:12px;color:#f4f4f5;font-weight:900;";
  const zSecondGrid = document.createElement("div");
  zSecondGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const zSecondWidth = makeInput("1920", "number");
  const zSecondHeight = makeInput("1080", "number");
  zSecondGrid.append(makeField("Width", zSecondWidth), makeField("Height", zSecondHeight));
  const zSeed = makeInput("1", "number");
  const zSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "fixed");
  const zBatchSize = makeInput("1", "number");
  zBatchSize.min = "1";
  zBatchSize.max = "16";
  zBatchSize.step = "1";
  const zSeedGrid = document.createElement("div");
  zSeedGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  zSeedGrid.append(makeField("Seed", zSeed), makeField("Seed mode", zSeedMode));
  const zLoraCount = makeInput("0", "number");
  zLoraCount.min = "0";
  zLoraCount.max = "4";
  const zUseLora = makeCheckbox("Use LoRAs?", false);
  const zLoraPanel = document.createElement("div");
  zLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zLoraRows = document.createElement("div");
  zLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 76px 76px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const firstPassStrength = makeInput("0.5", "number");
    firstPassStrength.step = "0.01";
    const secondPassStrength = makeInput("1", "number");
    secondPassStrength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Pass 1", firstPassStrength), makeField("Pass 2", secondPassStrength));
    zLoraRows.append(row);
    zLoraSlots.push({ row, picker, firstPassStrength, secondPassStrength });
  }
  const zUseImageToImage = makeCheckbox("Use image-to-image?", false);
  const zI2IPanel = document.createElement("div");
  zI2IPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zI2ISlider = document.createElement("input");
  zI2ISlider.type = "range";
  zI2ISlider.min = "1";
  zI2ISlider.max = "8";
  zI2ISlider.step = "1";
  zI2ISlider.value = "5";
  zI2ISlider.style.cssText = "width:100%;accent-color:#22d3ee;";
  const zI2IHint = document.createElement("div");
  zI2IHint.textContent = "1 = more creative, 8 = more like original";
  zI2IHint.style.cssText = "font-size:11px;color:#a1a1aa;";
  const zI2IStartStep = makeInput("5", "number");
  zI2IStartStep.min = "1";
  zI2IStartStep.max = "8";
  zI2IStartStep.step = "1";
  const zI2IPath = makeInput("");
  zI2IPath.placeholder = "Image-to-image source path...";
  zI2IPath.style.display = "none";
  const zI2IDrop = document.createElement("div");
  zI2IDrop.textContent = "Drop an image here, or drag a scene image from the timeline.";
  zI2IDrop.style.cssText = "border:1px dashed #155e75;border-radius:6px;background:#020617;color:#bae6fd;padding:10px;text-align:center;font-size:12px;";
  const zI2IActions = document.createElement("div");
  zI2IActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  const zI2ILoadButton = makeButton("Load I2I Image", "primary");
  zI2IActions.append(zI2ILoadButton);
  zLoraPanel.append(makeField("LoRA count", zLoraCount), zLoraRows);
  zI2IPanel.append(makeField("I2I similarity", zI2ISlider), zI2IHint, makeField("I2I start step", zI2IStartStep), zI2IPath, zI2IDrop, zI2IActions);
  const ernieImagePanel = document.createElement("div");
  ernieImagePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const ernieUnetPicker = makeSearchableLoraPicker("ernie\\ernie-image-turbo.safetensors");
  const ernieClipPicker = makeSearchableLoraPicker("ministral-3-3b.safetensors");
  const ernieVaePicker = makeSearchableLoraPicker("flux\\flux2-vae.safetensors");
  const ernieWidth = makeInput("1280", "number");
  const ernieHeight = makeInput("720", "number");
  const ernieSeed = makeInput("1", "number");
  const ernieSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "fixed");
  const ernieBatchSize = makeInput("1", "number");
  ernieBatchSize.min = "1";
  ernieBatchSize.max = "16";
  ernieBatchSize.step = "1";
  const ernieGrid = document.createElement("div");
  ernieGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  ernieGrid.append(makeField("Width", ernieWidth), makeField("Height", ernieHeight), makeField("Seed", ernieSeed), makeField("Seed mode", ernieSeedMode));
  const ernieUseLora = makeCheckbox("Use LoRAs?", false);
  const ernieLoraPanel = document.createElement("div");
  ernieLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const ernieLoraCount = makeInput("0", "number");
  ernieLoraCount.min = "0";
  ernieLoraCount.max = "4";
  const ernieLoraRows = document.createElement("div");
  ernieLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const ernieLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Strength", strength));
    ernieLoraRows.append(row);
    ernieLoraSlots.push({ row, picker, strength });
  }
  ernieLoraPanel.append(makeField("LoRA count", ernieLoraCount), ernieLoraRows);
  const ernieUseImageToImage = makeCheckbox("Use image-to-image?", false);
  const ernieI2IPanel = document.createElement("div");
  ernieI2IPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const ernieI2ISlider = document.createElement("input");
  ernieI2ISlider.type = "range";
  ernieI2ISlider.min = "1";
  ernieI2ISlider.max = "8";
  ernieI2ISlider.step = "1";
  ernieI2ISlider.value = "5";
  ernieI2ISlider.style.cssText = "width:100%;accent-color:#22d3ee;";
  const ernieI2IHint = document.createElement("div");
  ernieI2IHint.textContent = "1 = more creative, 8 = more like original";
  ernieI2IHint.style.cssText = "font-size:11px;color:#a1a1aa;";
  const ernieI2IStartStep = makeInput("5", "number");
  ernieI2IStartStep.min = "1";
  ernieI2IStartStep.max = "8";
  ernieI2IStartStep.step = "1";
  const ernieI2IPath = makeInput("");
  ernieI2IPath.placeholder = "Image-to-image source path...";
  ernieI2IPath.style.display = "none";
  const ernieI2IDrop = document.createElement("div");
  ernieI2IDrop.textContent = "Drop an image here, or drag a scene image from the timeline.";
  ernieI2IDrop.style.cssText = zI2IDrop.style.cssText;
  const ernieI2IActions = document.createElement("div");
  ernieI2IActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  const ernieI2ILoadButton = makeButton("Load I2I Image", "primary");
  ernieI2IActions.append(ernieI2ILoadButton);
  ernieI2IPanel.append(makeField("I2I similarity", ernieI2ISlider), ernieI2IHint, makeField("I2I start step", ernieI2IStartStep), ernieI2IPath, ernieI2IDrop, ernieI2IActions);
  const fluxKleinPanel = document.createElement("div");
  fluxKleinPanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const useFluxKlein = makeCheckbox("Build image using Flux/Klein?", false);
  const fluxIngredientFileInput = document.createElement("input");
  fluxIngredientFileInput.type = "file";
  fluxIngredientFileInput.accept = "image/png,image/jpeg,image/webp";
  fluxIngredientFileInput.multiple = true;
  fluxIngredientFileInput.style.display = "none";
  shell.append(fluxIngredientFileInput);
  const fluxGlobalIngredientFileInput = document.createElement("input");
  fluxGlobalIngredientFileInput.type = "file";
  fluxGlobalIngredientFileInput.accept = "image/png,image/jpeg,image/webp";
  fluxGlobalIngredientFileInput.multiple = true;
  fluxGlobalIngredientFileInput.style.display = "none";
  shell.append(fluxGlobalIngredientFileInput);
  const useFluxGlobalIngredients = makeCheckbox("Use global image ingredients", false);
  const fluxGlobalIngredientPanel = document.createElement("div");
  fluxGlobalIngredientPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const fluxGlobalIngredientDrop = document.createElement("div");
  fluxGlobalIngredientDrop.innerHTML = `<b>Global image ingredients</b><br><span>Drop character, face, costume, or style references here to use them in every Flux/Klein scene.</span>`;
  fluxGlobalIngredientDrop.style.cssText = "border:1px dashed #7c3aed;border-radius:6px;background:#12091f;color:#ddd6fe;padding:12px;text-align:center;font-size:12px;line-height:1.45;";
  const fluxGlobalIngredientList = document.createElement("div");
  fluxGlobalIngredientList.style.cssText = "display:flex;flex-direction:column;gap:6px;";
  const fluxGlobalIngredientActions = document.createElement("div");
  fluxGlobalIngredientActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const fluxGlobalIngredientButton = makeButton("Upload Global Images", "primary");
  const fluxGlobalIngredientClearButton = makeButton("Clear Globals");
  fluxGlobalIngredientActions.append(fluxGlobalIngredientButton, fluxGlobalIngredientClearButton);
  fluxGlobalIngredientPanel.append(
    fluxGlobalIngredientDrop,
    fluxGlobalIngredientActions,
    fluxGlobalIngredientList,
  );
  const nbUseGlobalIngredients = makeCheckbox("Use global Nano B reference images", false);
  const nbGlobalIngredientPanel = document.createElement("div");
  nbGlobalIngredientPanel.style.cssText = fluxGlobalIngredientPanel.style.cssText;
  const nbGlobalIngredientDrop = document.createElement("div");
  nbGlobalIngredientDrop.innerHTML = `<b>Global Nano B reference images</b><br><span>Drop character, face, costume, or style references here to use them in every Nano B scene.</span>`;
  nbGlobalIngredientDrop.style.cssText = fluxGlobalIngredientDrop.style.cssText;
  const nbGlobalIngredientList = document.createElement("div");
  nbGlobalIngredientList.style.cssText = fluxGlobalIngredientList.style.cssText;
  const nbGlobalIngredientActions = document.createElement("div");
  nbGlobalIngredientActions.style.cssText = fluxGlobalIngredientActions.style.cssText;
  const nbGlobalIngredientButton = makeButton("Upload Global References", "primary");
  const nbGlobalIngredientClearButton = makeButton("Clear Globals");
  nbGlobalIngredientActions.append(nbGlobalIngredientButton, nbGlobalIngredientClearButton);
  nbGlobalIngredientPanel.append(
    nbGlobalIngredientDrop,
    nbGlobalIngredientActions,
    nbGlobalIngredientList,
  );
  const fluxIngredientDrop = document.createElement("div");
  fluxIngredientDrop.innerHTML = `<b>Image ingredients</b><br><span>Drop images here: character, background, props, style references, or anything else Flux/Klein should use.</span>`;
  fluxIngredientDrop.style.cssText = "border:1px dashed #155e75;border-radius:6px;background:#020617;color:#bae6fd;padding:12px;text-align:center;font-size:12px;line-height:1.45;";
  const fluxIngredientList = document.createElement("div");
  fluxIngredientList.style.cssText = "display:flex;flex-direction:column;gap:6px;";
  const fluxIngredientActions = document.createElement("div");
  fluxIngredientActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const fluxIngredientButton = makeButton("Upload Images", "primary");
  const fluxIngredientClearButton = makeButton("Clear Images");
  fluxIngredientActions.append(fluxIngredientButton, fluxIngredientClearButton);
  const fluxNotes = document.createElement("textarea");
  fluxNotes.placeholder = "Optional pose, camera, wardrobe, lighting, or mood notes...";
  fluxNotes.style.cssText = "width:100%;box-sizing:border-box;min-height:72px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  const fluxUseTextOnlyGemmaPrompt = makeCheckbox("Use text-only Gemma for Flux prompts", false);
  const fluxGemmaModelSelect = makeSelect([""], "");
  const fluxMmprojSelect = makeSelect([""], "");
  const fluxPrompt = document.createElement("textarea");
  fluxPrompt.placeholder = "Flux/Klein prompt...";
  fluxPrompt.style.cssText = fluxNotes.style.cssText;
  const fluxUnetPicker = makeSearchableLoraPicker("flux\\flux-2-klein-4b-fp8.safetensors");
  const fluxClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const fluxVaePicker = makeSearchableLoraPicker("flux\\flux2-vae.safetensors");
  const fluxUseLora = makeCheckbox("Use Flux/Klein LoRAs?", false);
  const fluxLoraPanel = document.createElement("div");
  fluxLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const fluxLoraCount = makeInput("0", "number");
  fluxLoraCount.min = "0";
  fluxLoraCount.max = "4";
  const fluxLoraRows = document.createElement("div");
  fluxLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const fluxLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Strength", strength));
    fluxLoraRows.append(row);
    fluxLoraSlots.push({ row, picker, strength });
  }
  fluxLoraPanel.append(makeField("LoRA count", fluxLoraCount), fluxLoraRows);
  const fluxWidth = makeInput("1024", "number");
  const fluxHeight = makeInput("576", "number");
  const fluxSeed = makeInput("100", "number");
  const fluxGrid = document.createElement("div");
  fluxGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  fluxGrid.append(makeField("Width", fluxWidth), makeField("Height", fluxHeight), makeField("Seed", fluxSeed));
  const createFluxPromptButton = makeButton("Gemma Flux Prompt", "primary");
  const previewFluxButton = makeButton("Create with Flux/Klein", "primary");
  const sendFluxPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const nbImagePanel = document.createElement("div");
  nbImagePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const nbApiKey = makeInput("");
  nbApiKey.type = "password";
  nbApiKey.placeholder = "NanoBanana API key...";
  const nbModelSelect = makeSelect(NB_IMAGE_MODELS, DEFAULT_NB_IMAGE_MODEL);
  const nbGemmaModelSelect = makeSelect([""], "");
  const nbMmprojSelect = makeSelect([""], "");
  const nbUseTextOnlyGemmaPrompt = makeCheckbox("Use text-only Gemma for Nano B prompts", false);
  const nbNotes = document.createElement("textarea");
  nbNotes.placeholder = "Optional camera, framing, pose, scene, or edit notes for NanoBanana...";
  nbNotes.style.cssText = fluxNotes.style.cssText;
  const nbPrompt = document.createElement("textarea");
  nbPrompt.placeholder = "NanoBanana prompt...";
  nbPrompt.style.cssText = fluxPrompt.style.cssText;
  const nbIngredientDrop = document.createElement("div");
  nbIngredientDrop.innerHTML = `<b>NanoBanana reference images</b><br><span>Drop character and scene references here. Reference Builder images are also included when enabled for this scene.</span>`;
  nbIngredientDrop.style.cssText = fluxIngredientDrop.style.cssText;
  const nbIngredientList = document.createElement("div");
  nbIngredientList.style.cssText = fluxIngredientList.style.cssText;
  const nbIngredientActions = document.createElement("div");
  nbIngredientActions.style.cssText = fluxIngredientActions.style.cssText;
  const nbIngredientButton = makeButton("Upload References", "primary");
  const nbIngredientClearButton = makeButton("Clear References");
  nbIngredientActions.append(nbIngredientButton, nbIngredientClearButton);
  const createNBPromptButton = makeButton("Gemma NB Prompt", "primary");
  const previewNBButton = makeButton("Create with NanoBanana", "primary");
  const sendNBPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const fluxImageRefsPanel = document.createElement("div");
  fluxImageRefsPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  fluxImageRefsPanel.append(
    useFluxGlobalIngredients.wrapper,
    fluxGlobalIngredientPanel,
    fluxIngredientDrop,
    fluxIngredientActions,
    fluxIngredientList,
  );
  const zEnhancePanel = document.createElement("div");
  zEnhancePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const zEnhanceTitle = document.createElement("div");
  zEnhanceTitle.textContent = "Upscale / Enhance selected image";
  zEnhanceTitle.style.cssText = "font-size:12px;color:#f4f4f5;font-weight:900;";
  const zEnhancePromptPreview = document.createElement("textarea");
  zEnhancePromptPreview.placeholder = "Enhance prompt copied from the selected scene...";
  zEnhancePromptPreview.style.cssText = "width:100%;box-sizing:border-box;min-height:92px;resize:vertical;border:1px solid #27272a;border-radius:6px;background:#18181b;color:#d4d4d8;padding:8px;font-size:11px;line-height:1.35;";
  const zEnhanceGemmaNotes = document.createElement("textarea");
  zEnhanceGemmaNotes.placeholder = "Optional notes for Gemma: what to preserve, improve, restyle, or emphasize from the selected image...";
  zEnhanceGemmaNotes.style.cssText = zEnhancePromptPreview.style.cssText;
  const zEnhanceGemmaModelSelect = makeSelect([""], "");
  const zEnhanceMmprojSelect = makeSelect([""], "");
  const zEnhanceGemmaButton = makeButton("Gemma Enhance Prompt", "primary");
  const zEnhanceUnetPicker = makeSearchableLoraPicker("z_image_turbo_bf16.safetensors");
  const zEnhanceClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const zEnhanceVaePicker = makeSearchableLoraPicker("ae.safetensors");
  const zEnhanceWidth = makeInput("1920", "number");
  const zEnhanceHeight = makeInput("1080", "number");
  const zEnhanceSeed = makeInput("1", "number");
  const zEnhanceSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "randomize");
  const zEnhanceAmount = document.createElement("input");
  zEnhanceAmount.type = "range";
  zEnhanceAmount.min = "1";
  zEnhanceAmount.max = "20";
  zEnhanceAmount.step = "1";
  zEnhanceAmount.value = "8";
  zEnhanceAmount.style.cssText = "width:100%;accent-color:#22d3ee;";
  const zEnhanceAmountValue = document.createElement("div");
  zEnhanceAmountValue.style.cssText = "font-size:11px;color:#a1a1aa;";
  const zEnhanceHint = document.createElement("div");
  zEnhanceHint.textContent = "Higher values keep closer to the original. Lower values are more creative. Adding a ZImage character LoRA can act like a face swap plus enhancement.";
  zEnhanceHint.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const zEnhanceGrid = document.createElement("div");
  zEnhanceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  zEnhanceGrid.append(makeField("Width", zEnhanceWidth), makeField("Height", zEnhanceHeight), makeField("Seed", zEnhanceSeed), makeField("Seed mode", zEnhanceSeedMode));
  const zEnhanceUseLora = makeCheckbox("Use LoRAs?", false);
  const zEnhanceLoraPanel = document.createElement("div");
  zEnhanceLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zEnhanceLoraCount = makeInput("0", "number");
  zEnhanceLoraCount.min = "0";
  zEnhanceLoraCount.max = "4";
  const zEnhanceLoraRows = document.createElement("div");
  zEnhanceLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zEnhanceLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.step = "0.01";
    row.append(makeField(`Enhance LoRA ${slot}`, picker.wrapper), makeField("Strength", strength));
    zEnhanceLoraRows.append(row);
    zEnhanceLoraSlots.push({ row, picker, strength });
  }
  zEnhanceLoraPanel.append(makeField("LoRA count", zEnhanceLoraCount), zEnhanceLoraRows);
  const zEnhanceButton = makeButton("Upscale / Enhance Image", "primary");
  const imageModelChooserWrap = document.createElement("div");
  imageModelChooserWrap.style.cssText = "display:flex;flex-direction:column;gap:5px;min-width:0;";
  const imageModelChooserLabel = document.createElement("div");
  imageModelChooserLabel.textContent = "Model";
  imageModelChooserLabel.style.cssText = "font-size:11px;font-weight:900;color:#bae6fd;letter-spacing:0;";
  const imageModelChooser = document.createElement("div");
  imageModelChooser.style.cssText = "display:flex;gap:6px;overflow-x:auto;overflow-y:hidden;max-width:100%;padding:0 0 3px;scrollbar-width:thin;";
  function makeImageModelCard(label, value) {
    const card = document.createElement("button");
    card.type = "button";
    card.dataset.model = value;
    card.textContent = label;
    card.style.cssText = "height:34px;min-width:72px;flex:0 0 auto;border:1px solid #3f3f46;border-radius:999px;background:#27272a;color:#f4f4f5;font-size:12px;font-weight:900;cursor:pointer;padding:0 10px;white-space:nowrap;";
    return card;
  }
  const zImageCard = makeImageModelCard("ZImage", "zimage");
  const fluxKleinCard = makeImageModelCard("Flux Klein", "flux_klein");
  const nbImageCard = makeImageModelCard("Nano B", "nano_banana");
  const ernieImageCard = makeImageModelCard("Ernie", "ernie_image");
  const zEnhanceCard = makeImageModelCard("Enhance", "z_enhance");
  const loadCustomImageButton = makeImageModelCard("+ Custom", "custom_image");
  loadCustomImageButton.title = "Load a custom image for the selected scene";
  imageModelChooser.append(zImageCard, fluxKleinCard, nbImageCard, ernieImageCard, zEnhanceCard, loadCustomImageButton);
  imageModelChooserWrap.append(imageModelChooserLabel, imageModelChooser);
  const zImageModePanel = document.createElement("div");
  zImageModePanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  const fluxKleinModePanel = document.createElement("div");
  fluxKleinModePanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const ernieImageModePanel = document.createElement("div");
  ernieImageModePanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const startInput = makeInput("0", "number");
  startInput.step = "0.01";
  const endInput = makeInput("4", "number");
  endInput.step = "0.01";
  const notesInput = document.createElement("textarea");
  notesInput.placeholder = "Scene notes for Gemma / prompt direction...";
  notesInput.style.cssText = "width:100%;box-sizing:border-box;min-height:82px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  const i2vNotesInput = document.createElement("textarea");
  i2vNotesInput.placeholder = "Extra video motion notes, camera movement, character movement...";
  i2vNotesInput.style.cssText = notesInput.style.cssText;
  const t2iTextGemmaModelSelect = makeSelect([""], "");
  const gemmaModelSelect = makeSelect([""], "");
  const mmprojSelect = makeSelect([""], "");
  const ernieTextGemmaModelSelect = makeSelect([""], "");
  const ernieGemmaModelSelect = makeSelect([""], "");
  const ernieMmprojSelect = makeSelect([""], "");
  const i2vTextGemmaModelSelect = makeSelect([""], "");
  const i2vGemmaModelSelect = makeSelect([""], "");
  const i2vMmprojSelect = makeSelect([""], "");
  const useVisionReference = makeCheckbox("Use vision reference image?", false);
  const useI2VVisionReference = makeCheckbox("Use image reference for I2V prompt?", true);
  const i2vReferenceNote = document.createElement("div");
  i2vReferenceNote.textContent = "When checked, Gemma looks at the scene image and your video notes to create the I2V prompt. When unchecked, it uses the T2I prompt text and your video notes instead.";
  i2vReferenceNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const useT2VVisionReference = makeCheckbox("Use image reference for T2V Gemma prompt?", false);
  const t2vReferenceNote = document.createElement("div");
  t2vReferenceNote.textContent = "Optional: Gemma looks at a reference image for pose, framing, mood, or visual direction while still creating a text-to-video prompt.";
  t2vReferenceNote.style.cssText = i2vReferenceNote.style.cssText;
  const refImageInput = makeInput("");
  refImageInput.style.display = "none";
  const refImagePanel = document.createElement("div");
  refImagePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const refImageNote = document.createElement("div");
  refImageNote.textContent = "Optional: give Gemma a visual reference for the direction you want.";
  refImageNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const refImageDrop = document.createElement("div");
  refImageDrop.textContent = "Drop a reference image here, or drag a scene image from the timeline.";
  refImageDrop.style.cssText = zI2IDrop.style.cssText;
  const refImageLoadButton = makeButton("Load Reference Image", "primary");
  refImagePanel.append(refImageNote, refImageDrop, refImageLoadButton, refImageInput);
  const createT2IButton = makeButton("Gemma T2I", "primary");
  const createI2VButton = makeButton("Gemma I2V", "primary");
  const sendT2IPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const t2iPrompt = document.createElement("textarea");
  t2iPrompt.placeholder = "Text-to-image prompt...";
  t2iPrompt.style.cssText = notesInput.style.cssText;
  const ernieNotesInput = document.createElement("textarea");
  ernieNotesInput.placeholder = notesInput.placeholder;
  ernieNotesInput.style.cssText = notesInput.style.cssText;
  const ernieT2IPrompt = document.createElement("textarea");
  ernieT2IPrompt.placeholder = t2iPrompt.placeholder;
  ernieT2IPrompt.style.cssText = t2iPrompt.style.cssText;
  const ernieUseVisionReference = makeCheckbox("Use vision reference image?", false);
  const ernieRefImagePanel = document.createElement("div");
  ernieRefImagePanel.style.cssText = refImagePanel.style.cssText;
  const ernieRefImageNote = document.createElement("div");
  ernieRefImageNote.textContent = refImageNote.textContent;
  ernieRefImageNote.style.cssText = refImageNote.style.cssText;
  const ernieRefImageDrop = document.createElement("div");
  ernieRefImageDrop.textContent = refImageDrop.textContent;
  ernieRefImageDrop.style.cssText = refImageDrop.style.cssText;
  const ernieRefImageLoadButton = makeButton("Load Reference Image", "primary");
  ernieRefImagePanel.append(ernieRefImageNote, ernieRefImageDrop, ernieRefImageLoadButton);
  const t2vRefImagePanel = document.createElement("div");
  t2vRefImagePanel.style.cssText = refImagePanel.style.cssText;
  const t2vRefImageNote = document.createElement("div");
  t2vRefImageNote.textContent = "Drop/load the image Gemma should look at while writing the T2V prompt.";
  t2vRefImageNote.style.cssText = refImageNote.style.cssText;
  const t2vRefImageDrop = document.createElement("div");
  t2vRefImageDrop.textContent = refImageDrop.textContent;
  t2vRefImageDrop.style.cssText = refImageDrop.style.cssText;
  const t2vRefImageLoadButton = makeButton("Load Reference Image", "primary");
  t2vRefImagePanel.append(t2vRefImageNote, t2vRefImageDrop, t2vRefImageLoadButton);
  const ernieCreateT2IButton = makeButton("Gemma T2I", "primary");
  const ernieSendT2IPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const i2vPrompt = document.createElement("textarea");
  i2vPrompt.placeholder = "Image-to-video prompt...";
  i2vPrompt.style.cssText = notesInput.style.cssText;
  const videoModeChooser = document.createElement("div");
  videoModeChooser.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;";
  const imageToVideoCard = makeImageModelCard("Image to Video", "i2v");
  const textToVideoCard = makeImageModelCard("Text to Video", "t2v");
  videoModeChooser.append(imageToVideoCard, textToVideoCard);
  const i2vUnetPicker = makeSearchableLoraPicker("");
  const i2vVaePicker = makeSearchableLoraPicker("");
  const i2vClip1Picker = makeSearchableLoraPicker("");
  const i2vClip2Picker = makeSearchableLoraPicker("");
  const i2vUpscalePicker = makeSearchableLoraPicker("");
  const i2vAudioVaePicker = makeSearchableLoraPicker("");
  const i2vFpsInput = makeInput("24", "number");
  const i2vWidthInput = makeInput("1920", "number");
  const i2vHeightInput = makeInput("1080", "number");
  const i2vSeedInput = makeInput("69", "number");
  const i2vUseLora = makeCheckbox("Use video LoRAs?", false);
  const i2vLoraPanel = document.createElement("div");
  i2vLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const i2vLoraHintRow = document.createElement("div");
  i2vLoraHintRow.style.cssText = "display:flex;justify-content:flex-end;";
  const i2vLoraHintButton = makeButton("?", "neutral");
  i2vLoraHintButton.title = "Why use different strengths per pass?";
  i2vLoraHintButton.style.cssText += "width:34px;padding:7px 0;";
  i2vLoraHintRow.append(i2vLoraHintButton);
  const i2vLoraCount = makeInput("0", "number");
  i2vLoraCount.min = "0";
  i2vLoraCount.max = "4";
  const i2vLoraRows = document.createElement("div");
  i2vLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const i2vLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 84px 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const firstPassStrength = makeInput("1", "number");
    firstPassStrength.step = "0.01";
    const secondPassStrength = makeInput("1", "number");
    secondPassStrength.step = "0.01";
    row.append(
      makeField(`Video LoRA ${slot}`, picker.wrapper),
      makeField("Pass 1", firstPassStrength),
      makeField("Pass 2", secondPassStrength)
    );
    i2vLoraRows.append(row);
    i2vLoraSlots.push({ row, picker, firstPassStrength, secondPassStrength });
  }
  i2vLoraPanel.append(i2vLoraHintRow, makeField("Video LoRA count", i2vLoraCount), i2vLoraRows);
  const i2vSettingsGrid = document.createElement("div");
  i2vSettingsGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  i2vSettingsGrid.append(makeField("FPS", i2vFpsInput), makeField("Seed", i2vSeedInput), makeField("Width", i2vWidthInput), makeField("Height", i2vHeightInput));
  const createSceneVideoButton = makeButton("Create Scene Video", "primary");
  const createSceneVideoButtons = [createSceneVideoButton];
  function makeCreateSceneVideoButton() {
    const button = makeButton("Create Scene Video", "primary");
    createSceneVideoButtons.push(button);
    return button;
  }
  const previewButton = makeButton("Create Z-Image", "primary");
  const zCreateButtons = [previewButton];
  function makeZCreateButton() {
    const button = makeButton("Create Z-Image", "primary");
    zCreateButtons.push(button);
    return button;
  }
  const ernieCreateButton = makeButton("Create with Ernie", "primary");
  const ernieCreateButtons = [ernieCreateButton];
  function makeErnieCreateButton() {
    const button = makeButton("Create with Ernie", "primary");
    ernieCreateButtons.push(button);
    return button;
  }
  const fluxCreateButtons = [previewFluxButton];
  function makeFluxCreateButton() {
    const button = makeButton("Create with Flux/Klein", "primary");
    fluxCreateButtons.push(button);
    return button;
  }
  const nbCreateButtons = [previewNBButton];
  function makeNBCreateButton() {
    const button = makeButton("Create with NanoBanana", "primary");
    nbCreateButtons.push(button);
    return button;
  }
  function setButtonGroupState(buttons, { disabled = false, text = "" } = {}) {
    for (const button of buttons) {
      button.disabled = disabled;
      if (text) button.textContent = text;
    }
  }
  const inspectorActions = document.createElement("div");
  inspectorActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:6px;";
  for (const button of [previewButton]) {
    button.style.padding = "7px 8px";
    button.style.fontSize = "11px";
  }
  const timingGrid = document.createElement("div");
  timingGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  timingGrid.append(makeField("Start", startInput), makeField("End", endInput));
  const audioSummary = document.createElement("div");
  audioSummary.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#d4d4d8;padding:9px;font-size:12px;line-height:1.45;overflow-wrap:anywhere;";
  const globalAudioSummary = document.createElement("div");
  globalAudioSummary.style.cssText = audioSummary.style.cssText;
  const openSceneAudioOptionsButton = makeButton("Open Scene Audio Options", "primary");
  const inspectorTabs = document.createElement("div");
  inspectorTabs.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;position:sticky;top:0;z-index:3;background:#202024;padding-bottom:2px;";
  const sceneTabButton = makeButton("Scene");
  const imageTabButton = makeButton("Image");
  const videoTabButton = makeButton("Video");
  const audioTabButton = makeButton("Audio");
  inspectorTabs.append(sceneTabButton, imageTabButton, videoTabButton, audioTabButton);
  const scenePanel = document.createElement("div");
  const imagePanel = document.createElement("div");
  const videoPanel = document.createElement("div");
  const audioPanel = document.createElement("div");
  const noSceneNotice = document.createElement("div");
  noSceneNotice.style.cssText = "display:none;min-height:220px;align-items:center;justify-content:center;text-align:center;border:1px dashed #3f3f46;border-radius:8px;background:#18181b;color:#a1a1aa;padding:24px;font-size:13px;line-height:1.5;";
  noSceneNotice.innerHTML = `<div><div style="color:#e4e4e7;font-weight:900;font-size:15px;margin-bottom:6px;">Select a scene</div><div>Choose a scene from the list or timeline to edit its prompts, images, video, audio, and timing.</div></div>`;
  for (const panel of [scenePanel, imagePanel, videoPanel, audioPanel]) {
    panel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  }
  function syncInspectorPanels() {
    const hasScene = Boolean(activeSegment());
    const tabName = state.inspectorTab || "scene";
    noSceneNotice.style.display = hasScene ? "none" : "flex";
    inspectorTabs.style.opacity = hasScene ? "1" : ".45";
    inspectorTabs.style.pointerEvents = hasScene ? "auto" : "none";
    scenePanel.style.display = hasScene && tabName === "scene" ? "flex" : "none";
    imagePanel.style.display = hasScene && tabName === "image" ? "flex" : "none";
    videoPanel.style.display = hasScene && tabName === "video" ? "flex" : "none";
    audioPanel.style.display = hasScene && tabName === "audio" ? "flex" : "none";
  }
  function setInspectorTab(tabName) {
    const activeColor = "#06b6d4";
    const inactiveColor = "#27272a";
    state.inspectorTab = tabName;
    syncInspectorPanels();
    if (tabName === "image" || tabName === "video" || tabName === "audio") {
      state.rightPanelWidth = Math.max(state.rightPanelWidth || 360, 460);
    }
    applyLayoutSizes();
    for (const [button, name] of [[sceneTabButton, "scene"], [imageTabButton, "image"], [videoTabButton, "video"], [audioTabButton, "audio"]]) {
      const active = name === tabName;
      button.style.background = active ? activeColor : inactiveColor;
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
  }
  sceneTabButton.onclick = () => setInspectorTab("scene");
  imageTabButton.onclick = () => setInspectorTab("image");
  videoTabButton.onclick = () => setInspectorTab("video");
  audioTabButton.onclick = () => setInspectorTab("audio");
  const zImageModelsSection = makeSettingsPanel([
    makeSettingsSection("ZImage Models", [
      makeField("ZImage model", zUnetPicker.wrapper),
      makeField("CLIP", zClipPicker.wrapper),
      makeField("VAE", zVaePicker.wrapper),
    ]),
    makeSettingsSection("LLM Models", [
      makeField("Non-Vision text Gemma model", t2iTextGemmaModelSelect),
      makeField("Vision Gemma model", gemmaModelSelect),
      makeField("Vision mmproj", mmprojSelect),
    ]),
    zUseLora.wrapper,
    zLoraPanel,
    makeZCreateButton(),
  ]);
  const zImageSettingsSection = makeSettingsPanel([
    makeField("Image trigger phrase", imageTriggerInput),
    useSceneZImageSettings.wrapper,
    zFirstTitle,
    zFirstGrid,
    zSecondTitle,
    zSecondGrid,
    zSeedGrid,
    makeField("Batch size", zBatchSize),
    zUseImageToImage.wrapper,
    zI2IPanel,
    inspectorActions,
  ]);
  const zImagePromptSection = makeSettingsPanel([
    makeField("Notes", notesInput),
    useVisionReference.wrapper,
    refImagePanel,
    createT2IButton,
    makeField("T2I prompt", t2iPrompt),
    sendT2IPromptToEnhanceButton,
    makeZCreateButton(),
  ]);
  const zImageSubTabs = makeSubTabs([
    { label: "Models", value: "models", content: zImageModelsSection },
    { label: "Image Settings", value: "settings", content: zImageSettingsSection },
    { label: "LLM Prompting", value: "prompting", content: zImagePromptSection },
  ]);
  zimageSettingsPanel.append(zImageSubTabs.wrapper);
  zImageModePanel.append(zimageSettingsPanel);
  fluxKleinModePanel.append(fluxKleinPanel);
  ernieImageModePanel.append(ernieImagePanel);
  const ernieImageSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeSettingsSection("Ernie Models", [
          makeField("Ernie model", ernieUnetPicker.wrapper),
          makeField("CLIP", ernieClipPicker.wrapper),
          makeField("VAE", ernieVaePicker.wrapper),
        ]),
        makeSettingsSection("LLM Models", [
          makeField("Non-Vision text Gemma model", ernieTextGemmaModelSelect),
          makeField("Vision Gemma model", ernieGemmaModelSelect),
          makeField("Vision mmproj", ernieMmprojSelect),
        ]),
        ernieUseLora.wrapper,
        ernieLoraPanel,
        makeErnieCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        useSceneErnieImageSettings.wrapper,
        makeField("Image trigger phrase", ernieImageTriggerInput),
        ernieGrid,
        makeField("Batch size", ernieBatchSize),
        ernieUseImageToImage.wrapper,
        ernieI2IPanel,
        ernieCreateButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Notes", ernieNotesInput),
        ernieUseVisionReference.wrapper,
        ernieRefImagePanel,
        ernieCreateT2IButton,
        makeField("T2I prompt", ernieT2IPrompt),
        ernieSendT2IPromptToEnhanceButton,
        makeErnieCreateButton(),
      ]),
    },
  ]);
  ernieImagePanel.append(ernieImageSubTabs.wrapper);
  const fluxKleinSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeSettingsSection("Flux/Klein Models", [
          makeField("Flux model", fluxUnetPicker.wrapper),
          makeField("Flux CLIP", fluxClipPicker.wrapper),
          makeField("Flux VAE", fluxVaePicker.wrapper),
        ]),
        makeSettingsSection("Vision LLM Models", [
          makeField("Gemma vision model", fluxGemmaModelSelect),
          makeField("Vision mmproj", fluxMmprojSelect),
        ]),
        fluxUseLora.wrapper,
        fluxLoraPanel,
        makeFluxCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        useSceneFluxKleinSettings.wrapper,
        fluxUseTextOnlyGemmaPrompt.wrapper,
        makeField("Image trigger phrase", fluxImageTriggerInput),
        fluxImageRefsPanel,
        fluxGrid,
        previewFluxButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Flux/Klein notes", fluxNotes),
        createFluxPromptButton,
        makeField("Flux/Klein prompt", fluxPrompt),
        sendFluxPromptToEnhanceButton,
        makeFluxCreateButton(),
      ]),
    },
  ]);
  fluxKleinPanel.append(
    fluxKleinSubTabs.wrapper,
  );
  const nbImageSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        useSceneNBImageSettings.wrapper,
        makeSettingsSection("NanoBanana", [
          makeField("Google Cloud API key", nbApiKey),
          makeField("Model", nbModelSelect),
        ]),
        makeSettingsSection("Vision LLM Models", [
          makeField("Gemma vision model", nbGemmaModelSelect),
          makeField("Vision mmproj", nbMmprojSelect),
        ]),
        makeNBCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        nbUseGlobalIngredients.wrapper,
        nbUseTextOnlyGemmaPrompt.wrapper,
        nbGlobalIngredientPanel,
        nbIngredientDrop,
        nbIngredientActions,
        nbIngredientList,
        makeNBCreateButton(),
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("NanoBanana notes", nbNotes),
        createNBPromptButton,
        makeField("NanoBanana prompt", nbPrompt),
        sendNBPromptToEnhanceButton,
        previewNBButton,
      ]),
    },
  ]);
  nbImagePanel.append(nbImageSubTabs.wrapper);
  fluxKleinModePanel.append(nbImagePanel);
  const zEnhanceSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeField("ZImage model", zEnhanceUnetPicker.wrapper),
        makeField("CLIP", zEnhanceClipPicker.wrapper),
        makeField("VAE", zEnhanceVaePicker.wrapper),
        makeSettingsSection("Vision LLM Models", [
          makeField("Gemma vision model", zEnhanceGemmaModelSelect),
          makeField("Vision mmproj", zEnhanceMmprojSelect),
        ]),
        zEnhanceUseLora.wrapper,
        zEnhanceLoraPanel,
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        zEnhanceGrid,
        makeField("Enhance amount", zEnhanceAmount),
        zEnhanceAmountValue,
        zEnhanceHint,
        zEnhanceButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Gemma notes", zEnhanceGemmaNotes),
        zEnhanceGemmaButton,
        makeField("Enhance prompt", zEnhancePromptPreview),
      ]),
    },
  ]);
  zEnhancePanel.append(
    zEnhanceTitle,
    zEnhanceSubTabs.wrapper,
  );
  scenePanel.append(
    makeField("Scene label", labelInput),
    freezeTimingControl.wrapper,
    timingGrid,
    makeEditField("Prompt JSON path", promptJsonInput, editPromptJsonButton),
    importPromptJsonButton,
    makeEditField("I2V motion notes JSON path", i2vMotionJsonInput, editI2VMotionJsonButton),
    importI2VMotionJsonButton,
    useVrgdgTextContext.wrapper,
    loadVrgdgContextButton,
    makeEditField("Global theme/style text file", themeStyleInput, editThemeStyleButton),
    makeEditField("Global story idea text file", storyIdeaInput, editStoryIdeaButton),
    makeEditField("Global subject/scene text file", subjectSceneInput, editSubjectSceneButton),
  );
  imagePanel.append(
    imageModelChooserWrap,
    zImageModePanel,
    fluxKleinModePanel,
    ernieImageModePanel,
    zEnhancePanel,
    inspectorActions,
  );
  const videoSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        useSceneI2VVideoSettings.wrapper,
        useSceneI2VVideoSettingsNote,
        makeSettingsSection("Video Models", [
          makeField("Unet model", i2vUnetPicker.wrapper),
          makeField("Video VAE", i2vVaePicker.wrapper),
          makeField("Clip model 1", i2vClip1Picker.wrapper),
          makeField("Clip model 2", i2vClip2Picker.wrapper),
          makeField("Latent upscaler", i2vUpscalePicker.wrapper),
          makeField("Audio VAE", i2vAudioVaePicker.wrapper),
        ]),
        makeSettingsSection("Non-Vision LLM Models", [
          makeField("Non-Vision text Gemma model", i2vTextGemmaModelSelect),
        ]),
        makeSettingsSection("Vision LLM Models", [
          makeField("Vision Gemma model", i2vGemmaModelSelect),
          makeField("Vision mmproj", i2vMmprojSelect),
        ]),
        i2vUseLora.wrapper,
        i2vLoraPanel,
        createSceneVideoButton,
      ]),
    },
    {
      label: "Video Settings",
      value: "settings",
      content: makeSettingsPanel([
        videoSettingsScopeNote,
        makeField("Video trigger phrase", videoTriggerInput),
        i2vSettingsGrid,
        makeCreateSceneVideoButton(),
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Video motion notes", i2vNotesInput),
        useI2VVisionReference.wrapper,
        i2vReferenceNote,
        useT2VVisionReference.wrapper,
        t2vReferenceNote,
        t2vRefImagePanel,
        createI2VButton,
        makeField("Video prompt", i2vPrompt),
        makeCreateSceneVideoButton(),
      ]),
    },
  ]);
  videoPanel.append(videoModeChooser, videoSubTabs.wrapper);
  audioPanel.append(
    makeSettingsSection("Scene Audio", [
      audioSummary,
      openSceneAudioOptionsButton,
    ]),
    makeSettingsSection("Timeline Audio", [
      globalAudioSummary,
    ], false),
  );
  inspector.append(inspectorTabs, noSceneNotice, scenePanel, imagePanel, videoPanel, audioPanel);

  const timeline = document.createElement("div");
  timeline.style.cssText = "display:grid;grid-template-rows:7px auto 1fr;border-top:1px solid #27272a;background:#111113;min-height:0;";
  const timelineResizeHandle = document.createElement("div");
  timelineResizeHandle.title = "Drag to resize timeline";
  timelineResizeHandle.style.cssText = "cursor:row-resize;background:#18181b;border-bottom:1px solid #27272a;";
  const timelineHeader = document.createElement("div");
  timelineHeader.style.cssText = "display:grid;grid-template-columns:repeat(13,auto) minmax(260px,1fr) auto auto;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid #27272a;font-size:12px;";
  const bulkSegmentsButton = makeButton("Bulk Segments");
  const addSegmentButton = makeButton("Add Segment", "primary");
  const addOverlaySegmentButton = makeButton("Add Insert", "primary");
  const undoButton = makeButton("Undo");
  const redoButton = makeButton("Redo");
  const playButton = makeButton("Play");
  const stopButton = makeButton("Stop");
  const multiSelectButton = makeButton("Select Multi");
  const multiSelectHintButton = makeButton("?");
  const deleteSegmentButton = makeButton("Del");
  const zoomOutButton = makeButton("-");
  const zoomInButton = makeButton("+");
  bulkSegmentsButton.title = "Create many manual timeline scenes from pasted durations or start/end times.";
  addSegmentButton.title = "Add segment";
  addOverlaySegmentButton.title = "Add an insert/overlay segment at the playhead without changing the base timeline.";
  undoButton.title = "Undo";
  redoButton.title = "Redo";
  playButton.title = "Play / Pause";
  stopButton.title = "Stop";
  multiSelectButton.title = "Select multiple scenes, then batch-apply image/video settings or stitch a preview.";
  multiSelectHintButton.title = "What does Select Multi do?";
  deleteSegmentButton.title = "Delete selected segment";
  zoomOutButton.title = "Zoom out timeline";
  zoomInButton.title = "Zoom in timeline";
  bulkSegmentsButton.textContent = "Bulk Segments";
  addSegmentButton.textContent = "+ Segment";
  addOverlaySegmentButton.textContent = "+ Insert";
  undoButton.textContent = "↶";
  redoButton.textContent = "↷";
  playButton.textContent = "▶";
  stopButton.textContent = "■";
  multiSelectButton.textContent = "Select Multi";
  multiSelectHintButton.textContent = "?";
  deleteSegmentButton.textContent = "×";
  deleteSegmentButton.style.borderColor = "#7f1d1d";
  deleteSegmentButton.style.color = "#fecaca";
  for (const button of [bulkSegmentsButton, addSegmentButton, addOverlaySegmentButton, undoButton, redoButton, playButton, stopButton, multiSelectButton, multiSelectHintButton, deleteSegmentButton, zoomOutButton, zoomInButton]) {
    button.style.padding = "7px 10px";
    button.style.minWidth = "0";
  }
  bulkSegmentsButton.style.width = "max-content";
  addSegmentButton.style.width = "max-content";
  addOverlaySegmentButton.style.width = "max-content";
  for (const button of [undoButton, redoButton, playButton, stopButton, deleteSegmentButton, zoomOutButton, zoomInButton]) {
    button.style.width = "34px";
  }
  const waveformModeSelect = makeSelect(Object.keys(WAVEFORM_MODES), "medium");
  waveformModeSelect.style.minWidth = "126px";
  for (const option of waveformModeSelect.options) {
    option.textContent = WAVEFORM_MODES[option.value]?.label || option.value;
  }
  const snapToBeatsControl = makeCheckbox("Snap beats", true);
  snapToBeatsControl.wrapper.style.margin = "0";
  const beatMarkersButton = makeButton("^");
  beatMarkersButton.title = "Show or hide beat markers";
  beatMarkersButton.style.width = "34px";
  beatMarkersButton.style.padding = "7px 10px";
  const globalScrub = document.createElement("input");
  globalScrub.type = "range";
  globalScrub.min = "0";
  globalScrub.max = "0";
  globalScrub.step = "0.01";
  globalScrub.value = "0";
  globalScrub.style.cssText = "width:100%;accent-color:#22d3ee;";
  const globalScrubWrap = document.createElement("label");
  globalScrubWrap.style.cssText = "display:grid;grid-template-columns:auto minmax(160px,1fr) auto;align-items:center;gap:8px;color:#d4d4d8;font-size:12px;font-weight:800;background:#18181b;border-top:1px solid #27272a;padding:8px 10px;";
  const globalScrubLabel = document.createElement("span");
  globalScrubLabel.textContent = "Global video scrub";
  globalScrubLabel.style.cssText = "white-space:nowrap;";
  const globalScrubTime = document.createElement("span");
  globalScrubTime.textContent = "00:00.00";
  globalScrubTime.style.cssText = "color:#67e8f9;font-variant-numeric:tabular-nums;white-space:nowrap;";
  globalScrubWrap.append(globalScrubLabel, globalScrub, globalScrubTime);
  preview.append(previewStage, globalScrubWrap);
  const timelineInfo = document.createElement("div");
  timelineInfo.textContent = "No audio loaded";
  timelineInfo.style.cssText = "color:#67e8f9;font-variant-numeric:tabular-nums;";
  const selectedMediaTools = document.createElement("div");
  selectedMediaTools.style.cssText = "margin-left:auto;display:flex;gap:8px;align-items:center;border:1px solid #27272a;border-radius:6px;background:#111113;padding:6px 8px;";
  const selectedMediaLabel = document.createElement("span");
  selectedMediaLabel.textContent = "Selected media: none";
  selectedMediaLabel.style.cssText = "font-size:12px;color:#a1a1aa;white-space:nowrap;";
  const deleteSelectedMediaButton = makeButton("Delete Image/Video");
  deleteSelectedMediaButton.style.padding = "6px 10px";
  deleteSelectedMediaButton.style.borderColor = "#7f1d1d";
  deleteSelectedMediaButton.style.color = "#fecaca";
  selectedMediaTools.append(selectedMediaLabel, deleteSelectedMediaButton);
  const zoomWrap = document.createElement("div");
  zoomWrap.style.cssText = "display:flex;gap:4px;align-items:center;";
  zoomWrap.append(zoomOutButton, zoomInButton);
  timelineHeader.append(bulkSegmentsButton, addSegmentButton, addOverlaySegmentButton, undoButton, redoButton, playButton, stopButton, multiSelectButton, multiSelectHintButton, waveformModeSelect, snapToBeatsControl.wrapper, beatMarkersButton, zoomWrap, timelineInfo, deleteSegmentButton, selectedMediaTools);
  const timelineViewport = document.createElement("div");
  timelineViewport.style.cssText = "position:relative;overflow:auto;min-height:0;padding:12px;";
  const timelineCanvas = document.createElement("canvas");
  timelineCanvas.height = TIMELINE_HEIGHT;
  timelineCanvas.style.cssText = `display:block;height:${TIMELINE_HEIGHT}px;background:#09090b;border:1px solid #27272a;border-radius:6px;`;
  const segmentLayer = document.createElement("div");
  segmentLayer.style.cssText = `position:absolute;left:12px;top:12px;height:${TIMELINE_HEIGHT}px;pointer-events:none;`;
  const playhead = document.createElement("div");
  playhead.style.cssText = `position:absolute;left:12px;top:12px;height:${TIMELINE_HEIGHT}px;width:3px;background:#f4f4f5;box-shadow:0 0 10px rgba(103,232,249,.8);cursor:ew-resize;z-index:5;`;
  timelineViewport.append(timelineCanvas, segmentLayer, playhead);
  timeline.append(timelineResizeHandle, timelineHeader, timelineViewport);
  const audio = document.createElement("audio");
  audio.preload = "metadata";
  const sceneAudio = document.createElement("audio");
  sceneAudio.preload = "metadata";

  shell.append(topbar, main, timeline);
  overlay.append(shell);
  document.body.append(overlay);

  function applyBuilderFullscreen(enabled) {
    builderFullscreen = Boolean(enabled);
    shell.style.cssText = builderFullscreen ? fullscreenShellStyle : normalShellStyle;
    overlay.style.alignItems = builderFullscreen ? "stretch" : "center";
    overlay.style.justifyContent = builderFullscreen ? "stretch" : "center";
    overlay.style.background = builderFullscreen ? "#09090b" : "rgba(0,0,0,.72)";
    fullscreenButton.textContent = builderFullscreen ? "Exit Fullscreen" : "Fullscreen";
    fullscreenButton.title = builderFullscreen
      ? "Return the Video Creator to the normal floating panel size."
      : "Expand the Video Creator to fill the browser window without closing or resetting anything.";
    applyLayoutSizes();
    render();
  }

  setTimeout(() => {
    showStartupWelcome().catch((error) => {
      console.warn("[VRGDG Music Builder] Startup welcome failed:", error);
      toast(`Video Creator startup failed:\n${String(error?.message || error)}`, true);
    });
  }, 250);
  for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
    overlay.addEventListener(eventName, (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      if (event.target?.closest?.("[data-vrgdg-file-drop-zone='true']")) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
    }, true);
  }

  function defaultZImageSettings() {
    return {
      unet_name: "z_image_turbo_bf16.safetensors",
      clip_name: "qwen_3_4b.safetensors",
      vae_name: "ae.safetensors",
      first_pass_width: 1280,
      first_pass_height: 720,
      second_pass_width: 1920,
      second_pass_height: 1080,
      seed: 1,
      seed_mode: "fixed",
      batch_size: 1,
      use_loras: false,
      lora_count: 0,
      loras: [],
      use_image_to_image: false,
      image_to_image_start_at_step: 5,
      image_to_image_path: "",
      image_to_image_data: "",
      image_to_image_name: "",
      image_trigger_phrase: "",
    };
  }

  function defaultFluxKleinSettings() {
    return {
      enabled: false,
      image_model_mode: "",
      use_text_only_gemma_prompt: false,
      unet_name: "flux\\flux-2-klein-4b-fp8.safetensors",
      clip_name: "qwen_3_4b.safetensors",
      vae_name: "flux\\flux2-vae.safetensors",
      width: 1024,
      height: 576,
      seed: 100,
      use_loras: false,
      lora_count: 0,
      loras: [],
      image_trigger_phrase: "",
    };
  }

  function defaultErnieImageSettings() {
    return {
      unet_name: "ernie\\ernie-image-turbo.safetensors",
      clip_name: "ministral-3-3b.safetensors",
      vae_name: "flux\\flux2-vae.safetensors",
      width: 1280,
      height: 720,
      seed: 1,
      seed_mode: "fixed",
      batch_size: 1,
      use_loras: false,
      lora_count: 0,
      loras: [],
      use_image_to_image: false,
      image_to_image_start_at_step: 5,
      image_to_image_path: "",
      image_to_image_data: "",
      image_to_image_name: "",
      image_trigger_phrase: "",
    };
  }

  function defaultZEnhanceSettings() {
    return {
      unet_name: "z_image_turbo_bf16.safetensors",
      clip_name: "qwen_3_4b.safetensors",
      vae_name: "ae.safetensors",
      width: 1920,
      height: 1080,
      seed: 1,
      seed_mode: "randomize",
      enhance_amount: 8,
      use_loras: false,
      lora_count: 0,
      loras: [],
      video_trigger_phrase: "",
    };
  }

  function defaultI2VVideoSettings() {
    return {
      unet_name: DEFAULT_I2V_UNET,
      vae_name: "LTX23_video_vae_bf16.safetensors",
      clip_name1: "gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors",
      clip_name2: "ltx-2.3_text_projection_bf16.safetensors",
      upscale_model_name: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
      audio_vae_name: "LTX23_audio_vae_bf16.safetensors",
      fps: 24,
      width: 1920,
      height: 1080,
      seed: 69,
      use_loras: false,
      lora_count: 0,
      loras: [],
    };
  }

  const state = {
    duration: 0,
    peaks: [],
    beats: [],
    segments: [],
    overlaySegments: [],
    activeId: "",
    activeTrack: "base",
    multiSelectMode: false,
    selectedSegmentIds: [],
    inspectorTab: "scene",
    pxPerSecond: 45,
    timelineZoom: 45,
    waveformMode: "medium",
    snapToBeats: true,
    showBeatMarkers: false,
    leftPanelWidth: 260,
    rightPanelWidth: 360,
    timelinePanelHeight: 300,
    projectFolder: projectInput.value,
    sessionPath: "",
    srtPath: "",
    autoSaveEnabled: true,
    isScrubbing: false,
    isClipScrubbing: false,
    sceneAudioMode: false,
    sceneAudioSegmentId: "",
    sceneAudioGlobalTime: 0,
    timingFrozen: false,
    srtMode: false,
    promptJsonPath: "",
    i2vMotionJsonPath: "",
    lyricSegmentsPath: "",
    imageTriggerPhrase: "",
    videoTriggerPhrase: "",
    useVrgdgTextContext: true,
    themeStylePath: "",
    storyIdeaPath: "",
    subjectScenePath: "",
    textGemmaRunner: "builtin",
    lmStudioBaseUrl: "http://127.0.0.1:1234/v1",
    lmStudioModel: "",
    lmStudioApiKey: "",
    imageModelMode: "zimage",
    zimageSettings: defaultZImageSettings(),
    fluxKleinSettings: defaultFluxKleinSettings(),
    ernieImageSettings: defaultErnieImageSettings(),
    nbImageSettings: defaultNBImageSettings(),
    useFluxGlobalImageIngredients: false,
    fluxGlobalImageIngredients: [],
    fluxReferenceBuilder: defaultFluxReferenceBuilder(),
    zEnhanceSettings: defaultZEnhanceSettings(),
    videoModelMode: "i2v",
    i2vVideoSettings: defaultI2VVideoSettings(),
    promptToolsHintPrefs: {},
    undoStack: [],
    redoStack: [],
    isRestoringHistory: false,
    batchCancelled: false,
  };
  const LAST_PROJECT_KEY = "vrgdg_music_builder_last_project_folder";

  function textGemmaRunnerPayload() {
    return {
      text_runner: state.textGemmaRunner || "builtin",
      lmstudio_base_url: state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
      lmstudio_model: state.lmStudioModel || "",
      lmstudio_api_key: state.lmStudioApiKey || "",
    };
  }

  function defaultNBImageSettings() {
    return {
      api_key: "",
      model: DEFAULT_NB_IMAGE_MODEL,
      use_text_only_gemma_prompt: false,
    };
  }

  function gemmaRunnerLabel(options = {}) {
    if (options.forceBuiltin) return options.vision ? "Built-in GGUF vision" : "Built-in GGUF";
    if (options.vision) return state.textGemmaRunner === "lm_studio" ? "LM Studio vision" : "Built-in GGUF vision";
    return state.textGemmaRunner === "lm_studio" ? "LM Studio" : "Built-in GGUF";
  }

  function gemmaRunnerLine(options = {}) {
    return `Runner: ${gemmaRunnerLabel(options)}`;
  }

  function isLikelyEmbeddingModelId(modelId) {
    const text = String(modelId || "").toLowerCase();
    return text.includes("embedding") || text.includes("embed") || text.includes("nomic-embed") || text.includes("bge-") || text.includes("e5-");
  }

  function rememberLastProject(projectFolder = "") {
    const folder = String(projectFolder || "").trim();
    if (!folder) return;
    try {
      localStorage.setItem(LAST_PROJECT_KEY, folder);
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not save last project folder:", error);
    }
  }

  function getLastProject() {
    try {
      return localStorage.getItem(LAST_PROJECT_KEY) || "";
    } catch {
      return "";
    }
  }

  function allEditableSegments() {
    return [...state.segments, ...state.overlaySegments];
  }

  function segmentTrack(segment) {
    if (!segment) return "base";
    if (state.overlaySegments.some((item) => item.id === segment.id)) return "overlay";
    return "base";
  }

  function segmentIndexInfo(segment) {
    if (!segment) return { track: "base", index: -1 };
    const baseIndex = state.segments.findIndex((item) => item.id === segment.id);
    if (baseIndex >= 0) return { track: "base", index: baseIndex };
    const overlayIndex = state.overlaySegments.findIndex((item) => item.id === segment.id);
    if (overlayIndex >= 0) return { track: "overlay", index: overlayIndex };
    return { track: segment.track === "overlay" ? "overlay" : "base", index: -1 };
  }

  function sceneSlotNumber(segment) {
    const info = segmentIndexInfo(segment);
    if (info.track === "overlay") return 10000 + Math.max(0, info.index) + 1;
    return Math.max(0, info.index) + 1;
  }

  function activeSegment() {
    return allEditableSegments().find((segment) => segment.id === state.activeId) || null;
  }

  function selectedSegmentsForBatch({ baseOnly = false } = {}) {
    const ids = new Set(Array.isArray(state.selectedSegmentIds) ? state.selectedSegmentIds : []);
    const items = allEditableSegments().filter((segment) => ids.has(segment.id));
    return baseOnly ? items.filter((segment) => segmentTrack(segment) !== "overlay") : items;
  }

  function isSegmentMultiSelected(segment) {
    return Boolean(segment?.id && Array.isArray(state.selectedSegmentIds) && state.selectedSegmentIds.includes(segment.id));
  }

  function updateMultiSelectButton() {
    const count = selectedSegmentsForBatch().length;
    multiSelectButton.textContent = state.multiSelectMode ? `Multi ${count}` : "Select Multi";
    multiSelectButton.style.background = state.multiSelectMode ? "#0e7490" : "#27272a";
    multiSelectButton.style.borderColor = state.multiSelectMode ? "#22d3ee" : "#3f3f46";
    multiSelectButton.style.color = state.multiSelectMode ? "#ecfeff" : "#f4f4f5";
  }

  function toggleMultiSegmentSelection(segment) {
    if (!segment?.id) return;
    const ids = new Set(Array.isArray(state.selectedSegmentIds) ? state.selectedSegmentIds : []);
    if (ids.has(segment.id)) ids.delete(segment.id);
    else ids.add(segment.id);
    state.selectedSegmentIds = Array.from(ids);
    if (!state.activeId || !ids.has(state.activeId)) {
      state.activeId = segment.id;
      state.activeTrack = segmentTrack(segment);
      syncInspector();
    }
    render();
  }

  function handleSegmentPick(segment) {
    if (state.multiSelectMode) toggleMultiSegmentSelection(segment);
    else setActiveSegment(segment);
  }

  function applyImageSettingsToMultiSelection(kind, settings) {
    if (!state.multiSelectMode) return 0;
    const targets = selectedSegmentsForBatch();
    if (targets.length <= 1) return 0;
    for (const segment of targets) {
      if (kind === "zimage") {
        segment.use_scene_zimage_settings = true;
        segment.zimage_settings = cloneZImageSettings(settings);
      } else if (kind === "ernie_image") {
        segment.use_scene_ernie_image_settings = true;
        segment.ernie_image_settings = { ...settings, loras: Array.isArray(settings.loras) ? settings.loras.map((item) => ({ ...item })) : [] };
      } else if (kind === "flux_klein") {
        segment.use_scene_flux_klein_settings = true;
        segment.flux_klein_settings = { ...settings, loras: Array.isArray(settings.loras) ? settings.loras.map((item) => ({ ...item })) : [] };
      } else if (kind === "nano_banana") {
        segment.use_scene_nb_image_settings = true;
        segment.nb_image_settings = cloneNBImageSettings(settings);
      }
    }
    return targets.length;
  }

  function hasMultiSceneBatchSelection() {
    return state.multiSelectMode && selectedSegmentsForBatch().length > 1;
  }

  function applyVideoSettingsToMultiSelection(settings) {
    if (!state.multiSelectMode) return 0;
    const targets = selectedSegmentsForBatch();
    if (targets.length <= 1) return 0;
    for (const segment of targets) {
      segment.use_scene_i2v_video_settings = true;
      segment.i2v_video_settings = cloneI2VVideoSettings(settings);
    }
    return targets.length;
  }

  function usingSceneAudioMode() {
    return state.segments.some((segment) => String(segment.custom_audio_path || "").trim());
  }

  function audioTimelineStart(segment) {
    if (!segment) return 0;
    const value = Number(segment.custom_audio_timeline_start);
    return Number.isFinite(value) ? value : Number(segment.start || 0);
  }

  function audioSourceStart(segment) {
    if (!segment) return 0;
    const value = Number(segment.custom_audio_source_start);
    return Number.isFinite(value) ? Math.max(0, value) : 0;
  }

  function audioChunkDuration(segment) {
    if (!segment) return 0;
    const value = Number(segment.custom_audio_duration);
    if (Number.isFinite(value) && value > 0) return value;
    return Math.max(0, Number(segment.end || 0) - Number(segment.start || 0));
  }

  function timelineSegmentDuration(segment) {
    if (!segment) return 0;
    return Math.max(0, Number(segment.end || 0) - Number(segment.start || 0));
  }

  function audioTimelineEnd(segment) {
    return audioTimelineStart(segment) + audioChunkDuration(segment);
  }

  function audioSegmentAtTime(time) {
    const current = Number(time || 0);
    return state.segments.find((segment, index) => {
      if (!String(segment.custom_audio_path || "").trim()) return false;
      const start = audioTimelineStart(segment);
      const end = audioTimelineEnd(segment);
      const isLast = index === state.segments.length - 1;
      return current >= start && (current < end || (isLast && current <= end));
    }) || null;
  }

  function mediaPathKey(path) {
    return String(path || "").trim().replace(/\\/g, "/").replace(/\/+/g, "/").toLowerCase();
  }

  function isBackupSceneVideoPath(path) {
    return mediaPathKey(path).includes("/rendered_scene_videos_backup/");
  }

  function normalizeSegmentVideoHistory(segment) {
    if (!segment) return [];
    const currentPath = String(segment.video_path || "").trim();
    const currentKey = mediaPathKey(currentPath);
    const previousHistory = Array.isArray(segment.video_history) ? segment.video_history : [];
    const previousIndex = Number(segment.video_history_index);
    const previousSelectedPath = Number.isFinite(previousIndex) && previousIndex >= 0 ? String(previousHistory[previousIndex] || "").trim() : "";
    const previousSelectedKey = mediaPathKey(previousSelectedPath);
    const seen = new Set();
    const cleaned = [];
    const candidates = [
      ...(Array.isArray(segment.video_backup_paths) ? segment.video_backup_paths : []),
      ...(Array.isArray(segment.video_history) ? segment.video_history : []),
      currentPath,
    ];
    for (const item of candidates) {
      const path = String(item || "").trim();
      const key = mediaPathKey(path);
      if (!path || !key || seen.has(key)) continue;
      seen.add(key);
      cleaned.push(path);
    }
    segment.video_backup_paths = cleaned.filter(isBackupSceneVideoPath);
    segment.video_history = cleaned;
    if (!cleaned.length) {
      segment.video_history_index = -1;
    } else if (previousSelectedKey) {
      const selectedIndex = cleaned.findIndex((item) => mediaPathKey(item) === previousSelectedKey);
      segment.video_history_index = selectedIndex >= 0 ? selectedIndex : Math.max(0, Math.min(cleaned.length - 1, Number(segment.video_history_index || 0)));
    } else {
      const currentIndex = currentKey ? cleaned.findIndex((item) => mediaPathKey(item) === currentKey) : -1;
      segment.video_history_index = currentIndex >= 0 ? currentIndex : Math.max(0, Math.min(cleaned.length - 1, Number(segment.video_history_index || 0)));
    }
    return cleaned;
  }

  function timelineDuration() {
    const segmentEnd = state.segments.reduce((max, segment) => Math.max(max, Number(segment.end || 0)), 0);
    const overlayEnd = state.overlaySegments.reduce((max, segment) => Math.max(max, Number(segment.end || 0)), 0);
    const sceneAudioEnd = state.segments.reduce((max, segment) => Math.max(max, segment.custom_audio_path ? audioTimelineEnd(segment) : 0), 0);
    return Math.max(segmentEnd, overlayEnd, sceneAudioEnd, Number(state.duration || 0), Number(audio.duration || 0));
  }

  function currentGlobalTime() {
    if (usingSceneAudioMode()) return Number(state.sceneAudioGlobalTime || 0);
    return Number(audio.currentTime || 0);
  }

  function isTimelinePlaying() {
    return (audio.src && !audio.paused) || (sceneAudio.src && !sceneAudio.paused);
  }

  function updatePlayPauseButton() {
    const playing = isTimelinePlaying();
    playButton.textContent = playing ? "Ⅱ" : "▶";
    playButton.title = playing ? "Pause" : "Play";
  }

  function pauseAllAudio() {
    audio.pause();
    sceneAudio.pause();
    updatePlayPauseButton();
  }

  function applyLayoutSizes() {
    const left = Math.max(180, Math.min(520, Number(state.leftPanelWidth || 260)));
    const right = Math.max(280, Math.min(720, Number(state.rightPanelWidth || 360)));
    const timelineHeight = Math.max(190, Math.min(520, Number(state.timelinePanelHeight || 300)));
    state.leftPanelWidth = left;
    state.rightPanelWidth = right;
    state.timelinePanelHeight = timelineHeight;
    main.style.gridTemplateColumns = `${left}px 7px minmax(0,1fr) 7px ${right}px`;
    shell.style.gridTemplateRows = `auto minmax(0,1fr) ${timelineHeight}px`;
    drawWaveform();
  }

  function setTimelineZoom(value, anchorTime = currentGlobalTime()) {
    const oldZoom = Math.max(1, Number(state.pxPerSecond || 45));
    const zoom = Math.max(8, Math.min(260, Number(value || 45)));
    state.timelineZoom = zoom;
    state.pxPerSecond = zoom;
    drawWaveform();
    renderSegments();
    const previousScroll = timelineViewport.scrollLeft;
    const anchorX = anchorTime * oldZoom;
    const viewportAnchor = anchorX - previousScroll;
    timelineViewport.scrollLeft = Math.max(0, anchorTime * zoom - viewportAnchor);
    autoSaveSessionQuiet("timeline zoom changed");
  }

  function makePanelResize(handle, mode) {
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture?.(event.pointerId);
      const startX = event.clientX;
      const startY = event.clientY;
      const startLeft = state.leftPanelWidth;
      const startRight = state.rightPanelWidth;
      const startTimeline = state.timelinePanelHeight;
      const move = (moveEvent) => {
        if (mode === "left") {
          state.leftPanelWidth = startLeft + (moveEvent.clientX - startX);
        } else if (mode === "right") {
          state.rightPanelWidth = startRight - (moveEvent.clientX - startX);
        } else if (mode === "timeline") {
          state.timelinePanelHeight = startTimeline - (moveEvent.clientY - startY);
        }
        applyLayoutSizes();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        autoSaveSessionQuiet("layout resized");
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  }

  function isInternalApprovedImagePath(path) {
    return /(^|[\\/])zimage_approved([\\/]|$)/i.test(String(path || ""));
  }

  function ensureSegmentRuntimeFields(segment) {
    if (!segment) return segment;
    if (!Number.isFinite(Number(segment.start))) segment.start = 0;
    if (!Number.isFinite(Number(segment.end))) segment.end = Number(segment.start || 0) + 4;
    if (Number(segment.end) <= Number(segment.start)) segment.end = Number(segment.start || 0) + 0.1;
    if (segment.label == null) segment.label = "New scene";
    if (segment.lyric_text == null) segment.lyric_text = "";
    if (segment.id == null || !String(segment.id).trim()) segment.id = `seg_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    if (!Array.isArray(segment.image_history)) segment.image_history = [];
    const approvedImagePath = String(segment.approved_image_path || "");
    segment.image_history = segment.image_history.filter((item, index, list) => {
      const path = String(item || "");
      return path && path !== approvedImagePath && !isInternalApprovedImagePath(path) && list.indexOf(item) === index;
    });
    if (!Number.isFinite(Number(segment.image_history_index))) segment.image_history_index = segment.image_history.length ? segment.image_history.length - 1 : -1;
    segment.image_history_index = segment.image_history.length
      ? Math.max(0, Math.min(segment.image_history.length - 1, Number(segment.image_history_index || 0)))
      : -1;
    if (segment.enhance_notes == null) segment.enhance_notes = "";
    if (segment.enhance_prompt == null) segment.enhance_prompt = "";
    if (segment.custom_audio_path == null) segment.custom_audio_path = "";
    if (segment.custom_audio_name == null) segment.custom_audio_name = "";
    if (!Number.isFinite(Number(segment.custom_audio_duration))) segment.custom_audio_duration = 0;
    if (!Number.isFinite(Number(segment.custom_audio_full_duration))) segment.custom_audio_full_duration = Number(segment.custom_audio_duration || 0);
    if (!Number.isFinite(Number(segment.custom_audio_timeline_start))) segment.custom_audio_timeline_start = Number(segment.start || 0);
    if (!Number.isFinite(Number(segment.custom_audio_source_start))) segment.custom_audio_source_start = 0;
    if (!Array.isArray(segment.custom_audio_peaks)) segment.custom_audio_peaks = [];
    if (!Array.isArray(segment.custom_audio_beats)) segment.custom_audio_beats = [];
    if (!Array.isArray(segment.flux_image_ingredients)) {
      segment.flux_image_ingredients = [];
      if (segment.flux_subject_image_path || segment.flux_subject_image_data || segment.flux_subject_image_name) {
        segment.flux_image_ingredients.push({
          path: segment.flux_subject_image_path || "",
          data: segment.flux_subject_image_data || "",
          name: segment.flux_subject_image_name || "subject.png",
        });
      }
      if (segment.flux_location_image_path || segment.flux_location_image_data || segment.flux_location_image_name) {
        segment.flux_image_ingredients.push({
          path: segment.flux_location_image_path || "",
          data: segment.flux_location_image_data || "",
          name: segment.flux_location_image_name || "location.png",
        });
      }
    }
    if (segment.flux_notes == null) segment.flux_notes = "";
    if (segment.flux_prompt == null) segment.flux_prompt = "";
    if (segment.use_scene_zimage_settings == null) segment.use_scene_zimage_settings = false;
    if (segment.zimage_settings && typeof segment.zimage_settings !== "object") segment.zimage_settings = null;
    if (segment.use_scene_ernie_image_settings == null) segment.use_scene_ernie_image_settings = false;
    if (segment.ernie_image_settings && typeof segment.ernie_image_settings !== "object") segment.ernie_image_settings = null;
    if (segment.use_scene_flux_klein_settings == null) segment.use_scene_flux_klein_settings = false;
    if (segment.flux_klein_settings && typeof segment.flux_klein_settings !== "object") segment.flux_klein_settings = null;
    if (segment.nb_notes == null) segment.nb_notes = "";
    if (segment.nb_prompt == null) segment.nb_prompt = "";
    if (segment.use_scene_nb_image_settings == null) segment.use_scene_nb_image_settings = false;
    if (segment.nb_image_settings && typeof segment.nb_image_settings !== "object") segment.nb_image_settings = null;
    if (segment.use_scene_i2v_video_settings == null) segment.use_scene_i2v_video_settings = false;
    if (segment.i2v_video_settings && typeof segment.i2v_video_settings !== "object") segment.i2v_video_settings = null;
    if (!["image", "video"].includes(segment.preview_mode)) segment.preview_mode = segment.video_path ? "video" : "image";
    if (segment.video_path == null) segment.video_path = "";
    if (segment.video_folder == null) segment.video_folder = "";
    if (!Array.isArray(segment.video_history)) segment.video_history = [];
    if (segment.video_output == null) segment.video_output = null;
    if (segment.video_status == null) segment.video_status = segment.video_path ? "done" : "none";
    if (!segment.video_path && segment.video_output && typeof segment.video_output === "object") {
      segment.video_path = segment.video_output.path || segment.video_output.filename || "";
    }
    if (segment.video_path && !segment.video_history.includes(segment.video_path)) {
      segment.video_history.push(segment.video_path);
    }
    normalizeSegmentVideoHistory(segment);
    return segment;
  }

  function ensureAllSegmentRuntimeFields() {
    state.segments = (Array.isArray(state.segments) ? state.segments : [])
      .filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment))
      .filter((segment, index) => {
        const recoveredId = String(segment.id || "").match(/^recovered_scene_(\d+)$/i);
        if (recoveredId && Number(recoveredId[1]) >= 10000) return false;
        if (segment.track === "overlay") return false;
        return index < 10000;
      })
      .map((segment) => {
        segment.track = "base";
        return ensureSegmentRuntimeFields(segment);
      });
    state.overlaySegments = (Array.isArray(state.overlaySegments) ? state.overlaySegments : [])
      .filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment))
      .map((segment) => {
        segment.track = "overlay";
        return ensureSegmentRuntimeFields(segment);
      });
    sortSegments(state.overlaySegments);
  }

  function cloneZImageSettings(settings) {
    const source = settings || {};
    return {
      unet_name: source.unet_name || "z_image_turbo_bf16.safetensors",
      clip_name: source.clip_name || "qwen_3_4b.safetensors",
      vae_name: source.vae_name || "ae.safetensors",
      first_pass_width: Number(source.first_pass_width || 1280),
      first_pass_height: Number(source.first_pass_height || 720),
      second_pass_width: Number(source.second_pass_width || 1920),
      second_pass_height: Number(source.second_pass_height || 1080),
      seed: Number(source.seed || 1),
      seed_mode: source.seed_mode || "fixed",
      batch_size: Math.max(1, Math.min(16, Number(source.batch_size || 1))),
      use_loras: Boolean(source.use_loras),
      lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
      loras: Array.isArray(source.loras) ? source.loras.map((item) => ({
        name: item?.name || "[none]",
        first_pass_strength: Number(item?.first_pass_strength ?? item?.strength ?? 0.5),
        second_pass_strength: Number(item?.second_pass_strength ?? item?.strength ?? 1),
        strength: Number(item?.second_pass_strength ?? item?.strength ?? 1),
      })) : [],
      use_image_to_image: Boolean(source.use_image_to_image),
      image_to_image_start_at_step: Math.max(1, Math.min(8, Number(source.image_to_image_start_at_step || 5))),
      image_to_image_path: source.image_to_image_path || "",
      image_to_image_data: source.image_to_image_data || "",
      image_to_image_name: source.image_to_image_name || "",
      image_trigger_phrase: source.image_trigger_phrase || state.imageTriggerPhrase || "",
    };
  }

  function cloneErnieImageSettings(settings) {
    const source = settings || {};
    return {
      ...defaultErnieImageSettings(),
      ...source,
      width: Number(source.width || 1280),
      height: Number(source.height || 720),
      seed: Number(source.seed || 1),
      batch_size: Math.max(1, Math.min(16, Number(source.batch_size || 1))),
      lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
      loras: Array.isArray(source.loras) ? source.loras.map((item) => ({ name: item?.name || "[none]", strength: Number(item?.strength ?? 1) })) : [],
      image_trigger_phrase: source.image_trigger_phrase || state.imageTriggerPhrase || "",
    };
  }

  function cloneNBImageSettings(settings) {
    const source = settings || {};
    return {
      ...defaultNBImageSettings(),
      ...source,
      api_key: source.api_key || "",
      model: source.model || DEFAULT_NB_IMAGE_MODEL,
      use_text_only_gemma_prompt: Boolean(source.use_text_only_gemma_prompt),
    };
  }

  function cloneFluxKleinSettings(settings) {
    const source = settings || {};
    return {
      ...defaultFluxKleinSettings(),
      ...source,
      width: Number(source.width || 1024),
      height: Number(source.height || 576),
      seed: Number(source.seed || 100),
      use_text_only_gemma_prompt: Boolean(source.use_text_only_gemma_prompt),
      use_loras: Boolean(source.use_loras),
      lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
      loras: Array.isArray(source.loras) ? source.loras.map((item) => ({
        name: item?.name || "[none]",
        strength: Number(item?.strength ?? 1),
      })) : [],
      image_trigger_phrase: source.image_trigger_phrase || state.imageTriggerPhrase || "",
    };
  }

  function cloneI2VVideoSettings(settings) {
    const source = settings || {};
    return {
      ...defaultI2VVideoSettings(),
      ...source,
      fps: Number(source.fps || 24),
      width: Number(source.width || 1920),
      height: Number(source.height || 1080),
      seed: Number(source.seed || 69),
      lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
      loras: Array.isArray(source.loras) ? source.loras.map((item) => ({
        name: item?.name || "[none]",
        first_pass_strength: Number(item?.first_pass_strength ?? item?.strength ?? 1),
        second_pass_strength: Number(item?.second_pass_strength ?? item?.strength ?? 1),
        strength: Number(item?.second_pass_strength ?? item?.strength ?? 1),
      })) : [],
      video_trigger_phrase: source.video_trigger_phrase || state.videoTriggerPhrase || "",
    };
  }

  function applyModelDefaults(defaults) {
    if (!defaults || typeof defaults !== "object" || Array.isArray(defaults)) return false;
    if (defaults.text_gemma_runner || defaults.textGemmaRunner) {
      state.textGemmaRunner = defaults.text_gemma_runner || defaults.textGemmaRunner || state.textGemmaRunner || "builtin";
    }
    if (defaults.lm_studio_base_url || defaults.lmStudioBaseUrl) {
      state.lmStudioBaseUrl = defaults.lm_studio_base_url || defaults.lmStudioBaseUrl || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    }
    if (defaults.lm_studio_model || defaults.lmStudioModel) {
      state.lmStudioModel = defaults.lm_studio_model || defaults.lmStudioModel || state.lmStudioModel || "";
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "lm_studio_api_key") || Object.prototype.hasOwnProperty.call(defaults, "lmStudioApiKey")) {
      state.lmStudioApiKey = defaults.lm_studio_api_key ?? defaults.lmStudioApiKey ?? state.lmStudioApiKey ?? "";
    }
    state.imageModelMode = defaults.image_model_mode || defaults.imageModelMode || defaults.flux_klein_settings?.image_model_mode || defaults.fluxKleinSettings?.image_model_mode || state.imageModelMode || "zimage";
    if (defaults.zimage_settings || defaults.zimageSettings) {
      state.zimageSettings = cloneZImageSettings(defaults.zimage_settings || defaults.zimageSettings);
    }
    if (defaults.flux_klein_settings || defaults.fluxKleinSettings) {
      state.fluxKleinSettings = cloneFluxKleinSettings(defaults.flux_klein_settings || defaults.fluxKleinSettings);
    }
    if (defaults.ernie_image_settings || defaults.ernieImageSettings) {
      state.ernieImageSettings = cloneErnieImageSettings(defaults.ernie_image_settings || defaults.ernieImageSettings);
    }
    if (defaults.nb_image_settings || defaults.nbImageSettings) {
      state.nbImageSettings = cloneNBImageSettings(defaults.nb_image_settings || defaults.nbImageSettings);
    }
    if (defaults.z_enhance_settings || defaults.zEnhanceSettings) {
      state.zEnhanceSettings = {
        ...defaultZEnhanceSettings(),
        ...(defaults.z_enhance_settings || defaults.zEnhanceSettings || {}),
      };
    }
    state.videoModelMode = defaults.video_model_mode || defaults.videoModelMode || state.videoModelMode || "i2v";
    if (defaults.i2v_video_settings || defaults.i2vVideoSettings) {
      state.i2vVideoSettings = cloneI2VVideoSettings(defaults.i2v_video_settings || defaults.i2vVideoSettings);
    }
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncNBImagePanel();
    syncZEnhanceSettingsPanel();
    syncI2VVideoSettingsPanel();
    syncVideoModePanel();
    return true;
  }

  async function loadGlobalModelDefaultsQuiet() {
    try {
      const data = await getJson("/vrgdg/music_builder/model_defaults");
      const applied = applyModelDefaults(data.defaults || {});
      if (applied) {
        console.log("[VRGDG Music Builder] Loaded global model defaults:", data.path || "");
      }
      return applied;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not load global model defaults:", error);
      return false;
    }
  }

  function activeZImageSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_zimage_settings) {
      if (!segment.zimage_settings) segment.zimage_settings = cloneZImageSettings(state.zimageSettings);
      return segment.zimage_settings;
    }
    return state.zimageSettings;
  }

  function activeErnieImageSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_ernie_image_settings) {
      if (!segment.ernie_image_settings) segment.ernie_image_settings = cloneErnieImageSettings(state.ernieImageSettings);
      return segment.ernie_image_settings;
    }
    return state.ernieImageSettings;
  }

  function activeNBImageSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_nb_image_settings) {
      if (!segment.nb_image_settings) segment.nb_image_settings = cloneNBImageSettings(state.nbImageSettings);
      return segment.nb_image_settings;
    }
    return state.nbImageSettings;
  }

  function activeFluxKleinSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_flux_klein_settings) {
      if (!segment.flux_klein_settings) segment.flux_klein_settings = cloneFluxKleinSettings(state.fluxKleinSettings);
      return segment.flux_klein_settings;
    }
    return state.fluxKleinSettings;
  }

  function activeI2VVideoSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_i2v_video_settings) {
      if (!segment.i2v_video_settings) segment.i2v_video_settings = cloneI2VVideoSettings(state.i2vVideoSettings);
      return segment.i2v_video_settings;
    }
    return state.i2vVideoSettings;
  }

  function videoVisionReferenceEnabled(segment) {
    return currentVideoMode() === "t2v"
      ? Boolean(segment?.use_t2v_vision_reference)
      : segment?.use_i2v_vision_reference !== false;
  }

  function setVideoVisionReferenceEnabled(segment, enabled) {
    if (!segment) return;
    if (currentVideoMode() === "t2v") segment.use_t2v_vision_reference = Boolean(enabled);
    else segment.use_i2v_vision_reference = Boolean(enabled);
  }

  function historySnapshot() {
    return JSON.stringify({
      segments: state.segments,
      overlaySegments: state.overlaySegments,
      activeId: state.activeId,
      activeTrack: state.activeTrack,
      timingFrozen: state.timingFrozen,
      srtMode: state.srtMode,
      promptJsonPath: state.promptJsonPath,
      i2vMotionJsonPath: state.i2vMotionJsonPath,
      imageTriggerPhrase: state.imageTriggerPhrase,
      videoTriggerPhrase: state.videoTriggerPhrase,
      useVrgdgTextContext: state.useVrgdgTextContext,
      themeStylePath: state.themeStylePath,
      storyIdeaPath: state.storyIdeaPath,
      subjectScenePath: state.subjectScenePath,
      textGemmaRunner: state.textGemmaRunner,
      lmStudioBaseUrl: state.lmStudioBaseUrl,
      lmStudioModel: state.lmStudioModel,
      lmStudioApiKey: state.lmStudioApiKey,
      waveformMode: state.waveformMode,
      snapToBeats: state.snapToBeats,
      showBeatMarkers: state.showBeatMarkers,
      leftPanelWidth: state.leftPanelWidth,
      rightPanelWidth: state.rightPanelWidth,
      timelinePanelHeight: state.timelinePanelHeight,
      timelineZoom: state.timelineZoom,
      autoSaveEnabled: state.autoSaveEnabled,
      imageModelMode: state.imageModelMode,
      zimageSettings: state.zimageSettings,
      fluxKleinSettings: state.fluxKleinSettings,
      nbImageSettings: state.nbImageSettings,
      ernieImageSettings: state.ernieImageSettings,
      useFluxGlobalImageIngredients: state.useFluxGlobalImageIngredients,
      fluxGlobalImageIngredients: state.fluxGlobalImageIngredients,
      zEnhanceSettings: state.zEnhanceSettings,
      videoModelMode: state.videoModelMode,
      i2vVideoSettings: state.i2vVideoSettings,
      promptToolsHintPrefs: state.promptToolsHintPrefs,
    });
  }

  function restoreHistorySnapshot(snapshot) {
    const data = JSON.parse(snapshot);
    state.isRestoringHistory = true;
    state.segments = data.segments || [];
    state.overlaySegments = data.overlaySegments || data.overlay_segments || [];
    ensureAllSegmentRuntimeFields();
    state.activeId = data.activeId || state.segments[0]?.id || "";
    state.activeTrack = data.activeTrack || data.active_track || segmentTrack(activeSegment()) || "base";
    state.timingFrozen = Boolean(data.timingFrozen);
    state.srtMode = Boolean(data.srtMode);
    state.promptJsonPath = data.promptJsonPath || "";
    state.i2vMotionJsonPath = data.i2vMotionJsonPath || "";
    state.imageTriggerPhrase = data.imageTriggerPhrase || "";
    state.videoTriggerPhrase = data.videoTriggerPhrase || "";
    state.useVrgdgTextContext = data.useVrgdgTextContext ?? true;
    state.themeStylePath = data.themeStylePath || "";
    state.storyIdeaPath = data.storyIdeaPath || "";
    state.subjectScenePath = data.subjectScenePath || "";
    state.textGemmaRunner = data.textGemmaRunner || data.text_gemma_runner || state.textGemmaRunner || "builtin";
    state.lmStudioBaseUrl = data.lmStudioBaseUrl || data.lm_studio_base_url || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    state.lmStudioModel = data.lmStudioModel || data.lm_studio_model || state.lmStudioModel || "";
    state.lmStudioApiKey = data.lmStudioApiKey || data.lm_studio_api_key || state.lmStudioApiKey || "";
    state.waveformMode = data.waveformMode || state.waveformMode || "medium";
    state.snapToBeats = data.snapToBeats ?? state.snapToBeats ?? true;
    state.peaks = Array.isArray(data.peaks) ? data.peaks : state.peaks;
    state.beats = Array.isArray(data.beats) ? data.beats : state.beats;
    setBeatMarkersVisible(data.showBeatMarkers ?? state.showBeatMarkers ?? false);
    state.leftPanelWidth = data.leftPanelWidth || state.leftPanelWidth || 260;
    state.rightPanelWidth = data.rightPanelWidth || state.rightPanelWidth || 360;
    state.timelinePanelHeight = data.timelinePanelHeight || state.timelinePanelHeight || 300;
    state.timelineZoom = data.timelineZoom || state.timelineZoom || 45;
    state.autoSaveEnabled = data.autoSaveEnabled ?? state.autoSaveEnabled ?? true;
    state.imageModelMode = data.imageModelMode || data.fluxKleinSettings?.image_model_mode || state.imageModelMode || "zimage";
    state.pxPerSecond = state.timelineZoom;
    waveformModeSelect.value = state.waveformMode;
    snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
    autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
    applyLayoutSizes();
    state.zimageSettings = data.zimageSettings || state.zimageSettings;
    state.fluxKleinSettings = data.fluxKleinSettings || state.fluxKleinSettings;
    state.nbImageSettings = data.nbImageSettings || data.nb_image_settings || state.nbImageSettings;
    state.ernieImageSettings = data.ernieImageSettings || state.ernieImageSettings;
    state.useFluxGlobalImageIngredients = Boolean(data.useFluxGlobalImageIngredients);
    state.fluxGlobalImageIngredients = Array.isArray(data.fluxGlobalImageIngredients) ? data.fluxGlobalImageIngredients : [];
    state.zEnhanceSettings = data.zEnhanceSettings || state.zEnhanceSettings;
    state.videoModelMode = data.videoModelMode || data.video_model_mode || state.videoModelMode || "i2v";
    state.i2vVideoSettings = data.i2vVideoSettings || state.i2vVideoSettings;
    state.promptToolsHintPrefs = data.promptToolsHintPrefs || data.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncZEnhanceSettingsPanel();
    syncI2VVideoSettingsPanel();
    syncVideoModePanel();
    syncInspector();
    render();
    updateHistoryButtons();
    state.isRestoringHistory = false;
  }

  function pushHistory() {
    if (state.isRestoringHistory) return;
    const snapshot = historySnapshot();
    if (state.undoStack[state.undoStack.length - 1] === snapshot) return;
    state.undoStack.push(snapshot);
    if (state.undoStack.length > 50) state.undoStack.shift();
    state.redoStack = [];
    updateHistoryButtons();
  }

  function undo() {
    if (!state.undoStack.length) return;
    const current = historySnapshot();
    const previous = state.undoStack.pop();
    state.redoStack.push(current);
    if (state.redoStack.length > 50) state.redoStack.shift();
    restoreHistorySnapshot(previous);
    syncPromptJsonFromSegments("undo");
    syncI2VMotionJsonFromSegments("undo");
  }

  function redo() {
    if (!state.redoStack.length) return;
    const current = historySnapshot();
    const next = state.redoStack.pop();
    state.undoStack.push(current);
    if (state.undoStack.length > 50) state.undoStack.shift();
    restoreHistorySnapshot(next);
    syncPromptJsonFromSegments("redo");
    syncI2VMotionJsonFromSegments("redo");
  }

  function updateHistoryButtons() {
    undoButton.disabled = !state.undoStack.length;
    redoButton.disabled = !state.redoStack.length;
    undoButton.style.opacity = undoButton.disabled ? ".55" : "1";
    redoButton.style.opacity = redoButton.disabled ? ".55" : "1";
  }

  function setBeatMarkersVisible(visible) {
    state.showBeatMarkers = Boolean(visible);
    beatMarkersButton.style.background = state.showBeatMarkers ? "#164e63" : "#27272a";
    beatMarkersButton.style.borderColor = state.showBeatMarkers ? "#0891b2" : "#3f3f46";
    beatMarkersButton.style.color = state.showBeatMarkers ? "#cffafe" : "#fafafa";
  }

  function showBeatMarkersIfAvailable() {
    if (Array.isArray(state.beats) && state.beats.length) {
      setBeatMarkersVisible(true);
    }
  }

  async function reloadBeatMarkersFromAudio() {
    const audioPath = String(audioInput.value || getWidget(node, "audio_path")?.value || "").trim();
    if (!audioPath) {
      toast("No audio path is loaded, so beat markers cannot be analyzed.", true);
      return false;
    }
    try {
      const data = await postJson("/vrgdg/music_builder/analyze_audio", {
        audio_path: audioPath,
        project_folder: projectInput.value || state.projectFolder || "",
        target_peaks: 1800,
      }, 90000);
      audioInput.value = data.audio_path || audioInput.value;
      setWidgetValue(node, "audio_path", audioInput.value);
      state.duration = Math.max(Number(state.duration || 0), Number(data.duration || 0));
      state.peaks = Array.isArray(data.peaks) ? data.peaks : [];
      state.beats = Array.isArray(data.beats) ? data.beats : [];
      setBeatMarkersVisible(Boolean(state.beats.length));
      if (!state.beats.length) {
        toast("Audio analysis finished, but no beat markers were detected.", true);
        return false;
      }
      render();
      toast(`Loaded ${state.beats.length} beat marker${state.beats.length === 1 ? "" : "s"}.`);
      await autoSaveSessionQuiet("beat markers refreshed");
      return true;
    } catch (error) {
      toast(`Could not reload beat markers:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  function cleanGeneratedPromptText(prompt) {
    let text = String(prompt || "").trim();
    text = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
    text = text.replace(/<\/?thought>/gi, "").trim();
    text = text.replace(/^[\s\-–—_=~.*#|:;,+/\\]{16,}(?=[\p{L}\p{N}])/u, "").trim();
    const controlPatterns = [
      /^\s*<?\/?end[_\-][a-z0-9_\-]*>?\s*/i,
      /^\s*_?name\s*[:=]\s*/i,
      /^\s*\d+\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*/i,
      /^\s*(?:0f|of)[_\-\s]*(?:thought|analysis|reasoning)\s*[:\-]?\s*/i,
      /^\s*_?\s*(?:thought|analysis|reasoning)\s*<channel\|>\s*/i,
      /^\s*_?\s*<\|?channel\|?>\s*(?:thought|analysis|reasoning)?\s*/i,
      /^\s*_?\s*<channel\|>\s*(?:thought|analysis|reasoning)?\s*/i,
      /^\s*_?\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*/i,
      /^(?:Assistant|Answer|Final prompt)\s*:\s*/i,
    ];
    let previous = "";
    while (text && previous !== text) {
      previous = text;
      for (const pattern of controlPatterns) text = text.replace(pattern, "").trim();
    }
    return text;
  }

  function looksLikeGeneratedPromptJunk(prompt) {
    const text = String(prompt || "").toLowerCase().replace(/\s+/g, " ").trim();
    if (!text) return false;
    const compact = text.replace(/[^a-z0-9_<>\-|]+/g, "");
    const markers = [
      "completion-completion-completion",
      "thought-thought-thought",
      "de-facto-de-facto-de-facto",
      "de-fleshed",
      "thoughtthoughtthought",
      "ownnessownnessownness",
      "nessnessnessness",
      "end_anow",
      "<|channel>",
      "<channel|>",
    ];
    if (markers.some((marker) => compact.includes(marker) || text.includes(marker))) return true;
    if (/([a-z]{2,16})\1{5,}/i.test(compact)) return true;
    const tokens = text.match(/[\p{L}\p{N}_']+/gu) || [];
    if (tokens.length >= 16) {
      const counts = new Map();
      for (const token of tokens) counts.set(token, (counts.get(token) || 0) + 1);
      const maxCount = Math.max(...counts.values());
      if (maxCount >= 10 && maxCount / tokens.length >= 0.20) return true;
      for (const size of [2, 3, 4]) {
        if (tokens.length < size * 4) continue;
        const phraseCounts = new Map();
        for (let index = 0; index <= tokens.length - size; index += 1) {
          const phrase = tokens.slice(index, index + size).join(" ");
          phraseCounts.set(phrase, (phraseCounts.get(phrase) || 0) + 1);
        }
        if (Math.max(...phraseCounts.values()) >= 8) return true;
      }
    }
    return false;
  }

  function isRecoverableBuildGemmaError(error) {
    const message = String(error?.message || error || "").toLowerCase();
    if (!message) return false;
    const recoverable = [
      "gemma returned repeated/thought junk",
      "gemma returned repeated/thought text",
      "repeated/thought junk",
      "repeated/thought text",
      "thought junk",
      "request timed out",
      "backend may still be processing",
      "failed to create a usable prompt",
      "returned an empty i2v prompt",
      "returned an empty t2v prompt",
      "returned an empty flux/klein prompt",
    ];
    if (recoverable.some((item) => message.includes(item))) return true;
    if (/gemma[\s\S]{0,80}(thought|junk|empty|timed out|timeout|repeated)/i.test(message)) return true;
    return false;
  }

  async function recoverFromBuildGemmaError(error, attempt, maxRetries, progress) {
    const message = String(error?.message || error);
    const windowTitle = `Build Full Video recovery ${attempt}/${maxRetries}`;
    const recoveryProgress = progress || createProgressWindow(windowTitle);
    recoveryProgress.set(`Recoverable Gemma error detected:\n${message}\n\nInterrupting any stuck backend job and clearing pending queue...`, 100);
    await cancelComfyExecutionAndWaitIdle((status) => {
      recoveryProgress.set(`Recoverable Gemma error detected:\n${message}\n\n${status}`, 100);
    }, { shouldCancel: () => state.batchCancelled });
    recoveryProgress.set(`Recoverable Gemma error detected:\n${message}\n\nClearing memory before retry ${attempt + 1}/${maxRetries + 1}...`, 100);
    try {
      const cleanupOutput = await runClearMemoryWorkflowQuiet(recoveryProgress, `Build Full Video retry ${attempt}/${maxRetries}`, 100);
      recoveryProgress.set(`${cleanupOutput}\n\nRetrying Build Full Video in resume mode so finished work is kept...`, 100);
    } catch (cleanupError) {
      recoveryProgress.set(`Cleanup failed after recoverable Gemma error:\n${String(cleanupError?.message || cleanupError)}\n\nRetrying anyway in resume mode...`, 100);
    }
    await autoSaveSessionQuiet("Build Full Video auto retry recovery").catch(() => null);
    recoveryProgress.close(1600);
  }

  async function runGemmaImagePromptPassWithRetry(segment, progress, basePercent, label, generator, options = {}) {
    const maxRetries = Math.max(1, Number(options.maxRetries || 3));
    let lastError = null;
    let lastErrorWasRecoverable = false;
    for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
      assertBatchNotStopped();
      try {
        const retryLabel = attempt === 1 ? label : `${label} retry ${attempt}/${maxRetries}`;
        const retryOptions = {
          ...(options.generatorOptions || {}),
          clearBeforeLoad: attempt === 1 ? options.clearBeforeLoad !== false : true,
          unloadAfter: attempt === maxRetries ? options.unloadAfter !== false : false,
          seed: Math.floor((Date.now() + Math.random() * 1000000 + attempt * 9973) % 2147483647),
          temperature: Math.min(0.95, Number(options.temperature ?? 0.25) + (attempt - 1) * 0.18),
          topP: Math.max(0.72, Math.min(0.98, Number(options.topP ?? 0.95) - (attempt - 1) * 0.04)),
        };
        progress?.set(`${retryLabel}\n${attempt === 1 ? "Keeping Gemma loaded until this prompt pass finishes..." : "Fresh retry after cleanup..."}`, basePercent);
        return await generator(segment, progress, Math.min(96, basePercent + 4), retryLabel, retryOptions);
      } catch (error) {
        lastError = error;
        lastErrorWasRecoverable = isRecoverableBuildGemmaError(error);
        if (!lastErrorWasRecoverable || attempt >= maxRetries) break;
        progress?.set(`${label}: Gemma returned a recoverable bad response on attempt ${attempt}/${maxRetries}.\nInterrupting and clearing memory before retry...`, Math.min(96, basePercent + 2));
        await cancelComfyExecutionAndWaitIdle((status) => {
          progress?.set(`${label}: cancelling failed Gemma job before retry...\n${status}`, Math.min(96, basePercent + 2));
        }, { shouldCancel: () => state.batchCancelled });
        await runClearMemoryWorkflowQuiet(progress, `${label} retry ${attempt}/${maxRetries}`, Math.min(96, basePercent + 3));
      }
    }
    if (lastErrorWasRecoverable && options.textFallback !== false) {
      progress?.set(`${label}: vision Gemma kept returning junk.\nSwitching to text-only Gemma for this scene only...`, Math.min(96, basePercent + 5));
      await cancelComfyExecutionAndWaitIdle((status) => {
        progress?.set(`${label}: cancelling vision Gemma before text-only fallback...\n${status}`, Math.min(96, basePercent + 5));
      }, { shouldCancel: () => state.batchCancelled });
      await runImageMemoryCleanupQuiet(progress, `${label} text-only fallback`, Math.min(96, basePercent + 6));
      try {
        return await generateTextOnlyImagePromptFallbackForSegment(
          segment,
          progress,
          Math.min(96, basePercent + 8),
          `${label}: text-only fallback`,
          { imageMode: options.imageMode || state.imageModelMode || "zimage" },
        );
      } catch (fallbackError) {
        progress?.set(`${label}: text-only Gemma fallback also failed.\nSaving a local notes-based prompt so the batch can continue...`, Math.min(96, basePercent + 9));
        return buildEmergencyImagePromptForSegment(segment, options.imageMode || state.imageModelMode || "zimage");
      }
    }
    throw lastError || new Error(`${label}: failed to create a usable prompt.`);
  }

  function applyTriggerPhrase(prompt, trigger, options = {}) {
    const promptText = cleanGeneratedPromptText(prompt);
    if (options.validateJunk !== false && looksLikeGeneratedPromptJunk(promptText)) {
      throw new Error("Gemma returned repeated/thought junk instead of a usable prompt. Try again or shorten the notes.");
    }
    const triggerText = String(trigger || "").trim().replace(/\s+/g, " ");
    if (!triggerText) return promptText;
    if (!promptText) return triggerText;
    if (promptText.toLowerCase().startsWith(triggerText.toLowerCase())) return promptText;
    return `${triggerText}, ${promptText}`;
  }

  function imageTriggerPhraseForSegment(segment = activeSegment(), imageMode = state.imageModelMode) {
    if (!segment) return state.imageTriggerPhrase || "";
    if (imageMode === "flux_klein") return fluxKleinSettingsForSegment(segment).image_trigger_phrase || state.imageTriggerPhrase || "";
    if (imageMode === "nano_banana") return "";
    if (imageMode === "ernie_image") return (segment.use_scene_ernie_image_settings ? segment.ernie_image_settings?.image_trigger_phrase : state.ernieImageSettings?.image_trigger_phrase) || state.imageTriggerPhrase || "";
    return (segment.use_scene_zimage_settings ? segment.zimage_settings?.image_trigger_phrase : state.zimageSettings?.image_trigger_phrase) || state.imageTriggerPhrase || "";
  }

  function applyImageTriggerToPrompt(prompt, segment = activeSegment(), imageMode = state.imageModelMode, options = {}) {
    return applyTriggerPhrase(prompt, imageTriggerPhraseForSegment(segment, imageMode), options);
  }

  function syncSegmentT2IPrompt(segment, prompt) {
    if (!segment) return "";
    const cleanPrompt = String(prompt || "").trim();
    segment.t2i_prompt = cleanPrompt;
    segment.flux_prompt = cleanPrompt;
    segment.nb_prompt = cleanPrompt;
    segment.enhance_prompt = cleanPrompt;
    if (segment.id === activeSegment()?.id) {
      t2iPrompt.value = cleanPrompt;
      ernieT2IPrompt.value = cleanPrompt;
      fluxPrompt.value = cleanPrompt;
      nbPrompt.value = cleanPrompt;
      zEnhancePromptPreview.value = cleanPrompt;
    }
    return cleanPrompt;
  }

  function ensureSegmentT2IPromptHasTrigger(segment, imageMode = state.imageModelMode, fallback = "") {
    const rawPrompt = String(
      imageMode === "flux_klein"
        ? (segment?.flux_prompt || segment?.t2i_prompt || fallback)
        : imageMode === "nano_banana"
          ? (segment?.nb_prompt || segment?.t2i_prompt || fallback)
        : (segment?.t2i_prompt || segment?.flux_prompt || fallback)
    ).trim();
    const prompt = applyImageTriggerToPrompt(rawPrompt, segment, imageMode, { validateJunk: false });
    return syncSegmentT2IPrompt(segment, prompt);
  }

  function videoTriggerPhraseForSegment(segment = activeSegment()) {
    if (!segment) return state.videoTriggerPhrase || "";
    return (segment.use_scene_i2v_video_settings ? segment.i2v_video_settings?.video_trigger_phrase : state.i2vVideoSettings?.video_trigger_phrase) || state.videoTriggerPhrase || "";
  }

  function conceptPromptsTextFromSegments() {
    const prompts = {};
    state.segments.forEach((segment, index) => {
      prompts[`Prompt${index + 1}`] = String(segment?.notes || "");
    });
    return JSON.stringify(prompts, null, 2);
  }

  function i2vMotionNotesTextFromSegments() {
    const notes = {};
    state.segments.forEach((segment, index) => {
      notes[`Motion${index + 1}`] = String(segment?.i2v_notes || "");
    });
    return JSON.stringify(notes, null, 2);
  }

  function hasAnyI2VMotionNotes(segments = state.segments) {
    return Array.isArray(segments) && segments.some((segment) => String(segment?.i2v_notes || "").trim());
  }

  async function loadI2VMotionNotesFromPath(path) {
    const notePath = String(path || "").trim();
    if (!notePath) return [];
    const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
      prompt_json_path: notePath,
    });
    return Array.isArray(data.prompts) ? data.prompts : [];
  }

  async function loadPromptJsonFromPath(path) {
    const promptPath = String(path || "").trim();
    if (!promptPath) return [];
    const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
      prompt_json_path: promptPath,
    });
    return Array.isArray(data.prompts) ? data.prompts : [];
  }

  async function loadLyricSegmentsFromPath(path) {
    const lyricPath = String(path || "").trim();
    if (!lyricPath) return [];
    const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
      prompt_json_path: lyricPath,
    });
    return Array.isArray(data.prompts) ? data.prompts : [];
  }

  function isInstrumentalLyricText(text) {
    const value = String(text || "").trim().toLowerCase().replace(/\s+/g, " ");
    return value === "instrumental" || value === "[instrumental]" || value === "instrumental section" || value === "instrumental section.";
  }

  function videoGemmaNotesForSegment(segment) {
    const notes = String(segment?.i2v_notes || "").trim();
    if (!isInstrumentalLyricText(segment?.lyric_text)) return notes;
    const instrumentalNote = "Lyric/performance status: instrumental / no sung lyrics. Do not make the subject sing or lip-sync unless the user notes explicitly ask for singing; use visual acting, camera motion, environmental motion, dancing, posing, walking, or atmosphere instead.";
    return notes ? `${instrumentalNote}\n\n${notes}` : instrumentalNote;
  }

  function defaultFluxReferenceBuilder() {
    return {
      use_subject_reference: false,
      use_location_references: false,
      include_manual_ingredients: true,
      subject: { description: "", image: { path: "", data: "", name: "" } },
      locations: [],
      scene_map: {},
    };
  }

  function normalizeFluxReferenceBuilder(value = {}) {
    const source = value && typeof value === "object" ? value : {};
    const normalized = defaultFluxReferenceBuilder();
    normalized.use_subject_reference = Boolean(source.use_subject_reference);
    normalized.use_location_references = Boolean(source.use_location_references);
    normalized.include_manual_ingredients = source.include_manual_ingredients !== false;
    const subject = source.subject && typeof source.subject === "object" ? source.subject : {};
    const subjectImage = subject.image && typeof subject.image === "object" ? subject.image : {};
    normalized.subject = {
      description: String(subject.description || ""),
      image: {
        path: String(subjectImage.path || ""),
        data: String(subjectImage.data || ""),
        name: String(subjectImage.name || ""),
      },
    };
    normalized.locations = Array.isArray(source.locations) ? source.locations
      .filter((item) => item && typeof item === "object")
      .map((item, index) => {
        const image = item.image && typeof item.image === "object" ? item.image : {};
        return {
          id: String(item.id || `loc_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
          name: String(item.name || `Location ${index + 1}`),
          description: String(item.description || ""),
          image: {
            path: String(image.path || ""),
            data: String(image.data || ""),
            name: String(image.name || ""),
          },
        };
      }) : [];
    normalized.scene_map = source.scene_map && typeof source.scene_map === "object" ? { ...source.scene_map } : {};
    return normalized;
  }

  async function syncPromptJsonFromSegments(reason = "") {
    const path = String(promptJsonInput.value || state.promptJsonPath || "").trim();
    if (!path) return false;
    try {
      const result = await postJson("/vrgdg/music_builder/save_text_file", {
        path,
        content: conceptPromptsTextFromSegments(),
      });
      promptJsonInput.value = result.path || path;
      state.promptJsonPath = promptJsonInput.value;
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync ConceptPrompts after ${reason || "segment change"}:`, error);
      toast(`Could not update ConceptPrompts.txt:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncI2VMotionJsonFromSegments(reason = "") {
    const path = String(i2vMotionJsonInput.value || state.i2vMotionJsonPath || "").trim();
    if (!path) return false;
    try {
      if (!hasAnyI2VMotionNotes()) {
        try {
          const existingNotes = await loadI2VMotionNotesFromPath(path);
          if (existingNotes.some((note) => String(note || "").trim())) {
            console.warn(`[VRGDG Music Builder] Skipped blank I2VMotionNotes sync after ${reason || "segment change"} because the existing file has motion notes.`);
            return false;
          }
        } catch (_error) {
          // If the file does not exist yet, allow the normal save path below.
        }
      }
      const result = await postJson("/vrgdg/music_builder/save_text_file", {
        path,
        content: i2vMotionNotesTextFromSegments(),
      });
      i2vMotionJsonInput.value = result.path || path;
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync I2VMotionNotes after ${reason || "segment change"}:`, error);
      toast(`Could not update I2VMotionNotes.txt:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  function setActiveSegment(segment) {
    if (state.activeId && state.activeId !== segment?.id) {
      saveI2VVideoSettingsFromPanel();
    }
    state.activeId = segment?.id || "";
    state.activeTrack = segment ? segmentTrack(segment) : state.activeTrack || "base";
    syncInspector();
    render();
  }

  function moveActiveSceneSelection(direction) {
    const track = state.activeTrack === "overlay" ? "overlay" : "base";
    const list = track === "overlay" ? state.overlaySegments : state.segments;
    if (!list.length) return false;
    const currentIndex = list.findIndex((segment) => segment.id === state.activeId);
    const fallbackIndex = direction > 0 ? -1 : list.length;
    const nextIndex = Math.max(0, Math.min(list.length - 1, (currentIndex >= 0 ? currentIndex : fallbackIndex) + direction));
    const next = list[nextIndex];
    if (!next || next.id === state.activeId) return false;
    setActiveSegment(next);
    return true;
  }

  function clearActiveSegment() {
    if (!state.activeId) return;
    state.activeId = "";
    syncInspector();
    render();
  }

  function setMultiSelectMode(enabled) {
    state.multiSelectMode = Boolean(enabled);
    if (!state.multiSelectMode) {
      state.selectedSegmentIds = [];
    } else if (state.activeId && !state.selectedSegmentIds.includes(state.activeId)) {
      state.selectedSegmentIds = [state.activeId];
    }
    render();
  }

  function segmentAtTime(time) {
    const current = Number(time || 0);
    return state.segments.find((segment, index) => {
      const start = Number(segment.start || 0);
      const end = Number(segment.end || 0);
      const isLast = index === state.segments.length - 1;
      return current >= start && (current < end || (isLast && current <= end));
    }) || null;
  }

  function playbackSegmentAtTime(time) {
    const current = Number(time || 0);
    const overlay = state.overlaySegments
      .slice()
      .sort((a, b) => Number(b.start || 0) - Number(a.start || 0))
      .find((segment) => {
        const start = Number(segment.start || 0);
        const end = Number(segment.end || 0);
        return current >= start && current < end;
      });
    return overlay || segmentAtTime(current);
  }

  function localPlaybackTime(segment, globalTime) {
    if (!segment) return 0;
    const duration = Math.max(0.1, Number(segment.end || 0) - Number(segment.start || 0));
    return Math.max(0, Math.min(duration, Number(globalTime || 0) - Number(segment.start || 0)));
  }

  function hasLockedVideo(segment) {
    return Boolean(segment?.video_path);
  }

  function selectedSegmentVideoPath(segment) {
    if (!segment) return "";
    if (!Array.isArray(segment.video_history)) normalizeSegmentVideoHistory(segment);
    const history = Array.isArray(segment?.video_history) ? segment.video_history : [];
    const index = Math.max(0, Math.min(history.length - 1, Number(segment?.video_history_index || 0)));
    return history[index] || segment?.video_path || "";
  }
  function selectedSegmentVideoCacheKey(segment, videoPath = "") {
    const path = String(videoPath || selectedSegmentVideoPath(segment) || "").trim();
    if (!path) return "";
    return `${path}|${String(segment?.video_cache_bust || "")}`;
  }

  function setPreviewVideoSource(segment, videoPath) {
    const cacheKey = selectedSegmentVideoCacheKey(segment, videoPath);
    if (!videoPath || !cacheKey) return;
    if (previewVideo.dataset.cacheKey !== cacheKey) {
      previewVideo.pause();
      previewVideo.src = makeEditorVideoUrl(videoPath);
      previewVideo.dataset.path = videoPath;
      previewVideo.dataset.cacheKey = cacheKey;
      previewVideo.load();
    }
  }

  function selectedSegmentImagePath(segment) {
    ensureSegmentRuntimeFields(segment);
    const history = Array.isArray(segment?.image_history) ? segment.image_history : [];
    const index = Math.max(0, Math.min(history.length - 1, Number(segment?.image_history_index || 0)));
    return history[index] || segment?.approved_image_path || segment?.custom_image_path || "";
  }

  function selectedSegmentImageThumbnailPath(segment) {
    ensureSegmentRuntimeFields(segment);
    return segment?.image_history?.[segment.image_history_index]
      || segment?.image_history?.[segment.image_history.length - 1]
      || segment?.custom_image_path
      || segment?.approved_image_path
      || "";
  }

  function selectedSegmentVideoThumbnailPath(segment) {
    return selectedSegmentVideoPath(segment);
  }

  function mediaThumbnailHtml(segment, height = 56) {
    const imagePath = selectedSegmentImageThumbnailPath(segment);
    if (imagePath) {
      return `<img src="${escapeHtml(makeEditorImageUrl(imagePath))}" style="width:100%;height:${height}px;object-fit:cover;border-radius:4px;margin-top:6px;background:#050505;">`;
    }
    const videoPath = selectedSegmentVideoThumbnailPath(segment);
    if (!videoPath) return "";
    return `<video src="${escapeHtml(makeEditorVideoUrl(videoPath))}" preload="metadata" muted playsinline style="width:100%;height:${height}px;object-fit:cover;border-radius:4px;margin-top:6px;background:#050505;display:block;"></video>`;
  }

  function appendTimelineVideoThumbnail(block, segment) {
    const videoPath = selectedSegmentVideoThumbnailPath(segment);
    if (!videoPath) return;
    const video = document.createElement("video");
    video.src = makeEditorVideoUrl(videoPath);
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.72;pointer-events:none;z-index:0;";
    const shade = document.createElement("span");
    shade.style.cssText = "position:absolute;inset:0;background:rgba(0,0,0,.24);pointer-events:none;z-index:1;";
    block.append(video, shade);
  }

  function selectedMediaForDelete() {
    const segment = activeSegment();
    if (!segment) return { segment: null, type: "", path: "" };
    if (segment.preview_mode === "video") {
      return { segment, type: "video", path: selectedSegmentVideoPath(segment) };
    }
    return { segment, type: "image", path: selectedSegmentImagePath(segment) };
  }

  function updateSelectedMediaTools() {
    const media = selectedMediaForDelete();
    const label = media.type ? `Selected media: ${media.type}` : "Selected media: none";
    selectedMediaLabel.textContent = media.path ? label : `${label} missing`;
    deleteSelectedMediaButton.textContent = media.type === "video" ? "Delete Video" : "Delete Image";
    deleteSelectedMediaButton.disabled = !media.path;
    deleteSelectedMediaButton.style.opacity = media.path ? "1" : ".55";
  }

  function syncPreview(segment) {
    ensureSegmentRuntimeFields(segment);
    const videoPath = selectedSegmentVideoPath(segment);
    if (segment?.preview_mode !== "image" && videoPath) {
      setPreviewVideoSource(segment, videoPath);
      previewVideo.muted = false;
      previewVideo.style.display = "block";
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
      return;
    }
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
    if (segment?.custom_image_data) {
      previewImage.src = segment.custom_image_data;
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    if (segment?.custom_image_path && !segment?.image?.filename) {
      previewImage.src = makeEditorImageUrl(segment.custom_image_path);
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    const selectedSource = segmentImageSource(segment);
    if (selectedSource?.path) {
      previewImage.src = makeEditorImageUrl(selectedSource.path);
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    if (selectedSource?.data) {
      previewImage.src = selectedSource.data;
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    const image = segment?.image || null;
    if (image?.filename) {
      previewImage.src = makeImageViewUrl(image);
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    previewEmpty.style.display = "block";
  }

  function syncInspector() {
    const segment = activeSegment();
    const disabled = !segment;
    for (const control of [labelInput, startInput, endInput, notesInput, ernieNotesInput, nbNotes, i2vNotesInput, t2iPrompt, ernieT2IPrompt, nbPrompt, i2vPrompt, zEnhanceGemmaNotes, zEnhancePromptPreview, previewButton, ernieCreateButton, previewNBButton, deleteSegmentButton, createSceneVideoButton]) {
      control.disabled = disabled;
    }
    loadCustomImageButton.disabled = disabled;
    openSceneAudioOptionsButton.disabled = disabled;
    for (const control of [t2iTextGemmaModelSelect, gemmaModelSelect, mmprojSelect, ernieTextGemmaModelSelect, ernieGemmaModelSelect, ernieMmprojSelect, zEnhanceGemmaModelSelect, zEnhanceMmprojSelect, i2vTextGemmaModelSelect, i2vGemmaModelSelect, i2vMmprojSelect, nbApiKey, nbModelSelect, nbGemmaModelSelect, nbMmprojSelect, fluxUseTextOnlyGemmaPrompt.input, nbUseTextOnlyGemmaPrompt.input, useVisionReference.input, ernieUseVisionReference.input, useI2VVisionReference.input, useT2VVisionReference.input, useSceneZImageSettings.input, useSceneErnieImageSettings.input, useSceneFluxKleinSettings.input, useSceneNBImageSettings.input, useSceneI2VVideoSettings.input, refImageInput, createT2IButton, ernieCreateT2IButton, createNBPromptButton, createI2VButton, zEnhanceGemmaButton]) {
      control.disabled = disabled;
    }
    const lockedByVideo = hasLockedVideo(segment);
    const isOverlay = segmentTrack(segment) === "overlay";
    startInput.disabled = disabled || (!isOverlay && state.timingFrozen) || lockedByVideo;
    endInput.disabled = disabled || (!isOverlay && state.timingFrozen) || lockedByVideo;
    freezeTimingControl.input.checked = Boolean(state.timingFrozen);
    promptJsonInput.value = state.promptJsonPath || "";
    i2vMotionJsonInput.value = state.i2vMotionJsonPath || "";
    useSceneErnieImageSettings.input.checked = Boolean(segment?.use_scene_ernie_image_settings);
    useSceneFluxKleinSettings.input.checked = Boolean(segment?.use_scene_flux_klein_settings);
    useSceneNBImageSettings.input.checked = Boolean(segment?.use_scene_nb_image_settings);
    useSceneI2VVideoSettings.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
    useVrgdgTextContext.input.checked = Boolean(state.useVrgdgTextContext);
    themeStyleInput.value = state.themeStylePath || "";
    storyIdeaInput.value = state.storyIdeaPath || "";
    subjectSceneInput.value = state.subjectScenePath || "";
    globalAudioSummary.innerHTML = `
      <div><strong>Global audio:</strong> ${escapeHtml(audioInput.value || "Not loaded")}</div>
      <div style="margin-top:6px;"><strong>SRT:</strong> ${escapeHtml(srtInput.value || "Not loaded")}</div>
      <div style="margin-top:6px;color:#a1a1aa;">Use the Settings button to load or change these files.</div>
    `;
    syncInspectorPanels();
    if (!segment) {
      labelInput.value = "";
      startInput.value = "0";
      endInput.value = "4";
      notesInput.value = "";
      ernieNotesInput.value = "";
      nbNotes.value = "";
      i2vNotesInput.value = "";
      t2iPrompt.value = "";
      ernieT2IPrompt.value = "";
      nbPrompt.value = "";
      i2vPrompt.value = "";
      useVisionReference.input.checked = false;
      ernieUseVisionReference.input.checked = false;
      useI2VVisionReference.input.checked = true;
      useT2VVisionReference.input.checked = false;
      useSceneZImageSettings.input.checked = false;
      useSceneNBImageSettings.input.checked = false;
      refImageInput.value = "";
      refImagePanel.style.display = "none";
      ernieRefImagePanel.style.display = "none";
      t2vRefImagePanel.style.display = "none";
      audioSummary.textContent = "Select a scene to view or edit scene audio.";
      syncZImageSettingsPanel();
      syncFluxKleinPanel();
      syncErnieImagePanel();
      syncVideoModePanel();
      syncPreview(null);
      return;
    }
    labelInput.value = segment.label || "";
    startInput.value = segment.start;
    endInput.value = segment.end;
    notesInput.value = segment.notes || "";
    ernieNotesInput.value = segment.notes || "";
    nbNotes.value = segment.nb_notes || segment.flux_notes || segment.notes || "";
    i2vNotesInput.value = segment.i2v_notes || "";
    t2iPrompt.value = segment.t2i_prompt || "";
    ernieT2IPrompt.value = segment.t2i_prompt || "";
    fluxPrompt.value = segment.t2i_prompt || segment.flux_prompt || "";
    nbPrompt.value = segment.t2i_prompt || segment.nb_prompt || "";
    i2vPrompt.value = segment.i2v_prompt || "";
    useVisionReference.input.checked = Boolean(segment.use_vision_reference);
    ernieUseVisionReference.input.checked = Boolean(segment.use_vision_reference);
    useI2VVisionReference.input.checked = segment.use_i2v_vision_reference !== false;
    useT2VVisionReference.input.checked = Boolean(segment.use_t2v_vision_reference);
    useSceneZImageSettings.input.checked = Boolean(segment.use_scene_zimage_settings);
    refImageInput.value = segment.ref_image_path || "";
    refImagePanel.style.display = useVisionReference.input.checked ? "flex" : "none";
    ernieRefImagePanel.style.display = ernieUseVisionReference.input.checked ? "flex" : "none";
    audioSummary.innerHTML = segment.custom_audio_path
      ? `
        <div><strong>Scene audio:</strong> ${escapeHtml(segment.custom_audio_name || segment.custom_audio_path)}</div>
        <div style="margin-top:6px;"><strong>Timeline:</strong> ${formatTime(audioTimelineStart(segment))} - ${formatTime(audioTimelineEnd(segment))}</div>
        <div style="margin-top:6px;"><strong>Duration:</strong> ${formatDurationSeconds(audioTimelineStart(segment), audioTimelineEnd(segment))} seconds</div>
        <div style="margin-top:6px;color:#a1a1aa;">Click the purple waveform on the timeline for cut/delete audio options.</div>
      `
      : "No custom scene audio selected. Use Scene Audio Options to drop or load audio for this scene.";
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncVideoModePanel();
    syncPreview(segment);
    updateAudioScrubbers();
  }

  function syncPreviewPlayback(current) {
    const playing = isTimelinePlaying();
    const segment = playing ? playbackSegmentAtTime(current) : activeSegment();
    const videoPath = selectedSegmentVideoPath(segment);
    if (!segment || !videoPath) {
      if (!previewVideo.paused) previewVideo.pause();
      return;
    }
    if (previewVideo.dataset.cacheKey !== selectedSegmentVideoCacheKey(segment, videoPath)) {
      setPreviewVideoSource(segment, videoPath);
      previewVideo.muted = false;
      previewVideo.style.display = "block";
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
    }
    const local = localPlaybackTime(segment, current);
    if (Number.isFinite(local) && Math.abs(Number(previewVideo.currentTime || 0) - local) > 0.2) {
      try {
        previewVideo.currentTime = local;
      } catch {
        // Some browsers reject seeking until metadata is ready. The next timeupdate will retry.
      }
    }
    if (isTimelinePlaying()) {
      previewVideo.play().catch(() => {});
    } else if (!previewVideo.paused) {
      previewVideo.pause();
    }
  }

  function updateAudioScrubbers() {
    const current = currentGlobalTime();
    const maxTime = timelineDuration();
    const followPlayback = Boolean((audio.src && !audio.paused) || (sceneAudio.src && !sceneAudio.paused) || state.isScrubbing);
    if (followPlayback) {
      const playbackSegment = playbackSegmentAtTime(current);
      if (playbackSegment && playbackSegment.id !== state.activeId) {
        state.activeId = playbackSegment.id;
        syncInspector();
        render();
        syncPreviewPlayback(current);
        return;
      }
    }
    globalScrub.max = String(Math.max(0, maxTime));
    if (!state.isScrubbing) globalScrub.value = String(current);
    globalScrubTime.textContent = `${formatTime(current)} / ${formatTime(maxTime)}`;
    playhead.style.left = `${12 + current * state.pxPerSecond}px`;
    syncPreviewPlayback(current);
  }

  function seekGlobalTimelineFromEvent(event) {
    const maxTime = timelineDuration();
    if (maxTime <= 0) return;
    const rect = timelineCanvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    setGlobalPlaybackTime(Math.max(0, Math.min(maxTime, x / state.pxPerSecond)));
    updateAudioScrubbers();
  }

  function setGlobalPlaybackTime(value) {
    const maxTime = timelineDuration();
    const time = Math.max(0, Math.min(maxTime, Number(value || 0)));
    state.sceneAudioGlobalTime = time;
    if (usingSceneAudioMode()) {
      const segment = audioSegmentAtTime(time) || playbackSegmentAtTime(time) || state.segments[state.segments.length - 1] || null;
      if (segment) state.activeId = segment.id;
      if (segment?.custom_audio_path) {
        const local = Math.max(0, Math.min(audioChunkDuration(segment), time - audioTimelineStart(segment))) + audioSourceStart(segment);
        if (state.sceneAudioSegmentId !== segment.id) {
          sceneAudio.src = audioUrl(segment.custom_audio_path);
          sceneAudio.load();
          state.sceneAudioSegmentId = segment.id;
        }
        try {
          sceneAudio.currentTime = local;
        } catch {
          // Browser may need metadata first; playback will retry on play.
        }
      } else {
        sceneAudio.pause();
        sceneAudio.removeAttribute("src");
        state.sceneAudioSegmentId = "";
      }
    } else if (audio.src) {
      audio.currentTime = time;
    }
  }

  function playSceneAudioFrom(time = currentGlobalTime()) {
    const maxTime = timelineDuration();
    const segment = audioSegmentAtTime(time) || playbackSegmentAtTime(time) || state.segments.find((item) => audioTimelineEnd(item) > time && item.custom_audio_path) || state.segments[0] || null;
    if (!segment) return;
    state.activeId = segment.id;
    state.sceneAudioGlobalTime = Math.max(audioTimelineStart(segment), Math.min(maxTime, Number(time || 0)));
    syncInspector();
    render();
    if (!segment.custom_audio_path) {
      sceneAudio.pause();
      sceneAudio.removeAttribute("src");
      state.sceneAudioSegmentId = "";
      const next = state.segments.find((item) => audioTimelineStart(item) > audioTimelineStart(segment) && item.custom_audio_path);
      if (next) {
        playSceneAudioFrom(audioTimelineStart(next));
        return;
      }
      updateAudioScrubbers();
      return;
    }
    const local = Math.max(0, state.sceneAudioGlobalTime - audioTimelineStart(segment)) + audioSourceStart(segment);
    sceneAudio.src = audioUrl(segment.custom_audio_path);
    sceneAudio.load();
    state.sceneAudioSegmentId = segment.id;
    const start = () => {
      try {
        sceneAudio.currentTime = local;
      } catch {
        // Metadata not ready yet.
      }
      sceneAudio.play().catch((error) => toast(String(error?.message || error), true));
    };
    if (Number.isFinite(sceneAudio.duration)) start();
    else sceneAudio.onloadedmetadata = start;
  }

  function beginGlobalTimelineScrub(event) {
    if (event.button !== 0) return;
    if (event.target !== timelineCanvas && event.target !== playhead) return;
    event.preventDefault();
    event.stopPropagation();
    state.isScrubbing = true;
    seekGlobalTimelineFromEvent(event);
    const move = (moveEvent) => seekGlobalTimelineFromEvent(moveEvent);
    const up = () => {
      state.isScrubbing = false;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      updateAudioScrubbers();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function syncZImageSettingsPanel() {
    const segment = activeSegment();
    useSceneZImageSettings.input.checked = Boolean(segment?.use_scene_zimage_settings);
    const settings = activeZImageSettings() || {};
    imageTriggerInput.value = settings.image_trigger_phrase || state.imageTriggerPhrase || "";
    zUnetPicker.input.value = settings.unet_name || "z_image_turbo_bf16.safetensors";
    zClipPicker.input.value = settings.clip_name || "qwen_3_4b.safetensors";
    zVaePicker.input.value = settings.vae_name || "ae.safetensors";
    zFirstWidth.value = settings.first_pass_width || 1280;
    zFirstHeight.value = settings.first_pass_height || 720;
    zSecondWidth.value = settings.second_pass_width || 1920;
    zSecondHeight.value = settings.second_pass_height || 1080;
    zSeed.value = settings.seed || 1;
    zSeedMode.value = settings.seed_mode || "fixed";
    zBatchSize.value = Math.max(1, Math.min(16, Number(settings.batch_size || 1)));
    zUseLora.input.checked = Boolean(settings.use_loras);
    zLoraCount.value = Number(settings.lora_count || 0);
    zLoraPanel.style.display = zUseLora.input.checked ? "flex" : "none";
    zLoraRows.style.display = zUseLora.input.checked && Number(zLoraCount.value || 0) > 0 ? "flex" : "none";
    zLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      const legacyStrength = config.strength;
      slot.row.style.display = index < Number(zLoraCount.value || 0) ? "grid" : "none";
      slot.picker.input.value = config.name || "[none]";
      slot.firstPassStrength.value = config.first_pass_strength ?? legacyStrength ?? 0.5;
      slot.secondPassStrength.value = config.second_pass_strength ?? legacyStrength ?? 1;
    });
    zUseImageToImage.input.checked = Boolean(settings.use_image_to_image);
    zI2IPanel.style.display = zUseImageToImage.input.checked ? "flex" : "none";
    const startStep = Math.max(1, Math.min(8, Number(settings.image_to_image_start_at_step || 5)));
    zI2ISlider.value = String(startStep);
    zI2IStartStep.value = String(startStep);
    zI2IPath.value = settings.image_to_image_path || settings.image_to_image_name || "";
  }

  function updateZLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(zLoraCount.value || 0)));
    zLoraPanel.style.display = zUseLora.input.checked ? "flex" : "none";
    zLoraRows.style.display = zUseLora.input.checked && count > 0 ? "flex" : "none";
    zLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function updateZImageToImageVisibility() {
    zI2IPanel.style.display = zUseImageToImage.input.checked ? "flex" : "none";
  }

  function saveZImageSettingsFromPanel() {
    pushHistory();
    const count = Math.max(0, Math.min(4, Number(zLoraCount.value || 0)));
    const currentSettings = activeZImageSettings() || {};
    const i2iPathValue = zI2IPath.value || "";
    const keepDataSource = Boolean(currentSettings.image_to_image_data && i2iPathValue === currentSettings.image_to_image_name);
    const settings = {
      unet_name: zUnetPicker.input.value || "z_image_turbo_bf16.safetensors",
      clip_name: zClipPicker.input.value || "qwen_3_4b.safetensors",
      vae_name: zVaePicker.input.value || "ae.safetensors",
      first_pass_width: Number(zFirstWidth.value || 1280),
      first_pass_height: Number(zFirstHeight.value || 720),
      second_pass_width: Number(zSecondWidth.value || 1920),
      second_pass_height: Number(zSecondHeight.value || 1080),
      seed: Number(zSeed.value || 1),
      seed_mode: zSeedMode.value || "fixed",
      batch_size: Math.max(1, Math.min(16, Number(zBatchSize.value || 1))),
      image_trigger_phrase: imageTriggerInput.value || "",
      use_loras: Boolean(zUseLora.input.checked),
      lora_count: count,
      loras: zLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        first_pass_strength: Number(slot.firstPassStrength.value || 0.5),
        second_pass_strength: Number(slot.secondPassStrength.value || 1),
        strength: Number(slot.secondPassStrength.value || 1),
      })),
      use_image_to_image: Boolean(zUseImageToImage.input.checked),
      image_to_image_start_at_step: Math.max(1, Math.min(8, Number(zI2IStartStep.value || zI2ISlider.value || 5))),
      image_to_image_path: keepDataSource ? "" : i2iPathValue,
      image_to_image_data: keepDataSource ? currentSettings.image_to_image_data || "" : "",
      image_to_image_name: keepDataSource ? currentSettings.image_to_image_name || "" : "",
    };
    const segment = activeSegment();
    if (segment?.use_scene_zimage_settings || hasMultiSceneBatchSelection()) {
      segment.zimage_settings = settings;
      if (segment) segment.use_scene_zimage_settings = true;
    } else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.zimageSettings = settings;
    }
    applyImageSettingsToMultiSelection("zimage", settings);
    updateZLoraVisibility();
    updateZImageToImageVisibility();
    renderList();
    return settings;
  }

  function advanceZImageSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    zSeed.value = String(settings.seed);
    if (activeSegment()?.use_scene_zimage_settings) {
      activeSegment().zimage_settings = settings;
    } else {
      state.zimageSettings = settings;
    }
  }

  function syncErnieImagePanel() {
    const segment = activeSegment();
    useSceneErnieImageSettings.input.checked = Boolean(segment?.use_scene_ernie_image_settings);
    const settings = activeErnieImageSettings() || {};
    ernieImageTriggerInput.value = settings.image_trigger_phrase || state.imageTriggerPhrase || "";
    ernieUnetPicker.input.value = chooseModelValue(
      ernieUnetPicker.options || [],
      settings.unet_name || "ernie\\ernie-image-turbo.safetensors",
      ["ernie\\ernie-image-turbo.safetensors", "ernie-image-turbo.safetensors"],
    ) || settings.unet_name || "";
    ernieClipPicker.input.value = settings.clip_name || "ministral-3-3b.safetensors";
    ernieVaePicker.input.value = chooseModelValue(
      ernieVaePicker.options || [],
      settings.vae_name || "flux\\flux2-vae.safetensors",
      ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"],
    ) || settings.vae_name || "";
    ernieWidth.value = settings.width || 1280;
    ernieHeight.value = settings.height || 720;
    ernieSeed.value = settings.seed || 1;
    ernieSeedMode.value = settings.seed_mode || "fixed";
    ernieBatchSize.value = Math.max(1, Math.min(16, Number(settings.batch_size || 1)));
    ernieUseLora.input.checked = Boolean(settings.use_loras);
    ernieLoraCount.value = Number(settings.lora_count || 0);
    updateErnieLoraVisibility();
    ernieLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.picker.input.value = config.name || "[none]";
      slot.strength.value = config.strength ?? 1;
    });
    ernieUseImageToImage.input.checked = Boolean(settings.use_image_to_image);
    updateErnieImageToImageVisibility();
    const startStep = Math.max(1, Math.min(8, Number(settings.image_to_image_start_at_step || 5)));
    ernieI2ISlider.value = String(startStep);
    ernieI2IStartStep.value = String(startStep);
    ernieI2IPath.value = settings.image_to_image_path || settings.image_to_image_name || "";
  }

  function updateErnieLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(ernieLoraCount.value || 0)));
    ernieLoraPanel.style.display = ernieUseLora.input.checked ? "flex" : "none";
    ernieLoraRows.style.display = ernieUseLora.input.checked && count > 0 ? "flex" : "none";
    ernieLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function updateErnieImageToImageVisibility() {
    ernieI2IPanel.style.display = ernieUseImageToImage.input.checked ? "flex" : "none";
  }

  function saveErnieImageSettingsFromPanel() {
    pushHistory();
    const count = Math.max(0, Math.min(4, Number(ernieLoraCount.value || 0)));
    const segment = activeSegment();
    const currentSettings = activeErnieImageSettings() || {};
    const i2iPathValue = ernieI2IPath.value || "";
    const keepDataSource = Boolean(currentSettings.image_to_image_data && i2iPathValue === currentSettings.image_to_image_name);
    const settings = {
      unet_name: ernieUnetPicker.input.value || "ernie\\ernie-image-turbo.safetensors",
      clip_name: ernieClipPicker.input.value || "ministral-3-3b.safetensors",
      vae_name: ernieVaePicker.input.value || "flux\\flux2-vae.safetensors",
      width: Number(ernieWidth.value || 1280),
      height: Number(ernieHeight.value || 720),
      seed: Number(ernieSeed.value || 1),
      seed_mode: ernieSeedMode.value || "fixed",
      batch_size: Math.max(1, Math.min(16, Number(ernieBatchSize.value || 1))),
      image_trigger_phrase: ernieImageTriggerInput.value || "",
      use_loras: Boolean(ernieUseLora.input.checked),
      lora_count: count,
      loras: ernieLoraSlots.map((slot) => ({ name: slot.picker.input.value || "[none]", strength: Number(slot.strength.value || 1) })),
      use_image_to_image: Boolean(ernieUseImageToImage.input.checked),
      image_to_image_start_at_step: Math.max(1, Math.min(8, Number(ernieI2IStartStep.value || ernieI2ISlider.value || 5))),
      image_to_image_path: keepDataSource ? "" : i2iPathValue,
      image_to_image_data: keepDataSource ? currentSettings.image_to_image_data || "" : "",
      image_to_image_name: keepDataSource ? currentSettings.image_to_image_name || "" : "",
    };
    if (segment?.use_scene_ernie_image_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_ernie_image_settings = true;
        segment.ernie_image_settings = settings;
      }
    }
    else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.ernieImageSettings = settings;
    }
    applyImageSettingsToMultiSelection("ernie_image", settings);
    updateErnieLoraVisibility();
    updateErnieImageToImageVisibility();
    return settings;
  }

  function advanceErnieSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    ernieSeed.value = String(settings.seed);
    if (activeSegment()?.use_scene_ernie_image_settings) activeSegment().ernie_image_settings = settings;
    else state.ernieImageSettings = settings;
  }

  function advanceZEnhanceSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    zEnhanceSeed.value = String(settings.seed);
    state.zEnhanceSettings = settings;
  }

  function renderFluxIngredientRows(listElement, ingredients, emptyText, onRemove) {
    listElement.innerHTML = "";
    if (!ingredients.length) {
      const empty = document.createElement("div");
      empty.textContent = emptyText;
      empty.style.cssText = "border:1px solid #27272a;border-radius:6px;background:#18181b;color:#a1a1aa;padding:8px;font-size:12px;";
      listElement.append(empty);
      return;
    }
    ingredients.forEach((item, index) => {
      const row = document.createElement("div");
      row.style.cssText = "display:grid;grid-template-columns:52px minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #27272a;border-radius:6px;background:#18181b;padding:8px;";
      const thumbWrap = document.createElement("div");
      thumbWrap.style.cssText = "width:52px;height:52px;border:1px solid #3f3f46;border-radius:6px;background:#09090b;overflow:hidden;display:flex;align-items:center;justify-content:center;";
      const imageSrc = item?.data || (item?.path ? makeEditorImageUrl(item.path) : "");
      if (imageSrc) {
        const thumb = document.createElement("img");
        thumb.alt = "";
        thumb.src = imageSrc;
        thumb.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
        thumbWrap.append(thumb);
      } else {
        const placeholder = document.createElement("div");
        placeholder.textContent = "IMG";
        placeholder.style.cssText = "font-size:10px;font-weight:900;color:#71717a;";
        thumbWrap.append(placeholder);
      }
      const label = document.createElement("div");
      label.textContent = `${index + 1}. ${item?.name || item?.path || "image ingredient"}`;
      label.title = item?.path || item?.name || "";
      label.style.cssText = "font-size:12px;color:#e4e4e7;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      const remove = makeMiniButton("Remove");
      remove.onclick = () => {
        onRemove(index);
      };
      row.append(thumbWrap, label, remove);
      listElement.append(row);
    });
  }

  function randomSeedValue() {
    return Math.floor(Math.random() * 2147483647) + 1;
  }

  function setImageSeedForCurrentMode(imageMode = state.imageModelMode || "zimage") {
    const seed = randomSeedValue();
    if (imageMode === "flux_klein") {
      const settings = activeFluxKleinSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_flux_klein_settings) activeSegment().flux_klein_settings = settings;
      else state.fluxKleinSettings = settings;
      fluxSeed.value = String(seed);
    } else if (imageMode === "ernie_image") {
      const settings = activeErnieImageSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_ernie_image_settings) activeSegment().ernie_image_settings = settings;
      else state.ernieImageSettings = settings;
      ernieSeed.value = String(seed);
    } else {
      const settings = activeZImageSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_zimage_settings) activeSegment().zimage_settings = settings;
      else state.zimageSettings = settings;
      zSeed.value = String(seed);
    }
    return seed;
  }

  function setVideoSeedRandom(segment = activeSegment()) {
    const settings = segment?.use_scene_i2v_video_settings
      ? cloneI2VVideoSettings(segment.i2v_video_settings || state.i2vVideoSettings)
      : cloneI2VVideoSettings(state.i2vVideoSettings);
    settings.seed = randomSeedValue();
    if (segment?.use_scene_i2v_video_settings) segment.i2v_video_settings = settings;
    else state.i2vVideoSettings = settings;
    if (!segment || segment.id === activeSegment()?.id) i2vSeedInput.value = String(settings.seed);
    return settings.seed;
  }

  function renderFluxIngredientList(segment = activeSegment()) {
    const ingredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
    renderFluxIngredientRows(fluxIngredientList, ingredients, "No scene-specific image ingredients loaded for this scene.", (index) => {
      const active = activeSegment();
      if (!active || !Array.isArray(active.flux_image_ingredients)) return;
      pushHistory();
      active.flux_image_ingredients.splice(index, 1);
      renderFluxIngredientList(active);
      renderNBIngredientList(active);
      render();
    });
  }

  function renderNBIngredientList(segment = activeSegment()) {
    const ingredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
    renderFluxIngredientRows(nbIngredientList, ingredients, "No scene-specific NanoBanana reference images loaded for this scene.", (index) => {
      const active = activeSegment();
      if (!active || !Array.isArray(active.flux_image_ingredients)) return;
      pushHistory();
      active.flux_image_ingredients.splice(index, 1);
      renderFluxIngredientList(active);
      renderNBIngredientList(active);
      render();
    });
  }

  function renderFluxGlobalIngredientList() {
    const ingredients = Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [];
    renderFluxIngredientRows(fluxGlobalIngredientList, ingredients, "No global Flux/Klein image ingredients loaded.", (index) => {
      pushHistory();
      state.fluxGlobalImageIngredients.splice(index, 1);
      renderFluxGlobalIngredientList();
      render();
    });
    renderFluxIngredientRows(nbGlobalIngredientList, ingredients, "No global Nano B reference images loaded.", (index) => {
      pushHistory();
      state.fluxGlobalImageIngredients.splice(index, 1);
      renderFluxGlobalIngredientList();
      render();
    });
  }

  function syncFluxGlobalIngredientPanel() {
    useFluxGlobalIngredients.input.checked = Boolean(state.useFluxGlobalImageIngredients);
    nbUseGlobalIngredients.input.checked = Boolean(state.useFluxGlobalImageIngredients);
    fluxGlobalIngredientPanel.style.display = state.useFluxGlobalImageIngredients ? "flex" : "none";
    nbGlobalIngredientPanel.style.display = state.useFluxGlobalImageIngredients ? "flex" : "none";
  }

  function mergedFluxImageIngredients(segment = activeSegment()) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const ingredients = [];
    const addUnique = (item) => {
      if (!item || typeof item !== "object") return;
      const path = String(item.path || "");
      const data = String(item.data || "");
      const name = String(item.name || "");
      if (!path && !data) return;
      const key = path || data || name;
      if (ingredients.some((existing) => (existing.path || existing.data || existing.name) === key)) return;
      ingredients.push({ path, data, name: name || path?.split?.(/[\\/]/)?.pop?.() || "reference.png" });
    };
    if (refs.use_subject_reference) addUnique(refs.subject?.image);
    if (refs.use_location_references && segment) {
      const locId = refs.scene_map?.[segment.id] || refs.scene_map?.[String(segmentIndexInfo(segment).index + 1)] || "";
      const location = refs.locations.find((item) => item.id === locId);
      addUnique(location?.image);
    }
    if (refs.include_manual_ingredients !== false) {
      const globalIngredients = state.useFluxGlobalImageIngredients && Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [];
      const sceneIngredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
      globalIngredients.forEach(addUnique);
      sceneIngredients.forEach(addUnique);
    }
    return ingredients;
  }

  function fluxReferenceContextForSegment(segment = activeSegment()) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const context = {
      has_subject_reference: false,
      has_location_reference: false,
      subject_description: "",
      location_name: "",
      location_description: "",
    };
    const subjectImage = refs.subject?.image || {};
    if (refs.use_subject_reference && (subjectImage.path || subjectImage.data)) {
      context.has_subject_reference = true;
      context.subject_description = refs.subject?.description || "";
    }
    if (refs.use_location_references && segment) {
      const locId = refs.scene_map?.[segment.id] || refs.scene_map?.[String(segmentIndexInfo(segment).index + 1)] || "";
      const location = refs.locations.find((item) => item.id === locId);
      const image = location?.image || {};
      if (location && (image.path || image.data)) {
        context.has_location_reference = true;
        context.location_name = location.name || "";
        context.location_description = location.description || "";
      }
    }
    return context;
  }

  function syncFluxKleinPanel() {
    const segment = activeSegment();
    const settings = activeFluxKleinSettings() || {};
    useSceneFluxKleinSettings.input.checked = Boolean(segment?.use_scene_flux_klein_settings);
    fluxImageTriggerInput.value = settings.image_trigger_phrase || state.imageTriggerPhrase || "";
    const mode = state.imageModelMode || settings.image_model_mode || "zimage";
    state.imageModelMode = mode;
    settings.image_model_mode = mode;
    settings.enabled = mode === "flux_klein";
    zImageModePanel.style.display = mode === "zimage" ? "flex" : "none";
    fluxKleinModePanel.style.display = mode === "flux_klein" || mode === "nano_banana" ? "flex" : "none";
    ernieImageModePanel.style.display = mode === "ernie_image" ? "flex" : "none";
    zEnhancePanel.style.display = mode === "z_enhance" ? "flex" : "none";
    previewButton.style.display = mode === "zimage" ? "" : "none";
    ernieCreateButton.style.display = mode === "ernie_image" ? "" : "none";
    useFluxKlein.input.checked = mode === "flux_klein";
    fluxKleinPanel.style.display = mode === "flux_klein" ? "flex" : "none";
    nbImagePanel.style.display = mode === "nano_banana" ? "flex" : "none";
    ernieImagePanel.style.display = mode === "ernie_image" ? "flex" : "none";
    for (const card of [zImageCard, fluxKleinCard, nbImageCard, ernieImageCard, zEnhanceCard]) {
      const active = card.dataset.model === mode;
      card.style.borderColor = active ? "#0891b2" : "#3f3f46";
      card.style.background = active ? "#06b6d4" : "#27272a";
      card.style.color = active ? "#082f49" : "#f4f4f5";
      card.style.boxShadow = active ? "inset 0 0 0 1px rgba(8,47,73,.22)" : "none";
    }
    loadCustomImageButton.style.borderColor = "#3f3f46";
    loadCustomImageButton.style.background = "#27272a";
    loadCustomImageButton.style.color = "#f4f4f5";
    loadCustomImageButton.style.boxShadow = "none";
    syncFluxGlobalIngredientPanel();
    renderFluxGlobalIngredientList();
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    fluxNotes.value = segment?.flux_notes || "";
    fluxPrompt.value = segment?.t2i_prompt || segment?.flux_prompt || "";
    fluxUnetPicker.input.value = chooseModelValue(
      fluxUnetPicker.options || [],
      settings.unet_name || "flux\\flux-2-klein-4b-fp8.safetensors",
      ["flux\\flux-2-klein-4b-fp8.safetensors", "flux-2-klein-4b-fp8.safetensors"],
    ) || settings.unet_name || "";
    fluxClipPicker.input.value = settings.clip_name || "qwen_3_4b.safetensors";
    fluxVaePicker.input.value = chooseModelValue(
      fluxVaePicker.options || [],
      settings.vae_name || "flux\\flux2-vae.safetensors",
      ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"],
    ) || settings.vae_name || "";
    fluxWidth.value = settings.width || 1024;
    fluxHeight.value = settings.height || 576;
    fluxSeed.value = settings.seed || 100;
    fluxUseTextOnlyGemmaPrompt.input.checked = Boolean(settings.use_text_only_gemma_prompt);
    fluxUseLora.input.checked = Boolean(settings.use_loras);
    fluxLoraCount.value = Number(settings.lora_count || 0);
    fluxLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.row.style.display = index < Number(fluxLoraCount.value || 0) ? "grid" : "none";
      slot.picker.input.value = config.name || "[none]";
      slot.strength.value = config.strength ?? 1;
    });
    updateFluxLoraVisibility();
    syncErnieImagePanel();
    syncNBImagePanel();
  }

  function updateFluxLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(fluxLoraCount.value || 0)));
    fluxLoraPanel.style.display = fluxUseLora.input.checked ? "flex" : "none";
    fluxLoraRows.style.display = fluxUseLora.input.checked && count > 0 ? "flex" : "none";
    fluxLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function saveFluxKleinSettingsFromPanel() {
    pushHistory();
    const count = Math.max(0, Math.min(4, Number(fluxLoraCount.value || 0)));
    const current = activeFluxKleinSettings() || {};
    const segment = activeSegment();
    if (segment) {
      if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
      segment.flux_notes = fluxNotes.value || "";
      segment.flux_prompt = fluxPrompt.value || "";
      segment.t2i_prompt = segment.flux_prompt;
    }
    const settings = {
      enabled: Boolean(useFluxKlein.input.checked),
      image_model_mode: state.imageModelMode || current.image_model_mode || (useFluxKlein.input.checked ? "flux_klein" : "zimage"),
      unet_name: fluxUnetPicker.input.value || "",
      clip_name: fluxClipPicker.input.value || "",
      vae_name: fluxVaePicker.input.value || "",
      width: Number(fluxWidth.value || 1024),
      height: Number(fluxHeight.value || 576),
      seed: Number(fluxSeed.value || 100),
      use_text_only_gemma_prompt: Boolean(fluxUseTextOnlyGemmaPrompt.input.checked),
      use_loras: Boolean(fluxUseLora.input.checked),
      lora_count: count,
      loras: fluxLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        strength: Number(slot.strength.value || 1),
      })),
      image_trigger_phrase: fluxImageTriggerInput.value || "",
    };
    if (segment?.use_scene_flux_klein_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_flux_klein_settings = true;
        segment.flux_klein_settings = settings;
      }
    }
    else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.fluxKleinSettings = settings;
    }
    applyImageSettingsToMultiSelection("flux_klein", settings);
    updateFluxLoraVisibility();
    return {
      ...settings,
      image_ingredients: mergedFluxImageIngredients(segment),
      use_global_image_ingredients: Boolean(state.useFluxGlobalImageIngredients),
      global_image_ingredients: Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [],
      scene_image_ingredients: Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [],
      notes: segment?.flux_notes || "",
      prompt: segment?.flux_prompt || "",
      reference_context: fluxReferenceContextForSegment(segment),
    };
  }

  function syncNBImagePanel() {
    const segment = activeSegment();
    const settings = activeNBImageSettings() || {};
    useSceneNBImageSettings.input.checked = Boolean(segment?.use_scene_nb_image_settings);
    nbApiKey.value = settings.api_key || "";
    nbModelSelect.value = NB_IMAGE_MODELS.includes(settings.model) ? settings.model : DEFAULT_NB_IMAGE_MODEL;
    nbUseTextOnlyGemmaPrompt.input.checked = Boolean(settings.use_text_only_gemma_prompt);
    nbNotes.value = segment?.nb_notes || segment?.flux_notes || segment?.notes || "";
    nbPrompt.value = segment?.nb_prompt || segment?.t2i_prompt || "";
    renderNBIngredientList(segment);
  }

  function saveNBImageSettingsFromPanel() {
    pushHistory();
    const current = activeNBImageSettings() || {};
    const segment = activeSegment();
    if (segment) {
      if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
      segment.nb_notes = nbNotes.value || "";
      segment.nb_prompt = nbPrompt.value || "";
      segment.t2i_prompt = segment.nb_prompt;
    }
    const settings = {
      ...current,
      api_key: nbApiKey.value || "",
      model: nbModelSelect.value || DEFAULT_NB_IMAGE_MODEL,
      use_text_only_gemma_prompt: Boolean(nbUseTextOnlyGemmaPrompt.input.checked),
    };
    if (segment?.use_scene_nb_image_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_nb_image_settings = true;
        segment.nb_image_settings = settings;
      }
    } else {
      state.nbImageSettings = settings;
    }
    applyImageSettingsToMultiSelection("nano_banana", settings);
    return {
      ...settings,
      image_ingredients: mergedFluxImageIngredients(segment),
      notes: segment?.nb_notes || segment?.flux_notes || segment?.notes || "",
      prompt: segment?.nb_prompt || "",
      reference_context: fluxReferenceContextForSegment(segment),
    };
  }

  function activeScenePromptForEnhance({ copyFallback = false } = {}) {
    const segment = activeSegment();
    const explicitEnhancePrompt = String(segment?.enhance_prompt || "").trim();
    if (explicitEnhancePrompt) return { prompt: explicitEnhancePrompt, source: "enhance" };

    const scenePrompt = String(segment?.t2i_prompt || "").trim();
    if (scenePrompt) {
      if (copyFallback && segment) {
        segment.enhance_prompt = scenePrompt;
        zEnhancePromptPreview.value = scenePrompt;
      }
      return { prompt: scenePrompt, source: "scene" };
    }

    return { prompt: "", source: "" };
  }

  function syncZEnhanceSettingsPanel() {
    const settings = state.zEnhanceSettings || {};
    const promptInfo = activeScenePromptForEnhance();
    const segment = activeSegment();
    zEnhanceGemmaNotes.value = segment?.enhance_notes || "";
    zEnhancePromptPreview.value = promptInfo.prompt || "";
    zEnhanceUnetPicker.input.value = settings.unet_name || "z_image_turbo_bf16.safetensors";
    zEnhanceClipPicker.input.value = settings.clip_name || "qwen_3_4b.safetensors";
    zEnhanceVaePicker.input.value = settings.vae_name || "ae.safetensors";
    zEnhanceWidth.value = settings.width || 1920;
    zEnhanceHeight.value = settings.height || 1080;
    zEnhanceSeed.value = settings.seed || 1;
    zEnhanceSeedMode.value = settings.seed_mode || "randomize";
    zEnhanceAmount.value = Math.max(1, Math.min(20, Number(settings.enhance_amount || 8)));
    zEnhanceAmountValue.textContent = `Enhance amount: ${zEnhanceAmount.value}`;
    zEnhanceUseLora.input.checked = Boolean(settings.use_loras);
    zEnhanceLoraCount.value = Number(settings.lora_count || 0);
    zEnhanceLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.picker.input.value = config.name || "[none]";
      slot.strength.value = config.strength ?? 1;
    });
    updateZEnhanceLoraVisibility();
  }

  function saveZEnhanceSettingsFromPanel() {
    const segment = activeSegment();
    if (segment) {
      segment.enhance_notes = zEnhanceGemmaNotes.value || "";
      segment.enhance_prompt = zEnhancePromptPreview.value || "";
    }
    const count = Math.max(0, Math.min(4, Number(zEnhanceLoraCount.value || 0)));
    state.zEnhanceSettings = {
      unet_name: zEnhanceUnetPicker.input.value || "",
      clip_name: zEnhanceClipPicker.input.value || "",
      vae_name: zEnhanceVaePicker.input.value || "",
      width: Number(zEnhanceWidth.value || 1920),
      height: Number(zEnhanceHeight.value || 1080),
      seed: Number(zEnhanceSeed.value || 1),
      seed_mode: zEnhanceSeedMode.value || "fixed",
      enhance_amount: Math.max(1, Math.min(20, Number(zEnhanceAmount.value || 8))),
      use_loras: Boolean(zEnhanceUseLora.input.checked),
      lora_count: count,
      loras: zEnhanceLoraSlots.map((slot) => ({ name: slot.picker.input.value || "[none]", strength: Number(slot.strength.value || 1) })),
    };
    updateZEnhanceLoraVisibility();
    zEnhanceAmountValue.textContent = `Enhance amount: ${state.zEnhanceSettings.enhance_amount}`;
    return state.zEnhanceSettings;
  }

  function syncI2VVideoSettingsPanel() {
    const segment = activeSegment();
    useSceneI2VVideoSettings.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
    videoSettingsScopeNote.textContent = segment?.use_scene_i2v_video_settings
      ? "This scene is using custom video models, settings, and LoRAs from the Models tab."
      : "This scene is using global video models, settings, and LoRAs. Enable custom scene video settings in the Models tab.";
    const settings = activeI2VVideoSettings() || {};
    videoTriggerInput.value = settings.video_trigger_phrase || state.videoTriggerPhrase || "";
    i2vUnetPicker.input.value = BAD_I2V_UNET_ALIASES.has(settings.unet_name) ? DEFAULT_I2V_UNET : settings.unet_name || "";
    i2vVaePicker.input.value = settings.vae_name || "";
    i2vClip1Picker.input.value = settings.clip_name1 || "";
    i2vClip2Picker.input.value = settings.clip_name2 || "";
    i2vUpscalePicker.input.value = settings.upscale_model_name || "";
    i2vAudioVaePicker.input.value = settings.audio_vae_name || "";
    i2vFpsInput.value = settings.fps || 24;
    i2vWidthInput.value = settings.width || 1920;
    i2vHeightInput.value = settings.height || 1080;
    i2vSeedInput.value = settings.seed || 69;
    i2vUseLora.input.checked = Boolean(settings.use_loras);
    i2vLoraCount.value = Number(settings.lora_count || 0);
    i2vLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.picker.input.value = config.name || "[none]";
      const legacyStrength = config.strength ?? 1;
      slot.firstPassStrength.value = config.first_pass_strength ?? legacyStrength;
      slot.secondPassStrength.value = config.second_pass_strength ?? legacyStrength;
    });
    updateI2VLoraVisibility();
  }

  function saveI2VVideoSettingsFromPanel() {
    const count = Math.max(0, Math.min(4, Number(i2vLoraCount.value || 0)));
    const segment = activeSegment();
    const settings = {
      unet_name: BAD_I2V_UNET_ALIASES.has(i2vUnetPicker.input.value) ? DEFAULT_I2V_UNET : i2vUnetPicker.input.value || "",
      vae_name: i2vVaePicker.input.value || "",
      clip_name1: i2vClip1Picker.input.value || "",
      clip_name2: i2vClip2Picker.input.value || "",
      upscale_model_name: i2vUpscalePicker.input.value || "",
      audio_vae_name: i2vAudioVaePicker.input.value || "",
      fps: Number(i2vFpsInput.value || 24),
      width: Number(i2vWidthInput.value || 1920),
      height: Number(i2vHeightInput.value || 1080),
      seed: Number(i2vSeedInput.value || 69),
      video_trigger_phrase: videoTriggerInput.value || "",
      use_loras: Boolean(i2vUseLora.input.checked),
      lora_count: count,
      loras: i2vLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        first_pass_strength: Number(slot.firstPassStrength.value || 1),
        second_pass_strength: Number(slot.secondPassStrength.value || 1),
      })),
    };
    if (segment?.use_scene_i2v_video_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_i2v_video_settings = true;
        segment.i2v_video_settings = settings;
      }
    }
    else {
      state.videoTriggerPhrase = settings.video_trigger_phrase || "";
      state.i2vVideoSettings = settings;
    }
    applyVideoSettingsToMultiSelection(settings);
    updateI2VLoraVisibility();
    return settings;
  }

  function currentVideoMode() {
    return state.videoModelMode === "t2v" ? "t2v" : "i2v";
  }

  function syncVideoModePanel() {
    const mode = currentVideoMode();
    state.videoModelMode = mode;
    for (const card of [imageToVideoCard, textToVideoCard]) {
      const active = card.dataset.model === mode;
      card.style.borderColor = active ? "#71717a" : "#3f3f46";
      card.style.background = active ? "#52525b" : "#27272a";
      card.style.color = "#f4f4f5";
      card.style.boxShadow = active ? "inset 0 0 0 1px rgba(244,244,245,.12)" : "none";
    }
    const isT2V = mode === "t2v";
    useI2VVisionReference.wrapper.style.display = isT2V ? "none" : "flex";
    i2vReferenceNote.style.display = isT2V ? "none" : "";
    useT2VVisionReference.wrapper.style.display = isT2V ? "flex" : "none";
    t2vReferenceNote.style.display = isT2V ? "" : "none";
    t2vRefImagePanel.style.display = isT2V && useT2VVisionReference.input.checked ? "flex" : "none";
    createI2VButton.textContent = isT2V ? "Gemma T2V" : "Gemma I2V";
    i2vNotesInput.placeholder = isT2V
      ? "Extra text-to-video motion notes, camera movement, character movement..."
      : "Extra video motion notes, camera movement, character movement...";
    i2vPrompt.placeholder = isT2V ? "Text-to-video prompt..." : "Image-to-video prompt...";
  }

  function updateActiveFromInputs() {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.label = labelInput.value || "Scene";
    const isOverlay = segmentTrack(segment) === "overlay";
    if ((!state.timingFrozen || isOverlay) && !hasLockedVideo(segment)) {
      segment.start = Math.max(0, Number(startInput.value || 0));
      segment.end = Math.max(segment.start + 0.1, Number(endInput.value || segment.start + 4));
    }
    segment.notes = notesInput.value || "";
    if (state.imageModelMode === "ernie_image") {
      segment.notes = ernieNotesInput.value || "";
    } else if (state.imageModelMode === "nano_banana") {
      segment.nb_notes = nbNotes.value || "";
    }
    segment.i2v_notes = i2vNotesInput.value || "";
    const editedT2IPrompt = state.imageModelMode === "ernie_image"
      ? ernieT2IPrompt.value || ""
      : state.imageModelMode === "flux_klein"
        ? fluxPrompt.value || ""
        : state.imageModelMode === "nano_banana"
          ? nbPrompt.value || ""
        : t2iPrompt.value || "";
    segment.t2i_prompt = editedT2IPrompt;
    segment.flux_prompt = editedT2IPrompt;
    segment.nb_prompt = editedT2IPrompt;
    if (t2iPrompt.value !== editedT2IPrompt) t2iPrompt.value = editedT2IPrompt;
    if (ernieT2IPrompt.value !== editedT2IPrompt) ernieT2IPrompt.value = editedT2IPrompt;
    if (fluxPrompt.value !== editedT2IPrompt) fluxPrompt.value = editedT2IPrompt;
    if (nbPrompt.value !== editedT2IPrompt) nbPrompt.value = editedT2IPrompt;
    segment.i2v_prompt = i2vPrompt.value || "";
    segment.enhance_notes = zEnhanceGemmaNotes.value || "";
    segment.enhance_prompt = zEnhancePromptPreview.value || segment.enhance_prompt || "";
    segment.use_vision_reference = Boolean(useVisionReference.input.checked);
    if (state.imageModelMode === "ernie_image") {
      segment.use_vision_reference = Boolean(ernieUseVisionReference.input.checked);
    }
    segment.use_i2v_vision_reference = Boolean(useI2VVisionReference.input.checked);
    segment.use_t2v_vision_reference = Boolean(useT2VVisionReference.input.checked);
    segment.ref_image_path = refImageInput.value || "";
    refImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
    ernieRefImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
    t2vRefImagePanel.style.display = currentVideoMode() === "t2v" && segment.use_t2v_vision_reference ? "flex" : "none";
    if (!state.timingFrozen && !hasLockedVideo(segment) && !isOverlay) normalizeSegments(segment);
    if (isOverlay) sortSegments(state.overlaySegments);
    render();
  }

  function normalizeSegments(changedSegment, changedIndex = null) {
    sortSegments(state.segments);
    const minDuration = 0.1;
    const active = changedSegment || activeSegment();
    const activeIndex = changedIndex ?? state.segments.findIndex((segment) => segment.id === active?.id);
    if (activeIndex < 0) return;

    active.start = Math.max(0, Number(active.start || 0));
    active.end = Math.max(active.start + minDuration, Number(active.end || active.start + 4));

    const prev = state.segments[activeIndex - 1] || null;
    const next = state.segments[activeIndex + 1] || null;

    if (!prev) {
      active.start = 0;
    } else {
      if (hasLockedVideo(prev)) {
        active.start = Math.max(active.start, Number(prev.end || 0));
      } else {
        const prevStart = Number(prev.start || 0);
        active.start = Math.max(active.start, prevStart + minDuration);
        prev.end = active.start;
      }
    }

    active.end = Math.max(active.start + minDuration, active.end);

    if (next) {
      if (hasLockedVideo(next)) {
        active.end = Math.min(active.end, Number(next.start || active.end));
      } else {
        const nextEnd = Number(next.end || active.end + minDuration);
        active.end = Math.min(active.end, nextEnd - minDuration);
      }
      active.end = Math.max(active.start + minDuration, active.end);
      if (!hasLockedVideo(next)) next.start = active.end;
    } else if (state.duration > 0) {
      active.end = Math.min(active.end, state.duration);
      active.end = Math.max(active.start + minDuration, active.end);
    }

    if (prev && !hasLockedVideo(prev)) prev.end = active.start;
    if (next) {
      if (!hasLockedVideo(next)) {
        next.start = active.end;
        if (next.end < next.start + minDuration) next.end = next.start + minDuration;
      }
    }
    sortSegments(state.segments);
  }

  function shiftSegmentTiming(segment, delta) {
    const amount = Number(delta || 0);
    if (!segment || !amount) return;
    segment.start = Number(segment.start || 0) + amount;
    segment.end = Number(segment.end || 0) + amount;
    if (Number.isFinite(Number(segment.custom_audio_timeline_start))) {
      segment.custom_audio_timeline_start = Number(segment.custom_audio_timeline_start || 0) + amount;
    }
  }

  function timelineHeight() {
    return WAVEFORM_MODES[state.waveformMode]?.height || WAVEFORM_MODES.medium.height;
  }

  function snapTimeToBeat(time) {
    if (!state.snapToBeats || !state.showBeatMarkers || !Array.isArray(state.beats) || !state.beats.length) return time;
    const value = Number(time || 0);
    let best = value;
    let bestDelta = Infinity;
    for (const beat of state.beats) {
      const beatTime = Number(beat || 0);
      const delta = Math.abs(beatTime - value);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = beatTime;
      }
    }
    return bestDelta <= 0.14 ? best : value;
  }

  function drawWaveform() {
    const height = timelineHeight();
    const width = Math.max(900, Math.ceil(Math.max(1, timelineDuration()) * state.pxPerSecond));
    timelineCanvas.height = height;
    timelineCanvas.width = width;
    timelineCanvas.style.height = `${height}px`;
    timelineCanvas.style.width = `${width}px`;
    segmentLayer.style.height = `${height}px`;
    segmentLayer.style.width = `${width}px`;
    playhead.style.height = `${height}px`;
    const ctx = timelineCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, timelineCanvas.height);
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, width, timelineCanvas.height);
    ctx.strokeStyle = "#164e63";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const waveTop = usingSceneAudioMode()
      ? TIMELINE_SCENE_AUDIO_TOP + TIMELINE_SCENE_AUDIO_HEIGHT + 14
      : TIMELINE_SEGMENT_TOP + TIMELINE_SEGMENT_HEIGHT + 14;
    const waveBottom = timelineCanvas.height - 10;
    const waveHeight = Math.max(24, waveBottom - waveTop);
    const mid = waveTop + waveHeight / 2;
    const peaks = state.peaks.length && !usingSceneAudioMode() ? state.peaks : [0];
    const gain = WAVEFORM_MODES[state.waveformMode]?.gain || 1;
    for (let x = 0; x < width; x++) {
      const index = Math.floor((x / width) * peaks.length);
      const amp = Math.min(1, Math.max(0.02, (peaks[index] || 0) * gain));
      ctx.moveTo(x, mid - amp * (waveHeight / 2));
      ctx.lineTo(x, mid + amp * (waveHeight / 2));
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(103,232,249,.16)";
    ctx.fillRect(0, waveTop - 1, width, 1);
    ctx.fillStyle = "#67e8f9";
    ctx.font = "11px sans-serif";
    for (let sec = 0; sec <= timelineDuration(); sec += 10) {
      const x = sec * state.pxPerSecond;
      ctx.fillRect(x, 0, 1, timelineCanvas.height);
      ctx.fillText(formatTime(sec), x + 3, 12);
    }
  }

  function drawSegmentAudioWaveform(canvas, peaks) {
    const values = Array.isArray(peaks) && peaks.length ? peaks : [];
    const width = Math.max(1, canvas.width || 1);
    const height = Math.max(1, canvas.height || 1);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    if (!values.length) return;
    const mid = height / 2;
    ctx.strokeStyle = "rgba(216, 180, 254, .9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const index = Math.floor((x / width) * values.length);
      const amp = Math.min(1, Math.max(0.03, Number(values[index] || 0) * 1.6));
      ctx.moveTo(x, mid - amp * (height / 2));
      ctx.lineTo(x, mid + amp * (height / 2));
    }
    ctx.stroke();
  }

  function renderBeatMarkersOverlay() {
    const visibleBeats = state.showBeatMarkers && Array.isArray(state.beats) ? state.beats : [];
    if (!visibleBeats.length) return;
    const top = Math.max(2, TIMELINE_SEGMENT_TOP - 18);
    for (const beatTime of visibleBeats) {
      const x = Number(beatTime || 0) * state.pxPerSecond;
      if (!Number.isFinite(x)) continue;
      const marker = document.createElement("div");
      marker.title = `Beat ${formatTime(beatTime)}`;
      marker.style.cssText = `
        position:absolute;left:${x}px;top:${top}px;width:2px;height:12px;
        z-index:4;pointer-events:none;background:rgba(244,244,245,.9);
        border-radius:2px;box-shadow:0 0 4px rgba(244,244,245,.35);
      `;
      segmentLayer.append(marker);
    }
  }

  function renderSegments() {
    segmentLayer.textContent = "";
    ensureAllSegmentRuntimeFields();
    const overlayLabel = document.createElement("div");
    overlayLabel.textContent = "INSERTS";
    overlayLabel.style.cssText = `position:absolute;left:4px;top:${TIMELINE_OVERLAY_TOP - 11}px;color:#a5f3fc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
    const baseLabel = document.createElement("div");
    baseLabel.textContent = "BASE";
    baseLabel.style.cssText = `position:absolute;left:4px;top:${TIMELINE_SEGMENT_TOP - 14}px;color:#a5f3fc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
    segmentLayer.append(overlayLabel, baseLabel);
    for (const segment of [...state.overlaySegments, ...state.segments]) {
      const isOverlay = segmentTrack(segment) === "overlay";
      const blockTop = isOverlay ? TIMELINE_OVERLAY_TOP : TIMELINE_SEGMENT_TOP;
      const blockHeight = isOverlay ? TIMELINE_OVERLAY_HEIGHT : TIMELINE_SEGMENT_HEIGHT;
      const block = document.createElement("button");
      block.type = "button";
      block.innerHTML = `<span style="position:relative;z-index:2;display:block;font-weight:900;">${escapeHtml(segment.label || (isOverlay ? "Insert" : "Scene"))}</span><span style="position:relative;z-index:2;display:block;margin-top:3px;font-size:10px;color:#d4d4d8;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</span>`;
      const left = segment.start * state.pxPerSecond;
      const width = Math.max(24, (segment.end - segment.start) * state.pxPerSecond);
      const previewThumbPath = selectedSegmentImageThumbnailPath(segment);
      const thumb = previewThumbPath ? makeEditorImageUrl(previewThumbPath) : "";
      const videoThumbPath = thumb ? "" : selectedSegmentVideoThumbnailPath(segment);
      const inserted = !isOverlay && state.srtMode && segment.source !== "srt";
      const lockedByVideo = hasLockedVideo(segment);
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      const isMultiSelected = isSegmentMultiSelected(segment);
      const borderColor = isActive || isMultiSelected ? "#ef4444" : lockedByVideo ? "#a3e635" : isOverlay ? "#f97316" : inserted ? "#f59e0b" : "#0891b2";
      const borderWidth = isActive || isMultiSelected ? "3px" : "1px";
      const shadow = isActive || isMultiSelected ? "0 0 0 2px rgba(239,68,68,.28), 0 0 18px rgba(239,68,68,.55)" : "none";
      block.style.cssText = `
        position:absolute;left:${left}px;top:${blockTop}px;width:${width}px;height:${blockHeight}px;
        border:${borderWidth} solid ${borderColor};
        border-radius:5px;background:${thumb ? `linear-gradient(rgba(0,0,0,.18),rgba(0,0,0,.18)), url("${thumb}") center / auto 100% repeat-x` : isOverlay ? "#7c2d12" : inserted ? "#92400e" : segment.image ? "#166534" : "#164e63"};
        color:#f4f4f5;font-size:11px;font-weight:800;overflow:hidden;cursor:pointer;pointer-events:auto;
        box-shadow:${shadow};
      `;
      if (!thumb && videoThumbPath) appendTimelineVideoThumbnail(block, segment);
      block.title = lockedByVideo ? "This scene has a generated video, so timing is locked." : "";
      const dragImageSource = segmentImageSource(segment);
      if (dragImageSource) {
        block.draggable = true;
        block.ondragstart = (event) => {
          event.dataTransfer.setData("application/x-vrgdg-segment-id", segment.id);
          event.dataTransfer.setData("text/plain", segment.label || "Scene image");
          event.dataTransfer.effectAllowed = "copy";
        };
      }
      const hasPreviewImage = Boolean(segmentImageSource(segment));
      const hasPreviewVideo = Boolean(selectedSegmentVideoPath(segment));
      if ((segment.preview_mode === "video" && segment.video_history.length) || (segment.preview_mode !== "video" && segment.image_history.length)) {
        const historyButton = document.createElement("span");
        const isVideoMode = segment.preview_mode === "video";
        const count = isVideoMode ? segment.video_history.length : segment.image_history.length;
        const index = isVideoMode ? Number(segment.video_history_index || 0) : Number(segment.image_history_index || 0);
        historyButton.textContent = `${Math.max(1, index + 1)}/${count}`;
        historyButton.title = isVideoMode ? "Cycle generated video previews for this scene." : "Cycle generated image previews for this scene.";
        historyButton.style.cssText = `position:absolute;right:13px;top:5px;min-width:34px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid ${isVideoMode ? "#a78bfa" : "#67e8f9"};border-radius:4px;background:${isVideoMode ? "rgba(59,7,100,.92)" : "rgba(8,47,73,.92)"};color:${isVideoMode ? "#f3e8ff" : "#e0f2fe"};font-size:10px;font-weight:900;z-index:3;`;
        historyButton.onpointerdown = (event) => event.stopPropagation();
        historyButton.onclick = (event) => {
          event.stopPropagation();
          if (isVideoMode) cycleSegmentVideoHistory(segment);
          else cycleSegmentImageHistory(segment);
        };
        block.append(historyButton);
      }
      if (hasPreviewImage && hasPreviewVideo) {
        const modeButton = document.createElement("span");
        modeButton.textContent = segment.preview_mode === "image" ? "I" : "V";
        modeButton.title = "Switch preview between image and video.";
        modeButton.style.cssText = "position:absolute;right:13px;top:29px;min-width:34px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid #a3e635;border-radius:4px;background:rgba(20,83,45,.92);color:#ecfccb;font-size:10px;font-weight:900;z-index:3;";
        modeButton.onpointerdown = (event) => event.stopPropagation();
        modeButton.onclick = (event) => {
          event.stopPropagation();
          toggleSegmentPreviewMode(segment);
        };
        block.append(modeButton);
      }
      const leftHandle = document.createElement("div");
      leftHandle.style.cssText = "position:absolute;left:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.25);cursor:ew-resize;z-index:4;";
      const rightHandle = document.createElement("div");
      rightHandle.style.cssText = "position:absolute;right:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.25);cursor:ew-resize;z-index:4;";
      block.append(leftHandle, rightHandle);
      block.onclick = () => handleSegmentPick(segment);
      block.oncontextmenu = (event) => openSegmentContextMenu(event, segment);
      enableImageDrop(block, segment);
      makeDragHandle(block, segment, "move");
      makeDragHandle(leftHandle, segment, "start");
      makeDragHandle(rightHandle, segment, "end");
      segmentLayer.append(block);
      if (!isOverlay && segment.custom_audio_peaks?.length) {
        const audioStart = audioTimelineStart(segment);
        const audioDuration = Math.max(0.1, audioChunkDuration(segment));
        const audioLeft = audioStart * state.pxPerSecond;
        const audioWidth = Math.max(24, audioDuration * state.pxPerSecond);
        const audioWave = document.createElement("canvas");
        audioWave.width = Math.max(1, Math.floor(audioWidth));
        audioWave.height = TIMELINE_SCENE_AUDIO_HEIGHT;
        audioWave.title = "Custom scene audio waveform. Click for audio cut/delete options.";
        audioWave.style.cssText = `
          position:absolute;left:${audioLeft}px;top:${TIMELINE_SCENE_AUDIO_TOP}px;width:${audioWidth}px;height:${TIMELINE_SCENE_AUDIO_HEIGHT}px;
          z-index:1;pointer-events:auto;cursor:pointer;background:rgba(88,28,135,.36);border:1px solid rgba(216,180,254,.35);border-radius:4px;
        `;
        audioWave.onclick = (event) => openAudioContextMenu(event, segment);
        audioWave.oncontextmenu = (event) => openAudioContextMenu(event, segment);
        segmentLayer.append(audioWave);
        drawSegmentAudioWaveform(audioWave, segment.custom_audio_peaks);
      }
    }
    renderBeatMarkersOverlay();
  }

  function openSceneOptions(segment) {
    if (!segment) return;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:12px;display:flex;flex-direction:column;gap:10px;";
    const title = document.createElement("div");
    title.textContent = `${segment.label || "Scene"} options`;
    title.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
    const customAudioStatus = document.createElement("div");
    customAudioStatus.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#d4d4d8;padding:9px;font-size:12px;overflow-wrap:anywhere;";
    const updateCustomAudioStatus = () => {
      customAudioStatus.textContent = segment.custom_audio_path
        ? `Custom audio: ${segment.custom_audio_name || segment.custom_audio_path}`
        : "No custom scene audio selected.";
    };
    updateCustomAudioStatus();
    const audioDrop = document.createElement("div");
    audioDrop.textContent = "Drag audio file here";
    audioDrop.style.cssText = "border:1px dashed #38bdf8;border-radius:6px;background:#082f49;color:#e0f2fe;padding:16px;text-align:center;font-size:12px;font-weight:900;";
    const sceneAudioFileInput = document.createElement("input");
    sceneAudioFileInput.type = "file";
    sceneAudioFileInput.accept = "audio/wav,audio/mpeg,audio/flac,audio/mp4,audio/ogg,.wav,.mp3,.flac,.m4a,.ogg";
    sceneAudioFileInput.style.display = "none";
    box.append(sceneAudioFileInput);
    const pickCustomAudio = makeButton("Load Audio File");
    const clearCustomAudio = makeButton("Clear");
    const saveOptions = makeButton("Save", "primary");
    const closeOptions = makeButton("Close");
    const note = document.createElement("div");
    note.textContent = "Drop or load an audio file for this scene. It will be copied into the project folder, sent to LTX for this scene, and used for final stitching when scene-audio mode is active.";
    note.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;";
    actions.append(pickCustomAudio, clearCustomAudio, saveOptions, closeOptions);
    box.append(title, customAudioStatus, audioDrop, note, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const saveAudioFile = (file) => {
      if (!file) return;
      const projectFolder = projectInput.value || state.projectFolder;
      if (!projectFolder) {
        toast("Set a project folder first so the scene audio can be copied there.", true);
        return;
      }
      const sceneNumber = sceneSlotNumber(segment);
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const data = await postJson("/vrgdg/music_builder/save_scene_audio", {
            project_folder: projectFolder,
            scene_number: sceneNumber,
            audio_data: String(reader.result || ""),
            audio_name: file.name || "scene_audio.wav",
          }, 180000);
          pushHistory();
          segment.custom_audio_path = data.saved_path || "";
          segment.custom_audio_name = file.name || "";
          segment.custom_audio_duration = Number(data.duration || 0);
          segment.custom_audio_full_duration = Number(data.duration || 0);
          segment.custom_audio_timeline_start = Number(segment.start || 0);
          segment.custom_audio_source_start = 0;
          segment.custom_audio_peaks = Array.isArray(data.peaks) ? data.peaks : [];
          segment.custom_audio_beats = Array.isArray(data.beats) ? data.beats : [];
          if (segment.custom_audio_duration > 0 && !hasLockedVideo(segment)) {
            segment.end = Number(segment.start || 0) + segment.custom_audio_duration;
            normalizeSegments(segment);
          }
          updateCustomAudioStatus();
          render();
          toast(`Custom scene audio saved:\n${segment.custom_audio_path}`);
        } catch (error) {
          toast(String(error?.message || error), true);
        }
      };
      reader.onerror = () => toast("Failed to read the audio file.", true);
      reader.readAsDataURL(file);
    };
    pickCustomAudio.onclick = () => sceneAudioFileInput.click();
    sceneAudioFileInput.onchange = () => {
      saveAudioFile(sceneAudioFileInput.files?.[0]);
      sceneAudioFileInput.value = "";
    };
    audioDrop.addEventListener("dragover", (event) => {
      if (!audioFileFromDrop(event)) return;
      event.preventDefault();
      event.stopPropagation();
      audioDrop.style.borderColor = "#a3e635";
    });
    audioDrop.addEventListener("dragleave", () => {
      audioDrop.style.borderColor = "#38bdf8";
    });
    audioDrop.addEventListener("drop", (event) => {
      const file = audioFileFromDrop(event);
      if (!file) return;
      event.preventDefault();
      event.stopPropagation();
      audioDrop.style.borderColor = "#38bdf8";
      saveAudioFile(file);
    });
    clearCustomAudio.onclick = () => {
      pushHistory();
      segment.custom_audio_path = "";
      segment.custom_audio_name = "";
      segment.custom_audio_duration = 0;
      segment.custom_audio_full_duration = 0;
      segment.custom_audio_timeline_start = Number(segment.start || 0);
      segment.custom_audio_source_start = 0;
      segment.custom_audio_peaks = [];
      segment.custom_audio_beats = [];
      updateCustomAudioStatus();
      render();
    };
    closeOptions.onclick = () => backdrop.remove();
    saveOptions.onclick = () => {
      pushHistory();
      render();
      backdrop.remove();
      toast(segment.custom_audio_path ? `Custom scene audio saved:\n${segment.custom_audio_path}` : "Custom scene audio cleared.");
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  function clearSegmentAudio(segment) {
    segment.custom_audio_path = "";
    segment.custom_audio_name = "";
    segment.custom_audio_duration = 0;
    segment.custom_audio_full_duration = 0;
    segment.custom_audio_timeline_start = Number(segment.start || 0);
    segment.custom_audio_source_start = 0;
    segment.custom_audio_peaks = [];
    segment.custom_audio_beats = [];
  }

  function splitAudioPeaks(peaks, ratio) {
    const values = Array.isArray(peaks) ? peaks : [];
    const index = Math.max(1, Math.min(values.length - 1, Math.round(values.length * ratio)));
    return [values.slice(0, index), values.slice(index)];
  }

  function splitAudioBeats(beats, cutLocal, totalDuration) {
    const values = Array.isArray(beats) ? beats : [];
    const left = [];
    const right = [];
    for (const beat of values) {
      const value = Number(beat || 0);
      if (value < cutLocal) left.push(value);
      else if (value <= totalDuration) right.push(Math.max(0, value - cutLocal));
    }
    return [left, right];
  }

  function openAudioContextMenu(event, segment) {
    event.preventDefault();
    event.stopPropagation();
    setActiveSegment(segment);
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    if (!segment?.custom_audio_path) return;
    const menu = document.createElement("div");
    menu.className = "vrgdg-builder-context-menu";
    menu.style.cssText = "position:fixed;z-index:100010;min-width:190px;border:1px solid #7e22ce;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";
    menu.style.left = `${Math.min(window.innerWidth - 200, event.clientX)}px`;
    menu.style.top = `${Math.min(window.innerHeight - 150, event.clientY)}px`;
    const addItem = (label, action, disabled = false) => {
      const button = makeButton(label);
      button.disabled = disabled;
      button.style.justifyContent = "flex-start";
      button.style.textAlign = "left";
      button.onclick = () => {
        menu.remove();
        action();
      };
      menu.append(button);
    };
    const rect = event.currentTarget?.getBoundingClientRect?.();
    const ratio = rect ? Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))) : 0.5;
    const duration = audioChunkDuration(segment);
    const cutLocal = ratio * duration;
    addItem("Cut audio here", () => {
      if (cutLocal < 0.25 || duration - cutLocal < 0.25) {
        toast("Cut point is too close to the edge of the audio chunk.", true);
        return;
      }
      pushHistory();
      const audioStart = audioTimelineStart(segment);
      const cutTime = audioStart + cutLocal;
      const rightDuration = duration - cutLocal;
      const right = newSegment(cutTime, cutTime + rightDuration);
      right.label = `${segment.label || "Scene"} audio`;
      right.source = state.srtMode ? "inserted" : "manual";
      right.custom_audio_path = segment.custom_audio_path;
      right.custom_audio_name = segment.custom_audio_name;
      right.custom_audio_full_duration = Number(segment.custom_audio_full_duration || duration);
      right.custom_audio_timeline_start = cutTime;
      right.custom_audio_source_start = audioSourceStart(segment) + cutLocal;
      right.custom_audio_duration = rightDuration;
      const [leftPeaks, rightPeaks] = splitAudioPeaks(segment.custom_audio_peaks, ratio);
      const [leftBeats, rightBeats] = splitAudioBeats(segment.custom_audio_beats, cutLocal, duration);
      segment.custom_audio_peaks = leftPeaks;
      segment.custom_audio_beats = leftBeats;
      right.custom_audio_peaks = rightPeaks;
      right.custom_audio_beats = rightBeats;
      segment.custom_audio_duration = cutLocal;
      segment.end = Math.min(Number(segment.end || cutTime), cutTime);
      const index = state.segments.findIndex((item) => item.id === segment.id);
      state.segments.splice(index + 1, 0, right);
      state.activeId = right.id;
      render();
      syncInspector();
    });
    addItem("Delete audio chunk", () => {
      pushHistory();
      clearSegmentAudio(segment);
      render();
      syncInspector();
    });
    addItem("Scene options", () => openSceneOptions(segment));
    document.body.append(menu);
    const close = (closeEvent) => {
      if (!menu.contains(closeEvent.target)) {
        menu.remove();
        window.removeEventListener("pointerdown", close);
      }
    };
    setTimeout(() => window.addEventListener("pointerdown", close), 0);
  }

  function openSegmentContextMenu(event, segment) {
    event.preventDefault();
    event.stopPropagation();
    setActiveSegment(segment);
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    const menu = document.createElement("div");
    menu.className = "vrgdg-builder-context-menu";
    menu.style.cssText = "position:fixed;z-index:100010;min-width:180px;border:1px solid #155e75;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";
    menu.style.left = `${Math.min(window.innerWidth - 190, event.clientX)}px`;
    menu.style.top = `${Math.min(window.innerHeight - 150, event.clientY)}px`;
    const addItem = (label, action, disabled = false) => {
      const button = makeButton(label);
      button.disabled = disabled;
      button.style.justifyContent = "flex-start";
      button.style.textAlign = "left";
      button.onclick = () => {
        menu.remove();
        action();
      };
      menu.append(button);
    };
    const rect = event.currentTarget?.getBoundingClientRect?.();
    const ratio = rect ? Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))) : 0.5;
    const cutTime = Number(segment.start || 0) + ratio * Math.max(0.1, Number(segment.end || 0) - Number(segment.start || 0));
    const isOverlay = segmentTrack(segment) === "overlay";
    const locked = hasLockedVideo(segment) || (state.timingFrozen && !isOverlay);
    addItem("Cut here", () => {
      if (locked) {
        toast("Timing is frozen or this scene has video, so it cannot be cut.", true);
        return;
      }
      const durationA = cutTime - Number(segment.start || 0);
      const durationB = Number(segment.end || 0) - cutTime;
      if (durationA < 0.25 || durationB < 0.25) {
        toast("Cut point is too close to the edge of the scene.", true);
        return;
      }
      pushHistory();
      const next = newSegment(cutTime, Number(segment.end || cutTime + 4));
      next.label = `${segment.label || "Scene"} copy`;
      next.notes = segment.notes || "";
      next.track = isOverlay ? "overlay" : "base";
      next.source = isOverlay ? "overlay" : state.srtMode ? "inserted" : "manual";
      segment.end = cutTime;
      const collection = isOverlay ? state.overlaySegments : state.segments;
      const index = collection.findIndex((item) => item.id === segment.id);
      collection.splice(index + 1, 0, next);
      state.activeId = next.id;
      render();
    }, locked);
    addItem("Scene options", () => openSceneOptions(segment));
    addItem("Delete scene", deleteSegment);
    document.body.append(menu);
    const close = (closeEvent) => {
      if (!menu.contains(closeEvent.target)) {
        menu.remove();
        window.removeEventListener("pointerdown", close);
      }
    };
    setTimeout(() => window.addEventListener("pointerdown", close), 0);
  }

  function makeDragHandle(element, segment, mode) {
    element.addEventListener("pointerdown", (event) => {
      const isOverlay = segmentTrack(segment) === "overlay";
      if (state.timingFrozen && !isOverlay) {
        toast("Timing is frozen. Unfreeze timing before editing segment lengths.", true);
        return;
      }
      if (hasLockedVideo(segment)) {
        toast("This scene already has a generated video, so its timing is locked.", true);
        return;
      }
      event.stopPropagation();
      element.setPointerCapture?.(event.pointerId);
      const startX = event.clientX;
      const start = segment.start;
      const end = segment.end;
      let historySaved = false;
      const move = (moveEvent) => {
        if (!historySaved) {
          pushHistory();
          historySaved = true;
        }
        const delta = (moveEvent.clientX - startX) / state.pxPerSecond;
        if (mode === "start") {
          segment.start = Math.max(0, Math.min(end - 0.1, snapTimeToBeat(start + delta)));
        } else if (mode === "end") {
          segment.end = Math.min(state.duration || end + 100, Math.max(start + 0.1, snapTimeToBeat(end + delta)));
        } else {
          const duration = end - start;
          segment.start = Math.max(0, Math.min((state.duration || 9999) - duration, snapTimeToBeat(start + delta)));
          segment.end = segment.start + duration;
        }
        if (isOverlay) sortSegments(state.overlaySegments);
        else normalizeSegments(segment);
        syncInspector();
        render();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  }

  function renderList() {
    segmentList.textContent = "";
    ensureAllSegmentRuntimeFields();
    for (const [index, segment] of state.segments.entries()) {
      const row = document.createElement("div");
      row.role = "button";
      row.tabIndex = 0;
      const thumb = mediaThumbnailHtml(segment, 56);
      const inserted = state.srtMode && segment.source !== "srt";
      const t2iDone = Boolean(segmentImageSource(segment));
      const i2vDone = Boolean(String(segment.i2v_prompt || "").trim());
      const videoDone = Boolean(segment.video_path);
      const historyStatus = segment.image_history.length ? `<span style="border:1px solid #67e8f9;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#bae6fd;">IMG ${segment.image_history.length}</span>` : "";
      const videoHistoryStatus = segment.video_history.length ? `<span style="border:1px solid #a78bfa;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#f3e8ff;">VID ${segment.video_history.length}</span>` : "";
      const zStatus = segment.use_scene_zimage_settings ? `<span style="border:1px solid #f59e0b;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#fde68a;">Z custom</span>` : "";
      const audioStatus = segment.custom_audio_path ? `<span style="border:1px solid #a78bfa;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#ddd6fe;">AUD</span>` : "";
      const status = `
        <div style="display:flex;gap:6px;margin-top:6px;align-items:center;">
          <span style="border:1px solid ${t2iDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${t2iDone ? "#bbf7d0" : "#a1a1aa"};">T2I ${t2iDone ? "OK" : "--"}</span>
          <span style="border:1px solid ${i2vDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${i2vDone ? "#bbf7d0" : "#a1a1aa"};">I2V ${i2vDone ? "OK" : "--"}</span>
          <span style="border:1px solid ${videoDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${videoDone ? "#bbf7d0" : "#a1a1aa"};">VID ${videoDone ? "OK" : "--"}</span>
          ${historyStatus}
          ${videoHistoryStatus}
          ${zStatus}
          ${audioStatus}
        </div>
      `;
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      const isMultiSelected = isSegmentMultiSelected(segment);
      row.style.cssText = `width:100%;text-align:left;border:${isActive || isMultiSelected ? "3px" : "1px"} solid ${isActive || isMultiSelected ? "#ef4444" : inserted ? "#f59e0b" : "#3f3f46"};border-radius:7px;background:${isActive || isMultiSelected ? "#3f1d24" : inserted ? "#451a03" : "#27272a"};color:#fafafa;padding:8px;margin-bottom:8px;cursor:pointer;box-shadow:${isActive || isMultiSelected ? "0 0 0 2px rgba(239,68,68,.25), 0 0 18px rgba(239,68,68,.42)" : "none"};`;
      row.innerHTML = `<div style="font-weight:800;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${index + 1}. ${escapeHtml(segment.label || "Scene")}</div><div style="font-size:11px;color:#a1a1aa;margin-top:4px;">Duration in seconds: ${formatDurationSeconds(segment.start, segment.end)}</div><div style="font-size:11px;color:#71717a;margin-top:2px;">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>${status}${thumb}`;
      row.onclick = () => handleSegmentPick(segment);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleSegmentPick(segment);
        }
      };
      const optionsButton = document.createElement("span");
      optionsButton.textContent = "Options";
      optionsButton.title = "Scene options";
      optionsButton.style.cssText = "display:inline-flex;margin-top:6px;border:1px solid #3f3f46;border-radius:4px;padding:3px 6px;font-size:10px;font-weight:900;color:#f4f4f5;background:#18181b;";
      optionsButton.onclick = (event) => {
        event.stopPropagation();
        setActiveSegment(segment);
        openSceneOptions(segment);
      };
      row.append(optionsButton);
      const dragImageSource = segmentImageSource(segment);
      if (dragImageSource) {
        row.draggable = true;
        row.ondragstart = (event) => {
          event.dataTransfer.setData("application/x-vrgdg-segment-id", segment.id);
          event.dataTransfer.setData("text/plain", segment.label || "Scene image");
          event.dataTransfer.effectAllowed = "copy";
        };
      }
      enableImageDrop(row, segment);
      segmentList.append(row);
    }
    if (state.overlaySegments.length) {
      const header = document.createElement("div");
      header.textContent = "Insert timeline";
      header.style.cssText = "margin:10px 0 8px;color:#fdba74;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.08em;";
      segmentList.append(header);
    }
    for (const [index, segment] of state.overlaySegments.entries()) {
      const row = document.createElement("div");
      row.role = "button";
      row.tabIndex = 0;
      const thumb = mediaThumbnailHtml(segment, 50);
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      const isMultiSelected = isSegmentMultiSelected(segment);
      row.style.cssText = `width:100%;text-align:left;border:${isActive || isMultiSelected ? "3px" : "1px"} solid ${isActive || isMultiSelected ? "#ef4444" : "#f97316"};border-radius:7px;background:${isActive || isMultiSelected ? "#3f1d24" : "#431407"};color:#fafafa;padding:8px;margin-bottom:8px;cursor:pointer;box-shadow:${isActive || isMultiSelected ? "0 0 0 2px rgba(239,68,68,.25), 0 0 18px rgba(239,68,68,.42)" : "none"};`;
      row.innerHTML = `<div style="font-weight:800;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Insert ${index + 1}. ${escapeHtml(segment.label || "Insert")}</div><div style="font-size:11px;color:#fed7aa;margin-top:4px;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</div>${thumb}`;
      row.onclick = () => handleSegmentPick(segment);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleSegmentPick(segment);
        }
      };
      enableImageDrop(row, segment);
      segmentList.append(row);
    }
  }

  function imageFileFromDrop(event) {
    const files = Array.from(event.dataTransfer?.files || []);
    return files.find((file) => /^image\//i.test(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name || ""));
  }

  function audioFileFromDrop(event) {
    const files = Array.from(event.dataTransfer?.files || []);
    return files.find((file) => /^audio\//i.test(file.type) || /\.(wav|mp3|flac|m4a|ogg)$/i.test(file.name || ""));
  }

  function loadCustomImageFile(file, segment = activeSegment()) {
    if (!segment || !file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      pushHistory();
      const imageData = String(reader.result || "");
      const projectFolder = projectInput.value || state.projectFolder;
      if (projectFolder) {
        try {
          const sceneNumber = sceneSlotNumber(segment);
          const saved = await postJson("/vrgdg/music_builder/archive_scene_image", {
            image_data: imageData,
            project_folder: projectFolder,
            scene_number: sceneNumber,
          });
          if (saved.saved_path) {
            addSceneImageHistoryPath(segment, saved.saved_path);
            segment.custom_image_path = saved.saved_path;
            segment.custom_image_data = "";
            segment.custom_image_name = file.name || "custom_image";
            segment.image = null;
          }
        } catch (error) {
          console.warn("[VRGDG Music Builder] Failed to archive custom image:", error);
          segment.custom_image_data = imageData;
          segment.custom_image_name = file.name || "custom_image";
          segment.custom_image_path = "";
          segment.image = null;
        }
      } else {
        segment.custom_image_data = imageData;
        segment.custom_image_name = file.name || "custom_image";
        segment.custom_image_path = "";
        segment.image = null;
      }
      segment.approved_image_path = "";
      segment.preview_mode = "image";
      setActiveSegment(segment);
      syncPreview(segment);
      render();
      toast(`Loaded custom image for ${segment.label || "scene"}:\n${file.name}`);
      autoSaveSessionQuiet("custom image load");
    };
    reader.onerror = () => toast("Failed to read the dropped image.", true);
    reader.readAsDataURL(file);
  }

  function setImageToImageSource({ path = "", data = "", name = "" } = {}) {
    const useErnie = (state.imageModelMode || "") === "ernie_image";
    const settings = useErnie ? (state.ernieImageSettings || defaultErnieImageSettings()) : activeZImageSettings();
    pushHistory();
    settings.use_image_to_image = true;
    settings.image_to_image_path = path || "";
    settings.image_to_image_data = data || "";
    settings.image_to_image_name = name || "";
    settings.image_to_image_start_at_step = Math.max(1, Math.min(8, Number(
      useErnie ? (ernieI2IStartStep.value || ernieI2ISlider.value || settings.image_to_image_start_at_step || 5) : (zI2IStartStep.value || zI2ISlider.value || settings.image_to_image_start_at_step || 5)
    )));
    if (useErnie) {
      state.ernieImageSettings = settings;
      syncErnieImagePanel();
    } else {
      syncZImageSettingsPanel();
    }
    renderList();
    toast(`Image-to-image source set${path || name ? `:\n${path || name}` : "."}`);
  }

  function segmentImageSource(segment) {
    ensureSegmentRuntimeFields(segment);
    const historyPath = segment.image_history?.[segment.image_history_index] || segment.image_history?.[segment.image_history.length - 1] || "";
    if (historyPath) {
      return { path: historyPath };
    }
    if (segment.custom_image_path) {
      return { path: segment.custom_image_path };
    }
    if (segment.custom_image_data) {
      return { data: segment.custom_image_data, name: segment.custom_image_name || "custom_image.png" };
    }
    if (segment.approved_image_path) {
      return { path: segment.approved_image_path };
    }
    return null;
  }

  function loadImageToImageFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageToImageSource({ data: String(reader.result || ""), name: file.name || "image.png" });
    reader.onerror = () => toast("Failed to read the image-to-image source.", true);
    reader.readAsDataURL(file);
  }

  async function setVisionReferenceSource({ path = "", data = "", name = "", forT2V = false } = {}) {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    if (path) {
      segment.ref_image_path = path;
    } else if (data) {
      const sceneNumber = sceneSlotNumber(segment);
      const saved = await postJson("/vrgdg/music_builder/archive_scene_image", {
        image_data: data,
        project_folder: projectInput.value || state.projectFolder,
        scene_number: sceneNumber,
      });
      segment.ref_image_path = saved.saved_path || "";
    }
    if (forT2V) {
      segment.use_t2v_vision_reference = true;
      useT2VVisionReference.input.checked = true;
    } else {
      segment.use_vision_reference = true;
      useVisionReference.input.checked = true;
      ernieUseVisionReference.input.checked = true;
    }
    refImageInput.value = segment.ref_image_path || name || "";
    refImagePanel.style.display = useVisionReference.input.checked ? "flex" : "none";
    ernieRefImagePanel.style.display = ernieUseVisionReference.input.checked ? "flex" : "none";
    t2vRefImagePanel.style.display = currentVideoMode() === "t2v" && useT2VVisionReference.input.checked ? "flex" : "none";
    renderList();
    toast(`Vision reference set${segment.ref_image_path ? `:\n${segment.ref_image_path}` : "."}`);
  }

  function loadVisionReferenceFile(file, options = {}) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setVisionReferenceSource({ data: String(reader.result || ""), name: file.name || "reference.png", forT2V: Boolean(options.forT2V) }).catch((error) => toast(String(error?.message || error), true));
    };
    reader.onerror = () => toast("Failed to read the vision reference image.", true);
    reader.readAsDataURL(file);
  }

  function sceneIdFromDrop(event) {
    return event.dataTransfer?.getData("application/x-vrgdg-segment-id") || "";
  }

  function droppedSceneImageSource(event) {
    const id = sceneIdFromDrop(event);
    if (!id) return null;
    const segment = state.segments.find((item) => item.id === id);
    return segment ? segmentImageSource(segment) : null;
  }

  function addFluxIngredient({ path = "", data = "", name = "", global = false } = {}) {
    pushHistory();
    const candidate = {
      path: path || "",
      data: data || "",
      name: name || path?.split?.(/[\\/]/)?.pop?.() || "image.png",
    };
    const ingredientKey = (item) => String(item?.path || item?.data || item?.name || "").trim();
    const candidateKey = ingredientKey(candidate);
    if (global) {
      if (!Array.isArray(state.fluxGlobalImageIngredients)) state.fluxGlobalImageIngredients = [];
      if (candidateKey && state.fluxGlobalImageIngredients.some((item) => ingredientKey(item) === candidateKey)) {
        renderFluxGlobalIngredientList();
        toast("That global image ingredient is already loaded.");
        return;
      }
      state.fluxGlobalImageIngredients.push(candidate);
      renderFluxGlobalIngredientList();
      render();
      toast(`Global image ingredient added${path || name ? `:\n${path || name}` : "."}`);
      return;
    }
    const segment = activeSegment();
    if (!segment) {
      toast("Add or select a scene first.", true);
      return;
    }
    if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
    if (candidateKey && segment.flux_image_ingredients.some((item) => ingredientKey(item) === candidateKey)) {
      renderFluxIngredientList(segment);
      renderNBIngredientList(segment);
      toast("That image ingredient is already loaded for this scene.");
      return;
    }
    segment.flux_image_ingredients.push(candidate);
    const settings = state.fluxKleinSettings || {};
    settings.enabled = true;
    state.fluxKleinSettings = settings;
    syncFluxKleinPanel();
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    render();
    toast(`Image ingredient added${path || name ? `:\n${path || name}` : "."}`);
  }

  function loadFluxIngredientFile(file, { global = false } = {}) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => addFluxIngredient({ data: String(reader.result || ""), name: file.name || "image.png", global });
    reader.onerror = () => toast("Failed to read the image ingredient.", true);
    reader.readAsDataURL(file);
  }

  function enableFluxIngredientDrop(element, { global = false } = {}) {
    element.dataset.vrgdgFileDropZone = "true";
    element.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []).map((item) => String(item).toLowerCase());
      if (!types.includes("files") && !types.includes("application/x-vrgdg-segment-id")) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.borderColor = "#a3e635";
    });
    element.addEventListener("dragleave", () => {
      element.style.borderColor = "#155e75";
    });
    element.addEventListener("drop", (event) => {
      const sceneSource = droppedSceneImageSource(event);
      const files = Array.from(event.dataTransfer?.files || []).filter((file) => file.type?.startsWith?.("image/"));
      if (!sceneSource && !files.length) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.borderColor = "#155e75";
      if (sceneSource) {
        addFluxIngredient({
          path: sceneSource.path || "",
          data: sceneSource.data || "",
          name: sceneSource.name || "scene_image.png",
          global,
        });
        return;
      }
      for (const file of files) loadFluxIngredientFile(file, { global });
    });
  }

  function enableImageDrop(element, segment) {
    element.addEventListener("dragover", (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.outline = "2px solid #a3e635";
    });
    element.addEventListener("dragleave", (event) => {
      event.stopPropagation();
      element.style.outline = "";
    });
    element.addEventListener("drop", (event) => {
      const file = imageFileFromDrop(event);
      if (!file) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.outline = "";
      loadCustomImageFile(file, segment);
    });
  }

  function projectContextPath(filename) {
    const folder = String(projectInput.value || state.projectFolder || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}project_context${separator}${filename}`;
  }

  async function loadContextTextQuiet(path) {
    const filePath = String(path || "").trim();
    if (!filePath) return "";
    try {
      const data = await postJson("/vrgdg/music_builder/load_text_file", { path: filePath });
      return String(data.content || "").trim();
    } catch {
      return "";
    }
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
        progress = createProgressWindow(title.replace(/^Edit\s+/i, "Gemma4 "));
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
          n_ctx: 13000,
          max_new_tokens: 8000,
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

  function projectPromptBackupPath(kind, name) {
    const folder = String(projectInput.value || state.projectFolder || "").trim().replace(/[\\/]+$/, "");
    if (!folder) return "";
    const separator = folder.includes("\\") ? "\\" : "/";
    return `${folder}${separator}prompts${separator}backups${separator}${kind}_prompts_${name}.txt`;
  }

  function promptKindLabel(kind) {
    return kind === "i2v" ? "image-to-video" : "text-to-image";
  }

  function promptKindFile(kind) {
    return projectPromptsPath(kind === "i2v" ? "i2v_prompts.txt" : "t2i_prompts.txt");
  }

  function promptKindPrefix(kind) {
    return kind === "i2v" ? "I2V" : "Prompt";
  }

  function formatPromptBlocks(kind, prompts = null) {
    const values = Array.isArray(prompts)
      ? prompts
      : allEditableSegments().map((segment) => kind === "i2v" ? segment.i2v_prompt : (segment.t2i_prompt || segment.flux_prompt));
    return values.map((item) => String(item || "").trim()).join("\n\n").replace(/\s+$/g, "") + "\n";
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

  function applyPromptBlocksToSegments(kind, prompts) {
    const values = Array.isArray(prompts) ? prompts : [];
    if (!values.length) throw new Error(`No ${promptKindLabel(kind)} prompts were found in that text.`);
    const segments = allEditableSegments();
    pushHistory();
    for (let index = 0; index < segments.length && index < values.length; index += 1) {
      const value = String(values[index] || "").trim();
      if (kind === "i2v") {
        segments[index].i2v_prompt = value;
      } else {
        segments[index].t2i_prompt = value;
        segments[index].flux_prompt = value;
      }
    }
    syncInspector();
    render();
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

  async function savePromptTextFile(path, content) {
    const filePath = String(path || "").trim();
    if (!filePath) throw new Error("Create or load a project first.");
    return await postJson("/vrgdg/music_builder/save_text_file", { path: filePath, content: String(content || "") });
  }

  function promptTimestamp() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
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
      effect.textContent = kind === "i2v"
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
      heading.textContent = kind === "i2v" ? "Edit Image-to-Video Prompts" : "Edit Text-to-Image Prompts";
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
          await savePromptTextFile(path, textarea.value);
          const prompts = parsePromptTextBlocks(textarea.value, kind);
          applyPromptBlocksToSegments(kind, prompts);
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
      await saveSession({ quiet: true, throwOnError: true });
      toast(`Reloaded ${prompts.length} ${promptKindLabel(kind)} prompt${prompts.length === 1 ? "" : "s"} into the scene boxes.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
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

  function openFluxReferenceBuilderModal() {
    const referenceBuilderTargetLabel = state.imageModelMode === "nano_banana" ? "Nano B" : "Flux/Klein";
    state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const refs = state.fluxReferenceBuilder;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1180px,calc(100vw - 42px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Reference Image Builder</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Use a global subject reference plus per-scene location references as automatic ${escapeHtml(referenceBuilderTargetLabel)} reference images.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);

    const usage = document.createElement("div");
    usage.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;";
    const useSubject = makeCheckbox("Use subject reference", refs.use_subject_reference);
    const useLocations = makeCheckbox("Use mapped location references", refs.use_location_references);
    const includeManual = makeCheckbox("Also include manually loaded reference images", refs.include_manual_ingredients !== false);
    for (const item of [useSubject.wrapper, useLocations.wrapper, includeManual.wrapper]) {
      item.style.cssText += "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;";
    }
    usage.append(useSubject.wrapper, useLocations.wrapper, includeManual.wrapper);
    const inlineProgress = document.createElement("div");
    inlineProgress.style.cssText = "display:none;border:1px solid #155e75;border-radius:7px;background:#07111f;padding:9px 10px;gap:7px;flex-direction:column;";
    const inlineProgressText = document.createElement("div");
    inlineProgressText.style.cssText = "font-size:12px;color:#e0f2fe;white-space:pre-wrap;line-height:1.35;";
    const inlineProgressTrack = document.createElement("div");
    inlineProgressTrack.style.cssText = "height:8px;border-radius:999px;background:#164e63;overflow:hidden;";
    const inlineProgressBar = document.createElement("div");
    inlineProgressBar.style.cssText = "height:100%;width:0%;background:#22d3ee;border-radius:999px;transition:width .18s ease;";
    inlineProgressTrack.append(inlineProgressBar);
    inlineProgress.append(inlineProgressText, inlineProgressTrack);
    const setInlineProgress = (message, percent = 8) => {
      inlineProgress.style.display = "flex";
      inlineProgressText.textContent = message || "Working...";
      inlineProgressBar.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
    };
    const hideInlineProgress = (delay = 1200) => {
      setTimeout(() => {
        inlineProgress.style.display = "none";
        inlineProgressBar.style.width = "0%";
      }, delay);
    };

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:minmax(280px,.9fr) minmax(420px,1.35fr) minmax(300px,1fr);gap:12px;align-items:start;";
    const cardStyle = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";

    const modalDragGuard = (event) => {
      const path = typeof event.composedPath === "function" ? event.composedPath() : [];
      const inDropZone = path.some((item) => item?.dataset?.vrgdgFileDropZone === "true")
        || event.target?.closest?.("[data-vrgdg-file-drop-zone='true']");
      if (!inDropZone) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    };
    for (const eventName of ["dragenter", "dragover", "drop"]) {
      backdrop.addEventListener(eventName, modalDragGuard, true);
      box.addEventListener(eventName, modalDragGuard, true);
    }

    const subjectCard = document.createElement("div");
    subjectCard.style.cssText = cardStyle;
    const subjectTitle = document.createElement("div");
    subjectTitle.textContent = "Subject Reference";
    subjectTitle.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
    const subjectDescription = document.createElement("textarea");
    subjectDescription.value = refs.subject.description || String(subjectSceneInput.value || "").split(/\n+/).find((line) => line.trim()) || "";
    subjectDescription.placeholder = "Subject/character description...";
    subjectDescription.style.cssText = "min-height:92px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:9px;font-size:12px;";
    const subjectDrop = document.createElement("div");
    subjectDrop.style.cssText = "min-height:118px;border:1px dashed #0891b2;border-radius:7px;background:#061620;color:#cffafe;display:flex;align-items:center;justify-content:center;text-align:center;padding:10px;overflow:hidden;";
    const subjectButtons = document.createElement("div");
    subjectButtons.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const createSubjectZImage = makeButton("Create Subject with ZImage", "primary");
    const uploadSubject = makeButton("Upload Subject Image", "primary");
    const clearSubject = makeButton("Clear Subject");
    subjectButtons.append(createSubjectZImage, uploadSubject, clearSubject);
    subjectCard.append(subjectTitle, makeField("Subject description", subjectDescription), subjectDrop, subjectButtons);

    const locationsCard = document.createElement("div");
    locationsCard.style.cssText = cardStyle;
    const locationsHeader = document.createElement("div");
    locationsHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
    const locationsTitle = document.createElement("div");
    locationsTitle.textContent = "Location References";
    locationsTitle.style.cssText = subjectTitle.style.cssText;
    const extractLocations = makeButton("Extract Locations", "primary");
    const autoMapLocations = makeButton("Auto Map Locations with Gemma", "primary");
    const addLocation = makeButton("Add Location", "primary");
    const locationActions = document.createElement("div");
    locationActions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;";
    locationActions.append(extractLocations, autoMapLocations, addLocation);
    locationsHeader.append(locationsTitle, locationActions);
    const locationsList = document.createElement("div");
    locationsList.style.cssText = "display:flex;flex-direction:column;gap:10px;max-height:560px;overflow:auto;padding-right:4px;";
    locationsCard.append(locationsHeader, locationsList);

    const mappingCard = document.createElement("div");
    mappingCard.style.cssText = cardStyle;
    const mappingTitle = document.createElement("div");
    mappingTitle.textContent = "Scene Mapping";
    mappingTitle.style.cssText = subjectTitle.style.cssText;
    const mappingNote = document.createElement("div");
    mappingNote.textContent = `Choose which location image ${referenceBuilderTargetLabel} should receive for each scene. Use Unassigned for no location reference.`;
    mappingNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const mappingList = document.createElement("div");
    mappingList.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:560px;overflow:auto;padding-right:4px;";
    mappingCard.append(mappingTitle, mappingNote, mappingList);

    grid.append(subjectCard, locationsCard, mappingCard);
    const footer = document.createElement("div");
    footer.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Save Reference Builder", "primary");
    footer.append(cancel, save);
    box.append(header, usage, inlineProgress, grid, footer);
    backdrop.append(box);
    document.body.append(backdrop);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/png,image/jpeg,image/webp";
    fileInput.style.display = "none";
    box.append(fileInput);
    let pendingImageTarget = null;

    const imageLabel = (image) => String(image?.name || image?.path || (image?.data ? "Custom image loaded" : ""));
    const imageSrc = (image) => image?.data || (image?.path ? makeEditorImageUrl(image.path) : "");
    const renderDrop = (drop, image, emptyText) => {
      const src = imageSrc(image);
      drop.style.flexDirection = "column";
      drop.style.gap = "6px";
      drop.innerHTML = src
        ? `<img src="${src}" draggable="false" style="max-width:100%;max-height:150px;object-fit:contain;border-radius:6px;"><div style="font-size:11px;color:#a5f3fc;overflow-wrap:anywhere;word-break:break-word;max-width:100%;">${escapeHtml(imageLabel(image))}</div>`
        : `<div><strong>${emptyText}</strong><br><span style="color:#94a3b8;font-size:12px;">Drop an image here or upload one.</span></div>`;
    };
    const updateSubjectDrop = () => renderDrop(subjectDrop, refs.subject.image, "Drop subject reference image here");
    const setImageTargetFromSource = (target, source = {}) => {
      if (!target) return;
      if (!target.image) target.image = { path: "", data: "", name: "" };
      target.image.path = source.path || "";
      target.image.data = source.data || "";
      target.image.name = source.name || source.path?.split?.(/[\\/]/)?.pop?.() || "reference.png";
      renderAll();
    };
    const setImageTarget = (target, file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        setImageTargetFromSource(target, {
          path: "",
          data: String(reader.result || ""),
          name: file.name || "reference.png",
        });
      };
      reader.onerror = () => toast("Failed to read the reference image.", true);
      reader.readAsDataURL(file);
    };
    const droppedImageFile = (event) => {
      const files = Array.from(event.dataTransfer?.files || []);
      const fromFiles = files.find((item) => /^image\//i.test(item.type) || /\.(png|jpe?g|webp)$/i.test(item.name || ""));
      if (fromFiles) return fromFiles;
      const items = Array.from(event.dataTransfer?.items || []);
      for (const item of items) {
        if (item.kind !== "file") continue;
        const file = item.getAsFile?.();
        if (file && (/^image\//i.test(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name || ""))) return file;
      }
      return null;
    };
    const droppedImageText = (event) => {
      return String(
        event.dataTransfer?.getData("text/uri-list")
        || event.dataTransfer?.getData("URL")
        || event.dataTransfer?.getData("text/plain")
        || ""
      ).split(/\r?\n/).map((line) => line.trim()).find((line) => line && !line.startsWith("#")) || "";
    };
    const setImageTargetFromDroppedUrl = async (target, urlText) => {
      const url = String(urlText || "").trim();
      if (!url) return false;
      if (/^data:image\//i.test(url)) {
        setImageTargetFromSource(target, { path: "", data: url, name: "reference.png" });
        return true;
      }
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        if (!/^image\//i.test(blob.type)) return false;
        const cleanName = url.split(/[?#]/)[0].split("/").pop() || "reference.png";
        setImageTarget(target, new File([blob], cleanName, { type: blob.type || "image/png" }));
        return true;
      } catch (error) {
        console.warn("[VRGDG Music Builder] Could not read dropped reference image URL:", error);
        return false;
      }
    };
    const wireDrop = (drop, target) => {
      drop.dataset.vrgdgFileDropZone = "true";
      for (const eventName of ["dragenter", "dragover", "dragleave"]) {
        drop.addEventListener(eventName, (event) => {
          event.preventDefault();
          event.stopPropagation();
          event.stopImmediatePropagation?.();
          if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        }, true);
      }
      drop.addEventListener("dragover", (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        drop.style.borderColor = "#22d3ee";
      });
      drop.addEventListener("dragleave", (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        drop.style.borderColor = "#0891b2";
      });
      drop.addEventListener("drop", (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        drop.style.borderColor = "#0891b2";
        const sceneSource = droppedSceneImageSource(event);
        if (sceneSource) {
          setImageTargetFromSource(target, sceneSource);
          return;
        }
        const file = droppedImageFile(event);
        if (file) {
          setImageTarget(target, file);
          return;
        }
        const urlText = droppedImageText(event);
        if (urlText) {
          setImageTargetFromDroppedUrl(target, urlText).then((ok) => {
            if (!ok) toast("That drop did not contain a readable image. Use Upload Image or drop a PNG/JPG/WebP file.", true);
          });
          return;
        }
        toast("That drop did not contain a readable image. Use Upload Image or drop a PNG/JPG/WebP file.", true);
      });
    };
    fileInput.onchange = () => {
      setImageTarget(pendingImageTarget, fileInput.files?.[0]);
      fileInput.value = "";
      pendingImageTarget = null;
    };
    const uploadFor = (target) => {
      pendingImageTarget = target;
      fileInput.click();
    };
    const locationKey = (name) => String(name || "").trim().toLowerCase().replace(/\s+/g, " ");
    const locationByName = (name) => refs.locations.find((item) => locationKey(item.name) === locationKey(name));
    const createLocation = (name = "", description = "") => {
      const location = {
        id: `loc_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
        name: String(name || `Location ${refs.locations.length + 1}`).trim(),
        description: String(description || "").trim(),
        image: { path: "", data: "", name: "" },
      };
      refs.locations.push(location);
      return location;
    };

    const currentZImageReferenceSettings = () => {
      const settings = cloneZImageSettings(saveZImageSettingsFromPanel());
      settings.use_image_to_image = false;
      settings.image_to_image_path = "";
      settings.image_to_image_data = "";
      settings.image_to_image_name = "";
      return settings;
    };

    const zimageReferencePayload = (prompt, settings) => {
      const zSettings = cloneZImageSettings(settings);
      const useLoras = Boolean(zSettings.use_loras && zSettings.lora_count > 0);
      const trigger = zSettings.image_trigger_phrase || state.imageTriggerPhrase || "";
      const payload = {
        prompt: applyTriggerPhrase(prompt, trigger, { validateJunk: true }),
        unet_name: zSettings.unet_name || "",
        clip_name: zSettings.clip_name || "",
        vae_name: zSettings.vae_name || "",
        first_pass_width: zSettings.first_pass_width,
        first_pass_height: zSettings.first_pass_height,
        second_pass_width: zSettings.second_pass_width,
        second_pass_height: zSettings.second_pass_height,
        seed: zSettings.seed,
        seed_mode: zSettings.seed_mode || "fixed",
        batch_size: zSettings.batch_size || 1,
        use_custom_loras: useLoras,
        lora_count: useLoras ? zSettings.lora_count : 0,
        ltx_two_pass_mode: false,
        use_image_to_image: false,
        image_to_image_start_at_step: zSettings.image_to_image_start_at_step || 5,
        image_to_image_path: "",
        image_to_image_data: "",
        image_to_image_name: "",
      };
      for (let index = 0; index < 4; index += 1) {
        const lora = zSettings.loras?.[index] || {};
        payload[`lora_${index + 1}`] = useLoras && index < zSettings.lora_count ? (lora.name || "[none]") : "[none]";
        payload[`first_pass_strength_${index + 1}`] = Number(lora.first_pass_strength ?? lora.strength ?? 0.5);
        payload[`second_pass_strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
        payload[`strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
      }
      return payload;
    };

    async function createFluxReferenceWithZImage(referenceType, target, sourceText, name = "") {
      const text = String(sourceText || "").trim();
      if (!text) {
        toast(referenceType === "subject" ? "Enter a subject description first." : "Enter a location description first.", true);
        return;
      }
      const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
      if (!modelFile && state.textGemmaRunner !== "lm_studio") {
        toast("Choose a non-vision Gemma model first.", true);
        return;
      }
      let progress = null;
      let ranZImage = false;
      try {
        setInlineProgress(referenceType === "subject" ? "Creating subject prompt with Gemma..." : "Creating location prompt with Gemma...", 8);
        progress = createProgressWindow(referenceType === "subject" ? "Creating subject reference" : "Creating location reference", { zIndex: 100008 });
        progress.set(`Creating ZImage prompt with Gemma...\n${gemmaRunnerLine()}`, 8);
        const styleTheme = state.useVrgdgTextContext ? await loadContextTextQuiet(themeStyleInput.value) : "";
        const promptData = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          reference_type: referenceType,
          source_text: text,
          style_theme: styleTheme,
          unload_after: true,
        }, 3 * 60 * 1000);
        const zSettings = currentZImageReferenceSettings();
        setInlineProgress("Building ZImage reference workflow...", 28);
        progress.set("Building ZImage reference workflow...", 28);
        const built = await postJson("/vrgdg/workflow_runner/build_zimage_prompt", zimageReferencePayload(promptData.prompt, zSettings));
        if (Number.isFinite(Number(built.used_seed))) {
          zSettings.seed = Number(built.used_seed);
          state.zimageSettings = zSettings;
          zSeed.value = String(zSettings.seed);
        }
        setInlineProgress("Queueing ZImage reference workflow...", 42);
        progress.set("Queueing ZImage reference workflow...", 42);
        const queued = await queueWorkflowPrompt(built.prompt);
        const promptId = queued?.prompt_id;
        if (!promptId) throw new Error("ComfyUI queued the ZImage reference but did not return a prompt_id.");
        ranZImage = true;
        const images = await waitForImages(promptId, (message) => {
          setInlineProgress(`${message}\nPrompt ID: ${promptId}`, 66);
          progress?.set(`${message}\nPrompt ID: ${promptId}`, 66);
        });
        const image = images[images.length - 1];
        if (!image) throw new Error("ZImage did not return a reference image.");
        setInlineProgress("Saving reference image into the project...", 88);
        progress.set("Saving reference image into the project...", 88);
        const saved = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
          project_folder: projectInput.value || state.projectFolder,
          reference_type: referenceType,
          name: name || text.slice(0, 48) || referenceType,
          image,
        });
        target.image.path = saved.saved_path || "";
        target.image.data = "";
        target.image.name = `${name || referenceType}.png`;
        advanceZImageSeedAfterRun(zSettings);
        syncZImageSettingsPanel();
        renderAll();
        setInlineProgress("Cleaning memory after ZImage reference...", 94);
        await runImageMemoryCleanupQuiet(progress, "ZImage reference", 94);
        setInlineProgress("Reference image ready.", 100);
        progress.set("Reference image ready.", 100);
        progress.close(1300);
        hideInlineProgress();
        toast(referenceType === "subject" ? "Subject reference created with ZImage." : "Location reference created with ZImage.");
      } catch (error) {
        if (ranZImage) {
          setInlineProgress("Cleaning memory after failed ZImage reference...", 100);
          await runImageMemoryCleanupQuiet(progress, "failed ZImage reference", 100);
        }
        setInlineProgress(`Error:\n${String(error?.message || error)}`, 100);
        progress?.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      }
    }

    async function extractLocationsWithGemma() {
      let progress = null;
      extractLocations.disabled = true;
      autoMapLocations.disabled = true;
      extractLocations.textContent = "Extracting...";
      progress = createProgressWindow("Extracting locations", { zIndex: 100008 });
      progress.set("Checking scene concept prompts and location context...", 5);
      const scenes = allEditableSegments().map((segment, index) => ({
        id: segment.id,
        label: segment.label || `Scene ${index + 1}`,
        concept: sceneConceptPromptText(segment),
        notes: segment.notes || "",
      }));
      const usableScenes = scenes.filter((scene) => String(scene.concept || scene.notes || "").trim());
      if (!usableScenes.length) {
        const message = "Extract Locations needs scene concept prompts or notes first.";
        progress.set(`Error:\n${message}`, 100);
        toast(message, true);
        extractLocations.disabled = false;
        autoMapLocations.disabled = false;
        extractLocations.textContent = "Extract Locations";
        return;
      }
      const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
      if (!modelFile && state.textGemmaRunner !== "lm_studio") {
        const message = "Choose a non-vision Gemma model first, or use LM Studio in Gemma Runner.";
        progress.set(`Error:\n${message}`, 100);
        toast(message, true);
        extractLocations.disabled = false;
        autoMapLocations.disabled = false;
        extractLocations.textContent = "Extract Locations";
        return;
      }
      try {
        progress.set(`Asking Gemma for a reusable location list...\n${gemmaRunnerLine()}`, 15);
        const data = await postJson("/vrgdg/music_builder/flux_reference_extract_locations", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          scenes: usableScenes,
          subject_scene_text: subjectSceneInput.value || "",
          existing_locations: refs.locations.map((item) => ({ name: item.name || "", description: item.description || "" })),
          unload_after: true,
        }, 10 * 60 * 1000);
        progress.set("Adding extracted locations to the Reference Builder...", 78);
        let added = 0;
        let updated = 0;
        for (const item of data.locations || []) {
          const name = String(item.name || "").trim();
          if (!name) continue;
          let location = locationByName(name);
          if (!location) {
            location = createLocation(name, item.description || "");
            added += 1;
          } else if (!String(location.description || "").trim() && item.description) {
            location.description = String(item.description || "");
            updated += 1;
          }
        }
        renderAll();
        progress.set(`Extracted locations ready.\nAdded: ${added}\nUpdated: ${updated}\n\nReview/edit the locations, add or create images, then run Auto Map.`, 100);
        progress.close(2600);
        toast(`Extracted ${added} new location${added === 1 ? "" : "s"}. Review them before Auto Map.`);
      } catch (error) {
        progress?.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      } finally {
        extractLocations.disabled = false;
        autoMapLocations.disabled = false;
        extractLocations.textContent = "Extract Locations";
      }
    }

    async function autoMapLocationsWithGemma() {
      let progress = null;
      autoMapLocations.disabled = true;
      autoMapLocations.textContent = "Mapping...";
      progress = createProgressWindow("Auto mapping locations", { zIndex: 100008 });
      progress.set("Checking scene concept prompts, location list, and Gemma settings...\nNo reference images are required for Auto Map.", 5);
      const scenes = allEditableSegments().map((segment, index) => ({
        id: segment.id,
        label: segment.label || `Scene ${index + 1}`,
        concept: sceneConceptPromptText(segment),
        notes: segment.notes || "",
      }));
      const usableScenes = scenes.filter((scene) => String(scene.concept || scene.notes || "").trim());
      if (!usableScenes.length) {
        const message = "Auto Map needs scene concept prompts or notes first. It does not need images yet, but it does need scene text to choose locations.";
        progress.set(`Error:\n${message}`, 100);
        toast(message, true);
        autoMapLocations.disabled = false;
        autoMapLocations.textContent = "Auto Map Locations with Gemma";
        return;
      }
      if (!refs.locations.length) {
        const message = "Auto Map needs at least one location first. Click Extract Locations or Add Location, then run Auto Map.";
        progress.set(`Error:\n${message}`, 100);
        toast(message, true);
        autoMapLocations.disabled = false;
        autoMapLocations.textContent = "Auto Map Locations with Gemma";
        return;
      }
      const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
      if (!modelFile && state.textGemmaRunner !== "lm_studio") {
        const message = "Choose a non-vision Gemma model first, or use LM Studio in Gemma Runner.";
        progress.set(`Error:\n${message}`, 100);
        toast(message, true);
        autoMapLocations.disabled = false;
        autoMapLocations.textContent = "Auto Map Locations with Gemma";
        return;
      }
      try {
        progress.set(`Sending scene concepts to Gemma...\n${gemmaRunnerLine()}`, 10);
        const data = await postJson("/vrgdg/music_builder/flux_reference_location_map", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          scenes: usableScenes,
          subject_scene_text: subjectSceneInput.value || "",
          existing_locations: refs.locations.map((item) => ({ name: item.name || "", description: item.description || "" })),
          unload_after: true,
        }, 10 * 60 * 1000);
        progress.set("Applying location map...", 80);
        const nameToId = new Map();
        for (const item of data.locations || []) {
          const name = String(item.name || "").trim();
          if (!name) continue;
          let location = locationByName(name);
          if (!location) location = createLocation(name, item.description || "");
          else if (!String(location.description || "").trim() && item.description) location.description = String(item.description || "");
          nameToId.set(locationKey(location.name), location.id);
        }
        for (const [sceneId, locationName] of Object.entries(data.scene_map || {})) {
          const id = nameToId.get(locationKey(locationName)) || locationByName(locationName)?.id || "";
          if (id) refs.scene_map[sceneId] = id;
        }
        refs.use_location_references = true;
        useLocations.input.checked = true;
        renderAll();
        progress.set("Location mapping ready. Review/edit the mappings before saving.", 100);
        progress.close(1600);
        toast(`Auto mapped ${(data.locations || []).length} location${(data.locations || []).length === 1 ? "" : "s"}. Review, add images, then save.`);
      } catch (error) {
        progress?.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      } finally {
        autoMapLocations.disabled = false;
        autoMapLocations.textContent = "Auto Map Locations with Gemma";
      }
    }

    function renderLocations() {
      locationsList.innerHTML = "";
      if (!refs.locations.length) {
        const empty = document.createElement("div");
        empty.textContent = "No locations yet. Add locations/scenes";
        empty.style.cssText = "font-size:12px;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:10px;";
        locationsList.append(empty);
        return;
      }
      refs.locations.forEach((location, index) => {
        const row = document.createElement("div");
        row.style.cssText = "border:1px solid #334155;border-radius:7px;background:#111827;padding:10px;display:flex;flex-direction:column;gap:8px;";
        const name = makeInput(location.name || `Location ${index + 1}`);
        const description = document.createElement("textarea");
        description.value = location.description || "";
        description.placeholder = "Location description / generation prompt...";
        description.style.cssText = "min-height:64px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;";
        const usedBy = allEditableSegments()
          .map((segment, sceneIndex) => refs.scene_map?.[segment.id] === location.id ? `Scene ${sceneIndex + 1}` : "")
          .filter(Boolean)
          .join(", ") || "Not mapped yet";
        const used = document.createElement("div");
        used.textContent = `Used by: ${usedBy}`;
        used.style.cssText = "font-size:11px;color:#a5f3fc;";
        const drop = document.createElement("div");
        drop.style.cssText = subjectDrop.style.cssText;
        const target = { image: location.image };
        renderDrop(drop, location.image, "Drop location image here");
        wireDrop(drop, target);
        const buttons = document.createElement("div");
        buttons.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
        const createZImage = makeButton("Create with ZImage", "primary");
        const upload = makeButton("Upload", "primary");
        const clear = makeButton("Clear");
        const remove = makeButton("Remove");
        createZImage.onclick = () => createFluxReferenceWithZImage("location", location, `${location.name || ""}\n${location.description || ""}`, location.name || `location_${index + 1}`);
        upload.onclick = () => uploadFor(target);
        clear.onclick = () => {
          location.image = { path: "", data: "", name: "" };
          renderAll();
        };
        remove.onclick = () => {
          refs.locations.splice(index, 1);
          for (const segment of allEditableSegments()) {
            if (refs.scene_map?.[segment.id] === location.id) refs.scene_map[segment.id] = "";
          }
          renderAll();
        };
        name.addEventListener("input", () => {
          location.name = name.value;
          renderMapping();
        });
        description.addEventListener("input", () => {
          location.description = description.value;
        });
        buttons.append(createZImage, upload, clear, remove);
        row.append(makeField("Location name", name), makeField("Description / prompt", description), used, drop, buttons);
        locationsList.append(row);
      });
    }

    function renderMapping() {
      mappingList.innerHTML = "";
      allEditableSegments().forEach((segment, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:minmax(100px,.9fr) minmax(120px,1fr);gap:8px;align-items:center;";
        const label = document.createElement("div");
        label.textContent = `${index + 1}. ${segment.label || `Scene ${index + 1}`}`;
        label.style.cssText = "font-size:12px;color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const select = document.createElement("select");
        select.style.cssText = "width:100%;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#f8fafc;padding:8px;font-size:12px;";
        select.append(new Option("Unassigned", ""));
        refs.locations.forEach((location) => select.append(new Option(location.name || "Location", location.id)));
        select.value = refs.scene_map?.[segment.id] || "";
        select.onchange = () => {
          refs.scene_map[segment.id] = select.value;
          renderLocations();
        };
        row.append(label, select);
        mappingList.append(row);
      });
    }

    function renderAll() {
      refs.subject.description = subjectDescription.value;
      updateSubjectDrop();
      renderLocations();
      renderMapping();
    }

    uploadSubject.onclick = () => uploadFor(refs.subject);
    createSubjectZImage.onclick = () => createFluxReferenceWithZImage("subject", refs.subject, refs.subject.description || subjectDescription.value || "", "subject_reference");
    clearSubject.onclick = () => {
      refs.subject.image = { path: "", data: "", name: "" };
      renderAll();
    };
    addLocation.onclick = () => {
      createLocation();
      renderAll();
    };
    extractLocations.onclick = extractLocationsWithGemma;
    autoMapLocations.onclick = autoMapLocationsWithGemma;
    useSubject.input.onchange = () => { refs.use_subject_reference = Boolean(useSubject.input.checked); };
    useLocations.input.onchange = () => { refs.use_location_references = Boolean(useLocations.input.checked); };
    includeManual.input.onchange = () => { refs.include_manual_ingredients = Boolean(includeManual.input.checked); };
    subjectDescription.addEventListener("input", () => { refs.subject.description = subjectDescription.value; });
    wireDrop(subjectDrop, refs.subject);
    close.onclick = () => backdrop.remove();
    cancel.onclick = () => backdrop.remove();
    save.onclick = async () => {
      refs.subject.description = subjectDescription.value;
      refs.use_subject_reference = Boolean(useSubject.input.checked);
      refs.use_location_references = Boolean(useLocations.input.checked);
      refs.include_manual_ingredients = Boolean(includeManual.input.checked);
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      renderFluxIngredientList(activeSegment());
      renderNBIngredientList(activeSegment());
      render();
      await autoSaveSessionQuiet(`${referenceBuilderTargetLabel} reference builder`);
      toast(`${referenceBuilderTargetLabel} reference builder saved.`);
      backdrop.remove();
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
    renderAll();
  }

  function openPromptOptionsModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Prompt Options";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Edit or reload the final generated prompt text files for this project. Reload updates the scene boxes immediately and saves the session.";
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;";
    const imageGroup = document.createElement("div");
    imageGroup.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;display:flex;flex-direction:column;gap:8px;";
    const videoGroup = document.createElement("div");
    videoGroup.style.cssText = imageGroup.style.cssText;
    const imageHeading = document.createElement("div");
    imageHeading.textContent = "Image";
    imageHeading.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
    const videoHeading = document.createElement("div");
    videoHeading.textContent = "Video";
    videoHeading.style.cssText = imageHeading.style.cssText;
    const editT2I = makeButton("Edit Text to Image Prompts", "primary");
    const reloadT2I = makeButton("Reload Text to Image Prompts");
    const originalT2I = makeButton("Reload Original T2I Prompts");
    const clearT2I = makeButton("Clear All T2I Prompts");
    clearT2I.style.borderColor = "#7f1d1d";
    clearT2I.style.color = "#fecaca";
    const editI2V = makeButton("Edit Image to Video Prompts", "primary");
    const reloadI2V = makeButton("Reload Image to Video Prompts");
    const originalI2V = makeButton("Reload Original I2V Prompts");
    const clearI2V = makeButton("Clear All I2V Prompts");
    clearI2V.style.borderColor = "#7f1d1d";
    clearI2V.style.color = "#fecaca";
    imageGroup.append(imageHeading, editT2I, reloadT2I, originalT2I, clearT2I);
    videoGroup.append(videoHeading, editI2V, reloadI2V, originalI2V, clearI2V);
    grid.append(imageGroup, videoGroup);
    box.append(header, note, grid);
    backdrop.append(box);
    document.body.append(backdrop);
    const run = (action) => {
      backdrop.remove();
      action();
    };
    close.onclick = () => backdrop.remove();
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
    editT2I.onclick = () => run(() => editFinalPromptList("t2i"));
    reloadT2I.onclick = () => run(() => reloadFinalPromptList("t2i", false));
    originalT2I.onclick = () => run(() => reloadFinalPromptList("t2i", true));
    clearT2I.onclick = () => run(() => clearFinalPromptList("t2i"));
    editI2V.onclick = () => run(() => editFinalPromptList("i2v"));
    reloadI2V.onclick = () => run(() => reloadFinalPromptList("i2v", false));
    originalI2V.onclick = () => run(() => reloadFinalPromptList("i2v", true));
    clearI2V.onclick = () => run(() => clearFinalPromptList("i2v"));
  }

  function render() {
    drawWaveform();
    renderSegments();
    renderList();
    updateSelectedMediaTools();
    updateMultiSelectButton();
    timelineInfo.textContent = `${state.segments.length} base / ${state.overlaySegments.length} insert${state.overlaySegments.length === 1 ? "" : "s"} | ${formatTime(state.duration)}`;
  }

  async function loadAudio() {
    try {
      loadButton.disabled = true;
      loadButton.textContent = "Loading...";
      if (!audioInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_audio_srt_paths");
        audioInput.value = paths.audio_path || "";
        if (!audioInput.value.trim()) {
          throw new Error(`No temp audio file found.\nAudio folder: ${paths.audio_folder}`);
        }
      }
      const data = await postJson("/vrgdg/music_builder/analyze_audio", {
        audio_path: audioInput.value,
        project_folder: projectInput.value || state.projectFolder || "",
        target_peaks: 1800,
      }, 90000);
      audioInput.value = data.audio_path || audioInput.value;
      state.duration = Number(data.duration || 0);
      state.peaks = data.peaks || [];
      state.beats = data.beats || [];
      showBeatMarkersIfAvailable();
      audio.src = audioUrl(data.audio_path || audioInput.value);
      audio.load();
      globalScrub.max = String(Math.max(0, state.duration));
      setWidgetValue(node, "audio_path", data.audio_path || audioInput.value);
      if (!state.segments.length) {
        state.segments.push(newSegment(0, Math.min(4, Math.max(1, state.duration || 4))));
        state.activeId = state.segments[0].id;
      }
      syncInspector();
      render();
      toast(`Loaded audio: ${formatTime(state.duration)}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      loadButton.disabled = false;
      loadButton.textContent = "Load Audio";
    }
  }

  async function loadSrt(options = {}) {
    try {
      if (!srtInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_audio_srt_paths");
        srtInput.value = paths.srt_path || "";
        if (!srtInput.value.trim()) {
          throw new Error(`No temp SRT file found.\nSRT folder: ${paths.srt_folder}`);
        }
      }
      const data = await postJson("/vrgdg/music_builder/load_srt", {
        srt_path: srtInput.value,
      });
      pushHistory();
      state.segments = data.segments || [];
      state.overlaySegments = [];
      state.srtPath = data.srt_path || "";
      state.activeId = state.segments[0]?.id || "";
      state.timingFrozen = true;
      state.srtMode = true;
      freezeTimingControl.input.checked = true;
      syncInspector();
      render();
      toast(`Loaded ${state.segments.length} SRT segment${state.segments.length === 1 ? "" : "s"}.\nTiming is frozen.`);
    } catch (error) {
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
    }
  }

  async function pickPath(kind, input) {
    try {
      const data = await postJson("/vrgdg/music_builder/pick_path", { kind });
      if (data.path) {
        input.value = data.path;
        return data.path;
      }
      return "";
    } catch (error) {
      toast(String(error?.message || error), true);
      return "";
    }
  }

  function chooseProjectAudioFile(file) {
    if (!file) return;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) {
      toast("Set the project folder first so the audio can be copied there.", true);
      return;
    }
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const data = await postJson("/vrgdg/music_builder/save_project_audio", {
          project_folder: projectFolder,
          audio_data: String(reader.result || ""),
          audio_name: file.name || "project_audio.wav",
        }, 180000);
        audioInput.value = data.saved_path || "";
        state.duration = Number(data.duration || 0);
        state.peaks = data.peaks || [];
        state.beats = data.beats || [];
        state.sceneAudioGlobalTime = 0;
        audio.src = audioUrl(audioInput.value);
        audio.load();
        setWidgetValue(node, "audio_path", audioInput.value);
        render();
        toast(`Loaded audio:\n${audioInput.value}`);
      } catch (error) {
        toast(String(error?.message || error), true);
      }
    };
    reader.onerror = () => toast("Failed to read the audio file.", true);
    reader.readAsDataURL(file);
  }

  function chooseProjectSrtFile(file) {
    if (!file) return;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) {
      toast("Set the project folder first so the SRT can be copied there.", true);
      return;
    }
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const data = await postJson("/vrgdg/music_builder/save_project_srt", {
          project_folder: projectFolder,
          srt_text: String(reader.result || ""),
        }, 60000);
        pushHistory();
        srtInput.value = data.srt_path || "";
        state.srtPath = srtInput.value;
        state.segments = data.segments || [];
        state.overlaySegments = [];
        state.activeId = state.segments[0]?.id || "";
        state.timingFrozen = true;
        state.srtMode = true;
        setWidgetValue(node, "srt_path", state.srtPath);
        syncInspector();
        render();
        toast(`Loaded ${state.segments.length} SRT segment${state.segments.length === 1 ? "" : "s"}.\n${state.srtPath}`);
      } catch (error) {
        toast(String(error?.message || error), true);
      }
    };
    reader.onerror = () => toast("Failed to read the SRT file.", true);
    reader.readAsText(file);
  }

  function openSettingsModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.58);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(980px,calc(100vw - 40px));max-height:calc(100vh - 48px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:14px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.textContent = "Builder Settings";
    title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;";
    const modalClose = makeButton("Close");
    header.append(title, modalClose);
    const pathGrid = document.createElement("div");
    pathGrid.style.cssText = "display:grid;grid-template-columns:1fr;gap:10px;";
    pathGrid.append(
      makePickerField("Audio file path", audioInput, pickAudioButton),
      makeField("Project folder", projectInput),
      makePickerField("SRT path", srtInput, pickSrtButton),
    );
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:8px;";
    actions.append(loadButton, loadSrtButton);
    const note = document.createElement("div");
    note.textContent = "These are workflow setup paths and one-time loading actions. Keeping them here leaves more room for scene editing.";
    note.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.45;";
    box.append(header, pathGrid, actions, note);
    backdrop.append(box);
    document.body.append(backdrop);
    modalClose.onclick = () => backdrop.remove();
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  async function importPromptJson(options = {}) {
    try {
      if (!promptJsonInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_context_paths");
        promptJsonInput.value = paths.concept_prompts_path || "";
        state.promptJsonPath = promptJsonInput.value;
      }
      const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
        prompt_json_path: promptJsonInput.value,
      });
      const prompts = data.prompts || [];
      if (options.pushHistory !== false) pushHistory();
      state.promptJsonPath = data.prompt_json_path || promptJsonInput.value;
      for (let index = 0; index < state.segments.length && index < prompts.length; index++) {
        const segment = state.segments[index];
        segment.notes = prompts[index];
        segment.flux_notes = prompts[index];
        segment.nb_notes = prompts[index];
        if (!state.segments[index].label || /^Prompt\s+\d+$/i.test(state.segments[index].label)) {
          state.segments[index].label = `Scene ${index + 1}`;
        }
      }
      syncInspector();
      render();
      if (!options.quiet) toast(`Imported ${prompts.length} prompt${prompts.length === 1 ? "" : "s"} into segment notes.`);
      return prompts;
    } catch (error) {
      if (options.throwOnError) throw error;
      if (!options.quiet) toast(String(error?.message || error), true);
      return [];
    }
  }

  async function loadDefaultContextPaths() {
    try {
      const data = await getJson("/vrgdg/music_builder/default_context_paths");
      promptJsonInput.value = data.concept_prompts_path || "";
      i2vMotionJsonInput.value = data.i2v_motion_notes_path || "";
      themeStyleInput.value = data.theme_style_path || "";
      storyIdeaInput.value = data.story_idea_path || "";
      subjectSceneInput.value = data.subject_scene_path || "";
      state.promptJsonPath = promptJsonInput.value;
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      state.themeStylePath = themeStyleInput.value;
      state.storyIdeaPath = storyIdeaInput.value;
      state.subjectScenePath = subjectSceneInput.value;
      state.useVrgdgTextContext = true;
      pushHistory();
      useVrgdgTextContext.input.checked = true;
      toast("Using VRGDG_TEMP TextFiles paths as Gemma context.");
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  function clearGeneratedSceneOutputsForImport() {
    for (const segment of allEditableSegments()) {
      ensureSegmentRuntimeFields(segment);
      segment.t2i_prompt = "";
      segment.i2v_prompt = "";
      segment.enhance_prompt = "";
      segment.approved_image_path = "";
      segment.custom_image_path = "";
      segment.custom_image_data = "";
      segment.custom_image_name = "";
      segment.image = null;
      segment.image_history = [];
      segment.image_history_index = -1;
      segment.video_path = "";
      segment.video_history = [];
      segment.video_history_index = -1;
      segment.video_source_path = "";
      segment.video_folder = "";
      segment.video_output = null;
      segment.video_status = "none";
      segment.preview_mode = "image";
      segment.flux_prompt = "";
      segment.flux_image_ingredients = [];
    }
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
  }

  async function autoLoadAll(options = {}) {
    try {
      autoLoadAllButton.disabled = true;
      autoLoadAllButton.textContent = "Importing...";
      pushHistory();
      let paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
        project_folder: projectInput.value || state.projectFolder || "",
      });
      let exists = paths.exists || {};
      let sourceLabel = "this project";
      const sourceProjectFolder = String(options.sourceProjectFolder || "").trim();
      let currentPrompts = [];
      let currentMotionNotes = [];
      if (exists.concept_prompts_path) {
        try {
          currentPrompts = await loadPromptJsonFromPath(paths.concept_prompts_path);
        } catch (_error) {
          currentPrompts = [];
        }
      }
      if (exists.i2v_motion_notes_path) {
        try {
          currentMotionNotes = await loadI2VMotionNotesFromPath(paths.i2v_motion_notes_path);
        } catch (_error) {
          currentMotionNotes = [];
        }
      }
      const hasCurrentPrompts = currentPrompts.some((item) => String(item || "").trim());
      const hasCurrentMotionNotes = currentMotionNotes.some((item) => String(item || "").trim());
      if (sourceProjectFolder) {
        const targetProjectFolder = String(projectInput.value || state.projectFolder || "").trim();
        if (targetProjectFolder && targetProjectFolder.replace(/[\\/]+$/, "").toLowerCase() === sourceProjectFolder.replace(/[\\/]+$/, "").toLowerCase()) {
          paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
            project_folder: sourceProjectFolder,
          });
        } else {
          try {
            paths = await postJson("/vrgdg/music_builder/copy_prompt_creator_outputs", {
              project_folder: targetProjectFolder,
              source_project_folder: sourceProjectFolder,
            }, 90000);
          } catch (error) {
            if (/\b405\b/.test(String(error?.message || error))) {
              throw new Error("The Prompt Creator handoff backend route is not loaded yet. Fully restart ComfyUI, refresh the browser, then try Send To Video Creator again.");
            }
            throw error;
          }
        }
        exists = paths.exists || {};
        sourceLabel = paths.source_project_folder ? `selected Prompt Creator project:\n${paths.source_project_folder}` : "selected Prompt Creator project";
        if (!exists.srt_path || !exists.concept_prompts_path) {
          throw new Error("The selected Prompt Creator project does not have saved SRT and concept prompt outputs yet. Run or save Prompt Creator outputs first, then send it to Video Creator.");
        }
      } else if (!exists.srt_path || !exists.concept_prompts_path || !hasCurrentPrompts || !hasCurrentMotionNotes) {
        paths = await postJson("/vrgdg/music_builder/import_latest_prompt_creator_outputs", {
          project_folder: projectInput.value || state.projectFolder || "",
        }, 90000);
        exists = paths.exists || {};
        sourceLabel = paths.source_project_folder ? `latest Prompt Creator project:\n${paths.source_project_folder}` : "latest Prompt Creator project";
        if (!exists.srt_path || !exists.concept_prompts_path) {
          throw new Error("No previous Prompt Creator output was found. Run Prompt Creator first, then import it into this project.");
        }
      }
      if (paths.audio_path) audioInput.value = paths.audio_path;
      if (paths.srt_path) srtInput.value = paths.srt_path;
      promptJsonInput.value = paths.concept_prompts_path || "";
      i2vMotionJsonInput.value = exists.i2v_motion_notes_path ? paths.i2v_motion_notes_path || "" : "";
      state.lyricSegmentsPath = exists.lyric_segments_path ? paths.lyric_segments_path || "" : "";
      themeStyleInput.value = exists.theme_style_path ? paths.theme_style_path || "" : "";
      storyIdeaInput.value = exists.story_idea_path ? paths.story_idea_path || "" : "";
      subjectSceneInput.value = exists.subject_scene_path ? paths.subject_scene_path || "" : "";
      state.promptJsonPath = promptJsonInput.value;
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      state.themeStylePath = themeStyleInput.value;
      state.storyIdeaPath = storyIdeaInput.value;
      state.subjectScenePath = subjectSceneInput.value;
      state.useVrgdgTextContext = true;
      useVrgdgTextContext.input.checked = true;
      if (paths.audio_path) await loadAudio();
      await loadSrt({ throwOnError: true });
      let importedPrompts = [];
      let importedMotionNotes = [];
      let importedLyrics = [];
      if (promptJsonInput.value) {
        importedPrompts = await importPromptJson({ quiet: true, pushHistory: false });
      }
      if (i2vMotionJsonInput.value) {
        importedMotionNotes = await importI2VMotionJson({ quiet: true, pushHistory: false });
      }
      if (state.lyricSegmentsPath) {
        try {
          importedLyrics = await loadLyricSegmentsFromPath(state.lyricSegmentsPath);
        } catch (error) {
          console.warn("[VRGDG Music Builder] Could not import lyric segment status:", error);
          importedLyrics = [];
        }
      }
      for (let index = 0; index < state.segments.length && index < importedLyrics.length; index += 1) {
        state.segments[index].lyric_text = String(importedLyrics[index] || "").trim();
      }
      const nonEmptyPrompts = importedPrompts.filter((item) => String(item || "").trim()).length;
      const nonEmptyMotionNotes = importedMotionNotes.filter((item) => String(item || "").trim()).length;
      if (!nonEmptyPrompts) {
        throw new Error(
          `Prompt Creator import found ${promptJsonInput.value || "ConceptPrompts.txt"}, but it contains no usable prompts. Run Prompt Creator first, then import it into this project.`
        );
      }
      clearGeneratedSceneOutputsForImport();
      syncInspector();
      render();
      await autoSaveSessionQuiet("prompt creator import");
      const parts = [
        `Imported ${nonEmptyPrompts} concept prompt${nonEmptyPrompts === 1 ? "" : "s"}`,
        nonEmptyMotionNotes
          ? `${nonEmptyMotionNotes} I2V motion note${nonEmptyMotionNotes === 1 ? "" : "s"}`
          : "no I2V motion notes found",
      ];
      toast(`${parts.join(" and ")} from ${sourceLabel}. Previous generated images/videos were cleared from this project session.`);
    } catch (error) {
      toast(String(error?.message || error), true);
      if (options.throwOnError) throw error;
    } finally {
      autoLoadAllButton.disabled = false;
      autoLoadAllButton.textContent = "Import Data From Prompt Creator";
    }
  }

  async function runClearMemoryWorkflow() {
    let progress = null;
    try {
      clearMemoryButton.disabled = true;
      clearMemoryButton.textContent = "Clearing...";
      progress = createProgressWindow("Clearing memory");
      progress.set("Clearing memory directly without queueing a workflow...", 35);
      const data = await postJson("/vrgdg/music_builder/clear_memory_direct", {}, 120000);
      progress.set(data.message || "Memory cleanup finished.", 100);
      progress.close(4500);
      toast("Memory cleanup workflow finished.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      clearMemoryButton.disabled = false;
      clearMemoryButton.textContent = "Clear Memory";
    }
  }

  async function runClearMemoryWorkflowQuiet(progress, label, percent = 95) {
    progress?.set(`Clearing memory directly after ${label}...`, percent);
    const data = await postJson("/vrgdg/music_builder/clear_memory_direct", {}, 120000);
    const output = data.message || `Memory cleanup finished after ${label}.`;
    progress?.set(output, percent);
    return output;
  }

  async function runImageMemoryCleanupQuiet(progress, label, percent = 95) {
    try {
      return await runClearMemoryWorkflowQuiet(progress, label, percent);
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Memory cleanup after ${label} failed:`, error);
      return "";
    }
  }

  async function importI2VMotionJson(options = {}) {
    try {
      if (!i2vMotionJsonInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_context_paths");
        i2vMotionJsonInput.value = paths.i2v_motion_notes_path || "";
        state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      }
      const notes = await loadI2VMotionNotesFromPath(i2vMotionJsonInput.value);
      if (options.pushHistory !== false) pushHistory();
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      for (let index = 0; index < state.segments.length && index < notes.length; index++) {
        state.segments[index].i2v_notes = notes[index];
      }
      syncInspector();
      render();
      if (!options.quiet) toast(`Imported ${notes.length} I2V motion note${notes.length === 1 ? "" : "s"} into scenes.`);
      return notes;
    } catch (error) {
      if (options.throwOnError) throw error;
      if (!options.quiet) toast(String(error?.message || error), true);
      return [];
    }
  }

  function currentSessionData() {
    return {
      segments: state.segments,
      overlay_segments: state.overlaySegments,
      active_track: state.activeTrack,
      timing_frozen: state.timingFrozen,
      srt_mode: state.srtMode,
      prompt_json_path: state.promptJsonPath,
      i2v_motion_json_path: state.i2vMotionJsonPath,
      image_trigger_phrase: state.imageTriggerPhrase,
      video_trigger_phrase: state.videoTriggerPhrase,
      use_vrgdg_text_context: state.useVrgdgTextContext,
      theme_style_path: state.themeStylePath,
      story_idea_path: state.storyIdeaPath,
      subject_scene_path: state.subjectScenePath,
      text_gemma_runner: state.textGemmaRunner || "builtin",
      lm_studio_base_url: state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
      lm_studio_model: state.lmStudioModel || "",
      lm_studio_api_key: state.lmStudioApiKey || "",
      waveform_mode: state.waveformMode,
      snap_to_beats: state.snapToBeats,
      show_beat_markers: state.showBeatMarkers,
      audio_peaks: Array.isArray(state.peaks) ? state.peaks : [],
      beat_markers: Array.isArray(state.beats) ? state.beats : [],
      left_panel_width: state.leftPanelWidth,
      right_panel_width: state.rightPanelWidth,
      timeline_panel_height: state.timelinePanelHeight,
      timeline_zoom: state.timelineZoom,
      auto_save_enabled: state.autoSaveEnabled,
      image_model_mode: state.imageModelMode,
      zimage_settings: state.zimageSettings,
      flux_klein_settings: state.fluxKleinSettings,
      nb_image_settings: state.nbImageSettings,
      ernie_image_settings: state.ernieImageSettings,
      use_flux_global_image_ingredients: Boolean(state.useFluxGlobalImageIngredients),
      flux_global_image_ingredients: Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [],
      flux_reference_builder: normalizeFluxReferenceBuilder(state.fluxReferenceBuilder),
      z_enhance_settings: state.zEnhanceSettings,
      video_model_mode: state.videoModelMode || "i2v",
      i2v_video_settings: state.i2vVideoSettings,
      prompt_tools_hint_prefs: state.promptToolsHintPrefs || {},
    };
  }

  function activeProjectFolderForSave() {
    const folder = String(state.projectFolder || "").trim();
    if (folder) {
      projectInput.value = folder;
      setWidgetValue(node, "project_folder", folder);
    }
    return folder;
  }

  async function stopCurrentWorkflow() {
    state.batchCancelled = true;
    pauseAllAudio();
    let progress = null;
    try {
      stopWorkflowButton.disabled = true;
      stopWorkflowButton.textContent = "Stopping...";
      progress = createProgressWindow("Stopping workflow");
      progress.set("Interrupting ComfyUI and clearing pending queue...", 20);
      await cancelComfyExecutionAndWaitIdle((status) => {
        progress.set(`${status}`, 45);
      }, { shouldCancel: () => false });
      progress.set("Clearing memory after stop...", 45);
      await runClearMemoryWorkflowQuiet(progress, "stop request", 85);
      progress.set("Stop requested and memory cleanup finished.", 100);
      progress.close(3000);
      toast("Stop requested. Memory cleanup ran.");
    } catch (error) {
      progress?.set(`Error while stopping:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      stopWorkflowButton.disabled = false;
      stopWorkflowButton.textContent = "Stop";
    }
  }

  async function saveSession(options = {}) {
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      const projectFolder = activeProjectFolderForSave();
      if (!projectFolder) {
        const message = "Create a new project, load a project, or use Save Project As before Quick Save.";
        if (options.throwOnError) throw new Error(message);
        if (!options.quiet) toast(message, true);
        return null;
      }
      const data = await postJson("/vrgdg/music_builder/save_session", {
        audio_path: audioInput.value,
        project_folder: projectFolder,
        session: currentSessionData(),
      }, 60000);
      await syncPromptJsonFromSegments("session save");
      await syncI2VMotionJsonFromSegments("session save");
      state.projectFolder = data.project_folder || "";
      state.sessionPath = data.session_path || "";
      state.srtPath = data.srt_path || "";
      if (data.session) {
        state.overlaySegments = Array.isArray(data.session.overlay_segments) ? data.session.overlay_segments : state.overlaySegments;
        ensureAllSegmentRuntimeFields();
        state.activeTrack = data.session.active_track || state.activeTrack || "base";
        state.timingFrozen = Boolean(data.session.timing_frozen);
        state.srtMode = Boolean(data.session.srt_mode);
        state.promptJsonPath = data.session.prompt_json_path || state.promptJsonPath;
        state.i2vMotionJsonPath = data.session.i2v_motion_json_path || state.i2vMotionJsonPath;
        state.imageTriggerPhrase = data.session.image_trigger_phrase || state.imageTriggerPhrase || "";
        state.videoTriggerPhrase = data.session.video_trigger_phrase || state.videoTriggerPhrase || "";
        state.useVrgdgTextContext = data.session.use_vrgdg_text_context ?? state.useVrgdgTextContext;
        state.themeStylePath = data.session.theme_style_path || state.themeStylePath;
        state.storyIdeaPath = data.session.story_idea_path || state.storyIdeaPath;
        state.subjectScenePath = data.session.subject_scene_path || state.subjectScenePath;
        state.textGemmaRunner = data.session.text_gemma_runner || state.textGemmaRunner || "builtin";
        state.lmStudioBaseUrl = data.session.lm_studio_base_url || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
        state.lmStudioModel = data.session.lm_studio_model || state.lmStudioModel || "";
        state.lmStudioApiKey = data.session.lm_studio_api_key || state.lmStudioApiKey || "";
        state.waveformMode = data.session.waveform_mode || state.waveformMode;
        state.snapToBeats = data.session.snap_to_beats ?? state.snapToBeats;
        state.peaks = Array.isArray(data.session.audio_peaks) ? data.session.audio_peaks : state.peaks;
        state.beats = Array.isArray(data.session.beat_markers) ? data.session.beat_markers : state.beats;
        setBeatMarkersVisible(data.session.show_beat_markers ?? state.showBeatMarkers);
        state.leftPanelWidth = data.session.left_panel_width || state.leftPanelWidth;
        state.rightPanelWidth = data.session.right_panel_width || state.rightPanelWidth;
        state.timelinePanelHeight = data.session.timeline_panel_height || state.timelinePanelHeight;
        state.timelineZoom = data.session.timeline_zoom || state.timelineZoom;
        state.autoSaveEnabled = data.session.auto_save_enabled ?? state.autoSaveEnabled;
        state.imageModelMode = data.session.image_model_mode || data.session.flux_klein_settings?.image_model_mode || state.imageModelMode || "zimage";
        state.pxPerSecond = state.timelineZoom;
        waveformModeSelect.value = state.waveformMode;
        snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
        autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
        applyLayoutSizes();
        state.zimageSettings = data.session.zimage_settings || state.zimageSettings;
        state.fluxKleinSettings = data.session.flux_klein_settings || state.fluxKleinSettings;
        state.nbImageSettings = data.session.nb_image_settings || state.nbImageSettings;
        state.ernieImageSettings = data.session.ernie_image_settings || state.ernieImageSettings;
        state.useFluxGlobalImageIngredients = Boolean(data.session.use_flux_global_image_ingredients);
        state.fluxGlobalImageIngredients = Array.isArray(data.session.flux_global_image_ingredients) ? data.session.flux_global_image_ingredients : [];
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(data.session.flux_reference_builder);
        state.zEnhanceSettings = data.session.z_enhance_settings || state.zEnhanceSettings;
        state.videoModelMode = data.session.video_model_mode || state.videoModelMode || "i2v";
        state.i2vVideoSettings = data.session.i2v_video_settings || state.i2vVideoSettings;
        state.promptToolsHintPrefs = data.session.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
        syncZImageSettingsPanel();
        syncFluxKleinPanel();
        syncErnieImagePanel();
        syncI2VVideoSettingsPanel();
        syncVideoModePanel();
        syncInspector();
      }
      projectInput.value = state.projectFolder;
      setWidgetValue(node, "project_folder", state.projectFolder);
      setWidgetValue(node, "session_path", state.sessionPath);
      setWidgetValue(node, "srt_path", state.srtPath);
      rememberLastProject(state.projectFolder);
      if (!options.quiet) toast(`Saved builder session and SRT.\n${state.srtPath}`);
      return data;
    } catch (error) {
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
      return null;
    }
  }

  async function saveSessionForSceneVideo() {
    updateActiveFromInputs();
    saveI2VVideoSettingsFromPanel();
    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) throw new Error("Create or load a project before rendering scene videos.");
    const data = await postJson("/vrgdg/music_builder/save_session", {
      audio_path: audioInput.value,
      project_folder: projectFolder,
      session: currentSessionData(),
    }, 60000);
    await syncPromptJsonFromSegments("scene video save");
    await syncI2VMotionJsonFromSegments("scene video save");
    state.projectFolder = data.project_folder || state.projectFolder;
    state.sessionPath = data.session_path || state.sessionPath;
    state.srtPath = data.srt_path || state.srtPath;
    projectInput.value = state.projectFolder;
    srtInput.value = state.srtPath;
    setWidgetValue(node, "project_folder", state.projectFolder);
    setWidgetValue(node, "session_path", state.sessionPath);
    setWidgetValue(node, "srt_path", state.srtPath);
    rememberLastProject(state.projectFolder);
    return state.srtPath;
  }

  async function autoSaveSessionQuiet(reason = "") {
    if (!state.autoSaveEnabled) return false;
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      const projectFolder = activeProjectFolderForSave();
      if (!projectFolder) {
        console.warn(`[VRGDG Music Builder] Autosave skipped before ${reason || "action"} because no active project is set.`);
        return false;
      }
      const data = await postJson("/vrgdg/music_builder/save_session", {
        audio_path: audioInput.value,
        project_folder: projectFolder,
        session: currentSessionData(),
      }, 60000);
      state.projectFolder = data.project_folder || state.projectFolder;
      state.sessionPath = data.session_path || state.sessionPath;
      state.srtPath = data.srt_path || state.srtPath;
      projectInput.value = state.projectFolder || projectInput.value;
      srtInput.value = state.srtPath || srtInput.value;
      setWidgetValue(node, "project_folder", state.projectFolder);
      setWidgetValue(node, "session_path", state.sessionPath);
      setWidgetValue(node, "srt_path", state.srtPath);
      rememberLastProject(state.projectFolder);
      await syncPromptJsonFromSegments(`autosave ${reason || "session"}`);
      await syncI2VMotionJsonFromSegments(`autosave ${reason || "session"}`);
      if (reason) console.log(`[VRGDG Music Builder] Autosaved session/SRT: ${reason}`, state.sessionPath || "");
      return true;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Autosave failed:", error);
      toast(`Autosave failed before ${reason || "this action"}:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function loadSessionFromProject(projectFolder) {
    try {
      const folder = String(projectFolder || "").trim();
      if (!folder) {
        toast("Choose a project folder that contains vrgdg_builder_session.json.", true);
        return false;
      }
      const data = await postJson("/vrgdg/music_builder/load_session", {
        project_folder: folder,
      });
      const session = data.session || {};
      pushHistory();
      state.segments = Array.isArray(session.segments) ? session.segments : [];
      state.overlaySegments = Array.isArray(session.overlay_segments) ? session.overlay_segments : [];
      ensureAllSegmentRuntimeFields();
      state.projectFolder = data.project_folder || folder;
      state.sessionPath = data.session_path || "";
      state.srtPath = data.srt_path || session.srt_path || state.srtPath;
      state.timingFrozen = Boolean(session.timing_frozen);
      state.srtMode = Boolean(session.srt_mode);
      state.promptJsonPath = session.prompt_json_path || "";
      state.i2vMotionJsonPath = session.i2v_motion_json_path || "";
      state.imageTriggerPhrase = session.image_trigger_phrase || "";
      state.videoTriggerPhrase = session.video_trigger_phrase || "";
      state.useVrgdgTextContext = session.use_vrgdg_text_context ?? true;
      state.themeStylePath = session.theme_style_path || "";
      state.storyIdeaPath = session.story_idea_path || "";
      state.subjectScenePath = session.subject_scene_path || "";
      state.textGemmaRunner = session.text_gemma_runner || state.textGemmaRunner || "builtin";
      state.lmStudioBaseUrl = session.lm_studio_base_url || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = session.lm_studio_model || state.lmStudioModel || "";
      state.lmStudioApiKey = session.lm_studio_api_key || state.lmStudioApiKey || "";
      state.waveformMode = session.waveform_mode || state.waveformMode || "medium";
      state.snapToBeats = session.snap_to_beats ?? state.snapToBeats ?? true;
      state.peaks = Array.isArray(session.audio_peaks) ? session.audio_peaks : state.peaks;
      state.beats = Array.isArray(session.beat_markers) ? session.beat_markers : state.beats;
      setBeatMarkersVisible(session.show_beat_markers ?? state.showBeatMarkers ?? false);
      state.leftPanelWidth = session.left_panel_width || state.leftPanelWidth || 260;
      state.rightPanelWidth = session.right_panel_width || state.rightPanelWidth || 360;
      state.timelinePanelHeight = session.timeline_panel_height || state.timelinePanelHeight || 300;
      state.timelineZoom = session.timeline_zoom || state.timelineZoom || 45;
      state.autoSaveEnabled = session.auto_save_enabled ?? state.autoSaveEnabled ?? true;
      state.imageModelMode = session.image_model_mode || session.flux_klein_settings?.image_model_mode || state.imageModelMode || "zimage";
      state.pxPerSecond = state.timelineZoom;
      waveformModeSelect.value = state.waveformMode;
      snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
      autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
      applyLayoutSizes();
      state.zimageSettings = session.zimage_settings || state.zimageSettings;
      state.fluxKleinSettings = session.flux_klein_settings || state.fluxKleinSettings;
      state.nbImageSettings = session.nb_image_settings || state.nbImageSettings;
      state.ernieImageSettings = session.ernie_image_settings || state.ernieImageSettings;
      state.useFluxGlobalImageIngredients = Boolean(session.use_flux_global_image_ingredients);
      state.fluxGlobalImageIngredients = Array.isArray(session.flux_global_image_ingredients) ? session.flux_global_image_ingredients : [];
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(session.flux_reference_builder);
      state.zEnhanceSettings = session.z_enhance_settings || state.zEnhanceSettings;
      state.videoModelMode = session.video_model_mode || state.videoModelMode || "i2v";
      state.i2vVideoSettings = session.i2v_video_settings || state.i2vVideoSettings;
      state.promptToolsHintPrefs = session.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
      if (session.audio_path) {
        audioInput.value = session.audio_path;
        setWidgetValue(node, "audio_path", session.audio_path);
        try {
          const audioData = await postJson("/vrgdg/music_builder/analyze_audio", {
            audio_path: session.audio_path,
            project_folder: projectInput.value || state.projectFolder || "",
            target_peaks: 1800,
          });
          audioInput.value = audioData.audio_path || session.audio_path;
          setWidgetValue(node, "audio_path", audioInput.value);
          state.duration = Number(audioData.duration || 0);
          state.peaks = Array.isArray(audioData.peaks) && audioData.peaks.length ? audioData.peaks : state.peaks;
          state.beats = Array.isArray(audioData.beats) && audioData.beats.length ? audioData.beats : state.beats;
          showBeatMarkersIfAvailable();
          audio.src = audioUrl(audioInput.value);
          audio.load();
        } catch (error) {
          toast(`Loaded session, but audio waveform failed:\n${String(error?.message || error)}`, true);
        }
      }
      try {
        const scan = await postJson("/vrgdg/music_builder/scan_scene_videos", {
          project_folder: state.projectFolder,
        });
        const videos = scan.videos || {};
        const videoBackups = scan.video_backups || {};
        let restored = 0;
        for (const [index, segment] of state.segments.entries()) {
          const sceneKey = String(index + 1);
          const videoPath = videos[sceneKey] || "";
          segment.video_backup_paths = Array.isArray(videoBackups[sceneKey]) ? videoBackups[sceneKey] : [];
          if (!videoPath) continue;
          segment.video_path = videoPath;
          segment.video_folder = scan.video_folder || segment.video_folder || "";
          segment.video_status = "done";
          normalizeSegmentVideoHistory(segment);
          restored += 1;
        }
        for (const [index, segment] of state.overlaySegments.entries()) {
          const sceneKey = String(10000 + index + 1);
          const videoPath = videos[sceneKey] || "";
          segment.video_backup_paths = Array.isArray(videoBackups[sceneKey]) ? videoBackups[sceneKey] : [];
          if (!videoPath) continue;
          segment.video_path = videoPath;
          segment.video_folder = scan.video_folder || segment.video_folder || "";
          segment.video_status = "done";
          normalizeSegmentVideoHistory(segment);
          restored += 1;
        }
        if (restored) console.log(`[VRGDG Music Builder] Restored ${restored} scene video path(s) from project folder.`);
      } catch (error) {
        console.warn("[VRGDG Music Builder] Scene video scan failed:", error);
      }
      projectInput.value = state.projectFolder;
      srtInput.value = state.srtPath;
      setWidgetValue(node, "project_folder", state.projectFolder);
      setWidgetValue(node, "session_path", state.sessionPath);
      setWidgetValue(node, "srt_path", state.srtPath);
      rememberLastProject(state.projectFolder);
      if (!state.i2vMotionJsonPath) {
        try {
          const paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
            project_folder: state.projectFolder,
          });
          if (paths?.exists?.i2v_motion_notes_path && paths.i2v_motion_notes_path) {
            state.i2vMotionJsonPath = paths.i2v_motion_notes_path;
          }
        } catch (error) {
          console.warn("[VRGDG Music Builder] Could not find project I2V motion notes path during load:", error);
        }
      }
      i2vMotionJsonInput.value = state.i2vMotionJsonPath || i2vMotionJsonInput.value || "";
      if (state.i2vMotionJsonPath && !hasAnyI2VMotionNotes(state.segments)) {
        const restoredNotes = await importI2VMotionJson({ quiet: true, pushHistory: false });
        if (restoredNotes.some((note) => String(note || "").trim())) {
          console.log(`[VRGDG Music Builder] Restored ${restoredNotes.length} I2V motion note(s) from project file.`);
        }
      }
      state.activeTrack = session.active_track || "base";
      state.activeId = state.segments[0]?.id || state.overlaySegments[0]?.id || "";
      syncZImageSettingsPanel();
      syncFluxKleinPanel();
      syncZEnhanceSettingsPanel();
      syncI2VVideoSettingsPanel();
      syncVideoModePanel();
      syncInspector();
      render();
      toast(`Loaded builder session.\n${state.sessionPath}`);
      return true;
    } catch (error) {
      toast(String(error?.message || error), true);
      return false;
    }
  }

  async function showStartupWelcome() {
    try {
      resetProjectState("", "", "");
    } catch (error) {
      console.warn("[VRGDG Music Builder] Fresh startup reset failed:", error);
      state.projectFolder = "";
      state.sessionPath = "";
      state.srtPath = "";
      state.segments = [newSegment(0, 4)];
      state.overlaySegments = [];
      state.activeId = state.segments[0]?.id || "";
    }
    let projects = [];
    try {
      const data = await getJson("/vrgdg/music_builder/list_projects");
      projects = Array.isArray(data.projects) ? data.projects : [];
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not list existing projects:", error);
    }
    const choice = await showWelcomeProjectModal(projects);
    if (!choice) return false;
    if (choice.action === "new") {
      const mode = await window.VRGDGMusicVideoPromptCreator?.chooseNewProjectMode?.();
      if (!mode) return false;
      const created = await newProject();
      if (created && mode === "prompt_creator") openPromptCreatorPanel();
      return Boolean(created);
    }
    if (choice.action === "load" && choice.project_folder) {
      return await loadSessionFromProject(choice.project_folder);
    }
    return false;
  }

  async function loadSession() {
    try {
      let projects = [];
      try {
        const data = await getJson("/vrgdg/music_builder/list_projects");
        projects = Array.isArray(data.projects) ? data.projects : [];
      } catch (error) {
        console.warn("[VRGDG Music Builder] Could not list existing projects:", error);
      }
      const choice = await showLoadProjectModal(projects);
      if (!choice?.project_folder) return;
      await loadSessionFromProject(choice.project_folder);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function loadLastProject() {
    const folder = getLastProject() || projectInput.value || state.projectFolder;
    if (!String(folder || "").trim()) {
      toast("No last project has been saved yet. Use Load Project first.", true);
      return;
    }
    await loadSessionFromProject(folder);
  }

  async function archiveGeneratedSceneImage(segment, imageInfo) {
    ensureSegmentRuntimeFields(segment);
    if (!segment || !imageInfo?.filename) return null;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) return null;
    try {
      const sceneNumber = sceneSlotNumber(segment);
      const data = await postJson("/vrgdg/music_builder/archive_scene_image", {
        image: imageInfo,
        project_folder: projectFolder,
        scene_number: sceneNumber,
      });
      if (data.saved_path && !segment.image_history.includes(data.saved_path)) {
        segment.image_history.push(data.saved_path);
        segment.image_history_index = segment.image_history.length - 1;
      }
      if (data.saved_path) segment.preview_mode = "image";
      return data.saved_path || null;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Failed to archive preview image:", error);
      return null;
    }
  }

  function addSceneImageHistoryPath(segment, imagePath) {
    ensureSegmentRuntimeFields(segment);
    if (!segment || !imagePath) return;
    if (!segment.image_history.includes(imagePath)) {
      segment.image_history.push(imagePath);
      segment.image_history_index = segment.image_history.length - 1;
    }
    segment.preview_mode = "image";
  }

  function cycleSegmentImageHistory(segment) {
    ensureSegmentRuntimeFields(segment);
    if (!segment?.image_history.length) return;
    pushHistory();
    const nextIndex = (Math.max(-1, Number(segment.image_history_index ?? -1)) + 1) % segment.image_history.length;
    const imagePath = segment.image_history[nextIndex];
    segment.image_history_index = nextIndex;
    segment.custom_image_path = imagePath;
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.image = null;
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
  }

  function addSegmentVideoHistoryPath(segment, videoPath) {
    ensureSegmentRuntimeFields(segment);
    if (!segment || !videoPath || isBackupSceneVideoPath(videoPath)) return;
    segment.video_path = videoPath;
    segment.video_cache_bust = Date.now();
    normalizeSegmentVideoHistory(segment);
    const currentIndex = segment.video_history.findIndex((item) => mediaPathKey(item) === mediaPathKey(videoPath));
    if (currentIndex >= 0) segment.video_history_index = currentIndex;
  }

  function cycleSegmentVideoHistory(segment) {
    ensureSegmentRuntimeFields(segment);
    if (!segment?.video_history.length) return;
    pushHistory();
    const nextIndex = (Math.max(-1, Number(segment.video_history_index ?? -1)) + 1) % segment.video_history.length;
    segment.video_history_index = nextIndex;
    segment.video_path = segment.video_history[nextIndex] || segment.video_path || "";
    segment.preview_mode = "video";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
  }

  function toggleSegmentPreviewMode(segment) {
    ensureSegmentRuntimeFields(segment);
    if (!segment) return;
    const hasImage = Boolean(segmentImageSource(segment));
    const hasVideo = Boolean(selectedSegmentVideoPath(segment));
    if (!hasImage || !hasVideo) return;
    pushHistory();
    segment.preview_mode = segment.preview_mode === "image" ? "video" : "image";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
  }

  function requireActiveSegment() {
    const segment = activeSegment();
    if (!segment) {
      toast("Hey, add a segment first.", true);
      return null;
    }
    return segment;
  }

  function isBlankStarterProject() {
    const segments = allEditableSegments();
    if (state.overlaySegments.length) return false;
    if (segments.length !== 1) return false;
    const segment = segments[0] || {};
    const emptyMedia = !segmentImageSource(segment) && !String(selectedSegmentVideoPath(segment) || "").trim();
    const emptyText = !String(segment.notes || segment.t2i_prompt || segment.flux_prompt || segment.i2v_notes || segment.i2v_prompt || "").trim();
    return emptyMedia && emptyText && /^new scene$/i.test(String(segment.label || "").trim());
  }

  function assertBatchNotStopped() {
    if (state.batchCancelled) throw new Error("Stopped by user.");
  }

  function t2iMissingReason(segment) {
    if (segment?.use_vision_reference && !String(segment.ref_image_path || "").trim()) {
      return "vision reference is turned on, but no reference image is loaded.";
    }
    if (!segment?.use_vision_reference && !String(segment?.notes || "").trim() && !state.useVrgdgTextContext) {
      return "scene notes are missing.";
    }
    if (!String(segment?.notes || "").trim() && !segment?.use_vision_reference) {
      return "scene notes are missing.";
    }
    return "";
  }

  function sceneConceptPromptText(segment) {
    return String(segment?.t2i_prompt || segment?.flux_prompt || segment?.notes || segment?.flux_notes || "").trim();
  }

  function textOnlyFallbackNotesForSegment(segment, imageMode = state.imageModelMode || "zimage") {
    const parts = [];
    const add = (title, value) => {
      const text = String(value || "").trim();
      if (text) parts.push(`${title}:\n${text}`);
    };
    add("Scene notes", segment?.notes);
    add("Flux/Klein notes", segment?.flux_notes);
    add("NanoBanana notes", segment?.nb_notes);
    if (imageMode === "flux_klein" || imageMode === "nano_banana") {
      const context = imageMode === "nano_banana" ? nbImageSettingsForSegment(segment).reference_context : fluxReferenceContextForSegment(segment);
      add("Reference subject description", context?.subject_description);
      add("Reference location name", context?.location_name);
      add("Reference location description", context?.location_description);
    }
    if (!parts.length) {
      parts.push(`Scene:\n${sceneDisplayName(segment, segmentIndexInfo(segment).index)}`);
      parts.push("Direction:\nCreate a cinematic image prompt that fits this scene.");
    }
    return parts.join("\n\n");
  }

  function buildEmergencyImagePromptForSegment(segment, imageMode = state.imageModelMode || "zimage") {
    const source = textOnlyFallbackNotesForSegment(segment, imageMode)
      .replace(/^[^:\n]{1,48}:\s*/gm, "")
      .replace(/\s+/g, " ")
      .trim();
    const base = source || `cinematic image for ${sceneDisplayName(segment, segmentIndexInfo(segment).index)}`;
    const clipped = base.length > 850 ? `${base.slice(0, 847).trim()}...` : base;
    const prompt = `cinematic image, ${clipped}, visually specific composition, clear subject, atmospheric lighting, high detail`;
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(prompt, segment, imageMode, { validateJunk: false }));
    render();
    return { prompt: segment.t2i_prompt, used_local_notes_fallback: true };
  }

  async function generateTextOnlyImagePromptFallbackForSegment(segment, progress = null, percent = 30, label = "Gemma text-only fallback", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const textModelSelect = imageMode === "ernie_image" ? ernieTextGemmaModelSelect : t2iTextGemmaModelSelect;
    const userNotes = textOnlyFallbackNotesForSegment(segment, imageMode);
    const referenceContext = imageMode === "nano_banana" ? nbImageSettingsForSegment(segment).reference_context : imageMode === "flux_klein" ? fluxReferenceContextForSegment(segment) : {};
    progress?.set(`${label}: creating prompt from notes with non-vision Gemma...\n${gemmaRunnerLine({ vision: false })}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_t2i", {
      ...textGemmaRunnerPayload(),
      model_file: textModelSelect.value,
      mmproj_file: "",
      use_vision: false,
      ref_image_path: "",
      prompt_mode: imageMode,
      reference_context: referenceContext || {},
      repair_model_file: textModelSelect.value,
      user_notes: userNotes,
      theme_style_path: state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: true,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, 120000);
    pushHistory();
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(data.prompt, segment, imageMode, { validateJunk: true }));
    render();
    return { ...data, used_text_only_fallback: true };
  }

  async function generateT2IPromptForSegment(segment, progress = null, percent = 30, label = "Gemma T2I", options = {}) {
    const missing = t2iMissingReason(segment);
    if (missing) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${missing}`);
    state.activeId = segment.id;
    syncInspector();
    const useVision = Boolean(segment.use_vision_reference);
    progress?.set(`${label}: preparing Gemma input...\n${gemmaRunnerLine({ vision: useVision })}`, percent);
    const gemmaSelect = state.imageModelMode === "ernie_image"
      ? (useVision ? ernieGemmaModelSelect : ernieTextGemmaModelSelect)
      : (useVision ? gemmaModelSelect : t2iTextGemmaModelSelect);
    const mmprojSelectForMode = state.imageModelMode === "ernie_image" ? ernieMmprojSelect : mmprojSelect;
    const data = await postJson("/vrgdg/music_builder/generate_t2i", {
      ...textGemmaRunnerPayload(),
      model_file: gemmaSelect.value,
      mmproj_file: useVision ? mmprojSelectForMode.value : "",
      use_vision: useVision,
      ref_image_path: segment.ref_image_path || "",
      repair_model_file: state.imageModelMode === "ernie_image" ? ernieTextGemmaModelSelect.value : t2iTextGemmaModelSelect.value,
      user_notes: segment.notes || "",
      theme_style_path: state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: options.unloadAfter !== false,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, useVision ? 10 * 60 * 1000 : 120000);
    pushHistory();
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(data.prompt, segment, state.imageModelMode, { validateJunk: true }));
    render();
    return data;
  }

  async function createZImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "ZImage") {
    state.activeId = segment.id;
    syncInspector();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "zimage", segment.notes || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: preparing ZImage settings...`, percentBase);
    const zSettings = saveZImageSettingsFromPanel();
    const useLoras = Boolean(zSettings.use_loras && zSettings.lora_count > 0);
    const payload = {
      prompt,
      unet_name: zSettings.unet_name || "",
      clip_name: zSettings.clip_name || "",
      vae_name: zSettings.vae_name || "",
      first_pass_width: zSettings.first_pass_width,
      first_pass_height: zSettings.first_pass_height,
      second_pass_width: zSettings.second_pass_width,
      second_pass_height: zSettings.second_pass_height,
      seed: zSettings.seed,
      seed_mode: zSettings.seed_mode || "fixed",
      batch_size: zSettings.batch_size || 1,
      use_custom_loras: useLoras,
      lora_count: useLoras ? zSettings.lora_count : 0,
      ltx_two_pass_mode: false,
      use_image_to_image: Boolean(zSettings.use_image_to_image),
      image_to_image_start_at_step: zSettings.image_to_image_start_at_step || 5,
      image_to_image_path: zSettings.image_to_image_path || "",
      image_to_image_data: zSettings.image_to_image_data || "",
      image_to_image_name: zSettings.image_to_image_name || "",
    };
    zLoraSlots.forEach((slot, index) => {
      payload[`lora_${index + 1}`] = useLoras && index < zSettings.lora_count ? slot.picker.input.value : "[none]";
      payload[`first_pass_strength_${index + 1}`] = Number(slot.firstPassStrength.value || 0.5);
      payload[`second_pass_strength_${index + 1}`] = Number(slot.secondPassStrength.value || 1);
      payload[`strength_${index + 1}`] = Number(slot.secondPassStrength.value || 1);
    });
    progress?.set(`${label}: building hidden ZImage workflow...`, percentBase + percentSpan * 0.25);
    const built = await postJson("/vrgdg/workflow_runner/build_zimage_prompt", payload);
    if (Number.isFinite(Number(built.used_seed))) {
      zSettings.seed = Number(built.used_seed);
      zSeed.value = String(zSettings.seed);
    }
    progress?.set(`${label}: queueing ZImage workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the preview but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    for (const image of images) {
      await archiveGeneratedSceneImage(segment, image);
    }
    syncSegmentT2IPrompt(segment, prompt);
    segment.image = images[images.length - 1] || null;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    syncPreview(segment);
    render();
    advanceZImageSeedAfterRun(zSettings);
    return images;
  }

  async function previewZImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
    if (!prompt) {
      toast("Hey, you need a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or add scene notes.", true);
      return;
    }
    let progress = null;
    let ranZImage = false;
    try {
      setButtonGroupState(zCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating ZImage preview");
      progress.set("Autosaving session/SRT before ZImage...", 8);
      await autoSaveSessionQuiet("ZImage preview");
      ranZImage = true;
      await createZImageForSegment(segment, progress, 15, 75, "ZImage preview");
      await autoSaveSessionQuiet("ZImage preview complete");
      await runImageMemoryCleanupQuiet(progress, "ZImage preview", 94);
      progress.set("ZImage preview ready.", 100);
      progress.close(900);
      toast("ZImage preview ready.");
    } catch (error) {
      if (ranZImage) {
        await runImageMemoryCleanupQuiet(progress, "failed ZImage preview", 100);
      }
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(zCreateButtons, { disabled: false, text: "Create Z-Image" });
    }
  }

  async function createErnieImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Ernie") {
    state.activeId = segment.id;
    syncInspector();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "ernie_image", segment.notes || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: preparing Ernie settings...`, percentBase);
    const settings = saveErnieImageSettingsFromPanel();
    const useLoras = Boolean(settings.use_loras && settings.lora_count > 0);
    const payload = {
      prompt,
      unet_name: settings.unet_name || "",
      clip_name: settings.clip_name || "",
      vae_name: settings.vae_name || "",
      width: settings.width,
      height: settings.height,
      seed: settings.seed,
      seed_mode: settings.seed_mode || "fixed",
      batch_size: settings.batch_size || 1,
      use_custom_loras: useLoras,
      lora_count: useLoras ? settings.lora_count : 0,
      use_image_to_image: Boolean(settings.use_image_to_image),
      image_to_image_start_at_step: settings.image_to_image_start_at_step || 5,
      image_to_image_path: settings.image_to_image_path || "",
      image_to_image_data: settings.image_to_image_data || "",
      image_to_image_name: settings.image_to_image_name || "",
    };
    ernieLoraSlots.forEach((slot, index) => {
      payload[`lora_${index + 1}`] = useLoras && index < settings.lora_count ? slot.picker.input.value : "[none]";
      payload[`strength_${index + 1}`] = Number(slot.strength.value || 1);
    });
    progress?.set(`${label}: building hidden Ernie workflow...`, percentBase + percentSpan * 0.25);
    let built;
    try {
      built = await postJson("/vrgdg/workflow_runner/build_ernie_image_prompt", payload);
    } catch (error) {
      if (/\b405\b/.test(String(error?.message || error))) {
        throw new Error("Ernie backend route is not loaded yet. Fully restart ComfyUI so the new Ernie workflow route is registered, then refresh the browser.");
      }
      throw error;
    }
    if (Number.isFinite(Number(built.used_seed))) {
      settings.seed = Number(built.used_seed);
      ernieSeed.value = String(settings.seed);
    }
    progress?.set(`${label}: queueing Ernie workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the Ernie image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    for (const image of images) {
      await archiveGeneratedSceneImage(segment, image);
    }
    segment.enhance_prompt = prompt;
    zEnhancePromptPreview.value = prompt;
    segment.image = images[images.length - 1] || null;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    syncPreview(segment);
    render();
    advanceErnieSeedAfterRun(settings);
    return images;
  }

  async function previewErnieImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
    if (!prompt) {
      toast("Hey, you need a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or add scene notes.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(ernieCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating Ernie image");
      progress.set("Autosaving session/SRT before Ernie...", 8);
      await autoSaveSessionQuiet("Ernie image");
      await createErnieImageForSegment(segment, progress, 15, 75, "Ernie image");
      await autoSaveSessionQuiet("Ernie image complete");
      progress.set("Ernie image ready.", 100);
      progress.close(900);
      toast("Ernie image ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(ernieCreateButtons, { disabled: false, text: "Create with Ernie" });
    }
  }

  async function createFluxKleinPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveFluxKleinSettingsFromPanel();
    if (!settings.use_text_only_gemma_prompt && (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length)) {
      toast("Load at least one image ingredient first.", true);
      return;
    }
    let progress = null;
    try {
      createFluxPromptButton.disabled = true;
      createFluxPromptButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating Flux/Klein prompt");
      progress.set("Autosaving session/SRT before Gemma Flux/Klein...", 8);
      await autoSaveSessionQuiet("Gemma Flux/Klein prompt");
      const data = await generateFluxKleinPromptForSegment(segment, progress, 25, "Gemma Flux/Klein", { unloadAfter: true });
      progress.set("Flux/Klein prompt ready.", 100);
      await autoSaveSessionQuiet("Gemma Flux/Klein prompt complete");
      progress.close(900);
      render();
      toast("Gemma created the Flux/Klein prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createFluxPromptButton.disabled = false;
      createFluxPromptButton.textContent = "Gemma Flux Prompt";
    }
  }

  function fluxKleinSettingsForSegment(segment = activeSegment()) {
    const current = segment?.use_scene_flux_klein_settings ? (segment.flux_klein_settings || cloneFluxKleinSettings(state.fluxKleinSettings)) : (state.fluxKleinSettings || {});
    return {
      ...current,
      image_ingredients: mergedFluxImageIngredients(segment),
      notes: segment?.flux_notes || "",
      prompt: segment?.flux_prompt || "",
    };
  }

  function fluxKleinLoraPayload(settings = {}) {
    const count = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
    const useLoras = Boolean(settings.use_loras && count > 0);
    const payload = {
      use_custom_loras: useLoras,
      lora_count: useLoras ? count : 0,
    };
    for (let slot = 1; slot <= 4; slot++) {
      const config = settings.loras?.[slot - 1] || {};
      payload[`lora_${slot}`] = config.name || "[none]";
      payload[`strength_${slot}`] = Number(config.strength ?? 1);
    }
    return payload;
  }

  async function generateFluxKleinPromptForSegment(segment, progress = null, percent = 25, label = "Flux/Klein Gemma", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = fluxKleinSettingsForSegment(segment);
    if (settings.use_text_only_gemma_prompt) {
      return await generateTextOnlyImagePromptFallbackForSegment(segment, progress, percent, `${label}: text-only Gemma`, { imageMode: "flux_klein" });
    }
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: add at least one global or scene Flux/Klein image ingredient.`);
    }
    progress?.set(`${label}: combining global and scene image ingredients for Gemma vision...\n${gemmaRunnerLine({ vision: true })}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_flux_klein_prompt", {
      model_file: fluxGemmaModelSelect.value,
      mmproj_file: fluxMmprojSelect.value,
      image_ingredients: settings.image_ingredients || [],
      reference_context: settings.reference_context || {},
      repair_model_file: t2iTextGemmaModelSelect.value,
      user_notes: settings.notes || segment.notes || "",
      clear_before_load: options.clearBeforeLoad !== false,
      unload_after: options.unloadAfter !== false,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, FLUX_GEMMA_TIMEOUT_MS);
    pushHistory();
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(data.prompt, segment, "flux_klein", { validateJunk: true }));
    render();
    return data;
  }

  async function createFluxKleinImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Flux/Klein") {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = fluxKleinSettingsForSegment(segment);
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "flux_klein", settings.prompt || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Flux/Klein prompt is missing.`);
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: add at least one global or scene Flux/Klein image ingredient.`);
    }
    progress?.set(`${label}: building hidden Flux/Klein workflow...`, percentBase + percentSpan * 0.25);
    const built = await postJson("/vrgdg/workflow_runner/build_flux_klein_prompt", {
      prompt,
      image_ingredients: settings.image_ingredients || [],
      unet_name: settings.unet_name || "",
      clip_name: settings.clip_name || "",
      vae_name: settings.vae_name || "",
      width: settings.width || 1024,
      height: settings.height || 576,
      seed: settings.seed || 100,
      ...fluxKleinLoraPayload(settings),
    });
    progress?.set(`${label}: queueing Flux/Klein workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the Flux/Klein image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    pushHistory();
    segment.image = images[images.length - 1] || null;
    await archiveGeneratedSceneImage(segment, segment.image);
    syncSegmentT2IPrompt(segment, prompt);
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) {
      syncPreview(segment);
    }
    render();
    return images;
  }

  function nbImageSettingsForSegment(segment = activeSegment()) {
    const current = segment?.use_scene_nb_image_settings
      ? (segment.nb_image_settings || cloneNBImageSettings(state.nbImageSettings))
      : (state.nbImageSettings || {});
    return {
      ...cloneNBImageSettings(current),
      image_ingredients: mergedFluxImageIngredients(segment),
      notes: segment?.nb_notes || segment?.flux_notes || segment?.notes || "",
      prompt: segment?.nb_prompt || segment?.t2i_prompt || "",
      reference_context: fluxReferenceContextForSegment(segment),
    };
  }

  async function createNBPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveNBImageSettingsFromPanel();
    if (!settings.use_text_only_gemma_prompt && (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length)) {
      toast("Load at least one NanoBanana reference image first.", true);
      return;
    }
    let progress = null;
    try {
      createNBPromptButton.disabled = true;
      createNBPromptButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating NanoBanana prompt");
      progress.set("Autosaving session/SRT before Gemma NanoBanana...", 8);
      await autoSaveSessionQuiet("Gemma NanoBanana prompt");
      const data = await generateNBPromptForSegment(segment, progress, 25, "Gemma NanoBanana", { unloadAfter: true });
      progress.set("NanoBanana prompt ready.", 100);
      await autoSaveSessionQuiet("Gemma NanoBanana prompt complete");
      progress.close(900);
      render();
      toast("Gemma created the NanoBanana prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createNBPromptButton.disabled = false;
      createNBPromptButton.textContent = "Gemma NB Prompt";
    }
  }

  async function generateNBPromptForSegment(segment, progress = null, percent = 25, label = "NanoBanana Gemma", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = nbImageSettingsForSegment(segment);
    if (settings.use_text_only_gemma_prompt) {
      return await generateTextOnlyImagePromptFallbackForSegment(segment, progress, percent, `${label}: text-only Gemma`, { imageMode: "nano_banana" });
    }
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: add at least one NanoBanana reference image.`);
    }
    progress?.set(`${label}: creating structured NanoBanana prompt from reference images...\n${gemmaRunnerLine({ vision: true })}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_nb_image_prompt", {
      ...textGemmaRunnerPayload(),
      model_file: nbGemmaModelSelect.value || fluxGemmaModelSelect.value,
      mmproj_file: nbMmprojSelect.value || fluxMmprojSelect.value,
      lmstudio_base_url: state.lmStudioBaseUrl || "",
      lmstudio_model: state.lmStudioModel || "",
      lmstudio_api_key: state.lmStudioApiKey || "",
      image_ingredients: settings.image_ingredients || [],
      reference_context: {},
      repair_model_file: t2iTextGemmaModelSelect.value,
      user_notes: settings.notes || segment.notes || "",
      clear_before_load: options.clearBeforeLoad !== false,
      unload_after: options.unloadAfter !== false,
      n_ctx: 8000,
      max_new_tokens: 900,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, 180000);
    pushHistory();
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(data.prompt, segment, "nano_banana", { validateJunk: true }));
    render();
    return data;
  }

  async function createNBImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "NanoBanana") {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = nbImageSettingsForSegment(segment);
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "nano_banana", settings.prompt || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: NanoBanana prompt is missing.`);
    if (!String(settings.api_key || "").trim()) throw new Error("NanoBanana API key is missing.");
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: add at least one NanoBanana reference image.`);
    }
    progress?.set(`${label}: building hidden NanoBanana workflow...`, percentBase + percentSpan * 0.25);
    const built = await postJson("/vrgdg/workflow_runner/build_nb_image_prompt", {
      api_key: settings.api_key || "",
      model: settings.model || DEFAULT_NB_IMAGE_MODEL,
      prompt,
      image_ingredients: settings.image_ingredients || [],
    });
    progress?.set(`${label}: queueing NanoBanana workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the NanoBanana image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    pushHistory();
    segment.image = images[images.length - 1] || null;
    await archiveGeneratedSceneImage(segment, segment.image);
    syncSegmentT2IPrompt(segment, prompt);
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) {
      syncPreview(segment);
    }
    render();
    return images;
  }

  async function previewNBImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveNBImageSettingsFromPanel();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "nano_banana", settings.prompt || nbPrompt.value || "");
    if (!prompt) {
      toast("Hey, you need a NanoBanana prompt first. Click Gemma NB Prompt or type one into the NanoBanana prompt box.", true);
      return;
    }
    if (!String(settings.api_key || "").trim()) {
      toast("NanoBanana API key is missing.", true);
      return;
    }
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      toast("Hey, you need at least one NanoBanana reference image first.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(nbCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating NanoBanana image");
      progress.set("Autosaving session/SRT before NanoBanana image...", 8);
      await autoSaveSessionQuiet("NanoBanana image");
      await createNBImageForSegment(segment, progress, 15, 75, "NanoBanana image");
      await autoSaveSessionQuiet("NanoBanana image complete");
      progress.set("NanoBanana image ready.", 100);
      progress.close(900);
      toast("NanoBanana image preview ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(nbCreateButtons, { disabled: false, text: "Create with NanoBanana" });
    }
  }

  async function previewFluxKleinImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveFluxKleinSettingsFromPanel();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "flux_klein", settings.prompt || fluxPrompt.value || "");
    if (!prompt) {
      toast("Hey, you need a Flux/Klein prompt first. Click Gemma Flux Prompt or type one into the Flux/Klein prompt box.", true);
      return;
    }
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      toast("Hey, you need at least one image ingredient first. Load or drop character, background, or reference images.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(fluxCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating Flux/Klein image");
      progress.set("Autosaving session/SRT before Flux/Klein image...", 8);
      await autoSaveSessionQuiet("Flux/Klein image");
      progress.set("Building hidden Flux/Klein workflow...", 30);
      const built = await postJson("/vrgdg/workflow_runner/build_flux_klein_prompt", {
        prompt,
        image_ingredients: settings.image_ingredients || [],
        unet_name: settings.unet_name || "",
        clip_name: settings.clip_name || "",
        vae_name: settings.vae_name || "",
        width: settings.width || 1024,
        height: settings.height || 576,
        seed: settings.seed || 100,
        ...fluxKleinLoraPayload(settings),
      });
      progress.set("Queueing Flux/Klein workflow...", 50);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the Flux/Klein image but did not return a prompt_id.");
      progress.set(`Queued prompt ID:\n${promptId}\n\nWaiting for image...`, 65);
      const images = await waitForImages(promptId, (message) => progress.set(`${message}\nPrompt ID: ${promptId}`, 80));
      pushHistory();
      segment.image = images[images.length - 1] || null;
      await archiveGeneratedSceneImage(segment, segment.image);
      syncSegmentT2IPrompt(segment, prompt);
      segment.custom_image_path = "";
      segment.custom_image_data = "";
      segment.custom_image_name = "";
      segment.approved_image_path = "";
      segment.preview_mode = "image";
      syncPreview(segment);
      render();
      await autoSaveSessionQuiet("Flux/Klein image complete");
      progress.set("Flux/Klein preview ready.", 100);
      progress.close(900);
      toast("Flux/Klein image preview ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(fluxCreateButtons, { disabled: false, text: "Create with Flux/Klein" });
    }
  }

  function currentEnhanceSource(segment) {
    const source = segmentImageSource(segment);
    if (source?.path || source?.data) return source;
    return null;
  }

  async function generateEnhancePromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    let source = currentEnhanceSource(segment);
    if (!source?.path && !source?.data && segment.image?.filename) {
      const archived = await archiveGeneratedSceneImage(segment, segment.image);
      if (archived) source = { path: archived, name: "scene_image.png" };
    }
    if (!source?.path && !source?.data) {
      toast("Hey, load or create an image first so Gemma can see what to enhance.", true);
      return;
    }
    const modelFile = String(zEnhanceGemmaModelSelect.value || "").trim();
    const mmprojFile = String(zEnhanceMmprojSelect.value || "").trim();
    if (!modelFile || !mmprojFile) {
      toast("Choose the Enhance vision Gemma model and mmproj in the Enhance Models tab first.", true);
      return;
    }
    const notes = String(zEnhanceGemmaNotes.value || "").trim();
    let progress = null;
    try {
      zEnhanceGemmaButton.disabled = true;
      zEnhanceGemmaButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating Enhance prompt");
      progress.set(`Reading selected image with Gemma vision...\n${gemmaRunnerLine({ vision: true })}`, 20);
      const userNotes = [
        "Create a text-to-image prompt for image-to-image upscale/enhance.",
        "Describe the visible subject, setting, lighting, clothing, pose, composition, and mood from the image.",
        "Preserve the important identity and scene details from the image.",
        notes ? `User enhancement notes:\n${notes}` : "User enhancement notes:\nKeep the image identity, improve cinematic detail, clarity, lighting, and polish.",
      ].join("\n\n");
      const data = await postJson("/vrgdg/music_builder/generate_t2i", {
        model_file: modelFile,
        mmproj_file: mmprojFile,
        use_vision: true,
        ref_image_path: source.path || "",
        ref_image_data: source.data || "",
        user_notes: userNotes,
        unload_after: true,
        max_new_tokens: 1000,
      }, 10 * 60 * 1000);
      pushHistory();
      segment.enhance_prompt = applyTriggerPhrase(data.prompt, state.imageTriggerPhrase);
      zEnhancePromptPreview.value = segment.enhance_prompt;
      progress.set("Enhance prompt ready.", 100);
      progress.close(900);
      render();
      await autoSaveSessionQuiet("Gemma enhance prompt");
      toast("Gemma created the Enhance prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      zEnhanceGemmaButton.disabled = false;
      zEnhanceGemmaButton.textContent = "Gemma Enhance Prompt";
    }
  }

  async function upscaleEnhanceImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    let source = currentEnhanceSource(segment);
    if (!source?.path && !source?.data && segment.image?.filename) {
      const archived = await archiveGeneratedSceneImage(segment, segment.image);
      if (archived) source = { path: archived };
    }
    if (!source?.path && !source?.data) {
      toast("Hey, you need an image first. Create, save, load, or choose a scene image before using upscale/enhance.", true);
      return;
    }
    const enhancePromptInfo = activeScenePromptForEnhance({ copyFallback: true });
    const enhancePrompt = enhancePromptInfo.prompt;
    if (!enhancePrompt) {
      toast("Hey, this scene needs a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or create a Flux/Klein prompt.", true);
      return;
    }
    const settings = saveZEnhanceSettingsFromPanel();
    const useLoras = Boolean(settings.use_loras && settings.lora_count > 0);
    let progress = null;
    try {
      zEnhanceButton.disabled = true;
      zEnhanceButton.textContent = "Enhancing...";
      if (source.path) addSceneImageHistoryPath(segment, source.path);
      progress = createProgressWindow("Upscale / Enhance image");
      progress.set("Autosaving session/SRT before Enhance...", 8);
      await autoSaveSessionQuiet("upscale/enhance");
      progress.set("Building hidden upscale/enhance workflow...", 25);
      const payload = {
        prompt: enhancePrompt,
        source_image_path: source.path || "",
        source_image_data: source.data || "",
        source_image_name: source.name || "source.png",
        unet_name: settings.unet_name || "",
        clip_name: settings.clip_name || "",
        vae_name: settings.vae_name || "",
        width: settings.width || 1920,
        height: settings.height || 1080,
        seed: settings.seed || 1,
        seed_mode: settings.seed_mode || "fixed",
        enhance_amount: settings.enhance_amount || 8,
        use_custom_loras: useLoras,
        lora_count: useLoras ? settings.lora_count : 0,
      };
      zEnhanceLoraSlots.forEach((slot, index) => {
        payload[`lora_${index + 1}`] = useLoras && index < settings.lora_count ? slot.picker.input.value : "[none]";
        payload[`strength_${index + 1}`] = Number(slot.strength.value || 1);
      });
      const built = await postJson("/vrgdg/workflow_runner/build_z_upscale_enhance_prompt", payload);
      if (Number.isFinite(Number(built.used_seed))) {
        settings.seed = Number(built.used_seed);
        zEnhanceSeed.value = String(settings.seed);
      }
      progress.set("Queueing upscale/enhance workflow...", 45);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the upscale/enhance workflow but did not return a prompt_id.");
      progress.set(`Queued prompt ID:\n${promptId}\n\nWaiting for enhanced image...`, 65);
      const images = await waitForImages(promptId, (message) => progress.set(`${message}\nPrompt ID: ${promptId}`, 80));
      pushHistory();
      for (const image of images) {
        await archiveGeneratedSceneImage(segment, image);
      }
      segment.image = images[images.length - 1] || null;
      segment.custom_image_path = "";
      segment.custom_image_data = "";
      segment.custom_image_name = "";
      segment.approved_image_path = "";
      segment.preview_mode = "image";
      syncPreview(segment);
      render();
      advanceZEnhanceSeedAfterRun(settings);
      await autoSaveSessionQuiet("upscale/enhance complete");
      progress.set("Enhanced image ready.", 100);
      progress.close(900);
      toast("Upscale/enhance image ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      zEnhanceButton.disabled = false;
      zEnhanceButton.textContent = "Upscale / Enhance Image";
    }
  }

  async function createT2IPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const missing = t2iMissingReason(segment);
    if (missing) {
      toast(`Hey, ${missing}`, true);
      return;
    }
    let progress = null;
    try {
      createT2IButton.disabled = true;
      createT2IButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating T2I prompt");
      progress.set("Autosaving session/SRT before Gemma T2I...", 8);
      await autoSaveSessionQuiet("Gemma T2I");
      const data = await generateT2IPromptForSegment(segment, progress, 45, segment.use_vision_reference ? "Gemma with reference image" : "Gemma from notes");
      await autoSaveSessionQuiet("Gemma T2I complete");
      progress.set("T2I prompt ready.", 100);
      progress.close(900);
      toast(data.used_reference_image ? "Gemma created T2I from reference image." : "Gemma created T2I from notes.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createT2IButton.disabled = false;
      createT2IButton.textContent = "Gemma T2I";
    }
  }

  function getI2VImageReference(segment) {
    if (!segment) return { path: "", data: "" };
    const source = segmentImageSource(segment);
    if (source?.data) return { path: "", data: source.data };
    if (source?.path) return { path: source.path, data: "" };
    return { path: "", data: "" };
  }

  async function createI2VPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const isT2V = currentVideoMode() === "t2v";
    const modeLabel = isT2V ? "T2V" : "I2V";
    const useImageReference = isT2V ? Boolean(segment.use_t2v_vision_reference) : segment.use_i2v_vision_reference !== false;
    const imageReference = useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
    const conceptPrompt = sceneConceptPromptText(segment);
    if (useImageReference && !imageReference.path && !imageReference.data) {
      toast(isT2V
        ? "Hey, T2V Gemma image reference is on, but no reference image is loaded. Drop/load a reference image or turn it off."
        : "Hey, you need a scene image first. Save/load an image, or turn off image reference to create I2V from the T2I prompt instead.", true);
      return;
    }
    if ((isT2V || !useImageReference) && !conceptPrompt) {
      toast(isT2V
        ? "Hey, you need a T2I/concept prompt first so Gemma has scene content to turn into a text-to-video prompt."
        : "Hey, you need a T2I prompt first. Create/type one, or turn image reference back on and use a saved/custom image.", true);
      return;
    }
    let progress = null;
    try {
      createI2VButton.disabled = true;
      createI2VButton.textContent = "Gemma...";
      progress = createProgressWindow(`Creating ${modeLabel} prompt`);
      progress.set(`Autosaving session/SRT before Gemma ${modeLabel}...`, 8);
      await autoSaveSessionQuiet(`Gemma ${modeLabel}`);
      progress.set(useImageReference ? "Preparing image reference and motion notes..." : "Preparing T2I prompt and motion notes...", 20);
      progress.set(useImageReference
        ? `Running Gemma vision ${modeLabel} prompt generation...\n${gemmaRunnerLine({ vision: true })}`
        : `Running Gemma text-only ${modeLabel} prompt generation...\n${gemmaRunnerLine()}`, 50);
      const data = await postJson(isT2V ? "/vrgdg/music_builder/generate_t2v" : "/vrgdg/music_builder/generate_i2v", {
        ...textGemmaRunnerPayload(),
        model_file: useImageReference ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
        mmproj_file: useImageReference ? i2vMmprojSelect.value : "",
        t2i_prompt: isT2V ? conceptPrompt : useImageReference ? "" : conceptPrompt,
        image_reference_path: imageReference.path,
        image_reference_data: imageReference.data,
        repair_model_file: i2vTextGemmaModelSelect.value,
        user_notes: videoGemmaNotesForSegment(segment),
        theme_style_path: useImageReference && !isT2V ? "" : state.useVrgdgTextContext ? state.themeStylePath || "" : "",
        story_idea_path: useImageReference && !isT2V ? "" : state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
        subject_scene_path: useImageReference && !isT2V ? "" : state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
        unload_after: true,
      });
      pushHistory();
      segment.i2v_prompt = applyTriggerPhrase(data.prompt, videoTriggerPhraseForSegment(segment));
      i2vPrompt.value = segment.i2v_prompt;
      render();
      await autoSaveSessionQuiet(`Gemma ${modeLabel} complete`);
      progress.set(`${modeLabel} prompt ready.`, 100);
      progress.close(900);
      toast(data.used_image_reference ? "Gemma created I2V prompt from the image reference." : `Gemma created ${modeLabel} prompt from the T2I prompt.`);
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createI2VButton.disabled = false;
      syncVideoModePanel();
    }
  }

  async function generateTextOnlyI2VPromptForSegment(segment, progress = null, percent = 50, label = "Gemma I2V", options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    const t2iText = String(segment.t2i_prompt || "").trim();
    if (!t2iText) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: converting T2I prompt to I2V prompt without vision...\n${gemmaRunnerLine()}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_i2v", {
      ...textGemmaRunnerPayload(),
      model_file: i2vTextGemmaModelSelect.value,
      mmproj_file: "",
      t2i_prompt: t2iText,
      image_reference_path: "",
      image_reference_data: "",
      repair_model_file: i2vTextGemmaModelSelect.value,
      user_notes: videoGemmaNotesForSegment(segment),
      theme_style_path: state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: options.unloadAfter !== false,
    });
    pushHistory();
    segment.i2v_prompt = applyTriggerPhrase(data.prompt, videoTriggerPhraseForSegment(segment));
    if (!segment.i2v_prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty I2V prompt.`);
    if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt;
    render();
    return data;
  }

  async function generateI2VPromptForSegment(segment, progress = null, percent = 50, label = "Gemma I2V", options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    const isT2V = currentVideoMode() === "t2v";
    const modeLabel = isT2V ? "T2V" : "I2V";
    const forceTextOnly = Boolean(options.forceTextOnly);
    const forceVision = Boolean(options.forceVision);
    const useImageReference = forceVision ? true : forceTextOnly ? false : (isT2V ? Boolean(segment.use_t2v_vision_reference) : segment.use_i2v_vision_reference !== false);
    const imageReference = useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
    const t2iText = sceneConceptPromptText(segment);
    if (useImageReference && !imageReference.path && !imageReference.data) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${modeLabel} image reference is enabled, but no reference image was found.`);
    }
    if ((isT2V || !useImageReference) && !t2iText) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I/concept prompt is missing.`);
    }
    progress?.set(useImageReference
      ? `${label}: creating ${modeLabel} prompt from reference image, concept, and motion notes...\n${gemmaRunnerLine({ vision: true })}`
      : `${label}: converting T2I prompt to ${modeLabel} prompt without vision...\n${gemmaRunnerLine()}`, percent);
    const data = await postJson(isT2V ? "/vrgdg/music_builder/generate_t2v" : "/vrgdg/music_builder/generate_i2v", {
      ...textGemmaRunnerPayload(),
      model_file: useImageReference ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
      mmproj_file: useImageReference ? i2vMmprojSelect.value : "",
      t2i_prompt: isT2V ? t2iText : useImageReference ? "" : t2iText,
      image_reference_path: imageReference.path,
      image_reference_data: imageReference.data,
      repair_model_file: i2vTextGemmaModelSelect.value,
      user_notes: videoGemmaNotesForSegment(segment),
      theme_style_path: useImageReference && !isT2V ? "" : state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: useImageReference && !isT2V ? "" : state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: useImageReference && !isT2V ? "" : state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: options.unloadAfter !== false,
    });
    pushHistory();
    segment.i2v_prompt = applyTriggerPhrase(data.prompt, videoTriggerPhraseForSegment(segment));
    if (!segment.i2v_prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty ${modeLabel} prompt.`);
    if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt;
    render();
    return data;
  }

  async function i2vAllScenes(options = {}) {
    const isT2V = currentVideoMode() === "t2v";
    const modeLabel = isT2V ? "T2V" : "I2V";
    const progress = options.progress || createProgressWindow(`Gemma ${modeLabel} All Scenes`);
    const closeProgress = !options.progress;
    const allScenes = allEditableSegments();
    const redoPrompts = options.i2vRunMode === "redo_prompts";
    const forceTextOnly = Boolean(options.forceTextOnly);
    const forceVision = Boolean(options.forceVision);
    if (redoPrompts) {
      allScenes.forEach((segment) => {
        segment.i2v_prompt = "";
      });
    }
    const scenes = allScenes.filter((segment) => redoPrompts || !String(segment?.i2v_prompt || "").trim());
    const missing = [];
    if (!allScenes.length) missing.push("No scenes found. Add or load scenes first.");
    scenes.forEach((segment) => {
      const index = segmentIndexInfo(segment).index;
      const useImageReference = forceVision ? true : forceTextOnly ? false : videoVisionReferenceEnabled(segment);
      const imageReference = useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
      if (useImageReference && !imageReference.path && !imageReference.data) {
        missing.push(`${sceneDisplayName(segment, index)}: ${modeLabel} image reference is enabled, but no reference image was found.`);
      }
      if ((isT2V || !useImageReference) && !sceneConceptPromptText(segment)) {
        missing.push(`${sceneDisplayName(segment, index)}: T2I/concept prompt is missing.`);
      }
    });
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Gemma ${escapeHtml(modeLabel)} All cannot start yet.</div>
          <div>Fix these first:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      createI2VButton.disabled = true;
      progress.set(`Autosaving session/SRT before Gemma ${modeLabel} All...`, 3);
      await saveSessionForSceneVideo();
      if (!scenes.length) {
        progress.set(`All scenes already have ${modeLabel} prompts. Skipping Gemma ${modeLabel} All.`, 100);
        if (closeProgress) progress.close(1800);
        toast(`All scenes already have ${modeLabel} prompts. Gemma ${modeLabel} skipped.`);
        return;
      }
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const segment = scenes[index];
        state.activeId = segment.id;
        syncInspector();
        render();
        const base = Math.floor((index / scenes.length) * 100);
        const useImageReference = forceVision ? true : forceTextOnly ? false : videoVisionReferenceEnabled(segment);
        const displayIndex = segmentIndexInfo(segment).index;
        progress.set(`Gemma ${modeLabel} All ${index + 1}/${scenes.length}: ${sceneDisplayName(segment, displayIndex)}\nBatch mode: ${forceVision ? "vision" : forceTextOnly ? "text only" : "scene checkbox"}\n${useImageReference ? "Using image reference plus T2I prompt/motion notes." : "Using T2I prompt text only."}`, base);
        await generateI2VPromptForSegment(segment, progress, Math.min(98, base + 30), `Gemma ${modeLabel} All ${index + 1}/${scenes.length}`, { unloadAfter: false, forceTextOnly, forceVision });
        await autoSaveSessionQuiet(`Gemma ${modeLabel} All ${sceneDisplayName(segment, displayIndex)}`);
      }
      await runClearMemoryWorkflowQuiet(progress, `Gemma ${modeLabel} prompt pass`, 96);
      await autoSaveSessionQuiet(`Gemma ${modeLabel} All complete`);
      progress.set(`Gemma ${modeLabel} All complete.`, 100);
      if (closeProgress) progress.close(1800);
    } catch (error) {
      progress.set(`Stopped/Error:\n${String(error?.message || error)}`, 100);
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
    } finally {
      createI2VButton.disabled = false;
    }
  }

  function promptAllModeTargets(promptRunMode = "missing_only", imageMode = state.imageModelMode || "zimage") {
    const scenes = allEditableSegments().map((segment) => ({ segment, index: segmentIndexInfo(segment).index }));
    if (promptRunMode === "redo_all") return scenes;
    return scenes.filter(({ segment }) => {
      if (imageMode === "flux_klein") return !String(segment.flux_prompt || segment.t2i_prompt || "").trim();
      if (imageMode === "nano_banana") return !String(segment.nb_prompt || segment.t2i_prompt || "").trim();
      return !String(segment.t2i_prompt || "").trim();
    });
  }

  async function gemmaT2IAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageMode = state.imageModelMode || "zimage";
    const modelLabel = imageMode === "flux_klein" ? "Flux/Klein" : imageMode === "nano_banana" ? "NanoBanana" : imageMode === "ernie_image" ? "Ernie" : "ZImage";
    const promptRunMode = options.promptRunMode || "redo_all";
    const redoPrompts = promptRunMode === "redo_all";
    const allScenes = allEditableSegments().map((segment) => ({ segment, index: segmentIndexInfo(segment).index }));
    const targetScenes = promptAllModeTargets(promptRunMode, imageMode);
    const progress = createProgressWindow(`Gemma T2I All (${modelLabel})`);
    const missing = [];
    if (!allScenes.length) missing.push("No scenes found. Add or load scenes first.");
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    targetScenes.forEach(({ segment, index }) => {
      if (imageMode === "flux_klein") {
        const settings = fluxKleinSettingsForSegment(segment);
        const ingredients = settings.image_ingredients || [];
        if (!settings.use_text_only_gemma_prompt && (!Array.isArray(ingredients) || !ingredients.length)) {
          missing.push(`${sceneDisplayName(segment, index)}: add at least one global or scene Flux/Klein image ingredient.`);
        }
      } else if (imageMode === "nano_banana") {
        const settings = nbImageSettingsForSegment(segment);
        const ingredients = settings.image_ingredients || [];
        if (!settings.use_text_only_gemma_prompt && (!Array.isArray(ingredients) || !ingredients.length)) {
          missing.push(`${sceneDisplayName(segment, index)}: add at least one NanoBanana reference image.`);
        }
      } else {
        const reason = t2iMissingReason(segment);
        if (reason) missing.push(`${sceneDisplayName(segment, index)}: ${reason}`);
      }
    });
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Gemma T2I All cannot start yet.</div>
          <div>Fix these first:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Gemma T2I All needs scene inputs first.", true);
      return;
    }
    try {
      state.batchCancelled = false;
      gemmaT2IAllButton.disabled = true;
      zImageAllButton.disabled = true;
      createT2IButton.disabled = true;
      ernieCreateT2IButton.disabled = true;
      createFluxPromptButton.disabled = true;
      progress.set("Autosaving session/SRT before Gemma T2I All...", 3);
      await saveSessionForSceneVideo();
      if (redoPrompts) {
        targetScenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = redoPrompts ? targetScenes : promptAllModeTargets("missing_only", imageMode);
      if (!promptScenes.length) {
        progress.set("All scenes already have T2I prompts. Nothing to do.", 100);
        progress.close(1800);
        toast("All scenes already have T2I prompts.");
        return;
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 5 + Math.floor((index / promptScenes.length) * 88);
        state.activeId = segment.id;
        syncInspector();
        render();
        const promptLabel = `Gemma T2I All ${index + 1}/${promptScenes.length}: ${sceneLabel}`;
        if (imageMode === "flux_klein") {
          await runGemmaImagePromptPassWithRetry(segment, progress, base, promptLabel, generateFluxKleinPromptForSegment, {
            clearBeforeLoad: index === 0,
            unloadAfter: false,
            generatorOptions: {},
          });
        } else if (imageMode === "nano_banana") {
          await runGemmaImagePromptPassWithRetry(segment, progress, base, promptLabel, generateNBPromptForSegment, {
            clearBeforeLoad: index === 0,
            unloadAfter: false,
            generatorOptions: {},
          });
        } else {
          await runGemmaImagePromptPassWithRetry(segment, progress, base, promptLabel, generateT2IPromptForSegment, {
            unloadAfter: false,
          });
        }
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Gemma T2I All ${sceneLabel}`);
      }
      await runClearMemoryWorkflowQuiet(progress, "Gemma T2I prompt pass", 96);
      await autoSaveSessionQuiet("Gemma T2I All complete");
      progress.set("Gemma T2I All complete. Review or edit the image prompts before creating images.", 100);
      progress.close(2500);
      toast("Gemma T2I All complete.");
    } catch (error) {
      const message = String(error?.message || error);
      const stopped = /stopped by user/i.test(message);
      progress.set(`${stopped ? "Stopped" : "Error"}:\n${message}\n\nRunning memory cleanup...`, 100);
      toast(message, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Gemma T2I All" : "Gemma T2I All error", 100);
        progress.set(`${stopped ? "Stopped" : "Error"}:\n${message}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Gemma T2I All failed:", cleanupError);
      }
    } finally {
      gemmaT2IAllButton.disabled = false;
      zImageAllButton.disabled = false;
      createT2IButton.disabled = false;
      ernieCreateT2IButton.disabled = false;
      createFluxPromptButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function gemmaVideoAllTextOnly(options = {}) {
    updateActiveFromInputs();
    gemmaVideoAllButton.disabled = true;
    const changedReferenceFlags = [];
    try {
      if (options.gemmaInputMode === "vision") {
        allEditableSegments().forEach((segment) => {
          const previous = videoVisionReferenceEnabled(segment);
          if (!previous) {
            changedReferenceFlags.push({ segment, previous });
            setVideoVisionReferenceEnabled(segment, true);
          }
        });
      }
      await i2vAllScenes({
        i2vRunMode: options.promptRunMode === "missing_only" ? "missing_only" : "redo_prompts",
        forceTextOnly: options.gemmaInputMode !== "vision",
        forceVision: options.gemmaInputMode === "vision",
      });
    } finally {
      changedReferenceFlags.forEach(({ segment, previous }) => setVideoVisionReferenceEnabled(segment, previous));
      gemmaVideoAllButton.disabled = false;
    }
  }

  function i2vImagesFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\zimage_approved`;
  }

  function i2vVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\image_to_video_clips`;
  }

  function t2vVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\text_to_video_clips`;
  }

  function activeVideoOutputFolder(mode = currentVideoMode()) {
    return mode === "t2v" ? t2vVideoOutputFolder() : i2vVideoOutputFolder();
  }

  function collectedSceneVideoFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\rendered_scene_videos`;
  }

  function sceneVideoDetailsHtml(segment, sceneIndex, srtPath, outputFolder, statusText = "Preparing hidden video workflow...", details = {}) {
    const videoMode = details.videoMode === "t2v" ? "t2v" : "i2v";
    const promptNumber = Number(details.promptNumber || sceneIndex + 1);
    const imageIndex = sceneSlotNumber(segment) - 1;
    const imageSource = segmentImageSource(segment);
    const imagePath = imageSource?.path || "";
    const imageSrc = imagePath ? makeEditorImageUrl(imagePath) : imageSource?.data || "";
    const audioPath = String(details.audioPath || audioInput.value || "");
    const audioMode = String(details.audioMode || (segment.custom_audio_path ? "Custom scene audio" : "Global/project audio"));
    return `
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="font-weight:900;color:#cffafe;">${escapeHtml(statusText)}</div>
        ${videoMode === "i2v" && imageSrc ? `<img src="${imageSrc}" style="width:180px;max-height:110px;object-fit:cover;border:1px solid #155e75;border-radius:6px;background:#050505;">` : ""}
        <div style="display:grid;grid-template-columns:150px minmax(0,1fr);gap:5px 10px;font-size:11px;">
          <div style="color:#67e8f9;font-weight:900;">Scene</div><div>${escapeHtml(segment.label || `Scene ${promptNumber}`)}</div>
          <div style="color:#67e8f9;font-weight:900;">Video mode</div><div>${videoMode === "t2v" ? "Text to Video" : "Image to Video"}</div>
          ${videoMode === "i2v" ? `<div style="color:#67e8f9;font-weight:900;">Image index</div><div>${imageIndex} (0 based)</div>` : ""}
          <div style="color:#67e8f9;font-weight:900;">SRT prompt #</div><div>${promptNumber} (1 based)</div>
          <div style="color:#67e8f9;font-weight:900;">Audio mode</div><div>${escapeHtml(audioMode)}</div>
          ${videoMode === "i2v" ? `<div style="color:#67e8f9;font-weight:900;">Image folder</div><div style="overflow-wrap:anywhere;">${escapeHtml(i2vImagesFolder())}</div>` : ""}
          ${videoMode === "i2v" ? `<div style="color:#67e8f9;font-weight:900;">Image path</div><div style="overflow-wrap:anywhere;">${escapeHtml(imagePath || imageSource?.name || "")}</div>` : ""}
          <div style="color:#67e8f9;font-weight:900;">Audio sent to LTX</div><div style="overflow-wrap:anywhere;">${escapeHtml(audioPath)}</div>
          <div style="color:#67e8f9;font-weight:900;">SRT path</div><div style="overflow-wrap:anywhere;">${escapeHtml(srtPath || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Save folder</div><div style="overflow-wrap:anywhere;">${escapeHtml(outputFolder || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Collected clips</div><div style="overflow-wrap:anywhere;">${escapeHtml(collectedSceneVideoFolder())}</div>
        </div>
        <div>
          <div style="color:#67e8f9;font-weight:900;margin-bottom:4px;">${videoMode === "t2v" ? "T2V prompt" : "I2V prompt"}</div>
          <div style="border:1px solid #155e75;border-radius:6px;background:#020617;color:#e0f2fe;padding:8px;max-height:130px;overflow:auto;white-space:pre-wrap;">${escapeHtml(segment.i2v_prompt || "")}</div>
        </div>
      </div>
    `;
  }

  function i2vVideoSettingsForSegment(segment = activeSegment()) {
    if (!segment || segment.id === activeSegment()?.id) {
      return saveI2VVideoSettingsFromPanel();
    }
    if (segment.use_scene_i2v_video_settings) {
      return cloneI2VVideoSettings(segment.i2v_video_settings || state.i2vVideoSettings);
    }
    return cloneI2VVideoSettings(state.i2vVideoSettings);
  }

  function i2vVideoSettingsPayload(segment = activeSegment()) {
    const settings = i2vVideoSettingsForSegment(segment);
    const useLoras = Boolean(settings.use_loras && Number(settings.lora_count || 0) > 0);
    const count = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
    const payload = {
      unet_name: settings.unet_name || "",
      vae_name: settings.vae_name || "",
      clip_name1: settings.clip_name1 || "",
      clip_name2: settings.clip_name2 || "",
      upscale_model_name: settings.upscale_model_name || "",
      audio_vae_name: settings.audio_vae_name || "",
      fps: Number(settings.fps || 24),
      width: Number(settings.width || 1920),
      height: Number(settings.height || 1080),
      seed: Number(settings.seed || 1),
      use_custom_loras: useLoras,
      lora_count: useLoras ? count : 0,
    };
    for (let index = 0; index < 4; index += 1) {
      const lora = settings.loras?.[index] || {};
      payload[`lora_${index + 1}`] = useLoras && index < count ? (lora.name || "[none]") : "[none]";
      payload[`first_pass_strength_${index + 1}`] = Number(lora.first_pass_strength ?? lora.strength ?? 1);
      payload[`second_pass_strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
    }
    return payload;
  }

  function sceneDisplayName(segment, sceneIndex) {
    const info = segmentIndexInfo(segment);
    if (info.track === "overlay") {
      const index = info.index >= 0 ? info.index : sceneIndex;
      return `Insert ${index + 1}. ${segment?.label || `Insert ${index + 1}`}`;
    }
    const index = sceneIndex >= 0 ? sceneIndex : info.index;
    return `${index + 1}. ${segment?.label || `Scene ${index + 1}`}`;
  }

  function validateSceneReadyForVideo(segment, sceneIndex) {
    const name = sceneDisplayName(segment, sceneIndex);
    const missing = [];
    const mode = currentVideoMode();
    if (mode !== "t2v" && !segmentImageSource(segment)) missing.push(`${name}: selected scene image is missing.`);
    if (!String(segment?.i2v_prompt || "").trim()) missing.push(`${name}: ${mode === "t2v" ? "T2V" : "I2V"} prompt is missing.`);
    return missing;
  }

  async function validateSrtTimingForSceneVideo({ segment, sceneIndex, srtPath, promptNumber, expectedDuration }) {
    const promptIndex = Math.max(0, Number(promptNumber || 1) - 1);
    const uiDuration = Number(expectedDuration ?? timelineSegmentDuration(segment));
    const data = await postJson("/vrgdg/music_builder/load_srt", { srt_path: srtPath }, 60000);
    const srtSegment = Array.isArray(data.segments) ? data.segments[promptIndex] : null;
    if (!srtSegment) {
      throw new Error(
        `SRT timing check failed before creating video.\n\n` +
        `${sceneDisplayName(segment, sceneIndex)} is using prompt #${promptNumber}, but that prompt was not found in:\n${srtPath}`
      );
    }
    const srtDuration = Math.max(0, Number(srtSegment.end || 0) - Number(srtSegment.start || 0));
    const diff = Math.abs(srtDuration - uiDuration);
    console.log("[VRGDG Music Builder] SRT timing check", {
      scene: sceneIndex + 1,
      promptNumber,
      srtPath,
      uiStart: Number(segment.start || 0),
      uiEnd: Number(segment.end || 0),
      uiDuration,
      srtStart: Number(srtSegment.start || 0),
      srtEnd: Number(srtSegment.end || 0),
      srtDuration,
      diff,
    });
    if (!Number.isFinite(uiDuration) || uiDuration <= 0) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} has an invalid UI duration: ${uiDuration}`);
    }
    if (!Number.isFinite(srtDuration) || srtDuration <= 0) {
      throw new Error(`SRT prompt #${promptNumber} has an invalid duration in:\n${srtPath}`);
    }
    if (diff > 0.25) {
      throw new Error(
        `SRT timing mismatch before creating video.\n\n` +
        `${sceneDisplayName(segment, sceneIndex)} UI duration: ${uiDuration.toFixed(3)}s\n` +
        `SRT prompt #${promptNumber} duration: ${srtDuration.toFixed(3)}s\n\n` +
        `UI timing: ${Number(segment.start || 0).toFixed(3)} -> ${Number(segment.end || 0).toFixed(3)}\n` +
        `SRT timing: ${Number(srtSegment.start || 0).toFixed(3)} -> ${Number(srtSegment.end || 0).toFixed(3)}\n\n` +
        `SRT path being sent to hidden workflow:\n${srtPath}\n\n` +
        `Stopping before LTX runs so it cannot accidentally render the wrong/huge clip.`
      );
    }
    return { srt_segment: srtSegment, srt_duration: srtDuration, ui_duration: uiDuration, srt_path: data.srt_path || srtPath };
  }

  async function ensureSelectedImageForSceneVideo(segment, sceneIndex) {
    const source = segmentImageSource(segment);
    if (!source) throw new Error(`${sceneDisplayName(segment, sceneIndex)}: selected scene image is missing.`);
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) throw new Error("Project folder is missing.");
    const data = await postJson("/vrgdg/music_builder/save_scene_image", {
      source_path: source.path || "",
      image_data: source.data || "",
      project_folder: projectFolder,
      scene_number: sceneSlotNumber(segment),
    });
    segment.approved_image_path = data.saved_path || "";
    ensureSegmentRuntimeFields(segment);
    return segment.approved_image_path;
  }

  function validateRenderAllReady(options = {}) {
    const missing = [];
    const allScenes = allEditableSegments();
    if (!allScenes.length) missing.push("No scenes found. Add or load scenes first.");
    const scenesToRender = allScenes
      .map((segment, index) => ({ segment, index }))
      .filter(({ segment }) => options.forceVideos || !String(selectedSegmentVideoPath(segment) || "").trim());
    const sceneAudioMode = usingSceneAudioMode();
    if (!sceneAudioMode && !String(audioInput.value || "").trim()) missing.push("Audio file path is missing.");
    if (sceneAudioMode) {
      state.segments.forEach((segment, index) => {
        if (!String(segment.custom_audio_path || "").trim()) {
          missing.push(`${sceneDisplayName(segment, index)}: scene audio is missing.`);
        }
      });
    }
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    scenesToRender.forEach(({ segment }) => {
      missing.push(...validateSceneReadyForVideo(segment, segmentIndexInfo(segment).index));
    });
    return missing;
  }

  function renderMissingListHtml(missing) {
    return `
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="font-weight:900;color:#fecaca;">Render All cannot start yet.</div>
        <div>Fix these first, then press Render All again:</div>
        <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
      </div>
    `;
  }

  function imageAllSegmentsForMode(mode = "resume_missing", imageMode = state.imageModelMode || "zimage") {
    const scenes = allEditableSegments().map((segment) => ({ segment, index: segmentIndexInfo(segment).index }));
    if (mode === "redo_prompts_images" || mode === "keep_prompts_redo_images") return scenes;
    return scenes.filter(({ segment }) => !segmentImageSource(segment));
  }

  function validateZImageAllReady(options = {}) {
    const mode = options.imageRunMode || "resume_missing";
    const missing = [];
    if (!allEditableSegments().length) missing.push("No scenes found. Add or load scenes first.");
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    imageAllSegmentsForMode(mode).forEach(({ segment }) => {
      if (mode !== "redo_prompts_images" && String(segment.t2i_prompt || "").trim()) return;
      const reason = t2iMissingReason(segment);
      if (reason) missing.push(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${reason}`);
    });
    return missing;
  }

  async function prepareSceneAudioMix(progress, label = "Preparing scene audio mix", options = {}) {
    const sceneAudioMode = usingSceneAudioMode();
    if (!sceneAudioMode) {
      return {
        audioPath: audioInput.value,
        srtPath: state.srtPath || srtInput.value,
        usedSceneAudio: false,
      };
    }
    progress?.set(`${label}...`, 6);
    const data = await postJson("/vrgdg/music_builder/prepare_scene_audio_mix", {
      project_folder: projectInput.value,
      segments: state.segments,
      allow_missing_scene_audio: !!options.allowMissingSceneAudio,
    }, 180000);
    if (data.audio_path) {
      audioInput.value = data.audio_path;
      setWidgetValue(node, "audio_path", data.audio_path);
      audio.src = audioUrl(data.audio_path);
      audio.load();
      state.duration = Math.max(state.duration || 0, Number(data.duration || 0));
      state.peaks = Array.isArray(data.peaks) ? data.peaks : state.peaks;
      state.beats = Array.isArray(data.beats) ? data.beats : state.beats;
      showBeatMarkersIfAvailable();
    }
    if (data.srt_path) {
      state.srtPath = data.srt_path;
      srtInput.value = data.srt_path;
      setWidgetValue(node, "srt_path", data.srt_path);
    }
    render();
    return {
      audioPath: data.audio_path || audioInput.value,
      srtPath: data.srt_path || state.srtPath || srtInput.value,
      usedSceneAudio: true,
    };
  }

  async function renderSceneVideoWithProgress(segment, sceneIndex, progress, options = {}) {
    const progressBase = Number(options.progressBase ?? 0);
    const progressSpan = Number(options.progressSpan ?? 100);
    const batchLabel = options.batchLabel ? `${options.batchLabel}\n` : "";
    const pct = (value) => Math.min(100, progressBase + (progressSpan * value / 100));
    const slotNumber = sceneSlotNumber(segment);
    state.activeId = segment.id;
    syncInspector();
    updateActiveFromInputs();
    const videoMode = currentVideoMode();
    const modeLabel = videoMode === "t2v" ? "T2V" : "I2V";
    const missing = validateSceneReadyForVideo(segment, sceneIndex);
    if (missing.length) throw new Error(missing.join("\n"));
    segment.video_status = "running";
    renderList();
    progress?.set(`${batchLabel}Saving current UI session/SRT timing...`, pct(8));
    let srtPath = await saveSessionForSceneVideo();
    if (!srtPath) throw new Error("The builder SRT path was not created.");
    if (videoMode === "i2v") {
      progress?.set(`${batchLabel}Preparing selected scene image for I2V...`, pct(12));
      await ensureSelectedImageForSceneVideo(segment, sceneIndex);
    } else {
      progress?.set(`${batchLabel}Preparing text-to-video render...`, pct(12));
    }
    let audioPathForScene = options.audioPathOverride || audioInput.value;
    let promptNumberForScene = sceneIndex + 1;
    let audioModeForScene = options.audioPathOverride ? "Combined scene-audio track" : "Global/project audio trimmed for this scene";
    if (options.srtPathOverride) {
      srtPath = options.srtPathOverride;
    }
    if (!options.audioPathOverride) {
      const sourceAudioPath = segment.custom_audio_path || audioInput.value;
      const sceneDuration = Math.max(0.1, timelineSegmentDuration(segment) || 4);
      const sourceStart = segment.custom_audio_path ? audioSourceStart(segment) : Number(segment.start || 0);
      if (!String(sourceAudioPath || "").trim()) {
        throw new Error(`${sceneDisplayName(segment, sceneIndex)}: no audio path is being sent to LTX. Add custom scene audio or load project/global audio before creating the video.`);
      }
      const trimmedAudio = await postJson("/vrgdg/music_builder/trim_scene_audio", {
        project_folder: projectInput.value,
        scene_number: slotNumber,
        source_path: sourceAudioPath,
        start: sourceStart,
        duration: sceneDuration,
      }, 120000);
      audioPathForScene = trimmedAudio.audio_path || sourceAudioPath;
      const singleSrt = await postJson("/vrgdg/music_builder/save_single_scene_srt", {
        project_folder: projectInput.value,
        scene_number: slotNumber,
        duration: sceneDuration,
        label: segment.label || `Scene ${sceneIndex + 1}`,
      }, 60000);
      srtPath = singleSrt.srt_path || srtPath;
      promptNumberForScene = 1;
      audioModeForScene = segment.custom_audio_path ? "Custom scene audio trimmed for this scene" : "Global/project audio trimmed for this scene";
    }
    if (!String(audioPathForScene || "").trim()) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)}: no audio path is being sent to LTX. Add custom scene audio or load project/global audio before creating the video.`);
    }
    progress?.set(`${batchLabel}Checking SRT timing before hidden ${modeLabel}...`, pct(14));
    const expectedDurationForScene = !options.audioPathOverride
      ? Math.max(0.1, timelineSegmentDuration(segment) || 4)
      : timelineSegmentDuration(segment);
    const timingCheck = await validateSrtTimingForSceneVideo({
      segment,
      sceneIndex,
      srtPath,
      promptNumber: promptNumberForScene,
      expectedDuration: expectedDurationForScene,
    });
    const payload = {
      ...i2vVideoSettingsPayload(segment),
      i2v_prompt: segment.i2v_prompt,
      t2v_prompt: segment.i2v_prompt,
      audio_path: audioPathForScene,
      prompt_number_one_based: promptNumberForScene,
      srt_path: srtPath,
      project_folder: projectInput.value,
    };
    if (videoMode === "i2v") {
      payload.image_folder = i2vImagesFolder();
      payload.image_index_zero_based = slotNumber - 1;
    }
    const workflowDetails = {
      audioPath: audioPathForScene,
      promptNumber: promptNumberForScene,
      audioMode: audioModeForScene,
      videoMode,
    };
    const defaultOutputFolder = activeVideoOutputFolder(videoMode);
    const buildEndpoint = videoMode === "t2v" ? "/vrgdg/workflow_runner/build_t2v_prompt" : "/vrgdg/workflow_runner/build_i2v_prompt";
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, defaultOutputFolder, `${batchLabel}Preparing hidden ${modeLabel} workflow...\nSRT timing verified: ${timingCheck.srt_duration.toFixed(3)}s`, workflowDetails), pct(15));
    const built = await postJson(buildEndpoint, payload);
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}Queueing hidden ${modeLabel} workflow...`, workflowDetails), pct(40));
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the video but did not return a prompt_id.");
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}Queued prompt ID: ${promptId}\nWaiting for video...`, workflowDetails), pct(60));
    const videos = await waitForVideos(
      promptId,
      (message) => progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}${message}\nPrompt ID: ${promptId}`, workflowDetails), pct(80)),
      () => state.batchCancelled
    );
    const video = videos[videos.length - 1] || null;
    const videoPath = resolveComfyVideoPath(video);
    if (!videoPath) throw new Error(`The ${modeLabel} workflow finished, but no video path was found in history.`);
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}Collecting scene video into builder folder...`, workflowDetails), pct(90));
    const collected = await postJson("/vrgdg/workflow_runner/collect_scene_video", {
      source_path: videoPath,
      project_folder: projectInput.value,
      scene_number: slotNumber,
      existing_action: options.existingVideoAction || "overwrite",
    }, 120000);
    pushHistory();
    if (collected.backup_path) {
      if (!Array.isArray(segment.video_backup_paths)) segment.video_backup_paths = [];
      if (!segment.video_backup_paths.some((item) => mediaPathKey(item) === mediaPathKey(collected.backup_path))) {
        segment.video_backup_paths.push(collected.backup_path);
      }
    }
    segment.video_output = video;
    segment.video_source_path = videoPath;
    segment.video_path = collected.video_path || videoPath;
    segment.video_cache_bust = Date.now();
    segment.video_folder = collected.video_folder || collectedSceneVideoFolder();
    normalizeSegmentVideoHistory(segment);
    const currentVideoIndex = segment.video_history.findIndex((item) => mediaPathKey(item) === mediaPathKey(segment.video_path));
    if (currentVideoIndex >= 0) segment.video_history_index = currentVideoIndex;
    segment.preview_mode = "video";
    segment.video_status = "done";
    syncPreview(segment);
    render();
    if (options.autoSaveAfter !== false) {
      await autoSaveSessionQuiet(options.autoSaveReason || "scene video complete");
    }
    const backupNote = collected.backup_path ? `\n\nPrevious video backed up to:\n${collected.backup_path}` : "";
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, segment.video_folder || collectedSceneVideoFolder(), `${batchLabel}Scene video ready.\n${segment.video_path}${backupNote}`, workflowDetails), pct(100));
    return segment.video_path;
  }

  async function stitchRenderedScenes(progress, options = {}) {
    const baseSegments = Array.isArray(options.segments) && options.segments.length ? options.segments : state.segments;
    const overlaySegments = Array.isArray(options.overlaySegments) ? options.overlaySegments : state.overlaySegments;
    const timelineOffset = Number(options.timelineOffset || 0);
    const paths = baseSegments.map((segment) => String(selectedSegmentVideoPath(segment) || "").trim());
    const overlayItems = overlaySegments
      .filter((segment) => String(selectedSegmentVideoPath(segment) || "").trim())
      .map((segment, index) => ({
        path: String(selectedSegmentVideoPath(segment) || "").trim(),
        start: Math.max(0, Number(segment.start || 0) - timelineOffset),
        end: Math.max(0.05, Number(segment.end || 0) - timelineOffset),
        source_start: Math.max(0, Number(segment.overlay_source_start || 0)),
        label: segment.label || `Insert ${index + 1}`,
      }));
    const sceneAudioMode = usingSceneAudioMode();
    const audioPaths = sceneAudioMode ? baseSegments.map((segment) => String(segment.custom_audio_path || "").trim()) : [];
    const audioItems = sceneAudioMode ? baseSegments.map((segment) => ({
      path: String(segment.custom_audio_path || "").trim(),
      start: audioSourceStart(segment),
      duration: audioChunkDuration(segment),
    })) : [];
    const missing = [];
    paths.forEach((path, index) => {
      if (!path) missing.push(`${sceneDisplayName(baseSegments[index], index)}: rendered scene video is missing.`);
      if (sceneAudioMode && !audioPaths[index]) missing.push(`${sceneDisplayName(baseSegments[index], index)}: scene audio is missing.`);
    });
    if (missing.length) throw new Error(missing.join("\n"));
    progress?.set(sceneAudioMode ? "Stitching rendered scene videos with scene audio clips..." : "Stitching rendered scene videos with original audio...", 94);
    const data = await postJson("/vrgdg/workflow_runner/stitch_scene_videos", {
      scene_paths: paths,
      audio_path: audioInput.value,
      scene_audio_paths: audioPaths,
      scene_audio_items: audioItems,
      overlay_items: overlayItems,
      project_folder: projectInput.value,
      audio_start: Number(options.audioStart || 0),
      audio_duration: Number(options.audioDuration || 0),
      output_prefix: options.outputPrefix || "FINAL_VIDEO",
    }, 20 * 60 * 1000);
    state.finalVideoPath = data.final_video_path || "";
    return data;
  }

  async function createSceneVideo() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const info = segmentIndexInfo(segment);
    const sceneIndex = info.index;
    if (sceneIndex < 0) return;
    const missing = [
      ...validateSceneReadyForVideo(segment, sceneIndex),
      ...(!String(segment.custom_audio_path || audioInput.value || "").trim() || !String(projectInput.value || "").trim() ? ["Load global audio or add custom audio for this scene, and set the project folder first."] : []),
    ];
    if (missing.length) {
      toast(missing.join("\n"), true);
      return;
    }
    let existingVideoAction = "overwrite";
    if (String(segment.video_path || "").trim()) {
      existingVideoAction = await askExistingSceneVideoAction(segment, sceneIndex);
      if (existingVideoAction === "cancel") return;
    }
    let progress = null;
    try {
      state.batchCancelled = false;
      setButtonGroupState(createSceneVideoButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating scene video");
      const videoPath = await renderSceneVideoWithProgress(segment, sceneIndex, progress, {
        existingVideoAction,
      });
      progress.close(900);
      toast(`Scene video ready:\n${videoPath}`);
    } catch (error) {
      const stopped = /stopped by user/i.test(String(error?.message || error));
      segment.video_status = stopped ? "none" : "error";
      progress?.set(stopped ? "Scene video creation stopped by user." : `Error:\n${String(error?.message || error)}`, 100);
      toast(stopped ? "Scene video creation stopped." : String(error?.message || error), !stopped);
      renderList();
    } finally {
      setButtonGroupState(createSceneVideoButtons, { disabled: false, text: "Create Scene Video" });
    }
  }

  function overlaySegmentsForPreviewRange(startTime, endTime) {
    return state.overlaySegments
      .filter((segment) => String(selectedSegmentVideoPath(segment) || "").trim())
      .filter((segment) => Number(segment.end || 0) > startTime && Number(segment.start || 0) < endTime)
      .map((segment) => ({
        ...segment,
        overlay_source_start: Math.max(0, startTime - Number(segment.start || 0)),
        start: Math.max(startTime, Number(segment.start || 0)),
        end: Math.min(endTime, Number(segment.end || 0)),
      }));
  }

  async function stitchPreviewFromSegments(segments, label = "selected") {
    const baseSegments = (Array.isArray(segments) ? segments : []).filter((segment) => segmentTrack(segment) !== "overlay");
    if (!baseSegments.length) {
      toast("Choose at least one base scene for the preview stitch.", true);
      return;
    }
    const sorted = baseSegments.slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const baseIndexes = sorted.map((segment) => state.segments.findIndex((item) => item.id === segment.id));
    const nonContiguous = baseIndexes.some((index) => index < 0) || baseIndexes.some((index, itemIndex) => itemIndex > 0 && index !== baseIndexes[itemIndex - 1] + 1);
    if (nonContiguous) {
      toast("Preview stitching needs contiguous base scenes. Select one continuous scene range, or use Start scene / End scene.", true);
      return;
    }
    const startTime = Math.min(...sorted.map((segment) => Number(segment.start || 0)));
    const endTime = Math.max(...sorted.map((segment) => Number(segment.end || 0)));
    const overlays = overlaySegmentsForPreviewRange(startTime, endTime);
    const progress = createProgressWindow("Stitch Preview");
    try {
      progress.set(`Stitching preview from ${label}...\nScenes: ${sorted.length}`, 15);
      const stitched = await stitchRenderedScenes(progress, {
        segments: sorted,
        overlaySegments: overlays,
        timelineOffset: startTime,
        audioStart: startTime,
        audioDuration: Math.max(0.1, endTime - startTime),
        outputPrefix: `PREVIEW_SCENES_${label.replace(/[^a-z0-9_-]+/gi, "_")}`,
      });
      progress.set(`Preview stitch complete.\n\nPreview video:\n${stitched.final_video_path}`, 100);
      progress.close(6500);
      toast(`Preview stitch complete:\n${stitched.final_video_path}`);
      showFinalVideoReadyModal(stitched.final_video_path);
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  function openStitchPreviewModal() {
    const selectedBase = selectedSegmentsForBatch({ baseOnly: true })
      .slice()
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Stitch Preview";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Creates a preview video from already-rendered scene videos. Inserts are included automatically. This does not run Gemma, create images, or render new videos.";
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const selectedCount = selectedBase.length;
    const selectedButton = makeButton(`Use Selected Scenes${selectedCount ? ` (${selectedCount})` : ""}`, "primary");
    selectedButton.disabled = !selectedCount;
    const startInput = makeInput("1");
    const endInput = makeInput(String(Math.max(1, state.segments.length)));
    startInput.type = "number";
    endInput.type = "number";
    startInput.min = "1";
    endInput.min = "1";
    startInput.max = String(Math.max(1, state.segments.length));
    endInput.max = String(Math.max(1, state.segments.length));
    const rangeGrid = document.createElement("div");
    rangeGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    rangeGrid.append(makeField("Start scene", startInput), makeField("End scene", endInput));
    const rangeButton = makeButton("Use Scene Range", "primary");
    const cancel = makeButton("Cancel");
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    actions.append(cancel, rangeButton);
    box.append(header, note, selectedButton, rangeGrid, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    close.onclick = () => backdrop.remove();
    cancel.onclick = () => backdrop.remove();
    selectedButton.onclick = () => {
      backdrop.remove();
      stitchPreviewFromSegments(selectedBase, "selected");
    };
    rangeButton.onclick = () => {
      const start = Math.max(1, Math.min(state.segments.length, Number(startInput.value || 1)));
      const end = Math.max(start, Math.min(state.segments.length, Number(endInput.value || start)));
      const scenes = state.segments.slice(start - 1, end);
      backdrop.remove();
      stitchPreviewFromSegments(scenes, `${String(start).padStart(3, "0")}-${String(end).padStart(3, "0")}`);
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  function showMultiSelectHint() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Select Multi";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:10px;font-size:12px;line-height:1.45;color:#d4d4d8;";
    const batch = document.createElement("div");
    batch.innerHTML = `<strong style="color:#e0f2fe;">Batch scene settings</strong><br>Turn Select Multi on, click the scenes you want, then change image or video model settings. Those settings are saved only to the selected scenes as custom scene settings.`;
    const preview = document.createElement("div");
    preview.innerHTML = `<strong style="color:#e0f2fe;">Stitch Preview</strong><br>Use the Stitch Preview menu option to make a quick complete video from selected scenes or from a start/end scene range. Inserts are included automatically, and no Gemma, image generation, or video rendering is run.`;
    const note = document.createElement("div");
    note.textContent = "Selected scenes turn red. Turn Select Multi off when you want normal single-scene editing again.";
    note.style.cssText = "border:1px solid #334155;border-radius:6px;background:#0f172a;padding:9px;color:#cbd5e1;";
    body.append(batch, preview, note);
    const ok = makeButton("Got it", "primary");
    box.append(header, body, ok);
    backdrop.append(box);
    document.body.append(backdrop);
    close.onclick = () => backdrop.remove();
    ok.onclick = () => backdrop.remove();
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  async function renderAllScenes(options = {}) {
    updateActiveFromInputs();
    saveI2VVideoSettingsFromPanel();
    const forceVideos = Boolean(options.forceVideos);
    const randomizeVideoSeed = Boolean(options.randomizeVideoSeed);
    const missing = validateRenderAllReady({ forceVideos });
    const progress = createProgressWindow("Render All Scenes");
    if (missing.length) {
      progress.setHtml(renderMissingListHtml(missing), 100);
      toast("Render All needs a few things fixed first.", true);
      return;
    }
    try {
      state.batchCancelled = false;
      renderAllButton.disabled = true;
      renderAllButton.textContent = "Rendering...";
      setButtonGroupState(createSceneVideoButtons, { disabled: true });
      progress.set("Autosaving session/SRT before Render All...", 3);
      await saveSessionForSceneVideo();
      const preparedAudio = await prepareSceneAudioMix(progress, "Preparing combined scene-audio track for LTX");
      const scenes = allEditableSegments()
        .map((segment) => ({ segment, index: segmentIndexInfo(segment).index }))
        .filter(({ segment }) => forceVideos || !String(selectedSegmentVideoPath(segment) || "").trim());
      if (!scenes.length) {
        progress.set("All scenes already have video. Stitching existing scene videos...", 80);
      }
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = Math.floor((index / scenes.length) * 100);
        const span = Math.max(1, Math.floor(80 / scenes.length));
        if (randomizeVideoSeed) setVideoSeedRandom(segment);
        progress.set(`Rendering ${sceneLabel} (${index + 1} of ${scenes.length}; ${forceVideos ? "creating a new video version" : "existing videos skipped"})...`, base);
        await renderSceneVideoWithProgress(segment, sceneIndex, progress, {
          progressBase: base,
          progressSpan: span,
          batchLabel: `Render All ${index + 1}/${scenes.length}: ${segment.label || `Scene ${sceneIndex + 1}`}`,
          autoSaveAfter: false,
          existingVideoAction: forceVideos ? "backup" : "overwrite",
          audioPathOverride: segmentTrack(segment) === "overlay" ? "" : preparedAudio.audioPath,
          srtPathOverride: segmentTrack(segment) === "overlay" ? "" : preparedAudio.srtPath,
        });
        assertBatchNotStopped();
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      assertBatchNotStopped();
      const stitched = await stitchRenderedScenes(progress);
      await autoSaveSessionQuiet("render all final stitch complete");
      progress.set(`Render All complete.\n\nFinal video:\n${stitched.final_video_path}\n\nScene clips:\n${stitched.video_folder}`, 100);
      progress.close(6500);
      toast(`Render All complete:\n${stitched.final_video_path}`);
      showFinalVideoReadyModal(stitched.final_video_path);
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      renderAllButton.disabled = false;
      renderAllButton.textContent = "Render All";
      setButtonGroupState(createSceneVideoButtons, { disabled: false, text: "Create Scene Video" });
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function zImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const missing = validateZImageAllReady({ imageRunMode });
    const progress = createProgressWindow("Z-Image All Scenes");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Z-Image All cannot start yet.</div>
          <div>Fix these first, then press Z-Image All again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Z-Image All needs scene notes first.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Z-Imaging...";
      setButtonGroupState(zCreateButtons, { disabled: true });
      createT2IButton.disabled = true;
      progress.set("Autosaving session/SRT before Z-Image All...", 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "zimage");
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Z-Image All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Z-Image All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing T2I prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have T2I prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        state.activeId = segment.id;
        syncInspector();
        render();
        await runGemmaImagePromptPassWithRetry(
          segment,
          progress,
          base,
          `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
          generateT2IPromptForSegment,
          { unloadAfter: false },
        );
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} ZImage image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        state.activeId = segment.id;
        syncInspector();
        render();
        if (forceNewImages) setImageSeedForCurrentMode("zimage");
        progress.set(`Z-Image image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved T2I prompt...`, base);
        await createZImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Z-Image All ${index + 1}/${scenes.length}: ZImage`);
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Z-Image All scene ${sceneIndex + 1}`);
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Z-Image All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Z-Image All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Z-Image All" : "Z-Image All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Z-Image All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(zCreateButtons, { disabled: false, text: "Create Z-Image" });
      createT2IButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function ernieImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const missing = validateZImageAllReady({ imageRunMode });
    const progress = createProgressWindow("Ernie Image All Scenes");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Ernie Image All cannot start yet.</div>
          <div>Fix these first, then press Image All again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Ernie Image All needs scene notes first.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Ernie...";
      setButtonGroupState(ernieCreateButtons, { disabled: true });
      createT2IButton.disabled = true;
      ernieCreateT2IButton.disabled = true;
      progress.set("Autosaving session/SRT before Ernie Image All...", 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "ernie_image");
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Ernie Image All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Ernie Image All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing T2I prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have T2I prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        state.activeId = segment.id;
        syncInspector();
        render();
        await runGemmaImagePromptPassWithRetry(
          segment,
          progress,
          base,
          `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
          generateT2IPromptForSegment,
          { unloadAfter: false },
        );
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} Ernie image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        state.activeId = segment.id;
        syncInspector();
        render();
        if (forceNewImages) setImageSeedForCurrentMode("ernie_image");
        progress.set(`Ernie image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved T2I prompt...`, base);
        await createErnieImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Ernie Image All ${index + 1}/${scenes.length}: Ernie`);
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Ernie Image All scene ${sceneIndex + 1}`);
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Ernie Image All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Ernie Image All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Ernie Image All" : "Ernie Image All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Ernie Image All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(ernieCreateButtons, { disabled: false, text: "Create with Ernie" });
      createT2IButton.disabled = false;
      ernieCreateT2IButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function fluxKleinAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const progress = createProgressWindow("Flux/Klein All Scenes");
    if (!allEditableSegments().some((segment) => mergedFluxImageIngredients(segment).length)) {
      const message = "Flux/Klein All needs at least one Reference Builder, global, or scene image ingredient.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Fluxing...";
      setButtonGroupState(fluxCreateButtons, { disabled: true });
      createFluxPromptButton.disabled = true;
      progress.set("Autosaving session/SRT before Flux/Klein All...", 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "flux_klein");
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Flux/Klein All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Flux/Klein All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.flux_prompt || segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing Flux/Klein prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have image prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        try {
          await runGemmaImagePromptPassWithRetry(
            segment,
            progress,
            base,
            `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
            generateFluxKleinPromptForSegment,
            { clearBeforeLoad: index === 0, unloadAfter: false },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
        } catch (error) {
          throw new Error(`Image All prompt pass stopped at ${sceneLabel} (${index + 1}/${promptScenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} Flux/Klein image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        try {
          if (forceNewImages) setImageSeedForCurrentMode("flux_klein");
          progress.set(`Flux/Klein image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved Flux/Klein prompt...`, base);
          await createFluxKleinImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Flux/Klein All ${index + 1}/${scenes.length}: Image`);
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Flux/Klein All scene ${sceneIndex + 1}`);
          await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
        } catch (error) {
          throw new Error(`Flux/Klein image pass stopped at ${sceneLabel} (${index + 1}/${scenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      await autoSaveSessionQuiet("Flux/Klein All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Flux/Klein All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Flux/Klein All" : "Flux/Klein All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Flux/Klein All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(fluxCreateButtons, { disabled: false, text: "Create with Flux/Klein" });
      createFluxPromptButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function nbImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const progress = createProgressWindow("NanoBanana All Scenes");
    const allHaveReferences = allEditableSegments().some((segment) => mergedFluxImageIngredients(segment).length);
    if (!allHaveReferences) {
      const message = "NanoBanana All needs at least one Reference Builder, global, or scene reference image.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    if (!String((state.nbImageSettings || {}).api_key || "").trim() && !allEditableSegments().some((segment) => String(segment.nb_image_settings?.api_key || "").trim())) {
      const message = "NanoBanana All needs a NanoBanana API key in the NB Models tab.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "NanoBanana...";
      setButtonGroupState(nbCreateButtons, { disabled: true });
      createNBPromptButton.disabled = true;
      progress.set("Autosaving session/SRT before NanoBanana All...", 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "nano_banana");
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping NanoBanana All.", 100);
        progress.close(1800);
        toast("All scenes already have images. NanoBanana All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.nb_prompt || segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing NanoBanana prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have image prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        try {
          await runGemmaImagePromptPassWithRetry(
            segment,
            progress,
            base,
            `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
            generateNBPromptForSegment,
            { clearBeforeLoad: index === 0, unloadAfter: false },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
        } catch (error) {
          throw new Error(`Image All prompt pass stopped at ${sceneLabel} (${index + 1}/${promptScenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} NanoBanana image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        try {
          progress.set(`NanoBanana image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved NanoBanana prompt...`, base);
          await createNBImageForSegmentWithRetry(
            segment,
            progress,
            base + span * 0.35,
            span * 0.45,
            `NanoBanana All ${index + 1}/${scenes.length}: ${sceneLabel}`,
            { maxRetries: 10 },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`NanoBanana All scene ${sceneIndex + 1}`);
          await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
        } catch (error) {
          throw new Error(`NanoBanana image pass stopped at ${sceneLabel} (${index + 1}/${scenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      await autoSaveSessionQuiet("NanoBanana All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("NanoBanana All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped NanoBanana All" : "NanoBanana All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after NanoBanana All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(nbCreateButtons, { disabled: false, text: "Create with NanoBanana" });
      createNBPromptButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function createNBImageForSegmentWithRetry(segment, progress, percentBase, percentSpan, label, options = {}) {
    const maxRetries = Math.max(1, Number(options.maxRetries || 10));
    let lastError = null;
    for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
      assertBatchNotStopped();
      try {
        const attemptLabel = attempt === 1 ? label : `${label} retry ${attempt}/${maxRetries}`;
        progress?.set(`${attemptLabel}: creating NanoBanana image...`, percentBase);
        return await createNBImageForSegment(segment, progress, percentBase, percentSpan, attemptLabel);
      } catch (error) {
        lastError = error;
        const message = String(error?.message || error || "Unknown error");
        if (attempt >= maxRetries) break;
        progress?.set(
          `${label}: NanoBanana failed on attempt ${attempt}/${maxRetries}.\n${message}\n\nClearing memory before retry ${attempt + 1}/${maxRetries}...`,
          Math.min(99, percentBase + percentSpan),
        );
        await cancelComfyExecutionAndWaitIdle((status) => {
          progress?.set(`${label}: cancelling failed job before retry...\n${status}`, Math.min(99, percentBase + percentSpan));
        }, { shouldCancel: () => state.batchCancelled });
        try {
          await runClearMemoryWorkflowQuiet(progress, `${label} failed attempt ${attempt}/${maxRetries}`, Math.min(99, percentBase + percentSpan));
        } catch (cleanupError) {
          console.warn("[VRGDG Music Builder] Cleanup before NanoBanana retry failed:", cleanupError);
          progress?.set(
            `${label}: cleanup failed before retry ${attempt + 1}/${maxRetries}, retrying anyway.\n${String(cleanupError?.message || cleanupError)}`,
            Math.min(99, percentBase + percentSpan),
          );
        }
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }
    progress?.set(`${label}: failed after ${maxRetries} NanoBanana attempts. Clearing memory before stopping...`, 100);
    try {
      await runClearMemoryWorkflowQuiet(progress, `${label} failed after ${maxRetries} attempts`, 100);
    } catch (cleanupError) {
      console.warn("[VRGDG Music Builder] Final cleanup after NanoBanana retry failure failed:", cleanupError);
    }
    throw new Error(`${label}: NanoBanana failed after ${maxRetries} attempts.\nLast error:\n${String(lastError?.message || lastError || "Unknown error")}`);
  }

  async function buildFullVideoPipeline(options = {}) {
    let buildMode = options.buildMode || "resume_missing";
    const maxAutoRetries = Math.max(0, Math.min(5, Number(options.maxAutoRetries ?? 3)));
    let attempt = 0;
    let progress = null;
    try {
      fullBuildButton.disabled = true;
      fullBuildButton.textContent = "Building...";
      renderAllButton.disabled = true;
      zImageAllButton.disabled = true;
      state.batchCancelled = false;
      while (true) {
        attempt += 1;
        progress = createProgressWindow(attempt > 1 ? `Build Full Video retry ${attempt}/${maxAutoRetries + 1}` : "Build Full Video");
        try {
          const videoMode = currentVideoMode();
          if (videoMode === "t2v") {
            progress.set("Stage 1/3: Text-to-video mode skips image generation.", 20);
          } else {
            const imageStage = (state.imageModelMode || "") === "flux_klein" ? "Flux/Klein image pass" : state.imageModelMode === "nano_banana" ? "NanoBanana image pass" : state.imageModelMode === "ernie_image" ? "Ernie image pass" : "Z-Image pass";
            progress.set(`Stage 1/3: ${imageStage}...`, 5);
            const imageMode = state.imageModelMode || "zimage";
            const imageRunMode = buildMode === "fresh_rebuild" ? "redo_prompts_images" : "resume_missing";
            if (imageMode === "flux_klein") {
              await fluxKleinAllScenes({ throwOnError: true, imageRunMode });
            } else if (imageMode === "nano_banana") {
              await nbImageAllScenes({ throwOnError: true, imageRunMode });
            } else if (imageMode === "ernie_image") {
              await ernieImageAllScenes({ throwOnError: true, imageRunMode });
            } else {
              await zImageAllScenes({ throwOnError: true, imageRunMode });
            }
          }
          assertBatchNotStopped();
          const videoPromptStage = currentVideoMode() === "t2v" ? "creating T2V prompts" : "creating I2V prompts";
          progress.set(`Stage 2/3: ${videoPromptStage}...`, 38);
          await i2vAllScenes({
            throwOnError: true,
            i2vRunMode: buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" ? "redo_prompts" : "resume_missing",
          });
          assertBatchNotStopped();
          progress.set("Stage 3/3: rendering and stitching scene videos...", 68);
          progress.close(300);
          progress = null;
          const shouldRandomizeVideoSeed = buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" || buildMode === "redo_videos";
          await renderAllScenes({
            forceVideos: buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" || buildMode === "redo_videos",
            randomizeVideoSeed: shouldRandomizeVideoSeed && options.videoSeedMode !== "keep",
          });
          toast(attempt > 1 ? `Build Full Video complete after ${attempt} attempts.` : "Build Full Video complete.");
          break;
        } catch (error) {
          const errorMessage = String(error?.message || error);
          const canRetry = !state.batchCancelled && attempt <= maxAutoRetries && isRecoverableBuildGemmaError(error);
          if (!canRetry) throw error;
          progress?.set(`Build Full Video hit a recoverable Gemma error on attempt ${attempt}/${maxAutoRetries + 1}:\n${errorMessage}`, 100);
          await recoverFromBuildGemmaError(error, attempt, maxAutoRetries, progress);
          progress = null;
          buildMode = "resume_missing";
        }
      }
    } catch (error) {
      const errorMessage = String(error?.message || error);
      progress?.set(`Build Full Video stopped:\n${errorMessage}`, 100);
      toast(`Full video build stopped:\n${errorMessage}`, true);
    } finally {
      fullBuildButton.disabled = false;
      fullBuildButton.textContent = "Build Full Video";
      renderAllButton.disabled = false;
      zImageAllButton.disabled = false;
      state.batchCancelled = false;
    }
  }

  async function loadCustomImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    customImageFileInput.value = "";
    customImageFileInput.click();
  }

  async function addSegment() {
    const active = activeSegment();
    const duration = 4;
    let insertIndex = state.segments.length;
    let start = state.segments[state.segments.length - 1]?.end || 0;
    if (active) {
      const activeIndex = state.segments.findIndex((segment) => segment.id === active.id);
      if (activeIndex >= 0) {
        const choice = await showAddSegmentPositionModal(active.label || `Scene ${activeIndex + 1}`);
        if (!choice) return;
        if (choice === "before") {
          insertIndex = Math.max(0, activeIndex);
          start = active.start;
          for (let index = activeIndex; index < state.segments.length; index += 1) {
            shiftSegmentTiming(state.segments[index], duration);
          }
        } else {
          insertIndex = activeIndex + 1;
          start = active.end;
          for (let index = activeIndex + 1; index < state.segments.length; index += 1) {
            shiftSegmentTiming(state.segments[index], duration);
          }
        }
      }
    }
    const end = start + duration;
    const segment = newSegment(start, end);
    segment.source = state.srtMode ? "inserted" : "manual";
    pushHistory();
    state.segments.splice(insertIndex, 0, segment);
    state.duration = Math.max(Number(state.duration || 0), end, ...state.segments.map((item) => Number(item.end || 0)));
    sortSegments(state.segments);
    setActiveSegment(segment);
    await syncPromptJsonFromSegments("segment added");
    await syncI2VMotionJsonFromSegments("segment added");
    autoSaveSessionQuiet("segment added");
  }

  function parseBulkTimeValue(raw) {
    const text = String(raw || "").trim().replace(",", ".");
    if (!text) return NaN;
    const parts = text.split(":").map((part) => part.trim());
    if (parts.length === 1) return Number(parts[0]);
    if (parts.length === 2) return Number(parts[0]) * 60 + Number(parts[1]);
    if (parts.length === 3) return Number(parts[0]) * 3600 + Number(parts[1]) * 60 + Number(parts[2]);
    return NaN;
  }

  function cleanBulkTimingLines(text) {
    return String(text || "")
      .split(/\r?\n/)
      .map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s*/, "").trim())
      .filter((line) => line && !line.startsWith("#"));
  }

  function parseBulkSegmentTimings(text, mode, appendStart = 0) {
    const lines = cleanBulkTimingLines(text);
    const segments = [];
    if (mode === "durations") {
      let cursor = Number(appendStart || 0);
      for (const line of lines) {
        const duration = parseBulkTimeValue(line);
        if (!Number.isFinite(duration) || duration <= 0) {
          throw new Error(`Invalid duration: ${line}`);
        }
        segments.push({ start: cursor, end: cursor + duration });
        cursor += duration;
      }
    } else if (mode === "ranges") {
      for (const line of lines) {
        const match = line.match(/^(.+?)\s*(?:-->|-|to)\s*(.+)$/i);
        if (!match) throw new Error(`Invalid start/end row: ${line}`);
        const start = parseBulkTimeValue(match[1]);
        const end = parseBulkTimeValue(match[2]);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
          throw new Error(`Invalid start/end row: ${line}`);
        }
        segments.push({ start, end });
      }
    } else {
      const markers = lines.map(parseBulkTimeValue);
      if (markers.some((value) => !Number.isFinite(value))) {
        throw new Error("One or more timestamp markers could not be read.");
      }
      for (let index = 0; index < markers.length - 1; index += 1) {
        const start = markers[index];
        const end = markers[index + 1];
        if (end <= start) throw new Error("Timestamp markers must go from earliest to latest.");
        segments.push({ start, end });
      }
    }
    if (!segments.length) throw new Error("No segments were found in the box.");
    return segments;
  }

  async function applyBulkSegmentTimings(timings, action) {
    const replacing = action === "replace";
    const newSegments = timings.map((timing, index) => {
      const segment = newSegment(Number(timing.start || 0), Number(timing.end || 0));
      segment.label = `Scene ${replacing ? index + 1 : state.segments.length + index + 1}`;
      segment.source = "manual";
      return segment;
    });
    pushHistory();
    if (replacing) {
      state.segments = newSegments;
      state.srtMode = false;
      state.timingFrozen = false;
      state.activeId = newSegments[0]?.id || "";
    } else {
      state.segments.push(...newSegments);
      state.activeId = newSegments[0]?.id || state.activeId;
    }
    sortSegments(state.segments);
    state.duration = Math.max(
      Number(state.duration || 0),
      ...state.segments.map((segment) => Number(segment.end || 0)),
      ...state.overlaySegments.map((segment) => Number(segment.end || 0)),
    );
    state.activeTrack = "base";
    syncInspector();
    render();
    await syncPromptJsonFromSegments("bulk segments");
    await syncI2VMotionJsonFromSegments("bulk segments");
    await autoSaveSessionQuiet("bulk segments");
  }

  function openBulkSegmentsModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:calc(100vh - 60px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);display:flex;flex-direction:column;overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid #155e75;background:#083f4f;";
    const heading = document.createElement("div");
    heading.textContent = "Bulk Manual Segments";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px 16px;display:flex;flex-direction:column;gap:12px;overflow:auto;";
    const explanation = document.createElement("div");
    explanation.innerHTML = `
      <div><strong style="color:#e0f2fe;">What this does</strong></div>
      <div>Create many manual base timeline scenes from pasted timing text instead of clicking + Segment over and over.</div>
      <div style="margin-top:6px;"><strong style="color:#fecaca;">Replace current base timeline</strong> rebuilds the base scenes and clears generated scene outputs from those new scenes. Inserts stay in the insert track.</div>
      <div><strong style="color:#bbf7d0;">Append after last scene</strong> adds duration-based scenes after the current last base scene.</div>
    `;
    explanation.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;color:#d4d4d8;font-size:12px;line-height:1.45;";
    const modeSelect = makeSelect(["durations", "ranges", "markers"], "durations");
    modeSelect.options[0].textContent = "Durations";
    modeSelect.options[1].textContent = "Start - End rows";
    modeSelect.options[2].textContent = "Timestamp markers";
    const actionSelect = makeSelect(["replace", "append"], "replace");
    actionSelect.options[0].textContent = "Replace current base timeline";
    actionSelect.options[1].textContent = "Append after last scene";
    const controls = document.createElement("div");
    controls.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    controls.append(makeField("Input format", modeSelect), makeField("Apply mode", actionSelect));
    const examples = document.createElement("div");
    examples.style.cssText = "border:1px solid #334155;border-radius:7px;background:#111827;padding:10px;color:#cbd5e1;font-size:12px;line-height:1.45;";
    const textarea = document.createElement("textarea");
    textarea.style.cssText = "min-height:220px;resize:vertical;border:1px solid #3f3f46;border-radius:7px;background:#09090b;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const setExample = () => {
      if (modeSelect.value === "durations") {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Durations</strong><br>One scene duration per line. Values can be seconds, <code>mm:ss</code>, or <code>hh:mm:ss</code>.`;
        if (!textarea.value.trim()) textarea.value = "4\n6.5\n3\n8\n5";
        actionSelect.disabled = false;
      } else if (modeSelect.value === "ranges") {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Start - End rows</strong><br>One exact scene range per line, like <code>0 - 4</code> or <code>00:04.00 - 00:08.50</code>. These are absolute timeline times.`;
        if (!textarea.value.trim()) textarea.value = "0 - 4\n4 - 8.5\n8.5 - 13\n13 - 20";
        actionSelect.value = "replace";
        actionSelect.disabled = true;
      } else {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Timestamp markers</strong><br>One marker per line. Scenes are created between each neighboring pair. Example: 0, 4, 8.5 creates two scenes.`;
        if (!textarea.value.trim()) textarea.value = "0\n4\n8.5\n13\n20";
        actionSelect.value = "replace";
        actionSelect.disabled = true;
      }
    };
    const preview = document.createElement("div");
    preview.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;color:#a5f3fc;font-size:12px;white-space:pre-wrap;min-height:42px;";
    const updatePreview = () => {
      try {
        const appendStart = actionSelect.value === "append" ? Number(state.segments[state.segments.length - 1]?.end || 0) : 0;
        const timings = parseBulkSegmentTimings(textarea.value, modeSelect.value, appendStart);
        const first = timings[0];
        const last = timings[timings.length - 1];
        preview.textContent = `Ready: ${timings.length} scene${timings.length === 1 ? "" : "s"} | ${formatTime(first.start)} - ${formatTime(last.end)} | total ${(last.end - first.start).toFixed(2)}s`;
      } catch (error) {
        preview.textContent = `Preview: ${String(error?.message || error)}`;
      }
    };
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px;border-top:1px solid #1f2937;";
    const cancel = makeButton("Cancel");
    const apply = makeButton("Apply Bulk Segments", "primary");
    actions.append(cancel, apply);
    body.append(explanation, controls, examples, textarea, preview);
    box.append(header, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const closeModal = () => backdrop.remove();
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    modeSelect.onchange = () => {
      textarea.value = "";
      setExample();
      updatePreview();
    };
    actionSelect.onchange = updatePreview;
    textarea.addEventListener("input", updatePreview);
    apply.onclick = async () => {
      try {
        const appendStart = actionSelect.value === "append" ? Number(state.segments[state.segments.length - 1]?.end || 0) : 0;
        const timings = parseBulkSegmentTimings(textarea.value, modeSelect.value, appendStart);
        await applyBulkSegmentTimings(timings, actionSelect.value);
        toast(`Created ${timings.length} manual scene${timings.length === 1 ? "" : "s"}.`);
        closeModal();
      } catch (error) {
        toast(String(error?.message || error), true);
        updatePreview();
      }
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
    setExample();
    updatePreview();
  }

  async function addOverlaySegment() {
    const duration = 4;
    const start = Math.max(0, snapTimeToBeat(currentGlobalTime()));
    const end = Math.min(Math.max(start + duration, start + 0.1), Math.max(start + duration, timelineDuration() || start + duration));
    const segment = newSegment(start, end);
    segment.track = "overlay";
    segment.source = "overlay";
    segment.label = `Insert ${state.overlaySegments.length + 1}`;
    pushHistory();
    state.overlaySegments.push(segment);
    sortSegments(state.overlaySegments);
    state.duration = Math.max(Number(state.duration || 0), end);
    setActiveSegment(segment);
    await autoSaveSessionQuiet("insert segment added");
  }

  async function deleteSegment() {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    if (segmentTrack(segment) === "overlay") {
      state.overlaySegments = state.overlaySegments.filter((item) => item.id !== segment.id);
      state.activeId = state.overlaySegments[0]?.id || state.segments[0]?.id || "";
    } else {
      state.segments = state.segments.filter((item) => item.id !== segment.id);
      state.activeId = state.segments[0]?.id || state.overlaySegments[0]?.id || "";
    }
    state.activeTrack = segmentTrack(activeSegment());
    syncInspector();
    render();
    await syncPromptJsonFromSegments("segment deleted");
    await syncI2VMotionJsonFromSegments("segment deleted");
    autoSaveSessionQuiet("segment deleted");
  }

  async function deleteSelectedMedia() {
    const media = selectedMediaForDelete();
    if (!media.segment || !media.path) {
      toast("No selected image or video to delete.", true);
      return;
    }
    const ok = await confirmDeleteMediaAction(media.type, media.path);
    if (!ok) return;
    try {
      deleteSelectedMediaButton.disabled = true;
      deleteSelectedMediaButton.textContent = "Deleting...";
      await postJson("/vrgdg/music_builder/delete_project_media", {
        project_folder: projectInput.value,
        path: media.path,
      });
      pushHistory();
      if (media.type === "video") {
        media.segment.video_history = (media.segment.video_history || []).filter((item) => item !== media.path);
        media.segment.video_history_index = Math.min(Math.max(0, Number(media.segment.video_history_index || 0)), media.segment.video_history.length - 1);
        if (media.segment.video_history_index < 0) media.segment.video_history_index = -1;
        if (media.segment.video_path === media.path) {
          media.segment.video_path = media.segment.video_history[media.segment.video_history_index] || "";
          media.segment.video_source_path = "";
          media.segment.video_output = null;
        }
        if (!media.segment.video_path) media.segment.video_status = "none";
        media.segment.preview_mode = media.segment.image_history?.length ? "image" : "video";
      } else {
        media.segment.image_history = (media.segment.image_history || []).filter((item) => item !== media.path);
        media.segment.image_history_index = Math.min(Math.max(0, Number(media.segment.image_history_index || 0)), media.segment.image_history.length - 1);
        if (media.segment.image_history_index < 0) media.segment.image_history_index = -1;
        if (media.segment.approved_image_path === media.path) media.segment.approved_image_path = media.segment.image_history[media.segment.image_history_index] || "";
        if (media.segment.custom_image_path === media.path) media.segment.custom_image_path = "";
        media.segment.preview_mode = "image";
      }
      syncPreview(media.segment);
      render();
      await autoSaveSessionQuiet(`${media.type} deleted`);
      toast(`Deleted ${media.type} from project.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      deleteSelectedMediaButton.disabled = false;
      updateSelectedMediaTools();
    }
  }

  function sendPromptToEnhance(sourceLabel, promptText) {
    const segment = requireActiveSegment();
    if (!segment) return;
    const prompt = String(promptText || "").trim();
    if (!prompt) {
      toast(`${sourceLabel} prompt is empty.`, true);
      return;
    }
    pushHistory();
    segment.enhance_prompt = prompt;
    zEnhancePromptPreview.value = prompt;
    syncZEnhanceSettingsPanel();
    toast(`${sourceLabel} prompt sent to Enhance.`);
  }

  function timestampForProjectName() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
  }

  function resetProjectState(projectFolder, sessionPath = "", srtPath = "") {
    pauseAllAudio();
    audio.removeAttribute("src");
    sceneAudio.removeAttribute("src");
    const cleanProjectFolder = String(projectFolder || "").trim();
    const contextPath = (filename) => {
      const folder = cleanProjectFolder.replace(/[\\/]+$/, "");
      if (!folder) return "";
      const separator = folder.includes("\\") ? "\\" : "/";
      return `${folder}${separator}project_context${separator}${filename}`;
    };
    state.duration = 0;
    state.peaks = [];
    state.beats = [];
    setBeatMarkersVisible(false);
    state.srtMode = false;
    state.timingFrozen = false;
    state.promptJsonPath = contextPath("ConceptPrompts.txt");
    state.i2vMotionJsonPath = contextPath("I2VMotionNotes.txt");
    state.imageTriggerPhrase = "";
    state.videoTriggerPhrase = "";
    state.themeStylePath = contextPath("themestyle.txt");
    state.storyIdeaPath = contextPath("storyconcept.txt");
    state.subjectScenePath = contextPath("subjectsandscenes.txt");
    state.useVrgdgTextContext = true;
    state.projectFolder = cleanProjectFolder;
    state.sessionPath = sessionPath || "";
    state.srtPath = srtPath || "";
    state.segments = [newSegment(0, 4)];
    state.overlaySegments = [];
    state.activeTrack = "base";
    state.activeId = state.segments[0]?.id || "";
    state.sceneAudioMode = false;
    state.sceneAudioSegmentId = "";
    state.sceneAudioGlobalTime = 0;
    state.zimageSettings = defaultZImageSettings();
    state.fluxKleinSettings = defaultFluxKleinSettings();
    state.ernieImageSettings = defaultErnieImageSettings();
    state.useFluxGlobalImageIngredients = false;
    state.fluxGlobalImageIngredients = [];
    state.fluxReferenceBuilder = defaultFluxReferenceBuilder();
    state.zEnhanceSettings = defaultZEnhanceSettings();
    state.videoModelMode = "i2v";
    state.i2vVideoSettings = defaultI2VVideoSettings();
    state.promptToolsHintPrefs = {};
    projectInput.value = state.projectFolder;
    srtInput.value = state.srtPath;
    setWidgetValue(node, "audio_path", "");
    setWidgetValue(node, "project_folder", state.projectFolder);
    setWidgetValue(node, "session_path", state.sessionPath);
    setWidgetValue(node, "srt_path", state.srtPath);
    audioInput.value = "";
    promptJsonInput.value = state.promptJsonPath;
    i2vMotionJsonInput.value = state.i2vMotionJsonPath;
    imageTriggerInput.value = "";
    ernieImageTriggerInput.value = "";
    fluxImageTriggerInput.value = "";
    videoTriggerInput.value = "";
    themeStyleInput.value = state.themeStylePath;
    storyIdeaInput.value = state.storyIdeaPath;
    subjectSceneInput.value = state.subjectScenePath;
    useVrgdgTextContext.input.checked = true;
    state.undoStack = [];
    state.redoStack = [];
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncZEnhanceSettingsPanel();
    syncI2VVideoSettingsPanel();
    syncVideoModePanel();
    syncInspector();
    render();
  }

  async function newProject() {
    const defaultName = `VRGDG_Project_${timestampForProjectName()}`;
    const projectName = await showTextInputModal({
      title: "New Project",
      label: "Project name or full project folder path",
      value: defaultName,
      confirmLabel: "Create Project",
    });
    if (projectName === null) return false;
    try {
      const data = await postJson("/vrgdg/music_builder/new_project", {
        project_folder: projectName,
      }, 60000);
      resetProjectState(data.project_folder || "", data.session_path || "", data.srt_path || "");
      await loadGlobalModelDefaultsQuiet();
      if (data.concept_prompts_path) {
        promptJsonInput.value = data.concept_prompts_path;
        state.promptJsonPath = data.concept_prompts_path;
      }
      if (data.i2v_motion_notes_path) {
        i2vMotionJsonInput.value = data.i2v_motion_notes_path;
        state.i2vMotionJsonPath = data.i2v_motion_notes_path;
      }
      if (data.theme_style_path) {
        themeStyleInput.value = data.theme_style_path;
        state.themeStylePath = data.theme_style_path;
      }
      if (data.story_idea_path) {
        storyIdeaInput.value = data.story_idea_path;
        state.storyIdeaPath = data.story_idea_path;
      }
      if (data.subject_scene_path) {
        subjectSceneInput.value = data.subject_scene_path;
        state.subjectScenePath = data.subject_scene_path;
      }
      rememberLastProject(state.projectFolder);
      await saveSession({ quiet: true });
      toast(`New project created.\n${state.projectFolder}`);
      return true;
    } catch (error) {
      toast(String(error?.message || error), true);
      return false;
    }
  }

  function openPromptCreatorPanel() {
    const creator = window.VRGDGMusicVideoPromptCreator;
    if (!creator?.open) {
      toast("Prompt Creator UI is not loaded yet. Refresh ComfyUI and try again.", true);
      return;
    }
    creator.open({
      projectFolder: projectInput.value || state.projectFolder || "",
      onSaved: (result) => {
        if (result?.project_folder) {
          projectInput.value = result.project_folder;
          state.projectFolder = result.project_folder;
          setWidgetValue(node, "project_folder", state.projectFolder);
        }
        if (result?.srt_path) {
          srtInput.value = result.srt_path;
          state.srtPath = result.srt_path;
          setWidgetValue(node, "srt_path", state.srtPath);
        }
        if (result?.files) {
          promptJsonInput.value = result.files["ConceptPrompts.txt"] || promptJsonInput.value;
          i2vMotionJsonInput.value = result.files["I2VMotionNotes.txt"] || i2vMotionJsonInput.value;
          themeStyleInput.value = result.files["themestyle.txt"] || themeStyleInput.value;
          storyIdeaInput.value = result.files["storyconcept.txt"] || storyIdeaInput.value;
          subjectSceneInput.value = result.files["subjectsandscenes.txt"] || subjectSceneInput.value;
          state.promptJsonPath = promptJsonInput.value;
          state.i2vMotionJsonPath = i2vMotionJsonInput.value;
          state.themeStylePath = themeStyleInput.value;
          state.storyIdeaPath = storyIdeaInput.value;
          state.subjectScenePath = subjectSceneInput.value;
        }
      },
      onSendToVideoCreator: async (result) => {
        const sourceProject = String(result?.project_folder || "").trim();
        const currentProject = String(projectInput.value || state.projectFolder || "").trim();
        if (sourceProject && (!currentProject || isBlankStarterProject())) {
          projectInput.value = sourceProject;
          state.projectFolder = sourceProject;
          setWidgetValue(node, "project_folder", state.projectFolder);
        }
        await autoLoadAll({ sourceProjectFolder: result?.project_folder || "", throwOnError: true });
      },
    });
  }

  async function saveProjectAs() {
    const currentProject = String(projectInput.value || state.projectFolder || "").trim();
    if (!currentProject) {
      toast("Create or load a project before using Save Project As.", true);
      return;
    }
    const currentName = currentProject.split(/[\\/]/).filter(Boolean).pop() || "VRGDG_Project";
    const targetProject = await showSaveProjectAsModal(`${currentName}_${timestampForProjectName()}`);
    if (targetProject === null) return;
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      const data = await postJson("/vrgdg/music_builder/save_project_as", {
        source_project_folder: currentProject,
        target_project_folder: targetProject,
        audio_path: audioInput.value,
        session: currentSessionData(),
      }, 120000);
      await loadSessionFromProject(data.project_folder || targetProject);
      toast(`Project saved as.\n${state.projectFolder}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  function showRemakeModeComingSoon() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.58);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(420px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const title = document.createElement("div");
    title.textContent = "Remake Mode";
    title.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const body = document.createElement("div");
    body.textContent = "Coming soon. This will let you select scenes and remake images, videos, or both.";
    body.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const close = makeButton("Close");
    close.onclick = () => backdrop.remove();
    box.append(title, body, close);
    backdrop.append(box);
    document.body.append(backdrop);
  }

  function confirmLongBatchAction({ title, lines = [], confirmLabel = "Continue" } = {}) {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = title || "Start Batch Process?";
      heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
      const body = document.createElement("div");
      body.style.cssText = "display:flex;flex-direction:column;gap:8px;font-size:13px;color:#d4d4d8;line-height:1.45;";
      for (const line of lines) {
        const item = document.createElement("div");
        item.textContent = line;
        body.append(item);
      }
      const note = document.createElement("div");
      note.textContent = "This can take a long time. You can use Stop if you need to interrupt it.";
      note.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;color:#fde68a;font-size:12px;";
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const confirm = makeButton(confirmLabel, "primary");
      cancel.onclick = () => {
        backdrop.remove();
        resolve(false);
      };
      confirm.onclick = () => {
        backdrop.remove();
        resolve(true);
      };
      actions.append(cancel, confirm);
      box.append(heading, body, note, actions);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  function chooseBatchModeAction({ title, intro = "", choices = [], confirmLabel = "Continue", extraGroups = [], returnAll = false, defaultValue = "" } = {}) {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(680px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = title || "Choose Batch Mode";
      heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
      const text = document.createElement("div");
      text.textContent = intro || "";
      text.style.cssText = `display:${intro ? "block" : "none"};font-size:13px;color:#d4d4d8;line-height:1.45;`;
      let selected = choices.some((choice) => choice.value === defaultValue) ? defaultValue : choices[0]?.value || "";
      const extraSelected = {};
      const list = document.createElement("div");
      list.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:420px;overflow:auto;";
      choices.forEach((choice, index) => {
        const id = `vrgdg_batch_choice_${Date.now()}_${index}`;
        const label = document.createElement("label");
        label.htmlFor = id;
        label.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:10px;align-items:start;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;cursor:pointer;";
        const input = document.createElement("input");
        input.type = "radio";
        input.name = "vrgdg_batch_choice";
        input.id = id;
        input.value = choice.value;
        input.checked = choice.value === selected;
        input.style.marginTop = "3px";
        input.onchange = () => {
          if (input.checked) selected = choice.value;
        };
        const copy = document.createElement("div");
        const name = document.createElement("div");
        name.textContent = choice.label || choice.value;
        name.style.cssText = "font-weight:900;color:#f8fafc;font-size:13px;";
        const desc = document.createElement("div");
        desc.textContent = choice.description || "";
        desc.style.cssText = "margin-top:4px;color:#cbd5e1;font-size:12px;line-height:1.45;";
        copy.append(name, desc);
        label.append(input, copy);
        label.onclick = () => {
          input.checked = true;
          selected = choice.value;
        };
        list.append(label);
      });
      for (const group of extraGroups) {
        const groupBox = document.createElement("div");
        groupBox.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;";
        const groupTitle = document.createElement("div");
        groupTitle.textContent = group.label || "Options";
        groupTitle.style.cssText = "font-weight:900;color:#cffafe;font-size:13px;";
        const groupNote = document.createElement("div");
        groupNote.textContent = group.description || "";
        groupNote.style.cssText = `display:${group.description ? "block" : "none"};color:#cbd5e1;font-size:12px;line-height:1.45;`;
        const groupChoices = document.createElement("div");
        groupChoices.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;";
        const key = group.key || `extra_${Object.keys(extraSelected).length}`;
        extraSelected[key] = group.choices?.[0]?.value || "";
        (group.choices || []).forEach((choice, index) => {
          const id = `vrgdg_batch_extra_${key}_${Date.now()}_${index}`;
          const label = document.createElement("label");
          label.htmlFor = id;
          label.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;align-items:start;border:1px solid #334155;border-radius:7px;background:#111827;padding:9px;cursor:pointer;";
          const input = document.createElement("input");
          input.type = "radio";
          input.name = `vrgdg_batch_extra_${key}`;
          input.id = id;
          input.value = choice.value;
          input.checked = index === 0;
          input.style.marginTop = "3px";
          input.onchange = () => {
            if (input.checked) extraSelected[key] = choice.value;
          };
          const copy = document.createElement("div");
          const name = document.createElement("div");
          name.textContent = choice.label || choice.value;
          name.style.cssText = "font-weight:900;color:#f8fafc;font-size:12px;";
          const desc = document.createElement("div");
          desc.textContent = choice.description || "";
          desc.style.cssText = `display:${choice.description ? "block" : "none"};margin-top:3px;color:#cbd5e1;font-size:11px;line-height:1.35;`;
          copy.append(name, desc);
          label.append(input, copy);
          label.onclick = () => {
            input.checked = true;
            extraSelected[key] = choice.value;
          };
          groupChoices.append(label);
        });
        groupBox.append(groupTitle, groupNote, groupChoices);
        list.append(groupBox);
      }
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const confirm = makeButton(confirmLabel, "primary");
      cancel.onclick = () => {
        backdrop.remove();
        resolve("");
      };
      confirm.onclick = () => {
        backdrop.remove();
        resolve(returnAll ? { mode: selected, ...extraSelected } : selected);
      };
      actions.append(cancel, confirm);
      box.append(heading, text, list, actions);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  function openGemmaRunnerModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(640px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Gemma Runner</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">LM Studio is used for text-only Gemma steps. Vision/image-reference Gemma stays on the built-in GGUF runner for now.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);
    const runner = makeSelect(["builtin", "lm_studio"], state.textGemmaRunner || "builtin");
    const baseUrl = makeInput(state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1");
    const model = makeInput(state.lmStudioModel || "");
    const modelSelect = makeSelect([""], "");
    const loadModels = makeButton("Load LM Studio Models");
    const modelPickerRow = document.createElement("div");
    modelPickerRow.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    const apiKey = makeInput(state.lmStudioApiKey || "", "password");
    const lmPanel = document.createElement("div");
    lmPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const note = document.createElement("div");
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    note.textContent = "In LM Studio, load your Gemma GGUF model, open the Local Server tab, start the server, then copy the model name shown there. No extra Python install is needed.";
    const test = makeButton("Test LM Studio", "primary");
    modelSelect.onchange = () => {
      if (modelSelect.value) model.value = modelSelect.value;
    };
    loadModels.onclick = async () => {
      loadModels.disabled = true;
      loadModels.textContent = "Loading...";
      try {
        const data = await postJson("/vrgdg/music_builder/lm_studio_models", {
          lmstudio_base_url: baseUrl.value || "http://127.0.0.1:1234/v1",
          lmstudio_api_key: apiKey.value || "",
        }, 45000);
        const allIds = Array.isArray(data?.models) ? data.models.map((item) => String(item || "").trim()).filter(Boolean) : [];
        const ids = allIds.filter((id) => !isLikelyEmbeddingModelId(id));
        if (!allIds.length) throw new Error("LM Studio returned no models. Load a chat model in LM Studio and make sure the local server is running.");
        if (!ids.length) throw new Error("LM Studio only returned embedding models. Load a chat/text-generation model, then click Load LM Studio Models again.");
        modelSelect.innerHTML = "";
        ids.forEach((id) => {
          const option = document.createElement("option");
          option.value = id;
          option.textContent = id;
          modelSelect.append(option);
        });
        const current = String(model.value || "").trim();
        if (current && ids.includes(current)) modelSelect.value = current;
        else {
          modelSelect.value = ids[0];
          model.value = ids[0];
        }
        toast(`Loaded ${ids.length} LM Studio model${ids.length === 1 ? "" : "s"}.`);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        loadModels.disabled = false;
        loadModels.textContent = "Load LM Studio Models";
      }
    };
    modelPickerRow.append(makeField("Available LM Studio models", modelSelect), loadModels);
    lmPanel.append(
      note,
      makeField("LM Studio base URL", baseUrl),
      modelPickerRow,
      makeField("LM Studio model name", model),
      makeField("API key (usually blank for local LM Studio)", apiKey),
      test,
    );
    const syncVisibility = () => {
      lmPanel.style.display = runner.value === "lm_studio" ? "flex" : "none";
    };
    runner.onchange = syncVisibility;
    test.onclick = async () => {
      state.textGemmaRunner = "lm_studio";
      state.lmStudioBaseUrl = baseUrl.value || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = model.value || "";
      state.lmStudioApiKey = apiKey.value || "";
      let progress = null;
      try {
        progress = createProgressWindow("Testing LM Studio");
        progress.set("Sending a tiny text prompt to LM Studio...", 35);
        const data = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
          ...textGemmaRunnerPayload(),
          model_file: t2iTextGemmaModelSelect.value || "",
          reference_type: "location",
          source_text: "small empty test room",
          style_theme: "",
          max_new_tokens: 120,
        }, 60000);
        progress.set(`LM Studio responded successfully:\n${data.prompt}`, 100);
        progress.close(4000);
        toast("LM Studio text runner works.");
      } catch (error) {
        progress?.set(`LM Studio test failed:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      }
    };
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Save Runner", "primary");
    actions.append(cancel, save);
    box.append(header, makeField("Text Gemma runner", runner), lmPanel, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    syncVisibility();
    close.onclick = cancel.onclick = () => backdrop.remove();
    save.onclick = async () => {
      state.textGemmaRunner = runner.value || "builtin";
      state.lmStudioBaseUrl = baseUrl.value || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = model.value || "";
      state.lmStudioApiKey = apiKey.value || "";
      await autoSaveSessionQuiet("Gemma runner settings");
      toast(state.textGemmaRunner === "lm_studio" ? "Gemma text runner set to LM Studio." : "Gemma text runner set to built-in GGUF.");
      backdrop.remove();
    };
  }

  function confirmDeleteMediaAction(type, path) {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = `Delete selected ${type}?`;
      heading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
      const body = document.createElement("div");
      body.textContent = `This deletes the file from the current project folder and removes it from this scene history.`;
      body.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
      const pathBox = document.createElement("div");
      pathBox.textContent = path;
      pathBox.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;";
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const confirm = makeButton(`Delete ${type}`, "danger");
      confirm.style.borderColor = "#7f1d1d";
      confirm.style.background = "#991b1b";
      confirm.style.color = "#fee2e2";
      cancel.onclick = () => {
        backdrop.remove();
        resolve(false);
      };
      confirm.onclick = () => {
        backdrop.remove();
        resolve(true);
      };
      actions.append(cancel, confirm);
      box.append(heading, body, pathBox, actions);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  function askExistingSceneVideoAction(segment, sceneIndex) {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = "This scene already has a video";
      heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
      const body = document.createElement("div");
      body.style.cssText = "display:flex;flex-direction:column;gap:8px;font-size:13px;color:#d4d4d8;line-height:1.45;";
      const scene = document.createElement("div");
      scene.textContent = `${sceneDisplayName(segment, sceneIndex)} already has a rendered clip.`;
      const path = document.createElement("div");
      path.textContent = String(segment?.video_path || "");
      path.style.cssText = "border:1px solid #303038;border-radius:6px;background:#18181b;padding:8px;color:#a5f3fc;overflow-wrap:anywhere;font-size:12px;";
      const note = document.createElement("div");
      note.textContent = "Choose Backup and replace to keep the old clip, or Overwrite to replace it without saving a backup.";
      note.style.cssText = "color:#fde68a;";
      body.append(scene, path, note);
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const overwrite = makeButton("Overwrite");
      const backup = makeButton("Backup and replace", "primary");
      cancel.onclick = () => {
        backdrop.remove();
        resolve("cancel");
      };
      overwrite.onclick = () => {
        backdrop.remove();
        resolve("overwrite");
      };
      backup.onclick = () => {
        backdrop.remove();
        resolve("backup");
      };
      actions.append(cancel, overwrite, backup);
      box.append(heading, body, actions);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  async function confirmAndRunZImageAll() {
    const imageMode = state.imageModelMode || "zimage";
    const useFluxKleinMode = imageMode === "flux_klein";
    const useNBMode = imageMode === "nano_banana";
    const useErnieMode = imageMode === "ernie_image";
    const modelLabel = useFluxKleinMode ? "Flux/Klein" : useNBMode ? "NanoBanana" : useErnieMode ? "Ernie" : "ZImage";
    const imageModeChoices = [
      {
        value: "zimage",
        label: "ZImage",
        description: "Use the ZImage image workflow. Does not require Nano B reference images.",
      },
      {
        value: "flux_klein",
        label: "Flux/Klein",
        description: "Use Flux/Klein and its image ingredients/reference images.",
      },
      {
        value: "nano_banana",
        label: "Nano B",
        description: "Use Nano B and its required reference images/API key.",
      },
      {
        value: "ernie_image",
        label: "Ernie",
        description: "Use the Ernie image workflow.",
      },
    ];
    const currentChoice = imageModeChoices.find((choice) => choice.value === imageMode) || imageModeChoices[0];
    const orderedImageModeChoices = [
      currentChoice,
      ...imageModeChoices.filter((choice) => choice.value !== currentChoice.value),
    ];
    const action = await chooseBatchModeAction({
      title: "Run Image All?",
      intro: `Image All only works on the image stage. It does not create I2V prompts, render videos, or stitch the final video. Current image model: ${modelLabel}. Flux ingredients, model selections, LoRAs, notes, and project paths are not reset.`,
      confirmLabel: "Run Image All",
      returnAll: true,
      choices: [
        {
          value: "resume_missing",
          label: "Resume missing images",
          description: "Safe resume mode. Keep existing image prompts and selected images. Only run Gemma/create images for scenes that do not already have an image.",
        },
        {
          value: "keep_prompts_redo_images",
          label: "Keep prompts, redo images",
          description: "Keep saved ZImage/Ernie/Flux prompts, randomize image seeds, and create a new image version for every scene.",
        },
        {
          value: "redo_prompts_images",
          label: "Redo image prompts and images",
          description: "Regenerate image prompts with Gemma, randomize image seeds, and create a new image version for every scene.",
        },
      ],
      extraGroups: [
        {
          key: "imageMode",
          label: "Image model to run",
          description: "This explicit choice controls which Image All pipeline runs.",
          choices: orderedImageModeChoices,
        },
      ],
    });
    if (!action?.mode) return;
    const selectedImageMode = ["zimage", "flux_klein", "nano_banana", "ernie_image"].includes(action.imageMode) ? action.imageMode : imageMode;
    state.imageModelMode = selectedImageMode;
    state.fluxKleinSettings.image_model_mode = selectedImageMode;
    state.fluxKleinSettings.enabled = selectedImageMode === "flux_klein";
    syncFluxKleinPanel();
    if (selectedImageMode === "flux_klein") await fluxKleinAllScenes({ imageRunMode: action.mode });
    else if (selectedImageMode === "nano_banana") await nbImageAllScenes({ imageRunMode: action.mode });
    else if (selectedImageMode === "ernie_image") await ernieImageAllScenes({ imageRunMode: action.mode });
    else await zImageAllScenes({ imageRunMode: action.mode });
  }

  async function confirmAndRunGemmaT2IAll() {
    const imageMode = state.imageModelMode || "zimage";
    const modelLabel = imageMode === "flux_klein" ? "Flux/Klein" : imageMode === "nano_banana" ? "NanoBanana" : imageMode === "ernie_image" ? "Ernie" : "ZImage";
    const mode = await chooseBatchModeAction({
      title: "Run Gemma T2I All?",
      intro: `This only creates text-to-image prompts for review. It will not create images, videos, or the final stitched video. Current image model: ${modelLabel}.`,
      confirmLabel: "Run Gemma T2I All",
      choices: [
        {
          value: "missing_only",
          label: "Missing prompts only",
          description: "Keep existing T2I/Flux prompts. Only run Gemma for scenes with no saved image prompt.",
        },
        {
          value: "redo_all",
          label: "Redo all T2I prompts",
          description: "Replace every scene's saved T2I/Flux prompt with a fresh Gemma prompt. Images and videos stay untouched.",
        },
      ],
    });
    if (mode) await gemmaT2IAllScenes({ promptRunMode: mode });
  }

  async function confirmAndRunGemmaVideoAll() {
    const videoLabel = currentVideoMode() === "t2v" ? "T2V" : "I2V";
    const targets = allEditableSegments().filter((segment) => !String(segment.i2v_prompt || "").trim());
    const hasVisionTargets = targets.length
      ? targets.some((segment) => videoVisionReferenceEnabled(segment))
      : allEditableSegments().some((segment) => videoVisionReferenceEnabled(segment));
    const textChoices = [
      {
        value: "missing_text",
        label: "Missing, text only",
        description: `Keep existing ${videoLabel} prompts. Use the selected text runner and T2I/concept prompt plus motion notes for scenes with no saved video prompt.`,
      },
      {
        value: "redo_text",
        label: `Redo all, text only`,
        description: "Replace every scene's saved video prompt using the selected text runner. Images and videos stay untouched.",
      },
    ];
    const visionChoices = [
      {
        value: "missing_vision",
        label: "Missing, vision",
        description: `Keep existing ${videoLabel} prompts. Use the built-in vision GGUF to look at each selected scene/reference image plus motion notes for missing prompts.`,
      },
      {
        value: "redo_vision",
        label: `Redo all, vision`,
        description: "Replace every scene's saved video prompt using the built-in vision GGUF and each scene/reference image. Images and videos stay untouched.",
      },
    ];
    const action = await chooseBatchModeAction({
      title: `Run Gemma ${videoLabel} All?`,
      intro: `This only creates ${videoLabel} prompts for review. It will not render videos or stitch the final video. ${hasVisionTargets ? "Image-reference scenes were detected, so vision options are listed first." : "No image-reference scenes were detected, so text-only options are listed first."}`,
      confirmLabel: `Run Gemma ${videoLabel} All`,
      defaultValue: hasVisionTargets ? "missing_vision" : "missing_text",
      choices: hasVisionTargets ? [...visionChoices, ...textChoices] : [...textChoices, ...visionChoices],
    });
    if (!action) return;
    await gemmaVideoAllTextOnly({
      promptRunMode: action.startsWith("missing") ? "missing_only" : "redo_all",
      gemmaInputMode: action.endsWith("vision") ? "vision" : "text",
    });
  }

  async function confirmAndRunRenderAll() {
    const videoLabel = currentVideoMode() === "t2v" ? "T2V" : "I2V";
    const ok = await confirmLongBatchAction({
      title: "Run Render All?",
      lines: [
        "Render All only works on the video/render stage.",
        currentVideoMode() === "t2v"
          ? "It uses existing T2V prompts and does not require scene images."
          : "It uses the current selected images and existing I2V prompts.",
        "Scenes that already have a selected video are skipped.",
        `Scenes missing video are rendered with ${videoLabel}, then the final video is stitched.`,
        "If every scene already has video, this only stitches the final video.",
      ],
      confirmLabel: "Run Render All",
    });
    if (ok) await renderAllScenes();
  }

  async function confirmAndRunFullBuild() {
    const t2vMode = currentVideoMode() === "t2v";
    const options = await chooseBatchModeAction({
      title: "Build Full Video?",
      intro: t2vMode
        ? "Build Full Video is in Text-to-Video mode. It skips image generation, creates T2V prompts, renders scene videos, and stitches the final video. Model selections, LoRAs, notes, and project paths are not reset."
        : "Build Full Video can run the whole pipeline: image prompts, images, I2V prompts, scene videos, and final stitching. Choose how much to regenerate. Flux ingredients, model selections, LoRAs, notes, and project paths are not reset.",
      confirmLabel: "Build Full Video",
      returnAll: true,
      choices: [
        {
          value: "resume_missing",
          label: "Resume missing only",
          description: "Safest resume mode. Keep existing prompts, selected images, and selected videos. Only create whatever is missing, then stitch the final video.",
        },
        {
          value: "fresh_rebuild",
          label: "Fresh full rebuild",
          description: t2vMode
            ? "Start fresh for generated video outputs. Regenerate T2V prompts and videos. Video seeds can be randomized below."
            : "Start fresh for generated outputs. Regenerate image prompts, images, I2V prompts, and videos. Image seeds are randomized.",
        },
        {
          value: "redo_i2v_prompts_videos",
          label: t2vMode ? "Redo T2V prompts and videos" : "Keep images, redo I2V prompts and videos",
          description: t2vMode
            ? "Regenerate T2V prompts with Gemma, then create new video versions."
            : "Use the current selected images. Regenerate I2V prompts with Gemma, then create new video versions.",
        },
        {
          value: "redo_videos",
          label: t2vMode ? "Keep prompts, redo videos" : "Keep images and prompts, redo videos",
          description: t2vMode
            ? "Use existing T2V prompts. Only create new video versions, then stitch."
            : "Use the current selected images and existing I2V prompts. Only create new video versions, then stitch.",
        },
      ],
      extraGroups: [
        {
          key: "videoSeedMode",
          label: "Video seed behavior",
          description: "Used only when this build creates new scene videos.",
          choices: [
            {
              value: "keep",
              label: "Keep current video seed",
              description: "Best when you like the motion and only changed LoRAs or model settings.",
            },
            {
              value: "random",
              label: "Randomize video seed",
              description: "Best when you want new motion variations.",
            },
          ],
        },
      ],
    });
    if (options?.mode) await buildFullVideoPipeline({ buildMode: options.mode, videoSeedMode: options.videoSeedMode || "keep" });
  }

  for (const control of [labelInput, startInput, endInput, notesInput, ernieNotesInput, i2vNotesInput, t2iPrompt, ernieT2IPrompt, i2vPrompt, zEnhanceGemmaNotes, zEnhancePromptPreview]) {
    control.addEventListener("input", updateActiveFromInputs);
    control.addEventListener("change", updateActiveFromInputs);
  }
  freezeTimingControl.input.addEventListener("change", () => {
    pushHistory();
    state.timingFrozen = Boolean(freezeTimingControl.input.checked);
    syncInspector();
    render();
    toast(state.timingFrozen ? "Timing frozen." : "Timing unlocked for editing.");
  });
  promptJsonInput.addEventListener("input", () => {
    pushHistory();
    state.promptJsonPath = promptJsonInput.value || "";
  });
  i2vMotionJsonInput.addEventListener("input", () => {
    pushHistory();
    state.i2vMotionJsonPath = i2vMotionJsonInput.value || "";
  });
  imageTriggerInput.addEventListener("input", saveZImageSettingsFromPanel);
  ernieImageTriggerInput.addEventListener("input", saveErnieImageSettingsFromPanel);
  fluxImageTriggerInput.addEventListener("input", saveFluxKleinSettingsFromPanel);
  fluxUseTextOnlyGemmaPrompt.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  nbApiKey.addEventListener("input", saveNBImageSettingsFromPanel);
  nbModelSelect.addEventListener("change", saveNBImageSettingsFromPanel);
  nbUseTextOnlyGemmaPrompt.input.addEventListener("change", saveNBImageSettingsFromPanel);
  nbNotes.addEventListener("input", saveNBImageSettingsFromPanel);
  nbPrompt.addEventListener("input", saveNBImageSettingsFromPanel);
  videoTriggerInput.addEventListener("input", saveI2VVideoSettingsFromPanel);
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
    helpText: "Save this file to re-import the updated concept prompts into the scene notes.",
    afterSave: async () => {
      state.promptJsonPath = promptJsonInput.value || "";
      await importPromptJson();
      await autoSaveSessionQuiet("prompt JSON edited");
    },
  });
  editI2VMotionJsonButton.onclick = () => editContextTextFile(i2vMotionJsonInput, "Edit I2V Motion Notes JSON", "I2VMotionNotes.txt", null, {
    showGemma: false,
    helpText: "Save this file to re-import the updated I2V motion notes into the scene motion boxes.",
    afterSave: async () => {
      state.i2vMotionJsonPath = i2vMotionJsonInput.value || "";
      await importI2VMotionJson();
      await autoSaveSessionQuiet("I2V motion notes edited");
    },
  });
  useVisionReference.input.addEventListener("change", updateActiveFromInputs);
  ernieUseVisionReference.input.addEventListener("change", updateActiveFromInputs);
  useI2VVisionReference.input.addEventListener("change", updateActiveFromInputs);
  useT2VVisionReference.input.addEventListener("change", updateActiveFromInputs);
  useSceneZImageSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_zimage_settings = Boolean(useSceneZImageSettings.input.checked);
    if (segment.use_scene_zimage_settings && !segment.zimage_settings) {
      segment.zimage_settings = cloneZImageSettings(state.zimageSettings);
    }
    syncZImageSettingsPanel();
    renderList();
    toast(segment.use_scene_zimage_settings ? "This scene now has custom ZImage settings." : "This scene is using global ZImage settings again.");
  });
  useSceneErnieImageSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_ernie_image_settings = Boolean(useSceneErnieImageSettings.input.checked);
    if (segment.use_scene_ernie_image_settings && !segment.ernie_image_settings) {
      segment.ernie_image_settings = cloneErnieImageSettings(state.ernieImageSettings);
    }
    syncErnieImagePanel();
    renderList();
    toast(segment.use_scene_ernie_image_settings ? "This scene now has custom Ernie settings." : "This scene is using global Ernie settings again.");
  });
  useSceneFluxKleinSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_flux_klein_settings = Boolean(useSceneFluxKleinSettings.input.checked);
    if (segment.use_scene_flux_klein_settings && !segment.flux_klein_settings) {
      segment.flux_klein_settings = cloneFluxKleinSettings(state.fluxKleinSettings);
    }
    syncFluxKleinPanel();
    renderList();
    toast(segment.use_scene_flux_klein_settings ? "This scene now has custom Flux/Klein settings." : "This scene is using global Flux/Klein settings again.");
  });
  useSceneNBImageSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_nb_image_settings = Boolean(useSceneNBImageSettings.input.checked);
    if (segment.use_scene_nb_image_settings && !segment.nb_image_settings) {
      segment.nb_image_settings = cloneNBImageSettings(state.nbImageSettings);
    }
    syncNBImagePanel();
    renderList();
    toast(segment.use_scene_nb_image_settings ? "This scene now has custom NanoBanana settings." : "This scene is using global NanoBanana settings again.");
  });
  function setSceneI2VVideoSettingsEnabled(enabled) {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    saveI2VVideoSettingsFromPanel();
    segment.use_scene_i2v_video_settings = Boolean(enabled);
    if (segment.use_scene_i2v_video_settings && !segment.i2v_video_settings) {
      segment.i2v_video_settings = cloneI2VVideoSettings(state.i2vVideoSettings);
    }
    syncI2VVideoSettingsPanel();
    renderList();
    toast(segment.use_scene_i2v_video_settings ? "This scene now has custom video models, settings, and LoRAs." : "This scene is using global video models, settings, and LoRAs again.");
  }
  useSceneI2VVideoSettings.input.addEventListener("change", () => {
    setSceneI2VVideoSettingsEnabled(Boolean(useSceneI2VVideoSettings.input.checked));
  });
  refImageInput.addEventListener("input", updateActiveFromInputs);
  refImageInput.addEventListener("change", updateActiveFromInputs);
  menuButton.onclick = (event) => {
    event.stopPropagation();
    menuDropdown.style.display = menuDropdown.style.display === "flex" ? "none" : "flex";
  };
  menuDropdown.addEventListener("click", (event) => {
    if (event.target === autoSaveControl.input || autoSaveControl.wrapper.contains(event.target)) return;
    menuDropdown.style.display = "none";
  });
  window.addEventListener("pointerdown", (event) => {
    if (menuDropdown.style.display !== "flex") return;
    if (menuDropdown.contains(event.target) || menuButton.contains(event.target)) return;
    menuDropdown.style.display = "none";
  });
  loadButton.onclick = () => {
    if (loadButton.disabled) return;
    loadAudio();
  };
  settingsButton.onclick = openSettingsModal;
  newProjectButton.onclick = async () => {
    const mode = await window.VRGDGMusicVideoPromptCreator?.chooseNewProjectMode?.();
    if (!mode) return;
    const created = await newProject();
    if (created && mode === "prompt_creator") openPromptCreatorPanel();
  };
  saveProjectAsButton.onclick = saveProjectAs;
  autoSaveControl.input.addEventListener("change", () => {
    state.autoSaveEnabled = Boolean(autoSaveControl.input.checked);
    if (state.projectFolder) {
      saveSession({ quiet: true });
    }
  });
  undoButton.onclick = undo;
  redoButton.onclick = redo;
  loadSrtButton.onclick = loadSrt;
  loadSessionButton.onclick = loadSession;
  loadLastProjectButton.onclick = loadLastProject;
  promptCreatorButton.onclick = openPromptCreatorPanel;
  fluxReferenceBuilderButton.onclick = openFluxReferenceBuilderModal;
  promptOptionsButton.onclick = openPromptOptionsModal;
  gemmaRunnerButton.onclick = openGemmaRunnerModal;
  autoLoadAllButton.onclick = autoLoadAll;
  clearMemoryButton.onclick = runClearMemoryWorkflow;
  renderAllButton.onclick = confirmAndRunRenderAll;
  stitchPreviewButton.onclick = openStitchPreviewModal;
  gemmaT2IAllButton.onclick = confirmAndRunGemmaT2IAll;
  gemmaVideoAllButton.onclick = confirmAndRunGemmaVideoAll;
  zImageAllButton.onclick = confirmAndRunZImageAll;
  fullBuildButton.onclick = confirmAndRunFullBuild;
  remakeModeButton.onclick = showRemakeModeComingSoon;
  stopWorkflowButton.onclick = stopCurrentWorkflow;
  downloadModelsButton.onclick = showModelDownloadModal;
  fullscreenButton.onclick = () => applyBuilderFullscreen(!builderFullscreen);
  openSceneAudioOptionsButton.onclick = () => {
    const segment = requireActiveSegment();
    if (segment) openSceneOptions(segment);
  };
  pickAudioButton.onclick = () => projectAudioFileInput.click();
  pickSrtButton.onclick = () => projectSrtFileInput.click();
  projectAudioFileInput.onchange = () => {
    chooseProjectAudioFile(projectAudioFileInput.files?.[0]);
    projectAudioFileInput.value = "";
  };
  projectSrtFileInput.onchange = () => {
    chooseProjectSrtFile(projectSrtFileInput.files?.[0]);
    projectSrtFileInput.value = "";
  };
  saveButton.onclick = saveSession;
  importPromptJsonButton.onclick = importPromptJson;
  importI2VMotionJsonButton.onclick = importI2VMotionJson;
  bulkSegmentsButton.onclick = openBulkSegmentsModal;
  addSegmentButton.onclick = addSegment;
  addOverlaySegmentButton.onclick = addOverlaySegment;
  createT2IButton.onclick = createT2IPromptWithGemma;
  ernieCreateT2IButton.onclick = createT2IPromptWithGemma;
  createI2VButton.onclick = createI2VPromptWithGemma;
  sendT2IPromptToEnhanceButton.onclick = () => sendPromptToEnhance("T2I", t2iPrompt.value);
  ernieSendT2IPromptToEnhanceButton.onclick = () => sendPromptToEnhance("T2I", ernieT2IPrompt.value);
  sendFluxPromptToEnhanceButton.onclick = () => sendPromptToEnhance("Flux/Klein", fluxPrompt.value);
  zEnhanceGemmaButton.onclick = generateEnhancePromptWithGemma;
  createFluxPromptButton.onclick = createFluxKleinPromptWithGemma;
  createNBPromptButton.onclick = createNBPromptWithGemma;
  for (const button of createSceneVideoButtons) button.onclick = createSceneVideo;
  loadCustomImageButton.onclick = loadCustomImage;
  for (const button of zCreateButtons) button.onclick = previewZImage;
  for (const button of ernieCreateButtons) button.onclick = previewErnieImage;
  for (const button of fluxCreateButtons) button.onclick = previewFluxKleinImage;
  for (const button of nbCreateButtons) button.onclick = previewNBImage;
  customImageFileInput.addEventListener("change", () => {
    const file = customImageFileInput.files?.[0];
    if (file) loadCustomImageFile(file);
  });
  i2iImageFileInput.addEventListener("change", () => {
    const file = i2iImageFileInput.files?.[0];
    if (file) loadImageToImageFile(file);
    i2iImageFileInput.value = "";
  });
  zI2ILoadButton.onclick = () => i2iImageFileInput.click();
  ernieI2ILoadButton.onclick = () => i2iImageFileInput.click();
  zImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "zimage";
    state.fluxKleinSettings.image_model_mode = "zimage";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to ZImage").catch(() => null);
  };
  fluxKleinCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "flux_klein";
    state.fluxKleinSettings.image_model_mode = "flux_klein";
    state.fluxKleinSettings.enabled = true;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Flux/Klein").catch(() => null);
  };
  ernieImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "ernie_image";
    state.fluxKleinSettings.image_model_mode = "ernie_image";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Ernie").catch(() => null);
  };
  zEnhanceCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "z_enhance";
    state.fluxKleinSettings.image_model_mode = "z_enhance";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Enhance").catch(() => null);
  };
  nbImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "nano_banana";
    state.fluxKleinSettings.image_model_mode = "nano_banana";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to NanoBanana").catch(() => null);
  };
  imageToVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "i2v";
    syncVideoModePanel();
  };
  textToVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "t2v";
    syncVideoModePanel();
  };
  fluxIngredientFileInput.addEventListener("change", () => {
    const files = Array.from(fluxIngredientFileInput.files || []);
    for (const file of files) loadFluxIngredientFile(file);
    fluxIngredientFileInput.value = "";
  });
  fluxGlobalIngredientFileInput.addEventListener("change", () => {
    const files = Array.from(fluxGlobalIngredientFileInput.files || []);
    for (const file of files) loadFluxIngredientFile(file, { global: true });
    fluxGlobalIngredientFileInput.value = "";
  });
  fluxIngredientButton.onclick = () => fluxIngredientFileInput.click();
  fluxGlobalIngredientButton.onclick = () => fluxGlobalIngredientFileInput.click();
  nbGlobalIngredientButton.onclick = () => fluxGlobalIngredientFileInput.click();
  fluxGlobalIngredientClearButton.onclick = () => {
    pushHistory();
    state.fluxGlobalImageIngredients = [];
    renderFluxGlobalIngredientList();
    render();
    toast("Global Flux/Klein image ingredients cleared.");
  };
  nbGlobalIngredientClearButton.onclick = () => {
    pushHistory();
    state.fluxGlobalImageIngredients = [];
    renderFluxGlobalIngredientList();
    render();
    toast("Global Nano B reference images cleared.");
  };
  useFluxGlobalIngredients.input.addEventListener("change", () => {
    pushHistory();
    state.useFluxGlobalImageIngredients = Boolean(useFluxGlobalIngredients.input.checked);
    syncFluxGlobalIngredientPanel();
    render();
  });
  nbUseGlobalIngredients.input.addEventListener("change", () => {
    pushHistory();
    state.useFluxGlobalImageIngredients = Boolean(nbUseGlobalIngredients.input.checked);
    syncFluxGlobalIngredientPanel();
    render();
  });
  fluxIngredientClearButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    pushHistory();
    segment.flux_image_ingredients = [];
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    render();
    toast("Flux/Klein image ingredients cleared for this scene.");
  };
  enableFluxIngredientDrop(fluxGlobalIngredientDrop, { global: true });
  enableFluxIngredientDrop(nbGlobalIngredientDrop, { global: true });
  enableFluxIngredientDrop(fluxIngredientDrop);
  nbIngredientButton.onclick = () => fluxIngredientFileInput.click();
  nbIngredientClearButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    pushHistory();
    segment.flux_image_ingredients = [];
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    render();
    toast("NanoBanana reference images cleared for this scene.");
  };
  enableFluxIngredientDrop(nbIngredientDrop);
  zI2IDrop.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
    event.preventDefault();
    event.stopPropagation();
    zI2IDrop.style.borderColor = "#a3e635";
  });
  zI2IDrop.addEventListener("dragleave", () => {
    zI2IDrop.style.borderColor = "#155e75";
  });
  zI2IDrop.addEventListener("drop", (event) => {
    const sceneSource = droppedSceneImageSource(event);
    if (sceneSource) {
      event.preventDefault();
      event.stopPropagation();
      zI2IDrop.style.borderColor = "#155e75";
      setImageToImageSource(sceneSource);
      return;
    }
    const file = imageFileFromDrop(event);
    if (!file) return;
    event.preventDefault();
    event.stopPropagation();
    zI2IDrop.style.borderColor = "#155e75";
    loadImageToImageFile(file);
  });
  ernieI2IDrop.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
    event.preventDefault();
    event.stopPropagation();
    ernieI2IDrop.style.borderColor = "#a3e635";
  });
  ernieI2IDrop.addEventListener("dragleave", () => {
    ernieI2IDrop.style.borderColor = "#155e75";
  });
  ernieI2IDrop.addEventListener("drop", (event) => {
    const sceneSource = droppedSceneImageSource(event);
    if (sceneSource) {
      event.preventDefault();
      event.stopPropagation();
      ernieI2IDrop.style.borderColor = "#155e75";
      setImageToImageSource(sceneSource);
      return;
    }
    const file = imageFileFromDrop(event);
    if (!file) return;
    event.preventDefault();
    event.stopPropagation();
    ernieI2IDrop.style.borderColor = "#155e75";
    loadImageToImageFile(file);
  });
  const t2vVisionRefFileInput = document.createElement("input");
  t2vVisionRefFileInput.type = "file";
  t2vVisionRefFileInput.accept = "image/*";
  t2vVisionRefFileInput.style.display = "none";
  document.body.append(t2vVisionRefFileInput);
  refImageLoadButton.onclick = () => visionRefFileInput.click();
  ernieRefImageLoadButton.onclick = () => visionRefFileInput.click();
  t2vRefImageLoadButton.onclick = () => t2vVisionRefFileInput.click();
  visionRefFileInput.addEventListener("change", () => {
    loadVisionReferenceFile(visionRefFileInput.files?.[0]);
    visionRefFileInput.value = "";
  });
  t2vVisionRefFileInput.addEventListener("change", () => {
    loadVisionReferenceFile(t2vVisionRefFileInput.files?.[0], { forT2V: true });
    t2vVisionRefFileInput.value = "";
  });
  function wireVisionReferenceDrop(dropElement, options = {}) {
    dropElement.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []);
      if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
      event.preventDefault();
      event.stopPropagation();
      dropElement.style.borderColor = "#a3e635";
    });
    dropElement.addEventListener("dragleave", () => {
      dropElement.style.borderColor = "#155e75";
    });
    dropElement.addEventListener("drop", (event) => {
      const sceneSource = droppedSceneImageSource(event);
      if (sceneSource) {
        event.preventDefault();
        event.stopPropagation();
        dropElement.style.borderColor = "#155e75";
        setVisionReferenceSource({ ...sceneSource, forT2V: Boolean(options.forT2V) }).catch((error) => toast(String(error?.message || error), true));
        return;
      }
      const file = imageFileFromDrop(event);
      if (!file) return;
      event.preventDefault();
      event.stopPropagation();
      dropElement.style.borderColor = "#155e75";
      loadVisionReferenceFile(file, { forT2V: Boolean(options.forT2V) });
    });
  }
  wireVisionReferenceDrop(refImageDrop);
  wireVisionReferenceDrop(ernieRefImageDrop);
  wireVisionReferenceDrop(t2vRefImageDrop, { forT2V: true });
  deleteSegmentButton.onclick = deleteSegment;
  deleteSelectedMediaButton.onclick = deleteSelectedMedia;
  playButton.onclick = () => {
    if (isTimelinePlaying()) {
      pauseAllAudio();
      updateAudioScrubbers();
      return;
    }
    if (usingSceneAudioMode()) {
      audio.pause();
      playSceneAudioFrom(currentGlobalTime());
      updatePlayPauseButton();
      return;
    }
    if (!audio.src) {
      toast("Load audio first, or add custom audio to scenes.", true);
      return;
    }
    audio.play().then(updatePlayPauseButton).catch((error) => toast(String(error?.message || error), true));
  };
  multiSelectButton.onclick = () => {
    setMultiSelectMode(!state.multiSelectMode);
    toast(state.multiSelectMode
      ? "Multi-select is on. Click scenes to add/remove them. Image and video model/settings changes will apply to selected scenes."
      : "Multi-select is off.");
  };
  multiSelectHintButton.onclick = showMultiSelectHint;
  stopButton.onclick = () => {
    pauseAllAudio();
    audio.currentTime = 0;
    sceneAudio.currentTime = 0;
    state.sceneAudioGlobalTime = 0;
    state.sceneAudioSegmentId = "";
    if (!previewVideo.paused) previewVideo.pause();
    updateAudioScrubbers();
  };
  timelineCanvas.addEventListener("pointerdown", beginGlobalTimelineScrub);
  playhead.addEventListener("pointerdown", beginGlobalTimelineScrub);
  timelineViewport.addEventListener("click", (event) => {
    if (event.target === timelineViewport || event.target === timelineCanvas) clearActiveSegment();
  });
  segmentList.addEventListener("click", (event) => {
    if (event.target === segmentList) clearActiveSegment();
  });
  previewStage.addEventListener("click", (event) => {
    if (event.target === previewStage || event.target === previewEmpty) clearActiveSegment();
  });
  globalScrub.addEventListener("pointerdown", () => {
    state.isScrubbing = true;
  });
  globalScrub.addEventListener("input", () => {
    setGlobalPlaybackTime(Number(globalScrub.value || 0));
    updateAudioScrubbers();
  });
  globalScrub.addEventListener("change", () => {
    state.isScrubbing = false;
    updateAudioScrubbers();
  });
  waveformModeSelect.onchange = () => {
    state.waveformMode = waveformModeSelect.value || "medium";
    render();
  };
  zoomOutButton.onclick = () => setTimelineZoom(state.pxPerSecond / 1.25);
  zoomInButton.onclick = () => setTimelineZoom(state.pxPerSecond * 1.25);
  beatMarkersButton.onclick = async () => {
    if (state.showBeatMarkers && (!state.beats || !state.beats.length)) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) setBeatMarkersVisible(false);
      render();
      return;
    }
    const shouldShow = !state.showBeatMarkers;
    setBeatMarkersVisible(shouldShow);
    if (state.showBeatMarkers && (!state.beats || !state.beats.length)) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) setBeatMarkersVisible(false);
    }
    render();
  };
  snapToBeatsControl.input.onchange = () => {
    state.snapToBeats = Boolean(snapToBeatsControl.input.checked);
  };
  audio.addEventListener("timeupdate", updateAudioScrubbers);
  audio.addEventListener("loadedmetadata", updateAudioScrubbers);
  audio.addEventListener("play", updatePlayPauseButton);
  audio.addEventListener("pause", () => {
    updatePlayPauseButton();
    updateAudioScrubbers();
  });
  audio.addEventListener("ended", () => {
    if (!previewVideo.paused) previewVideo.pause();
    updatePlayPauseButton();
    updateAudioScrubbers();
  });
  sceneAudio.addEventListener("play", updatePlayPauseButton);
  sceneAudio.addEventListener("pause", updatePlayPauseButton);
  sceneAudio.addEventListener("timeupdate", () => {
    const segment = state.segments.find((item) => item.id === state.sceneAudioSegmentId) || activeSegment();
    if (segment) {
      const sourceLocal = Math.max(0, Number(sceneAudio.currentTime || 0) - audioSourceStart(segment));
      state.sceneAudioGlobalTime = audioTimelineStart(segment) + sourceLocal;
      if (sourceLocal >= audioChunkDuration(segment) - 0.03) {
        sceneAudio.pause();
        sceneAudio.dispatchEvent(new Event("ended"));
        return;
      }
    }
    updateAudioScrubbers();
  });
  sceneAudio.addEventListener("ended", () => {
    const current = currentGlobalTime();
    const next = state.segments.find((segment) => audioTimelineStart(segment) >= current - 0.02 && segment.id !== state.sceneAudioSegmentId && segment.custom_audio_path) ||
      state.segments.find((segment) => audioTimelineStart(segment) > current && segment.custom_audio_path);
    if (next) {
      playSceneAudioFrom(audioTimelineStart(next));
    } else {
      if (!previewVideo.paused) previewVideo.pause();
      updateAudioScrubbers();
    }
  });
  overlay.addEventListener("keydown", (event) => {
    const tag = String(event.target?.tagName || "").toLowerCase();
    const isTyping = tag === "input" || tag === "textarea" || tag === "select" || event.target?.isContentEditable;
    if (isTyping) return;
    if (event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === "z") {
      event.preventDefault();
      undo();
    } else if ((event.ctrlKey && event.key.toLowerCase() === "y") || (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "z")) {
      event.preventDefault();
      redo();
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowRight") {
      if (moveActiveSceneSelection(1)) event.preventDefault();
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowLeft") {
      if (moveActiveSceneSelection(-1)) event.preventDefault();
    }
  });
  overlay.tabIndex = -1;
  setTimeout(() => overlay.focus(), 0);

  getJson("/vrgdg/music_builder/gemma_choices").then((data) => {
    const models = data.models || [];
    const mmproj = data.mmproj || [];
    const validMmproj = mmproj.filter((item) => item && !/^\[No mmproj/i.test(item));
    const singleMmproj = validMmproj.length === 1 ? validMmproj[0] : "";
    for (const select of [t2iTextGemmaModelSelect, gemmaModelSelect, ernieTextGemmaModelSelect, ernieGemmaModelSelect, zEnhanceGemmaModelSelect, i2vTextGemmaModelSelect, i2vGemmaModelSelect, fluxGemmaModelSelect, nbGemmaModelSelect]) {
      select.textContent = "";
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        select.append(option);
      }
    }
    const preferredNonVision = models.find((model) => model === DEFAULT_NON_VISION_GEMMA_MODEL)
      || models.find((model) => /supergemma4.*fast.*q4_k_m/i.test(model))
      || models.find((model) => /supergemma/i.test(model))
      || "";
    if (preferredNonVision) {
      for (const select of [t2iTextGemmaModelSelect, ernieTextGemmaModelSelect, i2vTextGemmaModelSelect]) {
        select.value = preferredNonVision;
      }
    }
    for (const select of [mmprojSelect, ernieMmprojSelect, zEnhanceMmprojSelect, i2vMmprojSelect, fluxMmprojSelect, nbMmprojSelect]) {
      const previousValue = select.value;
      select.textContent = "";
      for (const item of mmproj) {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        select.append(option);
      }
      if (previousValue && mmproj.includes(previousValue)) {
        select.value = previousValue;
      } else if (singleMmproj) {
        select.value = singleMmproj;
      }
    }
  }).catch((error) => {
    toast(`Could not load Gemma model choices:\n${String(error?.message || error)}`, true);
  });

  function renderSearchableSuggestions(picker, onSelect = null) {
    const query = String(picker.input.value || "").trim().toLowerCase();
    const options = picker.options || [];
    const matches = options
      .filter((name) => !query || String(name).toLowerCase().includes(query))
      .slice(0, 50);
    picker.matches = matches;
    if (!matches.length) picker.activeIndex = -1;
    else if (!Number.isInteger(picker.activeIndex) || picker.activeIndex < 0 || picker.activeIndex >= matches.length) picker.activeIndex = 0;
    picker.list.textContent = "";
    const choose = (name) => {
      picker.input.value = name;
      picker.list.style.display = "none";
      onSelect?.();
    };
    for (const [index, name] of matches.entries()) {
      const item = document.createElement("button");
      item.type = "button";
      item.textContent = name;
      item.title = name;
      item.style.cssText = `display:block;width:100%;text-align:left;border:0;background:${index === picker.activeIndex ? "#0e7490" : "#18181b"};color:#fafafa;padding:7px 8px;font-size:12px;line-height:1.35;cursor:pointer;white-space:normal;overflow-wrap:anywhere;`;
      item.onmouseenter = () => {
        picker.activeIndex = index;
        Array.from(picker.list.children).forEach((button, childIndex) => {
          button.style.background = childIndex === picker.activeIndex ? "#0e7490" : "#18181b";
        });
      };
      item.onpointerdown = (event) => {
        event.preventDefault();
        choose(name);
      };
      item.onclick = () => choose(name);
      picker.list.append(item);
    }
    picker.list.style.display = matches.length ? "block" : "none";
    const activeButton = picker.list.children[picker.activeIndex];
    activeButton?.scrollIntoView?.({ block: "nearest" });
  }

  function wireSearchablePicker(picker, onChange = null) {
    picker.input.addEventListener("focus", () => renderSearchableSuggestions(picker, onChange));
    picker.input.addEventListener("input", () => {
      picker.activeIndex = 0;
      renderSearchableSuggestions(picker, onChange);
      onChange?.();
    });
    picker.input.addEventListener("keydown", (event) => {
      if (picker.list.style.display !== "block") {
        if (event.key === "ArrowDown") {
          event.preventDefault();
          renderSearchableSuggestions(picker, onChange);
        }
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        picker.activeIndex = Math.min((picker.matches?.length || 1) - 1, (picker.activeIndex < 0 ? 0 : picker.activeIndex + 1));
        renderSearchableSuggestions(picker, onChange);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        picker.activeIndex = Math.max(0, picker.activeIndex <= 0 ? 0 : picker.activeIndex - 1);
        renderSearchableSuggestions(picker, onChange);
      } else if (event.key === "Enter") {
        const selected = picker.matches?.[picker.activeIndex];
        if (selected) {
          event.preventDefault();
          picker.input.value = selected;
          picker.list.style.display = "none";
          onChange?.();
        }
      } else if (event.key === "Escape") {
        picker.list.style.display = "none";
      }
    });
    picker.input.addEventListener("blur", () => {
      setTimeout(() => { picker.list.style.display = "none"; }, 180);
    });
  }

  function basenameOnly(value) {
    return String(value || "").replaceAll("\\", "/").split("/").pop();
  }

  function chooseModelValue(options = [], current = "", preferred = []) {
    const values = (options || []).filter((item) => String(item || "").trim());
    if (!values.length) return "";
    const exact = values.find((item) => item === current);
    if (exact) return exact;
    const currentBase = basenameOnly(current);
    if (currentBase) {
      const sameBase = values.find((item) => basenameOnly(item) === currentBase);
      if (sameBase) return sameBase;
    }
    for (const item of preferred) {
      const direct = values.find((value) => value === item);
      if (direct) return direct;
      const base = basenameOnly(item);
      const sameBase = values.find((value) => basenameOnly(value) === base);
      if (sameBase) return sameBase;
    }
    for (const item of preferred) {
      const needle = basenameOnly(item).toLowerCase().replace(/\.(safetensors|gguf|ckpt)$/i, "");
      const partial = values.find((value) => String(value || "").toLowerCase().includes(needle));
      if (partial) return partial;
    }
    return values[0] || "";
  }

  getJson("/vrgdg/workflow_runner/lora_list").then((data) => {
    const loras = data.loras || ["[none]"];
    for (const slot of [...zLoraSlots, ...ernieLoraSlots, ...fluxLoraSlots, ...i2vLoraSlots, ...zEnhanceLoraSlots]) {
      const current = slot.picker.input.value || "[none]";
      slot.picker.options = loras;
      slot.picker.input.value = loras.includes(current) ? current : current;
    }
  }).catch((error) => {
    toast(`Could not load LoRA choices:\n${String(error?.message || error)}`, true);
  });

  getJson("/vrgdg/workflow_runner/i2v_choices").then((data) => {
    const setOptions = (picker, options, preferred = []) => {
      const preferredList = Array.isArray(preferred) ? preferred : [preferred];
      const values = Array.from(new Set((options || []).filter((item) => String(item || "").trim())));
      picker.options = values;
      const current = BAD_I2V_UNET_ALIASES.has(picker.input.value) ? "" : picker.input.value;
      picker.input.value = chooseModelValue(values, current, preferredList);
    };
    setOptions(i2vUnetPicker, data.unets, DEFAULT_I2V_UNET);
    setOptions(i2vVaePicker, data.vae, "LTX23_video_vae_bf16.safetensors");
    setOptions(i2vClip1Picker, data.clip, "gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors");
    setOptions(i2vClip2Picker, data.clip, "ltx-2.3_text_projection_bf16.safetensors");
    setOptions(i2vUpscalePicker, data.upscale_models, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors");
    setOptions(i2vAudioVaePicker, data.vae, "LTX23_audio_vae_bf16.safetensors");
    setOptions(fluxUnetPicker, data.unets, ["flux\\flux-2-klein-4b-fp8.safetensors", "flux-2-klein-4b-fp8.safetensors"]);
    setOptions(fluxClipPicker, data.clip, ["qwen_3_4b.safetensors", "flux\\qwen_3_4b.safetensors"]);
    setOptions(fluxVaePicker, data.vae, ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"]);
    setOptions(zUnetPicker, data.unets, "z_image_turbo_bf16.safetensors");
    setOptions(zClipPicker, data.clip, "qwen_3_4b.safetensors");
    setOptions(zVaePicker, data.vae, "ae.safetensors");
    setOptions(zEnhanceUnetPicker, data.unets, "z_image_turbo_bf16.safetensors");
    setOptions(zEnhanceClipPicker, data.clip, "qwen_3_4b.safetensors");
    setOptions(zEnhanceVaePicker, data.vae, "ae.safetensors");
    setOptions(ernieUnetPicker, data.unets, ["ernie\\ernie-image-turbo.safetensors", "ernie-image-turbo.safetensors"]);
    setOptions(ernieClipPicker, data.clip, ["ministral-3-3b.safetensors", "ernie\\ministral-3-3b.safetensors"]);
    setOptions(ernieVaePicker, data.vae, ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"]);
  }).catch((error) => {
    toast(`Could not load I2V model choices:\n${String(error?.message || error)}`, true);
  });

  for (const control of [zFirstWidth, zFirstHeight, zSecondWidth, zSecondHeight, zSeed, zSeedMode, zBatchSize, zLoraCount, zI2IStartStep, zI2IPath]) {
    control.addEventListener("input", saveZImageSettingsFromPanel);
    control.addEventListener("change", saveZImageSettingsFromPanel);
  }
  for (const control of [ernieWidth, ernieHeight, ernieSeed, ernieSeedMode, ernieBatchSize, ernieLoraCount, ernieI2IStartStep, ernieI2IPath]) {
    control.addEventListener("input", saveErnieImageSettingsFromPanel);
    control.addEventListener("change", saveErnieImageSettingsFromPanel);
  }
  for (const picker of [zUnetPicker, zClipPicker, zVaePicker]) {
    wireSearchablePicker(picker, saveZImageSettingsFromPanel);
    picker.input.addEventListener("change", saveZImageSettingsFromPanel);
  }
  for (const picker of [ernieUnetPicker, ernieClipPicker, ernieVaePicker]) {
    wireSearchablePicker(picker, saveErnieImageSettingsFromPanel);
    picker.input.addEventListener("change", saveErnieImageSettingsFromPanel);
  }
  zI2ISlider.addEventListener("input", () => {
    zI2IStartStep.value = zI2ISlider.value;
    saveZImageSettingsFromPanel();
  });
  zI2IStartStep.addEventListener("input", () => {
    const value = Math.max(1, Math.min(8, Number(zI2IStartStep.value || 5)));
    zI2ISlider.value = String(value);
  });
  zUseLora.input.addEventListener("change", saveZImageSettingsFromPanel);
  zUseImageToImage.input.addEventListener("change", saveZImageSettingsFromPanel);
  ernieI2ISlider.addEventListener("input", () => {
    ernieI2IStartStep.value = ernieI2ISlider.value;
    saveErnieImageSettingsFromPanel();
  });
  ernieI2IStartStep.addEventListener("input", () => {
    const value = Math.max(1, Math.min(8, Number(ernieI2IStartStep.value || 5)));
    ernieI2ISlider.value = String(value);
  });
  ernieUseLora.input.addEventListener("change", saveErnieImageSettingsFromPanel);
  ernieUseImageToImage.input.addEventListener("change", saveErnieImageSettingsFromPanel);
  for (const slot of zLoraSlots) {
    wireSearchablePicker(slot.picker, saveZImageSettingsFromPanel);
    slot.firstPassStrength.addEventListener("input", saveZImageSettingsFromPanel);
    slot.firstPassStrength.addEventListener("change", saveZImageSettingsFromPanel);
    slot.secondPassStrength.addEventListener("input", saveZImageSettingsFromPanel);
    slot.secondPassStrength.addEventListener("change", saveZImageSettingsFromPanel);
  }
  for (const slot of ernieLoraSlots) {
    wireSearchablePicker(slot.picker, saveErnieImageSettingsFromPanel);
    slot.strength.addEventListener("input", saveErnieImageSettingsFromPanel);
    slot.strength.addEventListener("change", saveErnieImageSettingsFromPanel);
  }

  for (const picker of [i2vUnetPicker, i2vVaePicker, i2vClip1Picker, i2vClip2Picker, i2vUpscalePicker, i2vAudioVaePicker]) {
    wireSearchablePicker(picker, saveI2VVideoSettingsFromPanel);
    picker.input.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  makePanelResize(leftResizeHandle, "left");
  makePanelResize(rightResizeHandle, "right");
  makePanelResize(timelineResizeHandle, "timeline");
  applyLayoutSizes();
  setInspectorTab("scene");
  for (const picker of [fluxUnetPicker, fluxClipPicker, fluxVaePicker]) {
    wireSearchablePicker(picker, saveFluxKleinSettingsFromPanel);
    picker.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
  for (const control of [fluxNotes, fluxPrompt, fluxWidth, fluxHeight, fluxSeed, fluxLoraCount]) {
    control.addEventListener("input", saveFluxKleinSettingsFromPanel);
    control.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
  useFluxKlein.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  fluxUseLora.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  for (const slot of fluxLoraSlots) {
    wireSearchablePicker(slot.picker, saveFluxKleinSettingsFromPanel);
    slot.strength.addEventListener("input", saveFluxKleinSettingsFromPanel);
    slot.strength.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
  i2vLoraHintButton.addEventListener("click", () => {
    showInfoModal({
      title: "Two-Pass LoRA Strengths",
      lines: [
        "Some LoRAs can affect motion when they are too strong on the first pass. Using a lower Pass 1 value can help preserve motion, then a stronger Pass 2 value can bring back the LoRA details.",
        "Style LoRAs trained on images often work better around 0.5 on Pass 1 and 1.0 on Pass 2.",
      ],
    });
  });

  function updateI2VLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(i2vLoraCount.value || 0)));
    i2vLoraPanel.style.display = i2vUseLora.input.checked ? "flex" : "none";
    i2vLoraRows.style.display = i2vUseLora.input.checked && count > 0 ? "flex" : "none";
    i2vLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function updateZEnhanceLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(zEnhanceLoraCount.value || 0)));
    zEnhanceLoraPanel.style.display = zEnhanceUseLora.input.checked ? "flex" : "none";
    zEnhanceLoraRows.style.display = zEnhanceUseLora.input.checked && count > 0 ? "flex" : "none";
    zEnhanceLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  zEnhanceButton.onclick = upscaleEnhanceImage;
  zEnhanceUseLora.input.addEventListener("change", saveZEnhanceSettingsFromPanel);
  for (const control of [zEnhanceWidth, zEnhanceHeight, zEnhanceSeed, zEnhanceSeedMode, zEnhanceAmount, zEnhanceLoraCount]) {
    control.addEventListener("input", saveZEnhanceSettingsFromPanel);
    control.addEventListener("change", saveZEnhanceSettingsFromPanel);
  }
  for (const picker of [zEnhanceUnetPicker, zEnhanceClipPicker, zEnhanceVaePicker]) {
    wireSearchablePicker(picker, saveZEnhanceSettingsFromPanel);
  }
  for (const slot of zEnhanceLoraSlots) {
    wireSearchablePicker(slot.picker, saveZEnhanceSettingsFromPanel);
    slot.strength.addEventListener("input", saveZEnhanceSettingsFromPanel);
    slot.strength.addEventListener("change", saveZEnhanceSettingsFromPanel);
  }

  i2vUseLora.input.addEventListener("change", updateI2VLoraVisibility);
  i2vUseLora.input.addEventListener("change", saveI2VVideoSettingsFromPanel);
  i2vLoraCount.addEventListener("input", saveI2VVideoSettingsFromPanel);
  i2vLoraCount.addEventListener("change", saveI2VVideoSettingsFromPanel);
  for (const slot of i2vLoraSlots) {
    wireSearchablePicker(slot.picker, saveI2VVideoSettingsFromPanel);
    slot.firstPassStrength.addEventListener("input", saveI2VVideoSettingsFromPanel);
    slot.firstPassStrength.addEventListener("change", saveI2VVideoSettingsFromPanel);
    slot.secondPassStrength.addEventListener("input", saveI2VVideoSettingsFromPanel);
    slot.secondPassStrength.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  for (const control of [i2vFpsInput, i2vWidthInput, i2vHeightInput, i2vSeedInput]) {
    control.addEventListener("input", saveI2VVideoSettingsFromPanel);
    control.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  updateI2VLoraVisibility();

  syncInspector();
  syncZImageSettingsPanel();
  syncFluxKleinPanel();
  syncZEnhanceSettingsPanel();
  syncI2VVideoSettingsPanel();
  syncVideoModePanel();
  updateHistoryButtons();
  render();
}

function ensureButton(node) {
  const buttonName = "Open Music Video Builder";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => openBuilder(node));
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
}

app.registerExtension({
  name: "vrgdg.MusicVideoBuilderUI",
  loadedGraphNode(node) {
    if ((node?.comfyClass || node?.type) === NODE_NAME) ensureButton(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
  },
});
