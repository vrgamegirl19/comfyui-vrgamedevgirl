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

function createProgressWindow(title) {
  const box = document.createElement("div");
  box.style.cssText = `
    position: fixed;
    left: 50%;
    top: 54px;
    transform: translateX(-50%);
    z-index: 100004;
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
  restore.style.cssText = "position:fixed;left:50%;top:54px;transform:translateX(-50%);z-index:100004;display:none;max-width:min(850px,calc(100vw - 560px));min-width:260px;border:1px solid #155e75;border-radius:8px;background:#083344;color:#cffafe;padding:8px 12px;font-size:12px;font-weight:900;box-shadow:0 16px 48px rgba(0,0,0,.45);cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
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
    const groupTitle = document.createElement("div");
    groupTitle.textContent = group.title;
    groupTitle.style.cssText = "font-size:22px;font-weight:900;color:#f8fafc;";
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
    card.append(groupTitle, note, buttons);
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
      const item = document.createElement("div");
      item.textContent = line;
      body.append(item);
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

async function queueWorkflowPrompt(prompt) {
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
    use_scene_zimage_settings: false,
    zimage_settings: null,
    use_scene_ernie_image_settings: false,
    ernie_image_settings: null,
    use_scene_flux_klein_settings: false,
    flux_klein_settings: null,
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
  shell.style.cssText = `
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
  const closeButton = makeButton("Close");
  closeButton.onclick = () => overlay.remove();
  const promptCreatorButton = makeButton("Prompt Creator");
  const autoLoadAllButton = makeButton("Import Data From Prompt Creator");
  const promptOptionsButton = makeButton("Prompt Options");
  const clearMemoryButton = makeButton("Clear Memory");
  const renderAllButton = makeButton("Render All");
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
  for (const button of [newProjectButton, loadSessionButton, loadLastProjectButton, saveProjectAsButton, settingsButton, promptCreatorButton, autoLoadAllButton, zImageAllButton, renderAllButton, fullBuildButton, remakeModeButton]) {
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
  importActions.append(promptOptionsButton);
  const utilityActions = document.createElement("div");
  utilityActions.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
  utilityActions.append(stopWorkflowButton, downloadModelsButton, clearMemoryButton, closeButton);
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
  previewVideo.muted = true;
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
  const useSceneI2VVideoSettings = makeCheckbox("Use custom video models/settings/LoRAs for this scene", false);
  const useSceneI2VVideoSettingsModels = makeCheckbox("Use custom video models/settings/LoRAs for this scene", false);
  const useSceneI2VVideoSettingsNote = document.createElement("div");
  useSceneI2VVideoSettingsNote.textContent = "Applies the Models tab choices too, including video model files, LoRAs, LoRA count, and both pass strengths.";
  useSceneI2VVideoSettingsNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const useSceneI2VVideoSettingsModelsNote = document.createElement("div");
  useSceneI2VVideoSettingsModelsNote.textContent = useSceneI2VVideoSettingsNote.textContent;
  useSceneI2VVideoSettingsModelsNote.style.cssText = useSceneI2VVideoSettingsNote.style.cssText;
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
  const fluxGemmaModelSelect = makeSelect([""], "");
  const fluxMmprojSelect = makeSelect([""], "");
  const fluxPrompt = document.createElement("textarea");
  fluxPrompt.placeholder = "Flux/Klein prompt...";
  fluxPrompt.style.cssText = fluxNotes.style.cssText;
  const fluxUnetPicker = makeSearchableLoraPicker("flux\\flux-2-klein-4b-fp8.safetensors");
  const fluxClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const fluxVaePicker = makeSearchableLoraPicker("flux\\flux2-vae.safetensors");
  const fluxWidth = makeInput("1024", "number");
  const fluxHeight = makeInput("576", "number");
  const fluxSeed = makeInput("100", "number");
  const fluxGrid = document.createElement("div");
  fluxGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  fluxGrid.append(makeField("Width", fluxWidth), makeField("Height", fluxHeight), makeField("Seed", fluxSeed));
  const createFluxPromptButton = makeButton("Gemma Flux Prompt", "primary");
  const previewFluxButton = makeButton("Create with Flux/Klein", "primary");
  const sendFluxPromptToEnhanceButton = makeMiniButton("Send to Enhance");
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
  const imageModelChooser = document.createElement("div");
  imageModelChooser.style.cssText = "display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;";
  function makeImageModelCard(label, value) {
    const card = document.createElement("button");
    card.type = "button";
    card.dataset.model = value;
    card.textContent = label;
    card.style.cssText = "min-height:38px;border:1px solid #3f3f46;border-radius:6px;background:#27272a;color:#f4f4f5;font-size:12px;font-weight:900;cursor:pointer;padding:8px 10px;";
    return card;
  }
  const zImageCard = makeImageModelCard("ZImage", "zimage");
  const fluxKleinCard = makeImageModelCard("Flux Klein", "flux_klein");
  const ernieImageCard = makeImageModelCard("Ernie", "ernie_image");
  const zEnhanceCard = makeImageModelCard("Enhance", "z_enhance");
  const loadCustomImageButton = makeImageModelCard("Load Custom", "custom_image");
  loadCustomImageButton.title = "Load a custom image for the selected scene";
  imageModelChooser.append(zImageCard, fluxKleinCard, ernieImageCard, zEnhanceCard, loadCustomImageButton);
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
  const ernieCreateT2IButton = makeButton("Gemma T2I", "primary");
  const ernieSendT2IPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const i2vPrompt = document.createElement("textarea");
  i2vPrompt.placeholder = "Image-to-video prompt...";
  i2vPrompt.style.cssText = notesInput.style.cssText;
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
        makeFluxCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        useSceneFluxKleinSettings.wrapper,
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
    imageModelChooser,
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
        useSceneI2VVideoSettingsModels.wrapper,
        useSceneI2VVideoSettingsModelsNote,
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
        useSceneI2VVideoSettings.wrapper,
        useSceneI2VVideoSettingsNote,
        makeField("Video trigger phrase", videoTriggerInput),
        i2vSettingsGrid,
        makeCreateSceneVideoButton(),
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("I2V motion notes", i2vNotesInput),
        useI2VVisionReference.wrapper,
        i2vReferenceNote,
        createI2VButton,
        makeField("I2V prompt", i2vPrompt),
        makeCreateSceneVideoButton(),
      ]),
    },
  ]);
  videoPanel.append(videoSubTabs.wrapper);
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
  timelineHeader.style.cssText = "display:grid;grid-template-columns:auto auto auto auto auto auto auto auto auto auto minmax(260px,1fr) auto auto;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid #27272a;font-size:12px;";
  const addSegmentButton = makeButton("Add Segment", "primary");
  const addOverlaySegmentButton = makeButton("Add Insert", "primary");
  const undoButton = makeButton("Undo");
  const redoButton = makeButton("Redo");
  const playButton = makeButton("Play");
  const stopButton = makeButton("Stop");
  const deleteSegmentButton = makeButton("Del");
  const zoomOutButton = makeButton("-");
  const zoomInButton = makeButton("+");
  addSegmentButton.title = "Add segment";
  addOverlaySegmentButton.title = "Add an insert/overlay segment at the playhead without changing the base timeline.";
  undoButton.title = "Undo";
  redoButton.title = "Redo";
  playButton.title = "Play / Pause";
  stopButton.title = "Stop";
  deleteSegmentButton.title = "Delete selected segment";
  zoomOutButton.title = "Zoom out timeline";
  zoomInButton.title = "Zoom in timeline";
  addSegmentButton.textContent = "+ Segment";
  addOverlaySegmentButton.textContent = "+ Insert";
  undoButton.textContent = "↶";
  redoButton.textContent = "↷";
  playButton.textContent = "▶";
  stopButton.textContent = "■";
  deleteSegmentButton.textContent = "×";
  deleteSegmentButton.style.borderColor = "#7f1d1d";
  deleteSegmentButton.style.color = "#fecaca";
  for (const button of [addSegmentButton, addOverlaySegmentButton, undoButton, redoButton, playButton, stopButton, deleteSegmentButton, zoomOutButton, zoomInButton]) {
    button.style.padding = "7px 10px";
    button.style.minWidth = "0";
  }
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
  timelineHeader.append(addSegmentButton, addOverlaySegmentButton, undoButton, redoButton, playButton, stopButton, waveformModeSelect, snapToBeatsControl.wrapper, beatMarkersButton, zoomWrap, timelineInfo, deleteSegmentButton, selectedMediaTools);
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
  setTimeout(() => {
    showStartupWelcome().catch((error) => {
      console.warn("[VRGDG Music Builder] Startup welcome failed:", error);
      toast(`Video Creator startup failed:\n${String(error?.message || error)}`, true);
    });
  }, 250);
  for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
    overlay.addEventListener(eventName, (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
    });
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
      unet_name: "flux\\flux-2-klein-4b-fp8.safetensors",
      clip_name: "qwen_3_4b.safetensors",
      vae_name: "flux\\flux2-vae.safetensors",
      width: 1024,
      height: 576,
      seed: 100,
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
    imageTriggerPhrase: "",
    videoTriggerPhrase: "",
    useVrgdgTextContext: true,
    themeStylePath: "",
    storyIdeaPath: "",
    subjectScenePath: "",
    imageModelMode: "zimage",
    zimageSettings: defaultZImageSettings(),
    fluxKleinSettings: defaultFluxKleinSettings(),
    ernieImageSettings: defaultErnieImageSettings(),
    useFluxGlobalImageIngredients: false,
    fluxGlobalImageIngredients: [],
    zEnhanceSettings: defaultZEnhanceSettings(),
    i2vVideoSettings: defaultI2VVideoSettings(),
    promptToolsHintPrefs: {},
    undoStack: [],
    redoStack: [],
    isRestoringHistory: false,
    batchCancelled: false,
  };
  const LAST_PROJECT_KEY = "vrgdg_music_builder_last_project_folder";

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

  function cloneFluxKleinSettings(settings) {
    const source = settings || {};
    return {
      ...defaultFluxKleinSettings(),
      ...source,
      width: Number(source.width || 1024),
      height: Number(source.height || 576),
      seed: Number(source.seed || 100),
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
      ernieImageSettings: state.ernieImageSettings,
      useFluxGlobalImageIngredients: state.useFluxGlobalImageIngredients,
      fluxGlobalImageIngredients: state.fluxGlobalImageIngredients,
      zEnhanceSettings: state.zEnhanceSettings,
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
    state.ernieImageSettings = data.ernieImageSettings || state.ernieImageSettings;
    state.useFluxGlobalImageIngredients = Boolean(data.useFluxGlobalImageIngredients);
    state.fluxGlobalImageIngredients = Array.isArray(data.fluxGlobalImageIngredients) ? data.fluxGlobalImageIngredients : [];
    state.zEnhanceSettings = data.zEnhanceSettings || state.zEnhanceSettings;
    state.i2vVideoSettings = data.i2vVideoSettings || state.i2vVideoSettings;
    state.promptToolsHintPrefs = data.promptToolsHintPrefs || data.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncZEnhanceSettingsPanel();
    syncI2VVideoSettingsPanel();
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
        target_peaks: 1800,
      }, 90000);
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

  function applyTriggerPhrase(prompt, trigger) {
    const promptText = cleanGeneratedPromptText(prompt);
    if (looksLikeGeneratedPromptJunk(promptText)) {
      throw new Error("Gemma returned repeated/thought junk instead of a usable prompt. Try again or shorten the notes.");
    }
    const triggerText = String(trigger || "").trim().replace(/\s+/g, " ");
    if (!triggerText) return promptText;
    if (!promptText) return triggerText;
    if (promptText.toLowerCase().startsWith(triggerText.toLowerCase())) return promptText;
    return `${triggerText}, ${promptText}`;
  }

  function imageTriggerPhraseForSegment(segment = activeSegment()) {
    if (!segment) return state.imageTriggerPhrase || "";
    if (state.imageModelMode === "flux_klein") return fluxKleinSettingsForSegment(segment).image_trigger_phrase || state.imageTriggerPhrase || "";
    if (state.imageModelMode === "ernie_image") return (segment.use_scene_ernie_image_settings ? segment.ernie_image_settings?.image_trigger_phrase : state.ernieImageSettings?.image_trigger_phrase) || state.imageTriggerPhrase || "";
    return (segment.use_scene_zimage_settings ? segment.zimage_settings?.image_trigger_phrase : state.zimageSettings?.image_trigger_phrase) || state.imageTriggerPhrase || "";
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

  function selectedSegmentImagePath(segment) {
    ensureSegmentRuntimeFields(segment);
    const history = Array.isArray(segment?.image_history) ? segment.image_history : [];
    const index = Math.max(0, Math.min(history.length - 1, Number(segment?.image_history_index || 0)));
    return history[index] || segment?.approved_image_path || segment?.custom_image_path || "";
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
      if (previewVideo.dataset.path !== videoPath) {
        previewVideo.src = makeEditorVideoUrl(videoPath);
        previewVideo.dataset.path = videoPath;
      }
      previewVideo.muted = true;
      previewVideo.style.display = "block";
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
      return;
    }
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.dataset.path = "";
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
    for (const control of [labelInput, startInput, endInput, notesInput, ernieNotesInput, i2vNotesInput, t2iPrompt, ernieT2IPrompt, i2vPrompt, zEnhanceGemmaNotes, zEnhancePromptPreview, previewButton, ernieCreateButton, deleteSegmentButton, createSceneVideoButton]) {
      control.disabled = disabled;
    }
    loadCustomImageButton.disabled = disabled;
    openSceneAudioOptionsButton.disabled = disabled;
    for (const control of [t2iTextGemmaModelSelect, gemmaModelSelect, mmprojSelect, ernieTextGemmaModelSelect, ernieGemmaModelSelect, ernieMmprojSelect, zEnhanceGemmaModelSelect, zEnhanceMmprojSelect, i2vTextGemmaModelSelect, i2vGemmaModelSelect, i2vMmprojSelect, useVisionReference.input, ernieUseVisionReference.input, useI2VVisionReference.input, useSceneZImageSettings.input, useSceneErnieImageSettings.input, useSceneFluxKleinSettings.input, useSceneI2VVideoSettings.input, useSceneI2VVideoSettingsModels.input, refImageInput, createT2IButton, ernieCreateT2IButton, createI2VButton, zEnhanceGemmaButton]) {
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
    useSceneI2VVideoSettings.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
    useSceneI2VVideoSettingsModels.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
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
      i2vNotesInput.value = "";
      t2iPrompt.value = "";
      ernieT2IPrompt.value = "";
      i2vPrompt.value = "";
      useVisionReference.input.checked = false;
      ernieUseVisionReference.input.checked = false;
      useI2VVisionReference.input.checked = true;
      useSceneZImageSettings.input.checked = false;
      refImageInput.value = "";
      refImagePanel.style.display = "none";
      ernieRefImagePanel.style.display = "none";
      audioSummary.textContent = "Select a scene to view or edit scene audio.";
      syncZImageSettingsPanel();
      syncFluxKleinPanel();
      syncErnieImagePanel();
      syncPreview(null);
      return;
    }
    labelInput.value = segment.label || "";
    startInput.value = segment.start;
    endInput.value = segment.end;
    notesInput.value = segment.notes || "";
    ernieNotesInput.value = segment.notes || "";
    i2vNotesInput.value = segment.i2v_notes || "";
    t2iPrompt.value = segment.t2i_prompt || "";
    ernieT2IPrompt.value = segment.t2i_prompt || "";
    fluxPrompt.value = segment.t2i_prompt || segment.flux_prompt || "";
    i2vPrompt.value = segment.i2v_prompt || "";
    useVisionReference.input.checked = Boolean(segment.use_vision_reference);
    ernieUseVisionReference.input.checked = Boolean(segment.use_vision_reference);
    useI2VVisionReference.input.checked = segment.use_i2v_vision_reference !== false;
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
    if (previewVideo.dataset.path !== videoPath) {
      previewVideo.src = makeEditorVideoUrl(videoPath);
      previewVideo.dataset.path = videoPath;
      previewVideo.muted = true;
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
    if (segment?.use_scene_zimage_settings) {
      segment.zimage_settings = settings;
    } else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.zimageSettings = settings;
    }
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
    if (segment?.use_scene_ernie_image_settings) segment.ernie_image_settings = settings;
    else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.ernieImageSettings = settings;
    }
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

  function setVideoSeedRandom() {
    const settings = activeI2VVideoSettings() || {};
    settings.seed = randomSeedValue();
    if (activeSegment()?.use_scene_i2v_video_settings) activeSegment().i2v_video_settings = settings;
    else state.i2vVideoSettings = settings;
    i2vSeedInput.value = String(settings.seed);
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
  }

  function syncFluxGlobalIngredientPanel() {
    useFluxGlobalIngredients.input.checked = Boolean(state.useFluxGlobalImageIngredients);
    fluxGlobalIngredientPanel.style.display = state.useFluxGlobalImageIngredients ? "flex" : "none";
  }

  function mergedFluxImageIngredients(segment = activeSegment()) {
    const globalIngredients = state.useFluxGlobalImageIngredients && Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [];
    const sceneIngredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
    return [...globalIngredients, ...sceneIngredients];
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
    fluxKleinModePanel.style.display = mode === "flux_klein" ? "flex" : "none";
    ernieImageModePanel.style.display = mode === "ernie_image" ? "flex" : "none";
    zEnhancePanel.style.display = mode === "z_enhance" ? "flex" : "none";
    previewButton.style.display = mode === "zimage" ? "" : "none";
    ernieCreateButton.style.display = mode === "ernie_image" ? "" : "none";
    useFluxKlein.input.checked = mode === "flux_klein";
    fluxKleinPanel.style.display = mode === "flux_klein" ? "flex" : "none";
    ernieImagePanel.style.display = mode === "ernie_image" ? "flex" : "none";
    for (const card of [zImageCard, fluxKleinCard, ernieImageCard, zEnhanceCard]) {
      const active = card.dataset.model === mode;
      card.style.borderColor = active ? "#71717a" : "#3f3f46";
      card.style.background = active ? "#52525b" : "#27272a";
      card.style.color = "#f4f4f5";
      card.style.boxShadow = active ? "inset 0 0 0 1px rgba(244,244,245,.12)" : "none";
    }
    loadCustomImageButton.style.borderColor = "#3f3f46";
    loadCustomImageButton.style.background = "#27272a";
    loadCustomImageButton.style.color = "#f4f4f5";
    loadCustomImageButton.style.boxShadow = "none";
    syncFluxGlobalIngredientPanel();
    renderFluxGlobalIngredientList();
    renderFluxIngredientList(segment);
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
    syncErnieImagePanel();
  }

  function saveFluxKleinSettingsFromPanel() {
    pushHistory();
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
      image_trigger_phrase: fluxImageTriggerInput.value || "",
    };
    if (segment?.use_scene_flux_klein_settings) segment.flux_klein_settings = settings;
    else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.fluxKleinSettings = settings;
    }
    return {
      ...settings,
      image_ingredients: mergedFluxImageIngredients(segment),
      use_global_image_ingredients: Boolean(state.useFluxGlobalImageIngredients),
      global_image_ingredients: Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [],
      scene_image_ingredients: Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [],
      notes: segment?.flux_notes || "",
      prompt: segment?.flux_prompt || "",
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
    useSceneI2VVideoSettingsModels.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
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
    if (segment?.use_scene_i2v_video_settings) segment.i2v_video_settings = settings;
    else {
      state.videoTriggerPhrase = settings.video_trigger_phrase || "";
      state.i2vVideoSettings = settings;
    }
    updateI2VLoraVisibility();
    return settings;
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
    }
    segment.i2v_notes = i2vNotesInput.value || "";
    const editedT2IPrompt = state.imageModelMode === "ernie_image"
      ? ernieT2IPrompt.value || ""
      : state.imageModelMode === "flux_klein"
        ? fluxPrompt.value || ""
        : t2iPrompt.value || "";
    segment.t2i_prompt = editedT2IPrompt;
    segment.flux_prompt = editedT2IPrompt;
    if (t2iPrompt.value !== editedT2IPrompt) t2iPrompt.value = editedT2IPrompt;
    if (ernieT2IPrompt.value !== editedT2IPrompt) ernieT2IPrompt.value = editedT2IPrompt;
    if (fluxPrompt.value !== editedT2IPrompt) fluxPrompt.value = editedT2IPrompt;
    segment.i2v_prompt = i2vPrompt.value || "";
    segment.enhance_notes = zEnhanceGemmaNotes.value || "";
    segment.enhance_prompt = zEnhancePromptPreview.value || segment.enhance_prompt || "";
    segment.use_vision_reference = Boolean(useVisionReference.input.checked);
    if (state.imageModelMode === "ernie_image") {
      segment.use_vision_reference = Boolean(ernieUseVisionReference.input.checked);
    }
    segment.use_i2v_vision_reference = Boolean(useI2VVisionReference.input.checked);
    segment.ref_image_path = refImageInput.value || "";
    refImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
    ernieRefImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
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
      block.innerHTML = `<span style="display:block;font-weight:900;">${escapeHtml(segment.label || (isOverlay ? "Insert" : "Scene"))}</span><span style="display:block;margin-top:3px;font-size:10px;color:#d4d4d8;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</span>`;
      const left = segment.start * state.pxPerSecond;
      const width = Math.max(24, (segment.end - segment.start) * state.pxPerSecond);
      const previewThumbPath = segment.image_history?.[segment.image_history_index] || segment.image_history?.[segment.image_history.length - 1] || segment.custom_image_path || segment.approved_image_path || "";
      const thumb = previewThumbPath ? makeEditorImageUrl(previewThumbPath) : "";
      const inserted = !isOverlay && state.srtMode && segment.source !== "srt";
      const lockedByVideo = hasLockedVideo(segment);
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      block.style.cssText = `
        position:absolute;left:${left}px;top:${blockTop}px;width:${width}px;height:${blockHeight}px;
        border:${isActive ? "3px" : "1px"} solid ${isActive ? "#ef4444" : lockedByVideo ? "#a3e635" : isOverlay ? "#f97316" : inserted ? "#f59e0b" : "#0891b2"};
        border-radius:5px;background:${thumb ? `linear-gradient(rgba(0,0,0,.18),rgba(0,0,0,.18)), url("${thumb}") center / auto 100% repeat-x` : isOverlay ? "#7c2d12" : inserted ? "#92400e" : segment.image ? "#166534" : "#164e63"};
        color:#f4f4f5;font-size:11px;font-weight:800;overflow:hidden;cursor:pointer;pointer-events:auto;
        box-shadow:${isActive ? "0 0 0 2px rgba(239,68,68,.28), 0 0 18px rgba(239,68,68,.55)" : "none"};
      `;
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
      leftHandle.style.cssText = "position:absolute;left:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.25);cursor:ew-resize;";
      const rightHandle = document.createElement("div");
      rightHandle.style.cssText = "position:absolute;right:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.25);cursor:ew-resize;";
      block.append(leftHandle, rightHandle);
      block.onclick = () => setActiveSegment(segment);
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
      const previewThumbPath = segment.image_history?.[segment.image_history_index] || segment.image_history?.[segment.image_history.length - 1] || segment.custom_image_path || segment.approved_image_path || "";
      const thumb = previewThumbPath ? `<img src="${makeEditorImageUrl(previewThumbPath)}" style="width:100%;height:56px;object-fit:cover;border-radius:4px;margin-top:6px;background:#050505;">` : "";
      const inserted = state.srtMode && segment.source !== "srt";
      const t2iDone = Boolean(segmentImageSource(segment));
      const i2vDone = Boolean(String(segment.i2v_prompt || "").trim());
      const videoDone = Boolean(segment.video_path);
      const historyStatus = segment.image_history.length ? `<span style="border:1px solid #67e8f9;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#bae6fd;">IMG ${segment.image_history.length}</span>` : "";
      const zStatus = segment.use_scene_zimage_settings ? `<span style="border:1px solid #f59e0b;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#fde68a;">Z custom</span>` : "";
      const audioStatus = segment.custom_audio_path ? `<span style="border:1px solid #a78bfa;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#ddd6fe;">AUD</span>` : "";
      const status = `
        <div style="display:flex;gap:6px;margin-top:6px;align-items:center;">
          <span style="border:1px solid ${t2iDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${t2iDone ? "#bbf7d0" : "#a1a1aa"};">T2I ${t2iDone ? "OK" : "--"}</span>
          <span style="border:1px solid ${i2vDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${i2vDone ? "#bbf7d0" : "#a1a1aa"};">I2V ${i2vDone ? "OK" : "--"}</span>
          <span style="border:1px solid ${videoDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${videoDone ? "#bbf7d0" : "#a1a1aa"};">VID ${videoDone ? "OK" : "--"}</span>
          ${historyStatus}
          ${zStatus}
          ${audioStatus}
        </div>
      `;
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      row.style.cssText = `width:100%;text-align:left;border:${isActive ? "3px" : "1px"} solid ${isActive ? "#ef4444" : inserted ? "#f59e0b" : "#3f3f46"};border-radius:7px;background:${isActive ? "#3f1d24" : inserted ? "#451a03" : "#27272a"};color:#fafafa;padding:8px;margin-bottom:8px;cursor:pointer;box-shadow:${isActive ? "0 0 0 2px rgba(239,68,68,.25), 0 0 18px rgba(239,68,68,.42)" : "none"};`;
      row.innerHTML = `<div style="font-weight:800;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${index + 1}. ${escapeHtml(segment.label || "Scene")}</div><div style="font-size:11px;color:#a1a1aa;margin-top:4px;">Duration in seconds: ${formatDurationSeconds(segment.start, segment.end)}</div><div style="font-size:11px;color:#71717a;margin-top:2px;">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>${status}${thumb}`;
      row.onclick = () => setActiveSegment(segment);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setActiveSegment(segment);
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
      const previewThumbPath = segment.image_history?.[segment.image_history_index] || segment.image_history?.[segment.image_history.length - 1] || segment.custom_image_path || segment.approved_image_path || "";
      const thumb = previewThumbPath ? `<img src="${makeEditorImageUrl(previewThumbPath)}" style="width:100%;height:50px;object-fit:cover;border-radius:4px;margin-top:6px;background:#050505;">` : "";
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      row.style.cssText = `width:100%;text-align:left;border:${isActive ? "3px" : "1px"} solid ${isActive ? "#ef4444" : "#f97316"};border-radius:7px;background:${isActive ? "#3f1d24" : "#431407"};color:#fafafa;padding:8px;margin-bottom:8px;cursor:pointer;box-shadow:${isActive ? "0 0 0 2px rgba(239,68,68,.25), 0 0 18px rgba(239,68,68,.42)" : "none"};`;
      row.innerHTML = `<div style="font-weight:800;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Insert ${index + 1}. ${escapeHtml(segment.label || "Insert")}</div><div style="font-size:11px;color:#fed7aa;margin-top:4px;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</div>${thumb}`;
      row.onclick = () => setActiveSegment(segment);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setActiveSegment(segment);
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

  async function setVisionReferenceSource({ path = "", data = "", name = "" } = {}) {
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
    segment.use_vision_reference = true;
    useVisionReference.input.checked = true;
    ernieUseVisionReference.input.checked = true;
    refImageInput.value = segment.ref_image_path || name || "";
    refImagePanel.style.display = "flex";
    ernieRefImagePanel.style.display = "flex";
    renderList();
    toast(`Vision reference set${segment.ref_image_path ? `:\n${segment.ref_image_path}` : "."}`);
  }

  function loadVisionReferenceFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setVisionReferenceSource({ data: String(reader.result || ""), name: file.name || "reference.png" }).catch((error) => toast(String(error?.message || error), true));
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
    if (global) {
      if (!Array.isArray(state.fluxGlobalImageIngredients)) state.fluxGlobalImageIngredients = [];
      state.fluxGlobalImageIngredients.push({
        path: path || "",
        data: data || "",
        name: name || path?.split?.(/[\\/]/)?.pop?.() || "image.png",
      });
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
    segment.flux_image_ingredients.push({
      path: path || "",
      data: data || "",
      name: name || path?.split?.(/[\\/]/)?.pop?.() || "image.png",
    });
    const settings = state.fluxKleinSettings || {};
    settings.enabled = true;
    state.fluxKleinSettings = settings;
    syncFluxKleinPanel();
    renderFluxIngredientList(segment);
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
    element.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []);
      if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
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
        progress.set("Creating draft from your notes...", 25);
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
    const editI2V = makeButton("Edit Image to Video Prompts", "primary");
    const reloadI2V = makeButton("Reload Image to Video Prompts");
    const originalI2V = makeButton("Reload Original I2V Prompts");
    imageGroup.append(imageHeading, editT2I, reloadT2I, originalT2I);
    videoGroup.append(videoHeading, editI2V, reloadI2V, originalI2V);
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
    editI2V.onclick = () => run(() => editFinalPromptList("i2v"));
    reloadI2V.onclick = () => run(() => reloadFinalPromptList("i2v", false));
    originalI2V.onclick = () => run(() => reloadFinalPromptList("i2v", true));
  }

  function render() {
    drawWaveform();
    renderSegments();
    renderList();
    updateSelectedMediaTools();
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
        target_peaks: 1800,
      }, 90000);
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

  async function loadSrt() {
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
    previewVideo.style.display = "none";
  }

  async function autoLoadAll() {
    try {
      autoLoadAllButton.disabled = true;
      autoLoadAllButton.textContent = "Importing...";
      pushHistory();
      let paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
        project_folder: projectInput.value || state.projectFolder || "",
      });
      let exists = paths.exists || {};
      let sourceLabel = "this project";
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
      if (!exists.srt_path || !exists.concept_prompts_path || !hasCurrentPrompts || !hasCurrentMotionNotes) {
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
      await loadSrt();
      let importedPrompts = [];
      let importedMotionNotes = [];
      if (promptJsonInput.value) {
        importedPrompts = await importPromptJson({ quiet: true, pushHistory: false });
      }
      if (i2vMotionJsonInput.value) {
        importedMotionNotes = await importI2VMotionJson({ quiet: true, pushHistory: false });
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
      progress.set("Building cleanup workflow...", 20);
      const built = await postJson("/vrgdg/workflow_runner/build_clear_memory_prompt", {});
      progress.set("Queueing cleanup workflow...", 40);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the cleanup workflow but did not return a prompt_id.");
      progress.set(`Cleanup queued.\nPrompt ID: ${promptId}`, 60);
      const text = await waitForText(promptId, (message) => {
        progress.set(`${message}\nPrompt ID: ${promptId}`, 78);
      });
      progress.set(`Cleanup finished.\nPrompt ID: ${promptId}\n\n${text.join("\n\n")}`, 100);
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
    progress?.set(`Clearing memory after ${label}...`, percent);
    const built = await postJson("/vrgdg/workflow_runner/build_clear_memory_prompt", {});
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the cleanup workflow but did not return a prompt_id.");
    const text = await waitForText(promptId, (message) => {
      progress?.set(`${message}\nPrompt ID: ${promptId}`, percent);
    });
    const output = `Memory cleanup finished after ${label}.\nPrompt ID: ${promptId}\n\n${text.join("\n\n")}`;
    progress?.set(output, percent);
    return output;
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
      ernie_image_settings: state.ernieImageSettings,
      use_flux_global_image_ingredients: Boolean(state.useFluxGlobalImageIngredients),
      flux_global_image_ingredients: Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [],
      z_enhance_settings: state.zEnhanceSettings,
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
      progress.set("Sending interrupt to ComfyUI...", 20);
      await api.fetchApi("/interrupt", { method: "POST" }).catch(() => null);
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
        state.ernieImageSettings = data.session.ernie_image_settings || state.ernieImageSettings;
        state.useFluxGlobalImageIngredients = Boolean(data.session.use_flux_global_image_ingredients);
        state.fluxGlobalImageIngredients = Array.isArray(data.session.flux_global_image_ingredients) ? data.session.flux_global_image_ingredients : [];
        state.zEnhanceSettings = data.session.z_enhance_settings || state.zEnhanceSettings;
        state.i2vVideoSettings = data.session.i2v_video_settings || state.i2vVideoSettings;
        state.promptToolsHintPrefs = data.session.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
        syncZImageSettingsPanel();
        syncFluxKleinPanel();
        syncErnieImagePanel();
        syncI2VVideoSettingsPanel();
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
      state.pxPerSecond = state.timelineZoom;
      waveformModeSelect.value = state.waveformMode;
      snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
      autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
      applyLayoutSizes();
      state.zimageSettings = session.zimage_settings || state.zimageSettings;
      state.fluxKleinSettings = session.flux_klein_settings || state.fluxKleinSettings;
      state.ernieImageSettings = session.ernie_image_settings || state.ernieImageSettings;
      state.useFluxGlobalImageIngredients = Boolean(session.use_flux_global_image_ingredients);
      state.fluxGlobalImageIngredients = Array.isArray(session.flux_global_image_ingredients) ? session.flux_global_image_ingredients : [];
      state.zEnhanceSettings = session.z_enhance_settings || state.zEnhanceSettings;
      state.i2vVideoSettings = session.i2v_video_settings || state.i2vVideoSettings;
      state.promptToolsHintPrefs = session.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
      if (session.audio_path) {
        audioInput.value = session.audio_path;
        setWidgetValue(node, "audio_path", session.audio_path);
        try {
          const audioData = await postJson("/vrgdg/music_builder/analyze_audio", {
            audio_path: session.audio_path,
            target_peaks: 1800,
          });
          state.duration = Number(audioData.duration || 0);
          state.peaks = Array.isArray(audioData.peaks) && audioData.peaks.length ? audioData.peaks : state.peaks;
          state.beats = Array.isArray(audioData.beats) && audioData.beats.length ? audioData.beats : state.beats;
          showBeatMarkersIfAvailable();
          audio.src = audioUrl(audioData.audio_path || session.audio_path);
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

  async function generateT2IPromptForSegment(segment, progress = null, percent = 30, label = "Gemma T2I", options = {}) {
    const missing = t2iMissingReason(segment);
    if (missing) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${missing}`);
    state.activeId = segment.id;
    syncInspector();
    progress?.set(`${label}: preparing Gemma input...`, percent);
    const useVision = Boolean(segment.use_vision_reference);
    const gemmaSelect = state.imageModelMode === "ernie_image"
      ? (useVision ? ernieGemmaModelSelect : ernieTextGemmaModelSelect)
      : (useVision ? gemmaModelSelect : t2iTextGemmaModelSelect);
    const mmprojSelectForMode = state.imageModelMode === "ernie_image" ? ernieMmprojSelect : mmprojSelect;
    const data = await postJson("/vrgdg/music_builder/generate_t2i", {
      model_file: gemmaSelect.value,
      mmproj_file: useVision ? mmprojSelectForMode.value : "",
      use_vision: useVision,
      ref_image_path: segment.ref_image_path || "",
      user_notes: segment.notes || "",
      theme_style_path: state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: options.unloadAfter !== false,
    }, useVision ? 10 * 60 * 1000 : 120000);
    pushHistory();
    segment.t2i_prompt = applyTriggerPhrase(data.prompt, imageTriggerPhraseForSegment(segment));
    segment.enhance_prompt = segment.t2i_prompt;
    t2iPrompt.value = segment.t2i_prompt;
    ernieT2IPrompt.value = segment.t2i_prompt;
    fluxPrompt.value = segment.t2i_prompt;
    zEnhancePromptPreview.value = segment.enhance_prompt;
    render();
    return data;
  }

  async function createZImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "ZImage") {
    state.activeId = segment.id;
    syncInspector();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
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
    try {
      setButtonGroupState(zCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating ZImage preview");
      progress.set("Autosaving session/SRT before ZImage...", 8);
      await autoSaveSessionQuiet("ZImage preview");
      await createZImageForSegment(segment, progress, 15, 75, "ZImage preview");
      await autoSaveSessionQuiet("ZImage preview complete");
      progress.set("ZImage preview ready.", 100);
      progress.close(900);
      toast("ZImage preview ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(zCreateButtons, { disabled: false, text: "Create Z-Image" });
    }
  }

  async function createErnieImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Ernie") {
    state.activeId = segment.id;
    syncInspector();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
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
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
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
      progress.set("Combining image ingredients for Gemma vision...", 25);
      const data = await postJson("/vrgdg/music_builder/generate_flux_klein_prompt", {
        model_file: fluxGemmaModelSelect.value,
        mmproj_file: fluxMmprojSelect.value,
        image_ingredients: settings.image_ingredients || [],
        user_notes: settings.notes || "",
        unload_after: true,
      }, FLUX_GEMMA_TIMEOUT_MS);
      pushHistory();
      segment.flux_prompt = applyTriggerPhrase(data.prompt, imageTriggerPhraseForSegment(segment));
      fluxPrompt.value = segment.flux_prompt;
      segment.t2i_prompt = segment.flux_prompt;
      segment.enhance_prompt = segment.flux_prompt;
      t2iPrompt.value = segment.t2i_prompt;
      ernieT2IPrompt.value = segment.t2i_prompt;
      fluxPrompt.value = segment.t2i_prompt;
      zEnhancePromptPreview.value = segment.enhance_prompt;
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

  async function generateFluxKleinPromptForSegment(segment, progress = null, percent = 25, label = "Flux/Klein Gemma", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = fluxKleinSettingsForSegment(segment);
    if (!Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: add at least one global or scene Flux/Klein image ingredient.`);
    }
    progress?.set(`${label}: combining global and scene image ingredients for Gemma vision...`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_flux_klein_prompt", {
      model_file: fluxGemmaModelSelect.value,
      mmproj_file: fluxMmprojSelect.value,
      image_ingredients: settings.image_ingredients || [],
      user_notes: settings.notes || segment.notes || "",
      clear_before_load: options.clearBeforeLoad !== false,
      unload_after: options.unloadAfter !== false,
    }, FLUX_GEMMA_TIMEOUT_MS);
    pushHistory();
    segment.flux_prompt = applyTriggerPhrase(data.prompt, imageTriggerPhraseForSegment(segment));
    segment.t2i_prompt = segment.flux_prompt;
    segment.enhance_prompt = segment.flux_prompt;
    if (segment.id === activeSegment()?.id) {
      fluxPrompt.value = segment.flux_prompt;
      t2iPrompt.value = segment.flux_prompt;
      ernieT2IPrompt.value = segment.flux_prompt;
      zEnhancePromptPreview.value = segment.enhance_prompt;
    }
    render();
    return data;
  }

  async function createFluxKleinImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Flux/Klein") {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = fluxKleinSettingsForSegment(segment);
    const prompt = String(settings.prompt || segment.flux_prompt || segment.t2i_prompt || "").trim();
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
    segment.t2i_prompt = prompt;
    segment.enhance_prompt = prompt;
    segment.flux_prompt = prompt;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) {
      t2iPrompt.value = prompt;
      ernieT2IPrompt.value = prompt;
      fluxPrompt.value = prompt;
      zEnhancePromptPreview.value = prompt;
      syncPreview(segment);
    }
    render();
    return images;
  }

  async function previewFluxKleinImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveFluxKleinSettingsFromPanel();
    const prompt = String(settings.prompt || fluxPrompt.value || "").trim();
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
      segment.t2i_prompt = prompt;
      segment.enhance_prompt = prompt;
      segment.flux_prompt = prompt;
      segment.custom_image_path = "";
      segment.custom_image_data = "";
      segment.custom_image_name = "";
      segment.approved_image_path = "";
      segment.preview_mode = "image";
      t2iPrompt.value = prompt;
      ernieT2IPrompt.value = prompt;
      fluxPrompt.value = prompt;
      zEnhancePromptPreview.value = prompt;
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
      progress.set("Reading selected image with Gemma vision...", 20);
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
    const useImageReference = segment.use_i2v_vision_reference !== false;
    const imageReference = useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
    if (useImageReference && !imageReference.path && !imageReference.data) {
      toast("Hey, you need a scene image first. Save/load an image, or turn off image reference to create I2V from the T2I prompt instead.", true);
      return;
    }
    if (!useImageReference && !segment.t2i_prompt) {
      toast("Hey, you need a T2I prompt first. Create/type one, or turn image reference back on and use a saved/custom image.", true);
      return;
    }
    let progress = null;
    try {
      createI2VButton.disabled = true;
      createI2VButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating I2V prompt");
      progress.set("Autosaving session/SRT before Gemma I2V...", 8);
      await autoSaveSessionQuiet("Gemma I2V");
      progress.set(useImageReference ? "Preparing image reference and motion notes..." : "Preparing T2I prompt and motion notes...", 20);
      progress.set(useImageReference ? "Running Gemma vision I2V prompt generation..." : "Running Gemma text-only I2V prompt generation...", 50);
      const data = await postJson("/vrgdg/music_builder/generate_i2v", {
        model_file: useImageReference ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
        mmproj_file: useImageReference ? i2vMmprojSelect.value : "",
        t2i_prompt: useImageReference ? "" : segment.t2i_prompt,
        image_reference_path: imageReference.path,
        image_reference_data: imageReference.data,
        user_notes: segment.i2v_notes || "",
        theme_style_path: useImageReference ? "" : state.useVrgdgTextContext ? state.themeStylePath || "" : "",
        story_idea_path: useImageReference ? "" : state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
        subject_scene_path: useImageReference ? "" : state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
        unload_after: true,
      });
      pushHistory();
      segment.i2v_prompt = applyTriggerPhrase(data.prompt, videoTriggerPhraseForSegment(segment));
      i2vPrompt.value = segment.i2v_prompt;
      render();
      await autoSaveSessionQuiet("Gemma I2V complete");
      progress.set("I2V prompt ready.", 100);
      progress.close(900);
      toast(data.used_image_reference ? "Gemma created I2V prompt from the image reference." : "Gemma created I2V prompt from the T2I prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createI2VButton.disabled = false;
      createI2VButton.textContent = "Gemma I2V";
    }
  }

  async function generateTextOnlyI2VPromptForSegment(segment, progress = null, percent = 50, label = "Gemma I2V", options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    const t2iText = String(segment.t2i_prompt || "").trim();
    if (!t2iText) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: converting T2I prompt to I2V prompt without vision...`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_i2v", {
      model_file: i2vTextGemmaModelSelect.value,
      mmproj_file: "",
      t2i_prompt: t2iText,
      image_reference_path: "",
      image_reference_data: "",
      user_notes: segment.i2v_notes || "",
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
    const useImageReference = segment.use_i2v_vision_reference !== false;
    const imageReference = useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
    const t2iText = String(segment.t2i_prompt || "").trim();
    if (useImageReference && !imageReference.path && !imageReference.data) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: I2V image reference is enabled, but no scene image was found.`);
    }
    if (!useImageReference && !t2iText) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    }
    progress?.set(useImageReference
      ? `${label}: creating I2V prompt from scene image and motion notes...`
      : `${label}: converting T2I prompt to I2V prompt without vision...`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_i2v", {
      model_file: useImageReference ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
      mmproj_file: useImageReference ? i2vMmprojSelect.value : "",
      t2i_prompt: useImageReference ? "" : t2iText,
      image_reference_path: imageReference.path,
      image_reference_data: imageReference.data,
      user_notes: segment.i2v_notes || "",
      theme_style_path: useImageReference ? "" : state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: useImageReference ? "" : state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: useImageReference ? "" : state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: options.unloadAfter !== false,
    });
    pushHistory();
    segment.i2v_prompt = applyTriggerPhrase(data.prompt, videoTriggerPhraseForSegment(segment));
    if (!segment.i2v_prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty I2V prompt.`);
    if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt;
    render();
    return data;
  }

  async function i2vAllScenes(options = {}) {
    const progress = options.progress || createProgressWindow("Gemma I2V All Scenes");
    const closeProgress = !options.progress;
    const allScenes = allEditableSegments();
    const redoPrompts = options.i2vRunMode === "redo_prompts";
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
      const useImageReference = segment.use_i2v_vision_reference !== false;
      const imageReference = useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
      if (useImageReference && !imageReference.path && !imageReference.data) {
        missing.push(`${sceneDisplayName(segment, index)}: I2V image reference is enabled, but no scene image was found.`);
      }
      if (!useImageReference && !String(segment.t2i_prompt || "").trim()) {
        missing.push(`${sceneDisplayName(segment, index)}: T2I prompt is missing.`);
      }
    });
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Gemma I2V All cannot start yet.</div>
          <div>Fix these first:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      createI2VButton.disabled = true;
      progress.set("Autosaving session/SRT before Gemma I2V All...", 3);
      await saveSessionForSceneVideo();
      if (!scenes.length) {
        progress.set("All scenes already have I2V prompts. Skipping Gemma I2V All.", 100);
        if (closeProgress) progress.close(1800);
        toast("All scenes already have I2V prompts. Gemma I2V skipped.");
        return;
      }
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const segment = scenes[index];
        state.activeId = segment.id;
        syncInspector();
        render();
        const base = Math.floor((index / scenes.length) * 100);
        const useImageReference = segment.use_i2v_vision_reference !== false;
        const displayIndex = segmentIndexInfo(segment).index;
        progress.set(`Gemma I2V All ${index + 1}/${scenes.length}: ${sceneDisplayName(segment, displayIndex)}\n${useImageReference ? "Using scene image reference." : "Using T2I prompt text only."}`, base);
        await generateI2VPromptForSegment(segment, progress, Math.min(98, base + 30), `Gemma I2V All ${index + 1}/${scenes.length}`, { unloadAfter: false });
        await autoSaveSessionQuiet(`Gemma I2V All ${sceneDisplayName(segment, displayIndex)}`);
      }
      await runClearMemoryWorkflowQuiet(progress, "Gemma I2V prompt pass", 96);
      await autoSaveSessionQuiet("Gemma I2V All complete");
      progress.set("Gemma I2V All complete.", 100);
      if (closeProgress) progress.close(1800);
    } catch (error) {
      progress.set(`Stopped/Error:\n${String(error?.message || error)}`, 100);
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
    } finally {
      createI2VButton.disabled = false;
    }
  }

  function i2vImagesFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\zimage_approved`;
  }

  function i2vVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\image_to_video_clips`;
  }

  function collectedSceneVideoFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\rendered_scene_videos`;
  }

  function sceneVideoDetailsHtml(segment, sceneIndex, srtPath, outputFolder, statusText = "Preparing hidden I2V workflow...", details = {}) {
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
        ${imageSrc ? `<img src="${imageSrc}" style="width:180px;max-height:110px;object-fit:cover;border:1px solid #155e75;border-radius:6px;background:#050505;">` : ""}
        <div style="display:grid;grid-template-columns:150px minmax(0,1fr);gap:5px 10px;font-size:11px;">
          <div style="color:#67e8f9;font-weight:900;">Scene</div><div>${escapeHtml(segment.label || `Scene ${promptNumber}`)}</div>
          <div style="color:#67e8f9;font-weight:900;">Image index</div><div>${imageIndex} (0 based)</div>
          <div style="color:#67e8f9;font-weight:900;">SRT prompt #</div><div>${promptNumber} (1 based)</div>
          <div style="color:#67e8f9;font-weight:900;">Audio mode</div><div>${escapeHtml(audioMode)}</div>
          <div style="color:#67e8f9;font-weight:900;">Image folder</div><div style="overflow-wrap:anywhere;">${escapeHtml(i2vImagesFolder())}</div>
          <div style="color:#67e8f9;font-weight:900;">Image path</div><div style="overflow-wrap:anywhere;">${escapeHtml(imagePath || imageSource?.name || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Audio sent to LTX</div><div style="overflow-wrap:anywhere;">${escapeHtml(audioPath)}</div>
          <div style="color:#67e8f9;font-weight:900;">SRT path</div><div style="overflow-wrap:anywhere;">${escapeHtml(srtPath || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Save folder</div><div style="overflow-wrap:anywhere;">${escapeHtml(outputFolder || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Collected clips</div><div style="overflow-wrap:anywhere;">${escapeHtml(collectedSceneVideoFolder())}</div>
        </div>
        <div>
          <div style="color:#67e8f9;font-weight:900;margin-bottom:4px;">I2V prompt</div>
          <div style="border:1px solid #155e75;border-radius:6px;background:#020617;color:#e0f2fe;padding:8px;max-height:130px;overflow:auto;white-space:pre-wrap;">${escapeHtml(segment.i2v_prompt || "")}</div>
        </div>
      </div>
    `;
  }

  function i2vVideoSettingsPayload() {
    const settings = saveI2VVideoSettingsFromPanel();
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
    if (!segmentImageSource(segment)) missing.push(`${name}: selected scene image is missing.`);
    if (!String(segment?.i2v_prompt || "").trim()) missing.push(`${name}: I2V prompt is missing.`);
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
    const missing = validateSceneReadyForVideo(segment, sceneIndex);
    if (missing.length) throw new Error(missing.join("\n"));
    segment.video_status = "running";
    renderList();
    progress?.set(`${batchLabel}Saving current UI session/SRT timing...`, pct(8));
    let srtPath = await saveSessionForSceneVideo();
    if (!srtPath) throw new Error("The builder SRT path was not created.");
    progress?.set(`${batchLabel}Preparing selected scene image for I2V...`, pct(12));
    await ensureSelectedImageForSceneVideo(segment, sceneIndex);
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
    progress?.set(`${batchLabel}Checking SRT timing before hidden I2V...`, pct(14));
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
      ...i2vVideoSettingsPayload(),
      i2v_prompt: segment.i2v_prompt,
      audio_path: audioPathForScene,
      image_folder: i2vImagesFolder(),
      image_index_zero_based: slotNumber - 1,
      prompt_number_one_based: promptNumberForScene,
      srt_path: srtPath,
      project_folder: projectInput.value,
    };
    const workflowDetails = {
      audioPath: audioPathForScene,
      promptNumber: promptNumberForScene,
      audioMode: audioModeForScene,
    };
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, i2vVideoOutputFolder(), `${batchLabel}Preparing hidden I2V workflow...\nSRT timing verified: ${timingCheck.srt_duration.toFixed(3)}s`, workflowDetails), pct(15));
    const built = await postJson("/vrgdg/workflow_runner/build_i2v_prompt", payload);
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || i2vVideoOutputFolder(), `${batchLabel}Queueing hidden I2V workflow...`, workflowDetails), pct(40));
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the video but did not return a prompt_id.");
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || i2vVideoOutputFolder(), `${batchLabel}Queued prompt ID: ${promptId}\nWaiting for video...`, workflowDetails), pct(60));
    const videos = await waitForVideos(
      promptId,
      (message) => progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || i2vVideoOutputFolder(), `${batchLabel}${message}\nPrompt ID: ${promptId}`, workflowDetails), pct(80)),
      () => state.batchCancelled
    );
    const video = videos[videos.length - 1] || null;
    const videoPath = resolveComfyVideoPath(video);
    if (!videoPath) throw new Error("The I2V workflow finished, but no video path was found in history.");
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || i2vVideoOutputFolder(), `${batchLabel}Collecting scene video into builder folder...`, workflowDetails), pct(90));
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

  async function stitchRenderedScenes(progress) {
    const paths = state.segments.map((segment) => String(selectedSegmentVideoPath(segment) || "").trim());
    const overlayItems = state.overlaySegments
      .filter((segment) => String(selectedSegmentVideoPath(segment) || "").trim())
      .map((segment, index) => ({
        path: String(selectedSegmentVideoPath(segment) || "").trim(),
        start: Number(segment.start || 0),
        end: Number(segment.end || 0),
        label: segment.label || `Insert ${index + 1}`,
      }));
    const sceneAudioMode = usingSceneAudioMode();
    const audioPaths = sceneAudioMode ? state.segments.map((segment) => String(segment.custom_audio_path || "").trim()) : [];
    const audioItems = sceneAudioMode ? state.segments.map((segment) => ({
      path: String(segment.custom_audio_path || "").trim(),
      start: audioSourceStart(segment),
      duration: audioChunkDuration(segment),
    })) : [];
    const missing = [];
    paths.forEach((path, index) => {
      if (!path) missing.push(`${sceneDisplayName(state.segments[index], index)}: rendered scene video is missing.`);
      if (sceneAudioMode && !audioPaths[index]) missing.push(`${sceneDisplayName(state.segments[index], index)}: scene audio is missing.`);
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
        if (randomizeVideoSeed) setVideoSeedRandom();
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
        progress.set(`Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}\nKeeping Gemma loaded until this prompt pass finishes...`, base);
        await generateT2IPromptForSegment(segment, progress, Math.min(38, base + 3), `Image All prompt pass ${index + 1}/${promptScenes.length}: Gemma`, { unloadAfter: false });
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
        progress.set(`Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}\nKeeping Gemma loaded until this prompt pass finishes...`, base);
        await generateT2IPromptForSegment(segment, progress, Math.min(38, base + 3), `Image All prompt pass ${index + 1}/${promptScenes.length}: Gemma`, { unloadAfter: false });
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
    const hasAnyGlobal = Array.isArray(state.fluxGlobalImageIngredients) && state.fluxGlobalImageIngredients.length > 0;
    if (!hasAnyGlobal && !allEditableSegments().some((segment) => Array.isArray(segment.flux_image_ingredients) && segment.flux_image_ingredients.length)) {
      const message = "Flux/Klein All needs at least one global or scene image ingredient.";
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
          progress.set(`Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}\nKeeping Gemma loaded until this prompt pass finishes...`, base);
          await generateFluxKleinPromptForSegment(
            segment,
            progress,
            Math.min(38, base + 3),
            `Image All prompt pass ${index + 1}/${promptScenes.length}: Gemma`,
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

  async function buildFullVideoPipeline(options = {}) {
    const buildMode = options.buildMode || "resume_missing";
    let progress = null;
    try {
      fullBuildButton.disabled = true;
      fullBuildButton.textContent = "Building...";
      renderAllButton.disabled = true;
      zImageAllButton.disabled = true;
      state.batchCancelled = false;
      progress = createProgressWindow("Build Full Video");
      const imageStage = (state.imageModelMode || "") === "flux_klein" ? "Flux/Klein image pass" : state.imageModelMode === "ernie_image" ? "Ernie image pass" : "Z-Image pass";
      progress.set(`Stage 1/3: ${imageStage}...`, 5);
      const imageMode = state.imageModelMode || "zimage";
      const imageRunMode = buildMode === "fresh_rebuild" ? "redo_prompts_images" : "resume_missing";
      if (imageMode === "flux_klein") {
        await fluxKleinAllScenes({ throwOnError: true, imageRunMode });
      } else if (imageMode === "ernie_image") {
        await ernieImageAllScenes({ throwOnError: true, imageRunMode });
      } else {
        await zImageAllScenes({ throwOnError: true, imageRunMode });
      }
      assertBatchNotStopped();
      progress.set("Stage 2/3: creating I2V prompts...", 38);
      await i2vAllScenes({
        throwOnError: true,
        i2vRunMode: buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" ? "redo_prompts" : "resume_missing",
      });
      assertBatchNotStopped();
      progress.set("Stage 3/3: rendering and stitching scene videos...", 68);
      const shouldRandomizeVideoSeed = buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" || buildMode === "redo_videos";
      await renderAllScenes({
        forceVideos: buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" || buildMode === "redo_videos",
        randomizeVideoSeed: shouldRandomizeVideoSeed && options.videoSeedMode !== "keep",
      });
      progress.set("Build Full Video complete.", 100);
      progress.close(4500);
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
    state.zEnhanceSettings = defaultZEnhanceSettings();
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

  function chooseBatchModeAction({ title, intro = "", choices = [], confirmLabel = "Continue", extraGroups = [], returnAll = false } = {}) {
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
      let selected = choices[0]?.value || "";
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
        input.checked = index === 0;
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
    const useErnieMode = imageMode === "ernie_image";
    const modelLabel = useFluxKleinMode ? "Flux/Klein" : useErnieMode ? "Ernie" : "ZImage";
    const mode = await chooseBatchModeAction({
      title: "Run Image All?",
      intro: `Image All only works on the image stage. It does not create I2V prompts, render videos, or stitch the final video. Current image model: ${modelLabel}. Flux ingredients, model selections, LoRAs, notes, and project paths are not reset.`,
      confirmLabel: "Run Image All",
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
    });
    if (!mode) return;
    if (useFluxKleinMode) await fluxKleinAllScenes({ imageRunMode: mode });
    else if (useErnieMode) await ernieImageAllScenes({ imageRunMode: mode });
    else await zImageAllScenes({ imageRunMode: mode });
  }

  async function confirmAndRunRenderAll() {
    const ok = await confirmLongBatchAction({
      title: "Run Render All?",
      lines: [
        "Render All only works on the video/render stage.",
        "It uses the current selected images and existing I2V prompts.",
        "Scenes that already have a selected video are skipped.",
        "Scenes missing video are rendered, then the final video is stitched.",
        "If every scene already has video, this only stitches the final video.",
      ],
      confirmLabel: "Run Render All",
    });
    if (ok) await renderAllScenes();
  }

  async function confirmAndRunFullBuild() {
    const options = await chooseBatchModeAction({
      title: "Build Full Video?",
      intro: "Build Full Video can run the whole pipeline: image prompts, images, I2V prompts, scene videos, and final stitching. Choose how much to regenerate. Flux ingredients, model selections, LoRAs, notes, and project paths are not reset.",
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
          description: "Start fresh for generated outputs. Regenerate image prompts, images, I2V prompts, and videos. Image seeds are randomized.",
        },
        {
          value: "redo_i2v_prompts_videos",
          label: "Keep images, redo I2V prompts and videos",
          description: "Use the current selected images. Regenerate I2V prompts with Gemma, then create new video versions.",
        },
        {
          value: "redo_videos",
          label: "Keep images and prompts, redo videos",
          description: "Use the current selected images and existing I2V prompts. Only create new video versions, then stitch.",
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
  useSceneI2VVideoSettingsModels.input.addEventListener("change", () => {
    setSceneI2VVideoSettingsEnabled(Boolean(useSceneI2VVideoSettingsModels.input.checked));
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
  promptOptionsButton.onclick = openPromptOptionsModal;
  autoLoadAllButton.onclick = autoLoadAll;
  clearMemoryButton.onclick = runClearMemoryWorkflow;
  renderAllButton.onclick = confirmAndRunRenderAll;
  zImageAllButton.onclick = confirmAndRunZImageAll;
  fullBuildButton.onclick = confirmAndRunFullBuild;
  remakeModeButton.onclick = showRemakeModeComingSoon;
  stopWorkflowButton.onclick = stopCurrentWorkflow;
  downloadModelsButton.onclick = showModelDownloadModal;
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
  for (const button of createSceneVideoButtons) button.onclick = createSceneVideo;
  loadCustomImageButton.onclick = loadCustomImage;
  for (const button of zCreateButtons) button.onclick = previewZImage;
  for (const button of ernieCreateButtons) button.onclick = previewErnieImage;
  for (const button of fluxCreateButtons) button.onclick = previewFluxKleinImage;
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
  };
  fluxKleinCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "flux_klein";
    state.fluxKleinSettings.image_model_mode = "flux_klein";
    state.fluxKleinSettings.enabled = true;
    syncFluxKleinPanel();
  };
  ernieImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "ernie_image";
    state.fluxKleinSettings.image_model_mode = "ernie_image";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
  };
  zEnhanceCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "z_enhance";
    state.fluxKleinSettings.image_model_mode = "z_enhance";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
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
  fluxGlobalIngredientClearButton.onclick = () => {
    pushHistory();
    state.fluxGlobalImageIngredients = [];
    renderFluxGlobalIngredientList();
    render();
    toast("Global Flux/Klein image ingredients cleared.");
  };
  useFluxGlobalIngredients.input.addEventListener("change", () => {
    pushHistory();
    state.useFluxGlobalImageIngredients = Boolean(useFluxGlobalIngredients.input.checked);
    syncFluxGlobalIngredientPanel();
    render();
  });
  fluxIngredientClearButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    pushHistory();
    segment.flux_image_ingredients = [];
    renderFluxIngredientList(segment);
    render();
    toast("Flux/Klein image ingredients cleared for this scene.");
  };
  enableFluxIngredientDrop(fluxGlobalIngredientDrop, { global: true });
  enableFluxIngredientDrop(fluxIngredientDrop);
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
  refImageLoadButton.onclick = () => visionRefFileInput.click();
  ernieRefImageLoadButton.onclick = () => visionRefFileInput.click();
  visionRefFileInput.addEventListener("change", () => {
    loadVisionReferenceFile(visionRefFileInput.files?.[0]);
    visionRefFileInput.value = "";
  });
  function wireVisionReferenceDrop(dropElement) {
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
        setVisionReferenceSource(sceneSource).catch((error) => toast(String(error?.message || error), true));
        return;
      }
      const file = imageFileFromDrop(event);
      if (!file) return;
      event.preventDefault();
      event.stopPropagation();
      dropElement.style.borderColor = "#155e75";
      loadVisionReferenceFile(file);
    });
  }
  wireVisionReferenceDrop(refImageDrop);
  wireVisionReferenceDrop(ernieRefImageDrop);
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
    for (const select of [t2iTextGemmaModelSelect, gemmaModelSelect, ernieTextGemmaModelSelect, ernieGemmaModelSelect, zEnhanceGemmaModelSelect, i2vTextGemmaModelSelect, i2vGemmaModelSelect, fluxGemmaModelSelect]) {
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
    for (const select of [mmprojSelect, ernieMmprojSelect, zEnhanceMmprojSelect, i2vMmprojSelect, fluxMmprojSelect]) {
      select.textContent = "";
      for (const item of mmproj) {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        select.append(option);
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
    for (const slot of [...zLoraSlots, ...ernieLoraSlots, ...i2vLoraSlots, ...zEnhanceLoraSlots]) {
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
  for (const control of [fluxNotes, fluxPrompt, fluxWidth, fluxHeight, fluxSeed]) {
    control.addEventListener("input", saveFluxKleinSettingsFromPanel);
    control.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
  useFluxKlein.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
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
