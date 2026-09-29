import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "VRGDGMiniMaxUpscalerControlPanel";
const UPLOAD_URL = "/vrgdg/long_video/upload";
const CHUNK_SIZE = 50 * 1024 * 1024;

const FIELD_NAMES = [
  "video_path", "video_mode", "video_denoise", "overlap_preset", "custom_window",
  "custom_overlap", "blend_mode", "steps", "sampler_denoise", "sampler", "seed",
  "width", "height", "force_rate", "frame_load_cap", "tile_batch_size",
];

function panelValue(node, name) {
  return node.widgets?.find((item) => item.name === name)?.value;
}

function setValue(node, name, value) {
  const target = node.widgets?.find((item) => item.name === name);
  if (!target) return false;
  target.value = value;
  target.callback?.(value);
  return true;
}

function allNodes() {
  return app.graph?._nodes || [];
}

function firstNode(predicate) {
  return allNodes().find(predicate);
}

function byType(...types) {
  return firstNode((node) => types.includes(node.comfyClass || node.type));
}

function byTitle(text) {
  const needle = text.toLowerCase();
  return firstNode((node) => String(node.title || "").toLowerCase().includes(needle));
}

function applyToWorkflow(panel) {
  const video = byType("VRGDGLongVideoMetaBatchLoader", "VHS_LoadVideo", "VHS_LoadVideoPath") || byTitle("upload video");
  const preset = byType("VRGDGOverlapPreset") || byTitle("overlap preset");
  const avPrepare = byType("MiniMaxH3SourceAVPrepareT8") || byTitle("lock source audio");
  const scheduler = byType("BasicScheduler") || byTitle("steps / ");
  const sampler = byType("KSamplerSelect") || byTitle("solver");
  const noise = byType("RandomNoise") || byTitle("fixed seed");
  const decoder = byType("H3FastVAEDecode") || byTitle("vae decode fast");

  const videoPath = panelValue(panel, "video_path");
  if (video && videoPath) setValue(video, "video", videoPath);
  if (video) {
    setValue(video, "custom_width", Number(panelValue(panel, "width")) || 0);
    setValue(video, "custom_height", Number(panelValue(panel, "height")) || 0);
    setValue(video, "force_rate", Number(panelValue(panel, "force_rate")) || 0);
    setValue(video, "frame_load_cap", Number(panelValue(panel, "frame_load_cap")) || 0);
  }
  if (preset) {
    setValue(preset, "preset", panelValue(panel, "overlap_preset"));
    setValue(preset, "custom_window", Number(panelValue(panel, "custom_window")));
    setValue(preset, "custom_overlap", Number(panelValue(panel, "custom_overlap")));
    setValue(preset, "blend_mode", panelValue(panel, "blend_mode"));
  }
  if (avPrepare) {
    setValue(avPrepare, "video_mode", panelValue(panel, "video_mode"));
    setValue(avPrepare, "video_denoise_strength", Number(panelValue(panel, "video_denoise")));
    setValue(avPrepare, "denoise", Number(panelValue(panel, "video_denoise")));
  }
  if (scheduler) {
    setValue(scheduler, "steps", Number(panelValue(panel, "steps")));
    setValue(scheduler, "denoise", Number(panelValue(panel, "sampler_denoise")));
  }
  if (sampler) setValue(sampler, "sampler_name", panelValue(panel, "sampler"));
  if (noise) setValue(noise, "noise_seed", Number(panelValue(panel, "seed")));
  if (decoder) setValue(decoder, "tile_batch_size", Number(panelValue(panel, "tile_batch_size")));
  app.graph?.setDirtyCanvas(true, true);
  return { video, preset, avPrepare, scheduler, sampler, noise, decoder };
}

async function uploadVideo(file, panel, status) {
  const uploadId = crypto.randomUUID();
  const total = Math.ceil(file.size / CHUNK_SIZE);
  for (let chunk = 0; chunk < total; chunk += 1) {
    status.textContent = `Uploading ${chunk + 1}/${total}…`;
    const response = await api.fetchApi(
      `${UPLOAD_URL}?upload_id=${encodeURIComponent(uploadId)}&chunk=${chunk}&total=${total}&filename=${encodeURIComponent(file.name)}`,
      { method: "POST", body: file.slice(chunk * CHUNK_SIZE, Math.min(file.size, (chunk + 1) * CHUNK_SIZE)) },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) throw new Error(data.error || response.statusText || "Upload failed.");
    if (data.complete) setValue(panel, "video_path", data.path);
  }
  status.textContent = `Loaded: ${file.name}`;
}

function makeModal(panel) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;font:14px sans-serif;color:#eee";
  const box = document.createElement("div");
  box.style.cssText = "background:#202633;border:1px solid #65728a;border-radius:12px;padding:20px;width:560px;max-height:90vh;overflow:auto;box-shadow:0 12px 50px #000";
  const title = document.createElement("h2");
  title.textContent = "MiniMax H3 Upscaler Settings";
  title.style.marginTop = "0";
  box.append(title);
  const status = document.createElement("div");
  status.style.cssText = "min-height:22px;color:#9ed0ff;margin:8px 0";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v,.gif";
  fileInput.style.display = "none";
  const loadButton = document.createElement("button");
  loadButton.textContent = "Load Video";
  loadButton.onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    if (!fileInput.files?.[0]) return;
    try { await uploadVideo(fileInput.files[0], panel, status); }
    catch (error) { status.textContent = `Upload failed: ${error.message || error}`; }
  };
  box.append(loadButton, fileInput, status);

  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin-top:12px";
  for (const name of FIELD_NAMES) {
    const source = panel.widgets?.find((item) => item.name === name);
    if (!source) continue;
    const label = document.createElement("label");
    label.textContent = name.replaceAll("_", " ");
    label.style.cssText = "display:flex;flex-direction:column;gap:4px;text-transform:capitalize";
    const input = document.createElement(source.type === "combo" ? "select" : "input");
    if (source.type === "combo") {
      for (const option of source.options?.values || []) {
        const item = document.createElement("option"); item.value = option; item.textContent = option; input.append(item);
      }
    } else {
      input.type = typeof source.value === "number" ? "number" : "text";
      if (input.type === "number") input.step = String(source.options?.step || (String(source.value).includes(".") ? 0.01 : 1));
    }
    input.value = source.value ?? "";
    input.oninput = () => {
      let value = input.value;
      if (input.type === "number") value = Number(value);
      setValue(panel, name, value);
    };
    label.append(input); grid.append(label);
  }
  box.append(grid);
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;justify-content:flex-end;gap:10px;margin-top:18px";
  const cancel = document.createElement("button"); cancel.textContent = "Cancel"; cancel.onclick = () => overlay.remove();
  const apply = document.createElement("button"); apply.textContent = "Apply & Close"; apply.onclick = () => { const found = applyToWorkflow(panel); status.textContent = `Applied to ${Object.values(found).filter(Boolean).length} workflow nodes.`; setTimeout(() => overlay.remove(), 500); };
  actions.append(cancel, apply); box.append(actions); overlay.append(box);
  overlay.onclick = (event) => { if (event.target === overlay) overlay.remove(); };
  return overlay;
}

app.registerExtension({
  name: "vrgdg.minimax_upscaler_control_panel",
  nodeCreated(node) {
    if ((node.comfyClass || node.type) !== NODE_TYPE || node._vrgdgPanelButton) return;
    const button = node.addWidget("button", "Open H3 Settings", null, () => document.body.append(makeModal(node)));
    node._vrgdgPanelButton = button;
  },
});
