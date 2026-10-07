import { getJson, makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeInput, makeMiniButton, toast } from "./controls.mjs";
import {
  ERNIE_MODEL_DOWNLOADS,
  FLUX_KLEIN_4B_MODEL_DOWNLOADS,
  FLUX_KLEIN_9B_MODEL_DOWNLOADS,
  GEMMA_LLM_MODEL_DOWNLOADS,
  KREA2_MODEL_DOWNLOADS,
  LTX_23_MODEL_DOWNLOADS,
  LTX_25_MODEL_DOWNLOADS,
  MINIMAX_H3_MODEL_DOWNLOADS,
  MODEL_FOLDER_HINTS,
  QWEN_LLM_MODEL_DOWNLOADS,
  VIDEO_BUILDER_CUSTOM_NODES,
  ZIMAGE_MODEL_DOWNLOADS,
} from "./models.mjs";

// Batch prompt generation keeps recoverable Gemma failures per scene so one
// bad response does not discard the successful scenes from the same pass.
export function gemmaBatchFailureStore() {
  if (!window.__vrgdgGemmaBatchFailures) window.__vrgdgGemmaBatchFailures = {};
  return window.__vrgdgGemmaBatchFailures;
}

export function recordGemmaBatchFailure(key, segment, sceneLabel, error, debugPath = "") {
  const store = gemmaBatchFailureStore();
  const raw = String(error?.rawGemmaPrompt ?? error?.rawPrompt ?? "").trim();
  const cleaned = String(error?.cleanedGemmaPrompt ?? error?.cleanedPrompt ?? "").trim();
  store[key] = {
    key,
    segmentId: String(segment?.id || ""),
    sceneLabel: sceneLabel || "Unknown scene",
    error: String(error?.message || error || "Unknown Gemma error"),
    raw: raw || cleaned || "(raw Gemma output was not attached)",
    cleaned,
    debugPath: String(debugPath || error?.gemmaDebugPath || ""),
    timestamp: new Date().toISOString(),
  };
  return store[key];
}

export function looksLikeUnfilledMiniMaxTemplate(text) {
  const value = String(text || "").toLowerCase();
  return /\[(?:subject|setting(?:\/environment)?|environment|time(?:\/weather)?|weather|camera motion|dynamic performance|subject visibility|framing|clothing|hair)\]/i.test(value);
}

export function showGemmaBatchFailures(failures, { retryHandler }) {
  const items = Array.isArray(failures) ? failures : [];
  if (!items.length) return;
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:18px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(980px,calc(100vw - 36px));max-height:calc(100vh - 36px);overflow:auto;border:1px solid #991b1b;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 22px 80px rgba(0,0,0,.65);padding:16px;box-sizing:border-box;";
  const title = document.createElement("div");
  title.innerHTML = `<div style="font-size:17px;font-weight:900;color:#fecaca;">LLM skipped ${items.length} scene${items.length === 1 ? "" : "s"}</div><div style="font-size:12px;color:#cbd5e1;margin-top:5px;">Successful scenes were kept. Only these scenes will be retried.</div>`;
  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:12px;margin-top:14px;";
  items.forEach((item) => {
    const card = document.createElement("details");
    card.open = true;
    card.style.cssText = "border:1px solid #7f1d1d;border-radius:7px;background:#1f0808;padding:9px;";
    const summary = document.createElement("summary");
    summary.style.cssText = "cursor:pointer;font-weight:900;color:#fca5a5;";
    summary.textContent = `${item.sceneLabel}: ${item.error}`;
    const raw = document.createElement("pre");
    raw.style.cssText = "white-space:pre-wrap;word-break:break-word;max-height:240px;overflow:auto;margin:9px 0 0;color:#fecaca;font-size:11px;line-height:1.4;";
    raw.textContent = item.raw;
    card.append(summary, raw);
    if (item.debugPath) {
      const path = document.createElement("div");
      path.style.cssText = "margin-top:7px;color:#fda4af;font-size:11px;word-break:break-all;";
      path.textContent = `Debug file: ${item.debugPath}`;
      card.append(path);
    }
    list.append(card);
  });
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:16px;";
  const close = makeButton("Close");
  const retry = makeButton(`Retry ${items.length} Failed Scene${items.length === 1 ? "" : "s"}`);
  retry.style.background = "#7f1d1d";
  retry.onclick = async () => {
    retry.disabled = true;
    retry.textContent = "Retrying...";
    try {
      await retryHandler(items);
      backdrop.remove();
    } catch (error) {
      toast(String(error?.message || error), true);
      retry.disabled = false;
      retry.textContent = `Retry ${items.length} Failed Scene${items.length === 1 ? "" : "s"}`;
    }
  };
  close.onclick = () => backdrop.remove();
  actions.append(close, retry);
  box.append(title, list, actions);
  backdrop.append(box);
  document.body.append(backdrop);
}

// The newest progress window. Close only hides it, so the video view's "Render status" button can bring it back.
// A window is removed for good when its job finishes (close(delay)) or when a newer one replaces a hidden one.
let lastProgressWindow = null;

export function showLastProgressWindow() {
  if (!lastProgressWindow?.isAlive()) return false;
  lastProgressWindow.show();
  return true;
}

export function createBaseProgressWindow(title, options = {}) {
  if (lastProgressWindow?.isHidden()) lastProgressWindow.remove();
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
  body.style.cssText = "padding:12px;font-size:12px;line-height:1.45;max-height:min(72vh,760px);overflow:auto;";
  const status = document.createElement("div");
  status.style.cssText = "white-space:pre-wrap;";
  status.textContent = "Starting...";
  const sceneDetails = document.createElement("div");
  sceneDetails.style.cssText = "display:none;margin-top:12px;padding-top:12px;border-top:1px solid #164e63;";
  body.append(status, sceneDetails);
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
  const hideForLater = () => {
    box.style.display = "none";
    restore.style.display = "none";
  };
  close.title = "Hide this window. Use the Render status button in the video view to bring it back.";
  close.onclick = hideForLater;
  const registration = {
    isAlive: () => box.isConnected,
    isHidden: () => box.isConnected && box.style.display === "none" && restore.style.display === "none",
    remove: removeAll,
    show() {
      restore.style.display = "none";
      box.style.display = "block";
    },
  };
  lastProgressWindow = registration;
  return {
    set(message, percent = null) {
      status.textContent = message;
      if (percent !== null) barInner.style.width = `${Math.max(5, Math.min(100, percent))}%`;
      const firstLine = String(message || title).split(/\r?\n/)[0] || title;
      restore.textContent = `${title}: ${firstLine}`;
    },
    setHtml(html, percent = null) {
      status.innerHTML = html;
      if (percent !== null) barInner.style.width = `${Math.max(5, Math.min(100, percent))}%`;
      restore.textContent = title;
    },
    setSceneDetails(details = {}) {
      const images = Array.isArray(details.images) ? details.images : [];
      const prompt = String(details.prompt || "").trim();
      const lyric = String(details.lyric || "").trim();
      const sceneLabel = String(details.sceneLabel || "Current scene").trim();
      const modeLabel = String(details.modeLabel || "").trim();
      sceneDetails.replaceChildren();
      sceneDetails.style.display = "block";
      box.style.width = "min(1100px, calc(100vw - 48px))";

      const summary = document.createElement("div");
      summary.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px;";
      const sceneName = document.createElement("div");
      sceneName.textContent = sceneLabel;
      sceneName.style.cssText = "font-size:13px;font-weight:900;color:#e0f2fe;";
      const mode = document.createElement("div");
      mode.textContent = modeLabel;
      mode.style.cssText = "font-size:11px;color:#67e8f9;text-align:right;";
      summary.append(sceneName, mode);
      sceneDetails.append(summary);

      const sectionTitle = (text) => {
        const label = document.createElement("div");
        label.textContent = text;
        label.style.cssText = "margin:10px 0 6px;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;";
        return label;
      };

      sceneDetails.append(sectionTitle(`${String(details.imagesTitle || "Reference images")} (${images.length})`));
      if (images.length) {
        const strip = document.createElement("div");
        strip.style.cssText = "display:flex;gap:8px;overflow-x:auto;padding:2px 0 8px;";
        images.forEach((item, index) => {
          const card = document.createElement("div");
          card.style.cssText = "flex:0 0 118px;border:1px solid #334155;border-radius:7px;background:#020617;overflow:hidden;";
          const image = document.createElement("img");
          const imagePath = String(item?.path || "").trim();
          const imageData = String(item?.data || "").trim();
          image.src = String(item?.url || "").trim() || (imagePath ? makeEditorImageUrl(imagePath) : imageData);
          image.alt = String(item?.label || `Image ${index + 1}`);
          image.title = [item?.label, imagePath].filter(Boolean).join("\n");
          image.style.cssText = "display:block;width:118px;height:86px;object-fit:cover;background:#0f172a;";
          const caption = document.createElement("div");
          caption.textContent = String(item?.caption || "").trim() || `Image ${index + 1}: ${String(item?.label || "Reference")}`;
          caption.title = image.title;
          caption.style.cssText = "padding:6px;font-size:10px;color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
          card.append(image, caption);
          strip.append(card);
        });
        sceneDetails.append(strip);
      } else {
        const empty = document.createElement("div");
        empty.textContent = "No image references are being sent for this scene.";
        empty.style.cssText = "padding:8px;border:1px dashed #334155;border-radius:6px;color:#94a3b8;background:#020617;";
        sceneDetails.append(empty);
      }

      sceneDetails.append(sectionTitle("Lyric / dialogue sent"));
      const lyricBox = document.createElement("div");
      lyricBox.textContent = lyric || "No lyric or dialogue line is assigned to this scene.";
      lyricBox.style.cssText = "padding:9px 10px;border:1px solid #334155;border-radius:6px;background:#020617;color:#f8fafc;white-space:pre-wrap;";
      sceneDetails.append(lyricBox);

      const promptDetails = document.createElement("details");
      promptDetails.open = true;
      promptDetails.style.cssText = "margin-top:10px;border:1px solid #334155;border-radius:6px;background:#020617;overflow:hidden;";
      const promptSummary = document.createElement("summary");
      promptSummary.textContent = "Exact prompt sent to MiniMax H3";
      promptSummary.style.cssText = "padding:9px 10px;cursor:pointer;font-weight:900;color:#67e8f9;background:#0b1220;";
      const promptText = document.createElement("pre");
      promptText.textContent = prompt || "No prompt is available.";
      promptText.style.cssText = "margin:0;padding:10px;max-height:260px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;color:#e2e8f0;font:11px/1.45 monospace;";
      promptDetails.append(promptSummary, promptText);
      sceneDetails.append(promptDetails);
    },
    close(delay = 0) {
      setTimeout(() => {
        removeAll();
        if (lastProgressWindow === registration) lastProgressWindow = null;
      }, delay);
    },
  };
}

export function showFinalVideoReadyModal(videoPath) {
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

export function showModelDownloadModal() {
  const tabs = [
    {
      id: "ltx",
      label: "LTX",
      subTabs: [
        {
          id: "ltx-25",
          label: "LTX 2.5",
          groups: [
            { title: "LTX 2.5", note: "LTX 2.5 distilled transformer, Gemma 4 text encoders, video/audio VAEs, and x2 latent upscaler.", downloads: LTX_25_MODEL_DOWNLOADS },
          ],
        },
        {
          id: "ltx-23",
          label: "LTX 2.3",
          groups: [
            { title: "LTX 2.3", note: "Legacy LTX 2.3 model files for existing workflows.", downloads: LTX_23_MODEL_DOWNLOADS },
          ],
        },
      ],
    },
    {
      id: "image-models",
      label: "Image Models",
      groups: [
        { title: "ZImage", note: "Core ZImage Turbo diffusion model, Qwen text encoder, and VAE.", downloads: ZIMAGE_MODEL_DOWNLOADS },
        { title: "Krea2", note: "Krea2 text-to-image model used by the Reference Builder Krea2 + ZImage enhancer option.", downloads: KREA2_MODEL_DOWNLOADS },
        { title: "Flux/Klein 9B", note: "9B is higher quality. 4B is smaller and lighter.", downloads: FLUX_KLEIN_9B_MODEL_DOWNLOADS },
        { title: "Flux/Klein 4B", note: "4B is smaller and lighter.", downloads: FLUX_KLEIN_4B_MODEL_DOWNLOADS },
        { title: "Ernie Image", note: "Ernie diffusion model, Ministral text encoder, and VAE.", downloads: ERNIE_MODEL_DOWNLOADS },
      ],
    },
    {
      id: "llm",
      label: "LLM Models",
      subTabs: [
        {
          id: "gemma",
          label: "Gemma",
          groups: [
            { title: "LLM / Gemma", note: "SuperGemma is used for text prompting. Gemma Vision GGUF plus its matching mmproj are used for image-reference prompting.", downloads: GEMMA_LLM_MODEL_DOWNLOADS },
          ],
        },
        {
          id: "qwen",
          label: "Qwen",
          groups: [
            { title: "LLM / Qwen", note: "Qwen3.8 requires both a chosen GGUF model (including all shards for that quantization) and its matching vision mmproj. Rename the projector to qwen-mmproj-BF16.gguf so it is not confused with Gemma's mmproj-BF16.gguf.", downloads: QWEN_LLM_MODEL_DOWNLOADS },
          ],
        },
      ],
    },
    {
      id: "minimax",
      label: "MiniMax H3",
      groups: [
        { title: "MiniMax H3", note: "Required diffusion model, Qwen3-VL text encoder, video VAE, and audio VAE for MiniMax H3 rendering.", downloads: MINIMAX_H3_MODEL_DOWNLOADS },
      ],
    },
    { id: "custom-nodes", label: "Custom Nodes", custom: true },
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
  body.id = "vrgdg-model-downloads-panel";
  body.setAttribute("role", "tabpanel");
  body.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;padding:18px 20px 20px;";
  const tabBar = document.createElement("div");
  tabBar.setAttribute("role", "tablist");
  tabBar.setAttribute("aria-label", "Model download categories");
  tabBar.style.cssText = "position:sticky;top:78px;z-index:1;display:flex;flex-wrap:wrap;gap:10px;padding:14px 20px;border-bottom:1px solid #334155;background:#0b1220;";
  const tabButtons = new Map();
  const activeSubTabs = new Map();

  const renderGroups = (groups) => {
    body.replaceChildren();
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
  };

  const renderCustomNodes = async () => {
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.gap = "14px";
    body.replaceChildren();
    const intro = document.createElement("div");
    intro.textContent = "These are the external custom-node packs used by the Video Builder workflows. Open a repository for details, or install only the missing packs with the button below.";
    intro.style.cssText = "font-size:14px;line-height:1.45;color:#cbd5e1;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;";
    const installAll = makeMiniButton("Install Missing Nodes");
    installAll.style.cssText += ";font-size:15px;font-weight:900;padding:11px 15px;background:#15803d;border-color:#4ade80;color:#f0fdf4;";
    const refresh = makeMiniButton("Refresh Status");
    refresh.style.cssText += ";font-size:14px;font-weight:800;padding:11px 15px;";
    actions.append(installAll, refresh);
    const restartNote = document.createElement("div");
    restartNote.textContent = "Installs run through ComfyUI-Manager (cm_cli): packs in the Comfy Registry install by registry id, the rest from GitHub. Fully close and restart ComfyUI, then refresh your browser, before using newly installed nodes.";
    restartNote.style.cssText = "padding:10px 12px;border:1px solid #854d0e;border-radius:7px;background:#422006;color:#fde68a;font-size:12px;line-height:1.4;";
    const managerMissing = document.createElement("div");
    managerMissing.textContent = "ComfyUI-Manager is not installed in ComfyUI’s Python, so nodes cannot be installed from here. Start ComfyUI with --enable-manager, or run pip install -r manager_requirements.txt from the ComfyUI folder, then restart ComfyUI.";
    managerMissing.style.cssText = "display:none;padding:10px 12px;border:1px solid #b91c1c;border-radius:7px;background:#450a0a;color:#fecaca;font-size:12px;line-height:1.4;";
    const list = document.createElement("div");
    list.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;";
    body.append(intro, actions, restartNote, managerMissing, list);
    let managerAvailable = true;

    const draw = (statuses = []) => {
      managerMissing.style.display = managerAvailable ? "none" : "";
      installAll.disabled = !managerAvailable;
      const byId = new Map(statuses.map((item) => [item.id, item]));
      list.replaceChildren();
      for (const item of VIDEO_BUILDER_CUSTOM_NODES) {
        const status = byId.get(item.id);
        const card = document.createElement("div");
        card.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:9px;background:#111827;padding:15px;";
        const heading = document.createElement("div");
        heading.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
        const name = document.createElement("div");
        name.textContent = item.label;
        name.style.cssText = "font-size:17px;font-weight:900;color:#f8fafc;";
        const badge = document.createElement("span");
        const installed = item.current || status?.installed;
        badge.textContent = installed ? "Installed" : "Missing";
        badge.style.cssText = `font-size:11px;font-weight:900;padding:4px 7px;border-radius:999px;background:${installed ? "#14532d" : "#7f1d1d"};color:${installed ? "#bbf7d0" : "#fecaca"};`;
        heading.append(name, badge);
        const note = document.createElement("div");
        note.textContent = item.note;
        note.style.cssText = "font-size:13px;line-height:1.4;color:#cbd5e1;min-height:36px;";
        const source = document.createElement("div");
        source.textContent = status?.source === "registry"
          ? `Source: Comfy Registry (${status.install})`
          : status?.source === "github" ? "Source: GitHub (not in the Comfy Registry)" : "";
        source.style.cssText = "font-size:12px;color:#94a3b8;";
        const buttons = document.createElement("div");
        buttons.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-top:auto;";
        const repo = makeMiniButton("Open Repository");
        repo.addEventListener("click", () => window.open(item.url, "_blank", "noopener,noreferrer"));
        const install = makeMiniButton(item.current ? "Included" : (status?.installed ? "Reinstall Requirements" : "Install"));
        install.disabled = Boolean(item.current) || !managerAvailable;
        install.style.background = installed ? "#334155" : "#1d4ed8";
        install.style.borderColor = installed ? "#64748b" : "#60a5fa";
        if (item.current) install.title = "This is the currently installed Video Builder pack.";
        install.addEventListener("click", async () => {
          install.disabled = true;
          install.textContent = "Installing...";
          try {
            const result = await postJson("/vrgdg/video_builder/custom_nodes/install", { ids: [item.id] }, 1200000);
            toast(`${item.label} installed. Restart ComfyUI before using it.`);
            const refreshed = await getJson("/vrgdg/video_builder/custom_nodes/status");
            managerAvailable = refreshed.manager_available !== false;
            draw(refreshed.nodes || []);
          } catch (error) {
            toast(String(error?.message || error), true);
          } finally {
            install.disabled = !managerAvailable;
          install.textContent = status?.installed ? "Reinstall Requirements" : "Install";
          }
        });
        buttons.append(repo, install);
        card.append(heading, note, source, buttons);
        list.append(card);
      }
    };
    const load = async () => {
      refresh.disabled = true;
      try {
        const result = await getJson("/vrgdg/video_builder/custom_nodes/status");
        if (result.custom_nodes_dir) {
          intro.textContent = `Checking ComfyUI custom nodes folder: ${result.custom_nodes_dir}`;
        }
        managerAvailable = result.manager_available !== false;
        draw(result.nodes || []);
      } catch (error) {
        draw([]);
        toast(`Could not read custom-node status: ${String(error?.message || error)}`, true);
      } finally {
        refresh.disabled = false;
      }
    };
    refresh.onclick = load;
    installAll.onclick = async () => {
      installAll.disabled = true;
      installAll.textContent = "Installing Missing...";
      try {
        const current = await getJson("/vrgdg/video_builder/custom_nodes/status");
        const missing = (current.nodes || []).filter((item) => !item.installed).map((item) => item.id);
        if (!missing.length) {
          toast("All Video Builder custom nodes are already installed.");
        } else {
          await postJson("/vrgdg/video_builder/custom_nodes/install", { ids: missing }, 3600000);
          toast("Missing custom nodes installed. Restart ComfyUI before using them.");
        }
        await load();
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        installAll.disabled = !managerAvailable;
        installAll.textContent = "Install Missing Nodes";
      }
    };
    await load();
  };

  const renderTab = (tab) => {
    if (tab.custom) {
      renderCustomNodes();
      return;
    }
    body.style.display = "grid";
    body.style.flexDirection = "";
    if (!tab.subTabs?.length) {
      renderGroups(tab.groups || []);
      return;
    }

    const selectedSubTab = tab.subTabs.find((subTab) => subTab.id === activeSubTabs.get(tab.id)) || tab.subTabs[0];
    activeSubTabs.set(tab.id, selectedSubTab.id);
    const subTabBar = document.createElement("div");
    subTabBar.setAttribute("role", "tablist");
    subTabBar.setAttribute("aria-label", `${tab.label} versions`);
    subTabBar.style.cssText = "grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;padding:2px 0 4px;";

    renderGroups(selectedSubTab.groups || []);
    for (const subTab of tab.subTabs) {
      const button = makeMiniButton(subTab.label);
      const active = subTab.id === selectedSubTab.id;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
      button.style.cssText += `;font-size:14px;font-weight:900;padding:9px 15px;border-radius:999px;background:${active ? "#7c3aed" : "#1e293b"};border-color:${active ? "#c4b5fd" : "#475569"};color:${active ? "#f5f3ff" : "#cbd5e1"};`;
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        activeSubTabs.set(tab.id, subTab.id);
        renderTab(tab);
      });
      subTabBar.append(button);
    }
    body.prepend(subTabBar);
  };

  const activateTab = (tabId) => {
    const selected = tabs.find((tab) => tab.id === tabId) || tabs[0];
    for (const [id, button] of tabButtons) {
      const active = id === selected.id;
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
      button.style.background = active ? "#0e7490" : "#1e293b";
      button.style.borderColor = active ? "#22d3ee" : "#475569";
      button.style.color = active ? "#ecfeff" : "#cbd5e1";
    }
    renderTab(selected);
  };

  for (const tab of tabs) {
    const button = makeMiniButton(tab.label);
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", body.id);
    button.style.cssText += ";font-size:15px;font-weight:900;padding:10px 16px;border-radius:7px;";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      activateTab(tab.id);
    });
    tabButtons.set(tab.id, button);
    tabBar.append(button);
  }

  box.append(header, tabBar, body);
  backdrop.append(box);
  document.body.append(backdrop);
  activateTab("ltx");
  close.onclick = () => backdrop.remove();
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) backdrop.remove();
  });
}

export function showTextInputModal({ title, label, value = "", placeholder = "", confirmLabel = "Continue" } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
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

export function showLargeTextModal({ title, label, value = "", placeholder = "", confirmLabel = "Continue" } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(720px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = title || "Paste Text";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const fieldLabel = document.createElement("label");
    fieldLabel.textContent = label || "Text";
    fieldLabel.style.cssText = "font-size:12px;font-weight:900;color:#d4d4d8;";
    const input = document.createElement("textarea");
    input.value = value || "";
    input.placeholder = placeholder || "";
    input.style.cssText = "width:100%;box-sizing:border-box;min-height:260px;resize:vertical;border:1px solid #374151;border-radius:7px;background:#0f172a;color:#e5e7eb;padding:10px;font-size:12px;line-height:1.4;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;";
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
      if (event.key === "Escape") finish(null);
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") finish(input.value.trim());
    });
    actions.append(cancel, confirm);
    box.append(heading, fieldLabel, input, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    input.focus();
  });
}

export function showInfoModal({ title, lines = [], confirmLabel = "Got it" } = {}) {
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

export function showAddSegmentPositionModal(sceneLabel = "selected scene") {
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

export function showLongSegmentConfirm(duration) {
  return new Promise((resolve) => {
    const seconds = Math.max(0, Number(duration || 0));
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(500px,calc(100vw - 40px));border:1px solid #92400e;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Insert Long Segment?";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#fde68a;";
    const body = document.createElement("div");
    body.textContent = `Are you sure you want to insert this segment? It will be ${seconds.toFixed(2)} seconds long.`;
    body.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton("Insert Segment", "primary");
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    cancel.onclick = () => finish(false);
    confirm.onclick = () => finish(true);
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(false);
    });
    actions.append(cancel, confirm);
    box.append(heading, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    cancel.focus();
  });
}

export function pickProjectSessionFile() {
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

export function showSaveProjectAsModal(defaultName = "") {
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

export function showWelcomeProjectModal(projects = [], projectsLoader = null) {
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
    let finished = false;

    const finish = (result) => {
      finished = true;
      backdrop.remove();
      resolve(result);
    };
    create.onclick = () => finish({ action: "new" });
    close.onclick = () => finish(null);
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(null);
    });

    const showEmpty = (text = "No existing projects were found in the ComfyUI output folder.") => {
      list.replaceChildren();
      const empty = document.createElement("div");
      empty.textContent = text;
      empty.style.cssText = "border:1px dashed #3f3f46;border-radius:7px;padding:14px;color:#a1a1aa;font-size:12px;text-align:center;";
      list.append(empty);
    };

    const renderProjects = (items = []) => {
      list.replaceChildren();
      if (!items.length) {
        showEmpty();
        return;
      }
      for (const project of items) {
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
        del.disabled = project.can_delete === false;
        if (del.disabled) del.title = "For safety, projects outside the ComfyUI output folder must be deleted manually.";
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
    };

    box.append(heading, note, actions, listTitle, list);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.tabIndex = -1;
    backdrop.focus();

    if (projectsLoader && typeof projectsLoader.then === "function") {
      showEmpty("Loading existing projects...");
      projectsLoader
        .then((loadedProjects) => {
          if (finished) return;
          renderProjects(Array.isArray(loadedProjects) ? loadedProjects : []);
        })
        .catch((error) => {
          console.warn("[VRGDG Music Builder] Could not list existing projects:", error);
          if (!finished) showEmpty("Could not load existing projects. You can still create a new project.");
        });
    } else {
      renderProjects(projects);
    }
  });
}

export function showLoadProjectModal(projects = []) {
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
        del.disabled = project.can_delete === false;
        if (del.disabled) del.title = "For safety, projects outside the ComfyUI output folder must be deleted manually.";
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
