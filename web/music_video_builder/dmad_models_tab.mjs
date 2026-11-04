import { getJson, postJson } from "./comfy_api.mjs";
import { makeMiniButton, toast } from "./controls.mjs";
import { DMAD_MODEL_FILES } from "./models.mjs";

// The Models dialog's "Dmad Models" tab. It shows the two DMAD LoRAs, installs them into ComfyUI's models/loras folder
// through the Builder's own routes (core/model_downloads.py) and follows the download until it is done.

export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(0)} MB`;
  return `${Math.max(0, Math.round(value / 1024))} KB`;
}

// One line for a model's state: Installed, Not installed, a download in progress, or why it failed.
export function dmadModelStateText(model) {
  const job = model?.job || {};
  if (job.state === "downloading") {
    const total = Number(job.total) || 0;
    const done = Number(job.downloaded) || 0;
    if (!total) return done ? `Downloading… ${formatBytes(done)}` : "Starting…";
    return `Downloading… ${Math.min(100, Math.floor((done / total) * 100))}% (${formatBytes(done)} of ${formatBytes(total)})`;
  }
  if (model?.installed) return `Installed (${formatBytes(model.size)})`;
  if (job.state === "error") return `Failed: ${job.error || "unknown error"}`;
  if (job.state === "cancelled") return "Cancelled";
  return "Not installed";
}

// Every file shown, from the server's list. The static list keeps the cards (and their links) when the server cannot be reached.
export function mergeDmadModels(serverModels = []) {
  const byId = new Map((Array.isArray(serverModels) ? serverModels : []).map((item) => [item.id, item]));
  return DMAD_MODEL_FILES.map((file) => ({ ...file, installed: false, size: 0, destination: "", job: {}, ...(byId.get(file.id) || {}) }));
}

const isBusy = (model) => model?.job?.state === "downloading";

// Fills `body` with the tab and returns a function that stops its polling (call it when the tab is left or the dialog closes).
export function renderDmadModelsTab(body) {
  body.style.display = "flex";
  body.style.flexDirection = "column";
  body.style.gap = "14px";
  body.replaceChildren();

  const intro = document.createElement("div");
  intro.style.cssText = "font-size:14px;line-height:1.45;color:#cbd5e1;";
  intro.textContent = "DMAD sampling needs two LoRA files: the DMAD 4-step LoRA for pass 1 and the refiner LoRA for pass 2. Install puts them straight into your ComfyUI models/loras folder.";
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;";
  const installAll = makeMiniButton("Install Missing Files");
  installAll.style.cssText += ";font-size:15px;font-weight:900;padding:11px 15px;background:#15803d;border-color:#4ade80;color:#f0fdf4;";
  const refresh = makeMiniButton("Refresh Status");
  refresh.style.cssText += ";font-size:14px;font-weight:800;padding:11px 15px;";
  actions.append(installAll, refresh);
  const restartNote = document.createElement("div");
  restartNote.textContent = "When a file finishes, it shows up in the LoRA lists. If a list still misses it, restart ComfyUI or refresh your browser.";
  restartNote.style.cssText = "padding:10px 12px;border:1px solid #854d0e;border-radius:7px;background:#422006;color:#fde68a;font-size:12px;line-height:1.4;";
  const problem = document.createElement("div");
  problem.style.cssText = "display:none;padding:10px 12px;border:1px solid #b91c1c;border-radius:7px;background:#450a0a;color:#fecaca;font-size:12px;line-height:1.4;";
  const list = document.createElement("div");
  list.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;";
  body.append(intro, actions, restartNote, problem, list);

  let models = mergeDmadModels();
  let timer = 0;
  let stopped = false;
  const announced = new Set();

  const stop = () => {
    stopped = true;
    if (timer) window.clearTimeout(timer);
    timer = 0;
  };

  const callServer = async (send) => {
    try {
      const result = await send();
      problem.style.display = "none";
      return result;
    } catch (error) {
      problem.textContent = `Could not reach the Builder's model installer: ${String(error?.message || error)}`;
      problem.style.display = "";
      return null;
    }
  };

  const draw = () => {
    const anyMissing = models.some((model) => !model.installed && !isBusy(model));
    installAll.disabled = !anyMissing;
    list.replaceChildren();
    for (const model of models) {
      const busy = isBusy(model);
      const card = document.createElement("div");
      card.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:9px;background:#111827;padding:15px;";
      const heading = document.createElement("div");
      heading.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
      const name = document.createElement("div");
      name.textContent = model.label;
      name.style.cssText = "font-size:17px;font-weight:900;color:#f8fafc;";
      const badge = document.createElement("span");
      badge.textContent = busy ? "Downloading" : model.installed ? "Installed" : "Missing";
      const badgeColors = busy ? ["#1e3a8a", "#bfdbfe"] : model.installed ? ["#14532d", "#bbf7d0"] : ["#7f1d1d", "#fecaca"];
      badge.style.cssText = `font-size:11px;font-weight:900;padding:4px 7px;border-radius:999px;background:${badgeColors[0]};color:${badgeColors[1]};`;
      heading.append(name, badge);
      const file = document.createElement("div");
      file.textContent = model.filename;
      file.style.cssText = "font-size:12px;color:#94a3b8;overflow-wrap:anywhere;";
      const note = document.createElement("div");
      note.textContent = model.note;
      note.style.cssText = "font-size:13px;line-height:1.4;color:#cbd5e1;";
      const state = document.createElement("div");
      state.textContent = dmadModelStateText(model);
      state.style.cssText = `font-size:12px;font-weight:700;color:${model.job?.state === "error" ? "#fca5a5" : "#e2e8f0"};`;
      const where = document.createElement("div");
      where.textContent = model.installed && model.path ? `In: ${model.path}` : model.destination ? `Installs to: ${model.destination}` : "";
      where.style.cssText = "font-size:12px;color:#94a3b8;overflow-wrap:anywhere;";
      card.append(heading, file, note, state);
      if (busy) {
        const track = document.createElement("div");
        track.style.cssText = "height:8px;border-radius:999px;background:#1e293b;overflow:hidden;";
        const total = Number(model.job.total) || 0;
        const percent = total ? Math.min(100, (Number(model.job.downloaded) / total) * 100) : 4;
        const fill = document.createElement("div");
        fill.style.cssText = `height:100%;width:${percent}%;background:#38bdf8;`;
        track.append(fill);
        card.append(track);
      }
      if (where.textContent) card.append(where);
      const buttons = document.createElement("div");
      buttons.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-top:auto;";
      const install = makeMiniButton(model.installed ? "Installed" : busy ? "Cancel" : model.job?.state === "error" ? "Try Again" : "Install");
      install.disabled = Boolean(model.installed);
      install.style.background = model.installed ? "#334155" : busy ? "#7f1d1d" : "#1d4ed8";
      install.style.borderColor = model.installed ? "#64748b" : busy ? "#f87171" : "#60a5fa";
      install.addEventListener("click", () => (busy ? cancel([model.id]) : start([model.id])));
      const page = makeMiniButton("Open Model Page");
      page.addEventListener("click", () => window.open(model.page, "_blank", "noopener,noreferrer"));
      buttons.append(install, page);
      card.append(buttons);
      list.append(card);
    }
  };

  // A model that was downloading and is now installed: tell the person, and let the panels refresh their LoRA lists.
  const announceFinished = (before, after) => {
    const finished = after.filter((model) => model.installed && before.get(model.id)?.job?.state === "downloading" && !announced.has(model.id));
    for (const model of finished) {
      announced.add(model.id);
      toast(`${model.label} installed in ${model.destination || "models/loras"}.`);
    }
    if (finished.length) window.dispatchEvent(new CustomEvent("vrgdg-models-changed", { detail: { ids: finished.map((model) => model.id) } }));
    for (const model of after) {
      const was = before.get(model.id)?.job?.state;
      if (was === "downloading" && model.job?.state === "error") toast(`${model.label}: ${model.job.error}`, true);
    }
  };

  const load = async () => {
    const result = await callServer(() => getJson("/vrgdg/video_builder/models/status"));
    if (stopped) return;
    if (result) {
      const before = new Map(models.map((model) => [model.id, model]));
      models = mergeDmadModels(result.models);
      announceFinished(before, models);
    }
    draw();
    // Follow a running download once a second. Stop when nothing runs, or when the tab was left.
    if (!stopped && result && models.some(isBusy)) timer = window.setTimeout(load, 1000);
  };

  async function start(ids) {
    const result = await callServer(() => postJson("/vrgdg/video_builder/models/install", { ids }, 60000));
    if (result) {
      models = mergeDmadModels(result.models);
      for (const id of ids) announced.delete(id);
      draw();
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(load, 500);
    }
  }

  async function cancel(ids) {
    await callServer(() => postJson("/vrgdg/video_builder/models/cancel", { ids }, 30000));
    if (timer) window.clearTimeout(timer);
    timer = window.setTimeout(load, 300);
  }

  installAll.onclick = () => start(models.filter((model) => !model.installed && !isBusy(model)).map((model) => model.id));
  refresh.onclick = () => {
    if (timer) window.clearTimeout(timer);
    load();
  };
  draw();
  load();
  return stop;
}
