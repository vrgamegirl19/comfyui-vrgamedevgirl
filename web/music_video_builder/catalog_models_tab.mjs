import { getJson, postJson } from "./comfy_api.mjs";
import { makeMiniButton, toast } from "./controls.mjs";
import { dmadModelStateText, formatBytes } from "./dmad_models_tab.mjs";
import { renderHfTokenBox } from "./hf_token_box.mjs";

// The Models dialog's MiniMax H3, Image Models and LLM Models tabs. Every file is a row with a check box. "Install Selected" puts the ticked
// files where ComfyUI looks for them (diffusion_models, text_encoders, vae, loras, latent_upscale_models, models/LLM), one after another, and
// "Install Recommended" does the same for the files the Builder uses by default. The files and their folders come from the Builder's own
// routes (core/model_downloads.py reads core/model_catalog.json), the same ones as the Dmad Models tab.

const CATALOG_STATUS_URL = "/vrgdg/video_builder/models/status";
const CATALOG_INSTALL_URL = "/vrgdg/video_builder/models/install";
const CATALOG_CANCEL_URL = "/vrgdg/video_builder/models/cancel";

export const isDownloading = (model) => model?.job?.state === "downloading";

// A file Hugging Face keeps behind a login (`gated`) can be installed only when the Builder has a Hugging Face token for it (`token_saved`: the one
// saved in the box at the top of the tab, or in H3 Studio's App Settings). Without one the file only opens its page.
export const needsHfToken = (model) => Boolean(model?.gated) && !model?.token_saved;

// A file that can be started now: not installed, not running, and not waiting for a Hugging Face token.
export const canInstallModel = (model) => Boolean(model) && !model.installed && !isDownloading(model) && !needsHfToken(model);

// The files of one card, in the order the server lists them.
export function catalogModelsForGroup(models, category, title) {
  return (Array.isArray(models) ? models : []).filter((model) => model.category === category && (model.groups || []).includes(title));
}

export function modelSizeText(model) {
  const bytes = Number(model?.download_size) || 0;
  return bytes ? formatBytes(bytes) : "size unknown";
}

// How many files and how many bytes starting `models` would download.
export function downloadTotals(models) {
  const startable = models.filter(canInstallModel);
  return { count: startable.length, bytes: startable.reduce((sum, model) => sum + (Number(model.download_size) || 0), 0) };
}

// The files "Install Recommended" starts: the ones the Builder uses by default that are not installed yet, once each.
export function catalogRecommended(models) {
  return models.filter((model) => model.default && canInstallModel(model));
}

// The ticked files that can still be installed, once each (a file shared by two cards is one file).
export function catalogSelected(models, selected) {
  return models.filter((model) => selected.has(model.id) && canInstallModel(model));
}

export const catalogInstallLabel = (name, models) => {
  const { count, bytes } = downloadTotals(models);
  return count ? `${name} (${count} ${count === 1 ? "file" : "files"}, ${formatBytes(bytes)})` : name;
};

// One line under a card's title: how much of it is there and how much is still to download.
export function catalogGroupSummary(models) {
  const installed = models.filter((model) => model.installed).length;
  const { bytes } = downloadTotals(models);
  const left = bytes ? ` · ${formatBytes(bytes)} to download` : "";
  return `${installed} of ${models.length} installed${left}`;
}

// Fills `body` with the tab and returns a function that stops its polling (call it when the tab is left or the dialog closes).
// `groups` are the cards: `{ title, note, hint }` where `title` is the group name in the catalog and `hint` the key of the folder hint. When the
// installer has no files of `category` (an older Builder), `fallback` shows the plain download links instead.
export function renderCatalogTab(body, { category, groups, intro = "", fallback = () => {}, showFolders = () => {} }) {
  body.style.display = "flex";
  body.style.flexDirection = "column";
  body.style.gap = "14px";
  body.replaceChildren();

  const introText = document.createElement("div");
  introText.style.cssText = "font-size:14px;line-height:1.45;color:#cbd5e1;";
  introText.textContent = intro;
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;";
  const installRecommended = makeMiniButton("Install Recommended");
  installRecommended.style.cssText += ";font-size:15px;font-weight:900;padding:11px 15px;background:#15803d;border-color:#4ade80;color:#f0fdf4;";
  const installSelected = makeMiniButton("Install Selected");
  installSelected.style.cssText += ";font-size:15px;font-weight:900;padding:11px 15px;background:#1d4ed8;border-color:#60a5fa;color:#eff6ff;";
  const clearSelection = makeMiniButton("Clear Selection");
  clearSelection.style.cssText += ";font-size:14px;font-weight:800;padding:11px 15px;";
  const cancelAll = makeMiniButton("Cancel Downloads");
  cancelAll.style.cssText += ";font-size:14px;font-weight:800;padding:11px 15px;background:#7f1d1d;border-color:#f87171;color:#fee2e2;";
  const refresh = makeMiniButton("Refresh Status");
  refresh.style.cssText += ";font-size:14px;font-weight:800;padding:11px 15px;";
  actions.append(installRecommended, installSelected, clearSelection, cancelAll, refresh);
  const restartNote = document.createElement("div");
  restartNote.textContent = "Files are saved straight into the right ComfyUI model folder. When one finishes it shows up in the model lists. If a list still misses it, refresh your browser or restart ComfyUI. Large files take a while and are checked when they finish.";
  restartNote.style.cssText = "padding:10px 12px;border:1px solid #854d0e;border-radius:7px;background:#422006;color:#fde68a;font-size:12px;line-height:1.4;";
  const problem = document.createElement("div");
  problem.style.cssText = "display:none;padding:10px 12px;border:1px solid #b91c1c;border-radius:7px;background:#450a0a;color:#fecaca;font-size:12px;line-height:1.4;";
  const list = document.createElement("div");
  list.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px;align-items:start;";
  // The Hugging Face token the gated files wait for. The box shows once the Builder says whether it holds one.
  const tokenBox = renderHfTokenBox({
    onChange: () => {
      if (timer) window.clearTimeout(timer);
      load();
    },
  });
  body.append(introText, tokenBox.element, actions, restartNote, problem, list);

  let models = [];
  let loaded = false;
  let timer = 0;
  let stopped = false;
  const selected = new Set();
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

  const cardsModels = () => groups.flatMap((group) => catalogModelsForGroup(models, category, group.title));
  const unique = (items) => [...new Map(items.map((item) => [item.id, item])).values()];

  const drawActions = () => {
    const all = unique(cardsModels());
    const recommended = catalogRecommended(all);
    installRecommended.style.display = all.some((model) => model.default) ? "" : "none";
    installRecommended.textContent = catalogInstallLabel("Install Recommended", recommended);
    installRecommended.disabled = !recommended.length;
    const chosen = catalogSelected(all, selected);
    installSelected.textContent = catalogInstallLabel("Install Selected", chosen);
    installSelected.disabled = !chosen.length;
    clearSelection.disabled = !selected.size;
    cancelAll.style.display = all.some(isDownloading) ? "" : "none";
  };

  const drawRow = (model) => {
    const busy = isDownloading(model);
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-direction:column;gap:6px;border:1px solid #1f2937;border-radius:8px;background:#0b1220;padding:10px 12px;";
    const top = document.createElement("div");
    top.style.cssText = "display:flex;align-items:flex-start;gap:10px;";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = selected.has(model.id) && canInstallModel(model);
    box.disabled = !canInstallModel(model);
    box.title = model.installed ? "Already installed" : needsHfToken(model) ? "Needs a Hugging Face login: save your token in the box at the top of this tab, or download it from its page" : "Tick to install it with Install Selected";
    box.style.cssText = "width:18px;height:18px;margin-top:2px;flex:none;";
    box.addEventListener("change", () => {
      if (box.checked) selected.add(model.id);
      else selected.delete(model.id);
      draw();
    });
    const text = document.createElement("div");
    text.style.cssText = "flex:1;min-width:0;display:flex;flex-direction:column;gap:3px;";
    const name = document.createElement("div");
    name.textContent = `${model.label}${model.default ? " ★" : ""}`;
    name.title = model.default ? "Recommended: the Builder uses it by default" : "";
    name.style.cssText = "font-size:15px;font-weight:800;color:#f8fafc;";
    const file = document.createElement("div");
    file.textContent = `${model.filename} · ${modelSizeText(model)}`;
    file.style.cssText = "font-size:12px;color:#94a3b8;overflow-wrap:anywhere;";
    text.append(name, file);
    const badge = document.createElement("span");
    badge.textContent = busy ? "Downloading" : model.installed ? "Installed" : needsHfToken(model) ? "Login needed" : "Missing";
    const colors = busy ? ["#1e3a8a", "#bfdbfe"] : model.installed ? ["#14532d", "#bbf7d0"] : needsHfToken(model) ? ["#78350f", "#fde68a"] : ["#7f1d1d", "#fecaca"];
    badge.style.cssText = `flex:none;font-size:11px;font-weight:900;padding:4px 7px;border-radius:999px;background:${colors[0]};color:${colors[1]};`;
    top.append(box, text, badge);
    row.append(top);
    if (model.note) {
      const note = document.createElement("div");
      note.textContent = model.note;
      note.style.cssText = "font-size:12px;line-height:1.4;color:#cbd5e1;margin-left:28px;";
      row.append(note);
    }
    const state = document.createElement("div");
    state.textContent = needsHfToken(model) && !model.installed ? model.gated_message || "Hugging Face asks for a login before it gives this file out." : dmadModelStateText(model);
    state.style.cssText = `font-size:12px;font-weight:700;margin-left:28px;color:${model.job?.state === "error" ? "#fca5a5" : "#e2e8f0"};`;
    row.append(state);
    if (busy) {
      const track = document.createElement("div");
      track.style.cssText = "height:8px;border-radius:999px;background:#1e293b;overflow:hidden;margin-left:28px;";
      const total = Number(model.job.total) || 0;
      const percent = total ? Math.min(100, (Number(model.job.downloaded) / total) * 100) : 4;
      const fill = document.createElement("div");
      fill.style.cssText = `height:100%;width:${percent}%;background:#38bdf8;`;
      track.append(fill);
      row.append(track);
    }
    const where = document.createElement("div");
    where.textContent = model.installed && model.path ? `In: ${model.path}` : model.destination ? `Installs to: ${model.destination}` : "";
    where.style.cssText = "font-size:12px;color:#94a3b8;overflow-wrap:anywhere;margin-left:28px;";
    if (where.textContent) row.append(where);
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-left:28px;";
    if (!needsHfToken(model) || model.installed || busy) {
      const install = makeMiniButton(model.installed ? "Installed" : busy ? "Cancel" : model.job?.state === "error" ? "Try Again" : "Install");
      install.disabled = Boolean(model.installed);
      install.style.background = model.installed ? "#334155" : busy ? "#7f1d1d" : "#1d4ed8";
      install.style.borderColor = model.installed ? "#64748b" : busy ? "#f87171" : "#60a5fa";
      install.addEventListener("click", () => (busy ? cancel([model.id]) : start([model.id])));
      buttons.append(install);
    }
    const page = makeMiniButton("Open Model Page");
    page.addEventListener("click", () => window.open(model.page, "_blank", "noopener,noreferrer"));
    buttons.append(page);
    row.append(buttons);
    return row;
  };

  const drawCard = (group) => {
    const members = catalogModelsForGroup(models, category, group.title);
    const card = document.createElement("div");
    card.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:10px;background:#111827;padding:16px;";
    const titleRow = document.createElement("div");
    titleRow.style.cssText = "display:flex;align-items:center;gap:10px;";
    const selectable = members.filter(canInstallModel);
    const groupBox = document.createElement("input");
    groupBox.type = "checkbox";
    groupBox.checked = selectable.length > 0 && selectable.every((model) => selected.has(model.id));
    groupBox.disabled = !selectable.length;
    groupBox.title = "Tick or untick every file of this card that is not installed yet";
    groupBox.style.cssText = "width:20px;height:20px;flex:none;";
    groupBox.addEventListener("change", () => {
      for (const model of selectable) {
        if (groupBox.checked) selected.add(model.id);
        else selected.delete(model.id);
      }
      draw();
    });
    const title = document.createElement("div");
    title.textContent = group.title;
    title.style.cssText = "flex:1;font-size:20px;font-weight:900;color:#f8fafc;";
    const folders = makeMiniButton("Folders");
    folders.style.fontSize = "13px";
    folders.style.padding = "8px 10px";
    folders.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      showFolders(group);
    });
    titleRow.append(groupBox, title, folders);
    card.append(titleRow);
    if (group.note) {
      const note = document.createElement("div");
      note.textContent = group.note;
      note.style.cssText = "font-size:14px;line-height:1.4;color:#c7d2fe;";
      card.append(note);
    }
    const summary = document.createElement("div");
    summary.textContent = catalogGroupSummary(members);
    summary.style.cssText = "font-size:12px;font-weight:700;color:#94a3b8;";
    card.append(summary);
    for (const model of members) card.append(drawRow(model));
    // A card with recommended files has its own button, so one card (Gemma, say) can be installed without the others.
    const recommended = catalogRecommended(members);
    if (members.some((model) => model.default)) {
      const own = makeMiniButton(catalogInstallLabel("Install Recommended", recommended));
      own.disabled = !recommended.length;
      own.style.cssText += ";background:#15803d;border-color:#4ade80;color:#f0fdf4;font-weight:900;";
      own.addEventListener("click", () => start(recommended.map((model) => model.id)));
      card.append(own);
    }
    return card;
  };

  const draw = () => {
    drawActions();
    list.replaceChildren();
    if (!loaded) return;
    for (const group of groups) list.append(drawCard(group));
  };

  // A file that was downloading and is now installed: tell the person, and let the panels refresh their model lists.
  const announceFinished = (before, after) => {
    const finished = after.filter((model) => model.installed && before.get(model.id)?.job?.state === "downloading" && !announced.has(model.id));
    for (const model of finished) {
      announced.add(model.id);
      selected.delete(model.id);
      toast(`${model.label} installed in ${model.destination || model.folder}.`);
    }
    if (finished.length) window.dispatchEvent(new CustomEvent("vrgdg-models-changed", { detail: { ids: finished.map((model) => model.id) } }));
    for (const model of after) {
      if (before.get(model.id)?.job?.state === "downloading" && model.job?.state === "error") toast(`${model.label}: ${model.job.error}`, true);
    }
  };

  const load = async () => {
    const result = await callServer(() => getJson(CATALOG_STATUS_URL));
    if (stopped) return;
    if (!result && !loaded) {
      // No installer to ask (an older Builder, or ComfyUI is not reachable): the plain download links still work.
      toast(problem.textContent, true);
      stop();
      fallback();
      return;
    }
    if (result) {
      const before = new Map(models.map((model) => [model.id, model]));
      models = Array.isArray(result.models) ? result.models : [];
      tokenBox.setSaved(result.hf_token_saved);
      announceFinished(before, models);
      loaded = true;
      if (!cardsModels().length) {
        // An older Builder: its installer has none of these files. Show the plain links.
        stop();
        fallback();
        return;
      }
    }
    draw();
    // Follow running downloads once a second. Stop when nothing runs, or when the tab was left.
    if (!stopped && result && cardsModels().some(isDownloading)) timer = window.setTimeout(load, 1000);
  };

  async function start(ids) {
    const result = await callServer(() => postJson(CATALOG_INSTALL_URL, { ids }, 60000));
    if (result) {
      models = Array.isArray(result.models) ? result.models : models;
      for (const id of ids) announced.delete(id);
      draw();
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(load, 500);
    }
  }

  async function cancel(ids) {
    await callServer(() => postJson(CATALOG_CANCEL_URL, { ids }, 30000));
    if (timer) window.clearTimeout(timer);
    timer = window.setTimeout(load, 300);
  }

  installRecommended.onclick = () => start(catalogRecommended(unique(cardsModels())).map((model) => model.id));
  installSelected.onclick = () => start(catalogSelected(unique(cardsModels()), selected).map((model) => model.id));
  clearSelection.onclick = () => {
    selected.clear();
    draw();
  };
  cancelAll.onclick = () => cancel(unique(cardsModels()).filter(isDownloading).map((model) => model.id));
  refresh.onclick = () => {
    if (timer) window.clearTimeout(timer);
    load();
  };
  draw();
  load();
  return stop;
}
