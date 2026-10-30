// The screen that lists what a folder holds and lets the person choose what to import (Post Process, Overlays, Import folder). The app reads the
// folder first: a folder of effect packs gives the overlay packs it can use, a plain folder gives its video files. Nothing is copied until the
// person chooses and presses Import. What was imported before is not listed again. Each entry is a tile (the pack's picture and its name, four across)
// that plays a live preview while the pointer is over it. The rules at the top are plain functions so they can be tested without a page.

import { api } from "../../../scripts/api.js";
import { postJson } from "./comfy_api.mjs";
import { makeButton, toast } from "./controls.mjs";

const PACK_IMPORT_TIMEOUT_MS = 15 * 60 * 1000;
export const PACK_TILE_COLUMNS = 4;
// The pointer has to rest on a tile this long before its preview is asked for, so moving across the grid does not start a preview for every tile.
export const PACK_PREVIEW_DELAY_MS = 120;

// "12.3 MB", "850 KB", "1.4 GB".
export function packImportBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

// What the scan listed, as rows: packs, or files when the folder has no packs.
export function packImportEntries(scan) {
  const list = scan?.mode === "files" ? scan.files : scan?.packs;
  return (Array.isArray(list) ? list : []).map((entry) => ({
    id: String(entry.id),
    name: String(entry.name || entry.id),
    folder: String(entry.folder || ""),
    size: Number(entry.size_bytes) || 0,
    image: entry.image_url ? String(entry.image_url) : "",
    preview: entry.preview_url ? String(entry.preview_url) : "",
  }));
}

// The entries whose name or folder holds every word of the search, whatever the case.
export function packImportMatches(entries, query) {
  const words = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return entries;
  return entries.filter((entry) => {
    const text = `${entry.name} ${entry.folder}`.toLowerCase();
    return words.every((word) => text.includes(word));
  });
}

// Choose or clear every entry that is shown. What is chosen but hidden by the search stays chosen.
export function packImportSelectShown(selected, shown, on) {
  const next = new Set(selected);
  for (const entry of shown) {
    if (on) next.add(entry.id);
    else next.delete(entry.id);
  }
  return next;
}

// "3 chosen, 54.0 MB" for the button line.
export function packImportChosenText(entries, selected) {
  const chosen = entries.filter((entry) => selected.has(entry.id));
  if (!chosen.length) return "Nothing chosen yet";
  return `${chosen.length} chosen, ${packImportBytes(chosen.reduce((total, entry) => total + entry.size, 0))}`;
}

// The line under the title: what was found, and what is left out and why.
export function packImportSummary(scan, count) {
  const entries = Number.isFinite(count) ? count : packImportEntries(scan).length;
  const hidden = Number(scan?.hidden) || 0;
  const parts = [];
  if (scan?.mode === "files") parts.push(`${entries} video${entries === 1 ? "" : "s"} found`);
  else parts.push(`${entries} overlay${entries === 1 ? "" : "s"} can be imported`);
  if (hidden) parts.push(`${hidden} already imported ${hidden === 1 ? "is" : "are"} not shown`);
  if (Number(scan?.unsupported)) parts.push(`${scan.unsupported} overlay pack${scan.unsupported === 1 ? "" : "s"} in a format this app cannot use yet`);
  if (Number(scan?.not_overlays)) parts.push(`${scan.not_overlays} other pack${scan.not_overlays === 1 ? "" : "s"} (not overlays) left out`);
  if (scan?.truncated) parts.push("only the first 3000 are shown");
  return parts.join(". ") + ".";
}

const PACK_BOX = "box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#fafafa;padding:6px 8px;font-size:12px;";

// Opens the screen for a folder. `onImported(items)` gets the library's clips after every import, so the library redraws behind the screen.
export async function openPackImport({ folder, onImported }) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:24px;";
  const card = document.createElement("div");
  card.style.cssText = "display:flex;flex-direction:column;gap:10px;width:min(1120px,100%);max-height:100%;box-sizing:border-box;padding:16px;border:1px solid #3f3f46;border-radius:10px;background:#18181b;color:#f4f4f5;";
  overlay.append(card);
  const onKey = (event) => {
    if (event.key === "Escape" && !state.busy) close();
  };
  const close = () => {
    state.stopPreview?.();
    window.removeEventListener("keydown", onKey, true);
    overlay.remove();
  };
  overlay.onclick = (event) => {
    if (event.target === overlay && !state.busy) close();
  };
  window.addEventListener("keydown", onKey, true);
  document.body.append(overlay);

  // `stopPreview` ends the preview that is playing, if any. Only one tile plays at a time.
  const state = { scan: null, entries: [], selected: new Set(), query: "", busy: "", message: "", failures: [], stopPreview: null };

  const title = document.createElement("div");
  title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;";
  const where = document.createElement("div");
  where.textContent = folder;
  where.style.cssText = "font-size:10px;color:#71717a;overflow-wrap:anywhere;";
  const summary = document.createElement("div");
  summary.style.cssText = "font-size:11px;line-height:1.45;color:#a1a1aa;";
  const tools = document.createElement("div");
  tools.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;";
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search by name";
  search.style.cssText = `${PACK_BOX}flex:1;min-width:160px;`;
  const all = makeButton("Choose all shown");
  const none = makeButton("Clear all shown");
  tools.append(search, all, none);
  const list = document.createElement("div");
  list.style.cssText = `display:grid;grid-template-columns:repeat(${PACK_TILE_COLUMNS},minmax(0,1fr));gap:8px;align-content:start;overflow:auto;min-height:200px;max-height:60vh;border:1px solid #27272a;border-radius:6px;padding:8px;background:#0b0b0d;`;
  const status = document.createElement("div");
  status.style.cssText = "font-size:11px;line-height:1.45;color:#67e8f9;white-space:pre-wrap;overflow-wrap:anywhere;";
  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap;";
  const chosen = document.createElement("div");
  chosen.style.cssText = "font-size:12px;font-weight:700;color:#e4e4e7;";
  const buttons = document.createElement("div");
  buttons.style.cssText = "display:flex;gap:8px;";
  const importButton = makeButton("Import chosen", "primary");
  const closeButton = makeButton("Close");
  buttons.append(importButton, closeButton);
  footer.append(chosen, buttons);
  card.append(title, where, summary, tools, list, status, footer);

  // One tile: the picture of the pack with a check box on it, and its name. While the pointer is on it, a preview of the pack plays over the picture.
  function tile(entry) {
    const item = document.createElement("label");
    item.dataset.packTile = entry.id;
    const border = (picked) => (picked ? "#22d3ee" : "#27272a");
    item.style.cssText = `display:flex;flex-direction:column;gap:5px;min-width:0;padding:4px;border:2px solid ${border(state.selected.has(entry.id))};border-radius:8px;background:#18181b;cursor:pointer;`;
    const view = document.createElement("div");
    view.style.cssText = "position:relative;aspect-ratio:16/9;border-radius:5px;background:#27272a;overflow:hidden;";
    if (entry.image) {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = api.apiURL(entry.image);
      image.alt = "";
      image.style.cssText = "width:100%;height:100%;object-fit:cover;";
      image.onerror = () => image.remove();
      view.append(image);
    } else if (entry.preview) {
      const idle = document.createElement("div");
      idle.textContent = "Point to preview";
      idle.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:10px;color:#71717a;";
      view.append(idle);
    }
    const hint = document.createElement("div");
    hint.style.cssText = "position:absolute;left:6px;bottom:5px;z-index:1;font-size:10px;color:#e4e4e7;text-shadow:0 1px 2px #000;pointer-events:none;";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.selected.has(entry.id);
    box.disabled = Boolean(state.busy);
    box.style.cssText = "position:absolute;top:6px;left:6px;z-index:2;width:18px;height:18px;margin:0;accent-color:#06b6d4;";
    box.onchange = () => {
      if (box.checked) state.selected.add(entry.id);
      else state.selected.delete(entry.id);
      item.style.borderColor = border(box.checked);
      redrawFooter();
    };
    view.append(hint, box);
    const name = document.createElement("div");
    name.dataset.packName = "";
    name.textContent = entry.name;
    name.title = entry.name;
    name.style.cssText = "padding:0 2px 2px;font-size:12px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    item.append(view, name);

    let timer = 0;
    let player = null;
    let turn = 0;
    const stop = () => {
      clearTimeout(timer);
      turn += 1;
      if (player) {
        player.pause();
        player.removeAttribute("src");
        player.load();
        player.remove();
        player = null;
      }
      hint.textContent = "";
      if (state.stopPreview === stop) state.stopPreview = null;
    };
    item.onmouseenter = () => {
      if (!entry.preview) return;
      state.stopPreview?.();
      state.stopPreview = stop;
      timer = setTimeout(() => {
        const mine = ++turn;
        hint.textContent = "Loading preview...";
        const video = document.createElement("video");
        video.muted = true;
        video.loop = true;
        video.autoplay = true;
        video.playsInline = true;
        video.dataset.packPreview = "";
        video.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .12s;";
        video.onplaying = () => {
          if (mine !== turn) return;
          video.style.opacity = "1";
          hint.textContent = "";
        };
        video.onerror = () => {
          if (mine === turn) hint.textContent = "No preview";
        };
        video.src = api.apiURL(entry.preview);
        player = video;
        view.insertBefore(video, hint);
      }, PACK_PREVIEW_DELAY_MS);
    };
    item.onmouseleave = stop;
    return item;
  }

  function redrawFooter() {
    chosen.textContent = packImportChosenText(state.entries, state.selected);
    importButton.disabled = Boolean(state.busy) || !state.selected.size;
    importButton.textContent = state.selected.size ? `Import ${state.selected.size} chosen` : "Import chosen";
    closeButton.disabled = Boolean(state.busy);
    all.disabled = none.disabled = search.disabled = Boolean(state.busy);
  }

  function redraw() {
    state.stopPreview?.();
    title.textContent = state.scan?.mode === "files" ? "Import videos" : "Import overlays";
    summary.textContent = state.scan ? packImportSummary(state.scan, state.entries.length) : "Reading the folder...";
    status.textContent = state.busy || [state.message, ...state.failures].filter(Boolean).join("\n");
    status.style.color = state.busy || !state.failures.length ? "#67e8f9" : "#fecaca";
    const shown = packImportMatches(state.entries, state.query);
    list.replaceChildren();
    if (!shown.length) {
      const empty = document.createElement("div");
      empty.style.cssText = `grid-column:1 / ${PACK_TILE_COLUMNS + 1};padding:16px;font-size:11px;color:#a1a1aa;text-align:center;`;
      empty.textContent = !state.scan ? "" : state.entries.length ? "Nothing matches the search." : "There is nothing new to import in this folder.";
      list.append(empty);
    }
    for (const entry of shown) list.append(tile(entry));
    redrawFooter();
  }

  search.oninput = () => {
    state.query = search.value;
    redraw();
  };
  all.onclick = () => {
    state.selected = packImportSelectShown(state.selected, packImportMatches(state.entries, state.query), true);
    redraw();
  };
  none.onclick = () => {
    state.selected = packImportSelectShown(state.selected, packImportMatches(state.entries, state.query), false);
    redraw();
  };
  closeButton.onclick = close;

  importButton.onclick = async () => {
    const ids = state.entries.filter((entry) => state.selected.has(entry.id)).map((entry) => entry.id);
    if (!ids.length || state.busy) return;
    let imported = 0;
    state.failures = [];
    state.message = "";
    for (const [index, id] of ids.entries()) {
      const entry = state.entries.find((candidate) => candidate.id === id);
      state.busy = `Importing ${index + 1} of ${ids.length}: ${entry?.name || id}`;
      redraw();
      try {
        const reply = await postJson("/vrgdg/music_builder/overlay_library/packs/import", { path: folder, ids: [id] }, PACK_IMPORT_TIMEOUT_MS);
        for (const failure of reply.failed || []) state.failures.push(`${failure.name || failure.id}: ${failure.error}`);
        const done = (reply.added?.length || 0) + (reply.already_there?.length || 0);
        if (done) {
          imported += done;
          state.entries = state.entries.filter((candidate) => candidate.id !== id);
          state.selected.delete(id);
        }
        if (Array.isArray(reply.items)) onImported?.(reply.items);
      } catch (error) {
        state.failures.push(`${entry?.name || id}: ${String(error?.message || error)}`);
      }
    }
    state.busy = "";
    state.message = imported ? `Imported ${imported}. ${state.entries.length ? "They will not be listed again." : "That was everything new in this folder."}` : "Nothing was imported.";
    if (imported) toast(`Imported ${imported} overlay${imported === 1 ? "" : "s"} into the library.`);
    redraw();
  };

  redraw();
  try {
    state.scan = await postJson("/vrgdg/music_builder/overlay_library/packs/scan", { path: folder }, PACK_IMPORT_TIMEOUT_MS);
    state.entries = packImportEntries(state.scan);
  } catch (error) {
    state.message = String(error?.message || error);
  }
  redraw();
}
