// The screen that lists what a folder holds and lets the person choose what to import (Post Process, Overlays, Import folder). The app reads the
// folder first: a folder of effect packs gives the overlay packs it can use, a plain folder gives its video files. Nothing is copied until the
// person chooses and presses Import. What was imported before is not listed again. The rules at the top are plain functions so they can be tested
// without a page.

import { api } from "../../../scripts/api.js";
import { postJson } from "./comfy_api.mjs";
import { makeButton, toast } from "./controls.mjs";

const PACK_IMPORT_TIMEOUT_MS = 15 * 60 * 1000;

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
    note: String(entry.note || ""),
    image: entry.image_url ? String(entry.image_url) : "",
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
  card.style.cssText = "display:flex;flex-direction:column;gap:10px;width:min(860px,100%);max-height:100%;box-sizing:border-box;padding:16px;border:1px solid #3f3f46;border-radius:10px;background:#18181b;color:#f4f4f5;";
  overlay.append(card);
  const onKey = (event) => {
    if (event.key === "Escape" && !state.busy) close();
  };
  const close = () => {
    window.removeEventListener("keydown", onKey, true);
    overlay.remove();
  };
  overlay.onclick = (event) => {
    if (event.target === overlay && !state.busy) close();
  };
  window.addEventListener("keydown", onKey, true);
  document.body.append(overlay);

  const state = { scan: null, entries: [], selected: new Set(), query: "", busy: "", message: "", failures: [] };

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
  list.style.cssText = "display:flex;flex-direction:column;gap:4px;overflow:auto;min-height:160px;max-height:52vh;border:1px solid #27272a;border-radius:6px;padding:4px;background:#0b0b0d;";
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

  function row(entry) {
    const item = document.createElement("label");
    item.style.cssText = "display:flex;align-items:center;gap:10px;padding:5px 6px;border-radius:5px;cursor:pointer;background:#18181b;";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.selected.has(entry.id);
    box.disabled = Boolean(state.busy);
    box.onchange = () => {
      if (box.checked) state.selected.add(entry.id);
      else state.selected.delete(entry.id);
      redrawFooter();
    };
    const picture = document.createElement("div");
    picture.style.cssText = "width:64px;height:36px;flex:none;border-radius:3px;background:#27272a;overflow:hidden;";
    if (entry.image) {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = api.apiURL(entry.image);
      image.alt = "";
      image.style.cssText = "width:100%;height:100%;object-fit:cover;";
      image.onerror = () => image.remove();
      picture.append(image);
    }
    const text = document.createElement("div");
    text.style.cssText = "min-width:0;flex:1;display:flex;flex-direction:column;gap:1px;";
    const name = document.createElement("div");
    name.textContent = entry.name;
    name.style.cssText = "font-size:12px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    const meta = document.createElement("div");
    meta.textContent = [entry.folder, packImportBytes(entry.size), entry.note].filter(Boolean).join(" · ");
    meta.style.cssText = "font-size:10px;color:#a1a1aa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    meta.title = meta.textContent;
    text.append(name, meta);
    item.append(box, picture, text);
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
    title.textContent = state.scan?.mode === "files" ? "Import videos" : "Import overlays";
    summary.textContent = state.scan ? packImportSummary(state.scan, state.entries.length) : "Reading the folder...";
    status.textContent = state.busy || [state.message, ...state.failures].filter(Boolean).join("\n");
    status.style.color = state.busy || !state.failures.length ? "#67e8f9" : "#fecaca";
    const shown = packImportMatches(state.entries, state.query);
    list.replaceChildren();
    if (!shown.length) {
      const empty = document.createElement("div");
      empty.style.cssText = "padding:16px;font-size:11px;color:#a1a1aa;text-align:center;";
      empty.textContent = !state.scan ? "" : state.entries.length ? "Nothing matches the search." : "There is nothing new to import in this folder.";
      list.append(empty);
    }
    for (const entry of shown) list.append(row(entry));
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
