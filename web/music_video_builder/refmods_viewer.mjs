import { api } from "../../../scripts/api.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { confirmDestructiveAction } from "./confirm_dialog.mjs";
import { makeButton, makeInput, makeSelect, toast } from "./controls.mjs";
import { loadRefmodLibrary, refmodLibraryChanged, refmodPreviewUrl } from "./refmod_card.mjs";
import { REFMOD_ACTIVE_TYPES } from "./refmods_studio.mjs";
import {
  filterRefmods, formatBytes, formatCanvas, formatTokens, groupByType, prettyRefmodName, refmodKindLabel, typeLabel, VIEWER_SORTS,
} from "./refmods_viewer_data.mjs";

// RefMods Viewer: browse the saved RefMods by category with their pictures. Read only: RefMods are made in RefMods
// Studio. The left list is the categories with a count, the middle is a grid of cards, and clicking a card shows its
// details on the right. Data comes from GET /vrgdg/refmod/library, pictures from GET /vrgdg/refmod/preview.

const TYPE_LABELS = Object.fromEntries(REFMOD_ACTIVE_TYPES.map((item) => [item.value, item.label]));
const TYPE_ORDER = REFMOD_ACTIVE_TYPES.map((item) => item.value);
const TYPE_HINTS = Object.fromEntries(REFMOD_ACTIVE_TYPES.map((item) => [item.value, item.hint]));

const BACKDROP_STYLE = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:16px;";
const BOX_STYLE = `width:min(1500px,calc(100vw - 32px));height:calc(100vh - 32px);box-sizing:border-box;overflow:hidden;border:1px solid #3f3f46;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;
const PANEL_STYLE = "border:1px solid #27272a;border-radius:8px;background:#18181b;min-height:0;min-width:0;";
const BADGE_STYLE = "display:inline-block;border-radius:999px;padding:2px 8px;font-size:10px;font-weight:800;line-height:1.5;";

const div = (css, text) => {
  const element = document.createElement("div");
  if (css) element.style.cssText = css;
  if (text !== undefined) element.textContent = text;
  return element;
};

const badge = (text, background, color) => div(`${BADGE_STYLE}background:${background};color:${color};`, text);

// A picture that falls back to the first letter of the name when the RefMod has no preview or it fails to load.
export function pictureElement(item, css) {
  const wrap = div(`${css}position:relative;overflow:hidden;background:#09090b;display:flex;align-items:center;justify-content:center;`);
  const placeholder = div("font-size:34px;font-weight:900;color:#3f3f46;", prettyRefmodName(item.name).charAt(0) || "?");
  wrap.append(placeholder);
  if (item.has_preview) {
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = prettyRefmodName(item.name);
    image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;";
    image.onload = () => placeholder.remove();
    image.onerror = () => image.remove();
    image.src = refmodPreviewUrl(item.name);
    wrap.append(image);
  }
  return wrap;
}

export function openRefModsViewer({ onOpenStudio } = {}) {
  document.getElementById("vrgdg-refmods-viewer")?.remove();

  let library = [];
  let loaded = false;
  let activeType = "";
  let selectedName = "";
  const view = { query: "", sort: "name" };

  const backdrop = document.createElement("div");
  backdrop.id = "vrgdg-refmods-viewer";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.style.cssText = BACKDROP_STYLE;
  const box = div(BOX_STYLE);

  const close = () => {
    document.removeEventListener("keydown", onKey, true);
    backdrop.remove();
  };
  const onKey = (event) => {
    // Escape belongs to the delete confirmation while it is open.
    if (event.key === "Escape" && !document.querySelector('[role="alertdialog"]')) {
      event.stopPropagation();
      close();
    }
  };

  // Header: title, search, sort, refresh, close.
  const header = div("display:flex;align-items:center;gap:12px;flex:0 0 auto;flex-wrap:wrap;");
  const titleWrap = div("flex:1 1 220px;min-width:0;");
  const title = div("font-size:18px;font-weight:900;", "RefMods Viewer");
  const subtitle = div("font-size:12px;color:#a1a1aa;margin-top:2px;", "Loading RefMods...");
  titleWrap.append(title, subtitle);
  const search = makeInput("");
  search.placeholder = "Search name, description or tag";
  search.style.maxWidth = "280px";
  const sort = makeSelect(VIEWER_SORTS, "name");
  sort.title = "Sort the cards.";
  sort.style.width = "140px";
  const refresh = makeButton("Refresh");
  refresh.title = "Read the RefMods folders again.";
  const studioButton = makeButton("Open RefMods Studio");
  studioButton.onclick = () => {
    close();
    onOpenStudio?.();
  };
  if (!onOpenStudio) studioButton.style.display = "none";
  const closeButton = makeButton("Close");
  closeButton.onclick = close;
  header.append(titleWrap, search, sort, refresh, studioButton, closeButton);

  // Body: categories | grid | details.
  const body = div("display:grid;grid-template-columns:210px minmax(0,1fr) 340px;gap:12px;flex:1 1 auto;min-height:0;");
  const categories = div(`${PANEL_STYLE}padding:8px;display:flex;flex-direction:column;gap:4px;overflow:auto;`);
  const gridPanel = div(`${PANEL_STYLE}padding:12px;overflow:auto;`);
  const grid = div("display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;align-content:start;");
  gridPanel.append(grid);
  const details = div(`${PANEL_STYLE}padding:12px;overflow:auto;display:flex;flex-direction:column;gap:10px;`);
  body.append(categories, gridPanel, details);

  const emptyMessage = (headline, text, withStudio) => {
    const wrap = div("grid-column:1/-1;display:flex;flex-direction:column;align-items:center;gap:8px;padding:48px 16px;text-align:center;");
    wrap.append(div("font-size:15px;font-weight:800;color:#e4e4e7;", headline), div("font-size:12px;color:#a1a1aa;max-width:420px;line-height:1.5;", text));
    if (withStudio && onOpenStudio) {
      const open = makeButton("Open RefMods Studio", "primary");
      open.onclick = () => {
        close();
        onOpenStudio();
      };
      wrap.append(open);
    }
    return wrap;
  };

  const renderCategories = () => {
    categories.replaceChildren();
    const groups = groupByType(library, TYPE_ORDER);
    const entries = [{ type: "", label: "All RefMods", count: library.length }, ...groups.map((group) => ({ ...group, label: typeLabel(group.type, TYPE_LABELS) }))];
    for (const entry of entries) {
      const active = entry.type === activeType;
      const row = document.createElement("button");
      row.type = "button";
      row.style.cssText = `display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;box-sizing:border-box;text-align:left;border:1px solid ${active ? "#0891b2" : "transparent"};border-radius:6px;background:${active ? "#164e63" : "transparent"};color:${active ? "#ecfeff" : "#d4d4d8"};font-family:${BUILDER_FONT_STACK};font-size:13px;font-weight:${active ? 800 : 600};padding:8px 10px;cursor:pointer;`;
      row.title = entry.type ? TYPE_HINTS[entry.type] || "" : "Every category.";
      row.append(div("min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", entry.label), badge(String(entry.count), active ? "#0e7490" : "#27272a", "#f4f4f5"));
      row.onclick = () => {
        activeType = entry.type;
        render();
      };
      categories.append(row);
    }
  };

  const card = (item) => {
    const selected = item.name === selectedName;
    const element = document.createElement("button");
    element.type = "button";
    element.style.cssText = `display:flex;flex-direction:column;gap:0;padding:0;overflow:hidden;text-align:left;border:2px solid ${selected ? "#22d3ee" : "#27272a"};border-radius:8px;background:#0f172a;color:#f8fafc;font-family:${BUILDER_FONT_STACK};cursor:pointer;`;
    element.title = item.description || item.name;
    element.append(pictureElement(item, "width:100%;aspect-ratio:1/1;"));
    const info = div("padding:8px 10px;display:flex;flex-direction:column;gap:5px;min-width:0;");
    info.append(div("font-size:13px;font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", prettyRefmodName(item.name)));
    const badges = div("display:flex;gap:5px;flex-wrap:wrap;");
    badges.append(
      badge(typeLabel(item.type, TYPE_LABELS), "#134e4a", "#99f6e4"),
      badge(item.kind === "video" ? "Video" : "Picture", item.kind === "video" ? "#4c1d95" : "#1e3a8a", item.kind === "video" ? "#ddd6fe" : "#bfdbfe"),
    );
    info.append(badges);
    element.append(info);
    element.onclick = () => {
      selectedName = item.name;
      render();
    };
    return element;
  };

  const detailRow = (label, value) => {
    const row = div("display:flex;justify-content:space-between;gap:10px;font-size:12px;padding:5px 0;border-bottom:1px solid #27272a;");
    row.append(div("color:#a1a1aa;flex:0 0 auto;", label), div("color:#f4f4f5;font-weight:700;text-align:right;min-width:0;overflow-wrap:anywhere;", value));
    return row;
  };

  const renderDetails = (visible) => {
    details.replaceChildren();
    const item = visible.find((entry) => entry.name === selectedName) || library.find((entry) => entry.name === selectedName);
    if (!item) {
      details.append(
        div("font-size:14px;font-weight:800;color:#e4e4e7;", "Select a RefMod"),
        div("font-size:12px;color:#a1a1aa;line-height:1.5;", "Click a card to see its picture, type, size, description and tags."),
      );
      return;
    }
    details.append(pictureElement(item, "width:100%;aspect-ratio:1/1;border-radius:8px;flex:0 0 auto;"));
    details.append(div("font-size:16px;font-weight:900;overflow-wrap:anywhere;", prettyRefmodName(item.name)));
    details.append(div("font-size:11px;color:#71717a;overflow-wrap:anywhere;margin-top:-6px;", item.name));
    const rows = [
      ["Category", typeLabel(item.type, TYPE_LABELS)],
      ["Kind", refmodKindLabel(item)],
      ["Tokens", formatTokens(item.tokens)],
      ["Canvas", formatCanvas(item.canvas)],
      ["File size", formatBytes(item.size)],
      ["Mode", item.mode],
    ].filter(([, value]) => value);
    for (const [label, value] of rows) details.append(detailRow(label, value));
    if (item.description) {
      details.append(div("font-size:11px;font-weight:800;color:#a1a1aa;margin-top:4px;", "DESCRIPTION"), div("font-size:12px;line-height:1.5;color:#e4e4e7;white-space:pre-wrap;", item.description));
    }
    if (item.tags?.length) {
      const tags = div("display:flex;gap:5px;flex-wrap:wrap;");
      for (const tag of item.tags) tags.append(badge(tag, "#27272a", "#d4d4d8"));
      details.append(div("font-size:11px;font-weight:800;color:#a1a1aa;margin-top:4px;", "TAGS"), tags);
    }
    const copy = makeButton("Copy name");
    copy.title = "Copy this RefMod's name (type/name).";
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(item.name);
        toast(`Copied ${item.name}`);
      } catch (error) {
        toast("Could not copy to the clipboard.", true);
      }
    };
    const remove = makeButton("Delete RefMod");
    remove.title = "Permanently delete this RefMod from models/refmods.";
    remove.style.cssText += "border-color:#7f1d1d;background:#450a0a;color:#fecaca;";
    remove.onclick = () => deleteRefmod(item);
    const actions = div("display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;");
    actions.append(copy, remove);
    details.append(actions);
  };

  const deleteRefmod = async (item) => {
    const { confirmed } = await confirmDestructiveAction({
      title: "Delete this RefMod?",
      message: [
        `This permanently deletes the RefMod "${prettyRefmodName(item.name)}" and its picture from your models/refmods folder.`,
        "This cannot be undone. Scenes and Reference Builder cards that use it will no longer find it.",
      ],
      details: [item.name],
      confirmLabel: "Delete RefMod",
    });
    if (!confirmed) return;
    try {
      const response = await api.fetchApi("/vrgdg/refmod/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: item.name }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result?.ok) throw new Error(result?.error || `Delete failed (${response.status}).`);
    } catch (error) {
      toast(String(error?.message || error), true);
      return;
    }
    refmodLibraryChanged();
    if (selectedName === item.name) selectedName = "";
    toast(`Deleted ${prettyRefmodName(item.name)}`);
    await load(true);
  };

  const render = () => {
    const visible = filterRefmods(library, { type: activeType, query: view.query, sort: view.sort });
    const total = library.length;
    subtitle.textContent = !loaded
      ? "Loading RefMods..."
      : total
        ? `${visible.length} of ${total} RefMod${total === 1 ? "" : "s"}${activeType ? ` in ${typeLabel(activeType, TYPE_LABELS)}` : ""}`
        : "No RefMods yet";
    renderCategories();
    grid.replaceChildren();
    if (!loaded) {
      grid.append(emptyMessage("Loading...", "Reading the RefMods folders.", false));
    } else if (!total) {
      grid.append(emptyMessage("No RefMods yet", "RefMods you make in RefMods Studio are saved under models/refmods and show up here with their pictures.", true));
    } else if (!visible.length) {
      grid.append(emptyMessage("Nothing matches", view.query ? `No RefMod matches "${view.query}"${activeType ? " in this category" : ""}.` : "This category is empty.", false));
    } else {
      for (const item of visible) grid.append(card(item));
    }
    renderDetails(visible);
  };

  const load = async (force) => {
    refresh.disabled = true;
    try {
      library = await loadRefmodLibrary(force);
      loaded = true;
      if (activeType && !library.some((item) => item.type === activeType)) activeType = "";
      if (selectedName && !library.some((item) => item.name === selectedName)) selectedName = "";
    } finally {
      refresh.disabled = false;
    }
    render();
  };

  search.addEventListener("input", () => {
    view.query = search.value;
    render();
  });
  sort.addEventListener("change", () => {
    view.sort = sort.value;
    render();
  });
  refresh.onclick = () => load(true);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) close();
  });
  document.addEventListener("keydown", onKey, true);

  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  render();
  load(true);
  search.focus();
}
