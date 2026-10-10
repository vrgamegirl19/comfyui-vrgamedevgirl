import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeButton, makeInput } from "./controls.mjs";
import { loadRefmodLibrary } from "./refmod_card.mjs";
import { entriesForScope } from "./refmod_quick_add_data.mjs";
import { pictureElement } from "./refmods_viewer.mjs";
import { filterRefmods, groupByType, prettyRefmodName, typeLabel } from "./refmods_viewer_data.mjs";
import { REFMOD_ACTIVE_TYPES } from "./refmods_studio.mjs";

// Quick Add RefMod: pick saved RefMods and add each as a ready card in the Reference Builder, with its name,
// description, type and RefMod already filled in. `scope` is "subject" (everything but backgrounds) or "location"
// (backgrounds). `usedNames` are RefMods a card already uses: they are shown as Added and cannot be picked again.
// `onAdd(entries)` receives the chosen library entries. Click a card to select it, double-click to add just that one.

const TYPE_LABELS = Object.fromEntries(REFMOD_ACTIVE_TYPES.map((item) => [item.value, item.label]));
const TYPE_ORDER = REFMOD_ACTIVE_TYPES.map((item) => item.value);

const div = (css, text) => {
  const element = document.createElement("div");
  if (css) element.style.cssText = css;
  if (text !== undefined) element.textContent = text;
  return element;
};

export function openRefmodQuickAdd({ scope = "subject", usedNames = new Set(), onAdd } = {}) {
  document.getElementById("vrgdg-refmod-quick-add")?.remove();
  const noun = scope === "location" ? "location" : "card";
  let library = [];
  let loaded = false;
  let activeType = "";
  const picked = new Set();
  const view = { query: "" };

  const backdrop = document.createElement("div");
  backdrop.id = "vrgdg-refmod-quick-add";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:16px;";
  const box = div(`width:min(1040px,calc(100vw - 32px));height:min(720px,calc(100vh - 32px));box-sizing:border-box;overflow:hidden;border:1px solid #3f3f46;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`);

  const close = () => {
    document.removeEventListener("keydown", onKey, true);
    backdrop.remove();
  };
  const onKey = (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      close();
    }
  };

  const header = div("display:flex;align-items:center;gap:12px;flex-wrap:wrap;flex:0 0 auto;");
  const titleWrap = div("flex:1 1 240px;min-width:0;");
  titleWrap.append(
    div("font-size:18px;font-weight:900;", "Quick Add RefMod"),
    div("font-size:12px;color:#a1a1aa;margin-top:2px;", scope === "location"
      ? "Pick background RefMods. Each becomes a location card with its name and description filled in."
      : "Pick RefMods. Each becomes a card with its name, description and type filled in."),
  );
  const search = makeInput("");
  search.placeholder = "Search name, description or tag";
  search.style.maxWidth = "260px";
  header.append(titleWrap, search);

  const chips = div("display:flex;gap:6px;flex-wrap:wrap;flex:0 0 auto;");
  const gridPanel = div("border:1px solid #27272a;border-radius:8px;background:#18181b;padding:12px;overflow:auto;flex:1 1 auto;min-height:0;");
  const grid = div("display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;align-content:start;");
  gridPanel.append(grid);

  const footer = div("display:flex;align-items:center;justify-content:space-between;gap:12px;flex:0 0 auto;");
  const status = div("font-size:12px;color:#a1a1aa;");
  const buttons = div("display:flex;gap:8px;");
  const cancel = makeButton("Cancel");
  cancel.onclick = close;
  const add = makeButton("Add", "primary");
  buttons.append(cancel, add);
  footer.append(status, buttons);

  const commit = (names) => {
    const chosen = library.filter((entry) => names.includes(entry.name));
    if (!chosen.length) return;
    close();
    onAdd?.(chosen);
  };
  add.onclick = () => commit([...picked]);

  const render = () => {
    const scoped = entriesForScope(library, scope);
    const groups = groupByType(scoped, TYPE_ORDER);
    if (activeType && !groups.some((group) => group.type === activeType)) activeType = "";
    chips.replaceChildren();
    if (groups.length > 1) {
      for (const entry of [{ type: "", label: "All", count: scoped.length }, ...groups.map((group) => ({ ...group, label: typeLabel(group.type, TYPE_LABELS) }))]) {
        const active = entry.type === activeType;
        const chip = document.createElement("button");
        chip.type = "button";
        chip.textContent = `${entry.label} (${entry.count})`;
        chip.style.cssText = `border:1px solid ${active ? "#0891b2" : "#3f3f46"};border-radius:999px;background:${active ? "#164e63" : "#18181b"};color:${active ? "#ecfeff" : "#d4d4d8"};font-family:${BUILDER_FONT_STACK};font-size:12px;font-weight:700;padding:5px 11px;cursor:pointer;`;
        chip.onclick = () => {
          activeType = entry.type;
          render();
        };
        chips.append(chip);
      }
    }
    const visible = filterRefmods(scoped, { type: activeType, query: view.query, sort: "name" });
    grid.replaceChildren();
    if (!loaded) {
      grid.append(div("grid-column:1/-1;padding:40px;text-align:center;color:#a1a1aa;font-size:13px;", "Loading RefMods..."));
    } else if (!scoped.length) {
      grid.append(div("grid-column:1/-1;padding:40px;text-align:center;color:#a1a1aa;font-size:13px;line-height:1.5;", scope === "location"
        ? "No background RefMods yet. Make one in RefMods Studio."
        : "No RefMods yet. Make one in RefMods Studio."));
    } else if (!visible.length) {
      grid.append(div("grid-column:1/-1;padding:40px;text-align:center;color:#a1a1aa;font-size:13px;", "Nothing matches."));
    }
    for (const entry of visible) {
      const used = usedNames.has(entry.name);
      const selected = picked.has(entry.name);
      const tile = document.createElement("button");
      tile.type = "button";
      tile.disabled = used;
      tile.title = used ? `${prettyRefmodName(entry.name)} is already in this project.` : entry.description || entry.name;
      tile.style.cssText = `position:relative;display:flex;flex-direction:column;padding:0;overflow:hidden;text-align:left;border:2px solid ${selected ? "#22d3ee" : "#27272a"};border-radius:8px;background:#0f172a;color:#f8fafc;font-family:${BUILDER_FONT_STACK};cursor:${used ? "default" : "pointer"};opacity:${used ? 0.45 : 1};`;
      tile.append(pictureElement(entry, "width:100%;aspect-ratio:1/1;"));
      const name = div("padding:7px 9px;font-size:12px;font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", prettyRefmodName(entry.name));
      tile.append(name);
      const mark = div(`position:absolute;top:6px;right:6px;min-width:20px;height:20px;border-radius:999px;display:flex;align-items:center;justify-content:center;padding:0 6px;font-size:11px;font-weight:900;box-sizing:border-box;${used ? "background:#3f3f46;color:#e4e4e7;" : selected ? "background:#22d3ee;color:#082f49;" : "background:rgba(9,9,11,.7);color:transparent;border:1px solid #71717a;"}`, used ? "Added" : selected ? "✓" : "·");
      tile.append(mark);
      if (!used) {
        tile.onclick = () => {
          if (picked.has(entry.name)) picked.delete(entry.name);
          else picked.add(entry.name);
          render();
        };
        tile.ondblclick = () => commit([entry.name]);
      }
      grid.append(tile);
    }
    const count = picked.size;
    add.textContent = count ? `Add ${count} ${noun}${count === 1 ? "" : "s"}` : "Add";
    add.disabled = !count;
    add.style.opacity = count ? "1" : "0.5";
    status.textContent = loaded && scoped.length
      ? `${visible.length} RefMod${visible.length === 1 ? "" : "s"} shown. Click to select, double-click to add one right away.`
      : "";
  };

  search.addEventListener("input", () => {
    view.query = search.value;
    render();
  });
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) close();
  });
  document.addEventListener("keydown", onKey, true);
  box.append(header, chips, gridPanel, footer);
  backdrop.append(box);
  document.body.append(backdrop);
  render();
  loadRefmodLibrary(true).then((entries) => {
    library = entries;
    loaded = true;
    render();
  });
  search.focus();
}
