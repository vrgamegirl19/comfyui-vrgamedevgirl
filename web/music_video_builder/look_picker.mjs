// The dialog that chooses an effect or a filter: a grid of the looks drawn on a real frame of the scene at the playhead, grouped by category and searchable, a live
// preview of the chosen one that loops over the block's length, its strength, the length and the look's own settings.
//   mode "add": put the chosen look on the timeline at `at` seconds (the playhead when not given)
//   mode "all": put it over the whole video
//   mode "edit": change the block `id`

import { freshLookId, lookEntries, lookSelectedRange, lookTimelineEnd, removeLook } from "./look_actions.mjs";
import { loadLookLibrary } from "./look_library.mjs";
import { createLookGl } from "./look_gl.mjs";
import { LOOK_KINDS, LOOK_MAX_SECONDS, LOOK_MIN_SECONDS, normalizeLookEntries, placeNewLook, trimLookEnd } from "./look_model.mjs";
import { transitionShared } from "./transition_actions.mjs";
import { transitionElement, transitionName, transitionSceneName, transitionSeconds } from "./transition_dom.mjs";
import { transitionCover, transitionPlaceholder, transitionSceneFrame } from "./transition_frames.mjs";
import { transitionParamRow } from "./transition_picker.mjs";

const LOOK_PREVIEW_WIDTH = 520;
const LOOK_PREVIEW_MAX_HEIGHT = 340;
const LOOK_TILE_WIDTH = 192;
const LOOK_TILE_HEIGHT = 108;
const LOOK_TILES_PER_FRAME = 4;

let lookPicker = null;
const lookLastSeconds = {};

// The text a look is searched by.
const lookSearchText = (item) => `${item.id} ${transitionName(item.id)} ${item.category || ""} ${(item.tags || []).join(" ")}`.toLowerCase();

export async function openLookPicker(kind, { state, pushHistory, commit, mode = "add", id = "", at = null }) {
  lookPicker?.close();
  const config = LOOK_KINDS[kind];
  const segments = state.segments || [];
  const existing = mode === "edit" ? lookEntries(state, kind).find((item) => item.id === id) : null;
  if (!segments.length || (mode === "edit" && !existing)) return;
  const context = { state, pushHistory, commit };
  const playhead = existing ? existing.start : at ?? (Number(transitionShared.clock()) || 0);
  const scene = segments.find((segment) => playhead >= Number(segment.start) && playhead < Number(segment.end)) || segments[0];
  const sceneIndex = Math.max(0, segments.indexOf(scene));
  const button = (text, primary) => transitionElement("button", `border-radius:5px;padding:7px 14px;cursor:pointer;font-weight:800;border:1px solid ${primary ? config.border : "#52525b"};background:${primary ? config.color : "#27272a"};color:#fff;`, { type: "button", textContent: text });

  const backdrop = transitionElement("div", "position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;", { className: "h3-look-picker" });
  const dialog = transitionElement("div", "width:min(1040px,96vw);height:min(700px,94vh);background:#18181b;color:#f4f4f5;border:1px solid #52525b;border-radius:8px;display:flex;flex-direction:column;box-shadow:0 12px 40px rgba(0,0,0,.6);font-size:13px;");
  backdrop.append(dialog);
  const article = config.singular === "effect" ? "an" : "a";
  const title = mode === "edit" ? `Change the ${config.singular}: ${transitionName(existing.look)}` : mode === "all" ? `Put ${article} ${config.singular} over the whole video` : `Add ${article} ${config.singular}`;
  const header = transitionElement("div", "display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #3f3f46;");
  header.append(transitionElement("div", "font-weight:900;", { textContent: title }));
  const closeButton = transitionElement("button", "background:none;border:1px solid #52525b;color:#e4e4e7;border-radius:4px;padding:2px 9px;cursor:pointer;", { type: "button", textContent: "Close" });
  header.append(closeButton);
  const body = transitionElement("div", "flex:1;min-height:0;display:grid;grid-template-columns:minmax(300px,420px) 1fr;");
  dialog.append(header, body);
  const left = transitionElement("div", "display:flex;flex-direction:column;min-height:0;border-right:1px solid #3f3f46;");
  const right = transitionElement("div", "padding:12px 14px;overflow:auto;display:flex;flex-direction:column;gap:10px;");
  body.append(left, right);
  document.body.append(backdrop);

  let frameId = 0;
  let tileFrameId = 0;
  let closed = false;
  let preview = null;
  let thumbs = null;
  let observer = null;
  const close = () => {
    if (closed) return;
    closed = true;
    cancelAnimationFrame(frameId);
    cancelAnimationFrame(tileFrameId);
    observer?.disconnect();
    preview?.dispose();
    thumbs?.dispose();
    backdrop.remove();
    document.removeEventListener("keydown", onKey, true);
    if (lookPicker?.close === close) lookPicker = null;
  };
  const onKey = (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      close();
    }
  };
  document.addEventListener("keydown", onKey, true);
  backdrop.onpointerdown = (event) => {
    if (event.target === backdrop) close();
  };
  closeButton.onclick = close;
  lookPicker = { close };

  left.append(transitionElement("div", "padding:14px;color:#a1a1aa;", { textContent: `Loading the ${config.plural}...` }));
  let items;
  try {
    items = await loadLookLibrary(kind);
  } catch (error) {
    left.textContent = "";
    left.append(transitionElement("div", "padding:14px;color:#fca5a5;", { textContent: String(error.message || error) }));
    return;
  }
  if (closed) return;
  left.textContent = "";

  lookLastSeconds[kind] ??= config.defaultSeconds;
  let selected = existing && items.some((item) => item.id === existing.look) ? existing.look : "";
  let seconds = existing?.seconds ?? lookLastSeconds[kind];
  let intensity = existing?.intensity ?? 1;
  let overrides = existing ? { ...(existing.params || {}) } : {};
  let search = "";
  let category = "";
  const byId = (name) => items.find((item) => item.id === name) || null;

  // -- left: search, categories and the grid
  const searchBox = transitionElement("input", "margin:10px 10px 6px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:6px 8px;", { type: "search", placeholder: `Search ${config.plural}` });
  const categoryCounts = new Map();
  items.forEach((item) => categoryCounts.set(item.category || "Other", (categoryCounts.get(item.category || "Other") || 0) + 1));
  const categoryBox = transitionElement("select", "margin:0 10px 8px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:5px 6px;");
  categoryBox.append(new Option(`All ${config.plural} (${items.length})`, ""));
  [...categoryCounts.entries()].sort((a, b) => a[0].localeCompare(b[0])).forEach(([name, count]) => categoryBox.append(new Option(`${name} (${count})`, name)));
  const grid = transitionElement("div", "flex:1;min-height:0;overflow:auto;padding:0 10px 10px;");
  left.append(searchBox, categoryBox, grid);

  // -- right: preview, strength, length, settings
  const canvas = transitionElement("canvas", "width:100%;max-width:520px;align-self:center;background:#000;border-radius:6px;");
  canvas.width = LOOK_PREVIEW_WIDTH;
  canvas.height = Math.round(LOOK_PREVIEW_WIDTH * 9 / 16);
  const info = transitionElement("div", "color:#a1a1aa;font-size:12px;min-height:34px;");
  const sliderRow = (name, low, high, step, value, format, onInput) => {
    const row = transitionElement("label", "display:grid;grid-template-columns:150px 1fr 60px;gap:8px;align-items:center;");
    const slider = transitionElement("input", "min-width:0;", { type: "range", min: low, max: Math.max(high, value), step, value });
    const readout = transitionElement("span", "text-align:right;font-variant-numeric:tabular-nums;", { textContent: format(value) });
    slider.oninput = () => {
      readout.textContent = format(Number(slider.value));
      onInput(Number(slider.value));
    };
    row.append(transitionElement("span", "", { textContent: name }), slider, readout);
    return row;
  };
  const strengthRow = sliderRow("Strength", 0, 1, 0.01, intensity, (value) => `${Math.round(value * 100)}%`, (value) => { intensity = value; });
  const lengthRow = sliderRow("Length (seconds)", LOOK_MIN_SECONDS, 10, 0.05, seconds, transitionSeconds, (value) => { seconds = value; });
  const note = transitionElement("div", "color:#a1a1aa;font-size:11px;", {
    textContent: mode === "edit"
      ? "The block keeps its place on the timeline. Drag its edges there to stretch it."
      : `The block is a clip of the timeline: move it and stretch it there. It changes the video where it lies and not its length.`,
  });
  const settings = transitionElement("div", "display:flex;flex-direction:column;gap:7px;");
  const buttons = transitionElement("div", "display:flex;gap:8px;flex-wrap:wrap;margin-top:auto;padding:8px 0 12px;position:sticky;bottom:-12px;background:#18181b;border-top:1px solid #27272a;");
  const message = transitionElement("div", "color:#fca5a5;font-size:12px;min-height:16px;");
  right.append(canvas, info, strengthRow, lengthRow, settings, note, message, buttons);

  // The tiles are drawn on a first frame of the scene, a few per animation frame, as they come into view.
  const tiles = new Map();
  const visible = new Set();
  const queue = [];
  const pumpTiles = () => {
    tileFrameId = 0;
    for (let count = 0; count < LOOK_TILES_PER_FRAME && queue.length && thumbs; count += 1) {
      const tile = queue.shift();
      if (tile.drawn || !tile.node.isConnected) continue;
      thumbs.draw([{ item: tile.item, progress: 0.5, time: 0.45, intensity: 1, params: {} }]);
      tile.art.getContext("2d").drawImage(thumbCanvas, 0, 0);
      tile.drawn = true;
      const failure = thumbs.failure(tile.item);
      if (failure) {
        tile.label.style.color = "#fca5a5";
        tile.node.title = `This ${config.singular} did not compile in the preview: ${failure}`;
      }
    }
    if (queue.length && !closed) tileFrameId = requestAnimationFrame(pumpTiles);
  };
  const enqueue = (tile) => {
    if (tile.drawn || queue.includes(tile)) return;
    queue.push(tile);
    if (!tileFrameId && !closed) tileFrameId = requestAnimationFrame(pumpTiles);
  };
  observer = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const tile = tiles.get(entry.target);
      if (!tile) continue;
      if (entry.isIntersecting) {
        visible.add(tile);
        enqueue(tile);
      } else visible.delete(tile);
    }
  }, { root: grid, rootMargin: "120px" }) : null;

  const refreshGrid = () => {
    observer?.disconnect();
    tiles.clear();
    visible.clear();
    queue.length = 0;
    grid.textContent = "";
    const query = search.trim().toLowerCase();
    const shown = items.filter((item) => (!query || lookSearchText(item).includes(query)) && (!category || (item.category || "Other") === category));
    if (!shown.length) {
      grid.append(transitionElement("div", "padding:14px;color:#a1a1aa;", { textContent: `No ${config.singular} matches.` }));
      return;
    }
    const groups = new Map();
    for (const item of [...shown].sort((a, b) => a.id.localeCompare(b.id))) {
      const name = item.category || "Other";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(item);
    }
    if (mode === "edit" && !query && !category) {
      const none = transitionElement("div", "padding:6px 8px;margin:4px 0;border-radius:5px;cursor:pointer;background:#27272a;", { textContent: `None (remove the ${config.singular} from the timeline)` });
      none.onclick = () => choose("");
      grid.append(none);
    }
    for (const [name, group] of [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
      grid.append(transitionElement("div", `font-weight:900;color:${config.text};margin:10px 0 6px;`, { textContent: `${name} (${group.length})` }));
      const tilesBox = transitionElement("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;");
      for (const item of group) {
        const node = transitionElement("div", `border-radius:6px;cursor:pointer;overflow:hidden;background:#09090b;border:2px solid ${item.id === selected ? config.border : "transparent"};`, { title: item.description || "" });
        node.dataset.lookId = item.id;
        const art = transitionElement("canvas", "display:block;width:100%;aspect-ratio:16/9;background:#27272a;");
        art.width = LOOK_TILE_WIDTH;
        art.height = LOOK_TILE_HEIGHT;
        const label = transitionElement("div", "padding:3px 5px;font-size:11px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", { textContent: transitionName(item.id) });
        node.append(art, label);
        node.onclick = () => choose(item.id);
        tiles.set(node, { item, node, art, label, drawn: false });
        observer?.observe(node);
        tilesBox.append(node);
      }
      grid.append(tilesBox);
    }
    if (!observer) tiles.forEach(enqueue);
  };

  const paintSelection = () => {
    for (const tile of tiles.values()) tile.node.style.borderColor = tile.item.id === selected ? config.border : "transparent";
  };

  const refreshInfo = () => {
    const item = byId(selected);
    info.textContent = item
      ? `${item.description || ""} ${item.origin === "original" ? "An H3 Studio original." : `By ${item.author}, ${item.license} license.`}${preview?.failure(item) ? ` The preview could not compile it: ${preview.failure(item)}` : ""}`
      : mode === "edit" ? `Choosing none removes the block from the timeline.` : `Choose ${article} ${config.singular} from the list.`;
  };

  const refreshSettings = () => {
    settings.textContent = "";
    const item = byId(selected);
    if (!item?.params?.length) return;
    settings.append(transitionElement("div", `font-weight:900;color:${config.text};`, { textContent: "Settings" }));
    for (const param of item.params) {
      settings.append(transitionParamRow(param, overrides[param.name] ?? param.default, (value) => {
        overrides = { ...overrides, [param.name]: value };
      }));
    }
    const reset = button("Reset settings");
    reset.style.alignSelf = "flex-start";
    reset.onclick = () => {
      overrides = {};
      refreshSettings();
    };
    settings.append(reset);
  };

  function choose(name) {
    if (name !== selected) overrides = {};
    selected = name;
    paintSelection();
    refreshInfo();
    refreshSettings();
    refreshButtons();
  }

  searchBox.oninput = () => {
    search = searchBox.value;
    refreshGrid();
  };
  categoryBox.onchange = () => {
    category = categoryBox.value;
    refreshGrid();
  };

  // -- saving
  const finish = (reason) => {
    close();
    commit?.(reason);
  };
  const made = () => ({ look: selected, seconds: Math.round(seconds * 100) / 100, intensity: Math.round(intensity * 100) / 100, params: { ...overrides } });
  const addHere = button(mode === "all" ? "Put it over the whole video" : "Add to the timeline", true);
  addHere.onclick = () => {
    if (!selected) return;
    lookLastSeconds[kind] = made().seconds;
    const current = lookEntries(state, kind);
    const total = lookTimelineEnd(state, kind);
    const spot = mode === "all"
      ? placeNewLook({ total, span: { start: 0, end: total } })
      : placeNewLook({ playhead, seconds: made().seconds, total, range: at === null ? lookSelectedRange(state) : null });
    const entry = { id: freshLookId(current, kind), ...made(), start: spot.start, seconds: spot.seconds };
    state[config.activeKey] = entry.id;
    pushHistory?.();
    state[config.listKey] = normalizeLookEntries([...current, entry], kind);
    finish(`${config.label.slice(0, -1)} added`);
  };
  const apply = button("Apply", true);
  apply.onclick = () => {
    if (!selected) {
      removeLook(state, kind, id, context);
      close();
      return;
    }
    lookLastSeconds[kind] = made().seconds;
    const current = lookEntries(state, kind);
    const stretched = trimLookEnd(existing, existing.start + made().seconds, lookTimelineEnd(state, kind));
    pushHistory?.();
    state[config.listKey] = normalizeLookEntries(current.map((item) => (item.id === id ? { ...item, ...made(), seconds: stretched.seconds } : item)), kind);
    finish(`${config.label.slice(0, -1)} changed`);
  };
  const remove = button("Remove from the timeline");
  remove.onclick = () => {
    removeLook(state, kind, id, context);
    close();
  };
  function refreshButtons() {
    buttons.textContent = "";
    if (mode === "edit") buttons.append(apply, remove);
    else {
      addHere.disabled = !selected;
      addHere.style.opacity = selected ? "1" : "0.5";
      buttons.append(addHere);
    }
  }

  // -- the previews: a real frame of the scene, the chosen look looping over the length of the block
  const aspectOf = (frame) => (frame ? frame.width / frame.height : 16 / 9);
  const fitSize = (aspect) => {
    const width = Math.min(LOOK_PREVIEW_WIDTH, Math.round(LOOK_PREVIEW_MAX_HEIGHT * aspect));
    return [width, Math.round(width / aspect)];
  };
  const thumbCanvas = transitionElement("canvas");
  thumbCanvas.width = LOOK_TILE_WIDTH;
  thumbCanvas.height = LOOK_TILE_HEIGHT;
  thumbs = createLookGl(thumbCanvas);
  preview = createLookGl(canvas);
  if (!preview) {
    info.textContent = `This computer's browser engine has no WebGL 2, so there is no preview. The ${config.singular} still renders in the video.`;
  } else {
    const [w, h] = fitSize(16 / 9);
    canvas.width = w;
    canvas.height = h;
    const placeholder = transitionPlaceholder(transitionSceneName(scene, sceneIndex), 200, w, h);
    preview.setFrame(placeholder);
    thumbs?.setFrame(transitionCover(placeholder, LOOK_TILE_WIDTH, LOOK_TILE_HEIGHT));
    transitionSceneFrame(scene, false).then((frame) => {
      if (closed || !frame) return;
      const [width, height] = fitSize(aspectOf(frame));
      canvas.width = width;
      canvas.height = height;
      preview.setFrame(transitionCover(frame, width, height));
      thumbs?.setFrame(transitionCover(frame, LOOK_TILE_WIDTH, LOOK_TILE_HEIGHT));
      tiles.forEach((tile) => { tile.drawn = false; });
      visible.forEach(enqueue);
    });
    const begin = performance.now();
    const tick = (now) => {
      if (closed) return;
      const length = Math.max(LOOK_MIN_SECONDS, Math.min(LOOK_MAX_SECONDS, seconds)) * 1000;
      const position = (now - begin) % length;
      const item = byId(selected);
      preview.draw(item ? [{ item, progress: position / length, time: position / 1000, intensity, params: overrides }] : []);
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
  }

  refreshGrid();
  refreshInfo();
  refreshSettings();
  refreshButtons();
  grid.querySelector(`[data-look-id="${selected}"]`)?.scrollIntoView({ block: "center" });
}
