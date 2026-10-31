// The dialog that chooses a transition, with a preview on the real frames of the scenes around a cut, the length and the transition's own settings.
//   mode "add": put the chosen transition on the timeline at `at` seconds (the playhead when not given)
//   mode "all": put it on every cut
//   mode "edit": change the block `id`

import {
  freshTransitionId, removeTransition, transitionCutTimes, transitionEntries, transitionSelectedRange, transitionShared, transitionTimelineEnd,
} from "./transition_actions.mjs";
import { transitionElement, transitionFromHex, transitionName, transitionSceneName, transitionSeconds, transitionToHex } from "./transition_dom.mjs";
import { transitionCover, transitionPlaceholder, transitionSceneFrame } from "./transition_frames.mjs";
import { createTransitionGl } from "./transition_gl.mjs";
import { loadTransitionLibrary } from "./transition_library.mjs";
import {
  TRANSITION_DEFAULT_SECONDS, TRANSITION_FPS, TRANSITION_MIN_SECONDS, normalizeTransitionEntries, placeNewTransition, transitionEnd, transitionSnapToFrame, transitionWindow, trimTransitionEnd,
} from "./transition_model.mjs";

const TRANSITION_HOLD_MS = 350;

const TRANSITION_MIN_PREVIEW_SECONDS = 0.9;

const TRANSITION_PREVIEW_WIDTH = 520;

const TRANSITION_PREVIEW_MAX_HEIGHT = 340;

let transitionPicker = null;
let transitionLastSeconds = TRANSITION_DEFAULT_SECONDS;

// `onChange(value)` is called while the control moves, `onCommit()` when it is let go.
export function transitionParamRow(param, current, onChange, onCommit) {
  const row = transitionElement("label", "display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;font-size:12px;color:#d4d4d8;");
  const name = transitionElement("span", "overflow:hidden;text-overflow:ellipsis;", { textContent: param.description || transitionName(param.name), title: param.name });
  const field = transitionElement("div", "display:flex;gap:6px;align-items:center;min-width:0;");
  row.append(name, field);
  const numberBox = (value, step, change) => {
    const box = transitionElement("input", "width:70px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:3px 5px;", { type: "number", value, step });
    box.oninput = () => change(Number(box.value));
    box.onchange = () => onCommit?.();
    return box;
  };
  if (param.type === "bool") {
    const box = transitionElement("input", "", { type: "checkbox", checked: Boolean(current) });
    box.onchange = () => {
      onChange(box.checked);
      onCommit?.();
    };
    field.append(box);
  } else if (param.type === "float") {
    const lo = param.min ?? Math.min(0, Number(param.default));
    const hi = param.max ?? Math.max(1, Math.abs(Number(param.default)) * 4);
    const step = param.step ?? (hi - lo) / 200;
    const slider = transitionElement("input", "flex:1;min-width:0;", { type: "range", min: lo, max: Math.max(hi, Number(current)), step, value: current });
    const readout = transitionElement("span", "width:44px;text-align:right;font-variant-numeric:tabular-nums;", { textContent: Number(current).toFixed(2) });
    slider.oninput = () => {
      readout.textContent = Number(slider.value).toFixed(2);
      onChange(Number(slider.value));
    };
    slider.onchange = () => onCommit?.();
    field.append(slider, readout);
  } else if (param.type === "int") {
    field.append(numberBox(current, 1, (v) => onChange(Math.round(v))));
  } else if ((param.type === "vec3" || param.type === "vec4") && param.color) {
    const colorBox = transitionElement("input", "width:46px;height:26px;padding:0;border:1px solid #3f3f46;background:none;", { type: "color", value: transitionToHex(current) });
    colorBox.oninput = () => onChange([...transitionFromHex(colorBox.value), ...current.slice(3)]);
    colorBox.onchange = () => onCommit?.();
    field.append(colorBox);
  } else if (Array.isArray(current)) {
    const isInt = param.type.startsWith("i");
    const boxes = current.map((value, i) => numberBox(value, isInt ? 1 : 0.05, (v) => {
      const next = [...current];
      next[i] = isInt ? Math.round(v) : v;
      current = next;
      onChange(next);
    }));
    field.append(...boxes);
  }
  return row;
}

// The dialog that chooses a transition, with a preview on the real frames of the scenes around a cut.
//   mode "add": put the chosen transition on the timeline at `at` seconds (the playhead when not given)
//   mode "all": put it on every cut
//   mode "edit": change the block `id`
export async function openTransitionPicker({ state, pushHistory, commit, mode = "add", id = "", at = null }) {
  transitionPicker?.close();
  const segments = state.segments || [];
  const existing = mode === "edit" ? transitionEntries(state).find((item) => item.id === id) : null;
  if (!segments.length || (mode === "edit" && !existing)) return;
  const context = { state, pushHistory, commit };
  const playhead = at ?? (Number(transitionShared.clock()) || 0);

  // The scenes the preview plays between: those around the block's cut, else around the cut nearest to the playhead.
  let pair = 0;
  if (existing) {
    const w = transitionWindow(existing, segments);
    pair = Math.min(w.cut ?? w.scene, Math.max(0, segments.length - 2));
  } else {
    const cuts = transitionCutTimes(state).map((time, i) => [Math.abs(time - playhead), i]).sort((a, b) => a[0] - b[0]);
    pair = cuts.length ? cuts[0][1] : 0;
  }
  const first = segments[pair];
  const next = segments[pair + 1] || first;

  const backdrop = transitionElement("div", "position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;", { className: "h3-transition-picker" });
  const dialog = transitionElement("div", "width:min(980px,96vw);height:min(660px,94vh);background:#18181b;color:#f4f4f5;border:1px solid #52525b;border-radius:8px;display:flex;flex-direction:column;box-shadow:0 12px 40px rgba(0,0,0,.6);font-size:13px;");
  backdrop.append(dialog);
  const title = mode === "edit" ? `Change the transition: ${transitionName(existing.transition)}` : mode === "all" ? "Add a transition to every cut" : "Add a transition";
  const header = transitionElement("div", "display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #3f3f46;");
  header.append(transitionElement("div", "font-weight:900;", { textContent: title }));
  const closeButton = transitionElement("button", "background:none;border:1px solid #52525b;color:#e4e4e7;border-radius:4px;padding:2px 9px;cursor:pointer;", { type: "button", textContent: "Close" });
  header.append(closeButton);
  const body = transitionElement("div", "flex:1;min-height:0;display:grid;grid-template-columns:300px 1fr;");
  dialog.append(header, body);
  const left = transitionElement("div", "display:flex;flex-direction:column;min-height:0;border-right:1px solid #3f3f46;");
  const right = transitionElement("div", "padding:12px 14px;overflow:auto;display:flex;flex-direction:column;gap:10px;");
  body.append(left, right);
  document.body.append(backdrop);

  let frameId = 0;
  let closed = false;
  let preview = null;
  const close = () => {
    if (closed) return;
    closed = true;
    cancelAnimationFrame(frameId);
    preview?.dispose();
    backdrop.remove();
    document.removeEventListener("keydown", onKey, true);
    if (transitionPicker?.close === close) transitionPicker = null;
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
  transitionPicker = { close };

  left.append(transitionElement("div", "padding:14px;color:#a1a1aa;", { textContent: "Loading the transitions..." }));
  let items;
  try {
    items = await loadTransitionLibrary();
  } catch (error) {
    left.textContent = "";
    left.append(transitionElement("div", "padding:14px;color:#fca5a5;", { textContent: String(error.message || error) }));
    return;
  }
  if (closed) return;
  left.textContent = "";

  let selected = existing && items.some((item) => item.id === existing.transition) ? existing.transition : "";
  let seconds = existing?.seconds ?? transitionLastSeconds;
  let overrides = existing ? { ...(existing.params || {}) } : {};
  let search = "";
  let tag = "";

  const byId = (name) => items.find((item) => item.id === name) || null;

  // -- left: search, tags and the list
  const searchBox = transitionElement("input", "margin:10px 10px 6px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:6px 8px;", { type: "search", placeholder: "Search transitions" });
  const tagCounts = new Map();
  items.forEach((item) => (item.tags || []).forEach((t) => tagCounts.set(t, (tagCounts.get(t) || 0) + 1)));
  const tagBox = transitionElement("select", "margin:0 10px 8px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:5px 6px;");
  tagBox.append(new Option(`All styles (${items.length})`, ""));
  [...tagCounts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).forEach(([t, count]) => tagBox.append(new Option(`${t} (${count})`, t)));
  const list = transitionElement("div", "flex:1;min-height:0;overflow:auto;padding:0 6px 10px;");
  left.append(searchBox, tagBox, list);

  // -- right: preview, length, settings
  const canvas = transitionElement("canvas", "width:100%;max-width:520px;align-self:center;background:#000;border-radius:6px;");
  canvas.width = TRANSITION_PREVIEW_WIDTH;
  canvas.height = Math.round(TRANSITION_PREVIEW_WIDTH * 9 / 16);
  const info = transitionElement("div", "color:#a1a1aa;font-size:12px;min-height:34px;");
  const lengthRow = transitionElement("label", "display:grid;grid-template-columns:150px 1fr 60px;gap:8px;align-items:center;");
  const lengthSlider = transitionElement("input", "min-width:0;", { type: "range", min: TRANSITION_MIN_SECONDS, max: Math.max(5, seconds), step: 0.05, value: seconds });
  const lengthReadout = transitionElement("span", "text-align:right;font-variant-numeric:tabular-nums;");
  lengthRow.append(transitionElement("span", "", { textContent: "Length (seconds)" }), lengthSlider, lengthReadout);
  const note = transitionElement("div", "color:#a1a1aa;font-size:11px;", {
    textContent: mode === "edit"
      ? "The block keeps its place on the timeline. Drag its edges there to stretch it."
      : "The block is a clip of the timeline: move it and stretch it there. It replaces the same number of frames of the video, so the length of the video and the song stay in step.",
  });
  const settings = transitionElement("div", "display:flex;flex-direction:column;gap:7px;");
  const buttons = transitionElement("div", "display:flex;gap:8px;flex-wrap:wrap;margin-top:auto;padding:8px 0 12px;position:sticky;bottom:-12px;background:#18181b;border-top:1px solid #27272a;");
  const button = (text, primary) => transitionElement("button", `border-radius:5px;padding:7px 14px;cursor:pointer;font-weight:800;border:1px solid ${primary ? "#a78bfa" : "#52525b"};background:${primary ? "#7c3aed" : "#27272a"};color:#fff;`, { type: "button", textContent: text });
  const message = transitionElement("div", "color:#fca5a5;font-size:12px;min-height:16px;");
  right.append(canvas, info, lengthRow, settings, note, message, buttons);

  const refreshList = () => {
    list.textContent = "";
    const query = search.trim().toLowerCase();
    const row = (name, label, subtitle) => {
      const item = transitionElement("div", `padding:6px 8px;border-radius:5px;cursor:pointer;${name === selected ? "background:#5b21b6;" : ""}`);
      if (name === selected) item.dataset.selected = "1";
      item.append(transitionElement("div", "font-weight:800;", { textContent: label }));
      if (subtitle) item.append(transitionElement("div", "font-size:11px;color:#a1a1aa;", { textContent: subtitle }));
      item.onmouseenter = () => {
        if (name !== selected) item.style.background = "#27272a";
      };
      item.onmouseleave = () => {
        item.style.background = name === selected ? "#5b21b6" : "";
      };
      item.onclick = () => choose(name);
      list.append(item);
    };
    if (mode === "edit") row("", "None (hard cut)", "");
    // The app's own transitions first, then the collection, each in alphabetical order.
    const ordered = [...items].sort((a, b) => (a.origin === "original" ? 0 : 1) - (b.origin === "original" ? 0 : 1) || a.id.localeCompare(b.id));
    for (const item of ordered) {
      const text = `${item.id} ${transitionName(item.id)} ${(item.tags || []).join(" ")}`.toLowerCase();
      if (query && !text.includes(query)) continue;
      if (tag && !(item.tags || []).includes(tag)) continue;
      row(item.id, transitionName(item.id), (item.tags || []).join(", "));
    }
  };

  const refreshInfo = () => {
    const item = byId(selected);
    info.textContent = item
      ? `${item.description || ""} ${item.origin === "original" ? "An H3 Studio original." : `By ${item.author}, ${item.license} license.`}${preview?.failure() ? ` The preview could not compile it: ${preview.failure()}` : ""}`
      : mode === "edit" ? "A hard cut: choosing none removes the block from the timeline." : "Choose a transition from the list.";
  };

  const refreshSettings = () => {
    settings.textContent = "";
    const item = byId(selected);
    if (!item?.params?.length) return;
    settings.append(transitionElement("div", "font-weight:900;color:#c4b5fd;", { textContent: "Settings" }));
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

  const refreshLength = () => {
    lengthReadout.textContent = transitionSeconds(seconds);
  };

  function choose(name) {
    if (name !== selected) overrides = {};
    selected = name;
    preview?.setTransition(byId(name));
    refreshList();
    refreshInfo();
    refreshSettings();
    refreshButtons();
  }

  searchBox.oninput = () => {
    search = searchBox.value;
    refreshList();
  };
  tagBox.onchange = () => {
    tag = tagBox.value;
    refreshList();
  };
  lengthSlider.oninput = () => {
    seconds = Number(lengthSlider.value);
    refreshLength();
  };

  // -- saving
  const finish = (reason) => {
    close();
    commit?.(reason);
  };
  const made = () => ({ transition: selected, seconds: Math.round(seconds * 100) / 100, params: { ...overrides } });
  const addHere = button("Add to the timeline", true);
  addHere.onclick = () => {
    if (!selected) return;
    transitionLastSeconds = made().seconds;
    const current = transitionEntries(state);
    const spot = placeNewTransition(current, segments, { playhead, seconds: made().seconds, total: transitionTimelineEnd(state), range: at === null ? transitionSelectedRange(state) : null });
    if (!spot) {
      message.textContent = "There is no free room for it at the playhead. Move the playhead, or make room on the timeline.";
      return;
    }
    const entry = { id: freshTransitionId(current), ...made(), start: spot.start, seconds: spot.seconds };
    state.activeTransitionId = entry.id;
    pushHistory?.();
    state.sceneTransitions = normalizeTransitionEntries([...current, entry], segments);
    finish("Transition added");
  };
  const addAll = button("Add to every cut", true);
  addAll.onclick = () => {
    if (!selected) return;
    transitionLastSeconds = made().seconds;
    // Blocks that sit on a cut are replaced, the others stay.
    const kept = transitionEntries(state).filter((item) => transitionWindow(item, segments).cut === null);
    const added = [];
    transitionCutTimes(state).forEach((cut) => {
      const frames = Math.round(made().seconds * TRANSITION_FPS);
      const start = Math.max(0, transitionSnapToFrame(cut - Math.floor(frames / 2) / TRANSITION_FPS));
      const block = { id: freshTransitionId([...kept, ...added]), ...made(), start };
      if ([...kept, ...added].every((item) => block.start >= transitionEnd(item) - 0.0005 || transitionEnd(block) <= item.start + 0.0005)) added.push(block);
    });
    pushHistory?.();
    state.sceneTransitions = normalizeTransitionEntries([...kept, ...added], segments);
    finish("Transitions added");
  };
  const apply = button("Apply", true);
  apply.onclick = () => {
    if (!selected) {
      removeTransition(state, id, context);
      close();
      return;
    }
    transitionLastSeconds = made().seconds;
    const current = transitionEntries(state);
    const stretched = trimTransitionEnd(current, id, existing.start + made().seconds, transitionTimelineEnd(state));
    pushHistory?.();
    state.sceneTransitions = normalizeTransitionEntries(current.map((item) => (item.id === id ? { ...item, ...made(), seconds: stretched.seconds } : item)), segments);
    finish("Transition changed");
  };
  const remove = button("Remove from the timeline");
  remove.onclick = () => {
    removeTransition(state, id, context);
    close();
  };
  const refreshButtons = () => {
    buttons.textContent = "";
    if (mode === "edit") buttons.append(apply, remove);
    else {
      addHere.disabled = !selected;
      addAll.disabled = !selected;
      if (mode === "add") buttons.append(addHere);
      if (mode === "all" || segments.length > 1) buttons.append(addAll);
    }
    for (const node of [addHere, addAll, apply]) node.style.opacity = node.disabled ? "0.5" : "1";
  };

  // -- the preview: the scenes' real frames, played in a loop
  const aspectOf = (frame) => (frame ? frame.width / frame.height : 16 / 9);
  const fitSize = (aspect) => {
    const width = Math.min(TRANSITION_PREVIEW_WIDTH, Math.round(TRANSITION_PREVIEW_MAX_HEIGHT * aspect));
    return [width, Math.round(width / aspect)];
  };
  preview = createTransitionGl(canvas);
  if (!preview) {
    info.textContent = "This computer's browser engine has no WebGL 2, so there is no preview. The transition still renders in the video.";
  } else {
    const [w, h] = fitSize(16 / 9);
    canvas.width = w;
    canvas.height = h;
    preview.setFrames(transitionPlaceholder(transitionSceneName(first, pair), 200, w, h), transitionPlaceholder(transitionSceneName(next, pair + 1), 330, w, h));
    preview.setTransition(byId(selected));
    Promise.all([transitionSceneFrame(first, true), transitionSceneFrame(next, false)]).then(([a, b]) => {
      if (closed || (!a && !b)) return;
      const [width, height] = fitSize(aspectOf(a || b));
      canvas.width = width;
      canvas.height = height;
      preview.setFrames(transitionCover(a || transitionPlaceholder(transitionSceneName(first, pair), 200, width, height), width, height), transitionCover(b || transitionPlaceholder(transitionSceneName(next, pair + 1), 330, width, height), width, height));
    });
    const begin = performance.now();
    const tick = (now) => {
      if (closed) return;
      const length = Math.max(TRANSITION_MIN_PREVIEW_SECONDS, seconds) * 1000;
      const position = (now - begin) % (length + 2 * TRANSITION_HOLD_MS);
      const progress = position < TRANSITION_HOLD_MS ? 0 : position > TRANSITION_HOLD_MS + length ? 1 : (position - TRANSITION_HOLD_MS) / length;
      preview.draw(progress, overrides);
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
  }

  refreshLength();
  refreshList();
  refreshInfo();
  refreshSettings();
  refreshButtons();
  list.querySelector("[data-selected]")?.scrollIntoView({ block: "center" });
}
