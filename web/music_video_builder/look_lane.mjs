// The effect lane and the filter lane of the timeline: one block per look, on rows of their own under the transition lane. A block is dragged to move it, its edges
// are dragged to stretch it, a double click changes the look and a right click opens a menu (change, duplicate, delete). The Delete key removes the selected block.
// A block snaps to the scene edges and to the edges of other blocks, and to the beats when Snap beats is on. Blocks may overlap: they stack in rows. The rules are in
// look_model.mjs; this file draws the blocks and reads the pointer.

import { LOOK_ROW, duplicateLook, lookEntries, lookLaneHeight, lookLaneVisible, lookShared, removeLook, saveLooks, selectLook } from "./look_actions.mjs";
import { LOOK_KINDS, LOOK_ORDER, lookEnd, lookRows, lookSnapToFrame, moveLook, trimLookEnd, trimLookStart } from "./look_model.mjs";
import { openLookPicker } from "./look_picker.mjs";
import { transitionElement, transitionKeyIsTyping, transitionName, transitionSeconds } from "./transition_dom.mjs";
import { transitionShared } from "./transition_actions.mjs";
import { snapTransitionTime } from "./transition_model.mjs";

const LOOK_MENU_STYLE = "position:fixed;z-index:100010;min-width:200px;border:1px solid #3f3f46;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";

function openLookMenu(event, items) {
  document.querySelector(".vrgdg-builder-context-menu")?.remove();
  const menu = transitionElement("div", LOOK_MENU_STYLE, { className: "vrgdg-builder-context-menu" });
  menu.style.left = `${Math.min(window.innerWidth - 220, event.clientX)}px`;
  menu.style.top = `${Math.min(window.innerHeight - 40 * items.length - 20, event.clientY)}px`;
  const close = () => {
    menu.remove();
    window.removeEventListener("pointerdown", onPointer, true);
    window.removeEventListener("keydown", onKey, true);
  };
  const onPointer = (closeEvent) => {
    if (!menu.contains(closeEvent.target)) close();
  };
  const onKey = (keyEvent) => {
    if (keyEvent.key === "Escape") close();
  };
  for (const [text, action] of items) {
    const item = transitionElement("button", "text-align:left;border:0;border-radius:4px;background:transparent;color:inherit;padding:6px 8px;cursor:pointer;font:inherit;", { type: "button", textContent: text });
    item.onclick = () => {
      close();
      action();
    };
    menu.append(item);
  }
  document.body.append(menu);
  window.addEventListener("pointerdown", onPointer, true);
  window.addEventListener("keydown", onKey, true);
}

let lookDeleteKeyInstalled = false;

// The Delete (or Backspace) key removes the selected block, as the Delete item of its menu does. It leaves the key alone while the person types.
function installLookDeleteKey() {
  if (lookDeleteKeyInstalled) return;
  lookDeleteKeyInstalled = true;
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    if (event.ctrlKey || event.metaKey || event.altKey || event.defaultPrevented || transitionKeyIsTyping(event.target) || document.querySelector(".h3-look-picker")) return;
    for (const kind of LOOK_ORDER) {
      const context = lookShared.lanes[kind];
      const id = context?.state[LOOK_KINDS[kind].activeKey];
      if (!id || !document.querySelector(`.h3-look-lane[data-look-kind="${kind}"]`) || !lookEntries(context.state, kind).some((item) => item.id === id)) continue;
      event.preventDefault();
      event.stopPropagation();
      removeLook(context.state, kind, id, context);
      return;
    }
  }, true);
}

// Draws the lane of `kind` into `layer` (the timeline's segment layer, which does not take the pointer itself). `ctx` gives it the page:
//   state, top, total (seconds), edges (the scene edges to snap to), playheadTime(), pushHistory(), commit(reason) (redraw and save)
export function renderLookLane(kind, { layer, state, top, total, edges, playheadTime, pushHistory, commit }) {
  const config = LOOK_KINDS[kind];
  const context = { state, total, edges, playheadTime, pushHistory, commit };
  lookShared.lanes[kind] = context;
  transitionShared.deselectOthers = (current) => {
    for (const other of LOOK_ORDER) {
      if (current[LOOK_KINDS[other].activeKey]) selectLook(current, other, "");
    }
  };
  lookShared.panels[kind]?.();
  if (!lookLaneVisible(state, kind)) return null;
  installLookDeleteKey();
  const list = lookEntries(state, kind);
  const { rows } = lookRows(list);
  const lane = transitionElement(
    "div",
    `position:absolute;left:0;top:${top}px;width:${Math.max(900, Math.ceil(total * state.pxPerSecond))}px;height:${lookLaneHeight(state, kind)}px;pointer-events:auto;background:${config.color}14;border-top:1px solid ${config.color};border-bottom:1px solid ${config.color};`,
    { className: "h3-look-lane" },
  );
  lane.dataset.lookKind = kind;
  lane.onpointerdown = (event) => {
    if (event.target === lane) selectLook(state, kind, "");
  };
  lane.oncontextmenu = (event) => {
    if (event.target !== lane) return;
    event.preventDefault();
    const at = Math.max(0, (event.clientX - lane.getBoundingClientRect().left) / state.pxPerSecond);
    openLookMenu(event, [[`Add ${config.singular} here...`, () => openLookPicker(kind, { ...context, mode: "add", at })]]);
  };
  layer.append(transitionElement("div", `position:absolute;left:4px;top:${top - 12}px;color:${config.text};font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`, { textContent: config.laneLabel }));

  const secondsAt = (event, origin, startTime) => startTime + (event.clientX - origin) / state.pxPerSecond;
  for (const entry of list) {
    const selected = state[config.activeKey] === entry.id;
    const block = transitionElement(
      "div",
      `position:absolute;left:${entry.start * state.pxPerSecond}px;top:${3 + (rows[entry.id] || 0) * LOOK_ROW}px;width:${Math.max(18, entry.seconds * state.pxPerSecond)}px;height:${LOOK_ROW - 4}px;box-sizing:border-box;`
        + `border:1px solid ${selected ? "#f8fafc" : config.border};border-radius:5px;background:${config.color};color:#f8fafc;overflow:hidden;cursor:grab;user-select:none;touch-action:none;${selected ? "box-shadow:0 0 0 2px rgba(248,250,252,.6);" : ""}`,
      { className: "h3-look-block", tabIndex: 0 },
    );
    block.dataset.lookKind = kind;
    block.dataset.lookId = entry.id;
    block.append(transitionElement("div", "position:absolute;left:10px;right:10px;top:0;bottom:0;display:flex;align-items:center;font-size:11px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;", {
      textContent: transitionName(entry.look),
      title: `${transitionName(entry.look)}, ${transitionSeconds(entry.start)} to ${transitionSeconds(lookEnd(entry))}, ${Math.round(entry.intensity * 100)}% strong.`,
    }));
    for (const side of ["start", "end"]) {
      const handle = transitionElement("div", `position:absolute;${side === "start" ? "left" : "right"}:0;top:0;bottom:0;width:7px;cursor:ew-resize;background:rgba(248,250,252,.18);z-index:1;`);
      handle.dataset.edge = side;
      block.append(handle);
    }

    // One pointer gesture: a drag of the body moves the block, a drag of an edge stretches it. The block follows the pointer without a redraw and the change is
    // made once when the pointer is released.
    block.onpointerdown = (event) => {
      if (event.button !== 0) return;
      selectLook(state, kind, entry.id);
      const edge = event.target?.dataset?.edge || "";
      const origin = event.clientX;
      const base = edge === "end" ? lookEnd(entry) : entry.start;
      const snapOptions = {
        points: [...edges, ...list.filter((item) => item.id !== entry.id).flatMap((item) => [item.start, lookEnd(item)])],
        beats: state.showBeatMarkers === false ? [] : state.beats || [],
        snapBeats: Boolean(state.snapToBeats),
        pxPerSecond: state.pxPerSecond,
      };
      let result = { start: entry.start, seconds: entry.seconds };
      let moved = false;
      event.preventDefault();
      try {
        block.setPointerCapture(event.pointerId);
      } catch {
        /* the pointer is gone already: the events still reach the block */
      }
      const apply = (moveEvent) => {
        if (Math.abs(moveEvent.clientX - origin) > 2) moved = true;
        const wanted = secondsAt(moveEvent, origin, base);
        if (edge === "start") result = trimLookStart(entry, lookSnapToFrame(snapTransitionTime(wanted, snapOptions)));
        else if (edge === "end") result = trimLookEnd(entry, lookSnapToFrame(snapTransitionTime(wanted, snapOptions)), total);
        else {
          // Snap whichever end of the block is nearer a snap point.
          const start = snapTransitionTime(wanted, snapOptions);
          const end = snapTransitionTime(wanted + entry.seconds, snapOptions) - entry.seconds;
          result = { start: moveLook(entry, lookSnapToFrame(Math.abs(start - wanted) <= Math.abs(end - wanted) ? start : end), total), seconds: entry.seconds };
        }
        block.style.left = `${result.start * state.pxPerSecond}px`;
        block.style.width = `${Math.max(18, result.seconds * state.pxPerSecond)}px`;
      };
      const finish = () => {
        block.removeEventListener("pointermove", apply);
        block.removeEventListener("pointerup", finish);
        block.removeEventListener("pointercancel", finish);
        if (!moved) return;
        saveLooks(state, kind, list.map((item) => (item.id === entry.id ? { ...item, ...result } : item)), `${config.label.slice(0, -1)} ${edge ? "stretched" : "moved"}`, context);
      };
      block.addEventListener("pointermove", apply);
      block.addEventListener("pointerup", finish);
      block.addEventListener("pointercancel", finish);
    };
    block.ondblclick = (event) => {
      event.stopPropagation();
      openLookPicker(kind, { ...context, mode: "edit", id: entry.id });
    };
    block.oncontextmenu = (event) => {
      event.preventDefault();
      event.stopPropagation();
      selectLook(state, kind, entry.id);
      openLookMenu(event, [
        [`Change the ${config.singular}...`, () => openLookPicker(kind, { ...context, mode: "edit", id: entry.id })],
        ["Duplicate", () => duplicateLook(state, kind, entry.id, context)],
        ["Delete", () => removeLook(state, kind, entry.id, context)],
      ]);
    };
    lane.append(block);
  }
  layer.append(lane);
  return lane;
}
