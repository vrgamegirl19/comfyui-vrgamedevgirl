// Post Process > Transitions, next to Overlays: add a transition to the timeline, list the blocks and edit the selected one (its start, its length and
// the transition's own settings). The player shows a change as it is made and the change is saved when the control is let go.

import {
  centerTransitionOnCut, describeTransition, duplicateTransition, removeTransition, saveTransitions, selectTransition, transitionCutTimes, transitionEntries,
  transitionShared, transitionTimelineEnd, TRANSITION_BLOCK_STYLE,
} from "./transition_actions.mjs";
import { transitionButton, transitionElement, transitionName, transitionSeconds } from "./transition_dom.mjs";
import { loadTransitionLibrary, transitionLibrary } from "./transition_library.mjs";
import { TRANSITION_FPS, moveTransition, normalizeTransitionEntries, transitionEnd, transitionSnapToFrame, trimTransitionEnd } from "./transition_model.mjs";
import { openTransitionPicker, transitionParamRow } from "./transition_picker.mjs";

let transitionPanel = null;

function refreshTransitionPanel() {
  if (transitionPanel?.pane.isConnected && transitionPanel.pane.style.display !== "none") renderTransitionPanel();
}

function renderTransitionPanel() {
  const context = { ...transitionPanel.ctx, ...(transitionShared.lane || {}) };
  const { state } = context;
  const pane = transitionPanel.pane;
  pane.textContent = "";
  const segments = state.segments || [];
  const list = transitionEntries(state);
  pane.append(transitionElement("div", "color:#a1a1aa;font-size:12px;line-height:1.45;", {
    textContent: "Transitions are blocks on the timeline, like overlays. Add one at the playhead, drag it to move it and drag its edges to stretch it. On a cut it blends one scene into the next, inside a scene it is an effect on the footage. The player shows it while it plays, and Render All and Export draw it into the video.",
  }));
  if (!segments.length) {
    pane.append(transitionElement("div", "color:#fcd34d;", { textContent: "A project needs scenes before a transition can be added." }));
    return;
  }
  const bar = transitionElement("div", "display:flex;gap:8px;flex-wrap:wrap;");
  const add = transitionButton("Add a transition...", true);
  add.title = "Choose a transition and put it at the playhead (over the selected range of the timeline, if there is one).";
  add.onclick = () => openTransitionPicker({ ...context, mode: "add" });
  const every = transitionButton("Add to every cut...");
  every.disabled = segments.length < 2;
  every.onclick = () => openTransitionPicker({ ...context, mode: "all" });
  const clear = transitionButton("Clear all");
  clear.disabled = !list.length;
  clear.onclick = () => {
    state.activeTransitionId = "";
    saveTransitions(state, [], "Transitions cleared", context);
  };
  bar.append(add, every, clear);
  pane.append(bar);

  pane.append(transitionElement("div", "font-weight:900;color:#c4b5fd;", { textContent: list.length ? `On the timeline (${list.length})` : "Nothing on the timeline yet" }));
  const rows = transitionElement("div", "display:flex;flex-direction:column;gap:6px;");
  for (const entry of list) {
    const on = entry.id === state.activeTransitionId;
    const where = describeTransition(entry, segments);
    const row = transitionElement("div", `display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:6px;cursor:pointer;border:1px solid ${on ? "#f8fafc" : TRANSITION_BLOCK_STYLE[where.kind].border};background:${on ? "rgba(91,33,182,.5)" : "rgba(91,33,182,.22)"};`, { title: where.text });
    const text = transitionElement("div", "flex:1;min-width:0;");
    text.append(
      transitionElement("div", "font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", { textContent: `${where.kind === "cut" ? "⇆ " : ""}${transitionName(entry.transition)}` }),
      transitionElement("div", "font-size:12px;color:#ddd6fe;", { textContent: `${transitionSeconds(entry.start)} to ${transitionSeconds(transitionEnd(entry))}, ${where.kind === "cut" ? "on a cut" : "inside a scene"}` }),
    );
    row.append(text);
    row.onclick = () => selectTransition(state, entry.id);
    rows.append(row);
  }
  pane.append(rows);

  const entry = list.find((item) => item.id === state.activeTransitionId);
  if (entry) pane.append(renderTransitionInspector(entry, context));
}

// The settings of the selected block. The player shows a change as it is made and the change is saved when the control is let go.
function renderTransitionInspector(entry, context) {
  const { state, pushHistory, commit } = context;
  const box = transitionElement("div", "display:flex;flex-direction:column;gap:9px;border:1px solid #5b21b6;border-radius:7px;padding:10px;background:#18181b;");
  const item = transitionLibrary().find((candidate) => candidate.id === entry.transition);
  if (!transitionLibrary().length) loadTransitionLibrary().then(refreshTransitionPanel).catch(() => {});
  const head = transitionElement("div", "display:flex;align-items:center;gap:8px;");
  head.append(transitionElement("div", "flex:1;font-weight:900;", { textContent: transitionName(entry.transition) }));
  const change = transitionButton("Change...");
  change.onclick = () => openTransitionPicker({ ...context, mode: "edit", id: entry.id });
  head.append(change);
  box.append(head);
  if (item?.description) box.append(transitionElement("div", "font-size:12px;color:#a1a1aa;", { textContent: item.description }));

  let pending = false;
  const edit = (change) => {
    if (!pending) {
      pushHistory?.();
      pending = true;
    }
    state.sceneTransitions = transitionEntries(state).map((candidate) => (candidate.id === entry.id ? { ...candidate, ...change(candidate) } : candidate));
  };
  const finish = (reason) => {
    pending = false;
    state.sceneTransitions = normalizeTransitionEntries(state.sceneTransitions, state.segments || []);
    commit?.(reason);
  };
  const numberRow = (name, value, step, onChange) => {
    const row = transitionElement("label", "display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;font-size:12px;color:#d4d4d8;");
    const input = transitionElement("input", "width:90px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:3px 5px;", { type: "number", value: Number(value).toFixed(3), step });
    input.onchange = () => onChange(Number(input.value));
    row.append(transitionElement("span", "", { textContent: name }), input);
    return row;
  };
  const list = transitionEntries(state);
  box.append(
    numberRow("Starts at (seconds)", entry.start, 1 / TRANSITION_FPS, (value) => {
      const start = moveTransition(list, entry.id, transitionSnapToFrame(Math.max(0, value)), transitionTimelineEnd(state));
      edit(() => ({ start }));
      finish("Transition moved");
    }),
    numberRow("Length (seconds)", entry.seconds, 1 / TRANSITION_FPS, (value) => {
      const stretched = trimTransitionEnd(list, entry.id, entry.start + transitionSnapToFrame(value), transitionTimelineEnd(state));
      edit(() => ({ seconds: stretched.seconds }));
      finish("Transition stretched");
    }),
  );
  if (item?.params?.length) {
    box.append(transitionElement("div", "font-weight:900;color:#c4b5fd;", { textContent: "Settings" }));
    for (const param of item.params) {
      box.append(transitionParamRow(param, entry.params?.[param.name] ?? param.default, (value) => edit((candidate) => ({ params: { ...candidate.params, [param.name]: value } })), () => finish("Transition settings changed")));
    }
    const reset = transitionButton("Reset settings");
    reset.style.alignSelf = "flex-start";
    reset.onclick = () => {
      edit(() => ({ params: {} }));
      finish("Transition settings changed");
    };
    box.append(reset);
  }
  const actions = transitionElement("div", "display:flex;gap:8px;flex-wrap:wrap;");
  const center = transitionButton("Center on the nearest cut");
  center.disabled = transitionCutTimes(state).length === 0;
  center.onclick = () => centerTransitionOnCut(state, entry.id, context);
  const copy = transitionButton("Duplicate");
  copy.onclick = () => duplicateTransition(state, entry.id, context);
  const remove = transitionButton("Delete");
  remove.onclick = () => removeTransition(state, entry.id, context);
  actions.append(center, copy, remove);
  box.append(actions);
  return box;
}

// The Builder's Post Process tabs call this while they draw. The Transitions tab and its pane are made on the first call, beside the tab that is
// there already, and the Builder styles and shows them like its own.
export function createTransitionTab({ tabBar, pane, state, pushHistory, open, commit, redraw }) {
  if (!transitionPanel) {
    const sibling = [...tabBar.children].find((child) => child.tagName === "BUTTON");
    const tab = sibling ? sibling.cloneNode(false) : transitionElement("button", "", { type: "button" });
    tab.style.display = "";
    tab.textContent = "Transitions";
    tab.title = "Add transitions to the timeline, move them and stretch them like overlays.";
    const content = transitionElement("div", "display:none;overflow:auto;padding:10px;min-height:0;flex:1 1 auto;flex-direction:column;gap:10px;");
    tabBar.append(tab);
    pane.append(content);
    transitionPanel = { tab, pane: content, ctx: {} };
    transitionShared.refreshPanel = refreshTransitionPanel;
    tab.onclick = () => {
      const { state: current, redraw: draw } = transitionPanel.ctx;
      const opened = current.transitionLaneOpen;
      current.transitionLaneOpen = true;
      transitionPanel.ctx.open();
      if (!opened) draw?.();
    };
  }
  Object.assign(transitionPanel.ctx, { state, pushHistory, open, commit, redraw });
  return { tab: transitionPanel.tab, pane: transitionPanel.pane, render: renderTransitionPanel };
}
