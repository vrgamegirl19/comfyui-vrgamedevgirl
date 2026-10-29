// Post Process > Effects and Post Process > Filters, next to Overlays and Transitions: add a look to the timeline, list the blocks and edit the selected one (its
// start, its length, its strength and the look's own settings). The player shows a change as it is made and the change is saved when the control is let go.

import { duplicateLook, lookEntries, lookShared, lookTimelineEnd, removeLook, saveLooks, selectLook } from "./look_actions.mjs";
import { loadLookLibrary, lookLibrary } from "./look_library.mjs";
import { LOOK_FPS, LOOK_KINDS, lookEnd, lookSnapToFrame, moveLook, normalizeLookEntries, trimLookEnd } from "./look_model.mjs";
import { openLookPicker } from "./look_picker.mjs";
import { transitionButton, transitionElement, transitionName, transitionSeconds } from "./transition_dom.mjs";
import { transitionParamRow } from "./transition_picker.mjs";

const lookPanels = {};

function refreshLookPanel(kind) {
  const panel = lookPanels[kind];
  if (panel?.pane.isConnected && panel.pane.style.display !== "none") renderLookPanel(kind);
}

function renderLookPanel(kind) {
  const config = LOOK_KINDS[kind];
  const context = { ...lookPanels[kind].ctx, ...(lookShared.lanes[kind] || {}) };
  const { state } = context;
  const pane = lookPanels[kind].pane;
  pane.textContent = "";
  const list = lookEntries(state, kind);
  pane.append(transitionElement("div", "color:#a1a1aa;font-size:12px;line-height:1.45;", {
    textContent: `${config.label} are blocks on the timeline, like overlays. ${config.blurb} Add one at the playhead, drag it to move it and drag its edges to stretch it. Blocks may overlap. The player shows them while it plays, and Render All and Export draw them into the video.`,
  }));
  if (!(state.segments || []).length) {
    pane.append(transitionElement("div", "color:#fcd34d;", { textContent: `A project needs scenes before ${config.singular === "effect" ? "an" : "a"} ${config.singular} can be added.` }));
    return;
  }
  const button = (text, primary) => {
    const node = transitionButton(text, primary);
    if (primary) {
      node.style.background = config.color;
      node.style.borderColor = config.border;
    }
    return node;
  };
  const bar = transitionElement("div", "display:flex;gap:8px;flex-wrap:wrap;");
  const add = button(`Add ${config.singular === "effect" ? "an" : "a"} ${config.singular}...`, true);
  add.title = `Choose ${config.singular === "effect" ? "an" : "a"} ${config.singular} and put it at the playhead (over the selected range of the timeline, if there is one).`;
  add.onclick = () => openLookPicker(kind, { ...context, mode: "add" });
  const whole = button("Whole video...");
  whole.title = `Choose ${config.singular === "effect" ? "an" : "a"} ${config.singular} and put it over the whole video.`;
  whole.onclick = () => openLookPicker(kind, { ...context, mode: "all" });
  const clear = button("Clear all");
  clear.disabled = !list.length;
  clear.onclick = () => {
    state[config.activeKey] = "";
    saveLooks(state, kind, [], `${config.label} cleared`, context);
  };
  bar.append(add, whole, clear);
  pane.append(bar);

  pane.append(transitionElement("div", `font-weight:900;color:${config.text};`, { textContent: list.length ? `On the timeline (${list.length})` : "Nothing on the timeline yet" }));
  const rows = transitionElement("div", "display:flex;flex-direction:column;gap:6px;");
  for (const entry of list) {
    const on = entry.id === state[config.activeKey];
    const row = transitionElement("div", `display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:6px;cursor:pointer;border:1px solid ${on ? "#f8fafc" : config.border};background:${config.color}${on ? "88" : "38"};`);
    const text = transitionElement("div", "flex:1;min-width:0;");
    text.append(
      transitionElement("div", "font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", { textContent: transitionName(entry.look) }),
      transitionElement("div", `font-size:12px;color:${config.text};`, { textContent: `${transitionSeconds(entry.start)} to ${transitionSeconds(lookEnd(entry))}, ${Math.round(entry.intensity * 100)}% strong` }),
    );
    row.append(text);
    row.onclick = () => selectLook(state, kind, entry.id);
    rows.append(row);
  }
  pane.append(rows);

  const entry = list.find((item) => item.id === state[config.activeKey]);
  if (entry) pane.append(renderLookInspector(kind, entry, context, button));
}

// The settings of the selected block. The player shows a change as it is made and the change is saved when the control is let go.
function renderLookInspector(kind, entry, context, button) {
  const config = LOOK_KINDS[kind];
  const { state, pushHistory, commit } = context;
  const box = transitionElement("div", `display:flex;flex-direction:column;gap:9px;border:1px solid ${config.color};border-radius:7px;padding:10px;background:#18181b;`);
  const item = lookLibrary(kind).find((candidate) => candidate.id === entry.look);
  if (!lookLibrary(kind).length) loadLookLibrary(kind).then(() => refreshLookPanel(kind)).catch(() => {});
  const head = transitionElement("div", "display:flex;align-items:center;gap:8px;");
  head.append(transitionElement("div", "flex:1;font-weight:900;", { textContent: transitionName(entry.look) }));
  const change = button("Change...");
  change.onclick = () => openLookPicker(kind, { ...context, mode: "edit", id: entry.id });
  head.append(change);
  box.append(head);
  if (item?.description) box.append(transitionElement("div", "font-size:12px;color:#a1a1aa;", { textContent: item.description }));

  let pending = false;
  const edit = (change) => {
    if (!pending) {
      pushHistory?.();
      pending = true;
    }
    state[config.listKey] = lookEntries(state, kind).map((candidate) => (candidate.id === entry.id ? { ...candidate, ...change(candidate) } : candidate));
  };
  const finish = (reason) => {
    pending = false;
    state[config.listKey] = normalizeLookEntries(state[config.listKey], kind);
    commit?.(reason);
  };
  const numberRow = (name, value, step, onChange) => {
    const row = transitionElement("label", "display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;font-size:12px;color:#d4d4d8;");
    const input = transitionElement("input", "width:90px;background:#09090b;color:#f4f4f5;border:1px solid #3f3f46;border-radius:4px;padding:3px 5px;", { type: "number", value: Number(value).toFixed(3), step });
    input.onchange = () => onChange(Number(input.value));
    row.append(transitionElement("span", "", { textContent: name }), input);
    return row;
  };
  const total = lookTimelineEnd(state, kind);
  box.append(
    numberRow("Starts at (seconds)", entry.start, 1 / LOOK_FPS, (value) => {
      const start = moveLook(entry, lookSnapToFrame(Math.max(0, value)), total);
      edit(() => ({ start }));
      finish(`${config.label.slice(0, -1)} moved`);
    }),
    numberRow("Length (seconds)", entry.seconds, 1 / LOOK_FPS, (value) => {
      const stretched = trimLookEnd(entry, entry.start + lookSnapToFrame(value), total);
      edit(() => ({ seconds: stretched.seconds }));
      finish(`${config.label.slice(0, -1)} stretched`);
    }),
  );
  const strength = transitionElement("label", "display:grid;grid-template-columns:150px 1fr 44px;gap:8px;align-items:center;font-size:12px;color:#d4d4d8;");
  const slider = transitionElement("input", "min-width:0;", { type: "range", min: 0, max: 1, step: 0.01, value: entry.intensity });
  const readout = transitionElement("span", "text-align:right;font-variant-numeric:tabular-nums;", { textContent: `${Math.round(entry.intensity * 100)}%` });
  slider.oninput = () => {
    readout.textContent = `${Math.round(Number(slider.value) * 100)}%`;
    edit(() => ({ intensity: Number(slider.value) }));
  };
  slider.onchange = () => finish(`${config.label.slice(0, -1)} strength changed`);
  strength.append(transitionElement("span", "", { textContent: "Strength" }), slider, readout);
  box.append(strength);
  if (item?.params?.length) {
    box.append(transitionElement("div", `font-weight:900;color:${config.text};`, { textContent: "Settings" }));
    for (const param of item.params) {
      box.append(transitionParamRow(param, entry.params?.[param.name] ?? param.default, (value) => edit((candidate) => ({ params: { ...candidate.params, [param.name]: value } })), () => finish(`${config.label.slice(0, -1)} settings changed`)));
    }
    const reset = button("Reset settings");
    reset.style.alignSelf = "flex-start";
    reset.onclick = () => {
      edit(() => ({ params: {} }));
      finish(`${config.label.slice(0, -1)} settings changed`);
    };
    box.append(reset);
  }
  const actions = transitionElement("div", "display:flex;gap:8px;flex-wrap:wrap;");
  const copy = button("Duplicate");
  copy.onclick = () => duplicateLook(state, kind, entry.id, context);
  const remove = button("Delete");
  remove.onclick = () => removeLook(state, kind, entry.id, context);
  actions.append(copy, remove);
  box.append(actions);
  return box;
}

// The Builder's Post Process tabs call this while they draw. The tab of `kind` and its pane are made on the first call, beside the tabs that are there already, and
// the Builder styles and shows them like its own.
export function createLookTab(kind, { tabBar, pane, state, pushHistory, open, commit, redraw }) {
  const config = LOOK_KINDS[kind];
  if (!lookPanels[kind]) {
    const sibling = [...tabBar.children].find((child) => child.tagName === "BUTTON");
    const tab = sibling ? sibling.cloneNode(false) : transitionElement("button", "", { type: "button" });
    tab.style.display = "";
    tab.textContent = config.label;
    tab.title = `Add ${config.plural} to the timeline, move them and stretch them like overlays.`;
    const content = transitionElement("div", "display:none;overflow:auto;padding:10px;min-height:0;flex:1 1 auto;flex-direction:column;gap:10px;");
    tabBar.append(tab);
    pane.append(content);
    lookPanels[kind] = { tab, pane: content, ctx: {} };
    lookShared.panels[kind] = () => refreshLookPanel(kind);
    tab.onclick = () => {
      const { state: current, redraw: draw } = lookPanels[kind].ctx;
      const opened = current[config.openKey];
      current[config.openKey] = true;
      lookPanels[kind].ctx.open();
      if (!opened) draw?.();
    };
  }
  Object.assign(lookPanels[kind].ctx, { state, pushHistory, open, commit, redraw });
  return { tab: lookPanels[kind].tab, pane: lookPanels[kind].pane, render: () => renderLookPanel(kind) };
}
