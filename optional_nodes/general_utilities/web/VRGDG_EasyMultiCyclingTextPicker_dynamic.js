import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_EasyMultiCyclingTextPicker";
const MAX_PICKERS = 20;
const BASE_OUTPUTS = [
  { name: "combined_formatted_text", type: "STRING" },
  { name: "results_json", type: "STRING" },
];
const PRESET_ITEMS = {
  "Camera Motion": `Slow push-in
Track right
Track left
Dolly backward
Handheld follow
Over-the-shoulder push-in
Slow pan right
Slow pan left
Tilt up
Tilt down
Arc around subject
Orbit shot
Low-angle tracking shot
Crane rising move
Slow zoom-in`,
  "Character Movement/Motion": `Walks toward camera with confident swagger
Strides across the frame
Leans toward the camera
Points into the lens
Throws arms wide
Raises both hands overhead
Runs a hand through their hair
Slowly backs away from the camera
Drops to one knee
Throws their head back
Whips a jacket off one shoulder
Stomps forward with attitude
Tilts chin upward
Reaches toward the camera
Collapses dramatically to the floor`,
  Lighting: `Soft natural light
Hard direct sunlight
Warm tungsten light
Cool fluorescent light
Neon nightclub light
Moody low-key lighting
High-key studio lighting
Backlit silhouette
Rim lighting
Side lighting
Top-down lighting
Underlighting
Golden hour light
Blue hour light
Strobe lighting`,
  "Time of Day": `Pre-dawn
Dawn
Early morning
Mid-morning
Late morning
Noon
Early afternoon
Mid-afternoon
Late afternoon
Golden hour
Sunset
Dusk
Blue hour
Night
After midnight`,
  Weather: `Clear sky
Partly cloudy
Overcast
Light rain
Heavy rain
Thunderstorm
Drizzle
Fog
Mist
Snowfall
Blizzard
Hail
Strong wind
Dust storm
Humid haze`,
  Dialogue: "",
  "Facial Expression": `Calm expression
Serious expression
Confident smirk
Cold stare
Worried expression
Sad expression
Angry glare
Fearful expression
Surprised expression
Blank expression
Dreamy expression
Suspicious look
Pained expression
Defiant expression
Soft smile`,
  Emotion: `Joyful
Melancholic
Anxious
Furious
Heartbroken
Hopeful
Jealous
Lonely
Nostalgic
Conflicted
Euphoric
Ashamed
Determined
Vengeful
Peaceful`,
  Custom: "",
};
const PRESETS = Object.keys(PRESET_ITEMS);
const MODES = ["index", "random", "random no repeat"];
const FIELDS = ["preset", "items", "label", "selection_mode", "two_item_template", "pick_count"];
const MIN_NODE_WIDTH = 820;
const COLORS = {
  panel: "#151a21",
  panelBorder: "#344153",
  header: "#e5e7eb",
  text: "#d8dee9",
  dim: "#93a4b7",
  cell: "#202833",
  cellBorder: "#475569",
  button: "#283244",
  danger: "#cbd5e1",
};

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgEasyOriginalType")) {
    widget.__vrgdgEasyOriginalType = widget.type;
    widget.__vrgdgEasyOriginalComputeSize = widget.computeSize;
  }
  widget.hidden = !visible;
  widget.serialize = true;
  if (visible) {
    widget.type = widget.__vrgdgEasyOriginalType;
    if (widget.__vrgdgEasyOriginalComputeSize) widget.computeSize = widget.__vrgdgEasyOriginalComputeSize;
    else delete widget.computeSize;
  } else {
    widget.type = widget.__vrgdgEasyOriginalType;
    widget.computeSize = () => [0, -4];
  }
}

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
}

function getPickerCount(node) {
  const count = Number(getWidget(node, "picker_count")?.value ?? 0);
  return Math.max(0, Math.min(MAX_PICKERS, Number.isFinite(count) ? Math.round(count) : 0));
}

function setPickerCount(node, count) {
  setWidgetValue(node, "picker_count", Math.max(0, Math.min(MAX_PICKERS, Math.round(Number(count) || 0))));
}

function getPicker(node, index) {
  const preset = String(getWidget(node, `preset_${index}`)?.value || (index === 1 ? "Camera Motion" : "Custom"));
  const items = String(getWidget(node, `items_${index}`)?.value || "");
  const label = String(getWidget(node, `label_${index}`)?.value || "");
  const mode = String(getWidget(node, `selection_mode_${index}`)?.value || "index");
  const template = String(getWidget(node, `two_item_template_${index}`)?.value || "start with {item1} then follow with {item2}");
  const pickCount = Number(getWidget(node, `pick_count_${index}`)?.value ?? 1);
  return {
    preset,
    items,
    label,
    mode: MODES.includes(mode) ? mode : "index",
    template,
    pickCount: Math.max(1, Math.min(50, Number.isFinite(pickCount) ? Math.round(pickCount) : 1)),
  };
}

function getRows(node) {
  const count = getPickerCount(node);
  const rows = [];
  for (let i = 1; i <= count; i++) rows.push({ index: i, ...getPicker(node, i) });
  return rows;
}

function setPicker(node, index, values) {
  const preset = String(values.preset ?? "Custom");
  const label = String(values.label ?? (preset === "Custom" ? "" : preset));
  setWidgetValue(node, `preset_${index}`, preset);
  setWidgetValue(node, `items_${index}`, String(values.items ?? ""));
  setWidgetValue(node, `label_${index}`, label);
  setWidgetValue(node, `selection_mode_${index}`, values.mode || "index");
  setWidgetValue(node, `two_item_template_${index}`, String(values.template || "start with {item1} then follow with {item2}"));
  setWidgetValue(node, `pick_count_${index}`, values.pickCount || 1);
}

function applyPreset(node, index, preset) {
  const current = getPicker(node, index);
  setPicker(node, index, {
    ...current,
    preset,
    label: preset === "Custom" ? current.label : preset,
    items: preset === "Custom" ? current.items : PRESET_ITEMS[preset],
  });
}

function hideStorageWidgets(node) {
  setWidgetVisible(getWidget(node, "picker_count"), false);
  setWidgetVisible(getWidget(node, "joiner"), false);
  for (let i = 1; i <= MAX_PICKERS; i++) {
    for (const field of FIELDS) setWidgetVisible(getWidget(node, `${field}_${i}`), false);
  }
}

function ensureOutput(node, index, name, type) {
  const current = node.outputs?.[index];
  if (current) {
    current.name = name;
    current.type = type;
    return;
  }
  node.addOutput(name, type);
}

function refreshOutputs(node) {
  const count = getPickerCount(node);
  for (let i = 0; i < BASE_OUTPUTS.length; i++) ensureOutput(node, i, BASE_OUTPUTS[i].name, BASE_OUTPUTS[i].type);
  for (let i = 1; i <= count; i++) ensureOutput(node, BASE_OUTPUTS.length + i - 1, `formatted_text_${i}`, "STRING");
  for (let i = (node.outputs || []).length - 1; i >= BASE_OUTPUTS.length + count; i--) node.removeOutput(i);
}

function getGraphNodes(graph) {
  if (!graph) return [];
  if (Array.isArray(graph._nodes)) return graph._nodes;
  if (Array.isArray(graph.nodes)) return graph.nodes;
  if (typeof graph._nodes_by_id === "object" && graph._nodes_by_id) return Object.values(graph._nodes_by_id);
  return [];
}

function findSourceNode(graph, names, type = "INT") {
  const accepted = new Set(names.map((name) => String(name).trim().toLowerCase()));
  return getGraphNodes(graph).find((node) => {
    if (node === graph?.inputNode) return false;
    const title = String(node?.title || node?.name || node?.label || "").trim().toLowerCase();
    const nodeType = String(node?.type || node?.comfyClass || "").trim().toLowerCase();
    const outputs = Array.isArray(node?.outputs) ? node.outputs : [];
    const hasTypedOutput = outputs.some((output) => String(output?.type || "").toUpperCase() === type);
    const outputNameMatches = outputs.some((output) => accepted.has(String(output?.name || output?.label || "").trim().toLowerCase()));
    return hasTypedOutput && (accepted.has(title) || accepted.has(nodeType) || outputNameMatches);
  }) || null;
}

function findFirstOutputSlot(node, type = "INT") {
  return (node?.outputs || []).findIndex((output) => String(output?.type || "").toUpperCase() === type);
}

function connectSource(node, sourceNode, sourceSlot, targetInputName) {
  const targetSlot = (node?.inputs || []).findIndex((input) => input?.name === targetInputName);
  if (!sourceNode || sourceSlot < 0 || targetSlot < 0) return;
  if (node.inputs[targetSlot]?.link !== null && node.inputs[targetSlot]?.link !== undefined) return;
  sourceNode.connect?.(sourceSlot, node, targetSlot);
}

function autoWireIndexSeed(node) {
  const graph = node?.graph || app.graph;
  const indexNode = findSourceNode(graph, ["index", "get_index"], "INT");
  const seedNode = findSourceNode(graph, ["random seed", "seed", "random_seed"], "INT");
  const indexSlot = findFirstOutputSlot(indexNode, "INT");
  const seedSlot = findFirstOutputSlot(seedNode, "INT");
  const count = getPickerCount(node);
  for (let i = 1; i <= count; i++) {
    connectSource(node, indexNode, indexSlot, `index_${i}`);
    connectSource(node, seedNode, seedSlot, `seed_${i}`);
  }
}

function ensureInputSocket(node, name, type) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  if (inputs.some((input) => input?.name === name)) return;
  if (typeof node?.addInput === "function") {
    node.addInput(name, type);
    return;
  }
  node.inputs = inputs;
  node.inputs.push({ localized_name: name, name, type, widget: { name }, link: null });
}

function refreshIndexSeedInputs(node) {
  const socketCount = Math.max(1, getPickerCount(node));
  const keepNames = new Set();
  for (let i = 1; i <= socketCount; i++) {
    keepNames.add(`index_${i}`);
    keepNames.add(`seed_${i}`);
  }
  node.inputs = (node.inputs || []).filter((input) => !/^index_\d+$|^seed_\d+$/.test(String(input?.name || "")) || keepNames.has(input.name));
  for (let i = 1; i <= socketCount; i++) {
    ensureInputSocket(node, `index_${i}`, "INT");
    ensureInputSocket(node, `seed_${i}`, "INT");
  }
}

function requestDraw(node) {
  setWidgetValue(node, "joiner", "newline");
  refreshOutputs(node);
  refreshIndexSeedInputs(node);
  autoWireIndexSeed(node);
  node.setSize([Math.max(MIN_NODE_WIDTH, node.size?.[0] || MIN_NODE_WIDTH), node.computeSize()[1]]);
  app.graph?.setDirtyCanvas?.(true, true);
}

function drawRoundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function clipText(ctx, text, maxWidth) {
  const value = String(text || "-");
  if (ctx.measureText(value).width <= maxWidth) return value;
  let clipped = value;
  while (clipped.length > 1 && ctx.measureText(`${clipped}...`).width > maxWidth) {
    clipped = clipped.slice(0, -1);
  }
  return `${clipped}...`;
}

function getCellLines(ctx, value, maxWidth, maxLines = 2) {
  const words = String(value || "-").split(/\s+/).filter(Boolean);
  if (!words.length) return ["-"];
  const lines = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width <= maxWidth || !line) {
      line = next;
    } else {
      lines.push(line);
      line = word;
      if (lines.length >= maxLines - 1) break;
    }
  }
  if (line && lines.length < maxLines) lines.push(line);
  const remaining = words.slice(lines.join(" ").split(/\s+/).filter(Boolean).length).join(" ");
  if (remaining && lines.length) lines[lines.length - 1] = clipText(ctx, `${lines[lines.length - 1]} ${remaining}`, maxWidth);
  return lines;
}

function drawCell(ctx, box, value, muted = false) {
  drawRoundRect(ctx, box.x, box.y, box.w, box.h, 4);
  ctx.fillStyle = muted ? "rgba(32,40,51,0.55)" : COLORS.cell;
  ctx.fill();
  ctx.strokeStyle = COLORS.cellBorder;
  ctx.stroke();
  ctx.fillStyle = muted ? COLORS.dim : COLORS.text;
  ctx.font = "12px Arial";
  ctx.textBaseline = "middle";
  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x + 6, box.y, box.w - 12, box.h);
  ctx.clip();
  const lines = getCellLines(ctx, value, box.w - 16, box.h >= 30 ? 2 : 1);
  const lineHeight = 12;
  const startY = box.y + box.h / 2 - ((lines.length - 1) * lineHeight) / 2;
  lines.forEach((line, index) => {
    ctx.fillText(line, box.x + 8, startY + index * lineHeight + 0.5);
  });
  ctx.restore();
}

function openMenu(event, values, currentValue, callback) {
  if (globalThis.LiteGraph?.ContextMenu) {
    new globalThis.LiteGraph.ContextMenu(values, { event, title: currentValue, callback: (value) => callback(String(value)) });
    return;
  }
  const index = values.indexOf(currentValue);
  callback(values[(index + 1) % values.length] ?? values[0]);
}

function openTextEditor(title, value, callback) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.45); display: flex; align-items: center; justify-content: center;";
  const panel = document.createElement("div");
  panel.style.cssText = "width: min(720px, calc(100vw - 48px)); background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 14px; color: #e5e7eb; box-shadow: 0 20px 60px rgba(0,0,0,.45);";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font: 700 14px Arial; margin-bottom: 10px;";
  const textarea = document.createElement("textarea");
  textarea.value = value || "";
  textarea.style.cssText = "width: 100%; min-height: 260px; resize: vertical; box-sizing: border-box; background: #0b1118; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 10px; font: 12px monospace;";
  const buttons = document.createElement("div");
  buttons.style.cssText = "display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px;";
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel";
  const save = document.createElement("button");
  save.textContent = "Save";
  for (const button of [cancel, save]) {
    button.style.cssText = "background: #243044; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 8px 12px; cursor: pointer;";
  }
  cancel.onclick = () => overlay.remove();
  save.onclick = () => {
    callback(textarea.value);
    overlay.remove();
  };
  buttons.append(cancel, save);
  panel.append(heading, textarea, buttons);
  overlay.append(panel);
  document.body.appendChild(overlay);
  textarea.focus();
}

function openSingleLineEditor(title, value, callback) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.45); display: flex; align-items: center; justify-content: center;";
  const panel = document.createElement("div");
  panel.style.cssText = "width: min(460px, calc(100vw - 48px)); background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 14px; color: #e5e7eb; box-shadow: 0 20px 60px rgba(0,0,0,.45);";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font: 700 14px Arial; margin-bottom: 10px;";
  const input = document.createElement("input");
  input.value = value || "";
  input.style.cssText = "width: 100%; box-sizing: border-box; background: #0b1118; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 10px; font: 12px Arial;";
  const buttons = document.createElement("div");
  buttons.style.cssText = "display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px;";
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel";
  const save = document.createElement("button");
  save.textContent = "Save";
  for (const button of [cancel, save]) {
    button.style.cssText = "background: #243044; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 8px 12px; cursor: pointer;";
  }
  cancel.onclick = () => overlay.remove();
  save.onclick = () => {
    callback(input.value);
    overlay.remove();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") save.click();
    if (event.key === "Escape") cancel.click();
  });
  buttons.append(cancel, save);
  panel.append(heading, input, buttons);
  overlay.append(panel);
  document.body.appendChild(overlay);
  input.focus();
  input.select();
}

function openHelpModal() {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.45); display: flex; align-items: center; justify-content: center;";
  const panel = document.createElement("div");
  panel.style.cssText = "width: min(620px, calc(100vw - 48px)); background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 14px; color: #e5e7eb; box-shadow: 0 20px 60px rgba(0,0,0,.45);";
  const heading = document.createElement("div");
  heading.textContent = "Prompt Detail Lists Help";
  heading.style.cssText = "font: 700 15px Arial; margin-bottom: 10px;";
  const content = document.createElement("div");
  content.style.cssText = "font: 12px Arial; color: #cbd5e1; line-height: 1.5;";
  content.innerHTML = `
    <p><b>Preset</b> picks a starter category. Presets fill the list, but you can edit the list afterward.</p>
    <p><b>Label</b> is the name shown in the output, such as Camera Motion or Emotion.</p>
    <p><b>Mode</b> controls selection. Index follows the scene number. Random picks from the list. Random no repeat shuffles before repeating.</p>
    <p><b>Items</b> is how many entries this list adds to each prompt.</p>
    <p><b>Template</b> appears when Items is more than 1. Use placeholders like {item1} and {item2}.</p>
    <p><b>List</b> opens the list editor. Use one entry per line.</p>
    <p>The combined output always joins active lists with new lines.</p>
  `;
  const close = document.createElement("button");
  close.textContent = "Close";
  close.style.cssText = "background: #243044; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 8px 12px; cursor: pointer; margin-top: 8px;";
  close.onclick = () => overlay.remove();
  panel.append(heading, content, close);
  overlay.append(panel);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) overlay.remove();
  });
  document.body.appendChild(overlay);
}

function createPanelWidget(node) {
  const widget = {
    name: "easy_picker_panel",
    type: "custom",
    value: "",
    boxes: [],
    computeSize() {
      return [780, 96 + Math.max(1, getPickerCount(node)) * 40];
    },
    draw(ctx, drawNode, widgetWidth, y) {
      const rows = getRows(drawNode);
      setWidgetValue(drawNode, "joiner", "newline");
      if ((drawNode.size?.[0] || 0) < MIN_NODE_WIDTH) {
        drawNode.size[0] = MIN_NODE_WIDTH;
      }
      const width = Math.max(MIN_NODE_WIDTH - 20, widgetWidth || drawNode.size?.[0] || MIN_NODE_WIDTH);
      const x = 12;
      const top = y + 6;
      const panelWidth = width - 24;
      const rowHeight = 40;
      widget.boxes = [];

      drawRoundRect(ctx, x, top, panelWidth, 66 + Math.max(1, rows.length) * rowHeight, 6);
      ctx.fillStyle = COLORS.panel;
      ctx.fill();
      ctx.strokeStyle = COLORS.panelBorder;
      ctx.stroke();

      ctx.fillStyle = COLORS.header;
      ctx.font = "700 13px Arial";
      ctx.fillText("Prompt Detail Lists", x + 12, top + 20);
      const helpBox = { x: x + panelWidth - 34, y: top + 9, w: 22, h: 22, row: -1, field: "help" };
      drawCell(ctx, helpBox, "?");
      widget.boxes.push(helpBox);

      const columns = [
        { key: "preset", label: "Preset", w: 160 },
        { key: "label", label: "Label", w: 140 },
        { key: "mode", label: "Mode", w: 112 },
        { key: "pickCount", label: "Items", w: 54 },
        { key: "template", label: "Template", w: 78 },
      ];
      let colX = x + 36;
      ctx.fillStyle = COLORS.dim;
      ctx.font = "11px Arial";
      for (const column of columns) {
        ctx.fillText(column.label, colX + 2, top + 42);
        colX += column.w + 8;
      }

      rows.forEach((row, rowIndex) => {
        const rowY = top + 50 + rowIndex * rowHeight;
        ctx.fillStyle = COLORS.dim;
        ctx.font = "12px Arial";
        ctx.fillText(String(row.index), x + 14, rowY + 20);
        let cellX = x + 36;
        for (const column of columns) {
          const box = { x: cellX, y: rowY, w: column.w, h: 34, row: rowIndex, field: column.key };
          const isTemplate = column.key === "template";
          const value = isTemplate ? (row.pickCount > 1 ? "Template" : "-") : row[column.key];
          drawCell(ctx, box, value, (column.key === "label" && !row.label) || (isTemplate && row.pickCount <= 1));
          widget.boxes.push(box);
          cellX += column.w + 8;
        }
        const editBox = { x: cellX, y: rowY, w: 50, h: 34, row: rowIndex, field: "items" };
        drawCell(ctx, editBox, "List");
        widget.boxes.push(editBox);
        const deleteBox = { x: cellX + 58, y: rowY, w: 24, h: 34, row: rowIndex, field: "delete" };
        drawCell(ctx, deleteBox, "x");
        widget.boxes.push(deleteBox);
      });

      const buttonY = top + 60 + Math.max(1, rows.length) * rowHeight;
      const addBox = { x: x + 12, y: buttonY, w: 96, h: 26, row: -1, field: "add" };
      drawCell(ctx, addBox, "+ Add List");
      widget.boxes.push(addBox);
    },
    mouse(event, pos, drawNode) {
      if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
      const [px, py] = pos;
      const hit = widget.boxes.find((box) => px >= box.x && px <= box.x + box.w && py >= box.y && py <= box.y + box.h);
      if (!hit) return false;

      const rows = getRows(drawNode);
      if (hit.field === "help") {
        openHelpModal();
        return true;
      }
      if (hit.field === "add") {
        const next = rows.length + 1;
        if (next <= MAX_PICKERS) {
          setPickerCount(drawNode, next);
          setPicker(drawNode, next, { preset: "Custom", label: "", items: "", mode: "index", pickCount: 1 });
          requestDraw(drawNode);
        }
        return true;
      }
      const row = rows[hit.row];
      if (!row) return false;
      const index = row.index;
      if (hit.field === "delete") {
        for (let i = index; i < rows.length; i++) setPicker(drawNode, i, getPicker(drawNode, i + 1));
        setPicker(drawNode, rows.length, { preset: "Custom", label: "", items: "", mode: "index", pickCount: 1 });
        setPickerCount(drawNode, Math.max(0, rows.length - 1));
        requestDraw(drawNode);
        return true;
      }
      if (hit.field === "preset") {
        openMenu(event, PRESETS, row.preset, (value) => {
          applyPreset(drawNode, index, value);
          requestDraw(drawNode);
        });
        return true;
      }
      if (hit.field === "mode") {
        openMenu(event, MODES, row.mode, (value) => {
          setWidgetValue(drawNode, `selection_mode_${index}`, value);
          requestDraw(drawNode);
        });
        return true;
      }
      if (hit.field === "pickCount") {
        openSingleLineEditor("Items per prompt", String(row.pickCount), (next) => {
          const count = Math.max(1, Math.min(50, Math.round(Number(next) || 1)));
          setWidgetValue(drawNode, `pick_count_${index}`, count);
          requestDraw(drawNode);
        });
        return true;
      }
      if (hit.field === "template") {
        if (row.pickCount <= 1) return true;
        openSingleLineEditor("Item template", row.template, (next) => {
          setWidgetValue(drawNode, `two_item_template_${index}`, next || "start with {item1} then follow with {item2}");
          requestDraw(drawNode);
        });
        return true;
      }
      if (hit.field === "label") {
        openSingleLineEditor("Label", row.label || (row.preset !== "Custom" ? row.preset : ""), (next) => {
          setWidgetValue(drawNode, `label_${index}`, next);
          requestDraw(drawNode);
        });
        return true;
      }
      if (hit.field === "items") {
        openTextEditor(`List ${index}: ${row.label || row.preset}`, row.items, (value) => {
          setWidgetValue(drawNode, `items_${index}`, value);
          requestDraw(drawNode);
        });
        return true;
      }
      return false;
    },
  };
  return widget;
}

function ensurePanelWidget(node) {
  if ((node.widgets || []).some((widget) => widget.name === "easy_picker_panel")) return;
  const panel = createPanelWidget(node);
  if (typeof node.addCustomWidget === "function") node.addCustomWidget(panel);
  else {
    node.widgets = node.widgets || [];
    node.widgets.unshift(panel);
  }
}

function bindCount(node) {
  const countWidget = getWidget(node, "picker_count");
  if (!countWidget || countWidget.__vrgdgEasyPickerBound) return;
  const oldCallback = countWidget.callback;
  countWidget.callback = function () {
    oldCallback?.apply(this, arguments);
    requestDraw(node);
  };
  countWidget.__vrgdgEasyPickerBound = true;
}

function setupNode(node, initializeDefault = false) {
  node.serialize_widgets = true;
  setWidgetValue(node, "joiner", "newline");
  if (initializeDefault && getPickerCount(node) === 0) {
    setPickerCount(node, 1);
    applyPreset(node, 1, "Camera Motion");
  }
  hideStorageWidgets(node);
  bindCount(node);
  ensurePanelWidget(node);
  requestAnimationFrame(() => requestDraw(node));
}

app.registerExtension({
  name: "vrgdg.easy_multi_cycling_text_picker.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      setupNode(this, true);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = origOnConfigure?.apply(this, arguments);
      setupNode(this, false);
      return result;
    };
  },
});
