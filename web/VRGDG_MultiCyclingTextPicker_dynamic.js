import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_MultiCyclingTextPicker";
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
const PICKER_FIELDS = [
  "preset",
  "items",
  "label",
  "selection_mode",
  "two_item_template",
  "pick_count",
];

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function migrateWidgetValues(values) {
  if (!Array.isArray(values)) return values;

  const compactCount = 2 + MAX_PICKERS * 6;
  const currentCount = 2 + MAX_PICKERS * 10;
  const oldCount = 2 + MAX_PICKERS * 12;
  if (values.length !== currentCount && values.length !== oldCount) return values;

  const migrated = new Array(compactCount).fill("");
  migrated[0] = values[0] ?? 0;
  migrated[1] = values[1] ?? "newline";

  for (let i = 1; i <= MAX_PICKERS; i++) {
    const compactBase = 2 + (i - 1) * 6;
    if (values.length === currentCount) {
      const sourceBase = 2 + (i - 1) * 10;
      migrated[compactBase] = values[sourceBase] ?? (i === 1 ? "Camera Motion" : "Custom");
      migrated[compactBase + 1] = values[sourceBase + 1] ?? "";
      migrated[compactBase + 2] = values[sourceBase + 2] ?? "";
      migrated[compactBase + 3] = values[sourceBase + 5] ?? "index";
      migrated[compactBase + 4] = values[sourceBase + 7] ?? "start with {item1} then follow with {item2}";
      migrated[compactBase + 5] = values[sourceBase + 9] ?? 1;
      continue;
    }

    const sourceBase = 2 + (i - 1) * 12;
    migrated[compactBase] = values[sourceBase] ?? (i === 1 ? "Camera Motion" : "Custom");
    migrated[compactBase + 1] = values[sourceBase + 2] ?? "";
    migrated[compactBase + 2] = values[sourceBase + 3] ?? "";
    migrated[compactBase + 3] = values[sourceBase + 6] ?? "index";
    migrated[compactBase + 4] = values[sourceBase + 9] ?? "start with {item1} then follow with {item2}";
    migrated[compactBase + 5] = values[sourceBase + 11] ?? 1;
  }

  return migrated;
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgOriginalType")) {
    widget.__vrgdgOriginalType = widget.type;
    widget.__vrgdgOriginalComputeSize = widget.computeSize;
  }

  widget.hidden = !visible;
  widget.serialize = true;

  if (visible) {
    widget.type = widget.__vrgdgOriginalType;
    if (widget.__vrgdgOriginalComputeSize) {
      widget.computeSize = widget.__vrgdgOriginalComputeSize;
    } else {
      delete widget.computeSize;
    }
  } else {
    widget.type = widget.__vrgdgOriginalType;
    widget.computeSize = () => [0, -4];
  }
}

function getPickerCount(node) {
  const countWidget = getWidget(node, "picker_count");
  const rawCount = Number(countWidget?.value ?? 0);
  const count = Number.isFinite(rawCount) ? rawCount : 0;
  return Math.max(0, Math.min(MAX_PICKERS, Math.round(count)));
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
  for (let i = 0; i < BASE_OUTPUTS.length; i++) {
    ensureOutput(node, i, BASE_OUTPUTS[i].name, BASE_OUTPUTS[i].type);
  }
  for (let i = 1; i <= count; i++) {
    ensureOutput(node, BASE_OUTPUTS.length + i - 1, `formatted_text_${i}`, "STRING");
  }

  for (let i = (node.outputs || []).length - 1; i >= BASE_OUTPUTS.length + count; i--) {
    node.removeOutput(i);
  }
}

function getOwningSubgraph(node) {
  const graph = node?.graph;
  if (!graph) return null;
  if (Array.isArray(graph.inputs) && graph.inputNode) return graph;
  if (Array.isArray(graph.inputs) && String(graph.id || "").includes("-")) return graph;
  return null;
}

function getSubgraphLinksArray(subgraph) {
  if (Array.isArray(subgraph?.links)) return subgraph.links;
  if (Array.isArray(subgraph?._links)) return subgraph._links;
  return null;
}

function findSubgraphInputSlot(subgraph, names) {
  const accepted = new Set(names.map((name) => String(name).trim().toLowerCase()));
  const inputs = Array.isArray(subgraph?.inputs) ? subgraph.inputs : [];
  return inputs.findIndex((input) => {
    const name = String(input?.name || "").trim().toLowerCase();
    const label = String(input?.label || input?.localized_name || "").trim().toLowerCase();
    return accepted.has(name) || accepted.has(label);
  });
}

function getSubgraphNodes(subgraph) {
  if (!subgraph) return [];
  if (Array.isArray(subgraph._nodes)) return subgraph._nodes;
  if (Array.isArray(subgraph.nodes)) return subgraph.nodes;
  if (typeof subgraph._nodes_by_id === "object" && subgraph._nodes_by_id) return Object.values(subgraph._nodes_by_id);
  return [];
}

function findSubgraphSourceNode(subgraph, names, type = "INT") {
  const accepted = new Set(names.map((name) => String(name).trim().toLowerCase()));
  return getSubgraphNodes(subgraph).find((node) => {
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

function nextSubgraphLinkId(subgraph, links) {
  const current = Number(subgraph?.state?.lastLinkId || 0);
  const maxExisting = links.reduce((maxId, link) => Math.max(maxId, Number(link?.id || 0)), 0);
  const next = Math.max(current, maxExisting) + 1;
  subgraph.state = subgraph.state || {};
  subgraph.state.lastLinkId = next;
  return next;
}

function removeSubgraphTargetLink(subgraph, links, targetNode, targetSlot) {
  const input = targetNode?.inputs?.[targetSlot];
  const linkId = input?.link;
  if (linkId === null || linkId === undefined) return;

  const index = links.findIndex((link) => Number(link?.id) === Number(linkId));
  if (index >= 0) {
    const [oldLink] = links.splice(index, 1);
    const originInput = Array.isArray(subgraph?.inputs) ? subgraph.inputs[oldLink.origin_slot] : null;
    if (Array.isArray(originInput?.linkIds)) {
      originInput.linkIds = originInput.linkIds.filter((id) => Number(id) !== Number(linkId));
    }
  }
  input.link = null;
}

function ensureSubgraphInputLink(subgraph, sourceSlot, targetNode, targetInputName, type) {
  const links = getSubgraphLinksArray(subgraph);
  const targetSlot = (targetNode?.inputs || []).findIndex((input) => input?.name === targetInputName);
  if (!links || sourceSlot < 0 || targetSlot < 0) return false;

  const existing = links.find((link) =>
    Number(link?.origin_id) === -10 &&
    Number(link?.origin_slot) === Number(sourceSlot) &&
    Number(link?.target_id) === Number(targetNode.id) &&
    Number(link?.target_slot) === Number(targetSlot)
  );
  if (existing) {
    targetNode.inputs[targetSlot].link = existing.id;
    return false;
  }

  removeSubgraphTargetLink(subgraph, links, targetNode, targetSlot);
  const linkId = nextSubgraphLinkId(subgraph, links);
  links.push({
    id: linkId,
    origin_id: -10,
    origin_slot: sourceSlot,
    target_id: targetNode.id,
    target_slot: targetSlot,
    type,
  });

  targetNode.inputs[targetSlot].link = linkId;
  const sourceInput = Array.isArray(subgraph?.inputs) ? subgraph.inputs[sourceSlot] : null;
  if (sourceInput) {
    sourceInput.linkIds = Array.isArray(sourceInput.linkIds) ? sourceInput.linkIds : [];
    if (!sourceInput.linkIds.some((id) => Number(id) === Number(linkId))) {
      sourceInput.linkIds.push(linkId);
    }
  }
  return true;
}

function ensureSubgraphNodeLink(subgraph, sourceNode, sourceSlot, targetNode, targetInputName, type) {
  const targetSlot = (targetNode?.inputs || []).findIndex((input) => input?.name === targetInputName);
  if (!sourceNode || sourceSlot < 0 || targetSlot < 0) return false;

  if (typeof sourceNode.connect === "function") {
    const oldLinkId = targetNode.inputs?.[targetSlot]?.link;
    const result = sourceNode.connect(sourceSlot, targetNode, targetSlot);
    const newLinkId = targetNode.inputs?.[targetSlot]?.link;
    if (newLinkId !== null && newLinkId !== undefined) {
      return Number(oldLinkId) !== Number(newLinkId);
    }
    if (result) return true;
  }
  return false;
}

function ensureInputSocket(node, name, type) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  if (inputs.some((input) => input?.name === name)) return true;

  if (typeof node?.addInput === "function") {
    node.addInput(name, type);
    return true;
  }

  if (node) {
    node.inputs = inputs;
    node.inputs.push({
      localized_name: name,
      name,
      type,
      widget: { name },
      link: null,
    });
    return true;
  }

  return false;
}

function refreshIndexSeedInputs(node, count) {
  const socketCount = Math.max(1, count);
  const keepNames = new Set();
  for (let i = 1; i <= socketCount; i++) {
    keepNames.add(`index_${i}`);
    keepNames.add(`seed_${i}`);
  }

  node.inputs = (node.inputs || []).filter((input) => {
    if (!/^index_\d+$|^seed_\d+$/.test(String(input?.name || ""))) return true;
    return keepNames.has(input.name);
  });

  for (let i = 1; i <= socketCount; i++) {
    ensureInputSocket(node, `index_${i}`, "INT");
    ensureInputSocket(node, `seed_${i}`, "INT");
  }
}

function autoWireIndexSeed(node) {
  const subgraph = getOwningSubgraph(node);
  if (!subgraph) return;

  const count = getPickerCount(node);
  const indexNode = findSubgraphSourceNode(subgraph, ["index", "get_index"], "INT");
  const seedNode = findSubgraphSourceNode(subgraph, ["random seed", "seed", "random_seed"], "INT");
  const indexNodeSlot = findFirstOutputSlot(indexNode, "INT");
  const seedNodeSlot = findFirstOutputSlot(seedNode, "INT");

  for (let i = 1; i <= count; i++) {
    if (indexNode) ensureSubgraphNodeLink(subgraph, indexNode, indexNodeSlot, node, `index_${i}`, "INT");
    if (seedNode) ensureSubgraphNodeLink(subgraph, seedNode, seedNodeSlot, node, `seed_${i}`, "INT");
  }
}

function refreshWidgets(node) {
  const count = getPickerCount(node);
  for (let i = 1; i <= MAX_PICKERS; i++) {
    const visible = i <= count;
    for (const field of PICKER_FIELDS) {
      setWidgetVisible(getWidget(node, `${field}_${i}`), visible);
    }
  }

  refreshOutputs(node);
  refreshIndexSeedInputs(node, count);
  syncBlankLabels(node);
  autoWireIndexSeed(node);
  node.setSize([Math.max(430, node.size?.[0] || 430), node.computeSize()[1]]);
  app.graph?.setDirtyCanvas?.(true, true);
}

function applyPreset(node, index) {
  const presetWidget = getWidget(node, `preset_${index}`);
  const labelWidget = getWidget(node, `label_${index}`);
  const itemsWidget = getWidget(node, `items_${index}`);
  const preset = String(presetWidget?.value || "Custom");
  if (labelWidget) labelWidget.value = preset === "Custom" ? "" : preset;
  if (itemsWidget) itemsWidget.value = PRESET_ITEMS[preset] || "";
  app.graph?.setDirtyCanvas?.(true, true);
}

function syncBlankLabels(node) {
  const count = getPickerCount(node);
  for (let i = 1; i <= count; i++) {
    const presetWidget = getWidget(node, `preset_${i}`);
    const labelWidget = getWidget(node, `label_${i}`);
    const itemsWidget = getWidget(node, `items_${i}`);
    if (!labelWidget || String(labelWidget.value || "").trim()) continue;

    const preset = String(presetWidget?.value || "Custom").trim();
    if (preset && preset !== "Custom") {
      labelWidget.value = preset;
      continue;
    }

    const items = String(itemsWidget?.value || "").trim();
    for (const [presetName, presetItems] of Object.entries(PRESET_ITEMS)) {
      if (presetName !== "Custom" && items === String(presetItems || "").trim()) {
        labelWidget.value = presetName;
        break;
      }
    }
  }
}

function bindPickerCount(node) {
  const countWidget = getWidget(node, "picker_count");
  if (countWidget && !countWidget.__vrgdgMultiCyclingTextPickerBound) {
    const oldCallback = countWidget.callback;
    countWidget.callback = function () {
      oldCallback?.apply(this, arguments);
      refreshWidgets(node);
    };

    countWidget.__vrgdgMultiCyclingTextPickerBound = true;
  }

  for (let i = 1; i <= MAX_PICKERS; i++) {
    const presetWidget = getWidget(node, `preset_${i}`);
    if (!presetWidget || presetWidget.__vrgdgMultiCyclingTextPickerPresetBound) continue;

    const oldPresetCallback = presetWidget.callback;
    presetWidget.callback = function () {
      oldPresetCallback?.apply(this, arguments);
      applyPreset(node, i);
    };
    presetWidget.__vrgdgMultiCyclingTextPickerPresetBound = true;
  }
}

app.registerExtension({
  name: "vrgdg.multi_cycling_text_picker.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      bindPickerCount(this);
      setTimeout(() => refreshWidgets(this), 0);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      if (arguments[0]?.widgets_values) {
        arguments[0].widgets_values = migrateWidgetValues(arguments[0].widgets_values);
      }
      const result = origOnConfigure?.apply(this, arguments);
      this.widgets_values = migrateWidgetValues(this.widgets_values);
      bindPickerCount(this);
      refreshWidgets(this);
      return result;
    };
  },
});
