import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_SetGroupStateMulti";
const MAX_GROUP_SLOTS = 12;
const NONE_OPTION = "<none>";
const ACTION_VALUES = ["active", "mute", "bypass"];
const PRESET_CONFIGS = {
  VRGDG_MuteUnmute4PromptCreatorWF_0: {
    group_count: 3,
    auto_queue_next: true,
    queue_delay_seconds: 5.0,
    groups: [
      { title: "001-Lyrics, Theme, Story creator", action: "active" },
      { title: "002- Prompt Batcher - Text to image", action: "mute" },
      { title: "003 - image to video", action: "mute" },
    ],
  },
  VRGDG_MuteUnmute4PromptCreatorWF_1: {
    group_count: 2,
    auto_queue_next: true,
    queue_delay_seconds: 5.0,
    groups: [
      { title: "002- Prompt Batcher - Text to image", action: "active" },
      { title: "001-Lyrics, Theme, Story creator", action: "mute" },
    ],
  },
  VRGDG_MuteUnmute4PromptCreatorWF_2: {
    group_count: 2,
    auto_queue_next: true,
    queue_delay_seconds: 5.0,
    groups: [
      { title: "003 - image to video", action: "active" },
      { title: "002- Prompt Batcher - Text to image", action: "mute" },
    ],
  },
};

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function getWidgetByLabel(node, label) {
  return (node.widgets || []).find((w) => String(w?.name || w?.label || "") === label);
}

function getSlotPair(node, index) {
  const slots = node.__vrgdgSlotWidgets || {};
  return slots[index] || null;
}

function ensureSlotWidgets(node) {
  if (node.__vrgdgSlotWidgetsInitialized) return;
  node.__vrgdgSlotWidgets = {};

  for (let i = 1; i <= MAX_GROUP_SLOTS; i++) {
    const groupLabel = `Group ${i}`;
    const actionLabel = `Group ${i} Action`;

    const groupWidget =
      getWidgetByLabel(node, groupLabel) ||
      node.addWidget(
        "combo",
        groupLabel,
        NONE_OPTION,
        () => {
          syncHiddenFromSlots(node);
          updateTargets(node);
          app.graph.setDirtyCanvas(true, true);
        },
        { values: [NONE_OPTION], serialize: false }
      );
    const actionWidget =
      getWidgetByLabel(node, actionLabel) ||
      node.addWidget(
        "combo",
        actionLabel,
        "mute",
        () => {
          syncHiddenFromSlots(node);
          updateTargets(node);
          app.graph.setDirtyCanvas(true, true);
        },
        { values: ACTION_VALUES, serialize: false }
      );

    groupWidget.__vrgdgCustom = true;
    actionWidget.__vrgdgCustom = true;
    groupWidget.__vrgdgSlotIndex = i;
    actionWidget.__vrgdgSlotIndex = i;
    groupWidget.serializeValue = () => undefined;
    actionWidget.serializeValue = () => undefined;

    node.__vrgdgSlotWidgets[i] = { groupWidget, actionWidget };
  }

  node.__vrgdgSlotWidgetsInitialized = true;
}

function syncSlotsFromHidden(node) {
  for (let i = 1; i <= MAX_GROUP_SLOTS; i++) {
    const slot = getSlotPair(node, i);
    if (!slot) continue;
    const hiddenGroup = getWidget(node, `group_${i}`);
    const hiddenAction = getWidget(node, `group_${i}_action`);
    if (hiddenGroup && hiddenGroup.value != null) {
      slot.groupWidget.value = String(hiddenGroup.value || NONE_OPTION);
    }
    if (hiddenAction && hiddenAction.value != null) {
      slot.actionWidget.value = String(hiddenAction.value || "mute");
    }
  }
}

function syncHiddenFromSlots(node) {
  for (let i = 1; i <= MAX_GROUP_SLOTS; i++) {
    const slot = getSlotPair(node, i);
    if (!slot) continue;
    const hiddenGroup = getWidget(node, `group_${i}`);
    const hiddenAction = getWidget(node, `group_${i}_action`);
    if (hiddenGroup) hiddenGroup.value = String(slot.groupWidget.value || NONE_OPTION);
    if (hiddenAction) hiddenAction.value = String(slot.actionWidget.value || "mute");
  }
}

function isTargetNodeDef(nodeData) {
  const candidates = [nodeData?.name, nodeData?.display_name, nodeData?.displayName]
    .map((v) => String(v || ""));
  return candidates.some((v) => v === NODE_NAME || v.includes(NODE_NAME) || PRESET_CONFIGS[v]);
}

function isTargetNodeInstance(node) {
  const candidates = [node?.type, node?.comfyClass, node?.constructor?.type, node?.constructor?.title]
    .map((v) => String(v || ""));
  return candidates.some((v) => v === NODE_NAME || v.includes(NODE_NAME) || PRESET_CONFIGS[v]);
}

function getPresetConfig(node) {
  const candidates = [node?.type, node?.comfyClass, node?.constructor?.type, node?.constructor?.title]
    .map((v) => String(v || ""));
  for (const candidate of candidates) {
    if (PRESET_CONFIGS[candidate]) return PRESET_CONFIGS[candidate];
  }
  return null;
}

function applyPresetConfig(node) {
  const preset = getPresetConfig(node);
  if (!preset) return false;

  const countWidget = getWidget(node, "group_count");
  const autoQueueWidget = getWidget(node, "auto_queue_next");
  const delayWidget = getWidget(node, "queue_delay_seconds");

  if (countWidget) countWidget.value = Number(preset.group_count || preset.groups.length || 1);
  if (autoQueueWidget) autoQueueWidget.value = Boolean(preset.auto_queue_next);
  if (delayWidget) delayWidget.value = Number(preset.queue_delay_seconds || 0);

  ensureSlotWidgets(node);
  for (let i = 1; i <= MAX_GROUP_SLOTS; i++) {
    const slot = getSlotPair(node, i);
    if (!slot) continue;
    const group = preset.groups[i - 1];
    slot.groupWidget.value = group ? String(group.title || NONE_OPTION) : NONE_OPTION;
    slot.actionWidget.value = group ? String(group.action || "mute") : "mute";
  }
  syncHiddenFromSlots(node);
  updateTargets(node);
  return true;
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgOriginalType")) {
    widget.__vrgdgOriginalType = widget.type;
    widget.__vrgdgOriginalComputeSize = widget.computeSize;
  }

  if (visible) {
    widget.type = widget.__vrgdgOriginalType;
    if (widget.__vrgdgOriginalComputeSize) {
      widget.computeSize = widget.__vrgdgOriginalComputeSize;
    } else {
      delete widget.computeSize;
    }
  } else {
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
  }
}

function getCurrentGraph(node) {
  return node?.graph || app?.canvas?.getCurrentGraph?.() || app?.graph;
}

function toArrayMaybe(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  try {
    if (typeof value.values === "function") {
      return Array.from(value.values());
    }
  } catch (e) {
    // ignore
  }
  try {
    if (typeof value[Symbol.iterator] === "function") {
      return Array.from(value);
    }
  } catch (e) {
    // ignore
  }
  if (typeof value === "object") {
    try {
      return Object.values(value);
    } catch (e) {
      // ignore
    }
  }
  return [];
}

function collectGroupsFromGraph(graph) {
  if (!graph) return [];

  const groups = [];
  const seen = new Set();
  const pushGroups = (arr, ownerGraph) => {
    if (!Array.isArray(arr)) return;
    for (const g of arr) {
      if (!g) continue;
      const key = `${g.id ?? ""}|${String(g.title ?? "")}|${JSON.stringify(g._bounding ?? g.bounding ?? [])}`;
      if (seen.has(key)) continue;
      seen.add(key);
      // Keep graph ownership so group actions can target subgraph contents correctly.
      try {
        g.__vrgdgOwnerGraph = ownerGraph || g.__vrgdgOwnerGraph || graph;
      } catch (e) {
        // ignore readonly objects
      }
      groups.push(g);
    }
  };

  pushGroups(toArrayMaybe(graph._groups), graph);
  pushGroups(toArrayMaybe(graph.groups), graph);

  const subgraphs = graph.subgraphs?.values?.();
  if (subgraphs) {
    let sg;
    while ((sg = subgraphs.next().value)) {
      pushGroups(toArrayMaybe(sg._groups), sg);
      pushGroups(toArrayMaybe(sg.groups), sg);
    }
  }

  return groups;
}

function collectGroupsFromSerializedGraph(graph) {
  if (!graph || typeof graph.serialize !== "function") return [];
  try {
    const data = graph.serialize();
    const groups = toArrayMaybe(data?.groups);
    return groups
      .map((g) => ({
        id: g?.id,
        title: g?.title ?? g?.name ?? g?.label ?? "",
        _bounding: g?.bounding,
        bounding: g?.bounding,
        __vrgdgOwnerGraph: graph,
        __serialized: true,
      }))
      .filter((g) => String(g.title || "").trim() || Array.isArray(g._bounding));
  } catch (e) {
    return [];
  }
}

function getGroupsSortedAlpha(node) {
  const groups = [];
  const seen = new Set();
  const candidateGraphs = [
    node?.graph,
    app?.canvas?.getCurrentGraph?.(),
    app?.canvas?.graph,
    app?.graph,
  ].filter(Boolean);

  for (const graph of candidateGraphs) {
    for (const g of collectGroupsFromGraph(graph)) {
      const key = `${g?.id ?? ""}|${String(g?.title ?? "")}|${JSON.stringify(g?._bounding ?? g?.bounding ?? [])}`;
      if (seen.has(key)) continue;
      seen.add(key);
      groups.push(g);
    }
    for (const g of collectGroupsFromSerializedGraph(graph)) {
      const key = `${g?.id ?? ""}|${String(g?.title ?? "")}|${JSON.stringify(g?._bounding ?? g?.bounding ?? [])}`;
      if (seen.has(key)) continue;
      seen.add(key);
      groups.push(g);
    }
  }

  groups.sort((a, b) => String(a?.title || "").localeCompare(String(b?.title || "")));
  return groups;
}

function getNodesForGroup(node, group) {
  const graph = group?.__vrgdgOwnerGraph || getCurrentGraph(node);
  if (!graph || !group) return [];

  try {
    if (typeof group.recomputeInsideNodes === "function") {
      group.recomputeInsideNodes();
    }
  } catch (e) {
    // fall through to fallback strategy
  }

  const children = Array.from(group?._children || []).filter((n) => typeof n?.id === "number");
  if (children.length) return children;

  // Fallback for environments where _children is stale.
  const bounds = group?._bounding || group?.bounding;
  if (!Array.isArray(bounds) || bounds.length < 4) return [];
  const [gx, gy, gw, gh] = bounds;

  return (graph._nodes || []).filter((graphNode) => {
    if (typeof graphNode?.id !== "number") return false;
    const pos = graphNode.pos || [0, 0];
    const size = Array.isArray(graphNode.size) ? graphNode.size : [140, 80];
    const centerX = Number(pos[0] || 0) + Number(size[0] || 0) * 0.5;
    const centerY = Number(pos[1] || 0) + Number(size[1] || 0) * 0.5;
    return centerX >= gx && centerX < gx + gw && centerY >= gy && centerY < gy + gh;
  });
}

function updateGroupOptions(node) {
  const groups = getGroupsSortedAlpha(node);
  const seenTitles = new Set();
  const titles = [];
  for (const g of groups) {
    const t = String(g?.title || "").trim();
    if (!t) continue;
    const key = t.toLowerCase();
    if (seenTitles.has(key)) continue;
    seenTitles.add(key);
    titles.push(t);
  }

  const values = [NONE_OPTION, ...titles];

  for (let i = 1; i <= MAX_GROUP_SLOTS; i++) {
    const slot = getSlotPair(node, i);
    if (!slot) continue;

    if (!slot.groupWidget.options) slot.groupWidget.options = {};
    slot.groupWidget.options.values = values;
    if (!values.includes(String(slot.groupWidget.value || ""))) {
      slot.groupWidget.value = NONE_OPTION;
    }

    if (!slot.actionWidget.options) slot.actionWidget.options = {};
    slot.actionWidget.options.values = ACTION_VALUES;
    if (!ACTION_VALUES.includes(String(slot.actionWidget.value || ""))) {
      slot.actionWidget.value = "mute";
    }
  }

  syncHiddenFromSlots(node);
}

function collectSelectedGroupTargets(node) {
  const countWidget = getWidget(node, "group_count");
  const count = Math.max(1, Math.min(MAX_GROUP_SLOTS, Number(countWidget?.value ?? 1)));
  const groups = getGroupsSortedAlpha(node);

  const selected = [];
  for (let i = 1; i <= count; i++) {
    const slot = getSlotPair(node, i);
    if (!slot) continue;
    const title = String(slot.groupWidget.value || "").trim();
    if (!title || title === NONE_OPTION) continue;

    const action = String(slot.actionWidget?.value || "mute").toLowerCase();
    const matched = groups.filter((g) => String(g?.title || "").trim() === title);
    const nodeIds = [];
    const nodeKeys = [];
    for (const group of matched) {
      const ownerGraph = group?.__vrgdgOwnerGraph;
      const ownerGraphId = ownerGraph?.id != null ? String(ownerGraph.id) : null;
      for (const groupNode of getNodesForGroup(node, group)) {
        const nodeId = Number(groupNode?.id);
        if (Number.isInteger(nodeId) && nodeId >= 0 && !nodeIds.includes(nodeId)) {
          nodeIds.push(nodeId);
        }
        if (ownerGraphId != null && Number.isInteger(nodeId) && nodeId >= 0) {
          const key = `${ownerGraphId}:${nodeId}`;
          if (!nodeKeys.includes(key)) nodeKeys.push(key);
        }
      }
    }

    selected.push({ slot: i, title, action, node_ids: nodeIds, node_keys: nodeKeys });
  }
  return selected;
}

function updateTargets(node) {
  const targets = collectSelectedGroupTargets(node);
  const ids = [];
  for (const target of targets) {
    for (const nodeId of target.node_ids || []) {
      if (!ids.includes(nodeId)) {
        ids.push(nodeId);
      }
    }
  }

  const nodeIdsWidget = getWidget(node, "node_ids_csv");
  if (nodeIdsWidget) {
    nodeIdsWidget.value = ids.join(",");
  }

  const targetsWidget = getWidget(node, "group_targets_json");
  if (targetsWidget) {
    targetsWidget.value = JSON.stringify(targets);
  }
}

function refreshWidgets(node) {
  ensureSlotWidgets(node);
  const isPreset = applyPresetConfig(node);
  const countWidget = getWidget(node, "group_count");
  const count = Math.max(1, Math.min(MAX_GROUP_SLOTS, Number(countWidget?.value ?? 1)));

  setWidgetVisible(countWidget, !isPreset);
  setWidgetVisible(getWidget(node, "auto_queue_next"), !isPreset);
  setWidgetVisible(getWidget(node, "queue_delay_seconds"), !isPreset);

  for (let i = 1; i <= MAX_GROUP_SLOTS; i++) {
    const isVisible = i <= count;
    // Hide backend widgets; we render custom combo widgets instead.
    setWidgetVisible(getWidget(node, `group_${i}`), false);
    setWidgetVisible(getWidget(node, `group_${i}_action`), false);

    const slot = getSlotPair(node, i);
    if (!slot) continue;
    setWidgetVisible(slot.groupWidget, isVisible && !isPreset);
    setWidgetVisible(slot.actionWidget, isVisible && !isPreset);

    // Add breathing room after each pair except the last visible one.
    const actionWidget = slot.actionWidget;
    if (actionWidget) {
      const gap = isVisible && !isPreset && i < count ? 8 : 0;
      const baseComputeSize = actionWidget.__vrgdgOriginalComputeSize || actionWidget.computeSize;
      if (isVisible) {
        actionWidget.computeSize = function (...args) {
          const base = baseComputeSize ? baseComputeSize.apply(this, args) : [args?.[0] || 0, 20];
          return [base[0], base[1] + gap];
        };
      }
    }
  }

  // Keep helper widgets hidden; they are filled automatically.
  setWidgetVisible(getWidget(node, "node_ids_csv"), false);
  setWidgetVisible(getWidget(node, "group_targets_json"), false);
  // Legacy global action is hidden in favor of per-group actions.
  setWidgetVisible(getWidget(node, "group_action"), false);

  updateGroupOptions(node);
  if (isPreset) {
    applyPresetConfig(node);
  }
  updateTargets(node);

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCallbacks(node) {
  if (node.__vrgdgSetGroupStateMultiBound) return;
  ensureSlotWidgets(node);
  syncSlotsFromHidden(node);

  const countWidget = getWidget(node, "group_count");
  if (countWidget) {
    const old = countWidget.callback;
    countWidget.callback = function () {
      if (old) old.apply(this, arguments);
      refreshWidgets(node);
    };
  }

  const globalActionWidget = getWidget(node, "group_action");
  if (globalActionWidget) {
    const old = globalActionWidget.callback;
    globalActionWidget.callback = function () {
      if (old) old.apply(this, arguments);
      updateTargets(node);
      app.graph.setDirtyCanvas(true, true);
    };
  }

  node.__vrgdgSetGroupStateMultiBound = true;
}

function collectAllGraphs() {
  const out = [];
  const seen = new Set();

  const pushGraph = (g) => {
    if (!g) return;
    const key = String(g.id ?? "");
    const dedupeKey = `${key}|${g.constructor?.name || "graph"}`;
    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    out.push(g);

    const subgraphs = g.subgraphs?.values?.();
    if (subgraphs) {
      let sg;
      while ((sg = subgraphs.next().value)) {
        pushGraph(sg);
      }
    }
  };

  pushGraph(app?.canvas?.getCurrentGraph?.());
  pushGraph(app?.canvas?.graph);
  pushGraph(app?.graph);
  return out;
}

function setModeRecursive(node, mode) {
  if (!node) return;
  node.mode = mode;
  const sub = node.subgraph;
  if (sub?.nodes?.length) {
    for (const child of sub.nodes) {
      setModeRecursive(child, mode);
    }
  }
}

function actionToMode(action) {
  const a = String(action || "mute").toLowerCase();
  if (a === "active") return 0;
  if (a === "bypass") return 4;
  return 2;
}

async function applyNodeModesEvent(event) {
  const targets = Array.isArray(event?.detail?.targets) ? event.detail.targets : [];
  if (!targets.length) return;

  const graphs = collectAllGraphs();
  const graphById = new Map(graphs.map((g) => [String(g.id ?? ""), g]));
  const pseudoNode = { graph: app?.canvas?.getCurrentGraph?.() || app?.graph };
  const sortedGroups = getGroupsSortedAlpha(pseudoNode);
  const backendTargets = [];

  for (const target of targets) {
    if (!target || typeof target !== "object") continue;
    const mode = actionToMode(target.action);

    let nodeKeys = Array.isArray(target.node_keys) ? target.node_keys : [];
    let nodeIds = Array.isArray(target.node_ids) ? target.node_ids : [];
    const slot = Number(target.slot);
    const title = String(target.title || "").trim();

    for (const key of nodeKeys) {
      const text = String(key || "");
      const sep = text.indexOf(":");
      if (sep < 0) continue;
      const graphId = text.slice(0, sep);
      const nodeId = Number(text.slice(sep + 1));
      if (!Number.isInteger(nodeId)) continue;

      const graph = graphById.get(graphId);
      if (!graph) continue;
      const n = graph.getNodeById ? graph.getNodeById(nodeId) : graph?._nodes_by_id?.[nodeId];
      if (n) setModeRecursive(n, mode);
    }

    // Fallback: apply by id across all graphs when graph key is unavailable.
    for (const rawId of nodeIds) {
      const nodeId = Number(rawId);
      if (!Number.isInteger(nodeId)) continue;
      for (const graph of graphs) {
        const n = graph.getNodeById ? graph.getNodeById(nodeId) : graph?._nodes_by_id?.[nodeId];
        if (n) setModeRecursive(n, mode);
      }
    }

    // Preset fallback: resolve a group by slot first, then by exact title.
    if (!nodeKeys.length && !nodeIds.length && (Number.isInteger(slot) || title)) {
      let matchedGroups = [];
      if (Number.isInteger(slot) && slot > 0 && slot <= sortedGroups.length) {
        matchedGroups = [sortedGroups[slot - 1]];
      }
      if (!matchedGroups.length && title) {
        matchedGroups = sortedGroups.filter((g) => String(g?.title || "").trim() === title);
      }
      const resolved = [];
      for (const group of matchedGroups) {
        for (const groupNode of getNodesForGroup(pseudoNode, group)) {
          const nodeId = Number(groupNode?.id);
          if (Number.isInteger(nodeId) && nodeId >= 0 && !resolved.includes(nodeId)) {
            resolved.push(nodeId);
          }
        }
      }
      nodeIds = resolved;
      for (const rawId of nodeIds) {
        const nodeId = Number(rawId);
        if (!Number.isInteger(nodeId)) continue;
        for (const graph of graphs) {
          const n = graph.getNodeById ? graph.getNodeById(nodeId) : graph?._nodes_by_id?.[nodeId];
          if (n) setModeRecursive(n, mode);
        }
      }
    }

    if (nodeIds.length) {
      backendTargets.push({
        action: target.action,
        node_ids: nodeIds.filter((nodeId) => Number.isInteger(Number(nodeId))).map((nodeId) => Number(nodeId)),
      });
    }
  }

  app.graph?.setDirtyCanvas?.(true, true);
  if (backendTargets.length) {
    try {
      await api.fetchApi("/vrgdg/apply_node_modes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targets: backendTargets }),
      });
    } catch (e) {
      console.warn("[VRGDG] Failed to apply backend node modes", e);
    }
  }
}

app.registerExtension({
  name: "vrgdg.set_group_state_multi.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isTargetNodeDef(nodeData)) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;
    const origOnSerialize = nodeType.prototype.onSerialize;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindCallbacks(this);
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Groups")) {
        this.addWidget("button", "Refresh Groups", null, () => refreshWidgets(this));
      }
      // Refresh a few times because groups may not be ready on the first UI tick.
      setTimeout(() => refreshWidgets(this), 0);
      setTimeout(() => refreshWidgets(this), 100);
      setTimeout(() => refreshWidgets(this), 400);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCallbacks(this);
      refreshWidgets(this);
      setTimeout(() => refreshWidgets(this), 100);
      return r;
    };

    nodeType.prototype.onSerialize = function () {
      // Ensure backend (hidden) widgets hold latest custom UI values before save.
      try {
        syncHiddenFromSlots(this);
      } catch (e) {
        // ignore
      }
      return origOnSerialize?.apply(this, arguments);
    };
  },

  loadedGraphNode(node) {
    if (!isTargetNodeInstance(node)) return;
    bindCallbacks(node);
    setTimeout(() => refreshWidgets(node), 0);
    setTimeout(() => refreshWidgets(node), 100);
  },
});

if (!window.__vrgdgApplyNodeModesBound) {
  api.addEventListener("vrgdg-apply-node-modes", applyNodeModesEvent);
  window.__vrgdgApplyNodeModesBound = true;
}
