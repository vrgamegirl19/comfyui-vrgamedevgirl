import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_UpdateLatestCombinedJsonPrompts";
const EMPTY_OPTION = "<no files found>";
const MAX_SLOTS = 20;

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
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

function setWidgetGroupSpacing(widget) {
  if (!widget || !Object.prototype.hasOwnProperty.call(widget, "__vrgdgOriginalComputeSize")) return;
  const original = widget.__vrgdgOriginalComputeSize;
  if (!original || widget.type === "hidden") return;

  // Add a small gap after each prompt group for readability.
  widget.computeSize = function (...args) {
    const size = original.apply(this, args);
    return [size[0], size[1] + 10];
  };
}

function refreshInputVisibility(node) {
  const countWidget = getWidget(node, "prompt_count");
  const batchTypeWidget = getWidget(node, "batch_type");
  if (!countWidget) return;

  const count = Math.max(0, Math.min(MAX_SLOTS, Number(countWidget.value ?? 0)));
  const isText2Image = String(batchTypeWidget?.value || "Text2Image") === "Text2Image";
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const visible = i <= count;
    setWidgetVisible(getWidget(node, `prompt_number_${i}`), visible);
    setWidgetVisible(getWidget(node, `prompt_text_${i}`), visible);
    const imageWidget = getWidget(node, `prompt_image_index_${i}`);
    setWidgetVisible(imageWidget, visible && isText2Image);
    if (visible && isText2Image) {
      setWidgetGroupSpacing(imageWidget);
    }
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function clearPromptInputs(node) {
  const countWidget = getWidget(node, "prompt_count");
  if (countWidget) countWidget.value = 0;
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const numWidget = getWidget(node, `prompt_number_${i}`);
    const textWidget = getWidget(node, `prompt_text_${i}`);
    const imageIndexWidget = getWidget(node, `prompt_image_index_${i}`);
    if (numWidget) numWidget.value = i;
    if (textWidget) textWidget.value = "";
    if (imageIndexWidget) imageIndexWidget.value = "";
  }
}

function formatImageIndex(value) {
  if (!Array.isArray(value)) return "";
  return value
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .map((v) => String(Math.trunc(v)))
    .join(",");
}

async function refreshPromptValues(node) {
  const batchTypeWidget = getWidget(node, "batch_type");
  const fileWidget = getWidget(node, "combined_json_file");
  const countWidget = getWidget(node, "prompt_count");
  const batchType = String(batchTypeWidget?.value || "Text2Image");
  const selectedFile = String(fileWidget?.value || "");
  if (!selectedFile || selectedFile === EMPTY_OPTION) {
    clearPromptInputs(node);
    refreshInputVisibility(node);
    return;
  }

  const bt = encodeURIComponent(batchType);
  const file = encodeURIComponent(selectedFile);
  try {
    const res = await api.fetchApi(
      `/vrgdg/llm_batches/combined_file_prompt_values?batch_type=${bt}&combined_json_file=${file}`,
      { cache: "no-store" }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data?.ok) throw new Error(String(data?.error || `HTTP ${res.status}`));

    const prompts = Array.isArray(data.prompts) ? data.prompts : [];
    const promptByNumber = new Map(
      prompts
        .map((row) => [Number(row?.prompt_number), row])
        .filter(([n]) => Number.isFinite(n) && n > 0)
    );
    const selectedCount = Math.max(
      0,
      Math.min(MAX_SLOTS, Number(countWidget?.value ?? 0))
    );

    for (let i = 1; i <= selectedCount; i++) {
      const numWidget = getWidget(node, `prompt_number_${i}`);
      const textWidget = getWidget(node, `prompt_text_${i}`);
      const imageIndexWidget = getWidget(node, `prompt_image_index_${i}`);
      const promptNumber = Number(numWidget?.value ?? i);
      const row = promptByNumber.get(promptNumber) || null;

      if (textWidget) textWidget.value = row ? String(row.prompt ?? "") : "";
      if (imageIndexWidget) {
        imageIndexWidget.value = row ? formatImageIndex(row.image_index) : "";
      }
    }
  } catch (e) {
    clearPromptInputs(node);
  }

  refreshInputVisibility(node);
}

async function refreshFiles(node, keepSelection = true, loadPromptValues = true) {
  const typeWidget = getWidget(node, "batch_type");
  const fileWidget = getWidget(node, "combined_json_file");
  if (!typeWidget || !fileWidget) return;

  const batchType = encodeURIComponent(String(typeWidget.value || "Text2Image"));
  let options = [EMPTY_OPTION];
  try {
    const res = await api.fetchApi(`/vrgdg/llm_batches/combined_files?batch_type=${batchType}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.files) && data.files.length) {
      options = data.files.map((v) => String(v));
    }
  } catch (e) {
    options = [EMPTY_OPTION];
  }

  const current = String(fileWidget.value || "");
  fileWidget.options = fileWidget.options || {};
  fileWidget.options.values = [...options];

  if (keepSelection && options.includes(current)) {
    node.__vrgdgSkipNextFileRefresh = true;
    fileWidget.value = current;
  } else {
    node.__vrgdgSkipNextFileRefresh = true;
    fileWidget.value = options[0];
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
  if (loadPromptValues) {
    await refreshPromptValues(node);
  } else {
    clearPromptInputs(node);
    refreshInputVisibility(node);
  }
}

function collectUpdates(node) {
  const countWidget = getWidget(node, "prompt_count");
  const batchTypeWidget = getWidget(node, "batch_type");
  const isText2Image = String(batchTypeWidget?.value || "Text2Image") === "Text2Image";
  const count = Math.max(0, Math.min(MAX_SLOTS, Number(countWidget?.value ?? 0)));
  const updates = [];

  for (let i = 1; i <= count; i++) {
    const numWidget = getWidget(node, `prompt_number_${i}`);
    const textWidget = getWidget(node, `prompt_text_${i}`);
    const imageIndexWidget = getWidget(node, `prompt_image_index_${i}`);
    if (!numWidget || !textWidget) continue;

    const n = Number(numWidget.value ?? i);
    const prompt = String(textWidget.value ?? "");
    if (!Number.isFinite(n) || n <= 0 || !prompt.trim()) continue;

    updates.push({
      prompt_number: Math.trunc(n),
      prompt,
      ...(isText2Image ? { image_index: String(imageIndexWidget?.value ?? "") } : {}),
    });
  }

  return updates;
}

async function updateText(node) {
  const remakeModeWidget = getWidget(node, "remake_mode");
  const batchTypeWidget = getWidget(node, "batch_type");
  const fileWidget = getWidget(node, "combined_json_file");

  const remakeMode = Boolean(remakeModeWidget?.value);
  if (!remakeMode) {
    alert("[VRGDG] remake_mode is false. Enable it to apply updates.");
    return;
  }

  const selectedFile = String(fileWidget?.value || "");
  if (!selectedFile || selectedFile === EMPTY_OPTION) {
    alert("[VRGDG] Select a combined JSON file first.");
    return;
  }

  const updates = collectUpdates(node);
  if (!updates.length) {
    alert("[VRGDG] Provide at least one prompt number + prompt text.");
    return;
  }

  const payload = {
    remake_mode: remakeMode,
    batch_type: String(batchTypeWidget?.value || "Text2Image"),
    combined_json_file: selectedFile,
    updates,
  };

  try {
    const res = await api.fetchApi("/vrgdg/llm_batches/combined_file_update_prompts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      const err = data?.error ? String(data.error) : `HTTP ${res.status}`;
      alert(`[VRGDG] Update failed: ${err}`);
      return;
    }

    alert(
      `[VRGDG] Updated ${Number(data.updated || 0)} prompt(s).\n${String(data.file_path || "")}`
    );
  } catch (e) {
    alert(`[VRGDG] Update failed: ${String(e)}`);
  }
}

function bindCallbacks(node) {
  if (node.__vrgdgCombinedPromptEditorBound) return;

  const countWidget = getWidget(node, "prompt_count");
  if (countWidget) {
    const oldCount = countWidget.callback;
    countWidget.callback = function () {
      if (oldCount) oldCount.apply(this, arguments);
      refreshInputVisibility(node);
    };
  }

  const typeWidget = getWidget(node, "batch_type");
  if (typeWidget) {
    const oldType = typeWidget.callback;
    typeWidget.callback = function () {
      if (oldType) oldType.apply(this, arguments);
      refreshFiles(node, false, false);
    };
  }

  const fileWidget = getWidget(node, "combined_json_file");
  if (fileWidget) {
    const oldFile = fileWidget.callback;
    fileWidget.callback = function () {
      if (oldFile) oldFile.apply(this, arguments);
      if (node.__vrgdgSkipNextFileRefresh) {
        node.__vrgdgSkipNextFileRefresh = false;
        return;
      }
      refreshPromptValues(node);
    };
  }

  node.__vrgdgCombinedPromptEditorBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);

      bindCallbacks(this);
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Load Current Prompts")) {
        this.addWidget("button", "Load Current Prompts", null, () => {
          refreshPromptValues(this);
        });
      }
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Update Text")) {
        this.addWidget("button", "Update Text", null, () => updateText(this));
      }

      setTimeout(() => {
        refreshFiles(this, true, false);
      }, 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCallbacks(this);
      refreshFiles(this, true, false);
      return r;
    };
  },
});
