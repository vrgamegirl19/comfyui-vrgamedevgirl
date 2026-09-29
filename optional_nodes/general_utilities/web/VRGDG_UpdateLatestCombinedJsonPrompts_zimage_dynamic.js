import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_UpdateLatestCombinedJsonPrompts_zimage";
const EMPTY_OPTION = "<no files found>";
const MAX_SLOTS = 120;
const FIXED_BATCH_TYPE = "Text2Image";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function getConnectedStringValue(node, inputName) {
  const input = (node.inputs || []).find((entry) => entry?.name === inputName);
  const linkId = input?.link;
  if (!linkId || !app.graph?.links) return "";

  const linkInfo = app.graph.links[linkId];
  if (!linkInfo?.origin_id) return "";

  const sourceNode = app.graph.getNodeById?.(linkInfo.origin_id);
  if (!sourceNode) return "";

  const widgetValues = Array.isArray(sourceNode.widgets_values) ? sourceNode.widgets_values : [];
  const storedString = widgetValues.find((value) => typeof value === "string" && value.trim());
  if (storedString) return storedString;

  const liveWidget = (sourceNode.widgets || []).find(
    (widget) => typeof widget?.value === "string" && String(widget.value).trim()
  );
  return liveWidget ? String(liveWidget.value) : "";
}

function normalizeFolderPathInput(value) {
  let text = String(value || "").trim();
  if (text.startsWith("file:///")) {
    text = text.slice(8);
  }
  return text.trim().replace(/^["'`]+|["'`]+$/g, "").trim();
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

  widget.computeSize = function (...args) {
    const size = original.apply(this, args);
    return [size[0], size[1] + 10];
  };
}

function refreshInputVisibility(node) {
  const countWidget = getWidget(node, "prompt_count");
  if (!countWidget) return;

  const count = Math.max(0, Math.min(MAX_SLOTS, Number(countWidget.value ?? 0)));
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const visible = i <= count;
    setWidgetVisible(getWidget(node, `prompt_number_${i}`), visible);
    const textWidget = getWidget(node, `prompt_text_${i}`);
    setWidgetVisible(textWidget, visible);
    if (visible) {
      setWidgetGroupSpacing(textWidget);
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
    if (numWidget) numWidget.value = i;
    if (textWidget) textWidget.value = "";
  }
}

function applyPromptNumbers(node, promptNumbers) {
  const numbers = Array.isArray(promptNumbers)
    ? promptNumbers
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value) && value > 0)
        .map((value) => Math.trunc(value))
    : [];

  const countWidget = getWidget(node, "prompt_count");
  if (countWidget) {
    countWidget.value = Math.max(0, Math.min(MAX_SLOTS, numbers.length));
  }

  for (let i = 1; i <= MAX_SLOTS; i++) {
    const numWidget = getWidget(node, `prompt_number_${i}`);
    const textWidget = getWidget(node, `prompt_text_${i}`);
    if (numWidget) {
      numWidget.value = i <= numbers.length ? numbers[i - 1] : i;
    }
    if (textWidget) {
      textWidget.value = "";
    }
  }

  refreshInputVisibility(node);
}

async function refreshPromptValues(node) {
  const fileWidget = getWidget(node, "combined_json_file");
  const countWidget = getWidget(node, "prompt_count");
  const selectedFile = String(fileWidget?.value || "");
  if (!selectedFile || selectedFile === EMPTY_OPTION) {
    clearPromptInputs(node);
    refreshInputVisibility(node);
    return;
  }

  const bt = encodeURIComponent(FIXED_BATCH_TYPE);
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
    const selectedCount = Math.max(0, Math.min(MAX_SLOTS, Number(countWidget?.value ?? 0)));

    for (let i = 1; i <= selectedCount; i++) {
      const numWidget = getWidget(node, `prompt_number_${i}`);
      const textWidget = getWidget(node, `prompt_text_${i}`);
      const promptNumber = Number(numWidget?.value ?? i);
      const row = promptByNumber.get(promptNumber) || null;
      if (textWidget) textWidget.value = row ? String(row.prompt ?? "") : "";
    }
  } catch (e) {
    clearPromptInputs(node);
  }

  refreshInputVisibility(node);
}

async function refreshFiles(node, options = {}) {
  const fileWidget = getWidget(node, "combined_json_file");
  if (!fileWidget) return;
  const {
    keepSelection = true,
    loadPromptValues = true,
    clearInputsOnSelectionChange = false,
  } = options;

  const batchType = encodeURIComponent(FIXED_BATCH_TYPE);
  let fileOptions = [EMPTY_OPTION];
  try {
    const res = await api.fetchApi(`/vrgdg/llm_batches/combined_files?batch_type=${batchType}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.files) && data.files.length) {
      fileOptions = data.files.map((v) => String(v));
    }
  } catch (e) {
    fileOptions = [EMPTY_OPTION];
  }

  const current = String(fileWidget.value || "");
  fileWidget.options = fileWidget.options || {};
  fileWidget.options.values = [...fileOptions];
  const selectionPreserved = keepSelection && fileOptions.includes(current);

  if (selectionPreserved) {
    node.__vrgdgSkipNextFileRefresh = true;
    fileWidget.value = current;
  } else {
    node.__vrgdgSkipNextFileRefresh = true;
    fileWidget.value = fileOptions[0];
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
  if (loadPromptValues) {
    await refreshPromptValues(node);
  } else if (clearInputsOnSelectionChange && !selectionPreserved) {
    clearPromptInputs(node);
    refreshInputVisibility(node);
  } else {
    refreshInputVisibility(node);
  }
}

function collectUpdates(node) {
  const countWidget = getWidget(node, "prompt_count");
  const count = Math.max(0, Math.min(MAX_SLOTS, Number(countWidget?.value ?? 0)));
  const updates = [];

  for (let i = 1; i <= count; i++) {
    const numWidget = getWidget(node, `prompt_number_${i}`);
    const textWidget = getWidget(node, `prompt_text_${i}`);
    if (!numWidget || !textWidget) continue;

    const n = Number(numWidget.value ?? i);
    const prompt = String(textWidget.value ?? "");
    if (!Number.isFinite(n) || n <= 0 || !prompt.trim()) continue;

    updates.push({
      prompt_number: Math.trunc(n),
      prompt,
    });
  }

  return updates;
}

async function updateText(node) {
  const remakeModeWidget = getWidget(node, "remake_mode");
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
    batch_type: FIXED_BATCH_TYPE,
    use_plain_text: true,
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

    alert(`[VRGDG] Updated ${Number(data.updated || 0)} prompt(s).\n${String(data.file_path || "")}`);
  } catch (e) {
    alert(`[VRGDG] Update failed: ${String(e)}`);
  }
}

async function pullIndexesFromRemakeFolder(node) {
  const folderPathWidget = getWidget(node, "folder_path");
  const folderPath = normalizeFolderPathInput(
    getConnectedStringValue(node, "folder_path") || folderPathWidget?.value || ""
  );
  if (!folderPath) {
    alert("[VRGDG] Connect or enter a folder path first.");
    return;
  }

  try {
    const res = await api.fetchApi("/vrgdg/llm_batches/remake_prompt_indexes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: folderPath }),
      cache: "no-store",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data?.ok) {
      throw new Error(String(data?.error || `HTTP ${res.status}`));
    }

    const promptNumbers = Array.isArray(data.prompt_numbers) ? data.prompt_numbers : [];
    if (!promptNumbers.length) {
      alert(`[VRGDG] The remake folder is empty.\n${String(data.remake_folder || "")}`);
      return;
    }

    applyPromptNumbers(node, promptNumbers);
    alert(
      `[VRGDG] Loaded ${promptNumbers.length} prompt index(es) from remake.\n${String(
        data.remake_folder || ""
      )}`
    );
  } catch (e) {
    alert(`[VRGDG] Failed to pull indexes from remake folder: ${String(e)}`);
  }
}

function bindCallbacks(node) {
  if (node.__vrgdgCombinedPromptEditorZimageBound) return;

  const countWidget = getWidget(node, "prompt_count");
  if (countWidget) {
    const oldCount = countWidget.callback;
    countWidget.callback = function () {
      if (oldCount) oldCount.apply(this, arguments);
      refreshInputVisibility(node);
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

  node.__vrgdgCombinedPromptEditorZimageBound = true;
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
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Pull Index From Remake Folder")) {
        this.addWidget("button", "Pull Index From Remake Folder", null, () => {
          pullIndexesFromRemakeFolder(this);
        });
      }
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Update Text")) {
        this.addWidget("button", "Update Text", null, () => updateText(this));
      }

      setTimeout(() => {
        refreshFiles(this, { keepSelection: true, loadPromptValues: false, clearInputsOnSelectionChange: false });
      }, 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCallbacks(this);
      refreshFiles(this, { keepSelection: true, loadPromptValues: false, clearInputsOnSelectionChange: false });
      return r;
    };
  },
});
