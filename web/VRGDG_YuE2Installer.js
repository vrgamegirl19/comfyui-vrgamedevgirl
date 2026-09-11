import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_YuE2Installer";

function widget(node, name) {
  return (node.widgets || []).find((item) => item?.name === name);
}

function hide(item) {
  if (!item) return;
  if (!item.__vrgdgOriginalType) {
    item.__vrgdgOriginalType = item.type;
    item.__vrgdgOriginalComputeSize = item.computeSize;
  }
  item.type = "hidden";
  item.computeSize = () => [0, -4];
}

function setValue(node, name, value) {
  const item = widget(node, name);
  if (item) item.value = value ?? "";
}

function fillInstalledPaths(node, data, action) {
  const generation = action === "install_generation" || action === "install_all" || action === "verify";
  const cover = action === "install_cover" || action === "install_all" || action === "verify";
  const values = {
    target_root: data?.target_root,
    cache_dir: data?.cache,
  };
  if (generation) {
    values.yue2_python = data?.yue2_python;
    values.model = data?.model;
    values.vae = data?.vae;
  }
  if (cover) {
    values.sheetsage2_python = data?.sheetsage_python;
    values.sheetsage2_model = data?.sheetsage_model;
    values.mert_model = data?.mert_model;
  }
  for (const [name, value] of Object.entries(values)) {
    if (value != null && String(value).trim()) setValue(node, name, value);
  }
}

async function runInstaller(node, action) {
  const targetRoot = String(widget(node, "target_root")?.value || "").trim();
  if (!targetRoot) {
    window.alert("Choose a dedicated target_root first, for example D:\\Yue2.");
    return;
  }
  const actionNames = {
    install_generation: "YuE2 generation runtime and models",
    install_cover: "SheetSage2 cover runtime and models",
    install_all: "YuE2, cover tools, and all models",
    verify: "YuE2 installation verification",
  };
  if (action !== "verify") {
    const coverRequirement = action === "install_cover" || action === "install_all"
      ? "\n\nCover tools require a separately installed Python 3.10 or 3.11 with the Windows py launcher enabled. ComfyUI's bundled Python does not satisfy this prerequisite."
      : "";
    if (!window.confirm(
      `Install ${actionNames[action]} into:\n${targetRoot}${coverRequirement}\n\nThis downloads large packages and model weights and may take a while.`
    )) return;
  }

  console.log(`[VRGDG/YuE2 installer] Starting ${action}:`, targetRoot);
  try {
    const response = await api.fetchApi("/vrgdg/yue2/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_root: targetRoot, action }),
    });
    const data = await response.json();
    for (const line of data?.messages || []) console.log(line);
    if (data?.checks) console.log("[VRGDG/YuE2 installer] Checks:", data.checks);
    setValue(node, "status_text", data?.status || data?.error || "");
    setValue(node, "report_path_saved", data?.report_path || "");
    if (!response.ok || !data?.ok) throw new Error(data?.error || `HTTP ${response.status}`);
    fillInstalledPaths(node, data, action);
    node.graph?.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
    window.alert(`${actionNames[action]} completed successfully.\n\nInstalled paths were filled into the node.\n\nReport:\n${data.report_path}`);
  } catch (error) {
    console.error("[VRGDG/YuE2 installer] Failed:", error);
    window.alert(`YuE2 installer failed:\n${error.message || error}\n\nSee the ComfyUI console for details.`);
  }
}

function addButtons(node) {
  const definitions = [
    ["Install Generation + Models", "install_generation", "Install the isolated YuE2 generation environment and download YuE2-3B plus YuE2-Vae."],
    ["Install Cover Tools + Models [Requires Python 3.10/3.11]", "install_cover", "Requires a separately installed Python 3.10 or 3.11 with the Windows py launcher enabled. Installs the SheetSage2 transcription environment and downloads SheetSage2 plus MERT-v2-FullSong."],
    ["Install Everything [Cover Requires Python 3.10/3.11]", "install_all", "Installs both isolated environments and every model. The node installs packages and models, but it does not install system Python; cover tools require Python 3.10 or 3.11 first."],
    ["Verify Existing Installation", "verify", "Check Python environments, CUDA support, packages, and model files without installing or changing them."],
  ];
  for (const [name, action, tooltip] of definitions) {
    let button = (node.widgets || []).find((item) => item?.type === "button" && item?.name === name);
    if (!button) button = node.addWidget("button", name, null, () => runInstaller(node, action));
    button.serialize = false;
    button.serializeValue = () => undefined;
    button.tooltip = tooltip;
    // Action widgets must follow every server-defined input widget. ComfyUI's
    // API exporter associates widgets with backend inputs by position; placing
    // a null-valued button between target_root and runtime_mode makes export
    // attempt String.replace() on null.
    const buttonIndex = (node.widgets || []).indexOf(button);
    if (buttonIndex >= 0 && buttonIndex !== node.widgets.length - 1) {
      node.widgets.splice(buttonIndex, 1);
      node.widgets.push(button);
    }
  }
}

function prepare(node) {
  hide(widget(node, "status_text"));
  hide(widget(node, "report_path_saved"));
  addButtons(node);
  node.setSize?.([Math.max(node.size?.[0] || 420, 420), Math.max(node.size?.[1] || 260, 260)]);
}

app.registerExtension({
  name: "VRGDG.YuE2Installer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const created = nodeType.prototype.onNodeCreated;
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      prepare(this);
      setTimeout(() => prepare(this), 0);
      return result;
    };
    nodeType.prototype.onConfigure = function () {
      const result = configured?.apply(this, arguments);
      prepare(this);
      return result;
    };
  },
});
