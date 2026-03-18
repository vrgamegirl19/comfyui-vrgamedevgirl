import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_MusubiTunerInstaller";

console.log("[VRGDG] Musubi installer extension loaded");

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w?.name === name);
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

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value ?? "";
}

function ensureInstallerButton(node) {
  const widgets = node.widgets || [];
  const exists = widgets.some((w) => w?.type === "button" && w?.name === "Install Musubi-Tuner");
  if (exists) return;

  const buttonWidget = node.addWidget("button", "Install Musubi-Tuner", null, () => installMusubi(node));
  const buttonIndex = (node.widgets || []).indexOf(buttonWidget);
  if (buttonIndex >= 0) {
    node.widgets.splice(buttonIndex, 1);
    node.widgets.splice(Math.min(2, node.widgets.length), 0, buttonWidget);
  }
}

async function installMusubi(node) {
  const targetWidget = getWidget(node, "target_root");
  const downloadWidget = getWidget(node, "download_models");
  const targetRoot = String(targetWidget?.value || "").trim();
  const downloadModels = Boolean(downloadWidget?.value);

  if (!targetRoot) {
    window.alert("target_root is empty.");
    return;
  }

  console.log("[VRGDG] Musubi installer started:", targetRoot);

  try {
    const response = await api.fetchApi("/vrgdg/musubi/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_root: targetRoot,
        download_models: downloadModels,
      }),
    });

    const data = await response.json();
    if (Array.isArray(data?.messages)) {
      for (const line of data.messages) {
        console.log(line);
      }
    }
    if (Array.isArray(data?.checks)) {
      console.log("[VRGDG] Verification checks:", data.checks);
    }
    if (data?.report_path) {
      console.log("[VRGDG] Verification report:", data.report_path);
    }

    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `HTTP ${response.status}`);
    }

    setWidgetValue(node, "install_root", data.install_path || "");
    setWidgetValue(node, "checkpoint_path", data.checkpoint_path || "");
    setWidgetValue(node, "gemma_root_out", data.gemma_root || "");
    setWidgetValue(node, "report_path", data.report_path || "");
    setWidgetValue(node, "status_text", data.status || "Musubi-Tuner installed successfully.");

    console.log("[VRGDG] Musubi installer completed:", data.install_path);
    window.alert(`Musubi-Tuner installed successfully at:\n${data.install_path}`);
  } catch (error) {
    console.error("[VRGDG] Musubi installer failed:", error);
    window.alert(`Musubi-Tuner install failed: ${error.message || error}`);
  }
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".Installer",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    function hideBackingWidgets(node) {
      for (const name of ["install_root", "checkpoint_path", "gemma_root_out", "report_path", "status_text"]) {
        setWidgetVisible(getWidget(node, name), false);
      }
      node?.setSize?.(node.size);
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      hideBackingWidgets(this);
      setTimeout(() => hideBackingWidgets(this), 0);
      ensureInstallerButton(this);

      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = origOnConfigure?.apply(this, arguments);
      hideBackingWidgets(this);
      ensureInstallerButton(this);
      return result;
    };
  },
});
