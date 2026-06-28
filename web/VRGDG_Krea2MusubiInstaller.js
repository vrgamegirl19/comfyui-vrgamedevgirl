import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_Krea2MusubiInstaller";

console.log("[VRGDG] Krea 2 Musubi installer extension loaded");

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w?.name === name);
}

function normalizeWidgetValues(values) {
  if (!Array.isArray(values)) return values;
  return values.map((value) => (value == null ? "" : value));
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

function ensureInstallerButtons(node) {
  const widgets = node.widgets || [];
  const buttonDefs = [
    { name: "Install Krea 2 Musubi", action: "install_tuner" },
    { name: "Download Krea 2 Models", action: "download_models" },
    { name: "Install Krea 2 + Models", action: "install_and_download" },
  ];

  for (const def of buttonDefs) {
    const existingButton = widgets.find((w) => w?.type === "button" && w?.name === def.name);
    if (existingButton) {
      existingButton.serialize = false;
      continue;
    }

    const buttonWidget = node.addWidget("button", def.name, null, () => installKrea2(node, def.action));
    buttonWidget.serialize = false;
    const buttonIndex = (node.widgets || []).indexOf(buttonWidget);
    if (buttonIndex >= 0) {
      node.widgets.splice(buttonIndex, 1);
      node.widgets.splice(Math.min(2, node.widgets.length), 0, buttonWidget);
    }
  }
}

async function installKrea2(node, action = "install_and_download") {
  const targetRoot = String(getWidget(node, "target_root")?.value || "").trim();
  const modelsRoot = String(getWidget(node, "models_root")?.value || "").trim();

  if (!targetRoot) {
    window.alert("target_root is empty.");
    return;
  }

  console.log("[VRGDG] Krea 2 installer started:", targetRoot, modelsRoot, action);

  try {
    const response = await api.fetchApi("/vrgdg/krea2/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_root: targetRoot,
        model_root: modelsRoot,
        action,
      }),
    });

    const data = await response.json();
    if (Array.isArray(data?.messages)) {
      for (const line of data.messages) console.log(line);
    }
    if (Array.isArray(data?.checks)) {
      console.log("[VRGDG] Krea 2 verification checks:", data.checks);
    }
    if (data?.report_path) {
      console.log("[VRGDG] Krea 2 install report:", data.report_path);
    }

    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `HTTP ${response.status}`);
    }

    setWidgetValue(node, "install_root", data.install_path || "");
    setWidgetValue(node, "raw_dit_path", data.raw_dit_path || "");
    setWidgetValue(node, "turbo_dit_path", data.turbo_dit_path || "");
    setWidgetValue(node, "vae_path", data.vae_path || "");
    setWidgetValue(node, "text_encoder_path", data.text_encoder_path || "");
    setWidgetValue(node, "models_root", data.model_root || modelsRoot);
    setWidgetValue(node, "report_path", data.report_path || "");
    setWidgetValue(node, "status_text", data.status || "Krea 2 install complete.");

    const message =
      action === "download_models"
        ? `Krea 2 model assets downloaded successfully at:\n${data.model_root || modelsRoot}`
        : action === "install_tuner"
          ? `Krea 2-ready Musubi installed successfully at:\n${data.install_path}`
          : `Krea 2-ready Musubi and model assets installed successfully.\n\nMusubi:\n${data.install_path}\n\nModels:\n${data.model_root || modelsRoot}`;
    window.alert(message);
  } catch (error) {
    console.error("[VRGDG] Krea 2 installer failed:", error);
    window.alert(`Krea 2 install failed: ${error.message || error}`);
  }
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".Installer",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;
    const origConfigure = nodeType.prototype.configure;

    function hideBackingWidgets(node) {
      for (const name of ["install_root", "raw_dit_path", "turbo_dit_path", "vae_path", "text_encoder_path", "report_path", "status_text"]) {
        setWidgetVisible(getWidget(node, name), false);
      }
      node?.setSize?.(node.size);
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      hideBackingWidgets(this);
      setTimeout(() => hideBackingWidgets(this), 0);
      ensureInstallerButtons(this);
      return result;
    };

    nodeType.prototype.configure = function () {
      if (arguments[0]?.widgets_values) {
        arguments[0].widgets_values = normalizeWidgetValues(arguments[0].widgets_values);
      }
      const result = origConfigure?.apply(this, arguments);
      this.widgets_values = normalizeWidgetValues(this.widgets_values);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      if (arguments[0]?.widgets_values) {
        arguments[0].widgets_values = normalizeWidgetValues(arguments[0].widgets_values);
      }
      const result = origOnConfigure?.apply(this, arguments);
      this.widgets_values = normalizeWidgetValues(this.widgets_values);
      hideBackingWidgets(this);
      ensureInstallerButtons(this);
      return result;
    };
  },
});
