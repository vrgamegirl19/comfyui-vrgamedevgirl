import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_Krea2AIToolkitInstaller";
const widget = (node, name) => (node.widgets || []).find((item) => item?.name === name);

function hide(item) {
  if (!item) return;
  item.type = "hidden";
  item.computeSize = () => [0, -4];
}

function set(node, name, value) {
  const item = widget(node, name);
  if (item) item.value = value ?? "";
}

function addButton(node) {
  if ((node.widgets || []).some((item) => item?.type === "button" && item?.name === "Install AI Toolkit for Krea 2 Edit")) return;
  const item = node.addWidget("button", "Install AI Toolkit for Krea 2 Edit", null, async () => {
    const targetRoot = String(widget(node, "target_root")?.value || "").trim();
    if (!targetRoot) return window.alert("target_root is empty.");
    item.disabled = true;
    try {
      const response = await api.fetchApi("/vrgdg/ai_toolkit/install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_root: targetRoot, branch: "main" }),
      });
      const data = await response.json();
      for (const line of data.messages || []) console.log(line);
      if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
      set(node, "install_root", data.install_path);
      set(node, "python_path", data.python_path);
      set(node, "status_text", data.status);
      set(node, "report_path", data.report_path);
      window.alert(`AI Toolkit installed successfully.\n\n${data.install_path}\n\nKrea 2 Edit training is now available in Krea 2 Studio.`);
    } catch (error) {
      window.alert(`AI Toolkit installation failed:\n${error.message || error}\n\nSee the ComfyUI console for the installation log.`);
    } finally {
      item.disabled = false;
    }
  });
  item.serialize = false;
}

app.registerExtension({
  name: "vrgdg.Krea2AIToolkitInstaller",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const created = nodeType.prototype.onNodeCreated;
    const configured = nodeType.prototype.onConfigure;
    const setup = (node) => {
      for (const name of ["install_root", "python_path", "status_text", "report_path"]) hide(widget(node, name));
      addButton(node);
      node.setSize?.(node.size);
    };
    nodeType.prototype.onNodeCreated = function () { const result = created?.apply(this, arguments); setup(this); setTimeout(() => setup(this), 0); return result; };
    nodeType.prototype.onConfigure = function () { const result = configured?.apply(this, arguments); setup(this); return result; };
  },
});
