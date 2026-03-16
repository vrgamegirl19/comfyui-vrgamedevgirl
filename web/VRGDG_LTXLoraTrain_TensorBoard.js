import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_LTXLoraTrainChunk";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w?.name === name);
}

async function openTensorBoard(node) {
  const workspaceWidget = getWidget(node, "workspace_dir");
  const workspaceDir = String(workspaceWidget?.value || "").trim();

  if (!workspaceDir) {
    window.alert("workspace_dir is empty.");
    return;
  }

  try {
    const response = await api.fetchApi("/vrgdg/ltx/tensorboard/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspace_dir: workspaceDir,
        port: 6006,
      }),
    });

    const data = await response.json();
    if (!response.ok || !data?.ok) {
      throw new Error(data?.error || `HTTP ${response.status}`);
    }

    if (data.url) {
      window.open(data.url, "_blank");
    }
  } catch (error) {
    console.error("[VRGDG] TensorBoard launch failed:", error);
    window.alert(`TensorBoard launch failed: ${error.message || error}`);
  }
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".TensorBoard",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);

      const exists = (this.widgets || []).some(
        (w) => w?.type === "button" && w?.name === "Show TensorBoard"
      );
      if (!exists) {
        this.addWidget("button", "Show TensorBoard", null, () => openTensorBoard(this));
      }

      return result;
    };
  },
});
