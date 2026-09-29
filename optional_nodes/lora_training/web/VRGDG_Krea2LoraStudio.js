import { app } from "../../scripts/app.js";

const NODE_NAME = "VRGDG_Krea2LoraStudio";

function ensureStudioButton(node) {
  const existing = (node.widgets || []).find((w) => w?.type === "button" && w?.name === "Open Krea 2 Studio");
  if (existing) {
    existing.serialize = false;
    return;
  }
  // The studio lives in krea2_studio/studio.mjs and is imported on first open so it does not load with every page.
  const button = node.addWidget("button", "Open Krea 2 Studio", null, () => {
    import("./krea2_studio/studio.mjs")
      .then(({ Krea2Studio }) => new Krea2Studio(node).open())
      .catch((error) => window.alert(`Krea 2 Studio failed to open: ${error.message || error}`));
  });
  button.serialize = false;
}

app.registerExtension({
  name: "vrgdg.Krea2LoraStudio",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      ensureStudioButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = origOnConfigure?.apply(this, arguments);
      ensureStudioButton(this);
      return result;
    };
  },
});
