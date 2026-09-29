import { app } from "../../scripts/app.js";

const NODE_NAME = "VRGDG_StartImageStoryboard";

// The creator lives in start_image_storyboard/storyboard_creator.mjs and is imported on first open so it does not load with every page.
async function openStoryboardCreator(...args) {
  const { openStoryboardCreator } = await import("./start_image_storyboard/storyboard_creator.mjs");
  return openStoryboardCreator(...args);
}

app.registerExtension({
  name: "vrgdg.StartImageStoryboard",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      this.size = [320, 92];
      this.addWidget("button", "Open Storyboard Creator", null, openStoryboardCreator);
    };
  },
});

window.VRGDGStartImageStoryboard = { open: openStoryboardCreator };
