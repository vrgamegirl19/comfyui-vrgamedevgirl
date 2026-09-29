import { app } from "../../scripts/app.js";

const NODE_NAME = "VRGDG_PromptCreatorUI_V2";
const PART2_NODE_NAME = "VRGDG_Part2WorkflowUI";
const PART3_NODE_NAME = "VRGDG_Part3WorkflowUI";

function attachButton(node) {
  const buttonName = "Open Prompt Creator UI V2";
  const openUi = async () => {
    const modal = await ensureModal();
    modal.__vrgdgOpenForNode(node);
  };
  node.widgets = (node.widgets || []).filter((widget) => !(widget.type === "button" && widget.name === buttonName));

  const button = node.addWidget("button", buttonName, null, openUi);
  if (button) button.serialize = false;
}

function attachWorkflowButton(node, workflowKind) {
  const isPart3 = workflowKind === "part3";
  const buttonName = isPart3 ? "Open Workflow 3 UI" : "Open Part 2 Workflow UI";
  const openUi = async () => {
    const modal = await ensurePart2Modal();
    modal.__vrgdgOpenPart2(workflowKind, node);
  };
  node.widgets = (node.widgets || []).filter((widget) => !(widget.type === "button" && widget.name === buttonName));

  const button = node.addWidget("button", buttonName, null, openUi);
  if (button) button.serialize = false;
}

function attachUiForNode(node) {
  const nodeTypeName = node?.comfyClass || node?.type;
  if (nodeTypeName === PART2_NODE_NAME) attachWorkflowButton(node, "part2");
  else if (nodeTypeName === PART3_NODE_NAME) attachWorkflowButton(node, "part3");
  else if (nodeTypeName === NODE_NAME) attachButton(node);
}

// The modals live in prompt_creator_v2/prompt_creator.mjs and are imported on first open so they do not load with every page.
async function ensureModal(...args) {
  const { ensureModal } = await import("./prompt_creator_v2/prompt_creator.mjs");
  return ensureModal(...args);
}

async function ensurePart2Modal(...args) {
  const { ensurePart2Modal } = await import("./prompt_creator_v2/prompt_creator.mjs");
  return ensurePart2Modal(...args);
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  loadedGraphNode(node) {
    attachUiForNode(node);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME && nodeData.name !== PART2_NODE_NAME && nodeData.name !== PART3_NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      this.properties = this.properties || {};
      attachUiForNode(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.properties = this.properties || {};
      attachUiForNode(this);
      return result;
    };
  },
});
