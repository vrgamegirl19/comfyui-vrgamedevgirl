import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LoadVideos";

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this);

      const getWidget = (n) => (this.widgets || []).find((w) => w.name === n);

      const refresh = () => {
        const scW = getWidget("scene_count");
        if (scW && (scW.value == null || scW.value === "")) {
          scW.value = 3; // default
        }

        // always keep only one output: "video"
        if (!this.outputs || this.outputs.length === 0) {
          this.addOutput("video", "IMAGE");
        } else {
          // ensure correct name/type
          this.outputs = [{ name: "video", type: "IMAGE" }];
        }

        this.setSize([this.size[0], this.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);
      };

      // add refresh button if not present
      if (!(this.widgets || []).some((w) => w.type === "button" && w.label === "Refresh Outputs")) {
        this.addWidget("button", "Refresh Outputs", undefined, () => refresh());
      }

      // bind refresh on scene_count change
      const scW = getWidget("scene_count");
      if (scW && !scW._vrgdg_bound) {
        const orig = scW.callback;
        scW.callback = (v, w) => {
          orig?.(v, w);
          refresh();
        };
        scW._vrgdg_bound = true;
      }

      // run refresh after load
      setTimeout(refresh, 50);
      return r;
    };
  },
});
