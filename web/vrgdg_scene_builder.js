import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_SceneBuilder";

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this);

      this.addWidget("button", "Build Scenes", null, () => {
        const metaLink = this.inputs?.find(i => i.name === "meta")?.link;
        if (!metaLink) {
          alert("[SceneBuilder] No meta connected!");
          return;
        }

        const linkInfo = app.graph.links[metaLink];
        if (!linkInfo) {
          alert("[SceneBuilder] Invalid meta link!");
          return;
        }

        const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
        if (!sourceNode) {
          alert("[SceneBuilder] Could not find source of meta link!");
          return;
        }

        const meta = sourceNode.widgets_values?.[0]; // placeholder: you’ll likely extend this
        console.log("[SceneBuilder] Triggered build with meta:", meta);

        // TODO: build logic
        // - parse meta.scene_count, meta.durations
        // - find [SCENE_IN] and [SCENE_OUT] template chain
        // - clone it N times
        // - wire LoadAudioSplitDynamic.audio_i -> SCENE_IN.audio
        // - wire SCENE_OUT.images -> VRGDG_CombinevideosV2.video_i
      });

      return r;
    };
  },
});
