import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_CombinevideosV2";
const MAX_SCENES = 50;

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this);
      const getWidget = (n) => (this.widgets || []).find((w) => w.name === n);

      // remove non-video ports permanently
      const stripNonVideoPorts = () => {
        if (!Array.isArray(this.inputs)) return;
        for (let i = this.inputs.length - 1; i >= 0; i--) {
          const n = this.inputs[i]?.name;
          if (n === "pad_short_videos" || n === "fps" || n === "scene_count") {
            if (this.inputs[i].link != null) this.disconnectInput(i);
            this.removeInput(i);
          }
        }
      };

      // ensure exactly video_1..video_count exist (no more, no less)
      const ensureVideoPorts = (count) => {
        if (!Array.isArray(this.inputs)) this.inputs = [];

        // 1) remove extras (> count)
        for (let i = this.inputs.length - 1; i >= 0; i--) {
          const inp = this.inputs[i];
          const name = inp?.name || "";
          if (name.startsWith("video_")) {
            const idx = Number(name.split("_")[1] || 0);
            if (!Number.isFinite(idx) || idx < 1 || idx > count) {
              if (inp.link != null) this.disconnectInput(i);
              this.removeInput(i);
            }
          }
        }

        // 2) collect existing indices
        const have = new Set(
          (this.inputs || [])
            .map((inp) => inp?.name)
            .filter((n) => n?.startsWith?.("video_"))
            .map((n) => Number(n.split("_")[1]))
        );

        // 3) add missing 1..count
        for (let i = 1; i <= count; i++) {
          if (!have.has(i)) {
            this.addInput(`video_${i}`, "IMAGE");
          }
        }

        // 4) sort ports by index (video_1..video_n)
        this.inputs.sort((a, b) => {
          const ai = Number((a?.name || "").split("_")[1] || 0);
          const bi = Number((b?.name || "").split("_")[1] || 0);
          return ai - bi;
        });
      };

        const rebuildDurationWidgets = (count) => {
        const keep = new Set(["fps", "pad_short_videos", "scene_count"]);
        for (let i = 1; i <= count; i++) keep.add(`duration_${i}`);

        // keep buttons + needed widgets
        this.widgets = (this.widgets || []).filter(
            (w) => w?.type === "button" || !w?.name || keep.has(w.name)
        );

        // ensure duration widgets exist (force 1-decimal precision)
        const ensureDuration = (name) => {
            const w = (this.widgets || []).find((x) => x.name === name);
            if (!w) {
            this.addWidget(
                "number",
                name,
                0.0,
                (v, ww) => (ww.value = Math.max(0, Number(v) || 0)),
                { min: 0.0, step: 0.1, precision: 1 } // <-- enforce 0.0 style
            );
            } else {
            // normalize existing ones the UI already had
            w.options = w.options || {};
            w.options.min = 0.0;
            w.options.step = 0.1;
            w.options.precision = 1; // <-- key line
            }
        };

        for (let i = 1; i <= count; i++) ensureDuration(`duration_${i}`);

        // (optional) also normalize fps to 1 decimal:
        const fpsW = (this.widgets || []).find((w) => w.name === "fps");
        if (fpsW) {
            fpsW.options = fpsW.options || {};
            fpsW.options.min = 1.0;
            fpsW.options.step = 0.1;
            fpsW.options.precision = 1;
        }

        // order widgets
        const order = ["fps", "pad_short_videos", "scene_count", ...Array.from({ length: count }, (_, k) => `duration_${k + 1}`)];
        const btns = (this.widgets || []).filter((w) => w.type === "button");
        const others = (this.widgets || []).filter((w) => w.type !== "button");
        others.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
        this.widgets = [...others, ...btns];
        };


      const refresh = () => {
        const sc = Number(getWidget("scene_count")?.value ?? 2);
        const count = Math.max(2, Math.min(MAX_SCENES, sc));

        stripNonVideoPorts();
        ensureVideoPorts(count);
        rebuildDurationWidgets(count);

        // redraw
        this.setSize([this.size[0], this.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);
      };

      // add refresh button once
      if (!(this.widgets || []).some((w) => w.type === "button" && w.label === "Refresh Inputs")) {
        this.addWidget("button", "Refresh Inputs", undefined, () => refresh());
      }

      // auto-refresh when scene_count changes
      const scW = getWidget("scene_count");
      if (scW && !scW._vrgdg_bound) {
        const orig = scW.callback;
        scW.callback = (v, w) => { orig?.(v, w); refresh(); };
        scW._vrgdg_bound = true;
      }

      // initial normalize
      setTimeout(refresh, 50);
      return r;
    };
  },
});
