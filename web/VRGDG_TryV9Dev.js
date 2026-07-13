import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_TryV9Dev";

app.registerExtension({
  name: "vrgdg.TryV9Dev",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      this.setSize([380, 180]);

      const status = this.addWidget("text", "status", "Ready to switch this installation to V9 Dev.", () => {}, {
        multiline: true,
      });
      status.disabled = true;
      status.serialize = false;

      this.addWidget("button", "What does this do?", null, () => {
        window.alert(
          "This switches the existing comfyui-vrgamedevgirl folder from the main branch to V9 Dev.\n\n" +
          "It runs:\n" +
          "git fetch origin\n" +
          "git switch dev/music-video-builder-ui-test-v9\n" +
          "git pull\n\n" +
          "It does not install a second copy, run git reset, run git clean, or delete files you created. Git stops if switching would overwrite conflicting work."
        );
      });

      const updateButton = this.addWidget("button", "Switch to V9 Dev", null, async () => {
        if (!window.confirm(
          "Switch this VRGDG installation from main to V9 Dev?\n\n" +
          "This runs the same fetch, switch, and pull commands as the manual instructions. Continue?"
        )) return;

        updateButton.disabled = true;
        status.value = "Fetching and switching to V9 Dev...";
        this.setDirtyCanvas(true, true);
        try {
          const response = await api.fetchApi("/vrgdg/try_v9_dev", { method: "POST" });
          let payload = {};
          try { payload = await response.json(); } catch (_) { payload = {}; }
          if (!response.ok || !payload.ok) {
            throw new Error(payload.error || `Update failed (HTTP ${response.status}).`);
          }
          status.value = "V9 Dev installed. Fully restart ComfyUI and hard-refresh the browser.";
          window.alert(
            "Switched to V9 Dev successfully.\n\nFully stop and restart ComfyUI, then hard-refresh the browser page."
          );
        } catch (error) {
          status.value = `Switch did not complete: ${error?.message || error}`;
          window.alert(`Switch to V9 Dev did not complete:\n\n${error?.message || error}`);
        } finally {
          updateButton.disabled = false;
          this.setDirtyCanvas(true, true);
        }
      });
    };
  },
});
