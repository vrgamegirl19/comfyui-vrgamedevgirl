import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "VRGDGLongVideoMetaBatchLoader";
const UPLOAD_URL = "/vrgdg/long_video/upload";
const CHUNK_SIZE = 50 * 1024 * 1024;

function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

app.registerExtension({
  name: "vrgdg.long_video_meta_batch_loader",
  nodeCreated(node) {
    if ((node.comfyClass || node.type) !== NODE_TYPE || node._vrgdgLongVideoButton) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v,.gif";
    input.style.display = "none";
    document.body.appendChild(input);
    const button = node.addWidget("button", "Upload long video", null, () => input.click());
    node._vrgdgLongVideoButton = button;
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      input.value = "";
      if (!file || node._vrgdgLongVideoUploadPending) return;
      node._vrgdgLongVideoUploadPending = true;
      const original = button.label;
      try {
        const id = crypto.randomUUID();
        const total = Math.ceil(file.size / CHUNK_SIZE);
        let completedPath = "";
        for (let index = 0; index < total; index += 1) {
          button.label = `Uploading ${index + 1}/${total}`;
          const start = index * CHUNK_SIZE;
          const response = await api.fetchApi(
            `${UPLOAD_URL}?upload_id=${encodeURIComponent(id)}&chunk=${index}&total=${total}&filename=${encodeURIComponent(file.name)}`,
            { method: "POST", body: file.slice(start, Math.min(file.size, start + CHUNK_SIZE)) },
          );
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || response.statusText || "Upload failed.");
          if (data.complete) completedPath = data.path;
        }
        const videoWidget = widget(node, "video");
        if (!completedPath) throw new Error("Upload completed without a server file path.");
        if (videoWidget) {
          videoWidget.value = completedPath;
          videoWidget.callback?.(completedPath);
        }
        button.label = `Uploaded: ${file.name}`;
        app.graph?.setDirtyCanvas(true, true);
      } catch (error) {
        button.label = original;
        alert(`Long video upload failed: ${error.message || error}`);
      } finally {
        node._vrgdgLongVideoUploadPending = false;
      }
    });
  },
});
