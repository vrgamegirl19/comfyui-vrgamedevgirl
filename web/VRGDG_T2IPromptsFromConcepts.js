import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_T2IPromptsFromConcepts";

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget?.name === name);
}

function showMessage(title, message, isError = false) {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.62);
  `;

  const panel = document.createElement("div");
  panel.style.cssText = `
    width: min(560px, calc(100vw - 32px));
    border: 1px solid ${isError ? "#7f1d1d" : "#334155"};
    border-radius: 8px;
    background: #111827;
    box-shadow: 0 20px 70px rgba(0, 0, 0, 0.48);
    padding: 18px;
    color: #f8fafc;
  `;

  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-size: 16px; font-weight: 800; margin-bottom: 10px;";

  const body = document.createElement("div");
  body.textContent = message;
  body.style.cssText = `
    white-space: pre-wrap;
    color: ${isError ? "#fecaca" : "#cbd5e1"};
    font-size: 13px;
    line-height: 1.45;
    word-break: break-word;
    margin-bottom: 14px;
  `;

  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.style.cssText = `
    float: right;
    border: 1px solid #475569;
    border-radius: 8px;
    background: #1f2937;
    color: #e5e7eb;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
  `;
  close.addEventListener("click", () => overlay.remove());
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) overlay.remove();
  });

  panel.append(heading, body, close);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
}

function showProgress(message) {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.62);
  `;

  const panel = document.createElement("div");
  panel.style.cssText = `
    width: min(460px, calc(100vw - 32px));
    border: 1px solid #334155;
    border-radius: 8px;
    background: #111827;
    box-shadow: 0 20px 70px rgba(0, 0, 0, 0.48);
    padding: 18px;
    color: #f8fafc;
  `;

  const title = document.createElement("div");
  title.textContent = "Creating T2I prompts";
  title.style.cssText = "font-size: 16px; font-weight: 800; margin-bottom: 8px;";

  const body = document.createElement("div");
  body.textContent = message;
  body.style.cssText = "font-size: 13px; color: #cbd5e1; line-height: 1.45; white-space: pre-wrap;";

  const meter = document.createElement("div");
  meter.style.cssText = `
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: #1f2937;
    margin-top: 14px;
  `;
  const fill = document.createElement("div");
  fill.style.cssText = `
    width: 8%;
    height: 100%;
    border-radius: 999px;
    background: #10b981;
    transition: width 0.2s ease;
  `;
  meter.appendChild(fill);

  overlay.__vrgdgSetProgress = (data) => {
    const current = Number(data?.current || 0);
    const total = Number(data?.total || 0);
    const key = String(data?.current_key || "");
    const outputPath = String(data?.output_path || "");
    const lines = [String(data?.message || message)];
    if (total > 0) lines.push(`Step ${Math.min(current, total)} of ${total}${key ? `: ${key}` : ""}`);
    if (outputPath) lines.push("", `Saving to:\n${outputPath}`);
    body.textContent = lines.join("\n");
    fill.style.width = total > 0 ? `${Math.max(8, Math.min(100, (current / total) * 100))}%` : "18%";
  };

  panel.append(title, body, meter);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  return overlay;
}

function requestStart() {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
      position: fixed;
      inset: 0;
      z-index: 100000;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(2, 6, 23, 0.62);
    `;

    const panel = document.createElement("div");
    panel.style.cssText = `
      width: min(560px, calc(100vw - 32px));
      border: 1px solid #334155;
      border-radius: 8px;
      background: #111827;
      box-shadow: 0 20px 70px rgba(0, 0, 0, 0.48);
      padding: 18px;
      color: #f8fafc;
    `;

    const title = document.createElement("div");
    title.textContent = "Create T2I prompts from concepts";
    title.style.cssText = "font-size: 16px; font-weight: 800; margin-bottom: 10px;";

    const body = document.createElement("div");
    body.textContent = [
      "Gemma will read ConceptPrompts.txt one row at a time.",
      "It will use themestyle.txt and storyconcept.txt for context.",
      "For each row, it will create camera motion and character motion, then give those to Gemma while making the final T2I prompt.",
      "",
      "Output:",
      "VRGDG_TEMP/TextFiles/t2i_Prompts/t2i_Prompts.txt",
      "",
      "Gemma unloads after the final prompt.",
    ].join("\n");
    body.style.cssText = "white-space: pre-wrap; color: #cbd5e1; font-size: 13px; line-height: 1.45; margin-bottom: 14px;";

    const textarea = document.createElement("textarea");
    textarea.rows = 6;
    textarea.placeholder = "Optional: describe camera motion, character motion, or anything else you want Gemma to use. Example: fast camera movement, quick handheld pushes, intense character motion, energetic performance.";
    textarea.style.cssText = `
      width: 100%;
      box-sizing: border-box;
      resize: vertical;
      border: 1px solid #334155;
      border-radius: 8px;
      background: #0d1217;
      color: #f3f4f6;
      padding: 10px 12px;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 14px;
    `;

    const actions = document.createElement("div");
    actions.style.cssText = "display: flex; justify-content: flex-end; gap: 8px;";

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    cancel.style.cssText = "border: 1px solid #475569; border-radius: 8px; background: #1f2937; color: #e5e7eb; padding: 8px 12px; cursor: pointer; font-size: 12px; font-weight: 700;";

    const create = document.createElement("button");
    create.type = "button";
    create.textContent = "Create Prompts";
    create.style.cssText = "border: 1px solid #059669; border-radius: 8px; background: #10b981; color: #052e1b; padding: 8px 12px; cursor: pointer; font-size: 12px; font-weight: 800;";

    function finish(value) {
      overlay.remove();
      resolve(value);
    }

    cancel.addEventListener("click", () => finish(null));
    create.addEventListener("click", () => finish(String(textarea.value || "").trim()));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) finish(null);
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(null);
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) finish(String(textarea.value || "").trim());
    });

    actions.append(cancel, create);
    panel.append(title, body, textarea, actions);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    setTimeout(() => textarea.focus(), 0);
  });
}

function startProgressPolling(progress) {
  let stopped = false;
  async function poll() {
    if (stopped) return;
    try {
      const response = await api.fetchApi("/vrgdg/t2i_from_concepts/progress", { cache: "no-store" });
      const data = await response.json().catch(() => ({}));
      if (data?.ok) progress.__vrgdgSetProgress?.(data);
    } catch (error) {
      console.warn("[VRGDG] T2I concept progress poll failed:", error);
    }
    setTimeout(poll, 1000);
  }
  poll();
  return () => {
    stopped = true;
  };
}

async function createT2IPrompts(node) {
  const modelFile = String(getWidget(node, "model_file")?.value || "").trim();
  if (!modelFile) {
    showMessage("Missing Gemma model", "Choose a Gemma4 model first.", true);
    return;
  }

  const extraUserInput = await requestStart();
  if (extraUserInput === null) return;

  const progress = showProgress("Starting Gemma text-to-image prompt creation...");
  const stopPolling = startProgressPolling(progress);
  try {
    const response = await api.fetchApi("/vrgdg/t2i_from_concepts/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_file: modelFile, extra_user_input: extraUserInput }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data?.ok) {
      throw new Error(String(data?.error || `Request failed (${response.status})`));
    }
    stopPolling();
    progress.remove();
    showMessage(
      "T2I prompts created",
      [
        `Created ${data.count} prompt${data.count === 1 ? "" : "s"}.`,
        "",
        "Saved to:",
        data.output_path,
      ].join("\n")
    );
  } catch (error) {
    stopPolling();
    progress.remove();
    showMessage("T2I prompt creation failed", String(error?.message || error), true);
  }
}

function ensureButton(node) {
  const buttonName = "Create T2I Prompts";
  const runButton = () => createT2IPrompts(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));

  const button = node.addWidget("button", buttonName, null, runButton);
  if (button) button.serialize = false;
}

app.registerExtension({
  name: "vrgdg.T2IPromptsFromConcepts",

  loadedGraphNode(node) {
    if ((node?.comfyClass || node?.type) === NODE_NAME) ensureButton(node);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      ensureButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
  },
});
