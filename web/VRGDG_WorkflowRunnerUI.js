import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_ZImageWorkflowRunnerUI";
const CLEAR_MEMORY_NODE_NAME = "VRGDG_ClearMemoryButtonUI";
const MAX_LORA_SLOTS = 20;
const INTERNAL_WIDGETS = new Set([
  "workflow_path",
  "save_folder",
  "prompt",
  "first_pass_width",
  "first_pass_height",
  "second_pass_width",
  "second_pass_height",
  "batch_size",
  "use_custom_loras",
  "lora_count",
  "ltx_two_pass_mode",
]);
for (let i = 1; i <= MAX_LORA_SLOTS; i++) {
  INTERNAL_WIDGETS.add(`lora_${i}`);
  INTERNAL_WIDGETS.add(`strength_${i}`);
}

function getWidget(node, name) {
  return (node?.widgets || []).find((widget) => widget?.name === name);
}

function getWidgetValue(node, name, fallback = "") {
  const widget = getWidget(node, name);
  return widget?.value ?? fallback;
}

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
  const index = (node.widgets || []).indexOf(widget);
  if (Array.isArray(node.widgets_values) && index >= 0) node.widgets_values[index] = value;
  app.graph?.setDirtyCanvas?.(true, true);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgRunnerOriginalType")) {
    widget.__vrgdgRunnerOriginalType = widget.type;
    widget.__vrgdgRunnerOriginalComputeSize = widget.computeSize;
    widget.__vrgdgRunnerOriginalDraw = widget.draw;
  }
  widget.serialize = true;
  widget.hidden = !visible;
  if (visible) {
    widget.type = widget.__vrgdgRunnerOriginalType;
    if (widget.__vrgdgRunnerOriginalComputeSize) widget.computeSize = widget.__vrgdgRunnerOriginalComputeSize;
    else delete widget.computeSize;
    if (widget.__vrgdgRunnerOriginalDraw) widget.draw = widget.__vrgdgRunnerOriginalDraw;
    else delete widget.draw;
    return;
  }
  widget.type = "hidden";
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function hideInternalWidgets(node) {
  for (const widget of node?.widgets || []) {
    if (INTERNAL_WIDGETS.has(widget?.name)) setWidgetVisible(widget, false);
  }
  const width = Math.max(420, node?.size?.[0] || 420);
  node?.setSize?.([width, 96]);
  app.graph?.setDirtyCanvas?.(true, true);
}

function toast(message, isError = false) {
  const element = document.createElement("div");
  element.textContent = message;
  element.style.cssText = `
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 100003;
    max-width: min(560px, calc(100vw - 36px));
    border: 1px solid ${isError ? "#991b1b" : "#155e75"};
    border-radius: 8px;
    background: ${isError ? "#450a0a" : "#083344"};
    color: ${isError ? "#fecaca" : "#cffafe"};
    padding: 12px 14px;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 18px 60px rgba(0,0,0,.45);
  `;
  document.body.appendChild(element);
  setTimeout(() => element.remove(), 6500);
}

async function getJson(url) {
  const response = await api.fetchApi(url);
  const data = await response.json();
  if (!response.ok || data?.ok === false) throw new Error(data?.error || `Request failed: ${response.status}`);
  return data;
}

async function postJson(url, payload) {
  const response = await api.fetchApi(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data?.ok === false) throw new Error(data?.error || `Request failed: ${response.status}`);
  return data;
}

function makeImageViewUrl(image) {
  const params = new URLSearchParams();
  params.set("filename", image.filename || "");
  params.set("type", image.type || "output");
  if (image.subfolder) params.set("subfolder", image.subfolder);
  params.set("rand", String(Date.now()));
  return `/view?${params.toString()}`;
}

function extractImagesFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const images = [];
  for (const output of Object.values(outputs)) {
    if (Array.isArray(output?.images)) {
      for (const image of output.images) images.push(image);
    }
  }
  return images;
}

function extractTextFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const values = [];
  for (const output of Object.values(outputs)) {
    const text = output?.text ?? output?.ui?.text;
    if (Array.isArray(text)) values.push(...text);
    else if (text != null) values.push(text);
  }
  return values.flat(Infinity).map((value) => String(value ?? "")).filter((value) => value.trim());
}

async function waitForImages(promptId, onStatus) {
  const started = Date.now();
  while (Date.now() - started < 20 * 60 * 1000) {
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(`History request failed: ${response.status}`);
    const images = extractImagesFromHistory(data, promptId);
    if (images.length) return images;
    onStatus?.("Waiting for generated image...");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Timed out waiting for the generated image.");
}

async function waitForText(promptId, onStatus) {
  const started = Date.now();
  while (Date.now() - started < 5 * 60 * 1000) {
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(`History request failed: ${response.status}`);
    const text = extractTextFromHistory(data, promptId);
    if (text.length) return text;
    onStatus?.("Waiting for cleanup result...");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for the cleanup result.");
}

function makeButton(label, kind = "neutral") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.style.cssText = `
    border: 1px solid ${kind === "primary" ? "#0891b2" : "#3f3f46"};
    border-radius: 6px;
    background: ${kind === "primary" ? "#06b6d4" : "#27272a"};
    color: ${kind === "primary" ? "#082f49" : "#f4f4f5"};
    font-size: 12px;
    font-weight: 800;
    padding: 8px 11px;
    cursor: pointer;
  `;
  return button;
}

function makeInput(value = "", type = "text") {
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
  return input;
}

function makeSelect(options, value) {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
  for (const optionValue of options) {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = optionValue;
    select.append(option);
  }
  select.value = value;
  return select;
}

function makeField(label, control) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;color:#d4d4d8;font-weight:700;";
  const text = document.createElement("span");
  text.textContent = label;
  wrapper.append(text, control);
  return wrapper;
}

function makeGrid() {
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;";
  return grid;
}

function makeCheckbox(label, checked) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;align-items:center;gap:8px;font-size:12px;color:#f4f4f5;font-weight:800;";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(checked);
  wrapper.append(input, document.createTextNode(label));
  return { wrapper, input };
}

function collectPayload(node, controls) {
  const payload = {
    workflow_path: controls.workflowPath.value.trim(),
    save_folder: controls.saveFolder.value.trim(),
    prompt: controls.prompt.value.trim(),
    first_pass_width: Number(controls.firstWidth.value || 1280),
    first_pass_height: Number(controls.firstHeight.value || 720),
    second_pass_width: Number(controls.secondWidth.value || 1920),
    second_pass_height: Number(controls.secondHeight.value || 1080),
    batch_size: Number(controls.batchSize.value || 1),
    use_custom_loras: controls.useCustomLoras.checked,
    lora_count: Number(controls.loraCount.value || 0),
    ltx_two_pass_mode: controls.twoPass.checked,
  };
  for (let i = 1; i <= MAX_LORA_SLOTS; i++) {
    payload[`lora_${i}`] = controls.loraSlots[i - 1].select.value;
    payload[`strength_${i}`] = Number(controls.loraSlots[i - 1].strength.value || 1);
  }
  for (const [name, value] of Object.entries(payload)) setWidgetValue(node, name, value);
  return payload;
}

async function queueWorkflowPrompt(prompt) {
  const clientId = api.clientId || app?.clientId || crypto.randomUUID();
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId }),
  });
  const data = await response.json();
  if (!response.ok || data?.error) {
    throw new Error(data?.error?.message || data?.error || `Queue failed: ${response.status}`);
  }
  return data;
}

function showClearMemoryPopup(title = "Clearing memory") {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100004;
    background: rgba(0,0,0,.45);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 18px;
  `;

  const modal = document.createElement("div");
  modal.style.cssText = `
    width: min(620px, 94vw);
    border: 1px solid #155e75;
    border-radius: 8px;
    background: #0f172a;
    color: #e5e7eb;
    box-shadow: 0 24px 80px rgba(0,0,0,.55);
    overflow: hidden;
  `;

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;background:#083344;border-bottom:1px solid #155e75;padding:12px 14px;";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-weight:900;font-size:14px;";
  const close = makeButton("Close");
  close.onclick = () => overlay.remove();
  header.append(heading, close);

  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;";
  const status = document.createElement("div");
  status.textContent = "Building cleanup workflow...";
  status.style.cssText = "white-space:pre-wrap;font-size:12px;line-height:1.45;";
  const barOuter = document.createElement("div");
  barOuter.style.cssText = "height:8px;border-radius:999px;background:#164e63;overflow:hidden;";
  const bar = document.createElement("div");
  bar.style.cssText = "height:100%;width:28%;border-radius:999px;background:#22d3ee;transition:width .25s ease;";
  barOuter.append(bar);
  body.append(status, barOuter);
  modal.append(header, body);
  overlay.append(modal);
  document.body.append(overlay);

  return {
    setStatus(message, percent = 50) {
      status.textContent = message;
      bar.style.width = `${Math.max(5, Math.min(100, percent))}%`;
    },
    closeSoon() {
      setTimeout(() => overlay.remove(), 4500);
    },
  };
}

async function runClearMemoryWorkflow(button) {
  const popup = showClearMemoryPopup();
  try {
    if (button) {
      button.disabled = true;
      button.name = "Clearing...";
    }
    popup.setStatus("Building cleanup workflow...", 20);
    const built = await postJson("/vrgdg/workflow_runner/build_clear_memory_prompt", {});
    popup.setStatus("Queueing cleanup workflow...", 38);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the cleanup workflow but did not return a prompt_id.");
    popup.setStatus(`Cleanup queued.\nPrompt ID: ${promptId}`, 58);
    const text = await waitForText(promptId, (message) => {
      popup.setStatus(`${message}\nPrompt ID: ${promptId}`, 76);
    });
    popup.setStatus(`Cleanup finished.\nPrompt ID: ${promptId}\n\n${text.join("\n\n")}`, 100);
    toast("Memory cleanup workflow finished.");
    popup.closeSoon();
  } catch (error) {
    popup.setStatus(`Error:\n${error.message}`, 100);
    toast(error.message, true);
  } finally {
    if (button) {
      button.disabled = false;
      button.name = "Clear Memory";
    }
  }
}

function syncLoraSlotVisibility(controls) {
  const count = Math.max(0, Math.min(MAX_LORA_SLOTS, Number(controls.loraCount.value || 0)));
  controls.loraSlotContainer.style.display = controls.useCustomLoras.checked && count > 0 ? "grid" : "none";
  controls.loraSlots.forEach((slot, index) => {
    slot.row.style.display = index < count ? "grid" : "none";
  });
}

async function openRunner(node) {
  let loras = ["[none]"];
  try {
    const data = await getJson("/vrgdg/workflow_runner/lora_list");
    loras = data.loras || loras;
  } catch (error) {
    toast(error.message, true);
  }

  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100002;
    background: rgba(0,0,0,.62);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 22px;
  `;

  const modal = document.createElement("div");
  modal.style.cssText = `
    width: min(1180px, 96vw);
    max-height: 92vh;
    display: flex;
    flex-direction: column;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    background: #18181b;
    color: #f4f4f5;
    box-shadow: 0 24px 90px rgba(0,0,0,.55);
    overflow: hidden;
    font-family: sans-serif;
  `;

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid #3f3f46;";
  const title = document.createElement("div");
  title.textContent = "Z-Image Workflow Runner";
  title.style.cssText = "font-size:16px;font-weight:900;";
  const closeButton = makeButton("Close");
  header.append(title, closeButton);

  const body = document.createElement("div");
  body.style.cssText = "overflow:auto;padding:14px 16px;display:flex;flex-direction:column;gap:14px;";

  const workflowPath = makeInput(getWidgetValue(node, "workflow_path", ""));
  const saveFolder = makeInput(getWidgetValue(node, "save_folder", "VRGDG_WorkflowRunner_Saved"));
  const prompt = document.createElement("textarea");
  prompt.value = getWidgetValue(node, "prompt", "");
  prompt.placeholder = "Paste or edit the text-to-image prompt to send into the Z-Image workflow...";
  prompt.style.cssText = "width:100%;min-height:170px;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#101013;color:#fafafa;padding:10px;font-size:12px;line-height:1.45;resize:vertical;";

  const resolutionGrid = makeGrid();
  const firstWidth = makeInput(getWidgetValue(node, "first_pass_width", 1280), "number");
  const firstHeight = makeInput(getWidgetValue(node, "first_pass_height", 720), "number");
  const secondWidth = makeInput(getWidgetValue(node, "second_pass_width", 1920), "number");
  const secondHeight = makeInput(getWidgetValue(node, "second_pass_height", 1080), "number");
  resolutionGrid.append(
    makeField("First pass width", firstWidth),
    makeField("First pass height", firstHeight),
    makeField("Second pass width", secondWidth),
    makeField("Second pass height", secondHeight),
  );

  const miscGrid = makeGrid();
  const batchSize = makeInput(getWidgetValue(node, "batch_size", 1), "number");
  const loraCount = makeInput(getWidgetValue(node, "lora_count", 0), "number");
  loraCount.min = "0";
  loraCount.max = String(MAX_LORA_SLOTS);
  const useCustomLoras = makeCheckbox("Use custom LoRAs", getWidgetValue(node, "use_custom_loras", false));
  const twoPass = makeCheckbox("LTX two-pass LoRA mode", getWidgetValue(node, "ltx_two_pass_mode", false));
  miscGrid.append(makeField("Batch size", batchSize), makeField("LoRA count", loraCount), useCustomLoras.wrapper, twoPass.wrapper);

  const loraSlotContainer = document.createElement("div");
  loraSlotContainer.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
  const loraSlots = [];
  for (let i = 1; i <= MAX_LORA_SLOTS; i++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 100px;gap:8px;align-items:end;border:1px solid #27272a;border-radius:6px;padding:8px;background:#202024;";
    const select = makeSelect(loras, getWidgetValue(node, `lora_${i}`, "[none]"));
    const strength = makeInput(getWidgetValue(node, `strength_${i}`, 1), "number");
    strength.step = "0.01";
    row.append(makeField(`LoRA ${i}`, select), makeField("Strength", strength));
    loraSlotContainer.append(row);
    loraSlots.push({ row, select, strength });
  }

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 16px;border-top:1px solid #3f3f46;";
  const status = document.createElement("div");
  status.textContent = "Ready";
  status.style.cssText = "font-size:12px;color:#a1a1aa;white-space:pre-wrap;";
  const footerButtons = document.createElement("div");
  footerButtons.style.cssText = "display:flex;align-items:center;gap:8px;";
  const saveButton = makeButton("Save Image");
  saveButton.disabled = true;
  saveButton.style.opacity = ".55";
  const runButton = makeButton("Run / Retry Image", "primary");
  footerButtons.append(saveButton, runButton);
  footer.append(status, footerButtons);

  const resultPanel = document.createElement("div");
  resultPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#101013;padding:10px;";
  const resultTitle = document.createElement("div");
  resultTitle.textContent = "Generated image";
  resultTitle.style.cssText = "font-size:12px;color:#d4d4d8;font-weight:900;";
  const resultImage = document.createElement("img");
  resultImage.alt = "";
  resultImage.style.cssText = "display:none;width:100%;max-height:520px;object-fit:contain;background:#050505;border-radius:6px;";
  const resultEmpty = document.createElement("div");
  resultEmpty.textContent = "Run the image workflow to preview the result here.";
  resultEmpty.style.cssText = "min-height:160px;display:flex;align-items:center;justify-content:center;color:#71717a;font-size:12px;border:1px dashed #3f3f46;border-radius:6px;";
  resultPanel.append(resultTitle, resultImage, resultEmpty);

  body.append(
    makeField("Workflow template", workflowPath),
    makeField("Save approved images to", saveFolder),
    makeField("Positive prompt", prompt),
    resultPanel,
    resolutionGrid,
    miscGrid,
    loraSlotContainer,
  );
  modal.append(header, body, footer);
  overlay.append(modal);
  document.body.append(overlay);

  const controls = {
    workflowPath,
    saveFolder,
    prompt,
    firstWidth,
    firstHeight,
    secondWidth,
    secondHeight,
    batchSize,
    useCustomLoras: useCustomLoras.input,
    loraCount,
    twoPass: twoPass.input,
    loraSlotContainer,
    loraSlots,
  };
  let currentImage = null;
  syncLoraSlotVisibility(controls);

  closeButton.onclick = () => overlay.remove();
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) overlay.remove();
  });
  useCustomLoras.input.onchange = () => syncLoraSlotVisibility(controls);
  loraCount.oninput = () => syncLoraSlotVisibility(controls);

  runButton.onclick = async () => {
    try {
      runButton.disabled = true;
      saveButton.disabled = true;
      saveButton.style.opacity = ".55";
      currentImage = null;
      resultImage.style.display = "none";
      resultEmpty.style.display = "flex";
      resultEmpty.textContent = "Queued image workflow. Waiting for result...";
      status.textContent = "Building workflow prompt...";
      const payload = collectPayload(node, controls);
      const built = await postJson("/vrgdg/workflow_runner/build_zimage_prompt", payload);
      status.textContent = "Queueing image workflow...";
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the workflow but did not return a prompt_id.");
      status.textContent = `Queued image workflow: ${promptId}`;
      const images = await waitForImages(promptId, (message) => {
        status.textContent = `${message}\nPrompt ID: ${promptId}`;
      });
      currentImage = images[images.length - 1];
      resultImage.src = makeImageViewUrl(currentImage);
      resultImage.style.display = "block";
      resultEmpty.style.display = "none";
      saveButton.disabled = false;
      saveButton.style.opacity = "1";
      status.textContent = `Image ready.\nPrompt ID: ${promptId}`;
      toast("Generated image is ready to review.");
    } catch (error) {
      status.textContent = error.message;
      resultEmpty.textContent = error.message;
      resultEmpty.style.display = "flex";
      toast(error.message, true);
    } finally {
      runButton.disabled = false;
    }
  };

  saveButton.onclick = async () => {
    if (!currentImage) return;
    try {
      saveButton.disabled = true;
      status.textContent = "Saving approved image...";
      const result = await postJson("/vrgdg/workflow_runner/save_image", {
        image: currentImage,
        save_folder: saveFolder.value.trim(),
      });
      setWidgetValue(node, "save_folder", saveFolder.value.trim());
      status.textContent = `Saved image:\n${result.saved_path}`;
      toast(`Saved approved image:\n${result.saved_path}`);
    } catch (error) {
      status.textContent = error.message;
      toast(error.message, true);
    } finally {
      saveButton.disabled = false;
    }
  };
}

app.registerExtension({
  name: "vrgdg.workflow.runner.ui",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === CLEAR_MEMORY_NODE_NAME) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = onNodeCreated?.apply(this, args);
        this.addWidget("button", "Clear Memory", null, () => runClearMemoryWorkflow(getWidget(this, "Clear Memory")));
        this.setSize?.([260, 78]);
        return result;
      };
      return;
    }

    if (nodeData?.name !== NODE_NAME) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);
      this.addWidget("button", "Open Z-Image Runner", null, () => openRunner(this));
      hideInternalWidgets(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      setTimeout(() => hideInternalWidgets(this), 0);
      return result;
    };
  },
});
