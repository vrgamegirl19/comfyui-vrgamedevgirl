import { app } from "../../scripts/app.js";

const STORAGE_KEY = "vrgdg_node_canvas_prototype_v1";
const COMFY_NODE_NAME = "VRGDG_VideoBuilderNodeCanvas";

const NODE_DEFS = {
  prompt: {
    title: "Prompt",
    color: "#f1c40f",
    inputs: [],
    outputs: [{ name: "Prompt", type: "prompt" }],
    width: 330,
    height: 230,
  },
  imageRef: {
    title: "Image Ref",
    color: "#35c47d",
    inputs: [],
    outputs: [{ name: "Image", type: "image" }],
    width: 280,
    height: 250,
  },
  mode: {
    title: "Mode",
    color: "#5aa2ff",
    inputs: [],
    outputs: [{ name: "Mode", type: "mode" }],
    width: 260,
    height: 170,
  },
  sceneCard: {
    title: "Scene Card",
    color: "#ff7a4d",
    inputs: [
      { name: "Prompt", type: "prompt" },
      { name: "Refs", type: "image" },
      { name: "Mode", type: "mode" },
    ],
    outputs: [{ name: "Scene", type: "scene" }],
    width: 360,
    height: 400,
  },
};

const DEFAULT_GRAPH = {
  nextId: 5,
  nodes: [
    {
      id: 1,
      type: "prompt",
      x: 150,
      y: 260,
      data: { prompt: "A cinematic shot of waves crashing on a beach" },
    },
    {
      id: 2,
      type: "mode",
      x: 160,
      y: 80,
      data: { mode: "Existing Workflow Mode" },
    },
    {
      id: 3,
      type: "imageRef",
      x: 540,
      y: 90,
      data: { label: "Drop image here", imageName: "", imageData: "" },
    },
    {
      id: 4,
      type: "sceneCard",
      x: 640,
      y: 270,
      data: { title: "Scene Card Prototype" },
    },
  ],
  links: [
    { from: 1, fromPort: 0, to: 4, toPort: 0 },
    { from: 2, fromPort: 0, to: 4, toPort: 2 },
    { from: 3, fromPort: 0, to: 4, toPort: 1 },
  ],
};

function cloneGraph(graph) {
  return JSON.parse(JSON.stringify(graph));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

class VRGDGNodeCanvasPrototype {
  constructor() {
    this.graph = this.loadGraph();
    this.root = null;
    this.canvas = null;
    this.stage = null;
    this.linksSvg = null;
    this.contextMenu = null;
    this.drag = null;
    this.pan = { x: 0, y: 0 };
    this.tempLink = null;
    this.selectedNodeId = null;
  }

  init() {
    this.injectStyles();
    window.vrgdgNodeCanvasPrototype = this;
  }

  loadGraph() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch (error) {
      console.warn("[VRGDG Node Canvas] Failed to load saved prototype graph", error);
    }
    return cloneGraph(DEFAULT_GRAPH);
  }

  saveGraph() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.graph));
  }

  resetGraph() {
    this.graph = cloneGraph(DEFAULT_GRAPH);
    this.saveGraph();
    this.render();
  }

  open() {
    if (this.root) {
      this.root.classList.remove("hidden");
      this.render();
      return;
    }

    this.root = el("div", "vrgdg-node-root");
    this.root.innerHTML = `
      <div class="vrgdg-node-topbar">
        <div class="vrgdg-node-brand">
          <span>VRGDG Node Canvas</span>
          <small>standalone prototype - not connected to Video Builder</small>
        </div>
        <div class="vrgdg-node-actions">
          <button data-action="add-prompt">Prompt</button>
          <button data-action="add-image">Image Ref</button>
          <button data-action="add-mode">Mode</button>
          <button data-action="add-scene">Scene Card</button>
          <button data-action="reset">Reset</button>
          <button data-action="close">Close</button>
        </div>
      </div>
      <div class="vrgdg-node-canvas">
        <svg class="vrgdg-node-links"></svg>
        <div class="vrgdg-node-stage"></div>
        <div class="vrgdg-node-hint">Right click the canvas to add nodes. Drag from colored ports to connect.</div>
      </div>
    `;

    document.body.appendChild(this.root);
    this.canvas = this.root.querySelector(".vrgdg-node-canvas");
    this.stage = this.root.querySelector(".vrgdg-node-stage");
    this.linksSvg = this.root.querySelector(".vrgdg-node-links");

    this.root.querySelector("[data-action='close']").addEventListener("click", () => this.close());
    this.root.querySelector("[data-action='reset']").addEventListener("click", () => this.resetGraph());
    this.root.querySelector("[data-action='add-prompt']").addEventListener("click", () => this.addNode("prompt"));
    this.root.querySelector("[data-action='add-image']").addEventListener("click", () => this.addNode("imageRef"));
    this.root.querySelector("[data-action='add-mode']").addEventListener("click", () => this.addNode("mode"));
    this.root.querySelector("[data-action='add-scene']").addEventListener("click", () => this.addNode("sceneCard"));

    this.canvas.addEventListener("contextmenu", (event) => this.showContextMenu(event));
    this.canvas.addEventListener("pointerdown", (event) => {
      if (event.target === this.canvas || event.target === this.linksSvg) {
        this.selectedNodeId = null;
        this.hideContextMenu();
      }
    });
    window.addEventListener("pointermove", (event) => this.onPointerMove(event));
    window.addEventListener("pointerup", () => this.onPointerUp());
    window.addEventListener("keydown", (event) => this.onKeyDown(event));

    this.render();
  }

  close() {
    this.root?.classList.add("hidden");
  }

  addNode(type, x, y) {
    const def = NODE_DEFS[type];
    if (!def) return;

    const node = {
      id: this.graph.nextId++,
      type,
      x: x ?? 220 + this.graph.nodes.length * 24,
      y: y ?? 140 + this.graph.nodes.length * 18,
      data: {},
    };

    if (type === "prompt") node.data.prompt = "";
    if (type === "imageRef") node.data = { label: "Drop image here", imageName: "", imageData: "" };
    if (type === "mode") node.data.mode = "Existing Workflow Mode";
    if (type === "sceneCard") node.data.title = "Scene Card";

    this.graph.nodes.push(node);
    this.selectedNodeId = node.id;
    this.saveGraph();
    this.render();
  }

  deleteSelectedNode() {
    if (!this.selectedNodeId) return;
    this.graph.nodes = this.graph.nodes.filter((node) => node.id !== this.selectedNodeId);
    this.graph.links = this.graph.links.filter(
      (link) => link.from !== this.selectedNodeId && link.to !== this.selectedNodeId
    );
    this.selectedNodeId = null;
    this.saveGraph();
    this.render();
  }

  onKeyDown(event) {
    if (!this.root || this.root.classList.contains("hidden")) return;
    if (event.key === "Escape") this.close();
    if ((event.key === "Delete" || event.key === "Backspace") && this.selectedNodeId) {
      event.preventDefault();
      this.deleteSelectedNode();
    }
  }

  showContextMenu(event) {
    event.preventDefault();
    this.hideContextMenu();
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const menu = el("div", "vrgdg-node-menu");
    menu.innerHTML = `
      <button data-type="prompt">Prompt Node</button>
      <button data-type="imageRef">Image Ref Node</button>
      <button data-type="mode">Mode Node</button>
      <button data-type="sceneCard">Scene Card Node</button>
    `;
    menu.style.left = `${event.clientX}px`;
    menu.style.top = `${event.clientY}px`;
    menu.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        this.addNode(button.dataset.type, x, y);
        this.hideContextMenu();
      });
    });
    document.body.appendChild(menu);
    this.contextMenu = menu;
  }

  hideContextMenu() {
    this.contextMenu?.remove();
    this.contextMenu = null;
  }

  render() {
    if (!this.stage) return;
    this.stage.innerHTML = "";
    this.linksSvg.innerHTML = "";
    for (const node of this.graph.nodes) {
      this.stage.appendChild(this.renderNode(node));
    }
    this.renderLinks();
  }

  renderNode(node) {
    const def = NODE_DEFS[node.type];
    const card = el("div", `vrgdg-node-card ${node.type === "sceneCard" ? "scene" : ""}`);
    card.dataset.nodeId = String(node.id);
    card.style.left = `${node.x}px`;
    card.style.top = `${node.y}px`;
    card.style.width = `${def.width}px`;
    card.style.minHeight = `${def.height}px`;
    card.style.setProperty("--node-color", def.color);
    if (this.selectedNodeId === node.id) card.classList.add("selected");

    const header = el("div", "vrgdg-node-header");
    header.innerHTML = `<span>${def.title}</span><small>#${node.id}</small>`;
    header.addEventListener("pointerdown", (event) => this.startNodeDrag(event, node));
    card.appendChild(header);

    if (def.inputs.length) {
      const inputs = el("div", "vrgdg-node-ports inputs");
      def.inputs.forEach((port, index) => inputs.appendChild(this.renderPort(node, port, index, "input")));
      card.appendChild(inputs);
    }

    const body = el("div", "vrgdg-node-body");
    this.renderBody(node, body);
    card.appendChild(body);

    if (def.outputs.length) {
      const outputs = el("div", "vrgdg-node-ports outputs");
      def.outputs.forEach((port, index) => outputs.appendChild(this.renderPort(node, port, index, "output")));
      card.appendChild(outputs);
    }

    card.addEventListener("pointerdown", () => {
      this.selectedNodeId = node.id;
      this.render();
    });

    return card;
  }

  renderBody(node, body) {
    if (node.type === "prompt") {
      const textarea = el("textarea", "vrgdg-node-textarea");
      textarea.placeholder = "Write prompt text...";
      textarea.value = node.data.prompt || "";
      textarea.addEventListener("input", () => {
        node.data.prompt = textarea.value;
        this.saveGraph();
        this.renderLinks();
      });
      body.appendChild(textarea);
      return;
    }

    if (node.type === "mode") {
      const select = el("select", "vrgdg-node-select");
      ["Existing Workflow Mode", "Flux / Image", "Nano / Image", "Flow GPT / Prompt", "Wan / Video"].forEach((mode) => {
        const option = el("option", "", mode);
        option.value = mode;
        select.appendChild(option);
      });
      select.value = node.data.mode || "Existing Workflow Mode";
      select.addEventListener("change", () => {
        node.data.mode = select.value;
        this.saveGraph();
        this.renderLinks();
      });
      body.appendChild(select);
      return;
    }

    if (node.type === "imageRef") {
      const drop = el("div", "vrgdg-node-drop");
      drop.textContent = node.data.imageName || node.data.label || "Drop image here";
      if (node.data.imageData) {
        const img = el("img", "vrgdg-node-preview");
        img.src = node.data.imageData;
        drop.textContent = "";
        drop.appendChild(img);
        const caption = el("span", "vrgdg-node-caption", node.data.imageName || "Image ref");
        drop.appendChild(caption);
      }
      drop.addEventListener("dragover", (event) => {
        event.preventDefault();
        drop.classList.add("dragging");
      });
      drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
      drop.addEventListener("drop", (event) => this.handleImageDrop(event, node));
      body.appendChild(drop);
      return;
    }

    if (node.type === "sceneCard") {
      const resolved = this.resolveSceneCard(node);
      const title = el("input", "vrgdg-node-input");
      title.value = node.data.title || "Scene Card";
      title.addEventListener("input", () => {
        node.data.title = title.value;
        this.saveGraph();
      });
      body.appendChild(title);

      const preview = el("div", "vrgdg-node-scene-preview");
      if (resolved.imageData) {
        const img = el("img", "vrgdg-scene-image");
        img.src = resolved.imageData;
        preview.appendChild(img);
      } else {
        preview.textContent = "Scene preview";
      }
      body.appendChild(preview);

      body.appendChild(this.sceneField("Mode", resolved.mode || "No mode connected"));
      body.appendChild(this.sceneField("Prompt", resolved.prompt || "No prompt connected"));
      body.appendChild(this.sceneField("Refs", resolved.refs.length ? resolved.refs.join(", ") : "No refs connected"));
      return;
    }
  }

  sceneField(label, value) {
    const wrap = el("div", "vrgdg-scene-field");
    wrap.appendChild(el("label", "", label));
    wrap.appendChild(el("div", "", value));
    return wrap;
  }

  handleImageDrop(event, node) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      node.data.imageName = file.name;
      node.data.imageData = String(reader.result || "");
      this.saveGraph();
      this.render();
    };
    reader.readAsDataURL(file);
  }

  renderPort(node, port, index, direction) {
    const wrap = el("div", "vrgdg-node-port");
    wrap.dataset.nodeId = String(node.id);
    wrap.dataset.portIndex = String(index);
    wrap.dataset.direction = direction;
    wrap.dataset.portType = port.type;
    wrap.innerHTML = direction === "input"
      ? `<i></i><span>${port.name}</span>`
      : `<span>${port.name}</span><i></i>`;
    wrap.addEventListener("pointerdown", (event) => this.startLink(event, node, index, direction, port.type));
    return wrap;
  }

  startNodeDrag(event, node) {
    event.preventDefault();
    this.selectedNodeId = node.id;
    this.drag = {
      node,
      startX: event.clientX,
      startY: event.clientY,
      nodeX: node.x,
      nodeY: node.y,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    this.render();
  }

  startLink(event, node, portIndex, direction, type) {
    event.preventDefault();
    event.stopPropagation();
    this.tempLink = {
      nodeId: node.id,
      portIndex,
      direction,
      type,
      x: event.clientX,
      y: event.clientY,
    };
  }

  onPointerMove(event) {
    if (this.drag) {
      const dx = event.clientX - this.drag.startX;
      const dy = event.clientY - this.drag.startY;
      this.drag.node.x = Math.round(this.drag.nodeX + dx);
      this.drag.node.y = Math.round(this.drag.nodeY + dy);
      this.render();
      return;
    }

    if (this.tempLink) {
      this.tempLink.x = event.clientX;
      this.tempLink.y = event.clientY;
      this.renderLinks();
    }
  }

  onPointerUp() {
    if (this.drag) {
      this.saveGraph();
      this.drag = null;
    }

    if (this.tempLink) {
      const target = document.elementFromPoint(this.tempLink.x, this.tempLink.y)?.closest?.(".vrgdg-node-port");
      if (target) this.finishLink(target);
      this.tempLink = null;
      this.render();
    }
  }

  finishLink(target) {
    const targetNodeId = Number(target.dataset.nodeId);
    const targetPortIndex = Number(target.dataset.portIndex);
    const targetDirection = target.dataset.direction;
    const targetType = target.dataset.portType;
    const source = this.tempLink;
    if (!source || targetNodeId === source.nodeId || targetDirection === source.direction) return;
    if (targetType !== source.type) return;

    const link = source.direction === "output"
      ? { from: source.nodeId, fromPort: source.portIndex, to: targetNodeId, toPort: targetPortIndex }
      : { from: targetNodeId, fromPort: targetPortIndex, to: source.nodeId, toPort: source.portIndex };

    this.graph.links = this.graph.links.filter((existing) => {
      return !(existing.to === link.to && existing.toPort === link.toPort);
    });
    this.graph.links.push(link);
    this.saveGraph();
  }

  getPortCenter(nodeId, portIndex, direction) {
    const port = this.stage.querySelector(
      `.vrgdg-node-port[data-node-id='${nodeId}'][data-port-index='${portIndex}'][data-direction='${direction}'] i`
    );
    if (!port) return null;
    const canvasRect = this.canvas.getBoundingClientRect();
    const rect = port.getBoundingClientRect();
    return {
      x: rect.left + rect.width / 2 - canvasRect.left,
      y: rect.top + rect.height / 2 - canvasRect.top,
    };
  }

  renderLinks() {
    if (!this.linksSvg || !this.canvas) return;
    this.linksSvg.innerHTML = "";
    const rect = this.canvas.getBoundingClientRect();
    this.linksSvg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);

    for (const link of this.graph.links) {
      const from = this.getPortCenter(link.from, link.fromPort, "output");
      const to = this.getPortCenter(link.to, link.toPort, "input");
      if (from && to) this.linksSvg.appendChild(this.linkPath(from, to, "#e5b900"));
    }

    if (this.tempLink) {
      const start = this.getPortCenter(this.tempLink.nodeId, this.tempLink.portIndex, this.tempLink.direction);
      if (start) {
        const canvasRect = this.canvas.getBoundingClientRect();
        const end = {
          x: clamp(this.tempLink.x - canvasRect.left, 0, canvasRect.width),
          y: clamp(this.tempLink.y - canvasRect.top, 0, canvasRect.height),
        };
        const from = this.tempLink.direction === "output" ? start : end;
        const to = this.tempLink.direction === "output" ? end : start;
        this.linksSvg.appendChild(this.linkPath(from, to, "#ffffff", true));
      }
    }
  }

  linkPath(from, to, color, dashed) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const dx = Math.max(80, Math.abs(to.x - from.x) * 0.5);
    path.setAttribute("d", `M ${from.x} ${from.y} C ${from.x + dx} ${from.y}, ${to.x - dx} ${to.y}, ${to.x} ${to.y}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", "3");
    path.setAttribute("stroke-linecap", "round");
    if (dashed) path.setAttribute("stroke-dasharray", "8 8");
    return path;
  }

  resolveSceneCard(sceneNode) {
    const incoming = this.graph.links.filter((link) => link.to === sceneNode.id);
    const resolved = { prompt: "", mode: "", refs: [], imageData: "" };
    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (!source) continue;
      if (source.type === "prompt") resolved.prompt = source.data.prompt || "";
      if (source.type === "mode") resolved.mode = source.data.mode || "";
      if (source.type === "imageRef") {
        resolved.refs.push(source.data.imageName || "Image ref");
        if (!resolved.imageData) resolved.imageData = source.data.imageData || "";
      }
    }
    return resolved;
  }

  injectStyles() {
    if (document.getElementById("vrgdg-node-canvas-styles")) return;
    const style = el("style");
    style.id = "vrgdg-node-canvas-styles";
    style.textContent = `
      .vrgdg-node-root {
        position: fixed;
        inset: 0;
        z-index: 8999;
        display: flex;
        flex-direction: column;
        background: #0d0f12;
        color: #e8ebef;
        font-family: Arial, Helvetica, sans-serif;
      }

      .vrgdg-node-root.hidden { display: none; }

      .vrgdg-node-topbar {
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 0 18px;
        border-bottom: 1px solid rgba(255,255,255,0.09);
        background: #15181d;
      }

      .vrgdg-node-brand {
        display: flex;
        flex-direction: column;
        gap: 3px;
      }

      .vrgdg-node-brand span {
        font-size: 16px;
        font-weight: 800;
      }

      .vrgdg-node-brand small {
        color: #9da6b2;
        font-size: 12px;
      }

      .vrgdg-node-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .vrgdg-node-actions button,
      .vrgdg-node-menu button {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 7px;
        background: #252a31;
        color: #f3f6f8;
        padding: 8px 11px;
        font: 700 12px Arial, sans-serif;
        cursor: pointer;
      }

      .vrgdg-node-actions button:hover,
      .vrgdg-node-menu button:hover {
        background: #303740;
      }

      .vrgdg-node-canvas {
        position: relative;
        flex: 1;
        overflow: hidden;
        background-color: #0b0d10;
        background-image: radial-gradient(rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 24px 24px;
      }

      .vrgdg-node-stage,
      .vrgdg-node-links {
        position: absolute;
        inset: 0;
      }

      .vrgdg-node-links {
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
      }

      .vrgdg-node-stage {
        z-index: 2;
      }

      .vrgdg-node-card {
        position: absolute;
        border-radius: 10px;
        background: #202328;
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        overflow: visible;
      }

      .vrgdg-node-card.selected {
        outline: 2px solid rgba(255,255,255,0.34);
      }

      .vrgdg-node-card.scene {
        background: #1b1e22;
      }

      .vrgdg-node-header {
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 14px;
        border-radius: 10px 10px 0 0;
        background: color-mix(in srgb, var(--node-color) 18%, #202328);
        cursor: grab;
        user-select: none;
      }

      .vrgdg-node-header span {
        font-size: 14px;
        font-weight: 800;
      }

      .vrgdg-node-header small {
        color: #aab2bd;
        font-size: 11px;
      }

      .vrgdg-node-body {
        padding: 12px 14px 14px;
      }

      .vrgdg-node-textarea {
        width: 100%;
        height: 145px;
        box-sizing: border-box;
        resize: none;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 7px;
        background: #111316;
        color: #f5f6f7;
        padding: 10px;
        font: 14px/1.35 Arial, sans-serif;
      }

      .vrgdg-node-select,
      .vrgdg-node-input {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 7px;
        background: #111316;
        color: #f5f6f7;
        padding: 9px 10px;
        font: 13px Arial, sans-serif;
      }

      .vrgdg-node-drop {
        height: 165px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        border: 1px dashed rgba(255,255,255,0.24);
        border-radius: 8px;
        background: #111316;
        color: #aeb7c2;
        text-align: center;
        overflow: hidden;
      }

      .vrgdg-node-drop.dragging {
        border-color: #35c47d;
        background: #132019;
      }

      .vrgdg-node-preview,
      .vrgdg-scene-image {
        max-width: 100%;
        max-height: 132px;
        object-fit: contain;
        display: block;
      }

      .vrgdg-node-caption {
        max-width: 92%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: #d7dde4;
        font-size: 12px;
      }

      .vrgdg-node-ports {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 8px 0;
      }

      .vrgdg-node-port {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #adb6c0;
        font-size: 12px;
        user-select: none;
      }

      .vrgdg-node-ports.inputs .vrgdg-node-port {
        margin-left: -9px;
      }

      .vrgdg-node-ports.outputs .vrgdg-node-port {
        justify-content: flex-end;
        margin-right: -9px;
      }

      .vrgdg-node-port i {
        width: 14px;
        height: 14px;
        display: inline-block;
        border-radius: 50%;
        background: var(--node-color);
        border: 3px solid #202328;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.16);
        cursor: crosshair;
      }

      .vrgdg-node-scene-preview {
        height: 138px;
        margin: 11px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: #101216;
        color: #6f7782;
        overflow: hidden;
      }

      .vrgdg-scene-field {
        margin-top: 9px;
      }

      .vrgdg-scene-field label {
        display: block;
        margin-bottom: 4px;
        color: #9099a5;
        font-size: 11px;
        text-transform: uppercase;
      }

      .vrgdg-scene-field div {
        min-height: 28px;
        max-height: 76px;
        overflow: auto;
        border-radius: 6px;
        background: #12151a;
        color: #dce2e8;
        padding: 8px;
        font-size: 12px;
        line-height: 1.35;
      }

      .vrgdg-node-hint {
        position: absolute;
        left: 18px;
        bottom: 18px;
        z-index: 3;
        color: #8c96a3;
        font-size: 12px;
        background: rgba(12,14,17,0.72);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 7px;
        padding: 8px 10px;
        pointer-events: none;
      }

      .vrgdg-node-menu {
        position: fixed;
        z-index: 10001;
        display: flex;
        flex-direction: column;
        gap: 5px;
        min-width: 180px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        background: #171a1f;
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
      }
    `;
    document.head.appendChild(style);
  }
}

app.registerExtension({
  name: "VRGDG.VideoBuilderNodeCanvasPrototype",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== COMFY_NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;

    function ensureOpenButton(node) {
      const buttonName = "Open Node Canvas";
      node.widgets = (node.widgets || []).filter(
        (widget) => !(widget?.type === "button" && widget?.name === buttonName)
      );
      const widget = node.addWidget("button", buttonName, null, () => {
        window.vrgdgNodeCanvasPrototype?.open();
      });
      if (widget) widget.serialize = false;
      node.size = [
        Math.max(node.size?.[0] || 320, 320),
        Math.max(node.size?.[1] || 120, 120),
      ];
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      ensureOpenButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      ensureOpenButton(this);
      return result;
    };
  },

  async setup() {
    const prototype = new VRGDGNodeCanvasPrototype();
    prototype.init();
    console.log("[VRGDG] Node canvas prototype loaded");
  },
});
