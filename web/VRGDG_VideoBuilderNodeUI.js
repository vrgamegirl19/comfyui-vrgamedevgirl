import { app } from "../../scripts/app.js";

const STORAGE_KEY = "vrgdg_node_canvas_prototype_v1";
const COMFY_NODE_NAME = "VRGDG_VideoBuilderNodeCanvas";
const DEFAULT_I2V_PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0";
const DEFAULT_I2V_PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0";
const I2V_SAMPLER_OPTIONS = ["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde"];
const I2V_TAB_ALIASES = {
  model: "models",
  prompt: "llm_prompting",
  motion: "video_settings",
  output: "llm_prompting",
};

function defaultI2VNodeData() {
  return {
    tab: "models",
    use_scene_i2v_video_settings: false,
    use_gguf_model: true,
    unet_name: "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
    diffusion_model_name: "LTX_8bit\\ltx-2.3-22b-dev_transformer_only_int8_convrot.safetensors",
    vae_name: "LTX23_video_vae_bf16.safetensors",
    clip_name1: "gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors",
    clip_name2: "ltx-2.3_text_projection_bf16.safetensors",
    upscale_model_name: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    audio_vae_name: "LTX23_audio_vae_bf16.safetensors",
    text_gemma_model: "",
    vision_gemma_model: "",
    mmproj_file: "",
    use_loras: false,
    lora_count: 0,
    msr_lora_name: "licon\\LTX-2.3-Licon-MSR-V1.safetensors",
    ingredients_lora_name: "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
    id_lora_name: "lora_weights.safetensors",
    fps: 24,
    width: 1920,
    height: 1080,
    seed: 69,
    video_trigger_phrase: "",
    tail_loss_frames: 25,
    pre_frames: 50,
    pass1_sampler: "euler_ancestral",
    pass1_sigmas: DEFAULT_I2V_PASS1_SIGMAS,
    pass1_strength: 1,
    pass1_bypass: false,
    pass2_sampler: "euler_ancestral",
    pass2_sigmas: DEFAULT_I2V_PASS2_SIGMAS,
    pass2_strength: 1,
    pass2_bypass: false,
    use_i2v_prompt_enhancement_pass: false,
    i2v_notes: "",
    use_i2v_vision_reference: true,
    i2v_prompt: "",
  };
}

function defaultImageModeNodeData() {
  return {
    image_model_mode: "zimage",
    tab: "models",
    imageData: "",
    imageName: "",
    zimage_model: "",
    zimage_clip: "",
    zimage_vae: "",
    zimage_text_gemma_model: "",
    zimage_vision_gemma_model: "",
    zimage_mmproj: "",
    zimage_use_lora: false,
    zimage_lora_count: 0,
    zimage_trigger_phrase: "",
    zimage_batch_size: 1,
    zimage_use_image_to_image: false,
    zimage_notes: "",
    zimage_use_vision_reference: true,
    zimage_prompt: "",
    ernie_model: "",
    ernie_clip: "",
    ernie_vae: "",
    ernie_text_gemma_model: "",
    ernie_vision_gemma_model: "",
    ernie_mmproj: "",
    ernie_use_lora: false,
    ernie_lora_count: 0,
    ernie_trigger_phrase: "",
    ernie_batch_size: 1,
    ernie_use_image_to_image: false,
    ernie_notes: "",
    ernie_use_vision_reference: true,
    ernie_prompt: "",
    krea2_model: "",
    krea2_clip: "",
    krea2_vae: "",
    krea2_text_gemma_model: "",
    krea2_vision_gemma_model: "",
    krea2_mmproj: "",
    krea2_use_lora: false,
    krea2_lora_count: 0,
    krea2_trigger_phrase: "",
    krea2_use_image_to_image: false,
    krea2_notes: "",
    krea2_use_vision_reference: true,
    krea2_prompt: "",
    flux_model: "",
    flux_clip: "",
    flux_vae: "",
    flux_vision_gemma_model: "",
    flux_mmproj: "",
    flux_use_lora: false,
    flux_lora_count: 0,
    flux_use_text_only_gemma_prompt: false,
    flux_use_director_notes: true,
    flux_trigger_phrase: "",
    flux_reference_images: "",
    flux_notes: "",
    flux_prompt: "",
    nano_api_key: "",
    nano_model: "gemini-2.5-flash-image-preview",
    nano_vision_gemma_model: "",
    nano_mmproj: "",
    nano_use_global_ingredients: true,
    nano_use_text_only_gemma_prompt: false,
    nano_use_director_notes: true,
    nano_ingredients: "",
    nano_notes: "",
    nano_prompt: "",
    flow_provider: "chatgpt",
    flow_aspect_ratio: "16:9",
    flow_timeout_seconds: 300,
    flow_retries: 1,
    flow_failure_mode: "pause",
    flow_prompt: "",
    flow_manual_mode: false,
    flow_manual_auto_advance: true,
    zenhance_model: "",
    zenhance_clip: "",
    zenhance_vae: "",
    zenhance_vision_gemma_model: "",
    zenhance_mmproj: "",
    zenhance_use_lora: false,
    zenhance_lora_count: 0,
    zenhance_amount: 0.45,
    zenhance_notes: "",
    zenhance_prompt: "",
  };
}

function defaultSceneCardData() {
  return {
    tab: "scene",
    title: "Scene 01",
    scene_number: "01",
    duration: 5,
    lyrics: "",
    story_beat: "A cinematic shot of waves crashing on a beach",
    subject_notes: "",
    setting_notes: "beach, ocean waves, cinematic light",
    image_prompt: "A cinematic shot of waves crashing on a beach",
    video_prompt: "Ocean waves crash onto the beach with natural rolling motion.",
    motion_notes: "Slow camera push, natural wave motion, soft wind.",
  };
}

function defaultStoryDefaultsNodeData() {
  return {
    tab: "scene_defaults",
    still_shot_flow: "Intimate character shots",
    image_aesthetic: "Default cinematic still",
    global_consistency_phrase: "",
    global_performance_style: "Default cinematic",
    global_facial_performance: "Default natural",
    custom_facial_text: "",
    use_in_gemma_prompts: true,
    lyric_story_strength: 7,
    user_story_arc: "",
    song_story_brief: "",
  };
}

function defaultLoadImageNodeData() {
  return {
    label: "Load Image",
    imageName: "",
    imageData: "",
    source_mode: "ComfyUI input/upload",
    image_role: "start image",
    fit_mode: "contain",
    use_as_vision_reference: true,
    notes: "",
  };
}

function defaultLoadAudioNodeData() {
  return {
    label: "Load Audio",
    audioName: "",
    audioUrl: "",
    source_mode: "ComfyUI input/upload",
    sample_rate: "keep source",
    channel_mode: "stereo",
    normalize_audio: false,
    trim_start_seconds: 0,
    trim_end_seconds: 0,
    notes: "",
  };
}

function defaultTranscriptionNodeData() {
  return {
    engine: "Whisper / stable-ts",
    model: "large-v3",
    language: "auto",
    device: "cuda",
    compute_type: "float16",
    use_vad: true,
    word_timestamps: true,
    regroup_words: true,
    max_line_width: 42,
    srt_output_name: "",
    transcript_preview: "",
  };
}

function defaultLyricMappingNodeData() {
  return {
    mapping_mode: "lyrics to scenes",
    split_mode: "detect sections",
    scene_count: 8,
    timing_offset_seconds: 0,
    beat_padding_seconds: 0.25,
    attach_to_scene_cards: true,
    use_story_layer: true,
    full_lyrics: "",
    mapping_notes: "",
    map_preview: "",
  };
}

const NODE_DEFS = {
  storyDefaults: {
    title: "Story Defaults",
    color: "#16c6d8",
    inputs: [],
    outputs: [{ name: "Story Context", type: "story" }],
    width: 430,
    height: 470,
  },
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
  loadImage: {
    title: "Load Image",
    color: "#35c47d",
    inputs: [],
    outputs: [{ name: "Image", type: "image" }],
    width: 310,
    height: 360,
  },
  loadAudio: {
    title: "Load Audio",
    color: "#f59e0b",
    inputs: [],
    outputs: [{ name: "Audio", type: "audio" }],
    width: 330,
    height: 390,
  },
  transcription: {
    title: "Transcription",
    color: "#60a5fa",
    inputs: [{ name: "Audio", type: "audio" }],
    outputs: [
      { name: "Transcript", type: "transcript" },
      { name: "SRT", type: "srt" },
    ],
    width: 380,
    height: 430,
  },
  lyricMapping: {
    title: "Lyric Mapping",
    color: "#f472b6",
    inputs: [
      { name: "Transcript", type: "transcript" },
      { name: "Story Context", type: "story" },
    ],
    outputs: [
      { name: "Lyric Map", type: "lyric_map" },
      { name: "Scene Beats", type: "scene_beats" },
    ],
    width: 400,
    height: 470,
  },
  imageMode: {
    title: "Image Mode",
    color: "#18c2d6",
    inputs: [
      { name: "Scene Context", type: "scene" },
      { name: "Prompt Override", type: "prompt" },
    ],
    outputs: [{ name: "Image", type: "image" }],
    width: 410,
    height: 500,
  },
  imageToVideo: {
    title: "Image to Video",
    color: "#b56cff",
    inputs: [
      { name: "Start Image", type: "image" },
      { name: "Scene Context", type: "scene" },
      { name: "Prompt Override", type: "prompt" },
    ],
    outputs: [{ name: "Video", type: "video" }],
    width: 390,
    height: 470,
  },
  displayVideo: {
    title: "Display Video",
    color: "#35b6ff",
    inputs: [{ name: "Video", type: "video" }],
    outputs: [],
    width: 340,
    height: 300,
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
      { name: "Story Context", type: "story" },
      { name: "Lyric Map", type: "lyric_map" },
    ],
    outputs: [{ name: "Scene Context", type: "scene" }],
    width: 390,
    height: 500,
  },
};

const DEFAULT_GRAPH = {
  nextId: 9,
  nodes: [
    {
      id: 1,
      type: "storyDefaults",
      x: 100,
      y: 60,
      data: defaultStoryDefaultsNodeData(),
    },
    {
      id: 2,
      type: "sceneCard",
      x: 580,
      y: 80,
      data: defaultSceneCardData(),
    },
    {
      id: 3,
      type: "imageMode",
      x: 1030,
      y: 40,
      data: defaultImageModeNodeData(),
    },
    {
      id: 4,
      type: "imageToVideo",
      x: 1490,
      y: 110,
      data: defaultI2VNodeData(),
    },
    {
      id: 5,
      type: "displayVideo",
      x: 1940,
      y: 160,
      data: { label: "Video Preview" },
    },
    {
      id: 6,
      type: "loadAudio",
      x: 100,
      y: 610,
      data: defaultLoadAudioNodeData(),
    },
    {
      id: 7,
      type: "transcription",
      x: 500,
      y: 610,
      data: defaultTranscriptionNodeData(),
    },
    {
      id: 8,
      type: "lyricMapping",
      x: 960,
      y: 610,
      data: defaultLyricMappingNodeData(),
    },
  ],
  links: [
    { from: 1, fromPort: 0, to: 2, toPort: 0 },
    { from: 2, fromPort: 0, to: 3, toPort: 0 },
    { from: 2, fromPort: 0, to: 4, toPort: 1 },
    { from: 3, fromPort: 0, to: 4, toPort: 0 },
    { from: 4, fromPort: 0, to: 5, toPort: 0 },
    { from: 6, fromPort: 0, to: 7, toPort: 0 },
    { from: 7, fromPort: 0, to: 8, toPort: 0 },
    { from: 1, fromPort: 0, to: 8, toPort: 1 },
    { from: 8, fromPort: 0, to: 2, toPort: 1 },
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

function isInteractiveTarget(target) {
  return Boolean(target?.closest?.("button,input,select,textarea,label"));
}

function scrollableAncestor(target) {
  let node = target;
  while (node && node !== document.body) {
    if (node instanceof HTMLElement) {
      const style = window.getComputedStyle(node);
      const canScrollY = /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight;
      const canScrollX = /(auto|scroll)/.test(style.overflowX) && node.scrollWidth > node.clientWidth;
      if (canScrollY || canScrollX) return node;
    }
    node = node.parentElement;
  }
  return null;
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
    this.panDrag = null;
    this.pan = { x: 0, y: 0 };
    this.tool = "select";
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
          <button data-action="add-story-defaults">Story Defaults</button>
          <button data-action="add-prompt">Prompt</button>
          <button data-action="add-load-image">Load Image</button>
          <button data-action="add-load-audio">Load Audio</button>
          <button data-action="add-transcription">Transcription</button>
          <button data-action="add-lyric-map">Lyric Map</button>
          <button data-action="add-image-mode">Image Mode</button>
          <button data-action="add-image">Image Ref</button>
          <button data-action="add-i2v">Image to Video</button>
          <button data-action="add-video-display">Display Video</button>
          <button data-action="add-mode">Mode</button>
          <button data-action="add-scene">Scene Card</button>
          <button data-action="reset">Reset</button>
          <button data-action="close">Close</button>
        </div>
      </div>
      <div class="vrgdg-node-canvas">
        <svg class="vrgdg-node-links"></svg>
        <div class="vrgdg-node-stage"></div>
        <div class="vrgdg-node-toolstrip" aria-label="Canvas tools">
          <button data-tool="select" title="Select and edit nodes">Select</button>
          <button data-tool="pan" title="Move the canvas">Pan</button>
        </div>
        <div class="vrgdg-node-hint">Right click the canvas to add nodes. Drag from colored ports to connect.</div>
      </div>
    `;

    document.body.appendChild(this.root);
    this.canvas = this.root.querySelector(".vrgdg-node-canvas");
    this.stage = this.root.querySelector(".vrgdg-node-stage");
    this.linksSvg = this.root.querySelector(".vrgdg-node-links");

    this.root.querySelector("[data-action='close']").addEventListener("click", () => this.close());
    this.root.querySelector("[data-action='reset']").addEventListener("click", () => this.resetGraph());
    this.root.querySelector("[data-action='add-story-defaults']").addEventListener("click", () => this.addNode("storyDefaults"));
    this.root.querySelector("[data-action='add-prompt']").addEventListener("click", () => this.addNode("prompt"));
    this.root.querySelector("[data-action='add-load-image']").addEventListener("click", () => this.addNode("loadImage"));
    this.root.querySelector("[data-action='add-load-audio']").addEventListener("click", () => this.addNode("loadAudio"));
    this.root.querySelector("[data-action='add-transcription']").addEventListener("click", () => this.addNode("transcription"));
    this.root.querySelector("[data-action='add-lyric-map']").addEventListener("click", () => this.addNode("lyricMapping"));
    this.root.querySelector("[data-action='add-image-mode']").addEventListener("click", () => this.addNode("imageMode"));
    this.root.querySelector("[data-action='add-image']").addEventListener("click", () => this.addNode("imageRef"));
    this.root.querySelector("[data-action='add-i2v']").addEventListener("click", () => this.addNode("imageToVideo"));
    this.root.querySelector("[data-action='add-video-display']").addEventListener("click", () => this.addNode("displayVideo"));
    this.root.querySelector("[data-action='add-mode']").addEventListener("click", () => this.addNode("mode"));
    this.root.querySelector("[data-action='add-scene']").addEventListener("click", () => this.addNode("sceneCard"));
    this.root.querySelector(".vrgdg-node-toolstrip")?.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
    });
    this.root.querySelectorAll("[data-tool]").forEach((button) => {
      button.addEventListener("click", () => this.setTool(button.dataset.tool));
    });

    this.canvas.addEventListener("contextmenu", (event) => this.showContextMenu(event));
    this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    this.canvas.addEventListener("dragover", (event) => this.onCanvasDragOver(event));
    this.canvas.addEventListener("drop", (event) => this.onCanvasDrop(event));
    this.canvas.addEventListener("pointerdown", (event) => {
      if (this.tool === "pan") {
        if (event.target.closest?.(".vrgdg-node-toolstrip")) return;
        this.startPan(event);
        return;
      }
      if (event.target === this.canvas || event.target === this.linksSvg) {
        this.selectedNodeId = null;
        this.hideContextMenu();
      }
    });
    document.addEventListener("pointerdown", (event) => {
      if (!this.contextMenu) return;
      if (this.contextMenu.contains(event.target)) return;
      this.hideContextMenu();
    });
    window.addEventListener("pointermove", (event) => this.onPointerMove(event));
    window.addEventListener("pointerup", () => this.onPointerUp());
    window.addEventListener("keydown", (event) => this.onKeyDown(event));

    this.render();
    this.updateToolButtons();
    this.applyPan();
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
      x: x != null ? x - this.pan.x : 220 + this.graph.nodes.length * 24,
      y: y != null ? y - this.pan.y : 140 + this.graph.nodes.length * 18,
      data: {},
    };

    if (type === "storyDefaults") node.data = defaultStoryDefaultsNodeData();
    if (type === "prompt") node.data.prompt = "";
    if (type === "loadImage") node.data = defaultLoadImageNodeData();
    if (type === "loadAudio") node.data = defaultLoadAudioNodeData();
    if (type === "transcription") node.data = defaultTranscriptionNodeData();
    if (type === "lyricMapping") node.data = defaultLyricMappingNodeData();
    if (type === "imageMode") node.data = defaultImageModeNodeData();
    if (type === "imageRef") node.data = { label: "Drop image here", imageName: "", imageData: "" };
    if (type === "imageToVideo") {
      node.data = defaultI2VNodeData();
    }
    if (type === "displayVideo") node.data = { label: "Video Preview" };
    if (type === "mode") node.data.mode = "Existing Workflow Mode";
    if (type === "sceneCard") node.data = defaultSceneCardData();

    this.graph.nodes.push(node);
    this.selectedNodeId = node.id;
    this.saveGraph();
    this.render();
    return node;
  }

  setTool(tool) {
    this.tool = tool === "pan" ? "pan" : "select";
    this.hideContextMenu();
    this.updateToolButtons();
  }

  updateToolButtons() {
    if (!this.root) return;
    this.root.querySelectorAll("[data-tool]").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === this.tool);
    });
    this.canvas?.classList.toggle("pan-mode", this.tool === "pan");
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
    if (this.tool === "pan") return;
    this.hideContextMenu();
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const menu = el("div", "vrgdg-node-menu");
    menu.addEventListener("pointerdown", (menuEvent) => menuEvent.stopPropagation());
    menu.innerHTML = `
      <button data-type="storyDefaults">Story Defaults Node</button>
      <button data-type="prompt">Prompt Node</button>
      <button data-type="loadImage">Load Image Node</button>
      <button data-type="loadAudio">Load Audio Node</button>
      <button data-type="transcription">Transcription Node</button>
      <button data-type="lyricMapping">Lyric Mapping Node</button>
      <button data-type="imageMode">Image Mode Node</button>
      <button data-type="imageRef">Image Ref Node</button>
      <button data-type="imageToVideo">Image to Video Node</button>
      <button data-type="displayVideo">Display Video Node</button>
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
    this.applyPan();
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

    card.addEventListener("pointerdown", (event) => {
      if (this.tool === "pan") return;
      if (isInteractiveTarget(event.target)) return;
      this.selectedNodeId = node.id;
      this.render();
    });

    return card;
  }

  renderBody(node, body) {
    if (node.type === "storyDefaults") {
      node.data.tab = node.data.tab || "scene_defaults";
      const tabs = el("div", "vrgdg-node-tabs story-default-tabs");
      [
        ["scene_defaults", "Scene Defaults"],
        ["story_layer", "Story Layer"],
      ].forEach(([tab, label]) => {
        const button = el("button", "", label);
        button.classList.toggle("active", node.data.tab === tab);
        button.addEventListener("pointerdown", (event) => event.stopPropagation());
        button.addEventListener("click", () => {
          node.data.tab = tab;
          this.saveGraph();
          this.render();
        });
        tabs.appendChild(button);
      });
      body.appendChild(tabs);

      if (node.data.tab === "scene_defaults") {
        body.appendChild(this.settingsSection("Scene Defaults", [
          this.selectField(node, "Still shot flow", "still_shot_flow", [
            "Intimate character shots",
            "Balanced cinematic",
            "Wide establishing shots",
            "Performance focused",
            "Action / movement",
          ]),
          this.selectField(node, "Image aesthetic", "image_aesthetic", [
            "Default cinematic still",
            "Soft romantic still",
            "High contrast dramatic still",
            "Documentary natural still",
            "Music video glossy still",
          ]),
          this.textField(node, "Global consistency phrase", "global_consistency_phrase", "e.g. soft glittery eye makeup, wet-look hair..."),
          this.selectField(node, "Global performance style", "global_performance_style", [
            "Default cinematic",
            "Subtle emotional",
            "High energy",
            "Dreamy",
            "Grounded natural",
          ]),
          this.selectField(node, "Global facial performance", "global_facial_performance", [
            "Default natural",
            "Expressive emotional",
            "Calm neutral",
            "Intense eye contact",
            "Soft vulnerable",
          ]),
          this.textAreaField(node, "Custom facial text", "custom_facial_text", "Optional custom facial performance text..."),
        ]));
        body.appendChild(this.nodeActionRow(["Fill Missing", "Replace All"]));
      }

      if (node.data.tab === "story_layer") {
        body.appendChild(this.settingsSection("Story Layer", [
          this.checkboxField(node, "Use in Gemma prompts", "use_in_gemma_prompts"),
          this.numberField(node, "Lyric Story Strength", "lyric_story_strength", 0, 10, 1),
          this.textAreaField(node, "User Story Arc", "user_story_arc", "Optional user story arc, e.g. Verse 1: she feels trapped. Chorus: she breaks free..."),
          this.textAreaField(node, "Song Story Brief", "song_story_brief", "Gemma-created song story brief..."),
        ]));
        body.appendChild(this.nodeActionRow([
          "Create User Story Arc",
          "Create Story Brief",
          "Create Missing Scene Beats",
          "Replace All Scene Beats",
          "Detect Lyric Sections",
        ]));
      }
      return;
    }

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

    if (node.type === "loadImage") {
      const drop = this.imageDropZone(node, "Drop image file here");
      body.appendChild(drop);
      body.appendChild(this.settingsSection("Image Source", [
        this.textField(node, "Loaded file", "imageName", "No image selected"),
        this.selectField(node, "Source mode", "source_mode", [
          "ComfyUI input/upload",
          "Project asset",
          "Generated image output",
          "External path placeholder",
        ]),
        this.selectField(node, "Image role", "image_role", [
          "start image",
          "reference image",
          "character reference",
          "style reference",
          "background plate",
        ]),
      ]));
      body.appendChild(this.settingsSection("Prep Settings", [
        this.selectField(node, "Fit mode", "fit_mode", ["contain", "cover", "center crop", "stretch"]),
        this.checkboxField(node, "Use as vision reference", "use_as_vision_reference"),
        this.textAreaField(node, "Image notes", "notes", "Optional note for how this image should be used..."),
      ]));
      return;
    }

    if (node.type === "loadAudio") {
      const drop = this.audioDropZone(node);
      body.appendChild(drop);
      body.appendChild(this.settingsSection("Audio Source", [
        this.textField(node, "Loaded file", "audioName", "No audio selected"),
        this.selectField(node, "Source mode", "source_mode", [
          "ComfyUI input/upload",
          "Project asset",
          "External path placeholder",
          "Video file audio track",
        ]),
      ]));
      body.appendChild(this.settingsSection("Audio Prep", [
        this.selectField(node, "Sample rate", "sample_rate", ["keep source", "44100", "48000"]),
        this.selectField(node, "Channel mode", "channel_mode", ["stereo", "mono", "keep source"]),
        this.checkboxField(node, "Normalize audio before transcription", "normalize_audio"),
        this.numberField(node, "Trim start seconds", "trim_start_seconds", 0, 9999, 0.1),
        this.numberField(node, "Trim end seconds", "trim_end_seconds", 0, 9999, 0.1),
        this.textAreaField(node, "Audio notes", "notes", "Optional notes about song version, stems, or timing..."),
      ]));
      return;
    }

    if (node.type === "transcription") {
      const audio = this.audioContextForNode(node.id);
      body.appendChild(this.connectionNotice("Audio input", Boolean(audio)));
      if (audio) body.appendChild(this.sceneField("Audio", audio.audioName || "Connected audio"));
      body.appendChild(this.settingsSection("Transcription Engine", [
        this.selectField(node, "Engine", "engine", ["Whisper / stable-ts", "Whisper", "Existing SRT", "Manual transcript"]),
        this.selectField(node, "Model", "model", ["large-v3", "large-v3-turbo", "medium", "small", "base"]),
        this.selectField(node, "Language", "language", ["auto", "en", "es", "fr", "de", "ja", "ko"]),
        this.selectField(node, "Device", "device", ["cuda", "cpu"]),
        this.selectField(node, "Compute type", "compute_type", ["float16", "int8_float16", "int8", "float32"]),
      ]));
      body.appendChild(this.settingsSection("Timing / Captions", [
        this.checkboxField(node, "Use VAD filtering", "use_vad"),
        this.checkboxField(node, "Word timestamps", "word_timestamps"),
        this.checkboxField(node, "Regroup words into lyric lines", "regroup_words"),
        this.numberField(node, "Max SRT line width", "max_line_width", 12, 120, 1),
        this.textField(node, "SRT output name", "srt_output_name", "song_transcript.srt"),
      ]));
      body.appendChild(this.settingsSection("Transcript Preview", [
        this.textAreaField(node, "Transcript text placeholder", "transcript_preview", "Timed transcript preview will show here..."),
      ]));
      body.appendChild(this.nodeActionRow(["Transcribe Audio", "Import SRT", "Clear Transcript"]));
      return;
    }

    if (node.type === "lyricMapping") {
      const transcript = this.transcriptContextForNode(node.id);
      const storyDefaults = this.storyContextForNode(node.id);
      body.appendChild(this.settingsSection("Inputs", [
        this.connectionNotice("Transcript input", Boolean(transcript)),
        this.connectionNotice("Story Context input", Boolean(storyDefaults)),
      ]));
      body.appendChild(this.settingsSection("Mapping Settings", [
        this.selectField(node, "Mapping mode", "mapping_mode", ["lyrics to scenes", "sections to scenes", "beats to scenes", "manual timing"]),
        this.selectField(node, "Split mode", "split_mode", ["detect sections", "line groups", "fixed scene count", "manual beats"]),
        this.numberField(node, "Target scene count", "scene_count", 1, 99, 1),
        this.numberField(node, "Timing offset seconds", "timing_offset_seconds", -30, 30, 0.1),
        this.numberField(node, "Beat padding seconds", "beat_padding_seconds", 0, 10, 0.05),
        this.checkboxField(node, "Attach lyric map to Scene Cards", "attach_to_scene_cards"),
        this.checkboxField(node, "Use Story Layer when creating beats", "use_story_layer"),
      ]));
      body.appendChild(this.settingsSection("Lyrics / Beat Notes", [
        this.textAreaField(node, "Full lyrics override", "full_lyrics", "Optional full lyrics if transcript is incomplete..."),
        this.textAreaField(node, "Mapping notes", "mapping_notes", "Optional notes for sections, story beats, or chorus/verse structure..."),
        this.textAreaField(node, "Lyric map preview", "map_preview", "Scene-by-scene lyric timing preview will show here..."),
      ]));
      body.appendChild(this.nodeActionRow(["Detect Lyric Sections", "Map Lyrics To Scenes", "Create Missing Scene Beats"]));
      return;
    }

    if (node.type === "imageMode") {
      const mode = node.data.image_model_mode || "zimage";
      const imageResolved = this.resolveImageMode(node);
      node.data.tab = node.data.tab || "models";
      body.appendChild(this.selectField(node, "Image model", "image_model_mode", [
        "zimage",
        "ernie_image",
        "krea2_2pass",
        "flux_klein",
        "nano_banana",
        "flow_gpt",
        "z_enhance",
      ]));

      const tabs = el("div", "vrgdg-node-tabs");
      [
        ["models", "Models"],
        ["settings", "Image Settings"],
        ["prompting", "LLM Prompting"],
      ].forEach(([tab, label]) => {
        const button = el("button", "", label);
        button.classList.toggle("active", node.data.tab === tab);
        button.addEventListener("pointerdown", (event) => event.stopPropagation());
        button.addEventListener("click", () => {
          node.data.tab = tab;
          this.saveGraph();
          this.render();
        });
        tabs.appendChild(button);
      });
      body.appendChild(tabs);
      if (node.data.tab === "prompting") {
        body.appendChild(this.connectionNotice("Scene Context input", Boolean(imageResolved.scene)));
        if (imageResolved.scene) {
          body.appendChild(this.sceneField("Scene", `${imageResolved.scene.sceneNumber || ""} ${imageResolved.scene.title || ""}`.trim() || "Connected scene"));
          body.appendChild(this.sceneField("Scene Image Prompt", imageResolved.scene.imagePrompt || "No scene image prompt"));
        }
      }
      this.renderImageModeBody(node, body, mode, node.data.tab);
      const output = this.imageModeOutputPreview(node);
      if (output) body.appendChild(output);
      return;
    }

    if (node.type === "imageToVideo") {
      const resolved = this.resolveImageToVideo(node);
      node.data.tab = I2V_TAB_ALIASES[node.data.tab] || node.data.tab || "models";
      const tabs = el("div", "vrgdg-node-tabs");
      [
        ["models", "Models"],
        ["video_settings", "Video Settings"],
        ["llm_prompting", "LLM Prompting"],
      ].forEach(([tab, label]) => {
        const button = el("button", "", label);
        button.classList.toggle("active", (node.data.tab || "models") === tab);
        button.addEventListener("pointerdown", (event) => event.stopPropagation());
        button.addEventListener("click", () => {
          node.data.tab = tab;
          this.saveGraph();
          this.render();
        });
        tabs.appendChild(button);
      });
      body.appendChild(tabs);

      const activeTab = node.data.tab || "models";
      if (activeTab === "models") {
        body.appendChild(this.settingsSection("Video Models", [
          this.checkboxField(node, "Use custom video models/settings/LoRAs for this scene", "use_scene_i2v_video_settings"),
          this.checkboxField(node, "Use GGUF model?", "use_gguf_model"),
          this.textField(node, "Unet model", "unet_name", "LTX GGUF model"),
          this.textField(node, "Diffusion model", "diffusion_model_name", "LTX diffusion model"),
          this.textField(node, "Video VAE", "vae_name", "Video VAE"),
          this.textField(node, "Clip model 1", "clip_name1", "CLIP / Gemma text encoder"),
          this.textField(node, "Clip model 2", "clip_name2", "Text projection"),
          this.textField(node, "Latent upscaler", "upscale_model_name", "Latent upscaler"),
          this.textField(node, "Audio VAE", "audio_vae_name", "Audio VAE"),
        ]));
        body.appendChild(this.settingsSection("LLM Models", [
          this.textField(node, "Non-Vision text Gemma model", "text_gemma_model", "Text Gemma model"),
          this.textField(node, "Vision Gemma model", "vision_gemma_model", "Vision Gemma model"),
          this.textField(node, "Vision mmproj", "mmproj_file", "Vision mmproj"),
        ]));
        body.appendChild(this.settingsSection("Video LoRAs", [
          this.checkboxField(node, "Use video LoRAs?", "use_loras"),
          this.numberField(node, "Video LoRA count", "lora_count", 0, 4, 1),
          this.textField(node, "Required MSR LoRA", "msr_lora_name", "MSR LoRA"),
          this.textField(node, "Ingredients LoRA", "ingredients_lora_name", "Ingredients LoRA"),
          this.textField(node, "ID-LoRA I2V LoRA", "id_lora_name", "ID LoRA"),
        ]));
      }
      if (activeTab === "video_settings") {
        body.appendChild(this.settingsSection("Render Size / Timing", [
          this.numberField(node, "FPS", "fps", 1, 120, 1),
          this.numberField(node, "Seed", "seed", 0, 999999999, 1),
          this.numberField(node, "Width", "width", 64, 4096, 8),
          this.numberField(node, "Height", "height", 64, 4096, 8),
          this.textField(node, "Video trigger phrase", "video_trigger_phrase", "Optional trigger phrase"),
        ]));
        body.appendChild(this.settingsSection("Warm/Cool Frames", [
          this.numberField(node, "Cool Down Frames", "tail_loss_frames", 0, 999, 1),
          this.numberField(node, "Warm Up Frames", "pre_frames", 0, 999, 1),
        ]));
        body.appendChild(this.settingsSection("Pass 1", [
          this.selectField(node, "Sampler", "pass1_sampler", I2V_SAMPLER_OPTIONS),
          this.textField(node, "Sigmas", "pass1_sigmas", DEFAULT_I2V_PASS1_SIGMAS),
          this.numberField(node, "Strength", "pass1_strength", 0, 1, 0.01),
          this.checkboxField(node, "Bypass", "pass1_bypass"),
        ]));
        body.appendChild(this.settingsSection("Pass 2", [
          this.selectField(node, "Sampler", "pass2_sampler", I2V_SAMPLER_OPTIONS),
          this.textField(node, "Sigmas", "pass2_sigmas", DEFAULT_I2V_PASS2_SIGMAS),
          this.numberField(node, "Strength", "pass2_strength", 0, 1, 0.01),
          this.checkboxField(node, "Bypass", "pass2_bypass"),
        ]));
      }
      if (activeTab === "llm_prompting") {
        body.appendChild(this.settingsSection("Prompt Inputs", [
          this.connectionNotice("Scene Context input", resolved.sceneConnected),
          this.connectionNotice("Prompt override input", resolved.promptConnected),
          this.connectionNotice("Start image input", resolved.imageConnected),
          this.checkboxField(node, "I2V prompt enhancement pass", "use_i2v_prompt_enhancement_pass"),
          this.checkboxField(node, "Use image reference for I2V prompt?", "use_i2v_vision_reference"),
        ]));
        body.appendChild(this.settingsSection("Video Motion Notes", [
          this.textAreaField(node, "Video motion notes", "i2v_notes", "Extra video motion notes, camera movement, character movement..."),
        ]));
        body.appendChild(this.settingsSection("Video Prompt", [
          this.textAreaField(node, "Video prompt fallback", "i2v_prompt", "Used if no Scene Card video prompt or Prompt Override is connected...", resolved.promptConnected || resolved.sceneConnected),
        ]));
        if (resolved.scene) {
          body.appendChild(this.sceneField("Scene", `${resolved.scene.sceneNumber || ""} ${resolved.scene.title || ""}`.trim() || "Connected scene"));
          body.appendChild(this.sceneField("Motion Notes", resolved.scene.motionNotes || "No scene motion notes"));
        }
        body.appendChild(this.sceneField("Resolved Prompt", resolved.prompt || "No prompt yet"));
      }
      if (activeTab === "llm_prompting_legacy") {
        body.appendChild(this.connectionNotice("Prompt input", resolved.promptConnected));
        const textarea = el("textarea", "vrgdg-node-textarea compact");
        textarea.placeholder = "Prompt fallback if no Prompt node is connected...";
        textarea.value = node.data.i2v_prompt || "";
        textarea.disabled = resolved.promptConnected;
        textarea.addEventListener("input", () => {
          node.data.i2v_prompt = textarea.value;
          this.saveGraph();
          this.renderLinks();
        });
        body.appendChild(textarea);
      }
      if (activeTab === "output") {
        body.appendChild(this.connectionNotice("Start image", resolved.imageConnected));
        const preview = el("div", "vrgdg-node-scene-preview");
        if (resolved.imageData) {
          const img = el("img", "vrgdg-scene-image");
          img.src = resolved.imageData;
          preview.appendChild(img);
        } else {
          preview.textContent = "No start image connected";
        }
        body.appendChild(preview);
        body.appendChild(this.sceneField("Resolved Prompt", resolved.prompt || "No prompt yet"));
      }
      return;
    }

    if (node.type === "displayVideo") {
      const resolved = this.resolveDisplayVideo(node);
      const title = el("input", "vrgdg-node-input");
      title.value = node.data.label || "Video Preview";
      title.addEventListener("input", () => {
        node.data.label = title.value;
        this.saveGraph();
      });
      body.appendChild(title);

      const preview = el("div", "vrgdg-node-video-preview");
      if (resolved.imageData) {
        const img = el("img", "vrgdg-scene-image");
        img.src = resolved.imageData;
        preview.appendChild(img);
        preview.appendChild(el("span", "", "I2V preview placeholder"));
      } else {
        preview.textContent = "Connect Image to Video output";
      }
      body.appendChild(preview);
      body.appendChild(this.sceneField("Model", resolved.model || "No video connected"));
      body.appendChild(this.sceneField("Prompt", resolved.prompt || "No prompt connected"));
      return;
    }

    if (node.type === "sceneCard") {
      const storyDefaults = this.storyContextForNode(node.id);
      const lyricMap = this.lyricMapContextForNode(node.id);
      node.data.tab = node.data.tab || "scene";
      const tabs = el("div", "vrgdg-node-tabs scene-tabs");
      [
        ["scene", "Scene"],
        ["story", "Story"],
        ["prompts", "Prompts"],
      ].forEach(([tab, label]) => {
        const button = el("button", "", label);
        button.classList.toggle("active", node.data.tab === tab);
        button.addEventListener("pointerdown", (event) => event.stopPropagation());
        button.addEventListener("click", () => {
          node.data.tab = tab;
          this.saveGraph();
          this.render();
        });
        tabs.appendChild(button);
      });
      body.appendChild(tabs);

      if (node.data.tab === "scene") {
        body.appendChild(this.connectionNotice("Story Context input", Boolean(storyDefaults)));
        body.appendChild(this.connectionNotice("Lyric Map input", Boolean(lyricMap)));
        if (storyDefaults) {
          body.appendChild(this.sceneField("Scene Defaults", [
            storyDefaults.stillShotFlow,
            storyDefaults.imageAesthetic,
            storyDefaults.globalPerformanceStyle,
            storyDefaults.globalFacialPerformance,
          ].filter(Boolean).join(" · ") || "Connected"));
        }
        if (lyricMap) {
          body.appendChild(this.sceneField("Lyric Mapping", [
            lyricMap.mappingMode,
            lyricMap.splitMode,
            lyricMap.sceneCount ? `${lyricMap.sceneCount} target scenes` : "",
          ].filter(Boolean).join(" · ") || "Connected lyric map"));
        }
        body.appendChild(this.settingsSection("Scene Identity", [
          this.textField(node, "Scene title", "title", "Scene title"),
          this.textField(node, "Scene number", "scene_number", "01"),
          this.numberField(node, "Duration", "duration", 0.1, 999, 0.1),
        ]));
      }
      if (node.data.tab === "story") {
        body.appendChild(this.settingsSection("Story Context", [
          this.textAreaField(node, "Lyrics / dialogue", "lyrics", "Lyrics, dialogue, or narration for this scene..."),
          this.textAreaField(node, "Story beat", "story_beat", "What happens in this scene..."),
          this.textAreaField(node, "Subject notes", "subject_notes", "Character/subject notes..."),
          this.textAreaField(node, "Setting notes", "setting_notes", "Location/environment notes..."),
        ]));
      }
      if (node.data.tab === "prompts") {
        body.appendChild(this.settingsSection("Prompts", [
          this.textAreaField(node, "Image prompt", "image_prompt", "Image prompt for the image model..."),
          this.textAreaField(node, "Video prompt", "video_prompt", "Video prompt for image-to-video..."),
          this.textAreaField(node, "Motion notes", "motion_notes", "Camera movement, character movement, action..."),
        ]));
      }
      return;
    }
  }

  renderImageModeBody(node, body, mode, tab) {
    const section = (...args) => this.settingsSection(...args);
    const text = (...args) => this.textField(node, ...args);
    const check = (...args) => this.checkboxField(node, ...args);
    const num = (...args) => this.numberField(node, ...args);
    const area = (...args) => this.textAreaField(node, ...args);

    if (tab === "models") {
      if (mode === "zimage") {
        body.appendChild(section("ZImage Models", [
          text("ZImage model", "zimage_model", "ZImage model"),
          text("CLIP", "zimage_clip", "CLIP"),
          text("VAE", "zimage_vae", "VAE"),
        ]));
        body.appendChild(section("LLM Models", [
          text("Non-Vision text Gemma model", "zimage_text_gemma_model", "Text Gemma model"),
          text("Vision Gemma model", "zimage_vision_gemma_model", "Vision Gemma model"),
          text("Vision mmproj", "zimage_mmproj", "Vision mmproj"),
        ]));
        body.appendChild(section("LoRAs", [check("Use ZImage LoRAs?", "zimage_use_lora"), num("LoRA count", "zimage_lora_count", 0, 8, 1)]));
      }
      if (mode === "ernie_image") {
        body.appendChild(section("Ernie Models", [
          text("Ernie model", "ernie_model", "Ernie model"),
          text("CLIP", "ernie_clip", "CLIP"),
          text("VAE", "ernie_vae", "VAE"),
        ]));
        body.appendChild(section("LLM Models", [
          text("Non-Vision text Gemma model", "ernie_text_gemma_model", "Text Gemma model"),
          text("Vision Gemma model", "ernie_vision_gemma_model", "Vision Gemma model"),
          text("Vision mmproj", "ernie_mmproj", "Vision mmproj"),
        ]));
        body.appendChild(section("LoRAs", [check("Use Ernie LoRAs?", "ernie_use_lora"), num("LoRA count", "ernie_lora_count", 0, 8, 1)]));
      }
      if (mode === "krea2_2pass") {
        body.appendChild(section("Krea 2 Models", [
          text("Krea2 model", "krea2_model", "Krea2 model"),
          text("CLIP", "krea2_clip", "CLIP"),
          text("VAE", "krea2_vae", "VAE"),
          check("Use Krea2 LoRAs?", "krea2_use_lora"),
          num("LoRA count", "krea2_lora_count", 0, 8, 1),
        ]));
        body.appendChild(section("LLM Models", [
          text("Non-Vision text Gemma model", "krea2_text_gemma_model", "Text Gemma model"),
          text("Vision Gemma model", "krea2_vision_gemma_model", "Vision Gemma model"),
          text("Vision mmproj", "krea2_mmproj", "Vision mmproj"),
        ]));
      }
      if (mode === "flux_klein") {
        body.appendChild(section("Flux/Klein Models", [
          text("Flux model", "flux_model", "Flux/Klein model"),
          text("Flux CLIP", "flux_clip", "Flux CLIP"),
          text("Flux VAE", "flux_vae", "Flux VAE"),
        ]));
        body.appendChild(section("Vision LLM Models", [
          text("Gemma vision model", "flux_vision_gemma_model", "Vision Gemma model"),
          text("Vision mmproj", "flux_mmproj", "Vision mmproj"),
        ]));
        body.appendChild(section("LoRAs", [check("Use Flux LoRAs?", "flux_use_lora"), num("LoRA count", "flux_lora_count", 0, 8, 1)]));
      }
      if (mode === "nano_banana") {
        body.appendChild(section("NanoBanana", [
          text("Google Cloud API key", "nano_api_key", "API key placeholder"),
          text("Model", "nano_model", "NanoBanana model"),
        ]));
        body.appendChild(section("Vision LLM Models", [
          text("Gemma vision model", "nano_vision_gemma_model", "Vision Gemma model"),
          text("Vision mmproj", "nano_mmproj", "Vision mmproj"),
        ]));
      }
      if (mode === "flow_gpt") {
        body.appendChild(section("Flow/GPT Provider", [
          this.selectField(node, "Provider", "flow_provider", ["chatgpt", "flow_gpt"]),
          this.connectionNotice("Browser automation setup", false),
        ]));
      }
      if (mode === "z_enhance") {
        body.appendChild(section("ZImage Enhance Models", [
          text("ZImage model", "zenhance_model", "ZImage model"),
          text("CLIP", "zenhance_clip", "CLIP"),
          text("VAE", "zenhance_vae", "VAE"),
        ]));
        body.appendChild(section("Vision LLM Models", [
          text("Gemma vision model", "zenhance_vision_gemma_model", "Vision Gemma model"),
          text("Vision mmproj", "zenhance_mmproj", "Vision mmproj"),
        ]));
        body.appendChild(section("LoRAs", [check("Use ZEnhance LoRAs?", "zenhance_use_lora"), num("LoRA count", "zenhance_lora_count", 0, 8, 1)]));
      }
    }

    if (tab === "settings") {
      if (mode === "zimage") {
        body.appendChild(section("Image Settings", [
          text("Image trigger phrase", "zimage_trigger_phrase", "Optional trigger phrase"),
          num("Batch size", "zimage_batch_size", 1, 16, 1),
          check("Use image-to-image?", "zimage_use_image_to_image"),
        ]));
      }
      if (mode === "ernie_image") {
        body.appendChild(section("Image Settings", [
          text("Image trigger phrase", "ernie_trigger_phrase", "Optional trigger phrase"),
          num("Batch size", "ernie_batch_size", 1, 16, 1),
          check("Use image-to-image?", "ernie_use_image_to_image"),
        ]));
      }
      if (mode === "krea2_2pass") {
        body.appendChild(section("Image Settings", [
          text("Image trigger phrase", "krea2_trigger_phrase", "Optional trigger phrase"),
          check("Use image-to-image?", "krea2_use_image_to_image"),
        ]));
      }
      if (mode === "flux_klein") {
        body.appendChild(section("Image Settings", [
          check("Use text-only Gemma prompt?", "flux_use_text_only_gemma_prompt"),
          check("Use director notes?", "flux_use_director_notes"),
          text("Image trigger phrase", "flux_trigger_phrase", "Optional trigger phrase"),
          area("Reference images", "flux_reference_images", "Reference image paths/notes..."),
        ]));
      }
      if (mode === "nano_banana") {
        body.appendChild(section("Image Settings", [
          check("Use global ingredients?", "nano_use_global_ingredients"),
          check("Use text-only Gemma prompt?", "nano_use_text_only_gemma_prompt"),
          check("Use director notes?", "nano_use_director_notes"),
          area("Ingredients", "nano_ingredients", "Ingredient refs/notes..."),
        ]));
      }
      if (mode === "flow_gpt") {
        body.appendChild(section("Image Settings", [
          this.selectField(node, "Aspect ratio", "flow_aspect_ratio", ["16:9", "9:16", "1:1", "4:3", "3:4"]),
          num("Timeout seconds", "flow_timeout_seconds", 1, 3600, 1),
          num("Retries", "flow_retries", 0, 10, 1),
          this.selectField(node, "After final failure", "flow_failure_mode", ["pause", "skip", "continue"]),
          check("Manual mode", "flow_manual_mode"),
          check("Manual auto advance", "flow_manual_auto_advance"),
        ]));
      }
      if (mode === "z_enhance") {
        body.appendChild(section("Image Settings", [
          num("Enhance amount", "zenhance_amount", 0, 1, 0.01),
        ]));
      }
    }

    if (tab === "prompting") {
      if (mode === "zimage") {
        body.appendChild(section("LLM Prompting", [
          area("Notes", "zimage_notes", "Scene/image notes..."),
          check("Use vision reference?", "zimage_use_vision_reference"),
          area("T2I prompt", "zimage_prompt", "T2I prompt..."),
        ]));
      }
      if (mode === "ernie_image") {
        body.appendChild(section("LLM Prompting", [
          area("Notes", "ernie_notes", "Ernie image notes..."),
          check("Use vision reference?", "ernie_use_vision_reference"),
          area("T2I prompt", "ernie_prompt", "Ernie T2I prompt..."),
        ]));
      }
      if (mode === "krea2_2pass") {
        body.appendChild(section("LLM Prompting", [
          area("Notes", "krea2_notes", "Krea 2 notes..."),
          check("Use vision reference?", "krea2_use_vision_reference"),
          area("T2I prompt", "krea2_prompt", "Krea 2 prompt..."),
        ]));
      }
      if (mode === "flux_klein") {
        body.appendChild(section("LLM Prompting", [
          area("Flux/Klein notes", "flux_notes", "Flux/Klein notes..."),
          area("Flux/Klein prompt", "flux_prompt", "Flux/Klein prompt..."),
        ]));
      }
      if (mode === "nano_banana") {
        body.appendChild(section("LLM Prompting", [
          area("NanoBanana notes", "nano_notes", "NanoBanana notes..."),
          area("NanoBanana prompt", "nano_prompt", "NanoBanana prompt..."),
        ]));
      }
      if (mode === "flow_gpt") {
        body.appendChild(section("LLM Prompting", [
          area("Browser prompt", "flow_prompt", "Flow/GPT browser prompt..."),
        ]));
      }
      if (mode === "z_enhance") {
        body.appendChild(section("LLM Prompting", [
          area("Gemma notes", "zenhance_notes", "Enhance notes..."),
          area("Enhance prompt", "zenhance_prompt", "Enhance prompt..."),
        ]));
      }
    }
  }

  imageModeOutputPreview(node) {
    const wrap = el("div", "vrgdg-node-drop compact-output");
    wrap.textContent = node.data.imageName || "Drop placeholder output image here";
    if (node.data.imageData) {
      const img = el("img", "vrgdg-node-preview");
      img.src = node.data.imageData;
      wrap.textContent = "";
      wrap.appendChild(img);
      wrap.appendChild(el("span", "vrgdg-node-caption", node.data.imageName || "Image output"));
    }
    wrap.addEventListener("dragover", (event) => {
      event.preventDefault();
      wrap.classList.add("dragging");
    });
    wrap.addEventListener("dragleave", () => wrap.classList.remove("dragging"));
    wrap.addEventListener("drop", (event) => this.handleImageDrop(event, node));
    return wrap;
  }

  sceneField(label, value) {
    const wrap = el("div", "vrgdg-scene-field");
    wrap.appendChild(el("label", "", label));
    wrap.appendChild(el("div", "", value));
    return wrap;
  }

  connectionNotice(label, connected) {
    const wrap = el("div", `vrgdg-connection-notice ${connected ? "connected" : ""}`);
    wrap.textContent = connected ? `${label}: connected` : `${label}: using local settings`;
    return wrap;
  }

  settingsSection(title, controls) {
    const section = el("div", "vrgdg-node-settings-section");
    section.appendChild(el("div", "vrgdg-node-settings-title", title));
    controls.forEach((control) => section.appendChild(control));
    return section;
  }

  nodeActionRow(labels) {
    const row = el("div", "vrgdg-node-action-row");
    labels.forEach((label) => {
      const button = el("button", "", label);
      button.type = "button";
      button.addEventListener("pointerdown", (event) => event.stopPropagation());
      button.addEventListener("click", () => {
        console.log(`[VRGDG Node Canvas] Placeholder action: ${label}`);
      });
      row.appendChild(button);
    });
    return row;
  }

  checkboxField(node, label, key) {
    const wrap = el("label", "vrgdg-node-checkbox");
    const input = el("input");
    input.type = "checkbox";
    input.checked = Boolean(node.data[key]);
    input.addEventListener("change", () => {
      node.data[key] = Boolean(input.checked);
      this.saveGraph();
      this.renderLinks();
    });
    wrap.appendChild(input);
    wrap.appendChild(el("span", "", label));
    return wrap;
  }

  selectField(node, label, key, values) {
    const wrap = el("label", "vrgdg-node-setting");
    wrap.appendChild(el("span", "", label));
    const select = el("select", "vrgdg-node-select");
    values.forEach((value) => {
      const option = el("option", "", value);
      option.value = value;
      select.appendChild(option);
    });
    select.value = node.data[key] || values[0];
    select.addEventListener("change", () => {
      node.data[key] = select.value;
      this.saveGraph();
      this.renderLinks();
    });
    wrap.appendChild(select);
    return wrap;
  }

  numberField(node, label, key, min, max, step) {
    const wrap = el("label", "vrgdg-node-setting");
    wrap.appendChild(el("span", "", label));
    const input = el("input", "vrgdg-node-input");
    input.type = "number";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = node.data[key] ?? "";
    input.addEventListener("input", () => {
      node.data[key] = input.value;
      this.saveGraph();
      this.renderLinks();
    });
    wrap.appendChild(input);
    return wrap;
  }

  textField(node, label, key, placeholder) {
    const wrap = el("label", "vrgdg-node-setting");
    wrap.appendChild(el("span", "", label));
    const input = el("input", "vrgdg-node-input");
    input.placeholder = placeholder || "";
    input.value = node.data[key] || "";
    input.addEventListener("input", () => {
      node.data[key] = input.value;
      this.saveGraph();
      this.renderLinks();
    });
    wrap.appendChild(input);
    return wrap;
  }

  textAreaField(node, label, key, placeholder, disabled = false) {
    const wrap = el("label", "vrgdg-node-setting");
    wrap.appendChild(el("span", "", label));
    const textarea = el("textarea", "vrgdg-node-textarea compact");
    textarea.placeholder = placeholder || "";
    textarea.value = node.data[key] || "";
    textarea.disabled = Boolean(disabled);
    textarea.addEventListener("input", () => {
      node.data[key] = textarea.value;
      this.saveGraph();
      this.renderLinks();
    });
    wrap.appendChild(textarea);
    return wrap;
  }

  handleImageDrop(event, node) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    this.applyImageFileToNode(file, node);
  }

  applyImageFileToNode(file, node) {
    const reader = new FileReader();
    reader.onload = () => {
      node.data.imageName = file.name;
      node.data.imageData = String(reader.result || "");
      this.saveGraph();
      this.render();
    };
    reader.readAsDataURL(file);
  }

  imageDropZone(node, label) {
    const drop = el("div", "vrgdg-node-drop");
    drop.textContent = node.data.imageName || label || "Drop image here";
    if (node.data.imageData) {
      const img = el("img", "vrgdg-node-preview");
      img.src = node.data.imageData;
      drop.textContent = "";
      drop.appendChild(img);
      drop.appendChild(el("span", "vrgdg-node-caption", node.data.imageName || "Loaded image"));
    }
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.stopPropagation();
      drop.classList.add("dragging");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (event) => {
      event.stopPropagation();
      drop.classList.remove("dragging");
      this.handleImageDrop(event, node);
    });
    return drop;
  }

  audioDropZone(node) {
    const drop = el("div", "vrgdg-node-drop audio");
    drop.textContent = node.data.audioName || "Drop audio file here";
    if (node.data.audioName) {
      drop.textContent = "";
      drop.appendChild(el("span", "vrgdg-node-caption", node.data.audioName));
      if (node.data.audioUrl) {
        const audio = el("audio", "vrgdg-audio-preview");
        audio.controls = true;
        audio.src = node.data.audioUrl;
        drop.appendChild(audio);
      } else {
        drop.appendChild(el("small", "", "Audio preview placeholder"));
      }
    }
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.stopPropagation();
      drop.classList.add("dragging");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      drop.classList.remove("dragging");
      const file = event.dataTransfer?.files?.[0];
      if (file?.type?.startsWith("audio/")) this.applyAudioFileToNode(file, node);
    });
    return drop;
  }

  applyAudioFileToNode(file, node) {
    node.data.audioName = file.name;
    if (node.data.audioUrl?.startsWith?.("blob:")) URL.revokeObjectURL(node.data.audioUrl);
    node.data.audioUrl = URL.createObjectURL(file);
    this.saveGraph();
    this.render();
  }

  onCanvasDragOver(event) {
    if (!event.dataTransfer?.files?.length) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }

  onCanvasDrop(event) {
    if (event.target.closest?.(".vrgdg-node-card")) return;
    const files = Array.from(event.dataTransfer?.files || []);
    if (!files.length) return;
    event.preventDefault();
    this.hideContextMenu();
    const rect = this.canvas.getBoundingClientRect();
    const baseX = event.clientX - rect.left;
    const baseY = event.clientY - rect.top;
    files.forEach((file, index) => {
      const x = baseX + index * 36;
      const y = baseY + index * 28;
      if (file.type.startsWith("image/")) {
        const node = this.addNode("loadImage", x, y);
        if (node) this.applyImageFileToNode(file, node);
      }
      if (file.type.startsWith("audio/")) {
        const node = this.addNode("loadAudio", x, y);
        if (node) this.applyAudioFileToNode(file, node);
      }
    });
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
    if (this.tool !== "select") return;
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
    if (this.tool !== "select") return;
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
    if (this.panDrag) {
      const dx = event.clientX - this.panDrag.startX;
      const dy = event.clientY - this.panDrag.startY;
      this.pan.x = Math.round(this.panDrag.panX + dx);
      this.pan.y = Math.round(this.panDrag.panY + dy);
      this.applyPan();
      this.renderLinks();
      return;
    }

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

  onWheel(event) {
    if (!this.root || this.root.classList.contains("hidden")) return;
    const scrollable = scrollableAncestor(event.target);
    if (scrollable && scrollable.closest?.(".vrgdg-node-card")) {
      event.preventDefault();
      event.stopPropagation();
      scrollable.scrollTop += event.deltaY;
      scrollable.scrollLeft += event.deltaX;
      return;
    }

    event.preventDefault();
    this.hideContextMenu();
    this.pan.x -= Math.round(event.deltaX);
    this.pan.y -= Math.round(event.deltaY);
    this.applyPan();
    this.renderLinks();
  }

  onPointerUp() {
    if (this.panDrag) {
      this.panDrag = null;
    }

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

  startPan(event) {
    event.preventDefault();
    this.hideContextMenu();
    this.panDrag = {
      startX: event.clientX,
      startY: event.clientY,
      panX: this.pan.x,
      panY: this.pan.y,
    };
    this.canvas?.setPointerCapture?.(event.pointerId);
  }

  applyPan() {
    if (this.stage) {
      this.stage.style.transform = `translate(${this.pan.x}px, ${this.pan.y}px)`;
    }
    if (this.canvas) {
      this.canvas.style.backgroundPosition = `${this.pan.x}px ${this.pan.y}px`;
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
    return this.resolveSceneContext(sceneNode);
  }

  resolveSceneContext(sceneNode) {
    if (!sceneNode) return {
      title: "",
      sceneNumber: "",
      duration: "",
      lyrics: "",
      storyBeat: "",
      subjectNotes: "",
      settingNotes: "",
      imagePrompt: "",
      videoPrompt: "",
      motionNotes: "",
    };
    return {
      title: sceneNode.data.title || "Scene",
      sceneNumber: sceneNode.data.scene_number || "",
      duration: sceneNode.data.duration || "",
      lyrics: sceneNode.data.lyrics || "",
      storyBeat: sceneNode.data.story_beat || "",
      subjectNotes: sceneNode.data.subject_notes || "",
      settingNotes: sceneNode.data.setting_notes || "",
      imagePrompt: sceneNode.data.image_prompt || "",
      videoPrompt: sceneNode.data.video_prompt || "",
      motionNotes: sceneNode.data.motion_notes || "",
    };
  }

  sceneContextForNode(nodeId) {
    const incoming = this.graph.links.filter((link) => link.to === nodeId);
    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (source?.type === "sceneCard") return this.resolveSceneContext(source);
    }
    return null;
  }

  storyContextForNode(nodeId) {
    const incoming = this.graph.links.filter((link) => link.to === nodeId);
    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (source?.type === "storyDefaults") return this.resolveStoryDefaults(source);
    }
    return null;
  }

  audioContextForNode(nodeId) {
    const incoming = this.graph.links.filter((link) => link.to === nodeId);
    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (source?.type === "loadAudio") {
        return {
          audioName: source.data.audioName || "",
          sourceMode: source.data.source_mode || "",
          sampleRate: source.data.sample_rate || "",
          channelMode: source.data.channel_mode || "",
        };
      }
    }
    return null;
  }

  transcriptContextForNode(nodeId) {
    const incoming = this.graph.links.filter((link) => link.to === nodeId);
    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (source?.type === "transcription") {
        return {
          engine: source.data.engine || "",
          model: source.data.model || "",
          language: source.data.language || "",
          transcriptPreview: source.data.transcript_preview || "",
        };
      }
    }
    return null;
  }

  lyricMapContextForNode(nodeId) {
    const incoming = this.graph.links.filter((link) => link.to === nodeId);
    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (source?.type === "lyricMapping") {
        return {
          mappingMode: source.data.mapping_mode || "",
          splitMode: source.data.split_mode || "",
          sceneCount: source.data.scene_count || "",
          mapPreview: source.data.map_preview || "",
        };
      }
    }
    return null;
  }

  resolveStoryDefaults(node) {
    if (!node) return null;
    return {
      stillShotFlow: node.data.still_shot_flow || "",
      imageAesthetic: node.data.image_aesthetic || "",
      globalConsistencyPhrase: node.data.global_consistency_phrase || "",
      globalPerformanceStyle: node.data.global_performance_style || "",
      globalFacialPerformance: node.data.global_facial_performance || "",
      customFacialText: node.data.custom_facial_text || "",
      useInGemmaPrompts: node.data.use_in_gemma_prompts !== false,
      lyricStoryStrength: node.data.lyric_story_strength || 0,
      userStoryArc: node.data.user_story_arc || "",
      songStoryBrief: node.data.song_story_brief || "",
    };
  }

  resolveImageMode(imageNode) {
    const scene = this.sceneContextForNode(imageNode.id);
    const incoming = this.graph.links.filter((link) => link.to === imageNode.id);
    const promptOverride = incoming
      .map((link) => this.graph.nodes.find((node) => node.id === link.from))
      .find((node) => node?.type === "prompt");
    const mode = imageNode.data.image_model_mode || "zimage";
    const promptKey = {
      zimage: "zimage_prompt",
      ernie_image: "ernie_prompt",
      krea2_2pass: "krea2_prompt",
      flux_klein: "flux_prompt",
      nano_banana: "nano_prompt",
      flow_gpt: "flow_prompt",
      z_enhance: "zenhance_prompt",
    }[mode] || "zimage_prompt";
    return {
      mode,
      label: `${this.imageModeLabel(mode)} output`,
      prompt: promptOverride?.data?.prompt || imageNode.data[promptKey] || scene?.imagePrompt || "",
      imageName: imageNode.data.imageName || `${this.imageModeLabel(mode)} image`,
      imageData: imageNode.data.imageData || "",
      scene,
    };
  }

  imageModeLabel(mode) {
    return {
      zimage: "ZImage",
      ernie_image: "Ernie",
      krea2_2pass: "Krea 2",
      flux_klein: "Flux/Klein",
      nano_banana: "NanoBanana",
      flow_gpt: "Flow/GPT",
      z_enhance: "ZEnhance",
    }[mode] || "Image";
  }

  resolveImageToVideo(videoNode) {
    const incoming = this.graph.links.filter((link) => link.to === videoNode.id);
    const resolved = {
      model: videoNode.data.use_gguf_model === false
        ? (videoNode.data.diffusion_model_name || "")
        : (videoNode.data.unet_name || videoNode.data.diffusion_model_name || ""),
      prompt: videoNode.data.i2v_prompt || "",
      imageName: "",
      imageData: "",
      promptConnected: false,
      imageConnected: false,
      sceneConnected: false,
      scene: null,
    };

    for (const link of incoming) {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      if (!source) continue;
      if (source.type === "sceneCard") {
        const scene = this.resolveSceneContext(source);
        resolved.scene = scene;
        resolved.sceneConnected = true;
        resolved.prompt = resolved.prompt || scene.videoPrompt || scene.motionNotes || scene.storyBeat || "";
      }
      if (source.type === "prompt") {
        resolved.prompt = source.data.prompt || "";
        resolved.promptConnected = true;
      }
      if (source.type === "imageRef") {
        resolved.imageName = source.data.imageName || "Image ref";
        resolved.imageData = source.data.imageData || "";
        resolved.imageConnected = true;
      }
      if (source.type === "loadImage") {
        resolved.imageName = source.data.imageName || "Loaded image";
        resolved.imageData = source.data.imageData || "";
        resolved.imageConnected = true;
      }
      if (source.type === "imageMode") {
        const image = this.resolveImageMode(source);
        resolved.imageName = image.imageName;
        resolved.imageData = image.imageData;
        resolved.imageConnected = true;
      }
    }
    return resolved;
  }

  resolveDisplayVideo(displayNode) {
    const incoming = this.graph.links.filter((link) => link.to === displayNode.id);
    const resolved = { model: "", prompt: "", imageData: "" };
    const videoLink = incoming.find((link) => {
      const source = this.graph.nodes.find((node) => node.id === link.from);
      return source?.type === "imageToVideo";
    });
    if (!videoLink) return resolved;

    const videoNode = this.graph.nodes.find((node) => node.id === videoLink.from);
    if (!videoNode) return resolved;
    const video = this.resolveImageToVideo(videoNode);
    resolved.model = video.model;
    resolved.prompt = video.prompt;
    resolved.imageData = video.imageData;
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
        min-height: 58px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 8px 18px;
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
        min-width: 0;
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

      .vrgdg-node-canvas.pan-mode {
        cursor: grab;
      }

      .vrgdg-node-canvas.pan-mode:active {
        cursor: grabbing;
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
        transform-origin: 0 0;
      }

      .vrgdg-node-toolstrip {
        position: absolute;
        left: 50%;
        bottom: 18px;
        transform: translateX(-50%);
        z-index: 4;
        display: flex;
        align-items: center;
        gap: 4px;
        height: 44px;
        padding: 5px;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 10px;
        background: rgba(29, 31, 35, 0.95);
        box-shadow: 0 12px 34px rgba(0,0,0,0.4);
      }

      .vrgdg-node-toolstrip button {
        width: 38px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid transparent;
        border-radius: 8px;
        background: transparent;
        color: #f2f5f7;
        font-size: 0;
        cursor: pointer;
      }

      .vrgdg-node-toolstrip button::before {
        font-size: 20px;
        line-height: 1;
      }

      .vrgdg-node-toolstrip button[data-tool="select"]::before {
        content: "↖";
      }

      .vrgdg-node-toolstrip button[data-tool="pan"]::before {
        content: "✋";
      }

      .vrgdg-node-toolstrip button:hover,
      .vrgdg-node-toolstrip button.active {
        background: #3a3d42;
        border-color: rgba(255,255,255,0.1);
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

      .vrgdg-node-card .vrgdg-node-body {
        max-height: 520px;
        overflow: auto;
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

      .vrgdg-node-setting {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-bottom: 10px;
      }

      .vrgdg-node-setting span {
        color: #aeb7c2;
        font-size: 12px;
        font-weight: 700;
      }

      .vrgdg-node-checkbox {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 9px;
        color: #d8dee6;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.3;
      }

      .vrgdg-node-checkbox input {
        width: 16px;
        height: 16px;
        accent-color: #13b8d0;
        flex: 0 0 auto;
      }

      .vrgdg-node-settings-section {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        background: rgba(255,255,255,0.025);
        padding: 10px;
        margin-bottom: 10px;
      }

      .vrgdg-node-settings-title {
        color: #cffafe;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 9px;
      }

      .vrgdg-node-action-row {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 10px;
      }

      .vrgdg-node-action-row button {
        border: 1px solid rgba(34,211,238,0.45);
        border-radius: 7px;
        background: #0e7490;
        color: #ecfeff;
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 900;
        cursor: pointer;
      }

      .vrgdg-node-action-row button:hover {
        background: #0891b2;
      }

      .vrgdg-node-tabs {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 5px;
        margin-bottom: 12px;
      }

      .vrgdg-node-tabs button {
        min-width: 0;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 6px;
        background: #15181d;
        color: #cfd6de;
        padding: 7px 4px;
        font-size: 11px;
        font-weight: 800;
        cursor: pointer;
      }

      .vrgdg-node-tabs button.active {
        background: #3b2459;
        border-color: rgba(181,108,255,0.55);
        color: #ffffff;
      }

      .vrgdg-node-textarea.compact {
        height: 118px;
      }

      .vrgdg-connection-notice {
        margin-bottom: 10px;
        border-radius: 6px;
        background: #15181d;
        color: #9da6b2;
        padding: 8px;
        font-size: 12px;
        font-weight: 700;
      }

      .vrgdg-connection-notice.connected {
        background: #132019;
        color: #8ff0b8;
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

      .vrgdg-node-drop.audio {
        border-color: rgba(245,158,11,0.38);
        background: #171411;
      }

      .vrgdg-audio-preview {
        width: 92%;
        height: 34px;
      }

      .vrgdg-node-drop.compact-output {
        height: 118px;
        margin-top: 10px;
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

      .vrgdg-node-video-preview {
        height: 150px;
        margin: 11px 0;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: #07090c;
        color: #6f7782;
        overflow: hidden;
      }

      .vrgdg-node-video-preview span {
        position: absolute;
        left: 10px;
        bottom: 9px;
        border-radius: 5px;
        background: rgba(0,0,0,0.62);
        color: #f3f6f8;
        padding: 5px 7px;
        font-size: 11px;
        font-weight: 800;
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
