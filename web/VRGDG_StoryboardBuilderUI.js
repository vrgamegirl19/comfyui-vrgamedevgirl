import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_StoryboardBuilderUI";
const HIDDEN_WIDGETS = new Set(["project_folder"]);

function hideInternalWidgets(node) {
  for (const widget of node.widgets || []) {
    if (!HIDDEN_WIDGETS.has(widget.name)) continue;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
  }
}

async function postJson(url, payload = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await api.fetchApi(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.ok === false) {
      throw new Error(data?.error || `Request failed (${response.status})`);
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

function makeButton(label, variant = "default") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  const bg = variant === "primary" ? "#12b5cb" : variant === "purple" ? "#0e7490" : "#2b2b30";
  const border = variant === "primary" ? "#0891b2" : variant === "purple" ? "#06b6d4" : "#3f3f46";
  button.style.cssText = `border:1px solid ${border};border-radius:6px;background:${bg};color:#f8fafc;padding:9px 13px;font-weight:800;cursor:pointer;`;
  return button;
}

function makeInput(value = "", placeholder = "") {
  const input = document.createElement("input");
  input.value = value || "";
  input.placeholder = placeholder;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#0b1220;color:#e5e7eb;padding:9px;font:12px monospace;";
  return input;
}

function makeTextarea(value = "", placeholder = "", rows = 4) {
  const textarea = document.createElement("textarea");
  textarea.value = value || "";
  textarea.placeholder = placeholder;
  textarea.rows = rows;
  textarea.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;border:1px solid #334155;border-radius:6px;background:#050814;color:#e5e7eb;padding:9px;font:12px monospace;line-height:1.45;";
  return textarea;
}

function makeSelect(options, value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;";
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    select.append(item);
  }
  select.value = value || options[0]?.value || "";
  return select;
}

function makeGroupedSelect(groups, value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;";
  for (const group of groups) {
    if (group.options) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group.label;
      for (const option of group.options) {
        const item = document.createElement("option");
        item.value = option.value ?? option;
        item.textContent = option.label ?? option;
        optgroup.append(item);
      }
      select.append(optgroup);
    } else {
      const item = document.createElement("option");
      item.value = group.value ?? "";
      item.textContent = group.label ?? "";
      select.append(item);
    }
  }
  select.value = value || "";
  return select;
}

function makeMultiSelect(options, values = []) {
  const select = document.createElement("select");
  select.multiple = true;
  select.size = Math.min(6, Math.max(3, options.length || 3));
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:7px;min-height:104px;";
  const selected = new Set(Array.isArray(values) ? values.map(String) : []);
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    item.selected = selected.has(String(option.value));
    select.append(item);
  }
  return select;
}

function makeCollapsiblePanel(title, summary = "", content = null, { open = false } = {}) {
  const panel = document.createElement("div");
  panel.style.cssText = "margin:8px 24px 0;border:1px solid #334155;border-radius:8px;background:#0f172a;overflow:hidden;";
  const header = document.createElement("button");
  header.type = "button";
  header.style.cssText = "width:100%;border:0;background:#0f172a;color:#e5e7eb;padding:9px 12px;display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center;text-align:left;cursor:pointer;";
  const caret = document.createElement("span");
  caret.style.cssText = "color:#67e8f9;font-size:13px;";
  const label = document.createElement("span");
  label.style.cssText = "font-weight:900;color:#cffafe;font-size:13px;white-space:nowrap;";
  label.textContent = title;
  const summaryNode = document.createElement("span");
  summaryNode.style.cssText = "color:#94a3b8;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  summaryNode.textContent = summary;
  const body = document.createElement("div");
  body.style.cssText = "border-top:1px solid #1f3347;padding:10px 12px;";
  if (content) body.append(content);
  let expanded = Boolean(open);
  const sync = () => {
    caret.textContent = expanded ? "▾" : "▸";
    body.style.display = expanded ? "" : "none";
  };
  header.onclick = () => {
    expanded = !expanded;
    sync();
  };
  header.append(caret, label, summaryNode);
  panel.append(header, body);
  panel.setSummary = (value) => {
    summaryNode.textContent = String(value || "");
  };
  panel.setOpen = (value) => {
    expanded = Boolean(value);
    sync();
  };
  panel.isOpen = () => expanded;
  sync();
  return panel;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function makeStoryboardImageUrl(path) {
  return `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}&rand=${Date.now()}`;
}

function storyboardReferenceImageSrc(image) {
  if (!image || typeof image !== "object") return "";
  const data = String(image.data || "").trim();
  if (data) return data.startsWith("data:") ? data : `data:image/png;base64,${data}`;
  const path = String(image.path || "").trim();
  return path ? makeStoryboardImageUrl(path) : "";
}

function normalizeReferenceImage(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const image = source.image && typeof source.image === "object" ? source.image : source;
  const hasTopLevelImage = Boolean(source.path || source.data || source.image_path || source.imagePath || source.image_data || source.imageData);
  return {
    path: String(image.path || source.image_path || source.imagePath || source.path || "").trim(),
    data: String(image.data || source.image_data || source.imageData || source.data || "").trim(),
    name: String(image.name || source.image_name || source.imageName || (hasTopLevelImage ? source.name : "") || "").trim(),
  };
}

function mergeReferenceImages(existing = {}, incoming = {}) {
  const left = normalizeReferenceImage(existing);
  const right = normalizeReferenceImage(incoming);
  return {
    path: right.path || left.path,
    data: right.data || left.data,
    name: right.name || left.name,
  };
}

function truncate(text, length = 130) {
  const clean = String(text || "").trim();
  if (clean.length <= length) return clean;
  return `${clean.slice(0, Math.max(0, length - 1)).trim()}...`;
}

function replaceLabeledPlanningLine(value, labelName, selectedValue) {
  const cleanLabel = String(labelName || "").trim();
  const cleanValue = String(selectedValue || "").trim();
  if (!cleanLabel || !cleanValue) return String(value || "").trim();
  const prefix = `${cleanLabel}:`;
  const replacement = `${prefix} ${cleanValue}.`;
  const lines = String(value || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => !line.trim().toLowerCase().startsWith(prefix.toLowerCase()));
  lines.push(replacement);
  return lines.map((line) => line.trim()).filter(Boolean).join("\n");
}

function tagsHtml(tags) {
  const list = Array.isArray(tags) ? tags : [];
  if (!list.length) return `<span style="color:#94a3b8;">-</span>`;
  return list.map((tag) => `<span style="display:inline-flex;border-radius:5px;background:#1e1b4b;color:#ddd6fe;padding:4px 7px;margin:2px;font-size:11px;">${escapeHtml(tag)}</span>`).join("");
}

function storyboardSubjectNamesFromRefs(subjectRefs = []) {
  return Array.from(new Set(
    (Array.isArray(subjectRefs) ? subjectRefs : [])
      .map((subject) => String(subject?.name || "").trim())
      .filter(Boolean)
  ));
}

function storyboardReferenceId(prefix, name = "") {
  const slug = String(name || prefix || "reference")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || prefix;
  return `${prefix}_story_${Date.now()}_${slug}`;
}

function readStoryboardImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file || !String(file.type || "").startsWith("image/")) {
      reject(new Error("Choose an image file."));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read that image file."));
    reader.readAsDataURL(file);
  });
}

const IMAGE_SHOT_TYPES = [
  "close-up shot",
  "extreme close-up shot",
  "medium close-up shot",
  "medium shot",
  "medium wide shot",
  "wide shot",
  "extreme wide shot",
  "full shot",
  "long shot",
  "extreme long shot",
  "establishing shot",
  "master shot",
  "two-shot",
  "three-shot",
  "over-the-shoulder shot",
  "point-of-view shot",
  "first-person shot",
  "insert shot",
  "cutaway shot",
  "reaction shot",
  "detail shot",
  "beauty shot",
  "hero shot",
  "profile shot",
  "frontal shot",
  "rear shot",
  "side shot",
  "low-angle shot",
  "high-angle shot",
  "eye-level shot",
  "bird's-eye view shot",
  "worm's-eye view shot",
  "aerial shot",
  "drone shot",
  "overhead shot",
  "top-down shot",
  "ground-level shot",
  "Dutch angle shot",
  "tilted shot",
  "symmetrical shot",
  "centered shot",
  "off-center shot",
  "silhouette shot",
  "reflection shot",
  "mirror shot",
  "shadow shot",
  "through-the-window shot",
  "through-the-doorway shot",
  "frame-within-a-frame shot",
  "single shot",
  "two-person shot",
  "group shot",
  "crowd shot",
  "face shot",
  "head shot",
  "head-and-shoulders shot",
  "bust shot",
  "waist-up shot",
  "chest-up shot",
  "knee-up shot",
  "cowboy shot",
  "American shot",
  "full-body shot",
  "feet shot",
  "hands shot",
  "eyes shot",
  "mouth shot",
  "object shot",
  "product shot",
  "environment shot",
  "landscape shot",
  "cityscape shot",
  "room shot",
  "hallway shot",
  "doorway shot",
  "car interior shot",
  "dashboard shot",
  "passenger-seat shot",
  "driver-seat shot",
  "cinematic wide shot",
  "moody close-up shot",
  "dramatic low-angle shot",
  "intimate close-up shot",
  "documentary-style shot",
  "surveillance-style shot",
  "security-camera shot",
  "CCTV shot",
  "found-footage shot",
  "vlog-style shot",
  "selfie shot",
  "webcam shot",
  "interview shot",
  "talking-head shot",
  "news-style shot",
  "broadcast-style shot",
  "commercial product shot",
  "lifestyle shot",
  "montage opening shot",
  "transition shot",
  "dreamlike shot",
  "blurred foreground shot",
  "shallow-depth-of-field shot",
  "deep-focus shot",
  "soft-focus shot",
  "backlit shot",
  "lens-flare shot",
  "natural-light shot",
  "night shot",
  "golden-hour shot",
  "blue-hour shot",
];

const VIDEO_SHOT_TYPES = [
  ...IMAGE_SHOT_TYPES,
  "static shot",
  "locked-off shot",
  "handheld shot",
  "tracking shot",
  "dolly shot",
  "dolly-in shot",
  "dolly-out shot",
  "push-in shot",
  "pull-out shot",
  "zoom-in shot",
  "zoom-out shot",
  "pan shot",
  "whip pan shot",
  "tilt-up shot",
  "tilt-down shot",
  "crane shot",
  "jib shot",
  "Steadicam shot",
  "gimbal shot",
  "follow shot",
  "lead shot",
  "arc shot",
  "orbit shot",
  "360-degree shot",
  "reveal shot",
  "rack-focus shot",
  "focus-pull shot",
  "slow-motion shot",
  "time-lapse shot",
  "hyperlapse shot",
];

const CAMERA_MOTION_GROUPS = [
  { value: "", label: "Choose camera motion..." },
  {
    label: "Basic Camera Motions",
    options: [
      "pan left", "pan right", "pan up", "pan down", "tilt up", "tilt down",
      "push in", "pull back", "pull out", "dolly in", "dolly out",
      "dolly left", "dolly right", "truck left", "truck right",
      "pedestal up", "pedestal down", "zoom in", "zoom out",
      "slow zoom in", "slow zoom out", "quick zoom in", "snap zoom",
      "crash zoom", "whip pan", "whip left", "whip right", "whip up", "whip down",
    ],
  },
  {
    label: "Orbit / Rotation Motions",
    options: [
      "orbit left", "orbit right", "orbit around subject", "rotate around subject",
      "circle around subject", "180-degree rotation", "360-degree rotation",
      "half-circle orbit", "full-circle orbit", "clockwise orbit",
      "counterclockwise orbit", "spiral around subject", "arc left", "arc right",
      "arc around subject", "wraparound move", "sweeping circular move",
    ],
  },
  {
    label: "Tracking / Following Motions",
    options: [
      "track forward", "track backward", "track left", "track right",
      "tracking shot", "follow shot", "follow behind", "follow in front",
      "lead shot", "side-follow shot", "over-the-shoulder follow",
      "chase shot", "pursuit shot", "walk-and-talk tracking",
      "handheld follow", "gimbal follow", "steadicam follow",
      "smooth follow", "shaky follow",
    ],
  },
  {
    label: "Reveal Motions",
    options: [
      "reveal upward", "reveal downward", "reveal left", "reveal right",
      "slide reveal", "dolly reveal", "pan reveal", "tilt reveal",
      "pull-back reveal", "push-in reveal", "orbit reveal", "crane reveal",
      "rack-focus reveal", "foreground reveal", "doorway reveal",
      "window reveal", "object reveal", "character reveal", "environment reveal",
    ],
  },
  {
    label: "Vertical / Height Motions",
    options: [
      "crane up", "crane down", "jib up", "jib down", "rise up",
      "descend down", "boom up", "boom down", "lift upward", "drop downward",
      "float upward", "sink downward", "aerial rise", "aerial descent",
      "drone rise", "drone descend", "top-down descent",
      "ground-to-sky tilt", "sky-to-ground tilt",
    ],
  },
  {
    label: "Drone / Aerial Motions",
    options: [
      "drone flyover", "drone push in", "drone pull back", "drone rise",
      "drone descend", "drone orbit", "drone circle", "drone follow",
      "drone chase", "drone pass-through", "drone reveal", "aerial tracking",
      "aerial pan", "aerial tilt", "overhead drift", "top-down tracking",
      "bird's-eye pullback", "sweeping aerial move",
    ],
  },
  {
    label: "Handheld / Style Motions",
    options: [
      "handheld shake", "subtle handheld movement", "shaky cam",
      "smooth handheld", "floating camera move", "drifting camera move",
      "breathing camera movement", "documentary-style movement",
      "natural handheld sway", "nervous handheld move", "chaotic handheld move",
      "stabilized gimbal move", "steadicam glide", "slow cinematic glide",
      "smooth cinematic drift",
    ],
  },
  {
    label: "Focus / Lens Motions",
    options: [
      "rack focus", "focus pull", "focus shift",
      "foreground-to-background focus", "background-to-foreground focus",
      "shallow-focus drift", "zoom with focus pull", "dolly zoom",
      "vertigo effect", "crash zoom with focus", "soft-focus transition",
      "focus reveal",
    ],
  },
  {
    label: "POV / Subjective Motions",
    options: [
      "POV walk forward", "POV turn left", "POV turn right", "POV look up",
      "POV look down", "POV stumble", "POV run", "POV chase", "POV fall",
      "POV rise", "POV scan the room", "POV peek around corner",
      "POV lean in", "POV look over shoulder",
    ],
  },
  {
    label: "Transition Motions",
    options: [
      "whip pan transition", "match move", "push-through transition",
      "pass-through transition", "foreground wipe", "camera wipe",
      "object wipe", "spin transition", "rotate transition", "zoom transition",
      "crash zoom transition", "tilt transition", "pan transition",
      "motion blur transition",
    ],
  },
];

const STILL_CAMERA_STYLE_GROUPS = [
  { value: "", label: "Choose still camera style..." },
  {
    label: "Composition / Framing",
    options: [
      "clean portrait composition", "editorial fashion composition", "cinematic still frame",
      "rule-of-thirds composition", "centered symmetrical composition", "negative space composition",
      "foreground framing", "frame-within-a-frame composition", "environmental portrait",
      "intimate close portrait", "wide environmental still", "dramatic silhouette composition",
    ],
  },
  {
    label: "Lens / Depth",
    options: [
      "shallow depth of field", "deep focus photography", "soft background bokeh",
      "wide-angle perspective", "telephoto compression", "macro detail photography",
      "natural lens perspective", "cinematic anamorphic lens look", "soft-focus portrait lens",
      "crisp studio lens detail",
    ],
  },
  {
    label: "Lighting / Exposure",
    options: [
      "natural window light", "golden-hour photography", "blue-hour photography",
      "high-contrast studio lighting", "soft diffused key light", "dramatic rim lighting",
      "backlit portrait", "low-key lighting", "high-key photography",
      "moody practical lighting", "neon-lit still photography",
    ],
  },
  {
    label: "Still Photography Style",
    options: [
      "editorial magazine photo", "fine-art portrait photography", "documentary still photo",
      "album-cover photography", "cinematic production still", "glossy commercial photo",
      "gritty street photography", "dreamlike fashion editorial", "dramatic character portrait",
      "atmospheric location photography",
    ],
  },
];

const CHARACTER_MOTION_GROUPS = [
  { value: "", label: "Choose character motion..." },
  {
    label: "Basic Locomotion",
    options: [
      "standing still", "walking", "running", "jogging", "sprinting", "pacing",
      "strolling", "wandering", "marching", "limping", "sneaking", "crawling",
      "climbing", "jumping", "landing", "falling", "tripping", "stumbling",
      "sliding", "spinning", "turning around", "looking around", "backing away",
      "moving forward", "moving sideways", "approaching camera",
      "walking away from camera",
    ],
  },
  {
    label: "Dance / Performance",
    options: [
      "dancing", "freestyle dancing", "slow dancing", "breakdancing",
      "hip-hop dancing", "club dancing", "swaying to music", "head nodding",
      "shoulder bouncing", "foot tapping", "hand waving", "arm swinging",
      "body rolling", "spinning while dancing", "jumping to the beat",
      "performing on stage", "singing into microphone", "rapping into microphone",
      "playing guitar", "playing piano", "playing drums", "DJing", "crowd surfing",
    ],
  },
  {
    label: "Gestures",
    options: [
      "pointing", "waving", "clapping", "snapping fingers", "giving thumbs up",
      "crossing arms", "raising arms", "reaching out", "holding hands up",
      "covering face", "touching chest", "touching head", "brushing hair back",
      "adjusting jacket", "adjusting sunglasses", "putting hands in pockets",
      "throwing hands up", "making hand signs", "beckoning", "saluting",
    ],
  },
  {
    label: "Facial Expression / Head Movement",
    options: [
      "smiling", "laughing", "crying", "frowning", "smirking", "shouting",
      "whispering", "looking at camera", "looking away",
      "looking down", "looking up", "turning head", "tilting head", "nodding",
      "shaking head", "closing eyes", "opening eyes", "blinking",
      "staring intensely",
    ],
  },
  {
    label: "Environment Interaction",
    options: [
      "opening a door", "closing a door", "leaning on a wall", "sitting on a chair",
      "standing up", "sitting down", "lying down", "kneeling", "picking something up",
      "dropping something", "throwing something", "pushing something",
      "pulling something", "carrying something", "leaning over a railing",
      "looking out a window", "walking through smoke", "walking through rain",
      "splashing through water", "kicking dust", "touching a wall",
      "running fingers along a surface",
    ],
  },
  {
    label: "Object Interaction",
    options: [
      "holding microphone", "holding phone", "looking at phone", "taking a photo",
      "recording video", "holding flowers", "holding money", "counting money",
      "holding a drink", "drinking", "smoking", "lighting a cigarette",
      "wearing headphones", "putting on headphones", "removing sunglasses",
      "putting on sunglasses", "holding a weapon prop", "holding a bag",
      "carrying luggage", "tossing keys", "spinning keys", "reading a note",
    ],
  },
  {
    label: "Emotional Action",
    options: [
      "collapsing to knees", "reaching toward camera", "running away",
      "chasing someone", "being chased", "searching for someone", "hiding",
      "waiting", "hesitating", "reacting in shock", "celebrating", "arguing",
      "fighting", "hugging", "pushing away", "walking alone",
      "standing in silence", "looking heartbroken", "looking confident",
      "looking angry", "looking lost",
    ],
  },
  {
    label: "Camera-Facing Motion",
    options: [
      "walking toward camera", "walking past camera", "turning to face camera",
      "looking directly into lens", "reaching toward lens", "pointing at camera",
      "singing to camera", "dancing toward camera", "moving in slow motion",
      "freezing in place", "silhouette movement", "hair blowing in wind",
      "clothing flowing in wind", "walking through frame", "entering frame",
      "exiting frame", "crossing foreground", "moving in background",
    ],
  },
  {
    label: "Group Movement",
    options: [
      "crowd dancing", "crowd jumping", "crowd waving arms", "crowd clapping",
      "people walking around", "people running past", "group marching",
      "group circling character", "group following character",
      "group surrounding character", "backup dancers performing",
      "band performing", "audience cheering", "friends walking together",
      "couple dancing", "couple arguing", "couple embracing",
    ],
  },
  {
    label: "Vehicle / Travel",
    options: [
      "driving", "riding in car", "getting into car", "getting out of car",
      "leaning out car window", "walking beside car", "sitting on car hood",
      "riding motorcycle", "riding bicycle", "skateboarding", "roller skating",
      "riding elevator", "walking down stairs", "walking up stairs",
      "riding escalator", "running through tunnel", "walking across street",
    ],
  },
  {
    label: "Stylized / Surreal Motion",
    options: [
      "floating", "levitation", "falling in slow motion", "spinning in place",
      "walking in reverse", "glitching", "teleporting", "duplicating",
      "morphing pose", "freeze-frame pose", "dramatic turn", "slow-motion walk",
      "wind-swept pose", "hero pose", "shadow dancing", "silhouette dancing",
      "smoke reveal", "light reveal", "walking through sparks",
      "dancing in rain", "falling backward into darkness", "reaching through light",
      "moving like a puppet", "robotic movement", "fluid dreamlike movement",
    ],
  },
];

export const STORYBOARD_CAMERA_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill missing shot or camera motion fields.",
    sequence: [],
  },
  balanced: {
    label: "Balanced cinematic flow",
    description: "Alternates wide, medium, close, lateral, reveal, and reset shots without repeating inward zooms.",
    sequence: [
      { shot: "wide shot", camera: "slow cinematic drift" },
      { shot: "medium close-up shot", camera: "pull back" },
      { shot: "tracking shot", camera: "side-follow shot" },
      { shot: "close-up shot", camera: "slow orbit left" },
      { shot: "medium wide shot", camera: "dolly right" },
      { shot: "profile shot", camera: "pan reveal" },
      { shot: "low-angle shot", camera: "crane up" },
      { shot: "intimate close-up shot", camera: "slow zoom out" },
      { shot: "over-the-shoulder shot", camera: "reveal right" },
      { shot: "full-body shot", camera: "track backward" },
    ],
  },
  music_video: {
    label: "Music video movement",
    description: "More performance energy with tracking, handheld, whip, reveal, lateral moves, and orbit changes.",
    sequence: [
      { shot: "medium shot", camera: "handheld follow" },
      { shot: "medium close-up shot", camera: "handheld follow" },
      { shot: "wide shot", camera: "whip pan transition" },
      { shot: "close-up shot", camera: "orbit right" },
      { shot: "low-angle shot", camera: "track left" },
      { shot: "side shot", camera: "track left" },
      { shot: "full-body shot", camera: "smooth follow" },
      { shot: "reaction shot", camera: "rack focus" },
      { shot: "dramatic low-angle shot", camera: "crane reveal" },
      { shot: "moody close-up shot", camera: "drifting camera move" },
    ],
  },
  quiet: {
    label: "Quiet dramatic",
    description: "Slower restrained camera choices for emotional, eerie, or cinematic scenes.",
    sequence: [
      { shot: "establishing shot", camera: "slow cinematic drift" },
      { shot: "medium wide shot", camera: "locked-off shot" },
      { shot: "profile shot", camera: "subtle handheld movement" },
      { shot: "intimate close-up shot", camera: "slow zoom out" },
      { shot: "reflection shot", camera: "focus pull" },
      { shot: "centered shot", camera: "pull back" },
      { shot: "silhouette shot", camera: "tilt up" },
      { shot: "close-up shot", camera: "drifting camera move" },
    ],
  },
  energetic: {
    label: "Fast energetic",
    description: "Bigger changes between scenes with fast moves, reveals, tracking, and punchier reframing.",
    sequence: [
      { shot: "wide shot", camera: "whip pan transition" },
      { shot: "medium shot", camera: "track left" },
      { shot: "close-up shot", camera: "whip right" },
      { shot: "low-angle shot", camera: "orbit reveal" },
      { shot: "full-body shot", camera: "dolly left" },
      { shot: "Dutch angle shot", camera: "push-through transition" },
      { shot: "medium wide shot", camera: "crane up" },
      { shot: "reaction shot", camera: "rack focus" },
      { shot: "tracking shot", camera: "chase shot" },
      { shot: "detail shot", camera: "snap zoom" },
    ],
  },
};

export const PERFORMANCE_STYLE_PRESETS = [
  {
    value: "",
    label: "Default cinematic",
    direction: "Use a natural cinematic music-video performance with visible emotion, expressive face, motivated body language, and camera energy that fits the scene.",
  },
  {
    value: "rock_punk",
    label: "Rock / punk",
    direction: "Use raw rock performance energy: intense facial emotion, head movement, sharp gestures, defiant posture, and gritty stage-like body language.",
  },
  {
    value: "metal_screaming",
    label: "Metal / screaming",
    direction: "Use aggressive high-intensity performance energy: fierce expression, powerful stance, forceful gestures, hair and clothing reacting to motion, and heavy dramatic presence.",
  },
  {
    value: "rap_hiphop",
    label: "Rap / hip-hop",
    direction: "Use rap-style energy instead of soft singing: confident direct-to-camera presence, expressive hand gestures, head nods, shoulder movement, and sharper body language.",
  },
  {
    value: "pop_performance",
    label: "Pop performance",
    direction: "Use polished pop performance energy: expressive singing, clean confident movement, controlled gestures, direct eye contact, stylish body language, and camera-friendly emotion.",
  },
  {
    value: "ballad_emotional",
    label: "Ballad / emotional",
    direction: "Use emotional ballad performance energy: vulnerable facial expression, slower gestures, longing eyes, subtle hand movement, restrained body language, and intimate camera presence.",
  },
  {
    value: "rnb_smooth",
    label: "R&B / smooth",
    direction: "Use smooth R&B performance energy: relaxed confident expression, controlled sensual movement, gentle hand gestures, soft rhythmic body motion, and close emotional intensity.",
  },
  {
    value: "edm_club",
    label: "EDM / club",
    direction: "Use energetic club performance energy: rhythmic movement, dance-like gestures, bright reactive expression, beat-driven body language, and dynamic camera motion.",
  },
  {
    value: "spoken_word",
    label: "Spoken word",
    direction: "Use spoken-word energy instead of singing: focused eyes, intentional gestures, restrained intensity, and poetic performance presence.",
  },
  {
    value: "no_vocals_broll",
    label: "No vocals / B-roll",
    direction: "Do not include singing, rapping, speaking, lip-sync, mouth movement, microphones, or vocal performance. Use visual action, environment interaction, and mood-driven movement only.",
  },
];

export function storyboardPerformancePreset(value = "") {
  return PERFORMANCE_STYLE_PRESETS.find((item) => item.value === value) || PERFORMANCE_STYLE_PRESETS[0];
}

export const FACIAL_PERFORMANCE_PRESETS = [
  {
    value: "",
    label: "Default natural",
    description: "Natural expressive face",
    direction: "Use natural expressive facial performance: engaged eyes, subtle natural eye movement, active brows, subtle cheek and jaw movement, visible emotion that fits the lyric or scene, and occasional natural blinking.",
  },
  {
    value: "pop_polished",
    label: "Pop / polished stage",
    description: "Camera-ready pop emotion",
    direction: "Use polished pop-star facial performance: bright eyes, subtle natural eye movement, direct camera gaze, soft confident smile, playful smirk, relaxed brows, slight head tilts, lips slightly parted while singing, charming camera-ready expression, and occasional natural blinking.",
  },
  {
    value: "pop_flirty",
    label: "Pop / playful flirty",
    description: "Playful, charming pop face",
    direction: "Use playful pop facial performance: flirty smile, coy glance, subtle natural eye movement, light pout, glossy pout, raised brows, charming direct gaze, playful smirk, subtle head tilt, lips slightly parted while singing, and occasional natural blinking.",
  },
  {
    value: "love_tender",
    label: "Love song / tender",
    description: "Soft romantic expression",
    direction: "Use tender love-song facial performance: softened eyes, subtle natural eye movement, warm smile, affectionate gaze, raised inner brows, gentle head tilt, relaxed cheeks, subtle vulnerable emotion, and occasional natural blinking.",
  },
  {
    value: "sad_wounded",
    label: "Sad / wounded",
    description: "Grief, hurt, vulnerability",
    direction: "Use wounded sad-song facial performance: lowered gaze, heavy or watery eyes, subtle natural eye movement, raised inner brows, pinched brows, downturned mouth, trembling lips or chin when appropriate, defeated expression, and occasional natural blinking.",
  },
  {
    value: "happy_joyful",
    label: "Happy / joyful",
    description: "Bright and joyful",
    direction: "Use joyful facial performance: bright smile, smiling eyes, subtle natural eye movement, raised cheeks, delighted expression, playful gaze, lifted mouth corners, relaxed brows, head tilt with smile, and occasional natural blinking.",
  },
  {
    value: "rock_intense",
    label: "Rock / intense",
    description: "Gritty rock intensity",
    direction: "Use intense rock facial performance: focused stare, subtle natural eye movement, furrowed brows, defiant smirk, clenched jaw, gritty emotional strain, sharp eye contact, forceful singing expression, and occasional natural blinking.",
  },
  {
    value: "metal_rage",
    label: "Metal / rage",
    description: "Aggressive heavy metal face",
    direction: "Use aggressive heavy metal facial performance: fierce stare, subtle natural eye movement, furrowed brows, wild eyes, clenched jaw, snarling mouth shapes during vocals, bared teeth on powerful notes, flared nostrils, strained neck intensity, raw emotional scream expression, and occasional natural blinking.",
  },
  {
    value: "rap_high_intensity",
    label: "Rap / high intensity",
    description: "Sharp rap delivery",
    direction: "Use high-intensity rap facial performance: intense stare, sharp eye contact, subtle natural eye movement, furrowed brows, animated eyes, confident smirk, tight jaw, mouth open mid-verse, fast-moving mouth during delivery, challenging look, victory grin, and occasional natural blinking.",
  },
  {
    value: "custom",
    label: "Custom",
    description: "Use custom facial text",
    direction: "",
  },
];

export function storyboardFacialPerformancePreset(value = "") {
  return FACIAL_PERFORMANCE_PRESETS.find((item) => item.value === value) || FACIAL_PERFORMANCE_PRESETS[0];
}

export const ID_LORA_PERFORMANCE_STYLE_PRESETS = [
  {
    value: "dialogue_naturalism",
    label: "Dialogue naturalism",
    direction: "Use grounded short-film acting: conversational timing, motivated gestures, lived-in posture, subtle emotional shifts, and behavior that feels observed rather than performed.",
  },
  {
    value: "tense_confrontation",
    label: "Tense confrontation",
    direction: "Use restrained confrontation energy: clipped gestures, guarded posture, controlled anger, charged pauses, and body language that suggests pressure under the surface.",
  },
  {
    value: "indie_drama",
    label: "Indie drama",
    direction: "Use intimate indie-film acting: small revealing gestures, vulnerable stillness, natural imperfections, quiet tension, and emotionally specific reactions.",
  },
  {
    value: "noir_restraint",
    label: "Noir restraint",
    direction: "Use noir-style restraint: low-key confidence, suspicious glances, minimal gestures, guarded delivery, and tension carried through posture and eyes.",
  },
  {
    value: "comedic_awkwarness",
    label: "Comedic awkward",
    direction: "Use dry comedic acting: awkward pauses, slightly mismatched reactions, contained embarrassment, small nervous gestures, and believable conversational timing.",
  },
  {
    value: "emotional_confession",
    label: "Emotional confession",
    direction: "Use confession-scene acting: exposed emotion, hesitant gestures, wavering confidence, visible vulnerability, and a line delivery that feels personally risky.",
  },
  {
    value: "suspense_dread",
    label: "Suspense dread",
    direction: "Use suspense-film tension: alert posture, careful stillness, anxious scanning, controlled breathing, and reactions that imply something important is about to break.",
  },
  {
    value: "punk_bar_attitude",
    label: "Punk bar attitude",
    direction: "Use gritty punk-bar acting: defiant posture, sharp side-eye, casual toughness, impatient gestures, and messy lived-in confidence without turning it into a stage performance.",
  },
];

export const ID_LORA_FACIAL_PERFORMANCE_PRESETS = [
  {
    value: "",
    label: "Default screen acting",
    description: "Natural film face",
    direction: "Use grounded screen-acting facial detail: attentive eyes, small brow changes, readable thought, subtle jaw tension, natural mouth shapes for speech, and emotion that fits the dialogue.",
  },
  {
    value: "curious_inquisitive",
    label: "Curious / inquisitive",
    description: "Curious screen expression",
    direction: "Use curious facial performance: bright attentive eyes, slight head angle, lifted brow, searching gaze, relaxed mouth between words, and a sense of active listening.",
  },
  {
    value: "guarded_suspicious",
    label: "Guarded / suspicious",
    description: "Guarded tension",
    direction: "Use guarded facial performance: narrowed eyes, tight jaw, controlled mouth, skeptical brow, held gaze, and restrained suspicion under the dialogue.",
  },
  {
    value: "defiant_controlled",
    label: "Defiant / controlled",
    description: "Controlled defiance",
    direction: "Use controlled defiance: steady eye contact, tense mouth corners, lifted chin, compressed jaw, and a look that refuses to back down.",
  },
  {
    value: "vulnerable_confession",
    label: "Vulnerable confession",
    description: "Exposed emotion",
    direction: "Use vulnerable confession facial performance: softened eyes, raised inner brows, small uncertain mouth movements, visible hesitation, and emotion barely held together.",
  },
  {
    value: "dry_comedic",
    label: "Dry comedic",
    description: "Subtle comedy face",
    direction: "Use dry comedic facial performance: tiny reaction beats, restrained disbelief, awkward half-smile, quick eye shifts, and understated embarrassment.",
  },
  {
    value: "custom",
    label: "Custom",
    description: "Use custom facial text",
    direction: "",
  },
];

function storyboardMotionFamily(motion = "") {
  const text = String(motion || "").toLowerCase();
  if (/push|dolly in|zoom in|track forward|crash zoom|snap zoom/.test(text)) return "in";
  if (/pull|dolly out|zoom out|track backward/.test(text)) return "out";
  if (/orbit|arc|circle|rotation|rotate/.test(text)) return "orbit";
  if (/track|follow|dolly left|dolly right|truck/.test(text)) return "track";
  if (/reveal|tilt|crane|jib|rise|descend/.test(text)) return "reveal";
  if (/focus|rack/.test(text)) return "focus";
  return text.split(/\s+/).slice(0, 2).join(" ");
}

export function storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion = "") {
  const preset = STORYBOARD_CAMERA_FLOW_PRESETS[profileKey] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
  const sequence = preset.sequence || [];
  if (!sequence.length) return null;
  let entry = sequence[sceneIndex % sequence.length];
  if (previousMotion && storyboardMotionFamily(entry.camera) === storyboardMotionFamily(previousMotion)) {
    entry = sequence[(sceneIndex + 1) % sequence.length] || entry;
  }
  return entry;
}

export const STORYBOARD_IMAGE_SHOT_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill still-image shot/composition fields.",
    sequence: [],
  },
  intimate: {
    label: "Intimate character shots",
    description: "Close, emotional stills for faces, hands, expressions, and quiet character moments.",
    sequence: [
      "intimate close-up shot",
      "medium close-up shot",
      "eyes shot",
      "hands shot",
      "profile shot",
      "head-and-shoulders shot",
      "reflection shot",
      "moody close-up shot",
    ],
  },
  music_video_stills: {
    label: "Music video stills",
    description: "Album-cover and performance-friendly framing with cinematic variety but no camera movement.",
    sequence: [
      "medium shot",
      "low-angle shot",
      "wide shot",
      "hero shot",
      "Dutch angle shot",
      "silhouette shot",
      "full-body shot",
      "dramatic low-angle shot",
      "centered shot",
      "beauty shot",
    ],
  },
  editorial: {
    label: "Editorial fashion",
    description: "Stylized portrait, fashion, and magazine-like compositions.",
    sequence: [
      "editorial fashion composition",
      "beauty shot",
      "full-body shot",
      "profile shot",
      "wide environmental still",
      "centered symmetrical composition",
      "negative space composition",
      "commercial product shot",
    ],
  },
  cinematic_story: {
    label: "Cinematic story frames",
    description: "Film-still composition for locations, story beats, and emotionally readable scenes.",
    sequence: [
      "establishing shot",
      "medium wide shot",
      "over-the-shoulder shot",
      "frame-within-a-frame shot",
      "environment shot",
      "reflection shot",
      "silhouette shot",
      "detail shot",
      "wide shot",
    ],
  },
  film_dialogue_coverage: {
    label: "Film dialogue coverage",
    description: "Short-film coverage for story-heavy music videos: readable faces, eyelines, reactions, and location context.",
    sequence: [
      "medium close-up dialogue shot",
      "over-the-shoulder shot",
      "reaction close-up",
      "two-shot dialogue frame",
      "profile close-up",
      "medium shot with foreground framing",
      "insert detail shot",
      "wide establishing film still",
    ],
  },
  intimate_drama: {
    label: "Intimate drama frames",
    description: "Close emotional film stills for confessions, quiet tension, and character-led music-video scenes.",
    sequence: [
      "tight close-up",
      "intimate medium close-up",
      "profile close-up",
      "hands and face detail shot",
      "reflection close-up",
      "seated conversation frame",
      "shallow-focus reaction shot",
      "low-key portrait frame",
    ],
  },
  noir_story_frames: {
    label: "Noir story frames",
    description: "Moody dramatic coverage with shadows, silhouettes, foregrounds, and tense blocking.",
    sequence: [
      "low-key medium shot",
      "silhouette dialogue frame",
      "over-the-shoulder noir shot",
      "frame-within-a-frame shot",
      "side-lit profile shot",
      "wide empty-space composition",
      "reflection shot",
      "detail insert shot",
    ],
  },
};

export const ID_LORA_IMAGE_SHOT_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill film-still composition fields.",
    sequence: [],
  },
  film_dialogue_coverage: {
    label: "Film dialogue coverage",
    description: "Short-film coverage for dialogue scenes: readable faces, eyelines, reactions, and location context.",
    sequence: [
      "medium close-up dialogue shot",
      "over-the-shoulder shot",
      "reaction close-up",
      "two-shot dialogue frame",
      "profile close-up",
      "medium shot with foreground framing",
      "insert detail shot",
      "wide establishing film still",
    ],
  },
  intimate_drama: {
    label: "Intimate drama frames",
    description: "Close emotional film stills for confessions, quiet tension, and character-led scenes.",
    sequence: [
      "tight close-up",
      "intimate medium close-up",
      "profile close-up",
      "hands and face detail shot",
      "reflection close-up",
      "seated conversation frame",
      "shallow-focus reaction shot",
      "low-key portrait frame",
    ],
  },
  noir_story_frames: {
    label: "Noir story frames",
    description: "Moody dramatic coverage with shadows, silhouettes, foregrounds, and tense blocking.",
    sequence: [
      "low-key medium shot",
      "silhouette dialogue frame",
      "over-the-shoulder noir shot",
      "frame-within-a-frame shot",
      "side-lit profile shot",
      "wide empty-space composition",
      "reflection shot",
      "detail insert shot",
    ],
  },
};

export const STORYBOARD_IMAGE_AESTHETIC_PRESETS = [
  { value: "", label: "Default cinematic still", description: "Balanced cinematic lighting, color, and texture for a polished text-to-image prompt.", prompt_guidance: "Create a polished cinematic still with clear subject placement, believable wardrobe and environment details, purposeful lighting, readable composition, lens/framing detail, and a strong music-video production still feeling." },
  { value: "music_video_gloss", label: "Glossy music video", description: "Glossy high-production music-video still, dramatic color contrast, stylish lighting, album-cover polish.", prompt_guidance: "Build a glossy high-production music-video still. Specify stylized wardrobe, intentional pose, dramatic color contrast, polished hair and makeup, expensive-looking lighting, reflective or atmospheric set details, album-cover composition, crisp lens choice, and cinematic depth. Do not merely say glossy music video." },
  { value: "dark_neon", label: "Dark neon", description: "Dark cinematic neon lighting, saturated color accents, glossy reflections, smoky atmosphere, night-club energy.", prompt_guidance: "Build a dark neon cinematic still. Use saturated colored light sources, glossy reflections, wet or polished surfaces, smoke/haze, rim light, deep shadows, vivid accent colors on the subject, and a nightlife or futuristic music-video atmosphere. Describe where the neon comes from and how it shapes the face, outfit, and environment." },
  { value: "editorial_fashion", label: "Editorial fashion", description: "High-fashion editorial photography, intentional posing, refined wardrobe detail, magazine-grade lighting.", prompt_guidance: "Build an editorial fashion photograph, not a plain portrait. Give the subject a deliberate model pose with body angles, hand placement, posture, and gaze. Describe refined wardrobe styling, fabric behavior, accessories, hair/makeup direction, fashion-magazine lighting, background styling, composition, lens/framing, and a strong art-directed theme." },
  { value: "editorial_fashion_photography", label: "Editorial fashion photography", description: "Editorial fashion photography with confident model posing, dramatic styling, creative wardrobe themes, magazine-grade composition, bold makeup and hair, and polished high-resolution lighting.", prompt_guidance: "Build a detailed editorial fashion photograph. Include a confident model pose, strong body line, hand/shoulder/hip placement, dramatic styling choices, creative wardrobe concept, fabric texture and silhouette, bold hair and makeup, accessories, modern magazine composition, art-directed setting, high-resolution studio or location lighting, and a clear fashion story. Do not just write 'editorial fashion composition'." },
  { value: "conceptual_portrait_photography", label: "Conceptual portrait photography", description: "Conceptual portrait photography built around a clear visual idea, symbolic prop, emotional pose, controlled environment, cinematic lighting, and a strong central portrait composition.", prompt_guidance: "Build a conceptual portrait around one clear visual idea. Choose a symbolic prop, object arrangement, or environmental metaphor that fits the scene. Describe the subject's pose, relation to the prop, wardrobe, hair/makeup, controlled setting, color palette, lighting direction, mood, lens/framing, and how the composition communicates the concept visually without explaining it." },
  { value: "avant_garde_fashion_photography", label: "Avant-garde fashion photography", description: "Avant-garde fashion photography with unusual makeup, sculptural hair, strange or powerful poses, experimental styling, abstract studio or surreal setting, and bold high-contrast lighting.", prompt_guidance: "Build an avant-garde fashion photograph. Use unusual makeup, sculptural or geometric hair, experimental wardrobe shape, exaggerated silhouette, strange powerful pose, asymmetrical composition, abstract studio or surreal set design, hard shadows or high-contrast light, unexpected materials, and a bold futuristic or theatrical fashion mood. Make it visually daring, not casual." },
  { value: "beauty_editorial_photography", label: "Beauty editorial photography", description: "Beauty editorial photography focused on close-up makeup, hair, skin texture, eyes, lips, jewelry or face details, soft luxury lighting, and clean magazine beauty composition.", prompt_guidance: "Build a beauty editorial photograph. Use close-up or tight portrait framing focused on eyes, lips, makeup, hair texture, jewelry, nails, skin glow, and face-framing styling. Describe makeup colors, glossy or matte finish, hair placement, accessories near the face, soft diffused lighting, clean backdrop, shallow depth of field, and luxury magazine composition." },
  { value: "high_fashion_editorial", label: "High fashion editorial", description: "High fashion editorial photography inspired by dramatic fashion competition shoots: couture wardrobe, expressive posing, epic location, wind or fabric movement, glamorous styling, and cinematic full-body framing.", prompt_guidance: "Build a high fashion editorial shoot like a dramatic fashion competition photo. Use couture-level wardrobe, exaggerated fabric movement, strong full-body or three-quarter pose, elongated body line, expressive hands and face, wind or motion in hair/fabric, glamorous accessories, bold makeup, epic location styling, low or cinematic camera angle, dramatic natural or studio lighting, and a clear fashion-story payoff. The prompt must describe the actual fashion shoot details, not just name the style." },
  { value: "creative_portrait_photography", label: "Creative portrait photography", description: "Creative portrait photography with a posed subject, strong visual theme, props or animals when appropriate, colorful art direction, expressive styling, and a memorable environment.", prompt_guidance: "Build a creative portrait photograph with a strong visual theme. Include a posed subject, purposeful prop or themed object if appropriate, color-directed wardrobe, expressive hair/makeup, layered environment details, playful or artistic composition, lens/framing, lighting style, and a memorable subject-environment relationship. If an animal or prop is used, make it clearly integrated into the scene concept." },
  { value: "gritty_analog", label: "Gritty analog", description: "Gritty analog film look, visible texture, natural imperfections, moody documentary realism.", prompt_guidance: "Build a gritty analog film still with imperfect realism: visible film grain, practical lighting, worn textures, imperfect surfaces, muted color response, handheld or documentary-feeling framing, natural body posture, atmospheric shadows, and a lived-in environment. Avoid overly polished studio language." },
  { value: "soft_dream_pop", label: "Soft dream pop", description: "Soft dreamy pop aesthetic, gentle bloom, pastel color, romantic haze, delicate cinematic lighting.", prompt_guidance: "Build a soft dream-pop still with gentle bloom, pastel color palette, romantic haze, delicate backlight, floating or soft fabric details, dreamy hair/makeup styling, graceful pose, shallow depth of field, soft environment edges, and a light emotional music-video mood." },
  { value: "high_contrast_drama", label: "High-contrast drama", description: "Bold shadows, sculpted highlights, intense facial emotion, dramatic production-still lighting.", prompt_guidance: "Build a high-contrast dramatic still with sculpted highlights, deep shadows, strong key light direction, visible tension in posture, intense facial emotion, dramatic wardrobe silhouette, textured environment, cinematic contrast ratio, and a composition that creates visual pressure." },
  { value: "surreal_symbolic", label: "Surreal symbolic", description: "Surreal symbolic music-video still, heightened atmosphere, poetic objects, dreamlike composition.", prompt_guidance: "Build a surreal symbolic music-video still. Use poetic visual motifs, dreamlike composition, unusual scale or placement of objects, symbolic set dressing, atmospheric light, controlled color palette, and a subject pose that feels ritualistic or uncanny. Keep the imagery visual and concrete rather than explanatory." },
  { value: "clean_studio", label: "Clean studio", description: "Clean studio photography, crisp subject detail, controlled lighting, uncluttered composition.", prompt_guidance: "Build a clean studio photograph with crisp subject detail, controlled lighting setup, precise wardrobe styling, polished hair/makeup, uncluttered backdrop, intentional pose, clear silhouette, lens/framing detail, and professional commercial or editorial clarity." },
  { value: "film_default", label: "Default film still", description: "Balanced short-film still lighting, believable production design, natural texture, and cinematic composition.", prompt_guidance: "Build a polished film-style music-video still. Use believable character blocking, grounded wardrobe, practical lighting, lens/framing detail, textured production design, natural color contrast, emotionally readable composition, and a cinematic story-frame finish." },
  { value: "indie_film_naturalism", label: "Indie film naturalism", description: "Naturalistic indie-drama still with lived-in details, imperfect realism, and intimate character focus.", prompt_guidance: "Build an indie-film music-video still with naturalistic lighting, lived-in wardrobe, imperfect textures, believable posture, intimate framing, subtle emotional detail, muted color response, and environment details that feel observed rather than staged." },
  { value: "neo_noir_dialogue", label: "Neo-noir dialogue", description: "Low-key shadows, practical neon, suspicious glances, dramatic contrast, and noir-style tension.", prompt_guidance: "Build a neo-noir dialogue still with low-key lighting, practical neon or sodium light, deep shadows, hard rim light, reflective surfaces, guarded facial expression, tense blocking, and a controlled color palette. Keep it cinematic and grounded." },
  { value: "gritty_punk_bar", label: "Gritty punk bar", description: "Worn bar textures, punk attitude, practical stage/neon light, smoky atmosphere, and analog grit.", prompt_guidance: "Build a gritty punk-bar film still with worn leather or denim styling, messy lived-in hair/makeup, scratched tables, stickers, posters, dim practical lights, colored neon spill, smoky air, visible texture, defiant posture, and a raw 35mm cinematic finish." },
  { value: "psychological_thriller", label: "Psychological thriller", description: "Uneasy framing, controlled color, negative space, tense facial detail, and subtle dread.", prompt_guidance: "Build a psychological-thriller still with uneasy composition, negative space, controlled color palette, tense facial detail, practical low light, slightly off-balance framing, foreground obstruction, and environmental details that imply pressure without explaining it." },
  { value: "warm_dialogue_drama", label: "Warm dialogue drama", description: "Warm practical interiors, soft skin tones, intimate framing, and emotionally readable acting.", prompt_guidance: "Build a warm dialogue-drama still with practical lamp, street, stage, or bar light, gentle skin tones, shallow depth of field, intimate framing, small emotional facial detail, believable wardrobe, textured surroundings, and a quiet cinematic finish." },
  { value: "35mm_analog_film", label: "35mm analog film", description: "Film grain, practical lighting, imperfect texture, grounded color, and documentary-like realism.", prompt_guidance: "Build a 35mm analog film still with visible grain, practical lighting, imperfect surfaces, grounded color response, natural posture, textured wardrobe, shallow lens character, and a lived-in environment. Avoid glossy music-video polish unless the scene asks for it." },
];

export const ID_LORA_IMAGE_AESTHETIC_PRESETS = [
  { value: "film_default", label: "Default film still", description: "Balanced short-film still lighting, believable production design, natural texture, and cinematic composition.", prompt_guidance: "Build a polished short-film still, not a music-video still. Use believable character blocking, grounded wardrobe, practical lighting, lens/framing detail, textured production design, natural color contrast, and emotionally readable composition." },
  { value: "indie_film_naturalism", label: "Indie film naturalism", description: "Naturalistic indie-drama still with lived-in details, imperfect realism, and intimate character focus.", prompt_guidance: "Build an indie-film still with naturalistic lighting, lived-in wardrobe, imperfect textures, believable posture, intimate framing, subtle emotional detail, muted color response, and environment details that feel observed rather than staged." },
  { value: "neo_noir_dialogue", label: "Neo-noir dialogue", description: "Low-key shadows, practical neon, suspicious glances, dramatic contrast, and noir-style tension.", prompt_guidance: "Build a neo-noir dialogue still with low-key lighting, practical neon or sodium light, deep shadows, hard rim light, reflective surfaces, guarded facial expression, tense blocking, and a controlled color palette. Keep it cinematic and grounded." },
  { value: "gritty_punk_bar", label: "Gritty punk bar", description: "Worn bar textures, punk attitude, practical stage/neon light, smoky atmosphere, and analog grit.", prompt_guidance: "Build a gritty punk-bar film still with worn leather or denim styling, messy lived-in hair/makeup, scratched tables, stickers, posters, dim practical lights, colored neon spill, smoky air, visible texture, defiant posture, and a raw 35mm cinematic finish." },
  { value: "psychological_thriller", label: "Psychological thriller", description: "Uneasy framing, controlled color, negative space, tense facial detail, and subtle dread.", prompt_guidance: "Build a psychological-thriller still with uneasy composition, negative space, controlled color palette, tense facial detail, practical low light, slightly off-balance framing, foreground obstruction, and environmental details that imply pressure without explaining it." },
  { value: "warm_dialogue_drama", label: "Warm dialogue drama", description: "Warm practical interiors, soft skin tones, intimate framing, and emotionally readable acting.", prompt_guidance: "Build a warm dialogue-drama still with practical lamp or bar light, gentle skin tones, shallow depth of field, intimate framing, small emotional facial detail, believable wardrobe, textured surroundings, and a quiet cinematic finish." },
  { value: "35mm_analog_film", label: "35mm analog film", description: "Film grain, practical lighting, imperfect texture, grounded color, and documentary-like realism.", prompt_guidance: "Build a 35mm analog film still with visible grain, practical lighting, imperfect surfaces, grounded color response, natural posture, textured wardrobe, shallow lens character, and a lived-in environment. Avoid glossy music-video polish." },
];

export function storyboardImageShotFlowEntry(profileKey, sceneIndex) {
  const preset = STORYBOARD_IMAGE_SHOT_FLOW_PRESETS[profileKey] || STORYBOARD_IMAGE_SHOT_FLOW_PRESETS.intimate;
  const sequence = preset.sequence || [];
  if (!sequence.length) return "";
  return sequence[sceneIndex % sequence.length] || "";
}

export function storyboardImageAestheticPreset(value = "") {
  return STORYBOARD_IMAGE_AESTHETIC_PRESETS.find((item) => item.value === value) || STORYBOARD_IMAGE_AESTHETIC_PRESETS[0];
}

function storyboardImageAestheticGuidance(value = "", options = {}) {
  const presets = options.idLoraMode ? ID_LORA_IMAGE_AESTHETIC_PRESETS : STORYBOARD_IMAGE_AESTHETIC_PRESETS;
  const preset = presets.find((item) => item.value === value) || presets[0] || storyboardImageAestheticPreset(value);
  return preset.prompt_guidance || preset.description || "";
}

function referenceChipHtml(ref, fallbackLabel = "Reference") {
  const image = storyboardReferenceImageSrc(ref?.image);
  const label = String(ref?.name || fallbackLabel || "Reference").trim();
  const thumb = image
    ? `<span style="width:34px;height:34px;border-radius:6px;border:1px solid #334155;background:#0f172a url('${escapeHtml(image)}') center/cover no-repeat;flex:0 0 auto;"></span>`
    : `<span style="width:34px;height:34px;border-radius:6px;border:1px dashed #334155;background:#07111f;color:#67e8f9;display:grid;place-items:center;font-size:12px;flex:0 0 auto;">▣</span>`;
  return `<span title="${escapeHtml(label)}" style="display:inline-flex;align-items:center;gap:7px;max-width:190px;border:1px solid #334155;border-radius:7px;background:#0f172a;color:#e5e7eb;padding:4px 7px;margin:3px 3px 3px 0;vertical-align:middle;">${thumb}<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:800;">${escapeHtml(label)}</span></span>`;
}

function subjectRefsHtml(scene) {
  const refs = Array.isArray(scene.subject_refs) ? scene.subject_refs : [];
  if (refs.length) return refs.map((ref, index) => referenceChipHtml(ref, `Subject ${index + 1}`)).join("");
  return tagsHtml(scene.subjects);
}

function settingRefHtml(scene) {
  if (scene.location_ref && typeof scene.location_ref === "object" && String(scene.location_ref.name || scene.location_ref.image?.path || scene.location_ref.image?.data || "").trim()) {
    return referenceChipHtml(scene.location_ref, scene.setting || "Location");
  }
  return escapeHtml(scene.setting || "-");
}

function normalizeReferenceBuilderCatalog(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const mergeReferenceList = (items = []) => {
    const byKey = new Map();
    const keyFor = (item) => {
      const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
      return name || String(item.id || "").trim().toLowerCase();
    };
    for (const item of items) {
      const key = keyFor(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        id: existing.id || item.id,
        name: existing.name || item.name,
        description: existing.description || item.description,
        trigger_phrase: existing.trigger_phrase || item.trigger_phrase,
        image: mergeReferenceImages(existing.image, item.image),
      });
    }
    return Array.from(byKey.values());
  };
  const subjects = mergeReferenceList(Array.isArray(source.subjects) ? source.subjects
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `subject_${index + 1}`),
      name: String(item.name || `Character ${index + 1}`),
      description: String(item.description || ""),
      trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
      trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
      extra_reference_for: String(item.extra_reference_for || item.extraReferenceFor || item.same_subject_as || item.sameSubjectAs || ""),
      image: normalizeReferenceImage(item),
    })).filter((item) => !item.extra_reference_for) : []);
  const locations = mergeReferenceList(Array.isArray(source.locations) ? source.locations
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `location_${index + 1}`),
      name: String(item.name || `Location ${index + 1}`),
      description: String(item.description || ""),
      trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
      trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
      image: normalizeReferenceImage(item),
    })) : []);
  return {
    subjects,
    locations: Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations) ? [] : locations,
    locations_cleared: Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations),
    trigger_position: String(source.trigger_position || source.triggerPosition || source.trigger_placement || "start") === "end" ? "end" : "start",
    subject_trigger_position: String(source.subject_trigger_position || source.subjectTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start",
    location_trigger_position: String(source.location_trigger_position || source.locationTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start",
  };
}

function mergeReferenceBuilderCatalog(base = {}, incoming = {}) {
  const normalizedBase = normalizeReferenceBuilderCatalog(base);
  const normalizedIncoming = normalizeReferenceBuilderCatalog(incoming);
  const mergeList = (left, right) => {
    const byKey = new Map();
    const keyFor = (item) => {
      const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
      return name || String(item.id || "").trim().toLowerCase();
    };
    for (const item of left) {
      const key = keyFor(item);
      if (key) byKey.set(key, { ...item, image: { ...(item.image || {}) } });
    }
    for (const item of right) {
      const key = keyFor(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        image: mergeReferenceImages(existing.image, item.image),
      });
    }
    return Array.from(byKey.values());
  };
  return {
    subjects: mergeList(normalizedBase.subjects, normalizedIncoming.subjects),
    locations: normalizedBase.locations_cleared ? [] : mergeList(normalizedBase.locations, normalizedIncoming.locations),
    locations_cleared: Boolean(normalizedBase.locations_cleared || normalizedIncoming.locations_cleared),
  };
}

function statusMeta(scene) {
  const hasImage = Boolean(String(scene.image_path || "").trim());
  const hasImagePrompt = Boolean(String(scene.image_prompt || "").trim());
  const hasVideoPrompt = Boolean(String(scene.video_prompt || "").trim());
  if (hasImage && hasVideoPrompt) return { label: "Ready for Video", color: "#22c55e" };
  if (hasImagePrompt && hasVideoPrompt) return { label: "Prompts Ready", color: "#22c55e" };
  if (hasVideoPrompt) return { label: "Video Prompt Ready", color: "#22c55e" };
  if (hasImagePrompt) return { label: "Image Prompt Ready", color: "#22c55e" };
  if (hasImage) return { label: "Image Ready", color: "#10b981" };
  return { label: "Draft", color: "#60a5fa" };
}

function storyboardIsInstrumentalText(value = "") {
  const text = String(value || "").trim();
  if (!text) return false;
  if (/^\[?\s*instrumental\s*\]?\.?$/i.test(text)) return true;
  if (/^\[?\s*(?:no vocals?|no singing|silence|music|intro|outro|interlude|break)\s*\]?\.?$/i.test(text)) return true;
  return /\binstrumental|no vocals?|no singing|silence\b/i.test(text);
}

function normalizeStoryboardPerformanceMode(value = "") {
  const text = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["speaking", "short_film", "dialogue", "dialog"].includes(text)) return "speaking";
  if (["no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"].includes(text)) return "no_lip_sync";
  return "singing";
}

function storyboardStillFacialDirection(value = "") {
  return String(value || "")
    .replace(/\bsubtle natural eye movement\b/gi, "clear eye direction")
    .replace(/\bsubtle eye movement\b/gi, "clear eye direction")
    .replace(/\boccasional natural blinking\b/gi, "natural eyelid detail")
    .replace(/\bnatural blinking\b/gi, "natural eyelid detail")
    .replace(/\bfast-moving mouth during delivery\b/gi, "mouth captured in a still expressive shape")
    .replace(/\bmouth open mid-verse\b/gi, "mouth captured in a still expressive shape")
    .replace(/\blips slightly parted while singing\b/gi, "lips slightly parted in a still performance expression")
    .replace(/\bsnarling mouth shapes during vocals\b/gi, "snarling still mouth expression")
    .replace(/\bbared teeth on powerful notes\b/gi, "bared teeth in a powerful still expression")
    .replace(/\braw emotional scream expression\b/gi, "raw emotional still expression")
    .replace(/\bforceful singing expression\b/gi, "forceful performance expression")
    .replace(/\bmovement\b/gi, "pose")
    .replace(/\bmoving\b/gi, "posed")
    .replace(/\bduring vocals?\b/gi, "in the expression")
    .replace(/\bwhile singing\b/gi, "in the expression")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function normalizeScene(scene = {}, index = 0) {
  const rawVideoType = String(scene.video_prompt_type || scene.video_type || scene.mode || "").trim();
  const videoPromptType = ["i2v", "id_lora", "t2v", "rtv", "ingredients"].includes(rawVideoType) ? rawVideoType : "i2v";
  const lyrics = scene.lyrics || scene.lyric_text || "";
  const lyricSingers = Array.isArray(scene.lyric_singers)
    ? scene.lyric_singers.map((item) => String(item || "").trim()).filter(Boolean)
    : String(scene.lyric_singers || scene.singers || "").split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
  const lyricNoLipSync = Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync || scene.noLipSync || scene.broll || scene.b_roll);
  const lyricInstrumental = Boolean(scene.lyric_instrumental || scene.instrumental || storyboardIsInstrumentalText(lyrics));
  const noCharacterPresent = Boolean(scene.no_character_present || scene.noCharacterPresent || scene.no_subject || scene.no_visible_subject);
  return {
    id: scene.id || `storyboard_scene_${index + 1}_${Date.now()}`,
    scene_number: Number(scene.scene_number || scene.number || index + 1),
    label: scene.label || `Scene ${index + 1}`,
    lyrics,
    lyric_section: scene.lyric_section || scene.section || scene.song_section || "",
    story_beat: scene.story_beat || scene.scene_story_beat || scene.narrative_beat || "",
    performance_mode: normalizeStoryboardPerformanceMode(scene.performance_mode || scene.performanceMode || scene.video_performance_mode || scene.videoPerformanceMode),
    lyric_singers: lyricSingers,
    lyric_no_lip_sync: lyricNoLipSync,
    lyric_instrumental: lyricInstrumental,
    no_character_present: noCharacterPresent,
    prompt_summary: scene.prompt_summary || scene.summary || "",
    motion_summary: scene.motion_summary || scene.video_notes || scene.i2v_notes || "",
    subjects: Array.isArray(scene.subjects) ? scene.subjects : String(scene.subjects || "").split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean),
    subject_refs: noCharacterPresent ? [] : Array.isArray(scene.subject_refs) ? scene.subject_refs.filter((item) => item && typeof item === "object") : [],
    setting: scene.setting || scene.location_ref?.description || scene.location_ref?.name || scene.location || "",
    location_ref: scene.location_ref && typeof scene.location_ref === "object" ? scene.location_ref : null,
    trigger_phrase: String(scene.trigger_phrase || scene.trigger || scene.Trigger || ""),
    trigger_position: String(scene.trigger_position || scene.triggerPosition || scene.trigger_placement || "start") === "end" ? "end" : "start",
    video_prompt_type: videoPromptType,
    shot_type: scene.shot_type || "",
    camera_motion: scene.camera_motion || scene.motion_preset || "",
    character_motion: scene.character_motion || scene.character_motion_preset || scene.subject_motion || "",
    performance_style: scene.performance_style || scene.song_style || scene.music_style || "",
    facial_performance: scene.facial_performance || scene.facialPerformance || scene.facial_expression || scene.facialExpression || "",
    facial_performance_custom: scene.facial_performance_custom || scene.facialPerformanceCustom || scene.facial_expression_custom || scene.facialExpressionCustom || "",
    include_microphone: Boolean(scene.include_microphone || scene.use_microphone || scene.microphone),
    status: scene.status || "draft",
    image_prompt: scene.image_prompt || scene.t2i_prompt || "",
    video_prompt: scene.video_prompt || scene.i2v_prompt || scene.t2v_prompt || "",
    image_path: scene.image_path || scene.approved_image_path || "",
    image_data: scene.image_data || scene.image_reference_data || "",
    notes: scene.notes || "",
    id_lora_character_id: scene.id_lora_character_id || scene.character_id || scene.subject_id || "",
    id_lora_location_id: scene.id_lora_location_id || scene.location_id || "",
  };
}

function storyboardReferenceOpening(scene = {}) {
  const normalized = normalizeScene(scene, 0);
  const subjectCount = normalized.no_character_present
    ? 0
    : normalized.subject_refs.filter((subject) => {
        const image = subject?.image || subject || {};
        return Boolean(image.path || image.data || subject?.image_path || subject?.image_data);
      }).length;
  const locationImage = normalized.location_ref?.image || normalized.location_ref || {};
  const hasLocation = Boolean(locationImage.path || locationImage.data || normalized.location_ref?.image_path || normalized.location_ref?.image_data);
  if (!subjectCount && !hasLocation) return "";
  const characterPhrase = subjectCount > 1 ? "character reference images" : "character reference image";
  if (subjectCount && hasLocation) return `Using the provided ${characterPhrase} and location reference image`;
  if (subjectCount) return `Using the provided ${characterPhrase}`;
  return "Using the provided location reference image";
}

function storyboardImageModeUsesReferenceOpening(imageMode = "") {
  return ["nano_banana", "flux_klein", "flow_gpt"].includes(String(imageMode || "").trim());
}

function ensureStoryboardReferenceOpening(prompt, scene = {}, imageMode = "") {
  if (!storyboardImageModeUsesReferenceOpening(imageMode)) return String(prompt || "").trim();
  const opening = storyboardReferenceOpening(scene);
  let text = String(prompt || "").trim();
  if (!opening || !text) return text;
  text = text.replace(
    /^Using the provided\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?)(?:\s+and\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?))*\s*,?\s*(?:create\s+)?/i,
    "",
  ).trim();
  text = text.replace(
    /^and\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?)\s*,?\s*(?:create\s+)?/i,
    "",
  ).trim();
  text = text.replace(/^(?:create|make|generate)\b\s*/i, "").trim();
  if (!text) return `${opening}, create a cinematic still image.`;
  return `${opening}, create ${text.slice(0, 1).toLowerCase()}${text.slice(1)}`;
}

function scenesFromBuilderPayload(payload = {}) {
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  return scenes.map((scene, index) => normalizeScene({
    id: scene.id,
    scene_number: index + 1,
    label: scene.label || `Scene ${index + 1}`,
    lyrics: scene.lyric_text || scene.lyrics || "",
    lyric_section: scene.lyric_section || scene.section || scene.song_section || "",
    story_beat: scene.story_beat || scene.scene_story_beat || scene.narrative_beat || "",
    performance_mode: scene.performance_mode || scene.performanceMode || payload.performance_mode || payload.performanceMode || "",
    lyric_singers: scene.lyric_singers || scene.singers || [],
    lyric_no_lip_sync: Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync),
    lyric_instrumental: Boolean(scene.lyric_instrumental || scene.instrumental),
    no_character_present: Boolean(scene.no_character_present || scene.noCharacterPresent || scene.no_subject || scene.no_visible_subject),
    prompt_summary: scene.notes || scene.director_note || scene.t2i_prompt || "",
    motion_summary: scene.video_notes || scene.i2v_notes || "",
    subjects: scene.lyric_singers || scene.subjects || "",
    subject_refs: scene.subject_refs || [],
    setting: scene.location || scene.location_ref?.description || scene.location_ref?.name || "",
    location_ref: scene.location_ref || null,
    video_prompt_type: scene.video_prompt_type || scene.video_type || "",
      shot_type: scene.shot_type || "",
      camera_motion: scene.camera_motion || scene.motion_preset || "",
      character_motion: scene.character_motion || scene.character_motion_preset || scene.subject_motion || "",
      performance_style: scene.performance_style || scene.song_style || scene.music_style || "",
      facial_performance: scene.facial_performance || scene.facialPerformance || scene.facial_expression || scene.facialExpression || "",
      facial_performance_custom: scene.facial_performance_custom || scene.facialPerformanceCustom || scene.facial_expression_custom || scene.facialExpressionCustom || "",
      include_microphone: Boolean(scene.include_microphone || scene.use_microphone || scene.microphone),
      image_prompt: scene.t2i_prompt || "",
    video_prompt: scene.i2v_prompt || scene.t2v_prompt || "",
    image_path: scene.image_path || scene.approved_image_path || "",
    image_data: scene.image_data || scene.image_reference_data || "",
    notes: scene.notes || "",
  }, index));
}

function createToast(message, error = false) {
  const toast = document.createElement("div");
  toast.textContent = message;
  toast.style.cssText = `position:fixed;right:24px;bottom:24px;z-index:100020;max-width:520px;border:1px solid ${error ? "#991b1b" : "#155e75"};border-radius:8px;background:${error ? "#3f0808" : "#083344"};color:#f8fafc;padding:12px 14px;box-shadow:0 12px 40px rgba(0,0,0,.45);white-space:pre-wrap;font-size:13px;`;
  document.body.append(toast);
  setTimeout(() => toast.remove(), error ? 8500 : 4200);
}

function createStoryboardProgressWindow(title = "Storyboard LLM") {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.18);pointer-events:none;display:flex;align-items:flex-start;justify-content:center;padding-top:72px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(760px,calc(100vw - 48px));border:1px solid #0891b2;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 22px 70px rgba(0,0,0,.55);overflow:hidden;pointer-events:auto;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;background:#083f4f;border-bottom:1px solid #0891b2;";
  const titleEl = document.createElement("div");
  titleEl.textContent = title;
  titleEl.style.cssText = "font-weight:900;color:#cffafe;";
  const close = makeButton("Close");
  close.style.padding = "8px 12px";
  header.append(titleEl, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;";
  const message = document.createElement("div");
  message.style.cssText = "white-space:pre-wrap;line-height:1.45;font-size:13px;color:#e2e8f0;min-height:38px;";
  const track = document.createElement("div");
  track.style.cssText = "height:8px;border-radius:999px;background:#155e75;overflow:hidden;";
  const fill = document.createElement("div");
  fill.style.cssText = "height:100%;width:0%;background:#22d3ee;border-radius:999px;transition:width .18s ease;";
  track.append(fill);
  body.append(message, track);
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  close.onclick = () => backdrop.remove();
  return {
    set(text, percent = 0) {
      message.textContent = String(text || "");
      const pct = Number.isFinite(Number(percent)) ? Math.max(0, Math.min(100, Number(percent))) : 0;
      fill.style.width = `${pct}%`;
    },
    close(delay = 0) {
      if (delay > 0) {
        setTimeout(() => backdrop.remove(), delay);
      } else {
        backdrop.remove();
      }
    },
  };
}

function storyboardPayloadFromBuilder(payload = {}) {
  return {
    project_folder: payload.projectFolder || payload.project_folder || "",
    scenes: scenesFromBuilderPayload(payload),
  };
}

function slimReferenceForRequest(ref) {
  if (!ref || typeof ref !== "object") return null;
  return {
    id: String(ref.id || ""),
    name: String(ref.name || ""),
    description: String(ref.description || ""),
    trigger_phrase: String(ref.trigger_phrase || ref.trigger || ref.Trigger || ""),
    trigger_position: String(ref.trigger_position || ref.triggerPosition || ref.trigger_placement || "start") === "end" ? "end" : "start",
    image: {
      path: String(ref.image?.path || ""),
      name: String(ref.image?.name || ""),
      data: "",
    },
  };
}

function slimSceneForRequest(scene, index = 0) {
  const normalized = normalizeScene(scene, index);
  return {
    ...normalized,
    subject_refs: (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map(slimReferenceForRequest)
      .filter(Boolean),
    location_ref: slimReferenceForRequest(normalized.location_ref),
  };
}

function normalizeStoryLayer(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const lyricStoryStrength = Math.max(0, Math.min(10, Number(source.lyric_story_strength ?? source.lyricStoryStrength ?? 7)));
  return {
    enabled: source.enabled !== false,
    user_story_arc: String(source.user_story_arc || source.userStoryArc || ""),
    song_story_brief: String(source.song_story_brief || source.songStoryBrief || ""),
    lyric_story_strength: Number.isFinite(lyricStoryStrength) ? lyricStoryStrength : 7,
  };
}

function storyboardSpeedValue(value, fallback = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(10, number)) : fallback;
}

function storyboardSpeedLabel(value, kind = "motion") {
  const speed = storyboardSpeedValue(value);
  if (speed <= 0) return kind === "camera" ? "0 / static camera" : "0 / still subject";
  if (speed <= 3) return `${speed} / subtle`;
  if (speed <= 6) return `${speed} / active`;
  if (speed <= 8) return `${speed} / energetic`;
  return `${speed} / fast action`;
}

function storyboardSpeedGuidance(value, kind = "motion") {
  const speed = storyboardSpeedValue(value);
  if (kind === "camera") {
    if (speed <= 0) return "Camera speed 0/10: locked-off static camera, no camera movement.";
    if (speed <= 3) return `Camera speed ${speed}/10: slow, gentle camera motion; one simple move at most.`;
    if (speed <= 6) return `Camera speed ${speed}/10: controlled cinematic movement such as tracking, pan, dolly, crane, or orbit, usually one clear move.`;
    if (speed <= 8) return `Camera speed ${speed}/10: energetic camera motion with stronger tracking, orbit, whip pan, rise, reveal, or compound movement.`;
    return `Camera speed ${speed}/10: fast action camera language; use two or more coordinated camera actions in one scene when readable, such as whip pan into fast tracking plus orbit, reveal, pan, tilt, crane, or pullback. Do not end with "then holds", "holds on", "settles into a hold", static hold, or steady hold unless the user explicitly asks for a hold.`;
  }
  if (speed <= 0) return "Character motion speed 0/10: subject stays still or holds a pose; only facial expression or tiny gestures.";
  if (speed <= 3) return `Character motion speed ${speed}/10: subtle body motion such as shifting weight, hand gestures, turning, swaying, reaching, or small steps.`;
  if (speed <= 6) return `Character motion speed ${speed}/10: active body performance; walking, dancing, interacting with objects, using the set, expressive arms and torso.`;
  if (speed <= 8) return `Character motion speed ${speed}/10: energetic character action; running, dancing hard, climbing, struggling, spinning, crossing the space, or forceful environmental interaction.`;
  return `Character motion speed ${speed}/10: fast action character movement; require clear full-body action such as sprinting, explosive dance, striding, sharp turns, crossing the space, chase/action beats, rapid direction changes, forceful gestures, or intense physical set interaction when it fits the scene. Avoid only poised, still, standing, subtle, quiet, steady, or restrained body language.`;
}

function enforceHighMotionPromptLanguage(prompt, scene = {}, state = {}) {
  let text = String(prompt || "").trim();
  if (!text) return text;
  const cameraSpeed = storyboardSpeedValue(scene.camera_motion_speed ?? scene.cameraMotionSpeed ?? state.cameraMotionSpeed, 4);
  const characterSpeed = storyboardSpeedValue(scene.character_motion_speed ?? scene.characterMotionSpeed ?? state.characterMotionSpeed, 4);
  if (cameraSpeed >= 9) {
    text = text
      .replace(/\bthen\s+holds?\s+on\b/gi, "then continues moving across")
      .replace(/\bthen\s+holds?\b/gi, "then continues moving")
      .replace(/\bsettles?\s+into\s+a\s+(?:static\s+|steady\s+)?hold\b/gi, "flows into another coordinated camera move")
      .replace(/\b(?:static|steady)\s+hold\b/gi, "continued camera motion")
      .replace(/\bholds?\s+on\s+her\s+steady,\s*powerful\s+gaze\b/gi, "tracks her powerful gaze while the camera keeps moving")
      .replace(/\bholds?\s+on\s+(his|her|their|the)\s+([^,.]+)\b/gi, "keeps moving around $1 $2");
    if (!/\b(?:tracking|orbit|whip pan|pan|tilt|crane|pullback|push|dolly|handheld|reveal)\b.*\b(?:tracking|orbit|whip pan|pan|tilt|crane|pullback|push|dolly|handheld|reveal)\b/i.test(text)) {
      text = text.replace(/\.+\s*$/, "");
      text += ", with the camera chaining multiple readable moves instead of stopping on a hold.";
    }
  }
  if (characterSpeed >= 9) {
    text = text
      .replace(/\bmoves?\s+with\s+a\s+quiet,\s*poised\s+authority\b/gi, "moves with forceful, physically active authority")
      .replace(/\bmoves?\s+with\s+quiet,\s*poised\s+authority\b/gi, "moves with forceful, physically active authority")
      .replace(/\bquiet,\s*poised\s+authority\b/gi, "forceful, physically active authority")
      .replace(/\bquiet\s+poised\s+authority\b/gi, "forceful physical authority")
      .replace(/\bpoised,\s*unyielding\s+head\s+position\b/gi, "forward-driving head posture with sharp turns")
      .replace(/\bpoised\s+posture\b/gi, "active, commanding posture")
      .replace(/\bsubtle\s+body\s+motion\b/gi, "clear full-body movement")
      .replace(/\bstands?\s+still\b/gi, "moves through the space");
    if (!/\b(?:strides?|runs?|sprints?|dances?|turns?|crosses?|lunges?|reaches?|pushes?|pulls?|climbs?|fights?|brushing|sweeping|gestures?)\b/i.test(text)) {
      text = text.replace(/\.+\s*$/, "");
      text += ", while the subject performs clear full-body movement through the set.";
    }
  }
  return text.replace(/\s{2,}/g, " ").trim();
}

function mergeStoryLayers(primary = {}, fallback = {}) {
  const primaryLayer = normalizeStoryLayer(primary);
  const fallbackLayer = normalizeStoryLayer(fallback);
  return normalizeStoryLayer({
    enabled: primaryLayer.enabled !== false,
    user_story_arc: primaryLayer.user_story_arc || fallbackLayer.user_story_arc,
    song_story_brief: primaryLayer.song_story_brief || fallbackLayer.song_story_brief,
  });
}

function slimStoryboardForRequest(state) {
  return {
    mode: state.mode,
    performance_mode: normalizeStoryboardPerformanceMode(state.performanceMode || state.performance_mode),
    camera_flow: state.cameraFlow || "balanced",
    image_shot_flow: state.imageShotFlow || "intimate",
    image_aesthetic: state.imageAesthetic || "",
    global_consistency_phrase: state.globalConsistencyPhrase || "",
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    performance_style_default: state.performanceStyle || "",
    facial_performance_default: state.facialPerformance || "",
    facial_performance_custom_default: state.facialPerformanceCustom || "",
    story_layer: normalizeStoryLayer(state.storyLayer),
    reference_builder: {
      subjects: (state.referenceBuilder?.subjects || []).map(slimReferenceForRequest).filter(Boolean),
      locations: (state.referenceBuilder?.locations || []).map(slimReferenceForRequest).filter(Boolean),
    },
    motion_defaults: {
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
    },
    scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
  };
}

const STORYBOARD_GPT_URL = "https://chatgpt.com/g/g-6a28d15f04e88191a2375d564ff8d90c-ltx-2-3-video-builder-from-storyboard-builder";
const STORYBOARD_IMAGE_GPT_URL = "https://chatgpt.com/g/g-6a40129fc12c81919878b79eaa5ae94f-text-to-image-prompt-builder-for-krea-2";

function storyboardReferenceForGpt(ref, options = {}) {
  if (!ref) return null;
  const name = String(ref.name || "").trim();
  const description = String(ref.description || "").trim();
  const triggerPhrase = String(ref.trigger_phrase || ref.trigger || ref.Trigger || "").trim();
  const promptName = options.subject && triggerPhrase ? triggerPhrase : name;
  if (!promptName && !description) return null;
  return {
    name: promptName,
    display_name: name,
    description,
    trigger_phrase: triggerPhrase,
    prompt_name_source: options.subject && triggerPhrase ? "subject_trigger_phrase" : "reference_name",
  };
}

function storyboardVideoPromptTypeLabel(type) {
  const key = String(type || "").toLowerCase();
  if (key === "id_lora") return "ID-LoRA image to video";
  if (key === "ingredients") return "ingredients to video";
  if (key === "t2v") return "text to video";
  if (key === "rtv") return "reference to video";
  if (key === "i2v") return "image to video";
  return key || "image to video";
}

function storyboardScenesForGpt(state) {
  const imageMode = state.mode !== "image_to_video_prep";
  const idLoraMode = String(state.videoPromptType || state.video_prompt_type || "").trim() === "id_lora"
    || state.scenes.some((scene) => String(scene?.video_prompt_type || "").trim() === "id_lora");
  const performancePresets = idLoraMode ? ID_LORA_PERFORMANCE_STYLE_PRESETS : PERFORMANCE_STYLE_PRESETS;
  const facialPresets = idLoraMode ? ID_LORA_FACIAL_PERFORMANCE_PRESETS : FACIAL_PERFORMANCE_PRESETS;
  const performancePreset = (value = "") => performancePresets.find((item) => item.value === value) || performancePresets[0] || PERFORMANCE_STYLE_PRESETS[0];
  const facialPresetForPayload = (value = "") => facialPresets.find((item) => item.value === value) || facialPresets[0] || FACIAL_PERFORMANCE_PRESETS[0];
  let previousCameraMotion = "";
  return state.scenes.map((scene, index) => {
    const normalized = normalizeScene(scene, index);
    const sceneNumberIndex = Math.max(0, Number(normalized.scene_number || index + 1) - 1);
    const cameraFallback = storyboardCameraFlowEntry(state.cameraFlow || "balanced", sceneNumberIndex, previousCameraMotion);
    const shotType = normalized.shot_type || cameraFallback?.shot || "";
    const cameraMotion = normalized.camera_motion || (imageMode ? "" : cameraFallback?.camera) || "";
    if (!imageMode) previousCameraMotion = cameraMotion || previousCameraMotion;
    const lyricText = String(normalized.lyrics || "").trim();
    const performanceMode = normalizeStoryboardPerformanceMode(normalized.performance_mode || state.performanceMode || state.videoType || state.performance_mode);
    const facialPreset = facialPresetForPayload(normalized.facial_performance || state.facialPerformance);
    const facialCustom = String(normalized.facial_performance_custom || state.facialPerformanceCustom || "").trim();
    const facialDirection = (normalized.facial_performance || state.facialPerformance) === "custom" && facialCustom
      ? facialCustom
      : [facialPreset.direction, facialCustom].filter(Boolean).join(" ");
    const instrumental = Boolean(normalized.lyric_instrumental);
    const noLipSync = Boolean(normalized.lyric_no_lip_sync || performanceMode === "no_lip_sync");
    const noCharacterPresent = Boolean(normalized.no_character_present);
    const shouldLipSync = !imageMode && performanceMode !== "no_lip_sync" && Boolean(lyricText) && !instrumental && !noLipSync && !noCharacterPresent;
    const subjectRefs = noCharacterPresent ? [] : (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map((ref) => storyboardReferenceForGpt(ref, { subject: true }))
      .filter(Boolean);
    const subjectFallbacks = noCharacterPresent ? [] : (Array.isArray(normalized.subjects) ? normalized.subjects : [])
      .map((name) => ({ name: String(name || "").trim(), description: "" }))
      .filter((item) => item.name);
    const subjectNames = subjectRefs.length
      ? subjectRefs.map((subject) => subject.name).filter(Boolean)
      : subjectFallbacks.map((subject) => subject.name).filter(Boolean);
    const subjectCount = subjectRefs.length || subjectFallbacks.length;
    const subjectPromptNameByLabel = new Map(
      subjectRefs
        .map((subject) => [String(subject.display_name || subject.name || "").trim().toLowerCase(), subject.name])
        .filter(([label, promptName]) => label && promptName),
    );
    const explicitSingers = (Array.isArray(normalized.lyric_singers) ? normalized.lyric_singers : [])
      .map((name) => String(name || "").trim())
      .map((name) => subjectPromptNameByLabel.get(name.toLowerCase()) || name)
      .filter(Boolean);
    const singers = shouldLipSync ? (explicitSingers.length ? explicitSingers : subjectNames) : [];
    const singerKeySet = new Set(singers.map((name) => String(name || "").trim().toLowerCase()));
    const nonSingingSubjects = shouldLipSync
      ? subjectNames.filter((name) => !singerKeySet.has(String(name || "").trim().toLowerCase()))
      : subjectNames;
    const locationRef = storyboardReferenceForGpt(normalized.location_ref);
    return {
      scene_number: normalized.scene_number,
      label: normalized.label,
      prompt_type: imageMode ? "text to image" : storyboardVideoPromptTypeLabel(normalized.video_prompt_type),
      performance_mode: performanceMode,
      lyric_line_to_sing: shouldLipSync && performanceMode === "singing" ? lyricText : "",
      line_to_say: shouldLipSync && performanceMode === "speaking" ? lyricText : "",
      vocal_status: {
        performance_mode: performanceMode,
        lyric_text: lyricText,
        lyric_section: normalized.lyric_section,
        singers,
        instrumental,
        no_lip_sync: noLipSync,
        should_lip_sync: shouldLipSync,
        no_character_present: noCharacterPresent,
      },
      vocal_direction: {
        mode: imageMode
          ? "still image / no singing"
          : performanceMode === "speaking" && shouldLipSync
            ? "say exact dialogue line"
            : performanceMode === "no_lip_sync"
              ? "visual only / no lip sync"
              : (shouldLipSync ? "sing exact lyric line" : (instrumental ? "instrumental / no vocals" : (noLipSync ? "b-roll / no lip sync" : "no lyric line provided"))),
        lyric_line: lyricText,
        singers,
        non_singing_visible_subjects: nonSingingSubjects,
        instruction: imageMode
          ? "This is a text-to-image still prompt, not a video or lip-sync prompt. Use lyric_line only for mood, symbolism, emotion, and visual direction. Do not mention singing, lip-syncing, performing vocals, singing the line, mouth movement, blinking, eye movement, or animation. The subject can hold a natural still pose, show a clear expression, or appear in a fashion/editorial scene, but should not be described as singing unless the scene notes explicitly ask for a live singing still."
          : performanceMode === "speaking" && shouldLipSync
            ? "Treat lyric_line as dialogue being said. The listed singer(s) field means the visible speaker(s). Use only wording like 'as she says \"...\"', 'as he says \"...\"', or 'as [subject name] says \"...\"'. Do not use alternate verbs for the dialogue line; use says only. Never use singing, rapping, music, lyric, vocal, or performance wording for speaking mode. Every non_singing_visible_subjects entry must still appear in the scene as visible non-speaking subjects who react, watch, move, or share the moment silently. Do not describe mouth shapes or mouth position."
            : performanceMode === "no_lip_sync"
              ? "Visual-only scene. Do not quote lyric_line. Do not mention saying, speaking, dialogue, singing, rapping, lyrics, vocals, lip-syncing, mouth movement, or no-vocal status. Use lyric_line only as hidden mood or story context."
              : shouldLipSync
                ? "Treat lyric_line as words being sung, not as literal scene action. The listed singer(s) should visibly sing this line with expressive facial emotion, gestures, performance energy, and facial performance guidance when provided. In the singer face sentence, include subtle natural eye movement and occasional natural blinking beside the eyes/brows/gaze description; do not append blinking or eye movement to an environment sentence. Every non_singing_visible_subjects entry must still appear in the scene as a visible non-singing subject who reacts, watches, moves, or shares the moment without singing. Use mouth-shape or jaw/lip wording only for the listed singer(s), never for non-singing subjects."
                + " Do not describe visible singing as quiet; use controlled, focused, intimate, restrained, inward, tender, or simmering intensity instead."
                : "Do not mention singing, lip-syncing, mouth movement, or vocal performance for this scene. Every listed subject must still appear as a visible non-singing subject unless no_character_present is true.",
      },
      scene_summary: imageMode ? "" : normalized.prompt_summary,
      story_layer: {
        lyric_section: normalized.lyric_section,
        scene_story_beat: normalized.story_beat,
        song_story_brief: state.storyLayer?.enabled === false ? "" : String(state.storyLayer?.song_story_brief || ""),
        user_story_arc: state.storyLayer?.enabled === false ? "" : String(state.storyLayer?.user_story_arc || ""),
        lyric_story_strength: normalizeStoryLayer(state.storyLayer).lyric_story_strength,
        instruction: "Use the story brief and scene story beat as narrative guidance. Lyric story strength controls how literally to follow lyric_line: 0 ignores lyrics, 1-3 uses mood only, 4-6 balances lyrics with story, 7-8 strongly follows lyric meaning, and 9-10 uses concrete lyric objects/actions/emotions whenever possible. Do not turn the prompt into plot exposition.",
      },
      motion_summary: imageMode ? "" : normalized.motion_summary,
      still_image_notes: imageMode ? normalized.motion_summary : "",
      image_aesthetic: imageMode ? storyboardImageAestheticGuidance(state.imageAesthetic, { idLoraMode }) : "",
      image_aesthetic_instruction: imageMode
        ? "Translate the selected image aesthetic into concrete prompt details: pose, wardrobe styling, hair, makeup, accessories, lighting setup, lens/framing, composition, environment treatment, texture, weather/time if useful, and art direction. Do not merely name the preset or append it as a short tag."
        : "",
      global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
      global_consistency_instruction: String(state.globalConsistencyPhrase || "").trim()
        ? "Incorporate the global_consistency_phrase naturally into the prompt where it fits. Preserve its key wording, but do not force it to the beginning unless that is the most natural phrasing."
        : "",
      performance_style: performancePreset(normalized.performance_style || state.performanceStyle).label,
      performance_direction: performancePreset(normalized.performance_style || state.performanceStyle).direction,
      facial_performance: facialPreset.label,
      facial_performance_direction: imageMode ? storyboardStillFacialDirection(facialDirection) : facialDirection,
      facial_performance_custom: imageMode ? storyboardStillFacialDirection(facialCustom) : facialCustom,
      microphone: {
        include: Boolean(normalized.include_microphone),
        instruction: normalized.include_microphone
          ? "A microphone may be included if it naturally fits the scene, stage, or performance setup."
          : "Do not mention or add a microphone, mic stand, headset mic, studio mic, or any microphone prop unless the scene notes explicitly ask for one.",
      },
      subject_count: subjectCount,
      subject_instruction: noCharacterPresent
        ? (imageMode
          ? "No main character or mapped subject is present in this scene. Do not include, mention, imply, or describe the mapped character/singer/subject. Use the location, props, environment, objects, atmosphere, and still-image composition instead."
          : "No main character or mapped subject is present in this scene. Do not include, mention, imply, or describe the mapped character/singer/subject. Use the location, props, environment, objects, atmosphere, and camera motion instead.")
        : subjectCount === 1
        ? "This scene has exactly one mapped subject. Use the exact visible_subjects name/phrase as the subject phrase in the prompt. If that phrase came from a subject trigger_phrase, treat it as the subject identity, e.g. 'a photo of TRIGGER_PHRASE' instead of 'a photo of a woman'. Do not rewrite it as 'one woman', 'a woman', 'one man', 'a man', or any generic count phrase. Treat that exact subject phrase as one individual person and do not create duplicates, groups, backup singers, or multiple versions of the subject."
        : "This scene has multiple mapped subjects. Every listed subject must be visibly present in the prompt. Use each exact visible_subjects name/phrase when referring to them. If a phrase came from a subject trigger_phrase, treat it as that subject's identity. Do not drop any listed subject, rename them, or replace them with generic count phrases. Only the names in vocal_status.singers should sing; the other listed subjects should be visible but not singing. Do not add extra people unless the scene notes explicitly ask for them.",
      subject_name_rule: "Preserve mapped subject prompt names exactly as provided in visible_subjects and subjects.name. For subjects with trigger_phrase, subjects.name is already the prompt-facing trigger phrase and must be used as the subject instead of generic wording like 'a woman' or 'a man'.",
      visible_subjects: subjectNames,
      subjects: subjectRefs.length ? subjectRefs : subjectFallbacks,
      setting: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      location_ref: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      shot_type: shotType,
      camera_motion: imageMode ? "" : cameraMotion,
      still_camera_style: imageMode ? cameraMotion : "",
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      camera_motion_speed_guidance: imageMode ? "" : storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      camera_guidance: imageMode
        ? {
            selected_still_camera_style: cameraMotion,
            instruction: "Use this as still photography composition, lens, lighting, or framing guidance only. Do not turn it into camera movement.",
          }
        : {
            selected_camera_motion: cameraMotion,
            camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
            camera_motion_speed_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
            avoid_default_inward_moves: true,
            instruction: "Use the selected camera motion as written. Do not add zoom-in, push-in, dolly-in, crash-zoom, or a close-up ending unless that exact inward motion is selected or requested in notes.",
          },
      character_motion: imageMode ? "" : normalized.character_motion,
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      character_motion_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
      first_frame_visual_inventory: imageMode
        ? ""
        : {
            source: "text_to_image_prompt",
            text: normalized.image_prompt,
            instruction: "Use only for visible first-frame inventory: subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition. Do not use this field for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing.",
          },
      text_to_image_prompt: imageMode ? normalized.image_prompt : "",
      video_prompt: normalized.video_prompt,
      notes: normalized.notes,
    };
  });
}

export function storyboardGptPayload(state, scenesOverride = null) {
  const payloadState = scenesOverride ? { ...state, scenes: scenesOverride } : state;
  const selectedScene = scenesOverride?.length === 1 ? normalizeScene(scenesOverride[0], 0) : null;
  const imageMode = state.mode !== "image_to_video_prep";
  const selectedImageMode = String(state.imageMode || state.image_mode || "zimage").trim() || "zimage";
  const selectedImageModeLabel = String(state.imageModeLabel || state.image_mode_label || selectedImageMode).trim() || selectedImageMode;
  const imagePromptTarget = selectedImageMode === "flow_gpt"
    ? "Flow/GPT browser image prompt"
    : selectedImageMode === "nano_banana"
      ? "NanoBanana image prompt"
      : `${selectedImageModeLabel} image prompt`;
  return {
    scope: selectedScene ? "single_scene" : "all_scenes",
    selected_scene_number: selectedScene ? selectedScene.scene_number : null,
    performance_mode: normalizeStoryboardPerformanceMode(selectedScene?.performance_mode || state.performanceMode || state.videoType || state.performance_mode),
    storyboard_mode: state.mode === "image_to_video_prep" ? "video prompt planning" : "text-to-image prompt planning",
    image_model_mode: selectedImageMode,
    image_model_label: selectedImageModeLabel,
    image_prompt_target: imagePromptTarget,
    ...(imageMode
      ? {
        task_instruction: `Create detailed ${imagePromptTarget}s for Image Prep using advanced Krea 2-style still-image prompting. These are still-image prompts, not video or lip-sync prompts. Use lyrics and story beats for mood, symbolism, emotion, styling, and scene direction only. Do not say the subject is singing, lip-syncing, performing vocals, or singing the lyric unless the scene notes explicitly ask for a live singing image. Preserve mapped subject prompt names exactly as provided in each scene's visible_subjects and subjects.name. When a subject has a trigger_phrase, that trigger phrase is the subject identity for prompt wording, so write natural phrases like 'a photo of TRIGGER_PHRASE' instead of 'a photo of a woman'. Do not rename 'the woman' as 'one woman' or 'a woman', and do not rename trigger phrases. If global_consistency_phrase is present, weave it naturally into the prompt where it fits instead of slapping it onto the front.`,
        output_format: {
          type: "image_prompt_import_json",
          instruction: "Return only a JSON code block with an array of objects. Include every scene. Each object must have scene_number and image_prompt. Do not include prose outside the JSON code block.",
          example: [
            { scene_number: 1, image_prompt: `Full detailed ${imagePromptTarget} for scene 1...` },
            { scene_number: 2, image_prompt: `Full detailed ${imagePromptTarget} for scene 2...` },
          ],
        },
      }
      : {
        task_instruction: "Create detailed image-to-video prompts for Video Prep using a strict source hierarchy. The first_frame_visual_inventory field is only a first-frame inventory: visible subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition. Do not use first_frame_visual_inventory or any image prompt wording for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing. Build the video prompt in this order: 1) subject and vocal/performance sentence from vocal_status, performance_direction, and facial_performance_direction; 2) character movement sentence from character_motion, character_motion_guidance, character_motion_speed, and scene_story_beat; 3) camera movement sentence from camera_motion, camera_guidance, and camera_motion_speed_guidance; 4) environment/lighting sentence from first_frame_visual_inventory and location_ref; 5) final mood/style sentence from story_layer and image aesthetic only where visual. Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across multiple sentences. If an idea appears in the face sentence, do not repeat it in the body, camera, environment, or atmosphere sentence; use a different concrete visual detail instead. Do not duplicate adjacent words such as 'tall, tall'. The motion priority is character_motion_guidance + camera_motion_speed_guidance + camera_guidance + performance_direction + vocal_status + scene_story_beat above story_layer, and all of those above first_frame_visual_inventory. At camera speed 9-10, do not write 'then holds', 'holds on', or static hold endings; use multiple coordinated readable camera moves. At character speed 9-10, do not leave the subject merely poised or standing; include clear full-body action or set interaction.",
      }),
    story_layer: normalizeStoryLayer(state.storyLayer),
    scenes: storyboardScenesForGpt(payloadState),
  };
}

function openStoryboardGptUrl(payload) {
  const isImagePayload = String(payload?.storyboard_mode || "").toLowerCase().includes("text-to-image")
    || String(payload?.scenes?.[0]?.prompt_type || "").toLowerCase().includes("text to image");
  window.open(isImagePayload ? STORYBOARD_IMAGE_GPT_URL : STORYBOARD_GPT_URL, "_blank", "noopener,noreferrer");
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.cssText = "position:fixed;left:-9999px;top:-9999px;";
  document.body.append(textarea);
  textarea.focus();
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function openStoryboardBuilder(payload = {}) {
  const projectFolder = String(payload.projectFolder || payload.project_folder || "").trim();
  const payloadVideoPromptType = ["i2v", "id_lora", "t2v", "rtv", "ingredients"].includes(String(payload.videoPromptType || payload.video_prompt_type || "").trim())
    ? String(payload.videoPromptType || payload.video_prompt_type || "").trim()
    : "";
  const isIdLoraMode = payloadVideoPromptType === "id_lora";
  const payloadPerformanceMode = normalizeStoryboardPerformanceMode(payload.performanceMode || payload.performance_mode || payload.videoType || payload.video_type);
  const state = {
    projectFolder,
    mode: "storyboard_prompts",
    scenes: scenesFromBuilderPayload(payload).map((scene) => ({
      ...scene,
      video_prompt_type: payloadVideoPromptType || scene.video_prompt_type,
      performance_mode: scene.performance_mode || payloadPerformanceMode,
    })),
    referenceBuilder: normalizeReferenceBuilderCatalog(payload.referenceBuilder || payload.reference_builder || {}),
    storyLayer: normalizeStoryLayer(payload.storyLayer || payload.story_layer || {}),
    onReferenceMappingsChanged: typeof payload.onReferenceMappingsChanged === "function" ? payload.onReferenceMappingsChanged : null,
    onStoryLayerChanged: typeof payload.onStoryLayerChanged === "function" ? payload.onStoryLayerChanged : null,
    onPromptsExported: typeof payload.onPromptsExported === "function" ? payload.onPromptsExported : null,
    onApplyIdLoraDialoguePlan: typeof payload.onApplyIdLoraDialoguePlan === "function" ? payload.onApplyIdLoraDialoguePlan : null,
    onCreateVideoPrompt: typeof payload.onCreateVideoPrompt === "function" ? payload.onCreateVideoPrompt : null,
    query: "",
    selected: new Set(),
    saving: false,
    gemmaSettings: payload.gemmaSettings || payload.gemma_settings || {},
    cameraFlow: String(payload.cameraFlow || payload.camera_flow || "balanced"),
    imageShotFlow: String(payload.imageShotFlow || payload.image_shot_flow || (isIdLoraMode ? "film_dialogue_coverage" : "intimate")),
    imageAesthetic: String(payload.imageAesthetic || payload.image_aesthetic || (isIdLoraMode ? "film_default" : "")),
    globalConsistencyPhrase: String(payload.globalConsistencyPhrase || payload.global_consistency_phrase || ""),
    performanceStyle: String(payload.performanceStyle || payload.performance_style || payload.performance_style_default || (isIdLoraMode ? "dialogue_naturalism" : "")),
    facialPerformance: String(payload.facialPerformance || payload.facial_performance || payload.facial_performance_default || ""),
    facialPerformanceCustom: String(payload.facialPerformanceCustom || payload.facial_performance_custom || payload.facial_performance_custom_default || ""),
    cameraMotionSpeed: storyboardSpeedValue(payload.cameraMotionSpeed ?? payload.camera_motion_speed ?? payload.motion_defaults?.camera_motion_speed, 4),
    characterMotionSpeed: storyboardSpeedValue(payload.characterMotionSpeed ?? payload.character_motion_speed ?? payload.motion_defaults?.character_motion_speed, 4),
    performanceMode: payloadPerformanceMode,
    videoPromptType: payloadVideoPromptType,
    imageMode: String(payload.imageMode || payload.image_mode || "zimage").trim() || "zimage",
    imageModeLabel: String(payload.imageModeLabel || payload.image_mode_label || "").trim(),
  };

  const promptRunnerName = () => {
    const runner = String(state.gemmaSettings?.text_runner || state.gemmaSettings?.gemma_runner || "builtin").trim().toLowerCase();
    if (runner === "lm_studio" || runner === "lmstudio" || runner === "lm-studio") return "LM Studio";
    if (runner === "llm_api" || runner === "llmapi" || runner === "llm-api" || runner === "api") return "API LLM";
    return "Gemma";
  };
  const imageShotFlowPresets = isIdLoraMode ? ID_LORA_IMAGE_SHOT_FLOW_PRESETS : STORYBOARD_IMAGE_SHOT_FLOW_PRESETS;
  const imageAestheticPresets = isIdLoraMode ? ID_LORA_IMAGE_AESTHETIC_PRESETS : STORYBOARD_IMAGE_AESTHETIC_PRESETS;
  const performanceStylePresets = isIdLoraMode ? ID_LORA_PERFORMANCE_STYLE_PRESETS : PERFORMANCE_STYLE_PRESETS;
  const facialPerformancePresets = isIdLoraMode ? ID_LORA_FACIAL_PERFORMANCE_PRESETS : FACIAL_PERFORMANCE_PRESETS;
  const imageShotFlowPresetForMode = (value = "") => imageShotFlowPresets[value] || imageShotFlowPresets[Object.keys(imageShotFlowPresets)[0]] || STORYBOARD_IMAGE_SHOT_FLOW_PRESETS.intimate;
  const imageAestheticPresetForMode = (value = "") => imageAestheticPresets.find((item) => item.value === value) || imageAestheticPresets[0] || STORYBOARD_IMAGE_AESTHETIC_PRESETS[0];
  const performancePresetForMode = (value = "") => performanceStylePresets.find((item) => item.value === value) || performanceStylePresets[0] || PERFORMANCE_STYLE_PRESETS[0];
  const facialPresetForMode = (value = "") => facialPerformancePresets.find((item) => item.value === value) || facialPerformancePresets[0] || FACIAL_PERFORMANCE_PRESETS[0];
  if (!imageShotFlowPresets[state.imageShotFlow]) state.imageShotFlow = Object.keys(imageShotFlowPresets)[0] || "off";
  if (!imageAestheticPresets.some((item) => item.value === state.imageAesthetic)) state.imageAesthetic = imageAestheticPresets[0]?.value || "";
  if (!performanceStylePresets.some((item) => item.value === state.performanceStyle)) state.performanceStyle = performanceStylePresets[0]?.value || "";
  if (!facialPerformancePresets.some((item) => item.value === state.facialPerformance)) state.facialPerformance = facialPerformancePresets[0]?.value || "";
  const storyboardDefaultsPayload = () => ({
    builder_storyboard_defaults: {
      global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
      performance_style: String(state.performanceStyle || ""),
      camera_flow: String(state.cameraFlow || ""),
      image_shot_flow: String(state.imageShotFlow || ""),
      image_aesthetic: String(state.imageAesthetic || ""),
    },
    global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
    performance_style_default: String(state.performanceStyle || ""),
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    motion_defaults: {
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
    },
  });
  const promptRunnerGenericName = () => promptRunnerName() === "Gemma" ? "Gemma" : "LLM";
  const promptAllButtonText = () => {
    const kind = state.mode === "image_to_video_prep" ? "Video" : "Image";
    return `${promptRunnerName()} ${kind} All`;
  };

  const absorbSceneReferencesIntoCatalog = (scenes = []) => {
    const refs = normalizeReferenceBuilderCatalog(state.referenceBuilder || {});
    const locationIds = new Set(refs.locations.map((location) => String(location.id || "")).filter(Boolean));
    const locationByName = new Map(
      refs.locations
        .map((location) => [String(location.name || "").trim().toLowerCase().replace(/\s+/g, " "), location])
        .filter(([name]) => Boolean(name)),
    );
    const subjectIds = new Set(refs.subjects.map((subject) => String(subject.id || "")).filter(Boolean));
    for (const scene of scenes || []) {
      let location = scene?.location_ref;
      if ((!location || typeof location !== "object") && String(scene?.setting || "").trim()) {
        location = {
          id: "",
          name: String(scene.setting || "").trim(),
          description: String(scene.setting || "").trim(),
          image: { path: "", data: "", name: "" },
        };
      }
    if (location && typeof location === "object" && String(location.id || location.name || location.description || "").trim()) {
      const locationNameKey = String(location.name || scene.setting || "").trim().toLowerCase().replace(/\s+/g, " ");
      const existingLocation = locationNameKey ? locationByName.get(locationNameKey) : null;
      const id = String(existingLocation?.id || location.id || `location_from_scene_${scene.scene_number || refs.locations.length + 1}`).trim();
      location.id = id;
      scene.location_ref = location;
      if (!locationIds.has(id)) {
        const addedLocation = {
            id,
            name: String(location.name || scene.setting || "Saved location"),
            description: String(location.description || ""),
            trigger_phrase: String(location.trigger_phrase || ""),
            trigger_position: String(location.trigger_position || "start") === "end" ? "end" : "start",
            image: normalizeReferenceImage(location),
          };
          refs.locations.push(addedLocation);
          locationIds.add(id);
          const addedNameKey = String(addedLocation.name || "").trim().toLowerCase().replace(/\s+/g, " ");
          if (addedNameKey) locationByName.set(addedNameKey, addedLocation);
        }
      }
      for (const subject of Array.isArray(scene?.subject_refs) ? scene.subject_refs : []) {
        if (!subject || typeof subject !== "object") continue;
        const id = String(subject.id || subject.name || "").trim();
        if (!id || subjectIds.has(id)) continue;
        refs.subjects.push({
          id,
          name: String(subject.name || "Saved subject"),
          description: String(subject.description || ""),
          trigger_phrase: String(subject.trigger_phrase || ""),
          trigger_position: String(subject.trigger_position || "start") === "end" ? "end" : "start",
          image: normalizeReferenceImage(subject),
        });
        subjectIds.add(id);
      }
    }
    state.referenceBuilder = normalizeReferenceBuilderCatalog(refs);
  };

  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:stretch;justify-content:center;padding:18px;";
  const shell = document.createElement("div");
  shell.style.cssText = "width:min(1820px,calc(100vw - 36px));height:calc(100vh - 36px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#e5e7eb;box-shadow:0 28px 90px rgba(0,0,0,.62);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,sans-serif;";

  const header = document.createElement("div");
  header.style.cssText = "display:grid;grid-template-columns:minmax(280px,1fr) auto auto;gap:22px;align-items:center;padding:24px;border-bottom:1px solid #1f3b46;background:linear-gradient(180deg,#083344,#111827);";
  const titleBlock = document.createElement("div");
  titleBlock.innerHTML = `
    <div style="display:flex;gap:14px;align-items:center;">
      <div style="width:52px;height:52px;border-radius:12px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;">▣</div>
      <div>
        <div style="font-size:26px;font-weight:900;color:#cffafe;">Storyboard Builder <span id="vrgdg-storyboard-mode-pill" style="font-size:13px;border-radius:999px;background:#164e63;color:#a5f3fc;padding:5px 9px;vertical-align:middle;">Planning</span></div>
        <div id="vrgdg-storyboard-subtitle" style="color:#cbd5e1;font-size:14px;margin-top:3px;">Write scene cards, image prompts, and video prompts before sending them to the Video Creator.</div>
      </div>
    </div>
  `;
  const steps = document.createElement("div");
  steps.style.cssText = "display:flex;gap:14px;align-items:center;min-width:520px;";
  const stepPrompts = makeButton("Image Prep", "purple");
  const stepPrep = makeButton("Video Prep");
  stepPrompts.style.minWidth = "220px";
  stepPrep.style.minWidth = "190px";
  steps.append(stepPrompts, stepPrep);
  const headerActions = document.createElement("div");
  headerActions.style.cssText = "display:flex;gap:10px;align-items:center;justify-content:flex-end;";
  const search = makeInput("", "Search scenes...");
  search.style.width = "260px";
  const gptButton = makeButton("GPT All", "purple");
  gptButton.title = "Copy all Storyboard scene-card inputs as JSON for your custom GPT.";
  const importImagePromptsButton = makeButton("Import prompts from GPT", "purple");
  importImagePromptsButton.title = "Paste JSON from the Text to Image Prompt Builder GPT and update Image Prep prompts.";
  const gemmaAllButton = makeButton(promptAllButtonText(), "primary");
  gemmaAllButton.title = "Use the selected LLM runner to create prompts for every storyboard scene.";
  const clearPromptsButton = makeButton("Clear Prompts");
  clearPromptsButton.title = "Clear Storyboard scene-card prompt summaries, generated prompts, and extra notes without changing subjects, locations, camera, motion, or lyrics.";
  clearPromptsButton.style.borderColor = "#991b1b";
  clearPromptsButton.style.background = "#3f0808";
  const keepGemmaLoadedLabel = document.createElement("label");
  keepGemmaLoadedLabel.style.cssText = "display:flex;align-items:center;gap:6px;color:#cbd5e1;font-size:12px;font-weight:800;white-space:nowrap;";
  const keepGemmaLoadedInput = document.createElement("input");
  keepGemmaLoadedInput.type = "checkbox";
  keepGemmaLoadedInput.checked = Boolean(state.gemmaSettings?.keep_loaded_for_storyboard_all);
  keepGemmaLoadedLabel.append(keepGemmaLoadedInput, document.createTextNode("Keep local LLM loaded"));
  keepGemmaLoadedLabel.title = "When checked, local Gemma keeps the text model loaded until the batch finishes. This has no effect on API runners.";
  const add = makeButton("+ Add Scene", "purple");
  const close = makeButton("Close");
  headerActions.append(gptButton, importImagePromptsButton, gemmaAllButton, clearPromptsButton, keepGemmaLoadedLabel, search, add, close);
  header.append(titleBlock, steps, headerActions);

  const note = document.createElement("div");
  note.style.cssText = "margin:18px 24px 0;border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:12px 14px;font-size:13px;";
  const middleContent = document.createElement("div");
  middleContent.style.cssText = "min-height:0;overflow-y:auto;overflow-x:hidden;padding-bottom:18px;scrollbar-width:thin;";

  const cameraFlowBar = document.createElement("div");
  cameraFlowBar.style.cssText = "display:grid;grid-template-columns:auto minmax(280px,1fr);gap:8px 12px;align-items:center;color:#cbd5e1;font-size:12px;";
  const imageShotControls = document.createElement("div");
  imageShotControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageShotLabel = document.createElement("div");
  imageShotLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageShotLabel.textContent = "Still shot flow";
  const imageShotSelect = makeSelect(
    Object.entries(imageShotFlowPresets).map(([value, preset]) => ({ value, label: preset.label })),
    state.imageShotFlow,
  );
  imageShotSelect.style.width = "max-content";
  imageShotSelect.style.minWidth = "180px";
  const imageShotApply = makeButton("Fill Missing", "primary");
  imageShotApply.title = "Fill only blank shot/composition fields for Image Prep. Existing manual choices are kept.";
  const imageShotReplace = makeButton("Replace All");
  imageShotReplace.title = "Replace every scene's shot/composition field with the selected still shot flow.";
  imageShotControls.append(imageShotLabel, imageShotSelect, imageShotApply, imageShotReplace);
  const imageShotInfo = document.createElement("div");
  imageShotInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageAestheticControls = document.createElement("div");
  imageAestheticControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageAestheticLabel = document.createElement("div");
  imageAestheticLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageAestheticLabel.textContent = "Image aesthetic";
  const imageAestheticSelect = makeSelect(imageAestheticPresets, state.imageAesthetic);
  imageAestheticSelect.style.width = "max-content";
  imageAestheticSelect.style.minWidth = "180px";
  const imageAestheticApply = makeButton("Fill Missing", "primary");
  imageAestheticApply.title = "Fill only scenes without a still camera style/aesthetic note.";
  const imageAestheticReplace = makeButton("Replace All");
  imageAestheticReplace.title = "Replace each scene's generated image aesthetic note.";
  imageAestheticControls.append(imageAestheticLabel, imageAestheticSelect, imageAestheticApply, imageAestheticReplace);
  const imageAestheticInfo = document.createElement("div");
  imageAestheticInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const consistencyControls = document.createElement("div");
  consistencyControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const consistencyLabel = document.createElement("div");
  consistencyLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  consistencyLabel.textContent = "Global consistency phrase";
  const consistencyInput = makeInput(state.globalConsistencyPhrase, "e.g. soft glittery eye makeup, wet-look hair, chrome jewelry");
  consistencyInput.style.minWidth = "520px";
  consistencyControls.append(consistencyLabel, consistencyInput);
  const consistencyInfo = document.createElement("div");
  consistencyInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cameraFlowControls = document.createElement("div");
  cameraFlowControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cameraFlowLabel = document.createElement("div");
  cameraFlowLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cameraFlowLabel.textContent = "Auto camera flow";
  const cameraFlowSelect = makeSelect(
    Object.entries(STORYBOARD_CAMERA_FLOW_PRESETS).map(([value, preset]) => ({ value, label: preset.label })),
    state.cameraFlow,
  );
  cameraFlowSelect.style.width = "max-content";
  cameraFlowSelect.style.minWidth = "180px";
  const cameraFlowApply = makeButton("Fill Missing", "primary");
  cameraFlowApply.title = "Fill only blank shot type and camera motion fields. Existing manual choices are kept.";
  const cameraFlowReplace = makeButton("Replace All");
  cameraFlowReplace.title = "Replace every scene's shot type and camera motion with the selected auto camera flow.";
  cameraFlowControls.append(cameraFlowLabel, cameraFlowSelect, cameraFlowApply, cameraFlowReplace);
  const cameraFlowInfo = document.createElement("div");
  cameraFlowInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cameraSpeedControls = document.createElement("div");
  cameraSpeedControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cameraSpeedLabel = document.createElement("div");
  cameraSpeedLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cameraSpeedLabel.textContent = "Camera motion speed";
  const cameraSpeedInput = makeInput(String(storyboardSpeedValue(state.cameraMotionSpeed, 4)));
  cameraSpeedInput.type = "range";
  cameraSpeedInput.min = "0";
  cameraSpeedInput.max = "10";
  cameraSpeedInput.step = "1";
  cameraSpeedInput.style.minWidth = "360px";
  cameraSpeedInput.style.accentColor = "#22d3ee";
  const cameraSpeedValue = document.createElement("div");
  cameraSpeedValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:120px;";
  const cameraSpeedHint = makeButton("Hint");
  cameraSpeedHint.title = "Explain camera motion speed.";
  cameraSpeedControls.append(cameraSpeedLabel, cameraSpeedInput, cameraSpeedValue, cameraSpeedHint);
  const cameraSpeedInfo = document.createElement("div");
  cameraSpeedInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const performanceControls = document.createElement("div");
  performanceControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const performanceLabel = document.createElement("div");
  performanceLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  performanceLabel.textContent = isIdLoraMode ? "Global acting style" : "Global performance style";
  const performanceSelect = makeSelect(performanceStylePresets, state.performanceStyle);
  performanceSelect.style.width = "max-content";
  performanceSelect.style.minWidth = "180px";
  const performanceApply = makeButton("Fill Missing", "primary");
  performanceApply.title = isIdLoraMode ? "Fill only blank per-scene acting style fields. Existing scene choices are kept." : "Fill only blank per-scene performance/song style fields. Existing scene choices are kept.";
  const performanceReplace = makeButton("Replace All");
  performanceReplace.title = isIdLoraMode ? "Replace every scene's acting style with the selected global style." : "Replace every scene's performance/song style with the selected global style.";
  performanceControls.append(performanceLabel, performanceSelect, performanceApply, performanceReplace);
  const performanceInfo = document.createElement("div");
  performanceInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const characterSpeedControls = document.createElement("div");
  characterSpeedControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const characterSpeedLabel = document.createElement("div");
  characterSpeedLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  characterSpeedLabel.textContent = "Character motion speed";
  const characterSpeedInput = makeInput(String(storyboardSpeedValue(state.characterMotionSpeed, 4)));
  characterSpeedInput.type = "range";
  characterSpeedInput.min = "0";
  characterSpeedInput.max = "10";
  characterSpeedInput.step = "1";
  characterSpeedInput.style.minWidth = "360px";
  characterSpeedInput.style.accentColor = "#22d3ee";
  const characterSpeedValue = document.createElement("div");
  characterSpeedValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:120px;";
  const characterSpeedHint = makeButton("Hint");
  characterSpeedHint.title = "Explain character motion speed.";
  characterSpeedControls.append(characterSpeedLabel, characterSpeedInput, characterSpeedValue, characterSpeedHint);
  const characterSpeedInfo = document.createElement("div");
  characterSpeedInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const facialControls = document.createElement("div");
  facialControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const facialLabel = document.createElement("div");
  facialLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  facialLabel.textContent = isIdLoraMode ? "Global screen face" : "Global facial performance";
  const facialSelect = makeSelect(facialPerformancePresets, state.facialPerformance);
  facialSelect.style.width = "max-content";
  facialSelect.style.minWidth = "180px";
  const facialApply = makeButton("Fill Missing", "primary");
  facialApply.title = "Fill only blank per-scene facial performance fields.";
  const facialReplace = makeButton("Replace All");
  facialReplace.title = "Replace every scene's facial performance with the selected global facial preset.";
  facialControls.append(facialLabel, facialSelect, facialApply, facialReplace);
  const facialInfo = document.createElement("div");
  facialInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const facialCustomControls = document.createElement("div");
  facialCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;white-space:nowrap;";
  const facialCustomLabel = document.createElement("div");
  facialCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:8px;";
  facialCustomLabel.textContent = "Custom facial text";
  const facialCustomInput = makeTextarea(state.facialPerformanceCustom || "", "Optional custom facial performance text, e.g. expressive eyes, active brows, natural blinking...", 3);
  facialCustomInput.style.minWidth = "520px";
  facialCustomControls.append(facialCustomLabel, facialCustomInput);
  const facialCustomInfo = document.createElement("div");
  facialCustomInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  cameraFlowBar.append(imageShotControls, imageShotInfo, imageAestheticControls, imageAestheticInfo, consistencyControls, consistencyInfo, cameraFlowControls, cameraFlowInfo, cameraSpeedControls, cameraSpeedInfo, performanceControls, performanceInfo, characterSpeedControls, characterSpeedInfo, facialControls, facialInfo, facialCustomControls, facialCustomInfo);

  const storyLayerBar = document.createElement("div");
  storyLayerBar.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:12px;color:#cbd5e1;font-size:12px;";
  const storyLayerHeader = document.createElement("div");
  storyLayerHeader.style.cssText = "grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:12px;";
  const storyLayerTitle = document.createElement("div");
  storyLayerTitle.innerHTML = isIdLoraMode
    ? `<div style="font-weight:900;color:#cffafe;font-size:15px;">Short Film Story Layer</div><div style="color:#94a3b8;margin-top:2px;">Dialogue-first planning for ID-LoRA scenes, characters, and locations.</div>`
    : `<div style="font-weight:900;color:#cffafe;font-size:15px;">Story Layer</div><div style="color:#94a3b8;margin-top:2px;">Optional narrative context for connecting lyrics, sections, subjects, and locations across scenes.</div>`;
  const storyLayerEnabledLabel = document.createElement("label");
  storyLayerEnabledLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;white-space:nowrap;";
  const storyLayerEnabledInput = document.createElement("input");
  storyLayerEnabledInput.type = "checkbox";
  storyLayerEnabledInput.checked = state.storyLayer.enabled !== false;
  storyLayerEnabledLabel.append(storyLayerEnabledInput, document.createTextNode("Use in Gemma prompts"));
  storyLayerHeader.append(storyLayerTitle, storyLayerEnabledLabel);
  const userStoryArcInput = makeTextarea(
    state.storyLayer.user_story_arc || "",
    isIdLoraMode ? "Short film premise, conflict, tone, character goal..." : "Optional user story arc, e.g. Verse 1: she feels trapped. Chorus: she breaks free...",
    5,
  );
  const songStoryBriefInput = makeTextarea(
    state.storyLayer.song_story_brief || "",
    isIdLoraMode ? "Gemma-created short film story brief..." : "Gemma-created song story brief...",
    5,
  );
  const lyricStoryStrengthInput = makeInput(String(normalizeStoryLayer(state.storyLayer).lyric_story_strength));
  lyricStoryStrengthInput.type = "range";
  lyricStoryStrengthInput.min = "0";
  lyricStoryStrengthInput.max = "10";
  lyricStoryStrengthInput.step = "1";
  lyricStoryStrengthInput.style.accentColor = "#22d3ee";
  const lyricStoryStrengthValue = document.createElement("div");
  lyricStoryStrengthValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:105px;text-align:right;";
  const lyricStoryStrengthHintButton = makeButton("Hint");
  lyricStoryStrengthHintButton.title = "Explain Lyric Story Strength.";
  const lyricStoryStrengthText = (value) => {
    const strength = Math.max(0, Math.min(10, Number(value || 7)));
    if (strength <= 0) return "0 / ignore lyrics";
    if (strength <= 3) return `${strength} / mood only`;
    if (strength <= 6) return `${strength} / balanced`;
    if (strength <= 8) return `${strength} / strong lyric story`;
    return `${strength} / literal lyric anchors`;
  };
  const syncLyricStoryStrengthLabel = () => {
    lyricStoryStrengthValue.textContent = lyricStoryStrengthText(lyricStoryStrengthInput.value);
  };
  const storyField = (label, control) => {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:900;color:#cbd5e1;";
    wrap.textContent = label;
    wrap.append(control);
    return wrap;
  };
  syncLyricStoryStrengthLabel();
  const lyricStoryStrengthRow = document.createElement("div");
  lyricStoryStrengthRow.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:end;";
  lyricStoryStrengthRow.append(storyField("Lyric Story Strength", lyricStoryStrengthInput), lyricStoryStrengthValue, lyricStoryStrengthHintButton);
  lyricStoryStrengthRow.style.display = isIdLoraMode ? "none" : "grid";
  const idLoraDialoguePlanner = document.createElement("div");
  idLoraDialoguePlanner.style.cssText = "grid-column:1/-1;display:none;border:1px solid #155e75;border-radius:8px;background:#082f49;padding:12px;gap:10px;align-items:center;grid-template-columns:minmax(0,1fr) auto;";
  const idLoraDialoguePlannerText = document.createElement("div");
  idLoraDialoguePlannerText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Plan Dialogue Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, Gemma invents a short-film dialogue scene plan from your ID-LoRA characters and locations.</div>`;
  const idLoraDialogueControls = document.createElement("div");
  idLoraDialogueControls.style.cssText = "display:flex;gap:8px;align-items:end;flex-wrap:wrap;justify-content:flex-end;";
  const idLoraDialogueSceneCount = makeInput("6");
  idLoraDialogueSceneCount.type = "number";
  idLoraDialogueSceneCount.min = "1";
  idLoraDialogueSceneCount.max = "24";
  idLoraDialogueSceneCount.step = "1";
  idLoraDialogueSceneCount.style.width = "76px";
  const planDialogueScenesButton = makeButton("Plan Dialogue Scenes", "primary");
  planDialogueScenesButton.title = "ID-LoRA only. Create a preview scene plan from the premise/script and ID-LoRA Ref Builder characters.";
  const applyDialoguePlanButton = makeButton("Apply Dialogue Plan", "primary");
  applyDialoguePlanButton.title = "Apply the reviewed ID-LoRA dialogue scenes to Video Builder.";
  applyDialoguePlanButton.style.display = "none";
  idLoraDialogueControls.append(storyField("Scenes", idLoraDialogueSceneCount), planDialogueScenesButton, applyDialoguePlanButton);
  idLoraDialoguePlanner.append(idLoraDialoguePlannerText, idLoraDialogueControls);
  const storyActions = document.createElement("div");
  storyActions.style.cssText = "grid-column:1/-1;display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
  const createStoryArcButton = makeButton("Create User Story Arc", "primary");
  const createStoryBriefButton = makeButton("Create Story Brief", "primary");
  const createMissingBeatsButton = makeButton("Create Missing Scene Beats", "purple");
  const replaceBeatsButton = makeButton("Replace All Scene Beats");
  const detectSectionsButton = makeButton("Detect Lyric Sections");
  storyActions.append(createStoryArcButton, createStoryBriefButton, createMissingBeatsButton, replaceBeatsButton, detectSectionsButton);
  storyLayerBar.append(
    storyLayerHeader,
    lyricStoryStrengthRow,
    storyField("User Story Arc", userStoryArcInput),
    storyField("Song Story Brief", songStoryBriefInput),
    idLoraDialoguePlanner,
    storyActions,
  );

  const sceneDefaultsPanel = makeCollapsiblePanel("Scene Defaults", "", cameraFlowBar, { open: false });
  const hasStoryLayerContent = Boolean(String(state.storyLayer.user_story_arc || "").trim() || String(state.storyLayer.song_story_brief || "").trim());
  const storyLayerPanel = makeCollapsiblePanel("Story Layer", "", storyLayerBar, { open: hasStoryLayerContent });

  const tableWrap = document.createElement("div");
  tableWrap.style.cssText = "margin:10px 24px 18px;overflow:auto;border:1px solid #334155;border-radius:10px;background:#0b1220;min-height:0;";

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 24px;border-top:1px solid #334155;background:#111827;";
  const stats = document.createElement("div");
  stats.style.cssText = "color:#cbd5e1;font-size:13px;";
  const footerActions = document.createElement("div");
  footerActions.style.cssText = "display:flex;gap:10px;align-items:center;";
  const save = makeButton("Save Storyboard");
  const exportPrompts = makeButton("Export Prompt Files", "purple");
  exportPrompts.title = "Export text prompt files plus clean JSON files keyed by scene number.";
  footerActions.append(save, exportPrompts);
  footer.append(stats, footerActions);

  middleContent.append(sceneDefaultsPanel, storyLayerPanel, tableWrap);
  shell.append(header, note, middleContent, footer);
  backdrop.append(shell);
  document.body.append(backdrop);

  const setMode = (mode) => {
    state.mode = mode;
    const isVideoPrepMode = mode === "image_to_video_prep";
    stepPrompts.style.background = mode === "storyboard_prompts" ? "#0e7490" : "#2b2b30";
    stepPrompts.style.borderColor = mode === "storyboard_prompts" ? "#06b6d4" : "#3f3f46";
    stepPrep.style.background = mode === "image_to_video_prep" ? "#0e7490" : "#2b2b30";
    stepPrep.style.borderColor = mode === "image_to_video_prep" ? "#06b6d4" : "#3f3f46";
    shell.querySelector("#vrgdg-storyboard-mode-pill").textContent = mode === "image_to_video_prep" ? "Video Prep" : "Planning";
    shell.querySelector("#vrgdg-storyboard-subtitle").textContent = mode === "image_to_video_prep"
      ? "Use scene images with vision guidance to create video prompts before rendering."
      : "Create text-to-image prompts for each scene before image generation.";
    note.textContent = mode === "image_to_video_prep"
      ? "Video Prep uses existing scene images when available, plus subjects, locations, lyrics, story beats, and motion notes to create video prompts."
      : "Image Prep creates text-to-image prompts from subjects, locations, lyrics, story beats, shot direction, and the story layer.";
    gemmaAllButton.textContent = promptAllButtonText();
    gemmaAllButton.title = mode === "image_to_video_prep"
      ? "Create video prompts for the visible scenes. If a scene has an image path, local vision uses it as guidance."
      : "Create text-to-image prompts for the visible scenes with the selected LLM runner.";
    gptButton.textContent = mode === "image_to_video_prep" ? "GPT Video All" : "GPT Image All";
    gptButton.title = mode === "image_to_video_prep"
      ? "Copy all Storyboard scene-card inputs as JSON and open the video prompt GPT."
      : "Copy all Image Prep scene-card inputs as JSON and open the Krea 2 text-to-image prompt GPT.";
    importImagePromptsButton.style.display = isVideoPrepMode ? "none" : "";
    imageShotControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageShotInfo.style.display = isVideoPrepMode ? "none" : "";
    imageAestheticControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageAestheticInfo.style.display = isVideoPrepMode ? "none" : "";
    cameraFlowControls.style.display = isVideoPrepMode ? "flex" : "none";
    cameraFlowInfo.style.display = isVideoPrepMode ? "" : "none";
    cameraSpeedControls.style.display = isVideoPrepMode ? "flex" : "none";
    cameraSpeedInfo.style.display = isVideoPrepMode ? "" : "none";
    characterSpeedControls.style.display = isVideoPrepMode ? "flex" : "none";
    characterSpeedInfo.style.display = isVideoPrepMode ? "" : "none";
    refreshConsistencyInfo();
    refreshSetupPanelSummaries();
    renderTable();
  };

  const cameraFlowEntryForScene = (profileKey, sceneIndex, previousMotion = "") => {
    return storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion);
  };

  const sceneLooksLikeStarterPlaceholder = (scene = {}) => {
    const text = [
      scene.lyrics,
      scene.story_beat,
      scene.prompt_summary,
      scene.motion_summary,
      scene.image_prompt,
      scene.video_prompt,
      scene.image_path,
      scene.setting,
    ].map((item) => String(item || "").trim()).join("");
    return !text;
  };

  const shouldShowIdLoraDialoguePlanner = () => {
    return isIdLoraMode
      && state.scenes.length > 0
      && state.scenes.length <= 2
      && state.scenes.every(sceneLooksLikeStarterPlaceholder);
  };
  const hasIdLoraDialoguePlan = () => {
    return isIdLoraMode
      && state.scenes.some((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim())
      && state.scenes.some((scene) => String(scene.video_prompt_type || "") === "id_lora");
  };

  const refreshSetupPanelSummaries = () => {
    const cameraPreset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const imageShotPreset = imageShotFlowPresetForMode(state.imageShotFlow);
    const imageAestheticPreset = imageAestheticPresetForMode(state.imageAesthetic);
    const performancePreset = performancePresetForMode(state.performanceStyle);
    const facialPreset = facialPresetForMode(state.facialPerformance);
    sceneDefaultsPanel.setSummary(state.mode === "image_to_video_prep"
      ? `${cameraPreset.label || "Camera flow"} · camera ${storyboardSpeedValue(state.cameraMotionSpeed, 4)}/10 · character ${storyboardSpeedValue(state.characterMotionSpeed, 4)}/10 · ${performancePreset.label || "Performance style"} · ${facialPreset.label || "Facial performance"}${state.globalConsistencyPhrase ? " · consistency phrase" : ""}`
      : `${imageShotPreset.label || "Still shot flow"} · ${imageAestheticPreset.label || "Image aesthetic"} · ${performancePreset.label || "Performance style"} · ${facialPreset.label || "Facial performance"}${state.globalConsistencyPhrase ? " · consistency phrase" : ""}`);
    const beatCount = state.scenes.filter((scene) => String(scene.story_beat || "").trim()).length;
    const sectionCount = state.scenes.filter((scene) => String(scene.lyric_section || "").trim()).length;
    const hasBrief = Boolean(String(state.storyLayer.song_story_brief || "").trim());
    const hasArc = Boolean(String(state.storyLayer.user_story_arc || "").trim());
    const lyricStrength = normalizeStoryLayer(state.storyLayer).lyric_story_strength;
    const idLoraPlannerVisible = shouldShowIdLoraDialoguePlanner();
    idLoraDialoguePlanner.style.display = (idLoraPlannerVisible || hasIdLoraDialoguePlan()) ? "grid" : "none";
    applyDialoguePlanButton.style.display = hasIdLoraDialoguePlan() && state.onApplyIdLoraDialoguePlan ? "" : "none";
    createStoryArcButton.textContent = isIdLoraMode ? "Create Story Premise" : "Create User Story Arc";
    createStoryBriefButton.textContent = isIdLoraMode ? "Create Short Film Brief" : "Create Story Brief";
    createMissingBeatsButton.textContent = isIdLoraMode ? "Create Missing Scene Beats" : "Create Missing Scene Beats";
    replaceBeatsButton.textContent = isIdLoraMode ? "Replace All Scene Beats" : "Replace All Scene Beats";
    detectSectionsButton.style.display = isIdLoraMode ? "none" : "";
    storyLayerPanel.setSummary(isIdLoraMode
      ? `${state.storyLayer.enabled === false ? "Off" : "On"} · ID-LoRA dialogue story · ${beatCount}/${state.scenes.length} beats${hasBrief ? " · brief" : ""}${hasArc ? " · premise" : ""}${idLoraPlannerVisible ? " · starter scenes" : ""}`
      : `${state.storyLayer.enabled === false ? "Off" : "On"} · lyric ${lyricStrength}/10 · ${beatCount}/${state.scenes.length} beats · ${sectionCount}/${state.scenes.length} sections${hasBrief ? " · brief" : ""}${hasArc ? " · user arc" : ""}`);
  };

  const refreshCameraFlowInfo = () => {
    const preset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const count = preset.sequence?.length || 0;
    cameraFlowInfo.textContent = state.cameraFlow === "off"
      ? preset.description
      : `${preset.description} For any scene count, it cycles through ${count} camera beats and only fills blank fields.`;
    refreshSetupPanelSummaries();
  };

  const refreshCameraSpeedInfo = () => {
    cameraSpeedValue.textContent = storyboardSpeedLabel(state.cameraMotionSpeed, "camera");
    cameraSpeedInfo.textContent = storyboardSpeedGuidance(state.cameraMotionSpeed, "camera");
    refreshSetupPanelSummaries();
  };

  const refreshImageShotInfo = () => {
    const preset = imageShotFlowPresetForMode(state.imageShotFlow);
    const count = preset.sequence?.length || 0;
    imageShotInfo.textContent = state.imageShotFlow === "off"
      ? preset.description
      : `${preset.description} Cycles through ${count} still compositions and only fills blank shot fields.`;
    refreshSetupPanelSummaries();
  };

  const refreshImageAestheticInfo = () => {
    const preset = imageAestheticPresetForMode(state.imageAesthetic);
    imageAestheticInfo.textContent = `${preset.description} Used as still-image aesthetic guidance for Image Prep.`;
    refreshSetupPanelSummaries();
  };

  const refreshConsistencyInfo = () => {
    consistencyInfo.textContent = state.globalConsistencyPhrase
      ? "Gemma will incorporate this phrase into every generated prompt while keeping the wording as intact as the scene allows."
      : "Optional phrase Gemma should preserve across every prompt, such as makeup, styling, texture, wardrobe detail, or visual motif.";
    refreshSetupPanelSummaries();
  };

  const refreshPerformanceInfo = () => {
    const preset = performancePresetForMode(state.performanceStyle);
    performanceInfo.textContent = state.performanceStyle
      ? `${preset.description} Used by Gemma/GPT for scenes without a per-scene ${isIdLoraMode ? "acting" : "performance"} style.`
      : `${preset.description} Pick a style here to use it as the default for blank scenes.`;
    refreshSetupPanelSummaries();
  };

  const refreshCharacterSpeedInfo = () => {
    characterSpeedValue.textContent = storyboardSpeedLabel(state.characterMotionSpeed, "character");
    characterSpeedInfo.textContent = storyboardSpeedGuidance(state.characterMotionSpeed, "character");
    refreshSetupPanelSummaries();
  };

  const refreshFacialInfo = () => {
    const preset = facialPresetForMode(state.facialPerformance);
    facialInfo.textContent = state.facialPerformance
      ? `${preset.description} Used by Gemma/GPT for scenes without a per-scene facial performance preset.`
      : `${preset.description} Pick a preset here to use it as the default for blank scenes.`;
    facialCustomInfo.textContent = state.facialPerformanceCustom
      ? "Custom facial text is appended to the selected preset, or used directly when Custom is selected."
      : "Optional custom wording for eyes, brows, cheeks, jaw, mouth behavior, emotion, and blinking.";
    refreshSetupPanelSummaries();
  };

  const syncStoryLayerFromInputs = ({ notify = false } = {}) => {
    state.storyLayer = normalizeStoryLayer({
      enabled: storyLayerEnabledInput.checked,
      user_story_arc: userStoryArcInput.value,
      song_story_brief: songStoryBriefInput.value,
      lyric_story_strength: lyricStoryStrengthInput.value,
    });
    if (notify && state.onStoryLayerChanged) {
      state.onStoryLayerChanged({
        ...storyboardDefaultsPayload(),
        story_layer: normalizeStoryLayer(state.storyLayer),
        facial_performance_default: state.facialPerformance || "",
        facial_performance_custom_default: state.facialPerformanceCustom || "",
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
      });
    }
    refreshSetupPanelSummaries();
  };
  const notifyStoryboardDefaultsChanged = () => {
    if (!state.onStoryLayerChanged) return;
    state.onStoryLayerChanged({
      ...storyboardDefaultsPayload(),
      story_layer: normalizeStoryLayer(state.storyLayer),
      facial_performance_default: state.facialPerformance || "",
      facial_performance_custom_default: state.facialPerformanceCustom || "",
      scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
    });
  };

  const lyricsForStoryBrief = () => {
    return state.scenes
      .map((scene, index) => {
        const normalized = normalizeScene(scene, index);
        const section = String(normalized.lyric_section || "").trim();
        const lyric = String(normalized.lyrics || "").trim();
        if (!section && !lyric) return "";
        return `${section ? `[${section}]\n` : ""}${lyric}`;
      })
      .filter(Boolean)
      .join("\n\n");
  };

  const sectionMapFromLyrics = () => {
    const map = new Map();
    let current = "";
    state.scenes.forEach((scene, index) => {
      const lyric = String(scene.lyrics || "").trim();
      const explicit = String(scene.lyric_section || "").trim();
      const header = lyric.match(/^\s*\[([^\]]{2,80})\]\s*$/);
      if (explicit) current = explicit;
      else if (header) current = header[1].trim();
      else if (current) map.set(scene.id || `scene_${index + 1}`, current);
    });
    return map;
  };

  const detectLyricSections = () => {
    const map = sectionMapFromLyrics();
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const key = scene.id || `scene_${index + 1}`;
      const section = map.get(key);
      const lyric = String(scene.lyrics || "").trim();
      const header = lyric.match(/^\s*\[([^\]]{2,80})\]\s*$/);
      if (header && !String(scene.lyric_section || "").trim()) {
        scene.lyric_section = header[1].trim();
        changed += 1;
      } else if (section && !String(scene.lyric_section || "").trim()) {
        scene.lyric_section = section;
        changed += 1;
      }
    });
    renderTable();
    syncStoryLayerFromInputs();
    createToast(changed ? `Detected lyric sections for ${changed} scene${changed === 1 ? "" : "s"}.` : "No missing lyric sections were detected.");
  };

  const createStoryBriefWithGemma = async () => {
    syncStoryLayerFromInputs();
    const progress = createStoryboardProgressWindow("Story Brief Gemma");
    try {
      progress.set("Creating compact song story brief from lyrics, sections, and your story arc...", 18);
      const data = await postJson("/vrgdg/storyboard/story_brief", {
        ...(state.gemmaSettings || {}),
        story_layer: normalizeStoryLayer(state.storyLayer),
        lyrics: lyricsForStoryBrief(),
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        unload_after: true,
        max_new_tokens: 800,
      }, 240000);
      state.storyLayer.song_story_brief = String(data.story_brief || "").trim();
      songStoryBriefInput.value = state.storyLayer.song_story_brief;
      syncStoryLayerFromInputs({ notify: true });
      progress.set("Story brief saved into the Story Layer.", 100);
      progress.close(1600);
      createToast("Story brief created.");
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      createToast(`Story brief failed:\n${String(error?.message || error)}`, true);
    }
  };

  const createStoryArcWithGemma = async () => {
    syncStoryLayerFromInputs();
    const progress = createStoryboardProgressWindow("Story Arc Gemma");
    try {
      progress.set("Creating a short song-structure story arc from lyrics, subjects, and locations...", 18);
      const data = await postJson("/vrgdg/storyboard/story_arc", {
        ...(state.gemmaSettings || {}),
        story_layer: normalizeStoryLayer(state.storyLayer),
        storyboard: slimStoryboardForRequest(state),
        story_idea: userStoryArcInput.value,
        lyrics: lyricsForStoryBrief(),
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        reference_builder: state.referenceBuilder || {},
        camera_flow: state.cameraFlow || "",
        camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
        character_motion: storyboardSpeedValue(state.characterMotionSpeed, 4),
        character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
        performance_style: state.performanceStyle || "",
        facial_performance: state.facialPerformance || "",
        facial_performance_custom: state.facialPerformanceCustom || "",
        unload_after: true,
        max_new_tokens: 900,
      }, 240000);
      state.storyLayer.user_story_arc = String(data.story_arc || "").trim();
      userStoryArcInput.value = state.storyLayer.user_story_arc;
      syncStoryLayerFromInputs({ notify: true });
      progress.set("Story arc saved into the Story Layer.", 100);
      progress.close(1600);
      createToast("Story arc created.");
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      createToast(`Story arc failed:\n${String(error?.message || error)}`, true);
    }
  };

  const sceneBeatGemmaPayload = (scene, overrides = {}) => ({
    ...(state.gemmaSettings || {}),
    ...overrides,
    story_layer: normalizeStoryLayer(state.storyLayer),
    storyboard_payload: storyboardGptPayload(state, [scene]),
    max_new_tokens: 360,
    temperature: 0.35,
    top_p: 0.90,
  });

  const createSceneBeatWithGemma = async (scene, { quiet = false, unloadAfter = true, previousBeat = "", nextLyrics = "", progress = null, progressPercent = 35, progressLabel = "" } = {}) => {
    syncStoryLayerFromInputs();
    const normalized = normalizeScene(scene, 0);
    try {
      progress?.set(`${progressLabel || normalized.label || "Scene"}: creating scene story beat...`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/scene_story_beat", sceneBeatGemmaPayload(scene, {
        unload_after: unloadAfter,
        previous_beat: previousBeat,
        next_lyrics: nextLyrics,
      }), 240000);
      scene.story_beat = String(data.story_beat || "").trim();
      if (!scene.story_beat) throw new Error("Gemma returned an empty scene story beat.");
      if (!quiet) createToast(`Scene story beat created for ${normalized.label || "scene"}.`);
      return scene.story_beat;
    } catch (error) {
      if (!quiet) createToast(`Scene story beat failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  };

  const createAllSceneBeatsWithGemma = async ({ overwrite = false } = {}) => {
    syncStoryLayerFromInputs();
    const scenes = currentRows().filter((scene) => overwrite || !String(scene.story_beat || "").trim());
    if (!scenes.length) {
      createToast(overwrite ? "No scenes found." : "No scene story beats are missing.");
      return;
    }
    const progress = createStoryboardProgressWindow(overwrite ? "Replace Scene Beats" : "Create Missing Scene Beats");
    let created = 0;
    try {
      progress.set(`Creating ${scenes.length} scene story beat${scenes.length === 1 ? "" : "s"}...`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        const scene = scenes[index];
        const allIndex = state.scenes.findIndex((item) => item.id === scene.id);
        const previousBeat = allIndex > 0 ? String(state.scenes[allIndex - 1]?.story_beat || "") : "";
        const nextLyrics = allIndex >= 0 && allIndex < state.scenes.length - 1 ? String(state.scenes[allIndex + 1]?.lyrics || "") : "";
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        await createSceneBeatWithGemma(scene, {
          quiet: true,
          unloadAfter: index === scenes.length - 1,
          previousBeat,
          nextLyrics,
          progress,
          progressPercent: base,
          progressLabel: `Scene Beat ${index + 1}/${scenes.length}: ${scene.label || `Scene ${scene.scene_number || index + 1}`}`,
        });
        created += 1;
      }
      progress.set("Saving story beats...", 96);
      await saveStoryboard();
      progress.set(`Scene beats complete.\nCreated ${created} story beat${created === 1 ? "" : "s"}.`, 100);
      progress.close(1600);
      createToast(`Created ${created} scene story beat${created === 1 ? "" : "s"}.`);
    } catch (error) {
      progress.set(`Scene beats stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, 100);
      createToast(`Scene beats stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, true);
    }
  };

  const planIdLoraDialogueScenesWithGemma = async () => {
    if (!isIdLoraMode) return;
    syncStoryLayerFromInputs();
    const sceneCount = Math.max(1, Math.min(24, Number(idLoraDialogueSceneCount.value || 6)));
    idLoraDialogueSceneCount.value = String(sceneCount);
    const progress = createStoryboardProgressWindow("ID-LoRA Dialogue Scenes");
    try {
      progress.set(`Planning ${sceneCount} ID-LoRA dialogue scene${sceneCount === 1 ? "" : "s"} with ${promptRunnerName()}...`, 8);
      const data = await postJson("/vrgdg/storyboard/id_lora_dialogue_scenes", {
        ...(state.gemmaSettings || {}),
        story_source: [userStoryArcInput.value, songStoryBriefInput.value].map((item) => String(item || "").trim()).filter(Boolean).join("\n\n"),
        story_layer: normalizeStoryLayer(state.storyLayer),
        reference_builder: state.referenceBuilder || {},
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        storyboard: slimStoryboardForRequest(state),
        scene_count: sceneCount,
        video_prompt_type: "id_lora",
        performance_mode: "speaking",
        unload_after: true,
        max_new_tokens: Math.max(2200, sceneCount * 520),
        temperature: 0.55,
        top_p: 0.92,
      }, 240000);
      const generated = Array.isArray(data.scenes) ? data.scenes : [];
      if (!generated.length) throw new Error("Gemma returned no dialogue scenes.");
      state.scenes = generated.map((scene, index) => {
        const normalized = normalizeScene({ ...scene, video_prompt_type: "id_lora", performance_mode: "speaking" }, index);
        normalized.id_lora_character_id = scene.id_lora_character_id || scene.character_id || scene.subject_id || "";
        normalized.id_lora_location_id = scene.id_lora_location_id || scene.location_id || "";
        return normalized;
      });
      state.selected.clear();
      if (String(data.premise || "").trim() && !String(songStoryBriefInput.value || "").trim()) {
        state.storyLayer.song_story_brief = String(data.premise || "").trim();
        songStoryBriefInput.value = state.storyLayer.song_story_brief;
      }
      setMode("storyboard_prompts");
      renderTable();
      refreshSetupPanelSummaries();
      progress.set(`Dialogue scene plan ready.\nCreated ${state.scenes.length} preview scene${state.scenes.length === 1 ? "" : "s"}.`, 96);
      await saveStoryboard();
      progress.set(`Dialogue scene plan saved for review.\nNext step will apply this plan to Video Builder.`, 100);
      progress.close(1800);
      createToast(`ID-LoRA dialogue plan created with ${state.scenes.length} preview scene${state.scenes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      progress.set(`ID-LoRA dialogue planning failed:\n${String(error?.message || error)}`, 100);
      createToast(`ID-LoRA dialogue planning failed:\n${String(error?.message || error)}`, true);
    }
  };

  const applyIdLoraDialoguePlanToVideoBuilder = async () => {
    if (!isIdLoraMode || !state.onApplyIdLoraDialoguePlan) return;
    const scenes = state.scenes
      .map((scene, index) => slimSceneForRequest(scene, index))
      .filter((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim());
    if (!scenes.length) {
      createToast("No ID-LoRA dialogue plan scenes found.", true);
      return;
    }
    const confirmed = window.confirm(`Apply ${scenes.length} ID-LoRA dialogue scene${scenes.length === 1 ? "" : "s"} to Video Builder?\n\nThis will replace the blank starter timeline scene when the project is still empty. If real scenes already exist, Video Builder will ask/guard before replacing.`);
    if (!confirmed) return;
    try {
      applyDialoguePlanButton.disabled = true;
      const result = await state.onApplyIdLoraDialoguePlan({
        story_layer: normalizeStoryLayer(state.storyLayer),
        scenes,
      });
      createToast(result?.message || `Applied ${scenes.length} ID-LoRA dialogue scene${scenes.length === 1 ? "" : "s"} to Video Builder.`);
    } catch (error) {
      createToast(`Apply Dialogue Plan failed:\n${String(error?.message || error)}`, true);
    } finally {
      applyDialoguePlanButton.disabled = false;
    }
  };

  const applyCameraFlow = ({ overwrite = false } = {}) => {
    if (state.mode !== "image_to_video_prep") {
      createToast("Auto camera flow is only available in Video Prep.");
      return;
    }
    const profileKey = state.cameraFlow || "balanced";
    if (profileKey === "off") {
      createToast("Auto camera flow is off.");
      return;
    }
    let previousMotion = "";
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const entry = cameraFlowEntryForScene(profileKey, index, previousMotion);
      if (!entry) return;
      const hadShot = Boolean(String(scene.shot_type || "").trim());
      const hadCamera = Boolean(String(scene.camera_motion || "").trim());
      if ((overwrite || !hadShot) && entry.shot) {
        scene.shot_type = entry.shot;
        changed += 1;
      }
      if ((overwrite || !hadCamera) && entry.camera) {
        scene.camera_motion = entry.camera;
        changed += 1;
      }
      previousMotion = String(scene.camera_motion || entry.camera || previousMotion);
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Auto camera flow replaced ${changed} field${changed === 1 ? "" : "s"}.` : "No camera fields were changed.");
    } else {
      createToast(changed ? `Auto camera flow filled ${changed} blank field${changed === 1 ? "" : "s"}.` : "No blank shot or camera fields needed filling.");
    }
  };

  const applyImageShotFlow = ({ overwrite = false } = {}) => {
    if (state.mode === "image_to_video_prep") {
      createToast("Still shot flow is only available in Image Prep.");
      return;
    }
    const profileKey = state.imageShotFlow || "intimate";
    if (profileKey === "off") {
      createToast("Still shot flow is off.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const sequence = imageShotFlowPresetForMode(profileKey).sequence || [];
      const shot = sequence[index % sequence.length] || "";
      if (!shot) return;
      if (!overwrite && String(scene.shot_type || "").trim()) return;
      scene.shot_type = shot;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Still shot flow replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No shot fields were changed.");
    } else {
      createToast(changed ? `Still shot flow filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank shot fields needed filling.");
    }
  };

  const applyImageAesthetic = ({ overwrite = false } = {}) => {
    if (state.mode === "image_to_video_prep") {
      createToast("Image aesthetic is only available in Image Prep.");
      return;
    }
    const preset = imageAestheticPresetForMode(state.imageAesthetic);
    const value = String(preset.description || "").trim();
    if (!value) {
      createToast("Choose an image aesthetic first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      const existing = String(scene.motion_summary || "");
      const hasAesthetic = existing.split(/\r?\n/).some((line) => line.trim().toLowerCase().startsWith("image aesthetic:"));
      if (!overwrite && hasAesthetic) return;
      scene.motion_summary = replaceLabeledPlanningLine(existing, "Image aesthetic", value);
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Image aesthetic replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No image aesthetic notes were changed.");
    } else {
      createToast(changed ? `Image aesthetic filled ${changed} scene${changed === 1 ? "" : "s"}.` : "No blank image aesthetic notes needed filling.");
    }
  };

  const applyPerformanceStyle = ({ overwrite = false } = {}) => {
    const value = String(state.performanceStyle || "").trim();
    if (!value) {
      createToast(isIdLoraMode ? "Choose a global acting style first." : "Choose a global performance style first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      if (!overwrite && String(scene.performance_style || "").trim()) return;
      scene.performance_style = value;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `${isIdLoraMode ? "Acting" : "Performance"} style replaced ${changed} scene${changed === 1 ? "" : "s"}.` : `No ${isIdLoraMode ? "acting" : "performance"} style fields were changed.`);
    } else {
      createToast(changed ? `${isIdLoraMode ? "Acting" : "Performance"} style filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : `No blank ${isIdLoraMode ? "acting" : "performance"} style fields needed filling.`);
    }
  };

  const applyFacialPerformance = ({ overwrite = false } = {}) => {
    const value = String(state.facialPerformance || "").trim();
    const custom = String(state.facialPerformanceCustom || "").trim();
    if (!value && !custom) {
      createToast("Choose a global facial performance preset or enter custom facial text first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      const hasPreset = String(scene.facial_performance || "").trim();
      const hasCustom = String(scene.facial_performance_custom || "").trim();
      if (!overwrite && (hasPreset || hasCustom)) return;
      scene.facial_performance = value;
      scene.facial_performance_custom = custom;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Facial performance replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No facial performance fields were changed.");
    } else {
      createToast(changed ? `Facial performance filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank facial performance fields needed filling.");
    }
  };

  const confirmClearStoryboardPrompts = () => new Promise((resolve) => {
    const confirmBackdrop = document.createElement("div");
    confirmBackdrop.style.cssText = "position:fixed;inset:0;z-index:100040;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:22px;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(620px,calc(100vw - 44px));border:1px solid #991b1b;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "padding:14px 16px;background:#3f0808;border-bottom:1px solid #991b1b;font-weight:900;color:#fecaca;";
    header.textContent = "Clear all Storyboard prompts and notes?";
    const body = document.createElement("div");
    body.style.cssText = "padding:16px;line-height:1.45;color:#e2e8f0;font-size:13px;";
    body.textContent = "This clears prompt summaries, generated image/video prompts, and extra notes inside every scene card. It keeps lyrics, subjects, locations, reference images, shot type, camera motion, character motion, performance style, and microphone settings.";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 16px 16px;";
    const cancel = makeButton("Cancel");
    const clear = makeButton("Yes, clear prompts", "primary");
    clear.style.borderColor = "#991b1b";
    clear.style.background = "#991b1b";
    actions.append(cancel, clear);
    panel.append(header, body, actions);
    confirmBackdrop.append(panel);
    document.body.append(confirmBackdrop);
    const closeConfirm = (value) => {
      confirmBackdrop.remove();
      resolve(value);
    };
    cancel.onclick = () => closeConfirm(false);
    clear.onclick = () => closeConfirm(true);
    confirmBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === confirmBackdrop) closeConfirm(false);
    });
  });

  const clearAllStoryboardPrompts = async () => {
    const confirmed = await confirmClearStoryboardPrompts();
    if (!confirmed) return;
    let changed = 0;
    for (const scene of state.scenes) {
      const before = [
        scene.prompt_summary,
        scene.motion_summary,
        scene.image_prompt,
        scene.video_prompt,
        scene.notes,
      ].map((value) => String(value || "")).join("\n");
      scene.prompt_summary = "";
      scene.motion_summary = "";
      scene.image_prompt = "";
      scene.video_prompt = "";
      scene.notes = "";
      if (scene.status && scene.status !== "draft") scene.status = "draft";
      const after = [
        scene.prompt_summary,
        scene.motion_summary,
        scene.image_prompt,
        scene.video_prompt,
        scene.notes,
      ].map((value) => String(value || "")).join("\n");
      if (before !== after) changed += 1;
    }
    renderTable();
    syncReferenceMappingsToVideoCreator();
    if (state.projectFolder) {
      try {
        await postJson("/vrgdg/storyboard/save", {
          project_folder: state.projectFolder,
          storyboard: slimStoryboardForRequest(state),
        });
        createToast(`Cleared prompts/notes in ${changed} scene${changed === 1 ? "" : "s"} and saved Storyboard.`);
      } catch (error) {
        createToast(`Cleared prompts/notes in this session, but could not save Storyboard:\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast(`Cleared prompts/notes in ${changed} scene${changed === 1 ? "" : "s"}. Save the project to keep this change.`);
    }
  };

  const currentRows = () => {
    const q = state.query.trim().toLowerCase();
    if (!q) return state.scenes;
    return state.scenes.filter((scene) => [
      scene.label,
      scene.lyrics,
      scene.lyric_section,
      scene.story_beat,
      scene.prompt_summary,
      scene.motion_summary,
      scene.setting,
      scene.shot_type,
      ...(scene.subjects || []),
    ].join(" ").toLowerCase().includes(q));
  };

  function syncReferenceMappingsToVideoCreator() {
    if (!state.onReferenceMappingsChanged) return;
    state.onReferenceMappingsChanged({
      reference_builder: normalizeReferenceBuilderCatalog(state.referenceBuilder),
      scenes: state.scenes.map((scene) => ({
        id: scene.id,
        scene_number: scene.scene_number,
        no_character_present: Boolean(scene.no_character_present),
        subject_ids: scene.no_character_present ? [] : (Array.isArray(scene.subject_refs) ? scene.subject_refs : []).map((ref) => String(ref?.id || "")).filter(Boolean),
        location_id: String(scene.location_ref?.id || ""),
        trigger: String(scene.trigger_phrase || ""),
        trigger_position: String(scene.trigger_position || "start") === "end" ? "end" : "start",
      })),
    });
  }

  const videoPromptTypeLabel = (type) => {
    if (type === "id_lora") return "ID-LoRA I2V";
    if (type === "t2v") return "T2V";
    if (type === "rtv") return "RTV";
    return "I2V";
  };

  const videoPromptTypeHint = (type) => {
    if (type === "id_lora") {
      return "ID-LoRA uses a scene image plus per-scene dialogue and a character voice sample from the ID-LoRA Ref Builder.";
    }
    if (type === "t2v") {
      return "T2V has no first frame, so choose an opening shot and describe the motion clearly.";
    }
    if (type === "rtv") {
      return "Reference to Video uses subject/location references plus an opening shot and motion direction.";
    }
    return "I2V already has a first frame, so use this mostly for camera movement, framing, and continuity.";
  };

  const chooseStoryboardImageFile = () => new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    document.body.append(input);
    input.onchange = () => {
      const file = input.files?.[0] || null;
      input.remove();
      resolve(file);
    };
    input.click();
  });

  const promptStoryboardReferenceDetails = ({ kind, file, defaultName = "", defaultDescription = "" } = {}) => new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:12px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.58);overflow:hidden;";
    const title = kind === "location" ? "Add Location Reference" : "Add Subject Reference";
    box.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;background:#083344;border-bottom:1px solid #155e75;">
        <div>
          <div style="font-size:18px;font-weight:900;color:#cffafe;">${escapeHtml(title)}</div>
          <div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Name and describe this image so Storyboard Builder and Reference Builder can both use it.</div>
        </div>
      </div>
    `;
    const body = document.createElement("div");
    body.style.cssText = "padding:16px;display:flex;flex-direction:column;gap:12px;";
    const name = makeInput(defaultName || String(file?.name || "").replace(/\.[^.]+$/, ""), kind === "location" ? "Location name" : "Subject name");
    const description = makeTextarea(defaultDescription, kind === "location" ? "Location description..." : "Subject description...", 5);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Use Image", "primary");
    actions.append(cancel, save);
    body.append(
      (() => {
        const preview = document.createElement("div");
        preview.style.cssText = "height:150px;border:1px dashed #155e75;border-radius:10px;background:#07111f center/contain no-repeat;";
        return preview;
      })(),
      (() => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:900;color:#cbd5e1;";
        wrap.append("Name", name);
        return wrap;
      })(),
      (() => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:900;color:#cbd5e1;";
        wrap.append("Description", description);
        return wrap;
      })(),
      actions,
    );
    box.append(body);
    backdrop.append(box);
    document.body.append(backdrop);
    const preview = body.firstChild;
    readStoryboardImageFile(file)
      .then((dataUrl) => {
        preview.style.backgroundImage = `url("${dataUrl}")`;
        save.onclick = () => {
          const cleanName = String(name.value || "").trim();
          if (!cleanName) {
            createToast("Give this reference a name first.", true);
            return;
          }
          backdrop.remove();
          resolve({
            name: cleanName,
            description: String(description.value || "").trim(),
            image: { path: "", data: dataUrl, name: String(file?.name || cleanName) },
          });
        };
      })
      .catch((error) => {
        backdrop.remove();
        createToast(String(error?.message || error), true);
        resolve(null);
      });
    cancel.onclick = () => {
      backdrop.remove();
      resolve(null);
    };
  });

  const upsertStoryboardReference = (kind, reference) => {
    if (!reference) return null;
    const list = kind === "location" ? state.referenceBuilder.locations : state.referenceBuilder.subjects;
    const name = String(reference.name || "").trim();
    const existing = list.find((item) => String(item.name || "").trim().toLowerCase() === name.toLowerCase());
    const merged = {
      ...(existing || {}),
      ...reference,
      id: existing?.id || reference.id || storyboardReferenceId(kind === "location" ? "loc" : "subj", name),
      name,
      description: String(reference.description || existing?.description || ""),
      image: reference.image || existing?.image || { path: "", data: "", name: "" },
    };
    if (existing) {
      Object.assign(existing, merged);
      return existing;
    }
    list.push(merged);
    return merged;
  };

  const addStoryboardReferenceFromFile = async (kind, scene) => {
    const file = await chooseStoryboardImageFile();
    if (!file) return null;
    const details = await promptStoryboardReferenceDetails({ kind, file });
    if (!details) return null;
    let reference = details;
    if (state.projectFolder) {
      try {
        const saved = await postJson("/vrgdg/storyboard/import_reference_image", {
          project_folder: state.projectFolder,
          kind,
          name: details.name,
          description: details.description,
          image_data: details.image?.data || "",
          file_name: details.image?.name || file.name || details.name,
        }, 120000);
        reference = saved.reference || reference;
      } catch (error) {
        createToast(`Could not save this reference image into the project folder. It will stay in this session only.\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast("Save the Video Creator project first if you want imported Storyboard references to persist.", true);
    }
    const ref = upsertStoryboardReference(kind, reference);
    if (!ref || !scene) return ref;
    if (kind === "location") {
      scene.location_ref = ref;
      scene.setting = ref.description || ref.name || scene.setting || "";
    } else {
      const refs = Array.isArray(scene.subject_refs) ? scene.subject_refs.slice() : [];
      if (!refs.some((item) => String(item.id || "") === String(ref.id || ""))) refs.push(ref);
      scene.subject_refs = refs;
      scene.subjects = storyboardSubjectNamesFromRefs(refs);
    }
    syncReferenceMappingsToVideoCreator();
    renderTable();
    createToast(`${kind === "location" ? "Location" : "Subject"} reference added to ${scene.label || `Scene ${scene.scene_number}`}.`);
    return ref;
  };

  const openSceneEditor = (scene) => {
    const isVideoPrepMode = state.mode === "image_to_video_prep";
    const isImagePrepMode = !isVideoPrepMode;
    absorbSceneReferencesIntoCatalog([scene]);
    const editorBackdrop = document.createElement("div");
    editorBackdrop.style.cssText = "position:fixed;inset:0;z-index:100012;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const editor = document.createElement("div");
    editor.style.cssText = "width:min(1420px,calc(100vw - 42px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #0e7490;border-radius:16px;background:linear-gradient(135deg,#07111f,#0f172a 46%,#071827);color:#f8fafc;box-shadow:0 28px 90px rgba(0,0,0,.68);padding:18px;display:flex;flex-direction:column;gap:12px;";
    const label = makeInput(scene.label, "Scene label");
    const lyricSection = makeInput(scene.lyric_section || "", "Verse 1, Chorus, Bridge, Outro...");
    const lyrics = makeTextarea(scene.lyrics, "Lyrics, script, or beat for this scene...", 4);
    const storyBeat = makeTextarea(scene.story_beat || "", "Scene story beat for this scene...", 4);
    const summary = makeTextarea(scene.prompt_summary, "Image prompt summary...", 3);
    const motion = makeTextarea(scene.motion_summary, isImagePrepMode ? "Still photography notes..." : "Motion/video summary...", 3);
    const cameraGroups = isImagePrepMode ? STILL_CAMERA_STYLE_GROUPS : CAMERA_MOTION_GROUPS;
    const cameraMotionOptions = cameraGroups.flatMap((group) => group.options || []);
    const cameraMotionValue = scene.camera_motion || cameraMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const cameraMotionPreset = makeGroupedSelect(cameraGroups, cameraMotionValue);
    const customCameraMotion = makeInput(scene.camera_motion || "", isImagePrepMode ? "Custom still camera style" : "Custom camera motion");
    const characterMotionOptions = CHARACTER_MOTION_GROUPS.flatMap((group) => group.options || []);
    const characterMotionValue = scene.character_motion || characterMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const characterMotionPreset = makeGroupedSelect(CHARACTER_MOTION_GROUPS, characterMotionValue);
    const customCharacterMotion = makeInput(scene.character_motion || "", "Custom character motion");
    const performanceStyle = makeSelect(performanceStylePresets, scene.performance_style || "");
    const facialPerformance = makeSelect(facialPerformancePresets, scene.facial_performance || "");
    const facialPerformanceCustom = makeTextarea(scene.facial_performance_custom || "", "Optional custom facial expression/movement text for this scene...", 3);
    const includeMicLabel = document.createElement("label");
    includeMicLabel.style.cssText = "display:flex;align-items:center;gap:8px;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:9px 10px;font-size:12px;font-weight:900;";
    const includeMic = document.createElement("input");
    includeMic.type = "checkbox";
    includeMic.checked = Boolean(scene.include_microphone);
    includeMicLabel.append(includeMic, document.createTextNode("Include microphone in prompt"));
    const noCharacterLabel = document.createElement("label");
    noCharacterLabel.style.cssText = includeMicLabel.style.cssText;
    const noCharacterInput = document.createElement("input");
    noCharacterInput.type = "checkbox";
    noCharacterInput.checked = Boolean(scene.no_character_present);
    noCharacterLabel.append(noCharacterInput, document.createTextNode("No character present"));
    const videoPromptType = makeSelect([
      { value: "i2v", label: "Image to Video" },
      { value: "id_lora", label: "ID-LoRA I2V" },
      { value: "t2v", label: "Text to Video" },
      { value: "rtv", label: "Reference to Video" },
      { value: "ingredients", label: "Ingredients to Video" },
    ], scene.video_prompt_type || "i2v");
    const subjects = makeInput((scene.subjects || []).join(", "), "Subjects, comma separated");
    const subjectDetails = makeTextarea(
      (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
        .map((subject) => `${subject.name || "Subject"}: ${subject.description || ""}`.trim())
        .filter(Boolean)
        .join("\n\n"),
      "Character descriptions from Reference Builder...",
      4,
    );
    const setting = makeInput(scene.setting || scene.location_ref?.description || scene.location_ref?.name || "", "Location / setting");
    const locationDetails = makeTextarea(
      scene.location_ref
        ? `${scene.location_ref.name || "Location"}: ${scene.location_ref.description || ""}`.trim()
        : "",
      "Location description from Reference Builder...",
      4,
    );
    const shot = makeInput(scene.shot_type, "Shot type");
    const shotPreset = makeSelect([{ value: "", label: "Choose a preset..." }, { value: "__custom__", label: "Custom / keep typed value" }], "__custom__");
    const imagePrompt = makeTextarea(scene.image_prompt, "Full text-to-image prompt...", 7);
    const videoPrompt = makeTextarea(scene.video_prompt, "Full video prompt...", 7);
    const imagePath = makeInput(scene.image_path, "Image path");
    const triggerPhrase = makeInput(scene.trigger_phrase || "", "Optional scene trigger phrase");
    const triggerPosition = makeSelect([
      { value: "start", label: "Add trigger to start" },
      { value: "end", label: "Add trigger to end" },
    ], scene.trigger_position || "start");
    const notes = makeTextarea(scene.notes, "Extra planning notes...", 3);
    const selectedSubjectIds = scene.no_character_present ? [] : (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
      .map((ref) => String(ref?.id || ""))
      .filter(Boolean);
    const subjectSelect = makeMultiSelect(
      state.referenceBuilder.subjects.map((subject) => ({ value: subject.id, label: subject.name })),
      selectedSubjectIds,
    );
    const savedLocationId = String(scene.location_ref?.id || "");
    const locationOptions = [
      { value: "", label: "Unassigned" },
      ...state.referenceBuilder.locations.map((location) => ({ value: location.id, label: location.name })),
    ];
    const locationSelect = makeSelect(locationOptions, savedLocationId);
    const field = (name, control) => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:800;color:#cbd5e1;";
      wrap.textContent = name;
      wrap.append(control);
      return wrap;
    };
    const section = (number, title, content, { collapsible = false, open = false } = {}) => {
      const wrap = collapsible ? document.createElement("details") : document.createElement("section");
      if (collapsible) wrap.open = open;
      wrap.style.cssText = "border:1px solid #1f3b46;border-radius:10px;background:linear-gradient(135deg,rgba(8,51,68,.34),rgba(15,23,42,.9));padding:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03);";
      const heading = collapsible ? document.createElement("summary") : document.createElement("div");
      heading.style.cssText = "display:flex;align-items:center;gap:12px;color:#e2e8f0;font-size:20px;font-weight:900;cursor:pointer;list-style:none;";
      const badge = document.createElement("span");
      badge.textContent = String(number);
      badge.style.cssText = "width:30px;height:30px;border-radius:999px;background:#155e75;color:#cffafe;display:grid;place-items:center;font-size:15px;flex:0 0 auto;";
      const text = document.createElement("span");
      text.textContent = title;
      heading.append(badge, text);
      if (collapsible) {
        const chevron = document.createElement("span");
        chevron.textContent = "⌄";
        chevron.style.cssText = "margin-left:auto;color:#cbd5e1;font-size:22px;";
        heading.append(chevron);
      }
      const body = document.createElement("div");
      body.style.cssText = "margin-top:12px;";
      body.append(content);
      wrap.append(heading, body);
      return wrap;
    };
    const twoCol = () => {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;";
      return grid;
    };
    const threeCol = () => {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px 28px;";
      return grid;
    };
    const iconField = (icon, control) => {
      const row = document.createElement("div");
      row.style.cssText = "display:grid;grid-template-columns:44px 1fr;gap:8px;align-items:center;";
      const ico = document.createElement("div");
      ico.textContent = icon;
      ico.style.cssText = "width:42px;height:42px;border:1px solid #155e75;border-radius:8px;background:#083344;color:#22d3ee;display:grid;place-items:center;font-size:20px;";
      row.append(ico, control);
      return row;
    };
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const videoTypeHint = document.createElement("div");
    videoTypeHint.style.cssText = "grid-column:1/-1;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;font-size:12px;line-height:1.45;padding:9px 10px;";
    const shotPresetField = field("Shot type preset", shotPreset);
    const shotCustomField = field("Custom shot type", shot);
    const cameraMotionField = field(isImagePrepMode ? "Still camera style preset" : "Camera motion preset", cameraMotionPreset);
    const characterMotionField = field("Character motion preset", characterMotionPreset);
    const customCharacterMotionField = field("Custom character motion", customCharacterMotion);
    const performanceStyleField = field("Performance / song style", performanceStyle);
    const facialPerformanceField = field("Facial performance", facialPerformance);
    const facialPerformanceCustomField = field("Custom facial performance", facialPerformanceCustom);
    const imagePathField = field("Image path", imagePath);
    const motionField = field("Motion / video summary", motion);
    const t2iPromptField = field("T2I prompt", imagePrompt);
    if (isVideoPrepMode) {
      grid.append(field("Video prompt type", videoPromptType), field("Setting", setting), videoTypeHint, field("Subjects", subjects), performanceStyleField, facialPerformanceField, facialPerformanceCustomField, includeMicLabel, noCharacterLabel, shotPresetField, shotCustomField, cameraMotionField, characterMotionField, customCharacterMotionField, imagePathField, field("Scene trigger phrase", triggerPhrase), field("Trigger placement", triggerPosition));
    } else {
      grid.append(field("Setting", setting), field("Subjects", subjects), performanceStyleField, facialPerformanceField, facialPerformanceCustomField, includeMicLabel, noCharacterLabel, shotPresetField, shotCustomField, cameraMotionField, field("Scene trigger phrase", triggerPhrase), field("Trigger placement", triggerPosition));
    }
    const referenceGrid = document.createElement("div");
    referenceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;";
    if (state.referenceBuilder.subjects.length || state.referenceBuilder.locations.length) {
      referenceGrid.append(
        field("Reference Builder characters", subjectSelect),
        field("Reference Builder location", locationSelect),
      );
    } else {
      referenceGrid.innerHTML = `<div style="grid-column:1/-1;color:#94a3b8;font-size:12px;">No Reference Builder subjects or locations are available yet. Add them in Reference Builder first, then reopen Storyboard Builder.</div>`;
    }
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;";
    const gemmaBeat = makeButton(`${promptRunnerGenericName()} Story Beat`, "primary");
    const gemma = makeButton("Generate Prompt", "purple");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Save Scene Card", "primary");
    actions.append(cancel, gemmaBeat, gemma, apply);
    const closeEditor = makeButton("×");
    closeEditor.style.cssText += "font-size:26px;line-height:1;width:44px;height:44px;padding:0;border-radius:8px;";
    const header = document.createElement("div");
    header.style.cssText = "display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;";
    const headerIcon = document.createElement("div");
    headerIcon.textContent = "▣";
    headerIcon.style.cssText = "width:54px;height:54px;border-radius:14px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;";
    const headerText = document.createElement("div");
    headerText.innerHTML = `<div style="font-size:28px;font-weight:900;color:#f8fafc;">Edit Scene Card</div><div style="color:#cbd5e1;margin-top:3px;">${isVideoPrepMode ? "Define the details for this scene to generate a rich video prompt." : "Define the details for this scene to generate a rich text-to-image prompt."}</div>`;
    header.append(headerIcon, headerText, closeEditor);

    const basicsGrid = twoCol();
    basicsGrid.append(field("Scene label", label), field("Lyric section", lyricSection), field("Scene / lyrics", lyrics), field("Scene story beat", storyBeat));
    if (isVideoPrepMode) {
      basicsGrid.append(field("Prompt mode", iconField("▣", videoPromptType)), field("Performance / song style", performanceStyle), field("Facial performance", facialPerformance), field("Custom facial performance", facialPerformanceCustom), includeMicLabel, noCharacterLabel, videoTypeHint);
    } else {
      const imagePromptType = makeInput("Text to Image", "Text to Image");
      imagePromptType.readOnly = true;
      basicsGrid.append(field("Image prompt type", iconField("▣", imagePromptType)), field("Performance / song style", performanceStyle), field("Facial performance", facialPerformance), field("Custom facial performance", facialPerformanceCustom), includeMicLabel, noCharacterLabel);
    }

    const addSubject = makeButton("+ Add subject");
    addSubject.style.background = "#0f172a";
    addSubject.style.borderStyle = "dashed";
    const addLocation = makeButton("+ Add location");
    addLocation.style.background = "#0f172a";
    addLocation.style.borderStyle = "dashed";
    const subjectChip = document.createElement("div");
    const locationChip = document.createElement("div");
    const refreshReferenceChips = () => {
      const selectedSubjects = Array.from(subjectSelect.selectedOptions).map((option) => state.referenceBuilder.subjects.find((subject) => subject.id === option.value)).filter(Boolean);
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
      subjectChip.innerHTML = noCharacterInput.checked
        ? `<span style="color:#fca5a5;">No character present</span>`
        : selectedSubjects.length
        ? selectedSubjects.map((ref) => referenceChipHtml(ref, "Subject")).join("")
        : `<span style="color:#94a3b8;">No subject selected</span>`;
      locationChip.innerHTML = selectedLocation
        ? referenceChipHtml(selectedLocation, "Location")
        : `<span style="color:#94a3b8;">No location selected</span>`;
    };
    const refreshNoCharacterState = () => {
      subjectSelect.disabled = Boolean(noCharacterInput.checked);
      subjects.disabled = Boolean(noCharacterInput.checked);
      subjectDetails.disabled = Boolean(noCharacterInput.checked);
      if (noCharacterInput.checked) {
        for (const option of subjectSelect.options) option.selected = false;
        subjects.value = "";
        subjectDetails.value = "";
      }
      refreshReferenceChips();
    };
    const referencesGrid = twoCol();
    const subjectPick = document.createElement("div");
    subjectPick.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;";
    subjectPick.append(field("Subject(s)", subjectChip), addSubject);
    const locationPick = document.createElement("div");
    locationPick.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;";
    locationPick.append(field("Setting / Location", locationChip), addLocation);
    referencesGrid.append(subjectPick, locationPick, ...Array.from(referenceGrid.children));
    refreshReferenceChips();

    const motionGrid = isVideoPrepMode ? threeCol() : twoCol();
    if (isVideoPrepMode) {
      motionGrid.append(
        field("Starting shot preset", iconField("▣", shotPreset)),
        field("Camera motion preset", iconField("▣", cameraMotionPreset)),
        field("Character motion preset", iconField("♟", characterMotionPreset)),
        field("Custom starting shot (optional)", shot),
        field("Custom camera motion (optional)", customCameraMotion),
        field("Custom character motion (optional)", customCharacterMotion),
      );
    } else {
      motionGrid.append(
        field("Shot / composition preset", iconField("▣", shotPreset)),
        field("Still camera / photography preset", iconField("▣", cameraMotionPreset)),
        field("Custom shot / composition (optional)", shot),
        field("Custom still camera style (optional)", customCameraMotion),
      );
    }

    const advancedGrid = twoCol();
    if (isVideoPrepMode) {
      advancedGrid.append(field("Prompt summary", summary), field("Motion / video prompt summary", motion), field("Character details", subjectDetails), field("Location details", locationDetails), imagePathField, t2iPromptField, field("Video prompt", videoPrompt));
    } else {
      advancedGrid.append(t2iPromptField, field("Character details", subjectDetails), field("Location details", locationDetails), field("Still photography notes", motion));
    }
    const notesWrap = document.createElement("div");
    notesWrap.append(notes);
    editor.replaceChildren(
      header,
      section(1, "Scene Basics", basicsGrid),
      section(2, "References", referencesGrid),
      section(3, isVideoPrepMode ? "Camera & Motion" : "Shot & Still Camera", motionGrid),
      section(4, "Advanced Options", advancedGrid, { collapsible: true, open: false }),
      section(5, "Notes", notesWrap),
      actions,
    );
    editorBackdrop.append(editor);
    document.body.append(editorBackdrop);
    closeEditor.onclick = () => editorBackdrop.remove();
    const refreshShotPresetForVideoType = () => {
      const type = videoPromptType.value || "i2v";
      const options = isImagePrepMode ? IMAGE_SHOT_TYPES : (type === "i2v" ? VIDEO_SHOT_TYPES : Array.from(new Set([...IMAGE_SHOT_TYPES, ...VIDEO_SHOT_TYPES])));
      const current = shot.value || scene.shot_type || "";
      shotPreset.replaceChildren();
      for (const option of [
        { value: "", label: isImagePrepMode ? "Choose shot / composition preset..." : (type === "i2v" ? "Choose camera/motion preset..." : "Choose starting shot preset...") },
        ...options.map((item) => ({ value: item, label: item })),
        { value: "__custom__", label: "Custom / keep typed value" },
      ]) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        shotPreset.append(item);
      }
      shotPreset.value = options.includes(current) ? current : "__custom__";
      shotPresetField.firstChild.textContent = isImagePrepMode ? "Shot / composition preset" : (type === "i2v" ? "Camera / motion preset" : "Starting shot preset");
      shotCustomField.firstChild.textContent = isImagePrepMode ? "Custom shot / composition" : (type === "i2v" ? "Custom camera / motion" : "Custom starting shot");
      videoTypeHint.textContent = videoPromptTypeHint(type);
      motionField.firstChild.textContent = isImagePrepMode
        ? "Still photography notes"
        : type === "i2v"
          ? "Motion / camera direction"
          : type === "rtv"
            ? "Motion / camera direction with references"
            : "Motion / camera direction";
      t2iPromptField.style.display = isImagePrepMode || (type !== "t2v" && type !== "rtv") ? "flex" : "none";
      imagePathField.style.display = isVideoPrepMode && type !== "t2v" && type !== "rtv" ? "flex" : "none";
      videoPrompt.style.display = isVideoPrepMode ? "" : "none";
      videoPrompt.placeholder = type === "t2v"
        ? "Full text-to-video prompt..."
        : type === "rtv"
          ? "Full reference-to-video prompt..."
          : "Full image-to-video prompt...";
    };
    refreshShotPresetForVideoType();
    videoPromptType.addEventListener("change", refreshShotPresetForVideoType);
    const refreshSubjectDetailsFromSelection = () => {
      const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => option.value).filter(Boolean);
      const selectedSubjects = selectedIds
        .map((id) => state.referenceBuilder.subjects.find((subject) => subject.id === id))
        .filter(Boolean);
      subjectDetails.value = selectedSubjects
        .map((subject) => `${subject.name || "Subject"}: ${subject.description || ""}`.trim())
        .filter(Boolean)
        .join("\n\n");
    };
    subjectSelect.addEventListener("change", refreshSubjectDetailsFromSelection);
    subjectSelect.addEventListener("change", refreshReferenceChips);
    noCharacterInput.addEventListener("change", refreshNoCharacterState);
    refreshNoCharacterState();
    shotPreset.addEventListener("change", () => {
      if (shotPreset.value && shotPreset.value !== "__custom__") shot.value = shotPreset.value;
    });
    cameraMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(cameraMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCameraMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      motion.value = replaceLabeledPlanningLine(currentMotion, isImagePrepMode ? "Still camera style" : "Camera motion", selectedMotion);
    });
    characterMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(characterMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCharacterMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      motion.value = replaceLabeledPlanningLine(currentMotion, "Character motion", selectedMotion);
    });
    locationSelect.addEventListener("change", () => {
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
      if (selectedLocation) {
        setting.value = selectedLocation.description || selectedLocation.name || "";
        locationDetails.value = `${selectedLocation.name || "Location"}: ${selectedLocation.description || ""}`.trim();
      } else {
        locationDetails.value = "";
      }
      refreshReferenceChips();
    });
    addSubject.onclick = async () => {
      saveEditorFieldsToScene();
      const ref = await addStoryboardReferenceFromFile("subject", scene);
      if (!ref) return;
      let option = Array.from(subjectSelect.options).find((item) => item.value === ref.id);
      if (!option) {
        option = document.createElement("option");
        option.value = ref.id;
        option.textContent = ref.name;
        subjectSelect.append(option);
      }
      option.selected = true;
      refreshSubjectDetailsFromSelection();
      refreshReferenceChips();
    };
    addLocation.onclick = async () => {
      saveEditorFieldsToScene();
      const ref = await addStoryboardReferenceFromFile("location", scene);
      if (!ref) return;
      let option = Array.from(locationSelect.options).find((item) => item.value === ref.id);
      if (!option) {
        option = document.createElement("option");
        option.value = ref.id;
        option.textContent = ref.name;
        locationSelect.append(option);
      }
      locationSelect.value = ref.id;
      setting.value = ref.description || ref.name || "";
      refreshReferenceChips();
    };
    const saveEditorFieldsToScene = () => {
      scene.label = label.value.trim() || scene.label;
      scene.lyric_section = lyricSection.value.trim();
      scene.lyrics = lyrics.value.trim();
      scene.story_beat = storyBeat.value.trim();
      scene.prompt_summary = isVideoPrepMode ? summary.value.trim() : "";
      scene.motion_summary = motion.value.trim();
      scene.video_prompt_type = isVideoPrepMode ? (videoPromptType.value || "i2v") : "i2v";
      scene.no_character_present = Boolean(noCharacterInput.checked);
      scene.subjects = scene.no_character_present ? [] : subjects.value.split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
      scene.setting = setting.value.trim();
      if (state.referenceBuilder.subjects.length && !scene.no_character_present) {
        const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => option.value).filter(Boolean);
        scene.subject_refs = selectedIds
          .map((id) => state.referenceBuilder.subjects.find((subject) => subject.id === id))
          .filter(Boolean);
        const detailsByName = new Map(
          subjectDetails.value
            .split(/\n{2,}/)
            .map((block) => {
              const parts = block.split(":");
              const name = String(parts.shift() || "").trim();
              const description = parts.join(":").trim();
              return name ? [name.toLowerCase(), description] : null;
            })
            .filter(Boolean)
        );
        scene.subject_refs = scene.subject_refs.map((subject) => ({
          ...subject,
          description: detailsByName.get(String(subject.name || "").toLowerCase()) ?? subject.description,
        }));
        if (scene.subject_refs.length) {
          scene.subjects = storyboardSubjectNamesFromRefs(scene.subject_refs);
        }
      } else if (scene.no_character_present) {
        scene.subject_refs = [];
      }
      if (state.referenceBuilder.locations.length) {
        const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
        const locationParts = String(locationDetails.value || "").split(":");
        const locationName = String(locationParts.shift() || "").trim();
        const locationDescription = locationParts.join(":").trim();
        scene.location_ref = selectedLocation
          ? {
              ...selectedLocation,
              name: locationName || selectedLocation.name,
              description: locationDescription || selectedLocation.description || "",
            }
          : null;
        if (selectedLocation) scene.setting = selectedLocation.description || selectedLocation.name || scene.setting;
        if (scene.location_ref) scene.setting = scene.location_ref.description || scene.location_ref.name || scene.setting;
      }
      scene.shot_type = shot.value.trim();
      scene.camera_motion = customCameraMotion.value.trim() || cameraMotionPreset.value.trim();
      scene.character_motion = isVideoPrepMode ? (customCharacterMotion.value.trim() || characterMotionPreset.value.trim()) : "";
      scene.performance_style = performanceStyle.value || "";
      scene.facial_performance = facialPerformance.value || "";
      scene.facial_performance_custom = facialPerformanceCustom.value.trim();
      scene.include_microphone = Boolean(includeMic.checked);
      scene.trigger_phrase = triggerPhrase.value.trim();
      scene.trigger_position = triggerPosition.value === "end" ? "end" : "start";
      scene.image_prompt = imagePrompt.value.trim();
      if (isVideoPrepMode) scene.video_prompt = videoPrompt.value.trim();
      if (isVideoPrepMode) scene.image_path = imagePath.value.trim();
      scene.notes = notes.value.trim();
    };
    cancel.onclick = () => editorBackdrop.remove();
    gemma.onclick = async () => {
      const previous = gemma.textContent;
      gemma.disabled = true;
      const runnerName = promptRunnerName();
      gemma.textContent = `Running ${runnerName}...`;
      const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
      try {
        saveEditorFieldsToScene();
        progress.set(`Preparing ${scene.label || "scene"} for ${runnerName}...`, 12);
        await createScenePromptForActiveMode(scene, { progress, progressPercent: 32 });
        progress.set(state.mode === "image_to_video_prep" ? "Storyboard video prompt ready." : "Storyboard image prompt ready.", 100);
        progress.close(1200);
        imagePrompt.value = scene.image_prompt || "";
        videoPrompt.value = scene.video_prompt || "";
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemma.disabled = false;
        gemma.textContent = previous;
      }
    };
    gemmaBeat.onclick = async () => {
      const previous = gemmaBeat.textContent;
      gemmaBeat.disabled = true;
      gemmaBeat.textContent = "Creating...";
      const progress = createStoryboardProgressWindow("Scene Story Beat");
      try {
        saveEditorFieldsToScene();
        await createSceneBeatWithGemma(scene, { progress, progressPercent: 35 });
        storyBeat.value = scene.story_beat || "";
        progress.set("Scene story beat ready.", 100);
        progress.close(1200);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemmaBeat.disabled = false;
        gemmaBeat.textContent = previous;
      }
    };
    apply.onclick = () => {
      saveEditorFieldsToScene();
      syncReferenceMappingsToVideoCreator();
      syncStoryLayerFromInputs({ notify: true });
      editorBackdrop.remove();
      renderTable();
    };
  };

  function renderTable() {
    const rows = currentRows();
    const mode = state.mode;
    const head = mode === "image_to_video_prep"
      ? ["", "#", "Image", "Scene / Lyrics", "Motion / Video Prompt Summary", "Subjects", "Setting", "Shot Type", "Prompt Status", "Actions"]
      : ["#", "Reference", "Scene / Lyrics", "Prompt Summary", "Subjects", "Setting", "Shot Type", "Prompt Status", "Actions"];
    const table = document.createElement("table");
    table.style.cssText = "width:100%;border-collapse:collapse;min-width:1250px;font-size:13px;";
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>${head.map((item) => `<th style="position:sticky;top:0;background:#111827;border-bottom:1px solid #334155;color:#cffafe;text-align:left;padding:13px;font-weight:900;">${escapeHtml(item)}</th>`).join("")}</tr>`;
    const tbody = document.createElement("tbody");
    for (const scene of rows) {
      const meta = statusMeta(scene);
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid #1e293b";
      tr.style.background = "#0b1220";
      const imageCell = scene.image_path
        ? `<div style="width:170px;height:78px;border-radius:6px;background:#0f172a url('${escapeHtml(makeStoryboardImageUrl(scene.image_path))}') center/cover no-repeat;"></div>`
        : `<div style="width:170px;height:78px;border:1px dashed #334155;border-radius:6px;display:grid;place-items:center;color:#94a3b8;font-size:12px;text-align:center;background:#07111f;">No image in storyboard<br>Optional reference</div>`;
      const sceneActionStyle = "border:1px solid #155e75;border-radius:6px;background:#0f172a;color:#a5f3fc;padding:8px 10px;font-weight:800;cursor:pointer;";
      const sceneGptStyle = "border:1px solid #06b6d4;border-radius:6px;background:#0e7490;color:#f8fafc;padding:8px 10px;font-weight:900;cursor:pointer;";
      const sceneGemmaStyle = "border:1px solid #22c55e;border-radius:6px;background:#166534;color:#f0fdf4;padding:8px 10px;font-weight:900;cursor:pointer;";
      const runnerName = promptRunnerName();
      const gemmaTitle = mode === "image_to_video_prep"
        ? `Create this scene's video prompt with ${runnerName}. If the scene has an image, local vision uses it as guidance.`
        : `Create this scene's text-to-image prompt with ${runnerName}.`;
      const actionHtml = `
        <div style="display:flex;align-items:center;gap:7px;white-space:nowrap;">
          <button data-action="edit" style="${sceneActionStyle}">Open Scene Card</button>
          <button data-action="gemma" style="${sceneGemmaStyle}" title="${escapeHtml(gemmaTitle)}">${escapeHtml(runnerName)}</button>
          <button data-action="gpt" style="${sceneGptStyle}" title="Copy only this scene card as GPT JSON.">GPT</button>
        </div>`;
      const status = `<span style="display:inline-flex;align-items:center;gap:6px;color:${meta.color};font-weight:900;"><span style="width:8px;height:8px;border-radius:999px;background:${meta.color};display:inline-block;"></span>${escapeHtml(meta.label)}</span>`;
      const miniRefButtonStyle = "margin-top:7px;border:1px dashed #155e75;border-radius:6px;background:#07111f;color:#a5f3fc;padding:5px 7px;font-size:11px;font-weight:900;cursor:pointer;";
      const subjectCell = `<div>${subjectRefsHtml(scene)}</div><button data-action="load-subject-ref" title="Load a subject image for this scene" style="${miniRefButtonStyle}">+ Subject</button>`;
      const settingCell = `<div>${settingRefHtml(scene)}</div><button data-action="load-location-ref" title="Load a location image for this scene" style="${miniRefButtonStyle}">+ Location</button>`;
      const videoType = videoPromptTypeLabel(scene.video_prompt_type || "i2v");
      const shotCell = `<div style="display:flex;flex-direction:column;gap:4px;"><span style="align-self:flex-start;border:1px solid #155e75;border-radius:999px;background:#0f172a;color:#a5f3fc;font-size:11px;font-weight:900;padding:2px 7px;">${escapeHtml(videoType)}</span><strong style="color:#f8fafc;">${escapeHtml(scene.shot_type || "-")}</strong></div>`;
      const storyPreview = `${scene.lyric_section ? `<div style="margin-top:5px;color:#67e8f9;font-size:11px;font-weight:900;">${escapeHtml(scene.lyric_section)}</div>` : ""}${scene.story_beat ? `<div style="margin-top:5px;color:#94a3b8;font-size:11px;">Beat: ${escapeHtml(truncate(scene.story_beat, 90))}</div>` : ""}`;
      if (mode === "image_to_video_prep") {
        tr.innerHTML = `
          <td style="padding:13px;"><input type="checkbox" data-action="select" ${state.selected.has(scene.id) ? "checked" : ""}></td>
          <td style="padding:13px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:13px;">${imageCell}</td>
          <td style="padding:13px;max-width:210px;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 95))}</span>${storyPreview}</td>
          <td style="padding:13px;max-width:270px;color:#d4d4d8;">${escapeHtml(truncate(scene.motion_summary || scene.video_prompt, 150))}</td>
          <td style="padding:13px;max-width:230px;">${subjectCell}</td>
          <td style="padding:13px;color:#d4d4d8;max-width:210px;">${settingCell}</td>
          <td style="padding:13px;">${shotCell}</td>
          <td style="padding:13px;">${status}</td>
          <td style="padding:13px;white-space:nowrap;">${actionHtml}</td>
        `;
      } else {
        tr.innerHTML = `
          <td style="padding:13px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:13px;">${imageCell}</td>
          <td style="padding:13px;max-width:220px;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 95))}</span>${storyPreview}</td>
          <td style="padding:13px;max-width:280px;color:#d4d4d8;">${escapeHtml(truncate(scene.prompt_summary || scene.image_prompt, 150))}</td>
          <td style="padding:13px;max-width:230px;">${subjectCell}</td>
          <td style="padding:13px;color:#d4d4d8;max-width:210px;">${settingCell}</td>
          <td style="padding:13px;">${shotCell}</td>
          <td style="padding:13px;">${status}</td>
          <td style="padding:13px;white-space:nowrap;">${actionHtml}</td>
        `;
      }
      tr.querySelector('[data-action="edit"]')?.addEventListener("click", () => openSceneEditor(scene));
      tr.querySelector('[data-action="load-subject-ref"]')?.addEventListener("click", () => addStoryboardReferenceFromFile("subject", scene));
      tr.querySelector('[data-action="load-location-ref"]')?.addEventListener("click", () => addStoryboardReferenceFromFile("location", scene));
      tr.querySelector('[data-action="gemma"]')?.addEventListener("click", async () => {
        const runnerName = promptRunnerName();
        const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
        try {
          progress.set(`Preparing ${scene.label || "scene"} for ${runnerName}...`, 12);
          await createScenePromptForActiveMode(scene, { progress, progressPercent: 32 });
          progress.set(state.mode === "image_to_video_prep" ? "Storyboard video prompt ready." : "Storyboard image prompt ready.", 100);
          progress.close(1200);
        } catch (error) {
          progress.set(`Error:\n${String(error?.message || error)}`, 100);
        }
      });
      tr.querySelector('[data-action="gpt"]')?.addEventListener("click", () => copySceneForGpt(scene));
      tr.querySelector('[data-action="select"]')?.addEventListener("change", (event) => {
        if (event.target.checked) state.selected.add(scene.id);
        else state.selected.delete(scene.id);
        renderTable();
      });
      tbody.append(tr);
    }
    table.append(thead, tbody);
    tableWrap.replaceChildren(table);
    const readyCount = state.scenes.filter((scene) => String(scene.image_prompt || scene.video_prompt || "").trim()).length;
    const imageCount = state.scenes.filter((scene) => String(scene.image_path || "").trim()).length;
    stats.textContent = `${state.scenes.length} scenes  |  ${imageCount} images linked  |  ${readyCount} scenes with prompts  |  ${state.selected.size} selected`;
    refreshSetupPanelSummaries();
  }

  async function loadExisting() {
    if (!state.projectFolder) {
      renderTable();
      return;
    }
    try {
      const incomingScenes = state.scenes.map((scene) => normalizeScene(scene));
      const data = await postJson("/vrgdg/storyboard/load", { project_folder: state.projectFolder });
      const saved = data.storyboard || {};
      const savedReferences = normalizeReferenceBuilderCatalog(saved.reference_builder || saved.referenceBuilder || {});
      const currentHasSubjects = Array.isArray(state.referenceBuilder?.subjects) && state.referenceBuilder.subjects.length > 0;
      const currentHasLocations = Array.isArray(state.referenceBuilder?.locations) && state.referenceBuilder.locations.length > 0;
      const currentLocationsCleared = Boolean(state.referenceBuilder?.locations_cleared);
      if ((!currentHasSubjects && savedReferences.subjects.length) || (!currentHasLocations && !currentLocationsCleared && savedReferences.locations.length)) {
        const nextReferences = {
          subjects: currentHasSubjects ? state.referenceBuilder.subjects : savedReferences.subjects,
          locations: currentLocationsCleared ? [] : (currentHasLocations ? state.referenceBuilder.locations : savedReferences.locations),
          locations_cleared: currentLocationsCleared,
        };
        state.referenceBuilder = normalizeReferenceBuilderCatalog(nextReferences);
      } else if (!currentHasSubjects && !currentHasLocations && !currentLocationsCleared && (savedReferences.subjects.length || savedReferences.locations.length)) {
        state.referenceBuilder = mergeReferenceBuilderCatalog(state.referenceBuilder, savedReferences);
      }
      if (Array.isArray(saved.scenes) && saved.scenes.length) {
        const savedScenes = saved.scenes.map((scene, index) => normalizeScene(scene, index));
        const scenesToShow = incomingScenes.length ? incomingScenes : savedScenes;
        state.scenes = scenesToShow.map((fresh, index) => {
          const normalized = savedScenes.find((item) => item.id === fresh.id)
            || savedScenes.find((item) => Number(item.scene_number) === Number(fresh.scene_number))
            || null;
          if (!normalized) return normalizeScene(fresh, index);
          const subjectRefs = incomingScenes.length ? (fresh.subject_refs || []) : (fresh.subject_refs?.length ? fresh.subject_refs : normalized.subject_refs);
          const subjects = subjectRefs?.length
            ? storyboardSubjectNamesFromRefs(subjectRefs)
            : Array.from(new Set([
              ...(fresh.subjects || []),
              ...(normalized.subjects || []),
            ].map((item) => String(item || "").trim()).filter(Boolean)));
          return {
            ...normalized,
            id: fresh.id || normalized.id,
            scene_number: fresh.scene_number || normalized.scene_number,
            label: fresh.label || normalized.label,
            video_prompt_type: payloadVideoPromptType || fresh.video_prompt_type || normalized.video_prompt_type,
            lyrics: fresh.lyrics || normalized.lyrics,
            lyric_section: fresh.lyric_section || normalized.lyric_section,
            story_beat: fresh.story_beat || normalized.story_beat,
            performance_mode: fresh.performance_mode || normalized.performance_mode || state.performanceMode,
            prompt_summary: state.mode === "image_to_video_prep" ? (fresh.prompt_summary || normalized.prompt_summary) : "",
            motion_summary: fresh.motion_summary || normalized.motion_summary,
            image_path: fresh.image_path || normalized.image_path,
            no_character_present: Boolean(fresh.no_character_present || normalized.no_character_present),
            subjects,
            subject_refs: fresh.no_character_present || normalized.no_character_present ? [] : subjectRefs,
            setting: currentLocationsCleared ? "" : (fresh.location_ref?.name || normalized.setting || fresh.setting),
            location_ref: currentLocationsCleared ? null : (incomingScenes.length ? fresh.location_ref : (fresh.location_ref || normalized.location_ref)),
          };
        });
        if (currentLocationsCleared) {
          state.scenes.forEach((scene) => {
            scene.location_ref = null;
            scene.setting = "";
          });
        }
        absorbSceneReferencesIntoCatalog(state.scenes);
      }
      state.mode = saved.mode || state.mode;
      state.performanceMode = normalizeStoryboardPerformanceMode(saved.performance_mode || saved.performanceMode || state.performanceMode);
      if (saved.camera_flow && STORYBOARD_CAMERA_FLOW_PRESETS[saved.camera_flow]) {
        state.cameraFlow = saved.camera_flow;
        cameraFlowSelect.value = state.cameraFlow;
      }
      if (saved.image_shot_flow && imageShotFlowPresets[saved.image_shot_flow]) {
        state.imageShotFlow = saved.image_shot_flow;
        imageShotSelect.value = state.imageShotFlow;
      }
      state.imageAesthetic = String(saved.image_aesthetic || saved.imageAesthetic || state.imageAesthetic || "");
      if (!imageAestheticPresets.some((preset) => preset.value === state.imageAesthetic)) state.imageAesthetic = imageAestheticPresets[0]?.value || "";
      imageAestheticSelect.value = state.imageAesthetic;
      state.globalConsistencyPhrase = String(saved.global_consistency_phrase || saved.globalConsistencyPhrase || state.globalConsistencyPhrase || "");
      consistencyInput.value = state.globalConsistencyPhrase;
      state.performanceStyle = String(saved.performance_style_default || saved.performance_style || state.performanceStyle || "");
      if (!performanceStylePresets.some((preset) => preset.value === state.performanceStyle)) state.performanceStyle = performanceStylePresets[0]?.value || "";
      performanceSelect.value = state.performanceStyle;
      state.facialPerformance = String(saved.facial_performance_default || saved.facial_performance || state.facialPerformance || "");
      if (!facialPerformancePresets.some((preset) => preset.value === state.facialPerformance)) state.facialPerformance = facialPerformancePresets[0]?.value || "";
      state.facialPerformanceCustom = String(saved.facial_performance_custom_default || saved.facial_performance_custom || state.facialPerformanceCustom || "");
      facialSelect.value = state.facialPerformance;
      facialCustomInput.value = state.facialPerformanceCustom;
      state.cameraMotionSpeed = storyboardSpeedValue(saved.camera_motion_speed ?? saved.motion_defaults?.camera_motion_speed ?? state.cameraMotionSpeed, 4);
      state.characterMotionSpeed = storyboardSpeedValue(saved.character_motion_speed ?? saved.motion_defaults?.character_motion_speed ?? state.characterMotionSpeed, 4);
      cameraSpeedInput.value = String(state.cameraMotionSpeed);
      characterSpeedInput.value = String(state.characterMotionSpeed);
      state.storyLayer = normalizeStoryLayer(saved.story_layer || saved.storyLayer || {});
      storyLayerEnabledInput.checked = state.storyLayer.enabled !== false;
      userStoryArcInput.value = state.storyLayer.user_story_arc || "";
      songStoryBriefInput.value = state.storyLayer.song_story_brief || "";
      lyricStoryStrengthInput.value = String(state.storyLayer.lyric_story_strength ?? 7);
      syncLyricStoryStrengthLabel();
      refreshCameraFlowInfo();
      refreshImageShotInfo();
      refreshImageAestheticInfo();
      refreshConsistencyInfo();
      refreshCameraSpeedInfo();
      refreshPerformanceInfo();
      refreshCharacterSpeedInfo();
      refreshFacialInfo();
      setMode(state.mode);
      syncReferenceMappingsToVideoCreator();
    } catch (error) {
      createToast(String(error?.message || error), true);
      renderTable();
    }
  }

  async function saveStoryboard() {
    if (!state.projectFolder) {
      createToast("Save the Video Creator project first so Storyboard Builder knows where to write files.", true);
      return;
    }
    state.saving = true;
    save.disabled = true;
    try {
      syncStoryLayerFromInputs();
      state.scenes.forEach((scene) => {
        if (String(scene.video_prompt || "").trim()) scene.video_prompt = enforceStoryboardVideoFacialRequirements(scene.video_prompt, scene);
      });
      const data = await postJson("/vrgdg/storyboard/save", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      syncStoryLayerFromInputs({ notify: true });
      createToast(`Storyboard saved:\n${data.storyboard?.path || ""}`);
    } catch (error) {
      createToast(String(error?.message || error), true);
    } finally {
      save.disabled = false;
      state.saving = false;
    }
  }

  async function exportPromptFiles() {
    if (!state.projectFolder) {
      createToast("Save the Video Creator project first so Storyboard Builder knows where to export prompt files.", true);
      return;
    }
    exportPrompts.disabled = true;
    try {
      state.scenes.forEach((scene) => {
        if (String(scene.image_prompt || "").trim()) scene.image_prompt = ensureStoryboardReferenceOpening(scene.image_prompt, scene, state.imageMode);
        if (String(scene.video_prompt || "").trim()) scene.video_prompt = enforceStoryboardVideoFacialRequirements(scene.video_prompt, scene);
      });
      const data = await postJson("/vrgdg/storyboard/export_prompts", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      if (state.onPromptsExported) {
        state.onPromptsExported({
          ...storyboardDefaultsPayload(),
          story_layer: normalizeStoryLayer(state.storyLayer),
          scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        });
      }
      createToast(`Exported ${data.scene_count || 0} scene prompt rows:\nText:\n${data.t2i_prompts_path}\n${data.i2v_prompts_path}\nJSON:\n${data.t2i_prompts_json_path || ""}\n${data.video_prompts_json_path || ""}`);
    } catch (error) {
      createToast(String(error?.message || error), true);
    } finally {
      exportPrompts.disabled = false;
    }
  }

  async function copyStoryboardForGpt() {
    try {
      const payload = storyboardGptPayload(state);
      const text = JSON.stringify(payload, null, 2);
      await copyTextToClipboard(text);
      openStoryboardGptUrl(payload);
      createToast(`Copied Storyboard GPT JSON for ${payload.scenes.length} scenes and opened GPT.`);
    } catch (error) {
      createToast(`Could not copy Storyboard GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  async function copySceneForGpt(scene) {
    try {
      const normalized = normalizeScene(scene, 0);
      const payload = storyboardGptPayload(state, [scene]);
      const text = JSON.stringify(payload, null, 2);
      await copyTextToClipboard(text);
      openStoryboardGptUrl(payload);
      createToast(`Copied GPT JSON for ${normalized.label || `Scene ${normalized.scene_number}`} and opened GPT.`);
    } catch (error) {
      createToast(`Could not copy scene GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  function imagePromptImportJsonText(rawText) {
    const text = String(rawText || "").trim();
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced) return fenced[1].trim();
    const firstArray = text.indexOf("[");
    const firstObject = text.indexOf("{");
    const starts = [firstArray, firstObject].filter((index) => index >= 0);
    if (!starts.length) return text;
    const start = Math.min(...starts);
    const end = Math.max(text.lastIndexOf("]"), text.lastIndexOf("}"));
    return end > start ? text.slice(start, end + 1).trim() : text.slice(start).trim();
  }

  function parseImagePromptImportJson(rawText) {
    const text = imagePromptImportJsonText(rawText);
    if (!text) return [];
    const data = JSON.parse(text);
    const source = Array.isArray(data)
      ? data
      : Array.isArray(data.prompts)
        ? data.prompts
        : Array.isArray(data.scenes)
          ? data.scenes
          : data && typeof data === "object"
            ? Object.entries(data).map(([key, value]) => {
              if (value && typeof value === "object") return { scene: key, ...value };
              return { scene: key, prompt: value };
            })
            : [];
    const rows = [];
    for (const item of source) {
      if (!item || typeof item !== "object") continue;
      const sceneRaw = item.scene_number ?? item.sceneNumber ?? item.scene ?? item.number ?? item.id ?? "";
      const sceneNumber = Number(String(sceneRaw).match(/\d+/)?.[0] || sceneRaw || 0);
      const prompt = String(
        item.image_prompt
        ?? item.text_to_image_prompt
        ?? item.t2i_prompt
        ?? item.prompt
        ?? item.text
        ?? "",
      ).trim();
      if (!sceneNumber || !prompt) continue;
      rows.push({ sceneNumber, prompt });
    }
    return rows;
  }

  function openImportImagePromptsFromGptModal() {
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100013;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(840px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = `<div style="font-size:17px;font-weight:900;color:#cffafe;">Import Image Prompts From GPT</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Paste the JSON code block from the Krea 2 text-to-image GPT. This updates Image Prep prompts only.</div>`;
    const importClose = makeButton("Close");
    importHeader.append(importTitle, importClose);
    const help = document.createElement("div");
    help.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    help.innerHTML = `Accepted examples:<br><code>[{"scene":1,"image_prompt":"..."},{"scene":2,"prompt":"..."}]</code><br><code>{"scene1":"prompt text","scene2":"prompt text"}</code>`;
    const input = document.createElement("textarea");
    input.placeholder = "Paste GPT JSON output here...";
    input.spellcheck = false;
    input.style.cssText = "min-height:340px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const apply = makeButton("Import Image Prompts", "purple");
    actions.append(cancel, apply);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto;";
    body.append(help, input, status, actions);
    importBox.append(importHeader, body);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    const closeImport = () => importBackdrop.remove();
    importClose.onclick = closeImport;
    cancel.onclick = closeImport;
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) closeImport();
    });
    apply.onclick = () => {
      try {
        const rows = parseImagePromptImportJson(input.value);
        if (!rows.length) throw new Error("No usable image prompts found. Make sure each row has a scene number and image_prompt or prompt.");
        let updated = 0;
        const missing = [];
        for (const row of rows) {
          const scene = state.scenes.find((item) => Number(item.scene_number) === Number(row.sceneNumber));
          if (!scene) {
            missing.push(row.sceneNumber);
            continue;
          }
          scene.image_prompt = row.prompt;
          scene.prompt_summary = "";
          scene.status = "image_prompt_ready";
          updated += 1;
        }
        renderTable();
        status.textContent = `Updated ${updated} Image Prep prompt${updated === 1 ? "" : "s"}${missing.length ? `; missing scenes: ${missing.join(", ")}` : ""}.`;
        status.style.color = updated ? "#67e8f9" : "#fbbf24";
        createToast(`Imported ${updated} image prompt${updated === 1 ? "" : "s"} from GPT.`);
        if (updated) closeImport();
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
        createToast(String(error?.message || error), true);
      }
    };
    input.focus();
  }

  function storyboardGemmaPayload(scene, overrides = {}) {
    const payload = storyboardGptPayload(state, [scene]);
    return {
      ...(state.gemmaSettings || {}),
      ...overrides,
      storyboard_payload: payload,
      max_new_tokens: 2000,
      temperature: 0.35,
      top_p: 0.90,
    };
  }

  async function createSceneImagePromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending image scene card to ${runnerName}...\nThis creates the text-to-image prompt for Image Prep.`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/gemma_image_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter, max_new_tokens: 1200 }), 240000);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: ${genericName} response received.\nRunner: ${data.runner || runnerName}\nSaving image prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const prompt = ensureStoryboardReferenceOpening(applyStoryboardTriggerPhrases(data.prompt, scene), scene, state.imageMode);
      if (!prompt) throw new Error(`${genericName} returned an empty Storyboard image prompt.`);
      scene.image_prompt = prompt;
      scene.prompt_summary = "";
      scene.status = "image_prompt_ready";
      if (!quiet) createToast(`${genericName} created image prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || runnerName}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`${genericName} Storyboard image prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  function enforceStoryboardVideoFacialRequirements(prompt, scene) {
    let text = String(prompt || "").trim();
    const normalized = normalizeScene(scene, 0);
    const promptMentionsFace = /\b(?:woman|man|girl|boy|person|subject|singer|rapper|performer|speaker|character|face|eyes?|brows?|gaze|mouth|jaw|cheeks?|expression|smile|frown|sings?|singing|says|speaks?)\b/i.test(text);
    const hasCharacter = !normalized.no_character_present && (
      (Array.isArray(normalized.subject_refs) && normalized.subject_refs.length)
      || (Array.isArray(normalized.subjects) && normalized.subjects.length)
      || promptMentionsFace
    );
    if (!text || !hasCharacter) return text;
    const vocalStatus = normalized.vocal_status || {};
    const promptSaysSinging = /\b(?:sings?|singing|raps?|rapping)\b/i.test(text);
    const isSinging = promptSaysSinging || (String(normalized.performance_mode || vocalStatus.performance_mode || state.performanceMode || "").trim() === "singing"
      && vocalStatus.should_lip_sync !== false
      && !vocalStatus.instrumental
      && !vocalStatus.no_lip_sync
      && !normalized.lyric_no_lip_sync
      && Boolean(String(vocalStatus.lyric_text || normalized.lyrics || "").trim()));
    if (isSinging) {
      text = text
        .replace(/\bwith\s+a\s+quiet,\s*internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bwith\s+quiet\s+internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bquiet,\s*internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+intensity\b/gi, "controlled intensity")
        .replace(/\bquiet\s+performance\b/gi, "controlled performance")
        .replace(/\bquiet\s+emotion\b/gi, "restrained emotion")
        .replace(/\bquiet\s+singing\b/gi, "focused singing");
    }
    const hasBlink = /\bblink\w*\b/i.test(text);
    const hasEyeMovement = /\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b/i.test(text);
    const additions = [];
    if (!hasEyeMovement) additions.push("subtle natural eye movement");
    if (!hasBlink) additions.push("occasional natural blinking");
    if (additions.length) {
      const insert = `, ${additions.join(", ")}`;
      const faceSentence = text.match(/([^.]*(?:face|eyes?|brows?|gaze|expression)[^.]*)(\.)/i);
      if (faceSentence && typeof faceSentence.index === "number") {
        const nextSentence = `${faceSentence[1].trimEnd()}${insert}`;
        text = `${text.slice(0, faceSentence.index)}${nextSentence}${text.slice(faceSentence.index + faceSentence[1].length)}`;
      } else {
        text = `${text.replace(/\.+\s*$/, "")} with ${additions.join(", ")}.`;
      }
    }
    return text.replace(/\s{2,}/g, " ").trim();
  }

  function applyStoryboardTriggerPhrases(prompt, scene) {
    let text = enforceStoryboardVideoFacialRequirements(prompt, scene);
    const normalized = normalizeScene(scene, 0);
    const refs = normalizeReferenceBuilderCatalog(state.referenceBuilder || {});
    const parts = { start: [], end: [] };
    const add = (trigger, position = "start") => {
      const value = String(trigger || "").trim();
      if (!value) return;
      const key = position === "end" ? "end" : "start";
      if (!parts[key].some((item) => item.toLowerCase() === value.toLowerCase())) parts[key].push(value);
    };
    const subjectPosition = refs.subject_trigger_position === "end" ? "end" : "start";
    const locationPosition = refs.location_trigger_position === "end" ? "end" : "start";
    (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : []).forEach((subject) => {
      add(subject.trigger_phrase || subject.trigger || subject.Trigger, subjectPosition);
    });
    if (normalized.location_ref) {
      add(normalized.location_ref.trigger_phrase || normalized.location_ref.trigger || normalized.location_ref.Trigger, locationPosition);
    }
    add(normalized.trigger_phrase || normalized.trigger || normalized.Trigger, normalized.trigger_position === "end" ? "end" : "start");
    const stripBoundaryTrigger = (value, trigger) => {
      let current = String(value || "").trim();
      const escaped = String(trigger || "").trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!escaped) return current;
      const leading = new RegExp(`^\\s*${escaped}\\s*(?:,\\s*)?`, "i");
      const trailing = new RegExp(`(?:,\\s*)?${escaped}\\s*$`, "i");
      let previous = "";
      while (current && current !== previous) {
        previous = current;
        current = current.replace(leading, "").replace(trailing, "").trim();
      }
      return current;
    };
    [...parts.start, ...parts.end]
      .sort((a, b) => b.length - a.length)
      .forEach((trigger) => {
        text = stripBoundaryTrigger(text, trigger);
      });
    if (parts.start.length) {
      const prefix = parts.start.join(", ");
      if (!text.toLowerCase().startsWith(prefix.toLowerCase())) text = text ? `${prefix}, ${text}` : prefix;
    }
    if (parts.end.length) {
      const suffix = parts.end.join(", ");
      if (!text.toLowerCase().endsWith(suffix.toLowerCase())) text = text ? `${text}, ${suffix}` : suffix;
    }
    return text;
  }

  async function createSceneVideoPromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending scene card to ${runnerName}...\nThis can take a minute depending on runner/model speed.`, progressPercent);
      const callbackPayload = storyboardGptPayload(state, [scene]);
      const data = state.onCreateVideoPrompt
        ? await state.onCreateVideoPrompt(scene, {
          unloadAfter,
          storyboardPayload: callbackPayload,
          progress,
          progressPercent,
          progressLabel,
        })
        : await postJson("/vrgdg/storyboard/gemma_video_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter }), 240000);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: ${genericName} response received.\nRunner: ${data.runner || runnerName}\nSaving prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const rawPrompt = String(data?.prompt || data || "").trim();
      const prompt = data?.already_finalized ? rawPrompt : applyStoryboardTriggerPhrases(rawPrompt, scene);
      if (!prompt) throw new Error(`${genericName} returned an empty Storyboard video prompt.`);
      scene.video_prompt = prompt;
      scene.status = "video_prompt_ready";
      if (!quiet) createToast(`${genericName} created video prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || runnerName}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`${genericName} Storyboard prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  async function createScenePromptForActiveMode(scene, options = {}) {
    return state.mode === "image_to_video_prep"
      ? createSceneVideoPromptWithGemma(scene, options)
      : createSceneImagePromptWithGemma(scene, options);
  }

  async function createAllPromptsWithGemma() {
    const scenes = currentRows();
    if (!scenes.length) {
      createToast("No storyboard scenes found.", true);
      return;
    }
    const videoMode = state.mode === "image_to_video_prep";
    const promptKind = videoMode ? "video" : "image";
    gemmaAllButton.disabled = true;
    const previousText = gemmaAllButton.textContent;
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    const progress = createStoryboardProgressWindow(`Storyboard ${runnerName} All`);
    let created = 0;
    try {
      const keepLoaded = Boolean(keepGemmaLoadedInput.checked);
      progress.set(`Starting Storyboard ${runnerName} All...\nMode: ${videoMode ? "Video Prep" : "Image Prep"}\nScenes: ${scenes.length}\nKeep local LLM loaded: ${keepLoaded ? "yes" : "no"}`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        gemmaAllButton.textContent = `${runnerName} ${index + 1}/${scenes.length}`;
        const unloadAfter = keepLoaded ? index === scenes.length - 1 : true;
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        const label = `${runnerName} All ${index + 1}/${scenes.length}: ${scenes[index].label || `Scene ${scenes[index].scene_number || index + 1}`}`;
        progress.set(`${label}\nCreating storyboard ${promptKind} prompt...`, base);
        await createScenePromptForActiveMode(scenes[index], { quiet: true, unloadAfter, progress, progressPercent: base, progressLabel: label });
        created += 1;
      }
      progress.set("Saving storyboard prompts...", 96);
      await saveStoryboard();
      progress.set(`${runnerName} All complete.\nCreated ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}.`, 100);
      progress.close(1800);
      createToast(`${genericName} created ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}.`);
    } catch (error) {
      progress.set(`${runnerName} All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, 100);
      createToast(`${runnerName} All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, true);
    } finally {
      gemmaAllButton.disabled = false;
      gemmaAllButton.textContent = previousText;
      renderTable();
    }
  }

  stepPrompts.onclick = () => setMode("storyboard_prompts");
  stepPrep.onclick = () => setMode("image_to_video_prep");
  search.oninput = () => {
    state.query = search.value || "";
    renderTable();
  };
  cameraFlowSelect.onchange = () => {
    state.cameraFlow = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowSelect.value] ? cameraFlowSelect.value : "balanced";
    cameraFlowSelect.value = state.cameraFlow;
    refreshCameraFlowInfo();
    notifyStoryboardDefaultsChanged();
  };
  imageShotSelect.onchange = () => {
    state.imageShotFlow = imageShotFlowPresets[imageShotSelect.value] ? imageShotSelect.value : Object.keys(imageShotFlowPresets)[0] || "off";
    imageShotSelect.value = state.imageShotFlow;
    refreshImageShotInfo();
    notifyStoryboardDefaultsChanged();
  };
  imageAestheticSelect.onchange = () => {
    state.imageAesthetic = imageAestheticPresets.some((preset) => preset.value === imageAestheticSelect.value) ? imageAestheticSelect.value : imageAestheticPresets[0]?.value || "";
    imageAestheticSelect.value = state.imageAesthetic;
    refreshImageAestheticInfo();
    notifyStoryboardDefaultsChanged();
  };
  consistencyInput.addEventListener("input", () => {
    state.globalConsistencyPhrase = consistencyInput.value.trim();
    refreshConsistencyInfo();
  });
  consistencyInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cameraSpeedInput.addEventListener("input", () => {
    state.cameraMotionSpeed = storyboardSpeedValue(cameraSpeedInput.value, 4);
    cameraSpeedInput.value = String(state.cameraMotionSpeed);
    refreshCameraSpeedInfo();
  });
  cameraSpeedInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cameraSpeedHint.onclick = () => {
    window.alert([
      "Camera Motion Speed controls how much movement Gemma/GPT should put into the camera plan for Video Prep.",
      "",
      "0: locked-off static camera.",
      "1-3: slow, gentle camera motion; one simple move at most.",
      "4-6: controlled cinematic movement like tracking, pan, dolly, crane, reveal, or orbit.",
      "7-8: energetic movement with stronger tracking, orbit, whip pan, rise, reveal, or compound motion.",
      "9-10: fast action camera language; multiple coordinated moves can happen in one scene while keeping the subject readable.",
    ].join("\n"));
  };
  cameraFlowApply.onclick = () => applyCameraFlow({ overwrite: false });
  cameraFlowReplace.onclick = () => applyCameraFlow({ overwrite: true });
  imageShotApply.onclick = () => applyImageShotFlow({ overwrite: false });
  imageShotReplace.onclick = () => applyImageShotFlow({ overwrite: true });
  imageAestheticApply.onclick = () => applyImageAesthetic({ overwrite: false });
  imageAestheticReplace.onclick = () => applyImageAesthetic({ overwrite: true });
  performanceSelect.onchange = () => {
    state.performanceStyle = String(performanceSelect.value || "");
    refreshPerformanceInfo();
    notifyStoryboardDefaultsChanged();
  };
  characterSpeedInput.addEventListener("input", () => {
    state.characterMotionSpeed = storyboardSpeedValue(characterSpeedInput.value, 4);
    characterSpeedInput.value = String(state.characterMotionSpeed);
    refreshCharacterSpeedInfo();
  });
  characterSpeedInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  characterSpeedHint.onclick = () => {
    window.alert([
      "Character Motion Speed controls how active the subject's body movement should be.",
      "",
      "0: subject stays still or holds a pose.",
      "1-3: subtle motion like gestures, turns, swaying, reaching, or small steps.",
      "4-6: active performance like walking, dancing, interacting with objects, or using the set.",
      "7-8: energetic action like running, hard dancing, climbing, struggling, spinning, or crossing the space.",
      "9-10: fast action movement like sprinting, explosive dance, chase beats, rapid direction changes, or intense physical performance.",
    ].join("\n"));
  };
  facialSelect.onchange = () => {
    state.facialPerformance = String(facialSelect.value || "");
    refreshFacialInfo();
    notifyStoryboardDefaultsChanged();
  };
  facialCustomInput.oninput = () => {
    state.facialPerformanceCustom = String(facialCustomInput.value || "");
    refreshFacialInfo();
  };
  facialCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  performanceApply.onclick = () => applyPerformanceStyle({ overwrite: false });
  performanceReplace.onclick = () => applyPerformanceStyle({ overwrite: true });
  facialApply.onclick = () => applyFacialPerformance({ overwrite: false });
  facialReplace.onclick = () => applyFacialPerformance({ overwrite: true });
  add.onclick = () => {
    const next = normalizeScene({ scene_number: state.scenes.length + 1, label: `Scene ${state.scenes.length + 1}` }, state.scenes.length);
    state.scenes.push(next);
    openSceneEditor(next);
    renderTable();
  };
  gptButton.onclick = copyStoryboardForGpt;
  importImagePromptsButton.onclick = openImportImagePromptsFromGptModal;
  gemmaAllButton.onclick = createAllPromptsWithGemma;
  clearPromptsButton.onclick = clearAllStoryboardPrompts;
  storyLayerEnabledInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  lyricStoryStrengthInput.addEventListener("input", () => {
    syncLyricStoryStrengthLabel();
    syncStoryLayerFromInputs();
  });
  lyricStoryStrengthInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  lyricStoryStrengthHintButton.onclick = () => {
    window.alert([
      "Lyric Story Strength controls how literally Gemma should follow the lyrics when creating the story arc, story brief, scene beats, and prompt context.",
      "",
      "0: do not use lyrics as story source.",
      "1-3: use lyrics as mood and emotional timing only.",
      "4-6: balance lyrics with the story arc, subjects, and locations.",
      "7-8: lyrics strongly shape the scene story; include recognizable lyric anchors when possible.",
      "9-10: use lyrics as literally as possible; non-instrumental scenes should include a concrete object, action, emotion, or situation from the exact lyric line whenever possible.",
    ].join("\n"));
  };
  userStoryArcInput.addEventListener("input", syncStoryLayerFromInputs);
  userStoryArcInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  songStoryBriefInput.addEventListener("input", syncStoryLayerFromInputs);
  songStoryBriefInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  createStoryArcButton.onclick = createStoryArcWithGemma;
  createStoryBriefButton.onclick = createStoryBriefWithGemma;
  createMissingBeatsButton.onclick = () => createAllSceneBeatsWithGemma({ overwrite: false });
  replaceBeatsButton.onclick = () => createAllSceneBeatsWithGemma({ overwrite: true });
  detectSectionsButton.onclick = detectLyricSections;
  planDialogueScenesButton.onclick = planIdLoraDialogueScenesWithGemma;
  applyDialoguePlanButton.onclick = applyIdLoraDialoguePlanToVideoBuilder;
  keepGemmaLoadedInput.onchange = () => {
    state.gemmaSettings = {
      ...(state.gemmaSettings || {}),
      keep_loaded_for_storyboard_all: Boolean(keepGemmaLoadedInput.checked),
    };
  };
  save.onclick = saveStoryboard;
  exportPrompts.onclick = exportPromptFiles;
  close.onclick = () => backdrop.remove();
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) backdrop.remove();
  });
  refreshCameraFlowInfo();
  refreshImageShotInfo();
  refreshImageAestheticInfo();
  refreshConsistencyInfo();
  refreshCameraSpeedInfo();
  refreshPerformanceInfo();
  refreshCharacterSpeedInfo();
  refreshFacialInfo();
  setMode(state.mode || "storyboard_prompts");
  loadExisting();
}

window.VRGDGStoryboardBuilder = window.VRGDGStoryboardBuilder || {};
window.VRGDGStoryboardBuilder.open = openStoryboardBuilder;

function ensureButton(node) {
  const buttonName = "Open Storyboard Builder";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => {
    const projectWidget = (node.widgets || []).find((item) => item.name === "project_folder");
    openStoryboardBuilder({ projectFolder: projectWidget?.value || "" });
  });
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
}

app.registerExtension({
  name: "vrgdg.StoryboardBuilderUI",
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
