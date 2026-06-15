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

const STORYBOARD_CAMERA_FLOW_PRESETS = {
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

const PERFORMANCE_STYLE_PRESETS = [
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
    direction: "Use rap-style delivery instead of soft singing: confident direct-to-camera energy, expressive hand gestures, head nods, shoulder movement, and sharper body language.",
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
    direction: "Use smooth R&B performance energy: relaxed confident expression, controlled sensual movement, gentle hand gestures, soft rhythmic body motion, and close emotional delivery.",
  },
  {
    value: "edm_club",
    label: "EDM / club",
    direction: "Use energetic club performance energy: rhythmic movement, dance-like gestures, bright reactive expression, beat-driven body language, and dynamic camera motion.",
  },
  {
    value: "spoken_word",
    label: "Spoken word",
    direction: "Use spoken-word delivery instead of singing: focused eyes, intentional gestures, restrained intensity, and poetic performance energy.",
  },
  {
    value: "no_vocals_broll",
    label: "No vocals / B-roll",
    direction: "Do not include singing, rapping, speaking, lip-sync, mouth movement, microphones, or vocal performance. Use visual action, environment interaction, and mood-driven movement only.",
  },
];

function storyboardPerformancePreset(value = "") {
  return PERFORMANCE_STYLE_PRESETS.find((item) => item.value === value) || PERFORMANCE_STYLE_PRESETS[0];
}

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

function storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion = "") {
  const preset = STORYBOARD_CAMERA_FLOW_PRESETS[profileKey] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
  const sequence = preset.sequence || [];
  if (!sequence.length) return null;
  let entry = sequence[sceneIndex % sequence.length];
  if (previousMotion && storyboardMotionFamily(entry.camera) === storyboardMotionFamily(previousMotion)) {
    entry = sequence[(sceneIndex + 1) % sequence.length] || entry;
  }
  return entry;
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
  const subjects = Array.isArray(source.subjects) ? source.subjects
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `subject_${index + 1}`),
      name: String(item.name || `Character ${index + 1}`),
      description: String(item.description || ""),
      image: normalizeReferenceImage(item),
    })) : [];
  const locations = Array.isArray(source.locations) ? source.locations
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `location_${index + 1}`),
      name: String(item.name || `Location ${index + 1}`),
      description: String(item.description || ""),
      image: normalizeReferenceImage(item),
    })) : [];
  return { subjects, locations };
}

function mergeReferenceBuilderCatalog(base = {}, incoming = {}) {
  const normalizedBase = normalizeReferenceBuilderCatalog(base);
  const normalizedIncoming = normalizeReferenceBuilderCatalog(incoming);
  const mergeList = (left, right) => {
    const byKey = new Map();
    const keyFor = (item) => String(item.id || item.name || "").trim().toLowerCase();
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
    locations: mergeList(normalizedBase.locations, normalizedIncoming.locations),
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

function normalizeScene(scene = {}, index = 0) {
  const rawVideoType = String(scene.video_prompt_type || scene.video_type || scene.mode || "").trim();
  const videoPromptType = ["i2v", "t2v", "rtv"].includes(rawVideoType) ? rawVideoType : "i2v";
  const lyrics = scene.lyrics || scene.lyric_text || "";
  const lyricSingers = Array.isArray(scene.lyric_singers)
    ? scene.lyric_singers.map((item) => String(item || "").trim()).filter(Boolean)
    : String(scene.lyric_singers || scene.singers || "").split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
  const lyricNoLipSync = Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync || scene.noLipSync || scene.broll || scene.b_roll);
  const lyricInstrumental = Boolean(scene.lyric_instrumental || scene.instrumental || storyboardIsInstrumentalText(lyrics));
  return {
    id: scene.id || `storyboard_scene_${index + 1}_${Date.now()}`,
    scene_number: Number(scene.scene_number || scene.number || index + 1),
    label: scene.label || `Scene ${index + 1}`,
    lyrics,
    lyric_singers: lyricSingers,
    lyric_no_lip_sync: lyricNoLipSync,
    lyric_instrumental: lyricInstrumental,
    prompt_summary: scene.prompt_summary || scene.summary || "",
    motion_summary: scene.motion_summary || scene.video_notes || scene.i2v_notes || "",
    subjects: Array.isArray(scene.subjects) ? scene.subjects : String(scene.subjects || "").split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean),
    subject_refs: Array.isArray(scene.subject_refs) ? scene.subject_refs.filter((item) => item && typeof item === "object") : [],
    setting: scene.setting || scene.location_ref?.description || scene.location_ref?.name || scene.location || "",
    location_ref: scene.location_ref && typeof scene.location_ref === "object" ? scene.location_ref : null,
    video_prompt_type: videoPromptType,
    shot_type: scene.shot_type || "",
    camera_motion: scene.camera_motion || scene.motion_preset || "",
    character_motion: scene.character_motion || scene.character_motion_preset || scene.subject_motion || "",
    performance_style: scene.performance_style || scene.song_style || scene.music_style || "",
    include_microphone: Boolean(scene.include_microphone || scene.use_microphone || scene.microphone),
    status: scene.status || "draft",
    image_prompt: scene.image_prompt || scene.t2i_prompt || "",
    video_prompt: scene.video_prompt || scene.i2v_prompt || scene.t2v_prompt || "",
    image_path: scene.image_path || scene.approved_image_path || "",
    notes: scene.notes || "",
  };
}

function scenesFromBuilderPayload(payload = {}) {
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  return scenes.map((scene, index) => normalizeScene({
    id: scene.id,
    scene_number: index + 1,
    label: scene.label || `Scene ${index + 1}`,
    lyrics: scene.lyric_text || scene.lyrics || "",
    lyric_singers: scene.lyric_singers || scene.singers || [],
    lyric_no_lip_sync: Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync),
    lyric_instrumental: Boolean(scene.lyric_instrumental || scene.instrumental),
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
      include_microphone: Boolean(scene.include_microphone || scene.use_microphone || scene.microphone),
      image_prompt: scene.t2i_prompt || "",
    video_prompt: scene.i2v_prompt || scene.t2v_prompt || "",
    image_path: scene.image_path || scene.approved_image_path || "",
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

function createStoryboardProgressWindow(title = "Storyboard Gemma") {
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

function slimStoryboardForRequest(state) {
  return {
    mode: state.mode,
    camera_flow: state.cameraFlow || "balanced",
    performance_style_default: state.performanceStyle || "",
    reference_builder: {
      subjects: (state.referenceBuilder?.subjects || []).map(slimReferenceForRequest).filter(Boolean),
      locations: (state.referenceBuilder?.locations || []).map(slimReferenceForRequest).filter(Boolean),
    },
    scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
  };
}

const STORYBOARD_GPT_URL = "https://chatgpt.com/g/g-6a28d15f04e88191a2375d564ff8d90c-ltx-2-3-video-builder-from-storyboard-builder";

function storyboardReferenceForGpt(ref) {
  if (!ref) return null;
  const name = String(ref.name || "").trim();
  const description = String(ref.description || "").trim();
  if (!name && !description) return null;
  return { name, description };
}

function storyboardVideoPromptTypeLabel(type) {
  const key = String(type || "").toLowerCase();
  if (key === "t2v") return "text to video";
  if (key === "rtv") return "reference to video";
  if (key === "i2v") return "image to video";
  return key || "image to video";
}

function storyboardScenesForGpt(state) {
  let previousCameraMotion = "";
  return state.scenes.map((scene, index) => {
    const normalized = normalizeScene(scene, index);
    const sceneNumberIndex = Math.max(0, Number(normalized.scene_number || index + 1) - 1);
    const cameraFallback = storyboardCameraFlowEntry(state.cameraFlow || "balanced", sceneNumberIndex, previousCameraMotion);
    const shotType = normalized.shot_type || cameraFallback?.shot || "";
    const cameraMotion = normalized.camera_motion || cameraFallback?.camera || "";
    previousCameraMotion = cameraMotion || previousCameraMotion;
    const lyricText = String(normalized.lyrics || "").trim();
    const instrumental = Boolean(normalized.lyric_instrumental);
    const noLipSync = Boolean(normalized.lyric_no_lip_sync);
    const shouldLipSync = Boolean(lyricText) && !instrumental && !noLipSync;
    const subjectRefs = (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map(storyboardReferenceForGpt)
      .filter(Boolean);
    const subjectFallbacks = (Array.isArray(normalized.subjects) ? normalized.subjects : [])
      .map((name) => ({ name: String(name || "").trim(), description: "" }))
      .filter((item) => item.name);
    const subjectNames = subjectRefs.length
      ? subjectRefs.map((subject) => subject.name).filter(Boolean)
      : subjectFallbacks.map((subject) => subject.name).filter(Boolean);
    const subjectCount = subjectRefs.length || subjectFallbacks.length;
    const explicitSingers = (Array.isArray(normalized.lyric_singers) ? normalized.lyric_singers : [])
      .map((name) => String(name || "").trim())
      .filter(Boolean);
    const singers = shouldLipSync ? (explicitSingers.length ? explicitSingers : subjectNames) : [];
    const locationRef = storyboardReferenceForGpt(normalized.location_ref);
    return {
      scene_number: normalized.scene_number,
      label: normalized.label,
      prompt_type: storyboardVideoPromptTypeLabel(normalized.video_prompt_type),
      lyric_line_to_sing: shouldLipSync ? lyricText : "",
      vocal_status: {
        lyric_text: lyricText,
        singers,
        instrumental,
        no_lip_sync: noLipSync,
        should_lip_sync: shouldLipSync,
      },
      vocal_direction: {
        mode: shouldLipSync ? "sing exact lyric line" : (instrumental ? "instrumental / no vocals" : (noLipSync ? "b-roll / no lip sync" : "no lyric line provided")),
        lyric_line: lyricText,
        singers,
        instruction: shouldLipSync
          ? "Treat lyric_line as words being sung, not as literal scene action. The listed singer(s) should visibly sing this line with expressive facial emotion, gestures, and performance energy. Do not describe mouth shapes or mouth position."
          : "Do not mention singing, lip-syncing, mouth movement, or vocal performance for this scene.",
      },
      scene_summary: normalized.prompt_summary,
      motion_summary: normalized.motion_summary,
      performance_style: storyboardPerformancePreset(normalized.performance_style || state.performanceStyle).label,
      performance_direction: storyboardPerformancePreset(normalized.performance_style || state.performanceStyle).direction,
      microphone: {
        include: Boolean(normalized.include_microphone),
        instruction: normalized.include_microphone
          ? "A microphone may be included if it naturally fits the scene, stage, or performance setup."
          : "Do not mention or add a microphone, mic stand, headset mic, studio mic, or any microphone prop unless the scene notes explicitly ask for one.",
      },
      subject_count: subjectCount,
      subject_instruction: subjectCount === 1
        ? "This scene has exactly one subject. Treat the listed subject as one individual person even if the label sounds plural. Do not create a group, duplicates, backup singers, or multiple versions of the subject. Use singular wording and do not use they/them/their for this one subject."
        : "Only include the listed subjects. Do not add extra people unless the scene notes explicitly ask for them.",
      subjects: subjectRefs.length ? subjectRefs : subjectFallbacks,
      setting: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      shot_type: shotType,
      camera_motion: cameraMotion,
      camera_guidance: {
        selected_camera_motion: cameraMotion,
        avoid_default_inward_moves: true,
        instruction: "Use the selected camera motion as written. Do not add zoom-in, push-in, dolly-in, crash-zoom, or a close-up ending unless that exact inward motion is selected or requested in notes.",
      },
      character_motion: normalized.character_motion,
      text_to_image_prompt: normalized.image_prompt,
      video_prompt: normalized.video_prompt,
      notes: normalized.notes,
    };
  });
}

function storyboardGptPayload(state, scenesOverride = null) {
  const payloadState = scenesOverride ? { ...state, scenes: scenesOverride } : state;
  const selectedScene = scenesOverride?.length === 1 ? normalizeScene(scenesOverride[0], 0) : null;
  return {
    scope: selectedScene ? "single_scene" : "all_scenes",
    selected_scene_number: selectedScene ? selectedScene.scene_number : null,
    storyboard_mode: state.mode === "image_to_video_prep" ? "video prompt planning" : "image and video prompt planning",
    scenes: storyboardScenesForGpt(payloadState),
  };
}

function openStoryboardGptUrl(payload) {
  window.open(STORYBOARD_GPT_URL, "_blank", "noopener,noreferrer");
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
  const state = {
    projectFolder,
    mode: "storyboard_prompts",
    scenes: scenesFromBuilderPayload(payload),
    referenceBuilder: normalizeReferenceBuilderCatalog(payload.referenceBuilder || payload.reference_builder || {}),
    onReferenceMappingsChanged: typeof payload.onReferenceMappingsChanged === "function" ? payload.onReferenceMappingsChanged : null,
    onPromptsExported: typeof payload.onPromptsExported === "function" ? payload.onPromptsExported : null,
    query: "",
    selected: new Set(),
    saving: false,
    gemmaSettings: payload.gemmaSettings || payload.gemma_settings || {},
    cameraFlow: String(payload.cameraFlow || payload.camera_flow || "balanced"),
    performanceStyle: String(payload.performanceStyle || payload.performance_style || payload.performance_style_default || ""),
  };

  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:stretch;justify-content:center;padding:18px;";
  const shell = document.createElement("div");
  shell.style.cssText = "width:min(1820px,calc(100vw - 36px));height:calc(100vh - 36px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#e5e7eb;box-shadow:0 28px 90px rgba(0,0,0,.62);display:grid;grid-template-rows:auto auto 1fr auto;overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,sans-serif;";

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
  const gemmaAllButton = makeButton("Gemma All", "primary");
  gemmaAllButton.title = "Use Gemma4 to create video prompts for every storyboard scene.";
  const keepGemmaLoadedLabel = document.createElement("label");
  keepGemmaLoadedLabel.style.cssText = "display:flex;align-items:center;gap:6px;color:#cbd5e1;font-size:12px;font-weight:800;white-space:nowrap;";
  const keepGemmaLoadedInput = document.createElement("input");
  keepGemmaLoadedInput.type = "checkbox";
  keepGemmaLoadedInput.checked = Boolean(state.gemmaSettings?.keep_loaded_for_storyboard_all);
  keepGemmaLoadedLabel.append(keepGemmaLoadedInput, document.createTextNode("Keep Gemma loaded"));
  keepGemmaLoadedLabel.title = "When checked, Gemma All keeps the text model loaded until the batch finishes. Turn this off for lower VRAM systems.";
  const add = makeButton("+ Add Scene", "purple");
  const close = makeButton("Close");
  headerActions.append(gptButton, gemmaAllButton, keepGemmaLoadedLabel, search, add, close);
  header.append(titleBlock, steps, headerActions);

  const note = document.createElement("div");
  note.style.cssText = "margin:18px 24px 0;border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:12px 14px;font-size:13px;";

  const cameraFlowBar = document.createElement("div");
  cameraFlowBar.style.cssText = "margin:10px 24px 0;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:10px 12px;display:grid;grid-template-columns:auto minmax(280px,1fr);gap:8px 12px;align-items:center;color:#cbd5e1;font-size:12px;";
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
  const performanceControls = document.createElement("div");
  performanceControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const performanceLabel = document.createElement("div");
  performanceLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  performanceLabel.textContent = "Global performance style";
  const performanceSelect = makeSelect(PERFORMANCE_STYLE_PRESETS, state.performanceStyle);
  performanceSelect.style.width = "max-content";
  performanceSelect.style.minWidth = "180px";
  const performanceApply = makeButton("Fill Missing", "primary");
  performanceApply.title = "Fill only blank per-scene performance/song style fields. Existing scene choices are kept.";
  const performanceReplace = makeButton("Replace All");
  performanceReplace.title = "Replace every scene's performance/song style with the selected global style.";
  performanceControls.append(performanceLabel, performanceSelect, performanceApply, performanceReplace);
  const performanceInfo = document.createElement("div");
  performanceInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  cameraFlowBar.append(cameraFlowControls, cameraFlowInfo, performanceControls, performanceInfo);

  const tableWrap = document.createElement("div");
  tableWrap.style.cssText = "margin:18px 24px;overflow:auto;border:1px solid #334155;border-radius:10px;background:#0b1220;";

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 24px;border-top:1px solid #334155;background:#111827;";
  const stats = document.createElement("div");
  stats.style.cssText = "color:#cbd5e1;font-size:13px;";
  const footerActions = document.createElement("div");
  footerActions.style.cssText = "display:flex;gap:10px;align-items:center;";
  const save = makeButton("Save Storyboard");
  const exportPrompts = makeButton("Export T2I + Video Text Files", "purple");
  footerActions.append(save, exportPrompts);
  footer.append(stats, footerActions);

  shell.append(header, note, cameraFlowBar, tableWrap, footer);
  backdrop.append(shell);
  document.body.append(backdrop);

  const setMode = (mode) => {
    state.mode = mode;
    stepPrompts.style.background = mode === "storyboard_prompts" ? "#0e7490" : "#2b2b30";
    stepPrompts.style.borderColor = mode === "storyboard_prompts" ? "#06b6d4" : "#3f3f46";
    stepPrep.style.background = mode === "image_to_video_prep" ? "#0e7490" : "#2b2b30";
    stepPrep.style.borderColor = mode === "image_to_video_prep" ? "#06b6d4" : "#3f3f46";
    shell.querySelector("#vrgdg-storyboard-mode-pill").textContent = mode === "image_to_video_prep" ? "Video Prep" : "Planning";
    shell.querySelector("#vrgdg-storyboard-subtitle").textContent = mode === "image_to_video_prep"
      ? "Review I2V, T2V, and Reference-to-Video scene cards, then refine video prompts before creation."
      : "Write and organize prompts. Image and video rendering stays in the Video Creator workspace.";
    note.textContent = mode === "image_to_video_prep"
      ? "Video prep mode supports I2V, T2V, and Reference to Video per scene. Open a scene card to choose the video prompt type, shot direction, and motion notes."
      : "Storyboard prompt mode is the planning space. Build stronger scene cards first, then export prompt text files for the existing Video Creator.";
    renderTable();
  };

  const cameraFlowEntryForScene = (profileKey, sceneIndex, previousMotion = "") => {
    return storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion);
  };

  const refreshCameraFlowInfo = () => {
    const preset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const count = preset.sequence?.length || 0;
    cameraFlowInfo.textContent = state.cameraFlow === "off"
      ? preset.description
      : `${preset.description} For any scene count, it cycles through ${count} camera beats and only fills blank fields.`;
  };

  const refreshPerformanceInfo = () => {
    const preset = storyboardPerformancePreset(state.performanceStyle);
    performanceInfo.textContent = state.performanceStyle
      ? `${preset.description} Used by Gemma/GPT for scenes without a per-scene performance style.`
      : `${preset.description} Pick a style here to use it as the default for blank scenes.`;
  };

  const applyCameraFlow = ({ overwrite = false } = {}) => {
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

  const applyPerformanceStyle = ({ overwrite = false } = {}) => {
    const value = String(state.performanceStyle || "").trim();
    if (!value) {
      createToast("Choose a global performance style first.");
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
      createToast(changed ? `Performance style replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No performance style fields were changed.");
    } else {
      createToast(changed ? `Performance style filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank performance style fields needed filling.");
    }
  };

  const currentRows = () => {
    const q = state.query.trim().toLowerCase();
    if (!q) return state.scenes;
    return state.scenes.filter((scene) => [
      scene.label,
      scene.lyrics,
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
        subject_ids: (Array.isArray(scene.subject_refs) ? scene.subject_refs : []).map((ref) => String(ref?.id || "")).filter(Boolean),
        location_id: String(scene.location_ref?.id || ""),
      })),
    });
  }

  const videoPromptTypeLabel = (type) => {
    if (type === "t2v") return "T2V";
    if (type === "rtv") return "RTV";
    return "I2V";
  };

  const videoPromptTypeHint = (type) => {
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
    const editorBackdrop = document.createElement("div");
    editorBackdrop.style.cssText = "position:fixed;inset:0;z-index:100012;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const editor = document.createElement("div");
    editor.style.cssText = "width:min(1420px,calc(100vw - 42px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #0e7490;border-radius:16px;background:linear-gradient(135deg,#07111f,#0f172a 46%,#071827);color:#f8fafc;box-shadow:0 28px 90px rgba(0,0,0,.68);padding:18px;display:flex;flex-direction:column;gap:12px;";
    const label = makeInput(scene.label, "Scene label");
    const lyrics = makeTextarea(scene.lyrics, "Lyrics, script, or beat for this scene...", 4);
    const summary = makeTextarea(scene.prompt_summary, "Image prompt summary...", 3);
    const motion = makeTextarea(scene.motion_summary, "Motion/video summary...", 3);
    const cameraMotionOptions = CAMERA_MOTION_GROUPS.flatMap((group) => group.options || []);
    const cameraMotionValue = scene.camera_motion || cameraMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const cameraMotionPreset = makeGroupedSelect(CAMERA_MOTION_GROUPS, cameraMotionValue);
    const customCameraMotion = makeInput(scene.camera_motion || "", "Custom camera motion");
    const characterMotionOptions = CHARACTER_MOTION_GROUPS.flatMap((group) => group.options || []);
    const characterMotionValue = scene.character_motion || characterMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const characterMotionPreset = makeGroupedSelect(CHARACTER_MOTION_GROUPS, characterMotionValue);
    const customCharacterMotion = makeInput(scene.character_motion || "", "Custom character motion");
    const performanceStyle = makeSelect(PERFORMANCE_STYLE_PRESETS, scene.performance_style || "");
    const includeMicLabel = document.createElement("label");
    includeMicLabel.style.cssText = "display:flex;align-items:center;gap:8px;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:9px 10px;font-size:12px;font-weight:900;";
    const includeMic = document.createElement("input");
    includeMic.type = "checkbox";
    includeMic.checked = Boolean(scene.include_microphone);
    includeMicLabel.append(includeMic, document.createTextNode("Include microphone in prompt"));
    const videoPromptType = makeSelect([
      { value: "i2v", label: "Image to Video" },
      { value: "t2v", label: "Text to Video" },
      { value: "rtv", label: "Reference to Video" },
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
    const shot = makeInput(scene.shot_type, "Shot type");
    const shotPreset = makeSelect([{ value: "", label: "Choose a preset..." }, { value: "__custom__", label: "Custom / keep typed value" }], "__custom__");
    const imagePrompt = makeTextarea(scene.image_prompt, "Full text-to-image prompt...", 7);
    const videoPrompt = makeTextarea(scene.video_prompt, "Full video prompt...", 7);
    const imagePath = makeInput(scene.image_path, "Image path");
    const notes = makeTextarea(scene.notes, "Extra planning notes...", 3);
    const selectedSubjectIds = (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
      .map((ref) => String(ref?.id || ""))
      .filter(Boolean);
    const subjectSelect = makeMultiSelect(
      state.referenceBuilder.subjects.map((subject) => ({ value: subject.id, label: subject.name })),
      selectedSubjectIds,
    );
    const locationSelect = makeSelect(
      [
        { value: "", label: "Unassigned" },
        ...state.referenceBuilder.locations.map((location) => ({ value: location.id, label: location.name })),
      ],
      String(scene.location_ref?.id || ""),
    );
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
    const cameraMotionField = field("Camera motion preset", cameraMotionPreset);
    const characterMotionField = field("Character motion preset", characterMotionPreset);
    const customCharacterMotionField = field("Custom character motion", customCharacterMotion);
    const performanceStyleField = field("Performance / song style", performanceStyle);
    const imagePathField = field("Image path", imagePath);
    const motionField = field("Motion / video summary", motion);
    const t2iPromptField = field("T2I prompt", imagePrompt);
    grid.append(field("Video prompt type", videoPromptType), field("Setting", setting), videoTypeHint, field("Subjects", subjects), performanceStyleField, includeMicLabel, shotPresetField, shotCustomField, cameraMotionField, characterMotionField, customCharacterMotionField, imagePathField);
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
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;";
    const gemma = makeButton("Generate Prompt", "purple");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Save Scene Card", "primary");
    actions.append(cancel, gemma, apply);
    const closeEditor = makeButton("×");
    closeEditor.style.cssText += "font-size:26px;line-height:1;width:44px;height:44px;padding:0;border-radius:8px;";
    const header = document.createElement("div");
    header.style.cssText = "display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;";
    const headerIcon = document.createElement("div");
    headerIcon.textContent = "▣";
    headerIcon.style.cssText = "width:54px;height:54px;border-radius:14px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;";
    const headerText = document.createElement("div");
    headerText.innerHTML = `<div style="font-size:28px;font-weight:900;color:#f8fafc;">Edit Scene Card</div><div style="color:#cbd5e1;margin-top:3px;">Define the details for this scene to generate a rich video prompt.</div>`;
    header.append(headerIcon, headerText, closeEditor);

    const basicsGrid = twoCol();
    basicsGrid.append(field("Scene label", label), field("Scene / lyrics", lyrics), field("Prompt mode", iconField("▣", videoPromptType)), field("Performance / song style", performanceStyle), includeMicLabel, videoTypeHint);

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
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || null;
      subjectChip.innerHTML = selectedSubjects.length
        ? selectedSubjects.map((ref) => referenceChipHtml(ref, "Subject")).join("")
        : `<span style="color:#94a3b8;">No subject selected</span>`;
      locationChip.innerHTML = selectedLocation
        ? referenceChipHtml(selectedLocation, "Location")
        : `<span style="color:#94a3b8;">No location selected</span>`;
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

    const motionGrid = threeCol();
    motionGrid.append(
      field("Starting shot preset", iconField("▣", shotPreset)),
      field("Camera motion preset", iconField("▣", cameraMotionPreset)),
      field("Character motion preset", iconField("♟", characterMotionPreset)),
      field("Custom starting shot (optional)", shot),
      field("Custom camera motion (optional)", customCameraMotion),
      field("Custom character motion (optional)", customCharacterMotion),
    );

    const advancedGrid = twoCol();
    advancedGrid.append(field("Prompt summary", summary), field("Motion / video prompt summary", motion), field("Character details", subjectDetails), imagePathField, t2iPromptField, field("Video prompt", videoPrompt));
    const notesWrap = document.createElement("div");
    notesWrap.append(notes);
    editor.replaceChildren(
      header,
      section(1, "Scene Basics", basicsGrid),
      section(2, "References", referencesGrid),
      section(3, "Camera & Motion", motionGrid),
      section(4, "Advanced Options", advancedGrid, { collapsible: true, open: false }),
      section(5, "Notes", notesWrap),
      actions,
    );
    editorBackdrop.append(editor);
    document.body.append(editorBackdrop);
    closeEditor.onclick = () => editorBackdrop.remove();
    const refreshShotPresetForVideoType = () => {
      const type = videoPromptType.value || "i2v";
      const options = type === "i2v" ? VIDEO_SHOT_TYPES : Array.from(new Set([...IMAGE_SHOT_TYPES, ...VIDEO_SHOT_TYPES]));
      const current = shot.value || scene.shot_type || "";
      shotPreset.replaceChildren();
      for (const option of [
        { value: "", label: type === "i2v" ? "Choose camera/motion preset..." : "Choose starting shot preset..." },
        ...options.map((item) => ({ value: item, label: item })),
        { value: "__custom__", label: "Custom / keep typed value" },
      ]) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        shotPreset.append(item);
      }
      shotPreset.value = options.includes(current) ? current : "__custom__";
      shotPresetField.firstChild.textContent = type === "i2v" ? "Camera / motion preset" : "Starting shot preset";
      shotCustomField.firstChild.textContent = type === "i2v" ? "Custom camera / motion" : "Custom starting shot";
      videoTypeHint.textContent = videoPromptTypeHint(type);
      motionField.firstChild.textContent = type === "i2v"
        ? "Motion / camera direction"
        : type === "rtv"
          ? "Motion / camera direction with references"
          : "Motion / camera direction";
      t2iPromptField.style.display = type === "t2v" || type === "rtv" ? "none" : "flex";
      imagePathField.style.display = type === "t2v" || type === "rtv" ? "none" : "flex";
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
    shotPreset.addEventListener("change", () => {
      if (shotPreset.value && shotPreset.value !== "__custom__") shot.value = shotPreset.value;
    });
    cameraMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(cameraMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCameraMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      if (currentMotion.toLowerCase().includes(selectedMotion.toLowerCase())) return;
      motion.value = currentMotion ? `${currentMotion}\nCamera motion: ${selectedMotion}.` : `Camera motion: ${selectedMotion}.`;
    });
    characterMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(characterMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCharacterMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      if (currentMotion.toLowerCase().includes(selectedMotion.toLowerCase())) return;
      motion.value = currentMotion ? `${currentMotion}\nCharacter motion: ${selectedMotion}.` : `Character motion: ${selectedMotion}.`;
    });
    locationSelect.addEventListener("change", () => {
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || null;
      if (selectedLocation) setting.value = selectedLocation.description || selectedLocation.name || "";
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
      scene.lyrics = lyrics.value.trim();
      scene.prompt_summary = summary.value.trim();
      scene.motion_summary = motion.value.trim();
      scene.video_prompt_type = videoPromptType.value || "i2v";
      scene.subjects = subjects.value.split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
      scene.setting = setting.value.trim();
      if (state.referenceBuilder.subjects.length) {
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
      }
      if (state.referenceBuilder.locations.length) {
        const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || null;
        scene.location_ref = selectedLocation;
        if (selectedLocation) scene.setting = selectedLocation.description || selectedLocation.name || scene.setting;
      }
      scene.shot_type = shot.value.trim();
      scene.camera_motion = customCameraMotion.value.trim() || cameraMotionPreset.value.trim();
      scene.character_motion = customCharacterMotion.value.trim() || characterMotionPreset.value.trim();
      scene.performance_style = performanceStyle.value || "";
      scene.include_microphone = Boolean(includeMic.checked);
      scene.image_prompt = imagePrompt.value.trim();
      scene.video_prompt = videoPrompt.value.trim();
      scene.image_path = imagePath.value.trim();
      scene.notes = notes.value.trim();
    };
    cancel.onclick = () => editorBackdrop.remove();
    gemma.onclick = async () => {
      const previous = gemma.textContent;
      gemma.disabled = true;
      gemma.textContent = "Running Gemma...";
      const progress = createStoryboardProgressWindow("Storyboard Gemma");
      try {
        saveEditorFieldsToScene();
        progress.set(`Preparing ${scene.label || "scene"} for Gemma...`, 12);
        await createSceneVideoPromptWithGemma(scene, { progress, progressPercent: 32 });
        progress.set("Storyboard video prompt ready.", 100);
        progress.close(1200);
        videoPrompt.value = scene.video_prompt || "";
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemma.disabled = false;
        gemma.textContent = previous;
      }
    };
    apply.onclick = () => {
      saveEditorFieldsToScene();
      syncReferenceMappingsToVideoCreator();
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
      const actionHtml = `
        <div style="display:flex;align-items:center;gap:7px;white-space:nowrap;">
          <button data-action="edit" style="${sceneActionStyle}">Open Scene Card</button>
          <button data-action="gemma" style="${sceneGemmaStyle}" title="Create this scene's video prompt with Gemma4.">Gemma</button>
          <button data-action="gpt" style="${sceneGptStyle}" title="Copy only this scene card as GPT JSON.">GPT</button>
        </div>`;
      const status = `<span style="display:inline-flex;align-items:center;gap:6px;color:${meta.color};font-weight:900;"><span style="width:8px;height:8px;border-radius:999px;background:${meta.color};display:inline-block;"></span>${escapeHtml(meta.label)}</span>`;
      const miniRefButtonStyle = "margin-top:7px;border:1px dashed #155e75;border-radius:6px;background:#07111f;color:#a5f3fc;padding:5px 7px;font-size:11px;font-weight:900;cursor:pointer;";
      const subjectCell = `<div>${subjectRefsHtml(scene)}</div><button data-action="load-subject-ref" title="Load a subject image for this scene" style="${miniRefButtonStyle}">+ Subject</button>`;
      const settingCell = `<div>${settingRefHtml(scene)}</div><button data-action="load-location-ref" title="Load a location image for this scene" style="${miniRefButtonStyle}">+ Location</button>`;
      const videoType = videoPromptTypeLabel(scene.video_prompt_type || "i2v");
      const shotCell = `<div style="display:flex;flex-direction:column;gap:4px;"><span style="align-self:flex-start;border:1px solid #155e75;border-radius:999px;background:#0f172a;color:#a5f3fc;font-size:11px;font-weight:900;padding:2px 7px;">${escapeHtml(videoType)}</span><strong style="color:#f8fafc;">${escapeHtml(scene.shot_type || "-")}</strong></div>`;
      if (mode === "image_to_video_prep") {
        tr.innerHTML = `
          <td style="padding:13px;"><input type="checkbox" data-action="select" ${state.selected.has(scene.id) ? "checked" : ""}></td>
          <td style="padding:13px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:13px;">${imageCell}</td>
          <td style="padding:13px;max-width:210px;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 95))}</span></td>
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
          <td style="padding:13px;max-width:220px;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 95))}</span></td>
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
        const progress = createStoryboardProgressWindow("Storyboard Gemma");
        try {
          progress.set(`Preparing ${scene.label || "scene"} for Gemma...`, 12);
          await createSceneVideoPromptWithGemma(scene, { progress, progressPercent: 32 });
          progress.set("Storyboard video prompt ready.", 100);
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
      if (savedReferences.subjects.length || savedReferences.locations.length) {
        state.referenceBuilder = mergeReferenceBuilderCatalog(state.referenceBuilder, savedReferences);
      }
      if (Array.isArray(saved.scenes) && saved.scenes.length) {
        state.scenes = saved.scenes.map((scene, index) => {
          const normalized = normalizeScene(scene, index);
          const fresh = incomingScenes.find((item) => item.id === normalized.id)
            || incomingScenes.find((item) => Number(item.scene_number) === Number(normalized.scene_number))
            || null;
          if (!fresh) return normalized;
          const subjectRefs = fresh.subject_refs?.length ? fresh.subject_refs : normalized.subject_refs;
          const subjects = subjectRefs?.length
            ? storyboardSubjectNamesFromRefs(subjectRefs)
            : Array.from(new Set([
              ...(fresh.subjects || []),
              ...(normalized.subjects || []),
            ].map((item) => String(item || "").trim()).filter(Boolean)));
          return {
            ...normalized,
            subjects,
            subject_refs: subjectRefs,
            setting: fresh.location_ref?.name || normalized.setting || fresh.setting,
            location_ref: fresh.location_ref || normalized.location_ref,
          };
        });
      }
      state.mode = saved.mode || state.mode;
      if (saved.camera_flow && STORYBOARD_CAMERA_FLOW_PRESETS[saved.camera_flow]) {
        state.cameraFlow = saved.camera_flow;
        cameraFlowSelect.value = state.cameraFlow;
      }
      state.performanceStyle = String(saved.performance_style_default || saved.performance_style || state.performanceStyle || "");
      performanceSelect.value = state.performanceStyle;
      refreshCameraFlowInfo();
      refreshPerformanceInfo();
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
      const data = await postJson("/vrgdg/storyboard/save", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
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
      const data = await postJson("/vrgdg/storyboard/export_prompts", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      if (state.onPromptsExported) {
        state.onPromptsExported({ scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)) });
      }
      createToast(`Exported ${data.scene_count || 0} scene prompt rows:\n${data.t2i_prompts_path}\n${data.i2v_prompts_path}`);
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

  function storyboardGemmaPayload(scene, overrides = {}) {
    const payload = storyboardGptPayload(state, [scene]);
    return {
      ...(state.gemmaSettings || {}),
      ...overrides,
      storyboard_payload: payload,
      max_new_tokens: 1400,
      temperature: 0.35,
      top_p: 0.90,
    };
  }

  async function createSceneVideoPromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending scene card to Gemma...\nThis can take a minute depending on runner/model speed.`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/gemma_video_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter }), 240000);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: Gemma response received.\nRunner: ${data.runner || "Gemma"}\nSaving prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const prompt = String(data.prompt || "").trim();
      if (!prompt) throw new Error("Gemma returned an empty Storyboard video prompt.");
      scene.video_prompt = prompt;
      scene.status = "video_prompt_ready";
      if (!quiet) createToast(`Gemma created video prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || "Gemma"}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`Gemma Storyboard prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  async function createAllVideoPromptsWithGemma() {
    const scenes = currentRows();
    if (!scenes.length) {
      createToast("No storyboard scenes found.", true);
      return;
    }
    gemmaAllButton.disabled = true;
    const previousText = gemmaAllButton.textContent;
    const progress = createStoryboardProgressWindow("Storyboard Gemma All");
    let created = 0;
    try {
      const keepLoaded = Boolean(keepGemmaLoadedInput.checked);
      progress.set(`Starting Storyboard Gemma All...\nScenes: ${scenes.length}\nKeep Gemma loaded: ${keepLoaded ? "yes" : "no"}`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        gemmaAllButton.textContent = `Gemma ${index + 1}/${scenes.length}`;
        const unloadAfter = keepLoaded ? index === scenes.length - 1 : true;
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        const label = `Gemma All ${index + 1}/${scenes.length}: ${scenes[index].label || `Scene ${scenes[index].scene_number || index + 1}`}`;
        progress.set(`${label}\nCreating storyboard video prompt...`, base);
        await createSceneVideoPromptWithGemma(scenes[index], { quiet: true, unloadAfter, progress, progressPercent: base, progressLabel: label });
        created += 1;
      }
      progress.set("Saving storyboard prompts...", 96);
      await saveStoryboard();
      progress.set(`Gemma All complete.\nCreated ${created} storyboard video prompt${created === 1 ? "" : "s"}.`, 100);
      progress.close(1800);
      createToast(`Gemma created ${created} storyboard video prompt${created === 1 ? "" : "s"}.`);
    } catch (error) {
      progress.set(`Gemma All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, 100);
      createToast(`Gemma All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, true);
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
  };
  cameraFlowApply.onclick = () => applyCameraFlow({ overwrite: false });
  cameraFlowReplace.onclick = () => applyCameraFlow({ overwrite: true });
  performanceSelect.onchange = () => {
    state.performanceStyle = String(performanceSelect.value || "");
    refreshPerformanceInfo();
  };
  performanceApply.onclick = () => applyPerformanceStyle({ overwrite: false });
  performanceReplace.onclick = () => applyPerformanceStyle({ overwrite: true });
  add.onclick = () => {
    const next = normalizeScene({ scene_number: state.scenes.length + 1, label: `Scene ${state.scenes.length + 1}` }, state.scenes.length);
    state.scenes.push(next);
    openSceneEditor(next);
    renderTable();
  };
  gptButton.onclick = copyStoryboardForGpt;
  gemmaAllButton.onclick = createAllVideoPromptsWithGemma;
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
  refreshPerformanceInfo();
  setMode(state.scenes.some((scene) => scene.image_path) ? "image_to_video_prep" : "storyboard_prompts");
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
