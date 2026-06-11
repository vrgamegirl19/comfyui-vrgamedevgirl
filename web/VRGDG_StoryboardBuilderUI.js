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
      "whispering", "lip-syncing", "looking at camera", "looking away",
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
      image: item.image && typeof item.image === "object" ? {
        path: String(item.image.path || ""),
        data: String(item.image.data || ""),
        name: String(item.image.name || ""),
      } : { path: "", data: "", name: "" },
    })) : [];
  const locations = Array.isArray(source.locations) ? source.locations
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `location_${index + 1}`),
      name: String(item.name || `Location ${index + 1}`),
      description: String(item.description || ""),
      image: item.image && typeof item.image === "object" ? {
        path: String(item.image.path || ""),
        data: String(item.image.data || ""),
        name: String(item.image.name || ""),
      } : { path: "", data: "", name: "" },
    })) : [];
  return { subjects, locations };
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
  return state.scenes.map((scene, index) => {
    const normalized = normalizeScene(scene, index);
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
          ? "Treat lyric_line as words being sung, not as literal scene action. The listed singer(s) should visibly sing this line with clear mouth movement and expressive performance."
          : "Do not mention singing, lip-syncing, mouth movement, or vocal performance for this scene.",
      },
      scene_summary: normalized.prompt_summary,
      motion_summary: normalized.motion_summary,
      subject_count: subjectCount,
      subject_instruction: subjectCount === 1
        ? "This scene has exactly one subject. Treat the listed subject as one individual person even if the label sounds plural. Do not create a group, duplicates, backup singers, or multiple versions of the subject. Use singular wording and do not use they/them/their for this one subject."
        : "Only include the listed subjects. Do not add extra people unless the scene notes explicitly ask for them.",
      subjects: subjectRefs.length ? subjectRefs : subjectFallbacks,
      setting: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      shot_type: normalized.shot_type,
      camera_motion: normalized.camera_motion,
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

  shell.append(header, note, tableWrap, footer);
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

  const openSceneEditor = (scene) => {
    const editorBackdrop = document.createElement("div");
    editorBackdrop.style.cssText = "position:fixed;inset:0;z-index:100012;background:rgba(0,0,0,.58);display:flex;align-items:center;justify-content:center;padding:18px;";
    const editor = document.createElement("div");
    editor.style.cssText = "width:min(920px,calc(100vw - 42px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #155e75;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.62);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const label = makeInput(scene.label, "Scene label");
    const lyrics = makeTextarea(scene.lyrics, "Lyrics, script, or beat for this scene...", 4);
    const summary = makeTextarea(scene.prompt_summary, "Image prompt summary...", 3);
    const motion = makeTextarea(scene.motion_summary, "Motion/video summary...", 3);
    const cameraMotionOptions = CAMERA_MOTION_GROUPS.flatMap((group) => group.options || []);
    const cameraMotionValue = scene.camera_motion || cameraMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const cameraMotionPreset = makeGroupedSelect(CAMERA_MOTION_GROUPS, cameraMotionValue);
    const characterMotionOptions = CHARACTER_MOTION_GROUPS.flatMap((group) => group.options || []);
    const characterMotionValue = scene.character_motion || characterMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const characterMotionPreset = makeGroupedSelect(CHARACTER_MOTION_GROUPS, characterMotionValue);
    const customCharacterMotion = makeInput(scene.character_motion || "", "Custom character motion");
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
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const videoTypeHint = document.createElement("div");
    videoTypeHint.style.cssText = "grid-column:1/-1;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;font-size:12px;line-height:1.45;padding:9px 10px;";
    const shotPresetField = field("Shot type preset", shotPreset);
    const shotCustomField = field("Custom shot type", shot);
    const cameraMotionField = field("Camera motion preset", cameraMotionPreset);
    const characterMotionField = field("Character motion preset", characterMotionPreset);
    const customCharacterMotionField = field("Custom character motion", customCharacterMotion);
    const imagePathField = field("Image path", imagePath);
    const motionField = field("Motion / video summary", motion);
    const t2iPromptField = field("T2I prompt", imagePrompt);
    grid.append(field("Video prompt type", videoPromptType), field("Setting", setting), videoTypeHint, field("Subjects", subjects), shotPresetField, shotCustomField, cameraMotionField, characterMotionField, customCharacterMotionField, imagePathField);
    const referenceGrid = document.createElement("div");
    referenceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:10px;";
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
    const gemma = makeButton("Gemma Video Prompt", "primary");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Save Scene Card", "primary");
    actions.append(cancel, gemma, apply);
    editor.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;"><div style="font-size:18px;font-weight:900;color:#cffafe;">Edit Storyboard Scene ${scene.scene_number}</div></div>`;
    editor.append(field("Scene label", label), field("Scene / lyrics", lyrics), field("Prompt summary", summary), motionField, grid, field("Character details", subjectDetails), referenceGrid, t2iPromptField, field("Video prompt", videoPrompt), field("Notes", notes), actions);
    editorBackdrop.append(editor);
    document.body.append(editorBackdrop);
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
    shotPreset.addEventListener("change", () => {
      if (shotPreset.value && shotPreset.value !== "__custom__") shot.value = shotPreset.value;
    });
    cameraMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(cameraMotionPreset.value || "").trim();
      if (!selectedMotion) return;
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
    });
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
      scene.camera_motion = cameraMotionPreset.value.trim();
      scene.character_motion = customCharacterMotion.value.trim() || characterMotionPreset.value.trim();
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
      try {
        saveEditorFieldsToScene();
        await createSceneVideoPromptWithGemma(scene);
        videoPrompt.value = scene.video_prompt || "";
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
      const subjectCell = subjectRefsHtml(scene);
      const settingCell = settingRefHtml(scene);
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
      tr.querySelector('[data-action="gemma"]')?.addEventListener("click", () => createSceneVideoPromptWithGemma(scene));
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
      setMode(state.mode);
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

  async function createSceneVideoPromptWithGemma(scene, { quiet = false, unloadAfter = true } = {}) {
    const normalized = normalizeScene(scene, 0);
    try {
      const data = await postJson("/vrgdg/storyboard/gemma_video_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter }), 240000);
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
    let created = 0;
    try {
      const keepLoaded = Boolean(keepGemmaLoadedInput.checked);
      for (let index = 0; index < scenes.length; index += 1) {
        gemmaAllButton.textContent = `Gemma ${index + 1}/${scenes.length}`;
        const unloadAfter = keepLoaded ? index === scenes.length - 1 : true;
        await createSceneVideoPromptWithGemma(scenes[index], { quiet: true, unloadAfter });
        created += 1;
      }
      await saveStoryboard();
      createToast(`Gemma created ${created} storyboard video prompt${created === 1 ? "" : "s"}.`);
    } catch (error) {
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
