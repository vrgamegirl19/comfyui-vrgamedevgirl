import { api } from "../../../scripts/api.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeEditorImageUrl } from "./comfy_api.mjs";
import { confirmDestructiveAction } from "./confirm_dialog.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, toast } from "./controls.mjs";
import { refmodLibraryChanged } from "./refmod_card.mjs";
import { showRefmodCreatedCard } from "./refmod_created_card.mjs";
import { openRefModsRules } from "./refmods_rules.mjs";
import {
  clampBox, DEFAULT_QUALITY, estimateTokens, expandBox, MIN_CROP_SIZE, QUALITY_PRESETS, REF_TOKEN_CAP, trimBoxFromPixels,
} from "./refmod_trim.mjs";

// RefMods Studio: create a RefMod (.safetensors) and file it under models/refmods/<type>.
// The subfolder is always the type. Reference resolution, pool grid and max tokens are fixed by the app.
// Trim: an automatic crop around the subject, which the user can drag to adjust. The token readout updates as the
// boxes move, and Create RefMod sends the boxes so the file is built from exactly what the preview shows.
// Create calls POST /vrgdg/refmod/create. The order of the images matters: the first image anchors the frame
// size and leads the reference, so Identity has three fixed slots (front, left, right) before the bulk images.

// Concept types that can be created now. The other types the RefMod pack knows (voice, singing,
// music_style, sound_fx, ambience) are listed as unavailable until we decide how to handle them.
export const REFMOD_ACTIVE_TYPES = [
  { value: "identity", label: "Identity", hint: "A specific character or person." },
  { value: "clothing_men", label: "Clothing (men)", hint: "A men's outfit or garment, separate from who is wearing it." },
  { value: "clothing_women", label: "Clothing (women)", hint: "A women's outfit or garment, separate from who is wearing it." },
  { value: "background", label: "Background", hint: "An environment, set or location." },
  { value: "style", label: "Style", hint: "A look, grade or animation style rather than a concrete subject." },
  { value: "object", label: "Object", hint: "A held or placed object." },
  { value: "prop", label: "Prop", hint: "Set dressing or a prop." },
  { value: "vehicle", label: "Vehicle", hint: "A car, bike, boat or other vehicle." },
  { value: "creature", label: "Creature", hint: "An animal, monster or other creature." },
  { value: "pose_motion", label: "Pose / Motion", hint: "A pose, dance, gesture or camera move." },
  { value: "generic", label: "Generic", hint: "Unspecified or mixed content." },
];
export const REFMOD_INACTIVE_TYPES = ["voice", "singing", "music_style", "sound_fx", "ambience"];

// Types that load their images in a fixed order. Each slot holds one image and is loaded before the extras.
export const REFMOD_SLOTS_BY_TYPE = {
  identity: [
    { key: "front", label: "1. Front", hint: "Close-up of the face, looking at the camera." },
    { key: "left", label: "2. Left", hint: "Close-up of the left side of the face." },
    { key: "right", label: "3. Right", hint: "Close-up of the right side of the face." },
  ],
};
const EXTRAS_TITLE_BY_TYPE = {
  identity: "Bulk images of the same character",
};
const EXTRAS_NOTE_BY_TYPE = {
  identity: "Any other shots: full body, other outfits, other angles, expressions. Add as many as you like.",
};

const EXTRACTION_MODES = [
  { value: "Full Reference", label: "Full Reference (stores the full encode)" },
  { value: "Compressed Reference", label: "Compressed Reference (smaller, refined)" },
];
const IMAGE_FILE_PATTERN = /\.(png|jpe?g|webp|bmp|gif)$/i;

const BACKDROP_STYLE = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:16px;";
const BOX_STYLE = `width:min(1720px,calc(100vw - 32px));height:calc(100vh - 32px);box-sizing:border-box;overflow:hidden;border:1px solid #3f3f46;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;
const PANEL_STYLE = "border:1px solid #27272a;border-radius:8px;background:#18181b;padding:12px;display:flex;flex-direction:column;gap:10px;";
const SELECT_STYLE = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;`;
const DROP_IDLE = { border: "#3f3f46", background: "#0b1220" };
const DROP_ACTIVE = { border: "#06b6d4", background: "#0c2a3a" };

function sanitizeName(value) {
  return String(value || "").trim().replace(/[\\/:*?"<>|]/g, "_");
}

function fileName(path) {
  return String(path || "").split(/[\\/]/).pop();
}

function makeSelect(options, value) {
  const select = document.createElement("select");
  select.style.cssText = SELECT_STYLE;
  for (const item of options) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    if (item.disabled) option.disabled = true;
    select.append(option);
  }
  select.value = value;
  return select;
}

function sectionTitle(text) {
  const element = document.createElement("div");
  element.textContent = text;
  element.style.cssText = "font-size:13px;font-weight:800;color:#e4e4e7;";
  return element;
}

function noteText(text) {
  const element = document.createElement("div");
  element.textContent = text;
  element.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  return element;
}

function setButtonEnabled(button, enabled) {
  button.disabled = !enabled;
  button.style.opacity = enabled ? "1" : "0.5";
  button.style.cursor = enabled ? "pointer" : "not-allowed";
}

function smallButton(label) {
  const button = makeButton(label);
  button.style.padding = "5px 9px";
  button.style.fontSize = "11px";
  return button;
}

// Let a file drop onto `element`. `onFiles` receives the image files that were dropped.
function attachDropTarget(element, onFiles) {
  let depth = 0;
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
  const paint = (colors) => {
    element.style.borderColor = colors.border;
    element.style.background = colors.background;
  };
  element.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth += 1;
    paint(DROP_ACTIVE);
  });
  element.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  element.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) paint(DROP_IDLE);
  });
  element.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.stopPropagation();
    depth = 0;
    paint(DROP_IDLE);
    const files = Array.from(event.dataTransfer.files || []);
    const images = files.filter((file) => file.type.startsWith("image/") || IMAGE_FILE_PATTERN.test(file.name));
    if (files.length > images.length) toast(`${files.length - images.length} dropped file(s) were not images and were skipped.`, true);
    if (images.length) onFiles(images);
  });
}

// Animated bar for work with no known percentage. The keyframes are added to the page once.
function ensureBusyStyle() {
  if (document.getElementById("vrgdg-refmods-busy-style")) return;
  const style = document.createElement("style");
  style.id = "vrgdg-refmods-busy-style";
  style.textContent = "@keyframes vrgdg-refmods-slide{0%{left:-40%}100%{left:100%}}";
  document.head.append(style);
}

function makeBusyBar() {
  ensureBusyStyle();
  const track = document.createElement("div");
  track.style.cssText = "position:relative;height:6px;border-radius:999px;background:#1e293b;overflow:hidden;display:none;";
  const mover = document.createElement("div");
  mover.style.cssText = "position:absolute;top:0;bottom:0;width:40%;border-radius:999px;background:#22d3ee;animation:vrgdg-refmods-slide 1.1s ease-in-out infinite;";
  track.append(mover);
  return { element: track, show: (visible) => { track.style.display = visible ? "block" : "none"; } };
}

// Upload one file with real progress (fetch cannot report upload progress). Resolves with the parsed JSON reply.
function uploadWithProgress(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("image", file, file.name);
    const request = new XMLHttpRequest();
    request.open("POST", api.apiURL("/vrgdg/refmod/upload_image"));
    if (api.user) request.setRequestHeader("Comfy-User", api.user);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      try {
        const result = JSON.parse(request.responseText);
        if (request.status >= 200 && request.status < 300 && result.ok) resolve(result);
        else reject(new Error(result.error || "Upload failed."));
      } catch (error) {
        reject(new Error("Upload failed."));
      }
    };
    request.onerror = () => reject(new Error("Upload failed."));
    request.send(form);
  });
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

export function openRefModsStudio({ runnerPayload } = {}) {
  if (document.getElementById("vrgdg-refmods-studio")) return;

  ensureBusyStyle();
  const studio = {
    type: "identity",
    name: "",
    // An image is { id, name, url, path, objectUrl, status: "ready" | "uploading" | "failed" }.
    slots: {},
    extras: [],
    previewHeight: 260,
    describing: false,
    creating: false,
    statusText: "",
    prompts: {},
    // Trim preview settings (preview only).
    trim: { show: true, view: "outline", margin: 16, hideBoxes: false },
    quality: DEFAULT_QUALITY,
  };
  let nextImageId = 1;

  const slotDefs = () => REFMOD_SLOTS_BY_TYPE[studio.type] || [];
  // The images in the order they are sent: the slots first, then the extras.
  const orderedImages = () => [...slotDefs().map((slot) => studio.slots[slot.key]).filter(Boolean), ...studio.extras];
  const readyImages = () => orderedImages().filter((image) => image.status === "ready");
  const readyPaths = () => readyImages().map((image) => image.path);
  // An image is still being prepared while it uploads or until its size has been measured.
  const isPending = (image) => image.status === "uploading" || (image.status !== "failed" && !image.trim);
  const isUploading = () => orderedImages().some(isPending);
  const releaseImage = (image) => {
    if (image?.objectUrl) URL.revokeObjectURL(image.objectUrl);
  };

  const backdrop = document.createElement("div");
  backdrop.id = "vrgdg-refmods-studio";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.style.cssText = BACKDROP_STYLE;
  const box = document.createElement("div");
  box.style.cssText = BOX_STYLE;

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;flex:0 0 auto;";
  const titleWrap = document.createElement("div");
  const title = document.createElement("div");
  title.textContent = "RefMods Studio";
  title.style.cssText = "font-size:18px;font-weight:900;";
  titleWrap.append(title, noteText("Create a RefMod from your own images. It is saved under models/refmods in a folder named after its type."));
  const closeButton = makeButton("Close");
  const rulesButton = makeButton("Rules and uses");
  rulesButton.title = "How RefMods work, what goes in each type, and how to fix common problems.";
  rulesButton.onclick = openRefModsRules;
  const headerButtons = document.createElement("div");
  headerButtons.style.cssText = "display:flex;align-items:center;gap:8px;";
  headerButtons.append(rulesButton, closeButton);
  header.append(titleWrap, headerButtons);

  const columns = document.createElement("div");
  columns.style.cssText = "display:grid;grid-template-columns:400px minmax(0,1fr);gap:14px;flex:1 1 auto;min-height:0;";
  const left = document.createElement("div");
  left.style.cssText = "display:flex;flex-direction:column;gap:12px;min-width:0;min-height:0;overflow:auto;padding-right:2px;";

  // Left: type, name, description, extraction.
  const typePanel = document.createElement("div");
  typePanel.style.cssText = PANEL_STYLE;
  const typeSelect = makeSelect(
    [
      ...REFMOD_ACTIVE_TYPES.map((item) => ({ value: item.value, label: item.label })),
      ...REFMOD_INACTIVE_TYPES.map((value) => ({ value, label: `${value} (not available yet)`, disabled: true })),
    ],
    studio.type,
  );
  const typeHint = noteText("");
  const nameInput = makeInput("");
  nameInput.placeholder = "my_character";
  const savePath = document.createElement("div");
  savePath.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#0f172a;padding:8px;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;";
  typePanel.append(
    sectionTitle("Type and name"),
    makeField("Type", typeSelect),
    typeHint,
    makeField("Name", nameInput),
    savePath,
  );

  const descriptionPanel = document.createElement("div");
  descriptionPanel.style.cssText = PANEL_STYLE;
  const descriptionInput = document.createElement("textarea");
  descriptionInput.rows = 7;
  descriptionInput.style.cssText = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;resize:vertical;`;
  const describeButton = makeButton("Describe from images");
  const describeStatus = noteText("");
  const describeBusy = makeBusyBar();
  const promptDetails = document.createElement("details");
  promptDetails.style.cssText = "font-size:11px;color:#a1a1aa;";
  const promptSummary = document.createElement("summary");
  promptSummary.textContent = "What the model is asked";
  promptSummary.style.cssText = "cursor:pointer;";
  const promptText = document.createElement("div");
  promptText.style.cssText = "margin-top:6px;line-height:1.45;color:#d4d4d8;white-space:pre-wrap;";
  promptDetails.append(promptSummary, promptText);
  descriptionPanel.append(
    sectionTitle("Description"),
    noteText("Stored inside the file so you can read it back later. RefMods keep no trigger words."),
    descriptionInput,
    describeButton,
    describeStatus,
    describeBusy.element,
    promptDetails,
  );

  const settingsPanel = document.createElement("div");
  settingsPanel.style.cssText = PANEL_STYLE;
  const modeSelect = makeSelect(EXTRACTION_MODES, "Full Reference");
  const stepsInput = makeInput("500", "number");
  stepsInput.min = "0";
  stepsInput.max = "2000";
  stepsInput.step = "50";
  const stepsNote = noteText("");
  const qualitySelect = makeSelect(
    QUALITY_PRESETS.map((preset) => ({ value: preset.key, label: preset.key === DEFAULT_QUALITY ? `${preset.label} (recommended)` : preset.label })),
    DEFAULT_QUALITY,
  );
  const qualityNote = noteText("");
  settingsPanel.append(
    sectionTitle("Extraction"),
    makeField("Quality", qualitySelect),
    qualityNote,
    makeField("Mode", modeSelect),
    makeField("Refinement steps", stepsInput),
    stepsNote,
  );
  const trimPanel = document.createElement("div");
  trimPanel.style.cssText = PANEL_STYLE;
  const resetCropsButton = makeButton("Reset all boxes to automatic");
  const trimShow = makeCheckbox("Trim to subject", true);
  const trimViewSelect = makeSelect([
    { value: "outline", label: "Outline on the original" },
    { value: "trimmed", label: "Trimmed result" },
  ], "outline");
  const trimMarginInput = makeInput("16", "number");
  trimMarginInput.min = "0";
  trimMarginInput.max = "96";
  trimMarginInput.step = "4";
  trimPanel.append(
    sectionTitle("Trim to subject"),
    noteText("A crop around the subject is found for every image. Grab any edge or corner of the dashed box on an image to change it, or drag inside the box to move it. Create RefMod builds the file from these boxes, so what you see is what is kept."),
    trimShow.wrapper,
    makeField("View", trimViewSelect),
    makeField("Margin around the subject (px)", trimMarginInput),
    noteText("A new margin applies to images whose box you have not moved."),
    resetCropsButton,
  );
  left.append(typePanel, descriptionPanel, settingsPanel, trimPanel);

  // Right: the image sections. Images keep their own aspect ratio and files can be dropped on any section.
  const imagesPanel = document.createElement("div");
  imagesPanel.style.cssText = `${PANEL_STYLE}min-width:0;min-height:0;overflow:auto;`;
  const imageBar = document.createElement("div");
  imageBar.style.cssText = "display:flex;gap:10px;align-items:center;flex-wrap:wrap;flex:0 0 auto;";
  const clearAllButton = makeButton("Clear all images");
  const hideBoxesButton = makeButton("Hide boxes");
  hideBoxesButton.title = "Hide the trim boxes to see the images clearly. The trim still applies. Grab any edge of a box to change it.";
  const imageCount = noteText("");
  imageCount.style.fontSize = "12px";
  const sizeSlider = document.createElement("input");
  sizeSlider.type = "range";
  sizeSlider.min = "120";
  sizeSlider.max = "640";
  sizeSlider.step = "20";
  sizeSlider.value = String(studio.previewHeight);
  sizeSlider.style.cssText = "width:160px;";
  const sizeWrap = document.createElement("div");
  sizeWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin-left:auto;";
  sizeWrap.append(noteText("Preview size"), sizeSlider);
  imageBar.append(clearAllButton, hideBoxesButton, imageCount, sizeWrap);
  const busyStrip = document.createElement("div");
  busyStrip.style.cssText = "display:none;flex-direction:column;gap:5px;flex:0 0 auto;";
  const busyText = noteText("");
  busyText.style.fontSize = "12px";
  busyText.style.color = "#a5f3fc";
  const busyBar = makeBusyBar();
  busyStrip.append(busyText, busyBar.element);
  const tokenBox = document.createElement("div");
  tokenBox.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#0f172a;padding:8px 10px;font-size:12px;color:#d4d4d8;line-height:1.5;display:none;flex:0 0 auto;";
  const sections = document.createElement("div");
  sections.style.cssText = "display:flex;flex-direction:column;gap:16px;flex:0 0 auto;";
  const canvasMapBox = document.createElement("div");
  canvasMapBox.style.cssText = "border:1px solid #27272a;border-radius:8px;background:#0b1220;padding:12px;display:none;flex-direction:column;gap:10px;flex:0 0 auto;";
  imagesPanel.append(sectionTitle("Reference images"), imageBar, busyStrip, tokenBox, sections, canvasMapBox);
  columns.append(left, imagesPanel);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;flex:0 0 auto;";
  const footerNote = noteText("");
  footerNote.style.fontSize = "12px";
  const createButton = makeButton("Create RefMod", "primary");
  setButtonEnabled(createButton, false);
  const footerStatus = document.createElement("div");
  footerStatus.style.cssText = "display:flex;flex-direction:column;gap:6px;flex:1 1 320px;min-width:240px;";
  const createBusy = makeBusyBar();
  footerStatus.append(footerNote, createBusy.element);
  footer.append(footerStatus, createButton);

  box.append(header, columns, footer);
  backdrop.append(box);
  document.body.append(backdrop);

  let creationSignature = () => "";
  const requirementText = () => {
    if (!sanitizeName(studio.name)) return "Enter a name to create the RefMod.";
    if (!readyPaths().length) return "Add at least one image.";
    const active = activeEstimate();
    if (active && modeSelect.value === "Full Reference" && active.tokens > REF_TOKEN_CAP) {
      return `Over the ${REF_TOKEN_CAP.toLocaleString()} token limit (${Math.round(active.tokens).toLocaleString()}). Choose a lower quality, trim closer, or remove images.`;
    }
    return "";
  };

  const refreshSavePath = () => {
    savePath.textContent = `models/refmods/${studio.type}/${sanitizeName(studio.name) || "<name>"}.safetensors`;
  };

  const refreshActions = () => {
    const uploading = isUploading();
    const busy = studio.describing || studio.creating;
    setButtonEnabled(describeButton, !busy && !uploading && readyPaths().length > 0);
    describeButton.textContent = studio.describing ? "Describing..." : "Describe from images";
    const requirement = requirementText();
    // Right after a RefMod is created the same request would only replace it, so the button stays off until the name,
    // images or settings change.
    const alreadyCreated = Boolean(studio.createdSignature) && !requirement && !uploading && creationSignature() === studio.createdSignature;
    setButtonEnabled(createButton, !busy && !uploading && !requirement && !alreadyCreated);
    createButton.textContent = studio.creating ? "Creating..." : alreadyCreated ? "RefMod created" : "Create RefMod";
    footerNote.textContent = studio.statusText || (uploading ? "Preparing images..." : requirement || (alreadyCreated ? "Change the name, images or settings to create another." : ""));
    createBusy.show(studio.creating || uploading);
    describeBusy.show(studio.describing);
    updateBusyStrip();
  };

  // The strip above the images: shown while any image is uploading or being measured.
  const updateBusyStrip = () => {
    const pending = orderedImages().filter(isPending);
    busyStrip.style.display = pending.length ? "flex" : "none";
    busyBar.show(pending.length > 0);
    if (!pending.length) return;
    const uploading = pending.filter((image) => image.status === "uploading").length;
    busyText.textContent = `Adding ${pending.length} image${pending.length === 1 ? "" : "s"}`
      + (uploading ? `, ${uploading} uploading` : "") + ". Large files take a few seconds.";
  };

  const refreshType = () => {
    const entry = REFMOD_ACTIVE_TYPES.find((item) => item.value === studio.type);
    typeHint.textContent = entry?.hint || "";
    descriptionInput.placeholder = {
      identity: "The character: age and build, face and hair, clothing, distinctive marks.",
      clothing_men: "The clothing only: garments, colours, fabrics, accessories.",
      clothing_women: "The clothing only: garments, colours, fabrics, accessories.",
      object: "The object: what it is, shape, materials, colours, markings.",
      prop: "The prop: what it is, size, materials, colours, condition.",
      vehicle: "The vehicle: type, body, paint, wheels, trim, wear.",
      creature: "The creature: type, build, skin or fur, colours, distinctive features.",
      background: "The setting: location, features, colours, atmosphere.",
      style: "The look: medium, palette, line quality, shading, mood.",
      pose_motion: "The pose or movement.",
      generic: "What the images have in common.",
    }[studio.type] || "";
    promptText.textContent = studio.prompts[studio.type] || "Loading...";
    refreshSavePath();
  };

  const refreshQualityNote = () => {
    qualityNote.textContent = QUALITY_PRESETS.find((preset) => preset.key === studio.quality)?.detail || "";
  };

  const refreshMode = () => {
    const compressed = modeSelect.value === "Compressed Reference";
    stepsInput.disabled = !compressed;
    stepsInput.style.opacity = compressed ? "1" : "0.5";
    stepsNote.textContent = compressed
      ? "How many steps refine the compressed latent. More steps take longer."
      : "Refinement steps only apply to Compressed Reference.";
  };

  // Measure where the subject sits in an image (cached on the image). Runs on a reduced copy for speed.
  const setStage = (image, text) => {
    image.stageText = text;
    if (image.ui) image.ui.text.textContent = text;
  };

  const measureImage = (image) => {
    if (image.trim || image.measuring || image.status === "failed" || !image.url) return;
    image.measuring = true;
    setStage(image, "Loading image");
    const element = new Image();
    element.onload = async () => {
      setStage(image, "Finding the subject");
      await nextFrame();
      try {
        const natW = element.naturalWidth;
        const natH = element.naturalHeight;
        const scale = Math.min(1, 800 / Math.max(natW, natH));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(natW * scale));
        canvas.height = Math.max(1, Math.round(natH * scale));
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(element, 0, 0, canvas.width, canvas.height);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
        const found = trimBoxFromPixels(pixels.data, canvas.width, canvas.height);
        const box = found
          ? { x0: found.x0 / scale, y0: found.y0 / scale, x1: found.x1 / scale, y1: found.y1 / scale }
          : { x0: 0, y0: 0, x1: natW, y1: natH };
        image.trim = { natW, natH, box, el: element };
      } catch (error) {
        image.trim = { natW: element.naturalWidth, natH: element.naturalHeight, box: null, el: element };
      }
      image.measuring = false;
      renderImages();
    };
    element.onerror = () => {
      image.measuring = false;
      image.trim = { natW: 0, natH: 0, box: null, el: null, failed: true };
      renderImages();
    };
    element.src = image.url;
  };

  // The crop box for an image: the one the user dragged, or the automatic box plus the margin.
  const trimmedBoxFor = (image) => {
    const info = image.trim;
    if (!info || !info.natW) return null;
    return image.crop || expandBox(info.box, studio.trim.margin, info.natW, info.natH);
  };

  // Token estimate for the images as they are right now. Null until every image has been measured.
  const currentEstimate = () => {
    const images = orderedImages();
    if (!images.length || images.some((image) => !image.trim?.natW)) return null;
    const sizes = images.map((image) => ({ width: image.trim.natW, height: image.trim.natH }));
    const cropSizes = images.map((image) => {
      const box = trimmedBoxFor(image);
      return { width: Math.max(1, Math.round(box.x1 - box.x0)), height: Math.max(1, Math.round(box.y1 - box.y0)) };
    });
    return estimateTokens({ sizes, cropSizes, quality: studio.quality });
  };

  // What Create RefMod will use: the trimmed numbers when trimming is on, the whole images otherwise.
  function activeEstimate() {
    const estimate = currentEstimate();
    if (!estimate) return null;
    return studio.trim.show ? estimate.trimmed : estimate.whole;
  }

  const updateTokenBox = () => {
    const images = orderedImages();
    if (!images.length) {
      tokenBox.style.display = "none";
      tokenBox.textContent = "";
      canvasMapBox.style.display = "none";
      return;
    }
    tokenBox.style.display = "block";
    if (images.some((image) => image.trim?.failed)) {
      tokenBox.textContent = "Could not read the size of one of the images.";
      canvasMapBox.style.display = "none";
      return;
    }
    const estimate = currentEstimate();
    if (!estimate) {
      tokenBox.textContent = "Measuring images...";
      canvasMapBox.style.display = "none";
      return;
    }
    const fmt = (value) => Math.round(value).toLocaleString();
    const active = studio.trim.show ? estimate.trimmed : estimate.whole;
    const over = active.tokens > REF_TOKEN_CAP && modeSelect.value === "Full Reference";
    const qualityLabel = QUALITY_PRESETS.find((preset) => preset.key === studio.quality)?.label || "";
    const lines = [];
    lines.push(`Quality ${qualityLabel}. Whole images, nothing cut: ${estimate.whole.frames} x ${estimate.whole.canvas[0]}x${estimate.whole.canvas[1]} = ${fmt(estimate.whole.tokens)} tokens`);
    if (studio.trim.show) {
      const saved = estimate.whole.tokens ? Math.round((1 - estimate.trimmed.tokens / estimate.whole.tokens) * 100) : 0;
      lines.push(`With your trim boxes: ${estimate.trimmed.frames} x ${estimate.trimmed.canvas[0]}x${estimate.trimmed.canvas[1]} = ${fmt(estimate.trimmed.tokens)} tokens`
        + (saved > 0 ? ` (${saved}% less)` : saved < 0 ? ` (${-saved}% more)` : ""));
    }
    lines.push(`At each quality${studio.trim.show ? " (with your trim boxes)" : ""}: ${estimate.byQuality.map((entry) => `${entry.label} ${fmt(entry.tokens)}`).join(", ")}`);
    lines.push(`Create RefMod will use ${studio.trim.show ? "the trim boxes" : "the whole images"} at ${qualityLabel}. Limit ${fmt(REF_TOKEN_CAP)} tokens.${over ? " Over the limit: choose a lower quality, trim closer or remove images." : ""}`);
    updateCanvasMap();
    tokenBox.style.borderColor = over ? "#b45309" : "#3f3f46";
    tokenBox.replaceChildren(...lines.map((line, index) => {
      const row = document.createElement("div");
      row.textContent = line;
      if (index === lines.length - 1 && over) row.style.color = "#fbbf24";
      return row;
    }));
  };

  // The canvas map: every image sits on one shared canvas as wide as the widest image and as tall as the tallest, and
  // every frame costs the whole canvas. The map shows each image's size on that canvas so the cost is visible.
  const MAP_COLOURS = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#818cf8", "#fb923c", "#2dd4bf", "#e879f9"];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const svgElement = (name, attributes = {}) => {
    const element = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    return element;
  };

  const updateCanvasMap = () => {
    const images = orderedImages();
    const estimate = currentEstimate();
    if (!images.length || !estimate) {
      canvasMapBox.style.display = "none";
      return;
    }
    const active = studio.trim.show ? estimate.trimmed : estimate.whole;
    const [canvasW, canvasH] = active.canvas;
    const fit = active.fit;
    const items = images.map((image, index) => {
      const box = studio.trim.show ? trimmedBoxFor(image) : { x0: 0, y0: 0, x1: image.trim.natW, y1: image.trim.natH };
      const width = Math.min(canvasW, (box.x1 - box.x0) * fit);
      const height = Math.min(canvasH, (box.y1 - box.y0) * fit);
      return { index, name: image.name, width, height, colour: MAP_COLOURS[index % MAP_COLOURS.length] };
    });
    const widest = items.reduce((best, item) => (item.width > best.width ? item : best), items[0]);
    const tallest = items.reduce((best, item) => (item.height > best.height ? item : best), items[0]);

    const scale = Math.min(360 / canvasW, 300 / canvasH);
    const svg = svgElement("svg", { viewBox: `0 0 ${canvasW} ${canvasH}`, width: Math.round(canvasW * scale), height: Math.round(canvasH * scale) });
    svg.style.cssText = "flex:0 0 auto;background:#020617;border-radius:4px;";
    svg.append(svgElement("rect", { x: 0, y: 0, width: canvasW, height: canvasH, fill: "#0f172a", stroke: "#94a3b8", "stroke-width": 2, "vector-effect": "non-scaling-stroke" }));
    for (const item of items) {
      const x = (canvasW - item.width) / 2;
      const y = (canvasH - item.height) / 2;
      const setsSize = item === widest || item === tallest;
      svg.append(svgElement("rect", {
        x, y, width: item.width, height: item.height, fill: item.colour, "fill-opacity": 0.14, stroke: item.colour,
        "stroke-opacity": 0.95, "stroke-width": setsSize ? 3 : 1.5, "vector-effect": "non-scaling-stroke",
      }));
    }
    for (const item of items) {
      const x = (canvasW - item.width) / 2;
      const y = (canvasH - item.height) / 2;
      const label = svgElement("text", { x: x + 5 / scale, y: y + 15 / scale, fill: item.colour, "font-size": 13 / scale, "font-weight": 700 });
      label.textContent = String(item.index + 1);
      svg.append(label);
    }

    const details = document.createElement("div");
    details.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:240px;flex:1 1 280px;font-size:12px;color:#d4d4d8;line-height:1.45;";
    const fmt = (value) => Math.round(value).toLocaleString();
    const perFrame = (canvasW / 32) * (canvasH / 32);
    const headline = document.createElement("div");
    headline.style.cssText = "font-size:13px;color:#f4f4f5;font-weight:700;";
    headline.textContent = `Canvas ${canvasW}x${canvasH}: ${fmt(perFrame)} tokens per frame x ${items.length} frame${items.length === 1 ? "" : "s"} = ${fmt(active.tokens)} tokens`;
    details.append(headline);
    const explain = document.createElement("div");
    explain.style.cssText = "color:#a1a1aa;";
    explain.textContent = items.length > 1
      ? `Width comes from image ${widest.index + 1} and height from image ${tallest.index + 1} (the thick outlines). Every frame costs the whole canvas, so a smaller image still pays for the empty space around it.`
      : "One image sets the canvas. Add more images to see how they share it.";
    details.append(explain);
    for (const item of items) {
      const unused = Math.max(0, Math.round((1 - (item.width * item.height) / (canvasW * canvasH)) * 100));
      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:8px;align-items:baseline;";
      const dot = document.createElement("span");
      dot.textContent = String(item.index + 1);
      dot.style.cssText = `flex:0 0 18px;text-align:center;font-weight:800;color:${item.colour};`;
      const text = document.createElement("span");
      text.style.cssText = "flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      text.textContent = `${item.name}: ${Math.round(item.width)}x${Math.round(item.height)}, ${unused}% of its frame is padding`;
      text.title = text.textContent;
      row.append(dot, text);
      details.append(row);
    }
    const share = Math.min(1, active.tokens / REF_TOKEN_CAP);
    const barLabel = document.createElement("div");
    barLabel.textContent = `${fmt(active.tokens)} of ${fmt(REF_TOKEN_CAP)} tokens`;
    barLabel.style.cssText = "margin-top:4px;color:#a1a1aa;";
    const track = document.createElement("div");
    track.style.cssText = "height:8px;border-radius:999px;background:#1e293b;overflow:hidden;";
    const fill = document.createElement("div");
    const over = active.tokens > REF_TOKEN_CAP;
    fill.style.cssText = `height:100%;width:${Math.round(share * 100)}%;background:${over ? "#f59e0b" : "#22d3ee"};`;
    track.append(fill);
    details.append(barLabel, track);

    const title = document.createElement("div");
    title.textContent = "Canvas map";
    title.style.cssText = "font-size:13px;font-weight:800;color:#e4e4e7;";
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;";
    row.append(svg, details);
    canvasMapBox.style.display = "flex";
    canvasMapBox.replaceChildren(title, row);
  };

  // Drag handling for one crop box. Moves the whole box or one edge or corner, keeps it inside the image, and updates the
  // outline, the label and the token readout live, without rebuilding the page while the pointer is down.
  // Invisible grab zones along each edge and corner. The edges are listed first so the corners sit on top of them.
  const HANDLES = [
    ["n", "left:14px;right:14px;top:-3px;height:11px;cursor:ns-resize;"],
    ["s", "left:14px;right:14px;bottom:-3px;height:11px;cursor:ns-resize;"],
    ["w", "top:14px;bottom:14px;left:-3px;width:11px;cursor:ew-resize;"],
    ["e", "top:14px;bottom:14px;right:-3px;width:11px;cursor:ew-resize;"],
    ["nw", "left:-3px;top:-3px;width:17px;height:17px;cursor:nwse-resize;"],
    ["ne", "right:-3px;top:-3px;width:17px;height:17px;cursor:nesw-resize;"],
    ["se", "right:-3px;bottom:-3px;width:17px;height:17px;cursor:nwse-resize;"],
    ["sw", "left:-3px;bottom:-3px;width:17px;height:17px;cursor:nesw-resize;"],
  ];
  const OUTLINE_IDLE = "2px dashed #22d3ee";
  const OUTLINE_HOT = "2px solid #a5f3fc";

  const attachCropDrag = (image, stage, outline, label) => {
    outline.addEventListener("pointerdown", (event) => {
      const mode = event.target === outline ? "move" : event.target.dataset?.cropHandle;
      if (!mode) return;
      event.preventDefault();
      event.stopPropagation();
      const info = image.trim;
      const start = { ...trimmedBoxFor(image) };
      const rect = stage.getBoundingClientRect();
      const scaleX = info.natW / rect.width;
      const scaleY = info.natH / rect.height;
      const originX = event.clientX;
      const originY = event.clientY;
      outline.setPointerCapture?.(event.pointerId);
      let frame = 0;
      const apply = (moveEvent) => {
        const dx = (moveEvent.clientX - originX) * scaleX;
        const dy = (moveEvent.clientY - originY) * scaleY;
        let { x0, y0, x1, y1 } = start;
        if (mode === "move") {
          const width = x1 - x0;
          const height = y1 - y0;
          x0 = Math.min(Math.max(0, start.x0 + dx), info.natW - width);
          y0 = Math.min(Math.max(0, start.y0 + dy), info.natH - height);
          x1 = x0 + width;
          y1 = y0 + height;
        } else {
          if (mode.includes("w")) x0 = start.x0 + dx;
          if (mode.includes("e")) x1 = start.x1 + dx;
          if (mode.includes("n")) y0 = start.y0 + dy;
          if (mode.includes("s")) y1 = start.y1 + dy;
          const minimum = Math.min(MIN_CROP_SIZE, info.natW, info.natH);
          if (x1 - x0 < minimum) { if (mode.includes("w")) x0 = x1 - minimum; else x1 = x0 + minimum; }
          if (y1 - y0 < minimum) { if (mode.includes("n")) y0 = y1 - minimum; else y1 = y0 + minimum; }
          ({ x0, y0, x1, y1 } = clampBox({ x0, y0, x1, y1 }, info.natW, info.natH));
        }
        image.crop = { x0, y0, x1, y1 };
        positionOutline(image, outline);
        label.textContent = cropLabel(image);
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
          updateTokenBox();
          refreshActions();
        });
      };
      const finish = () => {
        outline.removeEventListener("pointermove", apply);
        outline.removeEventListener("pointerup", finish);
        outline.removeEventListener("pointercancel", finish);
        renderImages();
      };
      outline.addEventListener("pointermove", apply);
      outline.addEventListener("pointerup", finish);
      outline.addEventListener("pointercancel", finish);
    });
  };

  const positionOutline = (image, outline) => {
    const box = trimmedBoxFor(image);
    const { natW, natH } = image.trim;
    outline.style.left = `${(box.x0 / natW) * 100}%`;
    outline.style.top = `${(box.y0 / natH) * 100}%`;
    outline.style.width = `${((box.x1 - box.x0) / natW) * 100}%`;
    outline.style.height = `${((box.y1 - box.y0) / natH) * 100}%`;
  };

  const cropLabel = (image) => {
    const box = trimmedBoxFor(image);
    const base = image.status === "failed" ? `${image.name} (failed)` : image.name;
    if (!box || !studio.trim.show) return base;
    return `${base} | ${image.trim.natW}x${image.trim.natH} to ${Math.round(box.x1 - box.x0)}x${Math.round(box.y1 - box.y0)}${image.crop ? " (edited)" : ""}`;
  };

  // One image shown at its own aspect ratio, with an adjustable crop box or the trimmed result when trimming is on.
  const buildImageView = (image, { removable, onRemove }) => {
    const card = document.createElement("div");
    card.style.cssText = "position:relative;border:1px solid #27272a;border-radius:6px;background:#09090b;overflow:hidden;flex:0 0 auto;";
    const box = studio.trim.show ? trimmedBoxFor(image) : null;
    const showOutline = box && !studio.trim.hideBoxes;
    const label = document.createElement("div");
    label.style.cssText = `position:absolute;left:0;right:0;bottom:0;padding:3px 6px;background:rgba(9,9,11,.72);font-size:10px;color:${image.status === "failed" ? "#fca5a5" : "#d4d4d8"};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;`;
    label.textContent = cropLabel(image);
    label.title = image.path || image.name;
    let visual;
    if (box && studio.trim.view === "trimmed" && image.trim.el) {
      const cropW = Math.max(1, box.x1 - box.x0);
      const cropH = Math.max(1, box.y1 - box.y0);
      visual = document.createElement("canvas");
      visual.height = Math.round(studio.previewHeight);
      visual.width = Math.max(1, Math.round(studio.previewHeight * cropW / cropH));
      visual.style.cssText = "display:block;max-width:100%;";
      visual.getContext("2d").drawImage(image.trim.el, box.x0, box.y0, cropW, cropH, 0, 0, visual.width, visual.height);
    } else {
      visual = document.createElement("div");
      visual.style.cssText = "position:relative;overflow:hidden;line-height:0;touch-action:none;";
      const thumb = document.createElement("img");
      thumb.alt = image.name;
      thumb.src = image.url;
      thumb.draggable = false;
      thumb.style.cssText = `display:block;height:${studio.previewHeight}px;width:auto;`;
      thumb.onerror = () => { thumb.style.visibility = "hidden"; };
      visual.append(thumb);
      if (showOutline) {
        const outline = document.createElement("div");
        outline.style.cssText = `position:absolute;border:${OUTLINE_IDLE};box-shadow:0 0 0 9999px rgba(0,0,0,.45);box-sizing:border-box;cursor:move;touch-action:none;`;
        positionOutline(image, outline);
        for (const [mode, css] of HANDLES) {
          const zone = document.createElement("div");
          zone.dataset.cropHandle = mode;
          zone.style.cssText = `position:absolute;background:transparent;box-sizing:border-box;${css}`;
          // The edge you are about to grab turns solid, so you can tell where it is without any visible handle.
          zone.addEventListener("pointerenter", () => { outline.style.border = OUTLINE_HOT; });
          zone.addEventListener("pointerleave", () => { outline.style.border = OUTLINE_IDLE; });
          outline.append(zone);
        }
        visual.append(outline);
        attachCropDrag(image, thumb, outline, label);
      }
    }
    card.append(visual, label);
    if (removable) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "x";
      remove.title = "Remove this image";
      remove.style.cssText = "position:absolute;top:4px;right:4px;width:22px;height:22px;border:0;border-radius:4px;background:rgba(9,9,11,.8);color:#fecaca;cursor:pointer;font-weight:800;line-height:1;z-index:2;";
      remove.onclick = onRemove;
      card.append(remove);
    }
    if (box && image.crop) {
      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = "Reset box";
      reset.title = "Go back to the automatic box for this image";
      reset.style.cssText = "position:absolute;top:4px;left:4px;padding:2px 7px;border:0;border-radius:4px;background:rgba(9,9,11,.8);color:#a5f3fc;cursor:pointer;font-size:10px;font-weight:700;z-index:2;";
      reset.onclick = () => {
        image.crop = null;
        renderImages();
      };
      card.append(reset);
    }
    image.ui = null;
    if (isPending(image)) {
      // Something shows at once, so a slow image never looks like a dead click.
      card.style.minWidth = "170px";
      card.style.minHeight = `${Math.round(studio.previewHeight)}px`;
      const overlay = document.createElement("div");
      overlay.style.cssText = "position:absolute;inset:0;background:rgba(2,6,23,.72);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:12px;z-index:1;";
      const text = document.createElement("div");
      text.style.cssText = "font-size:12px;font-weight:700;color:#a5f3fc;text-align:center;";
      text.textContent = image.stageText || "Loading image";
      const track = document.createElement("div");
      track.style.cssText = "position:relative;width:80%;max-width:200px;height:6px;border-radius:999px;background:#1e293b;overflow:hidden;";
      const fill = document.createElement("div");
      if (image.status === "uploading") {
        fill.style.cssText = `height:100%;width:${Math.round((image.progress || 0) * 100)}%;border-radius:999px;background:#22d3ee;transition:width .15s linear;`;
      } else {
        fill.style.cssText = "position:absolute;top:0;bottom:0;width:40%;border-radius:999px;background:#22d3ee;animation:vrgdg-refmods-slide 1.1s ease-in-out infinite;";
      }
      track.append(fill);
      overlay.append(text, track);
      card.append(overlay);
      image.ui = { text, fill };
    }
    return card;
  };

  const startUpload = async (image, file) => {
    image.progress = 0;
    setStage(image, "Uploading 0%");
    try {
      const result = await uploadWithProgress(file, (fraction) => {
        image.progress = fraction;
        const percent = Math.round(fraction * 100);
        setStage(image, `Uploading ${percent}%`);
        if (image.ui) image.ui.fill.style.width = `${percent}%`;
      });
      image.path = result.path;
      image.status = "ready";
      image.progress = 1;
    } catch (error) {
      image.status = "failed";
      toast(`Could not add ${file.name}: ${error?.message || error}`, true);
    }
    renderImages();
  };

  // target: a slot key, or "extras".
  const placeImage = (target, image) => {
    if (target === "extras") {
      studio.extras.push(image);
      return;
    }
    releaseImage(studio.slots[target]);
    studio.slots[target] = image;
  };

  const addDroppedFiles = (target, files) => {
    const list = target === "extras" ? files : files.slice(0, 1);
    if (target !== "extras" && files.length > 1) toast("This slot holds one image. Only the first was used.");
    for (const file of list) {
      const objectUrl = URL.createObjectURL(file);
      const image = { id: nextImageId++, name: file.name, path: "", url: objectUrl, objectUrl, status: "uploading" };
      placeImage(target, image);
      startUpload(image, file);
    }
    renderImages();
  };

  const addPickedPaths = (target, paths) => {
    const list = target === "extras" ? paths : paths.slice(0, 1);
    for (const path of list) {
      if (target === "extras" && orderedImages().some((image) => image.path === path)) continue;
      placeImage(target, {
        id: nextImageId++, name: fileName(path), path, url: makeEditorImageUrl(path), objectUrl: "", status: "ready",
      });
    }
    renderImages();
  };

  const browseInto = async (target, button) => {
    setButtonEnabled(button, false);
    const previous = button.textContent;
    button.textContent = "Waiting...";
    try {
      const response = await api.fetchApi("/vrgdg/refmod/pick_images", { method: "POST" });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "The file dialog could not open.");
      addPickedPaths(target, result.paths || []);
    } catch (error) {
      toast(`Could not choose images: ${error?.message || error}`, true);
    } finally {
      setButtonEnabled(button, true);
      button.textContent = previous;
    }
  };

  const buildSlotCard = (slot) => {
    const image = studio.slots[slot.key];
    const card = document.createElement("div");
    card.style.cssText = "border:2px dashed #3f3f46;border-radius:8px;background:#0b1220;padding:10px;display:flex;flex-direction:column;gap:8px;min-width:0;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;flex-direction:column;gap:2px;";
    const label = document.createElement("div");
    label.textContent = slot.label;
    label.style.cssText = "font-size:13px;font-weight:800;color:#f4f4f5;";
    head.append(label, noteText(slot.hint));
    const stage = document.createElement("div");
    stage.style.cssText = `display:flex;align-items:center;justify-content:center;min-height:${studio.previewHeight}px;`;
    if (image) {
      stage.append(buildImageView(image, {
        removable: true,
        onRemove: () => {
          releaseImage(image);
          delete studio.slots[slot.key];
          renderImages();
        },
      }));
    } else {
      const empty = document.createElement("div");
      empty.textContent = `Drop the ${slot.label.replace(/^\d+\.\s*/, "").toLowerCase()} image here`;
      empty.style.cssText = "color:#71717a;font-size:13px;text-align:center;padding:20px;";
      stage.append(empty);
    }
    const browse = smallButton(image ? "Replace..." : "Browse...");
    browse.onclick = () => browseInto(slot.key, browse);
    card.append(head, stage, browse);
    attachDropTarget(card, (files) => addDroppedFiles(slot.key, files));
    return card;
  };

  const buildExtrasSection = () => {
    const section = document.createElement("div");
    section.style.cssText = "border:2px dashed #3f3f46;border-radius:8px;background:#0b1220;padding:10px;display:flex;flex-direction:column;gap:10px;min-height:160px;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;gap:10px;flex-wrap:wrap;";
    const titleBlock = document.createElement("div");
    titleBlock.style.cssText = "display:flex;flex-direction:column;gap:2px;margin-right:auto;";
    const titleText = document.createElement("div");
    titleText.textContent = EXTRAS_TITLE_BY_TYPE[studio.type] || "Images";
    titleText.style.cssText = "font-size:13px;font-weight:800;color:#f4f4f5;";
    titleBlock.append(titleText, noteText(EXTRAS_NOTE_BY_TYPE[studio.type] || "Drop images here or browse. Different sizes are fine."));
    const browse = smallButton("Browse images...");
    browse.onclick = () => browseInto("extras", browse);
    const clear = smallButton("Clear");
    clear.onclick = () => {
      studio.extras.forEach(releaseImage);
      studio.extras = [];
      renderImages();
    };
    head.append(titleBlock, browse, clear);
    const grid = document.createElement("div");
    grid.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;";
    if (!studio.extras.length) {
      const empty = document.createElement("div");
      empty.textContent = "Drop images here";
      empty.style.cssText = "margin:auto;color:#71717a;font-size:13px;padding:24px;";
      grid.append(empty);
    }
    for (const image of studio.extras) {
      grid.append(buildImageView(image, {
        removable: true,
        onRemove: () => {
          releaseImage(image);
          studio.extras = studio.extras.filter((item) => item.id !== image.id);
          renderImages();
        },
      }));
    }
    section.append(head, grid);
    attachDropTarget(section, (files) => addDroppedFiles("extras", files));
    return section;
  };

  function renderImages() {
    sections.replaceChildren();
    const defs = slotDefs();
    if (defs.length) {
      const slotRow = document.createElement("div");
      slotRow.style.cssText = `display:grid;grid-template-columns:repeat(${defs.length},minmax(0,1fr));gap:12px;`;
      for (const slot of defs) slotRow.append(buildSlotCard(slot));
      sections.append(slotRow);
    }
    if (defs.length) {
      const slotNote = noteText("Front, left and right are optional. Use any of them, or none and add bulk images below. Close-ups of the face give the strongest identity, and the slots are sent first.");
      sections.insertBefore(slotNote, sections.firstChild);
    }
    sections.append(buildExtrasSection());
    const total = orderedImages().length;
    const uploading = orderedImages().filter((image) => image.status === "uploading").length;
    const preparing = orderedImages().filter(isPending).length;
    imageCount.textContent = total ? `${total} image${total === 1 ? "" : "s"}${preparing ? `, ${preparing} loading` : ""}` : "No images yet";
    orderedImages().forEach(measureImage);
    updateTokenBox();
    refreshActions();
  }

  // Leaving a slotted type keeps its images: they move to the front of the extras in their slot order.
  const changeType = (nextType) => {
    const hadSlots = slotDefs().length > 0;
    if (hadSlots && !(REFMOD_SLOTS_BY_TYPE[nextType] || []).length) {
      const moved = slotDefs().map((slot) => studio.slots[slot.key]).filter(Boolean);
      studio.extras = [...moved, ...studio.extras];
      studio.slots = {};
    }
    studio.type = nextType;
    studio.statusText = "";
    refreshType();
    renderImages();
  };

  const close = () => {
    document.removeEventListener("keydown", onKeyDown, true);
    orderedImages().forEach(releaseImage);
    backdrop.remove();
  };
  const onKeyDown = (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    close();
  };
  document.addEventListener("keydown", onKeyDown, true);
  closeButton.onclick = close;
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) close();
  });
  // A file dropped outside a section must not make the browser open it.
  for (const eventName of ["dragover", "drop"]) {
    backdrop.addEventListener(eventName, (event) => {
      if (Array.from(event.dataTransfer?.types || []).includes("Files")) event.preventDefault();
    });
  }

  typeSelect.onchange = () => changeType(typeSelect.value);
  nameInput.oninput = () => {
    studio.name = nameInput.value;
    studio.statusText = "";
    refreshSavePath();
    refreshActions();
  };
  modeSelect.onchange = () => {
    refreshMode();
    renderImages();
  };
  hideBoxesButton.onclick = () => {
    studio.trim.hideBoxes = !studio.trim.hideBoxes;
    hideBoxesButton.textContent = studio.trim.hideBoxes ? "Show boxes" : "Hide boxes";
    renderImages();
  };
  resetCropsButton.onclick = () => {
    for (const image of orderedImages()) image.crop = null;
    renderImages();
  };
  trimShow.input.onchange = () => {
    studio.trim.show = trimShow.input.checked;
    renderImages();
  };
  trimViewSelect.onchange = () => {
    studio.trim.view = trimViewSelect.value;
    renderImages();
  };
  trimMarginInput.oninput = () => {
    studio.trim.margin = Math.min(96, Math.max(0, Number(trimMarginInput.value) || 0));
    renderImages();
  };
  qualitySelect.onchange = () => {
    studio.quality = qualitySelect.value;
    refreshQualityNote();
    renderImages();
  };
  sizeSlider.oninput = () => {
    studio.previewHeight = Number(sizeSlider.value) || 260;
    renderImages();
  };
  clearAllButton.onclick = () => {
    orderedImages().forEach(releaseImage);
    studio.slots = {};
    studio.extras = [];
    renderImages();
  };
  describeButton.onclick = async () => {
    const paths = readyPaths();
    if (!paths.length || studio.describing) return;
    studio.describing = true;
    const llm = typeof runnerPayload === "function" ? runnerPayload() : null;
    const runnerName = { llm_api: "the LLM API model", own_server: "your custom server model" }[llm?.text_runner] || "the model loaded in LM Studio";
    describeStatus.textContent = `Asking ${runnerName} to describe the ${studio.type}...`;
    refreshActions();
    try {
      const response = await api.fetchApi("/vrgdg/refmod/describe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths, concept_type: studio.type, llm }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "The description failed.");
      descriptionInput.value = result.description;
      describeStatus.textContent = "Description added. Edit it if needed.";
    } catch (error) {
      describeStatus.textContent = "";
      toast(`Could not describe the images: ${error?.message || error}`, true);
    } finally {
      studio.describing = false;
      refreshActions();
    }
  };

  // Everything the create request is built from, so an identical second request can be told from a changed one.
  const creationRequest = () => ({
    type: studio.type,
    name: sanitizeName(studio.name),
    paths: readyPaths(),
    crops: readyImages().map((image) => {
      const box = studio.trim.show ? trimmedBoxFor(image) : null;
      return box ? [Math.round(box.x0), Math.round(box.y0), Math.round(box.x1), Math.round(box.y1)] : null;
    }),
    quality: studio.quality,
    description: descriptionInput.value.trim(),
    mode: modeSelect.value,
    steps: Number(stepsInput.value) || 0,
  });
  creationSignature = () => JSON.stringify(creationRequest());
  const createRefMod = async (overwrite) => {
    const response = await api.fetchApi("/vrgdg/refmod/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...creationRequest(), overwrite }),
    });
    return { response, result: await response.json() };
  };
  for (const input of [descriptionInput, stepsInput]) input.addEventListener("input", refreshActions);
  modeSelect.addEventListener("change", refreshActions);
  createButton.onclick = async () => {
    if (createButton.disabled) return;
    studio.creating = true;
    studio.statusText = "Creating the RefMod. The first run loads the video VAE, so it can take a minute.";
    refreshActions();
    try {
      let { response, result } = await createRefMod(false);
      if (response.status === 409 && result.exists) {
        const answer = await confirmDestructiveAction({
          title: "Replace this RefMod?",
          message: "A RefMod with this name already exists in this folder. Creating it again replaces the file.",
          details: [result.path],
          confirmLabel: "Replace",
        });
        if (!answer.confirmed) {
          studio.statusText = "";
          return;
        }
        ({ response, result } = await createRefMod(true));
      }
      if (!response.ok || !result.ok) throw new Error(result.error || "The RefMod could not be created.");
      // Reference Builder cards built from now on list it straight away (an open dropdown re-reads when opened).
      refmodLibraryChanged();
      const canvasNote = result.canvas ? `, ${result.quality || studio.quality} quality, canvas ${result.canvas[0]}x${result.canvas[1]}` : "";
      studio.statusText = `Saved ${result.path} (${result.tokens} tokens${canvasNote}).`;
      studio.createdSignature = creationSignature();
      studio.creating = false;
      refreshActions();
      await showRefmodCreatedCard({
        ...result,
        imageCount: readyPaths().length,
        mode: modeSelect.value,
        description: descriptionInput.value.trim(),
        typeLabel: REFMOD_ACTIVE_TYPES.find((item) => item.value === studio.type)?.label,
      });
    } catch (error) {
      studio.statusText = "";
      toast(`Could not create the RefMod: ${error?.message || error}`, true);
    } finally {
      studio.creating = false;
      refreshActions();
    }
  };

  api.fetchApi("/vrgdg/refmod/describe_prompts")
    .then((response) => response.json())
    .then((result) => {
      studio.prompts = result?.prompts || {};
      promptText.textContent = studio.prompts[studio.type] || "";
    })
    .catch(() => {
      promptText.textContent = "Could not load the prompts.";
    });

  refreshType();
  refreshMode();
  refreshQualityNote();
  renderImages();
  nameInput.focus();
}
