// Thumbnail intro. The Thumbnail button opens a dialog: scrub the video to the exact moment you want and take that frame, let the
// Builder's LLM look at it and write a YouTube-style edit prompt, make the thumbnail from the frame (Overlay, Qwen Image Edit,
// NanoBanana or the image API of the LLM Runner's provider), keep the result.
// When the intro is on, the final stitch puts the picture in front of the video for a few seconds
// (`POST /vrgdg/music_builder/thumbnail/prepend`). The scenes, the timeline and the audio sync are never touched. The pure parts
// (the saved settings, the timeline of frames) are in thumbnail_state.mjs.

import { api } from "../../../scripts/api.js";
import { makeEditorImageUrl, makeEditorVideoUrl, postJson, queueWorkflowPrompt, waitForImages } from "./comfy_api.mjs";
import { BUILDER_FONT_STACK, DEFAULT_NB_IMAGE_MODEL } from "./constants.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, toast } from "./controls.mjs";
import {
  THUMBNAIL_ENGINE_LABELS, thumbnailApiImageProvider, thumbnailApiImageRequest, thumbnailEngineRequest, thumbnailOverlayRequest,
} from "./thumbnail_engines.mjs";
import {
  ensureThumbnailTitle, formatThumbnailTime, normalizeThumbnailApiImage, normalizeThumbnailIntro, normalizeThumbnailOverlay, thumbnailEngineKind,
  thumbnailFrameRange, thumbnailHistoryWith, thumbnailIntroForStitch, thumbnailItemAt, thumbnailLocalSeconds, thumbnailTimelineItems,
  THUMBNAIL_DEFAULT_ENGINE, THUMBNAIL_ENGINES, THUMBNAIL_FPS,
} from "./thumbnail_state.mjs";

const THUMBNAIL_PREPEND_TIMEOUT_MS = 30 * 60 * 1000;
const THUMBNAIL_PROMPT_TIMEOUT_MS = 10 * 60 * 1000;
const THUMBNAIL_FRAME_TIMEOUT_MS = 2 * 60 * 1000;
const THUMBNAIL_API_IMAGE_TIMEOUT_MS = 10 * 60 * 1000;
// Where inside a frame's time slot the player is parked, so it shows that frame and not the one before it.
const FRAME_SEEK_OFFSET = 0.25;
const ENGINE_OPTIONS = THUMBNAIL_ENGINES.map((value) => ({ value, label: THUMBNAIL_ENGINE_LABELS[value] }));
// The AI workflows redraw the frame, so the person, the face and the scene can drift. Overlay is the one that cannot change them.
const REDRAW_WARNING = " It redraws the frame, so the person, the face and the scene can change: use Overlay to keep them exactly.";
const EDIT_HINT = `This workflow edits the frame by instruction, so the prompt tells it what to change.${REDRAW_WARNING}`;
const QWEN_EDIT_HINT = "Qwen Image Edit edits the frame by instruction and tends to keep the person and the scene as they are. Its prompt is about the change only, mostly the title lettering, so put the lettering you want in Style notes (for example: wildstyle graffiti, orange to red gradient, neon cyan outline, drips, splatter, swirl lines in cyan, magenta and yellow). It uses the newest Qwen Image Edit model you have installed (2511 before 2509) and its Lightning LoRA. It still redraws pixels, so fine detail can change a little: use Overlay when nothing may change.";
const OVERLAY_HINT = "Overlay keeps every pixel of your frame and puts the layers on top, like a thumbnail made in an image editor: a color grade, a glow on the brights, a vignette and the title with an outline and a glow, placed in the calmest part of the frame. No AI redraws anything, so the person and the scene cannot change. It needs no prompt.";
const OVERLAY_COLOR_OPTIONS = [
  { value: "yellow", label: "Yellow" }, { value: "white", label: "White" }, { value: "orange", label: "Orange" },
  { value: "cyan", label: "Cyan" }, { value: "red", label: "Red" },
];
const OVERLAY_POSITION_OPTIONS = [
  { value: "auto", label: "Automatic (the calmest part of the frame)" }, { value: "top", label: "Top" }, { value: "bottom", label: "Bottom" },
  { value: "left", label: "Left" }, { value: "right", label: "Right" },
];
const API_IMAGE_HINT = "The pack sends your frame and the prompt to the image API of the provider set in the LLM Runner (Gemini, Grok or OpenAI) with that provider's key, and keeps the picture. The provider and the key are changed in the LLM Runner. The image model is below: leave it empty for the provider's first one, or type any image model the provider has." + REDRAW_WARNING;

// Asks whether this build can make thumbnails, whether it can take one exact frame of a video, make an overlay and use an image API
// (`api_image` has the providers and their image models). A build without the routes answers 404, which is an answer here and not an
// error for the project's error log, so the button stays hidden and nothing else changes.
export async function loadThumbnailCapabilities(state) {
  try {
    const response = await api.fetchApi("/vrgdg/music_builder/thumbnail/capabilities", { timeoutMs: null });
    const data = response.ok ? await response.json().catch(() => null) : null;
    state.thumbnailAvailable = Boolean(data && data.ok !== false && data.version);
    state.thumbnailFrameAvailable = Boolean(state.thumbnailAvailable && data.frame);
    state.thumbnailOverlayAvailable = Boolean(state.thumbnailAvailable && data.overlay);
    state.thumbnailApiImage = state.thumbnailAvailable && data.api_image?.providers ? data.api_image : null;
  } catch {
    state.thumbnailAvailable = false;
    state.thumbnailFrameAvailable = false;
    state.thumbnailOverlayAvailable = false;
    state.thumbnailApiImage = null;
  }
  return state.thumbnailAvailable;
}

// `stitched` is what the stitch route returned. It comes back unchanged apart from `thumbnail_intro`, and `final_video_path` now
// names the video with the picture as its first frame (one frame, never a length of time). When it cannot be added the plain
// stitched video stays and the reason is shown.
export async function applyThumbnailIntroAfterStitch(state, stitched, { progress = null, projectFolder = "" } = {}) {
  const intro = thumbnailIntroForStitch(state);
  const path = String(stitched?.final_video_path || "");
  if (!intro || !path) return stitched;
  progress?.set("Making the thumbnail the first frame of the video...", 98);
  try {
    const reply = await postJson("/vrgdg/music_builder/thumbnail/prepend", {
      input_path: path,
      image_path: intro.image_path,
      replace_source: true,
      project_folder: projectFolder,
    }, THUMBNAIL_PREPEND_TIMEOUT_MS);
    return { ...stitched, thumbnail_intro: reply };
  } catch (error) {
    toast(`The thumbnail could not be added, so the video starts with the first scene. ${String(error?.message || error)}`, true);
    return stitched;
  }
}

// Sends the chosen frame through the chosen image workflow and returns the ComfyUI view descriptor of the result. `settings` is the
// Builder's own settings for that workflow, so it runs with the models, LoRAs and size the user already chose for it.
async function generateThumbnailImage({ engine, prompt, sourcePath, settings, onStatus }) {
  const { url, payload, label } = thumbnailEngineRequest({
    engine, prompt, sourcePath, settings,
    seed: Math.floor(Math.random() * 2 ** 31),
    defaultNbModel: DEFAULT_NB_IMAGE_MODEL,
  });
  onStatus(`Building the ${label} workflow...`);
  const built = await postJson(url, payload);
  onStatus("Queueing the image workflow...");
  const queued = await queueWorkflowPrompt(built.prompt);
  const promptId = queued?.prompt_id;
  if (!promptId) throw new Error("ComfyUI queued the thumbnail image but did not return a prompt_id.");
  const images = await waitForImages(promptId, (message) => onStatus(message));
  const image = images[images.length - 1];
  if (!image) throw new Error("The image workflow finished without an image.");
  return image;
}

const PANEL_CSS = `width:min(1040px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;box-sizing:border-box;border:1px solid #155e75;border-radius:10px;background:#0f172a;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;
const COLUMN_CSS = "display:flex;flex-direction:column;gap:8px;min-width:0;";
const FRAME_CSS = "display:block;width:100%;aspect-ratio:16/9;object-fit:contain;border:1px solid #3f3f46;border-radius:6px;background:#09090b;";
const EMPTY_FRAME_CSS = "display:flex;align-items:center;justify-content:center;width:100%;aspect-ratio:16/9;box-sizing:border-box;border:1px dashed #3f3f46;border-radius:6px;color:#71717a;font-size:12px;text-align:center;padding:8px;";

function textArea(value, rows, placeholder) {
  const area = document.createElement("textarea");
  area.rows = rows;
  area.value = value;
  area.placeholder = placeholder;
  area.style.cssText = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;line-height:1.45;resize:vertical;`;
  return area;
}

function sectionTitle(text) {
  const title = document.createElement("div");
  title.textContent = text;
  title.style.cssText = "font-size:12px;font-weight:600;color:#67e8f9;letter-spacing:.02em;";
  return title;
}

// The dialog. `deps` come from the Builder: state, projectInput, runnerPayload() (the LLM Runner settings), sceneImagePath(segment),
// sceneVideoPath(segment) (its rendered video), sceneLabel(segment, index), engineSettings(engine) (the Builder's own settings for an
// image workflow), save() (a quiet autosave) and createProgressWindow(title).
export function openThumbnailDialog(deps) {
  const { state, projectInput, runnerPayload, sceneImagePath, sceneVideoPath, sceneLabel, engineSettings, save, createProgressWindow } = deps;
  // The settings an image workflow runs with: the Builder's own for it.
  const settingsFor = (engine) => engineSettings?.(engine) || {};
  const projectFolder = () => String(projectInput?.value || state.projectFolder || "").trim();
  if (!projectFolder()) {
    toast("Create or load a project before making a thumbnail.", true);
    return null;
  }
  // The scenes the picker moves through. Without the frame route a scene can only offer its image.
  const canFrame = Boolean(state.thumbnailFrameAvailable);
  const items = thumbnailTimelineItems((state.segments || []).map((segment, index) => ({
    segment, index, start: segment.start, end: segment.end,
    video: canFrame ? String(sceneVideoPath?.(segment) || "").trim() : "",
    image: String(sceneImagePath?.(segment) || "").trim(),
  })));
  if (!items.length) {
    toast("No scene has a rendered video or an image yet. Render a scene or make a scene image first, then open Thumbnail.", true);
    return null;
  }
  const { min: minFrame, max: maxFrame } = thumbnailFrameRange(items);
  const intro = normalizeThumbnailIntro(state.thumbnailIntro);
  let busy = false;
  let frame = Math.max(minFrame, Math.min(maxFrame, intro.source_kind ? intro.source_frame : minFrame));
  let loadToken = 0;

  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = PANEL_CSS;
  backdrop.addEventListener("keydown", (event) => event.stopPropagation());
  backdrop.addEventListener("keyup", (event) => event.stopPropagation());

  const title = document.createElement("div");
  title.textContent = "Thumbnail Intro";
  title.style.cssText = "font-size:17px;font-weight:600;color:#cffafe;";
  const note = document.createElement("div");
  note.textContent = canFrame
    ? "1. Scrub the video to the exact moment you want and click Use this frame. 2. Choose Overlay to keep that frame exactly and put a grade and the title on top, or an image workflow (the LLM looks at the frame and writes its prompt) to redraw it. 3. Make the thumbnail. When it is on, the final video starts with the thumbnail."
    : "This build cannot take a frame of a video, so pick the scene whose image you want to start from. Choose Overlay to keep that image exactly and put a grade and the title on top, or an image workflow (the LLM looks at it and writes its prompt) to redraw it.";
  note.style.cssText = "font-size:12px;line-height:1.45;color:#cbd5e1;";

  // Step 1: the player and the scrubber over the whole timeline.
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.style.cssText = `${FRAME_CSS}display:none;`;
  const still = document.createElement("img");
  still.alt = "";
  still.style.cssText = `${FRAME_CSS}display:none;`;
  const scrub = document.createElement("input");
  scrub.type = "range";
  scrub.min = String(minFrame);
  scrub.max = String(maxFrame);
  scrub.step = "1";
  scrub.style.cssText = "width:100%;accent-color:#22d3ee;";
  scrub.setAttribute("aria-label", "Moment of the video");
  const readout = document.createElement("div");
  readout.style.cssText = "font-size:12px;color:#e2e8f0;font-variant-numeric:tabular-nums;";
  const stepRow = document.createElement("div");
  stepRow.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;align-items:center;";
  const useButton = makeButton("Use this frame", "primary");

  const sceneItemAt = (at) => thumbnailItemAt(items, at);
  const paintReadout = (item) => {
    const seconds = frame / THUMBNAIL_FPS;
    const kind = item.video ? "video" : "scene image";
    readout.textContent = `${sceneLabel(item.segment, item.index)}  ·  ${formatThumbnailTime(seconds)}  ·  frame ${frame}  (${kind})`;
  };
  // Shows the frame: the scene's video parked on that frame, or the scene's image for a scene with no video.
  const showFrame = (next) => {
    frame = Math.max(minFrame, Math.min(maxFrame, Math.round(Number(next) || 0)));
    scrub.value = String(frame);
    const item = sceneItemAt(frame);
    paintReadout(item);
    const token = ++loadToken;
    if (item.video) {
      still.style.display = "none";
      video.style.display = "block";
      const seek = () => {
        if (token === loadToken) video.currentTime = (thumbnailLocalSeconds(item, frame, THUMBNAIL_FPS) * THUMBNAIL_FPS + FRAME_SEEK_OFFSET) / THUMBNAIL_FPS;
      };
      if (video.dataset.path !== item.video) {
        video.dataset.path = item.video;
        video.addEventListener("loadedmetadata", seek, { once: true });
        video.src = makeEditorVideoUrl(item.video, item.segment.video_cache_bust);
      } else {
        seek();
      }
    } else {
      video.style.display = "none";
      still.style.display = "block";
      still.src = makeEditorImageUrl(item.image);
    }
  };
  scrub.addEventListener("input", () => showFrame(Number(scrub.value)));
  // Shift with the arrow keys moves a second at a time. The plain arrows move one frame (the slider's own step).
  scrub.addEventListener("keydown", (event) => {
    if (!event.shiftKey || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) return;
    event.preventDefault();
    showFrame(frame + (event.key === "ArrowRight" ? THUMBNAIL_FPS : -THUMBNAIL_FPS));
  });
  const stepButton = (label, title, move) => {
    const button = makeButton(label);
    button.title = title;
    button.style.padding = "6px 9px";
    button.onclick = () => showFrame(move());
    stepRow.append(button);
    return button;
  };
  stepButton("◀ Scene", "Jump to the start of the previous scene (or to the start of this one).", () => {
    const item = sceneItemAt(frame);
    const before = items.filter((other) => other.startFrame < item.startFrame).pop();
    return frame > item.startFrame ? item.startFrame : (before || item).startFrame;
  });
  stepButton("− 1 s", "Back one second.", () => frame - THUMBNAIL_FPS);
  stepButton("− 1 frame", "Back one frame.", () => frame - 1);
  stepButton("+ 1 frame", "Forward one frame.", () => frame + 1);
  stepButton("+ 1 s", "Forward one second.", () => frame + THUMBNAIL_FPS);
  stepButton("Scene ▶", "Jump to the start of the next scene.", () => {
    const item = sceneItemAt(frame);
    return (items.find((other) => other.startFrame > item.startFrame) || item).startFrame;
  });
  stepRow.append(useButton);

  // The frame in use: what the LLM looks at and what the image workflow starts from.
  const sourceImage = document.createElement("img");
  sourceImage.alt = "";
  sourceImage.style.cssText = `${FRAME_CSS}display:none;`;
  const sourceEmpty = document.createElement("div");
  sourceEmpty.textContent = "No frame chosen yet. Scrub to the moment you want and click Use this frame.";
  sourceEmpty.style.cssText = EMPTY_FRAME_CSS;
  const sourceCaption = document.createElement("div");
  sourceCaption.style.cssText = "font-size:12px;color:#93c5fd;line-height:1.4;";

  // Step 2 and 3: the prompt and the picture. A build without the overlay or the API image route does not offer that engine, and a
  // project saved with it opens on the default engine instead.
  const engineOptions = ENGINE_OPTIONS.filter((option) => (option.value !== "overlay" || state.thumbnailOverlayAvailable)
    && (option.value !== "api_image" || state.thumbnailApiImage));
  const engineSelect = makeSelect(engineOptions, engineOptions.some((option) => option.value === intro.engine) ? intro.engine : THUMBNAIL_DEFAULT_ENGINE);
  const engineHint = document.createElement("div");
  engineHint.style.cssText = "font-size:11px;line-height:1.4;color:#94a3b8;";
  // The API image engine's choices: the provider is the LLM Runner's (shown, not chosen here), the image model is typed or picked from
  // that provider's list.
  const apiModelInput = makeInput(intro.api_image.model);
  const apiModelList = document.createElement("datalist");
  apiModelList.id = "vrgdg-thumbnail-api-models";
  apiModelInput.setAttribute("list", apiModelList.id);
  const apiProviderLine = document.createElement("div");
  apiProviderLine.style.cssText = "font-size:12px;line-height:1.4;";
  const apiGroup = document.createElement("div");
  apiGroup.style.cssText = COLUMN_CSS;
  apiGroup.append(apiProviderLine, makeField("Image model", apiModelInput), apiModelList);
  const paintApiImage = () => {
    const provider = thumbnailApiImageProvider(runnerPayload?.());
    const info = provider ? state.thumbnailApiImage?.providers?.[provider.name] : null;
    apiProviderLine.textContent = provider
      ? `Provider: ${provider.label}, with the key from the LLM Runner.`
      : "The LLM Runner's provider cannot draw pictures. Pick Gemini (google), Grok or OpenAI in the LLM Runner and add its key.";
    apiProviderLine.style.color = provider ? "#93c5fd" : "#fcd34d";
    apiModelList.replaceChildren(...(info?.models || []).map((model) => {
      const option = document.createElement("option");
      option.value = model;
      return option;
    }));
    apiModelInput.placeholder = info ? `Empty: ${info.default}` : "";
  };
  // The Overlay engine's choices: the color of the title, where it goes, and how strong the grade, the glow and the vignette are.
  const overlayColor = makeSelect(OVERLAY_COLOR_OPTIONS, intro.overlay.color);
  const overlayPosition = makeSelect(OVERLAY_POSITION_OPTIONS, intro.overlay.position);
  const overlayStrength = document.createElement("input");
  overlayStrength.type = "range";
  overlayStrength.min = "0";
  overlayStrength.max = "100";
  overlayStrength.step = "5";
  overlayStrength.value = String(intro.overlay.strength);
  overlayStrength.style.cssText = "width:100%;accent-color:#22d3ee;";
  overlayStrength.setAttribute("aria-label", "How strong the grade, the glow and the vignette are");
  const overlayStrengthField = makeField("Strength", overlayStrength);
  const paintOverlayStrength = () => {
    const label = overlayStrengthField.children[0];
    if (label) label.textContent = `Strength: ${overlayStrength.value}%  (the grade, the glow and the vignette; 0 leaves the frame alone)`;
  };
  overlayStrength.addEventListener("input", paintOverlayStrength);
  const overlayGroup = document.createElement("div");
  overlayGroup.style.cssText = COLUMN_CSS;
  overlayGroup.append(makeField("Title color", overlayColor), makeField("Title position", overlayPosition), overlayStrengthField);
  // What an engine that follows a prompt needs and Overlay does not: the style notes, the prompt and the LLM button.
  const styleInput = textArea(intro.style_notes, 2, "Optional look: neon city, moody blue and orange, glowing eyes, comic outline ...");
  const promptInput = textArea(intro.prompt, 7, "The edit prompt. Click Write prompt with LLM, or type your own.");
  const writeButton = makeButton("Write prompt with LLM");
  const styleField = makeField("Style notes", styleInput);
  const promptField = makeField("Edit prompt", promptInput);
  const kindOfEngine = () => thumbnailEngineKind(engineSelect.value);
  const paintEngine = () => {
    const overlayOn = engineSelect.value === "overlay";
    overlayGroup.style.display = overlayOn ? "flex" : "none";
    apiGroup.style.display = engineSelect.value === "api_image" ? "flex" : "none";
    styleField.style.display = overlayOn ? "none" : "flex";
    promptField.style.display = overlayOn ? "none" : "flex";
    writeButton.style.display = overlayOn ? "none" : "";
    if (overlayOn) engineHint.textContent = OVERLAY_HINT;
    else if (engineSelect.value === "qwen_image_edit") engineHint.textContent = QWEN_EDIT_HINT;
    else if (engineSelect.value === "api_image") engineHint.textContent = API_IMAGE_HINT;
    else engineHint.textContent = EDIT_HINT;
    paintApiImage();
    paintOverlayStrength();
  };
  // The two kinds want different prompts, so a prompt written for the other kind is flagged instead of being sent as it is.
  let promptKind = kindOfEngine();
  engineSelect.addEventListener("change", () => {
    paintEngine();
    if (kindOfEngine() !== promptKind) {
      promptKind = kindOfEngine();
      // Overlay has no prompt, so there is nothing to flag when it is chosen.
      if (promptInput.value.trim() && promptKind !== "overlay") {
        status.style.color = "#fcd34d";
        status.textContent = "This workflow needs a different kind of prompt. Click Write prompt with LLM again, or edit the one you have.";
      }
    }
  });
  const titleInput = makeInput(intro.title);
  titleInput.maxLength = 120;
  titleInput.placeholder = "Words on the thumbnail, for example NEVER LOOK BACK (leave empty for no text)";
  const generateButton = makeButton("Generate thumbnail", "primary");
  const status = document.createElement("div");
  status.style.cssText = "min-height:16px;font-size:12px;color:#93c5fd;white-space:pre-wrap;";

  const preview = document.createElement("img");
  preview.alt = "";
  preview.style.cssText = `${FRAME_CSS}display:none;`;
  const previewEmpty = document.createElement("div");
  previewEmpty.textContent = "No thumbnail yet.";
  previewEmpty.style.cssText = EMPTY_FRAME_CSS;
  const historyStrip = document.createElement("div");
  historyStrip.style.cssText = "display:flex;gap:6px;overflow-x:auto;";

  const enableBox = makeCheckbox("Make the thumbnail the first frame of the final video (one frame, no more)", intro.enabled);

  const current = () => normalizeThumbnailIntro(state.thumbnailIntro);
  const commit = (changes, { persist = true } = {}) => {
    state.thumbnailIntro = normalizeThumbnailIntro({ ...current(), ...changes });
    if (persist) Promise.resolve(save?.()).catch(() => {});
  };
  const readInputs = () => ({
    enabled: enableBox.input.checked,
    engine: engineSelect.value,
    overlay: normalizeThumbnailOverlay({ color: overlayColor.value, position: overlayPosition.value, strength: Number(overlayStrength.value) }),
    api_image: normalizeThumbnailApiImage({ model: apiModelInput.value }),
    title: titleInput.value,
    style_notes: styleInput.value,
    prompt: promptInput.value,
  });
  // The scene the chosen frame came from, for its lyrics.
  const sourceSegment = () => (state.segments || []).find((segment) => segment.id === current().scene_id) || null;

  const paintSource = () => {
    const saved = current();
    const has = Boolean(saved.source_path);
    sourceImage.style.display = has ? "block" : "none";
    sourceEmpty.style.display = has ? "none" : "flex";
    if (has) sourceImage.src = makeEditorImageUrl(saved.source_path);
    const segment = sourceSegment();
    const index = segment ? (state.segments || []).indexOf(segment) : -1;
    sourceCaption.textContent = has
      ? `${segment ? sceneLabel(segment, index) : "Scene"}  ·  ${formatThumbnailTime(saved.source_seconds)}  ·  frame ${saved.source_frame}  (${saved.source_kind === "frame" ? "video frame" : "scene image"})`
      : "";
  };
  const showImage = (path) => {
    const has = Boolean(path);
    preview.style.display = has ? "block" : "none";
    previewEmpty.style.display = has ? "none" : "flex";
    if (has) preview.src = makeEditorImageUrl(path);
  };
  const paintHistory = () => {
    historyStrip.replaceChildren();
    const { history, image_path: active } = current();
    for (const item of history.slice().reverse()) {
      const pick = document.createElement("img");
      pick.src = makeEditorImageUrl(item.path);
      pick.alt = "";
      pick.title = "Use this thumbnail";
      pick.style.cssText = `flex:0 0 auto;width:96px;aspect-ratio:16/9;object-fit:cover;border-radius:4px;cursor:pointer;border:2px solid ${item.path === active ? "#22d3ee" : "#3f3f46"};`;
      pick.onclick = () => {
        commit({ ...readInputs(), image_path: item.path, prompt: item.prompt || promptInput.value });
        promptInput.value = current().prompt;
        showImage(item.path);
        paintHistory();
      };
      historyStrip.append(pick);
    }
  };
  const setBusy = (value, text = "") => {
    busy = value;
    for (const control of [useButton, writeButton, generateButton, engineSelect, apiModelInput, overlayColor, overlayPosition, overlayStrength, titleInput, styleInput, promptInput]) {
      control.disabled = value;
    }
    status.style.color = "#93c5fd";
    status.textContent = text;
  };
  const fail = (error) => {
    setBusy(false);
    status.style.color = "#fca5a5";
    status.textContent = String(error?.message || error);
    toast(String(error?.message || error), true);
  };
  const requireSource = () => {
    if (current().source_path) return true;
    toast("Choose the frame first: scrub the video to the moment you want and click Use this frame.", true);
    return false;
  };

  useButton.onclick = async () => {
    if (busy) return;
    const item = sceneItemAt(frame);
    setBusy(true, item.video ? "Taking that frame of the video..." : "Using the scene image...");
    try {
      let kind = "image";
      let path = item.image;
      if (item.video) {
        const reply = await postJson("/vrgdg/music_builder/thumbnail/frame", {
          project_folder: projectFolder(),
          video_path: item.video,
          seconds: thumbnailLocalSeconds(item, frame, THUMBNAIL_FPS),
        }, THUMBNAIL_FRAME_TIMEOUT_MS);
        kind = "frame";
        path = reply.image_path;
      }
      commit({
        ...readInputs(), source_kind: kind, source_path: path, source_frame: frame, source_seconds: frame / THUMBNAIL_FPS,
        scene_id: item.segment.id,
      });
      paintSource();
      setBusy(false, "Frame chosen. Write the prompt with the LLM, or scrub again to pick a different moment.");
    } catch (error) {
      fail(error);
    }
  };

  writeButton.onclick = async () => {
    if (busy || !requireSource()) return;
    const saved = current();
    setBusy(true, "The LLM is looking at the frame and writing the prompt...");
    try {
      const reply = await postJson("/vrgdg/music_builder/thumbnail/prompt", {
        ...(runnerPayload?.() || {}),
        image_path: saved.source_path,
        engine_kind: kindOfEngine(),
        title: titleInput.value,
        style_notes: styleInput.value,
        project_name: projectFolder().split(/[\\/]/).filter(Boolean).pop() || "",
        scene_lyrics: String(sourceSegment()?.lyric_text || sourceSegment()?.lyrics || ""),
      }, THUMBNAIL_PROMPT_TIMEOUT_MS);
      // The title text is mandatory: the LLM may have left it out or spelled it differently, so it is made certain here.
      promptInput.value = ensureThumbnailTitle(String(reply.prompt || ""), titleInput.value, kindOfEngine(), styleInput.value);
      commit(readInputs());
      setBusy(false, `Prompt ready (${promptInput.value.length} characters). Edit it if you like, then Generate.`);
    } catch (error) {
      fail(error);
    }
  };

  // The Overlay engine: the Builder keeps the frame and puts the layers on it (no prompt, no LLM, no ComfyUI) and saves the result.
  const makeOverlayThumbnail = async () => {
    commit(readInputs());
    setBusy(true, "Putting the layers on the frame...");
    let progress = null;
    try {
      progress = createProgressWindow?.("Creating the thumbnail");
      progress?.set("Putting the layers on the frame...", 50);
      const { url, payload } = thumbnailOverlayRequest({
        sourcePath: current().source_path, title: titleInput.value, overlay: current().overlay, projectFolder: projectFolder(),
      });
      const saved = await postJson(url, payload);
      const entry = { path: saved.image_path, prompt: "", engine: "overlay", scene_id: current().scene_id };
      commit({ ...readInputs(), image_path: saved.image_path, history: thumbnailHistoryWith(current().history, entry) });
      showImage(saved.image_path);
      paintHistory();
      progress?.set("Thumbnail ready.", 100);
      progress?.close?.(600);
      setBusy(false, titleInput.value.trim()
        ? "Thumbnail ready. Change the color, position or strength and make it again, or pick an earlier one below."
        : "Thumbnail ready, with no title text because the title is empty.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      fail(error);
    }
  };

  generateButton.onclick = async () => {
    if (busy || !requireSource()) return;
    if (engineSelect.value === "overlay") {
      await makeOverlayThumbnail();
      return;
    }
    const typed = promptInput.value.trim();
    if (!typed) {
      toast("Write the edit prompt first: click Write prompt with LLM or type one.", true);
      return;
    }
    // The title text is mandatory: a prompt that lost it (or never had it) gets it back before anything is sent, and the box shows it.
    const prompt = ensureThumbnailTitle(typed, titleInput.value, kindOfEngine(), styleInput.value);
    if (prompt !== typed) {
      promptInput.value = prompt;
      toast("The title text was missing from the prompt, so it was added.");
    }
    commit(readInputs());
    setBusy(true, "Starting...");
    let progress = null;
    try {
      progress = createProgressWindow?.("Creating the thumbnail");
      const onStatus = (message) => {
        status.textContent = message;
        progress?.set(message, 50);
      };
      let saved;
      if (engineSelect.value === "api_image") {
        // The pack calls the provider's image API itself and keeps the picture in the project, so there is no ComfyUI queue to wait on.
        const { url, payload } = thumbnailApiImageRequest({
          prompt, sourcePath: current().source_path, projectFolder: projectFolder(), runner: runnerPayload?.(), model: apiModelInput.value,
        });
        onStatus(`Asking ${thumbnailApiImageProvider(runnerPayload?.()).label} to draw the thumbnail...`);
        saved = await postJson(url, payload, THUMBNAIL_API_IMAGE_TIMEOUT_MS);
      } else {
        const image = await generateThumbnailImage({
          engine: engineSelect.value, prompt, sourcePath: current().source_path, settings: settingsFor(engineSelect.value), onStatus,
        });
        onStatus("Saving the thumbnail in the project...");
        saved = await postJson("/vrgdg/music_builder/thumbnail/save", { project_folder: projectFolder(), image });
      }
      const entry = { path: saved.image_path, prompt, engine: engineSelect.value, scene_id: current().scene_id };
      commit({ ...readInputs(), image_path: saved.image_path, history: thumbnailHistoryWith(current().history, entry) });
      showImage(saved.image_path);
      paintHistory();
      progress?.set("Thumbnail ready.", 100);
      progress?.close?.(600);
      setBusy(false, "Thumbnail ready. Generate again for another take, or pick an earlier one below.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      fail(error);
    }
  };

  const closeButton = makeButton("Done", "primary");
  closeButton.onclick = () => {
    if (busy && !window.confirm("A thumbnail is still being made. Close this window anyway? It keeps running in the background.")) return;
    commit(readInputs());
    video.pause();
    backdrop.remove();
  };

  // Layout: the picker (player on the left, the frame in use on the right), then the prompt and the result, then the footer.
  const pickerGrid = document.createElement("div");
  pickerGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:14px;align-items:start;";
  const playerColumn = document.createElement("div");
  playerColumn.style.cssText = COLUMN_CSS;
  playerColumn.append(sectionTitle(canFrame ? "1. Pick the moment" : "1. Pick the scene"), video, still, scrub, readout, stepRow);
  const sourceColumn = document.createElement("div");
  sourceColumn.style.cssText = COLUMN_CSS;
  sourceColumn.append(sectionTitle("The frame in use"), sourceEmpty, sourceImage, sourceCaption);
  pickerGrid.append(playerColumn, sourceColumn);

  const settingsGrid = document.createElement("div");
  settingsGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:14px;align-items:start;";
  const left = document.createElement("div");
  left.style.cssText = COLUMN_CSS;
  const buttons = document.createElement("div");
  buttons.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
  buttons.append(writeButton, generateButton);
  left.append(sectionTitle("2. Choose how, 3. Make the thumbnail"), makeField("Image workflow", engineSelect), engineHint,
    makeField("Title text", titleInput), overlayGroup, apiGroup, styleField, promptField, buttons, status);
  const right = document.createElement("div");
  right.style.cssText = COLUMN_CSS;
  right.append(sectionTitle("The thumbnail"), previewEmpty, preview, historyStrip);
  settingsGrid.append(left, right);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;gap:14px;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;border-top:1px solid #1e293b;padding-top:10px;";
  const footerLeft = document.createElement("div");
  footerLeft.style.cssText = "display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;";
  footerLeft.append(enableBox.wrapper);
  footer.append(footerLeft, closeButton);

  box.append(title, note, pickerGrid, settingsGrid, footer);
  backdrop.append(box);
  document.body.append(backdrop);
  showFrame(frame);
  paintSource();
  paintEngine();
  showImage(intro.image_path);
  paintHistory();
  return backdrop;
}

// Shows the Thumbnail button when this build can make thumbnails, and opens the dialog when it is clicked.
export function wireThumbnail({ button, state, ...deps }) {
  if (!button) return;
  loadThumbnailCapabilities(state).then((available) => {
    // inline-flex is the toolbar button's own layout (styleCompactToolbarButton), so showing it restores that, not the default.
    button.style.display = available ? "inline-flex" : "none";
  });
  button.onclick = () => {
    openThumbnailDialog({ state, ...deps });
  };
}
