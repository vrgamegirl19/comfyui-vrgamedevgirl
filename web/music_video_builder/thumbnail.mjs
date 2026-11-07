// Thumbnail intro. The Thumbnail button opens a dialog: scrub the video to the exact moment you want and take that frame, let the
// Builder's LLM look at it and write a YouTube-style edit prompt, send the frame through NanoBanana or Flux/Klein, keep the result.
// When the intro is on, the final stitch puts the picture in front of the video for a few seconds
// (`POST /vrgdg/music_builder/thumbnail/prepend`). The scenes, the timeline and the audio sync are never touched. The pure parts
// (the saved settings, the timeline of frames) are in thumbnail_state.mjs.

import { api } from "../../../scripts/api.js";
import { makeEditorImageUrl, makeEditorVideoUrl, postJson, queueWorkflowPrompt, waitForImages } from "./comfy_api.mjs";
import { BUILDER_FONT_STACK, DEFAULT_NB_IMAGE_MODEL } from "./constants.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, toast } from "./controls.mjs";
import { fluxKleinLoraPayload } from "./image_generation.mjs";
import { loadMossReferenceSettings } from "./moss_settings.mjs";
import { THUMBNAIL_ENGINE_LABELS, thumbnailEngineRequest } from "./thumbnail_engines.mjs";
import {
  clampThumbnailChange, ensureThumbnailTitle, formatThumbnailTime, normalizeThumbnailIntro, thumbnailEngineKind, thumbnailFrameRange,
  thumbnailHistoryWith, thumbnailIntroForStitch, thumbnailItemAt, thumbnailLocalSeconds, thumbnailTimelineItems, THUMBNAIL_ENGINES,
  THUMBNAIL_FPS,
} from "./thumbnail_state.mjs";

const THUMBNAIL_PREPEND_TIMEOUT_MS = 30 * 60 * 1000;
const THUMBNAIL_PROMPT_TIMEOUT_MS = 10 * 60 * 1000;
const THUMBNAIL_FRAME_TIMEOUT_MS = 2 * 60 * 1000;
// Where inside a frame's time slot the player is parked, so it shows that frame and not the one before it.
const FRAME_SEEK_OFFSET = 0.25;
const ENGINE_OPTIONS = THUMBNAIL_ENGINES.map((value) => ({ value, label: THUMBNAIL_ENGINE_LABELS[value] }));
const EDIT_HINT = "This workflow edits the frame by instruction, so the prompt tells it what to change.";
const RESTYLE_HINT = "This workflow restyles the frame from a description of the finished picture. The prompt describes the thumbnail, and Change sets how far it moves from the frame.";
const MOSS_EDIT_HINT = "MOSS-Image Edit changes the frame by a short instruction (the Edit LoRA under Models > MOSS-Image, and ComfyUI 0.43 or newer). It keeps the picture and the composition and does what the prompt adds, including the title text.";
const MOSS_RESTYLE_HINT = `${RESTYLE_HINT} A restyle keeps the frame's composition and does not draw the title text reliably: use MOSS-Image Edit when the thumbnail needs text.`;
// The settings of the two MOSS-Image workflows are the ones saved by the MOSS-Image tab of Models > Select models.
const MOSS_ENGINES = ["moss_image", "moss_image_edit"];

// Asks whether this build can make thumbnails, and whether it can take one exact frame of a video. A build without the routes
// answers 404, which is an answer here and not an error for the project's error log, so the button stays hidden and nothing
// else changes.
export async function loadThumbnailCapabilities(state) {
  try {
    const response = await api.fetchApi("/vrgdg/music_builder/thumbnail/capabilities", { timeoutMs: null });
    const data = response.ok ? await response.json().catch(() => null) : null;
    state.thumbnailAvailable = Boolean(data && data.ok !== false && data.version);
    state.thumbnailFrameAvailable = Boolean(state.thumbnailAvailable && data.frame);
  } catch {
    state.thumbnailAvailable = false;
    state.thumbnailFrameAvailable = false;
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
async function generateThumbnailImage({ engine, prompt, sourcePath, change, settings, onStatus }) {
  const { url, payload, label } = thumbnailEngineRequest({
    engine, prompt, sourcePath, change, settings,
    seed: Math.floor(Math.random() * 2 ** 31),
    loraPayload: fluxKleinLoraPayload,
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
  // The settings an image workflow runs with: the Builder's own for it (MOSS-Image's live in the Reference Builder's saved choices).
  const settingsFor = (engine) => (MOSS_ENGINES.includes(engine) ? loadMossReferenceSettings() : engineSettings?.(engine)) || {};
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
    ? "1. Scrub the video to the exact moment you want and click Use this frame. 2. The LLM looks at that frame and writes the edit prompt. 3. NanoBanana or Flux/Klein turns it into a thumbnail. When it is on, the final video starts with the thumbnail."
    : "This build cannot take a frame of a video, so pick the scene whose image you want to start from. The LLM looks at it and writes the edit prompt, and NanoBanana or Flux/Klein turns it into a thumbnail.";
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

  // Step 2 and 3: the prompt and the picture.
  const engineSelect = makeSelect(ENGINE_OPTIONS, intro.engine);
  const engineHint = document.createElement("div");
  engineHint.style.cssText = "font-size:11px;line-height:1.4;color:#94a3b8;";
  // How far a restyle workflow may move away from the frame. Edit workflows follow an instruction instead, so it is hidden for them.
  const changeInput = document.createElement("input");
  changeInput.type = "range";
  changeInput.min = "0";
  changeInput.max = "100";
  changeInput.step = "5";
  changeInput.value = String(Math.round(intro.change * 100));
  changeInput.style.cssText = "width:100%;accent-color:#22d3ee;";
  changeInput.setAttribute("aria-label", "How much to change the frame");
  const changeField = makeField("Change", changeInput);
  const paintChange = () => {
    const label = changeField.children[0];
    if (label) label.textContent = `Change: ${changeInput.value}%  (lower keeps the frame, higher restyles more)`;
  };
  const kindOfEngine = () => thumbnailEngineKind(engineSelect.value);
  const paintEngine = () => {
    const restyle = kindOfEngine() === "restyle";
    changeField.style.display = restyle ? "flex" : "none";
    if (engineSelect.value === "moss_image_edit") engineHint.textContent = MOSS_EDIT_HINT;
    else if (engineSelect.value === "moss_image") engineHint.textContent = MOSS_RESTYLE_HINT;
    else engineHint.textContent = restyle ? RESTYLE_HINT : EDIT_HINT;
    paintChange();
  };
  changeInput.addEventListener("input", paintChange);
  // The two kinds want different prompts, so a prompt written for the other kind is flagged instead of being sent as it is.
  let promptKind = kindOfEngine();
  engineSelect.addEventListener("change", () => {
    paintEngine();
    if (kindOfEngine() !== promptKind) {
      promptKind = kindOfEngine();
      if (promptInput.value.trim()) {
        status.style.color = "#fcd34d";
        status.textContent = "This workflow needs a different kind of prompt. Click Write prompt with LLM again, or edit the one you have.";
      }
    }
  });
  const titleInput = makeInput(intro.title);
  titleInput.maxLength = 120;
  titleInput.placeholder = "Words on the thumbnail, for example NEVER LOOK BACK (leave empty for no text)";
  const styleInput = textArea(intro.style_notes, 2, "Optional look: neon city, moody blue and orange, glowing eyes, comic outline ...");
  const promptInput = textArea(intro.prompt, 7, "The edit prompt. Click Write prompt with LLM, or type your own.");
  const writeButton = makeButton("Write prompt with LLM");
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
    change: clampThumbnailChange(Number(changeInput.value) / 100),
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
    for (const control of [useButton, writeButton, generateButton, engineSelect, changeInput, titleInput, styleInput, promptInput]) control.disabled = value;
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
      promptInput.value = ensureThumbnailTitle(String(reply.prompt || ""), titleInput.value, kindOfEngine());
      commit(readInputs());
      setBusy(false, `Prompt ready (${promptInput.value.length} characters). Edit it if you like, then Generate.`);
    } catch (error) {
      fail(error);
    }
  };

  generateButton.onclick = async () => {
    if (busy || !requireSource()) return;
    const typed = promptInput.value.trim();
    if (!typed) {
      toast("Write the edit prompt first: click Write prompt with LLM or type one.", true);
      return;
    }
    // The title text is mandatory: a prompt that lost it (or never had it) gets it back before anything is sent, and the box shows it.
    const prompt = ensureThumbnailTitle(typed, titleInput.value, kindOfEngine());
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
      const image = await generateThumbnailImage({
        engine: engineSelect.value, prompt, sourcePath: current().source_path, change: current().change,
        settings: settingsFor(engineSelect.value), onStatus,
      });
      onStatus("Saving the thumbnail in the project...");
      const saved = await postJson("/vrgdg/music_builder/thumbnail/save", { project_folder: projectFolder(), image });
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
  left.append(sectionTitle("2. Write the prompt, 3. Make the thumbnail"), makeField("Image workflow", engineSelect), engineHint, changeField,
    makeField("Title text", titleInput), makeField("Style notes", styleInput), makeField("Edit prompt", promptInput), buttons, status);
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
