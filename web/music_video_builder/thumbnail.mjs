// Thumbnail intro. The Thumbnail button opens a dialog: pick a scene, let the Builder's LLM look at its image and write a
// YouTube-style edit prompt, send the image through NanoBanana or Flux/Klein, keep the result. When the intro is on, the
// final stitch puts the picture in front of the video for a few seconds (`POST /vrgdg/music_builder/thumbnail/prepend`).
// The scenes, the timeline and the audio sync are never touched. The pure parts are in thumbnail_state.mjs.

import { api } from "../../../scripts/api.js";
import { makeEditorImageUrl, postJson, queueWorkflowPrompt, waitForImages } from "./comfy_api.mjs";
import { BUILDER_FONT_STACK, DEFAULT_NB_IMAGE_MODEL } from "./constants.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, toast } from "./controls.mjs";
import { fluxKleinLoraPayload } from "./image_generation.mjs";
import {
  clampThumbnailSeconds, normalizeThumbnailIntro, thumbnailHistoryWith, thumbnailIntroForStitch,
  THUMBNAIL_MAX_SECONDS, THUMBNAIL_MIN_SECONDS,
} from "./thumbnail_state.mjs";

const THUMBNAIL_PREPEND_TIMEOUT_MS = 30 * 60 * 1000;
const THUMBNAIL_PROMPT_TIMEOUT_MS = 10 * 60 * 1000;
const THUMBNAIL_WIDTH = 1280;
const THUMBNAIL_HEIGHT = 720;
const ENGINE_LABELS = [
  { value: "nano_banana", label: "NanoBanana (API)" },
  { value: "flux_klein", label: "Flux/Klein (local)" },
];

// Asks whether this build can make thumbnails. A build without the routes answers 404, which is an answer here and not an
// error for the project's error log, so the button stays hidden and nothing else changes.
export async function loadThumbnailCapabilities(state) {
  try {
    const response = await api.fetchApi("/vrgdg/music_builder/thumbnail/capabilities", { timeoutMs: null });
    const data = response.ok ? await response.json().catch(() => null) : null;
    state.thumbnailAvailable = Boolean(data && data.ok !== false && data.version);
  } catch {
    state.thumbnailAvailable = false;
  }
  return state.thumbnailAvailable;
}

// `stitched` is what the stitch route returned. It comes back unchanged apart from `thumbnail_intro`, and `final_video_path` now
// names the video with the picture in front. When it cannot be added the plain stitched video stays and the reason is shown.
export async function applyThumbnailIntroAfterStitch(state, stitched, { progress = null, projectFolder = "" } = {}) {
  const intro = thumbnailIntroForStitch(state);
  const path = String(stitched?.final_video_path || "");
  if (!intro || !path) return stitched;
  progress?.set("Adding the thumbnail to the front of the video...", 98);
  try {
    const reply = await postJson("/vrgdg/music_builder/thumbnail/prepend", {
      input_path: path,
      image_path: intro.image_path,
      seconds: intro.seconds,
      replace_source: true,
      project_folder: projectFolder,
    }, THUMBNAIL_PREPEND_TIMEOUT_MS);
    return { ...stitched, thumbnail_intro: reply };
  } catch (error) {
    toast(`The thumbnail could not be added, so the video starts with the first scene. ${String(error?.message || error)}`, true);
    return stitched;
  }
}

// Sends the scene image through the chosen image workflow and returns the ComfyUI view descriptor of the result.
async function generateThumbnailImage({ state, engine, prompt, sourcePath, onStatus }) {
  const ingredients = [{ path: sourcePath }];
  let built;
  if (engine === "nano_banana") {
    const settings = state.nbImageSettings || {};
    if (!String(settings.api_key || "").trim()) {
      throw new Error("The NanoBanana API key is missing. Enter it in the NanoBanana image settings, or choose Flux/Klein.");
    }
    onStatus("Building the NanoBanana workflow...");
    built = await postJson("/vrgdg/workflow_runner/build_nb_image_prompt", {
      api_key: settings.api_key,
      model: settings.model || DEFAULT_NB_IMAGE_MODEL,
      prompt,
      image_ingredients: ingredients,
    });
  } else {
    const settings = state.fluxKleinSettings || {};
    onStatus("Building the Flux/Klein workflow...");
    built = await postJson("/vrgdg/workflow_runner/build_flux_klein_prompt", {
      prompt,
      image_ingredients: ingredients,
      unet_name: settings.unet_name || "",
      clip_name: settings.clip_name || "",
      vae_name: settings.vae_name || "",
      width: THUMBNAIL_WIDTH,
      height: THUMBNAIL_HEIGHT,
      seed: Math.floor(Math.random() * 2 ** 31),
      ...fluxKleinLoraPayload(settings),
    });
  }
  onStatus("Queueing the image workflow...");
  const queued = await queueWorkflowPrompt(built.prompt);
  const promptId = queued?.prompt_id;
  if (!promptId) throw new Error("ComfyUI queued the thumbnail image but did not return a prompt_id.");
  const images = await waitForImages(promptId, (message) => onStatus(message));
  const image = images[images.length - 1];
  if (!image) throw new Error("The image workflow finished without an image.");
  return image;
}

const PANEL_CSS = `width:min(980px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;box-sizing:border-box;border:1px solid #155e75;border-radius:10px;background:#0f172a;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;

function textArea(value, rows, placeholder) {
  const area = document.createElement("textarea");
  area.rows = rows;
  area.value = value;
  area.placeholder = placeholder;
  area.style.cssText = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;line-height:1.45;resize:vertical;`;
  return area;
}

// The dialog. `deps` come from the Builder: state, projectInput, runnerPayload() (the LLM Runner settings), sceneImagePath(segment),
// sceneLabel(segment, index), save() (a quiet autosave) and createProgressWindow(title).
export function openThumbnailDialog(deps) {
  const { state, projectInput, runnerPayload, sceneImagePath, sceneLabel, save, createProgressWindow } = deps;
  const projectFolder = () => String(projectInput?.value || state.projectFolder || "").trim();
  if (!projectFolder()) {
    toast("Create or load a project before making a thumbnail.", true);
    return null;
  }
  const scenes = (state.segments || [])
    .map((segment, index) => ({ segment, index, path: String(sceneImagePath(segment) || "").trim() }))
    .filter((item) => item.path);
  if (!scenes.length) {
    toast("No scene has an image yet. Make or choose a scene image first, then open Thumbnail.", true);
    return null;
  }
  const intro = normalizeThumbnailIntro(state.thumbnailIntro);
  let busy = false;

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
  note.textContent = "Pick a scene. The LLM looks at its image and writes the edit prompt, then NanoBanana or Flux/Klein turns it into a thumbnail. When it is on, the final video starts with the thumbnail.";
  note.style.cssText = "font-size:12px;line-height:1.45;color:#cbd5e1;";

  // Scene picker: one tile per scene that has an image.
  const strip = document.createElement("div");
  strip.style.cssText = "display:flex;gap:8px;overflow-x:auto;padding-bottom:6px;";
  const tiles = new Map();
  let selected = scenes.find((item) => item.segment.id === intro.scene_id) || scenes[0];
  const paintTiles = () => {
    for (const [id, tile] of tiles) tile.style.borderColor = id === selected.segment.id ? "#22d3ee" : "#3f3f46";
  };
  for (const item of scenes) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.title = sceneLabel(item.segment, item.index);
    tile.style.cssText = "flex:0 0 auto;width:132px;border:2px solid #3f3f46;border-radius:6px;background:#18181b;color:#e2e8f0;padding:4px;cursor:pointer;font-size:11px;font-family:inherit;";
    const picture = document.createElement("img");
    picture.src = makeEditorImageUrl(item.path);
    picture.alt = "";
    picture.style.cssText = "display:block;width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:3px;";
    const label = document.createElement("div");
    label.textContent = sceneLabel(item.segment, item.index);
    label.style.cssText = "margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    tile.append(picture, label);
    tile.onclick = () => {
      selected = item;
      paintTiles();
    };
    tiles.set(item.segment.id, tile);
    strip.append(tile);
  }
  paintTiles();

  const engineSelect = makeSelect(ENGINE_LABELS, intro.engine);
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
  preview.style.cssText = "display:none;width:100%;aspect-ratio:16/9;object-fit:contain;border:1px solid #3f3f46;border-radius:6px;background:#09090b;";
  const previewEmpty = document.createElement("div");
  previewEmpty.textContent = "No thumbnail yet.";
  previewEmpty.style.cssText = "display:flex;align-items:center;justify-content:center;width:100%;aspect-ratio:16/9;border:1px dashed #3f3f46;border-radius:6px;color:#71717a;font-size:12px;";
  const historyStrip = document.createElement("div");
  historyStrip.style.cssText = "display:flex;gap:6px;overflow-x:auto;";

  const enableBox = makeCheckbox("Add the thumbnail to the front of the final video", intro.enabled);
  const secondsInput = makeInput(String(intro.seconds), "number");
  secondsInput.min = String(THUMBNAIL_MIN_SECONDS);
  secondsInput.max = String(THUMBNAIL_MAX_SECONDS);
  secondsInput.step = "0.5";
  secondsInput.style.width = "90px";
  const secondsField = makeField("Seconds shown", secondsInput);

  const current = () => normalizeThumbnailIntro(state.thumbnailIntro);
  const commit = (changes, { persist = true } = {}) => {
    state.thumbnailIntro = normalizeThumbnailIntro({ ...current(), ...changes });
    if (persist) Promise.resolve(save?.()).catch(() => {});
  };
  const readInputs = () => ({
    enabled: enableBox.input.checked,
    seconds: clampThumbnailSeconds(secondsInput.value),
    engine: engineSelect.value,
    title: titleInput.value,
    style_notes: styleInput.value,
    prompt: promptInput.value,
    scene_id: selected.segment.id,
  });

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
    for (const control of [writeButton, generateButton, engineSelect, titleInput, styleInput, promptInput]) control.disabled = value;
    status.style.color = "#93c5fd";
    status.textContent = text;
  };
  const fail = (error) => {
    setBusy(false);
    status.style.color = "#fca5a5";
    status.textContent = String(error?.message || error);
    toast(String(error?.message || error), true);
  };

  writeButton.onclick = async () => {
    if (busy) return;
    setBusy(true, "The LLM is looking at the scene image and writing the prompt...");
    try {
      const reply = await postJson("/vrgdg/music_builder/thumbnail/prompt", {
        ...(runnerPayload?.() || {}),
        image_path: selected.path,
        title: titleInput.value,
        style_notes: styleInput.value,
        project_name: projectFolder().split(/[\\/]/).filter(Boolean).pop() || "",
        scene_lyrics: String(selected.segment.lyric_text || selected.segment.lyrics || ""),
      }, THUMBNAIL_PROMPT_TIMEOUT_MS);
      promptInput.value = String(reply.prompt || "");
      commit(readInputs());
      setBusy(false, `Prompt ready (${reply.characters || promptInput.value.length} characters). Edit it if you like, then Generate.`);
    } catch (error) {
      fail(error);
    }
  };

  generateButton.onclick = async () => {
    if (busy) return;
    const prompt = promptInput.value.trim();
    if (!prompt) {
      toast("Write the edit prompt first: click Write prompt with LLM or type one.", true);
      return;
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
      const image = await generateThumbnailImage({ state, engine: engineSelect.value, prompt, sourcePath: selected.path, onStatus });
      onStatus("Saving the thumbnail in the project...");
      const saved = await postJson("/vrgdg/music_builder/thumbnail/save", { project_folder: projectFolder(), image });
      const entry = { path: saved.image_path, prompt, engine: engineSelect.value, scene_id: selected.segment.id };
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
    backdrop.remove();
  };

  const settingsGrid = document.createElement("div");
  settingsGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:14px;align-items:start;";
  const left = document.createElement("div");
  left.style.cssText = "display:flex;flex-direction:column;gap:10px;min-width:0;";
  const buttons = document.createElement("div");
  buttons.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
  buttons.append(writeButton, generateButton);
  left.append(makeField("Image workflow", engineSelect), makeField("Title text", titleInput),
    makeField("Style notes", styleInput), makeField("Edit prompt", promptInput), buttons, status);
  const right = document.createElement("div");
  right.style.cssText = "display:flex;flex-direction:column;gap:8px;min-width:0;";
  right.append(previewEmpty, preview, historyStrip);
  settingsGrid.append(left, right);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;gap:14px;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;border-top:1px solid #1e293b;padding-top:10px;";
  const footerLeft = document.createElement("div");
  footerLeft.style.cssText = "display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;";
  footerLeft.append(enableBox.wrapper, secondsField);
  footer.append(footerLeft, closeButton);

  box.append(title, note, strip, settingsGrid, footer);
  backdrop.append(box);
  document.body.append(backdrop);
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
