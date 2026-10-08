import { makeButton, makeField, makeSelect, toast } from "./controls.mjs";
import { MINIMAX_I2V_TRANSITION_OPTIONS } from "./minimax_h3.mjs";
import { postJson } from "./comfy_api.mjs";
import { miniMaxI2VFrameMode, miniMaxI2VLastFrame } from "./minimax_keyframe_state.mjs";

export function createMiniMaxKeyframes() {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:9px;";
  const mode = makeSelect([{value: "normal", label: "Normal"}, {value: "flf", label: "First / Last Frame (FLF)"}], "normal");
  const note = document.createElement("div");
  note.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  note.textContent = "Frames belong to this scene. Generate images in the Image tab, then choose them below, or upload images. Single and 2 Pass use the same frames.";
  const cards = {};
  for (const role of ["first", "last"]) {
    const card = document.createElement("div");
    card.style.cssText = "display:flex;flex-direction:column;gap:6px;border:1px solid #3f3f46;border-radius:6px;padding:8px;min-width:0;";
    const preview = document.createElement("img");
    preview.alt = `${role} frame`;
    preview.style.cssText = "width:100%;height:110px;object-fit:contain;background:#09090b;display:none;";
    const label = document.createElement("div");
    label.style.cssText = "font-size:11px;color:#cbd5e1;overflow-wrap:anywhere;";
    const existing = makeSelect([{value:"",label:"Choose existing image…"}], "");
    const upload = makeButton(`Upload ${role} frame`);
    const clear = makeButton("Clear last frame");
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/png,image/jpeg,image/webp";
    file.style.display = "none";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
    actions.append(upload);
    if (role === "last") actions.append(clear);
    card.append(makeField(role === "first" ? "First frame — scene image" : "Last frame — this scene only", preview), label, existing, actions, file);
    wrapper.append(card);
    cards[role] = {card, preview, label, existing, upload, clear, file};
  }
  const transitionStyle = makeSelect(MINIMAX_I2V_TRANSITION_OPTIONS, "natural");
  const transitionDirection = document.createElement("textarea");
  transitionDirection.rows = 3;
  transitionDirection.placeholder = "Optional: describe how the first image becomes the last…";
  transitionDirection.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;";
  const transitionScope = document.createElement("div");
  transitionScope.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const transitionSection = document.createElement("div");
  transitionSection.style.cssText = "display:flex;flex-direction:column;gap:6px;";
  transitionSection.append(makeField("FLF transition style", transitionStyle), makeField("Transition direction (optional)", transitionDirection), transitionScope);
  wrapper.insertBefore(transitionSection, cards.last.card);
  wrapper.prepend(makeField("I2V mode — this scene", mode), note);
  return {wrapper, mode, cards, transitionStyle, transitionDirection, transitionScope, transitionSection, sync: () => {}};
}

export function wireMiniMaxKeyframes({controls, activeSegment, segmentImageSource, allEditableSegments,
  loadCustomImageFile, loadFirstLastFrameEndFile, projectFolder, sceneSlotNumber, pushHistory,
  autoSaveSessionQuiet, syncMiniMaxH3Panel, renderList, renderTimeline = () => {}}) {
  let shownSegment = null;
  let choices = [];
  const save = async (segment, role, source) => {
    const folder = projectFolder();
    if (!folder) throw new Error("Save the project before choosing keyframes.");
    const archived = await postJson("/vrgdg/music_builder/archive_scene_image", {
      project_folder: folder, scene_number: sceneSlotNumber(segment),
      ...(source.path ? {source_path: source.path} : {image_data: source.data}),
    });
    pushHistory();
    if (role === "first") {
      segment.custom_image_path = archived.saved_path;
      segment.custom_image_data = "";
      segment.approved_image_path = "";
      segment.image_history = [...(segment.image_history || []), archived.saved_path];
      segment.image_history_index = segment.image_history.length - 1;
    } else {
      segment.first_last_frame_end_image_path = archived.saved_path;
      segment.first_last_frame_end_image_data = "";
      segment.first_last_frame_end_image_name = source.name || "last_frame.png";
      segment.minimax_h3_i2v_frame_mode = "flf";
    }
    await autoSaveSessionQuiet(`MiniMax ${role} frame selected`);
    renderList();
    renderTimeline();
    syncMiniMaxH3Panel();
  };
  controls.sync = (segment, mode, settings = {}) => {
    if (controls.transitionSection) {
      controls.transitionSection.style.display = mode === "image_to_video" && (!segment || miniMaxI2VFrameMode(segment) === "flf") ? "flex" : "none";
      controls.transitionStyle.value = settings.i2v_transition_style || "natural";
      controls.transitionDirection.value = settings.i2v_transition_direction || "";
      controls.transitionScope.textContent = (segment?.use_scene_minimax_h3_settings
        ? "Locked scene override. " : "Global: applies to all unlocked FLF scenes. ")
        + "Regenerate Create Prompt after changing the transition. Frames remain per scene.";
    }
    shownSegment = segment;
    const enabled = Boolean(segment) && mode === "image_to_video";
    controls.mode.disabled = !enabled;
    controls.mode.value = miniMaxI2VFrameMode(segment);
    choices = [];
    const seen = new Set();
    for (const item of allEditableSegments()) {
      const sources = [segmentImageSource(item), ...(item.image_history || []).map(path => ({path}))];
      if (item.first_last_frame_end_image_path) sources.push({path:item.first_last_frame_end_image_path});
      for (const source of sources) {
        const key = source?.path || source?.data;
        if (!key || seen.has(key)) continue;
        seen.add(key);
        choices.push({...source, name: `${item.label || "Scene"} — ${source.path?.split(/[\\/]/).pop() || "uploaded image"}`});
      }
    }
    for (const [role, card] of Object.entries(controls.cards)) {
      const source = role === "first" ? segmentImageSource(segment) : miniMaxI2VLastFrame(segment);
      card.card.style.display = role === "last" && controls.mode.value !== "flf" ? "none" : "flex";
      const url = source?.data || (source?.path ? `/vrgdg/video_editor/image?path=${encodeURIComponent(source.path)}&cb=${segment?.video_cache_bust || 0}` : "");
      if (url) card.preview.src = url; else card.preview.removeAttribute("src");
      card.preview.style.display = url ? "block" : "none";
      card.label.textContent = !segment ? "Select a scene to assign frames." : source?.path?.split(/[\\/]/).pop() || (source?.data ? "Uploaded image" : `Choose a ${role}-frame image.`);
      card.existing.replaceChildren(new Option("Choose existing image…", ""), ...choices.map((item, index) => new Option(item.name, String(index))));
      card.existing.disabled = !enabled || !choices.length;
      card.upload.disabled = !enabled;
      card.clear.style.display = role === "last" && url ? "" : "none";
    }
  };
  controls.mode.onchange = async () => {
    if (!shownSegment) return;
    pushHistory();
    shownSegment.minimax_h3_i2v_frame_mode = controls.mode.value;
    await autoSaveSessionQuiet("MiniMax I2V frame mode");
    renderList();
    renderTimeline();
    syncMiniMaxH3Panel();
  };
  for (const [role, card] of Object.entries(controls.cards)) {
    let uploadSegment = null;
    card.upload.onclick = () => { uploadSegment = shownSegment || activeSegment(); card.file.click(); };
    card.file.onchange = async () => {
      try {
        const file = card.file.files?.[0];
        if (!file || !uploadSegment) return;
        if (!projectFolder()) throw new Error("Save the project before uploading keyframes.");
        if (role === "first") await loadCustomImageFile(file, uploadSegment);
        else await loadFirstLastFrameEndFile(file, uploadSegment);
        renderTimeline();
        syncMiniMaxH3Panel();
      } catch (error) { toast(String(error?.message || error), true); }
      finally { card.file.value = ""; }
    };
    card.existing.onchange = async () => {
      const segment = shownSegment;
      const source = card.existing.value === "" ? null : choices[Number(card.existing.value)];
      if (!segment || !source) return;
      try { await save(segment, role, source); }
      catch (error) { toast(String(error?.message || error), true); }
    };
    card.clear.onclick = async () => {
      if (!shownSegment) return;
      pushHistory();
      for (const key of ["path", "data", "name"]) shownSegment[`first_last_frame_end_image_${key}`] = "";
      await autoSaveSessionQuiet("MiniMax last frame cleared");
      renderList();
      renderTimeline();
      syncMiniMaxH3Panel();
    };
  }
}
