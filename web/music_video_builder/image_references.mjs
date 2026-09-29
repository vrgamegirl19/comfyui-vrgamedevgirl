import { makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import { makeMiniButton, toast } from "./controls.mjs";
import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import { cloneI2VVideoSettings } from "./model_settings.mjs";
import { expandSubjectReferencesForRender, normalizeFluxReferenceBuilder } from "./reference_data.mjs";

function renderFluxIngredientRows(listElement, ingredients, emptyText, onRemove) {
  listElement.innerHTML = "";
  if (!ingredients.length) {
    const empty = document.createElement("div");
    empty.textContent = emptyText;
    empty.style.cssText = "border:1px solid #27272a;border-radius:6px;background:#18181b;color:#a1a1aa;padding:8px;font-size:12px;";
    listElement.append(empty);
    return;
  }
  ingredients.forEach((item, index) => {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:52px minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #27272a;border-radius:6px;background:#18181b;padding:8px;";
    const thumbWrap = document.createElement("div");
    thumbWrap.style.cssText = "width:52px;height:52px;border:1px solid #3f3f46;border-radius:6px;background:#09090b;overflow:hidden;display:flex;align-items:center;justify-content:center;";
    const imageSrc = item?.data || (item?.path ? makeEditorImageUrl(item.path) : "");
    if (imageSrc) {
      const thumb = document.createElement("img");
      thumb.alt = "";
      thumb.src = imageSrc;
      thumb.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
      thumbWrap.append(thumb);
    } else {
      const placeholder = document.createElement("div");
      placeholder.textContent = "IMG";
      placeholder.style.cssText = "font-size:10px;font-weight:900;color:#71717a;";
      thumbWrap.append(placeholder);
    }
    const label = document.createElement("div");
    label.textContent = `${index + 1}. ${item?.name || item?.path || "image ingredient"}`;
    label.title = item?.path || item?.name || "";
    label.style.cssText = "font-size:12px;color:#e4e4e7;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const remove = makeMiniButton("Remove");
    remove.onclick = () => {
      onRemove(index);
    };
    row.append(thumbWrap, label, remove);
    listElement.append(row);
  });
}

function randomSeedValue() {
  return Math.floor(Math.random() * 2147483647) + 1;
}

export function rtvReferenceImagePayload(image = {}) {
  const source = image && typeof image === "object" ? image : {};
  return {
    path: String(source.path || ""),
    data: String(source.data || ""),
    name: String(source.name || ""),
  };
}

export function normalizeRTVReferenceBehavior(value, legacyAnchor = false) {
  const normalized = String(value || "").trim();
  if (["none", "character_anchor", "first_last_frame"].includes(normalized)) return normalized;
  return legacyAnchor ? "character_anchor" : "none";
}

export function createImageReferences({
  activeErnieImageSettings, activeFluxKleinSettings, activeKrea2TwoPassSettings, activeSegment,
  activeZImageSettings, autoSaveSessionQuiet, currentVideoMode, ernieSeed, fluxGlobalIngredientList,
  fluxGlobalIngredientPanel, fluxIngredientList, fluxSeed, i2vSeedInput, i2vVideoSettingsForSegment,
  krea2TwoPassSeed, logicalSubjectIdsForScene, miniMaxH3SettingsForSegment, nbGlobalIngredientList,
  nbGlobalIngredientPanel, nbIngredientList, nbUseGlobalIngredients, previousAutoChainSourceSegment,
  projectInput, pushHistory, render, renderList, sceneReferenceMapValue, sceneSlotNumber,
  sceneVideoConceptPromptText, segmentImageSource, state, storyboardVideoExtraNotesForSegment,
  syncMiniMaxH3Panel, syncRTVSceneImageAnchorPanel, useFluxGlobalIngredients, videoGemmaNotesForSegment,
  zSeed,
}) {
  function setImageSeedForCurrentMode(imageMode = state.imageModelMode || "zimage") {
    const seed = randomSeedValue();
    if (imageMode === "flux_klein") {
      const settings = activeFluxKleinSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_flux_klein_settings) activeSegment().flux_klein_settings = settings;
      else state.fluxKleinSettings = settings;
      fluxSeed.value = String(seed);
    } else if (imageMode === "ernie_image") {
      const settings = activeErnieImageSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_ernie_image_settings) activeSegment().ernie_image_settings = settings;
      else state.ernieImageSettings = settings;
      ernieSeed.value = String(seed);
    } else if (imageMode === "krea2_2pass") {
      const settings = activeKrea2TwoPassSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_krea2_2pass_settings) activeSegment().krea2_2pass_settings = settings;
      else state.krea2TwoPassSettings = settings;
      krea2TwoPassSeed.value = String(seed);
    } else {
      const settings = activeZImageSettings() || {};
      settings.seed = seed;
      if (activeSegment()?.use_scene_zimage_settings) activeSegment().zimage_settings = settings;
      else state.zimageSettings = settings;
      zSeed.value = String(seed);
    }
    return seed;
  }

  function setVideoSeedRandom(segment = activeSegment()) {
    const settings = segment?.use_scene_i2v_video_settings
      ? cloneI2VVideoSettings(segment.i2v_video_settings || state.i2vVideoSettings)
      : cloneI2VVideoSettings(state.i2vVideoSettings);
    settings.seed = randomSeedValue();
    if (segment?.use_scene_i2v_video_settings) segment.i2v_video_settings = settings;
    else state.i2vVideoSettings = settings;
    if (!segment || segment.id === activeSegment()?.id) i2vSeedInput.value = String(settings.seed);
    return settings.seed;
  }

  function setMiniMaxH3SeedRandom(segment = activeSegment()) {
    const settings = miniMaxH3SettingsForSegment(segment);
    settings.seed = randomSeedValue();
    settings.two_pass_pass1_seed = randomSeedValue();
    settings.two_pass_pass2_seed = randomSeedValue();
    settings.advanced_two_pass_pass1_seed = randomSeedValue();
    settings.advanced_two_pass_pass2_seed = randomSeedValue();
    if (segment?.use_scene_minimax_h3_settings) {
      segment.minimax_h3_settings = cloneMiniMaxH3Settings(settings);
      segment.minimax_h3_mode = segment.minimax_h3_settings.video_mode;
    } else {
      state.miniMaxH3Settings = cloneMiniMaxH3Settings(settings);
    }
    if (!segment || segment.id === activeSegment()?.id) syncMiniMaxH3Panel();
    return settings.seed;
  }

  function renderFluxIngredientList(segment = activeSegment()) {
    const ingredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
    renderFluxIngredientRows(fluxIngredientList, ingredients, "No scene-specific image ingredients loaded for this scene.", (index) => {
      const active = activeSegment();
      if (!active || !Array.isArray(active.flux_image_ingredients)) return;
      pushHistory();
      active.flux_image_ingredients.splice(index, 1);
      renderFluxIngredientList(active);
      renderNBIngredientList(active);
      render();
    });
  }

  function renderNBIngredientList(segment = activeSegment()) {
    const ingredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
    renderFluxIngredientRows(nbIngredientList, ingredients, "No scene-specific NanoBanana reference images loaded for this scene.", (index) => {
      const active = activeSegment();
      if (!active || !Array.isArray(active.flux_image_ingredients)) return;
      pushHistory();
      active.flux_image_ingredients.splice(index, 1);
      renderFluxIngredientList(active);
      renderNBIngredientList(active);
      render();
    });
  }

  function renderFluxGlobalIngredientList() {
    const ingredients = Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [];
    renderFluxIngredientRows(fluxGlobalIngredientList, ingredients, "No global Flux/Klein image ingredients loaded.", (index) => {
      pushHistory();
      state.fluxGlobalImageIngredients.splice(index, 1);
      renderFluxGlobalIngredientList();
      render();
    });
    renderFluxIngredientRows(nbGlobalIngredientList, ingredients, "No global Nano B reference images loaded.", (index) => {
      pushHistory();
      state.fluxGlobalImageIngredients.splice(index, 1);
      renderFluxGlobalIngredientList();
      render();
    });
  }

  function syncFluxGlobalIngredientPanel() {
    useFluxGlobalIngredients.input.checked = Boolean(state.useFluxGlobalImageIngredients);
    nbUseGlobalIngredients.input.checked = Boolean(state.useFluxGlobalImageIngredients);
    fluxGlobalIngredientPanel.style.display = state.useFluxGlobalImageIngredients ? "flex" : "none";
    nbGlobalIngredientPanel.style.display = state.useFluxGlobalImageIngredients ? "flex" : "none";
  }

  function mergedFluxImageIngredients(segment = activeSegment()) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const ingredients = [];
    const addUnique = (item) => {
      if (!item || typeof item !== "object") return;
      const path = String(item.path || "");
      const data = String(item.data || "");
      const name = String(item.name || "");
      if (!path && !data) return;
      const key = path || data || name;
      if (ingredients.some((existing) => (existing.path || existing.data || existing.name) === key)) return;
      ingredients.push({ path, data, name: name || path?.split?.(/[\\/]/)?.pop?.() || "reference.png" });
    };
    if (refs.use_subject_reference) {
      if (refs.subject_count > 1 && segment) {
        const subjectIds = logicalSubjectIdsForScene(refs, segment);
        const idSet = new Set(Array.isArray(subjectIds) ? subjectIds : [subjectIds].filter(Boolean));
        const mapped = refs.subjects.filter((item) => idSet.has(item.id));
        expandSubjectReferencesForRender(refs, mapped).forEach((item) => addUnique(item.image));
      } else {
        addUnique(refs.subject?.image || refs.subjects?.[0]?.image);
      }
    }
    if (refs.use_location_references && segment) {
      const locId = String(sceneReferenceMapValue(refs.scene_map, segment) || "");
      const location = refs.locations.find((item) => item.id === locId);
      addUnique(location?.image);
    }
    if (refs.include_manual_ingredients !== false) {
      const globalIngredients = state.useFluxGlobalImageIngredients && Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [];
      const sceneIngredients = Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [];
      globalIngredients.forEach(addUnique);
      sceneIngredients.forEach(addUnique);
    }
    return ingredients;
  }

  function rtvReferenceBehaviorForSegment(segment = activeSegment()) {
    return normalizeRTVReferenceBehavior(segment?.rtv_reference_behavior, segment?.use_scene_image_as_rtv_ref);
  }

  function firstLastFrameEndImageSource(segment = activeSegment()) {
    if (!segment) return null;
    return {
      path: String(segment.first_last_frame_end_image_path || ""),
      data: String(segment.first_last_frame_end_image_data || ""),
      name: String(segment.first_last_frame_end_image_name || "last_frame.png"),
    };
  }

  function hasFirstLastFrameEndImage(segment = activeSegment()) {
    const image = firstLastFrameEndImageSource(segment);
    return Boolean(image?.path || image?.data);
  }

  function firstLastFrameAssignedEndImageSource(segment = activeSegment()) {
    const explicitEnd = firstLastFrameEndImageSource(segment) || {};
    if (explicitEnd.path || explicitEnd.data) return explicitEnd;
    const assignedImage = segmentImageSource(segment) || {};
    return (assignedImage.path || assignedImage.data)
      ? { ...assignedImage, resolved_from_scene_image: true }
      : explicitEnd;
  }

  function firstLastFrameResolvedEndImageSource(segment = activeSegment()) {
    const explicitEnd = firstLastFrameEndImageSource(segment) || {};
    if (explicitEnd.path || explicitEnd.data) return explicitEnd;
    const previous = previousAutoChainSourceSegment(segment);
    const mayUseAssignedSceneImage = Boolean(previous) && (
      flfPreGeneratePromptsEnabled(segment)
      || flfRenderChainStartSource(segment) === "previous_image"
    );
    if (mayUseAssignedSceneImage) {
      return firstLastFrameAssignedEndImageSource(segment);
    }
    return explicitEnd;
  }

  function promoteChainedFLFSceneImageToEndFrame(segment = activeSegment()) {
    if (!segment || currentVideoMode() !== "flf" || hasFirstLastFrameEndImage(segment)) return false;
    const inheritedStart = firstLastFrameStartImageSource(segment) || {};
    if (!inheritedStart.chained_from_scene_id && !inheritedStart.chained_from_rendered_video) return false;
    const ownImage = segmentImageSource(segment) || {};
    if (!ownImage.path && !ownImage.data) return false;
    const inheritedKey = String(inheritedStart.path || inheritedStart.data || "");
    const ownKey = String(ownImage.path || ownImage.data || "");
    if (!ownKey || ownKey === inheritedKey) return false;
    segment.first_last_frame_end_image_path = ownImage.path || "";
    segment.first_last_frame_end_image_data = ownImage.path ? "" : (ownImage.data || "");
    segment.first_last_frame_end_image_name = ownImage.name || String(ownImage.path || "").split(/[\\/]/).pop() || "last_frame.png";
    return true;
  }

  async function loadFirstLastFrameEndFile(file, segment = activeSegment()) {
    if (!file || !segment) return;
    const isImage = String(file.type || "").startsWith("image/") || /\.(?:png|jpe?g|webp)$/i.test(String(file.name || ""));
    if (!isImage) throw new Error("Drop a PNG, JPG, JPEG, or WEBP image for the end frame.");
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("Could not read the end-frame image."));
      reader.readAsDataURL(file);
    });
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    const saved = projectFolder ? await postJson("/vrgdg/music_builder/archive_scene_image", {
      image_data: data,
      project_folder: projectFolder,
      scene_number: sceneSlotNumber(segment),
    }) : { saved_path: "" };
    pushHistory();
    segment.first_last_frame_end_image_path = saved.saved_path || "";
    segment.first_last_frame_end_image_data = saved.saved_path ? "" : data;
    segment.first_last_frame_end_image_name = file.name || "end_frame.png";
    segment.flf_end_frame_prompt = "";
    segment.flf_end_frame_stale = false;
    segment.flf_final_prompt_ready = false;
    syncRTVSceneImageAnchorPanel();
    renderList();
    if (projectFolder) await autoSaveSessionQuiet("First Last Frame end image loaded");
    toast("End frame loaded. No image provider was used.");
  }

  function flfChainingEnabled(segment = activeSegment()) {
    return state.i2vVideoSettings?.flf_chain_previous_end_frame !== false;
  }

  function flfPreGeneratePromptsEnabled(segment = activeSegment()) {
    return state.i2vVideoSettings?.flf_pregenerate_prompts_from_scene_images === true;
  }

  function flfRenderChainStartSource(segment = activeSegment()) {
    return state.i2vVideoSettings?.flf_render_chain_start_source === "previous_image" ? "previous_image" : "rendered_frame";
  }

  function firstLastFrameStartImageSource(segment = activeSegment()) {
    if (!segment) return null;
    if (currentVideoMode() === "flf" && flfChainingEnabled(segment)) {
      const previous = previousAutoChainSourceSegment(segment);
      if (previous) {
        if (flfRenderChainStartSource(segment) === "previous_image") {
          const previousEnd = firstLastFrameAssignedEndImageSource(previous);
          if (previousEnd?.path || previousEnd?.data) {
            return { ...previousEnd, chained_from_scene_id: previous.id || "", render_chain_source: "previous_image" };
          }
          return null;
        }
        if (segment.flf_rendered_start_frame_path || segment.flf_rendered_start_frame_data) {
          return {
            path: String(segment.flf_rendered_start_frame_path || ""),
            data: String(segment.flf_rendered_start_frame_data || ""),
            name: String(segment.flf_rendered_start_frame_name || "rendered_previous_end.png"),
            chained_from_rendered_video: true,
            render_chain_source: "rendered_frame",
          };
        }
        return null;
      }
    }
    return segmentImageSource(segment);
  }

  function firstLastFramePromptReferences(segment = activeSegment()) {
    let firstFrame = firstLastFrameStartImageSource(segment);
    const previous = previousAutoChainSourceSegment(segment);
    if (previous && (flfChainingEnabled(segment) || flfPreGeneratePromptsEnabled(segment))) {
      if (flfPreGeneratePromptsEnabled(segment)) {
        const provisional = firstLastFrameAssignedEndImageSource(previous);
        if (provisional?.path || provisional?.data) firstFrame = { ...provisional, prompt_only_provisional: true };
      } else if (!segment?.flf_rendered_start_frame_path && !segment?.flf_rendered_start_frame_data) {
        firstFrame = null;
      }
    }
    const lastFrame = firstLastFrameResolvedEndImageSource(segment);
    const refs = [];
    if (firstFrame?.path || firstFrame?.data) {
      refs.push({
        path: firstFrame.path || "",
        data: firstFrame.data || "",
        name: firstFrame.name || "first_frame.png",
        label: "Opening visual state",
        role: "first_frame",
      });
    }
    if (lastFrame?.path || lastFrame?.data) {
      refs.push({
        path: lastFrame.path || "",
        data: lastFrame.data || "",
        name: lastFrame.name || "last_frame.png",
        label: "Ending visual state",
        role: "last_frame",
      });
    }
    return refs;
  }

  function flfTransitionLoraActive(segment = activeSegment()) {
    const settings = i2vVideoSettingsForSegment(segment) || {};
    if (!settings.use_loras) return false;
    const count = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
    return (settings.loras || []).slice(0, count).some((item) => {
      const name = String(item?.name || "").trim().toLowerCase();
      return name && name !== "[none]" && /ltx[^\n]*2[._-]?3[^\n]*transition|transition[^\n]*lora|ltx2[._-]?3-transition/.test(name);
    });
  }

  function flfTransitionPromptDirection(segment = activeSegment()) {
    const sceneType = String(segment?.flf_transition_type || "global");
    const settings = segment?.use_scene_i2v_video_settings
      ? (segment.i2v_video_settings || state.i2vVideoSettings || {})
      : (state.i2vVideoSettings || {});
    const type = sceneType === "global"
      ? (["smooth", "morph"].includes(settings.flf_global_transition_type) ? settings.flf_global_transition_type : "auto")
      : sceneType;
    if (type === "smooth") {
      return "First Last Frame transition type: SMOOTH TRANSITION. Treat both images as the same continuous subject and world. Preserve identity, wardrobe, environment, lighting logic, and physical continuity while describing a seamless camera move, reframing, or natural action that arrives exactly at the last frame. Do not morph identities or invent a surreal transformation.";
    }
    if (type === "morph") {
      return "First Last Frame transition type: SURREAL MORPH. Describe a creative, continuous, visually legible transformation from every important element of the first frame into the last frame. Use organic shape changes, material transformations, match-motion, and cinematic surrealism while arriving exactly at the last image.";
    }
    return "First Last Frame transition type: AUTO. Inspect both images. If they show the same subject or setting from different framing, angles, poses, or camera distances, create a smooth cinematic continuity transition. If they are substantially different, create a seamless surreal morph from the first visual state into the last. Choose the approach that best fits the images and describe continuous motion rather than listing them.";
  }

  function flfGemmaContextMode(segment = activeSegment()) {
    const settings = i2vVideoSettingsForSegment(segment) || state.i2vVideoSettings || {};
    const mode = String(settings.flf_gemma_context_mode || "images_story").trim();
    return ["images_only", "images_story", "full"].includes(mode) ? mode : "images_story";
  }

  function flfGemmaVisualNotes(segment) {
    const mode = flfGemmaContextMode(segment);
    const transition = flfTransitionPromptDirection(segment);
    if (mode === "images_only") return transition;
    if (mode === "images_story") {
      const storyBeat = String(segment?.story_beat || "").trim();
      return [transition, storyBeat ? `Scene story beat (bridge guidance only; images override conflicts):\n${storyBeat}` : ""].filter(Boolean).join("\n\n");
    }
    return [videoGemmaNotesForSegment(segment), transition, flfStoryboardVideoDirection(segment), storyboardVideoExtraNotesForSegment(segment)].filter(Boolean).join("\n\n");
  }

  function flfGemmaSceneConcept(segment) {
    const mode = flfGemmaContextMode(segment);
    if (mode === "images_only") return "Create one continuous physical transition that begins at the visible LEFT image and completes at the visible RIGHT image.";
    if (mode === "images_story") return String(segment?.story_beat || "").trim() || "Create one continuous physical transition between the visible endpoints.";
    return sceneVideoConceptPromptText(segment);
  }

  function rtvSceneImageAnchorPayload(segment = activeSegment()) {
    if (rtvReferenceBehaviorForSegment(segment) !== "character_anchor") return null;
    const image = segmentImageSource(segment);
    if (!image?.path && !image?.data) return null;
    return {
      ...rtvReferenceImagePayload(image),
      label: "Scene image: secondary character anchor",
      reference_type: "character",
      scene_image_anchor: true,
    };
  }

  function flfStoryboardVideoDirection(segment) {
    if (currentVideoMode() !== "flf" || !segment) return "";
    const startState = String(segment.flf_start_state || "").trim();
    const transformation = String(segment.flf_transformation || "").trim();
    const endState = String(segment.flf_end_state || "").trim();
    const carryForward = String(segment.flf_carry_forward || "").trim();
    if (![startState, transformation, endState, carryForward].some(Boolean)) return "";
    return [
      "STORYBOARD FLF MOTION CONTRACT:",
      startState ? `Opening state:\n${startState}` : "",
      transformation ? `Required continuous transformation:\n${transformation}` : "",
      endState ? `Required destination state:\n${endState}` : "",
      carryForward ? `Continuity that must survive into the destination:\n${carryForward}` : "",
      "Use the two supplied images as visual truth. Use the required transformation as the primary motion plan connecting them.",
      "Begin the physical action and camera travel immediately, progress continuously, and complete at the exact destination. Do not replace the transformation with a generic dissolve, fade, cut, texture swap, or unrelated invented action.",
      "Keep all singing, exact lyric, performance, and selected facial-performance directions active when they are supplied.",
    ].filter(Boolean).join("\n\n");
  }

  return {
    firstLastFrameEndImageSource, firstLastFramePromptReferences, firstLastFrameResolvedEndImageSource,
    firstLastFrameStartImageSource, flfChainingEnabled, flfGemmaContextMode, flfGemmaSceneConcept,
    flfGemmaVisualNotes, flfPreGeneratePromptsEnabled, flfRenderChainStartSource, flfTransitionLoraActive,
    hasFirstLastFrameEndImage, loadFirstLastFrameEndFile, mergedFluxImageIngredients,
    promoteChainedFLFSceneImageToEndFrame, renderFluxGlobalIngredientList, renderFluxIngredientList,
    renderNBIngredientList, rtvReferenceBehaviorForSegment, rtvSceneImageAnchorPayload,
    setImageSeedForCurrentMode, setMiniMaxH3SeedRandom, setVideoSeedRandom, syncFluxGlobalIngredientPanel,
  };
}
