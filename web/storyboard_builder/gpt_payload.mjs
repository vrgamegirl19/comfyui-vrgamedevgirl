import { isRefmodCard } from "../music_video_builder/refmod_labels.mjs";
import {
  FACIAL_PERFORMANCE_PRESETS,
  ID_LORA_FACIAL_PERFORMANCE_PRESETS,
  ID_LORA_PERFORMANCE_STYLE_PRESETS,
  PERFORMANCE_STYLE_PRESETS,
} from "./performance_presets.mjs";
import {
  normalizeScene,
  normalizeStoryboardMiniMaxH3Mode,
  normalizeStoryboardPerformanceMode,
  normalizeStoryboardProjectVideoEngine,
  normalizeStoryboardShortFilmPlanningMode,
  normalizeStoryLayer,
  slimReferenceForRequest,
  slimSceneForRequest,
  storyboardSceneCardContext,
  storyboardTimelineNotesForRequest,
  STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTION,
  storyboardCameraMotionForSpeed,
  storyboardCutPlanForDuration,
  storyboardSpeedGuidance,
  storyboardSpeedValue,
  storyboardStillFacialDirection,
} from "./scenes.mjs";
import { storyboardRefmodLabels } from "./references.mjs";
import {
  STORYBOARD_CAMERA_FLOW_PRESETS,
  storyboardCameraFlowEntry,
  storyboardImageAestheticGuidance,
} from "./shot_presets.mjs";
import {
  storyboardMiniMaxVideoStylePreset,
  storyboardMiniMaxVideoStyleVerbiage,
  storyboardSceneSupportsVideoStyle,
  storyboardTemporalWorldEffectForScene,
} from "./video_style.mjs";

const STORYBOARD_GPT_URL = "https://chatgpt.com/g/g-6a28d15f04e88191a2375d564ff8d90c-ltx-2-3-video-builder-from-storyboard-builder";
const STORYBOARD_IMAGE_GPT_URL = "https://chatgpt.com/g/g-6a40129fc12c81919878b79eaa5ae94f-text-to-image-prompt-builder-for-krea-2";
export const STORY_LAYER_CHATGPT_URL = "https://chatgpt.com/";

export function storyLayerGptPayload(state) {
  const storyboardPayload = storyboardGptPayload(state);
  const scenes = state.scenes.map((scene, index) => slimSceneForRequest(scene, index));
  const orderedLyrics = scenes
    .map((scene) => String(scene.lyrics || scene.lyric_text || "").trim())
    .filter(Boolean);
  const sourceLyrics = String(state.lineMappingLyrics || state.lyricMapper?.source_text || "").trim();
  const lyricSections = scenes.map((scene, index) => ({
    scene_number: Number(scene.scene_number || index + 1),
    section: String(scene.lyric_section || scene.label || "").trim(),
    lyrics: String(scene.lyrics || scene.lyric_text || "").trim(),
  })).filter((item) => item.section || item.lyrics);
  const presetDetails = {
    image_aesthetic: state.imageAesthetic || "",
    image_shot_flow: state.imageShotFlow || "",
    video_style: state.videoStyle || "",
    performance_style: state.performanceStyle || "",
    facial_performance: state.facialPerformance || "",
    camera_flow: state.cameraFlow || "",
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    lyric_story_strength: normalizeStoryLayer(state.storyLayer).lyric_story_strength,
  };
  return {
    payload_type: "story_layer_planning",
    execution_mode: "execute_immediately",
    user_request: "Process this story-layer planning payload now. Do not ask what I want done and do not ask for more lyrics or project details. Generate the final story-layer JSON response immediately.",
    task_instruction: "Use the supplied project context to create or revise a coherent music-video story. The explicit project_inputs fields are the source of truth. Execute this task immediately and return only valid JSON matching output_format. Keep the user's overall story idea when it is present, use the ordered lyric sections as the structural spine, and use the style, subject, location, performance, and camera preset details as creative constraints.",
    output_format: {
      overall_story_idea: "A concise premise or overall story idea. Preserve the supplied idea when it is usable; otherwise create one.",
      user_story_arc: "A section-by-section story arc using the supplied lyric section labels in order.",
      song_story_brief: "A compact production brief explaining the premise, emotional progression, visual world, recurring motifs, and ending."
    },
    current_story_layer: normalizeStoryLayer(state.storyLayer),
    project_inputs: {
      overall_story_idea: String(state.storyLayer?.overall_story_idea || "").trim(),
      ordered_lyrics: orderedLyrics,
      source_lyrics: sourceLyrics,
      lyrics_instruction: "When scenes are present, use ordered_lyrics for scene alignment. When ordered_lyrics is empty, use source_lyrics as the complete pasted song/script and create the lyric sections yourself from its headings and structure.",
      lyric_sections: lyricSections,
      timeline_markers: storyboardTimelineNotesForRequest(state),
      timeline_notes_instruction: "Use Timeline Notes as user story-event and timing guidance. Apply each note to the scenes and lyric sections overlapping its start/end seconds; a note without an end marks an event at its start. Preserve the required lyric-section order, mapped cast and locations, and exact imported dialogue. Do not spread a timed event across unrelated sections.",
      subjects: (state.referenceBuilder?.subjects || []).map(slimReferenceForRequest).filter(Boolean),
      locations: (state.referenceBuilder?.locations || []).map(slimReferenceForRequest).filter(Boolean),
      preset_details: presetDetails,
      scenes,
    },
    project_context: storyboardPayload,
  };
}

const REFMOD_PIPELINE_INSTRUCTION = "This project renders from saved RefMods instead of reference images. Each visible subject in subject_refs has a refmod_label such as <Video 1> or <Picture 2>. In every shot, write that subject's label right after the first mention of their name, for example: Brad <Video 1> walks to the window. Use only the labels listed in refmod_pipeline.references and never invent or renumber one. Name a subject only if it is listed for this scene.";

// What the GPT needs to know about this scene's RefMods: each card's label and name, and how to write the labels.
function storyboardRefmodPipelineForGpt(scene, catalogSubjects, labels) {
  const seen = new Set();
  const references = [];
  for (const card of [...(scene.subject_refs || []), scene.location_ref, ...catalogSubjects]) {
    const id = String(card?.id || "");
    if (!id || seen.has(id) || !labels.has(id) || !isRefmodCard(card)) continue;
    seen.add(id);
    references.push({ label: labels.get(id), name: String(card.name || "").trim(), refmod_type: card.refmod.type || card.reference_type || "", refmod_kind: card.refmod.kind });
  }
  return { enabled: true, instruction: REFMOD_PIPELINE_INSTRUCTION, references };
}

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
    ...(options.refmodLabel && isRefmodCard(ref) ? { refmod_label: options.refmodLabel, refmod_kind: ref.refmod.kind, refmod_type: ref.refmod.type || ref.reference_type || "" } : {}),
  };
}

function storyboardVideoPromptTypeLabel(type) {
  const key = String(type || "").toLowerCase();
  if (key === "text_to_video") return "MiniMax H3 text to video";
  if (key === "image_to_video") return "MiniMax H3 image to video";
  if (key === "reference_to_video") return "MiniMax H3 reference to video";
  if (key === "video_to_video") return "MiniMax H3 video to video";
  if (key === "id_lora") return "ID-LoRA image to video";
  if (key === "ingredients") return "ingredients to video";
  if (key === "t2v") return "text to video";
  if (key === "rtv") return "reference to video";
  if (key === "flf") return "first / last frame video";
  if (key === "i2v") return "image to video";
  return key || "image to video";
}

function storyboardStartingShotInstruction(shotType) {
  const shot = String(shotType || "").trim();
  if (!shot) return "";
  if (shot.toLowerCase() === "eyes shot") {
    return "The literal first generated frame must already be an extreme close-up of the subject's eyes. Do not use a wider or farther-away lead-in. The selected camera motion must begin from that opening framing.";
  }
  return `The literal first generated frame must already be a ${shot}. Do not use a wider, farther-away, establishing, or full-body lead-in before reaching that framing. The selected camera motion must begin from that opening framing.`;
}

function storyboardLtxStartingFraming(shotType) {
  const shot = String(shotType || "").trim();
  if (!shot) return "";
  const movementClause = shot.match(/^(.*?),\s*(?:(?:then|before)\s+)?(?:slowly\s+)?(?:pulling|panning|tilting|sliding|tracking|orbiting|zooming|dollying|crane|moving|drifting)\b/i);
  return movementClause ? movementClause[1].trim() : shot;
}

function storyboardLtxEmbeddedCameraMotion(shotType) {
  const shot = String(shotType || "").trim();
  const movementClause = shot.match(/^.*?,\s*((?:(?:then|before)\s+)?(?:slowly\s+)?(?:pulling|panning|tilting|sliding|tracking|orbiting|zooming|dollying|crane|moving|drifting)\b.*)$/i);
  return movementClause ? movementClause[1].replace(/^(?:then|before)\s+/i, "").trim() : "";
}

function storyboardScenesForGpt(state) {
  const imageMode = state.mode !== "image_to_video_prep";
  const idLoraMode = String(state.videoPromptType || state.video_prompt_type || "").trim() === "id_lora"
    || state.scenes.some((scene) => String(scene?.video_prompt_type || "").trim() === "id_lora");
  const miniMaxShortFilmMode = normalizeStoryboardProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
    && normalizeStoryboardPerformanceMode(state.performanceMode || state.performance_mode) === "speaking";
  const filmPlanningProfile = idLoraMode || miniMaxShortFilmMode;
  const fullyCustomShortFilm = miniMaxShortFilmMode
    && normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode) === "fully_custom";
  const performancePresets = filmPlanningProfile ? ID_LORA_PERFORMANCE_STYLE_PRESETS : PERFORMANCE_STYLE_PRESETS;
  const facialPresets = filmPlanningProfile ? ID_LORA_FACIAL_PERFORMANCE_PRESETS : FACIAL_PERFORMANCE_PRESETS;
  const performancePreset = (value = "") => performancePresets.find((item) => item.value === value) || performancePresets[0] || PERFORMANCE_STYLE_PRESETS[0];
  const facialPresetForPayload = (value = "") => facialPresets.find((item) => item.value === value) || facialPresets[0] || FACIAL_PERFORMANCE_PRESETS[0];
  const cameraFlowKey = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] ? state.cameraFlow : "balanced";
  const cameraFlowPreset = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowKey];
  const explicitLyricSections = state.scenes.map((scene, index) => String(normalizeScene(scene, index).lyric_section || "").trim());
  const effectiveLyricSection = (index) => {
    if (explicitLyricSections[index]) return explicitLyricSections[index];
    for (let next = index + 1; next < explicitLyricSections.length; next += 1) {
      if (explicitLyricSections[next]) return explicitLyricSections[next];
    }
    for (let previous = index - 1; previous >= 0; previous -= 1) {
      if (explicitLyricSections[previous]) return explicitLyricSections[previous];
    }
    return "";
  };
  let previousCameraMotion = "";
  return state.scenes.map((scene, index) => {
    const normalized = normalizeScene(scene, index);
    const sceneVideoEngine = normalizeStoryboardProjectVideoEngine(normalized.project_video_engine || state.projectVideoEngine);
    const lyricSection = effectiveLyricSection(index);
    if (!explicitLyricSections[index] && lyricSection && scene && typeof scene === "object") {
      scene.lyric_section = lyricSection;
    }
    const sceneNumberIndex = Math.max(0, Number(normalized.scene_number || index + 1) - 1);
    const cameraFallback = fullyCustomShortFilm ? null : storyboardCameraFlowEntry(state.cameraFlow || "balanced", sceneNumberIndex, previousCameraMotion, state.customCameraFlowSequence);
    const shotType = normalized.shot_type || cameraFallback?.shot || "";
    const promptShotType = sceneVideoEngine === "ltx" ? storyboardLtxStartingFraming(shotType) : shotType;
    const requiresStartingShot = !imageMode && normalized.video_prompt_type !== "i2v" && Boolean(promptShotType);
    const embeddedLtxCameraMotion = sceneVideoEngine === "ltx" ? storyboardLtxEmbeddedCameraMotion(shotType) : "";
    const rawCameraMotion = normalized.camera_motion || (imageMode ? "" : embeddedLtxCameraMotion || cameraFallback?.camera) || "";
    const cameraMotion = imageMode || fullyCustomShortFilm
      ? rawCameraMotion
      : storyboardCameraMotionForSpeed(rawCameraMotion, state.cameraMotionSpeed);
    const motionSummary = String(normalized.motion_summary || "").trim();
    const cameraMotionForPrompt = motionSummary ? "" : cameraMotion;
    if (!imageMode) previousCameraMotion = cameraMotion || previousCameraMotion;
    const lyricText = String(normalized.lyrics || "").trim();
    const performanceMode = normalizeStoryboardPerformanceMode(normalized.performance_mode || state.performanceMode || state.videoType || state.performance_mode);
    const selectedFacialPerformance = normalized.facial_performance || (fullyCustomShortFilm ? "" : state.facialPerformance);
    const facialPreset = facialPresetForPayload(selectedFacialPerformance);
    const facialCustom = String(normalized.facial_performance_custom || state.facialPerformanceCustom || "").trim();
    const facialDirection = selectedFacialPerformance === "off"
      ? ""
      : selectedFacialPerformance === "custom" && facialCustom
      ? facialCustom
      : [facialPreset.direction, facialCustom].filter(Boolean).join(" ");
    const selectedPerformanceStyle = normalized.performance_style || (fullyCustomShortFilm ? "" : state.performanceStyle);
    const selectedPerformancePreset = performancePreset(selectedPerformanceStyle);
    const supportsVideoStyle = storyboardSceneSupportsVideoStyle(normalized);
    const selectedVideoStyle = supportsVideoStyle ? String(state.videoStyle || normalized.video_style || "") : "";
    const selectedVideoStyleCustom = selectedVideoStyle === "custom"
      ? String(state.videoStyle === "custom" ? state.videoStyleCustom : (normalized.video_style_custom || state.videoStyleCustom || "")).trim()
      : "";
    const selectedVideoStylePreset = storyboardMiniMaxVideoStylePreset(selectedVideoStyle);
    const selectedVideoStyleVerbiage = storyboardMiniMaxVideoStyleVerbiage(selectedVideoStyle, selectedVideoStyleCustom);
    const temporalWorldEffect = !imageMode ? storyboardTemporalWorldEffectForScene(normalized, state) : null;
    const exactSceneDuration = Math.max(
      0,
      Number(normalized.exact_duration || 0) || Number(normalized.timeline_end || 0) - Number(normalized.timeline_start || 0),
    );
    const cutPlan = !imageMode
      ? storyboardCutPlanForDuration(exactSceneDuration, state.cutFrequency, sceneVideoEngine)
      : null;
    const instrumental = Boolean(normalized.lyric_instrumental);
    const noLipSync = Boolean(normalized.lyric_no_lip_sync || performanceMode === "no_lip_sync");
    const noCharacterPresent = Boolean(normalized.no_character_present);
    const shouldLipSync = !imageMode && performanceMode !== "no_lip_sync" && Boolean(lyricText) && !instrumental && !noLipSync && !noCharacterPresent;
    const refmodActive = Boolean(state.refmodPipeline) && sceneVideoEngine === "minimax_h3";
    const refmodLabels = refmodActive ? storyboardRefmodLabels(normalized, state.referenceBuilder?.subjects || []) : new Map();
    const subjectRefs = noCharacterPresent ? [] : (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map((ref) => storyboardReferenceForGpt(ref, { subject: true, refmodLabel: refmodLabels.get(String(ref?.id || "")) }))
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
      scene_card: storyboardSceneCardContext(normalized, index),
      scene_card_instruction: STORYBOARD_SCENE_CARD_CONTEXT_INSTRUCTION,
      director_note: normalized.timeline_note,
      label: normalized.label,
      lyric_section: lyricSection,
      prompt_type: imageMode ? "text to image" : storyboardVideoPromptTypeLabel(normalized.video_prompt_type),
      project_video_engine: normalizeStoryboardProjectVideoEngine(normalized.project_video_engine || state.projectVideoEngine),
      minimax_h3_mode: normalizeStoryboardMiniMaxH3Mode(normalized.minimax_h3_mode),
      ...(cutPlan ? {
        minimax_h3_cut_frequency: cutPlan.frequency,
        cut_plan: cutPlan,
      } : {}),
      exact_duration: Number(normalized.exact_duration || 0),
      timeline_start: Number(normalized.timeline_start || 0),
      timeline_end: Number(normalized.timeline_end || 0),
      performance_mode: performanceMode,
      short_film_planning_mode: miniMaxShortFilmMode ? normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode) : "",
      manual_scene_contract: fullyCustomShortFilm
        ? "Every populated scene-card field is authoritative. Format the supplied material only. Do not invent, rewrite, reorder, merge, omit, or replace dialogue, speakers, actions, story beats, shot/framing, camera motion, setting, references, sound direction, or continuity. Leave unspecified details unspecified instead of filling them in."
        : "",
      lyric_line_to_sing: shouldLipSync && performanceMode === "singing" ? lyricText : "",
      line_to_say: shouldLipSync && performanceMode === "speaking" ? lyricText : "",
      // Preserve the mapped performer assignment even in Image Prep. Image
      // prompts intentionally disable lip-sync, but Scene Beat generation
      // still needs to know which visible subject is the singer.
      lyric_singers: explicitSingers,
      performer_assignment: {
        singing: explicitSingers,
        silent: nonSingingSubjects,
        instruction: explicitSingers.length === 1
          ? `${explicitSingers[0]} is the only singing performer. Every other visible subject is silent and must not sing.`
          : explicitSingers.length > 1
            ? `Only these mapped performers sing: ${explicitSingers.join(", ")}. Every other visible subject is silent and must not sing.`
            : "No mapped subject is assigned to sing. All visible subjects are silent.",
      },
      vocal_status: {
        performance_mode: performanceMode,
        lyric_text: lyricText,
        lyric_section: lyricSection,
        singers,
        instrumental,
        no_lip_sync: noLipSync,
        should_lip_sync: shouldLipSync,
        no_character_present: noCharacterPresent,
        lyric_cue_map: normalized.lyric_cue_map,
        lyric_shot_word_timing_enabled: normalized.lyric_shot_word_timing_enabled,
        lyric_performance_mode: normalized.lyric_performance_mode,
        timed_lyric_cue_contract: normalized.timed_lyric_cue_contract,
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
      ...(refmodActive && !imageMode && refmodLabels.size ? { refmod_pipeline: storyboardRefmodPipelineForGpt(normalized, state.referenceBuilder?.subjects || [], refmodLabels) } : {}),
      scene_summary: imageMode ? "" : normalized.prompt_summary,
      story_layer: {
        lyric_section: lyricSection,
        scene_story_beat: normalized.story_beat,
        flf_start_state: normalized.flf_start_state,
        flf_transformation: normalized.flf_transformation,
        flf_end_state: normalized.flf_end_state,
        flf_carry_forward: normalized.flf_carry_forward,
        song_story_brief: state.storyLayer?.enabled === false ? "" : String(state.storyLayer?.song_story_brief || ""),
        user_story_arc: state.storyLayer?.enabled === false ? "" : String(state.storyLayer?.user_story_arc || ""),
        lyric_story_strength: normalizeStoryLayer(state.storyLayer).lyric_story_strength,
        instruction: "Use the story brief and scene story beat as narrative guidance. Lyric story strength controls how literally to follow lyric_line: 0 ignores lyrics, 1-3 uses mood only, 4-6 balances lyrics with story, 7-8 strongly follows lyric meaning, and 9-10 uses concrete lyric objects/actions/emotions whenever possible. Do not turn the prompt into plot exposition.",
      },
      motion_summary: imageMode ? "" : motionSummary,
      still_image_notes: imageMode ? motionSummary : "",
      image_aesthetic: imageMode ? storyboardImageAestheticGuidance(state.imageAesthetic, { idLoraMode: filmPlanningProfile }) : "",
      image_aesthetic_instruction: imageMode
        ? "Translate the selected image aesthetic into concrete prompt details: pose, wardrobe styling, hair, makeup, accessories, lighting setup, lens/framing, composition, environment treatment, texture, weather/time if useful, and art direction. Do not merely name the preset or append it as a short tag."
        : "",
      video_style: !imageMode && supportsVideoStyle && selectedVideoStyle ? selectedVideoStylePreset.label : "",
      video_style_custom: !imageMode && supportsVideoStyle && selectedVideoStyle === "custom" ? selectedVideoStyleCustom : "",
      video_style_guidance: !imageMode && supportsVideoStyle ? selectedVideoStyleVerbiage : "",
      video_style_verbiage: !imageMode && supportsVideoStyle ? selectedVideoStyleVerbiage : "",
      video_style_instruction: !imageMode && supportsVideoStyle && selectedVideoStyleVerbiage
        ? "This exact video_style_verbiage is mandatory. Copy it word-for-word into the final prompt and use it only as the governing visual-appearance contract for lighting, color, texture, materials, production design, grading, and image finish. Do not paraphrase, shorten, rename, or omit it. Do not use it to select, replace, or modify camera motion, character motion, shot timing, editing, or transitions."
        : "",
      temporal_world_effect: temporalWorldEffect || { enabled: false },
      temporal_world_effect_verbiage: temporalWorldEffect?.exact_verbiage || "",
      temporal_world_effect_instruction: temporalWorldEffect
        ? "This exact temporal_world_effect_verbiage is mandatory and will be appended by the builder. Do not recreate, paraphrase, duplicate, or distribute this contract through the shot descriptions. Write only the creative shot action; the appended contract governs temporal behavior."
        : "",
      global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
      global_consistency_instruction: String(state.globalConsistencyPhrase || "").trim()
        ? "Incorporate the global_consistency_phrase naturally into the prompt where it fits. Preserve its key wording, but do not force it to the beginning unless that is the most natural phrasing."
        : "",
      performance_style: !selectedPerformanceStyle || selectedPerformanceStyle === "off" ? "" : selectedPerformancePreset.label,
      performance_direction: !selectedPerformanceStyle || selectedPerformanceStyle === "off" ? "" : selectedPerformancePreset.direction,
      facial_performance: !selectedFacialPerformance || selectedFacialPerformance === "off" ? "" : facialPreset.label,
      facial_performance_direction: !selectedFacialPerformance ? "" : (imageMode ? storyboardStillFacialDirection(facialDirection) : facialDirection),
      facial_performance_custom: !selectedFacialPerformance || selectedFacialPerformance === "off" ? "" : (imageMode ? storyboardStillFacialDirection(facialCustom) : facialCustom),
      emotion_expression_tags: String(normalized.emotion_expression_tags || "").trim(),
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
      extra_subjects: normalized.extra_subjects,
      extra_subject_instruction: normalized.extra_subjects.length
        ? "Use every mapped extra in the scene's action and blocking according to interaction. Describe direct, dancing_with, and alongside roles individually; extras sharing background or background_dancing roles may be grouped by name. Keep identity wording brief and use it only to distinguish people. Do not assign singing, dialogue, or speaker IDs to extras unless explicitly supplied elsewhere."
        : "No mapped extras are assigned to this scene.",
      setting: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      location_ref: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      camera_flow: cameraFlowKey,
      camera_flow_guidance: String(cameraFlowPreset?.guidance || "").trim(),
      shot_type: promptShotType,
      starting_shot: requiresStartingShot
        ? {
            required: true,
            selected_starting_shot: promptShotType,
            instruction: storyboardStartingShotInstruction(promptShotType),
          }
        : null,
      camera_motion: imageMode ? "" : cameraMotionForPrompt,
      still_camera_style: imageMode ? cameraMotion : "",
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      camera_motion_speed_guidance: imageMode || (fullyCustomShortFilm && !cameraMotionForPrompt) ? "" : storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      camera_guidance: imageMode
        ? {
            selected_still_camera_style: cameraMotion,
            instruction: "Use this as still photography composition, lens, lighting, or framing guidance only. Do not turn it into camera movement.",
          }
        : {
            selected_camera_motion: cameraMotionForPrompt,
            camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
            camera_motion_speed_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
            avoid_default_inward_moves: true,
            instruction: motionSummary
              ? "The custom motion_summary is authoritative. Do not add or reuse the scene-default camera motion preset."
              : "Use the selected camera motion as written. Do not add zoom-in, push-in, dolly-in, crash-zoom, or a close-up ending unless that exact inward motion is selected or requested in notes.",
          },
      character_motion: imageMode ? "" : normalized.character_motion,
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      character_motion_guidance: fullyCustomShortFilm && !normalized.character_motion ? "" : storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
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
      audio_direction: normalized.audio_direction,
      continuity: normalized.continuity,
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
    short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
    storyboard_mode: state.mode === "image_to_video_prep" ? "video prompt planning" : "text-to-image prompt planning",
    image_model_mode: selectedImageMode,
    image_model_label: selectedImageModeLabel,
    image_prompt_target: imagePromptTarget,
    ...(imageMode
      ? {
        task_instruction: `Create detailed ${imagePromptTarget}s for Image Prep using advanced Krea 2-style still-image prompting. These are still-image prompts, not video or lip-sync prompts. Use lyrics and story beats for mood, symbolism, emotion, styling, and scene direction only. The mapped location_ref is the required physical set for each scene: do not replace it with a location from story_layer, scene_story_beat, song_story_brief, user_story_arc, or lyrics. If story context mentions another place, translate only its emotion, symbolism, pose, or action into the mapped location_ref environment. Do not say the subject is singing, lip-syncing, performing vocals, or singing the lyric unless the scene notes explicitly ask for a live singing image. Preserve mapped subject prompt names exactly as provided in each scene's visible_subjects and subjects.name. When a subject has a trigger_phrase, that trigger phrase is the subject identity for prompt wording, so write natural phrases like 'a photo of TRIGGER_PHRASE' instead of 'a photo of a woman'. Do not rename 'the woman' as 'one woman' or 'a woman', and do not rename trigger phrases. If global_consistency_phrase is present, weave it naturally into the prompt where it fits instead of slapping it onto the front.`,
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
        task_instruction: "Create detailed image-to-video prompts for Video Prep using a strict source hierarchy. The mapped location_ref is the required physical set for each scene: do not replace it with a location from story_layer, scene_story_beat, song_story_brief, user_story_arc, lyrics, or previous/next scene context. If story context mentions another place, translate only its emotion, tension, symbolism, or action into the mapped location_ref environment. The first_frame_visual_inventory field is only a first-frame inventory: visible subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition. Do not use first_frame_visual_inventory or any image prompt wording for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing. Follow camera_flow_guidance as a hard framing constraint for the entire shot, every camera move, and the ending composition. Follow cut_plan.instruction exactly for every video engine. MiniMax uses its timestamped CUT TO structure. LTX uses ordinary chronological language such as 'then cut to' and must not use the MiniMax timestamp schema. A zero/effectively-zero plan is one continuous take with no cuts. When starting_shot.required is true, the first sentence must explicitly state that the video begins with starting_shot.selected_starting_shot; do not merely imply that framing or use it later. For an eyes shot, explicitly say the video begins with an extreme close-up of the subject's eyes. The selected camera motion begins from that opening framing. Then build the rest of the video prompt in this order: 1) subject and vocal/performance sentence from vocal_status, performance_direction, and facial_performance_direction; 2) character movement sentence from character_motion, character_motion_guidance, character_motion_speed, and scene_story_beat; 3) camera movement sentence from camera_motion, camera_guidance, and camera_motion_speed_guidance; 4) environment/lighting sentence from first_frame_visual_inventory and location_ref; 5) final mood/style sentence from story_layer and image aesthetic only where visual. Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across multiple sentences. If an idea appears in the face sentence, do not repeat it in the body, camera, environment, or atmosphere sentence; use a different concrete visual detail instead. Do not duplicate adjacent words such as 'tall, tall'. The motion priority is character_motion_guidance + camera_motion_speed_guidance + camera_guidance + performance_direction + vocal_status + scene_story_beat above story_layer, and all of those above first_frame_visual_inventory. At camera speed 7-8, do not use slow, gentle, subtle, restrained, locked-off, static, or hold camera wording; use energetic active movement. At camera speed 9-10, use multiple coordinated readable camera moves. At character speed 4 or higher, include at least one clear physical body action, gesture, step, or set interaction; facial movement alone does not count.",
        output_format: {
          type: "video_prompt_import_json",
          instruction: "Return only a JSON code block with an array of objects. Include every scene. Each object must have scene_number and video_prompt. Do not include prose outside the JSON code block.",
          example: [
            { scene_number: 1, video_prompt: "Full video prompt for scene 1..." },
            { scene_number: 2, video_prompt: "Full video prompt for scene 2..." },
          ],
        },
      }),
    story_layer: normalizeStoryLayer(state.storyLayer),
    scenes: storyboardScenesForGpt(payloadState),
  };
}

export function openStoryboardGptUrl(payload) {
  const isImagePayload = String(payload?.storyboard_mode || "").toLowerCase().includes("text-to-image")
    || String(payload?.scenes?.[0]?.prompt_type || "").toLowerCase().includes("text to image");
  window.open(isImagePayload ? STORYBOARD_IMAGE_GPT_URL : STORYBOARD_GPT_URL, "_blank", "noopener,noreferrer");
}
