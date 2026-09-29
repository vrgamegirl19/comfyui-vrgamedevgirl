import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import { normalizeProjectVideoEngine, normalizeVideoType, toast } from "./controls.mjs";
import {
  miniMaxH3InstructionKey,
  miniMaxH3ModeLabel,
  normalizeMiniMaxH3AudioMode,
  normalizeMiniMaxH3Mode,
  normalizeMiniMaxShortFilmPlanningMode,
  normalizeMiniMaxSpeakerAssignments,
} from "./minimax_h3.mjs";
import { syncMiniMaxSpeakerAssignmentLegacyFields } from "./minimax_speaker_cues.mjs";
import { normalizeBuilderStoryboardDefaults, normalizeBuilderStoryLayer } from "./model_settings.mjs";
import {
  applyTriggerPhrase,
  flattenLyricForPrompt,
  isInstrumentalLyricText,
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  segmentUsesNoLipSyncPerformance,
} from "./prompt_text.mjs";
import {
  estimateIdLoraDialogueDuration,
  normalizeFluxReferenceBuilder,
  normalizeIdLoraReferenceBuilder,
} from "./reference_data.mjs";
import { newSegment, normalizeVideoPromptOrigin, sortSegments } from "./segments.mjs";

export let ACTIVE_STORYBOARD_PROMPT_PIPELINE = null;

export function createStoryboardBridge({
  activeProjectFolderForSave, activeSegment, allEditableSegments, applyMappedTriggerPhrases,
  applyMiniMaxH3NativeVoiceBlock, assertMiniMaxH3ReferenceDescriptionsReady, autoSaveSessionQuiet,
  buildI2VPromptRequestForSegment, currentVideoMode, effectiveVideoPerformanceModeForSegment,
  ensureAllSegmentRuntimeFields, ensureAutoTimedSingerCuesBeforePrompt, ensureBuilderManagedFx,
  ensureSegmentRuntimeFields, finalizeVideoPromptForSegment, gemmaRunnerLine, getI2VImageReference,
  i2vMmprojSelect, imageModeDisplayLabel, llmApiVisionModelSelected, miniMaxH3CutPlanForSegment,
  miniMaxH3ModeForSegment, miniMaxH3PromptVisionImages, miniMaxH3PromptVisionImagesForRunner,
  miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment, miniMaxOrderedImageReferenceItemsForSegment,
  mmprojSelect, projectInput, pushHistory, render, runMiniMaxH3PromptGeneration,
  saveI2VVideoSettingsFromPanel, saveSession, sceneDisplayName, segmentImageSource, segmentIndexInfo,
  selectedSegmentImagePath, setSegmentPromptForEdit, state, storyboardPipeline,
  storyboardReferenceBuilderWithIdLoraRefs, storyboardScenePayload, syncI2VVideoSettingsPanel, syncInspector,
  syncVideoModePanel, textGemmaRunnerPayload, timelineDuration, updateActiveFromInputs,
  videoTriggerPhraseForSegment,
}) {
  function openStoryboardBuilderFromProject(options = {}) {
    if (!window.VRGDGStoryboardBuilder?.open) {
      toast("Storyboard Builder UI is not loaded yet. Refresh ComfyUI and try again.", true);
      return;
    }
    updateActiveFromInputs();
    saveI2VVideoSettingsFromPanel();
    const storyboardOpeningMiniMaxMode = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
      ? miniMaxH3ModeForSegment(activeSegment())
      : "";
    const applyStoryboardReferenceMappings = (updates = {}) => {
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const incomingSource = updates.reference_builder || updates.referenceBuilder || {};
      const normalizeIncomingImage = (item = {}) => {
        const sourceItem = item && typeof item === "object" ? item : {};
        const image = sourceItem.image && typeof sourceItem.image === "object" ? sourceItem.image : sourceItem;
        const hasTopLevelImage = Boolean(sourceItem.path || sourceItem.data || sourceItem.image_path || sourceItem.imagePath || sourceItem.image_data || sourceItem.imageData);
        return {
          path: String(image.path || sourceItem.image_path || sourceItem.imagePath || sourceItem.path || ""),
          data: String(image.data || sourceItem.image_data || sourceItem.imageData || sourceItem.data || ""),
          name: String(image.name || sourceItem.image_name || sourceItem.imageName || (hasTopLevelImage ? sourceItem.name : "") || ""),
        };
      };
      const mergeIncomingImage = (existing = {}, incoming = {}) => {
        const left = normalizeIncomingImage(existing);
        const right = normalizeIncomingImage(incoming);
        return {
          path: right.path || left.path,
          data: right.data || left.data,
          name: right.name || left.name,
        };
      };
      const normalizeIncomingRefList = (items = []) => Array.isArray(items)
        ? items
          .filter((item) => item && typeof item === "object")
          .map((item, index) => {
            return {
              id: String(item.id || item.name || `storyboard_ref_${index + 1}`),
              name: String(item.name || `Reference ${index + 1}`),
              description: String(item.description || ""),
              trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
              trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
              image: normalizeIncomingImage(item),
            };
          })
        : [];
      const incomingRefs = {
        subjects: normalizeIncomingRefList(incomingSource.subjects),
        locations: normalizeIncomingRefList(incomingSource.locations),
      };
      const mergeReferenceList = (current = [], incoming = []) => {
        const byKey = new Map();
        const keyFor = (item) => {
          const name = String(item?.name || "").trim().toLowerCase().replace(/\s+/g, " ");
          return name || String(item?.id || "").trim().toLowerCase();
        };
        for (const item of current) {
          const key = keyFor(item);
          if (key) byKey.set(key, { ...item, image: { ...(item.image || {}) } });
        }
        for (const item of incoming) {
          const key = keyFor(item);
          if (!key) continue;
          const existing = byKey.get(key) || {};
          byKey.set(key, {
            ...existing,
            ...item,
            image: mergeIncomingImage(existing.image, item.image),
          });
        }
        return Array.from(byKey.values());
      };
      if (incomingRefs.subjects.length) {
        refs.subjects = mergeReferenceList(refs.subjects, incomingRefs.subjects);
        refs.subject_count = refs.subjects.length;
      }
      if (incomingRefs.locations.length && !refs.locations_cleared) {
        refs.locations = mergeReferenceList(refs.locations, incomingRefs.locations);
      }
      if (!refs.subject_scene_map || typeof refs.subject_scene_map !== "object") refs.subject_scene_map = {};
      if (!refs.scene_map || typeof refs.scene_map !== "object") refs.scene_map = {};
      if (!refs.scene_trigger_map || typeof refs.scene_trigger_map !== "object") refs.scene_trigger_map = {};
      const segments = allEditableSegments();
      for (const item of Array.isArray(updates.scenes) ? updates.scenes : []) {
        const segment = segments.find((candidate) => candidate.id === item.id)
          || segments.find((candidate, index) => Number(index + 1) === Number(item.scene_number));
        if (!segment) continue;
        segment.no_character_present = Boolean(item.no_character_present || item.noCharacterPresent || item.no_subject || item.no_visible_subject);
        const validSubjectIds = new Set((refs.subjects || []).map((subject) => String(subject.id || "").trim()).filter(Boolean));
        const subjectIds = Array.isArray(item.subject_ids) ? item.subject_ids.map(String).filter((id) => validSubjectIds.has(id)) : [];
        if (!segment.no_character_present && subjectIds.length) refs.subject_scene_map[segment.id] = subjectIds;
        else delete refs.subject_scene_map[segment.id];
        const locationId = String(item.location_id || "").trim();
        if (locationId && !refs.locations_cleared) refs.scene_map[segment.id] = locationId;
        else delete refs.scene_map[segment.id];
        if (item.trigger_position) refs.location_trigger_position = String(item.trigger_position || "start") === "end" ? "end" : "start";
        if (Array.isArray(item.speaker_assignments) || Array.isArray(item.minimax_speaker_assignments) || Array.isArray(item.dialogue_cues)) {
          segment.minimax_speaker_assignments = normalizeMiniMaxSpeakerAssignments(item.speaker_assignments || item.minimax_speaker_assignments || item.dialogue_cues);
          syncMiniMaxSpeakerAssignmentLegacyFields(segment);
        }
      }
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      render();
      autoSaveSessionQuiet("Storyboard reference mapping update");
    };
    const applyStoryboardPrompts = (updates = {}) => {
      const segments = allEditableSegments()
        .slice()
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      // Storyboard prompt/beat application is allowed to update visual fields,
      // but it must never import lyric text from its older storyboard payload.
      // Keep the live line-review text as the source of truth for this operation.
      const lyricTextBySegmentId = new Map(
        segments.map((segment) => [String(segment.id || ""), String(segment.lyric_text || "")]),
      );
      let applied = 0;
      let storyChanged = false;
      let facialChanged = false;
      let beatChanged = false;
      let speakerChanged = false;
      const hasIncomingFacialDefault = Object.prototype.hasOwnProperty.call(updates, "facial_performance_default")
        || Object.prototype.hasOwnProperty.call(updates, "facialPerformance")
        || Object.prototype.hasOwnProperty.call(updates, "default_facial_performance");
      const hasIncomingFacialCustomDefault = Object.prototype.hasOwnProperty.call(updates, "facial_performance_custom_default")
        || Object.prototype.hasOwnProperty.call(updates, "facialPerformanceCustom")
        || Object.prototype.hasOwnProperty.call(updates, "default_facial_performance_custom");
      const hasIncomingStoryboardDefaults = Object.prototype.hasOwnProperty.call(updates, "builder_storyboard_defaults")
        || Object.prototype.hasOwnProperty.call(updates, "storyboard_defaults")
        || Object.prototype.hasOwnProperty.call(updates, "motion_defaults")
        || Object.prototype.hasOwnProperty.call(updates, "global_consistency_phrase")
        || Object.prototype.hasOwnProperty.call(updates, "performance_style_default")
        || Object.prototype.hasOwnProperty.call(updates, "performanceStyle")
        || Object.prototype.hasOwnProperty.call(updates, "video_style")
        || Object.prototype.hasOwnProperty.call(updates, "videoStyle")
        || Object.prototype.hasOwnProperty.call(updates, "video_style_custom")
        || Object.prototype.hasOwnProperty.call(updates, "videoStyleCustom")
        || Object.prototype.hasOwnProperty.call(updates, "temporal_world_effect")
        || Object.prototype.hasOwnProperty.call(updates, "temporalWorldEffect")
        || Object.prototype.hasOwnProperty.call(updates, "short_film_planning_mode")
        || Object.prototype.hasOwnProperty.call(updates, "shortFilmPlanningMode")
        || Object.prototype.hasOwnProperty.call(updates, "minimax_h3_cut_frequency")
        || Object.prototype.hasOwnProperty.call(updates, "cutFrequency");
      if (hasIncomingFacialDefault) {
        const nextDefault = String(updates.facial_performance_default ?? updates.facialPerformance ?? updates.default_facial_performance ?? "").trim();
        if (state.defaultFacialPerformance !== nextDefault) {
          state.defaultFacialPerformance = nextDefault;
          facialChanged = true;
        }
      }
      if (hasIncomingFacialCustomDefault) {
        const nextCustomDefault = String(updates.facial_performance_custom_default ?? updates.facialPerformanceCustom ?? updates.default_facial_performance_custom ?? "").trim();
        if (state.defaultFacialPerformanceCustom !== nextCustomDefault) {
          state.defaultFacialPerformanceCustom = nextCustomDefault;
          facialChanged = true;
        }
      }
      if (hasIncomingStoryboardDefaults) {
        state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults({
          ...state.builderStoryboardDefaults,
          ...(updates.builder_storyboard_defaults || updates.storyboard_defaults || {}),
          motion_defaults: updates.motion_defaults || updates.builder_storyboard_defaults?.motion_defaults || updates.storyboard_defaults?.motion_defaults || state.builderStoryboardDefaults?.motion_defaults || {},
          global_consistency_phrase: updates.global_consistency_phrase ?? updates.globalConsistencyPhrase ?? state.builderStoryboardDefaults?.global_consistency_phrase,
          performance_style: updates.performance_style_default ?? updates.performance_style ?? updates.performanceStyle ?? state.builderStoryboardDefaults?.performance_style,
          video_style: updates.video_style ?? updates.videoStyle ?? updates.builder_storyboard_defaults?.video_style ?? updates.storyboard_defaults?.video_style ?? state.builderStoryboardDefaults?.video_style,
          video_style_custom: updates.video_style_custom ?? updates.videoStyleCustom ?? updates.builder_storyboard_defaults?.video_style_custom ?? updates.storyboard_defaults?.video_style_custom ?? state.builderStoryboardDefaults?.video_style_custom,
          temporal_world_effect: updates.temporal_world_effect ?? updates.temporalWorldEffect ?? updates.builder_storyboard_defaults?.temporal_world_effect ?? updates.storyboard_defaults?.temporal_world_effect ?? state.builderStoryboardDefaults?.temporal_world_effect,
          temporal_world_effect_custom: updates.temporal_world_effect_custom ?? updates.temporalWorldEffectCustom ?? updates.builder_storyboard_defaults?.temporal_world_effect_custom ?? updates.storyboard_defaults?.temporal_world_effect_custom ?? state.builderStoryboardDefaults?.temporal_world_effect_custom,
          temporal_allow_background_extras: updates.temporal_allow_background_extras ?? updates.temporalAllowBackgroundExtras ?? updates.builder_storyboard_defaults?.temporal_allow_background_extras ?? updates.storyboard_defaults?.temporal_allow_background_extras ?? state.builderStoryboardDefaults?.temporal_allow_background_extras,
          temporal_background_intensity: updates.temporal_background_intensity ?? updates.temporalBackgroundIntensity ?? updates.builder_storyboard_defaults?.temporal_background_intensity ?? updates.storyboard_defaults?.temporal_background_intensity ?? state.builderStoryboardDefaults?.temporal_background_intensity,
          temporal_environment_time_passage: updates.temporal_environment_time_passage ?? updates.temporalEnvironmentTimePassage ?? updates.builder_storyboard_defaults?.temporal_environment_time_passage ?? updates.storyboard_defaults?.temporal_environment_time_passage ?? state.builderStoryboardDefaults?.temporal_environment_time_passage,
          temporal_protected_characters: updates.temporal_protected_characters ?? updates.temporalProtectedCharacters ?? updates.builder_storyboard_defaults?.temporal_protected_characters ?? updates.storyboard_defaults?.temporal_protected_characters ?? state.builderStoryboardDefaults?.temporal_protected_characters,
          temporal_protected_custom: updates.temporal_protected_custom ?? updates.temporalProtectedCustom ?? updates.builder_storyboard_defaults?.temporal_protected_custom ?? updates.storyboard_defaults?.temporal_protected_custom ?? state.builderStoryboardDefaults?.temporal_protected_custom,
          short_film_planning_mode: updates.short_film_planning_mode ?? updates.shortFilmPlanningMode ?? updates.builder_storyboard_defaults?.short_film_planning_mode ?? updates.storyboard_defaults?.short_film_planning_mode ?? state.builderStoryboardDefaults?.short_film_planning_mode,
          minimax_h3_cut_frequency: updates.minimax_h3_cut_frequency ?? updates.cut_frequency ?? updates.cutFrequency ?? updates.builder_storyboard_defaults?.minimax_h3_cut_frequency ?? updates.storyboard_defaults?.minimax_h3_cut_frequency ?? state.builderStoryboardDefaults?.minimax_h3_cut_frequency,
          camera_motion_speed: updates.camera_motion_speed ?? updates.cameraMotionSpeed ?? updates.motion_defaults?.camera_motion_speed ?? state.builderStoryboardDefaults?.camera_motion_speed,
          character_motion_speed: updates.character_motion_speed ?? updates.characterMotionSpeed ?? updates.motion_defaults?.character_motion_speed ?? state.builderStoryboardDefaults?.character_motion_speed,
        });
        storyChanged = true;
      }
      for (const scene of Array.isArray(updates.scenes) ? updates.scenes : []) {
        const segment = segments.find((candidate) => candidate.id === scene.id)
          || segments.find((candidate, index) => Number(index + 1) === Number(scene.scene_number));
        if (!segment) continue;
        ensureSegmentRuntimeFields(segment);
        if (Object.prototype.hasOwnProperty.call(scene, "lyric_no_lip_sync") || Object.prototype.hasOwnProperty.call(scene, "no_lip_sync")) {
          segment.lyric_no_lip_sync = Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync);
        }
        if (Array.isArray(scene.speaker_assignments) || Array.isArray(scene.minimax_speaker_assignments) || Array.isArray(scene.dialogue_cues)) {
          segment.minimax_speaker_assignments = normalizeMiniMaxSpeakerAssignments(scene.speaker_assignments || scene.minimax_speaker_assignments || scene.dialogue_cues);
          syncMiniMaxSpeakerAssignmentLegacyFields(segment);
          speakerChanged = true;
        }
        const imagePrompt = String(scene.image_prompt || "").trim();
        const videoPrompt = String(scene.video_prompt || scene.i2v_prompt || scene.t2v_prompt || "").trim();
        const videoType = String(scene.video_prompt_type || "").trim();
        segment.lyric_section = String(scene.lyric_section || scene.section || scene.song_section || segment.lyric_section || "").trim();
        const hasIncomingStoryBeat = Object.prototype.hasOwnProperty.call(scene, "story_beat")
          || Object.prototype.hasOwnProperty.call(scene, "scene_story_beat")
          || Object.prototype.hasOwnProperty.call(scene, "narrative_beat");
        const incomingStoryBeat = Object.prototype.hasOwnProperty.call(scene, "story_beat")
          ? scene.story_beat
          : Object.prototype.hasOwnProperty.call(scene, "scene_story_beat")
            ? scene.scene_story_beat
            : scene.narrative_beat;
        const nextStoryBeat = hasIncomingStoryBeat
          ? String(incomingStoryBeat || "").trim()
          : String(segment.story_beat || "").trim();
        if (segment.story_beat !== nextStoryBeat) beatChanged = true;
        segment.story_beat = nextStoryBeat;
        if (Object.prototype.hasOwnProperty.call(scene, "audio_direction")) segment.audio_direction = String(scene.audio_direction || "").trim();
        if (Object.prototype.hasOwnProperty.call(scene, "continuity") || Object.prototype.hasOwnProperty.call(scene, "continuity_direction")) {
          segment.continuity = String(scene.continuity || scene.continuity_direction || "").trim();
        }
        segment.flf_start_state = String(scene.flf_start_state || segment.flf_start_state || "").trim();
        segment.flf_transformation = String(scene.flf_transformation || segment.flf_transformation || "").trim();
        segment.flf_end_state = String(scene.flf_end_state || segment.flf_end_state || "").trim();
        segment.flf_carry_forward = String(scene.flf_carry_forward || segment.flf_carry_forward || "").trim();
        const nextFacial = String(scene.facial_performance ?? scene.facialPerformance ?? segment.facial_performance ?? "").trim();
        const nextFacialCustom = String(scene.facial_performance_custom ?? scene.facialPerformanceCustom ?? segment.facial_performance_custom ?? "").trim();
        if (segment.facial_performance !== nextFacial || segment.facial_performance_custom !== nextFacialCustom) facialChanged = true;
        segment.facial_performance = nextFacial;
        segment.facial_performance_custom = nextFacialCustom;
        if (Object.prototype.hasOwnProperty.call(scene, "video_style")) segment.minimax_h3_video_style = String(scene.video_style || "").trim();
        if (Object.prototype.hasOwnProperty.call(scene, "video_style_custom")) segment.minimax_h3_video_style_custom = String(scene.video_style_custom || "").trim();
        if (Object.prototype.hasOwnProperty.call(scene, "temporal_world_effect_override")) segment.temporal_world_effect_override = String(scene.temporal_world_effect_override || "global").trim();
        if (Object.prototype.hasOwnProperty.call(scene, "temporal_world_effect_custom")) segment.temporal_world_effect_custom = String(scene.temporal_world_effect_custom || "").trim();
        if (imagePrompt) {
          setSegmentPromptForEdit(segment, "t2i", imagePrompt);
          applied += 1;
        }
        if (videoPrompt) {
          if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
            segment.minimax_h3_prompt = applyMiniMaxH3NativeVoiceBlock(videoPrompt, segment);
            segment.minimax_h3_prompt_origin = normalizeVideoPromptOrigin(scene.video_prompt_origin);
          } else {
            setSegmentPromptForEdit(segment, "i2v", videoPrompt, {
              origin: scene.video_prompt_origin,
            });
          }
          applied += 1;
        }
        if (Object.prototype.hasOwnProperty.call(scene, "minimax_h3_pass2_prompt") || Object.prototype.hasOwnProperty.call(scene, "pass2_prompt")) {
          segment.minimax_h3_pass2_prompt = String(scene.minimax_h3_pass2_prompt ?? scene.pass2_prompt ?? "");
        }
        if (["i2v", "id_lora", "t2v", "rtv", "ingredients"].includes(videoType)) segment.video_prompt_type = videoType;
        if (String(scene.shot_type || "").trim()) segment.shot_type = String(scene.shot_type || "").trim();
        if (String(scene.camera_motion || "").trim()) segment.camera_motion = String(scene.camera_motion || "").trim();
      }
      for (const segment of segments) {
        const savedLyricText = lyricTextBySegmentId.get(String(segment.id || ""));
        if (savedLyricText !== undefined) segment.lyric_text = savedLyricText;
      }
      if (updates.story_layer || updates.storyLayer) {
        state.builderStoryLayer = normalizeBuilderStoryLayer(updates.story_layer || updates.storyLayer);
        storyChanged = true;
      }
      if (applied || storyChanged || facialChanged || beatChanged || speakerChanged) {
        ensureAllSegmentRuntimeFields();
        syncInspector();
        render();
        autoSaveSessionQuiet(applied ? "Storyboard prompt export" : "Storyboard scene beat export");
        if (applied) toast(`Storyboard prompts copied into Video Builder for ${applied} prompt field${applied === 1 ? "" : "s"}.`);
        else if (beatChanged) toast("Storyboard scene beats copied into Video Builder.");
      }
    };
    const findStoryboardSegment = (scene = {}) => {
      const segments = allEditableSegments()
        .slice()
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      return segments.find((candidate) => candidate.id === scene.id)
        || segments.find((candidate, index) => Number(index + 1) === Number(scene.scene_number));
    };
    const storyboardVideoExtraNotes = (scene = {}, storyboardPayload = {}, canonicalSegment = null) => {
      const selectedScene = Array.isArray(storyboardPayload?.scenes) && storyboardPayload.scenes.length ? storyboardPayload.scenes[0] : {};
      const fullyCustomShortFilm = normalizeMiniMaxShortFilmPlanningMode(storyboardPayload?.short_film_planning_mode) === "fully_custom";
      const storyLayer = selectedScene.story_layer || {};
      const vocalStatus = selectedScene.vocal_status || {};
      const ltxScene = normalizeProjectVideoEngine(selectedScene.project_video_engine || storyboardPayload?.project_video_engine || state.projectVideoEngine) === "ltx";
      const startingShot = selectedScene.starting_shot && typeof selectedScene.starting_shot === "object"
        ? selectedScene.starting_shot
        : null;
      const add = (parts, title, value) => {
        const text = String(value || "").trim();
        if (text) parts.push(`${title}:\n${text}`);
      };
      const parts = [];
      if (ltxScene) {
        const mappedSubjects = Array.isArray(selectedScene.subjects) ? selectedScene.subjects : [];
        const singleSubjectText = mappedSubjects.length === 1
          ? [mappedSubjects[0]?.name, mappedSubjects[0]?.description].filter(Boolean).join(" ").toLowerCase()
          : "";
        const singularPronouns = /\b(?:woman|girl|female|feminine|she|her)\b/.test(singleSubjectText)
          ? "she/her"
          : /\b(?:man|boy|male|masculine|he|him|his)\b/.test(singleSubjectText)
            ? "he/him"
            : "singular wording that repeats the mapped subject name when needed";
        const vocalContract = vocalStatus.should_lip_sync === true
          ? `This is a visible singing/lip-sync scene. The mapped performer visibly vocalizes ${JSON.stringify(String(vocalStatus.lyric_text || scene.lyrics || "").trim())} in sync with the audio using natural mouth, lip, cheek, and jaw movement. Never describe closed, still, sealed, relaxed-closed, or unmoving lips, and never call the performance silent or instrumental.`
          : "This is not a visible singing/lip-sync scene. Do not say anyone sings, vocalizes, mouths words, or lip-syncs, and do not quote the lyric as performed dialogue.";
        parts.push([
          "AUTHORITATIVE LTX ONE-PASS OUTPUT CONTRACT:",
          "Return one polished generation-ready prompt only. Do not print headings, field names, metadata labels, explanations, or this contract.",
          String(selectedScene.cut_plan?.instruction || "").trim() || "Use one continuous shot unless the scene explicitly requests a cut.",
          vocalContract,
          mappedSubjects.length === 1
            ? `This scene contains exactly one mapped subject. Use ${singularPronouns} consistently. Never use they/them/their or plural agreement for that person.`
            : "Use pronouns and singular/plural agreement that exactly match the mapped subject count.",
          "Integrate facial and performance guidance as natural visual prose; never output labels such as 'Facial performance direction:'.",
          "The first sentence is the sole opening-shot statement. After it, continue directly with new subject action; never restate that the subject is first shown, already shown, framed, introduced, or seen in the same opening shot.",
          "Describe each camera action exactly once. Do not repeat the opening framing, reveal, pull-back, or other camera direction.",
          "Use natural possessive anatomy phrasing such as 'the woman's eye' or 'the subject's eye'; never write awkward constructions such as 'one eye of the woman'.",
          "Write only complete grammatical sentences. Attach short descriptive additions with grammatical wording such as 'with subtle natural eye movement'; never append a bare comma fragment such as ', subtle natural eye movement.'.",
          "Treat first-frame visual inventory as optional visible detail, not a checklist. Mention only anatomy, wardrobe, accessories, and props that are actually inside the current framing. An eye, face, or upper-body shot must not claim that shoes, heels, feet, lower-body clothing, or other off-frame details are visible.",
        ].filter(Boolean).join("\n"));
      }
      if (fullyCustomShortFilm) {
        parts.push("FULLY CUSTOM SHORT FILM SOURCE CONTRACT:\nUse only the populated manual scene-card fields below. Do not infer or invent missing dialogue, speakers, actions, story beats, shots, camera moves, settings, sound, or continuity.");
      }
      if (startingShot?.required) {
        add(
          parts,
          "REQUIRED Storyboard opening shot",
          startingShot.instruction || `The video must explicitly begin with a ${startingShot.selected_starting_shot || scene.shot_type}. Begin all camera motion from that framing.`,
        );
      }
      add(parts, "Storyboard scene story beat", storyLayer.scene_story_beat || scene.story_beat);
      add(parts, "REQUIRED Storyboard video style", selectedScene.video_style);
      add(parts, "MANDATORY exact Storyboard video style verbiage — copy word-for-word", selectedScene.video_style_verbiage);
      add(parts, "MANDATORY exact temporal / world effect verbiage — copy word-for-word", selectedScene.temporal_world_effect_verbiage);
      if (!ltxScene) {
        const canonicalCutPlan = canonicalSegment ? miniMaxH3CutPlanForSegment(canonicalSegment) : null;
        add(parts, "MANDATORY Storyboard editing / cut plan", canonicalCutPlan?.instruction || selectedScene.cut_plan?.instruction);
      }
      const customMotionSummary = String(selectedScene.motion_summary || scene.motion_summary || scene.video_notes || "").trim();
      add(parts, "Storyboard motion/video summary", customMotionSummary);
      if (!customMotionSummary) add(parts, "Storyboard camera motion", selectedScene.camera_motion || scene.camera_motion);
      if (!fullyCustomShortFilm) add(parts, "REQUIRED Storyboard camera-flow framing", selectedScene.camera_flow_guidance);
      add(parts, "Storyboard camera motion speed guidance", selectedScene.camera_motion_speed_guidance || selectedScene.camera_guidance?.camera_motion_speed_guidance);
      add(parts, "Storyboard character motion guidance", selectedScene.character_motion_guidance || scene.character_motion);
      add(parts, "Storyboard performance direction", selectedScene.performance_direction || scene.performance_style);
      add(parts, "Storyboard facial performance direction", selectedScene.facial_performance_direction || scene.facial_performance_custom || scene.facial_performance);
      add(parts, "Storyboard lyric section", vocalStatus.lyric_section || scene.lyric_section);
      add(parts, "Storyboard first-frame visual inventory", selectedScene.first_frame_visual_inventory?.text || "");
      add(parts, "Exact manual audio / sound direction", selectedScene.audio_direction || scene.audio_direction);
      add(parts, "Exact manual continuity requirements", selectedScene.continuity || scene.continuity || scene.continuity_direction);
      return parts.join("\n\n");
    };
    const ensureStoryboardRequiredStartingShot = (prompt, segment, scene = {}, storyboardPayload = {}) => {
      const text = String(prompt || "").trim();
      const selectedScene = Array.isArray(storyboardPayload?.scenes) && storyboardPayload.scenes.length ? storyboardPayload.scenes[0] : {};
      const requirement = selectedScene.starting_shot && typeof selectedScene.starting_shot === "object"
        ? selectedScene.starting_shot
        : null;
      const shot = String(requirement?.selected_starting_shot || "").trim();
      if (!text || !requirement?.required || !shot) return text;
      const visibleSubject = Array.isArray(selectedScene.visible_subjects)
        ? String(selectedScene.visible_subjects.find((value) => String(value || "").trim()) || "").trim()
        : "";
      const subject = visibleSubject || String(scene.subjects || scene.subject || "").split(/[,;\n]+/)[0].trim() || "the subject";
      const shotKey = shot.toLowerCase().replace(/[\s_-]+/g, " ").trim();
      let sentence = "";
      if (shotKey === "eyes shot") sentence = `The video begins with an extreme close-up of ${subject}'s eyes.`;
      else if (shotKey === "mouth shot") sentence = `The video begins with an extreme close-up of ${subject}'s mouth.`;
      else if (shotKey === "hands shot") sentence = `The video begins with a close-up of ${subject}'s hands.`;
      else if (shotKey === "feet shot") sentence = `The video begins with a close-up of ${subject}'s feet.`;
      else sentence = `The video begins with ${/^[aeiou]/i.test(shot) ? "an" : "a"} ${shot} of ${subject === "the subject" ? "the scene" : subject}.`;
      const opening = text.slice(0, 500);
      const hasOpeningMarker = /\b(?:the\s+video\s+)?(?:begins?|starts?|opens?)\s+with\b|\b(?:opening|first)\s+(?:shot|frame)\b/i.test(opening);
      const shotWords = shotKey.split(/[^a-z0-9]+/).filter((word) => word && word !== "shot");
      const hasRequiredFraming = shotKey === "eyes shot"
        ? /\beyes?\b/i.test(opening)
        : shotWords.length > 0 && shotWords.every((word) => new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(opening));
      if (hasOpeningMarker && hasRequiredFraming) return text;
      if (currentVideoMode() === "id_lora" && /\[VISUAL\]\s*:?/i.test(text)) {
        return text.replace(/(\[VISUAL\]\s*:?\s*)/i, `$1${sentence} `);
      }
      const ensured = `${sentence} ${text}`.trim();
      return applyMappedTriggerPhrases(
        applyTriggerPhrase(ensured, videoTriggerPhraseForSegment(segment)),
        segment,
        { ensureTransitionLast: true },
      );
    };
    const ensureStoryboardRequiredVideoStyle = (prompt, storyboardPayload = {}) => {
      const text = String(prompt || "").trim();
      const selectedScene = Array.isArray(storyboardPayload?.scenes) && storyboardPayload.scenes.length
        ? storyboardPayload.scenes[0]
        : {};
      const verbiage = String(selectedScene.video_style_verbiage || "").trim();
      if (!text || !verbiage || text.includes(verbiage)) return text;
      const requiredLine = verbiage;
      const firstParagraphEnd = text.indexOf("\n\n");
      if (firstParagraphEnd >= 0) {
        return `${text.slice(0, firstParagraphEnd)}\n\n${requiredLine}${text.slice(firstParagraphEnd)}`;
      }
      const firstLineEnd = text.indexOf("\n");
      if (firstLineEnd >= 0) {
        return `${text.slice(0, firstLineEnd)}\n\n${requiredLine}\n${text.slice(firstLineEnd + 1)}`;
      }
      return `${text}\n\n${requiredLine}`;
    };
    const ensureStoryboardRequiredTemporalWorldEffect = (prompt, storyboardPayload = {}, scene = {}) => {
      let text = String(prompt || "").trim();
      const storyboardScenes = Array.isArray(storyboardPayload?.scenes) ? storyboardPayload.scenes : [];
      const selectedScene = storyboardScenes.find((candidate) => candidate?.id && candidate.id === scene?.id)
        || storyboardScenes.find((candidate) => Number(candidate?.scene_number) === Number(scene?.scene_number))
        || (storyboardScenes.length ? storyboardScenes[0] : scene);
      const effect = selectedScene.temporal_world_effect || {};
      const verbiage = String(selectedScene.temporal_world_effect_verbiage || effect.exact_verbiage || "").trim();
      if (!text || !verbiage) return text;

      // Canonicalize the exact contract at the end of the prompt. Gemma sometimes
      // embeds or paraphrases it inside a shot; remove that copy and append the
      // builder-owned contract once in its canonical form.
      text = text.split(verbiage).join("").replace(/\n{3,}/g, "\n\n").trim();
      text = `${text}\n\n${verbiage}`.trim();

      return text.trim();
    };
    const normalizeStoryboardInputAudioVocalTimeline = (prompt, segment) => {
      let text = String(prompt || "").trim();
      const settings = miniMaxH3SettingsForSegment(segment);
      const lyric = isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
      if (!text || !lyric || settings.audio_mode !== "input_audio" || effectiveVideoPerformanceModeForSegment(segment) !== "singing") return text;
      const timestampPattern = /(\[\d+(?:\.\d+)?s\s*[-–—]\s*\d+(?:\.\d+)?s\]\s*\n)([\s\S]*?)(?=\n\s*\[\d+(?:\.\d+)?s\s*[-–—]\s*\d+(?:\.\d+)?s\]|\n\s*Audio:|\n\s*Continuity:|$)/g;
      const blockCount = Array.from(text.matchAll(timestampPattern)).length;
      if (!blockCount) return text;
      const repeatedVocalBoilerplate = /During only the portion of this interval where the sung lyric is audible in Audio 1,[\s\S]*?never stretch, restart, or repeat the line to fill the interval\./gi;
      const quotedLyricVariants = [`“${lyric}”`, `\"${lyric}\"`];
      let blockIndex = 0;
      text = text.replace(timestampPattern, (whole, header, body) => {
        const phaseDirection = blockIndex === 0
          ? "Only while vocals are actually audible in this interval, the assigned singer begins the currently audible portion of the assigned lyric in exact sync with Audio 1."
          : blockIndex === blockCount - 1
            ? "Only while vocals remain audible in this interval, the assigned singer completes any remaining portion of the assigned lyric in exact sync with Audio 1; once the supplied vocal ends, the mouth closes or relaxes naturally."
            : "Only while vocals are actually audible in this interval, the assigned singer continues the currently audible portion of the assigned lyric in exact sync with Audio 1 without restarting or repeating it.";
        blockIndex += 1;
        let cleanBody = String(body || "").replace(repeatedVocalBoilerplate, "").trim();
        quotedLyricVariants.forEach((quoted) => {
          cleanBody = cleanBody.split(quoted).join("the assigned lyric");
        });
        if (!cleanBody.includes(phaseDirection)) cleanBody = `${cleanBody}\n\n${phaseDirection}`.trim();
        return `${header}${cleanBody}\n`;
      });
      return text.trim();
    };
    const storyboardSceneCloneForI2V = (segment, scene = {}) => {
      const clone = {
        ...segment,
        lyric_text: String(scene.lyrics || scene.lyric_text || segment.lyric_text || "").trim(),
        lyric_section: String(scene.lyric_section || scene.section || scene.song_section || segment.lyric_section || "").trim(),
        lyric_singers: Array.isArray(segment.lyric_singers) && segment.lyric_singers.length
          ? [...segment.lyric_singers]
          : (Array.isArray(scene.lyric_singers) ? [...scene.lyric_singers] : []),
        lyric_shot_word_timing_enabled: Array.isArray(segment.lyric_cue_map) && segment.lyric_cue_map.length
          ? Boolean(segment.lyric_shot_word_timing_enabled)
          : Boolean(scene.lyric_shot_word_timing_enabled ?? segment.lyric_shot_word_timing_enabled),
        lyric_performance_mode: Array.isArray(segment.lyric_cue_map) && segment.lyric_cue_map.length
          ? "cue_map"
          : String(scene.lyric_performance_mode || segment.lyric_performance_mode || "together"),
        lyric_cue_map: Array.isArray(segment.lyric_cue_map) && segment.lyric_cue_map.length
          ? segment.lyric_cue_map.map((cue) => ({ ...cue }))
          : (Array.isArray(scene.lyric_cue_map) ? scene.lyric_cue_map.map((cue) => ({ ...cue })) : []),
        minimax_speaker_assignments: normalizeMiniMaxSpeakerAssignments(scene.speaker_assignments || scene.minimax_speaker_assignments || segment.minimax_speaker_assignments),
        lyric_no_lip_sync: Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync || segmentUsesNoLipSyncPerformance(segment)),
        no_character_present: Boolean(scene.no_character_present || scene.noCharacterPresent || segment.no_character_present),
        story_beat: String(scene.story_beat || scene.scene_story_beat || scene.narrative_beat || segment.story_beat || "").trim(),
        flf_start_state: String(scene.flf_start_state || segment.flf_start_state || "").trim(),
        flf_transformation: String(scene.flf_transformation || segment.flf_transformation || "").trim(),
        flf_end_state: String(scene.flf_end_state || segment.flf_end_state || "").trim(),
        flf_carry_forward: String(scene.flf_carry_forward || segment.flf_carry_forward || "").trim(),
        shot_type: String(scene.shot_type || segment.shot_type || "").trim(),
        camera_motion: String(scene.camera_motion || segment.camera_motion || "").trim(),
        camera_motion_speed: Number(scene.camera_motion_speed ?? segment.camera_motion_speed ?? state.builderStoryboardDefaults?.camera_motion_speed ?? 4),
        camera_motion_speed_guidance: String(scene.camera_motion_speed_guidance || segment.camera_motion_speed_guidance || state.builderStoryboardDefaults?.camera_guidance || "").trim(),
        character_motion: String(scene.character_motion || segment.character_motion || "").trim(),
        character_motion_speed: Number(scene.character_motion_speed ?? segment.character_motion_speed ?? state.builderStoryboardDefaults?.character_motion_speed ?? 4),
        character_motion_guidance: String(scene.character_motion_guidance || segment.character_motion_guidance || state.builderStoryboardDefaults?.character_guidance || "").trim(),
        subject_refs: Array.isArray(scene.subject_refs) ? scene.subject_refs : Array.isArray(segment.subject_refs) ? segment.subject_refs : [],
        mapped_subjects: Array.isArray(scene.subject_refs)
          ? scene.subject_refs.map((subject) => [subject?.name, subject?.description].filter(Boolean).join(": ")).filter(Boolean).join("\n")
          : String(segment.mapped_subjects || segment.subject || ""),
        facial_performance: String(scene.facial_performance ?? scene.facialPerformance ?? segment.facial_performance ?? "").trim(),
        facial_performance_custom: String(scene.facial_performance_custom ?? scene.facialPerformanceCustom ?? segment.facial_performance_custom ?? "").trim(),
        performance_mode: String(scene.performance_mode || scene.performanceMode || segment.performance_mode || "").trim(),
        prompt_summary: String(scene.prompt_summary || scene.summary || segment.prompt_summary || segment.summary || "").trim(),
        summary: String(scene.prompt_summary || scene.summary || segment.prompt_summary || segment.summary || "").trim(),
        audio_direction: String(scene.audio_direction || segment.audio_direction || "").trim(),
        continuity: String(scene.continuity || scene.continuity_direction || segment.continuity || "").trim(),
        temporal_world_effect_override: String(scene.temporal_world_effect_override || segment.temporal_world_effect_override || "global").trim(),
        temporal_world_effect_custom: String(scene.temporal_world_effect_custom || segment.temporal_world_effect_custom || "").trim(),
      };
      const imageData = String(scene.image_data || scene.image_reference_data || "").trim();
      const imagePath = String(scene.image_path || scene.approved_image_path || "").trim();
      if (imageData) {
        clone.image_history = [];
        clone.image_history_index = 0;
        clone.custom_image_data = imageData;
        clone.custom_image_path = "";
        clone.approved_image_path = "";
      } else if (imagePath) {
        clone.image_history = [imagePath];
        clone.image_history_index = 0;
        clone.custom_image_data = "";
        clone.custom_image_path = "";
        clone.approved_image_path = "";
      }
      const imagePrompt = String(scene.image_prompt || scene.t2i_prompt || "").trim();
      if (imagePrompt) {
        clone.t2i_prompt = imagePrompt;
        clone.flux_prompt = imagePrompt;
        clone.notes = imagePrompt;
      } else if (clone.prompt_summary && !String(clone.notes || "").trim()) {
        clone.notes = clone.prompt_summary;
      }
      return clone;
    };
    const createStoryboardVideoPromptViaBuilder = async (scene = {}, options = {}) => {
      const segment = findStoryboardSegment(scene);
      if (!segment) throw new Error(`Scene ${scene.scene_number || ""}: matching Video Builder scene was not found.`);
      await ensureAutoTimedSingerCuesBeforePrompt(segment);
      const workingSegment = storyboardSceneCloneForI2V(segment, scene);
      const extraUserNotes = storyboardVideoExtraNotes(scene, options.storyboardPayload || {}, workingSegment);
      if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        const mode = miniMaxH3ModeForSegment(segment);
        const modeLabel = miniMaxH3ModeLabel(mode);
        const storyboardPlanningMode = normalizeMiniMaxShortFilmPlanningMode(
          options.storyboardPayload?.short_film_planning_mode
          || options.storyboardPayload?.shortFilmPlanningMode
          || state.builderStoryboardDefaults?.short_film_planning_mode,
        );
        const storyboardInstructionKey = miniMaxH3InstructionKey(mode);
        const customSourceContract = storyboardPlanningMode === "fully_custom"
          ? "FULLY CUSTOM SHORT FILM: Every populated scene-card field is authoritative. Format only what the user supplied. Do not invent, rewrite, reorder, merge, omit, or replace dialogue, speakers, story beats, actions, shot/framing, camera motion, setting, references, audio direction, sound, or continuity. Leave unspecified choices unspecified."
          : "GUIDED SHORT FILM: Preserve exact dialogue and speaker order while using the supplied film-planning fields to stage a coherent narrative scene.";
        workingSegment.minimax_h3_mode = mode;
        workingSegment.minimax_h3_reference_keys = Array.isArray(segment.minimax_h3_reference_keys)
          ? [...segment.minimax_h3_reference_keys]
          : null;
        workingSegment.minimax_h3_framing_shot_ids = Array.isArray(segment.minimax_h3_framing_shot_ids)
          ? [...segment.minimax_h3_framing_shot_ids]
          : [];
        workingSegment.minimax_h3_video_references = (Array.isArray(segment.minimax_h3_video_references)
          ? segment.minimax_h3_video_references
          : []).map((item) => ({ ...item }));
        const rendererPromptImages = miniMaxH3PromptVisionImages(workingSegment, mode);
        const visionImages = miniMaxH3PromptVisionImagesForRunner(workingSegment, mode);
        const rendererReferenceImages = ["reference_to_video", "image_reference_to_video"].includes(mode)
          ? miniMaxOrderedImageReferenceItemsForSegment(workingSegment, mode)
          : [];
        const sceneImageUse = miniMaxH3SceneImageUseForSegment(workingSegment);
        const sceneImageSourceAvailable = Boolean(segmentImageSource(workingSegment)?.path || segmentImageSource(workingSegment)?.data);
        if (mode === "image_to_video" && !rendererPromptImages.length) {
          throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: MiniMax Image to Video needs a selected scene image.`);
        }
        if (mode === "image_reference_to_video" && !sceneImageSourceAvailable) {
          throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: MiniMax Image to Video 2 Pass needs a selected scene image.`);
        }
        if (mode === "reference_to_video" && sceneImageUse !== "off" && !sceneImageSourceAvailable) {
          throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: the selected scene-image mode needs a timeline image for LLM prompting.`);
        }
        if (mode === "reference_to_video" && !rendererReferenceImages.length) {
          throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: MiniMax Reference to Video needs ordered Reference Builder images.`);
        }
        assertMiniMaxH3ReferenceDescriptionsReady(workingSegment, mode);
        if (mode === "video_to_video" && !workingSegment.minimax_h3_video_references.some((item) => String(item?.path || "").trim())) {
          throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: MiniMax Video to Video needs a reference video.`);
        }
        if (visionImages.length && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
          throw new Error("MiniMax image/reference prompting needs a vision-capable API model selected in LLM Runner.");
        }
        options.progress?.set(
          `${options.progressLabel || scene.label || sceneDisplayName(segment, segmentIndexInfo(segment).index)}: creating MiniMax ${modeLabel} prompt with the scene's H3 instructions...\n${gemmaRunnerLine({ vision: Boolean(visionImages.length) })}`,
          Math.min(92, Number(options.progressPercent || 35) + 18),
        );
        const data = await runMiniMaxH3PromptGeneration(workingSegment, mode, {
          projectFolder: activeProjectFolderForSave(),
          sceneId: segment.id || "",
          builderInstructionKey: storyboardInstructionKey,
          contextOptions: {
            storyboardContext: extraUserNotes,
            storyboardPayload: options.storyboardPayload || {},
            storyboardScene: scene,
          },
          userNotes: customSourceContract,
          finalizePrompt: (prompt) => ensureStoryboardRequiredTemporalWorldEffect(
            ensureBuilderManagedFx(prompt, scene),
            options.storyboardPayload || {},
            scene,
          ),
          unloadAfter: options.unloadAfter !== false,
          emptyPromptMessage: `${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: LLM returned an empty MiniMax ${modeLabel} prompt.`,
        });
        const prompt = data.prompt;
        if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: LLM returned an empty MiniMax ${modeLabel} prompt.`);
        if (Array.isArray(workingSegment.minimax_h3_framing_shot_ids)) {
          segment.minimax_h3_framing_shot_ids = [...workingSegment.minimax_h3_framing_shot_ids];
        }
        return {
          ...data,
          prompt,
          already_finalized: true,
          minimax_h3_mode: mode,
          used_minimax_h3_instructions: true,
        };
      }
      const storyboardImageReference = getI2VImageReference(workingSegment);
      const forceTextOnly = currentVideoMode() === "i2v" && !storyboardImageReference.path && !storyboardImageReference.data;
      const request = buildI2VPromptRequestForSegment(workingSegment, {
        unloadAfter: options.unloadAfter !== false,
        extraUserNotes,
        skipStoryboardExtraNotes: true,
        forceTextOnly,
      });
      options.progress?.set(`${options.progressLabel || scene.label || sceneDisplayName(segment, segmentIndexInfo(segment).index)}: creating prompt through Video Builder ${request.modeLabel} payload...\n${forceTextOnly ? "No scene image found, using text-only storyboard fallback.\n" : ""}${gemmaRunnerLine({ vision: request.useImageReference })}`, Math.min(92, Number(options.progressPercent || 35) + 18));
      const data = await postJson(request.endpoint, request.payload, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
      const finalizedPrompt = await finalizeVideoPromptForSegment(
        workingSegment,
        data.prompt,
        options.progress || null,
        Math.min(98, Number(options.progressPercent || 35) + 35),
        `${options.progressLabel || "Storyboard Gemma"}: enhancement pass`,
        { unloadAfter: request.useImageReference ? true : options.unloadAfter !== false },
      );
      const promptWithStartingShot = ensureStoryboardRequiredStartingShot(
        finalizedPrompt,
        workingSegment,
        scene,
        options.storyboardPayload || {},
      );
      const prompt = ensureBuilderManagedFx(promptWithStartingShot, scene);
      if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty I2V prompt.`);
      return {
        ...data,
        prompt,
        already_finalized: true,
        used_video_builder_i2v_payload: true,
      };
    };
    storyboardPipeline.runner = createStoryboardVideoPromptViaBuilder;
    ACTIVE_STORYBOARD_PROMPT_PIPELINE = createStoryboardVideoPromptViaBuilder;
    if (options.registerPromptPipelineOnly) return;
    const applyIdLoraDialoguePlanFromStoryboard = async (updates = {}) => {
      const scenes = Array.isArray(updates.scenes)
        ? updates.scenes.filter((scene) => String(scene?.lyrics || scene?.story_beat || scene?.image_prompt || "").trim())
        : [];
      if (!scenes.length) throw new Error("No reviewed ID-LoRA dialogue scenes were found to apply.");

      const hasSceneWork = (segment) => {
        return [
          segment?.lyric_text,
          segment?.lyric_note,
          segment?.lyrics,
          segment?.story_beat,
          segment?.notes,
          segment?.timeline_note,
          segment?.t2i_prompt,
          segment?.flux_prompt,
          segment?.flux_klein_prompt,
          segment?.nb_prompt,
          segment?.flow_gpt_prompt,
          segment?.ernie_t2i_prompt,
          segment?.i2v_prompt,
          segment?.t2v_prompt,
          segment?.video_path,
          selectedSegmentImagePath(segment),
        ].some((value) => String(value || "").trim());
      };
      const currentBase = Array.isArray(state.segments) ? state.segments : [];
      const currentOverlays = Array.isArray(state.overlaySegments) ? state.overlaySegments : [];
      const starterIsBlank = currentBase.length <= 1 && currentOverlays.length === 0 && currentBase.every((segment) => !hasSceneWork(segment));
      if (!starterIsBlank) {
        const ok = window.confirm(
          `Apply ${scenes.length} ID-LoRA dialogue scene${scenes.length === 1 ? "" : "s"} to Video Builder?\n\nThis will replace the current base timeline scenes and clear insert scenes.`,
        );
        if (!ok) return { message: "ID-LoRA dialogue plan apply cancelled." };
      }

      const idForStoryboardRef = (rawId, items = [], prefix = "") => {
        const raw = String(rawId || "").trim();
        if (!raw) return "";
        const clean = (value) => String(value || "").replace(/[^a-z0-9_-]+/gi, "_");
        const direct = items.find((item) => String(item?.id || "") === raw);
        if (direct) return String(direct.id || "");
        const prefixed = items.find((item) => `${prefix}_${clean(item?.id)}` === raw);
        if (prefixed) return String(prefixed.id || "");
        const byName = items.find((item) => clean(item?.name).toLowerCase() === clean(raw).toLowerCase());
        return byName ? String(byName.id || "") : "";
      };
      const sceneImagePrompt = (scene = {}) => String(
        scene.image_prompt || scene.t2i_prompt || scene.prompt_summary || scene.story_beat || "",
      ).trim();
      const sceneVideoNotes = (scene = {}) => [
        String(scene.motion_summary || "").trim(),
        String(scene.video_notes || "").trim(),
        String(scene.camera_motion || "").trim() ? `Camera: ${String(scene.camera_motion || "").trim()}` : "",
        String(scene.facial_performance_custom || scene.facial_performance || "").trim()
          ? `Performance: ${String(scene.facial_performance_custom || scene.facial_performance || "").trim()}`
          : "",
      ].filter(Boolean).join("\n");

      pushHistory();
      const refs = normalizeIdLoraReferenceBuilder(state.idLoraReferenceBuilder);
      refs.scene_map = {};
      const nextSegments = [];
      let cursor = 0;
      scenes.forEach((scene, index) => {
        const dialogue = String(scene.lyrics || scene.dialogue || scene.lyric_text || "").trim();
        const duration = estimateIdLoraDialogueDuration(dialogue || scene.story_beat || scene.image_prompt);
        const segment = ensureSegmentRuntimeFields(newSegment(cursor, cursor + duration));
        segment.label = String(scene.label || `Scene ${index + 1}`).trim() || `Scene ${index + 1}`;
        segment.lyric_text = dialogue;
        segment.lyric_section = String(scene.lyric_section || scene.section || scene.song_section || "").trim();
        segment.story_beat = String(scene.story_beat || scene.scene_story_beat || scene.narrative_beat || "").trim();
        segment.notes = sceneImagePrompt(scene);
        segment.i2v_notes = sceneVideoNotes(scene);
        segment.video_prompt_type = "id_lora";
        segment.facial_performance = String(scene.facial_performance ?? scene.facialPerformance ?? "").trim();
        segment.facial_performance_custom = String(scene.facial_performance_custom ?? scene.facialPerformanceCustom ?? "").trim();
        segment.shot_type = String(scene.shot_type || "").trim();
        segment.camera_motion = String(scene.camera_motion || "").trim();
        const imagePrompt = sceneImagePrompt(scene);
        if (imagePrompt) {
          setSegmentPromptForEdit(segment, "t2i", imagePrompt);
          segment.flux_prompt = imagePrompt;
          segment.flux_klein_prompt = imagePrompt;
          segment.nb_prompt = imagePrompt;
          segment.flow_gpt_prompt = imagePrompt;
          segment.ernie_t2i_prompt = imagePrompt;
        }
        const videoPrompt = String(scene.video_prompt || scene.i2v_prompt || scene.t2v_prompt || "").trim();
        if (videoPrompt) {
          setSegmentPromptForEdit(segment, "i2v", videoPrompt, {
            origin: scene.video_prompt_origin || scene.i2v_prompt_origin,
          });
        }
        const imagePath = String(scene.image_path || scene.approved_image_path || "").trim();
        const imageData = String(scene.image_data || scene.image_reference_data || "").trim();
        if (imageData) {
          segment.custom_image_data = imageData;
          segment.custom_image_path = "";
          segment.image_history = [];
          segment.image_history_index = 0;
        } else if (imagePath) {
          segment.image_history = [imagePath];
          segment.image_history_index = 0;
          segment.custom_image_path = "";
          segment.custom_image_data = "";
        }

        const subjectRefs = Array.isArray(scene.subject_refs) ? scene.subject_refs : [];
        const rawCharacterId = scene.id_lora_character_id || scene.character_id || scene.subject_id || subjectRefs[0]?.id || subjectRefs[0]?.name || "";
        const rawLocationId = scene.id_lora_location_id || scene.location_id || scene.location_ref?.id || scene.location_ref?.name || "";
        refs.scene_map[segment.id] = {
          character_id: idForStoryboardRef(rawCharacterId, refs.characters, "id_lora_character"),
          location_id: idForStoryboardRef(rawLocationId, refs.locations, "id_lora_location"),
          dialogue,
          auto_duration: true,
          manual_duration: duration,
          estimated_duration: duration,
        };
        nextSegments.push(segment);
        cursor += duration;
      });

      state.videoModelMode = "id_lora";
      state.segments = nextSegments;
      state.overlaySegments = [];
      state.activeId = nextSegments[0]?.id || "";
      state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(refs);
      if (updates.story_layer || updates.storyLayer) state.builderStoryLayer = normalizeBuilderStoryLayer(updates.story_layer || updates.storyLayer);
      sortSegments(state.segments);
      state.duration = timelineDuration();
      ensureAllSegmentRuntimeFields();
      syncVideoModePanel();
      syncI2VVideoSettingsPanel();
      syncInspector();
      render();
      await autoSaveSessionQuiet("ID-LoRA dialogue plan applied from storyboard");
      return {
        message: `Applied ${nextSegments.length} ID-LoRA dialogue scene${nextSegments.length === 1 ? "" : "s"} to Video Builder.`,
      };
    };
    const applyMiniMaxDialoguePlanFromStoryboard = async (updates = {}) => {
      const scenes = Array.isArray(updates.scenes)
        ? updates.scenes.filter((scene) => String(scene?.lyrics || scene?.story_beat || scene?.image_prompt || "").trim())
        : [];
      if (!scenes.length) throw new Error("No reviewed MiniMax storyboard scenes were found to create timeline segments.");
      const hasSceneWork = (segment) => [
        segment?.lyric_text,
        segment?.story_beat,
        segment?.notes,
        segment?.i2v_notes,
        segment?.t2i_prompt,
        segment?.minimax_h3_prompt,
        segment?.video_path,
        selectedSegmentImagePath(segment),
      ].some((value) => String(value || "").trim());
      const currentBase = Array.isArray(state.segments) ? state.segments : [];
      const currentOverlays = Array.isArray(state.overlaySegments) ? state.overlaySegments : [];
      const starterIsBlank = currentBase.length <= 1 && currentOverlays.length === 0 && currentBase.every((segment) => !hasSceneWork(segment));
      if (!starterIsBlank) {
        const ok = window.confirm(`Create ${scenes.length} MiniMax timeline segment${scenes.length === 1 ? "" : "s"}?\n\nThis will replace the current base timeline scenes and clear insert scenes.`);
        if (!ok) return { message: "Create MiniMax timeline segments cancelled." };
      }
      pushHistory();
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      refs.subject_scene_map = {};
      refs.extra_scene_map = {};
      refs.scene_map = {};
      const validSubjectIds = new Set((refs.subjects || []).map((subject) => String(subject.id || "")).filter(Boolean));
      const validLocationIds = new Set((refs.locations || []).map((location) => String(location.id || "")).filter(Boolean));
      const nextSegments = [];
      let cursor = 0;
      scenes.forEach((scene, index) => {
        const dialogue = String(scene.lyrics || scene.dialogue || scene.lyric_text || "").trim();
        const requestedDuration = Number(scene.exact_duration || scene.duration || 0);
        const duration = requestedDuration > 0
          ? Math.max(1, Math.min(15, requestedDuration))
          : Math.max(1, Math.min(15, estimateIdLoraDialogueDuration(dialogue || scene.story_beat || scene.image_prompt)));
        const segment = ensureSegmentRuntimeFields(newSegment(cursor, cursor + duration));
        segment.label = String(scene.label || `Scene ${index + 1}`).trim() || `Scene ${index + 1}`;
        segment.lyric_text = dialogue;
        segment.lyric_section = String(scene.lyric_section || "").trim();
        segment.lyric_singers = Array.isArray(scene.lyric_singers) ? scene.lyric_singers.map(String).filter(Boolean) : [];
        segment.minimax_speaker_assignments = normalizeMiniMaxSpeakerAssignments(scene.speaker_assignments || scene.dialogue_cues || []);
        syncMiniMaxSpeakerAssignmentLegacyFields(segment);
        segment.performance_mode = "speaking";
        segment.story_beat = String(scene.story_beat || "").trim();
        segment.notes = String(scene.prompt_summary || scene.summary || scene.notes || scene.image_prompt || "").trim();
        segment.i2v_notes = [scene.motion_summary, scene.character_motion, scene.camera_motion ? `Camera: ${scene.camera_motion}` : ""]
          .map((value) => String(value || "").trim()).filter(Boolean).join("\n");
        segment.audio_direction = String(scene.audio_direction || "").trim();
        segment.continuity = String(scene.continuity || scene.continuity_direction || "").trim();
        segment.minimax_h3_mode = normalizeMiniMaxH3Mode(scene.minimax_h3_mode || state.miniMaxH3Settings?.video_mode);
        segment.facial_performance = String(scene.facial_performance || "").trim();
        segment.facial_performance_custom = String(scene.facial_performance_custom || "").trim();
        segment.shot_type = String(scene.shot_type || "").trim();
        segment.camera_motion = String(scene.camera_motion || "").trim();
        segment.character_motion = String(scene.character_motion || "").trim();
        segment.temporal_world_effect_override = String(scene.temporal_world_effect_override || "global").trim();
        segment.temporal_world_effect_custom = String(scene.temporal_world_effect_custom || "").trim();
        const imagePrompt = String(scene.image_prompt || scene.t2i_prompt || "").trim();
        if (imagePrompt) setSegmentPromptForEdit(segment, "t2i", imagePrompt);
        const videoPrompt = String(scene.video_prompt || "").trim();
        if (videoPrompt) {
          segment.minimax_h3_prompt = applyMiniMaxH3NativeVoiceBlock(videoPrompt, segment);
          segment.minimax_h3_prompt_origin = normalizeVideoPromptOrigin(scene.video_prompt_origin);
        }
        if (Object.prototype.hasOwnProperty.call(scene, "minimax_h3_pass2_prompt") || Object.prototype.hasOwnProperty.call(scene, "pass2_prompt")) {
          segment.minimax_h3_pass2_prompt = String(scene.minimax_h3_pass2_prompt ?? scene.pass2_prompt ?? "");
        }
        const subjectIds = (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
          .map((subject) => String(subject?.id || ""))
          .filter((id) => validSubjectIds.has(id));
        if (subjectIds.length) refs.subject_scene_map[segment.id] = subjectIds;
        const locationId = String(scene.location_ref?.id || scene.location_id || "").trim();
        if (locationId && validLocationIds.has(locationId)) refs.scene_map[segment.id] = locationId;
        nextSegments.push(segment);
        cursor += duration;
      });
      state.segments = nextSegments;
      state.overlaySegments = [];
      state.activeId = nextSegments[0]?.id || "";
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      state.builderStoryLayer = normalizeBuilderStoryLayer(updates.story_layer || updates.storyLayer || state.builderStoryLayer);
      state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults({
        ...state.builderStoryboardDefaults,
        short_film_planning_mode: updates.short_film_planning_mode || "guided_film",
      });
      sortSegments(state.segments);
      state.duration = timelineDuration();
      ensureAllSegmentRuntimeFields();
      syncVideoModePanel();
      syncI2VVideoSettingsPanel();
      syncInspector();
      render();
      await autoSaveSessionQuiet("MiniMax timeline segments created from storyboard");
      return { message: `Created ${nextSegments.length} MiniMax timeline segment${nextSegments.length === 1 ? "" : "s"} from the reviewed storyboard.` };
    };
    const storyboardRunnerSettings = textGemmaRunnerPayload();
    const storyboardUsesQwen = state.textGemmaRunner === "qwen_local";
    const storyboardSelectedModel = storyboardUsesQwen
      ? String(storyboardRunnerSettings.qwen_model_file || "").trim()
      : String(storyboardRunnerSettings.gemma_model_file || "").trim();
    const storyboardSelectedMmproj = storyboardUsesQwen
      ? String(storyboardRunnerSettings.qwen_mmproj_file || "").trim()
      : String(i2vMmprojSelect.value || mmprojSelect.value || "").trim();
    window.VRGDGStoryboardBuilder.open({
      promptActionOnly: options.promptActionOnly === true,
      focusedSection: options.focusedSection,
      allowImagePrep: options.allowImagePrep,
      onClose: options.onClose,
      onFocusedSave: async (updates) => {
        applyStoryboardPrompts(updates);
        await saveSession({ quiet: true, throwOnError: true });
      },
      focusSceneId: String(options.focusSceneId || "").trim(),
      projectFolder: projectInput.value || state.projectFolder || "",
      projectVideoEngine: normalizeProjectVideoEngine(state.projectVideoEngine),
      renderedSceneCount: state.segments.filter((segment) => String(segment.video_path || "").trim()).length,
      lineMappingLyrics: String(options.sourceLyrics || state.lyricMapper?.source_text || ""),
      imageMode: state.imageModelMode || "zimage",
      imageModeLabel: imageModeDisplayLabel(state.imageModelMode || "zimage"),
      videoPromptType: currentVideoMode(),
      miniMaxH3Mode: storyboardOpeningMiniMaxMode,
      miniMaxH3AudioMode: normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
        ? normalizeMiniMaxH3AudioMode(miniMaxH3SettingsForSegment(activeSegment()).audio_mode)
        : "",
      performanceMode: normalizeVideoType(state.videoType),
      videoType: normalizeVideoType(state.videoType),
      shortFilmPlanningMode: state.builderStoryboardDefaults?.short_film_planning_mode || "guided_film",
      facialPerformance: state.defaultFacialPerformance || "",
      facialPerformanceCustom: state.defaultFacialPerformanceCustom || "",
      globalConsistencyPhrase: state.builderStoryboardDefaults?.global_consistency_phrase || "",
      performanceStyle: state.builderStoryboardDefaults?.performance_style || "",
      videoStyle: state.builderStoryboardDefaults?.video_style || "",
      videoStyleCustom: state.builderStoryboardDefaults?.video_style_custom || "",
      temporalWorldEffect: state.builderStoryboardDefaults?.temporal_world_effect || "",
      temporalWorldEffectCustom: state.builderStoryboardDefaults?.temporal_world_effect_custom || "",
      temporalAllowBackgroundExtras: state.builderStoryboardDefaults?.temporal_allow_background_extras !== false,
      temporalBackgroundIntensity: state.builderStoryboardDefaults?.temporal_background_intensity ?? 8,
      temporalEnvironmentTimePassage: state.builderStoryboardDefaults?.temporal_environment_time_passage !== false,
      temporalProtectedCharacters: state.builderStoryboardDefaults?.temporal_protected_characters || "all_referenced",
      temporalProtectedCustom: state.builderStoryboardDefaults?.temporal_protected_custom || "",
      fxPreset: state.builderStoryboardDefaults?.fx_preset || "",
      fxCustomJson: state.builderStoryboardDefaults?.fx_custom_json || "",
      cameraMotionSpeed: state.builderStoryboardDefaults?.camera_motion_speed ?? 4,
      characterMotionSpeed: state.builderStoryboardDefaults?.character_motion_speed ?? 4,
      cutFrequency: state.builderStoryboardDefaults?.minimax_h3_cut_frequency ?? 0,
      motion_defaults: {
        camera_motion_speed: state.builderStoryboardDefaults?.camera_motion_speed ?? 4,
        character_motion_speed: state.builderStoryboardDefaults?.character_motion_speed ?? 4,
        minimax_h3_cut_frequency: state.builderStoryboardDefaults?.minimax_h3_cut_frequency ?? 0,
        camera_guidance: state.builderStoryboardDefaults?.camera_guidance || "",
        character_guidance: state.builderStoryboardDefaults?.character_guidance || "",
      },
      scenes: storyboardScenePayload(),
      referenceBuilder: storyboardReferenceBuilderWithIdLoraRefs(state.fluxReferenceBuilder),
      storyLayer: normalizeBuilderStoryLayer(state.builderStoryLayer),
      gemmaSettings: {
        ...storyboardRunnerSettings,
        model_file: storyboardSelectedModel,
        vision_model_file: storyboardSelectedModel,
        mmproj_file: storyboardSelectedMmproj,
        n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
        n_gpu_layers: normalizeGemmaGpuLayers(state.gemmaGpuLayers),
        n_threads: 8,
        unload_after: true,
      },
      onReferenceMappingsChanged: applyStoryboardReferenceMappings,
      onStoryLayerChanged: applyStoryboardPrompts,
      onPrepareStoryContext: typeof options.onPrepareStoryContext === "function" ? options.onPrepareStoryContext : null,
      onPromptsExported: applyStoryboardPrompts,
      onCreateVideoPrompt: createStoryboardVideoPromptViaBuilder,
      onBeforeCreateVideoPrompt: async (scene) => {
        const segment = findStoryboardSegment(scene);
        if (segment) await ensureAutoTimedSingerCuesBeforePrompt(segment);
      },
      onApplyIdLoraDialoguePlan: applyIdLoraDialoguePlanFromStoryboard,
      onApplyMiniMaxDialoguePlan: applyMiniMaxDialoguePlanFromStoryboard,
    }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
  }

  return { openStoryboardBuilderFromProject };
}
