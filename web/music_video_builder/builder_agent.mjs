import { postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeSelect, toast } from "./controls.mjs";
import { formatTime } from "./format.mjs";
import { loadContextTextQuiet } from "./project_files.mjs";
import { normalizeGemmaContextLimit } from "./prompt_text.mjs";
import { newSegment, newTimelineMarker, renumberGenericBaseSceneLabels, sortSegments } from "./segments.mjs";
import { hasLockedVideo, selectedSegmentVideoPath } from "./selection_preview.mjs";
import { markerContext, markerOverlapsRange, normalizeTimelineMarkers } from "./timeline_state.mjs";

export function createBuilderAgent({
  activeSegment, addFluxIngredient, allEditableSegments, audioInput, autoSaveSessionQuiet,
  builderStorySourcePath, chooseProjectAudioFile, createErnieImageForSegment, createFluxKleinImageForSegment,
  createKrea2TwoPassImageForSegment, createNBImageForSegment, createProgressWindow, createSceneVideo,
  createStoryScenesFromSource, createZImageForSegment, currentVideoMode, droppedSceneImageSource,
  editContextTextFile, ernieNotesInput, fluxGemmaModelSelect, fluxMmprojSelect, fluxNotes, gemmaModelSelect,
  gemmaRunnerLabel, generateFluxKleinPromptForSegment, generateI2VPromptForSegment,
  generateNBPromptForSegment, generateT2IPromptForSegment, i2vGemmaModelSelect, i2vMmprojSelect,
  i2vNotesInput, loadBuilderStorySource, mergedFluxImageIngredients, mmprojSelect, nbGemmaModelSelect,
  nbMmprojSelect, nbNotes, notesInput, projectAudioFileInput, projectContextPath, projectInput,
  promptRunnerActionName, pushHistory, render, renderFluxIngredientList, renderNBIngredientList,
  runConceptPromptCreator, runImageMemoryCleanupQuiet, runMotionNoteCreator, saveBuilderStorySource,
  sceneDisplayName, sceneSlotNumber, segmentImageSource, segmentIndexInfo, segmentTrack,
  selectedTimelineRangeInfo, setInspectorTab, shell, state, storyIdeaInput, subjectSceneInput,
  syncFluxKleinPanel, syncI2VMotionJsonFromSegments, syncInspector, syncPromptJsonFromSegments,
  syncVideoModePanel, t2iTextGemmaModelSelect, textGemmaRunnerPayload, themeStyleInput, videoModeDisplayLabel,
}) {
  function builderAgentSceneContext(segment) {
    if (!segment) return null;
    const info = segmentIndexInfo(segment);
    const mergedRefs = mergedFluxImageIngredients(segment);
    return {
      id: segment.id || "",
      index: info.index,
      label: sceneDisplayName(segment, info.index),
      start: Number(segment.start || 0),
      end: Number(segment.end || 0),
      lyric_text: String(segment.lyric_text || "").trim(),
      director_note: String(segment.timeline_note || "").trim(),
      scene_notes: String(segment.notes || "").trim(),
      image_prompt: String(segment.t2i_prompt || segment.flux_prompt || segment.nb_prompt || "").trim(),
      flux_notes: String(segment.flux_notes || "").trim(),
      flux_prompt: String(segment.flux_prompt || "").trim(),
      nano_banana_notes: String(segment.nb_notes || "").trim(),
      nano_banana_prompt: String(segment.nb_prompt || "").trim(),
      video_notes: String(segment.i2v_notes || "").trim(),
      video_prompt: String(segment.i2v_prompt || "").trim(),
      image_model_mode: state.imageModelMode || "zimage",
      video_model_mode: state.videoModelMode || "i2v",
      reference_image_count: mergedRefs.length,
      has_reference_images: mergedRefs.length > 0,
      timeline_markers: normalizeTimelineMarkers(state.timelineMarkers)
        .filter((marker) => markerOverlapsRange(marker, Number(segment.start || 0), Number(segment.end || 0)))
        .slice(0, 10)
        .map(markerContext),
    };
  }

  function builderAgentProjectStatus() {
    const scenes = allEditableSegments();
    const count = scenes.length;
    const withLyrics = scenes.filter((segment) => String(segment.lyric_text || "").trim()).length;
    const withNotes = scenes.filter((segment) => String(segment.notes || segment.flux_notes || segment.nb_notes || "").trim()).length;
    const withImagePrompts = scenes.filter((segment) => String(segment.t2i_prompt || segment.flux_prompt || segment.nb_prompt || "").trim()).length;
    const withImages = scenes.filter((segment) => Boolean(segmentImageSource(segment)?.path || segmentImageSource(segment)?.data)).length;
    const withVideoPrompts = scenes.filter((segment) => String(segment.i2v_prompt || "").trim()).length;
    const withVideos = scenes.filter((segment) => String(selectedSegmentVideoPath(segment) || "").trim()).length;
    const selectedRange = selectedTimelineRangeInfo();
    return {
      has_project_folder: Boolean(String(state.projectFolder || projectInput.value || "").trim()),
      has_audio: Boolean(String(audioInput.value || "").trim()),
      has_story_source: Boolean(String(state.builderStorySourcePreview || "").trim()),
      has_selected_timeline_range: Boolean(selectedRange),
      timeline_marker_count: normalizeTimelineMarkers(state.timelineMarkers).length,
      scene_count: count,
      scenes_with_lyrics: withLyrics,
      scenes_with_notes: withNotes,
      scenes_with_image_prompts: withImagePrompts,
      scenes_with_images: withImages,
      scenes_with_video_prompts: withVideoPrompts,
      scenes_with_videos: withVideos,
      image_model_mode: state.imageModelMode || "zimage",
      video_model_mode: state.videoModelMode || "i2v",
    };
  }

  function builderAgentContext(scope = "active_scene") {
    const segment = activeSegment();
    const storySourcePath = String(state.builderStorySourcePath || projectContextPath("AgentStorySource.txt") || "").trim();
    const timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
    const activeTimelineMarker = timelineMarkers.find((marker) => marker.id === state.activeTimelineMarkerId);
    const context = {
      project_status: builderAgentProjectStatus(),
      active_scene: builderAgentSceneContext(segment),
      scene_directory: allEditableSegments().map((item) => ({
        id: item.id || "",
        index: segmentIndexInfo(item).index,
        number: segmentIndexInfo(item).index + 1,
        label: sceneDisplayName(item, segmentIndexInfo(item).index),
      })),
      selected_timeline_range: selectedTimelineRangeInfo(),
      active_timeline_marker: activeTimelineMarker ? markerContext(activeTimelineMarker) : null,
      project_folder: state.projectFolder || "",
      image_model_mode: state.imageModelMode || "zimage",
      video_model_mode: state.videoModelMode || "i2v",
      story_source: {
        path: storySourcePath,
        has_source: Boolean(String(state.builderStorySourcePreview || "").trim()),
        preview: String(state.builderStorySourcePreview || "").slice(0, 500),
      },
      story_references: {
        image_count: Array.isArray(state.builderStoryReferenceImages) ? state.builderStoryReferenceImages.length : 0,
        notes: String(state.builderStoryReferenceNotes || "").slice(0, 1800),
      },
      context_prompts: {
        theme_style_path: state.themeStylePath || "",
        story_idea_path: state.storyIdeaPath || "",
        subject_scene_path: state.subjectScenePath || "",
      },
      timeline_markers: timelineMarkers
        .slice(0, 60)
        .map(markerContext),
      agent_reference_stash_count: Array.isArray(state.builderAgentReferenceImages) ? state.builderAgentReferenceImages.length : 0,
    };
    if (scope === "neighbors" && segment) {
      const index = state.segments.findIndex((item) => item.id === segment.id);
      context.neighbor_scenes = [state.segments[index - 1], state.segments[index + 1]]
        .filter(Boolean)
        .map((item) => builderAgentSceneContext(item));
    }
    if (scope === "project_brief") {
      context.scene_count = state.segments.length;
      context.project_brief = state.segments.slice(0, 80).map((item) => ({
        id: item.id || "",
        index: segmentIndexInfo(item).index,
        label: sceneDisplayName(item, segmentIndexInfo(item).index),
        lyric_text: String(item.lyric_text || "").trim(),
        director_note: String(item.timeline_note || "").trim(),
        scene_notes: String(item.notes || "").trim(),
        timeline_markers: normalizeTimelineMarkers(state.timelineMarkers)
          .filter((marker) => markerOverlapsRange(marker, Number(item.start || 0), Number(item.end || 0)))
          .slice(0, 8)
          .map(markerContext),
      }));
    }
    if (scope === "full_scene_plan") {
      context.scene_count = state.segments.length;
      context.full_scene_plan = state.segments.slice(0, 120).map((item) => ({
        id: item.id || "",
        index: segmentIndexInfo(item).index,
        number: segmentIndexInfo(item).index + 1,
        label: sceneDisplayName(item, segmentIndexInfo(item).index),
        lyric_text: String(item.lyric_text || "").trim(),
        director_note: String(item.timeline_note || "").trim(),
        scene_notes: String(item.notes || "").trim(),
        flux_notes: String(item.flux_notes || "").trim(),
        nb_notes: String(item.nb_notes || "").trim(),
        video_notes: String(item.i2v_notes || "").trim(),
        timeline_markers: normalizeTimelineMarkers(state.timelineMarkers)
          .filter((marker) => markerOverlapsRange(marker, Number(item.start || 0), Number(item.end || 0)))
          .slice(0, 8)
          .map(markerContext),
      }));
    }
    return context;
  }

  async function applyBuilderAgentActions(actions = []) {
    const applied = [];
    const skipped = [];
    const list = Array.isArray(actions) ? actions : [];
    if (!list.length) return { applied, skipped };
    const imageModeLabel = (mode = state.imageModelMode || "zimage") => {
      if (mode === "flux_klein") return "Flux/Klein";
      if (mode === "nano_banana") return "Nano B";
      if (mode === "ernie_image") return "Ernie";
      if (mode === "krea2_2pass") return "Krea 2";
      if (mode === "flow_gpt") return "Flow/GPT";
      if (mode === "z_enhance") return "Z Enhance";
      return "ZImage";
    };
    const normalizeImageMode = (mode) => {
      const value = String(mode || "").trim().toLowerCase();
      if (["zimage", "flux_klein", "nano_banana", "ernie_image", "krea2_2pass", "flow_gpt"].includes(value)) return value;
      if (value === "flux" || value === "klein") return "flux_klein";
      if (value === "nano" || value === "nanobanana" || value === "nano_b") return "nano_banana";
      if (value === "ernie") return "ernie_image";
      if (value === "krea" || value === "krea2" || value === "krea_2" || value === "krea2_2_pass" || value === "krea2 two pass") return "krea2_2pass";
      if (value === "flow" || value === "gpt" || value === "flowgpt" || value === "flow_gpt" || value === "browser") return "flow_gpt";
      return "";
    };
    const setAgentImageMode = (mode, push = true) => {
      const normalized = normalizeImageMode(mode);
      if (!normalized) return "";
      if (push) markChanging();
      state.imageModelMode = normalized;
      state.fluxKleinSettings.image_model_mode = normalized;
      state.fluxKleinSettings.enabled = normalized === "flux_klein";
      syncFluxKleinPanel();
      return normalized;
    };
    const normalizeVideoMode = (mode) => {
      const value = String(mode || "").trim().toLowerCase();
      if (value === "t2v" || value === "text_to_video" || value === "text-to-video") return "t2v";
      if (value === "i2v" || value === "image_to_video" || value === "image-to-video") return "i2v";
      if (value === "id_lora" || value === "id-lora" || value === "identity_i2v" || value === "identity-driven") return "id_lora";
      if (value === "rtv" || value === "reference_to_video" || value === "reference-to-video") return "rtv";
      if (value === "ingredients" || value === "ingredients_to_video" || value === "ingredients-to-video") return "ingredients";
      if (value === "import" || value === "import_custom_video" || value === "custom_video") return "import";
      return "";
    };
    const setAgentVideoMode = (mode, push = true) => {
      const normalized = normalizeVideoMode(mode);
      if (!normalized) return "";
      if (push) markChanging();
      state.videoModelMode = normalized;
      syncVideoModePanel();
      return normalized;
    };
    const ingredientKey = (item) => String(item?.path || item?.data || item?.name || "").trim();
    const ensureAgentRefsForScene = (segment, imageMode) => {
      if (!segment || !["flux_klein", "nano_banana"].includes(imageMode)) return 0;
      if (mergedFluxImageIngredients(segment).length) return 0;
      const refs = Array.isArray(state.builderAgentReferenceImages) ? state.builderAgentReferenceImages : [];
      if (!refs.length) return 0;
      markChanging();
      if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
      let added = 0;
      for (const ref of refs) {
        const key = ingredientKey(ref);
        if (!key || segment.flux_image_ingredients.some((item) => ingredientKey(item) === key)) continue;
        segment.flux_image_ingredients.push({
          path: ref.path || "",
          data: ref.data || "",
          name: ref.name || "agent_reference.png",
        });
        added += 1;
      }
      if (added) {
        renderFluxIngredientList(segment);
        renderNBIngredientList(segment);
        render();
      }
      return added;
    };
    const agentSceneNumberFromValue = (value) => {
      if (Number.isFinite(Number(value)) && Number(value) > 0) return Math.floor(Number(value));
      const text = String(value || "").trim();
      if (!text) return 0;
      const match = text.match(/\bscene\s*(\d+)(?:\b|\s|\.|:|-)/i);
      if (match) return Number(match[1]);
      const wordMatch = text.toLowerCase().match(/\bscene\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b/);
      if (wordMatch) {
        return {
          one: 1,
          two: 2,
          three: 3,
          four: 4,
          five: 5,
          six: 6,
          seven: 7,
          eight: 8,
          nine: 9,
          ten: 10,
          eleven: 11,
          twelve: 12,
        }[wordMatch[1]] || 0;
      }
      return 0;
    };
    const agentSceneNumberFromAction = (action) => {
      let number = agentSceneNumberFromValue(action?.scene_number);
      if (number) return number;
      const sceneFields = [
        "scene_id",
        "scene",
        "scene_label",
        "scene_name",
        "scene_title",
        "target_scene",
        "target_scene_label",
        "target",
        "label",
        "name",
        "title",
        "instruction",
        "summary",
        "reason",
      ];
      for (const key of sceneFields) {
        number = agentSceneNumberFromValue(action?.[key]);
        if (number) return number;
      }
      const sceneIndex = Number(action?.scene_index ?? action?.index);
      if (Number.isFinite(sceneIndex) && sceneIndex >= 0) return Math.floor(sceneIndex) + 1;
      try {
        number = agentSceneNumberFromValue(JSON.stringify(action || {}));
      } catch {
        number = 0;
      }
      return number || 0;
    };
    const findAgentScene = (action) => {
      const sceneId = String(action?.scene_id || "").trim();
      if (sceneId) {
        const byId = allEditableSegments().find((item) => item.id === sceneId);
        if (byId) return byId;
      }
      const number = agentSceneNumberFromAction(action);
      if (Number.isFinite(number) && number > 0) {
        // Scene numbers shown in the timeline are positional. Resolve that
        // visible base-scene slot before consulting labels, which may be stale
        // after a split/merge or may contain a custom title.
        const byIndex = state.segments[number - 1] || null;
        if (byIndex) return byIndex;
        const labelPattern = new RegExp(`(?:^|\\b|\\.)\\s*scene\\s*${number}(?:\\b|\\s|\\.|:|-)`, "i");
        const byLabel = allEditableSegments().find((item) => {
          const info = segmentIndexInfo(item);
          return labelPattern.test(String(item.label || "")) || labelPattern.test(sceneDisplayName(item, info.index));
        });
        if (byLabel) return byLabel;
        const byAnyIndex = allEditableSegments()[number - 1] || null;
        if (byAnyIndex) return byAnyIndex;
        return allEditableSegments().find((item) => segmentIndexInfo(item).index + 1 === number || sceneSlotNumber(item) === number) || null;
      }
      return null;
    };
    let historyPushed = false;
    const markChanging = () => {
      if (!historyPushed) {
        pushHistory();
        historyPushed = true;
      }
    };
    const mergeSceneValues = (items, key) => {
      const values = [];
      for (const item of items) {
        const value = String(item?.[key] || "").trim();
        if (value && !values.includes(value)) values.push(value);
      }
      return values.join("\n\n");
    };
    const contextPromptInfo = (contextType = "") => {
      const type = String(contextType || "").trim();
      if (type === "theme_style") return { type, label: "Theme/style", filename: "themestyle.txt", input: themeStyleInput, stateKey: "themeStylePath" };
      if (type === "story_idea") return { type, label: "Story idea", filename: "storyconcept.txt", input: storyIdeaInput, stateKey: "storyIdeaPath" };
      if (type === "subject_scene") return { type, label: "Subject/scene", filename: "subjectsandscenes.txt", input: subjectSceneInput, stateKey: "subjectScenePath" };
      return null;
    };
    const ensureContextPromptPath = (info) => {
      if (!info) return "";
      const existing = String(state[info.stateKey] || info.input.value || "").trim();
      if (existing) return existing;
      const path = projectContextPath(info.filename);
      if (path) {
        state[info.stateKey] = path;
        info.input.value = path;
      }
      return path;
    };
    const setContextPromptText = async (contextType, text, modeValue = "replace") => {
      const info = contextPromptInfo(contextType);
      if (!info) {
        skipped.push("unknown context prompt");
        return false;
      }
      const path = ensureContextPromptPath(info);
      if (!path) {
        skipped.push("create or load a project before updating context prompts");
        return false;
      }
      const modeName = String(modeValue || "replace").trim().toLowerCase();
      const nextText = String(text || "").trim();
      if (!nextText) {
        skipped.push(`${info.label} text missing`);
        return false;
      }
      let content = nextText;
      if (modeName === "append") {
        const current = await loadContextTextQuiet(path);
        content = [current, nextText].filter(Boolean).join("\n\n");
      }
      markChanging();
      const result = await postJson("/vrgdg/music_builder/save_text_file", { path, content });
      info.input.value = result.path || path;
      state[info.stateKey] = info.input.value;
      applied.push(`${info.label} context ${modeName === "append" ? "appended" : "updated"}`);
      return true;
    };
    const getContextPromptText = async (contextType = "all") => {
      const types = contextType === "all" || !contextType
        ? ["theme_style", "story_idea", "subject_scene"]
        : [contextType];
      const parts = [];
      for (const type of types) {
        const info = contextPromptInfo(type);
        if (!info) continue;
        const path = ensureContextPromptPath(info);
        const text = path ? await loadContextTextQuiet(path) : "";
        parts.push(`${info.label} context:\n${text || "[empty]"}`);
      }
      if (parts.length) applied.push(parts.join("\n\n"));
      else skipped.push("context prompt not found");
    };
    const normalizeDualVocalDirectorNotes = (replacement = "male and female") => {
      const nextText = String(replacement || "male and female").trim() || "male and female";
      const dualPattern = /\b(?:male(?:\s+(?:singer|vocalist|character))?\s+and\s+female(?:\s+(?:singer|vocalist|character))?|female(?:\s+(?:singer|vocalist|character))?\s+and\s+male(?:\s+(?:singer|vocalist|character))?|both\s+(?:characters|singers|vocals?|vocalists)|dual[-\s]?vocal|two\s+(?:characters|singers|vocalists))\b/i;
      let changed = 0;
      markChanging();
      for (const segment of state.segments) {
        const note = String(segment.timeline_note || "").trim();
        if (!note || !dualPattern.test(note)) continue;
        if (note !== nextText) {
          segment.timeline_note = nextText;
          changed += 1;
        }
      }
      if (changed) applied.push(`updated ${changed} dual-vocal director note${changed === 1 ? "" : "s"} to ${nextText}`);
      else applied.push(`dual-vocal director notes already use ${nextText}`);
      return changed;
    };
    const replaceDirectorNoteText = (findText = "", replaceText = "") => {
      const findValue = String(findText || "").trim();
      const replaceValue = String(replaceText || "").trim();
      if (!findValue || !replaceValue) {
        skipped.push("director note replacement text missing");
        return 0;
      }
      let changed = 0;
      markChanging();
      const escaped = findValue.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const pattern = new RegExp(escaped, "gi");
      for (const segment of state.segments) {
        const note = String(segment.timeline_note || "");
        if (!note || !pattern.test(note)) continue;
        pattern.lastIndex = 0;
        const nextNote = note.replace(pattern, replaceValue).trim();
        if (nextNote !== note) {
          segment.timeline_note = nextNote;
          changed += 1;
        }
      }
      if (changed) applied.push(`replaced director note text in ${changed} scene${changed === 1 ? "" : "s"}`);
      else applied.push(`no director notes contained ${findValue}`);
      return changed;
    };
    const syncScenesToTimelineMarkers = (action = {}, sourceLabel = "timeline note timing") => {
      const markers = normalizeTimelineMarkers(state.timelineMarkers)
        .filter((marker) => Number.isFinite(Number(marker.start)) && Number.isFinite(Number(marker.end)) && Number(marker.end) > Number(marker.start))
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      if (!markers.length) {
        skipped.push("no timeline notes to sync");
        return false;
      }
      const baseScenes = state.segments.filter((item) => segmentTrack(item) !== "overlay").sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      const createMissing = action?.create_missing !== false;
      let synced = 0;
      let created = 0;
      let locked = 0;
      let firstSyncedId = "";
      markChanging();
      for (let index = 0; index < markers.length; index += 1) {
        const marker = markers[index];
        let targetScene = baseScenes[index] || null;
        if (!targetScene && createMissing) {
          targetScene = newSegment(Number(marker.start || 0), Number(marker.end || marker.start || 0));
          targetScene.label = `Scene ${state.segments.length + created + 1}`;
          targetScene.source = "agent_timeline_marker";
          state.segments.push(targetScene);
          baseScenes.push(targetScene);
          created += 1;
        }
        if (!targetScene) {
          skipped.push(`timeline note ${index + 1}: no scene available`);
          continue;
        }
        if (hasLockedVideo(targetScene)) {
          locked += 1;
          continue;
        }
        const markerStart = Number(marker.start || 0);
        const markerStop = Number(marker.end || markerStart);
        targetScene.start = Number(markerStart.toFixed(3));
        targetScene.end = Number(Math.max(markerStart + 0.05, markerStop).toFixed(3));
        if (Number.isFinite(Number(targetScene.custom_audio_timeline_start))) targetScene.custom_audio_timeline_start = targetScene.start;
        const markerText = [marker.type, marker.label, marker.note]
          .map((value) => String(value || "").trim())
          .filter(Boolean)
          .join(": ");
        if (markerText) targetScene.timeline_note = markerText;
        if (!firstSyncedId) firstSyncedId = targetScene.id || "";
        synced += 1;
      }
      sortSegments(state.segments);
      renumberGenericBaseSceneLabels(state.segments);
      state.duration = Math.max(Number(state.duration || 0), ...state.segments.map((item) => Number(item.end || 0)), ...markers.map((marker) => Number(marker.end || marker.start || 0)));
      if (synced) {
        state.activeId = firstSyncedId || state.activeId;
        applied.push(`synced ${synced} scene${synced === 1 ? "" : "s"} to ${sourceLabel}${created ? `, created ${created} missing scene${created === 1 ? "" : "s"}` : ""}`);
      }
      if (locked) skipped.push(`${locked} scene${locked === 1 ? "" : "s"} skipped because generated video timing is locked`);
      return synced > 0;
    };
    const splitSceneIntoSubscenes = (targetScene, action = {}) => {
      if (!targetScene) {
        skipped.push("scene not found");
        return false;
      }
      if (hasLockedVideo(targetScene)) {
        skipped.push(`${sceneDisplayName(targetScene, segmentIndexInfo(targetScene).index)} has generated video, so timing is locked`);
        return false;
      }
      const count = Math.max(2, Math.min(24, Math.floor(Number(action?.scene_count || action?.count || 0))));
      if (!count) {
        skipped.push("scene count missing");
        return false;
      }
      const start = Number(targetScene.start || 0);
      const end = Number(targetScene.end || start);
      const total = Math.max(0, end - start);
      if (total <= 0.1) {
        skipped.push(`${sceneDisplayName(targetScene, segmentIndexInfo(targetScene).index)} is too short to split`);
        return false;
      }
      const track = segmentTrack(targetScene);
      const listToUpdate = track === "overlay" ? state.overlaySegments : state.segments;
      const originalIndex = listToUpdate.findIndex((item) => item.id === targetScene.id);
      if (originalIndex < 0) {
        skipped.push("scene not found");
        return false;
      }
      markChanging();
      const labelPrefix = String(action?.label_prefix || targetScene.label || sceneDisplayName(targetScene, segmentIndexInfo(targetScene).index)).trim() || "Scene";
      const sceneNotes = Array.isArray(action?.scene_notes) ? action.scene_notes : [];
      const directorNotes = Array.isArray(action?.director_notes) ? action.director_notes : [];
      const originalTimelineNote = String(targetScene.timeline_note || "").trim();
      const originalSceneNotes = String(targetScene.notes || "").trim();
      const originalFluxNotes = String(targetScene.flux_notes || "").trim();
      const originalNBNotes = String(targetScene.nb_notes || "").trim();
      const originalVideoNotes = String(targetScene.i2v_notes || "").trim();
      const duration = total / count;
      const created = [];
      for (let index = 0; index < count; index += 1) {
        const itemStart = Number((start + duration * index).toFixed(3));
        const itemEnd = Number((index === count - 1 ? end : start + duration * (index + 1)).toFixed(3));
        const segment = index === 0 ? targetScene : newSegment(itemStart, itemEnd);
        segment.start = itemStart;
        segment.end = itemEnd;
        segment.track = track;
        segment.label = `${labelPrefix}.${index + 1}`;
        segment.source = "agent_scene_split";
        segment.timeline_note = String(directorNotes[index] || originalTimelineNote).trim();
        segment.notes = String(sceneNotes[index] || originalSceneNotes).trim();
        segment.flux_notes = originalFluxNotes;
        segment.nb_notes = originalNBNotes;
        segment.i2v_notes = originalVideoNotes;
        if (index > 0) created.push(segment);
      }
      listToUpdate.splice(originalIndex + 1, 0, ...created);
      sortSegments(listToUpdate);
      if (track !== "overlay") renumberGenericBaseSceneLabels(state.segments);
      state.activeId = targetScene.id;
      applied.push(`split ${labelPrefix} into ${count} sub-scene${count === 1 ? "" : "s"}`);
      return true;
    };
    const mergeScenes = (action = {}) => {
      let sceneNumbers = Array.isArray(action?.scene_numbers) ? action.scene_numbers.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0) : [];
      const startNumber = Number(action?.start_scene_number);
      const endNumber = Number(action?.end_scene_number);
      if (!sceneNumbers.length && Number.isFinite(startNumber) && Number.isFinite(endNumber) && startNumber > 0 && endNumber >= startNumber) {
        sceneNumbers = Array.from({ length: endNumber - startNumber + 1 }, (_, index) => startNumber + index);
      }
      if (!sceneNumbers.length && Number(action?.scene_number) > 0) sceneNumbers = [Number(action.scene_number)];
      sceneNumbers = Array.from(new Set(sceneNumbers)).sort((a, b) => a - b);
      const scenes = sceneNumbers.map((number) => findAgentScene({ scene_number: number })).filter(Boolean);
      const uniqueScenes = Array.from(new Map(scenes.map((segment) => [segment.id, segment])).values())
        .filter((segment) => segmentTrack(segment) !== "overlay")
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      if (uniqueScenes.length < 2) {
        skipped.push("merge needs at least two base scenes");
        return false;
      }
      const locked = uniqueScenes.filter((segment) => hasLockedVideo(segment));
      if (locked.length) {
        skipped.push(`${locked.length} scene${locked.length === 1 ? "" : "s"} skipped because generated video timing is locked`);
        return false;
      }
      markChanging();
      const target = uniqueScenes[0];
      const start = Math.min(...uniqueScenes.map((segment) => Number(segment.start || 0)));
      const end = Math.max(...uniqueScenes.map((segment) => Number(segment.end || 0)));
      target.start = Number(start.toFixed(3));
      target.end = Number(Math.max(start + 0.05, end).toFixed(3));
      target.label = String(action?.label || target.label || `Scene ${sceneNumbers[0] || segmentIndexInfo(target).index + 1}`).trim();
      target.timeline_note = mergeSceneValues(uniqueScenes, "timeline_note");
      target.notes = mergeSceneValues(uniqueScenes, "notes");
      target.flux_notes = mergeSceneValues(uniqueScenes, "flux_notes");
      target.nb_notes = mergeSceneValues(uniqueScenes, "nb_notes");
      target.i2v_notes = mergeSceneValues(uniqueScenes, "i2v_notes");
      const removeIds = new Set(uniqueScenes.slice(1).map((segment) => segment.id));
      state.segments = state.segments.filter((segment) => !removeIds.has(segment.id));
      sortSegments(state.segments);
      renumberGenericBaseSceneLabels(state.segments);
      state.activeId = target.id;
      applied.push(`merged scenes ${sceneNumbers.join(", ")} into ${target.label || sceneDisplayName(target, segmentIndexInfo(target).index)}`);
      return true;
    };
    for (const action of list) {
      const type = String(action?.type || "").trim();
      const sceneId = String(action?.scene_id || "").trim();
      const actionTextForType = () => {
        if (String(action?.text || "").trim()) return String(action.text).trim();
        const aliases = {
          set_scene_notes: ["scene_notes", "notes", "concept", "concept_prompt", "prompt"],
          set_flux_notes: ["flux_notes", "notes", "prompt", "concept", "concept_prompt"],
          set_nb_notes: ["nb_notes", "nano_banana_notes", "notes", "prompt", "concept", "concept_prompt"],
          set_video_notes: ["video_notes", "i2v_notes", "motion_notes", "notes", "prompt", "concept", "concept_prompt"],
        }[type] || [];
        for (const key of [...aliases, "note", "value", "content", "description", "new_text", "updated_text"]) {
          const value = String(action?.[key] || "").trim();
          if (value) return value;
        }
        return "";
      };
      const text = actionTextForType();
      if (type === "sync_existing_scenes_to_timeline_markers") {
        syncScenesToTimelineMarkers(action, "timeline note timing");
        continue;
      }
      if (type === "renumber_scene_labels") {
        markChanging();
        const changed = renumberGenericBaseSceneLabels(state.segments);
        if (changed) applied.push("renumbered generic scene labels");
        else applied.push("scene labels already sequential");
        continue;
      }
      if (type === "normalize_dual_vocal_director_notes") {
        normalizeDualVocalDirectorNotes(action?.replacement || "male and female");
        continue;
      }
      if (type === "replace_director_note_text") {
        replaceDirectorNoteText(action?.find || action?.from || "", action?.replace || action?.to || "");
        continue;
      }
      if (type === "merge_scenes") {
        mergeScenes(action);
        continue;
      }
      if (type === "select_scene") {
        const nextScene = findAgentScene(action);
        if (nextScene) {
          state.activeId = nextScene.id;
          syncInspector();
          render();
          applied.push(`selected ${sceneDisplayName(nextScene, segmentIndexInfo(nextScene).index)}`);
        } else {
          skipped.push("scene not found");
        }
        continue;
      }
      if (type === "get_scene_lyrics") {
        const lyricScene = findAgentScene(action);
        if (lyricScene) {
          const lyricText = String(lyricScene.lyric_text || "").trim() || "[no lyric text for this scene]";
          applied.push(`${sceneDisplayName(lyricScene, segmentIndexInfo(lyricScene).index)} lyrics:\n${lyricText}`);
        } else {
          skipped.push("scene lyrics not found");
        }
        continue;
      }
      if (type === "get_scene_context") {
        const contextScene = findAgentScene(action);
        if (contextScene) {
          applied.push(`${sceneDisplayName(contextScene, segmentIndexInfo(contextScene).index)} context:\n${JSON.stringify(builderAgentSceneContext(contextScene), null, 2)}`);
        } else {
          skipped.push("scene context not found");
        }
        continue;
      }
      if (type === "get_story_source") {
        const storySource = await loadBuilderStorySource();
        if (storySource.trim()) {
          applied.push(`Story Source:\n${storySource.slice(0, 12000)}`);
        } else {
          skipped.push("Story Source is empty");
        }
        continue;
      }
      if (type === "get_context_prompts") {
        await getContextPromptText(String(action?.context_type || "all").trim() || "all");
        continue;
      }
      if (type === "set_context_prompt") {
        await setContextPromptText(action?.context_type, action?.text || "", action?.mode || "replace");
        continue;
      }
      if (type === "get_selected_timeline_range") {
        const range = selectedTimelineRangeInfo();
        if (range) applied.push(`Selected timeline range:\n${JSON.stringify(range, null, 2)}`);
        else skipped.push("no selected timeline range");
        continue;
      }
      if (type === "assign_selected_range_note") {
        const range = selectedTimelineRangeInfo();
        if (!range) {
          skipped.push("no selected timeline range");
          continue;
        }
        const noteText = String(action?.note || action?.text || "").trim();
        const labelText = String(action?.label || action?.type_label || "").trim() || "Timeline note";
        const markerType = String(action?.marker_type || action?.category || "note").trim() || "note";
        if (!noteText && !labelText) {
          skipped.push("timeline note text missing");
          continue;
        }
        markChanging();
        const marker = newTimelineMarker(range.start, range.end);
        marker.type = markerType;
        marker.label = labelText;
        marker.note = noteText;
        state.timelineMarkers.push(marker);
        state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
        state.activeTimelineMarkerId = marker.id;
        applied.push(`timeline note: ${labelText} (${formatTime(range.start)} - ${formatTime(range.end)})`);
        continue;
      }
      if (type === "create_scene_from_selected_range") {
        const range = selectedTimelineRangeInfo();
        if (!range) {
          skipped.push("no selected timeline range");
          continue;
        }
        markChanging();
        const segment = newSegment(range.start, range.end);
        segment.label = String(action?.label || `Scene ${state.segments.length + 1}`).trim() || `Scene ${state.segments.length + 1}`;
        segment.timeline_note = String(action?.director_note || action?.note || action?.text || "").trim();
        segment.notes = String(action?.scene_notes || "").trim();
        segment.flux_notes = String(action?.flux_notes || "").trim();
        segment.nb_notes = String(action?.nb_notes || "").trim();
        segment.i2v_notes = String(action?.video_notes || "").trim();
        segment.source = "agent_range";
        state.segments.push(segment);
        sortSegments(state.segments);
        state.activeId = segment.id;
        applied.push(`created scene from selected range (${formatTime(range.start)} - ${formatTime(range.end)})`);
        continue;
      }
      if (type === "set_active_scene_to_selected_range") {
        const range = selectedTimelineRangeInfo();
        const targetScene = findAgentScene(action) || activeSegment();
        if (!range) {
          skipped.push("no selected timeline range");
          continue;
        }
        if (!targetScene) {
          skipped.push("no scene selected");
          continue;
        }
        markChanging();
        targetScene.start = range.start;
        targetScene.end = range.end;
        if (Number.isFinite(Number(targetScene.custom_audio_timeline_start))) targetScene.custom_audio_timeline_start = range.start;
        sortSegments(segmentTrack(targetScene) === "overlay" ? state.overlaySegments : state.segments);
        state.activeId = targetScene.id;
        applied.push(`${sceneDisplayName(targetScene, segmentIndexInfo(targetScene).index)} timing set to selected range`);
        continue;
      }
      if (type === "split_scene_into_subscenes") {
        splitSceneIntoSubscenes(findAgentScene(action), action);
        continue;
      }
      if (type === "split_selected_range_into_scenes") {
        const range = selectedTimelineRangeInfo();
        const count = Math.max(1, Math.min(24, Math.floor(Number(action?.scene_count || action?.count || 0))));
        if (!range) {
          const targetScene = findAgentScene(action);
          if (targetScene && count > 1) {
            splitSceneIntoSubscenes(targetScene, action);
          } else if (!count && normalizeTimelineMarkers(state.timelineMarkers).some((marker) => Number(marker.end || 0) > Number(marker.start || 0))) {
            syncScenesToTimelineMarkers({ ...action, create_missing: action?.create_missing !== false }, "timeline note timing");
          } else {
            skipped.push("no selected timeline range or target scene");
          }
          continue;
        }
        if (!count) {
          skipped.push("scene count missing");
          continue;
        }
        markChanging();
        const labelPrefix = String(action?.label_prefix || "Scene").trim() || "Scene";
        const sceneNotes = Array.isArray(action?.scene_notes) ? action.scene_notes : [];
        const directorNotes = Array.isArray(action?.director_notes) ? action.director_notes : [];
        const duration = range.duration / count;
        const created = [];
        for (let index = 0; index < count; index += 1) {
          const start = range.start + duration * index;
          const end = index === count - 1 ? range.end : range.start + duration * (index + 1);
          const segment = newSegment(Number(start.toFixed(3)), Number(end.toFixed(3)));
          segment.label = `${labelPrefix} ${state.segments.length + created.length + 1}`;
          segment.source = "agent_range_split";
          segment.notes = String(sceneNotes[index] || "").trim();
          segment.timeline_note = String(directorNotes[index] || "").trim();
          created.push(segment);
        }
        state.segments.push(...created);
        sortSegments(state.segments);
        state.activeId = created[0]?.id || state.activeId;
        applied.push(`split selected range into ${created.length} scene${created.length === 1 ? "" : "s"}`);
        continue;
      }
      if (type === "save_story_source") {
        try {
          const savedPath = await saveBuilderStorySource(text);
          applied.push(`Story Source saved: ${savedPath}`);
        } catch (error) {
          skipped.push(`Story Source save failed (${String(error?.message || error)})`);
        }
        continue;
      }
      if (type === "create_story_scenes_from_source") {
        try {
          const count = await createStoryScenesFromSource(action?.scene_count || 12);
          applied.push(`created ${count} Story Builder scene${count === 1 ? "" : "s"}`);
        } catch (error) {
          skipped.push(`Story Builder scene creation failed (${String(error?.message || error)})`);
        }
        continue;
      }
      if (type === "create_concept_prompts") {
        try {
          const result = await runConceptPromptCreator({
            sourceMode: action?.source_mode || "all",
            scope: action?.scope || "all",
            batchSize: action?.batch_size || 5,
            useStory: action?.use_story !== false,
            useTheme: action?.use_theme !== false,
          });
          if (result?.updated) applied.push(`created ${result.updated} concept prompt${result.updated === 1 ? "" : "s"}`);
          else skipped.push(result?.error || "concept prompt creation did not update any scenes");
        } catch (error) {
          skipped.push(`concept prompt creation failed (${String(error?.message || error)})`);
        }
        continue;
      }
      if (type === "create_motion_notes") {
        try {
          const result = await runMotionNoteCreator({
            sourceMode: action?.source_mode || "concept",
            scope: action?.scope || "all",
            batchSize: action?.batch_size || 5,
            useStory: action?.use_story !== false,
            useTheme: action?.use_theme !== false,
          });
          if (result?.updated) applied.push(`created ${result.updated} motion note${result.updated === 1 ? "" : "s"}`);
          else skipped.push(result?.error || "motion note creation did not update any scenes");
        } catch (error) {
          skipped.push(`motion note creation failed (${String(error?.message || error)})`);
        }
        continue;
      }
      if (type === "set_image_model_mode") {
        const nextMode = setAgentImageMode(action?.image_mode || text);
        if (nextMode) applied.push(`image mode: ${imageModeLabel(nextMode)}`);
        else skipped.push("unknown image mode");
        continue;
      }
      if (type === "set_video_model_mode") {
        const nextMode = setAgentVideoMode(action?.video_mode || text);
        if (nextMode) applied.push(`video mode: ${videoModeDisplayLabel(nextMode, true)}`);
        else skipped.push("unknown video mode");
        continue;
      }
      const setActionTypes = new Set(["set_scene_notes", "set_flux_notes", "set_nb_notes", "set_video_notes", "set_scene_plan"]);
      const hasSceneTarget = Boolean(
        String(action?.scene_id || "").trim()
        || agentSceneNumberFromAction(action)
      );
      const segment = findAgentScene(action)
        || state.segments.find((item) => item.id === sceneId)
        || (setActionTypes.has(type) && !hasSceneTarget ? activeSegment() : null)
        || (setActionTypes.has(type) && agentSceneNumberFromAction(action) === 1 ? state.segments[0] || allEditableSegments()[0] || null : null);
      const needsText = type.startsWith("set_") && type !== "set_scene_plan";
      if (!type) {
        skipped.push("unknown action");
        continue;
      }
      if (!segment) {
        skipped.push(`${type}: scene not found`);
        continue;
      }
      if (needsText && !text) {
        skipped.push(`${type}: text missing`);
        continue;
      }
      const label = sceneDisplayName(segment, segmentIndexInfo(segment).index);
      if (type === "set_scene_notes") {
        markChanging();
        segment.notes = text;
        if (segment.id === activeSegment()?.id) {
          notesInput.value = text;
          ernieNotesInput.value = text;
        }
        applied.push(`${label}: scene notes`);
      } else if (type === "set_flux_notes") {
        markChanging();
        segment.flux_notes = text;
        if (segment.id === activeSegment()?.id) fluxNotes.value = text;
        applied.push(`${label}: Flux notes`);
      } else if (type === "set_nb_notes") {
        markChanging();
        segment.nb_notes = text;
        if (segment.id === activeSegment()?.id) nbNotes.value = text;
        applied.push(`${label}: Nano B notes`);
      } else if (type === "set_video_notes") {
        markChanging();
        segment.i2v_notes = text;
        if (segment.id === activeSegment()?.id) i2vNotesInput.value = text;
        applied.push(`${label}: video notes`);
      } else if (type === "set_scene_plan") {
        const updates = [];
        const directorNote = String(action?.director_note || "").trim();
        const sceneNotes = String(action?.scene_notes || action?.text || "").trim();
        const nextFluxNotes = String(action?.flux_notes || "").trim();
        const nextNBNotes = String(action?.nb_notes || "").trim();
        const nextVideoNotes = String(action?.video_notes || "").trim();
        markChanging();
        if (directorNote) {
          segment.timeline_note = directorNote;
          updates.push("director note");
        }
        if (sceneNotes) {
          segment.notes = sceneNotes;
          updates.push("scene notes");
        }
        if (nextFluxNotes) {
          segment.flux_notes = nextFluxNotes;
          updates.push("Flux notes");
        }
        if (nextNBNotes) {
          segment.nb_notes = nextNBNotes;
          updates.push("Nano B notes");
        }
        if (nextVideoNotes) {
          segment.i2v_notes = nextVideoNotes;
          updates.push("video notes");
        }
        if (segment.id === activeSegment()?.id) {
          notesInput.value = segment.notes || "";
          ernieNotesInput.value = segment.notes || "";
          fluxNotes.value = segment.flux_notes || "";
          nbNotes.value = segment.nb_notes || "";
          i2vNotesInput.value = segment.i2v_notes || "";
        }
        applied.push(`${label}: scene plan${updates.length ? ` (${updates.join(", ")})` : ""}`);
      } else if (type === "request_reference_images") {
        const requestedMode = normalizeImageMode(action?.image_mode) || "nano_banana";
        setAgentImageMode(requestedMode);
        state.activeId = segment.id;
        syncInspector();
        setInspectorTab("image");
        const modeName = imageModeLabel(requestedMode);
        applied.push(`${label}: ready for optional ${modeName} reference images`);
        toast(`${modeName} reference images are optional. Drop/upload them if you want reference guidance, or ask the agent to continue text-only.`);
      } else if (type === "generate_image_prompt_for_current_mode") {
        const previousActiveId = state.activeId;
        const requestedMode = normalizeImageMode(action?.image_mode);
        if (requestedMode) setAgentImageMode(requestedMode);
        const imageMode = state.imageModelMode || "zimage";
        const modeLabel = imageModeLabel(imageMode);
        const copiedRefs = ensureAgentRefsForScene(segment, imageMode);
        state.activeId = segment.id;
        syncInspector();
        const progress = createProgressWindow(`Builder Agent image prompt`);
        try {
          progress.set(`${label}: generating ${modeLabel} prompt with current settings...`, 20);
          if (imageMode === "flux_klein") {
            await generateFluxKleinPromptForSegment(segment, progress, 35, "Builder Agent Flux/Klein", { unloadAfter: true });
          } else if (imageMode === "nano_banana" || imageMode === "flow_gpt") {
            await generateNBPromptForSegment(segment, progress, 35, imageMode === "flow_gpt" ? "Builder Agent Flow/GPT" : "Builder Agent NanoBanana", { unloadAfter: true });
          } else {
            await generateT2IPromptForSegment(segment, progress, 35, "Builder Agent T2I", { unloadAfter: true });
          }
          progress.set(`${label}: ${modeLabel} prompt ready.`, 100);
          progress.close(900);
          applied.push(`${label}: generated ${modeLabel} prompt${copiedRefs ? `, attached ${copiedRefs} recent Agent ref${copiedRefs === 1 ? "" : "s"}` : ""}`);
        } catch (error) {
          progress?.set(`Error:\n${String(error?.message || error)}`, 100);
          skipped.push(`${label}: image prompt failed (${String(error?.message || error)})`);
        } finally {
          state.activeId = previousActiveId || state.activeId;
          syncInspector();
        }
      } else if (type === "run_image_for_current_mode") {
        const previousActiveId = state.activeId;
        const requestedMode = normalizeImageMode(action?.image_mode);
        if (requestedMode) setAgentImageMode(requestedMode);
        const imageMode = state.imageModelMode || "zimage";
        const modeLabel = imageModeLabel(imageMode);
        const copiedRefs = ensureAgentRefsForScene(segment, imageMode);
        state.activeId = segment.id;
        syncInspector();
        const progress = createProgressWindow(`Builder Agent ${modeLabel} image`);
        try {
          progress.set(`${label}: running ${modeLabel} image workflow...`, 12);
          await autoSaveSessionQuiet(`Builder Agent ${modeLabel} image`);
          if (imageMode === "flux_klein") {
            await createFluxKleinImageForSegment(segment, progress, 18, 72, "Builder Agent Flux/Klein image");
          } else if (imageMode === "nano_banana") {
            await createNBImageForSegment(segment, progress, 18, 72, "Builder Agent NanoBanana image");
          } else if (imageMode === "ernie_image") {
            await createErnieImageForSegment(segment, progress, 18, 72, "Builder Agent Ernie image");
          } else if (imageMode === "krea2_2pass") {
            await createKrea2TwoPassImageForSegment(segment, progress, 18, 72, "Builder Agent Krea 2 image");
          } else {
            await createZImageForSegment(segment, progress, 18, 72, "Builder Agent ZImage");
            await runImageMemoryCleanupQuiet(progress, "Builder Agent ZImage", 94);
          }
          await autoSaveSessionQuiet(`Builder Agent ${modeLabel} image complete`);
          progress.set(`${label}: ${modeLabel} image ready.`, 100);
          progress.close(900);
          applied.push(`${label}: ran ${modeLabel} image${copiedRefs ? `, attached ${copiedRefs} recent Agent ref${copiedRefs === 1 ? "" : "s"}` : ""}`);
        } catch (error) {
          progress?.set(`Error:\n${String(error?.message || error)}`, 100);
          skipped.push(`${label}: ${modeLabel} image failed (${String(error?.message || error)})`);
        } finally {
          state.activeId = previousActiveId || state.activeId;
          syncInspector();
        }
      } else if (type === "generate_video_prompt_for_current_mode") {
        const previousActiveId = state.activeId;
        const requestedVideoMode = normalizeVideoMode(action?.video_mode);
        if (requestedVideoMode) setAgentVideoMode(requestedVideoMode);
        state.activeId = segment.id;
        syncInspector();
        const progress = createProgressWindow(`Builder Agent video prompt`);
        try {
          const videoLabel = videoModeDisplayLabel(currentVideoMode(), true);
          progress.set(`${label}: generating ${videoLabel} prompt with current video settings...`, 20);
          await generateI2VPromptForSegment(segment, progress, 35, `Builder Agent ${videoLabel}`, { unloadAfter: true });
          progress.set(`${label}: ${videoLabel} prompt ready.`, 100);
          progress.close(900);
          applied.push(`${label}: generated ${videoLabel} prompt`);
        } catch (error) {
          progress?.set(`Error:\n${String(error?.message || error)}`, 100);
          skipped.push(`${label}: video prompt failed (${String(error?.message || error)})`);
        } finally {
          state.activeId = previousActiveId || state.activeId;
          syncInspector();
        }
      } else if (type === "run_video_for_current_mode") {
        const previousActiveId = state.activeId;
        const requestedVideoMode = normalizeVideoMode(action?.video_mode);
        if (requestedVideoMode) setAgentVideoMode(requestedVideoMode);
        const videoLabel = videoModeDisplayLabel(currentVideoMode(), true);
        state.activeId = segment.id;
        syncInspector();
        try {
          await createSceneVideo();
          applied.push(`${label}: ran ${videoLabel} video`);
        } catch (error) {
          skipped.push(`${label}: ${videoLabel} video failed (${String(error?.message || error)})`);
        } finally {
          state.activeId = previousActiveId || state.activeId;
          syncInspector();
        }
      } else {
        skipped.push(`${label}: ${type}`);
      }
    }
    if (applied.length) {
      syncInspector();
      render();
      await syncPromptJsonFromSegments("Builder Agent auto apply").catch(() => null);
      await syncI2VMotionJsonFromSegments("Builder Agent auto apply").catch(() => null);
      autoSaveSessionQuiet("Builder Agent auto apply").catch(() => null);
    }
    return { applied, skipped };
  }

  function openBuilderAgentModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:transparent;display:flex;align-items:center;justify-content:flex-end;pointer-events:none;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 28px));height:calc(100vh - 48px);margin:34px 14px 14px;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);display:flex;flex-direction:column;overflow:hidden;pointer-events:auto;min-height:360px;";
    const agentTopbar = document.createElement("div");
    agentTopbar.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;min-height:28px;box-sizing:border-box;cursor:move;";
    const heading = document.createElement("div");
    heading.style.cssText = "min-width:0;flex:1 1 auto;";
    heading.innerHTML = `<div style="font-size:13px;font-weight:900;color:#cffafe;line-height:1.1;">Builder Agent</div>`;
    const headerActions = document.createElement("div");
    headerActions.style.cssText = "display:flex;align-items:center;gap:8px;flex:0 0 auto;white-space:nowrap;";
    const hint = makeButton("Hint");
    const dock = makeButton("Dock");
    const popout = makeButton("Pop Out");
    const minimize = makeButton("Min");
    minimize.title = "Minimize Builder Agent without closing the chat.";
    const close = makeButton("Close");
    dock.title = "Dock Builder Agent back to the right side of the editor.";
    popout.title = "Move Builder Agent to a separate browser window for another monitor.";
    headerActions.append(hint, dock, popout, minimize, close);
    agentTopbar.append(heading, headerActions);
    const restore = document.createElement("button");
    restore.type = "button";
    restore.textContent = "Builder Agent";
    restore.title = "Restore Builder Agent";
    restore.style.cssText = "position:fixed;right:18px;bottom:18px;z-index:100006;display:none;pointer-events:auto;border:1px solid #155e75;border-radius:8px;background:#083344;color:#cffafe;padding:10px 13px;font-size:12px;font-weight:900;box-shadow:0 14px 40px rgba(0,0,0,.5);cursor:pointer;";

    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:8px 14px 10px;border-bottom:1px solid #1f2937;background:#0f172a;flex:0 0 auto;box-sizing:border-box;";
    const controlGrid = document.createElement("div");
    controlGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr) auto;gap:8px;align-items:end;min-width:0;";
    const scope = makeSelect(["active_scene", "neighbors", "project_brief", "full_scene_plan"], "active_scene");
    const scopeLabels = {
      active_scene: "Active scene only",
      neighbors: "Active scene + neighbors",
      project_brief: "Project brief",
      full_scene_plan: "Full scene plan",
    };
    for (const option of scope.options) option.textContent = scopeLabels[option.value] || option.value;
    const mode = makeSelect(["manual", "auto"], state.builderAgentAutoApply ? "auto" : "manual");
    mode.options[0].textContent = "Manual: suggest only";
    mode.options[1].textContent = "Auto: update fields";
    const purpose = makeSelect(["walkthrough", "scene_work", "story_builder", "troubleshoot"], state.builderAgentPurpose || "scene_work");
    purpose.options[0].textContent = "Walkthrough";
    purpose.options[1].textContent = "Scene work";
    purpose.options[2].textContent = "Story Builder";
    purpose.options[3].textContent = "Troubleshoot";
    if (purpose.value === "story_builder" && scope.value === "active_scene") {
      scope.value = "full_scene_plan";
    }
    const clear = makeButton("Clear Chat");
    controlGrid.append(makeField(`Context sent to ${promptRunnerActionName()}`, scope), makeField("Agent mode", mode), makeField("Purpose", purpose), clear);
    controls.append(agentTopbar, controlGrid);
    purpose.addEventListener("change", () => {
      state.builderAgentPurpose = purpose.value || "scene_work";
      if (purpose.value === "story_builder" && scope.value === "active_scene") {
        scope.value = "full_scene_plan";
      }
      syncAgentStoryBuilderTools();
      renderMessages();
    });

    function openAgentHintPopup() {
      const hintBackdrop = document.createElement("div");
      hintBackdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center;padding:18px;";
      const panel = document.createElement("div");
      panel.style.cssText = "width:min(760px,calc(100vw - 36px));max-height:calc(100vh - 36px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#e5e7eb;box-shadow:0 18px 60px rgba(0,0,0,.58);";
      const top = document.createElement("div");
      top.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:13px 14px;border-bottom:1px solid #1f2937;";
      const title = document.createElement("div");
      title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;";
      title.textContent = "Builder Agent Hints";
      const hintClose = makeButton("Close");
      top.append(title, hintClose);
      const body = document.createElement("div");
      body.style.cssText = "padding:14px;display:grid;gap:12px;font-size:12px;line-height:1.5;color:#cbd5e1;";
      const sections = [
        {
          title: `Context sent to ${promptRunnerActionName()}`,
          lines: [
            "Active scene only: sends the selected scene details. Best when you want focused edits.",
            "Active scene + neighbors: also sends nearby scenes. Best for continuity between scenes.",
            "Project brief: sends a smaller overall project summary. Best for broad direction without stuffing the chat.",
            "Full scene plan: sends lyrics and planning fields for up to 120 scenes. Best when assigning characters across a whole video.",
          ],
        },
        {
          title: "Agent mode",
          lines: [
            "Manual: suggest only. The agent can explain and draft ideas, but it should not apply field changes.",
            "Auto: update fields. The agent can update notes/prompts, select scenes, switch modes, and run supported actions.",
          ],
        },
        {
          title: "Purpose",
          lines: [
            "Walkthrough: helps a new user step through the workflow and asks what they are making first.",
            "Scene work: helps plan scenes, update notes, generate prompts, and keep creative continuity.",
            "Story Builder: plans multi-character projects scene by scene, assigning characters, concepts, and motion notes.",
            "Troubleshoot: focuses on setup problems, missing refs, wrong mode, failed prompts, or blocked runs.",
          ],
        },
        {
          title: "Scene Timing",
          lines: [
            "Try: split Scene 12 into 3.",
            "Try: merge Scenes 2, 3, and 4 into Scene 2.",
            "Try: sync existing scenes to the timeline notes.",
            "Try: renumber scene labels.",
            "The agent can change scene timing, split one scene, merge multiple scenes, and repair generic scene numbers.",
          ],
        },
        {
          title: "Director Notes",
          lines: [
            "Try: set Scene 8 director note to male and female.",
            "Try: change all dual-vocal director notes to male and female.",
            "Try: get the notes for Scene 14.",
            "Director notes are the best place for subject mapping, who is singing, continuity, and story beats.",
          ],
        },
        {
          title: "Prompt Fields",
          lines: [
            "Try: update Scene 5 Flux notes to low angle, harsh stage light, male vocalist.",
            "Try: write Nano B notes for Scene 9 using the director note.",
            "Try: rewrite video motion notes for Scene 3.",
            "Try: generate the image prompt for Scene 10.",
            "The agent should update notes first, then use the normal prompt generator for the selected image/video mode.",
          ],
        },
        {
          title: "Images And Video",
          lines: [
            "Try: run Flux image for Scene 4.",
            "Try: run Nano B for Scene 6.",
            "Try: generate the video prompt for Scene 10.",
            "Try: run video for Scene 10.",
            "Flux/Klein and Nano B can run with or without reference images. Drop refs only when you want character/location/style guidance.",
          ],
        },
        {
          title: "Project Lookup",
          lines: [
            "Try: get lyrics for Scene 12.",
            "Try: what context does Scene 7 have?",
            "Try: select Scene 18.",
            "Try: what should I do next for this scene?",
            "Lookup requests do not edit the project. They let the agent fetch a slice of project data instead of keeping everything in chat memory.",
          ],
        },
        {
          title: "Context Prompts",
          lines: [
            "Try: show me the current context prompts.",
            "Try: update the Theme/style context to neon courtroom, harsh spotlights, VHS grit.",
            "Try: append this to Subject/scene context: the male rapper wears a black jacket and gold chain.",
            "Try: update the Story idea context to a rap battle between two rivals in a dreamlike city.",
            "Context prompts are global project files used by the prompt generators, separate from per-scene notes.",
          ],
        },
        {
          title: "Good Phrasing",
          lines: [
            "Be specific about the target: say Scene 12, selected scene, all scenes, or timeline notes.",
            "Be specific about the field: director note, scene notes, Flux notes, Nano B notes, video notes, image prompt, or video prompt.",
            "For Auto mode, use action words like set, update, split, merge, sync, generate, run, select, or get.",
            "If you only want advice, switch to Manual mode or say do not update anything yet.",
          ],
        },
      ];
      for (const section of sections) {
        const block = document.createElement("div");
        block.style.cssText = "border:1px solid #1f2937;border-radius:7px;background:#020617;padding:11px 12px;";
        const blockTitle = document.createElement("div");
        blockTitle.style.cssText = "font-weight:900;color:#f8fafc;margin-bottom:6px;";
        blockTitle.textContent = section.title;
        block.append(blockTitle);
        for (const line of section.lines) {
          const item = document.createElement("div");
          item.style.cssText = "margin-top:5px;";
          item.textContent = line;
          block.append(item);
        }
        body.append(block);
      }
      panel.append(top, body);
      hintBackdrop.append(panel);
      document.body.append(hintBackdrop);
      const dismiss = () => hintBackdrop.remove();
      hintClose.onclick = dismiss;
      hintBackdrop.addEventListener("click", (event) => {
        if (event.target === hintBackdrop) dismiss();
      });
    }

    const refTools = document.createElement("div");
    refTools.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto auto auto;gap:8px;align-items:center;padding:10px 14px;border-bottom:1px solid #1f2937;background:#111827;";
    const agentRefDrop = document.createElement("div");
    agentRefDrop.dataset.vrgdgFileDropZone = "true";
    agentRefDrop.style.cssText = "border:1px dashed #155e75;border-radius:7px;background:#020617;color:#cffafe;padding:10px;text-align:center;font-size:12px;line-height:1.4;";
    agentRefDrop.textContent = "Drop reference image or global audio";
    const agentRefUpload = makeButton("Upload Ref");
    const agentAudioUpload = makeButton("Add Audio");
    agentAudioUpload.title = "Load global/timeline audio for music videos, songs, and visualizers.";
    const storySourceButton = makeButton("Story Source");
    storySourceButton.title = "Paste or edit lyrics/script/source text saved to the project for Story Builder.";
    const agentRefInput = document.createElement("input");
    agentRefInput.type = "file";
    agentRefInput.accept = "image/png,image/jpeg,image/webp";
    agentRefInput.multiple = true;
    agentRefInput.style.display = "none";
    shell.append(agentRefInput);
    const storyRefInput = document.createElement("input");
    storyRefInput.type = "file";
    storyRefInput.accept = "image/png,image/jpeg,image/webp";
    storyRefInput.multiple = true;
    storyRefInput.style.display = "none";
    shell.append(storyRefInput);
    refTools.append(agentRefDrop, agentRefUpload, agentAudioUpload, storySourceButton);

    const storyRefTools = document.createElement("div");
    storyRefTools.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:center;padding:10px 14px;border-bottom:1px solid #1f2937;background:#0f172a;";
    const storyRefDrop = document.createElement("div");
    storyRefDrop.dataset.vrgdgFileDropZone = "true";
    storyRefDrop.style.cssText = "border:1px dashed #155e75;border-radius:7px;background:#020617;color:#cffafe;padding:10px;text-align:center;font-size:12px;line-height:1.4;";
    const renderStoryRefDropLabel = () => {
      const count = Array.isArray(state.builderStoryReferenceImages) ? state.builderStoryReferenceImages.length : 0;
      storyRefDrop.textContent = count
        ? `${count} Story/Style image${count === 1 ? "" : "s"} loaded`
        : "Drop performers, characters, locations, or aesthetic images";
    };
    const storyRefUpload = makeButton("Upload Story Images");
    const storyRefAnalyze = makeButton("Analyze Images");
    storyRefAnalyze.title = "Use the selected vision Gemma model once to turn these images into compact notes for the text-only agent.";
    renderStoryRefDropLabel();
    storyRefTools.append(storyRefDrop, storyRefUpload, storyRefAnalyze);
    const syncAgentStoryBuilderTools = () => {
      const isStoryBuilder = purpose.value === "story_builder";
      agentAudioUpload.style.display = isStoryBuilder ? "" : "none";
      storySourceButton.style.display = isStoryBuilder ? "" : "none";
      storyRefTools.style.display = isStoryBuilder ? "grid" : "none";
      agentRefDrop.textContent = isStoryBuilder ? "Drop reference image or global audio" : "Drop reference image for active scene";
    };
    syncAgentStoryBuilderTools();

    const log = document.createElement("div");
    log.style.cssText = "min-height:0;overflow-y:auto;overflow-x:hidden;padding:14px;display:flex;flex-direction:column;gap:10px;background:#111827;";
    const composer = document.createElement("div");
    composer.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;min-height:98px;box-sizing:border-box;padding:12px 14px;border-top:1px solid #1f2937;background:#0f172a;flex:0 0 auto;";
    const input = document.createElement("textarea");
    input.placeholder = "Ask the agent about this scene, lyrics, image prompt, video motion, continuity, or what to do next...";
    input.style.cssText = "min-height:72px;max-height:170px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;";
    const send = makeButton("Send", "primary");
    send.style.minWidth = "92px";
    composer.append(input, send);
    box.append(controls, refTools, storyRefTools, log, composer);
    const ensureAgentChromeVisible = () => {
      if (controls.parentNode !== box || box.firstElementChild !== controls) {
        box.insertBefore(controls, box.firstElementChild || null);
      }
      if (agentTopbar.parentNode !== controls || controls.firstElementChild !== agentTopbar) {
        controls.insertBefore(agentTopbar, controls.firstElementChild || null);
      }
      if (controlGrid.parentNode !== controls || controlGrid.previousElementSibling !== agentTopbar) {
        controls.insertBefore(controlGrid, agentTopbar.nextElementSibling || null);
      }
      if (composer.parentNode !== box || box.lastElementChild !== composer) {
        box.append(composer);
      }
      controls.style.display = "flex";
      controls.style.flexDirection = "column";
      controls.style.visibility = "visible";
      controls.style.opacity = "1";
      agentTopbar.style.display = "flex";
      agentTopbar.style.visibility = "visible";
      agentTopbar.style.opacity = "1";
      controlGrid.style.display = "grid";
      controlGrid.style.visibility = "visible";
      controlGrid.style.opacity = "1";
      composer.style.display = "grid";
      composer.style.visibility = "visible";
      composer.style.opacity = "1";
      log.style.minHeight = "0";
      log.style.overflowY = "auto";
    };
    ensureAgentChromeVisible();
    backdrop.append(box);
    backdrop.append(restore);
    document.body.append(backdrop);
    let agentPopout = null;
    let agentPopoutBeforeUnload = null;
    let agentDockParent = backdrop;
    const resetDockedAgentStyle = () => {
      box.style.position = "";
      box.style.left = "";
      box.style.top = "";
      box.style.right = "";
      box.style.bottom = "";
      box.style.width = "min(620px,calc(100vw - 28px))";
      box.style.height = "calc(100vh - 48px)";
      box.style.margin = "34px 14px 14px";
      state.builderAgentFloating = null;
    };
    const reattachFromPopout = () => {
      const popup = agentPopout;
      if (popup && agentPopoutBeforeUnload) {
        try { popup.removeEventListener("beforeunload", agentPopoutBeforeUnload); } catch {}
      }
      agentPopout = null;
      agentPopoutBeforeUnload = null;
      if (!document.body.contains(backdrop)) document.body.append(backdrop);
      if (box.ownerDocument !== document) {
        try { document.adoptNode(box); } catch {}
      }
      if (box.parentNode !== agentDockParent) agentDockParent.insertBefore(box, restore);
      ensureAgentChromeVisible();
      backdrop.style.display = "flex";
      return popup;
    };
    const dockAgent = () => {
      const popup = reattachFromPopout();
      resetDockedAgentStyle();
      if (popup && !popup.closed) {
        try { popup.close(); } catch {}
      }
      input.focus();
    };
    const floatAgentAt = (left, top) => {
      const width = Math.min(620, Math.max(360, window.innerWidth - 28));
      const height = Math.max(360, window.innerHeight - 28);
      const nextLeft = Math.max(8, Math.min(window.innerWidth - 120, Number(left || 0)));
      const nextTop = Math.max(8, Math.min(window.innerHeight - 80, Number(top || 0)));
      box.style.position = "fixed";
      box.style.left = `${nextLeft}px`;
      box.style.top = `${nextTop}px`;
      box.style.right = "auto";
      box.style.bottom = "auto";
      box.style.width = `min(${width}px,calc(100vw - 16px))`;
      box.style.height = `min(${height}px,calc(100vh - 16px))`;
      box.style.margin = "0";
      state.builderAgentFloating = { left: nextLeft, top: nextTop };
    };
    if (state.builderAgentFloating) {
      floatAgentAt(state.builderAgentFloating.left, state.builderAgentFloating.top);
    }
    agentTopbar.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target?.closest?.("button,select,input,textarea")) return;
      if (agentPopout && !agentPopout.closed) return;
      event.preventDefault();
      const rect = box.getBoundingClientRect();
      const offsetX = event.clientX - rect.left;
      const offsetY = event.clientY - rect.top;
      floatAgentAt(rect.left, rect.top);
      const move = (moveEvent) => {
        floatAgentAt(moveEvent.clientX - offsetX, moveEvent.clientY - offsetY);
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    const popOutAgent = () => {
      if (agentPopout && !agentPopout.closed) {
        try { agentPopout.focus(); } catch {}
        return;
      }
      const popup = window.open("", "vrgdg_builder_agent", "width=680,height=900,resizable=yes,scrollbars=no");
      if (!popup) {
        toast("Popup was blocked. Allow popups for ComfyUI, then try Pop Out again.", true);
        return;
      }
      agentPopout = popup;
      popup.document.open();
      popup.document.write(`<!doctype html><html><head><title>Builder Agent</title></head><body style="margin:0;background:#0f172a;overflow:hidden;"></body></html>`);
      popup.document.close();
      popup.document.body.appendChild(box);
      ensureAgentChromeVisible();
      box.style.position = "";
      box.style.left = "";
      box.style.top = "";
      box.style.width = "100vw";
      box.style.height = "100vh";
      box.style.margin = "0";
      backdrop.style.display = "none";
      agentPopoutBeforeUnload = () => {
        reattachFromPopout();
        resetDockedAgentStyle();
      };
      popup.addEventListener("beforeunload", agentPopoutBeforeUnload);
      popup.focus();
    };

    const renderMessages = (pending = "") => {
      log.textContent = "";
      const messages = state.builderAgentMessages || [];
      if (!messages.length && !pending) {
        const empty = document.createElement("div");
        empty.style.cssText = "border:1px dashed #334155;border-radius:8px;color:#94a3b8;padding:18px;text-align:center;font-size:13px;line-height:1.5;";
        empty.textContent = purpose.value === "walkthrough"
          ? "Walkthrough gives one setup step at a time based on the current project state."
          : purpose.value === "story_builder"
          ? "Story Builder can turn lyrics and your story idea into per-scene character assignments, concept notes, and motion notes."
          : mode.value === "auto"
          ? "Auto mode can update scene notes, image prompts, Flux/Nano B prompts, and video prompts when you ask it to."
          : "Manual mode only suggests text. It will not edit the project.";
        log.append(empty);
      }
      for (const item of messages) {
        const bubble = document.createElement("div");
        const isUser = item.role === "user";
        bubble.style.cssText = `align-self:${isUser ? "flex-end" : "flex-start"};max-width:92%;border:1px solid ${isUser ? "#0e7490" : "#334155"};border-radius:8px;background:${isUser ? "#164e63" : "#020617"};color:#f8fafc;padding:10px 11px;font-size:12px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;`;
        bubble.textContent = item.content || "";
        log.append(bubble);
      }
      if (pending) {
        const bubble = document.createElement("div");
        bubble.style.cssText = "align-self:flex-start;max-width:92%;border:1px solid #334155;border-radius:8px;background:#020617;color:#94a3b8;padding:10px 11px;font-size:12px;line-height:1.5;";
        bubble.textContent = pending;
        log.append(bubble);
      }
      log.scrollTop = log.scrollHeight;
    };

    const sendMessage = async () => {
      const message = String(input.value || "").trim();
      if (!message) return;
      if (!["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner) && !String(t2iTextGemmaModelSelect.value || "").trim()) {
        toast("Choose a non-vision text Gemma model first, or use LM Studio/LLM API in LLM Runner.", true);
        return;
      }
      if (purpose.value === "story_builder") {
        await loadBuilderStorySource().catch(() => "");
      }
      const history = (state.builderAgentMessages || []).slice(-8);
      const autoApply = mode.value === "auto";
      state.builderAgentAutoApply = autoApply;
      state.builderAgentPurpose = purpose.value || "scene_work";
      state.builderAgentMessages.push({ role: "user", content: message });
      input.value = "";
      send.disabled = true;
      send.textContent = "Thinking...";
      renderMessages("Builder Agent is thinking...");
      try {
        const makeAgentPayload = (extraMessage = message, extraHistory = history, allowActions = autoApply) => ({
          ...textGemmaRunnerPayload(),
          model_file: t2iTextGemmaModelSelect.value || "",
          context: builderAgentContext(scope.value || "active_scene"),
          messages: extraHistory,
          message: extraMessage,
          auto_apply: allowActions,
          agent_purpose: purpose.value || "scene_work",
          unload_after: true,
          n_ctx: scope.value === "project_brief" || scope.value === "full_scene_plan" ? 12000 : 8000,
          max_new_tokens: purpose.value === "story_builder" && allowActions ? 1800 : allowActions ? 700 : 450,
          temperature: 0.55,
          top_p: 0.95,
        });
        let data = await postJson("/vrgdg/music_builder/agent_chat", makeAgentPayload(), 180000);
        if (autoApply && /\b(director\s+notes?|notes?|reword|rename|text)\b/i.test(message) && /\b(male(?:\s+(?:singer|vocalist|character))?\s+and\s+female|female(?:\s+(?:singer|vocalist|character))?\s+and\s+male|both\s+(?:characters|singers|vocals?|vocalists)|dual[-\s]?vocal)\b/i.test(message)) {
          data.actions = (Array.isArray(data.actions) ? data.actions : [])
            .filter((action) => !["sync_existing_scenes_to_timeline_markers", "split_selected_range_into_scenes", "split_scene_into_subscenes", "merge_scenes"].includes(String(action?.type || "")));
          const exactReplaceMatch = message.match(/\b(?:director\s+notes?|director\s+note|note)\s+says?\s+["']([^"']+)["'].*?\b(?:changed?\s+to|to|instead)\s+["']([^"']+)["']/i);
          if (exactReplaceMatch && !data.actions.some((action) => String(action?.type || "") === "replace_director_note_text")) {
            data.actions.push({ type: "replace_director_note_text", find: exactReplaceMatch[1].trim(), replace: exactReplaceMatch[2].trim() });
          }
          const replacementMatch = message.match(/\b(?:says?|to|instead|changed?\s+to)\s+["']?((?:fe)?male\s+and\s+(?:fe)?male)["']?/i);
          const replacement = replacementMatch ? replacementMatch[1].trim().toLowerCase() : "male and female";
          if (!data.actions.some((action) => String(action?.type || "") === "normalize_dual_vocal_director_notes")) {
            data.actions.push({ type: "normalize_dual_vocal_director_notes", replacement });
          }
        }
        let result = autoApply ? await applyBuilderAgentActions(data.actions || []) : { applied: [], skipped: [] };
        let reply = String(data.reply || "").trim() || "(No reply.)";
        if (autoApply && !result.applied.length && !result.skipped.length) {
          const combinedIntent = `${message}\n${reply}`;
          const fallbackActions = [];
          const mergeIntent = /\b(combine|merge|join|consolidate)\b/i.test(message);
          const splitIntent = /\b(split|break|divide|sub[-\s]?scenes?|sub[-\s]?segments?)\b/i.test(message) || /\b(split|break|divide|sub[-\s]?scenes?|sub[-\s]?segments?)\b/i.test(reply);
          if (mergeIntent) {
            const rangeMatch = message.match(/\bscenes?\s+(\d+)\s*(?:-|to|through)\s*(\d+)\b/i);
            let sceneNumbers = [];
            if (rangeMatch) {
              const startNumber = Number(rangeMatch[1]);
              const endNumber = Number(rangeMatch[2]);
              if (Number.isFinite(startNumber) && Number.isFinite(endNumber) && endNumber >= startNumber) {
                sceneNumbers = Array.from({ length: endNumber - startNumber + 1 }, (_, index) => startNumber + index);
              }
            }
            if (!sceneNumbers.length) {
              const sequenceMatch = message.match(/\bscenes?\s+((?:\d+\s*(?:,|and)?\s*){2,})/i);
              if (sequenceMatch) sceneNumbers = (sequenceMatch[1].match(/\d+/g) || []).map((value) => Number(value)).filter(Boolean);
            }
            if (sceneNumbers.length >= 2) {
              fallbackActions.push({ type: "merge_scenes", scene_numbers: Array.from(new Set(sceneNumbers)), label: `Scene ${sceneNumbers[0]}` });
            }
          } else if (splitIntent) {
            const sceneMatch = combinedIntent.match(/\bscene\s+(\d+)\b/i);
            const countMatch = combinedIntent.match(/\b(?:into|in|to)\s+(\d+)\s*(?:sub[-\s]?scenes?|scenes?|segments?)\b/i)
              || combinedIntent.match(/\b(\d+)\s*(?:sub[-\s]?scenes?|sub[-\s]?segments?)\b/i);
            if (sceneMatch && countMatch) {
              const sceneNumber = Number(sceneMatch[1]);
              const sceneCount = Math.max(2, Math.min(24, Number(countMatch[1])));
              if (Number.isFinite(sceneNumber) && Number.isFinite(sceneCount)) {
                fallbackActions.push({ type: "split_scene_into_subscenes", scene_number: sceneNumber, scene_count: sceneCount, label_prefix: `Scene ${sceneNumber}` });
              }
            }
          }
          if (fallbackActions.length) {
            result = await applyBuilderAgentActions(fallbackActions);
            data.actions = fallbackActions;
          }
        }
        const fetchedResults = result.applied.filter((item) => /\b(?:lyrics|context):\n/.test(item) || /^Story Source:\n/.test(item));
        if (fetchedResults.length) {
          renderMessages("Builder Agent is reading the fetched scene data...");
          const toolMessage = [
            "Local tool results were fetched from the project. Answer the user's original request using these results.",
            "Keep the reply short and do not request the same data again.",
            "",
            fetchedResults.join("\n\n"),
          ].join("\n");
          const secondHistory = [
            ...history,
            { role: "user", content: message },
            { role: "assistant", content: reply },
          ].slice(-8);
          const secondData = await postJson("/vrgdg/music_builder/agent_chat", makeAgentPayload(toolMessage, secondHistory, false), 180000);
          if (String(secondData.reply || "").trim()) {
            data = secondData;
            reply = String(secondData.reply || "").trim();
          }
        }
        if (result.applied.length) {
          const updates = result.applied.filter((item) => !/\b(?:lyrics|context):\n/.test(item));
          if (updates.length) {
            reply += `\n\nUpdated: ${updates.join("; ")}`;
            toast(`Builder Agent updated ${updates.length} field${updates.length === 1 ? "" : "s"}.`);
          }
        }
        if (autoApply && !result.applied.length && Array.isArray(data.actions) && data.actions.length) {
          reply += "\n\nNo supported changes were applied.";
        }
        if (result.skipped.length) {
          reply += `\n\nSkipped: ${result.skipped.slice(0, 3).join("; ")}`;
        }
        state.builderAgentMessages.push({ role: "assistant", content: reply });
      } catch (error) {
        state.builderAgentMessages.push({ role: "assistant", content: `Agent failed:\n${String(error?.message || error)}` });
      } finally {
        send.disabled = false;
        send.textContent = "Send";
        autoSaveSessionQuiet("Builder Agent chat").catch(() => null);
        renderMessages();
      }
    };

    const rememberAgentReference = (source) => {
      if (!source?.path && !source?.data) return;
      if (!Array.isArray(state.builderAgentReferenceImages)) state.builderAgentReferenceImages = [];
      const key = String(source.path || source.data || source.name || "").trim();
      if (key && !state.builderAgentReferenceImages.some((item) => String(item.path || item.data || item.name || "").trim() === key)) {
        state.builderAgentReferenceImages.push({
          path: source.path || "",
          data: source.data || "",
          name: source.name || "agent_reference.png",
        });
        state.builderAgentReferenceImages = state.builderAgentReferenceImages.slice(-12);
      }
    };
    const attachAgentReferenceSource = (source, labelText = "reference image") => {
      const segment = activeSegment();
      if (!segment) {
        toast("Select a scene before attaching an Agent reference image.", true);
        return;
      }
      rememberAgentReference(source);
      addFluxIngredient(source);
      state.builderAgentMessages.push({
        role: "assistant",
        content: `Attached ${labelText} to ${sceneDisplayName(segment, segmentIndexInfo(segment).index)}. I can also reuse recent Agent refs if you ask me to work on another scene.`,
      });
      renderMessages();
    };
    const attachAgentReferenceFile = (file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => attachAgentReferenceSource({ data: String(reader.result || ""), name: file.name || "image.png" }, file.name || "reference image");
      reader.onerror = () => toast("Failed to read the Agent reference image.", true);
      reader.readAsDataURL(file);
    };
    const rememberStoryReferenceFile = (file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        if (!Array.isArray(state.builderStoryReferenceImages)) state.builderStoryReferenceImages = [];
        const item = { data: String(reader.result || ""), name: file.name || "story_reference.png" };
        const key = item.data || item.name;
        if (!state.builderStoryReferenceImages.some((existing) => String(existing.data || existing.path || existing.name || "") === key)) {
          state.builderStoryReferenceImages.push(item);
          state.builderStoryReferenceImages = state.builderStoryReferenceImages.slice(-24);
        }
        renderStoryRefDropLabel();
        state.builderAgentMessages.push({ role: "assistant", content: `Added Story/Style image: ${item.name}. Use Analyze Images when you want me to turn these into compact notes.` });
        renderMessages();
        autoSaveSessionQuiet("Story Builder reference image added").catch(() => null);
      };
      reader.onerror = () => toast("Failed to read the Story/Style reference image.", true);
      reader.readAsDataURL(file);
    };
    const analyzeStoryReferences = async () => {
      const refs = Array.isArray(state.builderStoryReferenceImages) ? state.builderStoryReferenceImages : [];
      if (!refs.length) {
        toast("Drop or upload Story/Style images first.", true);
        return;
      }
      const modelFile = String(gemmaModelSelect.value || fluxGemmaModelSelect.value || nbGemmaModelSelect.value || i2vGemmaModelSelect.value || "").trim();
      const mmprojFile = String(mmprojSelect.value || fluxMmprojSelect.value || nbMmprojSelect.value || i2vMmprojSelect.value || "").trim();
      if (!["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner) && !modelFile) {
        toast("Choose a vision Gemma model first.", true);
        return;
      }
      let progress = null;
      try {
        storyRefAnalyze.disabled = true;
        storyRefAnalyze.textContent = "Analyzing...";
        progress = createProgressWindow("Story Builder reference images");
        progress.set(`Compressing Story/Style images into 512px tiles and running ${gemmaRunnerLabel({ vision: true })}...`, 18);
        const data = await postJson("/vrgdg/music_builder/analyze_story_references", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          mmproj_file: mmprojFile,
          image_ingredients: refs,
          user_notes: "These are performer, character, location, and aesthetic references for the whole project.",
          unload_after: true,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          max_new_tokens: 500,
          temperature: 0.25,
          top_p: 0.95,
        }, 10 * 60 * 1000);
        state.builderStoryReferenceNotes = String(data.notes || "").trim();
        await autoSaveSessionQuiet("Story Builder reference notes analyzed");
        progress.set(data.unloaded ? "Story/Style notes saved. Vision model unloaded." : "Story/Style notes saved.", 100);
        progress.close(1200);
        state.builderAgentMessages.push({
          role: "assistant",
          content: `Story/Style image notes saved for the text-only agent:\n${state.builderStoryReferenceNotes || "(empty)"}`,
        });
        renderMessages();
      } catch (error) {
        progress?.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      } finally {
        storyRefAnalyze.disabled = false;
        storyRefAnalyze.textContent = "Analyze Images";
      }
    };
    const openStorySourceEditor = () => {
      const storyInput = document.createElement("input");
      storyInput.value = builderStorySourcePath();
      editContextTextFile(storyInput, "Edit Story Builder Source", "AgentStorySource.txt", null, {
        showGemma: false,
        helpText: "Paste lyrics, script, battle verses, shot outline, or story source here. Story Builder saves this to the project and fetches it only when needed.",
        afterSave: async (result, content) => {
          state.builderStorySourcePath = result.path || storyInput.value || builderStorySourcePath();
          state.builderStorySourcePreview = String(content || "").trim().slice(0, 500);
          state.builderAgentMessages.push({ role: "assistant", content: `Story Source saved. I can fetch it when you ask me to split scenes or plan the video.` });
          renderMessages();
        },
      });
    };
    agentRefUpload.onclick = () => agentRefInput.click();
    agentAudioUpload.onclick = () => projectAudioFileInput.click();
    storySourceButton.onclick = openStorySourceEditor;
    storyRefUpload.onclick = () => storyRefInput.click();
    storyRefAnalyze.onclick = analyzeStoryReferences;
    agentRefInput.addEventListener("change", () => {
      const files = Array.from(agentRefInput.files || []).filter((file) => file.type?.startsWith?.("image/"));
      files.forEach(attachAgentReferenceFile);
      agentRefInput.value = "";
    });
    storyRefInput.addEventListener("change", () => {
      const files = Array.from(storyRefInput.files || []).filter((file) => file.type?.startsWith?.("image/"));
      files.forEach(rememberStoryReferenceFile);
      storyRefInput.value = "";
    });
    storyRefDrop.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []).map((item) => String(item).toLowerCase());
      if (!types.includes("files")) return;
      event.preventDefault();
      event.stopPropagation();
      storyRefDrop.style.borderColor = "#67e8f9";
    });
    storyRefDrop.addEventListener("dragleave", () => {
      storyRefDrop.style.borderColor = "#155e75";
    });
    storyRefDrop.addEventListener("drop", (event) => {
      const files = Array.from(event.dataTransfer?.files || []).filter((file) => file.type?.startsWith?.("image/"));
      if (!files.length) return;
      event.preventDefault();
      event.stopPropagation();
      storyRefDrop.style.borderColor = "#155e75";
      files.forEach(rememberStoryReferenceFile);
    });
    agentRefDrop.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []).map((item) => String(item).toLowerCase());
      if (!types.includes("files") && !types.includes("application/x-vrgdg-segment-id")) return;
      event.preventDefault();
      event.stopPropagation();
      agentRefDrop.style.borderColor = "#67e8f9";
    });
    agentRefDrop.addEventListener("dragleave", () => {
      agentRefDrop.style.borderColor = "#155e75";
    });
    agentRefDrop.addEventListener("drop", (event) => {
      const sceneSource = droppedSceneImageSource(event);
      const droppedFiles = Array.from(event.dataTransfer?.files || []);
      const audioFile = purpose.value === "story_builder"
        ? droppedFiles.find((file) => file.type?.startsWith?.("audio/") || /\.(wav|mp3|flac|m4a|ogg)$/i.test(file.name || ""))
        : null;
      const files = droppedFiles.filter((file) => file.type?.startsWith?.("image/"));
      if (!sceneSource && !files.length && !audioFile) return;
      event.preventDefault();
      event.stopPropagation();
      agentRefDrop.style.borderColor = "#155e75";
      if (audioFile) {
        chooseProjectAudioFile(audioFile);
        state.builderAgentMessages.push({ role: "assistant", content: `Global audio added from ${audioFile.name || "audio file"}. For music videos, I will use that as the timeline audio.` });
        renderMessages();
        return;
      }
      const segment = activeSegment();
      if (!segment) {
        toast("Select a scene before attaching an Agent reference image.", true);
        return;
      }
      if (sceneSource) {
        attachAgentReferenceSource({
          path: sceneSource.path || "",
          data: sceneSource.data || "",
          name: sceneSource.name || "scene_image.png",
        }, "scene image reference");
        return;
      }
      files.forEach(attachAgentReferenceFile);
    });

    const minimizeAgent = () => {
      const popup = agentPopout && !agentPopout.closed ? reattachFromPopout() : null;
      if (popup && !popup.closed) {
        try { popup.close(); } catch {}
      }
      box.style.display = "none";
      restore.style.display = "block";
    };
    const closeAgent = () => {
      autoSaveSessionQuiet("Builder Agent closed").catch(() => null);
      const popup = agentPopout && !agentPopout.closed ? reattachFromPopout() : null;
      if (popup && !popup.closed) {
        try { popup.close(); } catch {}
      }
      backdrop.remove();
    };
    hint.onclick = openAgentHintPopup;
    minimize.onclick = minimizeAgent;
    restore.onclick = () => {
      restore.style.display = "none";
      box.style.display = "flex";
      ensureAgentChromeVisible();
      input.focus();
    };
    dock.onclick = dockAgent;
    popout.onclick = popOutAgent;
    close.onclick = closeAgent;
    mode.onchange = () => {
      state.builderAgentAutoApply = mode.value === "auto";
      autoSaveSessionQuiet("Builder Agent mode").catch(() => null);
      renderMessages();
    };
    purpose.onchange = () => {
      state.builderAgentPurpose = purpose.value || "scene_work";
      if (purpose.value === "story_builder" && scope.value === "active_scene") {
        scope.value = "full_scene_plan";
      }
      syncAgentStoryBuilderTools();
      autoSaveSessionQuiet("Builder Agent purpose").catch(() => null);
      renderMessages();
    };
    clear.onclick = () => {
      state.builderAgentMessages = [];
      autoSaveSessionQuiet("Builder Agent chat cleared").catch(() => null);
      renderMessages();
    };
    send.onclick = sendMessage;
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendMessage();
      }
    });
    renderMessages();
    setTimeout(() => input.focus(), 0);
  }

  return { openBuilderAgentModal };
}
