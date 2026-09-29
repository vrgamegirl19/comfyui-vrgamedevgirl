import { normalizeStoryboardScriptImportState } from "./script_import.mjs";
import { storyboardTemporalIntensity, storyboardTemporalProtectedMode } from "./video_style.mjs";

function statusMeta(scene) {
  const hasImage = Boolean(String(scene.image_path || "").trim() || String(scene.image_data || scene.image_reference_data || "").trim());
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

export function normalizeStoryboardPerformanceMode(value = "") {
  const text = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["speaking", "short_film", "dialogue", "dialog"].includes(text)) return "speaking";
  if (["no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"].includes(text)) return "no_lip_sync";
  return "singing";
}

export function normalizeStoryboardProjectVideoEngine(value = "") {
  return String(value || "").trim().toLowerCase() === "minimax_h3" ? "minimax_h3" : "ltx";
}

export function normalizeStoryboardMiniMaxH3Mode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ["text_to_video", "image_to_video", "reference_to_video", "video_to_video"].includes(clean)
    ? clean
    : "text_to_video";
}

export function normalizeStoryboardMiniMaxH3AudioMode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ["built_in_audio", "native_audio", "generated_audio"].includes(clean) ? "built_in_audio" : "input_audio";
}

export function normalizeStoryboardShortFilmPlanningMode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return clean === "fully_custom" || clean === "custom" ? "fully_custom" : "guided_film";
}

export function normalizeStoryboardSpeakerAssignments(value = []) {
  return (Array.isArray(value) ? value : [])
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || item.cue_id || item.cueId || `speaker_cue_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
      speaker_id: String(item.speaker_id || item.speakerId || item.subject_id || item.subjectId || ""),
      speaker_name: String(item.speaker_name || item.speakerName || item.speaker || item.character || "").trim(),
      text: String(item.text || item.dialogue || item.line || item.lyric || "").trim(),
    }))
    .slice(0, 40);
}

export function storyboardStillFacialDirection(value = "") {
  return String(value || "")
    .replace(/\bsubtle natural eye movement\b/gi, "clear eye direction")
    .replace(/\bsubtle eye movement\b/gi, "clear eye direction")
    .replace(/\boccasional natural blinking\b/gi, "natural eyelid detail")
    .replace(/\bnatural blinking\b/gi, "natural eyelid detail")
    .replace(/\bfast-moving mouth during delivery\b/gi, "mouth captured in a still expressive shape")
    .replace(/\bmouth open mid-verse\b/gi, "mouth captured in a still expressive shape")
    .replace(/\blips slightly parted while singing\b/gi, "lips slightly parted in a still performance expression")
    .replace(/\bsnarling mouth shapes during vocals\b/gi, "snarling still mouth expression")
    .replace(/\bbared teeth on powerful notes\b/gi, "bared teeth in a powerful still expression")
    .replace(/\braw emotional scream expression\b/gi, "raw emotional still expression")
    .replace(/\bforceful singing expression\b/gi, "forceful performance expression")
    .replace(/\bmovement\b/gi, "pose")
    .replace(/\bmoving\b/gi, "posed")
    .replace(/\bduring vocals?\b/gi, "in the expression")
    .replace(/\bwhile singing\b/gi, "in the expression")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export function normalizeVideoPromptOrigin(value) {
  return String(value || "").trim().toLowerCase() === "gemma" ? "gemma" : "manual";
}

export function normalizeScene(scene = {}, index = 0) {
  const rawVideoType = String(scene.video_prompt_type || scene.video_type || scene.mode || "").trim();
  const videoPromptType = ["i2v", "id_lora", "t2v", "rtv", "ingredients", "flf"].includes(rawVideoType) ? rawVideoType : "i2v";
  const lyrics = scene.lyrics || scene.lyric_text || "";
  const lyricSingers = Array.isArray(scene.lyric_singers)
    ? scene.lyric_singers.map((item) => String(item || "").trim()).filter(Boolean)
    : String(scene.lyric_singers || scene.singers || "").split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
  const lyricNoLipSync = Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync || scene.noLipSync || scene.broll || scene.b_roll);
  const lyricInstrumental = Boolean(scene.lyric_instrumental || scene.instrumental || storyboardIsInstrumentalText(lyrics));
  const lyricCueMap = Array.isArray(scene.lyric_cue_map)
    ? scene.lyric_cue_map.map((cue) => ({ ...cue }))
    : [];
  const noCharacterPresent = Boolean(scene.no_character_present || scene.noCharacterPresent || scene.no_subject || scene.no_visible_subject);
  const extraSubjects = noCharacterPresent || !Array.isArray(scene.extra_subjects || scene.extraSubjects)
    ? []
    : (scene.extra_subjects || scene.extraSubjects).filter((item) => item && typeof item === "object").map((item, extraIndex) => ({
        id: String(item.id || `extra_${extraIndex + 1}`).trim(),
        name: String(item.name || item.title || `Extra ${extraIndex + 1}`).replace(/\s+/g, " ").trim(),
        count: Math.max(1, Math.min(100, Math.round(Number(item.count) || 1))),
        interaction: ["background", "background_dancing", "alongside", "dancing_with", "direct"].includes(String(item.interaction || "").trim())
          ? String(item.interaction).trim()
          : "background",
        identity: String(item.identity || item.description || "").replace(/\s+/g, " ").trim().slice(0, 240),
      }));
  return {
    id: scene.id || `storyboard_scene_${index + 1}_${Date.now()}`,
    scene_number: Number(scene.scene_number || scene.number || index + 1),
    label: scene.label || `Scene ${index + 1}`,
    lyrics,
    lyric_section: scene.lyric_section || scene.section || scene.song_section || "",
    story_beat: scene.story_beat || scene.scene_story_beat || scene.narrative_beat || "",
    flf_start_state: scene.flf_start_state || scene.first_frame_state || "",
    flf_transformation: scene.flf_transformation || scene.transition_action || "",
    flf_end_state: scene.flf_end_state || scene.last_frame_state || "",
    flf_carry_forward: scene.flf_carry_forward || scene.carry_forward_state || "",
    performance_mode: normalizeStoryboardPerformanceMode(scene.performance_mode || scene.performanceMode || scene.video_performance_mode || scene.videoPerformanceMode),
    lyric_singers: lyricSingers,
    lyric_cue_map: lyricCueMap,
    lyric_shot_word_timing_enabled: Boolean(scene.lyric_shot_word_timing_enabled),
    lyric_performance_mode: String(scene.lyric_performance_mode || ""),
    timed_lyric_cue_contract: String(scene.timed_lyric_cue_contract || ""),
    speaker_assignments: normalizeStoryboardSpeakerAssignments(scene.speaker_assignments || scene.minimax_speaker_assignments || scene.dialogue_cues),
    lyric_no_lip_sync: lyricNoLipSync,
    lyric_instrumental: lyricInstrumental,
    no_character_present: noCharacterPresent,
    prompt_summary: scene.prompt_summary || scene.summary || "",
    motion_summary: scene.motion_summary || scene.video_notes || scene.i2v_notes || "",
    subjects: Array.isArray(scene.subjects) ? scene.subjects : String(scene.subjects || "").split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean),
    subject_refs: noCharacterPresent ? [] : Array.isArray(scene.subject_refs) ? scene.subject_refs.filter((item) => item && typeof item === "object") : [],
    extra_subjects: extraSubjects,
    setting: scene.setting || scene.location_ref?.description || scene.location_ref?.name || scene.location || "",
    location_ref: scene.location_ref && typeof scene.location_ref === "object" ? scene.location_ref : null,
    trigger_phrase: String(scene.trigger_phrase || scene.trigger || scene.Trigger || ""),
    trigger_position: String(scene.trigger_position || scene.triggerPosition || scene.trigger_placement || "start") === "end" ? "end" : "start",
    video_prompt_type: videoPromptType,
    project_video_engine: normalizeStoryboardProjectVideoEngine(scene.project_video_engine || scene.projectVideoEngine),
    minimax_h3_mode: normalizeStoryboardMiniMaxH3Mode(scene.minimax_h3_mode || scene.minimaxH3Mode),
    minimax_h3_audio_mode: normalizeStoryboardMiniMaxH3AudioMode(scene.minimax_h3_audio_mode || scene.minimaxH3AudioMode),
    video_style: String(scene.video_style || scene.videoStyle || ""),
    video_style_custom: String(scene.video_style_custom || scene.videoStyleCustom || ""),
    temporal_world_effect_override: String(scene.temporal_world_effect_override || scene.temporalWorldEffectOverride || "global"),
    temporal_world_effect_custom: String(scene.temporal_world_effect_custom || scene.temporalWorldEffectCustom || ""),
    timeline_start: Number(scene.timeline_start ?? scene.start ?? 0),
    timeline_end: Number(scene.timeline_end ?? scene.end ?? 0),
    exact_duration: Math.max(0, Number(scene.exact_duration ?? scene.duration ?? 0)),
    shot_type: scene.shot_type || "",
    camera_motion: scene.camera_motion || scene.motion_preset || "",
    character_motion: scene.character_motion || scene.character_motion_preset || scene.subject_motion || "",
    performance_style: scene.performance_style || scene.song_style || scene.music_style || "",
    facial_performance: scene.facial_performance || scene.facialPerformance || scene.facial_expression || scene.facialExpression || "",
    facial_performance_custom: scene.facial_performance_custom || scene.facialPerformanceCustom || scene.facial_expression_custom || scene.facialExpressionCustom || "",
    include_microphone: Boolean(scene.include_microphone || scene.use_microphone || scene.microphone),
    status: scene.status || "draft",
    image_prompt: scene.image_prompt || scene.t2i_prompt || "",
    video_prompt: scene.video_prompt || scene.i2v_prompt || scene.t2v_prompt || "",
    video_prompt_origin: normalizeVideoPromptOrigin(scene.video_prompt_origin || scene.i2v_prompt_origin),
    image_path: scene.image_path || scene.approved_image_path || "",
    image_data: scene.image_data || scene.image_reference_data || "",
    notes: scene.notes || "",
    audio_direction: scene.audio_direction || scene.audioDirection || "",
    continuity: scene.continuity || scene.continuity_direction || scene.continuityDirection || "",
    id_lora_character_id: scene.id_lora_character_id || scene.character_id || scene.subject_id || "",
    id_lora_location_id: scene.id_lora_location_id || scene.location_id || "",
  };
}

function storyboardReferenceOpening(scene = {}) {
  const normalized = normalizeScene(scene, 0);
  const subjectCount = normalized.no_character_present
    ? 0
    : normalized.subject_refs.filter((subject) => {
        const image = subject?.image || subject || {};
        return Boolean(image.path || image.data || subject?.image_path || subject?.image_data);
      }).length;
  const locationImage = normalized.location_ref?.image || normalized.location_ref || {};
  const hasLocation = Boolean(locationImage.path || locationImage.data || normalized.location_ref?.image_path || normalized.location_ref?.image_data);
  if (!subjectCount && !hasLocation) return "";
  const characterPhrase = subjectCount > 1 ? "character reference images" : "character reference image";
  if (subjectCount && hasLocation) return `Using the provided ${characterPhrase} and location reference image`;
  if (subjectCount) return `Using the provided ${characterPhrase}`;
  return "Using the provided location reference image";
}

function storyboardImageModeUsesReferenceOpening(imageMode = "") {
  return ["nano_banana", "flux_klein", "flow_gpt"].includes(String(imageMode || "").trim());
}

export function ensureStoryboardReferenceOpening(prompt, scene = {}, imageMode = "") {
  if (!storyboardImageModeUsesReferenceOpening(imageMode)) return String(prompt || "").trim();
  const opening = storyboardReferenceOpening(scene);
  let text = String(prompt || "").trim();
  if (!opening || !text) return text;
  text = text.replace(
    /^Using the provided\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?)(?:\s+and\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?))*\s*,?\s*(?:create\s+)?/i,
    "",
  ).trim();
  text = text.replace(
    /^and\s+(?:(?:character|location|scene|reference)\s+)*(?:images?|references?)\s*,?\s*(?:create\s+)?/i,
    "",
  ).trim();
  text = text.replace(/^(?:create|make|generate)\b\s*/i, "").trim();
  if (!text) return `${opening}, create a cinematic still image.`;
  return `${opening}, create ${text.slice(0, 1).toLowerCase()}${text.slice(1)}`;
}

export function scenesFromBuilderPayload(payload = {}) {
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  return scenes.map((scene, index) => normalizeScene({
    id: scene.id,
    scene_number: index + 1,
    label: scene.label || `Scene ${index + 1}`,
    lyrics: scene.lyric_text || scene.lyrics || "",
    lyric_section: scene.lyric_section || scene.section || scene.song_section || "",
    story_beat: scene.story_beat || scene.scene_story_beat || scene.narrative_beat || "",
    flf_start_state: scene.flf_start_state || scene.first_frame_state || "",
    flf_transformation: scene.flf_transformation || scene.transition_action || "",
    flf_end_state: scene.flf_end_state || scene.last_frame_state || "",
    flf_carry_forward: scene.flf_carry_forward || scene.carry_forward_state || "",
    performance_mode: scene.performance_mode || scene.performanceMode || payload.performance_mode || payload.performanceMode || "",
    lyric_singers: scene.lyric_singers || scene.singers || [],
    speaker_assignments: scene.speaker_assignments || scene.minimax_speaker_assignments || scene.dialogue_cues || [],
    lyric_no_lip_sync: Boolean(scene.lyric_no_lip_sync || scene.no_lip_sync),
    lyric_instrumental: Boolean(scene.lyric_instrumental || scene.instrumental),
    no_character_present: Boolean(scene.no_character_present || scene.noCharacterPresent || scene.no_subject || scene.no_visible_subject),
    prompt_summary: scene.notes || scene.director_note || scene.t2i_prompt || "",
    motion_summary: scene.video_notes || scene.i2v_notes || "",
    subjects: scene.lyric_singers || scene.subjects || "",
    subject_refs: scene.subject_refs || [],
    setting: scene.location || scene.location_ref?.description || scene.location_ref?.name || "",
    location_ref: scene.location_ref || null,
    project_video_engine: scene.project_video_engine || scene.projectVideoEngine || payload.project_video_engine || payload.projectVideoEngine || "",
    minimax_h3_mode: scene.minimax_h3_mode || scene.minimaxH3Mode || "",
    minimax_h3_audio_mode: scene.minimax_h3_audio_mode || scene.minimaxH3AudioMode || payload.minimax_h3_audio_mode || payload.miniMaxH3AudioMode || "",
    video_style: scene.video_style || scene.videoStyle || "",
    video_style_custom: scene.video_style_custom || scene.videoStyleCustom || "",
    temporal_world_effect_override: scene.temporal_world_effect_override || scene.temporalWorldEffectOverride || "global",
    temporal_world_effect_custom: scene.temporal_world_effect_custom || scene.temporalWorldEffectCustom || "",
    timeline_start: scene.timeline_start ?? scene.start ?? 0,
    timeline_end: scene.timeline_end ?? scene.end ?? 0,
    exact_duration: scene.exact_duration ?? scene.duration ?? 0,
    video_prompt_type: scene.video_prompt_type || scene.video_type || "",
      shot_type: scene.shot_type || "",
      camera_motion: scene.camera_motion || scene.motion_preset || "",
      character_motion: scene.character_motion || scene.character_motion_preset || scene.subject_motion || "",
      performance_style: scene.performance_style || scene.song_style || scene.music_style || "",
      facial_performance: scene.facial_performance || scene.facialPerformance || scene.facial_expression || scene.facialExpression || "",
      facial_performance_custom: scene.facial_performance_custom || scene.facialPerformanceCustom || scene.facial_expression_custom || scene.facialExpressionCustom || "",
      include_microphone: Boolean(scene.include_microphone || scene.use_microphone || scene.microphone),
      image_prompt: scene.t2i_prompt || "",
    video_prompt: scene.i2v_prompt || scene.t2v_prompt || "",
    video_prompt_origin: normalizeVideoPromptOrigin(scene.video_prompt_origin || scene.i2v_prompt_origin),
    image_path: scene.image_path || scene.approved_image_path || "",
    image_data: scene.image_data || scene.image_reference_data || "",
    notes: scene.notes || "",
    audio_direction: scene.audio_direction || "",
    continuity: scene.continuity || scene.continuity_direction || "",
  }, index));
}

function storyboardPayloadFromBuilder(payload = {}) {
  return {
    project_folder: payload.projectFolder || payload.project_folder || "",
    scenes: scenesFromBuilderPayload(payload),
  };
}

export function slimReferenceForRequest(ref) {
  if (!ref || typeof ref !== "object") return null;
  return {
    id: String(ref.id || ""),
    name: String(ref.name || ""),
    description: String(ref.description || ""),
    minimax_voice: ref.minimax_voice && typeof ref.minimax_voice === "object" ? { ...ref.minimax_voice } : {},
    trigger_phrase: String(ref.trigger_phrase || ref.trigger || ref.Trigger || ""),
    trigger_position: String(ref.trigger_position || ref.triggerPosition || ref.trigger_placement || "start") === "end" ? "end" : "start",
    image: {
      path: String(ref.image?.path || ""),
      name: String(ref.image?.name || ""),
      data: "",
    },
  };
}

export function slimSceneForRequest(scene, index = 0) {
  const normalized = normalizeScene(scene, index);
  return {
    ...normalized,
    subject_refs: (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map(slimReferenceForRequest)
      .filter(Boolean),
    location_ref: slimReferenceForRequest(normalized.location_ref),
  };
}

export function normalizeStoryLayer(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const lyricStoryStrength = Math.max(0, Math.min(10, Number(source.lyric_story_strength ?? source.lyricStoryStrength ?? 7)));
  const imageWorldStyle = ["natural", "surreal_subject", "balanced_surreal", "full_surreal", "abstract", "custom"].includes(String(source.image_world_style || source.imageWorldStyle || "natural"))
    ? String(source.image_world_style || source.imageWorldStyle || "natural")
    : "natural";
  return {
    enabled: source.enabled !== false,
    overall_story_idea: String(source.overall_story_idea || source.overallStoryIdea || source.story_idea || source.storyIdea || ""),
    user_story_arc: String(source.user_story_arc || source.userStoryArc || ""),
    song_story_brief: String(source.song_story_brief || source.songStoryBrief || ""),
    lyric_story_strength: Number.isFinite(lyricStoryStrength) ? lyricStoryStrength : 7,
    image_world_style: imageWorldStyle,
    image_custom_style_direction: String(source.image_custom_style_direction || source.imageCustomStyleDirection || ""),
  };
}

export function storyboardSpeedValue(value, fallback = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(10, number)) : fallback;
}

export function storyboardCutFrequencyValue(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(10, Math.round(number))) : fallback;
}

export function storyboardCutFrequencyLabel(value) {
  const frequency = storyboardCutFrequencyValue(value);
  if (frequency <= 0) return "0 / continuous shot";
  if (frequency >= 10) return "10 / cut every second";
  if (frequency <= 3) return `${frequency} / occasional cuts`;
  if (frequency <= 6) return `${frequency} / moderate cuts`;
  return `${frequency} / frequent cuts`;
}

export function storyboardCutPlanForDuration(durationValue, frequencyValue, engineValue = "minimax_h3") {
  const duration = Math.max(0, Number(durationValue) || 0);
  const frequency = storyboardCutFrequencyValue(frequencyValue);
  const maximumCuts = Math.max(0, Math.ceil(Math.max(0, duration - 0.000001)) - 1);
  const nonMaximumCutLimit = Math.max(1, maximumCuts - 1);
  const cutCount = frequency <= 0 || maximumCuts <= 0
    ? 0
    : frequency >= 10
      ? maximumCuts
      : Math.min(nonMaximumCutLimit, Math.max(1, Math.round(maximumCuts * frequency / 10)));
  let cutTimes = [];
  if (cutCount > 0) {
    cutTimes = cutCount === maximumCuts
      ? Array.from({ length: cutCount }, (_, index) => index + 1)
      : Array.from({ length: cutCount }, (_, index) => duration * (index + 1) / (cutCount + 1));
    cutTimes = cutTimes.map((time) => Number(time.toFixed(3)));
  }
  const exactDuration = Number(duration.toFixed(3));
  const timingText = cutTimes.map((time) => `${time}s`).join(", ");
  const miniMax = normalizeStoryboardProjectVideoEngine(engineValue) === "minimax_h3";
  const instruction = cutCount > 0
    ? miniMax
      ? `EDITING / CUT PLAN — MANDATORY: Cut frequency ${frequency}/10 for this exact ${exactDuration}-second segment requires exactly ${cutCount} hard CUT TO transition${cutCount === 1 ? "" : "s"}, at approximately ${timingText}, creating ${cutCount + 1} coherent shots. Begin with shot 1 at 0s. At every listed time, write an explicit new timestamp block beginning with CUT TO: and change to a clearly different but continuity-preserving angle, framing, or story detail within the same scene and ongoing action. Preserve identity, wardrobe, location, lighting, props, spatial direction, and action continuity. Do not add extra cuts, montage beats, dissolves, scene changes, or transitions outside this schedule.`
      : `EDITING / CUT PLAN — MANDATORY FOR LTX: Cut frequency ${frequency}/10 for this exact ${exactDuration}-second segment requires exactly ${cutCount} clear cut${cutCount === 1 ? "" : "s"}, creating ${cutCount + 1} coherent shots. Write the edits in ordinary natural language in chronological order, using phrases such as "then cut to" or "cut to a different angle". Do not use MiniMax timestamp blocks or a special prompt schema. Each cut must introduce a clearly different but continuity-preserving angle, framing, or story detail within the same location and ongoing action. Preserve identity, wardrobe, lighting, props, spatial direction, and action continuity. Do not add extra cuts, montage beats, dissolves, scene changes, or transitions.`
    : `EDITING / CUT PLAN — MANDATORY: Use one smooth, continuous, uninterrupted shot for the full ${exactDuration}-second segment. Use no hard cut, angle reset, montage, dissolve, scene change, or transition. Camera and character movement may develop inside the same continuous take.`;
  return {
    frequency,
    exact_duration_seconds: exactDuration,
    maximum_one_per_second_cuts: maximumCuts,
    cut_count: cutCount,
    shot_count: cutCount + 1,
    cut_times_seconds: cutTimes,
    continuous_shot: cutCount === 0,
    instruction,
  };
}

export function storyboardSpeedLabel(value, kind = "motion") {
  const speed = storyboardSpeedValue(value);
  if (speed <= 0) return kind === "camera" ? "0 / static camera" : "0 / still subject";
  if (speed <= 3) return `${speed} / subtle`;
  if (speed <= 6) return `${speed} / active`;
  if (speed <= 8) return `${speed} / energetic`;
  return `${speed} / fast action`;
}

export function storyboardSpeedGuidance(value, kind = "motion") {
  const speed = storyboardSpeedValue(value);
  if (kind === "camera") {
    if (speed <= 0) return "Camera speed 0/10: locked-off static camera, no camera movement.";
    if (speed <= 3) return `Camera speed ${speed}/10: slow, gentle camera motion; one simple move at most.`;
    if (speed <= 6) return `Camera speed ${speed}/10: controlled cinematic movement such as tracking, pan, dolly, crane, or orbit, usually one clear move.`;
    if (speed <= 8) return `Camera speed ${speed}/10: energetic camera motion with stronger tracking, orbit, whip pan, rise, reveal, or compound movement.`;
    return `Camera speed ${speed}/10: fast action camera language; use two or more coordinated camera actions in one scene when readable, such as whip pan into fast tracking plus orbit, reveal, pan, tilt, crane, or pullback. Do not end with "then holds", "holds on", "settles into a hold", static hold, or steady hold unless the user explicitly asks for a hold.`;
  }
  if (speed <= 0) return "Character motion speed 0/10: subject stays still or holds a pose; only facial expression or tiny gestures.";
  if (speed <= 3) return `Character motion speed ${speed}/10: subtle body motion such as shifting weight, hand gestures, turning, swaying, reaching, or small steps.`;
  if (speed <= 6) return `Character motion speed ${speed}/10: active body performance; walking, dancing, interacting with objects, using the set, expressive arms and torso.`;
  if (speed <= 8) return `Character motion speed ${speed}/10: energetic character action; running, dancing hard, climbing, struggling, spinning, crossing the space, or forceful environmental interaction.`;
  return `Character motion speed ${speed}/10: fast action character movement; require clear full-body action such as sprinting, explosive dance, striding, sharp turns, crossing the space, chase/action beats, rapid direction changes, forceful gestures, or intense physical set interaction when it fits the scene. Avoid only poised, still, standing, subtle, quiet, steady, or restrained body language.`;
}

export function storyboardCameraMotionForSpeed(value, speedValue) {
  let motion = String(value || "").trim();
  const speed = storyboardSpeedValue(speedValue, 4);
  if (!motion || speed < 7) return motion;
  return motion
    .replace(/\bslow cinematic drift\b/gi, "energetic cinematic tracking drift")
    .replace(/\bslow orbit\b/gi, "energetic orbit")
    .replace(/\bslow (left|right) orbit\b/gi, "energetic $1 orbit")
    .replace(/\bslow zoom out\b/gi, "brisk pull-back reveal")
    .replace(/\bslow (left|right|side|lateral) drift\b/gi, "brisk $1 tracking drift")
    .replace(/\bslow (pan|tilt|track|tracking|pull[ -]?back|drift)\b/gi, "brisk $1")
    .replace(/\bgentle lateral drift\b/gi, "energetic lateral tracking")
    .replace(/\bgentle pan reveal\b/gi, "brisk pan reveal")
    .replace(/\bgentle (pan|tilt|orbit|drift|camera movement)\b/gi, "brisk $1")
    .replace(/\bsubtle handheld movement\b/gi, "active handheld tracking")
    .replace(/\bsubtle handheld camera\b/gi, "active handheld camera")
    .replace(/\bsubtle handheld follow\b/gi, "energetic handheld follow")
    .replace(/\bsubtle rack focus\b/gi, "quick rack focus")
    .replace(/\bsubtle energetic orbit\b/gi, "energetic orbit")
    .replace(/\bsubtle settling pause\b/gi, "active reframing beat")
    .replace(/\bsubtle orbit movement\b/gi, "energetic orbit movement")
    .replace(/\b(?:quiet handheld hold|locked-off reaction hold|locked-off shot)\b/gi, "active handheld reaction tracking")
    .replace(/\brestrained pan\b/gi, "brisk pan")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function enforceHighMotionPromptLanguage(prompt, scene = {}, state = {}) {
  let text = String(prompt || "").trim();
  if (!text) return text;
  const cameraSpeed = storyboardSpeedValue(scene.camera_motion_speed ?? scene.cameraMotionSpeed ?? state.cameraMotionSpeed, 4);
  const characterSpeed = storyboardSpeedValue(scene.character_motion_speed ?? scene.characterMotionSpeed ?? state.characterMotionSpeed, 4);
  if (cameraSpeed >= 7) {
    text = storyboardCameraMotionForSpeed(text, cameraSpeed)
      .replace(/\bthen\s+holds?\s+on\b/gi, "then continues moving across")
      .replace(/\bthen\s+holds?\b/gi, "then continues moving")
      .replace(/\bsettles?\s+into\s+a\s+(?:static\s+|steady\s+)?hold\b/gi, "flows into another coordinated camera move")
      .replace(/\b(?:static|steady)\s+hold\b/gi, "continued camera motion")
      .replace(/\bholds?\s+on\s+her\s+steady,\s*powerful\s+gaze\b/gi, "tracks her powerful gaze while the camera keeps moving")
      .replace(/\bholds?\s+on\s+(his|her|their|the)\s+([^,.]+)\b/gi, "keeps moving around $1 $2");
    if (!/\b(?:tracking|orbit|whip pan|pan|tilt|crane|pullback|pull-back|push|dolly|handheld|reveal)\b/i.test(text)) {
      text = text.replace(/\.+\s*$/, "");
      text += ", with energetic camera tracking that keeps moving instead of settling into a static hold.";
    }
  }
  if (characterSpeed >= 4) {
    text = text
      .replace(/\bmoves?\s+with\s+a\s+quiet,\s*poised\s+authority\b/gi, "moves with forceful, physically active authority")
      .replace(/\bmoves?\s+with\s+quiet,\s*poised\s+authority\b/gi, "moves with forceful, physically active authority")
      .replace(/\bquiet,\s*poised\s+authority\b/gi, "forceful, physically active authority")
      .replace(/\bquiet\s+poised\s+authority\b/gi, "forceful physical authority")
      .replace(/\bpoised,\s*unyielding\s+head\s+position\b/gi, "forward-driving head posture with sharp turns")
      .replace(/\bpoised\s+posture\b/gi, "active, commanding posture")
      .replace(/\bsubtle\s+body\s+motion\b/gi, "clear full-body movement")
      .replace(/\bstands?\s+still\b/gi, "moves through the space");
    if (!/\b(?:walks?|steps?|strides?|runs?|sprints?|dances?|crosses?|lunges?|reaches?|pushes?|pulls?|climbs?|fights?|brushes?|sweeps?|gestures?|interacts?|grabs?|lifts?|paces?)\b/i.test(text)) {
      text = text.replace(/\.+\s*$/, "");
      text += ", while the subject performs a clear physical action with the body, hands, or surrounding set instead of relying on facial movement alone.";
    }
  }
  return text.replace(/\s{2,}/g, " ").trim();
}

function mergeStoryLayers(primary = {}, fallback = {}) {
  const primaryLayer = normalizeStoryLayer(primary);
  const fallbackLayer = normalizeStoryLayer(fallback);
  return normalizeStoryLayer({
    enabled: primaryLayer.enabled !== false,
    overall_story_idea: primaryLayer.overall_story_idea || fallbackLayer.overall_story_idea,
    user_story_arc: primaryLayer.user_story_arc || fallbackLayer.user_story_arc,
    song_story_brief: primaryLayer.song_story_brief || fallbackLayer.song_story_brief,
  });
}

export function slimStoryboardForRequest(state) {
  return {
    mode: state.mode,
    project_video_engine: normalizeStoryboardProjectVideoEngine(state.projectVideoEngine),
    performance_mode: normalizeStoryboardPerformanceMode(state.performanceMode || state.performance_mode),
    short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
    camera_flow: state.cameraFlow || "balanced",
    image_shot_flow: state.imageShotFlow || "intimate",
    image_aesthetic: state.imageAesthetic || "",
    video_style: state.videoStyle || "",
    video_style_custom: state.videoStyleCustom || "",
    temporal_world_effect: state.temporalWorldEffect || "",
    temporal_world_effect_custom: state.temporalWorldEffectCustom || "",
    temporal_allow_background_extras: state.temporalAllowBackgroundExtras !== false,
    temporal_background_intensity: storyboardTemporalIntensity(state.temporalBackgroundIntensity),
    temporal_environment_time_passage: state.temporalEnvironmentTimePassage !== false,
    temporal_protected_characters: storyboardTemporalProtectedMode(state.temporalProtectedCharacters),
    temporal_protected_custom: state.temporalProtectedCustom || "",
    global_consistency_phrase: state.globalConsistencyPhrase || "",
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    minimax_h3_cut_frequency: storyboardCutFrequencyValue(state.cutFrequency),
    performance_style_default: state.performanceStyle || "",
    facial_performance_default: state.facialPerformance || "",
    facial_performance_custom_default: state.facialPerformanceCustom || "",
    story_layer: normalizeStoryLayer(state.storyLayer),
    script_import: normalizeStoryboardScriptImportState(state.scriptImport),
    reference_builder: {
      subjects: (state.referenceBuilder?.subjects || []).map(slimReferenceForRequest).filter(Boolean),
      locations: (state.referenceBuilder?.locations || []).map(slimReferenceForRequest).filter(Boolean),
    },
    motion_defaults: {
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
    },
    scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
  };
}
