export const issuedSegmentIds = new Set();
let fallbackSegmentIdSequence = 0;

export function createUniqueSegmentId() {
  let id = "";
  do {
    if (globalThis.crypto?.randomUUID) {
      id = `seg_${globalThis.crypto.randomUUID()}`;
    } else {
      fallbackSegmentIdSequence += 1;
      id = `seg_${Date.now()}_${fallbackSegmentIdSequence}_${Math.floor(Math.random() * 0x100000000).toString(16)}`;
    }
  } while (issuedSegmentIds.has(id));
  issuedSegmentIds.add(id);
  return id;
}

export function normalizeVideoPromptOrigin(value) {
  return String(value || "").trim().toLowerCase() === "gemma" ? "gemma" : "manual";
}

export function newSegment(start = 0, end = 4) {
  return {
    id: createUniqueSegmentId(),
    track: "base",
    start,
    end,
    label: "New scene",
    notes: "",
    timeline_note: "",
    lyric_text: "",
    lyric_section: "",
    story_beat: "",
    flf_endpoint_mode: "auto",
    flf_custom_end_direction: "",
    flf_motion_plan: "",
    flf_end_frame_prompt: "",
    flf_end_frame_stale: false,
    flf_final_prompt_ready: false,
    lyric_singers: [],
    minimax_speaker_assignments: [],
    facial_performance: "",
    facial_performance_custom: "",
    emotion_expression_tags: "",
    no_character_present: false,
    i2v_notes: "",
    t2i_prompt: "",
    enhance_notes: "",
    enhance_prompt: "",
    i2v_prompt: "",
    i2v_prompt_origin: "manual",
    minimax_h3_mode: "text_to_video",
    use_scene_minimax_h3_settings: false,
    minimax_h3_settings: null,
    minimax_h3_prompt: "",
    minimax_h3_pass2_prompt: "",
    minimax_h3_prompt_origin: "manual",
    minimax_h3_reference_keys: null,
    minimax_h3_use_scene_image_as_start_frame: false,
    minimax_h3_scene_image_use: "off",
    minimax_h3_start_frame_character_influence: "full_character",
    minimax_h3_video_references: [],
    ref_image_path: "",
    use_vision_reference: false,
    use_i2v_vision_reference: true,
    custom_image_path: "",
    custom_image_data: "",
    custom_image_name: "",
    image: null,
    image_history: [],
    image_history_index: -1,
    preview_mode: "image",
    video_path: "",
    video_thumbnail_path: "",
    video_history: [],
    video_thumbnail_history: [],
    video_backup_paths: [],
    video_backup_thumbnail_paths: [],
    video_history_index: -1,
    video_output: null,
    video_status: "none",
    custom_audio_path: "",
    custom_audio_name: "",
    custom_audio_duration: 0,
    custom_audio_full_duration: 0,
    custom_audio_timeline_start: start,
    custom_audio_source_start: 0,
    custom_audio_peaks: [],
    custom_audio_beats: [],
    overlay_slot_number: 0,
    flux_image_ingredients: [],
    flux_notes: "",
    flux_prompt: "",
    nb_notes: "",
    nb_prompt: "",
    use_scene_zimage_settings: false,
    zimage_settings: null,
    use_scene_ernie_image_settings: false,
    ernie_image_settings: null,
    use_scene_krea2_2pass_settings: false,
    krea2_2pass_settings: null,
    use_scene_flux_klein_settings: false,
    flux_klein_settings: null,
    use_scene_nb_image_settings: false,
    nb_image_settings: null,
    use_scene_i2v_video_settings: false,
    i2v_video_settings: null,
    source: "manual",
  };
}

export function newTimelineMarker(start = 0, end = null) {
  const now = Date.now();
  const cleanStart = Math.max(0, Number(start || 0));
  const cleanEnd = Number.isFinite(Number(end)) && Number(end) > cleanStart ? Number(end) : null;
  return {
    id: `mark_${now}_${Math.floor(Math.random() * 10000)}`,
    start: cleanStart,
    end: cleanEnd,
    type: "note",
    label: "Timeline note",
    note: "",
  };
}

function scenePathKey(value) {
  return String(value).replace(/\\/g, "/").toLowerCase();
}

// Point every stored path at the file's renumbered name; exact renames win over their parent folder's rename.
export function rewriteRenamedScenePaths(node, renamed) {
  const rename = (value) => {
    const key = scenePathKey(value);
    const exact = renamed.find(([from]) => scenePathKey(from) === key);
    if (exact) return exact[1];
    const parent = renamed.find(([from]) => key.startsWith(`${scenePathKey(from)}/`));
    return parent ? parent[1] + value.slice(parent[0].length) : value;
  };
  for (const [key, value] of Object.entries(node)) {
    if (typeof value === "string") node[key] = rename(value);
    else if (value && typeof value === "object") rewriteRenamedScenePaths(value, renamed);
  }
}

export function sortSegments(segments) {
  segments.sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
}

// Keep generic base-scene labels ("", "Scene 7", "7. Rooftop chorus") in step with timeline position; custom names stay.
export function renumberGenericBaseSceneLabels(segments) {
  sortSegments(segments);
  let changed = false;
  segments.forEach((segment, index) => {
    const current = String(segment.label || "").trim();
    const numberedDescription = current.match(/^\d+\.\s*(.+)$/);
    const nextLabel = numberedDescription
      ? `${index + 1}. ${numberedDescription[1]}`
      : `Scene ${index + 1}`;
    if (!current || /^scene(?:\s+\d+(?:\.\d+)?)?$/i.test(current) || numberedDescription) {
      if (segment.label !== nextLabel) {
        segment.label = nextLabel;
        changed = true;
      }
    }
  });
  return changed;
}
