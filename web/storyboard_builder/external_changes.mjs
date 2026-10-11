// Shows Agent API / MCP scene-card edits in the open Storyboard.
//
// The Video Builder merges an API edit into its timeline first (music_video_builder/external_changes.mjs)
// and then sends "vrgdg:storyboard-external-change" with the scene fields that changed. Each changed card
// field is merged against the value this window last loaded or saved, so unsaved edits are kept: a card
// edited here and through the API keeps this window's value and the user is told. A card whose editor is
// open is left alone until the editor closes. Values come from the saved storyboard.json when the API wrote
// the card there, otherwise from the live timeline scene.

export const STORYBOARD_EXTERNAL_CHANGE_EVENT = "vrgdg:storyboard-external-change";

// Timeline segment keys and the Storyboard card keys they fill (the reverse of applyStoryboardPrompts).
export const SEGMENT_TO_CARD_KEYS = {
  label: "label", lyric_text: "lyrics", lyric_section: "lyric_section", story_beat: "story_beat",
  timeline_note: "timeline_note", i2v_notes: "motion_summary", video_notes: "motion_summary", notes: "notes",
  shot_type: "shot_type", camera_motion: "camera_motion", character_motion: "character_motion",
  performance_mode: "performance_mode", performance_style: "performance_style",
  facial_performance: "facial_performance", facial_performance_custom: "facial_performance_custom",
  emotion_expression_tags: "emotion_expression_tags",
  include_microphone: "include_microphone", audio_direction: "audio_direction", continuity: "continuity",
  flf_start_state: "flf_start_state", flf_transformation: "flf_transformation", flf_end_state: "flf_end_state",
  flf_carry_forward: "flf_carry_forward", minimax_h3_video_style: "video_style",
  minimax_h3_video_style_custom: "video_style_custom", temporal_world_effect_override: "temporal_world_effect_override",
  temporal_world_effect_custom: "temporal_world_effect_custom", no_character_present: "no_character_present",
  lyric_no_lip_sync: "lyric_no_lip_sync", lyric_instrumental: "lyric_instrumental", lyric_singers: "lyric_singers",
  lyric_cue_map: "lyric_cue_map", lyric_shot_word_timing_enabled: "lyric_shot_word_timing_enabled",
  lyric_performance_mode: "lyric_performance_mode", minimax_speaker_assignments: "speaker_assignments",
  video_prompt_type: "video_prompt_type", t2i_prompt: "image_prompt", flux_prompt: "image_prompt",
  nb_prompt: "image_prompt", flow_gpt_prompt: "image_prompt", minimax_h3_prompt: "video_prompt",
  i2v_prompt: "video_prompt", minimax_h3_prompt_origin: "video_prompt_origin", i2v_prompt_origin: "video_prompt_origin",
  minimax_h3_pass2_prompt: "minimax_h3_pass2_prompt",
};
const REFERENCE_CARD_KEYS = ["subject_refs", "subjects", "location_ref", "setting"];

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function sameValue(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

// Card keys to refresh per scene id, and which of them the API wrote into storyboard.json.
export function changedCardKeys(changes = []) {
  const byScene = {};
  for (const change of changes) {
    if (String(change?.kind || "") !== "scene_fields") continue;
    for (const [sceneId, scene] of Object.entries(change.scenes || {})) {
      const entry = byScene[sceneId] || (byScene[sceneId] = { keys: new Set(), fromFile: new Set() });
      for (const key of scene.segment || []) if (SEGMENT_TO_CARD_KEYS[key]) entry.keys.add(SEGMENT_TO_CARD_KEYS[key]);
      if ((scene.references || []).length) REFERENCE_CARD_KEYS.forEach((key) => entry.keys.add(key));
      for (const key of scene.card || []) {
        entry.keys.add(key);
        entry.fromFile.add(key);
      }
    }
  }
  return byScene;
}

// Merge changed card values into the open cards. ``saved`` maps scene id -> the card as last loaded/saved.
export function mergeCardChanges({ scenes = [], saved, byScene, fileCards = new Map(), liveCards = new Map(), editingSceneId = "" }) {
  const result = { applied: [], conflicts: [], deferred: [] };
  for (const scene of scenes) {
    const entry = byScene[String(scene?.id)];
    if (!entry) continue;
    const base = saved.get(String(scene.id)) || {};
    for (const key of entry.keys) {
      const source = entry.fromFile.has(key) ? fileCards.get(String(scene.id)) : liveCards.get(String(scene.id));
      if (!source || !Object.prototype.hasOwnProperty.call(source, key)) continue;
      const fresh = source[key];
      if (sameValue(scene[key], fresh)) {
        base[key] = clone(fresh);
        continue;
      }
      if (String(scene.id) === String(editingSceneId)) {
        result.deferred.push({ sceneId: String(scene.id), key });
        continue;  // the open editor will save its own fields; tell the user instead of changing it underneath
      }
      if (!sameValue(scene[key], base[key])) {
        result.conflicts.push({ sceneId: String(scene.id), key });
      } else {
        scene[key] = clone(fresh);
        result.applied.push({ sceneId: String(scene.id), key });
      }
      base[key] = clone(fresh);
    }
    saved.set(String(scene.id), base);
  }
  return result;
}

export function createStoryboardExternalSync({
  state, postJson, normalizeScene, scenesFromBuilderPayload, getBuilderScenes, renderTable, createToast, isOpen,
}) {
  let saved = new Map();
  // Record the cards as saved: all of them, or one card after its edit was applied to the timeline.
  const markClean = (sceneId = "") => {
    (state.scenes || []).forEach((scene, index) => {
      if (!sceneId || String(scene.id) === String(sceneId)) saved.set(String(scene.id), clone(normalizeScene(scene, index)));
    });
  };

  async function apply(detail = {}) {
    const changes = Array.isArray(detail.changes) ? detail.changes : [];
    const byScene = changedCardKeys(changes);
    const fileWrites = changes.filter((change) => Object.values(change?.scenes || {}).some((scene) => (scene.card || []).length)).length;
    if (!Object.keys(byScene).length) {
      if (changes.some((change) => ["project", "storyboard"].includes(String(change?.kind || "")))) {
        createToast("The project was changed through the Agent API / MCP. Reopen the Storyboard to see story and storyboard-wide changes.");
      }
      return;
    }
    let fileCards = new Map();
    let fileRevision = null;
    if (fileWrites && state.projectFolder) {
      const data = await postJson("/vrgdg/storyboard/load", { project_folder: state.projectFolder });
      const board = data.storyboard || {};
      fileRevision = Number(board.revision || 0);
      fileCards = new Map((board.scenes || []).map((scene, index) => [String(scene.id), normalizeScene(scene, index)]));
    }
    const builderScenes = typeof getBuilderScenes === "function" ? getBuilderScenes() : [];
    const liveCards = new Map(scenesFromBuilderPayload({ scenes: builderScenes }).map((scene) => [String(scene.id), scene]));
    if (!isOpen()) return;
    const editor = state.openSceneEditor;
    const editingSceneId = editor?.element?.isConnected ? editor.id : "";
    const result = mergeCardChanges({ scenes: state.scenes, saved, byScene, fileCards, liveCards, editingSceneId });
    // The file holds only this window's last save plus the API card edits just merged.
    if (fileRevision !== null && Number(state.storyboardRevision) + fileWrites === fileRevision) state.storyboardRevision = fileRevision;
    renderTable();
    const label = (item) => {
      const scene = state.scenes.find((candidate) => String(candidate.id) === item.sceneId);
      return `${scene?.label || item.sceneId}: ${item.key}`;
    };
    if (result.conflicts.length || result.deferred.length) {
      createToast([
        result.conflicts.length ? `Agent API / MCP edits arrived for card fields you also changed here; your unsaved values were kept:\n${result.conflicts.map(label).join("\n")}` : "",
        result.deferred.length ? `The open scene editor was not changed. Cancel and reopen it to see the API values, or save to keep yours:\n${result.deferred.map(label).join("\n")}` : "",
      ].filter(Boolean).join("\n\n"), true);
    } else if (result.applied.length) {
      createToast(`Storyboard updated from the Agent API / MCP: ${result.applied.length} card field${result.applied.length === 1 ? "" : "s"}.`);
    }
  }

  return { apply, markClean };
}
