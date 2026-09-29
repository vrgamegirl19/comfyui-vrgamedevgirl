import { normalizeMiniMaxSpeakerAssignments } from "./minimax_h3.mjs";
import { flattenLyricForPrompt } from "./prompt_text.mjs";

export function syncMiniMaxSpeakerAssignmentLegacyFields(segment) {
  if (!segment) return;
  segment.minimax_speaker_assignments = normalizeMiniMaxSpeakerAssignments(segment.minimax_speaker_assignments);
  const filled = segment.minimax_speaker_assignments.filter((cue) => cue.text);
  // An empty Storyboard speaker plan means no dialogue override was supplied.
  // It must not erase lyrics/transcription already stored on the timeline.
  if (filled.length) {
    segment.lyric_text = filled.map((cue) => cue.text).join("\n");
    segment.lyric_singers = Array.from(new Set(filled.map((cue) => cue.speaker_name).filter(Boolean)));
  }
}

export function syncLyricTextFromCueMap(segment) {
  if (!segment || !Array.isArray(segment.lyric_cue_map) || !segment.lyric_cue_map.length) return;
  const joined = segment.lyric_cue_map.map((cue) => flattenLyricForPrompt(cue?.text)).filter(Boolean).join(" ");
  if (joined) segment.lyric_text = joined;
}
