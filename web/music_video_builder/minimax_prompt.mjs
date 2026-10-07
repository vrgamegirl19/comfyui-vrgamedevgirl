import {
  normalizeStoryboardCustomCameraFlowSequence,
  STORYBOARD_CAMERA_FLOW_PRESETS,
} from "../storyboard_builder/shot_presets.mjs";
import { storyboardFxContract } from "../storyboard_builder/video_style.mjs";
import { normalizeVideoType, toast } from "./controls.mjs";
import { miniMaxEffectiveCueEnd, miniMaxH3CueTimingText, miniMaxH3PerformerLabel } from "./lyric_cues.mjs";
import {
  MINIMAX_H3_VIDEO_REFERENCE_PURPOSES,
  miniMaxH3ContinuationStartSeconds,
  miniMaxH3ModeLabel,
  normalizeMiniMaxH3ContinuityMode,
  normalizeMiniMaxH3Mode,
  normalizeMiniMaxH3Pipeline,
  normalizeMiniMaxH3VideoPurpose,
  normalizeMiniMaxH3Voice,
  normalizeMiniMaxSpeakerAssignments,
} from "./minimax_h3.mjs";
import { normalizeBuilderStoryboardDefaults } from "./model_settings.mjs";
import {
  escapeRegExp,
  flattenLyricForPrompt,
  isInstrumentalLyricText,
  segmentUsesNoLipSyncPerformance,
} from "./prompt_text.mjs";
import { castWallText, stripCastLeaks } from "./cast_guard.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { attachRefmodLabels } from "./refmod_labels.mjs";
import { mediaPathKey } from "./timeline_state.mjs";

export function miniMaxDialogueAssignmentsForSegment(segment) {
  return normalizeMiniMaxSpeakerAssignments(
    segment?.minimax_speaker_assignments || segment?.speaker_assignments || segment?.dialogue_cues || [],
  ).filter((cue) => cue.text);
}

export function miniMaxDialogueOrderText(segment) {
  const cues = miniMaxDialogueAssignmentsForSegment(segment);
  if (!cues.length) return "";
  return cues.map((cue, index) => {
    const timing = miniMaxH3CueTimingText(cue, segment, cues, index);
    return `${index + 1}. ${timing}${cue.speaker_name || "The assigned speaker"} says exactly: “${cue.text}”`;
  }).join("\n");
}

function stripMiniMaxH3ManagedPromptBlock(prompt, headingPattern) {
  const nextSection = String.raw`(?=\n+(?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|Audio(?:\s+1)?\s*:|Native\s+audio\s*:|Continuity\s*:|MINIMAX NATIVE VOICE IDENTITY — MANDATORY AND VERBATIM:|REFERENCE SUBJECT COUNT — MANDATORY:|VISUAL-ONLY B-ROLL / NO-LIP-SYNC SAFETY — MANDATORY AND FINAL:|PREVIOUS-SCENE (?:SPATIAL|EXACT FRAME) CONTINUITY — MANDATORY:)|$)`;
  return String(prompt || "").replace(
    new RegExp(`\\n*${headingPattern}:[\\s\\S]*?${nextSection}`, "g"),
    "",
  ).trim();
}

function stripMiniMaxH3VisualOnlyTimelineVocalDirections(prompt) {
  const positiveVocal = /\b(?:sing(?:s|ing)?|sang|sung|rap(?:s|ping)?|lip[ -]?sync(?:s|ing)?|speak(?:s|ing)?|say(?:s|ing)?|said|mouth(?:s|ed|ing)?|whisper(?:s|ing)?|perform(?:s|ing)?\s+(?:the\s+)?(?:saved\s+|exact\s+)?(?:lyric|dialogue|words?))\b/i;
  const negativeSafety = /\b(?:no|not|never|without|does\s+not|do\s+not|must\s+not|cannot|can['’]t|don['’]t)\b/i;
  const timestampPattern = /(\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]\s*\n)([\s\S]*?)(?=\n\n(?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]|Audio\s*:|Continuity\s*:)|$)/gi;
  return String(prompt || "").replace(timestampPattern, (match, header, body) => {
    const kept = String(body || "")
      .split(/(?<=[.!?])\s+|\n+/)
      .map((part) => part.trim())
      .filter((part) => part && !(positiveVocal.test(part) && !negativeSafety.test(part)));
    if (!kept.length) kept.push("Continue the requested visual action, camera movement, and environmental motion naturally.");
    kept.push("All visible subjects remain silent and do not sing, speak, rap, mouth words, or lip-sync; every mouth stays naturally relaxed or closed.");
    return `${header}${kept.join(" ")}`;
  });
}

function miniMaxH3ContinuityPromptBlock(segment, continuityInput, imageNumber) {
  const continuityMode = normalizeMiniMaxH3ContinuityMode(continuityInput?.continuityMode);
  const number = Math.max(1, Math.trunc(Number(imageNumber || 1)));
  if (continuityMode === "off" || !continuityInput?.framePath) return "";
  const sharedIdentityRule = `Image ${number} contains the same named subjects already assigned by the character references; it does not introduce additional people. Never duplicate or clone a person because they appear in both a character sheet and Image ${number}.`;
  if (continuityMode === "exact_start_frame") {
    return [
      "PREVIOUS-SCENE EXACT FRAME CONTINUITY — MANDATORY:",
      `Image ${number} is the actual final frame of the immediately previous rendered scene and is the sole exact opening frame for this scene. Begin on Image ${number} exactly, preserving its composition, camera angle, subject positions, poses, orientations, screen direction, distances, props, environment layout, lighting, and visual state.`,
      "Continue naturally from that exact frame without resetting, teleporting, mirroring, swapping sides, changing wardrobe, or rearranging the set. Character and location reference images remain identity and environment support only.",
      `Image ${number} is only the opening frame, never an ending target. Progress the requested action and camera direction away from that opening state and finish on a newly advanced frame; do not return to the same composition at the end.`,
      sharedIdentityRule,
    ].join("\n");
  }
  return [
    "PREVIOUS-SCENE SPATIAL CONTINUITY — MANDATORY:",
    `Image ${number} is a blocking-and-layout reference extracted from the immediately previous rendered scene. It is not a start frame, opening-frame anchor, end frame, or destination target. Use it only to understand each subject's relative physical position, orientation, distance from the others, screen-direction relationship, nearby props, and the environment's physical layout.`,
    `The literal first generated frame must be a visibly different composition from Image ${number}, using the new shot type, angle, crop, distance, height, and lens requested by this scene. Do not reproduce Image ${number}'s camera view, framing, or pixel composition at the beginning or at any later point in the clip.`,
    "Preserve the established blocking when photographing it from the new viewpoint. Do not reset, teleport, mirror, swap sides, reverse screen direction, rearrange the set, or move a subject behind/in front of another unless the requested action visibly causes that movement.",
    `Progress naturally from the new first frame and finish on a newly advanced composition. Do not recreate, return to, or converge on Image ${number}. Do not invent a distant establishing shot or an automatic push-in; use only the scene's requested camera motion.`,
    sharedIdentityRule,
  ].join("\n");
}

function applyMiniMaxH3ContinuityPromptBlock(prompt, segment, continuityInput, imageNumber) {
  const clean = stripMiniMaxH3ManagedPromptBlock(
    prompt,
    "PREVIOUS-SCENE (?:SPATIAL|EXACT FRAME) CONTINUITY — MANDATORY",
  );
  const block = miniMaxH3ContinuityPromptBlock(segment, continuityInput, imageNumber);
  return block ? `${clean}\n\n${block}`.trim() : clean;
}

export function miniMaxH3Timecode(seconds = 0) {
  const total = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  const secs = total - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(3).padStart(6, "0")}`;
}

const MINIMAX_H3_LIFE_MOVEMENTS_SUBTLE = [
  "breath catching before a held note", "weight shifting from one foot to the other", "fingers tightening then loosening on an object",
  "a glance away and back", "shoulders dropping after a held pose", "a slow blink", "a head tilt on a sustained note",
  "a hand brushing hair or collar", "a small smile that fades", "jaw tension then release", "a lean toward or away from the camera",
  "an unfinished gesture",
];
const MINIMAX_H3_LIFE_MOVEMENTS_ACTIVE = [
  "a snap turn that whips the hair", "a quick step then a hard stop", "shoulders driving on the downbeat", "a spin that ends in a pose",
  "a fast reach and grab", "a head throw back on the high note", "a jump landing with bent knees", "a quick glance at the other person mid-stride",
  "hands slicing the air with the rhythm", "a lean into the camera then a push away", "a stumble that turns into a dance step",
  "a laugh breaking out mid-motion",
];

// Picks six life-like movements scaled to the character motion speed; the segment id varies the pick between scenes.
function miniMaxH3LifeMovementBank(characterMotionSpeed, seed = "") {
  const bank = Number(characterMotionSpeed) >= 7
    ? MINIMAX_H3_LIFE_MOVEMENTS_ACTIVE
    : Number(characterMotionSpeed) >= 4 ? [...MINIMAX_H3_LIFE_MOVEMENTS_SUBTLE.slice(0, 6), ...MINIMAX_H3_LIFE_MOVEMENTS_ACTIVE.slice(0, 6)] : MINIMAX_H3_LIFE_MOVEMENTS_SUBTLE;
  const offset = Array.from(String(seed)).reduce((total, char) => total + char.charCodeAt(0), 0) % bank.length;
  return Array.from({ length: 6 }, (_, index) => bank[(offset + index) % bank.length]);
}

// Energy comes from the scene builder sliders, never from the scene length.
function miniMaxH3MotionEnergyText(cameraMotionSpeed, characterMotionSpeed) {
  const camera = Number.isFinite(Number(cameraMotionSpeed)) ? Number(cameraMotionSpeed) : 4;
  const character = Number.isFinite(Number(characterMotionSpeed)) ? Number(characterMotionSpeed) : 4;
  const cameraText = camera >= 7
    ? "The camera is fast and energetic: whips, orbits, push-ins, and quick tracking."
    : camera >= 4 ? "The camera moves at a steady pace with a clear direction." : "The camera moves slowly, or holds a locked frame.";
  const characterText = character >= 7
    ? "The performers are high energy: pop, fast motion, jumps, spins, runs, dance hits, and quick reactions."
    : character >= 4 ? "The performers use steady physical action: walking, turning, reaching, and set interaction." : "The performers use small movements and held poses with life detail.";
  return `${cameraText} ${characterText}`;
}

function miniMaxH3OfficialShotScheduleLines(cutPlan = {}) {
  const cuts = Array.isArray(cutPlan.cut_times_seconds) ? cutPlan.cut_times_seconds : [];
  const lines = ["[Shot 1] starts at 00:00.000 and has no timestamp after the label."];
  cuts.forEach((time, index) => {
    lines.push(`[Shot ${index + 2}] At ${miniMaxH3Timecode(time)}, begin a new continuity-preserving cut.`);
  });
  return lines;
}

function miniMaxH3OfficialShotPlan(cutPlan = {}) {
  const cuts = Array.isArray(cutPlan.cut_times_seconds) ? cutPlan.cut_times_seconds : [];
  return [
    { number: 1, time: 0, timecode: "00:00.000" },
    ...cuts.map((time, index) => ({
      number: index + 2,
      time: Number(time || 0),
      timecode: miniMaxH3Timecode(time),
    })),
  ];
}

export function miniMaxH3OfficialCutPlanInstruction(cutPlan = {}) {
  const duration = Number(cutPlan.exact_duration_seconds || 0);
  const exactDuration = Number.isFinite(duration) ? Number(duration.toFixed(3)) : 0;
  const cuts = Array.isArray(cutPlan.cut_times_seconds) ? cutPlan.cut_times_seconds : [];
  if (cutPlan.cue_driven) {
    const timingText = cuts.map((time) => miniMaxH3Timecode(time)).join(", ");
    return `EDITING / CUT PLAN — MANDATORY: The timed singer/lyric cue map controls this exact ${exactDuration}-second segment. Create exactly ${cuts.length + 1} shot${cuts.length ? "s" : ""}${cuts.length ? ` with hard cuts at ${timingText}` : ""}. Each shot corresponds to one cue row in order: rows without assigned words use only their visual action and camera direction, while lyric rows use the assigned subject and exact cue. The builder will write [Shot 1] and every later [Shot N] At MM:SS.mmm label. Return only the creative description for each shot. Do not omit, merge, add, reorder, or shift cue shots.`;
  }
  if (!cuts.length) {
    return `EDITING / CUT PLAN — MANDATORY: Use one smooth, continuous, uninterrupted shot for the full ${exactDuration}-second segment. Output only [Shot 1]. Use no additional shot label, hard cut, angle reset, montage, dissolve, scene change, or transition. Camera and character movement may develop inside the same continuous take.`;
  }
  const timingText = cuts.map((time) => miniMaxH3Timecode(time)).join(", ");
  return `EDITING / CUT PLAN — MANDATORY: Cut frequency ${cutPlan.frequency}/10 for this exact ${exactDuration}-second segment requires exactly ${cuts.length} hard cut${cuts.length === 1 ? "" : "s"} at ${timingText}, creating ${cuts.length + 1} coherent shots. The builder will write [Shot 1] and every later [Shot N] At MM:SS.mmm label. Return only the creative description for each shot. Each later description should stage a continuity-preserving camera/shot cut. Do not omit, merge, add, or shift a scheduled cut. Do not add extra shots, montage beats, dissolves, scene changes, or transitions outside this schedule.`;
}

function miniMaxH3VideoAssignmentLines(segment) {
  return (Array.isArray(segment?.minimax_h3_video_references) ? segment.minimax_h3_video_references : [])
    .filter((item) => String(item?.path || "").trim())
    .slice(0, 3)
    .map((item, index) => {
      const purpose = normalizeMiniMaxH3VideoPurpose(item.purpose);
      const purposeLabel = MINIMAX_H3_VIDEO_REFERENCE_PURPOSES.find((option) => option.value === purpose)?.label || "Continuation / Extension";
      const timing = Number(item.start_seconds || 0) > 0 || Number(item.duration || 0) > 0
        ? ` Use from ${Math.max(0, Number(item.start_seconds || 0))}s${Number(item.duration || 0) > 0 ? ` for ${Math.max(0, Number(item.duration || 0))}s` : " onward"}.`
        : "";
      const audio = item.use_audio ? " Its embedded audio is also supplied as a paired reference." : " Do not use its embedded audio.";
      return `<Video ${index + 1}>: ${purposeLabel}.${timing}${audio}`;
    });
}

function stripMiniMaxH3FixedSectionsFromCreative(prompt) {
  let text = String(prompt || "").trim();
  text = text.replace(/^\s*Generate\s+(?:an?|the)\s+[\s\S]*?(?=\n\s*(?:Image\s+\d+\s*:|\[\s*\d|CUT TO:|Audio(?:\s+1)?\s*:|Native audio\s*:|Continuity\s*:)|$)/i, "").trim();
  text = text.replace(/(?:^|\n)\s*(?:Ordered [^\n]*assignments?[^\n]*:\s*)?(?:Image\s+\d+(?:\s*\([^)]*\))?\s*:[^\n]*(?:\n|$))+/gi, "\n").trim();
  text = text.replace(/(?:^|\n)\s*(?:Ordered video assignments?[^\n]*:\s*)?(?:Video\s+\d+\s*:[^\n]*(?:\n|$))+/gi, "\n").trim();
  text = text.replace(/(?:^|\n)\s*Audio\s+1\s*:[\s\S]*?(?=\n\s*(?:\[\s*\d|CUT TO:|Audio\s*:|Continuity\s*:|MINIMAX|REFERENCE SUBJECT COUNT|VISUAL-ONLY|$))/gi, "\n").trim();
  text = text.replace(/(?:^|\n)\s*Audio\s*:[\s\S]*?(?=\n\s*(?:Continuity\s*:|MINIMAX|REFERENCE SUBJECT COUNT|VISUAL-ONLY|$))/gi, "\n").trim();
  text = text.replace(/(?:^|\n)\s*Native audio\s*:[\s\S]*?(?=\n\s*(?:\[\s*\d|CUT TO:|Audio\s*:|Continuity\s*:|MINIMAX|REFERENCE SUBJECT COUNT|VISUAL-ONLY|$))/gi, "\n").trim();
  text = text.replace(/(?:^|\n)\s*Continuity\s*:[\s\S]*?(?=\n\s*(?:MINIMAX|REFERENCE SUBJECT COUNT|VISUAL-ONLY|$))/gi, "\n").trim();
  text = stripMiniMaxH3ManagedPromptBlock(text, "MINIMAX NATIVE VOICE IDENTITY — MANDATORY AND VERBATIM");
  text = stripMiniMaxH3ManagedPromptBlock(text, "REFERENCE SUBJECT COUNT — MANDATORY");
  text = stripMiniMaxH3ManagedPromptBlock(text, "VISUAL-ONLY B-ROLL / NO-LIP-SYNC SAFETY — MANDATORY AND FINAL");
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

function miniMaxH3InjectMissingExtraLabels(descriptions, extras = []) {
  const shots = descriptions.map((description) => String(description || "").trim());
  const combined = shots.join("\n");
  const missing = extras.filter((item) => !combined.includes(item.label));
  const mainContactExtras = extras.filter((item) => ["dancing_with", "direct"].includes(String(item.interaction || "")));
  if ((!missing.length && !mainContactExtras.length) || !shots.length) return shots;
  const framingScore = (description) => {
    const text = String(description || "").toLowerCase();
    if (/\b(?:extreme close-up|close-up|macro|insert|detail shot|eyes shot|mouth shot)\b/.test(text)) return 0;
    if (/\b(?:wide|full-body|full body|establishing|group|formation|dance floor)\b/.test(text)) return 3;
    if (/\b(?:medium|tracking|two-shot|three-shot)\b/.test(text)) return 2;
    return 1;
  };
  let targetIndex = 0;
  let bestScore = -1;
  shots.forEach((description, index) => {
    const score = framingScore(description);
    if (score > bestScore) {
      bestScore = score;
      targetIndex = index;
    }
  });
  const groups = new Map();
  for (const extra of missing) {
    const interaction = String(extra.interaction || "background");
    if (!groups.has(interaction)) groups.set(interaction, []);
    groups.get(interaction).push(extra.label);
  }
  const clauses = [];
  for (const [interaction, labels] of groups.entries()) {
    if (["dancing_with", "direct"].includes(interaction)) continue;
    const subjects = labels.join(", ");
    const plural = labels.length > 1;
    if (bestScore === 0) {
      clauses.push(`${subjects} ${plural ? "remain" : "remains"} present in the same scene outside this close framing`);
    } else if (interaction === "background_dancing") {
      clauses.push(`${subjects} ${plural ? "perform" : "performs"} backup choreography in the background`);
    } else if (interaction === "alongside") {
      clauses.push(`${subjects} ${plural ? "dance" : "dances"} alongside <Subject 1> with independent choreography and comfortable personal spacing`);
    } else {
      clauses.push(`${subjects} ${plural ? "remain" : "remains"} visibly present in the background`);
    }
  }
  for (const extra of mainContactExtras) {
    const ensemble = Number(extra.count || 1) > 1;
    if (extra.interaction === "dancing_with") {
      clauses.push(`${extra.label} performs sensual adult nightclub dancing specifically with <Subject 1>${ensemble ? ", with its members alternating close hip-led grinding, dancing behind her with hands at her hips or waist, and face-to-face body-close movement" : ", using close hip-led grinding, dancing behind her with hands at her hips or waist, or face-to-face body-close movement"}; <Subject 1> remains the exclusive dance partner`);
    } else {
      clauses.push(`${ensemble ? `each member of ${extra.label} takes a turn making` : `${extra.label} makes`} clearly visible physical contact specifically with <Subject 1>; <Subject 1> remains the exclusive interaction partner`);
    }
  }
  const shotSentence = shots[targetIndex].replace(/\s+$/g, "").replace(/[.!?…]+$/g, "");
  const normalizedClauses = clauses.map((clause) => String(clause || "").replace(/^([a-z])/, (letter) => letter.toUpperCase()));
  const injected = `${shotSentence}. ${normalizedClauses.join("; ")}.`.trim();
  shots[targetIndex] = normalizeMiniMaxH3ShotDescription(injected);
  return shots;
}

function stripMiniMaxH3NegativePromptSentences(description) {
  const text = String(description || "").trim();
  if (!text) return "";
  const negativeWording = /\b(?:do\s+not|don['’]t|never|without|avoid|must\s+not|cannot|can['’]t|not|no)\b/i;
  const dialogueTags = [];
  // Quoted lyric words are sung text, not instructions: "don't" or "no" inside quotes is never a reason to drop a sentence.
  const maskedText = text.replace(/<d>[\s\S]*?<\/d>|["“][^"”]*["”]/gi, (tag) => {
    const token = `VRGDGDIALOGUE${dialogueTags.length}TOKEN`;
    dialogueTags.push(tag);
    return token;
  });
  const restoreDialogue = (value) => String(value || "").replace(/VRGDGDIALOGUE(\d+)TOKEN/g, (_match, index) => dialogueTags[Number(index)] || "");
  // A decimal point (1.5 seconds) is not a sentence end.
  const sentences = (maskedText.replace(/(\d)\.(?=\d)/g, "$1VRGDGDECIMALTOKEN").match(/[^.!?…]+[.!?…]+|[^.!?…]+$/g) || [maskedText])
    .map((sentence) => sentence.replace(/VRGDGDECIMALTOKEN/g, "."));
  const kept = [];
  for (const sentence of sentences) {
    const sentenceTokens = sentence.match(/VRGDGDIALOGUE\d+TOKEN/g) || [];
    const visualProse = sentence.replace(/VRGDGDIALOGUE\d+TOKEN/g, " ");
    if (!negativeWording.test(visualProse)) {
      kept.push(restoreDialogue(sentence.trim()));
      continue;
    }
    if (sentenceTokens.length) {
      kept.push(`The assigned performer visibly delivers ${restoreDialogue(sentenceTokens.join(" "))} with natural synchronized mouth, jaw, cheek, and facial movement.`);
    }
  }
  return kept.join(" ").replace(/\s{2,}/g, " ").trim();
}

function normalizeMiniMaxH3DialogueTags(text) {
  return String(text || "")
  .replace(/<\|[^<>]*\|>/g, "")
  .replace(/<\s*tool_call\|?>+/gi, "")
  .replace(/<d>\s*\[+\s*([A-Za-z][A-Za-z -]{1,30})\]\s*/gi, "<d>[$1] ")
  .replace(/<d>\s*\[([^\]]+)\]\s*([^<]*?)\s*<\/d>/gi, (_match, language, lyric) => {
    const cleanLanguage = String(language || "English").trim() || "English";
    const cleanLyric = miniMaxH3CapitalizeCueText(miniMaxH3PunctuatedCueText(lyric));
    return cleanLyric ? `<d>[${cleanLanguage}] ${cleanLyric}</d>` : "";
  }).replace(/<d>\s*(?!\[[^\]]+\]\s*)([^<]*?)\s*<\/d>/gi, (_match, lyric) => {
    const cleanLyric = miniMaxH3CapitalizeCueText(miniMaxH3PunctuatedCueText(lyric));
    return cleanLyric ? `<d>[English] ${cleanLyric}</d>` : "";
  });
}

export function miniMaxH3PunctuatedCueText(value) {
  let text = String(value || "").trim();
  if (text && !/[.!?…]["')\]]?$/.test(text)) text += ".";
  return text;
}

function miniMaxH3CapitalizeCueText(value) {
  return String(value || "").replace(/^(\s*["'“‘(]*)([a-z])/, (_match, prefix, letter) => `${prefix}${letter.toUpperCase()}`);
}

function miniMaxH3CleanSubjectNoun(value, fallback = "reference") {
  const text = String(value || "").replace(/\s+/g, " ").trim() || fallback;
  return text.replace(/^(?:the\s+)+/i, "the ");
}

function normalizeMiniMaxH3ShotDescription(text) {
  return normalizeMiniMaxH3DialogueTags(String(text || "")
    .replace(/<\s*(Subject\s+\d+)\s*\(\s*([^)<>]+)\s*>\s*'s/gi, "<$1> ($2)'s")
    .replace(/<\s*(Subject\s+\d+)\s*\(\s*([^)<>]+)\s*>/gi, "<$1> ($2)")
    .replace(/(?<!<)\bSubject\s+(\d+)\b(?!\s*>)/gi, "<Subject $1>")
    .replace(/\bto\s+Audio\s+1\b/gi, "to <Audio 1>")
    .replace(/\bfrom\s+Audio\s+1\b/gi, "from <Audio 1>")
    .replace(/\bwith\s+Audio\s+1\b/gi, "with <Audio 1>")
    .replace(/\bin\s+Audio\s+1\b/gi, "in <Audio 1>")
    .replace(/\bAudio\s+1\b/g, "<Audio 1>")
    .replace(/<+Audio 1>+/g, "<Audio 1>")
    .replace(/\bImage\s+\d+\b[,.]?/gi, "")
    .replace(/\s+/g, " ")
    .trim());
}

function miniMaxH3SentenceFragmentAfterCut(text) {
  let clean = miniMaxH3StripLeadingCutDirective(text);
  clean = clean.replace(/^(?:a|an|the)\s+/i, (match) => match.toLowerCase());
  return clean;
}

function miniMaxH3StripLeadingCutDirective(text) {
  let clean = normalizeMiniMaxH3ShotDescription(text);
  let previous = "";
  // The JSON task asks only for shot descriptions, but Storyboard/Gemma can
  // still echo one or several transition directives. The Builder owns the
  // official cut phrase and timestamp, so remove every echoed prefix first.
  while (clean && clean !== previous) {
    previous = clean;
    clean = clean.replace(/^(?:(?:the\s+camera\s+)?cuts?\s+to|cut\s+to)\s*(?::|[-–—.]|\s)*/i, "").trim();
  }
  return clean;
}

function miniMaxH3CleanPostCutGrammar(text) {
  let clean = String(text || "").replace(/\s+/g, " ").trim();
  clean = clean.replace(/\b(the camera cuts to\s+(?:a|an|the)\s+(?:(?:[a-z-]+)\s+){0,8}(?:close-up|shot|portrait|view|angle))\s+(?:holds?\s+on|frames?|captures?|isolates?|features?|shows?|finds?)\b/gi, "$1 of");
  clean = clean.replace(/\bthe camera cuts\s+to\s+(?=(?:the\s+)?camera\b)/gi, "the camera cuts. ");
  clean = clean.replace(/\bthe camera cuts\.\s*(?:the\s+camera\s+cuts(?:\s+to)?\.?\s*)+/gi, "the camera cuts. ");
  clean = clean.replace(/\bthe camera cuts\s+to\s+((?:a|an|the)\s+(?:(?:extreme|tight|wide|medium|close|low-angle|high-angle|over-the-shoulder|tracking|panning|orbiting|dolly|handheld|static|locked-off|profile|two-shot|single|insert|detail|wide-angle|telephoto)[\s-]+){0,6}(?:close-up|shot|view|angle|frame|framing)(?:\s+of\b[^.]{0,140})?\s+(?:shows|captures|reveals|frames|focuses|follows|tracks|pans|pushes|orbits|opens|begins)\b)/gi, (_match, fragment) => {
    return `the camera cuts. ${miniMaxH3CapitalizeCueText(fragment)}`;
  });
  clean = clean.replace(/\bthe camera cuts\.\s+([a-z])/g, (_match, letter) => `the camera cuts. ${letter.toUpperCase()}`);
  return clean;
}

function miniMaxH3PostCutShotText(text) {
  const clean = miniMaxH3StripLeadingCutDirective(text);
  // LLM shot descriptions are complete clauses, not guaranteed noun phrases.
  // A period is grammatical for both "The pursuit continues..." and
  // "A lower angle follows..."; blindly inserting "cuts to" is not.
  return miniMaxH3CleanPostCutGrammar(`the camera cuts. ${miniMaxH3CapitalizeCueText(clean)}`);
}

export function miniMaxH3ExtraDisplayTitle(value, fallback = "background performer") {
  const clean = String(value || "").replace(/\s+/g, " ").replace(/[,;:.]+$/g, "").trim() || fallback;
  const parts = clean.split(/\s+-\s+/).map((part) => part.trim()).filter(Boolean);
  if (parts.length >= 2 && /^(?:extra|dancer|performer)\s*\d*$/i.test(parts[0]) && /^(?:female|male|woman|man|nonbinary|non-binary)$/i.test(parts[1])) {
    return `${parts[0]} (${parts[1]})`;
  }
  return parts[0] || clean;
}

function miniMaxH3CleanExtraDescription(value, title = "") {
  let clean = String(value || "").replace(/\s+/g, " ").replace(/\s*,\s*,+/g, ", ").trim();
  clean = clean.replace(/[.;,\s]+$/g, "");
  const baseTitle = String(title || "").split(/\s+-\s+/)[0]?.trim() || "";
  const labels = [String(title || "").trim(), miniMaxH3ExtraDisplayTitle(title, ""), baseTitle].filter(Boolean).sort((a, b) => b.length - a.length);
  for (const label of labels) {
    const escaped = escapeRegExp(label);
    clean = clean.replace(new RegExp(`^${escaped}\\s+(?:has|is|wears?|with)\\s+`, "i"), "");
  }
  clean = clean
    .replace(/^(?:he|she|they)\s+wears?\s+/i, "wearing ")
    .replace(/^(?:he|she|they)\s+has\s+/i, "with ")
    .replace(/^(?:he|she|they)\s+is\s+(?:dressed|styled)\s+in\s+/i, "wearing ")
    .replace(/\b(?:he|she)\s+wears\b/gi, "they wear")
    .replace(/\b(?:he|she)\s+has\b/gi, "they have")
    .replace(/\b(?:he|she)\s+is\b/gi, "they are")
    .replace(/\b(?:his|her)\b/gi, "their")
    .replace(/\s+/g, " ")
    .trim();
  return clean;
}

function miniMaxH3CompactReferenceDescription(value, maxChars = 180) {
  const limit = Math.max(80, Math.round(Number(maxChars) || 180));
  let clean = String(value || "")
    .replace(/\s+/g, " ")
    .replace(/\b(?:Preserve this locked appearance|used as character identity, face, hair, clothing, and body-proportion reference)\s*:?\s*/gi, "")
    .replace(/\b(?:finished|completed) with\b/gi, "with")
    .replace(/\b(?:multiple|various)\s+(?=(?:flap |zippered |utility )?(?:pockets|straps|buckles|rings)\b)/gi, "")
    .replace(/\s*,\s*,+/g, ", ")
    .replace(/[.;,\s]+$/g, "")
    .trim();
  if (!clean || clean.length <= limit) return clean;
  const clip = (text, budget) => {
    const source = String(text || "").trim();
    if (source.length <= budget) return source.replace(/[.;,\s]+$/g, "");
    const slice = source.slice(0, budget + 1);
    const comma = Math.max(slice.lastIndexOf(","), slice.lastIndexOf(";"));
    const space = slice.lastIndexOf(" ");
    const cut = comma >= Math.floor(budget * 0.55) ? comma : space;
    return slice.slice(0, cut > 0 ? cut : budget).replace(/[.;,\s]+$/g, "").trim();
  };
  const wardrobeMatch = clean.match(/\b(?:they|he|she|[A-Z][\w-]*)\s+wears?\b|\bwearing\b/i);
  if (wardrobeMatch && wardrobeMatch.index > 20) {
    const appearanceBudget = Math.max(60, Math.floor(limit * 0.46));
    const appearance = clip(clean.slice(0, wardrobeMatch.index), appearanceBudget);
    const wardrobeSource = clean.slice(wardrobeMatch.index)
      .replace(/^(?:they|he|she|[A-Z][\w-]*)\s+wears?\s+/i, "")
      .replace(/^wearing\s+/i, "");
    const separator = "; wearing ";
    const wardrobe = clip(wardrobeSource, Math.max(40, limit - appearance.length - separator.length));
    return `${appearance}${separator}${wardrobe}`.replace(/[.;,\s]+$/g, "");
  }
  return clip(clean, limit);
}

export function miniMaxH3CompactExtraIdentity(value, title = "", maxChars = 150) {
  const limit = Math.max(90, Math.round(Number(maxChars) || 150));
  const clean = miniMaxH3CleanExtraDescription(value, title)
    .replace(/\s+/g, " ")
    .replace(/[.;,\s]+$/g, "")
    .trim();
  const searchable = `${title} ${clean}`;
  const genderMatch = searchable.match(/\b(female|male|woman|man|nonbinary|non-binary)\b/i);
  const gender = genderMatch ? `${genderMatch[1].toLowerCase().replace("non-binary", "nonbinary")} performer` : "background performer";
  const hairSearchable = `${title}; ${clean}`;
  const hairStyleTerm = /\b(?:hair|bob(?:bed)?|pixie cut|buzz cut|braids?|dreadlocks?|locs?|afro|shaved head|bald)\b/i;
  const titleHairSuffix = String(title || "")
    .replace(/^.*?\b(?:extra|dancer|performer)\s*\d*\s*-\s*/i, "")
    .replace(/^(?:(?:female|male|woman|man|nonbinary|non-binary)\s*-\s*)+/i, "")
    .replace(/[.;,\s]+$/g, "")
    .trim();
  const descriptionHairMatch = clean.match(/\b(?:with|has|having)\s+([^.;]{2,100}?\b(?:hair|bob(?:bed)?|pixie cut|buzz cut|braids?|dreadlocks?|locs?|afro|shaved head|bald)\b)/i);
  const fallbackHairMatch = hairSearchable.match(/\b(?:(?:very\s+)?(?:long|short|medium(?:-length)?|shoulder-length|waist-length|chin-length|black|brown|dark-brown|light-brown|blonde|blond|red|auburn|gray|grey|white|silver|blue|pink|purple|green|straight|wavy|curly|coily|braided|shaved|sleek|layered|textured|voluminous|tightly|closely|cropped|bobbed)[,\s-]+){1,8}hair\b/i)
    || hairSearchable.match(/\b(?:bald|shaved head|buzz cut|braids?|dreadlocks?|locs?|afro|pixie cut|(?:sleek[,\s-]+)?(?:chin-length[,\s-]+)?(?:black|brown|blonde|red|auburn)?[,\s-]*bob)\b/i);
  const hair = (hairStyleTerm.test(titleHairSuffix)
    ? titleHairSuffix
    : descriptionHairMatch?.[1] || fallbackHairMatch?.[0] || "")
    .replace(/\s+/g, " ")
    .replace(/[.;,\s]+$/g, "")
    .trim();
  const wardrobeStart = clean.search(/\b(?:wearing|wears?|dressed in)\b/i);
  const outfit = wardrobeStart >= 0
    ? clean.slice(wardrobeStart).replace(/^(?:wearing|wears?|dressed in)\s+/i, "")
    : "";
  const garmentNoun = /\b(?:crop top|tank top|halter top|sleeveless top|top|cargo pants|pants|trousers|jeans|shorts|skirt|dress|gown|jacket|shirt|blouse|bodysuit|jumpsuit|suit|boots|shoes|heels|camisole|halter|hoodie|sweater)\b/i;
  const completeGarmentClause = (value) => {
    const clause = String(value || "").replace(/\s+/g, " ").trim();
    const match = garmentNoun.exec(clause);
    if (!match) return "";
    return clause.slice(0, match.index + match[0].length).replace(/[.;,\s]+$/g, "").trim();
  };
  const outfitClauses = outfit
    ? outfit.split(/[.;]/, 1)[0].split(/,\s*/).map(completeGarmentClause).filter(Boolean)
    : [];
  const fullOutfit = outfitClauses.slice(0, 2).join(", ");
  const primaryOutfit = outfitClauses[0] || "";
  const buildIdentity = (wardrobe, compact = false) => {
    if (compact) {
      const genderWord = genderMatch ? genderMatch[1].toLowerCase().replace("non-binary", "nonbinary") : "performer";
      return [genderWord, hair, wardrobe ? `wearing ${wardrobe}` : ""].filter(Boolean).join("; ");
    }
    let identity = gender;
    if (hair) identity += ` with ${hair}`;
    if (wardrobe) identity += `, wearing ${wardrobe}`;
    return identity;
  };
  const candidates = [
    buildIdentity(fullOutfit),
    buildIdentity(primaryOutfit),
    buildIdentity(primaryOutfit, true),
    buildIdentity("", true),
  ].filter(Boolean);
  const result = candidates.find((candidate) => candidate.length <= limit) || candidates[candidates.length - 1] || gender;
  return result
    .replace(/[.;,\s]+$/g, "")
    .trim();
}

function miniMaxH3OfficialIntegratedDescription(segment, mode, creative) {
  const normalizedMode = normalizeMiniMaxH3Mode(mode);
  const prefix = normalizedMode === "image_to_video"
    ? `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n`
    : "";
  return `${prefix}integrated_multimodal_description:\n${creative}`;
}

export function createMiniMaxPrompt({
  assertMiniMaxH3ReferenceCapacity, autoTimeMiniMaxSingerCuesForSegment, firstLastFrameEndImageSource,
  isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode, logicalExtraSubjectsForScene,
  logicalSubjectIdsForScene, miniMaxH3CueShotContractText, miniMaxH3CutPlanForSegment,
  miniMaxH3FrameContinuityPromptEnabled, miniMaxH3ModeForSegment, miniMaxH3SceneImageIsPromptInspiration,
  miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment,
  miniMaxH3StartFrameCharacterInfluenceForSegment, miniMaxH3SubjectLabelMapForSegment,
  miniMaxH3VocalCueMapText, miniMaxOrderedImageReferenceItemsForSegment,
  miniMaxReferenceBuilderImagePathsForSegment, miniMaxReferencePurposeText, normalizeLyricCueMapForSegment,
  previousAutoChainSourceSegment, sceneDisplayName, sceneVideoConceptPromptText, segmentImageSource,
  segmentIndexInfo, segmentMappedLocationText, segmentMappedSubjectText, sceneCastGuardForSegment, selectedCastCoverageContract,
  selectedPerformerSubjectsForSegment, selectedSegmentImagePath, state, storyboardReferenceDataForSegment,
}) {
  // Story beats and the story arc are written for the whole song and can name characters who are not
  // selected for this scene. This tells the model the selected cast is the only cast.
  function sceneCastRestrictionText(segment) {
    if (!segment) return "";
    const guard = sceneCastGuardForSegment(segment);
    if (guard) return castWallText(guard);
    if (segment.no_character_present) return "No characters are in this scene. Ignore every character named in the story beat or scene idea.";
    const names = String(segmentMappedSubjectText(segment) || "").split(/\r?\n/).map((line) => line.split(":")[0].trim()).filter(Boolean);
    if (!names.length) return "";
    return `Only ${names.join(", ")} ${names.length > 1 ? "are" : "is"} in this scene. The story beat and scene idea may mention other characters from the wider story. Do not show, mention, imply, or describe any character who is not in this scene's cast.`;
  }

  // Removes any sentence that refers to a character who is not selected for this scene.
  // Story beats are model-written for the whole song, so they also lose sentences about invented people.
  function castSafeText(segment, text, invented = false) {
    return stripCastLeaks(String(text || ""), sceneCastGuardForSegment(segment), invented);
  }

  function miniMaxBuiltInDialogueCueMapText(segment) {
    if (!isMiniMaxBuiltInSpeakerAssignmentMode(segment)) return "";
    const cues = normalizeMiniMaxSpeakerAssignments(segment?.minimax_speaker_assignments || segment?.speaker_assignments || segment?.dialogue_cues || []);
    if (!cues.length) return "";
    const lines = cues.map((cue, index) => {
      const timing = miniMaxH3CueTimingText(cue, segment, cues, index);
      if (cue.type === "instrumental") {
        const note = cue.action_note ? ` Visual/audio action note: ${cue.action_note}` : "";
        return `${timing}Instrumental / no dialogue cue. No visible subject speaks or lip-syncs.${note}`;
      }
      return `${timing}${cue.speaker_name || "The assigned speaker"} speaks exactly: "${cue.text}"`;
    });
    return [
      "Native MiniMax dialogue cue map:",
      ...lines,
      "Only the assigned speaker talks during each dialogue cue. Other visible characters remain silent, mouth closed or naturally reacting.",
    ].join("\n");
  }

  function miniMaxH3NativeVoiceAssignments(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    if (settings.audio_mode !== "built_in_audio" || normalizeVideoType(segment?.performance_mode || state.videoType) !== "speaking") return [];
    const subjectRefs = storyboardReferenceDataForSegment(segment).subject_refs || [];
    const configured = subjectRefs
      .map((subject) => ({ ...subject, minimax_voice: normalizeMiniMaxH3Voice(subject.minimax_voice) }))
      .filter((subject) => subject.minimax_voice.preset_id !== "none"
        && subject.minimax_voice.preset_name
        && subject.minimax_voice.description);
    if (!configured.length) return [];
    const dialogueSpeakers = miniMaxDialogueAssignmentsForSegment(segment).map((cue) => cue.speaker_name).filter(Boolean);
    const speakerKeys = (dialogueSpeakers.length
      ? dialogueSpeakers
      : Array.isArray(segment?.lyric_singers)
        ? segment.lyric_singers
      : String(segment?.lyric_singers || "").split(/[,;\n]+/))
      .map((value) => String(value || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, " ").trim())
      .filter(Boolean);
    if (!speakerKeys.length) return configured.length === 1 ? configured : configured;
    const matched = configured.filter((subject) => {
      const nameKey = String(subject.name || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
      return nameKey && speakerKeys.some((speaker) => speaker === nameKey || speaker.includes(nameKey) || nameKey.includes(speaker));
    });
    return matched.length ? matched : configured;
  }

  function miniMaxH3NativeVoiceBlock(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    if (segmentUsesNoLipSyncPerformance(segment)
      || settings.audio_mode !== "built_in_audio"
      || normalizeVideoType(segment?.performance_mode || state.videoType) !== "speaking") return "";
    const assignments = miniMaxH3NativeVoiceAssignments(segment);
    const lyricText = isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
    const dialogueOrder = miniMaxDialogueOrderText(segment);
    if (!assignments.length && !dialogueOrder && !lyricText) return "";
    const lines = ["MINIMAX NATIVE VOICE IDENTITY — MANDATORY AND VERBATIM:"];
    if (assignments.length) {
      lines.push(
        ...assignments.map((subject) => `${subject.name || "The speaking character"}: voice preset “${subject.minimax_voice.preset_name}” — ${subject.minimax_voice.description}`),
        "Use each preset name and its complete voice description exactly as written for that character. Do not paraphrase, shorten, blend, or transfer voices between characters.",
      );
    }
    if (dialogueOrder) {
      lines.push("DIALOGUE RELIABILITY — MANDATORY:");
      lines.push("Use the single authoritative exact dialogue script written in the Native audio section. Do not print or quote that script again anywhere else in this prompt.");
      lines.push("Generate those cues in exactly the assigned speaker order. Do not merge speakers, reorder cues, rewrite, restart, repeat, improvise, omit, or add any word.");
      lines.push("Speak only in the language used by the supplied dialogue. Do not translate, switch languages, generate gibberish, babble, invented syllables, filler, phonetic substitutions, or extra vocalizations.");
      lines.push("Begin the first cue within the first 0.15 seconds, pace every supplied word naturally across the available clip, and land the final spoken word approximately 0.15-0.30 seconds before the clip ends. Never finish early and fill time with invented speech.");
      lines.push("If a character appears in more than one cue, preserve that character’s same assigned voice in every turn.");
      lines.push("If the dialogue finishes before the video ends, fill the remaining time only with natural silence, breathing, facial reaction, physical action, and low environmental ambience—never additional speech.");
    } else if (lyricText) {
      lines.push(`Generate only the exact spoken words “${lyricText}”. Do not add, repeat, improvise, or replace any word.`);
      lines.push("If the spoken line finishes before the video ends, fill the remaining time only with natural silence, breathing, facial reaction, physical action, and low environmental ambience—never additional speech.");
    }
    return lines.join("\n");
  }

  function miniMaxH3VisualOnlySafetyBlock(segment) {
    if (!segmentUsesNoLipSyncPerformance(segment)) return "";
    const builtInAudio = miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio";
    return [
      "VISUAL-ONLY B-ROLL / NO-LIP-SYNC SAFETY — MANDATORY AND FINAL:",
      "This scene is explicitly B-roll / visual-only. This safety block overrides any conflicting lyric, singer, speaker, dialogue, voice, timestamp, facial-performance, or mouth instruction elsewhere in the prompt.",
      "No visible subject sings, speaks, raps, whispers, lip-syncs, mouths words, or performs the saved lyric. Keep every visible mouth naturally relaxed or closed and use only visual acting, physical action, dancing without vocal performance, camera motion, and environmental motion.",
      builtInAudio
        ? "Built-in MiniMax Audio: generate only requested environmental ambience and sound effects. Do not generate speech, singing, dialogue, lyrics, narration, voices, or vocal layers."
        : "Input Audio: preserve Audio 1 completely unchanged as the soundtrack and timing reference, including any audible vocals, but treat those vocals as off-screen soundtrack only. No visible subject may synchronize to them.",
      "Treat the saved lyric only as hidden mood/story context; never quote it as dialogue or describe anyone performing it.",
    ].join("\n");
  }

  function applyMiniMaxH3NativeVoiceBlock(prompt, segment) {
    let clean = stripMiniMaxH3ManagedPromptBlock(
      prompt,
      "MINIMAX NATIVE VOICE IDENTITY — MANDATORY AND VERBATIM",
    );
    clean = stripMiniMaxH3ManagedPromptBlock(
      clean,
      "REFERENCE SUBJECT COUNT — MANDATORY",
    );
    clean = stripMiniMaxH3ManagedPromptBlock(
      clean,
      "VISUAL-ONLY B-ROLL / NO-LIP-SYNC SAFETY — MANDATORY AND FINAL",
    );
    const visualOnlyBlock = miniMaxH3VisualOnlySafetyBlock(segment);
    const safePrompt = visualOnlyBlock ? stripMiniMaxH3VisualOnlyTimelineVocalDirections(clean) : clean;
    const block = visualOnlyBlock ? "" : miniMaxH3NativeVoiceBlock(segment);
    return [safePrompt, block, visualOnlyBlock].filter(Boolean).join("\n\n").trim();
  }

  function miniMaxH3FramingSubjectReference(segment) {
    if (!segment || segment.no_character_present) return "the subject";
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const performerSubjects = selectedPerformerSubjectsForSegment(segment, refs);
    const mappedIds = logicalSubjectIdsForScene(refs, segment, Math.max(0, segmentIndexInfo(segment).index));
    const mappedSubjects = mappedIds
      .map((id) => refs.subjects.find((subject) => String(subject?.id || "") === String(id)))
      .filter(Boolean);
    const subject = performerSubjects[0] || mappedSubjects[0] || refs.subjects[0];
    if (!subject) return "the subject";
    const subjectId = String(subject.id || "");
    const mappedIndex = mappedIds.findIndex((id) => String(id) === subjectId);
    const fallbackIndex = refs.subjects.findIndex((item) => String(item?.id || "") === subjectId);
    const subjectNumber = (mappedIndex >= 0 ? mappedIndex : fallbackIndex) + 1;
    if (subjectNumber < 1) return "the subject";
    return `<Subject ${subjectNumber}> (S${subjectNumber})`;
  }

  function miniMaxH3ResolveFramingEntry(segment, entry) {
    const reference = miniMaxH3FramingSubjectReference(segment);
    const possessive = reference === "the subject" ? "the subject's" : `${reference}'s`;
    const shot = String(entry?.shot || "")
      .replace(/\{\{subject_possessive\}\}/gi, possessive)
      .replace(/\{\{subject\}\}/gi, reference)
      .replace(/\bthe subject[’']s\b/gi, possessive)
      .replace(/\bthe subject\b/gi, reference);
    return { ...entry, shot };
  }

  function miniMaxH3SelectedFramingEntries(segment, shotPlan = []) {
    if (!Array.isArray(shotPlan) || !shotPlan.length) return [];
    const cameraFlowKey = String(state.builderStoryboardDefaults?.camera_flow || segment?.camera_flow || "").trim();
    const preset = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowKey];
    if (!preset?.framing_candidates) return shotPlan.map(() => null);
    const customSequence = cameraFlowKey === "custom"
      ? normalizeStoryboardCustomCameraFlowSequence(segment?.custom_camera_flow_sequence || state.builderStoryboardDefaults?.custom_camera_flow_sequence)
      : [];
    const sequence = (cameraFlowKey === "custom" ? customSequence : (Array.isArray(preset?.sequence) ? preset.sequence : []))
      .filter((entry) => entry?.shot);
    if (!sequence.length) return shotPlan.map(() => null);
    const sceneText = [
      segment?.lyric_text,
      segment?.story_beat,
      segment?.notes,
      segment?.director_note,
      segment?.i2v_notes,
      segment?.timeline_note,
      segment?.motion_summary,
      segment?.character_motion_guidance,
      Number(segment?.character_motion_speed) >= 6 ? "active physical movement" : "",
      segment?.shot_type,
      segment?.camera_motion,
      sceneVideoConceptPromptText(segment),
    ].map((value) => String(value || "").toLowerCase()).join(" ");
    const semanticThemes = [
      { terms: /walk|walking|feet|step|stride|cross|move through/, match: /feet|walking/ },
      { terms: /sit|sitting|seated|chair|kneel|kneeling|ground|floor|curl|curled/, match: /seated|knees|curled/ },
      { terms: /lie|lying|recline|reclining|sleep|fallen|on her side|on the side/, match: /lying|side|overhead/ },
      { terms: /hair|wind|brush|touch.*hair|fingers/, match: /hair|fingers/ },
      { terms: /hand|hands|touch|reach|grip|hold|clothing|hip|chest/, match: /hand|hands|hip|chest|clothing/ },
      { terms: /eye|eyes|gaze|look|watch|stare|see/, match: /eye|eyes/ },
      { terms: /mouth|lip|lips|sing|sings|sings|whisper|speak|speaks/, match: /mouth|lips/ },
      { terms: /mirror|reflection|double|memory|remember/, match: /reflection|mirror/ },
      { terms: /behind|turn|turns|profile|sideways|back/, match: /behind|profile|side/ },
      { terms: /shadow|silhouette|dark|backlit|night/, match: /silhouette/ },
      { terms: /above|down|overhead|looking down/, match: /high-angle|overhead/ },
      { terms: /below|low angle|powerful|towering/, match: /low-angle/ },
      { terms: /dance|dancing|performance|rhythm|music video|singing/, match: /performance|dancing|rhythmic|vocal/ },
      { terms: /door|doorway|enter|exit|location|environment|room|street|landscape/, match: /doorway|location|environment|ground|background/ },
      { terms: /fisheye|distort|warped|curved|lens|glass|wide perspective/, match: /fisheye|distort|warped|curved|glass|lens/ },
      { terms: /approach|approaches|lean|leans|crouch|crouching|reach|reaching|bend|bends|look down/, match: /approach|lean|crouch|reach|lens|tilt/ },
    ];
    const normalizeShotId = (entry, index) => String(entry.id || `${cameraFlowKey}_${index + 1}`).trim();
    const entries = sequence.map((entry, index) => ({ entry: miniMaxH3ResolveFramingEntry(segment, entry), index, id: normalizeShotId(entry, index) }));
    const previousUsedIds = new Set(
      (Array.isArray(state.segments) ? state.segments : [])
        .filter((candidate) => candidate !== segment)
        .flatMap((candidate) => Array.isArray(candidate?.minimax_h3_framing_shot_ids) ? candidate.minimax_h3_framing_shot_ids : [])
        .map((value) => String(value || "").trim())
        .filter(Boolean),
    );
    const existingIds = Array.isArray(segment?.minimax_h3_framing_shot_ids)
      ? segment.minimax_h3_framing_shot_ids.map((value) => String(value || "").trim()).filter(Boolean)
      : [];
    const existingMatchesPlan = existingIds.length === shotPlan.length && existingIds.every((id) => entries.some((item) => item.id === id));
    const chosenIds = existingMatchesPlan ? existingIds : [];
    const usedInThisScene = new Set(chosenIds);
    const stableTieBreak = (id, shotNumber) => {
      const text = `${segment?.id || segment?.start || "scene"}:${shotNumber}:${id}`;
      let hash = 0;
      for (let index = 0; index < text.length; index += 1) hash = ((hash << 5) - hash + text.charCodeAt(index)) | 0;
      return Math.abs(hash);
    };
    const chooseEntry = (shot, shotIndex) => {
      if (chosenIds[shotIndex]) return entries.find((item) => item.id === chosenIds[shotIndex]) || entries[0];
      const available = entries.filter((item) => !usedInThisScene.has(item.id));
      const candidates = available.length ? available : entries.filter((item) => !usedInThisScene.has(item.id));
      const scored = (candidates.length ? candidates : entries).map((item) => {
        const shotText = `${item.entry.shot} ${item.entry.camera}`.toLowerCase();
        let score = previousUsedIds.has(item.id) ? -28 : 28;
        semanticThemes.forEach((theme) => {
          if (theme.terms.test(sceneText) && theme.match.test(shotText)) score += 22;
        });
        const shotWords = shotText.split(/[^a-z0-9]+/).filter((word) => word.length > 3);
        shotWords.forEach((word) => {
          if (sceneText.includes(word)) score += 3;
        });
        if (item.id === chosenIds[shotIndex - 1]) score -= 18;
        return { item, score, tie: stableTieBreak(item.id, shot.number) };
      }).sort((left, right) => right.score - left.score || left.tie - right.tie);
      const selected = scored[0]?.item || entries[shotIndex % entries.length];
      chosenIds[shotIndex] = selected.id;
      usedInThisScene.add(selected.id);
      return selected;
    };
    const selectedEntries = shotPlan.map((shot, index) => {
      const selected = chooseEntry(shot, index);
      return selected?.entry || null;
    });
    if (!existingMatchesPlan) segment.minimax_h3_framing_shot_ids = chosenIds;
    return selectedEntries;
  }

  function miniMaxH3PerShotFramingLines(segment, shotPlan = [], continuation = false) {
    const selectedEntries = miniMaxH3SelectedFramingEntries(segment, shotPlan);
    const framingLines = selectedEntries.map((entry, index) => {
      const shot = shotPlan[index];
      // A continued scene opens on the previous scene's last frame. The framing preset must not restage Shot 1.
      if (continuation && shot && Number(shot.number) === 1) {
        return "Shot 1 framing: begin exactly as Attached Picture 1 shows it, with the same shot size, camera angle, and subject pose, and keep that framing. Change it only gradually through camera movement, never by cutting.";
      }
      return entry && shot
        ? `Shot ${shot.number} framing: ${entry.shot}${entry.camera ? ` (camera: ${entry.camera})` : ""}.`
        : "";
    }).filter(Boolean);
    if (!framingLines.length) return [];
    const cameraFlowKey = String(state.builderStoryboardDefaults?.camera_flow || segment?.camera_flow || "").trim();
    const preset = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowKey];
    if (!preset?.framing_candidates) return [];
    return [
      `MANDATORY per-shot framing variety (${preset.label}):\n${framingLines.join("\n")}`,
      (continuation ? "Shot 1 follows Attached Picture 1 and the FRAME-TO-FRAME CONTINUITY contract above, not a framing preset. Any later shot uses its listed framing as exact cinematic direction." : preset.guidance) || "Use the listed framing as exact cinematic direction for each shot. Do not choose, broaden, replace, or contradict it. Add the character's emotion, performance, and action around the specified framing. Do not repeat a framing within this segment. A previously used framing may recur only when it is the strongest contextual fit or the available framing pool has been exhausted.",
    ];
  }

  // Seconds a continued scene simply carries on before its own movement begins. The author sets it per scene: at least
  // 0.5 s in and at most half of the scene, 0.5 s when nothing is set.
  function miniMaxH3ContinuationHoldSeconds(segment) {
    const sceneSeconds = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    return miniMaxH3ContinuationStartSeconds(sceneSeconds, segment?.minimax_h3_continuation_start_seconds);
  }

  // The author's own direction for a continued scene, placed after the hold so the take is never cut.
  function miniMaxH3ContinuationDirectionText(segment) {
    const direction = String(segment?.minimax_h3_continuation_direction || "").replace(/\s+/g, " ").trim();
    if (!direction) return "";
    const holdSeconds = miniMaxH3ContinuationHoldSeconds(segment);
    const sceneSeconds = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    const secondsLeft = Math.max(0, Math.round((sceneSeconds - holdSeconds) * 100) / 100);
    const sceneLength = Math.round(sceneSeconds * 100) / 100;
    return (
      `AUTHOR'S DIRECTION FOR THIS SCENE — MANDATORY, THE FINISHED DESCRIPTION MUST CONTAIN IT: "${direction}"\n`
      + `SCENE TIMING: This scene is ${sceneLength} seconds long. The direction starts at ${holdSeconds} seconds and has to be completely finished before the scene ends, which leaves ${secondsLeft} seconds for it. Perform every action of the direction in the author's order at a brisk pace that fits those ${secondsLeft} seconds, and end the shot with the last action fully done, never cut off or left unfinished. `
      + `Write the one shot description in two timed parts and put the timing in the text itself. First: "For the first ${holdSeconds} seconds, ..." continuing the opening frame's action with the same camera motion, framing, and pace. Then: "At about ${holdSeconds} seconds, ..." performing every action in the direction above, in the author's order and with the author's own verbs and objects, as one smooth continuous movement in the same take. `
      + `If the direction needs a body position or facing different from Attached Picture 1 (for example standing up, walking, or turning), first describe the natural movement that gets the subject there, inside the same take. `
      + `Do not skip or replace any action in the direction. If it is a lot for the time left, perform its actions in quicker succession rather than leaving any out. Never cut, change shot, or restart the action to reach it. Write it as the shot's one movement: if the mapped location differs from the previous scene's, let that same movement carry the shot into the new location instead of adding a second one. `
      + `Finish by stating where the shot ends.`
      + miniMaxH3MaskedPerformanceText(segment, "transition")
    );
  }

  // Vocal performance the continuation must keep going. Empty for instrumental, b-roll and no lip-sync scenes.
  function miniMaxH3MaskedPerformanceText(segment, part) {
    if (segmentUsesNoLipSyncPerformance(segment)) return "";
    const dialogue = miniMaxDialogueAssignmentsForSegment(segment);
    const lyric = dialogue.length
      ? dialogue.map((cue) => cue.text).join(" ")
      : isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
    if (!String(lyric || "").trim()) return "";
    const action = dialogue.length ? "speaking their dialogue" : "singing the scene's lyrics";
    return part === "transition"
      ? ` The performer keeps ${action} on camera through the whole movement, lips and mouth in continuous sync with <Audio 1>, and the movement keeps their face in view (it may travel around them but never hides the face for more than a moment).`
      : ` The performer is mid-performance, so keep ${action} without a pause from the first frame, lips and mouth in continuous sync with <Audio 1>.`;
  }

  function miniMaxH3FrameLocationContinuityContract(segment) {
    const previousSegment = previousAutoChainSourceSegment(segment);
    const currentLocation = storyboardReferenceDataForSegment(segment)?.location_ref || null;
    const previousLocation = previousSegment
      ? storyboardReferenceDataForSegment(previousSegment)?.location_ref || null
      : null;
    const locationKey = (location) => String(location?.id || location?.name || location?.description || "")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, " ");
    const currentKey = locationKey(currentLocation);
    const previousKey = locationKey(previousLocation);
    const locationIdentity = (location, fallback) => {
      const name = String(location?.name || fallback).trim();
      const description = String(location?.description || "").trim().replace(/\s+/g, " ");
      if (!description) return name;
      const firstSentence = description.match(/^.*?[.!?](?:\s|$)/)?.[0]?.trim() || description;
      const compactDescription = firstSentence.length > 240
        ? `${firstSentence.slice(0, 237).trim()}...`
        : firstSentence;
      return `${name} (${compactDescription})`;
    };
    const currentName = locationIdentity(currentLocation, "the current mapped location");
    const previousName = locationIdentity(previousLocation, "the preceding mapped location");
    if (!currentKey) return "";
    if (previousKey && previousKey !== currentKey) {
      const transitionSettings = miniMaxH3SettingsForSegment(segment);
      const preset = transitionSettings.location_transition_preset;
      const customDirection = transitionSettings.location_transition_custom;
      const holdSeconds = miniMaxH3ContinuationHoldSeconds(segment);
      const performanceLine = miniMaxH3MaskedPerformanceText(segment, "transition");
      const commonEnding = `Complete the transition inside this uninterrupted scene and end looking deeper into ${currentName}, with its mapped geography filling the image as the location inherited by the following scene.`;
      const directions = {
        normal: (
          `LOCATION PHASE — PHYSICAL THRESHOLD TURN: This scene performs the single physical passage from ${previousName} into ${currentName}. `
          + `Track the subject to a visible doorway, gateway, solid foreground edge, or natural threshold already present in the opening frame. `
          + `After the subject crosses its plane, let that foreground edge sweep fully across the image as a natural full-frame occlusion. During that continuous occlusion, cross the threshold and execute a clear camera arc around the subject toward ${currentName}. `
          + `As the occluding edge clears, the camera faces forward into ${currentName}; ${previousName} has passed fully beyond the rear camera plane while the subject and destination reappear through coherent motion and parallax. Describe the threshold crossing, full-frame occlusion, camera arc, and final viewing direction explicitly.`
        ),
        surreal: (
          `LOCATION PHASE — SURREAL MATERIAL TRANSFORMATION: Transform ${previousName} progressively into ${currentName} through imaginative, visually connected material changes that travel across the frame. `
          + `Use forms, textures, weather, particles, light, and movement visible in the actual opening image as the transformation source. Preserve the subject's continuous identity, action, position, and camera momentum while each physical element evolves into a corresponding element of ${currentName}. Make the transformation richly creative, spatially progressive, and complete.`
        ),
        cinematic: (
          `LOCATION PHASE — CINEMATIC CONCEAL AND REVEAL: Choose a visually suitable detail present in the actual opening image, such as the subject's eye, hair, clothing, a shadow, bright light, fog, doorway, or foreground object. `
          + `Move the camera into that detail until it fills the complete frame, carry the camera continuously through the concealed moment, then pull or glide outward to reveal the subject physically present in ${currentName}. Preserve the subject's action and camera momentum across the full-frame concealment.`
        ),
        inner_world: (
          `LOCATION PHASE — INNER WORLD PORTAL: Reveal ${currentName} living inside a visually suitable surface already present in the opening image, such as an eye reflection, mirror, window, pool, pendant, crystal, smoke formation, or luminous opening. `
          + `Let the destination become visibly dimensional inside that surface, move the camera continuously through it, and emerge with the subject inside the full-scale geography of ${currentName}. Treat the portal as one coherent passage with continuous scale, perspective, light, and motion.`
        ),
        match: (
          `LOCATION PHASE — VISUAL MATCH TRANSITION: Inspect the actual opening image and find a strong shared shape, color, texture, light pattern, or motion that can connect ${previousName} to ${currentName}. `
          + `Track that matching element as it fills or commands the composition, then let the same visual form resolve seamlessly as a real element in ${currentName}. Carry the subject's movement and camera trajectory through the visual correspondence with precise composition and spatial flow.`
        ),
        motion: (
          `LOCATION PHASE — MOTION-DRIVEN TRANSITION: Use an energetic camera action suited to the actual opening frame, such as a whip pan, rapid orbit, fast push, foreground sweep, or close pass around the subject. `
          + `Let directional motion and natural motion blur carry the complete image across the location boundary, then resolve the same movement and screen direction clearly inside ${currentName}. Preserve the subject's action, rhythm, and camera momentum throughout.`
        ),
        masked: (
          `LOCATION PHASE — MASKED CONTINUATION TRANSITION: The renderer already holds the previous scene's last moments, so for the first ${holdSeconds} seconds simply continue the opening frame's action in ${previousName} with the same subject, framing, pace, and camera motion. No cut, no change of angle, and no new setup. `
          + `At about ${holdSeconds} seconds, begin ONE smooth, motivated movement that carries the shot into ${currentName}. Choose the movement that best suits the opening frame: a camera move (push-in, pull-back, pan, tilt, track, or arc around the subject) or a natural subject move (turning, stepping through, walking on) that the camera follows. `
          + `Let the surroundings change progressively through that movement and its parallax, with no wipe, flash, portal, morph, or cut. Describe the move and where the shot has arrived when it finishes.${performanceLine}`
        ),
        creative_auto: (
          `LOCATION PHASE — CREATIVE IMAGE-AWARE TRANSITION: Inspect the actual opening image, ${previousName}, and ${currentName}, then choose the most visually convincing imaginative transition for their specific forms, materials, lighting, subject action, and camera trajectory. `
          + `Build one explicit on-screen mechanism with readable progression and continuous movement, using a physical passage, material transformation, cinematic concealment, portal, visual match, motion bridge, or an equally coherent original idea. Make the chosen mechanism concrete in the shot description.`
        ),
        custom: customDirection
          ? (
            `LOCATION PHASE — CUSTOM TRANSITION: Apply this scene's authored transition direction: ${customDirection} `
            + `Translate it into explicit positive visual action that carries the actual opening image from ${previousName} into ${currentName} through one coherent continuous camera experience.`
          )
          : (
            `LOCATION PHASE — CUSTOM TRANSITION FALLBACK: Inspect the actual opening image, ${previousName}, and ${currentName}, then create one highly imaginative, visually readable transition whose on-screen mechanism follows the subject and camera momentum into the destination.`
          ),
      };
      return `${directions[preset]} ${commonEnding}`;
    }
    return (
      `LOCATION PHASE — ESTABLISHED CURRENT LOCATION: ${currentName} is now the complete established environment. `
      + `Continue forward through the exact visible opening state and deeper into its mapped geography. Build every newly revealed environmental feature from ${currentName}, preserve its established spatial logic, and let the camera trajectory create the next composition. `
      + `End looking deeper into ${currentName}, fully grounded in that location.`
    );
  }

  function miniMaxH3CreativePromptContextForSegment(segment, mode, options = {}) {
    const settings = miniMaxH3SettingsForSegment(segment);
    const nativeAudio = settings.audio_mode === "built_in_audio";
    const duration = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    const exactDuration = String(Number(duration.toFixed(3)));
    const aspectRatio = String(settings.aspect_ratio || "16:9").match(/\d+\s*:\s*\d+/)?.[0]?.replace(/\s+/g, "") || "16:9";
    const visualOnly = segmentUsesNoLipSyncPerformance(segment);
    const dialogueAssignments = visualOnly ? [] : miniMaxDialogueAssignmentsForSegment(segment);
    const lyricText = dialogueAssignments.length
      ? dialogueAssignments.map((cue) => cue.text).join(" ")
      : isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
    const performanceMode = visualOnly
      ? "no_lip_sync"
      : normalizeVideoType(segment?.performance_mode || state.videoType);
    const cameraMotionSpeed = Number(segment?.camera_motion_speed ?? state.builderStoryboardDefaults?.camera_motion_speed ?? 4);
    const characterMotionSpeed = Number(segment?.character_motion_speed ?? state.builderStoryboardDefaults?.character_motion_speed ?? 4);
    const cameraMotionGuidance = String(segment?.camera_motion_speed_guidance || state.builderStoryboardDefaults?.camera_guidance || "").trim();
    const characterMotionGuidance = String(segment?.character_motion_guidance || state.builderStoryboardDefaults?.character_guidance || "").trim();
    const cutPlan = miniMaxH3CutPlanForSegment(segment);
    const shotPlan = miniMaxH3OfficialShotPlan(cutPlan);
    const compact = (value, limit = 900) => {
      const text = String(value || "").replace(/\s+/g, " ").trim();
      return text.length > limit ? `${text.slice(0, limit).trim()}...` : text;
    };
    const add = (parts, label, value, limit = 900) => {
      const text = compact(value, limit);
      if (text) parts.push(`${label}:\n${text}`);
    };
    const parts = [
      "MiniMax H3 shot-description task.",
      `Return exactly ${shotPlan.length} JSON shot description${shotPlan.length === 1 ? "" : "s"}: {"shots":[{"description":"..."}]}`,
      "Write only creative shot text. Do not include [Shot N] labels, cut times, markdown, or analysis.",
      `Mode: ${miniMaxH3ModeLabel(mode)}`,
      `Duration: ${exactDuration}s, ${aspectRatio}`,
      `Audio mode: ${nativeAudio ? "Built-in MiniMax audio" : "Input Audio 1 preserved by Builder"}`,
      `Shot count: ${shotPlan.length}`,
    ];
    if (options.frameContinuityPrompt) {
      const hasPromptInspiration = mode === "reference_to_video" && miniMaxH3SceneImageIsPromptInspiration(segment);
      const firstRendererAttachment = hasPromptInspiration ? 3 : 2;
      const locationContinuityContract = miniMaxH3FrameLocationContinuityContract(segment);
      const maskedLatentContinuation = settings.continuity_mode === "latent_continuation_masked";
      const hasContinuationDirection = Boolean(miniMaxH3ContinuationDirectionText(segment));
      parts.push(
        maskedLatentContinuation
          ? (
            "FRAME-TO-FRAME CONTINUITY — HIGHEST PRIORITY:\n"
            + "Attached Picture 1 is the previous rendered scene's actual final frame. The renderer already holds the last moments of the previous scene's motion and audio as the start of this render, so this scene is the very next moment of the same uninterrupted take, not a new shot. "
            + "Begin the returned description with exactly: ‘Continuing seamlessly from the previous shot, the camera maintains its established course as’ and immediately name the same camera movement continuing at the same speed. "
            + "Keep the subject's action, pace, pose, framing, camera angle, lighting, wardrobe, and environment exactly as Attached Picture 1 shows them, "
            + (hasContinuationDirection ? "then carry on exactly as the AUTHOR'S DIRECTION at the end of this scene concept says. " : "then advance them one small natural step at a time. ")
            + "Do not restage, reset, re-establish, change framing, or cut. "
            + (hasContinuationDirection ? "The author's direction decides what happens next." : "The current story beat, lyrics/audio timing, and mapped location decide where the action goes next, reached through the continuing motion.")
            + `${miniMaxH3MaskedPerformanceText(segment, "opening")} `
            + "Write every finished shot sentence as a positive description of the desired visible result. Attached Picture 1 remains an LLM-only observation source; finished prose uses direct visual description and the documented renderer labels."
          )
          : "FRAME-TO-FRAME CONTINUITY — HIGHEST PRIORITY:\n"
        + "Attached Picture 1 is the previous rendered scene's actual final frame. It is the visual truth for the first instant of this scene. Begin from its exact subject position, pose, expression, camera angle, framing, lighting, wardrobe, environment geometry, foreground layers, and camera momentum. "
        + "Continue as one seamless uninterrupted take. Begin the returned description with exactly: ‘Continuing seamlessly from the previous shot, the camera maintains its established course as’ and immediately specify the next physical camera movement. "
        + "The current story beat, lyrics/audio timing, mapped location, and supporting references determine the destination while the visible opening state supplies the exact starting point. "
        + "Express the complete transition through continuous camera travel, physical subject motion, stable geometry, progressive reveal, and coherent parallax. Write every finished shot sentence as a positive description of the desired visible result. Attached Picture 1 remains an LLM-only observation source; finished prose uses direct visual description and the documented renderer labels."
      );
      parts.push(
        "OPENING SUBJECT VISIBILITY — IMAGE-AWARE: Inspect Attached Picture 1 and inventory the human or character subjects actually visible in its opening composition. "
        + "Visible subjects continue from their exact observed position and state. Every currently mapped subject absent from Attached Picture 1 begins physically offscreen. "
        + "Bring an offscreen subject into view through a clearly described continuous screen-space event: an entrance through a frame edge, doorway, path, foreground layer, or the selected transition mechanism, or a camera pan, track, or orbit that reaches and reveals them in a connected position. "
        + "State the observed empty or partially occupied opening composition first, then the exact entrance or camera-reveal route, then that subject's performance action. Their first visible moment occurs through that physical entrance or reveal."
      );
      if (locationContinuityContract) parts.push(locationContinuityContract);
      if (hasPromptInspiration) {
        parts.push("Attached Picture 2 is the scene-image inspiration used only under its existing environment/framing limits. Renderer Image 1 begins at Attached Picture 3.");
      } else {
        parts.push(`Renderer Image 1 begins at Attached Picture ${firstRendererAttachment}; later renderer images follow in order.`);
      }
    }
    const characterBudget = miniMaxH3PromptCharacterBudget(segment, mode, options.h3TargetLimit ?? 6500);
    const perShotBudget = Math.max(1, Math.floor(characterBudget.shotDescriptionChars / Math.max(1, shotPlan.length)));
    parts.push(
      `MANDATORY CHARACTER BUDGET: The combined text inside all ${shotPlan.length} JSON description values must not exceed ${characterBudget.shotDescriptionChars} characters total (about ${perShotBudget} per shot). Stay within this combined limit; be concise without omitting required subjects, actions, camera direction, or vocal cues.`
    );
    const lifeMovements = miniMaxH3LifeMovementBank(characterMotionSpeed, String(segment?.id || segment?.label || ""));
    parts.push(
      "SHOT FORMAT — MANDATORY: Write each shot as 2 to 4 complete sentences, about 60 to 110 words, in this order. "
      + "1) Camera: the opening framing, one named camera move with its direction and speed, and the ending framing. "
      + "2) Subject action: each person in the cast does their own continuous physical action, written by what the body does. "
      + `3) Life detail: one or two small human movements placed inside the action, for example ${lifeMovements.join("; ")}. `
      + "4) Light and set: one short line using only the mapped location's own light and objects. "
      + "Describe only what the camera sees. Show emotion only as visible movement. Do not use feeling words such as feel, grief, longing, memory, soul, or emotional. "
      + "Write complete grammatical prose, not notes, labels, or fragments, and do not replace named subjects with S1/S2 shorthand. "
      + "Character appearance is already carried by the reference images and the Builder. Do not list clothing, hair, accessories, jewelry, or facial features. Mention a garment or feature only when it moves or reacts in the action, in one brief clause at most."
    );
    parts.push(
      `MOTION ENERGY — MANDATORY: ${miniMaxH3MotionEnergyText(cameraMotionSpeed, characterMotionSpeed)} `
      + `This scene is ${exactDuration} seconds long. Fill the full shot length with continuous action at this energy. Do not slow down or hold still. The length sets how many actions are chained, and the speed values set how energetic each one is.`
    );
    const cutTimes = shotPlan.slice(1).map((shot) => shot.timecode);
    if (cutTimes.length) {
      parts.push(`Builder cut times for your planning only: ${cutTimes.join(", ")}. Do not write these times.`);
    } else {
      parts.push("Continuous shot: return one description only.");
    }
    const selectedCastContract = selectedCastCoverageContract(segment, {
      shotPlan,
      labelMap: miniMaxH3SubjectLabelMapForSegment(segment, mode),
    });
    if (selectedCastContract) parts.push(selectedCastContract);
    const subjectLabelMap = miniMaxH3SubjectLabelMapForSegment(segment, mode);
    const subjectLabelEntries = Array.from(new Map(
      Array.from(subjectLabelMap.values())
        .filter((item) => (item.kind === "subject" || item.kind === "extra") && item.label)
        .map((item) => [item.label, item]),
    ).values());
    const labelForSubject = (subject) => (
      subjectLabelMap.get(String(subject?.id || "").trim()) || subjectLabelMap.get(String(subject?.name || "").trim().toLowerCase())
    )?.label;
    const performerLabelSet = new Set(selectedPerformerSubjectsForSegment(segment).map(labelForSubject).filter(Boolean));
    const speakerLabelSet = new Set(dialogueAssignments
      .map((cue) => subjectLabelMap.get(String(cue.speaker_name || "").trim().toLowerCase())?.label)
      .filter(Boolean));
    const cueMapMode = String(segment?.lyric_performance_mode || "together") === "cue_map";
    // Each subject's vocal role comes straight from the Map Performers / Review Lines assignments.
    const subjectRole = (item) => {
      if (item.kind === "extra") return "supporting extra; never sings or speaks";
      if (visualOnly || performanceMode === "no_lip_sync") return "does not sing, speak, or lip sync; acts and reacts silently with a relaxed closed mouth";
      if (performanceMode === "speaking" && dialogueAssignments.length) {
        return speakerLabelSet.has(item.label)
          ? "SPEAKS only its own assigned lines from the Exact dialogue order, with visible lip sync"
          : "does not speak or lip sync; listens and reacts silently with a relaxed closed mouth";
      }
      if (lyricText && performerLabelSet.size) {
        if (!performerLabelSet.has(item.label)) return "does NOT sing, speak, or lip sync; acts and reacts silently with a relaxed closed mouth";
        return cueMapMode
          ? "SINGS and lip-syncs only its assigned cue lines from the Performer / vocal cue map, with a relaxed closed mouth at every other moment"
          : "SINGS and lip-syncs the exact lyric line with visible mouth, lip, and jaw movement";
      }
      return "";
    };
    // Character names in narrative scene text become their labels before the LLM sees them (quoted lyrics and dialogue are left untouched).
    const labelize = (value) => {
      const source = String(value || "");
      if (!source || !subjectLabelEntries.length) return source;
      return source.split(/(“[^”]*”|"[^"]*")/).map((chunk, index) => {
        if (index % 2 === 1) return chunk;
        let text = chunk;
        for (const item of subjectLabelEntries) {
          const name = String(item.name || "").trim();
          if (!name) continue;
          const core = name.replace(/^the\s+/i, "");
          const pattern = core !== name ? `\\bthe\\s+${escapeRegExp(core)}\\b` : `\\b${escapeRegExp(name)}\\b`;
          text = text.replace(new RegExp(pattern, "gi"), item.label);
        }
        return text;
      }).join("");
    };
    if (subjectLabelEntries.length) {
      const firstEntry = subjectLabelEntries[0];
      parts.push([
        `CAST FOR THIS SCENE — MANDATORY: Character names in the scene text below have been replaced by these labels. Refer to every character in the shot text by label, writing the label followed by the assigned name in parentheses on its first mention in each shot, for example "${firstEntry.label} (${firstEntry.name || "assigned name"})", then the label alone. Never refer to a character only by name, "he", or "she". Each role below comes from the user's Map Performers and Review Lines settings and must be followed exactly.`,
        ...subjectLabelEntries.map((item) => {
          const role = subjectRole(item);
          return `- ${item.label} (${item.name || "mapped subject"})${role ? `: ${role}` : ""}`;
        }),
        `ONLY THESE PEOPLE APPEAR — MANDATORY: ${subjectLabelEntries.map((item) => `${item.label} (${item.name || "mapped subject"})`).join(", ")}. No other person, hand, arm, shadow, reflection, silhouette, or crowd appears in any shot. Refer to people only by label, never by pronoun.`,
      ].join("\n"));
      if (subjectLabelEntries.length > 1) {
        parts.push(
          "INDEPENDENT SUBJECT ACTION — MANDATORY: In every shot, give each visible subject their own physical action, eyeline, and small movement that could be filmed on its own, and show how each responds to the other. "
          + "Do not describe one subject only as a passive object of the other's action. Keep each subject's action tied to their label so the viewer can follow who does what and when."
        );
      }
    }
    const framingLines = miniMaxH3PerShotFramingLines(segment, shotPlan, Boolean(options.frameContinuityPrompt));
    if (framingLines.length) parts.push(...framingLines);
    parts.push(`Camera speed: ${Number.isFinite(cameraMotionSpeed) ? cameraMotionSpeed : 4}/10${cameraMotionGuidance ? ` - ${compact(cameraMotionGuidance, 240)}` : ""}.`);
    parts.push(`Character speed: ${Number.isFinite(characterMotionSpeed) ? characterMotionSpeed : 4}/10${characterMotionGuidance ? ` - ${compact(characterMotionGuidance, 240)}` : ""}.`);
    if (cameraMotionSpeed >= 7) {
      parts.push("Camera rule: use energetic, visibly active camera movement; avoid slow/static/locked-off language.");
    }
    const cueShotContract = miniMaxH3CueShotContractText(segment, mode);
    if (characterMotionSpeed >= 4) {
      if (cueShotContract) {
        parts.push("Character rule: include clear body action, gesture, step, or set interaction. Singing and lip sync occur only in vocal cue shots; instrumental cue shots remain completely non-vocal.");
      } else if (visualOnly || segment?.no_character_present || !lyricText) {
        parts.push("Character rule: include clear body action, gesture, step, or set interaction when a character is visible. Do not add singing, speaking, or lip sync.");
      } else if (performanceMode === "speaking") {
        parts.push("Character rule: include clear body action, gesture, step, or set interaction in addition to the required dialogue lip sync. Mouth movement alone is not enough.");
      } else {
        parts.push("Character rule: include clear body action, gesture, step, or set interaction in addition to the required singing lip sync. Mouth movement alone is not enough.");
      }
    }
    if (segment?.no_character_present) {
      parts.push("Vocal performance: no visible character / no lip sync. Do not invent a visible singer or speaker.");
    } else if (performanceMode === "no_lip_sync") {
      parts.push("Vocal performance: visual-only / no lip sync. Use lyric only as hidden mood/story context.");
    } else if (lyricText && performanceMode === "speaking") {
      parts.push("Vocal performance: speaking with exact dialogue lip sync. Use the stable visible subject label, and place every spoken cue inside <d>[English] exact words with final punctuation.</d>. Mention the visible speaking action naturally in the shot descriptions.");
      add(parts, "Exact dialogue order", labelize(miniMaxDialogueOrderText(segment)) || `The assigned speaker says exactly: "${lyricText}"`);
      add(parts, "Timed native dialogue cue map", miniMaxBuiltInDialogueCueMapText(segment), 1800);
    } else if (lyricText && cueShotContract) {
      parts.push("TIMED VOCAL CUES ONLY: Follow the timed singer/shot contract exactly. Never place, anticipate, continue, or repeat a lyric in an instrumental shot. A performer sings and lip-syncs only inside the specifically assigned vocal cue shot.");
    } else if (lyricText) {
      parts.push("MANDATORY VOCAL PERFORMANCE: The assigned subject is visibly singing the exact supplied lyric/audio during this scene. Use the stable visible subject label, " + (nativeAudio
        ? "and place every performed lyric cue inside <d>[English] exact words with final punctuation.</d>. "
        : "and write the exact lyric words, unchanged, inside the shot description in double quotes, introduced like this: <Subject 1> sings the lyric line, \"the exact words\". Every lyric line given below must appear in a shot, in order. ") + "Show clear, natural mouth, lip, jaw, and facial movement synchronized to the audible vocal. Never describe the lips as closed, still, motionless, or sealed while the assigned vocal is being performed. Body action is required in addition to lip sync; it does not replace lip sync. Non-verbal vocals such as oooh, ah, humming, and sustained notes still require visible mouth movement. Mention the visible singing action naturally in the shot descriptions.");
      add(parts, "Exact lyric line", lyricText);
    } else {
      parts.push("Vocal performance: no exact lyric or dialogue is assigned to this scene.");
    }
    if (cueShotContract) add(parts, "Timed singer / shot contract", cueShotContract, 1800);
    const vocalCueMap = miniMaxH3VocalCueMapText(segment, mode);
    if (vocalCueMap) {
      add(parts, "Performer / vocal cue map", vocalCueMap);
      if (selectedPerformerSubjectsForSegment(segment).length >= 2) {
        parts.push("Multi-performer rule: use exact subject labels such as <Subject 1>, <Subject 2>, and <Audio 1> in the shot descriptions. For each vocal cue, write that the assigned subject precisely lip-syncs to <Audio 1>, singing or speaking the cue inside <d>[English] cue.</d>. Apply no performance direction to other performers during that cue.");
        parts.push("Shot wording rule: do not begin descriptions with 'The camera cuts to' or 'The camera...'. Start with the resulting framing or subject action, e.g. 'A panning medium shot shows...' or '<Subject 2> (S2) steps forward...'.");
      }
    }
    add(parts, "Scene idea", labelize(castSafeText(segment, sceneVideoConceptPromptText(segment))));
    add(parts, "Scene notes", labelize(segment?.notes || segment?.director_note));
    add(parts, "Storyboard Builder context", labelize(castSafeText(segment, options.storyboardContext || options.extraStoryboardNotes)), 2200);
    add(parts, "Motion/camera request", labelize(segment?.i2v_notes));
    add(parts, "Story beat", labelize(castSafeText(segment, segment?.story_beat, true)));
    add(parts, "Scene cast restriction (mandatory)", sceneCastRestrictionText(segment));
    add(parts, "Lyric section", segment?.lyric_section);
    add(parts, "Subject (identity context only; do not restate appearance or clothing in the shot text)", segment?.no_character_present ? "No main character is visible in this scene." : segmentMappedSubjectText(segment));
    if (["reference_to_video", "image_reference_to_video", "video_to_video"].includes(normalizeMiniMaxH3Mode(mode))) {
      add(parts, "Mapped extra subjects — mandatory (who must appear; do not restate appearance or clothing in the shot text)", segmentMappedExtraSubjectText(segment, mode), 6000);
    }
    add(parts, "Location", segmentMappedLocationText(segment));
    const referenceLabels = miniMaxH3ReferenceAssignmentLines(segment, mode)
      .map((line) => String(line || "").split(":")[0].trim())
      .filter(Boolean);
    if (referenceLabels.length) {
      parts.push(`Available renderer reference labels: ${referenceLabels.join(", ")}. Use every selected subject label required by the selected-cast coverage and per-shot rotation contract; other labels need only be mentioned when useful. Do not define labels in the shot text.`);
    }
    if (mode === "video_to_video") {
      const videoAssignments = miniMaxH3VideoAssignmentLines(segment);
      if (videoAssignments.length) parts.push(`Available video reference labels: ${videoAssignments.map((line) => String(line || "").split(":")[0].trim()).filter(Boolean).join(", ")}.`);
    }
    add(parts, "Manual audio direction for staging only", segment?.audio_direction);
    add(parts, "Continuity notes for staging only", labelize(segment?.continuity));
    if (options.frameContinuityPrompt) {
      // Last in the concept, where the model weighs it most.
      const continuationDirection = miniMaxH3ContinuationDirectionText(segment);
      if (continuationDirection) parts.push(continuationDirection);
    }
    return parts.join("\n\n");
  }

  function miniMaxH3ReferenceAssignmentLines(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    if (normalizedMode === "image_to_video") {
      return ["<Picture 1>: exact start frame and authoritative opening composition, character identity, clothing, location, lighting, and visual-state anchor."];
    }
    if (normalizedMode === "image_reference_to_video") {
      return miniMaxH3ImageReferencePromptItems(segment).map((item, index) => {
        const pictureLabel = `<Picture ${index + 1}>`;
        if (item.kind === "start_frame") return `${pictureLabel}: exact start frame and authoritative opening composition, location, lighting, and visual-state anchor.`;
        if (item.kind === "end_frame") return `${pictureLabel}: exact end frame and authoritative final composition, subject placement, lighting, and visual-state anchor.`;
        const detail = [item.label, item.description].map((value) => String(value || "").trim()).filter(Boolean).join(" — ");
        return `${pictureLabel}: ${miniMaxReferencePurposeText(item, segment)}${detail ? `. ${detail}` : ""}`;
      });
    }
    if (normalizedMode !== "reference_to_video" && normalizedMode !== "video_to_video") return [];
    return miniMaxOrderedImageReferenceItemsForSegment(segment, normalizedMode).map((item, index) => {
      const detail = [item.label, item.description].map((value) => String(value || "").trim()).filter(Boolean).join(" — ");
      return `<Picture ${index + 1}>: ${miniMaxReferencePurposeText(item, segment)}${detail ? `. ${detail}` : ""}`;
    });
  }

  function miniMaxH3OpeningLine(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    const duration = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    const exactDuration = String(Number(duration.toFixed(3)));
    const aspectRatio = String(settings.aspect_ratio || "16:9").match(/\d+\s*:\s*\d+/)?.[0]?.replace(/\s+/g, "") || "16:9";
    const style = String(segment?.minimax_h3_video_style || state.builderStoryboardDefaults?.video_style || "").trim();
    const styleText = style || "photorealistic";
    return `Generate a ${exactDuration}-second ${aspectRatio} ${styleText} video.`;
  }

  function miniMaxH3AudioAssignmentBlock(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    const nativeAudio = settings.audio_mode === "built_in_audio";
    const visualOnly = segmentUsesNoLipSyncPerformance(segment);
    const dialogueAssignments = visualOnly ? [] : miniMaxDialogueAssignmentsForSegment(segment);
    const dialogueOrder = visualOnly ? "" : miniMaxDialogueOrderText(segment);
    const lyricText = dialogueAssignments.length
      ? dialogueAssignments.map((cue) => cue.text).join(" ")
      : isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
    const performanceMode = visualOnly
      ? "no_lip_sync"
      : normalizeVideoType(segment?.performance_mode || state.videoType);
    const singerNames = (dialogueAssignments.length
      ? dialogueAssignments.map((cue) => cue.speaker_name)
      : Array.isArray(segment?.lyric_singers)
        ? segment.lyric_singers
        : String(segment?.lyric_singers || "").split(/[,;\n]+/))
      .map((value) => String(value || "").trim())
      .filter(Boolean);
    const performerLabel = singerNames.length
      ? singerNames.join(singerNames.length === 2 ? " and " : ", ")
      : "the visible performer";
    if (segment?.no_character_present) {
      return nativeAudio
        ? "Native audio: generate only low environmental ambience and requested sound effects. Do not invent a singer, speaker, dialogue, lyrics, or voice."
        : "Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, rhythm, phrasing, tone, and duration. Do not invent a visible singer or speaker.";
    }
    if (performanceMode === "no_lip_sync") {
      return nativeAudio
        ? "Native audio: generate only ambience and requested sound effects. Do not generate speech, singing, lyrics, dialogue, or visible mouth synchronization."
        : "Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, rhythm, phrasing, tone, and duration, but do not show singing, speaking, or mouth synchronization.";
    }
    if (lyricText && performanceMode === "speaking") {
      return nativeAudio
        ? `Native audio: generate the exact dialogue only. ${dialogueOrder ? `Mandatory dialogue order:\n${dialogueOrder}` : `${performerLabel} says the exact line “${lyricText}”.`} Do not merge speakers, reorder cues, replace, alter, repeat, extend, improvise, omit, or add words. Use silence, breathing, facial reaction, physical action, and low ambience for all remaining time.`
        : `Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, rhythm, phrasing, tone, and duration, and use it as the exact voice, timing, and lip-sync reference. ${performerLabel} says the exact line “${lyricText}”. Synchronize lips, mouth shapes, jaw movement, facial muscles, and breathing precisely to that spoken line in Audio 1. Do not replace, alter, extend, or add words.`;
    }
    if (lyricText) {
      return `Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, rhythm, phrasing, tone, and duration, and use it as the exact vocal, timing, and lip-sync reference. ${performerLabel} is singing the exact line “${lyricText}”. Synchronize lips, mouth shapes, jaw movement, facial muscles, and breathing precisely only while that sung line is audible in Audio 1. Never stretch, restart, or repeat the full lyric across every timestamp. When the vocal ends, the mouth closes or relaxes naturally while physical action continues. Do not generate, add, replace, remix, or extend any music, ambience, dialogue, vocals, or sound effects.`;
    }
    return nativeAudio
      ? "Native audio: generate only ambience and requested sound effects. Do not invent lyrics, dialogue, or voices."
      : "Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, rhythm, phrasing, tone, and duration. Do not generate, add, replace, remix, or extend any music, ambience, dialogue, vocals, or sound effects.";
  }

  function miniMaxH3FinalAudioBlock(segment) {
    const block = miniMaxH3AudioAssignmentBlock(segment);
    if (!block) return "";
    return block.replace(/^Audio 1:/, "Audio:").replace(/^Native audio:/, "Native audio:");
  }

  function miniMaxH3ContinuityBlock(segment) {
    const parts = [];
    const subjectText = segment?.no_character_present ? "" : segmentMappedSubjectText(segment);
    const locationText = segmentMappedLocationText(segment);
    const manualContinuity = String(segment?.continuity || "").trim();
    const lyricSingers = Array.isArray(segment?.lyric_singers)
      ? segment.lyric_singers.map((value) => String(value || "").trim()).filter(Boolean)
      : String(segment?.lyric_singers || "").split(/[,;\n]+/).map((value) => String(value || "").trim()).filter(Boolean);
    if (subjectText) parts.push(`Preserve the mapped character identity, wardrobe, hair, body proportions, accessories, and performance continuity exactly:\n${subjectText}`);
    else if (lyricSingers.length && !segment?.no_character_present) parts.push(`Preserve the visible performer identity for ${lyricSingers.join(", ")} consistently throughout the clip.`);
    if (locationText) parts.push(`Preserve the mapped location/environment, lighting, spatial orientation, materials, and atmosphere consistently:\n${locationText}`);
    if (manualContinuity) parts.push(manualContinuity);
    parts.push("Do not add, remove, duplicate, clone, or replace mapped subjects unless the creative shot action explicitly asks for it.");
    return `Continuity: ${parts.join("\n")}`;
  }

  function miniMaxH3FallbackShotDescription(segment, shotIndex = 0, mode = miniMaxH3ModeForSegment(segment)) {
    const cues = isMiniMaxSingerAssignmentMode(segment) && String(segment?.lyric_performance_mode || "together") === "cue_map"
      ? normalizeLyricCueMapForSegment(segment)
      : [];
    const cue = cues[shotIndex] || null;
    const labelMap = miniMaxH3SubjectLabelMapForSegment(segment, mode);
    const performers = selectedPerformerSubjectsForSegment(segment);
    const environment = "inside the mapped environment";
    if (cue?.type === "instrumental") {
      return normalizeMiniMaxH3ShotDescription(`An atmospheric cinematic shot shows the scene ${environment} during an instrumental passage. Every visible performer maintains a naturally relaxed closed mouth while the camera creates visual movement through posture, wind, clothing motion, and the surrounding environment.`);
    }
    if (cue) {
      const subject = performers.find((item) => String(item.id) === String(cue.singer_id)) || { id: cue.singer_id, name: cue.singer_name };
      const performer = miniMaxH3PerformerLabel(subject, labelMap);
      return normalizeMiniMaxH3ShotDescription(`A clear medium close-up shows only ${performer} ${environment}, with the face and mouth unobstructed while the camera stages the assigned performance moment. Other visible performers remain silent and naturally reactive.`);
    }
    const subject = performers[0] ? miniMaxH3PerformerLabel(performers[0], labelMap) : "the mapped performer";
    return normalizeMiniMaxH3ShotDescription(`A cinematic shot shows ${subject} ${environment}, preserving identity, wardrobe, lighting, and location continuity while the camera stages a clear music-video performance moment.`);
  }

  function parseMiniMaxH3ShotDescriptionPayload(rawPrompt, cutPlan = {}, segment = null, mode = miniMaxH3ModeForSegment(segment)) {
    const text = String(rawPrompt || "").trim();
    if (!text) throw new Error("The LLM returned an empty MiniMax shot-description payload.");
    const extractLeadingJson = (value) => {
      const source = String(value || "").trim();
      const start = source.indexOf("{");
      if (start < 0) return source;
      let depth = 0;
      let inString = false;
      let escaped = false;
      for (let index = start; index < source.length; index += 1) {
        const ch = source[index];
        if (inString) {
          if (escaped) escaped = false;
          else if (ch === "\\") escaped = true;
          else if (ch === "\"") inString = false;
          continue;
        }
        if (ch === "\"") {
          inString = true;
          continue;
        }
        if (ch === "{") depth += 1;
        else if (ch === "}") {
          depth -= 1;
          if (depth === 0) return source.slice(start, index + 1);
        }
      }
      return source;
    };
    const jsonText = extractLeadingJson(text);
    let parsed = null;
    try {
      parsed = JSON.parse(jsonText);
    } catch (error) {
      throw new Error(`The LLM did not return valid JSON shot descriptions. Raw output:\n${text}`);
    }
    const shotPlan = miniMaxH3OfficialShotPlan(cutPlan);
    const rawShots = Array.isArray(parsed?.shots) ? parsed.shots : [];
    if (rawShots.length > shotPlan.length) {
      throw new Error(`The LLM returned ${rawShots.length} shot description${rawShots.length === 1 ? "" : "s"}, but the builder expected ${shotPlan.length}.`);
    }
    const normalizedShots = rawShots.slice();
    if (normalizedShots.length < shotPlan.length) {
      const missingCount = shotPlan.length - normalizedShots.length;
      toast(`The LLM returned ${normalizedShots.length} of ${shotPlan.length} shot descriptions; Builder filled ${missingCount} missing shot${missingCount === 1 ? "" : "s"} so the scheduled cuts remain intact.`, true);
      while (normalizedShots.length < shotPlan.length) normalizedShots.push({});
    }
    const descriptions = normalizedShots.map((item, index) => {
      const description = typeof item === "string" ? item : String(item?.description || item?.text || item?.shot || "").trim();
      if (!description) {
        toast(`Gemma returned a blank description for shot ${index + 1}; Builder filled it from the singer cue map.`, true);
        return miniMaxH3FallbackShotDescription(segment, index, mode);
      }
      if (/\[\s*Shot\s+\d+\s*\]/i.test(description) || /\bAt\s+\d{1,2}:\d{2}(?:\.\d{1,3})?\b/i.test(description)) {
        throw new Error(`The LLM included shot labels or cut times inside shot ${index + 1}. Generate again so the builder can own labels/timing.`);
      }
      if (/\.\s+guides\s+(?:his|her|their|the)\s+exact\s+appearance\b/i.test(description)) {
        throw new Error(`Gemma returned an orphaned reference-purpose fragment in shot ${index + 1}. Generate again so every sentence has a clear subject.`);
      }
      const positiveDescription = stripMiniMaxH3NegativePromptSentences(description);
      if (!positiveDescription) {
        console.warn(`[VRGDG Music Builder] Removed all negative prompt wording from shot ${index + 1}; using the positive fallback shot.`);
        return miniMaxH3FallbackShotDescription(segment, index, mode);
      }
      if (positiveDescription !== description) {
        console.warn(`[VRGDG Music Builder] Removed negative prompt wording from shot ${index + 1} and continued with its positive visual instructions.`);
      }
      return normalizeMiniMaxH3ShotDescription(positiveDescription);
    });
    const requiredExtras = miniMaxH3CombinedSubjectPlan(segment, mode).subjects.filter((item) => item.kind === "extra");
    const normalizedDescriptions = descriptions.map((description, index) => {
      let clean = description;
      for (const extra of requiredExtras) {
        clean = clean.replace(new RegExp(`${escapeRegExp(extra.label)}\\s*\\(S\\d+\\)`, "gi"), extra.label);
      }
      return clean;
    });
    return miniMaxH3InjectMissingExtraLabels(normalizedDescriptions, requiredExtras);
  }

  function enforceMiniMaxH3CueOnShotDescription(segment, description, shotIndex, mode = miniMaxH3ModeForSegment(segment)) {
    let text = normalizeMiniMaxH3ShotDescription(description);
    if (miniMaxH3FrameContinuityPromptEnabled(segment)) return text;
    if (!isMiniMaxSingerAssignmentMode(segment) || String(segment?.lyric_performance_mode || "together") !== "cue_map") return text;
    const cues = normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
    const cue = cues[shotIndex];
    if (!cue) return text;
    const performers = selectedPerformerSubjectsForSegment(segment);
    const labelMap = miniMaxH3SubjectLabelMapForSegment(segment, mode);
    const subject = performers.find((item) => String(item.id) === String(cue.singer_id))
      || performers.find((item) => String(item.name || "").toLowerCase() === String(cue.singer_name || "").toLowerCase())
      || performers[0]
      || { id: cue.singer_id, name: cue.singer_name || "Singer" };
    const performer = miniMaxH3PerformerLabel(subject, labelMap);
    const vocalMarker = /<d>|<\/d>|\b(?:sings?|singing|sung|speaks?|speaking|says?|saying|performs?|performing|dialogue|lip[ -]?sync(?:s|ing|ed)?|lyrics?|vocals?|instrumental|no[- ]?vocal|mouth\s+(?:moves?|moving|shapes?|shaping|articulates?|articulating)|lips?\s+(?:moves?|moving|shapes?|shaping)|jaw\s+(?:moves?|moving|shapes?|shaping|articulates?|articulating))\b/i;
    if (cue.type === "instrumental") {
      const sentences = text.match(/[^.!?…]+[.!?…]+|[^.!?…]+$/g) || [];
      let clean = sentences.filter((sentence) => !vocalMarker.test(sentence)).join(" ").replace(/\s+/g, " ").trim();
      if (!clean) clean = miniMaxH3FallbackShotDescription(segment, shotIndex, mode);
      return normalizeMiniMaxH3ShotDescription(clean);
    }
    // The application owns the exact lyric placement, but the LLM's shot
    // sentence also contains valuable blocking, camera, and ensemble action.
    // Do not delete a whole sentence based on a vocal word: quoted lyric
    // punctuation can make a sentence splitter leave fragments such as
    // `"; S2 plays bass...`. Remove only the duplicate lyric/tag/timing
    // material, then append the one canonical cue below.
    const cueText = miniMaxH3CapitalizeCueText(miniMaxH3PunctuatedCueText(cue.text));
    // Remove a previously generated canonical contract before adding the
    // authoritative one below. This keeps retries idempotent.
    text = text.replace(/<Subject\s+\d+>[^.\n]*?is the only visible performer singing and lip-syncing[\s\S]*?every other visible performer remains silent\.?/gi, "");
    const cueVariants = [String(cue.text || "").trim(), cueText.replace(/[.!?…]+$/, "").trim()]
      .filter(Boolean)
      .sort((left, right) => right.length - left.length);
    for (const variant of cueVariants) {
      const escapedCue = escapeRegExp(variant);
      const timing = "(?:\\d+(?:\\.\\d+)?s?\\s*(?:to|[–—-])\\s*\\d+(?:\\.\\d+)?s?)";
      const vocalVerb = "(?:lip[ -]?sync(?:s|ing|ed)?|sing(?:s|ing)?|perform(?:s|ing|ed)?|speak(?:s|ing)?)";
      // Preserve the sentence and its timing, replacing only the repeated
      // lyric wording. For example: `she lip-syncs only \"line\" from
      // 0.940s–1.800s` becomes `she performs the assigned vocal cue from
      // 0.940s–1.800s`.
      text = text
        .replace(new RegExp(`\\b${vocalVerb}\\s+(?:only\\s+)?[“"']\\s*${escapedCue}[.!?…]*\\s*[”"']\\s+from\\s+(${timing})`, "gi"), "performs the assigned vocal cue from $1")
        .replace(new RegExp(`<d>\\s*(?:\\[[^\\]]+\\]\\s*)?${escapedCue}[.!?…]*\\s*<\\/d>`, "gi"), "the assigned vocal cue")
        .replace(new RegExp(`[“"']\\s*${escapedCue}[.!?…]*\\s*[”"']`, "gi"), "the assigned vocal cue");
    }
    let clean = text
      .replace(/<d>[\s\S]*?<\/d>/gi, "the assigned vocal cue")
      .replace(/\s+([,.;!?])/g, "$1")
      .replace(/([.!?])\s*;+/g, "$1")
      .replace(/\s{2,}/g, " ")
      .replace(/\s+([.!?])\s*([.!?])/g, "$1")
      .trim();
    if (!clean) clean = miniMaxH3FallbackShotDescription(segment, shotIndex, mode);
    clean = clean.replace(/[,;:]\s*([.!?…])/g, "$1").replace(/\s+,/g, ",");
    const vocalContract = `${performer} is the only visible performer singing and lip-syncing <d>[English] ${cueText}</d> from <Audio 1>; every other visible performer remains silent.`;
    return normalizeMiniMaxH3ShotDescription(`${clean.replace(/[.!?…]+$/g, "").trim()}. ${vocalContract}`);
  }

  // The saved prompt must say what the character sings, so the lyric goes in the shot in double quotes. The LLM is asked
  // to do this; this quotes a plain copy of the lyric, or adds the sentence, so it never depends on the LLM. Lyric lines
  // are shared across the cuts in order. Only plain singing scenes on the project audio are touched.
  function miniMaxH3EnsureQuotedLyricInShot(segment, description, shotIndex, shotCount, mode = miniMaxH3ModeForSegment(segment)) {
    const text = String(description || "");
    if (miniMaxH3FrameContinuityPromptEnabled(segment)) return text;
    if (segmentUsesNoLipSyncPerformance(segment) || segment?.no_character_present) return text;
    if (miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio") return text;
    if (isMiniMaxSingerAssignmentMode(segment) && String(segment?.lyric_performance_mode || "together") === "cue_map") return text;
    if (isInstrumentalLyricText(segment?.lyric_text)) return text;
    const lines = String(segment?.lyric_text || "").split(/\r?\n/)
      .map((line) => line.replace(/\s+/g, " ").trim())
      .filter((line) => line && !/^\[[^\]]*\]$/.test(line));
    if (!lines.length) return text;
    const count = Math.max(1, Number(shotCount) || 1);
    const size = Math.floor(lines.length / count);
    const extra = lines.length % count;
    let start = 0;
    for (let index = 0; index < shotIndex; index += 1) start += size + (index < extra ? 1 : 0);
    const chunk = lines.slice(start, start + size + (shotIndex < extra ? 1 : 0)).join(" ");
    if (!chunk) return text;
    const words = (value) => String(value || "").toLowerCase().replace(/’/g, "'").match(/[a-z0-9']+/g) || [];
    const chunkWords = words(chunk);
    if (!chunkWords.length) return text;
    const lead = chunkWords.slice(0, 6).join(" ");
    for (const match of text.matchAll(/["“]([^"”]+)["”]/g)) {
      if (words(match[1]).join(" ").includes(lead)) return text;
    }
    const plain = new RegExp(chunkWords.map((word) => escapeRegExp(word)).join("[\\W_]*"), "i").exec(text.replace(/’/g, "'"));
    if (plain) {
      return `${text.slice(0, plain.index)}"${text.slice(plain.index, plain.index + plain[0].length).trim()}"${text.slice(plain.index + plain[0].length)}`;
    }
    const performers = selectedPerformerSubjectsForSegment(segment);
    const label = performers.length ? miniMaxH3PerformerLabel(performers[0], miniMaxH3SubjectLabelMapForSegment(segment, mode)) : "The singer";
    return `${text.replace(/\s+$/g, "")} ${label} sings the lyric line, "${chunk.replace(/[ ,;]+$/g, "")}".`.trim();
  }

  function miniMaxH3OfficialShotBodyFromDescriptions(segment, descriptions = [], mode = miniMaxH3ModeForSegment(segment)) {
    const cutPlan = miniMaxH3CutPlanForSegment(segment);
    const shotPlan = miniMaxH3OfficialShotPlan(cutPlan);
    if (descriptions.length !== shotPlan.length) {
      throw new Error(`Cannot assemble MiniMax shots: expected ${shotPlan.length} description${shotPlan.length === 1 ? "" : "s"}, got ${descriptions.length}.`);
    }
    let body = shotPlan.map((shot, index) => {
      const description = miniMaxH3EnsureQuotedLyricInShot(segment, enforceMiniMaxH3CueOnShotDescription(segment, descriptions[index], index), index, shotPlan.length, mode);
      if (shot.number === 1) return `[Shot 1] ${description}`.trim();
      const postCutDescription = miniMaxH3PostCutShotText(description).replace(/^\s*([a-z])/, (_match, letter) => letter.toUpperCase());
      return `[Shot ${shot.number}] At ${shot.timecode}, ${postCutDescription}`.trim();
    }).join("\n\n");
    if (normalizeMiniMaxH3Mode(mode) === "image_reference_to_video") {
      body = body.replace(/^\[Shot 1\]\s*/, "[Shot 1] The shot begins exactly from <Picture 1>. ");
      const endFrame = firstLastFrameEndImageSource(segment);
      if (endFrame?.path || endFrame?.data) body = `${body.trim()} The final motion converges exactly on <Picture 2> as the last frame.`;
    }
    return body;
  }

  function miniMaxH3OfficialReferencePlan(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    const cutPlan = miniMaxH3CutPlanForSegment(segment);
    const shotList = miniMaxH3OfficialShotPlan(cutPlan).map((shot) => `[Shot ${shot.number}]`).join(", ");
    const pictureItems = normalizedMode === "image_to_video"
      ? [{ kind: "start_frame", label: "opening frame", description: "the exact first frame and visual composition anchor" }]
      : normalizedMode === "image_reference_to_video"
        ? miniMaxH3ImageReferencePromptItems(segment)
      : (normalizedMode === "reference_to_video" || normalizedMode === "video_to_video")
        ? miniMaxOrderedImageReferenceItemsForSegment(segment, normalizedMode)
        : [];
    let subjectNumber = 0;
    const pictureDefinitions = [];
    const subjectDefinitions = [];
    const retention = [];
    const subjects = [];
    pictureItems.forEach((item, index) => {
      const pictureLabel = `<Picture ${index + 1}>`;
      const name = miniMaxH3CleanSubjectNoun(item?.label || item?.name || `${item?.kind === "location" ? "location" : "reference"} ${index + 1}`);
      const isLocation = item?.kind === "location";
      const displayName = isLocation ? "environment" : name;
      const description = String(item?.description || "").trim();
      const purpose = miniMaxReferencePurposeText(item, segment);
      if (item?.kind === "start_frame") {
        pictureDefinitions.push(`${pictureLabel} is the first frame of [Shot 1], used as the exact opening composition and visual-state anchor.`);
        retention.push(`${pictureLabel} ([Shot 1] first frame): fully_preserved - the opening composition, subject placement, environment, lighting, and visual state are retained as the starting frame.`);
        return;
      }
      if (item?.kind === "end_frame") {
        const finalShot = miniMaxH3OfficialShotPlan(cutPlan).slice(-1)[0]?.number || 1;
        pictureDefinitions.push(`${pictureLabel} is the last frame of [Shot ${finalShot}], used as the exact final composition and visual-state anchor.`);
        retention.push(`${pictureLabel} ([Shot ${finalShot}] last frame): fully_preserved - the final composition, subject placement, environment, lighting, and visual state are reached exactly at the end.`);
        return;
      }
      subjectNumber += 1;
      const subjectLabel = `<Subject ${subjectNumber}>`;
      const noun = isLocation
        ? "environment"
        : item?.kind === "subject" || item?.kind === "extra"
          ? displayName
          : `${displayName} reference`;
      const nounPhrase = /^(?:the|a|an)\s+/i.test(noun) ? noun : `the ${noun}`;
      const isMainPictureSubject = item?.kind === "subject" && subjectNumber === 1;
      const compactDefinition = isMainPictureSubject
        ? ""
        : item?.kind === "extra"
        ? miniMaxH3CompactExtraIdentity(description, displayName, 130)
        : miniMaxH3CompactReferenceDescription(description, 240);
      const compactRetention = item?.kind === "extra"
        ? "identity and wardrobe"
        : item?.kind === "subject"
          ? "reference identity, face, hair, wardrobe, and accessories"
          : miniMaxH3CompactReferenceDescription(description, 140);
      subjectDefinitions.push(isMainPictureSubject
        ? `${subjectLabel} is ${nounPhrase} in ${pictureLabel}; ${pictureLabel} is the visual authority for ${purpose}.`
        : `${subjectLabel} is ${nounPhrase} in ${pictureLabel}, used as ${purpose}${compactDefinition ? `: ${compactDefinition}.` : "."}`);
      retention.push(item?.kind === "extra"
        ? `${subjectLabel} (appears in ${shotList || "[Shot 1]"}, visible when framing permits): fully_preserved - ${compactRetention} remain consistent.`
        : item?.kind === "subject"
          ? `${subjectLabel} (appears in ${shotList || "[Shot 1]"}): fully_preserved - ${compactRetention} remain consistent.`
          : `${subjectLabel} (appears in ${shotList || "[Shot 1]"}): fully_preserved - ${compactRetention || displayName} and its reference identity remain consistent.`);
      subjects.push({
        label: subjectLabel,
        name: displayName,
        kind: item?.kind || "reference",
        description,
        pictureLabel,
        extraId: item?.kind === "extra" ? String(item?.source_id || "") : "",
      });
    });
    const videoDefinitions = miniMaxH3VideoAssignmentLines(segment).map((line) => line.replace(/^<Video\s+(\d+)>\s*:\s*/, "<Video $1> is a reference video used as "));
    const videoRetention = miniMaxH3VideoAssignmentLines(segment).map((line, index) => {
      const purpose = line.replace(/^<Video\s+\d+>\s*:\s*/, "").trim();
      return `<Video ${index + 1}> (reference-video structure): weak_reference - ${purpose}`;
    });
    return {
      subjects,
      subjectDefinitions,
      pictureDefinitions,
      videoDefinitions,
      retention: [...retention, ...videoRetention],
    };
  }

  function miniMaxH3CombinedSubjectPlan(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    const plan = miniMaxH3OfficialReferencePlan(segment, normalizedMode);
    if (!["reference_to_video", "image_reference_to_video", "video_to_video"].includes(normalizedMode)) return plan;
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const mappedExtras = logicalExtraSubjectsForScene(refs, segment);
    if (!mappedExtras.length) return plan;
    const cutPlan = miniMaxH3CutPlanForSegment(segment);
    const shotList = miniMaxH3OfficialShotPlan(cutPlan).map((shot) => `[Shot ${shot.number}]`).join(", ") || "[Shot 1]";
    let subjectNumber = plan.subjects.length;
    const textOnly = [];
    for (const { extra, interaction } of mappedExtras) {
      const imageBacked = plan.subjects.find((item) => item.kind === "extra" && String(item.extraId || "") === String(extra.id || ""));
      if (imageBacked) {
        imageBacked.interaction = interaction;
        imageBacked.count = Math.max(1, Math.min(100, Math.round(Number(extra.count) || 1)));
        continue;
      }
      textOnly.push({ extra, interaction });
    }
    const groupedRoles = new Set(["background", "background_dancing", "alongside", "dancing_with", "direct"]);
    const ensembleTitleForRole = (interaction, count) => {
      const plural = count > 1;
      if (interaction === "background_dancing") return plural ? "backup-dancer ensemble" : "backup dancer";
      if (interaction === "alongside") return plural ? "alongside-dancer ensemble" : "alongside dancer";
      if (interaction === "dancing_with") return plural ? "partnered-dance ensemble" : "partnered dancer";
      if (interaction === "direct") return plural ? "direct-interaction ensemble" : "directly interacting performer";
      return plural ? "background-cast ensemble" : "background performer";
    };
    for (const interaction of groupedRoles) {
      const members = textOnly.filter((item) => item.interaction === interaction);
      if (!members.length) continue;
      subjectNumber += 1;
      const label = `<Subject ${subjectNumber}>`;
      const count = members.reduce((total, item) => total + Math.max(1, Math.min(100, Math.round(Number(item.extra.count) || 1))), 0);
      const title = ensembleTitleForRole(interaction, count);
      const distinctions = members.map(({ extra }) => {
        const name = miniMaxH3ExtraDisplayTitle(extra.title || "performer");
        const identity = miniMaxH3CompactExtraIdentity(extra.description, extra.title, 90);
        return `${name}—${identity}`;
      });
      const description = distinctions.join("; ");
      plan.subjectDefinitions.push(`${label} is the ${title}${count > 1 ? ` of ${count}` : ""}: ${description}.`);
      plan.retention.push(`${label} (present throughout ${shotList}, visible when framing permits): fully_preserved - identity and wardrobe remain consistent.`);
      plan.subjects.push({
        label,
        name: title,
        kind: "extra",
        description,
        interaction,
        count,
        extraId: "",
        members: members.map(({ extra }) => String(extra.id || "")).filter(Boolean),
        pictureLabel: "",
      });
    }
    for (const { extra, interaction } of textOnly.filter((item) => !groupedRoles.has(item.interaction))) {
      subjectNumber += 1;
      const label = `<Subject ${subjectNumber}>`;
      const count = Math.max(1, Math.min(100, Math.round(Number(extra.count) || 1)));
      const title = miniMaxH3ExtraDisplayTitle(extra.title || "background performer");
      const description = miniMaxH3CleanExtraDescription(extra.description, extra.title);
      const compactDefinition = miniMaxH3CompactExtraIdentity(description, extra.title, 150);
      plan.subjectDefinitions.push(`${label} is the individually tracked performer ${title}: ${compactDefinition}.`);
      plan.retention.push(`${label} (present throughout ${shotList}, visible when framing permits): fully_preserved - identity and wardrobe remain consistent.`);
      plan.subjects.push({ label, name: title, kind: "extra", description, interaction, count, extraId: extra.id, pictureLabel: "" });
    }
    return plan;
  }

  function segmentMappedExtraSubjectText(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    if (!["reference_to_video", "image_reference_to_video", "video_to_video"].includes(normalizedMode)) return "";
    const extras = miniMaxH3CombinedSubjectPlan(segment, normalizedMode).subjects.filter((item) => item.kind === "extra");
    if (!extras.length) return "";
    const lines = extras.map((item) => {
      const groupText = item.count > 1 ? `group of ${item.count}` : "single background performer";
      let interactionText = "Background only; no interaction with the main subject.";
      if (item.interaction === "background_dancing") {
        interactionText = "Active backup choreography in the background; no interaction with the main subject.";
      } else if (!segment?.no_character_present && item.interaction === "alongside") {
        interactionText = "Coordinated dancing alongside the main subject; no physical contact.";
      } else if (!segment?.no_character_present && item.interaction === "dancing_with") {
        interactionText = "REQUIRED SENSUAL ADULT CLUB DANCING WITH <Subject 1>: every member must dance physically with the main subject, alternating across shots when needed. Show unmistakable nightclub partner movement such as close hip-led grinding, dancing behind her with hands visibly placed at her hips or waist, or face-to-face body-close dancing. Keep it sexy and performance-oriented. Contact among extras does not count; do not partner extras with one another. Do not use 'no contact', 'without touching', 'non-contact', nearby-only staging, formal ballroom holds, or generic synchronized choreography for this subject.";
      } else if (!segment?.no_character_present && item.interaction === "direct") {
        interactionText = "REQUIRED CONTACT WITH <Subject 1>: every member must physically interact specifically with the main subject, alternating across shots when needed, through an unmistakable scene-appropriate touch. Contact among extras does not count; do not direct the mapped interaction toward one another. Do not use 'no contact', 'without touching', 'non-contact', or substitute a nearby gesture.";
      }
      return `${item.label} (${item.name}, ${groupText}): ${item.description}. Direction: ${interactionText}`;
    });
    const labels = extras.map((item) => item.label).join(", ");
    lines.push(`SHARED EXTRA CONTINUITY — MANDATORY: ${labels} remain continuously present across every cut. Show each in every shot where angle, framing, and occlusion reasonably permit; close-ups, inserts, obstructed angles, or viewpoints that exclude an extra may leave that extra off-screen without implying an exit. Preserve locked appearance and reasonable spatial continuity whenever visible again. Every listed label must appear exactly as written in at least one shot where that extra is visible and in every later visible reference; vague phrases such as dancers, guests, crowd, or people do not replace labels. Extras do not sing or speak and must never receive an (S1), (S2), or other speaker suffix. Do not restate full appearances or invent additional named/principal background characters.`);
    return lines.join("\n");
  }

  function miniMaxH3OfficialAudioDefinition(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    if (settings.audio_mode === "built_in_audio") return "";
    // Keep the audio definition deliberately standalone.  Cue timing and
    // singer assignments belong in the shot descriptions, never on the
    // <Audio 1> definition line in subject_definitions.
    return "<Audio 1> is the complete synchronized song and vocal track for the target video, reused as the target video's complete final soundtrack and timing reference.";
  }

  function miniMaxH3OfficialSummary(segment, mode, refs) {
    const settings = miniMaxH3SettingsForSegment(segment);
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    const taskTypes = [];
    if (normalizedMode === "video_to_video") taskTypes.push("video editing");
    if (normalizedMode === "image_to_video" || normalizedMode === "image_reference_to_video") taskTypes.push("keyframe completion");
    if (normalizedMode === "reference_to_video" || normalizedMode === "image_reference_to_video" || refs.subjects.length) taskTypes.push("reference generation");
    if (settings.audio_mode === "input_audio") taskTypes.push("audio reuse");
    if (!taskTypes.length) taskTypes.push("text generation");
    const subjectNames = refs.subjects.map((item) => item.kind === "location" ? `${item.label} (environment)` : `${item.label} (${item.name})`);
    const subjectText = subjectNames.length ? subjectNames.join(" and ") : "the described target scene";
    const audioText = settings.audio_mode === "input_audio"
      ? " <Audio 1> is reused as the complete soundtrack and timing reference."
      : " MiniMax generates the native audio requested by the scene.";
    return `[${Array.from(new Set(taskTypes)).join(" + ")}] The target video is a ${miniMaxH3OpeningStyle(segment)} scene featuring ${subjectText}.${audioText}`;
  }

  function miniMaxH3OpeningStyle(segment) {
    return String(segment?.minimax_h3_video_style || state.builderStoryboardDefaults?.video_style || "").trim() || "photorealistic cinematic";
  }

  function miniMaxH3OfficialSoundscape(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    if (settings.audio_mode === "input_audio") {
      return "overall_soundscape:\n<Audio 1> remains the sole complete audience-facing soundtrack with its original mix, timing, vocals, music, and dynamics intact.";
    }
    const audioDirection = String(segment?.audio_direction || "").trim();
    return `overall_soundscape:\n${audioDirection || "Subtle location-appropriate ambience, physical movement sounds, breathing, and expressive performance sounds support the scene."}`;
  }

  function miniMaxH3OfficialMusic(segment) {
    const settings = miniMaxH3SettingsForSegment(segment);
    if (settings.audio_mode === "input_audio") {
      return "non_diegetic_music:\n<Audio 1> is reused as the complete audience-facing song/music track.";
    }
    return "non_diegetic_music:\nThe scene's native soundtrack follows the requested musical and atmospheric direction.";
  }

  function assertValidMiniMaxH3FinalPrompt(prompt, segment, mode = miniMaxH3ModeForSegment(segment), options = {}) {
    const text = String(prompt || "").trim();
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    if (!text) throw new Error("The assembled MiniMax H3 prompt is empty.");
    assertMiniMaxH3ReferenceCapacity(segment, normalizedMode);
    if (text.length > 7000) {
      const error = new Error(`The MiniMax H3 prompt is ${text.length} characters, exceeding the 7,000-character maximum by ${text.length - 7000}.`);
      error.code = "MINIMAX_H3_PROMPT_TOO_LONG";
      error.promptLength = text.length;
      error.promptLimit = 7000;
      throw error;
    }
    if (isMiniMaxSingerAssignmentMode(segment) && String(segment?.lyric_performance_mode || "together") === "cue_map") {
      const cueMap = normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
      const creativeHeader = ["text_to_video", "image_to_video"].includes(normalizedMode)
        ? "integrated_multimodal_description:"
        : "detailed_description:";
      const creativeStart = text.indexOf(creativeHeader);
      const soundscapeStart = text.indexOf("overall_soundscape:", creativeStart);
      const cueCreativeText = creativeStart >= 0
        ? (soundscapeStart > creativeStart ? text.slice(creativeStart + creativeHeader.length, soundscapeStart) : text.slice(creativeStart + creativeHeader.length)).trim()
        : "";
      const normalizeCueTag = (value) => String(value || "")
        .replace(/\\</g, "<")
        .replace(/\\>/g, ">")
        .replace(/\\+/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .toLocaleLowerCase();
      const reportCueMismatch = (warning, failure) => {
        if (options.allowCueValidationWarnings) {
          console.warn(`[VRGDG Music Builder] ${warning}`);
          if (typeof options.onCueValidationWarning === "function") options.onCueValidationWarning(warning);
        } else {
          throw new Error(failure);
        }
      };
      const isSingleContinuousShot = miniMaxH3OfficialShotPlan(miniMaxH3CutPlanForSegment(segment)).length === 1;
      if (isSingleContinuousShot && cueMap.length > 1) {
        // A continuous shot may contain several vocal cues and may also allow instrumental intervals to coexist in the same shot.
        const expectedTags = cueMap
          .filter((cue) => cue.type === "vocal")
          .map((cue) => `<d>[English] ${miniMaxH3CapitalizeCueText(miniMaxH3PunctuatedCueText(cue.text))}</d>`);
        const actualTags = cueCreativeText.match(/<d>[\s\S]*?<\/d>/gi) || [];
        const tagsMatch = actualTags.length === expectedTags.length
          && expectedTags.every((tag, index) => normalizeCueTag(actualTags[index]) === normalizeCueTag(tag));
        if (!tagsMatch) {
          const warning = `continuous-shot lyric cue mismatch. Expected ${expectedTags.join(", ") || "no <d> cues"}; found ${actualTags.join(", ") || "no <d> cues"}.`;
          reportCueMismatch(warning, `The continuous shot must contain its assigned lyric cues in order: ${expectedTags.join(", ") || "none"}`);
        }
      } else cueMap.forEach((cue, index) => {
        const shotLabel = `[Shot ${index + 1}]`;
        const shotStart = cueCreativeText.indexOf(shotLabel);
        const nextShotStart = cueCreativeText.indexOf(`[Shot ${index + 2}]`, shotStart + shotLabel.length);
        const shotText = shotStart >= 0
          ? cueCreativeText.slice(shotStart, nextShotStart >= 0 ? nextShotStart : cueCreativeText.length)
          : "";
        const dialogueTags = shotText.match(/<d>[\s\S]*?<\/d>/gi) || [];
        if (cue.type === "instrumental" && dialogueTags.length) {
          throw new Error(`${shotLabel} is assigned as instrumental but contains sung dialogue.`);
        }
        if (cue.type === "vocal") {
          const expectedTag = `<d>[English] ${miniMaxH3CapitalizeCueText(miniMaxH3PunctuatedCueText(cue.text))}</d>`;
          const actualTag = dialogueTags.length === 1 ? dialogueTags[0] : "";
          if (dialogueTags.length !== 1 || normalizeCueTag(actualTag) !== normalizeCueTag(expectedTag)) {
            const warning = `${shotLabel} lyric cue mismatch. Expected ${expectedTag}; found ${actualTag || "no <d> cue"}.`;
            reportCueMismatch(warning, `${shotLabel} must contain exactly its assigned lyric cue: ${expectedTag}`);
          }
        }
      });
    }
    if (!state.failOnInvalidPromptFormats) return text;
    const headers = ["text_to_video", "image_to_video"].includes(normalizedMode)
      ? ["integrated_multimodal_description:"]
      : ["detailed_description:"];
    for (const header of headers) {
      if (!text.includes(header)) throw new Error(`The MiniMax H3 prompt is incomplete: missing required ${header.slice(0, -1)} section.`);
    }
    const allPossibleHeaders = ["text_to_video", "image_to_video"].includes(normalizedMode)
      ? ["integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"]
      : ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:", "non_diegetic_music:"];
    let previousIndex = -1;
    for (const header of allPossibleHeaders) {
      const index = text.indexOf(header);
      if (index >= 0) {
        if (index <= previousIndex) throw new Error(`The MiniMax H3 prompt has required sections out of order near ${header}`);
        previousIndex = index;
      }
    }
    const creativeHeader = ["text_to_video", "image_to_video"].includes(normalizedMode)
      ? "integrated_multimodal_description:"
      : "detailed_description:";
    const creativeStart = text.indexOf(creativeHeader);
    const soundscapeStart = text.indexOf("overall_soundscape:", creativeStart);
    const creativeText = creativeStart >= 0
      ? (soundscapeStart > creativeStart ? text.slice(creativeStart + creativeHeader.length, soundscapeStart) : text.slice(creativeStart + creativeHeader.length)).trim()
      : "";
    const retentionStart = text.indexOf("retention_analysis:");
    const detailedStart = text.indexOf("detailed_description:");
    const definitionsText = (normalizedMode === "reference_to_video" || normalizedMode === "image_reference_to_video" || normalizedMode === "video_to_video") && text.includes("subject_definitions:") && text.includes("summary:")
      ? text.slice(text.indexOf("subject_definitions:") + "subject_definitions:".length, text.indexOf("summary:")).trim()
      : "";
    const retentionText = retentionStart >= 0 && detailedStart > retentionStart
      ? text.slice(retentionStart + "retention_analysis:".length, detailedStart).trim()
      : "";
    const configuredShotPlan = miniMaxH3OfficialShotPlan(miniMaxH3CutPlanForSegment(segment));
    // A prompt can outlive the cut-frequency/cue settings that produced it. In
    // that case the prompt's own numbered schedule is the authoritative render
    // contract; rejecting it because the current UI defaults changed produces
    // the misleading "unexpected [Shot N]" error at render time.
    const promptShotMatches = Array.from(creativeText.matchAll(/\[Shot\s+(\d+)\]/g));
    const promptShotNumbers = [...new Set(promptShotMatches.map((match) => Number(match[1])))].sort((a, b) => a - b);
    const promptShotPlan = promptShotNumbers.length && promptShotNumbers.every((number, index) => number === index + 1)
      ? promptShotNumbers.map((number) => {
        if (number === 1) return { number, time: 0, timecode: "00:00.000" };
        const labelStart = creativeText.indexOf(`[Shot ${number}]`);
        const timingMatch = creativeText.slice(labelStart).match(/^\[Shot\s+\d+\]\s+At\s+(\d{2}:\d{2}\.\d{3}),/m);
        return { number, time: 0, timecode: timingMatch?.[1] || "" };
      })
      : [];
    const shotPlan = promptShotPlan.length ? promptShotPlan : configuredShotPlan;
    for (const shot of shotPlan) {
      const matches = creativeText.match(new RegExp(`\\[Shot\\s+${shot.number}\\]`, "g")) || [];
      if (matches.length !== 1) {
        throw new Error(`The MiniMax H3 prompt is incomplete: expected exactly one [Shot ${shot.number}] block, found ${matches.length}.`);
      }
      if (shot.number > 1 && !new RegExp(`\\[Shot\\s+${shot.number}\\]\\s+At\\s+${escapeRegExp(shot.timecode)},`).test(creativeText)) {
        throw new Error(`The MiniMax H3 prompt is not compliant: [Shot ${shot.number}] must begin with its exact cut time, At ${shot.timecode},.`);
      }
    }
    const unexpectedShot = Array.from(creativeText.matchAll(/\[Shot\s+(\d+)\]/g)).find((match) => !shotPlan.some((shot) => shot.number === Number(match[1])));
    if (unexpectedShot) throw new Error(`The MiniMax H3 prompt contains unexpected [Shot ${unexpectedShot[1]}].`);
    const missingExtraLabels = miniMaxH3CombinedSubjectPlan(segment, normalizedMode).subjects
      .filter((item) => item.kind === "extra" && !creativeText.includes(item.label))
      .map((item) => item.label);
    if (missingExtraLabels.length) {
      throw new Error(`The MiniMax H3 prompt does not use mapped extra label${missingExtraLabels.length === 1 ? "" : "s"} ${missingExtraLabels.join(", ")} in any shot and was not saved or sent.`);
    }
    if (definitionsText) {
      const referencePlan = miniMaxH3CombinedSubjectPlan(segment, normalizedMode);
      const requiredLabels = [
        ...referencePlan.subjects.map((item) => item.label),
        ...referencePlan.pictureDefinitions.map((line) => line.match(/^<Picture\s+\d+>/)?.[0]).filter(Boolean),
        ...referencePlan.videoDefinitions.map((line) => line.match(/^<Video\s+\d+>/)?.[0]).filter(Boolean),
        miniMaxH3OfficialAudioDefinition(segment) ? "<Audio 1>" : "",
      ].filter(Boolean);
      const definitionLines = definitionsText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      const retentionLines = retentionText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      const missingDefinitions = requiredLabels.filter((label) => !definitionLines.some((line) => line.startsWith(label)));
      const missingRetention = requiredLabels.filter((label) => !retentionLines.some((line) => {
        if (!line.startsWith(label)) return false;
        return label.startsWith("<Audio ")
          ? /:\s*(?:fully_copy|partially_copy|reference|weak_reference)\s+-/i.test(line)
          : /:\s*(?:fully_preserved|partially_preserved|attribute_transfer|weak_reference)\s+-/i.test(line);
      }));
      if (missingDefinitions.length) throw new Error(`The MiniMax H3 prompt is not compliant: missing definition lines for ${missingDefinitions.join(", ")}.`);
      if (missingRetention.length) throw new Error(`The MiniMax H3 prompt is not compliant: missing retention lines for ${missingRetention.join(", ")}.`);
      if (/\(S\d+\)/.test(retentionText)) throw new Error("The MiniMax H3 prompt is not compliant: speaker IDs must not appear in retention_analysis.");
    }
    const malformedReference = text.match(/<(?:Subject|Picture|Video|Audio)\b[^>\n]{0,40}(?:$|\n)/im);
    if (malformedReference) throw new Error(`The MiniMax H3 prompt ends with or contains an incomplete reference label: ${malformedReference[0].trim()}`);
    const dialogueOpenCount = (text.match(/<d>/g) || []).length;
    const dialogueCloseCount = (text.match(/<\/d>/g) || []).length;
    if (dialogueOpenCount !== dialogueCloseCount) throw new Error("The MiniMax H3 prompt contains an incomplete <d> dialogue or lyric tag.");
    const invalidDialogue = Array.from(text.matchAll(/<d>([\s\S]*?)<\/d>/g)).find((match) => !/^\[[^\]\r\n]+\]\s+[^\r\n]+[.!?]\s*$/.test(match[1].trim()));
    if (invalidDialogue) throw new Error("The MiniMax H3 prompt is not compliant: every <d> cue must include a [Language] tag and end with punctuation before </d>.");
    const finalShotLabel = `[Shot ${shotPlan[shotPlan.length - 1]?.number || 1}]`;
    const finalShotStart = creativeText.indexOf(finalShotLabel);
    const finalShotText = finalShotStart >= 0
      ? creativeText.slice(finalShotStart + finalShotLabel.length).trim()
      : "";
    if (finalShotText.length < 24 || !/[.!?…](?:\s*<\/d>)?\s*$/.test(finalShotText)) {
      throw new Error("The MiniMax H3 prompt has an incomplete final shot description and was not saved or sent.");
    }
    if (/\.\s+guides\s+(?:his|her|their|the)\s+exact\s+appearance\b/i.test(text)) {
      throw new Error("The MiniMax H3 prompt contains an orphaned reference-purpose sentence fragment and was not saved or sent.");
    }
    return text;
  }

  // In the RefMod pipeline every reference is a saved RefMod. The prompt names the cast as <Subject n> in scene
  // order, and each RefMod's own label (<Video n> or <Picture n>) is put next to its character so Text Encode with
  // RefMods can tell them apart.
  function isRefmodPipelineActive() {
    return normalizeMiniMaxH3Pipeline(state.miniMaxH3Settings?.pipeline) === "refmod";
  }

  function relabelRefmodPrompt(segment, mode, text) {
    if (!isRefmodPipelineActive()) return text;
    const items = miniMaxOrderedImageReferenceItemsForSegment(segment, mode).map((item) => item.refmod).filter(Boolean);
    return attachRefmodLabels(text, items);
  }

  function assembleMiniMaxH3OfficialPromptFromCreative(segment, mode, creativePrompt) {
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    const cutPlan = miniMaxH3CutPlanForSegment(segment);
    const shotDescriptions = parseMiniMaxH3ShotDescriptionPayload(creativePrompt, cutPlan, segment, normalizedMode);
    const creative = miniMaxH3OfficialShotBodyFromDescriptions(segment, shotDescriptions, normalizedMode);
    if (!creative) {
      throw new Error("The LLM returned no creative MiniMax scene body. Try again, or add more scene notes so it has action/camera material to write.");
    }
    if (normalizedMode === "text_to_video" || normalizedMode === "image_to_video") {
      const prompt = [
        miniMaxH3OfficialIntegratedDescription(segment, normalizedMode, creative),
        miniMaxH3OfficialSoundscape(segment),
        miniMaxH3OfficialMusic(segment),
      ].filter(Boolean).join("\n\n").trim();
      return assertValidMiniMaxH3FinalPrompt(prompt, segment, normalizedMode);
    }
    const prompt = relabelRefmodPrompt(segment, normalizedMode, `detailed_description:\nThe target video is in a ${miniMaxH3OpeningStyle(segment)} music-video style.\n\n${creative}`.trim());
    return assertValidMiniMaxH3FinalPrompt(prompt, segment, normalizedMode);
  }

  function assembleMiniMaxH3PromptFromCreative(segment, mode, creativePrompt) {
    return assembleMiniMaxH3OfficialPromptFromCreative(segment, mode, creativePrompt);
  }

  function ensureBuilderManagedFx(prompt, scene = {}) {
    let text = String(prompt || "").trim();
    const defaults = normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults);
    const preset = String(defaults.fx_preset || "").trim();
    if (!text || !preset) return text;
    const timestampPattern = /((?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|\[\s*Shot\s+\d+[^\]]*\])\s*\n?)([\s\S]*?)(?=\n\s*(?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|\[\s*Shot\s+\d+[^\]]*\])|\n\s*(?:Audio(?:\s+1)?|Native\s+audio|overall_soundscape|non_diegetic_music|Continuity)\s*:|$)/gi;
    let shotIndex = 0;
    let matched = false;
    text = text.replace(timestampPattern, (whole, header, body) => {
      matched = true;
      const contract = storyboardFxContract(preset, defaults.fx_custom_json, shotIndex++);
      if (!contract || String(body || "").includes(contract.cue)) return whole;
      const cue = `FX accent: ${contract.cue} Timing: ${contract.timing}. Keep the mapped subject stable and readable.`;
      return `${header}${String(body || "").trim()} ${cue}\n`;
    });
    if (matched) return text.trim();
    const contract = storyboardFxContract(preset, defaults.fx_custom_json, 0);
    return contract ? `${text}\n\nFX accent inside this shot: ${contract.cue} Keep the mapped subject stable and readable.`.trim() : text;
  }

  function miniMaxH3PromptContextForSegment(segment, mode) {
    const settings = miniMaxH3SettingsForSegment(segment);
    const nativeAudio = settings.audio_mode === "built_in_audio";
    const duration = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    const exactDuration = String(Number(duration.toFixed(3)));
    const aspectRatio = String(settings.aspect_ratio || "16:9").match(/\d+\s*:\s*\d+/)?.[0]?.replace(/\s+/g, "") || "16:9";
    const visualOnly = segmentUsesNoLipSyncPerformance(segment);
    const dialogueAssignments = visualOnly ? [] : miniMaxDialogueAssignmentsForSegment(segment);
    const dialogueOrder = visualOnly ? "" : miniMaxDialogueOrderText(segment);
    const lyricText = dialogueAssignments.length
      ? dialogueAssignments.map((cue) => cue.text).join(" ")
      : isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
    const performanceMode = visualOnly
      ? "no_lip_sync"
      : normalizeVideoType(segment?.performance_mode || state.videoType);
    const singerNames = (dialogueAssignments.length
      ? dialogueAssignments.map((cue) => cue.speaker_name)
      : Array.isArray(segment?.lyric_singers)
        ? segment.lyric_singers
      : String(segment?.lyric_singers || "").split(/[,;\n]+/))
      .map((value) => String(value || "").trim())
      .filter(Boolean);
    const performerLabel = singerNames.length
      ? singerNames.join(singerNames.length === 2 ? " and " : ", ")
      : "the visible performer";
    const parts = [
      "MiniMax H3 Builder scene context:",
      `Mode: ${miniMaxH3ModeLabel(mode)}`,
      `Exact duration: ${exactDuration} seconds`,
      `Timeline range: ${Number(segment?.start || 0).toFixed(3)}s to ${Number(segment?.end || 0).toFixed(3)}s`,
      `Aspect ratio: ${aspectRatio}`,
      `Audio mode: ${nativeAudio ? "Built-in MiniMax Audio — generate native scene audio" : "Input Audio — preserve the supplied Audio 1 unchanged"}`,
      "Visual style: follow the supplied scene/theme context; default to photorealistic when no style is specified.",
    ];
    const cameraMotionSpeed = Number(segment?.camera_motion_speed ?? state.builderStoryboardDefaults?.camera_motion_speed ?? 4);
    const characterMotionSpeed = Number(segment?.character_motion_speed ?? state.builderStoryboardDefaults?.character_motion_speed ?? 4);
    const cameraMotionGuidance = String(segment?.camera_motion_speed_guidance || state.builderStoryboardDefaults?.camera_guidance || "").trim();
    const characterMotionGuidance = String(segment?.character_motion_guidance || state.builderStoryboardDefaults?.character_guidance || "").trim();
    const cutPlan = miniMaxH3CutPlanForSegment(segment);
    parts.push(
      `Camera motion speed: ${Number.isFinite(cameraMotionSpeed) ? cameraMotionSpeed : 4}/10.`,
      cameraMotionGuidance || "Follow the selected camera speed as a hard motion requirement.",
      `Character motion speed: ${Number.isFinite(characterMotionSpeed) ? characterMotionSpeed : 4}/10.`,
      characterMotionGuidance || "Include physical character action appropriate to the selected character-motion speed.",
      cutPlan.instruction,
    );
    const cueShotContract = miniMaxH3CueShotContractText(segment, mode);
    if (cueShotContract) parts.push(cueShotContract);
    if (cameraMotionSpeed >= 7) {
      parts.push("MANDATORY CAMERA RULE: use energetic, visibly active camera movement. Slow, gentle, subtle, restrained, locked-off, static, and hold camera language contradicts this setting and must not appear.");
    }
    if (characterMotionSpeed >= 4) {
      parts.push(cueShotContract
        ? "MANDATORY CHARACTER ACTION RULE: include at least one clear physical body action, gesture, step, or interaction with the set. Lip sync occurs only inside assigned vocal cue shots; instrumental cue shots must remain non-vocal."
        : "MANDATORY CHARACTER ACTION RULE: include at least one clear physical body action, gesture, step, or interaction with the set in addition to visible singing and lip sync. Facial expression, blinking, breathing, and mouth movement alone do not satisfy character motion, but do not omit or suppress the required lip sync.");
    }
    if (segment?.no_character_present) {
      parts.push(
        "Vocal performance: NO VISIBLE CHARACTER / NO LIP SYNC.",
        nativeAudio
          ? "Native audio assignment: generate only low environmental ambience and requested sound effects. Do not invent a singer, speaker, dialogue, lyrics, or voice."
          : "Audio 1 assignment: use the exact supplied custom/project audio segment unchanged for timing. Do not invent a visible singer or speaker.",
      );
    } else if (performanceMode === "no_lip_sync") {
      parts.push(
        "Vocal performance: VISUAL ONLY / NO LIP SYNC.",
        nativeAudio
          ? "Native audio assignment: generate only ambience and requested sound effects. Do not generate speech, singing, lyrics, dialogue, or visible mouth synchronization."
          : "Audio 1 assignment: use the exact supplied custom/project audio segment unchanged for timing, but do not show singing, speaking, or mouth synchronization.",
      );
    } else if (lyricText && performanceMode === "speaking") {
      parts.push(
        "Vocal performance: SPEAKING WITH EXACT DIALOGUE LIP SYNC.",
        dialogueOrder ? `Mandatory dialogue order:\n${dialogueOrder}` : `Exact dialogue line: “${lyricText}”`,
        nativeAudio
          ? `Native audio assignment: MiniMax generates the voices and audio. Follow the mandatory speaker assignment and dialogue order exactly. Synchronize each assigned speaker’s lips, mouth shapes, jaw movement, facial muscles, and breathing precisely to that speaker’s generated line. Do not merge speakers, reorder cues, replace, alter, repeat, extend, improvise, omit, or add words. Use silence, breathing, facial reaction, physical action, and low ambience for all remaining time.`
          : `Audio 1 assignment: use as the exact voice, timing, and lip-sync reference. ${performerLabel} says the exact line “${lyricText}”. Synchronize lips, mouth shapes, jaw movement, facial muscles, and breathing precisely to that spoken line in Audio 1. Do not replace, alter, extend, or add words.`,
      );
    } else if (lyricText && cueShotContract) {
      parts.push(
        "Vocal performance: TIMED CUE MAP ONLY.",
        "Audio 1 is the exact timing reference. For each vocal cue, the assigned performer uses only the exact words inside that cue and only during its stated time range. Use only visual action and camera direction for other cue shots. Never stretch, restart, anticipate, or repeat the full lyric across shots.",
      );
    } else if (lyricText) {
      parts.push(
        "MANDATORY VOCAL PERFORMANCE: The assigned subject is visibly singing the exact supplied lyric/audio during this scene. Show clear, natural mouth, lip, jaw, and facial movement synchronized to the audible vocal. Never describe the lips as closed, still, motionless, or sealed while the assigned vocal is being performed. Non-verbal vocals such as oooh, ah, humming, and sustained notes still require visible mouth movement.",
        `Exact lyric line to sing: “${lyricText}”`,
        `Audio 1 assignment: use as the exact vocal, timing, and lip-sync reference. ${performerLabel} is singing the exact line “${lyricText}”. Synchronize lips, mouth shapes, jaw movement, facial muscles, and breathing precisely only while that sung line is audible in Audio 1. Never stretch, restart, or repeat the full lyric across every timestamp. When the vocal ends, the mouth closes or relaxes naturally while physical action continues. Do not replace, alter, extend, or add vocals.`,
      );
    } else {
      parts.push(
        "Vocal performance: no exact lyric or dialogue is assigned to this scene.",
        nativeAudio
          ? "Native audio assignment: generate only ambience and requested sound effects. Do not invent lyrics, dialogue, or voices."
          : "Audio 1 assignment: use the exact supplied custom/project audio segment unchanged for performance, movement, and camera timing. Do not invent lyrics, dialogue, or replacement audio.",
      );
    }
    const add = (label, value) => {
      const text = String(value || "").trim();
      if (text) parts.push(`${label}:\n${text}`);
    };
    add("Scene idea / image prompt", castSafeText(segment, sceneVideoConceptPromptText(segment)));
    add("Scene notes", segment?.notes || segment?.director_note);
    add("Exact manual audio / sound direction", segment?.audio_direction);
    add("Exact manual continuity requirements", segment?.continuity);
    add("Director timeline note", segment?.timeline_note);
    add("Motion and camera request", segment?.i2v_notes);
    add("Scene story beat", castSafeText(segment, segment?.story_beat, true));
    add("Scene cast restriction (mandatory)", sceneCastRestrictionText(segment));
    add(performanceMode === "no_lip_sync" ? "Hidden lyric mood/story context — never performed" : "Exact supplied lyric or dialogue", lyricText);
    add("Lyric section", segment?.lyric_section);
    add("Visible character context", segment?.no_character_present ? "No main character is visible in this scene." : segmentMappedSubjectText(segment));
    if (["reference_to_video", "image_reference_to_video", "video_to_video"].includes(normalizeMiniMaxH3Mode(mode))) {
      add("Background extras context", segmentMappedExtraSubjectText(segment, mode));
    }
    add("Location context", segmentMappedLocationText(segment));
    const nativeVoiceBlock = miniMaxH3NativeVoiceBlock(segment);
    if (nativeVoiceBlock) parts.push(nativeVoiceBlock);

    if (mode === "image_to_video") {
      parts.push("Ordered image assignment:\nImage 1: exact start frame and authoritative opening composition, character identity, clothing, location, lighting, and visual-state anchor.");
    }
    if (mode === "reference_to_video" || mode === "image_reference_to_video" || mode === "video_to_video") {
      const items = mode === "image_reference_to_video"
        ? miniMaxH3ImageReferencePromptItems(segment)
        : miniMaxOrderedImageReferenceItemsForSegment(segment, mode);
      const promptInspiration = mode === "reference_to_video" && miniMaxH3SceneImageIsPromptInspiration(segment);
      const assignments = items.map((item, index) => {
        const detail = [item.label, item.description].map((value) => String(value || "").trim()).filter(Boolean).join(" — ");
        const attachmentMapping = promptInspiration ? ` (attached Picture ${index + 2})` : "";
        const purpose = item.kind === "end_frame"
          ? "exact end frame and authoritative final composition, subject placement, lighting, and visual-state anchor"
          : miniMaxReferencePurposeText(item, segment);
        return `Image ${index + 1}${attachmentMapping}: ${purpose}${detail ? `. ${detail}` : ""}`;
      });
      if (assignments.length) parts.push(`${mode === "video_to_video" ? "Ordered supporting edit-image assignments" : "Ordered image assignments"} (exact connected order):\n${assignments.join("\n")}`);
      if (promptInspiration) {
        const includesFraming = miniMaxH3SceneImageUseForSegment(segment) === "environment_framing_inspiration";
        parts.push(
          "PROMPT-ONLY SCENE-IMAGE INSPIRATION — MANDATORY:\n"
          + "Attached Picture 1 is shown only to the prompt-writing vision LLM. It is NOT supplied to the MiniMax H3 renderer, is NOT a start frame, is NOT a renderer reference, consumes no MiniMax reference slot, and must never be identified or mentioned as Image 1 or any Image N in the finished prompt. "
          + "Use Attached Picture 1 only to extract location and environment, atmosphere and mood, lighting and weather, colors, materials and background details, and relevant objects or environmental activity. "
          + (includesFraming
            ? "Camera framing, shot distance, camera angle, lens, and image composition may be used only as optional inspiration and may be changed freely to suit the requested scene. "
            : "Explicitly ignore camera framing and shot distance, camera angle and lens, and image composition. Choose new camera coverage freely from the scene request and camera-motion settings. ")
          + "Always ignore every visible character's identity, face, hair, body, clothing, accessories, pose, placement, and activity in Attached Picture 1. The assigned character reference images are the sole authority for the rendered character's complete identity and appearance. "
          + "In the finished prompt, convert permitted environmental observations into direct scene description without mentioning Attached Picture 1, inspiration, source imagery, or analysis. Renderer Image 1 is attached as Picture 2, Renderer Image 2 as Picture 3, and so on; use only the renderer Image N labels in the finished prompt.",
        );
      }
      if (["reference_to_video", "image_reference_to_video"].includes(mode) && segment?.minimax_h3_use_scene_image_as_start_frame) {
        const characterInfluence = miniMaxH3StartFrameCharacterInfluenceForSegment(segment);
        if (characterInfluence === "face_hair_only") {
          parts.push(
            "START-FRAME / CHARACTER-REFERENCE PRIORITY — MANDATORY:\n"
            + "Image 1 remains authoritative for the subject's pose, body and body proportions, clothing, wardrobe styling, accessories, framing, composition, camera angle, environment, props, lighting, colors, and every other visible detail. Character-reference images may override Image 1 ONLY for the subject's face identity, facial features, and hair. The literal first generated frame must already depict the character-reference face and hair within Image 1's otherwise unchanged visual setup. Describe the intended character as directly present from the first frame. Do not use temporal comparison language such as 'now featuring,' 'becomes,' 'changes into,' or 'replaced by'; never describe or show a face swap, replacement process, morph, transition, or transformation. Never import the character reference's clothing, body, pose, accessories, framing, lighting, or background.",
          );
        } else {
          parts.push(
            "START-FRAME / CHARACTER-REFERENCE PRIORITY — MANDATORY:\n"
            + "Image 1 controls the exact opening composition, pose, camera angle, environment, and lighting. Character-reference images control the subject's face, hair, clothing, body proportions, and identity details, including details hidden in Image 1.",
          );
        }
      }
      if (mode === "image_reference_to_video") {
        const endFrame = firstLastFrameEndImageSource(segment);
        const hasEndFrame = Boolean(endFrame?.path || endFrame?.data);
        parts.push(
          "IMAGE + REFERENCE FRAME AUTHORITY — MANDATORY:\n"
          + "Image 1 is the exact first frame at 0.00 seconds and controls the opening composition, camera angle, environment, lighting, and initial visual state. "
          + (hasEndFrame
            ? "Image 2 is the exact final frame and controls the ending composition, subject placement, lighting, and final visual state. Describe a continuous physical and camera path from Image 1 to Image 2. Images 3 and later are visual references only and must not be treated as timeline anchors."
            : "Images 2 and later are visual references only and must not be treated as timeline anchors. Develop the action continuously forward from Image 1."),
        );
      }
    }
    if (mode === "video_to_video") {
      const assignments = (Array.isArray(segment?.minimax_h3_video_references) ? segment.minimax_h3_video_references : [])
        .filter((item) => String(item?.path || "").trim())
        .slice(0, 3)
        .map((item, index) => {
          const purpose = normalizeMiniMaxH3VideoPurpose(item.purpose);
          const purposeLabel = MINIMAX_H3_VIDEO_REFERENCE_PURPOSES.find((option) => option.value === purpose)?.label || "Continuation / Extension";
          const timing = Number(item.start_seconds || 0) > 0 || Number(item.duration || 0) > 0
            ? ` Use from ${Math.max(0, Number(item.start_seconds || 0))}s${Number(item.duration || 0) > 0 ? ` for ${Math.max(0, Number(item.duration || 0))}s` : " onward"}.`
            : "";
          const audio = item.use_audio ? " Its embedded audio is also supplied as a paired reference." : " Do not use its embedded audio.";
          return `Video ${index + 1}: ${purposeLabel}.${timing}${audio}`;
        });
      if (assignments.length) parts.push(`Ordered video assignments (exact connected order):\n${assignments.join("\n")}`);
    }
    return parts.join("\n\n");
  }

  function miniMaxH3PromptVisionImages(segment, mode) {
    if (mode === "image_to_video") {
      const source = segmentImageSource(segment);
      const path = String(source?.path || selectedSegmentImagePath(segment) || "").trim();
      const data = String(source?.data || "").trim();
      return path || data ? [{ path, data }] : [];
    }
    if (mode === "image_reference_to_video") {
      return miniMaxH3ImageReferencePromptItems(segment)
        .map((item) => ({
          path: String(item?.image?.path || "").trim(),
          data: String(item?.image?.data || "").trim(),
        }))
        .filter((item) => item.path || item.data)
        .slice(0, 9);
    }
    if (mode === "reference_to_video" || mode === "video_to_video") {
      const rendererImages = miniMaxOrderedImageReferenceItemsForSegment(segment, mode)
        .map((item) => ({
          path: String(item?.image?.path || "").trim(),
          data: String(item?.image?.data || "").trim(),
        }))
        .filter((item) => (item.path || item.data) && !item.path.startsWith("refmod://"))
        .slice(0, 9);
      if (mode !== "reference_to_video" || !miniMaxH3SceneImageIsPromptInspiration(segment)) return rendererImages;
      const inspiration = segmentImageSource(segment);
      const path = String(inspiration?.path || selectedSegmentImagePath(segment) || "").trim();
      const data = String(inspiration?.data || "").trim();
      return path || data ? [{ path, data, prompt_only_scene_inspiration: true }, ...rendererImages] : rendererImages;
    }
    return [];
  }

  function miniMaxH3PromptVisionImagesForRunner(segment, mode) {
    if ((state.textGemmaRunner || "builtin") === "builtin") return [];
    return miniMaxH3PromptVisionImages(segment, mode);
  }

  async function ensureAutoTimedSingerCuesBeforePrompt(segment, options = {}) {
    if ((!state.autoTimeSingerCuesBeforePrompt && !options.force) || !segment || segment.no_character_present) return;
    if (segment.no_character_present || normalizeVideoType(segment.performance_mode || state.videoType) !== "singing") return;
    const lyric = isInstrumentalLyricText(segment.lyric_text) ? "" : flattenLyricForPrompt(segment.lyric_text);
    const performers = selectedPerformerSubjectsForSegment(segment);
    const rawExisting = Array.isArray(segment.lyric_cue_map) ? segment.lyric_cue_map : [];
    const existing = normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
    const timingCues = existing.length ? existing : rawExisting;
    const existingTimingComplete = timingCues.length && timingCues.every((cue, index) => {
      const start = Number(cue.start);
      // The UI intentionally derives a final cue's end from the scene end when
      // the user leaves that last End field blank. Treat that displayed timing
      // as complete; never replace an already hand-authored cue map with a new
      // transcription just because the final raw `end` is null.
      const end = miniMaxEffectiveCueEnd(segment, timingCues, index, cue);
      return Number.isFinite(start) && Number.isFinite(Number(end)) && Number(end) > start;
    });
    if (existingTimingComplete) return true;
    if (!lyric) return false;
    if (!performers.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: auto-time-before-prompt needs a mapped singer.`);
    }
    segment.lyric_shot_word_timing_enabled = true;
    segment.lyric_performance_mode = "cue_map";
    if (!existing.some((cue) => cue.type !== "instrumental" && flattenLyricForPrompt(cue.text))) {
      const performer = performers[0] || {};
      segment.lyric_cue_map = [{
        type: "vocal",
        text: lyric,
        action_note: "",
        singer_id: performer.id || "",
        singer_name: performer.name || "",
        start: null,
        end: null,
      }];
    }
    const timed = await autoTimeMiniMaxSingerCuesForSegment(segment);
    if (!timed) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Whisper timing did not complete; prompt generation was stopped.`);
    return true;
  }
  function miniMaxPromptReferenceSignature(segment, mode = miniMaxH3ModeForSegment(segment)) {
    if (!["reference_to_video", "image_reference_to_video", "video_to_video"].includes(mode)) return "";
    const items = mode === "image_reference_to_video"
      ? miniMaxH3ImageReferencePromptItems(segment)
      : miniMaxOrderedImageReferenceItemsForSegment(segment, mode);
    const contract = JSON.stringify({
      mode,
      references: items.map((item) => ({
        kind: item.kind, id: item.source_id || item.id || "",
        name: item.label || item.name || "", description: item.description || "",
        path: item.image?.path || item.path || "", data: item.image?.path ? "" : (item.image?.data || ""),
      })),
      extraPaths: segment?.minimax_h3_image_paths ?? segment?.minimax_image_paths ?? [],
    });
    let hash = 2166136261;
    for (let index = 0; index < contract.length; index += 1) hash = Math.imul(hash ^ contract.charCodeAt(index), 16777619);
    return `${contract.length}:${hash >>> 0}`;
  }

  function miniMaxLegacyPromptReferenceMismatch(segment, prompt, mode, imagePaths) {
    const text = String(prompt || "");
    const definitions = text.match(/subject_definitions:\s*([\s\S]*?)(?=\n\s*(?:summary|retention_analysis|detailed_description|integrated_multimodal_description):|$)/i)?.[1] || "";
    if (!definitions) return "";
    const items = mode === "image_reference_to_video"
      ? miniMaxH3ImageReferencePromptItems(segment)
      : miniMaxOrderedImageReferenceItemsForSegment(segment, mode);
    const matches = [...definitions.matchAll(/<Subject\s+\d+>\s+is\s+([^\n]*?)\s+in\s+<Picture\s+(\d+)>/gi)];
    const frameMatches = [...definitions.matchAll(/<Picture\s+(\d+)>\s+is\s+the\s+(?:first|last)\s+frame\b/gi)];
    const assigned = new Set([...matches.map((match) => Number(match[2])), ...frameMatches.map((match) => Number(match[1]))]);
    if (!assigned.size) return "";
    const available = Array.isArray(imagePaths) ? imagePaths.length : items.length;
    const highest = Math.max(...assigned);
    if (highest > available) return `The prompt uses <Picture ${highest}>, but this render has only ${available} image reference${available === 1 ? "" : "s"}.`;
    if (assigned.size < items.length) return `The prompt defines ${assigned.size} image reference${assigned.size === 1 ? "" : "s"}, but the current scene maps ${items.length}.`;
    for (const match of matches) {
      const picture = Number(match[2]);
      const item = items[picture - 1];
      if (!item) continue;
      const definition = String(match[1] || "").toLowerCase();
      const saysEnvironment = /\benvironment\b|\blocation\b/.test(definition);
      if (saysEnvironment && item.kind !== "location") return `<Picture ${picture}> is described as an environment, but it now maps to ${item.label || item.kind}.`;
      if (!saysEnvironment && item.kind === "location") return `<Picture ${picture}> is described as a subject, but it now maps to the location.`;
      if (item.kind === "subject" || item.kind === "extra") {
        const definedName = definition.replace(/^(?:the|a|an)\s+/, "").trim();
        const namedAt = items.findIndex((candidate) => {
          if (candidate.kind !== "subject" && candidate.kind !== "extra") return false;
          const name = String(candidate.label || candidate.name || "").toLowerCase().trim();
          return name && (definedName === name || definedName.startsWith(`${name} `));
        });
        if (namedAt >= 0 && namedAt !== picture - 1) return `<Picture ${picture}> names ${items[namedAt].label}, but that subject is now <Picture ${namedAt + 1}>.`;
      }
    }
    return "";
  }

  function miniMaxRenderReferenceImagePaths(segment, mode, configuredPaths) {
    const normalizePathList = (value) => (Array.isArray(value) ? value : [])
      .map((item) => String(item?.path || item?.file || item || "").trim())
      .filter(Boolean);
    const usesReferenceBuilderImages = ["reference_to_video", "image_reference_to_video", "video_to_video"].includes(mode);
    const builderPaths = usesReferenceBuilderImages && configuredPaths === undefined
      ? miniMaxReferenceBuilderImagePathsForSegment(segment)
      : [];
    const extraPaths = normalizePathList(configuredPaths !== undefined
      ? configuredPaths
      : (segment?.minimax_h3_image_paths ?? segment?.minimax_image_paths ?? []));
    const seen = new Set();
    let paths = usesReferenceBuilderImages ? [...builderPaths, ...extraPaths]
      .filter((path) => {
        const key = mediaPathKey(path);
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .slice(0, 9) : [];
    if (["image_to_video", "image_reference_to_video"].includes(mode)) {
      const selectedImage = String(selectedSegmentImagePath(segment) || "").trim();
      if (selectedImage) paths = [selectedImage, ...paths.filter((path) => mediaPathKey(path) !== mediaPathKey(selectedImage))].slice(0, 9);
    }
    return paths;
  }

  function miniMaxH3ImageReferencePromptItems(segment, maxImages = 9) {
    const ordered = [];
    const startFrame = segmentImageSource(segment);
    if (startFrame?.path || startFrame?.data) {
      ordered.push({
        key: "scene:start_frame",
        kind: "start_frame",
        label: "Scene start frame",
        description: "Exact first frame and authoritative opening composition, location, lighting, and visual-state anchor.",
        image: {
          path: String(startFrame.path || "").trim(),
          data: String(startFrame.data || "").trim(),
          name: String(startFrame.name || "start_frame.png"),
        },
      });
    }
    const endFrame = firstLastFrameEndImageSource(segment);
    if (endFrame?.path || endFrame?.data) {
      ordered.push({
        key: "scene:end_frame",
        kind: "end_frame",
        label: "Scene end frame",
        description: "Exact last frame and authoritative final composition, subject placement, lighting, and visual-state anchor.",
        image: {
          path: String(endFrame.path || "").trim(),
          data: String(endFrame.data || "").trim(),
          name: String(endFrame.name || "end_frame.png"),
        },
      });
    }
    ordered.push(...miniMaxOrderedImageReferenceItemsForSegment(segment, "image_reference_to_video"));
    const seen = new Set();
    return ordered.filter((item) => {
      const path = String(item?.image?.path || "").trim();
      const data = String(item?.image?.data || "").trim();
      const fingerprint = path ? `path:${mediaPathKey(path)}` : data ? `data:${data}` : "";
      if (!fingerprint || seen.has(fingerprint)) return false;
      seen.add(fingerprint);
      return true;
    }).slice(0, Math.max(0, Number(maxImages) || 9));
  }

  function miniMaxH3PromptCharacterBudget(segment, mode = miniMaxH3ModeForSegment(segment), requestedTargetLimit = 6500) {
    const hardLimit = 7000;
    const targetLimit = Math.max(0, Math.min(hardLimit, Math.round(Number(requestedTargetLimit) || 6500)));
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    const shotCount = miniMaxH3OfficialShotPlan(miniMaxH3CutPlanForSegment(segment)).length;
    const emptyCreative = miniMaxH3OfficialShotBodyFromDescriptions(segment, Array.from({ length: shotCount }, () => ""), normalizedMode);
    let fixedPrompt = "";
    if (["text_to_video", "image_to_video"].includes(normalizedMode)) {
      fixedPrompt = [
        miniMaxH3OfficialIntegratedDescription(segment, normalizedMode, emptyCreative),
        miniMaxH3OfficialSoundscape(segment),
        miniMaxH3OfficialMusic(segment),
      ].filter(Boolean).join("\n\n").trim();
    } else {
      fixedPrompt = `detailed_description:\nThe target video is in a ${miniMaxH3OpeningStyle(segment)} music-video style.\n\n${emptyCreative}`.trim();
    }
    // The RefMod pipeline adds each RefMod's label after the writer is done. A character's label is added in every
    // shot. A RefMod the shots never name (clothing, background, style) gets one short sentence. Both are reserved here,
    // so the writer is given room and the finished prompt stays under the limit.
    const refmodReserve = isRefmodPipelineActive()
      ? miniMaxOrderedImageReferenceItemsForSegment(segment, normalizedMode).map((item) => item.refmod).filter(Boolean)
        .reduce((total, item) => total + ((item.category === "character" || item.category === "extra")
          ? (item.label.length + 1) * shotCount
          : item.label.length + 47), 0)
      : 0;
    const fixedChars = fixedPrompt.length + refmodReserve;
    return {
      hardLimit,
      targetLimit,
      fixedChars,
      refmodReserve,
      shotDescriptionChars: Math.max(0, targetLimit - fixedChars),
      shotCount,
    };
  }

  function miniMaxPromptReferenceMismatch(segment, prompt, mode, imagePaths) {
    const signature = miniMaxPromptReferenceSignature(segment, mode);
    if (!signature || !String(prompt || "").trim()) return "";
    const saved = segment?.minimax_h3_prompt_reference_binding;
    if (saved?.prompt === String(prompt).trim() && saved.signature !== signature) {
      return "The scene's reference order or images changed after this prompt was generated.";
    }
    // RefMod prompts carry <Video n> labels, so the <Picture n> numbering check does not apply.
    if (typeof isRefmodPipelineActive === "function" && isRefmodPipelineActive()) return "";
    return miniMaxLegacyPromptReferenceMismatch(segment, prompt, mode, imagePaths);
  }

  return {
    applyMiniMaxH3NativeVoiceBlock, assembleMiniMaxH3PromptFromCreative, assertValidMiniMaxH3FinalPrompt,
    ensureAutoTimedSingerCuesBeforePrompt, ensureBuilderManagedFx, miniMaxH3CreativePromptContextForSegment,
    miniMaxH3ImageReferencePromptItems, miniMaxH3PromptCharacterBudget, miniMaxH3PromptVisionImages,
    miniMaxH3PromptVisionImagesForRunner, miniMaxPromptReferenceMismatch, miniMaxPromptReferenceSignature,
    miniMaxRenderReferenceImagePaths,
  };
}
