import { storyboardCutPlanForDuration } from "../storyboard_builder/scenes.mjs";
import { normalizeMiniMaxSpeakerAssignments } from "./minimax_h3.mjs";
import {
  miniMaxH3OfficialCutPlanInstruction,
  miniMaxH3PunctuatedCueText,
  miniMaxH3Timecode,
} from "./minimax_prompt.mjs";
import { flattenLyricForPrompt, isInstrumentalLyricText, segmentUsesNoLipSyncPerformance } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { newSegment } from "./segments.mjs";
import { timelineSegmentDuration } from "./timeline_state.mjs";

function parseLyricSegmentOutput(text) {
  const result = [];
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  for (const line of lines) {
    const match = line.match(/^\s*(?:lyricSegment|segment)\s*(\d+)\s*[:=]\s*(.*)\s*$/i);
    if (!match) continue;
    result[Number(match[1]) - 1] = String(match[2] || "").trim();
  }
  return result;
}

export function assertNoBundledReferenceLyrics(lyricValues, referenceLyrics = "") {
  const normalize = (value) => cleanTimestampedLyricText(value)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
  const referenceLines = Array.from(new Set(
    String(referenceLyrics || "")
      .replace(/\r\n?/g, "\n")
      .split("\n")
      .map(normalize)
      .filter((line) => line.length >= 4)
  ));
  if (referenceLines.length < 2) return;
  for (let index = 0; index < lyricValues.length; index += 1) {
    const value = normalize(lyricValues[index]);
    if (!value || referenceLines.includes(value)) continue;
    const containedLines = referenceLines.filter((line) => value.includes(line));
    if (containedLines.length >= 2) {
      throw new Error(
        `Transcription safety check stopped the update: Scene ${index + 1} received ${containedLines.length} pasted lyric lines in one result. `
        + "The existing timeline lyrics were left unchanged. Restart ComfyUI to load the corrected strict line-alignment node, then run Transcribe Lines again."
      );
    }
  }
}

function parseSrtTimestampToSeconds(value) {
  const match = String(value || "").trim().match(/^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$/);
  if (!match) return null;
  const hours = Number(match[1] || 0);
  const minutes = Number(match[2] || 0);
  const seconds = Number(match[3] || 0);
  const millis = Number(String(match[4] || "0").padEnd(3, "0").slice(0, 3));
  return (hours * 3600) + (minutes * 60) + seconds + (millis / 1000);
}

function parseSrtCueSegments(text = "") {
  const normalized = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  if (!normalized) return [];
  const blocks = normalized.split(/\n{2,}/);
  const cues = [];
  for (const block of blocks) {
    const lines = String(block || "").split("\n").map((line) => line.trim()).filter(Boolean);
    const timingIndex = lines.findIndex((line) => /-->\s*/.test(line));
    if (timingIndex < 0) continue;
    const timing = lines[timingIndex].match(/(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})/);
    if (!timing) continue;
    const start = parseSrtTimestampToSeconds(timing[1]);
    const end = parseSrtTimestampToSeconds(timing[2]);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
    const body = lines.slice(timingIndex + 1).join("\n").trim();
    cues.push({ start, end, text: body });
  }
  return cues.sort((a, b) => a.start - b.start || a.end - b.end);
}

export function extractPromptCreatorWhisperSrtText(textValues = []) {
  const values = (Array.isArray(textValues) ? textValues : [textValues])
    .flat(Infinity)
    .map((value) => String(value ?? ""))
    .filter((value) => value.trim());
  return {
    whisper: values.find((value) => /(?:lyricSegment|segment)\s*\d+\s*[:=]/i.test(value)) || "",
    srt: values.find((value) => /-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}/.test(value)) || "",
  };
}

export function createSegmentsFromBeatSrt(srtText = "") {
  const cues = parseSrtCueSegments(srtText);
  const created = [];
  cues.forEach((cue, index) => {
    const segment = newSegment(cue.start, cue.end);
    segment.label = `SCENE ${index + 1}`;
    segment.source = "beat_timestamped_lyrics";
    segment.lyric_text = "";
    segment.lyric_no_lip_sync = false;
    segment.timeline_note = "";
    created.push(segment);
  });
  return created;
}

function isNonVocalLyricMarkerLabel(label = "") {
  const clean = String(label || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").replace(/\s+/g, " ").trim();
  return /^(instrumental|break|interlude|solo|no vocal|no vocals|no lyrics|silence|b roll|music only)$/.test(clean);
}

function stripStructuralLyricHeaders(text = "") {
  return String(text || "").replace(/\[([^\]]{2,80})\]/g, (match, label) => (
    isNonVocalLyricMarkerLabel(label) ? match : " "
  ));
}

export function cleanTimestampedLyricText(text) {
  return stripStructuralLyricHeaders(text)
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([([{])\s+/g, "$1")
    .replace(/\s+([)\]}])/g, "$1")
    .trim();
}

function timestampedWordsFromPayload(payload) {
  const words = [];
  for (const segment of Array.isArray(payload?.segments) ? payload.segments : []) {
    if (String(segment?.type || "").toLowerCase() === "instrumental") continue;
    for (const word of Array.isArray(segment?.words) ? segment.words : []) {
      const text = String(word?.text || "").trim();
      const start = Number(word?.start);
      if (!text || !Number.isFinite(start)) continue;
      const end = Number.isFinite(Number(word?.end)) ? Number(word.end) : start;
      words.push({ text, start: Math.max(0, start), end: Math.max(start, end) });
    }
  }
  words.sort((a, b) => a.start - b.start || a.end - b.end);
  return words;
}

function timestampedTextForExistingScene(payload, scene, words = null, options = {}) {
  const sceneStart = Number(scene?.start || 0);
  const sceneEnd = Number(scene?.end || sceneStart);
  const safeEnd = Math.max(sceneStart, sceneEnd);
  const timedWords = words || timestampedWordsFromPayload(payload);
  const tailPadding = Math.max(0, Number(options?.tailPaddingSeconds || 0));
  const claimedWords = options?.claimedWords instanceof Set ? options.claimedWords : null;
  if (timedWords.length) {
    const selected = timedWords.filter((word) => (
      !claimedWords?.has(word)
      && word.start >= sceneStart - 0.03
      && word.start < safeEnd + tailPadding - 0.01
    ));
    if (claimedWords) selected.forEach((word) => claimedWords.add(word));
    return cleanTimestampedLyricText(selected.map((word) => word.text).join(" "));
  }

  let best = null;
  let bestOverlap = 0;
  for (const item of Array.isArray(payload?.segments) ? payload.segments : []) {
    if (String(item?.type || "").toLowerCase() === "instrumental") continue;
    const start = Number(item?.start);
    const end = Number(item?.end);
    const text = cleanTimestampedLyricText(item?.text);
    if (!text || !Number.isFinite(start) || !Number.isFinite(end)) continue;
    const overlap = Math.max(0, Math.min(safeEnd, end) - Math.max(sceneStart, start));
    if (overlap > bestOverlap) {
      best = text;
      bestOverlap = overlap;
    }
  }
  return bestOverlap > 0.05 ? best : "";
}

function snapExistingSceneBoundariesToTimestampedWords(segments, words = [], options = {}) {
  const items = Array.isArray(segments) ? segments : [];
  const timedWords = Array.isArray(words) ? words : [];
  if (items.length < 2 || !timedWords.length) return 0;
  const tailPadding = Math.max(0, Number(options?.tailPaddingSeconds || 0));
  const maxShift = Math.max(0.18, Math.min(1.5, tailPadding + 0.45));
  const wordLeadIn = 0.04;
  const minScene = 0.2;
  let adjusted = 0;

  for (let index = 0; index < items.length - 1; index += 1) {
    const current = items[index];
    const next = items[index + 1];
    const currentStart = Number(current?.start);
    const currentEnd = Number(current?.end);
    const nextStart = Number(next?.start);
    const nextEnd = Number(next?.end);
    if (![currentStart, currentEnd, nextStart, nextEnd].every(Number.isFinite)) continue;
    if (nextEnd <= currentStart + minScene) continue;

    const boundary = Math.max(currentStart, currentEnd);
    const sharedBoundary = Math.abs(nextStart - currentEnd) <= 0.25;
    if (!sharedBoundary) continue;

    const nextFirstWord = timedWords.find((word) => (
      Number(word.start) >= nextStart - maxShift
      && Number(word.start) < nextEnd + Math.min(tailPadding, 0.35)
    ));
    const crossingWord = timedWords.find((word) => (
      Number(word.start) < boundary + 0.03
      && Number(word.end) > boundary - 0.03
    ));

    let candidate = null;
    if (crossingWord && Math.abs(Number(crossingWord.start) - boundary) <= maxShift) {
      candidate = Number(crossingWord.start) - wordLeadIn;
    } else if (nextFirstWord && Math.abs(Number(nextFirstWord.start) - boundary) <= maxShift) {
      candidate = Number(nextFirstWord.start) - wordLeadIn;
    } else {
      const previousWords = timedWords.filter((word) => (
        Number(word.start) >= currentStart - 0.05
        && Number(word.start) < boundary + Math.min(tailPadding, 0.35)
      ));
      const previousLastWord = previousWords[previousWords.length - 1];
      if (previousLastWord && Number(previousLastWord.end) > boundary - maxShift && Number(previousLastWord.end) < nextEnd) {
        candidate = Number(previousLastWord.end) + 0.03;
      }
    }

    if (!Number.isFinite(candidate)) continue;
    candidate = Math.max(currentStart + minScene, Math.min(nextEnd - minScene, candidate));
    if (Math.abs(candidate - boundary) < 0.015 || Math.abs(candidate - boundary) > maxShift) continue;

    current.end = candidate;
    next.start = candidate;
    adjusted += 1;
  }
  return adjusted;
}

export function segmentSingerSubjectText(segment) {
  if (segment?.no_character_present) return "";
  if (Array.isArray(segment?.lyric_singers)) {
    return segment.lyric_singers.map((item) => String(item || "").trim()).filter(Boolean).join(", ");
  }
  return String(segment?.lyric_singers || "").trim();
}

export function miniMaxH3PerformerLabel(subject, labelMap) {
  const id = String(subject?.id || "").trim();
  const name = String(subject?.name || "").trim();
  const mapped = labelMap.get(id) || labelMap.get(name.toLowerCase());
  if (mapped?.label) return `${mapped.label}${mapped.alias ? ` ${mapped.alias}` : ""}`;
  return name || "the assigned performer";
}

export function lyricCueTextParts(text) {
  const clean = flattenLyricForPrompt(text);
  if (!clean) return [];
  const lines = String(text || "").split(/\r?\n+/).map((line) => flattenLyricForPrompt(line)).filter(Boolean);
  const splitIntoTwo = (items) => {
    if (items.length <= 1) return items;
    const midpoint = Math.ceil(items.length / 2);
    return [items.slice(0, midpoint).join(" "), items.slice(midpoint).join(" ")].map((item) => item.trim()).filter(Boolean);
  };
  if (lines.length > 1) return splitIntoTwo(lines);
  const words = clean.split(/\s+/).filter(Boolean);
  if (words.length <= 2) return [clean];
  return splitIntoTwo(words);
}

export function miniMaxH3CueTimingText(cue = {}, segment = null, cues = [], index = 0) {
  const start = Number(cue?.start);
  const end = miniMaxEffectiveCueEnd(segment, cues, index, cue);
  if (Number.isFinite(start) && Number.isFinite(end) && end > start) return `${start.toFixed(3)}s-${end.toFixed(3)}s: `;
  if (Number.isFinite(start)) return `from ${start.toFixed(3)}s: `;
  return "";
}

export function miniMaxEffectiveCueEnd(segment, cues = [], index = 0, cueOverride = null) {
  const cue = cueOverride || cues[index] || {};
  const start = Number(cue?.start);
  const explicitEnd = Number(cue?.end);
  if (Number.isFinite(explicitEnd) && (!Number.isFinite(start) || explicitEnd > start)) return explicitEnd;
  if (!segment) return Number.isFinite(explicitEnd) ? explicitEnd : null;
  const range = singerCuePlaybackRangeForCue(segment, cues, index);
  return Number.isFinite(Number(range?.end)) ? Number(range.end) : null;
}

export function miniMaxNextCueStartTime(segment, cues = []) {
  const sceneDuration = Math.max(0, Number(timelineSegmentDuration(segment) || 0));
  if (!Array.isArray(cues) || !cues.length) return 0;
  const lastIndex = cues.length - 1;
  const lastCue = cues[lastIndex] || {};
  const lastEnd = miniMaxEffectiveCueEnd(segment, cues, lastIndex, lastCue);
  if (Number.isFinite(Number(lastEnd))) return Math.max(0, Math.min(sceneDuration || Number(lastEnd), Number(lastEnd)));
  const lastStart = Number(lastCue.start);
  return Number.isFinite(lastStart) ? Math.max(0, Math.min(sceneDuration || lastStart, lastStart)) : 0;
}

export function syncCueEndBoundariesFromNextStarts(cues = []) {
  if (!Array.isArray(cues)) return cues;
  for (let index = 0; index < cues.length - 1; index += 1) {
    const current = cues[index];
    const nextStart = Number(cues[index + 1]?.start);
    const currentStart = Number(current?.start);
    if (current && Number.isFinite(nextStart) && (!Number.isFinite(currentStart) || nextStart > currentStart)) {
      current.end = nextStart;
    }
  }
  return cues;
}

export function formatCueTime(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(3)}s` : "--";
}

export function singerCuePlaybackRangeForCue(segment, cues = [], index = 0) {
  const cue = cues[index] || {};
  const sceneDuration = Math.max(0.1, Number(timelineSegmentDuration(segment) || 0.1));
  const hasStart = Number.isFinite(Number(cue.start));
  const start = Math.max(0, Math.min(sceneDuration, hasStart ? Number(cue.start) : 0));
  const explicitEnd = Number(cue.end);
  if (Number.isFinite(explicitEnd) && explicitEnd > start) {
    return { start, end: Math.min(sceneDuration, explicitEnd), hasStart };
  }
  const nextCue = cues.slice(index + 1).find((item) => Number.isFinite(Number(item?.start)) && Number(item.start) > start);
  if (nextCue) return { start, end: Math.min(sceneDuration, Number(nextCue.start)), hasStart };
  return { start, end: sceneDuration > start ? sceneDuration : null, hasStart };
}

function shiftCueLyricTextDown(cues = [], index = 0, text = "", defaultPerformer = {}) {
  const carried = flattenLyricForPrompt(text);
  if (!carried) return;
  cues.splice(index + 1, 0, { type: "vocal", text: carried, action_note: "", singer_id: defaultPerformer?.id || "", singer_name: defaultPerformer?.name || "", start: null, end: null });
}

export function timestampedCueSegments(payload = {}) {
  return (Array.isArray(payload?.segments) ? payload.segments : [])
    .map((item) => {
      const type = String(item?.type || "").trim().toLowerCase() === "instrumental" ? "instrumental" : "vocal";
      const start = Number(item?.start);
      const end = Number(item?.end);
      return {
        type,
        text: flattenLyricForPrompt(item?.text),
        start: Number.isFinite(start) ? Math.max(0, start) : null,
        end: Number.isFinite(end) ? Math.max(0, end) : null,
        duration: Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, end - start) : 0,
        words: (Array.isArray(item?.words) ? item.words : []).map((word) => ({
          text: String(word?.text || word?.word || "").trim(),
          start: Number.isFinite(Number(word?.start)) ? Math.max(0, Number(word.start)) : null,
          end: Number.isFinite(Number(word?.end)) ? Math.max(0, Number(word.end)) : null,
        })).filter((word) => word.text && Number.isFinite(word.start) && Number.isFinite(word.end) && word.end > word.start),
      };
    })
    .filter((item) => Number.isFinite(Number(item.start)) && Number.isFinite(Number(item.end)) && item.end > item.start);
}

export function rebuildSingerCueMapFromTimestampedSegments(segment, currentCues = [], timestamped = [], options = {}) {
  const sceneDuration = Math.max(0, Number(timelineSegmentDuration(segment) || 0));
  const minInstrumentalGap = Math.max(0, Number(options.minInstrumentalGap ?? 0.5));
  const trailingVocalTailMergeSeconds = Math.max(0, Number(options.trailingVocalTailMergeSeconds ?? 1.5));
  const vocalCues = currentCues.filter((cue) => cue.type !== "instrumental" && flattenLyricForPrompt(cue.text));
  const instrumentalNotes = currentCues
    .filter((cue) => cue.type === "instrumental")
    .map((cue) => String(cue.action_note || cue.note || "").trim())
    .filter(Boolean);
  let vocalIndex = 0;
  let instrumentalIndex = 0;
  const rebuilt = [];
  const absorbSkippedGap = (end) => {
    if (!rebuilt.length) return;
    const previous = rebuilt[rebuilt.length - 1];
    if (!Number.isFinite(Number(previous.end)) || Number(previous.end) < end) previous.end = end;
  };
  for (let itemIndex = 0; itemIndex < timestamped.length; itemIndex += 1) {
    const item = timestamped[itemIndex];
    const start = Math.max(0, Math.min(sceneDuration || item.start, Number(item.start)));
    const end = Math.max(start, Math.min(sceneDuration || item.end, Number(item.end)));
    if (item.type === "instrumental") {
      const hasLaterVocal = timestamped.slice(itemIndex + 1).some((next) => next.type !== "instrumental");
      const previous = rebuilt[rebuilt.length - 1];
      if (!hasLaterVocal && previous?.type === "vocal" && end - start <= trailingVocalTailMergeSeconds) {
        previous.end = end;
        continue;
      }
      if (end - start < minInstrumentalGap) {
        if (start <= 0.001 && rebuilt[0]) rebuilt[0].start = 0;
        else absorbSkippedGap(end);
        continue;
      }
      const note = instrumentalNotes[instrumentalIndex++] || "";
      rebuilt.push({ type: "instrumental", text: "", action_note: note, singer_id: "", singer_name: "", start, end });
      continue;
    }
    const source = vocalCues[vocalIndex++] || {};
    rebuilt.push({
      type: "vocal",
      text: flattenLyricForPrompt(source.text || item.text),
      action_note: "",
      singer_id: String(source.singer_id || "").trim(),
      singer_name: String(source.singer_name || "").trim(),
      start,
      end,
    });
  }
  while (vocalIndex < vocalCues.length) {
    const source = vocalCues[vocalIndex++];
    const start = miniMaxNextCueStartTime(segment, rebuilt);
    rebuilt.push({
      type: "vocal",
      text: flattenLyricForPrompt(source.text),
      action_note: "",
      singer_id: String(source.singer_id || "").trim(),
      singer_name: String(source.singer_name || "").trim(),
      start,
      end: null,
    });
  }
  if (rebuilt.length) {
    const last = rebuilt[rebuilt.length - 1];
    if (!Number.isFinite(Number(last.end)) || Number(last.end) <= Number(last.start)) last.end = sceneDuration || null;
  }
  return rebuilt.filter((cue) => cue.type === "instrumental" || cue.text);
}

export function rebuildSpeakerCueMapFromTimestampedSegments(segment, currentCues = [], timestamped = [], options = {}) {
  const sceneDuration = Math.max(0, Number(timelineSegmentDuration(segment) || 0));
  const minInstrumentalGap = Math.max(0, Number(options.minInstrumentalGap ?? 0.5));
  const trailingSpeechTailMergeSeconds = Math.max(0, Number(options.trailingSpeechTailMergeSeconds ?? 1.0));
  const normalizedCues = normalizeMiniMaxSpeakerAssignments(currentCues);
  const dialogueCues = normalizedCues.filter((cue) => cue.type !== "instrumental" && flattenLyricForPrompt(cue.text));
  const instrumentalNotes = normalizedCues
    .filter((cue) => cue.type === "instrumental")
    .map((cue) => String(cue.action_note || cue.note || "").trim())
    .filter(Boolean);
  let dialogueIndex = 0;
  let instrumentalIndex = 0;
  const rebuilt = [];
  const absorbSkippedGap = (end) => {
    if (!rebuilt.length) return;
    const previous = rebuilt[rebuilt.length - 1];
    if (!Number.isFinite(Number(previous.end)) || Number(previous.end) < end) previous.end = end;
  };
  for (let itemIndex = 0; itemIndex < timestamped.length; itemIndex += 1) {
    const item = timestamped[itemIndex];
    const start = Math.max(0, Math.min(sceneDuration || item.start, Number(item.start)));
    const end = Math.max(start, Math.min(sceneDuration || item.end, Number(item.end)));
    if (item.type === "instrumental") {
      const hasLaterDialogue = timestamped.slice(itemIndex + 1).some((next) => next.type !== "instrumental");
      const previous = rebuilt[rebuilt.length - 1];
      if (!hasLaterDialogue && previous?.type !== "instrumental" && end - start <= trailingSpeechTailMergeSeconds) {
        previous.end = end;
        continue;
      }
      if (end - start < minInstrumentalGap) {
        if (start <= 0.001 && rebuilt[0]) rebuilt[0].start = 0;
        else absorbSkippedGap(end);
        continue;
      }
      const note = instrumentalNotes[instrumentalIndex++] || "";
      rebuilt.push({ type: "instrumental", text: "", action_note: note, speaker_id: "", speaker_name: "", start, end });
      continue;
    }
    const source = dialogueCues[dialogueIndex++] || {};
    rebuilt.push({
      type: "dialogue",
      text: flattenLyricForPrompt(source.text || item.text),
      action_note: "",
      speaker_id: String(source.speaker_id || "").trim(),
      speaker_name: String(source.speaker_name || "").trim(),
      start,
      end,
    });
  }
  while (dialogueIndex < dialogueCues.length) {
    const source = dialogueCues[dialogueIndex++];
    const start = miniMaxNextCueStartTime(segment, rebuilt);
    rebuilt.push({
      type: "dialogue",
      text: flattenLyricForPrompt(source.text),
      action_note: "",
      speaker_id: String(source.speaker_id || "").trim(),
      speaker_name: String(source.speaker_name || "").trim(),
      start,
      end: null,
    });
  }
  if (rebuilt.length) {
    const last = rebuilt[rebuilt.length - 1];
    if (!Number.isFinite(Number(last.end)) || Number(last.end) <= Number(last.start)) last.end = sceneDuration || null;
  }
  return normalizeMiniMaxSpeakerAssignments(rebuilt).filter((cue) => cue.type === "instrumental" || cue.text);
}

export function createLyricCues({
  activeSegment, allEditableSegments, currentGlobalTime, i2vVideoSettingsForSegment,
  isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode, logicalReferenceSubjects,
  logicalSubjectIdsForScene, lyricSingersInput, miniMaxH3FrameContinuityPromptEnabled,
  miniMaxH3ModeForSegment, miniMaxOrderedImageReferenceItemsForSegment, sceneReferenceMapArray,
  sceneReferenceMapValue, segmentIndexInfo, state,
}) {
  function formatLyricSegmentText(lyrics = null) {
    const segments = allEditableSegments();
    return segments.map((segment, index) => {
      const value = Array.isArray(lyrics) ? lyrics[index] : segment?.lyric_text;
      return `lyricSegment${index + 1}=${String(value || "").trim()}`;
    }).join("\n") + "\n";
  }

  function selectedSceneSubjectsForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    if (!segment || segment.no_character_present) return [];
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    return logicalSubjectIdsForScene(normalizedRefs, segment, segmentIndexInfo(segment).index)
      .map((id) => logicalReferenceSubjects(normalizedRefs).find((subject) => String(subject?.id || "") === String(id)))
      .filter(Boolean);
  }

  function selectedCastCoverageContract(segment, options = {}) {
    const subjects = selectedSceneSubjectsForSegment(segment);
    if (!subjects.length) return "";
    const labelMap = options.labelMap instanceof Map ? options.labelMap : null;
    const subjectLabel = (subject) => {
      const id = String(subject?.id || "").trim();
      const name = String(subject?.name || "").trim();
      const mapped = labelMap ? (labelMap.get(id) || labelMap.get(name.toLowerCase())) : null;
      return mapped?.label || name || "selected subject";
    };
    const labels = subjects.map(subjectLabel);
    const performerIds = new Set(selectedPerformerSubjectsForSegment(segment).map((subject) => String(subject?.id || "").trim()).filter(Boolean));
    const performerLabels = subjects.filter((subject) => performerIds.has(String(subject?.id || "").trim())).map(subjectLabel);
    const featuredLabels = performerLabels.length ? performerLabels : labels;
    const shotPlan = Array.isArray(options.shotPlan) ? options.shotPlan : [];
    const lines = [
      `SELECTED CAST COVERAGE — MANDATORY: ${labels.join(", ")} are all selected visible subjects for this scene. Every selected subject must appear visibly at least once. Performer/singer assignment controls only who sings or lip-syncs; it never removes the other selected subjects from the scene. Non-singing selected subjects continue appropriate visible band, acting, reaction, movement, or instrument performance without lip-syncing.`,
      "VISIBLE BAND PERFORMANCE — MANDATORY: whenever a selected subject whose Reference Builder name or description identifies a musician, band role, or instrument is visible, show that subject actively and believably playing their assigned instrument. Preserve the exact instrument assignment; show purposeful hand, arm, and body interaction appropriate to that instrument, and do not leave the member merely standing, posing, holding the instrument idle, or reacting. A singer who is also assigned an instrument must continue playing it while singing unless the scene direction explicitly says otherwise. Do not invent an instrument for a subject whose reference does not assign one.",
    ];
    if (shotPlan.length <= 1) {
      lines.push(performerLabels.length
        ? `Single-shot performer rule: feature ${performerLabels.join(", ")} as the primary focus throughout the continuous shot. Keep the other selected subjects visible in appropriate supporting coverage when composition allows.`
        : `Single-shot cast rule: keep ${labels.join(", ")} visibly present together in the continuous shot. Do not isolate one member for the entire scene.`);
      return lines.join("\n");
    }
    const assignments = shotPlan.map((shot, index) => {
      const featured = featuredLabels[index % featuredLabels.length];
      const remaining = labels.filter((label) => label !== featured);
      return `Shot ${shot?.number || index + 1}: feature ${featured}; ${remaining.length ? `${remaining.join(", ")} may remain visible in supporting coverage` : "keep the selected subject visible"}.`;
    });
    lines.push(performerLabels.length
      ? "PERFORMER FOCUS — MANDATORY: an assigned performer/singer remains the featured primary subject on every lip-sync shot. If multiple performers are assigned, rotate only among those performers. Other selected cast members may appear in supporting, reaction, group, or instrument coverage, but must not replace an assigned performer as the featured subject."
      : "CUT ROTATION — MANDATORY: every cut must change the featured band member. Never keep the same member as the primary focus across consecutive shots while another selected member is available. Cycle through the selected cast; group or supporting coverage is allowed, but it does not replace the required featured-member rotation.",
      ...assignments);
    return lines.join("\n");
  }

  function ltx25SelectedCastCoverageContract(segment) {
    if (String(i2vVideoSettingsForSegment(segment)?.ltx_version || "2.5") === "2.3") return "";
    const base = selectedCastCoverageContract(segment);
    if (!base) return "";
    const hasAssignedPerformer = selectedPerformerSubjectsForSegment(segment).length > 0;
    return `${base}\n${hasAssignedPerformer
      ? "LTX 2.5 performer cut rule: if the prompt contains multiple shots or cuts, keep an assigned performer as the featured subject after every cut; rotate only between assigned performers when more than one is selected."
      : "LTX 2.5 cut rule: if the prompt contains multiple shots or cuts, each new shot must feature a different selected member from the preceding shot. If there is no cut, stage all selected members together in the continuous shot."}`;
  }

  function segmentMappedLocationReference(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    if (!segment) return null;
    const indexKey = String(segmentIndexInfo(segment).index + 1);
    const locId = String(sceneReferenceMapValue(refs.scene_map, segment, Number(indexKey) - 1) || "").trim();
    if (!locId) return null;
    const location = refs.locations.find((item) => String(item?.id || "").trim() === locId) || null;
    return location ? { ...location, id: locId } : null;
  }

  function syncPerformerInspectorForSegment(segment) {
    if (!segment || String(activeSegment()?.id || "") !== String(segment.id || "")) return;
    lyricSingersInput.value = Array.isArray(segment.lyric_singers) ? segment.lyric_singers.join(", ") : "";
    lyricSingersInput.dataset.vrgdgInspectorSegmentId = String(segment.id || "");
    lyricSingersInput.dataset.vrgdgUserEdited = "0";
  }

  function miniMaxH3SubjectLabelMapForSegment(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const map = new Map();
    let subjectNumber = 0;
    miniMaxOrderedImageReferenceItemsForSegment(segment, mode).forEach((item) => {
      if (item?.kind === "start_frame") return;
      subjectNumber += 1;
      const rawKey = String(item?.key || "").trim();
      const subjectId = String(item?.id || item?.subject_id || item?.subjectId || rawKey.replace(/^subject:/, "") || "").trim();
      const name = String(item?.label || item?.name || "").trim();
      const value = {
        label: `<Subject ${subjectNumber}>`,
        alias: "",
        name,
        kind: item?.kind || "reference",
      };
      if (subjectId) map.set(subjectId, value);
      if (name) map.set(name.toLowerCase(), value);
    });
    return map;
  }

  function miniMaxH3VocalCueMapText(segment, mode = miniMaxH3ModeForSegment(segment), options = {}) {
    const performers = selectedPerformerSubjectsForSegment(segment);
    const lyricText = isInstrumentalLyricText(segment?.lyric_text) ? "" : flattenLyricForPrompt(segment?.lyric_text);
    const cueMap = normalizeLyricCueMapForSegment(segment);
    if (segmentUsesNoLipSyncPerformance(segment) || segment?.no_character_present || !performers.length) return "";
    if (!lyricText && !cueMap.length) return "";
    const labelMap = miniMaxH3SubjectLabelMapForSegment(segment, mode);
    if (String(segment?.lyric_performance_mode || "together") === "cue_map" && cueMap.length) {
      const lines = cueMap.map((cue, index) => {
        const timing = miniMaxH3CueTimingText(cue, segment, cueMap, index);
        if (cue.type === "instrumental") {
          const note = cue.action_note ? ` Action note: ${cue.action_note}` : "";
          return `${timing}Use only the assigned visual action and camera direction for this interval.${note}`;
        }
        const subject = performers.find((item) => String(item.id) === String(cue.singer_id)) || { id: cue.singer_id, name: cue.singer_name };
        const vocalStart = Number(cue.vocal_start);
        const vocalEnd = Number(cue.vocal_end);
        const vocalWindow = Number.isFinite(vocalStart) && Number.isFinite(vocalEnd) && vocalEnd > vocalStart
          ? ` The assigned performer lip-syncs from ${vocalStart.toFixed(3)}s-${vocalEnd.toFixed(3)}s.`
          : "";
        return `${timing}${miniMaxH3PerformerLabel(subject, labelMap)} performs "${miniMaxH3PunctuatedCueText(cue.text)}" from <Audio 1>.${vocalWindow}`;
      });
      if (options.compact) return lines.join(" ");
      return [
        "Vocal cue map:",
        ...lines,
        "Apply performer assignments only to vocal cues; use visual action and camera direction for all other cue rows.",
      ].join("\n");
    }
    if (performers.length === 1 && lyricText) {
      const performer = miniMaxH3PerformerLabel(performers[0], labelMap);
      return `${performer} performs the exact full lyric/dialogue line from <Audio 1>: "${miniMaxH3PunctuatedCueText(lyricText)}"`;
    }
    const performerText = performers.map((subject) => miniMaxH3PerformerLabel(subject, labelMap)).join(" and ");
    return `${performerText} perform the same complete lyric/dialogue line together from <Audio 1>: "${miniMaxH3PunctuatedCueText(lyricText)}"`;
  }

  function miniMaxH3CutPlanForSegment(segment) {
    const duration = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    const fallback = storyboardCutPlanForDuration(duration, state.builderStoryboardDefaults?.minimax_h3_cut_frequency);
    if (miniMaxH3FrameContinuityPromptEnabled(segment)) {
      const continuous = { ...fallback, frequency: 0, cut_times_seconds: [], cue_driven: false };
      continuous.instruction = miniMaxH3OfficialCutPlanInstruction(continuous);
      return continuous;
    }
    if (isMiniMaxBuiltInSpeakerAssignmentMode(segment)) {
      const speakerCues = normalizeMiniMaxSpeakerAssignments(segment?.minimax_speaker_assignments || segment?.speaker_assignments || segment?.dialogue_cues || [])
        .filter((cue) => cue.type === "instrumental" || cue.text);
      const cuts = [];
      speakerCues.forEach((cue, index) => {
        const start = Number(cue?.start);
        if (index > 0 && Number.isFinite(start) && start > 0.04 && start < duration - 0.04 && !cuts.some((time) => Math.abs(time - start) < 0.04)) {
          cuts.push(Number(start.toFixed(3)));
        }
      });
      if (cuts.length) {
        cuts.sort((a, b) => a - b);
        return {
          ...fallback,
          frequency: state.builderStoryboardDefaults?.minimax_h3_cut_frequency ?? fallback.frequency,
          cut_times_seconds: cuts,
          cue_driven: true,
          cue_count: speakerCues.length,
          instruction: `EDITING / CUT PLAN — MANDATORY: The timed speaker/dialogue cue map controls this exact ${Number(duration.toFixed(3))}-second segment. Create exactly ${cuts.length + 1} shot${cuts.length ? "s" : ""}${cuts.length ? ` with hard cuts at ${cuts.map((time) => miniMaxH3Timecode(time)).join(", ")}` : ""}. Dialogue rows mean only the assigned speaker talks during that cue; instrumental rows mean no visible lip-sync or spoken dialogue. The builder will write [Shot 1] and every later [Shot N] At MM:SS.mmm label. Return only the creative description for each shot. Do not omit, merge, add, reorder, or shift cue shots.`,
        };
      }
    }
    if (!isMiniMaxSingerAssignmentMode(segment) || String(segment?.lyric_performance_mode || "together") !== "cue_map") return fallback;
    const cues = normalizeLyricCueMapForSegment(segment);
    if (cues.length < 2) return fallback;
    const cuts = [];
    cues.forEach((cue, index) => {
      const start = Number(cue?.start);
      if (index > 0 && Number.isFinite(start) && start > 0.04 && start < duration - 0.04 && !cuts.some((time) => Math.abs(time - start) < 0.04)) {
        cuts.push(Number(start.toFixed(3)));
      }
    });
    if (!cuts.length) return fallback;
    cuts.sort((a, b) => a - b);
    return {
      ...fallback,
      frequency: state.builderStoryboardDefaults?.minimax_h3_cut_frequency ?? fallback.frequency,
      cut_times_seconds: cuts,
      cue_driven: true,
      cue_count: cues.length,
      instruction: miniMaxH3OfficialCutPlanInstruction({ ...fallback, cut_times_seconds: cuts, cue_driven: true, cue_count: cues.length }),
    };
  }

  function miniMaxH3CueShotContractText(segment, mode = miniMaxH3ModeForSegment(segment)) {
    if (!isMiniMaxSingerAssignmentMode(segment) || String(segment?.lyric_performance_mode || "together") !== "cue_map") return "";
    const cues = normalizeLyricCueMapForSegment(segment);
    if (!cues.length) return "";
    const performers = selectedPerformerSubjectsForSegment(segment);
    const labelMap = miniMaxH3SubjectLabelMapForSegment(segment, mode);
    const duration = Math.max(0, Number(segment?.end || 0) - Number(segment?.start || 0));
    const continuousFramePrompt = miniMaxH3FrameContinuityPromptEnabled(segment);
    const lines = cues.map((cue, index) => {
      const range = singerCuePlaybackRangeForCue(segment, cues, index);
      const start = Number.isFinite(Number(cue.start)) ? Number(cue.start) : range.start;
      const end = Number.isFinite(Number(cue.end)) && Number(cue.end) > start ? Number(cue.end) : range.end;
      const timing = Number.isFinite(end) && end > start
        ? `${start.toFixed(3)}s-${Math.min(duration || end, end).toFixed(3)}s`
        : `from ${start.toFixed(3)}s`;
      if (cue.type === "instrumental") {
        const note = String(cue.action_note || "").trim();
        return `${continuousFramePrompt ? `Continuous-shot cue ${index + 1}` : `[Shot ${index + 1}]`} ${timing}: use only the assigned visual action and camera direction.${note ? ` Visual action note: ${note}` : ""}`;
      }
      const subject = performers.find((item) => String(item.id) === String(cue.singer_id)) || { id: cue.singer_id, name: cue.singer_name };
      const vocalStart = Number(cue.vocal_start);
      const vocalEnd = Number(cue.vocal_end);
      const vocalWindow = Number.isFinite(vocalStart) && Number.isFinite(vocalEnd) && vocalEnd > vocalStart
        ? ` The singer lip-syncs from ${vocalStart.toFixed(3)}s to ${vocalEnd.toFixed(3)}s.`
        : "";
      return `${continuousFramePrompt ? `Continuous-shot cue ${index + 1}` : `[Shot ${index + 1}]`} ${timing}: ${miniMaxH3PerformerLabel(subject, labelMap)} is the only performer singing/lip-syncing <d>[English] ${miniMaxH3PunctuatedCueText(cue.text)}</d> from <Audio 1>.${vocalWindow} Other visible performers remain silent, mouth closed or naturally reacting.`;
    });
    return [
      "Timed singer/lyric shot contract — authoritative:",
      ...lines,
      continuousFramePrompt
        ? "All listed cues occur inside the same uninterrupted shot at their assigned times. Do not add a cut, reset, or new setup between cues. Do not swap singers, merge lyric cues, or anticipate a later vocal cue."
        : "Each listed cue is its own shot. Do not swap singers, merge lyric cues, or anticipate a later vocal cue. Apply vocal direction only to the assigned vocal row and its exact timing.",
    ].join("\n");
  }

  function buildShotAlignedSingerCueMap(segment, timestamped = [], performer = {}) {
    const sceneDuration = Math.max(0, Number(timelineSegmentDuration(segment) || 0));
    if (!sceneDuration) return [];
    const words = timestamped.flatMap((item) => item.type === "instrumental" ? [] : (Array.isArray(item.words) ? item.words : []))
      .filter((word) => word.text && Number.isFinite(Number(word.start)) && Number.isFinite(Number(word.end)))
      .sort((a, b) => Number(a.start) - Number(b.start));
    if (!words.length) return [];
    const cutPlan = storyboardCutPlanForDuration(sceneDuration, state.builderStoryboardDefaults?.minimax_h3_cut_frequency);
    const cuts = (Array.isArray(cutPlan?.cut_times_seconds) ? cutPlan.cut_times_seconds : [])
      .map(Number)
      .filter((time) => Number.isFinite(time) && time > 0.001 && time < sceneDuration - 0.001)
      .sort((a, b) => a - b);
    const boundaries = [0, ...cuts, sceneDuration];
    const cues = [];
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const shotStart = boundaries[index];
      const shotEnd = boundaries[index + 1];
      const shotWords = words.filter((word) => {
        const midpoint = (Number(word.start) + Number(word.end)) / 2;
        return midpoint >= shotStart && (index === boundaries.length - 2 ? midpoint <= shotEnd : midpoint < shotEnd);
      });
      if (!shotWords.length) {
        cues.push({ type: "instrumental", text: "", action_note: "", singer_id: "", singer_name: "", start: shotStart, end: shotEnd, vocal_start: null, vocal_end: null });
        continue;
      }
      cues.push({
        type: "vocal",
        text: flattenLyricForPrompt(shotWords.map((word) => word.text).join(" ")),
        action_note: "",
        singer_id: String(performer?.id || "").trim(),
        singer_name: String(performer?.name || "").trim(),
        start: shotStart,
        end: shotEnd,
        vocal_start: Math.max(shotStart, Number(shotWords[0].start)),
        vocal_end: Math.min(shotEnd, Number(shotWords[shotWords.length - 1].end)),
      });
    }
    return cues;
  }

  function segmentMappedSubjectText(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    if (!segment) return "";
    if (segment.no_character_present) return "";
    const indexKey = String(segmentIndexInfo(segment).index + 1);
    const subjectIds = logicalSubjectIdsForScene(refs, segment, Number(indexKey) - 1);
    const subjects = subjectIds
      .map((id) => refs.subjects.find((item) => item.id === id))
      .filter(Boolean)
      .map((subject) => {
        const name = String(subject.name || "").trim();
        const description = String(subject.description || "").trim();
        return name && description ? `${name}: ${description}` : name || description;
      })
      .filter(Boolean);
    if (subjects.length) return subjects.join("\n");
    return segmentSingerSubjectText(segment);
  }

  function segmentMappedLocationText(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    if (!segment) return "";
    const locId = refs.scene_map?.[segment.id] || refs.scene_map?.[String(segmentIndexInfo(segment).index + 1)] || "";
    if (!locId) return "";
    const location = refs.locations.find((item) => item.id === locId);
    if (!location) return "";
    const name = String(location.name || "").trim();
    const description = String(location.description || "").trim();
    if (name && description) return `${name}: ${description}`;
    return name || description;
  }

  function selectedPerformerSubjectsForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const subjects = logicalReferenceSubjects(normalizedRefs);
    const performerIds = sceneReferenceMapArray(normalizedRefs.performer_scene_map, segment);
    if (performerIds.length) {
      const validIds = new Set(performerIds.map(String));
      return subjects.filter((subject) => validIds.has(String(subject?.id || "")));
    }
    const selected = new Set((Array.isArray(segment?.lyric_singers) ? segment.lyric_singers : String(segment?.lyric_singers || "").split(/[,;\n]+/))
      .map((value) => String(value || "").trim().toLowerCase())
      .filter(Boolean));
    return subjects.filter((subject) => {
      const id = String(subject?.id || "").trim();
      const name = String(subject?.name || "").trim();
      return selected.has(id.toLowerCase()) || selected.has(name.toLowerCase());
    });
  }

  function normalizeLyricCueMapForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), options = {}) {
    const performers = selectedPerformerSubjectsForSegment(segment, refs);
    const existing = Array.isArray(segment?.lyric_cue_map) ? segment.lyric_cue_map : [];
    // An explicit cue map is authoritative even for one singer. The old
    // checkbox gate silently discarded manually assigned instrumental/lyric
    // rows, so the LLM only saw the scene's full lyric and singer name.
    if (String(segment?.lyric_performance_mode || "together") !== "cue_map") return [];
    // Existing rows already carry their singer ID/name and must survive even
    // when a transient Storyboard clone has no performer list. Only generated
    // (not manually authored) cue maps require performer discovery here.
    if (!existing.length && (performers.length < 2 && !segment?.lyric_shot_word_timing_enabled)) return [];
    const parts = existing.length ? existing : lyricCueTextParts(segment?.lyric_text).map((text, index) => {
      const performer = performers[index % performers.length] || performers[0] || {};
      return { text, singer_id: performer.id || "", singer_name: performer.name || "" };
    });
    return parts.map((cue, index) => {
      const type = String(cue?.type || "").trim() === "instrumental" ? "instrumental" : "vocal";
      const performer = performers.find((subject) => String(subject.id) === String(cue?.singer_id || ""))
        || performers.find((subject) => String(subject.name || "").toLowerCase() === String(cue?.singer_name || "").toLowerCase())
        || performers[index % performers.length]
        || null;
      const cueTime = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value))
        ? Math.max(0, Number(value))
        : null;
      return {
        type,
        text: flattenLyricForPrompt(cue?.text),
        action_note: String(cue?.action_note || cue?.actionNote || cue?.note || "").trim(),
        singer_id: String(performer?.id || cue?.singer_id || "").trim(),
        singer_name: String(performer?.name || cue?.singer_name || "").trim(),
        start: cueTime(cue?.start),
        end: cueTime(cue?.end),
        vocal_start: type === "vocal" ? cueTime(cue?.vocal_start) : null,
        vocal_end: type === "vocal" ? cueTime(cue?.vocal_end) : null,
      };
    }).filter((cue) => options.preserveBlank || cue.type === "instrumental" || cue.text);
  }

  function singerCueRelativePlayheadTime(segment) {
    const global = Math.max(0, Number(currentGlobalTime() || 0));
    const sceneStart = Math.max(0, Number(segment?.start || 0));
    const duration = Math.max(0, Number(segment?.end || sceneStart) - sceneStart);
    return Math.max(0, Math.min(duration || Number.MAX_SAFE_INTEGER, global - sceneStart));
  }

  return {
    buildShotAlignedSingerCueMap, formatLyricSegmentText, ltx25SelectedCastCoverageContract,
    miniMaxH3CueShotContractText, miniMaxH3CutPlanForSegment, miniMaxH3SubjectLabelMapForSegment,
    miniMaxH3VocalCueMapText, normalizeLyricCueMapForSegment, segmentMappedLocationReference,
    segmentMappedLocationText, segmentMappedSubjectText, selectedCastCoverageContract,
    selectedPerformerSubjectsForSegment, singerCueRelativePlayheadTime, syncPerformerInspectorForSegment,
  };
}
