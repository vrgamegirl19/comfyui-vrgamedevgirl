import { sanitizeWizardReferenceLyrics } from "./wizard.mjs";
import { postJson, queueWorkflowPrompt, waitForText } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, normalizeVideoType, toast } from "./controls.mjs";
import {
  assertNoBundledReferenceLyrics,
  cleanTimestampedLyricText,
  createSegmentsFromBeatSrt,
  extractPromptCreatorWhisperSrtText,
  miniMaxEffectiveCueEnd,
  rebuildSingerCueMapFromTimestampedSegments,
  segmentSingerSubjectText,
  timestampedCueSegments,
} from "./lyric_cues.mjs";
import { syncLyricTextFromCueMap } from "./minimax_speaker_cues.mjs";
import { savePromptTextFile } from "./project_files.mjs";
import { cleanGeneratedPromptText, flattenLyricForPrompt, isInstrumentalLyricText } from "./prompt_text.mjs";
import { normalizedLyricMatchText, normalizeFluxReferenceBuilder, normalizeLyricMapper } from "./reference_data.mjs";
import { newSegment, sortSegments } from "./segments.mjs";
import { audioChunkDuration, audioSourceStart, timelineSegmentDuration } from "./timeline_state.mjs";

export async function runTimestampedCueWorkflow(audioPath, referenceLyrics, progress = null, maxSceneSeconds = 8) {
  progress?.set("Building Stable-ts timestamp workflow...", 22);
  const built = await postJson("/vrgdg/workflow_runner/build_timestamped_transcribe_prompt", {
    audio_path: audioPath,
    reference_lyrics: referenceLyrics,
    language: "english",
    segment_mode: "reference_scene_words",
    include_instrumental_gaps: true,
    instrumental_text: "[instrumental]",
    min_gap_seconds: 0.5,
    min_scene_seconds: 1.0,
    max_scene_seconds: Math.max(1, Number(maxSceneSeconds || 8)),
    vocal_tail_padding_seconds: 0.0,
    model_name: "large-v3",
  }, 60000);
  progress?.set("Queueing Stable-ts timestamp workflow...", 32);
  const queued = await queueWorkflowPrompt(built.prompt, {
    onStatus: (message) => progress?.set(message, 34),
    idleTimeoutMs: 10 * 60 * 1000,
  });
  const promptId = queued.prompt_id;
  if (!promptId) throw new Error("ComfyUI queued the timestamp workflow but did not return a prompt_id.");
  const textValues = await waitForText(
    promptId,
    (message) => progress?.set(`${message}\nPrompt ID: ${promptId}`, 58),
    () => false,
    45 * 60 * 1000,
  );
  return parseTimestampedLyricsOutput(textValues.join("\n"));
}

function hasParenthesizedReferenceText(value) {
  return /(?:\([^)\r\n]*\S[^)\r\n]*\)|（[^）\r\n]*\S[^）\r\n]*）)/.test(String(value || ""));
}

function confirmParenthesizedReferenceLyrics(value) {
  if (!hasParenthesizedReferenceText(value)) return true;
  return window.confirm(
    "Parenthesized lyric check\n\n"
    + "I found text inside parentheses in the reference lyrics.\n\n"
    + "Text inside parentheses will be treated as lyrics. If it is not actually sung or spoken, it can cause transcription and timing problems.\n\n"
    + "Please click Cancel, remove any parenthesized lines that are NOT lyrics, and then try again.\n\n"
    + "Click OK only if all text inside parentheses is actual lyrics.",
  );
}

function timestampedLyricsHintHtml() {
  return `
      <div style="display:flex;flex-direction:column;gap:10px;font-size:12px;line-height:1.45;color:#cbd5e1;">
        <div><strong style="color:#cffafe;">Language</strong><br>Whisper/stable-ts language hint. Use <code>english</code> for English songs. Use <code>auto</code> if you do not know the language, but a specific language is usually more stable.</div>
        <div><strong style="color:#cffafe;">Segment mode</strong><br>
          <code>whisper_chunks</code>: uses the natural chunks detected by stable-ts. Good when you do not have lyrics.<br>
          <code>reference_lines</code>: each non-empty pasted lyric line becomes one scene. The lyric line overrides minimum/maximum scene duration, so a short line stays short and a long line is not split.<br>
          <code>exact_reference_lines</code>: uses the normal lyric transcription/alignment and timing cleanup, but each non-empty pasted line stays one vocal scene with the exact pasted text and scene minimum/maximum duration limits are not applied.<br>
          <code>reference_stanzas</code>: blank-line-separated lyric blocks become scenes. Each stanza stays intact even when it falls outside minimum/maximum scene duration.<br>
          <code>beat_scenes</code>: creates beat-timed scenes with the same Prompt Creator Whisper/SRT timing workflow, then attaches the transcribed lyrics afterward.
        </div>
        <div><strong style="color:#cffafe;">Include instrumental gaps</strong><br>When enabled, the node inserts no-vocal scenes for detected timing gaps between vocal chunks. This is useful for intros, breaks, and outros.</div>
        <div><strong style="color:#cffafe;">Instrumental text</strong><br>The text used for no-vocal scenes, usually <code>[instrumental]</code>. Gemma/LTX treats this as a no-singing section.</div>
        <div><strong style="color:#cffafe;">Min gap seconds</strong><br>Only gaps at least this long become instrumental scenes. Example: <code>2.0</code> means tiny pauses are ignored, but a 10 second intro becomes a real scene.</div>
        <div><strong style="color:#cffafe;">Min scene seconds</strong><br>Used for Whisper chunks, beat timing, instrumental scenes, and approximate unmatched timing. A pasted lyric line or stanza may be shorter because its boundary takes priority.</div>
        <div><strong style="color:#cffafe;">Max scene seconds</strong><br>Used for Whisper chunks, beat timing, instrumental scenes, and approximate unmatched timing. A pasted lyric line or stanza is not split just to satisfy this value.</div>
        <div><strong style="color:#cffafe;">Vocal tail padding</strong><br>Adds a small amount of time after the final detected word in a vocal chunk so sung/held last words do not feel cut off. It is clamped before the next vocal word.</div>
        <div><strong style="color:#cffafe;">Manual instrumental sections</strong><br>Only <code>[instrumental]</code> or <code>[instrumental break]</code> on its own line forces a no-vocal section. <code>[intro]</code>, <code>[outro]</code>, and <code>[break]</code> are section headers only because those sections may contain lyrics. Blank lines are just spacing/stanza separators.</div>
      </div>
    `;
}

export function showTimestampedLyricsHintModal() {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 42px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.58);";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid #164e63;background:#083344;";
    const title = document.createElement("div");
    title.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    title.textContent = "Timestamped Line Settings";
    const close = makeButton("Close");
    header.append(title, close);
    const body = document.createElement("div");
    body.style.cssText = "padding:16px;";
    body.innerHTML = timestampedLyricsHintHtml();
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px;border-top:1px solid #1f2937;";
    const cancel = makeButton("Cancel");
    const ok = makeButton("Continue", "primary");
    actions.append(cancel, ok);
    box.append(header, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    close.onclick = () => finish(false);
    cancel.onclick = () => finish(false);
    ok.onclick = () => finish(true);
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) finish(false);
    });
  });
}

function parseTimestampedLyricsOutput(text) {
  const raw = String(text || "").trim();
  const direct = (() => {
    try { return JSON.parse(raw); } catch (_) { return null; }
  })();
  if (direct && Array.isArray(direct.segments)) return direct;
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start >= 0 && end > start) {
    const chunk = raw.slice(start, end + 1);
    try {
      const parsed = JSON.parse(chunk);
      if (parsed && Array.isArray(parsed.segments)) return parsed;
    } catch (_) {
      // Fall through to the friendly error below.
    }
  }
  throw new Error("Timestamped transcription finished, but no timestamped lyrics JSON was found.");
}

function referenceVocalLines(referenceLyrics = "") {
  return String(referenceLyrics || "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => String(line || "").trim())
    .filter((line) => line && !/^\[[^\]]+\]$/.test(line))
    .map((line) => cleanTimestampedLyricText(line))
    .filter(Boolean);
}

function mapTimestampedReferenceLyricsToExistingScenes(payload, scenes = [], referenceLyrics = "", instrumentalText = "[instrumental]") {
  const referenceLines = referenceVocalLines(referenceLyrics);
  if (!referenceLines.length) {
    throw new Error("Reference lyrics are required to transcribe existing scenes without dropping lines.");
  }

  const vocalSegments = (Array.isArray(payload?.segments) ? payload.segments : [])
    .filter((segment) => String(segment?.type || "").toLowerCase() === "vocal")
    .map((segment) => ({
      start: Number(segment?.start),
      end: Number(segment?.end),
      text: cleanTimestampedLyricText(segment?.text || ""),
      words: (Array.isArray(segment?.words) ? segment.words : [])
        .map((word) => ({
          start: Number(word?.start),
          end: Number(word?.end ?? word?.start),
          text: String(word?.text || "").trim(),
        }))
        .filter((word) => word.text && Number.isFinite(word.start))
        .sort((a, b) => a.start - b.start || a.end - b.end),
    }));
  if (vocalSegments.length !== referenceLines.length) {
    throw new Error(
      `Reference-preserving transcription returned ${vocalSegments.length} vocal line${vocalSegments.length === 1 ? "" : "s"}, `
      + `but ${referenceLines.length} reference lyric line${referenceLines.length === 1 ? " was" : "s were"} supplied. The existing timeline was left unchanged.`
    );
  }
  for (let index = 0; index < referenceLines.length; index += 1) {
    const expected = normalizedLyricMatchText(referenceLines[index]);
    const received = normalizedLyricMatchText(vocalSegments[index]?.text || "");
    if (!expected || expected !== received) {
      throw new Error(
        `Reference-preserving transcription changed or reordered lyric line ${index + 1}. The existing timeline was left unchanged.`
      );
    }
  }

  const orderedScenes = (Array.isArray(scenes) ? scenes : [])
    .filter(Boolean)
    .map((segment, originalIndex) => ({
      segment,
      originalIndex,
      start: Number(segment?.start),
      end: Number(segment?.end),
    }))
    .filter(({ start, end }) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((a, b) => a.start - b.start || a.end - b.end || a.originalIndex - b.originalIndex);
  if (!orderedScenes.length) throw new Error("There are no valid existing scene windows to transcribe.");

  const assignments = orderedScenes.map(() => []);
  const assignedLineIndices = orderedScenes.map(() => new Set());
  const sceneIndexForTime = (time) => {
    const exactIndex = orderedScenes.findIndex((scene, index) => (
      time >= scene.start - 0.001
      && (time < scene.end - 0.001 || (index === orderedScenes.length - 1 && time <= scene.end + 0.001))
    ));
    if (exactIndex >= 0) return exactIndex;
    let closestIndex = 0;
    let closestDistance = Number.POSITIVE_INFINITY;
    orderedScenes.forEach((scene, index) => {
      const distance = time < scene.start ? scene.start - time : Math.max(0, time - scene.end);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestIndex = index;
      }
    });
    return closestIndex;
  };

  vocalSegments.forEach((line, lineIndex) => {
    const expected = normalizedLyricMatchText(referenceLines[lineIndex]);
    const timedWordText = normalizedLyricMatchText(line.words.map((word) => word.text).join(" "));
    if (line.words.length && timedWordText === expected) {
      line.words.forEach((word, wordIndex) => {
        // Put each word in the scene containing most of that word. Using the
        // word midpoint prevents a word that merely begins just before a cut
        // from being shown entirely in the previous scene.
        const wordTime = Number.isFinite(word.end) && word.end > word.start
          ? (word.start + word.end) / 2
          : word.start;
        let sceneIndex = sceneIndexForTime(wordTime);

        // Whisper often truncates the timestamp of a sustained final sung
        // word just before a cut even though the detected vocal tail extends
        // into the next scene. Honor that tail when it crosses an adjacent
        // boundary so words such as "here" (9.96s with a 10.56s vocal tail)
        // are not put in the prior scene and the audible scene marked empty.
        const isFinalWord = wordIndex === line.words.length - 1;
        const nextScene = orderedScenes[sceneIndex + 1];
        const nextBoundary = Number(nextScene?.start);
        const lineEnd = Number(line.end);
        const wordEnd = Number(word.end);
        if (
          isFinalWord
          && nextScene
          && Number.isFinite(nextBoundary)
          && Number.isFinite(lineEnd)
          && Number.isFinite(wordEnd)
          && wordEnd <= nextBoundary + 0.001
          && nextBoundary - wordEnd <= 0.25
          && lineEnd > nextBoundary + 0.001
        ) {
          sceneIndex += 1;
        }
        assignments[sceneIndex].push(word.text);
        assignedLineIndices[sceneIndex].add(lineIndex);
      });
      return;
    }

    // Rare unaligned lines have no trustworthy word timestamps. Keep their
    // exact reference text and place them by the best available line start
    // instead of dropping them.
    const fallbackTime = Number.isFinite(line.start) ? line.start : orderedScenes[0].start;
    const sceneIndex = sceneIndexForTime(fallbackTime);
    assignments[sceneIndex].push(referenceLines[lineIndex]);
    assignedLineIndices[sceneIndex].add(lineIndex);
  });

  const mappedLineIndices = Array.from(new Set(assignedLineIndices.flatMap((indices) => Array.from(indices)))).sort((a, b) => a - b);
  if (mappedLineIndices.length !== referenceLines.length || mappedLineIndices.some((lineIndex, index) => lineIndex !== index)) {
    throw new Error("Reference lyric coverage validation failed. The existing timeline was left unchanged.");
  }

  const emptyText = String(instrumentalText || "[instrumental]").trim() || "[instrumental]";
  return orderedScenes.map((entry, index) => ({
    segment: entry.segment,
    lyricText: assignments[index].length ? assignments[index].join(" ") : emptyText,
    referenceLineCount: assignedLineIndices[index].size,
  }));
}

export function mergeTimestampedLyricText(a, b, instrumentalText = "[instrumental]") {
  const values = [a, b].map((value) => String(value || "").trim()).filter(Boolean);
  const nonInstrumental = values.filter((value) => !isInstrumentalLyricText(value));
  if (!nonInstrumental.length) return values[0] || instrumentalText;
  return Array.from(new Set(nonInstrumental)).join("\n");
}

function createSegmentsFromTimestampedLyricsPayload(payload, options = {}) {
  const sourceSegments = Array.isArray(payload?.segments) ? payload.segments : [];
  const segmentMode = String(options.segmentMode || payload?.segment_mode || payload?.segmentMode || "");
  const preservesReferenceUnits = ["reference_lines", "exact_reference_lines", "reference_stanzas"].includes(segmentMode);
  const orderedSource = sourceSegments
    .map((item) => ({
      item,
      start: Math.max(0, Number(item?.start || 0)),
      end: Math.max(0, Number(item?.end || 0)),
    }))
    .filter(({ start, end }) => Number.isFinite(start) && Number.isFinite(end) && end > start + 0.01)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const created = [];
  const instrumentalText = String(options.instrumentalText || payload?.instrumental_text || payload?.instrumentalText || "[instrumental]").trim() || "[instrumental]";
  const shouldFillGaps = options.includeInstrumentalGaps !== false && payload?.include_instrumental_gaps !== false && payload?.includeInstrumentalGaps !== false;
  const minGap = Math.max(0, Number(options.minGapSeconds ?? payload?.min_gap_seconds ?? payload?.minGapSeconds ?? 0.25) || 0);
  const maxScene = Math.max(0.5, Number(options.maxSceneSeconds ?? payload?.max_scene_seconds ?? payload?.maxSceneSeconds ?? 8) || 8);
  const softMaxScene = maxScene + 2;
  const epsilon = 0.03;
  const ordered = [];
  for (const source of orderedSource) {
    const duration = source.end - source.start;
    // Reference modes define scene boundaries in the pasted text. Their Python
    // extractor has already handled timing, and exact mode explicitly promises
    // that a reference line will never be split by duration limits.
    if (preservesReferenceUnits || duration <= softMaxScene + epsilon) {
      ordered.push(source);
      continue;
    }
    const timedWords = (Array.isArray(source.item?.words) ? source.item.words : [])
      .map((word) => ({
        ...word,
        start: Number(word?.start),
        end: Number(word?.end ?? word?.start),
      }))
      .filter((word) => Number.isFinite(word.start) && Number.isFinite(word.end) && word.end > word.start)
      .sort((a, b) => a.start - b.start || a.end - b.end);
    let partStart = source.start;
    let wordCursor = 0;
    while (source.end - partStart > softMaxScene + epsilon) {
      const target = partStart + maxScene;
      const limit = partStart + softMaxScene;
      const candidates = timedWords
        .map((word, index) => ({ word, index }))
        .filter(({ word, index }) => index >= wordCursor && word.end > partStart + 0.1 && word.end <= limit + epsilon);
      const chosen = candidates.length
        ? candidates.reduce((best, candidate) => Math.abs(candidate.word.end - target) < Math.abs(best.word.end - target) ? candidate : best)
        : null;
      const partEnd = chosen ? chosen.word.end : target;
      const partWords = chosen ? timedWords.slice(wordCursor, chosen.index + 1) : [];
      const partItem = { ...source.item };
      if (partWords.length) {
        partItem.words = partWords;
        partItem.text = partWords.map((word) => String(word.text || word.word || "").trim()).filter(Boolean).join(" ") || source.item?.text || "";
        wordCursor = chosen.index + 1;
      } else {
        partItem.words = [];
      }
      partItem.timing_warning = `Long transcription chunk split near a word boundary to honor the ${maxScene.toFixed(2)} second maximum.`;
      ordered.push({ item: partItem, start: partStart, end: partEnd });
      partStart = partEnd;
    }
    const finalItem = { ...source.item };
    const finalWords = timedWords.slice(wordCursor).filter((word) => word.end > partStart - epsilon);
    if (finalWords.length) {
      finalItem.words = finalWords;
      finalItem.text = finalWords.map((word) => String(word.text || word.word || "").trim()).filter(Boolean).join(" ") || source.item?.text || "";
    }
    finalItem.timing_warning = `Continuation of a long transcription chunk split near word boundaries.`;
    ordered.push({ item: finalItem, start: partStart, end: source.end });
  }
  let cursor = 0;
  const addTimestampedSegment = (start, end, item = null, forcedText = "") => {
    const cleanStart = Math.max(0, Number(start || 0));
    const cleanEnd = Math.max(cleanStart + 0.05, Number(end || cleanStart + 4));
    const segment = newSegment(cleanStart, cleanEnd);
    segment.label = `SCENE ${created.length + 1}`;
    segment.source = "timestamped_lyrics";
    segment.timeline_note = item ? String(item?.timing_warning || "").trim() : "";
    segment.lyric_text = cleanTimestampedLyricText(forcedText || item?.text || "") || instrumentalText;
    segment.lyric_no_lip_sync = !item || String(item?.type || "").toLowerCase() === "instrumental" || isInstrumentalLyricText(segment.lyric_text);
    created.push(segment);
  };
  const addInstrumentalGap = (start, end) => {
    if (!shouldFillGaps) return false;
    const cleanStart = Math.max(0, Number(start || 0));
    const cleanEnd = Math.max(cleanStart, Number(end || cleanStart));
    if (cleanEnd - cleanStart < Math.max(minGap, epsilon)) return false;
    let gapStart = cleanStart;
    let added = false;
    while (gapStart < cleanEnd - epsilon) {
      const gapEnd = Math.min(cleanEnd, gapStart + maxScene);
      addTimestampedSegment(gapStart, gapEnd, null, instrumentalText);
      added = true;
      gapStart = gapEnd;
    }
    return added;
  };
  for (const { item, start, end } of ordered) {
    let segmentStart = start;
    if (start > cursor + epsilon) {
      const filledGap = addInstrumentalGap(cursor, start);
      if (!filledGap) {
        if (created.length) {
          created[created.length - 1].end = start;
        } else {
          segmentStart = cursor;
        }
      }
    }
    addTimestampedSegment(segmentStart, end, item);
    cursor = Math.max(cursor, end);
  }
  const duration = Number(payload?.duration || 0);
  if (Number.isFinite(duration) && duration > cursor + epsilon) {
    const filledTail = addInstrumentalGap(cursor, duration);
    if (!filledTail && created.length) created[created.length - 1].end = duration;
  }
  sortSegments(created);
  created.forEach((segment, index) => {
    segment.label = `SCENE ${index + 1}`;
  });
  return created;
}

function normalizeLyricSectionLookupText(value = "") {
  return String(value || "")
    .toLowerCase()
    .replace(/[\u2018\u2019'`]/g, "")
    .replace(/[^\p{L}\p{N}' ]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function lyricSectionMapFromReferenceText(text = "") {
  const map = new Map();
  let currentSection = "";
  for (const rawLine of String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const tags = Array.from(line.matchAll(/\[([^\]]{1,80})\]/g));
    const tagPrefix = tags.length && tags[0].index === 0
      ? line.match(/^(?:\s*\[[^\]]{1,80}\])+\s*/)?.[0] || ""
      : "";
    const structural = tags
      .map((match) => String(match[1] || "").replace(/\s+/g, " ").trim())
      .find((label) => /^(?:intro|verse|pre[\s-]?chorus|chorus|post[\s-]?chorus|bridge|outro|refrain|hook|breakdown|drop|interlude|instrumental(?:\s+break)?|solo|break|spoken(?:\s+word)?|rap)(?:\s+(?:\d+|[ivxlcdm]+))?$/i.test(label));
    const terminal = tags.some((match) => /^(?:end|end of song)$/i.test(String(match[1] || "").trim()));
    if (structural && tagPrefix) {
      currentSection = structural;
      const lyricRemainder = line.slice(tagPrefix.length).trim();
      const key = normalizeLyricSectionLookupText(lyricRemainder);
      if (key && !map.has(key)) map.set(key, currentSection);
      continue;
    }
    if (terminal && tagPrefix && !line.slice(tagPrefix.length).trim()) {
      currentSection = "";
      continue;
    }
    const key = normalizeLyricSectionLookupText(line);
    if (key && currentSection && !map.has(key)) map.set(key, currentSection);
  }
  return map;
}

export function applyLyricSectionsFromReferenceText(segments = [], referenceText = "") {
  const sectionMap = lyricSectionMapFromReferenceText(referenceText);
  if (!sectionMap.size) return 0;
  const sectionEntries = Array.from(sectionMap.entries());
  let applied = 0;
  for (const segment of segments) {
    if (!segment) continue;
    const existingSection = String(segment.lyric_section || "").trim().toLowerCase();
    const lyricText = String(segment.lyric_text || "").trim();
    // Instrumental is a derived status when a scene contains no lyric text.
    // Do not let that old marker block a later lyric correction from getting
    // its real section assigned.
    if (existingSection && existingSection !== "instrumental") continue;
    if (!lyricText || isInstrumentalLyricText(lyricText)) {
      segment.lyric_section = "instrumental";
      continue;
    }
    const lyricLines = String(segment.lyric_text || "")
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    for (const line of lyricLines) {
      const header = line.match(/^\[([^\]]{2,80})\]$/);
      if (header) {
        segment.lyric_section = header[1].trim();
        applied += 1;
        break;
      }
      const section = sectionMap.get(normalizeLyricSectionLookupText(line));
      if (section) {
        segment.lyric_section = section;
        applied += 1;
        break;
      }
      const key = normalizeLyricSectionLookupText(line);
      if (key.length >= 6) {
        const fuzzy = sectionEntries.find(([referenceKey]) =>
          referenceKey.length >= 6 && (referenceKey.includes(key) || key.includes(referenceKey))
        );
        if (fuzzy?.[1]) {
          segment.lyric_section = fuzzy[1];
          applied += 1;
          break;
        }
      }
    }
  }
  return applied;
}

export function createLyricTranscription({
  activateGlobalTimelineAudioPlayback, activeProjectFolderForSave, allEditableSegments,
  applyLyricMapperToSegments, audio, audioInput, autoSaveSessionQuiet, buildShotAlignedSingerCueMap,
  createProgressWindow, currentProjectAudioPath, currentVideoMode, ensureAutoTimedSingerCuesBeforePrompt,
  ensureSegmentRuntimeFields, flfTransitionLoraActive, formatLyricSegmentText, logicalSubjectIdsForScene,
  miniMaxAutoTimeAllScenesButton, normalizeLyricCueMapForSegment, pauseAllAudio,
  previousAutoChainSourceSegment, projectInput, projectPromptsPath, pushHistory, render,
  renderMiniMaxSpeakerAssignmentPanel, saveSession, sceneDisplayName, sceneSlotNumber, segmentIndexInfo,
  segmentMappedLocationReference, segmentMappedLocationText, selectedPerformerSubjectsForSegment,
  startSilentTimelinePlayback, state, syncInspector, syncLyricNoteControls, updateAudioScrubbers,
  updatePlayPauseButton,
}) {
  const singerCuePlayback = { button: null, label: "" };

  let singerCueStopAt = null;
  let singerCueStopTimer = 0;
  let singerCueStopRaf = 0;

  async function autoTimeAllMiniMaxSingerScenes() {
    const candidates = (Array.isArray(state.segments) ? state.segments : [])
      .filter((segment) => segment && !segment.no_character_present)
      .filter((segment) => normalizeVideoType(segment.performance_mode || state.videoType) === "singing")
      .filter((segment) => !isInstrumentalLyricText(segment.lyric_text) && flattenLyricForPrompt(segment.lyric_text));
    if (!candidates.length) {
      toast("No singing scenes with lyric text are available to auto-time.", true);
      return;
    }
    const originalLabel = miniMaxAutoTimeAllScenesButton.textContent;
    miniMaxAutoTimeAllScenesButton.disabled = true;
    miniMaxAutoTimeAllScenesButton.textContent = `Auto Timing 0/${candidates.length}`;
    let completed = 0;
    let skipped = 0;
    const failures = [];
    try {
      for (let index = 0; index < candidates.length; index += 1) {
        const segment = candidates[index];
        miniMaxAutoTimeAllScenesButton.textContent = `Auto Timing ${index + 1}/${candidates.length}`;
        const existing = normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
        const timingComplete = existing.length && existing.every((cue, cueIndex) => {
          const start = Number(cue.start);
          const end = miniMaxEffectiveCueEnd(segment, existing, cueIndex, cue);
          return Number.isFinite(start) && Number.isFinite(Number(end)) && Number(end) > start;
        });
        if (timingComplete) {
          skipped += 1;
          continue;
        }
        try {
          const timed = await ensureAutoTimedSingerCuesBeforePrompt(segment, { force: true });
          if (timed) completed += 1;
          else failures.push(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: no usable lyric timing was created.`);
        } catch (error) {
          failures.push(String(error?.message || error));
        }
      }
      renderMiniMaxSpeakerAssignmentPanel();
      syncInspector();
      render();
      await autoSaveSessionQuiet("MiniMax singer cues auto-timed for all scenes");
      const summary = `Auto Time All Scenes finished. Timed: ${completed}. Already timed: ${skipped}. Failed: ${failures.length}.`;
      toast(failures.length ? `${summary}\n${failures.slice(0, 3).join("\n")}${failures.length > 3 ? `\n…and ${failures.length - 3} more.` : ""}` : `${summary}\nReview each scene's cue rows before creating prompts.`, Boolean(failures.length));
    } finally {
      miniMaxAutoTimeAllScenesButton.disabled = false;
      miniMaxAutoTimeAllScenesButton.textContent = originalLabel;
    }
  }

  function clearSingerCuePlaybackStop() {
    if (singerCueStopTimer) window.clearTimeout(singerCueStopTimer);
    if (singerCueStopRaf) window.cancelAnimationFrame(singerCueStopRaf);
    singerCueStopTimer = 0;
    singerCueStopRaf = 0;
    singerCueStopAt = null;
  }

  function stopSingerCuePlaybackAtBoundary() {
    const end = singerCueStopAt;
    clearSingerCuePlaybackStop();
    audio.pause();
    if (Number.isFinite(end)) {
      try {
        audio.currentTime = end;
        state.sceneAudioGlobalTime = end;
      } catch {
        // Browser may reject seeks before metadata is settled.
      }
    }
    resetSingerCuePlayButton();
    updateAudioScrubbers();
  }

  function armSingerCuePlaybackStop(start, end) {
    clearSingerCuePlaybackStop();
    const safeStart = Math.max(0, Number(start) || 0);
    const safeEnd = Math.max(safeStart + 0.1, Number(end) || safeStart + 0.1);
    singerCueStopAt = safeEnd;
    singerCueStopTimer = window.setTimeout(stopSingerCuePlaybackAtBoundary, Math.max(40, (safeEnd - safeStart) * 1000 + 30));
    const tick = () => {
      if (singerCueStopAt == null) return;
      if (audio.currentTime >= singerCueStopAt - 0.012) {
        stopSingerCuePlaybackAtBoundary();
        return;
      }
      singerCueStopRaf = window.requestAnimationFrame(tick);
    };
    singerCueStopRaf = window.requestAnimationFrame(tick);
  }

  function segmentsShareMappedLocation(firstSegment, secondSegment) {
    const firstLocation = segmentMappedLocationReference(firstSegment);
    const secondLocation = segmentMappedLocationReference(secondSegment);
    return Boolean(firstLocation?.id && secondLocation?.id && String(firstLocation.id) === String(secondLocation.id));
  }

  function flfSameLocationCameraDiversityDirection(segment, target = "start") {
    if (currentVideoMode() !== "flf" || !segment) return "";
    const previous = previousAutoChainSourceSegment(segment);
    if (!previous || !segmentsShareMappedLocation(previous, segment)) return "";
    const location = segmentMappedLocationReference(segment);
    const targetLabel = String(target || "start").toLowerCase() === "end" ? "destination/end frame" : "start frame";
    return [
      `REQUIRED SAME-LOCATION CAMERA CHANGE: The immediately previous scene and this scene both use the mapped location \"${String(location?.name || "the same location").trim()}\".`,
      `For this scene's ${targetLabel}, preserve the location's identity but move to a genuinely different part of its navigable three-dimensional space. Do not reuse the previous scene's exact area, camera placement, viewing direction, framing, or subject blocking.`,
      "Change several composition variables together: camera position and height, viewing direction, shot size or lens, character placement, and foreground/background arrangement. Reveal different architecture, furniture, landmarks, depth layers, or environmental details whenever the location allows it.",
      "Treat the previous scene image only as environment and character continuity evidence plus a negative composition reference. Do not recreate it, closely imitate it, or merely make a small crop/zoom adjustment.",
      "Place the character physically inside the location with correct perspective, scale, floor contact, depth, contact shadows, reflections, environmental color spill, and natural occlusion. Never paste, composite, cut out, green-screen, or layer the character over the location reference.",
    ].join("\n");
  }

  function mappedTriggerPartsForSegment(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const parts = { start: [], end: [] };
    if (!segment) return parts;
    const positionFor = (kind) => {
      if (kind === "subject") return refs.subject_trigger_position === "end" ? "end" : "start";
      if (kind === "location") return refs.location_trigger_position === "end" ? "end" : "start";
      return refs.trigger_position === "end" ? "end" : "start";
    };
    const add = (trigger, kind = "location") => {
      const text = String(trigger || "").trim();
      if (!text) return;
      const key = positionFor(kind);
      if (!parts[key].some((item) => item.toLowerCase() === text.toLowerCase())) parts[key].push(text);
    };
    const indexKey = String(segmentIndexInfo(segment).index + 1);
    if (!segment.no_character_present) {
      const subjectIds = logicalSubjectIdsForScene(refs, segment, Number(indexKey) - 1);
      subjectIds
        .map((id) => refs.subjects.find((subject) => subject.id === id))
        .filter(Boolean)
        .forEach((subject) => add(subject.trigger_phrase, "subject"));
    }
    const locId = refs.scene_map?.[segment.id] || refs.scene_map?.[indexKey] || "";
    const location = refs.locations.find((item) => item.id === locId);
    if (location) add(location.trigger_phrase, "location");
    return parts;
  }

  function applyMappedTriggerPhrases(prompt, segment, options = {}) {
    let text = cleanGeneratedPromptText(prompt);
    const parts = mappedTriggerPartsForSegment(segment);
    const starts = parts.start.filter(Boolean);
    const ends = parts.end.filter(Boolean);
    const stripBoundaryTrigger = (value, trigger) => {
      let current = String(value || "").trim();
      const escaped = String(trigger || "").trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!escaped) return current;
      const leading = new RegExp(`^\\s*${escaped}\\s*(?:,\\s*)?`, "i");
      const trailing = new RegExp(`(?:,\\s*)?${escaped}\\s*$`, "i");
      let previous = "";
      while (current && current !== previous) {
        previous = current;
        current = current.replace(leading, "").replace(trailing, "").trim();
      }
      return current;
    };
    [...starts, ...ends]
      .sort((a, b) => b.length - a.length)
      .forEach((trigger) => {
        text = stripBoundaryTrigger(text, trigger);
      });
    if (starts.length) {
      const prefix = starts.join(", ");
      if (!text.toLowerCase().startsWith(prefix.toLowerCase())) text = text ? `${prefix}, ${text}` : prefix;
    }
    if (ends.length) {
      const suffix = ends.join(", ");
      if (!text.toLowerCase().endsWith(suffix.toLowerCase())) text = text ? `${text}, ${suffix}` : suffix;
    }
    return options.ensureTransitionLast ? ensureTransitionLoraTriggerIsLast(text, segment) : text;
  }

  function ensureTransitionLoraTriggerIsLast(prompt, segment) {
    let text = cleanGeneratedPromptText(prompt);
    if (!segment || !flfTransitionLoraActive(segment)) return text;
    const trigger = String(state.autoChainTransitionTrigger || "zhuanchang").trim().replace(/\s+/g, " ");
    if (!trigger) return text;
    const escaped = trigger.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    text = text
      .replace(new RegExp(`(^|[\\s,;:.!?])${escaped}(?=$|[\\s,;:.!?])`, "gi"), "$1")
      .replace(/\s+([,;:.!?])/g, "$1")
      .replace(/([,;])\s*([,;])+/g, "$1")
      .replace(/\s{2,}/g, " ")
      .trim()
      .replace(/[\s,;:.!?]+$/, "");
    return text ? `${text}, ${trigger}` : trigger;
  }

  function formatSubjectSegmentText(subjects = null) {
    const segments = allEditableSegments();
    return segments.map((segment, index) => {
      const value = Array.isArray(subjects) ? subjects[index] : segmentSingerSubjectText(segment);
      return `subjectSegment${index + 1}=${String(value || "").trim()}`;
    }).join("\n") + "\n";
  }

  function lyricSubjectJsonFromSegments() {
    const data = {};
    allEditableSegments().forEach((segment, index) => {
      data[`scene${index + 1}`] = {
        subject: segmentSingerSubjectText(segment),
        lyric: String(segment?.lyric_text || "").trim(),
      };
    });
    return JSON.stringify(data, null, 2) + "\n";
  }

  function locationJsonFromSegments() {
    const data = {};
    allEditableSegments().forEach((segment, index) => {
      data[`scene${index + 1}`] = {
        location: segmentMappedLocationText(segment),
      };
    });
    return JSON.stringify(data, null, 2) + "\n";
  }

  function subjectLocationJsonFromSegments() {
    const data = {};
    allEditableSegments().forEach((segment, index) => {
      data[`scene${index + 1}`] = {
        subject: segmentSingerSubjectText(segment),
        location: segmentMappedLocationText(segment),
      };
    });
    return JSON.stringify(data, null, 2) + "\n";
  }

  function lyricSubjectLocationJsonFromSegments() {
    const data = {};
    allEditableSegments().forEach((segment, index) => {
      data[`scene${index + 1}`] = {
        subject: segmentSingerSubjectText(segment),
        location: segmentMappedLocationText(segment),
        lyric: String(segment?.lyric_text || "").trim(),
      };
    });
    return JSON.stringify(data, null, 2) + "\n";
  }

  function projectLyricNotesPath() {
    return projectPromptsPath("lyric_notes.txt");
  }

  function projectSubjectNotesPath() {
    return projectPromptsPath("subject_notes.txt");
  }

  function projectLyricSubjectNotesPath() {
    return projectPromptsPath("lyric_subject_notes.json");
  }

  function projectLocationNotesPath() {
    return projectPromptsPath("location_notes.json");
  }

  function projectSubjectLocationNotesPath() {
    return projectPromptsPath("subject_location_notes.json");
  }

  function projectLyricSubjectLocationNotesPath() {
    return projectPromptsPath("lyric_subject_location_notes.json");
  }

  async function syncLyricNotesFromSegments(reason = "") {
    const path = projectLyricNotesPath();
    if (!path) return false;
    try {
      await savePromptTextFile(path, formatLyricSegmentText());
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync lyric notes after ${reason || "segment change"}:`, error);
      toast(`Could not update lyric_notes.txt:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncSubjectNotesFromSegments(reason = "") {
    const path = projectSubjectNotesPath();
    if (!path) return false;
    try {
      await savePromptTextFile(path, formatSubjectSegmentText());
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync subject notes after ${reason || "segment change"}:`, error);
      toast(`Could not update subject_notes.txt:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncLyricSubjectNotesFromSegments(reason = "") {
    const path = projectLyricSubjectNotesPath();
    if (!path) return false;
    try {
      await savePromptTextFile(path, lyricSubjectJsonFromSegments());
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync lyric/subject notes after ${reason || "segment change"}:`, error);
      toast(`Could not update lyric_subject_notes.json:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncLocationNotesFromSegments(reason = "") {
    const path = projectLocationNotesPath();
    if (!path) return false;
    try {
      await savePromptTextFile(path, locationJsonFromSegments());
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync location notes after ${reason || "segment change"}:`, error);
      toast(`Could not update location_notes.json:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncSubjectLocationNotesFromSegments(reason = "") {
    const path = projectSubjectLocationNotesPath();
    if (!path) return false;
    try {
      await savePromptTextFile(path, subjectLocationJsonFromSegments());
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync subject/location notes after ${reason || "segment change"}:`, error);
      toast(`Could not update subject_location_notes.json:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncLyricSubjectLocationNotesFromSegments(reason = "") {
    const path = projectLyricSubjectLocationNotesPath();
    if (!path) return false;
    try {
      await savePromptTextFile(path, lyricSubjectLocationJsonFromSegments());
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync lyric/subject/location notes after ${reason || "segment change"}:`, error);
      toast(`Could not update lyric_subject_location_notes.json:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncLyricAndSubjectNoteFiles(reason = "") {
    await syncLyricNotesFromSegments(reason);
    await syncSubjectNotesFromSegments(reason);
    await syncLyricSubjectNotesFromSegments(reason);
    await syncLocationNotesFromSegments(reason);
    await syncSubjectLocationNotesFromSegments(reason);
    await syncLyricSubjectLocationNotesFromSegments(reason);
  }

  function showTranscribeLyricsModal() {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(720px,calc(100vw - 40px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
      const title = document.createElement("div");
      title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Transcribe Lines For Timeline</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Uses the current audio and builder SRT timing to fill each scene's Line / lyric / dialogue field.</div>`;
      const close = makeButton("Close");
      header.append(title, close);
      const note = document.createElement("div");
      note.textContent = "Full reference lyrics or dialogue are required. Their exact line order is preserved while stable-ts determines where each line belongs in the existing scenes.";
      note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
      const lyrics = document.createElement("textarea");
      lyrics.value = String(state.lyricMapper?.source_text || "");
      lyrics.placeholder = "Required full reference lyrics or dialogue...";
      lyrics.style.cssText = "width:100%;box-sizing:border-box;min-height:210px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;font-family:monospace;";
      const language = makeInput("english");
      const mode = makeSelect(["fill_missing", "replace_all"], "replace_all");
      mode.options[0].textContent = "Fill missing only";
      mode.options[1].textContent = "Replace all line notes";
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
      grid.append(
        makeField("Language", language),
        makeField("Apply mode", mode),
      );
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const run = makeButton("Transcribe Lines", "primary");
      actions.append(cancel, run);
      box.append(header, note, makeField("Reference lyrics/dialogue", lyrics), grid, actions);
      backdrop.append(box);
      document.body.append(backdrop);
      const finish = (value) => {
        backdrop.remove();
        resolve(value);
      };
      close.onclick = () => finish(null);
      cancel.onclick = () => finish(null);
      run.onclick = () => {
        if (!confirmParenthesizedReferenceLyrics(lyrics.value)) {
          lyrics.focus();
          return;
        }
        finish({
          referenceLyrics: lyrics.value || "",
          language: language.value || "english",
          replaceAll: mode.value === "replace_all",
        });
      };
      backdrop.addEventListener("pointerdown", (event) => {
        if (event.target === backdrop) finish(null);
      });
    });
  }

  function showTimestampedTranscribeModal() {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(820px,calc(100vw - 40px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
      const title = document.createElement("div");
      title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Create Scenes From Timestamped Lines</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Uses audio and optional reference lyrics/dialogue to create timeline scenes before an SRT exists.</div>`;
      const close = makeButton("Close");
      header.append(title, close);
      const warning = document.createElement("div");
      warning.style.cssText = "font-size:12px;line-height:1.45;border:1px solid #92400e;border-radius:7px;background:#451a03;color:#fed7aa;padding:9px;";
      warning.textContent = "This replaces the current base timeline scenes with timestamped lyric scenes. Existing generated images/videos are not deleted, but they may no longer line up with the new timing.";
      const lyrics = document.createElement("textarea");
      lyrics.value = String(state.lyricMapper?.source_text || "");
      lyrics.placeholder = "Optional reference lyrics/dialogue. Put each desired scene chunk on its own line. Use [instrumental] or [instrumental break] for no-vocal sections.";
      lyrics.style.cssText = "width:100%;box-sizing:border-box;min-height:230px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;font-family:monospace;";
      const language = makeInput("english");
      const segmentMode = makeSelect(["whisper_chunks", "reference_lines", "exact_reference_lines", "reference_stanzas", "beat_scenes"], "reference_lines");
      segmentMode.options[0].textContent = "Whisper chunks";
      segmentMode.options[1].textContent = "One scene per lyric line";
      segmentMode.options[2].textContent = "Exact reference lyric lines (no duration limits)";
      segmentMode.options[3].textContent = "One scene per stanza";
      segmentMode.options[4].textContent = "Beat mode";
      const exactModeNote = document.createElement("div");
      exactModeNote.style.cssText = "display:none;border:1px solid #0e7490;border-radius:7px;background:#083344;color:#cffafe;padding:10px;font-size:12px;line-height:1.5;";
      exactModeNote.innerHTML = "<strong>Exact Reference Lyric Lines:</strong> uses the normal lyric transcription/alignment, instrumental-gap detection, vocal-tail padding, and gap/overlap cleanup. Every non-empty pasted lyric line remains one vocal scene using that exact text. The only disabled rules are minimum and maximum scene duration, so a line is not stretched, split, or merged just to satisfy those two limits.";
      const referenceDurationNote = document.createElement("div");
      referenceDurationNote.style.cssText = "display:none;border:1px solid #0e7490;border-radius:7px;background:#083344;color:#cffafe;padding:10px;font-size:12px;line-height:1.5;";
      const includeGaps = makeCheckbox("Include instrumental gaps", true);
      const instrumentalText = makeInput("[instrumental]");
      const minGap = makeInput("2.0");
      const minScene = makeInput("1.0");
      const maxScene = makeInput("8.0");
      const vocalTail = makeInput("0.6");
      const beatUseSrtDurations = makeCheckbox("Use beat/SRT durations", true);
      const beatFixedSceneDuration = makeInput("4");
      const beatMinDuration = makeInput("4");
      const beatMaxDuration = makeInput("10");
      const beatBias = makeInput("0.7");
      const beatDurationPreset = makeSelect(["varied_no_repeat", "impact_weighted", "clustered_no_repeat"], "varied_no_repeat");
      beatDurationPreset.options[0].textContent = "Varied, no repeat";
      beatDurationPreset.options[1].textContent = "Impact weighted";
      beatDurationPreset.options[2].textContent = "Clustered, no repeat";
      const beatEmptySegmentText = makeInput("Instrumental section.");
      const hint = makeButton("?");
      hint.title = "Explain timestamped line settings";
      hint.style.width = "44px";
      hint.onclick = async () => {
        await showTimestampedLyricsHintModal();
      };
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
      const instrumentalTextField = makeField("Instrumental text", instrumentalText);
      const minGapField = makeField("Min gap seconds", minGap);
      const minSceneField = makeField("Min scene seconds", minScene);
      const maxSceneField = makeField("Max scene seconds", maxScene);
      const vocalTailField = makeField("Vocal tail padding", vocalTail);
      const timestampFields = [instrumentalTextField, minGapField, minSceneField, maxSceneField, vocalTailField];
      grid.append(
        makeField("Language", language),
        makeField("Segment mode", segmentMode),
        ...timestampFields,
      );
      const beatGrid = document.createElement("div");
      beatGrid.style.cssText = "display:none;grid-template-columns:1fr 1fr;gap:10px;border:1px solid #334155;border-radius:7px;padding:10px;background:#0f172a;";
      const beatSrtFields = [
        makeField("Duration preset", beatDurationPreset),
        makeField("Min duration", beatMinDuration),
        makeField("Max duration", beatMaxDuration),
        makeField("Bias", beatBias),
      ];
      const beatFixedField = makeField("Fixed scene duration", beatFixedSceneDuration);
      beatGrid.append(
        beatUseSrtDurations.wrapper,
        ...beatSrtFields,
        beatFixedField,
        makeField("Empty lyric segment text", beatEmptySegmentText),
      );
      const updateModeUi = () => {
        const isBeatMode = segmentMode.value === "beat_scenes";
        const isExactMode = segmentMode.value === "exact_reference_lines";
        const usingBeatSrt = Boolean(beatUseSrtDurations.input.checked);
        beatGrid.style.display = isBeatMode ? "grid" : "none";
        timestampFields.forEach((field) => {
          field.style.display = isBeatMode ? "none" : "";
        });
        minSceneField.style.display = isBeatMode || isExactMode ? "none" : "";
        maxSceneField.style.display = isBeatMode || isExactMode ? "none" : "";
        vocalTailField.style.display = isBeatMode ? "none" : "";
        exactModeNote.style.display = isExactMode ? "block" : "none";
        const isReferenceLineMode = segmentMode.value === "reference_lines";
        const isReferenceStanzaMode = segmentMode.value === "reference_stanzas";
        referenceDurationNote.style.display = isReferenceLineMode || isReferenceStanzaMode ? "block" : "none";
        referenceDurationNote.innerHTML = isReferenceLineMode
          ? "<strong>Lyric-line duration:</strong> each pasted line stays one scene. If “Playing in the rain” is only 1.5 seconds long, it remains a 1.5-second scene. Minimum and maximum duration still guide instrumental or approximate timing, but they never split or merge pasted lyric lines."
          : "<strong>Stanza duration:</strong> each pasted stanza stays one scene even when it is shorter than the minimum or longer than the maximum. Duration limits still guide instrumental or approximate timing.";
        gapRow.style.display = isBeatMode ? "none" : "flex";
        beatSrtFields.forEach((field) => {
          field.style.display = usingBeatSrt ? "" : "none";
        });
        beatFixedField.style.display = usingBeatSrt ? "none" : "";
      };
      segmentMode.addEventListener("change", updateModeUi);
      beatUseSrtDurations.input.addEventListener("change", updateModeUi);
      const gapRow = document.createElement("div");
      gapRow.style.cssText = "display:flex;align-items:center;gap:10px;";
      gapRow.append(includeGaps.wrapper, hint);
      updateModeUi();
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const run = makeButton("Create Timeline Scenes", "primary");
      actions.append(cancel, run);
      box.append(header, warning, makeField("Reference lyrics/dialogue", lyrics), exactModeNote, referenceDurationNote, grid, beatGrid, gapRow, actions);
      backdrop.append(box);
      document.body.append(backdrop);
      const finish = (value) => {
        backdrop.remove();
        resolve(value);
      };
      close.onclick = () => finish(null);
      cancel.onclick = () => finish(null);
      run.onclick = async () => {
        if (!confirmParenthesizedReferenceLyrics(lyrics.value)) {
          lyrics.focus();
          return;
        }
        finish({
          referenceLyrics: lyrics.value || "",
          language: language.value || "english",
          segmentMode: segmentMode.value || "reference_lines",
          includeInstrumentalGaps: Boolean(includeGaps.input.checked),
          instrumentalText: instrumentalText.value || "[instrumental]",
          minGapSeconds: Number(minGap.value || 2.0),
          minSceneSeconds: Number(minScene.value || 1.0),
          maxSceneSeconds: Number(maxScene.value || 8.0),
          vocalTailPaddingSeconds: Number(vocalTail.value || 0.6),
          beatUseSrtDurations: Boolean(beatUseSrtDurations.input.checked),
          beatFixedSceneDuration: Number(beatFixedSceneDuration.value || 4),
          beatMinDuration: Number(beatMinDuration.value || 4),
          beatMaxDuration: Number(beatMaxDuration.value || 10),
          beatBias: Number(beatBias.value || 0.7),
          beatDurationPreset: beatDurationPreset.value || "varied_no_repeat",
          beatEmptySegmentText: beatEmptySegmentText.value || "Instrumental section.",
        });
      };
      backdrop.addEventListener("pointerdown", (event) => {
        if (event.target === backdrop) finish(null);
      });
    });
  }

  function normalizeTimestampedSceneDurations(segments = [], options = {}, payload = {}) {
    const segmentMode = String(options.segmentMode || payload?.segment_mode || payload?.segmentMode || "");
    const preservesReferenceUnits = ["reference_lines", "exact_reference_lines", "reference_stanzas"].includes(segmentMode);
    if (preservesReferenceUnits) {
      // The timestamp extractor already applies the requested duration rules to
      // reference-defined units. Merging short units here can combine multiple
      // pasted lyric lines/stanzas and then trip the strict transcription guard.
      return (Array.isArray(segments) ? segments : [])
        .filter(Boolean)
        .map((segment) => ensureSegmentRuntimeFields(segment))
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0) || Number(a.end || 0) - Number(b.end || 0));
    }
    const minSceneSeconds = Math.max(1, Number(options.minSceneSeconds ?? payload?.min_scene_seconds ?? payload?.minSceneSeconds ?? 1) || 1);
    const maxSceneSeconds = Math.max(minSceneSeconds, Number(options.maxSceneSeconds ?? payload?.max_scene_seconds ?? payload?.maxSceneSeconds ?? 8) || 8);
    const softMaxSceneSeconds = maxSceneSeconds + 2;
    const instrumentalText = String(options.instrumentalText || payload?.instrumental_text || payload?.instrumentalText || "[instrumental]").trim() || "[instrumental]";
    const items = (Array.isArray(segments) ? segments : [])
      .filter(Boolean)
      .map((segment) => ensureSegmentRuntimeFields(segment))
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0) || Number(a.end || 0) - Number(b.end || 0));
    if (!items.length) return items;

    for (let index = 0; index < items.length; index += 1) {
      const segment = items[index];
      segment.start = Math.max(0, Number(segment.start || 0));
      segment.end = Math.max(segment.start + 0.05, Number(segment.end || segment.start + 0.05));
      const duration = segment.end - segment.start;
      if (duration >= minSceneSeconds || items.length <= 1) continue;

      const prev = items[index - 1] || null;
      const next = items[index + 1] || null;
      const prevDuration = prev ? Math.max(0, Number(prev.end || 0) - Number(prev.start || 0)) : -1;
      const nextDuration = next ? Math.max(0, Number(next.end || 0) - Number(next.start || 0)) : -1;
      const segmentIsInstrumental = isInstrumentalLyricText(segment.lyric_text);
      const prevIsVocal = prev && !isInstrumentalLyricText(prev.lyric_text);
      const nextIsVocal = next && !isInstrumentalLyricText(next.lyric_text);
      const nearestTarget = next && (!prev || nextDuration <= prevDuration) ? next : prev;
      const target = segmentIsInstrumental
        ? nearestTarget
        : (prevIsVocal ? prev : (nextIsVocal ? next : nearestTarget));
      if (!target && !segmentIsInstrumental) {
        const nextStart = next ? Number(next.start || 0) : null;
        const desiredEnd = segment.start + minSceneSeconds;
        segment.end = nextStart && nextStart > segment.start ? Math.min(desiredEnd, nextStart) : desiredEnd;
        continue;
      }
      if (!target) continue;
      const mergedStart = Math.min(Number(target.start || 0), segment.start);
      const mergedEnd = Math.max(Number(target.end || target.start + minSceneSeconds), segment.end);
      if (mergedEnd - mergedStart > softMaxSceneSeconds + 0.001) continue;

      const targetStartedBeforeSegment = Number(target.start || 0) <= Number(segment.start || 0);
      target.start = Math.min(Number(target.start || 0), segment.start);
      target.end = Math.max(Number(target.end || target.start + minSceneSeconds), segment.end);
      target.lyric_text = targetStartedBeforeSegment
        ? mergeTimestampedLyricText(target.lyric_text, segment.lyric_text, instrumentalText)
        : mergeTimestampedLyricText(segment.lyric_text, target.lyric_text, instrumentalText);
      target.lyric_no_lip_sync = isInstrumentalLyricText(target.lyric_text);
      target.timeline_note = [target.timeline_note, segment.timeline_note]
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join("\n");
      items.splice(index, 1);
      index = Math.max(-1, index - 2);
    }

    for (let index = 0; index < items.length; index += 1) {
      const segment = items[index];
      const prev = items[index - 1] || null;
      if (prev) {
        segment.start = Math.max(Number(prev.end || 0), Number(segment.start || 0));
        if (segment.end <= segment.start + 0.05) segment.end = segment.start + minSceneSeconds;
      }
      segment.end = Math.max(Number(segment.start || 0) + 0.05, Number(segment.end || 0));
      ensureSegmentRuntimeFields(segment);
    }
    return items;
  }

  async function createScenesFromTimestampedLyrics(presetOptions = null) {
    const options = presetOptions || await showTimestampedTranscribeModal();
    if (!options) return false;
    if (presetOptions && !confirmParenthesizedReferenceLyrics(options.referenceLyrics || "")) {
      return false;
    }
    if (!String(audioInput.value || state.audioPath || "").trim()) {
      toast("Load an audio file first.", true);
      return false;
    }
    if (String(options.referenceLyrics || "").trim()) {
      state.lyricMapper = normalizeLyricMapper({
        source_text: String(options.referenceLyrics || "").trim(),
        lines: [],
      });
    }
    let progress = null;
    try {
      progress = createProgressWindow("Creating Scenes From Timestamped Lines");
      let created = [];
      let modeLabel = options.segmentMode || "reference_lines";
      let sourceDuration = 0;
      let beatTranscriptionResult = null;
      if (options.segmentMode === "beat_scenes") {
        if (!String(options.referenceLyrics || "").trim()) {
          throw new Error("Reference lyrics are required for Beat Mode so the created scenes can be transcribed without dropping lyric lines.");
        }
        progress.set("Building Prompt Creator beat/SRT workflow...", 8);
        const built = await postJson("/vrgdg/music_prompt_creator/build_whisper_prompt", {
          project_folder: activeProjectFolderForSave() || projectInput.value || state.projectFolder || "",
          audio_path: audioInput.value || state.audioPath || "",
          min_duration: Number(options.beatMinDuration || 4),
          max_duration: Number(options.beatMaxDuration || 10),
          bias: Number(options.beatBias || 0.7),
          duration_preset: options.beatDurationPreset || "varied_no_repeat",
          use_srt_durations: options.beatUseSrtDurations !== false,
          fixed_scene_duration: Number(options.beatFixedSceneDuration || 4),
          empty_segment_text: options.beatEmptySegmentText || options.instrumentalText || "Instrumental section.",
          whisper_language: options.language || "english",
          full_lyrics: options.referenceLyrics || "",
        }, 60000);
        progress.set("Queueing Prompt Creator beat/SRT workflow...", 16);
        const queued = await queueWorkflowPrompt(built.prompt, {
          onStatus: (message) => progress.set(message, 16),
          idleTimeoutMs: 10 * 60 * 1000,
        });
        const promptId = queued.prompt_id;
        if (!promptId) throw new Error("ComfyUI queued the Prompt Creator beat/SRT workflow but did not return a prompt_id.");
        const textValues = await waitForText(
          promptId,
          (message) => progress.set(`${message}\nPrompt ID: ${promptId}`, 55),
          () => false,
          45 * 60 * 1000,
        );
        const text = extractPromptCreatorWhisperSrtText(textValues);
        if (!text.srt) throw new Error("Beat mode finished, but no SRT timing text was found.");
        created = createSegmentsFromBeatSrt(text.srt);
        sourceDuration = Math.max(0, ...created.map((segment) => Number(segment.end || 0)));
        modeLabel = "Beat mode";
      } else {
        progress.set("Building hidden timestamp transcription workflow...", 8);
        const built = await postJson("/vrgdg/workflow_runner/build_timestamped_transcribe_prompt", {
          audio_path: audioInput.value || state.audioPath || "",
          reference_lyrics: options.referenceLyrics || "",
          language: options.language || "english",
          segment_mode: options.segmentMode || "reference_lines",
          include_instrumental_gaps: Boolean(options.includeInstrumentalGaps),
          instrumental_text: options.instrumentalText || "[instrumental]",
          min_gap_seconds: Number(options.minGapSeconds || 2.0),
          min_scene_seconds: Number(options.minSceneSeconds || 1.0),
          max_scene_seconds: Number(options.maxSceneSeconds || 8.0),
          vocal_tail_padding_seconds: Number(options.vocalTailPaddingSeconds || 0.6),
          model_name: "large-v3",
        }, 60000);
        progress.set("Queueing timestamp transcription workflow...", 16);
        const queued = await queueWorkflowPrompt(built.prompt, {
          onStatus: (message) => progress.set(message, 16),
          idleTimeoutMs: 10 * 60 * 1000,
        });
        const promptId = queued.prompt_id;
        if (!promptId) throw new Error("ComfyUI queued the timestamp transcription workflow but did not return a prompt_id.");
        const textValues = await waitForText(
          promptId,
          (message) => progress.set(`${message}\nPrompt ID: ${promptId}`, 55),
          () => false,
          45 * 60 * 1000,
        );
        const payload = parseTimestampedLyricsOutput(textValues.join("\n"));
        const timestampedSegments = createSegmentsFromTimestampedLyricsPayload(payload, options);
        created = normalizeTimestampedSceneDurations(
          timestampedSegments,
          options,
          payload,
        );
        sourceDuration = Number(payload.duration || 0);
        modeLabel = payload.segment_mode || options.segmentMode || "reference_lines";
      }
      if (!created.length) throw new Error("Timestamped lines did not produce any usable scene segments.");
      // Only the two line modes promise one pasted reference line per scene.
      // Stanza and beat modes intentionally allow several pasted lines in a
      // scene, while Whisper mode does not define scenes from reference lines.
      if (["reference_lines", "exact_reference_lines"].includes(options.segmentMode)) {
        assertNoBundledReferenceLyrics(created.map((segment) => segment.lyric_text || ""), options.referenceLyrics || "");
      }
      if (options.wizardStripLyricParentheses) {
        for (const segment of created) {
          segment.lyric_text = sanitizeWizardReferenceLyrics(segment.lyric_text);
          segment.lyric_no_lip_sync = isInstrumentalLyricText(segment.lyric_text);
        }
      }
      applyLyricSectionsFromReferenceText(created, options.referenceLyrics || "");
      pushHistory();
      state.segments = created;
      state.overlaySegments = [];
      state.activeTrack = "base";
      state.activeId = created[0]?.id || "";
      state.duration = Math.max(Number(sourceDuration || 0), ...created.map((segment) => Number(segment.end || 0)));
      state.showTimelineLyricNotes = true;
      syncLyricNoteControls();
      syncInspector();
      render();
      let lyricPath = projectLyricNotesPath();
      if (options.segmentMode === "beat_scenes") {
        progress.set(`Created ${created.length} beat-aligned scenes. Now transcribing those existing scenes...`, 62);
        beatTranscriptionResult = await transcribeExistingScenesWithOptions({
          referenceLyrics: options.referenceLyrics || "",
          language: options.language || "english",
          replaceAll: true,
          instrumentalText: options.beatEmptySegmentText || options.instrumentalText || "Instrumental section.",
        }, {
          progress,
          recordHistory: false,
        });
        lyricPath = beatTranscriptionResult.lyricPath || lyricPath;
      } else {
        if (lyricPath) await syncLyricAndSubjectNoteFiles("timestamped lyric scene creation");
        if (activeProjectFolderForSave()) {
          await saveSession({ quiet: true, throwOnError: true });
        }
      }
      const beatCoverage = beatTranscriptionResult
        ? `\nPreserved all ${beatTranscriptionResult.referenceLineCount} reference lyric lines by automatically transcribing the finished beat scenes.`
        : "";
      progress.set(`Created ${created.length} timeline scene${created.length === 1 ? "" : "s"} from timestamped lines.\nMode: ${modeLabel}${beatCoverage}\n${lyricPath ? `Saved line notes: ${lyricPath}` : "Line notes saved in session only."}`, 100);
      progress.close(1800);
      toast(`Created ${created.length} timestamped lyric scene${created.length === 1 ? "" : "s"}.`);
      return true;
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      return false;
    }
  }

  async function transcribeExistingScenesWithOptions(options = {}, runtime = {}) {
    const progress = runtime.progress || null;
    const audioPath = String(audioInput.value || state.audioPath || "").trim();
    const referenceLyrics = String(options.referenceLyrics || "").trim();
    const segments = Array.isArray(state.segments) ? state.segments : [];
    if (!audioPath) throw new Error("Load an audio file first.");
    if (!segments.length) throw new Error("Create or import timeline scenes first.");
    if (!referenceLyrics) {
      throw new Error("Reference lyrics are required for Transcribe Existing Scenes so lyric lines cannot be silently omitted.");
    }
    state.lyricMapper = normalizeLyricMapper({
      ...state.lyricMapper,
      source_text: referenceLyrics,
    });

    if (runtime.saveTimelineFirst !== false) {
      progress?.set("Saving current scene boundaries...", 5);
      await autoSaveSessionQuiet("timeline lyric transcription");
    }
    progress?.set("Aligning every reference lyric line...", 12);
    const instrumentalText = String(options.instrumentalText || "[instrumental]").trim() || "[instrumental]";
    const built = await postJson("/vrgdg/workflow_runner/build_timestamped_transcribe_prompt", {
      audio_path: audioPath,
      reference_lyrics: referenceLyrics,
      language: options.language || "english",
      segment_mode: "reference_scene_words",
      include_instrumental_gaps: true,
      instrumental_text: instrumentalText,
      min_gap_seconds: 1.0,
      min_scene_seconds: 1.0,
      max_scene_seconds: 8.0,
      vocal_tail_padding_seconds: 0.6,
      model_name: "large-v3",
    }, 60000);
    progress?.set("Queueing reference-preserving transcription workflow...", 18);
    const queued = await queueWorkflowPrompt(built.prompt, {
      onStatus: (message) => progress?.set(message, 18),
      idleTimeoutMs: 10 * 60 * 1000,
    });
    const promptId = queued.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the transcription workflow but did not return a prompt_id.");
    const textValues = await waitForText(
      promptId,
      (message) => progress?.set(`${message}\nPrompt ID: ${promptId}`, 55),
      () => false,
      45 * 60 * 1000,
    );
    const payload = parseTimestampedLyricsOutput(textValues.join("\n"));
    const sceneMappings = mapTimestampedReferenceLyricsToExistingScenes(
      payload,
      segments,
      referenceLyrics,
      instrumentalText,
    );

    if (runtime.recordHistory !== false) pushHistory();
    let applied = 0;
    for (const mapping of sceneMappings) {
      if (!options.replaceAll && String(mapping.segment.lyric_text || "").trim()) continue;
      mapping.segment.lyric_text = mapping.lyricText;
      mapping.segment.lyric_no_lip_sync = isInstrumentalLyricText(mapping.lyricText);
      applied += 1;
    }
    const sectioned = applyLyricSectionsFromReferenceText(segments, referenceLyrics);
    const mapped = applyLyricMapperToSegments({ overwriteSingers: true });
    state.showTimelineLyricNotes = true;
    syncLyricNoteControls();
    const lyricPath = projectLyricNotesPath();
    if (lyricPath) await syncLyricAndSubjectNoteFiles("timeline lyric transcription");
    syncInspector();
    render();
    if (activeProjectFolderForSave()) {
      await saveSession({ quiet: true, throwOnError: true });
    }
    return {
      applied,
      sectioned,
      mapped,
      lyricPath,
      referenceLineCount: referenceVocalLines(referenceLyrics).length,
    };
  }

  async function transcribeLyricsForTimeline() {
    const options = await showTranscribeLyricsModal();
    if (!options) return;
    let progress = null;
    try {
      progress = createProgressWindow("Transcribing Timeline Lines");
      const result = await transcribeExistingScenesWithOptions(options, { progress });
      progress.set(`Transcribed existing scene windows with all ${result.referenceLineCount} reference lyric lines preserved.\nApplied ${result.applied} scene line note${result.applied === 1 ? "" : "s"}.\nDetected sections on ${result.sectioned} scene${result.sectioned === 1 ? "" : "s"}.\nMapped performers on ${result.mapped} scene${result.mapped === 1 ? "" : "s"}.\nSaved: ${result.lyricPath || "session only"}`, 100);
      progress.close(1800);
      toast(`Transcribed ${result.applied} existing scene${result.applied === 1 ? "" : "s"}; preserved all ${result.referenceLineCount} reference lyric lines.`);
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  async function prepareSceneAudioClipForTimestamping(segment, progress = null) {
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    const customAudioPath = String(segment?.custom_audio_path || "").trim();
    const sourcePath = customAudioPath || currentProjectAudioPath();
    if (!sourcePath) throw new Error("Load project audio, add custom scene audio, or render this built-in-audio scene before auto-timing cues.");
    if (!projectFolder) throw new Error("Create or load a project before auto-timing cues.");
    const sceneNumber = sceneSlotNumber(segment);
    const start = customAudioPath ? audioSourceStart(segment) : Math.max(0, Number(segment.start || 0));
    const duration = customAudioPath ? audioChunkDuration(segment) : timelineSegmentDuration(segment);
    progress?.set(`Preparing ${duration.toFixed(3)}s scene audio clip...`, 10);
    const prepared = await postJson("/vrgdg/workflow_runner/prepare_scene_audio_clip", {
      audio_path: sourcePath,
      project_folder: projectFolder,
      scene_number: sceneNumber,
      start_seconds: start,
      duration_seconds: duration,
    }, 120000);
    return String(prepared.audio_path || "").trim();
  }

  async function autoTimeMiniMaxSingerCuesForSegment(segment) {
    if (!segment) return false;
    const progress = createProgressWindow("Auto Time Singer Cues");
    try {
      const exactShotTiming = Boolean(segment.lyric_shot_word_timing_enabled);
      const performer = selectedPerformerSubjectsForSegment(segment)[0] || {};
      let cues = normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
      if (exactShotTiming && !cues.some((cue) => cue.type !== "instrumental" && flattenLyricForPrompt(cue.text))) {
        const fullLyric = isInstrumentalLyricText(segment.lyric_text) ? "" : flattenLyricForPrompt(segment.lyric_text);
        if (fullLyric) cues = [{ type: "vocal", text: fullLyric, action_note: "", singer_id: performer.id || "", singer_name: performer.name || "", start: null, end: null }];
      }
      const vocalCues = cues.filter((cue) => cue.type !== "instrumental" && flattenLyricForPrompt(cue.text));
      if (!vocalCues.length) throw new Error("Add at least one lyric cue before auto-timing this scene.");
      progress.set("Preparing lyric cue text...", 5);
      const referenceLyrics = vocalCues.map((cue) => flattenLyricForPrompt(cue.text)).filter(Boolean).join("\n");
      const audioPath = await prepareSceneAudioClipForTimestamping(segment, progress);
      const payload = await runTimestampedCueWorkflow(audioPath, referenceLyrics, progress, timelineSegmentDuration(segment));
      const timestamped = timestampedCueSegments(payload);
      if (!timestamped.length) throw new Error("Stable-ts did not return usable cue timestamps for this scene.");
      progress.set("Applying timestamped cue rows...", 88);
      const shotAligned = exactShotTiming ? buildShotAlignedSingerCueMap(segment, timestamped, performer) : [];
      const rebuilt = shotAligned.length ? shotAligned : rebuildSingerCueMapFromTimestampedSegments(segment, cues, timestamped, { minInstrumentalGap: 0.5 });
      if (!rebuilt.length) throw new Error("No usable cue rows were created from the timestamped result.");
      pushHistory();
      segment.lyric_performance_mode = "cue_map";
      segment.lyric_cue_map = rebuilt;
      syncLyricTextFromCueMap(segment);
      renderMiniMaxSpeakerAssignmentPanel();
      syncInspector();
      render();
      await autoSaveSessionQuiet("MiniMax singer cues auto-timed");
      const vocalReturned = timestamped.filter((item) => item.type !== "instrumental").length;
      const instrumentalReturned = rebuilt.filter((item) => item.type === "instrumental").length;
      const warning = vocalReturned !== vocalCues.length
        ? `\nWarning: Stable-ts returned ${vocalReturned} vocal cue${vocalReturned === 1 ? "" : "s"} for ${vocalCues.length} mapped lyric cue${vocalCues.length === 1 ? "" : "s"}. Review before rendering.`
        : "";
      progress.set(`Auto timing complete.\nVocal cues: ${vocalReturned}\nInstrumental gaps: ${instrumentalReturned}${exactShotTiming && shotAligned.length ? "\nExact words matched to storyboard shots." : ""}${warning}`, 100);
      progress.close(2600);
      toast("Singer cue timing filled. Review with Play Cue before rendering.");
      return true;
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      progress.close(6000);
      toast(String(error?.message || error), true);
      return false;
    }
  }

  function resetSingerCuePlayButton() {
    if (singerCuePlayback.button) singerCuePlayback.button.textContent = singerCuePlayback.label || "Play";
    singerCuePlayback.button = null;
    singerCuePlayback.label = "";
  }

  function playSingerCueRange(segment, startRelative = 0, endRelative = null, button = null, label = "Play") {
    if (!segment) return;
    const sceneStart = Math.max(0, Number(segment.start || 0));
    const sceneDuration = Math.max(0.1, Number(segment.end || sceneStart + 0.1) - sceneStart);
    const start = sceneStart + Math.max(0, Math.min(sceneDuration, Number(startRelative) || 0));
    const hasEnd = Number.isFinite(Number(endRelative)) && Number(endRelative) > Number(startRelative);
    const end = hasEnd ? sceneStart + Math.max(0, Math.min(sceneDuration, Number(endRelative))) : null;
    if (button && singerCuePlayback.button === button && !audio.paused) {
      clearSingerCuePlaybackStop();
      audio.pause();
      resetSingerCuePlayButton();
      return;
    }
    pauseAllAudio();
    clearSingerCuePlaybackStop();
    activateGlobalTimelineAudioPlayback(start);
    if (button) {
      resetSingerCuePlayButton();
      singerCuePlayback.button = button;
      singerCuePlayback.label = label || button.textContent || "Play";
      button.textContent = "Pause";
    }
    if (end != null) armSingerCuePlaybackStop(start, end);
    audio.play().then(updatePlayPauseButton).catch(() => startSilentTimelinePlayback(start));
  }

  return {
    applyMappedTriggerPhrases, autoTimeAllMiniMaxSingerScenes, autoTimeMiniMaxSingerCuesForSegment,
    createScenesFromTimestampedLyrics, flfSameLocationCameraDiversityDirection, playSingerCueRange,
    prepareSceneAudioClipForTimestamping, projectLyricNotesPath, syncLyricAndSubjectNoteFiles,
    transcribeExistingScenesWithOptions, transcribeLyricsForTimeline,
  };
}
