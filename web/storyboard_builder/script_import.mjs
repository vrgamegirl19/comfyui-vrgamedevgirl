function storyboardScriptWordCount(value) {
  return (String(value || "").match(/[\p{L}\p{N}'’-]+/gu) || []).length;
}

export function storyboardScriptSpeakerMatchKey(value) {
  return String(value || "")
    .toLocaleLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/^(?:the|a|an)\s+/, "")
    .replace(/\s+/g, " ");
}

export function suggestStoryboardScriptSpeakerMatch(speakerName, characters = []) {
  const speakerKey = storyboardScriptSpeakerMatchKey(speakerName);
  if (!speakerKey) return null;
  const speakerTokens = speakerKey.split(" ").filter(Boolean);
  const scored = (Array.isArray(characters) ? characters : []).map((character) => {
    const characterKey = storyboardScriptSpeakerMatchKey(character?.name);
    if (!characterKey) return null;
    const characterTokens = characterKey.split(" ").filter(Boolean);
    let score = 0;
    if (speakerKey === characterKey) score = 100;
    else if (characterKey.startsWith(`${speakerKey} `) || characterKey.endsWith(` ${speakerKey}`) || characterKey.includes(` ${speakerKey} `)) score = 85;
    else if (speakerKey.startsWith(`${characterKey} `) || speakerKey.endsWith(` ${characterKey}`) || speakerKey.includes(` ${characterKey} `)) score = 80;
    else if (speakerTokens.length && speakerTokens.every((token) => characterTokens.includes(token))) score = 70;
    return score ? { character, score } : null;
  }).filter(Boolean).sort((left, right) => right.score - left.score);
  if (!scored.length) return null;
  if (scored.length > 1 && scored[0].score === scored[1].score) return null;
  return scored[0].character || null;
}

function estimateStoryboardScriptCueSeconds(text) {
  const value = String(text || "").trim();
  if (!value) return 0;
  const words = storyboardScriptWordCount(value);
  // MiniMax native dialogue tends to complete short lines faster than a measured
  // read. Keep the planned clip close to the spoken line so the model is not
  // given several empty seconds that it can fill with invented speech.
  const baseSeconds = (words / 160) * 60;
  const commaPauses = (value.match(/[,;]/g) || []).length * 0.12;
  const strongPauses = (value.match(/[.!?](?=\s|$)/g) || []).length * 0.22;
  const dramaticPauses = (value.match(/[—…]/g) || []).length * 0.18;
  return Math.max(0.75, baseSeconds + commaPauses + strongPauses + dramaticPauses);
}

function splitStoryboardScriptCueForDuration(cue, maxSpeechSeconds) {
  const text = String(cue?.text || "").trim();
  if (!text || estimateStoryboardScriptCueSeconds(text) <= maxSpeechSeconds) {
    return [{ ...cue, source_cue_index: Number(cue?.index || 0), part_index: 1, part_count: 1, was_split: false }];
  }
  const tokens = text.match(/\S+(?:\s+|$)/g) || [text];
  const tokenCount = tokens.length;
  const textCache = new Map();
  const durationCache = new Map();
  const chunkText = (start, end) => {
    const key = `${start}:${end}`;
    if (!textCache.has(key)) textCache.set(key, tokens.slice(start, end).join("").trim());
    return textCache.get(key);
  };
  const chunkDuration = (start, end) => {
    const key = `${start}:${end}`;
    if (!durationCache.has(key)) durationCache.set(key, estimateStoryboardScriptCueSeconds(chunkText(start, end)));
    return durationCache.get(key);
  };
  const boundaryPenalty = (end) => {
    if (end >= tokenCount) return 0;
    const previousToken = tokens[end - 1].trim();
    if (/[.!?]["')\]]?$/.test(previousToken)) return 0;
    if (/[;—…]["')\]]?$/.test(previousToken)) return 3;
    if (/[:,]["')\]]?$/.test(previousToken)) return 12;
    // Never favor a visually balanced split that leaves a dangling grammar word.
    // MiniMax is much more likely to mumble or restart when a clip ends on words
    // such as "a", "the", "and", or "to" instead of a complete spoken phrase.
    const normalizedPrevious = previousToken.toLocaleLowerCase().replace(/[^a-z']/g, "");
    if (/^(?:a|an|the|and|or|but|to|of|for|with|from|in|on|at|by|as|than|that|this|these|those|my|your|his|her|its|our|their)$/.test(normalizedPrevious)) return 900;
    const nextToken = tokens[end]?.trim().toLocaleLowerCase().replace(/[^a-z']/g, "") || "";
    if (/^(?:and|or|but|while|then)$/.test(nextToken)) return 40;
    return 400;
  };
  const bestFrom = Array(tokenCount + 1).fill(null);
  bestFrom[tokenCount] = { cost: 0, parts: [] };
  for (let start = tokenCount - 1; start >= 0; start -= 1) {
    let best = null;
    for (let end = start + 1; end <= tokenCount; end += 1) {
      const duration = chunkDuration(start, end);
      if (duration > maxSpeechSeconds + 1e-6 && end > start + 1) break;
      const remainder = bestFrom[end];
      if (!remainder) continue;
      const wordCount = storyboardScriptWordCount(chunkText(start, end));
      const isFinalPart = end === tokenCount;
      let shortPartPenalty = 0;
      if (wordCount <= 1) shortPartPenalty += isFinalPart ? 300 : 100;
      else if (wordCount === 2) shortPartPenalty += isFinalPart ? 40 : 20;
      else if (duration < 1.25) shortPartPenalty += isFinalPart ? 25 : 12;
      const fullness = Math.max(0, maxSpeechSeconds - duration) / Math.max(0.75, maxSpeechSeconds);
      const balancePenalty = fullness * fullness * 3;
      // The large per-part cost guarantees the fewest possible clips first.
      // Within that clip count, punctuation quality, orphan avoidance, and balance decide the split.
      const cost = 1000 + boundaryPenalty(end) + shortPartPenalty + balancePenalty + remainder.cost;
      if (!best || cost < best.cost) {
        best = {
          cost,
          parts: [chunkText(start, end), ...remainder.parts],
        };
      }
    }
    bestFrom[start] = best;
  }
  const parts = bestFrom[0]?.parts?.filter(Boolean) || [text];
  return parts.map((part, index) => ({
    ...cue,
    text: part,
    word_count: storyboardScriptWordCount(part),
    source_cue_index: Number(cue?.index || 0),
    part_index: index + 1,
    part_count: parts.length,
    was_split: parts.length > 1,
  }));
}

export function planStoryboardScriptScenes(cues = [], options = {}) {
  const maxSceneSeconds = Math.max(3, Math.min(15, Number(options.max_scene_seconds || 8)));
  const openingBuffer = 0.15;
  const closingBuffer = 0.25;
  const sameSpeakerGap = 0.12;
  const speakerChangeGap = 0.2;
  const maxSpeechSeconds = Math.max(0.75, maxSceneSeconds - openingBuffer - closingBuffer);
  const sourceCues = Array.isArray(cues) ? cues : [];
  const groupKeyForCue = (cue) => Number(cue?.scene_index || 0) > 0
    ? `scene:${Number(cue.scene_index)}`
    : String(cue?.scene_label || "").trim() ? `label:${String(cue.scene_label).trim().toLocaleLowerCase()}` : "script:1";
  const participantsByGroup = new Map();
  for (const cue of sourceCues) {
    const key = groupKeyForCue(cue);
    const participants = participantsByGroup.get(key) || new Map();
    const participantKey = String(cue?.speaker_id || storyboardScriptSpeakerMatchKey(cue?.speaker_name || cue?.speaker));
    if (participantKey) participants.set(participantKey, {
      id: String(cue?.speaker_id || ""),
      name: String(cue?.speaker_name || cue?.speaker || ""),
      alias: String(cue?.speaker_alias || cue?.speaker || ""),
    });
    participantsByGroup.set(key, participants);
  }
  const expandedCues = sourceCues.flatMap((cue) => splitStoryboardScriptCueForDuration(cue, maxSpeechSeconds));
  const scenes = [];
  const warnings = [];
  let pending = [];
  let pendingGroupKey = "";
  const estimatedPackedSeconds = (rows) => {
    if (!rows.length) return 0;
    let total = openingBuffer + closingBuffer;
    rows.forEach((cue, index) => {
      if (index) total += storyboardScriptSpeakerMatchKey(rows[index - 1]?.speaker) === storyboardScriptSpeakerMatchKey(cue?.speaker) ? sameSpeakerGap : speakerChangeGap;
      total += estimateStoryboardScriptCueSeconds(cue?.text);
    });
    return total;
  };
  const flushScene = () => {
    if (!pending.length) return;
    let cursor = openingBuffer;
    const timedCues = pending.map((cue, index) => {
      if (index) cursor += storyboardScriptSpeakerMatchKey(pending[index - 1]?.speaker) === storyboardScriptSpeakerMatchKey(cue?.speaker) ? sameSpeakerGap : speakerChangeGap;
      const startSeconds = cursor;
      const spokenSeconds = estimateStoryboardScriptCueSeconds(cue?.text);
      cursor += spokenSeconds;
      return {
        ...cue,
        planned_start_seconds: Number(startSeconds.toFixed(2)),
        planned_end_seconds: Number(cursor.toFixed(2)),
        estimated_spoken_seconds: Number(spokenSeconds.toFixed(2)),
      };
    });
    const rawDuration = cursor + closingBuffer;
    const duration = Math.min(maxSceneSeconds, Math.ceil((rawDuration - 1e-6) * 10) / 10);
    const previousScene = scenes[scenes.length - 1];
    const timelineStartSeconds = scenes.reduce((total, scene) => total + Number(scene.duration_seconds || 0), 0);
    const sourceCueIndexes = Array.from(new Set(timedCues.map((cue) => Number(cue.source_cue_index || cue.index || 0)).filter(Boolean)));
    const participants = Array.from(participantsByGroup.get(pendingGroupKey)?.values() || []);
    const sourceSceneLabel = String(timedCues[0]?.scene_label || "").trim();
    scenes.push({
      index: scenes.length + 1,
      label: sourceSceneLabel || `Script Segment ${scenes.length + 1}`,
      source_scene_index: Number(timedCues[0]?.scene_index || 0),
      source_scene_label: sourceSceneLabel,
      continuation_of_previous: Boolean(previousScene && previousScene.source_group_key === pendingGroupKey),
      source_group_key: pendingGroupKey,
      maximum_scene_seconds: maxSceneSeconds,
      duration_seconds: Number(duration.toFixed(2)),
      timeline_start_seconds: Number(timelineStartSeconds.toFixed(2)),
      timeline_end_seconds: Number((timelineStartSeconds + duration).toFixed(2)),
      estimated_dialogue_seconds: Number(timedCues.reduce((total, cue) => total + Number(cue.estimated_spoken_seconds || 0), 0).toFixed(2)),
      source_cue_indexes: sourceCueIndexes,
      participant_ids: participants.map((participant) => participant.id).filter(Boolean),
      participant_names: participants.map((participant) => participant.name).filter(Boolean),
      participants,
      speaker_assignments: timedCues.map((cue) => ({
        speaker_id: String(cue.speaker_id || ""),
        speaker_name: String(cue.speaker_name || cue.speaker || ""),
        speaker_alias: String(cue.speaker_alias || cue.speaker || ""),
        text: String(cue.text || ""),
        source_cue_index: Number(cue.source_cue_index || cue.index || 0),
        part_index: Number(cue.part_index || 1),
        part_count: Number(cue.part_count || 1),
        planned_start_seconds: Number(cue.planned_start_seconds || 0),
        planned_end_seconds: Number(cue.planned_end_seconds || 0),
        estimated_spoken_seconds: Number(cue.estimated_spoken_seconds || 0),
      })),
    });
    pending = [];
  };
  for (const cue of expandedCues) {
    const groupKey = groupKeyForCue(cue);
    if (pending.length && groupKey !== pendingGroupKey) flushScene();
    pendingGroupKey = groupKey;
    const pendingSourceCueIndex = Number(pending[0]?.source_cue_index || pending[0]?.index || 0);
    const incomingSourceCueIndex = Number(cue?.source_cue_index || cue?.index || 0);
    const crossesSplitCueBoundary = pending.length
      && pendingSourceCueIndex !== incomingSourceCueIndex
      && (pending.some((item) => item.was_split) || (cue.was_split && Number(cue.part_index || 1) > 1));
    if (crossesSplitCueBoundary) flushScene();
    pendingGroupKey = groupKey;
    if (pending.length && estimatedPackedSeconds([...pending, cue]) > maxSceneSeconds + 1e-6) flushScene();
    pendingGroupKey = groupKey;
    pending.push(cue);
  }
  flushScene();
  const splitSourceCueIndexes = Array.from(new Set(expandedCues.filter((cue) => cue.was_split).map((cue) => Number(cue.source_cue_index || 0)).filter(Boolean)));
  if (splitSourceCueIndexes.length) warnings.push(`${splitSourceCueIndexes.length} long dialogue cue${splitSourceCueIndexes.length === 1 ? " was" : "s were"} split at natural phrase boundaries when possible to stay within ${maxSceneSeconds} seconds.`);
  return {
    maximum_scene_seconds: maxSceneSeconds,
    scene_count: scenes.length,
    split_cue_count: splitSourceCueIndexes.length,
    estimated_total_seconds: Number(scenes.reduce((total, scene) => total + Number(scene.duration_seconds || 0), 0).toFixed(2)),
    scenes,
    warnings,
  };
}

export function normalizeStoryboardScriptImportState(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const maximumSceneSeconds = Math.max(3, Math.min(15, Number(source.maximum_scene_seconds || source.max_scene_seconds || 8) || 8));
  const cues = (Array.isArray(source.cues) ? source.cues : []).slice(0, 1000).map((cue, index) => ({
    index: Number(cue?.index || index + 1),
    line_number: Number(cue?.line_number || 0),
    scene_index: Number(cue?.scene_index || 0),
    scene_label: String(cue?.scene_label || "").trim(),
    speaker: String(cue?.speaker_alias || cue?.speaker || cue?.speaker_name || "").trim(),
    speaker_alias: String(cue?.speaker_alias || cue?.speaker || cue?.speaker_name || "").trim(),
    speaker_id: String(cue?.speaker_id || cue?.reference_subject_id || ""),
    speaker_name: String(cue?.speaker_name || cue?.reference_subject_name || cue?.speaker || "").trim(),
    reference_subject_id: String(cue?.reference_subject_id || cue?.speaker_id || ""),
    reference_subject_name: String(cue?.reference_subject_name || cue?.speaker_name || "").trim(),
    speaker_match_method: String(cue?.speaker_match_method || "manual"),
    text: String(cue?.text || cue?.dialogue || cue?.line || "").trim(),
    word_count: storyboardScriptWordCount(cue?.text || cue?.dialogue || cue?.line || ""),
  })).filter((cue) => cue.speaker && cue.text);
  const speakersByKey = new Map();
  for (const cue of cues) {
    const key = storyboardScriptSpeakerMatchKey(cue.speaker);
    const existing = speakersByKey.get(key) || {
      name: cue.speaker,
      speaker_alias: cue.speaker,
      cue_count: 0,
      word_count: 0,
      reference_subject_id: cue.reference_subject_id,
      reference_subject_name: cue.reference_subject_name,
      match_method: cue.reference_subject_id ? cue.speaker_match_method || "manual" : "unmatched",
    };
    existing.cue_count += 1;
    existing.word_count += cue.word_count;
    if (!existing.reference_subject_id && cue.reference_subject_id) {
      existing.reference_subject_id = cue.reference_subject_id;
      existing.reference_subject_name = cue.reference_subject_name;
      existing.match_method = cue.speaker_match_method || "manual";
    }
    speakersByKey.set(key, existing);
  }
  const speakers = Array.from(speakersByKey.values());
  const speakerMatches = speakers.map((speaker) => ({
    speaker_alias: speaker.speaker_alias,
    reference_subject_id: speaker.reference_subject_id,
    reference_subject_name: speaker.reference_subject_name,
    match_method: speaker.match_method,
  }));
  const scenePlan = planStoryboardScriptScenes(cues, { max_scene_seconds: maximumSceneSeconds });
  return {
    enabled: source.enabled !== false && cues.length > 0,
    authoritative: source.authoritative !== false,
    format: String(source.format || "text"),
    raw_text: String(source.raw_text || source.rawText || ""),
    imported_at: String(source.imported_at || source.importedAt || ""),
    maximum_scene_seconds: maximumSceneSeconds,
    word_count: cues.reduce((total, cue) => total + cue.word_count, 0),
    cues,
    speakers,
    speaker_matches: speakerMatches,
    unmatched_speakers: speakers.filter((speaker) => !speaker.reference_subject_id).map((speaker) => speaker.name),
    scene_plan: scenePlan,
  };
}

export function parseStoryboardScriptImport(value) {
  const source = String(value || "").replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  const result = {
    format: "text",
    cues: [],
    speakers: [],
    metadata: [],
    errors: [],
    word_count: 0,
    estimated_spoken_seconds: 0,
  };
  const speakerMap = new Map();
  const reservedLabels = new Set(["scene", "scene label", "location", "setting", "present", "characters", "action", "camera", "audio", "audio direction", "continuity"]);
  const addCue = (speakerValue, textValue, details = {}) => {
    const speaker = String(speakerValue || "").trim();
    const text = String(textValue || "").trim();
    if (!speaker || !text) {
      result.errors.push({
        line_number: Number(details.line_number || 0),
        source: String(details.source || "").trim(),
        message: !speaker ? "Speaker name is missing." : "Dialogue text is missing.",
      });
      return;
    }
    const wordCount = storyboardScriptWordCount(text);
    const cue = {
      index: result.cues.length + 1,
      line_number: Number(details.line_number || 0),
      scene_index: Number(details.scene_index || 0),
      scene_label: String(details.scene_label || "").trim(),
      speaker,
      text,
      word_count: wordCount,
    };
    result.cues.push(cue);
    const key = speaker.toLocaleLowerCase();
    const summary = speakerMap.get(key) || { name: speaker, cue_count: 0, word_count: 0 };
    summary.cue_count += 1;
    summary.word_count += wordCount;
    speakerMap.set(key, summary);
  };
  const parseTextLines = (textValue, details = {}) => {
    let activeSceneLabel = String(details.scene_label || "").trim();
    String(textValue || "").split("\n").forEach((rawLine, index) => {
      const line = String(rawLine || "").trim();
      if (!line) return;
      const lineNumber = Number(details.line_offset || 0) + index + 1;
      const match = line.match(/^([^:\n]{1,80}?)\s*:\s*(.*)$/);
      if (!match) {
        result.errors.push({ line_number: lineNumber, source: line, message: "Expected speaker: dialogue." });
        return;
      }
      const label = String(match[1] || "").trim();
      const text = String(match[2] || "").trim();
      const labelKey = label.toLocaleLowerCase();
      if (reservedLabels.has(labelKey)) {
        result.metadata.push({ label, value: text, line_number: lineNumber });
        if (labelKey === "scene" || labelKey === "scene label") activeSceneLabel = text;
        return;
      }
      addCue(label, text, {
        line_number: lineNumber,
        source: line,
        scene_index: details.scene_index,
        scene_label: activeSceneLabel,
      });
    });
  };
  const addJsonCueRows = (rows, details = {}) => {
    if (typeof rows === "string") {
      parseTextLines(rows, details);
      return;
    }
    if (!Array.isArray(rows)) {
      result.errors.push({ line_number: 0, source: details.scene_label || "JSON", message: "Dialogue cues must be an array or speaker: dialogue text." });
      return;
    }
    rows.forEach((item, index) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        result.errors.push({ line_number: 0, source: `JSON cue ${index + 1}`, message: "Cue must be an object with speaker and text fields." });
        return;
      }
      addCue(
        item.speaker_name || item.speaker || item.character || item.name,
        item.text || item.dialogue || item.line,
        {
          source: `JSON cue ${index + 1}`,
          scene_index: details.scene_index,
          scene_label: details.scene_label,
        },
      );
    });
  };

  const trimmed = source.trim();
  if (!trimmed) {
    result.errors.push({ line_number: 0, source: "", message: "Paste a script or load a .txt/.json file first." });
    return result;
  }
  if (/^[\[{]/.test(trimmed)) {
    result.format = "json";
    try {
      const parsed = JSON.parse(trimmed);
      const scenes = !Array.isArray(parsed) && Array.isArray(parsed?.scenes) ? parsed.scenes : null;
      if (scenes) {
        scenes.forEach((scene, sceneIndex) => {
          if (!scene || typeof scene !== "object" || Array.isArray(scene)) {
            result.errors.push({ line_number: 0, source: `JSON scene ${sceneIndex + 1}`, message: "Scene must be an object." });
            return;
          }
          const sceneLabel = String(scene.label || scene.scene_label || scene.title || `Scene ${sceneIndex + 1}`).trim();
          const rows = scene.speaker_assignments || scene.dialogue_cues || scene.cues || scene.dialogue || [];
          addJsonCueRows(rows, { scene_index: sceneIndex + 1, scene_label: sceneLabel });
        });
      } else {
        const rows = Array.isArray(parsed)
          ? parsed
          : parsed?.speaker_assignments || parsed?.dialogue_cues || parsed?.cues || parsed?.dialogue;
        addJsonCueRows(rows, {});
      }
    } catch (error) {
      result.errors.push({ line_number: 0, source: "JSON", message: `Invalid JSON: ${String(error?.message || error)}` });
    }
  } else {
    parseTextLines(source);
  }
  result.speakers = Array.from(speakerMap.values());
  result.word_count = result.cues.reduce((total, cue) => total + Number(cue.word_count || 0), 0);
  result.estimated_spoken_seconds = result.word_count ? (result.word_count / 145) * 60 : 0;
  return result;
}
