import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_StoryboardBuilderUI";
const HIDDEN_WIDGETS = new Set(["project_folder"]);
const STORYBOARD_GEMMA_TIMEOUT_MS = 600000;

function hideInternalWidgets(node) {
  for (const widget of node.widgets || []) {
    if (!HIDDEN_WIDGETS.has(widget.name)) continue;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
  }
}

async function postJson(url, payload = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await api.fetchApi(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.ok === false) {
      const requestError = new Error(data?.error || `Request failed (${response.status})`);
      if (data?.diagnostics && typeof data.diagnostics === "object") requestError.diagnostics = data.diagnostics;
      throw requestError;
    }
    return data;
  } catch (error) {
    if (timedOut || controller.signal.aborted || error?.name === "AbortError") {
      const timeoutSeconds = Math.max(1, Math.round(timeoutMs / 1000));
      const timeoutAmount = timeoutSeconds >= 60 ? Math.round(timeoutSeconds / 60) : timeoutSeconds;
      const timeoutUnit = timeoutSeconds >= 60 ? "minute" : "second";
      throw new Error(`Request timed out after ${timeoutAmount} ${timeoutUnit}${timeoutAmount === 1 ? "" : "s"}. The backend may still be processing it.`);
    }
    const message = String(error?.message || error || "");
    if (/NetworkError|Failed to fetch|fetch resource|Load failed/i.test(message)) {
      throw new Error("Connection to the ComfyUI backend was lost. Check that ComfyUI is still running and inspect its console. If this happened while loading a local LLM, lower its GPU layers or context limit and try again.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function makeButton(label, variant = "default") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  const bg = variant === "primary" ? "#12b5cb" : variant === "purple" ? "#0e7490" : "#2b2b30";
  const border = variant === "primary" ? "#0891b2" : variant === "purple" ? "#06b6d4" : "#3f3f46";
  button.style.cssText = `border:1px solid ${border};border-radius:6px;background:${bg};color:#f8fafc;padding:9px 13px;font-weight:800;cursor:pointer;`;
  return button;
}

function makeInput(value = "", placeholder = "") {
  const input = document.createElement("input");
  input.value = value || "";
  input.placeholder = placeholder;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#0b1220;color:#e5e7eb;padding:9px;font:12px monospace;";
  return input;
}

function makeTextarea(value = "", placeholder = "", rows = 4) {
  const textarea = document.createElement("textarea");
  textarea.value = value || "";
  textarea.placeholder = placeholder;
  textarea.rows = rows;
  textarea.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;border:1px solid #334155;border-radius:6px;background:#050814;color:#e5e7eb;padding:9px;font:12px monospace;line-height:1.45;";
  return textarea;
}

function makeSelect(options, value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;";
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    select.append(item);
  }
  select.value = value || options[0]?.value || "";
  return select;
}

function makeGroupedSelect(groups, value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;";
  for (const group of groups) {
    if (group.options) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group.label;
      for (const option of group.options) {
        const item = document.createElement("option");
        item.value = option.value ?? option;
        item.textContent = option.label ?? option;
        optgroup.append(item);
      }
      select.append(optgroup);
    } else {
      const item = document.createElement("option");
      item.value = group.value ?? "";
      item.textContent = group.label ?? "";
      select.append(item);
    }
  }
  select.value = value || "";
  return select;
}

function makeMultiSelect(options, values = []) {
  const select = document.createElement("select");
  select.multiple = true;
  select.size = Math.min(6, Math.max(3, options.length || 3));
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:7px;min-height:104px;";
  const selected = new Set(Array.isArray(values) ? values.map(String) : []);
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    item.selected = selected.has(String(option.value));
    select.append(item);
  }
  return select;
}

function makeCollapsiblePanel(title, summary = "", content = null, { open = false } = {}) {
  const panel = document.createElement("div");
  panel.style.cssText = "margin:8px 24px 0;border:1px solid #334155;border-radius:8px;background:#0f172a;overflow:hidden;min-width:0;max-width:100%;box-sizing:border-box;";
  const header = document.createElement("button");
  header.type = "button";
  header.style.cssText = "width:100%;min-width:0;box-sizing:border-box;border:0;background:#0f172a;color:#e5e7eb;padding:9px 12px;display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center;text-align:left;cursor:pointer;";
  const caret = document.createElement("span");
  caret.style.cssText = "color:#67e8f9;font-size:13px;";
  const label = document.createElement("span");
  label.style.cssText = "font-weight:900;color:#cffafe;font-size:13px;white-space:nowrap;";
  label.textContent = title;
  const summaryNode = document.createElement("span");
  summaryNode.style.cssText = "min-width:0;max-width:100%;color:#94a3b8;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  summaryNode.textContent = summary;
  const body = document.createElement("div");
  body.style.cssText = "min-width:0;max-width:100%;box-sizing:border-box;border-top:1px solid #1f3347;padding:10px 12px;";
  if (content) body.append(content);
  let expanded = Boolean(open);
  const sync = () => {
    caret.textContent = expanded ? "▾" : "▸";
    body.style.display = expanded ? "" : "none";
  };
  header.onclick = () => {
    expanded = !expanded;
    sync();
  };
  header.append(caret, label, summaryNode);
  panel.append(header, body);
  panel.setSummary = (value) => {
    summaryNode.textContent = String(value || "");
  };
  panel.setOpen = (value) => {
    expanded = Boolean(value);
    sync();
  };
  panel.isOpen = () => expanded;
  sync();
  return panel;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function storyboardScriptWordCount(value) {
  return (String(value || "").match(/[\p{L}\p{N}'’-]+/gu) || []).length;
}

function storyboardScriptSpeakerMatchKey(value) {
  return String(value || "")
    .toLocaleLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/^(?:the|a|an)\s+/, "")
    .replace(/\s+/g, " ");
}

function suggestStoryboardScriptSpeakerMatch(speakerName, characters = []) {
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

function planStoryboardScriptScenes(cues = [], options = {}) {
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

function normalizeStoryboardScriptImportState(value = {}) {
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

function parseStoryboardScriptImport(value) {
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

function makeStoryboardImageUrl(path) {
  return `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}&rand=${Date.now()}`;
}

function storyboardReferenceImageSrc(image) {
  if (!image || typeof image !== "object") return "";
  const data = String(image.data || "").trim();
  if (data) return data.startsWith("data:") ? data : `data:image/png;base64,${data}`;
  const path = String(image.path || "").trim();
  return path ? makeStoryboardImageUrl(path) : "";
}

function normalizeReferenceImage(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const image = source.image && typeof source.image === "object" ? source.image : source;
  const hasTopLevelImage = Boolean(source.path || source.data || source.image_path || source.imagePath || source.image_data || source.imageData);
  return {
    path: String(image.path || source.image_path || source.imagePath || source.path || "").trim(),
    data: String(image.data || source.image_data || source.imageData || source.data || "").trim(),
    name: String(image.name || source.image_name || source.imageName || (hasTopLevelImage ? source.name : "") || "").trim(),
  };
}

function mergeReferenceImages(existing = {}, incoming = {}) {
  const left = normalizeReferenceImage(existing);
  const right = normalizeReferenceImage(incoming);
  return {
    path: right.path || left.path,
    data: right.data || left.data,
    name: right.name || left.name,
  };
}

function truncate(text, length = 130) {
  const clean = String(text || "").trim();
  if (clean.length <= length) return clean;
  return `${clean.slice(0, Math.max(0, length - 1)).trim()}...`;
}

function replaceLabeledPlanningLine(value, labelName, selectedValue) {
  const cleanLabel = String(labelName || "").trim();
  const cleanValue = String(selectedValue || "").trim();
  if (!cleanLabel || !cleanValue) return String(value || "").trim();
  const prefix = `${cleanLabel}:`;
  const replacement = `${prefix} ${cleanValue}.`;
  const lines = String(value || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => !line.trim().toLowerCase().startsWith(prefix.toLowerCase()));
  lines.push(replacement);
  return lines.map((line) => line.trim()).filter(Boolean).join("\n");
}

function tagsHtml(tags) {
  const list = Array.isArray(tags) ? tags : [];
  if (!list.length) return `<span style="color:#94a3b8;">-</span>`;
  return list.map((tag) => `<span style="display:inline-flex;border-radius:5px;background:#1e1b4b;color:#ddd6fe;padding:4px 7px;margin:2px;font-size:11px;">${escapeHtml(tag)}</span>`).join("");
}

function storyboardSubjectNamesFromRefs(subjectRefs = []) {
  return Array.from(new Set(
    (Array.isArray(subjectRefs) ? subjectRefs : [])
      .map((subject) => String(subject?.name || "").trim())
      .filter(Boolean)
  ));
}

function storyboardReferenceId(prefix, name = "") {
  const slug = String(name || prefix || "reference")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || prefix;
  return `${prefix}_story_${Date.now()}_${slug}`;
}

function readStoryboardImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file || !String(file.type || "").startsWith("image/")) {
      reject(new Error("Choose an image file."));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read that image file."));
    reader.readAsDataURL(file);
  });
}

const IMAGE_SHOT_TYPES = [
  "close-up shot",
  "extreme close-up shot",
  "medium close-up shot",
  "medium shot",
  "medium wide shot",
  "wide shot",
  "extreme wide shot",
  "full shot",
  "long shot",
  "extreme long shot",
  "establishing shot",
  "master shot",
  "two-shot",
  "three-shot",
  "over-the-shoulder shot",
  "point-of-view shot",
  "first-person shot",
  "insert shot",
  "cutaway shot",
  "reaction shot",
  "detail shot",
  "beauty shot",
  "hero shot",
  "profile shot",
  "frontal shot",
  "rear shot",
  "side shot",
  "low-angle shot",
  "high-angle shot",
  "eye-level shot",
  "bird's-eye view shot",
  "worm's-eye view shot",
  "aerial shot",
  "drone shot",
  "overhead shot",
  "top-down shot",
  "ground-level shot",
  "Dutch angle shot",
  "tilted shot",
  "symmetrical shot",
  "centered shot",
  "off-center shot",
  "silhouette shot",
  "reflection shot",
  "mirror shot",
  "shadow shot",
  "through-the-window shot",
  "through-the-doorway shot",
  "frame-within-a-frame shot",
  "single shot",
  "two-person shot",
  "group shot",
  "crowd shot",
  "face shot",
  "head shot",
  "head-and-shoulders shot",
  "bust shot",
  "waist-up shot",
  "chest-up shot",
  "knee-up shot",
  "cowboy shot",
  "American shot",
  "full-body shot",
  "feet shot",
  "hands shot",
  "eyes shot",
  "mouth shot",
  "object shot",
  "product shot",
  "environment shot",
  "landscape shot",
  "cityscape shot",
  "room shot",
  "hallway shot",
  "doorway shot",
  "car interior shot",
  "dashboard shot",
  "passenger-seat shot",
  "driver-seat shot",
  "cinematic wide shot",
  "moody close-up shot",
  "dramatic low-angle shot",
  "intimate close-up shot",
  "documentary-style shot",
  "surveillance-style shot",
  "security-camera shot",
  "CCTV shot",
  "found-footage shot",
  "vlog-style shot",
  "selfie shot",
  "webcam shot",
  "interview shot",
  "talking-head shot",
  "news-style shot",
  "broadcast-style shot",
  "commercial product shot",
  "lifestyle shot",
  "montage opening shot",
  "transition shot",
  "dreamlike shot",
  "blurred foreground shot",
  "shallow-depth-of-field shot",
  "deep-focus shot",
  "soft-focus shot",
  "backlit shot",
  "lens-flare shot",
  "natural-light shot",
  "night shot",
  "golden-hour shot",
  "blue-hour shot",
];

const VIDEO_SHOT_TYPES = [
  ...IMAGE_SHOT_TYPES,
  "static shot",
  "locked-off shot",
  "handheld shot",
  "tracking shot",
  "dolly shot",
  "dolly-in shot",
  "dolly-out shot",
  "push-in shot",
  "pull-out shot",
  "zoom-in shot",
  "zoom-out shot",
  "pan shot",
  "whip pan shot",
  "tilt-up shot",
  "tilt-down shot",
  "crane shot",
  "jib shot",
  "Steadicam shot",
  "gimbal shot",
  "follow shot",
  "lead shot",
  "arc shot",
  "orbit shot",
  "360-degree shot",
  "reveal shot",
  "rack-focus shot",
  "focus-pull shot",
  "slow-motion shot",
  "time-lapse shot",
  "hyperlapse shot",
];

const CAMERA_MOTION_GROUPS = [
  { value: "", label: "Choose camera motion..." },
  {
    label: "Basic Camera Motions",
    options: [
      "pan left", "pan right", "pan up", "pan down", "tilt up", "tilt down",
      "push in", "pull back", "pull out", "dolly in", "dolly out",
      "dolly left", "dolly right", "truck left", "truck right",
      "pedestal up", "pedestal down", "zoom in", "zoom out",
      "slow zoom in", "slow zoom out", "quick zoom in", "snap zoom",
      "crash zoom", "whip pan", "whip left", "whip right", "whip up", "whip down",
    ],
  },
  {
    label: "Orbit / Rotation Motions",
    options: [
      "orbit left", "orbit right", "orbit around subject", "rotate around subject",
      "circle around subject", "180-degree rotation", "360-degree rotation",
      "half-circle orbit", "full-circle orbit", "clockwise orbit",
      "counterclockwise orbit", "spiral around subject", "arc left", "arc right",
      "arc around subject", "wraparound move", "sweeping circular move",
    ],
  },
  {
    label: "Tracking / Following Motions",
    options: [
      "track forward", "track backward", "track left", "track right",
      "tracking shot", "follow shot", "follow behind", "follow in front",
      "lead shot", "side-follow shot", "over-the-shoulder follow",
      "chase shot", "pursuit shot", "walk-and-talk tracking",
      "handheld follow", "gimbal follow", "steadicam follow",
      "smooth follow", "shaky follow",
    ],
  },
  {
    label: "Reveal Motions",
    options: [
      "reveal upward", "reveal downward", "reveal left", "reveal right",
      "slide reveal", "dolly reveal", "pan reveal", "tilt reveal",
      "pull-back reveal", "push-in reveal", "orbit reveal", "crane reveal",
      "rack-focus reveal", "foreground reveal", "doorway reveal",
      "window reveal", "object reveal", "character reveal", "environment reveal",
    ],
  },
  {
    label: "Vertical / Height Motions",
    options: [
      "crane up", "crane down", "jib up", "jib down", "rise up",
      "descend down", "boom up", "boom down", "lift upward", "drop downward",
      "float upward", "sink downward", "aerial rise", "aerial descent",
      "drone rise", "drone descend", "top-down descent",
      "ground-to-sky tilt", "sky-to-ground tilt",
    ],
  },
  {
    label: "Drone / Aerial Motions",
    options: [
      "drone flyover", "drone push in", "drone pull back", "drone rise",
      "drone descend", "drone orbit", "drone circle", "drone follow",
      "drone chase", "drone pass-through", "drone reveal", "aerial tracking",
      "aerial pan", "aerial tilt", "overhead drift", "top-down tracking",
      "bird's-eye pullback", "sweeping aerial move",
    ],
  },
  {
    label: "Handheld / Style Motions",
    options: [
      "handheld shake", "subtle handheld movement", "shaky cam",
      "smooth handheld", "floating camera move", "drifting camera move",
      "breathing camera movement", "documentary-style movement",
      "natural handheld sway", "nervous handheld move", "chaotic handheld move",
      "stabilized gimbal move", "steadicam glide", "slow cinematic glide",
      "smooth cinematic drift",
    ],
  },
  {
    label: "Focus / Lens Motions",
    options: [
      "rack focus", "focus pull", "focus shift",
      "foreground-to-background focus", "background-to-foreground focus",
      "shallow-focus drift", "zoom with focus pull", "dolly zoom",
      "vertigo effect", "crash zoom with focus", "soft-focus transition",
      "focus reveal",
    ],
  },
  {
    label: "POV / Subjective Motions",
    options: [
      "POV walk forward", "POV turn left", "POV turn right", "POV look up",
      "POV look down", "POV stumble", "POV run", "POV chase", "POV fall",
      "POV rise", "POV scan the room", "POV peek around corner",
      "POV lean in", "POV look over shoulder",
    ],
  },
  {
    label: "Transition Motions",
    options: [
      "whip pan transition", "match move", "push-through transition",
      "pass-through transition", "foreground wipe", "camera wipe",
      "object wipe", "spin transition", "rotate transition", "zoom transition",
      "crash zoom transition", "tilt transition", "pan transition",
      "motion blur transition",
    ],
  },
];

const STILL_CAMERA_STYLE_GROUPS = [
  { value: "", label: "Choose still camera style..." },
  {
    label: "Composition / Framing",
    options: [
      "clean portrait composition", "editorial fashion composition", "cinematic still frame",
      "rule-of-thirds composition", "centered symmetrical composition", "negative space composition",
      "foreground framing", "frame-within-a-frame composition", "environmental portrait",
      "intimate close portrait", "wide environmental still", "dramatic silhouette composition",
    ],
  },
  {
    label: "Lens / Depth",
    options: [
      "shallow depth of field", "deep focus photography", "soft background bokeh",
      "wide-angle perspective", "telephoto compression", "macro detail photography",
      "natural lens perspective", "cinematic anamorphic lens look", "soft-focus portrait lens",
      "crisp studio lens detail",
    ],
  },
  {
    label: "Lighting / Exposure",
    options: [
      "natural window light", "golden-hour photography", "blue-hour photography",
      "high-contrast studio lighting", "soft diffused key light", "dramatic rim lighting",
      "backlit portrait", "low-key lighting", "high-key photography",
      "moody practical lighting", "neon-lit still photography",
    ],
  },
  {
    label: "Still Photography Style",
    options: [
      "editorial magazine photo", "fine-art portrait photography", "documentary still photo",
      "album-cover photography", "cinematic production still", "glossy commercial photo",
      "gritty street photography", "dreamlike fashion editorial", "dramatic character portrait",
      "atmospheric location photography",
    ],
  },
];

const CHARACTER_MOTION_GROUPS = [
  { value: "", label: "Choose character motion..." },
  {
    label: "Basic Locomotion",
    options: [
      "standing still", "walking", "running", "jogging", "sprinting", "pacing",
      "strolling", "wandering", "marching", "limping", "sneaking", "crawling",
      "climbing", "jumping", "landing", "falling", "tripping", "stumbling",
      "sliding", "spinning", "turning around", "looking around", "backing away",
      "moving forward", "moving sideways", "approaching camera",
      "walking away from camera",
    ],
  },
  {
    label: "Dance / Performance",
    options: [
      "dancing", "freestyle dancing", "slow dancing", "breakdancing",
      "hip-hop dancing", "club dancing", "swaying to music", "head nodding",
      "shoulder bouncing", "foot tapping", "hand waving", "arm swinging",
      "body rolling", "spinning while dancing", "jumping to the beat",
      "performing on stage", "singing into microphone", "rapping into microphone",
      "playing guitar", "playing piano", "playing drums", "DJing", "crowd surfing",
    ],
  },
  {
    label: "Gestures",
    options: [
      "pointing", "waving", "clapping", "snapping fingers", "giving thumbs up",
      "crossing arms", "raising arms", "reaching out", "holding hands up",
      "covering face", "touching chest", "touching head", "brushing hair back",
      "adjusting jacket", "adjusting sunglasses", "putting hands in pockets",
      "throwing hands up", "making hand signs", "beckoning", "saluting",
    ],
  },
  {
    label: "Facial Expression / Head Movement",
    options: [
      "smiling", "laughing", "crying", "frowning", "smirking", "shouting",
      "whispering", "looking at camera", "looking away",
      "looking down", "looking up", "turning head", "tilting head", "nodding",
      "shaking head", "closing eyes", "opening eyes", "blinking",
      "staring intensely",
    ],
  },
  {
    label: "Environment Interaction",
    options: [
      "opening a door", "closing a door", "leaning on a wall", "sitting on a chair",
      "standing up", "sitting down", "lying down", "kneeling", "picking something up",
      "dropping something", "throwing something", "pushing something",
      "pulling something", "carrying something", "leaning over a railing",
      "looking out a window", "walking through smoke", "walking through rain",
      "splashing through water", "kicking dust", "touching a wall",
      "running fingers along a surface",
    ],
  },
  {
    label: "Object Interaction",
    options: [
      "holding microphone", "holding phone", "looking at phone", "taking a photo",
      "recording video", "holding flowers", "holding money", "counting money",
      "holding a drink", "drinking", "smoking", "lighting a cigarette",
      "wearing headphones", "putting on headphones", "removing sunglasses",
      "putting on sunglasses", "holding a weapon prop", "holding a bag",
      "carrying luggage", "tossing keys", "spinning keys", "reading a note",
    ],
  },
  {
    label: "Emotional Action",
    options: [
      "collapsing to knees", "reaching toward camera", "running away",
      "chasing someone", "being chased", "searching for someone", "hiding",
      "waiting", "hesitating", "reacting in shock", "celebrating", "arguing",
      "fighting", "hugging", "pushing away", "walking alone",
      "standing in silence", "looking heartbroken", "looking confident",
      "looking angry", "looking lost",
    ],
  },
  {
    label: "Camera-Facing Motion",
    options: [
      "walking toward camera", "walking past camera", "turning to face camera",
      "looking directly into lens", "reaching toward lens", "pointing at camera",
      "singing to camera", "dancing toward camera", "moving in slow motion",
      "freezing in place", "silhouette movement", "hair blowing in wind",
      "clothing flowing in wind", "walking through frame", "entering frame",
      "exiting frame", "crossing foreground", "moving in background",
    ],
  },
  {
    label: "Group Movement",
    options: [
      "crowd dancing", "crowd jumping", "crowd waving arms", "crowd clapping",
      "people walking around", "people running past", "group marching",
      "group circling character", "group following character",
      "group surrounding character", "backup dancers performing",
      "band performing", "audience cheering", "friends walking together",
      "couple dancing", "couple arguing", "couple embracing",
    ],
  },
  {
    label: "Vehicle / Travel",
    options: [
      "driving", "riding in car", "getting into car", "getting out of car",
      "leaning out car window", "walking beside car", "sitting on car hood",
      "riding motorcycle", "riding bicycle", "skateboarding", "roller skating",
      "riding elevator", "walking down stairs", "walking up stairs",
      "riding escalator", "running through tunnel", "walking across street",
    ],
  },
  {
    label: "Stylized / Surreal Motion",
    options: [
      "floating", "levitation", "falling in slow motion", "spinning in place",
      "walking in reverse", "glitching", "teleporting", "duplicating",
      "morphing pose", "freeze-frame pose", "dramatic turn", "slow-motion walk",
      "wind-swept pose", "hero pose", "shadow dancing", "silhouette dancing",
      "smoke reveal", "light reveal", "walking through sparks",
      "dancing in rain", "falling backward into darkness", "reaching through light",
      "moving like a puppet", "robotic movement", "fluid dreamlike movement",
    ],
  },
];

export function normalizeStoryboardCustomCameraFlowSequence(input) {
  let source = input;
  if (typeof source === "string") {
    const raw = source.trim();
    if (!raw) return [];
    try {
      source = JSON.parse(raw);
    } catch (_error) {
      source = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    }
  }
  if (source && !Array.isArray(source) && typeof source === "object") {
    source = source.shots || source.sequence || source.candidates || source.list || source.items || [];
  }
  if (!Array.isArray(source)) return [];
  return source.map((item) => {
    if (typeof item === "string") {
      const parts = item.trim().replace(/^\s*(?:[-*•]|\d+[.)])\s*/, "").split(/\s+(?:\||—|–|-|->|=>)\s+/);
      return { shot: String(parts[0] || "").trim(), camera: String(parts.slice(1).join(" | ") || "").trim() };
    }
    if (!item || typeof item !== "object") return null;
    return {
      shot: String(item.shot || item.framing || item.type || item.description || item.name || "").trim(),
      camera: String(item.camera || item.camera_motion || item.movement || item.motion || "").trim(),
    };
  }).filter((item) => item?.shot).map((item) => ({ shot: item.shot, camera: item.camera }));
}

export const STORYBOARD_CAMERA_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill missing shot or camera motion fields.",
    sequence: [],
  },
  balanced: {
    label: "Balanced cinematic flow",
    description: "Alternates wide, medium, close, lateral, reveal, and reset shots without repeating inward zooms.",
    guidance: "Use the selected starting shot as the literal first generated frame. Do not add a wider, farther-away, establishing, or full-body lead-in before it. Preserve the selected framing unless the selected camera move explicitly changes scale. Treat inward moves as rare accents, never as a default pattern.",
    sequence: [
      { shot: "wide shot", camera: "slow cinematic drift" },
      { shot: "medium close-up shot", camera: "pull back" },
      { shot: "tracking shot", camera: "side-follow shot" },
      { shot: "close-up shot", camera: "slow orbit left" },
      { shot: "medium wide shot", camera: "dolly right" },
      { shot: "profile shot", camera: "pan reveal" },
      { shot: "low-angle shot", camera: "crane up" },
      { shot: "intimate close-up shot", camera: "slow zoom out" },
      { shot: "over-the-shoulder shot", camera: "reveal right" },
      { shot: "full-body shot", camera: "track backward" },
    ],
  },
  intimate_closeups: {
    label: "Intimate close-ups",
    framing_candidates: true,
    description: "Uses only distinct frame-filling close-ups, body-detail reveals, tight seated poses, and tight upper-body compositions.",
    guidance: "Every shot remains close, intimate, and frame-filling. The furthest framing is a tightly composed upper-body shot. Never use a wide shot, distant shot, full-body shot, small-in-frame composition, or full environment view. Each shot must use a distinct framing, angle, subject detail, or camera movement.",
    sequence: [
      { shot: "extreme close-up of one eye, slowly pulling back to reveal the full face", camera: "slow pullback" },
      { shot: "extreme close-up of the mouth, slowly pulling back to the upper body", camera: "slow pullback" },
      { shot: "close-up of both eyes", camera: "slight sideways camera slide" },
      { shot: "tight face close-up from a three-quarter angle", camera: "slow three-quarter orbit" },
      { shot: "tight profile close-up of the face", camera: "gentle lateral drift" },
      { shot: "close-up of the hand resting on the hip, slowly panning upward to the face", camera: "slow upward pan" },
      { shot: "close-up of fingers brushing through the hair, tilting upward to the eyes", camera: "slow upward tilt" },
      { shot: "close-up of the shoulder and neck, panning upward to the face", camera: "slow upward pan" },
      { shot: "close-up of the lips, then tilting upward to the eyes", camera: "slow upward tilt" },
      { shot: "close-up of the eyes, slowly tilting downward to the hands", camera: "slow downward tilt" },
      { shot: "close-up of the feet walking, panning upward along the body to the face", camera: "slow upward pan" },
      { shot: "close-up of the feet standing still, slowly tilting upward to the upper body", camera: "slow upward tilt" },
      { shot: "close-up of one hand reaching toward the camera, revealing the face behind it", camera: "slow reveal" },
      { shot: "close-up of hands gripping clothing, panning upward to the face", camera: "slow upward pan" },
      { shot: "close-up of a hand touching the chest, tilting upward to the eyes", camera: "slow upward tilt" },
      { shot: "tight seated portrait with the knees, torso, and face filling the frame", camera: "slow lateral drift" },
      { shot: "seated side-profile shot with the body filling the entire frame", camera: "slow side pan" },
      { shot: "seated curled-up pose framed from knees to face", camera: "slow push-in" },
      { shot: "tight upper-body shot with the subject leaning toward the camera", camera: "slow push-in" },
      { shot: "tight upper-body shot from behind the shoulder, revealing the face in profile", camera: "slow shoulder reveal" },
      { shot: "low-angle close-up from the waist upward, keeping the face near the top of frame", camera: "slow upward tilt" },
      { shot: "high-angle close-up looking down at the subject’s face and upper body", camera: "slow downward drift" },
      { shot: "tight overhead shot of the subject lying down, filling the frame", camera: "slow overhead drift" },
      { shot: "close-up of the subject lying on their side, slowly panning from feet to face", camera: "slow lateral pan" },
      { shot: "close-up from behind as the subject turns their head toward the camera", camera: "slow turn reveal" },
      { shot: "tight front-facing upper-body shot with a slow push-in toward the eyes", camera: "slow push-in" },
      { shot: "tight side shot with a slow horizontal pan from shoulder to face", camera: "slow horizontal pan" },
      { shot: "close-up framed through the subject’s moving hair", camera: "gentle hair reveal" },
      { shot: "reflection close-up in a mirror, slowly moving from the reflection’s hands to face", camera: "slow reflection pan" },
      { shot: "tight silhouette close-up with the face and shoulders filling the frame", camera: "slow silhouette drift" },
    ],
  },
  music_video: {
    label: "Music video shots",
    framing_candidates: true,
    description: "Uses performance, movement, location, reveal, tracking, and rhythmic music-video shot ideas.",
    guidance: "Use only the selected music-video candidate framing for each shot. Choose the strongest fit for the lyrics, story, character motion, performance, and location. Preserve distinct shot variety across the scene and avoid unnecessary repetition across scenes; sensible reuse is allowed when the action clearly calls for it or the candidate pool is exhausted.",
    sequence: [
      { shot: "wide performance shot with the subject centered in the environment", camera: "slow performance drift" },
      { shot: "full-body shot walking toward the camera", camera: "track backward" },
      { shot: "side-tracking shot following the subject's movement", camera: "side track" },
      { shot: "low-angle full-body performance shot", camera: "low-angle tracking move" },
      { shot: "high-angle shot looking down at the subject", camera: "high-angle crane drift" },
      { shot: "slow push-in from wide to medium framing", camera: "slow push-in" },
      { shot: "pull-back revealing the full location", camera: "slow pull-back reveal" },
      { shot: "circular camera move around the subject", camera: "full circular orbit" },
      { shot: "profile shot while the subject walks", camera: "profile side track" },
      { shot: "rear tracking shot following the subject from behind", camera: "rear follow" },
      { shot: "overhead shot of the subject lying or moving on the ground", camera: "overhead tracking drift" },
      { shot: "Dutch-angle performance shot", camera: "tilted handheld drift" },
      { shot: "static shot with the subject moving through the frame", camera: "locked-off composition" },
      { shot: "camera crossing from behind the subject to the front", camera: "wraparound reveal" },
      { shot: "full-body dancing shot with rhythmic camera motion", camera: "rhythmic orbit" },
      { shot: "walking past the camera in profile", camera: "profile pass-by track" },
      { shot: "camera following the subject through a doorway", camera: "doorway follow-through" },
      { shot: "wide shot with the subject isolated in the environment", camera: "slow environmental drift" },
      { shot: "slow-motion full-body movement shot", camera: "smooth-motion follow" },
      { shot: "handheld roaming performance shot", camera: "roaming handheld move" },
      { shot: "silhouette shot against strong backlighting", camera: "slow silhouette reveal" },
      { shot: "mirror or reflection shot", camera: "reflection slide" },
      { shot: "long-lens shot compressing the background", camera: "compressed lateral drift" },
      { shot: "ground-level shot looking upward as the subject approaches", camera: "ground-level push-in" },
      { shot: "elevated crane-style reveal", camera: "crane rise" },
      { shot: "continuous one-take shot following the subject", camera: "continuous tracking move" },
      { shot: "whip-pan transition between movements or locations", camera: "whip pan" },
      { shot: "match cut connecting two poses or actions", camera: "match-cut reframing" },
    ],
  },
  fisheye_distorted: {
    label: "Fisheye and Distorted-Lens Shots",
    framing_candidates: true,
    description: "Uses only fisheye, warped-perspective, curved-reflection, and distorted-lens compositions.",
    guidance: "Use only the selected fisheye or distorted-lens candidate framing for each shot. Keep the warped perspective visibly intentional and choose the strongest fit for the lyrics, story, character motion, performance, and location. Preserve distinct shot variety across the scene and avoid unnecessary repetition across scenes; sensible reuse is allowed when the action clearly calls for it or the candidate pool is exhausted.",
    sequence: [
      { shot: "extreme fisheye shot with the subject leaning toward the lens", camera: "dramatic fisheye push-in" },
      { shot: "full-body fisheye shot with exaggerated perspective", camera: "wide fisheye drift" },
      { shot: "low-angle fisheye shot as the subject looks down into the camera", camera: "low fisheye tilt-up" },
      { shot: "crouching close to the camera and staring into it", camera: "fisheye push-in" },
      { shot: "reaching one hand toward the fisheye lens", camera: "hand-to-lens reveal" },
      { shot: "slowly circling around the stationary camera", camera: "orbit around fixed lens" },
      { shot: "camera placed on the floor as the subject walks around it", camera: "ground-level fisheye rotation" },
      { shot: "camera tilted upward as the subject bends toward the lens", camera: "upward fisheye tilt" },
      { shot: "distorted wide shot with the environment curving around the subject", camera: "curved-perspective drift" },
      { shot: "face close to the lens while the subject's body recedes behind it", camera: "fisheye pullback" },
      { shot: "camera rotating slightly as the subject stares into the lens", camera: "rolling fisheye rotation" },
      { shot: "moving past the lens with warped motion", camera: "warped pass-by move" },
      { shot: "camera positioned between the subject's feet as the subject looks down", camera: "upward fisheye tilt" },
      { shot: "fisheye shot from inside a doorway as the subject approaches", camera: "doorway fisheye push-in" },
      { shot: "looking through glass directly into the lens", camera: "glass distortion drift" },
      { shot: "framing the camera with both hands", camera: "hands-around-lens reveal" },
      { shot: "low fisheye shot with the subject's hair falling toward the camera", camera: "low fisheye tilt" },
      { shot: "leaning into the lens, then suddenly pulling away", camera: "rapid fisheye pullback" },
      { shot: "distorted reflection in curved glass or a mirror", camera: "curved reflection slide" },
      { shot: "camera placed on the ground while the subject walks around it", camera: "ground-level orbit" },
      { shot: "camera pushed toward the subject's face, then pulled back to full body", camera: "fisheye push-pull" },
      { shot: "fisheye lens pointed upward as the subject spins above it", camera: "upward spinning fisheye" },
      { shot: "singing directly into the lens with exaggerated perspective", camera: "fisheye vocal push-in" },
      { shot: "handheld camera held at arm's length as the subject looks into it", camera: "handheld fisheye sway" },
      { shot: "centered fisheye shot with the entire background bending around the subject", camera: "centered warped drift" },
    ],
  },
  custom: {
    label: "Custom",
    framing_candidates: true,
    description: "Uses the project-specific camera-shot list imported by the user.",
    guidance: "Use only the selected framing from the user's custom camera-shot list. Choose the strongest fit for the lyrics, story, character motion, performance, and location. Preserve distinct shot variety across the scene and avoid unnecessary repetition across scenes; sensible reuse is allowed when the action clearly calls for it or the candidate pool is exhausted.",
    sequence: [],
  },
  quiet: {
    label: "Quiet dramatic",
    description: "Slower restrained camera choices for emotional, eerie, or cinematic scenes.",
    sequence: [
      { shot: "establishing shot", camera: "slow cinematic drift" },
      { shot: "medium wide shot", camera: "locked-off shot" },
      { shot: "profile shot", camera: "subtle handheld movement" },
      { shot: "intimate close-up shot", camera: "slow zoom out" },
      { shot: "reflection shot", camera: "focus pull" },
      { shot: "centered shot", camera: "pull back" },
      { shot: "silhouette shot", camera: "tilt up" },
      { shot: "close-up shot", camera: "drifting camera move" },
    ],
  },
  energetic: {
    label: "Fast energetic",
    description: "Bigger changes between scenes with fast moves, reveals, tracking, and punchier reframing.",
    sequence: [
      { shot: "wide shot", camera: "whip pan transition" },
      { shot: "medium shot", camera: "track left" },
      { shot: "close-up shot", camera: "whip right" },
      { shot: "low-angle shot", camera: "orbit reveal" },
      { shot: "full-body shot", camera: "dolly left" },
      { shot: "Dutch angle shot", camera: "push-through transition" },
      { shot: "medium wide shot", camera: "crane up" },
      { shot: "reaction shot", camera: "rack focus" },
      { shot: "tracking shot", camera: "chase shot" },
      { shot: "detail shot", camera: "snap zoom" },
    ],
  },
};

export const PERFORMANCE_STYLE_PRESETS = [
  {
    value: "off",
    label: "Off",
    direction: "",
  },
  {
    value: "",
    label: "Default cinematic",
    direction: "Use a natural cinematic music-video performance with visible emotion, expressive face, motivated body language, and camera energy that fits the scene.",
  },
  {
    value: "rock_punk",
    label: "Rock / punk",
    direction: "Use raw rock performance energy: intense facial emotion, head movement, sharp gestures, defiant posture, and gritty stage-like body language.",
  },
  {
    value: "metal_screaming",
    label: "Metal / screaming",
    direction: "Use aggressive high-intensity performance energy: fierce expression, powerful stance, forceful gestures, hair and clothing reacting to motion, and heavy dramatic presence.",
  },
  {
    value: "rap_hiphop",
    label: "Rap / hip-hop",
    direction: "Use rap-style energy instead of soft singing: confident direct-to-camera presence, expressive hand gestures, head nods, shoulder movement, and sharper body language.",
  },
  {
    value: "pop_performance",
    label: "Pop performance",
    direction: "Use polished pop performance energy: expressive singing, clean confident movement, controlled gestures, direct eye contact, stylish body language, and camera-friendly emotion.",
  },
  {
    value: "ballad_emotional",
    label: "Ballad / emotional",
    direction: "Use emotional ballad performance energy: vulnerable facial expression, slower gestures, longing eyes, subtle hand movement, restrained body language, and intimate camera presence.",
  },
  {
    value: "rnb_smooth",
    label: "R&B / smooth",
    direction: "Use smooth R&B performance energy: relaxed confident expression, controlled sensual movement, gentle hand gestures, soft rhythmic body motion, and close emotional intensity.",
  },
  {
    value: "edm_club",
    label: "EDM / club",
    direction: "Use energetic club performance energy: rhythmic movement, dance-like gestures, bright reactive expression, beat-driven body language, and dynamic camera motion.",
  },
  {
    value: "spoken_word",
    label: "Spoken word",
    direction: "Use spoken-word energy instead of singing: focused eyes, intentional gestures, restrained intensity, and poetic performance presence.",
  },
  {
    value: "no_vocals_broll",
    label: "No vocals / B-roll",
    direction: "Do not include singing, rapping, speaking, lip-sync, mouth movement, microphones, or vocal performance. Use visual action, environment interaction, and mood-driven movement only.",
  },
];

export function storyboardPerformancePreset(value = "") {
  return PERFORMANCE_STYLE_PRESETS.find((item) => item.value === value) || PERFORMANCE_STYLE_PRESETS[0];
}

export const FACIAL_PERFORMANCE_PRESETS = [
  {
    value: "off",
    label: "Off",
    description: "Do not attach facial direction",
    direction: "",
  },
  {
    value: "",
    label: "Default natural",
    description: "Natural expressive face",
    direction: "Use natural expressive facial performance: engaged eyes, subtle natural eye movement, active brows, subtle cheek and jaw movement, visible emotion that fits the lyric or scene, and occasional natural blinking.",
  },
  {
    value: "pop_polished",
    label: "Pop / polished stage",
    description: "Camera-ready pop emotion",
    direction: "Use polished pop-star facial performance: bright eyes, subtle natural eye movement, direct camera gaze, soft confident smile, playful smirk, relaxed brows, slight head tilts, lips slightly parted while singing, charming camera-ready expression, and occasional natural blinking.",
  },
  {
    value: "pop_flirty",
    label: "Pop / playful flirty",
    description: "Playful, charming pop face",
    direction: "Use playful pop facial performance: flirty smile, coy glance, subtle natural eye movement, light pout, glossy pout, raised brows, charming direct gaze, playful smirk, subtle head tilt, lips slightly parted while singing, and occasional natural blinking.",
  },
  {
    value: "love_tender",
    label: "Love song / tender",
    description: "Soft romantic expression",
    direction: "Use tender love-song facial performance: softened eyes, subtle natural eye movement, warm smile, affectionate gaze, raised inner brows, gentle head tilt, relaxed cheeks, subtle vulnerable emotion, and occasional natural blinking.",
  },
  {
    value: "sad_wounded",
    label: "Sad / wounded",
    description: "Grief, hurt, vulnerability",
    direction: "Use wounded sad-song facial performance: lowered gaze, heavy or watery eyes, subtle natural eye movement, raised inner brows, pinched brows, downturned mouth, trembling lips or chin when appropriate, defeated expression, and occasional natural blinking.",
  },
  {
    value: "happy_joyful",
    label: "Happy / joyful",
    description: "Bright and joyful",
    direction: "Use joyful facial performance: bright smile, smiling eyes, subtle natural eye movement, raised cheeks, delighted expression, playful gaze, lifted mouth corners, relaxed brows, head tilt with smile, and occasional natural blinking.",
  },
  {
    value: "rock_intense",
    label: "Rock / intense",
    description: "Gritty rock intensity",
    direction: "Use intense rock facial performance: focused stare, subtle natural eye movement, furrowed brows, defiant smirk, clenched jaw, gritty emotional strain, sharp eye contact, forceful singing expression, and occasional natural blinking.",
  },
  {
    value: "metal_rage",
    label: "Metal / rage",
    description: "Aggressive heavy metal face",
    direction: "Use aggressive heavy metal facial performance: fierce stare, subtle natural eye movement, furrowed brows, wild eyes, clenched jaw, snarling mouth shapes during vocals, bared teeth on powerful notes, flared nostrils, strained neck intensity, raw emotional scream expression, and occasional natural blinking.",
  },
  {
    value: "rap_high_intensity",
    label: "Rap / high intensity",
    description: "Sharp rap delivery",
    direction: "Use high-intensity rap facial performance: intense stare, sharp eye contact, subtle natural eye movement, furrowed brows, animated eyes, confident smirk, tight jaw, mouth open mid-verse, fast-moving mouth during delivery, challenging look, victory grin, and occasional natural blinking.",
  },
  {
    value: "custom",
    label: "Custom",
    description: "Use custom facial text",
    direction: "",
  },
];

export function storyboardFacialPerformancePreset(value = "") {
  return FACIAL_PERFORMANCE_PRESETS.find((item) => item.value === value) || FACIAL_PERFORMANCE_PRESETS[0];
}

export const ID_LORA_PERFORMANCE_STYLE_PRESETS = [
  {
    value: "dialogue_naturalism",
    label: "Dialogue naturalism",
    direction: "Use grounded short-film acting: conversational timing, motivated gestures, lived-in posture, subtle emotional shifts, and behavior that feels observed rather than performed.",
  },
  {
    value: "tense_confrontation",
    label: "Tense confrontation",
    direction: "Use restrained confrontation energy: clipped gestures, guarded posture, controlled anger, charged pauses, and body language that suggests pressure under the surface.",
  },
  {
    value: "indie_drama",
    label: "Indie drama",
    direction: "Use intimate indie-film acting: small revealing gestures, vulnerable stillness, natural imperfections, quiet tension, and emotionally specific reactions.",
  },
  {
    value: "noir_restraint",
    label: "Noir restraint",
    direction: "Use noir-style restraint: low-key confidence, suspicious glances, minimal gestures, guarded delivery, and tension carried through posture and eyes.",
  },
  {
    value: "comedic_awkwarness",
    label: "Comedic awkward",
    direction: "Use dry comedic acting: awkward pauses, slightly mismatched reactions, contained embarrassment, small nervous gestures, and believable conversational timing.",
  },
  {
    value: "emotional_confession",
    label: "Emotional confession",
    direction: "Use confession-scene acting: exposed emotion, hesitant gestures, wavering confidence, visible vulnerability, and a line delivery that feels personally risky.",
  },
  {
    value: "suspense_dread",
    label: "Suspense dread",
    direction: "Use suspense-film tension: alert posture, careful stillness, anxious scanning, controlled breathing, and reactions that imply something important is about to break.",
  },
  {
    value: "punk_bar_attitude",
    label: "Punk bar attitude",
    direction: "Use gritty punk-bar acting: defiant posture, sharp side-eye, casual toughness, impatient gestures, and messy lived-in confidence without turning it into a stage performance.",
  },
];

export const ID_LORA_FACIAL_PERFORMANCE_PRESETS = [
  {
    value: "",
    label: "Default screen acting",
    description: "Natural film face",
    direction: "Use grounded screen-acting facial detail: attentive eyes, small brow changes, readable thought, subtle jaw tension, natural mouth shapes for speech, and emotion that fits the dialogue.",
  },
  {
    value: "curious_inquisitive",
    label: "Curious / inquisitive",
    description: "Curious screen expression",
    direction: "Use curious facial performance: bright attentive eyes, slight head angle, lifted brow, searching gaze, relaxed mouth between words, and a sense of active listening.",
  },
  {
    value: "guarded_suspicious",
    label: "Guarded / suspicious",
    description: "Guarded tension",
    direction: "Use guarded facial performance: narrowed eyes, tight jaw, controlled mouth, skeptical brow, held gaze, and restrained suspicion under the dialogue.",
  },
  {
    value: "defiant_controlled",
    label: "Defiant / controlled",
    description: "Controlled defiance",
    direction: "Use controlled defiance: steady eye contact, tense mouth corners, lifted chin, compressed jaw, and a look that refuses to back down.",
  },
  {
    value: "vulnerable_confession",
    label: "Vulnerable confession",
    description: "Exposed emotion",
    direction: "Use vulnerable confession facial performance: softened eyes, raised inner brows, small uncertain mouth movements, visible hesitation, and emotion barely held together.",
  },
  {
    value: "dry_comedic",
    label: "Dry comedic",
    description: "Subtle comedy face",
    direction: "Use dry comedic facial performance: tiny reaction beats, restrained disbelief, awkward half-smile, quick eye shifts, and understated embarrassment.",
  },
  {
    value: "custom",
    label: "Custom",
    description: "Use custom facial text",
    direction: "",
  },
];

function storyboardMotionFamily(motion = "") {
  const text = String(motion || "").toLowerCase();
  if (/push|dolly in|zoom in|track forward|crash zoom|snap zoom/.test(text)) return "in";
  if (/pull|dolly out|zoom out|track backward/.test(text)) return "out";
  if (/orbit|arc|circle|rotation|rotate/.test(text)) return "orbit";
  if (/track|follow|dolly left|dolly right|truck/.test(text)) return "track";
  if (/reveal|tilt|crane|jib|rise|descend/.test(text)) return "reveal";
  if (/focus|rack/.test(text)) return "focus";
  return text.split(/\s+/).slice(0, 2).join(" ");
}

export function storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion = "", customSequence = []) {
  const preset = STORYBOARD_CAMERA_FLOW_PRESETS[profileKey] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
  const customFlow = profileKey === "custom";
  const sequence = profileKey === "custom"
    ? normalizeStoryboardCustomCameraFlowSequence(customSequence)
    : (preset.sequence || []);
  if (!sequence.length) return null;
  let entry = sequence[sceneIndex % sequence.length];
  // A custom list is an authored shot-by-shot plan. Preserve its exact order;
  // the repetition-avoidance rule is only for generated preset sequences.
  if (!customFlow && previousMotion && storyboardMotionFamily(entry.camera) === storyboardMotionFamily(previousMotion)) {
    entry = sequence[(sceneIndex + 1) % sequence.length] || entry;
  }
  return entry;
}

export const STORYBOARD_IMAGE_SHOT_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill still-image shot/composition fields.",
    sequence: [],
  },
  intimate: {
    label: "Intimate character shots",
    description: "Close, emotional stills for faces, hands, expressions, and quiet character moments.",
    sequence: [
      "intimate close-up shot",
      "medium close-up shot",
      "eyes shot",
      "hands shot",
      "profile shot",
      "head-and-shoulders shot",
      "reflection shot",
      "moody close-up shot",
    ],
  },
  music_video_stills: {
    label: "Music video stills",
    description: "Album-cover and performance-friendly framing with cinematic variety but no camera movement.",
    sequence: [
      "medium shot",
      "low-angle shot",
      "wide shot",
      "hero shot",
      "Dutch angle shot",
      "silhouette shot",
      "full-body shot",
      "dramatic low-angle shot",
      "centered shot",
      "beauty shot",
    ],
  },
  editorial: {
    label: "Editorial fashion",
    description: "Stylized portrait, fashion, and magazine-like compositions.",
    sequence: [
      "editorial fashion composition",
      "beauty shot",
      "full-body shot",
      "profile shot",
      "wide environmental still",
      "centered symmetrical composition",
      "negative space composition",
      "commercial product shot",
    ],
  },
  cinematic_story: {
    label: "Cinematic story frames",
    description: "Film-still composition for locations, story beats, and emotionally readable scenes.",
    sequence: [
      "establishing shot",
      "medium wide shot",
      "over-the-shoulder shot",
      "frame-within-a-frame shot",
      "environment shot",
      "reflection shot",
      "silhouette shot",
      "detail shot",
      "wide shot",
    ],
  },
  film_dialogue_coverage: {
    label: "Film dialogue coverage",
    description: "Short-film coverage for story-heavy music videos: readable faces, eyelines, reactions, and location context.",
    sequence: [
      "medium close-up dialogue shot",
      "over-the-shoulder shot",
      "reaction close-up",
      "two-shot dialogue frame",
      "profile close-up",
      "medium shot with foreground framing",
      "insert detail shot",
      "wide establishing film still",
    ],
  },
  intimate_drama: {
    label: "Intimate drama frames",
    description: "Close emotional film stills for confessions, quiet tension, and character-led music-video scenes.",
    sequence: [
      "tight close-up",
      "intimate medium close-up",
      "profile close-up",
      "hands and face detail shot",
      "reflection close-up",
      "seated conversation frame",
      "shallow-focus reaction shot",
      "low-key portrait frame",
    ],
  },
  noir_story_frames: {
    label: "Noir story frames",
    description: "Moody dramatic coverage with shadows, silhouettes, foregrounds, and tense blocking.",
    sequence: [
      "low-key medium shot",
      "silhouette dialogue frame",
      "over-the-shoulder noir shot",
      "frame-within-a-frame shot",
      "side-lit profile shot",
      "wide empty-space composition",
      "reflection shot",
      "detail insert shot",
    ],
  },
};

export const ID_LORA_IMAGE_SHOT_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill film-still composition fields.",
    sequence: [],
  },
  film_dialogue_coverage: {
    label: "Film dialogue coverage",
    description: "Short-film coverage for dialogue scenes: readable faces, eyelines, reactions, and location context.",
    sequence: [
      "medium close-up dialogue shot",
      "over-the-shoulder shot",
      "reaction close-up",
      "two-shot dialogue frame",
      "profile close-up",
      "medium shot with foreground framing",
      "insert detail shot",
      "wide establishing film still",
    ],
  },
  intimate_drama: {
    label: "Intimate drama frames",
    description: "Close emotional film stills for confessions, quiet tension, and character-led scenes.",
    sequence: [
      "tight close-up",
      "intimate medium close-up",
      "profile close-up",
      "hands and face detail shot",
      "reflection close-up",
      "seated conversation frame",
      "shallow-focus reaction shot",
      "low-key portrait frame",
    ],
  },
  noir_story_frames: {
    label: "Noir story frames",
    description: "Moody dramatic coverage with shadows, silhouettes, foregrounds, and tense blocking.",
    sequence: [
      "low-key medium shot",
      "silhouette dialogue frame",
      "over-the-shoulder noir shot",
      "frame-within-a-frame shot",
      "side-lit profile shot",
      "wide empty-space composition",
      "reflection shot",
      "detail insert shot",
    ],
  },
};

export const STORYBOARD_IMAGE_AESTHETIC_PRESETS = [
  { value: "", label: "Default cinematic still", description: "Balanced cinematic lighting, color, and texture for a polished text-to-image prompt.", prompt_guidance: "Create a polished cinematic still with clear subject placement, believable wardrobe and environment details, purposeful lighting, readable composition, lens/framing detail, and a strong music-video production still feeling." },
  { value: "music_video_gloss", label: "Glossy music video", description: "Glossy high-production music-video still, dramatic color contrast, stylish lighting, album-cover polish.", prompt_guidance: "Build a glossy high-production music-video still. Specify stylized wardrobe, intentional pose, dramatic color contrast, polished hair and makeup, expensive-looking lighting, reflective or atmospheric set details, album-cover composition, crisp lens choice, and cinematic depth. Do not merely say glossy music video." },
  { value: "dark_neon", label: "Dark neon", description: "Dark cinematic neon lighting, saturated color accents, glossy reflections, smoky atmosphere, night-club energy.", prompt_guidance: "Build a dark neon cinematic still. Use saturated colored light sources, glossy reflections, wet or polished surfaces, smoke/haze, rim light, deep shadows, vivid accent colors on the subject, and a nightlife or futuristic music-video atmosphere. Describe where the neon comes from and how it shapes the face, outfit, and environment." },
  { value: "editorial_fashion", label: "Editorial fashion", description: "High-fashion editorial photography, intentional posing, refined wardrobe detail, magazine-grade lighting.", prompt_guidance: "Build an editorial fashion photograph, not a plain portrait. Give the subject a deliberate model pose with body angles, hand placement, posture, and gaze. Describe refined wardrobe styling, fabric behavior, accessories, hair/makeup direction, fashion-magazine lighting, background styling, composition, lens/framing, and a strong art-directed theme." },
  { value: "editorial_fashion_photography", label: "Editorial fashion photography", description: "Editorial fashion photography with confident model posing, dramatic styling, creative wardrobe themes, magazine-grade composition, bold makeup and hair, and polished high-resolution lighting.", prompt_guidance: "Build a detailed editorial fashion photograph. Include a confident model pose, strong body line, hand/shoulder/hip placement, dramatic styling choices, creative wardrobe concept, fabric texture and silhouette, bold hair and makeup, accessories, modern magazine composition, art-directed setting, high-resolution studio or location lighting, and a clear fashion story. Do not just write 'editorial fashion composition'." },
  { value: "conceptual_portrait_photography", label: "Conceptual portrait photography", description: "Conceptual portrait photography built around a clear visual idea, symbolic prop, emotional pose, controlled environment, cinematic lighting, and a strong central portrait composition.", prompt_guidance: "Build a conceptual portrait around one clear visual idea. Choose a symbolic prop, object arrangement, or environmental metaphor that fits the scene. Describe the subject's pose, relation to the prop, wardrobe, hair/makeup, controlled setting, color palette, lighting direction, mood, lens/framing, and how the composition communicates the concept visually without explaining it." },
  { value: "avant_garde_fashion_photography", label: "Avant-garde fashion photography", description: "Avant-garde fashion photography with unusual makeup, sculptural hair, strange or powerful poses, experimental styling, abstract studio or surreal setting, and bold high-contrast lighting.", prompt_guidance: "Build an avant-garde fashion photograph. Use unusual makeup, sculptural or geometric hair, experimental wardrobe shape, exaggerated silhouette, strange powerful pose, asymmetrical composition, abstract studio or surreal set design, hard shadows or high-contrast light, unexpected materials, and a bold futuristic or theatrical fashion mood. Make it visually daring, not casual." },
  { value: "beauty_editorial_photography", label: "Beauty editorial photography", description: "Beauty editorial photography focused on close-up makeup, hair, skin texture, eyes, lips, jewelry or face details, soft luxury lighting, and clean magazine beauty composition.", prompt_guidance: "Build a beauty editorial photograph. Use close-up or tight portrait framing focused on eyes, lips, makeup, hair texture, jewelry, nails, skin glow, and face-framing styling. Describe makeup colors, glossy or matte finish, hair placement, accessories near the face, soft diffused lighting, clean backdrop, shallow depth of field, and luxury magazine composition." },
  { value: "high_fashion_editorial", label: "High fashion editorial", description: "High fashion editorial photography inspired by dramatic fashion competition shoots: couture wardrobe, expressive posing, epic location, wind or fabric movement, glamorous styling, and cinematic full-body framing.", prompt_guidance: "Build a high fashion editorial shoot like a dramatic fashion competition photo. Use couture-level wardrobe, exaggerated fabric movement, strong full-body or three-quarter pose, elongated body line, expressive hands and face, wind or motion in hair/fabric, glamorous accessories, bold makeup, epic location styling, low or cinematic camera angle, dramatic natural or studio lighting, and a clear fashion-story payoff. The prompt must describe the actual fashion shoot details, not just name the style." },
  { value: "creative_portrait_photography", label: "Creative portrait photography", description: "Creative portrait photography with a posed subject, strong visual theme, props or animals when appropriate, colorful art direction, expressive styling, and a memorable environment.", prompt_guidance: "Build a creative portrait photograph with a strong visual theme. Include a posed subject, purposeful prop or themed object if appropriate, color-directed wardrobe, expressive hair/makeup, layered environment details, playful or artistic composition, lens/framing, lighting style, and a memorable subject-environment relationship. If an animal or prop is used, make it clearly integrated into the scene concept." },
  { value: "gritty_analog", label: "Gritty analog", description: "Gritty analog film look, visible texture, natural imperfections, moody documentary realism.", prompt_guidance: "Build a gritty analog film still with imperfect realism: visible film grain, practical lighting, worn textures, imperfect surfaces, muted color response, handheld or documentary-feeling framing, natural body posture, atmospheric shadows, and a lived-in environment. Avoid overly polished studio language." },
  { value: "soft_dream_pop", label: "Soft dream pop", description: "Soft dreamy pop aesthetic, gentle bloom, pastel color, romantic haze, delicate cinematic lighting.", prompt_guidance: "Build a soft dream-pop still with gentle bloom, pastel color palette, romantic haze, delicate backlight, floating or soft fabric details, dreamy hair/makeup styling, graceful pose, shallow depth of field, soft environment edges, and a light emotional music-video mood." },
  { value: "high_contrast_drama", label: "High-contrast drama", description: "Bold shadows, sculpted highlights, intense facial emotion, dramatic production-still lighting.", prompt_guidance: "Build a high-contrast dramatic still with sculpted highlights, deep shadows, strong key light direction, visible tension in posture, intense facial emotion, dramatic wardrobe silhouette, textured environment, cinematic contrast ratio, and a composition that creates visual pressure." },
  { value: "surreal_symbolic", label: "Surreal symbolic", description: "Surreal symbolic music-video still, heightened atmosphere, poetic objects, dreamlike composition.", prompt_guidance: "Build a surreal symbolic music-video still. Use poetic visual motifs, dreamlike composition, unusual scale or placement of objects, symbolic set dressing, atmospheric light, controlled color palette, and a subject pose that feels ritualistic or uncanny. Keep the imagery visual and concrete rather than explanatory." },
  { value: "clean_studio", label: "Clean studio", description: "Clean studio photography, crisp subject detail, controlled lighting, uncluttered composition.", prompt_guidance: "Build a clean studio photograph with crisp subject detail, controlled lighting setup, precise wardrobe styling, polished hair/makeup, uncluttered backdrop, intentional pose, clear silhouette, lens/framing detail, and professional commercial or editorial clarity." },
  { value: "film_default", label: "Default film still", description: "Balanced short-film still lighting, believable production design, natural texture, and cinematic composition.", prompt_guidance: "Build a polished film-style music-video still. Use believable character blocking, grounded wardrobe, practical lighting, lens/framing detail, textured production design, natural color contrast, emotionally readable composition, and a cinematic story-frame finish." },
  { value: "indie_film_naturalism", label: "Indie film naturalism", description: "Naturalistic indie-drama still with lived-in details, imperfect realism, and intimate character focus.", prompt_guidance: "Build an indie-film music-video still with naturalistic lighting, lived-in wardrobe, imperfect textures, believable posture, intimate framing, subtle emotional detail, muted color response, and environment details that feel observed rather than staged." },
  { value: "neo_noir_dialogue", label: "Neo-noir dialogue", description: "Low-key shadows, practical neon, suspicious glances, dramatic contrast, and noir-style tension.", prompt_guidance: "Build a neo-noir dialogue still with low-key lighting, practical neon or sodium light, deep shadows, hard rim light, reflective surfaces, guarded facial expression, tense blocking, and a controlled color palette. Keep it cinematic and grounded." },
  { value: "gritty_punk_bar", label: "Gritty punk bar", description: "Worn bar textures, punk attitude, practical stage/neon light, smoky atmosphere, and analog grit.", prompt_guidance: "Build a gritty punk-bar film still with worn leather or denim styling, messy lived-in hair/makeup, scratched tables, stickers, posters, dim practical lights, colored neon spill, smoky air, visible texture, defiant posture, and a raw 35mm cinematic finish." },
  { value: "psychological_thriller", label: "Psychological thriller", description: "Uneasy framing, controlled color, negative space, tense facial detail, and subtle dread.", prompt_guidance: "Build a psychological-thriller still with uneasy composition, negative space, controlled color palette, tense facial detail, practical low light, slightly off-balance framing, foreground obstruction, and environmental details that imply pressure without explaining it." },
  { value: "warm_dialogue_drama", label: "Warm dialogue drama", description: "Warm practical interiors, soft skin tones, intimate framing, and emotionally readable acting.", prompt_guidance: "Build a warm dialogue-drama still with practical lamp, street, stage, or bar light, gentle skin tones, shallow depth of field, intimate framing, small emotional facial detail, believable wardrobe, textured surroundings, and a quiet cinematic finish." },
  { value: "35mm_analog_film", label: "35mm analog film", description: "Film grain, practical lighting, imperfect texture, grounded color, and documentary-like realism.", prompt_guidance: "Build a 35mm analog film still with visible grain, practical lighting, imperfect surfaces, grounded color response, natural posture, textured wardrobe, shallow lens character, and a lived-in environment. Avoid glossy music-video polish unless the scene asks for it." },
];

const MINIMAX_VIDEO_STYLE_LABELS = [
  "Cinematic realism", "Gothic romance", "Dark fantasy", "Ethereal dreamscape", "Surrealism", "Cosmic horror",
  "Psychological horror", "Found footage", "Analog horror", "Body horror", "Occult ritual", "Silent Hill-inspired",
  "Cyberpunk", "Biopunk", "Dieselpunk", "Steampunk", "Post-apocalyptic", "Dystopian sci-fi", "Retro-futurism",
  "Y2K futurism", "Vaporwave", "Synthwave", "Dreamcore", "Weirdcore", "Liminal space", "Dark academia",
  "Cottagecore", "Fairycore", "Angelcore", "Goblincore", "Whimsigoth", "Baroque", "Rococo", "Art Nouveau",
  "Art Deco", "Victorian gothic", "Renaissance-inspired", "Medieval fantasy", "Mythological epic", "Film noir",
  "Neo-noir", "Expressionism", "Giallo horror", "Grindhouse", "1970s psychedelic", "1980s music video",
  "1990s grunge", "Early-2000s pop", "Indie sleaze", "Lo-fi VHS", "Super 8 film", "Vintage Hollywood",
  "High-fashion editorial", "Avant-garde fashion", "Runway glamour", "Luxury commercial", "Beauty campaign",
  "Pop-star music video", "Industrial metal", "Gothic metal", "Alternative rock", "Punk rock", "Dark pop",
  "Hyperpop", "K-pop-inspired", "R&B glamour", "Eerie claymation", "Stop-motion", "Paper-cut animation",
  "Hand-painted animation", "Anime-inspired", "Graphic novel", "Comic-book", "Cel-shaded 3D", "Photorealistic CGI",
  "Low-poly 3D", "Miniature diorama", "Dollhouse surrealism", "Liquid chrome", "Holographic iridescence",
  "Neon noir", "Monochrome minimalism", "High-key white studio", "Low-key chiaroscuro", "Soft pastel",
  "Desaturated melancholy", "Crimson-and-black", "Teal-and-orange blockbuster", "Golden-hour nostalgia",
  "Moonlit blue", "Underwater ethereal", "Elemental fantasy", "Nature mysticism", "Apocalyptic biblical",
  "Glitch art", "Datamosh", "CRT distortion", "Kaleidoscopic", "Double exposure", "Infrared", "Thermal vision",
  "Fisheye distortion", "Security-camera footage", "Documentary realism", "Social-media selfie", "TikTok transformation",
  "Dreamlike slow motion", "Frenetic montage", "One-take immersive", "Music-video performance",
  "Narrative short film", "Movie-trailer aesthetic",
];

const MINIMAX_VIDEO_STYLE_VERBIAGE = {
  "Cinematic realism": "Naturalistic practical lighting, restrained color grading, balanced contrast, subtle film grain, realistic skin texture, neutral tones, believable materials, and polished cinematic clarity throughout.",
  "Gothic romance": "Deep burgundy, black, and ivory tones, soft shadows, luminous highlights, ornate details, rich velvet and lace textures, candlelit atmosphere, and melancholic visual softness throughout.",
  "Dark fantasy": "Shadow-heavy lighting, desaturated earth tones, metallic accents, dramatic contrast, weathered textures, monumental fantasy production design, atmospheric haze, and richly cinematic grading throughout.",
  "Ethereal dreamscape": "Pastel colors, diffused highlights, soft focus, glowing edges, translucent layers, pearlescent haze, low contrast, and weightless dreamlike beauty throughout.",
  "Surrealism": "Unexpected proportions, distorted perspective, symbolic abstraction, unnatural colors, impossible spatial relationships, uncanny objects, and deliberately illogical dream imagery throughout.",
  "Cosmic horror": "Near-black palettes, cold highlights, immense scale, distorted geometry, oppressive shadows, ancient textures, unsettling negative space, and incomprehensible otherworldly detail throughout.",
  "Psychological horror": "Sickly restrained color, oppressive shadow, uneasy negative space, subtly distorted interiors, harsh practical light, clammy skin tones, ambiguous background details, and persistent visual dread throughout.",
  "Found footage": "Authentic in-world consumer-camera imagery with practical available light, imperfect exposure, mild autofocus softness, sensor noise, compression artifacts, clipped highlights, noisy shadows, subdued color, and an unpolished documentary texture. Keep faces readable; avoid glossy grading, studio polish, pristine sharpness, and cinematic glamour.",
  "Analog horror": "Degraded videotape imagery, faded color, tracking noise, scan lines, chromatic bleed, warped edges, crushed blacks, blown highlights, timestamp-like visual language without readable text, and ominous broadcast-era texture throughout.",
  "Body horror": "Visceral organic textures, pallid flesh tones, wet highlights, anatomical distortion, diseased surfaces, clinical details, bruised color accents, harsh close detail, and deeply unsettling physical materiality throughout.",
  "Occult ritual": "Candlelit darkness, ceremonial symbols, weathered stone, smoke, wax, ash, deep red and black accents, antique ritual objects, symmetrical arrangements, and secretive sacred atmosphere throughout.",
  "Silent Hill-inspired": "Dense pale fog, rusted industrial surfaces, damp concrete, peeling walls, muted gray-green color, dirty amber light, corroded metal, abandoned spaces, and oppressive psychological decay throughout.",
  "Cyberpunk": "Neon magenta, cyan, and electric blue light, rain-slick surfaces, holographic signage shapes without readable text, dense urban technology, reflective synthetic materials, high contrast, and gritty futuristic detail throughout.",
  "Biopunk": "Organic technology, translucent membranes, bone-like structures, cultured tissue, surgical hardware, sickly green and amber light, wet biological surfaces, laboratory grime, and engineered-life detail throughout.",
  "Dieselpunk": "Oil-stained metal, riveted machinery, soot, heavy industrial architecture, military-era styling, muted olive and rust colors, hard smoky light, analog gauges, and imposing mechanical detail throughout.",
  "Steampunk": "Aged brass, copper pipes, leather, polished wood, intricate gears, Victorian tailoring, warm amber light, steam-filled atmosphere, engraved ornament, and handcrafted mechanical detail throughout.",
  "Post-apocalyptic": "Sun-bleached ruins, scavenged materials, dust, rust, broken infrastructure, weathered clothing, harsh natural light, muted earth colors, and layered environmental decay throughout.",
  "Dystopian sci-fi": "Monumental controlled architecture, cold gray-blue palettes, severe uniforms, sterile surfaces, surveillance motifs without readable text, stark artificial light, rigid visual order, and oppressive technological detail throughout.",
  "Retro-futurism": "Optimistic vintage future design, chrome, molded plastic, analog controls, bold geometric forms, saturated period color, glowing panels, clean illustrative surfaces, and nostalgic speculative detail throughout.",
  "Y2K futurism": "Glossy silver, translucent plastics, icy blue and white palettes, bubble-like interfaces without readable text, chrome accessories, soft digital glow, clean synthetic surfaces, and early-digital optimism throughout.",
  "Vaporwave": "Pastel pink, lavender, aqua, and sunset gradients, marble surfaces, retro computer textures, classical-statue motifs, soft haze, luminous grid-like design, and nostalgic digital unreality throughout.",
  "Synthwave": "Hot magenta, violet, and electric cyan palettes, deep black silhouettes, neon grids, glossy reflections, dramatic sunset gradients, chrome accents, and polished retro-electronic atmosphere throughout.",
  "Dreamcore": "Soft familiar spaces, hazy pastel light, washed color, low-detail backgrounds, uncanny childhood objects, gentle bloom, empty interiors, and comforting yet disorienting dream imagery throughout.",
  "Weirdcore": "Low-resolution digital texture, awkward cropping, mismatched color, uncanny ordinary objects, liminal interiors, crude graphic shapes without readable text, visual noise, and deliberately unsettling internet-era imagery throughout.",
  "Liminal space": "Empty transitional architecture, fluorescent or sodium lighting, repetitive corridors, vacant rooms, muted institutional color, dated surfaces, deep vanishing points, and eerily familiar stillness throughout.",
  "Dark academia": "Deep brown, charcoal, forest green, and oxblood tones, old books, dark wood, worn leather, classical architecture, window light, dust, tweed textures, and scholarly melancholy throughout.",
  "Cottagecore": "Warm natural light, wildflowers, handmade fabrics, rustic wood, ceramics, baskets, soft earth colors, pastoral interiors, gentle weathering, and cozy rural detail throughout.",
  "Fairycore": "Mossy forests, tiny flowers, luminous dust, translucent wings, dewdrops, soft green and pastel color, miniature natural details, glowing mushrooms, and delicate enchanted atmosphere throughout.",
  "Angelcore": "Ivory and pale gold palettes, luminous white fabric, soft clouds, radiant backlight, delicate feathers, sacred ornament, pearlescent highlights, and serene celestial atmosphere throughout.",
  "Goblincore": "Muddy greens and browns, moss, mushrooms, stones, bones, jars, tarnished trinkets, damp forest textures, cluttered natural collections, and earthy mischievous detail throughout.",
  "Whimsigoth": "Midnight blue, plum, black, and antique gold tones, celestial patterns, velvet, candles, stained glass, ornate jewelry, mystical clutter, and romantic witchy atmosphere throughout.",
  "Baroque": "Deep jewel tones, dramatic light and shadow, gilded ornament, rich fabric, carved architecture, elaborate decoration, painterly highlights, and theatrical seventeenth-century grandeur throughout.",
  "Rococo": "Powder pink, pale blue, cream, and gold palettes, delicate florals, curved ornament, silk, porcelain, airy light, playful luxury, and ornate eighteenth-century elegance throughout.",
  "Art Nouveau": "Flowing botanical lines, stained glass, wrought metal, muted jewel colors, floral ornament, organic symmetry, decorative illustration, and elegant turn-of-the-century craftsmanship throughout.",
  "Art Deco": "Bold geometry, black and gold contrast, polished stone, lacquer, chrome, stepped forms, symmetrical ornament, rich jewel tones, and glamorous machine-age luxury throughout.",
  "Victorian gothic": "Black lace, dark carved wood, aged stone, gaslight, heavy drapery, mourning attire, tarnished silver, deep wine tones, and haunted nineteenth-century atmosphere throughout.",
  "Renaissance-inspired": "Warm earth pigments, rich red and blue fabric, classical architecture, fresco-like color, soft directional light, fine textile detail, balanced humanist elegance, and old-master visual richness throughout.",
  "Medieval fantasy": "Weathered stone, timber halls, chainmail, wool, leather, heraldic color, torchlight, misty landscapes, handcrafted props, and grounded legendary-world detail throughout.",
  "Mythological epic": "Monumental temples, heroic silhouettes, carved stone, bronze and gold accents, dramatic skies, ceremonial fabric, divine light, vast landscapes, and timeless legendary grandeur throughout.",
  "Film noir": "High-contrast black-and-white imagery, hard key light, venetian-blind shadows, wet streets, cigarette haze, deep blacks, bright highlights, period interiors, and morally shadowed atmosphere throughout.",
  "Neo-noir": "Deep shadows, saturated neon accents, reflective night surfaces, controlled color contrast, urban grime, practical light, smoky atmosphere, and sleek contemporary darkness throughout.",
  "Expressionism": "Angular sets, exaggerated shadows, distorted architecture, stark color or monochrome contrast, theatrical makeup, painted surfaces, and emotionally warped visual design throughout.",
  "Giallo horror": "Saturated red, yellow, blue, and green light, glossy black surfaces, ornate interiors, sharp shadows, glamorous styling, lurid practical effects, and stylish Italian horror atmosphere throughout.",
  "Grindhouse": "Faded color, heavy grain, scratched film, dirty highlights, crushed shadows, cheap practical effects, lurid wardrobe, distressed print texture, and raw exploitation-era finish throughout.",
  "1970s psychedelic": "Burnt orange, avocado, violet, and acid color, bold patterns, soft film grain, optical layering, warped graphic forms, warm haze, and richly hallucinatory period design throughout.",
  "1980s music video": "Saturated neon color, glossy highlights, smoky studio atmosphere, dramatic backlight, bold makeup, metallic wardrobe, soft diffusion, analog video texture, and theatrical pop imagery throughout.",
  "1990s grunge": "Muted dirty color, fluorescent interiors, distressed denim and flannel, photocopied graphic texture without readable text, harsh flash, visible grain, urban wear, and unpolished alternative-era realism throughout.",
  "Early-2000s pop": "Glossy candy color, icy highlights, metallic accessories, low-rise era styling, bright studio surfaces, soft skin diffusion, digital-camera crispness, and playful Y2K polish throughout.",
  "Indie sleaze": "Direct-flash nightlife imagery, blown skin highlights, deep black backgrounds, messy styling, grainy digital texture, smoky clubs, saturated accents, and deliberately careless downtown glamour throughout.",
  "Lo-fi VHS": "Soft analog resolution, tape grain, color bleed, scan lines, tracking instability, crushed blacks, clipped whites, oversaturated consumer color, and worn home-video texture throughout.",
  "Super 8 film": "Warm faded color, pronounced small-gauge grain, soft focus, halation, light leaks, flickering exposure texture, rounded highlights, and intimate home-movie character throughout.",
  "Vintage Hollywood": "Elegant studio lighting, luminous skin, rich black-and-white or restrained Technicolor tones, soft diffusion, tailored wardrobe, painted-set refinement, and classic star-era glamour throughout.",
  "High-fashion editorial": "Sculptural wardrobe, immaculate makeup, controlled color, premium fabric detail, bold graphic styling, clean luxury surfaces, precise beauty lighting, and magazine-grade visual polish throughout.",
  "Avant-garde fashion": "Experimental silhouettes, unexpected materials, abstract makeup, severe color blocking, sculptural sets, conceptual styling, high-detail fabric texture, and art-gallery fashion imagery throughout.",
  "Runway glamour": "Luxury garments, luminous skin, glossy hair, dramatic show lighting, polished surfaces, rich color, crisp textile detail, and elevated fashion-week spectacle throughout.",
  "Luxury commercial": "Pristine product-grade surfaces, controlled highlights, rich neutral color, immaculate materials, elegant reflections, premium environments, clean contrast, and expensive advertising polish throughout.",
  "Beauty campaign": "Luminous skin, refined makeup detail, soft controlled highlights, clean backgrounds, flattering color, glossy hair, delicate texture, and premium cosmetic-advertising finish throughout.",
  "Pop-star music video": "Bold saturated color, glamorous wardrobe, luminous skin, dramatic set lighting, glossy production design, metallic accents, atmospheric haze, and polished superstar imagery throughout.",
  "Industrial metal": "Cold steel, concrete, rust, oil, black leather, harsh white and red light, smoke, abrasive texture, heavy machinery, and severe high-contrast atmosphere throughout.",
  "Gothic metal": "Black leather and lace, deep crimson accents, cathedral stone, silver ornament, smoke, dramatic pale skin, low-key light, and dark romantic grandeur throughout.",
  "Alternative rock": "Lived-in rehearsal spaces, worn instruments, denim and leather, practical stage light, muted color, visible grain, textured walls, and grounded independent-band authenticity throughout.",
  "Punk rock": "Photocopied texture without readable text, torn fabric, studs, leather, raw club interiors, harsh flash, red and black accents, grime, and confrontational DIY visual energy throughout.",
  "Dark pop": "Deep black palettes, jewel-tone accents, glossy shadows, dramatic beauty light, surreal luxury details, refined makeup, controlled haze, and sleek ominous pop polish throughout.",
  "Hyperpop": "Acid neon color, chrome, glossy plastic, exaggerated digital texture, candy gradients, iridescent makeup, maximal graphic detail, and intensely synthetic internet-pop imagery throughout.",
  "K-pop-inspired": "Immaculate styling, vivid coordinated color, glossy sets, luminous skin, detailed fashion, polished hair and makeup, clean highlights, and high-budget pop perfection throughout.",
  "R&B glamour": "Warm bronze skin tones, black and gold accents, soft practical light, satin and velvet textures, elegant interiors, luminous highlights, and intimate luxury throughout.",
  "Eerie claymation": "Hand-sculpted clay surfaces, visible fingerprints, miniature sets, muted uncanny color, uneven handmade forms, soft practical miniature lighting, and tactile unsettling charm throughout.",
  "Stop-motion": "Tactile handcrafted materials, miniature practical sets, visible fabrication seams, slightly stepped pose character, controlled tabletop lighting, and charming physical-animation texture throughout.",
  "Paper-cut animation": "Layered cut-paper shapes, visible fibers, flat illustrated color, crisp silhouettes, handmade edges, shadowed paper depth, decorative patterns, and crafted collage texture throughout.",
  "Hand-painted animation": "Visible brushwork, layered pigment, painterly backgrounds, softened outlines, rich handcrafted color, canvas or watercolor texture, and expressive illustrated detail throughout.",
  "Anime-inspired": "Clean expressive linework, stylized facial features, cel-painted color, luminous eyes, graphic shadows, detailed illustrated backgrounds, controlled highlights, and polished animation-art finish throughout.",
  "Graphic novel": "Bold ink lines, dramatic shadow blocks, limited accent color, textured paper, illustrated crosshatching, high contrast, and sophisticated sequential-art atmosphere throughout.",
  "Comic-book": "Crisp outlines, saturated primary colors, halftone texture, graphic shadow shapes, stylized anatomy, printed-paper character, and energetic illustrated spectacle throughout.",
  "Cel-shaded 3D": "Three-dimensional forms with clean graphic outlines, flat color regions, stepped shadows, controlled highlights, simplified materials, and polished illustrated-game rendering throughout.",
  "Photorealistic CGI": "Physically accurate materials, realistic global illumination, detailed skin and hair, precise reflections, volumetric atmosphere, clean high-resolution rendering, and seamless digital realism throughout.",
  "Low-poly 3D": "Faceted geometry, simplified forms, flat-shaded surfaces, restrained texture, clean geometric color, stylized lighting, and intentionally economical digital design throughout.",
  "Miniature diorama": "Clearly handcrafted miniature environments, tiny scaled props, model-making textures, shallow miniature depth, painted surfaces, practical tabletop lighting, and charming physical detail throughout.",
  "Dollhouse surrealism": "Miniature domestic rooms, toy-like furniture, porcelain or plastic textures, artificial pastel color, uncanny scale relationships, pristine tiny details, and dreamlike domestic unease throughout.",
  "Liquid chrome": "Mirror-bright silver surfaces, fluid metallic forms, warped reflections, cool specular highlights, deep black contrast, futuristic polish, and glossy sculptural abstraction throughout.",
  "Holographic iridescence": "Prismatic rainbow highlights, pearlescent surfaces, translucent layers, shifting cyan-magenta color, glossy reflections, soft luminous haze, and futuristic iridescent finish throughout.",
  "Neon noir": "Near-black environments, saturated neon red, blue, and violet accents, wet reflections, hard silhouettes, smoky atmosphere, glossy urban surfaces, and brooding futuristic contrast throughout.",
  "Monochrome minimalism": "Single-hue or black-and-white palette, clean negative space, simple materials, restrained contrast, sparse production design, precise tonal separation, and elegant visual reduction throughout.",
  "High-key white studio": "Bright seamless white surroundings, soft wraparound light, low shadow density, clean neutral color, crisp product-grade detail, airy surfaces, and immaculate studio clarity throughout.",
  "Low-key chiaroscuro": "Deep black shadows, narrow pools of directional light, sculpted facial highlights, rich tonal contrast, restrained color, and dramatic painterly darkness throughout.",
  "Soft pastel": "Powder pink, pale blue, lavender, mint, and cream color, diffused light, gentle contrast, matte surfaces, delicate texture, and calm airy softness throughout.",
  "Desaturated melancholy": "Muted color, cool gray and faded earth tones, soft overcast light, low saturation, restrained highlights, subtle grain, weathered surfaces, and quiet visual sadness throughout.",
  "Crimson-and-black": "Dominant black surfaces with vivid crimson accents, deep shadows, hard red highlights, dark wardrobe, severe contrast, and intense graphic drama throughout.",
  "Teal-and-orange blockbuster": "Cool teal shadows, warm amber skin and highlights, strong complementary contrast, polished surfaces, atmospheric depth, controlled saturation, and large-scale commercial cinema finish throughout.",
  "Golden-hour nostalgia": "Warm amber sunlight, long soft shadows, gentle haze, faded earth color, glowing skin, subtle grain, sunlit dust, and tender memory-like warmth throughout.",
  "Moonlit blue": "Deep navy and cobalt tones, cool silver highlights, soft night haze, pale skin light, subdued warm accents, dark silhouettes, and luminous nocturnal atmosphere throughout.",
  "Underwater ethereal": "Aqua and deep blue palettes, diffused caustic light, suspended particles, translucent fabric, softened detail, pearlescent highlights, and immersive aquatic beauty throughout.",
  "Elemental fantasy": "Visually dominant fire, water, air, earth, ice, or lightning motifs, richly textured natural materials, luminous energy, dramatic atmospheric light, and mythic environmental detail throughout.",
  "Nature mysticism": "Ancient forests, moss, stone, roots, mist, filtered natural light, symbolic organic details, muted green and earth color, and sacred wilderness atmosphere throughout.",
  "Apocalyptic biblical": "Monumental skies, ash and fire, stark divine light, ancient stone, distressed earth tones, ceremonial silhouettes, vast destruction, and solemn prophetic grandeur throughout.",
  "Glitch art": "Digital fragmentation, RGB channel separation, block corruption, pixel noise, scan errors, broken color fields, displaced image sections, and deliberate electronic artifacting throughout.",
  "Datamosh": "Compressed digital smearing, macroblock trails, color displacement, broken codec texture, fragmented silhouettes, melted pixel fields, and aggressive corrupted-video appearance throughout.",
  "CRT distortion": "Curved glass-screen appearance, scan lines, phosphor glow, chromatic fringing, barrel distortion, soft analog resolution, bloom, static noise, and vintage monitor texture throughout.",
  "Kaleidoscopic": "Mirrored geometric repetition, radial symmetry, jewel-like color, layered reflections, intricate patterning, luminous fragments, and hypnotic prismatic imagery throughout.",
  "Double exposure": "Layered translucent imagery, overlapping silhouettes, blended environments, luminous tonal merging, photographic grain, controlled negative space, and poetic composite texture throughout.",
  "Infrared": "False-color foliage, pale luminous skin, dark skies, unusual magenta or cyan tonal mapping, high contrast, bright vegetation, and uncanny infrared-photography texture throughout.",
  "Thermal vision": "Heat-map color ranging from deep violet and blue through red, orange, yellow, and white, simplified surface detail, glowing warm bodies, and sensor-like thermal imaging throughout.",
  "Fisheye distortion": "Pronounced barrel distortion, curved edges, expanded center perspective, compressed borders, close spatial exaggeration, and distinctive ultra-wide optical appearance throughout.",
  "Security-camera footage": "Fixed surveillance-system image quality, high or corner-mounted viewpoint appearance, wide utilitarian lens, low resolution, digital noise, flat exposure, limited color, and institutional monitoring texture without readable overlays.",
  "Documentary realism": "Available practical light, natural skin and material texture, restrained color, modest contrast, believable environments, subtle sensor grain, and honest unembellished observational realism throughout.",
  "Social-media selfie": "Front-facing phone-camera appearance, close personal perspective, wide phone-lens facial character, automatic exposure, digital sharpening, casual available light, and immediate user-generated authenticity throughout.",
  "TikTok transformation": "Bright mobile-video color, crisp phone-camera detail, bold styling contrast, clean vertical-content polish without requiring a vertical aspect ratio, beauty-filter sheen, and highly legible before-and-after visual design throughout.",
  "Dreamlike slow motion": "Soft temporal blur, luminous highlight bloom, gentle pastel or muted color, floating particles, delicate fabric detail, low contrast, and romantic dreamlike image softness throughout.",
  "Frenetic montage": "Punchy high-contrast imagery, varied but coordinated color treatments, bold graphic details, sharp texture changes, intense highlights, and fragmented editorial visual energy throughout.",
  "One-take immersive": "Naturalistic spatial continuity, consistent practical lighting, coherent production design, believable environmental depth, uninterrupted visual realism, and an immediate lived-in atmosphere throughout.",
  "Music-video performance": "Expressive stage styling, dramatic practical and colored light, atmospheric haze, polished wardrobe and makeup, rich contrast, glossy highlights, and premium performance-world production design throughout.",
  "Narrative short film": "Grounded production design, believable wardrobe, motivated practical lighting, restrained cinematic grading, detailed environments, natural skin texture, and cohesive story-world realism throughout.",
  "Movie-trailer aesthetic": "Large-scale cinematic contrast, dramatic skies and practical light, rich production design, deep blacks, luminous highlights, atmospheric depth, premium color grading, and event-film visual polish throughout.",
};

export const MINIMAX_VIDEO_STYLE_PRESETS = [
  {
    value: "",
    label: "Default / let prompt decide",
    description: "No additional global video aesthetic is imposed.",
    prompt_guidance: "",
  },
  ...MINIMAX_VIDEO_STYLE_LABELS.map((label) => ({
    value: label.toLowerCase().replace(/&/g, "and").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""),
    label,
    description: `${label} visual direction for MiniMax video generation.`,
    prompt_guidance: MINIMAX_VIDEO_STYLE_VERBIAGE[label],
  })),
  {
    value: "custom",
    label: "Custom — type exact wording",
    description: "Use custom visual-style wording exactly as entered in every eligible prompt.",
    prompt_guidance: "",
  },
];

function storyboardMiniMaxVideoStylePreset(value = "") {
  return MINIMAX_VIDEO_STYLE_PRESETS.find((item) => item.value === value) || MINIMAX_VIDEO_STYLE_PRESETS[0];
}

function storyboardMiniMaxVideoStyleVerbiage(value = "", custom = "") {
  const preset = storyboardMiniMaxVideoStylePreset(value);
  const direction = preset.value === "custom" ? String(custom || "").trim() : String(preset.prompt_guidance || "").trim();
  if (!direction) return "";
  return preset.value === "custom" ? direction : `${preset.label}: ${direction}`;
}

function storyboardSceneSupportsVideoStyle(scene = {}) {
  const engine = normalizeStoryboardProjectVideoEngine(scene.project_video_engine || scene.projectVideoEngine);
  if (engine === "ltx") return true;
  return ["text_to_video", "reference_to_video"].includes(normalizeStoryboardMiniMaxH3Mode(scene.minimax_h3_mode || scene.minimaxH3Mode));
}

export const MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS = [
  { value: "", label: "Off / natural time", description: "All people and the environment move in the same natural time." },
  { value: "realtime_subjects_timelapse_world", label: "Real-time characters / time-lapse world", description: "Mapped characters stay natural while anonymous extras, location activity, and optional light changes race around them.", prompt_guidance: "Create a clearly separated two-speed reality: protected characters remain in natural real time while only the unprotected background world moves in smooth accelerated time-lapse." },
  { value: "frozen_world", label: "Characters move / world frozen", description: "Protected characters move naturally through a world held almost perfectly still.", prompt_guidance: "Protected characters move naturally through a world frozen at one instant. Unprotected people, particles, vehicles, liquids, smoke, weather, and environmental motion remain suspended unless the scene explicitly releases one element." },
  { value: "reverse_world", label: "Characters forward / world reverses", description: "Protected characters continue normally while unprotected background action runs backward.", prompt_guidance: "Protected characters move and perform forward in natural time while unprotected background people and environmental events visibly run in reverse, including recoverable spills, retracing traffic, returning debris, and reversed weather or smoke." },
  { value: "day_night_sweep", label: "Real-time characters / day-to-night sweep", description: "Characters stay natural while daylight, shadows, windows, and practical lights rapidly change.", prompt_guidance: "Protected characters remain in natural time while the environment passes rapidly through a readable day-to-night or night-to-day cycle, with accelerated sky color, sunlight angle, shadow travel, window light, and practical lights switching on or off." },
  { value: "seasonal_passage", label: "Real-time characters / seasons pass", description: "The location visibly crosses seasons around stable real-time characters.", prompt_guidance: "Protected characters remain in natural time while the environment transitions through accelerated seasonal change: vegetation, weather, ground cover, atmospheric color, and daylight evolve coherently without changing the characters' identities or wardrobe unless explicitly requested." },
  { value: "crowd_flow", label: "Real-time characters / crowd river", description: "Anonymous extras stream around the referenced cast while the cast remains readable.", prompt_guidance: "Protected characters remain sharply readable in natural time while anonymous unreferenced extras flow around them as an accelerated crowd river, forming continuous directional streams without duplicating, replacing, or obscuring the protected cast." },
  { value: "looping_background", label: "Real-time characters / looping background", description: "Background actions repeat in visible cycles while the referenced cast continues normally.", prompt_guidance: "Protected characters continue naturally while unprotected background actions repeat in deliberate seamless temporal loops. Keep each loop spatially anchored and visually distinct from the protected characters' unrepeated performance." },
  { value: "delayed_world", label: "Characters lead / world echoes behind", description: "The environment responds a beat late, creating a temporal echo.", prompt_guidance: "Protected characters move in natural time while unprotected environmental reactions lag behind them in visible delayed echoes, as though the world responds one beat late. Preserve physical readability and avoid duplicating the protected characters." },
  { value: "living_shadows", label: "Real-time characters / living shadows", description: "Characters remain normal while unprotected shadows move independently or at accelerated speed.", prompt_guidance: "Protected characters remain in natural time while cast shadows and environmental shadows move independently at accelerated speed, changing direction and shape without changing the protected characters' bodies, faces, or identity." },
  { value: "reflection_delay", label: "Real-time characters / delayed reflections", description: "Mirrors and reflective surfaces lag behind the real-time cast.", prompt_guidance: "Protected characters move in natural time while their reflections and the environment's reflections respond with a deliberate temporal delay. Keep the real characters singular and stable; the delayed imagery exists only inside physically plausible reflective surfaces." },
  { value: "gravity_separation", label: "Real-time characters / altered-gravity world", description: "The cast stays grounded while loose environmental objects behave in surreal gravity.", prompt_guidance: "Protected characters remain grounded and move in natural time while unprotected loose objects, dust, fabric scraps, droplets, leaves, and environmental debris rise, fall, or drift under visibly altered gravity." },
  { value: "custom", label: "Custom temporal effect", description: "Use the user's exact temporal-effect wording while retaining the selected protection and extras rules." },
];

// FX are injected by the Builder into each timestamped shot after prompt
// generation. They are intentionally separate from camera flow and temporal
// world effects so the LLM does not have to invent, place, or repeat them.
export const STORYBOARD_FX_PRESETS = [
  { value: "", label: "Off", description: "Do not add builder-managed FX.", cues: [] },
  { value: "lighting", label: "Lighting FX", description: "Practical-light flicker, sweeps, pulses, exposure and color changes.", cues: ["A brief practical-light flicker cascade sweeps through the environment and resolves into clean subject light.", "A controlled red-to-blue light sweep travels across the set, creating a readable cinematic exposure pulse.", "Neon reflections pulse once across the surfaces on the musical accent, without obscuring the subject's face."] },
  { value: "camera_lens", label: "Camera / Lens FX", description: "Lens flare, bloom, parallax, focus breathing, motion blur and optical artifacts.", cues: ["A restrained lens flare and foreground-parallax accent passes through the frame as the camera moves.", "A subtle lens-bloom pulse and focus-breathing shift accent the camera move while the subject remains sharp.", "A controlled motion-blur streak catches the edge of the frame during the camera movement, preserving subject readability."] },
  { value: "glitch", label: "Glitch / Digital FX", description: "Scan lines, signal tearing, RGB separation, frame stutter and digital distortion.", cues: ["A restrained scan-glitch briefly tears across the background, leaving the subject's face and body stable.", "A short RGB-separation and signal-noise burst flickers at the musical accent, then clears completely.", "A controlled digital frame stutter affects the environment for a moment without duplicating or deforming the subject."] },
  { value: "atmospheric", label: "Atmospheric FX", description: "Fog, smoke, dust, ash, rain, sparks and environmental particles.", cues: ["A thin layer of atmospheric mist curls through the background, catching the existing light without hiding the subject.", "Small airborne particles drift through the light beam and briefly sparkle around the environment.", "A restrained veil of smoke and dust crosses the deeper background, preserving the foreground performance clearly."] },
  { value: "energy", label: "Energy / Impact FX", description: "Beat flashes, sparks, shockwave-like light and controlled energy accents.", cues: ["A compact beat-synchronized light burst radiates through the environment and quickly settles.", "A brief ring of sparks and reflected light marks the musical accent without touching or altering the subject.", "A controlled impact pulse ripples through loose environmental particles while the subject continues naturally."] },
  { value: "film_texture", label: "Film / Texture FX", description: "Film grain, halation, gate weave, light leaks and shutter trails.", cues: ["A subtle film-grain and halation texture becomes visible in the highlights for this shot.", "A restrained analog gate-weave and soft light leak add a brief tactile film accent.", "A short shutter-trail texture catches the brightest movement while keeping the composition and subject readable."] },
  { value: "distortion", label: "Distortion FX", description: "Fisheye warp, heat haze, ripples, refraction and reality-bending lens effects.", cues: ["A localized lens distortion gently bends the outer edges of the environment while the subject remains stable.", "A brief heat-haze ripple passes through the background, refracting light without changing the subject's identity.", "A controlled wide-angle warp accentuates the perspective at the frame edges, then returns to a clean image."] },
  { value: "supernatural", label: "Supernatural FX", description: "Living shadows, aura, reality fractures, floating debris and uncanny visual accents.", cues: ["The environment's shadows shift independently for a brief uncanny accent while the subject remains physically natural.", "A faint supernatural glow gathers in the background atmosphere and fades without changing the subject's face or clothing.", "A few loose environmental fragments hover briefly in the air, creating a controlled reality-fracture accent around the subject."] },
  { value: "music_video", label: "Music Video FX", description: "A curated rhythmic mix of lighting, lens and restrained post-production accents.", cues: ["A rhythmic neon light pulse combines with a restrained lens flare on the musical accent.", "A brief bloom and light-streak accent sweeps across the frame, then resolves into clean cinematic contrast.", "A controlled mix of atmospheric particles and subtle chromatic color separation marks the beat without obscuring the subject."] },
  { value: "custom", label: "Custom FX JSON", description: "Use the exact custom FX JSON wording and Builder placement rules." },
];

export function storyboardFxPreset(value = "") {
  return STORYBOARD_FX_PRESETS.find((item) => item.value === String(value || "")) || STORYBOARD_FX_PRESETS[0];
}

export function normalizeStoryboardCustomFxJson(input) {
  let source = input;
  if (typeof source === "string") {
    try { source = JSON.parse(source); } catch { return null; }
  }
  if (Array.isArray(source)) source = { cues: source };
  if (!source || typeof source !== "object") return null;
  const rawCues = Array.isArray(source.cues) ? source.cues : (Array.isArray(source.effects) ? source.effects : []);
  const cues = rawCues.map((item) => {
    if (typeof item === "string") return item.trim();
    if (item && typeof item === "object") return String(item.text || item.cue || item.description || "").trim();
    return "";
  }).filter(Boolean).slice(0, 12);
  const primary = String(source.primary || source.primary_effect || "").trim();
  const secondary = String(source.secondary || source.secondary_effect || "").trim();
  if (primary) cues.unshift(primary);
  if (secondary) cues.push(secondary);
  if (!cues.length) return null;
  return {
    label: String(source.label || source.name || "Custom FX").trim().slice(0, 120) || "Custom FX",
    cues: Array.from(new Set(cues)).slice(0, 12),
    timing: String(source.timing || "on the strongest musical or action accent").trim().slice(0, 240),
    intensity: Math.max(0, Math.min(10, Number(source.intensity ?? 6))) || 6,
    avoid: String(source.avoid || source.avoid_text || "Do not obscure, duplicate, deform, or alter the mapped subject.").trim().slice(0, 500),
  };
}

export function storyboardFxContract(value = "", customInput = "", shotIndex = 0) {
  const preset = storyboardFxPreset(value);
  const custom = value === "custom" ? normalizeStoryboardCustomFxJson(customInput) : null;
  const source = custom || preset;
  if (!source?.cues?.length) return null;
  const cue = source.cues[Math.max(0, Number(shotIndex) || 0) % source.cues.length];
  const timing = custom ? custom.timing : "on a readable musical, lyric, or action accent when one is present";
  const intensity = custom ? custom.intensity : 6;
  const avoid = custom ? custom.avoid : "Keep the mapped subject's face, identity, body, wardrobe, performance, and lip sync stable and readable.";
  return {
    label: source.label || preset.label,
    cue,
    timing,
    intensity,
    exact_verbiage: `FX placement — ${source.label || preset.label}: Insert this FX directly inside the timestamped shot after the camera direction and before the lyric or performance action. Use one clear ${intensity}/10 visual accent: ${cue} Timing: ${timing} ${avoid}`,
  };
}

function storyboardTemporalWorldEffectPreset(value = "") {
  return MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.find((item) => item.value === value) || MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS[0];
}

function storyboardTemporalProtectedMode(value = "") {
  return ["all_referenced", "lead_only", "custom"].includes(String(value || "")) ? String(value) : "all_referenced";
}

function storyboardTemporalIntensity(value = 8) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(10, Math.round(number))) : 8;
}

function storyboardTemporalLocationExamples(scene = {}) {
  const location = scene.location_ref || scene.locationRef || {};
  const context = [
    location.name,
    location.description,
    scene.setting,
    scene.location,
    scene.story_beat,
    scene.prompt_summary,
  ].map((value) => String(value || "").toLowerCase()).join(" ");
  if (/store|shop|market|liquor|grocery|retail|checkout/.test(context)) {
    return "customers and staff crossing aisles, checkout activity, shelf restocking, changing window light, and accelerated reflections";
  }
  if (/kitchen|dining|restaurant|cafe|bar|diner/.test(context)) {
    return "location-appropriate patrons or household activity, staff movement, changing practical light, drifting steam, and accelerated shadows";
  }
  if (/street|road|alley|sidewalk|parking|overpass|city|urban/.test(context)) {
    return "pedestrians, passing traffic, moving reflections, changing signs or practical lights, fast clouds, and traveling shadows";
  }
  if (/laundromat|laundry/.test(context)) {
    return "customers cycling through the room, spinning machines, baskets changing position, shifting fluorescent light, and window reflections";
  }
  if (/bedroom|apartment|living room|house|home|hallway|corridor|stair/.test(context)) {
    return "location-appropriate household or neighbor activity, rapidly shifting window light, traveling shadows, changing practical lights, weather, and moving reflections";
  }
  if (/forest|woods|field|garden|park|outdoor|beach|mountain/.test(context)) {
    return "location-appropriate passersby when permitted, fast clouds, traveling sunlight or moonlight, moving shadows, weather, vegetation, smoke, and airborne particles";
  }
  return "location-appropriate anonymous activity when permitted, changing light and shadows, weather, traffic or reflections, smoke, particles, and moving environmental details";
}

function storyboardTemporalWorldEffectForScene(scene = {}, state = {}) {
  const override = String(scene.temporal_world_effect_override || scene.temporalWorldEffectOverride || "global").trim();
  const globalKey = String(state.temporalWorldEffect || state.temporal_world_effect || "").trim();
  const key = override === "off" ? "" : (override && override !== "global" ? override : globalKey);
  const preset = storyboardTemporalWorldEffectPreset(key);
  const custom = String(
    key === "custom" && override && override !== "global"
      ? (scene.temporal_world_effect_custom || scene.temporalWorldEffectCustom || "")
      : (state.temporalWorldEffectCustom || state.temporal_world_effect_custom || ""),
  ).trim();
  const baseDirection = key === "custom" ? custom : String(preset.prompt_guidance || "").trim();
  if (!key || !baseDirection) return null;

  const protectedMode = storyboardTemporalProtectedMode(state.temporalProtectedCharacters || state.temporal_protected_characters);
  const protectedCustom = String(state.temporalProtectedCustom || state.temporal_protected_custom || "").trim();
  const protectedDirection = protectedMode === "lead_only"
    ? "Protect only the first mapped/reference character at natural 1x real-time speed; other mapped characters may receive the selected temporal effect."
    : protectedMode === "custom" && protectedCustom
      ? `Protect only these named mapped/reference characters at natural 1x real-time speed: ${protectedCustom}.`
      : "Protect every mapped/reference character in the scene at natural 1x real-time speed, including secondary referenced characters. Never accelerate, freeze, reverse, echo, duplicate, or temporally distort any protected character.";
  const allowExtras = state.temporalAllowBackgroundExtras !== false && state.temporal_allow_background_extras !== false;
  const extrasDirection = allowExtras
    ? "Anonymous unreferenced background extras are allowed. Infer only extras that naturally belong in the mapped location—for example customers or staff in a store, family or household activity in a kitchen, and pedestrians or traffic on a street. Keep them clearly secondary; never turn an extra into a principal character or duplicate a mapped/reference character."
    : "Do not add anonymous background people. Apply the temporal effect only to existing unprotected scene elements and the environment.";
  const environmentTimePassage = state.temporalEnvironmentTimePassage !== false && state.temporal_environment_time_passage !== false;
  const environmentDirection = environmentTimePassage
    ? "Environmental time passage is enabled: when appropriate, accelerate or transform daylight, shadows, practical lighting, weather, traffic, smoke, particles, and location activity while preserving spatial continuity."
    : "Do not add a day/night, lighting, weather, or seasonal time passage unless the scene notes explicitly request it.";
  const intensity = storyboardTemporalIntensity(state.temporalBackgroundIntensity ?? state.temporal_background_intensity ?? 8);
  const intensityDirection = intensity <= 3
    ? `Use a subtle ${intensity}/10 background-effect intensity with restrained, readable temporal separation.`
    : intensity <= 6
      ? `Use a clear ${intensity}/10 background-effect intensity that is immediately visible but does not overpower the protected characters.`
      : intensity <= 8
        ? `Use a strong ${intensity}/10 background-effect intensity with unmistakable temporal separation while keeping protected faces and actions readable.`
        : `Use an extreme ${intensity}/10 background-effect intensity with dramatic temporal contrast, while protected characters remain stable, singular, and readable.`;
  const audioDirection = "Temporal speed separation is visual only. Keep supplied or generated dialogue, singing, lip sync, facial timing, and primary audio at normal speed; never time-stretch, reverse, or accelerate protected voices.";
  const locationExamples = storyboardTemporalLocationExamples(scene);
  const cueCount = intensity >= 9 ? "at least two concrete, clearly visible effect cues" : "at least one concrete, clearly visible effect cue";
  const stagingRequirement = `TIMESTAMP STAGING REQUIREMENT: Every timestamp block must actively show ${cueCount} affecting only the unprotected background/world while protected characters continue at natural 1x speed. Use the mapped location to choose actions such as ${locationExamples}. At intensity 7 or higher, subtle flicker, ambience, drifting particles, or a vague mention of time passage alone does not satisfy this requirement. The temporal contrast must be immediately visible in the action itself. Any phrase such as “no people” inside a location-reference description describes only the source image and does not prohibit anonymous background extras when this contract permits them.`;
  const timestampAction = `Visibly enact this temporal layer at ${intensity}/10: ${baseDirection} Use concrete location-appropriate activity such as ${locationExamples}. Protected mapped/reference characters remain singular and move at natural 1x speed; only the permitted unprotected background/world receives the effect.`;
  const continuityRequirement = allowExtras
    ? "CONTINUITY RULE: Do not add any new named, principal, mapped, or referenced characters. Anonymous unreferenced background extras are explicitly permitted and must remain secondary and subject to the selected temporal effect; never prohibit them with a generic ‘no new characters’ or ‘no people’ rule."
    : "CONTINUITY RULE: Do not add new named, mapped, referenced, principal, or anonymous characters.";
  // This is a builder-owned contract. Gemma should not spend tokens recreating
  // temporal rules, and the final renderer prompt should receive one canonical
  // block instead of repeated per-shot instructions.
  const verbiage = `Temporal / World Effect — ${preset.label} — Mandatory:\n\n${baseDirection}\n\n${protectedDirection}\n\nOnly the unprotected background/world may receive this effect, including location-appropriate environmental motion or anonymous extras when permitted. Keep the effect secondary; never alter, duplicate, replace, or obscure a mapped/reference character.\n\nTimestamp staging requirement: every shot must visibly show at least one concrete, readable effect affecting only the unprotected background/world while protected characters continue moving naturally at normal 1x speed.\n\n${audioDirection}`;
  return {
    enabled: true,
    key,
    label: preset.label,
    exact_verbiage: verbiage,
    protected_characters: protectedMode,
    protected_custom: protectedCustom,
    allow_background_extras: allowExtras,
    background_intensity: intensity,
    environment_time_passage: environmentTimePassage,
    timestamp_staging_requirement: stagingRequirement,
    timestamp_action: timestampAction,
    continuity_requirement: continuityRequirement,
  };
}

export const ID_LORA_IMAGE_AESTHETIC_PRESETS = [
  { value: "film_default", label: "Default film still", description: "Balanced short-film still lighting, believable production design, natural texture, and cinematic composition.", prompt_guidance: "Build a polished short-film still, not a music-video still. Use believable character blocking, grounded wardrobe, practical lighting, lens/framing detail, textured production design, natural color contrast, and emotionally readable composition." },
  { value: "indie_film_naturalism", label: "Indie film naturalism", description: "Naturalistic indie-drama still with lived-in details, imperfect realism, and intimate character focus.", prompt_guidance: "Build an indie-film still with naturalistic lighting, lived-in wardrobe, imperfect textures, believable posture, intimate framing, subtle emotional detail, muted color response, and environment details that feel observed rather than staged." },
  { value: "neo_noir_dialogue", label: "Neo-noir dialogue", description: "Low-key shadows, practical neon, suspicious glances, dramatic contrast, and noir-style tension.", prompt_guidance: "Build a neo-noir dialogue still with low-key lighting, practical neon or sodium light, deep shadows, hard rim light, reflective surfaces, guarded facial expression, tense blocking, and a controlled color palette. Keep it cinematic and grounded." },
  { value: "gritty_punk_bar", label: "Gritty punk bar", description: "Worn bar textures, punk attitude, practical stage/neon light, smoky atmosphere, and analog grit.", prompt_guidance: "Build a gritty punk-bar film still with worn leather or denim styling, messy lived-in hair/makeup, scratched tables, stickers, posters, dim practical lights, colored neon spill, smoky air, visible texture, defiant posture, and a raw 35mm cinematic finish." },
  { value: "psychological_thriller", label: "Psychological thriller", description: "Uneasy framing, controlled color, negative space, tense facial detail, and subtle dread.", prompt_guidance: "Build a psychological-thriller still with uneasy composition, negative space, controlled color palette, tense facial detail, practical low light, slightly off-balance framing, foreground obstruction, and environmental details that imply pressure without explaining it." },
  { value: "warm_dialogue_drama", label: "Warm dialogue drama", description: "Warm practical interiors, soft skin tones, intimate framing, and emotionally readable acting.", prompt_guidance: "Build a warm dialogue-drama still with practical lamp or bar light, gentle skin tones, shallow depth of field, intimate framing, small emotional facial detail, believable wardrobe, textured surroundings, and a quiet cinematic finish." },
  { value: "35mm_analog_film", label: "35mm analog film", description: "Film grain, practical lighting, imperfect texture, grounded color, and documentary-like realism.", prompt_guidance: "Build a 35mm analog film still with visible grain, practical lighting, imperfect surfaces, grounded color response, natural posture, textured wardrobe, shallow lens character, and a lived-in environment. Avoid glossy music-video polish." },
];

export function storyboardImageShotFlowEntry(profileKey, sceneIndex) {
  const preset = STORYBOARD_IMAGE_SHOT_FLOW_PRESETS[profileKey] || STORYBOARD_IMAGE_SHOT_FLOW_PRESETS.intimate;
  const sequence = preset.sequence || [];
  if (!sequence.length) return "";
  return sequence[sceneIndex % sequence.length] || "";
}

export function storyboardImageAestheticPreset(value = "") {
  return STORYBOARD_IMAGE_AESTHETIC_PRESETS.find((item) => item.value === value) || STORYBOARD_IMAGE_AESTHETIC_PRESETS[0];
}

function storyboardImageAestheticGuidance(value = "", options = {}) {
  const presets = options.idLoraMode ? ID_LORA_IMAGE_AESTHETIC_PRESETS : STORYBOARD_IMAGE_AESTHETIC_PRESETS;
  const preset = presets.find((item) => item.value === value) || presets[0] || storyboardImageAestheticPreset(value);
  return preset.prompt_guidance || preset.description || "";
}

function referenceChipHtml(ref, fallbackLabel = "Reference") {
  const image = storyboardReferenceImageSrc(ref?.image);
  const label = String(ref?.name || fallbackLabel || "Reference").trim();
  const thumb = image
    ? `<span style="width:34px;height:34px;border-radius:6px;border:1px solid #334155;background:#0f172a url('${escapeHtml(image)}') center/cover no-repeat;flex:0 0 auto;"></span>`
    : `<span style="width:34px;height:34px;border-radius:6px;border:1px dashed #334155;background:#07111f;color:#67e8f9;display:grid;place-items:center;font-size:12px;flex:0 0 auto;">▣</span>`;
  return `<span title="${escapeHtml(label)}" style="display:inline-flex;align-items:center;gap:7px;max-width:190px;border:1px solid #334155;border-radius:7px;background:#0f172a;color:#e5e7eb;padding:4px 7px;margin:3px 3px 3px 0;vertical-align:middle;">${thumb}<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:800;">${escapeHtml(label)}</span></span>`;
}

function subjectRefsHtml(scene) {
  const refs = Array.isArray(scene.subject_refs) ? scene.subject_refs : [];
  if (refs.length) return refs.map((ref, index) => referenceChipHtml(ref, `Subject ${index + 1}`)).join("");
  return tagsHtml(scene.subjects);
}

function settingRefHtml(scene) {
  if (scene.location_ref && typeof scene.location_ref === "object" && String(scene.location_ref.name || scene.location_ref.image?.path || scene.location_ref.image?.data || "").trim()) {
    return referenceChipHtml(scene.location_ref, scene.setting || "Location");
  }
  return escapeHtml(scene.setting || "-");
}

function normalizeReferenceBuilderCatalog(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const mergeReferenceList = (items = []) => {
    const byKey = new Map();
    const keyFor = (item) => {
      const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
      return name || String(item.id || "").trim().toLowerCase();
    };
    for (const item of items) {
      const key = keyFor(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        id: existing.id || item.id,
        name: existing.name || item.name,
        description: existing.description || item.description,
        trigger_phrase: existing.trigger_phrase || item.trigger_phrase,
        image: mergeReferenceImages(existing.image, item.image),
      });
    }
    return Array.from(byKey.values());
  };
  const subjects = mergeReferenceList(Array.isArray(source.subjects) ? source.subjects
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `subject_${index + 1}`),
      name: String(item.name || `Character ${index + 1}`),
      description: String(item.description || ""),
      minimax_voice: item.minimax_voice && typeof item.minimax_voice === "object" ? { ...item.minimax_voice } : {},
      trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
      trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
      extra_reference_for: String(item.extra_reference_for || item.extraReferenceFor || item.same_subject_as || item.sameSubjectAs || ""),
      image: normalizeReferenceImage(item),
    })).filter((item) => !item.extra_reference_for) : []);
  const locations = mergeReferenceList(Array.isArray(source.locations) ? source.locations
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `location_${index + 1}`),
      name: String(item.name || `Location ${index + 1}`),
      description: String(item.description || ""),
      trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
      trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
      image: normalizeReferenceImage(item),
    })) : []);
  return {
    subjects,
    locations: Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations) ? [] : locations,
    locations_cleared: Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations),
    trigger_position: String(source.trigger_position || source.triggerPosition || source.trigger_placement || "start") === "end" ? "end" : "start",
    subject_trigger_position: String(source.subject_trigger_position || source.subjectTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start",
    location_trigger_position: String(source.location_trigger_position || source.locationTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start",
  };
}

function mergeReferenceBuilderCatalog(base = {}, incoming = {}) {
  const normalizedBase = normalizeReferenceBuilderCatalog(base);
  const normalizedIncoming = normalizeReferenceBuilderCatalog(incoming);
  const mergeList = (left, right) => {
    const byKey = new Map();
    const keyFor = (item) => {
      const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
      return name || String(item.id || "").trim().toLowerCase();
    };
    for (const item of left) {
      const key = keyFor(item);
      if (key) byKey.set(key, { ...item, image: { ...(item.image || {}) } });
    }
    for (const item of right) {
      const key = keyFor(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        image: mergeReferenceImages(existing.image, item.image),
      });
    }
    return Array.from(byKey.values());
  };
  return {
    subjects: mergeList(normalizedBase.subjects, normalizedIncoming.subjects),
    locations: normalizedBase.locations_cleared ? [] : mergeList(normalizedBase.locations, normalizedIncoming.locations),
    locations_cleared: Boolean(normalizedBase.locations_cleared || normalizedIncoming.locations_cleared),
  };
}

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

function normalizeStoryboardPerformanceMode(value = "") {
  const text = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["speaking", "short_film", "dialogue", "dialog"].includes(text)) return "speaking";
  if (["no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"].includes(text)) return "no_lip_sync";
  return "singing";
}

function normalizeStoryboardProjectVideoEngine(value = "") {
  return String(value || "").trim().toLowerCase() === "minimax_h3" ? "minimax_h3" : "ltx";
}

function normalizeStoryboardMiniMaxH3Mode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ["text_to_video", "image_to_video", "reference_to_video", "video_to_video"].includes(clean)
    ? clean
    : "text_to_video";
}

function normalizeStoryboardMiniMaxH3AudioMode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ["built_in_audio", "native_audio", "generated_audio"].includes(clean) ? "built_in_audio" : "input_audio";
}

function normalizeStoryboardShortFilmPlanningMode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return clean === "fully_custom" || clean === "custom" ? "fully_custom" : "guided_film";
}

function normalizeStoryboardSpeakerAssignments(value = []) {
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

function storyboardStillFacialDirection(value = "") {
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

function normalizeVideoPromptOrigin(value) {
  return String(value || "").trim().toLowerCase() === "gemma" ? "gemma" : "manual";
}

function normalizeScene(scene = {}, index = 0) {
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

function ensureStoryboardReferenceOpening(prompt, scene = {}, imageMode = "") {
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

function scenesFromBuilderPayload(payload = {}) {
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

function createToast(message, error = false) {
  const toast = document.createElement("div");
  toast.textContent = message;
  toast.style.cssText = `position:fixed;right:24px;bottom:24px;z-index:100020;max-width:520px;border:1px solid ${error ? "#991b1b" : "#155e75"};border-radius:8px;background:${error ? "#3f0808" : "#083344"};color:#f8fafc;padding:12px 14px;box-shadow:0 12px 40px rgba(0,0,0,.45);white-space:pre-wrap;font-size:13px;`;
  document.body.append(toast);
  setTimeout(() => toast.remove(), error ? 8500 : 4200);
}

function createStoryboardProgressWindow(title = "Storyboard LLM") {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.18);pointer-events:none;display:flex;align-items:flex-start;justify-content:center;padding-top:72px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(760px,calc(100vw - 48px));border:1px solid #0891b2;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 22px 70px rgba(0,0,0,.55);overflow:hidden;pointer-events:auto;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;background:#083f4f;border-bottom:1px solid #0891b2;";
  const titleEl = document.createElement("div");
  titleEl.textContent = title;
  titleEl.style.cssText = "font-weight:900;color:#cffafe;";
  const close = makeButton("Close");
  close.style.padding = "8px 12px";
  header.append(titleEl, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;";
  const message = document.createElement("div");
  message.style.cssText = "white-space:pre-wrap;line-height:1.45;font-size:13px;color:#e2e8f0;min-height:38px;";
  const track = document.createElement("div");
  track.style.cssText = "height:8px;border-radius:999px;background:#155e75;overflow:hidden;";
  const fill = document.createElement("div");
  fill.style.cssText = "height:100%;width:0%;background:#22d3ee;border-radius:999px;transition:width .18s ease;";
  track.append(fill);
  body.append(message, track);
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  close.onclick = () => backdrop.remove();
  return {
    set(text, percent = 0) {
      message.textContent = String(text || "");
      const pct = Number.isFinite(Number(percent)) ? Math.max(0, Math.min(100, Number(percent))) : 0;
      fill.style.width = `${pct}%`;
    },
    showDiagnostics(diagnostics = {}) {
      if (!diagnostics || typeof diagnostics !== "object") return;
      const details = document.createElement("details");
      details.style.cssText = "border:1px solid #475569;border-radius:7px;background:#111827;padding:9px 10px;color:#cbd5e1;font-size:12px;";
      const summary = document.createElement("summary");
      summary.textContent = "Show raw model output (diagnostics)";
      summary.style.cssText = "cursor:pointer;font-weight:800;color:#bae6fd;";
      const meta = document.createElement("div");
      meta.style.cssText = "margin-top:8px;line-height:1.45;white-space:pre-wrap;";
      const runner = String(diagnostics.runner || "LLM");
      const expected = Array.isArray(diagnostics.expected_sections) ? diagnostics.expected_sections.join(" → ") : "";
      meta.textContent = `Runner: ${runner}${expected ? `\nExpected sections: ${expected}` : ""}`;
      const rawLabel = document.createElement("div");
      rawLabel.textContent = "Raw response:";
      rawLabel.style.cssText = "margin-top:8px;font-weight:800;color:#fda4af;";
      const raw = document.createElement("pre");
      raw.textContent = String(diagnostics.raw_output || "[empty]");
      raw.style.cssText = "max-height:220px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 8px;padding:8px;background:#020617;border-radius:5px;color:#fecdd3;";
      const cleanedLabel = document.createElement("div");
      cleanedLabel.textContent = "Cleaned response:";
      cleanedLabel.style.cssText = "font-weight:800;color:#bae6fd;";
      const cleaned = document.createElement("pre");
      cleaned.textContent = String(diagnostics.cleaned_output || "[empty]");
      cleaned.style.cssText = "max-height:180px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 8px;padding:8px;background:#020617;border-radius:5px;color:#bae6fd;";
      const copy = makeButton("Copy output");
      copy.style.padding = "5px 9px";
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(String(diagnostics.raw_output || "")); copy.textContent = "Copied"; setTimeout(() => { copy.textContent = "Copy output"; }, 1200); }
        catch { copy.textContent = "Copy failed"; }
      };
      details.append(summary, meta, rawLabel, raw, cleanedLabel, cleaned, copy);
      body.insertBefore(details, track);
    },
    close(delay = 0) {
      if (delay > 0) {
        setTimeout(() => backdrop.remove(), delay);
      } else {
        backdrop.remove();
      }
    },
  };
}

function storyboardPayloadFromBuilder(payload = {}) {
  return {
    project_folder: payload.projectFolder || payload.project_folder || "",
    scenes: scenesFromBuilderPayload(payload),
  };
}

function slimReferenceForRequest(ref) {
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

function slimSceneForRequest(scene, index = 0) {
  const normalized = normalizeScene(scene, index);
  return {
    ...normalized,
    subject_refs: (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map(slimReferenceForRequest)
      .filter(Boolean),
    location_ref: slimReferenceForRequest(normalized.location_ref),
  };
}

function normalizeStoryLayer(value = {}) {
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

function storyboardSpeedValue(value, fallback = 4) {
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

function storyboardSpeedLabel(value, kind = "motion") {
  const speed = storyboardSpeedValue(value);
  if (speed <= 0) return kind === "camera" ? "0 / static camera" : "0 / still subject";
  if (speed <= 3) return `${speed} / subtle`;
  if (speed <= 6) return `${speed} / active`;
  if (speed <= 8) return `${speed} / energetic`;
  return `${speed} / fast action`;
}

function storyboardSpeedGuidance(value, kind = "motion") {
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

function storyboardCameraMotionForSpeed(value, speedValue) {
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

function slimStoryboardForRequest(state) {
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

const STORYBOARD_GPT_URL = "https://chatgpt.com/g/g-6a28d15f04e88191a2375d564ff8d90c-ltx-2-3-video-builder-from-storyboard-builder";
const STORYBOARD_IMAGE_GPT_URL = "https://chatgpt.com/g/g-6a40129fc12c81919878b79eaa5ae94f-text-to-image-prompt-builder-for-krea-2";
const STORY_LAYER_CHATGPT_URL = "https://chatgpt.com/";

function storyLayerGptPayload(state) {
  const storyboardPayload = storyboardGptPayload(state);
  const scenes = state.scenes.map((scene, index) => slimSceneForRequest(scene, index));
  const orderedLyrics = scenes
    .map((scene) => String(scene.lyrics || scene.lyric_text || "").trim())
    .filter(Boolean);
  const sourceLyrics = String(state.lineMappingLyrics || state.lyricMapper?.source_text || "").trim();
  const lyricSections = scenes.map((scene, index) => ({
    scene_number: Number(scene.scene_number || index + 1),
    section: String(scene.lyric_section || scene.label || "").trim(),
    lyrics: String(scene.lyrics || scene.lyric_text || "").trim(),
  })).filter((item) => item.section || item.lyrics);
  const presetDetails = {
    image_aesthetic: state.imageAesthetic || "",
    image_shot_flow: state.imageShotFlow || "",
    video_style: state.videoStyle || "",
    performance_style: state.performanceStyle || "",
    facial_performance: state.facialPerformance || "",
    camera_flow: state.cameraFlow || "",
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    lyric_story_strength: normalizeStoryLayer(state.storyLayer).lyric_story_strength,
  };
  return {
    payload_type: "story_layer_planning",
    execution_mode: "execute_immediately",
    user_request: "Process this story-layer planning payload now. Do not ask what I want done and do not ask for more lyrics or project details. Generate the final story-layer JSON response immediately.",
    task_instruction: "Use the supplied project context to create or revise a coherent music-video story. The explicit project_inputs fields are the source of truth. Execute this task immediately and return only valid JSON matching output_format. Keep the user's overall story idea when it is present, use the ordered lyric sections as the structural spine, and use the style, subject, location, performance, and camera preset details as creative constraints.",
    output_format: {
      overall_story_idea: "A concise premise or overall story idea. Preserve the supplied idea when it is usable; otherwise create one.",
      user_story_arc: "A section-by-section story arc using the supplied lyric section labels in order.",
      song_story_brief: "A compact production brief explaining the premise, emotional progression, visual world, recurring motifs, and ending."
    },
    current_story_layer: normalizeStoryLayer(state.storyLayer),
    project_inputs: {
      overall_story_idea: String(state.storyLayer?.overall_story_idea || "").trim(),
      ordered_lyrics: orderedLyrics,
      source_lyrics: sourceLyrics,
      lyrics_instruction: "When scenes are present, use ordered_lyrics for scene alignment. When ordered_lyrics is empty, use source_lyrics as the complete pasted song/script and create the lyric sections yourself from its headings and structure.",
      lyric_sections: lyricSections,
      subjects: (state.referenceBuilder?.subjects || []).map(slimReferenceForRequest).filter(Boolean),
      locations: (state.referenceBuilder?.locations || []).map(slimReferenceForRequest).filter(Boolean),
      preset_details: presetDetails,
      scenes,
    },
    project_context: storyboardPayload,
  };
}

function storyboardReferenceForGpt(ref, options = {}) {
  if (!ref) return null;
  const name = String(ref.name || "").trim();
  const description = String(ref.description || "").trim();
  const triggerPhrase = String(ref.trigger_phrase || ref.trigger || ref.Trigger || "").trim();
  const promptName = options.subject && triggerPhrase ? triggerPhrase : name;
  if (!promptName && !description) return null;
  return {
    name: promptName,
    display_name: name,
    description,
    trigger_phrase: triggerPhrase,
    prompt_name_source: options.subject && triggerPhrase ? "subject_trigger_phrase" : "reference_name",
  };
}

function storyboardVideoPromptTypeLabel(type) {
  const key = String(type || "").toLowerCase();
  if (key === "text_to_video") return "MiniMax H3 text to video";
  if (key === "image_to_video") return "MiniMax H3 image to video";
  if (key === "reference_to_video") return "MiniMax H3 reference to video";
  if (key === "video_to_video") return "MiniMax H3 video to video";
  if (key === "id_lora") return "ID-LoRA image to video";
  if (key === "ingredients") return "ingredients to video";
  if (key === "t2v") return "text to video";
  if (key === "rtv") return "reference to video";
  if (key === "flf") return "first / last frame video";
  if (key === "i2v") return "image to video";
  return key || "image to video";
}

function storyboardStartingShotInstruction(shotType) {
  const shot = String(shotType || "").trim();
  if (!shot) return "";
  if (shot.toLowerCase() === "eyes shot") {
    return "The literal first generated frame must already be an extreme close-up of the subject's eyes. Do not use a wider or farther-away lead-in. The selected camera motion must begin from that opening framing.";
  }
  return `The literal first generated frame must already be a ${shot}. Do not use a wider, farther-away, establishing, or full-body lead-in before reaching that framing. The selected camera motion must begin from that opening framing.`;
}

function storyboardLtxStartingFraming(shotType) {
  const shot = String(shotType || "").trim();
  if (!shot) return "";
  const movementClause = shot.match(/^(.*?),\s*(?:(?:then|before)\s+)?(?:slowly\s+)?(?:pulling|panning|tilting|sliding|tracking|orbiting|zooming|dollying|crane|moving|drifting)\b/i);
  return movementClause ? movementClause[1].trim() : shot;
}

function storyboardLtxEmbeddedCameraMotion(shotType) {
  const shot = String(shotType || "").trim();
  const movementClause = shot.match(/^.*?,\s*((?:(?:then|before)\s+)?(?:slowly\s+)?(?:pulling|panning|tilting|sliding|tracking|orbiting|zooming|dollying|crane|moving|drifting)\b.*)$/i);
  return movementClause ? movementClause[1].replace(/^(?:then|before)\s+/i, "").trim() : "";
}

function storyboardScenesForGpt(state) {
  const imageMode = state.mode !== "image_to_video_prep";
  const idLoraMode = String(state.videoPromptType || state.video_prompt_type || "").trim() === "id_lora"
    || state.scenes.some((scene) => String(scene?.video_prompt_type || "").trim() === "id_lora");
  const miniMaxShortFilmMode = normalizeStoryboardProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
    && normalizeStoryboardPerformanceMode(state.performanceMode || state.performance_mode) === "speaking";
  const filmPlanningProfile = idLoraMode || miniMaxShortFilmMode;
  const fullyCustomShortFilm = miniMaxShortFilmMode
    && normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode) === "fully_custom";
  const performancePresets = filmPlanningProfile ? ID_LORA_PERFORMANCE_STYLE_PRESETS : PERFORMANCE_STYLE_PRESETS;
  const facialPresets = filmPlanningProfile ? ID_LORA_FACIAL_PERFORMANCE_PRESETS : FACIAL_PERFORMANCE_PRESETS;
  const performancePreset = (value = "") => performancePresets.find((item) => item.value === value) || performancePresets[0] || PERFORMANCE_STYLE_PRESETS[0];
  const facialPresetForPayload = (value = "") => facialPresets.find((item) => item.value === value) || facialPresets[0] || FACIAL_PERFORMANCE_PRESETS[0];
  const cameraFlowKey = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] ? state.cameraFlow : "balanced";
  const cameraFlowPreset = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowKey];
  const explicitLyricSections = state.scenes.map((scene, index) => String(normalizeScene(scene, index).lyric_section || "").trim());
  const effectiveLyricSection = (index) => {
    if (explicitLyricSections[index]) return explicitLyricSections[index];
    for (let next = index + 1; next < explicitLyricSections.length; next += 1) {
      if (explicitLyricSections[next]) return explicitLyricSections[next];
    }
    for (let previous = index - 1; previous >= 0; previous -= 1) {
      if (explicitLyricSections[previous]) return explicitLyricSections[previous];
    }
    return "";
  };
  let previousCameraMotion = "";
  return state.scenes.map((scene, index) => {
    const normalized = normalizeScene(scene, index);
    const sceneVideoEngine = normalizeStoryboardProjectVideoEngine(normalized.project_video_engine || state.projectVideoEngine);
    const lyricSection = effectiveLyricSection(index);
    if (!explicitLyricSections[index] && lyricSection && scene && typeof scene === "object") {
      scene.lyric_section = lyricSection;
    }
    const sceneNumberIndex = Math.max(0, Number(normalized.scene_number || index + 1) - 1);
    const cameraFallback = fullyCustomShortFilm ? null : storyboardCameraFlowEntry(state.cameraFlow || "balanced", sceneNumberIndex, previousCameraMotion, state.customCameraFlowSequence);
    const shotType = normalized.shot_type || cameraFallback?.shot || "";
    const promptShotType = sceneVideoEngine === "ltx" ? storyboardLtxStartingFraming(shotType) : shotType;
    const requiresStartingShot = !imageMode && normalized.video_prompt_type !== "i2v" && Boolean(promptShotType);
    const embeddedLtxCameraMotion = sceneVideoEngine === "ltx" ? storyboardLtxEmbeddedCameraMotion(shotType) : "";
    const rawCameraMotion = normalized.camera_motion || (imageMode ? "" : embeddedLtxCameraMotion || cameraFallback?.camera) || "";
    const cameraMotion = imageMode || fullyCustomShortFilm
      ? rawCameraMotion
      : storyboardCameraMotionForSpeed(rawCameraMotion, state.cameraMotionSpeed);
    const motionSummary = String(normalized.motion_summary || "").trim();
    const cameraMotionForPrompt = motionSummary ? "" : cameraMotion;
    if (!imageMode) previousCameraMotion = cameraMotion || previousCameraMotion;
    const lyricText = String(normalized.lyrics || "").trim();
    const performanceMode = normalizeStoryboardPerformanceMode(normalized.performance_mode || state.performanceMode || state.videoType || state.performance_mode);
    const selectedFacialPerformance = normalized.facial_performance || (fullyCustomShortFilm ? "" : state.facialPerformance);
    const facialPreset = facialPresetForPayload(selectedFacialPerformance);
    const facialCustom = String(normalized.facial_performance_custom || state.facialPerformanceCustom || "").trim();
    const facialDirection = selectedFacialPerformance === "off"
      ? ""
      : selectedFacialPerformance === "custom" && facialCustom
      ? facialCustom
      : [facialPreset.direction, facialCustom].filter(Boolean).join(" ");
    const selectedPerformanceStyle = normalized.performance_style || (fullyCustomShortFilm ? "" : state.performanceStyle);
    const selectedPerformancePreset = performancePreset(selectedPerformanceStyle);
    const supportsVideoStyle = storyboardSceneSupportsVideoStyle(normalized);
    const selectedVideoStyle = supportsVideoStyle ? String(state.videoStyle || normalized.video_style || "") : "";
    const selectedVideoStyleCustom = selectedVideoStyle === "custom"
      ? String(state.videoStyle === "custom" ? state.videoStyleCustom : (normalized.video_style_custom || state.videoStyleCustom || "")).trim()
      : "";
    const selectedVideoStylePreset = storyboardMiniMaxVideoStylePreset(selectedVideoStyle);
    const selectedVideoStyleVerbiage = storyboardMiniMaxVideoStyleVerbiage(selectedVideoStyle, selectedVideoStyleCustom);
    const temporalWorldEffect = !imageMode ? storyboardTemporalWorldEffectForScene(normalized, state) : null;
    const exactSceneDuration = Math.max(
      0,
      Number(normalized.exact_duration || 0) || Number(normalized.timeline_end || 0) - Number(normalized.timeline_start || 0),
    );
    const cutPlan = !imageMode
      ? storyboardCutPlanForDuration(exactSceneDuration, state.cutFrequency, sceneVideoEngine)
      : null;
    const instrumental = Boolean(normalized.lyric_instrumental);
    const noLipSync = Boolean(normalized.lyric_no_lip_sync || performanceMode === "no_lip_sync");
    const noCharacterPresent = Boolean(normalized.no_character_present);
    const shouldLipSync = !imageMode && performanceMode !== "no_lip_sync" && Boolean(lyricText) && !instrumental && !noLipSync && !noCharacterPresent;
    const subjectRefs = noCharacterPresent ? [] : (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : [])
      .map((ref) => storyboardReferenceForGpt(ref, { subject: true }))
      .filter(Boolean);
    const subjectFallbacks = noCharacterPresent ? [] : (Array.isArray(normalized.subjects) ? normalized.subjects : [])
      .map((name) => ({ name: String(name || "").trim(), description: "" }))
      .filter((item) => item.name);
    const subjectNames = subjectRefs.length
      ? subjectRefs.map((subject) => subject.name).filter(Boolean)
      : subjectFallbacks.map((subject) => subject.name).filter(Boolean);
    const subjectCount = subjectRefs.length || subjectFallbacks.length;
    const subjectPromptNameByLabel = new Map(
      subjectRefs
        .map((subject) => [String(subject.display_name || subject.name || "").trim().toLowerCase(), subject.name])
        .filter(([label, promptName]) => label && promptName),
    );
    const explicitSingers = (Array.isArray(normalized.lyric_singers) ? normalized.lyric_singers : [])
      .map((name) => String(name || "").trim())
      .map((name) => subjectPromptNameByLabel.get(name.toLowerCase()) || name)
      .filter(Boolean);
    const singers = shouldLipSync ? (explicitSingers.length ? explicitSingers : subjectNames) : [];
    const singerKeySet = new Set(singers.map((name) => String(name || "").trim().toLowerCase()));
    const nonSingingSubjects = shouldLipSync
      ? subjectNames.filter((name) => !singerKeySet.has(String(name || "").trim().toLowerCase()))
      : subjectNames;
    const locationRef = storyboardReferenceForGpt(normalized.location_ref);
    return {
      scene_number: normalized.scene_number,
      label: normalized.label,
      lyric_section: lyricSection,
      prompt_type: imageMode ? "text to image" : storyboardVideoPromptTypeLabel(normalized.video_prompt_type),
      project_video_engine: normalizeStoryboardProjectVideoEngine(normalized.project_video_engine || state.projectVideoEngine),
      minimax_h3_mode: normalizeStoryboardMiniMaxH3Mode(normalized.minimax_h3_mode),
      ...(cutPlan ? {
        minimax_h3_cut_frequency: cutPlan.frequency,
        cut_plan: cutPlan,
      } : {}),
      exact_duration: Number(normalized.exact_duration || 0),
      timeline_start: Number(normalized.timeline_start || 0),
      timeline_end: Number(normalized.timeline_end || 0),
      performance_mode: performanceMode,
      short_film_planning_mode: miniMaxShortFilmMode ? normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode) : "",
      manual_scene_contract: fullyCustomShortFilm
        ? "Every populated scene-card field is authoritative. Format the supplied material only. Do not invent, rewrite, reorder, merge, omit, or replace dialogue, speakers, actions, story beats, shot/framing, camera motion, setting, references, sound direction, or continuity. Leave unspecified details unspecified instead of filling them in."
        : "",
      lyric_line_to_sing: shouldLipSync && performanceMode === "singing" ? lyricText : "",
      line_to_say: shouldLipSync && performanceMode === "speaking" ? lyricText : "",
      // Preserve the mapped performer assignment even in Image Prep. Image
      // prompts intentionally disable lip-sync, but Scene Beat generation
      // still needs to know which visible subject is the singer.
      lyric_singers: explicitSingers,
      performer_assignment: {
        singing: explicitSingers,
        silent: nonSingingSubjects,
        instruction: explicitSingers.length === 1
          ? `${explicitSingers[0]} is the only singing performer. Every other visible subject is silent and must not sing.`
          : explicitSingers.length > 1
            ? `Only these mapped performers sing: ${explicitSingers.join(", ")}. Every other visible subject is silent and must not sing.`
            : "No mapped subject is assigned to sing. All visible subjects are silent.",
      },
      vocal_status: {
        performance_mode: performanceMode,
        lyric_text: lyricText,
        lyric_section: lyricSection,
        singers,
        instrumental,
        no_lip_sync: noLipSync,
        should_lip_sync: shouldLipSync,
        no_character_present: noCharacterPresent,
        lyric_cue_map: normalized.lyric_cue_map,
        lyric_shot_word_timing_enabled: normalized.lyric_shot_word_timing_enabled,
        lyric_performance_mode: normalized.lyric_performance_mode,
        timed_lyric_cue_contract: normalized.timed_lyric_cue_contract,
      },
      vocal_direction: {
        mode: imageMode
          ? "still image / no singing"
          : performanceMode === "speaking" && shouldLipSync
            ? "say exact dialogue line"
            : performanceMode === "no_lip_sync"
              ? "visual only / no lip sync"
              : (shouldLipSync ? "sing exact lyric line" : (instrumental ? "instrumental / no vocals" : (noLipSync ? "b-roll / no lip sync" : "no lyric line provided"))),
        lyric_line: lyricText,
        singers,
        non_singing_visible_subjects: nonSingingSubjects,
        instruction: imageMode
          ? "This is a text-to-image still prompt, not a video or lip-sync prompt. Use lyric_line only for mood, symbolism, emotion, and visual direction. Do not mention singing, lip-syncing, performing vocals, singing the line, mouth movement, blinking, eye movement, or animation. The subject can hold a natural still pose, show a clear expression, or appear in a fashion/editorial scene, but should not be described as singing unless the scene notes explicitly ask for a live singing still."
          : performanceMode === "speaking" && shouldLipSync
            ? "Treat lyric_line as dialogue being said. The listed singer(s) field means the visible speaker(s). Use only wording like 'as she says \"...\"', 'as he says \"...\"', or 'as [subject name] says \"...\"'. Do not use alternate verbs for the dialogue line; use says only. Never use singing, rapping, music, lyric, vocal, or performance wording for speaking mode. Every non_singing_visible_subjects entry must still appear in the scene as visible non-speaking subjects who react, watch, move, or share the moment silently. Do not describe mouth shapes or mouth position."
            : performanceMode === "no_lip_sync"
              ? "Visual-only scene. Do not quote lyric_line. Do not mention saying, speaking, dialogue, singing, rapping, lyrics, vocals, lip-syncing, mouth movement, or no-vocal status. Use lyric_line only as hidden mood or story context."
              : shouldLipSync
                ? "Treat lyric_line as words being sung, not as literal scene action. The listed singer(s) should visibly sing this line with expressive facial emotion, gestures, performance energy, and facial performance guidance when provided. In the singer face sentence, include subtle natural eye movement and occasional natural blinking beside the eyes/brows/gaze description; do not append blinking or eye movement to an environment sentence. Every non_singing_visible_subjects entry must still appear in the scene as a visible non-singing subject who reacts, watches, moves, or shares the moment without singing. Use mouth-shape or jaw/lip wording only for the listed singer(s), never for non-singing subjects."
                + " Do not describe visible singing as quiet; use controlled, focused, intimate, restrained, inward, tender, or simmering intensity instead."
                : "Do not mention singing, lip-syncing, mouth movement, or vocal performance for this scene. Every listed subject must still appear as a visible non-singing subject unless no_character_present is true.",
      },
      scene_summary: imageMode ? "" : normalized.prompt_summary,
      story_layer: {
        lyric_section: lyricSection,
        scene_story_beat: normalized.story_beat,
        flf_start_state: normalized.flf_start_state,
        flf_transformation: normalized.flf_transformation,
        flf_end_state: normalized.flf_end_state,
        flf_carry_forward: normalized.flf_carry_forward,
        song_story_brief: state.storyLayer?.enabled === false ? "" : String(state.storyLayer?.song_story_brief || ""),
        user_story_arc: state.storyLayer?.enabled === false ? "" : String(state.storyLayer?.user_story_arc || ""),
        lyric_story_strength: normalizeStoryLayer(state.storyLayer).lyric_story_strength,
        instruction: "Use the story brief and scene story beat as narrative guidance. Lyric story strength controls how literally to follow lyric_line: 0 ignores lyrics, 1-3 uses mood only, 4-6 balances lyrics with story, 7-8 strongly follows lyric meaning, and 9-10 uses concrete lyric objects/actions/emotions whenever possible. Do not turn the prompt into plot exposition.",
      },
      motion_summary: imageMode ? "" : motionSummary,
      still_image_notes: imageMode ? motionSummary : "",
      image_aesthetic: imageMode ? storyboardImageAestheticGuidance(state.imageAesthetic, { idLoraMode: filmPlanningProfile }) : "",
      image_aesthetic_instruction: imageMode
        ? "Translate the selected image aesthetic into concrete prompt details: pose, wardrobe styling, hair, makeup, accessories, lighting setup, lens/framing, composition, environment treatment, texture, weather/time if useful, and art direction. Do not merely name the preset or append it as a short tag."
        : "",
      video_style: !imageMode && supportsVideoStyle && selectedVideoStyle ? selectedVideoStylePreset.label : "",
      video_style_custom: !imageMode && supportsVideoStyle && selectedVideoStyle === "custom" ? selectedVideoStyleCustom : "",
      video_style_guidance: !imageMode && supportsVideoStyle ? selectedVideoStyleVerbiage : "",
      video_style_verbiage: !imageMode && supportsVideoStyle ? selectedVideoStyleVerbiage : "",
      video_style_instruction: !imageMode && supportsVideoStyle && selectedVideoStyleVerbiage
        ? "This exact video_style_verbiage is mandatory. Copy it word-for-word into the final prompt and use it only as the governing visual-appearance contract for lighting, color, texture, materials, production design, grading, and image finish. Do not paraphrase, shorten, rename, or omit it. Do not use it to select, replace, or modify camera motion, character motion, shot timing, editing, or transitions."
        : "",
      temporal_world_effect: temporalWorldEffect || { enabled: false },
      temporal_world_effect_verbiage: temporalWorldEffect?.exact_verbiage || "",
      temporal_world_effect_instruction: temporalWorldEffect
        ? "This exact temporal_world_effect_verbiage is mandatory and will be appended by the builder. Do not recreate, paraphrase, duplicate, or distribute this contract through the shot descriptions. Write only the creative shot action; the appended contract governs temporal behavior."
        : "",
      global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
      global_consistency_instruction: String(state.globalConsistencyPhrase || "").trim()
        ? "Incorporate the global_consistency_phrase naturally into the prompt where it fits. Preserve its key wording, but do not force it to the beginning unless that is the most natural phrasing."
        : "",
      performance_style: !selectedPerformanceStyle || selectedPerformanceStyle === "off" ? "" : selectedPerformancePreset.label,
      performance_direction: !selectedPerformanceStyle || selectedPerformanceStyle === "off" ? "" : selectedPerformancePreset.direction,
      facial_performance: !selectedFacialPerformance || selectedFacialPerformance === "off" ? "" : facialPreset.label,
      facial_performance_direction: !selectedFacialPerformance ? "" : (imageMode ? storyboardStillFacialDirection(facialDirection) : facialDirection),
      facial_performance_custom: !selectedFacialPerformance || selectedFacialPerformance === "off" ? "" : (imageMode ? storyboardStillFacialDirection(facialCustom) : facialCustom),
      microphone: {
        include: Boolean(normalized.include_microphone),
        instruction: normalized.include_microphone
          ? "A microphone may be included if it naturally fits the scene, stage, or performance setup."
          : "Do not mention or add a microphone, mic stand, headset mic, studio mic, or any microphone prop unless the scene notes explicitly ask for one.",
      },
      subject_count: subjectCount,
      subject_instruction: noCharacterPresent
        ? (imageMode
          ? "No main character or mapped subject is present in this scene. Do not include, mention, imply, or describe the mapped character/singer/subject. Use the location, props, environment, objects, atmosphere, and still-image composition instead."
          : "No main character or mapped subject is present in this scene. Do not include, mention, imply, or describe the mapped character/singer/subject. Use the location, props, environment, objects, atmosphere, and camera motion instead.")
        : subjectCount === 1
        ? "This scene has exactly one mapped subject. Use the exact visible_subjects name/phrase as the subject phrase in the prompt. If that phrase came from a subject trigger_phrase, treat it as the subject identity, e.g. 'a photo of TRIGGER_PHRASE' instead of 'a photo of a woman'. Do not rewrite it as 'one woman', 'a woman', 'one man', 'a man', or any generic count phrase. Treat that exact subject phrase as one individual person and do not create duplicates, groups, backup singers, or multiple versions of the subject."
        : "This scene has multiple mapped subjects. Every listed subject must be visibly present in the prompt. Use each exact visible_subjects name/phrase when referring to them. If a phrase came from a subject trigger_phrase, treat it as that subject's identity. Do not drop any listed subject, rename them, or replace them with generic count phrases. Only the names in vocal_status.singers should sing; the other listed subjects should be visible but not singing. Do not add extra people unless the scene notes explicitly ask for them.",
      subject_name_rule: "Preserve mapped subject prompt names exactly as provided in visible_subjects and subjects.name. For subjects with trigger_phrase, subjects.name is already the prompt-facing trigger phrase and must be used as the subject instead of generic wording like 'a woman' or 'a man'.",
      visible_subjects: subjectNames,
      subjects: subjectRefs.length ? subjectRefs : subjectFallbacks,
      extra_subjects: normalized.extra_subjects,
      extra_subject_instruction: normalized.extra_subjects.length
        ? "Use every mapped extra in the scene's action and blocking according to interaction. Describe direct, dancing_with, and alongside roles individually; extras sharing background or background_dancing roles may be grouped by name. Keep identity wording brief and use it only to distinguish people. Do not assign singing, dialogue, or speaker IDs to extras unless explicitly supplied elsewhere."
        : "No mapped extras are assigned to this scene.",
      setting: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      location_ref: locationRef || {
        name: String(normalized.setting || "").trim(),
        description: String(normalized.setting || "").trim(),
      },
      camera_flow: cameraFlowKey,
      camera_flow_guidance: String(cameraFlowPreset?.guidance || "").trim(),
      shot_type: promptShotType,
      starting_shot: requiresStartingShot
        ? {
            required: true,
            selected_starting_shot: promptShotType,
            instruction: storyboardStartingShotInstruction(promptShotType),
          }
        : null,
      camera_motion: imageMode ? "" : cameraMotionForPrompt,
      still_camera_style: imageMode ? cameraMotion : "",
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      camera_motion_speed_guidance: imageMode || (fullyCustomShortFilm && !cameraMotionForPrompt) ? "" : storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      camera_guidance: imageMode
        ? {
            selected_still_camera_style: cameraMotion,
            instruction: "Use this as still photography composition, lens, lighting, or framing guidance only. Do not turn it into camera movement.",
          }
        : {
            selected_camera_motion: cameraMotionForPrompt,
            camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
            camera_motion_speed_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
            avoid_default_inward_moves: true,
            instruction: motionSummary
              ? "The custom motion_summary is authoritative. Do not add or reuse the scene-default camera motion preset."
              : "Use the selected camera motion as written. Do not add zoom-in, push-in, dolly-in, crash-zoom, or a close-up ending unless that exact inward motion is selected or requested in notes.",
          },
      character_motion: imageMode ? "" : normalized.character_motion,
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      character_motion_guidance: fullyCustomShortFilm && !normalized.character_motion ? "" : storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
      first_frame_visual_inventory: imageMode
        ? ""
        : {
            source: "text_to_image_prompt",
            text: normalized.image_prompt,
            instruction: "Use only for visible first-frame inventory: subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition. Do not use this field for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing.",
          },
      text_to_image_prompt: imageMode ? normalized.image_prompt : "",
      video_prompt: normalized.video_prompt,
      notes: normalized.notes,
      audio_direction: normalized.audio_direction,
      continuity: normalized.continuity,
    };
  });
}

export function storyboardGptPayload(state, scenesOverride = null) {
  const payloadState = scenesOverride ? { ...state, scenes: scenesOverride } : state;
  const selectedScene = scenesOverride?.length === 1 ? normalizeScene(scenesOverride[0], 0) : null;
  const imageMode = state.mode !== "image_to_video_prep";
  const selectedImageMode = String(state.imageMode || state.image_mode || "zimage").trim() || "zimage";
  const selectedImageModeLabel = String(state.imageModeLabel || state.image_mode_label || selectedImageMode).trim() || selectedImageMode;
  const imagePromptTarget = selectedImageMode === "flow_gpt"
    ? "Flow/GPT browser image prompt"
    : selectedImageMode === "nano_banana"
      ? "NanoBanana image prompt"
      : `${selectedImageModeLabel} image prompt`;
  return {
    scope: selectedScene ? "single_scene" : "all_scenes",
    selected_scene_number: selectedScene ? selectedScene.scene_number : null,
    performance_mode: normalizeStoryboardPerformanceMode(selectedScene?.performance_mode || state.performanceMode || state.videoType || state.performance_mode),
    short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
    storyboard_mode: state.mode === "image_to_video_prep" ? "video prompt planning" : "text-to-image prompt planning",
    image_model_mode: selectedImageMode,
    image_model_label: selectedImageModeLabel,
    image_prompt_target: imagePromptTarget,
    ...(imageMode
      ? {
        task_instruction: `Create detailed ${imagePromptTarget}s for Image Prep using advanced Krea 2-style still-image prompting. These are still-image prompts, not video or lip-sync prompts. Use lyrics and story beats for mood, symbolism, emotion, styling, and scene direction only. The mapped location_ref is the required physical set for each scene: do not replace it with a location from story_layer, scene_story_beat, song_story_brief, user_story_arc, or lyrics. If story context mentions another place, translate only its emotion, symbolism, pose, or action into the mapped location_ref environment. Do not say the subject is singing, lip-syncing, performing vocals, or singing the lyric unless the scene notes explicitly ask for a live singing image. Preserve mapped subject prompt names exactly as provided in each scene's visible_subjects and subjects.name. When a subject has a trigger_phrase, that trigger phrase is the subject identity for prompt wording, so write natural phrases like 'a photo of TRIGGER_PHRASE' instead of 'a photo of a woman'. Do not rename 'the woman' as 'one woman' or 'a woman', and do not rename trigger phrases. If global_consistency_phrase is present, weave it naturally into the prompt where it fits instead of slapping it onto the front.`,
        output_format: {
          type: "image_prompt_import_json",
          instruction: "Return only a JSON code block with an array of objects. Include every scene. Each object must have scene_number and image_prompt. Do not include prose outside the JSON code block.",
          example: [
            { scene_number: 1, image_prompt: `Full detailed ${imagePromptTarget} for scene 1...` },
            { scene_number: 2, image_prompt: `Full detailed ${imagePromptTarget} for scene 2...` },
          ],
        },
      }
      : {
        task_instruction: "Create detailed image-to-video prompts for Video Prep using a strict source hierarchy. The mapped location_ref is the required physical set for each scene: do not replace it with a location from story_layer, scene_story_beat, song_story_brief, user_story_arc, lyrics, or previous/next scene context. If story context mentions another place, translate only its emotion, tension, symbolism, or action into the mapped location_ref environment. The first_frame_visual_inventory field is only a first-frame inventory: visible subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition. Do not use first_frame_visual_inventory or any image prompt wording for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing. Follow camera_flow_guidance as a hard framing constraint for the entire shot, every camera move, and the ending composition. Follow cut_plan.instruction exactly for every video engine. MiniMax uses its timestamped CUT TO structure. LTX uses ordinary chronological language such as 'then cut to' and must not use the MiniMax timestamp schema. A zero/effectively-zero plan is one continuous take with no cuts. When starting_shot.required is true, the first sentence must explicitly state that the video begins with starting_shot.selected_starting_shot; do not merely imply that framing or use it later. For an eyes shot, explicitly say the video begins with an extreme close-up of the subject's eyes. The selected camera motion begins from that opening framing. Then build the rest of the video prompt in this order: 1) subject and vocal/performance sentence from vocal_status, performance_direction, and facial_performance_direction; 2) character movement sentence from character_motion, character_motion_guidance, character_motion_speed, and scene_story_beat; 3) camera movement sentence from camera_motion, camera_guidance, and camera_motion_speed_guidance; 4) environment/lighting sentence from first_frame_visual_inventory and location_ref; 5) final mood/style sentence from story_layer and image aesthetic only where visual. Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across multiple sentences. If an idea appears in the face sentence, do not repeat it in the body, camera, environment, or atmosphere sentence; use a different concrete visual detail instead. Do not duplicate adjacent words such as 'tall, tall'. The motion priority is character_motion_guidance + camera_motion_speed_guidance + camera_guidance + performance_direction + vocal_status + scene_story_beat above story_layer, and all of those above first_frame_visual_inventory. At camera speed 7-8, do not use slow, gentle, subtle, restrained, locked-off, static, or hold camera wording; use energetic active movement. At camera speed 9-10, use multiple coordinated readable camera moves. At character speed 4 or higher, include at least one clear physical body action, gesture, step, or set interaction; facial movement alone does not count.",
      }),
    story_layer: normalizeStoryLayer(state.storyLayer),
    scenes: storyboardScenesForGpt(payloadState),
  };
}

function openStoryboardGptUrl(payload) {
  const isImagePayload = String(payload?.storyboard_mode || "").toLowerCase().includes("text-to-image")
    || String(payload?.scenes?.[0]?.prompt_type || "").toLowerCase().includes("text to image");
  window.open(isImagePayload ? STORYBOARD_IMAGE_GPT_URL : STORYBOARD_GPT_URL, "_blank", "noopener,noreferrer");
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.cssText = "position:fixed;left:-9999px;top:-9999px;";
  document.body.append(textarea);
  textarea.focus();
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function openStoryboardBuilder(payload = {}) {
  const projectFolder = String(payload.projectFolder || payload.project_folder || "").trim();
  const incomingProjectVideoEngine = String(payload.projectVideoEngine || payload.project_video_engine || "").trim();
  const hasIncomingProjectVideoEngine = Boolean(incomingProjectVideoEngine);
  const projectVideoEngine = normalizeStoryboardProjectVideoEngine(incomingProjectVideoEngine);
  const payloadMiniMaxH3Mode = projectVideoEngine === "minimax_h3"
    ? normalizeStoryboardMiniMaxH3Mode(payload.miniMaxH3Mode || payload.minimax_h3_mode || payload.videoPromptType || payload.video_prompt_type)
    : "";
  const payloadMiniMaxH3AudioMode = projectVideoEngine === "minimax_h3"
    ? normalizeStoryboardMiniMaxH3AudioMode(payload.miniMaxH3AudioMode || payload.minimax_h3_audio_mode)
    : "input_audio";
  const payloadVideoPromptType = ["i2v", "id_lora", "t2v", "rtv", "ingredients", "flf"].includes(String(payload.videoPromptType || payload.video_prompt_type || "").trim())
    ? String(payload.videoPromptType || payload.video_prompt_type || "").trim()
    : "";
  const isIdLoraMode = payloadVideoPromptType === "id_lora";
  const payloadPerformanceMode = normalizeStoryboardPerformanceMode(payload.performanceMode || payload.performance_mode || payload.videoType || payload.video_type);
  const isMiniMaxShortFilmMode = projectVideoEngine === "minimax_h3" && payloadPerformanceMode === "speaking";
  const usesFilmPlanningProfile = isIdLoraMode || isMiniMaxShortFilmMode;
  const openingMode = projectVideoEngine === "minimax_h3"
    ? (payloadMiniMaxH3Mode === "image_to_video" ? "storyboard_prompts" : "image_to_video_prep")
    : (payloadVideoPromptType === "i2v" ? "storyboard_prompts" : "image_to_video_prep");
  const state = {
    projectFolder,
    projectVideoEngine,
    miniMaxH3AudioMode: payloadMiniMaxH3AudioMode,
    lineMappingLyrics: String(payload.lineMappingLyrics || payload.line_mapping_lyrics || payload.lyricMapper?.source_text || payload.lyric_mapper?.source_text || ""),
    mode: openingMode,
    scenes: scenesFromBuilderPayload(payload).map((scene) => ({
      ...scene,
      video_prompt_type: payloadVideoPromptType || scene.video_prompt_type,
      performance_mode: scene.performance_mode || payloadPerformanceMode,
    })),
    referenceBuilder: normalizeReferenceBuilderCatalog(payload.referenceBuilder || payload.reference_builder || {}),
    storyLayer: normalizeStoryLayer(payload.storyLayer || payload.story_layer || {}),
    scriptImport: normalizeStoryboardScriptImportState(payload.scriptImport || payload.script_import || {}),
    onReferenceMappingsChanged: typeof payload.onReferenceMappingsChanged === "function" ? payload.onReferenceMappingsChanged : null,
    onStoryLayerChanged: typeof payload.onStoryLayerChanged === "function" ? payload.onStoryLayerChanged : null,
    onPrepareStoryContext: typeof payload.onPrepareStoryContext === "function" ? payload.onPrepareStoryContext : null,
    onPromptsExported: typeof payload.onPromptsExported === "function" ? payload.onPromptsExported : null,
    onApplyIdLoraDialoguePlan: typeof payload.onApplyIdLoraDialoguePlan === "function" ? payload.onApplyIdLoraDialoguePlan : null,
    onApplyMiniMaxDialoguePlan: typeof payload.onApplyMiniMaxDialoguePlan === "function" ? payload.onApplyMiniMaxDialoguePlan : null,
    onCreateVideoPrompt: typeof payload.onCreateVideoPrompt === "function" ? payload.onCreateVideoPrompt : null,
    onBeforeCreateVideoPrompt: typeof payload.onBeforeCreateVideoPrompt === "function" ? payload.onBeforeCreateVideoPrompt : null,
    query: "",
    selected: new Set(),
    saving: false,
    gemmaSettings: payload.gemmaSettings || payload.gemma_settings || {},
    cameraFlow: String(payload.cameraFlow || payload.camera_flow || "balanced"),
    customCameraFlowSequence: normalizeStoryboardCustomCameraFlowSequence(payload.customCameraFlowSequence || payload.custom_camera_flow_sequence || payload.builderStoryboardDefaults?.custom_camera_flow_sequence || payload.builder_storyboard_defaults?.custom_camera_flow_sequence),
    imageShotFlow: String(payload.imageShotFlow || payload.image_shot_flow || (usesFilmPlanningProfile ? "film_dialogue_coverage" : "intimate")),
    imageAesthetic: String(payload.imageAesthetic || payload.image_aesthetic || (usesFilmPlanningProfile ? "film_default" : "")),
    videoStyle: String(payload.videoStyle || payload.video_style || ""),
    videoStyleCustom: String(payload.videoStyleCustom || payload.video_style_custom || ""),
    temporalWorldEffect: String(payload.temporalWorldEffect || payload.temporal_world_effect || ""),
    temporalWorldEffectCustom: String(payload.temporalWorldEffectCustom || payload.temporal_world_effect_custom || ""),
    temporalAllowBackgroundExtras: (payload.temporalAllowBackgroundExtras ?? payload.temporal_allow_background_extras) !== false,
    temporalBackgroundIntensity: storyboardTemporalIntensity(payload.temporalBackgroundIntensity ?? payload.temporal_background_intensity ?? 8),
    temporalEnvironmentTimePassage: (payload.temporalEnvironmentTimePassage ?? payload.temporal_environment_time_passage) !== false,
    temporalProtectedCharacters: storyboardTemporalProtectedMode(payload.temporalProtectedCharacters || payload.temporal_protected_characters),
    temporalProtectedCustom: String(payload.temporalProtectedCustom || payload.temporal_protected_custom || ""),
    fxPreset: String(payload.fxPreset || payload.fx_preset || payload.builderStoryboardDefaults?.fx_preset || payload.builder_storyboard_defaults?.fx_preset || ""),
    fxCustomJson: String(payload.fxCustomJson || payload.fx_custom_json || payload.builderStoryboardDefaults?.fx_custom_json || payload.builder_storyboard_defaults?.fx_custom_json || ""),
    globalConsistencyPhrase: String(payload.globalConsistencyPhrase || payload.global_consistency_phrase || ""),
    performanceStyle: String(payload.performanceStyle || payload.performance_style || payload.performance_style_default || (usesFilmPlanningProfile ? "dialogue_naturalism" : "")),
    facialPerformance: String(payload.facialPerformance || payload.facial_performance || payload.facial_performance_default || ""),
    facialPerformanceCustom: String(payload.facialPerformanceCustom || payload.facial_performance_custom || payload.facial_performance_custom_default || ""),
    cameraMotionSpeed: storyboardSpeedValue(payload.cameraMotionSpeed ?? payload.camera_motion_speed ?? payload.motion_defaults?.camera_motion_speed, 4),
    characterMotionSpeed: storyboardSpeedValue(payload.characterMotionSpeed ?? payload.character_motion_speed ?? payload.motion_defaults?.character_motion_speed, 4),
    cutFrequency: storyboardCutFrequencyValue(payload.cutFrequency ?? payload.minimax_h3_cut_frequency ?? payload.builderStoryboardDefaults?.minimax_h3_cut_frequency ?? payload.builder_storyboard_defaults?.minimax_h3_cut_frequency),
    performanceMode: payloadPerformanceMode,
    shortFilmPlanningMode: normalizeStoryboardShortFilmPlanningMode(
      payload.shortFilmPlanningMode
      || payload.short_film_planning_mode
      || payload.builderStoryboardDefaults?.short_film_planning_mode
      || payload.builder_storyboard_defaults?.short_film_planning_mode,
    ),
    videoPromptType: payloadVideoPromptType,
    miniMaxH3Mode: payloadMiniMaxH3Mode,
    imageMode: String(payload.imageMode || payload.image_mode || "zimage").trim() || "zimage",
    imageModeLabel: String(payload.imageModeLabel || payload.image_mode_label || "").trim(),
  };

  const promptRunnerName = () => {
    const runner = String(state.gemmaSettings?.text_runner || state.gemmaSettings?.gemma_runner || "builtin").trim().toLowerCase();
    if (runner === "lm_studio" || runner === "lmstudio" || runner === "lm-studio") return "LM Studio";
    if (runner === "llm_api" || runner === "llmapi" || runner === "llm-api" || runner === "api") return "LLM API";
    if (["qwen", "qwen_local", "qwen-local", "qwen_gguf", "qwen_gguf_local"].includes(runner)) return "Qwen Local";
    if (["ownserver", "own-server", "own_server", "custom_openai", "openai_compatible", "custom_server", "my_server"].includes(runner)) return "Custom Server";
    return "Gemma Local";
  };
  const imageShotFlowPresets = usesFilmPlanningProfile ? ID_LORA_IMAGE_SHOT_FLOW_PRESETS : STORYBOARD_IMAGE_SHOT_FLOW_PRESETS;
  const imageAestheticPresets = usesFilmPlanningProfile ? ID_LORA_IMAGE_AESTHETIC_PRESETS : STORYBOARD_IMAGE_AESTHETIC_PRESETS;
  const performanceStylePresets = usesFilmPlanningProfile ? ID_LORA_PERFORMANCE_STYLE_PRESETS : PERFORMANCE_STYLE_PRESETS;
  const facialPerformancePresets = usesFilmPlanningProfile ? ID_LORA_FACIAL_PERFORMANCE_PRESETS : FACIAL_PERFORMANCE_PRESETS;
  const imageShotFlowPresetForMode = (value = "") => imageShotFlowPresets[value] || imageShotFlowPresets[Object.keys(imageShotFlowPresets)[0]] || STORYBOARD_IMAGE_SHOT_FLOW_PRESETS.intimate;
  const imageAestheticPresetForMode = (value = "") => imageAestheticPresets.find((item) => item.value === value) || imageAestheticPresets[0] || STORYBOARD_IMAGE_AESTHETIC_PRESETS[0];
  const performancePresetForMode = (value = "") => performanceStylePresets.find((item) => item.value === value) || performanceStylePresets[0] || PERFORMANCE_STYLE_PRESETS[0];
  const facialPresetForMode = (value = "") => facialPerformancePresets.find((item) => item.value === value) || facialPerformancePresets[0] || FACIAL_PERFORMANCE_PRESETS[0];
  if (!imageShotFlowPresets[state.imageShotFlow]) state.imageShotFlow = Object.keys(imageShotFlowPresets)[0] || "off";
  if (!imageAestheticPresets.some((item) => item.value === state.imageAesthetic)) state.imageAesthetic = imageAestheticPresets[0]?.value || "";
  if (!MINIMAX_VIDEO_STYLE_PRESETS.some((item) => item.value === state.videoStyle)) state.videoStyle = "";
  if (!MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.some((item) => item.value === state.temporalWorldEffect)) state.temporalWorldEffect = "";
  if (!STORYBOARD_FX_PRESETS.some((item) => item.value === state.fxPreset)) state.fxPreset = "";
  if (!performanceStylePresets.some((item) => item.value === state.performanceStyle)) state.performanceStyle = performanceStylePresets[0]?.value || "";
  if (!facialPerformancePresets.some((item) => item.value === state.facialPerformance)) state.facialPerformance = facialPerformancePresets[0]?.value || "";
  const storyboardDefaultsPayload = () => ({
    builder_storyboard_defaults: {
      global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      minimax_h3_cut_frequency: storyboardCutFrequencyValue(state.cutFrequency),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
      performance_style: String(state.performanceStyle || ""),
      short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
      camera_flow: String(state.cameraFlow || ""),
      custom_camera_flow_sequence: normalizeStoryboardCustomCameraFlowSequence(state.customCameraFlowSequence),
      image_shot_flow: String(state.imageShotFlow || ""),
      image_aesthetic: String(state.imageAesthetic || ""),
      video_style: String(state.videoStyle || ""),
      video_style_custom: String(state.videoStyleCustom || "").trim(),
      temporal_world_effect: String(state.temporalWorldEffect || ""),
      temporal_world_effect_custom: String(state.temporalWorldEffectCustom || "").trim(),
      temporal_allow_background_extras: state.temporalAllowBackgroundExtras !== false,
      temporal_background_intensity: storyboardTemporalIntensity(state.temporalBackgroundIntensity),
      temporal_environment_time_passage: state.temporalEnvironmentTimePassage !== false,
      temporal_protected_characters: storyboardTemporalProtectedMode(state.temporalProtectedCharacters),
      temporal_protected_custom: String(state.temporalProtectedCustom || "").trim(),
      fx_preset: String(state.fxPreset || ""),
      fx_custom_json: String(state.fxCustomJson || "").trim(),
    },
    global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
    performance_style_default: String(state.performanceStyle || ""),
    short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
    video_style: String(state.videoStyle || ""),
    video_style_custom: String(state.videoStyleCustom || "").trim(),
    temporal_world_effect: String(state.temporalWorldEffect || ""),
    temporal_world_effect_custom: String(state.temporalWorldEffectCustom || "").trim(),
    temporal_allow_background_extras: state.temporalAllowBackgroundExtras !== false,
    temporal_background_intensity: storyboardTemporalIntensity(state.temporalBackgroundIntensity),
    temporal_environment_time_passage: state.temporalEnvironmentTimePassage !== false,
    temporal_protected_characters: storyboardTemporalProtectedMode(state.temporalProtectedCharacters),
    temporal_protected_custom: String(state.temporalProtectedCustom || "").trim(),
    fx_preset: String(state.fxPreset || ""),
    fx_custom_json: String(state.fxCustomJson || "").trim(),
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    minimax_h3_cut_frequency: storyboardCutFrequencyValue(state.cutFrequency),
    custom_camera_flow_sequence: normalizeStoryboardCustomCameraFlowSequence(state.customCameraFlowSequence),
    motion_defaults: {
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
    },
  });
  const promptRunnerGenericName = () => promptRunnerName();
  const promptAllButtonText = () => {
    const kind = state.mode === "image_to_video_prep" ? "Video" : "Image";
    return `${promptRunnerName()} ${kind} All`;
  };

  const absorbSceneReferencesIntoCatalog = (scenes = []) => {
    const refs = normalizeReferenceBuilderCatalog(state.referenceBuilder || {});
    const locationIds = new Set(refs.locations.map((location) => String(location.id || "")).filter(Boolean));
    const locationByName = new Map(
      refs.locations
        .map((location) => [String(location.name || "").trim().toLowerCase().replace(/\s+/g, " "), location])
        .filter(([name]) => Boolean(name)),
    );
    const subjectIds = new Set(refs.subjects.map((subject) => String(subject.id || "")).filter(Boolean));
    for (const scene of scenes || []) {
      let location = scene?.location_ref;
      if ((!location || typeof location !== "object") && String(scene?.setting || "").trim()) {
        location = {
          id: "",
          name: String(scene.setting || "").trim(),
          description: String(scene.setting || "").trim(),
          image: { path: "", data: "", name: "" },
        };
      }
    if (location && typeof location === "object" && String(location.id || location.name || location.description || "").trim()) {
      const locationNameKey = String(location.name || scene.setting || "").trim().toLowerCase().replace(/\s+/g, " ");
      const existingLocation = locationNameKey ? locationByName.get(locationNameKey) : null;
      const id = String(existingLocation?.id || location.id || `location_from_scene_${scene.scene_number || refs.locations.length + 1}`).trim();
      location.id = id;
      scene.location_ref = location;
      if (!locationIds.has(id)) {
        const addedLocation = {
            id,
            name: String(location.name || scene.setting || "Saved location"),
            description: String(location.description || ""),
            trigger_phrase: String(location.trigger_phrase || ""),
            trigger_position: String(location.trigger_position || "start") === "end" ? "end" : "start",
            image: normalizeReferenceImage(location),
          };
          refs.locations.push(addedLocation);
          locationIds.add(id);
          const addedNameKey = String(addedLocation.name || "").trim().toLowerCase().replace(/\s+/g, " ");
          if (addedNameKey) locationByName.set(addedNameKey, addedLocation);
        }
      }
      for (const subject of Array.isArray(scene?.subject_refs) ? scene.subject_refs : []) {
        if (!subject || typeof subject !== "object") continue;
        const id = String(subject.id || subject.name || "").trim();
        if (!id || subjectIds.has(id)) continue;
        refs.subjects.push({
          id,
          name: String(subject.name || "Saved subject"),
          description: String(subject.description || ""),
          trigger_phrase: String(subject.trigger_phrase || ""),
          trigger_position: String(subject.trigger_position || "start") === "end" ? "end" : "start",
          image: normalizeReferenceImage(subject),
        });
        subjectIds.add(id);
      }
    }
    state.referenceBuilder = normalizeReferenceBuilderCatalog(refs);
  };

  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:stretch;justify-content:center;padding:18px;box-sizing:border-box;";
  const shell = document.createElement("div");
  shell.className = "vrgdg-storyboard-shell";
  shell.style.cssText = "width:min(1820px,calc(100vw - 36px));max-width:100%;min-width:0;height:calc(100vh - 36px);box-sizing:border-box;border:1px solid #155e75;border-radius:10px;background:#111827;color:#e5e7eb;box-shadow:0 28px 90px rgba(0,0,0,.62);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,sans-serif;";

  if (!document.getElementById("vrgdg-storyboard-responsive-styles")) {
    const responsiveStyles = document.createElement("style");
    responsiveStyles.id = "vrgdg-storyboard-responsive-styles";
    responsiveStyles.textContent = `
      @media (max-width: 1100px) {
        .vrgdg-storyboard-header {
          grid-template-columns:minmax(0,1fr) !important;
          grid-template-areas:"title" "steps" "actions" !important;
        }
        .vrgdg-storyboard-defaults-grid,
        .vrgdg-storyboard-story-grid {
          grid-template-columns:minmax(0,1fr) !important;
        }
      }
      @media (max-width: 700px) {
        .vrgdg-storyboard-header { padding:14px !important; }
        .vrgdg-storyboard-panel { margin-left:8px !important; margin-right:8px !important; }
        .vrgdg-storyboard-note { margin-left:8px !important; margin-right:8px !important; }
        .vrgdg-storyboard-footer { padding:12px !important; }
      }
    `;
    document.head.append(responsiveStyles);
  }

  const header = document.createElement("div");
  header.className = "vrgdg-storyboard-header";
  header.style.cssText = "display:grid;grid-template-columns:minmax(280px,.8fr) minmax(0,2.2fr);grid-template-areas:'title steps' 'title actions';gap:14px 22px;align-items:center;padding:20px 24px;border-bottom:1px solid #1f3b46;background:linear-gradient(180deg,#083344,#111827);min-width:0;";
  const titleBlock = document.createElement("div");
  titleBlock.style.cssText = "grid-area:title;min-width:0;overflow-wrap:anywhere;";
  titleBlock.innerHTML = `
    <div style="display:flex;gap:14px;align-items:center;min-width:0;">
      <div style="width:52px;height:52px;border-radius:12px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;">▣</div>
      <div style="min-width:0;">
        <div style="font-size:26px;font-weight:900;color:#cffafe;">Storyboard Builder <span id="vrgdg-storyboard-mode-pill" style="font-size:13px;border-radius:999px;background:#164e63;color:#a5f3fc;padding:5px 9px;vertical-align:middle;">Planning</span></div>
        <div id="vrgdg-storyboard-subtitle" style="color:#cbd5e1;font-size:14px;margin-top:3px;">Write scene cards, image prompts, and video prompts before sending them to the AI Video Builder.</div>
      </div>
    </div>
  `;
  const steps = document.createElement("div");
  steps.style.cssText = "grid-area:steps;display:flex;flex-wrap:wrap;gap:10px;align-items:center;min-width:0;width:100%;";
  const stepPrompts = makeButton("Image Prep", "purple");
  const stepPrep = makeButton("Video Prep");
  stepPrompts.style.cssText += "flex:1 1 180px;min-width:0;";
  stepPrep.style.cssText += "flex:1 1 160px;min-width:0;";
  steps.append(stepPrompts, stepPrep);
  const headerActions = document.createElement("div");
  headerActions.style.cssText = "grid-area:actions;display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:flex-end;min-width:0;width:100%;";
  const search = makeInput("", "Search scenes...");
  search.style.cssText += "flex:1 1 190px;width:auto;min-width:160px;max-width:260px;";
  const gptButton = makeButton("GPT All", "purple");
  gptButton.title = "Copy all Storyboard scene-card inputs as JSON for your custom GPT.";
  const importImagePromptsButton = makeButton("Import prompts from GPT", "purple");
  importImagePromptsButton.title = "Paste JSON from the Text to Image Prompt Builder GPT and update Image Prep prompts.";
  const gemmaAllButton = makeButton(promptAllButtonText(), "primary");
  gemmaAllButton.title = "Use the selected LLM runner to create prompts for every storyboard scene.";
  const clearPromptsButton = makeButton("Clear Prompts");
  clearPromptsButton.title = "Clear Storyboard scene-card prompt summaries, generated prompts, and extra notes without changing subjects, locations, camera, motion, or lyrics.";
  clearPromptsButton.style.borderColor = "#991b1b";
  clearPromptsButton.style.background = "#3f0808";
  const clearStoryBeatsButton = makeButton("Clear All Story Beats");
  clearStoryBeatsButton.title = "Clear the story beat from every Storyboard scene without changing lyrics, prompts, images, subjects, locations, or shot settings.";
  clearStoryBeatsButton.style.borderColor = "#991b1b";
  clearStoryBeatsButton.style.background = "#3f0808";
  const keepGemmaLoadedLabel = document.createElement("label");
  keepGemmaLoadedLabel.style.cssText = "display:flex;align-items:center;gap:6px;color:#cbd5e1;font-size:12px;font-weight:800;white-space:nowrap;";
  const keepGemmaLoadedInput = document.createElement("input");
  keepGemmaLoadedInput.type = "checkbox";
  keepGemmaLoadedInput.checked = Boolean(state.gemmaSettings?.keep_loaded_for_storyboard_all);
  keepGemmaLoadedLabel.append(keepGemmaLoadedInput, document.createTextNode("Keep local LLM loaded"));
  keepGemmaLoadedLabel.title = `When checked, ${promptRunnerName()} keeps a local text model loaded until the batch finishes. This has no effect on external runners.`;
  const add = makeButton("+ Add Scene", "purple");
  const close = makeButton("Close");
  headerActions.append(gptButton, importImagePromptsButton, gemmaAllButton, clearPromptsButton, clearStoryBeatsButton, keepGemmaLoadedLabel, search, add, close);
  for (const control of headerActions.children) {
    control.style.maxWidth = "100%";
    if (control.tagName === "BUTTON") control.style.whiteSpace = "normal";
  }
  header.append(titleBlock, steps, headerActions);

  const note = document.createElement("div");
  note.className = "vrgdg-storyboard-note";
  note.style.cssText = "margin:18px 24px 0;min-width:0;max-width:100%;box-sizing:border-box;border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:12px 14px;font-size:13px;overflow-wrap:anywhere;";
  const middleContent = document.createElement("div");
  middleContent.style.cssText = "min-width:0;min-height:0;overflow-y:auto;overflow-x:hidden;padding-bottom:18px;scrollbar-width:thin;";

  const cameraFlowBar = document.createElement("div");
  cameraFlowBar.className = "vrgdg-storyboard-defaults-grid";
  cameraFlowBar.style.cssText = "display:grid;grid-template-columns:minmax(420px,700px) minmax(0,1fr);gap:8px 12px;align-items:center;width:100%;min-width:0;max-width:100%;box-sizing:border-box;color:#cbd5e1;font-size:12px;";
  const imageShotControls = document.createElement("div");
  imageShotControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageShotLabel = document.createElement("div");
  imageShotLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageShotLabel.textContent = "Still shot flow";
  const imageShotSelect = makeSelect(
    Object.entries(imageShotFlowPresets).map(([value, preset]) => ({ value, label: preset.label })),
    state.imageShotFlow,
  );
  imageShotSelect.style.width = "max-content";
  imageShotSelect.style.minWidth = "180px";
  const imageShotApply = makeButton("Fill Missing", "primary");
  imageShotApply.title = "Fill only blank shot/composition fields for Image Prep. Existing manual choices are kept.";
  const imageShotReplace = makeButton("Replace All");
  imageShotReplace.title = "Replace every scene's shot/composition field with the selected still shot flow.";
  imageShotControls.append(imageShotLabel, imageShotSelect, imageShotApply, imageShotReplace);
  const imageShotInfo = document.createElement("div");
  imageShotInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageAestheticControls = document.createElement("div");
  imageAestheticControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageAestheticLabel = document.createElement("div");
  imageAestheticLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageAestheticLabel.textContent = "Image aesthetic";
  const imageAestheticSelect = makeSelect(imageAestheticPresets, state.imageAesthetic);
  imageAestheticSelect.style.width = "max-content";
  imageAestheticSelect.style.minWidth = "180px";
  const imageAestheticApply = makeButton("Fill Missing", "primary");
  imageAestheticApply.title = "Fill only scenes without a still camera style/aesthetic note.";
  const imageAestheticReplace = makeButton("Replace All");
  imageAestheticReplace.title = "Replace each scene's generated image aesthetic note.";
  imageAestheticControls.append(imageAestheticLabel, imageAestheticSelect, imageAestheticApply, imageAestheticReplace);
  const imageAestheticInfo = document.createElement("div");
  imageAestheticInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const videoStyleControls = document.createElement("div");
  videoStyleControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const videoStyleLabel = document.createElement("div");
  videoStyleLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  videoStyleLabel.textContent = "Video aesthetic";
  const videoStyleSelect = makeSelect(MINIMAX_VIDEO_STYLE_PRESETS, state.videoStyle);
  videoStyleSelect.style.width = "max-content";
  videoStyleSelect.style.minWidth = "220px";
  const videoStyleApply = makeButton("Fill Missing", "primary");
  videoStyleApply.title = "Fill blank style fields on eligible video scenes.";
  const videoStyleReplace = makeButton("Replace All");
  videoStyleReplace.title = "Replace the style on all eligible video scenes.";
  videoStyleControls.append(videoStyleLabel, videoStyleSelect, videoStyleApply, videoStyleReplace);
  const videoStyleCustomControls = document.createElement("div");
  videoStyleCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const videoStyleCustomLabel = document.createElement("div");
  videoStyleCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  videoStyleCustomLabel.textContent = "Custom style wording";
  const videoStyleCustomInput = makeTextarea(
    state.videoStyleCustom,
    "Type the exact visual-style wording that must appear unchanged in every eligible prompt...",
    3,
  );
  videoStyleCustomInput.style.minWidth = "520px";
  videoStyleCustomControls.append(videoStyleCustomLabel, videoStyleCustomInput);
  const videoStyleInfo = document.createElement("div");
  videoStyleInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const temporalEffectControls = document.createElement("div");
  temporalEffectControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const temporalEffectLabel = document.createElement("div");
  temporalEffectLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  temporalEffectLabel.textContent = "Temporal / world effect";
  const temporalEffectSelect = makeSelect(MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS, state.temporalWorldEffect);
  temporalEffectSelect.style.width = "max-content";
  temporalEffectSelect.style.minWidth = "300px";
  temporalEffectControls.append(temporalEffectLabel, temporalEffectSelect);
  const temporalEffectCustomControls = document.createElement("div");
  temporalEffectCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const temporalEffectCustomLabel = document.createElement("div");
  temporalEffectCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  temporalEffectCustomLabel.textContent = "Custom temporal wording";
  const temporalEffectCustomInput = makeTextarea(state.temporalWorldEffectCustom, "Describe the exact temporal separation or world behavior. Character protection and audio-safety rules will be added automatically...", 3);
  temporalEffectCustomInput.style.minWidth = "520px";
  temporalEffectCustomControls.append(temporalEffectCustomLabel, temporalEffectCustomInput);
  const fxControls = document.createElement("div");
  fxControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const fxLabel = document.createElement("div");
  fxLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  fxLabel.textContent = "Shot FX preset";
  const fxSelect = makeSelect(STORYBOARD_FX_PRESETS, state.fxPreset);
  fxSelect.style.width = "max-content";
  fxSelect.style.minWidth = "260px";
  fxControls.append(fxLabel, fxSelect);
  const fxCustomControls = document.createElement("div");
  fxCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const fxCustomLabel = document.createElement("div");
  fxCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  fxCustomLabel.textContent = "Custom FX JSON";
  const fxCustomInput = makeTextarea(state.fxCustomJson, "{\n  \"label\": \"Custom FX\",\n  \"cues\": [\"A brief effect crosses the background on the beat.\"],\n  \"timing\": \"on the musical accent\",\n  \"intensity\": 6\n}", 5);
  fxCustomInput.style.minWidth = "520px";
  fxCustomControls.append(fxCustomLabel, fxCustomInput);
  const fxInfo = document.createElement("div");
  fxInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const temporalEffectOptions = document.createElement("div");
  temporalEffectOptions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:9px 10px;border:1px solid #1f3347;border-radius:7px;background:#07111f;";
  const temporalExtrasLabel = document.createElement("label");
  temporalExtrasLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;";
  const temporalExtrasInput = document.createElement("input");
  temporalExtrasInput.type = "checkbox";
  temporalExtrasInput.checked = state.temporalAllowBackgroundExtras !== false;
  temporalExtrasLabel.append(temporalExtrasInput, document.createTextNode("Allow location-appropriate anonymous extras"));
  const temporalEnvironmentLabel = document.createElement("label");
  temporalEnvironmentLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;";
  const temporalEnvironmentInput = document.createElement("input");
  temporalEnvironmentInput.type = "checkbox";
  temporalEnvironmentInput.checked = state.temporalEnvironmentTimePassage !== false;
  temporalEnvironmentLabel.append(temporalEnvironmentInput, document.createTextNode("Allow lighting / weather / time passage"));
  const temporalIntensityLabel = document.createElement("label");
  temporalIntensityLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;min-width:290px;";
  const temporalIntensityInput = makeInput(String(storyboardTemporalIntensity(state.temporalBackgroundIntensity)));
  temporalIntensityInput.type = "range";
  temporalIntensityInput.min = "0";
  temporalIntensityInput.max = "10";
  temporalIntensityInput.step = "1";
  temporalIntensityInput.style.width = "170px";
  temporalIntensityInput.style.accentColor = "#22d3ee";
  const temporalIntensityValue = document.createElement("span");
  temporalIntensityValue.style.cssText = "color:#cffafe;font-weight:900;min-width:38px;";
  temporalIntensityLabel.append(document.createTextNode("World intensity"), temporalIntensityInput, temporalIntensityValue);
  const temporalProtectedLabel = document.createElement("label");
  temporalProtectedLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;min-width:360px;";
  const temporalProtectedSelect = makeSelect([
    { value: "all_referenced", label: "Protect all referenced characters (recommended)" },
    { value: "lead_only", label: "Protect first referenced character only" },
    { value: "custom", label: "Protect named referenced characters" },
  ], state.temporalProtectedCharacters);
  temporalProtectedSelect.style.minWidth = "280px";
  temporalProtectedLabel.append(document.createTextNode("Real-time cast"), temporalProtectedSelect);
  temporalEffectOptions.append(temporalExtrasLabel, temporalEnvironmentLabel, temporalIntensityLabel, temporalProtectedLabel);
  const temporalProtectedCustomControls = document.createElement("div");
  temporalProtectedCustomControls.style.cssText = "display:flex;gap:8px;align-items:center;";
  const temporalProtectedCustomLabel = document.createElement("div");
  temporalProtectedCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  temporalProtectedCustomLabel.textContent = "Protected names";
  const temporalProtectedCustomInput = makeInput(state.temporalProtectedCustom, "Exact mapped character names, comma separated");
  temporalProtectedCustomInput.style.minWidth = "520px";
  temporalProtectedCustomControls.append(temporalProtectedCustomLabel, temporalProtectedCustomInput);
  const temporalEffectInfo = document.createElement("div");
  temporalEffectInfo.style.cssText = "color:#94a3b8;line-height:1.35;white-space:pre-wrap;";
  const consistencyControls = document.createElement("div");
  consistencyControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const consistencyLabel = document.createElement("div");
  consistencyLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  consistencyLabel.textContent = "Global consistency phrase";
  const consistencyInput = makeInput(state.globalConsistencyPhrase, "e.g. soft glittery eye makeup, wet-look hair, chrome jewelry");
  consistencyInput.style.minWidth = "520px";
  consistencyControls.append(consistencyLabel, consistencyInput);
  const consistencyInfo = document.createElement("div");
  consistencyInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cameraFlowControls = document.createElement("div");
  cameraFlowControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cameraFlowLabel = document.createElement("div");
  cameraFlowLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cameraFlowLabel.textContent = "Auto camera flow";
  const cameraFlowSelect = makeSelect(
    Object.entries(STORYBOARD_CAMERA_FLOW_PRESETS).map(([value, preset]) => ({ value, label: preset.label })),
    state.cameraFlow,
  );
  cameraFlowSelect.style.width = "max-content";
  cameraFlowSelect.style.minWidth = "180px";
  const cameraFlowApply = makeButton("Fill Missing", "primary");
  cameraFlowApply.title = "Fill only blank shot type and camera motion fields. Existing manual choices are kept.";
  const cameraFlowReplace = makeButton("Replace All");
  cameraFlowReplace.title = "Replace every scene's shot type and camera motion with the selected auto camera flow.";
  cameraFlowControls.append(cameraFlowLabel, cameraFlowSelect, cameraFlowApply, cameraFlowReplace);
  const cameraFlowInfo = document.createElement("div");
  cameraFlowInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const openCustomCameraFlowDialog = () => new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100070;background:rgba(0,0,0,.74);display:flex;align-items:center;justify-content:center;padding:22px;box-sizing:border-box;";
    const panel = document.createElement("div");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.style.cssText = "width:min(900px,calc(100vw - 44px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:11px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 90px rgba(0,0,0,.7);";
    const header = document.createElement("div");
    header.style.cssText = "padding:16px 18px;background:#083f4f;border-bottom:1px solid #155e75;";
    const title = document.createElement("div");
    title.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    title.textContent = "Import Custom Camera Shot List";
    const subtitle = document.createElement("div");
    subtitle.style.cssText = "margin-top:5px;color:#bae6fd;font-size:12px;line-height:1.45;";
    subtitle.textContent = "Paste a JSON list, a JSON object containing shots/sequence/candidates, or one shot per line. Optional camera movement can follow a pipe, em dash, arrow, or =>.";
    header.append(title, subtitle);
    const body = document.createElement("div");
    body.style.cssText = "padding:16px 18px;display:grid;gap:12px;";
    const examples = document.createElement("pre");
    examples.style.cssText = "margin:0;padding:10px;border:1px solid #334155;border-radius:8px;background:#07111f;color:#cbd5e1;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;";
    examples.textContent = `Accepted examples:\n\nJSON array:\n[{"shot":"wide performance shot","camera":"slow push-in"},{"shot":"close-up of the eyes","camera":"pan left"}]\n\nJSON object:\n{"shots":[{"shot":"tracking shot","camera":"side follow"}]}\n\nPlain list:\n1. Wide shot — slow pull-back\n2. Close-up of the hands | slow tilt down\n3. Overhead shot -> slow drift`;
    const input = document.createElement("textarea");
    input.value = state.customCameraFlowSequence.length ? JSON.stringify(state.customCameraFlowSequence, null, 2) : "";
    input.placeholder = "Paste or enter your custom camera-shot list here...";
    input.style.cssText = "width:100%;min-height:260px;resize:vertical;box-sizing:border-box;border:1px solid #475569;border-radius:8px;background:#020617;color:#e2e8f0;padding:11px;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;outline:none;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;color:#94a3b8;font-size:12px;line-height:1.4;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:9px;";
    const cancel = makeButton("Cancel");
    const importButton = makeButton("Import Custom List", "primary");
    actions.append(cancel, importButton);
    body.append(examples, input, status, actions);
    panel.append(header, body);
    backdrop.append(panel);
    document.body.append(backdrop);
    const finish = (value) => {
      document.removeEventListener("keydown", onKeyDown, true);
      backdrop.remove();
      resolve(value);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        finish(null);
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    cancel.onclick = () => finish(null);
    importButton.onclick = () => {
      const sequence = normalizeStoryboardCustomCameraFlowSequence(input.value);
      if (!sequence.length) {
        status.textContent = "No valid shot entries were found. Add a shot description, then try Import again.";
        status.style.color = "#fca5a5";
        return;
      }
      finish(sequence);
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) finish(null);
    });
    input.focus();
  });
  const cameraSpeedControls = document.createElement("div");
  cameraSpeedControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cameraSpeedLabel = document.createElement("div");
  cameraSpeedLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cameraSpeedLabel.textContent = "Camera motion speed";
  const cameraSpeedInput = makeInput(String(storyboardSpeedValue(state.cameraMotionSpeed, 4)));
  cameraSpeedInput.type = "range";
  cameraSpeedInput.min = "0";
  cameraSpeedInput.max = "10";
  cameraSpeedInput.step = "1";
  cameraSpeedInput.style.minWidth = "360px";
  cameraSpeedInput.style.accentColor = "#22d3ee";
  const cameraSpeedValue = document.createElement("div");
  cameraSpeedValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:120px;";
  const cameraSpeedHint = makeButton("Hint");
  cameraSpeedHint.title = "Explain camera motion speed.";
  cameraSpeedControls.append(cameraSpeedLabel, cameraSpeedInput, cameraSpeedValue, cameraSpeedHint);
  const cameraSpeedInfo = document.createElement("div");
  cameraSpeedInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cutFrequencyControls = document.createElement("div");
  cutFrequencyControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cutFrequencyLabel = document.createElement("div");
  cutFrequencyLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cutFrequencyLabel.textContent = "Cut frequency";
  const cutFrequencyInput = makeInput(String(storyboardCutFrequencyValue(state.cutFrequency)));
  cutFrequencyInput.type = "range";
  cutFrequencyInput.min = "0";
  cutFrequencyInput.max = "10";
  cutFrequencyInput.step = "1";
  cutFrequencyInput.style.minWidth = "360px";
  cutFrequencyInput.style.accentColor = "#22d3ee";
  const cutFrequencyValue = document.createElement("div");
  cutFrequencyValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:150px;";
  const cutFrequencyHint = makeButton("Hint");
  cutFrequencyHint.title = "Explain MiniMax cut frequency.";
  cutFrequencyControls.append(cutFrequencyLabel, cutFrequencyInput, cutFrequencyValue, cutFrequencyHint);
  const cutFrequencyInfo = document.createElement("div");
  cutFrequencyInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const performanceControls = document.createElement("div");
  performanceControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const performanceLabel = document.createElement("div");
  performanceLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  performanceLabel.textContent = usesFilmPlanningProfile ? "Global acting style" : "Global performance style";
  const performanceSelect = makeSelect(performanceStylePresets, state.performanceStyle);
  performanceSelect.style.width = "max-content";
  performanceSelect.style.minWidth = "180px";
  const performanceApply = makeButton("Fill Missing", "primary");
  performanceApply.title = usesFilmPlanningProfile ? "Fill only blank per-scene acting style fields. Existing scene choices are kept." : "Fill only blank per-scene performance/song style fields. Existing scene choices are kept.";
  const performanceReplace = makeButton("Replace All");
  performanceReplace.title = usesFilmPlanningProfile ? "Replace every scene's acting style with the selected global style." : "Replace every scene's performance/song style with the selected global style.";
  performanceControls.append(performanceLabel, performanceSelect, performanceApply, performanceReplace);
  const performanceInfo = document.createElement("div");
  performanceInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const characterSpeedControls = document.createElement("div");
  characterSpeedControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const characterSpeedLabel = document.createElement("div");
  characterSpeedLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  characterSpeedLabel.textContent = "Character motion speed";
  const characterSpeedInput = makeInput(String(storyboardSpeedValue(state.characterMotionSpeed, 4)));
  characterSpeedInput.type = "range";
  characterSpeedInput.min = "0";
  characterSpeedInput.max = "10";
  characterSpeedInput.step = "1";
  characterSpeedInput.style.minWidth = "360px";
  characterSpeedInput.style.accentColor = "#22d3ee";
  const characterSpeedValue = document.createElement("div");
  characterSpeedValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:120px;";
  const characterSpeedHint = makeButton("Hint");
  characterSpeedHint.title = "Explain character motion speed.";
  characterSpeedControls.append(characterSpeedLabel, characterSpeedInput, characterSpeedValue, characterSpeedHint);
  const characterSpeedInfo = document.createElement("div");
  characterSpeedInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const facialControls = document.createElement("div");
  facialControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const facialLabel = document.createElement("div");
  facialLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  facialLabel.textContent = usesFilmPlanningProfile ? "Global screen face" : "Global facial performance";
  const facialSelect = makeSelect(facialPerformancePresets, state.facialPerformance);
  facialSelect.style.width = "max-content";
  facialSelect.style.minWidth = "180px";
  const facialApply = makeButton("Fill Missing", "primary");
  facialApply.title = "Fill only blank per-scene facial performance fields.";
  const facialReplace = makeButton("Replace All");
  facialReplace.title = "Replace every scene's facial performance with the selected global facial preset.";
  facialControls.append(facialLabel, facialSelect, facialApply, facialReplace);
  const facialInfo = document.createElement("div");
  facialInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const facialCustomControls = document.createElement("div");
  facialCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;white-space:nowrap;";
  const facialCustomLabel = document.createElement("div");
  facialCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:8px;";
  facialCustomLabel.textContent = "Custom facial text";
  const facialCustomInput = makeTextarea(state.facialPerformanceCustom || "", "Optional custom facial performance text, e.g. expressive eyes, active brows, natural blinking...", 3);
  facialCustomInput.style.minWidth = "520px";
  facialCustomControls.append(facialCustomLabel, facialCustomInput);
  const facialCustomInfo = document.createElement("div");
  facialCustomInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageWorldStyleControls = document.createElement("div");
  imageWorldStyleControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageWorldStyleLabel = document.createElement("div");
  imageWorldStyleLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageWorldStyleLabel.textContent = "Image world style";
  const imageWorldStyleSelect = makeSelect([
    { value: "natural", label: "Natural / realistic world" },
    { value: "surreal_subject", label: "Realistic world + surreal subject" },
    { value: "balanced_surreal", label: "Balanced surrealism" },
    { value: "full_surreal", label: "Fully surreal world" },
    { value: "abstract", label: "Abstract / nonliteral world" },
    { value: "custom", label: "Fully custom" },
  ], normalizeStoryLayer(state.storyLayer).image_world_style);
  imageWorldStyleSelect.style.minWidth = "240px";
  imageWorldStyleControls.append(imageWorldStyleLabel, imageWorldStyleSelect);
  const imageWorldStyleInfo = document.createElement("div");
  imageWorldStyleInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageCustomStyleControls = document.createElement("div");
  imageCustomStyleControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const imageCustomStyleLabel = document.createElement("div");
  imageCustomStyleLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  imageCustomStyleLabel.textContent = "Custom world direction";
  const imageCustomStyleInput = makeTextarea(normalizeStoryLayer(state.storyLayer).image_custom_style_direction, "Describe the whole visual world: environment, architecture, materials, lighting, color, perspective, subject styling, and anything to avoid...", 4);
  imageCustomStyleInput.style.minWidth = "520px";
  imageCustomStyleControls.append(imageCustomStyleLabel, imageCustomStyleInput);
  const imageCustomStyleInfo = document.createElement("div");
  imageCustomStyleInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const responsiveDefaultRows = [
    imageShotControls,
    imageAestheticControls,
    videoStyleControls,
    videoStyleCustomControls,
    temporalEffectControls,
    temporalEffectCustomControls,
    fxControls,
    fxCustomControls,
    temporalEffectOptions,
    temporalProtectedCustomControls,
    imageWorldStyleControls,
    imageCustomStyleControls,
    consistencyControls,
    cameraFlowControls,
    cameraSpeedControls,
    cutFrequencyControls,
    performanceControls,
    characterSpeedControls,
    facialControls,
    facialCustomControls,
  ];
  for (const row of responsiveDefaultRows) {
    row.style.flexWrap = "wrap";
    row.style.whiteSpace = "normal";
    row.style.minWidth = "0";
    row.style.maxWidth = "100%";
  }
  const responsiveDefaultInputs = [
    imageShotSelect,
    imageAestheticSelect,
    videoStyleSelect,
    videoStyleCustomInput,
    temporalEffectSelect,
    temporalEffectCustomInput,
    fxSelect,
    fxCustomInput,
    temporalIntensityInput,
    temporalProtectedSelect,
    temporalProtectedCustomInput,
    imageWorldStyleSelect,
    imageCustomStyleInput,
    consistencyInput,
    cameraFlowSelect,
    cameraSpeedInput,
    cutFrequencyInput,
    performanceSelect,
    characterSpeedInput,
    facialSelect,
    facialCustomInput,
  ];
  for (const control of responsiveDefaultInputs) {
    control.style.minWidth = "0";
    control.style.maxWidth = "100%";
  }
  for (const control of [videoStyleCustomInput, temporalEffectCustomInput, fxCustomInput, temporalProtectedCustomInput, imageCustomStyleInput, consistencyInput, cameraSpeedInput, cutFrequencyInput, characterSpeedInput, facialCustomInput]) {
    control.style.flex = "1 1 280px";
    control.style.width = "100%";
  }
  const responsiveDefaultInfo = [
    imageShotInfo,
    imageAestheticInfo,
    videoStyleInfo,
    temporalEffectInfo,
    fxInfo,
    imageWorldStyleInfo,
    imageCustomStyleInfo,
    consistencyInfo,
    cameraFlowInfo,
    cameraSpeedInfo,
    cutFrequencyInfo,
    performanceInfo,
    characterSpeedInfo,
    facialInfo,
    facialCustomInfo,
  ];
  for (const info of responsiveDefaultInfo) {
    info.style.minWidth = "0";
    info.style.maxWidth = "100%";
    info.style.overflowWrap = "anywhere";
  }
  cameraFlowBar.append(imageShotControls, imageShotInfo, imageAestheticControls, imageAestheticInfo, videoStyleControls, videoStyleCustomControls, videoStyleInfo, temporalEffectControls, temporalEffectCustomControls, temporalEffectOptions, temporalProtectedCustomControls, temporalEffectInfo, fxControls, fxCustomControls, fxInfo, imageWorldStyleControls, imageWorldStyleInfo, imageCustomStyleControls, imageCustomStyleInfo, consistencyControls, consistencyInfo, cameraFlowControls, cameraFlowInfo, cameraSpeedControls, cameraSpeedInfo, cutFrequencyControls, cutFrequencyInfo, performanceControls, performanceInfo, characterSpeedControls, characterSpeedInfo, facialControls, facialInfo, facialCustomControls, facialCustomInfo);

  const storyLayerBar = document.createElement("div");
  storyLayerBar.className = "vrgdg-storyboard-story-grid";
  storyLayerBar.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;min-width:0;max-width:100%;color:#cbd5e1;font-size:12px;";
  const storyLayerHeader = document.createElement("div");
  storyLayerHeader.style.cssText = "grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;min-width:0;max-width:100%;";
  const storyLayerTitle = document.createElement("div");
  storyLayerTitle.style.cssText = "flex:1 1 420px;min-width:0;max-width:100%;overflow-wrap:anywhere;";
  storyLayerTitle.innerHTML = usesFilmPlanningProfile
    ? `<div style="font-weight:900;color:#cffafe;font-size:15px;">Short Film Story Layer</div><div style="color:#94a3b8;margin-top:2px;">Dialogue-first planning for ${isIdLoraMode ? "ID-LoRA" : "MiniMax H3"} scenes, characters, and locations.</div>`
    : `<div style="font-weight:900;color:#cffafe;font-size:15px;">Story Layer</div><div style="color:#94a3b8;margin-top:2px;">Optional narrative context for connecting lyrics, sections, subjects, and locations across scenes.</div>`;
  const storyLayerEnabledLabel = document.createElement("label");
  storyLayerEnabledLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;white-space:normal;max-width:100%;";
  const storyLayerEnabledInput = document.createElement("input");
  storyLayerEnabledInput.type = "checkbox";
  storyLayerEnabledInput.checked = state.storyLayer.enabled !== false;
  storyLayerEnabledLabel.append(storyLayerEnabledInput, document.createTextNode(`Use in ${promptRunnerName()} prompts`));
  storyLayerHeader.append(storyLayerTitle, storyLayerEnabledLabel);
  const shortFilmPlanningModeWrap = document.createElement("div");
  shortFilmPlanningModeWrap.style.cssText = "grid-column:1/-1;display:none;grid-template-columns:minmax(180px,260px) minmax(0,1fr);gap:12px;align-items:start;border:1px solid #155e75;border-radius:8px;background:#071a2b;padding:12px;";
  const shortFilmPlanningModeSelect = makeSelect([
    { value: "guided_film", label: "Guided Film Automation" },
    { value: "fully_custom", label: "Fully Custom" },
  ], state.shortFilmPlanningMode);
  const shortFilmPlanningModeField = document.createElement("label");
  shortFilmPlanningModeField.style.cssText = "display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:900;color:#cbd5e1;";
  shortFilmPlanningModeField.textContent = "Short Film Authoring Mode";
  shortFilmPlanningModeField.append(shortFilmPlanningModeSelect);
  const shortFilmPlanningModeInfo = document.createElement("div");
  shortFilmPlanningModeInfo.style.cssText = "color:#bae6fd;line-height:1.45;min-width:0;overflow-wrap:anywhere;";
  shortFilmPlanningModeWrap.append(shortFilmPlanningModeField, shortFilmPlanningModeInfo);
  const overallStoryIdeaInput = makeTextarea(
    state.storyLayer.overall_story_idea || "",
    "Optional short premise, e.g. A woman navigates a surreal dream world.",
    3,
  );
  overallStoryIdeaInput.title = `Optional. Sets the overall premise, world, or theme that ${promptRunnerName()} develops through the real lyric sections.`;
  const userStoryArcInput = makeTextarea(
    state.storyLayer.user_story_arc || "",
    usesFilmPlanningProfile ? "Short film premise, conflict, tone, character goal, or pasted script..." : "Optional user story arc, e.g. Verse 1: she feels trapped. Chorus: she breaks free...",
    5,
  );
  const songStoryBriefInput = makeTextarea(
    state.storyLayer.song_story_brief || "",
    usesFilmPlanningProfile ? "LLM-created short film story brief..." : `${promptRunnerName()}-created song story brief...`,
    5,
  );
  const lyricStoryStrengthInput = makeInput(String(normalizeStoryLayer(state.storyLayer).lyric_story_strength));
  lyricStoryStrengthInput.type = "range";
  lyricStoryStrengthInput.min = "0";
  lyricStoryStrengthInput.max = "10";
  lyricStoryStrengthInput.step = "1";
  lyricStoryStrengthInput.style.accentColor = "#22d3ee";
  const lyricStoryStrengthValue = document.createElement("div");
  lyricStoryStrengthValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:105px;text-align:right;";
  const lyricStoryStrengthHintButton = makeButton("Hint");
  lyricStoryStrengthHintButton.title = "Explain Lyric Story Strength.";
  const lyricStoryStrengthText = (value) => {
    const strength = Math.max(0, Math.min(10, Number(value || 7)));
    if (strength <= 0) return "0 / ignore lyrics";
    if (strength <= 3) return `${strength} / mood only`;
    if (strength <= 6) return `${strength} / balanced`;
    if (strength <= 8) return `${strength} / strong lyric story`;
    return `${strength} / literal lyric anchors`;
  };
  const syncLyricStoryStrengthLabel = () => {
    lyricStoryStrengthValue.textContent = lyricStoryStrengthText(lyricStoryStrengthInput.value);
  };
  const storyField = (label, control) => {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:900;color:#cbd5e1;";
    wrap.textContent = label;
    wrap.append(control);
    return wrap;
  };
  const overallStoryIdeaField = storyField("Overall Story Idea (optional)", overallStoryIdeaInput);
  overallStoryIdeaField.style.gridColumn = "1/-1";
  const overallStoryIdeaHint = document.createElement("div");
  overallStoryIdeaHint.style.cssText = "font-size:11px;font-weight:500;color:#94a3b8;line-height:1.4;";
  overallStoryIdeaHint.textContent = `Sets the overall premise, world, or theme. ${promptRunnerName()} will develop it through the actual reference-lyric sections; leave blank for a lyric-led idea.`;
  overallStoryIdeaField.append(overallStoryIdeaHint);
  syncLyricStoryStrengthLabel();
  const lyricStoryStrengthRow = document.createElement("div");
  lyricStoryStrengthRow.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:end;";
  lyricStoryStrengthRow.append(storyField("Lyric Story Strength", lyricStoryStrengthInput), lyricStoryStrengthValue, lyricStoryStrengthHintButton);
  lyricStoryStrengthRow.style.display = usesFilmPlanningProfile ? "none" : "grid";
  const idLoraDialoguePlanner = document.createElement("div");
  idLoraDialoguePlanner.style.cssText = "grid-column:1/-1;display:none;border:1px solid #155e75;border-radius:8px;background:#082f49;padding:12px;gap:10px;align-items:center;grid-template-columns:minmax(0,1fr) auto;";
  const idLoraDialoguePlannerText = document.createElement("div");
  idLoraDialoguePlannerText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Plan Dialogue Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, the selected LLM invents a short-film dialogue scene plan from your ${isIdLoraMode ? "ID-LoRA" : "MiniMax H3"} characters and locations.</div>`;
  const idLoraDialogueControls = document.createElement("div");
  idLoraDialogueControls.style.cssText = "display:flex;gap:8px;align-items:end;flex-wrap:wrap;justify-content:flex-end;";
  const idLoraDialogueSceneCount = makeInput("6");
  idLoraDialogueSceneCount.type = "number";
  idLoraDialogueSceneCount.min = "1";
  idLoraDialogueSceneCount.max = "24";
  idLoraDialogueSceneCount.step = "1";
  idLoraDialogueSceneCount.style.width = "76px";
  const planDialogueScenesButton = makeButton("Plan Storyboard Scenes", "primary");
  planDialogueScenesButton.title = `${isIdLoraMode ? "ID-LoRA" : "MiniMax H3 Guided Film"}. Develop editable scene cards inside Storyboard Builder. This does not create Video Builder timeline segments.`;
  const applyDialoguePlanButton = makeButton("Create Timeline Segments", "primary");
  applyDialoguePlanButton.title = "Create real Video Builder timeline segments from the reviewed storyboard scenes.";
  applyDialoguePlanButton.style.display = "none";
  idLoraDialogueControls.append(storyField("Scenes", idLoraDialogueSceneCount), planDialogueScenesButton, applyDialoguePlanButton);
  idLoraDialoguePlanner.append(idLoraDialoguePlannerText, idLoraDialogueControls);
  const miniMaxGuidedWorkflowSteps = document.createElement("div");
  miniMaxGuidedWorkflowSteps.style.cssText = "grid-column:1/-1;display:none;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;border:1px solid #155e75;border-radius:8px;background:#041923;padding:10px;";
  miniMaxGuidedWorkflowSteps.innerHTML = `
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 1 — SCRIPT</strong><br><span style="color:#bae6fd;">Import or review the script, map speakers, then click <strong>Use This Script</strong>.</span></div>
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 2 — DEVELOP</strong><br><span style="color:#bae6fd;">Create the editable storyboard scene cards. The timeline is still unchanged.</span></div>
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 3 — REVIEW</strong><br><span style="color:#bae6fd;">Review and edit the scene cards, dialogue, references, shots, and continuity below.</span></div>
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 4 — TIMELINE</strong><br><span style="color:#bae6fd;">Create the real Video Builder timeline segments only after reviewing the cards.</span></div>
  `;
  const miniMaxScriptImporter = document.createElement("div");
  miniMaxScriptImporter.style.cssText = "grid-column:1/-1;display:none;border:1px solid #0e7490;border-radius:8px;background:#06283d;padding:12px;gap:10px;align-items:center;grid-template-columns:minmax(0,1fr) auto;";
  const miniMaxScriptImporterText = document.createElement("div");
  miniMaxScriptImporterText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Import Script / Script Mapper</div><div style="color:#bae6fd;line-height:1.4;margin-top:3px;">Paste a <strong>speaker: dialogue</strong> script or load a .txt/.json file. Validate exact cues, match speakers, and preview automatically timed MiniMax segments without changing the timeline.</div>`;
  const openMiniMaxScriptMapperButton = makeButton("Import Script / Script Mapper", "primary");
  openMiniMaxScriptMapperButton.title = "Import, map, time, and activate an exact dialogue script for MiniMax Guided Film Automation.";
  miniMaxScriptImporter.append(miniMaxScriptImporterText, openMiniMaxScriptMapperButton);
  const storyActions = document.createElement("div");
  storyActions.style.cssText = "grid-column:1/-1;display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
  const storyActionsLabel = document.createElement("div");
  storyActionsLabel.style.cssText = "flex:0 0 100%;font-size:12px;font-weight:900;color:#fcd34d;border-top:1px solid #334155;padding-top:10px;";
  storyActionsLabel.textContent = usesFilmPlanningProfile
    ? "OPTIONAL STORY PLANNING TOOLS — not required when using an imported authoritative script"
    : "OPTIONAL STORY PLANNING TOOLS";
  const createStoryArcButton = makeButton("Create User Story Arc", "primary");
  const createStoryBriefButton = makeButton("Create Story Brief", "primary");
  const gptStoryButton = makeButton("GPT Story");
  gptStoryButton.title = "Copy all story, lyric, scene, reference, and preset details as JSON, then open the Storyboard GPT.";
  const importStoryJsonButton = makeButton("Import story from json");
  importStoryJsonButton.title = "Paste or load GPT story JSON and fill the overall idea, story arc, and story brief.";
  const createMissingBeatsButton = makeButton("Create Missing Scene Beats", "purple");
  const replaceBeatsButton = makeButton("Replace All Scene Beats");
  const detectSectionsButton = makeButton("Detect Lyric Sections");
  storyActions.append(storyActionsLabel, createStoryArcButton, createStoryBriefButton, createMissingBeatsButton, replaceBeatsButton, detectSectionsButton, gptStoryButton, importStoryJsonButton);
  storyLayerBar.append(
    storyLayerHeader,
    shortFilmPlanningModeWrap,
    lyricStoryStrengthRow,
    overallStoryIdeaField,
    storyField("User Story Arc", userStoryArcInput),
    storyField("Song Story Brief", songStoryBriefInput),
    miniMaxGuidedWorkflowSteps,
    miniMaxScriptImporter,
    idLoraDialoguePlanner,
    storyActions,
  );

  const sceneDefaultsPanel = makeCollapsiblePanel("Scene Defaults", "", cameraFlowBar, { open: false });
  sceneDefaultsPanel.classList.add("vrgdg-storyboard-panel");
  const hasStoryLayerContent = Boolean(String(state.storyLayer.overall_story_idea || "").trim() || String(state.storyLayer.user_story_arc || "").trim() || String(state.storyLayer.song_story_brief || "").trim());
  const storyLayerPanel = makeCollapsiblePanel("Story Layer", "", storyLayerBar, { open: hasStoryLayerContent || isMiniMaxShortFilmMode });
  storyLayerPanel.classList.add("vrgdg-storyboard-panel");

  const tableWrap = document.createElement("div");
  tableWrap.style.cssText = "margin:10px 24px 18px;overflow:auto;border:1px solid #334155;border-radius:10px;background:#0b1220;min-height:0;";

  const footer = document.createElement("div");
  footer.className = "vrgdg-storyboard-footer";
  footer.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px;padding:16px 24px;border-top:1px solid #334155;background:#111827;min-width:0;";
  const stats = document.createElement("div");
  stats.style.cssText = "flex:1 1 260px;min-width:0;color:#cbd5e1;font-size:13px;overflow-wrap:anywhere;";
  const footerActions = document.createElement("div");
  footerActions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:flex-end;min-width:0;max-width:100%;";
  const save = makeButton("Save Storyboard");
  const exportPrompts = makeButton(state.onPromptsExported ? "Save Prompts to Timeline + Files" : "Export Prompt Files Only", "purple");
  exportPrompts.title = state.onPromptsExported
    ? "Copy prompts into matching Video Builder timeline segments and write TXT and JSON prompt files. This does not create or replace timeline segments."
    : "Write TXT and JSON prompt files only. This does not create or replace Video Builder timeline segments.";
  footerActions.append(save, exportPrompts);
  footer.append(stats, footerActions);

  middleContent.append(sceneDefaultsPanel, storyLayerPanel, tableWrap);
  shell.append(header, note, middleContent, footer);
  backdrop.append(shell);
  document.body.append(backdrop);

  const setMode = (mode) => {
    state.mode = mode;
    const isVideoPrepMode = mode === "image_to_video_prep";
    const videoStyleEligible = state.projectVideoEngine === "ltx"
      || (state.projectVideoEngine === "minimax_h3" && ["text_to_video", "reference_to_video"].includes(state.miniMaxH3Mode))
      || state.scenes.some((scene) => storyboardSceneSupportsVideoStyle(scene));
    stepPrompts.style.background = mode === "storyboard_prompts" ? "#0e7490" : "#2b2b30";
    stepPrompts.style.borderColor = mode === "storyboard_prompts" ? "#06b6d4" : "#3f3f46";
    stepPrep.style.background = mode === "image_to_video_prep" ? "#0e7490" : "#2b2b30";
    stepPrep.style.borderColor = mode === "image_to_video_prep" ? "#06b6d4" : "#3f3f46";
    shell.querySelector("#vrgdg-storyboard-mode-pill").textContent = mode === "image_to_video_prep" ? "Video Prep" : "Planning";
    shell.querySelector("#vrgdg-storyboard-subtitle").textContent = mode === "image_to_video_prep"
      ? "Use scene images with vision guidance to create video prompts before rendering."
      : "Create text-to-image prompts for each scene before image generation.";
    note.textContent = mode === "image_to_video_prep"
      ? "Video Prep uses existing scene images when available, plus subjects, locations, lyrics, story beats, and motion notes to create video prompts."
      : "Image Prep creates text-to-image prompts from subjects, locations, lyrics, story beats, shot direction, and the story layer.";
    gemmaAllButton.textContent = promptAllButtonText();
    gemmaAllButton.title = mode === "image_to_video_prep"
      ? "Choose whether to create only missing video prompts or redo all visible scenes. If a scene has an image path, local vision uses it as guidance."
      : "Create text-to-image prompts for the visible scenes with the selected LLM runner.";
    gptButton.textContent = mode === "image_to_video_prep" ? "GPT Video All" : "GPT Image All";
    gptButton.title = mode === "image_to_video_prep"
      ? "Copy all Storyboard scene-card inputs as JSON and open the video prompt GPT."
      : "Copy all Image Prep scene-card inputs as JSON and open the Krea 2 text-to-image prompt GPT.";
    importImagePromptsButton.style.display = isVideoPrepMode ? "none" : "";
    imageShotControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageShotInfo.style.display = isVideoPrepMode ? "none" : "";
    imageAestheticControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageAestheticInfo.style.display = isVideoPrepMode ? "none" : "";
    videoStyleControls.style.display = isVideoPrepMode && videoStyleEligible ? "flex" : "none";
    videoStyleCustomControls.style.display = isVideoPrepMode && videoStyleEligible && state.videoStyle === "custom" ? "flex" : "none";
    videoStyleInfo.style.display = isVideoPrepMode && videoStyleEligible ? "" : "none";
    const temporalEffectEligible = isVideoPrepMode;
    temporalEffectControls.style.display = temporalEffectEligible ? "flex" : "none";
    temporalEffectCustomControls.style.display = temporalEffectEligible && state.temporalWorldEffect === "custom" ? "flex" : "none";
    temporalEffectOptions.style.display = temporalEffectEligible && Boolean(state.temporalWorldEffect) ? "flex" : "none";
    temporalProtectedCustomControls.style.display = temporalEffectEligible && Boolean(state.temporalWorldEffect) && state.temporalProtectedCharacters === "custom" ? "flex" : "none";
    temporalEffectInfo.style.display = temporalEffectEligible ? "" : "none";
    fxControls.style.display = isVideoPrepMode ? "flex" : "none";
    fxInfo.style.display = isVideoPrepMode ? "" : "none";
    fxCustomControls.style.display = isVideoPrepMode && state.fxPreset === "custom" ? "flex" : "none";
    imageWorldStyleControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageWorldStyleInfo.style.display = isVideoPrepMode ? "none" : "";
    imageCustomStyleControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageCustomStyleInfo.style.display = isVideoPrepMode ? "none" : "";
    cameraFlowControls.style.display = isVideoPrepMode ? "flex" : "none";
    cameraFlowInfo.style.display = isVideoPrepMode ? "" : "none";
    cameraSpeedControls.style.display = isVideoPrepMode ? "flex" : "none";
    cameraSpeedInfo.style.display = isVideoPrepMode ? "" : "none";
    const cutFrequencyEligible = isVideoPrepMode;
    cutFrequencyControls.style.display = cutFrequencyEligible ? "flex" : "none";
    cutFrequencyInfo.style.display = cutFrequencyEligible ? "" : "none";
    characterSpeedControls.style.display = isVideoPrepMode ? "flex" : "none";
    characterSpeedInfo.style.display = isVideoPrepMode ? "" : "none";
    refreshConsistencyInfo();
    refreshSetupPanelSummaries();
    renderTable();
  };

  const cameraFlowEntryForScene = (profileKey, sceneIndex, previousMotion = "") => {
    return storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion, state.customCameraFlowSequence);
  };

  const sceneLooksLikeStarterPlaceholder = (scene = {}) => {
    const text = [
      scene.lyrics,
      scene.story_beat,
      scene.prompt_summary,
      scene.motion_summary,
      scene.image_prompt,
      scene.video_prompt,
      scene.image_path,
      scene.setting,
    ].map((item) => String(item || "").trim()).join("");
    return !text;
  };

  const isFullyCustomShortFilm = () => isMiniMaxShortFilmMode
    && normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode) === "fully_custom";
  const shouldShowFilmDialoguePlanner = () => {
    return (isIdLoraMode || (isMiniMaxShortFilmMode && !isFullyCustomShortFilm()))
      && state.scenes.length > 0
      && state.scenes.length <= 2
      && state.scenes.every(sceneLooksLikeStarterPlaceholder);
  };
  const hasFilmDialoguePlan = () => {
    return (isIdLoraMode || isMiniMaxShortFilmMode)
      && state.scenes.some((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim())
      && (isIdLoraMode
        ? state.scenes.some((scene) => String(scene.video_prompt_type || "") === "id_lora")
        : state.scenes.some((scene) => normalizeStoryboardProjectVideoEngine(scene.project_video_engine || state.projectVideoEngine) === "minimax_h3"));
  };

  const refreshSetupPanelSummaries = () => {
    const cameraPreset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const imageShotPreset = imageShotFlowPresetForMode(state.imageShotFlow);
    const imageAestheticPreset = imageAestheticPresetForMode(state.imageAesthetic);
    const videoStylePreset = storyboardMiniMaxVideoStylePreset(state.videoStyle);
    const temporalEffectPreset = storyboardTemporalWorldEffectPreset(state.temporalWorldEffect);
    const performancePreset = performancePresetForMode(state.performanceStyle);
    const facialPreset = facialPresetForMode(state.facialPerformance);
    sceneDefaultsPanel.setSummary(state.mode === "image_to_video_prep"
      ? `${cameraPreset.label || "Camera flow"}${state.videoStyle ? ` · ${videoStylePreset.label}` : ""}${state.temporalWorldEffect ? ` · ${temporalEffectPreset.label}` : ""}${state.fxPreset ? ` · ${storyboardFxPreset(state.fxPreset).label}` : ""} · camera ${storyboardSpeedValue(state.cameraMotionSpeed, 4)}/10 · cuts ${storyboardCutFrequencyValue(state.cutFrequency)}/10 · character ${storyboardSpeedValue(state.characterMotionSpeed, 4)}/10 · ${performancePreset.label || "Performance style"} · ${facialPreset.label || "Facial performance"}${state.globalConsistencyPhrase ? " · consistency phrase" : ""}`
      : `${imageShotPreset.label || "Still shot flow"} · ${imageAestheticPreset.label || "Image aesthetic"} · ${performancePreset.label || "Performance style"} · ${facialPreset.label || "Facial performance"}${state.globalConsistencyPhrase ? " · consistency phrase" : ""}`);
    const beatCount = state.scenes.filter((scene) => String(scene.story_beat || "").trim()).length;
    const sectionCount = state.scenes.filter((scene) => String(scene.lyric_section || "").trim()).length;
    const hasBrief = Boolean(String(state.storyLayer.song_story_brief || "").trim());
    const hasArc = Boolean(String(state.storyLayer.user_story_arc || "").trim());
    const lyricStrength = normalizeStoryLayer(state.storyLayer).lyric_story_strength;
    const filmPlannerVisible = shouldShowFilmDialoguePlanner();
    const fullyCustom = isFullyCustomShortFilm();
    const activeScriptImport = normalizeStoryboardScriptImportState(state.scriptImport);
    shortFilmPlanningModeWrap.style.display = isMiniMaxShortFilmMode ? "grid" : "none";
    miniMaxGuidedWorkflowSteps.style.display = isMiniMaxShortFilmMode && state.miniMaxH3AudioMode === "built_in_audio" && !fullyCustom ? "grid" : "none";
    miniMaxScriptImporter.style.display = isMiniMaxShortFilmMode && state.miniMaxH3AudioMode === "built_in_audio" ? "grid" : "none";
    miniMaxScriptImporterText.innerHTML = activeScriptImport.enabled
      ? `<div style="font-weight:900;color:#cffafe;">Authoritative Script Active</div><div style="color:#bae6fd;line-height:1.4;margin-top:3px;"><strong>${activeScriptImport.cues.length}</strong> exact dialogue cues are mapped into <strong>${activeScriptImport.scene_plan.scene_count}</strong> planned MiniMax segments at a ${activeScriptImport.maximum_scene_seconds}-second maximum. Guided Film may develop the visual story but cannot rewrite the dialogue.</div>`
      : `<div style="font-weight:900;color:#cffafe;">Import Script / Script Mapper</div><div style="color:#bae6fd;line-height:1.4;margin-top:3px;">Paste a <strong>speaker: dialogue</strong> script or load a .txt/.json file. Validate exact cues, match speakers, and preview automatically timed MiniMax segments without changing the timeline.</div>`;
    openMiniMaxScriptMapperButton.textContent = activeScriptImport.enabled ? "Step 1 — Review / Replace Script" : "Step 1 — Import / Activate Script";
    if (activeScriptImport.enabled) {
      idLoraDialogueSceneCount.value = String(activeScriptImport.scene_plan.scene_count || 1);
      idLoraDialogueSceneCount.disabled = true;
      idLoraDialogueSceneCount.title = "The authoritative Script Mapper plan controls the required segment count.";
      idLoraDialoguePlannerText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Step 2 — Develop Imported Script Storyboard</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">The LLM will create editable storyboard scene cards with visual story beats, actions, reactions, shots, camera direction, locations, ambience, and continuity for all ${activeScriptImport.scene_plan.scene_count} locked script sections. Exact dialogue, speakers, and order are enforced. The timeline remains unchanged. Afterward, complete <strong>Step 3</strong> by reviewing the cards below.</div>`;
      planDialogueScenesButton.textContent = `Step 2 — Develop ${activeScriptImport.scene_plan.scene_count} Scenes`;
      planDialogueScenesButton.title = `Develop ${activeScriptImport.scene_plan.scene_count} editable storyboard scene cards. After reviewing them, use Create ${activeScriptImport.scene_plan.scene_count} Timeline Segments.`;
    } else {
      idLoraDialogueSceneCount.disabled = false;
      idLoraDialogueSceneCount.title = "Number of guided dialogue scenes to create.";
      idLoraDialoguePlannerText.innerHTML = isMiniMaxShortFilmMode
        ? `<div style="font-weight:900;color:#cffafe;">Step 2 — Plan Storyboard Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, the selected LLM invents editable short-film storyboard scenes from your MiniMax H3 characters and locations. The timeline remains unchanged until Step 4.</div>`
        : `<div style="font-weight:900;color:#cffafe;">Plan Storyboard Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, the selected LLM invents editable short-film storyboard scenes from your ID-LoRA characters and locations. This does not alter the timeline until you choose Create Timeline Segments.</div>`;
      planDialogueScenesButton.textContent = isMiniMaxShortFilmMode ? "Step 2 — Plan Storyboard Scenes" : "Plan Storyboard Scenes";
      planDialogueScenesButton.title = "Develop editable storyboard scene cards. This does not create Video Builder timeline segments.";
    }
    shortFilmPlanningModeInfo.innerHTML = fullyCustom
      ? `<strong style="color:#cffafe;">Manual scene cards are authoritative.</strong><br>Enter dialogue in Speaker Assignment and fill the scene beat, action, shot, camera, setting, references, audio direction, and continuity yourself. Prompt generation formats your entries but may not invent or rewrite them.`
      : `<strong style="color:#cffafe;">The LLM can help plan the film.</strong><br>Use the premise/script, reference characters, film shot coverage, story beats, and dialogue planner to create scene cards. You can still edit every result manually before prompting.`;
    idLoraDialoguePlanner.style.display = !fullyCustom && (activeScriptImport.enabled || filmPlannerVisible || hasFilmDialoguePlan()) ? "grid" : "none";
    const hasApplyDialogueCallback = isIdLoraMode ? Boolean(state.onApplyIdLoraDialoguePlan) : Boolean(state.onApplyMiniMaxDialoguePlan);
    applyDialoguePlanButton.style.display = hasFilmDialoguePlan() && hasApplyDialogueCallback ? "" : "none";
    const plannedTimelineSceneCount = state.scenes.filter((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim()).length;
    applyDialoguePlanButton.textContent = plannedTimelineSceneCount
      ? `${isMiniMaxShortFilmMode ? "Step 4 — " : ""}Create ${plannedTimelineSceneCount} Timeline Segment${plannedTimelineSceneCount === 1 ? "" : "s"}`
      : `${isMiniMaxShortFilmMode ? "Step 4 — " : ""}Create Timeline Segments`;
    applyDialoguePlanButton.title = plannedTimelineSceneCount
      ? `Create ${plannedTimelineSceneCount} real Video Builder timeline segment${plannedTimelineSceneCount === 1 ? "" : "s"} from these storyboard scenes. This may replace existing base timeline scenes.`
      : "Create real Video Builder timeline segments from the reviewed storyboard scenes.";
    storyActions.style.display = fullyCustom ? "none" : "flex";
    createStoryArcButton.textContent = usesFilmPlanningProfile ? "Create Story Premise" : "Create User Story Arc";
    createStoryBriefButton.textContent = usesFilmPlanningProfile ? "Create Short Film Brief" : "Create Story Brief";
    createMissingBeatsButton.textContent = isIdLoraMode ? "Create Missing Scene Beats" : "Create Missing Scene Beats";
    replaceBeatsButton.textContent = isIdLoraMode ? "Replace All Scene Beats" : "Replace All Scene Beats";
    detectSectionsButton.style.display = usesFilmPlanningProfile ? "none" : "";
    storyLayerPanel.setSummary(usesFilmPlanningProfile
      ? `${state.storyLayer.enabled === false ? "Off" : "On"} · ${isIdLoraMode ? "ID-LoRA dialogue story" : (fullyCustom ? "MiniMax fully custom film" : "MiniMax guided film")} · ${beatCount}/${state.scenes.length} beats${hasBrief ? " · brief" : ""}${hasArc ? " · premise" : ""}${filmPlannerVisible ? " · starter scenes" : ""}`
      : `${state.storyLayer.enabled === false ? "Off" : "On"} · lyric ${lyricStrength}/10 · ${beatCount}/${state.scenes.length} beats · ${sectionCount}/${state.scenes.length} sections${hasBrief ? " · brief" : ""}${hasArc ? " · user arc" : ""}`);
  };

  const refreshCameraFlowInfo = () => {
    const preset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const count = state.cameraFlow === "custom"
      ? normalizeStoryboardCustomCameraFlowSequence(state.customCameraFlowSequence).length
      : (preset.sequence?.length || 0);
    cameraFlowInfo.textContent = state.cameraFlow === "off"
      ? preset.description
      : `${preset.description} ${state.cameraFlow === "custom" ? (count ? `The project list contains ${count} shot${count === 1 ? "" : "s"}.` : "Import a custom shot list to activate this flow.") : `For any scene count, it cycles through ${count} camera beats and only fills blank fields.`}`;
    refreshSetupPanelSummaries();
  };

  const refreshCameraSpeedInfo = () => {
    cameraSpeedValue.textContent = storyboardSpeedLabel(state.cameraMotionSpeed, "camera");
    cameraSpeedInfo.textContent = storyboardSpeedGuidance(state.cameraMotionSpeed, "camera");
    refreshSetupPanelSummaries();
  };

  const refreshCutFrequencyInfo = () => {
    const frequency = storyboardCutFrequencyValue(state.cutFrequency);
    const engineLabel = state.projectVideoEngine === "minimax_h3" ? "MiniMax" : "LTX";
    cutFrequencyValue.textContent = storyboardCutFrequencyLabel(frequency);
    cutFrequencyInfo.textContent = frequency <= 0
      ? `${engineLabel} prompts use one smooth, continuous shot for every segment. Existing prompts are not changed until regenerated.`
      : frequency >= 10
        ? `Maximum ${engineLabel} editing: each segment requests a new continuity-preserving shot every second. A 5-second segment gets four cuts.`
        : `${engineLabel} scales this ${frequency}/10 editing intensity to each segment's exact duration. LTX writes the cuts in ordinary language; MiniMax uses its structured CUT TO format. Existing prompts are not changed until regenerated.`;
    refreshSetupPanelSummaries();
  };

  const refreshImageShotInfo = () => {
    const preset = imageShotFlowPresetForMode(state.imageShotFlow);
    const count = preset.sequence?.length || 0;
    imageShotInfo.textContent = state.imageShotFlow === "off"
      ? preset.description
      : `${preset.description} Cycles through ${count} still compositions and only fills blank shot fields.`;
    refreshSetupPanelSummaries();
  };

  const refreshImageAestheticInfo = () => {
    const preset = imageAestheticPresetForMode(state.imageAesthetic);
    imageAestheticInfo.textContent = `${preset.description} Used as still-image aesthetic guidance for Image Prep.`;
    refreshSetupPanelSummaries();
  };

  const refreshImageWorldStyleInfo = () => {
    const labels = {
      natural: "Naturalistic world; surreal details appear only when the scene requires them.",
      surreal_subject: "The setting stays believable while the subject receives the strongest surreal treatment.",
      balanced_surreal: "Subject and environment are both visibly surreal while remaining spatially readable.",
      full_surreal: "Every visible layer follows dream logic—including environment, background, architecture, lighting, perspective, props, subject, and materials.",
      abstract: "A strongly nonliteral world built from symbolic form, impossible space, expressive material, color, and light.",
      custom: "Your custom direction is the primary whole-frame visual contract.",
    };
    imageWorldStyleInfo.textContent = labels[imageWorldStyleSelect.value] || labels.natural;
    imageCustomStyleInfo.textContent = imageCustomStyleInput.value.trim()
      ? "This custom direction is added to the selected preset and applies to the entire frame."
      : "Optional. Enter your complete style idea; select Fully custom when it should be the primary direction.";
    refreshSetupPanelSummaries();
  };

  const refreshConsistencyInfo = () => {
    consistencyInfo.textContent = state.globalConsistencyPhrase
      ? `${promptRunnerName()} will incorporate this phrase into every generated prompt while keeping the wording as intact as the scene allows.`
      : `Optional phrase ${promptRunnerName()} should preserve across every prompt, such as makeup, styling, texture, wardrobe detail, or visual motif.`;
    refreshSetupPanelSummaries();
  };

  const refreshPerformanceInfo = () => {
    const preset = performancePresetForMode(state.performanceStyle);
    const presetDescription = preset.description || preset.direction || preset.label || "Performance guidance";
    performanceInfo.textContent = state.performanceStyle
      ? `${presetDescription} Used by ${promptRunnerName()}/GPT for scenes without a per-scene ${isIdLoraMode ? "acting" : "performance"} style.`
      : `${presetDescription} Pick a style here to use it as the default for blank scenes.`;
    refreshSetupPanelSummaries();
  };

  const refreshCharacterSpeedInfo = () => {
    characterSpeedValue.textContent = storyboardSpeedLabel(state.characterMotionSpeed, "character");
    characterSpeedInfo.textContent = storyboardSpeedGuidance(state.characterMotionSpeed, "character");
    refreshSetupPanelSummaries();
  };

  const refreshFacialInfo = () => {
    const preset = facialPresetForMode(state.facialPerformance);
    facialInfo.textContent = state.facialPerformance
      ? `${preset.description} Used by ${promptRunnerName()}/GPT for scenes without a per-scene facial performance preset.`
      : `${preset.description} Pick a preset here to use it as the default for blank scenes.`;
    facialCustomInfo.textContent = state.facialPerformanceCustom
      ? "Custom facial text is appended to the selected preset, or used directly when Custom is selected."
      : "Optional custom wording for eyes, brows, cheeks, jaw, mouth behavior, emotion, and blinking.";
    refreshSetupPanelSummaries();
  };

  const syncStoryLayerFromInputs = ({ notify = false } = {}) => {
    state.storyLayer = normalizeStoryLayer({
      enabled: storyLayerEnabledInput.checked,
      overall_story_idea: overallStoryIdeaInput.value,
      user_story_arc: userStoryArcInput.value,
      song_story_brief: songStoryBriefInput.value,
      lyric_story_strength: lyricStoryStrengthInput.value,
      image_world_style: imageWorldStyleSelect.value,
      image_custom_style_direction: imageCustomStyleInput.value,
    });
    if (notify && state.onStoryLayerChanged) {
      state.onStoryLayerChanged({
        ...storyboardDefaultsPayload(),
        story_layer: normalizeStoryLayer(state.storyLayer),
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        facial_performance_default: state.facialPerformance || "",
        facial_performance_custom_default: state.facialPerformanceCustom || "",
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
      });
    }
    refreshSetupPanelSummaries();
  };
  const notifyStoryboardDefaultsChanged = () => {
    if (!state.onStoryLayerChanged) return;
    state.onStoryLayerChanged({
      ...storyboardDefaultsPayload(),
      story_layer: normalizeStoryLayer(state.storyLayer),
      script_import: normalizeStoryboardScriptImportState(state.scriptImport),
      facial_performance_default: state.facialPerformance || "",
      facial_performance_custom_default: state.facialPerformanceCustom || "",
      scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
    });
  };

  const lyricsForStoryBrief = () => {
    const blocks = [];
    state.scenes.forEach((scene, index) => {
      const normalized = normalizeScene(scene, index);
      const section = String(normalized.lyric_section || "").trim();
      const lyric = String(normalized.lyrics || "").trim();
      if (!section && !lyric) return;
      const previous = blocks[blocks.length - 1];
      if (section && previous?.section?.toLowerCase() === section.toLowerCase()) {
        if (lyric) previous.lyrics.push(lyric);
        return;
      }
      blocks.push({ section, lyrics: lyric ? [lyric] : [] });
    });
    return blocks
      .map(({ section, lyrics }) => `${section ? `[${section}]\n` : ""}${lyrics.join("\n")}`.trim())
      .filter(Boolean)
      .join("\n\n");
  };

  const refreshVideoStyleInfo = () => {
    const preset = storyboardMiniMaxVideoStylePreset(state.videoStyle);
    const exactVerbiage = storyboardMiniMaxVideoStyleVerbiage(state.videoStyle, state.videoStyleCustom);
    videoStyleCustomControls.style.display = state.mode === "image_to_video_prep"
      && (state.projectVideoEngine === "ltx"
        || (state.projectVideoEngine === "minimax_h3" && ["text_to_video", "reference_to_video"].includes(state.miniMaxH3Mode))
        || state.scenes.some((scene) => storyboardSceneSupportsVideoStyle(scene)))
      && state.videoStyle === "custom"
      ? "flex"
      : "none";
    videoStyleInfo.textContent = exactVerbiage
      ? `Required exact wording in every eligible prompt: ${exactVerbiage}`
      : "Optional. Choose the governing visual aesthetic for eligible video scenes.";
    refreshSetupPanelSummaries();
  };

  const refreshTemporalEffectInfo = () => {
    state.temporalBackgroundIntensity = storyboardTemporalIntensity(temporalIntensityInput.value);
    temporalIntensityValue.textContent = `${state.temporalBackgroundIntensity}/10`;
    temporalEffectCustomControls.style.display = state.mode === "image_to_video_prep"
      && state.temporalWorldEffect === "custom" ? "flex" : "none";
    temporalEffectOptions.style.display = state.mode === "image_to_video_prep"
      && Boolean(state.temporalWorldEffect) ? "flex" : "none";
    temporalProtectedCustomControls.style.display = state.mode === "image_to_video_prep"
      && Boolean(state.temporalWorldEffect)
      && state.temporalProtectedCharacters === "custom" ? "flex" : "none";
    const effect = storyboardTemporalWorldEffectForScene({}, state);
    temporalEffectInfo.textContent = effect
      ? `Required in every video prompt unless a scene overrides it.\n${effect.exact_verbiage}`
      : "Optional. Choose a global temporal/world effect. Existing projects and scenes remain natural-time while this is Off.";
    refreshSetupPanelSummaries();
  };

  const sectionMapFromLyrics = () => {
    const map = new Map();
    let current = "";
    state.scenes.forEach((scene, index) => {
      const lyric = String(scene.lyrics || "").trim();
      const explicit = String(scene.lyric_section || "").trim();
      const header = lyric.match(/^\s*\[([^\]]{2,80})\]\s*$/);
      if (explicit) current = explicit;
      else if (header) current = header[1].trim();
      else if (current) map.set(scene.id || `scene_${index + 1}`, current);
    });
    return map;
  };

  const detectLyricSections = () => {
    const map = sectionMapFromLyrics();
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const key = scene.id || `scene_${index + 1}`;
      const section = map.get(key);
      const lyric = String(scene.lyrics || "").trim();
      const header = lyric.match(/^\s*\[([^\]]{2,80})\]\s*$/);
      if (header && !String(scene.lyric_section || "").trim()) {
        scene.lyric_section = header[1].trim();
        changed += 1;
      } else if (section && !String(scene.lyric_section || "").trim()) {
        scene.lyric_section = section;
        changed += 1;
      }
    });
    renderTable();
    syncStoryLayerFromInputs();
    createToast(changed ? `Detected lyric sections for ${changed} scene${changed === 1 ? "" : "s"}.` : "No missing lyric sections were detected.");
  };

  const createStoryBriefWithGemma = async () => {
    syncStoryLayerFromInputs();
    const authoritativeScript = normalizeStoryboardScriptImportState(state.scriptImport);
    const progress = createStoryboardProgressWindow(`Story Brief — ${promptRunnerName()}`);
    try {
      progress.set(authoritativeScript.enabled
        ? "Creating a short-film production brief around the exact imported script..."
        : "Creating compact song story brief from lyrics, sections, and your story arc...", 18);
      const data = await postJson("/vrgdg/storyboard/story_brief", {
        ...(state.gemmaSettings || {}),
        story_layer: normalizeStoryLayer(state.storyLayer),
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        performance_mode: state.performanceMode,
        reference_builder: state.referenceBuilder || {},
        storyboard: slimStoryboardForRequest(state),
        lyrics: lyricsForStoryBrief(),
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        unload_after: true,
        max_new_tokens: authoritativeScript.enabled ? 1200 : 800,
      }, 240000);
      state.storyLayer.song_story_brief = String(data.story_brief || "").trim();
      songStoryBriefInput.value = state.storyLayer.song_story_brief;
      syncStoryLayerFromInputs({ notify: true });
      progress.set("Story brief saved into the Story Layer.", 100);
      progress.close(1600);
      createToast("Story brief created.");
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      createToast(`Story brief failed:\n${String(error?.message || error)}`, true);
    }
  };

  const createStoryArcWithGemma = async () => {
    syncStoryLayerFromInputs();
    const authoritativeScript = normalizeStoryboardScriptImportState(state.scriptImport);
    const progress = createStoryboardProgressWindow(`${authoritativeScript.enabled ? "Short Film Premise" : "Story Arc"} — ${promptRunnerName()}`);
    const storyArcSeed = Math.floor(Math.random() * 2147483647);
    const existingStoryArcText = String(userStoryArcInput.value || "").trim();
    const overallStoryIdea = String(overallStoryIdeaInput.value || "").trim();
    const storyLayerForRequest = normalizeStoryLayer({
      ...state.storyLayer,
      overall_story_idea: overallStoryIdea,
      user_story_arc: "",
    });
    try {
      progress.set(authoritativeScript.enabled
        ? `Creating a visual short-film premise around the exact imported dialogue...\nReroll seed: ${storyArcSeed}`
        : `Creating a short song-structure story arc from lyrics, subjects, and locations...\nReroll seed: ${storyArcSeed}`, 18);
      const data = await postJson("/vrgdg/storyboard/story_arc", {
        ...(state.gemmaSettings || {}),
        n_ctx: Math.max(16384, Number(state.gemmaSettings?.n_ctx) || 0),
        seed: storyArcSeed,
        story_arc_seed: storyArcSeed,
        story_layer: storyLayerForRequest,
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        performance_mode: state.performanceMode,
        storyboard: slimStoryboardForRequest(state),
        story_idea: overallStoryIdea,
        previous_story_arc: existingStoryArcText,
        lyrics: lyricsForStoryBrief(),
        project_folder: state.projectFolder,
        line_mapping_lyrics: state.lineMappingLyrics,
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        reference_builder: state.referenceBuilder || {},
        camera_flow: state.cameraFlow || "",
        camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
        character_motion: storyboardSpeedValue(state.characterMotionSpeed, 4),
        character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
        performance_style: state.performanceStyle || "",
        facial_performance: state.facialPerformance || "",
        facial_performance_custom: state.facialPerformanceCustom || "",
        unload_after: true,
        max_new_tokens: 2400,
      }, 240000);
      state.storyLayer.user_story_arc = String(data.story_arc || "").trim();
      userStoryArcInput.value = state.storyLayer.user_story_arc;
      syncStoryLayerFromInputs({ notify: true });
      progress.set(`${authoritativeScript.enabled ? "Short-film premise" : "Story arc"} saved into the Story Layer.\nSeed: ${storyArcSeed}`, 100);
      progress.close(1600);
      createToast(`${authoritativeScript.enabled ? "Short-film premise" : "Story arc"} created. Seed: ${storyArcSeed}`);
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      progress.showDiagnostics?.(error?.diagnostics);
      createToast(`Story arc failed:\n${String(error?.message || error)}`, true);
    }
  };

  const sceneBeatGemmaPayload = (scene, overrides = {}) => ({
    ...(state.gemmaSettings || {}),
    ...overrides,
    story_layer: normalizeStoryLayer(state.storyLayer),
    // A replacement request must not feed the old beat back to the model as
    // if it were authoritative. Lyrics, mappings, defaults, and references
    // remain; only the stale generated beat is cleared from this request.
    storyboard_payload: storyboardGptPayload(state, [{ ...scene, story_beat: "" }]),
    max_new_tokens: state.videoPromptType === "flf" ? 700 : 360,
    temperature: 0.35,
    top_p: 0.90,
  });

  const propagateFlfEndStateToNextScene = (scene) => {
    if (state.videoPromptType !== "flf" && scene?.video_prompt_type !== "flf") return;
    const sceneIndex = state.scenes.findIndex((item) => item.id === scene?.id);
    if (sceneIndex < 0 || sceneIndex >= state.scenes.length - 1) return;
    const endState = String(scene.flf_end_state || "").trim();
    if (!endState) return;
    state.scenes[sceneIndex + 1].flf_start_state = endState;
  };

  const createSceneBeatWithGemma = async (scene, { quiet = false, unloadAfter = true, previousBeat = "", previousLyrics = "", previousEndState = "", previousCarryForward = "", nextLyrics = "", progress = null, progressPercent = 35, progressLabel = "" } = {}) => {
    syncStoryLayerFromInputs();
    const normalized = normalizeScene(scene, 0);
    const sceneIndex = state.scenes.findIndex((item) => item.id === scene.id);
    if (!previousBeat && sceneIndex > 0) previousBeat = String(state.scenes[sceneIndex - 1]?.story_beat || "");
    if (!previousLyrics && sceneIndex > 0) previousLyrics = String(state.scenes[sceneIndex - 1]?.lyrics || "");
    if (!previousEndState && sceneIndex > 0) previousEndState = String(state.scenes[sceneIndex - 1]?.flf_end_state || "");
    if (!previousCarryForward && sceneIndex > 0) previousCarryForward = String(state.scenes[sceneIndex - 1]?.flf_carry_forward || "");
    if (!nextLyrics && sceneIndex >= 0 && sceneIndex < state.scenes.length - 1) nextLyrics = String(state.scenes[sceneIndex + 1]?.lyrics || "");
    if ((state.videoPromptType === "flf" || normalized.video_prompt_type === "flf") && sceneIndex > 0 && previousEndState.trim()) {
      scene.flf_start_state = previousEndState.trim();
    }
    try {
      progress?.set(`${progressLabel || normalized.label || "Scene"}: creating scene story beat with ${promptRunnerName()}...`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/scene_story_beat", sceneBeatGemmaPayload(scene, {
        unload_after: unloadAfter,
        previous_beat: previousBeat,
        previous_lyrics: previousLyrics,
        previous_end_state: previousEndState,
        previous_carry_forward: previousCarryForward,
        current_lyrics: normalized.lyrics,
        next_lyrics: nextLyrics,
        flf_mode: state.videoPromptType === "flf" || normalized.video_prompt_type === "flf",
      }), 240000);
      scene.story_beat = String(data.story_beat || "").trim();
      if (state.videoPromptType === "flf" || normalized.video_prompt_type === "flf") {
        scene.flf_start_state = sceneIndex > 0 && previousEndState.trim()
          ? previousEndState.trim()
          : String(data.flf_start_state || "").trim();
        scene.flf_transformation = String(data.flf_transformation || "").trim();
        scene.flf_end_state = String(data.flf_end_state || "").trim();
        scene.flf_carry_forward = String(data.flf_carry_forward || "").trim();
        propagateFlfEndStateToNextScene(scene);
      }
      if (!scene.story_beat) throw new Error(`${promptRunnerName()} returned an empty scene story beat.`);
      if (!quiet) createToast(`Scene story beat created for ${normalized.label || "scene"}.`);
      return scene.story_beat;
    } catch (error) {
      if (!quiet) createToast(`Scene story beat failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  };

  const createAllSceneBeatsWithGemma = async ({ overwrite = false, failedSceneIds = [] } = {}) => {
    syncStoryLayerFromInputs();
    const flfMode = state.videoPromptType === "flf";
    const failedIds = new Set(failedSceneIds.map((value) => String(value)));
    const scenes = currentRows().filter((scene) => failedIds.size
      ? failedIds.has(String(scene.id || ""))
      : overwrite
      || !String(scene.story_beat || "").trim()
      || (flfMode && [scene.flf_start_state, scene.flf_transformation, scene.flf_end_state, scene.flf_carry_forward].some((value) => !String(value || "").trim())));
    if (!scenes.length) {
      createToast(overwrite ? "No scenes found." : "No scene story beats are missing.");
      return;
    }
    const progress = createStoryboardProgressWindow(`${overwrite ? "Replace Scene Beats" : "Create Missing Scene Beats"} — ${promptRunnerName()}`);
    let created = 0;
    const failures = [];
    try {
      progress.set(`${failedIds.size ? "Retrying failed" : "Creating"} ${scenes.length} scene story beat${scenes.length === 1 ? "" : "s"}...`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        const scene = scenes[index];
        const allIndex = state.scenes.findIndex((item) => item.id === scene.id);
        const previousBeat = allIndex > 0 ? String(state.scenes[allIndex - 1]?.story_beat || "") : "";
        const previousLyrics = allIndex > 0 ? String(state.scenes[allIndex - 1]?.lyrics || "") : "";
        const previousEndState = allIndex > 0 ? String(state.scenes[allIndex - 1]?.flf_end_state || "") : "";
        const previousCarryForward = allIndex > 0 ? String(state.scenes[allIndex - 1]?.flf_carry_forward || "") : "";
        const nextLyrics = allIndex >= 0 && allIndex < state.scenes.length - 1 ? String(state.scenes[allIndex + 1]?.lyrics || "") : "";
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        try {
          await createSceneBeatWithGemma(scene, {
            quiet: true,
            unloadAfter: index === scenes.length - 1,
            previousBeat,
            previousLyrics,
            previousEndState,
            previousCarryForward,
            nextLyrics,
            progress,
            progressPercent: base,
            progressLabel: `Scene Beat ${index + 1}/${scenes.length}: ${scene.label || `Scene ${scene.scene_number || index + 1}`}`,
          });
          created += 1;
        } catch (error) {
          if (!isRecoverableStoryboardBatchError(error)) throw error;
          failures.push({ scene, error: String(error?.message || error) });
          progress.set(`Scene Beat ${index + 1}/${scenes.length} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      progress.set("Saving story beats...", 96);
      await saveStoryboard();
      progress.set(`Scene beats complete.\nCreated ${created} story beat${created === 1 ? "" : "s"}.${failures.length ? ` ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : ""}`, 100);
      progress.close(1600);
      createToast(`Created ${created} scene story beat${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      if (failures.length) showStoryboardBatchFailures(failures, (items) => createAllSceneBeatsWithGemma({
        failedSceneIds: items.map((item) => item.scene.id),
      }));
    } catch (error) {
      progress.set(`Scene beats stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, 100);
      createToast(`Scene beats stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, true);
    }
  };

  const openMiniMaxScriptMapper = () => {
    if (!isMiniMaxShortFilmMode || state.miniMaxH3AudioMode !== "built_in_audio") {
      createToast("Script Mapper is available for MiniMax Short Film with Built-in MiniMax Audio.", true);
      return;
    }
    const mapperBackdrop = document.createElement("div");
    mapperBackdrop.style.cssText = "position:fixed;inset:0;z-index:100070;background:rgba(0,0,0,.78);display:flex;align-items:stretch;justify-content:center;padding:18px;box-sizing:border-box;";
    const mapperShell = document.createElement("div");
    mapperShell.style.cssText = "width:min(1320px,calc(100vw - 36px));height:calc(100vh - 36px);min-height:520px;border:1px solid #0e7490;border-radius:11px;background:#07111f;color:#e5e7eb;box-shadow:0 24px 90px rgba(0,0,0,.72);overflow:hidden;display:grid;grid-template-rows:auto minmax(0,1fr) auto;";
    const mapperHeader = document.createElement("div");
    mapperHeader.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:center;padding:15px 18px;background:#083344;border-bottom:1px solid #155e75;";
    const mapperHeading = document.createElement("div");
    mapperHeading.innerHTML = `<div style="font-size:20px;font-weight:900;color:#cffafe;">Import Script / Script Mapper</div><div style="font-size:12px;color:#bae6fd;line-height:1.4;margin-top:3px;">Import, map, and time exact dialogue, then activate it as the authoritative source for Guided Film Automation. Activation alone does not change the Video Builder timeline.</div>`;
    const mapperCloseTop = makeButton("Close");
    mapperHeader.append(mapperHeading, mapperCloseTop);

    const mapperBody = document.createElement("div");
    mapperBody.style.cssText = "min-height:0;overflow:auto;padding:16px 18px;display:grid;grid-template-columns:minmax(360px,.8fr) minmax(480px,1.2fr);gap:14px;align-items:stretch;";
    const sourcePanel = document.createElement("div");
    sourcePanel.style.cssText = "min-width:0;border:1px solid #334155;border-radius:9px;background:#0b1220;padding:12px;display:flex;flex-direction:column;gap:10px;";
    const sourceTitle = document.createElement("div");
    sourceTitle.innerHTML = `<div style="font-weight:900;color:#cffafe;">Script source</div><div style="font-size:12px;color:#94a3b8;line-height:1.4;margin-top:3px;">Plain text uses <strong style="color:#e2e8f0;">speaker: exact dialogue</strong>. JSON accepts cues with speaker/speaker_name and text/dialogue fields.</div>`;
    const existingScriptImport = normalizeStoryboardScriptImportState(state.scriptImport);
    const scriptInput = makeTextarea(existingScriptImport.raw_text, "woman: Have you tried the new MiniMax H3 model yet?\n\nman: I have, and honestly...", 22);
    scriptInput.style.flex = "1 1 auto";
    scriptInput.style.minHeight = "360px";
    const sourceActions = document.createElement("div");
    sourceActions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;";
    const loadScriptButton = makeButton("Load .txt / .json", "primary");
    const parseScriptButton = makeButton("Parse + Plan Preview", "purple");
    const clearScriptButton = makeButton("Clear");
    const scriptFileInput = document.createElement("input");
    scriptFileInput.type = "file";
    scriptFileInput.accept = ".txt,.json,text/plain,application/json";
    scriptFileInput.style.display = "none";
    sourceActions.append(loadScriptButton, parseScriptButton, clearScriptButton, scriptFileInput);
    const sceneLengthSettings = document.createElement("div");
    sceneLengthSettings.style.cssText = "border:1px solid #0e7490;border-radius:7px;background:#082f49;padding:10px;display:grid;grid-template-columns:minmax(150px,.8fr) minmax(180px,1.2fr);gap:9px;align-items:center;";
    const sceneLengthLabel = document.createElement("div");
    sceneLengthLabel.innerHTML = `<div style="font-weight:900;color:#cffafe;">Maximum scene length</div><div style="font-size:11px;color:#bae6fd;line-height:1.35;margin-top:3px;">Hard ceiling for every planned MiniMax clip. Shorter clips can reduce VRAM pressure and generation time.</div>`;
    const sceneLengthControls = document.createElement("div");
    sceneLengthControls.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 96px;gap:8px;align-items:center;";
    const maxSceneLengthSelect = makeSelect([
      { value: "5", label: "5 seconds — low VRAM" },
      { value: "8", label: "8 seconds — recommended" },
      { value: "10", label: "10 seconds" },
      { value: "12", label: "12 seconds" },
      { value: "15", label: "15 seconds — maximum" },
      { value: "custom", label: "Custom..." },
    ], "8");
    const customSceneLengthInput = makeInput("8", "3–15");
    customSceneLengthInput.type = "number";
    customSceneLengthInput.min = "3";
    customSceneLengthInput.max = "15";
    customSceneLengthInput.step = "0.5";
    customSceneLengthInput.title = "Custom maximum scene length from 3 to 15 seconds";
    customSceneLengthInput.style.display = "none";
    const presetSceneLengths = new Set([5, 8, 10, 12, 15]);
    if (existingScriptImport.enabled) {
      if (presetSceneLengths.has(existingScriptImport.maximum_scene_seconds)) {
        maxSceneLengthSelect.value = String(existingScriptImport.maximum_scene_seconds);
      } else {
        maxSceneLengthSelect.value = "custom";
        customSceneLengthInput.value = String(existingScriptImport.maximum_scene_seconds);
        customSceneLengthInput.style.display = "";
      }
    }
    const currentMaximumSceneSeconds = () => Math.max(3, Math.min(15, Number(maxSceneLengthSelect.value === "custom" ? customSceneLengthInput.value : maxSceneLengthSelect.value) || 8));
    sceneLengthControls.append(maxSceneLengthSelect, customSceneLengthInput);
    sceneLengthSettings.append(sceneLengthLabel, sceneLengthControls);
    const sourceStatus = document.createElement("div");
    sourceStatus.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;line-height:1.4;";
    sourcePanel.append(sourceTitle, scriptInput, sourceActions, sceneLengthSettings, sourceStatus);

    const previewPanel = document.createElement("div");
    previewPanel.style.cssText = "min-width:0;border:1px solid #155e75;border-radius:9px;background:#071827;padding:12px;overflow:auto;";
    const renderEmptyPreview = () => {
      previewPanel.innerHTML = `<div style="height:100%;min-height:360px;display:grid;place-items:center;border:1px dashed #334155;border-radius:8px;color:#94a3b8;text-align:center;padding:24px;box-sizing:border-box;"><div><strong style="display:block;color:#cffafe;font-size:15px;margin-bottom:6px;">No parsed script yet</strong>Paste dialogue or load a file, then click Parse Preview.</div></div>`;
    };
    let lastParsed = null;
    const speakerMatches = new Map();
    for (const match of existingScriptImport.speaker_matches || []) {
      const aliasKey = storyboardScriptSpeakerMatchKey(match?.speaker_alias);
      if (!aliasKey) continue;
      speakerMatches.set(aliasKey, {
        subject_id: String(match?.reference_subject_id || ""),
        method: String(match?.match_method || (match?.reference_subject_id ? "manual" : "unmatched")),
      });
    }
    const referenceCharacters = normalizeReferenceBuilderCatalog(state.referenceBuilder).subjects;
    const copyParsedButton = makeButton("Copy Parsed JSON");
    copyParsedButton.disabled = true;
    const useGuidedScriptButton = makeButton("Use This Script in Guided Film", "primary");
    useGuidedScriptButton.disabled = true;
    const removeGuidedScriptButton = makeButton("Remove Active Script");
    removeGuidedScriptButton.style.display = existingScriptImport.enabled ? "" : "none";
    const renderParsedPreview = (parsed) => {
      const characterById = new Map(referenceCharacters.map((character) => [String(character.id || ""), character]));
      for (const speaker of Array.isArray(parsed?.speakers) ? parsed.speakers : []) {
        const aliasKey = storyboardScriptSpeakerMatchKey(speaker?.name);
        if (!aliasKey || speakerMatches.has(aliasKey)) continue;
        const suggested = suggestStoryboardScriptSpeakerMatch(speaker.name, referenceCharacters);
        speakerMatches.set(aliasKey, suggested
          ? { subject_id: String(suggested.id || ""), method: "auto" }
          : { subject_id: "", method: "unmatched" });
      }
      const mappedSpeakers = (Array.isArray(parsed?.speakers) ? parsed.speakers : []).map((speaker) => {
        const aliasKey = storyboardScriptSpeakerMatchKey(speaker?.name);
        const match = speakerMatches.get(aliasKey) || { subject_id: "", method: "unmatched" };
        const character = characterById.get(String(match.subject_id || ""));
        const method = character ? String(match.method || "manual") : "unmatched";
        return {
          ...speaker,
          speaker_alias: String(speaker.name || ""),
          reference_subject_id: character ? String(character.id || "") : "",
          reference_subject_name: character ? String(character.name || "") : "",
          match_method: method,
        };
      });
      const mappedSpeakerByKey = new Map(mappedSpeakers.map((speaker) => [storyboardScriptSpeakerMatchKey(speaker.name), speaker]));
      parsed.speakers = mappedSpeakers;
      parsed.cues = (Array.isArray(parsed?.cues) ? parsed.cues : []).map((cue) => {
        const mappedSpeaker = mappedSpeakerByKey.get(storyboardScriptSpeakerMatchKey(cue?.speaker));
        return {
          ...cue,
          speaker_alias: String(cue?.speaker || ""),
          speaker_id: String(mappedSpeaker?.reference_subject_id || ""),
          speaker_name: String(mappedSpeaker?.reference_subject_name || cue?.speaker || ""),
          reference_subject_id: String(mappedSpeaker?.reference_subject_id || ""),
          reference_subject_name: String(mappedSpeaker?.reference_subject_name || ""),
          speaker_match_method: String(mappedSpeaker?.match_method || "unmatched"),
        };
      });
      parsed.speaker_matches = mappedSpeakers.map((speaker) => ({
        speaker_alias: String(speaker.speaker_alias || speaker.name || ""),
        reference_subject_id: String(speaker.reference_subject_id || ""),
        reference_subject_name: String(speaker.reference_subject_name || ""),
        match_method: String(speaker.match_method || "unmatched"),
      }));
      parsed.unmatched_speakers = mappedSpeakers.filter((speaker) => !speaker.reference_subject_id).map((speaker) => String(speaker.name || ""));
      parsed.scene_plan = planStoryboardScriptScenes(parsed.cues, { max_scene_seconds: currentMaximumSceneSeconds() });
      lastParsed = parsed;
      copyParsedButton.disabled = !parsed?.cues?.length;
      const cueCount = Array.isArray(parsed?.cues) ? parsed.cues.length : 0;
      const speakerCount = Array.isArray(parsed?.speakers) ? parsed.speakers.length : 0;
      const wordCount = Number(parsed?.word_count || 0);
      const speechSeconds = Number(parsed?.estimated_spoken_seconds || 0);
      const errorCount = Array.isArray(parsed?.errors) ? parsed.errors.length : 0;
      const scenePlan = parsed.scene_plan || { scenes: [], warnings: [] };
      const plannedSceneCount = Number(scenePlan.scene_count || 0);
      const speakerHtml = speakerCount
        ? parsed.speakers.map((speaker) => `<span style="display:inline-flex;gap:6px;align-items:center;border:1px solid #0e7490;border-radius:999px;background:#083344;color:#cffafe;padding:5px 9px;font-size:11px;font-weight:900;">${escapeHtml(speaker.name)} <span style="color:#67e8f9;">${Number(speaker.cue_count || 0)} cue${Number(speaker.cue_count || 0) === 1 ? "" : "s"}</span></span>`).join("")
        : `<span style="color:#fca5a5;">No speakers detected.</span>`;
      const matchedCount = parsed.speakers.filter((speaker) => speaker.reference_subject_id).length;
      useGuidedScriptButton.disabled = !cueCount || errorCount > 0 || matchedCount !== speakerCount;
      useGuidedScriptButton.title = errorCount
        ? "Resolve every parse issue before activating the script."
        : matchedCount !== speakerCount
          ? "Match every script speaker to a Reference Builder character first."
          : "Save this exact script and timed segment plan as the authoritative source for Guided Film Automation.";
      const speakerMappingHtml = speakerCount
        ? parsed.speakers.map((speaker, index) => {
          const status = speaker.reference_subject_id
            ? speaker.match_method === "auto" ? "Auto matched" : "Manually matched"
            : "Needs a character";
          const statusColor = speaker.reference_subject_id ? "#86efac" : "#fbbf24";
          const options = [
            `<option value="">Choose Reference Builder character...</option>`,
            ...referenceCharacters.map((character) => `<option value="${escapeHtml(character.id)}"${String(character.id) === String(speaker.reference_subject_id) ? " selected" : ""}>${escapeHtml(character.name)}</option>`),
          ].join("");
          return `<div style="display:grid;grid-template-columns:minmax(130px,.65fr) minmax(210px,1.35fr) auto;gap:9px;align-items:center;border-top:${index ? "1px solid #1e3a5f" : "0"};padding:${index ? "9px 0 0" : "0"};margin-top:${index ? "9px" : "0"};"><div style="min-width:0;"><div style="font-size:10px;color:#94a3b8;text-transform:uppercase;font-weight:900;">Script speaker</div><div title="${escapeHtml(speaker.name)}" style="color:#cffafe;font-weight:900;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:3px;">${escapeHtml(speaker.name)}</div></div><select data-script-speaker-index="${index}" style="width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;"${referenceCharacters.length ? "" : " disabled"}>${options}</select><div style="color:${statusColor};font-size:11px;font-weight:900;white-space:nowrap;">${status}</div></div>`;
        }).join("")
        : `<div style="color:#fca5a5;">Parse at least one valid speaker before matching characters.</div>`;
      const rows = cueCount
        ? parsed.cues.map((cue) => `<tr style="border-top:1px solid #1e3a5f;"><td style="padding:8px;color:#67e8f9;font-weight:900;vertical-align:top;">${cue.index}</td><td style="padding:8px;color:#94a3b8;vertical-align:top;">${cue.line_number || "JSON"}</td><td style="padding:8px;color:#cffafe;font-weight:900;vertical-align:top;overflow-wrap:anywhere;">${escapeHtml(cue.speaker)}</td><td style="padding:8px;color:#e2e8f0;vertical-align:top;line-height:1.4;overflow-wrap:anywhere;">${escapeHtml(cue.text)}</td><td style="padding:8px;color:#a5f3fc;text-align:right;vertical-align:top;">${cue.word_count}</td></tr>`).join("")
        : `<tr><td colspan="5" style="padding:18px;color:#fca5a5;text-align:center;">No valid dialogue cues were parsed.</td></tr>`;
      const errorsHtml = errorCount
        ? `<div style="margin-top:12px;border:1px solid #991b1b;border-radius:7px;background:#3f0808;padding:10px;"><div style="font-weight:900;color:#fecaca;">${errorCount} issue${errorCount === 1 ? "" : "s"} found</div>${parsed.errors.map((error) => `<div style="margin-top:6px;color:#fecaca;font-size:12px;line-height:1.4;"><strong>${error.line_number ? `Line ${error.line_number}: ` : ""}</strong>${escapeHtml(error.message)}${error.source ? `<div style="color:#fca5a5;font-family:monospace;overflow-wrap:anywhere;">${escapeHtml(error.source)}</div>` : ""}</div>`).join("")}</div>`
        : `<div style="margin-top:12px;border:1px solid #166534;border-radius:7px;background:#052e16;color:#bbf7d0;padding:9px 10px;font-size:12px;font-weight:900;">All non-empty script lines parsed successfully. Exact dialogue was preserved.</div>`;
      const scenePlanHtml = plannedSceneCount
        ? scenePlan.scenes.map((scene) => {
          const participantHtml = Array.isArray(scene.participants) && scene.participants.length
            ? scene.participants.map((participant) => `<span style="display:inline-flex;border:1px solid #334155;border-radius:999px;background:#0f172a;color:#bae6fd;padding:3px 7px;font-size:10px;font-weight:900;">${escapeHtml(participant.name || participant.alias || "Unmatched speaker")}</span>`).join("")
            : `<span style="color:#fbbf24;font-size:11px;">No matched participants</span>`;
          const dialogueHtml = (Array.isArray(scene.speaker_assignments) ? scene.speaker_assignments : []).map((cue) => `<div style="display:grid;grid-template-columns:82px minmax(105px,.35fr) minmax(0,1fr);gap:8px;border-top:1px solid #1e3a5f;padding:7px 0;align-items:start;"><div style="color:#67e8f9;font:10px monospace;white-space:nowrap;">${Number(cue.planned_start_seconds || 0).toFixed(2)}–${Number(cue.planned_end_seconds || 0).toFixed(2)}s</div><div style="color:#cffafe;font-size:11px;font-weight:900;overflow-wrap:anywhere;">${escapeHtml(cue.speaker_name || cue.speaker_alias)}${Number(cue.part_count || 1) > 1 ? `<div style="color:#fbbf24;font-size:9px;margin-top:2px;">Split ${cue.part_index}/${cue.part_count}</div>` : ""}</div><div style="color:#e2e8f0;font-size:11px;line-height:1.4;overflow-wrap:anywhere;">${escapeHtml(cue.text)}</div></div>`).join("");
          return `<div style="border:1px solid #334155;border-radius:7px;background:#07111f;padding:9px;margin-top:8px;"><div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;"><div><div style="font-weight:900;color:#cffafe;">Segment ${scene.index}${scene.continuation_of_previous ? ` <span style="color:#fbbf24;font-size:10px;">CONTINUATION</span>` : ""}</div><div style="color:#94a3b8;font:10px monospace;margin-top:3px;">Timeline ${Number(scene.timeline_start_seconds || 0).toFixed(1)}–${Number(scene.timeline_end_seconds || 0).toFixed(1)}s</div><div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:5px;">${participantHtml}</div></div><div style="text-align:right;"><div style="font-weight:900;color:#67e8f9;">${Number(scene.duration_seconds || 0).toFixed(1)}s</div><div style="color:#94a3b8;font-size:10px;">max ${Number(scene.maximum_scene_seconds || 0).toFixed(1)}s</div></div></div><div style="margin-top:7px;">${dialogueHtml}</div></div>`;
        }).join("")
        : `<div style="color:#fca5a5;padding:10px;text-align:center;">No scenes could be planned from the parsed dialogue.</div>`;
      const planWarningsHtml = Array.isArray(scenePlan.warnings) && scenePlan.warnings.length
        ? `<div style="margin-top:8px;color:#fde68a;font-size:11px;line-height:1.4;">${scenePlan.warnings.map((warning) => escapeHtml(warning)).join("<br>")}</div>`
        : "";
      previewPanel.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;">
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Format</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${escapeHtml(String(parsed.format || "text").toUpperCase())}</div></div>
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Speakers</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${speakerCount}</div></div>
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Dialogue cues</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${cueCount}</div></div>
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Words / raw speech</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${wordCount} / ${speechSeconds.toFixed(1)}s</div></div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:11px;">${speakerHtml}</div>
        <div style="margin-top:12px;border:1px solid ${matchedCount === speakerCount && speakerCount ? "#166534" : "#92400e"};border-radius:7px;background:${matchedCount === speakerCount && speakerCount ? "#052e16" : "#291804"};padding:10px;">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:9px;"><div><div style="font-weight:900;color:${matchedCount === speakerCount && speakerCount ? "#bbf7d0" : "#fde68a"};">Speaker matching — ${matchedCount}/${speakerCount} matched</div><div style="font-size:11px;color:#cbd5e1;line-height:1.4;margin-top:3px;">Match each exact script name to the character that should speak it. Clear automatic matches can be changed manually.</div></div><div style="color:#94a3b8;font-size:11px;text-align:right;">${referenceCharacters.length} Reference Builder character${referenceCharacters.length === 1 ? "" : "s"}</div></div>
          ${referenceCharacters.length ? speakerMappingHtml : `<div style="border:1px solid #991b1b;border-radius:6px;background:#3f0808;color:#fecaca;padding:9px;font-size:12px;">No Reference Builder characters are available. Add or save the film characters in Reference Builder, then reopen Script Mapper.</div>`}
        </div>
        <div style="margin-top:12px;border:1px solid #0e7490;border-radius:7px;background:#06283d;padding:10px;">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;"><div><div style="font-weight:900;color:#cffafe;">Timed MiniMax scene plan</div><div style="font-size:11px;color:#bae6fd;line-height:1.4;margin-top:3px;">${plannedSceneCount} segment${plannedSceneCount === 1 ? "" : "s"}; ${Number(scenePlan.estimated_total_seconds || 0).toFixed(1)} estimated total seconds. Each clip stays at or below ${Number(scenePlan.maximum_scene_seconds || currentMaximumSceneSeconds()).toFixed(1)} seconds.</div></div><div style="color:#67e8f9;font-weight:900;white-space:nowrap;">${Number(scenePlan.split_cue_count || 0)} long cue${Number(scenePlan.split_cue_count || 0) === 1 ? "" : "s"} split</div></div>
          ${planWarningsHtml}
          <div style="margin-top:8px;max-height:620px;overflow:auto;padding-right:3px;">${scenePlanHtml}</div>
        </div>
        <div style="margin-top:12px;border:1px solid #334155;border-radius:7px;overflow:auto;max-height:520px;">
          <table style="width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px;">
            <thead><tr style="background:#0f172a;color:#bae6fd;text-align:left;"><th style="width:42px;padding:8px;">#</th><th style="width:58px;padding:8px;">Line</th><th style="width:150px;padding:8px;">Speaker</th><th style="padding:8px;">Exact dialogue</th><th style="width:54px;padding:8px;text-align:right;">Words</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        ${errorsHtml}`;
      previewPanel.querySelectorAll("select[data-script-speaker-index]").forEach((select) => {
        select.onchange = () => {
          const speaker = parsed.speakers[Number(select.dataset.scriptSpeakerIndex || 0)];
          if (!speaker) return;
          const aliasKey = storyboardScriptSpeakerMatchKey(speaker.name);
          speakerMatches.set(aliasKey, {
            subject_id: String(select.value || ""),
            method: select.value ? "manual" : "unmatched",
          });
          renderParsedPreview(parsed);
        };
      });
      sourceStatus.textContent = cueCount
        ? `Parsed ${cueCount} exact cue${cueCount === 1 ? "" : "s"} from ${speakerCount} speaker${speakerCount === 1 ? "" : "s"}; planned ${plannedSceneCount} MiniMax segment${plannedSceneCount === 1 ? "" : "s"} at a ${currentMaximumSceneSeconds()}s maximum. ${matchedCount}/${speakerCount} matched to Reference Builder.${errorCount ? ` Review ${errorCount} issue${errorCount === 1 ? "" : "s"}.` : ""}`
        : "No valid dialogue cues were found.";
      sourceStatus.style.color = cueCount && !errorCount && matchedCount === speakerCount ? "#67e8f9" : "#fbbf24";
    };
    renderEmptyPreview();
    mapperBody.append(sourcePanel, previewPanel);

    const mapperFooter = document.createElement("div");
    mapperFooter.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;padding:12px 18px;background:#0f172a;border-top:1px solid #334155;";
    const footerNote = document.createElement("div");
    footerNote.style.cssText = "color:#94a3b8;font-size:12px;line-height:1.4;";
    footerNote.textContent = "Use This Script makes the exact dialogue authoritative for Guided Film Automation. It still does not change the Video Builder timeline.";
    const footerActions = document.createElement("div");
    footerActions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
    const mapperDone = makeButton("Done", "primary");
    footerActions.append(removeGuidedScriptButton, copyParsedButton, useGuidedScriptButton, mapperDone);
    mapperFooter.append(footerNote, footerActions);
    mapperShell.append(mapperHeader, mapperBody, mapperFooter);
    mapperBackdrop.append(mapperShell);
    document.body.append(mapperBackdrop);

    const closeMapper = () => {
      document.removeEventListener("keydown", onMapperKeyDown, true);
      mapperBackdrop.remove();
    };
    const onMapperKeyDown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      closeMapper();
    };
    mapperCloseTop.onclick = closeMapper;
    mapperDone.onclick = closeMapper;
    mapperBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === mapperBackdrop) closeMapper();
    });
    document.addEventListener("keydown", onMapperKeyDown, true);
    loadScriptButton.onclick = () => scriptFileInput.click();
    maxSceneLengthSelect.onchange = () => {
      customSceneLengthInput.style.display = maxSceneLengthSelect.value === "custom" ? "" : "none";
      if (lastParsed?.cues?.length) renderParsedPreview(lastParsed);
    };
    customSceneLengthInput.onchange = () => {
      customSceneLengthInput.value = String(currentMaximumSceneSeconds());
      if (lastParsed?.cues?.length) renderParsedPreview(lastParsed);
    };
    scriptFileInput.onchange = async () => {
      const file = scriptFileInput.files?.[0];
      if (!file) return;
      try {
        scriptInput.value = await file.text();
        sourceStatus.textContent = `Loaded ${file.name}. Parsing preview...`;
        renderParsedPreview(parseStoryboardScriptImport(scriptInput.value));
      } catch (error) {
        sourceStatus.textContent = `Could not read ${file.name}: ${String(error?.message || error)}`;
        sourceStatus.style.color = "#fca5a5";
      } finally {
        scriptFileInput.value = "";
      }
    };
    parseScriptButton.onclick = () => renderParsedPreview(parseStoryboardScriptImport(scriptInput.value));
    clearScriptButton.onclick = () => {
      scriptInput.value = "";
      lastParsed = null;
      speakerMatches.clear();
      copyParsedButton.disabled = true;
      sourceStatus.textContent = "";
      renderEmptyPreview();
      scriptInput.focus();
    };
    copyParsedButton.onclick = async () => {
      if (!lastParsed?.cues?.length) return;
      await copyTextToClipboard(JSON.stringify(lastParsed, null, 2));
      createToast("Parsed script JSON copied. No timeline changes were made.");
    };
    useGuidedScriptButton.onclick = async () => {
      if (!lastParsed?.cues?.length || useGuidedScriptButton.disabled) return;
      useGuidedScriptButton.disabled = true;
      try {
        state.scriptImport = normalizeStoryboardScriptImportState({
          enabled: true,
          authoritative: true,
          format: lastParsed.format,
          raw_text: scriptInput.value,
          imported_at: new Date().toISOString(),
          maximum_scene_seconds: currentMaximumSceneSeconds(),
          cues: lastParsed.cues,
        });
        refreshSetupPanelSummaries();
        notifyStoryboardDefaultsChanged();
        if (state.projectFolder) {
          await postJson("/vrgdg/storyboard/save", {
            project_folder: state.projectFolder,
            storyboard: slimStoryboardForRequest(state),
          });
        }
        closeMapper();
        createToast(`Authoritative script activated: ${state.scriptImport.cues.length} exact cue${state.scriptImport.cues.length === 1 ? "" : "s"} across ${state.scriptImport.scene_plan.scene_count} planned MiniMax segment${state.scriptImport.scene_plan.scene_count === 1 ? "" : "s"}.`);
      } catch (error) {
        sourceStatus.textContent = `Could not activate the script: ${String(error?.message || error)}`;
        sourceStatus.style.color = "#fca5a5";
        useGuidedScriptButton.disabled = false;
      }
    };
    removeGuidedScriptButton.onclick = async () => {
      if (!window.confirm("Remove the authoritative imported script from Guided Film Automation?\n\nThis does not delete existing Storyboard or Video Builder scenes.")) return;
      removeGuidedScriptButton.disabled = true;
      try {
        state.scriptImport = normalizeStoryboardScriptImportState({});
        refreshSetupPanelSummaries();
        notifyStoryboardDefaultsChanged();
        if (state.projectFolder) {
          await postJson("/vrgdg/storyboard/save", {
            project_folder: state.projectFolder,
            storyboard: slimStoryboardForRequest(state),
          });
        }
        closeMapper();
        createToast("Authoritative script removed. Existing scenes were not changed.");
      } catch (error) {
        sourceStatus.textContent = `Could not remove the active script: ${String(error?.message || error)}`;
        sourceStatus.style.color = "#fca5a5";
        removeGuidedScriptButton.disabled = false;
      }
    };
    if (existingScriptImport.enabled && scriptInput.value.trim()) {
      renderParsedPreview(parseStoryboardScriptImport(scriptInput.value));
    } else {
      scriptInput.focus();
    }
  };

  const planFilmDialogueScenesWithLlm = async () => {
    if (!isIdLoraMode && !isMiniMaxShortFilmMode) return;
    if (isFullyCustomShortFilm()) {
      createToast("Fully Custom uses your manual scene cards. Switch to Guided Film Automation to ask the LLM to plan dialogue scenes.");
      return;
    }
    syncStoryLayerFromInputs();
    const authoritativeScript = normalizeStoryboardScriptImportState(state.scriptImport);
    const sceneCount = authoritativeScript.enabled
      ? Math.max(1, Math.min(80, Number(authoritativeScript.scene_plan.scene_count || 1)))
      : Math.max(1, Math.min(24, Number(idLoraDialogueSceneCount.value || 6)));
    idLoraDialogueSceneCount.value = String(sceneCount);
    const plannerLabel = isIdLoraMode ? "ID-LoRA Dialogue Scenes" : "MiniMax Short Film Scenes";
    const progress = createStoryboardProgressWindow(plannerLabel);
    try {
      progress.set(`Planning ${sceneCount} ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} dialogue scene${sceneCount === 1 ? "" : "s"} with ${promptRunnerName()}...`, 8);
      const data = await postJson(isIdLoraMode ? "/vrgdg/storyboard/id_lora_dialogue_scenes" : "/vrgdg/storyboard/minimax_dialogue_scenes", {
        ...(state.gemmaSettings || {}),
        story_source: authoritativeScript.enabled
          ? authoritativeScript.raw_text
          : [userStoryArcInput.value, songStoryBriefInput.value].map((item) => String(item || "").trim()).filter(Boolean).join("\n\n"),
        script_import: authoritativeScript,
        story_layer: normalizeStoryLayer(state.storyLayer),
        reference_builder: state.referenceBuilder || {},
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        storyboard: slimStoryboardForRequest(state),
        scene_count: sceneCount,
        project_video_engine: state.projectVideoEngine,
        minimax_h3_mode: state.miniMaxH3Mode,
        video_prompt_type: isIdLoraMode ? "id_lora" : state.videoPromptType,
        short_film_planning_mode: "guided_film",
        performance_mode: "speaking",
        unload_after: true,
        max_new_tokens: Math.max(2200, sceneCount * 520),
        temperature: 0.55,
        top_p: 0.92,
      }, 240000);
      const generated = Array.isArray(data.scenes) ? data.scenes : [];
      if (!generated.length) throw new Error(`${promptRunnerName()} returned no dialogue scenes.`);
      state.scenes = generated.map((scene, index) => {
        const normalized = normalizeScene({
          ...scene,
          video_prompt_type: isIdLoraMode ? "id_lora" : state.videoPromptType,
          project_video_engine: isIdLoraMode ? "ltx" : "minimax_h3",
          minimax_h3_mode: isIdLoraMode ? "" : state.miniMaxH3Mode,
          performance_mode: "speaking",
        }, index);
        normalized.id_lora_character_id = scene.id_lora_character_id || scene.character_id || scene.subject_id || "";
        normalized.id_lora_location_id = scene.id_lora_location_id || scene.location_id || "";
        return normalized;
      });
      state.selected.clear();
      if (String(data.premise || "").trim() && !String(songStoryBriefInput.value || "").trim()) {
        state.storyLayer.song_story_brief = String(data.premise || "").trim();
        songStoryBriefInput.value = state.storyLayer.song_story_brief;
      }
      setMode("storyboard_prompts");
      renderTable();
      refreshSetupPanelSummaries();
      progress.set(`Storyboard scenes ready.\nCreated ${state.scenes.length} editable storyboard scene${state.scenes.length === 1 ? "" : "s"}. The Video Builder timeline has not been changed.`, 96);
      await saveStoryboard();
      progress.set(`Storyboard saved for review.\nNext: click Create ${state.scenes.length} Timeline Segment${state.scenes.length === 1 ? "" : "s"} to build the Video Builder timeline.`, 100);
      progress.close(1800);
      createToast(`Created ${state.scenes.length} ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard scene${state.scenes.length === 1 ? "" : "s"}. The timeline is unchanged until you click Create ${state.scenes.length} Timeline Segment${state.scenes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      progress.set(`${isIdLoraMode ? "ID-LoRA" : "MiniMax"} dialogue planning failed:\n${String(error?.message || error)}`, 100);
      createToast(`${isIdLoraMode ? "ID-LoRA" : "MiniMax"} dialogue planning failed:\n${String(error?.message || error)}`, true);
    }
  };

  const applyFilmDialoguePlanToVideoBuilder = async () => {
    const applyCallback = isIdLoraMode ? state.onApplyIdLoraDialoguePlan : state.onApplyMiniMaxDialoguePlan;
    if (!applyCallback) return;
    const scenes = state.scenes
      .map((scene, index) => slimSceneForRequest(scene, index))
      .filter((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim());
    if (!scenes.length) {
      createToast(`No reviewed ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard scenes were found to create timeline segments.`, true);
      return;
    }
    const confirmed = window.confirm(`Create ${scenes.length} Video Builder timeline segment${scenes.length === 1 ? "" : "s"} from the reviewed ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard scenes?\n\nThis is the step that creates the timeline. Export Prompt Files Only does not create timeline segments.\n\nThe blank starter scene will be replaced. If real scenes already exist, Video Builder will ask before replacing them.`);
    if (!confirmed) return;
    try {
      applyDialoguePlanButton.disabled = true;
      const result = await applyCallback({
        story_layer: normalizeStoryLayer(state.storyLayer),
        short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
        scenes,
      });
      createToast(result?.message || `Created ${scenes.length} Video Builder timeline segment${scenes.length === 1 ? "" : "s"} from the reviewed ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard.`);
    } catch (error) {
      createToast(`Create Timeline Segments failed:\n${String(error?.message || error)}`, true);
    } finally {
      applyDialoguePlanButton.disabled = false;
    }
  };

  const applyCameraFlow = ({ overwrite = false } = {}) => {
    if (state.mode !== "image_to_video_prep") {
      createToast("Auto camera flow is only available in Video Prep.");
      return;
    }
    const profileKey = state.cameraFlow || "balanced";
    if (profileKey === "off") {
      createToast("Auto camera flow is off.");
      return;
    }
    let previousMotion = "";
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const entry = cameraFlowEntryForScene(profileKey, index, previousMotion);
      if (!entry) return;
      const hadShot = Boolean(String(scene.shot_type || "").trim());
      const hadCamera = Boolean(String(scene.camera_motion || "").trim());
      if ((overwrite || !hadShot) && entry.shot) {
        scene.shot_type = entry.shot;
        changed += 1;
      }
      if ((overwrite || !hadCamera) && entry.camera) {
        scene.camera_motion = entry.camera;
        changed += 1;
      }
      previousMotion = String(scene.camera_motion || entry.camera || previousMotion);
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Auto camera flow replaced ${changed} field${changed === 1 ? "" : "s"}.` : "No camera fields were changed.");
    } else {
      createToast(changed ? `Auto camera flow filled ${changed} blank field${changed === 1 ? "" : "s"}.` : "No blank shot or camera fields needed filling.");
    }
  };

  const applyImageShotFlow = ({ overwrite = false } = {}) => {
    if (state.mode === "image_to_video_prep") {
      createToast("Still shot flow is only available in Image Prep.");
      return;
    }
    const profileKey = state.imageShotFlow || "intimate";
    if (profileKey === "off") {
      createToast("Still shot flow is off.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const sequence = imageShotFlowPresetForMode(profileKey).sequence || [];
      const shot = sequence[index % sequence.length] || "";
      if (!shot) return;
      if (!overwrite && String(scene.shot_type || "").trim()) return;
      scene.shot_type = shot;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Still shot flow replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No shot fields were changed.");
    } else {
      createToast(changed ? `Still shot flow filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank shot fields needed filling.");
    }
  };

  const applyImageAesthetic = ({ overwrite = false } = {}) => {
    if (state.mode === "image_to_video_prep") {
      createToast("Image aesthetic is only available in Image Prep.");
      return;
    }
    const preset = imageAestheticPresetForMode(state.imageAesthetic);
    const value = String(preset.description || "").trim();
    if (!value) {
      createToast("Choose an image aesthetic first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      const existing = String(scene.motion_summary || "");
      const hasAesthetic = existing.split(/\r?\n/).some((line) => line.trim().toLowerCase().startsWith("image aesthetic:"));
      if (!overwrite && hasAesthetic) return;
      scene.motion_summary = replaceLabeledPlanningLine(existing, "Image aesthetic", value);
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Image aesthetic replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No image aesthetic notes were changed.");
    } else {
      createToast(changed ? `Image aesthetic filled ${changed} scene${changed === 1 ? "" : "s"}.` : "No blank image aesthetic notes needed filling.");
    }
  };

  const applyPerformanceStyle = ({ overwrite = false } = {}) => {
    const value = String(state.performanceStyle || "").trim();
    if (!value) {
      createToast(isIdLoraMode ? "Choose a global acting style first." : "Choose a global performance style first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      if (!overwrite && String(scene.performance_style || "").trim()) return;
      scene.performance_style = value;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `${isIdLoraMode ? "Acting" : "Performance"} style replaced ${changed} scene${changed === 1 ? "" : "s"}.` : `No ${isIdLoraMode ? "acting" : "performance"} style fields were changed.`);
    } else {
      createToast(changed ? `${isIdLoraMode ? "Acting" : "Performance"} style filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : `No blank ${isIdLoraMode ? "acting" : "performance"} style fields needed filling.`);
    }
  };

  const applyFacialPerformance = ({ overwrite = false } = {}) => {
    const value = String(state.facialPerformance || "").trim();
    const custom = String(state.facialPerformanceCustom || "").trim();
    if (!value && !custom) {
      createToast("Choose a global facial performance preset or enter custom facial text first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      const hasPreset = String(scene.facial_performance || "").trim();
      const hasCustom = String(scene.facial_performance_custom || "").trim();
      if (!overwrite && (hasPreset || hasCustom)) return;
      scene.facial_performance = value;
      scene.facial_performance_custom = custom;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Facial performance replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No facial performance fields were changed.");
    } else {
      createToast(changed ? `Facial performance filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank facial performance fields needed filling.");
    }
  };

  const confirmClearStoryboardPrompts = () => new Promise((resolve) => {
    const confirmBackdrop = document.createElement("div");
    confirmBackdrop.style.cssText = "position:fixed;inset:0;z-index:100040;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:22px;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(620px,calc(100vw - 44px));border:1px solid #991b1b;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "padding:14px 16px;background:#3f0808;border-bottom:1px solid #991b1b;font-weight:900;color:#fecaca;";
    header.textContent = "Clear all Storyboard prompts and notes?";
    const body = document.createElement("div");
    body.style.cssText = "padding:16px;line-height:1.45;color:#e2e8f0;font-size:13px;";
    body.textContent = "This clears prompt summaries, generated image/video prompts, and extra notes inside every scene card. It keeps lyrics, subjects, locations, reference images, shot type, camera motion, character motion, performance style, and microphone settings.";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 16px 16px;";
    const cancel = makeButton("Cancel");
    const clear = makeButton("Yes, clear prompts", "primary");
    clear.style.borderColor = "#991b1b";
    clear.style.background = "#991b1b";
    actions.append(cancel, clear);
    panel.append(header, body, actions);
    confirmBackdrop.append(panel);
    document.body.append(confirmBackdrop);
    const closeConfirm = (value) => {
      confirmBackdrop.remove();
      resolve(value);
    };
    cancel.onclick = () => closeConfirm(false);
    clear.onclick = () => closeConfirm(true);
    confirmBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === confirmBackdrop) closeConfirm(false);
    });
  });

  const clearAllStoryboardPrompts = async () => {
    const confirmed = await confirmClearStoryboardPrompts();
    if (!confirmed) return;
    let changed = 0;
    for (const scene of state.scenes) {
      const before = [
        scene.prompt_summary,
        scene.motion_summary,
        scene.image_prompt,
        scene.video_prompt,
        scene.notes,
      ].map((value) => String(value || "")).join("\n");
      scene.prompt_summary = "";
      scene.motion_summary = "";
      scene.image_prompt = "";
      scene.video_prompt = "";
      scene.video_prompt_origin = "manual";
      scene.notes = "";
      if (scene.status && scene.status !== "draft") scene.status = "draft";
      const after = [
        scene.prompt_summary,
        scene.motion_summary,
        scene.image_prompt,
        scene.video_prompt,
        scene.notes,
      ].map((value) => String(value || "")).join("\n");
      if (before !== after) changed += 1;
    }
    renderTable();
    syncReferenceMappingsToVideoCreator();
    if (state.projectFolder) {
      try {
        await postJson("/vrgdg/storyboard/save", {
          project_folder: state.projectFolder,
          storyboard: slimStoryboardForRequest(state),
        });
        createToast(`Cleared prompts/notes in ${changed} scene${changed === 1 ? "" : "s"} and saved Storyboard.`);
      } catch (error) {
        createToast(`Cleared prompts/notes in this session, but could not save Storyboard:\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast(`Cleared prompts/notes in ${changed} scene${changed === 1 ? "" : "s"}. Save the project to keep this change.`);
    }
  };

  const currentRows = () => {
    const q = state.query.trim().toLowerCase();
    if (!q) return state.scenes;
    return state.scenes.filter((scene) => [
      scene.label,
      scene.lyrics,
      scene.lyric_section,
      scene.story_beat,
      scene.prompt_summary,
      scene.motion_summary,
      scene.setting,
      scene.shot_type,
      ...(scene.subjects || []),
    ].join(" ").toLowerCase().includes(q));
  };

  function syncReferenceMappingsToVideoCreator() {
    if (!state.onReferenceMappingsChanged) return;
    state.onReferenceMappingsChanged({
      reference_builder: normalizeReferenceBuilderCatalog(state.referenceBuilder),
      scenes: state.scenes.map((scene) => ({
        id: scene.id,
        scene_number: scene.scene_number,
        no_character_present: Boolean(scene.no_character_present),
        subject_ids: scene.no_character_present ? [] : (Array.isArray(scene.subject_refs) ? scene.subject_refs : []).map((ref) => String(ref?.id || "")).filter(Boolean),
        location_id: String(scene.location_ref?.id || ""),
        trigger: String(scene.trigger_phrase || ""),
        trigger_position: String(scene.trigger_position || "start") === "end" ? "end" : "start",
        speaker_assignments: normalizeStoryboardSpeakerAssignments(scene.speaker_assignments),
        lyric_text: String(scene.lyrics || ""),
        lyric_singers: Array.isArray(scene.lyric_singers) ? [...scene.lyric_singers] : [],
      })),
    });
  }

  const videoPromptTypeLabel = (type) => {
    if (type === "text_to_video") return "H3 T2V";
    if (type === "image_to_video") return "H3 I2V";
    if (type === "reference_to_video") return "H3 Reference";
    if (type === "video_to_video") return "H3 V2V";
    if (type === "id_lora") return "ID-LoRA I2V";
    if (type === "t2v") return "T2V";
    if (type === "rtv") return "RTV";
    return "I2V";
  };

  const confirmClearAllStoryBeats = () => new Promise((resolve) => {
    const confirmBackdrop = document.createElement("div");
    confirmBackdrop.style.cssText = "position:fixed;inset:0;z-index:100040;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:22px;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(620px,calc(100vw - 44px));border:1px solid #991b1b;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.6);overflow:hidden;";
    const confirmHeader = document.createElement("div");
    confirmHeader.style.cssText = "padding:14px 16px;background:#3f0808;border-bottom:1px solid #991b1b;font-weight:900;color:#fecaca;";
    confirmHeader.textContent = "Clear all Storyboard story beats?";
    const body = document.createElement("div");
    body.style.cssText = "padding:16px;line-height:1.45;color:#e2e8f0;font-size:13px;";
    body.textContent = "This clears only the Scene Story Beat field in every scene. Lyrics, generated prompts, notes, images, subjects, locations, references, camera settings, and motion settings will remain unchanged.";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 16px 16px;";
    const cancel = makeButton("Cancel");
    const clear = makeButton("Yes, clear story beats", "primary");
    clear.style.borderColor = "#991b1b";
    clear.style.background = "#991b1b";
    const finish = (value) => { confirmBackdrop.remove(); resolve(value); };
    cancel.onclick = () => finish(false);
    clear.onclick = () => finish(true);
    confirmBackdrop.addEventListener("pointerdown", (event) => { if (event.target === confirmBackdrop) finish(false); });
    actions.append(cancel, clear);
    panel.append(confirmHeader, body, actions);
    confirmBackdrop.append(panel);
    document.body.append(confirmBackdrop);
  });

  const clearAllStoryboardStoryBeats = async () => {
    if (!await confirmClearAllStoryBeats()) return;
    let changed = 0;
    for (const scene of state.scenes) {
      if (String(scene.story_beat || "").trim()) changed += 1;
      scene.story_beat = "";
    }
    renderTable();
    if (state.onStoryLayerChanged) {
      state.onStoryLayerChanged({
        scenes: state.scenes.map((scene) => ({
          id: scene.id,
          scene_number: scene.scene_number,
          story_beat: "",
        })),
      });
    }
    if (state.projectFolder) {
      try {
        await postJson("/vrgdg/storyboard/save", {
          project_folder: state.projectFolder,
          storyboard: slimStoryboardForRequest(state),
        });
        createToast(`Cleared story beats in ${changed} scene${changed === 1 ? "" : "s"} and saved Storyboard.`);
      } catch (error) {
        createToast(`Cleared story beats in this session, but could not save Storyboard:\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast(`Cleared story beats in ${changed} scene${changed === 1 ? "" : "s"}. Save the project to keep this change.`);
    }
  };

  const videoPromptTypeHint = (type) => {
    if (type === "text_to_video") return "MiniMax H3 Text to Video uses scene text and <Audio 1> without picture or video references.";
    if (type === "image_to_video") return "MiniMax H3 Image to Video uses the scene image as <Picture 1> and the authoritative opening frame.";
    if (type === "reference_to_video") return "MiniMax H3 Reference to Video uses this scene's ordered Reference Builder pictures and their exact <Picture N> tags.";
    if (type === "video_to_video") return "MiniMax H3 Video to Video uses the reference-video paths and purposes configured for this scene in Video Builder.";
    if (type === "id_lora") {
      return "ID-LoRA uses a scene image plus per-scene dialogue and a character voice sample from the ID-LoRA Ref Builder.";
    }
    if (type === "t2v") {
      return "T2V has no first frame, so choose an opening shot and describe the motion clearly.";
    }
    if (type === "rtv") {
      return "Reference to Video uses subject/location references plus an opening shot and motion direction.";
    }
    return "I2V already has a first frame, so use this mostly for camera movement, framing, and continuity.";
  };

  const chooseStoryboardImageFile = () => new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    document.body.append(input);
    input.onchange = () => {
      const file = input.files?.[0] || null;
      input.remove();
      resolve(file);
    };
    input.click();
  });

  const promptStoryboardReferenceDetails = ({ kind, file, defaultName = "", defaultDescription = "" } = {}) => new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:12px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.58);overflow:hidden;";
    const title = kind === "location" ? "Add Location Reference" : "Add Subject Reference";
    box.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;background:#083344;border-bottom:1px solid #155e75;">
        <div>
          <div style="font-size:18px;font-weight:900;color:#cffafe;">${escapeHtml(title)}</div>
          <div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Name and describe this image so Storyboard Builder and Reference Builder can both use it.</div>
        </div>
      </div>
    `;
    const body = document.createElement("div");
    body.style.cssText = "padding:16px;display:flex;flex-direction:column;gap:12px;";
    const name = makeInput(defaultName || String(file?.name || "").replace(/\.[^.]+$/, ""), kind === "location" ? "Location name" : "Subject name");
    const description = makeTextarea(defaultDescription, kind === "location" ? "Location description..." : "Subject description...", 5);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Use Image", "primary");
    actions.append(cancel, save);
    body.append(
      (() => {
        const preview = document.createElement("div");
        preview.style.cssText = "height:150px;border:1px dashed #155e75;border-radius:10px;background:#07111f center/contain no-repeat;";
        return preview;
      })(),
      (() => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:900;color:#cbd5e1;";
        wrap.append("Name", name);
        return wrap;
      })(),
      (() => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:900;color:#cbd5e1;";
        wrap.append("Description", description);
        return wrap;
      })(),
      actions,
    );
    box.append(body);
    backdrop.append(box);
    document.body.append(backdrop);
    const preview = body.firstChild;
    readStoryboardImageFile(file)
      .then((dataUrl) => {
        preview.style.backgroundImage = `url("${dataUrl}")`;
        save.onclick = () => {
          const cleanName = String(name.value || "").trim();
          if (!cleanName) {
            createToast("Give this reference a name first.", true);
            return;
          }
          backdrop.remove();
          resolve({
            name: cleanName,
            description: String(description.value || "").trim(),
            image: { path: "", data: dataUrl, name: String(file?.name || cleanName) },
          });
        };
      })
      .catch((error) => {
        backdrop.remove();
        createToast(String(error?.message || error), true);
        resolve(null);
      });
    cancel.onclick = () => {
      backdrop.remove();
      resolve(null);
    };
  });

  const upsertStoryboardReference = (kind, reference) => {
    if (!reference) return null;
    const list = kind === "location" ? state.referenceBuilder.locations : state.referenceBuilder.subjects;
    const name = String(reference.name || "").trim();
    const existing = list.find((item) => String(item.name || "").trim().toLowerCase() === name.toLowerCase());
    const merged = {
      ...(existing || {}),
      ...reference,
      id: existing?.id || reference.id || storyboardReferenceId(kind === "location" ? "loc" : "subj", name),
      name,
      description: String(reference.description || existing?.description || ""),
      image: reference.image || existing?.image || { path: "", data: "", name: "" },
    };
    if (existing) {
      Object.assign(existing, merged);
      return existing;
    }
    list.push(merged);
    return merged;
  };

  const addStoryboardReferenceFromFile = async (kind, scene) => {
    const file = await chooseStoryboardImageFile();
    if (!file) return null;
    const details = await promptStoryboardReferenceDetails({ kind, file });
    if (!details) return null;
    let reference = details;
    if (state.projectFolder) {
      try {
        const saved = await postJson("/vrgdg/storyboard/import_reference_image", {
          project_folder: state.projectFolder,
          kind,
          name: details.name,
          description: details.description,
          image_data: details.image?.data || "",
          file_name: details.image?.name || file.name || details.name,
        }, 120000);
        reference = saved.reference || reference;
      } catch (error) {
        createToast(`Could not save this reference image into the project folder. It will stay in this session only.\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast("Save the AI Video Builder project first if you want imported Storyboard references to persist.", true);
    }
    const ref = upsertStoryboardReference(kind, reference);
    if (!ref || !scene) return ref;
    if (kind === "location") {
      scene.location_ref = ref;
      scene.setting = ref.description || ref.name || scene.setting || "";
    } else {
      const refs = Array.isArray(scene.subject_refs) ? scene.subject_refs.slice() : [];
      if (!refs.some((item) => String(item.id || "") === String(ref.id || ""))) refs.push(ref);
      scene.subject_refs = refs;
      scene.subjects = storyboardSubjectNamesFromRefs(refs);
    }
    syncReferenceMappingsToVideoCreator();
    renderTable();
    createToast(`${kind === "location" ? "Location" : "Subject"} reference added to ${scene.label || `Scene ${scene.scene_number}`}.`);
    return ref;
  };

  const applyVideoStyle = ({ overwrite = false } = {}) => {
    if (state.mode !== "image_to_video_prep") {
      createToast("Video style is only available in Video Prep.");
      return;
    }
    const value = String(state.videoStyle || "").trim();
    if (!value) {
      createToast("Choose a video style first.");
      return;
    }
    if (value === "custom" && !String(state.videoStyleCustom || "").trim()) {
      createToast("Type the exact custom style wording first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      if (!storyboardSceneSupportsVideoStyle(scene)) return;
      if (!overwrite && String(scene.video_style || "").trim()) return;
      scene.video_style = value;
      scene.video_style_custom = value === "custom" ? String(state.videoStyleCustom || "").trim() : "";
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Video style replaced ${changed} eligible scene${changed === 1 ? "" : "s"}.` : "No eligible video scene styles were changed.");
    } else {
      createToast(changed ? `Video style filled ${changed} eligible scene${changed === 1 ? "" : "s"}.` : "No blank eligible video scene styles needed filling.");
    }
  };

  const openStoryboardSubjectPicker = (scene) => {
    if (!scene) return;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:22px;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(980px,calc(100vw - 44px));max-height:calc(100vh - 48px);overflow:auto;border:1px solid #155e75;border-radius:10px;background:#0b1220;color:#f8fafc;padding:14px;display:flex;flex-direction:column;gap:12px;box-shadow:0 24px 80px rgba(0,0,0,.62);";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
    const heading = document.createElement("div");
    heading.textContent = `${scene.scene_number || 1}. ${scene.label || `Scene ${scene.scene_number || 1}`} — Characters Present`;
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Cancel");
    header.append(heading, close);

    const choices = document.createElement("div");
    choices.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;";
    const selected = new Set(
      (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
        .map((subject) => String(subject?.id || ""))
        .filter(Boolean),
    );
    const availableSubjects = Array.isArray(state.referenceBuilder.subjects)
      ? state.referenceBuilder.subjects
      : [];

    const renderChoices = () => {
      choices.replaceChildren();
      const clearSelection = document.createElement("button");
      clearSelection.type = "button";
      clearSelection.textContent = "Clear selection";
      clearSelection.style.cssText = `min-height:132px;border:2px dashed ${selected.size ? "#475569" : "#22d3ee"};border-radius:8px;background:${selected.size ? "#111827" : "#083344"};color:#94a3b8;cursor:pointer;font-weight:900;`;
      clearSelection.onclick = () => {
        selected.clear();
        renderChoices();
      };
      choices.append(clearSelection);

      availableSubjects.forEach((subject) => {
        const subjectId = String(subject.id || "");
        if (!subjectId) return;
        const active = selected.has(subjectId);
        const card = document.createElement("button");
        card.type = "button";
        card.style.cssText = `min-height:132px;border:2px solid ${active ? "#22d3ee" : "#334155"};border-radius:8px;background:${active ? "#083344" : "#111827"};color:#f8fafc;padding:8px;display:flex;flex-direction:column;gap:7px;align-items:center;cursor:pointer;`;
        const preview = document.createElement("div");
        preview.style.cssText = "width:96px;height:72px;border:1px solid #155e75;border-radius:6px;background:#061620;overflow:hidden;display:flex;align-items:center;justify-content:center;flex:0 0 auto;";
        const imageSource = storyboardReferenceImageSrc(subject.image || {});
        if (imageSource) {
          const image = document.createElement("img");
          image.src = imageSource;
          image.alt = subject.name || "Subject reference";
          image.draggable = false;
          image.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
          preview.append(image);
        } else {
          const empty = document.createElement("span");
          empty.textContent = "No image";
          empty.style.cssText = "font-size:11px;font-weight:900;color:#67e8f9;";
          preview.append(empty);
        }
        const name = document.createElement("div");
        name.textContent = subject.name || "Subject";
        name.style.cssText = "font-size:12px;font-weight:900;text-align:center;line-height:1.25;";
        card.append(preview, name);
        card.onclick = () => {
          if (selected.has(subjectId)) selected.delete(subjectId);
          else selected.add(subjectId);
          renderChoices();
        };
        choices.append(card);
      });

      if (!availableSubjects.length) {
        const empty = document.createElement("div");
        empty.textContent = "No subjects are in Reference Builder yet. Use Upload New Subject below to add the first one.";
        empty.style.cssText = "grid-column:1/-1;border:1px dashed #334155;border-radius:8px;padding:18px;color:#94a3b8;text-align:center;font-size:12px;";
        choices.append(empty);
      }
    };

    const footer = document.createElement("div");
    footer.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
    const upload = makeButton("Upload New Subject");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Apply Selection", "primary");
    const dismiss = () => backdrop.remove();
    close.onclick = dismiss;
    cancel.onclick = dismiss;
    upload.onclick = async () => {
      dismiss();
      await addStoryboardReferenceFromFile("subject", scene);
    };
    apply.onclick = () => {
      const selectedSubjects = availableSubjects.filter((subject) => selected.has(String(subject.id || "")));
      scene.subject_refs = selectedSubjects;
      scene.subjects = storyboardSubjectNamesFromRefs(selectedSubjects);
      if (selectedSubjects.length) scene.no_character_present = false;
      syncReferenceMappingsToVideoCreator();
      dismiss();
      renderTable();
      createToast(selectedSubjects.length
        ? `${selectedSubjects.length} subject${selectedSubjects.length === 1 ? "" : "s"} mapped to ${scene.label || `Scene ${scene.scene_number}`}.`
        : `Subject mapping cleared for ${scene.label || `Scene ${scene.scene_number}`}.`);
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) dismiss();
    });
    footer.append(upload, cancel, apply);
    panel.append(header, choices, footer);
    backdrop.append(panel);
    document.body.append(backdrop);
    renderChoices();
  };

  const openSceneEditor = (scene) => {
    const isVideoPrepMode = state.mode === "image_to_video_prep";
    const isImagePrepMode = !isVideoPrepMode;
    const editorSceneIndex = state.scenes.findIndex((item) => item.id === scene.id);
    const inheritedFlfStart = editorSceneIndex > 0 ? String(state.scenes[editorSceneIndex - 1]?.flf_end_state || "").trim() : "";
    if ((state.videoPromptType === "flf" || scene.video_prompt_type === "flf") && inheritedFlfStart) scene.flf_start_state = inheritedFlfStart;
    absorbSceneReferencesIntoCatalog([scene]);
    const editorBackdrop = document.createElement("div");
    editorBackdrop.style.cssText = "position:fixed;inset:0;z-index:100012;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const editor = document.createElement("div");
    editor.style.cssText = "width:min(1420px,calc(100vw - 42px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #0e7490;border-radius:16px;background:linear-gradient(135deg,#07111f,#0f172a 46%,#071827);color:#f8fafc;box-shadow:0 28px 90px rgba(0,0,0,.68);padding:18px;display:flex;flex-direction:column;gap:12px;";
    const label = makeInput(scene.label, "Scene label");
    const lyricSection = makeInput(scene.lyric_section || "", "Verse 1, Chorus, Bridge, Outro...");
    const lyrics = makeTextarea(scene.lyrics, "Lyrics, script, or beat for this scene...", 4);
    const storyBeat = makeTextarea(scene.story_beat || "", "Scene story beat for this scene...", 4);
    const flfStartState = makeTextarea(scene.flf_start_state || "", "What must be visible in this scene's first frame...", 3);
    const flfTransformation = makeTextarea(scene.flf_transformation || "", "What changes continuously between the two frames...", 3);
    const flfEndState = makeTextarea(scene.flf_end_state || "", "What must be visible in this scene's last frame...", 3);
    const flfCarryForward = makeTextarea(scene.flf_carry_forward || "", "Continuity details the next scene should inherit...", 3);
    if ((state.videoPromptType === "flf" || scene.video_prompt_type === "flf") && editorSceneIndex > 0) {
      flfStartState.readOnly = true;
      flfStartState.title = "Automatically inherited from the previous scene's end-frame state.";
      flfStartState.style.opacity = "0.78";
    }
    const summary = makeTextarea(scene.prompt_summary, "Image prompt summary...", 3);
    const motion = makeTextarea(
      scene.motion_summary,
      isImagePrepMode ? "Still photography notes..." : "Custom motion, camera, action, or LLM direction...",
      3,
    );
    const cameraGroups = isImagePrepMode ? STILL_CAMERA_STYLE_GROUPS : CAMERA_MOTION_GROUPS;
    const cameraMotionOptions = cameraGroups.flatMap((group) => group.options || []);
    const cameraMotionValue = scene.camera_motion || cameraMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const cameraMotionPreset = makeGroupedSelect(cameraGroups, cameraMotionValue);
    const customCameraMotion = makeInput(scene.camera_motion || "", isImagePrepMode ? "Custom still camera style" : "Custom camera motion");
    const characterMotionOptions = CHARACTER_MOTION_GROUPS.flatMap((group) => group.options || []);
    const characterMotionValue = scene.character_motion || characterMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const characterMotionPreset = makeGroupedSelect(CHARACTER_MOTION_GROUPS, characterMotionValue);
    const customCharacterMotion = makeInput(scene.character_motion || "", "Custom character motion");
    const performanceStyle = makeSelect(performanceStylePresets, scene.performance_style || "");
    const videoStyle = makeSelect(MINIMAX_VIDEO_STYLE_PRESETS, state.videoStyle || scene.video_style || "");
    videoStyle.disabled = Boolean(state.videoStyle);
    videoStyle.title = state.videoStyle ? "The global Video style is required for every eligible scene." : "Choose a style for this scene.";
    const videoStyleCustom = makeTextarea(
      state.videoStyle === "custom" ? state.videoStyleCustom : (scene.video_style_custom || state.videoStyleCustom || ""),
      "Type the exact style wording that must appear unchanged in this scene's prompt...",
      3,
    );
    videoStyleCustom.disabled = Boolean(state.videoStyle);
    const temporalEffectOverride = makeSelect([
      { value: "global", label: "Use global temporal effect" },
      { value: "off", label: "Off for this scene" },
      ...MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.filter((item) => item.value).map((item) => ({ value: item.value, label: item.label })),
    ], scene.temporal_world_effect_override || "global");
    const temporalEffectCustom = makeTextarea(scene.temporal_world_effect_custom || "", "Exact custom temporal behavior for only this scene...", 3);
    const facialPerformance = makeSelect(facialPerformancePresets, scene.facial_performance || "");
    const facialPerformanceCustom = makeTextarea(scene.facial_performance_custom || "", "Optional custom facial expression/movement text for this scene...", 3);
    const includeMicLabel = document.createElement("label");
    includeMicLabel.style.cssText = "display:flex;align-items:center;gap:8px;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:9px 10px;font-size:12px;font-weight:900;";
    const includeMic = document.createElement("input");
    includeMic.type = "checkbox";
    includeMic.checked = Boolean(scene.include_microphone);
    includeMicLabel.append(includeMic, document.createTextNode("Include microphone in prompt"));
    const noCharacterLabel = document.createElement("label");
    noCharacterLabel.style.cssText = includeMicLabel.style.cssText;
    const noCharacterInput = document.createElement("input");
    noCharacterInput.type = "checkbox";
    noCharacterInput.checked = Boolean(scene.no_character_present);
    noCharacterLabel.append(noCharacterInput, document.createTextNode("No character present"));
    const miniMaxProject = state.projectVideoEngine === "minimax_h3";
    const videoPromptType = makeSelect(miniMaxProject ? [
      { value: "text_to_video", label: "MiniMax H3 — Text to Video" },
      { value: "image_to_video", label: "MiniMax H3 — Image to Video" },
      { value: "reference_to_video", label: "MiniMax H3 — Reference to Video" },
      { value: "video_to_video", label: "MiniMax H3 — Video to Video" },
    ] : [
      { value: "i2v", label: "Image to Video" },
      { value: "id_lora", label: "ID-LoRA I2V" },
      { value: "t2v", label: "Text to Video" },
      { value: "rtv", label: "Reference to Video" },
      { value: "ingredients", label: "Ingredients to Video" },
    ], miniMaxProject ? normalizeStoryboardMiniMaxH3Mode(scene.minimax_h3_mode) : (scene.video_prompt_type || "i2v"));
    const subjects = makeInput((scene.subjects || []).join(", "), "Subjects, comma separated");
    const subjectDetails = makeTextarea(
      (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
        .map((subject) => `${subject.name || "Subject"}: ${subject.description || ""}`.trim())
        .filter(Boolean)
        .join("\n\n"),
      "Character descriptions from Reference Builder...",
      4,
    );
    const setting = makeInput(scene.setting || scene.location_ref?.description || scene.location_ref?.name || "", "Location / setting");
    const locationDetails = makeTextarea(
      scene.location_ref
        ? `${scene.location_ref.name || "Location"}: ${scene.location_ref.description || ""}`.trim()
        : "",
      "Location description from Reference Builder...",
      4,
    );
    const shot = makeInput(scene.shot_type, "Shot type");
    const shotPreset = makeSelect([{ value: "", label: "Choose a preset..." }, { value: "__custom__", label: "Custom / keep typed value" }], "__custom__");
    const imagePrompt = makeTextarea(scene.image_prompt, "Full text-to-image prompt...", 7);
    const videoPrompt = makeTextarea(scene.video_prompt, "Full video prompt...", 7);
    let editorVideoPromptOrigin = normalizeVideoPromptOrigin(scene.video_prompt_origin);
    videoPrompt.addEventListener("input", () => {
      editorVideoPromptOrigin = "manual";
    });
    const imagePath = makeInput(scene.image_path, "Image path");
    imagePath.type = "hidden";
    let sceneImageData = String(scene.image_data || scene.image_reference_data || "").trim();
    let sceneImageName = String(scene.image_name || scene.image_reference_name || "").trim();
    const startingImageControl = document.createElement("div");
    startingImageControl.dataset.vrgdgFileDropZone = "true";
    startingImageControl.style.cssText = "display:grid;grid-template-columns:112px 1fr;gap:12px;align-items:center;border:1px dashed #155e75;border-radius:9px;background:#07111f;padding:10px;cursor:pointer;";
    const startingImagePreview = document.createElement("div");
    startingImagePreview.style.cssText = "width:112px;height:82px;border:1px solid #334155;border-radius:7px;background:#020617 center/contain no-repeat;display:grid;place-items:center;color:#64748b;font-size:11px;text-align:center;overflow:hidden;";
    const startingImageDetails = document.createElement("div");
    startingImageDetails.style.cssText = "display:flex;flex-direction:column;gap:7px;min-width:0;";
    const startingImageButton = makeButton("Upload Image", "primary");
    startingImageButton.type = "button";
    const startingImageStatus = document.createElement("div");
    startingImageStatus.style.cssText = "font-size:11px;color:#94a3b8;overflow-wrap:anywhere;";
    const startingImageNote = document.createElement("div");
    startingImageNote.style.cssText = "font-size:11px;line-height:1.4;color:#cbd5e1;";
    startingImageNote.textContent = "This is the finished starting frame for this I2V scene. Storyboard Builder analyzes it when writing the video prompt so the motion matches what is actually visible.";
    startingImageDetails.append(startingImageButton, startingImageStatus, startingImageNote);
    startingImageControl.append(startingImagePreview, startingImageDetails, imagePath);
    const refreshStartingImage = () => {
      const source = sceneImageData
        ? (sceneImageData.startsWith("data:") ? sceneImageData : `data:image/png;base64,${sceneImageData}`)
        : (imagePath.value.trim() ? makeStoryboardImageUrl(imagePath.value.trim()) : "");
      startingImagePreview.style.backgroundImage = source ? `url("${source.replace(/"/g, "%22")}")` : "none";
      startingImagePreview.textContent = source ? "" : "Drop image here";
      startingImageStatus.textContent = source
        ? `Selected: ${sceneImageName || imagePath.value.trim().split(/[\\/]/).pop() || "uploaded image"}`
        : "No starting image selected. Drop a PNG, JPG, or WEBP here, or click Upload Image.";
    };
    const useStartingImageFile = async (file) => {
      if (!file) return;
      sceneImageData = await readStoryboardImageFile(file);
      sceneImageName = file.name || "starting_frame.png";
      imagePath.value = "";
      refreshStartingImage();
    };
    startingImageButton.onclick = async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await useStartingImageFile(await chooseStoryboardImageFile());
    };
    startingImageControl.onclick = async (event) => {
      if (event.target === startingImageButton) return;
      await useStartingImageFile(await chooseStoryboardImageFile());
    };
    startingImageControl.addEventListener("dragover", (event) => {
      if (!event.dataTransfer?.files?.length && !Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      startingImageControl.style.borderColor = "#22d3ee";
    });
    startingImageControl.addEventListener("dragleave", () => { startingImageControl.style.borderColor = "#155e75"; });
    startingImageControl.addEventListener("drop", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      startingImageControl.style.borderColor = "#155e75";
      await useStartingImageFile(Array.from(event.dataTransfer?.files || []).find((file) => String(file.type || "").startsWith("image/")) || null);
    });
    refreshStartingImage();
    const triggerPhrase = makeInput(scene.trigger_phrase || "", "Optional scene trigger phrase");
    const triggerPosition = makeSelect([
      { value: "start", label: "Add trigger to start" },
      { value: "end", label: "Add trigger to end" },
    ], scene.trigger_position || "start");
    const notes = makeTextarea(scene.notes, "Extra planning notes...", 3);
    const audioDirection = makeTextarea(scene.audio_direction || "", "Exact ambience, sound effects, silence, breathing, or audio behavior for this scene...", 4);
    const continuityDirection = makeTextarea(scene.continuity || "", "Exact identity, wardrobe, prop, location, screen-direction, and spatial continuity requirements...", 4);
    const selectedSubjectIds = scene.no_character_present ? [] : (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
      .map((ref) => String(ref?.id || ""))
      .filter(Boolean);
    const subjectSelect = makeMultiSelect(
      state.referenceBuilder.subjects.map((subject) => ({ value: subject.id, label: subject.name })),
      selectedSubjectIds,
    );
    const savedLocationId = String(scene.location_ref?.id || "");
    const locationOptions = [
      { value: "", label: "Unassigned" },
      ...state.referenceBuilder.locations.map((location) => ({ value: location.id, label: location.name })),
    ];
    const locationSelect = makeSelect(locationOptions, savedLocationId);
    const field = (name, control) => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:800;color:#cbd5e1;";
      wrap.textContent = name;
      wrap.append(control);
      return wrap;
    };
    const section = (number, title, content, { collapsible = false, open = false } = {}) => {
      const wrap = collapsible ? document.createElement("details") : document.createElement("section");
      if (collapsible) wrap.open = open;
      wrap.style.cssText = "border:1px solid #1f3b46;border-radius:10px;background:linear-gradient(135deg,rgba(8,51,68,.34),rgba(15,23,42,.9));padding:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03);";
      const heading = collapsible ? document.createElement("summary") : document.createElement("div");
      heading.style.cssText = "display:flex;align-items:center;gap:12px;color:#e2e8f0;font-size:20px;font-weight:900;cursor:pointer;list-style:none;";
      const badge = document.createElement("span");
      badge.textContent = String(number);
      badge.style.cssText = "width:30px;height:30px;border-radius:999px;background:#155e75;color:#cffafe;display:grid;place-items:center;font-size:15px;flex:0 0 auto;";
      const text = document.createElement("span");
      text.textContent = title;
      heading.append(badge, text);
      if (collapsible) {
        const chevron = document.createElement("span");
        chevron.textContent = "⌄";
        chevron.style.cssText = "margin-left:auto;color:#cbd5e1;font-size:22px;";
        heading.append(chevron);
      }
      const body = document.createElement("div");
      body.style.cssText = "margin-top:12px;";
      body.append(content);
      wrap.append(heading, body);
      return wrap;
    };
    const twoCol = () => {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;";
      return grid;
    };
    const threeCol = () => {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px 28px;";
      return grid;
    };
    const iconField = (icon, control) => {
      const row = document.createElement("div");
      row.style.cssText = "display:grid;grid-template-columns:44px 1fr;gap:8px;align-items:center;";
      const ico = document.createElement("div");
      ico.textContent = icon;
      ico.style.cssText = "width:42px;height:42px;border:1px solid #155e75;border-radius:8px;background:#083344;color:#22d3ee;display:grid;place-items:center;font-size:20px;";
      row.append(ico, control);
      return row;
    };
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const videoTypeHint = document.createElement("div");
    videoTypeHint.style.cssText = "grid-column:1/-1;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;font-size:12px;line-height:1.45;padding:9px 10px;";
    const shotPresetField = field("Shot type preset", shotPreset);
    const shotCustomField = field("Custom shot type", shot);
    const cameraMotionField = field(isImagePrepMode ? "Still camera style preset" : "Camera motion preset", cameraMotionPreset);
    const characterMotionField = field("Character motion preset", characterMotionPreset);
    const customCharacterMotionField = field("Custom character motion", customCharacterMotion);
    const performanceStyleField = field("Performance / song style", performanceStyle);
    const videoStyleField = field(state.videoStyle ? "Video aesthetic — global and required" : "Video aesthetic", videoStyle);
    const videoStyleCustomField = field("Custom style wording — copied exactly", videoStyleCustom);
    const temporalEffectField = field("Temporal / world effect", temporalEffectOverride);
    const temporalEffectCustomField = field("Custom temporal wording", temporalEffectCustom);
    const facialPerformanceField = field("Facial performance", facialPerformance);
    const facialPerformanceCustomField = field("Custom facial performance", facialPerformanceCustom);
    const imagePathField = field("Starting image", startingImageControl);
    const motionField = field(isImagePrepMode ? "Still photography notes" : "Motion Notes / LLM Direction", motion);
    const t2iPromptField = field("T2I prompt", imagePrompt);
    if (isVideoPrepMode) {
      grid.append(field("Video prompt type", videoPromptType), videoStyleField, videoStyleCustomField, field("Setting", setting), videoTypeHint, field("Subjects", subjects), performanceStyleField, facialPerformanceField, facialPerformanceCustomField, includeMicLabel, noCharacterLabel, shotPresetField, shotCustomField, cameraMotionField, characterMotionField, customCharacterMotionField, imagePathField, field("Scene trigger phrase", triggerPhrase), field("Trigger placement", triggerPosition));
    } else {
      grid.append(field("Setting", setting), field("Subjects", subjects), performanceStyleField, facialPerformanceField, facialPerformanceCustomField, includeMicLabel, noCharacterLabel, shotPresetField, shotCustomField, cameraMotionField, field("Scene trigger phrase", triggerPhrase), field("Trigger placement", triggerPosition));
    }
    const referenceGrid = document.createElement("div");
    referenceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;";
    if (state.referenceBuilder.subjects.length || state.referenceBuilder.locations.length) {
      referenceGrid.append(
        field("Reference Builder characters", subjectSelect),
        field("Reference Builder location", locationSelect),
      );
    } else {
      referenceGrid.innerHTML = `<div style="grid-column:1/-1;color:#94a3b8;font-size:12px;">No Reference Builder subjects or locations are available yet. Add them in Reference Builder first, then reopen Storyboard Builder.</div>`;
    }
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;";
    const gemmaBeat = makeButton(`${promptRunnerGenericName()} Story Beat`, "primary");
    const gemma = makeButton("Generate Prompt", "purple");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Save Scene Card", "primary");
    actions.append(cancel, gemmaBeat, gemma, apply);
    if (isFullyCustomShortFilm()) {
      gemmaBeat.style.display = "none";
      gemma.title = "Formats the manually entered scene card into a MiniMax H3 prompt without inventing or rewriting scene content.";
    }
    const closeEditor = makeButton("×");
    closeEditor.style.cssText += "font-size:26px;line-height:1;width:44px;height:44px;padding:0;border-radius:8px;";
    const header = document.createElement("div");
    header.style.cssText = "display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;";
    const headerIcon = document.createElement("div");
    headerIcon.textContent = "▣";
    headerIcon.style.cssText = "width:54px;height:54px;border-radius:14px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;";
    const headerText = document.createElement("div");
    headerText.innerHTML = `<div style="font-size:28px;font-weight:900;color:#f8fafc;">Edit Scene Card</div><div style="color:#cbd5e1;margin-top:3px;">${isVideoPrepMode ? "Define the details for this scene to generate a rich video prompt." : "Define the details for this scene to generate a rich text-to-image prompt."}</div>`;
    header.append(headerIcon, headerText, closeEditor);

    const basicsGrid = twoCol();
    basicsGrid.append(field("Scene label", label), field("Lyric section", lyricSection), field("Scene / lyrics", lyrics), field("Scene story beat", storyBeat));
    if (isVideoPrepMode) {
      basicsGrid.append(field("Prompt mode", iconField("▣", videoPromptType)), videoStyleField, videoStyleCustomField, temporalEffectField, temporalEffectCustomField, field("Performance / song style", performanceStyle), field("Facial performance", facialPerformance), field("Custom facial performance", facialPerformanceCustom), includeMicLabel, noCharacterLabel, videoTypeHint);
    } else {
      const imagePromptType = makeInput("Text to Image", "Text to Image");
      imagePromptType.readOnly = true;
      basicsGrid.append(field("Image prompt type", iconField("▣", imagePromptType)), field("Performance / song style", performanceStyle), field("Facial performance", facialPerformance), field("Custom facial performance", facialPerformanceCustom), includeMicLabel, noCharacterLabel);
    }

    const addSubject = makeButton("+ Add subject");
    addSubject.style.background = "#0f172a";
    addSubject.style.borderStyle = "dashed";
    const addLocation = makeButton("+ Add location");
    addLocation.style.background = "#0f172a";
    addLocation.style.borderStyle = "dashed";
    const subjectChip = document.createElement("div");
    const locationChip = document.createElement("div");
    const refreshReferenceChips = () => {
      const selectedSubjects = Array.from(subjectSelect.selectedOptions).map((option) => state.referenceBuilder.subjects.find((subject) => subject.id === option.value)).filter(Boolean);
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
      subjectChip.innerHTML = noCharacterInput.checked
        ? `<span style="color:#fca5a5;">No character present</span>`
        : selectedSubjects.length
        ? selectedSubjects.map((ref) => referenceChipHtml(ref, "Subject")).join("")
        : `<span style="color:#94a3b8;">No subject selected</span>`;
      locationChip.innerHTML = selectedLocation
        ? referenceChipHtml(selectedLocation, "Location")
        : `<span style="color:#94a3b8;">No location selected</span>`;
    };
    const refreshNoCharacterState = () => {
      subjectSelect.disabled = Boolean(noCharacterInput.checked);
      subjects.disabled = Boolean(noCharacterInput.checked);
      subjectDetails.disabled = Boolean(noCharacterInput.checked);
      if (noCharacterInput.checked) {
        for (const option of subjectSelect.options) option.selected = false;
        subjects.value = "";
        subjectDetails.value = "";
      }
      refreshReferenceChips();
    };
    const referencesGrid = twoCol();
    const subjectPick = document.createElement("div");
    subjectPick.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;";
    subjectPick.append(field("Subject(s)", subjectChip), addSubject);
    const locationPick = document.createElement("div");
    locationPick.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;";
    locationPick.append(field("Setting / Location", locationChip), addLocation);
    referencesGrid.append(subjectPick, locationPick, ...Array.from(referenceGrid.children));
    refreshReferenceChips();

    const speakerAssignmentEnabled = isVideoPrepMode
      && miniMaxProject
      && normalizeStoryboardPerformanceMode(scene.performance_mode || state.performanceMode) === "speaking"
      && normalizeStoryboardMiniMaxH3AudioMode(scene.minimax_h3_audio_mode || state.miniMaxH3AudioMode) === "built_in_audio";
    let speakerAssignments = normalizeStoryboardSpeakerAssignments(scene.speaker_assignments);
    const speakerAssignmentWrap = document.createElement("div");
    speakerAssignmentWrap.style.cssText = "display:flex;flex-direction:column;gap:9px;";
    const speakerAssignmentNote = document.createElement("div");
    speakerAssignmentNote.style.cssText = "border:1px solid #155e75;border-radius:8px;background:#07111f;color:#cbd5e1;padding:9px 10px;font-size:12px;line-height:1.45;";
    speakerAssignmentNote.textContent = "Drag cues into the exact speaking order. The same character can have multiple turns. Speaker choices come only from this scene’s mapped Reference Builder characters.";
    const speakerAssignmentRows = document.createElement("div");
    speakerAssignmentRows.style.cssText = "display:flex;flex-direction:column;gap:8px;";
    const addSpeakerAssignment = makeButton("Add Dialogue Cue", "primary");
    const mappedSpeakerOptions = () => {
      const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => String(option.value || "")).filter(Boolean);
      const selected = selectedIds
        .map((id) => state.referenceBuilder.subjects.find((subject) => String(subject.id || "") === id))
        .filter(Boolean);
      const fallback = Array.isArray(scene.subject_refs) ? scene.subject_refs : [];
      return (selected.length ? selected : fallback)
        .filter((subject) => subject && typeof subject === "object")
        .map((subject) => ({ id: String(subject.id || ""), name: String(subject.name || "Character").trim() || "Character" }));
    };
    const syncSpeakerAssignmentLegacy = () => {
      speakerAssignments = normalizeStoryboardSpeakerAssignments(speakerAssignments);
      scene.speaker_assignments = speakerAssignments;
      const filled = speakerAssignments.filter((cue) => cue.text);
      const combined = filled.map((cue) => cue.text).join("\n");
      lyrics.value = combined;
      scene.lyrics = combined;
      scene.lyric_singers = Array.from(new Set(filled.map((cue) => cue.speaker_name).filter(Boolean)));
    };
    const ensureSpeakerAssignments = () => {
      if (speakerAssignments.length) return;
      const speakers = mappedSpeakerOptions();
      const existingLine = String(scene.lyrics || lyrics.value || "").trim();
      if (existingLine) {
        const preferred = String((Array.isArray(scene.lyric_singers) ? scene.lyric_singers[0] : "") || "").trim();
        const speaker = speakers.find((item) => item.name.toLowerCase() === preferred.toLowerCase()) || speakers[0] || { id: "", name: preferred };
        speakerAssignments = normalizeStoryboardSpeakerAssignments([{ speaker_id: speaker.id, speaker_name: speaker.name, text: existingLine }]);
      } else if (speakers.length) {
        speakerAssignments = normalizeStoryboardSpeakerAssignments(speakers.map((speaker) => ({ speaker_id: speaker.id, speaker_name: speaker.name, text: "" })));
      }
      syncSpeakerAssignmentLegacy();
    };
    const renderSpeakerAssignments = () => {
      speakerAssignmentRows.replaceChildren();
      const speakers = mappedSpeakerOptions();
      addSpeakerAssignment.disabled = !speakers.length || Boolean(noCharacterInput.checked);
      ensureSpeakerAssignments();
      if (!speakers.length || noCharacterInput.checked) {
        const empty = document.createElement("div");
        empty.textContent = noCharacterInput.checked
          ? "This scene is marked No character present."
          : "Map one or more Reference Builder characters to this scene first.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:8px;padding:12px;color:#94a3b8;text-align:center;font-size:12px;";
        speakerAssignmentRows.append(empty);
        return;
      }
      let draggedIndex = -1;
      speakerAssignments.forEach((cue, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:34px 34px minmax(160px,.65fr) minmax(300px,1.5fr) 76px;gap:8px;align-items:center;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:8px;";
        row.addEventListener("dragover", (event) => {
          if (draggedIndex < 0 || draggedIndex === index) return;
          event.preventDefault();
          row.style.borderColor = "#22d3ee";
        });
        row.addEventListener("dragleave", () => { row.style.borderColor = "#334155"; });
        row.addEventListener("drop", (event) => {
          event.preventDefault();
          row.style.borderColor = "#334155";
          if (draggedIndex < 0 || draggedIndex === index) return;
          const [moved] = speakerAssignments.splice(draggedIndex, 1);
          speakerAssignments.splice(index, 0, moved);
          syncSpeakerAssignmentLegacy();
          renderSpeakerAssignments();
        });
        const handle = document.createElement("button");
        handle.type = "button";
        handle.textContent = "::";
        handle.title = "Drag to change speaking order";
        handle.draggable = true;
        handle.style.cssText = "height:38px;border:1px solid #334155;border-radius:6px;background:#07111f;color:#67e8f9;font-weight:900;cursor:grab;";
        handle.addEventListener("dragstart", () => { draggedIndex = index; row.style.opacity = ".55"; });
        handle.addEventListener("dragend", () => { draggedIndex = -1; row.style.opacity = ""; });
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-weight:900;color:#cffafe;text-align:center;";
        const speakerSelect = makeSelect(speakers.map((speaker) => ({ value: speaker.id, label: speaker.name })), cue.speaker_id);
        if (!speakers.some((speaker) => speaker.id === cue.speaker_id) && cue.speaker_name) {
          speakerSelect.prepend(new Option(`${cue.speaker_name} (not currently mapped)`, cue.speaker_id));
          speakerSelect.value = cue.speaker_id;
        }
        const line = makeInput(cue.text || "", "Exact words this character says...");
        const remove = makeButton("Remove");
        speakerSelect.addEventListener("change", () => {
          const speaker = speakers.find((item) => item.id === speakerSelect.value) || { id: speakerSelect.value, name: speakerSelect.selectedOptions[0]?.textContent || "" };
          cue.speaker_id = speaker.id;
          cue.speaker_name = speaker.name;
          syncSpeakerAssignmentLegacy();
        });
        line.addEventListener("input", () => {
          cue.text = line.value;
          syncSpeakerAssignmentLegacy();
        });
        remove.onclick = () => {
          speakerAssignments.splice(index, 1);
          syncSpeakerAssignmentLegacy();
          renderSpeakerAssignments();
        };
        row.append(handle, number, speakerSelect, line, remove);
        speakerAssignmentRows.append(row);
      });
    };
    addSpeakerAssignment.onclick = () => {
      const speaker = mappedSpeakerOptions()[0];
      if (!speaker) return;
      speakerAssignments.push(...normalizeStoryboardSpeakerAssignments([{ speaker_id: speaker.id, speaker_name: speaker.name, text: "" }]));
      syncSpeakerAssignmentLegacy();
      renderSpeakerAssignments();
    };
    speakerAssignmentWrap.append(speakerAssignmentNote, speakerAssignmentRows, addSpeakerAssignment);
    if (speakerAssignmentEnabled) {
      lyrics.readOnly = true;
      lyrics.title = "This value is built automatically from the ordered Speaker Assignment cues below.";
      lyrics.style.opacity = "0.78";
      renderSpeakerAssignments();
    }

    const motionGrid = isVideoPrepMode ? threeCol() : twoCol();
    if (isVideoPrepMode) {
      motionGrid.append(
        field("Starting shot preset", iconField("▣", shotPreset)),
        field("Camera motion preset", iconField("▣", cameraMotionPreset)),
        field("Character motion preset", iconField("♟", characterMotionPreset)),
        field("Custom starting shot (optional)", shot),
        field("Custom camera motion (optional)", customCameraMotion),
        field("Custom character motion (optional)", customCharacterMotion),
      );
    } else {
      motionGrid.append(
        field("Shot / composition preset", iconField("▣", shotPreset)),
        field("Still camera / photography preset", iconField("▣", cameraMotionPreset)),
        field("Custom shot / composition (optional)", shot),
        field("Custom still camera style (optional)", customCameraMotion),
      );
    }

    const advancedGrid = twoCol();
    if (isVideoPrepMode) {
      advancedGrid.append(field("Prompt summary", summary), motionField, field("Character details", subjectDetails), field("Location details", locationDetails), imagePathField, t2iPromptField, field("Video prompt", videoPrompt));
      if (isMiniMaxShortFilmMode) {
        advancedGrid.append(field("Manual audio / sound direction", audioDirection), field("Manual continuity requirements", continuityDirection));
      }
    } else {
      advancedGrid.append(t2iPromptField, field("Character details", subjectDetails), field("Location details", locationDetails), field("Still photography notes", motion));
    }
    const notesWrap = document.createElement("div");
    notesWrap.append(notes);
    const flfBeatGrid = twoCol();
    flfBeatGrid.append(
      field(editorSceneIndex > 0 ? "Start-frame state (inherited from previous end)" : "Start-frame state", flfStartState),
      field("Transformation during scene", flfTransformation),
      field("End-frame state", flfEndState),
      field("Carry-forward state", flfCarryForward),
    );
    const flfBeatSection = section(2, "First / Last Frame Endpoint Beat", flfBeatGrid);
    const editorSections = [
      header,
      section(1, "Scene Basics", basicsGrid),
    ];
    if (speakerAssignmentEnabled) editorSections.push(section("2", "Speaker Assignment", speakerAssignmentWrap));
    if (state.videoPromptType === "flf" || scene.video_prompt_type === "flf") editorSections.push(flfBeatSection);
    editorSections.push(
      section(3, "References", referencesGrid),
      section(4, isVideoPrepMode ? "Camera & Motion" : "Shot & Still Camera", motionGrid),
      section(5, "Advanced Options", advancedGrid, { collapsible: true, open: false }),
      section(6, "Notes", notesWrap),
      actions,
    );
    editor.replaceChildren(
      ...editorSections,
    );
    editorBackdrop.append(editor);
    document.body.append(editorBackdrop);
    closeEditor.onclick = () => editorBackdrop.remove();
    const refreshShotPresetForVideoType = () => {
      const type = videoPromptType.value || "i2v";
      const imageToVideoType = type === "i2v" || type === "image_to_video";
      const textToVideoType = type === "t2v" || type === "text_to_video";
      const referenceToVideoType = type === "rtv" || type === "reference_to_video";
      const videoStyleType = !miniMaxProject || textToVideoType || referenceToVideoType;
      const options = isImagePrepMode ? IMAGE_SHOT_TYPES : (imageToVideoType ? VIDEO_SHOT_TYPES : Array.from(new Set([...IMAGE_SHOT_TYPES, ...VIDEO_SHOT_TYPES])));
      const current = shot.value || scene.shot_type || "";
      shotPreset.replaceChildren();
      for (const option of [
        { value: "", label: isImagePrepMode ? "Choose shot / composition preset..." : (imageToVideoType ? "Choose camera/motion preset..." : "Choose starting shot preset...") },
        ...options.map((item) => ({ value: item, label: item })),
        { value: "__custom__", label: "Custom / keep typed value" },
      ]) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        shotPreset.append(item);
      }
      shotPreset.value = options.includes(current) ? current : "__custom__";
      shotPresetField.firstChild.textContent = isImagePrepMode ? "Shot / composition preset" : (imageToVideoType ? "Camera / motion preset" : "Starting shot preset");
      shotCustomField.firstChild.textContent = isImagePrepMode ? "Custom shot / composition" : (imageToVideoType ? "Custom camera / motion" : "Custom starting shot");
      videoTypeHint.textContent = videoPromptTypeHint(type);
      motionField.firstChild.textContent = isImagePrepMode
        ? "Still photography notes"
        : imageToVideoType
          ? "Motion Notes / LLM Direction"
          : referenceToVideoType
            ? "Motion Notes / LLM Direction (with references)"
            : "Motion Notes / LLM Direction";
      t2iPromptField.style.display = isImagePrepMode || (!textToVideoType && !referenceToVideoType) ? "flex" : "none";
      imagePathField.style.display = isVideoPrepMode && !textToVideoType && !referenceToVideoType ? "flex" : "none";
      videoStyleField.style.display = isVideoPrepMode && videoStyleType ? "flex" : "none";
      videoStyleCustomField.style.display = isVideoPrepMode && videoStyleType && videoStyle.value === "custom" ? "flex" : "none";
      temporalEffectField.style.display = isVideoPrepMode ? "flex" : "none";
      temporalEffectCustomField.style.display = isVideoPrepMode && temporalEffectOverride.value === "custom" ? "flex" : "none";
      videoPrompt.style.display = isVideoPrepMode ? "" : "none";
      videoPrompt.placeholder = textToVideoType
        ? "Full text-to-video prompt..."
        : referenceToVideoType
          ? "Full reference-to-video prompt..."
          : type === "video_to_video"
            ? "Full video-to-video prompt..."
          : "Full image-to-video prompt...";
    };
    refreshShotPresetForVideoType();
    videoPromptType.addEventListener("change", refreshShotPresetForVideoType);
    videoStyle.addEventListener("change", refreshShotPresetForVideoType);
    temporalEffectOverride.addEventListener("change", refreshShotPresetForVideoType);
    const refreshSubjectDetailsFromSelection = () => {
      const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => option.value).filter(Boolean);
      const selectedSubjects = selectedIds
        .map((id) => state.referenceBuilder.subjects.find((subject) => subject.id === id))
        .filter(Boolean);
      subjectDetails.value = selectedSubjects
        .map((subject) => `${subject.name || "Subject"}: ${subject.description || ""}`.trim())
        .filter(Boolean)
        .join("\n\n");
    };
    subjectSelect.addEventListener("change", refreshSubjectDetailsFromSelection);
    subjectSelect.addEventListener("change", refreshReferenceChips);
    noCharacterInput.addEventListener("change", refreshNoCharacterState);
    if (speakerAssignmentEnabled) {
      subjectSelect.addEventListener("change", renderSpeakerAssignments);
      noCharacterInput.addEventListener("change", renderSpeakerAssignments);
    }
    refreshNoCharacterState();
    shotPreset.addEventListener("change", () => {
      if (shotPreset.value && shotPreset.value !== "__custom__") shot.value = shotPreset.value;
    });
    cameraMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(cameraMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCameraMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      motion.value = replaceLabeledPlanningLine(currentMotion, isImagePrepMode ? "Still camera style" : "Camera motion", selectedMotion);
    });
    characterMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(characterMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCharacterMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      motion.value = replaceLabeledPlanningLine(currentMotion, "Character motion", selectedMotion);
    });
    locationSelect.addEventListener("change", () => {
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
      if (selectedLocation) {
        setting.value = selectedLocation.description || selectedLocation.name || "";
        locationDetails.value = `${selectedLocation.name || "Location"}: ${selectedLocation.description || ""}`.trim();
      } else {
        locationDetails.value = "";
      }
      refreshReferenceChips();
    });
    addSubject.onclick = async () => {
      saveEditorFieldsToScene();
      const ref = await addStoryboardReferenceFromFile("subject", scene);
      if (!ref) return;
      let option = Array.from(subjectSelect.options).find((item) => item.value === ref.id);
      if (!option) {
        option = document.createElement("option");
        option.value = ref.id;
        option.textContent = ref.name;
        subjectSelect.append(option);
      }
      option.selected = true;
      refreshSubjectDetailsFromSelection();
      refreshReferenceChips();
    };
    addLocation.onclick = async () => {
      saveEditorFieldsToScene();
      const ref = await addStoryboardReferenceFromFile("location", scene);
      if (!ref) return;
      let option = Array.from(locationSelect.options).find((item) => item.value === ref.id);
      if (!option) {
        option = document.createElement("option");
        option.value = ref.id;
        option.textContent = ref.name;
        locationSelect.append(option);
      }
      locationSelect.value = ref.id;
      setting.value = ref.description || ref.name || "";
      refreshReferenceChips();
    };
    const saveEditorFieldsToScene = () => {
      scene.label = label.value.trim() || scene.label;
      scene.lyric_section = lyricSection.value.trim();
      scene.lyrics = lyrics.value.trim();
      scene.story_beat = storyBeat.value.trim();
      scene.flf_start_state = flfStartState.value.trim();
      scene.flf_transformation = flfTransformation.value.trim();
      scene.flf_end_state = flfEndState.value.trim();
      scene.flf_carry_forward = flfCarryForward.value.trim();
      propagateFlfEndStateToNextScene(scene);
      scene.prompt_summary = isVideoPrepMode ? summary.value.trim() : "";
      scene.motion_summary = motion.value.trim();
      if (isVideoPrepMode && miniMaxProject) {
        scene.minimax_h3_mode = normalizeStoryboardMiniMaxH3Mode(videoPromptType.value);
        scene.project_video_engine = "minimax_h3";
      } else {
        scene.video_prompt_type = isVideoPrepMode ? (videoPromptType.value || "i2v") : "i2v";
      }
      scene.no_character_present = Boolean(noCharacterInput.checked);
      scene.subjects = scene.no_character_present ? [] : subjects.value.split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
      scene.setting = setting.value.trim();
      if (state.referenceBuilder.subjects.length && !scene.no_character_present) {
        const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => option.value).filter(Boolean);
        scene.subject_refs = selectedIds
          .map((id) => state.referenceBuilder.subjects.find((subject) => subject.id === id))
          .filter(Boolean);
        const detailsByName = new Map(
          subjectDetails.value
            .split(/\n{2,}/)
            .map((block) => {
              const parts = block.split(":");
              const name = String(parts.shift() || "").trim();
              const description = parts.join(":").trim();
              return name ? [name.toLowerCase(), description] : null;
            })
            .filter(Boolean)
        );
        scene.subject_refs = scene.subject_refs.map((subject) => ({
          ...subject,
          description: detailsByName.get(String(subject.name || "").toLowerCase()) ?? subject.description,
        }));
        if (scene.subject_refs.length) {
          scene.subjects = storyboardSubjectNamesFromRefs(scene.subject_refs);
        }
      } else if (scene.no_character_present) {
        scene.subject_refs = [];
      }
      if (state.referenceBuilder.locations.length) {
        const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
        const locationParts = String(locationDetails.value || "").split(":");
        const locationName = String(locationParts.shift() || "").trim();
        const locationDescription = locationParts.join(":").trim();
        scene.location_ref = selectedLocation
          ? {
              ...selectedLocation,
              name: locationName || selectedLocation.name,
              description: locationDescription || selectedLocation.description || "",
            }
          : null;
        if (selectedLocation) scene.setting = selectedLocation.description || selectedLocation.name || scene.setting;
        if (scene.location_ref) scene.setting = scene.location_ref.description || scene.location_ref.name || scene.setting;
      }
      scene.shot_type = shot.value.trim();
      scene.camera_motion = customCameraMotion.value.trim() || cameraMotionPreset.value.trim();
      scene.character_motion = isVideoPrepMode ? (customCharacterMotion.value.trim() || characterMotionPreset.value.trim()) : "";
      scene.performance_style = performanceStyle.value || "";
      scene.video_style = videoStyle.value || "";
      scene.video_style_custom = videoStyle.value === "custom" ? videoStyleCustom.value.trim() : "";
      scene.temporal_world_effect_override = temporalEffectOverride.value || "global";
      scene.temporal_world_effect_custom = temporalEffectOverride.value === "custom" ? temporalEffectCustom.value.trim() : "";
      scene.facial_performance = facialPerformance.value || "";
      scene.facial_performance_custom = facialPerformanceCustom.value.trim();
      scene.include_microphone = Boolean(includeMic.checked);
      scene.trigger_phrase = triggerPhrase.value.trim();
      scene.trigger_position = triggerPosition.value === "end" ? "end" : "start";
      scene.image_prompt = imagePrompt.value.trim();
      if (isVideoPrepMode) {
        scene.video_prompt = videoPrompt.value.trim();
        scene.video_prompt_origin = editorVideoPromptOrigin;
      }
      if (isVideoPrepMode) {
        scene.image_path = imagePath.value.trim();
        scene.image_data = sceneImageData;
        scene.image_name = sceneImageName;
      }
      if (speakerAssignmentEnabled) {
        scene.minimax_h3_audio_mode = "built_in_audio";
        syncSpeakerAssignmentLegacy();
      }
      scene.notes = notes.value.trim();
      scene.audio_direction = audioDirection.value.trim();
      scene.continuity = continuityDirection.value.trim();
    };
    cancel.onclick = () => editorBackdrop.remove();
    gemma.onclick = async () => {
      const previous = gemma.textContent;
      gemma.disabled = true;
      const runnerName = promptRunnerName();
      gemma.textContent = `Running ${runnerName}...`;
      const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
      try {
        saveEditorFieldsToScene();
        progress.set(`Preparing ${scene.label || "scene"} for ${runnerName}...`, 12);
        await createScenePromptForActiveMode(scene, { progress, progressPercent: 32 });
        progress.set(state.mode === "image_to_video_prep" ? "Storyboard video prompt ready." : "Storyboard image prompt ready.", 100);
        progress.close(1200);
        imagePrompt.value = scene.image_prompt || "";
        videoPrompt.value = scene.video_prompt || "";
        editorVideoPromptOrigin = normalizeVideoPromptOrigin(scene.video_prompt_origin);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemma.disabled = false;
        gemma.textContent = previous;
      }
    };
    gemmaBeat.onclick = async () => {
      const previous = gemmaBeat.textContent;
      gemmaBeat.disabled = true;
      gemmaBeat.textContent = "Creating...";
      const progress = createStoryboardProgressWindow("Scene Story Beat");
      try {
        saveEditorFieldsToScene();
        await createSceneBeatWithGemma(scene, { progress, progressPercent: 35 });
        storyBeat.value = scene.story_beat || "";
        flfStartState.value = scene.flf_start_state || "";
        flfTransformation.value = scene.flf_transformation || "";
        flfEndState.value = scene.flf_end_state || "";
        flfCarryForward.value = scene.flf_carry_forward || "";
        progress.set("Scene story beat ready.", 100);
        progress.close(1200);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemmaBeat.disabled = false;
        gemmaBeat.textContent = previous;
      }
    };
    apply.onclick = () => {
      saveEditorFieldsToScene();
      syncReferenceMappingsToVideoCreator();
      syncStoryLayerFromInputs({ notify: true });
      editorBackdrop.remove();
      renderTable();
    };
  };

  function renderTable() {
    const rows = currentRows();
    const mode = state.mode;
    const head = mode === "image_to_video_prep"
      ? ["", "#", "Image", "Scene / Lyrics", "Motion Notes", "Video Prompt", "Subjects", "Setting", "Shot Type", "Status", "Actions"]
      : ["#", "Reference", "Scene / Lyrics", "Prompt Summary", "Subjects", "Setting", "Shot Type", "Prompt Status", "Actions"];
    const table = document.createElement("table");
    table.style.cssText = mode === "image_to_video_prep"
      ? "width:100%;border-collapse:collapse;table-layout:fixed;min-width:1567px;font-size:13px;"
      : "width:100%;border-collapse:collapse;min-width:1250px;font-size:13px;";
    if (mode === "image_to_video_prep") {
      const colgroup = document.createElement("colgroup");
      [36, 50, 166, 190, 205, 185, 205, 150, 120, 54, 206].forEach((width) => {
        const col = document.createElement("col");
        col.style.width = `${width}px`;
        colgroup.appendChild(col);
      });
      table.appendChild(colgroup);
    }
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>${head.map((item) => `<th style="position:sticky;top:0;background:#111827;border-bottom:1px solid #334155;color:#cffafe;text-align:${item === "Status" ? "center" : "left"};padding:${mode === "image_to_video_prep" ? "11px 9px" : "13px"};font-weight:900;">${escapeHtml(item)}</th>`).join("")}</tr>`;
    const tbody = document.createElement("tbody");
    for (const scene of rows) {
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid #1e293b";
      tr.style.background = "#0b1220";
      const sceneImageSource = storyboardReferenceImageSrc({ path: scene.image_path, data: scene.image_data || scene.image_reference_data });
      const imageWidth = mode === "image_to_video_prep" ? 148 : 170;
      const imageCell = sceneImageSource
        ? `<div style="width:${imageWidth}px;height:78px;border-radius:6px;background:#0f172a url('${escapeHtml(sceneImageSource)}') center/cover no-repeat;"></div>`
        : `<div style="width:${imageWidth}px;height:78px;border:1px dashed #334155;border-radius:6px;display:grid;place-items:center;color:#94a3b8;font-size:12px;text-align:center;background:#07111f;">No image in storyboard<br>Optional reference</div>`;
      const sceneActionStyle = mode === "image_to_video_prep"
        ? "border:1px solid #155e75;border-radius:6px;background:#0f172a;color:#a5f3fc;width:76px;min-height:54px;padding:6px 7px;line-height:1.15;white-space:normal;font-weight:800;cursor:pointer;"
        : "border:1px solid #155e75;border-radius:6px;background:#0f172a;color:#a5f3fc;padding:8px 10px;font-weight:800;cursor:pointer;";
      const sceneGptStyle = "border:1px solid #06b6d4;border-radius:6px;background:#0e7490;color:#f8fafc;padding:7px 8px;font-weight:900;cursor:pointer;";
      const sceneGemmaStyle = "border:1px solid #22c55e;border-radius:6px;background:#166534;color:#f0fdf4;padding:7px 8px;font-weight:900;cursor:pointer;";
      const runnerName = promptRunnerName();
      const gemmaTitle = mode === "image_to_video_prep"
        ? `Create this scene's video prompt with ${runnerName}. If the scene has an image, local vision uses it as guidance.`
        : `Create this scene's text-to-image prompt with ${runnerName}.`;
      const actionHtml = `
        <div style="display:${mode === "image_to_video_prep" ? "grid" : "flex"};grid-template-columns:${mode === "image_to_video_prep" ? "76px minmax(62px, 1fr) 44px" : "none"};align-items:stretch;gap:6px;white-space:nowrap;">
          <button data-action="edit" style="${sceneActionStyle}">${mode === "image_to_video_prep" ? "Open Scene<br>Card" : "Open Scene Card"}</button>
          <button data-action="gemma" style="${sceneGemmaStyle}" title="${escapeHtml(gemmaTitle)}">${escapeHtml(runnerName)}</button>
          <button data-action="gpt" style="${sceneGptStyle}" title="Copy only this scene card as GPT JSON.">GPT</button>
        </div>`;
      const promptReady = Boolean(String(mode === "image_to_video_prep" ? scene.video_prompt : scene.image_prompt).trim());
      const promptStatusLabel = promptReady ? "Prompt ready" : "Prompt missing";
      const promptStatusColor = promptReady ? "#22c55e" : "#ef4444";
      const status = `<span role="img" aria-label="${promptStatusLabel}" title="${promptStatusLabel}" style="display:flex;align-items:center;justify-content:center;width:100%;"><span style="width:12px;height:12px;border-radius:999px;background:${promptStatusColor};box-shadow:0 0 0 2px ${promptReady ? "rgba(34,197,94,.16)" : "rgba(239,68,68,.16)"};display:inline-block;"></span></span>`;
      const miniRefButtonStyle = "margin-top:7px;border:1px dashed #155e75;border-radius:6px;background:#07111f;color:#a5f3fc;padding:5px 7px;font-size:11px;font-weight:900;cursor:pointer;";
      const subjectCell = `<div>${subjectRefsHtml(scene)}</div><button data-action="load-subject-ref" title="Choose subjects from Reference Builder or upload a new subject image" style="${miniRefButtonStyle}">+ Subject</button>`;
      const settingCell = `<div>${settingRefHtml(scene)}</div><button data-action="load-location-ref" title="Load a location image for this scene" style="${miniRefButtonStyle}">+ Location</button>`;
      const videoType = videoPromptTypeLabel(state.projectVideoEngine === "minimax_h3" ? scene.minimax_h3_mode : (scene.video_prompt_type || "i2v"));
      const shotCell = `<div style="display:flex;flex-direction:column;gap:4px;"><span style="align-self:flex-start;border:1px solid #155e75;border-radius:999px;background:#0f172a;color:#a5f3fc;font-size:11px;font-weight:900;padding:2px 7px;">${escapeHtml(videoType)}</span><strong style="color:#f8fafc;">${escapeHtml(scene.shot_type || "-")}</strong></div>`;
      const storyPreview = `${scene.lyric_section ? `<div style="margin-top:5px;color:#67e8f9;font-size:11px;font-weight:900;">${escapeHtml(scene.lyric_section)}</div>` : ""}${scene.story_beat ? `<div style="margin-top:5px;color:#94a3b8;font-size:11px;">Beat: ${escapeHtml(truncate(scene.story_beat, 90))}</div>` : ""}`;
      if (mode === "image_to_video_prep") {
        const motionNotes = `<textarea data-action="motion-notes" aria-label="Motion notes for ${escapeHtml(scene.label || `Scene ${scene.scene_number}`)}" placeholder="Custom motion or LLM direction..." style="display:block;width:100%;height:80px;box-sizing:border-box;resize:vertical;border:1px solid #334155;border-radius:6px;background:#07111f;color:#e2e8f0;padding:8px;font:inherit;line-height:1.35;outline:none;">${escapeHtml(scene.motion_summary || "")}</textarea>`;
        const videoPrompt = scene.video_prompt
          ? `<div title="${escapeHtml(scene.video_prompt)}" style="color:#d4d4d8;line-height:1.38;overflow-wrap:anywhere;">${escapeHtml(truncate(scene.video_prompt, 115))}</div>`
          : `<span style="color:#64748b;font-style:italic;">No video prompt yet.</span>`;
        tr.innerHTML = `
          <td style="padding:9px;text-align:center;"><input type="checkbox" data-action="select" ${state.selected.has(scene.id) ? "checked" : ""}></td>
          <td style="padding:9px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:9px;">${imageCell}</td>
          <td style="padding:9px;overflow:hidden;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 70))}</span>${storyPreview}</td>
          <td style="padding:9px;vertical-align:middle;">${motionNotes}</td>
          <td style="padding:9px;vertical-align:middle;">${videoPrompt}</td>
          <td style="padding:9px;overflow:hidden;">${subjectCell}</td>
          <td style="padding:9px;color:#d4d4d8;overflow:hidden;">${settingCell}</td>
          <td style="padding:9px;overflow:hidden;">${shotCell}</td>
          <td style="padding:9px;text-align:center;">${status}</td>
          <td style="padding:9px;white-space:nowrap;">${actionHtml}</td>
        `;
      } else {
        tr.innerHTML = `
          <td style="padding:13px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:13px;">${imageCell}</td>
          <td style="padding:13px;max-width:220px;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 95))}</span>${storyPreview}</td>
          <td style="padding:13px;max-width:280px;color:#d4d4d8;">${escapeHtml(truncate(scene.prompt_summary || scene.image_prompt, 150))}</td>
          <td style="padding:13px;max-width:230px;">${subjectCell}</td>
          <td style="padding:13px;color:#d4d4d8;max-width:210px;">${settingCell}</td>
          <td style="padding:13px;">${shotCell}</td>
          <td style="padding:13px;">${status}</td>
          <td style="padding:13px;white-space:nowrap;">${actionHtml}</td>
        `;
      }
      tr.querySelector('[data-action="edit"]')?.addEventListener("click", () => openSceneEditor(scene));
      const motionNotesInput = tr.querySelector('[data-action="motion-notes"]');
      if (motionNotesInput) {
        motionNotesInput.addEventListener("input", () => {
          scene.motion_summary = motionNotesInput.value;
        });
        motionNotesInput.addEventListener("change", () => {
          scene.motion_summary = motionNotesInput.value.trim();
        });
        motionNotesInput.addEventListener("keydown", (event) => event.stopPropagation());
      }
      tr.querySelector('[data-action="load-subject-ref"]')?.addEventListener("click", () => openStoryboardSubjectPicker(scene));
      tr.querySelector('[data-action="load-location-ref"]')?.addEventListener("click", () => addStoryboardReferenceFromFile("location", scene));
      tr.querySelector('[data-action="gemma"]')?.addEventListener("click", async () => {
        const runnerName = promptRunnerName();
        const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
        try {
          progress.set(`Preparing ${scene.label || "scene"} for ${runnerName}...`, 12);
          await createScenePromptForActiveMode(scene, { progress, progressPercent: 32 });
          progress.set(state.mode === "image_to_video_prep" ? "Storyboard video prompt ready." : "Storyboard image prompt ready.", 100);
          progress.close(1200);
        } catch (error) {
          progress.set(`Error:\n${String(error?.message || error)}`, 100);
        }
      });
      tr.querySelector('[data-action="gpt"]')?.addEventListener("click", () => copySceneForGpt(scene));
      tr.querySelector('[data-action="select"]')?.addEventListener("change", (event) => {
        if (event.target.checked) state.selected.add(scene.id);
        else state.selected.delete(scene.id);
        renderTable();
      });
      tbody.append(tr);
    }
    table.append(thead, tbody);
    tableWrap.replaceChildren(table);
    const readyCount = state.scenes.filter((scene) => String(scene.image_prompt || scene.video_prompt || "").trim()).length;
    const imageCount = state.scenes.filter((scene) => String(scene.image_path || "").trim()).length;
    stats.textContent = `${state.scenes.length} scenes  |  ${imageCount} images linked  |  ${readyCount} scenes with prompts  |  ${state.selected.size} selected`;
    refreshSetupPanelSummaries();
  }

  async function loadExisting() {
    if (!state.projectFolder) {
      renderTable();
      return;
    }
    try {
      const incomingScenes = state.scenes.map((scene) => normalizeScene(scene));
      const data = await postJson("/vrgdg/storyboard/load", { project_folder: state.projectFolder });
      const saved = data.storyboard || {};
      if (!hasIncomingProjectVideoEngine) {
        state.projectVideoEngine = normalizeStoryboardProjectVideoEngine(saved.project_video_engine || saved.projectVideoEngine || state.projectVideoEngine);
      }
      const savedReferences = normalizeReferenceBuilderCatalog(saved.reference_builder || saved.referenceBuilder || {});
      const currentHasSubjects = Array.isArray(state.referenceBuilder?.subjects) && state.referenceBuilder.subjects.length > 0;
      const currentHasLocations = Array.isArray(state.referenceBuilder?.locations) && state.referenceBuilder.locations.length > 0;
      const currentLocationsCleared = Boolean(state.referenceBuilder?.locations_cleared);
      if ((!currentHasSubjects && savedReferences.subjects.length) || (!currentHasLocations && !currentLocationsCleared && savedReferences.locations.length)) {
        const nextReferences = {
          subjects: currentHasSubjects ? state.referenceBuilder.subjects : savedReferences.subjects,
          locations: currentLocationsCleared ? [] : (currentHasLocations ? state.referenceBuilder.locations : savedReferences.locations),
          locations_cleared: currentLocationsCleared,
        };
        state.referenceBuilder = normalizeReferenceBuilderCatalog(nextReferences);
      } else if (!currentHasSubjects && !currentHasLocations && !currentLocationsCleared && (savedReferences.subjects.length || savedReferences.locations.length)) {
        state.referenceBuilder = mergeReferenceBuilderCatalog(state.referenceBuilder, savedReferences);
      }
      if (Array.isArray(saved.scenes) && saved.scenes.length) {
        const savedScenes = saved.scenes.map((scene, index) => normalizeScene(scene, index));
        const scenesToShow = incomingScenes.length ? incomingScenes : savedScenes;
        state.scenes = scenesToShow.map((fresh, index) => {
          const normalized = savedScenes.find((item) => item.id === fresh.id)
            || savedScenes.find((item) => Number(item.scene_number) === Number(fresh.scene_number))
            || null;
          if (!normalized) return normalizeScene(fresh, index);
          const subjectRefs = incomingScenes.length ? (fresh.subject_refs || []) : (fresh.subject_refs?.length ? fresh.subject_refs : normalized.subject_refs);
          const subjects = subjectRefs?.length
            ? storyboardSubjectNamesFromRefs(subjectRefs)
            : Array.from(new Set([
              ...(fresh.subjects || []),
              ...(normalized.subjects || []),
            ].map((item) => String(item || "").trim()).filter(Boolean)));
          return {
            ...normalized,
            id: fresh.id || normalized.id,
            scene_number: fresh.scene_number || normalized.scene_number,
            label: fresh.label || normalized.label,
            video_prompt_type: payloadVideoPromptType || fresh.video_prompt_type || normalized.video_prompt_type,
            project_video_engine: state.projectVideoEngine,
            minimax_h3_mode: normalizeStoryboardMiniMaxH3Mode(fresh.minimax_h3_mode || normalized.minimax_h3_mode),
            timeline_start: Number(fresh.timeline_start ?? normalized.timeline_start ?? 0),
            timeline_end: Number(fresh.timeline_end ?? normalized.timeline_end ?? 0),
            exact_duration: Math.max(0, Number(fresh.exact_duration ?? normalized.exact_duration ?? 0)),
            lyrics: fresh.lyrics || normalized.lyrics,
            lyric_section: fresh.lyric_section || normalized.lyric_section,
            story_beat: fresh.story_beat || normalized.story_beat,
            audio_direction: fresh.audio_direction || normalized.audio_direction,
            continuity: fresh.continuity || normalized.continuity,
            flf_start_state: fresh.flf_start_state || normalized.flf_start_state,
            flf_transformation: fresh.flf_transformation || normalized.flf_transformation,
            flf_end_state: fresh.flf_end_state || normalized.flf_end_state,
            flf_carry_forward: fresh.flf_carry_forward || normalized.flf_carry_forward,
            performance_mode: fresh.performance_mode || normalized.performance_mode || state.performanceMode,
            prompt_summary: state.mode === "image_to_video_prep" ? (fresh.prompt_summary || normalized.prompt_summary) : "",
            motion_summary: fresh.motion_summary || normalized.motion_summary,
            temporal_world_effect_override: fresh.temporal_world_effect_override || normalized.temporal_world_effect_override || "global",
            temporal_world_effect_custom: fresh.temporal_world_effect_custom || normalized.temporal_world_effect_custom || "",
            image_path: fresh.image_path || normalized.image_path,
            no_character_present: Boolean(fresh.no_character_present || normalized.no_character_present),
            subjects,
            subject_refs: fresh.no_character_present || normalized.no_character_present ? [] : subjectRefs,
            setting: currentLocationsCleared ? "" : (fresh.location_ref?.name || normalized.setting || fresh.setting),
            location_ref: currentLocationsCleared ? null : (incomingScenes.length ? fresh.location_ref : (fresh.location_ref || normalized.location_ref)),
          };
        });
        if (currentLocationsCleared) {
          state.scenes.forEach((scene) => {
            scene.location_ref = null;
            scene.setting = "";
          });
        }
        absorbSceneReferencesIntoCatalog(state.scenes);
      }
      // The current video mode decides the opening workspace every time. Do not
      // restore a stale Image Prep/Video Prep tab from an earlier visit.
      state.mode = openingMode;
      state.performanceMode = normalizeStoryboardPerformanceMode(saved.performance_mode || saved.performanceMode || state.performanceMode);
      state.shortFilmPlanningMode = normalizeStoryboardShortFilmPlanningMode(saved.short_film_planning_mode || saved.shortFilmPlanningMode || state.shortFilmPlanningMode);
      shortFilmPlanningModeSelect.value = state.shortFilmPlanningMode;
      state.customCameraFlowSequence = normalizeStoryboardCustomCameraFlowSequence(
        saved.custom_camera_flow_sequence
        || saved.customCameraFlowSequence
        || saved.builder_storyboard_defaults?.custom_camera_flow_sequence
        || saved.builderStoryboardDefaults?.custom_camera_flow_sequence
        || state.customCameraFlowSequence,
      );
      if (saved.camera_flow && STORYBOARD_CAMERA_FLOW_PRESETS[saved.camera_flow]) {
        state.cameraFlow = saved.camera_flow;
        cameraFlowSelect.value = state.cameraFlow;
      }
      if (saved.image_shot_flow && imageShotFlowPresets[saved.image_shot_flow]) {
        state.imageShotFlow = saved.image_shot_flow;
        imageShotSelect.value = state.imageShotFlow;
      }
      state.imageAesthetic = String(saved.image_aesthetic || saved.imageAesthetic || state.imageAesthetic || "");
      if (!imageAestheticPresets.some((preset) => preset.value === state.imageAesthetic)) state.imageAesthetic = imageAestheticPresets[0]?.value || "";
      imageAestheticSelect.value = state.imageAesthetic;
      state.videoStyle = String(saved.video_style || saved.videoStyle || state.videoStyle || "");
      if (!MINIMAX_VIDEO_STYLE_PRESETS.some((preset) => preset.value === state.videoStyle)) state.videoStyle = "";
      videoStyleSelect.value = state.videoStyle;
      state.videoStyleCustom = String(saved.video_style_custom || saved.videoStyleCustom || state.videoStyleCustom || "");
      videoStyleCustomInput.value = state.videoStyleCustom;
      state.temporalWorldEffect = String(saved.temporal_world_effect || saved.temporalWorldEffect || state.temporalWorldEffect || "");
      if (!MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.some((preset) => preset.value === state.temporalWorldEffect)) state.temporalWorldEffect = "";
      temporalEffectSelect.value = state.temporalWorldEffect;
      state.temporalWorldEffectCustom = String(saved.temporal_world_effect_custom || saved.temporalWorldEffectCustom || state.temporalWorldEffectCustom || "");
      temporalEffectCustomInput.value = state.temporalWorldEffectCustom;
      state.temporalAllowBackgroundExtras = (saved.temporal_allow_background_extras ?? saved.temporalAllowBackgroundExtras ?? state.temporalAllowBackgroundExtras) !== false;
      temporalExtrasInput.checked = state.temporalAllowBackgroundExtras;
      state.temporalBackgroundIntensity = storyboardTemporalIntensity(saved.temporal_background_intensity ?? saved.temporalBackgroundIntensity ?? state.temporalBackgroundIntensity);
      temporalIntensityInput.value = String(state.temporalBackgroundIntensity);
      state.temporalEnvironmentTimePassage = (saved.temporal_environment_time_passage ?? saved.temporalEnvironmentTimePassage ?? state.temporalEnvironmentTimePassage) !== false;
      temporalEnvironmentInput.checked = state.temporalEnvironmentTimePassage;
      state.temporalProtectedCharacters = storyboardTemporalProtectedMode(saved.temporal_protected_characters || saved.temporalProtectedCharacters || state.temporalProtectedCharacters);
      temporalProtectedSelect.value = state.temporalProtectedCharacters;
      state.temporalProtectedCustom = String(saved.temporal_protected_custom || saved.temporalProtectedCustom || state.temporalProtectedCustom || "");
      temporalProtectedCustomInput.value = state.temporalProtectedCustom;
      state.fxPreset = String(saved.fx_preset || saved.fxPreset || saved.builder_storyboard_defaults?.fx_preset || saved.builderStoryboardDefaults?.fx_preset || state.fxPreset || "");
      if (!STORYBOARD_FX_PRESETS.some((preset) => preset.value === state.fxPreset)) state.fxPreset = "";
      fxSelect.value = state.fxPreset;
      state.fxCustomJson = String(saved.fx_custom_json || saved.fxCustomJson || saved.builder_storyboard_defaults?.fx_custom_json || saved.builderStoryboardDefaults?.fx_custom_json || state.fxCustomJson || "");
      fxCustomInput.value = state.fxCustomJson;
      state.globalConsistencyPhrase = String(saved.global_consistency_phrase || saved.globalConsistencyPhrase || state.globalConsistencyPhrase || "");
      consistencyInput.value = state.globalConsistencyPhrase;
      state.performanceStyle = String(saved.performance_style_default || saved.performance_style || state.performanceStyle || "");
      if (!performanceStylePresets.some((preset) => preset.value === state.performanceStyle)) state.performanceStyle = performanceStylePresets[0]?.value || "";
      performanceSelect.value = state.performanceStyle;
      state.facialPerformance = String(saved.facial_performance_default || saved.facial_performance || state.facialPerformance || "");
      if (!facialPerformancePresets.some((preset) => preset.value === state.facialPerformance)) state.facialPerformance = facialPerformancePresets[0]?.value || "";
      state.facialPerformanceCustom = String(saved.facial_performance_custom_default || saved.facial_performance_custom || state.facialPerformanceCustom || "");
      facialSelect.value = state.facialPerformance;
      facialCustomInput.value = state.facialPerformanceCustom;
      state.cameraMotionSpeed = storyboardSpeedValue(saved.camera_motion_speed ?? saved.motion_defaults?.camera_motion_speed ?? state.cameraMotionSpeed, 4);
      state.characterMotionSpeed = storyboardSpeedValue(saved.character_motion_speed ?? saved.motion_defaults?.character_motion_speed ?? state.characterMotionSpeed, 4);
      state.cutFrequency = storyboardCutFrequencyValue(saved.minimax_h3_cut_frequency ?? saved.cut_frequency ?? state.cutFrequency);
      cameraSpeedInput.value = String(state.cameraMotionSpeed);
      characterSpeedInput.value = String(state.characterMotionSpeed);
      cutFrequencyInput.value = String(state.cutFrequency);
      state.storyLayer = normalizeStoryLayer(saved.story_layer || saved.storyLayer || {});
      state.scriptImport = normalizeStoryboardScriptImportState(saved.script_import || saved.scriptImport || state.scriptImport || {});
      storyLayerEnabledInput.checked = state.storyLayer.enabled !== false;
      overallStoryIdeaInput.value = state.storyLayer.overall_story_idea || "";
      userStoryArcInput.value = state.storyLayer.user_story_arc || "";
      songStoryBriefInput.value = state.storyLayer.song_story_brief || "";
      lyricStoryStrengthInput.value = String(state.storyLayer.lyric_story_strength ?? 7);
      syncLyricStoryStrengthLabel();
      refreshCameraFlowInfo();
      refreshImageShotInfo();
      refreshImageAestheticInfo();
      refreshVideoStyleInfo();
      refreshTemporalEffectInfo();
      refreshFxInfo();
      refreshConsistencyInfo();
      refreshCameraSpeedInfo();
      refreshCutFrequencyInfo();
      refreshPerformanceInfo();
      refreshCharacterSpeedInfo();
      refreshFacialInfo();
      setMode(state.mode);
      syncReferenceMappingsToVideoCreator();
    } catch (error) {
      createToast(String(error?.message || error), true);
      renderTable();
    }
  }

  async function saveStoryboard() {
    if (!state.projectFolder) {
      createToast("Save the AI Video Builder project first so Storyboard Builder knows where to write files.", true);
      return;
    }
    state.saving = true;
    save.disabled = true;
    try {
      syncStoryLayerFromInputs();
      state.scenes.forEach((scene) => {
        if (state.projectVideoEngine !== "minimax_h3" && String(scene.video_prompt || "").trim() && normalizeVideoPromptOrigin(scene.video_prompt_origin) === "gemma") {
          scene.video_prompt = enforceStoryboardVideoFacialRequirements(scene.video_prompt, scene);
        }
      });
      const data = await postJson("/vrgdg/storyboard/save", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      // The Storyboard is already saved. Do not trigger a redundant parent
      // Video Builder session save from this completed save action.
      syncStoryLayerFromInputs({ notify: false });
      createToast(`Storyboard saved:\n${data.storyboard?.path || ""}`);
    } catch (error) {
      createToast(String(error?.message || error), true);
    } finally {
      save.disabled = false;
      state.saving = false;
    }
  }

  async function exportPromptFiles() {
    if (!state.projectFolder) {
      createToast("Save the AI Video Builder project first so Storyboard Builder knows where to export prompt files.", true);
      return;
    }
    exportPrompts.disabled = true;
    try {
      state.scenes.forEach((scene) => {
        if (String(scene.image_prompt || "").trim()) scene.image_prompt = ensureStoryboardReferenceOpening(scene.image_prompt, scene, state.imageMode);
        if (state.projectVideoEngine !== "minimax_h3" && String(scene.video_prompt || "").trim() && normalizeVideoPromptOrigin(scene.video_prompt_origin) === "gemma") {
          scene.video_prompt = enforceStoryboardVideoFacialRequirements(scene.video_prompt, scene);
        }
      });
      const data = await postJson("/vrgdg/storyboard/export_prompts", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      if (state.onPromptsExported) {
        state.onPromptsExported({
          ...storyboardDefaultsPayload(),
          story_layer: normalizeStoryLayer(state.storyLayer),
          scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        });
      }
      const destination = state.onPromptsExported
        ? " and copied them into matching Video Builder timeline segments. Timeline segments were not created or replaced."
        : " to files only. The Video Builder timeline was not created or replaced.";
      createToast(`Exported ${data.scene_count || 0} scene prompt rows${destination}\n\nText:\n${data.t2i_prompts_path}\n${data.i2v_prompts_path}\nJSON:\n${data.t2i_prompts_json_path || ""}\n${data.video_prompts_json_path || ""}`);
    } catch (error) {
      createToast(String(error?.message || error), true);
    } finally {
      exportPrompts.disabled = false;
    }
  }

  async function copyStoryboardForGpt() {
    try {
      const payload = storyboardGptPayload(state);
      const text = JSON.stringify(payload, null, 2);
      await copyTextToClipboard(text);
      openStoryboardGptUrl(payload);
      createToast(`Copied Storyboard GPT JSON for ${payload.scenes.length} scenes and opened GPT.`);
    } catch (error) {
      createToast(`Could not copy Storyboard GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  async function copySceneForGpt(scene) {
    try {
      const normalized = normalizeScene(scene, 0);
      const payload = storyboardGptPayload(state, [scene]);
      const text = JSON.stringify(payload, null, 2);
      await copyTextToClipboard(text);
      openStoryboardGptUrl(payload);
      createToast(`Copied GPT JSON for ${normalized.label || `Scene ${normalized.scene_number}`} and opened GPT.`);
    } catch (error) {
      createToast(`Could not copy scene GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  function showStoryLayerGptHandoff(payloadJson, chatWindow = null) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100014;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(900px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 22px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
    const title = document.createElement("div");
    title.innerHTML = "<div style=\"font-size:17px;font-weight:900;color:#cffafe;\">GPT Story JSON</div><div style=\"font-size:12px;color:#cbd5e1;margin-top:3px;\">Attach or paste the JSON, then send an explicit request to process it.</div>";
    const close = makeButton("Close");
    header.append(title, close);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;overflow:auto;";
    const status = document.createElement("div");
    status.style.cssText = "font-size:12px;color:#94a3b8;min-height:18px;";
    const text = document.createElement("textarea");
    text.value = payloadJson;
    text.spellcheck = false;
    text.style.cssText = "min-height:360px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;white-space:pre;overflow:auto;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;";
    const copy = makeButton("Copy JSON", "primary");
    const copyRequest = makeButton("Copy Request");
    const openChat = makeButton("Open ChatGPT", "primary");
    actions.append(copy, copyRequest, openChat, close);
    body.append(status, text, actions);
    box.append(header, body);
    backdrop.append(box);
    document.body.append(backdrop);
    const closeModal = () => backdrop.remove();
    const copyJson = async () => {
      try {
        await copyTextToClipboard(text.value);
        status.textContent = "Copied JSON to clipboard. Paste it into ChatGPT.";
        status.style.color = "#67e8f9";
      } catch (error) {
        status.textContent = "Clipboard copy was blocked. Select the JSON above and copy it manually.";
        status.style.color = "#fbbf24";
      }
    };
    const copyRequestText = async () => {
      try {
        await copyTextToClipboard("Process the attached story-layer planning JSON now. Do not ask what I want done. Use its task_instruction and project_inputs, then return only the final JSON with overall_story_idea, user_story_arc, and song_story_brief.");
        status.textContent = "Copied the request text. Paste it into ChatGPT after attaching the JSON.";
        status.style.color = "#67e8f9";
      } catch (error) {
        status.textContent = "Clipboard copy was blocked. Manually type: Process the attached story-layer planning JSON now and return the final JSON.";
        status.style.color = "#fbbf24";
      }
    };
    close.onclick = closeModal;
    copy.onclick = copyJson;
    copyRequest.onclick = copyRequestText;
    openChat.onclick = () => {
      if (chatWindow && !chatWindow.closed) chatWindow.focus();
      else window.open(STORY_LAYER_CHATGPT_URL, "_blank", "noopener,noreferrer");
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
    text.focus();
    text.select();
  }

  async function copyStoryLayerForGpt() {
    const progress = createStoryboardProgressWindow("GPT Story Preparation");
    try {
      progress.set("Preparing Auto Mode story context...", 12);
      syncStoryLayerFromInputs();
      if (state.onPrepareStoryContext) {
        const prepared = await state.onPrepareStoryContext({
          setProgress: (message, percent) => progress.set(String(message || "Preparing story context..."), Number(percent || 50)),
        });
        if (prepared?.reference_builder || prepared?.referenceBuilder) {
          state.referenceBuilder = normalizeReferenceBuilderCatalog(prepared.reference_builder || prepared.referenceBuilder);
        }
        if (prepared?.source_lyrics || prepared?.sourceLyrics) {
          state.lineMappingLyrics = String(prepared.source_lyrics || prepared.sourceLyrics || "");
        }
      }
      progress.set("Packaging lyrics, descriptions, presets, and story settings...", 82);
      const payload = storyLayerGptPayload(state);
      const payloadText = JSON.stringify(payload, null, 2);
      let clipboardCopied = true;
      try {
        await copyTextToClipboard(payloadText);
      } catch (error) {
        clipboardCopied = false;
      }
      const lyricCount = payload.project_inputs.ordered_lyrics.length;
      const sceneCount = payload.project_inputs.scenes.length;
      const sourceLyricsPresent = Boolean(payload.project_inputs.source_lyrics);
      const lyricStatus = lyricCount ? `${lyricCount} lyric entries` : (sourceLyricsPresent ? "full pasted lyrics" : "no lyrics");
      progress.set("Story context ready. JSON review window opened.", 100);
      progress.close(1200);
      showStoryLayerGptHandoff(payloadText);
      createToast(`${clipboardCopied ? "Copied" : "Prepared"} Story Layer JSON (${sceneCount} scenes, ${lyricStatus}). Use Open ChatGPT in the JSON window.`);
    } catch (error) {
      progress.set(`Story preparation failed:\n${String(error?.message || error)}`, 100);
      progress.close(5000);
      createToast(`Could not copy Story Layer GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  function parseStoryLayerImportJson(rawText) {
    const data = JSON.parse(imagePromptImportJsonText(rawText));
    const source = data?.story_layer && typeof data.story_layer === "object"
      ? { ...data, ...data.story_layer }
      : data;
    if (!source || typeof source !== "object" || Array.isArray(source)) {
      throw new Error("Story JSON must be an object.");
    }
    const stringifyStoryValue = (raw) => {
      if (raw === undefined || raw === null) return "";
      if (typeof raw === "string" || typeof raw === "number" || typeof raw === "boolean") return String(raw).trim();
      if (Array.isArray(raw)) return raw.map(stringifyStoryValue).filter(Boolean).join("\n");
      if (typeof raw === "object") {
        return Object.entries(raw)
          .map(([key, value]) => {
            const text = stringifyStoryValue(value);
            return text ? `${key}:\n${text}` : "";
          })
          .filter(Boolean)
          .join("\n\n");
      }
      return "";
    };
    const value = (...keys) => {
      for (const key of keys) {
        if (source[key] !== undefined && source[key] !== null) return stringifyStoryValue(source[key]);
      }
      return "";
    };
    const result = {
      overall_story_idea: value("overall_story_idea", "overallStoryIdea", "story_idea", "storyIdea"),
      user_story_arc: value("user_story_arc", "userStoryArc", "story_arc", "storyArc"),
      song_story_brief: value("song_story_brief", "songStoryBrief", "story_brief", "storyBrief", "brief"),
    };
    if (!result.overall_story_idea && !result.user_story_arc && !result.song_story_brief) {
      throw new Error("No overall_story_idea, user_story_arc/story_arc, or song_story_brief/story_brief was found.");
    }
    return result;
  }

  function openImportStoryJsonModal() {
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100013;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(840px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = "<div style=\"font-size:17px;font-weight:900;color:#cffafe;\">Import Story JSON</div><div style=\"font-size:12px;color:#cbd5e1;margin-top:3px;\">Paste the GPT response or load a .json file. This fills the overall idea, story arc, and story brief.</div>";
    const importClose = makeButton("Close");
    importHeader.append(importTitle, importClose);
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json,application/json,text/plain";
    const input = document.createElement("textarea");
    input.placeholder = '{\n  "overall_story_idea": "...",\n  "user_story_arc": "...",\n  "song_story_brief": "..."\n}';
    input.spellcheck = false;
    input.style.cssText = "min-height:300px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const apply = makeButton("Import Story", "purple");
    actions.append(cancel, apply);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto;";
    body.append(fileInput, input, status, actions);
    importBox.append(importHeader, body);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    const closeImport = () => importBackdrop.remove();
    importClose.onclick = closeImport;
    cancel.onclick = closeImport;
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) closeImport();
    });
    fileInput.onchange = async () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      input.value = await file.text();
      status.textContent = `Loaded ${file.name}. Review it, then click Import Story.`;
    };
    apply.onclick = () => {
      try {
        const imported = parseStoryLayerImportJson(input.value);
        if (imported.overall_story_idea) {
          state.storyLayer.overall_story_idea = imported.overall_story_idea;
          overallStoryIdeaInput.value = imported.overall_story_idea;
        }
        if (imported.user_story_arc) {
          state.storyLayer.user_story_arc = imported.user_story_arc;
          userStoryArcInput.value = imported.user_story_arc;
        }
        if (imported.song_story_brief) {
          state.storyLayer.song_story_brief = imported.song_story_brief;
          songStoryBriefInput.value = imported.song_story_brief;
        }
        syncStoryLayerFromInputs({ notify: true });
        status.textContent = "Story Layer fields updated.";
        status.style.color = "#67e8f9";
        createToast("Imported story idea, story arc, and story brief.");
        closeImport();
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
      }
    };
    input.focus();
  }

  function imagePromptImportJsonText(rawText) {
    const text = String(rawText || "").trim();
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced) return fenced[1].trim();
    const firstArray = text.indexOf("[");
    const firstObject = text.indexOf("{");
    const starts = [firstArray, firstObject].filter((index) => index >= 0);
    if (!starts.length) return text;
    const start = Math.min(...starts);
    const end = Math.max(text.lastIndexOf("]"), text.lastIndexOf("}"));
    return end > start ? text.slice(start, end + 1).trim() : text.slice(start).trim();
  }

  function parseImagePromptImportJson(rawText) {
    const text = imagePromptImportJsonText(rawText);
    if (!text) return [];
    const data = JSON.parse(text);
    const source = Array.isArray(data)
      ? data
      : Array.isArray(data.prompts)
        ? data.prompts
        : Array.isArray(data.scenes)
          ? data.scenes
          : data && typeof data === "object"
            ? Object.entries(data).map(([key, value]) => {
              if (value && typeof value === "object") return { scene: key, ...value };
              return { scene: key, prompt: value };
            })
            : [];
    const rows = [];
    for (const item of source) {
      if (!item || typeof item !== "object") continue;
      const sceneRaw = item.scene_number ?? item.sceneNumber ?? item.scene ?? item.number ?? item.id ?? "";
      const sceneNumber = Number(String(sceneRaw).match(/\d+/)?.[0] || sceneRaw || 0);
      const prompt = String(
        item.image_prompt
        ?? item.text_to_image_prompt
        ?? item.t2i_prompt
        ?? item.prompt
        ?? item.text
        ?? "",
      ).trim();
      if (!sceneNumber || !prompt) continue;
      rows.push({ sceneNumber, prompt });
    }
    return rows;
  }

  function openImportImagePromptsFromGptModal() {
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100013;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(840px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = `<div style="font-size:17px;font-weight:900;color:#cffafe;">Import Image Prompts From GPT</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Paste the JSON code block from the Krea 2 text-to-image GPT. This updates Image Prep prompts only.</div>`;
    const importClose = makeButton("Close");
    importHeader.append(importTitle, importClose);
    const help = document.createElement("div");
    help.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    help.innerHTML = `Accepted examples:<br><code>[{"scene":1,"image_prompt":"..."},{"scene":2,"prompt":"..."}]</code><br><code>{"scene1":"prompt text","scene2":"prompt text"}</code>`;
    const input = document.createElement("textarea");
    input.placeholder = "Paste GPT JSON output here...";
    input.spellcheck = false;
    input.style.cssText = "min-height:340px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const apply = makeButton("Import Image Prompts", "purple");
    actions.append(cancel, apply);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto;";
    body.append(help, input, status, actions);
    importBox.append(importHeader, body);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    const closeImport = () => importBackdrop.remove();
    importClose.onclick = closeImport;
    cancel.onclick = closeImport;
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) closeImport();
    });
    apply.onclick = () => {
      try {
        const rows = parseImagePromptImportJson(input.value);
        if (!rows.length) throw new Error("No usable image prompts found. Make sure each row has a scene number and image_prompt or prompt.");
        let updated = 0;
        const missing = [];
        for (const row of rows) {
          const scene = state.scenes.find((item) => Number(item.scene_number) === Number(row.sceneNumber));
          if (!scene) {
            missing.push(row.sceneNumber);
            continue;
          }
          scene.image_prompt = row.prompt;
          scene.prompt_summary = "";
          scene.status = "image_prompt_ready";
          updated += 1;
        }
        renderTable();
        status.textContent = `Updated ${updated} Image Prep prompt${updated === 1 ? "" : "s"}${missing.length ? `; missing scenes: ${missing.join(", ")}` : ""}.`;
        status.style.color = updated ? "#67e8f9" : "#fbbf24";
        createToast(`Imported ${updated} image prompt${updated === 1 ? "" : "s"} from GPT.`);
        if (updated) closeImport();
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
        createToast(String(error?.message || error), true);
      }
    };
    input.focus();
  }

  function storyboardGemmaPayload(scene, overrides = {}) {
    const payload = storyboardGptPayload(state, [scene]);
    const imageStyle = normalizeStoryLayer(state.storyLayer);
    return {
      ...(state.gemmaSettings || {}),
      ...overrides,
      storyboard_payload: payload,
      image_world_style: imageStyle.image_world_style,
      image_custom_style_direction: imageStyle.image_custom_style_direction,
      max_new_tokens: 2000,
      temperature: 0.35,
      top_p: 0.90,
    };
  }

  async function createSceneImagePromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending image scene card to ${runnerName}...\nThis creates the text-to-image prompt for Image Prep.`, progressPercent);
      const data = await postJson("/vrgdg/storyboard/gemma_image_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter, max_new_tokens: 1200 }), STORYBOARD_GEMMA_TIMEOUT_MS);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: ${genericName} response received.\nRunner: ${data.runner || runnerName}\nSaving image prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const prompt = ensureStoryboardReferenceOpening(applyStoryboardTriggerPhrases(data.prompt, scene), scene, state.imageMode);
      if (!prompt) throw new Error(`${genericName} returned an empty Storyboard image prompt.`);
      scene.image_prompt = prompt;
      scene.prompt_summary = "";
      scene.status = "image_prompt_ready";
      if (!quiet) createToast(`${genericName} created image prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || runnerName}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`${genericName} Storyboard image prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  function enforceStoryboardVideoFacialRequirements(prompt, scene) {
    let text = String(prompt || "").trim();
    const normalized = normalizeScene(scene, 0);
    const promptMentionsFace = /\b(?:woman|man|girl|boy|person|subject|singer|rapper|performer|speaker|character|face|eyes?|brows?|gaze|mouth|jaw|cheeks?|expression|smile|frown|sings?|singing|says|speaks?)\b/i.test(text);
    const hasCharacter = !normalized.no_character_present && (
      (Array.isArray(normalized.subject_refs) && normalized.subject_refs.length)
      || (Array.isArray(normalized.subjects) && normalized.subjects.length)
      || promptMentionsFace
    );
    if (!text || !hasCharacter) return text;
    const vocalStatus = normalized.vocal_status || {};
    const promptSaysSinging = /\b(?:sings?|singing|raps?|rapping)\b/i.test(text);
    const isSinging = promptSaysSinging || (String(normalized.performance_mode || vocalStatus.performance_mode || state.performanceMode || "").trim() === "singing"
      && vocalStatus.should_lip_sync !== false
      && !vocalStatus.instrumental
      && !vocalStatus.no_lip_sync
      && !normalized.lyric_no_lip_sync
      && Boolean(String(vocalStatus.lyric_text || normalized.lyrics || "").trim()));
    if (isSinging) {
      text = text
        .replace(/\bwith\s+a\s+quiet,\s*internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bwith\s+quiet\s+internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bquiet,\s*internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+intensity\b/gi, "controlled intensity")
        .replace(/\bquiet\s+performance\b/gi, "controlled performance")
        .replace(/\bquiet\s+emotion\b/gi, "restrained emotion")
        .replace(/\bquiet\s+singing\b/gi, "focused singing");
    }
    const hasBlink = /\bblink\w*\b/i.test(text);
    const hasEyeMovement = /\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b/i.test(text);
    const additions = [];
    if (!hasEyeMovement) additions.push("subtle natural eye movement");
    if (!hasBlink) additions.push("occasional natural blinking");
    if (additions.length) {
      const insert = `, ${additions.join(", ")}`;
      const faceSentence = text.match(/([^.]*(?:face|eyes?|brows?|gaze|expression)[^.]*)(\.)/i);
      if (faceSentence && typeof faceSentence.index === "number") {
        const nextSentence = `${faceSentence[1].trimEnd()}${insert}`;
        text = `${text.slice(0, faceSentence.index)}${nextSentence}${text.slice(faceSentence.index + faceSentence[1].length)}`;
      } else {
        text = `${text.replace(/\.+\s*$/, "")} with ${additions.join(", ")}.`;
      }
    }
    return text.replace(/\s{2,}/g, " ").trim();
  }

  function applyStoryboardTriggerPhrases(prompt, scene) {
    let text = enforceStoryboardVideoFacialRequirements(prompt, scene);
    const normalized = normalizeScene(scene, 0);
    const refs = normalizeReferenceBuilderCatalog(state.referenceBuilder || {});
    const parts = { start: [], end: [] };
    const add = (trigger, position = "start") => {
      const value = String(trigger || "").trim();
      if (!value) return;
      const key = position === "end" ? "end" : "start";
      if (!parts[key].some((item) => item.toLowerCase() === value.toLowerCase())) parts[key].push(value);
    };
    const subjectPosition = refs.subject_trigger_position === "end" ? "end" : "start";
    const locationPosition = refs.location_trigger_position === "end" ? "end" : "start";
    (Array.isArray(normalized.subject_refs) ? normalized.subject_refs : []).forEach((subject) => {
      add(subject.trigger_phrase || subject.trigger || subject.Trigger, subjectPosition);
    });
    if (normalized.location_ref) {
      add(normalized.location_ref.trigger_phrase || normalized.location_ref.trigger || normalized.location_ref.Trigger, locationPosition);
    }
    add(normalized.trigger_phrase || normalized.trigger || normalized.Trigger, normalized.trigger_position === "end" ? "end" : "start");
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
    [...parts.start, ...parts.end]
      .sort((a, b) => b.length - a.length)
      .forEach((trigger) => {
        text = stripBoundaryTrigger(text, trigger);
      });
    if (parts.start.length) {
      const prefix = parts.start.join(", ");
      if (!text.toLowerCase().startsWith(prefix.toLowerCase())) text = text ? `${prefix}, ${text}` : prefix;
    }
    if (parts.end.length) {
      const suffix = parts.end.join(", ");
      if (!text.toLowerCase().endsWith(suffix.toLowerCase())) text = text ? `${text}, ${suffix}` : suffix;
    }
    return text;
  }

  function applyBuilderManagedFx(prompt, presetValue = "", customJson = "") {
    let text = String(prompt || "").trim();
    if (!text || !presetValue) return text;
    const timestampPattern = /((?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|\[\s*Shot\s+\d+[^\]]*\])\s*\n?)([\s\S]*?)(?=\n\s*(?:\[\s*\d+(?:\.\d+)?s?\s*[-–—]\s*\d+(?:\.\d+)?s?\s*\]|\[\s*Shot\s+\d+[^\]]*\])|\n\s*(?:Audio(?:\s+1)?|overall_soundscape|non_diegetic_music|Continuity)\s*:|$)/gi;
    let index = 0;
    let matched = false;
    text = text.replace(timestampPattern, (whole, header, body) => {
      matched = true;
      const contract = storyboardFxContract(presetValue, customJson, index++);
      if (!contract || String(body || "").includes(contract.cue)) return whole;
      return `${header}${String(body || "").trim()} FX accent: ${contract.cue} Keep the mapped subject stable and readable.\n`;
    });
    if (matched) return text.trim();
    const contract = storyboardFxContract(presetValue, customJson, 0);
    return contract ? `${text}\n\nFX accent inside this shot: ${contract.cue} Keep the mapped subject stable and readable.`.trim() : text;
  }

  async function createSceneVideoPromptWithGemma(scene, { quiet = false, unloadAfter = true, progress = null, progressPercent = 35, progressLabel = "" } = {}) {
    const normalized = normalizeScene(scene, 0);
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    try {
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: sending scene card to ${runnerName}...\nThis can take a minute depending on runner/model speed.`, progressPercent);
      const callbackPayload = storyboardGptPayload(state, [scene]);
      if (state.onBeforeCreateVideoPrompt) {
        await state.onBeforeCreateVideoPrompt(scene, {
          storyboardPayload: callbackPayload,
          progress,
          progressPercent,
          progressLabel,
        });
      }
      if (state.projectVideoEngine === "minimax_h3" && !state.onCreateVideoPrompt) {
        throw new Error("Open Storyboard Builder from the Video Builder so MiniMax can use the scene's H3 mode, ordered references, exact timing, and LLM instructions.");
      }
      const data = state.onCreateVideoPrompt
        ? await state.onCreateVideoPrompt(scene, {
          unloadAfter,
          storyboardPayload: callbackPayload,
          progress,
          progressPercent,
          progressLabel,
        })
        : await postJson("/vrgdg/storyboard/gemma_video_prompt", storyboardGemmaPayload(scene, { unload_after: unloadAfter }), STORYBOARD_GEMMA_TIMEOUT_MS);
      progress?.set(`${progressLabel || normalized.label || `Scene ${normalized.scene_number}`}: ${genericName} response received.\nRunner: ${data.runner || runnerName}\nSaving prompt into the scene card...`, Math.min(96, progressPercent + 45));
      const rawPrompt = String(data?.prompt || data || "").trim();
      const prompted = data?.already_finalized ? rawPrompt : applyStoryboardTriggerPhrases(rawPrompt, scene);
      const prompt = data?.already_finalized ? prompted : applyBuilderManagedFx(prompted, state.fxPreset, state.fxCustomJson);
      if (!prompt) throw new Error(`${genericName} returned an empty Storyboard video prompt.`);
      scene.video_prompt = prompt;
      scene.video_prompt_origin = "gemma";
      scene.status = "video_prompt_ready";
      if (!quiet) createToast(`${genericName} created video prompt for ${normalized.label || `Scene ${normalized.scene_number}`}.\nRunner: ${data.runner || runnerName}`);
      return prompt;
    } catch (error) {
      if (!quiet) createToast(`${genericName} Storyboard prompt failed:\n${String(error?.message || error)}`, true);
      throw error;
    } finally {
      renderTable();
    }
  }

  async function createScenePromptForActiveMode(scene, options = {}) {
    return state.mode === "image_to_video_prep"
      ? createSceneVideoPromptWithGemma(scene, options)
      : createSceneImagePromptWithGemma(scene, options);
  }

  const chooseVideoPromptGenerationScope = (scenes = []) => new Promise((resolve) => {
    const missingCount = scenes.filter((scene) => !String(scene.video_prompt || "").trim()).length;
    const completedCount = scenes.length - missingCount;
    const runnerName = promptRunnerName();
    const choiceBackdrop = document.createElement("div");
    choiceBackdrop.style.cssText = "position:fixed;inset:0;z-index:100060;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:22px;box-sizing:border-box;";
    const panel = document.createElement("div");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-labelledby", "vrgdg-video-all-choice-title");
    panel.style.cssText = "width:min(650px,calc(100vw - 44px));border:1px solid #155e75;border-radius:11px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 90px rgba(0,0,0,.7);overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "padding:16px 18px;background:#083f4f;border-bottom:1px solid #155e75;";
    const title = document.createElement("div");
    title.id = "vrgdg-video-all-choice-title";
    title.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    title.textContent = `${runnerName} Video All`;
    const subtitle = document.createElement("div");
    subtitle.style.cssText = "margin-top:4px;color:#bae6fd;font-size:12px;line-height:1.4;";
    subtitle.textContent = "Choose whether to preserve completed video prompts or regenerate every visible scene.";
    header.append(title, subtitle);
    const body = document.createElement("div");
    body.style.cssText = "padding:18px;display:flex;flex-direction:column;gap:14px;";
    const counts = document.createElement("div");
    counts.style.cssText = "border:1px solid #334155;border-radius:8px;background:#07111f;padding:12px;color:#e2e8f0;font-weight:800;line-height:1.45;";
    counts.textContent = `${missingCount} missing  •  ${completedCount} already complete  •  ${scenes.length} total visible scene${scenes.length === 1 ? "" : "s"}`;
    const guidance = document.createElement("div");
    guidance.style.cssText = "color:#cbd5e1;font-size:13px;line-height:1.5;";
    guidance.textContent = "Only Missing keeps every existing video prompt unchanged and creates prompts only for blank scenes. Redo All replaces the generated video prompt for every visible scene.";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;";
    const onlyMissing = makeButton(`Only Missing (${missingCount})`, "primary");
    onlyMissing.style.minHeight = "46px";
    onlyMissing.disabled = missingCount === 0;
    onlyMissing.title = missingCount
      ? "Keep completed prompts and create only the missing video prompts."
      : "Every visible scene already has a video prompt.";
    const redoAll = makeButton(`Redo All (${scenes.length})`);
    redoAll.style.cssText += "min-height:46px;border-color:#d97706;background:#78350f;color:#fef3c7;";
    redoAll.title = "Replace all existing video prompts for the visible scenes.";
    const cancel = makeButton("Cancel");
    cancel.style.cssText += "grid-column:1 / -1;min-height:40px;";
    actions.append(onlyMissing, redoAll, cancel);
    body.append(counts, guidance, actions);
    panel.append(header, body);
    choiceBackdrop.append(panel);
    document.body.append(choiceBackdrop);
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finish(null);
      }
    };
    const finish = (choice) => {
      document.removeEventListener("keydown", onKeyDown, true);
      choiceBackdrop.remove();
      resolve(choice);
    };
    onlyMissing.onclick = () => finish("missing");
    redoAll.onclick = () => finish("all");
    cancel.onclick = () => finish(null);
    choiceBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === choiceBackdrop) finish(null);
    });
    document.addEventListener("keydown", onKeyDown, true);
    requestAnimationFrame(() => (missingCount ? onlyMissing : redoAll).focus());
  });

  function isRecoverableStoryboardBatchError(error) {
    const message = String(error?.message || error || "").toLowerCase();
    return [
      "did not return valid json shot descriptions",
      "returned 0 shot descriptions",
      "returned an invalid number of shot descriptions",
      "returned an empty",
      "request timed out",
      "backend may still be processing",
      "repeated/thought",
      "unfilled template",
      "placeholder",
    ].some((phrase) => message.includes(phrase));
  }

  function showStoryboardBatchFailures(failures, retryHandler) {
    const items = Array.isArray(failures) ? failures : [];
    if (!items.length) return;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:18px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(980px,calc(100vw - 36px));max-height:calc(100vh - 36px);overflow:auto;border:1px solid #991b1b;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 22px 80px rgba(0,0,0,.65);padding:16px;box-sizing:border-box;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:17px;font-weight:900;color:#fecaca;">Storyboard skipped ${items.length} scene${items.length === 1 ? "" : "s"}</div><div style="font-size:12px;color:#cbd5e1;margin-top:5px;">Successful scenes were saved. Only these scenes will be retried.</div>`;
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:10px;margin-top:14px;";
    items.forEach((item) => {
      const card = document.createElement("details");
      card.open = true;
      card.style.cssText = "border:1px solid #7f1d1d;border-radius:7px;background:#1f0808;padding:9px;";
      const summary = document.createElement("summary");
      summary.style.cssText = "cursor:pointer;font-weight:900;color:#fca5a5;";
      summary.textContent = `${item.scene.label || `Scene ${item.scene.scene_number || "?"}`}: ${item.error}`;
      const raw = document.createElement("pre");
      raw.style.cssText = "white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto;margin:9px 0 0;color:#fecaca;font-size:11px;line-height:1.4;";
      raw.textContent = item.error;
      card.append(summary, raw);
      list.append(card);
    });
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:16px;";
    const close = makeButton("Close");
    const retry = makeButton(`Retry ${items.length} Failed Scene${items.length === 1 ? "" : "s"}`, "primary");
    retry.onclick = async () => {
      retry.disabled = true;
      retry.textContent = "Retrying...";
      try {
        backdrop.remove();
        await retryHandler(items);
      } catch (error) {
        createToast(String(error?.message || error), true);
        retry.disabled = false;
        retry.textContent = `Retry ${items.length} Failed Scene${items.length === 1 ? "" : "s"}`;
      }
    };
    close.onclick = () => backdrop.remove();
    actions.append(close, retry);
    box.append(title, list, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  }

  async function createAllPromptsWithGemma({ onlyMissing = false, failedSceneIds = [] } = {}) {
    const visibleScenes = currentRows();
    if (!visibleScenes.length) {
      createToast("No storyboard scenes found.", true);
      return;
    }
    const videoMode = state.mode === "image_to_video_prep";
    const promptKind = videoMode ? "video" : "image";
    const promptField = videoMode ? "video_prompt" : "image_prompt";
    const failedIds = new Set(failedSceneIds.map((value) => String(value)));
    const scenes = failedIds.size
      ? visibleScenes.filter((scene) => failedIds.has(String(scene.id || "")))
      : onlyMissing
      ? visibleScenes.filter((scene) => !String(scene[promptField] || "").trim())
      : visibleScenes;
    if (!scenes.length) {
      createToast(`Every visible scene already has a ${promptKind} prompt.`);
      return;
    }
    gemmaAllButton.disabled = true;
    const previousText = gemmaAllButton.textContent;
    const runnerName = promptRunnerName();
    const genericName = promptRunnerGenericName();
    const progress = createStoryboardProgressWindow(`Storyboard ${runnerName} All`);
    let created = 0;
    const failures = [];
    try {
      const keepLoaded = Boolean(keepGemmaLoadedInput.checked);
      progress.set(`${failedIds.size ? "Retrying failed Storyboard scenes" : `Starting Storyboard ${runnerName} All`}...\nMode: ${videoMode ? "Video Prep" : "Image Prep"}\nScope: ${failedIds.size ? `failed only (${scenes.length})` : onlyMissing ? `only missing (${scenes.length} of ${visibleScenes.length})` : `redo all (${scenes.length})`}\nKeep local LLM loaded: ${keepLoaded ? "yes" : "no"}`, 5);
      for (let index = 0; index < scenes.length; index += 1) {
        gemmaAllButton.textContent = `${runnerName} ${index + 1}/${scenes.length}`;
        const unloadAfter = keepLoaded ? index === scenes.length - 1 : true;
        const base = 8 + Math.round((index / Math.max(1, scenes.length)) * 84);
        const label = `${runnerName} All ${index + 1}/${scenes.length}: ${scenes[index].label || `Scene ${scenes[index].scene_number || index + 1}`}`;
        try {
          progress.set(`${label}\nCreating storyboard ${promptKind} prompt...`, base);
          await createScenePromptForActiveMode(scenes[index], { quiet: true, unloadAfter, progress, progressPercent: base, progressLabel: label });
          created += 1;
        } catch (error) {
          if (!isRecoverableStoryboardBatchError(error)) throw error;
          failures.push({ scene: scenes[index], error: String(error?.message || error) });
          progress.set(`${label} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      progress.set("Saving storyboard prompts...", 96);
      await saveStoryboard();
      progress.set(`${runnerName} All complete.\nCreated ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}${failures.length ? `. ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped` : ""}${onlyMissing ? "; existing prompts were preserved" : ""}.`, 100);
      progress.close(1800);
      createToast(`${genericName} created ${created} storyboard ${promptKind} prompt${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}${onlyMissing ? "; existing prompts were preserved" : ""}.`, Boolean(failures.length));
      if (failures.length) {
        showStoryboardBatchFailures(failures, (items) => createAllPromptsWithGemma({
          failedSceneIds: items.map((item) => item.scene.id),
        }));
      }
    } catch (error) {
      if (created > 0) {
        progress.set(`Saving ${created} completed prompt${created === 1 ? "" : "s"} before stopping...`, 96);
        await saveStoryboard();
      }
      progress.set(`${runnerName} All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, 100);
      createToast(`${runnerName} All stopped after ${created}/${scenes.length} scenes:\n${String(error?.message || error)}`, true);
    } finally {
      gemmaAllButton.disabled = false;
      gemmaAllButton.textContent = previousText;
      renderTable();
    }
  }

  async function startAllPromptsWithGemma() {
    const scenes = currentRows();
    if (!scenes.length) {
      createToast("No storyboard scenes found.", true);
      return;
    }
    if (state.mode !== "image_to_video_prep") {
      await createAllPromptsWithGemma();
      return;
    }
    const scope = await chooseVideoPromptGenerationScope(scenes);
    if (!scope) return;
    await createAllPromptsWithGemma({ onlyMissing: scope === "missing" });
  }

  stepPrompts.onclick = () => setMode("storyboard_prompts");
  stepPrep.onclick = () => setMode("image_to_video_prep");
  search.oninput = () => {
    state.query = search.value || "";
    renderTable();
  };
  cameraFlowSelect.onchange = async () => {
    const previous = state.cameraFlow;
    const next = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowSelect.value] ? cameraFlowSelect.value : "balanced";
    if (next === "custom") {
      state.cameraFlow = "custom";
      const imported = await openCustomCameraFlowDialog();
      if (imported) {
        state.customCameraFlowSequence = normalizeStoryboardCustomCameraFlowSequence(imported);
        state.cameraFlow = "custom";
      } else {
        state.cameraFlow = previous === "custom" && state.customCameraFlowSequence.length ? "custom" : (STORYBOARD_CAMERA_FLOW_PRESETS[previous] ? previous : "balanced");
      }
    } else {
      state.cameraFlow = next;
    }
    cameraFlowSelect.value = state.cameraFlow;
    refreshCameraFlowInfo();
    notifyStoryboardDefaultsChanged();
  };
  const refreshFxInfo = () => {
    const preset = storyboardFxPreset(state.fxPreset);
    const custom = state.fxPreset === "custom" ? normalizeStoryboardCustomFxJson(state.fxCustomJson) : null;
    fxInfo.textContent = state.fxPreset === ""
      ? preset.description
      : custom
        ? `${custom.label}: ${custom.cues.length} custom cue${custom.cues.length === 1 ? "" : "s"}. The Builder injects one cue into each finished timestamped shot after ${promptRunnerName()} returns.`
        : `${preset.description} The Builder injects one cue into each finished timestamped shot after ${promptRunnerName()} returns.`;
    fxCustomControls.style.display = state.fxPreset === "custom" ? "flex" : "none";
    refreshSetupPanelSummaries();
  };
  imageShotSelect.onchange = () => {
    state.imageShotFlow = imageShotFlowPresets[imageShotSelect.value] ? imageShotSelect.value : Object.keys(imageShotFlowPresets)[0] || "off";
    imageShotSelect.value = state.imageShotFlow;
    refreshImageShotInfo();
    notifyStoryboardDefaultsChanged();
  };
  imageAestheticSelect.onchange = () => {
    state.imageAesthetic = imageAestheticPresets.some((preset) => preset.value === imageAestheticSelect.value) ? imageAestheticSelect.value : imageAestheticPresets[0]?.value || "";
    imageAestheticSelect.value = state.imageAesthetic;
    refreshImageAestheticInfo();
    notifyStoryboardDefaultsChanged();
  };
  videoStyleSelect.onchange = () => {
    state.videoStyle = MINIMAX_VIDEO_STYLE_PRESETS.some((preset) => preset.value === videoStyleSelect.value) ? videoStyleSelect.value : "";
    videoStyleSelect.value = state.videoStyle;
    refreshVideoStyleInfo();
    notifyStoryboardDefaultsChanged();
  };
  videoStyleCustomInput.addEventListener("input", () => {
    state.videoStyleCustom = videoStyleCustomInput.value;
    refreshVideoStyleInfo();
  });
  videoStyleCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  temporalEffectSelect.onchange = () => {
    const previous = state.temporalWorldEffect;
    state.temporalWorldEffect = MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.some((preset) => preset.value === temporalEffectSelect.value)
      ? temporalEffectSelect.value
      : "";
    temporalEffectSelect.value = state.temporalWorldEffect;
    if (!previous && state.temporalWorldEffect) {
      state.temporalAllowBackgroundExtras = true;
      state.temporalEnvironmentTimePassage = true;
      temporalExtrasInput.checked = true;
      temporalEnvironmentInput.checked = true;
    }
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalEffectCustomInput.addEventListener("input", () => {
    state.temporalWorldEffectCustom = temporalEffectCustomInput.value;
    refreshTemporalEffectInfo();
  });
  temporalEffectCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  fxSelect.onchange = () => {
    state.fxPreset = STORYBOARD_FX_PRESETS.some((preset) => preset.value === fxSelect.value) ? fxSelect.value : "";
    fxSelect.value = state.fxPreset;
    refreshFxInfo();
    notifyStoryboardDefaultsChanged();
  };
  fxCustomInput.addEventListener("input", () => {
    state.fxCustomJson = fxCustomInput.value;
    refreshFxInfo();
  });
  fxCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  temporalExtrasInput.onchange = () => {
    state.temporalAllowBackgroundExtras = temporalExtrasInput.checked;
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalEnvironmentInput.onchange = () => {
    state.temporalEnvironmentTimePassage = temporalEnvironmentInput.checked;
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalIntensityInput.addEventListener("input", () => {
    state.temporalBackgroundIntensity = storyboardTemporalIntensity(temporalIntensityInput.value);
    refreshTemporalEffectInfo();
  });
  temporalIntensityInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  temporalProtectedSelect.onchange = () => {
    state.temporalProtectedCharacters = storyboardTemporalProtectedMode(temporalProtectedSelect.value);
    temporalProtectedSelect.value = state.temporalProtectedCharacters;
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalProtectedCustomInput.addEventListener("input", () => {
    state.temporalProtectedCustom = temporalProtectedCustomInput.value;
    refreshTemporalEffectInfo();
  });
  temporalProtectedCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  consistencyInput.addEventListener("input", () => {
    state.globalConsistencyPhrase = consistencyInput.value.trim();
    refreshConsistencyInfo();
  });
  consistencyInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cameraSpeedInput.addEventListener("input", () => {
    state.cameraMotionSpeed = storyboardSpeedValue(cameraSpeedInput.value, 4);
    cameraSpeedInput.value = String(state.cameraMotionSpeed);
    refreshCameraSpeedInfo();
  });
  cameraSpeedInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cameraSpeedHint.onclick = () => {
    window.alert([
      `Camera Motion Speed controls how much movement ${promptRunnerName()}/GPT should put into the camera plan for Video Prep.`,
      "",
      "0: locked-off static camera.",
      "1-3: slow, gentle camera motion; one simple move at most.",
      "4-6: controlled cinematic movement like tracking, pan, dolly, crane, reveal, or orbit.",
      "7-8: energetic movement with stronger tracking, orbit, whip pan, rise, reveal, or compound motion.",
      "9-10: fast action camera language; multiple coordinated moves can happen in one scene while keeping the subject readable.",
    ].join("\n"));
  };
  cutFrequencyInput.addEventListener("input", () => {
    state.cutFrequency = storyboardCutFrequencyValue(cutFrequencyInput.value);
    cutFrequencyInput.value = String(state.cutFrequency);
    refreshCutFrequencyInfo();
  });
  cutFrequencyInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cutFrequencyHint.onclick = () => {
    window.alert([
      "Cut Frequency controls editing inside each timeline segment.",
      "",
      "0: one smooth continuous take with no cuts.",
      "1-3: occasional cuts, scaled to the segment's exact duration.",
      "4-6: a moderate number of evenly spaced cuts.",
      "7-9: frequent cuts with short coherent coverage shots.",
      "10: maximum frequency — request a new continuity-preserving angle every second.",
      "",
      "Example: a 5-second segment at 10 starts with shot 1, then cuts at 1s, 2s, 3s, and 4s.",
      "LTX writes cuts in ordinary language such as 'then cut to'; MiniMax keeps its structured CUT TO format.",
      "Changing this setting affects newly generated prompts; it does not rewrite existing prompts automatically.",
    ].join("\n"));
  };
  cameraFlowApply.onclick = () => applyCameraFlow({ overwrite: false });
  cameraFlowReplace.onclick = () => applyCameraFlow({ overwrite: true });
  imageShotApply.onclick = () => applyImageShotFlow({ overwrite: false });
  imageShotReplace.onclick = () => applyImageShotFlow({ overwrite: true });
  imageAestheticApply.onclick = () => applyImageAesthetic({ overwrite: false });
  imageAestheticReplace.onclick = () => applyImageAesthetic({ overwrite: true });
  videoStyleApply.onclick = () => applyVideoStyle({ overwrite: false });
  videoStyleReplace.onclick = () => applyVideoStyle({ overwrite: true });
  performanceSelect.onchange = () => {
    state.performanceStyle = String(performanceSelect.value || "");
    refreshPerformanceInfo();
    notifyStoryboardDefaultsChanged();
  };
  characterSpeedInput.addEventListener("input", () => {
    state.characterMotionSpeed = storyboardSpeedValue(characterSpeedInput.value, 4);
    characterSpeedInput.value = String(state.characterMotionSpeed);
    refreshCharacterSpeedInfo();
  });
  characterSpeedInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  characterSpeedHint.onclick = () => {
    window.alert([
      "Character Motion Speed controls how active the subject's body movement should be.",
      "",
      "0: subject stays still or holds a pose.",
      "1-3: subtle motion like gestures, turns, swaying, reaching, or small steps.",
      "4-6: active performance like walking, dancing, interacting with objects, or using the set.",
      "7-8: energetic action like running, hard dancing, climbing, struggling, spinning, or crossing the space.",
      "9-10: fast action movement like sprinting, explosive dance, chase beats, rapid direction changes, or intense physical performance.",
    ].join("\n"));
  };
  facialSelect.onchange = () => {
    state.facialPerformance = String(facialSelect.value || "");
    refreshFacialInfo();
    notifyStoryboardDefaultsChanged();
  };
  facialCustomInput.oninput = () => {
    state.facialPerformanceCustom = String(facialCustomInput.value || "");
    refreshFacialInfo();
  };
  facialCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  performanceApply.onclick = () => applyPerformanceStyle({ overwrite: false });
  performanceReplace.onclick = () => applyPerformanceStyle({ overwrite: true });
  facialApply.onclick = () => applyFacialPerformance({ overwrite: false });
  facialReplace.onclick = () => applyFacialPerformance({ overwrite: true });
  add.onclick = () => {
    const next = normalizeScene({ scene_number: state.scenes.length + 1, label: `Scene ${state.scenes.length + 1}` }, state.scenes.length);
    state.scenes.push(next);
    openSceneEditor(next);
    renderTable();
  };
  gptButton.onclick = copyStoryboardForGpt;
  importImagePromptsButton.onclick = openImportImagePromptsFromGptModal;
  gptStoryButton.onclick = copyStoryLayerForGpt;
  importStoryJsonButton.onclick = openImportStoryJsonModal;
  gemmaAllButton.onclick = startAllPromptsWithGemma;
  clearPromptsButton.onclick = clearAllStoryboardPrompts;
  clearStoryBeatsButton.onclick = clearAllStoryboardStoryBeats;
  storyLayerEnabledInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  shortFilmPlanningModeSelect.addEventListener("change", () => {
    state.shortFilmPlanningMode = normalizeStoryboardShortFilmPlanningMode(shortFilmPlanningModeSelect.value);
    shortFilmPlanningModeSelect.value = state.shortFilmPlanningMode;
    refreshSetupPanelSummaries();
    notifyStoryboardDefaultsChanged();
    renderTable();
  });
  imageWorldStyleSelect.addEventListener("change", () => {
    refreshImageWorldStyleInfo();
    syncStoryLayerFromInputs({ notify: true });
  });
  imageCustomStyleInput.addEventListener("input", () => {
    refreshImageWorldStyleInfo();
    syncStoryLayerFromInputs();
  });
  imageCustomStyleInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  lyricStoryStrengthInput.addEventListener("input", () => {
    syncLyricStoryStrengthLabel();
    syncStoryLayerFromInputs();
  });
  lyricStoryStrengthInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  lyricStoryStrengthHintButton.onclick = () => {
    window.alert([
      `Lyric Story Strength controls how literally ${promptRunnerName()} should follow the lyrics when creating the story arc, story brief, scene beats, and prompt context.`,
      "",
      "0: do not use lyrics as story source.",
      "1-3: use lyrics as mood and emotional timing only.",
      "4-6: balance lyrics with the story arc, subjects, and locations.",
      "7-8: lyrics strongly shape the scene story; include recognizable lyric anchors when possible.",
      "9-10: use lyrics as literally as possible; non-instrumental scenes should include a concrete object, action, emotion, or situation from the exact lyric line whenever possible.",
    ].join("\n"));
  };
  overallStoryIdeaInput.addEventListener("input", syncStoryLayerFromInputs);
  overallStoryIdeaInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  userStoryArcInput.addEventListener("input", syncStoryLayerFromInputs);
  userStoryArcInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  songStoryBriefInput.addEventListener("input", syncStoryLayerFromInputs);
  songStoryBriefInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  createStoryArcButton.onclick = createStoryArcWithGemma;
  createStoryBriefButton.onclick = createStoryBriefWithGemma;
  createMissingBeatsButton.onclick = () => createAllSceneBeatsWithGemma({ overwrite: false });
  replaceBeatsButton.onclick = () => createAllSceneBeatsWithGemma({ overwrite: true });
  detectSectionsButton.onclick = detectLyricSections;
  openMiniMaxScriptMapperButton.onclick = openMiniMaxScriptMapper;
  planDialogueScenesButton.onclick = planFilmDialogueScenesWithLlm;
  applyDialoguePlanButton.onclick = applyFilmDialoguePlanToVideoBuilder;
  keepGemmaLoadedInput.onchange = () => {
    state.gemmaSettings = {
      ...(state.gemmaSettings || {}),
      keep_loaded_for_storyboard_all: Boolean(keepGemmaLoadedInput.checked),
    };
  };
  save.onclick = saveStoryboard;
  exportPrompts.onclick = exportPromptFiles;
  close.onclick = () => backdrop.remove();
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) backdrop.remove();
  });
  refreshCameraFlowInfo();
  refreshImageShotInfo();
  refreshImageAestheticInfo();
  refreshVideoStyleInfo();
  refreshTemporalEffectInfo();
  refreshFxInfo();
  refreshImageWorldStyleInfo();
  refreshConsistencyInfo();
  refreshCameraSpeedInfo();
  refreshCutFrequencyInfo();
  refreshPerformanceInfo();
  refreshCharacterSpeedInfo();
  refreshFacialInfo();
  setMode(state.mode || "storyboard_prompts");
  loadExisting();
}

window.VRGDGStoryboardBuilder = window.VRGDGStoryboardBuilder || {};
window.VRGDGStoryboardBuilder.open = openStoryboardBuilder;

function ensureButton(node) {
  const buttonName = "Open Storyboard Builder";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => {
    const projectWidget = (node.widgets || []).find((item) => item.name === "project_folder");
    openStoryboardBuilder({ projectFolder: projectWidget?.value || "" });
  });
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
}

app.registerExtension({
  name: "vrgdg.StoryboardBuilderUI",
  loadedGraphNode(node) {
    if ((node?.comfyClass || node?.type) === NODE_NAME) ensureButton(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
  },
});
